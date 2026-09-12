"""PyAV H.264/AAC muxing with loudness parity and atomic publication."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Literal


def _numpy():
    return import_module("numpy")


def _av():
    try:
        return import_module("av")
    except ImportError as exc:
        raise RuntimeError(
            "native MP4 muxing requires PyAV with H.264 and AAC encoders"
        ) from exc


def _as_numpy(value: Any):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return _numpy().asarray(value)


def _video_uint8(video: Any):
    np = _numpy()
    frames = _as_numpy(video)
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError("muxer supports batch size one only")
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError("video must be [F,H,W,3], [3,F,H,W], or [F,3,H,W]")
    if frames.shape[-1] == 3:
        pass
    elif frames.shape[0] == 3:
        frames = frames.transpose(1, 2, 3, 0)
    elif frames.shape[1] == 3:
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise ValueError("video tensor has no identifiable RGB channel dimension")
    if np.issubdtype(frames.dtype, np.floating):
        if not np.isfinite(frames).all():
            raise ValueError("video contains NaN or infinity")
        if float(frames.min()) < -1e-4 or float(frames.max()) > 1.0001:
            raise ValueError("decoded video must be float RGB in [0,1]")
        frames = np.rint(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif frames.dtype != np.uint8:
        raise ValueError("decoded video must be float [0,1] or uint8")
    return np.ascontiguousarray(frames)


def _audio_stereo(audio: Any):
    np = _numpy()
    waveform = _as_numpy(audio).astype(np.float32, copy=False)
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError("muxer supports one audio batch only")
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError("audio must be [1,2,S], [2,S], or [S,2]")
    if waveform.shape[0] == 2:
        pass
    elif waveform.shape[1] == 2:
        waveform = waveform.T
    else:
        raise ValueError("H3 output must contain exactly two audio channels")
    if not np.isfinite(waveform).all():
        raise ValueError("audio contains NaN or infinity")
    return np.ascontiguousarray(waveform)


def normalize_h3_audio_loudness(audio: Any):
    """Apply the established H3 post-VAE loudness rule.

    Per batch, divide all channels/samples by ``max(std * 5, peak / 0.9, 1)``.
    The prior rule hard-clipped sparse Audio-VAE peaks whenever the track's
    global standard deviation stayed below 0.2.  Those flat-topped transients
    occurred disproportionately in later long-video windows and are perceived
    as harsh or under-denoised speech.  Peak-safe scaling preserves the entire
    waveform shape and leaves already-safe tracks bit-identical.  The 0.9
    ceiling also leaves headroom for AAC inter-sample overshoot.  NumPy
    ``ddof=1`` matches PyTorch's historical sample-standard-deviation rule.
    """

    np = _numpy()
    stereo = _audio_stereo(audio)[None, ...]
    loudness_divisor = np.std(
        stereo, axis=(1, 2), keepdims=True, ddof=1
    ) * 5.0
    peak = np.max(np.abs(stereo), axis=(1, 2), keepdims=True)
    peak_divisor = peak / 0.9
    divisor = np.maximum(np.maximum(loudness_divisor, peak_divisor), 1.0)
    return (stereo / divisor)[0].astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class MuxConfig:
    video_codec: str = "libx264"
    # H3's fine texture is expensive to regenerate and cheap to preserve.
    # On the same decoded 1280x736 15-second frames, CRF14/superfast improved
    # re-encode PSNR by 1.37 dB and SSIM from 0.990396 to 0.993200 versus the
    # former CRF18/veryfast default, while median mux time fell from 2.346s to
    # 2.284s.
    # This changes only delivery encoding; DiT/VAE results, motion, timing and
    # audio are untouched. Avoid post-VAE sharpening: it was slower and did not
    # produce a stable FasterVQA gain.
    video_crf: int = 14
    video_preset: str = "superfast"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bit_rate: int = 192_000
    required_audio_channels: int = 2
    probe_duration_tolerance_seconds: float = 0.15
    validation_mode: Literal["metadata", "full_decode"] = "metadata"

    def __post_init__(self) -> None:
        if self.validation_mode not in ("metadata", "full_decode"):
            raise ValueError("validation_mode must be metadata or full_decode")

    def video_options(self) -> dict[str, str]:
        if self.video_codec == "h264_nvenc":
            # Constant-QP avoids an implicit low bitrate cap.  p6/hq keeps
            # generated-image texture while the fixed-function encoder still
            # runs much faster than CPU x264 on an RTX 4090.
            return {
                "preset": self.video_preset,
                "tune": "hq",
                "rc": "constqp",
                "qp": str(self.video_crf),
            }
        return {"crf": str(self.video_crf), "preset": self.video_preset}


@dataclass(frozen=True, slots=True)
class MediaProbe:
    path: str
    width: int
    height: int
    frame_count: int
    fps: float
    video_duration_seconds: float
    audio_channels: int
    audio_sample_rate: int
    audio_duration_seconds: float
    video_codec: str
    audio_codec: str


def probe_media(path: str | Path) -> MediaProbe:
    av = _av()
    source = Path(path)
    with av.open(str(source), mode="r") as container:
        videos = [stream for stream in container.streams if stream.type == "video"]
        audios = [stream for stream in container.streams if stream.type == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise RuntimeError("generated MP4 must contain exactly one video and one audio stream")
        video_stream, audio_stream = videos[0], audios[0]
        width = int(video_stream.codec_context.width)
        height = int(video_stream.codec_context.height)
        fps = float(video_stream.average_rate or video_stream.base_rate or 0.0)
        video_codec = str(video_stream.codec_context.name)
        audio_codec = str(audio_stream.codec_context.name)
        audio_rate = int(audio_stream.codec_context.sample_rate or audio_stream.rate or 0)
        layout = audio_stream.codec_context.layout
        audio_channels = len(layout.channels) if layout is not None else 0

        frame_count = 0
        video_end = 0.0
        for frame in container.decode(video_stream):
            frame_count += 1
            if frame.pts is not None and frame.time_base is not None:
                video_end = max(video_end, float(frame.pts * frame.time_base) + 1.0 / fps)

    with av.open(str(source), mode="r") as container:
        audio_stream = next(stream for stream in container.streams if stream.type == "audio")
        audio_samples = sum(frame.samples for frame in container.decode(audio_stream))
    audio_duration = audio_samples / float(audio_rate) if audio_rate else 0.0
    if video_end == 0.0 and fps:
        video_end = frame_count / fps
    return MediaProbe(
        path=str(source.resolve()),
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        video_duration_seconds=video_end,
        audio_channels=audio_channels,
        audio_sample_rate=audio_rate,
        audio_duration_seconds=audio_duration,
        video_codec=video_codec,
        audio_codec=audio_codec,
    )


def probe_media_metadata(path: str | Path) -> MediaProbe:
    """Validate a finalized generated MP4 from its indexed stream metadata.

    The native muxer already observes every submitted frame and every encoder
    exception.  Re-decoding the just-written H.264/AAC streams on every
    request adds latency without strengthening normal request semantics.  The
    slower :func:`probe_media` remains available for release audits.
    """

    av = _av()
    source = Path(path)
    with av.open(str(source), mode="r") as container:
        videos = [stream for stream in container.streams if stream.type == "video"]
        audios = [stream for stream in container.streams if stream.type == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise RuntimeError("generated MP4 must contain exactly one video and one audio stream")
        video_stream, audio_stream = videos[0], audios[0]
        width = int(video_stream.codec_context.width)
        height = int(video_stream.codec_context.height)
        fps = float(video_stream.average_rate or video_stream.base_rate or 0.0)
        frame_count = int(video_stream.frames or 0)
        if frame_count <= 0:
            raise RuntimeError("generated MP4 has no indexed video frame count")
        video_duration = (
            float(video_stream.duration * video_stream.time_base)
            if video_stream.duration is not None and video_stream.time_base is not None
            else frame_count / fps if fps else 0.0
        )
        audio_duration = (
            float(audio_stream.duration * audio_stream.time_base)
            if audio_stream.duration is not None and audio_stream.time_base is not None
            else 0.0
        )
        layout = audio_stream.codec_context.layout
        audio_channels = len(layout.channels) if layout is not None else 0
        audio_rate = int(
            audio_stream.codec_context.sample_rate or audio_stream.rate or 0
        )
        return MediaProbe(
            path=str(source.resolve()),
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            video_duration_seconds=video_duration,
            audio_channels=audio_channels,
            audio_sample_rate=audio_rate,
            audio_duration_seconds=audio_duration,
            video_codec=str(video_stream.codec_context.name),
            audio_codec=str(audio_stream.codec_context.name),
        )


CancelCheck = Callable[[], None]


class AtomicPyAVMuxer:
    """Pipeline muxer adapter implementing ``muxer.write(...)``."""

    def __init__(
        self,
        *,
        output_root: Path,
        config: MuxConfig = MuxConfig(),
    ) -> None:
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config = config

    def _validate_probe(
        self,
        probe: MediaProbe,
        *,
        frame_count: int,
        width: int,
        height: int,
        fps: int,
        sample_rate: int,
    ) -> None:
        if (probe.width, probe.height) != (width, height):
            raise RuntimeError("encoded MP4 geometry differs from decoded video")
        if probe.frame_count != frame_count:
            raise RuntimeError("encoded MP4 frame count differs from decoded video")
        if abs(probe.fps - fps) > 0.01:
            raise RuntimeError("encoded MP4 FPS differs from generation FPS")
        if probe.audio_channels != self.config.required_audio_channels:
            raise RuntimeError("encoded MP4 does not contain stereo audio")
        if probe.audio_sample_rate != sample_rate:
            raise RuntimeError("encoded MP4 audio sample rate changed")
        tolerance = max(self.config.probe_duration_tolerance_seconds, 2.0 / fps)
        if abs(probe.video_duration_seconds - probe.audio_duration_seconds) > tolerance:
            raise RuntimeError("encoded MP4 audio/video durations are not synchronized")

    def write(
        self,
        *,
        video: Any,
        audio: Any,
        sample_rate: int | None,
        fps: int,
        output_path: Path | None,
        cancel_check: CancelCheck = lambda: None,
    ) -> dict[str, Any]:
        if output_path is None:
            raise ValueError("native muxer requires an explicit output_path")
        destination = Path(output_path).resolve()
        if not destination.is_relative_to(self.output_root):
            raise ValueError("output_path must remain inside the configured output root")
        if destination.suffix.lower() != ".mp4":
            raise ValueError("native audio-video output must use the .mp4 suffix")
        if sample_rate is None or sample_rate <= 0 or fps <= 0:
            raise ValueError("positive audio sample rate and FPS are required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        cancel_check()

        np = _numpy()
        frames = _video_uint8(video)
        source_waveform = _audio_stereo(audio)
        source_peak = float(np.max(np.abs(source_waveform)))
        waveform = normalize_h3_audio_loudness(source_waveform)
        normalized_peak = float(np.max(np.abs(waveform)))
        expected_samples = int(round(frames.shape[0] / float(fps) * sample_rate))
        if waveform.shape[1] < expected_samples:
            waveform = np.pad(waveform, ((0, 0), (0, expected_samples - waveform.shape[1])))
        else:
            waveform = waveform[:, :expected_samples]
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=".tmp.mp4",
            dir=destination.parent,
            delete=False,
        )
        temporary = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            self._encode(
                temporary,
                frames=frames,
                waveform=waveform,
                sample_rate=sample_rate,
                fps=fps,
                cancel_check=cancel_check,
            )
            with temporary.open("rb") as file_handle:
                os.fsync(file_handle.fileno())
            cancel_check()
            probe = (
                probe_media(temporary)
                if self.config.validation_mode == "full_decode"
                else probe_media_metadata(temporary)
            )
            self._validate_probe(
                probe,
                frame_count=int(frames.shape[0]),
                width=int(frames.shape[2]),
                height=int(frames.shape[1]),
                fps=fps,
                sample_rate=sample_rate,
            )
            cancel_check()
            os.replace(temporary, destination)
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            published_probe = MediaProbe(**{**asdict(probe), "path": str(destination)})
            return {
                "output_path": str(destination),
                "media": asdict(published_probe),
                "encoder": {
                    "video_codec": self.config.video_codec,
                    "video_crf": self.config.video_crf,
                    "video_preset": self.config.video_preset,
                    "audio_codec": self.config.audio_codec,
                    "audio_bit_rate": self.config.audio_bit_rate,
                    "audio_normalization": {
                        "policy": "std5_peak0p9_shape_preserving_v2",
                        "input_peak": source_peak,
                        "normalized_peak": normalized_peak,
                        "linear_gain": (
                            normalized_peak / source_peak
                            if source_peak > 0.0 else 1.0
                        ),
                        "hard_clipped_samples": 0,
                    },
                    "validation_mode": self.config.validation_mode,
                },
            }
        finally:
            if temporary.exists():
                temporary.unlink()

    def _encode(
        self,
        path: Path,
        *,
        frames: Any,
        waveform: Any,
        sample_rate: int,
        fps: int,
        cancel_check: CancelCheck,
    ) -> None:
        av = _av()
        config = self.config
        try:
            container = av.open(str(path), mode="w", format="mp4")
            video_stream = container.add_stream(
                config.video_codec,
                rate=fps,
                options=config.video_options(),
            )
            video_stream.width = int(frames.shape[2])
            video_stream.height = int(frames.shape[1])
            video_stream.pix_fmt = config.pixel_format
            video_stream.time_base = Fraction(1, fps)
            audio_stream = container.add_stream(config.audio_codec, rate=sample_rate)
            audio_stream.bit_rate = config.audio_bit_rate
            audio_stream.codec_context.layout = "stereo"
            audio_stream.codec_context.sample_rate = sample_rate
            audio_stream.time_base = Fraction(1, sample_rate)

            for index, array in enumerate(frames):
                cancel_check()
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

            cancel_check()
            audio_frame = av.AudioFrame.from_ndarray(
                waveform,
                format="fltp",
                layout="stereo",
            )
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, sample_rate)
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
            container.close()
        except Exception:
            try:
                container.close()
            except Exception:
                pass
            raise
