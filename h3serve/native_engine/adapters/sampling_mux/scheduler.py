"""H3 simple sigma schedule and deterministic dual-modality noise."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


def _torch():
    return import_module("torch")


def simple_sigma_schedule(num_steps: int, shift: float) -> tuple[float, ...]:
    """Return Comfy's exact discrete ``simple`` rectified-flow schedule.

    ``num_steps`` consistently means model evaluations in the native service.
    This matches the product's explicit actual-step indices, which range from
    zero through ``num_steps - 1``.  H3's model sampler owns a 1000-entry
    shifted table.  Comfy's ``simple`` scheduler samples that table with
    ``int(step * 1000 / num_steps)``; directly shifting a ``linspace`` with
    ``num_steps + 1`` entries is close but not equivalent and changes the
    same-seed trajectory.
    """

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if shift <= 0.0:
        raise ValueError("sigma shift must be positive")
    torch = _torch()
    base = torch.arange(1, 1001, dtype=torch.float32, device="cpu") / 1000.0
    table = shift * base / (1.0 + (shift - 1.0) * base)
    stride = len(table) / num_steps
    values = tuple(
        float(table[-(1 + int(index * stride))])
        for index in range(num_steps)
    ) + (0.0,)
    if values[0] != 1.0 or values[-1] != 0.0:
        raise RuntimeError("H3 sigma schedule must span exactly [1, 0]")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise RuntimeError("H3 sigma schedule must be strictly decreasing")
    return values


def refinement_sigma_schedule(
    num_steps: int,
    denoise: float,
    shift: float,
    *,
    curve_power: float = 1.0,
) -> tuple[float, ...]:
    """Build a step-invariant low-noise Simple schedule.

    ``denoise`` owns the start point and ``num_steps`` owns only the numerical
    resolution between that point and zero.  This is the continuous form of
    ComfyUI's ``steps / denoise`` tail selection without its low-step integer
    quantisation (most visible at denoise=0.30).  ``curve_power`` leaves the
    endpoints and evaluation count unchanged; values above one move the
    intermediate evaluations toward the low-noise terminal region.
    """

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if not 0.0 < float(denoise) <= 1.0:
        raise ValueError("denoise must lie inside (0, 1]")
    if shift <= 0.0:
        raise ValueError("sigma shift must be positive")
    if not 0.5 <= float(curve_power) <= 3.0:
        raise ValueError("refinement sigma curve power must lie inside [0.5, 3]")

    def shifted(base: float) -> float:
        return shift * base / (1.0 + (shift - 1.0) * base)

    values = tuple(
        shifted(
            float(denoise)
            * (((num_steps - index) / num_steps) ** float(curve_power))
        )
        for index in range(num_steps)
    ) + (0.0,)
    if any(left <= right for left, right in zip(values, values[1:])):
        raise RuntimeError("refinement sigma schedule must be strictly decreasing")
    return values


def comfy_denoise_tail_sigma_schedule(
    num_steps: int,
    denoise: float,
    shift: float,
) -> tuple[float, ...]:
    """Return ComfyUI BasicScheduler's exact truncated ``simple`` schedule.

    BasicScheduler treats ``num_steps`` as the number of evaluations retained
    after truncation.  It first builds ``int(num_steps / denoise)`` simple
    steps, then keeps the final ``num_steps + 1`` sigma values.  FaceRefine's
    published four-step workflow relies on this discrete behaviour; the
    service's continuous refinement schedule is intentionally still available
    for ordinary H3 second sampling.
    """

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if not 0.0 < float(denoise) <= 1.0:
        raise ValueError("denoise must lie inside (0, 1]")
    total_steps = int(num_steps / float(denoise))
    if total_steps < num_steps:
        total_steps = num_steps
    return simple_sigma_schedule(total_steps, shift)[-(num_steps + 1) :]


@dataclass(frozen=True, slots=True)
class H3LatentGeometry:
    video_shape: tuple[int, int, int, int, int]
    audio_shape: tuple[int, int, int, int]
    actual_duration_seconds: float


@dataclass(frozen=True, slots=True)
class StepClock:
    index: int
    video_sigma: float
    video_sigma_next: float
    audio_sigma: float
    audio_sigma_next: float

    @property
    def video_timestep(self) -> float:
        return 1.0 - self.video_sigma

    @property
    def audio_timestep(self) -> float:
        return 1.0 - self.audio_sigma


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    sampler: str
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    actual_step_indices: tuple[int, ...]
    video_shift: float
    audio_shift: float
    step_index_offset: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.video_sigmas) != len(self.audio_sigmas):
            raise ValueError("video and audio sigma clocks must have equal length")
        if len(self.video_sigmas) < 2:
            raise ValueError("sampling plan requires at least one step")
        step_count = self.step_count
        if self.step_index_offset < 0:
            raise ValueError("sampling step offset cannot be negative")
        stop = self.step_index_offset + step_count
        if any(
            index < self.step_index_offset or index >= stop
            for index in self.actual_step_indices
        ):
            raise ValueError("actual step index falls outside the sigma schedule")

    @property
    def step_count(self) -> int:
        return len(self.video_sigmas) - 1

    def clock(self, index: int) -> StepClock:
        if index < 0 or index >= self.step_count:
            raise IndexError(index)
        return StepClock(
            index=index + self.step_index_offset,
            video_sigma=self.video_sigmas[index],
            video_sigma_next=self.video_sigmas[index + 1],
            audio_sigma=self.audio_sigmas[index],
            audio_sigma_next=self.audio_sigmas[index + 1],
        )


LayoutBuilder = Callable[[Any, H3LatentGeometry], Any]


class H3SimpleScheduler:
    """Pipeline scheduler adapter implementing ``scheduler.prepare(state)``.

    Noise is generated by one CPU generator in the historical order: complete
    video latent first, then channel-major stereo audio latent. Moving the result
    to CUDA happens only after both deterministic CPU draws are complete.
    """

    def __init__(
        self,
        *,
        execution_device: str = "cuda:0",
        video_channels: int = 24,
        audio_channels: int = 32,
        spatial_scale: int = 16,
        audio_latent_hz: float = 40.0,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        layout_builder: LayoutBuilder | None = None,
    ) -> None:
        if video_channels <= 0 or audio_channels <= 0 or spatial_scale <= 0:
            raise ValueError("latent dimensions must be positive")
        if audio_latent_hz <= 0.0:
            raise ValueError("audio_latent_hz must be positive")
        self.execution_device = execution_device
        self.video_channels = video_channels
        self.audio_channels = audio_channels
        self.spatial_scale = spatial_scale
        self.audio_latent_hz = audio_latent_hz
        self.video_shift = video_shift
        self.audio_shift = audio_shift
        self.layout_builder = layout_builder

    @staticmethod
    def video_latent_frames(num_frames: int) -> int:
        if num_frames <= 5:
            return 2
        return ((num_frames - 5) // 17) * 5 + 2

    def geometry(self, request: Any) -> H3LatentGeometry:
        if request.width % self.spatial_scale or request.height % self.spatial_scale:
            raise ValueError("canvas is not divisible by the H3 VAE spatial scale")
        duration = request.num_frames / float(request.fps)
        audio_frames = int(round(duration * self.audio_latent_hz))
        return H3LatentGeometry(
            video_shape=(
                1,
                self.video_channels,
                self.video_latent_frames(request.num_frames),
                request.height // self.spatial_scale,
                request.width // self.spatial_scale,
            ),
            audio_shape=(1, self.audio_channels, 2, audio_frames),
            actual_duration_seconds=duration,
        )

    def _plan(self, sampling: Any, *, seed: int = 0) -> SamplingPlan:
        video = simple_sigma_schedule(sampling.num_steps, self.video_shift)
        # Current H3 ModelSamplingAV carries both streams on the video sigma
        # schedule; the model boundary performs the audio-clock transport.
        audio = video
        if sampling.actual_step_indices is None:
            actual = tuple(range(sampling.num_steps))
        else:
            actual = tuple(sampling.actual_step_indices)
        return SamplingPlan(
            sampler=sampling.sampler,
            video_sigmas=video,
            audio_sigmas=audio,
            actual_step_indices=actual,
            video_shift=self.video_shift,
            audio_shift=self.audio_shift,
            seed=int(seed),
        )

    def prepare(self, state: Any) -> dict[str, Any]:
        request = state.request
        geometry = self.geometry(request)
        torch = _torch()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(request.seed) & ((1 << 64) - 1))
        video = torch.randn(
            geometry.video_shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        audio = torch.randn(
            geometry.audio_shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        if self.execution_device != "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA scheduler requested but torch CUDA is unavailable")
            video = video.pin_memory().to(self.execution_device, non_blocking=True)
            audio = audio.pin_memory().to(self.execution_device, non_blocking=True)

        plan = self._plan(request.sampling, seed=request.seed)
        state.metadata["sampling_plan"] = plan
        state.metadata["latent_geometry"] = geometry
        layout = self.layout_builder(state, geometry) if self.layout_builder else None
        return {
            "video_latents": video,
            "audio_latents": audio,
            "packed_layout": layout,
        }
