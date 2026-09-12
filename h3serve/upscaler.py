from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .backend import JobCancelled
from .config import ServicePaths


PROGRESS_PREFIX = "H3_UPSCALE_PROGRESS "
READY_PREFIX = "H3_UPSCALE_READY "
RESPONSE_PREFIX = "H3_UPSCALE_RESPONSE "
REQUIRED_MODEL_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
    "posi_prompt.pth",
)


class UpscaleError(RuntimeError):
    pass


class RetiredFlashVSRUpscaler:
    """Unavailable-method surface when the optional temporal runtime is absent.

    The historical class name is retained for import compatibility. New
    installations normally construct :class:`FlashVSRUpscaler`; this object is
    used only when its interpreter, source, or weights have not been installed.
    """

    def status(self) -> dict[str, Any]:
        return {
            "ready": False,
            "implementation": "temporal_second_sampling_unavailable",
            "resident_state": "unavailable",
            "missing": [],
            "remediation": "run scripts/install.sh and scripts/download_models.py",
            "last_error": None,
        }

    def configure_data_dir(self, _data_dir: Path) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def upscale(self, *_args: Any, **_kwargs: Any) -> UpscaleResult:
        raise UpscaleError(
            "temporal second sampling is unavailable; install the isolated "
            "FlashVSR runtime and model weights"
        )


@dataclass(frozen=True)
class UpscaleResult:
    output_path: Path
    elapsed_seconds: float
    width: int
    height: int
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    timings: dict[str, float] | None = None


class FlashVSRUpscaler:
    """Persistent, process-isolated FlashVSR runtime for a single GPU queue.

    FlashVSR keeps its immutable weights in CPU RAM between jobs. A request
    moves only the active modules to CUDA; completion offloads those modules
    and clears task-local caches. Process isolation keeps FlashVSR's Torch 2.6
    ABI separate from the H3 service's Torch 2.13 runtime.
    """

    def __init__(self, paths: ServicePaths) -> None:
        self.paths = paths
        self.worker = paths.release_root / "scripts/flashvsr_worker.py"
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._ready: asyncio.Future | None = None
        self._response: asyncio.Future | None = None
        self._active_request_id: str | None = None
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None
        self._model_load_seconds: float | None = None
        self._preload_wall_seconds: float | None = None
        self._requests_completed = 0
        self._last_error: str | None = None
        self._closing = False
        self._daemon_log = paths.data_dir / "logs/flashvsr-daemon.log"

    def status(self) -> dict[str, Any]:
        missing = []
        if not self.paths.flashvsr_python_executable.is_file():
            missing.append(str(self.paths.flashvsr_python_executable))
        if not self.worker.is_file():
            missing.append(str(self.worker))
        if not (self.paths.flashvsr_source_dir / "diffsynth/__init__.py").is_file():
            missing.append(str(self.paths.flashvsr_source_dir / "diffsynth"))
        for filename in REQUIRED_MODEL_FILES:
            path = self.paths.flashvsr_model_dir / filename
            if not path.is_file():
                missing.append(str(path))
        process_alive = self._process is not None and self._process.returncode is None
        ready = bool(
            process_alive and self._ready is not None and self._ready.done()
            and not self._ready.cancelled() and self._ready.exception() is None
        )
        return {
            "ready": not missing,
            "implementation": "vendored_flashvsr_v1.1_persistent_cpu_hot",
            "missing": missing,
            "resident_state": "ready" if ready else (
                "loading" if process_alive else "stopped"
            ),
            "model_load_seconds": self._model_load_seconds,
            "preload_wall_seconds": self._preload_wall_seconds,
            "requests_completed": self._requests_completed,
            "gpu_policy": "on_demand_release_after_each_task",
            "last_error": self._last_error,
        }

    def configure_data_dir(self, data_dir: Path) -> None:
        if self._process is not None and self._process.returncode is None:
            raise UpscaleError("cannot switch workspace while FlashVSR is running")
        self.paths = replace(self.paths, data_dir=data_dir.resolve())
        self._daemon_log = self.paths.data_dir / "logs/flashvsr-daemon.log"

    def _command(self) -> list[str]:
        return [
            str(self.paths.flashvsr_python_executable),
            str(self.worker),
            "--serve",
            "--source-root", str(self.paths.flashvsr_source_dir),
            "--model-root", str(self.paths.flashvsr_model_dir),
        ]

    async def start(self) -> None:
        """Preload model weights in RAM without claiming persistent VRAM."""
        state = self.status()
        if state["missing"]:
            raise UpscaleError("FlashVSR runtime is incomplete; run scripts/install.sh")
        async with self._start_lock:
            if self._process is not None and self._process.returncode is None:
                assert self._ready is not None
                await asyncio.shield(self._ready)
                return
            self._closing = False
            self._last_error = None
            loop = asyncio.get_running_loop()
            self._ready = loop.create_future()
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            self._daemon_log.parent.mkdir(parents=True, exist_ok=True)
            self._process = await asyncio.create_subprocess_exec(
                *self._command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
            )
            self._reader_task = asyncio.create_task(
                self._read_output(self._process), name="flashvsr-daemon-reader"
            )
            started = time.perf_counter()
            try:
                await asyncio.wait_for(asyncio.shield(self._ready), timeout=180.0)
                self._preload_wall_seconds = time.perf_counter() - started
            except BaseException:
                await self._terminate()
                raise

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        try:
            with self._daemon_log.open("a", encoding="utf-8") as log:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace")
                    log.write(text)
                    log.flush()
                    if text.startswith(READY_PREFIX):
                        event = json.loads(text[len(READY_PREFIX):])
                        self._model_load_seconds = float(event["model_load_seconds"])
                        if self._ready is not None and not self._ready.done():
                            self._ready.set_result(event)
                    elif text.startswith(PROGRESS_PREFIX):
                        try:
                            event = json.loads(text[len(PROGRESS_PREFIX):])
                        except json.JSONDecodeError:
                            continue
                        if event.get("request_id") != self._active_request_id:
                            continue
                        callback = self._progress_callback
                        if callback is not None:
                            callback(event)
                    elif text.startswith(RESPONSE_PREFIX):
                        try:
                            event = json.loads(text[len(RESPONSE_PREFIX):])
                        except json.JSONDecodeError:
                            continue
                        if event.get("request_id") != self._active_request_id:
                            continue
                        if self._response is not None and not self._response.done():
                            self._response.set_result(event)
            code = await process.wait()
            if not self._closing:
                raise UpscaleError(f"FlashVSR daemon exited with status {code}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._last_error = str(error)
            for future in (self._ready, self._response):
                if future is not None and not future.done():
                    future.set_exception(error)

    async def _terminate(self) -> None:
        process = self._process
        self._closing = True
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        current = asyncio.current_task()
        if self._reader_task is not None and self._reader_task is not current:
            if not self._reader_task.done():
                self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self._process = None
        self._reader_task = None
        self._ready = None
        self._response = None
        self._active_request_id = None
        self._progress_callback = None

    async def upscale(
        self,
        source: Path,
        *,
        target_width: int,
        target_height: int,
        cancel_event: asyncio.Event,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        output_path: Path | None = None,
    ) -> UpscaleResult:
        source = source.resolve()
        final = (
            output_path.resolve()
            if output_path is not None
            else source.with_name(
                f"{source.stem}_flashvsr_{target_width}x{target_height}.mp4"
            )
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".{final.stem}.upscaling.mp4")
        temporary.unlink(missing_ok=True)
        async with self._request_lock:
            started = time.perf_counter()
            request_id = uuid.uuid4().hex
            try:
                if cancel_event.is_set():
                    raise JobCancelled("cancelled before FlashVSR upscaling")
                await self.start()
                process = self._process
                if process is None or process.stdin is None:
                    raise UpscaleError("FlashVSR daemon did not expose stdin")
                self._active_request_id = request_id
                self._progress_callback = progress_callback
                self._response = asyncio.get_running_loop().create_future()
                request = {
                    "request_id": request_id,
                    "command": "upscale",
                    "input": str(source),
                    "output": str(temporary),
                    "target_width": target_width,
                    "target_height": target_height,
                    "gpu_input_limit_mib": int(
                        os.environ.get("H3_FLASHVSR_GPU_INPUT_LIMIT_MIB", "1536")
                    ),
                }
                process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                await process.stdin.drain()
                while True:
                    if cancel_event.is_set():
                        # The vendor pipeline is not cooperatively cancellable;
                        # terminating the isolated daemon reliably frees CUDA.
                        await self._terminate()
                        raise JobCancelled("cancelled during FlashVSR upscaling")
                    try:
                        response = await asyncio.wait_for(
                            asyncio.shield(self._response), timeout=0.25
                        )
                        break
                    except asyncio.TimeoutError:
                        continue
                if not response.get("ok") or not temporary.is_file():
                    raise UpscaleError(
                        str(response.get("error", "FlashVSR produced no output"))
                    )
                final.unlink(missing_ok=True)
                temporary.replace(final)
                self._requests_completed += 1
                return UpscaleResult(
                    output_path=final,
                    elapsed_seconds=time.perf_counter() - started,
                    width=target_width,
                    height=target_height,
                    peak_allocated_mib=float(response["peak_allocated_mib"]),
                    peak_reserved_mib=float(response["peak_reserved_mib"]),
                    timings={
                        str(name): float(seconds)
                        for name, seconds in response.get("timings", {}).items()
                    },
                )
            except (BrokenPipeError, ConnectionError) as error:
                await self._terminate()
                raise UpscaleError("FlashVSR daemon connection was lost") from error
            finally:
                self._active_request_id = None
                self._progress_callback = None
                self._response = None
                temporary.unlink(missing_ok=True)

    async def stop(self) -> None:
        async with self._request_lock:
            await self._terminate()
