"""ComfyUI-independent H3 audio-video sampler interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from .scheduler import SamplingPlan, StepClock


@dataclass(frozen=True, slots=True)
class AVPrediction:
    """Denoised x0 predictions returned by the model/forecast controller."""

    video_denoised: Any
    audio_denoised: Any


class AVPredictor(Protocol):
    def __call__(
        self,
        video: Any,
        audio: Any,
        clock: StepClock,
        *,
        step_index: int,
        is_actual_step: bool,
    ) -> AVPrediction: ...


CancelCheck = Callable[[], None]
StepCallback = Callable[[int, StepClock, Any, Any], None]
StepTransition = Callable[
    [int, StepClock, Any, Any, Any | None, Any | None],
    tuple[Any, Any, Any | None, Any | None],
]


def _euler(sample: Any, denoised: Any, sigma: float, sigma_next: float) -> Any:
    if sigma <= 0.0:
        raise ValueError("Euler source sigma must be positive")
    derivative = (sample - denoised) / sigma
    return sample + derivative * (sigma_next - sigma)


def _phi_values(value: float) -> tuple[float, float]:
    if abs(value) < 1e-7:
        return 1.0, 0.5
    phi1 = math.expm1(value) / value
    return phi1, (phi1 - 1.0) / value


def _res_step(
    sample: Any,
    denoised: Any,
    *,
    sigma: float,
    sigma_next: float,
    previous_sigma: float | None,
    previous_denoised: Any | None,
) -> Any:
    if sigma_next == 0.0 or previous_denoised is None or previous_sigma is None:
        return _euler(sample, denoised, sigma, sigma_next)

    t = -math.log(sigma)
    t_previous = -math.log(previous_sigma)
    t_next = -math.log(sigma_next)
    # eta=0 means the prior step's sigma_down is today's source sigma.
    t_old_down = t
    h = t_next - t
    if h == 0.0:
        raise ValueError("duplicate sigma endpoints are not supported")
    c2 = (t_previous - t_old_down) / h
    phi1, phi2 = _phi_values(-h)
    if c2 == 0.0 or not math.isfinite(c2):
        b1, b2 = phi1, 0.0
    else:
        b2 = phi2 / c2
        b1 = phi1 - b2
    return math.exp(-h) * sample + h * (b1 * denoised + b2 * previous_denoised)


class ResMultistepAVSampler:
    """Second-order RES multistep sampler over synchronized H3 clocks.

    The predictor is responsible for executing either a real DiT step or a
    forecast step according to ``is_actual_step``. Both paths return the same
    x0 contract, so sampler mathematics stays independent from acceleration.
    """

    name = "res_multistep"

    def sample(
        self,
        video: Any,
        audio: Any,
        plan: SamplingPlan,
        predict: AVPredictor,
        *,
        cancel_check: CancelCheck = lambda: None,
        callback: StepCallback | None = None,
        transition: StepTransition | None = None,
        initial_previous_video: Any | None = None,
        initial_previous_audio: Any | None = None,
        initial_previous_video_sigma: float | None = None,
        initial_previous_audio_sigma: float | None = None,
    ) -> tuple[Any, Any]:
        history_values = (
            initial_previous_video,
            initial_previous_audio,
            initial_previous_video_sigma,
            initial_previous_audio_sigma,
        )
        if any(value is not None for value in history_values) and not all(
            value is not None for value in history_values
        ):
            raise ValueError("initial RES history must provide both modalities and sigmas")
        previous_video = initial_previous_video
        previous_audio = initial_previous_audio
        previous_video_sigma = initial_previous_video_sigma
        previous_audio_sigma = initial_previous_audio_sigma
        actual = frozenset(plan.actual_step_indices)

        for index in range(plan.step_count):
            cancel_check()
            clock = plan.clock(index)
            global_index = clock.index
            prediction = predict(
                video,
                audio,
                clock,
                step_index=global_index,
                is_actual_step=global_index in actual,
            )
            next_video = _res_step(
                video,
                prediction.video_denoised,
                sigma=clock.video_sigma,
                sigma_next=clock.video_sigma_next,
                previous_sigma=previous_video_sigma,
                previous_denoised=previous_video,
            )
            next_audio = _res_step(
                audio,
                prediction.audio_denoised,
                sigma=clock.audio_sigma,
                sigma_next=clock.audio_sigma_next,
                previous_sigma=previous_audio_sigma,
                previous_denoised=previous_audio,
            )
            previous_video, previous_audio = (
                prediction.video_denoised,
                prediction.audio_denoised,
            )
            previous_video_sigma = clock.video_sigma
            previous_audio_sigma = clock.audio_sigma
            video, audio = next_video, next_audio
            if transition is not None:
                video, audio, previous_video, previous_audio = transition(
                    global_index,
                    clock,
                    video,
                    audio,
                    previous_video,
                    previous_audio,
                )
            if callback is not None:
                callback(global_index, clock, video, audio)
        return video, audio


def _flow_half_log_snr(sigma: float) -> float:
    """Return ComfyUI's half-logSNR for ``ModelSamplingDiscreteFlow``.

    Unlike the ``CONST`` flow parameterization, H3's discrete-flow sampler
    treats the solver state as unit-alpha data prediction.  SA-Solver therefore
    uses ``-log(sigma)`` here; using ``log((1-sigma)/sigma)`` silently changes
    every Adams coefficient.
    """

    if sigma <= 0.0:
        return math.inf
    return -math.log(sigma)


def _flow_percent_to_sigma(percent: float, shift: float) -> float:
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    timestep = 1.0 - percent
    return shift * timestep / (1.0 + (shift - 1.0) * timestep)


def _sa_coefficients(
    *,
    sigma_next: float,
    lambdas: tuple[float, ...],
    lambda_start: float,
    lambda_stop: float,
    tau: float,
) -> tuple[float, ...]:
    """Compute SA-Solver's exponential Adams coefficients.

    This is an independent AV adaptation of the Stochastic Adams equations
    used by ComfyUI's standard ``sa_solver``.  The coefficient solve is tiny
    (at most 4x4) and remains on CPU; only the resulting scalars touch the H3
    video/audio tensors.
    """

    import torch

    order = len(lambdas)
    if order <= 0:
        raise ValueError("SA-Solver requires at least one prediction")
    dtype = torch.float64
    device = "cpu"
    start = torch.tensor(lambda_start, dtype=dtype, device=device)
    stop = torch.tensor(lambda_stop, dtype=dtype, device=device)
    tau_multiplier = 1.0 + tau * tau
    depth = torch.arange(order, dtype=dtype, device=device)
    product = stop.pow(depth) - start.pow(depth) * torch.exp(
        -tau_multiplier * (stop - start)
    )
    recursive_depth = depth.unsqueeze(1) - depth.unsqueeze(0)
    log_factorial = torch.lgamma(depth + 1.0)
    recursive = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau > 0.0:
        recursive = recursive - recursive_depth * math.log(tau_multiplier)
    signs = torch.where(
        torch.remainder(recursive_depth, 2.0) == 0.0,
        torch.ones_like(recursive_depth),
        -torch.ones_like(recursive_depth),
    )
    exponential_integrals = (recursive.exp() * signs).tril() @ product
    lambda_tensor = torch.tensor(lambdas, dtype=dtype, device=device)
    vandermonde_t = torch.vander(
        lambda_tensor, N=order, increasing=True
    ).transpose(0, 1)
    lagrange_integrals = torch.linalg.solve(
        vandermonde_t, exponential_integrals
    )
    alpha_next = sigma_next * math.exp(lambda_stop)
    return tuple(float(value) for value in alpha_next * lagrange_integrals)


def _linear_combination(values: list[Any], coefficients: tuple[float, ...]) -> Any:
    if not values or len(values) != len(coefficients):
        raise ValueError("SA-Solver history and coefficients disagree")
    result = values[0] * coefficients[0]
    for value, coefficient in zip(values[1:], coefficients[1:]):
        result = result + value * coefficient
    return result


class SASolverAVSampler:
    """Stochastic Adams predictor-corrector for synchronized H3 AV latents.

    The implementation follows the standard ``sa_solver`` operating point:
    predictor order 3, corrector order 4, PECE disabled, and stochasticity in
    the 20%-80% flow-time interval. H3 second sampling normally starts near
    the low-noise boundary, so the common denoise=0.20 path is deterministic.
    As with RES, the predictor may supply either an exact DiT x0 or a forecast
    x0; both follow one solver contract and both remain in the Adams history.
    """

    name = "sa_solver"

    def __init__(self, *, predictor_order: int = 3, corrector_order: int = 4) -> None:
        if predictor_order <= 0 or corrector_order <= 0:
            raise ValueError("SA-Solver orders must be positive")
        self.predictor_order = int(predictor_order)
        self.corrector_order = int(corrector_order)

    def sample(
        self,
        video: Any,
        audio: Any,
        plan: SamplingPlan,
        predict: AVPredictor,
        *,
        cancel_check: CancelCheck = lambda: None,
        callback: StepCallback | None = None,
        transition: StepTransition | None = None,
        initial_previous_video: Any | None = None,
        initial_previous_audio: Any | None = None,
        initial_previous_video_sigma: float | None = None,
        initial_previous_audio_sigma: float | None = None,
    ) -> tuple[Any, Any]:
        if transition is not None or any(
            value is not None
            for value in (
                initial_previous_video,
                initial_previous_audio,
                initial_previous_video_sigma,
                initial_previous_audio_sigma,
            )
        ):
            raise ValueError(
                "SA-Solver second sampling does not support RES history or transitions"
            )
        import torch

        sigmas = tuple(float(value) for value in plan.video_sigmas)
        if plan.audio_sigmas != plan.video_sigmas:
            raise ValueError("SA-Solver requires the synchronized H3 AV clock")
        lambdas = tuple(_flow_half_log_snr(value) for value in sigmas)
        maximum_order = max(self.predictor_order, self.corrector_order)
        predicted_video = video
        predicted_audio = audio
        video_predictions: list[Any] = []
        audio_predictions: list[Any] = []
        previous_h = 0.0
        previous_tau = 0.0
        previous_video_noise: Any | None = None
        previous_audio_noise: Any | None = None
        generator = torch.Generator(device=video.device)
        # Match ComfyUI's default_noise_sampler exactly.  It offsets CPU seeds
        # by one while CUDA generators use the request seed directly.
        generator_seed = int(plan.seed)
        if video.device.type == "cpu":
            generator_seed += 1
        generator.manual_seed(generator_seed & ((1 << 63) - 1))
        stochastic_start = _flow_percent_to_sigma(0.20, plan.video_shift)
        stochastic_stop = _flow_percent_to_sigma(0.80, plan.video_shift)
        actual = frozenset(plan.actual_step_indices)

        for index in range(plan.step_count):
            cancel_check()
            clock = plan.clock(index)
            prediction = predict(
                predicted_video,
                predicted_audio,
                clock,
                step_index=clock.index,
                is_actual_step=clock.index in actual,
            )
            video_predictions.append(prediction.video_denoised)
            audio_predictions.append(prediction.audio_denoised)
            video_predictions = video_predictions[-maximum_order:]
            audio_predictions = audio_predictions[-maximum_order:]

            next_sigma = sigmas[index + 1]
            predictor_order = min(self.predictor_order, len(video_predictions))
            corrector_order = (
                0
                if index == 0 or next_sigma == 0.0
                else min(self.corrector_order, len(video_predictions))
            )
            # The zero endpoint lowers the method order near completion, as in
            # the upstream implementation, to avoid an unstable terminal fit.
            predictor_order = min(
                predictor_order, max(1, len(sigmas) - 2 - index)
            )
            corrector_order = min(
                corrector_order, max(0, len(sigmas) - 1 - index)
            )

            if corrector_order == 0:
                video = predicted_video
                audio = predicted_audio
            else:
                history_lambdas = lambdas[
                    index - corrector_order + 1 : index + 1
                ]
                coefficients = _sa_coefficients(
                    sigma_next=sigmas[index],
                    lambdas=history_lambdas,
                    lambda_start=lambdas[index - 1],
                    lambda_stop=lambdas[index],
                    tau=previous_tau,
                )
                scale = (
                    sigmas[index]
                    / sigmas[index - 1]
                    * math.exp(-(previous_tau * previous_tau) * previous_h)
                )
                video = scale * video + _linear_combination(
                    video_predictions[-corrector_order:], coefficients
                )
                audio = scale * audio + _linear_combination(
                    audio_predictions[-corrector_order:], coefficients
                )
                if previous_tau > 0.0:
                    assert previous_video_noise is not None
                    assert previous_audio_noise is not None
                    video = video + previous_video_noise
                    audio = audio + previous_audio_noise

            if next_sigma == 0.0:
                predicted_video = prediction.video_denoised
                predicted_audio = prediction.audio_denoised
                previous_video_noise = None
                previous_audio_noise = None
            else:
                tau = (
                    1.0
                    if stochastic_stop <= next_sigma <= stochastic_start
                    else 0.0
                )
                history_lambdas = lambdas[
                    index - predictor_order + 1 : index + 1
                ]
                coefficients = _sa_coefficients(
                    sigma_next=next_sigma,
                    lambdas=history_lambdas,
                    lambda_start=lambdas[index],
                    lambda_stop=lambdas[index + 1],
                    tau=tau,
                )
                previous_h = lambdas[index + 1] - lambdas[index]
                scale = (
                    next_sigma
                    / sigmas[index]
                    * math.exp(-(tau * tau) * previous_h)
                )
                predicted_video = scale * video + _linear_combination(
                    video_predictions[-predictor_order:], coefficients
                )
                predicted_audio = scale * audio + _linear_combination(
                    audio_predictions[-predictor_order:], coefficients
                )
                if tau > 0.0:
                    noise_scale = next_sigma * math.sqrt(
                        -math.expm1(-2.0 * tau * tau * previous_h)
                    )
                    if video.shape[0] != audio.shape[0]:
                        raise ValueError("SA-Solver requires equal AV batch sizes")
                    batch = int(video.shape[0])
                    video_width = video.numel() // batch
                    audio_width = audio.numel() // batch
                    # The reference sampler draws one packed AV noise tensor.
                    # Drawing video and audio separately is not numerically
                    # equivalent (even with the same generator), because the
                    # random kernel may consume a shape-dependent sequence.
                    packed_noise = torch.randn(
                        (batch, video_width + audio_width),
                        dtype=video.dtype,
                        layout=video.layout,
                        device=video.device,
                        generator=generator,
                    )
                    previous_video_noise = packed_noise[
                        :, :video_width
                    ].reshape_as(video) * noise_scale
                    previous_audio_noise = packed_noise[
                        :, video_width:
                    ].reshape_as(audio).to(dtype=audio.dtype) * noise_scale
                    predicted_video = predicted_video + previous_video_noise
                    predicted_audio = predicted_audio + previous_audio_noise
                else:
                    previous_video_noise = None
                    previous_audio_noise = None
                previous_tau = tau

            if callback is not None:
                callback(clock.index, clock, predicted_video, predicted_audio)

        return predicted_video, predicted_audio


class TurboClockMode(str, Enum):
    """How the Turbo predictor represents its audio denoised estimate."""

    DUAL_SHIFT = "dual_shift"
    SHARED_VIDEO = "shared_video"


def _time_shift_sigma(sigma: float, source_shift: float, target_shift: float) -> float:
    base = sigma / (source_shift + sigma * (1.0 - source_shift))
    return target_shift * base / (1.0 + (target_shift - 1.0) * base)


def _time_shift_slope(sigma: float, source_shift: float, target_shift: float) -> float:
    base = sigma / (source_shift + sigma * (1.0 - source_shift))
    numerator = target_shift * (1.0 + (source_shift - 1.0) * base) ** 2
    denominator = source_shift * (1.0 + (target_shift - 1.0) * base) ** 2
    return numerator / denominator


class TurboAVSampler:
    """First-order distilled sampler with explicit audio clock ownership."""

    name = "turbo"

    def __init__(self, clock_mode: TurboClockMode = TurboClockMode.DUAL_SHIFT) -> None:
        self.clock_mode = clock_mode

    def sample(
        self,
        video: Any,
        audio: Any,
        plan: SamplingPlan,
        predict: AVPredictor,
        *,
        cancel_check: CancelCheck = lambda: None,
        callback: StepCallback | None = None,
        transition: StepTransition | None = None,
        initial_previous_video: Any | None = None,
        initial_previous_audio: Any | None = None,
        initial_previous_video_sigma: float | None = None,
        initial_previous_audio_sigma: float | None = None,
    ) -> tuple[Any, Any]:
        if any(
            value is not None
            for value in (
                initial_previous_video,
                initial_previous_audio,
                initial_previous_video_sigma,
                initial_previous_audio_sigma,
            )
        ):
            raise ValueError(
                "Turbo sampler does not support RES history"
            )
        for index in range(plan.step_count):
            cancel_check()
            clock = plan.clock(index)
            global_index = clock.index
            prediction = predict(
                video,
                audio,
                clock,
                step_index=global_index,
                is_actual_step=True,
            )
            video_derivative = (video - prediction.video_denoised) / clock.video_sigma
            video = video + (
                clock.video_sigma_next - clock.video_sigma
            ) * video_derivative

            if self.clock_mode is TurboClockMode.SHARED_VIDEO:
                audio_derivative = (audio - prediction.audio_denoised) / clock.video_sigma
                audio = audio + (
                    clock.video_sigma_next - clock.video_sigma
                ) * audio_derivative
            else:
                # Matches the released Turbo adapter's legacy dual-clock
                # contract: model output is differentiated on the video clock,
                # then converted to the audio clock through d sigma_a/d sigma_v.
                slope = _time_shift_slope(
                    max(clock.video_sigma, 1e-6),
                    plan.video_shift,
                    plan.audio_shift,
                )
                audio_derivative = (
                    (audio - prediction.audio_denoised) / clock.video_sigma / slope
                )
                audio_sigma = _time_shift_sigma(
                    clock.video_sigma, plan.video_shift, plan.audio_shift
                )
                audio_sigma_next = _time_shift_sigma(
                    clock.video_sigma_next, plan.video_shift, plan.audio_shift
                )
                audio = audio + (audio_sigma_next - audio_sigma) * audio_derivative
            if transition is not None:
                video, audio, _, _ = transition(
                    global_index,
                    clock,
                    video,
                    audio,
                    None,
                    None,
                )
            if callback is not None:
                callback(global_index, clock, video, audio)
        return video, audio


def create_sampler(
    name: str,
    *,
    turbo_clock_mode: TurboClockMode = TurboClockMode.DUAL_SHIFT,
) -> ResMultistepAVSampler | SASolverAVSampler | TurboAVSampler:
    if name == ResMultistepAVSampler.name:
        return ResMultistepAVSampler()
    if name == TurboAVSampler.name:
        return TurboAVSampler(turbo_clock_mode)
    if name == SASolverAVSampler.name:
        return SASolverAVSampler()
    raise ValueError(f"unsupported native H3 sampler: {name!r}")
