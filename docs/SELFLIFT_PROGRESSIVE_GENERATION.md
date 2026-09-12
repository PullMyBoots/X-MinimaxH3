# SelfLift progressive H3 generation

The first-generation API can run one H3 Base RES or Larry Turbo trajectory at
two spatial resolutions. Set `selflift_enabled=true`, choose a smaller
`selflift_initial_resolution`, and provide the number of completed
low-resolution steps in `selflift_transition_step`.

For an eight-step 1080P request with a 540P start, the default transition is:

1. Run steps 1–6 on the aligned 960×544 canvas.
2. Take the clean H3 latent (`x0`) predicted by step 6.
3. Lift `x0` to the 1920×1088 latent grid with the pinned MiniMax H3 learned
   3D latent upscaler.
4. Reconstruct the next rectified-flow state as
   `x(sigma_next) = sigma_next * noise + (1 - sigma_next) * lifted_x0`.
5. Run steps 7–8 on the 1920×1088 canvas and decode once at the end.

Larry resumes its first-order Turbo trajectory directly. Base preserves the
second-order RES trajectory by replacing the boundary's low-resolution x0
history with the same learned high-resolution x0 used to rebuild the sampler
state. The boundary step and every step after it are forced to real DiT
evaluations; sparse Attention may still accelerate those evaluations.

The transformer and latent upscaler use separate residency phases, so their
weights do not occupy GPU memory together. The public receipt is stored under
`inference_plan.selflift`, including the two canvas sizes, transition sigma,
step split, noise seed, and latent shapes.

This integration follows the progressive clean-endpoint lifting mechanism in
[facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift), inspected
at revision `b143ed2731438e606c7a6c70ee580a1f86e7bf72`. No source code was copied
from that repository. The current H3 product route uses its practical learned
latent-lift path with `rho=0`; it does not claim to implement the optional
pixel-VAE SelfLift-zero correction.

Current constraints: H3 Base RES or Larry LoRA/Turbo, text-to-video, complete
execution, one physical generation window, and an initial canvas smaller than
the requested output canvas.
