"""
Diffusion pipeline components.
Submodules:
    diffusion_steps - Diffusion stepping algorithms (EulerDiffusionStep, EulerAncestralDiffusionStep)
    guiders         - Guidance strategies (CFGGuider, STGGuider, APG variants)
    noisers         - Noise samplers (GaussianNoiser)
    patchifiers     - Latent patchification (VideoLatentPatchifier, AudioPatchifier)
    protocols       - Protocol definitions (Patchifier, etc.)
    schedulers      - Sigma schedulers (LTX2Scheduler, LinearQuadraticScheduler)
"""
