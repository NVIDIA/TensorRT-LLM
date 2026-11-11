import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.protocols import DiffusionStepProtocol
from tensorrt_llm._torch.visual_gen.models.ltx_core.utils import to_velocity


def _get_ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(sigma_down, sigma_up)`` for one DDIM ancestral sampling step.
    Both inputs are in the rescaled parameterization ``sigma / alpha``.
    Returns ``sigma_down`` (deterministic component) and ``sigma_up``
    (stochastic component) in the same rescaled space.
    """
    if not eta:
        return sigma_to, torch.zeros_like(sigma_to)
    variance = sigma_to**2 * (sigma_from**2 - sigma_to**2).clamp(min=0) / sigma_from**2
    sigma_up = (eta * variance**0.5).clamp(max=sigma_to)
    sigma_down = (sigma_to**2 - sigma_up**2).clamp(min=0) ** 0.5
    return sigma_down, sigma_up


class EulerDiffusionStep(DiffusionStepProtocol):
    """
    First-order Euler method for diffusion sampling.
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int, **_kwargs
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)


class EulerAncestralDiffusionStep(DiffusionStepProtocol):
    """Ancestral (SDE) Euler step for rectified-flow models.
    Each step advances deterministically to an intermediate noise level
    ``sigma_down <= sigma_next`` and then renoises back up to ``sigma_next``,
    rescaling the signal component by ``alpha_next / alpha_down`` so the
    transition stays variance-preserving. ``eta`` interpolates between a plain
    Euler step (``eta=0``, ``sigma_down == sigma_next``, no noise added) and a
    fully ancestral step (``eta=1``).
    This is the rectified-flow parameterization (``alpha = 1 - sigma``), which
    is the one that applies to LTX-2. It deliberately does **not** reuse
    :func:`_get_ancestral_step`: that helper implements the DDIM/variance-
    exploding ancestral coefficients, which give a different ``sigma_down`` and
    a different amount of injected noise for the same ``eta``. The two agree
    only at ``eta=0``.
    """

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        noise: torch.Tensor | None = None,
        eta: float = 1.0,
        s_noise: float = 1.0,
        **_kwargs,
    ) -> torch.Tensor:
        """Advance one ancestral Euler step.
        Args:
            sample: Current noisy latent x_t.
            denoised_sample: Denoised prediction x_0.
            sigmas: Full sigma schedule tensor.
            step_index: Current step index.
            noise: Noise tensor for the renoise term. Required when ``eta > 0``;
                unused (and may be ``None``) when ``eta == 0``.
            eta: Stochastic noise injection strength. ``0`` is a plain Euler
                step; ``1`` is fully ancestral. Default ``1.0``.
            s_noise: Scale on the injected noise. Default ``1.0``. At ``0`` the
                step still applies its variance-preserving rescale.
        Returns:
            Updated latent x_{t-1}, or ``denoised_sample`` when the next sigma is 0.
        """
        sigma = sigmas[step_index].to(torch.float32)
        sigma_next = sigmas[step_index + 1].to(torch.float32)
        if sigma_next == 0:
            return denoised_sample.to(sample.dtype)
        if eta > 0 and noise is None:
            raise ValueError("EulerAncestralDiffusionStep requires a noise tensor when eta > 0")

        x = sample.to(torch.float32)
        denoised = denoised_sample.to(torch.float32)

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio

        # Euler step to sigma_down, expressed as an interpolation between x and x_0.
        sigma_down_ratio = sigma_down / sigma
        x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised

        if eta > 0:
            # Renoise from sigma_down back up to sigma_next.
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
            x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * s_noise * renoise_coeff
        return x_next.to(sample.dtype)


class Res2sDiffusionStep(DiffusionStepProtocol):
    """
    Second-order diffusion step for res_2s sampling with SDE noise injection.
    Used by the res_2s denoising loop. Advances the sample from the current
    sigma to the next by mixing a deterministic update (from the denoised
    prediction) with injected noise via ``get_sde_coeff``, producing
    variance-preserving transitions.
    """

    @staticmethod
    def get_sde_coeff(
        sigma_next: torch.Tensor,
        sigma_up: torch.Tensor | None = None,
        sigma_down: torch.Tensor | None = None,
        sigma_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SDE coefficients (alpha_ratio, sigma_down, sigma_up) for the step.
        Given either ``sigma_down`` or ``sigma_up``, returns the mixing
        coefficients used for variance-preserving noise injection. If
        ``sigma_up`` is provided, ``sigma_down`` and ``alpha_ratio`` are
        derived; if ``sigma_down`` is provided, ``sigma_up`` and
        ``alpha_ratio`` are derived.
        """
        if sigma_down is not None:
            alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
            sigma_up = (sigma_next**2 - sigma_down**2 * alpha_ratio**2).clamp(min=0) ** 0.5
        elif sigma_up is not None:
            # Fallback to avoid sqrt(neg_num)
            sigma_up.clamp_(max=sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else torch.ones_like(sigma_next)
            sigma_signal = sigmax - sigma_next
            sigma_residual = (sigma_next**2 - sigma_up**2).clamp(min=0) ** 0.5
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            sigma_down = sigma_next
            sigma_up = torch.zeros_like(sigma_next)

        sigma_up = torch.nan_to_num(sigma_up if sigma_up is not None else torch.zeros_like(sigma_next), 0.0)
        # Replace NaNs in sigma_down with corresponding sigma_next elements (float32)
        sigma_down = torch.where(torch.isnan(sigma_down), sigma_next.to(sigma_down.dtype), sigma_down)
        alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

        return alpha_ratio, sigma_down, sigma_up

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        noise: torch.Tensor,
        eta: float = 0.5,
    ) -> torch.Tensor:
        """Advance one step with SDE noise injection via get_sde_coeff.
        Args:
            sample: Current noisy sample.
            denoised_sample: Denoised prediction from the model.
            sigmas: Noise schedule tensor.
            step_index: Current step index in the schedule.
            noise: Random noise tensor for stochastic injection.
            eta: Controls stochastic noise injection strength (0=deterministic, 1=maximum). Default 0.5.
        Returns:
            Next sample with SDE noise injection applied.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        alpha_ratio, sigma_down, sigma_up = self.get_sde_coeff(sigma_next, sigma_up=sigma_next * eta)
        output_dtype = denoised_sample.dtype
        return_denoised = torch.any(sigma_up == 0) | torch.any(sigma_next == 0)

        # Extract epsilon prediction
        eps_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next

        # Mix deterministic and stochastic components
        x_noised = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
        return torch.where(return_denoised, denoised_sample, x_noised.to(output_dtype))


class EulerCfgPpDiffusionStep(DiffusionStepProtocol):
    """Euler step using the CFG++ correction for the ODE derivative.
    Instead of the standard velocity formula, the ODE derivative is computed
    from the unconditioned prediction, keeping the conditioned prediction as
    the target denoised state.  Ancestral (DDIM) noise injection is applied
    in the rescaled sigma parameterization (sigma / alpha).
    All diffusion quantities (alpha, ODE derivative, ancestral coefficients)
    are computed internally from ``sigmas`` and ``uncond_denoised``.
    Reference: CFG++ (https://arxiv.org/abs/2406.08070).
    """

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        uncond_denoised: torch.Tensor,
        noise: torch.Tensor | None = None,
        eta: float = 1.0,
        s_noise: float = 1.0,
        **_kwargs,
    ) -> torch.Tensor:
        """Advance one CFG++ Euler step.
        Args:
            sample: Current noisy latent x_t.
            denoised_sample: Conditioned denoised prediction x_0^cond.
            sigmas: Full sigma schedule tensor.
            step_index: Current step index.
            uncond_denoised: Unconditioned denoised prediction x_0^uncond,
                used to compute the ODE derivative direction.
            noise: Noise tensor for stochastic injection; ignored when
                ``eta=0`` or ``s_noise=0``.
            eta: Stochastic noise injection strength. Default ``1.0``.
            s_noise: Scale on the injected noise. Default ``1.0``.
        Returns:
            Updated latent x_{t-1}.
        """
        sigma_s = sigmas[step_index].to(torch.float32)
        sigma_t = sigmas[step_index + 1].to(torch.float32)
        _eps = torch.finfo(torch.float32).eps
        # Clamp to avoid division by zero when sigma == 1.0 exactly.
        alpha_s = (1.0 - sigma_s).clamp(min=_eps)
        alpha_t = (1.0 - sigma_t).clamp(min=_eps)

        x = sample.to(torch.float32)
        denoised = denoised_sample.to(torch.float32)
        uncond = uncond_denoised.to(torch.float32)

        # ODE derivative: direction toward noise using uncond prediction (CFG++ correction)
        d = (x - alpha_s * uncond) / sigma_s

        # Ancestral step in rescaled sigma space (sigma / alpha)
        sigma_down, sigma_up = _get_ancestral_step(sigma_s / alpha_s, sigma_t / alpha_t, eta=eta)
        sigma_down = alpha_t * sigma_down

        x_next = alpha_t * denoised + sigma_down * d
        if noise is not None and eta > 0 and s_noise > 0:
            x_next = x_next + alpha_t * noise.to(torch.float32) * s_noise * sigma_up
        return x_next.to(sample.dtype)
