import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.components.protocols import GuiderProtocol


@dataclass(frozen=True)
class CFGGuider(GuiderProtocol):
    """
    Classifier-free guidance (CFG) guider.
    Computes the guidance delta as (scale - 1) * (cond - uncond), steering the
    denoising process toward the conditioned prediction.
    Attributes:
        scale: Guidance strength. 1.0 means no guidance, higher values increase
            adherence to the conditioning.
    """

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        return (self.scale - 1) * (cond - uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class CFGStarRescalingGuider(GuiderProtocol):
    """
    Calculates the CFG delta between conditioned and unconditioned samples.
    To minimize offset in the denoising direction and move mostly along the
    conditioning axis within the distribution, the unconditioned sample is
    rescaled in accordance with the norm of the conditioned sample.
    Attributes:
        scale (float):
            Global guidance strength. A value of 1.0 corresponds to no extra
            guidance beyond the base model prediction. Values > 1.0 increase
            the influence of the conditioned sample relative to the
            unconditioned one.
    """

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        rescaled_neg = projection_coef(cond, uncond) * uncond
        return (self.scale - 1) * (cond - rescaled_neg)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class STGGuider(GuiderProtocol):
    """
    Calculates the STG delta between conditioned and perturbed denoised samples.
    Perturbed samples are the result of the denoising process with perturbations,
    e.g. attentions acting as passthrough for certain layers and modalities.
    Attributes:
        scale (float):
            Global strength of the STG guidance. A value of 0.0 disables the
            guidance. Larger values increase the correction applied in the
            direction of (pos_denoised - perturbed_denoised).
    """

    scale: float

    def delta(self, pos_denoised: torch.Tensor, perturbed_denoised: torch.Tensor) -> torch.Tensor:
        return self.scale * (pos_denoised - perturbed_denoised)

    def enabled(self) -> bool:
        return self.scale != 0.0


@dataclass(frozen=True)
class LtxAPGGuider(GuiderProtocol):
    """
    Calculates the APG (adaptive projected guidance) delta between conditioned
    and unconditioned samples.
    To minimize offset in the denoising direction and move mostly along the
    conditioning axis within the distribution, the (cond - uncond) delta is
    decomposed into components parallel and orthogonal to the conditioned
    sample. The `eta` parameter weights the parallel component, while `scale`
    is applied to the orthogonal component. Optionally, a norm threshold can
    be used to suppress guidance when the magnitude of the correction is small.
    Attributes:
        scale (float):
            Strength applied to the component of the guidance that is orthogonal
            to the conditioned sample. Controls how aggressively we move in
            directions that change semantics but stay consistent with the
            conditioning manifold.
        eta (float):
            Weight of the component of the guidance that is parallel to the
            conditioned sample. A value of 1.0 keeps the full parallel
            component; values in [0, 1] attenuate it, and values > 1.0 amplify
            motion along the conditioning direction.
        norm_threshold (float):
            Minimum L2 norm of the guidance delta below which the guidance
            can be reduced or ignored (depending on implementation).
            This is useful for avoiding noisy or unstable updates when the
            guidance signal is very small.
    """

    scale: float
    eta: float = 1.0
    norm_threshold: float = 0.0

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        guidance = cond - uncond
        if self.norm_threshold > 0:
            ones = torch.ones_like(guidance)
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor
        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth

        return g_apg * (self.scale - 1)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=False)
class LegacyStatefulAPGGuider(GuiderProtocol):
    """
    Calculates the APG (adaptive projected guidance) delta between conditioned
    and unconditioned samples.
    To minimize offset in the denoising direction and move mostly along the
    conditioning axis within the distribution, the (cond - uncond) delta is
    decomposed into components parallel and orthogonal to the conditioned
    sample. The `eta` parameter weights the parallel component, while `scale`
    is applied to the orthogonal component. Optionally, a norm threshold can
    be used to suppress guidance when the magnitude of the correction is small.
    Attributes:
        scale (float):
            Strength applied to the component of the guidance that is orthogonal
            to the conditioned sample. Controls how aggressively we move in
            directions that change semantics but stay consistent with the
            conditioning manifold.
        eta (float):
            Weight of the component of the guidance that is parallel to the
            conditioned sample. A value of 1.0 keeps the full parallel
            component; values in [0, 1] attenuate it, and values > 1.0 amplify
            motion along the conditioning direction.
        norm_threshold (float):
            Minimum L2 norm of the guidance delta below which the guidance
            can be reduced or ignored (depending on implementation).
            This is useful for avoiding noisy or unstable updates when the
            guidance signal is very small.
        momentum (float):
            Exponential moving-average coefficient for accumulating guidance
            over time. running_avg = momentum * running_avg + guidance
    """

    scale: float
    eta: float
    norm_threshold: float = 5.0
    momentum: float = 0.0
    # it is user's responsibility not to use same APGGuider for several denoisings or different modalities
    # in order not to share accumulated average across different denoisings or modalities
    running_avg: torch.Tensor | None = None

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        guidance = cond - uncond
        if self.momentum != 0:
            if self.running_avg is None:
                self.running_avg = guidance.clone()
            else:
                self.running_avg = self.momentum * self.running_avg + guidance
            guidance = self.running_avg

        if self.norm_threshold > 0:
            ones = torch.ones_like(guidance)
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor

        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth

        return g_apg * self.scale

    def enabled(self) -> bool:
        return self.scale != 0.0


@dataclass(frozen=True)
class MultiModalGuiderParams:
    """
    Parameters for the multi-modal guider.
    """

    cfg_scale: float = 1.0
    "CFG (Classifier-free guidance) scale controlling how strongly the model adheres to the prompt."
    stg_scale: float = 0.0
    "STG (Spatio-Temporal Guidance) scale controls how strongly the model reacts to the perturbation of the modality."
    stg_blocks: list[int] | None = field(default_factory=list)
    "Which transformer blocks to perturb for STG."
    rescale_scale: float = 0.0
    "Rescale scale controlling how strongly the model rescales the modality after applying other guidance."
    modality_scale: float = 1.0
    "Modality scale controlling how strongly the model reacts to the perturbation of the modality."
    skip_step: int = 0
    "Skip step controlling how often the model skips the step."


def _masked_std(x: torch.Tensor, active: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Unbiased std of ``x`` over its ``count`` ``active`` elements (``active`` broadcast to ``x``).
    Inactive elements are replaced by the masked mean, so they add nothing to the squared deviations,
    and ``std`` is rescaled from the full element count to ``count``. With every element active this is
    bit-identical to ``x.std()``. Boolean indexing (``x[active].std()``) would force a host sync.
    """
    mean = torch.where(active, x, 0.0).sum() / count
    return torch.where(active, x, mean).std() * ((x.numel() - 1) / (count - 1)).sqrt()


def _params_for_sigma_from_sorted_dict(
    sigma: float, params_by_sigma: Sequence[tuple[float, MultiModalGuiderParams]]
) -> MultiModalGuiderParams:
    """
    Return params for the given sigma from a sorted (sigma_upper_bound -> params) structure.
    Keys are sorted descending (bin upper bounds). Bin i is (key_{i+1}, key_i].
    Get all keys >= sigma; use last in list (smallest such key = upper bound of bin containing sigma),
    or last entry in the sequence if list is empty (sigma above max key).
    """
    if not params_by_sigma:
        raise ValueError("params_by_sigma must be non-empty")
    sigma = float(sigma)
    keys_desc = [k for k, _ in params_by_sigma]
    keys_ge_sigma = [k for k in keys_desc if k >= sigma]
    # sigma above all keys: use first bin (max key)
    key = keys_ge_sigma[-1] if keys_ge_sigma else keys_desc[0]
    return next(p for k, p in params_by_sigma if k == key)


@dataclass(frozen=True)
class MultiModalGuider:
    """
    Multi-modal guider with constant params per instance.
    For sigma-dependent params, use MultiModalGuiderFactory.build_from_sigma(sigma) to
    obtain a guider for each step.
    """

    params: MultiModalGuiderParams
    negative_context: torch.Tensor | None = None

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
        denoise_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The guider calculates the guidance delta as (scale - 1) * (cond - uncond) for cfg and modality cfg,
        and as scale * (cond - uncond) for stg, steering the denoising process away from the unconditioned
        prediction.
        When ``denoise_mask`` (broadcastable to ``cond``, e.g. ``(B, T, 1)``) is given, the rescale factor
        is estimated only over tokens with a non-zero mask, so fixed conditioning and reference tokens do
        not skew it. With no such tokens, no rescaling is applied.
        """
        dtype = cond.dtype
        cond = cond.float()
        uncond_text = uncond_text.float() if isinstance(uncond_text, torch.Tensor) else uncond_text
        uncond_perturbed = uncond_perturbed.float() if isinstance(uncond_perturbed, torch.Tensor) else uncond_perturbed
        uncond_modality = uncond_modality.float() if isinstance(uncond_modality, torch.Tensor) else uncond_modality
        pred = (
            cond
            + (self.params.cfg_scale - 1) * (cond - uncond_text)
            + self.params.stg_scale * (cond - uncond_perturbed)
            + (self.params.modality_scale - 1) * (cond - uncond_modality)
        )

        if self.params.rescale_scale != 0:
            if denoise_mask is None:
                factor = cond.std() / pred.std()
            else:
                active = (denoise_mask != 0).expand_as(cond)
                count = active.sum()
                masked_factor = _masked_std(cond, active, count) / _masked_std(pred, active, count)
                factor = torch.where(count > 1, masked_factor, 1.0)
            factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
            pred = pred * factor

        return pred.to(dtype)

    def do_unconditional_generation(self) -> bool:
        """Returns True if the guider is doing unconditional generation."""
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        """Returns True if the guider is doing perturbed generation."""
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        """Returns True if the guider is doing isolated modality generation."""
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        """Returns True if the guider should skip the step."""
        if self.params.skip_step == 0:
            return False
        return step % (self.params.skip_step + 1) != 0


@dataclass(frozen=True)
class MultiModalGuiderFactory:
    """
    Factory that creates a MultiModalGuider for a given sigma.
    Single source of truth: _params_by_sigma (schedule). Use constant() for
    one params for all sigma, from_dict() for sigma-binned params.
    """

    negative_context: torch.Tensor | None = None
    _params_by_sigma: tuple[tuple[float, MultiModalGuiderParams], ...] = ()

    @classmethod
    def constant(
        cls,
        params: MultiModalGuiderParams,
        negative_context: torch.Tensor | None = None,
    ) -> "MultiModalGuiderFactory":
        """Build a factory with constant params (same guider for all sigma)."""
        return cls(
            negative_context=negative_context,
            _params_by_sigma=((float("inf"), params),),
        )

    @classmethod
    def from_dict(
        cls,
        sigma_to_params: Mapping[float, MultiModalGuiderParams],
        negative_context: torch.Tensor | None = None,
    ) -> "MultiModalGuiderFactory":
        """
        Build a factory from a dict of sigma_value -> MultiModalGuiderParams.
        Keys are sorted descending and used for bin lookup in params(sigma).
        """
        if not sigma_to_params:
            raise ValueError("sigma_to_params must be non-empty")
        sorted_items = tuple(sorted(sigma_to_params.items(), key=lambda x: x[0], reverse=True))
        return cls(negative_context=negative_context, _params_by_sigma=sorted_items)

    def params(self, sigma: float | torch.Tensor) -> MultiModalGuiderParams:
        """Return params effective for the given sigma (getter; single source of truth)."""
        sigma_val = float(sigma.item() if isinstance(sigma, torch.Tensor) else sigma)
        return _params_for_sigma_from_sorted_dict(sigma_val, self._params_by_sigma)

    def build_from_sigma(self, sigma: float | torch.Tensor) -> MultiModalGuider:
        """Return a MultiModalGuider with params effective for the given sigma."""
        return MultiModalGuider(
            params=self.params(sigma),
            negative_context=self.negative_context,
        )


def create_multimodal_guider_factory(
    params: MultiModalGuiderParams | MultiModalGuiderFactory,
    negative_context: torch.Tensor | None = None,
) -> MultiModalGuiderFactory:
    """
    Create or return a MultiModalGuiderFactory. Pass constant params for a
    single-params factory (uses MultiModalGuiderFactory.constant), or an existing
    MultiModalGuiderFactory. When given a factory, returns it as-is unless
    negative_context is provided. For sigma-dependent params use
    MultiModalGuiderFactory.from_dict(...) and pass that as params.
    """
    if isinstance(params, MultiModalGuiderFactory):
        if negative_context is not None and params.negative_context is not negative_context:
            return MultiModalGuiderFactory.from_dict(dict(params._params_by_sigma), negative_context=negative_context)
        return params
    return MultiModalGuiderFactory.constant(params, negative_context=negative_context)


def projection_coef(to_project: torch.Tensor, project_onto: torch.Tensor) -> torch.Tensor:
    batch_size = to_project.shape[0]
    positive_flat = to_project.reshape(batch_size, -1)
    negative_flat = project_onto.reshape(batch_size, -1)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm
