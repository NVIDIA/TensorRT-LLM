from dataclasses import dataclass

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.patchifiers import get_pixel_coords
from tensorrt_llm._torch.visual_gen.models.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx_core.tools import LatentTools, SpatioTemporalScaleFactors
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape


@dataclass(frozen=True)
class TemporalRegionMask(ConditioningItem):
    """Conditioning item that sets ``denoise_mask = 0`` outside a time range
    and ``1`` inside, so only the specified temporal region is regenerated.
    Uses ``start_time`` and ``end_time`` in seconds. Works in *patchified*
    (token) space using the patchifier's ``get_patch_grid_bounds``: for video
    coords are latent frame indices (converted from seconds via ``fps``), for
    audio coords are already in seconds.
    """

    start_time: float  # seconds, inclusive
    end_time: float  # seconds, exclusive
    fps: float

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        coords = latent_tools.patchifier.get_patch_grid_bounds(
            latent_tools.target_shape, device=latent_state.denoise_mask.tensor.device
        )
        if isinstance(latent_tools.target_shape, AudioLatentShape):
            # Audio: patchifier get_patch_grid_bounds returns seconds
            t_boundaries = coords[:, 0]
        elif isinstance(latent_tools.target_shape, VideoLatentShape):
            # Video: patchifier get_patch_grid_bounds returns latent bounds, converting to frame numbers & pixel bounds
            scale_factors = getattr(latent_tools, "scale_factors", SpatioTemporalScaleFactors.default())
            pixel_bounds = get_pixel_coords(coords, scale_factors, causal_fix=getattr(latent_tools, "causal_fix", True))
            # converting frame numbers to seconds
            t_boundaries = pixel_bounds[:, 0] / self.fps
        else:
            raise ValueError("Unsupported LatentShape type, expected AudioLatentShape or VideoLatentShape")
        t_start, t_end = t_boundaries.unbind(dim=-1)  # [B, N]
        in_region = (t_end > self.start_time) & (t_start < self.end_time)
        state = latent_state.clone()
        mask_val = in_region.to(state.denoise_mask.tensor.dtype)
        if state.denoise_mask.tensor.dim() == 3:
            mask_val = mask_val.unsqueeze(-1)
        state.denoise_mask.tensor.copy_(mask_val)
        return state
