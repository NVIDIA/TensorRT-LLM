import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.conditioning.exceptions import ConditioningError
from tensorrt_llm._torch.visual_gen.models.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx_core.tools import LatentTools
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import LatentState


class VideoConditionByLatentIndex(ConditioningItem):
    """
    Conditions video generation by injecting latents at a specific latent frame index.
    Sets the clean latents at positions corresponding to latent_idx to the injected latents,
    sets denoise strength according to the strength parameter.
    """

    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        cond_batch, cond_channels, _, cond_height, cond_width = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_height, tgt_width = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_height, cond_width) != (tgt_batch, tgt_channels, tgt_height, tgt_width):
            raise ConditioningError(
                f"Can't apply image conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"shape is ({tgt_batch}, {tgt_channels}, {tgt_frames}, {tgt_height}, {tgt_width}). Make sure "
                "the image and latent have the same spatial shape."
            )

        return _apply_condition_by_latent_index(
            latent_state,
            latent_tools,
            latent=self.latent,
            strength=self.strength,
            latent_idx=self.latent_idx,
        )


class AudioConditionByLatentIndex(ConditioningItem):
    """Inject audio latents at a latent-frame index and set denoise strength on that span.
    Audio counterpart of :class:`VideoConditionByLatentIndex`. Full-coverage freeze is
    ``latent_idx=0`` with ``strength=1.0`` over the whole audio latent.
    """

    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        cond_batch, cond_channels, _, cond_mel = self.latent.shape
        tgt_batch, tgt_channels, _, tgt_mel = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_mel) != (tgt_batch, tgt_channels, tgt_mel):
            raise ConditioningError(
                f"Can't apply audio conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"matching batch/channels/mel (got conditioning {(cond_batch, cond_channels, cond_mel)})."
            )

        return _apply_condition_by_latent_index(
            latent_state,
            latent_tools,
            latent=self.latent,
            strength=self.strength,
            latent_idx=self.latent_idx,
        )


def _apply_condition_by_latent_index(
    latent_state: LatentState,
    latent_tools: LatentTools,
    *,
    latent: torch.Tensor,
    strength: float,
    latent_idx: int,
) -> LatentState:
    """Inject patchified ``latent`` at ``latent_idx`` and set denoise strength on that span."""
    tokens = latent_tools.patchifier.patchify(latent)
    start_token = latent_tools.patchifier.get_token_count(latent_tools.target_shape._replace(frames=latent_idx))
    stop_token = start_token + tokens.shape[1]

    latent_state = latent_state.clone()
    latent_state.clean_latent[:, start_token:stop_token] = tokens
    latent_state.denoise_mask.tensor[:, start_token:stop_token] = 1.0 - strength
    return latent_state
