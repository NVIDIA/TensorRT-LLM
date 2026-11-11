"""Wrapper conditioning item that adds attention masking to any inner conditioning."""

from dataclasses import replace

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.mask_utils import update_attention_mask
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import LatentTools
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import LatentState


class ConditioningItemAttentionStrengthWrapper(ConditioningItem):
    """Wraps a conditioning item to add an attention mask for its tokens.
    Separates the *attention-masking* concern from the underlying conditioning
    logic (token layout, positional encoding, denoise strength).  The inner
    conditioning item appends tokens to the latent sequence as usual, and this
    wrapper then builds or updates the self-attention mask so that the newly
    added tokens interact with the noisy tokens according to *attention_mask*.
    Args:
        conditioning: Any conditioning item that appends tokens to the latent.
        attention_mask: Per-token attention weight controlling how strongly the
            new conditioning tokens attend to/from noisy tokens.  Can be a
            scalar (float) applied uniformly, or a tensor of shape ``(B, M)``
            for spatial control, where ``M = F * H * W`` is the number of
            patchified conditioning tokens.  Values in ``[0, 1]``.
    Example::
        cond = ConditioningItemAttentionStrengthWrapper(
            VideoConditionByReferenceLatent(latent=ref, strength=1.0),
            attention_mask=0.5,
        )
        state = cond.apply_to(latent_state, latent_tools)
    """

    def __init__(
        self,
        conditioning: ConditioningItem,
        attention_mask: float | torch.Tensor,
    ):
        self.conditioning = conditioning
        self.attention_mask = attention_mask

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: LatentTools,
    ) -> LatentState:
        """Apply inner conditioning, then build the attention mask for its tokens."""
        # Snapshot the original state for mask building
        original_state = latent_state

        # Inner conditioning appends tokens (positions, denoise mask, etc.)
        new_state = self.conditioning.apply_to(latent_state, latent_tools)

        num_new_tokens = new_state.latent.shape[1] - original_state.latent.shape[1]
        if num_new_tokens == 0:
            return new_state

        # Build the attention mask using the *original* state as the reference
        # so that the block structure is computed correctly.
        new_attention_mask = update_attention_mask(
            latent_state=original_state,
            attention_mask=self.attention_mask,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=num_new_tokens,
            batch_size=new_state.latent.shape[0],
            device=new_state.latent.device,
            dtype=new_state.latent.dtype,
        )

        return replace(new_state, attention_mask=new_attention_mask)
