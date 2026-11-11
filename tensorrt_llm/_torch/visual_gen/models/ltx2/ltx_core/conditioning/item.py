from typing import Protocol

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import LatentTools
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import LatentState


class ConditioningItem(Protocol):
    """Protocol for conditioning items that modify latent state during diffusion."""

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        """
        Apply the conditioning to the latent state.
        Args:
            latent_state: The latent state to apply the conditioning to. This is state always patchified.
        Returns:
            The latent state after the conditioning has been applied.
        IMPORTANT: If the conditioning needs to add extra tokens to the latent, it should add them to the end of the
        latent.
        """
        ...
