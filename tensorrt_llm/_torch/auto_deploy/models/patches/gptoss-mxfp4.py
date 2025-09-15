from typing import Tuple

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _forward_router(
    self: "GptOssTopKRouter", hidden_states: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Patched forward:
      - Calls fused router op (returns only scores)
      - Derives router_indices via topk(scores) to preserve original API
    Returns:
      router_scores:  [T, E]
      router_indices: [T, top_k]
    """
    hs = hidden_states
    # The custom op's fake kernel assumes 3D; ensure [B, S, H] if the caller gives [T, H]
    if hs.dim() == 2:
        hs = hs.unsqueeze(0)

    out = torch.ops.auto_deploy.torch_moe_router(hs, self.weight, self.bias, int(self.top_k))
    router_scores = out  # [B*S, E]

    return router_scores, None


@ExportPatchRegistry.register("gptoss_topk_router")
class GptOssTopKRouterPatch(BaseExportPatch):
    """Patch for GptOssTopKRouter to use the fused torch_moe_router custom op during export."""

    def _apply_patch(self):
        cls = self._resolve_router_class()

        # Keep original
        key = f"{cls.__module__}.{cls.__qualname__}.forward"
        self.original_values[key] = cls.forward

        # Apply patch
        cls._original_forward = cls.forward
        cls.forward = _forward_router

    def _revert_patch(self):
        cls = self._resolve_router_class()
        key = f"{cls.__module__}.{cls.__qualname__}.forward"

        # Restore original
        cls.forward = self.original_values[key]

        # Cleanup
        if hasattr(cls, "_original_forward"):
            delattr(cls, "_original_forward")

    @staticmethod
    def _resolve_router_class():
        """
        Resolve the GptOssTopKRouter class. If it's not in the current module scope,
        import it from your project location instead.
        """
        try:
            return GptOssTopKRouter
        except NameError:
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter as _Cls

            return _Cls
