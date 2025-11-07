"""Patch for Llama4 Rotary Embedding to use custom polar op."""
# TODO(fridah): remove this patch once the issue introduced in https://github.com/pytorch/pytorch/pull/160894 is solved

import torch
from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding

from ...export.interface import BaseExportPatch, ExportPatchRegistry


@torch.library.custom_op("auto_deploy::polar", mutates_args=())
def polar_custom(abs: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Custom wrapper for torch.polar."""
    real = abs * torch.cos(angle)
    imag = abs * torch.sin(angle)
    return torch.complex(real, imag)


@polar_custom.register_fake
def polar_custom_fake(abs: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Fake implementation for export."""
    out_dtype = torch.complex64 if abs.dtype == torch.float32 else torch.complex128
    return torch.empty_like(abs, dtype=out_dtype)


def _forward_rope(self: Llama4TextRotaryEmbedding, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = (
        x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)

        # Use custom polar op instead of torch.polar
        freqs_cis = torch.ops.auto_deploy.polar(torch.ones_like(freqs), freqs)
        freqs_cis = freqs_cis * self.attention_scaling

    return freqs_cis


@ExportPatchRegistry.register("hf_llama4_rope")
class Llama4RopePatch(BaseExportPatch):
    """
    Patch for Llama4 Rotary Embedding to make it compatible with torch.export.

    This patch replaces torch.polar with our custom polar op in the
    Llama4TextRotaryEmbedding forward method to avoid shape prop issues
    with aten::polar Meta kernel during export/tracing.
    """

    def _apply_patch(self):
        """Apply the Llama4 RoPE patch."""
        self.original_values["Llama4TextRotaryEmbedding.forward"] = (
            Llama4TextRotaryEmbedding.forward
        )

        Llama4TextRotaryEmbedding._original_forward = Llama4TextRotaryEmbedding.forward

        Llama4TextRotaryEmbedding.forward = _forward_rope

    def _revert_patch(self):
        """Revert the Llama4 RoPE patch."""
        Llama4TextRotaryEmbedding.forward = self.original_values[
            "Llama4TextRotaryEmbedding.forward"
        ]

        if hasattr(Llama4TextRotaryEmbedding, "_original_forward"):
            delattr(Llama4TextRotaryEmbedding, "_original_forward")
