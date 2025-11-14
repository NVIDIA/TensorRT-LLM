"""Patch for torch.export.export to detect and replace hf attention_interface with unified attention."""

from typing import Optional

import torch
import torch.export as te
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..interface import BaseExportPatch, ExportPatchRegistry

# Kwargs mapping for HF attention_interface to auto_deploy::torch_attention
HF_ATTN_KWARGS_MAPPING = {
    "dropout": "dropout_p",
    "is_causal": "is_causal",
    "scaling": "scale",
    "scale": "scale",
    "s_aux": "sinks",
    "sinks": "sinks",
    "sliding_window": "sliding_window",
    "logit_cap": "logit_cap",
}


def torch_attention_hf_wrapper(
    self: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    """Wrapper of auto_deploy::torch_attention with HF attention_interface signature."""

    # Convert from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    ad_attn_kwargs = {
        HF_ATTN_KWARGS_MAPPING[k]: v for k, v in kwargs.items() if k in HF_ATTN_KWARGS_MAPPING
    }

    attn_output = torch.ops.auto_deploy.torch_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        layout="bsnd",
        **ad_attn_kwargs,
    )

    return attn_output, None


@ExportPatchRegistry.register("unified_attn")
class UnifiedAttnPatch(BaseExportPatch):
    """
    Patch on torch.export.export to replace attention_interface with torch.ops.auto_deploy.torch_attention.
    """

    def _apply_patch(self):
        """Apply the te.export patch."""
        # Store original torch.export.export
        self.original_values["te.export"] = te.export

        # Register the wrapper function
        ALL_ATTENTION_FUNCTIONS.register("ad_unified_attn", torch_attention_hf_wrapper)

        def _export_with_unified_attn(model, *args, **kwargs):
            # torch_export_to_gm is called at both export stage and attn matching stage
            # we only patch attn implementation for export stage
            if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "ad_unified_attn"
            return self.original_values["te.export"](model, *args, **kwargs)

        # Apply patch
        te.export = _export_with_unified_attn

    def _revert_patch(self):
        """Revert the te.export patch."""
        te.export = self.original_values["te.export"]
