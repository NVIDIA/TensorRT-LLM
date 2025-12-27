"""Patch for torch.export.export to detect and replace hf attention_interface with unified attention."""

import json
from typing import Dict, Optional

import torch
import torch.export as te
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..interface import BaseExportPatch, ExportPatchRegistry

# Default kwargs mapping for HF attention_interface to auto_deploy::torch_attention
# Model patches can override this by setting `_ad_attn_kwargs_mapping` on the attention class
HF_ATTN_KWARGS_MAPPING: Dict[str, str] = {
    "dropout": "dropout_p",
    "is_causal": "is_causal",
    "scaling": "scale",
    "scale": "scale",
    "s_aux": "sinks",
    "sinks": "sinks",
    "sliding_window": "sliding_window",
    "logit_cap": "logit_cap",
}


def _needs_custom_mask(attn_module: torch.nn.Module) -> bool:
    """Check if this attention module needs custom masking.

    Model patches set `_ad_needs_custom_mask = True` on attention classes
    that require custom mask generation (e.g., VLMs with bidirectional attention).

    Args:
        attn_module: The HF attention module (e.g., Gemma3Attention).

    Returns:
        True if custom mask marker should be inserted.
    """
    return getattr(attn_module, "_ad_needs_custom_mask", False)


def _get_layer_idx(attn_module: torch.nn.Module) -> int:
    """Extract layer index from attention module.

    HF attention modules have layer_idx attribute set in __init__.

    Args:
        attn_module: The HF attention module.

    Returns:
        Layer index, or -1 if not found.
    """
    return getattr(attn_module, "layer_idx", -1)


def _get_model_type(attn_module: torch.nn.Module) -> str:
    """Get model type from attention module's config.

    Args:
        attn_module: The HF attention module.

    Returns:
        Model type string (e.g., "gemma3_text").
    """
    config = getattr(attn_module, "config", None)
    if config is None:
        return "unknown"
    return getattr(config, "model_type", "unknown")


def _get_sliding_window(attn_module: torch.nn.Module) -> int:
    """Get sliding window size from attention module.

    Args:
        attn_module: The HF attention module.

    Returns:
        Sliding window size, or -1 if not using sliding window.
    """
    sliding_window = getattr(attn_module, "sliding_window", None)
    if sliding_window is None:
        return -1
    return sliding_window


def torch_attention_hf_wrapper(
    self: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
):
    """Wrapper of auto_deploy::torch_attention with HF attention_interface signature.

    This wrapper:
    1. Converts Q, K, V from HF layout [b, n, s, d] to AD layout [b, s, n, d]
    2. For models needing custom masks, inserts custom_mask_marker op
    3. Calls torch.ops.auto_deploy.torch_attention
    """
    # Convert from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    # Determine attention mask to use
    attn_mask = attention_mask

    # For models needing custom masks (e.g., VLMs with bidirectional image attention),
    # insert a marker that will be replaced during KV cache transformation
    if _needs_custom_mask(self):
        layer_idx = _get_layer_idx(self)
        model_type = _get_model_type(self)
        sliding_window = _get_sliding_window(self)

        # Build metadata dict with layer-specific info
        # The generator at transform time will use this
        metadata = {
            "sliding_window": sliding_window,
        }

        # Insert custom_mask_marker - this becomes an FX node during tracing
        # It will be replaced by actual mask computation during KV cache transform
        # Note: metadata is serialized to JSON because torch.library.custom_op
        # only supports primitive types (str, int, float, bool) and tensors
        attn_mask = torch.ops.auto_deploy.custom_mask_marker(
            model_type,
            layer_idx,
            json.dumps(metadata),
        )

    # Build kwargs for attention op
    # Use module-specific mapping if set by patch, otherwise use default
    kwargs_mapping = getattr(self, "_ad_attn_kwargs_mapping", HF_ATTN_KWARGS_MAPPING)
    ad_attn_kwargs = {kwargs_mapping[k]: v for k, v in kwargs.items() if k in kwargs_mapping}

    attn_output = torch.ops.auto_deploy.torch_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        layout="bsnd",
        **ad_attn_kwargs,
    )

    return attn_output, None


@ExportPatchRegistry.register("unified_attn")
class UnifiedAttnPatch(BaseExportPatch):
    """
    Patch on torch.export.export to replace attention_interface with torch.ops.auto_deploy.torch_attention.

    This patch:
    1. Registers torch_attention_hf_wrapper as "ad_unified_attn" in HF's attention registry
    2. Switches model's _attn_implementation to use our wrapper during export
    3. For VLM models, the wrapper inserts custom_mask_marker ops that capture layer metadata
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
