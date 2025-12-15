"""Export-time patch for Gemma3 to avoid vmap-based mask creation (torch.export incompat)."""

# ruff: noqa: I001

from ...export.interface import BaseExportPatch, ExportPatchRegistry

from transformers import masking_utils
from transformers.models.gemma3 import modeling_gemma3


def _noop_create_causal_mask(**kwargs):
    """Return None to skip vmap-based mask creation during export."""
    return None


def _noop_create_sliding_window_causal_mask(**kwargs):
    """Return None to skip vmap-based mask creation during export."""
    return None


@ExportPatchRegistry.register("hf_gemma3")
class Gemma3ModelPatch(BaseExportPatch):
    """Patch for Gemma3 to make mask functions export-compatible."""

    def _apply_patch(self):
        """Apply the Gemma3Model patch."""
        # Patch mask functions to return None (avoids vmap incompatibility with torch.export)
        # We need to patch both masking_utils AND the module-level references in modeling_gemma3
        self.original_values["mu.create_causal_mask"] = masking_utils.create_causal_mask
        self.original_values["mu.create_sliding_window"] = (
            masking_utils.create_sliding_window_causal_mask
        )
        self.original_values["mg.create_causal_mask"] = modeling_gemma3.create_causal_mask
        self.original_values["mg.create_sliding_window"] = (
            modeling_gemma3.create_sliding_window_causal_mask
        )

        masking_utils.create_causal_mask = _noop_create_causal_mask
        masking_utils.create_sliding_window_causal_mask = _noop_create_sliding_window_causal_mask
        modeling_gemma3.create_causal_mask = _noop_create_causal_mask
        modeling_gemma3.create_sliding_window_causal_mask = _noop_create_sliding_window_causal_mask

    def _revert_patch(self):
        """Revert the Gemma3Model patch."""
        masking_utils.create_causal_mask = self.original_values["mu.create_causal_mask"]
        masking_utils.create_sliding_window_causal_mask = self.original_values[
            "mu.create_sliding_window"
        ]
        modeling_gemma3.create_causal_mask = self.original_values["mg.create_causal_mask"]
        modeling_gemma3.create_sliding_window_causal_mask = self.original_values[
            "mg.create_sliding_window"
        ]
