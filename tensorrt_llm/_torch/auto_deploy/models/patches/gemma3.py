"""Export-time patch for Gemma3 to avoid vmap-based mask creation (torch.export incompat).

This patch also ensures token_type_ids can be accepted by the TextModel for VLM mask
generation inside the exported GraphModule. The export process (export_to_gm.py) injects
token_type_ids into the captured kwargs for VLM models.
"""

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


def _create_patched_text_model_forward(original_forward):
    """Create a patched TextModel forward that accepts token_type_ids.

    This allows token_type_ids to be captured as a graph input during export.
    The TextModel doesn't use token_type_ids directly - mask generation happens
    in the kvcache transform via the flashinfer_gemma3_mask_gen op.
    """
    import functools

    @functools.wraps(original_forward)
    def patched_forward(self, *args, token_type_ids=None, **kwargs):
        # Accept token_type_ids (will be captured as graph input during export)
        # but don't use it - mask generation happens in the kvcache transform
        return original_forward(self, *args, **kwargs)

    return patched_forward


@ExportPatchRegistry.register("hf_gemma3")
class Gemma3ModelPatch(BaseExportPatch):
    """Patch for Gemma3 to make mask functions export-compatible and accept token_type_ids."""

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

        # Patch Gemma3TextModel.forward to accept token_type_ids as an argument
        # This is necessary for torch.export to capture it as a graph input.
        # The export process (export_to_gm.py) injects token_type_ids into captured kwargs
        # for VLM models.
        if hasattr(modeling_gemma3, "Gemma3TextModel"):
            self.original_values["Gemma3TextModel.forward"] = (
                modeling_gemma3.Gemma3TextModel.forward
            )
            modeling_gemma3.Gemma3TextModel.forward = _create_patched_text_model_forward(
                modeling_gemma3.Gemma3TextModel.forward
            )

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

        # Revert forward patches
        if "Gemma3TextModel.forward" in self.original_values:
            modeling_gemma3.Gemma3TextModel.forward = self.original_values[
                "Gemma3TextModel.forward"
            ]
