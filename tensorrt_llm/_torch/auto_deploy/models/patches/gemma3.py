"""Export-time patch for Gemma3 to avoid vmap-based mask creation (torch.export incompat).

This patch ensures:
1. Mask creation functions return None (avoids vmap incompatibility with torch.export)
2. token_type_ids is always present in Gemma3TextModel kwargs during export, so it can be
   captured as a graph input for VLM custom mask generation.

The key insight is that we patch Gemma3TextModel.__call__ to inject token_type_ids into
kwargs BEFORE any pre-hooks run. This ensures the capture hook sees token_type_ids.
"""

# ruff: noqa: I001

import functools
import torch
from ...custom_ops.vlm_mask_registry import VlmMetadataKeys
from ...export.interface import BaseExportPatch, ExportPatchRegistry

from transformers.models.gemma3 import modeling_gemma3


def _create_patched_text_model_call(original_call):
    """Patch __call__ to inject token_type_ids into kwargs BEFORE pre-hooks run.

    This ensures the capture hook sees token_type_ids when capturing kwargs.
    """

    @functools.wraps(original_call)
    def patched_call(self, *args, **kwargs):
        # Inject token_type_ids before pre-hooks run
        if "token_type_ids" not in kwargs or kwargs.get("token_type_ids") is None:
            position_ids = kwargs.get("position_ids")
            if position_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(position_ids)
        return original_call(self, *args, **kwargs)

    return patched_call


@ExportPatchRegistry.register("hf_gemma3")
class Gemma3ModelPatch(BaseExportPatch):
    """Patch for Gemma3 to make mask functions export-compatible and pass token_type_ids.

    This patch sets `_vlm_input_names` on the model class to specify which kwargs
    should be captured as VLM inputs for mask generation.
    """

    # VLM inputs that this patch injects and expects to be captured
    VLM_INPUT_NAMES = ["token_type_ids"]

    def _apply_patch(self):
        """Apply the Gemma3Model patch."""

        # Patch Gemma3TextModel.forward to accept token_type_ids as an argument
        # This is necessary for torch.export to capture it as a graph input.
        if hasattr(modeling_gemma3, "Gemma3TextModel"):
            # Patch __call__ to inject token_type_ids BEFORE pre-hooks run
            # This ensures the capture hook sees token_type_ids in kwargs
            self.original_values["Gemma3TextModel.__call__"] = (
                modeling_gemma3.Gemma3TextModel.__call__
            )
            modeling_gemma3.Gemma3TextModel.__call__ = _create_patched_text_model_call(
                modeling_gemma3.Gemma3TextModel.__call__
            )

            # Set VLM input names on the class so _set_vlm_metadata can discover them
            attr_name = VlmMetadataKeys.MODULE_INPUT_NAMES
            self.original_values[f"Gemma3TextModel.{attr_name}"] = getattr(
                modeling_gemma3.Gemma3TextModel, attr_name, None
            )
            setattr(modeling_gemma3.Gemma3TextModel, attr_name, self.VLM_INPUT_NAMES)

    def _revert_patch(self):
        """Revert the Gemma3Model patch."""
        # Revert __call__ patch
        if "Gemma3TextModel.__call__" in self.original_values:
            modeling_gemma3.Gemma3TextModel.__call__ = self.original_values[
                "Gemma3TextModel.__call__"
            ]

        # Revert VLM input names
        attr_name = VlmMetadataKeys.MODULE_INPUT_NAMES
        key = f"Gemma3TextModel.{attr_name}"
        if key in self.original_values:
            original = self.original_values[key]
            if original is None:
                if hasattr(modeling_gemma3.Gemma3TextModel, attr_name):
                    delattr(modeling_gemma3.Gemma3TextModel, attr_name)
            else:
                setattr(modeling_gemma3.Gemma3TextModel, attr_name, original)
