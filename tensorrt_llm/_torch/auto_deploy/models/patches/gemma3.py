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
    in the kvcache transform via the flashinfer_vlm_mask_gen op.
    """

    @functools.wraps(original_forward)
    def patched_forward(self, *args, token_type_ids=None, **kwargs):
        # Accept token_type_ids (will be captured as graph input during export)
        # but don't use it - mask generation happens in the kvcache transform
        return original_forward(self, *args, **kwargs)

    return patched_forward


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


def _create_patched_gemma3_model_forward(original_forward):
    """Patch Gemma3Model.forward to pass token_type_ids to self.language_model().

    The HF Gemma3Model.forward receives token_type_ids (via **lm_kwargs) but doesn't
    pass it to self.language_model(). This patch ensures token_type_ids is included
    in the call to language_model so it can be captured during export.
    """

    @functools.wraps(original_forward)
    def patched_forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        **lm_kwargs,
    ):
        # Add token_type_ids to lm_kwargs so it gets passed to self.language_model()
        # The language_model (Gemma3TextModel) is patched to accept token_type_ids
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32, device=input_ids.device)
        lm_kwargs["token_type_ids"] = token_type_ids
        return original_forward(
            self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            **lm_kwargs,
        )

    return patched_forward


@ExportPatchRegistry.register("hf_gemma3")
class Gemma3ModelPatch(BaseExportPatch):
    """Patch for Gemma3 to make mask functions export-compatible and pass token_type_ids."""

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
        if hasattr(modeling_gemma3, "Gemma3TextModel"):
            self.original_values["Gemma3TextModel.forward"] = (
                modeling_gemma3.Gemma3TextModel.forward
            )
            modeling_gemma3.Gemma3TextModel.forward = _create_patched_text_model_forward(
                modeling_gemma3.Gemma3TextModel.forward
            )

            # Patch __call__ to inject token_type_ids BEFORE pre-hooks run
            # This ensures the capture hook sees token_type_ids in kwargs
            self.original_values["Gemma3TextModel.__call__"] = (
                modeling_gemma3.Gemma3TextModel.__call__
            )
            modeling_gemma3.Gemma3TextModel.__call__ = _create_patched_text_model_call(
                modeling_gemma3.Gemma3TextModel.__call__
            )

        # Patch Gemma3Model.forward to pass token_type_ids to self.language_model()
        # The HF implementation receives token_type_ids but doesn't forward it to the inner
        # Gemma3TextModel. This patch ensures it flows through via **lm_kwargs.
        if hasattr(modeling_gemma3, "Gemma3Model"):
            self.original_values["Gemma3Model.forward"] = modeling_gemma3.Gemma3Model.forward
            modeling_gemma3.Gemma3Model.forward = _create_patched_gemma3_model_forward(
                modeling_gemma3.Gemma3Model.forward
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
        if "Gemma3TextModel.__call__" in self.original_values:
            modeling_gemma3.Gemma3TextModel.__call__ = self.original_values[
                "Gemma3TextModel.__call__"
            ]
        if "Gemma3Model.forward" in self.original_values:
            modeling_gemma3.Gemma3Model.forward = self.original_values["Gemma3Model.forward"]
