"""Patch for transformers is_tracing to detect meta tensors during export.

This patch addresses an issue where transformers v5's `is_tracing()` function
doesn't detect meta tensors, causing `.all()` and `.item()` calls to fail
during AutoDeploy's meta-tensor based export.

The fix wraps `is_tracing` to also return True when a meta tensor is passed,
which prevents transformers from trying to materialize tensor values during export.
"""

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


def _is_meta_tensor(x) -> bool:
    """Check if tensor is on meta device."""
    try:
        return isinstance(x, torch.Tensor) and x.device.type == "meta"
    except Exception:
        return False


@ExportPatchRegistry.register("transformers_is_tracing")
class TransformersIsTracingPatch(BaseExportPatch):
    """Patch transformers.utils.import_utils.is_tracing to detect meta tensors.

    This patch wraps the transformers `is_tracing` function to also return True
    when a meta tensor is passed. This is needed because AutoDeploy uses meta
    tensors during export, and transformers v5's masking utilities call
    `.all()` or `.item()` on tensors when `is_tracing()` returns False,
    which fails on meta tensors.
    """

    def _apply_patch(self):
        """Apply the is_tracing patch."""
        try:
            from transformers.utils import import_utils

            # Store original is_tracing function
            self.original_values["is_tracing"] = import_utils.is_tracing

            # Create wrapped version that also checks for meta tensors
            original_is_tracing = import_utils.is_tracing

            def patched_is_tracing(tensor=None) -> bool:
                # First check meta tensor (fast path for our use case)
                if tensor is not None and _is_meta_tensor(tensor):
                    return True
                # Fall back to original is_tracing logic
                return original_is_tracing(tensor)

            # Apply patch
            import_utils.is_tracing = patched_is_tracing

            # Also patch the masking_utils module if it imports is_tracing directly
            try:
                from transformers import masking_utils

                if hasattr(masking_utils, "is_tracing"):
                    self.original_values["masking_utils.is_tracing"] = masking_utils.is_tracing
                    masking_utils.is_tracing = patched_is_tracing
            except ImportError:
                # masking_utils may not exist in older transformers versions
                pass

        except ImportError:
            # If transformers is not available, skip patch
            pass

    def _revert_patch(self):
        """Revert the is_tracing patch."""
        try:
            from transformers.utils import import_utils

            if "is_tracing" in self.original_values:
                import_utils.is_tracing = self.original_values["is_tracing"]

            if "masking_utils.is_tracing" in self.original_values:
                from transformers import masking_utils

                masking_utils.is_tracing = self.original_values["masking_utils.is_tracing"]

        except ImportError:
            # If transformers is not available, skip revert
            pass
