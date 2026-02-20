"""Patch to make torch.autocast a no-op during export.

This patch also handles torch.is_autocast_enabled for meta tensors,
which is needed because transformers v5 calls is_autocast_enabled(device_type)
before entering autocast contexts, and PyTorch doesn't recognize "meta"
as a valid autocast device type.
"""

from contextlib import nullcontext

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("autocast_noop")
class AutocastNoopPatch(BaseExportPatch):
    """Patch torch.autocast to be a no-op during export.

    This patch replaces torch.autocast with a null context manager
    that can interfere with export. It also wraps torch.is_autocast_enabled
    to handle meta device gracefully (returns False for meta device).
    """

    def _apply_patch(self):
        """Apply the autocast no-op patch."""
        # Store original functions
        self.original_values["torch.autocast"] = torch.autocast
        self.original_values["torch.is_autocast_enabled"] = torch.is_autocast_enabled

        # Apply autocast patch
        torch.autocast = lambda *args, **kwargs: nullcontext()

        # Apply is_autocast_enabled patch to handle meta device
        original_is_autocast_enabled = self.original_values["torch.is_autocast_enabled"]

        def patched_is_autocast_enabled(device_type=None):
            # Meta device doesn't support autocast, return False to avoid errors
            if device_type == "meta":
                return False
            return original_is_autocast_enabled(device_type)

        torch.is_autocast_enabled = patched_is_autocast_enabled

    def _revert_patch(self):
        """Revert the autocast no-op patch."""
        torch.autocast = self.original_values["torch.autocast"]
        torch.is_autocast_enabled = self.original_values["torch.is_autocast_enabled"]
