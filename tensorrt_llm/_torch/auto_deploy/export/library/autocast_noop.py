"""Patch to make torch.autocast a no-op during export."""

from contextlib import nullcontext

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("autocast_noop")
class AutocastNoopPatch(BaseExportPatch):
    """Patch torch.autocast to be a no-op during export.

    This patch replaces torch.autocast with a null context manager
    that can interfere with export.
    """

    def _apply_patch(self):
        """Apply the autocast no-op patch."""
        # Store original function
        self.original_values["torch.autocast"] = torch.autocast

        # Apply patch
        torch.autocast = lambda *args, **kwargs: nullcontext()

    def _revert_patch(self):
        """Revert the autocast no-op patch."""
        torch.autocast = self.original_values["torch.autocast"]
