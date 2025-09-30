"""Patch for torch.where to handle case where only condition is provided."""

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("torch_where")
class TorchWherePatch(BaseExportPatch):
    """Patch torch.where to handle the case where only condition is provided.

    This patch addresses the issue where torch.where(condition) should return
    torch.nonzero(condition, as_tuple=True) but the export process doesn't
    handle this correctly.
    """

    def _apply_patch(self):
        """Apply the torch.where patch."""
        # Store original function
        self.original_values["torch.where"] = torch.where

        # Create patched function
        def _torch_where_patch(condition: torch.Tensor, *args, **kwargs):
            if len(args) == 0 and len(kwargs) == 0:
                return torch.nonzero(condition, as_tuple=True)
            return self.original_values["torch.where"](condition, *args, **kwargs)

        # Apply patch
        torch.where = _torch_where_patch

    def _revert_patch(self):
        """Revert the torch.where patch."""
        torch.where = self.original_values["torch.where"]
