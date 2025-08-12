"""Patch for F.linear to use simpler implementation during export."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("linear")
class LinearPatch(BaseExportPatch):
    """Patch F.linear to use a simpler implementation for export.

    This patch replaces F.linear with a version that avoids exporting
    view operations used to flatten/unflatten multiple batch dimensions.
    """

    def _apply_patch(self):
        """Apply the linear patch."""
        # Store original function
        self.original_values["F.linear"] = F.linear

        # Create patched function
        def _torch_linear_patch(
            input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            return torch.ops.auto_deploy.torch_linear_simple(input, weight, bias)

        # Apply patch
        F.linear = _torch_linear_patch

    def _revert_patch(self):
        """Revert the linear patch."""
        F.linear = self.original_values["F.linear"]
