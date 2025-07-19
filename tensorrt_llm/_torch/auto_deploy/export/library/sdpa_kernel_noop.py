"""Patch to make torch.nn.attention.sdpa_kernel a no-op during export."""

from contextlib import nullcontext

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("sdpa_kernel_noop")
class SdpaKernelNoopPatch(BaseExportPatch):
    """Patch torch.nn.attention.sdpa_kernel to be a no-op during export.

    This patch replaces torch.nn.attention.sdpa_kernel with a null context manager
    that can interfere with export.
    """

    def _apply_patch(self):
        """Apply the sdpa_kernel no-op patch."""
        # Store original function
        self.original_values["torch.nn.attention.sdpa_kernel"] = torch.nn.attention.sdpa_kernel

        # Apply patch
        torch.nn.attention.sdpa_kernel = lambda *args, **kwargs: nullcontext()

    def _revert_patch(self):
        """Revert the sdpa_kernel no-op patch."""
        torch.nn.attention.sdpa_kernel = self.original_values["torch.nn.attention.sdpa_kernel"]
