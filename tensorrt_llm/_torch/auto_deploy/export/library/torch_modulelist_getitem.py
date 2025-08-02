"""Patch for nn.ModuleList.__getitem__ to handle slicing during export."""

import torch.nn as nn

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("torch_modulelist_getitem")
class TorchModuleListGetitemPatch(BaseExportPatch):
    """Patch nn.ModuleList.__getitem__ to handle slicing during export.

    This patch addresses a PyTorch issue where nn.ModuleList.__getitem__ with slice
    indexing doesn't work correctly during export. The workaround returns a simple
    list for slice operations.

    Reference: https://github.com/pytorch/pytorch/issues/142439
    """

    def _apply_patch(self):
        """Apply the ModuleList getitem patch."""
        # Store original function
        self.original_values["nn.ModuleList.__getitem__"] = nn.ModuleList.__getitem__

        # Capture the original function for use in closure
        original_getitem = nn.ModuleList.__getitem__

        # Create patched function
        def _torch_modulelist_getitem_patch(self: nn.ModuleList, idx):
            if isinstance(idx, slice):
                # return a simple list.
                # NOTE: this obviously only works for any use case where we access the sliced module list
                # like a regular list like a for-loop. For most other things, this hack will not work.
                return list(self._modules.values())[idx]
            else:
                # Call the original function
                return original_getitem(self, idx)

        # Apply patch (type ignore needed as return type differs for slice case)
        nn.ModuleList.__getitem__ = _torch_modulelist_getitem_patch  # type: ignore

    def _revert_patch(self):
        """Revert the ModuleList getitem patch."""
        nn.ModuleList.__getitem__ = self.original_values["nn.ModuleList.__getitem__"]
