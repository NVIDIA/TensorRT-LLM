"""Patch for torch.tensor to handle 0.0 on meta device."""

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("tensor_meta_device")
class TensorMetaDevicePatch(BaseExportPatch):
    """Patch torch.tensor to handle 0.0 or empty data on meta device.

    This patch addresses an issue where torch.tensor(0.0, device="meta")
    and torch.tensor([], device="meta") doesn't work and needs to be replaced with
    torch.zeros operator.
    """

    def _apply_patch(self):
        """Apply the tensor meta device patch."""
        # Store original function
        self.original_values["torch.tensor"] = torch.tensor

        # Create patched function
        def _torch_tensor_patch(data, **kwargs):
            device = kwargs.get("device", None)
            if device is not None and torch.device(device) == torch.device("meta"):
                if data == 0.0:
                    return torch.zeros((), **kwargs)
                if data == [] or data == ():
                    return torch.zeros((0,), **kwargs)
            return self.original_values["torch.tensor"](data, **kwargs)

        # Apply patch
        torch.tensor = _torch_tensor_patch

    def _revert_patch(self):
        """Revert the tensor meta device patch."""
        torch.tensor = self.original_values["torch.tensor"]
