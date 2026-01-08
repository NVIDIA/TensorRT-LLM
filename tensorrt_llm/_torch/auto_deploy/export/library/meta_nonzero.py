"""Patch to enable torch.nonzero() on meta tensors during export.

This patch addresses an issue where torch.nonzero() raises NotImplementedError
when tracing models that use nonzero on meta device. The fix sets the config
flag to assume all elements are non-zero, which enables export to proceed.
"""

import torch.fx.experimental._config as fx_config

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("meta_nonzero")
class MetaNonzeroPatch(BaseExportPatch):
    """Patch to enable torch.nonzero() meta registration during export.

    This patch sets torch.fx.experimental._config.meta_nonzero_assume_all_nonzero
    to True, allowing torch.nonzero() to work on meta tensors during tracing.
    The implementation assumes all elements are non-zero, which is acceptable
    for tracing purposes where only shapes matter.
    """

    def _apply_patch(self):
        """Apply the meta nonzero patch."""
        # Store original config value
        self.original_values["meta_nonzero_assume_all_nonzero"] = getattr(
            fx_config, "meta_nonzero_assume_all_nonzero", False
        )

        # Enable nonzero on meta tensors
        fx_config.meta_nonzero_assume_all_nonzero = True

    def _revert_patch(self):
        """Revert the meta nonzero patch."""
        fx_config.meta_nonzero_assume_all_nonzero = self.original_values[
            "meta_nonzero_assume_all_nonzero"
        ]
