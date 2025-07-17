"""Patch for F.scaled_dot_product_attention to use custom op."""

import torch
import torch.nn.functional as F

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("sdpa")
class SdpaPatch(BaseExportPatch):
    """Patch F.scaled_dot_product_attention to use custom op during export.

    This patch ensures that scaled_dot_product_attention is represented consistently
    in the exported graph by using a custom operation.
    """

    def _apply_patch(self):
        """Apply the SDPA patch."""
        # Store original function
        self.original_values["F.scaled_dot_product_attention"] = F.scaled_dot_product_attention

        # Apply patch
        F.scaled_dot_product_attention = torch.ops.auto_deploy.torch_attention_sdpa

    def _revert_patch(self):
        """Revert the SDPA patch."""
        F.scaled_dot_product_attention = self.original_values["F.scaled_dot_product_attention"]
