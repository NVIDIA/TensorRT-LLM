"""Patch to make torch.autocast a no-op during export."""

from contextlib import nullcontext

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("autocast_noop")
class AutocastNoopPatch(BaseExportPatch):
    """Patch torch.autocast to be a no-op during export.

    This patch replaces torch.autocast with a null context manager
    that can interfere with export.

    It also patches ``torch.is_autocast_enabled`` so that transformers 5.x
    helpers like ``maybe_autocast`` (used in RoPE embeddings) do not call the
    real function with an unknown/fake device type during ``torch.export``
    tracing, which would raise a ``RuntimeError``.
    """

    def _apply_patch(self):
        """Apply the autocast no-op patch."""
        # Store original functions
        self.original_values["torch.autocast"] = torch.autocast
        self.original_values["torch.is_autocast_enabled"] = torch.is_autocast_enabled

        # Apply patches
        torch.autocast = lambda *args, **kwargs: nullcontext()

        # torch.is_autocast_enabled(device_type) can fail during export when the
        # device_type is unknown (e.g. fake/meta tensors).  Return False so that
        # callers like transformers' ``maybe_autocast`` skip the autocast block.
        original_is_autocast = self.original_values["torch.is_autocast_enabled"]

        def _safe_is_autocast_enabled(*args, **kwargs):
            try:
                return original_is_autocast(*args, **kwargs)
            except RuntimeError:
                return False

        torch.is_autocast_enabled = _safe_is_autocast_enabled

    def _revert_patch(self):
        """Revert the autocast no-op patch."""
        torch.autocast = self.original_values["torch.autocast"]
        torch.is_autocast_enabled = self.original_values["torch.is_autocast_enabled"]
