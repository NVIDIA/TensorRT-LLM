"""Patch for modelopt's torch_export_context."""

from contextlib import nullcontext

from ..interface import ContextManagerPatch, ExportPatchRegistry


@ExportPatchRegistry.register("modelopt_context")
class ModeloptContextPatch(ContextManagerPatch):
    """Patch to apply modelopt's torch_export_context during export.

    This patch applies the modelopt quantization context manager around
    the export process when available, otherwise uses a null context.
    """

    def init_context_manager(self):
        """Initialize and return the modelopt context manager or nullcontext if not available."""
        try:
            from modelopt.torch.quantization.utils import export_torch_mode as torch_export_context

            return torch_export_context()
        except ImportError:
            return nullcontext()
