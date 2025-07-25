"""Patch for transformers SDPA mask to be export-compatible."""

import importlib.metadata

from packaging import version

from ..interface import BaseExportPatch, ExportPatchRegistry


def _transformers_version() -> str:
    """Get the version of transformers."""
    return version.parse(importlib.metadata.version("transformers")).base_version


@ExportPatchRegistry.register("transformers_sdpa_mask")
class TransformersSdpaMaskPatch(BaseExportPatch):
    """Patch transformers.masking_utils.sdpa_mask to be export-compatible.

    This patch replaces the transformers SDPA mask implementation with an
    export-compatible version for transformers >= 4.53.0.
    """

    def _apply_patch(self):
        """Apply the transformers SDPA mask patch."""
        # this patch is only needed+compatible for transformers >= 4.53.0
        if version.parse(_transformers_version()) < version.parse("4.53.0"):
            return  # Skip patch for older versions

        try:
            # imports only after version check
            from transformers import masking_utils
            from transformers.integrations.executorch import sdpa_mask_without_vmap

            # recall original implementation
            self.original_values["masking_utils.sdpa_mask"] = masking_utils.sdpa_mask

            # patch function and mask attention interface
            masking_utils.sdpa_mask = sdpa_mask_without_vmap

            if "sdpa" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._local_mapping:
                self.original_values["sdpa_local_original"] = (
                    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._local_mapping["sdpa"]
                )
            else:
                self.original_values["sdpa_local_original"] = None

            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask_without_vmap

        except ImportError:
            # If transformers is not available or doesn't have required modules, skip patch
            pass

    def _revert_patch(self):
        """Revert the transformers SDPA mask patch."""
        # this patch is only needed+compatible for transformers >= 4.53.0
        if version.parse(_transformers_version()) < version.parse("4.53.0"):
            return  # Skip revert for older versions

        try:
            # imports only after version check
            from transformers import masking_utils

            # revert patches
            if "masking_utils.sdpa_mask" in self.original_values:
                masking_utils.sdpa_mask = self.original_values["masking_utils.sdpa_mask"]

            if "sdpa_local_original" in self.original_values:
                if self.original_values["sdpa_local_original"] is None:
                    if "sdpa" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._local_mapping:
                        del masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
                else:
                    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = self.original_values[
                        "sdpa_local_original"
                    ]

        except ImportError:
            # If transformers is not available, skip revert
            pass
