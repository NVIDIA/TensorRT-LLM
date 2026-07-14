# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch for transformers SDPA mask to be export-compatible."""

from functools import partial

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("transformers_sdpa_mask")
class TransformersSdpaMaskPatch(BaseExportPatch):
    """Patch transformers.masking_utils.sdpa_mask to be export-compatible.

    Transformers 5.x removed ``sdpa_mask_without_vmap`` from
    ``integrations.executorch``; ``sdpa_mask(use_vmap=False)`` is the export-compatible
    replacement.
    """

    def _apply_patch(self):
        """Apply the transformers SDPA mask patch."""
        try:
            from transformers import masking_utils
        except ImportError:
            return

        sdpa_mask_without_vmap = partial(masking_utils.sdpa_mask, use_vmap=False)

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

    def _revert_patch(self):
        """Revert the transformers SDPA mask patch."""
        try:
            from transformers import masking_utils
        except ImportError:
            return

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
