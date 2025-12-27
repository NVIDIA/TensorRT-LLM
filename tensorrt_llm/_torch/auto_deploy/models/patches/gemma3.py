# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Export-time patch for Gemma3 models.

This patch sets model-specific attention kwargs mapping on Gemma3 attention classes.
Gemma3 has nested config (text_config) and uses specific kwargs like logit_cap.
"""

from typing import Sequence

import torch

from ...export.interface import AdditionalGraphInput, BaseExportPatch, ExportPatchRegistry


class TokenTypeIdsInput(AdditionalGraphInput):
    """token_type_ids input for bidirectional attention mask in Gemma3.

    In Gemma3 VLM, token_type_ids indicates whether each token is a text token (0)
    or an image token (1). This is used by the mask generator to create the correct
    bidirectional attention pattern for image tokens.

    For text-only requests, the placeholder is all zeros (all text tokens).
    """

    @property
    def name(self) -> str:
        return "token_type_ids"

    @property
    def needs_bypass(self) -> bool:
        """Gemma3Model consumes token_type_ids but doesn't pass it to language_model."""
        return True

    def create_placeholder(self, input_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        """Create all-zeros placeholder indicating all tokens are text tokens."""
        total_tokens = sum(len(seq) for seq in input_ids)
        return torch.zeros(total_tokens, dtype=torch.long)


# Gemma3-specific attention kwargs mapping
# Extends/overrides the default mapping for Gemma3's specific attention parameters
GEMMA3_ATTN_KWARGS_MAPPING = {
    "dropout": "dropout_p",
    "is_causal": "is_causal",
    "scaling": "scale",
    "scale": "scale",
    "s_aux": "sinks",
    "sinks": "sinks",
    "sliding_window": "sliding_window",
    "logit_cap": "logit_cap",  # Gemma3 uses soft-capping
}


@ExportPatchRegistry.register("hf_gemma3")
class Gemma3ModelPatch(BaseExportPatch):
    """Patch for Gemma3 models during export.

    Sets attributes on Gemma3Attention class:
    - `_ad_attn_kwargs_mapping`: Model-specific HF-to-AD kwargs mapping
    - `_ad_needs_custom_mask`: Enables custom mask marker insertion for VLM support
    """

    # Additional inputs added to the graph beyond the nn.Module forward signature.
    # TokenTypeIdsInput needs bypass (_ad_ prefix) because Gemma3Model consumes it
    # but doesn't pass it to the inner language_model where mask generation runs.
    ADDITIONAL_GRAPH_INPUTS = [TokenTypeIdsInput()]

    def _apply_patch(self):
        """Apply the Gemma3Model patch."""
        # Import Gemma3 attention class and set the mapping attribute
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

            # Store originals to revert later
            self.original_values["Gemma3Attention._ad_attn_kwargs_mapping"] = getattr(
                Gemma3Attention, "_ad_attn_kwargs_mapping", None
            )
            self.original_values["Gemma3Attention._ad_needs_custom_mask"] = getattr(
                Gemma3Attention, "_ad_needs_custom_mask", None
            )
            # Set model-specific attributes
            Gemma3Attention._ad_attn_kwargs_mapping = GEMMA3_ATTN_KWARGS_MAPPING
            Gemma3Attention._ad_needs_custom_mask = True
        except ImportError:
            # Gemma3 not available in this transformers version
            pass

    def _revert_patch(self):
        """Revert the Gemma3Model patch."""
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

            # Revert _ad_attn_kwargs_mapping
            original_mapping = self.original_values.get("Gemma3Attention._ad_attn_kwargs_mapping")
            if original_mapping is None:
                if hasattr(Gemma3Attention, "_ad_attn_kwargs_mapping"):
                    delattr(Gemma3Attention, "_ad_attn_kwargs_mapping")
            else:
                Gemma3Attention._ad_attn_kwargs_mapping = original_mapping

            # Revert _ad_needs_custom_mask
            original_mask = self.original_values.get("Gemma3Attention._ad_needs_custom_mask")
            if original_mask is None:
                if hasattr(Gemma3Attention, "_ad_needs_custom_mask"):
                    delattr(Gemma3Attention, "_ad_needs_custom_mask")
            else:
                Gemma3Attention._ad_needs_custom_mask = original_mask
        except ImportError:
            pass
