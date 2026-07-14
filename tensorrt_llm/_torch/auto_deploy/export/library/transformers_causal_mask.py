# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Patch for transformers causal mask to be export-compatible with meta tensors.

Transformers 5.x's ``masking_utils.create_causal_mask`` calls
``find_packed_sequence_indices`` which invokes ``.all()`` on the attention mask
tensor.  During ``torch.export`` tracing on meta tensors this raises because
``.item()`` / ``.all()`` are not supported on meta tensors.

This patch replaces ``find_packed_sequence_indices`` with a version that returns
early when the input tensors live on the meta device.
"""

import importlib.metadata

from packaging import version

from ..interface import BaseExportPatch, ExportPatchConfig, ExportPatchRegistry


def _transformers_version() -> str:
    """Get the version of transformers."""
    return version.parse(importlib.metadata.version("transformers")).base_version


@ExportPatchRegistry.register("transformers_causal_mask")
class TransformersCausalMaskPatch(BaseExportPatch):
    """Patch ``find_packed_sequence_indices`` to handle meta tensors during export."""

    @classmethod
    def get_config_class(cls):
        return ExportPatchConfig

    def _apply_patch(self):
        """Apply the causal mask patch for meta-tensor compatibility."""
        # Only needed for transformers >= 5.0.0 which introduced find_packed_sequence_indices
        if version.parse(_transformers_version()) < version.parse("4.53.0"):
            return

        try:
            from transformers import masking_utils

            if not hasattr(masking_utils, "find_packed_sequence_indices"):
                return

            original_fn = masking_utils.find_packed_sequence_indices
            self.original_values["find_packed_sequence_indices"] = original_fn

            def _meta_safe_find_packed_sequence_indices(*args, **kwargs):
                """Wrapper that returns None for meta tensors, delegates otherwise."""
                # The first positional arg is the attention_mask tensor
                mask = args[0] if args else kwargs.get("attention_mask")
                if mask is not None and hasattr(mask, "device") and mask.device.type == "meta":
                    return None
                return original_fn(*args, **kwargs)

            masking_utils.find_packed_sequence_indices = _meta_safe_find_packed_sequence_indices

        except (ImportError, AttributeError):
            pass

    def _revert_patch(self):
        """Revert the causal mask patch."""
        if version.parse(_transformers_version()) < version.parse("4.53.0"):
            return

        try:
            from transformers import masking_utils

            if "find_packed_sequence_indices" in self.original_values:
                masking_utils.find_packed_sequence_indices = self.original_values[
                    "find_packed_sequence_indices"
                ]
        except ImportError:
            pass
