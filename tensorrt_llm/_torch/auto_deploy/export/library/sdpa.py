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
