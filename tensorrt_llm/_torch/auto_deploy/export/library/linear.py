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
"""Patch for F.linear to use simpler implementation during export."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("linear")
class LinearPatch(BaseExportPatch):
    """Patch F.linear to use a simpler implementation for export.

    This patch replaces F.linear with a version that avoids exporting
    view operations used to flatten/unflatten multiple batch dimensions.
    """

    def _apply_patch(self):
        """Apply the linear patch."""
        # Store original function
        self.original_values["F.linear"] = F.linear

        # Create patched function
        def _torch_linear_patch(
            input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            return torch.ops.auto_deploy.torch_linear_simple(input, weight, bias)

        # Apply patch
        F.linear = _torch_linear_patch

    def _revert_patch(self):
        """Revert the linear patch."""
        F.linear = self.original_values["F.linear"]
