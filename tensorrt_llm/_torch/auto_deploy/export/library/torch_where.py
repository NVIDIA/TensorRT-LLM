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
"""Patch for torch.where to handle case where only condition is provided."""

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("torch_where")
class TorchWherePatch(BaseExportPatch):
    """Patch torch.where to handle the case where only condition is provided.

    This patch addresses the issue where torch.where(condition) should return
    torch.nonzero(condition, as_tuple=True) but the export process doesn't
    handle this correctly.
    """

    def _apply_patch(self):
        """Apply the torch.where patch."""
        # Store original function
        self.original_values["torch.where"] = torch.where

        # Create patched function
        def _torch_where_patch(condition: torch.Tensor, *args, **kwargs):
            if len(args) == 0 and len(kwargs) == 0:
                return torch.nonzero(condition, as_tuple=True)
            return self.original_values["torch.where"](condition, *args, **kwargs)

        # Apply patch
        torch.where = _torch_where_patch

    def _revert_patch(self):
        """Revert the torch.where patch."""
        torch.where = self.original_values["torch.where"]
