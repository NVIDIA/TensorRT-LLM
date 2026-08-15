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
"""Patch to make torch.nn.attention.sdpa_kernel a no-op during export."""

from contextlib import nullcontext

import torch

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("sdpa_kernel_noop")
class SdpaKernelNoopPatch(BaseExportPatch):
    """Patch torch.nn.attention.sdpa_kernel to be a no-op during export.

    This patch replaces torch.nn.attention.sdpa_kernel with a null context manager
    that can interfere with export.
    """

    def _apply_patch(self):
        """Apply the sdpa_kernel no-op patch."""
        # Store original function
        self.original_values["torch.nn.attention.sdpa_kernel"] = torch.nn.attention.sdpa_kernel

        # Apply patch
        torch.nn.attention.sdpa_kernel = lambda *args, **kwargs: nullcontext()

    def _revert_patch(self):
        """Revert the sdpa_kernel no-op patch."""
        torch.nn.attention.sdpa_kernel = self.original_values["torch.nn.attention.sdpa_kernel"]
