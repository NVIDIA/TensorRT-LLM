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
