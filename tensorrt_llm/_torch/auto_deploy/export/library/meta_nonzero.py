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
"""Patch to enable torch.nonzero() on meta tensors during export.

This patch addresses an issue where torch.nonzero() raises NotImplementedError
when tracing models that use nonzero on meta device. The fix sets the config
flag to assume all elements are non-zero, which enables export to proceed.
"""

import torch.fx.experimental._config as fx_config

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("meta_nonzero")
class MetaNonzeroPatch(BaseExportPatch):
    """Patch to enable torch.nonzero() meta registration during export.

    This patch sets torch.fx.experimental._config.meta_nonzero_assume_all_nonzero
    to True, allowing torch.nonzero() to work on meta tensors during tracing.
    The implementation assumes all elements are non-zero, which is acceptable
    for tracing purposes where only shapes matter.
    """

    def _apply_patch(self):
        """Apply the meta nonzero patch."""
        # Store original config value
        self.original_values["meta_nonzero_assume_all_nonzero"] = getattr(
            fx_config, "meta_nonzero_assume_all_nonzero", False
        )

        # Enable nonzero on meta tensors
        fx_config.meta_nonzero_assume_all_nonzero = True

    def _revert_patch(self):
        """Revert the meta nonzero patch."""
        fx_config.meta_nonzero_assume_all_nonzero = self.original_values[
            "meta_nonzero_assume_all_nonzero"
        ]
