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
"""Sparse attention fields that every TRT-LLM attention op reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .base import Metadata, _Registry
from .core import TrtllmCoreMetadata

if TYPE_CHECKING:
    from ..backends.sparse.params import SparseMetadataParams
    from .batch_inputs import BatchInputs


class SparseMetadata(Metadata):
    """What the attention ops read about sparse attention, whatever the model.

    Attached as is, it describes a dense model: no layer selects KV rows. A
    sparse algorithm derives from it and fills the fields.
    """

    requires = (TrtllmCoreMetadata,)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        self.sparse_metadata_params: Optional["SparseMetadataParams"] = (
            config.sparse_metadata_params
        )
        # KV rows each query selects; 0 without sparse attention.
        self.num_sparse_topk = 0
