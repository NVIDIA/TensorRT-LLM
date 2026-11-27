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
"""Recurrent-state metadata of the hybrid models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Metadata, _Registry
from .core import TrtllmCoreMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs


class RecurrentStateMetadata(Metadata):
    """The step's recurrent states of a hybrid model: Mamba, KDA.

    The recurrent layers read ``mamba_metadata``, an object of the class the
    model names; it is prepared from the lengths and the requests of the core.
    """

    requires = (TrtllmCoreMetadata,)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        self.mamba_metadata = config.mamba_metadata_cls(
            config.max_num_requests, config.mamba_chunk_size, max_num_tokens=config.max_num_tokens
        )

    def prepare(self, inputs: "BatchInputs") -> None:
        if inputs.stage_host:
            self.mamba_metadata.prepare(self.get_metadata(TrtllmCoreMetadata))
