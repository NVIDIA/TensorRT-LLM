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

from dataclasses import asdict, dataclass


# [numLayers, kv_factor, heads, tokens, dims_per_head]
@dataclass
class AttentionInfo:
    kv_heads_per_rank: int
    tokens_per_block: int
    dims_per_head: int
    element_bytes: int | float
    enable_attention_dp: bool
    is_mla: bool

    @property
    def kv_factor(self) -> int:
        return 2 if not self.is_mla else 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionInfo":
        return cls(**data)
