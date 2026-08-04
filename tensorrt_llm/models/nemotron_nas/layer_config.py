# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class AttentionImplementation(str, enum.Enum):
    ATTENTION = "attention"
    LINEAR = "linear"
    NO_OP = "no_op"


class FFNImplementation(str, enum.Enum):
    MLP = "mlp"
    LINEAR = "linear"
    NO_OP = "no_op"


@dataclass(frozen=True, kw_only=True)
class AttentionConfig:
    impl: AttentionImplementation = AttentionImplementation.ATTENTION
    num_key_value_heads: Optional[int] = None

    @property
    def needs_kv_cache(self) -> bool:
        return self.impl == AttentionImplementation.ATTENTION


@dataclass(frozen=True, kw_only=True)
class FFNConfig:
    impl: FFNImplementation = FFNImplementation.MLP
    intermediate_size: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class DeciLayerConfig:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeciLayerConfig":
        assert "attention" in d, "Missing attention configuration"
        assert "ffn" in d, "Missing mlp configuration"

        return cls(
            attention=AttentionConfig(**d["attention"]),
            ffn=FFNConfig(**d["ffn"]),
        )

    @property
    def is_attention_layer(self) -> bool:
        return self.attention.impl == AttentionImplementation.ATTENTION

    @property
    def is_mlp_layer(self) -> bool:
        return self.ffn.impl == FFNImplementation.MLP

    @property
    def is_noop_attention_layer(self) -> bool:
        return self.attention.impl == AttentionImplementation.NO_OP

    @property
    def is_linear_attention_layer(self) -> bool:
        return self.attention.impl == AttentionImplementation.LINEAR

    @property
    def is_noop_ffn_layer(self) -> bool:
        return self.ffn.impl == FFNImplementation.NO_OP

    @property
    def is_linear_ffn_layer(self) -> bool:
        return self.ffn.impl == FFNImplementation.LINEAR
