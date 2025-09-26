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
from dataclasses import asdict, dataclass, field

from ..mapping import Mapping
from ..module import Module
from ..quantization import QuantMode
from .moe import MOE, MoeConfig


@dataclass
class LanguageAdapterConfig:
    num_languages: int | None = None
    top_k: int = 1
    normalization_mode: MoeConfig.ExpertScaleNormalizationMode = MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED
    ffn_hidden_size: int = 1024
    language_list: list = field(default_factory=list)

    def validate(self) -> "LanguageAdapterConfig":
        if (self.num_languages is None or self.num_languages == 0
                or self.top_k == 0):
            raise ValueError(
                "LanguageAdapterConfig's num_languages and top_k must not be set to 0"
            )
        if (self.normalization_mode
                != MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED):
            raise ValueError(
                "LanguageAdapterConfig's normalization_mode must be set to DEVICE_LIMITED since it skips both softmax and renormalization"
            )
        if (len(self.language_list) == 0):
            raise ValueError(
                "LanguageAdapterConfig's language_list must not be empty")
        return self

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_dict(self):
        return asdict(self)

    def to_MOE_config(self):
        return MoeConfig(
            num_experts=self.num_languages,
            top_k=self.top_k,
            normalization_mode=self.normalization_mode,
        )


class LanguageAdapter(Module):
    """
    Language Adapter module that uses MOE plugin with static expert selection passed in as a parameter in request.
    A language MLP is selected by user for each request.
    see https://arxiv.org/pdf/2005.00052 for more details.
    """

    def __init__(
            self,
            language_adapter_config: LanguageAdapterConfig,
            hidden_size: int,
            hidden_act: str,
            mapping: Mapping = Mapping(),
            has_mlp_bias: bool = True,
            dtype=None,
            quant_mode=QuantMode(0),
    ):
        super().__init__()
        self.config = language_adapter_config
        self.config.validate()

        self.layers = MOE(
            hidden_size=hidden_size,
            ffn_hidden_size=language_adapter_config.ffn_hidden_size,
            hidden_act=hidden_act,
            dtype=dtype,
            bias=has_mlp_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            quant_mode=quant_mode,
            static_routing=True,
            moe_config=self.config.to_MOE_config(),
            mapping=mapping)
