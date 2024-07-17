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

from typing import Optional, Union

from ...layers import MoeConfig
from ..modeling_utils import PretrainedConfig


class DbrxConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 bias: bool = False,
                 clip_qkv: Optional[float] = None,
                 rotary_base: float = 500000.0,
                 rotary_scaling: Optional[dict] = None,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 **kwargs):
        self.bias = bias
        self.clip_qkv = clip_qkv
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        if moe is None:
            # Legacy MOE config fields
            moe = MoeConfig(
                num_experts=kwargs.pop('moe_num_experts', 0),
                top_k=kwargs.pop('moe_top_k', 0),
                normalization_mode=kwargs.pop(
                    'moe_normalization_mode',
                    MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE))
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in DbrxConfig
        output['bias'] = self.bias
        output['clip_qkv'] = self.clip_qkv
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['moe'] = self.moe.to_dict()
        return output
