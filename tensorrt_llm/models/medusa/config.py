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

from ..llama.config import LLaMAConfig
from ..qwen.config import QWenConfig


# MedusaConfig is a thin wrapper that picks parent class for GenericMedusaConfig
class QWenMedusaConfig(QWenConfig):

    def __init__(self, num_medusa_heads, num_medusa_layers, max_draft_len,
                 **kwargs):
        self.num_medusa_heads = num_medusa_heads
        self.num_medusa_layers = num_medusa_layers
        self.max_draft_len = max_draft_len
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in MedusaConfig
        output['num_medusa_heads'] = self.num_medusa_heads
        output['num_medusa_layers'] = self.num_medusa_layers
        output['max_draft_len'] = self.max_draft_len
        return output


class LLaMAMedusaConfig(LLaMAConfig):

    def __init__(self, num_medusa_heads, num_medusa_layers, max_draft_len,
                 **kwargs):
        self.num_medusa_heads = num_medusa_heads
        self.num_medusa_layers = num_medusa_layers
        self.max_draft_len = max_draft_len
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in MedusaConfig
        output['num_medusa_heads'] = self.num_medusa_heads
        output['num_medusa_layers'] = self.num_medusa_layers
        output['max_draft_len'] = self.max_draft_len
        return output


# Medusa-specific config is stored and retrieved from GenericMedusaConfig.
class MedusaConfig():

    def __init__(self,
                 *,
                 num_medusa_heads: int = 4,
                 num_medusa_layers: int = 1,
                 max_draft_len: int = 63,
                 **kwargs):
        GenericMedusaConfig = QWenMedusaConfig if "qwen" in kwargs[
            'model_type'] else LLaMAMedusaConfig

        self.config = GenericMedusaConfig(num_medusa_heads, num_medusa_layers,
                                          max_draft_len, **kwargs)

    # Specialization to redirect accesses to self.config
    def __getattribute__(self, name):
        if name == 'config' or '__' in name:
            return object.__getattribute__(self, name)
        else:
            config = object.__getattribute__(self, 'config')
            return config.__getattribute__(name)
