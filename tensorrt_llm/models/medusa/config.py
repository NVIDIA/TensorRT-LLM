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


# Medusa-specific config is stored and retrieved from GenericMedusaConfig.
class MedusaConfig():

    def __init__(self,
                 *,
                 num_medusa_heads: int = 4,
                 num_medusa_layers: int = 1,
                 max_draft_len: int = 63,
                 **kwargs):
        GenericMedusaConfig = QWenConfig if "qwen" in kwargs[
            'model_type'] else LLaMAConfig

        self.config = GenericMedusaConfig(**kwargs)

        # Add objects
        self.config.num_medusa_heads = num_medusa_heads
        self.config.num_medusa_layers = num_medusa_layers
        self.config.max_draft_len = max_draft_len

    def to_dict(self):
        output = self.config.to_dict()
        output['num_medusa_heads'] = self.config.num_medusa_heads
        output['num_medusa_layers'] = self.config.num_medusa_layers
        output['max_draft_len'] = self.config.max_draft_len
        return output

    # Specialization to redirect accesses to self.config
    def __getattr__(self, name):
        return getattr(self.config, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
