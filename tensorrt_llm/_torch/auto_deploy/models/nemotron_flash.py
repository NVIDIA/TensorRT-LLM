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
from .factory import ModelFactoryRegistry
from .hf import AutoModelForCausalLMFactory


@ModelFactoryRegistry.register("NemotronFlashForCausalLM")
class NemotronFlashForCausalLMFactory(AutoModelForCausalLMFactory):
    # TODO: custom tokenizer initialization system
    def init_tokenizer(self):
        if self.tokenizer is None:
            return None

        from .custom import NemotronFlashPreTrainedTokenizerFast

        model_config, _ = self._get_model_config()
        return NemotronFlashPreTrainedTokenizerFast.from_pretrained(
            self.tokenizer,
            **self.tokenizer_kwargs,
            num_memory_tokens=model_config.num_memory_tokens,
            vocab_size_model=model_config.vocab_size,
        )
