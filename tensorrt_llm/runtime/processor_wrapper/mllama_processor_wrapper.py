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

from .processor_wrapper import ProcessorWrapper


class MllamaProcessorWrapper(ProcessorWrapper):
    '''
    A slight wrapper for Huggingface MllamaProcessor to insert the prompt automatically for different kinds of models
    '''

    def __call__(self, **kwargs):
        return self.processor(**kwargs)

    def apply_chat_template(self, **kwargs):
        images = kwargs.get('images', None)
        text = kwargs.get('text')
        if '<|begin_of_text|>' in text:
            self.logger.warning(
                'find special token (<|begin_of_text|>) in the text, skipping to add the special tokens.'
            )
        else:
            if images is None:
                if self.processor.chat_template is not None:
                    text = self.tokenizer.apply_chat_template(
                        [{
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": text
                            }],
                        }],
                        add_generation_prompt=True,
                    )
                else:
                    text = text
            else:
                if self.processor.chat_template is not None:
                    self.logger.info(
                        'find chat_template. apply_chat_template to the text')
                    text = self.processor.apply_chat_template(
                        [{
                            "role":
                            "user",
                            "content": [{
                                "type": "image"
                            }, {
                                "type": "text",
                                "text": text
                            }],
                        }],
                        add_generation_prompt=True,
                    )
                else:
                    self.logger.info(
                        'cannot find chat_template. add <|image|><|begin_of_text|> to the text'
                    )
                    text = f"<|image|><|begin_of_text|>{text}"

        return text

    def decode(self, tensor, skip_special_tokens=True):
        return self.processor.decode(tensor,
                                     skip_special_tokens=skip_special_tokens)
