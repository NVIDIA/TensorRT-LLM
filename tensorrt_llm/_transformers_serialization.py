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
"""Runtime guards for Hugging Face chat-template serialization."""

from functools import wraps
from pathlib import Path


def _validate_chat_template_paths(chat_template, chat_template_dir) -> None:
    """Keep named template files inside their resolved template directory."""
    if not isinstance(chat_template, dict):
        return
    directory = Path(chat_template_dir)
    for name in chat_template:
        if name == "default":
            continue
        template_filepath = directory / f"{name}.jinja"
        if template_filepath.resolve().parent != directory.resolve():
            raise ValueError(f"Invalid chat template name: {name!r}")


def _install_chat_template_guards() -> None:
    """Guard both HF serializers in memory without changing installed files."""
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.utils import CHAT_TEMPLATE_DIR

    tokenizer_save = PreTrainedTokenizerBase.save_chat_templates
    if not getattr(tokenizer_save, "_trtllm_chat_template_guard", False):

        @wraps(tokenizer_save)
        def save_chat_templates(
            self, save_directory, tokenizer_config, filename_prefix, save_jinja_files
        ):
            if save_jinja_files:
                directory = Path(save_directory) / (
                    (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_DIR
                )
                _validate_chat_template_paths(self.chat_template, directory)
            return tokenizer_save(
                self, save_directory, tokenizer_config, filename_prefix, save_jinja_files
            )

        save_chat_templates._trtllm_chat_template_guard = True
        PreTrainedTokenizerBase.save_chat_templates = save_chat_templates

    processor_save = ProcessorMixin.save_pretrained
    if not getattr(processor_save, "_trtllm_chat_template_guard", False):

        @wraps(processor_save)
        def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
            _validate_chat_template_paths(
                self.chat_template, Path(save_directory) / CHAT_TEMPLATE_DIR
            )
            return processor_save(self, save_directory, push_to_hub=push_to_hub, **kwargs)

        save_pretrained._trtllm_chat_template_guard = True
        ProcessorMixin.save_pretrained = save_pretrained
