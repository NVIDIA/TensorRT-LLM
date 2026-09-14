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
"""Tokenizer wrapper for the MuseSparkOpen checkpoint.

The shipped tokenizer files are a plain fast tokenizer plus three loose ends
this class ties off, all of which are silent failures otherwise:

* The chat template ships as ``chat_template.jinja`` beside the weights
  and is **not** referenced from ``tokenizer_config.json``, so
  ``apply_chat_template`` would raise (or fall back to a default template)
  unless the file is read explicitly.
* No pad token is configured, so batched encoding fails. The checkpoint's
  ``generation_config.json`` names 200018 for padding.
* ``generation_config.json`` lists two EOS ids (200001 ``<|end_of_text|>`` and
  200008 ``<|eot|>``), while ``tokenizer_config.json`` names only the first.
  Both must terminate generation, and ``<|eom|>`` (200007) must not: it ends
  every reasoning block and non-final tool call, after which the model keeps
  going on its own.

Nothing here rewrites the vocabulary or the pre-tokenizer: in particular the
``fix_mistral_regex`` compatibility switch stays off, because enabling it
changes how digits are split.
"""

import json
import os
from typing import List, Optional

from tensorrt_llm.logger import logger

from .tokenizer import TransformersTokenizer

# The HF-standard sidecar name (recent transformers also loads it natively;
# this fallback covers tokenizers constructed without that path).
CHAT_TEMPLATE_FILENAME = "chat_template.jinja"

# ``<|end_of_text|>`` and ``<|eot|>``.
EOS_TOKEN_IDS = (200001, 200008)
# ``<|eom|>``: a channel boundary, never a stop token.
END_OF_MESSAGE_TOKEN_ID = 200007
PAD_TOKEN_ID = 200018


def read_chat_template(model_dir: str) -> Optional[str]:
    """Return the checkpoint's chat template text, or None if absent."""
    path = os.path.join(model_dir, CHAT_TEMPLATE_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def read_generation_eos_ids(model_dir: str) -> Optional[List[int]]:
    """EOS ids from ``generation_config.json``, or None if it has none."""
    path = os.path.join(model_dir, "generation_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        eos = json.load(handle).get("eos_token_id")
    if eos is None:
        return None
    return [int(eos)] if isinstance(eos, int) else [int(i) for i in eos]


class MuseSparkOpenTokenizer(TransformersTokenizer):
    """Fast tokenizer plus the checkpoint's template, pad token and stop ids."""

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_dir, **kwargs)
        tokenizer.configure_for_musespark_open(pretrained_model_dir)
        return tokenizer

    def configure_for_musespark_open(self, model_dir: str) -> None:
        inner = self.tokenizer

        if not getattr(inner, "chat_template", None):
            template = read_chat_template(model_dir)
            if template is None:
                raise ValueError(
                    f"{model_dir} has no chat template: neither "
                    f"tokenizer_config.json nor {CHAT_TEMPLATE_FILENAME}"
                )
            inner.chat_template = template

        if inner.pad_token_id is None:
            pad_token = inner.convert_ids_to_tokens(PAD_TOKEN_ID)
            if pad_token is None:
                raise ValueError(
                    f"{model_dir} has no token with id {PAD_TOKEN_ID} to use for padding"
                )
            inner.pad_token = pad_token

        eos_ids = read_generation_eos_ids(model_dir) or list(EOS_TOKEN_IDS)
        if END_OF_MESSAGE_TOKEN_ID in eos_ids:
            raise ValueError(
                f"{model_dir} lists <|eom|> ({END_OF_MESSAGE_TOKEN_ID}) as an "
                "EOS id; it is a channel boundary and would truncate every "
                "reasoning block"
            )
        self.musespark_open_eos_token_ids = eos_ids
        if inner.eos_token_id not in eos_ids:
            logger.warning(
                f"tokenizer_config eos_token_id {inner.eos_token_id} is not in "
                f"the checkpoint's generation_config eos ids {eos_ids}"
            )

    @property
    def stop_token_ids(self) -> List[int]:
        """Every id that ends a turn. ``<|eom|>`` is deliberately absent."""
        return list(getattr(self, "musespark_open_eos_token_ids", list(EOS_TOKEN_IDS)))
