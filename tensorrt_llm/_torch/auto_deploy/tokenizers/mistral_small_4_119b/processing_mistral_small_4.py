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

"""Temporary Mistral Small 4 tokenizer/processor bridge for AutoDeploy.

This exists only until TRT-LLM upgrades to a transformers release that natively exposes the
upstream ``TokenizersBackend`` class referenced by the checkpoint metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from tokenizers import Tokenizer
from transformers import PixtralImageProcessorFast, PixtralProcessor, PreTrainedTokenizerFast
from transformers.utils import cached_file

_PROCESSOR_CONFIG_FILE = "processor_config.json"
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_FILE = "tokenizer.json"
_SOURCE_MODEL_KEY = "source_model_name_or_path"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _read_sidecar_config(
    pretrained_model_name_or_path: str | Path, filename: str
) -> Dict[str, Any]:
    path = Path(pretrained_model_name_or_path)
    if not path.is_dir():
        return {}
    config_path = path / filename
    if not config_path.is_file():
        return {}
    return _load_json(config_path)


def _resolve_source_model(pretrained_model_name_or_path: str | Path, filename: str) -> str | Path:
    sidecar_config = _read_sidecar_config(pretrained_model_name_or_path, filename)
    return sidecar_config.get(_SOURCE_MODEL_KEY, pretrained_model_name_or_path)


def _cached_text(source_model_name_or_path: str | Path, filename: str, **kwargs) -> Optional[str]:
    resolved = cached_file(source_model_name_or_path, filename, **kwargs)
    if resolved is None:
        return None
    return Path(resolved).read_text()


class ADMistralSmall4Tokenizer(PreTrainedTokenizerFast):
    """Temporary replacement for the upstream v5 ``TokenizersBackend`` class."""

    vocab_files_names = {"tokenizer_file": _TOKENIZER_FILE}
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *inputs,
        **kwargs,
    ) -> "ADMistralSmall4Tokenizer":
        del inputs
        kwargs.pop("_from_auto", None)
        kwargs.pop("_commit_hash", None)
        kwargs.pop("trust_remote_code", None)

        source_model_name_or_path = _resolve_source_model(
            pretrained_model_name_or_path, _TOKENIZER_CONFIG_FILE
        )
        source_tokenizer_config_path = cached_file(
            source_model_name_or_path,
            _TOKENIZER_CONFIG_FILE,
            **kwargs,
        )
        assert source_tokenizer_config_path is not None
        source_tokenizer_config = _load_json(Path(source_tokenizer_config_path))

        tokenizer_file = cached_file(source_model_name_or_path, _TOKENIZER_FILE, **kwargs)
        assert tokenizer_file is not None

        tokenizer = cls(
            tokenizer_object=Tokenizer.from_file(tokenizer_file),
            name_or_path=str(source_model_name_or_path),
            bos_token=source_tokenizer_config.get("bos_token"),
            eos_token=source_tokenizer_config.get("eos_token"),
            unk_token=source_tokenizer_config.get("unk_token"),
            pad_token=source_tokenizer_config.get("pad_token"),
            additional_special_tokens=source_tokenizer_config.get("extra_special_tokens", []),
            clean_up_tokenization_spaces=source_tokenizer_config.get(
                "clean_up_tokenization_spaces", False
            ),
            model_max_length=source_tokenizer_config.get("model_max_length"),
            padding_side=source_tokenizer_config.get("padding_side", "left"),
            truncation_side=source_tokenizer_config.get("truncation_side", "left"),
        )

        chat_template = _cached_text(source_model_name_or_path, _CHAT_TEMPLATE_FILE, **kwargs)
        if chat_template is not None:
            tokenizer.chat_template = chat_template

        return tokenizer


class ADMistralSmall4Processor(PixtralProcessor):
    """Temporary Pixtral processor wired to the local tokenizer bridge."""

    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoProcessor") -> None:
        del auto_class

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "ADMistralSmall4Processor":
        kwargs.pop("_from_auto", None)
        kwargs.pop("_commit_hash", None)
        kwargs.pop("trust_remote_code", None)

        source_model_name_or_path = _resolve_source_model(
            pretrained_model_name_or_path, _PROCESSOR_CONFIG_FILE
        )
        source_processor_config_path = cached_file(
            source_model_name_or_path,
            _PROCESSOR_CONFIG_FILE,
            **kwargs,
        )
        assert source_processor_config_path is not None
        source_processor_config = _load_json(Path(source_processor_config_path))

        image_processor = PixtralImageProcessorFast.from_pretrained(
            source_model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )
        tokenizer = ADMistralSmall4Tokenizer.from_pretrained(source_model_name_or_path, **kwargs)

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=source_processor_config.get("patch_size", 16),
            spatial_merge_size=source_processor_config.get("spatial_merge_size", 1),
            chat_template=getattr(tokenizer, "chat_template", None),
            image_token=source_processor_config.get("image_token", "[IMG]"),
            image_break_token=source_processor_config.get("image_break_token", "[IMG_BREAK]"),
            image_end_token=source_processor_config.get("image_end_token", "[IMG_END]"),
        )
