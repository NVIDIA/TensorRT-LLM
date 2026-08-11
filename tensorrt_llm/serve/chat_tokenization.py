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

from __future__ import annotations

import bisect
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Hashable, Optional, Protocol, TypedDict, cast

from transformers import PretrainedConfig

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from tensorrt_llm.serve.harmony_adapter import HarmonyAdapter

ToolDict = dict[str, object]


class _OffsetEncoding(TypedDict):
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]


class OffsetTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into canonical token IDs."""
        ...

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> _OffsetEncoding:
        """Encode text and optionally return character offsets."""
        ...


class IncrementalTokenizationCache:
    """Boundary-safe rendered-prompt token cache shared by frontends and routers."""

    _PREFIX_SCAN_CHUNK_SIZE = 64 * 1024

    def __init__(
        self,
        *,
        name: str,
        enabled: bool,
        rollback_tokens: int = 1,
        verify_every: int = 0,
        max_entries: int = 1024,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.rollback_tokens = max(1, rollback_tokens)
        self.verify_every = max(0, verify_every)
        self.max_entries = max(1, max_entries)
        self._cache: OrderedDict[
            Hashable,
            tuple[str, list[int], list[tuple[int, int]] | None],
        ] = OrderedDict()
        self._lock = threading.Lock()
        self._verify_count = 0
        self._incremental_hits = 0

    @classmethod
    def from_environment(cls, name: str) -> "IncrementalTokenizationCache":
        """Create a cache from the shared incremental-tokenization environment flags."""
        return cls(
            name=name,
            enabled=os.environ.get("TRTLLM_INCREMENTAL_TOKENIZE", "0") == "1",
            rollback_tokens=int(os.environ.get("TRTLLM_INCREMENTAL_TOKENIZE_ROLLBACK_TOKENS", "1")),
            verify_every=int(os.environ.get("TRTLLM_INCREMENTAL_TOKENIZE_VERIFY_EVERY", "0")),
        )

    def encode(
        self,
        rendered: str,
        key: Hashable,
        tokenizer: OffsetTokenizer,
    ) -> list[int]:
        """Tokenize a rendered prompt while reusing a stable cached prefix."""
        with self._lock:
            entry = self._cache.get(key)

        used_incremental = False
        if entry is not None and rendered == entry[0]:
            ids = entry[1]
            offsets = entry[2]
        elif self.enabled and entry is not None and entry[2]:
            previous_text, previous_ids, previous_offsets = entry
            prefix_limit = min(len(previous_text), len(rendered))
            common_prefix_chars = 0
            chunk_size = self._PREFIX_SCAN_CHUNK_SIZE
            while common_prefix_chars + chunk_size <= prefix_limit:
                chunk_end = common_prefix_chars + chunk_size
                if (
                    previous_text[common_prefix_chars:chunk_end]
                    != rendered[common_prefix_chars:chunk_end]
                ):
                    break
                common_prefix_chars = chunk_end

            # Locate the exact mismatch within at most one bounded chunk.
            low = common_prefix_chars
            high = min(common_prefix_chars + chunk_size, prefix_limit)
            while low < high:
                middle = (low + high + 1) // 2
                if (
                    previous_text[common_prefix_chars:middle]
                    == rendered[common_prefix_chars:middle]
                ):
                    low = middle
                else:
                    high = middle - 1
            common_prefix_chars = low

            stable_tokens = bisect.bisect_right(
                previous_offsets,
                common_prefix_chars,
                key=lambda offset: offset[1],
            )
            # BPE merges may cross the character-level common-prefix boundary.
            cut_token = max(0, stable_tokens - self.rollback_tokens)
            cut_char = 0 if cut_token == 0 else previous_offsets[cut_token][0]
            suffix_ids, suffix_offsets = self._encode_with_offsets(rendered[cut_char:], tokenizer)
            if suffix_offsets is not None:
                ids = previous_ids[:cut_token] + suffix_ids
                offsets = previous_offsets[:cut_token] + [
                    (start + cut_char, end + cut_char) for start, end in suffix_offsets
                ]
                used_incremental = True
            else:
                ids, offsets = self._encode_with_offsets(rendered, tokenizer)
        elif self.enabled:
            ids, offsets = self._encode_with_offsets(rendered, tokenizer)
        else:
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            offsets = None

        with self._lock:
            self._verify_count += 1
            verify_count = self._verify_count

        if used_incremental and self.verify_every > 0 and verify_count % self.verify_every == 0:
            full_ids = tokenizer.encode(rendered, add_special_tokens=False)
            if ids != full_ids:
                logger.error(
                    f"Incremental tokenization mismatch in {self.name}; "
                    f"falling back to full encode for cache key {key!r}"
                )
                ids = full_ids
                offsets = None

        with self._lock:
            self._cache[key] = (rendered, ids, offsets)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            if used_incremental:
                self._incremental_hits += 1
                incremental_hits = self._incremental_hits
            else:
                incremental_hits = 0

        if incremental_hits and (incremental_hits == 1 or incremental_hits % 1000 == 0):
            logger.info(
                f"Incremental tokenization exercised in {self.name}: hits={incremental_hits}"
            )
        return ids

    @staticmethod
    def _encode_with_offsets(
        rendered: str, tokenizer: OffsetTokenizer
    ) -> tuple[list[int], list[tuple[int, int]] | None]:
        """Return canonical token IDs and validated offsets when supported."""
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        try:
            encoding = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offset_ids = list(encoding["input_ids"])
            offsets = list(encoding["offset_mapping"])
        except (KeyError, NotImplementedError, TypeError, ValueError):
            return ids, None
        if ids != offset_ids or len(ids) != len(offsets):
            return ids, None
        return ids, offsets


def resolve_model_type_from_config(model_name_or_path: str) -> Optional[str]:
    """Return the checkpoint's declared model type from its config metadata."""
    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    model_type = config_dict.get("model_type")
    return model_type if isinstance(model_type, str) else None


def uses_harmony_tokenization(
    use_harmony: Optional[bool] = None,
    model_type: Optional[str] = None,
    model_type_resolver: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    if os.getenv("DISABLE_HARMONY_ADAPTER", "0") == "1":
        return False
    if use_harmony is not None:
        return use_harmony
    if model_type is None and model_type_resolver is not None:
        model_type = model_type_resolver()
    return model_type == "gpt_oss"


def get_chat_completion_tool_dicts(
    request: ChatCompletionRequest, empty_as_none: bool = False
) -> Optional[list[ToolDict]]:
    if request.tools is None or (empty_as_none and not request.tools):
        return None
    tools: list[ToolDict] = []
    for tool in request.tools:
        if hasattr(tool, "model_dump"):
            tools.append(cast(ToolDict, tool.model_dump()))
        elif isinstance(tool, dict):
            tools.append(cast(ToolDict, tool))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool).__name__}")
    return tools


def tokenize_harmony_chat_request(
    request: ChatCompletionRequest,
    harmony_adapter: Optional["HarmonyAdapter"] = None,
    set_prompt_token_ids: bool = False,
) -> list[int]:
    if request.prompt_token_ids is not None:
        return request.prompt_token_ids

    from tensorrt_llm.serve import harmony_adapter as harmony_adapter_module

    adapter = harmony_adapter or harmony_adapter_module.get_harmony_adapter()
    result = adapter.openai_to_harmony_tokens(
        request.messages,
        get_chat_completion_tool_dicts(request, empty_as_none=True),
        reasoning_effort=harmony_adapter_module.maybe_transform_reasoning_effort(
            request.reasoning_effort
        ),
        tool_choice=request.tool_choice,
    )
    if set_prompt_token_ids:
        request.prompt_token_ids = result
    return result


def render_chat_request_for_tokenizer(
    request: ChatCompletionRequest, tokenizer: object
) -> str | list[int]:
    chat_template_kwargs = (
        dict(request.chat_template_kwargs) if getattr(request, "chat_template_kwargs", None) else {}
    )
    chat_template_kwargs["tools"] = get_chat_completion_tool_dicts(request)
    chat_template_kwargs["documents"] = request.documents
    if request.chat_template is not None:
        chat_template_kwargs["chat_template"] = request.chat_template
    rendered = tokenizer.apply_chat_template(
        [msg if isinstance(msg, dict) else dict(msg) for msg in request.messages],
        add_generation_prompt=request.add_generation_prompt,
        tokenize=False,
        return_dict=False,
        **chat_template_kwargs,
    )
    if isinstance(rendered, str):
        return rendered
    return list(rendered)


def tokenize_chat_request_for_serving(
    request: ChatCompletionRequest,
    tokenizer_factory: Callable[[], object],
    encode_rendered: Callable[[str, object], list[int]],
    use_harmony: Optional[bool] = None,
    model_type: Optional[str] = None,
    model_type_resolver: Optional[Callable[[], Optional[str]]] = None,
    harmony_adapter: Optional["HarmonyAdapter"] = None,
    set_prompt_token_ids: bool = True,
) -> list[int]:
    if request.prompt_token_ids is not None:
        return request.prompt_token_ids

    if uses_harmony_tokenization(
        use_harmony=use_harmony,
        model_type=model_type,
        model_type_resolver=model_type_resolver,
    ):
        return tokenize_harmony_chat_request(
            request,
            harmony_adapter=harmony_adapter,
            set_prompt_token_ids=set_prompt_token_ids,
        )

    tokenizer = tokenizer_factory()
    rendered = render_chat_request_for_tokenizer(request, tokenizer)
    result = encode_rendered(rendered, tokenizer) if isinstance(rendered, str) else rendered
    if set_prompt_token_ids:
        request.prompt_token_ids = result
    return result
