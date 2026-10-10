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
"""Everything the prompt-preparation core needs, loaded without an executor."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.serving_extensions import ServingExtension, get_serving_extension

from .fingerprint import compute_fingerprint


def harmony_enabled_for(model_type: Optional[str]) -> bool:
    """Whether prompts for ``model_type`` are rendered by the Harmony adapter."""
    if os.getenv("DISABLE_HARMONY_ADAPTER", "0") == "1":
        return False
    return model_type == "gpt_oss"


@dataclass(frozen=True)
class RenderResources:
    """CPU-only resources for rendering chat requests into prompts.

    The same object is built three ways: from a running server (``from_server``),
    from a bare tokenizer for a router (``from_tokenizer``), and from a checkpoint
    directory on a host with no GPU and no weights (``load``).
    """

    # Tokenizer used by the chat template. The engine's wrapper or a raw HF
    # tokenizer; the template path unwraps it.
    tokenizer: Any
    model_type: Optional[str] = None
    # Engine input processor. ``None`` means plain tokenization with ``tokenizer``.
    input_processor: Any = None
    processor: Any = None
    # Loaded HF model config, used to resolve the model type and media placeholders.
    hf_config: Any = None
    # Server-side ``--chat_template``; a request-level template wins over it.
    default_chat_template: Optional[str] = None
    extension: ServingExtension = field(default_factory=ServingExtension)
    use_harmony: bool = False
    # Harmony adapter to use instead of the process-wide one.
    harmony: Any = None
    allow_request_chat_template: bool = False
    multimodal_server_config: Any = None
    tool_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    custom_tokenizer: Optional[str] = None

    def legacy_view(self) -> "RenderResources":
        """Resources approximating the pre-merge inputs (``TRTLLM_RENDER_LEGACY=1``).

        The governor, Responses and multimodal-encoder routes rendered without
        the server-side chat template and without the model extension's rules.
        This restores those inputs only; the render and tokenize steps are still the
        shared ones (see :func:`~tensorrt_llm.serve.render.chat.legacy_render_enabled`).
        """
        return dataclasses.replace(self, default_chat_template=None, extension=ServingExtension())

    def fingerprint(self) -> dict:
        """Rendering-configuration fingerprint (see :mod:`.fingerprint`)."""
        return compute_fingerprint(self)

    @classmethod
    def from_server(cls, server: Any) -> "RenderResources":
        """Resources of a running ``OpenAIServer``, read at call time.

        Attributes the server does not have (a partially built server in tests,
        or a role without them) fall back to their defaults.
        """
        from tensorrt_llm.serve.chat_utils import resolve_top_level_model_type

        hf_config = getattr(server, "model_config", None)
        model_type = resolve_top_level_model_type(hf_config) if hf_config is not None else None
        generator = getattr(server, "generator", None)
        args = getattr(generator, "args", None)
        return cls(
            tokenizer=getattr(server, "tokenizer", None),
            model_type=model_type,
            input_processor=getattr(generator, "input_processor", None),
            processor=getattr(server, "processor", None),
            hf_config=hf_config,
            default_chat_template=getattr(server, "chat_template", None),
            extension=get_serving_extension(model_type),
            use_harmony=bool(getattr(server, "use_harmony", False)),
            harmony=getattr(server, "harmony_adapter", None),
            allow_request_chat_template=bool(getattr(server, "allow_request_chat_template", False)),
            multimodal_server_config=getattr(server, "multimodal_server_config", None),
            tool_parser=getattr(server, "tool_parser", None),
            reasoning_parser=getattr(args, "reasoning_parser", None),
            custom_tokenizer=getattr(args, "custom_tokenizer", None),
        )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        model_type: Optional[str] = None,
        use_harmony: Optional[bool] = None,
        custom_tokenizer: Optional[str] = None,
    ) -> "RenderResources":
        """Resources for a caller that has only a tokenizer (the KV-aware router).

        There is no processor, model config or server template here, so the
        fingerprint of these resources only matches a server that has none of
        those either; otherwise the caller must not trust ids it renders.
        """
        return cls(
            tokenizer=tokenizer,
            model_type=model_type,
            extension=get_serving_extension(model_type),
            use_harmony=harmony_enabled_for(model_type) if use_harmony is None else use_harmony,
            # The router does not police request-level templates; the worker
            # rejects a disallowed one when the request reaches it.
            allow_request_chat_template=True,
            custom_tokenizer=custom_tokenizer,
        )

    @classmethod
    def load(
        cls,
        model: str,
        *,
        tokenizer: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer_mode: str = "auto",
        custom_tokenizer: Optional[str] = None,
        chat_template: Optional[str] = None,
        checkpoint_format: Optional[str] = "HF",
        enable_tokenization_cache: bool = False,
        tool_parser: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
        allow_request_chat_template: bool = False,
    ) -> "RenderResources":
        """Load resources from a checkpoint directory without building an executor.

        Only the tokenizer, configuration and processor files are read; no model
        weights are opened and no GPU is touched. Mirrors what ``LLM`` and
        ``OpenAIServer`` load, so the same request renders to the same ids.
        """
        from transformers import AutoProcessor

        from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
        from tensorrt_llm.inputs.registry import create_input_processor
        from tensorrt_llm.llmapi.llm_utils import ModelLoader
        from tensorrt_llm.serve.chat_utils import load_chat_template, resolve_top_level_model_type
        from tensorrt_llm.tokenizer import load_custom_tokenizer

        tokenizer_path = tokenizer or model
        use_fast = tokenizer_mode != "slow"
        if custom_tokenizer:
            loaded = load_custom_tokenizer(
                custom_tokenizer, tokenizer_path, trust_remote_code=trust_remote_code
            )
        else:
            loaded = ModelLoader.load_hf_tokenizer(
                tokenizer_path, trust_remote_code=trust_remote_code, use_fast=use_fast
            )
        if loaded is None:
            raise ValueError(f"Could not load a tokenizer from {tokenizer_path!r}.")

        input_processor = create_input_processor(
            model,
            loaded,
            checkpoint_format,
            trust_remote_code=trust_remote_code,
            enable_tokenization_cache=enable_tokenization_cache,
        )
        # The engine takes its tokenizer from the input processor.
        loaded = getattr(input_processor, "tokenizer", loaded)

        processor = None
        if checkpoint_format not in ("mistral", "mistral_large_3"):
            try:
                processor = AutoProcessor.from_pretrained(
                    model, trust_remote_code=trust_remote_code
                )
            except Exception:
                logger.debug(f"No AutoProcessor for {model}.")
        # The model type selects the serving extension and Harmony. A worker cannot start
        # without its config, so a renderer that could not read it must not quietly fall
        # back to generic text rendering (a different fingerprint, a different prompt).
        try:
            hf_config = load_pretrained_config(
                model, trust_remote_code=trust_remote_code, checkpoint_format=checkpoint_format
            )
        except Exception as error:
            raise ValueError(
                f"Could not load the model configuration from {model!r}: {error}"
            ) from error
        model_type = resolve_top_level_model_type(hf_config)
        return cls(
            tokenizer=loaded,
            model_type=model_type,
            input_processor=input_processor,
            processor=processor,
            hf_config=hf_config,
            default_chat_template=load_chat_template(chat_template),
            extension=get_serving_extension(model_type),
            use_harmony=harmony_enabled_for(model_type),
            allow_request_chat_template=allow_request_chat_template,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            custom_tokenizer=custom_tokenizer,
        )
