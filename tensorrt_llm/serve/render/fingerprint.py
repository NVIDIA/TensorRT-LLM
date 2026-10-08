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
"""Rendering-configuration fingerprint.

Two processes produce the same prompt token ids for the same request only if
they render with the same chat template, tokenizer, model type, model-specific
extension and input processor. The fingerprint is a digest over exactly those
inputs, so a router or a worker can check that it may trust ids rendered
elsewhere. Settings that do not change the ids (the tool parser, for example)
are listed for information but excluded from the digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .resources import RenderResources

FINGERPRINT_VERSION = 1


def _sha256(value: Any) -> Optional[str]:
    """Stable digest of a JSON-serializable value; ``None`` stays ``None``."""
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenizer_digest(tokenizer: Any) -> Dict[str, Any]:
    """Identity of a tokenizer independent of where its files live."""
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    info: Dict[str, Any] = {"class": f"{type(inner).__module__}.{type(inner).__qualname__}"}
    get_vocab = getattr(inner, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = dict(get_vocab())
            info["vocab_size"] = len(vocab)
            info["vocab_sha256"] = _sha256(sorted(vocab.items()))
        except (TypeError, ValueError):
            # A tokenizer without a usable vocabulary is identified by its class.
            pass
    special = getattr(inner, "special_tokens_map", None)
    info["special_tokens_sha256"] = _sha256(special if isinstance(special, dict) else None)
    added = getattr(inner, "added_tokens_decoder", None)
    if isinstance(added, dict) and added:
        info["added_tokens_sha256"] = _sha256(sorted((int(k), str(v)) for k, v in added.items()))
    return info


def _template_sources(res: "RenderResources") -> Dict[str, Optional[str]]:
    """Digests of every place a chat template can come from."""
    inner = getattr(res.tokenizer, "tokenizer", res.tokenizer)
    return {
        "server": _sha256(res.default_chat_template),
        "processor": _sha256(getattr(res.processor, "chat_template", None)),
        "tokenizer": _sha256(getattr(inner, "chat_template", None)),
    }


def compute_fingerprint(res: "RenderResources") -> Dict[str, Any]:
    """Fingerprint of the rendering configuration held by ``res``."""
    input_processor = res.input_processor
    input_processor_kind = (
        "default" if input_processor is None else type(input_processor).__qualname__
    )
    compared: Dict[str, Any] = {
        "version": FINGERPRINT_VERSION,
        "model_type": res.model_type,
        "extension": type(res.extension).__qualname__,
        "use_harmony": bool(res.use_harmony),
    }
    if not res.use_harmony:
        # Harmony renders with its own encoding, independent of the chat
        # template, the tokenizer files and the input processor.
        compared.update(
            {
                "custom_tokenizer": res.custom_tokenizer,
                "input_processor": input_processor_kind,
                "templates": _template_sources(res),
                "tokenizer": _tokenizer_digest(res.tokenizer),
            }
        )
    return {
        "digest": _sha256(compared),
        "compared": compared,
        # Informational only: does not change the prompt token ids.
        "info": {"tool_parser": res.tool_parser, "reasoning_parser": res.reasoning_parser},
    }


def fingerprints_match(local: Optional[Dict[str, Any]], remote: Optional[Dict[str, Any]]) -> bool:
    """Whether ids rendered under ``remote`` may be trusted under ``local``.

    A missing fingerprint on either side counts as a mismatch.
    """
    if not local or not remote:
        return False
    digest = local.get("digest")
    return digest is not None and digest == remote.get("digest")
