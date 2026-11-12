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
elsewhere. The chat template is the one the renderer resolves, not every place
one could come from. Settings that do not change the ids (the tool parser, for
example) are listed for information but excluded from the digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .resources import RenderResources


def _sha256(value: Any) -> Optional[str]:
    """Stable digest of a JSON-serializable value; ``None`` stays ``None``."""
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Tokenizer init options that change the ids a text encodes to (an allowlist: the rest of
# ``init_kwargs`` is loading provenance such as paths).
_ENCODING_INIT_KWARGS = (
    "add_prefix_space",
    "add_bos_token",
    "add_eos_token",
    "legacy",
    "split_special_tokens",
    "do_lower_case",
    "strip_accents",
    "tokenize_chinese_chars",
)


def _encoding_digest(inner: Any) -> Optional[Dict[str, Any]]:
    """Digest of everything that turns text into ids, or ``None`` if it cannot be established.

    A vocabulary alone does not identify the transformation: the merges, normalizer,
    pre-tokenizer, post-processor and added-token matching rules also decide the ids.
    A fast (Rust) tokenizer serializes all of them; a tiktoken-based tokenizer is
    identified by its split pattern, mergeable ranks and special tokens.
    """
    backend = getattr(inner, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        try:
            serialized = json.loads(to_str())
            # Truncation and padding are request-time state: Transformers enables them on
            # the live backend while encoding a request and does not restore them, and the
            # serialization includes them. They belong to the request, not to the identity,
            # so a fingerprint must not depend on which requests the process has served.
            # (Canonicalized on the parsed copy; the live tokenizer is never touched.)
            serialized["truncation"] = None
            serialized["padding"] = None
            return {"kind": "tokenizers", "sha256": _sha256(serialized)}
        except Exception:  # noqa: BLE001 - fall through to the other identities
            pass
    for name in ("model", "tokenizer", "encoding", "_encoding", "tiktoken_model"):
        encoding = getattr(inner, name, None)
        ranks = getattr(encoding, "_mergeable_ranks", None)
        if isinstance(ranks, dict) and ranks:
            specials = getattr(encoding, "_special_tokens", None) or {}
            return {
                "kind": "tiktoken",
                "pattern_sha256": _sha256(getattr(encoding, "_pat_str", None)),
                "ranks_sha256": _sha256(sorted((bytes(k).hex(), v) for k, v in ranks.items())),
                "special_tokens_sha256": _sha256(sorted(specials.items())),
            }
    return None


def _tokenizer_digest(tokenizer: Any) -> Dict[str, Any]:
    """Identity of a tokenizer independent of where its files live.

    ``complete`` says whether the whole encoding configuration could be identified.
    Ids rendered under an incomplete identity are never trusted by
    :func:`fingerprints_match`, even when both sides report the same digest.
    """
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    info: Dict[str, Any] = {"class": f"{type(inner).__module__}.{type(inner).__qualname__}"}
    # Which end of the prompt ``truncate_prompt_tokens`` keeps; it is tokenizer state, not part
    # of the serialized encoding.
    info["truncation_side"] = getattr(inner, "truncation_side", None)
    get_vocab = getattr(inner, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = dict(get_vocab())
            info["vocab_size"] = len(vocab)
            info["vocab_sha256"] = _sha256(sorted(vocab.items()))
        except (TypeError, ValueError):
            pass
    special = getattr(inner, "special_tokens_map", None)
    info["special_tokens_sha256"] = _sha256(special if isinstance(special, dict) else None)
    added = getattr(inner, "added_tokens_decoder", None)
    if isinstance(added, dict) and added:
        # The matching rules (lstrip/rstrip/normalized/special) are part of the identity.
        info["added_tokens_sha256"] = _sha256(
            sorted(
                (int(k), repr(v), getattr(v, "__dict__", None) and str(v.__dict__))
                for k, v in added.items()
            )
        )
    init_kwargs = getattr(inner, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        # Only options that change how text is split into ids. ``init_kwargs`` also holds
        # loading provenance (``name_or_path``, resolved file paths), which differs for the
        # same checkpoint mounted at two locations and must not be part of the identity.
        info["init_kwargs_sha256"] = _sha256(
            {
                k: init_kwargs[k]
                for k in _ENCODING_INIT_KWARGS
                if isinstance(init_kwargs.get(k), (str, int, float, bool, type(None)))
                and k in init_kwargs
            }
        )
    encoding = _encoding_digest(inner)
    info["encoding"] = encoding
    info["complete"] = encoding is not None
    return info


def _harmony_identity() -> Dict[str, Any]:
    """Harmony renders with its own encoding, so its implementation is the identity."""
    try:
        from importlib.metadata import version

        package = version("openai-harmony")
    except Exception:  # noqa: BLE001
        package = None
    return {"encoding": "HARMONY_GPT_OSS", "package": "openai-harmony", "version": package}


def _resolved_templates(res: "RenderResources") -> Dict[str, Optional[str]]:
    """Digests of the chat template the renderer resolves, for a request without tools and with.

    The template is chosen by ``resolve_hf_chat_template`` (the server template, else the
    processor's, else the tokenizer's, and the tokenizer's tool variant when tools are given),
    so the fingerprint calls that function instead of listing where a template could come
    from. Listing the sources made a worker whose ``AutoProcessor`` fell back to the
    tokenizer (every text-only checkpoint) differ from a router that has no processor, though
    both render with the same text.
    """
    from tensorrt_llm.inputs.utils import resolve_hf_chat_template

    inner = getattr(res.tokenizer, "tokenizer", res.tokenizer)
    # A request can select a named template (``chat_template="alternate"``) from a dict-valued
    # tokenizer template, so every variant is part of the identity, not only the defaults.
    named = getattr(inner, "chat_template", None)
    resolved: Dict[str, Optional[str]] = {
        "named": _sha256(named if isinstance(named, dict) else None)
    }
    if not any(
        getattr(source, "chat_template", None) is not None for source in (res.processor, inner)
    ) and (res.default_chat_template is None):
        # No Jinja template anywhere (a tokenizer-native renderer): the tokenizer's own
        # identity covers it, and the resolver would only log a warning.
        return {**resolved, "without_tools": None, "with_tools": None}
    for shape, tools in (("without_tools", None), ("with_tools", [{}])):
        try:
            template = resolve_hf_chat_template(
                inner, res.processor, res.default_chat_template, tools
            )
        except Exception:  # noqa: BLE001 - a template that cannot be resolved is "none"
            template = None
        resolved[shape] = _sha256(template)
    return resolved


def compute_fingerprint(res: "RenderResources") -> Dict[str, Any]:
    """Fingerprint of the rendering configuration held by ``res``."""
    input_processor = res.input_processor
    # "No input processor" (a router that only has a tokenizer) and the engine's
    # DefaultInputProcessor are the same plain-text tokenization, so they share one identity.
    input_processor_kind = (
        "default"
        if input_processor is None or type(input_processor).__qualname__ == "DefaultInputProcessor"
        else type(input_processor).__qualname__
    )
    compared: Dict[str, Any] = {
        # Harmony renders the same for any model type, so a router that was told to use
        # Harmony without being able to resolve the type still matches its workers.
        "model_type": None if res.use_harmony else (res.model_type or None),
        "extension": type(res.extension).__qualname__,
        # Bumped by an extension whose rendering output changes between revisions.
        "extension_render_version": getattr(res.extension, "render_version", 1),
        "use_harmony": bool(res.use_harmony),
    }
    if res.use_harmony:
        compared["harmony"] = _harmony_identity()
    else:
        # Harmony renders with its own encoding, independent of the chat
        # template, the tokenizer files and the input processor.
        compared.update(
            {
                "custom_tokenizer": res.custom_tokenizer,
                "input_processor": input_processor_kind,
                "templates": _resolved_templates(res),
                "tokenizer": _tokenizer_digest(res.tokenizer),
            }
        )
    return {
        "digest": _sha256(compared),
        "compared": compared,
        # Informational only: does not change the prompt token ids.
        "info": {"tool_parser": res.tool_parser, "reasoning_parser": res.reasoning_parser},
    }


def fingerprint_mismatch(
    local: Optional[Dict[str, Any]], remote: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Why ids rendered under ``remote`` may not be trusted under ``local``; None if they may.

    A missing fingerprint on either side counts as a mismatch.
    """
    if not local:
        return "this side has no rendering fingerprint"
    if not remote:
        return "no rendering fingerprint reported"
    if not _identity_complete(local):
        # A tokenizer whose whole encoding configuration could not be identified is never
        # trusted, even when both sides happen to report the same digest.
        return "this side's tokenizer cannot be fully identified"
    if not _identity_complete(remote):
        return "the other side's tokenizer cannot be fully identified"
    digest = local.get("digest")
    if digest is None or digest != remote.get("digest"):
        return "renders differently (different fingerprint)"
    return None


def fingerprints_match(local: Optional[Dict[str, Any]], remote: Optional[Dict[str, Any]]) -> bool:
    """Whether ids rendered under ``remote`` may be trusted under ``local``."""
    return fingerprint_mismatch(local, remote) is None


def fingerprint_summary(fingerprint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The part of a fingerprint :func:`fingerprints_match` reads, small enough to publish.

    A coordinator sends this per worker to the frontends, which never fetch ``/server_info``
    themselves. A summary matches exactly like the full fingerprint it came from.
    """
    if not fingerprint:
        return None
    compared = fingerprint.get("compared") or {}
    return {
        "digest": fingerprint.get("digest"),
        "compared": {
            "use_harmony": bool(compared.get("use_harmony")),
            "tokenizer": {"complete": bool((compared.get("tokenizer") or {}).get("complete"))},
        },
        # Not compared; read by a router deciding whether a prompt's reasoning mode is lost.
        "reasoning_parser": (fingerprint.get("info") or {}).get("reasoning_parser"),
    }


def worker_resolves_thinking(
    fingerprint: Optional[Dict[str, Any]], resolved_thinking: Dict[str, bool]
) -> bool:
    """Whether the worker behind ``fingerprint`` reads a reasoning mode off the prompt.

    ``resolved_thinking`` is what the prompt carries, per reasoning parser that takes its
    mode from there. Only the parser the worker runs matters: a worker without a reasoning
    parser, or with one that does not read the prompt, decides nothing from it and may be
    given ids. A fingerprint that does not say which parser runs (an older worker) cannot
    be judged, so it counts as one that does.
    """
    if not resolved_thinking:
        return False
    info = (fingerprint or {}).get("info")
    if isinstance(info, dict) and "reasoning_parser" in info:
        parser = info["reasoning_parser"]
    elif "reasoning_parser" in (fingerprint or {}):
        parser = fingerprint["reasoning_parser"]
    else:
        return True
    return parser in resolved_thinking


def _identity_complete(fingerprint: Dict[str, Any]) -> bool:
    compared = fingerprint.get("compared") or {}
    if compared.get("use_harmony"):
        return True
    return bool((compared.get("tokenizer") or {}).get("complete"))
