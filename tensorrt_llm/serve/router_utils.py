# Copyright (c) 2025-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared router utilities: request tokenization and KV-cache block hashing.

Extracted from ``router.py`` so the surface routers there can share a single
implementation of block hashing without importing the whole router module.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

from tensorrt_llm.bindings.internal.batch_manager import BlockKey as _NativeBlockKey
from tensorrt_llm.bindings.internal.batch_manager import BlockKeyHasher as _NativeBlockKeyHasher
from tensorrt_llm.inputs.prefix_token_cache import PrefixTokenCache, create_prefix_token_cache
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import kv_cache_hash
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    KVCacheEventManager,
    ReuseScope,
    sequence_to_blockchain_keys,
)
from tensorrt_llm.serve.chat_tokenization import (
    resolve_model_type_from_config,
    uses_harmony_tokenization,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)
from tensorrt_llm.serve.render import (
    HarmonyConversionError,
    PreparedContext,
    RenderResources,
    UnrenderableMessagesError,
    UnsupportedRenderError,
    fingerprint_mismatch,
    render_chat,
    render_chat_for_length,
    worker_resolves_thinking,
)

KV_CACHE_HASH_ALGO_DEFAULT = kv_cache_hash.KV_CACHE_HASH_ALGO_DEFAULT
KV_CACHE_HASH_ALGO_V1 = kv_cache_hash.KV_CACHE_HASH_ALGO_V1
KV_CACHE_HASH_ALGO_V2 = kv_cache_hash.KV_CACHE_HASH_ALGO_V2
KV_CACHE_HASH_ALGO_V2_SHA256_64 = kv_cache_hash.KV_CACHE_HASH_ALGO_V2_SHA256_64
KV_CACHE_HASH_ALGOS = frozenset(
    {
        KV_CACHE_HASH_ALGO_V1,
        KV_CACHE_HASH_ALGO_V2,
        KV_CACHE_HASH_ALGO_V2_SHA256_64,
    }
)
hash_v1_block_key = kv_cache_hash.hash_v1_block_key
truncate_sha256_hash_to_int64 = kv_cache_hash.truncate_sha256_hash_to_int64

OpenAIRequest = Union[CompletionRequest, ChatCompletionRequest, ResponsesRequest]
BlockHash = Union[int, str]

__all__ = [
    "KV_CACHE_HASH_ALGO_DEFAULT",
    "KV_CACHE_HASH_ALGO_V1",
    "KV_CACHE_HASH_ALGO_V2",
    "KV_CACHE_HASH_ALGO_V2_SHA256_64",
    "KV_CACHE_HASH_ALGOS",
    "hash_v1_block_key",
    "truncate_sha256_hash_to_int64",
    "OpenAIRequest",
    "BlockHash",
    "responses_request_text",
    "get_request_num_tokens",
    "block_key_hasher",
    "BlockHashMixin",
    "PrefixBlockSet",
]


class PrefixBlockSet:
    """Single-owner block-hash index -- a flat ``set`` of held block hashes.

    A KV-cache block hash folds in its parent chain (the worker computes it with
    ``BlockKeyHasher.hash(block_key, parent_hash)``), so every block-hash *value*
    is globally unique to one position in one prefix path. A request's ordered
    block-hash list is itself the prefix path, so longest-common-prefix against a
    set of held blocks is just "walk the list until the first hash the owner
    doesn't hold" -- no explicit tree needed.

    This is the exact structure the orchestrator ``KvCacheAwareServerState`` uses
    (a ``set[block_hash]`` per server, walked until the first miss). It is the
    right index whenever there is only ONE logical owner -- e.g. the centralized
    router's per-instance ``combined_trie`` (owner = instance) and each rank's
    trie (owner = that rank). Those are only ever queried for a single owner's
    prefix depth (:meth:`match_one`), so a ``hash -> {owner}`` reverse map with
    per-depth set intersections would be pure overhead here.

    The ``owner_id`` argument on :meth:`add` / :meth:`remove` / :meth:`match_one`
    is accepted and ignored so this is a drop-in for the single-owner call sites.
    """

    __slots__ = ("_blocks",)

    def __init__(self) -> None:
        self._blocks: set[int] = set()

    def add(self, owner_id: str, block_hashes: Iterable[int]) -> None:
        self._blocks.update(block_hashes)

    def remove(self, owner_id: str, block_hashes: Iterable[int]) -> None:
        self._blocks.difference_update(block_hashes)

    def remove_worker(self, owner_id: str) -> None:
        self._blocks.clear()

    def match_one(self, owner_id: str, block_hashes: List[int]) -> int:
        """Consecutive prefix-block count held by the (single) owner.

        Identical to ``KvCacheAwareServerState.matched_tokens``: walk the query
        path, counting blocks present in the set, and stop at the first miss.
        """
        blocks = self._blocks
        depth = 0
        for h in block_hashes:
            if h not in blocks:
                break
            depth += 1
        return depth

    def has_worker(self, owner_id: str) -> bool:
        return bool(self._blocks)


def responses_request_text(request: ResponsesRequest) -> str:
    """Return the instructions and input as ``role:content`` lines, like a chat request's text.

    A later turn re-sends the earlier turns' input, so its text extends theirs.
    """
    from tensorrt_llm.serve.responses_utils import chat_messages_from_responses_input

    messages = chat_messages_from_responses_input(request, [])
    return "\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in messages)


def get_request_num_tokens(request: Optional[OpenAIRequest]) -> int:
    if request is None:
        return 0

    if isinstance(request, ResponsesRequest):
        # Without relayed token ids the text is counted in characters, as a
        # string completion prompt is below, which weighs a request several
        # times what its token count would. A router sees one kind only: the
        # context router routes before any ids exist, and the generation
        # router gets them relayed (context-first) or never (generation-first).
        token_ids = request.relayed_prompt_token_ids()
        return len(responses_request_text(request) if token_ids is None else token_ids)

    if (
        request.disaggregated_params is None
        or request.disaggregated_params.request_type == "context_only"
    ):
        if isinstance(request, ChatCompletionRequest):
            raise ValueError(
                "LoadBalancing router with tokens doesn't support ChatCompletionRequest yet"
            )

        if isinstance(request.prompt, str) or (
            isinstance(request.prompt, list)
            and (not request.prompt or isinstance(request.prompt[0], int))
        ):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        num_tokens = sum(len(prompt) for prompt in prompts)
    elif request.disaggregated_params.request_type == "generation_only":
        raise ValueError(
            "LoadBalancing router with tokens doesn't support generation_only requests"
        )
    else:
        raise ValueError(f"Unsupported request type: {request.disaggregated_params.request_type}")

    return num_tokens


def block_key_hasher(
    token_ids: list[int], parent_hash: Optional[int] = None, cache_salt_id: Optional[int] = None
) -> int:
    parent = 0 if parent_hash is None else parent_hash
    # Fast path: the native C++ BlockKeyHasher is bit-exact with
    # hash_v1_block_key and avoids the per-token Python loop. Its hash() binding
    # takes no cache_salt_id, so fall back to Python only when a salt is set
    # (rare opt-in; never in the unsalted agent/chat completion path).
    if cache_salt_id is None:
        return _NativeBlockKeyHasher.hash(_NativeBlockKey(token_ids), parent)
    return hash_v1_block_key(token_ids, parent_hash=parent, cache_salt_id=cache_salt_id)


@dataclass(frozen=True)
class WriteBackDecision:
    """Whether rendered ids may be forwarded to the workers, and what blocks them if not."""

    allowed: bool
    mismatched: dict[str, str] = field(default_factory=dict)  # server -> reason
    needs_prompt_text: list[str] = field(default_factory=list)  # reasoning parser reads the prompt


class BlockHashMixin:
    """Shared tokenization and block-hash computation.

    Used by routers that need KV-cache-aware prefix matching.
    """

    def _init_block_hashing(
        self,
        tokens_per_block: Optional[int] = None,
        custom_tokenizer: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        use_harmony: Optional[bool] = None,
        model_path: Optional[str] = None,
        enable_tokenization_cache: bool = False,
    ) -> None:
        env_tokens_per_block = os.environ.get("TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK")
        if env_tokens_per_block is not None:
            tokens_per_block = int(env_tokens_per_block)
        self._tpb_auto = tokens_per_block is None
        self._tokens_per_block = 32 if tokens_per_block is None else tokens_per_block
        self._tokenizers: dict = {}
        self._model_types: dict[str, Optional[str]] = {}
        self._custom_tokenizer = custom_tokenizer
        self._tokenizer_dir = tokenizer_dir
        self._model_path = model_path
        self._use_harmony = use_harmony
        self._tok_prefix_cache: OrderedDict = OrderedDict()
        self._enable_tokenization_cache = enable_tokenization_cache
        # id(tokenizer) -> (tokenizer, its prefix token cache or None)
        self._prefix_token_caches: dict[int, tuple[object, Optional[PrefixTokenCache]]] = {}
        logger.info(
            f"BlockHashMixin: tokens_per_block={self._tokens_per_block}"
            f"{' (auto, adopts worker)' if self._tpb_auto else ''}"
            f", custom_tokenizer={self._custom_tokenizer}"
            f", model_path={self._model_path}"
            f", use_harmony={self._use_harmony}"
            f", enable_tokenization_cache={self._enable_tokenization_cache}"
        )

    def _get_tokenizer(self, model: str):
        if model not in self._tokenizers:
            model_path = self._tokenizer_dir or model
            # Only an operator-configured tokenizer directory may run remote
            # tokenizer code; request model values are client-controlled.
            trust_remote_code = self._tokenizer_dir is not None
            if self._custom_tokenizer:
                from tensorrt_llm.tokenizer import load_custom_tokenizer

                self._tokenizers[model] = load_custom_tokenizer(
                    self._custom_tokenizer, model_path, trust_remote_code=trust_remote_code
                )
            else:
                from tensorrt_llm.tokenizer import load_hf_tokenizer

                # The loader a worker and the standalone renderer use, so the tokenizer's
                # init options (and so its fingerprint) are the same as theirs.
                tokenizer = load_hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
                if tokenizer is None:
                    raise ValueError(f"Could not load a tokenizer from {model_path!r}.")
                self._tokenizers[model] = tokenizer.tokenizer
        return self._tokenizers[model]

    def _prefix_token_cache(self, tokenizer) -> Optional[PrefixTokenCache]:
        """The prefix token cache of ``tokenizer`` with enable_tokenization_cache, else None."""
        if not self._enable_tokenization_cache:
            return None
        entry = self._prefix_token_caches.get(id(tokenizer))
        if entry is None or entry[0] is not tokenizer:
            entry = (tokenizer, create_prefix_token_cache(tokenizer))
            self._prefix_token_caches[id(tokenizer)] = entry
        return entry[1]

    def _encode_with_prefix_cache(self, rendered: str, key: int, tokenizer) -> list[int]:
        cache = self._tok_prefix_cache
        entry = cache.get(key)
        if entry is not None and rendered == entry[0]:
            ids = entry[1]
        else:
            prefix_token_cache = self._prefix_token_cache(tokenizer)
            if prefix_token_cache is not None:
                ids = prefix_token_cache.encode(tokenizer, rendered)
            else:
                # Tokenizing a suffix independently is not generally composable:
                # BPE/SentencePiece merges can cross the cached string boundary.
                ids = tokenizer.encode(rendered, add_special_tokens=False)
        cache[key] = (rendered, ids)
        cache.move_to_end(key)
        while len(cache) > 1024:
            cache.popitem(last=False)
        return ids

    def _get_model_type(self) -> Optional[str]:
        model_path = self._model_path or self._tokenizer_dir
        if model_path is None:
            return None
        if model_path not in self._model_types:
            try:
                self._model_types[model_path] = resolve_model_type_from_config(model_path)
            except (OSError, ValueError) as error:
                logger.warning(
                    "Unable to resolve model type from checkpoint config at %s: %s. "
                    "Set use_harmony explicitly if the checkpoint uses Harmony.",
                    model_path,
                    error,
                )
                self._model_types[model_path] = None
        return self._model_types[model_path]

    def _make_encode_rendered(self, request: ChatCompletionRequest):
        """Router-side tokenization with the exact-match prompt cache."""

        def encode_rendered(rendered: str, tokenizer: object) -> list[int]:
            key = hash(
                "".join(
                    str(
                        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                    )
                    for msg in request.messages[:2]
                )
            )
            return self._encode_with_prefix_cache(rendered, key, tokenizer)

        return encode_rendered

    def _length_chat_ids(self, request: ChatCompletionRequest) -> list[int]:
        """Routing ids from :func:`render_chat_for_length`; never forwarded to a worker.

        A request the request model rejects, or Harmony cannot convert, gets no ids. One the chat
        template rejects raises.
        """
        try:
            return render_chat_for_length(
                request,
                self._render_resources(request.model),
                encode_rendered=self._make_encode_rendered(request),
            ).token_ids
        except (UnsupportedRenderError, HarmonyConversionError) as error:
            logger.debug(f"Router has no length for the request ({error}); routing on no ids.")
            return []

    def _render_resources(self, model: str) -> RenderResources:
        """Rendering resources of this router for ``model``, built once."""
        cache = self.__dict__.setdefault("_render_resources_cache", {})
        resources = cache.get(model)
        if resources is None:
            try:
                model_type = self._get_model_type()
            except Exception as error:
                logger.warning(f"Router could not resolve the model type: {error}")
                model_type = None
            if model_type is None:
                # Not an intentional generic-text configuration: a model that renders through
                # an extension or Harmony would be rendered here without it, so its routing
                # ids are an estimate and its fingerprint will not match the workers'.
                logger.warning(
                    "Router has no resolvable model type (no checkpoint path or an unreadable "
                    "config); it renders as a generic text model. Models that need a serving "
                    "extension or Harmony route on estimated ids and never forward them."
                )
            use_harmony = uses_harmony_tokenization(
                use_harmony=self._use_harmony, model_type_resolver=self._get_model_type
            )
            resources = RenderResources.from_tokenizer(
                # Harmony renders token ids itself. Loading a tokenizer here would fail for a
                # client that sends the served name when no tokenizer directory is configured.
                None if use_harmony else self._get_tokenizer(model),
                model_type=model_type,
                use_harmony=use_harmony,
                custom_tokenizer=self._custom_tokenizer,
            )
            cache[model] = resources
        return resources

    def _write_back_decision(
        self, model: str, resolved_thinking: Optional[dict[str, bool]] = None
    ) -> "WriteBackDecision":
        """Whether prompt ids rendered here may replace the worker's own rendering.

        Only when every active worker reports the same rendering fingerprint as
        the router. A worker that reports none (an older version) or whose
        ``/server_info`` could not be fetched counts as a mismatch. Hashing the vocabulary is
        slow, so the router's own fingerprint is computed once per model.

        ``resolved_thinking`` is the reasoning mode the prompt carries, per parser that reads it
        off the prompt. A forwarded request has no prompt text for the worker to read it from,
        so ids go only to workers whose own parser does not need it.

        This runs on several threads at once (``_tokenize`` is called through
        ``asyncio.to_thread``), so the decision and the reasons for it are built from local
        values and returned, never kept on the instance.
        """
        published = self.__dict__.get("_published_render_fingerprints")
        servers = list(getattr(self, "_servers", None) or [])
        if published is not None:
            # A coordinator frontend: it never fetches /server_info, so it judges the
            # servers it was told about by the fingerprints the coordinator published.
            # A server the coordinator has not reported yet counts as a mismatch.
            remote = [published.get(server) for server in servers]
        else:
            # Every active worker, not only those whose /server_info was fetched: a failed
            # fetch leaves a worker active with no entry, and it must not be skipped.
            info = getattr(self, "_server_info", None) or {}
            remote = [(info.get(server) or {}).get("render_fingerprint") for server in servers]
        if not remote:
            return WriteBackDecision(allowed=False)
        cache = self.__dict__.setdefault("_render_fingerprint_cache", {})
        local = cache.get(model)
        if local is None:
            local = cache[model] = self._render_resources(model).fingerprint()
        mismatched: dict[str, str] = {}
        for server, fingerprint in zip(servers, remote):
            reason = fingerprint_mismatch(local, fingerprint)
            if reason is not None:
                mismatched[server] = reason
        if mismatched:
            return WriteBackDecision(allowed=False, mismatched=mismatched)
        needs_prompt_text = [
            server
            for server, fingerprint in zip(servers, remote)
            if worker_resolves_thinking(fingerprint, resolved_thinking or {})
        ]
        return WriteBackDecision(allowed=not needs_prompt_text, needs_prompt_text=needs_prompt_text)

    def _may_write_back_token_ids(
        self, model: str, resolved_thinking: Optional[dict[str, bool]] = None
    ) -> bool:
        return self._write_back_decision(model, resolved_thinking).allowed

    def set_published_render_fingerprints(self, fingerprints: dict[str, Optional[dict]]) -> None:
        """Worker fingerprints (summaries) a disagg coordinator published, by server."""
        self._published_render_fingerprints = dict(fingerprints)

    def _tokenize_chat(self, request: ChatCompletionRequest) -> list[int]:
        """Render a chat request with the shared core and decide on write-back."""
        if request.prompt_token_ids is not None:
            return list(request.prompt_token_ids)
        try:
            rendered = render_chat(
                request,
                self._render_resources(request.model),
                tokenize=True,
                encode_rendered=self._make_encode_rendered(request),
            )
        except UnrenderableMessagesError as error:
            # The exact render spent the request's message iterators; the worker has the raw body.
            logger.debug(f"Router could not render the messages ({error}); routing on no ids.")
            return []
        except (UnsupportedRenderError, ValueError) as error:
            # Media, an unsupported model, or an extension rejection: route on its text.
            logger.debug(f"Router could not render the request exactly ({error}).")
            return self._length_chat_ids(request)
        except HarmonyConversionError as error:
            # The router has no tokenizer for Harmony to fall back to; the worker answers it.
            logger.debug(f"Router could not convert the messages ({error}); routing on no ids.")
            return []
        # A forwarded request is an ordinary chat request carrying prompt_token_ids, so the
        # worker cannot see the rendered text and decides nothing from it: usage adjustments
        # (kimi_k3) and reasoning modes read off the prompt are lost. Ids whose serving
        # semantics cannot be preserved that way are not forwarded; the worker renders them.
        context = PreparedContext.model_validate(rendered.context)
        request_reason = None
        if getattr(request.tool_choice, "function", None) is not None:
            # A named function makes the worker append a forced tool-call prefix to the prompt
            # it renders, and the worker rejects such a request together with prompt_token_ids.
            request_reason = "a named tool_choice is rendered by the worker"
        elif context.prompt_tokens_excluded_from_usage:
            request_reason = "the worker adjusts the reported prompt usage from the prompt text"
        decision = (
            None
            if request_reason is not None
            else self._write_back_decision(request.model, context.resolved_thinking)
        )
        if decision is not None and decision.allowed:
            request.prompt_token_ids = rendered.token_ids
            self._last_fallback_reason = None
            return rendered.token_ids
        blockers = decision.mismatched if decision is not None else {}
        thinking_blockers = decision.needs_prompt_text if decision is not None else []
        if request_reason is not None:
            logger.debug(f"Router rendered prompt ids are not forwarded: {request_reason}.")
        elif not blockers and thinking_blockers:
            logger.debug(
                "Router rendered prompt ids are not forwarded: the reasoning parser of "
                f"{', '.join(thinking_blockers)} reads the reasoning mode off the prompt text."
            )
        else:
            reason = (
                ", ".join(f"{server} ({why})" for server, why in blockers.items())
                if blockers
                else "no workers are known yet"
            )
            # The state of the workers, not of this request: reported again only when it changes.
            if reason != getattr(self, "_last_fallback_reason", None):
                self._last_fallback_reason = reason
                logger.warning(
                    f"Router rendered prompt ids are not forwarded to workers: {reason}. "
                    "Workers render the prompt themselves; routing is based on the router's ids."
                )
        self._render_fallbacks = getattr(self, "_render_fallbacks", 0) + 1
        return rendered.token_ids

    def _tokenize(self, request: OpenAIRequest) -> list[list[int]]:
        if isinstance(request, ResponsesRequest):
            return [self._tokenize_responses_request(request)]

        # Handle ChatCompletionRequest (has messages, not prompt)
        if isinstance(request, ChatCompletionRequest):
            return [self._tokenize_chat(request)]

        # Handle CompletionRequest (has prompt)
        prompts = request.prompt
        if isinstance(prompts, list) and not prompts:
            return [prompts]
        if isinstance(prompts, list) and isinstance(prompts[0], list):
            return prompts
        elif isinstance(prompts, list) and isinstance(prompts[0], int):
            return [prompts]
        elif isinstance(prompts, str):
            prompts = [prompts]
        else:
            assert isinstance(prompts, list) and isinstance(prompts[0], str)

        tokenizer = self._get_tokenizer(request.model)
        token_lists = [tokenizer(prompt)["input_ids"] for prompt in prompts]
        # Replace string prompts with token IDs so the worker server
        # skips re-tokenization
        request.prompt = token_lists if len(token_lists) > 1 else token_lists[0]
        return token_lists

    def _tokenize_responses_request(self, request: ResponsesRequest) -> list[int]:
        """Return the token ids of the prompt a worker builds for a Responses request.

        The ids only drive routing and are not stored on the request, so the
        context worker still renders the prompt itself.
        """
        token_ids = request.relayed_prompt_token_ids()
        if token_ids is not None:
            return token_ids
        from tensorrt_llm.serve import responses_utils

        # A disaggregated server rejects previous_response_id, so there is no
        # stored history: the input is the whole conversation.
        if uses_harmony_tokenization(
            use_harmony=self._use_harmony, model_type_resolver=self._get_model_type
        ):
            return responses_utils._render_for_completion(
                responses_utils._construct_harmony_messages(request, None)
            )
        tools = responses_utils._get_chat_completion_function_tools(request.tools)
        # As on the worker, developer messages render as system for a chat
        # template that has no developer role.
        messages = responses_utils._render_developer_as_system(
            responses_utils.chat_messages_from_responses_input(request, []),
            self._get_tokenizer(request.model),
            None,
            [tool.model_dump() for tool in tools],
        )
        chat_request = ChatCompletionRequest.model_construct(
            model=request.model,
            messages=messages,
            tools=tools,
            chat_template_kwargs=responses_utils.reasoning_chat_template_kwargs(request),
            injected_chat_template_kwargs=list(
                responses_utils.reasoning_injected_chat_template_keys(request)
            ),
        )
        return self._tokenize(chat_request)[0]

    def _compute_block_hashes(
        self,
        token_lists: list[list[int]],
        hash_algo: str = KV_CACHE_HASH_ALGO_DEFAULT,
        cache_salt: Optional[str] = None,
    ) -> list[list[BlockHash]]:
        if hash_algo == KV_CACHE_HASH_ALGO_V1:
            block_hasher = block_key_hasher
            cache_salt_id = (
                None if cache_salt is None else KVCacheEventManager._hash_cache_salt(cache_salt)
            )
        elif hash_algo in (KV_CACHE_HASH_ALGO_V2, KV_CACHE_HASH_ALGO_V2_SHA256_64):
            # V2 keys chain from a reuse-scope root, which is exactly what
            # sequence_to_blockchain_keys yields; its first pair is the root itself, so
            # skip it. In KvCacheManager the last token is not part of any block key.
            reuse_scope = ReuseScope(salt=cache_salt)
            to_hash = truncate_sha256_hash_to_int64
            if hash_algo == KV_CACHE_HASH_ALGO_V2:

                def to_hash(key: bytes) -> BlockHash:
                    return key.hex()

            block_hashes: list[list[BlockHash]] = []
            for token_list in token_lists:
                keys = sequence_to_blockchain_keys(
                    self._tokens_per_block, reuse_scope, token_list[:-1]
                )
                next(keys, None)  # the root key labels no block
                block_hashes.append([to_hash(key) for _, key in keys])
            return block_hashes
        else:
            raise ValueError(f"Unsupported KV cache hash algorithm: {hash_algo}")

        block_hashes: list[list[BlockHash]] = []
        for token_list in token_lists:
            hash_list = []
            # in KvCacheManager, the last token is not included in the block key
            for t in range(0, len(token_list) - 1, self._tokens_per_block):
                t_end = min(t + self._tokens_per_block, len(token_list) - 1)
                hash_list.append(
                    block_hasher(
                        token_list[t:t_end], None if t == 0 else hash_list[-1], cache_salt_id
                    )
                )
            block_hashes.append(hash_list)
        return block_hashes

    def _tokenize_and_compute_block_hashes(
        self, request: OpenAIRequest
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Synchronous tokenize + block-hash, combined for thread offload.

        Factored into one method so ``get_next_server`` can offload the whole
        CPU-bound step via ``asyncio.to_thread`` in a single call, keeping
        the orchestrator's asyncio event loop free to dispatch other
        requests in parallel.
        """
        token_lists = self._tokenize(request)
        block_hashes = self._compute_block_hashes(token_lists)
        return token_lists, block_hashes

    def _tokenize_and_compute_block_hashes_with_salt(
        self,
        request: OpenAIRequest,
        cache_salt: Optional[str] = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        token_lists = self._tokenize(request)
        block_hashes = self._compute_block_hashes(token_lists, cache_salt=cache_salt)
        return token_lists, block_hashes

    def _tokenize_and_compute_block_hashes_by_algo(
        self,
        request: OpenAIRequest,
        hash_algos: Iterable[str],
        cache_salt: Optional[str] = None,
    ) -> tuple[list[list[int]], dict[str, list[list[BlockHash]]]]:
        """Synchronous tokenize + per-algorithm block hashes for thread offload."""
        token_lists = self._tokenize(request)
        return token_lists, {
            hash_algo: self._compute_block_hashes(token_lists, hash_algo, cache_salt=cache_salt)
            for hash_algo in set(hash_algos)
        }

    @staticmethod
    def _text_to_int_sequences(texts: list[str]) -> list[list[int]]:
        """Convert text strings to lists of unicode code points.

        Usable as input to ``_compute_block_hashes``.
        """
        return [[ord(c) for c in text] for text in texts]

    @staticmethod
    def _get_request_cache_salt(request: OpenAIRequest) -> Optional[str]:
        return getattr(request, "cache_salt", None)
