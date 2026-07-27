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
from typing import Iterable, List, Optional, Union

from tensorrt_llm.bindings.internal.batch_manager import BlockKey as _NativeBlockKey
from tensorrt_llm.bindings.internal.batch_manager import BlockKeyHasher as _NativeBlockKeyHasher
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import kv_cache_hash
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import Block as V2Block
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import ReuseScope
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import RootBlock as V2RootBlock
from tensorrt_llm.serve.chat_tokenization import (
    resolve_model_type_from_config,
    tokenize_chat_request_for_serving,
)
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest

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
get_cache_salt_id = kv_cache_hash.get_cache_salt_id
hash_v1_block_key = kv_cache_hash.hash_v1_block_key
truncate_sha256_hash_to_int64 = kv_cache_hash.truncate_sha256_hash_to_int64

OpenAIRequest = Union[CompletionRequest, ChatCompletionRequest]
BlockHash = Union[int, str]

__all__ = [
    "KV_CACHE_HASH_ALGO_DEFAULT",
    "KV_CACHE_HASH_ALGO_V1",
    "KV_CACHE_HASH_ALGO_V2",
    "KV_CACHE_HASH_ALGO_V2_SHA256_64",
    "KV_CACHE_HASH_ALGOS",
    "get_cache_salt_id",
    "hash_v1_block_key",
    "truncate_sha256_hash_to_int64",
    "OpenAIRequest",
    "BlockHash",
    "get_request_num_tokens",
    "block_key_hasher",
    "v2_sha256_block_hasher",
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


def get_request_num_tokens(request: Optional[OpenAIRequest]) -> int:
    if request is None:
        return 0

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


def v2_sha256_block_hasher(
    token_ids: list[int], parent_hash: Optional[str] = None, cache_salt_id: Optional[int] = None
) -> str:
    parent_key = (
        V2RootBlock.make_key(ReuseScope(salt=cache_salt_id))
        if parent_hash is None
        else bytes.fromhex(parent_hash)
    )
    return V2Block.make_key(parent_key, token_ids).hex()


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
        logger.info(
            f"BlockHashMixin: tokens_per_block={self._tokens_per_block}"
            f"{' (auto, adopts worker)' if self._tpb_auto else ''}"
            f", custom_tokenizer={self._custom_tokenizer}"
            f", model_path={self._model_path}"
            f", use_harmony={self._use_harmony}"
        )

    def _get_tokenizer(self, model: str):
        if model not in self._tokenizers:
            model_path = self._tokenizer_dir or model
            if self._custom_tokenizer:
                from tensorrt_llm.tokenizer import load_custom_tokenizer

                self._tokenizers[model] = load_custom_tokenizer(self._custom_tokenizer, model_path)
            else:
                from tensorrt_llm.tokenizer import TransformersTokenizer

                tokenizer = TransformersTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self._tokenizers[model] = tokenizer.tokenizer
        return self._tokenizers[model]

    def _encode_with_prefix_cache(self, rendered: str, key: int, tokenizer) -> list[int]:
        cache = getattr(self, "_tok_prefix_cache", None)
        if cache is None:
            cache = self._tok_prefix_cache = OrderedDict()
        entry = cache.get(key)
        if entry is not None and rendered == entry[0]:
            ids = entry[1]
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

    def _tokenize(self, request: OpenAIRequest) -> list[list[int]]:
        # Handle ChatCompletionRequest (has messages, not prompt)
        if isinstance(request, ChatCompletionRequest):

            def tokenizer_factory() -> object:
                return self._get_tokenizer(request.model)

            def encode_rendered(rendered: str, tokenizer: object) -> list[int]:
                key = hash(
                    "".join(
                        str(
                            msg.get("content")
                            if isinstance(msg, dict)
                            else getattr(msg, "content", "")
                        )
                        for msg in request.messages[:2]
                    )
                )
                return self._encode_with_prefix_cache(rendered, key, tokenizer)

            result = tokenize_chat_request_for_serving(
                request,
                tokenizer_factory=tokenizer_factory,
                encode_rendered=encode_rendered,
                use_harmony=self._use_harmony,
                model_type_resolver=self._get_model_type,
                set_prompt_token_ids=True,
            )
            return [result]

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

    def _compute_block_hashes(
        self,
        token_lists: list[list[int]],
        hash_algo: str = KV_CACHE_HASH_ALGO_DEFAULT,
        cache_salt_id: Optional[int] = None,
    ) -> list[list[BlockHash]]:
        if hash_algo == KV_CACHE_HASH_ALGO_V1:
            block_hasher = block_key_hasher
        elif hash_algo == KV_CACHE_HASH_ALGO_V2:
            block_hasher = v2_sha256_block_hasher
        elif hash_algo == KV_CACHE_HASH_ALGO_V2_SHA256_64:
            reuse_scope = ReuseScope(salt=cache_salt_id)
            block_hashes: list[list[BlockHash]] = []
            for token_list in token_lists:
                hash_list = []
                parent_key = V2RootBlock.make_key(reuse_scope)
                for t in range(0, len(token_list) - 1, self._tokens_per_block):
                    t_end = min(t + self._tokens_per_block, len(token_list) - 1)
                    parent_key = V2Block.make_key(parent_key, token_list[t:t_end])
                    hash_list.append(truncate_sha256_hash_to_int64(parent_key))
                block_hashes.append(hash_list)
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
        cache_salt_id: Optional[int] = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        token_lists = self._tokenize(request)
        block_hashes = self._compute_block_hashes(token_lists, cache_salt_id=cache_salt_id)
        return token_lists, block_hashes

    def _tokenize_and_compute_block_hashes_by_algo(
        self,
        request: OpenAIRequest,
        hash_algos: Iterable[str],
        cache_salt_id: Optional[int] = None,
    ) -> tuple[list[list[int]], dict[str, list[list[BlockHash]]]]:
        """Synchronous tokenize + per-algorithm block hashes for thread offload."""
        token_lists = self._tokenize(request)
        return token_lists, {
            hash_algo: self._compute_block_hashes(
                token_lists, hash_algo, cache_salt_id=cache_salt_id
            )
            for hash_algo in set(hash_algos)
        }

    @staticmethod
    def _text_to_int_sequences(texts: list[str]) -> list[list[int]]:
        """Convert text strings to lists of unicode code points.

        Usable as input to ``_compute_block_hashes``.
        """
        return [[ord(c) for c in text] for text in texts]

    @staticmethod
    def _get_request_cache_salt_id(request: OpenAIRequest) -> Optional[int]:
        cache_salt = getattr(request, "cache_salt", None)
        return None if cache_salt is None else get_cache_salt_id(cache_salt)
