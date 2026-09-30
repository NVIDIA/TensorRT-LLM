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
"""
Engram module implementation for TensorRT-LLM PyTorch backend.

The Engram module provides n-gram based hash embeddings that augment
transformer hidden states with local context information.

All operations run on device:
  - Hash IDs (GPU) → embedding → flatten → linear projections (GEMMs)
    → normed keys + projected value
  - GPU main stream forward: SDP gating + short conv
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
from tokenizers import Regex, normalizers
from torch import nn
from transformers import AutoTokenizer

from tensorrt_llm._torch.host_staging import copy_host_to_device
from tensorrt_llm._utils import prefer_pinned

from ._decode_hash import _decode_hash_kernel

# Default Engram embedding vocabulary size per n-gram level.
# Derived from the DeepSeek-V3 tokenizer vocab size (129280) scaled by 5.
_DEFAULT_ENGRAM_VOCAB_SIZE = 129280 * 5


@dataclass
class EngramConfig:
    """Configuration for the Engram module.

    Attributes:
        tokenizer_name_or_path: Path or name of the HuggingFace tokenizer.
        engram_vocab_size: List of vocabulary sizes for each n-gram level (2-gram, 3-gram, etc.).
        max_ngram_size: Maximum n-gram size to compute hashes for.
        n_embed_per_ngram: Embedding dimension for each n-gram.
        n_head_per_ngram: Number of attention heads per n-gram.
        layer_ids: List of layer indices where Engram modules are applied.
        pad_id: Token ID used for padding.
        seed: Random seed for hash multiplier generation.
        kernel_size: Kernel size for the short convolution.
        hidden_size: Hidden dimension of the backbone model.
        hc_mult: Hyper-connection multiplier (number of residual streams).
        norm_eps: Epsilon for RMSNorm.
        dtype: Data type for model parameters (e.g., torch.float32, torch.bfloat16).
    """

    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(
        default_factory=lambda: [_DEFAULT_ENGRAM_VOCAB_SIZE, _DEFAULT_ENGRAM_VOCAB_SIZE]
    )
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    hidden_size: int = 1024
    hc_mult: int = 4
    norm_eps: float = 1e-5
    dtype: Optional[torch.dtype] = None


class CompressedTokenizer:
    """Tokenizer wrapper that normalizes and compresses tokens.

    This class builds a lookup table mapping original token IDs to
    normalized/compressed token IDs, reducing vocabulary size for
    more efficient n-gram hashing.

    If a pre-built ``lookup_table`` (torch.Tensor of shape [vocab_size])
    and ``num_new_token`` count are provided, the tokenizer download is
    skipped entirely — useful to avoid cold-start network fetches.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        lookup_table: Optional[torch.Tensor] = None,
        num_new_token: Optional[int] = None,
    ):
        if lookup_table is not None and num_new_token is not None:
            self.lookup_table = lookup_table
            self.num_new_token = num_new_token
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self) -> int:
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = torch.tensor([old2new[tid] for tid in range(vocab_size)], dtype=torch.long)

        return lookup, len(new_tokens)

    def _compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        ids = input_ids.long()
        # Preserve negative padding IDs while compressing valid token IDs.
        vocab_size = len(self.lookup_table)
        compressed = self.lookup_table[ids.clamp(0, vocab_size - 1)]
        return torch.where(ids < 0, ids, compressed)

    def __call__(self, input_ids):
        return self._compress(input_ids)


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test for n < 3.3e24."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Write n-1 as 2^r * d
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Witnesses sufficient for n < 3.3e24
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set) -> int:
    """Find the next prime number greater than start that is not in seen_primes."""
    candidate = start + 1
    while True:
        if _is_prime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Computes n-gram hash indices for embedding lookup.

    This class generates per-layer, per-head hash mappings for n-grams,
    using prime moduli to reduce hash collisions across heads.
    """

    def __init__(
        self,
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_name_or_path: str,
        pad_id: int,
        seed: int,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id].item())

        max_long = torch.iinfo(torch.long).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers: Dict[int, torch.Tensor] = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            # Use numpy RNG for deterministic reproducibility with existing
            # checkpoints — torch.Generator uses a different algorithm and
            # would produce incompatible multipliers for the same seed.
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = torch.tensor(r * 2 + 1, dtype=torch.long)
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

    def _calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = _find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        x = input_ids.long()
        (T,) = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> torch.Tensor:
            if k == 0:
                return x
            return torch.nn.functional.pad(x, (k, 0), value=self.pad_id)[:T]

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes: List[torch.Tensor] = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                all_hashes.append(mix % mod)

        return torch.stack(all_hashes, dim=1)

    def hash(self, input_ids) -> Dict[int, torch.Tensor]:
        """Compute hash indices for all configured layers.

        Args:
            input_ids: Token IDs of shape ``[T]``.

        Returns:
            Dictionary mapping layer_id to hash indices of shape ``[T, num_heads]``.
        """
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

    def hash_single_layer(self, input_ids, layer_id: int) -> torch.Tensor:
        """Compute hash indices for a single layer.

        Args:
            input_ids: Token IDs of shape ``[T]`` (already compressed or raw).
            layer_id: The layer to compute hashes for.

        Returns:
            Hash indices of shape ``[T, num_heads]``.
        """
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return self._get_ngram_hashes(input_ids, layer_id=layer_id)


class EngramHashProvider:
    """Computes and caches n-gram hash indices for all Engram layers.

    Hash computation runs on GPU using PyTorch and fused decode kernels. CPU-side
    ``NgramHashMapping`` is used only at init to derive constants
    (lookup table, multipliers, moduli) and then discarded.

    Usage:
        # At model initialization
        hash_provider = EngramHashProvider(config)

        # At each forward pass
        hash_cache = hash_provider.compute_hashes(input_ids)

        # In each Engram layer
        precomputed = engram_layer.precompute(hash_cache[layer_id])
        output = engram_layer(hidden_states, precomputed=precomputed)
    """

    def __init__(self, config: EngramConfig):
        self.config = config

        # Use NgramHashMapping only to derive constants, then discard it.
        hash_mapping = NgramHashMapping(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_embed_per_ngram=config.n_embed_per_ngram,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.layer_ids,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            pad_id=config.pad_id,
            seed=config.seed,
        )

        # Store vocab sizes directly (needed by Engram layer init).
        self._vocab_size_across_layers = hash_mapping.vocab_size_across_layers

        # GPU tensors for on-device hash computation.
        # Created on CPU and lazily moved to GPU on first use.
        self._lookup_table = hash_mapping.compressed_tokenizer.lookup_table.clone()
        self._pad_id = hash_mapping.pad_id

        self._multipliers: Dict[int, torch.Tensor] = {}
        self._modules: Dict[int, List[torch.Tensor]] = {}
        # The same moduli as plain Python ints. They are static, so reading them
        # out of the device tensors above with `.item()` inside the per-head hash
        # loop costs one host sync per (layer, n-gram, head) -- 48 per forward on
        # V4.1-Flash -- for values known at construction time.
        self._moduli_ints: Dict[int, List[List[int]]] = {}
        for layer_id in config.layer_ids:
            self._multipliers[layer_id] = hash_mapping.layer_multipliers[layer_id].clone()
            self._modules[layer_id] = [
                torch.tensor(ngram_head_sizes, dtype=torch.long)
                for ngram_head_sizes in hash_mapping.vocab_size_across_layers[layer_id]
            ]
            self._moduli_ints[layer_id] = [
                [int(size) for size in ngram_head_sizes]
                for ngram_head_sizes in hash_mapping.vocab_size_across_layers[layer_id]
            ]

        self._device: Optional[torch.device] = None

        # Retain per-shape buffers so captured addresses survive shape changes;
        # _cached_hashes selects the most recently computed shape.
        self._cached_hashes: Optional[Dict[int, torch.Tensor]] = None
        self._cached_hashes_store: Dict[tuple, Dict[int, torch.Tensor]] = {}

        # Per-request history of compressed token ids, so that an n-gram at a
        # decode step can reach the tokens that preceded it. Without this the
        # look-back can only see the tokens inside the current forward, which is
        # correct for a prefill and wrong for every generation step: a decode
        # forward carries exactly one token per sequence, so every n-gram would
        # degenerate to (pad, ..., pad, tok). Mirrors ``NgramHashState.cache`` in
        # the reference implementation, which keeps the same per-sequence buffer
        # for the same reason.
        #
        # Rows are grown on demand rather than sized from `max_batch_size` up
        # front: the column count is `max_seq_len`, so a fixed allocation costs
        # hundreds of MiB on a long-context config that may never use it.
        self._history: Optional[torch.Tensor] = None
        self._history_row_of: Dict[int, int] = {}
        self._history_free_rows: List[int] = []
        self._pending_history_seeds: dict[
            int, tuple[int, tuple[int, ...], tuple[bool, ...] | None]
        ] = {}
        self._history_host_staging: dict = {}
        self._decode_row_map: tuple[tuple, torch.Tensor] | None = None

    def _ensure_on_device(self, device: torch.device):
        """Move hash tensors to the specified device (lazy, once)."""
        if self._device == device:
            return
        self._lookup_table = self._lookup_table.to(device)
        for layer_id in list(self._multipliers.keys()):
            self._multipliers[layer_id] = self._multipliers[layer_id].to(device)
            self._modules[layer_id] = [m.to(device) for m in self._modules[layer_id]]
        self._device = device

    # Excluded image tokens and unwritten history stop n-grams across forwards.
    _DEAD = -1

    def queue_history_seed(
        self,
        request_id: int,
        start_position: int,
        token_ids: list[int],
        token_mask: list[bool] | None = None,
    ) -> None:
        """Stage a new request's raw-token lookback before its first GEN forward.

        Only the final ``max_ngram_size - 1`` prompt tokens are needed. Delay
        device writes until the complete forward batch has reserved its rows;
        reserving just the new requests could evict an existing batch member.
        False ``token_mask`` entries mark image boundaries.
        """
        if start_position < 0 or len(token_ids) >= self.config.max_ngram_size:
            raise ValueError("Engram history seeds must be a nonnegative, bounded lookback")
        if token_mask is not None and len(token_mask) != len(token_ids):
            raise ValueError("Engram history token_mask must have one entry per token")
        self.release_request_state(request_id)
        self._pending_history_seeds[request_id] = (
            start_position,
            tuple(token_ids),
            tuple(token_mask) if token_mask is not None else None,
        )

    def release_request_state(self, request_id: int) -> None:
        """Discard pending and resident history ownership on completion or cancellation."""
        self._pending_history_seeds.pop(request_id, None)
        row = self._history_row_of.pop(request_id, None)
        if row is not None:
            self._history_free_rows.append(row)

    def _upload_history_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(value, device=self._history.device)
        copy_host_to_device(self._history_host_staging, (key, result.device), result, value)
        return result

    def _write_history(
        self,
        slots: list[int],
        positions: list[int],
        raw: list[int],
        text_mask: list[bool],
    ) -> None:
        if not raw:
            return
        # Upload the small lookback batch once. The pinned ring retains each
        # source until its DMA completes, including when the host runs ahead.
        fields = [slots, positions, raw]
        if text_mask:
            fields.append(text_mask)
        packed = self._upload_history_tensor(
            "lookback", torch.tensor(fields, dtype=torch.long, device="cpu")
        )
        row_ids, position_ids, raw_ids = packed[:3].unbind(0)
        compressed = self._lookup_table[raw_ids]
        if text_mask:
            compressed = torch.where(packed[3].bool(), compressed, self._DEAD)
        self._history[row_ids, position_ids] = compressed.to(torch.int32)

    def _consume_history_seeds(self, request_ids: list[int], rows: list[int]) -> None:
        slots, positions, raw, text_mask, consumed = [], [], [], [], []
        for request_id, row in zip(request_ids, rows, strict=True):
            seed = self._pending_history_seeds.get(request_id)
            if seed is None:
                continue
            start, token_ids, token_mask = seed
            end = start + len(token_ids)
            if end > self._history.shape[1]:
                raise ValueError("Engram history seed exceeds the configured sequence length")
            slots.extend([row] * len(token_ids))
            positions.extend(range(start, end))
            raw.extend(max(0, min(token, self._lookup_table.shape[0] - 1)) for token in token_ids)
            text_mask.extend(token_mask if token_mask is not None else [True] * len(token_ids))
            consumed.append(request_id)
        self._write_history(slots, positions, raw, text_mask)
        for request_id in consumed:
            del self._pending_history_seeds[request_id]

    def _reserve_history_rows(
        self,
        request_ids: List[int],
        min_columns: int,
        device: torch.device,
    ) -> List[int]:
        """Reserve active request rows, recycling inactive history when needed.

        Resumed contexts restore their lookback before hashing; unwritten cells
        remain ``_DEAD`` so recycled tokens cannot enter another request's hash.
        """
        rows = self._history.shape[0] if self._history is not None else 0
        columns = self._history.shape[1] if self._history is not None else 0
        needed_rows = max(rows, 1)
        while needed_rows < len(request_ids):
            needed_rows *= 2
        needed_columns = max(columns, 1)
        while needed_columns < min_columns:
            needed_columns *= 2

        if self._history is None or needed_rows > rows or needed_columns > columns:
            grown = torch.full(
                (needed_rows, needed_columns),
                self._DEAD,
                dtype=torch.int32,
                device=device,
            )
            if self._history is not None and self._history.device == device:
                grown[:rows, :columns] = self._history
            else:
                # A device change invalidates the row map along with the buffer.
                self._history_row_of = {}
            self._history = grown
            self._history_free_rows = [
                row for row in range(needed_rows) if row not in set(self._history_row_of.values())
            ]

        live = set(request_ids)
        assigned: List[int] = []
        for request_id in request_ids:
            row = self._history_row_of.get(request_id)
            if row is None:
                if not self._history_free_rows:
                    # Evict any row held by a request outside this batch. One is
                    # guaranteed to exist: the buffer has at least as many rows
                    # as the batch has requests.
                    stale = [rid for rid in self._history_row_of if rid not in live]
                    assert stale, (
                        "engram history has no free row and every row belongs to a "
                        f"request in this batch (rows={self._history.shape[0]}, "
                        f"batch={len(request_ids)})"
                    )
                    for rid in stale:
                        self._history_free_rows.append(self._history_row_of.pop(rid))
                row = self._history_free_rows.pop()
                # A recycled row still holds the previous request's tokens, which
                # would otherwise be hashed into this one's first n-grams.
                self._history[row].fill_(self._DEAD)
                self._history_row_of[request_id] = row
            assigned.append(row)
        return assigned

    def seed_context_history(
        self,
        request_ids: List[int],
        context_prefixes: Dict[int, Tuple[int, List[int]]],
        *,
        max_seq_len: int,
        device: torch.device,
        token_masks: dict[int, list[bool]] | None = None,
    ) -> None:
        """Restore bounded raw-token lookback before a context chunk is hashed.

        Each entry is ``request_id: (first_query_position, preceding_raw_ids)``.
        These tokens populate history only and never add attention/query rows.
        Reserving all active request IDs prevents evicting a generation peer.
        Optional per-request text masks preserve image boundaries.
        """
        if not context_prefixes:
            return
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Seed Engram context history before CUDA Graph capture")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Engram history requires unique active request IDs")
        lookback = self.config.max_ngram_size - 1
        for request_id, (start, tokens) in context_prefixes.items():
            if request_id not in request_ids or not 0 <= start < max_seq_len:
                raise ValueError("Engram lookback does not belong to the active context")
            if len(tokens) != min(start, lookback):
                raise ValueError("Engram lookback must contain every available preceding token")
            token_mask = token_masks.get(request_id) if token_masks is not None else None
            if token_mask is not None and len(token_mask) != len(tokens):
                raise ValueError("Engram history token_mask must have one entry per token")
            if any(token < 0 or token >= self._lookup_table.numel() for token in tokens):
                raise ValueError("Engram lookback token is outside the tokenizer vocabulary")
        self._ensure_on_device(device)
        rows = dict(zip(request_ids, self._reserve_history_rows(request_ids, max_seq_len, device)))
        slots, positions, raw, text_mask = [], [], [], []
        for request_id, (start, tokens) in context_prefixes.items():
            slots.extend([rows[request_id]] * len(tokens))
            positions.extend(range(start - len(tokens), start))
            raw.extend(tokens)
            if token_masks is not None:
                text_mask.extend(token_masks.get(request_id, [True] * len(tokens)))
        self._write_history(slots, positions, raw, text_mask)

    def _lookback_from_history(
        self,
        compressed: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lens_host: torch.Tensor,
        request_ids: List[int],
        max_seq_len: Optional[int],
    ) -> List[torch.Tensor]:
        """Return the shifted n-gram inputs, reading across forward boundaries.

        Writes this forward's compressed ids into each request's row at their
        absolute positions, then gathers the ``max_ngram_size`` look-back slots
        out of the row. This replaces left-padding the current forward's token
        window, which silently substitutes ``pad_id`` for real history.
        """
        device = compressed.device
        positions = position_ids.view(-1).long()
        num_tokens = compressed.shape[0]

        # Sizing from the runtime's sequence-length ceiling keeps this off the
        # critical path: reading the true maximum off `positions` would mean a
        # device-to-host sync on every forward, and the ceiling is known up
        # front. Only a caller that cannot supply it pays for the sync.
        min_columns = max_seq_len
        if min_columns is None:
            min_columns = int(positions.max().item()) + 1 if num_tokens else 1
        rows = self._reserve_history_rows(request_ids, min_columns, device)
        self._consume_history_seeds(request_ids, rows)
        # Expanded on the host and copied once, rather than by a device-side
        # `repeat_interleave`: the latter has to read the counts back to size its
        # output, which is a full sync on the critical path.
        row_of_token = self._upload_history_tensor(
            "token_rows",
            torch.repeat_interleave(
                torch.tensor(rows, dtype=torch.long, device="cpu"),
                seq_lens_host[: len(rows)].to(torch.long),
            ),
        )
        assert row_of_token.shape[0] == num_tokens, (
            f"engram history: seq_lens sum to {row_of_token.shape[0]} tokens but the "
            f"forward carries {num_tokens}"
        )

        self._history[row_of_token, positions] = compressed.to(torch.int32)

        shifts: List[torch.Tensor] = []
        # `blocked` accumulates: once a shorter look-back is unusable every longer
        # one is too, so a single dead cell truncates the n-gram rather than
        # leaving a hole in the middle of it. This is the reference's rule.
        blocked = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        for shift in range(self.config.max_ngram_size):
            source = self._history[row_of_token, (positions - shift).clamp_min(0)]
            blocked = blocked | (positions < shift) | (source == self._DEAD)
            shifts.append(
                torch.where(
                    blocked,
                    torch.full_like(source, self._pad_id),
                    source,
                ).to(compressed.dtype)
            )
        return shifts

    def _decode_history_rows(
        self, rows: list[int], lengths: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        # A stable batch needs no per-step DMA or host-buffer reuse wait. The
        # immutable map only runs on its allocation stream; a changed key gets
        # a new allocation, so pending readers cannot observe an overwrite.
        stream = torch.cuda.current_stream(device)
        key = (device, stream.cuda_stream, tuple(rows), lengths)
        if self._decode_row_map is not None and self._decode_row_map[0] == key:
            return self._decode_row_map[1]
        row_map = self._upload_history_tensor(
            "decode_rows", torch.tensor(rows, dtype=torch.long, device="cpu")
        )
        self._decode_row_map = (key, row_map)
        return row_map

    def _compute_decode_hashes(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        request_ids: list[int] | None,
        seq_lens_host: torch.Tensor | None,
        max_seq_len: int | None,
        token_mask: torch.Tensor | None,
        padded_num_tokens: int | None,
    ) -> dict[int, torch.Tensor] | None:
        # Packed contexts, speculative batches, and incomplete host metadata
        # keep the general path. Capture already returned its warmup buffers.
        # Unique rows fuse the scatter too; aliased rows retain Torch's scatter
        # before any hash reads, including repeated CUDA graph dummy requests.
        if (
            not input_ids.is_cuda
            or input_ids.ndim != 1
            or not input_ids.is_contiguous()
            or input_ids.dtype not in (torch.int32, torch.int64)
            or position_ids is None
            or position_ids.device != input_ids.device
            or not position_ids.is_contiguous()
            or position_ids.numel() != input_ids.numel()
            or position_ids.dtype not in (torch.int32, torch.int64)
            or request_ids is None
            or not request_ids
            or len(request_ids) != input_ids.numel()
            or seq_lens_host is None
            or seq_lens_host.device.type != "cpu"
            or seq_lens_host.ndim != 1
            or seq_lens_host.numel() < len(request_ids)
            or seq_lens_host.dtype not in (torch.int32, torch.int64)
            or type(max_seq_len) is not int
            or max_seq_len <= 0
            or self.config.max_ngram_size < 2
            or self.config.n_head_per_ngram < 1
            or not self.config.layer_ids
            or self._pad_id is None
            or (token_mask is not None and not token_mask.is_contiguous())
        ):
            return None
        lengths = tuple(seq_lens_host[: len(request_ids)].tolist())
        if any(length != 1 for length in lengths):
            return None

        # Constants can move independently of retained history and captured
        # outputs. Keep cross-device calls on the general path instead of
        # passing another device's storage to a raw Triton kernel.
        if self._history is not None and self._history.device != input_ids.device:
            return None
        num_tokens = input_ids.numel()
        num_output_tokens = padded_num_tokens if padded_num_tokens is not None else num_tokens
        num_heads = (self.config.max_ngram_size - 1) * self.config.n_head_per_ngram
        shape_key = ((num_output_tokens, num_heads),) * len(self.config.layer_ids)
        cached = self._cached_hashes_store.get(shape_key)
        if cached is not None and any(
            value.device != input_ids.device for value in cached.values()
        ):
            return None

        aliased_rows = len(set(request_ids)) != len(request_ids)
        if aliased_rows:
            # Preserve the general path's compression and int32 index_put
            # geometry/order. Repeated IDs may have different input tokens or
            # positions: hashing each token's register value would bypass the
            # shared history selected by that scatter. Equal-index writes keep
            # PyTorch's existing unspecified winner rather than inventing one.
            ids = input_ids.long().view(-1)
            compressed = self._lookup_table[ids.clamp(0, self._lookup_table.shape[0] - 1)]
            if token_mask is not None:
                compressed = torch.where(
                    token_mask.to(device=input_ids.device), compressed, self._DEAD
                )
            positions = position_ids.view(-1).long()

        rows = self._reserve_history_rows(request_ids, max_seq_len, input_ids.device)
        if not aliased_rows and len(set(rows)) != len(rows):
            return None
        # Row assignment and seed consumption must happen even on a map hit:
        # a released request can immediately reuse the same physical row.
        self._consume_history_seeds(request_ids, rows)
        row_map = self._decode_history_rows(rows, lengths, input_ids.device)
        if aliased_rows:
            self._history[row_map, positions] = compressed.to(torch.int32)
        elif token_mask is not None:
            token_mask = token_mask.to(device=input_ids.device)
        if cached is None:
            cached = {
                layer_id: torch.empty(
                    (num_output_tokens, num_heads), dtype=torch.long, device=input_ids.device
                )
                for layer_id in self.config.layer_ids
            }
            self._cached_hashes_store[shape_key] = cached
        for index, layer_id in enumerate(self.config.layer_ids):
            _decode_hash_kernel[(num_output_tokens,)](
                input_ids,
                position_ids,
                row_map,
                None if aliased_rows else token_mask,
                self._lookup_table,
                self._history,
                self._multipliers[layer_id],
                cached[layer_id],
                num_tokens,
                self._history.shape[1],
                VOCAB_SIZE=self._lookup_table.numel(),
                PAD_ID=self._pad_id,
                DEAD=self._DEAD,
                MODULI=tuple(tuple(moduli) for moduli in self._moduli_ints[layer_id]),
                N_HEAD=self.config.n_head_per_ngram,
                MAX_NGRAM=self.config.max_ngram_size,
                WRITE_HISTORY=index == 0 and not aliased_rows,
                READ_CURRENT_HISTORY=aliased_rows,
                HAS_MASK=token_mask is not None and not aliased_rows,
                BLOCK_HEAD=triton.next_power_of_2(self.config.n_head_per_ngram),
                num_warps=1,
            )
        self._cached_hashes = cached
        return cached

    def compute_hashes(
        self,
        input_ids: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        request_ids: Optional[List[int]] = None,
        seq_lens_host: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        token_mask: torch.Tensor | None = None,
        padded_num_tokens: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """Compute hash indices on GPU.

        Runs the compressed-tokenizer lookup, n-gram mixing, and modular
        hashing entirely on the device where ``input_ids`` resides.

        During CUDA graph capture, returns cached hashes from the warmup pass.

        Args:
            input_ids: Flattened token IDs of shape ``[T]`` on GPU.
                When the tensor contains multiple packed sequences, pass
                ``seq_lens`` so that n-gram hashing does not cross
                sequence boundaries.
            seq_lens: Optional 1-D tensor of per-sequence lengths.  When
                provided, shifted n-gram windows that would cross a
                sequence boundary are replaced with ``pad_id``.
            position_ids: Optional absolute position of each token, shape
                ``[T]``.  Together with ``request_ids`` this enables the
                per-request history buffer, which is what lets a generation
                step's n-grams reach the tokens that came before it.  Without
                both, the look-back is confined to this forward's tokens --
                correct for a whole-prompt prefill, wrong for decode.
            request_ids: Optional per-sequence request ids in the same order as
                ``seq_lens``, used to key the history buffer.
            token_mask: Optional ``[T]`` bool tensor, true for text. Image
                positions hash padding and block subsequent n-gram look-back.
            padded_num_tokens: Optional output row count for graph or collective
                padding. Only real input tokens update the request history.

        Returns:
            Dictionary mapping layer_id to hash indices as torch.Tensor
            of shape ``[T, num_heads]`` on the same device as input_ids.
        """
        if torch.cuda.is_current_stream_capturing():
            if self._cached_hashes is not None:
                return self._cached_hashes
            raise RuntimeError(
                "EngramHashProvider.compute_hashes() called during CUDA "
                "graph capture but no cached hashes available. "
                "Ensure warmup runs before capture."
            )

        device = input_ids.device
        self._ensure_on_device(device)

        T = input_ids.numel()
        if token_mask is not None and (token_mask.dtype != torch.bool or token_mask.shape != (T,)):
            raise ValueError("token_mask must be a bool tensor with one entry per token")
        if padded_num_tokens is not None and (
            type(padded_num_tokens) is not int or padded_num_tokens < T
        ):
            raise ValueError(
                "Engram padded_num_tokens must be an integer >= the input token count."
            )

        decoded = self._compute_decode_hashes(
            input_ids,
            position_ids,
            request_ids,
            seq_lens_host,
            max_seq_len,
            token_mask,
            padded_num_tokens,
        )
        if decoded is not None:
            return decoded

        # 1. Compress token ids via lookup table (1-D)
        ids = input_ids.long().view(-1)
        vocab_size = self._lookup_table.shape[0]
        compressed = self._lookup_table[ids.clamp(0, vocab_size - 1)]
        if token_mask is not None:
            compressed = torch.where(token_mask.to(device=device), compressed, self._DEAD)

        use_history = (
            position_ids is not None and request_ids is not None and seq_lens_host is not None
        )
        if use_history:
            # 2. Shifted versions read out of the per-request history, which
            # spans forwards. Absolute positions and per-request rows make the
            # sequence-boundary masking below unnecessary: each sequence owns a
            # row and its own position origin, so a look-back cannot reach
            # another sequence's tokens in the first place.
            shifts = self._lookback_from_history(
                compressed, position_ids, seq_lens_host, list(request_ids), max_seq_len
            )
        else:
            # 2. Pre-compute shifted versions (left-padded with pad_id)
            shifts = [compressed]
            for k in range(1, self.config.max_ngram_size):
                padded = torch.nn.functional.pad(compressed, (k, 0), value=self._pad_id)
                shifts.append(padded[:T])

        # 2b. When input_ids is a packed/flattened tensor containing multiple
        # sequences, mask out shifted positions that cross sequence boundaries
        # so n-gram hashes stay within each sequence.
        if not use_history and seq_lens is not None and seq_lens.numel() > 1:
            cum_lens = torch.cumsum(seq_lens, dim=0)
            seq_starts = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=cum_lens.dtype),
                    cum_lens[:-1],
                ]
            )
            positions = torch.arange(T, device=device)
            # Map each position to its owning sequence
            seq_idx = torch.searchsorted(cum_lens, positions, right=True)
            seq_idx = seq_idx.clamp(max=seq_lens.numel() - 1)
            pos_in_seq = positions - seq_starts[seq_idx]

            for k in range(1, self.config.max_ngram_size):
                boundary_mask = pos_in_seq < k  # [T]
                shifts[k][boundary_mask] = self._pad_id

        if not use_history and token_mask is not None:
            # Apply sequence boundaries first, so an image in the preceding
            # packed sequence cannot block this sequence's look-back.
            blocked = torch.zeros(T, dtype=torch.bool, device=device)
            for k, source in enumerate(shifts):
                blocked = blocked | (source == self._DEAD)
                shifts[k] = torch.where(blocked, self._pad_id, source)

        # 3. Compute hashes per layer
        result: Dict[int, torch.Tensor] = {}
        for layer_id in self.config.layer_ids:
            multipliers = self._multipliers[layer_id]
            all_hashes: List[torch.Tensor] = []

            for n in range(2, self.config.max_ngram_size + 1):
                n_gram_index = n - 2
                mix = shifts[0] * multipliers[0]
                for k in range(1, n):
                    mix = torch.bitwise_xor(mix, shifts[k] * multipliers[k])

                moduli = self._moduli_ints[layer_id][n_gram_index]
                for j in range(self.config.n_head_per_ngram):
                    head_hash = mix % moduli[j]
                    all_hashes.append(head_hash)

            result[layer_id] = torch.stack(all_hashes, dim=1)
            if padded_num_tokens is not None and padded_num_tokens > T:
                result[layer_id] = torch.nn.functional.pad(
                    result[layer_id], (0, 0, 0, padded_num_tokens - T)
                )

        # Update retained per-shape buffers in place to preserve captured addresses.
        shape_key = tuple(tuple(v.shape) for v in result.values())
        if shape_key in self._cached_hashes_store:
            cached = self._cached_hashes_store[shape_key]
            for layer_id, hashes in result.items():
                cached[layer_id].copy_(hashes)
            self._cached_hashes = cached
        else:
            self._cached_hashes_store[shape_key] = result
            self._cached_hashes = result
        return self._cached_hashes

    def refresh_captured_hashes(
        self,
        input_ids: torch.Tensor,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        request_ids: Optional[List[int]] = None,
        seq_lens_host: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        token_mask: torch.Tensor | None = None,
        padded_num_tokens: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """Refresh captured hash buffers before replay, after filling static inputs.

        Call outside capture: replay does not run Python or ``compute_hashes``.
        Reject new shapes, which allocate buffers while the graph reads old addresses.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "EngramHashProvider.refresh_captured_hashes() must not be called while the "
                "stream is capturing. It is the replay-path counterpart of the capture "
                "short-circuit in compute_hashes(), not a way around it."
            )

        known_shapes = set(self._cached_hashes_store)
        result = self.compute_hashes(
            input_ids,
            seq_lens=seq_lens,
            position_ids=position_ids,
            request_ids=request_ids,
            seq_lens_host=seq_lens_host,
            max_seq_len=max_seq_len,
            token_mask=token_mask,
            padded_num_tokens=padded_num_tokens,
        )
        shape_key = tuple(tuple(v.shape) for v in result.values())
        if shape_key not in known_shapes:
            raise RuntimeError(
                f"EngramHashProvider.refresh_captured_hashes() computed hashes of shape "
                f"{shape_key}, for which no captured graph is reading a buffer (known "
                f"shapes: {sorted(known_shapes)}). The recomputed values went into a fresh "
                "allocation, so the graph about to replay would still read stale hashes."
            )
        return result

    @property
    def layer_ids(self) -> List[int]:
        """Return the list of layer IDs that have Engram modules."""
        return self.config.layer_ids

    @property
    def vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        """Return the vocabulary sizes for each layer and head."""
        return self._vocab_size_across_layers


def _bucket_offsets(list_of_N: List[int]) -> torch.Tensor:
    """Return per-head base rows shared by the dense and FP8 table layouts."""
    offsets = [0]
    for n in list_of_N[:-1]:
        offsets.append(offsets[-1] + n)
    return torch.tensor(offsets, dtype=torch.long)


class _CudaArrayInterfaceView:
    """Expose a device pointer as bytes; NumPy typestr has no FP8 spelling."""

    def __init__(self, ptr: int, shape: Tuple[int, ...]):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "strides": None,
            "version": 3,
        }


def _uva_device_view(host_tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy CUDA view with the pinned tensor's shape and dtype.

    UVA maps pinned host memory at the same device address. The byte view
    requires contiguous single-byte storage to preserve element strides.
    """
    if host_tensor.device.type != "cpu":
        raise ValueError(
            f"A UVA device view requires a pinned CPU tensor, got {host_tensor.device}."
        )
    if not host_tensor.is_pinned():
        raise ValueError(
            "A UVA device view requires page-locked host memory; this tensor is "
            "ordinary pageable memory, whose host address is not valid on the device."
        )
    if not host_tensor.is_contiguous():
        raise ValueError("A UVA device view requires a contiguous tensor.")
    if host_tensor.element_size() != 1:
        raise ValueError(
            f"A UVA device view is restricted to single-byte dtypes, got "
            f"{host_tensor.dtype} ({host_tensor.element_size()} bytes)."
        )
    view = torch.as_tensor(
        _CudaArrayInterfaceView(host_tensor.data_ptr(), host_tensor.shape),
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    return view.view(host_tensor.dtype)


class MultiHeadEmbedding(nn.Module):
    """Multi-head embedding layer with per-head offset handling.

    Each head has its own embedding space, indexed by applying offsets
    to the input indices before a single embedding lookup.
    """

    def __init__(self, list_of_N: List[int], D: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        self.register_buffer("offsets", _bucket_offsets(list_of_N))

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for multi-head input indices.

        Args:
            input_ids: Indices of shape ``[T, num_heads]``.

        Returns:
            Embeddings of shape ``[T, num_heads, D]``.
        """
        shifted_input_ids = input_ids + self.offsets
        # Clamp to valid embedding range to prevent CUDA kernel OOB assertion.
        # head_hash values should be in [0, prime-1] by construction, but this
        # acts as a defensive guard in case of unexpected inputs.
        shifted_input_ids = shifted_input_ids.clamp(0, self.embedding.num_embeddings - 1)
        output = self.embedding(shifted_input_ids)
        return output


class ShortConv(nn.Module):
    """Short depthwise convolution with RMSNorm and SiLU activation.

    Applies a causal depthwise convolution across the sequence dimension
    with per-hyper-connection-stream normalization.

    Note: During token-by-token autoregressive generation (seq_len=1),
    the Conv1d sees only the current token plus zero padding — no ring
    buffer or conv state carries over from previous steps.  This means
    the short-range context capture is limited to prefill.  A proper
    conv state cache for generation is left as a future enhancement.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        self.hidden_size = hidden_size
        self.norm_eps = norm_eps

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            dtype=dtype,
        )

        # Stacked RMSNorm weight [hc_mult, D] — single vectorised norm
        # instead of hc_mult separate kernel launches.
        self.norm_weight = nn.Parameter(torch.ones(hc_mult, hidden_size, dtype=dtype))

        if self.activation:
            self.act_fn = nn.SiLU()

    def load_weights(self, weights):
        """Load weights, handling legacy ``norms.{i}.weight`` checkpoint keys.

        Legacy checkpoints store per-stream RMSNorm weights as
        ``norms.0.weight``, ``norms.1.weight``, …  This method stacks
        them into the single ``norm_weight`` parameter.
        """
        w = weights[0] if isinstance(weights, list) else weights
        # Check for legacy per-stream norm keys
        legacy_keys = [f"norms.{i}.weight" for i in range(self.hc_mult)]
        if legacy_keys[0] in w:
            stacked = torch.stack([w[k][:] for k in legacy_keys], dim=0)
            self.norm_weight.data.copy_(stacked)
            # Load remaining keys (e.g. conv.weight) via the generic path
            for n, p in self.named_parameters():
                if n == "norm_weight":
                    continue
                if n in w:
                    p.data.copy_(w[n][:])
        else:
            for n, p in self.named_parameters():
                if n in w:
                    p.data.copy_(w[n][:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply short convolution with normalization.

        Args:
            x: Input tensor of shape ``[T, HC_MULT, D]``.

        Returns:
            Output tensor of shape ``[T, HC_MULT, D]``.
        """
        T, G, C = x.shape

        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        x_normed = x * rms * self.norm_weight  # [hc_mult, D] broadcasts over [T, G, D]

        x_norm = x_normed.reshape(T, G * C)
        # Conv1d expects [N, C, L]; use N=1 so that L=T is the sequence dim.
        x_nct = x_norm.unsqueeze(0).transpose(1, 2)  # [1, G*C, T]
        y_nct = self.conv(x_nct)
        # Truncate to maintain causal masking
        y_nct = y_nct[..., :T]

        if self.activation:
            y_nct = self.act_fn(y_nct)
        y = y_nct.transpose(1, 2).squeeze(0).view(T, G, C).contiguous()

        return y


class Engram(nn.Module):
    """Engram module for n-gram based context augmentation.

    The Engram module computes n-gram hash embeddings from input tokens
    and uses gated attention to augment the hidden states of the model.

    All operations run on device. ``precompute()`` is called on a separate
    CUDA stream to overlap with other layers; the main stream performs
    only SDP gating + short conv.

    Known limitations:
      - **No TP sharding**: The embedding table is replicated on every rank.
        For large-scale deployments with TP > 1 this means redundant memory.
      - **No conv state for generation**: See ``ShortConv`` docstring.
    """

    def _make_multi_head_embedding(self, list_of_N: List[int], D: int) -> nn.Module:
        """Build the n-gram table.

        A hook rather than a class attribute because the alternative
        implementation (:class:`ShardedFp8MultiHeadEmbedding`) takes different
        constructor arguments, and because the table has to be built *instead of*
        the dense one, never alongside it: DeepSeek-V4.1's tables are 384 M rows,
        so materializing an ``nn.Embedding`` for them would try to allocate
        ~197 GiB before the subclass got a chance to replace it.
        """
        return MultiHeadEmbedding(list_of_N=list_of_N, D=D, dtype=self.config.dtype)

    def _make_short_conv(self) -> Optional[nn.Module]:
        """Build the depthwise convolution, or return None if there isn't one.

        DeepSeek-V4.1's Engram has no convolution at all (its checkpoint ships no
        conv weights), so its subclass returns None and overrides ``forward``.
        """
        config = self.config
        return ShortConv(
            hidden_size=config.hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
            hc_mult=config.hc_mult,
            norm_eps=config.norm_eps,
            dtype=config.dtype,
        )

    def _make_kv_projection(self, in_features: int, out_features: int) -> nn.Module:
        """Construct WKV without first allocating a replaced full-precision weight."""
        return nn.Linear(in_features, out_features, bias=False, dtype=self.config.dtype)

    def __init__(
        self,
        layer_id: int,
        config: EngramConfig,
        vocab_sizes_flat: Optional[List[int]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.stream = stream
        self.sync_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if stream is not None else None
        )

        # If vocab_sizes not provided, compute them (for standalone usage)
        if vocab_sizes_flat is None:
            hash_mapping = NgramHashMapping(
                engram_vocab_size=config.engram_vocab_size,
                max_ngram_size=config.max_ngram_size,
                n_embed_per_ngram=config.n_embed_per_ngram,
                n_head_per_ngram=config.n_head_per_ngram,
                layer_ids=config.layer_ids,
                tokenizer_name_or_path=config.tokenizer_name_or_path,
                pad_id=config.pad_id,
                seed=config.seed,
            )
            vocab_sizes_flat = [
                x for y in hash_mapping.vocab_size_across_layers[layer_id] for x in y
            ]
            self.hash_mapping = hash_mapping
        else:
            self.hash_mapping = None

        embed_dim_per_head = config.n_embed_per_ngram // config.n_head_per_ngram

        dtype = config.dtype

        # Kept so subclasses (and weight-loading checks) can see the per-head
        # bucket sizes without re-deriving them from the table's offsets.
        self.vocab_sizes_flat = list(vocab_sizes_flat)
        self.embed_dim_per_head = embed_dim_per_head

        self.multi_head_embedding = self._make_multi_head_embedding(
            vocab_sizes_flat, embed_dim_per_head
        )

        self.short_conv = self._make_short_conv()

        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        hc = config.hc_mult
        D = config.hidden_size

        # Fused projection: one GEMM produces value (1 head) + all keys (hc_mult heads).
        # Output layout: [value (D) | key_0 (D) | key_1 (D) | ... | key_{hc-1} (D)]
        self.kv_proj = self._make_kv_projection(engram_hidden_size, (1 + hc) * D)

        # Per-HC-stream RMSNorm weights for keys and queries, stored as
        # stacked parameters so we can apply a single vectorised norm
        # instead of hc_mult separate kernel launches.
        self.key_norm_weight = nn.Parameter(torch.ones(hc, D, dtype=dtype))
        self.query_norm_weight = nn.Parameter(torch.ones(hc, D, dtype=dtype))
        self.norm_eps = config.norm_eps

        # CUDA graph capture cache
        self._cached_embeddings: Optional[torch.Tensor] = None

    def precompute(
        self,
        hash_indices: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Pre-compute embeddings from hash indices.

        With an auxiliary stream, the caller must wait on ``sync_event`` before
        consuming the result. Otherwise, lookup runs on the current stream.

        Args:
            hash_indices: Hash indices of shape ``[T, num_heads]`` on GPU.
            dtype: Cast embeddings to this dtype after lookup.
                If None, uses the embedding's native dtype.
            all_rank_num_tokens: Unused for replicated tables; accepted for subclasses.

        Returns:
            Embedding tensor of shape ``[T, num_heads * embed_dim_per_head]``.
        """
        if self.stream is not None:
            # Wait for caller-stream hash production before lookup.
            caller_stream = torch.cuda.current_stream()
            self.stream.wait_stream(caller_stream)
            # Ordering does not prevent allocator reuse after the caller drops
            # its reference; keep the input alive through the side-stream read.
            hash_indices.record_stream(self.stream)
            with torch.cuda.stream(self.stream):
                embeddings = self.multi_head_embedding(hash_indices)
                embeddings = embeddings.flatten(start_dim=-2)
                if dtype is not None:
                    embeddings = embeddings.to(dtype)
                self.sync_event.record()
            # Keep side-stream output alive through the caller's pending reads.
            embeddings.record_stream(caller_stream)
        else:
            embeddings = self.multi_head_embedding(hash_indices)
            embeddings = embeddings.flatten(start_dim=-2)
            if dtype is not None:
                embeddings = embeddings.to(dtype)

        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the Engram module.

        Args:
            hidden_states: Hidden states of shape ``[T, HC_MULT, D]``.
            embeddings: Pre-computed embeddings from ``precompute()``.
            conv_state: Optional conv state from a previous decode step
                (shape ``[1, C_total, conv_state_size]``).
            use_cache: When ``True``, return ``(output, new_conv_state)``.

        Returns:
            Output tensor of shape ``[T, HC_MULT, D]`` to be added as
            residual, and optionally the updated conv state.
        """
        # Fused key/value projection: single GEMM replaces hc_mult + 1 separate GEMMs.
        # kv_proj output: [T, (1 + HC) * D]
        D = self.config.hidden_size
        HC = self.config.hc_mult
        kv = self.kv_proj(embeddings)
        value_raw, keys = kv.split([D, HC * D], dim=-1)
        keys = keys.view(*keys.shape[:-1], HC, D)  # [T, HC, D]

        # Vectorised RMSNorm for keys and queries (no per-HC kernel launches).
        # rms_norm(x, w) = x / rms(x) * w  where rms(x) = sqrt(mean(x^2) + eps)
        key_rms = keys.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        normed_keys = keys * key_rms * self.key_norm_weight  # [HC, D] broadcasts

        queries = hidden_states  # [T, HC, D]
        query_rms = queries.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        normed_queries = queries * query_rms * self.query_norm_weight

        # Gating: per-HC dot product between normed keys and queries.
        gates = (normed_keys * normed_queries).sum(dim=-1) / math.sqrt(D)  # [T, HC]
        gates = gates.abs().clamp_min(1e-6).sqrt() * gates.sign()
        gates = gates.sigmoid().unsqueeze(-1)  # [T, HC, 1]

        value = gates * value_raw.unsqueeze(-2)  # [T, HC, D]

        if use_cache:
            conv_out, new_conv_state = self.short_conv(value, conv_state=conv_state, use_cache=True)
            return value + conv_out, new_conv_state

        output = value + self.short_conv(value)
        return output


# Rows per program in the fused lookup below. A row is one (token, head) pair, so a
# block of 16 rows is 16 scattered gathers of `D` bytes; larger blocks buy nothing
# because the addresses are unrelated and each one misses on its own.
_LOOKUP_BLOCK_R = 16

# The exponent bitcast needs explicit ue8m0 code 0 (2**-127, FP32 denormal)
# and code 255 (NaN) handling.
_E8M0_DENORM = tl.constexpr(5.877471754111438e-39)


@lru_cache(maxsize=None)
def _multi_processor_count(device_index: int) -> int:
    """Cache the fixed per-device SM count to avoid repeated property queries."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "vocab_start",
        "vocab_end",
        "ids_stride_t",
        "ids_stride_h",
        "grid_size",
    ]
)
def _engram_lookup_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    offsets_ptr,
    out_ptr,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    grid_size,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DIM_PAD: tl.constexpr,
    QB: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Gather and dequantize into contiguous ``[T * LOCAL_HEADS, DIM]`` output.

    A capped grid-stride loop processes ``BLOCK_R`` rows per program, limiting
    TLB contention and reserving SMs for main-stream work on background calls.
    Head shards use rebased offsets. Row shards mask unowned values to zero
    with scale code 127 (1.0), so their contribution is exactly zero.
    """
    cols = tl.arange(0, DIM_PAD)
    in_dim = cols < DIM
    scale_cols = cols // QB
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, grid_size * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        # Contiguous [T, LOCAL_HEADS, DIM]: heads are adjacent within each token.
        head = rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids_ptr + token * ids_stride_t + (HEAD_START + head) * ids_stride_h,
            mask=valid,
            other=0,
        ).to(tl.int64)
        # Keep int64 addressing: ~48M rows * 256 lanes exceeds the int32 range.
        index += tl.load(offsets_ptr + head, mask=valid, other=0).to(tl.int64)
        if MASKED:
            owned = valid & (index >= vocab_start) & (index < vocab_end)
            local = tl.where(owned, index - vocab_start, 0)
        else:
            owned = valid
            local = index
        gather = owned[:, None] & in_dim[None, :]
        values = tl.load(weight_ptr + local[:, None] * DIM + cols[None, :], mask=gather, other=0.0)
        code = tl.load(
            scale_ptr + local[:, None] * (DIM // QB) + scale_cols[None, :],
            mask=gather,
            other=127,
        )
        # ue8m0 codes map to the FP32 exponent field, except 0 and 255 below.
        scale = (code.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scale = tl.where(code == 0, _E8M0_DENORM, scale)
        scale = tl.where(code == 255, float("nan"), scale)
        tl.store(
            out_ptr + rows[:, None] * DIM + cols[None, :],
            (values.to(tl.float32) * scale).to(out_ptr.dtype.element_ty),
            mask=valid[:, None] & in_dim[None, :],
        )


class ShardedFp8MultiHeadEmbedding(nn.Module):
    """Sharded E4M3 n-gram rows with one E8M0 scale per ``block_size`` lanes.

    With TP > 1 and ``num_heads % tp_size == 0``, each rank owns contiguous heads
    and their buckets. Lookup returns local heads for the caller's all-gather.
    Otherwise, ranks own contiguous row ranges, return zero for unowned rows,
    and the caller combines full-width contributions with a sum all-reduce.
    ``shard_heads`` selects the collective in ``DeepseekV41Engram.precompute``.

    Both shard layouts stay in pinned host memory; lookup reads them through UVA.
    The default scale block size is 32 lanes.
    """

    def __init__(
        self,
        list_of_N: List[int],
        D: int,
        block_size: int = 32,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        if D % block_size != 0:
            raise ValueError(
                f"Engram embedding dim {D} is not a multiple of the fp8 block size "
                f"{block_size}; the checkpoint stores one exponent per block, so a "
                "partial trailing block has no scale to dequantize it with."
            )
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.block_size = block_size
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # The HF loader fills pinned allocations in place; other loaders may replace them.
        self._requires_standard_hf_loading = True
        # Rebuild UVA views when host addresses change to avoid stale pointers.
        self._views: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._views_src: Optional[Tuple[int, int]] = None

        self.num_embeddings = sum(list_of_N)
        # Head all-gather requires equal widths; use row shards for uneven splits.
        self.shard_heads = tp_size > 1 and self.num_heads % tp_size == 0
        offsets = _bucket_offsets(list_of_N)

        if self.shard_heads:
            heads_per_rank = self.num_heads // tp_size
            self.head_start = tp_rank * heads_per_rank
            head_end = self.head_start + heads_per_rank
            self.num_local_heads = heads_per_rank
            self.vocab_start_idx = int(offsets[self.head_start])
            self.vocab_end_idx = self.vocab_start_idx + sum(list_of_N[self.head_start : head_end])
            self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
            # Owned heads use local row offsets; both layouts register the same buffer.
            offsets = offsets[self.head_start : head_end] - self.vocab_start_idx
        else:
            self.head_start = 0
            self.num_local_heads = self.num_heads
            # Match the reference's ceil-divided row ranges, with a short final shard.
            self.part_num_embeddings = (self.num_embeddings + tp_size - 1) // tp_size
            self.vocab_start_idx = tp_rank * self.part_num_embeddings
            self.vocab_end_idx = min(
                self.vocab_start_idx + self.part_num_embeddings, self.num_embeddings
            )

        self.register_buffer("offsets", offsets)

        # Override the model's CUDA construction context for table storage.
        if not prefer_pinned():
            raise RuntimeError("Engram tables require the pinned-memory runtime policy")
        self.weight = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                D,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                pin_memory=prefer_pinned(),
            ),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                D // block_size,
                dtype=torch.float8_e8m0fnu,
                device="cpu",
                pin_memory=prefer_pinned(),
            ),
            requires_grad=False,
        )

    def _materialize_pinned(self) -> None:
        """Materialize meta shards directly on the host, without an HBM staging copy."""
        for name in ("weight", "scale"):
            param = self._parameters[name]
            if param.device.type == "cpu" and param.is_pinned():
                continue
            if param.device.type != "meta":
                raise RuntimeError(
                    f"Engram {name} must be meta or pinned CPU memory, got {param.device}"
                )
            if not prefer_pinned():
                raise RuntimeError("Engram tables require the pinned-memory runtime policy")
            pinned = nn.Parameter(
                torch.empty(
                    param.shape,
                    device="cpu",
                    dtype=param.dtype,
                    pin_memory=prefer_pinned(),
                ),
                requires_grad=False,
            )
            if not pinned.is_pinned():
                raise RuntimeError(f"Engram {name} allocation is not pinned")
            self.register_parameter(name, pinned)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "ShardedFp8MultiHeadEmbedding":
        """Preserve table dtypes and pinned host addresses across model transforms."""
        # Materialization and model.to("cuda") must preserve pinned tables while
        # offsets and other device buffers still follow the model.
        host_parameters = {name: self._parameters.pop(name) for name in ("weight", "scale")}
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._parameters.update(host_parameters)
        self._materialize_pinned()
        return result

    def load_weights(self, weights: List[Dict]) -> None:
        """Load this rank's rows from full-height checkpoint weights and scales.

        Slice safetensors lazily so unowned rows are not read into host memory.
        """
        self._materialize_pinned()
        weights = weights[0] if isinstance(weights, list) else weights
        start, end = self.vocab_start_idx, self.vocab_end_idx
        for name, param in (("weight", self.weight), ("scale", self.scale)):
            src = weights[name]
            shard = src[start:end]
            if not isinstance(shard, torch.Tensor):
                shard = torch.as_tensor(shard)
            rows = shard.shape[0]
            if rows != param.shape[0]:
                # Only the final row shard may be short by the ceil-divide remainder.
                # Other mismatches violate the hash bucket layout.
                if self.shard_heads or self.tp_rank != self.tp_size - 1 or rows > param.shape[0]:
                    raise ValueError(
                        f"Engram {name}: rank {self.tp_rank}/{self.tp_size} expected "
                        f"{param.shape[0]} rows in [{start}, {end}) of a "
                        f"{self.num_embeddings}-row table but the checkpoint tensor "
                        f"yielded {rows}."
                    )
                param.data[:rows].copy_(shard.to(param.dtype))
                # Valid hashes cannot reach padding; zero it defensively.
                param.data[rows:].zero_()
            else:
                param.data.copy_(shard.to(param.dtype))

    def storage(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached UVA ``(weight, scale)`` views, refreshed if host addresses change."""
        src = (self.weight.data_ptr(), self.scale.data_ptr())
        if self._views_src != src:
            self._views = (
                _uva_device_view(self.weight.data),
                _uva_device_view(self.scale.data),
            )
            self._views_src = src
        assert self._views is not None
        return self._views

    def forward(self, input_ids: torch.Tensor, background: bool = False) -> torch.Tensor:
        """``[T, num_heads]`` bucket-local indices -> ``[T, num_local_heads, D]``.

        Head shards return only owned heads; row shards return all heads with
        unowned rows zeroed. The caller orders the matching collective after the
        lookup stream. Fused dequantization avoids a full FP32 intermediate.

        ``background`` caps side-stream occupancy to leave SMs for the main stream.
        """
        num_rows = input_ids.shape[0] * self.num_local_heads
        out = torch.empty(
            (input_ids.shape[0], self.num_local_heads, self.embedding_dim),
            dtype=self.dtype,
            device=input_ids.device,
        )
        if num_rows == 0:
            return out
        weight, scale = self.storage()
        num_sms = _multi_processor_count(input_ids.device.index)
        grid_size = min(
            triton.cdiv(num_rows, _LOOKUP_BLOCK_R),
            max(1, num_sms // 2) if background else num_sms,
        )
        _engram_lookup_kernel[(grid_size,)](
            weight,
            # Triton reads ue8m0 exponent bytes through uint8.
            scale.view(torch.uint8),
            input_ids,
            self.offsets,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            num_rows,
            input_ids.stride(0),
            input_ids.stride(1),
            grid_size,
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.num_local_heads,
            DIM=self.embedding_dim,
            DIM_PAD=triton.next_power_of_2(self.embedding_dim),
            QB=self.block_size,
            # Hashes must stay within each head's bucket. With rebased offsets,
            # head shards own every lookup and need no ownership mask.
            MASKED=not self.shard_heads,
            BLOCK_R=_LOOKUP_BLOCK_R,
        )
        return out
