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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tokenizers import Regex, normalizers
from torch import nn
from transformers import AutoTokenizer

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

    All hash computation runs on GPU using PyTorch ops. CPU-side
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
        for layer_id in config.layer_ids:
            self._multipliers[layer_id] = hash_mapping.layer_multipliers[layer_id].clone()
            self._modules[layer_id] = [
                torch.tensor(ngram_head_sizes, dtype=torch.long)
                for ngram_head_sizes in hash_mapping.vocab_size_across_layers[layer_id]
            ]

        self._device: Optional[torch.device] = None

        # Cache for CUDA graph capture - stores last computed hashes.
        # _cached_hashes points to the most-recently used entry.
        # _cached_hashes_store keeps ALL per-shape entries alive so that
        # CUDA graph tensor addresses remain valid after subsequent compute calls
        # change the active shape (e.g. warmup batch_size=4 → 3 → 2 → 1).
        self._cached_hashes: Optional[Dict[int, torch.Tensor]] = None
        self._cached_hashes_store: Dict[tuple, Dict[int, torch.Tensor]] = {}

    def _ensure_on_device(self, device: torch.device):
        """Move hash tensors to the specified device (lazy, once)."""
        if self._device == device:
            return
        self._lookup_table = self._lookup_table.to(device)
        for layer_id in list(self._multipliers.keys()):
            self._multipliers[layer_id] = self._multipliers[layer_id].to(device)
            self._modules[layer_id] = [m.to(device) for m in self._modules[layer_id]]
        self._device = device

    def compute_hashes(
        self,
        input_ids: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """Compute hash indices on GPU using PyTorch ops.

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

        # 1. Compress token ids via lookup table (1-D)
        ids = input_ids.long().view(-1)
        vocab_size = self._lookup_table.shape[0]
        compressed = self._lookup_table[ids.clamp(0, vocab_size - 1)]

        (T,) = compressed.shape

        # 2. Pre-compute shifted versions (left-padded with pad_id)
        shifts = [compressed]
        for k in range(1, self.config.max_ngram_size):
            padded = torch.nn.functional.pad(compressed, (k, 0), value=self._pad_id)
            shifts.append(padded[:T])

        # 2b. When input_ids is a packed/flattened tensor containing multiple
        # sequences, mask out shifted positions that cross sequence boundaries
        # so n-gram hashes stay within each sequence.
        if seq_lens is not None and seq_lens.numel() > 1:
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

                moduli = self._modules[layer_id][n_gram_index]
                for j in range(self.config.n_head_per_ngram):
                    head_hash = mix % int(moduli[j].item())
                    all_hashes.append(head_hash)

            result[layer_id] = torch.stack(all_hashes, dim=1)

        # Cache for CUDA graph capture.
        # Use a per-shape store so tensors captured by CUDA graphs are never
        # garbage-collected when a subsequent call uses a different batch size.
        # When the same shape is seen again, update the existing tensors
        # in-place so CUDA graphs can read the freshly computed values from
        # the same memory addresses used at capture time.
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

    @property
    def layer_ids(self) -> List[int]:
        """Return the list of layer IDs that have Engram modules."""
        return self.config.layer_ids

    @property
    def vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        """Return the vocabulary sizes for each layer and head."""
        return self._vocab_size_across_layers


class MultiHeadEmbedding(nn.Module):
    """Multi-head embedding layer with per-head offset handling.

    Each head has its own embedding space, indexed by applying offsets
    to the input indices before a single embedding lookup.
    """

    def __init__(self, list_of_N: List[int], D: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

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

        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=vocab_sizes_flat,
            D=embed_dim_per_head,
            dtype=dtype,
        )

        self.short_conv = ShortConv(
            hidden_size=config.hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
            hc_mult=config.hc_mult,
            norm_eps=config.norm_eps,
            dtype=dtype,
        )

        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        hc = config.hc_mult
        D = config.hidden_size

        # Fused projection: one GEMM produces value (1 head) + all keys (hc_mult heads).
        # Output layout: [value (D) | key_0 (D) | key_1 (D) | ... | key_{hc-1} (D)]
        self.kv_proj = nn.Linear(engram_hidden_size, (1 + hc) * D, bias=False, dtype=dtype)

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
    ) -> torch.Tensor:
        """Pre-compute embeddings from hash indices.

        When a stream was supplied at construction time, the work is
        dispatched onto that stream (overlapping with the main stream) and
        ``sync_event`` is recorded so the caller can synchronize before
        consuming the result.  When no stream is set the computation runs
        on the current stream synchronously and ``sync_event`` is None.

        Args:
            hash_indices: Hash indices of shape ``[T, num_heads]`` on GPU.
            dtype: Cast embeddings to this dtype after lookup.
                If None, uses the embedding's native dtype.

        Returns:
            Embedding tensor of shape ``[T, num_heads * embed_dim_per_head]``.
        """
        if self.stream is not None:
            # Fork: let the engram stream wait for any pending main-stream work
            # (e.g. hash computation) before we start the embedding lookup.
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                embeddings = self.multi_head_embedding(hash_indices)
                embeddings = embeddings.flatten(start_dim=-2)
                if dtype is not None:
                    embeddings = embeddings.to(dtype)
                self.sync_event.record()
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
