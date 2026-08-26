# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram short-convolution side path for Qwen3.8-Flash-Next.

The PLE (Parallel Layer Extension) side path is injected into the Hyper-Connection
hidden stream *before* attention at exactly one decoder layer: ``layer_id`` such
that ``(layer_id + 1) in config.ple_layer_ids``. For the released
The checkpoint uses ``ple_layer_ids == [2]``, so PLE is active at
``layer_id == 1`` only.

Source of truth (parity oracle): the sglang reference ``qwen4_exp.py``
(``Qwen4ExpPLELayer`` / ``Qwen4ExpNGramEmbedding`` / ``Qwen4ExpPLEGroupedNorm`` +
``_prepare_ple_batch`` / ``_commit_ple_batch``). This module is a faithful,
runtime-decoupled re-implementation of that math:

  * n-gram id hashing — splitmix64 layer multipliers, per-head prime vocab sizes
    (``nth-prime after ngram_vocab_size_base - 1``), eos-segmented right-shift, XOR
    mixing, ``mod prime + offset`` per head;
  * n-gram embedding lookup through the in-tree :class:`Embedding`, row-sharded
    over TP ranks; attention-DP uses all-gather / reduce-scatter to preserve
    rank-local token ownership;
  * key/value projection, a per-Hyper-Connection-stream grouped Gemma
    (``weight + 1``) RMSNorm gate (signed-sqrt sigmoid), and a **dilated causal
    depthwise short conv** over the ``hc_count * hidden`` stream with a carried
    conv state and n-gram-context history.

Why a NEW module (not ``modules/engram``): DSv4's ``Engram`` shares only the
signed-sqrt-sigmoid *gating* shape. It diverges on (a) hash multipliers
(numpy-RNG ``r*2+1`` vs Qwen4 splitmix64 ``2*(mix % half_bound)+1``), (b) prime
selection (dedup ``seen_primes`` set vs Qwen4 ``nth-prime after base-1`` indexed
by global head), (c) shift (plain left-pad vs eos-segmented), (d) no Gemma
``weight+1`` RMSNorm offset, and (e) **no conv-state carry-over across the
prefill->decode boundary** (a documented Engram limitation), so it cannot
reproduce the sglang reference for a decode step. Plan labels C4 "New".

Runtime contract: the caller
owns two per-layer recurrent-state pools indexed by ``PLEMetadata.state_indices``
and updated **in place** (mamba-style) by :meth:`Qwen4ExpPLE.forward`:

  * ``conv_state`` — shape ``(num_slots, *module.conv_state_shape)`` where
    ``conv_state_shape == (conv_channels, short_conv_state_len)``; init to 0.
  * ``ngram_context`` — shape ``(num_slots, module.ngram_context_len)`` int64
    token history; init to ``eos_token_id``.

``short_conv_state_len == (ple_conv_kernel_size - 1) * ngram_size`` (dilation ==
ngram_size), ``conv_channels == hc_count * hidden_size``, and
``ngram_context_len == ngram_size - 1`` — identical to the shapes
``config_utils.extract_qwen4_exp_ple_cache_params`` derives for the V2 cache.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.distributed import allgather, reducescatter
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm.mapping import Mapping

from .mamba.layernorm_gated import RMSNorm as TritonRMSNorm

# --- splitmix64 constants (verbatim from the sglang reference) ---
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(x: int) -> int:
    """The finalizer mix used to seed the per-layer n-gram hash multipliers."""
    x = (x + _SPLITMIX_GAMMA) & _MASK64
    x = ((x ^ (x >> 30)) * _SPLITMIX_M1) & _MASK64
    x = ((x ^ (x >> 27)) * _SPLITMIX_M2) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int) -> int:
    """Smallest prime strictly greater than ``start`` (matches sympy.nextprime)."""
    candidate = int(start) + 1
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _find_nth_prime_after(start: int, n: int) -> int:
    """The ``n``-th prime strictly greater than ``start`` (``n >= 1``).

    Equivalent to ``n`` successive ``sympy.nextprime`` calls from ``start``; a
    dependency-free primality test keeps the module import-light (the reference's
    ``sympy`` is not guaranteed present in the runtime container).
    """
    prime = int(start)
    for _ in range(n):
        prime = _next_prime(prime)
    return prime


def _pad_token_rows(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Zero-pad the leading (token) dim up to ``total_tokens`` (CUDA-graph pad)."""
    if x.shape[0] == total_tokens:
        return x
    out = x.new_zeros((total_tokens, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


@dataclasses.dataclass
class PLEMetadata:
    """Per-forward token layout for the PLE side path (runtime-decoupled).

    Re-derives the fields ``_prepare_ple_batch`` computes in the sglang reference,
    but from plain tensors instead of an sglang ``ForwardBatch``, so the module is
    usable both from focused unit tests and model wiring.
    All index/length tensors live on the same device as ``input_ids``.
    """

    is_decode: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor  # [num_seq] this-chunk token count per sequence
    row_width: int  # max chunk length (== 1 for decode)
    req_indices: torch.Tensor  # [processed_tokens] -> sequence index
    token_offsets: torch.Tensor  # [processed_tokens] -> offset within the chunk
    valid_tokens: torch.Tensor  # [processed_tokens] bool
    state_indices: torch.Tensor  # [num_seq] -> recurrent-state pool slot
    padded_tokens: torch.Tensor  # [num_seq, row_width] eos-padded token ids
    ngram_eos_token_id: int
    all_rank_num_tokens: Optional[list[int]] = None
    num_contexts: int = 0
    use_spec_decoding: bool = False

    @classmethod
    def build(
        cls,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        is_decode: bool,
        eos_token_id: int,
        physical_tokens: Optional[int] = None,
        num_contexts: int = 0,
        use_spec_decoding: bool = False,
        uniform_row_width: Optional[int] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
    ) -> "PLEMetadata":
        """Construct metadata from packed ``input_ids`` and per-sequence lengths.

        ``input_ids`` is the flattened, sequence-packed token id vector of length
        ``processed_tokens``. For prefill/extend, ``seq_lens[i]`` is sequence
        ``i``'s chunk length and the tokens are packed contiguously per sequence.
        For decode there is exactly one new token per sequence, so
        ``num_seq == processed_tokens`` and ``seq_lens`` is ignored.
        """
        input_ids = input_ids.reshape(-1)
        device = input_ids.device
        processed_tokens = int(input_ids.shape[0])
        if physical_tokens is None:
            physical_tokens = processed_tokens
        positions = torch.arange(processed_tokens, device=device, dtype=torch.long)

        if is_decode:
            num_seq = processed_tokens
            lengths = torch.ones(num_seq, dtype=torch.long, device=device)
            row_width = 1
            req_indices = positions
            token_offsets = torch.zeros_like(positions)
        elif uniform_row_width is not None:
            if uniform_row_width <= 0:
                raise ValueError("PLE uniform row width must be positive")
            num_seq = state_indices.shape[0]
            if processed_tokens != num_seq * uniform_row_width:
                raise ValueError(
                    "PLE uniform layout does not match the processed token count: "
                    f"{processed_tokens} != {num_seq} * {uniform_row_width}"
                )
            row_width = uniform_row_width
            lengths = torch.full((num_seq,), row_width, dtype=torch.long, device=device)
            req_indices = torch.div(positions, row_width, rounding_mode="floor")
            token_offsets = positions.remainder(row_width)
        else:
            lengths = seq_lens.to(device=device, dtype=torch.long)
            seq_lens_cpu = lengths.tolist()
            row_width = max(seq_lens_cpu) if seq_lens_cpu else processed_tokens
            num_seq = lengths.shape[0]
            query_start_loc = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)])
            req_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
            # Keep this tensor-only: ``processed_tokens`` can be symbolic under
            # torch.compile, and the empty-token assignment below is already a
            # no-op. A Python truth-value guard would force a host-side data
            # dependency and prevent Dynamo from compiling the model forward.
            req_indices = req_indices.clamp(min=0, max=num_seq - 1)
            token_offsets = positions - query_start_loc.index_select(0, req_indices)

        num_seq = lengths.shape[0]
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)
        state_indices = state_indices.to(device=device, dtype=torch.long)

        padded = input_ids.new_full((num_seq, row_width), eos_token_id)
        padded[req_indices, token_offsets] = torch.where(
            valid_tokens, input_ids, input_ids.new_full((), eos_token_id)
        )

        return cls(
            is_decode=is_decode,
            physical_tokens=physical_tokens,
            processed_tokens=processed_tokens,
            lengths=lengths,
            row_width=row_width,
            req_indices=req_indices,
            token_offsets=token_offsets,
            valid_tokens=valid_tokens,
            state_indices=state_indices,
            padded_tokens=padded,
            ngram_eos_token_id=eos_token_id,
            all_rank_num_tokens=all_rank_num_tokens,
            num_contexts=num_contexts,
            use_spec_decoding=use_spec_decoding,
        )


class Qwen4ExpPLEGroupedNorm(TritonRMSNorm):
    """Grouped Gemma (``weight + 1``) RMSNorm, computed per group in fp32.

    ``group_size`` groups the last dim into contiguous blocks (one per
    Hyper-Connection stream); ``group_size is None`` normalizes the whole last
    dim. CUDA tensors reuse TRT-LLM's fused grouped RMSNorm kernel; CPU tensors
    retain the native reference path for construction and parity tests.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ) -> None:
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        super().__init__(
            hidden_size,
            eps=eps,
            group_size=group_size,
            weight_is_delta=True,
        )
        # Gemma stores a delta around one. Construct a fresh tensor instead of
        # mutating the meta-initialized parameter in place.
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return super().forward(x)
        compute_dtype = x.dtype
        x_float = x.float()
        if self.group_size == self.hidden_size:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        else:
            group_shape = x_float.shape[:-1] + (-1, self.group_size)
            variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
            variance = variance.expand(group_shape).reshape_as(x_float)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        weight = self.weight.float() + 1.0
        return (x_norm * weight).to(compute_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashes per-token n-gram contexts to per-head ids and embeds them.

    The hashing (splitmix64 multipliers, prime head vocab sizes / offsets,
    eos-segmented right-shift, XOR mix) is a verbatim port of the sglang
    ``Qwen4ExpNGramEmbedding``. The embedding table is row-sharded across the
    model TP group. Ordinary TP uses masked lookup plus AllReduce; attention-DP
    uses token AllGather, masked local lookup, and ReduceScatter.
    """

    def __init__(
        self,
        config,
        embedding_dim: int,
        ple_layer_index: int = 0,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        self.eos_token_id = int(config.eos_token_id)
        self.seed = int(getattr(config, "seed", 1234))
        self.mapping = mapping
        self.tp_size = mapping.tp_size if mapping is not None else 1
        self.tp_rank = mapping.tp_rank if mapping is not None else 0
        self.use_attention_dp_sharding = bool(
            mapping is not None and mapping.enable_attention_dp and self.tp_size > 1
        )

        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )

        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(config.make_ngram_vocab_size_divisible_by)
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = self._build_head_vocab_and_offsets()
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.padded_vocab_size = padded_vocab_size
        if self.use_attention_dp_sharding:
            slice_width = math.ceil(padded_vocab_size / self.tp_size)
            self.vocab_start_index = self.tp_rank * slice_width
            self.vocab_end_index = min((self.tp_rank + 1) * slice_width, padded_vocab_size)
            self.ngram_embedding = Embedding(
                slice_width,
                self.head_dim_per_ngram,
                dtype=dtype,
            )
        elif self.tp_size > 1:
            self.ngram_embedding = Embedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
            )
            self.vocab_start_index = self.ngram_embedding.vocab_start_index
            self.vocab_end_index = self.ngram_embedding.vocab_end_index
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = padded_vocab_size
            self.ngram_embedding = Embedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                dtype=dtype,
            )

    def _build_layer_multipliers(self, size: int) -> torch.Tensor:
        max_long = (1 << 63) - 1
        m_max = max_long // max(self.unigram_vocab_size, 1)
        half_bound = max(1, m_max // 2)
        values = []
        base_seed = self.seed + _PRIME_1 * self.ple_layer_index
        for idx in range(size):
            x0 = (base_seed + _SPLITMIX_GAMMA * (idx + 1)) & _MASK64
            mixed = _splitmix64(x0)
            values.append(int(2 * (mixed % half_bound) + 1))
        return torch.tensor(values, dtype=torch.long)

    def _build_head_vocab_and_offsets(self):
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        """Right-shift each row by ``n`` within eos-delimited segments.

        Positions that would pull a token across an eos boundary (or off the left
        edge) are filled with ``eos_token_id``, so an n-gram never mixes tokens
        from two different documents. Verbatim port of the reference.
        """
        if n == 0:
            return tensor
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
        eos_mask = tensor == self.eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        pos_in_segment = idx.unsqueeze(0) - segment_start
        src_idx = idx - n
        gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
        return torch.where(valid_mask, shifted, tensor.new_full((), self.eos_token_id))

    def hash_contexts(self, contexts: torch.Tensor) -> torch.Tensor:
        """Hash n-gram windows ``[T, ngram_size]`` to head ids ``[T, ngram_heads]``."""
        contexts = contexts.to(torch.long)
        shifted_tokens = [contexts]
        for shift in range(1, self.ngram_size):
            shifted_tokens.append(self._shift_right_ignore_eos(contexts, shift))

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            ngram_idx = ngram - 2
            start_idx = ngram_idx * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mix = shifted_tokens[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted_tokens[pos] * self.layer_multipliers[pos])
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mix[:, -1:].unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            ngram_ids = ngram_ids + head_offsets.view(1, 1, -1)
            blocks.append(ngram_ids[:, 0])
        return torch.cat(blocks, dim=-1)

    def embed(
        self,
        ngram_ids: torch.Tensor,
        physical_tokens: Optional[int] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Embed head IDs while preserving each ADP rank's token ownership.

        Ordinary TP ranks process identical tokens, so the in-tree column-sharded
        :class:`Embedding` performs a masked local lookup followed by AllReduce.
        Attention-DP ranks process different requests. They therefore all-gather
        the physical token rows, look up only the locally-owned vocabulary rows,
        and reduce-scatter the summed embeddings back to the source rank.
        """
        if not self.use_attention_dp_sharding:
            return self.ngram_embedding(ngram_ids)

        if all_rank_num_tokens is None or len(all_rank_num_tokens) != self.tp_size:
            raise ValueError(
                "PLE row sharding under attention DP requires one token count per TP rank"
            )
        if physical_tokens is None:
            physical_tokens = ngram_ids.shape[0]
        if ngram_ids.shape[0] > physical_tokens:
            raise ValueError(
                f"PLE semantic token count {ngram_ids.shape[0]} exceeds physical "
                f"token count {physical_tokens}"
            )
        if all_rank_num_tokens[self.tp_rank] != physical_tokens:
            raise ValueError(
                "PLE local physical token count does not match attention-DP metadata: "
                f"{physical_tokens} != {all_rank_num_tokens[self.tp_rank]}"
            )

        physical_ids = ngram_ids.new_zeros((physical_tokens, ngram_ids.shape[1]))
        physical_ids[: ngram_ids.shape[0]] = ngram_ids
        sizes = None if len(set(all_rank_num_tokens)) == 1 else all_rank_num_tokens
        gathered_ids = allgather(
            physical_ids,
            self.mapping,
            dim=0,
            sizes=sizes,
        )
        owned = (gathered_ids >= self.vocab_start_index) & (gathered_ids < self.vocab_end_index)
        local_ids = torch.where(
            owned,
            gathered_ids - self.vocab_start_index,
            torch.zeros_like(gathered_ids),
        )
        partial = F.embedding(local_ids, self.ngram_embedding.weight)
        partial.masked_fill_(~owned.unsqueeze(-1), 0)
        local_embeddings = reducescatter(
            partial,
            self.mapping,
            dim=0,
            sizes=sizes,
        )
        return local_embeddings[: ngram_ids.shape[0]]


class Qwen4ExpPLE(nn.Module):
    """The Qwen4-Exp PLE n-gram short-conv side path (contract C4).

    :meth:`forward` consumes the Hyper-Connection hidden bundle
    ``[physical_tokens, hc_count * hidden]`` as the gate query, the per-forward
    :class:`PLEMetadata`, and the caller-owned ``conv_state`` / ``ngram_context``
    recurrent-state pools (updated in place, mamba-style), and returns the PLE
    contribution ``[physical_tokens, hc_count * hidden]`` to be **added** into the
    hidden stream before attention.
    """

    def __init__(
        self,
        config,
        dtype: Optional[torch.dtype] = None,
        ple_layer_index: int = 0,
        layer_id: Optional[int] = None,
        mapping: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.ple_embed_dim = int(config.ple_embed_dim)
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count

        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            self.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            dtype=dtype,
            mapping=mapping,
        )
        self.ngram_size = self.ple_embedding.ngram_size
        self.short_conv_dilation = self.ngram_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.conv_channels = self.hc_hidden_size
        # Recurrent-state pool contract for KVCacheManagerV2.
        self.ngram_context_len = self.ngram_size - 1
        self.conv_state_shape = (self.conv_channels, self.short_conv_state_len)

        self.key_proj = Linear(
            self.ple_embed_dim,
            self.conv_channels,
            bias=False,
            dtype=dtype,
        )
        self.value_proj = Linear(
            self.ple_embed_dim,
            self.hidden_size,
            bias=False,
            dtype=dtype,
        )
        norm_hidden = self.hc_hidden_size
        norm_group = self.hidden_size
        self.norm_key = Qwen4ExpPLEGroupedNorm(
            norm_hidden, eps=config.rms_norm_eps, group_size=norm_group
        )
        self.norm_query = Qwen4ExpPLEGroupedNorm(
            norm_hidden, eps=config.rms_norm_eps, group_size=norm_group
        )
        self.norm_conv = Qwen4ExpPLEGroupedNorm(
            norm_hidden, eps=config.rms_norm_eps, group_size=norm_group
        )
        # Only ``.weight`` (shape [conv_channels, 1, kernel]) is used; the
        # functional conv below prepends the carried state explicitly, so the
        # module's ``padding`` is never exercised (kept for weight-name parity).
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_channels,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
        )
        # Target verification advances through every proposed token. Retain the
        # per-prefix candidates until rejection sampling reports the accepted
        # length, then restore the accepted prefix state.
        self._pending_conv_states: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._pending_ngram_contexts: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            None
        )

    def _apply_ple_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Per-stream grouped norm over a ``[..., hc_count, hidden]`` tensor."""
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def _short_conv(
        self, x: torch.Tensor, metadata: PLEMetadata, conv_state: torch.Tensor
    ) -> torch.Tensor:
        """Dilated causal depthwise short conv with in-place conv-state update.

        ``x`` is ``[processed_tokens, conv_channels]``. ``conv_state`` is the
        caller-owned pool ``[num_slots, conv_channels, short_conv_state_len]``;
        the consumed slots are advanced in place. Returns
        ``silu(conv_output)[processed_tokens, conv_channels]``.
        """
        if x.shape[0] == 0:
            return x
        m = metadata
        num_seq = m.lengths.shape[0]
        state = conv_state.index_select(0, m.state_indices).to(dtype=x.dtype)

        padded_seq = x.new_zeros((num_seq, m.row_width, self.conv_channels))
        padded_seq[m.req_indices, m.token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            self.conv1d.weight.to(dtype=x.dtype),
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.conv_channels,
        ).transpose(1, 2)

        state_cols = torch.arange(self.short_conv_state_len, device=x.device, dtype=torch.long)

        def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
            return conv_input.gather(
                2,
                (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                .unsqueeze(1)
                .expand(-1, self.conv_channels, -1),
            )

        next_state = _gather_at(m.lengths)
        conv_state[m.state_indices] = next_state.to(dtype=conv_state.dtype)

        if m.use_spec_decoding and m.num_contexts < num_seq:
            candidates = conv_input.unfold(2, self.short_conv_state_len, 1)[
                :, :, 1 : m.row_width + 1
            ]
            candidates = candidates.permute(0, 2, 1, 3).contiguous()
            self._pending_conv_states = (
                conv_state,
                m.state_indices[m.num_contexts :],
                candidates[m.num_contexts :],
            )

        return F.silu(conv_output[m.req_indices, m.token_offsets])

    def _commit_ngram_context(
        self, combined: torch.Tensor, metadata: PLEMetadata, ngram_context: torch.Tensor
    ) -> None:
        """Slide each sequence's n-gram history forward by its chunk length.

        Mirrors the reference ``_commit_ple_batch``: the new history is the
        ``ngram_context_len`` columns of ``combined`` starting at ``lengths``
        (i.e. the last ``ngram_size - 1`` tokens the chunk exposed). Written back
        into the caller's pool in place.
        """
        m = metadata
        context_len = combined.shape[1] - m.row_width
        context_cols = torch.arange(context_len, device=combined.device, dtype=torch.long)
        next_context = combined.gather(1, m.lengths.unsqueeze(1) + context_cols.unsqueeze(0))
        ngram_context[m.state_indices] = next_context.to(dtype=ngram_context.dtype)

        if m.use_spec_decoding and m.num_contexts < m.lengths.shape[0]:
            candidates = combined.unfold(1, context_len, 1)[:, 1 : m.row_width + 1].contiguous()
            self._pending_ngram_contexts = (
                ngram_context,
                m.state_indices[m.num_contexts :],
                candidates[m.num_contexts :],
            )

    def commit_speculative_states(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        num_contexts: int,
    ) -> None:
        """Keep the PLE state produced by the accepted verification prefix."""
        del state_indices
        accepted = num_accepted_tokens[num_contexts:].to(torch.long) - 1
        for entry in (self._pending_conv_states, self._pending_ngram_contexts):
            if entry is None:
                continue
            state_pool, slots, candidates = entry
            num_gens = slots.shape[0]
            rows = torch.arange(num_gens, device=slots.device)
            selected = candidates[rows, accepted[:num_gens]]
            state_pool[slots] = selected.to(dtype=state_pool.dtype)
        self._pending_conv_states = None
        self._pending_ngram_contexts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: PLEMetadata,
        conv_state: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the PLE contribution and advance both recurrent-state pools.

        Args:
            hidden_states: ``[physical_tokens, hc_count * hidden]`` HC bundle used
                as the gate query.
            metadata: per-forward token layout (:class:`PLEMetadata`).
            conv_state: ``[num_slots, conv_channels, short_conv_state_len]`` pool,
                updated in place.
            ngram_context: ``[num_slots, ngram_size - 1]`` int64 token-history
                pool, updated in place.

        Returns:
            ``[physical_tokens, hc_count * hidden]`` — the value to add into the
            hidden stream before attention (zero for padded / invalid tokens).
        """
        m = metadata
        hc_dim = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] != hc_dim:
            raise RuntimeError(
                "PLE hidden size does not match its hyper-connection layout: "
                f"expected {hc_dim}, got {hidden_states.shape[-1]}"
            )
        hidden_states = hidden_states[: m.processed_tokens]

        # Build the [history | this-chunk] context grid and gather each token's
        # ngram_size-token window, then hash + embed it.
        history = ngram_context.index_select(0, m.state_indices).to(torch.long)
        combined = torch.cat([history, m.padded_tokens.to(torch.long)], dim=1)
        windows = combined.unfold(1, self.ngram_size, 1)
        contexts = windows[m.req_indices, m.token_offsets]
        ngram_ids = self.ple_embedding.hash_contexts(contexts)
        embeddings = self.ple_embedding.embed(
            ngram_ids,
            physical_tokens=m.physical_tokens,
            all_rank_num_tokens=m.all_rank_num_tokens,
        ).flatten(start_dim=-2)

        key = self.key_proj(embeddings)
        value = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key_normed = self._apply_ple_norm(self.norm_key, key)
        query_normed = self._apply_ple_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)
        gated_value = gate * value.unsqueeze(-2)
        gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(-2)
        gated_value_normed = gated_value_normed.flatten(-2)
        conv_output = self._short_conv(gated_value_normed, metadata, conv_state)
        output = gated_value + conv_output
        output = torch.where(m.valid_tokens.unsqueeze(-1), output, torch.zeros_like(output))

        self._commit_ngram_context(combined, metadata, ngram_context)
        return _pad_token_rows(output, m.physical_tokens)
