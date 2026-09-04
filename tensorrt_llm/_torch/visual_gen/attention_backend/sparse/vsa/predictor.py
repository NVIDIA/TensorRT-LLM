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

"""Shared Video Sparse Attention prediction and post-processing."""

from dataclasses import dataclass, field
from functools import cache
from math import ceil
from typing import Mapping, Optional

import torch

from .....attention_backend.block_sparse import BlockSparseForwardInputs
from .....attention_backend.interface import PredefinedAttentionMask
from .....attention_backend.sparse.params import SparseRuntimeParams
from ...trtllm import SparseForwardInputs
from .metadata import (
    _DEFAULT_MAX_CACHED_SHAPES,
    VSA_BLOCK_SIZE,
    VSAMetadata,
    get_vsa_forward_context,
)

_BITS_PER_WORD = 32
_SIGNED_INT32_MAX = torch.iinfo(torch.int32).max


def _mean_pool_cubes(
    x_tiled: torch.Tensor,
    variable_block_sizes: torch.LongTensor,
    prod_tile: int,
    num_cubes: int,
) -> torch.Tensor:
    batch_size, _padded, num_heads, head_dim = x_tiled.shape
    x_cubes = x_tiled.view(batch_size, num_cubes, prod_tile, num_heads, head_dim)
    # FP32 accumulation avoids perturbing the coarse softmax when inputs are BF16.
    x_sum = x_cubes.float().sum(dim=2)
    valid_counts = variable_block_sizes.float().clamp(min=1).view(1, num_cubes, 1, 1)
    return (x_sum / valid_counts).to(x_tiled.dtype)


class VSAPreprocessor:
    """Convert compact BSHD tensors between sequence-major and tile-major order."""

    @staticmethod
    def tile(
        x: torch.Tensor,
        non_pad_index: torch.LongTensor,
        gather_idx: torch.LongTensor,
        padded_seq_len: int,
    ) -> torch.Tensor:
        # index_select + index_copy_ keeps this path traceable by torch.compile.
        batch_size, _seq_len, num_heads, head_dim = x.shape
        x_valid = x.index_select(1, gather_idx)
        x_padded = x.new_zeros(batch_size, padded_seq_len, num_heads, head_dim)
        x_padded.index_copy_(1, non_pad_index, x_valid)
        return x_padded

    @staticmethod
    def untile(
        x: torch.Tensor,
        untile_idx: torch.LongTensor,
    ) -> torch.Tensor:
        return torch.index_select(x, 1, untile_idx)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VSAPostProcessContext:
    """Per-call tensors needed after the backend executes the fine stage."""

    coarse_output: torch.Tensor = field(repr=False)
    gate_compress: torch.Tensor = field(repr=False)
    gate_fine: Optional[torch.Tensor] = field(default=None, repr=False)
    untile_idx: Optional[torch.LongTensor] = field(default=None, repr=False)
    output_shape: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VSAForwardInputs(SparseForwardInputs):
    """Typed VSA prediction consumed by TRTLLM or CuTe DSL fine attention."""

    topk_indices: torch.IntTensor = field(repr=False)
    variable_block_sizes: torch.LongTensor = field(repr=False)
    cur_topk: int
    num_cubes: int
    post_context: VSAPostProcessContext = field(repr=False)


class _VSARouteBuilder:
    """Lower fixed-width VSA top-K tables into graph-stable BSR routes."""

    def __init__(self, max_cached_shapes: int = _DEFAULT_MAX_CACHED_SHAPES) -> None:
        if max_cached_shapes <= 0:
            raise ValueError("max_cached_shapes must be positive")
        self._max_cached_shapes = max_cached_shapes
        self._indptr_cache: dict[tuple[torch.device, int, int, int, int], torch.Tensor] = {}

    def from_selected_blocks(
        self,
        selected_blocks: torch.Tensor,
        kv_valid_bits: torch.Tensor,
    ) -> BlockSparseForwardInputs:
        batch_size, num_kv_heads, num_q_blocks, blocks_per_row = map(int, selected_blocks.shape)
        key = (
            selected_blocks.device,
            batch_size,
            num_kv_heads,
            num_q_blocks,
            blocks_per_row,
        )
        block_indptr = self._indptr_cache.get(key)
        if block_indptr is None:
            if len(self._indptr_cache) >= self._max_cached_shapes:
                raise RuntimeError(
                    "VSA route cache reached its "
                    f"{self._max_cached_shapes}-shape limit; restart the pipeline or "
                    "reuse a configured resolution/frame profile"
                )
            if selected_blocks.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "VSA route cache miss during CUDA Graph capture; "
                    "run an eager warmup with the same selected-block shape first"
                )
            total_entries = batch_size * num_kv_heads * num_q_blocks * blocks_per_row
            if total_entries > _SIGNED_INT32_MAX:
                raise OverflowError("VSA route offsets must fit in signed int32")
            row_offsets = torch.arange(
                num_q_blocks + 1,
                dtype=torch.int32,
                device=selected_blocks.device,
            ).reshape(1, 1, -1)
            head_offsets = torch.arange(
                batch_size * num_kv_heads,
                dtype=torch.int32,
                device=selected_blocks.device,
            ).reshape(batch_size, num_kv_heads, 1)
            block_indptr = (
                head_offsets * (num_q_blocks * blocks_per_row) + row_offsets * blocks_per_row
            ).contiguous()
            self._indptr_cache[key] = block_indptr
        return BlockSparseForwardInputs(
            q_block_size=VSA_BLOCK_SIZE,
            kv_block_size=VSA_BLOCK_SIZE,
            max_blocks_per_row=blocks_per_row,
            block_indptr=block_indptr,
            block_indices=torch.sort(selected_blocks, dim=-1).values.reshape(-1).contiguous(),
            kv_valid_bits=kv_valid_bits,
        )


@cache
def _get_bit_weights(device: torch.device) -> torch.Tensor:
    bit_positions = torch.arange(_BITS_PER_WORD, dtype=torch.int64, device=device)
    return torch.bitwise_left_shift(torch.ones_like(bit_positions), bit_positions)


def _pack_kv_token_mask(kv_token_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if kv_token_mask.ndim == 1:
        batched_mask = kv_token_mask.unsqueeze(0).expand(batch_size, -1)
    else:
        batched_mask = kv_token_mask
    seq_len_kv = int(batched_mask.shape[1])
    padded_length = ceil(seq_len_kv / _BITS_PER_WORD) * _BITS_PER_WORD
    if padded_length != seq_len_kv:
        batched_mask = torch.nn.functional.pad(batched_mask, (0, padded_length - seq_len_kv))
    words = (
        batched_mask.reshape(batch_size, -1, _BITS_PER_WORD).to(torch.int64)
        * _get_bit_weights(kv_token_mask.device)
    ).sum(dim=-1)
    return words.to(torch.uint32).contiguous()


class VSAPredictor:
    """Produce the complete per-call VSA block-attention input envelope."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_cached_shapes: int = _DEFAULT_MAX_CACHED_SHAPES,
    ) -> None:
        resolved_num_kv_heads = num_kv_heads or num_heads
        if resolved_num_kv_heads != num_heads:
            raise ValueError(
                "VSA coarse mean-pool assumes MHA (num_kv_heads == num_heads), "
                f"got num_kv_heads={resolved_num_kv_heads}, num_heads={num_heads}. "
                "GQA/MQA is not supported."
            )
        self._route_builder = _VSARouteBuilder(max_cached_shapes=max_cached_shapes)

    @torch.compiler.disable
    def get_metadata(self) -> VSAMetadata:
        metadata = get_vsa_forward_context()
        if metadata is None:
            raise RuntimeError(
                "VSA attention called without an active VSA forward context. "
                "Wrap each transformer call with set_vsa_forward_context()."
            )
        return metadata

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_compress: Optional[torch.Tensor],
        gate_fine: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if gate_compress is None:
            raise ValueError(
                "VSA requires gate_compress. "
                "Ensure to_gate_compress is wired in the transformer block."
            )
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("VSA requires Q, K, and V with the same BSHD shape.")
        if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
            raise ValueError("VSA requires Q, K, and V to share device and dtype.")
        if not isinstance(gate_compress, torch.Tensor):
            raise TypeError("VSA gate_compress must be a torch.Tensor.")
        if (
            gate_compress.shape != q.shape
            or gate_compress.device != q.device
            or gate_compress.dtype != q.dtype
        ):
            raise ValueError("VSA gate_compress must share Q's shape, device, and dtype.")
        if gate_fine is not None and (
            not isinstance(gate_fine, torch.Tensor)
            or gate_fine.shape != q.shape
            or gate_fine.device != q.device
            or gate_fine.dtype != q.dtype
        ):
            raise ValueError("VSA gate_fine must share Q's shape, device, and dtype.")
        return gate_compress, gate_fine

    def predict(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        seq_len_kv: int,
        attention_mask: PredefinedAttentionMask,
        gate_compress: Optional[torch.Tensor],
        gate_fine: Optional[torch.Tensor],
        use_sparse_fine: bool,
        produce_block_sparse_inputs: bool,
        forward_kwargs: Mapping[str, object],
        metadata: Optional[VSAMetadata] = None,
    ) -> VSAForwardInputs:
        """Predict routes, effective QKV, and the shared post-process context."""

        gate_compress, gate_fine = self._validate_inputs(q, k, v, gate_compress, gate_fine)
        if attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("VSA supports only full self-attention.")
        if seq_len_kv != seq_len:
            raise ValueError("VSA requires self-attention with matching Q and KV sequence lengths.")
        if tuple(q.shape[:2]) != (batch_size, seq_len):
            raise ValueError("VSA batch_size and seq_len must match the compact QKV tensors.")

        metadata = metadata or self.get_metadata()
        padded_len = metadata.padded_seq_length
        num_cubes = metadata.num_cubes
        cur_topk = max(1, ceil((1.0 - metadata.vsa_sparsity) * num_cubes))
        q_tiled = VSAPreprocessor.tile(q, metadata.non_pad_index, metadata.gather_idx, padded_len)
        k_tiled = VSAPreprocessor.tile(k, metadata.non_pad_index, metadata.gather_idx, padded_len)
        v_tiled = VSAPreprocessor.tile(v, metadata.non_pad_index, metadata.gather_idx, padded_len)

        q_coarse = _mean_pool_cubes(
            q_tiled, metadata.variable_block_sizes, VSA_BLOCK_SIZE, num_cubes
        )
        k_coarse = _mean_pool_cubes(
            k_tiled, metadata.variable_block_sizes, VSA_BLOCK_SIZE, num_cubes
        )
        v_coarse = _mean_pool_cubes(
            v_tiled, metadata.variable_block_sizes, VSA_BLOCK_SIZE, num_cubes
        )
        coarse_scores = torch.einsum("bnhd,bmhd->bhnm", q_coarse, k_coarse) * q.shape[-1] ** -0.5
        coarse_probs = coarse_scores.softmax(dim=-1)
        coarse_output = torch.einsum("bhnm,bmhd->bnhd", coarse_probs, v_coarse)
        topk_indices = coarse_probs.topk(cur_topk, dim=-1).indices.to(torch.int32)
        coarse_output_tiled = (
            coarse_output.unsqueeze(2)
            .expand(batch_size, num_cubes, VSA_BLOCK_SIZE, q.shape[2], q.shape[3])
            .reshape(batch_size, padded_len, q.shape[2], q.shape[3])
        )
        coarse_output_compact = VSAPreprocessor.untile(coarse_output_tiled, metadata.untile_idx)

        block_sparse_inputs = None
        if use_sparse_fine and produce_block_sparse_inputs:
            kv_valid_bits = _pack_kv_token_mask(metadata.kv_token_mask, batch_size)
            block_sparse_inputs = self._route_builder.from_selected_blocks(
                topk_indices,
                kv_valid_bits,
            )

        effective_q = q_tiled if use_sparse_fine else q
        effective_k = k_tiled if use_sparse_fine else k
        effective_v = v_tiled if use_sparse_fine else v
        effective_seq_len = padded_len if use_sparse_fine else seq_len
        return VSAForwardInputs(
            q=effective_q,
            k=effective_k,
            v=effective_v,
            batch_size=batch_size,
            seq_len=effective_seq_len,
            seq_len_kv=effective_seq_len,
            attention_mask=attention_mask,
            sparse_runtime_params=SparseRuntimeParams(
                block_sparse_inputs=block_sparse_inputs,
            ),
            forward_kwargs=forward_kwargs,
            topk_indices=topk_indices,
            variable_block_sizes=metadata.variable_block_sizes,
            cur_topk=cur_topk,
            num_cubes=num_cubes,
            post_context=VSAPostProcessContext(
                coarse_output=coarse_output_compact,
                gate_compress=gate_compress,
                gate_fine=gate_fine,
                untile_idx=metadata.untile_idx if use_sparse_fine else None,
                output_shape=tuple(q.shape),
            ),
        )


def vsa_post_process(output: torch.Tensor, inputs: VSAForwardInputs) -> torch.Tensor:
    """Combine coarse/fine VSA outputs and restore compact BSHD order."""

    context = inputs.post_context
    fine_output = output.reshape(
        inputs.batch_size,
        inputs.seq_len,
        context.output_shape[2],
        context.output_shape[3],
    )
    if context.untile_idx is not None:
        fine_output = VSAPreprocessor.untile(fine_output, context.untile_idx)
    if context.gate_fine is not None:
        fine_output = context.gate_fine * fine_output
    return context.gate_compress * context.coarse_output + fine_output


__all__ = [
    "VSAForwardInputs",
    "VSAPostProcessContext",
    "VSAPredictor",
    "vsa_post_process",
]
