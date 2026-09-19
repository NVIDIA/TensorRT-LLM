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
from math import ceil
from typing import Optional

import torch

from .....attention.backends.interface import PredefinedAttentionMask
from .....attention.backends.sparse.params import BlockSparseForwardInputs
from .kernels import blend_coarse_fine, sort_last_dim, tile_and_pool_cubes
from .metadata import (
    _DEFAULT_MAX_CACHED_SHAPES,
    VSA_BLOCK_SIZE,
    VSAMetadata,
    get_vsa_forward_context,
)

_SIGNED_INT32_MAX = torch.iinfo(torch.int32).max


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VSAPostProcessContext:
    """Per-call tensors needed after the backend executes the fine stage.

    ``coarse_output`` stays per cube (``[batch, num_cubes, heads, head_dim]``); the
    post-process gathers it to compact token order together with the fine output.
    """

    coarse_output: torch.Tensor = field(repr=False)
    gate_compress: torch.Tensor = field(repr=False)
    gate_fine: Optional[torch.Tensor] = field(default=None, repr=False)
    untile_idx: torch.LongTensor = field(repr=False)
    fine_is_tiled: bool


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VSAForwardInputs:
    """Typed VSA prediction consumed by TRTLLM or CuTe DSL fine attention.

    The envelope is structurally immutable. Tensor payloads remain live objects
    so CUDA Graph-compatible predictors can publish values into stable buffers.
    ``q``, ``k``, ``v``, and ``seq_len`` describe the effective fine-stage
    inputs: tiled when block-sparse routes are produced, compact otherwise.
    """

    q: torch.Tensor = field(repr=False)
    k: torch.Tensor = field(repr=False)
    v: torch.Tensor = field(repr=False)
    batch_size: int
    seq_len: int
    block_sparse_inputs: Optional[BlockSparseForwardInputs] = field(repr=False)
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
        kv_valid_words: torch.Tensor,
    ) -> BlockSparseForwardInputs:
        """Build the BSR carrier for one prediction.

        Args:
            selected_blocks: ``[batch, kv_heads, q_blocks, blocks_per_row]`` int32 selected
                KV cube per query cube, in any order within a row.
            kv_valid_words: ``[words]`` uint32 packed valid-token mask of the padded
                sequence, shared by every batch entry.
        """
        batch_size, num_kv_heads, num_q_blocks, blocks_per_row = map(int, selected_blocks.shape)
        key = (selected_blocks.device, batch_size, num_kv_heads, num_q_blocks, blocks_per_row)
        block_indptr = self._indptr_cache.get(key)
        if block_indptr is None:
            block_indptr = self._build_block_indptr(*key)
            self._indptr_cache[key] = block_indptr
        return BlockSparseForwardInputs(
            q_block_size=VSA_BLOCK_SIZE,
            kv_block_size=VSA_BLOCK_SIZE,
            max_blocks_per_row=blocks_per_row,
            block_indptr=block_indptr,
            block_indices=sort_last_dim(selected_blocks).reshape(-1),
            kv_valid_bits=kv_valid_words.unsqueeze(0).expand(batch_size, -1).contiguous(),
        )

    def _build_block_indptr(
        self,
        device: torch.device,
        batch_size: int,
        num_kv_heads: int,
        num_q_blocks: int,
        blocks_per_row: int,
    ) -> torch.Tensor:
        if len(self._indptr_cache) >= self._max_cached_shapes:
            raise RuntimeError(
                "VSA route cache reached its "
                f"{self._max_cached_shapes}-shape limit; restart the pipeline or "
                "reuse a configured resolution/frame profile"
            )
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VSA route cache miss during CUDA Graph capture; "
                "run an eager warmup with the same selected-block shape first"
            )
        if batch_size * num_kv_heads * num_q_blocks * blocks_per_row > _SIGNED_INT32_MAX:
            raise OverflowError("VSA route offsets must fit in signed int32")
        row_offsets = torch.arange(num_q_blocks + 1, dtype=torch.int32, device=device)
        head_offsets = torch.arange(batch_size * num_kv_heads, dtype=torch.int32, device=device)
        return (
            head_offsets.reshape(batch_size, num_kv_heads, 1) * (num_q_blocks * blocks_per_row)
            + row_offsets.reshape(1, 1, -1) * blocks_per_row
        ).contiguous()


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
        num_cubes = metadata.num_cubes
        cur_topk = max(1, ceil((1.0 - metadata.vsa_sparsity) * num_cubes))

        (q_tiled, q_coarse), (k_tiled, k_coarse), (v_tiled, v_coarse) = (
            tile_and_pool_cubes(
                x,
                metadata.tile_source_index,
                metadata.variable_block_sizes,
                cube_size=VSA_BLOCK_SIZE,
            )
            for x in (q, k, v)
        )

        coarse_scores = torch.einsum("bnhd,bmhd->bhnm", q_coarse, k_coarse) * q.shape[-1] ** -0.5
        coarse_probs = coarse_scores.softmax(dim=-1)
        coarse_output = torch.einsum("bhnm,bmhd->bnhd", coarse_probs, v_coarse)
        # BSR routes are re-sorted by cube index, so their value order is not needed; other
        # consumers keep receiving the selected cubes in descending probability order.
        topk_indices = coarse_probs.topk(
            cur_topk, dim=-1, sorted=not produce_block_sparse_inputs
        ).indices.to(torch.int32)

        block_sparse_inputs = None
        if use_sparse_fine and produce_block_sparse_inputs:
            block_sparse_inputs = self._route_builder.from_selected_blocks(
                topk_indices,
                metadata.kv_valid_words,
            )

        return VSAForwardInputs(
            q=q_tiled if use_sparse_fine else q,
            k=k_tiled if use_sparse_fine else k,
            v=v_tiled if use_sparse_fine else v,
            batch_size=batch_size,
            seq_len=metadata.padded_seq_length if use_sparse_fine else seq_len,
            block_sparse_inputs=block_sparse_inputs,
            topk_indices=topk_indices,
            variable_block_sizes=metadata.variable_block_sizes,
            cur_topk=cur_topk,
            num_cubes=num_cubes,
            post_context=VSAPostProcessContext(
                coarse_output=coarse_output,
                gate_compress=gate_compress,
                gate_fine=gate_fine,
                untile_idx=metadata.untile_idx,
                fine_is_tiled=use_sparse_fine,
            ),
        )


def vsa_post_process(output: torch.Tensor, inputs: VSAForwardInputs) -> torch.Tensor:
    """Combine coarse/fine VSA outputs and restore compact BSHD order."""

    context = inputs.post_context
    fine_output = output.reshape(
        inputs.batch_size, inputs.seq_len, *context.gate_compress.shape[2:]
    )
    return blend_coarse_fine(
        fine_output,
        context.coarse_output,
        context.gate_compress,
        context.gate_fine,
        context.untile_idx,
        cube_size=VSA_BLOCK_SIZE,
        fine_is_tiled=context.fine_is_tiled,
    )


__all__ = [
    "VSAForwardInputs",
    "VSAPostProcessContext",
    "VSAPredictor",
    "vsa_post_process",
]
