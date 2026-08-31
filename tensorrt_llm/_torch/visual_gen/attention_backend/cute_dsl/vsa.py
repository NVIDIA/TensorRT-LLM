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
Video Sparse Attention (VSA) backend for visual generation models.

VSAAttention implements hierarchical sparse attention:
  - Coarse branch: mean-pooled cube attention (always dense)
  - Fine branch: block-sparse top-K attention via CuTe JIT kernel (sm100+)
    or dense SDPA fallback when CuTe is unavailable / head_dim != 128.
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..interface import AttentionBackend, AttentionTensorLayout

_vsa_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )
except (ImportError, OSError) as e:
    block_sparse_attn_from_indices_cute = None
    is_cute_supported = None
    _vsa_import_error = e


# Must match the Blackwell kernel's block_size expectation.
VSA_TILE_SIZE: Tuple[int, int, int] = (4, 4, 4)

# Kernel's SMEM buffer for variable_block_sizes is fixed-size and unchecked,
# so num_cubes must stay <= this.
VSA_KERNEL_MAX_CUBES: int = 4 * 1024


def _get_tile_partition_indices(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    tT, tH, tW = tile_size
    nT, nH, nW = ceil(T / tT), ceil(H / tH), ceil(W / tW)

    bt = torch.arange(nT, device=device).view(nT, 1, 1, 1, 1, 1)
    bh = torch.arange(nH, device=device).view(1, nH, 1, 1, 1, 1)
    bw = torch.arange(nW, device=device).view(1, 1, nW, 1, 1, 1)
    lt = torch.arange(tT, device=device).view(1, 1, 1, tT, 1, 1)
    lh = torch.arange(tH, device=device).view(1, 1, 1, 1, tH, 1)
    lw = torch.arange(tW, device=device).view(1, 1, 1, 1, 1, tW)

    gt = bt * tT + lt
    gh = bh * tH + lh
    gw = bw * tW + lw
    valid = (gt < T) & (gh < H) & (gw < W)
    flat = gt * (H * W) + gh * W + gw
    out = torch.where(valid, flat, torch.full_like(flat, -1))
    return out.reshape(-1).to(torch.long)


def _construct_variable_block_sizes(
    dit_seq_shape: Tuple[int, int, int],
    num_tiles: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    tT, tH, tW = tile_size
    nT, nH, nW = num_tiles

    bt = torch.arange(nT, device=device)
    bh = torch.arange(nH, device=device)
    bw = torch.arange(nW, device=device)
    valid_t = (T - bt * tT).clamp(max=tT)
    valid_h = (H - bh * tH).clamp(max=tH)
    valid_w = (W - bw * tW).clamp(max=tW)
    sizes = valid_t.view(nT, 1, 1) * valid_h.view(1, nH, 1) * valid_w.view(1, 1, nW)
    return sizes.reshape(-1).to(torch.long)


@dataclass
class VSAMetadata:
    """Per-timestep metadata required by the VSA sparse path."""

    current_timestep: int
    dit_seq_shape: Tuple[int, int, int]
    vsa_sparsity: float
    num_tiles: Tuple[int, int, int]
    total_seq_length: int
    padded_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor
    gather_idx: torch.LongTensor


class VSAMetadataBuilder:
    """Builds VSAMetadata; caches per-shape index tensors so torch.compile
    guards stay stable across denoising steps."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[Tuple[int, int, int], str], Dict[str, object]] = {}

    def _build_shape_payload(
        self,
        dit_seq_shape: Tuple[int, int, int],
        device: torch.device,
    ) -> Dict[str, object]:
        T, H, W = dit_seq_shape
        tT, tH, tW = VSA_TILE_SIZE
        num_tiles = (ceil(T / tT), ceil(H / tH), ceil(W / tW))
        total_seq_length = T * H * W
        padded_seq_length = num_tiles[0] * num_tiles[1] * num_tiles[2] * tT * tH * tW

        tile_partition_indices = _get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        non_pad_index = (tile_partition_indices >= 0).nonzero(as_tuple=True)[0]
        gather_idx = tile_partition_indices[non_pad_index]

        reverse = torch.zeros(total_seq_length, dtype=torch.long, device=device)
        reverse[gather_idx] = torch.arange(len(non_pad_index), dtype=torch.long, device=device)

        variable_block_sizes = _construct_variable_block_sizes(
            dit_seq_shape, num_tiles, VSA_TILE_SIZE, device
        )

        return {
            "dit_seq_shape": dit_seq_shape,
            "num_tiles": num_tiles,
            "total_seq_length": total_seq_length,
            "padded_seq_length": padded_seq_length,
            "tile_partition_indices": tile_partition_indices,
            "reverse_tile_partition_indices": reverse,
            "variable_block_sizes": variable_block_sizes,
            "non_pad_index": non_pad_index,
            "gather_idx": gather_idx,
        }

    def build(
        self,
        current_timestep: int,
        raw_latent_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        vsa_sparsity: float,
        device: torch.device,
    ) -> VSAMetadata:
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        cache_key = (dit_seq_shape, str(device))
        payload = self._cache.get(cache_key)
        if payload is None:
            payload = self._build_shape_payload(dit_seq_shape, device)
            self._cache[cache_key] = payload

        return VSAMetadata(
            current_timestep=current_timestep,
            vsa_sparsity=vsa_sparsity,
            **payload,  # type: ignore[arg-type]
        )


_vsa_forward_context_var: contextvars.ContextVar[Optional[VSAMetadata]] = contextvars.ContextVar(
    "_vsa_forward_context", default=None
)


@contextmanager
def set_vsa_forward_context(metadata: VSAMetadata):
    token = _vsa_forward_context_var.set(metadata)
    try:
        yield
    finally:
        _vsa_forward_context_var.reset(token)


def get_vsa_forward_context() -> Optional[VSAMetadata]:
    return _vsa_forward_context_var.get(None)


def _mean_pool_cubes(
    x_tiled: torch.Tensor,
    variable_block_sizes: torch.LongTensor,
    prod_tile: int,
    num_cubes: int,
) -> torch.Tensor:
    B, _padded, H, D = x_tiled.shape
    x_cubes = x_tiled.view(B, num_cubes, prod_tile, H, D)
    # fp32 accumulation: bf16 sum over 64 tokens perturbs the coarse softmax.
    x_sum = x_cubes.float().sum(dim=2)
    valid_counts = variable_block_sizes.float().clamp(min=1).view(1, num_cubes, 1, 1)
    return (x_sum / valid_counts).to(x_tiled.dtype)


class VSAPreprocessor:
    """Reorders NHD tokens into tile-major layout and zero-pads to tile boundaries."""

    @staticmethod
    def tile(
        x: torch.Tensor,
        non_pad_index: torch.LongTensor,
        gather_idx: torch.LongTensor,
        padded_seq_len: int,
    ) -> torch.Tensor:
        # index_select + index_copy_ instead of chained advanced indexing so
        # torch.compile can trace this without a graph break.
        B, _S, H, D = x.shape
        x_valid = x.index_select(1, gather_idx)
        x_padded = x.new_zeros(B, padded_seq_len, H, D)
        x_padded.index_copy_(1, non_pad_index, x_valid)
        return x_padded

    @staticmethod
    def untile(
        x: torch.Tensor,
        reverse_tile_partition_indices: torch.LongTensor,
        non_pad_index: torch.LongTensor,
    ) -> torch.Tensor:
        return x.index_select(1, non_pad_index).index_select(1, reverse_tile_partition_indices)


class VSAAttention(AttentionBackend):
    """
    Video Sparse Attention (VSA) backend for diffusion models.

    Implements coarse mean-pool + fine block-sparse top-K attention.
    The fine branch uses a JIT-compiled CuTe kernel on sm100+ for
    head_dim=128 / fp16-bf16; otherwise falls back to dense SDPA.

    Requires an active VSA forward context (set_vsa_forward_context) during
    each forward call. Does not support LSE output.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        sparse_attention_config=None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.num_kv_heads == self.num_heads, (
            f"VSA coarse mean-pool assumes MHA (num_kv_heads == num_heads), "
            f"got num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}. "
            f"GQA/MQA is not supported."
        )
        self.dtype = dtype
        self.sparse_attention_config = sparse_attention_config

    # Dynamo can't guard on the module-level mutable global, so this read
    # runs in eager.
    @torch.compiler.disable
    def _get_vsa_inputs(self):
        ctx: Optional[VSAMetadata] = get_vsa_forward_context()
        if ctx is None:
            raise RuntimeError(
                "VSAAttention.forward called without an active VSA forward context. "
                "Wrap each transformer call with set_vsa_forward_context()."
            )
        return (
            ctx.non_pad_index,
            ctx.gather_idx,
            ctx.reverse_tile_partition_indices,
            ctx.variable_block_sizes,
            ctx.padded_seq_length,
            ctx.num_tiles[0] * ctx.num_tiles[1] * ctx.num_tiles[2],
            ctx.vsa_sparsity,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        gate_compress: Optional[torch.Tensor] = None,
        gate_fine: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        VSA forward: coarse mean-pool + fine block-sparse top-K.

        Args:
            q, k, v: [B, S, H, D] in original (un-tiled) token order.
            gate_compress: [B, S, H, D] G_c gate weighting the coarse branch O_c.
            gate_fine: Optional [B, S, H, D] G_f gate weighting the fine branch
                O_f. None means constant 1 (dense behavior preserved).

        Returns:
            [B, S, H, D] in the same original token order.
        """
        if gate_compress is None:
            raise ValueError(
                "VSAAttention requires gate_compress. "
                "Ensure to_gate_compress is wired in the transformer block."
            )

        (
            non_pad_index,
            gather_idx,
            reverse_tile_partition_indices,
            variable_block_sizes,
            padded_len,
            num_cubes,
            vsa_sparsity,
        ) = self._get_vsa_inputs()

        B, S, H, D = q.shape
        prod_tile = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]
        cur_topk = max(1, ceil((1.0 - vsa_sparsity) * num_cubes))

        q_t = VSAPreprocessor.tile(q, non_pad_index, gather_idx, padded_len)
        k_t = VSAPreprocessor.tile(k, non_pad_index, gather_idx, padded_len)
        v_t = VSAPreprocessor.tile(v, non_pad_index, gather_idx, padded_len)

        q_c = _mean_pool_cubes(q_t, variable_block_sizes, prod_tile, num_cubes)
        k_c = _mean_pool_cubes(k_t, variable_block_sizes, prod_tile, num_cubes)
        v_c = _mean_pool_cubes(v_t, variable_block_sizes, prod_tile, num_cubes)

        scale = D**-0.5
        scores_c = torch.einsum("bnhd,bmhd->bhnm", q_c, k_c) * scale
        attn_probs_c = scores_c.softmax(dim=-1)
        o_c = torch.einsum("bhnm,bmhd->bnhd", attn_probs_c, v_c)

        use_cute = (
            _vsa_import_error is None
            and is_cute_supported(q)
            and (q.dtype == k.dtype == v.dtype)
            and num_cubes <= VSA_KERNEL_MAX_CUBES
        )
        topk_indices = attn_probs_c.topk(cur_topk, dim=-1).indices.to(torch.int32)

        o_c_tiled = (
            o_c.unsqueeze(2).expand(B, num_cubes, prod_tile, H, D).reshape(B, padded_len, H, D)
        )

        if use_cute:
            q_hnd = q_t.transpose(1, 2).contiguous()
            k_hnd = k_t.transpose(1, 2).contiguous()
            v_hnd = v_t.transpose(1, 2).contiguous()
            q2k_num = torch.full((B, H, num_cubes), cur_topk, dtype=torch.int32, device=q.device)
            o_hnd, _lse = block_sparse_attn_from_indices_cute(
                q_hnd,
                k_hnd,
                v_hnd,
                q2k_idx=topk_indices.contiguous(),
                q2k_num=q2k_num,
                variable_block_sizes=variable_block_sizes.to(torch.int32),
            )
            o_f_tiled = o_hnd.transpose(1, 2)

            # Padded rows hold kernel garbage; zero-padded gates mask the coarse
            # term and untile discards padded positions from both branches.
            gate_c_t = VSAPreprocessor.tile(gate_compress, non_pad_index, gather_idx, padded_len)
            if gate_fine is not None:
                gate_f_t = VSAPreprocessor.tile(gate_fine, non_pad_index, gather_idx, padded_len)
                combined_tiled = gate_c_t * o_c_tiled + gate_f_t * o_f_tiled
            else:
                combined_tiled = gate_c_t * o_c_tiled + o_f_tiled
            return VSAPreprocessor.untile(
                combined_tiled, reverse_tile_partition_indices, non_pad_index
            )

        # SDPA must run on the un-tiled Q/K/V — padded zero K/V slots would
        # otherwise absorb softmax mass and pollute the output. Untile o_c so
        # both branches combine in original-flat order.
        o_c_full = VSAPreprocessor.untile(o_c_tiled, reverse_tile_partition_indices, non_pad_index)
        o_f = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        if gate_fine is not None:
            return gate_compress * o_c_full + gate_fine * o_f
        return gate_compress * o_c_full + o_f

    @classmethod
    def support_lse(cls) -> bool:
        return False

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
