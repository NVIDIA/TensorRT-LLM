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
CuTe DSL Backend for Visual Generation Models

CuTeDSLAttention runs the VSA sparse path when sparse_attention_config is set,
otherwise the dense cubin path (with optional QK16PV8 quantization).
Expects NHD layout ([B, S, H, D]) and supports float16/bfloat16.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_cute_dsl_import_error = None
try:
    import tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention as cute_dsl
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention.fmha import (
        _cute_runtime_import_error,
    )

    if _cute_runtime_import_error is not None:
        raise ImportError(_cute_runtime_import_error)
except (ImportError, OSError) as e:
    cute_dsl = None
    _cute_dsl_import_error = e


# VSA (Video Sparse Attention) sparse-path helpers

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


_vsa_forward_context: Optional[VSAMetadata] = None


@contextmanager
def set_vsa_forward_context(metadata: VSAMetadata):
    global _vsa_forward_context
    prev = _vsa_forward_context
    _vsa_forward_context = metadata
    try:
        yield
    finally:
        _vsa_forward_context = prev


def get_vsa_forward_context() -> Optional[VSAMetadata]:
    return _vsa_forward_context


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


class CuTeDSLAttention(AttentionBackend):
    """
    CuTe DSL (NVIDIA kernels) backend for diffusion models.

    Dense path uses pre-compiled cubins and requires head_dim=128. The VSA
    sparse path (sparse_attention_config set) uses a JIT-compiled CuTe kernel
    when head_dim=128 / fp16-bf16 / sm100+, and otherwise falls back to dense SDPA.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        sparse_attention_config=None,
        skip_softmax_threshold_scale: Optional[float] = None,
        **kwargs,
    ):
        # Dense path requires head_dim=128 (packaged cubins); the VSA sparse
        # path JIT-compiles per shape, so it has no such restriction.
        if sparse_attention_config is None and head_dim != 128:
            raise ValueError(f"CUTEDSL cubins require head_dim=128, got head_dim={head_dim}.")
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self.sparse_attention_config = sparse_attention_config
        self.skip_softmax_threshold_scale = skip_softmax_threshold_scale
        self.scale = 1.0 / math.sqrt(head_dim)

        # CuTe DSL expects [B, S, H, D] format
        self._preferred_layout = AttentionTensorLayout.NHD

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.dtype]:
        """Cast inputs to CuTeDSL-compatible dtype and resolve causal flag."""
        if _cute_dsl_import_error is not None:
            raise ImportError(
                f"CuTe DSL kernels are not available. Import error: {_cute_dsl_import_error}"
            ) from _cute_dsl_import_error

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        # Packaged cubins support float16 and bfloat16 only.
        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        return q, k, v, is_causal, origin_dtype

    # cute_dsl.cute_dsl_fmha_fwd is already decorated with @torch.compiler.disable
    # Allow torch.compile to fuse preceding linear/norm with quantization of V / seq-preprocess
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, num_heads, _ = q.shape
        _, seq_len_kv, _, value_head_dim = v.shape
        out = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            value_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            dtype=torch.float32,
            device=q.device,
        )

        # Options that instructs quantization of V
        scale_v = kwargs.get("scale_v", 1.0)
        if self.quant_attention_config is not None:
            v_qscale = 448.0 / v.abs().amax().clamp(min=1e-3)
            v = (v * v_qscale).to(torch.float8_e4m3fn)
            scale_v = scale_v / v_qscale

        # Sequence preproc.
        qo_indptr_host = [i * seq_len_q for i in range(batch_size + 1)]
        qo_indptr = torch.tensor(qo_indptr_host).to(device=q.device, dtype=torch.int32)
        kv_indptr_host = [i * seq_len_kv for i in range(batch_size + 1)]
        kv_indptr = torch.tensor(kv_indptr_host).to(device=q.device, dtype=torch.int32)

        # Skip softmax.
        skip_softmax_threshold_scale = self.skip_softmax_threshold_scale
        if skip_softmax_threshold_scale is not None and skip_softmax_threshold_scale <= 0.0:
            skip_softmax_threshold_scale = None

        cute_dsl.cute_dsl_fmha_fwd(
            q.flatten(0, 1).contiguous(),
            k.flatten(0, 1).contiguous(),
            v.flatten(0, 1).contiguous(),
            out.flatten(0, 1),
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse.flatten(0, 1).contiguous(),
            scale_q=kwargs.get("scale_q", 1.0),
            scale_k=kwargs.get("scale_k", 1.0),
            scale_v=scale_v,
            scale_o=kwargs.get("scale_o", 1.0),
            max_qo_len=seq_len_q,
            max_kv_len=seq_len_kv,
            is_persistent=False,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale,
        )
        return out, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        gate_compress: Optional[torch.Tensor] = None,
        gate_fine: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using CuTe DSL (NVIDIA kernels).

        Dimensions are derived from tensor shapes (NHD layout: [B, S, H, D]).
        Dispatches to _forward_vsa when sparse_attention_config is set
        (VSA sparse path); otherwise runs the dense cubins via forward_with_lse.

        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            v: Value tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            attention_mask: Attention mask type (CAUSAL or FULL) — dense path only.
            gate_compress: VSA path only — G_c gate for the coarse branch.
            gate_fine: VSA path only — G_f gate for the fine branch. None means
                constant 1.

        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
        if self.sparse_attention_config is not None:
            return self._forward_vsa(q, k, v, gate_compress=gate_compress, gate_fine=gate_fine)
        output, _ = self.forward_with_lse(q, k, v, attention_mask=attention_mask, **kwargs)
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and log-sum-exp (LSE). Dense path
        only — the VSA sparse path does not produce an LSE.

        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
            lse:    [batch_size, num_heads, seq_len] - log-sum-exp per query position,
                    always in float32. Used for numerically stable combination of
                    partial attention results in Attention2D parallelism.
        """
        if self.sparse_attention_config is not None:
            raise RuntimeError(
                "CuTeDSLAttention.forward_with_lse() does not support the VSA "
                "sparse path. Use forward() instead, or construct without "
                "sparse_attention_config to use the dense path."
            )
        q, k, v, is_causal, origin_dtype = self._prepare_inputs(q, k, v, attention_mask)
        output, lse = self._fwd(q, k, v, is_causal, **kwargs)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output, lse.transpose(1, 2)

    # Dynamo can't guard on the module-level mutable global, so this read
    # runs in eager.
    @torch.compiler.disable
    def _get_vsa_inputs(self):
        ctx: Optional[VSAMetadata] = get_vsa_forward_context()
        if ctx is None:
            raise RuntimeError(
                "CuTeDSLAttention._forward_vsa called without an active VSA forward context. "
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

    def _forward_vsa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        gate_compress: Optional[torch.Tensor],
        gate_fine: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """VSA forward: coarse mean-pool + fine block-sparse top-K.

        Args:
            q, k, v: [B, S, H, D] in original (un-tiled) token order.
            gate_compress: [B, S, H, D] G_c gate weighting the coarse branch O_c.
            gate_fine: Optional [B, S, H, D] G_f gate weighting the fine branch
                O_f. None means constant 1 (dense behavior preserved).

        Returns:
            [B, S, H, D] in the same original token order.
        """
        # Lazy import: the VSA kernels package is optional and may not be
        # importable in environments without the cute-dsl runtime.
        from ..cute_dsl_kernels.blackwell.video_sparse_attention import (
            block_sparse_attn_from_indices_cute,
            is_cute_supported,
        )

        if gate_compress is None:
            raise ValueError(
                "CuTeDSLAttention VSA path requires gate_compress. "
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

        use_cute = is_cute_supported(q) and (q.dtype == k.dtype == v.dtype)
        topk_indices = attn_probs_c.topk(cur_topk, dim=-1).indices.to(torch.int32)

        o_c_tiled = (
            o_c.unsqueeze(2).expand(B, num_cubes, prod_tile, H, D).reshape(B, padded_len, H, D)
        )

        if use_cute:
            assert num_cubes <= VSA_KERNEL_MAX_CUBES, (
                f"VSA CuTe kernel supports at most {VSA_KERNEL_MAX_CUBES} cubes "
                f"(SMEM-allocated variable_block_sizes buffer); got num_cubes={num_cubes}. "
                "Lower video resolution/length or fall back to dense SDPA."
            )
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
        return True

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
