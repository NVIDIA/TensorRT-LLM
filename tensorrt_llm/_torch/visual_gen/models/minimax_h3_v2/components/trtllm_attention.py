# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uncached full and block-sparse TRTLLM attention for H3."""

import math
import os

import torch
from torch.nn import functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseBackendForwardArgs,
)
from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig
from tensorrt_llm.visual_gen.sparse_attention import SolAttentionConfig

from ....catalog.attention.trtllm_context_attention import (
    TrtllmContextWorkspace,
    trtllm_context_attention,
)
from .skip_softmax_threshold import H3SkipSoftmax, dense_prefix
from .trtllm_attention_prims import H3PrimTS


@torch.compile
def _pack(q, k, v):
    return torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1).flatten(0, 1)


@torch.compile
def _compact(q, k, v):
    return q.contiguous(), k.contiguous(), v.contiguous()


class H3TrtllmAttention:
    """Full self-attention on NHD activations with one host prefix length per batch."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        attention_metadata_state: dict,
        layer_idx: int = 0,
        num_kv_heads: int | None = None,
        quant_config: QuantConfig | None = None,
        dtype: torch.dtype | None = None,
        attention_config: AttentionConfig | None = None,
        sparse_params: SolAttentionConfig | H3SkipSoftmax | None = None,
    ):
        if attention_metadata_state is None:
            raise ValueError("TRTLLM attention requires model-scoped attention metadata state.")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.quant_mode = int(quant_config.layer_quant_mode) if quant_config is not None else 0
        self.recipe = attention_config.quant_attention_config if attention_config else None
        self.sparse = sparse_params
        self.sol = isinstance(sparse_params, SolAttentionConfig)
        self.skip_softmax = isinstance(sparse_params, H3SkipSoftmax)
        self.state = attention_metadata_state
        self.prims = H3PrimTS(self.recipe, attention_metadata_state)
        self.skip_stats = None
        self.print_stats = os.getenv("TRTLLM_PRINT_SKIP_SOFTMAX_STAT") == "1"

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    def support_fused_qkv(self) -> bool:
        return self.recipe is None and not self.sol

    @torch.compiler.disable
    def _skip_threshold(self, timestep):
        assert isinstance(self.sparse, H3SkipSoftmax)
        return (
            0.0
            if dense_prefix(timestep, self.sparse.disabled_until_timestep)
            else self.sparse.threshold
        )

    @torch.compiler.disable
    def _sol_routes(self, q, k, v, timestep):
        from ....catalog.attention.sol_predictor import SOL_BLOCK_SIZE, sol_predictor

        assert isinstance(self.sparse, SolAttentionConfig)
        if self.layer_idx in (self.sparse.dense_layers or ()):
            return None
        if dense_prefix(timestep, self.sparse.disabled_until_timestep):
            return None
        bits, keys, values = sol_predictor(
            q,
            k,
            v,
            tau=self.sparse.tau,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            thresh_type=self.sparse.thresh_type,
        )
        return BlockSparseForwardInputs(
            q_block_size=SOL_BLOCK_SIZE,
            kv_block_size=SOL_BLOCK_SIZE,
            exact_block_bits=bits,
            k_summary=keys,
            v_summary=values,
        )

    @torch.compiler.disable
    def _thop(self, q, k, v, batch, length, threshold):
        if not self.prims.fallback_enabled:
            raise RuntimeError("Uncached H3 context attention requires the fallback FMHA library.")
        cache = self.state.setdefault("h3_thop", {})
        key = (q.device, batch, length)
        if key not in cache:
            cache[key] = TrtllmContextWorkspace(batch, length, q.device)
        if self.skip_stats is None or self.skip_stats.device != q.device:
            self.skip_stats = q.new_zeros((2,), dtype=torch.uint32)
        if self.print_stats:
            self.skip_stats.zero_()
        recipe = self.recipe
        packed_heads = self.num_heads + (2 * self.num_kv_heads if k is None else 0)
        out = trtllm_context_attention(
            q.view(batch, length, packed_heads, self.head_dim),
            k.view(batch, length, self.num_kv_heads, self.head_dim) if k is not None else None,
            v.view(batch, length, self.num_kv_heads, self.head_dim) if v is not None else None,
            cache[key],
            self.skip_stats,
            num_heads=self.num_heads,
            layer_idx=self.layer_idx,
            quant_mode=self.quant_mode,
            skip_threshold=threshold,
            sage_block_sizes=(recipe.q_block_size, recipe.k_block_size, recipe.v_block_size)
            if recipe
            else (0, 0, 0),
            sage_qk_int8=recipe is not None and recipe.qk_dtype == "int8",
            sage_smooth_k=recipe is not None and recipe.smooth_k,
        )
        if self.print_stats:
            total, skipped = self.skip_stats.tolist()
            if total:
                print(
                    f"SKIP_SOFTMAX_STAT: layer{self.layer_idx}: {skipped}/{total} = {skipped / total:.2f}%"
                )
        return out.flatten(2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        *,
        batch_size: int,
        seq_len: int,
        seq_len_kv: int | None = None,
        valid_seq_len: int | None = None,
        discard_padded_queries: bool = False,
        key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        sparse_backend_args: SparseBackendForwardArgs | None = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        if seq_len_kv not in (None, seq_len) or attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("H3 TRTLLM attention requires full self-attention.")
        if discard_padded_queries and (valid_seq_len is not None or key_padding_mask is not None):
            if (
                self.recipe is not None
                or (self.sparse is not None and not self.sol)
                or sparse_backend_args is not None
            ):
                raise ValueError(
                    "TRTLLM padding requires dense self-attention with discarded padded queries."
                )
            if type(valid_seq_len) is not int or not 0 <= valid_seq_len <= seq_len:
                raise ValueError("valid_seq_len must be a host integer in [0, sequence length].")
            if valid_seq_len < seq_len:
                if valid_seq_len == 0:
                    return q.new_zeros((batch_size, seq_len, self.num_heads * self.head_dim))
                out = self.forward(
                    q[:, :valid_seq_len].contiguous(),
                    k[:, :valid_seq_len].contiguous() if k is not None else None,
                    v[:, :valid_seq_len].contiguous() if v is not None else None,
                    batch_size=batch_size,
                    seq_len=valid_seq_len,
                    timestep=timestep,
                )
                return F.pad(out, (0, 0, 0, seq_len - valid_seq_len))
        threshold = self._skip_threshold(timestep) if self.skip_softmax else None
        routes = sparse_backend_args.block_sparse_inputs if sparse_backend_args else None
        if self.sol or routes is not None or self.recipe is not None:
            if k is None or v is None:
                raise ValueError("Quantized and sparse H3 attention require separate Q, K and V.")
            q, k, v = _compact(q, k, v)
            if self.sol:
                routes = self._sol_routes(q, k, v, timestep)
            if routes is not None or self.recipe is not None:
                out = self.prims(
                    q,
                    k,
                    v,
                    routes,
                    skip_softmax=self.skip_softmax,
                    skip_threshold_active=bool(threshold),
                )
                if out is not None:
                    return out.flatten(2)
                if routes is not None:
                    raise RuntimeError(
                        "H3 block-sparse attention requires PrimTS block-sparse FMHA."
                    )
            if self.recipe:
                return self._thop(
                    q.flatten(0, 1).flatten(1),
                    k.flatten(0, 1).flatten(1),
                    v.flatten(0, 1).flatten(1),
                    batch_size,
                    seq_len,
                    threshold,
                )
        qkv = q.reshape(batch_size * seq_len, -1) if k is None else _pack(q, k, v)
        return self._thop(qkv, None, None, batch_size, seq_len, threshold)
