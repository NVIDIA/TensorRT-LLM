# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch.nn import functional as F

from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.params import SparseRuntimeParams
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.bindings.internal import thop

__all__ = ["TrtllmContextWorkspace", "trtllm_context_attention"]


class TrtllmContextWorkspace:
    """Caller-owned native scratch space for a batch of equal-length sequences."""

    def __init__(self, batch: int, length: int, device: torch.device):
        self.batch = batch
        self.length = length
        host_lengths = torch.full((batch,), length, dtype=torch.int32, device="cpu")
        self.device_lengths = host_lengths.to(device)
        self.host_totals = torch.tensor([batch * length, 0], dtype=torch.int32, device="cpu")
        self.scratch = torch.empty(0, dtype=torch.int8, device=device)
        self.kv_scale = torch.ones(1, dtype=torch.float32, device=device)
        # The native FMHA reader consumes the same metadata schema for every phase.
        self.meta = SimpleNamespace(
            max_num_requests=batch,
            max_num_sequences=None,
            effective_beam_width=1,
            max_seq_len=length,
            max_context_length=length,
            max_context_q_len_override=None,
            kv_lens_runtime=host_lengths,
            prompt_lens_cpu_runtime=host_lengths,
            prompt_lens_cuda_runtime=self.device_lengths,
            is_cross=False,
            cache_indirection=None,
            is_spec_decoding_enabled=False,
            is_spec_dec_tree=False,
            force_prepare_spec_dec_tree_mask=False,
            max_total_draft_tokens=None,
            trtllm_gen_jit_warmup=False,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            kv_cache_block_offsets=None,
            block_ids_per_seq=None,
            flash_mla_tile_scheduler_metadata=None,
            flash_mla_num_splits=None,
            use_paged_context_fmha=False,
            num_sparse_topk=0,
            host_kv_cache_pool_pointers=None,
            host_kv_cache_pool_mapping=None,
        )


class _ContextLayer(SimpleNamespace):
    rope_params = RopeParams()
    q_scaling = 1.0
    predicted_tokens_per_seq = 1
    is_mla_enable = False
    skip_correction_threshold = 0.0
    uses_spcompress = False
    q_lora_rank = None
    kv_lora_rank = None
    qk_nope_head_dim = None
    qk_rope_head_dim = None
    rope_append = None
    rotary_inv_freq = None
    rotary_cos_sin = None
    position_embedding_type = 0
    attention_chunk_size = None


def _validate_inputs(q, k, v, workspace, skip_stats, num_heads, sage_block_sizes, sage_flags):
    batch, length, heads, dim = q.shape
    if (batch, length) != (
        workspace.batch,
        workspace.length,
    ) or q.device != workspace.scratch.device:
        raise ValueError(
            "Context attention workspace must match the input batch, length and device."
        )
    if (
        skip_stats.device != q.device
        or skip_stats.dtype != torch.uint32
        or skip_stats.numel() != 2
        or not skip_stats.is_contiguous()
    ):
        raise ValueError(
            "Skip-softmax statistics require two contiguous uint32 counters on the input device."
        )
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or dim == 0:
        raise ValueError(
            "Context attention requires floating-point Q/K/V with a positive head dimension."
        )

    if sage_block_sizes == (0, 0, 0):
        if k is not None or v is not None:
            raise ValueError("Dense context attention requires packed QKV.")
        if sage_flags:
            raise ValueError("Sage Q/K quantization options require explicit Sage block sizes.")
        kv_heads, remainder = divmod(heads - num_heads, 2)
        if remainder:
            raise ValueError("Packed QKV must contain the same number of K and V heads.")
    else:
        if k is None or v is None:
            raise ValueError("Sage context attention requires separate Q, K and V tensors.")
        q_block, k_block, v_block = sage_block_sizes
        if q_block <= 0 or k_block <= 0 or v_block != 1:
            raise ValueError("Sage requires positive Q and K block sizes and a V block size of 1.")
        if q.dtype not in (torch.bfloat16, torch.float16) or dim not in (64, 128, 256):
            raise ValueError("Sage Q/K/V require BF16 or FP16 with head dimension 64, 128 or 256.")
        if k.ndim != 4 or k.shape != v.shape or k.shape[:2] != (batch, length) or k.shape[3] != dim:
            raise ValueError(
                "Sage K and V must share shape [B,S,Hkv,D] with the batch, sequence and head dimension of Q."
            )
        if k.dtype != q.dtype or v.dtype != q.dtype or k.device != q.device or v.device != q.device:
            raise ValueError("Sage Q, K and V must share a dtype and device.")
        if num_heads != heads:
            raise ValueError("num_heads must match Q's head count for separate Q/K/V.")
        kv_heads = k.shape[2]
    if num_heads <= 0 or kv_heads <= 0 or num_heads % kv_heads != 0:
        raise ValueError(
            "The positive query head count must be a multiple of the positive KV head count."
        )
    return kv_heads


class TrtllmContextAttention(OpWrapper):
    """Full uncached attention with 1/sqrt(D) scaling on equal-length sequences.

    Dense attention takes packed [B,S,Hq+2*Hkv,D] QKV. Sage takes separate BSHD
    tensors. The output is [B,S,Hq,D] in the query dtype. Builds with skip
    statistics enabled accumulate total and skipped blocks in the two uint32 counters.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Batched packed QKV", dict(batch=2, length=65, heads=56, kv_heads=56)),
        Cell("Short packed QKV", dict(batch=1, length=5, heads=56, kv_heads=56)),
        Cell("Packed grouped-query heads", dict(batch=1, length=73, heads=8, kv_heads=2)),
        Cell(
            "Low-score tiles skipped",
            dict(batch=1, length=512, heads=4, kv_heads=4, skip_threshold=8.0),
        ),
        Cell(
            "Separate Q/K/V with block-scaled FP8 and centered K",
            dict(batch=1, length=257, heads=4, kv_heads=4, sage=True),
        ),
    )

    @torch.compiler.disable
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        workspace: TrtllmContextWorkspace,
        skip_stats: torch.Tensor,
        *,
        num_heads: int,
        layer_idx: int = 0,
        quant_mode: int = 0,
        skip_threshold: float | None = None,
        sage_block_sizes: tuple[int, int, int] = (0, 0, 0),
        sage_qk_int8: bool = False,
        sage_smooth_k: bool = False,
    ) -> torch.Tensor:
        num_kv_heads = _validate_inputs(
            q,
            k,
            v,
            workspace,
            skip_stats,
            num_heads,
            sage_block_sizes,
            sage_qk_int8 or sage_smooth_k,
        )
        batch, length, _, dim = q.shape
        rows = batch * length
        out = q.new_empty((batch, length, num_heads, dim))
        if rows == 0:
            return out
        layer = _ContextLayer(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=dim,
            v_head_dim=dim,
            quant_mode=quant_mode,
            skip_softmax_stat=skip_stats,
        )
        fwd = AttentionForwardArgs(
            output=out.view(rows, num_heads * dim),
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=length,
            is_fused_qkv=k is None,
            kv_scale_orig_quant=workspace.kv_scale,
            kv_scale_quant_orig=workspace.kv_scale,
            sage_attn_num_elts_per_blk_q=sage_block_sizes[0],
            sage_attn_num_elts_per_blk_k=sage_block_sizes[1],
            sage_attn_num_elts_per_blk_v=sage_block_sizes[2],
            sage_attn_qk_int8=sage_qk_int8,
            sage_attn_smooth_k=sage_smooth_k,
            sparse_runtime_params=SparseRuntimeParams(
                threshold_scale_factor_prefill=skip_threshold or 0.0
            ),
        )
        params = FmhaParams(
            attn=layer,
            meta=workspace.meta,
            fwd=fwd,
            workspace=workspace.scratch,
            qkv_input=q.reshape(rows, -1).contiguous(),
            key_input=k.reshape(rows, -1).contiguous() if k is not None else None,
            value_input=v.reshape(rows, -1).contiguous() if v is not None else None,
            output=out.view(rows, num_heads * dim),
            sequence_lengths=workspace.device_lengths,
            context_lengths=workspace.device_lengths,
            host_total_kv_lens=workspace.host_totals,
            input_seq_length=length,
            max_past_kv_length=length,
            max_attention_window_size=length,
            cyclic_attention_window_size=length,
            num_tokens=rows,
            tokens_per_block=0,
            kv_factor=1,
            batch_size=batch,
            num_requests=batch,
            local_layer_idx=layer_idx,
        )
        thop.run_context(params)
        return out

    def reference(
        self,
        q,
        k,
        v,
        workspace,
        skip_stats,
        *,
        num_heads,
        layer_idx=0,
        quant_mode=0,
        skip_threshold=None,
        sage_block_sizes=(0, 0, 0),
        sage_qk_int8=False,
        sage_smooth_k=False,
    ):
        """FP32 full attention before optional quantization or tile skipping."""
        if k is None:
            kv_heads = (q.shape[2] - num_heads) // 2
            q, k, v = q.split((num_heads, kv_heads, kv_heads), dim=2)
        result = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            enable_gqa=num_heads != k.shape[2],
        )
        return result.transpose(1, 2).to(q.dtype)

    def is_valid(
        self,
        q,
        k,
        v,
        workspace,
        skip_stats,
        *,
        num_heads,
        layer_idx=0,
        quant_mode=0,
        skip_threshold=None,
        sage_block_sizes=(0, 0, 0),
        sage_qk_int8=False,
        sage_smooth_k=False,
    ):
        _validate_inputs(
            q,
            k,
            v,
            workspace,
            skip_stats,
            num_heads,
            sage_block_sizes,
            sage_qk_int8 or sage_smooth_k,
        )


trtllm_context_attention = TrtllmContextAttention()
