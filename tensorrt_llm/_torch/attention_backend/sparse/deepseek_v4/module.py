# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-specific orchestration for the generic MLA module."""

import os
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.sparse.mla import (
    forward_context_sparse_mla,
    forward_generation_sparse_mla,
)
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._utils import is_sm_100f


def _fp8_block_scaling_bmm_out(*args, **kwargs):
    from tensorrt_llm._torch.modules.mla import fp8_block_scaling_bmm_out

    return fp8_block_scaling_bmm_out(*args, **kwargs)


_q_b_proj_cute_dsl_import_ok: Optional[bool] = None


def q_b_proj_cute_dsl_bf16(q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BF16 dense GEMM via CuTe DSL.

    Computes ``q @ weight.T`` for [M, K] @ [N, K]^T -> [M, N].

    Delegates to ``torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell`` (which
    runs its own autotune over (use_2cta, mma_tiler, cluster_shape)). Falls
    back to ``torch.nn.functional.linear`` if CuTe DSL is unavailable.
    """
    global _q_b_proj_cute_dsl_import_ok
    if _q_b_proj_cute_dsl_import_ok is None:
        try:
            from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

            _q_b_proj_cute_dsl_import_ok = IS_CUTLASS_DSL_AVAILABLE
        except ImportError:
            _q_b_proj_cute_dsl_import_ok = False
    if not _q_b_proj_cute_dsl_import_ok or not is_sm_100f():
        return torch.nn.functional.linear(q, weight)

    assert q.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
        "q_b_proj cute_dsl path requires bfloat16 inputs"
    )
    q = q.contiguous()
    weight = weight.contiguous()
    m, n = q.shape[0], weight.shape[0]
    out = q.new_empty((m, n), dtype=torch.bfloat16)
    torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell(q, weight, out)
    return out


def is_fused_q_fp8_quant_enabled(self, num_generations: int = 0) -> bool:
    # Context-only batches: the fused path leaves a placeholder bf16 q_buf
    # that forward_generation_sparse_mla would read uninitialized, so
    # mixed/gen batches must take the legacy unfused path.
    # `TRTLLM_DISABLE_FUSED_Q_FP8_QUANT=1` opts back into the legacy
    # two-kernel Q-quant path as a kill switch.
    if os.environ.get("TRTLLM_DISABLE_FUSED_Q_FP8_QUANT", "0") == "1":
        return False
    if not self.is_deepseek_v4:
        return False
    if self.qk_head_dim != 512 or self.kv_lora_rank != 448:
        return False
    if num_generations > 0:
        return False
    return bool(getattr(self.mqa, "has_fp8_kv_cache", False))


def deepseek_v4_q_b_layernorm_fused_fp8(self, q_proj: torch.Tensor):
    # Returns (placeholder_q, quant_q_buffer, q_pe, quant_scale_qkv).
    # `placeholder_q` keeps the [num_tokens, num_heads*head_dim] bf16 layout
    # the downstream `forward_absorption_context` needs for its `q.shape[0]`
    # check and `q.view().split()` call. Its contents are never read on the
    # fused FP8 path: the nope segment lives in `quant_q_buffer`, the rope
    # segment is passed in `q_pe`, and the split's `q_nope`/`q_pe` outputs
    # are either overridden by the caller or discarded by the DSv4 branch.
    # Reusing `q_proj` (q_b_proj output) avoids a ~num_tokens x hidden bf16
    # allocation per forward.
    assert q_proj.dim() == 2
    assert q_proj.shape[1] == self.num_heads_tp * self.qk_head_dim
    if getattr(self, "_quant_scale_qkv", None) is None:
        self._quant_scale_qkv = torch.tensor([1.0], dtype=torch.float32, device=q_proj.device)
    # q_pe is 3D so thop.attention's sparse-MLA context branch passes its
    # q_pe->dim() == 3 check; the kernel op consumes the flat 2D view.
    num_tokens = q_proj.shape[0]
    rope_dim = self.qk_head_dim - self.kv_lora_rank
    quant_q_buffer = q_proj.new_empty(
        (num_tokens, self.num_heads_tp * self.qk_head_dim), dtype=torch.float8_e4m3fn
    )
    q_pe = q_proj.new_empty((num_tokens, self.num_heads_tp, rope_dim))
    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q_proj,
        quant_q_buffer,
        q_pe.view(num_tokens, self.num_heads_tp * rope_dim),
        self.num_heads_tp,
        self.qk_head_dim,
        self.kv_lora_rank,
        float(self.q_b_layernorm.variance_epsilon),
        self._quant_scale_qkv,
    )
    # Both buffers must be live for the fused path; the downstream
    # absorption-context op switches on `quant_scale_qkv is not None`
    # to enable the C++ fusion (see trtllm.py `thop.attention` call).
    assert self._quant_scale_qkv is not None, (
        "fused FP8-Q quant requires _quant_scale_qkv to be set"
    )
    return q_proj, quant_q_buffer, q_pe, self._quant_scale_qkv


def deepseek_v4_q_b_layernorm(self, q: torch.Tensor) -> torch.Tensor:
    assert q.dim() == 2 and q.shape[1] == self.num_heads_tp * self.qk_head_dim
    return torch.ops.trtllm.deepseek_v4_q_norm(
        q, self.num_heads_tp, self.qk_head_dim, float(self.q_b_layernorm.variance_epsilon)
    )


def should_use_dsv4_epilogue_fusion(
    self, num_contexts: int, num_generations: int
) -> bool:
    if self._disable_dsv4_epilogue_fusion:
        return False
    if not self.is_deepseek_v4:
        return False
    if num_contexts == 0 and num_generations == 0:
        return False
    if num_contexts > 0 and num_generations > 0:
        # Context and generation use separate FMHA calls, but the fused
        # buffers do not carry token offsets for a mixed batch.
        return False
    if self.mapping.has_cp_helix():
        return False
    if not is_sm_100f():
        return False
    if not getattr(self.mapping, "enable_attention_dp", False):
        return False
    if self.num_heads != 128 or self.num_heads_tp != 128:
        return False
    if getattr(self.mqa, "sparse_params", None) is None:
        return False
    if not getattr(self.mqa, "has_fp8_kv_cache", False):
        return False
    if self.o_a_proj.dtype != torch.float8_e4m3fn:
        return False
    if self.kv_lora_rank != 448 or self.qk_rope_head_dim != 64:
        return False
    if self.qk_head_dim != 512 or self.v_head_dim != 512:
        return False
    if self.n_local_groups <= 0 or self.num_heads_tp % self.n_local_groups != 0:
        return False
    return not self.inverse_rotary_emb.is_neox


def create_dsv4_epilogue_buffers(
    self, q: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.n_local_groups <= 0 or self.num_heads_tp % self.n_local_groups != 0:
        raise ValueError(
            "DSv4 fused epilogue requires num_heads_tp to be divisible by n_local_groups."
        )
    heads_per_group = self.num_heads_tp // self.n_local_groups
    scale_buf_m = (num_tokens + 3) // 4 * 4
    fp8_o = q.new_empty(
        (self.n_local_groups, num_tokens, heads_per_group * self.v_head_dim),
        dtype=torch.float8_e4m3fn,
    )
    output_sf = q.new_empty(
        (
            self.n_local_groups,
            heads_per_group * (self.v_head_dim // 128),
            scale_buf_m,
        ),
        dtype=torch.float32,
    )
    return fp8_o, output_sf


def validate_dsv4_epilogue_buffers(
    self,
    num_tokens: int,
    dsv4_epilogue_output: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_o, output_sf = dsv4_epilogue_output
    scale_buf_m = (num_tokens + 3) // 4 * 4
    if fp8_o.shape[1] != num_tokens or output_sf.shape[2] != scale_buf_m:
        raise RuntimeError("Invalid DSv4 fused epilogue buffers for current token count.")
    return fp8_o, output_sf


def deepseek_v4_o_proj(
    self,
    attn_out_latent: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(attn_out_latent, tuple):
        attn_fp8, attn_scale = attn_out_latent
        num_tokens = attn_fp8.shape[1]
        o_lora = torch.empty(
            [num_tokens, self.n_local_groups, self.o_lora_rank],
            device=attn_fp8.device,
            dtype=self.dtype,
        )
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(
            attn_fp8,
            self.o_a_proj,
            attn_scale,
            self.o_a_proj_scale,
            o_lora.transpose(0, 1),
        )
        return self.o_b_proj(o_lora.flatten(1))

    assert position_ids is not None
    num_tokens = attn_out_latent.shape[0]
    attn_out_latent = attn_out_latent.view(num_tokens, self.num_heads_tp, -1)

    # When o_a_proj is FP8 on SM100 (which is always the case for DSv4
    # under FP8 block-scales after init), fuse the inverse-RoPE into the
    # FP8-quant epilogue (vLLM-ported Triton kernel) and call
    # cute_dsl_fp8_bmm_blackwell directly. Saves one BF16 read+write of
    # the latent vs the mla_rope_inplace +
    # fp8_batched_quantize_1x128_permute102 pair. Decoupled from
    # use_cute_dsl_blockscaling_bmm (which gates the separate K/V
    # absorption BMM kernel choice).
    fused_inv_rope_fp8 = self.o_a_proj.dtype == torch.float8_e4m3fn and is_sm_100f()
    if fused_inv_rope_fp8:
        heads_per_group = self.num_heads_tp // self.n_local_groups
        attn_fp8, attn_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
            attn_out_latent,
            position_ids.view(-1),
            self.inverse_rotary_emb.rotary_cos_sin,
            self.n_local_groups,
            heads_per_group,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            128,
            self.inverse_rotary_emb.is_neox,
        )
        o_lora = torch.empty(
            [num_tokens, self.n_local_groups, self.o_lora_rank],
            device=attn_out_latent.device,
            dtype=self.dtype,
        )
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(
            attn_fp8, self.o_a_proj, attn_scale, self.o_a_proj_scale, o_lora.transpose(0, 1)
        )
        o_lora = o_lora.flatten(1)
        return self.o_b_proj(o_lora)

    # Fused in-place inverse RoPE on the rope portion of each head
    torch.ops.trtllm.mla_rope_inplace(
        attn_out_latent,
        position_ids.view(-1),
        self.inverse_rotary_emb.rotary_cos_sin,
        self.num_heads_tp,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
        True,
        self.inverse_rotary_emb.is_neox,
    )

    # Output projections
    o_lora = torch.empty(
        [num_tokens, self.n_local_groups, self.o_lora_rank],
        device=attn_out_latent.device,
        dtype=attn_out_latent.dtype,
    )
    if self.o_a_proj.dtype == torch.bfloat16:
        # dim = head_dim * num_head // num_group
        # [num_groups, num_tokens, dim] x [num_groups, dim, o_lora_rank]
        # -> [num_groups, num_tokens, o_lora_rank]
        torch.ops.trtllm.bmm_out(
            attn_out_latent.view(num_tokens, self.n_local_groups, -1).transpose(0, 1),
            self.o_a_proj.transpose(1, 2),
            o_lora.transpose(0, 1),
        )
    elif self.o_a_proj.dtype == torch.float8_e4m3fn:
        _fp8_block_scaling_bmm_out(
            attn_out_latent.view(num_tokens, self.n_local_groups, -1),
            self.o_a_proj,
            self.o_a_proj_scale,
            o_lora.transpose(0, 1),
            self.o_a_proj_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.o_a_proj.dtype}.")
    o_lora = o_lora.flatten(1)
    output = self.o_b_proj(o_lora)
    return output


def forward_impl_with_deepseek_v4(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    dsv4_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    """
    Forward pass for the MLA module with DeepSeek-V4 (always in MQA mode).

    Args:
        position_ids (Optional[torch.IntTensor]): The position IDs.
        hidden_states (torch.Tensor): The hidden states.
        attn_metadata (AttentionMetadata): The attention metadata.

        output (torch.Tensor): Pre-allocated output tensor, written in-place
            when epilogue fusion is disabled.
        dsv4_epilogue_output: Caller-provided ``(fp8_o, output_sf)``
            buffers, written in-place when epilogue fusion is enabled.
    """
    assert self.mha is None and self.mqa is not None, "DeepSeek-V4 is only supported in MQA mode"
    # split q, k, v into context and gen batches
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    enable_dsv4_epilogue_fusion = dsv4_epilogue_output is not None
    if enable_dsv4_epilogue_fusion and ((num_contexts > 0) == (num_generations > 0)):
        raise RuntimeError(
            "DSv4 epilogue fusion requires a context-only or generation-only batch."
        )

    hidden_states = hidden_states[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    # TRTLLM_MLA_EXTRA_OVERLAP=1 reorders the V4 attention prologue so the
    # outer compressor and the ratio-4 indexer can execute concurrently
    # with q_b_proj + q_b_layernorm. The indexer is launched on a
    # dedicated stream and still uses a different aux stream for its
    # internal q-proj/weights-proj split.
    _v4_extra_overlap = (
        os.environ.get("TRTLLM_MLA_EXTRA_OVERLAP", "1") == "1"
        and self.compressor is not None
        and self.aux_stream is not None
    )
    _use_indexer_overlap = (
        _v4_extra_overlap
        and do_multi_stream()
        and self.indexer is not None
        and self.indexer_stream is not None
    )

    # Pre-launch the outer compressor on compressor_stream BEFORE
    # kv_a_proj_with_mqa. The compressor only reads hidden_states +
    # attn_metadata, so it has no data dependency on the kv_a_proj GEMM or
    # the downstream q_a/kv_a LN split. A dedicated stream (not aux_stream)
    # keeps kv_a_layernorm free to run on aux_stream in parallel.
    # _q_branch will be queued onto this same stream further down so it
    # runs strictly serial after the compressor; dsv4_compressor_event is
    # recorded only at the end of _q_branch, gating the caller's downstream
    # waits on both compressor + _q_branch completion.
    if _use_indexer_overlap:
        self.dsv4_compressor_start_event.record()
        with torch.cuda.stream(self.compressor_stream):
            self.dsv4_compressor_start_event.wait()
            self.compressor(hidden_states, attn_metadata)

    # Pre-launch the qr-independent half of the indexer prepare phase
    # (weights_proj + internal compressor + k_cache_update) on the
    # indexer's aux stream (self.indexer_aux_stream, wired into the
    # indexer module as its aux_stream). Only reads hidden_states +
    # attn_metadata, so it can overlap with the kv_a_proj -> LN -> split
    # chain on the caller stream and the outer compressor on
    # compressor_stream. The returned tuple is fed back into
    # self.indexer() via pre_aux so the later _indexer_branch skips its
    # own aux-stream launch.
    _indexer_pre_aux = None
    if _use_indexer_overlap:
        _indexer_pre_aux = self.indexer.precompute_aux(hidden_states, attn_metadata)

    q, kv = self.kv_a_proj_with_mqa(hidden_states).split(
        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], -1
    )

    q, kv = maybe_execute_in_parallel(
        lambda: self.q_a_layernorm(q),
        lambda: self.kv_a_layernorm(kv),
        self.ln_events[0],
        self.ln_events[1],
        self.aux_stream,
    )
    compressed_kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
    qr = q
    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

    # CuTe DSL path for q_b_proj (hardware-default cluster count).
    # Restricted to DSv4 CSA layers with compress_ratio=4 so the kernel
    # swap only kicks in where the prologue overlap is exercised; other
    # layers keep the cuBLAS path. Set TRTLLM_MLA_Q_B_PROJ_USE_CUTE_DSL=0
    # to disable. Bias and quantization are not handled.
    _use_q_b_cute = (
        self.has_dsv4_indexer
        and os.environ.get("TRTLLM_MLA_Q_B_PROJ_USE_CUTE_DSL", "1") == "1"
        and self.q_b_proj.bias is None
        and self.q_b_proj.weight.dtype == torch.bfloat16
    )

    def _q_branch():
        # CuTe DSL bf16 path is bench-only and intentionally bypasses the
        # FP8-fused-quant branch (weights are bf16, so the fused FP8 path
        # would never apply anyway, but assert to make the contract
        # explicit and catch any future config drift).
        if _use_q_b_cute:
            assert not is_fused_q_fp8_quant_enabled(self, num_generations=num_generations), (
                "CuTe DSL q_b_proj path is incompatible with the fused FP8 q-quant branch"
            )
            q_proj = q_b_proj_cute_dsl_bf16(q, self.q_b_proj.weight)
            # Cross-iter cleanup: forward_absorption_* downstream gates
            # the fused-FP8 attention path on these attrs being non-None.
            # The FP8 path cannot trigger when weights are bf16, but clear
            # them so stale buffers cannot silently re-enable fusion.
            self._fused_quant_q_buffer = None
            self._fused_q_pe = None
            return deepseek_v4_q_b_layernorm(self, q_proj)
        q_proj = self.q_b_proj(q)
        if is_fused_q_fp8_quant_enabled(self, num_generations=num_generations):
            placeholder_q, quant_q_buffer, q_pe, quant_scale_qkv = (
                deepseek_v4_q_b_layernorm_fused_fp8(self, q_proj)
            )
            self._fused_quant_q_buffer = quant_q_buffer
            self._fused_q_pe = q_pe
            self._quant_scale_qkv = quant_scale_qkv
            return placeholder_q
        self._fused_quant_q_buffer = None
        self._fused_q_pe = None
        return deepseek_v4_q_b_layernorm(self, q_proj)

    def _compressor_branch():
        self.compressor(hidden_states, attn_metadata)
        return None

    def _indexer_branch():
        return self.indexer(
            qr,
            hidden_states,
            attn_metadata,
            position_ids,
            pre_aux=_indexer_pre_aux,
        )

    topk_indices = None
    indexer_ran = False
    if _v4_extra_overlap:
        if _use_indexer_overlap:
            # Compressor + indexer-aux are already in flight from the
            # pre-launch block above. The outer compressor's tail is
            # deferred until after _q_branch so the single wait below
            # gates both compressor and _q_branch completion.
            self.dsv4_overlap_start_event.record()

            with torch.cuda.stream(self.indexer_stream):
                self.dsv4_overlap_start_event.wait()
                topk_indices = _indexer_branch()
                indexer_ran = True
                self.dsv4_indexer_event.record()

            # _q_branch reads qr (post-q_a_layernorm), so it must wait for
            # dsv4_overlap_start_event. Queuing it on compressor_stream
            # serializes compressor -> q_b_proj -> q_b_layernorm while
            # freeing the caller stream during the prologue window.
            with torch.cuda.stream(self.compressor_stream):
                self.dsv4_overlap_start_event.wait()
                q = _q_branch()
                self.dsv4_compressor_event.record()

            self.dsv4_compressor_event.wait()
            self.dsv4_indexer_event.wait()

            # q/topk_indices were produced on other streams; record on the
            # consuming stream so the caching allocator cannot recycle them mid-use.
            cur_stream = torch.cuda.current_stream()
            if q is not None:
                q.record_stream(cur_stream)
            if topk_indices is not None:
                topk_indices.record_stream(cur_stream)
        else:
            q, _ = maybe_execute_in_parallel(
                _q_branch,
                _compressor_branch,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
    else:
        q = _q_branch()
        if self.compressor is not None:
            self.compressor(hidden_states, attn_metadata)

    if self.indexer is not None:
        if not indexer_ran:
            topk_indices = _indexer_branch()

    assert q.shape[0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

    assert output is not None, "output must be provided"

    if num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        topk_indices_ctx = topk_indices[:num_ctx_tokens, :] if topk_indices is not None else None
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        ctx_position_ids = position_ids[..., :num_ctx_tokens] if position_ids is not None else None
        if self.apply_rotary_emb:
            assert ctx_position_ids is not None
            k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)

        forward_context_sparse_mla(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            position_ids=ctx_position_ids,
            latent_cache=latent_cache_ctx,
            topk_indices=topk_indices_ctx,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )

    if num_generations > 0:
        q_gen = q[num_ctx_tokens:, ...]
        topk_indices_gen = (
            topk_indices[num_ctx_tokens:num_tokens, :] if topk_indices is not None else None
        )
        compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
        k_pe_gen = k_pe[num_ctx_tokens:, ...]
        latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
        gen_position_ids = (
            position_ids[..., num_ctx_tokens:num_tokens] if position_ids is not None else None
        )
        if self.apply_rotary_emb:
            assert gen_position_ids is not None
            k_pe_gen = self.apply_rope(q_gen, k_pe_gen, gen_position_ids)

        forward_generation_sparse_mla(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            position_ids=gen_position_ids,
            latent_cache=latent_cache_gen,
            topk_indices=topk_indices_gen,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            dsv4_epilogue_output=dsv4_epilogue_output,
        )
