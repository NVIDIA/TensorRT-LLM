# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 integration for the shared MLA module."""

import os
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType, AttentionMetadata
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._utils import get_sm_version, is_sm_100f

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed import AllReduceParams

_q_b_proj_cute_dsl_import_ok: Optional[bool] = None


# Module initialization and weight lifecycle for DeepSeek-V4 MLA.


def initialize_sparse_attn(
    self,
    *,
    config,
    mapping,
    mapping_o,
    rms_norm_eps: float,
    quant_config,
    q_scaling: float,
    bias: bool,
    dtype: torch.dtype,
    reduce_output: bool,
    aux_stream: Optional[torch.cuda.Stream],
) -> None:
    """Initialize DeepSeek-V4 module state and remove unused dense modules."""
    del bias, q_scaling
    tp_size = mapping.tp_size
    if self.num_groups % tp_size != 0:
        raise ValueError(
            f"DeepSeek-V4 num_groups ({self.num_groups}) must be divisible by tp_size ({tp_size})."
        )
    if self.num_heads % self.num_groups != 0:
        raise ValueError(
            f"DeepSeek-V4 num_heads ({self.num_heads}) must be divisible by "
            f"num_groups ({self.num_groups})."
        )
    if self.is_lite:
        raise ValueError("DeepSeek-V4 does not support lite MLA")

    del self.kv_b_proj
    del self.v_b_proj
    del self.o_proj
    self.mha = None
    self.indexer = getattr(self.mqa, "indexer", None)
    self.compressor = getattr(self.mqa, "compressor", None)

    self.n_local_groups = self.num_groups // tp_size
    self.q_b_layernorm = RMSNorm(
        hidden_size=self.qk_head_dim,
        eps=rms_norm_eps,
        dtype=dtype,
        has_weights=False,
    )
    self.kv_a_layernorm = RMSNorm(
        hidden_size=self.kv_lora_rank + self.qk_rope_head_dim,
        dtype=dtype,
        eps=rms_norm_eps,
    )
    self.o_a_proj = nn.Parameter(
        torch.empty(
            (
                self.n_local_groups,
                self.o_lora_rank,
                self.num_heads * self.qk_head_dim // self.num_groups,
            ),
            dtype=dtype,
        ),
        requires_grad=False,
    )
    self.o_b_proj = Linear(
        self.num_groups * self.o_lora_rank,
        self.hidden_size,
        bias=False,
        dtype=dtype,
        mapping=mapping_o,
        tensor_parallel_mode=TensorParallelMode.ROW,
        quant_config=quant_config,
        skip_create_weights_in_init=config.skip_create_weights_in_init,
        reduce_output=reduce_output,
        allreduce_strategy=config.allreduce_strategy,
        force_dynamic_quantization=config.force_dynamic_quantization,
        use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
        use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
    )

    self.has_dsv4_indexer = (
        self.layer_idx is not None and self.sparse_params.compress_ratios[self.layer_idx] == 4
    )
    self.indexer_stream = None
    self.indexer_aux_stream = None
    self.compressor_stream = None
    if self.has_dsv4_indexer and aux_stream is not None:
        self.indexer_stream = torch.cuda.Stream(device=aux_stream.device)
        self.indexer_aux_stream = torch.cuda.Stream(device=aux_stream.device)
        self.compressor_stream = torch.cuda.Stream(device=aux_stream.device)
    if self.indexer_aux_stream is not None:
        assert self.indexer is not None
        self.indexer.aux_stream = self.indexer_aux_stream

    self.inverse_rotary_emb = RotaryEmbedding(
        self.pos_embd_params.rope,
        head_dim=self.qk_rope_head_dim,
        is_neox=self.pos_embd_params.is_neox,
        inverse=True,
    )
    self._disable_dsv4_epilogue_fusion = os.environ.get(
        "TRTLLM_DSV4_DISABLE_FMHA_EPILOGUE_FUSION", ""
    ).strip().lower() in ("1", "true", "on")
    self.dsv4_overlap_start_event = torch.cuda.Event()
    self.dsv4_compressor_start_event = torch.cuda.Event()
    self.dsv4_compressor_event = torch.cuda.Event()
    self.dsv4_indexer_event = torch.cuda.Event()
    self.attention_output_hidden_size = self.num_heads_tp_cp * self.v_head_dim


def create_sparse_attn_weights(self) -> None:
    has_fp8_block_scales = bool(
        self.o_b_proj.quant_config and self.o_b_proj.quant_config.quant_mode.has_fp8_block_scales()
    )
    self.o_a_proj_dequant = None
    if has_fp8_block_scales:
        self.o_a_proj_scale = nn.Parameter(
            torch.empty(
                (
                    self.n_local_groups,
                    self.o_lora_rank // 128,
                    self.num_heads * self.qk_head_dim // self.num_groups // 128,
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if is_sm_100f():
            self.o_a_proj = nn.Parameter(
                torch.empty(
                    (
                        self.n_local_groups,
                        self.o_lora_rank,
                        self.num_heads * self.qk_head_dim // self.num_groups,
                    ),
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
    else:
        self.o_a_proj_scale = None


def transform_sparse_attn_weights(self) -> None:
    """Skip the dense MLA weight transformation for DeepSeek-V4."""
    return None


# Fused epilogue buffer management and output projection.


def _validate_dsv4_epilogue_buffers(
    self,
    num_tokens: int,
    dsv4_epilogue_output: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_o, output_sf = dsv4_epilogue_output
    scale_buf_m = (num_tokens + 3) // 4 * 4
    if fp8_o.shape[1] != num_tokens or output_sf.shape[2] != scale_buf_m:
        raise RuntimeError("Invalid DSv4 fused epilogue buffers for current token count.")
    return fp8_o, output_sf


def prepare_sparse_attn_outputs(
    self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
) -> list[torch.Tensor]:
    def _should_use_dsv4_epilogue_fusion() -> bool:
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        if self._disable_dsv4_epilogue_fusion:
            return False
        if num_contexts == 0 and num_generations == 0:
            return False
        if num_contexts > 0 and num_generations > 0:
            # The fused buffers do not carry token offsets for a mixed batch.
            return False
        if self.mapping.has_cp_helix() or not is_sm_100f():
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

    def _create_dsv4_epilogue_buffers() -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_local_groups <= 0 or self.num_heads_tp % self.n_local_groups != 0:
            raise ValueError(
                "DSv4 fused epilogue requires num_heads_tp to be divisible by n_local_groups."
            )
        heads_per_group = self.num_heads_tp // self.n_local_groups
        num_tokens = attn_metadata.num_tokens
        scale_buf_m = (num_tokens + 3) // 4 * 4
        fp8_o = hidden_states.new_empty(
            (self.n_local_groups, num_tokens, heads_per_group * self.v_head_dim),
            dtype=torch.float8_e4m3fn,
        )
        output_sf = hidden_states.new_empty(
            (
                self.n_local_groups,
                heads_per_group * (self.v_head_dim // 128),
                scale_buf_m,
            ),
            dtype=torch.float32,
        )
        return fp8_o, output_sf

    if _should_use_dsv4_epilogue_fusion():
        attn_output = [self.create_output(hidden_states[:0], attn_metadata.num_contexts)]
        attn_output.extend(_create_dsv4_epilogue_buffers())
        return attn_output
    return [self.create_output(hidden_states, attn_metadata.num_contexts)]


def project_sparse_attn_output(
    self,
    attn_output: list[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
    attn_metadata: Optional[AttentionMetadata] = None,
    all_reduce_params: Optional["AllReduceParams"] = None,
) -> torch.Tensor:
    del attn_metadata, all_reduce_params
    if len(attn_output) > 1:
        attn_fp8, attn_scale = attn_output[1:]
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

    attn_output_tensor = attn_output[0]
    assert position_ids is not None
    num_tokens = attn_output_tensor.shape[0]
    attn_output_tensor = attn_output_tensor.view(num_tokens, self.num_heads_tp, -1)

    # Fuse inverse RoPE with FP8 quantization to avoid a BF16 latent read/write.
    # This is independent of the K/V absorption BMM implementation.
    fused_inv_rope_fp8 = self.o_a_proj.dtype == torch.float8_e4m3fn and is_sm_100f()
    if fused_inv_rope_fp8:
        heads_per_group = self.num_heads_tp // self.n_local_groups
        attn_fp8, attn_scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
            attn_output_tensor,
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
            device=attn_output_tensor.device,
            dtype=self.dtype,
        )
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(
            attn_fp8,
            self.o_a_proj,
            attn_scale,
            self.o_a_proj_scale,
            o_lora.transpose(0, 1),
        )
        o_lora = o_lora.flatten(1)
        return self.o_b_proj(o_lora)

    # Restore the RoPE portion before output projection.
    torch.ops.trtllm.mla_rope_inplace(
        attn_output_tensor,
        position_ids.view(-1),
        self.inverse_rotary_emb.rotary_cos_sin,
        self.num_heads_tp,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
        True,
        self.inverse_rotary_emb.is_neox,
    )

    o_lora = torch.empty(
        [num_tokens, self.n_local_groups, self.o_lora_rank],
        device=attn_output_tensor.device,
        dtype=attn_output_tensor.dtype,
    )
    if self.o_a_proj.dtype == torch.bfloat16:
        # [groups, tokens, dim] @ [groups, dim, rank] -> [groups, tokens, rank]
        torch.ops.trtllm.bmm_out(
            attn_output_tensor.view(num_tokens, self.n_local_groups, -1).transpose(0, 1),
            self.o_a_proj.transpose(1, 2),
            o_lora.transpose(0, 1),
        )
    elif self.o_a_proj.dtype == torch.float8_e4m3fn:
        from tensorrt_llm._torch.modules.mla import fp8_block_scaling_bmm_out

        fp8_block_scaling_bmm_out(
            attn_output_tensor.view(num_tokens, self.n_local_groups, -1),
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


# Context and generation attention execution paths.


def forward_generation_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    sparse_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the DeepSeek-V4 generation absorption path."""
    if get_sm_version() < 100:
        raise RuntimeError("DeepSeek-V4 is not supported on pre-Blackwell GPUs")
    del compressed_kv, k_pe
    num_tokens = q.shape[0]
    q_pe = q.view(-1, self.num_heads_tp, self.qk_head_dim)[..., self.qk_nope_head_dim :]

    num_seqs = attn_metadata.num_seqs
    cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
    cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)

    has_fp8_kv_cache = bool(getattr(self.mqa, "has_fp8_kv_cache", False))
    mla_bmm1_scale = None
    mla_bmm2_scale = None
    quant_q_buffer = None
    if has_fp8_kv_cache:
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
        quant_q_buffer = torch.empty(
            num_tokens,
            self.num_heads_tp,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=torch.uint8,
            device=q.device,
        )

    self.mqa.mla_rope_generation(
        q,
        q_pe,
        latent_cache,
        attn_metadata,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
    )

    attention_output = output
    output_sf = None
    inverse_rope_cos_sin = None
    if sparse_epilogue_output is not None:
        attention_output, output_sf = _validate_dsv4_epilogue_buffers(
            self, num_tokens, sparse_epilogue_output
        )
        inverse_rope_cos_sin = self.inverse_rotary_emb.rotary_cos_sin

    attn_out_latent = self._attn_forward_gen(
        self.mqa,
        q,
        None,
        None,
        position_ids,
        attn_metadata,
        attention_input_type=AttentionInputType.generation_only,
        out_scale=self.out_scale,
        output=attention_output,
        output_sf=output_sf,
        latent_cache=latent_cache,
        q_pe=q_pe,
        topk_indices=topk_indices,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
        dsv4_inv_rope_cos_sin_cache=inverse_rope_cos_sin,
        enable_dsv4_epilogue_fusion=sparse_epilogue_output is not None,
    )
    if sparse_epilogue_output is not None:
        return attn_out_latent

    if self.mapping.has_cp_helix():
        raise RuntimeError(
            "DeepSeek-V4 + CP Helix is not supported because the post-process "
            "does not preserve the pre-allocated output buffer."
        )
    assert attn_out_latent.data_ptr() == output.data_ptr(), (
        "Attention backend did not write into the provided output buffer."
    )
    return output


def forward_context_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    sparse_epilogue_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the DeepSeek-V4 context absorption path."""
    if get_sm_version() < 100:
        raise RuntimeError("DeepSeek-V4 is not supported on pre-Blackwell GPUs")
    del compressed_kv, k_pe
    num_tokens = q.shape[0]
    q_pe = q.view(-1, self.num_heads_tp, self.qk_head_dim)[..., self.qk_nope_head_dim :]

    quant_q_buffer = getattr(self, "_fused_quant_q_buffer", None)
    fused_q_pe = getattr(self, "_fused_q_pe", None)
    quant_scale_qkv = getattr(self, "_quant_scale_qkv", None)
    use_fused_q_fp8 = (
        quant_q_buffer is not None and fused_q_pe is not None and quant_scale_qkv is not None
    )
    if use_fused_q_fp8:
        q_pe = fused_q_pe[:num_tokens]
        quant_q_buffer = quant_q_buffer[:num_tokens].view(
            num_tokens,
            self.num_heads_tp,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
    else:
        quant_q_buffer = None
        quant_scale_qkv = None

    attention_output = output
    output_sf = None
    inverse_rope_cos_sin = None
    if sparse_epilogue_output is not None:
        attention_output, output_sf = _validate_dsv4_epilogue_buffers(
            self, num_tokens, sparse_epilogue_output
        )
        inverse_rope_cos_sin = self.inverse_rotary_emb.rotary_cos_sin

    attn_out_latent = self._attn_forward_gen(
        self.mqa,
        q,
        None,
        None,
        position_ids,
        attn_metadata,
        attention_input_type=AttentionInputType.context_only,
        out_scale=self.out_scale,
        output=attention_output,
        output_sf=output_sf,
        latent_cache=latent_cache,
        q_pe=q_pe,
        quant_q_buffer=quant_q_buffer,
        quant_scale_qkv=quant_scale_qkv,
        topk_indices=topk_indices,
        dsv4_inv_rope_cos_sin_cache=inverse_rope_cos_sin,
        enable_dsv4_epilogue_fusion=sparse_epilogue_output is not None,
    )
    self._fused_quant_q_buffer = None
    self._fused_q_pe = None

    if sparse_epilogue_output is not None:
        return attn_out_latent
    if self.mapping.has_cp_helix():
        raise RuntimeError(
            "DeepSeek-V4 + CP Helix is not supported because the post-process "
            "does not preserve the pre-allocated output buffer."
        )
    assert attn_out_latent.data_ptr() == output.data_ptr(), (
        "Attention backend did not write into the provided output buffer."
    )
    return output


# End-to-end DeepSeek-V4 forward scheduling hook.


def forward_sparse_attn(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    attn_output: list[torch.Tensor],
) -> None:
    """Run DeepSeek-V4 MLA and write into the algorithm-defined output buffers."""
    assert self.mha is None and self.mqa is not None, "DeepSeek-V4 is only supported in MQA mode"
    output = attn_output[0]
    sparse_epilogue_output = (attn_output[1], attn_output[2]) if len(attn_output) > 1 else None
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    if sparse_epilogue_output is not None and ((num_contexts > 0) == (num_generations > 0)):
        raise RuntimeError("DSv4 epilogue fusion requires a context-only or generation-only batch.")

    hidden_states = hidden_states[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    # TRTLLM_MLA_EXTRA_OVERLAP overlaps the compressor and ratio-4 indexer with
    # Q projection on dedicated streams.
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

    # The compressor depends only on hidden states and metadata, so start it before
    # KV projection. Q work later shares this stream and its completion event.
    if _use_indexer_overlap:
        self.dsv4_compressor_start_event.record()
        with torch.cuda.stream(self.compressor_stream):
            self.dsv4_compressor_start_event.wait()
            self.compressor(hidden_states, attn_metadata)

    # Precompute QR-independent indexer work while the caller stream prepares KV.
    # Passing pre_aux later prevents the indexer from launching this work again.
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

    # Use the CuTe BF16 Q projection only for ratio-4 CSA layers with unquantized,
    # bias-free weights. TRTLLM_MLA_Q_B_PROJ_USE_CUTE_DSL disables this path.
    _use_q_b_cute = (
        self.has_dsv4_indexer
        and os.environ.get("TRTLLM_MLA_Q_B_PROJ_USE_CUTE_DSL", "1") == "1"
        and self.q_b_proj.bias is None
        and self.q_b_proj.weight.dtype == torch.bfloat16
    )

    def _q_b_proj_cute_dsl_bf16(q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
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

    def _is_fused_q_fp8_quant_enabled() -> bool:
        # Mixed/generation batches must use q itself in the generation path.
        if os.environ.get("TRTLLM_DISABLE_FUSED_Q_FP8_QUANT", "0") == "1":
            return False
        if self.qk_head_dim != 512 or self.kv_lora_rank != 448:
            return False
        if num_generations > 0:
            return False
        return bool(getattr(self.mqa, "has_fp8_kv_cache", False))

    def _q_b_layernorm(q: torch.Tensor) -> torch.Tensor:
        assert q.dim() == 2 and q.shape[1] == self.num_heads_tp * self.qk_head_dim
        return torch.ops.trtllm.deepseek_v4_q_norm(
            q,
            self.num_heads_tp,
            self.qk_head_dim,
            float(self.q_b_layernorm.variance_epsilon),
        )

    def _q_b_layernorm_fused_fp8(
        q_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert q_proj.dim() == 2
        assert q_proj.shape[1] == self.num_heads_tp * self.qk_head_dim
        if getattr(self, "_quant_scale_qkv", None) is None:
            self._quant_scale_qkv = torch.tensor([1.0], dtype=torch.float32, device=q_proj.device)

        # The 3D q_pe layout is required by the sparse-MLA context op.
        num_q_tokens = q_proj.shape[0]
        rope_dim = self.qk_head_dim - self.kv_lora_rank
        quant_q_buffer = q_proj.new_empty(
            (num_q_tokens, self.num_heads_tp * self.qk_head_dim),
            dtype=torch.float8_e4m3fn,
        )
        q_pe = q_proj.new_empty((num_q_tokens, self.num_heads_tp, rope_dim))
        torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
            q_proj,
            quant_q_buffer,
            q_pe.view(num_q_tokens, self.num_heads_tp * rope_dim),
            self.num_heads_tp,
            self.qk_head_dim,
            self.kv_lora_rank,
            float(self.q_b_layernorm.variance_epsilon),
            self._quant_scale_qkv,
        )
        assert self._quant_scale_qkv is not None
        return q_proj, quant_q_buffer, q_pe, self._quant_scale_qkv

    def _q_branch():
        # CuTe BF16 projection is incompatible with fused FP8 Q quantization.
        if _use_q_b_cute:
            assert not _is_fused_q_fp8_quant_enabled(), (
                "CuTe DSL q_b_proj path is incompatible with the fused FP8 q-quant branch"
            )
            q_proj = _q_b_proj_cute_dsl_bf16(q, self.q_b_proj.weight)
            # The context path detects fusion from these buffers, so clear stale state.
            self._fused_quant_q_buffer = None
            self._fused_q_pe = None
            return _q_b_layernorm(q_proj)
        q_proj = self.q_b_proj(q)
        if _is_fused_q_fp8_quant_enabled():
            placeholder_q, quant_q_buffer, q_pe, quant_scale_qkv = _q_b_layernorm_fused_fp8(q_proj)
            self._fused_quant_q_buffer = quant_q_buffer
            self._fused_q_pe = q_pe
            self._quant_scale_qkv = quant_scale_qkv
            return placeholder_q
        self._fused_quant_q_buffer = None
        self._fused_q_pe = None
        return _q_b_layernorm(q_proj)

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
            # Compressor and indexer prework are already in flight. Run Q after the
            # compressor on the same stream, then synchronize both streams below.
            self.dsv4_overlap_start_event.record()

            with torch.cuda.stream(self.indexer_stream):
                self.dsv4_overlap_start_event.wait()
                topk_indices = _indexer_branch()
                indexer_ran = True
                self.dsv4_indexer_event.record()

            with torch.cuda.stream(self.compressor_stream):
                self.dsv4_overlap_start_event.wait()
                q = _q_branch()
                self.dsv4_compressor_event.record()

            self.dsv4_compressor_event.wait()
            self.dsv4_indexer_event.wait()

            # Keep cross-stream outputs alive on the consuming stream.
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

        forward_context_sparse_attn(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            position_ids=ctx_position_ids,
            latent_cache=latent_cache_ctx,
            topk_indices=topk_indices_ctx,
            sparse_epilogue_output=sparse_epilogue_output,
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

        forward_generation_sparse_attn(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            position_ids=gen_position_ids,
            latent_cache=latent_cache_gen,
            topk_indices=topk_indices_gen,
            sparse_epilogue_output=sparse_epilogue_output,
        )
