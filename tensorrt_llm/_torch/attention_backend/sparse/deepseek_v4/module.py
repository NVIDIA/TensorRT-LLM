# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 integration for the shared MLA module."""

from __future__ import annotations

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
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import get_sm_version, is_sm_100f

from ..hooks import MLASparseHooks, register_mla_sparse_hooks
from ..params import SparseBackendForwardArgs
from .flash_mla import DeepSeekV4FlashMLA

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed import AllReduceParams
    from tensorrt_llm._torch.modules.mla import MLA

_q_b_proj_cute_dsl_import_ok: Optional[bool] = None


# Module initialization and weight lifecycle for DeepSeek-V4 MLA.


def _has_dsv4_indexer(self) -> bool:
    return self.layer_idx is not None and self.sparse_params.compress_ratios[self.layer_idx] == 4


def initialize_sparse_attn(self) -> None:
    """Initialize DeepSeek-V4 module state."""
    tp_size = self.num_heads // self.num_heads_tp
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

    self.indexer = getattr(self.mqa, "indexer", None)
    self.compressor = getattr(self.mqa, "compressor", None)

    self.n_local_groups = self.num_groups // tp_size
    self.q_b_layernorm = RMSNorm(
        hidden_size=self.qk_head_dim,
        eps=self.rms_norm_eps,
        dtype=self.dtype,
        has_weights=False,
    )
    self.kv_a_layernorm = RMSNorm(
        hidden_size=self.kv_lora_rank + self.qk_rope_head_dim,
        dtype=self.dtype,
        eps=self.rms_norm_eps,
    )
    self.o_a_proj = nn.Parameter(
        torch.empty(
            (
                self.n_local_groups,
                self.o_lora_rank,
                self.num_heads * self.qk_head_dim // self.num_groups,
            ),
            dtype=self.dtype,
        ),
        requires_grad=False,
    )
    self.o_b_proj = Linear(
        self.num_groups * self.o_lora_rank,
        self.hidden_size,
        bias=False,
        dtype=self.dtype,
        mapping=self.mapping_o,
        tensor_parallel_mode=TensorParallelMode.ROW,
        quant_config=self.quant_config,
        skip_create_weights_in_init=self.skip_create_weights_in_init,
        reduce_output=self.reduce_output,
        allreduce_strategy=self.allreduce_strategy,
        force_dynamic_quantization=self.force_dynamic_quantization,
        use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
        use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
    )

    self.has_dsv4_indexer = _has_dsv4_indexer(self)
    self.indexer_stream = (
        self.aux_stream_dict.get(AuxStreamType.MlaIndexer) if self.has_dsv4_indexer else None
    )
    self.indexer_aux_stream = (
        self.aux_stream_dict.get(AuxStreamType.MlaIndexerAux) if self.has_dsv4_indexer else None
    )
    # Not gated on has_dsv4_indexer: HCA layers have no indexer but still pre-launch
    # the compressor on this stream, which is the case the pre-launch exists for.
    self.compressor_stream = (
        self.aux_stream_dict.get(AuxStreamType.MlaCompressor)
        if self.compressor is not None
        else None
    )
    if self.indexer_aux_stream is not None:
        assert self.indexer is not None
        self.indexer.aux_stream = self.indexer_aux_stream

    self.inverse_rotary_emb = RotaryEmbedding(
        self.pos_embd_params.rope,
        head_dim=self.qk_rope_head_dim,
        is_neox=self.pos_embd_params.is_neox,
        inverse=True,
    )
    self._dsv4_flash_mla = None
    sm_version = get_sm_version()
    if sm_version < 90:
        raise RuntimeError(f"DeepSeek-V4 requires Hopper or newer GPUs, got SM{sm_version}")
    if sm_version == 90:
        self._dsv4_flash_mla = DeepSeekV4FlashMLA(self.mqa, self.mqa.compress_ratio)
    self._disable_dsv4_epilogue_fusion = os.environ.get(
        "TRTLLM_DSV4_DISABLE_FMHA_EPILOGUE_FUSION", ""
    ).strip().lower() in ("1", "true", "on")
    self.dsv4_overlap_start_event = torch.cuda.Event()
    self.dsv4_compressor_start_event = torch.cuda.Event()
    self.dsv4_compressor_event = torch.cuda.Event()
    self.dsv4_indexer_event = torch.cuda.Event()
    # The hoisted fused kv-norm launch spans the whole Q branch, which itself uses
    # ln_events on some paths, so it needs its own pair.
    self.kv_gen_events = [torch.cuda.Event(), torch.cuda.Event()]
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


# Fused epilogue buffer management and output projection.


def _create_dsv4_epilogue_buffers(
    self,
    q: torch.Tensor,
    num_tokens: int,
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


def _run_dsv4_o_lora_bmms(
    self,
    o_lora_output: torch.Tensor,
    num_context_tokens: int,
    num_tokens: int,
    context_o_lora_bmm_input: Optional[tuple[torch.Tensor, torch.Tensor]],
    generation_o_lora_bmm_input: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    def run_o_lora_bmm(
        o_lora_bmm_input: tuple[torch.Tensor, torch.Tensor],
        phase_o_lora_output: torch.Tensor,
    ) -> None:
        attn_fp8, attn_scale = o_lora_bmm_input
        torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(
            attn_fp8,
            self.o_a_proj,
            attn_scale,
            self.o_a_proj_scale,
            phase_o_lora_output.transpose(0, 1),
        )

    if context_o_lora_bmm_input is not None:
        run_o_lora_bmm(
            context_o_lora_bmm_input,
            o_lora_output[:num_context_tokens],
        )
    if generation_o_lora_bmm_input is not None:
        run_o_lora_bmm(
            generation_o_lora_bmm_input,
            o_lora_output[num_context_tokens:num_tokens],
        )


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

    if _should_use_dsv4_epilogue_fusion():
        num_tokens = hidden_states.shape[0]
        return [
            torch.empty(
                [num_tokens, self.n_local_groups, self.o_lora_rank],
                device=hidden_states.device,
                dtype=self.dtype,
            )
        ]
    return [self.create_output(hidden_states, attn_metadata.num_contexts)]


def project_sparse_attn_output(
    self,
    attn_output: list[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
    attn_metadata: Optional[AttentionMetadata] = None,
    all_reduce_params: Optional["AllReduceParams"] = None,
) -> torch.Tensor:
    del attn_metadata, all_reduce_params
    attn_output_tensor = attn_output[0]
    # BCG/mixed-batch epilogue fusion runs o_a_proj at the end of attention,
    # so this 3D tensor is O-LoRA output and only o_b_proj remains.
    if attn_output_tensor.ndim == 3:
        return self.o_b_proj(attn_output_tensor.flatten(1))

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


# ---------------------------------------------------------------------------
# DSv4 prologue fusion: kv_a_layernorm folded into the KV kernels, q_nope FP8
# quantize and the Q RoPE folded into q_b_layernorm.
# ---------------------------------------------------------------------------


def _mla_gen_scheduler_scalars(self: MLA, device: torch.device, has_fp8_kv_cache: bool):
    """Persistent per-layer FMHA scheduler scalars for the generation path.

    `fmha_tile_counter` (the persistent-CTA tile dispenser) plus the bmm1/bmm2
    scales. All three are written from scratch by the RoPE kernel -- or, on the
    fused kv-norm path, by `mlaKvNormRopeQuantGenerationKernel` -- before the
    FMHA kernel reads them, so nothing carries across launches and a single
    buffer per layer is enough.
    """
    counter = getattr(self, "_mla_fmha_tile_counter", None)
    if counter is None or counter.device != device:
        counter = torch.empty(1, dtype=torch.uint32, device=device)
        self._mla_fmha_tile_counter = counter
    if not has_fp8_kv_cache:
        return counter, None, None
    bmm1 = getattr(self, "_mla_bmm1_scale", None)
    if bmm1 is None or bmm1.device != device:
        bmm1 = torch.empty(2, dtype=torch.float32, device=device)
        self._mla_bmm1_scale = bmm1
        self._mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
    return counter, bmm1, self._mla_bmm2_scale


def _is_fused_kv_norm_enabled(self: MLA, num_generations: int = 0) -> bool:
    # Mirrors `_is_fused_q_fp8_quant_enabled`. Two kernels, same warp-per-row shape:
    #   context    -> `mlaKvNormRopeQuantContextKernel`
    #   generation -> `mlaKvNormRopeQuantGenerationKernel`
    # Mixed batches are fine -- both read the same raw latent view.
    # The kernels describe the latent row with their K_DIM/ROPE_DIM template
    # constants, so these must be the 448/64 instantiation.
    if self.kv_lora_rank != 448 or self.qk_rope_head_dim != 64:
        return False
    # V4 normalizes the whole 512-wide latent; the weight must span it.
    if self.kv_a_layernorm.weight.shape[0] != self.kv_lora_rank + self.qk_rope_head_dim:
        return False
    return bool(getattr(self.mqa, "has_fp8_kv_cache", False))


def _is_fused_q_fp8_quant_enabled(
    self: MLA, num_generations: int = 0, num_contexts: int = 0
) -> bool:
    # The fused path leaves a placeholder bf16 q_buf, so consumers read the FP8
    # buffer and `_fused_q_pe`: context takes the prefix rows, generation the
    # suffix. A mixed batch launches once per phase over those disjoint ranges.
    if os.environ.get("TRTLLM_DISABLE_FUSED_Q_FP8_QUANT", "0") == "1":
        return False
    if self.qk_head_dim != 512 or self.kv_lora_rank != 448:
        return False
    # fp8_ds_mla (FlashInfer sparse MLA) does not use the fused Q FP8 path.
    if self.kv_cache_dtype == "fp8_ds_mla":
        return False
    return bool(getattr(self.mqa, "has_fp8_kv_cache", False))


def _is_fused_prologue_active(
    self: MLA, *, num_contexts: int, num_generations: int, rope_specs: list
) -> bool:
    # Coupled by correctness: the fused KV kernels leave the RAW latent, which the
    # un-fused RoPE kernels' Q region would read un-normalized. Safe only because
    # the fused Q path owns Q entirely -- so require both or neither. The Q fold in
    # turn needs the per-phase specs, which `_fused_q_rope_specs` withholds when the
    # metadata cannot supply ragged context positions.
    return (
        _is_fused_kv_norm_enabled(self, num_generations=num_generations)
        and _is_fused_q_fp8_quant_enabled(
            self, num_generations=num_generations, num_contexts=num_contexts
        )
        and bool(rope_specs)
    )


def _fused_q_rope_specs(
    self: MLA, attn_metadata: AttentionMetadata, num_contexts: int, num_generations: int
):
    """(cos_sin, specs) for the fused Q RoPE, one spec per phase in the batch.

    Each spec is `(rows, cache_lens, seq_len, cu_q_seqlens)` where `rows` is the
    row range of the batch it covers. Context rows take positions from a ragged
    token cumsum, generation rows from a uniform query length, so one launch
    cannot serve both -- a mixed batch yields two specs and the caller issues one
    launch per spec over disjoint row ranges of the same output buffers.

    Returns `(None, [])` when the batch does not qualify.
    """
    cache_lens = getattr(attn_metadata, "kv_lens_cuda_runtime", None)
    if cache_lens is None:
        return None, []

    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    specs = []

    if num_contexts > 0:
        # Context is ragged: positions come from a token-wise cumsum, not a uniform
        # seq_len. Deliberately NOT `ctx_uncached_token_indptr` -- it holds the same
        # values but only exists under `enable_context_mla_with_cached_kv`, so this
        # path would silently no-op without chunked prefill or block reuse. Paired
        # with the cache lengths, it recovers each row's cached offset.
        prep_ctx = getattr(attn_metadata, "mla_prepare_ctx_cu_seqlens", None)
        if prep_ctx is None:
            return None, []
        cu_q_seqlens = prep_ctx()
        if cu_q_seqlens is None:
            return None, []
        specs.append((slice(0, num_ctx_tokens), cache_lens[:num_contexts], 0, cu_q_seqlens))

    if num_generations > 0:
        num_gen_tokens = num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0 or num_gen_tokens % num_generations != 0:
            return None, []
        specs.append(
            (
                slice(num_ctx_tokens, num_tokens),
                cache_lens[num_contexts:],
                num_gen_tokens // num_generations,
                None,
            )
        )

    if not specs:
        return None, []

    # The table must already cover every position these rows will read.
    ensure = getattr(self.mqa, "_ensure_rope_table_size", None)
    if ensure is not None:
        ensure(attn_metadata.max_seq_len)
    cos_sin = getattr(self.mqa, "rotary_cos_sin", None)
    if cos_sin is None:
        return None, []
    return cos_sin, specs


def _deepseek_v4_q_b_layernorm_fused_fp8(
    self: MLA,
    q_proj: torch.Tensor,
    rope_cos_sin: Optional[torch.Tensor] = None,
    rope_specs: Optional[list] = None,
):
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
    q_pe_flat = q_pe.view(num_tokens, self.num_heads_tp * rope_dim)
    # With rope inputs the kernel also rotates the rope segment into
    # `quant_q_buffer` as FP8, leaving `q_pe` untouched. One spec per phase; the
    # row ranges are disjoint, so launches never overlap.
    launches = rope_specs if rope_specs else [(slice(None), None, 0, None)]
    for rows, cache_seq_lens, seq_len, cu_q_seqlens in launches:
        torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
            q_proj[rows],
            quant_q_buffer[rows],
            q_pe_flat[rows],
            self.num_heads_tp,
            self.qk_head_dim,
            self.kv_lora_rank,
            float(self.q_b_layernorm.variance_epsilon),
            self._quant_scale_qkv,
            rope_cos_sin,
            cache_seq_lens,
            seq_len,
            cu_q_seqlens,
        )
    # Both buffers must be live for the fused path; the downstream
    # absorption-context op switches on `quant_scale_qkv is not None`
    # to enable the C++ fusion (see trtllm.py `thop.attention` call).
    assert self._quant_scale_qkv is not None, (
        "fused FP8-Q quant requires _quant_scale_qkv to be set"
    )
    return q_proj, quant_q_buffer, q_pe, self._quant_scale_qkv


def _launch_fused_kv_norm_gen(
    self: MLA,
    latent_cache_gen: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> bool:
    """Run the fused kv-norm/RoPE/quant KV kernel ahead of the Q branch.

    The kernel reads the raw `kv_a_proj` latent and writes the paged KV cache;
    it never touches q_pe or `fused_q`. Nothing between here and FMHA reads the
    KV cache either, so it runs on `aux_stream` concurrently with
    q_a_layernorm -> q_b_proj -> q_b_layernorm, which is where the standalone
    `kv_a_layernorm` it replaced used to run.

    Returns True when the launch happened, so the later `mla_rope_generation`
    knows to run Q-only instead of doing the KV half again.
    """
    if self.aux_stream is None or not do_multi_stream():
        return False
    # The cu_seqlens buffers are Q-kernel state; the KV kernel never reads them.
    # Only the precomputing metadata exposes them at this point, so without the
    # hook the hoist is skipped rather than allocating here.
    if getattr(attn_metadata, "mla_prepare_scheduler_buffers", None) is None:
        return False

    has_fp8_kv_cache = getattr(self.mqa, "has_fp8_kv_cache", False)
    counter, bmm1_scale, bmm2_scale = _mla_gen_scheduler_scalars(
        self, latent_cache_gen.device, has_fp8_kv_cache
    )
    cu_q_seqlens, cu_kv_seqlens = attn_metadata.mla_prepare_scheduler_buffers(self.num_heads_tp)

    # The latent is produced on the caller stream; gate the aux stream on it.
    self.kv_gen_events[0].record()
    with torch.cuda.stream(self.aux_stream):
        self.kv_gen_events[0].wait()
        # Keeps the allocator from recycling the latent while the aux stream
        # still reads it.
        latent_cache_gen.record_stream(self.aux_stream)
        self.mqa.mla_rope_generation(
            None,
            None,
            latent_cache_gen,
            attn_metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            counter,
            bmm1_scale,
            bmm2_scale,
            None,
            kv_norm_weight=self.kv_a_layernorm.weight,
            kv_norm_eps=float(self.kv_a_layernorm.variance_epsilon),
            precomputed_cu_seqlens=True,
            precomputed_fmha_scheduler=(
                has_fp8_kv_cache and bmm1_scale is not None and bmm2_scale is not None
            ),
            kv_only=True,
        )
        self.kv_gen_events[1].record()
    return True


def forward_generation_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    enable_dsv4_epilogue_fusion: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the DeepSeek-V4 generation absorption path."""
    if get_sm_version() == 90:
        if self._dsv4_flash_mla is None:
            raise RuntimeError("DeepSeek-V4 Hopper FlashMLA helper is not initialized")
        return self._dsv4_flash_mla.forward_generation(
            q,
            latent_cache,
            attn_metadata,
            output,
            topk_indices,
            self.softmax_scale,
            position_ids=position_ids,
            rotary_cos_sin=self.inverse_rotary_emb.rotary_cos_sin,
            is_neox=self.inverse_rotary_emb.is_neox,
        )
    del compressed_kv, k_pe
    num_tokens = q.shape[0]
    q_pe = q.view(-1, self.num_heads_tp, self.qk_head_dim)[..., self.qk_nope_head_dim :]

    # Fused FP8-Q path: `q` is the placeholder q_b_proj output, so the view above is
    # un-normalized and must not be used. q_b_layernorm already emitted the normalized
    # rope segment (`_fused_q_pe`) and the FP8 nope segment of `_fused_quant_q_buffer`.
    fused_q_fp8_pe = getattr(self, "_fused_q_pe", None)
    fused_q_fp8_buf = getattr(self, "_fused_quant_q_buffer", None)
    fused_q_fp8_scale = getattr(self, "_quant_scale_qkv", None)
    use_fused_q_fp8 = (
        fused_q_fp8_pe is not None and fused_q_fp8_buf is not None and fused_q_fp8_scale is not None
    )
    if use_fused_q_fp8:
        # Suffix slice: the Q branch ran over the whole batch.
        gen_offset = fused_q_fp8_pe.shape[0] - num_tokens
        q_pe = fused_q_fp8_pe[gen_offset:]

    num_seqs = attn_metadata.num_seqs

    # Cumulative Q/KV seqlens are layer-invariant: the metadata fills fixed-address
    # buffers once per iteration and the kernel skips its per-layer recomputation.
    # Metadata without the hook falls back to per-layer allocation + in-kernel fill.
    _mla_prep = getattr(attn_metadata, "mla_prepare_scheduler_buffers", None)
    if _mla_prep is not None:
        cu_q_seqlens, cu_kv_seqlens = _mla_prep(self.num_heads_tp)
        precomputed_cu_seqlens = True
    else:
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        precomputed_cu_seqlens = False

    has_fp8_kv_cache = bool(getattr(self.mqa, "has_fp8_kv_cache", False))

    # Shape-invariant (one tile counter + three floats, rewritten every launch), so
    # they are layer state rather than per-forward allocations, and fixed addresses
    # suit CUDA graph capture.
    fmha_scheduler_counter, mla_bmm1_scale, mla_bmm2_scale = _mla_gen_scheduler_scalars(
        self, q.device, has_fp8_kv_cache
    )

    quant_q_buffer = None
    quant_scale_qkv = None
    if use_fused_q_fp8:
        # Already allocated and half-filled by q_b_layernorm; reusing it also drops a
        # per-layer num_tokens x heads x 512 allocation from the decode step.
        quant_q_buffer = fused_q_fp8_buf[gen_offset:].view(
            num_tokens,
            self.num_heads_tp,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        quant_scale_qkv = fused_q_fp8_scale
    elif has_fp8_kv_cache:
        quant_q_buffer = torch.empty(
            num_tokens,
            self.num_heads_tp,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=torch.uint8,
            device=q.device,
        )

    _fused_kv_norm = getattr(self, "_fused_kv_norm_active", False)
    # Already ran on aux_stream, overlapped with the Q branch; the op still has to be
    # told the KV half is done.
    _kv_hoisted = getattr(self, "_fused_kv_norm_hoisted", False)
    # q_b_layernorm already rotated the rope segment into the FP8 Q buffer, so no Q
    # work is left -- but the KV half still runs, hence kv_only.
    _q_rope_done = use_fused_q_fp8 and _fused_kv_norm
    assert not (_q_rope_done and not precomputed_cu_seqlens), (
        "fused Q RoPE drops the kernel that fills cu_q_seqlens; "
        "attention metadata must precompute them"
    )
    # fp8_ds_mla (FlashInfer sparse MLA) owns RoPE itself; both halves already done
    # elsewhere -> nothing left to launch at all.
    if self.kv_cache_dtype != "fp8_ds_mla" and not (_q_rope_done and _kv_hoisted):
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
            kv_norm_weight=(
                self.kv_a_layernorm.weight if _fused_kv_norm and not _kv_hoisted else None
            ),
            kv_norm_eps=float(self.kv_a_layernorm.variance_epsilon),
            kv_done_elsewhere=_kv_hoisted,
            # Non-None tells the launcher q_nope is already FP8, so it drops the q_nope
            # quantize rows (1024 of 1161 at head_num 128) from the grid.
            quant_scale_qkv=quant_scale_qkv,
            kv_only=_q_rope_done and not _kv_hoisted,
            precomputed_cu_seqlens=precomputed_cu_seqlens,
            # `_deepseek_v4_local_to_global_kernel` emits the tile counter and bmm
            # scales here -- last kernel before FMHA, a better home than block (0,0) of
            # the RoPE kernels. Same condition as the sparse_attn_predict branch.
            precomputed_fmha_scheduler=(
                has_fp8_kv_cache and mla_bmm1_scale is not None and mla_bmm2_scale is not None
            ),
        )

    if _kv_hoisted:
        # FMHA below reads the KV cache the hoisted kernel wrote.
        self.kv_gen_events[1].wait()

    dsv4_output = output
    o_lora_bmm_input_scale = None
    inverse_rope_cos_sin = None
    if enable_dsv4_epilogue_fusion:
        dsv4_output, o_lora_bmm_input_scale = _create_dsv4_epilogue_buffers(self, q, num_tokens)
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
        output=dsv4_output,
        output_sf=o_lora_bmm_input_scale,
        latent_cache=latent_cache,
        q_pe=q_pe,
        sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk_indices),
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
        dsv4_inv_rope_cos_sin_cache=inverse_rope_cos_sin,
        enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
    )
    if enable_dsv4_epilogue_fusion:
        assert dsv4_output is not None and o_lora_bmm_input_scale is not None
        return dsv4_output, o_lora_bmm_input_scale

    assert output is not None
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
    output: Optional[torch.Tensor],
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    enable_dsv4_epilogue_fusion: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the DeepSeek-V4 context absorption path."""
    if get_sm_version() == 90:
        if self._dsv4_flash_mla is None:
            raise RuntimeError("DeepSeek-V4 Hopper FlashMLA helper is not initialized")
        return self._dsv4_flash_mla.forward_context(
            q,
            latent_cache,
            attn_metadata,
            output,
            topk_indices,
            self.softmax_scale,
            position_ids=position_ids,
            rotary_cos_sin=self.inverse_rotary_emb.rotary_cos_sin,
            is_neox=self.inverse_rotary_emb.is_neox,
        )
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

    dsv4_output = output
    o_lora_bmm_input_scale = None
    inverse_rope_cos_sin = None
    if enable_dsv4_epilogue_fusion:
        dsv4_output, o_lora_bmm_input_scale = _create_dsv4_epilogue_buffers(self, q, num_tokens)
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
        output=dsv4_output,
        output_sf=o_lora_bmm_input_scale,
        latent_cache=latent_cache,
        q_pe=q_pe,
        quant_q_buffer=quant_q_buffer,
        quant_scale_qkv=quant_scale_qkv,
        sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk_indices),
        dsv4_inv_rope_cos_sin_cache=inverse_rope_cos_sin,
        enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
        # Fused kv-norm: `latent_cache` is the RAW kv_a_proj output, so the kernel must
        # apply the norm before RoPE/quant/write.
        kv_norm_weight=(
            self.kv_a_layernorm.weight if getattr(self, "_fused_kv_norm_active", False) else None
        ),
        kv_norm_eps=float(self.kv_a_layernorm.variance_epsilon),
    )
    # Do NOT clear the fused FP8-Q buffers or `_fused_kv_norm_active` here. In a mixed
    # batch this runs before the generation half, which reads the flag to decide whether
    # to pass the norm weight -- clearing it made the generation half write
    # UN-normalized rows into the KV cache. `forward_sparse_attn` clears it once both
    # halves are done.

    if enable_dsv4_epilogue_fusion:
        assert dsv4_output is not None and o_lora_bmm_input_scale is not None
        return dsv4_output, o_lora_bmm_input_scale

    assert output is not None
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
    # A 3D token-major output is the internal fusion marker, avoiding
    # algorithm-specific parameters in the shared MLA custom-op schema.
    enable_dsv4_epilogue_fusion = output.ndim == 3
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
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
    # Also pre-launch on HCA layers (no indexer, so no indexer overlap): left inside
    # the parallel LN block below it only started ~27 us into the prologue, whereas
    # compressor_stream runs it underneath the kv_a_proj GEMM. On those layers nothing
    # else is queued behind it, so record the completion event here.
    _prelaunch_compressor = _use_indexer_overlap or (
        _v4_extra_overlap and do_multi_stream() and self.compressor_stream is not None
    )
    if _prelaunch_compressor:
        self.dsv4_compressor_start_event.record()
        with torch.cuda.stream(self.compressor_stream):
            self.dsv4_compressor_start_event.wait()
            self.compressor(hidden_states, attn_metadata)
            if not _use_indexer_overlap:
                self.dsv4_compressor_event.record()

    # Precompute QR-independent indexer work while the caller stream prepares KV.
    # Passing pre_aux later prevents the indexer from launching this work again.
    _indexer_pre_aux = None
    if _use_indexer_overlap:
        _indexer_pre_aux = self.indexer.precompute_aux(hidden_states, attn_metadata)

    q, kv = self.kv_a_proj_with_mqa(hidden_states).split(
        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], -1
    )

    # Fused kv-norm: the KV kernels apply kv_a_layernorm themselves, so skip the
    # standalone RMSNorm and the concat and hand them the RAW latent. Resolved here,
    # ahead of the Q branch, because the KV decision is made first.
    self._fused_q_rope_cos_sin, self._fused_q_rope_specs_cached = _fused_q_rope_specs(
        self, attn_metadata, num_contexts, num_generations
    )
    self._fused_kv_norm_active = _is_fused_prologue_active(
        self,
        num_contexts=num_contexts,
        num_generations=num_generations,
        rope_specs=self._fused_q_rope_specs_cached,
    )
    self._fused_kv_norm_hoisted = False

    if self._fused_kv_norm_active:
        # Launched ahead of q_a_layernorm, not after: the KV kernel needs only the raw
        # latent, so recording the gating event first lets the two run concurrently on
        # their own streams. The join stays before the generation half. Pure-decode
        # only -- with contexts the context half writes the same cache first, and
        # decode is where the idle span exists.
        if num_generations > 0 and num_contexts == 0:
            self._fused_kv_norm_hoisted = _launch_fused_kv_norm_gen(
                self, kv[num_ctx_tokens:, ...], attn_metadata
            )
        q = self.q_a_layernorm(q)
    else:
        q, kv = maybe_execute_in_parallel(
            lambda: self.q_a_layernorm(q),
            lambda: self.kv_a_layernorm(kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
    compressed_kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
    qr = q
    if self._fused_kv_norm_active:
        # `kv` is already the [compressed_kv | k_pe] layout the kernels read as
        # `fuse_buf`; the views above are dead on V4 (the compressor and indexer read
        # `hidden_states`/`qr`, not the latent).
        #
        # It is a last-dim slice, so its row stride is q_lora_rank + 512, not 512. The
        # kernel reads stride(0) off the tensor -- calling .contiguous() would
        # reintroduce the copy this fusion removes.
        assert kv.stride(-1) == 1, "fused kv-norm needs a unit-stride latent row"
        latent_cache = kv
    else:
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

    # Use the CuTe BF16 Q projection only for ratio-4 CSA layers with unquantized,
    # bias-free weights. TRTLLM_MLA_Q_B_PROJ_USE_CUTE_DSL disables this path.
    # The fused FP8 Q branch owns q_b_proj's output, so the two are exclusive by
    # construction rather than by assertion.
    _use_q_b_cute = (
        self.has_dsv4_indexer
        and os.environ.get("TRTLLM_MLA_Q_B_PROJ_USE_CUTE_DSL", "1") == "1"
        and self.q_b_proj.bias is None
        and self.q_b_proj.weight.dtype == torch.bfloat16
        and not _is_fused_q_fp8_quant_enabled(
            self, num_generations=num_generations, num_contexts=num_contexts
        )
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

    def _fused_q_fp8_quant_enabled() -> bool:
        return _is_fused_q_fp8_quant_enabled(
            self, num_generations=num_generations, num_contexts=num_contexts
        )

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
        # Rope args leave `q_pe` unwritten, and only `_fused_kv_norm_active` makes
        # generation skip the un-fused RoPE that would read it. Dropping just the rope
        # args keeps the standalone q_nope FP8 quant, correct without the fold.
        coupled = self._fused_kv_norm_active
        return _deepseek_v4_q_b_layernorm_fused_fp8(
            self,
            q_proj,
            self._fused_q_rope_cos_sin if coupled else None,
            self._fused_q_rope_specs_cached if coupled else None,
        )

    def _q_branch():
        if _use_q_b_cute:
            q_proj = _q_b_proj_cute_dsl_bf16(q, self.q_b_proj.weight)
            # The context path detects fusion from these buffers, so clear stale state.
            self._fused_quant_q_buffer = None
            self._fused_q_pe = None
            return _q_b_layernorm(q_proj)
        q_proj = self.q_b_proj(q)
        if _fused_q_fp8_quant_enabled():
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

            # Keep cross-stream outputs alive on the consuming stream. The fused-Q
            # buffers count too: allocated on compressor_stream inside _q_branch, read
            # by FMHA on this one.
            cur_stream = torch.cuda.current_stream()
            for _crossed in (
                q,
                topk_indices,
                self._fused_quant_q_buffer,
                self._fused_q_pe,
            ):
                if _crossed is not None:
                    _crossed.record_stream(cur_stream)
        elif _prelaunch_compressor:
            # Already in flight on compressor_stream since before kv_a_proj, so the
            # caller stream only runs the Q branch and joins. Running
            # _compressor_branch here as well would execute the compressor twice.
            q = _q_branch()
            self.dsv4_compressor_event.wait()
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

    context_o_lora_bmm_input = None
    generation_o_lora_bmm_input = None
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

        context_o_lora_bmm_input = forward_context_sparse_attn(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            None if enable_dsv4_epilogue_fusion else output[:num_ctx_tokens, :],
            position_ids=ctx_position_ids,
            latent_cache=latent_cache_ctx,
            topk_indices=topk_indices_ctx,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
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

        generation_o_lora_bmm_input = forward_generation_sparse_attn(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            None if enable_dsv4_epilogue_fusion else output[num_ctx_tokens:num_tokens, :],
            position_ids=gen_position_ids,
            latent_cache=latent_cache_gen,
            topk_indices=topk_indices_gen,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
        )

    if enable_dsv4_epilogue_fusion:
        assert context_o_lora_bmm_input is None or isinstance(context_o_lora_bmm_input, tuple)
        assert generation_o_lora_bmm_input is None or isinstance(generation_o_lora_bmm_input, tuple)
        # The fused kernel output is group-first, which BCG cannot slice on
        # dim 0. Write O-LoRA as token-first so replay can slice the bucket.
        _run_dsv4_o_lora_bmms(
            self,
            output,
            num_ctx_tokens,
            num_tokens,
            context_o_lora_bmm_input,
            generation_o_lora_bmm_input,
        )

    # Both halves have consumed the fused-prologue state; clear it here rather than in
    # either half, so a mixed batch cannot have the context half hide the flag from the
    # generation half.
    self._fused_quant_q_buffer = None
    self._fused_q_pe = None
    self._fused_q_rope_cos_sin = None
    self._fused_q_rope_specs_cached = []
    self._fused_kv_norm_active = False
    self._fused_kv_norm_hoisted = False

    # Join the prev_topk copy forked in sparse_attn_indexer; CUDA graph
    # capture requires the join within the same layer's forward.
    if self.indexer is not None:
        self.indexer.maybe_join_prev_topk_copy()


class DeepSeekV4Hooks(MLASparseHooks):
    """Typed DeepSeek-V4 adapter for the shared MLA module."""

    mqa_rope_append = False
    need_absorption = False
    need_dense_mha = False
    need_default_o_proj = False

    def get_mqa_aux_stream(self, mla: MLA) -> Optional[torch.cuda.Stream]:
        if _has_dsv4_indexer(mla):
            return mla.aux_stream_dict.get(AuxStreamType.MlaIndexerAux)
        return mla.aux_stream

    def initialize(self, mla: MLA) -> None:
        initialize_sparse_attn(mla)

    def create_weights(self, mla: MLA) -> None:
        create_sparse_attn_weights(mla)

    def prepare_outputs(
        self,
        mla: MLA,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> list[torch.Tensor]:
        return prepare_sparse_attn_outputs(mla, hidden_states, attn_metadata)

    def forward(
        self,
        mla: MLA,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_output: list[torch.Tensor],
    ) -> None:
        forward_sparse_attn(mla, position_ids, hidden_states, attn_metadata, attn_output)

    def project_output(
        self,
        mla: MLA,
        attn_output: list[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        return project_sparse_attn_output(
            mla,
            attn_output,
            position_ids,
            attn_metadata,
            all_reduce_params,
        )


register_mla_sparse_hooks("deepseek_v4", DeepSeekV4Hooks)
