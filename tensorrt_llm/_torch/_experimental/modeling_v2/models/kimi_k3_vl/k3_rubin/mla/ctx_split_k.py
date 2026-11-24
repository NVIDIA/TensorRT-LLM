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
"""Kimi K3 MLA cached-prefix context on Rubin: the FP8 split-K context FMHA.

For an MLA layer with an NVFP4 latent cache, the base ``MLA.forward_context_with_chunked_prefill_fp4``
runs every cached-prefix partition, and then the current chunk, through the attention op's FP8
context MLA FMHA into a temporary output and merges it into the running output with
``merge_chunked_attention_for_mla``; K is the BF16 per-head concat of k_nope and the shared k_pe.
:class:`KimiK3MlaContextSplitKMixin` overrides it where :func:`supports_k3_fp8_split_k_context`
holds (SM107, matching Q / KV heads, head dimensions 128 / 64 / 128, the CuTe DSL kernel imports)
and Q and the output are BF16 without an output scale:

* each partition runs the Rubin FP8 split-K context FMHA (``trtllm::k3r_context_fmha_rubin``,
  ``kernels/k3_context_fmha.py``), which reads K-nope ``[T, H, 128]``, the shared K-pe
  ``[T, 1, 64]`` and V as they are and merges the partition into the output in its epilogue
  (``merge_op_tensor`` selects copy / merge / skip per request);
* its inputs are unit-scale E4M3: Q once per layer call, each partition's kv_b_proj output and
  K-pe. ``TRTLLM_K3_MLA_CTX_TRITON_QUANT=1`` quantizes Q and the K-pe rows (the current chunk's
  read in place from ``latent_cache``) with the Triton kernels of ``kernels/fp8_static_quant.py``,
  otherwise ``static_quantize_e4m3_per_tensor`` does; the bytes are identical;
* where kv_b_proj is a bias-free FP8 block-scale GEMM with its weight scale in the SM107 CuTe
  MXFP8 layout (:meth:`KimiK3MlaContextSplitKMixin._kv_b_takes_mxfp8`), each cached-prefix
  partition is gathered from the FP4 cache straight into MXFP8
  (``kernels/fp4_mla_context_gather.py``), the current chunk is quantized with
  ``fp8_quantize_1x128_packed_ue8m0``, and kv_b_proj runs as ``cute_dsl_mxfp8_gemm_rubin`` with
  FP8 output; otherwise the BF16 gather and kv_b_proj run and their output is quantized.

There is no switch: the path is taken whenever it applies. Elsewhere the base partition loop
runs, with ``KimiK3MlaContextFp8Mixin``'s inputs.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from tensorrt_llm._torch.attention.backends import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    FP4_MLA_K_RESIDUAL_DIM,
    _fp4_mla_attention_backend,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _validate_fp4_mla_kv_storage_shape,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _materialize_fp4_mla_device_page_table_for_forward,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..kernels import fp8_static_quant
from .ctx_fp8 import _TRITON_QUANT_FEATURE, _mla_ctx_triton_quant_enabled

# Head dimensions (K-nope, K-pe, V) the split-K context FMHA is built for.
_SPLIT_K_HEAD_DIMS = (128, 64, 128)


def split_k_context_unavailable_reason(attn) -> Optional[str]:
    """Why ``attn`` cannot run the Rubin FP8 split-K context FMHA, or None if it can."""
    sm_version = get_sm_version()
    if sm_version != 107:
        return f"requires SM107, got SM{sm_version}"
    if attn.num_heads <= 0 or attn.num_heads != attn.num_kv_heads:
        return (
            "requires matching positive Q/KV head counts per rank, "
            f"got {attn.num_heads}/{attn.num_kv_heads}"
        )
    head_dims = (attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim)
    if head_dims != _SPLIT_K_HEAD_DIMS:
        return f"requires head dimensions (K-nope, K-pe, V)={_SPLIT_K_HEAD_DIMS}, got {head_dims}"
    from ..kernels import k3_context_fmha_op

    if k3_context_fmha_op.IS_K3_CONTEXT_FMHA_AVAILABLE:
        return None
    if k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR is not None:
        return f"CuTe DSL kernel import failed: {k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR!r}"
    return "CuTe DSL kernel lacks split-K epilogue merge support"


def supports_k3_fp8_split_k_context(attn) -> bool:
    """Whether the attention backend ``attn`` can run the Rubin FP8 split-K context FMHA."""
    reason = split_k_context_unavailable_reason(attn)
    if reason is None:
        return True
    logger.info_once(
        f"Kimi K3 MLA: Rubin FP8 split-K context FMHA disabled: {reason}; NVFP4 KV-cache layers "
        "keep the FP8 context MLA FMHA partition loop.",
        key=("k3_mla_ctx_split_k_unavailable", reason),
    )
    return False


def run_k3_split_k_context_partition(
    attn,
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_stats: torch.Tensor,
    merge_op: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    *,
    max_q_len: int,
    max_kv_len: int,
    is_causal: bool,
) -> None:
    """Attend FP8 Q to one FP8 KV partition and merge it into ``output`` / ``softmax_stats``.

    ``q`` ``[T_q, H * 192]``, ``k_nope`` / ``v`` ``[T_kv, H * 128]`` with unit inner stride,
    ``k_pe`` ``[T_kv, 64]``, ``output`` ``[T_q, H * 128]`` BF16, ``softmax_stats``
    ``[T_q, H, 2]`` FP32; the softmax scale is the attention backend's.
    """
    from ..kernels import k3_context_fmha_op  # noqa: F401 registers trtllm::k3r_context_fmha_rubin

    sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + attn.qk_rope_head_dim) ** 0.5)
    num_heads = attn.num_heads
    logger.info_once(
        "Kimi K3 MLA context FMHA: trtllm::k3r_context_fmha_rubin (Rubin FP8 split-K, epilogue "
        f"merge) serves {'causal current-chunk' if is_causal else 'cached-prefix'} partitions, "
        f"{num_heads} heads per rank.",
        key=("k3_mla_ctx_split_k_fmha", is_causal, num_heads),
    )
    torch.ops.trtllm.k3r_context_fmha_rubin(
        q.view(q.shape[0], num_heads, 192),
        k_nope.view(k_nope.shape[0], num_heads, 128),
        k_pe.view(k_pe.shape[0], 1, 64),
        v.view(v.shape[0], num_heads, 128),
        output.view(q.shape[0], num_heads, 128),
        cu_q_seqlens,
        cu_kv_seqlens,
        softmax_stats,
        merge_op,
        max_q_len,
        max_kv_len,
        sm_scale,
        is_causal,
    )


def load_fp4_mla_chunked_kv_cache_mxfp8(
    metadata,
    layer_idx: int,
    *,
    num_ctx_cached_tokens: int,
    cu_chunked_seq_len: torch.Tensor,
    chunked_global_offset: torch.Tensor,
    chunked_max_seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Gather one cached-prefix partition of dense FP4 MLA V2 storage as MXFP8 kv_b_proj input.

    The partition and metadata are those of ``load_fp4_mla_chunked_kv_cache``. Returns
    ``((E4M3 [M, 512], UE8M0 [pad_up(M, 128) * 16]), E4M3 [M, 64])``: the BF16-rounded latent
    quantized 1x128 with scales in the K32 R128c4 layout ``cute_dsl_mxfp8_gemm_rubin`` reads, and
    the BF16-rounded RoPE part plus its residual in unit-scale E4M3.
    """
    if kv_lora_rank != 512 or qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "MXFP8 gather requires kv_lora_rank=512 and "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {kv_lora_rank} and {qk_rope_head_dim}."
        )
    if num_ctx_cached_tokens < 0:
        raise ValueError("FP4 MLA chunk gather token count must be non-negative.")
    if chunked_max_seq_len < 0:
        raise ValueError("FP4 MLA chunk gather max sequence length must be non-negative.")
    if num_ctx_cached_tokens > 0 and chunked_max_seq_len == 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires a positive max length.")
    num_contexts = int(metadata.num_contexts)
    if num_ctx_cached_tokens > 0 and num_contexts <= 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires context requests.")
    tensors = (cu_chunked_seq_len, chunked_global_offset)
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA chunk gather metadata must use CUDA tensors.")
    if (
        cu_chunked_seq_len.dtype != torch.int64
        or cu_chunked_seq_len.ndim != 1
        or cu_chunked_seq_len.numel() < num_contexts + 1
        or not cu_chunked_seq_len.is_contiguous()
    ):
        raise ValueError(
            "FP4 MLA chunk gather requires a contiguous int64 cumulative-length tensor."
        )
    if (
        chunked_global_offset.dtype != torch.int64
        or chunked_global_offset.ndim != 1
        or chunked_global_offset.numel() < num_contexts
        or not chunked_global_offset.is_contiguous()
    ):
        raise ValueError("FP4 MLA chunk gather requires contiguous int64 global offsets.")
    if cu_chunked_seq_len.device != chunked_global_offset.device:
        raise ValueError("FP4 MLA chunk gather metadata tensors must share one device.")

    device = cu_chunked_seq_len.device
    compressed_kv = torch.empty(
        (num_ctx_cached_tokens, kv_lora_rank), dtype=torch.float8_e4m3fn, device=device
    )
    compressed_sf = torch.empty(
        (triton.cdiv(num_ctx_cached_tokens, 128) * 128 * (kv_lora_rank // 32),),
        dtype=torch.uint8,
        device=device,
    )
    k_pe = torch.empty(
        (num_ctx_cached_tokens, qk_rope_head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    if num_ctx_cached_tokens == 0:
        return (compressed_kv, compressed_sf), k_pe

    state = metadata.fp4_mla_state
    if not bool(getattr(state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA chunk gather requires fixed-stride device page metadata.")
    _materialize_fp4_mla_device_page_table_for_forward(metadata)
    page_table_stride = int(state.page_table_stride)
    page_ids = state._paged_kv_indices
    if (
        page_table_stride <= 0
        or not isinstance(page_ids, torch.Tensor)
        or page_ids.dtype != torch.int32
        or not page_ids.is_cuda
        or page_ids.device != device
        or page_ids.numel() < num_contexts * page_table_stride
    ):
        raise RuntimeError("FP4 MLA chunk gather received invalid device page metadata.")

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=_fp4_mla_attention_backend(),
    )
    if storage_head_dim != head_dim + FP4_MLA_K_RESIDUAL_DIM:
        raise RuntimeError(
            "FP4 MLA chunk gather requires K residual storage for BF16 reconstruction."
        )
    from ..kernels.fp4_mla_context_gather import gather_mxfp8

    logger.info_once(
        "Kimi K3 MLA context gather: cached-prefix chunks are read from the FP4 cache as MXFP8 "
        "latent + FP8 K-pe (k3_rubin.kernels.fp4_mla_context_gather.gather_mxfp8).",
        key="k3_mla_ctx_gather_mxfp8",
    )
    gather_mxfp8(
        compressed_kv,
        compressed_sf,
        k_pe,
        kv_cache,
        sf_cache.view(torch.float8_e4m3fn),
        page_ids,
        cu_chunked_seq_len,
        chunked_global_offset,
        _get_fp4_mla_global_scale(metadata, kv_cache.device),
        chunked_max_seq_len,
        metadata.page_size,
        page_table_stride=page_table_stride,
        num_contexts=num_contexts,
    )
    return (compressed_kv, compressed_sf), k_pe


class KimiK3MlaContextSplitKMixin:
    """The Rubin FP8 split-K context FMHA for NVFP4-cache cached-prefix context (module doc).

    Mixed in before ``MLA``; it only overrides ``forward_context_with_chunked_prefill_fp4``.
    """

    def _uses_k3_split_k_context(self, q: torch.Tensor, output: torch.Tensor) -> bool:
        mha = self.mha
        return (
            q.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and self.out_scale is None
            and isinstance(mha, TrtllmAttention)
            and mha.uses_fp4_mla_attention
            and supports_k3_fp8_split_k_context(mha)
        )

    def _kv_b_takes_mxfp8(self) -> bool:
        """Whether kv_b_proj runs as the SM107 MXFP8 GEMM on an MXFP8 cached-prefix gather.

        Needs SM107 with the CuTe DSL Rubin kernels, the 512 / 64 latent / RoPE dimensions of the
        gather, and a bias-free FP8 block-scale kv_b_proj whose UE8M0 weight scale is in the CuTe
        MXFP8 layout: one that takes a prequantized CuTe-layout input, or a ``Linear`` on the
        CuTe DSL block-scale GEMM.
        """
        kv_b_proj = self.kv_b_proj
        quant_mode = getattr(getattr(kv_b_proj, "quant_config", None), "quant_mode", None)
        weight = getattr(kv_b_proj, "weight", None)
        weight_scale = getattr(kv_b_proj, "weight_scale", None)
        return (
            get_sm_version() == 107
            and IS_CUTLASS_DSL_RUBIN_AVAILABLE
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and getattr(kv_b_proj, "bias", None) is None
            and (
                getattr(kv_b_proj, "supports_prequantized_input", False)
                or (
                    quant_mode is not None
                    and quant_mode.has_fp8_block_scales()
                    and (
                        getattr(kv_b_proj, "use_cute_dsl_blockscaling_mm", False)
                        or getattr(kv_b_proj, "disable_deep_gemm", False)
                    )
                )
            )
            and weight is not None
            and weight.dtype == torch.float8_e4m3fn
            and weight_scale is not None
            and weight_scale.dtype == torch.uint8
        )

    def forward_context_with_chunked_prefill_fp4(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self._uses_k3_split_k_context(q, output):
            return super().forward_context_with_chunked_prefill_fp4(
                q, compressed_kv, latent_cache, attn_metadata, output
            )
        return self._forward_context_split_k_fp4(
            q, compressed_kv, latent_cache, attn_metadata, output
        )

    def _forward_context_split_k_fp4(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """FP4 MLA chunked prefill with every partition on the split-K context FMHA."""
        mha = self.mha
        if latent_cache is None:
            raise RuntimeError("FP4 MLA chunked prefill requires latent_cache.")
        use_triton_quant = _mla_ctx_triton_quant_enabled()
        kv_b_mxfp8 = self._kv_b_takes_mxfp8()
        logger.info_once(
            "Kimi K3 MLA cached-prefix context: NVFP4 KV-cache layers run the Rubin FP8 split-K "
            "context FMHA (trtllm::k3r_context_fmha_rubin); cached prefix: "
            + (
                "MXFP8 gather + FP8-output MXFP8 kv_b GEMM"
                if kv_b_mxfp8
                else "BF16 gather + kv_b_proj + FP8 quantization"
            )
            + "; Q FP8 quantization: "
            + (
                f"static_quantize_e4m3 (Triton, {_TRITON_QUANT_FEATURE})"
                if use_triton_quant
                else "static_quantize_e4m3_per_tensor"
            )
            + ".",
            key=("k3_mla_ctx_split_k", kv_b_mxfp8, use_triton_quant),
        )

        # Rotate Q and the current K in place and append the chunk to the FP4 cache; the current
        # chunk's rotated K-pe is then read from latent_cache.
        mha.append_fp4_mla_context_cache(q, latent_cache, attn_metadata)
        current_k_pe = latent_cache.reshape(-1, self.kv_lora_rank + self.qk_rope_head_dim)[
            :, self.kv_lora_rank :
        ]
        fp8_unit_scale = mha.kv_cache_scaling_factor
        if use_triton_quant:
            q_fp8 = fp8_static_quant.static_quantize_e4m3(q.contiguous(), fp8_unit_scale)
        else:
            q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, fp8_unit_scale)
        context_cu_q_seqlens = attn_metadata.mla_prepare_ctx_cu_seqlens()
        if context_cu_q_seqlens is None:
            raise RuntimeError("K3 split-K context FMHA requires context requests.")

        # The first partition of every request writes the output and its stats (merge op 2).
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device=q.device,
        )
        self.temp_softmax_stats_tensor = None
        kv_split = [self.num_heads_tp * self.qk_nope_head_dim, self.num_heads_tp * self.v_head_dim]

        def expand_kv(
            partition_compressed_kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            partition_k_pe: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if kv_b_mxfp8:
                if isinstance(partition_compressed_kv, tuple):
                    activation, activation_scale = partition_compressed_kv
                else:
                    activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
                        partition_compressed_kv.contiguous()
                    )
                # The FP8 output keeps the GEMM's BF16 rounding (unit output scale).
                partition_kv = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                    activation,
                    self.kv_b_proj.weight,
                    activation_scale,
                    self.kv_b_proj.weight_scale,
                    output_dtype=torch.float8_e4m3fn,
                )
            else:
                partition_kv, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    self.kv_b_proj(partition_compressed_kv), fp8_unit_scale
                )
            if partition_k_pe.dtype == torch.float8_e4m3fn:
                k_pe_source = "FP8 from the gather"
            elif use_triton_quant and fp8_static_quant.supports_rows(partition_k_pe):
                k_pe_source = "static_quantize_e4m3_rows (Triton)"
                partition_k_pe = fp8_static_quant.static_quantize_e4m3_rows(
                    partition_k_pe, fp8_unit_scale
                )
            else:
                k_pe_source = "static_quantize_e4m3_per_tensor"
                partition_k_pe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    partition_k_pe.contiguous(), fp8_unit_scale
                )
            logger.info_once(
                "Kimi K3 MLA context expand: per-chunk kv_b_proj runs as "
                + (
                    "cute_dsl_mxfp8_gemm_rubin (MXFP8 in, FP8 out)"
                    if kv_b_mxfp8
                    else "kv_b_proj (BF16) + static_quantize_e4m3_per_tensor"
                )
                + f"; K-pe: {k_pe_source}; K-nope / K-pe / V go to the split-K FMHA unconcatenated.",
                key=("k3_mla_ctx_split_k_expand", kv_b_mxfp8, k_pe_source),
            )
            partition_k_nope, partition_v = partition_kv.split(kv_split, -1)
            return partition_k_nope, partition_k_pe, partition_v

        def run_partition(
            partition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            cu_kv_seqlens: torch.Tensor,
            max_kv_len: int,
            merge_op: torch.Tensor,
            is_causal: bool,
        ) -> None:
            partition_k_nope, partition_k_pe, partition_v = partition
            run_k3_split_k_context_partition(
                mha,
                q_fp8,
                partition_k_nope,
                partition_k_pe,
                partition_v,
                output,
                self.softmax_stats_tensor,
                merge_op,
                context_cu_q_seqlens,
                cu_kv_seqlens,
                max_q_len=attn_metadata.max_ctx_seq_len,
                max_kv_len=max_kv_len,
                is_causal=is_causal,
            )

        for loop_idx in range(attn_metadata.chunked_loop_num):
            cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_chunked_tokens = int(
                attn_metadata.host_cu_chunked_seq_len[loop_idx, attn_metadata.num_contexts]
            )
            chunked_max_seq_len = int(attn_metadata.max_chunk_len_per_loop[loop_idx])
            if kv_b_mxfp8:
                chunked_compressed_kv, chunked_k_pe = load_fp4_mla_chunked_kv_cache_mxfp8(
                    attn_metadata,
                    mha.layer_idx,
                    num_ctx_cached_tokens=total_chunked_tokens,
                    cu_chunked_seq_len=cu_chunked_seq_len,
                    chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                    chunked_max_seq_len=chunked_max_seq_len,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                )
            else:
                chunked_compressed_kv, chunked_k_pe = mha.load_chunked_kv_cache_for_mla(
                    metadata=attn_metadata,
                    num_ctx_cached_tokens=total_chunked_tokens,
                    cu_chunked_seq_len=cu_chunked_seq_len,
                    chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                    chunked_max_seq_len=chunked_max_seq_len,
                    out_dtype=q.dtype,
                )
            partition = expand_kv(chunked_compressed_kv, chunked_k_pe)
            chunked_compressed_kv = None
            chunked_k_pe = None
            run_partition(
                partition,
                cu_chunked_seq_len,
                chunked_max_seq_len,
                attn_metadata.merge_op_tensor[loop_idx],
                is_causal=False,
            )

        run_partition(
            expand_kv(compressed_kv, current_k_pe),
            context_cu_q_seqlens,
            attn_metadata.max_ctx_seq_len,
            attn_metadata.merge_op_tensor[attn_metadata.chunked_loop_num],
            is_causal=True,
        )
        return output


__all__ = [
    "KimiK3MlaContextSplitKMixin",
    "load_fp4_mla_chunked_kv_cache_mxfp8",
    "run_k3_split_k_context_partition",
    "split_k_context_unavailable_reason",
    "supports_k3_fp8_split_k_context",
]
