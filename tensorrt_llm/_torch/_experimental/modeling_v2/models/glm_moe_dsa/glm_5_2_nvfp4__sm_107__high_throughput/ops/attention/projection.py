# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation projections with fused indexer rows and row-RMS q_b GEMM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.module import (
    DSAHooks,
    _forward_dsa_attn,
    should_use_short_mha,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

from ..gemm.low_m import _device_limits, low_m_gemm
from .kv_norm_pack import is_kv_norm_pack_supported, kv_norm_pack
from .q_a_norm import is_q_a_norm_early_supported, q_a_norm_early

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.mla import MLA


def build_projection_weights(mla) -> None:
    indexer = mla.mqa.indexer
    old = getattr(mla, "fused_a_indexer_weight", None)
    mla.fused_a_indexer_weight = None
    linear = mla.kv_a_proj_with_mqa
    if indexer is None or linear.has_any_quant or linear.bias is not None:
        return
    parts = (linear.weight, indexer.wk.weight, indexer.weights_proj.weight)
    if any(p.dtype != torch.bfloat16 or p.ndim != 2 or p.shape[1] != 6144 for p in parts):
        return
    shape = (sum(p.shape[0] for p in parts), 6144)
    if shape != (2784, 6144):
        return
    if old is None or tuple(old.shape) != shape or old.device != linear.weight.device:
        old = linear.weight.new_empty(shape)
    offset = 0
    with torch.no_grad():
        for part in parts:
            old[offset : offset + part.shape[0]].copy_(part)
            offset += part.shape[0]
    mla.fused_a_indexer_weight = old


def _q_b_norm_fold_ready(mla: MLA) -> bool:
    norm = mla.q_a_layernorm
    linear = mla.q_b_proj
    return (
        type(norm) is RMSNorm
        and not norm.use_gemma
        and not norm.use_cuda_tile
        and not norm.return_hp_output
        and not norm.is_nvfp4
        and mla._resolve_qa_fused_scale() is None
        and linear.bias is None
        and not linear.has_any_quant
        and linear.lora is None
        and linear._locality_domain_weight_shards is None
        and linear.weight.dtype == norm.weight.dtype == torch.bfloat16
        and linear.weight.ndim == 2
        and linear.weight.shape[1] == mla.q_lora_rank
        and mla.q_lora_rank % 128 == 0
        and linear.weight.is_contiguous()
        and norm.weight.is_contiguous()
        and linear.weight.data_ptr() % 32 == 0
        and norm.weight.data_ptr() % 16 == 0
    )


def _fold_q_b(mla: MLA, raw_q: torch.Tensor) -> bool:
    return (
        33 <= raw_q.shape[0] <= 64
        and _q_b_norm_fold_ready(mla)
        and raw_q.dtype == torch.bfloat16
        and raw_q.is_cuda
        and raw_q.stride(1) == 1
        and raw_q.stride(0) % 8 == 0
        and raw_q.data_ptr() % 32 == 0
        and not torch.is_grad_enabled()
    )


def _fused_a_supported(mla: MLA, inputs: torch.Tensor, weight: torch.Tensor) -> bool:
    linear = mla.kv_a_proj_with_mqa
    return (
        not linear.has_any_quant
        and linear.bias is None
        and linear.lora is None
        and linear._locality_domain_weight_shards is None
        and not linear.use_cute_dsl_bf16_gemm
        and inputs.ndim == weight.ndim == 2
        and inputs.is_cuda
        and inputs.device == weight.device
        and inputs.dtype == weight.dtype == torch.bfloat16
        and inputs.is_contiguous()
        and weight.is_contiguous()
        and inputs.shape[1] == weight.shape[1] == 6144
        and weight.shape[0] % 8 == 0
        and inputs.data_ptr() % 32 == weight.data_ptr() % 32 == 0
        and not torch.is_grad_enabled()
    )


def _q_b_with_norm(mla, raw_q: torch.Tensor) -> torch.Tensor:
    from ...kernels.gemm.splitk import row_rms_launch, run_splitk_row_rms

    weight = mla.q_b_proj.weight
    sm_count, smem_capacity = _device_limits(raw_q.device)
    launch = row_rms_launch(
        raw_q.shape[0],
        weight.shape[0],
        weight.shape[1],
        sm_count=sm_count,
        smem_capacity=smem_capacity,
    )
    output = raw_q.new_empty((raw_q.shape[0], weight.shape[0]))
    run_splitk_row_rms(
        raw_q,
        weight,
        mla.q_a_layernorm.weight,
        output,
        get_env_enable_pdl(),
        launch,
        float(mla.q_a_layernorm.variance_epsilon),
    )
    return output


def _forward_dsa_proj_fallback(
    mla, position_ids, hidden_states, attn_metadata, *, allow_k_cache_write: bool
) -> list[torch.Tensor]:
    q, compressed_kv, k_pe = mla.kv_a_proj_with_mqa(hidden_states).split(
        [mla.q_lora_rank, mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1
    )
    q_pair, compressed_kv = maybe_execute_in_parallel(
        lambda: mla._q_a_layernorm_maybe_fused(q, return_norm_out=True),
        lambda: mla.kv_a_layernorm(compressed_kv),
        mla.ln_events[0],
        mla.ln_events[1],
        mla.aux_stream,
    )
    q, qr = q_pair
    latent_cache = torch.cat((compressed_kv, k_pe), dim=-1)
    q = mla.q_b_proj(q)
    outputs = [q, compressed_kv, k_pe, latent_cache]
    if (
        should_use_short_mha(mla, attn_metadata, position_ids)
        and attn_metadata.num_generations == 0
    ):
        return outputs
    indexer = mla.mqa.indexer
    if indexer is not None:
        outputs.extend(
            indexer.pre_indexer_proj(
                qr,
                hidden_states,
                position_ids,
                k_cache_metadata=attn_metadata if allow_k_cache_write else None,
            )
        )
    return outputs


def forward_dsa_proj(
    mla, position_ids, hidden_states, attn_metadata, *, allow_k_cache_write: bool = False
) -> list[torch.Tensor]:
    if (
        attn_metadata.num_contexts != 0
        or attn_metadata.num_generations == 0
        or isinstance(hidden_states, Fp4QuantizedTensor)
        or hidden_states.dtype != torch.bfloat16
    ):
        return _forward_dsa_proj_fallback(
            mla,
            position_ids,
            hidden_states,
            attn_metadata,
            allow_k_cache_write=allow_k_cache_write,
        )

    rows = hidden_states.shape[0]
    indexer = mla.mqa.indexer
    skip_indexer = indexer is not None and indexer.can_skip_projection(attn_metadata, rows)
    extended = mla.fused_a_indexer_weight if not skip_indexer else None
    indexer_wk_wp = None
    fused_a_weight = extended if extended is not None else mla.kv_a_proj_with_mqa.weight
    if 1 <= rows <= 64 and _fused_a_supported(mla, hidden_states, fused_a_weight):
        if rows <= 16:
            projected = torch.ops.trtllm.dsv3_fused_a_gemm_op(
                hidden_states, fused_a_weight.t(), None, None
            )
        else:
            projected = low_m_gemm(
                hidden_states,
                fused_a_weight,
                None,
                release_early=rows > 32 and _q_b_norm_fold_ready(mla),
            )
        if extended is not None:
            kv_a = projected[:, : mla.q_lora_rank + mla.kv_lora_rank + mla.qk_rope_head_dim]
            indexer_wk_wp = projected[:, kv_a.shape[1] :]
        else:
            kv_a = projected
    else:
        kv_a = mla.kv_a_proj_with_mqa(hidden_states)
    raw_q, kv, k_pe = kv_a.split([mla.q_lora_rank, mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
    folded = _fold_q_b(mla, raw_q)
    aux_projection = []

    def q_path():
        if folded:
            return _q_b_with_norm(mla, raw_q), None
        if mla._resolve_qa_fused_scale() is None and is_q_a_norm_early_supported(
            raw_q, mla.q_a_layernorm
        ):
            q_norm = q_a_norm_early(
                raw_q, mla.q_a_layernorm.weight, mla.q_a_layernorm.variance_epsilon
            )
            qr = q_norm
        else:
            q_norm, qr = mla._q_a_layernorm_maybe_fused(raw_q, return_norm_out=True)
        return mla.q_b_proj(q_norm), qr

    def kv_path():
        if is_kv_norm_pack_supported(kv, k_pe, mla.kv_a_layernorm):
            result = kv_norm_pack(
                kv, k_pe, mla.kv_a_layernorm.weight, mla.kv_a_layernorm.variance_epsilon
            )
        else:
            normalized = mla.kv_a_layernorm(kv)
            result = normalized, torch.cat((normalized, k_pe), dim=-1)
        if folded and indexer is not None and not skip_indexer:
            qr = mla._q_a_layernorm_maybe_fused(raw_q, return_norm_out=True)[1]
            aux_projection.extend((qr, indexer.wq_b(qr)))
        return result

    (q, qr), (compressed_kv, latent_cache) = maybe_execute_in_parallel(
        q_path, kv_path, mla.ln_events[0], mla.ln_events[1], mla.aux_stream
    )
    outputs = [q, compressed_kv, k_pe, latent_cache]
    if indexer is None:
        return outputs
    if skip_indexer:
        outputs.extend(indexer.reused_projection_outputs(attn_metadata, rows))
        return outputs
    q_in = None
    if aux_projection:
        qr, q_in = aux_projection
    outputs.extend(
        indexer.pre_indexer_proj(
            qr,
            hidden_states,
            position_ids,
            q_in=q_in,
            wk_wp_out=indexer_wk_wp,
            k_cache_metadata=attn_metadata if allow_k_cache_write else None,
        )
    )
    return outputs


class RubinDSAHooks(DSAHooks):
    def forward(self, mla, position_ids, hidden_states, attn_metadata, attn_output) -> None:
        outputs = forward_dsa_proj(
            mla, position_ids, hidden_states, attn_metadata, allow_k_cache_write=True
        )
        _forward_dsa_attn(
            mla, *outputs[:4], outputs[4:], position_ids, attn_metadata, attn_output[0]
        )

    def forward_custom_op(
        self, mla, hidden_states, position_ids, attn_output, latent_cache_gen
    ) -> bool:
        from tensorrt_llm._torch.attention.backends.sparse.dsa.custom_ops import (
            maybe_bcg_mla_dsa_attn_inplace,
        )

        from .projection_op import indexer_cache_for_projection, mla_dsa_proj

        indexer_k_cache = indexer_cache_for_projection(mla.layer_idx_str)
        if isinstance(hidden_states, Fp4QuantizedTensor):
            outputs = mla_dsa_proj(
                hidden_states.unquantized_hidden_states,
                position_ids,
                mla.layer_idx_str,
                hidden_states.fp4_tensor,
                hidden_states.scaling_factor,
                indexer_k_cache,
            )
        else:
            outputs = mla_dsa_proj(
                hidden_states, position_ids, mla.layer_idx_str, indexer_k_cache=indexer_k_cache
            )
        maybe_bcg_mla_dsa_attn_inplace(
            *outputs[:4], outputs[4:], position_ids, mla.layer_idx_str, attn_output[0]
        )
        return True
