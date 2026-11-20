# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer execution state and native prefill calls."""

from __future__ import annotations

import math
from typing import Any, Literal

import flashinfer
import torch

_WORKSPACE_BYTES = 128 * 1024 * 1024
_LOG2_TO_LN = math.log(2.0)
_BlockScaledQKMode = Literal["mxfp8", "nvfp4"]


class FlashInferPrefillRunner:
    """Bind prefill geometry to caller-owned, per-stream plans and workspaces.

    Give a captured graph a separate cache if subsequent eager requests on its
    stream can replan the same geometry with a different padding mask.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        plan_cache: dict[str, Any],
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self._cache = plan_cache

    def plan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        stream = torch.cuda.current_stream(q.device)
        stream_key = (q.device, stream.cuda_stream)

        workspace_cache = self._cache.setdefault("v2_flashinfer_workspaces", {})
        workspace = workspace_cache.get(stream_key)
        if workspace is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=q.device,
            )
            workspace_cache[stream_key] = workspace

        plan_key = (
            *stream_key,
            batch_size,
            query_length,
            key_value_length,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q.dtype,
            is_causal,
            key_padding_mask is not None,
        )
        plan_cache = self._cache.setdefault("v2_flashinfer_plans", {})
        plan_entry = plan_cache.get(plan_key)
        if plan_entry is None:
            plan_entry = {
                "wrapper": flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD"),
                "mask_signature": None,
                "mask_reference": None,
                "is_planned": False,
            }
            plan_cache[plan_key] = plan_entry

        mask_signature = None
        version_tracked = key_padding_mask is None or not torch.is_inference(key_padding_mask)
        if key_padding_mask is not None and version_tracked:
            mask_signature = (
                id(key_padding_mask),
                key_padding_mask.data_ptr(),
                key_padding_mask._version,
            )
        if (
            not version_tracked
            or not plan_entry["is_planned"]
            or plan_entry["mask_signature"] != mask_signature
        ):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "FlashInfer CUDA capture requires a warmed plan and a version-tracked padding mask."
                )

            query_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * query_length
            )
            key_value_indptr = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * key_value_length
            )
            custom_mask = None
            if key_padding_mask is not None:
                custom_mask = (
                    key_padding_mask.to(device=q.device, dtype=torch.bool)
                    .unsqueeze(1)
                    .expand(-1, query_length, -1)
                    .reshape(-1)
                    .contiguous()
                )

            stream.synchronize()
            plan_entry["wrapper"].plan(
                query_indptr,
                key_value_indptr,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                custom_mask=custom_mask,
                causal=is_causal,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                o_data_type=q.dtype,
            )
            plan_entry["mask_signature"] = mask_signature
            plan_entry["mask_reference"] = key_padding_mask
            plan_entry["is_planned"] = True

        return plan_entry["wrapper"]


def run_single_prefill(
    runner: FlashInferPrefillRunner,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    custom_mask = None
    if key_padding_mask is not None:
        custom_mask = (
            key_padding_mask[0]
            .to(device=q.device, dtype=torch.bool)
            .unsqueeze(0)
            .expand(q.shape[1], -1)
            .contiguous()
        )
    output, lse = flashinfer.single_prefill_with_kv_cache(
        q[0].contiguous(),
        k[0].contiguous(),
        v[0].contiguous(),
        o_dtype=q.dtype,
        custom_mask=custom_mask,
        causal=is_causal,
        sm_scale=runner.scale,
        return_lse=True,
    )
    return output.unsqueeze(0), lse.transpose(0, 1).unsqueeze(0) * _LOG2_TO_LN


@torch.compiler.disable
def run_batch_prefill(
    runner: FlashInferPrefillRunner,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper = runner.plan(q, k, is_causal=is_causal, key_padding_mask=key_padding_mask)
    output, lse = wrapper.run(
        q.contiguous().view(-1, runner.num_heads, runner.head_dim),
        k.contiguous().view(-1, runner.num_kv_heads, runner.head_dim),
        v.contiguous().view(-1, runner.num_kv_heads, runner.head_dim),
        return_lse=True,
    )
    batch_size, query_length = q.shape[:2]
    output = output.view(batch_size, query_length, runner.num_heads, runner.head_dim)
    lse = lse.view(batch_size, query_length, runner.num_heads).transpose(1, 2)
    return output, lse * _LOG2_TO_LN


def _quantize_fp8_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    scale = (tensor.abs().amax().float() / fp8_info.max).clamp(min=1e-12)
    quantized = (tensor / scale).clamp(min=fp8_info.min, max=fp8_info.max)
    return quantized.to(fp8_dtype).contiguous(), scale


def run_blockscaled_sm10x(
    runner: FlashInferPrefillRunner,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qk_mode: _BlockScaledQKMode,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if runner.head_dim != 128:
        raise ValueError(
            "FlashInfer SM100/SM103 block-scaled attention requires head_dim=128, "
            f"got {runner.head_dim}."
        )
    from flashinfer.attention.cute_dsl.fmha_blockscaled import cute_dsl_fmha_blockscaled_prefill
    from flashinfer.cute_dsl.attention.fmha.quantize import quantize_blockscaled_qk

    q_store, k_store, q_sf, k_sf, q_scale, k_scale = quantize_blockscaled_qk(q, k, qk_mode)
    v_store, v_scale = _quantize_fp8_per_tensor(v)
    output = torch.empty_like(q)
    lse = torch.empty(
        (q.shape[0], q.shape[1], runner.num_heads),
        dtype=torch.float32,
        device=q.device,
    )
    cute_dsl_fmha_blockscaled_prefill(
        q_store,
        k_store,
        q_sf,
        k_sf,
        v_store,
        output,
        qk_mode=qk_mode,
        is_causal=is_causal,
        sm_scale=runner.scale,
        lse=lse,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
        scale_o=1.0,
    )
    return output, lse.transpose(1, 2).contiguous() * _LOG2_TO_LN


def run_nvfp4_sm12x(
    runner: FlashInferPrefillRunner,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "FlashInfer SM120/SM121 NVFP4 attention supports dense self-attention only."
        )
    if runner.num_heads != runner.num_kv_heads or runner.head_dim not in (64, 128):
        raise ValueError(
            "FlashInfer SM120/SM121 NVFP4 attention requires equal Q/K/V head counts "
            "and head_dim=64 or 128."
        )
    sequence_length = q.shape[1]
    # FlashInfer 0.6.16 pads internally but cannot mask the padded K/V tokens.
    if sequence_length % 128 != 0:
        raise ValueError(
            "FlashInfer SM120/SM121 NVFP4 attention requires sequence length "
            f"to be a multiple of 128, got {sequence_length}."
        )
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    quantized_qkv = flashinfer.nvfp4_attention_sm120_quantize_qkv(
        query,
        key,
        value,
        per_block_mean=False,
    )
    output, lse = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized_qkv,
        sm_scale=runner.scale,
        causal=is_causal,
        per_block_mean=False,
        out_dtype=q.dtype,
    )
    output = output[:, :, :sequence_length].transpose(1, 2).contiguous()
    return output, lse[:, :, :sequence_length]
