# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer prefill plans, workspaces and provider kernel entry points."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any, Literal

import flashinfer
import torch

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

_WORKSPACE_BYTES = 128 * 1024 * 1024
_LOG2_TO_LN = math.log(2.0)
_BlockScaledQKMode = Literal["mxfp8", "nvfp4"]


class FlashInferPrefillRunner:
    """Own the prefill head geometry and the plans and workspaces shared by the caller."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        static_global_scale: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.quant_attention_config = quant_attention_config
        self._metadata_state = (
            attention_metadata_state if attention_metadata_state is not None else {}
        )
        self.static_global_scale = static_global_scale
        if quant_attention_config is not None and quant_attention_config.qk_dtype not in (
            "mxfp8",
            "nvfp4",
        ):
            raise ValueError(
                "FlashInfer quantized attention supports qk_dtype='mxfp8' or 'nvfp4', "
                f"got {quant_attention_config.qk_dtype!r}."
            )

    def plan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        if self.static_global_scale:
            return self._capture_safe_plan(
                q, k, is_causal=is_causal, key_padding_mask=key_padding_mask
            )
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        stream = torch.cuda.current_stream(q.device)
        stream_key = (q.device, stream.cuda_stream)

        workspace_cache = self._metadata_state.setdefault("flashinfer_workspace_cache", {})
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
        plan_cache = self._metadata_state.setdefault("flashinfer_prefill_plan_cache", {})
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
        if key_padding_mask is not None:
            mask_signature = (
                id(key_padding_mask),
                key_padding_mask.data_ptr(),
                key_padding_mask._version,
            )
        if not plan_entry["is_planned"] or plan_entry["mask_signature"] != mask_signature:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "Cannot plan FlashInfer batched prefill while the CUDA stream is "
                    "capturing. Run a warmup forward pass before capture."
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

    def _prefill_mask_digest(self, mask: torch.Tensor | None, *, capturing: bool) -> str | None:
        if mask is None:
            return None
        version = None if torch.is_inference(mask) else mask._version
        identity = (id(mask), mask.data_ptr(), version)
        identities = self._metadata_state.setdefault("flashinfer_mask_identities", {})
        if capturing:
            if identity not in identities:
                raise RuntimeError("FlashInfer padding mask must be warmed before CUDA capture.")
            return identities[identity][1]
        digest = hashlib.sha256(mask.detach().to("cpu", torch.bool).numpy().tobytes()).hexdigest()
        # Retain the tensor so its identity cannot be recycled while a plan uses it.
        identities[identity] = (mask, digest)
        return digest

    def _capture_safe_plan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        """Reuse immutable mask-content plans across warmup and capture streams.

        Serialized forwards share a plan. Each new mask digest owns its wrapper,
        so a captured graph retains the plan used during capture.
        """
        capturing = torch.cuda.is_current_stream_capturing()
        mask_digest = self._prefill_mask_digest(key_padding_mask, capturing=capturing)
        batch_size, query_length = q.shape[:2]
        key_value_length = k.shape[1]
        plan_key = (
            q.device,
            batch_size,
            query_length,
            key_value_length,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q.dtype,
            is_causal,
            mask_digest,
        )
        plans = self._metadata_state.setdefault("flashinfer_immutable_prefill_plans", {})
        entry = plans.get(plan_key)
        if entry is None:
            if capturing:
                raise RuntimeError("FlashInfer batched prefill must be warmed before CUDA capture.")
            workspaces = self._metadata_state.setdefault("flashinfer_immutable_workspaces", {})
            workspace = workspaces.get(q.device)
            if workspace is None:
                workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
                workspaces[q.device] = workspace
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
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
                    .clone()
                )
            torch.cuda.current_stream(q.device).synchronize()
            wrapper.plan(
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
            entry = (wrapper, query_indptr, key_value_indptr, custom_mask)
            plans[plan_key] = entry
        return entry[0]


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


def _quantize_qk_static_global(
    tensor: torch.Tensor, mode: _BlockScaledQKMode
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode NHD Q/K with global one and native, input-dependent block scales."""
    from flashinfer.quantization import fp4_quantize, mxfp8_quantize

    batch, sequence, heads, dimension = tensor.shape
    padded_sequence = ((sequence + 127) // 128) * 128
    data = tensor.transpose(1, 2).contiguous()
    if padded_sequence != sequence:
        data = torch.nn.functional.pad(data, (0, 0, 0, padded_sequence - sequence))
    matrix = data.reshape(batch * heads * padded_sequence, dimension)
    if mode == "mxfp8":
        encoded, factors = mxfp8_quantize(
            matrix, is_sf_swizzled_layout=True, alignment=32, backend="cute-dsl"
        )
        encoded = encoded.view(batch, heads, padded_sequence, dimension)[:, :, :sequence]
        factors = factors.view(torch.float8_e8m0fnu)
    elif mode == "nvfp4":
        # Encoding and FMHA share a unit global scale and per-block native factors.
        inverse_global = torch.ones(1, device=tensor.device, dtype=torch.float32)
        encoded, factors = fp4_quantize(
            matrix,
            inverse_global,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
            backend="cute-dsl",
        )
        encoded = encoded.view(batch, heads, padded_sequence, dimension // 2)[:, :, :sequence]
        factors = factors.view(torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported static-global Q/K mode: {mode!r}.")
    encoded = encoded.transpose(1, 2).contiguous()
    if mode == "nvfp4":
        # Copy packed bytes before viewing Float4, whose deterministic fill is unsupported.
        encoded = encoded.view(torch.float4_e2m1fn_x2)
    return encoded, factors


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

    if runner.static_global_scale:
        q_store, q_sf = _quantize_qk_static_global(q, qk_mode)
        k_store, k_sf = _quantize_qk_static_global(k, qk_mode)
        v_store = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
        q_scale = k_scale = v_scale = 1.0
    else:
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
