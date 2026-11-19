# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.fmha_interface import (
    _check_cute_runtime_available,
    _quantize_blockscaled_one,
    _quantize_fp8,
    cute_dsl_fmha_fwd,
    cutlass,
    is_cute_fmha_supported,
)

__all__ = ["cute_fmha", "is_cute_fmha_supported"]


def _smooth_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    alpha: float = 0.7,
):
    """Balance per-channel Q/K ranges across the query heads sharing each KV head."""
    k = k.unflatten(2, (-1, 1))  # (B, S, H_K, 1, D)
    q = q.unflatten(2, (k.shape[2], -1))  # (B, S, H_K, H_R, D)
    q_max = q.abs().amax(dim=(0, 1, 3), keepdim=True).float().clamp_min(1e-4)  # (1, 1, H_K, 1, D)
    k_max = k.abs().amax(dim=(0, 1, 3), keepdim=True).float().clamp_min(1e-4)  # (1, 1, H_K, 1, D)
    s = (q_max.pow(alpha) / k_max.pow(1 - alpha)).clamp(1e-4, 1e4)
    q = q * s.reciprocal().bfloat16()
    # Per-channel shift commutes with the per-channel smooth scale `s`.
    k = k - k.mean(dim=(0, 1, 3), keepdim=True)
    k = k * s.bfloat16()
    return q.flatten(2, 3), k.flatten(2, 3)


class CuteFMHA(OpWrapper):
    """CuTe dense attention on BSHD tensors with FP32 base-2 logsumexp in BHS order.

    A Q/K recipe selects FP8 V. Block-scaled Q/K use a centered K and report
    the logsumexp of the centered scores. Causal queries attend to keys at
    the same or earlier position.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("BF16 causal GQA with unequal sequence lengths", dict(qk_dtype=None, is_causal=True)),
        Cell(
            "E4M3 Q/K/V with independent dequantization scales",
            dict(qk_dtype="fp8", prequantized=True),
        ),
        Cell("NVFP4 Q/K with per-channel E4M3 V scales", dict(qk_dtype="nvfp4", v_block_size=1)),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        is_causal: bool = False,
        qk_dtype: str | None = None,
        v_block_size: int | None = None,
        static_q_scale: torch.Tensor | None = None,
        static_k_scale: torch.Tensor | None = None,
        static_v_scale: torch.Tensor | None = None,
        skip_softmax_threshold_scale_factor: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_cute_runtime_available()
        fp8 = torch.float8_e4m3fn
        if (q.dtype == fp8 or k.dtype == fp8) and qk_dtype != "fp8":
            raise ValueError("Prequantized E4M3 Q/K require an FP8 Q/K attention recipe.")
        origin_dtype = torch.bfloat16 if q.dtype == fp8 else q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16, fp8):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        if qk_dtype in ("mxfp8", "nvfp4"):
            q, k = _smooth_qk(q, k)

        batch_size, seq_len_q, num_heads, _ = q.shape
        value_head_dim = v.shape[-1]
        output_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        out = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            value_head_dim,
            dtype=output_dtype,
            device=q.device,
        )
        lse = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            dtype=torch.float32,
            device=q.device,
        )

        # V is tensor-scaled by default. MXFP8/NVFP4 with v_block_size=1 use an (H, D) scale.
        v_dequant_scale = 1.0
        q_sf = k_sf = qk_cutlass_dtype = None
        qk_sf_vec = 0
        scale_v_channels = None
        if qk_dtype is not None:
            if qk_dtype == "fp8":
                if q.dtype == torch.float8_e4m3fn:
                    if static_q_scale is None:
                        raise ValueError("Prequantized E4M3 Q requires a Q dequant scale.")
                    q_dequant_scale = static_q_scale
                else:
                    q, q_dequant_scale, _ = _quantize_fp8(q, static_q_scale)
                if k.dtype == torch.float8_e4m3fn:
                    if static_k_scale is None:
                        raise ValueError("Prequantized E4M3 K requires a K dequant scale.")
                    k_dequant_scale = static_k_scale
                else:
                    k, k_dequant_scale, _ = _quantize_fp8(k, static_k_scale)
                sm_scale = sm_scale * q_dequant_scale * k_dequant_scale
            elif qk_dtype in ("mxfp8", "nvfp4"):
                qk_sf_vec = 32 if qk_dtype == "mxfp8" else 16
                q, q_sf, gs_q = _quantize_blockscaled_one(q, qk_sf_vec)
                k, k_sf, gs_k = _quantize_blockscaled_one(k, qk_sf_vec)
                sm_scale = sm_scale * gs_q * gs_k
                qk_cutlass_dtype = cutlass.Float4E2M1FN if qk_sf_vec == 16 else cutlass.Float8E4M3FN
            if v.dtype == torch.float8_e4m3fn:
                if static_v_scale is None:
                    raise ValueError("Prequantized E4M3 V requires a V dequant scale.")
                v_dequant_scale = static_v_scale
            else:
                v, v_dequant_scale, scale_v_channels = _quantize_fp8(
                    v,
                    dequant_scale=static_v_scale,
                    per_head_channel=qk_sf_vec != 0 and v_block_size == 1,
                )

        # Materialize packed-QKV views where Inductor can fuse the copies with quantization.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        cute_dsl_fmha_fwd(
            q,
            k,
            v,
            out,
            is_causal=is_causal,
            sm_scale=sm_scale,
            lse=lse,
            scale_output=v_dequant_scale,
            scale_v_channels=scale_v_channels,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            qk_sf_vec=qk_sf_vec,
            q_sf=q_sf,
            k_sf=k_sf,
            qk_cutlass_dtype=qk_cutlass_dtype,
        )
        return out.to(origin_dtype), lse.transpose(1, 2)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        is_causal: bool = False,
        qk_dtype: str | None = None,
        v_block_size: int | None = None,
        static_q_scale: torch.Tensor | None = None,
        static_k_scale: torch.Tensor | None = None,
        static_v_scale: torch.Tensor | None = None,
        skip_softmax_threshold_scale_factor: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense attention before lossy quantization or softmax skipping."""
        fp8 = torch.float8_e4m3fn
        origin_dtype = torch.bfloat16 if q.dtype == fp8 else q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16, fp8):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        def dequantize(tensor, scale):
            return tensor.double() * scale if tensor.dtype == fp8 else tensor.double()

        q = dequantize(q, static_q_scale)
        k = dequantize(k, static_k_scale)
        v = dequantize(v, static_v_scale)
        if qk_dtype in ("mxfp8", "nvfp4"):
            k = k - k.mean(dim=(0, 1), keepdim=True)
        group_size = q.size(2) // k.size(2)
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(group_size, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(group_size, dim=1)
        scores = (qh @ kh.transpose(-1, -2)) * sm_scale
        if is_causal:
            mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        out = (scores.softmax(-1) @ vh).transpose(1, 2)
        return out.to(origin_dtype), (scores.logsumexp(-1) / math.log(2)).float()

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        is_causal: bool = False,
        qk_dtype: str | None = None,
        v_block_size: int | None = None,
        static_q_scale: torch.Tensor | None = None,
        static_k_scale: torch.Tensor | None = None,
        static_v_scale: torch.Tensor | None = None,
        skip_softmax_threshold_scale_factor: float | None = None,
    ) -> None:
        for tensor, scale in ((q, static_q_scale), (k, static_k_scale), (v, static_v_scale)):
            if tensor.dtype == torch.float8_e4m3fn and scale is not None:
                assert scale.numel() == 1 and scale.dtype == torch.float32, (
                    "expected one FP32 dequant scale"
                )
                assert scale.device == tensor.device
        assert qk_dtype in (None, "bf16", "fp8", "mxfp8", "nvfp4")


cute_fmha = CuteFMHA()
