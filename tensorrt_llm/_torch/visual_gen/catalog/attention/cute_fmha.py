# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._cute_fmha_kernel import (
    _check_cute_runtime_available,
    _quantize_blockscaled_one,
    _quantize_fp8,
    cute_dsl_fmha_fwd,
    cutlass,
    is_cute_fmha_supported,
)

__all__ = ["cute_fmha", "is_cute_fmha_supported"]


def _smooth_qk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_max = q.abs().amax(dim=(0, 1), keepdim=True).float().clamp_min(1e-4)
    k_max = k.abs().amax(dim=(0, 1), keepdim=True).float().clamp_min(1e-4)
    scale = (q_max.pow(0.7) / k_max.pow(1 - 0.7)).clamp(1e-4, 1e4)
    return q * scale.reciprocal().bfloat16(), (
        k - k.mean(dim=(0, 1), keepdim=True)
    ) * scale.bfloat16()


class CuteFMHA(OpWrapper):
    """Full self-attention on equal, unpadded BSHD queries, keys and values.

    Returns BSHD output in the query dtype and FP32 base-2 logsumexp in BHS order.
    Block-scaled Q/K use centered keys and report logsumexp for the centered scores.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("BF16 self-attention", dict(qk_dtype=None)),
        Cell(
            "skip a low-score KV tile and retain a near-threshold tile",
            dict(qk_dtype=None, skip_softmax_threshold_scale_factor=4.0),
        ),
        Cell("BF16 Q/K and E4M3 V", dict(qk_dtype="bf16")),
        Cell("Dynamic E4M3 Q/K/V", dict(qk_dtype="fp8")),
        Cell("MXFP8 Q/K and E4M3 V", dict(qk_dtype="mxfp8")),
        Cell("NVFP4 Q/K with per-channel E4M3 V scales", dict(qk_dtype="nvfp4", v_block_size=1)),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        qk_dtype: str | None = None,
        v_block_size: int | None = None,
        skip_softmax_threshold_scale_factor: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_cute_runtime_available()
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
        if qk_dtype in ("mxfp8", "nvfp4"):
            q, k = _smooth_qk(q, k)

        # V is tensor-scaled by default. MXFP8/NVFP4 with v_block_size=1 use an (H, D) scale.
        v_dequant_scale = 1.0
        q_sf = k_sf = qk_cutlass_dtype = None
        qk_sf_vec = 0
        scale_v_channels = None
        if qk_dtype is not None:
            if qk_dtype == "fp8":
                q, q_dequant_scale, _ = _quantize_fp8(q)
                k, k_dequant_scale, _ = _quantize_fp8(k)
                sm_scale = sm_scale * q_dequant_scale * k_dequant_scale
            elif qk_dtype in ("mxfp8", "nvfp4"):
                qk_sf_vec = 32 if qk_dtype == "mxfp8" else 16
                q, q_sf, gs_q = _quantize_blockscaled_one(q, qk_sf_vec)
                k, k_sf, gs_k = _quantize_blockscaled_one(k, qk_sf_vec)
                sm_scale = sm_scale * gs_q * gs_k
                qk_cutlass_dtype = cutlass.Float4E2M1FN if qk_sf_vec == 16 else cutlass.Float8E4M3FN
            v, v_dequant_scale, scale_v_channels = _quantize_fp8(
                v, per_head_channel=qk_sf_vec != 0 and v_block_size == 1
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
        return out, lse.transpose(1, 2)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        qk_dtype: str | None = None,
        v_block_size: int | None = None,
        skip_softmax_threshold_scale_factor: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense attention before lossy quantization or softmax skipping."""
        dtype = q.dtype
        q, k, v = q.double(), k.double(), v.double()
        if qk_dtype in ("mxfp8", "nvfp4"):
            k = k - k.mean(dim=(0, 1), keepdim=True)
        scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * sm_scale
        out = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)
        return out.to(dtype), (scores.logsumexp(-1) / math.log(2)).float()

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        qk_dtype: str | None = None,
        v_block_size: int | None = None,
        skip_softmax_threshold_scale_factor: float | None = None,
    ) -> None:
        assert q.ndim == 4 and q.shape == k.shape == v.shape
        assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.bfloat16, torch.float16)
        assert q.device == k.device == v.device
        assert qk_dtype in (None, "bf16", "fp8", "mxfp8", "nvfp4")


cute_fmha = CuteFMHA()
