# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local choice of equivalent static-FP8 GEMM entry points."""

import json
from pathlib import Path

import cache_env  # noqa: F401 -- configure caches before importing CUDA libraries
import torch

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod

STATE = {"mode": 0, "validate": False, "fast_accum": True}
checks = {}


@torch.library.custom_op("h3_fp8_round2::gemm", mutates_args=())
def gemm(a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    n = w.shape[0]
    bit = 1 if n == 21504 else 2

    def native():
        return torch.ops.trtllm.cublas_scaled_mm(
            a, w.t(), scale_a=sa, scale_b=sb, bias=None, out_dtype=torch.bfloat16
        )

    def candidate():
        return torch._scaled_mm(
            a,
            w.t(),
            scale_a=sa,
            scale_b=sb,
            out_dtype=torch.bfloat16,
            use_fast_accum=STATE["fast_accum"],
        )

    key = str(w.data_ptr())
    if STATE["validate"] and key not in checks:
        ref = native()
        val = candidate()
        diff = val.float() - ref.float()
        checks[key] = dict(
            shape=[a.shape[0], n, a.shape[1]],
            scale_a=float(sa.item()),
            scale_b=float(sb.item()),
            byte_equal=bool(torch.equal(ref, val)),
            mismatch_fraction=float((val != ref).float().mean().item()),
            relative_l2=float(
                (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref.float())).item()
            ),
        )
        Path("/profile/results/real-gemm-checks.json").write_text(
            json.dumps(list(checks.values()), indent=2)
        )
    return candidate() if STATE["mode"] & bit else native()


@gemm.register_fake
def _(a, w, sa, sb):
    return torch.empty((a.shape[0], w.shape[0]), device=a.device, dtype=torch.bfloat16)


original = FP8QDQLinearMethod.apply


def apply(self, module, x, bias):
    scale = self.get_static_input_scale(module)
    if (
        module.weight.shape[0] not in (21504, 28672)
        or module.weight.shape[1] != 5376
        or module.dtype != torch.bfloat16
        or scale is None
        or getattr(module, "all_reduce", None) is not None
    ):
        return original(self, module, x, bias)
    shape = x.shape
    y = x.reshape(-1, shape[-1])
    if y.dtype != torch.float8_e4m3fn:
        y = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(y, scale)[0]
    out = gemm(y, module.weight, module.input_scale, module.weight_scale)
    if len(shape) > 2:
        out = out.reshape(*shape[:-1], out.shape[-1])
    return out + bias if bias is not None else out


def install():
    FP8QDQLinearMethod.apply = apply
