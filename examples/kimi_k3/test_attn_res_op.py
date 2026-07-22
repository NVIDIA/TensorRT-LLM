# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity test: in-tree fused attn_res thop op vs the pure-torch references.

Covers both consumers of the kernel contract:
  1. the ``KimiK3AttnResidual`` module reference (chunked, kernel layout), and
  2. the model's ``_apply_attn_res`` fp32 reference (HF layout), which is what
     GSM8K runs use today.
No external kernel collection is required.
"""
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_kimi_linear import (
    KimiK3RMSNorm, _apply_attn_res, _apply_attn_res_fused)

torch.manual_seed(0)
dev = "cuda"
H = 7168
EPS = 1e-6


def rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    print(f"{name}: cos={cos:.6f} rel_l2={rel:.3e} max_abs={(a - b).abs().max().item():.3e}")
    return cos, rel


proj = nn.Linear(H, 1, bias=False, dtype=torch.bfloat16, device=dev)
norm = KimiK3RMSNorm(H, eps=EPS).to(device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    proj.weight.mul_(0.02)

ok = True
for M, K in [(64, 0), (128, 3), (1024, 11), (300, 5), (16384, 11)]:
    prefix = torch.randn(M, H, dtype=torch.bfloat16, device=dev) * 0.05
    block = torch.randn(M, K, H, dtype=torch.bfloat16, device=dev) * 0.05

    ref = _apply_attn_res(prefix, block, proj, norm)  # fp32 reference path
    fused = _apply_attn_res_fused(prefix, block, proj, norm)
    if fused is None:
        print(f"M={M} K={K}: fused path unavailable")
        ok = False
        continue
    c, r = rep(f"M={M} K={K}", fused, ref)
    ok &= c > 0.999 and r < 3e-2

print("ATTN-RES-OP-PARITY:", "PASS" if ok else "FAIL")
