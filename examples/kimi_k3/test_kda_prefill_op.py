# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity test: in-tree KDA prefill CuTe DSL op (optimized path) vs FLA reference.

The in-tree ``trtllm::kda_prefill`` op needs no external kernel collection.
"""

import torch

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

torch.manual_seed(0)
dev = "cuda"
H, D, W = 96, 128, 4

common = dict(
    hidden_size=7168,
    num_heads=H,
    head_dim=D,
    conv_kernel_size=W,
    use_full_rank_gate=True,
    gate_lower_bound=-5.0,
    rms_norm_eps=1e-5,
    dtype=torch.bfloat16,
)

opt = KimiKDALinearAttention(**common).to(dev)
assert opt.prefill_kernel_path == "optimized", opt.prefill_kernel_path
ref = KimiKDALinearAttention(**common, use_optimized_prefill=False).to(dev)
ref.load_state_dict(opt.state_dict())
assert ref.prefill_kernel_path == "fla", ref.prefill_kernel_path
assert ref.decode_kernel_path == opt.decode_kernel_path


def rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    print(f"{name}: cos={cos:.6f} rel_l2={rel:.3e} max_abs={(a - b).abs().max().item():.3e}")
    return cos, rel


ok = True

with torch.no_grad():
    # Equal-length prefill: B=2 sequences, T multiple of 64.
    for B, T in [(2, 256), (1, 1024)]:
        x = torch.randn(B, T, 7168, dtype=torch.bfloat16, device=dev) * 0.05
        out_opt = opt.forward_prefill(x)
        out_ref = ref.forward_prefill(x)
        c, r = rep(f"eqlen B={B} T={T}", out_opt, out_ref)
        ok &= c > 0.999 and r < 3e-2

    # Equal-length with T not a multiple of 64 (B=1 padding path).
    x = torch.randn(1, 300, 7168, dtype=torch.bfloat16, device=dev) * 0.05
    out_opt = opt.forward_prefill(x)
    out_ref = ref.forward_prefill(x)
    c, r = rep("eqlen B=1 T=300 (pad path)", out_opt, out_ref)
    ok &= c > 0.999 and r < 3e-2

    # Varlen: packed B=1 with cu_seqlens (64-aligned lengths).
    lens = [128, 256, 192]
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()), dtype=torch.long, device=dev
    )
    x = torch.randn(1, sum(lens), 7168, dtype=torch.bfloat16, device=dev) * 0.05
    out_opt = opt.forward_prefill(x, cu_seqlens=cu)
    out_ref = ref.forward_prefill(x, cu_seqlens=cu)
    c, r = rep(f"varlen lens={lens}", out_opt, out_ref)
    ok &= c > 0.999 and r < 3e-2

print("prefill source (optimized):", opt._dispatch.get_prefill_source())
print("KDA-PREFILL-OP-PARITY:", "PASS" if ok else "FAIL")
