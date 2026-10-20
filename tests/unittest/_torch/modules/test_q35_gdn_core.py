# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 GDN target-verify core as a CUDA kernel (modules/fla/q35_gdn_core.py, TRTLLM_Q35_GDN_CORE) against the
Triton fused replay + gated RMSNorm (modules/fla/q35_gdn_fused.py, conv=False), per-rank TP8 / TP16 shapes.

The CUDA kernel uses the exact bf16 q / k rows in its MMAs and applies the l2norm scales to the results, where the
Triton kernel rounds the normalized rows to bf16 first, so the comparison is a tolerance check (the fp64-reference
comparison in a microbenchmark shows the CUDA kernel equal or closer to the exact result). Runs without pytest too:
``python3 test_q35_gdn_core.py``.
"""

import itertools

import torch

try:
    import pytest
except ImportError:  # images without pytest: run __main__
    pytest = None

from tensorrt_llm._torch.modules.fla import q35_gdn_core as gc
from tensorrt_llm._torch.modules.fla import q35_gdn_fused as gf

K = V = 128
HIST = 16
SLOTS = 16


def _supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


if pytest is not None:
    pytestmark = pytest.mark.skipif(not _supported(), reason="needs SM100/SM103")


def _case(tp16, n, t, pnat, seed):
    H, HV = (1, 4) if tp16 else (2, 8)
    g = torch.Generator(device="cuda").manual_seed(seed)
    cd = 2 * H * K + HV * V
    proj = (torch.randn(n * t, cd + HV * V, generator=g, device="cuda") * 0.7).to(torch.bfloat16)
    ba = torch.randn(n * t, 2 * HV, generator=g, device="cuda").to(torch.bfloat16)
    st = dict(
        a_log=torch.log(torch.rand(HV, generator=g, device="cuda") * 7 + 1).float(),
        dt_bias=(torch.rand(HV, generator=g, device="cuda") * 2 - 1).float(),
        norm_w=(torch.rand(V, generator=g, device="cuda") + 0.5).to(torch.bfloat16),
        fp8_scale=torch.tensor(0.02, dtype=torch.float32, device="cuda"),
        state_indices=torch.tensor([1, 4, 7, 10, 13, 2, 5, 8][:n], dtype=torch.int32, device="cuda"),
        pnat=torch.full((SLOTS,), pnat, dtype=torch.int32, device="cuda"),
        buf=torch.zeros(SLOTS, dtype=torch.int32, device="cuda"),
    )
    st["buf"][2::3] = 1
    kh = torch.randn(SLOTS, 2, HIST, H, K, generator=g, device="cuda")
    dec = torch.rand(SLOTS, 2, HV, HIST, generator=g, device="cuda") * 0.49 + 0.01
    mut = dict(
        ssm=(torch.randn(SLOTS, HV, V, K, generator=g, device="cuda") * 0.05).to(torch.bfloat16),
        old_k=(kh / kh.norm(dim=-1, keepdim=True)).to(torch.bfloat16),
        old_u=(torch.randn(SLOTS, 2, HIST, HV, V, generator=g, device="cuda") * 0.5).to(torch.bfloat16),
        old_g=(-torch.cumsum(dec, dim=-1)).float().contiguous(),
    )
    return H, HV, cd, proj, ba, st, mut


def _run_triton(H, HV, cd, n, t, proj, ba, st, mut):
    m = {k: v.clone() for k, v in mut.items()}
    o = torch.empty(n, t, HV, V, dtype=torch.bfloat16, device="cuda")
    conv_w = torch.zeros(cd, 4, dtype=torch.bfloat16, device="cuda")
    conv_s = torch.zeros(SLOTS, cd, 3, dtype=torch.bfloat16, device="cuda")
    inter = torch.zeros(n, t, cd, 3, dtype=torch.bfloat16, device="cuda")
    y = gf.q35_gdn_fused_update(
        proj, conv_w, conv_s, inter, ba[:, HV:], ba[:, :HV], st["a_log"], st["dt_bias"], m["ssm"], st["state_indices"],
        m["old_u"], m["old_k"], m["old_g"], st["buf"], st["pnat"], HIST, n, t, H, HV, K, V, output=o,
        z=proj[:, cd:].view(n * t, HV, V), norm_weight=st["norm_w"], norm_eps=1e-6, fp8_scale=st["fp8_scale"],
        counters=torch.zeros(64, dtype=torch.int32, device="cuda"), block_v=16, num_warps=4, prefetch_state=True,
        trigger=2, spec_hist=True, solve=2, conv=False, store_o=True)
    return o, y, m


def _run_core(H, HV, cd, n, t, proj, ba, st, mut):
    m = {k: v.clone() for k, v in mut.items()}
    o = torch.empty(n, t, HV, V, dtype=torch.bfloat16, device="cuda")
    y = gc.q35_gdn_core_update(
        proj, ba[:, HV:], ba[:, :HV], st["a_log"], st["dt_bias"], m["ssm"], st["state_indices"], m["old_u"], m["old_k"],
        m["old_g"], st["buf"], st["pnat"], HIST, n, t, H, HV, K, V, z=proj[:, cd:].view(n * t, HV, V),
        norm_weight=st["norm_w"], norm_eps=1e-6, fp8_scale=st["fp8_scale"], output=o, trigger=2)
    return o, y, m


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def check(tp16, n, pnat, t=7, seed=0):
    H, HV, cd, proj, ba, st, mut = _case(tp16, n, t, pnat, 1000 + seed)
    o_ref, y_ref, m_ref = _run_triton(H, HV, cd, n, t, proj, ba, st, mut)
    o, y, m = _run_core(H, HV, cd, n, t, proj, ba, st, mut)
    torch.cuda.synchronize()
    # replay output: bf16-level agreement
    assert _rel(o, o_ref) < 1e-2, f"o rel {_rel(o, o_ref)}"
    # FP8 norm output: identical up to FP8 rounding flips of near-tie values
    yf, yrf = y.float(), y_ref.float()
    assert _rel(yf, yrf) < 5e-2, f"y rel {_rel(yf, yrf)}"
    assert (y.view(torch.uint8) != y_ref.view(torch.uint8)).float().mean().item() < 0.2
    # history of this step and the (possibly committed) state
    assert _rel(m["old_u"], m_ref["old_u"]) < 1e-3, f"hist u {_rel(m['old_u'], m_ref['old_u'])}"
    assert _rel(m["old_k"], m_ref["old_k"]) < 1e-2, f"hist k {_rel(m['old_k'], m_ref['old_k'])}"
    assert torch.allclose(m["old_g"], m_ref["old_g"], rtol=1e-5, atol=1e-6), "hist G"
    assert _rel(m["ssm"], m_ref["ssm"]) < 1e-3, f"ssm {_rel(m['ssm'], m_ref['ssm'])}"
    if pnat + t > HIST:
        assert not torch.equal(m["ssm"], mut["ssm"]), "commit expected (history full)"
    # negative controls: (1) one value-input column changes the replay output; (2) one checkpoint-state element of a
    # used slot changes the output or the committed state (its weight in o is e^{G_t}, which the random gates can push
    # below bf16 resolution)
    proj2 = proj.clone()
    proj2[:, 2 * H * K + 3] += 1.0
    o2, _, _ = _run_core(H, HV, cd, n, t, proj2, ba, st, mut)
    mut2 = {k: v.clone() for k, v in mut.items()}
    mut2["ssm"][int(st["state_indices"][0]), 0, 3, 5] += 4.0
    o3, _, m3 = _run_core(H, HV, cd, n, t, proj, ba, st, mut2)
    torch.cuda.synchronize()
    assert not torch.equal(o2, o), "negative control (value input) not detected"
    assert not torch.equal(o3, o) or not torch.equal(m3["ssm"], m["ssm"]), "negative control (state) not detected"


if pytest is not None:

    @pytest.mark.parametrize("tp16", [False, True])
    @pytest.mark.parametrize("n", [1, 2, 4])
    @pytest.mark.parametrize("pnat", [0, 5, 12])
    def test_q35_gdn_core_vs_triton(tp16, n, pnat):
        check(tp16, n, pnat)

    @pytest.mark.parametrize("tp16", [False, True])
    def test_q35_gdn_core_t8(tp16):
        check(tp16, 2, 9, t=8)


if __name__ == "__main__":
    for tp16, n, pnat in itertools.product((False, True), (1, 2, 4), (0, 5, 12)):
        check(tp16, n, pnat)
        print(f"[test] tp16={tp16} n={n} pnat={pnat}: PASS", flush=True)
    for tp16 in (False, True):
        check(tp16, 2, 9, t=8)
        print(f"[test] tp16={tp16} n=2 pnat=9 T=8: PASS", flush=True)
    print("[test] ALL PASS", flush=True)
