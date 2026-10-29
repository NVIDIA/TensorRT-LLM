# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MK1 P0a switches change only where kernels trigger and what they load before
griddepcontrol.wait, so every output must be bitwise identical:

* ``TRTLLM_K3_ATTN_RES_EARLY_TRIGGER`` 0 / 1 / 2 (k3_attn_res, exact and fast kernels, with
  and without the FP8 output);
* ``TRTLLM_K3_GEMM_PREWAIT_STAGES`` (with the reserve and grid caps) on the PDL Rubin MXFP8 GEMM,
  base and split-K tactics, both operand orders;
* the Triton twin of the C++ 1x128 FP8 quantizer (``TRTLLM_K3_QUANT_EARLY_TRIGGER``): the same
  codes and R128c4 scale bytes, padding rows included, for every trigger position.
"""

import pytest
import torch
from test_kimi_k3_mxfp8_split_k_rubin import _k3_inputs, _rubin_available, _tactics

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _rubin_available(),
    reason="needs SM107 with the CuTe DSL Rubin helpers",
)


@pytest.fixture(autouse=True)
def _mk1_supported(monkeypatch):
    from tensorrt_llm._torch.modules import kimi_k3_mk1_gate

    monkeypatch.setattr(kimi_k3_mk1_gate, "mk1_platform_supported", lambda: True)


@pytest.mark.parametrize("trigger", [0, 1, 2])
# Row tiles of 16 and 128-row scale tiles: one partial tile, exact tiles, a tile plus one row,
# a full scale tile, and padding past several row tiles; K with a partial column group.
@pytest.mark.parametrize(
    "m,k",
    [
        (1, 7168),
        (6, 7168),
        (12, 7168),
        (16, 7168),
        (17, 1536),
        (48, 2048),
        (128, 7168),
        (130, 1536),
    ],
)
def test_fp8_quant_twin_matches_the_cpp_quantizer(m: int, k: int, trigger: int):
    from tensorrt_llm._torch.modules.kimi_k3_fp8_quant import fp8_quantize_1x128_r128c4

    g = torch.Generator(device="cuda").manual_seed(m * k + trigger)
    x = torch.randn((m, k), device="cuda", generator=g)
    # Per-row magnitudes from E4M3 subnormal territory to large values, one all-zero block.
    x = x * torch.logspace(-30, 30, m, base=2.0, device="cuda")[:, None]
    x[0, :128] = 0
    x = x.to(torch.bfloat16)
    ref_q, ref_s = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
    q, s = fp8_quantize_1x128_r128c4(x, trigger=trigger)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert s.numel() == ref_s.numel()
    assert torch.equal(s, ref_s.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("m,n", [(6, 5), (12, 9), (48, 3)])
def test_attn_res_trigger_modes_are_bitwise(m: int, n: int, mode: int, exact: bool, fp8: bool):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    g = torch.Generator(device="cuda").manual_seed(m + n)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    block, layer, add, add2 = (
        rn(n - 1, m, 7168),
        rn(m, 7168),
        rn(m, 7168, sc=0.5),
        rn(m, 7168, sc=0.5),
    )
    proj, norm_w = rn(7168, sc=0.05), (1.0 + rn(7168, sc=0.1).float()).to(torch.bfloat16)
    out_w = (1.0 + rn(7168, sc=0.1).float()).to(torch.bfloat16)
    addends = ((), (add,), (add, add2))[mode]
    outs = []
    for trigger in (0, 1, 2):
        out = torch.empty_like(layer)
        upd = torch.empty_like(layer) if addends else None
        # The FP8 output (TRTLLM_K3_ATTN_RES_FP8_OUT): codes and R128c4 scales, padding included.
        q = torch.empty((m, 7168), dtype=torch.float8_e4m3fn, device="cuda") if fp8 else None
        s = torch.full((k3_attn_res.fp8_scale_numel(m),), 0xA5, dtype=torch.uint8, device="cuda")
        k3_attn_res.run_k3_attn_res(
            layer, addends, block, proj, norm_w, out_w, 1e-6, 1e-5, out, upd,
            exact=exact, trigger=trigger, fp8_output=q, fp8_scale=s if fp8 else None,
        )  # fmt: skip
        outs.append((out, upd, q, s))
    for out, upd, q, s in outs[1:]:
        assert torch.equal(out, outs[0][0])
        if addends:
            assert torch.equal(upd, outs[0][1])
        if fp8:
            assert torch.equal(q.view(torch.uint8), outs[0][2].view(torch.uint8))
            assert torch.equal(s, outs[0][3])


def _staged_kernel_compiled(runner, tactic, stages: int) -> bool:
    """Whether the runner's kernel cache holds a staged kernel of this tactic's MMA tiler."""
    return any(
        isinstance(key, tuple)
        and "prewait" in key
        and key[key.index("prewait") + 1] == stages
        and tuple(tactic[1]) in key
        for key in runner.__class__.kernel_cache
    )


@pytest.mark.parametrize("n,k,m", [(6144, 7168, 12), (3584, 7168, 6), (2112, 7168, 48)])
def test_gemm_prewait_stages_are_bitwise(monkeypatch, n: int, k: int, m: int):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLKimiK3MXFP8RubinLinear

    inputs = _k3_inputs(n, k, m)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL", "1")
    monkeypatch.setenv("TRTLLM_CUTEDSL_RUBIN_GEMM_PDL_MAX_CTAS", "100000")
    monkeypatch.setenv("TRTLLM_K3_MXFP8_SPLIT_K", "1")
    runner = CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    tactics = _tactics(runner, inputs)
    # Staging needs the static scheduler. Base tactics are ("base", tiler, inst, cluster,
    # swap_ab, prefetch, scheduler, raster): one per (tile M, MMA M, cluster, swap AB), so 2-CTA
    # MMAs, clusters and both operand orders run staged; plus the split-K tactic with most splits.
    checked, seen = [], set()
    for t in tactics:
        if t[0] == "base" and t[6] == "static":
            signature = (t[1][0], t[2][0], tuple(t[3]), t[4])
            if signature not in seen:
                seen.add(signature)
                checked.append(t)
    checked += sorted((t for t in tactics if t[0] == "splitk"), key=lambda t: t[-1])[-1:]
    assert checked
    switches = (
        "TRTLLM_K3_GEMM_PREWAIT_STAGES",
        "TRTLLM_K3_GEMM_PREWAIT_RESERVE_KB",
        "TRTLLM_K3_GEMM_PREWAIT_MAX_CTAS",
    )
    # A 16-CTA cap makes CTAs loop over several tiles (staging only the first one).
    configs = ((2, 0, 0), (4, 8, 0), (16, 0, 0), (4, 8, 16), (0, 0, 16))
    for tactic in checked:
        for name in switches:
            monkeypatch.setenv(name, "0")
        ref = runner(inputs, tactic=tactic)
        for config in configs:
            for name, value in zip(switches, config):
                monkeypatch.setenv(name, str(value))
            out = runner(inputs, tactic=tactic)
            assert torch.equal(out, ref), (tactic, config)
            if config[0]:
                assert _staged_kernel_compiled(runner, tactic, config[0]), (tactic, config)
