# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_moe catalog entry.

The entry owns the contract: `fused_moe.CELLS` is what is certified,
`reference` is what it is certified against, and `_compare` / `_compare_step5`
below are the bands. This file owns what a cell cannot carry -- building up to
6.6 GB of weights once per geometry and reusing them across that geometry's
cells, and the hazards, which are claims about inputs no cell describes.
"""

import contextlib
import hashlib
import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_moe import fused_moe
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "fused_moe requires a CUDA device"

DEV = "cuda"
# The reference GEMMs must be true fp32; TF32 would leave the reference with
# 10 mantissa bits, coarser than the bf16 output it is meant to bound. The
# entry's reference also enters `true_fp32_matmul` itself.
torch.backends.cuda.matmul.allow_tf32 = False


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Wider than torch's default band, with the derivation rather than a number.

    Kernel and reference consume bit-identical operands and differ only in
    GEMM accumulation order and in which side of a rounding boundary each
    intermediate lands -- one flipped intermediate moves an output element
    by about one ulp of that row's scale. torch's default cannot express
    that: its bf16 `atol=1e-5` sits three orders of magnitude below one
    output ulp of a two-GEMM chain, and per-element `rtol` is meaningless
    where cancellation drives `|ref|` to zero.

    So: 8 ulp of the row's largest magnitude element-wise, 4 ulp of
    relative RMS in aggregate, for the R1 cells. Against the entry's
    reference, which also models the op's bf16 store of the FC1 output, the
    worst case over the R1 tactic space walked in `test_r1_mtp_tactic_space`
    is 3.97 / 1.15 ulp -- 2.0x and 3.5x margin (3.98 / 1.43 ulp against a
    reference that leaves the FC1 store out). Both gates were sized against
    deliberately wrong computations, which land two orders of magnitude out.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    ulp = ULP[ref.dtype]
    row_scale = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(
        out.float() / row_scale, ref.float() / row_scale, rtol=0.0, atol=8 * ulp
    )
    rel_rms = (
        (out.float() - ref.float()).pow(2).mean().sqrt()
        / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    ).item()
    assert rel_rms <= 4 * ulp, f"relative RMS {rel_rms:.3e} > {4 * ulp:.3e}"


def _make(num_tokens, hidden, inter, num_experts, top_k, seed=0, dtype=torch.bfloat16):
    """A small, cheap operand set for the hazard cases below.

    Deliberately not a certified geometry: the claims it serves -- ids outside
    the window, repeated ids, rejected domains -- are about routing
    bookkeeping, which does not vary with shape. Driving them at the R1
    geometry would cost 5.6 GB of weights to prove something shape-independent.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, device=DEV, generator=g).to(dtype)
    w31 = (torch.randn(num_experts, 2 * inter, hidden, device=DEV, generator=g) / hidden**0.5).to(
        dtype
    )
    w2 = (torch.randn(num_experts, hidden, inter, device=DEV, generator=g) / inter**0.5).to(dtype)
    logits = torch.randn(num_tokens, num_experts, device=DEV, generator=g)
    vals, ids = torch.topk(logits, top_k, dim=-1)
    return x, ids.to(torch.int32), torch.softmax(vals, dim=-1), w31, w2


# ─── The certified cells ──────────────────────────────────────────────────

# The R1-0528 MTP layer's MoE: hidden 7168, top-8, and at the dep4 share 64
# experts at the full 2048-wide intermediate. The cells add the other shares.
R1_H, R1_I, R1_K = 7168, 2048, 8
R1_E_LOCAL, R1_EP = 64, 4
R1_E_GLOBAL = R1_E_LOCAL * R1_EP


def _make_r1_weights(num_experts: int, seed: int, inter: int = R1_I):
    """One rank's [E, 2I, H] / [E, H, I] bf16 weight pair, built in chunks.

    5.6 GB at E = 64, I = 2048 and at E = 256, I = 512. Materializing the fp32
    randn for the whole stack at once would transiently need another 11 GB;
    chunking caps the temporary at 64 MB.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    w31 = torch.empty(num_experts, 2 * inter, R1_H, dtype=torch.bfloat16, device=DEV)
    w2 = torch.empty(num_experts, R1_H, inter, dtype=torch.bfloat16, device=DEV)
    chunk = max(1, (1 << 26) // (2 * inter * R1_H))
    for lo in range(0, num_experts, chunk):
        hi = min(lo + chunk, num_experts)
        w31[lo:hi] = (
            torch.randn(hi - lo, 2 * inter, R1_H, device=DEV, generator=g) / R1_H**0.5
        ).to(torch.bfloat16)
        w2[lo:hi] = (torch.randn(hi - lo, R1_H, inter, device=DEV, generator=g) / inter**0.5).to(
            torch.bfloat16
        )
    return w31, w2


def _make_r1_routing(num_tokens: int, num_experts: int, seed: int):
    """(x, ids, scales) with top-8 routing over `num_experts` global experts."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(num_tokens, R1_H, device=DEV, generator=g).to(torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts, device=DEV, generator=g)
    vals, ids = torch.topk(logits, R1_K, dim=-1)
    return (
        x,
        ids.to(torch.int32).contiguous(),
        torch.softmax(vals, dim=-1).contiguous(),
    )


# Step-5-Preview's routed experts: hidden 4096, top-8 over 352 global experts,
# the post-SiLU SwiGLU clamp at 7.0 on every expert. Real weights are not
# needed for the claims a cell makes; what the synthetic ones must reproduce is
# the regime the clamp acts in.
S5_H, S5_K, S5_E_GLOBAL = 4096, 8, 352
S5_LIMIT = 7.0
#: FC1 output standard deviation the weights are scaled to. At 4 the 7.0 clamp
#: bites: |up| > 7 on about 8% of the intermediate and silu(gate) > 7 on about
#: 4%, so a reference that dropped either clamp is measurably wrong.
S5_FC1_STD = 4.0
#: The activation's standard deviation the FC2 weights are scaled against, so
#: the layer output has unit scale.
S5_ACT_STD = 5.0
#: Standard deviation of a dequantized random NVFP4 weight: e2m1 codes drawn
#: uniformly (mean square 8.56) under block scales 1/4, 1/2 or 1 (mean square
#: 0.4375). Dividing by a global scale of `E2M1_UNIT_STD / target_std` gives a
#: weight of the target standard deviation.
E2M1_UNIT_STD = (8.5625 * 0.4375) ** 0.5
SV = 16  # NVFP4 block size


def _compare_step5(out: torch.Tensor, ref: torch.Tensor) -> None:
    """8 ulp element-wise against the row scale, 2 ulp of relative RMS.

    The reference reads the same packed operands and models every rounding
    the op performs -- the FC1 output and the activation stored in bf16, and
    on the NVFP4 path the activation requantized under `fc2_act_global` --
    so the two differ only in GEMM accumulation order, in an intermediate
    landing on the other side of a rounding boundary, and in the combine:
    the op adds each of the eight expert rows into the bf16 output, the
    reference sums them in fp32 and rounds once.

    Measured on the default tactics over every Step-5 cell (printed by
    `test_certified_cells`): at most 2.0 ulp element-wise and 0.72 ulp RMS.
    Walking each GEMM's whole tactic space (`test_step5_tactic_space`) moves
    that to 3.3 / 1.12 -- the band is 2.4x and 1.8x that. The wrong
    computations `test_step5_wrong_computations` drives land 47 or more RMS
    ulp out: no clamp 47.6 / 59.9 (nvfp4 / bf16), gate and up swapped 255 /
    251, an expert-index slip in `fc1_global` 113 or `fc2_global` 66.7, the
    expert window one rank off 363.
    """
    assert_within_ulp(out, ref, element_ulp=8.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


def _swizzle_sf(sf: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 swizzle of `[E, M, C]` uint8 block scales, as int32 `[E, M, C/4]`.

    Flat destination of scale `(e, m, c)` is
    `e*M*C + (m//128)*512*(C//4) + (c//4)*512 + (m%32)*16 + ((m%128)//32)*4 + (c%4)`.
    `test_scale_swizzle_matches_block_scale_interleave` pins it to the
    engine's own `block_scale_interleave`, which is what a target's weight
    loader runs.
    """
    e, m, c = sf.shape
    v = sf.reshape(e, m // 128, 4, 32, c // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return v.reshape(e, m, c).contiguous().view(torch.int32)


def _nvfp4_stack(e: int, n: int, k: int, gen: torch.Generator):
    """Random `[e, n, k]` NVFP4 weights: `(packed uint8 [e, n, k/2], scales uint8 [e, n, k/16])`.

    Codes and power-of-two e4m3 scales are drawn directly, so every weight
    value is exact and no quantizer is modelled on the weight side.
    """
    packed = torch.empty(e, n, k // 2, dtype=torch.uint8, device=DEV)
    sf = torch.empty(e, n, k // SV, dtype=torch.uint8, device=DEV)
    for lo in range(0, e, 8):
        hi = min(lo + 8, e)
        codes = torch.randint(0, 16, (hi - lo, n, k), dtype=torch.uint8, device=DEV, generator=gen)
        packed[lo:hi] = codes[..., 0::2] | (codes[..., 1::2] << 4)
        exps = torch.randint(-2, 1, (hi - lo, n, k // SV), device=DEV, generator=gen).float()
        sf[lo:hi] = torch.exp2(exps).to(torch.float8_e4m3fn).view(torch.uint8)
    return packed, sf


def _make_step5_weights(quant: str, num_experts: int, inter: int, hidden: int, seed: int) -> dict:
    """One rank's routed-expert stack, `[up | gate]` FC1 and FC2, at unit output scale.

    bf16: plain `[E, 2I, H]` / `[E, H, I]`. nvfp4: int64-packed codes, the
    swizzled int32 block scales, and a per-expert weight global scale per GEMM
    (spread over a factor of two, so an expert-index slip in `fc1_global` or
    `fc2_global` is a visible error). `raw` keeps the unswizzled pieces.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    if quant == "bf16":
        w31 = torch.empty(num_experts, 2 * inter, hidden, dtype=torch.bfloat16, device=DEV)
        w2 = torch.empty(num_experts, hidden, inter, dtype=torch.bfloat16, device=DEV)
        for lo in range(0, num_experts, 8):
            hi = min(lo + 8, num_experts)
            w31[lo:hi] = (
                torch.randn(hi - lo, 2 * inter, hidden, device=DEV, generator=gen)
                * (S5_FC1_STD / hidden**0.5)
            ).to(torch.bfloat16)
            w2[lo:hi] = (
                torch.randn(hi - lo, hidden, inter, device=DEV, generator=gen)
                / (S5_ACT_STD * inter**0.5)
            ).to(torch.bfloat16)
        return dict(quant=quant, w1=w31, w2=w2)
    p1, s1 = _nvfp4_stack(num_experts, 2 * inter, hidden, gen)
    p2, s2 = _nvfp4_stack(num_experts, hidden, inter, gen)
    spread1 = torch.exp2(torch.rand(num_experts, device=DEV, generator=gen) - 0.5)
    spread2 = torch.exp2(torch.rand(num_experts, device=DEV, generator=gen) - 0.5)
    return dict(
        quant=quant,
        w1=p1.view(torch.int64),
        w2=p2.view(torch.int64),
        sf1=_swizzle_sf(s1),
        sf2=_swizzle_sf(s2),
        gw1=E2M1_UNIT_STD * hidden**0.5 / S5_FC1_STD * spread1,
        gw2=E2M1_UNIT_STD * S5_ACT_STD * inter**0.5 * spread2,
        raw=(p1, s1, p2, s2),
    )


def _step5_call(
    weights: dict,
    num_tokens: int,
    num_global: int,
    seed: int,
    *,
    limit: float = S5_LIMIT,
    clamp_after_silu: bool = True,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    top_k: int = S5_K,
):
    """`(args, kwargs)` for one call, built the way a Step-5 target makes it.

    Routing is noaux_tc-shaped: top-k distinct global ids, sigmoid weights
    normalized and scaled by 3.0, fp32. On the NVFP4 path the input is
    quantized by the `fp4_quantize` entry under its own `448*6/amax` scale
    into the swizzled layout, and the FC2-input global scale is
    `448*6/limit**2` -- the clamp bounds the activation by `limit**2`, so that
    is the calibrated value.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    w1, w2 = weights["w1"], weights["w2"]
    num_local = w1.shape[0]
    hidden = w2.shape[1]
    x = torch.randn(num_tokens, hidden, device=DEV, generator=gen).to(torch.bfloat16)
    logits = torch.randn(num_tokens, num_global, device=DEV, generator=gen)
    vals, ids = torch.topk(logits, top_k, dim=-1)
    w = torch.sigmoid(vals)
    scales = (w / w.sum(dim=-1, keepdim=True) * 3.0).float().contiguous()
    ids = ids.to(torch.int32).contiguous()
    kwargs = dict(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        swiglu_limit=torch.full((num_local,), limit, dtype=torch.float32, device=DEV),
        swiglu_clamp_after_silu=clamp_after_silu,
    )
    if weights["quant"] == "bf16":
        return (x, ids, scales, w1, None, w2, None, torch.bfloat16, []), kwargs
    gs_in = 448.0 * 6.0 / x.abs().max().float()
    xq, xsf = fp4_quantize(x, gs_in.reshape(1), SV, False, True)
    g2 = torch.tensor(448.0 * 6.0 / limit**2, dtype=torch.float32, device=DEV)
    quant_scales = [
        gs_in.reshape(()).contiguous(),
        weights["sf1"],
        (1.0 / (gs_in * weights["gw1"])).float().contiguous(),
        g2,
        weights["sf2"],
        (1.0 / (g2 * weights["gw2"])).float().contiguous(),
    ]
    kwargs.update(input_sf=xsf, swizzled_input_sf=True)
    return (xq, ids, scales, w1, None, w2, None, torch.bfloat16, quant_scales), kwargs


def _step5_cell_call(spec: dict, weights: dict, seed: int):
    return _step5_call(
        weights,
        spec["num_tokens"],
        spec["experts_local"] * spec["ep_size"],
        seed,
        limit=spec["swiglu_limit"],
        clamp_after_silu=spec["clamp_after_silu"],
        tp_size=spec["tp_size"],
        tp_rank=spec["tp_rank"],
        ep_size=spec["ep_size"],
        ep_rank=spec["ep_rank"],
    )


def _step5_weights_for(spec: dict) -> dict:
    key = f"{spec['quant']}:{spec['experts_local']}:{spec['inter']}"
    return _make_step5_weights(
        spec["quant"],
        spec["experts_local"],
        spec["inter"],
        spec["hidden"],
        seed=zlib.crc32(key.encode()) % 2**31,
    )


def _cell(fragment: str):
    """The one cell whose `why` contains `fragment`."""
    found = [c for c in fused_moe.CELLS if fragment in c.why]
    assert len(found) == 1, (fragment, [c.why for c in found])
    return found[0]


#: The C++ runners the `use_fused_finalize=False` calls build and reuse. The op
#: caches one runner per dtype triple in `MoERunner.runner_dict`, a key that
#: leaves `use_fused_finalize` out, and the C++ runner fixes the flag when it is
#: built -- so in a process whose first call on a key fused, a later unfused
#: call on that key runs on the fused runner, whose GEMM2 tactics include ones
#: that fuse the finalize (`test_the_flag_reaches_a_cached_runner` pins that). A
#: target passes one value for the whole process; this file drives both, so it
#: gives the unfused calls a runner cache of their own.
_UNFUSED_RUNNERS: dict = {}

#: The kernel that sums each token's top-k expert rows when the fusion is off.
FINALIZE_KERNEL = "finalizeMoeRoutingKernel"


@contextlib.contextmanager
def _runners_for(fused: bool):
    """Route the op's runner cache to the one built for this `use_fused_finalize`."""
    if fused:
        yield
        return
    held = MoERunner.runner_dict
    MoERunner.runner_dict = _UNFUSED_RUNNERS
    try:
        yield
    finally:
        MoERunner.runner_dict = held


def _kernels(call) -> tuple[object, list[str]]:
    """`(result, names of the CUDA kernels it launched)`."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = call()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return result, names


def _ran_finalize_kernel(names: list[str]) -> bool:
    return any(FINALIZE_KERNEL in n for n in names)


@pytest.fixture(scope="module")
def _weights():
    """One weight set per geometry, built on first use and reused by the cells after it.

    Module-scoped because a cell is cheap and its operands are not: rebuilding
    per case would spend minutes materializing identical tensors. One geometry
    is held at a time -- the cells sharing one are listed back to back, and a
    new geometry frees the last before it is built -- so the peak stays at one
    set: 5.6 GB for an R1 pair, 6.6 GB for Step-5's bf16 ep2 stack.
    """
    held: dict = {}

    def get(spec: dict):
        if "quant" in spec:
            key = (spec["quant"], spec["experts_local"], spec["inter"])
        else:
            key = ("r1", spec["experts_local"], spec["inter"])
        if key not in held:
            held.clear()
            torch.cuda.empty_cache()
            if "quant" in spec:
                held[key] = _step5_weights_for(spec)
            else:
                held[key] = _make_r1_weights(spec["experts_local"], seed=1234, inter=spec["inter"])
        return held[key]

    yield get
    held.clear()
    torch.cuda.empty_cache()


def test_cell_ids_are_distinct() -> None:
    """Every cell keeps a test id of its own (the first 44 characters of `why`)."""
    ids = [c.why[:44] for c in fused_moe.CELLS]
    assert len(set(ids)) == len(ids), sorted(i for i in ids if ids.count(i) > 1)


def test_step5_cells_hold_the_sites_values() -> None:
    """Every Step-5 cell is a share of the 352-expert layer at hidden 4096, top-8,
    clamp 7.0 after SiLU -- the values the target passes, not nearby ones.
    """
    step5 = [c for c in fused_moe.CELLS if "quant" in c.spec]
    assert len(step5) == 32
    for cell in step5:
        spec = cell.spec
        assert spec["hidden"] == S5_H and spec["topk"] == S5_K, spec
        assert spec["experts_local"] * spec["ep_size"] == S5_E_GLOBAL, spec
        assert spec["inter"] * spec["tp_size"] == 1536, spec
        assert spec["swiglu_limit"] == S5_LIMIT and spec["clamp_after_silu"] is True, spec
        # the fused cells leave the flag at the op's default; the unfused ones name it
        assert spec.get("use_fused_finalize", True) is ("unfused" not in cell.why), cell.why
    assert sum("use_fused_finalize" in c.spec for c in step5) == 11


@pytest.mark.parametrize("cell", fused_moe.CELLS, ids=[c.why[:44] for c in fused_moe.CELLS])
def test_certified_cells(cell, certifying, _weights) -> None:  # noqa: F811
    """Every certified configuration, kernel against the entry's own reference."""
    certifying(fused_moe)
    spec = cell.spec
    if "quant" in spec and "use_fused_finalize" in spec:
        _certify_unfused_cell(cell, _weights(spec))
        return
    if "quant" in spec:
        args, kwargs = _step5_cell_call(
            spec, _weights(spec), seed=zlib.crc32(cell.why.encode()) % 2**31
        )
        out = fused_moe(*args, **kwargs)[0]
        ref = fused_moe.reference(*args, **kwargs)
        print(f"  {cell.why[:44]}: {_dev(out, ref)} (elt / rms ulp)")
        _compare_step5(out, ref)
        return
    w31, w2 = _weights(spec)
    x, ids, scales = _make_r1_routing(
        spec["num_tokens"], spec["experts_local"] * spec["ep_size"], seed=spec["num_tokens"]
    )
    parallel = dict(
        tp_size=spec["tp_size"],
        tp_rank=spec["tp_rank"],
        ep_size=spec["ep_size"],
        ep_rank=spec["ep_rank"],
    )
    out = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [], **parallel)[0]
    _compare(
        out,
        fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, [], **parallel),
    )


def _certify_unfused_cell(cell, weights: dict) -> None:
    """An unfused cell: the finalize kernel ran, the band holds, and the answer is
    reproducible to the bit -- which is what the switch is for.

    Same reference and same `_compare_step5` band as the fused twins: with the
    fusion off FC2 stores each expanded row in bf16 and the finalize kernel sums
    a token's eight rows in fp32 and rounds once, so against the fp32-summing
    reference the one rounding the reference leaves out is that per-row bf16 store.
    """
    spec = cell.spec
    assert spec["use_fused_finalize"] is False, spec
    args, kwargs = _step5_cell_call(spec, weights, seed=zlib.crc32(cell.why.encode()) % 2**31)
    kwargs = dict(kwargs, use_fused_finalize=False)
    with _runners_for(fused=False):
        first, names = _kernels(lambda: fused_moe(*args, **kwargs)[0])
        again = fused_moe(*args, **kwargs)[0]
    # the finalize kernel did the sum (on the default tactic a fused runner
    # launches it too; `test_step5_tactic_space` covers the tuned tactics)
    assert _ran_finalize_kernel(names), sorted(set(n[:80] for n in names))
    assert first.data_ptr() != again.data_ptr(), "two calls shared an output buffer"
    assert torch.equal(first, again), (
        f"two identical unfused calls differ in {int((first != again).sum())} elements"
    )
    ref = fused_moe.reference(*args, **kwargs)
    print(f"  {cell.why[:44]}: {_dev(first, ref)} (elt / rms ulp)")
    _compare_step5(first, ref)


# ─── Hazards: claims about inputs no cell describes ───────────────────────


def test_out_of_range_expert_ids_are_dropped() -> None:
    # Ids outside this rank's slot range are silently skipped, not clamped.
    x, ids, scales, w31, w2 = _make(32, 256, 128, 8, 4, seed=29)
    for bad_id in [8, 100, -1]:
        bad = ids.clone()
        bad[:, 1] = bad_id
        y = fused_moe(x, bad, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        kept = torch.zeros_like(bad, dtype=torch.bool)
        kept[:, 0] = True
        kept[:, 2:] = True
        dropped_scales = scales * kept
        _compare(
            y, fused_moe.reference(x, bad, dropped_scales, w31, None, w2, None, torch.bfloat16, [])
        )


def test_repeated_expert_ids() -> None:
    # A row may name the same expert twice; each slot is combined separately.
    # This holds only for num_tokens <= 256, which is what this case drives --
    # past that the op takes a different expert-map path and reads out of
    # bounds on a repeated id (see the contract's Notes).
    x, ids, scales, w31, w2 = _make(32, 256, 128, 8, 4, seed=30)
    dup = ids.clone()
    dup[:, 1] = dup[:, 0]
    y = fused_moe(x, dup, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _compare(y, fused_moe.reference(x, dup, scales, w31, None, w2, None, torch.bfloat16, []))


def test_inputs_untouched_and_deterministic() -> None:
    x, ids, scales, w31, w2 = _make(256, 512, 256, 32, 4, seed=31)
    snap = [t.clone() for t in (x, ids, scales, w31, w2)]
    y_a = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    y_b = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    for t, s in zip((x, ids, scales, w31, w2), snap):
        assert torch.equal(t, s), "an input tensor was mutated"
    assert y_a.data_ptr() != y_b.data_ptr(), "two calls shared an output buffer"
    assert torch.equal(y_a, y_b), "two identical calls disagreed"


def test_wrapper_rejects_output_dtype_mismatch() -> None:
    # Observed silent failure: the store happens in the activation dtype while
    # the buffer is allocated as output_dtype, so the bits are reinterpreted.
    x, ids, scales, w31, w2 = _make(8, 128, 64, 8, 2, seed=41)
    for bad in [torch.float16, torch.float32]:
        try:
            with validating(fused_moe):
                fused_moe(x, ids, scales, w31, None, w2, None, bad, [])
        except AssertionError:
            continue
        raise AssertionError(f"wrapper accepted output_dtype={bad} for a bf16 input")
    # positive control: the matching dtype still works
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _compare(y, fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, []))


def test_op_rejects_unsupported_domains() -> None:
    x, ids, scales, w31, w2 = _make(8, 128, 64, 8, 2, seed=43)
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    wide_x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEV)
    wide_ids = torch.zeros(8, 4, dtype=torch.int32, device=DEV)
    wide_scales = torch.zeros(8, 4, dtype=torch.float32, device=DEV)
    x_bad, ids_bad, sc_bad, w31_bad, w2_bad = _make(8, 136, 72, 8, 2, seed=44)
    x_0, ids_0, sc_0, w31_0, w2_0 = _make(0, 128, 64, 8, 2, seed=45)
    cases = [
        (
            "zero tokens",
            lambda: fused_moe(x_0, ids_0, sc_0, w31_0, None, w2_0, None, torch.bfloat16, []),
        ),
        ("cpu input", lambda: fused_moe(x.cpu(), *args[1:])),
        (
            "cpu weights",
            lambda: fused_moe(x, ids, scales, w31.cpu(), None, w2, None, torch.bfloat16, []),
        ),
        ("3D input", lambda: fused_moe(x.unsqueeze(0), *args[1:])),
        (
            "2D fc1",
            lambda: fused_moe(x, ids, scales, w31[0], None, w2, None, torch.bfloat16, []),
        ),
        ("non-contiguous input", lambda: fused_moe(wide_x[:, :128], *args[1:])),
        ("non-contiguous ids", lambda: fused_moe(x, wide_ids[:, :2], *args[2:])),
        (
            "non-contiguous scales",
            lambda: fused_moe(x, ids, wide_scales[:, :2], *args[3:]),
        ),
        (
            "non-contiguous fc1",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31.transpose(1, 2).contiguous().transpose(1, 2),
                None,
                w2,
                None,
                torch.bfloat16,
                [],
            ),
        ),
        ("int64 ids", lambda: fused_moe(x, ids.long(), *args[2:])),
        ("bf16 scales", lambda: fused_moe(x, ids, scales.bfloat16(), *args[3:])),
        (
            "fp32 activations",
            lambda: fused_moe(
                x.float(),
                ids,
                scales,
                w31.float(),
                None,
                w2.float(),
                None,
                torch.float32,
                [],
            ),
        ),
        (
            "mismatched weight dtype",
            lambda: fused_moe(
                x, ids, scales, w31.half(), None, w2.half(), None, torch.bfloat16, []
            ),
        ),
        (
            "fp32 bias",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31,
                torch.zeros(8, 128, device=DEV),
                w2,
                torch.zeros(8, 128, device=DEV),
                torch.bfloat16,
                [],
            ),
        ),
        (
            "fc1 bias without fc2 bias",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31,
                torch.zeros(8, 128, dtype=torch.bfloat16, device=DEV),
                w2,
                None,
                torch.bfloat16,
                [],
            ),
        ),
        (
            "fc2 bias without fc1 bias",
            lambda: fused_moe(
                x,
                ids,
                scales,
                w31,
                None,
                w2,
                torch.zeros(8, 128, dtype=torch.bfloat16, device=DEV),
                torch.bfloat16,
                [],
            ),
        ),
        (
            "expert count mismatch",
            lambda: fused_moe(
                x, ids, scales, w31, None, w2[:4].contiguous(), None, torch.bfloat16, []
            ),
        ),
        (
            "token count mismatch",
            lambda: fused_moe(x, ids[:4].contiguous(), scales[:4].contiguous(), *args[3:]),
        ),
        (
            "top-k mismatch",
            lambda: fused_moe(x, ids, scales[:, :1].contiguous(), *args[3:]),
        ),
        (
            "hidden_size not a multiple of 8",
            lambda: fused_moe(
                x_bad[:, :132].contiguous(),
                ids_bad,
                sc_bad,
                w31_bad[:, :, :132].contiguous(),
                None,
                w2_bad[:, :132].contiguous(),
                None,
                torch.bfloat16,
                [],
            ),
        ),
        ("min_latency_mode on bf16", lambda: fused_moe(*args, min_latency_mode=True)),
        (
            "deepseek fp8 block scale on bf16",
            lambda: fused_moe(*args, use_deepseek_fp8_block_scale=True),
        ),
        ("int8 woq on bf16", lambda: fused_moe(*args, use_int8_woq_per_channel=True)),
        (
            "mxfp8 weight scaling on bf16",
            lambda: fused_moe(*args, use_mxfp8_weight_scaling=True),
        ),
        (
            "swiglu_alpha with wrong length",
            lambda: fused_moe(*args, swiglu_alpha=torch.ones(1, device=DEV)),
        ),
        (
            "swiglu_alpha in bf16",
            lambda: fused_moe(*args, swiglu_alpha=torch.ones(8, dtype=torch.bfloat16, device=DEV)),
        ),
        # non-gated activation types want an [E, I, H] fc1 instead
        ("Identity activation", lambda: fused_moe(*args, activation_type=1)),
        ("Gelu activation", lambda: fused_moe(*args, activation_type=2)),
        ("Silu activation", lambda: fused_moe(*args, activation_type=4)),
        ("Relu2 activation", lambda: fused_moe(*args, activation_type=8)),
        ("unknown activation", lambda: fused_moe(*args, activation_type=99)),
        ("ep_rank >= ep_size", lambda: fused_moe(*args, ep_size=2, ep_rank=2)),
        (
            "cluster_size without min-latency",
            lambda: fused_moe(*args, cluster_size=2, cluster_rank=0),
        ),
        (
            "tuner_num_tokens without alltoall",
            lambda: fused_moe(*args, tuner_num_tokens=8),
        ),
        ("alltoall without tuner args", lambda: fused_moe(*args, enable_alltoall=True)),
        (
            "lora without max low rank",
            lambda: fused_moe(*args, fc1_lora_ranks=torch.zeros(1, dtype=torch.int32)),
        ),
        (
            "out_tensor with wrong dtype",
            lambda: fused_moe(
                *args, out_tensor=torch.empty(8, 128, dtype=torch.float32, device=DEV)
            ),
        ),
        (
            "out_tensor with wrong shape",
            lambda: fused_moe(
                *args, out_tensor=torch.empty(4, 128, dtype=torch.bfloat16, device=DEV)
            ),
        ),
        (
            "non-contiguous out_tensor",
            lambda: fused_moe(*args, out_tensor=wide_x[:, :128]),
        ),
        (
            "fp8 input without quant scales",
            lambda: fused_moe(
                x.to(torch.float8_e4m3fn),
                ids,
                scales,
                w31.to(torch.float8_e4m3fn),
                None,
                w2.to(torch.float8_e4m3fn),
                None,
                torch.bfloat16,
                [],
            ),
        ),
    ]
    for tag, call in cases:
        try:
            call()
        except (RuntimeError, ValueError, AssertionError):
            continue
        raise AssertionError(f"op accepted an unsupported domain: {tag}")


# ---------------------------------------------------------------------------
# DeepSeek-R1-0528 MTP layer (model.layers.61) routed-expert geometry.
# hidden 7168, moe_intermediate 2048, 256 routed experts, top-8, bf16 weights
# and activations. Under ep_size 4 each rank holds E = 64 of those experts and
# is handed global ids in [0, 256).


def _rejects(y: torch.Tensor, ref: torch.Tensor, tag: str) -> None:
    """Assert the gates reject `ref` as a description of `y`."""
    try:
        _compare(y, ref)
    except AssertionError:
        return
    raise AssertionError(f"tolerance accepted a wrong reference: {tag}")


def test_r1_mtp_expert_relabeling_is_bitwise() -> None:
    # Nothing in the permutation/gather path may key on which expert index a
    # weight sits at: permuting the 64-expert stack and relabelling the ids to
    # match must reproduce the same output. Measured bitwise identical.
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=2222)
    x, ids, scales = _make_r1_routing(512, R1_E_LOCAL, seed=31337)
    y = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
    _compare(y, fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, []))
    g = torch.Generator(device=DEV).manual_seed(4)
    perm = torch.randperm(R1_E_LOCAL, device=DEV, generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(R1_E_LOCAL, device=DEV)
    y_perm = fused_moe(
        x,
        inv[ids.long()].to(torch.int32).contiguous(),
        scales,
        w31[perm].contiguous(),
        None,
        w2[perm].contiguous(),
        None,
        torch.bfloat16,
        [],
    )[0]
    assert torch.equal(y, y_perm), "relabelling the experts changed the output"
    del w31, w2, x, ids, scales, y, y_perm, perm, inv
    torch.cuda.empty_cache()


def test_r1_mtp_autotuned_tactics() -> None:
    # Serving runs on the hot side of the tuner, which a cold receipt never
    # sees. One autotune() pass at this geometry moves the bits and still
    # lands inside the gates; clearing the cache restores the cold bits
    # exactly.
    #
    # How many entries that pass leaves in the cache is the tuner's business,
    # not this op's -- `misc/test_autotuner.py::test_bucket_mapping` owns the
    # bucketing rule. Asserting the count here would fail on a tuner change
    # that says nothing about fused_moe.
    tuner = AutoTuner.get()
    tuner.clear_cache()
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=1234)
    cases = {}
    for t in (1, 256, 8192):
        x, ids, scales = _make_r1_routing(t, R1_E_LOCAL, seed=t)
        cold = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        cases[t] = (
            x,
            ids,
            scales,
            cold,
            fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, []),
        )
        _compare(cold, cases[t][4])
    try:
        x, ids, scales = cases[8192][:3]
        with autotune():
            fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
        assert tuner.profiling_cache, "the autotune pass recorded nothing"
        for key, (_, tactic, _) in tuner.profiling_cache.cache.items():
            assert tactic >= 0, f"{key} kept the fallback tactic after tuning"
        moved = 0
        for t, (x, ids, scales, cold, ref) in cases.items():
            hot = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
            _compare(hot, ref)
            moved += int(not torch.equal(hot, cold))
        # the control this whole test rests on: if tuning had not changed a
        # single bit, a clean pass here would mean nothing
        assert moved > 0, "no warm result differed from its cold counterpart"
    finally:
        tuner.clear_cache()
    for t, (x, ids, scales, cold, _) in cases.items():
        again = fused_moe(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])[0]
        assert torch.equal(again, cold), f"clearing the cache did not restore T={t}"
    del w31, w2, cases
    torch.cuda.empty_cache()


def _dev(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against the row scale, relative RMS), in bf16 ulp."""
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r).abs() / row).max().item() / ULP[torch.bfloat16]
    rms = (o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)
    return round(elt, 2), round(rms.item() / ULP[torch.bfloat16], 3)


def _sweep_tactics(combos, args, ref, count_distinct: bool) -> int:
    """Drive the op once per (runner, tactic) config, gating every output."""
    distinct = set()
    worst = (0.0, 0.0)
    for cfg in combos:
        with AutoTuner.get().replay(cfg):
            y = fused_moe(*args)[0]
        _compare(y, ref)
        d = _dev(y, ref)
        worst = (max(worst[0], d[0]), max(worst[1], d[1]))
        if count_distinct:
            distinct.add(y.view(torch.int16).cpu().numpy().tobytes())
        del y
    print(f"    swept {len(combos)} configs, worst {worst} (elt / rms ulp)")
    return len(distinct)


def _split_index(combos) -> int:
    """Number of gemm2 tactics: itertools.product varies that context fastest."""
    return next(i for i in range(1, len(combos)) if combos[i][0][1] != combos[0][0][1])


def test_r1_mtp_tactic_space() -> None:
    # get_valid_tactics takes no shape, so the tactic population is a property
    # of the build, not of the geometry -- checked here by capturing it at a
    # certified small cell and at this one and comparing the sizes. Then every
    # tactic in it is driven against the torch reference, which makes the
    # receipt cover the warm path as a whole rather than one tuner outcome.
    tuner = AutoTuner.get()
    x_s, ids_s, scales_s, w31_s, w2_s = _make(64, 512, 256, 32, 4, seed=61)
    with tuner.capture() as cap_small:
        fused_moe(x_s, ids_s, scales_s, w31_s, None, w2_s, None, torch.bfloat16, [])
    small = list(cap_small)
    del x_s, ids_s, scales_s, w31_s, w2_s
    torch.cuda.empty_cache()

    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=1234)
    x, ids, scales = _make_r1_routing(256, R1_E_LOCAL, seed=99)
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    ref = fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    with tuner.capture() as cap:
        fused_moe(*args)
    combos = list(cap)
    assert len(combos) == len(small), (len(combos), len(small))
    n2 = _split_index(combos)
    n1 = len(combos) // n2
    assert n1 > 1 and n2 > 1 and n1 * n2 == len(combos), (n1, n2)

    # gemm2 carries nearly all of the spread (asserted below): measured 101
    # distinct outputs from its 309 tactics here against 2 from gemm1's 209,
    # the same counts the small shapes give. Worst deviation over the whole
    # space against the entry's reference: 3.59 / 1.15 ulp here and
    # 3.97 / 1.11 at 8192 tokens below -- inside the 8 / 4 ulp gates.
    d2 = _sweep_tactics(combos[:n2], args, ref, count_distinct=True)
    d1 = _sweep_tactics(combos[::n2], args, ref, count_distinct=True)
    # the blindness control: a harness that could not tell one tactic from
    # another would report the same clean sweep with nothing measured
    assert d2 > 1, d2
    assert d1 < d2, (d1, d2)
    del x, ids, scales, args, ref, combos
    torch.cuda.empty_cache()

    # and at the token count the caller chunks to
    x, ids, scales = _make_r1_routing(8192, R1_E_LOCAL, seed=8192)
    args = (x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    ref = fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    with tuner.capture() as cap:
        fused_moe(*args)
    combos = list(cap)
    _sweep_tactics(combos[: _split_index(combos)], args, ref, count_distinct=False)
    del w31, w2, x, ids, scales, args, ref, combos
    torch.cuda.empty_cache()


# ─── Step-5-Preview: the clamp, the NVFP4 operands, the warm path ──────────


def test_step5_wrong_computations() -> None:
    # The band is only as good as what it rejects. Each wrong computation below
    # is the entry's own reference handed a deliberately wrong argument, at a
    # cell's operands, and each must land well outside `_compare_step5`.
    for fragment in ("step5 nvfp4 tp4 352xI384 T=128", "step5 bf16 tp4 352xI384 T=128"):
        cell = _cell(fragment)
        spec = cell.spec
        weights = _step5_weights_for(spec)
        args, kwargs = _step5_cell_call(spec, weights, seed=zlib.crc32(cell.why.encode()) % 2**31)
        out = fused_moe(*args, **kwargs)[0]
        right = fused_moe.reference(*args, **kwargs)
        _compare_step5(out, right)
        inter = spec["inter"]
        if spec["quant"] == "bf16":
            w1 = weights["w1"]
            swapped = args[:3] + (torch.cat([w1[:, inter:], w1[:, :inter]], dim=1),) + args[4:]
        else:
            p1, s1 = weights["raw"][:2]
            swapped_qs = list(args[8])
            swapped_qs[1] = _swizzle_sf(torch.cat([s1[:, inter:], s1[:, :inter]], dim=1))
            swapped = (
                args[:3]
                + (torch.cat([p1[:, inter:], p1[:, :inter]], dim=1).view(torch.int64),)
                + args[4:8]
                + (swapped_qs,)
            )
        wrong = {
            "no clamp": fused_moe.reference(
                *args, **dict(kwargs, swiglu_limit=None, swiglu_clamp_after_silu=False)
            ),
            "gate/up halves swapped": fused_moe.reference(*swapped, **kwargs),
        }
        if spec["quant"] == "nvfp4":
            rolled = list(args[8])
            rolled[2] = rolled[2].roll(1)
            wrong["fc1_global one expert off"] = fused_moe.reference(*args[:8], rolled, **kwargs)
            rolled = list(args[8])
            rolled[5] = rolled[5].roll(1)
            wrong["fc2_global one expert off"] = fused_moe.reference(*args[:8], rolled, **kwargs)
        print(f"  {fragment}: right {_dev(out, right)}")
        for name, ref in wrong.items():
            elt, rms = _dev(out, ref)
            print(f"    {name:28s} {elt:8.2f} elt / {rms:7.2f} rms ulp")
            assert rms > 10.0, f"{name} is only {rms:.2f} rms ulp out -- not discriminated"
        # The clamp order: at limit 7 the two orders differ only where the
        # gate exceeds 7, by at most 7 * (1 - sigmoid(7)) = 0.0064 -- under a
        # bf16 ulp of the activation. Recorded here; pinned at a small limit
        # in test_step5_clamp_order_at_a_small_limit.
        pre = fused_moe.reference(*args, **dict(kwargs, swiglu_clamp_after_silu=False))
        print(f"    {'clamp before SiLU':28s} {_dev(out, pre)} (elt / rms ulp)")
        assert _dev(out, right)[1] < _dev(out, pre)[1], "post-SiLU is not the closer order"
        del weights, args, kwargs, out, right, wrong, swapped, pre
        torch.cuda.empty_cache()

    # And the expert window: rank 0 of ep4 answering as rank 1.
    cell = _cell("step5 nvfp4 ep4 r0 88xI1536 T=128")
    weights = _step5_weights_for(cell.spec)
    args, kwargs = _step5_cell_call(cell.spec, weights, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = fused_moe(*args, **kwargs)[0]
    elt, rms = _dev(out, fused_moe.reference(*args, **dict(kwargs, ep_rank=1)))
    print(f"    {'ep window one rank off':28s} {elt:8.2f} elt / {rms:7.2f} rms ulp")
    assert rms > 10.0, f"a shifted expert window is only {rms:.2f} rms ulp out"
    del weights, args, kwargs, out
    torch.cuda.empty_cache()


@pytest.mark.parametrize("quant", ["bf16", "nvfp4"])
def test_step5_clamp_order_at_a_small_limit(quant: str) -> None:
    # At limit 0.5 the two clamp orders are far apart (min(silu(g), 0.5) is
    # 0.5 for a large gate where silu(min(g, 0.5)) is 0.31), so this pins
    # which one `swiglu_clamp_after_silu` selects, which the limit-7 cells
    # cannot. Small geometry: the claim is about the activation, not shape.
    weights = _make_step5_weights(quant, num_experts=8, inter=128, hidden=256, seed=71)
    for after in (True, False):
        args, kwargs = _step5_call(
            weights, 64, 8, seed=72, limit=0.5, clamp_after_silu=after, top_k=4
        )
        out = fused_moe(*args, **kwargs)[0]
        _compare_step5(out, fused_moe.reference(*args, **kwargs))
        other = fused_moe.reference(*args, **dict(kwargs, swiglu_clamp_after_silu=not after))
        elt, rms = _dev(out, other)
        print(f"  {quant} clamp_after_silu={after} vs the other order: {elt:.1f} / {rms:.1f} ulp")
        assert rms > 10.0, f"the two clamp orders are only {rms:.2f} rms ulp apart"
    # The flag needs a limit to order against; without one the op refuses.
    with pytest.raises(RuntimeError, match="clamp"):
        fused_moe(*args, **dict(kwargs, swiglu_limit=None, swiglu_clamp_after_silu=True))


def test_scale_swizzle_matches_block_scale_interleave() -> None:
    # The cells build the weights' block-scale layout in torch; a target's
    # loader builds it with the engine's `block_scale_interleave`. The two must
    # be the same bytes, or the cells certify a layout no target hands the op.
    gen = torch.Generator(device=DEV).manual_seed(81)
    for e, m, k in ((3, 768, 4096), (2, 4096, 1536), (2, 256, 256)):
        _, sf = _nvfp4_stack(e, m, k, gen)
        mine = _swizzle_sf(sf).view(torch.uint8).reshape(-1)
        engine = torch.ops.trtllm.block_scale_interleave(sf)
        assert torch.equal(mine, engine.reshape(-1)), (e, m, k)
        # and per expert, the way the loader calls it
        one = torch.ops.trtllm.block_scale_interleave(sf[1].contiguous())
        assert torch.equal(_swizzle_sf(sf[1:2]).view(torch.uint8).reshape(-1), one.reshape(-1))


def test_fc1_act_global_is_inert_on_a_prequantized_input() -> None:
    # `quant_scales[0]` scales a bf16 input the op quantizes itself. The
    # certified input arrives quantized, so the value must not matter -- which
    # is why the reference does not read it. Measured bitwise inert.
    cell = _cell("step5 nvfp4 tp4 352xI384 T=128")
    weights = _step5_weights_for(cell.spec)
    args, kwargs = _step5_cell_call(cell.spec, weights, seed=zlib.crc32(cell.why.encode()) % 2**31)
    base = fused_moe(*args, **kwargs)[0]
    for factor in (2.0, 0.25):
        qs = list(args[8])
        qs[0] = qs[0] * factor
        assert torch.equal(fused_moe(*args[:8], qs, **kwargs)[0], base), factor
    del weights, args, kwargs, base
    torch.cuda.empty_cache()


def _sweep_step5(combos, args, kwargs, ref) -> dict:
    """Drive the op twice per (runner, tactic) config, gating every output.

    Returns the distinct outputs, the worst deviation, how many configs gave two
    different answers to two identical calls, and the finalize-kernel launches
    per call -- 1.0 when no config fused the finalize into GEMM2.
    """
    distinct = set()
    worst = (0.0, 0.0)
    unrepeatable = calls = 0
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for cfg in combos:
            with AutoTuner.get().replay(cfg):
                y = fused_moe(*args, **kwargs)[0]
                again = fused_moe(*args, **kwargs)[0]
            calls += 2
            unrepeatable += int(not torch.equal(y, again))
            _compare_step5(y, ref)
            d = _dev(y, ref)
            worst = (max(worst[0], d[0]), max(worst[1], d[1]))
            distinct.add(hashlib.blake2b(y.view(torch.int16).cpu().numpy().tobytes()).digest())
            del y, again
        torch.cuda.synchronize()
    launches = sum(
        FINALIZE_KERNEL in e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    )
    return dict(
        distinct=len(distinct),
        worst=worst,
        unrepeatable=unrepeatable,
        finalize_per_call=launches / calls,
    )


def _sweep_both_gemms(args, kwargs, ref, fused: bool):
    """`(gemm1 tactics, gemm2 tactics, gemm2 sweep, gemm1 sweep)` on the runner for `fused`."""
    with _runners_for(fused):
        with AutoTuner.get().capture() as cap:
            fused_moe(*args, **kwargs)
        combos = list(cap)
        n2 = _split_index(combos)
        n1 = len(combos) // n2
        assert n1 > 1 and n2 > 1 and n1 * n2 == len(combos), (n1, n2)
        return (
            n1,
            n2,
            _sweep_step5(combos[:n2], args, kwargs, ref),
            _sweep_step5(combos[::n2], args, kwargs, ref),
        )


@pytest.mark.parametrize(
    "fragment",
    [
        "step5 nvfp4 tp4 352xI384 T=128",
        "step5 nvfp4 ep4 r0 88xI1536 T=8192",
        "step5 bf16 tp4 352xI384 T=128",
        "step5 unfused nvfp4 tp4 352xI384 T=128",
        "step5 unfused nvfp4 ep2 r1 176xI1536 T=8192",
        "step5 unfused nvfp4 ep4 r0 88xI1536 T=8192",
        "step5 unfused bf16 tp4 352xI384 T=1",
        "step5 unfused bf16 ep2 r1 176xI1536 T=8192",
    ],
)
def test_step5_tactic_space(fragment: str) -> None:
    # Serving runs on whatever tactics the warm-up autotune pass picked, which
    # a cold cell never sees. So every gemm2 tactic (gemm1 at its first) and
    # every gemm1 tactic (gemm2 at its first) is driven against the reference
    # at Step-5 cells, twice each -- the receipt covers the warm path as a
    # whole rather than one tuner outcome. The tuner picks the two
    # independently; gemm1's tactics are asserted bitwise identical to each
    # other here, so FC2 sees the same input under any of them and a (gemm1,
    # gemm2) pair answers as (first, gemm2) does. Measured worst on the fused
    # runner: 3.3 / 1.11 ulp (nvfp4 tp4, T=128), 3.1 / 0.77 (nvfp4 ep4,
    # T=8192), 3.25 / 1.12 (bf16 tp4, T=128), from 71, 71 and 101 distinct
    # gemm2 outputs.
    #
    # `use_fused_finalize` decides the runner's GEMM2 tactic population. The
    # fused runner's includes tactics whose epilogue adds each expert row into
    # the output, skipping the finalize kernel; the unfused runner's has none,
    # so every unfused call launches the kernel, which sums a token's rows in
    # slot order. Measured: every unfused tactic of either GEMM returns the
    # same bits, call after call. An unfused fragment asserts exactly that,
    # with the fused runner swept at the same operands as its control: there
    # the same harness sees the tactics apart and the finalize skipped.
    cell = _cell(fragment)
    fused = cell.spec.get("use_fused_finalize", True)
    weights = _step5_weights_for(cell.spec)
    args, kwargs = _step5_cell_call(cell.spec, weights, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = fused_moe.reference(*args, **kwargs)
    n1, n2, g2, g1 = _sweep_both_gemms(args, kwargs, ref, fused=True)
    print(f"  {fragment}: fused runner {n1} gemm1 x {n2} gemm2 tactics; gemm2 {g2}; gemm1 {g1}")
    # the blindness control: a harness that could not tell one tactic from
    # another would report the same clean sweep with nothing measured
    assert g2["distinct"] > 1, g2
    assert g2["finalize_per_call"] < 1.0, f"no fused GEMM2 tactic skipped the kernel: {g2}"
    assert g1["distinct"] == 1, f"gemm1 tactics disagree ({g1}); pairs need their own sweep"
    if not fused:
        kwargs = dict(kwargs, use_fused_finalize=False)
        n1, n2, g2, g1 = _sweep_both_gemms(args, kwargs, ref, fused=False)
        print(
            f"  {fragment}: unfused runner {n1} gemm1 x {n2} gemm2 tactics; gemm2 {g2}; gemm1 {g1}"
        )
        for sweep in (g2, g1):
            assert sweep["finalize_per_call"] == 1.0, sweep
            assert sweep["unrepeatable"] == 0, sweep
            assert sweep["distinct"] == 1, sweep
    del weights, args, kwargs, ref
    torch.cuda.empty_cache()


# ─── The finalize switch ───────────────────────────────────────────────────


def _gemm2_finalize_per_call(args, kwargs) -> float:
    """Finalize-kernel launches per call over every GEMM2 tactic of the runner the
    op holds for this call's dtype triple (building it if there is none).
    """
    with AutoTuner.get().capture() as cap:
        fused_moe(*args, **kwargs)
    combos = list(cap)
    gemm2 = combos[: _split_index(combos)]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for cfg in gemm2:
            with AutoTuner.get().replay(cfg):
                fused_moe(*args, **kwargs)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return sum(FINALIZE_KERNEL in n for n in names) / len(gemm2)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="MoERunner.runner_dict keys its cached C++ runner without use_fused_finalize",
)
def test_the_flag_reaches_a_cached_runner() -> None:
    # Detector for an engine defect, not a claim about the cells. The op caches
    # one C++ runner per dtype triple, and the runner fixes `use_fused_finalize`
    # when it is built; after a fused call, an unfused call on the same triple
    # reuses the fused runner, so the tuner can hand it a GEMM2 tactic that
    # fuses the finalize -- the run-to-run variation the flag exists to remove.
    # The cells avoid it with a runner cache per flag (`_runners_for`). When the
    # cache key carries the flag this passes, the strict xfail fails, and this
    # becomes an ordinary assertion.
    weights = _make_step5_weights("bf16", num_experts=8, inter=128, hidden=256, seed=91)
    args, kwargs = _step5_call(weights, 64, 8, seed=92, top_k=4)
    unfused = dict(kwargs, use_fused_finalize=False)
    held = MoERunner.runner_dict
    try:
        MoERunner.runner_dict = {}
        fresh_unfused = _gemm2_finalize_per_call(args, unfused)
        MoERunner.runner_dict = {}
        fresh_fused = _gemm2_finalize_per_call(args, kwargs)
        after_fused = _gemm2_finalize_per_call(args, unfused)  # same cache, built fused
    finally:
        MoERunner.runner_dict = held
    print(f"  finalize launches per GEMM2 tactic: {fresh_unfused=} {fresh_fused=} {after_fused=}")
    # the controls, raised as something other than the xfail's AssertionError so
    # a broken control fails the test instead of passing as the expected failure
    if fresh_unfused != 1.0 or fresh_fused >= 1.0:
        raise RuntimeError(
            f"the launch count cannot tell the runners apart: {fresh_unfused=} {fresh_fused=}"
        )
    if after_fused != 1.0:
        raise AssertionError(
            f"use_fused_finalize=False ran on the cached fused runner ({after_fused=:.3f})"
        )
