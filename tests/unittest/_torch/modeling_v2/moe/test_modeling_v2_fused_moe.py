# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_moe catalog entry.

The entry owns the contract: `fused_moe.CELLS` is what is certified,
`reference` is what it is certified against, and `compare` is the band. This
file owns what a cell cannot carry -- building 5.6 GB of weights once and
reusing them, and the hazards, which are claims about inputs no cell
describes.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fused_moe import fused_moe
from tensorrt_llm._torch.autotuner import AutoTuner, autotune

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "fused_moe requires a CUDA device"

DEV = "cuda"
# The reference GEMMs must be true fp32; TF32 would leave the reference with
# 10 mantissa bits, coarser than the bf16 output it is meant to bound.
torch.backends.cuda.matmul.allow_tf32 = False


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

# The R1-0528 MTP layer's MoE: the one geometry this entry is certified at.
R1_H, R1_I, R1_K = 7168, 2048, 8
R1_E_LOCAL, R1_EP = 64, 4
R1_E_GLOBAL = R1_E_LOCAL * R1_EP


def _make_r1_weights(num_experts: int, seed: int):
    """One rank's [E, 2I, H] / [E, H, I] bf16 weight pair, built in chunks.

    5.6 GB at E = 64. Materializing the fp32 randn for the whole stack at once
    would transiently need another 11 GB; chunking caps the temporary at 64 MB.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    w31 = torch.empty(num_experts, 2 * R1_I, R1_H, dtype=torch.bfloat16, device=DEV)
    w2 = torch.empty(num_experts, R1_H, R1_I, dtype=torch.bfloat16, device=DEV)
    chunk = max(1, (1 << 26) // (2 * R1_I * R1_H))
    for lo in range(0, num_experts, chunk):
        hi = min(lo + chunk, num_experts)
        w31[lo:hi] = (torch.randn(hi - lo, 2 * R1_I, R1_H, device=DEV, generator=g) / R1_H**0.5).to(
            torch.bfloat16
        )
        w2[lo:hi] = (torch.randn(hi - lo, R1_H, R1_I, device=DEV, generator=g) / R1_I**0.5).to(
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


@pytest.fixture(scope="module")
def _r1_weights():
    """One 5.6 GB weight pair for every cell that shares the geometry.

    Module-scoped because a cell is cheap and its operands are not: rebuilding
    per case would spend minutes materializing identical tensors. Every cell
    in this entry is at the one geometry, so one pair serves all of them.
    """
    specs = {(c.spec["experts_local"], c.spec["hidden"], c.spec["inter"]) for c in fused_moe.CELLS}
    assert len(specs) == 1, f"cells no longer share one geometry: {specs}"
    w31, w2 = _make_r1_weights(R1_E_LOCAL, seed=1234)
    yield w31, w2
    del w31, w2
    torch.cuda.empty_cache()


@pytest.mark.parametrize("cell", fused_moe.CELLS, ids=[c.why[:36] for c in fused_moe.CELLS])
def test_certified_cells(cell, _r1_weights) -> None:
    """Every certified configuration, kernel against the entry's own reference."""
    w31, w2 = _r1_weights
    spec = cell.spec
    x, ids, scales = _make_r1_routing(
        spec["num_tokens"], spec["experts_local"] * spec["ep_size"], seed=spec["num_tokens"]
    )
    with validating(fused_moe):
        out = fused_moe(
            x,
            ids,
            scales,
            w31,
            None,
            w2,
            None,
            torch.bfloat16,
            [],
            ep_size=spec["ep_size"],
            ep_rank=spec["ep_rank"],
        )[0]
    fused_moe.compare(
        out,
        fused_moe.reference(
            x,
            ids,
            scales,
            w31,
            None,
            w2,
            None,
            torch.bfloat16,
            [],
            ep_size=spec["ep_size"],
            ep_rank=spec["ep_rank"],
        ),
    )


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
        fused_moe.compare(
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
    fused_moe.compare(
        y, fused_moe.reference(x, dup, scales, w31, None, w2, None, torch.bfloat16, [])
    )


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
    fused_moe.compare(
        y, fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    )


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
        fused_moe.compare(y, ref)
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
    fused_moe.compare(
        y, fused_moe.reference(x, ids, scales, w31, None, w2, None, torch.bfloat16, [])
    )
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
        fused_moe.compare(cold, cases[t][4])
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
            fused_moe.compare(hot, ref)
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


def _sweep_tactics(combos, args, ref, count_distinct: bool) -> int:
    """Drive the op once per (runner, tactic) config, gating every output."""
    distinct = set()
    for cfg in combos:
        with AutoTuner.get().replay(cfg):
            y = fused_moe(*args)[0]
        fused_moe.compare(y, ref)
        if count_distinct:
            distinct.add(y.view(torch.int16).cpu().numpy().tobytes())
        del y
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

    # gemm2 carries nearly all of the spread: measured 101 distinct outputs
    # from its 309 tactics here against 2 from gemm1's 209, the same counts the
    # small shapes give. Worst deviation over the whole space: 3.71 ulp
    # element-wise here and 3.98 at 8192 tokens below -- inside the 8 ulp gate
    # with 2.0x margin.
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
