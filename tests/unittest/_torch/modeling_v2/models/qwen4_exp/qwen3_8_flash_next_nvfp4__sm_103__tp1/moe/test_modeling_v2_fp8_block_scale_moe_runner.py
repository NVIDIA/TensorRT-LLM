# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_block_scale_moe_runner catalog entry.

The entry owns the contract: `CELLS`, `reference`, `is_valid`. This
file turns a cell into operands and holds the claims no cell carries: the
intermediate requantization pinned bit-exactly, how far each wrong reference
lands, the expert-window bookkeeping, and the guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    ACT_AMAX_FLOOR,
    BLOCK,
    E4M3_MAX,
    dequantize_block_weight,
    requantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "fp8_block_scale_moe_runner requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16 ulp element-wise, 0.75 ulp RMS, both in bf16 ulp of each row's max.

    Both statistics normalize each element by its reference row's largest
    magnitude: `d = (out - ref) / (rowmax * 2^-8)`, gated on `max|d|` and
    on `sqrt(mean(d^2))` -- the RMS of the row-normalized deviation, not
    `assert_within_ulp`'s ratio of norms, which reads higher on the same
    data (a row's max sits several sigma above its RMS). Kernel and
    reference consume bit-identical operands and model both
    requantizations, so they differ by accumulation order and by FC1 sums
    that land on an e4m3 rounding boundary.

    Measured worst over the cells: 6.65 element-wise (T=4096; the max over
    T*H samples grows with T -- 0.91 at T=1) and 0.306 RMS (H=I=512), the
    RMS sitting at 0.17-0.31 everywhere; earlier draws reached 10.03 and
    0.342. Wrong references at the checkpoint geometry, RMS: gate/up swap
    65, transposed activation scales 43, strided hidden_states 94, no
    FC1-output requant 3.5, neither requant 3.1, no activation requant 1.58
    -- the closest, which is why the intermediate requantization is also
    pinned bit-exactly by the test.
    """
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    o, r = out.float(), ref.float()
    rowmax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    d = (o - r) / (rowmax * 2.0**-8)
    elt, rms = d.abs().max().item(), d.pow(2).mean().sqrt().item()
    assert elt <= 16.0, f"{elt:.2f} ulp element-wise > 16"
    assert rms <= 0.75, f"{rms:.3f} ulp RMS > 0.75"


DEV = "cuda"
ULP = 2.0**-8
CKPT = dict(hidden=2560, inter=640, num_experts=512, top_k=10)

# ── operands ──────────────────────────────────────────────────────────────

_WEIGHTS: dict = {}


def _weights(hidden: int, inter: int, experts: int, seed: int):
    """`(w1, w1s, w2, w2s)` drawn directly in e4m3 / fp32, so every value is exact.

    Cached: the checkpoint stack is 2.5 GB and every checkpoint-geometry cell
    reads the same one. Drawn 32 experts at a time to cap the fp32 transient.
    """
    key = (hidden, inter, experts, seed)
    if key not in _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(seed)
        w1 = torch.empty(experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=DEV)
        w2 = torch.empty(experts, hidden, inter, dtype=torch.float8_e4m3fn, device=DEV)
        for lo in range(0, experts, 32):
            hi = min(lo + 32, experts)
            w1[lo:hi] = torch.randn(hi - lo, 2 * inter, hidden, device=DEV, generator=g)
            w2[lo:hi] = torch.randn(hi - lo, hidden, inter, device=DEV, generator=g)
        w1s = 0.01 + 0.02 * torch.rand(
            experts, 2 * inter // BLOCK, hidden // BLOCK, device=DEV, generator=g
        )
        w2s = 0.01 + 0.02 * torch.rand(
            experts, hidden // BLOCK, inter // BLOCK, device=DEV, generator=g
        )
        _WEIGHTS[key] = (w1, w1s, w2, w2s)
    return _WEIGHTS[key]


def _args(
    tokens,
    hidden,
    inter,
    num_experts,
    top_k,
    local_expert_offset=0,
    local_num_experts=None,
    gemm1_clamp_limit=None,
    foreign_ids=False,
    seed=0,
):
    """The op's arguments by name, pre-routed, for one configuration.

    Weights are drawn for the whole routing space and the window sliced out
    of them, so a window and the full stack hold the same experts.
    """
    local_num_experts = num_experts if local_num_experts is None else local_num_experts
    stack = max(num_experts, local_expert_offset + local_num_experts)
    w1, w1s, w2, w2s = _weights(hidden, inter, stack, seed=hidden + inter + stack)
    window = slice(local_expert_offset, local_expert_offset + local_num_experts)
    g = torch.Generator(device=DEV).manual_seed(seed)
    hs = (2.0 * torch.randn(tokens, hidden, device=DEV, generator=g)).to(torch.float8_e4m3fn)
    # [H/128, T]: the transposed layout fp8_quantize_1x128 returns here.
    hs_scale = 0.5 + torch.rand(hidden // BLOCK, tokens, device=DEV, generator=g)
    logits = torch.randn(tokens, num_experts, device=DEV, generator=g)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    ids = ids.to(torch.int32)
    if foreign_ids:
        ids[0, 0], ids[1, 0], ids[2, 0] = -1, num_experts, 40
    return dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_weights=w1[window].contiguous(),
        gemm1_weights_scale=w1s[window].contiguous(),
        gemm2_weights=w2[window].contiguous(),
        gemm2_weights_scale=w2s[window].contiguous(),
        num_experts=num_experts,
        top_k=top_k,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,
        topk_weights=weights.to(torch.bfloat16).contiguous(),
        topk_ids=ids.contiguous(),
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def _ulp(out: torch.Tensor, ref: torch.Tensor):
    """(max, RMS) deviation in bf16 ulp of the row max -- `_compare`'s statistics."""
    o, r = out.float(), ref.float()
    d = (o - r) / (r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) * ULP)
    return d.abs().max().item(), d.pow(2).mean().sqrt().item()


def _variant(
    a, swap=False, quant_fc1_out=True, quant_act=True, count_repeats=True, requant=requantize_1x128
):
    """`reference` with one step changed -- the wrong readings it must refute."""
    tokens, hidden = a["hidden_states"].shape
    inter, nb = a["intermediate_size"], hidden // BLOCK
    act_in = (
        a["hidden_states"].float().reshape(tokens, nb, BLOCK)
        * a["hidden_states_scale"].t().reshape(tokens, nb, 1)
    ).reshape(tokens, hidden)
    ids = a["topk_ids"].long()
    local = ids - a["local_expert_offset"]
    live = (local >= 0) & (local < a["local_num_experts"])
    if not count_repeats:
        first = torch.ones_like(live)
        for j in range(1, ids.shape[1]):
            first[:, j] = (ids[:, j : j + 1] != ids[:, :j]).all(dim=1)
        live &= first
    out = torch.zeros(tokens, hidden, device=DEV)
    with true_fp32_matmul():
        for e in torch.unique(local[live]).tolist():
            tok, slot = ((local == e) & live).nonzero(as_tuple=True)
            y1 = (
                act_in[tok]
                @ dequantize_block_weight(a["gemm1_weights"][e], a["gemm1_weights_scale"][e]).t()
            )
            if quant_fc1_out:
                y1 = requant(y1)
            up, gate = y1[:, :inter], y1[:, inter:]
            if swap:
                up, gate = gate, up
            act = torch.nn.functional.silu(gate) * up
            if quant_act:
                act = requant(act)
            y = (
                act
                @ dequantize_block_weight(a["gemm2_weights"][e], a["gemm2_weights_scale"][e]).t()
            )
            out.index_add_(0, tok, y * a["topk_weights"][tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode())


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _args(**cell.spec, seed=_seed(cell))
    out = op(**args)
    assert out.shape == args["hidden_states"].shape and out.dtype == torch.bfloat16
    ref = op.reference(**args)
    elt, rms = _ulp(out, ref)
    print(f"  {cell.why[:44]}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    _compare(out, ref)


# ── the two internal requantizations, read out bit-exactly ────────────────


def _pow2_requantize(x: torch.Tensor) -> torch.Tensor:
    """The MX-style round-up power-of-two block scale -- a wrong recipe here."""
    shp = x.shape
    xb = x.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    coded = (xb / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (coded.float() * scale).reshape(shp)


@pytest.mark.parametrize("tokens,width", [(16, 256), (64, 640), (256, 128)])
def test_intermediate_is_fp8_block_quantized(tokens, width):
    """Identity down projection, one slot of weight 1.0: the output is the
    requantized intermediate, one bf16 round away. The recipe reproduces
    >=99.9% of it bit-exactly (measured 99.95-100%); every neighbouring recipe
    reproduces under 10%. The misses are FC1 sums on an e4m3 rounding boundary.
    """
    num_experts = 8
    a = _args(tokens, width, width, num_experts, 1, seed=width + tokens)
    eye = torch.eye(width, device=DEV).to(torch.float8_e4m3fn)
    a["gemm2_weights"] = eye.expand(num_experts, width, width).contiguous()
    a["gemm2_weights_scale"] = torch.ones(
        num_experts, width // BLOCK, width // BLOCK, device=DEV, dtype=torch.float32
    )
    a["topk_weights"] = torch.ones(tokens, 1, device=DEV, dtype=torch.bfloat16)
    out = op(**a)
    exact = (op.reference(**a) == out).float().mean().item()
    print(f"  T={tokens} width={width}: recipe reproduces {100 * exact:.2f}%")
    assert exact >= 0.999, f"modelled recipe reproduced only {100 * exact:.2f}%"

    assert torch.equal(_variant(a), op.reference(**a)), "the variant model drifted from reference"
    wrong = {
        "power-of-two scale": _variant(a, requant=_pow2_requantize),
        "no activation requant": _variant(a, quant_act=False),
        "no FC1-output requant": _variant(a, quant_fc1_out=False),
        "gate/up swapped": _variant(a, swap=True),
    }
    for label, pred in wrong.items():
        frac = (pred == out).float().mean().item()
        print(f"    {label}: {100 * frac:.2f}%")
        assert frac <= 0.10, f"{label} reproduced {100 * frac:.2f}% -- not discriminated"


# ── what the gate separates ───────────────────────────────────────────────


def test_wrong_variants_land_outside_the_gate():
    """Every operand-level mistake, measured against one call at the checkpoint."""
    a = _args(1024, **CKPT, seed=77)
    out = op(**a)
    _compare(out, op.reference(**a))
    good = _ulp(out, op.reference(**a))[1]
    for label, kw in (
        ("gate/up swapped", dict(swap=True)),
        ("no FC1-output requant", dict(quant_fc1_out=False)),
        ("no activation requant", dict(quant_act=False)),
        ("neither requant", dict(quant_fc1_out=False, quant_act=False)),
    ):
        elt, rms = _ulp(out, _variant(a, **kw))
        print(f"  {label}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
        # The RMS gate separates all four; the element gate does not separate
        # a missing activation requant, hence the bit-exact pin above.
        assert rms > 0.75, f"{label} is inside the RMS gate"
        assert rms >= 2.0 * good, f"{label} is within 2x of the correct reference"


def test_transposed_activation_scale_is_a_silent_wrong_answer():
    """At `T == H/128` both of the op's dimension checks pass on the transpose."""
    a = _args(20, **CKPT, seed=99)
    assert a["hidden_states_scale"].shape == (20, 20)
    _compare(op(**a), op.reference(**a))
    swapped = op(**dict(a, hidden_states_scale=a["hidden_states_scale"].t().contiguous()))
    elt, rms = _ulp(swapped, op.reference(**a))
    print(f"  transposed: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    assert rms > 10.0, "the transposed layout must be visibly wrong"


# ── expert parallelism and the window ─────────────────────────────────────


def test_expert_parallel_windows_sum_to_the_full_stack():
    """Four 128-wide windows of 512: each matches its own reference, a token
    with no slot in the window is zero, and the four sum to the full call."""
    full = _args(64, **CKPT, seed=7)
    out = op(**full)
    _compare(out, op.reference(**full))
    ids = full["topk_ids"].long()
    summed = torch.zeros(out.shape, device=DEV)
    for offset in (0, 128, 256, 384):
        a = _args(64, **CKPT, local_expert_offset=offset, local_num_experts=128, seed=7)
        assert torch.equal(a["topk_ids"], full["topk_ids"])
        got = op(**a)
        _compare(got, op.reference(**a))
        dead = (got == 0).all(dim=-1)
        expected_dead = ~((ids >= offset) & (ids < offset + 128)).any(dim=-1)
        assert torch.equal(dead, expected_dead), "a token with no slot in the window must be zero"
        summed += got.float()
    # Each window rounds its partial to bf16 first: inside the gate, not bitwise.
    _compare(summed.to(torch.bfloat16), op.reference(**full))


def test_window_past_the_routing_space_is_all_zero():
    a = _args(64, 256, 128, 8, 2, local_expert_offset=8, local_num_experts=4, seed=22)
    assert bool((op(**a) == 0).all())


def test_repeated_expert_ids_are_counted_once_or_twice_by_token_count():
    """Counted once or twice depending on which routing kernel the token count
    selects. Records the verdict per token count; asserts only that each call
    matches one of the two readings, not that both readings occur."""
    seen = set()
    for tokens in (16, 256):
        a = _args(tokens, 256, 128, 8, 2, seed=tokens * 3 + 256)
        a["topk_ids"][:, 1] = a["topk_ids"][:, 0]
        out = op(**a)
        _, twice = _ulp(out, _variant(a, count_repeats=True))
        _, once = _ulp(out, _variant(a, count_repeats=False))
        verdict = "once" if once < twice else "twice"
        print(f"  T={tokens}: counted {verdict} (once={once:.3f} twice={twice:.3f} ulp RMS)")
        assert min(once, twice) <= 0.75, "neither reading of a repeat matches"
        seen.add(verdict)
    print(f"  verdicts across token counts: {sorted(seen)}")


# ── arguments ─────────────────────────────────────────────────────────────


def test_gemm1_clamp_limit_is_live():
    a = _args(16, **CKPT, gemm1_clamp_limit=1.0, seed=16)
    clamped = op(**a)
    _compare(clamped, op.reference(**a))
    _, rms = _ulp(clamped, op.reference(**dict(a, gemm1_clamp_limit=None)))
    print(f"  clamp=1.0 vs the unclamped reference: {rms:.3f} ulp RMS")
    assert rms > 0.75, "the clamp must change the result"
    b = dict(a)
    del b["gemm1_clamp_limit"]
    assert torch.equal(op(**dict(a, gemm1_clamp_limit=None)), op(**b))


def test_inert_arguments_are_bitwise_identical():
    a = _args(16, **CKPT, seed=16)
    base = op(**a)
    for label, kw in (
        ("routed_scaling_factor=1.0", dict(routed_scaling_factor=1.0)),
        ("routed_scaling_factor=2.5", dict(routed_scaling_factor=2.5)),
        ("routing_method_type=0", dict(routing_method_type=0)),
        ("routing_method_type=2", dict(routing_method_type=2)),
        ("routing_method_type=4", dict(routing_method_type=4)),
        ("n_group=1, topk_group=1", dict(n_group=1, topk_group=1)),
        ("tune_max_num_tokens=128", dict(tune_max_num_tokens=128)),
        ("use_dp=True", dict(use_dp=True)),
    ):
        assert torch.equal(op(**dict(a, **kw)), base), f"{label} changed the result"


def test_autotuned_tactic_matches_the_fallback_tactic():
    """A warm profiling cache changes the tactic, not the result."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    a = _args(256, **CKPT, seed=42)
    AutoTuner.get().clear_cache()
    try:
        cold = op(**a)
        with autotune():
            op(**a)
        entries = len(AutoTuner.get().profiling_cache)
        warm = op(**a)
        print(f"  warming wrote {entries} cache entries")
        assert entries > 0, "warming profiled nothing -- the comparison would be vacuous"
        assert torch.equal(warm, cold), "a tuned tactic changed the result"
        _compare(warm, op.reference(**a))
    finally:
        AutoTuner.get().clear_cache()


def test_registered_fake_returns_the_wrong_hidden_size():
    """An upstream defect, recorded so nothing relies on it: the fake assumes
    packed fp4 hidden states and doubles the width."""
    a = _args(16, 256, 128, 8, 2, seed=3)
    real = op(**a)
    with torch._subclasses.FakeTensorMode():
        fake_args = {
            k: (torch.empty(v.shape, dtype=v.dtype, device=DEV) if torch.is_tensor(v) else v)
            for k, v in a.items()
        }
        fake_shape = tuple(op(**fake_args).shape)
    print(f"  fake says {fake_shape}, the kernel returns {tuple(real.shape)}")
    assert fake_shape == (real.shape[0], 2 * real.shape[1])


def test_determinism_and_purity():
    a = _args(16, 512, 256, 16, 4, seed=25)
    snaps = {k: v.clone() for k, v in a.items() if torch.is_tensor(v)}
    assert torch.equal(op(**a), op(**a))
    for k, v in snaps.items():
        assert torch.equal(a[k], v), f"{k} was mutated"


def test_output_buffer_semantics():
    """`output=` is written in place and the call returns an empty `[0]` tensor;
    rows of a taller buffer past `T` are untouched."""
    a = _args(16, 256, 128, 8, 2, seed=3)
    fresh = op(**a)
    buf = torch.full((32, 256), 7.0, device=DEV, dtype=torch.bfloat16)
    ret = op(**dict(a, output=buf[:16]))
    assert ret.shape == (0,) and ret.dtype == torch.bfloat16
    assert torch.equal(buf[:16], fresh)
    assert bool((buf[16:] == 7.0).all()), "rows past T must be untouched"


def test_rejections():
    """The op's own loud checks; none of these needs a guard."""
    a = _args(16, 256, 128, 8, 2, seed=3)

    def refuses(label, **kw):
        with pytest.raises(RuntimeError) as excinfo:
            op(**dict(a, **kw))
            torch.cuda.synchronize()
        print(f"  {label}: {str(excinfo.value).splitlines()[0][:100]}")

    refuses("bf16 hidden_states", hidden_states=a["hidden_states"].to(torch.bfloat16))
    refuses("bf16 hidden_states_scale", hidden_states_scale=a["hidden_states_scale"].bfloat16())
    refuses("1-D hidden_states_scale", hidden_states_scale=a["hidden_states_scale"].flatten())
    refuses("fp16 weight scale", gemm1_weights_scale=a["gemm1_weights_scale"].half())
    refuses("int64 topk_ids", topk_ids=a["topk_ids"].long())
    refuses("fp32 topk_weights", topk_weights=a["topk_weights"].float())
    refuses("no routing given", topk_ids=None, topk_weights=None)
    refuses("intermediate_size mismatch", intermediate_size=256)
    refuses("num_experts == top_k", num_experts=2)
    refuses("fused shared experts pre-routed", num_fused_shared_experts=1)
    refuses("output of the wrong dtype", output=torch.empty(16, 256, device=DEV).half())
    refuses(
        "output of the wrong shape",
        output=torch.empty(16, 128, device=DEV, dtype=torch.bfloat16),
    )


# ── the guards ────────────────────────────────────────────────────────────


def _strided(t: torch.Tensor) -> torch.Tensor:
    wide = torch.zeros(*t.shape[:-1], 2 * t.shape[-1], device=DEV, dtype=t.dtype)
    wide[..., ::2] = t
    view = wide[..., ::2]
    assert not view.is_contiguous()
    return view


def test_unguarded_hazards_are_silent():
    """What `is_valid` exists for: the raw op takes each of these and answers."""
    a = _args(16, 256, 128, 8, 2, seed=11)
    good = op.reference(**a)
    _, rms = _ulp(op(**dict(a, hidden_states=_strided(a["hidden_states"]))), good)
    print(f"  strided hidden_states accepted: {rms:.3f} ulp RMS")
    assert rms > 10.0, "a strided operand must be visibly wrong, or the guard is pointless"
    base = op(**a)
    assert torch.equal(op(**dict(a, act_type=1)), base), "act_type is no longer ignored"
    bias = torch.tensor([1e3, -1e3] * 4, device=DEV, dtype=torch.bfloat16)
    assert torch.equal(op(**dict(a, routing_bias=bias)), base), (
        "routing_bias is no longer ignored on the pre-routed path"
    )


def test_guards_refuse():
    a = _args(16, 256, 128, 8, 2, seed=11)
    with validating(op):
        ok = op(**a)
        for name in (
            "hidden_states",
            "hidden_states_scale",
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
            "topk_weights",
            "topk_ids",
        ):
            with pytest.raises(AssertionError, match="must be contiguous"):
                op(**dict(a, **{name: _strided(a[name])}))
        with pytest.raises(AssertionError, match="silently ignored"):
            op(**dict(a, act_type=1))
        with pytest.raises(AssertionError, match="routing_bias is ignored"):
            op(**dict(a, routing_bias=torch.zeros(8, device=DEV, dtype=torch.bfloat16)))
        assert torch.equal(op(**a), ok)
