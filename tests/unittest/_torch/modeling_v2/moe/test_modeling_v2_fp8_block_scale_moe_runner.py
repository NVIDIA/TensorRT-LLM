# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_block_scale_moe_runner catalog entry.

The entry owns the contract: `CELLS`, `reference`, `is_valid`. This
file turns a cell into operands and holds the claims no cell carries: the
intermediate requantization pinned bit-exactly, how far each wrong reference
lands, the expert-window bookkeeping, the tep4 kernel path by token count,
tactic invariance, the experts past 2^31 bytes, and the guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    ACT_AMAX_FLOOR,
    BLOCK,
    E4M3_MAX,
    dequantize_block_weight,
    requantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner as op,
)

__extra_import_path__ = [".."]
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

    Measured at the tep4 geometry (H = I = 6144, top-4 of 128, 32 local): at
    most 5.70 element-wise (8192-9216 tokens; 2 or less at 1..16), RMS
    0.11-0.27. Wrong references at tep4 (1024 tokens), RMS: gate/up swap
    37.6, transposed scales 16.4 (48 tokens), no FC1-output requant 2.77,
    neither 2.48, no activation requant 1.23 -- the closest, which is why the
    intermediate requantization is also pinned bit-exactly by the test.
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
#: Mistral Large 4's routed experts on tep4 rank 1, as the tep4 cells call them.
TEP4 = dict(
    hidden=6144,
    inter=6144,
    num_experts=128,
    top_k=4,
    local_expert_offset=32,
    local_num_experts=32,
    tune_max_num_tokens=9216,
)

# ── operands ──────────────────────────────────────────────────────────────

_WEIGHTS: dict = {}


def _weights(hidden: int, inter: int, experts: int, seed: int):
    """`(w1, w1s, w2, w2s)` drawn directly in e4m3 / fp32, so every value is exact.

    Cached per geometry and seed. Drawn 32 experts at a time to cap the fp32
    transient.
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
    )


# ── operands at the tep4 geometry: the checkpoint's and the engine's recipe ──


def _window_weights(hidden: int, inter: int, local: int):
    """The rank's `local` experts only, with power-of-two 128x128 scales.

    The checkpoint's expert qscales are UE8M0 powers of two; 2^-8..2^-6 keeps
    the products in the e4m3 range the codes are drawn for. Every rank's
    window holds this one stack (the offset only names which global ids it
    answers for). Cached: 3.6 GB at H = I = 6144, 32 experts.
    """
    key = ("window", hidden, inter, local)
    if key not in _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(hidden + inter + local + 1)
        w1 = torch.empty(local, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=DEV)
        w2 = torch.empty(local, hidden, inter, dtype=torch.float8_e4m3fn, device=DEV)
        for e in range(local):
            w1[e] = torch.randn(2 * inter, hidden, device=DEV, generator=g)
            w2[e] = torch.randn(hidden, inter, device=DEV, generator=g)
        exps = dict(device=DEV, generator=g)
        w1s = torch.exp2(
            -torch.randint(6, 9, (local, 2 * inter // BLOCK, hidden // BLOCK), **exps).float()
        )
        w2s = torch.exp2(
            -torch.randint(6, 9, (local, hidden // BLOCK, inter // BLOCK), **exps).float()
        )
        _WEIGHTS[key] = (w1, w1s, w2, w2s)
    return _WEIGHTS[key]


def _quantize_1x128(x: torch.Tensor):
    """bf16 `[T, H]` -> e4m3 codes and `[H/128, T]` scales by the plain amax/448
    recipe `fp8_quantize_1x128(x, use_ue8m0=False)` applies -- so the codes
    span the e4m3 range the way the engine's do."""
    tokens, hidden = x.shape
    xb = x.float().reshape(tokens, hidden // BLOCK, BLOCK)
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR) / E4M3_MAX
    codes = (xb / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return codes.reshape(tokens, hidden), scale.reshape(tokens, hidden // BLOCK).t().contiguous()


def _tep4_args(spec, tokens: int, seed: int, far: bool = False):
    """The op's arguments for one tep4 call of `tokens` tokens.

    Routing is the renormalized top-k the target's router produces (top-k of
    the logits, softmax over the k), in bf16. Token 0 always has a slot in the
    window, so no call is vacuously all-zero; with `far`, slot 0 of every
    token is one of the window's last three experts (local 29, 30, 31), whose
    gemm1 rows start past 2^31 bytes at this geometry. Forced picks keep the
    weights of their unforced logits.
    """
    hidden, inter, top_k = spec["hidden"], spec["inter"], spec["top_k"]
    num_experts, offset = spec["num_experts"], spec["local_expert_offset"]
    local = spec["local_num_experts"]
    w1, w1s, w2, w2s = _window_weights(hidden, inter, local)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(tokens, hidden, device=DEV, generator=g).to(torch.bfloat16)
    hs, hs_scale = _quantize_1x128(x)
    logits = torch.randn(tokens, num_experts, device=DEV, generator=g)
    force = torch.zeros_like(logits)
    rows = torch.arange(tokens, device=DEV)
    if far:
        force[rows, offset + local - 3 + rows % 3] = 1e4
    else:
        force[0, offset + seed % local] = 1e4
    ids = torch.topk(logits + force, top_k, dim=-1).indices
    weights = torch.softmax(logits.gather(1, ids), dim=-1)
    return dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1s,
        gemm2_weights=w2,
        gemm2_weights_scale=w2s,
        num_experts=num_experts,
        top_k=top_k,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=offset,
        local_num_experts=local,
        routed_scaling_factor=None,
        routing_method_type=1,
        topk_weights=weights.to(torch.bfloat16).contiguous(),
        topk_ids=ids.to(torch.int32).contiguous(),
        tune_max_num_tokens=spec["tune_max_num_tokens"],
    )


#: The arguments that change every step; the rest is weights and configuration.
_PER_STEP = ("hidden_states", "hidden_states_scale", "topk_weights", "topk_ids")


def _prefix(a: dict, m: int) -> dict:
    """The same call on its first `m` tokens, every operand contiguous."""
    return dict(
        a,
        hidden_states=a["hidden_states"][:m],
        hidden_states_scale=a["hidden_states_scale"][:, :m].contiguous(),
        topk_weights=a["topk_weights"][:m],
        topk_ids=a["topk_ids"][:m],
    )


def _assert_window(out: torch.Tensor, a: dict) -> None:
    """A token with no slot in the window is bitwise zero; some token has one."""
    local = a["topk_ids"].long() - a["local_expert_offset"]
    live = ((local >= 0) & (local < a["local_num_experts"])).any(dim=-1)
    assert bool(live.any()), "no token has a slot in the window: the call checks nothing"
    assert bool((out[~live] == 0).all()), "a token with no slot in the window must be zero"


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


def _replay_tep4(spec, tokens: int, seed: int) -> None:
    """Capture one call, replay it on new tokens and new routing: bitwise the
    eager call on that data, which is gated against `reference`."""
    static = _tep4_args(spec, tokens, seed)
    fresh = _tep4_args(spec, tokens, seed + 7919)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(**static)  # the runner, its tactic and the kernels' modules, before capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(**static)
    for name in _PER_STEP:
        static[name].copy_(fresh[name])
    graph.replay()
    eager = op(**fresh)
    assert torch.equal(captured, eager), f"T={tokens}: graph replay differs from eager"
    ref = op.reference(**fresh)
    elt, rms = _ulp(eager, ref)
    print(f"  graph T={tokens}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    _compare(eager, ref)
    _assert_window(eager, fresh)


def _sweep_tep4(spec, seed: int) -> None:
    """The largest count gated against `reference`; every smaller count bitwise
    the first rows of that call, so each is gated too."""
    counts = spec["sweep"] if "sweep" in spec else (spec["tokens"],)
    top = max(counts)
    a = _tep4_args(spec, top, seed, far=spec.get("far", False))
    full = op(**a)
    assert full.shape == a["hidden_states"].shape and full.dtype == torch.bfloat16
    ref = op.reference(**a)
    elt, rms = _ulp(full, ref)
    print(f"  T={top}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS; {len(counts)} counts")
    _compare(full, ref)
    _assert_window(full, a)
    differ = [m for m in counts if m < top and not torch.equal(op(**_prefix(a, m)), full[:m])]
    assert not differ, f"token counts whose rows differ from the {top}-token call: {differ}"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    seed = _seed(cell)
    if "graph" in cell.spec:
        for tokens in cell.spec["graph"]:
            _replay_tep4(cell.spec, tokens, seed + tokens)
    else:
        _sweep_tep4(cell.spec, seed)


# ── the two internal requantizations, read out bit-exactly ────────────────


def _pow2_requantize(x: torch.Tensor) -> torch.Tensor:
    """The MX-style round-up power-of-two block scale -- a wrong recipe here."""
    shp = x.shape
    xb = x.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    coded = (xb / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (coded.float() * scale).reshape(shp)


@pytest.mark.parametrize("tokens,width", [(16, 256), (64, 640), (256, 128), (16, 6144)])
def test_intermediate_is_fp8_block_quantized(tokens, width):
    """Identity down projection, one slot of weight 1.0: the output is the
    requantized intermediate, one bf16 round away. The recipe reproduces
    >=99.9% of it bit-exactly (measured 99.95-100%; 99.987% at the tep4 width
    6144); every neighbouring recipe reproduces under 10%. The misses are FC1
    sums on an e4m3 rounding boundary.
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
    """Every operand-level mistake, measured against one 1024-token call."""
    a = _tep4_args(TEP4, 1024, seed=77)
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
    a = _tep4_args(TEP4, 48, seed=99)
    assert a["hidden_states_scale"].shape == (48, 48)
    _compare(op(**a), op.reference(**a))
    swapped = op(**dict(a, hidden_states_scale=a["hidden_states_scale"].t().contiguous()))
    elt, rms = _ulp(swapped, op.reference(**a))
    print(f"  transposed: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    assert rms > 10.0, "the transposed layout must be visibly wrong"


# ── the tep4 path: kernels, tactics and the far experts ───────────────────

_BLOCK_ROUTING, _DYN_ROUTING = "routingIndicesBlockKernel", "routingIndicesDynBlockKernel"
_CLUSTER_ROUTING, _COOP_ROUTING = "routingIndicesClusterKernel", "routingIndicesCoopKernel"
_EXPANDED_ACT, _PERMUTED_ACT = "activationDeepSeekKernel<", "activationDeepSeekPermutedWideKernel<"
_SCALAR_FIN, _VECTOR_FIN = "finalizeKernel<", "finalizeKernelVecLoad<"

#: (tokens, routing kernel, GEMM tile, activation kernel, finalize kernel) the
#: tep4 call launches on the cold path, at each side of every switch the
#: sweep cell's token counts claim to straddle.
_TEP4_KERNELS = (
    (4, _BLOCK_ROUTING, "t128x32x128", _EXPANDED_ACT, _SCALAR_FIN),
    (5, _DYN_ROUTING, "t128x32x128", _EXPANDED_ACT, _SCALAR_FIN),
    (12, _DYN_ROUTING, "t128x32x128", _EXPANDED_ACT, _SCALAR_FIN),
    (13, _CLUSTER_ROUTING, "t128x32x128", _EXPANDED_ACT, _SCALAR_FIN),
    (49, _CLUSTER_ROUTING, "t128x32x128", _EXPANDED_ACT, _SCALAR_FIN),
    (50, _CLUSTER_ROUTING, "t128x32x128", _EXPANDED_ACT, _VECTOR_FIN),
    (71, _CLUSTER_ROUTING, "t128x32x128", _EXPANDED_ACT, _VECTOR_FIN),
    (72, _CLUSTER_ROUTING, "t128x64x128", _EXPANDED_ACT, _VECTOR_FIN),
    (135, _CLUSTER_ROUTING, "t128x64x128", _EXPANDED_ACT, _VECTOR_FIN),
    (136, _CLUSTER_ROUTING, "t128x128x128", _EXPANDED_ACT, _VECTOR_FIN),
    (4095, _CLUSTER_ROUTING, "t128x128x128", _EXPANDED_ACT, _VECTOR_FIN),
    (4096, _CLUSTER_ROUTING, "t128x128x128", _PERMUTED_ACT, _VECTOR_FIN),
    (8192, _CLUSTER_ROUTING, "t128x128x128", _PERMUTED_ACT, _VECTOR_FIN),
    (8193, _COOP_ROUTING, "t128x128x128", _PERMUTED_ACT, _VECTOR_FIN),
)


def test_tep4_kernels_by_token_count():
    """The switches the sweep cell straddles are where it says: read back from
    the kernels each call launches, on the cold path the cells run."""
    from torch.profiler import ProfilerActivity, profile

    a = _tep4_args(TEP4, 8193, seed=5)
    for tokens, routing, tile, activation, finalize in _TEP4_KERNELS:
        call = _prefix(a, tokens)
        op(**call)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(**call)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
        gemms = [n for n in names if n.startswith("bmm_")]
        assert len(gemms) == 2 and all(f"_{tile}_" in n for n in gemms), (tokens, gemms)
        for want in (routing, activation, finalize):
            assert any(want in n for n in names), (tokens, want, names)


@pytest.mark.parametrize("tokens", [16, 9216])
def test_tep4_every_valid_tactic_is_bitwise_the_fallback(tokens):
    """What makes the cold cells certify the tuned path: whichever tactic the
    autotuner keeps, the bits are the cold fallback's (48 tactics at 16
    tokens, 20 at 9216, each run through the op's own tunable runner)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP8BlockScaleMoERunner

    a = _tep4_args(TEP4, tokens, seed=tokens)
    base = op(**a)
    _compare(base, op.reference(**a))
    runner = FP8BlockScaleMoERunner(
        a["num_experts"],
        a["top_k"],
        None,
        None,
        None,
        a["intermediate_size"],
        a["local_expert_offset"],
        a["local_num_experts"],
        None,
        a["routing_method_type"],
        0,
        tune_max_num_tokens=a["tune_max_num_tokens"],
    )
    inputs = [
        None,
        None,
        a["hidden_states"],
        a["hidden_states_scale"],
        a["gemm1_weights"],
        a["gemm1_weights_scale"],
        a["gemm2_weights"],
        a["gemm2_weights_scale"],
        a["topk_weights"],
        a["topk_ids"],
    ]
    valid = runner.get_runner().get_valid_configs(
        a["top_k"],
        None,
        a["hidden_states"].shape[1],
        a["intermediate_size"],
        a["local_num_experts"],
        tokens,
    )
    differ = [tuple(t) for t in valid if not torch.equal(runner.forward(inputs, list(t)), base)]
    print(f"  T={tokens}: {len(valid)} valid tactics; tiles {sorted({t[0] for t in valid})}")
    assert len({t[0] for t in valid}) > 1, "one tile only -- the comparison would be vacuous"
    assert not differ, f"tactics that change the result: {differ}"


def test_tep4_experts_past_2_31_bytes_give_the_same_bits_below_it():
    """The gemm1 rows of local experts 29..31 start past 2^31 bytes; moved to
    slot 1, each expert's tokens come back bitwise the same."""
    spec = dict(TEP4, local_expert_offset=0)
    a = _tep4_args(spec, 64, seed=31, far=True)
    out = op(**a)
    _compare(out, op.reference(**a))
    per_expert = a["gemm1_weights"][0].numel() * a["gemm1_weights"].element_size()
    assert 28 * per_expert < 2**31 < 29 * per_expert, "experts 29..31 must lie past 2^31 bytes"
    ids, offset = a["topk_ids"], a["local_expert_offset"]
    for far in (29, 30, 31):
        order = list(range(a["local_num_experts"]))
        order[1], order[far] = far, 1
        index = torch.tensor(order, device=DEV)
        near_id, far_id = offset + 1, offset + far
        swapped = torch.where(ids == far_id, near_id, torch.where(ids == near_id, far_id, ids))
        moved = dict(
            a,
            gemm1_weights=a["gemm1_weights"].index_select(0, index),
            gemm1_weights_scale=a["gemm1_weights_scale"].index_select(0, index),
            gemm2_weights=a["gemm2_weights"].index_select(0, index),
            gemm2_weights_scale=a["gemm2_weights_scale"].index_select(0, index),
            topk_ids=swapped.to(torch.int32).contiguous(),
        )
        assert bool((ids == far_id).any()), f"no token reaches expert {far}"
        assert torch.equal(op(**moved), out), f"expert {far} differs from itself at slot 1"
        del moved
        torch.cuda.empty_cache()


# ── expert parallelism and the window ─────────────────────────────────────


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
