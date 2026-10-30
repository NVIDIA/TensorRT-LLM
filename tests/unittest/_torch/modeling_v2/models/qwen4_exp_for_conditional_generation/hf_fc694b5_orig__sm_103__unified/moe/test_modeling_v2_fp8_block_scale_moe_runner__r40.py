# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4 unified target's fp8_block_scale_moe_runner variant
(ops/moe/fp8_block_scale_moe_runner__r40), copied from the tp1 target's test.

The entry owns the contract: `CELLS`, `reference`, `is_valid`. This
file turns a cell into operands and holds the claims no cell carries: a warm
autotuned tactic is bitwise the cold fallback the cells run (whole stack and
window), the unified target's four windows sum to the 512-expert layer, a
window call refuses the whole stack's tensors loudly, and the two guards
(contiguity, a window outside the routing space).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.moe.fp8_block_scale_moe_runner__r40 import (  # noqa: E501
    BLOCK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.moe.fp8_block_scale_moe_runner__r40 import (  # noqa: E501
    fp8_block_scale_moe_runner as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

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
    0.342. The unified target's 128-expert window cells on sm_103: 1.29 / 0.188
    (T=1), 3.93 / 0.212 (128), 4.40 / 0.221 (1024), 7.95 / 0.227 (8192); the
    four windows summed in fp32, at most 4.41 / 0.297. Wrong references at the checkpoint geometry, RMS: gate/up swap
    65, transposed activation scales 43, strided hidden_states 94, no
    FC1-output requant 3.5, neither requant 3.1, no activation requant 1.58
    -- the closest.
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
    tokens, hidden, inter, num_experts, top_k, local_expert_offset=0, local_num_experts=None, seed=0
):
    """The op's arguments by name, pre-routed over all `num_experts`.

    The call answers for `[local_expert_offset, local_expert_offset +
    local_num_experts)` and its weights are that window's slice of the one
    stack -- what a rank of the unified target holds. The routing, activations
    and stack depend on `seed` and the geometry only, so the windows of one
    seed split one layer.
    """
    local = num_experts if local_num_experts is None else local_num_experts
    window = slice(local_expert_offset, local_expert_offset + local)
    w1, w1s, w2, w2s = _weights(hidden, inter, num_experts, seed=hidden + inter + num_experts)
    w1, w1s, w2, w2s = w1[window], w1s[window], w2[window], w2s[window]
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
        local_expert_offset=local_expert_offset,
        local_num_experts=local,
        routed_scaling_factor=None,
        routing_method_type=1,
        topk_weights=weights.to(torch.bfloat16).contiguous(),
        topk_ids=ids.contiguous(),
        gemm1_clamp_limit=None,
    )


def _ulp(out: torch.Tensor, ref: torch.Tensor):
    """(max, RMS) deviation in bf16 ulp of the row max -- `_compare`'s statistics."""
    o, r = out.float(), ref.float()
    d = (o - r) / (r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) * ULP)
    return d.abs().max().item(), d.pow(2).mean().sqrt().item()


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode())


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _args(**cell.spec, seed=_seed(cell))
    out = op(**args)
    assert out.shape == args["hidden_states"].shape and out.dtype == torch.bfloat16
    # A window cell certifies the window only if some slot routes into it; a
    # row that routes none comes back bitwise zero.
    hit = _in_window(args)
    assert hit.any(), f"no slot routes into the window: {cell.why!r} would certify nothing"
    assert torch.equal(out[~hit.any(-1)], torch.zeros_like(out[~hit.any(-1)]))
    ref = op.reference(**args)
    elt, rms = _ulp(out, ref)
    print(
        f"  {cell.why[:44]}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS; "
        f"{hit.sum().item()} of {hit.numel()} slots in the window"
    )
    _compare(out, ref)


def _in_window(args) -> torch.Tensor:
    """`[T, top_k]`: which slots the call's window answers for."""
    local = args["topk_ids"].long() - args["local_expert_offset"]
    return (local >= 0) & (local < args["local_num_experts"])


# ── the unified target's windows ──────────────────────────────────────────

#: The qwen4 unified target's split: four 128-expert windows of the 512, one per rank.
WINDOWS = (0, 128, 256, 384)


@pytest.mark.parametrize("tokens", [1, 128, 1024])
def test_windows_sum_to_the_whole_stack(tokens):
    """The four windows' bf16 outputs, added in fp32 (what the target's
    all-reduce or reduce-scatter does), sit inside the gate against the
    512-expert reference -- and the sum with any one window left out does not.

    Not bitwise to the 512-expert call: each window rounds its own partial to
    bf16. The distance to it is printed.
    """
    seed = 900 + tokens
    whole = _args(tokens, **CKPT, seed=seed)
    parts = [
        op(**_args(tokens, **CKPT, local_expert_offset=o, local_num_experts=128, seed=seed))
        for o in WINDOWS
    ]
    total = torch.stack([p.float() for p in parts]).sum(0).to(torch.bfloat16)
    ref = op.reference(**whole)
    elt, rms = _ulp(total, ref)
    elt_call, rms_call = _ulp(total, op(**whole))
    print(
        f"  T={tokens}: windows summed vs reference {elt:.2f} / {rms:.3f} ulp; "
        f"vs the 512-expert call {elt_call:.2f} / {rms_call:.3f} ulp"
    )
    _compare(total, ref)

    # Negative control: drop each window that holds a slot.
    ids = whole["topk_ids"].long()
    for i, o in enumerate(WINDOWS):
        if not ((ids >= o) & (ids < o + 128)).any():
            continue
        short = torch.stack([p.float() for j, p in enumerate(parts) if j != i]).sum(0)
        with pytest.raises(AssertionError):
            _compare(short.to(torch.bfloat16), ref)


@pytest.mark.parametrize(
    "name", ["gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale"]
)
def test_window_call_refuses_the_whole_stack_loudly(name):
    """A window call handed the whole stack's tensor raises: unlike the NVFP4
    runner, this op compares every weight and scale's expert dim with
    `local_num_experts`, so `is_valid` need not."""
    a = _args(16, **CKPT, local_expert_offset=128, local_num_experts=128, seed=5)
    stack = dict(
        zip(
            ("gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale"),
            _weights(2560, 640, 512, seed=2560 + 640 + 512),
        )
    )
    with pytest.raises(RuntimeError, match=f"{name} has incorrect"):
        op(**dict(a, **{name: stack[name]}))


def test_guard_refuses_a_window_outside_the_routing_space():
    """`num_experts` passed as the window's own count: the raw op answers all
    zeros with no error, and `is_valid` refuses it.

    Probed at one token only. At 16 rows the same call reads out of bounds (an
    illegal memory access that would end this process), recorded in the note.
    """
    a = _args(1, **CKPT, local_expert_offset=128, local_num_experts=128, seed=3)
    assert _in_window(a).any(), "the probe needs a slot in the window"
    wrong = dict(a, num_experts=128)
    unguarded = op(**wrong)
    assert torch.count_nonzero(unguarded) == 0, "the hazard this guard covers is gone"
    assert torch.count_nonzero(op.reference(**a)) > 0
    with validating(op), pytest.raises(AssertionError, match="outside the routing space"):
        op(**wrong)


# ── what a cell cannot carry ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "offset, local", [(0, 512), (256, 128)], ids=["whole stack", "a 128-expert window"]
)
def test_autotuned_tactic_matches_the_fallback_tactic(offset, local):
    """A warm profiling cache changes the tactic, not the result -- for the
    whole stack and for a window, which tunes under its own key
    (`local_num_experts`) and its own config (`num_experts // local_num_experts`)."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    a = _args(256, **CKPT, local_expert_offset=offset, local_num_experts=local, seed=42)
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


# ── the guards ────────────────────────────────────────────────────────────


def _strided(t: torch.Tensor) -> torch.Tensor:
    wide = torch.zeros(*t.shape[:-1], 2 * t.shape[-1], device=DEV, dtype=t.dtype)
    wide[..., ::2] = t
    view = wide[..., ::2]
    assert not view.is_contiguous()
    return view


def test_guard_refuses_a_strided_hidden_states():
    """What `is_valid` exists for: the raw op reads a strided operand as dense."""
    a = _args(16, 256, 128, 8, 2, seed=11)
    with validating(op), pytest.raises(AssertionError, match="must be contiguous"):
        op(**dict(a, hidden_states=_strided(a["hidden_states"])))
