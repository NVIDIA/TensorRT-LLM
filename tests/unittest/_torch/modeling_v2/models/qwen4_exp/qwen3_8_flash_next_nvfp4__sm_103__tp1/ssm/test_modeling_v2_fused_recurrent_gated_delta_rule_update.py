# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_recurrent_gated_delta_rule_update entry.

The axis this entry exists for is more than one token per sequence per step --
the speculative verify call -- so the cells sweep `P`, and the two state sinks
(write-back into the pool, per-step states into the rollback buffer) are
driven separately and together.

Two tests answer "is this harness blind" rather than a correctness question:
the neighbouring `chunk_gated_delta_rule` is wrong on the first indexed
in-place call of a process (an autotuner trial launch consumes the state), so
`test_cold_call_in_a_fresh_process_is_correct` measures the first call of a
process that has made none, and `test_the_gate_rejects_a_doubly_applied_state`
shows `_compare` rejects exactly the damage that mechanism leaves.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch.modules.fla.fused_recurrent as _impl
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_recurrent_gated_delta_rule_update import (  # noqa: E501
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_recurrent_gated_delta_rule_update import (  # noqa: E501
    fused_recurrent_gated_delta_rule_update as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_recurrent_gated_delta_rule_update requires a CUDA device"


def _compare(out: tuple, ref: tuple) -> None:
    """`(returned, initial_state_source, intermediate_states_buffer)` as the
    call left them, against `reference`'s triple, each at torch's default
    dtype-aware band -- nothing is loosened.

    The kernel and the reference both accumulate in fp32 and differ only in
    reduction order. Measured over every cell shape: output <= 2.27e-4 at a
    bf16 absmax of 3.9e-2 .. 8.2e-2 (under one bf16 ulp there), written-back
    state <= 2.98e-7 at absmax ~1, per-step states <= 1.8e-7; a bf16
    per-step buffer costs one bf16 ulp (1.94e-3 at absmax 0.89). Eight wrong
    recurrences (q/k heads tiled not interleaved, q/k swapped, beta 1, no
    gating, reversed tokens, every token from the initial state, a
    transposed or absent initial state) land 1.56e-2 .. 8.17e-2 away on the
    output -- 128x the correct result's own deviation at the closest.
    """
    (o, pool, cache), (o_ref, pool_ref, cache_ref) = out, ref
    assert o.dtype == o_ref.dtype and o.shape == o_ref.shape, (
        (o.dtype, o.shape),
        (o_ref.dtype, o_ref.shape),
    )
    torch.testing.assert_close(o, o_ref)
    assert (pool is None) == (pool_ref is None)
    if pool is not None:
        torch.testing.assert_close(pool, pool_ref)
    assert (cache is None) == (cache_ref is None)
    if cache is not None:
        torch.testing.assert_close(cache, cache_ref)


DEV = "cuda"
REPEAT = NUM_V_HEADS // NUM_K_HEADS  # 3
MIXED_QKV_WIDTH = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240
CACHE_SENTINEL = -3.0


def _build(spec, seed):
    """`__call__`'s arguments by name for one spec, plus what the locks need.

    q/k/v are views over one packed `[T, 10240]` row block (or contiguous
    copies, as the target passes them); `g` is built the way the gating step
    builds it -- negative, so `exp(g) < 1` -- and `beta` is a sigmoid.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    mode, n = spec["mode"], spec["n"]
    lengths = spec.get("lengths")
    batch, tokens = (n, spec["p"]) if lengths is None else (1, sum(lengths))
    total = batch * tokens

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, device=DEV, dtype=dtype)

    mixed = randn(total, MIXED_QKV_WIDTH, dtype=torch.bfloat16)
    kd = NUM_K_HEADS * HEAD_K_DIM
    q = mixed[:, :kd].view(batch, tokens, NUM_K_HEADS, HEAD_K_DIM)
    k = mixed[:, kd : 2 * kd].view(batch, tokens, NUM_K_HEADS, HEAD_K_DIM)
    v = mixed[:, 2 * kd :].view(batch, tokens, NUM_V_HEADS, HEAD_V_DIM)
    if spec.get("qkv") == "contiguous":
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    a, a_log, dt_bias = randn(total, NUM_V_HEADS), randn(NUM_V_HEADS), randn(NUM_V_HEADS)
    g = -torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias)
    g = g.view(batch, tokens, NUM_V_HEADS)
    b = randn(total, NUM_V_HEADS, dtype=torch.bfloat16)
    beta_kind = spec.get("beta", "fp32")
    beta = (b.sigmoid() if beta_kind == "bf16" else b.float().sigmoid()).view(
        batch, tokens, NUM_V_HEADS
    )
    if beta_kind == "channel":
        beta = beta[..., None].expand(batch, tokens, NUM_V_HEADS, HEAD_V_DIM).contiguous()

    pool_slots = spec.get("pool_slots", SLOTS)
    pool = layered = None
    if pool_slots is not None:
        pool = randn(pool_slots, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM) * 0.05
        pool = pool.to(spec.get("pool_dtype", torch.float32))
        if spec.get("pool_view") == "layer_major":
            layered = torch.zeros((3,) + pool.shape, dtype=pool.dtype, device=DEV)
            layered[1].copy_(pool)
            pool = layered[1]
            assert pool.is_contiguous()

    slots = spec.get("slots") or tuple(range(n))
    indices = torch.tensor(slots, dtype=spec.get("index_dtype", torch.int32), device=DEV)
    cu_seqlens = None
    if lengths is not None:
        bounds = torch.tensor((0,) + tuple(lengths)).cumsum(0)
        cu_seqlens = bounds.to(device=DEV, dtype=spec.get("cu_dtype", torch.int64))

    buffer = cache_steps = None
    if mode in ("verify", "both"):
        rows = spec.get("cache_rows", SLOTS if pool_slots is None else pool_slots)
        cache_steps = tokens
        buffer = torch.full(
            (rows, cache_steps, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM),
            CACHE_SENTINEL,
            dtype=spec.get("cache_dtype", torch.float32),
            device=DEV,
        )

    output = out_buf = None
    if spec.get("output") == "prefix":
        # A length-prefix slice of a longer row-major buffer, as the stock mixer
        # passes it: contiguous, and the tail is not offered.
        out_buf = torch.full(
            (total + 5, NUM_V_HEADS, HEAD_V_DIM), -7.0, dtype=torch.bfloat16, device=DEV
        )
        output = out_buf[:total].view(batch, tokens, NUM_V_HEADS, HEAD_V_DIM)
    elif spec.get("output") == "fp32":
        output = torch.zeros(batch, tokens, NUM_V_HEADS, HEAD_V_DIM, device=DEV)

    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_source=pool,
        initial_state_indices=indices,
        scale=None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        disable_state_update=mode == "verify",
        disable_output_calculation=False,
        intermediate_states_buffer=buffer,
        cache_steps=cache_steps,
        output=output,
    )
    return kwargs, dict(
        layered=layered, out_buf=out_buf, total=total, live=[s for s in slots if s >= 0]
    )


def _seed(why):
    return zlib.crc32(why.encode())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    kw, ctx = _build(cell.spec, seed=_seed(cell.why))
    pool, buffer, live = kw["initial_state_source"], kw["intermediate_states_buffer"], ctx["live"]
    pool_before = None if pool is None else pool.clone()
    buffer_before = None if buffer is None else buffer.clone()
    layered_before = None if ctx["layered"] is None else ctx["layered"].clone()
    tail_before = None if ctx["out_buf"] is None else ctx["out_buf"][ctx["total"] :].clone()

    ref = op.reference(**kw)
    got = op(**kw)
    _compare((got, pool, buffer), ref)

    # The in-place locks, bitwise, independent of the reference.
    if kw["output"] is not None:
        assert got.data_ptr() == kw["output"].data_ptr(), "output must be written in place"
    if tail_before is not None:
        assert torch.equal(ctx["out_buf"][ctx["total"] :], tail_before), "buffer tail touched"
    if pool is not None:
        if kw["disable_state_update"]:
            assert torch.equal(pool, pool_before), "the pool must not move on a verify call"
        else:
            for s in live:
                assert not torch.equal(pool[s], pool_before[s]), f"slot {s} was not written"
            pool_before[live] = pool[live]
            assert torch.equal(pool, pool_before), "a slot outside the indices was written"
    if buffer is not None:
        for s in live:
            assert not (buffer[s] == CACHE_SENTINEL).any(), f"row {s} not fully cached"
        buffer_before[live] = buffer[live]
        assert torch.equal(buffer, buffer_before), "a buffer row outside the indices was written"
    if layered_before is not None:
        layered = ctx["layered"]
        assert torch.equal(layered[0], layered_before[0]) and torch.equal(
            layered[2], layered_before[2]
        ), "the neighbouring layers must be untouched"
    if cell.spec["mode"] == "both":
        # Independent of the reference: the last cached step is the stored state.
        assert torch.equal(buffer[live, cell.spec["p"] - 1], pool[live])


# ── the mechanism the neighbouring entry fails on ─────────────────────────


def test_the_wrapped_kernel_carries_no_autotuner() -> None:
    """Why the cold call is safe here: `triton.heuristics` + `triton.jit`, so
    there is no trial launch to consume the in-place state."""
    kernel = _impl.fused_recurrent_gated_delta_rule_update_fwd_kernel
    names = [cls.__name__ for cls in type(kernel).__mro__]
    assert "Autotuner" not in names, f"the wrapped kernel gained an autotuner ({names})"
    assert not hasattr(kernel, "configs"), "an autotuner config list appeared on the wrapped kernel"


_COLD_SPEC = dict(mode="write_back", n=2, p=3, slots=(5, 2))


def cold_call_measurement() -> dict:
    """The first indexed in-place call of the process, then the second.

    Public because the child process below calls it by name.
    """
    kw, _ = _build(_COLD_SPEC, seed=7)
    pool = kw["initial_state_source"]
    pool_before = pool.clone()
    ref = op.reference(**kw)
    cold_out = op(**kw).clone()
    cold_pool = pool.clone()
    try:
        _compare((cold_out, cold_pool, None), ref)
        cold_ok = True
    except AssertionError:
        cold_ok = False
    pool.copy_(pool_before)
    warm_out = op(**kw)
    return {
        "cold_compare_ok": cold_ok,
        "cold_out_err": (cold_out.float() - ref[0].float()).abs().max().item(),
        "cold_pool_err": (cold_pool - ref[1]).abs().max().item(),
        "out_bitwise_equal": bool(torch.equal(cold_out, warm_out)),
        "pool_bitwise_equal": bool(torch.equal(cold_pool, pool)),
    }


_COLD_CHILD = """
import importlib.util, json, sys
sys.path.insert(0, {root!r})
spec = importlib.util.spec_from_file_location("_cold_entry_test", {path!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["_cold_entry_test"] = mod
spec.loader.exec_module(mod)
print("COLD_RESULT " + json.dumps(mod.cold_call_measurement()))
"""


def test_cold_call_in_a_fresh_process_is_correct() -> None:
    """Measured in a child that has launched nothing: a warm-up here would
    hide exactly the defect the sibling prefill core has. Cold must pass the
    entry's own gate *and* be bitwise the warm call."""
    root = str(Path(__file__).resolve().parents[4])  # where `_validating` lives
    proc = subprocess.run(
        [sys.executable, "-c", _COLD_CHILD.format(root=root, path=__file__)],
        capture_output=True,
        text=True,
        timeout=480,
        check=False,
    )
    assert proc.returncode == 0, (
        f"cold-call child exited {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("COLD_RESULT ")]
    assert line, f"child produced no result\n{proc.stdout[-4000:]}"
    result = json.loads(line[0][len("COLD_RESULT ") :])
    print(f"\ncold call: {result}")
    assert result["cold_compare_ok"], result
    assert result["out_bitwise_equal"] and result["pool_bitwise_equal"], result


def test_the_gate_rejects_a_doubly_applied_state() -> None:
    """The damage an autotuner trial launch leaves -- the state applied twice --
    is rejected by `_compare`, so a cold-call error of that shape cannot pass."""
    kw, _ = _build(_COLD_SPEC, seed=8)
    pool = kw["initial_state_source"]
    ref = op.reference(**kw)
    _compare((op(**kw), pool, None), ref)  # one application is right

    doubled = op(**kw)
    out_err = (doubled.float() - ref[0].float()).abs().max().item()
    pool_err = (pool - ref[1]).abs().max().item()
    print(f"\ndoubly applied: output {out_err:.3e}, state {pool_err:.3e}")
    assert out_err > 1e-2 and pool_err > 1e-2
    with pytest.raises(AssertionError):
        _compare((doubled, pool, None), ref)


# ── properties beyond a single reference comparison ───────────────────────


def test_varlen_form_matches_the_batched_form_bitwise() -> None:
    """`cu_seqlens` at B = 1 is the same arithmetic as the uniform batch."""
    n, p = 3, 4
    kw, _ = _build(dict(mode="write_back", n=n, p=p, slots=(2, 6, 3)), seed=60)
    pool_before = kw["initial_state_source"].clone()
    batched = op(**kw)
    total = n * p
    flat = dict(
        kw,
        q=kw["q"].reshape(1, total, NUM_K_HEADS, HEAD_K_DIM),
        k=kw["k"].reshape(1, total, NUM_K_HEADS, HEAD_K_DIM),
        v=kw["v"].reshape(1, total, NUM_V_HEADS, HEAD_V_DIM),
        g=kw["g"].reshape(1, total, NUM_V_HEADS),
        beta=kw["beta"].reshape(1, total, NUM_V_HEADS),
        initial_state_source=pool_before.clone(),
        cu_seqlens=torch.arange(0, total + 1, p, device=DEV, dtype=torch.int64),
    )
    varlen = op(**flat)
    assert torch.equal(varlen.reshape(n, p, NUM_V_HEADS, HEAD_V_DIM), batched)
    assert torch.equal(flat["initial_state_source"], kw["initial_state_source"])


def test_scale_default_is_bitwise_head_dim_pow_minus_half() -> None:
    kw, _ = _build(dict(mode="write_back", n=2, p=3, slots=(1, 2)), seed=66)
    pool_a, pool_b = kw["initial_state_source"].clone(), kw["initial_state_source"].clone()
    out_a = op(**dict(kw, initial_state_source=pool_a))
    out_b = op(**dict(kw, initial_state_source=pool_b, scale=HEAD_K_DIM**-0.5))
    assert torch.equal(out_a, out_b) and torch.equal(pool_a, pool_b)


def test_deterministic() -> None:
    kw, _ = _build(dict(mode="both", n=3, p=4), seed=68)
    pool0, buf0 = kw["initial_state_source"].clone(), kw["intermediate_states_buffer"].clone()
    runs = []
    for _ in range(2):
        pool, buf = pool0.clone(), buf0.clone()
        out = op(**dict(kw, initial_state_source=pool, intermediate_states_buffer=buf))
        runs.append((out, pool, buf))
    assert all(torch.equal(x, y) for x, y in zip(*runs))


def test_duplicate_slot_indices_race_on_the_write_back() -> None:
    """Two sequences on one slot: the outputs stay right, the slot does not.
    Nothing raises, so it is a caller obligation recorded as a measurement."""
    kw, _ = _build(dict(mode="write_back", n=2, p=3, slots=(3, 3)), seed=69)
    pool = kw["initial_state_source"]
    pool_before = pool.clone()
    ref = op.reference(**kw)
    got = op(**kw)
    torch.testing.assert_close(got, ref[0])

    def one(i, source):
        rows = slice(i, i + 1)
        return op.reference(
            **dict(
                kw,
                q=kw["q"][rows],
                k=kw["k"][rows],
                v=kw["v"][rows],
                g=kw["g"][rows],
                beta=kw["beta"][rows],
                initial_state_source=source,
                initial_state_indices=kw["initial_state_indices"][rows],
            )
        )[1]

    after_both = one(1, one(0, pool_before))
    drift = (pool[3] - after_both[3]).abs().max().item()
    print(f"\nshared slot drift from a sequential application: {drift:.3e}")
    assert drift > 1e-2


# ── what the guard refuses, and what the op refuses itself ────────────────


def _small(seed, **spec):
    return _build(dict(dict(mode="write_back", n=2, p=3), **spec), seed=seed)[0]


def test_a_strided_output_clobbers_its_neighbour_and_is_refused() -> None:
    """Unguarded, a last-dim slice of a doubly wide buffer is written with
    packed arithmetic: the half never offered is overwritten."""
    kw = _small(70)
    wide = torch.zeros(2, 3, NUM_V_HEADS, 2 * HEAD_V_DIM, dtype=torch.bfloat16, device=DEV)
    strided = wide[..., :HEAD_V_DIM]
    op(**dict(kw, initial_state_source=kw["initial_state_source"].clone(), output=strided))
    assert (wide[..., HEAD_V_DIM:] != 0).any(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="output must be contiguous"):
        with validating(op):
            op(**dict(kw, output=wide[..., :HEAD_V_DIM]))


def test_output_with_the_calculation_off_is_never_written_and_is_refused() -> None:
    kw = _build(dict(mode="verify", n=2, p=3), seed=71)[0]
    buf = torch.full((2, 3, NUM_V_HEADS, HEAD_V_DIM), -7.0, dtype=torch.bfloat16, device=DEV)
    got = op(**dict(kw, disable_output_calculation=True, output=buf))
    assert tuple(got.shape) == (1, 1, 1, 1) and (buf == -7.0).all(), "the hazard is gone"
    with pytest.raises(AssertionError, match="mutually exclusive"):
        with validating(op):
            op(**dict(kw, disable_output_calculation=True, output=buf))


def test_a_strided_pool_loses_the_write_back_and_is_refused() -> None:
    """Unguarded, a `[::2]` slot view is replaced by a contiguous copy: the
    output is right, and the caller's pool comes back bitwise unchanged."""
    kw = _small(72)
    backing = torch.zeros(2 * SLOTS, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device=DEV)
    strided = backing[::2]
    strided.copy_(kw["initial_state_source"])
    before = strided.clone()
    ref = op.reference(**dict(kw, initial_state_source=before))
    got = op(**dict(kw, initial_state_source=strided))
    torch.testing.assert_close(got, ref[0])
    assert torch.equal(strided, before), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="initial_state_source must be contiguous"):
        with validating(op):
            op(**dict(kw, initial_state_source=strided))


def test_a_strided_step_buffer_loses_every_cached_state_and_is_refused() -> None:
    """The same copy-in trap on the per-step buffer: unguarded, the caller's
    buffer comes back bitwise unchanged and nothing raises."""
    kw = _build(dict(mode="verify", n=2, p=3), seed=73)[0]
    backing = torch.full(
        (2 * SLOTS, 3, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM), CACHE_SENTINEL, device=DEV
    )
    strided = backing[::2]
    op(**dict(kw, intermediate_states_buffer=strided))
    assert (backing == CACHE_SENTINEL).all(), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="intermediate_states_buffer must be contiguous"):
        with validating(op):
            op(**dict(kw, intermediate_states_buffer=strided))


def test_a_short_cache_steps_writes_into_the_next_row_and_is_refused() -> None:
    """Unguarded, cache_steps 2 at P 3: row 0's last step lands in row 1, a
    row the call never named."""
    kw = _build(dict(mode="verify", n=1, p=3), seed=74)[0]
    short = torch.full((SLOTS, 2, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM), CACHE_SENTINEL, device=DEV)
    op(**dict(kw, intermediate_states_buffer=short, cache_steps=2))
    assert not (short[1, 0] == CACHE_SENTINEL).all(), "the hazard this guard covers is gone"
    for cache_steps in (2, None):
        with pytest.raises(AssertionError, match="cache_steps must be at least"):
            with validating(op):
                op(**dict(kw, intermediate_states_buffer=short.clone(), cache_steps=cache_steps))


def test_cache_steps_off_the_buffer_step_axis_misplaces_rows_and_is_refused() -> None:
    """Unguarded, a 4-step buffer driven at cache_steps 2: row 1's states land in
    row 0's steps 2..3 and row 1 is never written -- the rows the promotion
    reads are not the ones the kernel filled."""
    kw = _build(dict(mode="verify", n=2, p=2), seed=75)[0]
    deep = torch.full((SLOTS, 4, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM), CACHE_SENTINEL, device=DEV)
    op(**dict(kw, intermediate_states_buffer=deep, cache_steps=2))
    assert (deep[1] == CACHE_SENTINEL).all() and not (deep[0, 2:] == CACHE_SENTINEL).any(), (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="must equal intermediate_states_buffer's step axis"):
        with validating(op):
            op(**dict(kw, intermediate_states_buffer=deep.clone(), cache_steps=2))


def test_op_rejects_what_it_really_rejects() -> None:
    """Three loud refusals; the guard admits each so the op's own error is what
    surfaces."""
    from triton.compiler.errors import CompilationError

    kw = _small(76)
    cu = torch.tensor([0, 3, 6], device=DEV, dtype=torch.int64)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        with validating(op):
            op(**dict(kw, cu_seqlens=cu))

    flat = _build(dict(mode="write_back", n=1, p=6), seed=77)[0]
    with pytest.raises(ValueError, match="number of initial states"):
        with validating(op):
            op(**dict(flat, cu_seqlens=cu))

    with pytest.raises(CompilationError):
        with validating(op):
            op(**dict(kw, initial_state_source=None))


def test_sequence_count_past_the_grid_cap_fails_loudly() -> None:
    """gridDim.z is `N * HV`, capped at 65535: 1365 sequences at HV 48. Above
    it the launch is refused, not truncated, and the context survives."""
    over = 1400
    assert over * NUM_V_HEADS > 65535
    kw = _build(dict(mode="write_back", n=over, p=1, slots=(0,)), seed=78)[0]
    indices = torch.arange(over, device=DEV, dtype=torch.int32) % SLOTS
    with pytest.raises(RuntimeError, match="invalid argument"):
        op(**dict(kw, initial_state_indices=indices, disable_state_update=True))

    kw, _ = _build(dict(mode="verify", n=2, p=2), seed=79)
    ref = op.reference(**kw)
    _compare((op(**kw), kw["initial_state_source"], kw["intermediate_states_buffer"]), ref)


# ── discrimination ────────────────────────────────────────────────────────


def _wrong_variants(kw, pool_before):
    """Plausible wrong recurrences, each expressed as `reference` driven on
    transformed inputs -- so the rival is the same arithmetic, misapplied."""
    base = dict(kw, initial_state_source=pool_before)
    n, p = kw["q"].shape[:2]

    def flip(t):
        return t.flip(1)

    def per_token(t):
        return t.reshape((n * p, 1) + tuple(t.shape[2:]))

    out = {
        "no_gating": op.reference(**dict(base, g=torch.zeros_like(kw["g"])))[0],
        "beta_one": op.reference(**dict(base, beta=torch.ones_like(kw["beta"])))[0],
        "qk_swapped": op.reference(**dict(base, q=kw["k"], k=kw["q"]))[0],
        "no_initial_state": op.reference(
            **dict(base, initial_state_source=torch.zeros_like(pool_before))
        )[0],
        "transposed_pool": op.reference(
            **dict(base, initial_state_source=pool_before.transpose(-1, -2).contiguous())
        )[0],
        # Tiled instead of interleaved: v-head hv reads q/k head hv % H.
        "repeat_not_interleave": op.reference(
            **dict(base, q=kw["q"].repeat(1, 1, REPEAT, 1), k=kw["k"].repeat(1, 1, REPEAT, 1))
        )[0],
        "reversed_tokens": flip(
            op.reference(**dict(base, **{x: flip(kw[x]) for x in ("q", "k", "v", "g", "beta")}))[0]
        ),
        # Every predicted token re-read from the same initial state.
        "independent_tokens": op.reference(
            **dict(
                base,
                **{x: per_token(kw[x]) for x in ("q", "k", "v", "g", "beta")},
                initial_state_indices=kw["initial_state_indices"].repeat_interleave(p),
                disable_state_update=True,
            )
        )[0].reshape(kw["v"].shape),
    }
    return out


def test_default_gate_separates_the_correct_reference_from_wrong_ones() -> None:
    """`_compare` runs at torch's default band, so it is only as good as its
    ability to reject a near miss. Two of these (`reversed_tokens`,
    `independent_tokens`) are wrong only when P > 1 -- this entry's axis."""
    kw, _ = _build(dict(mode="write_back", n=3, p=4, slots=(2, 5, 0)), seed=80)
    pool = kw["initial_state_source"]
    pool_before = pool.clone()
    ref = op.reference(**kw)
    got = op(**kw)
    _compare((got, pool, None), ref)
    own = (got.float() - ref[0].float()).abs().max().item()
    for name, wrong in _wrong_variants(kw, pool_before).items():
        gap = (wrong.float() - ref[0].float()).abs().max().item()
        print(f"{name}: {gap:.3e} from the reference (the kernel: {own:.3e})")
        assert gap > 1e-2, f"{name} is not a distinct reference (gap {gap:.3e})"
        with pytest.raises(AssertionError):
            _compare((got, pool, None), (wrong, ref[1], None))
