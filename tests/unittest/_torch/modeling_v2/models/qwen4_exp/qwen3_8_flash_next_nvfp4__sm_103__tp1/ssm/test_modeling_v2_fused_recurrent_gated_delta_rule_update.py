# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_recurrent_gated_delta_rule_update entry.

The axis this entry exists for is more than one token per sequence per step --
the speculative verify call -- so the cells are that call as the target makes it
(per-step states into the rollback buffer, the pool untouched) at both pool
dtypes. The write-back sink is driven only by the cold-call test below.

The neighbouring `chunk_gated_delta_rule` was once wrong on the first indexed
in-place call of a process (an autotuner trial launch consumes the state), so
`test_cold_call_in_a_fresh_process_is_correct` measures the first call of a
process that has made none.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

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
MIXED_QKV_WIDTH = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240
CACHE_SENTINEL = -3.0


def _build(spec, seed):
    """`__call__`'s arguments by name for one spec, plus the live slots.

    q/k/v are views over one packed `[T, 10240]` row block (or contiguous
    copies, as the target passes them); `g` is built the way the gating step
    builds it -- negative, so `exp(g) < 1` -- and `beta` is a sigmoid.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    mode, n, tokens = spec["mode"], spec["n"], spec["p"]
    total = n * tokens

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen, device=DEV, dtype=dtype)

    mixed = randn(total, MIXED_QKV_WIDTH, dtype=torch.bfloat16)
    kd = NUM_K_HEADS * HEAD_K_DIM
    q = mixed[:, :kd].view(n, tokens, NUM_K_HEADS, HEAD_K_DIM)
    k = mixed[:, kd : 2 * kd].view(n, tokens, NUM_K_HEADS, HEAD_K_DIM)
    v = mixed[:, 2 * kd :].view(n, tokens, NUM_V_HEADS, HEAD_V_DIM)
    if spec.get("qkv") == "contiguous":
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    a, a_log, dt_bias = randn(total, NUM_V_HEADS), randn(NUM_V_HEADS), randn(NUM_V_HEADS)
    g = -torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias)
    g = g.view(n, tokens, NUM_V_HEADS)
    b = randn(total, NUM_V_HEADS, dtype=torch.bfloat16)
    beta = b.sigmoid() if spec.get("beta", "fp32") == "bf16" else b.float().sigmoid()
    beta = beta.view(n, tokens, NUM_V_HEADS)

    pool_slots = spec.get("pool_slots", SLOTS)
    pool = randn(pool_slots, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM) * 0.05
    pool = pool.to(spec.get("pool_dtype", torch.float32))

    slots = spec.get("slots") or tuple(range(n))
    indices = torch.tensor(slots, dtype=torch.int32, device=DEV)

    buffer = cache_steps = None
    if mode == "verify":
        rows = spec.get("cache_rows", pool_slots)
        cache_steps = tokens
        buffer = torch.full(
            (rows, cache_steps, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM),
            CACHE_SENTINEL,
            dtype=spec.get("cache_dtype", torch.float32),
            device=DEV,
        )

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
        cu_seqlens=None,
        disable_state_update=mode == "verify",
        disable_output_calculation=False,
        intermediate_states_buffer=buffer,
        cache_steps=cache_steps,
        output=None,
    )
    return kwargs, list(slots)


def _seed(why):
    return zlib.crc32(why.encode())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    kw, live = _build(cell.spec, seed=_seed(cell.why))
    pool, buffer = kw["initial_state_source"], kw["intermediate_states_buffer"]
    pool_before, buffer_before = pool.clone(), buffer.clone()

    ref = op.reference(**kw)
    got = op(**kw)
    _compare((got, pool, buffer), ref)

    # The in-place locks, bitwise, independent of the reference.
    assert torch.equal(pool, pool_before), "the pool must not move on a verify call"
    for s in live:
        assert not (buffer[s] == CACHE_SENTINEL).any(), f"row {s} not fully cached"
    buffer_before[live] = buffer[live]
    assert torch.equal(buffer, buffer_before), "a buffer row outside the indices was written"


# ── the mechanism the neighbouring entry failed on ────────────────────────


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


# ── what the guard refuses ─────────────────────────────────────────────


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
