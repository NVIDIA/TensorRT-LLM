# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 gdn_decode_t1 entry.

Each cell against the entry's float64 reference from the same dequantized states:

* `o` in torch's default bf16 band: both compute `o` from the unrounded state, so they
  differ by the fp32 sums' reorder (~2^-20 of `|h| |q|`) under one bf16 rounding.
* the pool, element by element, within two int16 roundings (each side's own,
  `scale / 2`) plus 1e-6 of fp32 reorder; scales to 1e-5 relative. This is the
  derivation of the quantized pool's acceptance check, applied to one step.
* every slot the call did not name -- the pad row's included -- bitwise unchanged; with
  `out` given, the step output's other rows too.

Then the properties: CUDA-graph replay of three chained steps equal to eager and to
itself, and the guards (shared slots, a bf16 pool).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_decode_t1 import (  # noqa: E501
    POOL_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_decode_t1 import (  # noqa: E501
    gdn_decode_t1 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _gdn_checks import (  # noqa: E402
    DEV,
    HV,
    V,
    assert_q8_states_close,
    assert_untouched,
    coalesced_slots,
    decode_inputs,
    q8_pool,
    scattered_slots,
)
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

#: Context rows ahead of the riders in the mixed-step cell's output.
_PREFILL_ROWS = 6144


def _build(spec, seed):
    """`(args, extras)`: `__call__`'s positional arguments and the buffers they view."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["rows"]
    if spec.get("pool") == "coalesced":
        slots = coalesced_slots(rows)
        backing, pool = q8_pool(102, "coalesced", slots + [0, 101], gen)
    else:
        slots = scattered_slots(rows, POOL_SLOTS, gen, pad=spec.get("pad", False))
        backing, pool = q8_pool(POOL_SLOTS, "compact", [s for s in slots if s >= 0] + [0], gen)
    inp = decode_inputs(rows, gen)
    step_out = None
    out = None
    if spec.get("out") == "row_range":
        step_out = torch.full(
            (1, _PREFILL_ROWS + rows, HV, V), -7.0, dtype=torch.bfloat16, device=DEV
        )
        out = step_out[0, _PREFILL_ROWS:]
    slot_t = torch.tensor(slots, dtype=torch.int32, device=DEV)
    args = [inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], inp["A_log"], inp["dt_bias"],
            pool, slot_t, out]  # fmt: skip
    return args, dict(backing=backing, step_out=step_out, slots=slots)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    pool, out = args[7], args[9]
    pool_before = pool.clone()
    step_before = None if extras["step_out"] is None else extras["step_out"].clone()
    ref_o, ref_pool = op.reference(*args)

    o = op(*args)

    rows = cell.spec["rows"]
    assert o.shape == (rows, HV, V) and o.dtype == torch.bfloat16 and o.is_contiguous()
    if out is not None:
        assert o is out, "out= is written in place and returned"
        assert torch.equal(extras["step_out"][0, :_PREFILL_ROWS], step_before[0, :_PREFILL_ROWS]), (
            "rows of the step output outside out were written"
        )
    torch.testing.assert_close(o, ref_o)
    assert_q8_states_close(pool, ref_pool, extras["slots"])
    assert_untouched(pool, pool_before, extras["slots"])
    if cell.spec.get("pool") == "coalesced":
        assert not extras["backing"][:, 1:].any(), "another layer's states were written"
    for s in extras["slots"]:
        if s >= 0:  # non-vacuity: every named slot moved
            assert not torch.equal(pool[s], pool_before[s]), s


def test_cuda_graph_replay_of_chained_steps() -> None:
    """Three chained 4-row steps captured in one graph: every replay from the same pool
    equals the eager chain bitwise (outputs and pool), and replays equal each other."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    rows = 4
    slots = scattered_slots(rows, 16, gen)
    _, pool = q8_pool(16, "compact", slots, gen)
    pristine = pool.clone()
    steps = [decode_inputs(rows, gen) for _ in range(3)]
    slot_t = torch.tensor(slots, dtype=torch.int32, device=DEV)

    def chain():
        return [
            op(s["q"], s["k"], s["v"], s["a"], s["b"], s["A_log"], s["dt_bias"], pool, slot_t, None)
            for s in steps
        ]

    eager = [o.clone() for o in chain()]
    eager_pool = pool.clone()
    pool.copy_(pristine)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        chain()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    for replay in range(3):
        pool.copy_(pristine)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, eager_pool), f"pool after replay {replay}"
        for i, (got, want) in enumerate(zip(outs, eager)):
            assert torch.equal(got, want), f"step {i} output after replay {replay}"


def test_guard_refuses_shared_slots() -> None:
    """Two rows naming one slot race on its state (each program requantizes its rows from
    its own update); the guard refuses it."""
    args, _ = _build(dict(rows=3), seed=7)
    args[8] = torch.tensor([5, 9, 5], dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="two rows naming one slot"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_bf16_pool() -> None:
    """A bf16 `[S, 48, 128, 128]` pool would be read as int8 codes and scales; refused."""
    args, _ = _build(dict(rows=2), seed=8)
    args[7] = torch.zeros(8, HV, V, 128, dtype=torch.bfloat16, device=DEV)
    args[8] = torch.tensor([1, 2], dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="int8 pool"):
        with validating(op):
            op(*args)
