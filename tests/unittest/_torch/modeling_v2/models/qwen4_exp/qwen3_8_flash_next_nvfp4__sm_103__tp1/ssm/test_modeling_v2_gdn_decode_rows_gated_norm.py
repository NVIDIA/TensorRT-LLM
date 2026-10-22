# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_decode_rows_gated_norm entry.

Cells against the composition's fp64 reference (the gated delta rule's recurrence,
then the gated norm entry's) in torch's default band for the bf16 output and the pool's
dtype (fp32 or bf16), the pool compared whole so slots the call did not name are
gated too. Beyond the cells: a fixed row bitwise across batch sizes over a bf16 pool, a
stale-read check under graph replay (a producer rewrites the projection inside the graph
right before the op), and the guards. The `gdn_decode_rows` entry (the recurrence alone)
has its own cells below.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_decode_rows_gated_norm import (  # noqa: E501
    gdn_decode_rows as rows_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_decode_rows_gated_norm import (  # noqa: E501
    gdn_decode_rows_gated_norm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_decode_rows_gated_norm requires a CUDA device"

DEV = "cuda"
H, HV, K, V = 16, 48, 128, 128
SLOTS, EPS = 461, 1e-6
WIDTH = 2 * H * K + 2 * HV * V + 2 * HV


def _scattered(n: int, slots: int) -> tuple[int, ...]:
    """`n` distinct slots of a `slots`-row pool, both ends first, then a stride-37 walk
    over the interior (37 is coprime to every pool size here)."""
    rest = tuple((i * 37 + 4) % (slots - 2) + 1 for i in range(max(n - 2, 0)))
    return ((slots - 1, 0) + rest)[:n]


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["rows"]
    state = torch.bfloat16 if spec.get("state") == "bf16" else torch.float32
    proj = torch.randn(rows, WIDTH, generator=g, device=DEV).to(torch.bfloat16)
    if spec.get("pool") == "coalesced":
        # The KV-cache-v2 manager's view: every GDN layer's state of a slot in one page,
        # this layer's state one sub-page of it, so the slot stride is 36 layers' states
        # and slot 100 sits ~2.8G elements past the base (past int32).
        layers, slots = 36, 102
        page = torch.empty(slots * layers * HV * V * K, device=DEV, dtype=state)
        page[: HV * V * K * 4].normal_(0.0, 0.05, generator=g)
        pool = page.as_strided((slots, HV, V, K), (layers * HV * V * K, V * K, K, 1))
        for s in (100, 101):
            pool[s].copy_(torch.randn(HV, V, K, generator=g, device=DEV) * 0.05)
    else:
        pool = (torch.randn(SLOTS, HV, V, K, generator=g, device=DEV) * 0.05).to(state)
    if spec.get("pool") == "coalesced":
        slots = torch.tensor([100, 101][:rows], dtype=torch.int32, device=DEV)
    else:
        slots = torch.tensor(_scattered(rows, SLOTS), dtype=torch.int32, device=DEV)
    if spec.get("pad"):
        slots[1] = -1
    A_log = torch.empty(HV, device=DEV).uniform_(-5.5, 1.0, generator=g)
    dt_bias = torch.randn(HV, generator=g, device=DEV) * 4.0
    w = (1.0 + 0.2 * torch.randn(V, generator=g, device=DEV)).to(torch.bfloat16)
    return [proj, pool, slots, A_log, dt_bias, w, EPS, H, K]


def _compare(out, pool, ref_out, ref_pool) -> None:
    assert out.dtype == torch.bfloat16 and out.shape == ref_out.shape and out.is_contiguous()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(pool, ref_pool)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    ref_out, ref_pool = op.reference(*args)
    out = op(*args)
    _compare(out, args[1], ref_out, ref_pool)
    if cell.spec.get("pad"):
        assert (out[1] == 0).all(), "a pad row's output is zero"


#: Decode batch sizes up to the engine's largest CUDA-graph batch.
_BATCHES = (1, 2, 4, 5, 16, 33, 128, 256)
_FIXED_SLOT = 301


def test_bf16_pool_is_batch_invariant() -> None:
    """The property the bf16 decode step relies on: a row's output and state are bitwise
    independent of the batch it rides in. One fixed row (its projection and slot 301 of a
    bf16 pool) sits first and last in random batches of every size."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    cpu = torch.Generator().manual_seed(3)
    pool0 = (torch.randn(SLOTS, HV, V, K, generator=gen, device=DEV) * 0.05).to(torch.bfloat16)
    row = torch.randn(WIDTH, generator=gen, device=DEV).to(torch.bfloat16)
    rest = _build(dict(rows=1), seed=3)[3:]
    first = None
    for n in _BATCHES:
        for pos in sorted({0, n - 1}):
            g = torch.Generator(device=DEV).manual_seed(1000 * n + pos)
            proj = torch.randn(n, WIDTH, generator=g, device=DEV).to(torch.bfloat16)
            proj[pos] = row
            others = [s for s in torch.randperm(SLOTS, generator=cpu).tolist() if s != _FIXED_SLOT]
            others = others[:n]
            others[pos] = _FIXED_SLOT
            slots = torch.tensor(others, dtype=torch.int32, device=DEV)
            pool = pool0.clone()
            out = op(proj, pool, slots, *rest)
            got = (out[pos].clone(), pool[_FIXED_SLOT].clone())
            first = first or got
            assert torch.equal(got[0], first[0]) and torch.equal(got[1], first[1]), (
                f"the fixed row differs at batch {n}, position {pos}"
            )


def test_reads_after_the_wait_under_graph_replay() -> None:
    """A producer rewrites the projection inside the graph right before the op; with PDL
    a read of it before `griddepcontrol.wait` would see the previous replay's row. The
    pool advances one step per replay, as in decode."""
    args = _build(dict(rows=16), seed=41)
    src = args[0].clone()

    def step():
        args[0].copy_(src * 1.0)
        return op(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        src.copy_(_build(dict(rows=16), seed=100 + trial)[0])
        before = args[1].clone()
        graph.replay()
        torch.cuda.synchronize()
        after = args[1].clone()
        args[1].copy_(before)
        want = op(src.clone(), *args[1:])
        assert torch.equal(graph_out, want), f"replay {trial} output"
        assert torch.equal(after, args[1]), f"replay {trial} pool"


def test_guards() -> None:
    """An fp16 pool, int64 slot indices and a projection of the wrong width are refused."""
    args = _build(dict(rows=4), seed=5)
    with validating(op), pytest.raises(AssertionError):
        op(args[0], args[1].to(torch.float16), *args[2:])
    with validating(op), pytest.raises(AssertionError):
        op(args[0], args[1], args[2].long(), *args[3:])
    with validating(op), pytest.raises(AssertionError):
        op(args[0][:, :-2], *args[1:])


# ---- gdn_decode_rows: the recurrence alone, into a row range of a step's output ----------


def _rows_args(spec, seed):
    """`(args, buf)`: the entry's arguments, `out` rows 5.. of a `[1, rows + 9, HV, V]` buffer
    (the step's output, the call's rows a range of it), and that buffer."""
    proj, pool, slots, A_log, dt_bias, _, _, h, k = _build(spec, seed)
    rows = proj.shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    buf = torch.randn(1, rows + 9, HV, V, generator=g, device=DEV).to(torch.bfloat16)
    out = buf[0, 5 : 5 + rows].view(rows, HV * V)
    return [proj, pool, slots, A_log, dt_bias, h, k, out], buf


@pytest.mark.parametrize("cell", rows_op.CELLS, ids=[c.why[:44] for c in rows_op.CELLS])
def test_rows_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(rows_op)
    args, buf = _rows_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    outside = torch.cat([buf[0, :5], buf[0, 5 + args[0].shape[0] :]]).clone()
    ref_o, ref_pool = rows_op.reference(*args)
    got = rows_op(*args)
    assert got.data_ptr() == args[-1].data_ptr(), "written into out"
    torch.testing.assert_close(got, ref_o)
    torch.testing.assert_close(args[1], ref_pool)
    assert torch.equal(torch.cat([buf[0, :5], buf[0, 5 + args[0].shape[0] :]]), outside), (
        "rows outside the range untouched"
    )
    if cell.spec.get("pad"):
        assert (got[1] == 0).all(), "a pad row's output is zero"
