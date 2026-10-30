# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_decode_gated_norm_pdl entry.

Cells against the composition's fp64 reference (the gated delta rule's recurrence,
then the gated norm entry's) in torch's default band for the bf16 output and the fp32
pool, the pool compared whole so slots the call did not name are
gated too. Beyond the cells: a stale-read check under graph replay (a producer
rewrites the projection inside the graph right before the op), and the row guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.gdn_decode_gated_norm_pdl__r31 import (  # noqa: E501
    gdn_decode_gated_norm_pdl as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_decode_gated_norm_pdl requires a CUDA device"

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
    proj = torch.randn(rows, WIDTH, generator=g, device=DEV).to(torch.bfloat16)
    if spec.get("pool") == "coalesced":
        # The KV-cache-v2 manager's view: every GDN layer's state of a slot in one page,
        # this layer's state one sub-page of it, so the slot stride is 36 layers' states
        # and slot 100 sits ~2.8G elements past the base (past int32).
        layers, slots = 36, 102
        page = torch.empty(slots * layers * HV * V * K, device=DEV)
        page[: HV * V * K * 4].normal_(0.0, 0.05, generator=g)
        pool = page.as_strided((slots, HV, V, K), (layers * HV * V * K, V * K, K, 1))
        for s in (100, 101):
            pool[s].copy_(torch.randn(HV, V, K, generator=g, device=DEV) * 0.05)
    else:
        pool = torch.randn(SLOTS, HV, V, K, generator=g, device=DEV) * 0.05
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


def test_reads_after_the_wait_under_graph_replay() -> None:
    """A producer rewrites the projection inside the graph right before the op; with PDL
    a read of it before `griddepcontrol.wait` would see the previous replay's row. The
    pool advances one step per replay, as in decode."""
    args = _build(dict(rows=1), seed=41)
    src = args[0].clone()

    def step():
        args[0].copy_(src * 1.0)
        return op(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    pool0 = args[1].clone()
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        src.copy_(_build(dict(rows=1), seed=100 + trial)[0])
        before = args[1].clone()
        graph.replay()
        torch.cuda.synchronize()
        after = args[1].clone()
        args[1].copy_(before)
        want = op(src.clone(), *args[1:])
        assert torch.equal(graph_out, want), f"replay {trial} output"
        assert torch.equal(after, args[1]), f"replay {trial} pool"
    del pool0


def test_guards() -> None:
    """More rows than `MAX_ROWS` are refused."""
    args = _build(dict(rows=1), seed=5)
    with validating(op), pytest.raises(AssertionError):
        op(args[0].expand(5, -1).contiguous(), args[1], args[2].expand(5).contiguous(), *args[3:])
