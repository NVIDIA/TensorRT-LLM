# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_moe_dsa tp16 o_proj_publish entry.

One GPU stands in for the group: `lamport` is a plain device buffer laid out like an MNNVL
one-shot workspace (three Lamport buffers, every element the idle -0.0) and `flags` a plain int32
tensor, so the test can read back exactly which slots the op stored and that it stored nothing
else. On a real group `lamport` is the multicast address and the same stores reach every rank.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.gemm.low_m_gemm import (  # noqa: E501
    low_m_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.gemm.o_proj_publish import (  # noqa: E501
    o_proj_publish as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the glm_moe_dsa tp16 entries are certified on sm_107"
)

#: Lamport buffers of a one-shot workspace.
_BUFFERS = 3
#: Bytes past the slots in each buffer: a store past the end of the slot block lands there.
_SLACK_BYTES = 4096
#: The BF16 bit pattern of -0.0, the idle value of a Lamport buffer, as int16.
_IDLE = -32768


def _build(spec: dict, seed: int) -> dict:
    """The op's arguments for one cell: unit-variance attention outputs, a weight scaled to
    unit-variance outputs, a workspace with the slots of every token and rank plus slack."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, world = spec["m"], spec["k"], spec["n"], spec["world_size"]
    bytes_per_buffer = m * world * n * 2 + _SLACK_BYTES
    current = spec["current"]
    flags = torch.zeros(16, dtype=torch.int32, device="cuda")
    flags[:4] = torch.tensor([current, (current + 2) % _BUFFERS, bytes_per_buffer, 1])
    return dict(
        x=torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16),
        weight=(torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16),
        out=torch.empty(m, n, dtype=torch.bfloat16, device="cuda"),
        lamport=torch.full(
            (_BUFFERS * bytes_per_buffer // 2,), -0.0, dtype=torch.bfloat16, device="cuda"
        ),
        flags=flags,
        world_size=world,
        rank=spec["rank"],
    )


def _slots(args: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(this rank's [M, N] slots in the current buffer, a mask of every element of `lamport`
    outside them)."""
    m, n = args["x"].shape[0], args["weight"].shape[0]
    world, rank = args["world_size"], args["rank"]
    current, _, bytes_per_buffer, _ = args["flags"][:4].tolist()
    buffer = args["lamport"].view(_BUFFERS, bytes_per_buffer // 2)[current]
    slots = buffer[: m * world * n].view(m, world, n)
    elsewhere = torch.ones(args["lamport"].numel(), dtype=torch.bool, device="cuda")
    mask = elsewhere.view(_BUFFERS, -1)[current][: m * world * n].view(m, world, n)
    mask[:, rank, :] = False
    return slots[:, rank, :], elsewhere


def _compare(published: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, as for low_m_gemm.

    The FP32 sums of the kernel (split-K 2 over a 1024-long K) and of torch differ in order by
    about 1e-6 relative, which flips the BF16 rounding of a rare output by one ulp: 2 units of
    the band's 2**-8 at the row maximum, so 4 units element-wise and 2 units of relative RMS. A
    value in the wrong slot, a lost K slice or a wrong rounding is off by whole values.
    """
    assert_within_ulp(published, ref, element_ulp=4.0, rms_ulp=2.0)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    assert op(**args) is None
    published, elsewhere = _slots(args)
    assert bool((published.view(torch.int16) != _IDLE).all()), "a slot holds -0.0 (unwritten)"
    idle = args["lamport"].view(torch.int16)[elsewhere]
    assert bool((idle == _IDLE).all()), "the op stored outside this rank's slots"
    _compare(published, op.reference(**args))


def test_published_values_are_the_gemm_output_bit_for_bit() -> None:
    """The publish epilogue rounds as the plain epilogue does: the slots hold low_m_gemm's
    output at the same tactic, -0.0 written as +0.0."""
    args = _build(op.CELLS[0].spec, seed=61)
    with validating(op):
        op(**args)
    published, _ = _slots(args)
    plain = low_m_gemm(args["x"], args["weight"], torch.empty_like(args["out"]))
    plain = torch.where(plain == 0, torch.zeros_like(plain), plain)
    assert torch.equal(published.view(torch.int16), plain.view(torch.int16))


def test_cuda_graph_capture_and_replay() -> None:
    """Captured once, the op publishes into whichever buffer the flags mark at replay."""
    args = _build(op.CELLS[0].spec, seed=62)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(**args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)

    args["lamport"].fill_(-0.0)
    args["flags"][0] = (int(args["flags"][0]) + 1) % _BUFFERS
    args["x"].copy_(_build(op.CELLS[0].spec, seed=63)["x"])
    graph.replay()
    torch.cuda.synchronize()
    published, elsewhere = _slots(args)
    assert bool((args["lamport"].view(torch.int16)[elsewhere] == _IDLE).all())
    _compare(published, op.reference(**args))


def test_guard_refuses_a_rank_outside_the_group() -> None:
    """Rank 16 of 16 lands in the next token's slots."""
    args = _build(op.CELLS[0].spec, seed=64)
    args["rank"] = args["world_size"]
    with pytest.raises(AssertionError, match="outside a group of"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_buffer_smaller_than_the_slots() -> None:
    """The slots past the buffer's end land in the next Lamport buffer."""
    args = _build(op.CELLS[0].spec, seed=65)
    m, n = args["x"].shape[0], args["weight"].shape[0]
    args["flags"][2] = m * args["world_size"] * n * 2 - 16
    with pytest.raises(AssertionError, match="smaller than the"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_buffer_index_out_of_range() -> None:
    args = _build(op.CELLS[0].spec, seed=66)
    args["flags"][0] = _BUFFERS
    with pytest.raises(AssertionError, match="current buffer index"):
        with validating(op):
            op(**args)
