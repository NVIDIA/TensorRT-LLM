# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_draft_silu entry (the drafter MLP's plain SwiGLU as one C++ launch): bit for bit the Triton
silu_and_mul at its defaults, over every bf16 gate and every bf16 up pattern."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.spec.m3_draft_silu import (  # noqa: E501
    m3_draft_silu as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401


def seed_of(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


assert torch.cuda.is_available(), "m3_draft_silu requires a CUDA device"

ALL = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def _rows(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """[rows, 2d] gate | up from equally long flat gate and up values, d = 4608 (the TP4 drafter shard)."""
    d = 4608
    n = gate.numel()
    rows = -(-n // d)
    pad = rows * d - n
    g = torch.cat([gate, gate[:pad]]).view(rows, d)
    u = torch.cat([up, up[:pad]]).view(rows, d)
    return torch.cat([g, u], dim=1).contiguous()


def _assert_bitwise(x: torch.Tensor, what: str) -> None:
    got = op(x)
    assert got is not None, what
    ref = op.reference(x)
    torch.cuda.synchronize()
    diff = got.view(torch.int16) != ref.view(torch.int16)
    assert not bool(diff.any()), f"{what}: {int(diff.sum())} of {diff.numel()} differ"


@pytest.mark.parametrize("up", [0.0, 1.0, -1.0, 3.5, -0.0, 1e-3], ids=lambda v: f"up={v}")
def test_every_gate_pattern_is_bitwise_the_triton_kernel(up) -> None:
    gate = ALL.cuda()
    _assert_bitwise(_rows(gate, torch.full_like(gate, up)), f"gate sweep, up {up}")


@pytest.mark.parametrize("gate", [1.0, -1.0, 0.5, -3.0, 20.0, -0.0], ids=lambda v: f"gate={v}")
def test_every_up_pattern_is_bitwise_the_triton_kernel(gate) -> None:
    up = ALL.cuda()
    _assert_bitwise(_rows(torch.full_like(up, gate), up), f"up sweep, gate {gate}")


@pytest.mark.parametrize("seed", range(4))
def test_random_pairs_are_bitwise_the_triton_kernel(seed) -> None:
    g = gen(100 + seed)
    n = 1 << 22
    gate = torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda").to(torch.int16)
    up = torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda").to(torch.int16)
    _assert_bitwise(_rows(gate.view(torch.bfloat16), up.view(torch.bfloat16)), f"seed {seed}")


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The drafter's gate_up output range (and past it), in a CUDA graph as the drafter runs it."""
    g = gen(seed_of(cell.why))
    rows, d = cell.spec["rows"], cell.spec["d"]
    x = (4.0 * torch.randn(rows, 2 * d, generator=g, device="cuda")).to(torch.bfloat16)
    x[:, :6] = torch.tensor([60.0, -60.0, 0.0, -0.0, 88.0, -88.0], dtype=torch.bfloat16)
    certifying(op)
    ref = op.reference(x)
    eager = op(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(captured.view(torch.int16), ref.view(torch.int16))


def test_declines_what_it_does_not_cover() -> None:
    x = torch.zeros(2, 2 * 4608, dtype=torch.bfloat16, device="cuda")
    assert op(x[:, 8:]) is None  # strided rows
    assert (
        op(torch.zeros(2, 2 * 12, dtype=torch.bfloat16, device="cuda")) is None
    )  # d not a multiple of 8
    assert op(x.float()) is None
