# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_moe_residual_add entry: bitwise against its reference at every cell on SM107, and
with ``any_rows`` bitwise against the two torch adds it replaces on a context step, planted
extremes included; the reference, the torch path and the two kernels' arithmetic on the CPU."""

import ast
import inspect
import zlib

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_moe_residual as kernels,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_residual_add import (  # noqa: E501
    HIDDEN,
    rubin_moe_residual_add,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

# BF16 bit patterns where the two roundings can go wrong: signed zeros, subnormals (smallest,
# largest), the smallest normal, 1 and the half-ulp / ulp / 1.5-ulp steps next to it, the largest
# finite value and half its ulp (max + that is a tie that rounds to inf), infinities and a NaN.
_PLANTED = (
    0x0000,
    0x8000,
    0x0001,
    0x8001,
    0x007F,
    0x0080,
    0x8080,
    0x3F80,
    0xBF80,
    0x3F81,
    0x3B80,
    0xBB80,
    0x3C00,
    0x3C40,
    0x4000,
    0x7F7F,
    0xFF7F,
    0x7B00,
    0xFB00,
    0x7F80,
    0xFF80,
    0x7FC0,
)


def _operands(rows, seed, device):
    """Routed and shared outputs of similar size and a larger residual, so both roundings bite."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    routed = torch.randn(rows, HIDDEN, generator=g) * 0.3
    shared = torch.randn(rows, HIDDEN, generator=g) * 0.3
    residual = torch.randn(rows, HIDDEN, generator=g) * 4.0
    return tuple(t.to(torch.bfloat16).to(device) for t in (routed, shared, residual))


def _planted_operands(rows, seed, device):
    """Every (routed, shared, residual) triple of the planted patterns, then random BF16 bit
    patterns (every exponent, subnormals, infinities and NaNs included) to fill ``rows`` rows."""
    n = rows * HIDDEN
    planted = np.array(_PLANTED, dtype=np.uint16)
    triples = np.stack(np.meshgrid(planted, planted, planted, indexing="ij"), -1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 1 << 16, size=(n, 3), dtype=np.uint16)
    k = min(n, len(triples))
    bits[:k] = triples[:k]
    out = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
    return tuple(out[:, i].contiguous().view(rows, HIDDEN).to(device) for i in range(3))


def _same_bits(a, b):
    """Byte-equal, except that a NaN matches any NaN (the payload is not compared)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a_nan, b_nan = torch.isnan(a), torch.isnan(b)
    if not torch.equal(a_nan, b_nan):
        return False
    keep = ~a_nan
    return torch.equal(a.view(torch.int16)[keep], b.view(torch.int16)[keep])


@gpu
@pytest.mark.parametrize(
    "cell", rubin_moe_residual_add.CELLS, ids=[c.why[:40] for c in rubin_moe_residual_add.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_moe_residual_add)
    any_rows = cell.spec.get("any_rows", False)
    args = _operands(cell.spec["rows"], zlib.crc32(cell.why.encode()), "cuda")
    out = rubin_moe_residual_add(*args, any_rows=any_rows)
    ref = rubin_moe_residual_add.reference(*args, any_rows=any_rows)
    # Exact: both add in FP32 and round to BF16 at the same two points.
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    if cell.spec["rows"]:
        one_rounding = (args[2].float() + args[0].float() + args[1].float()).to(torch.bfloat16)
        assert not torch.equal(out, one_rounding), "the inputs do not exercise the first rounding"


@gpu
@pytest.mark.parametrize("rows", [16384, 16383, 4095, 97, 1, 0])
def test_any_rows_is_the_two_torch_adds(rows):
    """A context step's update, against what it replaces there: ``prefix + (routed + shared)``."""
    routed, shared, residual = _operands(rows, 1000 + rows, "cuda")
    today = residual + (routed + shared)
    out = rubin_moe_residual_add(routed, shared, residual, any_rows=True)
    assert out.shape == today.shape and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.int16), today.view(torch.int16))


@gpu
@pytest.mark.parametrize("rows", [16384, 97, 2])
def test_any_rows_planted_values(rows):
    """Half-ulp ties, signed zeros, subnormals, the largest finite value, sums that overflow to
    inf, infinities, NaNs and random bit patterns: bitwise the two torch adds (NaN as a class)."""
    routed, shared, residual = _planted_operands(rows, 7 + rows, "cuda")
    today = residual + (routed + shared)
    out = rubin_moe_residual_add(routed, shared, residual, any_rows=True)
    assert _same_bits(out, today)
    assert _same_bits(out, rubin_moe_residual_add.reference(routed, shared, residual))
    # Every planted triple fits in two rows; its outputs include inf, NaN and -0.0.
    assert torch.isinf(out).any() and torch.isnan(out).any()
    assert (out.view(torch.int16) == -0x8000).any()


class _Launches:
    """Counts the launches of one Triton kernel (``kernel[grid](...)``)."""

    def __init__(self, kernel):
        self.kernel, self.count = kernel, 0

    def __getitem__(self, grid):
        self.count += 1
        return self.kernel[grid]


@gpu
@pytest.mark.parametrize(
    "rows, any_rows, want_rows_kernel, want_gen_kernel",
    [
        (16384, True, 1, 0),
        (1, True, 1, 0),
        (0, True, 0, 0),
        (96, False, 0, 1),
        (97, False, 0, 0),
    ],
)
def test_the_launch_that_serves_the_rows(
    rows, any_rows, want_rows_kernel, want_gen_kernel, monkeypatch
):
    """``any_rows`` takes the runtime-size kernel at every row count above 0; without it the
    generation kernel serves up to 96 rows and torch the rest, as before."""
    rows_kernel = _Launches(kernels._moe_residual_add_rows_kernel)
    gen_kernel = _Launches(kernels._moe_residual_add_kernel)
    monkeypatch.setattr(kernels, "_moe_residual_add_rows_kernel", rows_kernel)
    monkeypatch.setattr(kernels, "_moe_residual_add_kernel", gen_kernel)
    args = _operands(rows, 5, "cuda")
    out = rubin_moe_residual_add(*args, any_rows=any_rows)
    assert (rows_kernel.count, gen_kernel.count) == (want_rows_kernel, want_gen_kernel)
    assert torch.equal(out.view(torch.int16), (args[2] + (args[0] + args[1])).view(torch.int16))


@gpu
@pytest.mark.parametrize("any_rows, rows", [(False, 8), (True, 1000)])
def test_cuda_graph_replay_matches_eager(any_rows, rows):
    args = _operands(rows, 4, "cuda")
    eager = rubin_moe_residual_add(*args, any_rows=any_rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rubin_moe_residual_add(*args, any_rows=any_rows)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("any_rows", [False, True])
def test_reference_is_the_unfused_module_math(rows, any_rows):
    routed, shared, residual = _operands(rows, rows, "cpu")
    ref = rubin_moe_residual_add.reference(routed, shared, residual, any_rows=any_rows)
    assert ref.dtype == torch.bfloat16
    assert torch.equal(ref.view(torch.int16), (residual + (routed + shared)).view(torch.int16))
    # Off the GPU the op takes its torch path, which is that same math.
    assert torch.equal(rubin_moe_residual_add(routed, shared, residual, any_rows=any_rows), ref)


def _rne_bf16(x: np.ndarray) -> np.ndarray:
    """FP32 -> BF16 bit patterns, round to nearest even (a NaN becomes 0x7FC0)."""
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[np.isnan(x)] = 0x7FC0
    return r


def _f32(bits: np.ndarray) -> np.ndarray:
    """BF16 bit patterns -> FP32 values (exact)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.cpu_only
def test_the_kernel_steps_are_the_reference_on_planted_values():
    """The kernels' per-element steps (FP32 add, round to BF16, FP32 add, round to BF16), done in
    IEEE FP32 here, give the reference's and torch's bits on every planted triple and on random
    bit patterns (NaN as a class)."""
    routed, shared, residual = _planted_operands(4, 11, "cpu")
    r, s, p = (
        t.view(torch.int16).numpy().view(np.uint16).reshape(-1) for t in (routed, shared, residual)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        moe = _rne_bf16(_f32(r) + _f32(s))
        steps = _rne_bf16(_f32(p) + _f32(moe))
    emulated = torch.from_numpy(steps.view(np.int16).copy()).view(torch.bfloat16).view(routed.shape)
    assert _same_bits(emulated, rubin_moe_residual_add.reference(routed, shared, residual))
    assert _same_bits(emulated, residual + (routed + shared))
    # The planted block reaches ties, overflow to inf, inf - inf and signed zeros.
    assert np.isinf(_f32(steps)).sum() > 1000 and np.isnan(_f32(steps)).sum() > 1000
    assert (steps == 0x8000).any() and (steps == 0x7F80).any()


@pytest.mark.cpu_only
def test_the_rows_kernel_is_the_generation_kernels_arithmetic():
    """The two kernels' bodies are the same statements; their signatures differ only in ``size``
    being a runtime argument. So ``any_rows`` changes the launch, never a rounding."""

    def parsed(fn):
        (node,) = ast.parse(inspect.getsource(fn.fn)).body
        return node

    def annotation(arg):
        return None if arg.annotation is None else ast.dump(arg.annotation)

    gen = parsed(kernels._moe_residual_add_kernel)
    rows = parsed(kernels._moe_residual_add_rows_kernel)
    assert [ast.dump(s) for s in gen.body] == [ast.dump(s) for s in rows.body]
    assert [a.arg for a in gen.args.args] == [a.arg for a in rows.args.args]
    differ = [
        a.arg for a, b in zip(gen.args.args, rows.args.args) if annotation(a) != annotation(b)
    ]
    assert differ == ["size"], differ
    assert [ast.dump(d) for d in gen.decorator_list] == [ast.dump(d) for d in rows.decorator_list]
    assert kernels._FUSED_MAX_ROWS == 96
    # int32 element offsets: the last block of the largest any-row launch ends below 2**31.
    assert kernels._ANY_ROWS_MAX * HIDDEN + 1024 < 2**31


@pytest.mark.cpu_only
def test_guard_refuses_a_broadcast():
    routed, shared, residual = _operands(4, 1, "cpu")
    rubin_moe_residual_add.is_valid(routed, shared, residual)
    rubin_moe_residual_add.is_valid(routed, shared, residual, any_rows=True)
    with pytest.raises(AssertionError):
        rubin_moe_residual_add.is_valid(routed[:1], shared, residual)
    with pytest.raises(AssertionError):
        rubin_moe_residual_add.is_valid(routed.float(), shared, residual, any_rows=True)
