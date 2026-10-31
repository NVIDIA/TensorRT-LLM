# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul catalog entry.

The entry owns what is certified. This file turns a cell's spec into tensors,
drives the inputs the guard exists to refuse, and holds the properties the
qwen4_exp shared expert's call site depends on: CUDA-graph replay at the captured
decode row counts, and the saturated tails of the gate.
"""

import zlib

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"

#: The MTP layer's shared expert: intermediate 640, gate and up concatenated.
_QWEN4_WIDTH = 2 * 640


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


def _qwen4(rows: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(shape=(rows, _QWEN4_WIDTH), dtype=torch.bfloat16))


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at attention TP8: the shared expert's gate+up, 2 x 2048/8",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16),
        ),
    ),
    # The qwen4_exp shared expert's SwiGLU, contiguous [rows, 2 x 640], as the tp1
    # target calls it in its trunk and MTP layer (certified for the unified target's
    # MTP layer as variant r44; that target now calls `silu_and_mul_rows` there): one
    # decode request, the largest captured batch, that batch's mtp7 verify rows
    # (128 x 8), and a prefill chunk at max_num_tokens (the envelope's largest row).
    "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__tp1": (
        _qwen4(1, "qwen4 MTP shared expert: one decode row, 2 x 640"),
        _qwen4(128, "qwen4 MTP shared expert: the 128-request captured batch"),
        _qwen4(1024, "qwen4 MTP shared expert: 128 x 8 mtp7 verify rows"),
        _qwen4(8192, "qwen4 MTP shared expert: an 8192-token prefill chunk"),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def test_the_entry_cells_are_all_driven() -> None:
    """`test_certified_cells` drives every cell the entry itself declares (`op.CELLS`),
    not only the targets' `TARGET_CELLS`: `certified_cells` keeps one cell per distinct spec."""
    driven = {repr(sorted(c.spec.items())) for c in CELLS}
    undriven = [c.why for c in op.CELLS if repr(sorted(c.spec.items())) not in driven]
    assert not undriven, f"entry cells test_certified_cells does not drive: {undriven}"


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x)
    assert out.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    torch.testing.assert_close(out, op.reference(x))


@pytest.mark.parametrize("rows", [1, 128, 1024])
def test_cuda_graph_capture_and_replay(rows) -> None:
    """The MTP layer's decode call site runs captured: one request, the 128-request
    batch, and that batch's mtp7 verify rows. Replay after a refill answers bitwise
    what an eager call does on the new input."""
    spec = dict(shape=(rows, _QWEN4_WIDTH), dtype=torch.bfloat16)
    x = _build(spec, 100 + rows)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(x)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x)
    x.copy_(_build(spec, 200 + rows))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, op(x))
    torch.testing.assert_close(out, op.reference(x))


def test_saturated_gates_match_the_reference() -> None:
    """The kernel's SiLU is `g / (1 + __expf(-g))`. From a gate of 20 up, the fp32
    sigmoid is exactly one, so the output is the bf16 rounding of `g * u`; at -100
    `__expf` overflows to inf and the quotient is a signed zero, where torch's
    `g * sigmoid(g)` is an fp32 denormal that bf16 rounds to zero; -20 and -88 stay
    small but representable and are held to the reference band. The gate half holds
    these values, the up half random ones."""
    specials = [0.0, -0.0, 20.0, -20.0, 88.0, -88.0, 100.0, -100.0]
    n = len(specials)
    x = _build(dict(shape=(4, _QWEN4_WIDTH), dtype=torch.bfloat16), 7)
    x[:, :n] = torch.tensor(specials, device="cuda").to(torch.bfloat16)
    out = op(x)
    torch.testing.assert_close(out, op.reference(x))

    gate, up = x[:, :n].float(), x[:, 640 : 640 + n].float()
    positive = gate >= 20.0
    assert torch.equal(out[:, :n][positive], (gate * up).to(torch.bfloat16)[positive])
    zero = (gate == 0.0) | (gate == -100.0)
    assert torch.equal(out[:, :n][zero].float(), torch.zeros_like(gate[zero]))


def test_the_band_refuses_a_wrong_activation() -> None:
    """The cells' band (torch's bf16 default) has to tell this op from its neighbours:
    the halves swapped (SiLU on the up projection) and GELU-tanh in place of SiLU."""
    x = _build(dict(shape=(128, _QWEN4_WIDTH), dtype=torch.bfloat16), 11)
    out = op(x)
    gate, up = x.float().chunk(2, dim=-1)
    for name, wrong in (
        ("swapped halves", (F.silu(up) * gate).to(x.dtype)),
        ("gelu_tanh", (F.gelu(gate, approximate="tanh") * up).to(x.dtype)),
    ):
        try:
            torch.testing.assert_close(out, wrong)
        except AssertionError:
            continue
        pytest.fail(f"the band accepted {name} in place of silu(gate) * up")


@pytest.mark.parametrize(
    "width,match",
    [
        (4095, "must be even"),
        # 24 bf16 elements is 48 bytes per row, which the op's own row check
        # accepts; the half is 24 bytes, so the second half starts mid-vector.
        (24, "16-byte vectors"),
        # The only width that reaches the third assert: for the half to be a
        # whole number of 16-byte vectors *and* smaller than one, it has to be
        # zero. Worth a case anyway -- it is the difference between a guard
        # that is unreachable and one that is merely narrow.
        (0, "at least one 16-byte vector"),
    ],
)
def test_guard_refuses_widths_the_kernel_faults_on(width, match) -> None:
    """A misaligned half kills the CUDA context rather than raising.

    That is why these are `is_valid` and not left to the op: the failure is
    unrecoverable and its message names neither the op nor the shape.
    """
    x = torch.randn(4, width, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(x)
