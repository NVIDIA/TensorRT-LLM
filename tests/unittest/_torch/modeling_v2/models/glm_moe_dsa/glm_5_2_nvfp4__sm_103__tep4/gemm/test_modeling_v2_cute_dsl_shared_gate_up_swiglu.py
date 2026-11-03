# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_shared_gate_up_swiglu entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "cute_dsl_shared_gate_up_swiglu runs an sm_103 CuTe DSL kernel", allow_module_level=True
    )

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_shared_gate_up_swiglu import (  # noqa: E402, E501
    MTP_ROWS,
    PREFILL_MAX_ROWS,
    PREFILL_MIN_ROWS,
    PrefillGateUpRunner,
    build,
    build_prefill,
    prefill_family,
    prepare_weight,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_shared_gate_up_swiglu import (  # noqa: E402, E501
    cute_dsl_shared_gate_up_swiglu as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, GATE_UP, OUT = 6144, 1024, 512


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The reference keeps the gate and up sums in fp32 and rounds once. The
    kernel sums the same exact products in another order and may round each
    sum to bf16 before SwiGLU, as the two-launch path does: up to one ulp on
    each factor, so about two on the product. 4 ulp element-wise against the
    row scale and 2 ulp relative RMS; a wrong pairing of gate and up rows, or
    the other layer's weight, moves outputs by O(1).

    The prefill kernels take the same band; this is the derivation for what
    they round. Each family sums exact bf16 products in fp32. The split-K
    family adds two fp32 partials of 3072 terms each, one more fp32 rounding.
    Both order effects stay near 2**-18 relative, far below a bf16 ulp. Every
    family then rounds the gate and up sums to bf16 before SwiGLU, as its
    kernel states. Each of those roundings moves a factor by at most 2**-8 of
    its value. SiLU uses fast exp and reciprocal, about 2**-22 relative.

    A unit below is 2**-8 of the row scale, as the gate counts. For `g > 0`,
    a relative error e in the gate moves `silu(g)` by at most 1.28 e. The two
    factor roundings then move an output by at most 2.28 x 2**-8 of its own
    magnitude, so at most 2.28 units. For `g < 0` the move is at most
    0.27 `|up|` x 2**-8, a fraction of a unit against a row scale of a few
    (the largest magnitude among a row's 512 outputs). Both sides then round
    to bf16. At the row maximum one bf16 ulp is 1 to 2 units. Values at most
    2.28 units apart round at most 2 ulps apart when the ulp exceeds 1.14
    units, and at most 3 ulps (3.42 units) otherwise. Smaller elements have
    smaller ulps and stay below 3.3 units. So the gap stays within 4 units,
    and the element band holds. Equality needs a power-of-two row maximum and
    every rounding at its extreme.

    For the RMS, each bf16 rounding adds about 0.42 x 2**-8 of relative
    error. The two factor roundings and the two output roundings give about
    0.9 x 2**-8, under the 2 ulp band. A CPU emulation of this epilogue over
    8192-row draws measured at most 3.4 ulp element-wise and 0.90 ulp RMS,
    with or without the split-K sum."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _checkpoint_weight(g: torch.Generator) -> torch.Tensor:
    """A `[gate | up]` checkpoint weight scaled so each sum is O(1)."""
    w = torch.randn(GATE_UP, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5
    return w.to(torch.bfloat16)


#: The prefill runner, built once per session: it compiles every family.
_PREFILL: dict = {}


def _prefill_runner() -> PrefillGateUpRunner:
    if "runner" not in _PREFILL:
        g = torch.Generator(device="cuda").manual_seed(0)
        _PREFILL["runner"] = build_prefill(prepare_weight(_checkpoint_weight(g)))
    return _PREFILL["runner"]


def _prefill_rows(family: str, edge: str) -> int:
    """The first or last row count at which the device SM count selects
    `family`. The cells name families, not row counts, because the family
    boundaries move with the SM count."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = [
        m
        for m in range(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS + 1)
        if prefill_family(m, num_sms) == family
    ]
    if not rows:
        pytest.skip(f"{family} is never selected on a {num_sms}-SM device")
    return rows[0] if edge == "first" else rows[-1]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    if "prefill" in cell.spec:
        rows = _prefill_rows(cell.spec["prefill"], cell.spec["edge"])
        runner = _prefill_runner()
        assert prefill_family(rows, runner.num_sms) == cell.spec["prefill"]
    else:
        rows = cell.spec["rows"]
        runner = build(rows, prepare_weight(_checkpoint_weight(g)))
    # Another layer's weight: the target builds once and runs every layer's
    # own prepared weight through the same runner.
    prepared = prepare_weight(_checkpoint_weight(g))
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    # NaN, so an element the kernel never writes fails the comparison.
    out = torch.full((rows, OUT), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        assert op(x, prepared, out, runner) is out, "the op writes `out` and returns it"
    _compare(out, op.reference(x, prepared, out, runner))


#: MTP5 at 1-3 requests (6 / 12 / 18 rows; MTP3 at 3 is 12): the target builds the fused cell
#: at these on an MTP3 / MTP5 engine (`_SHARED_FUSED_UP_SEQS`); 16, a decode cell, is the control.
MTP_DECODE_ROWS = (6, 12, 18, 16)


@pytest.mark.parametrize("rows", MTP_ROWS + MTP_DECODE_ROWS)
def test_mtp_rows_replay_bitwise_in_a_cuda_graph(rows) -> None:
    """The MTP3 / MTP5 runners (128 / 192, and the fused cells at the small-batch decode
    counts 6 / 12 / 18) run inside the decode CUDA graph: a replay writes the eager launch's
    bytes, and a replay after new operands are copied into the captured buffers writes the
    eager launch's product on them, within the band of the reference."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    prepared = prepare_weight(_checkpoint_weight(g))
    runner = build(rows, prepared)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    out = torch.empty(rows, OUT, dtype=torch.bfloat16, device="cuda")
    op(x, prepared, out, runner)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, prepared, out, runner)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "the replay differs from the eager launch"
    x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
    prepared.copy_(prepare_weight(_checkpoint_weight(g)))
    again = torch.empty_like(out)
    op(x, prepared, again, runner)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, again), "the replay on new operands differs from the eager launch"
    assert not torch.equal(again, eager), "the new operands did not change the product"
    _compare(out, op.reference(x, prepared, out, runner))


#: MTP3 / MTP5 decode row counts without a cell: each bucket's edges, the 24-request batches
#: (96, 144) and partial batches at widths 4 / 6.
WIDE_ROWS = (65, 66, 68, 84, 96, 100, 126, 127, 129, 144, 150, 190, 191)


def test_mtp_cells_at_other_wide_row_counts_are_the_cells_bitwise() -> None:
    """The MTP cells are the split-K kernel on a symbolic row count: the target runs a
    wide decode row count without a cell on the next cell up at its exact rows. Each row is
    bitwise what the cell writes for it at its own row count with the input padded by
    other (random) rows, nothing is written past the last row, and it is within the band."""
    g = torch.Generator(device="cuda").manual_seed(11)
    prepared = prepare_weight(_checkpoint_weight(g))
    cells = {m: build(m, prepared) for m in MTP_ROWS}
    for rows in WIDE_ROWS:
        cell = min(m for m in cells if m > rows)
        x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        parent = torch.full((rows + 8, OUT), -7.0, dtype=torch.bfloat16, device="cuda")
        out = parent[:rows]
        with validating(op):
            assert op(x, prepared, out, cells[cell]) is out
        padded = torch.randn(cell, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        padded[:rows] = x
        full = torch.empty(cell, OUT, dtype=torch.bfloat16, device="cuda")
        op(padded, prepared, full, cells[cell])
        torch.cuda.synchronize()
        assert torch.equal(out, full[:rows]), f"{rows} rows on the {cell}-row cell: not the cell's"
        assert bool((parent[rows:] == -7.0).all()), f"{rows} rows: wrote past the last row"
        _compare(out, op.reference(x, prepared, out, cells[cell]))


def test_prepare_weight_pairs_each_gate_row_with_its_up_row() -> None:
    """The reference reads gate at even rows and up at odd rows; this pins
    that to the checkpoint's `[gate_0..gate_511, up_0..up_511]` order."""
    w = _checkpoint_weight(torch.Generator(device="cuda").manual_seed(7))
    prepared = prepare_weight(w)
    assert torch.equal(prepared[0::2], w[:OUT])
    assert torch.equal(prepared[1::2], w[OUT:])


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _ok(rows: int = 8):
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    prepared = torch.zeros(GATE_UP, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows, OUT, dtype=torch.bfloat16, device="cuda")
    return x, prepared, out


def test_guard_refuses_a_strided_out() -> None:
    """A column slice of a wider buffer. The kernel checks layout on a
    runner's first call only, so nothing else stops it."""
    x, prepared, _ = _ok()
    wide = torch.empty(8, OUT + 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="out must be contiguous"):
        with validating(op):
            op(x, prepared, wide[:, :OUT], _unreachable)


def test_guard_refuses_a_misaligned_input() -> None:
    """The kernel's TMA needs 128-byte aligned operands; it checks only once."""
    _, prepared, out = _ok()
    flat = torch.zeros(8 * HIDDEN + 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="x must be 128-byte aligned"):
        with validating(op):
            op(flat[8:].view(8, HIDDEN), prepared, out, _unreachable)


def test_the_prefill_runner_refuses_a_decode_row_count_loudly() -> None:
    """The prefill kernels serve 65..8192 rows and check every call, so `is_valid`
    adds nothing for them; this pins that the refusal is loud. 64 rows fail the
    family selection before any kernel is looked up."""
    x, prepared, out = _ok(64)
    with pytest.raises(ValueError, match="65..8192"):
        op(x, prepared, out, PrefillGateUpRunner(num_sms=2, families={}))
