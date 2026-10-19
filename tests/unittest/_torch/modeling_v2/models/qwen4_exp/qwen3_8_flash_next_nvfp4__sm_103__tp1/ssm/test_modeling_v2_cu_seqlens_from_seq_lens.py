# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cu_seqlens_from_seq_lens entry.

Integer arithmetic, gated bitwise. Beyond the cells: a near-miss vector is
monotonic and plausible, so the tests that matter are the ones proving the
vector actually splits a packed buffer and differs from every near miss.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.cu_seqlens_from_seq_lens import (  # noqa: E501
    BATCHES,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.cu_seqlens_from_seq_lens import (  # noqa: E501
    cu_seqlens_from_seq_lens as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "cu_seqlens_from_seq_lens requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: integer arithmetic, no tolerance to loosen."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    assert torch.equal(out, ref), f"{out.tolist()} != {ref.tolist()}"


DEV = "cuda"


def _seq_lens(lengths, dtype=torch.int32):
    return torch.tensor(list(lengths), dtype=dtype, device=DEV)


def _build(spec):
    lengths, dtype = spec["lengths"], spec["in_dtype"]
    layout = spec.get("layout", "contiguous")
    if layout == "prefix":
        # The metadata buffer is sized for max_num_requests; the call site
        # passes its first `batch` entries.
        buf = torch.full((len(lengths) + 9,), 99, dtype=dtype, device=DEV)
        buf[: len(lengths)] = _seq_lens(lengths, dtype)
        return buf[: len(lengths)]
    if layout == "strided":
        buf = torch.full((2 * len(lengths),), 99, dtype=dtype, device=DEV)
        buf[::2] = _seq_lens(lengths, dtype)
        view = buf[::2]
        assert not view.is_contiguous()
        return view
    return _seq_lens(lengths, dtype)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    seq_lens = _build(spec)
    before = seq_lens.clone()
    out = op(seq_lens, spec["out_dtype"])
    assert torch.equal(seq_lens, before), "seq_lens is read-only"
    assert out.device.type == "cuda"
    _compare(out, op.reference(seq_lens, spec["out_dtype"]))

    # The entry's purpose: slicing a packed buffer with the vector recovers
    # exactly the sequences packed into it.
    lengths = spec["lengths"]
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(cell.why.encode()))
    pieces = [torch.randint(0, 10_000, (n,), generator=g, device=DEV) for n in lengths]
    packed = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64, device=DEV)
    bounds = out.tolist()
    assert bounds[-1] == packed.shape[0]
    for s, piece in enumerate(pieces):
        assert torch.equal(packed[bounds[s] : bounds[s + 1]], piece), f"sequence {s} mis-sliced"


def test_wrong_variants_all_differ() -> None:
    """Every near-miss spelling is monotonic and plausible; none is this one."""
    lengths = BATCHES["mixed_p2"]
    seq_lens = _seq_lens(lengths)
    out = op(seq_lens, torch.int32)
    ref = op.reference(seq_lens, torch.int32)
    _compare(out, ref)

    inclusive = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    no_total = torch.cat([torch.zeros(1, dtype=torch.int32, device=DEV), inclusive[:-1]])
    right_padded = torch.cat([inclusive, inclusive[-1:]])
    # The one-row-per-generation form `Mamba2Metadata.prepare` builds: a real
    # cumsum over the context prefix, then a constant plus arange.
    num_ctx, num_gen = 2, len(lengths) - 2
    engine = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEV)
    torch.cumsum(seq_lens[:num_ctx], dim=0, dtype=torch.int32, out=engine[1 : num_ctx + 1])
    engine[num_ctx + 1 :] = engine[num_ctx] + torch.arange(
        1, num_gen + 1, dtype=torch.int32, device=DEV
    )
    for name, wrong in {
        "inclusive scan": inclusive,
        "total dropped": no_total,
        "right-padded": right_padded,
        "one row per generation": engine,
        "shifted": torch.roll(out, 1),
    }.items():
        with pytest.raises(AssertionError):
            _compare(wrong, ref)
        assert wrong.shape != ref.shape or not torch.equal(wrong, ref), name

    # The metadata form agrees exactly when every generation request emits one
    # row, which is why the mistake is invisible until speculation is on.
    unspec = _seq_lens(list(lengths[:num_ctx]) + [1] * num_gen)
    assert torch.equal(op(unspec, torch.int32), engine)


def test_result_is_fresh_and_deterministic() -> None:
    seq_lens = _seq_lens(BATCHES["decode_p2"])
    results = [op(seq_lens, torch.int32) for _ in range(5)]
    assert len({r.data_ptr() for r in results} | {seq_lens.data_ptr()}) == 6
    for other in results[1:]:
        assert torch.equal(results[0], other)


def test_accumulation_happens_in_the_requested_dtype() -> None:
    """The hard-limit row: int64 widens an int32 input; int32 wraps silently."""
    seq_lens = _seq_lens([2_000_000_000, 2_000_000_000], torch.int32)
    assert op(seq_lens, torch.int64).tolist() == [0, 2_000_000_000, 4_000_000_000]
    assert op(seq_lens, torch.int32).tolist() == [0, 2_000_000_000, -294_967_296]


def test_cuda_graph_capture_and_replay() -> None:
    """A captured graph follows lengths rewritten in place: only the shape
    reaches Python, the values are read on device and never synchronized on."""
    static_lens = _seq_lens([2] * 8)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(static_lens, torch.int32)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_lens, torch.int32)
    for lengths in ([1] * 8, [2] * 8, [3, 1, 2, 1, 1, 1, 1, 1]):
        static_lens.copy_(_seq_lens(lengths))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, op.reference(static_lens, torch.int32))


def test_guard_refuses_a_seq_lens_that_is_not_1d() -> None:
    """Unguarded, a [1, 3] row vector returns a length-2 result, not length 4."""
    row = _seq_lens([7, 1, 13]).unsqueeze(0)
    with pytest.warns(UserWarning):
        unguarded = op(row, torch.int32)
    assert unguarded.shape == (2,), "the hazard this guard covers is gone"
    for bad in (row, torch.zeros(3, 2, dtype=torch.int32, device=DEV), _seq_lens([3])[0]):
        with pytest.raises(AssertionError, match="1-D per-request row count"):
            with validating(op):
                op(bad, torch.int32)


def test_guard_refuses_non_integer_seq_lens() -> None:
    """Unguarded, a float32 seq_lens is truncated toward zero before the scan."""
    floats = torch.tensor([7.9, 1.2, 13.6], dtype=torch.float32, device=DEV)
    assert op(floats, torch.int32).tolist() == [0, 7, 8, 21], "hazard gone"
    for dtype in (torch.float32, torch.bfloat16, torch.bool):
        with pytest.raises(AssertionError, match="seq_lens must have an integer dtype"):
            with validating(op):
                op(torch.ones(3, dtype=dtype, device=DEV), torch.int32)


def test_guard_refuses_a_non_integer_result_dtype() -> None:
    """Unguarded, a float `dtype` returns a float vector the consumers misread."""
    assert op(_seq_lens([1, 2, 3]), torch.float32).dtype is torch.float32
    for dtype in (torch.float32, torch.float64, torch.bool):
        with pytest.raises(AssertionError, match="dtype must be an integer dtype"):
            with validating(op):
                op(_seq_lens([1, 2, 3]), dtype)
