# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_indexer_topk_prefill catalog entry.

The cells marked `rereads` (Mistral Large 4's 1M envelope) also check which
kernel and path they drive: the kernel the op compiles for the width's bucket
(its candidate index width and shared candidate buffer), and how many rows
take REREAD's second pass, counted from the cell's own scores with the
kernel's coarse key. A control shows the gate refuses the answer a 16-bit
candidate index would give past 65536 columns.
"""

import zlib

import cutlass
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_topk_prefill import (  # noqa: E501
    cute_dsl_indexer_topk_prefill as op,
)
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLTopKPrefillSingleCTARunner
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.filtered_top_k_prefill_varlen import (
    FilteredTopKKernelVarlenPrefill,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "the entry's cells drive a CuTe DSL kernel on the GPU"

#: Written everywhere outside a row's window, its row padding included. A
#: kernel that read outside would rank these above every real score.
POISON = 1.0e30


def _row_stride(width: int, layout: str) -> int:
    """fp8_mqa_logits returns `[T, N]` as a view of rows padded to a multiple of
    256 plus 256 more (measured: 577 -> 1024, 3000 -> 3328, 9216 -> 9472,
    16384 -> 16640, 16385 -> 16896, 20000 -> 20480, 32768 -> 33024, 65537 ->
    66048, 131073 -> 131584, 524289 -> 524800, 983256 -> 983552, 983296 ->
    983552, 1024000 -> 1024256, 1048576 -> 1048832, 1048577 -> 1049088,
    2097152 -> 2097408, 2097153 -> 2097664, 2373713 -> 2374144)."""
    if layout == "compact":
        return width
    assert layout == "padded", layout
    return (width + 255) // 256 * 256 + 256


def _windows(requests):
    """`(row_starts, row_ends, width)` for prompts packed back to back in one K block.

    Prompt `(cached, new, taken)` owns `cached + new` keys starting where the
    previous one ended; its `j`-th new token sees `[offset, offset + cached + j +
    1)`, as an indexer prefill chunk's `cu_seqlen_ks` / `cu_seqlen_ke` give it.
    `taken` keeps only the prompt's last rows, like one row tile.
    """
    starts, ends, offset = [], [], 0
    for cached, new, taken in requests:
        first = 0 if taken is None else new - taken
        for j in range(first, new):
            starts.append(offset)
            ends.append(offset + cached + j + 1)
        offset += cached + new
    as_t = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    return as_t(starts), as_t(ends), offset


#: Rows argsorted per call while drawing the permutations. A whole 32768 x
#: 32768 block sorted at once took the cell past 40 GiB (an 8 GiB int64 result
#: plus the sort's scratch); each row's permutation is the same either way.
_SORT_ROWS = 4096


def _scores(rows: int, width: int, values: str, g: torch.Generator) -> torch.Tensor:
    """Per-row scores; every pattern but "ties" is distinct within a row."""
    perm = torch.rand(rows, width, generator=g, device="cuda")
    if values == "ties":
        del perm
        return torch.randint(0, 4, (rows, width), generator=g, device="cuda").float()
    for lo in range(0, rows, _SORT_ROWS):
        perm[lo : lo + _SORT_ROWS] = perm[lo : lo + _SORT_ROWS].argsort(dim=1).float()
    if values == "distinct":
        return perm
    if values == "narrow":
        # Distinct in fp32 (steps of 8 ulps at 1.0), one value in the fp16 the
        # kernel's coarse histogram keys on: every key lands in one bin. Past
        # 2^17 keys they span [1, 2), whose quarter binades (one coarse bin
        # each) still hold 2^18 keys at 2^20; past 2^20 they reach [2, 4), whose
        # quarter binades hold 2^19.
        return perm.mul_(2.0**-20).add_(1.0)
    if values == "spread":
        # 2^15 at the top, falling by a factor 2^(1/8192) per rank: one coarse
        # bin (a quarter binade) holds at most about 2640 keys, fewer than the
        # 4096-entry shared buffer of the 32-bit-index kernels. Distinct, and
        # normal fp32 down to 2^-113 at 2^20 keys; past about 1.15e6 keys the
        # lowest ranks go subnormal, then zero, far below any ranked window's
        # top_k in the cells.
        return perm.sub_(width - 1).mul_(2.0**-13).add_(15.0).exp2_()
    # Monotone in the permutation, so still distinct, from -1.6e6 to 1.6e6.
    assert values == "wide", values
    return perm.sub_(width / 2).mul_(30.0 / width).sinh_()


def _build(requests, spec, seed):
    """`(input_values, row_starts, row_ends, output_indices)` for one K block.

    `input_values` is a `[rows, W]` view whose row stride follows `layout`;
    outside each window, padding included, it holds `POISON`. `output_indices`
    starts at -7 so an unwritten slot shows.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    row_starts, row_ends, width = _windows(requests)
    rows = row_starts.shape[0]
    cols = torch.arange(width, device="cuda")
    outside = (cols < row_starts[:, None]) | (cols >= row_ends[:, None])
    scores = _scores(rows, width, spec["values"], g).masked_fill_(outside, POISON)
    del outside
    base = torch.full((rows, _row_stride(width, spec["layout"])), POISON, device="cuda")
    base[:, :width] = scores
    del scores
    output_indices = torch.full((rows, spec["top_k"]), -7, dtype=torch.int32, device="cuda")
    return base[:, :width], row_starts, row_ends, output_indices


#: The configuration the tests outside the cells run at: the cells' own.
_SPEC = dict(
    values="distinct",
    layout="padded",
    top_k=2048,
    num_copy_bits=128,
    overflow_policy="REREAD",
    cache_smem_values=False,
)


def _kwargs(spec):
    return dict(
        top_k=spec["top_k"],
        num_copy_bits=spec["num_copy_bits"],
        overflow_policy=spec["overflow_policy"],
        cache_smem_values=spec["cache_smem_values"],
    )


def _compare(out, ref, input_values, row_starts, row_ends, exact_sets: bool) -> None:
    """Exact, because selection does no arithmetic: there is nothing to round.

    A short row (`len <= top_k`) is fully specified, order included. A long
    row must hold `top_k` distinct window-local columns whose scores are,
    as a multiset, the window's `top_k` largest. Where every score in the row is
    distinct that pins the set itself, which `exact_sets` then checks; with ties
    at the k-th score any of the tied columns is a right answer.
    """
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    top_k = out.shape[1]
    lengths = (row_ends - row_starts).long()
    short = lengths <= top_k
    bad = (out[short] != ref[short]).any(dim=1)
    assert not bool(bad.any()), (
        f"{int(bad.sum())} of {int(short.sum())} short rows are not 0..len-1 then -1; "
        f"first at row {int(short.nonzero()[bad.nonzero()[0]])}"
    )
    long_rows = (~short).nonzero().flatten()
    if not long_rows.numel():
        return
    got = out[long_rows].long()
    span = lengths[long_rows, None]
    outside = ((got < 0) | (got >= span)).any(dim=1)
    assert not bool(outside.any()), (
        f"{int(outside.sum())} long rows hold a slot outside their window or unwritten; "
        f"first at row {int(long_rows[outside.nonzero()[0]])}"
    )
    got = got.sort(dim=1).values
    repeated = (got[:, 1:] == got[:, :-1]).any(dim=1)
    assert not bool(repeated.any()), f"{int(repeated.sum())} long rows select a column twice"
    start = row_starts.long()[long_rows, None]
    want = ref[long_rows].long()
    # Gathered in place: a copy of the long rows is 4 GiB at a 32768-column block.
    got_scores = input_values[long_rows[:, None], got + start].sort(dim=1).values
    want_scores = input_values[long_rows[:, None], want + start].sort(dim=1).values
    wrong = (got_scores != want_scores).any(dim=1)
    assert not bool(wrong.any()), (
        f"{int(wrong.sum())} of {long_rows.numel()} long rows select scores that are not "
        f"their window's top {top_k}; first at row {int(long_rows[wrong.nonzero()[0]])}"
    )
    if exact_sets:
        differ = (got != want.sort(dim=1).values).any(dim=1)
        assert not bool(differ.any()), f"{int(differ.sum())} long rows select another set"


def _compiled_key(width: int, spec) -> tuple:
    """The runner's cache key for the kernel a call at this width launches."""
    bucket = 1 << max(width - 1, 0).bit_length()
    return (
        cutlass.Float32,
        bucket,
        spec["top_k"],
        False,
        spec["num_copy_bits"],
        spec["overflow_policy"],
        spec["cache_smem_values"],
    )


def _kernel(width: int, spec) -> FilteredTopKKernelVarlenPrefill:
    """The kernel object the runner compiles for this width's bucket (built, not
    compiled), to read its candidate index type and shared buffer size."""
    _, bucket, top_k, _, num_copy_bits, overflow_policy, cache_smem_values = _compiled_key(
        width, spec
    )
    return FilteredTopKKernelVarlenPrefill(
        cutlass.Float32,
        bucket,
        top_k,
        num_copy_bits=num_copy_bits,
        overflow_policy=overflow_policy,
        cache_smem_values=cache_smem_values,
    )


def _rereading_rows(input_values, row_starts, row_ends, top_k: int, capacity: int) -> int:
    """How many rows overflow the kernel's shared candidate buffer, so REREAD rescans them.

    The kernel's coarse key is the score rounded to fp16, its bits flipped so a
    larger score has a smaller key, then the high byte (`to_coarse_key`). The
    threshold bin is the first whose running count, from the largest scores,
    passes `top_k`; when the bins before it hold fewer than `top_k`, its keys are
    the candidates, and more of them than `capacity` is an overflow
    (`prefix_sum_and_find_threshold_coarse`, `_filter_and_histogram_per_elem_coarse`).
    """
    rows, width = input_values.shape
    cols = torch.arange(width, device=input_values.device)
    step = max(1, 2**25 // width)
    count = 0
    for lo in range(0, rows, step):
        hi = min(lo + step, rows)
        starts, ends = row_starts[lo:hi, None].long(), row_ends[lo:hi, None].long()
        inside = ((cols >= starts) & (cols < ends)).long()
        bits = input_values[lo:hi].half().view(torch.int16).int() & 0xFFFF
        bins = torch.where(bits >= 0x8000, bits, ~bits & 0x7FFF).long() >> 8
        hist = torch.zeros(hi - lo, 256, dtype=torch.long, device=cols.device)
        hist.scatter_add_(1, bins, inside)
        del inside, bits, bins
        upto = hist.cumsum(1)
        first = (upto > top_k).int().argmax(1)
        before = torch.where(first > 0, upto.gather(1, (first - 1).clamp(min=0)[:, None])[:, 0], 0)
        in_bin = hist.gather(1, first[:, None])[:, 0]
        ranked = (ends - starts)[:, 0] > top_k
        count += int((ranked & (before < top_k) & (in_bin > capacity)).sum())
    return count


def _check_path(args, spec) -> None:
    """The kernel the width's bucket compiles, and the rows that take REREAD's
    second pass: all ranked rows (`rereads="all"`) or none (`"none"`)."""
    input_values, row_starts, row_ends = args[:3]
    kernel = _kernel(input_values.shape[1], spec)
    # Past 65536 columns the candidate index is 32 bits wide, which halves the
    # shared candidate buffer at the same shared-memory budget.
    wide = kernel.max_num_cols > 65536
    assert kernel.index_type == (cutlass.Uint32 if wide else cutlass.Uint16), kernel.index_type
    assert kernel.filtered_topk_smem_input_size == (4096 if wide else 8192), (
        kernel.filtered_topk_smem_input_size
    )
    assert kernel.enable_reread, "REREAD's second pass is compiled at this width"
    ranked = int((row_ends - row_starts > spec["top_k"]).sum())
    rereading = _rereading_rows(
        input_values, row_starts, row_ends, spec["top_k"], kernel.filtered_topk_smem_input_size
    )
    want = {"all": ranked, "none": 0}[spec["rereads"]]
    assert rereading == want, f"{rereading} of {ranked} ranked rows reread, expected {want}"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    for n, requests in enumerate(spec["blocks"]):
        args = _build(requests, spec, seed=zlib.crc32(f"{cell.why}/{n}".encode()) % 2**31)
        if "rereads" in spec:
            _check_path(args, spec)
        kw = _kwargs(spec)
        out = op(*args, **kw)
        assert out is args[3], "the op writes the caller's buffer and returns it"
        width = args[0].shape[1]
        assert _compiled_key(width, spec) in CuteDSLTopKPrefillSingleCTARunner.kernel_cache, (
            f"no CuTe DSL kernel was compiled for the {width}-column call"
        )
        _compare(out, op.reference(*args, **kw), *args[:3], exact_sets=spec["values"] != "ties")


def test_the_gate_sees_absolute_indices() -> None:
    """The defect a prefill top-k is most likely to have is answering in K-block
    columns instead of window-local ones; that is invisible on any request
    whose windows start at 0. Shift a correct answer that way on a block of
    two prompts and the gate must refuse it."""
    args = _build([(0, 1500, None), (0, 2596, None)], _SPEC, seed=31)
    out = op(*args, **_kwargs(_SPEC)).clone()
    ref = op.reference(*args, **_kwargs(_SPEC))
    _compare(out, ref, *args[:3], exact_sets=True)
    starts = args[1].long()[:, None]
    shifted = torch.where(out >= 0, out + starts, out).to(torch.int32)
    assert bool((starts > 0).any()), "the block must hold windows that start past 0"
    with pytest.raises(AssertionError):
        _compare(shifted, ref, *args[:3], exact_sets=True)


def test_the_gate_sees_a_16_bit_candidate_index() -> None:
    """Past 65536 columns the kernel carries each candidate's column in 32 bits;
    one carried in 16 would answer the column modulo 65536. Wrap a correct
    answer that way on the last 16 rows of a 1048576-column block, at the call
    site's copy width, and the gate must refuse it."""
    spec = dict(_SPEC, num_copy_bits=256)
    args = _build([(1015808, 32768, 16)], spec, seed=36)
    out = op(*args, **_kwargs(spec)).clone()
    ref = op.reference(*args, **_kwargs(spec))
    _compare(out, ref, *args[:3], exact_sets=True)
    wrapped = out & 0xFFFF
    assert bool((wrapped != out).any()), "the selection must hold columns past 65535"
    with pytest.raises(AssertionError):
        _compare(wrapped, ref, *args[:3], exact_sets=True)


@pytest.mark.parametrize(
    "requests",
    [[(0, 9216, 1820)], [(0, 32768, 512)]],
    ids=["W 9216 tile", "W 32768 tile"],
)
def test_guard_refuses_truncate_on_an_overflowing_block(requests) -> None:
    """The narrow cells reach REREAD's second pass rather than fitting in shared
    memory: on the same block "TRUNCATE", which drops the candidates that do
    not fit instead of rereading the row, selects a wrong set. The guard
    refuses it."""
    spec = dict(_SPEC, values="narrow")
    args = _build(requests, spec, seed=35)
    ref = op.reference(*args, **_kwargs(spec))
    _compare(op(*args, **_kwargs(spec)), ref, *args[:3], exact_sets=True)
    truncating = dict(_kwargs(spec), overflow_policy="TRUNCATE")
    with pytest.raises(AssertionError):
        _compare(op(*args, **truncating), ref, *args[:3], exact_sets=True)
    with pytest.raises(AssertionError, match="TRUNCATE drops candidates"):
        with validating(op):
            op(*args, **truncating)


def _padded_block(seed: int):
    """One 3000-token prompt in a padded view (row stride 3328) with distinct scores."""
    return _build([(0, 3000, None)], _SPEC, seed), _kwargs(_SPEC)


def test_guard_refuses_a_window_past_the_block() -> None:
    (logits, starts, ends, indices), kw = _padded_block(seed=32)
    width = logits.shape[1]
    ends = ends.clone()
    ends[-1] = width + 64  # still inside the row's own padding, which the test owns
    out = op(logits, starts, ends, indices, **kw)
    assert int((out[-1] >= width).sum()) == 64, "unguarded, the row ranks its padding"
    with pytest.raises(AssertionError, match="past the"):
        with validating(op):
            op(logits, starts, ends, indices, **kw)


def test_guard_refuses_a_window_before_the_row() -> None:
    (logits, starts, ends, indices), kw = _padded_block(seed=33)
    row = 2500  # a long row, so the kernel ranks rather than enumerates
    starts = starts.clone()
    starts[row] = -64  # reads the previous row's padding, which the test owns
    out = op(logits, starts, ends, indices, **kw)
    # The 64 POISON columns of the previous row win, and come back wrapped
    # through a 16-bit index as 65536 + i, outside the window.
    span = int(ends[row] - starts[row])
    assert int((out[row] >= span).sum()) == 64, "unguarded, the row ranks another row's tail"
    with pytest.raises(AssertionError, match="before its row"):
        with validating(op):
            op(logits, starts, ends, indices, **kw)


def _misaligned(logits, starts, ends, indices):
    odd = torch.zeros(logits.shape[0] + 1, logits.shape[1] + 1, device="cuda")
    view = odd[1:, : logits.shape[1]]
    assert view.data_ptr() % 32, "the view must start off a 32-byte boundary"
    view.copy_(logits)
    return view, starts, ends, indices


def _column_major(logits, starts, ends, indices):
    return logits.t().contiguous().t(), starts, ends, indices


def _int64_windows(logits, starts, ends, indices):
    return logits, starts.long(), ends.long(), indices


def _wider_output(logits, starts, ends, indices):
    return logits, starts, ends, torch.full((indices.shape[0], 4096), -7, **_i32())


def _output_column_slice(logits, starts, ends, indices):
    wide = torch.full((indices.shape[0], 4096), -7, **_i32())
    return logits, starts, ends, wide[:, : indices.shape[1]]


def _i32():
    return dict(dtype=torch.int32, device="cuda")


@pytest.mark.parametrize(
    "mangle,message",
    [
        (_misaligned, "Misaligned Tensor data on argument #0"),
        (_column_major, r"Mismatched input_values\.strides\[1\]"),
        (_int64_windows, "expected dtype=int32"),
        (_wider_output, r"Mismatched output_indices\.shape\[1\]"),
        (_output_column_slice, r"Mismatched output_indices\.strides\[0\]"),
    ],
    ids=["misaligned base", "column-major", "int64 windows", "wider output", "output col slice"],
)
def test_the_op_refuses_these_layouts_itself(mangle, message) -> None:
    """The preconditions the note leaves unguarded, because the op raises on them."""
    args, kw = _padded_block(seed=34)
    with pytest.raises(ValueError, match=message):
        op(*mangle(*args), **kw)
