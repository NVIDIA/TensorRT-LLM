# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_gvr_topk_decode and rubin_gvr_topk_prefill entries (self-sampling GVR Top-K,
deterministic kernels) and of rubin_topk_decode_cute (the radix Top-K of the helix key shard).

CPU: each reference against a plain sort on small rows (ties to the smaller column; the identity
and -1 tail of short rows), the guards. GPU (sm_107): every cell of each entry against its
reference as sets (the kernels' output order is their own); ``rubin_topk_decode_cute`` by the
multiset of selected values (its boundary ties may go either way).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_gvr_topk_decode import (  # noqa: E501
    rubin_gvr_topk_decode as decode_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_gvr_topk_prefill import (  # noqa: E501
    rubin_gvr_topk_prefill as prefill_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_topk_decode_cute import (  # noqa: E501
    rubin_topk_decode_cute as cute_op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)


def _sets(rows: torch.Tensor):
    return [set(r[r >= 0].tolist()) for r in rows.cpu()]


def _logits(rows, width, gen, ties=False, align=4):
    stride = -(-width // align) * align
    base = torch.empty(rows, stride)
    if ties:
        base.copy_(torch.randint(-3, 4, (rows, stride), generator=gen).float())
    else:
        base.copy_(torch.randn(rows, stride, generator=gen))
    return base[:, :width]


# --- CPU: references -------------------------------------------------------------------------


def test_decode_reference_short_rows_and_ties():
    logits = torch.tensor([[1.0, 5.0, 5.0, 2.0, 0.0, 5.0]] * 2)
    kv_lens = torch.tensor([6], dtype=torch.int32)
    out = torch.full((2, 2), 7, dtype=torch.int32)
    ref = decode_op.reference(logits, kv_lens, out, 2, 8, True)
    # Row 0 sees 5 keys: the two 5.0 at columns 1, 2 (column 5 is outside); row 1 sees all 6.
    assert ref[0].tolist() == [1, 2] and ref[1].tolist() == [1, 2]
    assert (out == 7).all(), "the reference leaves indices untouched"
    short = decode_op.reference(
        logits,
        torch.tensor([3], dtype=torch.int32),
        torch.empty(2, 4, dtype=torch.int32),
        2,
        8,
        True,
    )
    assert short[0].tolist() == [0, 1, -1, -1] and short[1].tolist() == [0, 1, 2, -1]


def test_prefill_reference_windows_are_local():
    logits = torch.arange(10.0).repeat(3, 1)
    starts = torch.tensor([0, 2, 5], dtype=torch.int32)
    ends = torch.tensor([10, 6, 7], dtype=torch.int32)
    ref = prefill_op.reference(logits, starts, ends, torch.empty(3, 3, dtype=torch.int32), True)
    assert ref[0].tolist() == [7, 8, 9]
    assert ref[1].tolist() == [1, 2, 3]  # columns 3, 4, 5 of the window [2, 6)
    assert ref[2].tolist() == [0, 1, -1]  # a window of 2: the identity, then -1


def test_cute_reference_per_row_lengths():
    logits = torch.tensor([[3.0, 1.0, 2.0, 9.0], [4.0, 0.0, 0.0, 0.0]])
    lens = torch.tensor([3, 1], dtype=torch.int32)
    ref = cute_op.reference(logits, lens, torch.empty(2, 2, dtype=torch.int32), 2, 1)
    assert ref[0].tolist() == [0, 2] and ref[1].tolist() == [0, -1]


def test_guards():
    logits = torch.zeros(1, 6)
    with pytest.raises(AssertionError, match="float4"):
        decode_op.is_valid(
            logits,
            torch.tensor([6], dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int32),
            1,
            8,
            True,
        )
    with pytest.raises(AssertionError, match="past max_seq_len"):
        decode_op.is_valid(
            torch.zeros(2, 8),
            torch.tensor([9], dtype=torch.int32),
            torch.zeros(2, 4, dtype=torch.int32),
            2,
            8,
            True,
        )
    with pytest.raises(AssertionError, match="inside the logits row"):
        prefill_op.is_valid(
            torch.zeros(2, 8),
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([8, 9], dtype=torch.int32),
            torch.zeros(2, 4, dtype=torch.int32),
            True,
        )
    with pytest.raises(AssertionError, match="past the logits' width"):
        cute_op.is_valid(
            torch.zeros(2, 8),
            torch.tensor([9, 1], dtype=torch.int32),
            torch.zeros(2, 4, dtype=torch.int32),
            4,
            1,
        )


def test_reference_of_each_cell_shape_runs_on_cpu():
    """The cells' specs build valid inputs (shapes, lengths inside the envelope)."""
    for cell in decode_op.CELLS:
        s = cell.spec
        assert s["kv_lens"][1] <= s["max_seq_len"] and s["batch"] * s["next_n"] > 0
    for cell in prefill_op.CELLS:
        s = cell.spec
        assert all(0 <= a <= b <= s["width"] for a, b in s["windows"])
    for cell in cute_op.CELLS:
        s = cell.spec
        assert s["row_lens"][1] <= s["width"]


# --- GPU: the certified cells ----------------------------------------------------------------


@on_sm107
@pytest.mark.parametrize("cell", decode_op.CELLS, ids=[c.why[:48] for c in decode_op.CELLS])
def test_decode_certified_cells(cell, certifying):  # noqa: F811
    certifying(decode_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["batch"] * 31 + s["next_n"])
    lo, hi = s["kv_lens"]
    kv_lens = torch.randint(lo, hi + 1, (s["batch"],), generator=gen, dtype=torch.int32)
    rows = s["batch"] * s["next_n"]
    width = -(-s["max_seq_len"] // 256) * 256
    logits = _logits(rows, width, gen, s.get("data") == "ties", 256)[:, : s["max_seq_len"]].cuda()
    out = torch.full((rows, s["top_k"]), -7, dtype=torch.int32, device="cuda")
    decode_op(logits, kv_lens.cuda(), out, s["next_n"], s["max_seq_len"], True)
    ref = decode_op.reference(logits, kv_lens, out, s["next_n"], s["max_seq_len"], True)
    assert _sets(out) == _sets(ref)


@on_sm107
@pytest.mark.parametrize("cell", prefill_op.CELLS, ids=[c.why[:48] for c in prefill_op.CELLS])
def test_prefill_certified_cells(cell, certifying):  # noqa: F811
    certifying(prefill_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(len(s["windows"]))
    logits = _logits(len(s["windows"]), s["width"], gen, s.get("data") == "ties").cuda()
    starts = torch.tensor([a for a, _ in s["windows"]], dtype=torch.int32).cuda()
    ends = torch.tensor([b for _, b in s["windows"]], dtype=torch.int32).cuda()
    out = torch.full((len(s["windows"]), s["top_k"]), -7, dtype=torch.int32, device="cuda")
    prefill_op(logits, starts, ends, out, True)
    ref = prefill_op.reference(logits, starts, ends, out, True)
    assert _sets(out) == _sets(ref)


@on_sm107
@pytest.mark.parametrize("cell", cute_op.CELLS, ids=[c.why[:48] for c in cute_op.CELLS])
def test_cute_certified_cells(cell, certifying):  # noqa: F811
    certifying(cute_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["rows"])
    lo, hi = s["row_lens"]
    lens = torch.randint(lo, hi + 1, (s["rows"],), generator=gen, dtype=torch.int32)
    logits = _logits(s["rows"], s["width"], gen).cuda()
    out = torch.full((s["rows"], s["top_k"]), -7, dtype=torch.int32, device="cuda")
    cute_op(logits, lens.cuda(), out, s["top_k"], s["next_n"])
    ref = cute_op.reference(logits, lens, out, s["top_k"], s["next_n"])
    for r in range(s["rows"]):
        got, want = out[r][out[r] >= 0].long().cpu(), ref[r][ref[r] >= 0].long().cpu()
        assert got.numel() == want.numel()
        row = logits[r].cpu()
        assert torch.equal(torch.sort(row[got]).values, torch.sort(row[want]).values)
