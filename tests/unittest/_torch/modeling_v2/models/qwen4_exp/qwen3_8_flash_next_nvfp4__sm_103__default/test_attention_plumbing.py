# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's attention plumbing around its kernels.

* **The FP8 KV range probe** (`TRTLLM_MODELING_V2_VALIDATE` over an fp8 pool). The pool stores
  K/V at the unit scale, so a magnitude past the e4m3 maximum saturates. The probe reports the
  largest magnitude of the first eager forward's attention layers, once, and never reads a
  tensor under CUDA graph capture.
* **The context pass over an NVFP4 pool, a group of requests at a time**, at one rank (24 / 2
  heads, the family's row-tile pass) and at a tp4 / tep4 rank (6 / 1 heads, the target's). The
  context rows read a bf16 scratch of their requests' pages (`_nvfp4_context_plan`), filled and
  attended one group of consecutive requests at a time, so it holds the largest group's pages
  rather than the step's. Under a budget of a few pages, with one request longer than the
  budget, a prefill step's rows match one group over the whole step bitwise: its requests'
  rows come in multiples of the row-tile pass's groups (10 rows at 24 / 2 heads, 20 at 6 / 1),
  so a launch per group splits none of them and every row takes the same pass as in the one
  launch. At 6 / 1 heads the pass agrees with the NVFP4 per-row prefill kernel.
* **MTP IndexShare.** The draft selection (`_selection`) in its fused and its two-launch form,
  the capture of each request's draft-extend row into the shared buffers (decode and
  mixed-batch forms), and the reuse entry's order over canonical captures. The targets are
  built without weights: the capture methods read only their arguments, the metadata and the
  core's draft binding.
* **Score tables past the int32 range.** A part whose rows times the table's columns would
  reach 2^31 is scored and selected in chunks under it, each row once, with its part's
  context flag; at the checkpoint's `max_seq_len` (65568 columns) 32752 rows are one call.
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_mtp_reuse_topk import (  # noqa: E501
    qsa_mtp_reuse_topk as reuse,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs on a CUDA device")
DEV = "cuda"


# ---- the FP8 KV range probe ---------------------------------------------------------------

LAST = 11


def _kv(peak: float = 1.0):
    """K and V of one layer, 5 rows x 2 heads x 8 dims, with `peak` as V's largest value."""
    k = torch.linspace(-1.0, 1.0, 80, device=DEV).reshape(5, 2, 8).to(torch.bfloat16)
    v = torch.linspace(-0.5, 0.5, 80, device=DEV).reshape(5, 2, 8).to(torch.bfloat16)
    v[3, 1, 4] = peak
    return k, v


def _run_forward(probe, peaks):
    """One forward's attention layers 3, 7 and LAST; returns each `observe` result."""
    return [probe.observe(layer, *_kv(peak)) for layer, peak in zip((3, 7, LAST), peaks)]


@_cuda
def test_a_forward_within_the_e4m3_range_is_silent(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(M, "logger", logger)

    assert _run_forward(M._KvRangeProbe(LAST), (1.0, 300.0, 448.0)) == [True, True, False]

    logger.warning.assert_not_called()


@_cuda
@pytest.mark.parametrize("peak", [-600.0, float("nan")])
def test_a_magnitude_past_the_e4m3_maximum_is_reported_once(monkeypatch, peak):
    """A negative value counts by its magnitude; a NaN counts as out of range."""
    logger = Mock()
    monkeypatch.setattr(M, "logger", logger)

    # The oversized value is in a middle layer: the report waits for the forward's last one.
    assert _run_forward(M._KvRangeProbe(LAST), (1.0, peak, 1.0)) == [True, True, False]

    logger.warning.assert_called_once()
    assert "e4m3 maximum 448" in logger.warning.call_args.args[0]


@pytest.mark.cpu_only
def test_the_probe_runs_only_when_asked_over_an_fp8_pool():
    core = SimpleNamespace(quant_mode=M.QM_FP8_KV_CACHE, attn_layers=[3, 7, LAST])
    assert M._KvRangeProbe.for_target(core, True)._last == LAST
    assert M._KvRangeProbe.for_target(core, False) is None
    for quant_mode in (0, M.QM_NVFP4_KV_CACHE):
        other = SimpleNamespace(quant_mode=quant_mode, attn_layers=core.attn_layers)
        assert M._KvRangeProbe.for_target(other, True) is None


@_cuda
def test_observing_under_capture_reads_nothing(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(M, "logger", logger)
    probe = M._KvRangeProbe(LAST)
    k, v = _kv(1000.0)
    scratch = torch.zeros(1, device=DEV)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scratch.add_(1)  # warm the allocator off the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    kept = []
    with torch.cuda.graph(graph):
        scratch.add_(1)
        # The last layer, with an out-of-range value: a read here would fail the capture
        # (a host sync) and a report would fire.
        kept.append(probe.observe(LAST, k, v))

    assert kept == [True], "a captured forward does not end the probe"
    logger.warning.assert_not_called()
    # The first eager forward after capture is what reports.
    assert _run_forward(probe, (1.0, 1.0, 1000.0)) == [True, True, False]
    logger.warning.assert_called_once()


# ---- the context pass over an NVFP4 pool ---------------------------------------------------

TPB, D, RATIO, TOPK = 32, 256, 4, 2048
# (cached, new) tokens a context request: 10, 30, 1 (2 at tp4), 47 and 4 pages. Each request's
# rows are a multiple of the row-tile pass's groups: 10 rows at 24 / 2 heads, 20 at 6 / 1.
SEGMENTS = ((0, 300), (900, 40), (20, 10), (1400, 100), (64, 60))
TP4_SEGMENTS = ((0, 300), (900, 40), (20, 20), (1400, 100), (64, 60))
# (one rank, query heads, KV heads, segments, the whole step's pages)
GEOMETRIES = {
    "tp1": (True, 24, 2, SEGMENTS, 92),
    "tp4": (False, 6, 1, TP4_SEGMENTS, 93),
}
BUDGET_PAGES = 32  # request 3's 47 pages form a group of their own


def _context_step(geometry: str, seed: int = 5):
    one_rank, qh, hkv, segments, _ = GEOMETRIES[geometry]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    kv_lens = [s + n for s, n in segments]
    pages = [math.ceil(k / TPB) for k in kv_lens]
    pool_pages = sum(pages) + 3
    shape = (pool_pages, 2, hkv, TPB)
    data = torch.randint(0, 256, (*shape, D // 2), generator=gen, device=DEV).to(torch.uint8)
    scales = torch.randint(0x08, 0x31, (*shape, D // 16), generator=gen, device=DEV)
    scales = scales.to(torch.uint8)
    slots = 2 * len(segments)
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))
    slot_of = slot_of[: len(segments)].to(DEV, torch.int32)
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, max(pages)), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), pages):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    new = torch.tensor([n for _, n in segments], dtype=torch.int32)
    md = SimpleNamespace(
        num_contexts=len(segments),
        kv_lens_runtime=torch.tensor(kv_lens, dtype=torch.int32),
        kv_lens_cuda_runtime=torch.tensor(kv_lens, dtype=torch.int32, device=DEV),
        seq_lens=new,
        seq_lens_cuda=new.to(DEV),
    )
    logical = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    rows = logical.numel()
    # Every visible token, as the expansion lays out a selection under its top-k.
    column = torch.arange(TOPK + RATIO - 1, device=DEV)[None, :]
    selected = torch.where(column <= logical[:, None], column, -1).to(torch.int32)
    core = SimpleNamespace(
        one_rank=one_rank,
        heads_kv=hkv,
        head_dim=D,
        q_width=qh * D,
        kv_scale_quant_orig=torch.ones(3, dtype=torch.float32, device=DEV),
        softmax_scale=D**-0.5,
        compress_ratio=RATIO,
    )
    kv = SimpleNamespace(k=data[:, 0], v=data[:, 1], k_scale=scales[:, 0], v_scale=scales[:, 1])
    return dict(
        core=core,
        kv=kv,
        md=md,
        block_table=block_table,
        req_idx=slot_of.repeat_interleave(new.to(DEV).long()),
        q=torch.randn(rows, qh, D, generator=gen, device=DEV).to(torch.bfloat16),
        selected=selected,
        logical=logical.to(torch.int32),
        rows=rows,
    )


def _context_pass(step):
    plan = M._nvfp4_context_plan(
        step["core"], step["md"], step["req_idx"], step["block_table"], TPB
    )
    out = torch.empty(step["rows"], step["core"].q_width, dtype=torch.bfloat16, device=DEV)
    M._sparse_gqa_nvfp4_context(
        step["core"],
        step["kv"],
        plan,
        step["q"],
        step["block_table"],
        step["selected"],
        step["req_idx"],
        step["logical"],
        TPB,
        step["rows"],
        out,
        1,
    )
    return plan, out


@_cuda
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_grouped_context_pass_equals_one_group(monkeypatch, geometry) -> None:
    step = _context_step(geometry)
    whole, want = _context_pass(step)
    assert len(whole[0]) == 1 and whole[2].shape[0] == GEOMETRIES[geometry][4]

    page_bytes = 2 * step["core"].heads_kv * TPB * D * 2
    monkeypatch.setattr(M, "_NVFP4_CONTEXT_SCRATCH_BYTES", BUDGET_PAGES * page_bytes)
    (groups, _, k_scratch, _), got = _context_pass(step)
    # Requests [0], [1, 2], [3] (over the budget, alone), [4].
    ends = [0, 300]
    for _, n in GEOMETRIES[geometry][3][1:]:
        ends.append(ends[-1] + n)
    del ends[2]  # requests 1 and 2 share a group
    assert [(g[0], g[1]) for g in groups] == list(zip(ends, ends[1:]))
    assert k_scratch.shape[0] == 47
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@_cuda
def test_the_tp4_context_pass_matches_the_per_row_kernel() -> None:
    """At a tp4 / tep4 rank (6 / 1 heads) the context rows take the target's row-tile pass over
    the dequantized scratch, and agree with the NVFP4 per-row prefill kernel over the pool."""
    step = _context_step("tp4")
    _, got = _context_pass(step)
    want = M._sparse_gqa(
        step["core"],
        step["kv"],
        step["q"],
        step["block_table"],
        step["selected"],
        step["req_idx"],
        TPB,
        step["logical"],
    ).reshape(got.shape)
    err = (got.float() - want.float()).norm() / want.float().norm()
    assert err < 1e-2, err


# ---- MTP IndexShare -----------------------------------------------------------------------

K = 512
SLOT = 1
COLS = 2080


def _target(cls):
    target = object.__new__(cls)
    target.core = SimpleNamespace(_draft_bound=SimpleNamespace(share_slot=SLOT))
    return target


def _metadata(batch: int, accepted, seq_lens=None, num_contexts: int = 0):
    return SimpleNamespace(
        num_seqs=batch,
        num_contexts=num_contexts,
        seq_lens_cuda=None
        if seq_lens is None
        else torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        qsa_mtp_num_accepted=torch.tensor(accepted, dtype=torch.int32, device=DEV),
        qsa_shared_topk_indices=torch.full((3, 8, K), -7, dtype=torch.int32, device=DEV),
        qsa_shared_topk_visible_blocks=torch.full((3, 8), -7, dtype=torch.int32, device=DEV),
        qsa_shared_topk_captured_slots=set(),
    )


def _scores_and_windows(rows: int, gen: torch.Generator):
    """Relu-shaped scores (half zeros: ties at the budget's last score whenever fewer than
    K are positive) over windows of 0 to 3K groups; the columns past a window are NaN."""
    ends = torch.randint(0, COLS + 1, (rows,), generator=gen).to(torch.int32).to(DEV)
    scores = torch.relu(torch.randn(rows, COLS, generator=gen)).to(DEV)
    scores = scores * (torch.rand(rows, COLS, generator=gen).to(DEV) < 0.3)
    outside = torch.arange(COLS, device=DEV)[None, :] >= ends[:, None]
    scores.masked_fill_(outside, float("nan"))
    return scores, torch.zeros(rows, dtype=torch.int32, device=DEV), ends


def _select(scores, starts, ends) -> torch.Tensor:
    out = torch.empty((scores.shape[0], K), dtype=torch.int32, device=DEV)
    M._select(SimpleNamespace(block_topk=K), scores, starts, ends, out)
    return out


@_cuda
@pytest.mark.parametrize("width", [2080, 65600])
@pytest.mark.parametrize("fused", [False, True])
def test_selection_matches_the_canonical_composition(width, fused, monkeypatch):
    """The draft selection keeps the canonical tie order and expansion in the fused launch
    and in the selection and the expansion apart."""
    positions = torch.tensor([2043, 2047, 84000, 99999], dtype=torch.int64, device=DEV)
    visible = ((positions + 1) // 4).to(torch.int32)
    visible.clamp_(max=width)
    positions = torch.minimum(positions, (visible.to(torch.int64) * 4 + 2))
    lengths = (positions + 3).to(torch.int32)
    starts = torch.zeros_like(visible)
    generator = torch.Generator(device=DEV).manual_seed(85)
    scores = torch.relu(torch.randn((4, width), generator=generator, device=DEV) - 1.8)
    cols = torch.arange(width, device=DEV)
    scores.masked_fill_(cols[None, :] >= visible[:, None], float("nan"))
    expected = torch.empty((4, K), dtype=torch.int32, device=DEV)
    core = SimpleNamespace(block_topk=K, compress_ratio=4, token_topk=2048)
    M._select(core, scores, starts, visible, expected)
    expected_tokens = M.qsa_expand_block_indices(
        expected, positions, lengths, compress_ratio=4, token_topk=2048
    )
    if not fused:
        monkeypatch.setattr(M, "_topk_expand_supports", lambda scores: False)
    result = torch.full_like(expected, -7)
    selected = torch.full_like(expected_tokens, -7)
    M._selection(core, scores, visible, positions, lengths, result, starts, selected)
    assert torch.equal(result, expected)
    assert torch.equal(selected, expected_tokens)


@_cuda
def test_decode_capture_at_draft_seven_takes_row_s_times_8_plus_a_minus_1():
    """max_draft_len 7: eight rows per request, accepted 1..8, so the captured row is
    `s * 8 + a - 1` -- the first and the last row of a block included."""
    p, batch = 8, 4
    accepted = [1, 8, 5, 3]
    topk = torch.arange(batch * p * K, dtype=torch.int32, device=DEV).reshape(batch * p, K)
    visible = torch.arange(300, 300 + batch * p, dtype=torch.int32, device=DEV)
    md = _metadata(batch, accepted)

    _target(M.DecodeTarget)._capture_selection(md, topk, visible, batch, p)

    rows = [s * p + a - 1 for s, a in enumerate(accepted)]
    assert rows == [0, 15, 20, 26]
    torch.testing.assert_close(md.qsa_shared_topk_indices[SLOT, :batch], topk[rows])
    torch.testing.assert_close(md.qsa_shared_topk_visible_blocks[SLOT, :batch], visible[rows])
    # Other slots and rows past the batch are untouched.
    assert torch.all(md.qsa_shared_topk_indices[SLOT, batch:] == -7)
    assert torch.all(md.qsa_shared_topk_indices[0] == -7)
    assert md.qsa_shared_topk_captured_slots == {SLOT}


@_cuda
@pytest.mark.parametrize("compact", [False, True])
def test_prefill_capture_at_draft_seven(compact):
    """A mixed batch at max_draft_len 7: two context chunks of 5 and 3 rows, then three
    generation requests of 8 rows with 8, 1 and 6 accepted. A context request captures its
    chunk's last row whatever accepted count the worker left there. A `compact` selection
    (the draft-extend step's kept rows under attention TP) holds each context request's last
    row, then every generation row."""
    seq_lens = [5, 3, 8, 8, 8]
    accepted = [2, 3, 8, 1, 6]
    rows = [4, 7, 8 + 7, 16 + 0, 24 + 5]
    if compact:
        seq_lens_rows = [1, 1, 8, 8, 8]
        rows = [0, 1, 2 + 7, 10 + 0, 18 + 5]
    else:
        seq_lens_rows = seq_lens
    rows_total = sum(seq_lens_rows)
    topk = torch.arange(rows_total * K, dtype=torch.int32, device=DEV).reshape(rows_total, K)
    visible = torch.arange(400, 400 + rows_total, dtype=torch.int32, device=DEV)
    md = _metadata(5, accepted, seq_lens=seq_lens, num_contexts=2)
    md.num_ctx_tokens = 8

    _target(M.PrefillTarget)._capture_selection(md, topk, visible, compact=compact)

    torch.testing.assert_close(md.qsa_shared_topk_indices[SLOT, :5], topk[rows])
    torch.testing.assert_close(md.qsa_shared_topk_visible_blocks[SLOT, :5], visible[rows])
    assert md.qsa_shared_topk_captured_slots == {SLOT}


def _reuse_case(fresh: int):
    """A canonical capture of 16 requests and the reuse call's arguments for it."""
    gen = torch.Generator().manual_seed(11 + fresh)
    batch = 16
    scores, starts, ends = _scores_and_windows(batch, gen)
    captured_visible = ends.cpu()
    captured = _select(scores, starts, ends)
    shared = torch.full((8, 32, K), -1, dtype=torch.int32, device=DEV)
    shared[SLOT, :batch] = captured
    shared_visible = torch.zeros((8, 32), dtype=torch.int32, device=DEV)
    shared_visible[SLOT, :batch] = captured_visible.to(DEV)
    args = dict(
        shared_indices=shared[SLOT],
        shared_visible_blocks=shared_visible[SLOT],
        request_indices=torch.arange(batch, dtype=torch.int32, device=DEV),
        visible_blocks=(captured_visible + fresh).to(DEV),
        output=torch.empty((batch, K), dtype=torch.int32, device=DEV),
    )
    return args, captured_visible


@_cuda
def test_reuse_of_a_canonical_capture_stays_ascending():
    """A canonical capture plus the groups completed since is still ascending: appended ids
    are newer than every captured one, and a full row gives up its oldest ids to them, so
    the newest captured groups -- the ones next to the new ones -- are kept."""
    for fresh in (0, 2):
        args, captured_visible = _reuse_case(fresh)

        out = reuse(**args)

        assert torch.equal(out, reuse.reference(**args))
        for row, shared_row, vis in zip(
            out.cpu().tolist(), args["shared_indices"].cpu().tolist(), captured_visible.tolist()
        ):
            ids = [i for i in row if i >= 0]
            assert row == ids + [-1] * (K - len(ids)), "padding must stay last"
            assert ids == sorted(ids) and len(set(ids)) == len(ids)
            assert len(ids) == min(vis + fresh, K)
            assert all(i < vis + fresh for i in ids)
            captured = [i for i in shared_row if i >= 0]
            kept = captured[max(0, len(captured) + fresh - K) :]
            assert ids == kept + list(range(vis, vis + fresh))


# ---- score tables past the int32 range ----------------------------------------------------


def _chunked_calls(monkeypatch, rows: int, pages: int, parts, limit=None):
    """`_score_and_select` over `rows` rows and a `pages`-page table (32 tokens, compression
    4), with the scoring and the selection recorded: `(first row, rows, context_rows)` per
    scoring call, and `selected` with each row's position in its first column."""
    columns = pages * 32 // 4
    calls = []

    def scores_op(*, q, query_positions, context_rows, **_):
        calls.append((int(query_positions[0]), q.shape[0], context_rows))
        assert q.shape[0] * columns < (limit or M._SCORE_TABLE_ELEMENTS)
        return q[:, 0, :1].expand(q.shape[0], columns)

    def selection(core, scores, visible, logical, seq_lens, topk, starts, out, live_cols):
        assert scores.shape == (logical.shape[0], columns) and live_cols == 7
        out.copy_(logical[:, None].to(out.dtype).expand_as(out))

    monkeypatch.setattr(M, "qsa_paged_index_scores", scores_op)
    monkeypatch.setattr(M, "_selection", selection)
    if limit is not None:
        monkeypatch.setattr(M, "_SCORE_TABLE_ELEMENTS", limit)
    logical = torch.arange(rows)
    selected = torch.full((rows, 3), -1, dtype=torch.int32)
    table = torch.zeros(2, pages, dtype=torch.int32)
    rows_q = logical.float()[:, None, None]
    M._score_and_select(
        SimpleNamespace(compress_ratio=4),
        None,
        rows_q,
        table,
        logical,
        logical,
        32,
        parts,
        logical,
        logical,
        logical,
        logical,
        selected,
        7,
    )
    assert torch.equal(selected[:, 0], logical.to(torch.int32)), "every row selected once"
    return calls


@pytest.mark.cpu_only
def test_a_part_past_the_score_table_bound_goes_in_chunks(monkeypatch) -> None:
    # 4 pages: 32 columns; a bound of 5 x 32 + 1 elements takes 5 rows a call.
    calls = _chunked_calls(monkeypatch, 13, 4, [(0, 11, True), (11, 13, False)], 5 * 32 + 1)
    assert calls == [(0, 5, True), (5, 5, True), (10, 1, True), (11, 2, False)]


@pytest.mark.cpu_only
def test_at_max_seq_len_262144_a_call_takes_32752_rows(monkeypatch) -> None:
    # 8196 pages (262144 tokens plus the generation lead, padded to 4 pages): 65568 columns.
    assert 32752 * 65568 < 2**31 <= 32753 * 65568
    assert _chunked_calls(monkeypatch, 8192, 8196, [(0, 8192, True)]) == [(0, 8192, True)]
    assert _chunked_calls(monkeypatch, 40000, 8196, [(0, 40000, True)]) == [
        (0, 32752, True),
        (32752, 7248, True),
    ]
