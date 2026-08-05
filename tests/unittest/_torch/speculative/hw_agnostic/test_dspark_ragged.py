# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ragged (per-request) verification layout."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_ragged import (
    RaggedVerifyLayout,
    build_qo_indptr,
    choose_ragged_capture_shape,
    count_accepted_ragged,
    exceeds_captured_buckets,
    fill_padded_rows_onehot,
    round_up_to_bucket,
    row_ids_from_lens,
    scatter_ragged_to_padded,
)
from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)

BLOCK = 7


def _layout(lens, bucket=None, buckets=None):
    t = torch.tensor(lens, dtype=torch.int32)
    return RaggedVerifyLayout.from_verify_lens(
        t, graph_num_tokens=bucket, buckets=buckets, total_verify_tokens=sum(lens)
    )


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lens, expected",
    [
        ([3, 1, 4, 2], [0, 3, 4, 8, 10]),
        ([], [0]),  # empty batch still yields the leading 0
    ],
)
def test_qo_indptr_is_the_exclusive_prefix_sum(lens, expected):
    assert build_qo_indptr(torch.tensor(lens, dtype=torch.int32)).tolist() == expected


def test_build_qo_indptr_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        build_qo_indptr(torch.ones(2, 3, dtype=torch.int32))


# --------------------------------------------------------------------------
# bucket selection
# --------------------------------------------------------------------------


def test_round_up_picks_the_smallest_fitting_bucket():
    b = [8, 16, 32, 64]
    assert round_up_to_bucket(1, b) == 8
    assert round_up_to_bucket(8, b) == 8
    assert round_up_to_bucket(9, b) == 16
    assert round_up_to_bucket(64, b) == 64


def test_round_up_raises_past_the_largest_bucket():
    """Clamping would silently drop the step out of graph replay into eager."""
    with pytest.raises(ValueError, match="exceeds the largest captured bucket"):
        round_up_to_bucket(65, [8, 16, 32, 64])
    assert exceeds_captured_buckets(65, [8, 16, 32, 64])
    assert not exceeds_captured_buckets(64, [8, 16, 32, 64])


def test_layout_derives_the_bucket_from_the_host_total():
    lay = RaggedVerifyLayout.from_verify_lens(
        torch.tensor([3, 1, 4], dtype=torch.int32), buckets=[8, 16, 32], total_verify_tokens=8
    )
    assert lay.graph_num_tokens == 8
    # extend_start_loc is the offset form some attention backends want directly.
    assert torch.equal(lay.extend_start_loc, lay.qo_indptr[:-1])


def test_layout_refuses_to_sync_for_the_total():
    with pytest.raises(ValueError, match="would sync"):
        RaggedVerifyLayout.from_verify_lens(torch.tensor([3, 1], dtype=torch.int32))


# --------------------------------------------------------------------------
# padding reclamation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lens, bucket",
    [
        ([2, 1, 1], 12),  # 4 used, 8 spare: padding becomes real verification
        ([3, 2, 3], 8),  # already exact: the fill is a no-op
    ],
)
def test_fill_bucket_converts_padding_into_real_verification(lens, bucket):
    """The step already pays for the bucket -- spare slots must not be wasted."""
    lay = _layout(lens, bucket=bucket)
    filled = lay.fill_bucket(max_verify_len=BLOCK)
    assert filled.total_verify_tokens == bucket
    assert int(filled.verify_lens.sum()) == bucket
    filled.validate(max_verify_len=BLOCK, exact_fill=True)


def test_fill_bucket_spreads_over_real_requests_first():
    """Round-robin keeps the extra where survival is still plausible."""
    lay = _layout([1, 1, 1, 1], bucket=8)
    filled = lay.fill_bucket(max_verify_len=BLOCK)
    lens = filled.verify_lens.tolist()
    assert sum(lens) == 8
    assert max(lens) - min(lens) <= 1, f"unbalanced: {lens}"


def test_fill_bucket_adds_pad_rows_to_reach_the_captured_batch_size():
    """Row count must hit padded_bs; slack goes to real rows before pad rows."""
    lay = _layout([3, 2], bucket=12)
    filled = lay.fill_bucket(max_verify_len=BLOCK, padded_bs=4)
    assert filled.bs == 4
    assert filled.num_real_requests == 2
    assert filled.num_pad_requests == 2
    assert int(filled.verify_lens.sum()) == 12
    filled.validate(max_verify_len=BLOCK, exact_fill=True)
    # Real rows absorb all the slack; the pad rows stay at the 1-token minimum
    # (a pad row with 0 tokens would break qo_indptr slicing).
    lens = filled.verify_lens.tolist()
    assert lens[2:] == [1, 1]
    assert lens[0] + lens[1] == 10


def test_fill_bucket_raises_when_the_bucket_cannot_be_reached():
    """Silently returning a short batch would desync attention from the MoE."""
    lay = _layout([1, 1], bucket=100)  # capacity 2*7 = 14 < 100
    with pytest.raises(ValueError, match="exceeds what .* rows can absorb"):
        lay.fill_bucket(max_verify_len=BLOCK)


@pytest.mark.parametrize(
    "lens, bucket, padded_bs",
    [
        ([5, 5], 8, None),  # 10 real tokens already > 8
        # 8 real tokens + 6 one-token pad rows needs 14, not 10: the pad-row
        # floor counts against the bucket too. A short batch would desync
        # attention from the MoE.
        ([4, 4], 10, 8),
    ],
)
def test_fill_bucket_raises_when_the_bucket_is_too_small(lens, bucket, padded_bs):
    lay = _layout(lens, bucket=bucket)
    with pytest.raises(ValueError, match="too small"):
        lay.fill_bucket(max_verify_len=BLOCK, padded_bs=padded_bs)


def test_fill_bucket_rejects_a_padded_bs_below_the_real_count():
    lay = _layout([2, 2, 2], bucket=8)
    with pytest.raises(ValueError, match="cannot hold this batch"):
        lay.fill_bucket(max_verify_len=BLOCK, padded_bs=2)


def test_validate_exact_fill_catches_a_short_batch():
    """The check that would otherwise only surface as an attention/MoE mismatch."""
    lay = _layout([2, 2], bucket=8)  # 4 != 8
    lay.validate(max_verify_len=BLOCK)  # non-exact form still passes
    with pytest.raises(ValueError, match="do not fill the captured bucket"):
        lay.validate(max_verify_len=BLOCK, exact_fill=True)


def test_fill_bucket_needs_the_host_total():
    lay = RaggedVerifyLayout.from_verify_lens(
        torch.tensor([2, 2], dtype=torch.int32), graph_num_tokens=8
    )
    with pytest.raises(ValueError, match="host-side total"):
        lay.fill_bucket(max_verify_len=BLOCK)


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


def test_validate_rejects_a_request_that_verifies_nothing():
    """verify_len 0 means the request makes no progress -- it would stall."""
    lay = _layout([2, 0, 3], bucket=8)
    with pytest.raises(ValueError, match="anchor token"):
        lay.validate()


def test_validate_rejects_overflowing_the_bucket():
    lay = _layout([5, 5], bucket=8)
    with pytest.raises(ValueError, match="exceed the captured bucket"):
        lay.validate()


def test_validate_rejects_a_stale_indptr():
    lay = _layout([2, 3], bucket=8)
    lay.qo_indptr = torch.tensor([0, 1, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="not the prefix sum"):
        lay.validate()


# --------------------------------------------------------------------------
# end-to-end: confidence -> per-request lengths -> packed layout
# --------------------------------------------------------------------------


def test_ragged_beats_uniform_at_equal_budget():
    """Pins the reason ragged exists: heterogeneous confidence must yield
    heterogeneous lengths that pack into a valid layout and out-earn the best
    uniform split at the same token spend."""
    conf = torch.tensor(
        [
            [0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92],  # deep, worth verifying
            [0.55, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10],  # shallow
            [0.90, 0.85, 0.70, 0.55, 0.45, 0.35, 0.25],
            [0.60, 0.45, 0.35, 0.28, 0.22, 0.18, 0.12],
        ]
    )
    surv = compute_survival(conf)
    cfg = DSparkScheduleConfig(block_size=BLOCK, min_verify_len=1)
    budget, bs = 8, conf.shape[0]

    lens = schedule_verify_lens_topk(survival=surv, budget=budget, cfg=cfg)
    assert lens[0] > lens[1], "the confident request should verify deeper"

    # The lengths pack into a valid captured layout.
    total = int(lens.sum())
    lay = RaggedVerifyLayout.from_verify_lens(
        lens, buckets=[8, 16, 32, 64], total_verify_tokens=total
    )
    lay.validate(max_verify_len=BLOCK)
    assert lay.graph_num_tokens >= total

    # Same tokens verified, more expected yield than the best uniform length.
    tau_ragged = sum(float(surv[r, : int(lens[r])].sum()) for r in range(bs))
    budget_tokens = bs * cfg.min_verify_len + budget
    tau_uniform = max(
        float(surv[:, :L].sum()) for L in range(1, BLOCK + 1) if bs * L <= budget_tokens
    )
    assert int(lens.sum()) <= budget_tokens
    assert tau_ragged > tau_uniform, (
        f"ragged {tau_ragged:.2f} should beat uniform {tau_uniform:.2f} at the same token spend"
    )


# --------------------------------------------------------------------------
# packed <-> padded, and ragged acceptance counting
# --------------------------------------------------------------------------


def test_row_ids_map_each_packed_token_to_its_request():
    lens = torch.tensor([3, 1, 2], dtype=torch.int32)
    assert row_ids_from_lens(lens, total=6).tolist() == [0, 0, 0, 1, 2, 2]


def test_row_ids_requires_the_host_total():
    """``total`` is what keeps the expansion sync-free, so it cannot be optional.

    Without ``output_size`` torch reads the cumulative sum back to the host to
    size its output. DSpark's acceptance runs inside the target's captured CUDA
    graph, where that sync is illegal -- so the signature forces every caller to
    supply a total it already knows.
    """
    lens = torch.tensor([3, 1, 2], dtype=torch.int32)
    with pytest.raises(TypeError):
        row_ids_from_lens(lens)


def test_scatter_unpacks_a_flat_batch_into_rows():
    lens = torch.tensor([3, 1, 2], dtype=torch.int32)
    lay = _layout([3, 1, 2], bucket=8)
    flat = torch.tensor([10, 11, 12, 20, 30, 31])
    out = scatter_ragged_to_padded(
        flat, verify_lens=lens, qo_indptr=lay.qo_indptr, max_len=4, pad_value=-1
    )
    assert out.tolist() == [[10, 11, 12, -1], [20, -1, -1, -1], [30, 31, -1, -1]]


@pytest.mark.parametrize(
    "draft, target, lens, expected",
    [
        # the run of matches stops at the first mismatch
        ([[1, 2, 3, 4]], [[1, 2, 9, 4]], [4], [2]),
        # a request only gets credit inside its window, even when the stale
        # padding beyond it happens to match -- unmasked padding would credit
        # an acceptance the target never made (wrong output, no error).
        ([[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]], [4, 2], [4, 2]),
    ],
)
def test_accept_count_respects_each_request_window(draft, target, lens, expected):
    got = count_accepted_ragged(
        draft_tokens=torch.tensor(draft),
        target_tokens=torch.tensor(target),
        verify_lens=torch.tensor(lens),
    )
    assert got.tolist() == expected


def test_accept_count_matches_the_uniform_path_when_lengths_are_equal():
    """Equal lengths must reproduce the existing rectangular cumprod exactly."""
    torch.manual_seed(9)
    bs, L = 6, 5
    draft = torch.randint(0, 4, (bs, L))
    target = torch.randint(0, 4, (bs, L))
    lens = torch.full((bs,), L)
    ragged = count_accepted_ragged(draft_tokens=draft, target_tokens=target, verify_lens=lens)
    uniform = torch.cumprod((draft == target).int(), dim=-1).sum(1)
    assert torch.equal(ragged, uniform)


def test_accept_count_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same padded shape"):
        count_accepted_ragged(
            draft_tokens=torch.zeros(2, 3, dtype=torch.long),
            target_tokens=torch.zeros(2, 4, dtype=torch.long),
            verify_lens=torch.tensor([3, 3]),
        )


# ---------------------------------------------------------------------------
# choose_ragged_capture_shape
# ---------------------------------------------------------------------------

BS_BUCKETS = [1, 2, 4, 8, 16]
TOK_BUCKETS = [8, 16, 32, 64, 128]


def _shape(num_real, total, peers=None):
    return choose_ragged_capture_shape(
        num_real_requests=num_real,
        total_verify_tokens=total,
        bs_buckets=BS_BUCKETS,
        token_buckets=TOK_BUCKETS,
        peer_stats=peers,
    )


@pytest.mark.parametrize(
    "num_real, total, want_bs, want_bucket",
    [
        # 5 -> 8 rows; slack = 20 - 5 = 15; needed = 8 + 15 = 23 -> 32.
        (5, 20, 8, 32),
        # exact captured values are not rounded further: 8 requests with
        # slack 8 -> needed 16, both axes already captured.
        (8, 16, 8, 16),
    ],
)
def test_shape_rounds_both_axes_up_to_captured_values(num_real, total, want_bs, want_bucket):
    got = _shape(num_real=num_real, total=total)
    assert got.padded_bs == want_bs
    assert got.bucket == want_bucket


def test_every_rank_computes_the_same_shape():
    # The ADP invariant: ranks with different batches must agree. Each rank
    # passes the same peer list and must get an identical answer.
    peers = [(3, 9), (6, 30), (5, 11)]
    shapes = {_shape(num_real=real, total=total, peers=peers) for real, total in peers}
    assert len(shapes) == 1


@pytest.mark.parametrize(
    "num_real, total, peers, want_bs, want_bucket",
    [
        # This rank is small (3 requests, 9 tokens) but must size for the
        # widest peer: rows from max real (6 -> 8), slack from the max
        # (30 - 6 = 24) -> 8 + 24 = 32, exactly a captured bucket.
        (3, 9, [(3, 9), (6, 30), (5, 11)], 8, 32),
        # The widest axes can come from different ranks: rank A has the most
        # rows (7 -> 8), rank B the most slack (18) -> 8 + 18 = 26 -> 32.
        (7, 8, [(7, 8), (2, 20)], 8, 32),
    ],
)
def test_shape_is_driven_by_the_widest_rank_not_the_local_one(
    num_real, total, peers, want_bs, want_bucket
):
    got = _shape(num_real=num_real, total=total, peers=peers)
    assert got.padded_bs == want_bs
    assert got.bucket == want_bucket


@pytest.mark.parametrize(
    "num_real, total, match",
    [
        (17, 17, "exceed the largest captured batch"),  # rows overflow
        (16, 1000, "exceeds the largest captured bucket"),  # tokens overflow
        (5, 4, "cannot be below"),  # fewer tokens than requests
    ],
)
def test_shape_raises_when_the_batch_exceeds_captured_rows(num_real, total, match):
    with pytest.raises(ValueError, match=match):
        _shape(num_real=num_real, total=total)


def test_shape_requires_bs_buckets():
    with pytest.raises(ValueError, match="requires bs_buckets"):
        choose_ragged_capture_shape(
            num_real_requests=1, total_verify_tokens=1, bs_buckets=[], token_buckets=TOK_BUCKETS
        )


# ---------------------------------------------------------------------------
# group-consistent bucket choice (max_verify_len given)
#
# The regression these pin: the chooser used to pick the smallest bucket >=
# needed and leave "can MY rank decompose it into real + pad rows" to a LOCAL
# check in fit_ragged_verify_lens. Pad rows all share one window (single
# shared dummy), so decomposability is a divisibility question -- sharpest
# under a full-block pin where real rows have zero slack -- and a rank that
# failed it declined ALONE while its peers published the bucket. The ADP
# shape gate then took the whole group eager: 70/962 steps on job 2586075.
# ---------------------------------------------------------------------------

MAXLEN_BUCKETS = [16, 24, 32, 40, 48]   # padded_bs 8, tiers 1..5 -> 8*(t+1)


def _shape6(num_real, total, peers):
    return choose_ragged_capture_shape(
        num_real_requests=num_real,
        total_verify_tokens=total,
        bs_buckets=BS_BUCKETS,
        token_buckets=MAXLEN_BUCKETS,
        peer_stats=peers,
        max_verify_len=6,
    )


def test_pinned_ranks_agree_on_a_bucket_every_rank_can_decompose():
    """The job-2586075 divergence, miniaturised.

    Both ranks are pinned to the full block (floor == capacity). needed =
    8 + 25 = 33 rounds to 40, which rank (5, 30) cannot decompose (3 pad rows,
    (40-30)/3 is not an integer window) while rank (4, 24) can (4 rows at
    window 4 exactly) -- the old code split the group here. The chooser must
    now walk past 40 and land BOTH ranks on 48.
    """
    peers = [(5, 30), (4, 24)]
    shapes = {_shape6(num_real=r, total=t, peers=peers) for r, t in peers}
    assert len(shapes) == 1
    assert shapes.pop().bucket == 48


def test_slacky_batch_keeps_the_smallest_bucket():
    """Feasibility filtering must not grow the bucket when real rows have
    slack to absorb the remainder -- the ordinary trimmed batch keeps the
    exact answer the ungated chooser gave."""
    peers = [(3, 9), (6, 30), (5, 11)]
    got = _shape6(num_real=3, total=9, peers=peers)
    ungated = choose_ragged_capture_shape(
        num_real_requests=3, total_verify_tokens=9,
        bs_buckets=BS_BUCKETS, token_buckets=MAXLEN_BUCKETS,
        peer_stats=peers)
    assert got.bucket == ungated.bucket == 32
    assert got.padded_bs == ungated.padded_bs == 8


def test_group_declines_together_when_nothing_decomposes():
    """When no captured bucket works for every rank, every rank must see the
    same ValueError -- a group-consistent decline replays the uniform graph,
    where a lone decline used to split the graph keys.

    A full-block-pinned rank can always decompose the FULL bucket (pad rows
    at the max window), so total infeasibility requires a grid whose top tier
    is missing -- which capture produces for real when max_num_tokens
    truncates the ladder at the largest batch size.
    """
    truncated = [16, 24, 32, 40]   # top tier 48 not captured
    peers = [(5, 30), (4, 24)]     # pinned: 40 fails (5,30), 32 fails it too
    for r, t in peers:
        with pytest.raises(ValueError, match="decomposable by every rank"):
            choose_ragged_capture_shape(
                num_real_requests=r, total_verify_tokens=t,
                bs_buckets=BS_BUCKETS, token_buckets=truncated,
                peer_stats=peers, max_verify_len=6)


def test_uniform_batch_shape_matches_the_one_dimensional_rule():
    # With a uniform K the token total is bs * (K + 1), so the joint rule must
    # not pick anything larger than the uniform path would have needed.
    k_plus_1 = 4
    for num_real in (1, 3, 8):
        got = _shape(num_real=num_real, total=num_real * k_plus_1)
        assert got.padded_bs >= num_real
        assert got.bucket >= num_real * k_plus_1


# ---------------------------------------------------------------------------
# fill_padded_rows_onehot
# ---------------------------------------------------------------------------


def test_padding_rows_become_a_valid_distribution():
    probs = torch.zeros(2, 3, 5)
    probs[0, :2] = torch.tensor([0.2, 0.3, 0.1, 0.2, 0.2])
    probs[1, :1] = torch.tensor([0.5, 0.1, 0.1, 0.2, 0.1])
    fill_padded_rows_onehot(probs, verify_lens=torch.tensor([2, 1]))

    # Real positions untouched.
    assert torch.allclose(probs[0, 0], torch.tensor([0.2, 0.3, 0.1, 0.2, 0.2]))
    # Padding positions sum to 1 instead of 0.
    assert torch.allclose(probs[0, 2].sum(), torch.tensor(1.0))
    assert torch.allclose(probs[1, 1].sum(), torch.tensor(1.0))
    assert torch.allclose(probs[1, 2].sum(), torch.tensor(1.0))

    # Full-length rows are left entirely alone (guards the >= boundary).
    full = torch.full((2, 3, 4), 0.25)
    before = full.clone()
    fill_padded_rows_onehot(full, verify_lens=torch.tensor([3, 3]))
    assert torch.equal(full, before)


# ---------------------------------------------------------------------------
# Bucket fitting: the runtime batch must land on a captured token total
# ---------------------------------------------------------------------------


def _fit(real_token_lens, bucket, padded_bs, max_verify_len):
    """Mirror of ModelEngine.fit_ragged_verify_lens' core."""
    lens = torch.tensor(real_token_lens, dtype=torch.int32)
    layout = RaggedVerifyLayout.from_verify_lens(
        lens, graph_num_tokens=bucket, total_verify_tokens=sum(real_token_lens)
    )
    return layout.fill_bucket(max_verify_len=max_verify_len, padded_bs=padded_bs)


def test_fit_never_exceeds_the_per_request_ceiling():
    got = _fit([1, 1, 1], bucket=12, padded_bs=4, max_verify_len=3)
    assert max(got.verify_lens.tolist()) <= 3


def test_warmup_split_reaches_any_captured_bucket():
    # Capture has to hit the bucket exactly or the graph is keyed on a total no
    # runtime batch produces, and every ragged step silently runs eager.
    tiers = [1, 3, 5]
    for bs in (1, 2, 8):
        for t in tiers:
            bucket = bs * (t + 1)
            got = _fit([1] * bs, bucket=bucket, padded_bs=bs, max_verify_len=1 + tiers[-1])
            assert int(got.verify_lens.sum()) == bucket


def _pad_len(*, bucket, n_real, n_pad, max_verify_len, floor=None):
    """Mirror of the pad-row window ModelEngine.fit_ragged_verify_lens derives.

    Pad rows take the one-token minimum so real requests keep the slack, and
    only grow when the real rows physically cannot absorb the bucket. Returns
    None when no integer window leaves the real rows a reachable target -- the
    scheduler then declines and the step stays uniform.
    """
    real_capacity = n_real * max_verify_len
    floor = n_real if floor is None else floor
    if n_pad == 0:
        pad_len = 1
    else:
        lo = max(1, -(-(bucket - real_capacity) // n_pad))  # ceil division
        hi = min(max_verify_len, (bucket - floor) // n_pad)
        if lo > hi:
            return None
        pad_len = lo
    real_target = bucket - n_pad * pad_len
    if real_target < floor or real_target > real_capacity:
        return None
    return pad_len


@pytest.mark.parametrize(
    "bucket, expected_pad_len",
    [
        # 3 real rows capped at 6 hold up to 18; bucket 18 minus 5 one-token
        # pad rows leaves 13, which the real rows absorb -- pads stay minimal.
        (18, 1),
        # The case that used to have no solution: bucket 24 with 5 one-token
        # pad rows would demand 19 from real rows that cap at 18, so the pad
        # rows must absorb what the real rows cannot.
        (24, 2),
    ],
)
def test_pad_rows_grow_when_the_real_rows_cannot_absorb_the_bucket(bucket, expected_pad_len):
    padded_bs, max_verify_len = 8, 6
    real = [2, 3, 2]
    n_pad = padded_bs - len(real)
    pad_len = _pad_len(bucket=bucket, n_real=len(real), n_pad=n_pad, max_verify_len=max_verify_len)
    assert pad_len == expected_pad_len

    real_target = bucket - n_pad * pad_len
    assert sum(real) <= real_target <= len(real) * max_verify_len
    got = _fit(real, bucket=real_target, padded_bs=len(real), max_verify_len=max_verify_len)
    # The total the forward sees must be exactly the captured bucket.
    assert int(got.verify_lens.sum()) + n_pad * pad_len == bucket


def test_pad_length_never_yields_an_infeasible_real_target():
    """Exhaustive sweep of every shape the scheduler can actually produce.

    A wrong pad window is the worst failure mode here: the batch's token total
    silently misses every captured bucket and the step drops out of graph
    replay into eager. So sweep the real bucket grid (one bucket per tier at
    each captured batch size), pick the bucket the way the scheduler does, and
    require that the derived split either reconstructs the bucket exactly or
    declines outright.
    """
    max_verify_len = 6
    checked = declined = 0
    for padded_bs in (1, 2, 4, 8, 16, 32, 64):
        grid = sorted({padded_bs * (t + 1) for t in range(max_verify_len)})
        for n_real in range(1, padded_bs + 1):
            n_pad = padded_bs - n_real
            for floor in range(n_real, n_real * max_verify_len + 1):
                # The scheduler rounds up from "what the batch already needs
                # plus one token per pad row".
                fits = [b for b in grid if b >= floor + n_pad]
                if not fits:
                    continue
                bucket = fits[0]
                checked += 1
                pad_len = _pad_len(
                    bucket=bucket,
                    n_real=n_real,
                    n_pad=n_pad,
                    max_verify_len=max_verify_len,
                    floor=floor,
                )
                if pad_len is None:
                    declined += 1
                    continue
                real_target = bucket - n_pad * pad_len
                assert floor <= real_target <= n_real * max_verify_len, (
                    f"padded_bs={padded_bs} n_real={n_real} bucket={bucket} "
                    f"floor={floor} pad_len={pad_len} -> {real_target}"
                )
                assert real_target + n_pad * pad_len == bucket

    assert checked > 10000, f"sweep too small to be meaningful: {checked}"
    # Declining is safe (the step stays uniform) but should stay rare, or the
    # bucket grid is too coarse to be worth capturing.
    assert declined / checked < 0.15, f"{declined}/{checked} shapes have no valid split"


def test_extra_peer_stat_fields_are_ignored_here():
    """The payload grows; this function's contract does not.

    peer_stats gained a third element -- the group's answer to whether row
    padding will take the cross-rank maximum at all -- which the caller uses
    before reaching this function. A two-element unpack here turned that into
    a ValueError on the first ragged step.
    """
    two = _shape(num_real=3, total=12, peers=[[3, 12], [4, 16]])
    three = _shape(num_real=3, total=12, peers=[[3, 12, 1], [4, 16, 1]])
    assert (three.padded_bs, three.bucket) == (two.padded_bs, two.bucket)
