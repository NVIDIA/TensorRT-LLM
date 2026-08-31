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
"""DSpark ragged verification layout and packing tests."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_ragged import (
    RaggedVerifyLayout,
    build_qo_indptr,
    build_row_maps_device,
    choose_ragged_capture_shape,
    count_accepted_ragged,
    fill_bucket_device,
    fill_padded_rows_onehot,
    resolve_ragged_pad_split,
    round_up_to_bucket,
    row_ids_from_lens,
    scatter_ragged_to_padded,
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
# packed <-> padded, and ragged acceptance counting
# --------------------------------------------------------------------------


def test_row_ids_map_each_packed_token_to_its_request():
    lens = torch.tensor([3, 1, 2], dtype=torch.int32)
    assert row_ids_from_lens(lens, total=6).tolist() == [0, 0, 0, 1, 2, 2]


def test_row_ids_requires_the_host_total():
    """``total`` keeps the expansion sync-free -- without it torch reads the
    cumsum back to the host, which is illegal inside the captured graph."""
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
# group-consistent bucket choice (max_verify_len given): pad rows share one
# window, so decomposability is a divisibility question every rank must answer
# identically -- a lone decline used to take the whole ADP group eager.
# ---------------------------------------------------------------------------

MAXLEN_BUCKETS = [16, 24, 32, 40, 48]  # padded_bs 8, tiers 1..5 -> 8*(t+1)


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
    """Full-block-pinned ranks must land together on a bucket every rank can
    decompose (48 here), not split the group at the first fitting bucket (40)."""
    peers = [(5, 30), (4, 24)]
    shapes = {_shape6(num_real=r, total=t, peers=peers) for r, t in peers}
    assert len(shapes) == 1
    assert shapes.pop().bucket == 48


def test_slacky_batch_keeps_the_smallest_bucket():
    """Feasibility filtering must not grow the bucket when real rows have slack
    to absorb the remainder -- the trimmed batch keeps the ungated answer."""
    peers = [(3, 9), (6, 30), (5, 11)]
    got = _shape6(num_real=3, total=9, peers=peers)
    ungated = choose_ragged_capture_shape(
        num_real_requests=3,
        total_verify_tokens=9,
        bs_buckets=BS_BUCKETS,
        token_buckets=MAXLEN_BUCKETS,
        peer_stats=peers,
    )
    assert got.bucket == ungated.bucket == 32
    assert got.padded_bs == ungated.padded_bs == 8


def test_group_declines_together_when_nothing_decomposes():
    """When no captured bucket works for every rank, every rank must see the
    same ValueError -- a group-consistent decline replays the uniform graph."""
    truncated = [16, 24, 32, 40]  # top tier 48 not captured
    peers = [(5, 30), (4, 24)]  # pinned: 40 fails (5,30), 32 fails it too
    for r, t in peers:
        with pytest.raises(ValueError, match="decomposable by every rank"):
            choose_ragged_capture_shape(
                num_real_requests=r,
                total_verify_tokens=t,
                bs_buckets=BS_BUCKETS,
                token_buckets=truncated,
                peer_stats=peers,
                max_verify_len=6,
            )


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
    split = resolve_ragged_pad_split(
        bucket=bucket,
        num_real_requests=len(real),
        total_real_tokens=sum(real),
        padded_bs=padded_bs,
        max_verify_len=max_verify_len,
    )
    assert split is not None
    assert split.pad_len == expected_pad_len

    real_target = split.real_target
    assert sum(real) <= real_target <= len(real) * max_verify_len
    got = _fit(real, bucket=real_target, padded_bs=len(real), max_verify_len=max_verify_len)
    # The total the forward sees must be exactly the captured bucket.
    assert int(got.verify_lens.sum()) + n_pad * split.pad_len == bucket


def test_pad_length_never_yields_an_infeasible_real_target():
    """Over every shape the scheduler can produce, the derived split must either
    reconstruct the bucket exactly or decline outright (never a missed bucket)."""
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
                split = resolve_ragged_pad_split(
                    bucket=bucket,
                    num_real_requests=n_real,
                    total_real_tokens=floor,
                    padded_bs=padded_bs,
                    max_verify_len=max_verify_len,
                )
                if split is None:
                    declined += 1
                    continue
                real_target = split.real_target
                assert floor <= real_target <= n_real * max_verify_len, (
                    f"padded_bs={padded_bs} n_real={n_real} bucket={bucket} "
                    f"floor={floor} pad_len={split.pad_len} -> {real_target}"
                )
                assert real_target + n_pad * split.pad_len == bucket

    assert checked > 10000, f"sweep too small to be meaningful: {checked}"
    # Declining is safe (the step stays uniform) but should stay rare, or the
    # bucket grid is too coarse to be worth capturing.
    assert declined / checked < 0.15, f"{declined}/{checked} shapes have no valid split"


def test_extra_peer_stat_fields_are_ignored_here():
    """peer_stats may grow extra elements consumed by the caller; a rigid
    two-element unpack here would raise on the first ragged step."""
    two = _shape(num_real=3, total=12, peers=[[3, 12], [4, 16]])
    three = _shape(num_real=3, total=12, peers=[[3, 12, 1], [4, 16, 1]])
    assert (three.padded_bs, three.bucket) == (two.padded_bs, two.bucket)


# ---------------------------------------------------------------------------
# device fill / row-map parity: fill_bucket_device and build_row_maps_device
# must reproduce the host path token-for-token.
# ---------------------------------------------------------------------------


def _host_fill(lens, padded_bs, bucket, max_verify_len):
    layout = RaggedVerifyLayout.from_verify_lens(
        torch.tensor(lens, dtype=torch.int32),
        graph_num_tokens=bucket,
        total_verify_tokens=sum(lens),
    )
    return layout.fill_bucket(max_verify_len=max_verify_len, padded_bs=padded_bs)


def _device_fill(lens, padded_bs, bucket, max_verify_len):
    padded = torch.ones(padded_bs, dtype=torch.int32)
    padded[: len(lens)] = torch.tensor(lens, dtype=torch.int32)
    return fill_bucket_device(
        padded,
        num_real=torch.tensor(len(lens)),
        graph_num_tokens=bucket,
        max_verify_len=max_verify_len,
    )


def _feasible(lens, padded_bs, bucket, max_verify_len):
    n_pad = padded_bs - len(lens)
    return sum(lens) + n_pad <= bucket <= padded_bs * max_verify_len


def _all_lens(n_real, max_verify_len):
    """Every [1, max]^n_real length vector."""
    if n_real == 0:
        yield []
        return
    for head in range(1, max_verify_len + 1):
        for tail in _all_lens(n_real - 1, max_verify_len):
            yield [head] + tail


@pytest.mark.parametrize("max_verify_len", [2, 4, 6])
def test_fill_parity_exhaustive_small(max_verify_len):
    """Exact host/device agreement over every feasible small case."""
    checked = 0
    for n_real in (1, 2, 3):
        for padded_bs in (n_real, n_real + 1, n_real + 3):
            for lens in _all_lens(n_real, max_verify_len):
                for bucket in range(padded_bs, padded_bs * max_verify_len + 1):
                    if not _feasible(lens, padded_bs, bucket, max_verify_len):
                        continue
                    host = _host_fill(lens, padded_bs, bucket, max_verify_len)
                    dev = _device_fill(lens, padded_bs, bucket, max_verify_len)
                    assert torch.equal(dev, host.verify_lens), (
                        f"lens={lens} padded_bs={padded_bs} bucket={bucket} "
                        f"max={max_verify_len}: device {dev.tolist()} != "
                        f"host {host.verify_lens.tolist()}"
                    )
                    checked += 1
    assert checked > 100


def test_fill_parity_randomized_large():
    """Production-shaped cases: bs up to 128, max_verify_len 6."""
    gen = torch.Generator().manual_seed(20260806)
    max_verify_len = 6
    for _ in range(200):
        n_real = int(torch.randint(1, 129, (1,), generator=gen))
        padded_bs = n_real + int(torch.randint(0, 9, (1,), generator=gen))
        lens = torch.randint(1, max_verify_len + 1, (n_real,), generator=gen).tolist()
        lo, hi = sum(lens) + (padded_bs - n_real), padded_bs * max_verify_len
        bucket = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        host = _host_fill(lens, padded_bs, bucket, max_verify_len)
        dev = _device_fill(lens, padded_bs, bucket, max_verify_len)
        assert torch.equal(dev, host.verify_lens)
        assert int(dev.sum()) == bucket


def test_row_maps_match_prepare_semantics():
    """req_idx/correction must equal what prepare() stages: token o of a window
    v gets correction o - v + 1, so the extent walks kv_len - v + 1 .. kv_len."""
    gen = torch.Generator().manual_seed(7)
    for _ in range(50):
        bs = int(torch.randint(1, 65, (1,), generator=gen))
        lens = torch.randint(1, 7, (bs,), generator=gen).to(torch.int32)
        total = int(lens.sum())
        req_idx, corr = build_row_maps_device(lens, graph_num_tokens=total)
        want_req, want_corr = [], []
        for r, v in enumerate(lens.tolist()):
            for o in range(v):
                want_req.append(r)
                want_corr.append(o - v + 1)
        assert req_idx.tolist() == want_req
        assert corr.tolist() == want_corr
        # Composing with a kv_lens gather is refresh_ragged_row_kv_lens:
        # the extents must end exactly at each request's kv_len.
        kv_lens = torch.randint(10, 1000, (bs,), generator=gen).to(torch.int32)
        extents = kv_lens[req_idx] + corr
        ends = torch.cumsum(lens.to(torch.long), 0) - 1
        assert torch.equal(extents[ends], kv_lens)
        assert int(extents.min()) >= 1


def test_fill_pad_fill_constrained_split():
    """With pad_fill given, every pad row carries exactly that many tokens, the
    real rows absorb all slack, and the total still hits the bucket."""
    gen = torch.Generator().manual_seed(3)
    max_verify_len = 6
    for _ in range(100):
        n_real = int(torch.randint(1, 65, (1,), generator=gen))
        n_pad = int(torch.randint(0, 9, (1,), generator=gen))
        padded_bs = n_real + n_pad
        pad_fill = int(torch.randint(1, max_verify_len + 1, (1,), generator=gen))
        lens = torch.randint(2, max_verify_len + 1, (n_real,), generator=gen)
        lo = int(lens.sum()) + n_pad * pad_fill
        hi = n_real * max_verify_len + n_pad * pad_fill
        if lo > hi:
            continue
        bucket = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        padded = torch.ones(padded_bs, dtype=torch.int32)
        padded[:n_real] = lens.to(torch.int32)
        filled = fill_bucket_device(
            padded,
            num_real=torch.tensor(n_real),
            graph_num_tokens=bucket,
            max_verify_len=max_verify_len,
            pad_fill=pad_fill,
        )
        assert int(filled.sum()) == bucket
        assert (filled[n_real:] == pad_fill).all()
        assert (filled[:n_real] >= lens.to(torch.int32)).all()
        assert int(filled[:n_real].max()) <= max_verify_len
