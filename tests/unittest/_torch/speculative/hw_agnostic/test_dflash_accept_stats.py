# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DFlash/DSpark acceptance-statistics aggregation
math (tensorrt_llm/_torch/speculative/accept_stats.py).

accept_stats.py is import-light (no torch), so these tests also run on
hosts without a GPU stack; the fallback loader below keeps them runnable
even where the tensorrt_llm package itself cannot be imported.
"""

import json
import os

import pytest

try:
    from tensorrt_llm._torch.speculative import accept_stats
except ImportError:  # pragma: no cover - torch-less host fallback
    import importlib.util

    _path = os.path.join(
        os.path.dirname(__file__),
        *([os.pardir] * 5),
        "tensorrt_llm",
        "_torch",
        "speculative",
        "accept_stats.py",
    )
    _spec = importlib.util.spec_from_file_location("accept_stats", os.path.abspath(_path))
    accept_stats = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(accept_stats)


def test_summarize_hist_basic():
    # K = 3; 10 steps: 4 steps accepted 0 drafts, 3 accepted 1,
    # 2 accepted 2, 1 accepted 3.
    hist = [4, 3, 2, 1]
    s = accept_stats.summarize_hist(hist)
    assert s["num_steps"] == 10
    assert s["mean_accepted_draft"] == pytest.approx(1.0)
    assert s["al"] == pytest.approx(2.0)
    # AR_k = P(accepted_draft >= k): [6/10, 3/10, 1/10]
    assert s["ar_per_position"] == pytest.approx([0.6, 0.3, 0.1])
    # Monotone non-increasing by construction (prefix acceptance).
    ar = s["ar_per_position"]
    assert all(a >= b for a, b in zip(ar, ar[1:]))


def test_summarize_hist_empty_and_dummy_regimes():
    assert accept_stats.summarize_hist([0, 0, 0])["al"] == 0.0
    # Dummy-drafter regime: every step accepts 0 drafts -> AL exactly 1.0
    # (bonus token only), AR 0 at every position.
    s = accept_stats.summarize_hist([100, 0, 0, 0, 0, 0, 0, 0])
    assert s["al"] == pytest.approx(1.0)
    assert s["ar_per_position"] == pytest.approx([0.0] * 7)
    # Perfect drafter: every step accepts all K=7 drafts -> AL = 8.
    s = accept_stats.summarize_hist([0, 0, 0, 0, 0, 0, 0, 50])
    assert s["al"] == pytest.approx(8.0)
    assert s["ar_per_position"] == pytest.approx([1.0] * 7)


def test_recorder_accumulation_and_flush(tmp_path):
    rec = accept_stats.DFlashAcceptStatsRecorder(
        str(tmp_path), max_draft_len=3, rank=0, flush_every=1000, num_conf_bins=4
    )
    # Step 1: req 7 accepts bonus+2 drafts, req 8 bonus only.
    rec.on_draft_confidence([7, 8], [[0.9, 0.6, 0.1], [0.2, 0.2, 0.2]])
    rec.on_accept([7, 8], [3, 1])
    # Step 2: req 7 accepts everything (bonus + 3 drafts, clamped input 5).
    rec.on_accept([7], [5])

    snap = rec.snapshot()
    assert snap["num_steps"] == 2
    assert snap["accepted_draft_hist"] == [1, 0, 1, 1]
    assert snap["per_request"]["7"] == {"steps": 2, "accepted_draft": 5}
    assert snap["per_request"]["8"] == {"steps": 1, "accepted_draft": 0}

    cc = snap["confidence_calibration"]
    # req 7: conf (0.9, 0.6, 0.1) -> bins (3, 2, 0); accepted at pos 1,2 only.
    assert cc["attempts"][0][3] == 1 and cc["accepted"][0][3] == 1
    assert cc["attempts"][1][2] == 1 and cc["accepted"][1][2] == 1
    assert cc["attempts"][2][0] == 2 and cc["accepted"][2][0] == 0
    # req 8: conf 0.2 -> bin 0 at each position, nothing accepted.
    assert cc["attempts"][0][0] == 1 and cc["accepted"][0][0] == 0
    # Step 2 had no pending confidences for req 7 (consumed in step 1).
    assert sum(sum(r) for r in cc["attempts"]) == 6

    rec.flush()
    with open(rec.path) as f:
        assert json.load(f) == snap


def test_merge_snapshots_and_load(tmp_path):
    r0 = accept_stats.DFlashAcceptStatsRecorder(str(tmp_path), 2, rank=0, num_conf_bins=4)
    r1 = accept_stats.DFlashAcceptStatsRecorder(str(tmp_path), 2, rank=1, num_conf_bins=4)
    r0.on_draft_confidence([1], [[0.9, 0.9]])
    r0.on_accept([1], [3])  # 2 drafts accepted
    r1.on_accept([2], [1])  # 0 drafts accepted
    r0.flush()
    r1.flush()

    snaps = accept_stats.load_rank_snapshots(str(tmp_path))
    assert len(snaps) == 2
    merged = accept_stats.merge_snapshots(snaps)
    assert merged["num_steps"] == 2
    assert merged["accepted_draft_hist"] == [1, 0, 1]
    assert merged["per_request"]["1"]["accepted_draft"] == 2
    assert merged["per_request"]["2"]["steps"] == 1
    s = accept_stats.summarize_hist(merged["accepted_draft_hist"])
    assert s["al"] == pytest.approx(2.0)
    assert s["ar_per_position"] == pytest.approx([0.5, 0.5])

    with pytest.raises(ValueError):
        accept_stats.merge_snapshots([])


def test_merge_snapshots_shape_mismatch(tmp_path):
    a = accept_stats.DFlashAcceptStatsRecorder(str(tmp_path), 2, rank=0).snapshot()
    b = accept_stats.DFlashAcceptStatsRecorder(str(tmp_path), 3, rank=1).snapshot()
    with pytest.raises(ValueError):
        accept_stats.merge_snapshots([a, b])


def test_calibration_table():
    # One position, 4 bins; perfectly calibrated in bins 1 and 3.
    attempts = [[0, 10, 0, 8]]
    accepted = [[0, 4, 0, 7]]  # empirical 0.4 vs center 0.375; 0.875 = center
    table = accept_stats.calibration_table(attempts, accepted)
    assert table["bin_centers"] == pytest.approx([0.125, 0.375, 0.625, 0.875])
    pos = table["per_position"][0]
    assert pos["empirical_acceptance"][0] is None
    assert pos["empirical_acceptance"][1] == pytest.approx(0.4)
    assert pos["empirical_acceptance"][3] == pytest.approx(0.875)
    # ECE = (10*|0.375-0.4| + 8*|0.875-0.875|) / 18
    assert pos["ece"] == pytest.approx(10 * 0.025 / 18)


def test_recorder_excludes_dummy_request_id_zero(tmp_path):
    # Executor-warmup dummies and idle attention-DP ranks' padding requests
    # carry request id 0 and must not pollute any counter.
    rec = accept_stats.DFlashAcceptStatsRecorder(
        str(tmp_path), max_draft_len=3, rank=0, num_conf_bins=4
    )
    rec.on_draft_confidence([0, 5], [[0.9, 0.9, 0.9], [0.4, 0.4, 0.4]])
    rec.on_accept([0, 5], [4, 2])  # id 0: all-K padding "accept" -> dropped
    snap = rec.snapshot()
    assert snap["accepted_draft_hist"] == [0, 1, 0, 0]
    assert list(snap["per_request"]) == ["5"]
    assert sum(sum(r) for r in snap["confidence_calibration"]["attempts"]) == 3


def test_confidence_provider_none_collects_nothing(tmp_path):
    # Default recorder (no provider): record_draft_confidence is a no-op,
    # the calibration table stays empty, AL/AR still accumulate.
    rec = accept_stats.DFlashAcceptStatsRecorder(
        str(tmp_path), max_draft_len=2, rank=0, num_conf_bins=4
    )
    assert rec.confidence_provider is None
    rec.record_draft_confidence([1], "model", "hidden", "prev", "drafts")
    rec.on_accept([1], [2])
    snap = rec.snapshot()
    assert snap["accepted_draft_hist"] == [0, 1, 0]
    assert sum(sum(r) for r in snap["confidence_calibration"]["attempts"]) == 0


def test_confidence_provider_decline_and_forwarding(tmp_path):
    # A provider may decline a step by returning None; the recorder must
    # forward the draft-site arguments verbatim (they are opaque to it).
    calls = []

    def declining_provider(*args):
        calls.append(args)
        return None

    rec = accept_stats.DFlashAcceptStatsRecorder(
        str(tmp_path),
        max_draft_len=2,
        rank=0,
        num_conf_bins=4,
        confidence_provider=declining_provider,
    )
    sentinel = ("model", "hidden", "prev", "drafts")
    rec.record_draft_confidence([1], *sentinel)
    assert calls == [sentinel]
    assert not rec._pending_conf


def test_mocked_calibration_end_to_end(tmp_path):
    """Integration: scripted mock provider + scripted accept outcomes,
    driven through the recorder exactly as the DFlash worker drives it
    (verify previous block via on_accept, then draft a new block via
    record_draft_confidence), then flushed / loaded / merged / tabulated.
    Asserts exact bin counts and calibration-table contents."""
    K, NBINS = 3, 4  # bins: [0,.25) [.25,.5) [.5,.75) [.75,1]

    scripted_rows = [
        # step 1 block: rows for requests (1, 2)
        [[0.9, 0.6, 0.1], [0.3, 0.3, 0.3]],
        # step 2 block
        [[0.8, 0.2, 0.55], [0.7, 0.1, 0.9]],
    ]
    provider_calls = []

    def mock_provider(draft_model, gen_hidden, first_prev_tokens, gen_draft_tokens):
        provider_calls.append(draft_model)
        return scripted_rows[len(provider_calls) - 1]

    rec = accept_stats.DFlashAcceptStatsRecorder(
        str(tmp_path),
        max_draft_len=K,
        rank=0,
        flush_every=1000,
        num_conf_bins=NBINS,
        confidence_provider=mock_provider,
    )

    rids = [1, 2]
    # Step 1: nothing pending yet; both requests accept 0 drafts (n=1).
    rec.on_accept(rids, [1, 1])
    rec.record_draft_confidence(rids, "m", "h", "p", "d")
    # Step 2: req 1 accepts 2 drafts (n=3), req 2 accepts 0 (n=1).
    rec.on_accept(rids, [3, 1])
    rec.record_draft_confidence(rids, "m", "h", "p", "d")
    # Step 3: req 1 accepts 1 draft (n=2), req 2 accepts all 3 (n=4).
    rec.on_accept(rids, [2, 4])

    assert len(provider_calls) == 2
    rec.flush()

    snaps = accept_stats.load_rank_snapshots(str(tmp_path))
    merged = accept_stats.merge_snapshots(snaps)
    assert merged["num_steps"] == 3
    assert merged["accepted_draft_hist"] == [3, 1, 1, 1]

    cc = merged["confidence_calibration"]
    # Exact per-(position, bin) attempt/accept counts (bin = int(4c)):
    # step-1 rows joined with step-2 outcomes, step-2 rows with step-3.
    assert cc["attempts"] == [
        [0, 1, 1, 2],  # pos 1: 0.3 | 0.7 | 0.9, 0.8
        [2, 1, 1, 0],  # pos 2: 0.2, 0.1 | 0.3 | 0.6
        [1, 1, 1, 1],  # pos 3: 0.1 | 0.3 | 0.55 | 0.9
    ]
    assert cc["accepted"] == [
        [0, 0, 1, 2],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    table = accept_stats.calibration_table(cc["attempts"], cc["accepted"])
    assert table["bin_centers"] == pytest.approx([0.125, 0.375, 0.625, 0.875])
    p1, p2, p3 = table["per_position"]
    assert p1["empirical_acceptance"] == [None, 0.0, 1.0, 1.0]
    assert p2["empirical_acceptance"] == [0.5, 0.0, 1.0, None]
    assert p3["empirical_acceptance"] == [0.0, 0.0, 0.0, 1.0]
    assert p1["num_samples"] == [0, 1, 1, 2]
    # pos-1 ECE: (1*|.375-0| + 1*|.625-1| + 2*|.875-1|) / 4
    assert p1["ece"] == pytest.approx((0.375 + 0.375 + 2 * 0.125) / 4)


def test_default_provider_absent_in_core_tree():
    # The optional dspark_confidence module ships in a separate MR; on this
    # tree the default provider must resolve to None (not raise).
    assert accept_stats._resolve_default_confidence_provider() is None


def test_conf_bin_edges():
    assert accept_stats._conf_bin(-0.1, 20) == 0
    assert accept_stats._conf_bin(0.0, 20) == 0
    assert accept_stats._conf_bin(0.999, 20) == 19
    assert accept_stats._conf_bin(1.0, 20) == 19
    assert accept_stats._conf_bin(1.5, 20) == 19
