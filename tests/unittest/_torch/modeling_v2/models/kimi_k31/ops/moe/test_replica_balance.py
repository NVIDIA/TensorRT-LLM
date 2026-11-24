# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-layer dynamic EPLB load record of the Kimi K3.1 routed experts (``ops/moe/replica_balance``).

* CPU: the window statistics (row-weighted ``max / mean`` before and after the plans, the per layer-forward
  percentiles over large layer-forwards, the helper share) and the sampling knob.
* one GPU: the device record of logical routes and physical slots, the window copy and the flush, with an EP group
  of four ranks emulated by a communicator that hands back four copies of this rank's counts.
"""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import replica_balance
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.replica_balance import (
    MIN_TOKENS,
    ReplicaBalanceLog,
    balance_summary,
    sampling_stride,
)

EP, TOP_K = 4, 16


def _counts(before, after, helper):
    """One window of one layer-forward per row: ``[samples, 1, 2 * EP + 1]``."""
    rows = [list(b) + list(a) + [h] for b, a, h in zip(before, after, helper)]
    return np.array(rows, dtype=np.int64)[:, None, :]


@pytest.mark.cpu_only
def test_summary_weights_by_rows_and_keeps_large_layer_forwards_for_percentiles() -> None:
    big = MIN_TOKENS * TOP_K
    counts = _counts(
        before=[(big, big, big, big), (2 * big, big, big, 0), (3, 1, 0, 0)],
        after=[(big, big, big, big), (big, big, big, big), (1, 1, 1, 1)],
        helper=[0, big, 2],
    )
    s = balance_summary(counts, EP, TOP_K)
    assert s["rows"] == 4 * big + 4 * big + 4
    assert s["layer_forwards"] == 3
    # max / mean summed over layer-forwards: (big + 2 big + 3) / (big + big + 1).
    assert s["before"] == pytest.approx((3 * big + 3) / (2 * big + 1))
    assert s["after"] == pytest.approx((2 * big + 1) / (2 * big + 1))
    assert s["helper_share"] == pytest.approx((big + 2) / s["rows"])
    # The near-empty layer-forward stays out of the percentiles.
    assert s["large_layer_forwards"] == 2
    assert s["before_max"] == pytest.approx(2.0)
    assert s["after_max"] == pytest.approx(1.0)
    for key in ("max_before", "max_after", "mean"):
        assert s[key] > 0


@pytest.mark.cpu_only
def test_summary_of_an_empty_window() -> None:
    s = balance_summary(np.zeros((2, 3, 2 * EP + 1), dtype=np.int64), EP, TOP_K)
    assert s["rows"] == 0 and s["before"] == 0.0 and s["after"] == 0.0 and s["helper_share"] == 0.0


@pytest.mark.cpu_only
@pytest.mark.parametrize("value, every", [("", 8), ("0", 0), ("3", 3)])
def test_sampling_stride(monkeypatch, value, every) -> None:
    monkeypatch.setenv(replica_balance.EVERY_ENV, value)
    assert sampling_stride() == every


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["-1", "x"])
def test_sampling_stride_refuses_bad_values(monkeypatch, value) -> None:
    monkeypatch.setenv(replica_balance.EVERY_ENV, value)
    with pytest.raises(ValueError, match=replica_balance.EVERY_ENV):
        sampling_stride()


class _FourRanks:
    """The EP group's communicator stand-in: every rank sent what this one did."""

    def __init__(self):
        self.calls = 0

    def allgather(self, value):
        self.calls += 1
        return [value] * EP


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the record runs on the GPU")
def test_record_window_and_flush(monkeypatch) -> None:
    home, slots = 2, 3  # 2 resident experts + 1 helper slot per rank, 8 experts over 4 ranks
    comm = _FourRanks()
    log = ReplicaBalanceLog(
        ep_rank=0,
        ep_size=EP,
        home_experts=home,
        compute_slots=slots,
        num_layers=2,
        top_k=TOP_K,
        ep_comm=lambda: comm,
        every=2,
        window=2,
    )
    seen = []
    real = replica_balance.balance_summary
    monkeypatch.setattr(
        replica_balance,
        "balance_summary",
        lambda counts, *a: seen.append(np.array(counts)) or real(counts, *a),
    )
    # Every pick on expert 1 (rank 0), half of them moved to rank 2's helper slot (slot 2 * 3 + 2 = 8).
    logical = torch.ones(4, TOP_K, dtype=torch.int32, device="cuda")
    physical = torch.full((4, TOP_K), 1, dtype=torch.int32, device="cuda")
    physical[:, ::2] = 8
    for forward in range(5):
        log.begin_forward()
        for layer in range(2):
            log.record(layer, logical, physical)
        log.end_forward()
    torch.cuda.synchronize()
    # Forwards 0, 2 sampled (window full after forward 2), flushed at the start of forward 3; forward 4 opens the next.
    assert log.forwards == 5 and log.plans == 10
    assert comm.calls == 1 and len(seen) == 1
    window = seen[0]
    assert window.shape == (2, 2, 2 * EP + 1)
    rows = 4 * TOP_K
    expected = np.array([rows, 0, 0, 0, rows // 2, 0, rows // 2, 0, rows // 2]) * EP
    assert (window == expected).all()
    assert log.totals["max_before"] == pytest.approx(4 * rows * EP)
    assert log.totals["max_after"] == pytest.approx(4 * rows * EP / 2)
