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

"""Unit tests for EPGroupHealth (WideEP fault tolerance, PR 1a.1).

Covers single-threaded correctness of the public API and concurrent update
races as required by the WideEP FT implementation plan section 8.
"""

import threading

import pytest

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import (
    EP_MASK_NUM_WORDS,
    EPGroupHealth,
    EPGroupHealthSnapshot,
)

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moe_world_size",
    [1, 2, 8, 64, 72, 128],
    ids=["ep1", "ep2", "ep8", "ep64_one_word_full", "nvl72", "ep128_two_words_full"],
)
def test_initial_state_all_active(moe_world_size: int) -> None:
    """Verify a new health object starts with every rank active."""
    h = EPGroupHealth(moe_world_size)
    assert h.moe_world_size == moe_world_size
    assert len(h) == moe_world_size
    assert h.get_active_count() == moe_world_size
    assert h.get_failed_ranks() == frozenset()
    assert h.all_active() is True
    assert h.generation == 0
    assert h.get_mask() == (1 << moe_world_size) - 1
    for r in range(moe_world_size):
        assert h.is_active(r) is True


@pytest.mark.parametrize(
    "bad_size",
    [0, -1, -100],
    ids=["zero", "neg_one", "large_neg"],
)
def test_init_rejects_non_positive_moe_world_size(bad_size: int) -> None:
    """Reject zero and negative MoE world sizes."""
    with pytest.raises(ValueError):
        EPGroupHealth(bad_size)


@pytest.mark.parametrize(
    "bad_rank",
    [-1, 8, 9, 100],
    ids=["neg_one", "at_moe_world_size", "above_moe_world_size", "way_above"],
)
def test_methods_reject_out_of_range_rank(bad_rank: int) -> None:
    """Reject invalid ranks across rank-indexed APIs."""
    h = EPGroupHealth(8)
    with pytest.raises(ValueError):
        h.mark_failed(bad_rank)
    with pytest.raises(ValueError):
        h.mark_active(bad_rank)
    with pytest.raises(ValueError):
        h.is_active(bad_rank)


# ---------------------------------------------------------------------------
# Single-threaded correctness: mark_failed / mark_active
# ---------------------------------------------------------------------------


def test_mark_failed_basic() -> None:
    """Marking an active rank failed updates mask, count, set, and generation."""
    h = EPGroupHealth(8)
    assert h.mark_failed(3) is True
    assert h.is_active(3) is False
    assert h.get_active_count() == 7
    assert h.get_failed_ranks() == frozenset({3})
    assert h.all_active() is False
    assert h.generation == 1
    # bit 3 cleared, all other bits set
    assert h.get_mask() == ((1 << 8) - 1) & ~(1 << 3)


def test_mark_failed_idempotent() -> None:
    """Repeated failure marks for the same rank are no-ops."""
    h = EPGroupHealth(8)
    assert h.mark_failed(2) is True
    gen_after_first = h.generation
    # Second call is a no-op and reports it.
    assert h.mark_failed(2) is False
    assert h.generation == gen_after_first
    assert h.get_active_count() == 7
    assert h.get_failed_ranks() == frozenset({2})


def test_mark_active_reverses_mark_failed() -> None:
    """Reactivating a failed rank restores the all-active state."""
    h = EPGroupHealth(8)
    h.mark_failed(5)
    assert h.is_active(5) is False
    assert h.mark_active(5) is True
    assert h.is_active(5) is True
    assert h.get_active_count() == 8
    assert h.get_failed_ranks() == frozenset()
    assert h.all_active() is True
    # Two state changes total: fail + reactivate.
    assert h.generation == 2


def test_mark_active_idempotent_on_already_active() -> None:
    """Reactivating an already active rank is a no-op."""
    h = EPGroupHealth(8)
    # Brand-new health: all ranks active. Reactivating any rank is a no-op.
    assert h.mark_active(0) is False
    assert h.generation == 0
    assert h.get_active_count() == 8


def test_multiple_failures_accumulate() -> None:
    """Independent rank failures accumulate in the mask and failed-rank set."""
    h = EPGroupHealth(16)
    for r in [1, 4, 7, 12]:
        assert h.mark_failed(r) is True
    assert h.get_active_count() == 12
    assert h.get_failed_ranks() == frozenset({1, 4, 7, 12})
    expected_mask = ((1 << 16) - 1) & ~((1 << 1) | (1 << 4) | (1 << 7) | (1 << 12))
    assert h.get_mask() == expected_mask
    assert h.generation == 4


def test_get_failed_ranks_returns_snapshot() -> None:
    """Failed-rank queries return immutable snapshots."""
    h = EPGroupHealth(8)
    h.mark_failed(2)
    snap = h.get_failed_ranks()
    assert isinstance(snap, frozenset)
    # Snapshot is a frozenset; it cannot be mutated by the caller.
    with pytest.raises(AttributeError):
        snap.add(3)  # type: ignore[attr-defined]
    # Subsequent state changes do not affect the prior snapshot.
    h.mark_failed(3)
    assert snap == frozenset({2})
    assert h.get_failed_ranks() == frozenset({2, 3})


# ---------------------------------------------------------------------------
# get_mask_words (kernel-passing format)
# ---------------------------------------------------------------------------


def _expected_words(moe_world_size: int, num_words: int, fail_ranks: list[int]) -> tuple[int, ...]:
    """Reference computation: bit i set iff i < moe_world_size and i not in fail_ranks,
    then split into ``num_words`` little-endian uint64 words.
    """
    full = (1 << moe_world_size) - 1
    for r in fail_ranks:
        full &= ~(1 << r)
    word_mask = (1 << 64) - 1
    return tuple((full >> (i * 64)) & word_mask for i in range(num_words))


@pytest.mark.parametrize(
    "moe_world_size,num_words,fail_ranks",
    [
        # default 2 words, all active -> (0xFF, 0)
        (8, 2, []),
        # one word, fully populated -> ((1<<64)-1,)
        (64, 1, []),
        # NVL72, low-word failure (rank 37 in word 0)
        (72, 2, [37]),
        # NVL72, high-word failure (rank 70 in word 1, bit 6)
        (72, 2, [70]),
        # extra words past moe_world_size are zero
        (8, 4, []),
        # moe_world_size > 128 (beyond default kernel ABI), one failure per word
        (256, 4, [10, 70, 130, 200]),
    ],
    ids=[
        "ep8_default_all_active",
        "ep64_one_word_full",
        "nvl72_low_word_failure",
        "nvl72_high_word_failure",
        "ep8_extra_words_zero",
        "ep256_one_failure_per_word",
    ],
)
def test_get_mask_words_layout(moe_world_size: int, num_words: int, fail_ranks: list[int]) -> None:
    """Single-word, multi-word, NVL72 boundary, and >128 cases all share the same
    invariant: get_mask_words() splits the bitmask into the requested number of
    little-endian uint64 words. Each word must fit in uint64 and the words must
    reassemble to the underlying get_mask() Python int.
    """
    h = EPGroupHealth(moe_world_size)
    for r in fail_ranks:
        assert h.mark_failed(r) is True

    words = h.get_mask_words(num_words=num_words)
    assert words == _expected_words(moe_world_size, num_words, fail_ranks)
    assert len(words) == num_words
    for w in words:
        assert 0 <= w < (1 << 64)
    # Round-trip: words reassembled equal the underlying Python int.
    assert h.get_mask() == sum(w << (i * 64) for i, w in enumerate(words))


@pytest.mark.parametrize(
    "bad_num_words",
    [0, -1],
    ids=["zero_words", "negative_words"],
)
def test_get_mask_words_rejects_non_positive(bad_num_words: int) -> None:
    """Reject non-positive mask word counts."""
    h = EPGroupHealth(8)
    with pytest.raises(ValueError):
        h.get_mask_words(num_words=bad_num_words)


@pytest.mark.parametrize(
    "moe_world_size,num_words",
    [
        (72, 1),  # NVL72: 72 bits do not fit in a single uint64
        (256, 2),  # moe_world_size > 128 does not fit in the default 2 words
        (256, 3),  # nor in 3 words
    ],
    ids=[
        "nvl72_in_one_word",
        "ep256_in_default_two_words",
        "ep256_in_three_words",
    ],
)
def test_get_mask_words_rejects_too_few_words(moe_world_size: int, num_words: int) -> None:
    """Reject mask word counts that cannot cover the MoE world."""
    h = EPGroupHealth(moe_world_size)
    with pytest.raises(ValueError):
        h.get_mask_words(num_words=num_words)


# ---------------------------------------------------------------------------
# Generation counter semantics
# ---------------------------------------------------------------------------


def test_generation_only_bumps_on_effective_change() -> None:
    """Only effective active/failed transitions bump generation."""
    h = EPGroupHealth(8)
    assert h.generation == 0

    assert h.mark_failed(3) is True
    assert h.generation == 1
    assert h.mark_failed(3) is False  # idempotent
    assert h.generation == 1

    assert h.mark_active(0) is False  # already active
    assert h.generation == 1

    assert h.mark_active(3) is True
    assert h.generation == 2


def test_repr_is_informative() -> None:
    """repr includes the main state fields for debugging."""
    h = EPGroupHealth(4)
    h.mark_failed(1)
    s = repr(h)
    assert "moe_world_size=4" in s
    assert "active_count=3" in s
    assert "failed_ranks=[1]" in s
    assert "generation=1" in s


# ---------------------------------------------------------------------------
# Concurrent update races
# ---------------------------------------------------------------------------


def test_concurrent_mark_failed_distinct_ranks() -> None:
    """Many threads, each marking a distinct rank; all should land exactly once."""
    moe_world_size = 128
    h = EPGroupHealth(moe_world_size)

    barrier = threading.Barrier(moe_world_size)

    def worker(rank: int) -> None:
        """Mark one distinct rank failed after the synchronized start."""
        # Synchronize start so threads contend on the lock simultaneously.
        barrier.wait()
        h.mark_failed(rank)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(moe_world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert h.get_active_count() == 0
    assert h.get_failed_ranks() == frozenset(range(moe_world_size))
    assert h.get_mask() == 0
    assert h.generation == moe_world_size


def test_concurrent_mark_failed_same_rank_idempotent() -> None:
    """Many threads racing on the same rank; exactly one effective change."""
    h = EPGroupHealth(8)
    n_threads = 64
    barrier = threading.Barrier(n_threads)
    effective_changes: list[bool] = []
    changes_lock = threading.Lock()

    def worker() -> None:
        """Race with other workers to mark rank 3 failed."""
        barrier.wait()
        changed = h.mark_failed(3)
        with changes_lock:
            effective_changes.append(changed)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(effective_changes) == 1
    assert h.get_active_count() == 7
    assert h.get_failed_ranks() == frozenset({3})
    assert h.generation == 1


def test_concurrent_mixed_fail_and_reactivate() -> None:
    """Stress: alternating fail/active calls converge to a deterministic state.

    Each rank is touched by one fail-then-active pair; the final state must be
    all ranks active, with generation == 2 * moe_world_size. The reactivate thread
    waits on a per-rank threading.Event signaled by the fail thread, so we
    avoid any busy-wait that could spin under adversarial scheduling.
    """
    moe_world_size = 32
    h = EPGroupHealth(moe_world_size)

    barrier = threading.Barrier(moe_world_size * 2)
    fail_done = [threading.Event() for _ in range(moe_world_size)]

    def fail(rank: int) -> None:
        """Mark one rank failed and signal its paired reactivation thread."""
        barrier.wait()
        h.mark_failed(rank)
        fail_done[rank].set()

    def reactivate(rank: int) -> None:
        """Wait for the paired failure before marking the rank active again."""
        barrier.wait()
        # Wait for the corresponding fail to land, then reactivate. Bounded
        # wait so a regression that prevented the fail signal cannot wedge CI.
        assert fail_done[rank].wait(timeout=10.0), f"rank {rank} fail did not signal"
        h.mark_active(rank)

    fail_threads = [threading.Thread(target=fail, args=(r,)) for r in range(moe_world_size)]
    react_threads = [threading.Thread(target=reactivate, args=(r,)) for r in range(moe_world_size)]

    for t in fail_threads + react_threads:
        t.start()
    for t in fail_threads + react_threads:
        t.join()

    assert h.all_active() is True
    assert h.get_active_count() == moe_world_size
    assert h.get_failed_ranks() == frozenset()
    # Each rank contributed exactly two effective state changes.
    assert h.generation == 2 * moe_world_size


# ---------------------------------------------------------------------------
# Oscillation (fail -> active -> fail -> active)
# ---------------------------------------------------------------------------


def test_oscillation_preserves_generation_monotonicity() -> None:
    """Fail -> active -> fail -> active on the same rank.

    Verifies that generation increments by exactly one per effective
    transition and that the final state is fully restored.
    """
    h = EPGroupHealth(8)
    rank = 5
    expected_gen = 0

    for cycle in range(4):
        if cycle % 2 == 0:
            assert h.mark_failed(rank) is True
            assert h.is_active(rank) is False
        else:
            assert h.mark_active(rank) is True
            assert h.is_active(rank) is True
        expected_gen += 1
        assert h.generation == expected_gen

    # Ended on an active transition; rank should be back.
    assert h.is_active(rank) is True
    assert h.all_active() is True
    assert h.generation == 4


# ---------------------------------------------------------------------------
# snapshot() - atomic multi-field read
# ---------------------------------------------------------------------------


def test_snapshot_returns_all_fields_at_one_instant() -> None:
    """snapshot returns one coherent immutable state view."""
    h = EPGroupHealth(16)
    h.mark_failed(3)
    h.mark_failed(7)

    snap = h.snapshot()
    assert isinstance(snap, EPGroupHealthSnapshot)
    assert snap.mask == ((1 << 16) - 1) & ~((1 << 3) | (1 << 7))
    assert snap.active_count == 14
    assert snap.failed_ranks == frozenset({3, 7})
    assert snap.generation == 2

    # Field aliases still work via attribute access.
    assert snap[0] == snap.mask
    assert snap[3] == snap.generation


def test_snapshot_is_immutable() -> None:
    """snapshot results cannot be mutated by callers."""
    h = EPGroupHealth(8)
    h.mark_failed(1)
    snap = h.snapshot()

    # NamedTuples are immutable; reassigning a field must fail.
    with pytest.raises(AttributeError):
        snap.generation = 99  # type: ignore[misc]
    # The frozenset slot cannot be mutated either.
    with pytest.raises(AttributeError):
        snap.failed_ranks.add(2)  # type: ignore[attr-defined]

    # And later state changes do not retroactively alter the snapshot.
    h.mark_failed(2)
    assert snap.failed_ranks == frozenset({1})
    assert snap.generation == 1


# ---------------------------------------------------------------------------
# __len__ semantics
# ---------------------------------------------------------------------------


def test_len_returns_moe_world_size_not_active_count() -> None:
    """__len__ is the MoE world size; failures must not change it."""
    h = EPGroupHealth(8)
    assert len(h) == 8
    h.mark_failed(0)
    h.mark_failed(1)
    h.mark_failed(2)
    assert h.get_active_count() == 5  # active count moves
    assert len(h) == 8  # moe_world_size does not


def test_ep_mask_num_words_constant_default_value() -> None:
    """Lock in the kernel-ABI default. If this changes, kMaxRanks must too."""
    assert EP_MASK_NUM_WORDS == 2
