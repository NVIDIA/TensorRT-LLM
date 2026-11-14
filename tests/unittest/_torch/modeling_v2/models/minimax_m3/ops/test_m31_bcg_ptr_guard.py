# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3 S4 family module ops/bcg_ptr_guard (CPU): addresses noted at capture are re-read for every noted key before every
replay; ``reset`` (the graphs were released) forgets them, so the next executor's captures note a new pool's
addresses."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.bcg_ptr_guard import (
    StaticPointerGuard,
)

pytestmark = pytest.mark.cpu_only


def test_an_unarmed_guard_checks_nothing():
    guard = StaticPointerGuard("t")
    assert not guard.armed
    assert guard.check(lambda key: pytest.fail("read without a capture")) is False


NOTED = {3: (10, 20, 30, 40), 4: (10, 21, 31, 41)}


def test_every_noted_key_is_checked_on_every_call():
    guard = StaticPointerGuard("t")
    for key, ptrs in NOTED.items():
        guard.note(key, ptrs)
    reads = []

    def ptrs(key):
        reads.append(key)
        return NOTED[key]

    for _ in range(3):
        assert guard.check(ptrs) is True
    assert reads == [3, 4] * 3 and guard.checks == 3
    with pytest.raises(RuntimeError, match="buffers moved since capture"):
        guard.check(lambda key: (10, 20, 30, 99))


def test_a_later_key_that_moved_fails_closed_while_the_first_holds():
    """CodeRabbit K3 r3: each layer has its own cache views, so the first noted layer alone does not vouch for the
    others."""
    guard = StaticPointerGuard("t")
    for key, ptrs in NOTED.items():
        guard.note(key, ptrs)
    moved = {3: NOTED[3], 4: (10, 21, 31, 99)}
    with pytest.raises(RuntimeError, match="4 buffers moved since capture"):
        guard.check(moved.__getitem__)
    assert guard.checks == 0


def test_a_moved_buffer_fails_closed():
    guard = StaticPointerGuard("A1_FUSED")
    guard.note(3, (10, 20, 30, 40))
    with pytest.raises(RuntimeError, match="A1_FUSED: 3 buffers moved since capture"):
        guard.check(lambda key: (10, 20, 30, 99))
    assert guard.checks == 0


def test_a_capture_that_sees_other_addresses_raises():
    guard = StaticPointerGuard("t")
    guard.note(3, (1, 2, 3, 4))
    guard.note(3, (1, 2, 3, 4))  # the same addresses for another bucket
    with pytest.raises(RuntimeError, match="moved between captures"):
        guard.note(3, (1, 2, 3, 5))


def test_a_reset_lets_the_next_captures_note_new_addresses():
    """KV-cache estimation: the first executor's graphs noted its pool; they are released (reset) before the final
    executor captures on a new pool, whose addresses must not be compared with the released ones."""
    guard = StaticPointerGuard("t")
    guard.note(3, (1, 2, 3, 4))
    assert guard.check(lambda key: (1, 2, 3, 4)) is True
    guard.reset()
    assert not guard.armed
    assert guard.check(lambda key: pytest.fail("read with no live capture")) is False
    guard.note(4, (5, 6, 7, 8))  # the final executor's notes alone are compared
    guard.note(3, (9, 10, 11, 12))
    final = {4: (5, 6, 7, 8), 3: (9, 10, 11, 12)}
    assert guard.check(final.__getitem__) is True
    with pytest.raises(RuntimeError, match="moved between captures"):
        guard.note(3, (1, 2, 3, 4))
