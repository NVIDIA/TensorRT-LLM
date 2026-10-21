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
"""Host-side issue order of the DWDP expert prefetch (``PrefetchSchedule``)."""

import random
from typing import Dict, List, Optional

import pytest

from tensorrt_llm._torch.modules.dwdp.prefetch_schedule import PrefetchSchedule, ScheduleStep

pytestmark = pytest.mark.cpu_only


def _schedule(num_layers: int, num_slots: int, repeated=()) -> PrefetchSchedule:
    """The weight buffer's mapping: MoE order ``o`` uses slot ``o % num_slots``."""
    return PrefetchSchedule([order % num_slots for order in range(num_layers)], repeated)


def _run_pass(schedule: PrefetchSchedule, num_layers: int) -> None:
    """One complete forward pass in MoE order."""
    schedule.start_forward()
    for order in range(num_layers):
        schedule.bind(order)
        assert schedule.finish(order).release == schedule.slot(order)


@pytest.mark.parametrize("num_slots", [2, 3, 4])
@pytest.mark.parametrize("num_layers", list(range(1, 13)) + [92])
def test_each_layer_is_gathered_once_per_pass(num_layers, num_slots):
    schedule = _schedule(num_layers, num_slots)
    first = schedule.start_forward()
    assert first.prefetch == list(range(min(num_layers, num_slots)))
    issued = list(first.prefetch)
    for order in range(num_layers):
        issued += schedule.bind(order).prefetch
        step = schedule.finish(order)
        assert step.release == order % num_slots
        issued += step.prefetch
    for _ in range(3):
        # Steady state: the previous pass already issued the first layers of this one.
        assert schedule.start_forward().prefetch == []
        for order in range(num_layers):
            assert schedule.bind(order).prefetch == []
            issued += schedule.finish(order).prefetch
    # One cyclic sequence: every pass gathers each layer once, in MoE order.
    assert issued[: 4 * num_layers] == list(range(num_layers)) * 4
    assert schedule.num_restarts == 0


@pytest.mark.parametrize(
    "num_slots, wrap",
    [
        (2, {}),
        (4, {}),
        # 92 = 30 * 3 + 2: orders 90 and 91 hold slots 0 and 1, so the next pass's order 0 waits
        # for MoE(90) and orders 1 and 2 for MoE(91).
        (3, {89: [], 90: [0], 91: [1, 2]}),
    ],
)
def test_finish_issues_the_layer_num_slots_ahead_and_wraps(num_slots, wrap):
    num_layers = 92
    schedule = _schedule(num_layers, num_slots)
    _run_pass(schedule, num_layers)
    assert schedule.start_forward().prefetch == []
    for order in range(num_layers):
        assert schedule.bind(order).prefetch == []
        expected = wrap.get(order, [(order + num_slots) % num_layers])
        assert schedule.finish(order).prefetch == expected


def test_two_slots_wrap_explicitly():
    # 5 layers, 2 slots: slots 0 1 0 1 0 | 0 1 0 1 0.
    schedule = _schedule(5, 2)
    assert schedule.start_forward().prefetch == [0, 1]
    expected_after_finish = {0: [2], 1: [3], 2: [4], 3: [], 4: [0, 1]}
    for order in range(5):
        assert schedule.bind(order).prefetch == []
        assert schedule.finish(order).prefetch == expected_after_finish[order]
    assert schedule.start_forward().prefetch == []
    # 4 layers, 3 slots: slots 0 1 2 0 | 0 1 2 0.
    schedule = _schedule(4, 3)
    assert schedule.start_forward().prefetch == [0, 1, 2]
    expected_after_finish = {0: [3], 1: [], 2: [], 3: [0, 1, 2]}
    for order in range(4):
        schedule.bind(order)
        assert schedule.finish(order).prefetch == expected_after_finish[order]
    assert schedule.start_forward().prefetch == []


def test_pass_without_start_forward_issues_on_first_bind():
    schedule = _schedule(4, 2)
    step = schedule.bind(0)
    assert step == ScheduleStep(False, None, [0, 1])
    schedule.finish(0)
    assert schedule.num_restarts == 0


def test_out_of_order_bind_restarts_at_that_layer():
    schedule = _schedule(6, 2)
    _ = schedule.start_forward()
    schedule.bind(0)
    schedule.finish(0)
    step = schedule.bind(3)
    assert step == ScheduleStep(False, None, [3, 4])
    assert schedule.num_restarts == 1
    assert schedule.finish(3).prefetch == [5]


def test_pass_stopped_inside_the_sequence_restarts_at_start_forward():
    schedule = _schedule(6, 2)
    schedule.start_forward()
    for order in range(3):
        schedule.bind(order)
        schedule.finish(order)
    step = schedule.start_forward()
    assert step == ScheduleStep(False, None, [0, 1])
    assert schedule.num_restarts == 1


def test_abandoned_bind_requests_a_device_sync():
    schedule = _schedule(4, 2)
    schedule.start_forward()
    schedule.bind(0)
    # The MoE of layer 0 raised before finish: its slot may still be read.
    step = schedule.start_forward()
    assert step == ScheduleStep(True, None, [0, 1])
    schedule.bind(0)
    assert schedule.bind(0).synchronize


def test_finish_without_matching_bind_raises():
    schedule = _schedule(4, 2)
    with pytest.raises(RuntimeError):
        schedule.finish(0)
    schedule.bind(0)
    with pytest.raises(RuntimeError):
        schedule.finish(1)
    with pytest.raises(ValueError):
        schedule.bind(4)


def test_invalidate_reissues_the_first_layers():
    schedule = _schedule(4, 2)
    _run_pass(schedule, 4)
    assert schedule.invalidate() == ScheduleStep(False, None, [])
    assert schedule.start_forward().prefetch == [0, 1]
    assert schedule.num_restarts == 0
    schedule.bind(0)
    assert schedule.invalidate().synchronize


def test_constructor_validates_arguments():
    with pytest.raises(ValueError):
        PrefetchSchedule([])
    with pytest.raises(ValueError):
        PrefetchSchedule([0, -1])
    with pytest.raises(ValueError):
        PrefetchSchedule([0, 1], repeated_orders=[2])


@pytest.mark.parametrize("num_slots", [2, 3])
@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_repeated_layer_rebinds_its_resident_copy(num_slots, repeats):
    """MTP-Eagle pattern ``..., N, N, N``: every repeat reads the copy still in the slot."""
    num_layers = 5
    last = num_layers - 1
    schedule = _schedule(num_layers, num_slots, repeated=[last])
    model = _SlotModel(num_layers, num_slots)
    copies = []
    for _ in range(4):
        step = schedule.start_forward()
        model.apply(step)
        copies += step.prefetch
        for order in list(range(num_layers)) + [last] * (repeats - 1):
            step = schedule.bind(order)
            model.apply(step)
            model.read(order)
            copies += step.prefetch
            step = schedule.finish(order)
            if order == last:
                # The slot stays held: no release and no copy until a different layer runs.
                assert step == ScheduleStep(False, None, [])
            model.apply(step)
            copies += step.prefetch
    assert copies[: 4 * num_layers] == list(range(num_layers)) * 4
    assert schedule.num_restarts == 0


def test_held_slot_is_released_by_the_next_call():
    # 5 layers, 2 slots: order 4 holds slot 0 after it finishes.
    schedule = _schedule(5, 2, repeated=[4])
    schedule.start_forward()
    for order in range(5):
        schedule.bind(order)
        schedule.finish(order)
    assert schedule.bind(4) == ScheduleStep(False, None, [])
    assert schedule.finish(4) == ScheduleStep(False, None, [])
    # The next pass releases slot 0 before it reissues the slot.
    assert schedule.start_forward() == ScheduleStep(False, 0, [0, 1])
    # A bind of the next layer releases it too (pass without start_forward).
    for order in range(5):
        schedule.bind(order)
        schedule.finish(order)
    assert schedule.bind(0) == ScheduleStep(False, 0, [0, 1])
    schedule.finish(0)
    # An abandoned repeat still synchronizes before its slot is reused.
    for order in range(1, 5):
        schedule.bind(order)
        schedule.finish(order)
    schedule.bind(4)
    assert schedule.start_forward() == ScheduleStep(True, 0, [0, 1])
    assert schedule.num_restarts == 1


def test_undeclared_repeat_restarts():
    """Without the declaration a repeat is out of sequence and gathers the layer again."""
    schedule = _schedule(5, 2)
    schedule.start_forward()
    for order in range(5):
        schedule.bind(order)
        schedule.finish(order)
    assert schedule.bind(4).prefetch == [4]
    assert schedule.num_restarts == 1


def test_slots_come_from_the_given_mapping():
    """The schedule uses the slot table it is given, not a formula of its own."""
    slots = [1, 0, 2, 1, 0, 2]
    schedule = PrefetchSchedule(slots)
    assert [schedule.slot(order) for order in range(len(slots))] == slots
    assert schedule.start_forward().prefetch == [0, 1, 2]
    for order in range(len(slots)):
        schedule.bind(order)
        assert schedule.finish(order).release == slots[order]


class _SlotModel:
    """Device-side model of the slots: checks RAW at every bind and WAR at every copy.

    A copy into a slot may only start after every MoE that read the slot released it (WAR), unless
    the caller synchronized the device; a bind must find its own layer as the last copy into the
    slot (RAW).
    """

    def __init__(self, num_layers: int, num_slots: int) -> None:
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.last_copy: List[Optional[int]] = [None] * num_slots
        # Readers of each slot whose release was not recorded yet.
        self.unreleased: List[set] = [set() for _ in range(num_slots)]
        self.copies_per_pass: Dict[int, int] = {}

    def apply(self, step: ScheduleStep) -> None:
        if step.synchronize:
            for readers in self.unreleased:
                readers.clear()
        if step.release is not None:
            self.unreleased[step.release].clear()
        for order in step.prefetch:
            slot = order % self.num_slots
            assert not self.unreleased[slot], f"WAR: copy of {order} into slot {slot} in use"
            self.last_copy[slot] = order
            self.copies_per_pass[order] = self.copies_per_pass.get(order, 0) + 1

    def read(self, order: int) -> None:
        slot = order % self.num_slots
        assert self.last_copy[slot] == order, f"RAW: slot {slot} holds {self.last_copy[slot]}"
        self.unreleased[slot].add(order)


@pytest.mark.parametrize("seed", range(40))
def test_randomized_passes_keep_raw_and_war(seed):
    rng = random.Random(seed)
    num_layers = rng.randint(1, 9)
    num_slots = rng.randint(2, 4)
    # Half of the seeds repeat the last layer of each pass, as one-model MTP-Eagle does.
    repeated = [num_layers - 1] if seed % 2 else []
    schedule = _schedule(num_layers, num_slots, repeated)
    model = _SlotModel(num_layers, num_slots)
    for _ in range(40):
        kind = rng.random()
        if rng.random() < 0.8:
            model.apply(schedule.start_forward())
        if kind < 0.1:
            model.apply(schedule.invalidate())
            continue
        stop = num_layers
        if kind < 0.3:
            stop = rng.randint(0, num_layers)
        orders = list(range(stop))
        if kind > 0.95 and orders:
            orders = sorted(rng.sample(orders, rng.randint(1, len(orders))))
        if repeated and orders and orders[-1] == repeated[0]:
            orders += repeated * rng.randint(0, 3)
        for order in orders:
            model.apply(schedule.bind(order))
            model.read(order)
            if kind < 0.2 and order == orders[-1]:
                break  # The MoE raised: no finish.
            model.apply(schedule.finish(order))
