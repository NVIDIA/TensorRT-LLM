# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collecting the device work of a step's input preparation."""

import threading

import pytest

from tensorrt_llm._torch.utils import (
    begin_device_work,
    end_device_work,
    run_device_work,
    run_device_work_items,
)

pytestmark = pytest.mark.cpu_only


def test_runs_immediately_outside_a_region():
    calls = []
    run_device_work(calls.append, "a")
    assert calls == ["a"]


def test_collects_in_order_without_running():
    calls = []
    work = begin_device_work()
    try:
        run_device_work(calls.append, "a")
        run_device_work(calls.append, "b")
        assert calls == []
        assert len(work) == 2
    finally:
        end_device_work()
    assert calls == []

    run_device_work_items(work)
    assert calls == ["a", "b"]


def test_forwards_positional_and_keyword_arguments():
    seen = []

    def record(first, second, *, third):
        seen.append((first, second, third))

    work = begin_device_work()
    try:
        run_device_work(record, 1, 2, third=3)
        run_device_work(record, 4, 5, third=6)
    finally:
        end_device_work()
    run_device_work_items(work)
    assert seen == [(1, 2, 3), (4, 5, 6)]


def test_collected_work_can_be_replayed():
    calls = []
    work = begin_device_work()
    try:
        run_device_work(calls.append, "a")
    finally:
        end_device_work()
    run_device_work_items(work)
    run_device_work_items(work)
    assert calls == ["a", "a"]


def test_a_replaying_region_drops_the_work():
    calls = []
    work = begin_device_work(collect=False)
    try:
        assert work is None
        run_device_work(calls.append, "a")
        run_device_work(calls.append, "b")
    finally:
        end_device_work()
    # Nothing ran and nothing was recorded: the caller's graph is what
    # performs these.
    assert calls == []

    # The region really closed.
    run_device_work(calls.append, "c")
    assert calls == ["c"]


def test_region_ends_after_an_exception():
    calls = []
    with pytest.raises(ValueError):
        begin_device_work()
        try:
            run_device_work(calls.append, "a")
            raise ValueError
        finally:
            end_device_work()
    assert calls == []

    run_device_work(calls.append, "b")
    assert calls == ["b"]


def test_regions_do_not_nest():
    begin_device_work()
    try:
        with pytest.raises(AssertionError):
            begin_device_work()
        # The failed inner region left the outer one intact.
        calls = []
        run_device_work(calls.append, "a")
        assert calls == []
    finally:
        end_device_work()


def test_a_region_does_not_reach_another_thread():
    calls = []
    work = begin_device_work()
    try:
        thread = threading.Thread(target=run_device_work, args=(calls.append, "other"))
        thread.start()
        thread.join()
    finally:
        end_device_work()
    assert calls == ["other"]
    assert work == []
