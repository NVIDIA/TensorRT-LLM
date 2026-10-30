# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict, deque

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.prefix_load_completion import (
    PrefixLoadCompletionTracker,
)

pytestmark = pytest.mark.cpu_only


class Report:
    def __init__(self, ids):
        self.ids = ids
        self.received = False
        self.ready = True

    def test(self):
        return self.received, None

    def wait(self):
        assert self.received

    def irecv(self):
        return ReceivedReport(self)


class ReceivedReport:
    def __init__(self, report):
        self.report = report

    def test(self):
        if not self.report.ready:
            return False, None
        self.report.received = True
        return True, self.report.ids

    def wait(self):
        assert self.report.ready


class Mailbox:
    def __init__(self, rank, queues, size=2):
        self.rank = rank
        self.queues = queues
        self.size = size
        self.probes = 0
        self.sent = []
        self.freed = False

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def isend(self, ids, dest, tag):
        assert dest == 0 and tag == 0
        report = Report(ids)
        self.queues[self.rank].append(report)
        self.sent.append(report)
        return report

    def improbe(self, source, tag):
        assert tag == 0
        self.probes += 1
        return self.queues[source].popleft() if self.queues[source] else None

    def Free(self):
        self.freed = True


@pytest.fixture
def trackers():
    queues = defaultdict(deque)
    return [PrefixLoadCompletionTracker(Mailbox(rank, queues)) for rank in range(2)]


def test_slow_worker_does_not_block_other_completed_loads(trackers):
    leader, worker = trackers
    for tracker in trackers:
        tracker.track(1)
        tracker.track(2)
    leader.report({1, 2})
    worker.report({2})
    worker.poll()
    assert leader.take_completed() == [2]
    leader.forget(2)
    assert leader.take_completed() == []
    worker.report({1})
    worker.poll()
    assert leader.take_completed() == [1]


def test_pending_send_preserves_new_reports_without_waiting(trackers):
    leader, worker = trackers
    for reservation_id in (1, 2):
        leader.track(reservation_id)
    leader.report({1, 2})
    worker.report({1})
    worker.poll()
    worker.report({2})
    worker.poll()
    assert len(worker._comm.sent) == 1
    assert worker._comm.sent[0].ids == {1}
    assert leader.take_completed() == [1]
    worker.poll()
    assert leader.take_completed() == [2]


def test_matched_receive_is_polled_without_waiting(trackers):
    leader, worker = trackers
    leader.track(1)
    leader.report({1})
    worker.report({1})
    worker.poll()
    report = worker._comm.sent[0]
    report.ready = False
    assert leader.take_completed() == []
    assert leader.take_completed() == []
    report.ready = True
    assert leader.take_completed() == [1]


def test_stale_or_duplicate_reports_cannot_retire_another_load(trackers):
    leader, worker = trackers
    leader.track(2)
    leader.report({2})
    worker.report({1, 1000})
    worker.poll()
    assert leader.take_completed() == []
    worker.report({2})
    worker.poll()
    assert leader.take_completed() == [2]
    leader.forget(2)
    worker.report({2})
    worker.poll()
    leader.track(3)
    leader.report({3})
    assert leader.take_completed() == []


def test_idle_poll_does_not_send_or_probe(trackers):
    for tracker in trackers:
        tracker.poll()
        assert tracker.take_completed() == []
        assert tracker._comm.probes == 0
        assert tracker._comm.sent == []


def test_single_worker_needs_no_communicator():
    tracker = PrefixLoadCompletionTracker()
    tracker.track(4)
    assert tracker.take_completed() == []
    tracker.report({4})
    assert tracker.take_completed() == [4]
    tracker.forget(4)
    tracker.report({4})
    assert tracker.take_completed() == []
    tracker.close()


def test_shutdown_completes_delivered_sends_and_frees_communicator(trackers):
    leader, worker = trackers
    leader.track(1)
    leader.report({1})
    worker.report({1})
    worker.poll()
    assert leader.take_completed() == [1]
    for tracker in trackers:
        comm = tracker._comm
        tracker.close()
        assert comm.freed
