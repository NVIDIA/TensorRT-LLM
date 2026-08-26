# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CBTS context broadcast channel."""

import contextlib
import itertools
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

COVERAGE_UTILS = (Path(__file__).resolve().parents[3] / "jenkins" / "scripts" / "cbts" /
                  "coverage_utils")
sys.path.insert(0, str(COVERAGE_UTILS))

from cbts_channel import ContextServer, ContextSubscriber  # noqa: E402


class Recorder:
    """Collects the contexts a subscriber is handed, and whether it was stopped."""

    def __init__(self):
        self.contexts = []
        self.stopped = threading.Event()
        self._seen = threading.Condition()

    def on_context(self, nodeid):
        with self._seen:
            self.contexts.append(nodeid)
            self._seen.notify_all()

    def on_stop(self):
        self.stopped.set()

    def wait_for(self, count, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self._seen:
            while len(self.contexts) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._seen.wait(remaining)
        return True


def _wait_for_subscribers(server, count, timeout=5.0):
    """connect() returns before the producer has accepted; wait for registration."""
    deadline = time.monotonic() + timeout
    while server.subscriber_count != count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.subscriber_count == count, server.subscriber_count


_ADDRESS_COUNTER = itertools.count()


@pytest.fixture
def server():
    # An explicit address per test: the production default is derived from CBTS_STAGE so both
    # ends agree without propagating it, which would make every test here share one socket.
    # Kept short because AF_UNIX paths are capped near 108 bytes.
    address = os.path.join(tempfile.gettempdir(), f"cbts-{os.getuid()}",
                           f"ut-{os.getpid()}-{next(_ADDRESS_COUNTER)}.sock")
    instance = ContextServer.start(address)
    assert instance is not None, "could not bind the context socket"
    yield instance
    instance.close(drain_timeout=2.0)


@contextlib.contextmanager
def subscribed(server, count=1):
    """One or more live subscribers, joined on exit so no reader thread outlives the test."""
    recorders = [Recorder() for _ in range(count)]
    subscribers = []
    try:
        for index, recorder in enumerate(recorders, start=1):
            subscriber = ContextSubscriber.subscribe(
                server.address, on_context=recorder.on_context, on_stop=recorder.on_stop)
            assert subscriber is not None
            subscribers.append(subscriber)
            _wait_for_subscribers(server, index)
        yield recorders if count > 1 else recorders[0]
    finally:
        for subscriber in subscribers:
            subscriber.close()


def test_subscriber_sees_the_context_current_at_subscribe_time(server):
    server.announce("suite.py::test_first", ack_timeout=0)
    with subscribed(server) as recorder:
        # Delivered during subscribe(), so a caller is on the right context before it proceeds.
        assert recorder.contexts == ["suite.py::test_first"]


def test_announcements_reach_every_subscriber(server):
    with subscribed(server, count=3) as recorders:
        server.announce("suite.py::test_second")
        for recorder in recorders:
            assert recorder.wait_for(2)
            assert recorder.contexts[-1] == "suite.py::test_second"


def test_announce_waits_until_every_subscriber_acknowledges(server):
    with subscribed(server):
        assert server.announce("suite.py::test_third", ack_timeout=5.0) is True


def test_announce_reports_when_an_acknowledgement_is_missing(server):
    """A consumer that is alive but never reads must not stall the suite indefinitely."""
    mute = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mute.connect(server.address)
    _wait_for_subscribers(server, 1)
    try:
        assert server.announce("suite.py::test_mute", ack_timeout=0.3) is False
    finally:
        mute.close()


def test_a_wedged_consumer_is_ejected_rather_than_waited_on_again(server):
    """One hung process must not charge every later announcement the ack timeout."""
    mute = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mute.connect(server.address)
    _wait_for_subscribers(server, 1)
    try:
        assert server.announce("suite.py::test_one", ack_timeout=0.3) is False
        _wait_for_subscribers(server, 0)
        started = time.monotonic()
        assert server.announce("suite.py::test_two", ack_timeout=5.0) is True
        assert time.monotonic() - started < 1.0, "producer waited on the ejected consumer"
    finally:
        mute.close()


def test_unsubscribe_stops_further_delivery(server):
    recorder = Recorder()
    subscriber = ContextSubscriber.subscribe(
        server.address, on_context=recorder.on_context, on_stop=recorder.on_stop)
    assert subscriber is not None
    _wait_for_subscribers(server, 1)
    subscriber.close()
    _wait_for_subscribers(server, 0)

    delivered = len(recorder.contexts)
    server.announce("suite.py::test_after_unsubscribe", ack_timeout=0)
    time.sleep(0.2)
    assert len(recorder.contexts) == delivered


def test_a_crashed_consumer_is_dropped_without_a_timeout(server):
    """The kernel closes a dead process's socket, so the producer never waits on it."""
    victim = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    victim.connect(server.address)
    _wait_for_subscribers(server, 1)
    # Abort without any protocol-level goodbye, as a SIGKILLed process would.
    victim.close()
    _wait_for_subscribers(server, 0)
    started = time.monotonic()
    assert server.announce("suite.py::test_after_crash", ack_timeout=5.0) is True
    assert time.monotonic() - started < 1.0, "producer waited on a dead consumer"


def test_stop_ends_subscribers_and_refuses_new_ones(server):
    recorder = Recorder()
    subscriber = ContextSubscriber.subscribe(
        server.address, on_context=recorder.on_context, on_stop=recorder.on_stop)
    assert subscriber is not None
    _wait_for_subscribers(server, 1)
    try:
        assert server.close(drain_timeout=5.0) is True
        assert recorder.stopped.wait(5.0), "subscriber was not told to stop"
    finally:
        subscriber.close()

    late = Recorder()
    assert ContextSubscriber.subscribe(
        server.address, on_context=late.on_context, on_stop=late.on_stop) is None


def test_producer_exit_stops_a_subscriber(server):
    """A consumer that outlives the producer still saves rather than hanging."""
    recorder = Recorder()
    subscriber = ContextSubscriber.subscribe(
        server.address, on_context=recorder.on_context, on_stop=recorder.on_stop)
    assert subscriber is not None
    _wait_for_subscribers(server, 1)
    try:
        server.close(drain_timeout=0.0)
        assert recorder.stopped.wait(5.0)
    finally:
        subscriber.close()


def test_subscribe_without_a_producer_returns_none(tmp_path):
    recorder = Recorder()
    assert ContextSubscriber.subscribe(
        str(tmp_path / "absent.sock"),
        on_context=recorder.on_context,
        on_stop=recorder.on_stop,
    ) is None
