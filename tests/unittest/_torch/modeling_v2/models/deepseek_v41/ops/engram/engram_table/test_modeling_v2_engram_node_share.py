# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The node-shared Engram table's ``NodeShare`` over real abstract ``SOCK_SEQPACKET`` sockets, on CPU.

The targets' table is the engine's, so its ``NodeShare`` is the engine's ``modules/engram/_node_share.py`` (no family
copy). Proved here: the targets' table joins through it, with its 30-minute owner deadline; a joiner of an owner that
listens but never publishes, and of an owner that publishes but never marks the content ready, gives up with
``TimeoutError`` once the owner has been silent for ``_OWNER_TIMEOUT_S`` (the table then loads a private copy) instead
of blocking the weight load; an owner that publishes and marks ready within the bound is shared normally, and an owner
that leaves before marking ready releases its joiner with ``False`` (the joiner elects again).
"""

import os
import sys
import threading
import uuid

import pytest

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
)
from tensorrt_llm._torch.modules.engram import _node_share as node_share
from tensorrt_llm._torch.modules.engram._node_share import NodeShare

pytestmark = pytest.mark.cpu_only

# Every joiner call must return or raise within this; a hang fails the test instead of blocking the suite.
_BOUND_S = 10.0


@pytest.fixture
def short_owner_timeout(monkeypatch):
    """An owner silent for 0.5 s is given up on (stock: 30 min), checked every 0.1 s."""
    monkeypatch.setattr(node_share, "_OWNER_TIMEOUT_S", 0.5)
    monkeypatch.setattr(node_share, "_WAIT_LOG_INTERVAL_S", 0.1)


def test_the_targets_table_joins_through_the_engines_node_share():
    """The family's copy of ``NodeShare`` is gone: the targets' table elects and waits on the engine's, which gives up
    on a silent owner after 30 minutes (the engine's deadline), logging every minute."""
    impl = sys.modules[ShardedFp8MultiHeadEmbedding._load_node_shared.__module__]
    assert impl.NodeShare is NodeShare
    assert node_share._OWNER_TIMEOUT_S == 30 * 60
    assert node_share._WAIT_LOG_INTERVAL_S == 60


def _name() -> str:
    return f"trtllm-test-engram-node-share-{os.getpid()}-{uuid.uuid4().hex}"


def _bounded(call):
    """Run ``call`` on a daemon thread; return ``("returned", value)`` or ``("raised", error)``, or fail the test
    if it is still blocked after ``_BOUND_S``."""
    result = {}

    def run():
        try:
            result["returned"] = call()
        except BaseException as error:  # noqa: BLE001 - reported to the test
            result["raised"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(_BOUND_S)
    assert not thread.is_alive(), f"joiner still blocked after {_BOUND_S} s"
    ((kind, value),) = result.items()
    return kind, value


def _publish(owner: NodeShare) -> None:
    read_end, write_end = os.pipe()
    os.close(write_end)
    owner.publish([read_end], {"size": 1, "granularity": None})


def test_owner_that_never_publishes_times_out(short_owner_timeout):
    name = _name()
    owner = NodeShare(name)
    assert owner.join(0.0) is None  # claimed: listening, not yet serving
    joiner = NodeShare(name)
    try:
        kind, value = _bounded(lambda: joiner.join(0.0))
        assert kind == "raised" and isinstance(value, TimeoutError), (kind, value)
        assert "publish" in str(value)
    finally:
        joiner.close()
        owner.close()


def test_owner_that_never_marks_ready_times_out(short_owner_timeout):
    name = _name()
    owner = NodeShare(name)
    assert owner.join(0.0) is None
    _publish(owner)
    joiner = NodeShare(name)
    try:
        kind, received = _bounded(lambda: joiner.join(0.0))
        assert kind == "returned" and received is not None
        fds, meta = received
        for fd in fds:
            os.close(fd)
        assert meta == {"size": 1, "granularity": None}
        kind, value = _bounded(joiner.wait_ready)
        assert kind == "raised" and isinstance(value, TimeoutError), (kind, value)
        assert "finish" in str(value)
    finally:
        joiner.close()
        owner.close()


def test_owner_that_marks_ready_is_shared(short_owner_timeout):
    name = _name()
    owner = NodeShare(name)
    assert owner.join(0.0) is None
    _publish(owner)
    joiner = NodeShare(name)
    try:
        kind, received = _bounded(lambda: joiner.join(0.0))
        assert kind == "returned" and received is not None
        for fd in received[0]:
            os.close(fd)
        timer = threading.Timer(
            0.2, owner.mark_ready
        )  # within the bound, after the joiner started waiting
        timer.start()
        assert _bounded(joiner.wait_ready) == ("returned", True)
        timer.join()
    finally:
        joiner.close()
        owner.close()


def test_owner_that_leaves_before_ready_releases_the_joiner(short_owner_timeout):
    name = _name()
    owner = NodeShare(name)
    assert owner.join(0.0) is None
    _publish(owner)
    joiner = NodeShare(name)
    try:
        kind, received = _bounded(lambda: joiner.join(0.0))
        assert kind == "returned" and received is not None
        for fd in received[0]:
            os.close(fd)
        timer = threading.Timer(0.2, owner.close)
        timer.start()
        assert _bounded(joiner.wait_ready) == ("returned", False)
        timer.join()
    finally:
        joiner.close()
        owner.close()
