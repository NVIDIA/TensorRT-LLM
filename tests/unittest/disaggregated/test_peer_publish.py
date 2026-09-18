# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the send side reaches the common contract.

The mirror of the receive adapter: one piece per call and a handle back for each. What differs is
where the session comes from -- a send session may already exist from an earlier chunk, so the
adapter is built around one instead of opening it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base import (
    CacheExtent,
    CacheKind,
    Chunk,
    Delivered,
    Publishes,
    TokenRange,
)
from tensorrt_llm._torch.disaggregation.native.publish import PeerPublish
from tensorrt_llm._torch.disaggregation.native.transfer import TaskStatus, TxSession

pytestmark = pytest.mark.cpu_only

TOKENS = 8


def _stub_sender():
    """Only what a send session calls on its sender; dispatch is what the peers would do."""
    sender = MagicMock()
    sender._get_req_info = MagicMock(return_value=None)
    return sender


def _request(rid: int = 42):
    """Only what ``get_unique_rid`` reads off a request."""
    return SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid), request_id=rid
    )


def _extent(rid: int = 42, end: int = TOKENS, is_last: bool = True) -> CacheExtent:
    return CacheExtent(
        name=rid,
        local=Chunk(
            block_ids_per_layer_groups=[np.array([0, 1], dtype=np.int64)],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(start=0, end=end),
            is_last=is_last,
        ),
    )


def _sending(rid: int = 42) -> TxSession:
    return TxSession(
        request_id=rid,
        params=DisaggregatedParams(disagg_request_id=rid),
        sender=_stub_sender(),
        prompt_len=TOKENS,
    )


def test_the_adapter_satisfies_the_contract():
    assert isinstance(PeerPublish(_sending(), _request()), Publishes)


def test_each_piece_gets_its_own_handle():
    """Pieces of one request share a session, so a handle has to hold the piece, not the session."""
    session = _sending()
    publish = PeerPublish(session, _request())

    first = publish.publish(_extent(end=4, is_last=False))
    second = publish.publish(_extent(end=TOKENS))

    assert len(session.kv_tasks) == 2
    assert first is not second
    session.kv_tasks[0].complete()
    assert isinstance(first.poll(), Delivered)
    assert first.poll().token_end == 4
    assert second.poll() is None


def test_the_piece_offered_is_the_extent_given():
    session = _sending()

    PeerPublish(session, _request()).publish(_extent())

    assert session.kv_tasks[0]._chunk.token_range.end == TOKENS
    assert session.kv_tasks[0].status is TaskStatus.INIT


def test_the_session_stays_reachable_for_what_the_contract_skips():
    """Packing the auxiliary buffer and the sweep's own closing are not on this protocol."""
    session = _sending()

    assert PeerPublish(session, _request()).session is session


def test_an_extent_for_another_request_is_refused():
    """Checked against the derivation the builder used, not against the session's id.

    Those two differ whenever a request carries a context id but no disaggregated one, and the
    session's is the name the peer resolves rather than the one the extent was built with.
    """
    session = _sending()
    extent = _extent()
    extent.name = 99

    with pytest.raises(ValueError, match="bound to 42"):
        PeerPublish(session, _request(42)).publish(extent)
    assert not session.kv_tasks
