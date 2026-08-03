"""Tests for disaggregated request-id handling.

Covers ExecutorRequestQueue._get_request_id and the Receiver's sender_req_id
fallback in the native KV transceiver (nvbugs/6482576).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_queue(max_batch_size=128):
    """Create an ExecutorRequestQueue with mocked Distributed."""
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue

    mock_dist = MagicMock()
    mock_dist.rank = 0
    return ExecutorRequestQueue(
        dist=mock_dist,
        max_batch_size=max_batch_size,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
    )


def test_get_request_id_with_disagg_id():
    """When disagg_request_id is a nonzero int, it is returned directly."""
    q = _make_queue()
    mock_request = MagicMock()
    mock_request.disagg_request_id = 42

    rid = q._get_request_id(mock_request)
    assert rid == 42


def test_get_request_id_without_disagg_id():
    """When disagg_request_id is None, auto-increment is used."""
    q = _make_queue(max_batch_size=128)
    mock_request = MagicMock()
    mock_request.disagg_request_id = None

    rid = q._get_request_id(mock_request)
    # Should return next_request_id which starts at max_batch_size
    assert rid == 128


def test_get_request_id_zero_bug():
    """BUG: disagg_request_id=0 is falsy so it falls through to auto-increment.

    This documents the known issue: `if request and request.disagg_request_id`
    evaluates to False when disagg_request_id is 0, because 0 is falsy in Python.
    The correct check should be `is not None`.
    """
    q = _make_queue(max_batch_size=128)
    mock_request = MagicMock()
    mock_request.disagg_request_id = 0

    rid = q._get_request_id(mock_request)
    # BUG: should return 0, but returns auto-incremented id instead
    assert rid != 0, "If this fails, the bug has been fixed — update this test"
    assert rid == 128  # falls through to auto-increment


# --------------------------------------------------------------------------- #
# Receiver._build_recv_req_info: sender_req_id fallback (nvbugs/6482576)
# --------------------------------------------------------------------------- #
def _make_recv_task(ctx_request_id, disagg_request_id, unique_rid=123):
    """Minimal stand-in for KVRecvTask with only the fields the method reads."""
    return SimpleNamespace(
        _params=SimpleNamespace(ctx_request_id=ctx_request_id, disagg_request_id=disagg_request_id),
        _unique_rid=unique_rid,
        _kv_slice=SimpleNamespace(block_ids_per_layer_groups=[], mamba_state_index=None),
        _aux_slot=None,
        slice_id=0,
    )


def _build_recv_req_info(tfr, task):
    """Call the unbound Receiver method against a mocked registrar."""
    recv_self = SimpleNamespace(
        _registrar=SimpleNamespace(
            self_rank_info=SimpleNamespace(instance_name="gen-0", instance_rank=0)
        )
    )
    return tfr.Receiver._build_recv_req_info(recv_self, task)


def test_build_recv_req_info_prefers_ctx_request_id():
    """Normal disagg flow: ctx_request_id keys the sender's TxSession."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    info = _build_recv_req_info(tfr, _make_recv_task(ctx_request_id=7, disagg_request_id=99))
    assert info.sender_req_id == 7


def test_build_recv_req_info_falls_back_to_disagg_request_id():
    """nvbugs/6482576: fall back to disagg_request_id when ctx_request_id is None."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    info = _build_recv_req_info(tfr, _make_recv_task(ctx_request_id=None, disagg_request_id=99))
    assert info.sender_req_id == 99


def test_build_recv_req_info_both_ids_none_raises():
    """Raise when neither request id is available (survives python -O)."""
    tfr = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
    with pytest.raises(ValueError):
        _build_recv_req_info(tfr, _make_recv_task(ctx_request_id=None, disagg_request_id=None))
