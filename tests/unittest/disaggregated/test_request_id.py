"""Tests for _get_request_id in ExecutorRequestQueue.

Demonstrates the known bug: disagg_request_id=0 is falsy and gets skipped.
"""

from unittest.mock import MagicMock


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
