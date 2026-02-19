import pytest

from tensorrt_llm._torch.disaggregation.base.kv_transfer import (
    KVSlice,
    LayerRange,
    SessionState,
    SessionStatus,
    TokenRange,
)


def test_token_range_valid():
    tr = TokenRange(start=0, end=10)
    assert tr.start == 0
    assert tr.end == 10


def test_token_range_invalid_negative():
    with pytest.raises(ValueError, match="non-negative"):
        TokenRange(start=-1, end=5)
    with pytest.raises(ValueError, match="non-negative"):
        TokenRange(start=0, end=-1)


def test_token_range_invalid_start_ge_end():
    with pytest.raises(ValueError, match="Invalid range"):
        TokenRange(start=5, end=5)
    with pytest.raises(ValueError, match="Invalid range"):
        TokenRange(start=10, end=3)


def test_layer_range_valid():
    lr = LayerRange(start=0, end=32)
    assert lr.start == 0
    assert lr.end == 32


def test_layer_range_invalid_negative():
    with pytest.raises(ValueError, match="non-negative"):
        LayerRange(start=-1, end=5)
    with pytest.raises(ValueError, match="non-negative"):
        LayerRange(start=0, end=-1)


def test_layer_range_invalid_start_ge_end():
    with pytest.raises(ValueError, match="Invalid range"):
        LayerRange(start=5, end=5)
    with pytest.raises(ValueError, match="Invalid range"):
        LayerRange(start=10, end=3)


def test_kv_slice_construction():
    tr = TokenRange(0, 128)
    lr = LayerRange(0, 32)
    s = KVSlice(token_range=tr, layer_range=lr, block_ids=[1, 2, 3], is_last_slice=True)
    assert s.token_range == tr
    assert s.layer_range == lr
    assert s.block_ids == [1, 2, 3]
    assert s.is_last_slice is True

    # Test defaults
    s2 = KVSlice()
    assert s2.token_range is None
    assert s2.layer_range is None
    assert s2.block_ids == []
    assert s2.is_last_slice is False


def test_session_status_enum():
    expected = [
        "INIT",
        "READY",
        "TRANSFERRING",
        "TRANSFERRED",
        "AUX_TRANSFERRED",
        "COMPLETED",
        "CANCELED",
        "ERROR",
    ]
    for name in expected:
        assert hasattr(SessionStatus, name)
        assert SessionStatus[name].value == name
    assert len(SessionStatus) == 8


def test_session_state_construction():
    state = SessionState(status=SessionStatus.INIT, finished_tasks=[])
    assert state.status == SessionStatus.INIT
    assert state.finished_tasks == []

    state2 = SessionState(status=SessionStatus.COMPLETED, finished_tasks=[1, 2, 3])
    assert state2.status == SessionStatus.COMPLETED
    assert state2.finished_tasks == [1, 2, 3]
