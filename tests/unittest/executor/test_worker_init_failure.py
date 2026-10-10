from unittest.mock import Mock

import pytest

from tensorrt_llm.executor import worker as worker_module


class _FailingWorker:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("worker initialization failed")


def test_nonleader_init_failure_aborts_mpi(monkeypatch):
    comm = Mock()
    monkeypatch.setattr(worker_module, "mpi_comm", lambda: comm)
    monkeypatch.setattr(worker_module, "mpi_rank", lambda: 3)
    monkeypatch.setattr(worker_module, "set_mpi_session_cpp", Mock())

    with pytest.raises(RuntimeError, match="worker initialization failed"):
        worker_module.worker_main(
            engine=Mock(),
            worker_queues=Mock(),
            log_level="info",
            worker_cls=_FailingWorker,
        )

    assert comm.barrier.call_count == 2
    comm.Abort.assert_called_once_with(1)
