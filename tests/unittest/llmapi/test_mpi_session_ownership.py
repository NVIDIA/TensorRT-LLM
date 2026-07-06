from unittest.mock import Mock

import pytest

from tensorrt_llm.executor.utils import get_mpi_session
from tensorrt_llm.llmapi.llm import BaseLLM


def _llm_for_shutdown(executor, mpi_session, owns_mpi_session):
    llm = object.__new__(BaseLLM)
    llm._executor = executor
    llm._encoder_executor = None
    llm.mpi_session = mpi_session
    llm._owns_mpi_session = owns_mpi_session
    return llm


def test_owned_session_is_cleaned_when_executor_shutdown_fails():
    executor = Mock()
    executor.shutdown.side_effect = RuntimeError("worker failed")
    session = Mock()
    llm = _llm_for_shutdown(executor, session, owns_mpi_session=True)

    with pytest.raises(RuntimeError, match="worker failed"):
        llm.shutdown()

    executor.shutdown.assert_called_once_with()
    session.shutdown.assert_called_once_with()
    assert llm._executor is None
    assert llm.mpi_session is None


def test_borrowed_session_is_not_shutdown_by_llm():
    executor = Mock()
    session = Mock()
    llm = _llm_for_shutdown(executor, session, owns_mpi_session=False)

    llm.shutdown()

    executor.shutdown.assert_called_once_with()
    session.shutdown.assert_not_called()
    assert llm._executor is None
    assert llm.mpi_session is None


def test_external_session_is_borrowed_and_validated():
    session = Mock()
    session.n_workers = 2

    returned_session, owns_session = get_mpi_session(2, session)

    assert returned_session is session
    assert owns_session is False

    with pytest.raises(ValueError, match="needs world_size=4"):
        get_mpi_session(4, session)
