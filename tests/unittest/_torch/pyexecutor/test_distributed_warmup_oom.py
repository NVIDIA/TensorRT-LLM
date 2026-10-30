# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module

pytestmark = pytest.mark.cpu_only


@contextlib.contextmanager
def _guard_env(
    *, global_size: int, symmetric_crash: bool = False
) -> Iterator[tuple[mock.Mock, mock.Mock]]:
    with (
        mock.patch.object(py_executor_module, "start_rank_crash_kill_watchdog") as watchdog,
        mock.patch.object(py_executor_module, "propagate_hard_kill") as hard_kill,
        mock.patch.object(py_executor_module, "global_mpi_size", return_value=global_size),
        mock.patch.object(
            py_executor_module, "all_ranks_crashed", return_value=symmetric_crash
        ) as probe,
    ):
        yield watchdog, hard_kill, probe


def _run_guard(*, world_size: int, dwdp_size: int, error: BaseException) -> None:
    dist = SimpleNamespace(world_size=world_size)
    mapping = SimpleNamespace(dwdp_enabled=dwdp_size > 1)
    with py_executor_module._distributed_warmup_guard(dist, mapping):
        raise error


@pytest.mark.parametrize(
    "world_size,dwdp_size,global_size,expected_peer_count",
    [
        (1, 0, 1, None),
        (2, 0, 1, 2),
        (1, 4, 4, 4),
    ],
)
def test_guard_topology_policy(
    world_size: int,
    dwdp_size: int,
    global_size: int,
    expected_peer_count: int | None,
) -> None:
    error = ValueError("warmup failed")
    with _guard_env(global_size=global_size) as (watchdog, hard_kill, _probe):
        with pytest.raises(ValueError) as excinfo:
            _run_guard(world_size=world_size, dwdp_size=dwdp_size, error=error)

    assert excinfo.value is error
    # The watchdog's grace period is what lets the setup/RPC path report the
    # original exception; aborting the world here would replace it with an
    # exit code.
    hard_kill.assert_not_called()
    if expected_peer_count is None:
        watchdog.assert_not_called()
    else:
        watchdog.assert_called_once_with(expected_peer_count, error_delivered=None)


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_guard_leaves_teardown_signals_unarmed(signal: type) -> None:
    with _guard_env(global_size=4) as (watchdog, hard_kill, _probe):
        with pytest.raises(signal):
            _run_guard(world_size=4, dwdp_size=0, error=signal())

    watchdog.assert_not_called()
    hard_kill.assert_not_called()


def test_guard_skips_the_kill_on_a_symmetric_crash() -> None:
    """Every rank crashed the same way: nobody is stranded, so no kill."""
    error = ValueError("warmup failed on every rank")
    with _guard_env(global_size=4, symmetric_crash=True) as (
        watchdog,
        hard_kill,
        probe,
    ):
        with pytest.raises(ValueError):
            _run_guard(world_size=4, dwdp_size=0, error=error)

    probe.assert_called_once_with(4)
    watchdog.assert_not_called()
    hard_kill.assert_not_called()


def test_guard_arms_the_kill_when_the_crash_is_not_proven_symmetric() -> None:
    """A False probe (stranded peer, MPI unavailable, timeout) keeps the kill."""
    with _guard_env(global_size=4, symmetric_crash=False) as (
        watchdog,
        _hard_kill,
        probe,
    ):
        with pytest.raises(ValueError):
            _run_guard(world_size=4, dwdp_size=0, error=ValueError("boom"))

    probe.assert_called_once_with(4)
    watchdog.assert_called_once_with(4, error_delivered=None)
