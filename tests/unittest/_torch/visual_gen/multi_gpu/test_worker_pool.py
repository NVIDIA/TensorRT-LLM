# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-path tests for the rank-process pool (_worker_pool.py), on CPU with gloo.

Every scenario in the pool's failure contract gets a case: reuse across tests,
a rank raising while its peer blocks in a collective, a rank dying without a
report, a hung test hitting the deadline, the kill switch / use cap, the between-test
reset of process state, and ranks dying with their parent.
"""

import os
import signal
import subprocess
import sys
import time

import pytest
import torch.distributed as dist

from . import _worker_pool as pool_mod
from ._worker_pool import WorkerPoolError, run_distributed

pytestmark = pytest.mark.cpu_only

# Short deadlines: these tests exercise timeouts on purpose.
_TIMEOUT = 20.0


@pytest.fixture(autouse=True)
def _fresh_pool_cache():
    pool_mod.shutdown_all()
    yield
    pool_mod.shutdown_all()


# ---- rank-side functions (module-level so they pickle by reference) ----


def _allreduce_ok(rank, world_size, value):
    import torch

    t = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(t)
    assert t.item() == value * world_size


def _rank1_raises_rank0_waits(rank, world_size):
    import torch

    if rank == 1:
        raise ValueError("boom on rank 1")
    # Rank 0 enters a collective that rank 1 never joins.
    dist.all_reduce(torch.zeros(1))


def _rank1_dies(rank, world_size):
    if rank == 1:
        os._exit(7)
    time.sleep(60)


def _hang_forever(rank, world_size):
    time.sleep(3600)


def _write_pid(rank, world_size, path):
    with open(f"{path}.{rank}", "w") as f:
        f.write(str(os.getpid()))


def _write_pid_then_hang(rank, world_size, path):
    _write_pid(rank, world_size, path)
    time.sleep(3600)


def _set_env(rank, world_size):
    os.environ["TRTLLM_TEST_POOL_LEAK"] = "1"


def _assert_env_clean(rank, world_size):
    assert "TRTLLM_TEST_POOL_LEAK" not in os.environ


def _write_random(rank, world_size, path):
    import torch

    with open(f"{path}.{rank}", "w") as f:
        f.write(repr(torch.rand(1).item()))


def _read_values(path, world_size):
    out = []
    for rank in range(world_size):
        with open(f"{path}.{rank}") as f:
            out.append(f.read())
    return out


def _make_extra_group(rank, world_size):
    from torch.distributed import distributed_c10d as c10d

    dist.new_group(list(range(world_size)))
    assert len(c10d._world.pg_map) == 2


def _assert_only_default_group(rank, world_size):
    from torch.distributed import distributed_c10d as c10d

    assert len(c10d._world.pg_map) == 1


def _read_pids(path, world_size):
    pids = []
    for rank in range(world_size):
        with open(f"{path}.{rank}") as f:
            pids.append(int(f.read()))
    return pids


# ---- tests ----


def test_reuses_ranks_across_runs(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    run_distributed(2, _allreduce_ok, 3, backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    assert _read_pids(p, 2) == first


def test_rank_failure_kills_pool_and_reports_peer(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    before = _read_pids(p, 2)

    t0 = time.monotonic()
    with pytest.raises(WorkerPoolError) as exc_info:
        run_distributed(2, _rank1_raises_rank0_waits, backend="gloo", timeout=_TIMEOUT)
    elapsed = time.monotonic() - t0
    msg = str(exc_info.value)
    assert "boom on rank 1" in msg
    assert "rank 0: no report (killed after peer failure)" in msg
    # Rank 0 was stuck in a collective; we must not have waited for the deadline.
    assert elapsed < _TIMEOUT

    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(before)


def test_dead_rank_is_detected():
    t0 = time.monotonic()
    with pytest.raises(WorkerPoolError) as exc_info:
        run_distributed(2, _rank1_dies, backend="gloo", timeout=_TIMEOUT)
    assert time.monotonic() - t0 < _TIMEOUT
    assert "rank 1: exited with code 7 without reporting" in str(exc_info.value)


def test_hung_test_hits_deadline():
    with pytest.raises(WorkerPoolError, match="timed out after 3s"):
        run_distributed(2, _hang_forever, backend="gloo", timeout=3.0)


def test_kill_switch_gives_fresh_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_REUSE_VISUALGEN_WORKERS", "0")
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(first)


def test_use_cap_retires_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_VISUALGEN_POOL_MAX_USES", "2")
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    assert _read_pids(p, 2) == first
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(first)


def test_env_change_retires_pool(tmp_path, monkeypatch):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    monkeypatch.setenv("TRTLLM_TEST_POOL_PROBE", "changed")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(first)


def test_env_is_restored_between_runs():
    run_distributed(2, _set_env, backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _assert_env_clean, backend="gloo", timeout=_TIMEOUT)


def test_rng_is_reseeded_between_runs(tmp_path):
    """Fresh ranks start from torch's per-process seed; every run after the
    first starts from the reset's fixed seed, identical on all ranks."""
    p = str(tmp_path / "rand")
    run_distributed(2, _write_random, p, backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _write_random, p, backend="gloo", timeout=_TIMEOUT)
    second = _read_values(p, 2)
    run_distributed(2, _write_random, p, backend="gloo", timeout=_TIMEOUT)
    third = _read_values(p, 2)
    assert second == third
    assert len(set(second)) == 1


def test_extra_process_groups_are_destroyed():
    run_distributed(2, _make_extra_group, backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _assert_only_default_group, backend="gloo", timeout=_TIMEOUT)


def test_reuse_false_gives_fresh_ranks_and_drains(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT)
    cached = _read_pids(p, 2)
    run_distributed(2, _write_pid, p, backend="gloo", timeout=_TIMEOUT, reuse=False)
    private = _read_pids(p, 2)
    assert set(private).isdisjoint(cached)
    assert not pool_mod._POOLS
    for pid in cached + private:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _pid_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_ranks_die_with_parent(tmp_path):
    """SIGKILL the pytest-side owner while its ranks are busy in a test body."""
    p = str(tmp_path / "pid")
    script = f"""
import signal, threading, time, os
from multi_gpu._worker_pool import run_distributed
from multi_gpu.test_worker_pool import _write_pid_then_hang
t = threading.Thread(target=run_distributed, args=(2, _write_pid_then_hang, {p!r}),
                     kwargs=dict(backend="gloo", timeout=3600), daemon=True)
t.start()
while not all(os.path.exists({p!r} + f".{{r}}") for r in range(2)):
    time.sleep(0.2)
print("READY", flush=True)
signal.pause()
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    owner = subprocess.Popen(
        [sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 120
        while owner.stdout.readline().strip() != "READY":
            assert time.monotonic() < deadline and owner.poll() is None, "owner did not start"
        ranks = _read_pids(p, 2)
        assert not any(_pid_gone(pid) for pid in ranks)
        owner.send_signal(signal.SIGKILL)
        owner.wait(timeout=10)
        deadline = time.monotonic() + 15
        while not all(_pid_gone(pid) for pid in ranks):
            assert time.monotonic() < deadline, f"ranks {ranks} outlived their parent"
            time.sleep(0.2)
    finally:
        if owner.poll() is None:
            owner.kill()
        for pid in _read_pids(p, 2):
            if not _pid_gone(pid):
                os.kill(pid, signal.SIGKILL)
