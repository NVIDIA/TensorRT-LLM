# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-path tests for the rank-process pool (_worker_pool.py), on CPU with gloo.

Every scenario in the pool's failure contract gets a case: reuse across tests,
a rank raising while its peer blocks in a collective, a rank dying without a
report, a hung test hitting the deadline, the pool lifetime knob, the between-test
reset of process state, and ranks dying with their parent.
"""

import multiprocessing
import os
import pickle
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch.distributed as dist

from . import _worker_pool as pool_mod
from ._worker_pool import WorkerPoolError, run_distributed

pytestmark = pytest.mark.cpu_only

# Short deadlines: these tests exercise timeouts on purpose.
_TIMEOUT = 20.0


@pytest.fixture(autouse=True)
def _fresh_pool_cache(monkeypatch):
    # These tests assert reuse semantics; pin the lifetime they assume rather
    # than inheriting whatever the environment set (under lifetime "test" the
    # reset tests would pass without the reset ever running).
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("pool reuse is forced off under pytest-xdist")
    monkeypatch.setenv("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "session")
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


def _read_pids(path, world_size):
    return [int(v) for v in _read_values(path, world_size)]


def _proc_state(pid):
    """Kernel process state letter, or None if the pid no longer exists."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _pid_gone(pid):
    """Dead, counting zombies: an orphan reparented to a PID 1 that never reaps
    (as in some CI containers) stays a zombie forever but holds nothing."""
    if sys.platform.startswith("linux"):
        return _proc_state(pid) in (None, "Z", "X")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _make_extra_group(rank, world_size):
    from torch.distributed import distributed_c10d as c10d

    dist.new_group(list(range(world_size)))
    assert len(c10d._world.pg_map) == 2


def _assert_only_default_group(rank, world_size):
    from torch.distributed import distributed_c10d as c10d

    assert len(c10d._world.pg_map) == 1


# ---- tests ----


def test_reuses_ranks_across_runs(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    run_distributed(2, _allreduce_ok, 3, dist_backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert _read_pids(p, 2) == first


def test_rank_failure_kills_pool_and_reports_peer(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    before = _read_pids(p, 2)

    t0 = time.monotonic()
    with pytest.raises(WorkerPoolError) as exc_info:
        run_distributed(2, _rank1_raises_rank0_waits, dist_backend="gloo", timeout=120)
    elapsed = time.monotonic() - t0
    msg = str(exc_info.value)
    assert "boom on rank 1" in msg
    assert "rank 0: no report (killed after peer failure)" in msg
    # Rank 0 was stuck in a collective; we must not have waited for the deadline,
    # only the grace period for its report.
    assert elapsed < 2 * pool_mod._GRACE_SECONDS

    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(before)


def test_dead_rank_is_detected():
    run_distributed(
        2, _allreduce_ok, 1, dist_backend="gloo", timeout=_TIMEOUT
    )  # spawn outside the timed window
    t0 = time.monotonic()
    with pytest.raises(WorkerPoolError) as exc_info:
        run_distributed(2, _rank1_dies, dist_backend="gloo", timeout=_TIMEOUT)
    assert time.monotonic() - t0 < _TIMEOUT
    assert "rank 1: exited with code 7 without reporting" in str(exc_info.value)


def test_hung_test_hits_deadline():
    t0 = time.monotonic()
    with pytest.raises(WorkerPoolError, match="deadline reached"):
        run_distributed(2, _hang_forever, dist_backend="gloo", timeout=15)
    assert time.monotonic() - t0 < 15 + pool_mod._GRACE_SECONDS


@pytest.mark.timeout(60)
def test_default_deadline_follows_pytest_timeout():
    """With a 60 s pytest-timeout the pool must give up first (at 60 s minus
    the kill path's margin), so the test fails on its own instead of
    pytest-timeout killing the whole process."""
    t0 = time.monotonic()
    with pytest.raises(WorkerPoolError, match="deadline reached"):
        run_distributed(2, _hang_forever, dist_backend="gloo")
    assert time.monotonic() - t0 < 60 - pool_mod._GRACE_SECONDS


def test_lifetime_test_gives_fresh_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "test")
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    assert not pool_mod._POOLS
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(first)


def test_lifetime_module_reuses_within_a_module_and_drains_at_its_end(tmp_path, monkeypatch):
    from . import conftest

    monkeypatch.setenv("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "module")
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert _read_pids(p, 2) == first
    assert pool_mod._POOLS
    conftest.end_of_module()  # what the module-scoped fixture runs at teardown
    assert not pool_mod._POOLS
    for pid in first:
        assert _pid_gone(pid)
    monkeypatch.setenv("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "session")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert pool_mod._POOLS
    conftest.end_of_module()
    assert pool_mod._POOLS  # session lifetime: untouched at module end


def test_lifetime_value_is_validated(monkeypatch):
    monkeypatch.setenv("TRTLLM_TEST_VISUALGEN_POOL_LIFETIME", "forever")
    with pytest.raises(ValueError, match="TRTLLM_TEST_VISUALGEN_POOL_LIFETIME"):
        pool_mod.lifetime()


def test_rank_dead_while_idle_retires_pool(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    rank1 = next(iter(pool_mod._POOLS.values()))._ranks[1].proc
    os.kill(first[1], signal.SIGKILL)
    # Wait on what the pool checks: is_alive() reaps, so it turns False only
    # once every thread of the rank has exited (a /proc "Z" can come earlier).
    deadline = time.monotonic() + _TIMEOUT
    while rank1.is_alive():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(first)


def test_env_change_retires_pool(tmp_path, monkeypatch):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    first = _read_pids(p, 2)
    monkeypatch.setenv("TRTLLM_TEST_POOL_PROBE", "changed")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    assert set(_read_pids(p, 2)).isdisjoint(first)


def test_env_is_restored_between_runs():
    run_distributed(2, _set_env, dist_backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _assert_env_clean, dist_backend="gloo", timeout=_TIMEOUT)


def test_rng_is_seeded_before_every_run(tmp_path):
    """Every run, the first included, starts from the fixed seed on every rank."""
    p = str(tmp_path / "rand")
    draws = []
    for _ in range(3):
        run_distributed(2, _write_random, p, dist_backend="gloo", timeout=_TIMEOUT)
        draws.append(_read_values(p, 2))
    assert draws[0] == draws[1] == draws[2]
    assert len(set(draws[0])) == 1


def test_extra_process_groups_are_destroyed():
    run_distributed(2, _make_extra_group, dist_backend="gloo", timeout=_TIMEOUT)
    run_distributed(2, _assert_only_default_group, dist_backend="gloo", timeout=_TIMEOUT)


def test_reuse_false_gives_fresh_ranks_and_drains(tmp_path):
    p = str(tmp_path / "pid")
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT)
    cached = _read_pids(p, 2)
    run_distributed(2, _write_pid, p, dist_backend="gloo", timeout=_TIMEOUT, reuse=False)
    private = _read_pids(p, 2)
    assert set(private).isdisjoint(cached)
    assert not pool_mod._POOLS
    for pid in cached + private:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_ranks_die_with_parent(tmp_path):
    """SIGKILL the pytest-side owner while its ranks are busy in a test body."""
    p = str(tmp_path / "pid")
    script = f"""
import signal, threading, time, os
from multi_gpu._worker_pool import run_distributed
from multi_gpu.test_worker_pool import _write_pid_then_hang
t = threading.Thread(target=run_distributed, args=(2, _write_pid_then_hang, {p!r}),
                     kwargs=dict(dist_backend="gloo", timeout=3600), daemon=True)
t.start()
while not all(os.path.exists({p!r} + f".{{r}}") for r in range(2)):
    time.sleep(0.2)
print("READY", flush=True)
signal.pause()
"""
    # Only what the owner needs: the package root (multi_gpu) and tests/ (test_common).
    # Handing it the whole sys.path can make defs.common importable, which drags the
    # full library import into the owner before it prints READY.
    here = Path(__file__).resolve()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(here.parents[1]), str(here.parents[4])]))
    owner = subprocess.Popen(
        [sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, bufsize=0
    )
    try:
        deadline = time.monotonic() + 120
        seen = b""
        while b"READY\n" not in seen:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and owner.poll() is None, "owner did not start"
            if select.select([owner.stdout], [], [], remaining)[0]:
                chunk = os.read(owner.stdout.fileno(), 4096)
                assert chunk, "owner closed stdout before READY"
                seen += chunk
        ranks = _read_pids(p, 2)
        assert not any(_pid_gone(pid) for pid in ranks)
        owner.send_signal(signal.SIGKILL)
        owner.wait(timeout=10)
        deadline = time.monotonic() + 15
        while not all(_pid_gone(pid) for pid in ranks):
            assert time.monotonic() < deadline, (
                f"ranks {ranks} outlived their parent; states {[_proc_state(pid) for pid in ranks]}"
            )
            time.sleep(0.2)
    finally:
        if owner.poll() is None:
            owner.kill()
        for rank in range(2):
            try:
                with open(f"{p}.{rank}") as f:
                    pid = int(f.read())
            except FileNotFoundError:
                continue  # startup failed before this rank wrote its pid
            if not _pid_gone(pid):
                os.kill(pid, signal.SIGKILL)


def test_unpicklable_fn_touches_no_ranks():
    before = len(multiprocessing.active_children())
    with pytest.raises((pickle.PicklingError, AttributeError)):
        run_distributed(2, lambda rank, world_size: None, dist_backend="gloo", timeout=_TIMEOUT)
    assert not pool_mod._POOLS
    assert len(multiprocessing.active_children()) == before


def test_call_deadline_is_capped_by_test_deadline(monkeypatch):
    now = time.monotonic()
    monkeypatch.setattr(pool_mod, "_test_deadline_at", now + 30)
    assert abs(pool_mod._deadline_for(None) - (now + 30)) < 1
    assert abs(pool_mod._deadline_for(5) - (now + 5)) < 1
    assert abs(pool_mod._deadline_for(100) - (now + 30)) < 1
    monkeypatch.setattr(pool_mod, "_test_deadline_at", None)
    assert pool_mod._deadline_for(None) is None
    assert abs(pool_mod._deadline_for(5) - (now + 5)) < 1
