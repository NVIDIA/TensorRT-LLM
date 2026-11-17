# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RankWorker: inside a pass, worker processes build the kernels this process asks for, and it
loads them.

The kernels are rank_worker_targets.py's picklable stand-ins, so no GPU is needed.
"""

import os
import threading
import time

import pytest
import rank_worker_targets as targets

from tensorrt_llm import _jit_share
from tensorrt_llm._torch import rank_worker
from tensorrt_llm._torch.rank_worker import RankWorker

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _every_build_to_a_worker(monkeypatch):
    """Workers take every build, however short; the tests of that rule set it themselves."""
    monkeypatch.setenv("TLLM_KERNEL_BUILDER_MIN_BUILD_S", "0")


@pytest.fixture
def worker_env(monkeypatch):
    """Worker processes that import rank_worker_targets, started from a launcher's environment."""
    original = rank_worker._worker_env
    here = os.path.dirname(os.path.abspath(targets.__file__))

    def worker_env():
        env = original()
        env[targets.IN_WORKER_ENV] = "1"
        env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
        return env

    monkeypatch.setattr(rank_worker, "_worker_env", worker_env)
    monkeypatch.setattr(_jit_share, "_load_kernel", targets.load_kernel)
    monkeypatch.setattr(_jit_share, "_cute_identity", lambda: targets.IDENTITY)
    for name in targets.LAUNCHER_VARS:
        monkeypatch.setenv(name, "1")
    targets.compile_kernel.cache_clear()
    yield
    targets.compile_kernel.cache_clear()


@pytest.fixture
def worker(worker_env, monkeypatch, tmp_path):
    """This process's RankWorker: up to two worker processes and a store of its own."""
    rank = RankWorker(workers=2, enabled=True, store=str(tmp_path / "store"))
    monkeypatch.setattr(RankWorker, "_instance", rank)
    yield rank
    rank.stop()


def test_a_submitted_kernel_is_built_by_a_worker_and_loaded_here(worker):
    with worker.pass_scope():
        assert targets.compile_kernel.submit(1)
        assert not targets.compile_kernel.submit(1)
        worker.join()
        kernel = targets.compile_kernel(1)
    assert kernel.value == 1
    assert kernel.pid != os.getpid()
    # The launcher's variables stay out of the workers: importing TensorRT-LLM starts MPI.
    assert kernel.launcher_vars == ()
    assert targets.compile_kernel(1) is kernel
    assert targets.compile_kernel.cache_info()[:2] == (1, 1)
    assert worker.counts() == {"built": 1}


def test_outside_a_pass_no_worker_takes_a_build(worker, tmp_path):
    assert not worker.takes((2,))
    assert not targets.compile_kernel.submit(2)
    assert worker.run("record", f"{targets.__name__}:record", (str(tmp_path / "r"), 2)) is None
    assert targets.compile_kernel(2).pid == os.getpid()
    assert worker.counts() == {}


def test_with_workers_off_no_worker_takes_a_build(worker_env, monkeypatch, tmp_path):
    rank = RankWorker(workers=2, enabled=False, store=str(tmp_path / "store"))
    monkeypatch.setattr(RankWorker, "_instance", rank)
    with rank.pass_scope():
        assert not targets.compile_kernel.submit(3)
        assert targets.compile_kernel(3).pid == os.getpid()
    assert rank.counts() == {}


def test_a_build_that_fails_in_a_worker_is_compiled_here(worker):
    with worker.pass_scope():
        assert targets.compile_kernel.submit(4, "raise")
        worker.join()
        assert targets.compile_kernel(4, "raise").pid == os.getpid()
    assert worker.counts() == {"failed": 1}


def test_the_other_workers_carry_on_when_one_ends(worker):
    with worker.pass_scope():
        assert targets.compile_kernel.submit(5, "exit")
        worker.join()
        assert targets.compile_kernel.submit(6)
        worker.join()
        assert targets.compile_kernel(6).pid != os.getpid()
        assert targets.compile_kernel(5, "exit").pid == os.getpid()
    assert worker.counts() == {"failed": 1, "built": 1}


def test_the_queued_builds_fail_when_no_worker_is_left(worker):
    worker.max_workers = 1
    with worker.pass_scope():
        assert targets.compile_kernel.submit(7, "exit")
        assert targets.compile_kernel.submit(8)
        worker.join()
    assert targets.compile_kernel(8).pid == os.getpid()
    assert worker.counts() == {"failed": 2}


def test_the_workers_stop_when_the_outermost_pass_ends(worker):
    with worker.pass_scope():
        with worker.pass_scope():
            assert targets.compile_kernel.submit(9)
            worker.join()
            first = targets.compile_kernel(9).pid
        # An inner pass leaves the workers running.
        os.kill(first, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(first, 0)
    with worker.pass_scope():
        assert targets.compile_kernel.submit(10)
        worker.join()
        assert targets.compile_kernel(10).pid not in (first, os.getpid())


def test_a_build_no_worker_has_started_is_taken_back_and_made_here(worker, tmp_path):
    worker.max_workers = 1
    started, release = tmp_path / "started", tmp_path / "release"
    with worker.pass_scope():
        busy = worker.run("busy", f"{targets.__name__}:hold", (str(started), str(release)))
        try:
            deadline = time.monotonic() + 60
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert targets.compile_kernel.submit(40)
            # The only worker is busy: the kernel is made here at once.
            assert targets.compile_kernel(40).pid == os.getpid()
        finally:
            release.touch()
        assert busy.result()[0] == "done"
    assert worker.counts() == {"done": 1, "taken": 1}


def test_the_workers_start_as_builds_come_in_up_to_the_cap(worker):
    with worker.pass_scope():
        for value in range(11, 17):
            assert targets.compile_kernel.submit(value)
        worker.join()
    assert worker.counts() == {"built": 6}
    assert 1 <= worker._workers_started <= worker.max_workers


def test_a_run_builds_in_a_worker_and_this_process_loads_its_kernel(worker, tmp_path):
    path = tmp_path / "result"
    with worker.pass_scope():
        recorded = worker.run("record", f"{targets.__name__}:record", (str(path), 20))
        ran = worker.run("by-running", f"{targets.__name__}:build_by_running", (21,))
        assert worker.run("record", f"{targets.__name__}:record", (str(path), 0)) is recorded
        assert recorded.result()[0] == "done" and ran.result()[0] == "done"
        assert worker.state("record") == "done"
    value, pid = path.read_text().split()
    assert value == "20" and int(pid) != os.getpid()
    # The kernel the run compiled went to the store, and loads here.
    assert targets.compile_kernel(21).pid != os.getpid()
    assert worker.counts() == {"done": 2}


def test_a_run_that_fails_reports_the_error(worker):
    with worker.pass_scope():
        status, _, error = worker.run("fail", f"{targets.__name__}:fail", ()).result()
    assert status == "failed" and "ValueError: a build that fails" in error


def test_a_build_a_worker_cannot_load_fails_alone(worker, tmp_path):
    # One worker at most: the next build waits for it rather than starting another.
    worker.max_workers = 1
    with worker.pass_scope():
        unloadable = worker.run(
            "unloadable", f"{targets.__name__}:record", (targets.LoadsOnlyHere(), 0)
        )
        assert unloadable.result()[0] == "failed"
        record = worker.run("record", f"{targets.__name__}:record", (str(tmp_path / "r"), 7))
        assert record.result()[0] == "done"
    assert (tmp_path / "r").read_text().split()[0] == "7"
    assert worker._workers_started == 1


def test_a_worker_that_loses_its_cuda_context_ends_and_another_takes_the_builds(worker, tmp_path):
    with worker.pass_scope():
        lost = worker.run("lost", f"{targets.__name__}:lose_context", ())
        assert lost.result()[0] == "failed"
        record = worker.run("record", f"{targets.__name__}:record", (str(tmp_path / "r"), 8))
        assert record.result()[0] == "done"
    assert (tmp_path / "r").read_text().split()[0] == "8"
    assert worker._workers_started == 2


def test_no_worker_starts_while_the_gpu_lacks_room(worker, monkeypatch):
    monkeypatch.setattr(rank_worker, "_gpu_has_room", lambda: False)
    with worker.pass_scope():
        assert not targets.compile_kernel.submit(5)
    assert targets.compile_kernel(5).pid == os.getpid()
    assert worker._workers_started == 0


def test_a_worker_takes_only_builds_long_enough_to_repay_it(worker):
    worker.min_build_s = 5.0
    with worker.pass_scope():
        # Neither measured nor expected: built where it is first needed.
        assert not worker.worth_a_worker("new")
        assert worker.worth_a_worker("new", expected_s=30.0)
        assert not worker.worth_a_worker("new", expected_s=1.0)
        # Once measured, the measure decides.
        worker.record_build("new", 0.5)
        assert not worker.worth_a_worker("new", expected_s=30.0)
        worker.record_build("slow", 10.0)
        assert worker.worth_a_worker("slow")
    assert not worker.worth_a_worker("slow")
    worker.min_build_s = 0.0
    with worker.pass_scope():
        assert worker.worth_a_worker("new")


def test_a_process_built_function_measures_its_compiles(worker):
    worker.min_build_s = 5.0
    with worker.pass_scope():
        # Never measured: the call compiles it here, and measures it.
        assert not targets.compile_kernel.submit(50)
        assert targets.compile_kernel(50).pid == os.getpid()
        assert worker.build_seconds(targets.NAME) < 5.0
        assert not targets.compile_kernel.submit(51)
    assert worker.counts() == {}


def test_a_build_in_a_worker_measures_its_kind(worker):
    with worker.pass_scope():
        assert targets.compile_kernel.submit(52)
        worker.join()
    assert worker.build_seconds(targets.NAME) is not None


def test_arguments_that_do_not_pickle_are_built_here(worker):
    with worker.pass_scope():
        assert worker.takes((1,)) and not worker.takes((lambda: 0,))
        assert not worker.submit(targets.NAME, targets.WHERE, (lambda: 0,))
        assert worker.run("lambda", f"{targets.__name__}:record", (lambda: 0,)) is None
    assert worker.counts() == {}


@pytest.fixture
def share_dir(worker_env, monkeypatch, tmp_path):
    """A launch whose processes share their JIT builds through a directory."""
    share = tmp_path / "share"
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(share))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)
    return share


def test_ranks_sharing_a_store_build_each_kernel_once(share_dir):
    ranks = [RankWorker(workers=1, enabled=True), RankWorker(workers=1, enabled=True)]
    try:
        submitted = []
        for rank in ranks:
            with rank.pass_scope():
                submitted.append(rank.submit(targets.NAME, targets.WHERE, (30,)))
        # The second rank finds the kernel the first one built in the store.
        assert submitted == [True, False]
        kernels = [rank.take(targets.NAME, (30,)) for rank in ranks]
        assert kernels[0] == kernels[1]
        assert kernels[0].pid != os.getpid()
        assert [rank.counts() for rank in ranks] == [{"built": 1}, {"present": 1}]
    finally:
        for rank in ranks:
            rank.stop()
    assert os.path.dirname(ranks[0]._paths(targets.NAME, (30,))[1].obj) == str(share_dir / "cute")


def test_a_rank_loads_a_kernel_another_process_claimed_once_it_is_built(share_dir):
    rank = RankWorker(workers=1, enabled=True)
    _, paths = rank._paths(targets.NAME, (31,))
    os.makedirs(os.path.dirname(paths.claim), exist_ok=True)
    open(paths.claim, "w").close()
    try:
        with rank.pass_scope():
            assert rank.submit(targets.NAME, targets.WHERE, (31,))
        assert rank.counts() == {"claimed": 1}
        other = targets.Kernel(31, -1, ())
        build = threading.Timer(0.5, targets.export_kernel, (other, paths.obj, "", paths.meta))
        build.start()
        assert rank.take(targets.NAME, (31,)) == other
        build.join()
    finally:
        rank.stop()


def test_a_kernel_claimed_by_a_process_that_never_builds_it_is_compiled_here(
    share_dir, monkeypatch
):
    monkeypatch.setenv("TLLM_JIT_SHARE_WAIT_S", "0.5")
    rank = RankWorker(workers=1, enabled=True)
    _, paths = rank._paths(targets.NAME, (32,))
    os.makedirs(os.path.dirname(paths.claim), exist_ok=True)
    open(paths.claim, "w").close()
    try:
        with rank.pass_scope():
            assert rank.submit(targets.NAME, targets.WHERE, (32,))
        start = time.monotonic()
        assert rank.take(targets.NAME, (32,)) is None
        assert time.monotonic() - start < 30
    finally:
        rank.stop()


def test_without_shared_builds_each_rank_keeps_a_store_of_its_own(
    worker_env, monkeypatch, tmp_path
):
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, "/nonexistent-share-dir")
    monkeypatch.setattr(_jit_share, "enabled", lambda: False)
    ranks = [RankWorker(workers=1, enabled=True) for _ in range(2)]
    stores = [rank._store_dir() for rank in ranks]
    assert stores[0] != stores[1]
    assert not any(store.startswith("/nonexistent-share-dir") for store in stores)
    # Once a worker takes a build, the shared kernel caches keep CuTe DSL kernels where the
    # workers build them, until the pass ends.
    with ranks[0].pass_scope():
        record = ranks[0].run("record", f"{targets.__name__}:record", (str(tmp_path / "r"), 1))
        assert record is not None
        assert _jit_share._cute_store() == stores[0]
    assert record.result()[0] == "done"
    assert _jit_share._cute_store() == os.path.join("/nonexistent-share-dir", "cute")


@pytest.mark.parametrize("first", ["run", "submit"])
def test_a_store_that_cannot_be_made_leaves_every_build_here(
    worker_env, monkeypatch, tmp_path, first
):
    not_a_directory = tmp_path / "share"
    not_a_directory.write_text("")
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(not_a_directory))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)
    rank = RankWorker(workers=1, enabled=True)
    monkeypatch.setattr(RankWorker, "_instance", rank)
    try:
        # The pass ends: no build waits for a worker.
        with rank.pass_scope():
            if first == "run":
                record = (f"{targets.__name__}:record", (str(tmp_path / "r"), 1))
                assert rank.run("record", *record) is None
            else:
                assert not targets.compile_kernel.submit(33)
            # The first build found the store unusable: the workers are off.
            assert not rank.accepting()
            assert targets.compile_kernel(33).pid == os.getpid()
    finally:
        rank.stop()
    assert rank._futures == {}
    assert rank.counts() == {} and rank._workers_started == 0
