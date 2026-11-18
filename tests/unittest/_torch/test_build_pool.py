# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BuildPool: inside a pass, worker processes make the builds this process asks for, in
parallel, and the call returns once they have ended; this process then loads the kernels.

The kernels are build_pool_targets.py's picklable stand-ins, so no GPU is needed.
"""

import os
import threading
import time

import build_pool_targets as targets
import pytest

from tensorrt_llm import _jit_share
from tensorrt_llm._torch import build_pool
from tensorrt_llm._torch.build_pool import Build, BuildPool, BuildResult

pytestmark = pytest.mark.cpu_only

_RECORD = f"{targets.__name__}:record"
_HOLD = f"{targets.__name__}:hold"


@pytest.fixture(autouse=True)
def _every_build_to_a_worker(monkeypatch):
    """Workers take every build, however short; the tests of that rule set it themselves."""
    monkeypatch.setenv("TLLM_KERNEL_BUILDER_MIN_BUILD_S", "0")


@pytest.fixture
def worker_env(monkeypatch):
    """Worker processes that import build_pool_targets, started from a launcher's environment."""
    original = build_pool._worker_env
    here = os.path.dirname(os.path.abspath(targets.__file__))

    def worker_env():
        env = original()
        env[targets.IN_WORKER_ENV] = "1"
        env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
        return env

    monkeypatch.setattr(build_pool, "_worker_env", worker_env)
    monkeypatch.setattr(_jit_share, "_load_kernel", targets.load_kernel)
    monkeypatch.setattr(_jit_share, "_cute_identity", lambda: targets.IDENTITY)
    for name in targets.LAUNCHER_VARS:
        monkeypatch.setenv(name, "1")
    targets.compile_kernel.cache_clear()
    yield
    targets.compile_kernel.cache_clear()


@pytest.fixture
def pool(worker_env, monkeypatch, tmp_path):
    """This process's BuildPool: up to two worker processes and a store of its own."""
    rank = BuildPool(workers=2, enabled=True, store=str(tmp_path / "store"))
    monkeypatch.setattr(BuildPool, "_instance", rank)
    yield rank
    rank.stop()


def _record(path, value):
    return Build(_RECORD, (str(path), value))


def _build_kernels(*arg_lists):
    """Build the stand-in kernels of ``arg_lists`` on the workers, together, then call them:
    each call loads its kernel, or compiles one no worker built."""
    arg_lists = list(dict.fromkeys(arg_lists))
    BuildPool.get().build_kernels([(targets.compile_kernel, args) for args in arg_lists])
    for args in arg_lists:
        targets.compile_kernel(*args)
    return len(arg_lists)


def test_kernels_built_together_are_built_by_workers_and_loaded_here(pool):
    with pool.pass_scope():
        assert _build_kernels((1,), (2,), (1,)) == 2
        # Built, and loaded by the calls.
        assert targets.compile_kernel.cache_info().currsize == 2
        kernels = [targets.compile_kernel(1), targets.compile_kernel(2)]
    assert [kernel.value for kernel in kernels] == [1, 2]
    assert all(kernel.pid != os.getpid() for kernel in kernels)
    # The launcher's variables stay out of the workers: importing TensorRT-LLM starts MPI.
    assert all(kernel.launcher_vars == () for kernel in kernels)
    assert targets.compile_kernel.cache_info()[:2] == (2, 2)
    assert pool.counts() == {"built": 2}


def test_kernels_get_a_result_each(pool):
    assert pool.build_kernels([(targets.compile_kernel, (60,))]) == [BuildResult("refused")]
    with pool.pass_scope():
        [built] = pool.build_kernels([(targets.compile_kernel, (60,))])
        # In the store already.
        [present] = pool.build_kernels([(targets.compile_kernel, (60,))])
    assert (built.status, present.status) == ("built", "present")


def test_a_build_returns_every_result_in_order_once_all_have_ended(pool, tmp_path):
    ended = []
    builds = [_record(tmp_path / f"r{v}", v) for v in range(4)]
    builds.append(Build(f"{targets.__name__}:fail"))
    with pool.pass_scope():
        results = pool.build(builds, on_done=lambda i, result: ended.append((i, result.status)))
    assert [result.status for result in results] == ["built"] * 4 + ["failed"]
    assert "ValueError: a build that fails" in results[-1].error
    assert sorted(ended) == [(i, "built") for i in range(4)] + [(4, "failed")]
    for v in range(4):
        value, pid = (tmp_path / f"r{v}").read_text().split()
        assert int(value) == v and int(pid) != os.getpid()


def test_the_builds_of_one_call_run_in_parallel(pool, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    # Each build waits for the other to start: both end only if two workers run them together.
    with pool.pass_scope():
        results = pool.build([Build(_HOLD, (a, b, 60.0)), Build(_HOLD, (b, a, 60.0))])
    assert [result.status for result in results] == ["built", "built"]


def test_outside_a_pass_no_worker_takes_a_build(pool, tmp_path):
    assert not pool.takes((2,))
    assert pool.build([_record(tmp_path / "r", 2)]) == [BuildResult("refused")]
    assert _build_kernels((2,)) == 1
    assert targets.compile_kernel(2).pid == os.getpid()
    assert not (tmp_path / "r").exists()
    assert pool.counts() == {"refused": 1} and pool._workers_started == 0


def test_with_workers_off_no_worker_takes_a_build(worker_env, monkeypatch, tmp_path):
    rank = BuildPool(workers=2, enabled=False, store=str(tmp_path / "store"))
    monkeypatch.setattr(BuildPool, "_instance", rank)
    with rank.pass_scope():
        _build_kernels((3,))
        assert targets.compile_kernel(3).pid == os.getpid()
    assert rank._workers_started == 0


def test_a_kernel_whose_build_fails_in_a_worker_is_compiled_here(pool):
    with pool.pass_scope():
        _build_kernels((4, "raise"))
        assert targets.compile_kernel(4, "raise").pid == os.getpid()
    assert pool.counts() == {"failed": 1}


def test_the_other_workers_carry_on_when_one_ends(pool):
    with pool.pass_scope():
        _build_kernels((5, "exit"))
        _build_kernels((6,))
        assert targets.compile_kernel(6).pid != os.getpid()
        assert targets.compile_kernel(5, "exit").pid == os.getpid()
    assert pool.counts() == {"failed": 1, "built": 1}


def test_the_builds_queued_when_no_worker_is_left_are_refused(pool):
    pool.max_workers = 1
    with pool.pass_scope():
        _build_kernels((7, "exit"), (8,))
    assert targets.compile_kernel(8).pid == os.getpid()
    assert pool.counts() == {"failed": 1, "refused": 1}


def test_the_workers_stop_when_the_outermost_pass_ends(pool):
    with pool.pass_scope():
        with pool.pass_scope():
            _build_kernels((9,))
            first = targets.compile_kernel(9).pid
        # An inner pass leaves the workers running.
        os.kill(first, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(first, 0)
    with pool.pass_scope():
        _build_kernels((10,))
        assert targets.compile_kernel(10).pid not in (first, os.getpid())


def test_the_workers_start_with_the_builds_up_to_the_cap(pool):
    with pool.pass_scope():
        _build_kernels(*((value,) for value in range(11, 17)))
    assert pool.counts() == {"built": 6}
    assert 1 <= pool._workers_started <= pool.max_workers


def test_a_build_that_runs_an_op_leaves_its_kernel_for_this_process(pool):
    with pool.pass_scope():
        [result] = pool.build([Build(f"{targets.__name__}:build_by_running", (21,))])
        assert result.status == "built"
        # The kernel the op compiled went to the store, and loads here.
        assert targets.compile_kernel(21).pid != os.getpid()


def test_a_build_a_worker_cannot_load_fails_alone(pool, tmp_path):
    # One worker at most: the next build waits for it rather than starting another.
    pool.max_workers = 1
    with pool.pass_scope():
        results = pool.build(
            [Build(_RECORD, (targets.LoadsOnlyHere(), 0)), _record(tmp_path / "r", 7)]
        )
    assert [result.status for result in results] == ["failed", "built"]
    assert (tmp_path / "r").read_text().split()[0] == "7"
    assert pool._workers_started == 1


def test_a_worker_that_loses_its_cuda_context_ends_and_another_takes_the_builds(pool, tmp_path):
    with pool.pass_scope():
        [lost] = pool.build([Build(f"{targets.__name__}:lose_context")])
        assert lost.status == "failed"
        [record] = pool.build([_record(tmp_path / "r", 8)])
        assert record.status == "built"
    assert (tmp_path / "r").read_text().split()[0] == "8"
    assert pool._workers_started == 2


def test_no_worker_starts_while_the_gpu_lacks_room(pool, monkeypatch):
    monkeypatch.setattr(build_pool, "_gpu_has_room", lambda: False)
    with pool.pass_scope():
        _build_kernels((5,))
    assert targets.compile_kernel(5).pid == os.getpid()
    assert pool._workers_started == 0
    assert pool.counts() == {"refused": 1}


def test_a_worker_takes_only_builds_long_enough_to_repay_it(pool):
    pool.min_build_s = 5.0
    with pool.pass_scope():
        # Neither measured nor expected: the caller measures one first.
        assert not pool.worth_a_worker("new") and pool.unmeasured("new")
        assert pool.worth_a_worker("new", expected_s=30.0)
        assert not pool.worth_a_worker("new", expected_s=1.0)
        assert not pool.unmeasured("new", expected_s=1.0)
        # Once measured, the measure decides.
        pool.record_build("new", 0.5)
        assert not pool.worth_a_worker("new", expected_s=30.0) and not pool.unmeasured("new")
        pool.record_build("slow", 10.0)
        assert pool.worth_a_worker("slow")
    assert not pool.worth_a_worker("slow")
    pool.min_build_s = 0.0
    with pool.pass_scope():
        assert pool.worth_a_worker("new") and not pool.unmeasured("other")


def test_a_build_in_a_worker_measures_its_kind(pool, tmp_path):
    with pool.pass_scope():
        pool.build([Build(_RECORD, (str(tmp_path / "r"), 1), "recorded")])
    assert pool.build_seconds("recorded") is not None


def test_arguments_that_do_not_pickle_are_built_here(pool):
    with pool.pass_scope():
        assert pool.takes((1,)) and not pool.takes((lambda: 0,))
        assert pool.build([Build(_RECORD, (lambda: 0,))]) == [BuildResult("refused")]
    assert pool.counts() == {"refused": 1} and pool._workers_started == 0


@pytest.fixture
def share_dir(worker_env, monkeypatch, tmp_path):
    """A launch whose processes share their JIT builds through a directory."""
    share = tmp_path / "share"
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(share))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)
    return share


def test_ranks_sharing_a_store_build_each_kernel_once(share_dir, monkeypatch):
    ranks = [BuildPool(workers=1, enabled=True), BuildPool(workers=1, enabled=True)]
    kernels = []
    for rank in ranks:
        monkeypatch.setattr(BuildPool, "_instance", rank)
        targets.compile_kernel.cache_clear()
        with rank.pass_scope():
            _build_kernels((30,))
            kernels.append(targets.compile_kernel(30))
    # The second rank finds the kernel the first one built in the store.
    assert kernels[0] == kernels[1] and kernels[0].pid != os.getpid()
    assert [rank.counts() for rank in ranks] == [{"built": 1}, {}]
    assert os.path.dirname(ranks[0]._paths(targets.NAME, (30,))[1].obj) == str(share_dir / "cute")


def _claimed_elsewhere(rank, args):
    """Claim the kernel of ``args`` for another rank, which has not built it."""
    _, paths = rank._paths(targets.NAME, args)
    os.makedirs(os.path.dirname(paths.claim), exist_ok=True)
    open(paths.claim, "w").close()
    return paths


def test_a_rank_loads_a_kernel_another_rank_claimed_once_it_is_built(share_dir, monkeypatch):
    rank = BuildPool(workers=1, enabled=True)
    monkeypatch.setattr(BuildPool, "_instance", rank)
    paths = _claimed_elsewhere(rank, (31,))
    other = targets.Kernel(31, -1, ())
    build = rank.build
    timers = []

    def build_then_built_elsewhere(builds, on_done=None):
        results = build(builds, on_done)
        # The other rank's build ends once this rank's worker found it claimed.
        timers.append(
            threading.Timer(0.5, targets.export_kernel, (other, paths.obj, "", paths.meta))
        )
        timers[-1].start()
        return results

    monkeypatch.setattr(rank, "build", build_then_built_elsewhere)
    try:
        with rank.pass_scope():
            _build_kernels((31,))
            assert targets.compile_kernel(31) == other
        assert rank.counts() == {"claimed": 1}
    finally:
        for timer in timers:
            timer.join()
        rank.stop()


def test_a_kernel_claimed_by_a_rank_that_never_builds_it_is_compiled_here(share_dir, monkeypatch):
    monkeypatch.setenv("TLLM_JIT_SHARE_WAIT_S", "0.5")
    rank = BuildPool(workers=1, enabled=True)
    monkeypatch.setattr(BuildPool, "_instance", rank)
    _claimed_elsewhere(rank, (32,))
    with rank.pass_scope():
        start = time.monotonic()
        _build_kernels((32,))
        assert targets.compile_kernel(32).pid == os.getpid()
    assert time.monotonic() - start < 60
    assert rank.counts() == {"claimed": 1}


def test_without_shared_builds_each_rank_keeps_a_store_of_its_own(
    worker_env, monkeypatch, tmp_path
):
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, "/nonexistent-share-dir")
    monkeypatch.setattr(_jit_share, "enabled", lambda: False)
    ranks = [BuildPool(workers=1, enabled=True) for _ in range(2)]
    stores = [rank._store_dir() for rank in ranks]
    assert stores[0] != stores[1]
    assert not any(store.startswith("/nonexistent-share-dir") for store in stores)
    # Once a worker takes a build, the shared kernel caches keep CuTe DSL kernels where the
    # workers build them, until the pass ends.
    with ranks[0].pass_scope():
        [result] = ranks[0].build([_record(tmp_path / "r", 1)])
        assert result.status == "built"
        assert _jit_share._cute_store() == stores[0]
    assert _jit_share._cute_store() == os.path.join("/nonexistent-share-dir", "cute")


@pytest.mark.parametrize("first", ["build", "build_kernels"])
def test_a_store_that_cannot_be_made_leaves_every_build_here(
    worker_env, monkeypatch, tmp_path, first
):
    not_a_directory = tmp_path / "share"
    not_a_directory.write_text("")
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(not_a_directory))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)
    rank = BuildPool(workers=1, enabled=True)
    monkeypatch.setattr(BuildPool, "_instance", rank)
    try:
        with rank.pass_scope():
            if first == "build":
                assert rank.build([_record(tmp_path / "r", 1)]) == [BuildResult("refused")]
            else:
                _build_kernels((33,))
            # The first build found the store unusable: the workers are off.
            assert not rank.accepting()
            assert targets.compile_kernel(33).pid == os.getpid()
    finally:
        rank.stop()
    assert rank._workers_started == 0
