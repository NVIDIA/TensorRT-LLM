# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BuildPool: inside a pass, worker processes make the builds this process asks for, in
parallel, and the call returns once they have ended; this process then loads the kernels.

The kernels are build_pool_targets.py's picklable stand-ins, so no GPU is needed.
"""

import errno
import importlib.util
import os
import signal
import subprocess  # nosec B404 - starts a stand-in rank: sys.executable, a fixed argv, no shell
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import build_pool_targets as targets
import pytest

from tensorrt_llm import _jit_share
from tensorrt_llm._torch import build_pool
from tensorrt_llm._torch.build_pool import Build, BuildPool, BuildResult, Claim

pytestmark = pytest.mark.cpu_only

_RECORD = f"{targets.__name__}:record"
_HOLD = f"{targets.__name__}:hold"


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


def test_an_error_serving_a_build_fails_it_and_ends_its_worker(
    pool: BuildPool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An error in the thread that serves a worker, here reading the GPU's free memory after a
    sticky CUDA error, fails the build it serves and ends the worker, whose queued builds are
    refused: the pass does not wait for builds no thread serves."""

    def sticky_error(device: object) -> int:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(build_pool, "_free_gpu_bytes", sticky_error)
    pool.max_workers = 1
    with pool.pass_scope():
        results = pool.build([_record(tmp_path / "a", 1), _record(tmp_path / "b", 2)])
    assert [result.status for result in results] == ["failed", "refused"]
    assert "serving the build raised RuntimeError" in results[0].error
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_no_worker_starts_while_the_gpu_lacks_room(pool, monkeypatch):
    monkeypatch.setattr(build_pool, "_gpu_has_room", lambda reserved=0: False)
    with pool.pass_scope():
        _build_kernels((5,))
    assert targets.compile_kernel(5).pid == os.getpid()
    assert pool._workers_started == 0
    assert pool.counts() == {"refused": 1}


_SPAN = f"{targets.__name__}:span"


def _spans(paths):
    """The (start, end) times of the builds that wrote ``paths``, in start order."""
    return sorted(tuple(map(float, path.read_text().split())) for path in paths)


def _overlap(spans):
    return any(later[0] < earlier[1] for earlier, later in zip(spans, spans[1:]))


@pytest.fixture
def gpu(monkeypatch):
    """The rank's GPU as the pool sees it: ``gpu["free"]`` bytes free."""
    gpu = {"free": 0}
    monkeypatch.setattr(build_pool, "_free_gpu_bytes", lambda device: gpu["free"])
    return gpu


def test_builds_that_do_not_fit_together_run_one_at_a_time(pool, gpu, tmp_path):
    # Room for one 7 GiB build beside the 8 GiB the rank keeps free, not for two.
    gpu["free"] = 20 << 30
    pool.record_peak("large", 7 << 30)
    paths = [tmp_path / f"s{i}" for i in range(2)]
    with pool.pass_scope():
        results = pool.build([Build(_SPAN, (str(path), 0.5), "large") for path in paths])
    assert [result.status for result in results] == ["built", "built"]
    assert not _overlap(_spans(paths))


def test_a_build_that_does_not_fit_alone_is_left_to_the_caller(pool, gpu, tmp_path):
    gpu["free"] = 10 << 30
    pool.record_peak("huge", 4 << 30)
    with pool.pass_scope():
        [result] = pool.build([Build(_SPAN, (str(tmp_path / "s"), 0.0), "huge")])
    assert result.status == "refused" and not (tmp_path / "s").exists()


def test_a_kind_of_build_not_measured_yet_builds_one_at_a_time(pool, gpu, tmp_path):
    gpu["free"] = 64 << 30
    first = [tmp_path / f"s{i}" for i in range(2)]
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    with pool.pass_scope():
        pool.build([Build(_SPAN, (str(path), 0.5), "new") for path in first])
        assert not _overlap(_spans(first))
        # Measured (no GPU memory here): its builds now run together, or neither ends.
        results = pool.build([Build(_HOLD, (a, b, 60.0), "new"), Build(_HOLD, (b, a, 60.0), "new")])
    assert [result.status for result in results] == ["built", "built"]
    assert pool.peak_bytes("new") == 0


def test_a_workers_first_build_also_needs_room_for_its_cuda_context(
    pool: BuildPool, gpu: dict, tmp_path: Path
) -> None:
    """A worker makes its CUDA context in its first build: that build needs room for the context
    beside the 8 GiB the rank keeps free, and the worker's later builds do not."""
    pool.max_workers = 1
    gpu["free"] = (8 << 30) + (1 << 29)
    with pool.pass_scope():
        [cold] = pool.build([_record(tmp_path / "a", 1)])
    assert cold.status == "refused"
    gpu["free"] = (9 << 30) + (1 << 29)
    with pool.pass_scope():
        [first] = pool.build([_record(tmp_path / "b", 2)])
        gpu["free"] = (8 << 30) + (1 << 29)
        [warm] = pool.build([_record(tmp_path / "c", 3)])
    assert (first.status, warm.status) == ("built", "built")


def test_new_workers_whose_contexts_do_not_fit_together_build_one_at_a_time(
    pool: BuildPool, gpu: dict, tmp_path: Path
) -> None:
    # Room for one worker's context beside the 8 GiB the rank keeps free, not for two.
    gpu["free"] = (9 << 30) + (1 << 29)
    paths = [tmp_path / f"s{i}" for i in range(2)]
    with pool.pass_scope():
        results = pool.build([Build(_SPAN, (str(path), 0.5)) for path in paths])
    assert [result.status for result in results] == ["built", "built"]
    assert not _overlap(_spans(paths))
    # Room for two: each build waits for the other to start, so both end only if run together.
    gpu["free"] = (10 << 30) + (1 << 29)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    with pool.pass_scope():
        results = pool.build([Build(_HOLD, (a, b, 60.0)), Build(_HOLD, (b, a, 60.0))])
    assert [result.status for result in results] == ["built", "built"]


def test_a_later_batch_starts_the_workers_the_gpu_had_no_room_for(
    pool: BuildPool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """While the GPU has room for one worker, that worker builds a batch alone; once the GPU has
    room, a later batch starts the workers its builds need, however many builds the first worker
    made."""
    room = {"workers": 1}
    monkeypatch.setattr(
        build_pool, "_gpu_has_room", lambda reserved=0: pool._workers_started < room["workers"]
    )
    with pool.pass_scope():
        first = pool.build([_record(tmp_path / f"r{v}", v) for v in range(3)])
        assert [result.status for result in first] == ["built"] * 3
        assert pool._workers_started == 1
        room["workers"] = 2
        # Each build waits for the other to start: both end only if two workers run them.
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        second = pool.build([Build(_HOLD, (a, b, 60.0)), Build(_HOLD, (b, a, 60.0))])
    assert [result.status for result in second] == ["built", "built"]
    assert pool._workers_started == 2


def test_a_worker_that_cannot_start_leaves_no_descriptor_open(
    pool: BuildPool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker whose start fails part way, here for want of a file for its stderr, closes the
    pipes it opened, and the pass starts no other: its builds are left to the caller."""
    starts = []

    def no_file() -> None:
        starts.append(1)
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(build_pool.tempfile, "TemporaryFile", no_file)
    before = len(os.listdir("/proc/self/fd"))
    with pool.pass_scope():
        results = pool.build([_record(tmp_path / "a", 1)]) + pool.build(
            [_record(tmp_path / "b", 2)]
        )
    assert [result.status for result in results] == ["refused", "refused"]
    assert len(os.listdir("/proc/self/fd")) == before
    assert len(starts) == 1 and pool._workers_started == 0


@pytest.mark.parametrize(
    "value,seconds",
    [
        (None, 900.0),
        ("0", None),
        ("2.5", 2.5),
        ("1e30", threading.TIMEOUT_MAX),
        # Neither 0 nor a positive, finite number of seconds: the default limit stands.
        ("-1", 900.0),
        ("nan", 900.0),
        ("inf", 900.0),
        ("soon", 900.0),
    ],
)
def test_only_zero_lets_a_worker_take_any_time_to_reply(
    monkeypatch: pytest.MonkeyPatch, value: str | None, seconds: float | None
) -> None:
    if value is None:
        monkeypatch.delenv("TLLM_KERNEL_BUILDER_BUILD_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("TLLM_KERNEL_BUILDER_BUILD_TIMEOUT_S", value)
    assert build_pool._reply_timeout() == seconds


def test_a_worker_that_stops_replying_is_stopped_and_its_build_fails(
    pool: BuildPool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker that does not reply within TLLM_KERNEL_BUILDER_BUILD_TIMEOUT_S seconds, stuck in
    a build here, is stopped and its build fails, so the pass goes on; another worker takes the
    next build."""
    monkeypatch.delenv("TLLM_KERNEL_BUILDER_BUILD_TIMEOUT_S", raising=False)
    with pool.pass_scope():
        # A first build with the default limit, which leaves room for the worker's imports.
        [first] = pool.build([_record(tmp_path / "a", 1)])
        pool._pool._reply_timeout = 2.0
        start = time.monotonic()
        [stuck] = pool.build([Build(f"{targets.__name__}:stall", (600.0,))])
        waited = time.monotonic() - start
        # No limit for the next worker, whose first reply waits for its imports.
        pool._pool._reply_timeout = None
        [after] = pool.build([_record(tmp_path / "b", 2)])
    assert first.status == "built"
    assert stuck.status == "failed" and "did not reply within 2s" in stuck.error
    assert waited < 60
    assert after.status == "built" and pool._workers_started == 2


def test_a_stopped_worker_marks_its_claim_failed_and_leaves_others_alone(
    pool: BuildPool, tmp_path: Path
) -> None:
    """A worker stopped in a build it claimed leaves that claim failed, so that the ranks waiting
    for the kernel build it at once rather than at the end of their wait; a claim another
    process made stands."""
    mine, theirs = (
        Claim(*(str(tmp_path / f"{who}.{end}") for end in ("claim", "done", "failed")))
        for who in ("mine", "theirs")
    )
    with pool.pass_scope():
        [first] = pool.build([_record(tmp_path / "a", 1)])
        pool._pool._reply_timeout = 2.0
        [stuck] = pool.build([Build(f"{targets.__name__}:stall", (600.0,), None, mine)])
    assert first.status == "built" and stuck.status == "failed"
    assert os.path.exists(mine.claim) and os.path.exists(mine.failed)
    with open(theirs.claim, "w") as f:
        f.write(f"{os.uname().nodename} 1\n")
    build_pool._abandon_claim(theirs, SimpleNamespace(pid=2), "stopped")
    assert not os.path.exists(theirs.failed)


def _claim_paths(tmp_path: Path, who: str) -> Claim:
    return Claim(*(str(tmp_path / f"{who}.{end}") for end in ("claim", "done", "failed")))


def _running(pid: int) -> bool:
    """Whether process ``pid`` runs: neither gone nor a zombie no one has reaped yet."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_a_stopped_worker_goes_with_the_processes_its_build_started(
    pool: BuildPool, tmp_path: Path
) -> None:
    """A worker stopped in a build ends with the processes the build started, a compiler's as
    here, which share its process group: none of them is left to write into a cache that a
    retry of the build, on this rank or another, uses."""
    child = tmp_path / "child"
    with pool.pass_scope():
        [first] = pool.build([_record(tmp_path / "a", 1)])
        pool._pool._reply_timeout = 2.0
        [stuck] = pool.build([Build(f"{targets.__name__}:start_and_stall", (str(child), 600.0))])
    assert first.status == "built" and stuck.status == "failed"
    pid = int(child.read_text())
    deadline = time.monotonic() + 30.0
    while _running(pid):
        assert time.monotonic() < deadline, f"the build's process {pid} outlived its worker"
        time.sleep(0.05)


# A rank whose one worker runs start_and_stall(<child>, 600): the build starts a process, writes
# its pid to <child>, and stalls.
_STALLING_RANK = """
import sys
from tensorrt_llm._torch.build_pool import Build, BuildPool
store, child = sys.argv[1:]
pool = BuildPool(workers=1, enabled=True, store=store)
with pool.pass_scope():
    pool.build([Build("build_pool_targets:start_and_stall", (child, 600.0))])
"""


def _parent(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[1])


def test_a_worker_whose_rank_is_gone_ends_with_the_processes_its_build_started(
    tmp_path: Path,
) -> None:
    """A worker leads a process group of its own, which the end of the rank's group does not
    reach, and it reads its task pipe only between builds: once the rank is gone, killed in a
    build, the worker ends itself and the processes the build started."""
    child = tmp_path / "child"
    here = os.path.dirname(os.path.abspath(targets.__file__))
    env = dict(os.environ)
    env[targets.IN_WORKER_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
    rank = subprocess.Popen(  # nosec B603 - sys.executable, a fixed argv
        [sys.executable, "-c", _STALLING_RANK, str(tmp_path / "store"), str(child)], env=env
    )
    worker = None
    try:
        deadline = time.monotonic() + 300.0
        while not (child.exists() and child.read_text()):
            assert rank.poll() is None, "the rank ended before its build started"
            assert time.monotonic() < deadline, "the rank's build did not start"
            time.sleep(0.1)
        started = int(child.read_text())
        worker = _parent(started)
        rank.kill()
        rank.wait()
        deadline = time.monotonic() + 30.0
        while _running(worker) or _running(started):
            assert time.monotonic() < deadline, (
                "the worker or its build's process outlived the rank"
            )
            time.sleep(0.1)
    finally:
        if rank.poll() is None:
            rank.kill()
            rank.wait()
        if worker is not None and _running(worker):
            os.killpg(worker, signal.SIGKILL)


def test_a_worker_whose_serving_thread_does_not_start_is_not_counted(
    pool: BuildPool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker whose serving thread does not start ends at once and is not counted: the pass
    starts no other, its builds are left to the caller, and its pipes are closed."""
    start = threading.Thread.start

    def no_serving_thread(thread: threading.Thread) -> None:
        if thread.name.startswith("tllm-build-pool-"):
            raise RuntimeError("can't start new thread")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", no_serving_thread)
    before = len(os.listdir("/proc/self/fd"))
    with pool.pass_scope():
        results = pool.build([_record(tmp_path / "a", 1)]) + pool.build(
            [_record(tmp_path / "b", 2)]
        )
    assert [result.status for result in results] == ["refused", "refused"]
    assert pool._workers_started == 0
    assert len(os.listdir("/proc/self/fd")) == before


def test_a_build_that_finds_its_kernel_made_ends_present(pool: BuildPool, tmp_path: Path) -> None:
    """A build function that returns False found what it builds made already (a compiler's
    cache hit): the build ends present, not built, so that the counts tell the two apart; one
    that returns nothing built. Either way its claim is done."""
    claim = _claim_paths(tmp_path, "found")
    with pool.pass_scope():
        results = pool.build(
            [
                Build(f"{targets.__name__}:found", (str(tmp_path / "found"),), None, claim),
                _record(tmp_path / "a", 1),
            ]
        )
    assert [result.status for result in results] == ["present", "built"]
    assert (tmp_path / "found").exists() and os.path.exists(claim.done)
    assert pool.counts() == {"present": 1, "built": 1}


def test_the_kernels_of_an_edited_compile_function_file_get_other_names(tmp_path: Path) -> None:
    """A process_built function's kernels are also named by its file's contents: a kernel
    compiled from another version of the file, which an editable install does not tell by its
    version, is not loaded for this one."""

    def store_name(module: str, source: str) -> str:
        path = tmp_path / f"{module}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(module, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded.compile_kernel._compiler.store_name

    source = (
        "from tensorrt_llm._torch.build_pool import process_built\n\n\n"
        "@process_built('edited_file_test')\n"
        "def compile_kernel(value):\n"
        "    return value\n"
    )
    assert store_name("first", source) == store_name("first_again", source)
    assert store_name("edited", source + "# edited\n") != store_name("first", source)


def test_a_kernel_is_the_same_however_a_call_passes_its_arguments(pool: BuildPool) -> None:
    with pool.pass_scope():
        # Two spellings of one kernel: one build.
        results = pool.build_kernels(
            [(targets.compile_kernel, (40,)), (targets.compile_kernel, (40, ""))]
        )
        assert [result.status for result in results] == ["built", "present"]
        kernel = targets.compile_kernel(40)
        assert targets.compile_kernel(40, "") is kernel
        assert targets.compile_kernel(value=40) is kernel
        assert targets.compile_kernel(40, in_worker="") is kernel
    assert kernel.pid != os.getpid()
    assert targets.compile_kernel.cache_info().currsize == 1
    assert pool.counts() == {"built": 1}


def test_process_built_takes_only_functions_of_positional_parameters() -> None:
    def keyword_only(value: int, *, mode: str = "") -> int:
        return value

    with pytest.raises(TypeError):
        build_pool.process_built("keyword_only")(keyword_only)


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
    compiler = targets.compile_kernel._compiler
    # The store names a kernel by its function's file and its call's arguments, defaults filled in.
    _, paths = rank._paths(compiler.store_name, compiler.bind(args, {}))
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
    # A directory this process does not share builds through is no store.
    assert _jit_share._cute_store() is None


def test_a_shared_store_that_cannot_be_made_leaves_the_rank_a_store_of_its_own(
    worker_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The workers still build where the directory the ranks share cannot hold the store: into
    a store of this rank's own, from which it loads the kernels, which are not shared."""
    not_a_directory = tmp_path / "share"
    not_a_directory.write_text("")
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(not_a_directory))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)
    rank = BuildPool(workers=1, enabled=True)
    monkeypatch.setattr(BuildPool, "_instance", rank)
    try:
        with rank.pass_scope():
            _build_kernels((34,))
            assert rank.accepting()
            assert targets.compile_kernel(34).pid != os.getpid()
    finally:
        rank.stop()
    assert rank.counts() == {"built": 1}
    assert not rank._shared and not rank._store.startswith(str(not_a_directory))


@pytest.mark.parametrize("first", ["build", "build_kernels"])
def test_a_store_that_cannot_be_made_leaves_every_build_here(
    worker_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: str
) -> None:
    not_a_directory = tmp_path / "share"
    not_a_directory.write_text("")
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(not_a_directory))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)

    def no_directory(*args: object, **kwargs: object) -> str:
        raise OSError(28, "No space left on device")

    # Nor can a store of the rank's own be made.
    monkeypatch.setattr(build_pool.tempfile, "mkdtemp", no_directory)
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
