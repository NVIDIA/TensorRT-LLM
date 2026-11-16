# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KernelBuilder: in build mode, an op's calls list the kernels of the shapes its BuildConfig
sweeps, and the rank's worker processes build them; in inference mode nothing is listed.

The builds are rank_worker_targets.py's, which write a file, so no GPU is needed.
"""

import os
import time

import pytest
import rank_worker_targets as targets

from tensorrt_llm._torch import rank_worker
from tensorrt_llm._torch.kernel_builder import BuildConfig, BuildTask, DynamicSpec, KernelBuilder
from tensorrt_llm._torch.rank_worker import RankWorker

pytestmark = pytest.mark.cpu_only

_RECORD = f"{targets.__name__}:record"
_HOLD = f"{targets.__name__}:hold"


@pytest.fixture
def builder(monkeypatch, tmp_path):
    """This process's KernelBuilder, and a RankWorker of up to two workers that import
    rank_worker_targets."""
    original = rank_worker._worker_env
    here = os.path.dirname(os.path.abspath(targets.__file__))

    def worker_env():
        env = original()
        env[targets.IN_WORKER_ENV] = "1"
        env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
        return env

    monkeypatch.setattr(rank_worker, "_worker_env", worker_env)
    rank = RankWorker(workers=2, enabled=True, store=str(tmp_path / "store"))
    monkeypatch.setattr(RankWorker, "_instance", rank)
    kb = KernelBuilder(enabled=True)
    monkeypatch.setattr(KernelBuilder, "_instance", kb)
    yield kb
    rank.stop()


def _lister(tmp_path, calls):
    """A list_builds that records the shapes it is asked for and builds one file per shape."""

    def list_builds(shapes):
        calls.append([shape["m"] for shape in shapes])
        return [
            ((shape["m"],), BuildTask(_RECORD, (str(tmp_path / f"m{shape['m']}"), shape["m"])))
            for shape in shapes
        ]

    return list_builds


_CONFIG = BuildConfig((DynamicSpec("m", lambda m: range(1, m + 1)),))


def test_a_build_config_sweeps_the_buckets_with_the_call_value():
    config = BuildConfig((DynamicSpec("m", (1, 4)), DynamicSpec("splits", (1, 2))))
    shapes = config.shapes({"m": 3, "splits": 1})
    assert shapes == [{"m": m, "splits": s} for m in (1, 3, 4) for s in (1, 2)]


def test_in_inference_mode_an_op_lists_nothing(builder, tmp_path):
    calls = []
    builder.prebuild("op", "sig", _CONFIG, {"m": 3}, _lister(tmp_path, calls))
    assert calls == []
    assert not builder.submit("op", "k", BuildTask(_RECORD, (str(tmp_path / "x"), 0)))


def test_in_build_mode_the_workers_build_each_shape_swept(builder, tmp_path):
    calls = []
    list_builds = _lister(tmp_path, calls)
    with builder.build_pass("test"):
        builder.prebuild("op", "sig", _CONFIG, {"m": 3}, list_builds)
        # The same call lists nothing new; a larger one lists only the shapes not listed yet.
        builder.prebuild("op", "sig", _CONFIG, {"m": 3}, list_builds)
        builder.prebuild("op", "sig", _CONFIG, {"m": 5}, list_builds)
        RankWorker.get().join()
        assert builder.state("op", (5,)) == "done"
    assert calls == [[1, 2, 3], [4, 5]]
    for m in range(1, 6):
        value, pid = (tmp_path / f"m{m}").read_text().split()
        assert int(value) == m and int(pid) != os.getpid()
    stats = builder.stats("op")
    assert (stats.signatures, stats.submitted) == (1, 5)
    # The pass stopped the workers.
    assert RankWorker.get()._pool is None


def test_an_op_waiting_for_a_build_no_worker_has_started_takes_it_back(builder, tmp_path):
    RankWorker.get().max_workers = 1
    started, release = tmp_path / "started", tmp_path / "release"
    with builder.build_pass("test"):
        try:
            assert builder.submit("op", "busy", BuildTask(_HOLD, (str(started), str(release))))
            deadline = time.monotonic() + 60
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert builder.submit("op", "k", BuildTask(_RECORD, (str(tmp_path / "k"), 1)))
            # The only worker is busy: the op builds the kernel itself, at once.
            assert builder.wait("op", "k") == 0.0
            assert builder.state("op", "k") == "taken"
        finally:
            release.touch()
    assert not (tmp_path / "k").exists()


def test_a_listing_that_fails_leaves_the_kernels_to_the_op(builder, tmp_path):
    def list_builds(shapes):
        raise RuntimeError("no listing")

    with builder.build_pass("test"):
        builder.prebuild("op", "sig", _CONFIG, {"m": 2}, list_builds)
    assert builder.stats("op").submitted == 0


def test_a_build_the_workers_do_not_take_is_left_to_the_op(builder, tmp_path):
    with builder.build_pass("test"):
        assert not builder.submit("op", "lambda", BuildTask(_RECORD, (lambda: 0, 0)))
    stats = builder.stats("op")
    assert (stats.submitted, stats.inline) == (0, 1)


def test_only_the_memory_profiling_warmup_is_build_mode(builder):
    # A warmup with the final KV cache builds in this process.
    with builder.warmup_pass("final_executor"):
        assert not builder.active()
    with builder.warmup_pass("memory_profiling"):
        assert builder.active()


def test_with_kernel_builder_off_nothing_is_listed(monkeypatch, tmp_path):
    monkeypatch.setenv("TLLM_KERNEL_BUILDER", "0")
    kb = KernelBuilder()
    calls = []
    with kb.build_pass("test"):
        assert not kb.active()
        kb.prebuild("op", "sig", _CONFIG, {"m": 2}, _lister(tmp_path, calls))
    assert calls == []
