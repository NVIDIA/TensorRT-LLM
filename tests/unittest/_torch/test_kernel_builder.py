# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KernelBuilder: in build mode, an op's calls list the kernels of the shapes its BuildConfig
sweeps and build them on the rank's worker processes, and go on once they are built; in
inference mode nothing is listed.

The builds are build_pool_targets.py's, which write a file, so no GPU is needed.
"""

import os
import threading
import time

import build_pool_targets as targets
import pytest

from tensorrt_llm import _jit_share
from tensorrt_llm._torch import build_pool, kernel_builder
from tensorrt_llm._torch.build_pool import BuildPool
from tensorrt_llm._torch.kernel_builder import (
    BuildConfig,
    BuildStrategy,
    BuildTask,
    DynamicSpec,
    KernelBuilder,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _every_build_to_a_worker(monkeypatch):
    """Workers take every build, however short; the tests of that rule set it themselves."""
    monkeypatch.setenv("TLLM_KERNEL_BUILDER_MIN_BUILD_S", "0")


_RECORD = f"{targets.__name__}:record"
_FAIL = f"{targets.__name__}:fail"


@pytest.fixture
def builder(monkeypatch, tmp_path):
    """This process's KernelBuilder, and a BuildPool of up to two workers that import
    build_pool_targets."""
    original = build_pool._worker_env
    here = os.path.dirname(os.path.abspath(targets.__file__))

    def worker_env():
        env = original()
        env[targets.IN_WORKER_ENV] = "1"
        env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
        return env

    monkeypatch.setattr(build_pool, "_worker_env", worker_env)
    rank = BuildPool(workers=2, enabled=True, store=str(tmp_path / "store"))
    monkeypatch.setattr(BuildPool, "_instance", rank)
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


def _record(path, value, config=_CONFIG):
    return BuildTask(_RECORD, (str(path), value)), config


def test_a_build_config_sweeps_the_buckets_with_the_call_value():
    config = BuildConfig((DynamicSpec("m", (1, 4)), DynamicSpec("splits", (1, 2))))
    shapes = config.shapes({"m": 3, "splits": 1})
    assert shapes == [{"m": m, "splits": s} for m in (1, 3, 4) for s in (1, 2)]


def test_in_inference_mode_an_op_lists_and_builds_nothing(builder, tmp_path):
    calls = []
    builder.prebuild("op", "sig", _CONFIG, {"m": 3}, _lister(tmp_path, calls))
    builder.build("op", [("k", *_record(tmp_path / "x", 0))])
    assert calls == [] and not (tmp_path / "x").exists()


def test_in_build_mode_a_call_goes_on_once_the_workers_built_each_shape_swept(builder, tmp_path):
    calls = []
    list_builds = _lister(tmp_path, calls)
    with builder.build_pass("test"):
        builder.prebuild("op", "sig", _CONFIG, {"m": 3}, list_builds)
        assert all((tmp_path / f"m{m}").exists() for m in (1, 2, 3))
        # The same call lists nothing new; a larger one lists only the shapes not listed yet.
        builder.prebuild("op", "sig", _CONFIG, {"m": 3}, list_builds)
        builder.prebuild("op", "sig", _CONFIG, {"m": 5}, list_builds)
        assert (tmp_path / "m5").exists()
    assert calls == [[1, 2, 3], [4, 5]]
    for m in range(1, 6):
        value, pid = (tmp_path / f"m{m}").read_text().split()
        assert int(value) == m and int(pid) != os.getpid()
    stats = builder.stats("op")
    assert (stats.signatures, stats.built) == (1, 5)
    # The pass stopped the workers.
    assert BuildPool.get()._pool is None


def test_a_build_asked_for_again_is_not_built_again(builder, tmp_path):
    with builder.build_pass("test"):
        builder.build("op", [("k", *_record(tmp_path / "k", 1))])
        (tmp_path / "k").unlink()
        builder.build("op", [("k", *_record(tmp_path / "k", 1))])
    assert not (tmp_path / "k").exists()
    assert builder.stats("op").built == 1


def test_a_listing_that_fails_leaves_the_kernels_to_the_op(builder, tmp_path):
    def list_builds(shapes):
        raise RuntimeError("no listing")

    with builder.build_pass("test"):
        builder.prebuild("op", "sig", _CONFIG, {"m": 2}, list_builds)
    assert builder.stats("op").built == 0


def test_a_build_the_workers_do_not_take_or_that_fails_is_left_to_the_op(builder, tmp_path):
    with builder.build_pass("test"):
        builder.build(
            "op",
            [
                ("lambda", BuildTask(_RECORD, (lambda: 0, 0)), _CONFIG),
                ("fails", BuildTask(_FAIL), _CONFIG),
            ],
        )
    stats = builder.stats("op")
    assert (stats.built, stats.failed, stats.inline) == (0, 1, 1)


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


def test_builds_too_short_for_a_worker_are_not_listed(builder, tmp_path):
    BuildPool.get().min_build_s = 5.0
    calls = []
    with builder.build_pass("test"):
        # Neither measured nor expected: the op builds them on first use.
        builder.prebuild("op", "sig", _CONFIG, {"m": 2}, _lister(tmp_path, calls))
        BuildPool.get().record_build("measured", 6.0)
        builder.prebuild("measured", "sig", _CONFIG, {"m": 2}, _lister(tmp_path, calls))
        expected = BuildConfig(_CONFIG.dynamic_specs, build_seconds=30.0)
        builder.prebuild("expected", "sig", expected, {"m": 1}, _lister(tmp_path, calls))
    assert calls == [[1, 2], [1]]


@pytest.fixture
def shared(monkeypatch, tmp_path):
    """Ranks that share their JIT builds through a directory."""
    share = tmp_path / "share"
    share.mkdir()
    monkeypatch.setenv(_jit_share.SHARE_DIR_ENV, str(share))
    monkeypatch.setattr(_jit_share, "enabled", lambda: True)
    return share


def _parallel(scope="here"):
    return BuildConfig(strategy=BuildStrategy.PARALLEL, share_scope=lambda: scope)


def test_ranks_split_parallel_builds(builder, shared, tmp_path):
    other = KernelBuilder(enabled=True)  # Another rank of the launch.
    with builder.build_pass("a"):
        builder.build("op", [("k", *_record(tmp_path / "k", 1, _parallel()))])
    with other.build_pass("b"):
        # The other rank finds the build made, and makes nothing.
        other.build("op", [("k", *_record(tmp_path / "k2", 1, _parallel()))])
    assert (tmp_path / "k").exists() and not (tmp_path / "k2").exists()
    assert (builder.stats("op").built, other.stats("op").elsewhere) == (1, 1)


def test_ranks_whose_kernels_land_apart_build_their_own(builder, shared, tmp_path):
    other = KernelBuilder(enabled=True)
    with builder.build_pass("a"):
        builder.build("op", [("k", *_record(tmp_path / "a", 1, _parallel("a")))])
    with other.build_pass("b"):
        other.build("op", [("k", *_record(tmp_path / "b", 1, _parallel("b")))])
    assert (tmp_path / "a").exists() and (tmp_path / "b").exists()


def test_a_listed_build_of_its_own_config_is_split_as_its_config_says(builder, shared, tmp_path):
    def list_builds(shapes):
        return [
            ("own", *_record(tmp_path / "own", 1, _parallel())),
            ("op's", *_record(tmp_path / "op", 2)[:1]),
        ]

    with builder.build_pass("a"):
        builder.prebuild("op", "sig", _CONFIG, {"m": 1}, list_builds)
    assert (tmp_path / "own").exists() and (tmp_path / "op").exists()
    # Only the PARALLEL build was claimed.
    own = kernel_builder._claim_files("op", "own", "here")
    claims = sorted(os.listdir(shared / "kernel_builder"))
    assert claims == sorted(os.path.basename(path) for path in (own.claim, own.done))


def test_a_parallel_build_marks_its_end_for_the_other_ranks(builder, shared, tmp_path):
    with builder.build_pass("a"):
        builder.build(
            "op",
            [
                ("made", *_record(tmp_path / "made", 1, _parallel())),
                ("fails", BuildTask(_FAIL), _parallel()),
            ],
        )
    made = kernel_builder._claim_files("op", "made", "here")
    fails = kernel_builder._claim_files("op", "fails", "here")
    assert all(os.path.exists(path) for path in (made.claim, made.done))
    assert not os.path.exists(made.failed)
    assert all(os.path.exists(path) for path in (fails.claim, fails.failed))
    assert not os.path.exists(fails.done)


def _claimed_elsewhere(key):
    """Claim ``("op", key)`` for another rank, whose worker has started it."""
    claim = kernel_builder._claim_files("op", key, "here")
    os.makedirs(os.path.dirname(claim.claim), exist_ok=True)
    open(claim.claim, "w").close()
    return claim


def _ends_elsewhere(monkeypatch, path):
    """Once this rank's workers found its builds claimed, the other rank's build creates
    ``path`` (its done or failed mark)."""
    pool = BuildPool.get()
    build = pool.build
    timers = []

    def build_then_end_elsewhere(builds, on_done=None):
        results = build(builds, on_done)
        timers.append(threading.Timer(0.5, lambda: open(path, "w").close()))
        timers[-1].start()
        return results

    monkeypatch.setattr(pool, "build", build_then_end_elsewhere)
    return timers


def test_a_rank_waits_for_a_build_another_rank_claimed(builder, shared, tmp_path, monkeypatch):
    claim = _claimed_elsewhere("k")
    timers = _ends_elsewhere(monkeypatch, claim.done)
    with builder.build_pass("a"):
        builder.build("op", [("k", *_record(tmp_path / "x", 1, _parallel()))])
        # The call went on once the other rank's build had ended.
        assert os.path.exists(claim.done)
    for timer in timers:
        timer.join()
    assert not (tmp_path / "x").exists()
    assert builder.stats("op").elsewhere == 1


def test_a_rank_stops_waiting_for_a_build_that_failed_elsewhere(
    builder, shared, tmp_path, monkeypatch
):
    monkeypatch.setenv("TLLM_JIT_SHARE_WAIT_S", "120")
    claim = _claimed_elsewhere("k")
    timers = _ends_elsewhere(monkeypatch, claim.failed)
    with builder.build_pass("a"):
        start = time.monotonic()
        builder.build("op", [("k", *_record(tmp_path / "x", 1, _parallel()))])
        assert time.monotonic() - start < 60
    for timer in timers:
        timer.join()
    # The op builds it on first use.
    assert not (tmp_path / "x").exists()
