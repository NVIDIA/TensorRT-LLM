# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KernelBuilder: builds an op lists for the shapes it sweeps, run on worker threads."""

import sys
import threading
import types

import pytest
import torch

from tensorrt_llm._torch import kernel_builder
from tensorrt_llm._torch.kernel_builder import (
    BuildConfig,
    DynamicSpec,
    KernelBuilder,
    builds_by_running,
)

pytestmark = pytest.mark.cpu_only


def _prebuild(builder, signature, list_builds):
    """``prebuild`` for an op without build dimensions."""
    builder.prebuild("op", signature, BuildConfig(), {}, lambda shapes: list_builds())


def _recorder(built, key, compiled=True):
    def build():
        built.append(key)
        return compiled

    return build


def test_kernels_are_listed_once_per_signature_inside_a_pass():
    builder = KernelBuilder(threads=2, enabled=True)
    listed, built = [], []

    def list_builds():
        listed.append(True)
        return [(name, _recorder(built, name)) for name in ("a", "b", "c")]

    _prebuild(builder, "sig", list_builds)
    assert listed == []  # no pass, no listing

    with builder.build_pass("test"):
        _prebuild(builder, "sig", list_builds)
        _prebuild(builder, "sig", list_builds)
        # A kernel another signature also reaches is built once.
        _prebuild(builder, "sig2", lambda: [("a", _recorder(built, "a"))])
    # The pass waited for the builds.
    assert listed == [True]
    assert sorted(built) == ["a", "b", "c"]
    assert builder.state("op", "b") == "done"
    assert not builder.active()
    stats = builder.stats("op")
    assert (stats.signatures, stats.submitted, stats.built) == (2, 3, 3)


def test_a_failed_build_is_left_to_the_op():
    builder = KernelBuilder(threads=2, enabled=True)
    built = []

    def fail():
        raise RuntimeError("nvcc failed")

    def exit_():
        raise SystemExit(1)

    builds = [("bad", fail), ("exit", exit_), ("good", _recorder(built, "good"))]
    with builder.build_pass("test"):
        _prebuild(builder, "sig", lambda: builds)
    # The pass ended: even an item raising outside Exception was settled.
    assert built == ["good"]
    assert builder.state("op", "bad") == builder.state("op", "exit") == "failed"
    # The op compiling the failed kernel itself counts as a late build, not a wait.
    builder.op_compiled("op", "bad", 80.0)
    stats = builder.stats("op")
    assert (stats.submitted, stats.built, stats.failed, stats.late, stats.waited) == (3, 1, 2, 1, 0)


def test_a_listing_failure_leaves_the_kernels_to_the_op():
    builder = KernelBuilder(threads=1, enabled=True)

    def list_builds():
        raise ValueError("unsupported shape")

    with builder.build_pass("test"):
        _prebuild(builder, "sig", list_builds)
    assert builder.state("op", "anything") is None


def test_the_op_waits_for_a_build_in_flight():
    builder = KernelBuilder(threads=1, enabled=True)
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(timeout=30)
        return True

    builder.submit("op", "k", slow)
    assert started.wait(timeout=30)
    assert builder.state("op", "k") == "queued"
    builder.op_compiled("op", "k", 5.0)
    release.set()
    builder.join(timeout=30)
    assert builder.state("op", "k") == "done"
    stats = builder.stats("op")
    assert (stats.built, stats.waited, stats.waited_s, stats.late) == (1, 1, 5.0, 0)


def test_an_op_builds_the_shapes_its_dimensions_sweep_from_the_call():
    config = BuildConfig(
        (
            DynamicSpec("m", lambda m: [x for x in (1, 2, 4, 8) if x <= m]),
            DynamicSpec("splits", (1, 2)),
        )
    )
    # A function's buckets come from the call's value, a tuple's are fixed, and the call's
    # own value is always one of them.
    assert config.shapes({"m": 6, "splits": 1}) == [
        {"m": m, "splits": splits} for m in (1, 2, 4, 6) for splits in (1, 2)
    ]
    builder = KernelBuilder(threads=2, enabled=True)
    swept, built = [], []

    def list_builds(shapes):
        swept.extend(shapes)
        # Shapes that reach the same kernel list it once.
        return [(shape["m"], _recorder(built, shape["m"])) for shape in shapes]

    with builder.build_pass("test"):
        builder.prebuild("op", "sig", config, {"m": 6, "splits": 1}, list_builds)
        # A smaller call of the signature sweeps nothing new; a larger one only its new shapes.
        builder.prebuild("op", "sig", config, {"m": 4, "splits": 1}, list_builds)
        assert len(swept) == 8
        builder.prebuild("op", "sig", config, {"m": 9, "splits": 1}, list_builds)
    assert swept[8:] == [
        {"m": 8, "splits": 1},
        {"m": 8, "splits": 2},
        {"m": 9, "splits": 1},
        {"m": 9, "splits": 2},
    ]
    assert sorted(built) == [1, 2, 4, 6, 8, 9]
    assert builder.stats("op").signatures == 1


def test_a_rehearsal_leaves_its_builds_running_for_the_next_pass():
    builder = KernelBuilder(threads=1, enabled=True)
    release = threading.Event()

    def slow():
        release.wait(timeout=30)
        return True

    assert not builder.rehearsing()
    with builder.rehearsal():
        assert builder.rehearsing() and builder.active()
        _prebuild(builder, "sig", lambda: [("k", slow)])
    # The rehearsal did not wait: the build is still running.
    assert not builder.rehearsing() and not builder.active()
    assert builder.state("op", "k") == "queued"
    release.set()
    with builder.build_pass("warmup"):
        # The signature was listed in the rehearsal.
        _prebuild(builder, "sig", lambda: pytest.fail("listed twice"))
    # The warmup pass waited for it.
    assert builder.state("op", "k") == "done"


def test_the_rehearsal_runs_for_a_model_an_op_family_reaches(monkeypatch):
    reached = []
    family = types.ModuleType("family")
    family.reaches = (
        lambda model_config, model: reached.append((model_config, model)) or model_config == "msa"
    )
    monkeypatch.setitem(sys.modules, "family", family)
    monkeypatch.setattr(kernel_builder, "_FAMILIES", {"op": "family:reaches"})
    builder = KernelBuilder(threads=1, enabled=True)
    assert builder.wants_rehearsal("msa", "model")
    assert not builder.wants_rehearsal("dense", "model")
    assert reached == [("msa", "model"), ("dense", "model")]
    assert not KernelBuilder(threads=1, enabled=True, rehearse=False).wants_rehearsal("msa")
    assert not KernelBuilder(threads=1, enabled=False).wants_rehearsal("msa")


def test_join_can_wait_only_for_the_builds_that_use_the_gpu():
    builder = KernelBuilder(threads=2, enabled=True)
    release = threading.Event()
    builder.submit("compile", "k", lambda: release.wait(timeout=30))
    builder.submit("run", "k", lambda: True, uses_gpu=True)
    assert builder.join(timeout=30, gpu_only=True) < 30
    assert builder.state("run", "k") == "done"
    assert builder.state("compile", "k") == "queued"
    release.set()
    builder.join(timeout=30)
    assert builder.state("compile", "k") == "done"


def test_an_op_built_by_running_runs_once_per_swept_shape_in_a_rehearsal(monkeypatch):
    builder = KernelBuilder(threads=2, enabled=True)
    monkeypatch.setattr(KernelBuilder, "_instance", builder)
    runs = []

    @builds_by_running(
        BuildConfig((DynamicSpec("m", (1, 8)),)),
        dims={"m": [(0, 0), ("out", 0)]},
        output=lambda x, w, out: torch.zeros(x.shape[0], w.shape[1]),
        name="op",
    )
    def op(x, w, out):
        runs.append(
            (x.clone(), out.shape[0], threading.get_ident(), torch.is_inference_mode_enabled())
        )
        return x @ w

    x = torch.arange(12.0).reshape(6, 2)
    w = torch.ones(2, 3)
    # Outside a rehearsal the op runs as it is.
    assert torch.equal(op(x, w, out=torch.empty(6, 3)), x @ w)
    runs.clear()
    with builder.rehearsal():
        assert torch.equal(op(x, w, out=torch.empty(6, 3)), torch.zeros(6, 3))
        # A call of the same signature at a swept size queues nothing new.
        op(x[:1], w, out=torch.empty(1, 3))
    builder.join(timeout=30)
    # The call's own size and the swept ones, each run once, off the calling thread.
    assert sorted(run[0].shape[0] for run in runs) == [1, 6, 8]
    assert all(run[1] == run[0].shape[0] for run in runs)
    assert all(run[2] != threading.get_ident() and run[3] for run in runs)
    # A larger size repeats the call's rows.
    tiled = next(run[0] for run in runs if run[0].shape[0] == 8)
    assert torch.equal(tiled, torch.cat([x, x[:2]]))
    assert builder.stats("op").submitted == 3


def test_disabled():
    builder = KernelBuilder(threads=1, enabled=False)
    listed = []
    with builder.build_pass("test"):
        assert not builder.active()
        _prebuild(builder, "sig", lambda: listed.append(True) or [])
    with builder.rehearsal():
        assert not builder.rehearsing()
    assert listed == []
    assert not builder.submit("op", "k", lambda: True)
