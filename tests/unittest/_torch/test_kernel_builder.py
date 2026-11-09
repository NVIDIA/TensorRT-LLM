# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KernelBuilder: builds listed by an op on first sight, run on worker threads."""

import threading

import pytest

from tensorrt_llm._torch.kernel_builder import KernelBuilder

pytestmark = pytest.mark.cpu_only


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

    builder.prebuild("op", "sig", list_builds)
    assert listed == []  # no pass, no listing

    with builder.build_pass("test"):
        builder.prebuild("op", "sig", list_builds)
        builder.prebuild("op", "sig", list_builds)
        # A kernel another signature also reaches is built once.
        builder.prebuild("op", "sig2", lambda: [("a", _recorder(built, "a"))])
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
        builder.prebuild("op", "sig", lambda: builds)
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
        builder.prebuild("op", "sig", list_builds)
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


def test_disabled():
    builder = KernelBuilder(threads=1, enabled=False)
    listed = []
    with builder.build_pass("test"):
        assert not builder.active()
        builder.prebuild("op", "sig", lambda: listed.append(True) or [])
    assert listed == []
    assert not builder.submit("op", "k", lambda: True)
