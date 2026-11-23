# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's GEN-step shell (``ops/common/k3_ull_runtime.py``), host-side.

Covered without a GPU: ``make_runtime_hooks`` declares the served shell; the executor-pinning core
choice equals served ``ull_shell.pin_executor_cpu`` on synthetic topologies; the pin and the GIL
switch interval are applied; the provenance manifest names functions that exist on both sides. The
engine hook points the shell uses are tested in ``tests/unittest/_torch/pyexecutor/``
(``test_runtime_hooks.py``, ``test_step_inputs.py``)."""

import ast
import json
import pathlib
import subprocess
import sys
import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.common import (  # noqa: E501
    k3_ull_runtime as runtime,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks

pytestmark = pytest.mark.cpu_only

_FAM_DIR = pathlib.Path(runtime.__file__).resolve().parents[2]
_REPO = _FAM_DIR.parents[4]
_MANIFEST = _FAM_DIR / "FLAT_RUNTIME_SOURCE.json"
SERVED_COMMIT = "6904cd9672"


def test_make_runtime_hooks_declares_the_served_shell():
    base = RuntimeHooks(skip_greedy_rng_upload=True)
    hooks = runtime.make_runtime_hooks(base)
    assert hooks.pack_step_inputs and hooks.pack_spec_rng_uploads
    assert callable(hooks.step_input_pack_factory)
    assert isinstance(hooks.spec_step_fold, runtime.OverlapFixupFold)
    assert isinstance(hooks.executor_events, runtime.StepShellEvents)
    assert callable(getattr(hooks.executor_events, "on_iteration", None))
    assert hooks.skip_greedy_rng_upload is base.skip_greedy_rng_upload
    # Nothing else of the measured run's engine changes.
    for name in (
        "pack_parallel_scatter",
        "pack_in_graph",
        "loop_on_execution_stream",
        "skip_unchanged_h2d",
    ):
        assert getattr(hooks, name) == getattr(DEFAULT_RUNTIME_HOOKS, name), name
    assert runtime.GIL_SWITCH_US == 200
    assert runtime.InputCommitPack.issue_after_prepare


def test_fold_declines_without_an_open_commit():
    fold = runtime.OverlapFixupFold()
    assert fold.begin_step([0], 1) is False
    assert fold.sampler_rows_stored({}, 0, [0]) is False
    assert fold.scatter(*([None] * 5)) is False
    t = torch.zeros(1, dtype=torch.int32)
    assert fold.gather(t, t, t, t, t, 1, 8, 7, 0, 0, 0) is False
    assert runtime.fold_block_offsets_copy(t, t, t, t, t) is False


def test_commit_pack_declines_host_destinations():
    pack = object.__new__(runtime.InputCommitPack)
    pack._dtypes = frozenset({torch.int32})
    pack.commit = None  # a declined copy never reaches the commit
    assert pack.stage(torch.zeros(2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)) is False


def _topology(siblings_of):
    return lambda cpu: set(siblings_of.get(cpu, {cpu})) | {cpu}


_TOPOLOGIES = [
    # 8 CPUs, SMT pairs (i, i + 4): physical cores 0..3, core 0 skipped.
    {i: {i % 4, i % 4 + 4} for i in range(8)},
    # No SMT.
    {},
    # A NUMA set with CPU 0's core outside it.
    {i: {i, i + 72} for i in range(72)} | {i + 72: {i, i + 72} for i in range(72)},
]


@pytest.mark.parametrize("topo", range(len(_TOPOLOGIES)))
@pytest.mark.parametrize("gpu", range(4))
def test_pinned_core_choice(topo, gpu, monkeypatch):
    monkeypatch.setattr(runtime, "smt_siblings", _topology(_TOPOLOGIES[topo]))
    available = set(range(8)) if topo < 2 else set(range(36, 72)) | set(range(108, 144))
    cpu = runtime.pin_executor_cpu(available, gpu_index=gpu)
    assert cpu in available
    assert 0 not in runtime.smt_siblings(cpu) or available == {0}
    expected = {0: [1, 2, 3, 1], 1: [1, 2, 3, 4], 2: [36 + gpu] * 4}[topo][gpu]
    assert cpu == expected


def _served_ull_shell():
    out = subprocess.run(
        ["git", "show", f"{SERVED_COMMIT}:tensorrt_llm/_torch/pyexecutor/ull_shell.py"],
        cwd=_REPO,
        capture_output=True,
    )
    if out.returncode:
        pytest.skip(f"served commit {SERVED_COMMIT} is not in this repository")
    module = types.ModuleType("served_ull_shell")
    exec(compile(out.stdout.decode(), "served_ull_shell.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("topo", range(len(_TOPOLOGIES)))
def test_pinned_core_choice_equals_served(topo, monkeypatch):
    served = _served_ull_shell()
    fake = _topology(_TOPOLOGIES[topo])
    monkeypatch.setattr(runtime, "smt_siblings", fake)
    monkeypatch.setattr(served, "_smt_siblings", fake)
    monkeypatch.setattr(served, "_SHELL", served.StepShell(pin_executor=True))
    available = set(range(8)) if topo < 2 else set(range(36, 72)) | set(range(108, 144))
    for gpu in range(8):
        ours = runtime.pin_executor_cpu(available, gpu_index=gpu)
        assert ours == served.pin_executor_cpu(available, gpu_index=gpu)


@pytest.mark.parametrize(
    ("visible", "device", "expected_cpu"),
    [
        (None, 2, 3),
        ("6,1", 1, 2),
        ("3", 0, 4),
        ("GPU-00000000-0000-0000-0000-000000000001", 0, None),
        ("MIG-00000000-0000-0000-0000-000000000001", 0, None),
        ("3", None, None),
    ],
)
def test_executor_pin_requires_a_known_gpu_index(visible, device, expected_cpu, monkeypatch):
    cuda = types.SimpleNamespace(
        is_available=lambda: device is not None, current_device=lambda: device
    )
    monkeypatch.setattr(runtime, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runtime, "smt_siblings", lambda cpu: {cpu})
    affinity_changes = []
    monkeypatch.setattr(
        runtime,
        "os",
        types.SimpleNamespace(
            environ={} if visible is None else {"CUDA_VISIBLE_DEVICES": visible},
            sched_getaffinity=lambda pid: set(range(1, 9)),
            sched_setaffinity=lambda pid, cpus: affinity_changes.append((pid, cpus)),
            listdir=lambda path: [],
        ),
    )

    pin = runtime.ExecutorThreadPin.create()
    if expected_cpu is None:
        assert pin is None
        assert affinity_changes == []
    else:
        assert pin is not None and pin.cpu == expected_cpu
        assert affinity_changes == [(0, {expected_cpu})]


def test_executor_pin_moves_the_other_threads(monkeypatch):
    monkeypatch.setattr(runtime, "smt_siblings", _topology(_TOPOLOGIES[0]))
    monkeypatch.setattr(runtime, "physical_gpu_index", lambda: 1)
    me = runtime.threading.get_native_id()
    tasks = [str(me), "111", "222"]
    calls = []
    fake_os = types.SimpleNamespace(
        sched_getaffinity=lambda pid: set(range(8)),
        sched_setaffinity=lambda pid, cpus: calls.append((pid, frozenset(cpus))),
        listdir=lambda path: list(tasks),
        environ={},
    )
    monkeypatch.setattr(runtime, "os", fake_os)
    pin = runtime.ExecutorThreadPin.create()
    assert pin is not None and pin.cpu == 2
    others = frozenset(range(8)) - {2, 6}
    assert calls == [(0, frozenset({2})), (111, others), (222, others)]
    calls.clear()
    tasks.append("333")
    pin.maybe_refresh(runtime.PIN_REFRESH_ITERS - 1)
    assert calls == []
    pin.maybe_refresh(runtime.PIN_REFRESH_ITERS)
    assert calls == [(333, others)]


def test_before_overlap_loop_sets_the_switch_interval(monkeypatch):
    monkeypatch.setattr(runtime.ExecutorThreadPin, "create", classmethod(lambda cls: None))
    previous = sys.getswitchinterval()
    try:
        events = runtime.StepShellEvents()
        events.before_overlap_loop()
        # CPython keeps the interval in whole microseconds (200e-6 reads back as 199e-6).
        assert round(sys.getswitchinterval() * 1e6) in (199, 200)
        events.on_iteration(runtime.PIN_REFRESH_ITERS)  # no pin: only the fold counters
        events.after_warmup()
        events.at_shutdown()
    finally:
        sys.setswitchinterval(previous)


def test_the_only_environment_read_is_the_gpu_index():
    tree = ast.parse(pathlib.Path(runtime.__file__).read_text())
    reads = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
    ]
    assert reads == ["os.environ"]
    assert '"CUDA_VISIBLE_DEVICES"' in pathlib.Path(runtime.__file__).read_text()


def test_served_step_kernels_are_byte_copies():
    files = json.loads((_FAM_DIR / "kernels" / "SOURCE.json").read_text())["files"]
    for name in ("kernels/k3r_step/input_commit.py", "kernels/k3r_step/step_upload_kernels.py"):
        entry = files[name]
        assert entry["source"].endswith(f"@{SERVED_COMMIT}")
        assert entry["local_edits"]
    text = (_FAM_DIR / "kernels/k3r_step/step_upload_kernels.py").read_text()
    assert "os.environ" not in text.split("def _cdiv")[0]


def _defined(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                names |= {
                    f"{node.name}.{item.name}"
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                }
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {t.id for t in targets if isinstance(t, ast.Name)}
    return names


def test_manifest_names_exist():
    manifest = json.loads(_MANIFEST.read_text())
    ours = _defined(ast.parse(pathlib.Path(runtime.__file__).read_text()))
    for name, source in manifest["functions"].items():
        assert name.startswith("k3_ull_runtime.") and name[len("k3_ull_runtime.") :] in ours, name
        path, func = source.split(":")
        out = subprocess.run(
            ["git", "show", f"{SERVED_COMMIT}:{path}"], cwd=_REPO, capture_output=True
        )
        if out.returncode:
            pytest.skip(f"served commit {SERVED_COMMIT} is not in this repository")
        served = _defined(ast.parse(out.stdout.decode()))
        assert func in served, f"{name}: {path}:{func}"
