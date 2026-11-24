# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real launcher worker ownership without requiring CUDA collective work."""

import ast
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn, Optional

import pytest

pytestmark = pytest.mark.cpu_only
_FAMILY = Path(__file__).parent
_LAUNCHERS = sorted(
    path
    for target in _FAMILY.glob("minimax_m3_nvfp4__sm_107__tp*")
    for path in target.rglob("_m3*_op_matrix.py")
)


def test_collective_launchers_are_discovered() -> None:
    expected = {
        path
        for target in _FAMILY.glob("minimax_m3_nvfp4__sm_107__tp*")
        for path in target.rglob("*.py")
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "_spawn_ranks"
            for node in ast.parse(path.read_text()).body
        )
    }
    assert expected, "no MiniMax SM107 collective launchers discovered"
    assert set(_LAUNCHERS) == expected


def _launcher(path):
    # The modules themselves require CUDA. Execute their actual stdlib-only launcher
    # function and literal configuration, leaving the GPU worker body isolated.
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_spawn_ranks"
    )
    namespace = dict(
        os=os,
        signal=signal,
        socket=socket,
        subprocess=subprocess,
        sys=sys,
        time=time,
        Path=Path,
        Optional=Optional,
        __file__=str(path),
    )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("path", _LAUNCHERS, ids=lambda p: str(p.relative_to(_FAMILY)))
@pytest.mark.parametrize("failure", ["interrupt", "partial_spawn"])
def test_launcher_reaps_workers_on_exception(monkeypatch, path, failure):
    namespace = _launcher(path)
    tp = int(path.relative_to(_FAMILY).parts[0].rsplit("tp", 1)[1])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(tp))))
    real_popen = subprocess.Popen
    children = []

    def spawn(command, **kwargs):
        if failure == "partial_spawn" and children:
            raise OSError("controlled partial launch failure")
        child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    class Clock:
        monotonic = staticmethod(time.monotonic)

        @staticmethod
        def sleep(delay):
            raise KeyboardInterrupt("controlled launcher interruption")

    monkeypatch.setattr(subprocess, "Popen", spawn)
    namespace["time"] = Clock
    try:
        expected = KeyboardInterrupt if failure == "interrupt" else OSError
        with pytest.raises(expected):
            namespace["_spawn_ranks"]()
        assert children
        assert all(child.poll() is not None for child in children), "launcher leaked a worker"
    finally:
        # Even a regression must not leave test-owned workers behind.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)


@pytest.mark.parametrize("explicit_directory", [False, True], ids=["default", "caller-directory"])
@pytest.mark.parametrize(
    "runner",
    [
        *sorted(_FAMILY.glob("minimax_m3_nvfp4__sm_*/comm/_rank_job.py")),
        _FAMILY.parent / "glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/comm/_rank_job.py",
    ],
    ids=lambda p: p.parent.parent.name,
)
def test_rank_runner_resolves_launcher_without_entry_name_table(
    runner: Path, explicit_directory: bool, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    tree = ast.parse(runner.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    selected = []

    class LauncherLocated(Exception):
        pass

    def load_constants(path: Path) -> NoReturn:
        selected.append(path)
        raise LauncherLocated

    namespace = {
        "Path": Path,
        "os": os,
        "sys": sys,
        "__file__": str(runner),
        "WORLD_SIZES": (2, 4, 8),
        "WORLD_SIZE": 2,
        "require_world": lambda size: "0,1",
        "_devices": lambda: "0,1",
        "_load_launcher_constants": load_constants,
        "importlib": SimpleNamespace(
            util=SimpleNamespace(
                find_spec=lambda name: SimpleNamespace(origin=str(tmp_path / "pkg/__init__.py"))
            )
        ),
    }
    module = ast.Module(body=[run], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(runner), "exec"), namespace)
    args = {"world_size": 2} if any(a.arg == "world_size" for a in run.args.args) else {}
    if explicit_directory:
        args["directory"] = tmp_path
    with pytest.raises(LauncherLocated):
        namespace["run"]("new_collective", **args)
    base = tmp_path if explicit_directory else runner.parent
    assert selected == [base.resolve() / "_new_collective_op_matrix.py"]
