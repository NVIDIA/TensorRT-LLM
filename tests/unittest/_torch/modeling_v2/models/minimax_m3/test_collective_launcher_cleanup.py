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
from typing import Optional

import pytest

pytestmark = pytest.mark.cpu_only
_FAMILY = Path(__file__).parent
_LAUNCHERS = sorted(_FAMILY.glob("minimax_m3_nvfp4__sm_107__tp*/comm/_m3*_op_matrix.py"))


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
    tp = int(path.parents[1].name.rsplit("tp", 1)[1])
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
