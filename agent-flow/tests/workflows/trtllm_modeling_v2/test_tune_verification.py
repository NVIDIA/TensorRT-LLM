# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verification policy: one measured A/B, fail closed, and fresh final e2e."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .test_tune_stage import _microbench


def test_wrapper_times_callables_not_claimed_numbers(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import microbench

    script = Path(_microbench(tmp_path)["script"])
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "microbench.py",
            "--script",
            str(script),
            "--output",
            str(output),
            "--tree",
            str(tmp_path),
            "--commit",
            "revision",
            "--harness-sha256",
            digest,
        ],
    )
    identity = Mock()
    monkeypatch.setattr(microbench, "_identity", identity)
    monkeypatch.setattr(
        microbench.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(tmp_path / "tensorrt_llm/__init__.py")),
    )
    event = Mock(elapsed_time=Mock(return_value=0.125))
    cuda = SimpleNamespace(Event=Mock(return_value=event), synchronize=Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    operation = Mock()
    monkeypatch.setattr(
        microbench.runpy, "run_path", lambda path: {"benchmark": lambda: {"case": operation}}
    )
    microbench.main()
    assert operation.call_count == 110 and cuda.synchronize.call_count == 2
    assert identity.call_count == 2
    assert microbench.load_result(output, commit="revision", harness=digest, cases=["case"])[
        "latency_us"
    ] == {"case": 125.0}
    monkeypatch.setattr(
        microbench.runpy, "run_path", lambda path: {"benchmark": lambda: {"case": 0.001}}
    )
    with pytest.raises(ValueError, match="not claimed timings"):
        microbench.main()
