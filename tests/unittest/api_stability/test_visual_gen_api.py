# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live VisualGen Python API gate; import errors must fail rather than skip."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from visual_gen_api_snapshot import check_reference, collect_snapshot, render_snapshot

pytestmark = pytest.mark.cpu_only


def test_visual_gen_api_matches_reference() -> None:
    check_reference(collect_snapshot())


def test_visual_gen_snapshot_is_deterministic() -> None:
    assert render_snapshot(collect_snapshot()) == render_snapshot(collect_snapshot())


def test_visual_gen_snapshot_does_not_start_runtime() -> None:
    code = """
import multiprocessing
import torch
from tensorrt_llm import VisualGen
from visual_gen_api_snapshot import collect_snapshot

def forbidden(*args, **kwargs):
    raise AssertionError("The API guard must not construct an engine")

VisualGen.__new__ = staticmethod(forbidden)
collect_snapshot()
assert not torch.cuda.is_initialized()
assert not torch.distributed.is_initialized()
assert not multiprocessing.active_children()
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRTLLM_NO_USAGE_STATS="1")
    paths = [str(Path(__file__).parent)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
