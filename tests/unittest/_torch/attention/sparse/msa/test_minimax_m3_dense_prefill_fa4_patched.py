# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged FA4 accuracy suite with its pinned patch enabled in an isolated process."""

import hashlib
import importlib.util
import os
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="SM107 FA4 numerical validation",
)
def test_dense_prefill_fa4_patched_accuracy_suite() -> None:
    """Require executed numerical checks; an optional-package skip cannot satisfy this gate."""
    root = Path(__file__).resolve().parents[6]
    setup = root / "examples/minimax_m3/rubin/setup_fa4.sh"
    suite = Path(__file__).with_name("test_minimax_m3_dense_prefill_fa4.py")
    pin = re.search(r'^readonly PATCHED_SHA256="([0-9a-f]{64})"$', setup.read_text(), re.MULTILINE)
    assert pin is not None
    spec = importlib.util.find_spec("flash_attn.cute")
    assert spec is not None and spec.submodule_search_locations, "pinned flash-attn-4 is required"
    installed = Path(next(iter(spec.submodule_search_locations))) / "flash_fwd_sm100.py"
    runtime = importlib.util.find_spec("tensorrt_llm")
    assert runtime is not None and runtime.origin is not None
    runtime_root = Path(runtime.origin).resolve().parents[1]
    already_patched = hashlib.sha256(installed.read_bytes()).hexdigest() == pin[1]
    with tempfile.TemporaryDirectory(prefix="trtllm-fa4-validation-", dir="/tmp") as temporary:
        scratch = Path(temporary)
        binaries = scratch / "bin"
        binaries.mkdir()
        interpreter = binaries / "python3"
        interpreter.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n')
        interpreter.chmod(0o755)
        env = {
            **os.environ,
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "TMPDIR": str(scratch),
            "TLLM_DISABLE_MPI": "1",
            "TRTLLM_MSA_DISABLE_DENSE_FA4": "0",
        }
        for name in tuple(env):
            if name.startswith(("OMPI_", "PMI_", "PMIX_")) or name in {
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
            }:
                del env[name]
        for name in (
            "PYTHONPYCACHEPREFIX",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUTE_DSL_CACHE_DIR",
            "XDG_CACHE_HOME",
            "CUDA_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "FLASHINFER_WORKSPACE_BASE",
        ):
            env[name] = str(scratch / name.lower())
        overlay = scratch / "fa4"
        if not already_patched:
            prepared = subprocess.run(
                ["bash", str(setup), str(overlay)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            assert prepared.returncode == 0, prepared.stdout + prepared.stderr
        env["PYTHONPATH"] = os.pathsep.join(
            entry
            for entry in (
                str(overlay) if not already_patched else "",
                str(runtime_root),
                *(
                    entry
                    for entry in env.get("PYTHONPATH", "").split(os.pathsep)
                    if entry and (runtime_root == root or Path(entry).resolve() != root)
                ),
            )
            if entry
        )
        report = scratch / "junit.xml"
        # A fresh interpreter loads the patched package before any FA4 imports and runs only the numeric module.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rs",
                str(suite),
                "--junitxml",
                str(report),
                "--basetemp",
                str(scratch / "pytest"),
                "-o",
                f"cache_dir={scratch / 'pytest-cache'}",
            ],
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert report.is_file(), output
        cases = list(ET.parse(report).iter("testcase"))
        skipped = sum(case.find("skipped") is not None for case in cases)
        failed = sum(
            case.find("failure") is not None or case.find("error") is not None for case in cases
        )
        assert len(cases) >= 6 and skipped == 0 and failed == 0, (
            f"FA4 numeric cases={len(cases)}, skipped={skipped}, failed={failed}\n{output}"
        )
        print(f"Patched FA4 accuracy: {len(cases)} passed, zero skips")
