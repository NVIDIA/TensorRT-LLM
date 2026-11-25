# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start a deepseek_v41 multi-rank launcher (`_*_matrix.py` and friends) in a fresh interpreter, as
`comm/_rank_job.py` starts the catalog's.

* The ranks import tensorrt_llm absolutely. The checkout is put on their path only when it is the tensorrt_llm this
  process resolves (an in-tree build, or nothing installed): CI runs from an unbuilt checkout, whose tensorrt_llm/ has
  no compiled bindings, beside an installed wheel, and a checkout first on the path breaks every rank's import.
* The subprocess deadline is the launcher's own `DEADLINE_S` plus a grace, so the launcher's "(wedged)" diagnostics
  win the race; each launcher keeps its deadline below the CI timeouts around it (the inner pytest's 2400 s and its
  test-db entry's TIMEOUT).
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

# tests/unittest/_torch/modeling_v2/models/deepseek_v41 -> repo root
REPO_ROOT = Path(__file__).resolve().parents[6]
GRACE_S = 120


def rank_env(devices: Optional[str] = None) -> dict:
    """This process's environment for the ranks: `CUDA_VISIBLE_DEVICES` = `devices`, and the checkout on PYTHONPATH
    only when it is the tensorrt_llm this process imports."""
    env = dict(os.environ)
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    installed = importlib.util.find_spec("tensorrt_llm")
    if installed is None or Path(installed.origin).resolve().parents[1] == REPO_ROOT:
        assert (REPO_ROOT / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ under {REPO_ROOT}"
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
        )
    return env


def launcher_deadline(launcher: Path) -> int:
    """The launcher's module-level `DEADLINE_S`, read from its source (it must not be imported: MPI_Init)."""
    match = re.search(r"^DEADLINE_S = (\d+)", Path(launcher).read_text(), re.M)
    assert match, f"{launcher.name} states no DEADLINE_S"
    return int(match.group(1))


def run_launcher(
    launcher: Path, devices: str, args: Sequence[str] = (), env: Optional[dict] = None
) -> subprocess.CompletedProcess:
    """Run `launcher` over `devices` and fail with its output unless it exits 0."""
    done = subprocess.run(
        [sys.executable, str(launcher), *args],
        env={**rank_env(devices), **(env or {})},
        capture_output=True,
        text=True,
        timeout=launcher_deadline(launcher) + GRACE_S,
    )
    assert done.returncode == 0, (
        f"{Path(launcher).name} exited {done.returncode}\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    )
    return done
