# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark datasets, staged in the background from preflight on, never a blocker.

Every accuracy gate runs offline (``HF_DATASETS_OFFLINE=1``), so the dataset must be on
disk where the evaluator looks, before the first benchmark: ``trtllm-eval`` takes it as
``--dataset_path``, and the tree's own accuracy tests (the disaggregated-serving one among
them) read it under ``$LLM_MODELS_ROOT/datasets/...`` (``accuracy_core.py``). A run that
found it missing there lost its benchmarks to it.

:func:`start` runs beside the run's first turns: per benchmark, the spec's
``accuracy.dataset_path``, else the copy under ``$LLM_MODELS_ROOT``, else a download of the
dataset's repository into the workspace (``datasets/<repo>``), first from the local
HuggingFace cache. ``datasets/status.json`` says where each stands. A gate that needs one
still being staged waits for it (:func:`wait`); one that could not be staged runs as
before, and the journal says why. Where the models root lacks the dataset, a root that
holds it beside the models (:func:`models_root`) is what the tree's own tests read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ...infra.files import write_text_atomic

DIRNAME = "datasets"
STATUS_FILENAME = "status.json"
#: Benchmark -> the dataset repository and where the tree's tests read it under the models root.
KNOWN: dict[str, str] = {"gsm8k": "openai/gsm8k"}
#: How long a gate waits for a dataset still being staged.
WAIT_S = 30 * 60.0
STAGING, READY, FAILED = "staging", "ready", "failed"

_DOWNLOAD = """
import sys
from huggingface_hub import snapshot_download
repo, target = sys.argv[1], sys.argv[2]
try:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=target, local_files_only=True)
except Exception:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=target)
"""


def status(workspace: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((Path(workspace) / DIRNAME / STATUS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _set(workspace: Path, benchmark: str, **entry: Any) -> None:
    data = status(workspace)
    data[benchmark] = {**entry, "at": time.time()}
    (Path(workspace) / DIRNAME).mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(workspace) / DIRNAME / STATUS_FILENAME, json.dumps(data, indent=1))


def _holds_files(path: Path) -> bool:
    try:
        return path.is_dir() and any(
            p.is_file() for p in path.rglob("*") if ".cache" not in p.parts
        )
    except OSError:
        return False


def benchmarks(task_data: Mapping[str, Any]) -> list[str]:
    """The known benchmarks the run's units score."""
    value = (task_data.get("accuracy") or {}).get("benchmark") or "gsm8k"
    names = value.values() if isinstance(value, Mapping) else [value]
    return sorted({str(n).lower() for n in names if str(n).lower() in KNOWN})


def stage(
    task_data: Mapping[str, Any],
    workspace: Path,
    benchmark: str,
    *,
    python: str | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Stage one benchmark's dataset; its status entry."""
    accuracy = task_data.get("accuracy") or {}
    repo = KNOWN[benchmark]
    given = accuracy.get("dataset_path")
    if given and Path(str(given)).exists():
        return {"state": READY, "path": str(given), "source": "accuracy.dataset_path"}
    env = (task_data.get("environment") or {}).get("env") or {}
    root = env.get("LLM_MODELS_ROOT") or os.environ.get("LLM_MODELS_ROOT")
    if root and _holds_files(Path(root) / DIRNAME / repo):
        return {"state": READY, "path": str(Path(root) / DIRNAME / repo), "source": "models root"}
    target = Path(workspace) / DIRNAME / repo
    if _holds_files(target):
        return {"state": READY, "path": str(target), "source": "staged before"}
    target.mkdir(parents=True, exist_ok=True)
    last = ""
    for interpreter in dict.fromkeys(p for p in (python, sys.executable) if p):
        try:
            done = run(
                [interpreter, "-c", _DOWNLOAD, repo, str(target)],
                capture_output=True,
                text=True,
                timeout=WAIT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last = str(exc)
            continue
        last = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or [""]
        last = last[0]
        if done.returncode == 0 and _holds_files(target):
            return {"state": READY, "path": str(target), "source": f"downloaded {repo}"}
    return {"state": FAILED, "path": "", "detail": f"could not stage {repo}: {last}"[:400]}


def start(
    task_data: Mapping[str, Any],
    workspace: Path,
    *,
    python: str | None = None,
    note: Callable[[str], None] = lambda said: None,
) -> threading.Thread | None:
    """Stage every benchmark's dataset in a background thread; ``None`` with nothing to stage."""
    names = benchmarks(task_data)
    if not names:
        return None
    for name in names:
        _set(workspace, name, state=STAGING)

    def work() -> None:
        for name in names:
            try:
                entry = stage(task_data, workspace, name, python=python)
            except Exception as exc:  # a dataset is never the run's failure
                entry = {"state": FAILED, "path": "", "detail": f"{type(exc).__name__}: {exc}"}
            _set(workspace, name, **entry)
            note(
                f"{name}: {entry['path']} ({entry.get('source')})"
                if entry["state"] == READY
                else f"{name}: {entry.get('detail')}; its gates run as they would without it"
            )

    thread = threading.Thread(target=work, name="mv2-datasets", daemon=True)
    thread.start()
    return thread


def wait(workspace: Path, benchmark: str, *, timeout: float = WAIT_S, poll: float = 5.0) -> str:
    """The staged dataset's path once it is ready; "" if it failed, timed out or was never asked."""
    deadline = time.monotonic() + timeout
    while True:
        entry = status(workspace).get(benchmark)
        if not entry or entry.get("state") == FAILED:
            return ""
        if entry.get("state") == READY:
            return str(entry.get("path") or "")
        if time.monotonic() >= deadline:
            return ""
        time.sleep(poll)


def models_root(workspace: Path, root: str, benchmark: str, path: str) -> str:
    """A models root holding ``root``'s entries and the staged dataset; ``root`` if it has it.

    The tree's own tests read ``$LLM_MODELS_ROOT/datasets/<repo>``: where the cluster's root
    lacks the dataset, a root of links to its entries, with the staged copy under
    ``datasets/``, stands in for it.
    """
    repo = KNOWN[benchmark]
    if not root or not path or _holds_files(Path(root) / DIRNAME / repo):
        return root
    overlay = Path(workspace) / DIRNAME / "models_root"
    try:
        overlay.mkdir(parents=True, exist_ok=True)
        for entry in Path(root).iterdir():
            link = overlay / entry.name
            if entry.name != DIRNAME and not link.exists():
                link.symlink_to(entry)
        datasets = overlay / DIRNAME
        datasets.mkdir(exist_ok=True)
        original = Path(root) / DIRNAME
        if original.is_dir():
            for entry in original.iterdir():
                link = datasets / entry.name
                if entry.name != repo.split("/")[0] and not link.exists():
                    link.symlink_to(entry)
        owner = datasets / repo.split("/")[0]
        owner.mkdir(exist_ok=True)
        link = owner / repo.split("/")[1]
        if not link.exists():
            link.symlink_to(Path(path))
    except OSError:
        return root
    return str(overlay)


__all__ = ["DIRNAME", "KNOWN", "benchmarks", "models_root", "stage", "start", "status", "wait"]
