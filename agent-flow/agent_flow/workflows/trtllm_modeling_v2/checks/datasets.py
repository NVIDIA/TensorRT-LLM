# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a benchmark's dataset is on this host.

Every accuracy check runs offline (``HF_DATASETS_OFFLINE=1``), so the dataset must be on
disk before the engine loads. :func:`find` looks where it can be: the benchmark's stated
``dataset_path``, the copy the tree's own tests read under ``$LLM_MODELS_ROOT/datasets``,
the HuggingFace hub cache, the datasets an earlier evaluation left prepared. A known
benchmark whose dataset is nowhere is a host problem, said before a load is spent on it
and never a verdict about the target. Nothing is downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

#: Benchmark -> its dataset's repository, as the tree's tests and the hub name it.
KNOWN: dict[str, str] = {"gsm8k": "openai/gsm8k"}


def _holds_files(path: Path) -> bool:
    try:
        return path.is_dir() and any(p.is_file() or p.is_symlink() for p in path.rglob("*"))
    except OSError:
        return False


def find(
    benchmark: str, stated: str | None = None, env: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """``(path, said)``: the dataset's directory and how it was found.

    ``("", "")`` for a benchmark this module does not know: the evaluator looks it up
    itself. ``("", why)`` when a known or a stated one is not on this host.
    """
    if stated:
        if Path(stated).exists():
            return str(stated), "the benchmark's `dataset_path`"
        return "", f"`dataset_path` {stated} does not exist on this host"
    repo = KNOWN.get(benchmark)
    if repo is None:
        return "", ""
    env = {**os.environ, **(env or {})}
    if env.get("LLM_MODELS_ROOT") and _holds_files(
        Path(env["LLM_MODELS_ROOT"]) / "datasets" / repo
    ):
        return str(Path(env["LLM_MODELS_ROOT"]) / "datasets" / repo), "the models root"
    home = env.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    hubs = [hub for hub in (env.get("HF_HUB_CACHE"), str(Path(home) / "hub")) if hub]
    for hub in hubs:
        snapshots = Path(hub) / f"datasets--{repo.replace('/', '--')}" / "snapshots"
        for snapshot in sorted(snapshots.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if _holds_files(snapshot):
                return str(snapshot), "the HuggingFace cache"
    # What an earlier evaluation left prepared is the evaluator's own to find, offline too.
    prepared = Path(env.get("HF_DATASETS_CACHE") or Path(home) / "datasets")
    if _holds_files(prepared / repo.replace("/", "___").lower()):
        return "", ""
    return "", (
        f"the dataset of `{benchmark}` ({repo}) is not on this host, and accuracy runs "
        f"offline: put it under $LLM_MODELS_ROOT/datasets/{repo}, or state `dataset_path`"
    )
