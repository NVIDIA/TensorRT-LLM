# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The substrate probe: what every gate's launch needs of the host, with no model loaded.

Launched the way a gate is (one rank bare; more under ``mpirun`` and
``trtllm-llmapi-launch``, whose leader runs this script while the other ranks
wait for work), it checks what failed runs on hosts that could not have served
any model: the build imports, the leader sees the world the launch made, a
device takes a tensor, the cache and scratch directories take a file, the
scratch path leaves room for a socket, and the benchmark's dataset loads
offline. It prints one ``MV2FLOW_SUBSTRATE <json>`` line, ``{"facts": {...},
"problems": [...]}``, and exits 1 when a problem stands.

    python env_probe.py <spec.json>

The spec: ``world_size`` (the ranks the launch made), ``max_tmpdir_len``,
``dataset`` (``{"path": ...}`` or ``{"hf": [name, config, split]}``; optional).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

#: The variables naming directories a gate writes into: each must take a file.
CACHE_VARIABLES = (
    "TMPDIR",
    "TRITON_CACHE_DIR",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "XDG_CACHE_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
)


def _writable(directory: Path) -> str:
    """Why ``directory`` takes no file, or ``""``."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory):
            pass
    except OSError as exc:
        return f"{directory}: {exc.strerror or exc}"
    return ""


def probe(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    facts: dict[str, Any] = {}
    problems: list[str] = []
    try:
        import tensorrt_llm

        facts["tensorrt_llm"] = (
            f"{tensorrt_llm.__file__} {getattr(tensorrt_llm, '__version__', '')}"
        )
    except Exception as exc:  # whatever the import raises is the finding
        problems.append(f"import tensorrt_llm: {type(exc).__name__}: {exc}")
    try:
        import torch

        facts["devices"] = torch.cuda.device_count()
        torch.zeros(1, device="cuda")
    except Exception as exc:  # no device, a driver mismatch, a CUDA error
        problems.append(f"CUDA: {type(exc).__name__}: {exc}")
    wanted = int(spec.get("world_size") or 1)
    if wanted > 1:
        try:
            from mpi4py import MPI

            seen = MPI.COMM_WORLD.Get_size()
            facts["mpi_world"] = seen
            if seen != wanted:
                problems.append(
                    f"MPI: the leader sees a world of {seen}, the launch made {wanted}: the "
                    "engine would try to spawn the rest itself (MPI_ERR_SPAWN)"
                )
        except Exception as exc:  # mpi4py missing, or MPI unable to start
            problems.append(f"MPI: {type(exc).__name__}: {exc}")
    homes = [Path.home() / ".cache"]
    homes += [Path(os.environ[name]) for name in CACHE_VARIABLES if os.environ.get(name)]
    problems += [why for why in map(_writable, dict.fromkeys(homes)) if why]
    scratch = tempfile.gettempdir()
    limit = int(spec.get("max_tmpdir_len") or 0)
    if limit and len(scratch) > limit:
        problems.append(
            f"scratch {scratch} is {len(scratch)} characters, past {limit}: a socket under it "
            "cannot be bound"
        )
    dataset = spec.get("dataset") or {}
    if dataset.get("path") and not Path(str(dataset["path"])).exists():
        problems.append(f"dataset: {dataset['path']} does not exist")
    elif dataset.get("hf"):
        try:
            from datasets import load_dataset

            name, config, split = (list(dataset["hf"]) + [None, None])[:3]
            facts["dataset_rows"] = len(load_dataset(name, config, split=split))
        except Exception as exc:  # offline cache missing, or an unreadable one
            problems.append(f"dataset {dataset['hf']} offline: {type(exc).__name__}: {exc}")
    return facts, problems


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8")) if len(argv) > 1 else {}
    facts, problems = probe(spec)
    print("MV2FLOW_SUBSTRATE " + json.dumps({"facts": facts, "problems": problems}), flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
