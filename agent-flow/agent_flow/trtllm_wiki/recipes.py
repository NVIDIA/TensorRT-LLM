# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving configurations the checkout already ships, indexed by model, GPU and workload.

Three sources, one record shape:

* ``examples/configs/database/lookup.yaml``: pareto-optimized ``trtllm-serve``
  configs per model, GPU, ISL/OSL and concurrency;
* ``tests/scripts/perf-sanity/aggregated/*.yaml``: CI perf configs, one record
  per (server config, client config);
* ``tests/scripts/perf-sanity/disaggregated/*.yaml``: CI disaggregated configs
  with their context and generation workers.

A record keeps the path of the YAML (and the server config's name) so an agent
reads the exact file; the indexed fields are what a nearest-neighbour search
needs: architecture or model name, GPU, GPU count, parallel layout, ISL, OSL,
concurrency, speculative decoding and whether attention DP is on.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

LOOKUP_REL = "examples/configs/database/lookup.yaml"
PERF_SANITY_REL = "tests/scripts/perf-sanity"


def _load(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None


def _layout(config: dict[str, Any]) -> dict[str, Any]:
    spec = config.get("speculative_config") or {}
    return {
        "tp": config.get("tensor_parallel_size"),
        "ep": config.get("moe_expert_parallel_size"),
        "attention_dp": bool(config.get("enable_attention_dp")),
        "speculative": str(spec.get("decoding_type") or "") if isinstance(spec, dict) else "",
        "draft_len": spec.get("max_draft_len") if isinstance(spec, dict) else None,
        "max_batch_size": config.get("max_batch_size"),
    }


def _lookup(repo: Path) -> list[dict[str, Any]]:
    entries = _load(repo / LOOKUP_REL) or []
    records = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        config = _load(repo / str(entry.get("config_path") or "")) or {}
        records.append(
            {
                "source": "config-database",
                "path": str(entry.get("config_path") or ""),
                "name": "",
                "model": str(entry.get("model") or ""),
                "arch": str(entry.get("arch") or ""),
                "gpu": str(entry.get("gpu") or ""),
                "gpus": entry.get("num_gpus"),
                "isl": entry.get("isl"),
                "osl": entry.get("osl"),
                "concurrency": entry.get("concurrency"),
                "benchmark": "",
                **_layout(config if isinstance(config, dict) else {}),
            }
        )
    return records


def _aggregated(repo: Path, path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = data.get("metadata") or {}
    gpus = [str(g) for g in metadata.get("supported_gpus") or []]
    records = []
    for server in data.get("server_configs") or []:
        if not isinstance(server, dict):
            continue
        for client in server.get("client_configs") or [{}]:
            records.append(
                {
                    "source": "perf-sanity",
                    "path": path.relative_to(repo).as_posix(),
                    "name": str(server.get("name") or ""),
                    "model": str(server.get("model_name") or metadata.get("model_name") or ""),
                    "arch": "",
                    "gpu": ",".join(gpus),
                    "gpus": server.get("gpus"),
                    "isl": client.get("isl"),
                    "osl": client.get("osl"),
                    "concurrency": client.get("concurrency"),
                    "benchmark": str(metadata.get("benchmark_type") or ""),
                    **_layout(server),
                }
            )
    return records


def _disaggregated(repo: Path, path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = data.get("metadata") or {}
    benchmark = data.get("benchmark") or {}
    workers = data.get("worker_config") or {}
    gen = workers.get("gen") if isinstance(workers.get("gen"), dict) else {}
    ctx = workers.get("ctx") if isinstance(workers.get("ctx"), dict) else {}
    concurrency = str(benchmark.get("concurrency_list") or "").split()
    return [
        {
            "source": "perf-sanity-disagg",
            "path": path.relative_to(repo).as_posix(),
            "name": "",
            "model": str(metadata.get("model_name") or ""),
            "arch": "",
            "gpu": ",".join(str(g) for g in metadata.get("supported_gpus") or []),
            "gpus": None,
            "isl": benchmark.get("input_length"),
            "osl": benchmark.get("output_length"),
            "concurrency": int(c) if c.isdigit() else None,
            "benchmark": str(metadata.get("benchmark_type") or ""),
            **_layout(gen),
            "ctx": _layout(ctx),
        }
        for c in concurrency or [""]
    ]


def extract_recipes(repo: Path) -> list[dict[str, Any]]:
    repo = Path(repo)
    records = _lookup(repo)
    for path in sorted((repo / PERF_SANITY_REL).glob("*/*.yaml")):
        data = _load(path)
        if not isinstance(data, dict):
            continue
        if "server_configs" in data:
            records.extend(_aggregated(repo, path, data))
        elif "worker_config" in data:
            records.extend(_disaggregated(repo, path, data))
    return records


def _distance(record: dict[str, Any], key: str, wanted: Any) -> float:
    have = record.get(key)
    if wanted is None:
        return 0.0
    if not isinstance(have, (int, float)) or have <= 0 or wanted <= 0:
        return 2.0
    return abs(math.log2(have) - math.log2(wanted))


def rank_recipes(
    records: list[dict[str, Any]],
    *,
    text: str = "",
    gpu: str = "",
    gpus: int | None = None,
    isl: int | None = None,
    osl: int | None = None,
    concurrency: int | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """The ``limit`` records closest to the request: name/GPU words must match, numbers rank."""
    words = [w.lower() for w in text.replace("/", " ").split() if w]
    chosen = []
    for record in records:
        haystack = f"{record['model']} {record['arch']} {record['path']} {record['name']}".lower()
        if any(w not in haystack for w in words):
            continue
        if gpu and gpu.lower() not in record["gpu"].lower():
            continue
        score = (
            _distance(record, "gpus", gpus)
            + _distance(record, "isl", isl)
            + _distance(record, "osl", osl)
            + _distance(record, "concurrency", concurrency)
        )
        chosen.append((score, record["path"], record["name"], record))
    chosen.sort(key=lambda item: item[:3])
    return [{**record, "distance": round(score, 2)} for score, _, _, record in chosen[:limit]]
