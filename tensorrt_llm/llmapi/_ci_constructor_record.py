# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local constructor snapshots for instrumented integration tests."""

import json
import os
from pathlib import Path
from uuid import uuid4

from tensorrt_llm.usage.llmapi_config import collect_llm_api_config_payloads


def _model_name(model: str | Path) -> str | None:
    text = os.fspath(model)
    if "://" in text or "?" in text or "#" in text:
        return None
    path = Path(text)
    root = os.environ.get("LLM_MODELS_ROOT")
    if path.is_absolute():
        if root and path.is_relative_to(root):
            return str(path.relative_to(root))
        return path.name or None
    return str(path)


def write_constructor_record(context: str, model: str | Path, args) -> None:
    directory, stage, nodeid = json.loads(context)
    config, metadata = collect_llm_api_config_payloads(args)
    event_id = uuid4().hex
    record = {
        "version": 1,
        "event_id": event_id,
        "stage": stage,
        "nodeid": nodeid,
        "phase": os.environ.get("PYTEST_CURRENT_TEST", "").rpartition(" (")[2].rstrip(")"),
        "model": _model_name(model),
        "config": json.loads(config),
        "capture": json.loads(metadata),
        "configured": {
            key: getattr(args, key) is not None
            for key in ("cuda_graph_config", "speculative_config", "torch_compile_config")
        },
    }
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / (event_id + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(target / (event_id + ".json"))
