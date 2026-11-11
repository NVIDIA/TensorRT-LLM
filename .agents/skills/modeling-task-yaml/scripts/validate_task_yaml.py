#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate a task.yaml with agent-flow's own modeling-bringup schema.

Loads agent-flow's task_schema.py straight from its file: importing it
through the agent_flow package also imports claude_agent_sdk, which is not
installed everywhere. task_schema.py itself needs only PyYAML.

Usage:
    python3 validate_task_yaml.py /path/to/task.yaml [--agent-flow DIR]

Without --agent-flow, the schema is looked up under the task's
trtllm_repo_path, then under this checkout.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

SCHEMA_RELPATH = Path("agent_flow/workflows/modeling_bringup/task_schema.py")
THIS_CHECKOUT = Path(__file__).resolve().parents[4]


def _candidate_agent_flow_dirs(task_path: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    candidates = []
    try:
        data = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    repo = data.get("trtllm_repo_path") if isinstance(data, dict) else None
    if isinstance(repo, str) and repo.strip():
        candidates.append(Path(repo) / "agent-flow")
    candidates.append(THIS_CHECKOUT / "agent-flow")
    return candidates


def _load_schema_module(agent_flow_dir: Path):
    schema_file = agent_flow_dir / SCHEMA_RELPATH
    spec = importlib.util.spec_from_file_location("task_schema", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", type=Path, help="task.yaml to validate")
    parser.add_argument("--agent-flow", type=Path, help="agent-flow directory holding the schema")
    args = parser.parse_args()

    candidates = _candidate_agent_flow_dirs(args.task, args.agent_flow)
    agent_flow_dir = next((d for d in candidates if (d / SCHEMA_RELPATH).is_file()), None)
    if agent_flow_dir is None:
        tried = ", ".join(str(d / SCHEMA_RELPATH) for d in candidates)
        print(f"error: task_schema.py not found; tried: {tried}", file=sys.stderr)
        return 2

    schema = _load_schema_module(agent_flow_dir)
    try:
        schema.load_and_validate_task_yaml(args.task)
    except schema.TaskSchemaError as exc:
        print(f"INVALID\n{exc}")
        return 1
    print(f"VALID ({agent_flow_dir / SCHEMA_RELPATH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
