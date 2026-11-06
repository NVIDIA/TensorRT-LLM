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
"""Route snapshot of every trtllm-serve app configuration.

A route record is one string: ``"<METHODS> <path> <class>"``:

- ``<METHODS>``: the route's methods, sorted and comma-joined (``GET,HEAD`` for
  FastAPI's documentation routes); ``MOUNT`` for a ``Mount``.
- ``<path>``: the effective path from :func:`effective_routes`.
- ``<class>``: the route's bare class name, on purpose: moving the route class
  to another module must not change the snapshot.
- When two routes in one configuration share ``<METHODS> <path>``, both records
  get `` #<route.name>`` appended so their order can be compared. Names are not
  recorded otherwise, so moving a handler into a router does not change the
  snapshot.

Regenerate the reference with::

    PYTHONPATH=tests python3 -m test_common.serve.routes --write

run from the repository root with the checkout's build importable.
"""

import argparse
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import yaml
from fastapi import FastAPI

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from test_common.serve.app_factory import (
    AggConfig,
    CoordinatorConfig,
    DisaggConfig,
    isolated_serve_env,
)

__all__ = [
    "CONFIGS",
    "VISUAL_GEN_CONFIGS",
    "SnapshotConfig",
    "compare",
    "effective_routes",
    "read_reference",
    "reference_path",
    "route_record",
    "snapshot_config",
    "snapshot_routes",
    "write_reference",
]

REPO_ROOT = Path(__file__).resolve().parents[3]

REFERENCE_HEADER = """\
# Routes of every trtllm-serve app configuration. Generated, do not edit by hand:
#   PYTHONPATH=tests python3 -m test_common.serve.routes --write
# Each route is "<METHODS> <path> <route class>", in registration order. "#<name>" is
# added only when two routes share methods and path. The checker compares the order
# only for routes that can match the same request. Changing this file changes the
# HTTP surface of trtllm-serve; the API review committee approves it.
"""


def reference_path(repo_root: Path = REPO_ROOT) -> Path:
    return (
        repo_root
        / "tests"
        / "unittest"
        / "api_stability"
        / "references"
        / "trtllm_serve_routes.yaml"
    )


# ---------------------------------------------------------------------------
# Walking routes
# ---------------------------------------------------------------------------


def effective_routes(app_or_routes, prefix: str = "") -> list:
    """(effective path, leaf route) in match order, on FastAPI 0.136.x and 0.139.x.

    FastAPI 0.139 keeps one ``_IncludedRouter`` object per ``include_router``
    call in ``app.routes``; it has no ``path`` attribute, so reading ``.path``
    off ``app.routes`` raises once any router is included. This walks into such
    objects and applies their include prefix. On 0.136.x included routes are
    already flat, so the recursion never runs and the result equals
    ``app.routes``. The duck-typed check below uses two attribute names of a
    private FastAPI class; if a later FastAPI renames them,
    ``test_effective_routes_flattens_included_routers`` fails first, naming this
    function.
    """
    routes = app_or_routes.routes if isinstance(app_or_routes, FastAPI) else app_or_routes
    out = []
    for route in routes:
        included = getattr(route, "original_router", None)
        context = getattr(route, "include_context", None)
        if included is not None and context is not None:  # FastAPI 0.139 _IncludedRouter
            out.extend(effective_routes(included.routes, prefix + context.prefix))
            continue
        # Mounts keep their path and are not descended into.
        out.append((prefix + getattr(route, "path", ""), route))
    return out


def route_record(path: str, route) -> str:
    """``"<METHODS> <path> <class>"`` for one route, without the ``#<name>`` suffix."""
    methods = getattr(route, "methods", None)
    methods_text = ",".join(sorted(methods)) if methods else "MOUNT"
    return f"{methods_text} {path} {type(route).__name__}"


def snapshot_routes(app: FastAPI) -> List[str]:
    """Route records of ``app`` in registration order, disambiguated by name."""
    pairs = effective_routes(app)
    records = [route_record(path, route) for path, route in pairs]
    key_counts = Counter(record.rsplit(" ", 1)[0] for record in records)
    out = []
    for record, (path, route) in zip(records, pairs):
        methods_and_path = record.rsplit(" ", 1)[0]
        if key_counts[methods_and_path] > 1:
            record = f"{record} #{getattr(route, 'name', '')}"
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotConfig:
    name: str
    kind: Literal["openai_server", "disagg_server", "coordinator_server"]
    build: Union[AggConfig, DisaggConfig, CoordinatorConfig]


CONFIGS: Tuple[SnapshotConfig, ...] = (
    SnapshotConfig("llm", "openai_server", AggConfig()),
    SnapshotConfig("llm_context", "openai_server", AggConfig(role=ServerRole.CONTEXT)),
    SnapshotConfig("llm_generation", "openai_server", AggConfig(role=ServerRole.GENERATION)),
    SnapshotConfig("llm_gpt_oss", "openai_server", AggConfig(model_type="gpt_oss")),
    SnapshotConfig(
        "llm_gpt_oss_harmony_disabled",
        "openai_server",
        AggConfig(model_type="gpt_oss", env=(("DISABLE_HARMONY_ADAPTER", "1"),)),
    ),
    SnapshotConfig("llm_governor", "openai_server", AggConfig(governor_queue=True)),
    SnapshotConfig("llm_rl_control", "openai_server", AggConfig(rl_control=True)),
    SnapshotConfig("llm_runtime_control", "openai_server", AggConfig(runtime_control=True)),
    SnapshotConfig(
        "llm_rl_and_runtime_control",
        "openai_server",
        AggConfig(rl_control=True, runtime_control=True),
    ),
    SnapshotConfig("llm_perf_metrics", "openai_server", AggConfig(return_perf_metrics=True)),
    SnapshotConfig(
        "llm_perf_metrics_enable_metrics_env",
        "openai_server",
        AggConfig(return_perf_metrics=True, env=(("ENABLE_METRICS", "true"),)),
    ),
    SnapshotConfig(
        "llm_perf_metrics_output_dir_only",
        "openai_server",
        AggConfig(perf_metrics_output_dir=True),
    ),
    SnapshotConfig(
        "llm_render_endpoints",
        "openai_server",
        AggConfig(env=(("TRTLLM_ENABLE_RENDER_ENDPOINTS", "1"),)),
    ),
    # An attached frontend of a multi-frontend server registers the launcher
    # forwarding routes before the local table; same method and path, so
    # registration order decides which handler serves them.
    SnapshotConfig(
        "llm_multi_frontend_attached",
        "openai_server",
        AggConfig(multi_frontend_attached=True),
    ),
    SnapshotConfig("mm_encoder", "openai_server", AggConfig(role=ServerRole.MM_ENCODER)),
    SnapshotConfig(
        "mm_encoder_rl_control",
        "openai_server",
        AggConfig(role=ServerRole.MM_ENCODER, rl_control=True),
    ),
    SnapshotConfig("embedding", "openai_server", AggConfig(role=ServerRole.EMBEDDING)),
    SnapshotConfig(
        "embedding_runtime_control",
        "openai_server",
        AggConfig(role=ServerRole.EMBEDDING, runtime_control=True),
    ),
    SnapshotConfig("disagg_local", "disagg_server", DisaggConfig()),
    SnapshotConfig(
        "disagg_local_perf_metrics", "disagg_server", DisaggConfig(return_perf_metrics=True)
    ),
    SnapshotConfig(
        "disagg_local_http_cluster_storage",
        "disagg_server",
        DisaggConfig(http_cluster_storage=True),
    ),
    SnapshotConfig("disagg_fleet", "disagg_server", DisaggConfig(fleet=True)),
    SnapshotConfig(
        "disagg_fleet_perf_metrics",
        "disagg_server",
        DisaggConfig(fleet=True, return_perf_metrics=True),
    ),
    SnapshotConfig(
        "disagg_fleet_http_cluster_storage",
        "disagg_server",
        DisaggConfig(fleet=True, http_cluster_storage=True),
    ),
    SnapshotConfig("coordinator", "coordinator_server", CoordinatorConfig()),
    SnapshotConfig(
        "coordinator_http_cluster_storage",
        "coordinator_server",
        CoordinatorConfig(http_cluster_storage=True),
    ),
)

VISUAL_GEN_CONFIGS: Tuple[SnapshotConfig, ...] = (
    SnapshotConfig(
        "visual_gen",
        "openai_server",
        AggConfig(role=ServerRole.VISUAL_GEN, model_name="fake-visual-gen"),
    ),
)


def snapshot_config(config: SnapshotConfig, tmp_dir: Path) -> Dict:
    """``{"kind": ..., "routes": [...]}`` or ``{"kind": ..., "error": "<Type>: <message>"}``."""
    with isolated_serve_env(tmp_dir) as env:
        try:
            if config.kind == "openai_server":
                app = env.build_openai_server(config.build).app
            elif config.kind == "disagg_server":
                app = env.build_disagg_server(config.build).app
            else:
                app = env.build_coordinator_server(config.build).app
        except ValueError as error:
            return {"kind": config.kind, "error": f"{type(error).__name__}: {error}"}
        return {"kind": config.kind, "routes": snapshot_routes(app)}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _parse_record(record: str) -> Tuple[str, str, str, str]:
    """(methods, path, class, name) of one record; name is "" when absent."""
    parts = record.split(" ")
    methods, path, cls = parts[0], parts[1], parts[2]
    name = parts[3][1:] if len(parts) > 3 and parts[3].startswith("#") else ""
    return methods, path, cls, name


def _literal_prefix(path: str) -> str:
    """The path up to its first ``{``."""
    brace = path.find("{")
    return path if brace < 0 else path[:brace]


def _paths_can_match_same_request(path_a: str, path_b: str) -> bool:
    segments_a = path_a.split("/")
    segments_b = path_b.split("/")
    for i in range(max(len(segments_a), len(segments_b))):
        a = segments_a[i] if i < len(segments_a) else None
        b = segments_b[i] if i < len(segments_b) else None
        # A {name:path} parameter matches the rest of the path.
        if a is not None and "{" in a and a.rstrip("}").endswith(":path"):
            return True
        if b is not None and "{" in b and b.rstrip("}").endswith(":path"):
            return True
        if a is None or b is None:
            # Different segment counts with no path parameter: no overlap.
            return False
        if "{" in a or "{" in b:
            continue  # a parameter matches any literal
        if a != b:
            return False
    return True


def _records_conflict(record_a: str, record_b: str) -> bool:
    """Whether two records can match the same request (over-approximated).

    Two records conflict when (a) their methods intersect or either is a
    mount, and (b) their path templates can match one request path.
    Over-approximating conflicts only locks more pairs, which is safe.
    """
    methods_a, path_a, _, _ = _parse_record(record_a)
    methods_b, path_b, _, _ = _parse_record(record_b)
    a_is_mount = methods_a == "MOUNT"
    b_is_mount = methods_b == "MOUNT"
    if not a_is_mount and not b_is_mount:
        if not set(methods_a.split(",")) & set(methods_b.split(",")):
            return False
        return _paths_can_match_same_request(path_a, path_b)
    # A mount at P conflicts with every route whose literal prefix starts
    # with P, or that P starts with.
    if a_is_mount and b_is_mount:
        return path_a.startswith(path_b) or path_b.startswith(path_a)
    mount_path = path_a if a_is_mount else path_b
    other_prefix = _literal_prefix(path_b if a_is_mount else path_a)
    return other_prefix.startswith(mount_path) or mount_path.startswith(other_prefix)


def compare(expected: Sequence[str], actual: Sequence[str]) -> List[str]:
    """Human-readable differences between the reference and the live records."""
    differences: List[str] = []
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    removed = list((expected_counts - actual_counts).elements())
    added = list((actual_counts - expected_counts).elements())

    # A removed and an added record with the same methods and path but another
    # class are one "route class changed" line.
    def _methods_path(record: str) -> str:
        methods, path, _, _ = _parse_record(record)
        return f"{methods} {path}"

    remaining_added = list(added)
    class_changes = []
    for removed_record in list(removed):
        for added_record in remaining_added:
            if _methods_path(added_record) == _methods_path(removed_record):
                old_cls = _parse_record(removed_record)[2]
                new_cls = _parse_record(added_record)[2]
                why = (
                    " (a plain APIRouter breaks msgpack bodies from the disaggregated frontend)"
                    if old_cls == "_MsgspecRoute"
                    else ""
                )
                class_changes.append(
                    f"route class changed: {_methods_path(removed_record)}: "
                    f"{old_cls} -> {new_cls}{why}"
                )
                removed.remove(removed_record)
                remaining_added.remove(added_record)
                break
    for record in remaining_added:
        differences.append(f"added:   {record}")
    for record in removed:
        differences.append(f"removed: {record}")
    differences.extend(class_changes)

    # Conflict-relevant order: for every conflicting pair a-before-b in
    # expected, a must come before b in actual, when both are present. For
    # repeated identical records, the k-th occurrence pairs with the k-th.
    actual_positions: Dict[str, List[int]] = {}
    for position, record in enumerate(actual):
        actual_positions.setdefault(record, []).append(position)
    expected_occurrence: List[Optional[int]] = []
    seen: Counter = Counter()
    for record in expected:
        positions = actual_positions.get(record, [])
        occurrence = seen[record]
        seen[record] += 1
        expected_occurrence.append(positions[occurrence] if occurrence < len(positions) else None)
    for i, record_a in enumerate(expected):
        position_a = expected_occurrence[i]
        if position_a is None:
            continue
        for j in range(i + 1, len(expected)):
            record_b = expected[j]
            position_b = expected_occurrence[j]
            if position_b is None or record_a == record_b:
                continue
            if _records_conflict(record_a, record_b) and position_a > position_b:
                differences.append(
                    f"order changed: {record_a} must come before {record_b} "
                    "(FastAPI matches in registration order)"
                )
    return differences


def format_failure(name: str, differences: Sequence[str]) -> str:
    lines = "\n".join(f"  {difference}" for difference in differences)
    return (
        f"Route snapshot of configuration '{name}' differs from "
        "tests/unittest/api_stability/references/trtllm_serve_routes.yaml:\n"
        f"{lines}\n"
        "If the change is intended, regenerate with "
        "`PYTHONPATH=tests python3 -m test_common.serve.routes --write`, commit the YAML, "
        "and request review from @NVIDIA/trt-llm-noncommitted-api-review-committee "
        "(owner of tests/unittest/api_stability/). Do not add routes or methods to "
        "OpenAIServer; see tensorrt_llm/serve/AGENTS.md."
    )


# ---------------------------------------------------------------------------
# Reference file
# ---------------------------------------------------------------------------


def build_reference_data(tmp_root: Path) -> Dict:
    configs: Dict[str, Dict] = {}
    for index, config in enumerate(CONFIGS + VISUAL_GEN_CONFIGS):
        configs[config.name] = snapshot_config(config, tmp_root / f"cfg{index}")
    return {"version": 1, "configs": configs}


def write_reference(data: Dict, path: Optional[Path] = None) -> Path:
    path = path or reference_path()
    body = yaml.safe_dump(data, sort_keys=False, width=4096)
    path.write_text(REFERENCE_HEADER + body)
    return path


def read_reference(path: Optional[Path] = None) -> Dict:
    path = path or reference_path()
    return yaml.safe_load(path.read_text())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m test_common.serve.routes",
        description="Check, rewrite or print the trtllm-serve route reference.",
    )
    parser.add_argument(
        "--write", action="store_true", help="rebuild every configuration and rewrite the reference"
    )
    parser.add_argument(
        "--print",
        dest="print_name",
        metavar="CONFIG",
        help="print one configuration's live records",
    )
    args = parser.parse_args(argv)

    by_name = {config.name: config for config in CONFIGS + VISUAL_GEN_CONFIGS}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        if args.print_name:
            if args.print_name not in by_name:
                print(f"unknown configuration '{args.print_name}'; one of: {sorted(by_name)}")
                return 2
            snapshot = snapshot_config(by_name[args.print_name], tmp_root / "print")
            for line in snapshot.get("routes", [snapshot.get("error", "")]):
                print(line)
            return 0
        data = build_reference_data(tmp_root)
        if args.write:
            path = write_reference(data)
            print(f"wrote {path}")
            return 0
        reference = read_reference()
        exit_code = 0
        for name, live in data["configs"].items():
            recorded = reference["configs"].get(name)
            if recorded is None:
                print(f"configuration '{name}' missing from the reference")
                exit_code = 1
                continue
            if recorded.get("error") != live.get("error"):
                print(
                    f"configuration '{name}': error changed: "
                    f"{recorded.get('error')!r} -> {live.get('error')!r}"
                )
                exit_code = 1
            differences = compare(recorded.get("routes", []), live.get("routes", []))
            if differences:
                print(format_failure(name, differences))
                exit_code = 1
        stale = set(reference["configs"]) - set(data["configs"])
        if stale:
            print(f"configurations only in the reference: {sorted(stale)}")
            exit_code = 1
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
