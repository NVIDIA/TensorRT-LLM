# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export direct DSpark whole-step costs or check existing artifacts.

This standalone command uses the actual schema-v2 consumer. Only the explicit
complete-loop producer can export new costs; graph replay timings stay separate.
Run it by file path to avoid importing the native TensorRT-LLM package on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from types import ModuleType

__all__ = [
    "check_cost_bytes",
    "main",
    "observed_engine_fingerprint",
    "recorded_source_origin",
    "source_origin_from_args",
    "source_origin_input_metadata",
    "runtime_cost_bytes",
]

WHOLE_STEP_SCOPE = "original-executor-loop-hostStepTimeMS"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_origin_input_metadata(
    recorded_origin: Path | None, recorded_origin_sha256: str | None
) -> dict[str, str | bool]:
    """Label recorded provenance separately from a fresh Git observation."""
    if (recorded_origin is None) != (recorded_origin_sha256 is None):
        raise ValueError("Recorded source origin requires both its path and SHA256")
    if recorded_origin is None:
        return {"method": "git", "recorded_input": False}
    if not isinstance(recorded_origin_sha256, str) or not re.fullmatch(
        "[0-9a-f]{64}", recorded_origin_sha256
    ):
        raise ValueError("Recorded source origin SHA256 must be explicit lowercase hex")
    return {
        "method": "recorded_provenance_input",
        "recorded_input": True,
        "input_sha256": recorded_origin_sha256,
        "fresh_git_diff_repeated": False,
    }


def source_origin_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Use one shared optional provenance input across the public drivers."""
    return recorded_source_origin(
        args.source_root,
        recorded_origin=getattr(args, "recorded_source_origin", None),
        recorded_origin_sha256=getattr(args, "recorded_source_origin_sha256", None),
    )


def recorded_source_origin(
    source_root: Path,
    *,
    recorded_origin: Path | None = None,
    recorded_origin_sha256: str | None = None,
) -> dict[str, str]:
    """Read Git provenance or an explicitly pinned record for this exact checkout.

    An optional record contains actual earlier HEAD/full-diff observation for
    the same immutable source root. It is recorded metadata, not proof of source
    freshness, GPU health, numerical accuracy or performance admission. Loaded
    module checks and caller-owned source byte validation remain independent.
    """
    source_origin_input_metadata(recorded_origin, recorded_origin_sha256)
    if recorded_origin is not None:
        if recorded_origin.is_symlink() or not recorded_origin.is_file():
            raise ValueError("Recorded source origin must be a regular non-symlink file")
        if recorded_origin.stat().st_size > 8192:
            raise ValueError("Recorded source origin must be a finite metadata record")
        data = recorded_origin.read_bytes()
        if hashlib.sha256(data).hexdigest() != recorded_origin_sha256:
            raise ValueError("Recorded source origin bytes differ from the supplied SHA256")
        utility = _source_module(
            "_dspark_recorded_origin_json", Path(__file__).with_name("verify_costs.py")
        )
        record = utility.parse_json(data)
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "source_root", "origin"}
            or record["schema"] != "trtllm-recorded-source-origin-v1"
            or record["source_root"] != str(source_root.resolve(strict=True))
            or not source_root.is_dir()
        ):
            raise ValueError("Recorded source origin belongs to a different checkout or schema")
        origin = record["origin"]
        if (
            not isinstance(origin, dict)
            or set(origin) != {"source_head", "source_diff_sha256"}
            or not isinstance(origin["source_head"], str)
            or not re.fullmatch("[0-9a-f]{40}", origin["source_head"])
            or not isinstance(origin["source_diff_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", origin["source_diff_sha256"])
        ):
            raise ValueError("Recorded source HEAD/full-diff digests are malformed")
        return origin
    head = (
        subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        .stdout.decode()
        .strip()
    )
    diff = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(source_root),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "HEAD",
            "--",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    return {"source_head": head, "source_diff_sha256": hashlib.sha256(diff).hexdigest()}


def observed_engine_fingerprint(
    identity: dict[str, object],
    *,
    source_root: Path,
    world_size: int,
    recorded_origin: Path | None = None,
    recorded_origin_sha256: str | None = None,
) -> dict[str, object]:
    """Record actual origin metadata, not a destination freshness certificate.

    Loaded module bytes must be those of the supplied public checkout. Native
    and runtime snapshots preserve observed library/configuration identity; no
    old artifact fingerprint is relabeled as current and no GPU health is implied.
    """
    topology = identity["topology"]
    if world_size > 1 and not topology["attention_dp"]:
        raise ValueError(
            "Schema-v2 global graph counts use attention-DP fanout; "
            "multi-rank attention-DP-off costs need a separate schema decision"
        )
    sources = identity["loaded_source_sha256"]
    for module, expected in sources.items():
        path = source_root / (module.replace(".", "/") + ".py")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Loaded public source differs from the recorded origin checkout")
    devices = identity["devices"]
    native = identity["rank_native_libraries"]
    if len(devices) != world_size or len(native) != world_size or not all(native):
        raise ValueError("Complete observed device/native-library rank coverage is required")
    names = {device["name"] for device in devices}
    if len(names) != 1:
        raise ValueError("A cost grid must record one observed GPU class")
    origin = recorded_source_origin(
        source_root, recorded_origin=recorded_origin, recorded_origin_sha256=recorded_origin_sha256
    )
    if identity.get("recorded_source_origin", origin) != origin:
        raise ValueError("Original recorded source changed during whole-step construction")
    graphs = sorted({cell["graph_batch_size"] for cell in identity["profile"]["cells"]})
    runtime = {
        "runtime": identity["runtime"],
        "rank_native_libraries": native,
        "loaded_source_sha256": sources,
        "runtime_configuration": identity["runtime_configuration"],
        "model_config_sha256": identity["model_config_sha256"],
        "profile": identity["profile"],
    }
    return {
        "global_graph_batch_sizes": [g * world_size for g in graphs],
        "gpu": next(iter(names)),
        "gpu_count": world_size,
        "gpu_snapshot_sha256": _canonical_sha256(devices),
        "max_draft_len": identity["profile"]["physical_k"],
        "rank_local_graph_batch_sizes": graphs,
        "runtime_snapshot": _canonical_sha256(runtime),
        **origin,
        "topology": json.dumps(topology, sort_keys=True, separators=(",", ":")),
    }


def _source_module(name: str, path: Path) -> ModuleType:
    """Load a fixed pure-Python dependency without native package initialization."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load cost tooling dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _whole_step_payload(
    data: bytes, summary: dict, *, minimum_predicted_gain: float
) -> dict[str, object]:
    """Map direct complete-loop cells without adding components or missing prices."""
    if summary.get("timing_scope") != WHOLE_STEP_SCOPE:
        raise ValueError(
            "Graph replay observations are not schema-v2 whole-step costs; "
            "use the original complete-loop whole-step producer, not relabeled timings"
        )
    identity = summary["identity"]
    fingerprint = identity["recorded_engine_fingerprint"]
    profile = identity["profile"]
    if profile["measurement_scope"] != "whole-step":
        raise ValueError("Whole-step export requires the original whole-step adapter")
    floors = summary["sample_floors"]
    if floors["per_generation"] < 12 or floors["pooled_per_shape"] < 24:
        raise ValueError("Original whole-step sample floors are 12 per generation and 24 pooled")
    tables, measurements, observed = {}, [], set()
    digest = hashlib.sha256(data).hexdigest()
    for cell in summary["cells"]:
        shape = cell["execution_shape"]
        g, v = shape["graph_batch_size"], shape["verifier_budget"]
        expected_layout = "uniform" if v == 0 else "packed_chain_rows"
        if (
            shape["workload"] != "dspark_chain"
            or shape["physical_k"] != profile["physical_k"]
            or shape["attention_layout"] != expected_layout
            or shape["captured_head"] != profile["record_confidence_head"]
            or (g, v) in observed
        ):
            raise ValueError(
                "Whole-step execution shape, K, layout or unique cell identity differs"
            )
        samples = cell["retained_rank_max_samples_ms"]
        if (
            len(samples) < floors["pooled_per_shape"]
            or cell["retained_count"] != len(samples)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value < float("inf")
                for value in samples
            )
            or cell["rank_max_median_ms"] != statistics.median(samples)
        ):
            raise ValueError("Whole-step median does not reproduce from original retained samples")
        observed.add((g, v))
        table = tables.setdefault(str(g), {"token_counts": [], "step_time_ms": []})
        table["token_counts"].append(v)
        table["step_time_ms"].append(cell["rank_max_median_ms"])
        measurements.append(
            {
                "rank_local_graph_batch_size": g,
                "rank_local_verifier_budget": v,
                "source_result_sha256": digest,
                "step_time_ms": cell["rank_max_median_ms"],
            }
        )
    declared = {(cell["graph_batch_size"], cell["verifier_budget"]) for cell in profile["cells"]}
    if observed != declared:
        raise ValueError(
            "Every declared native/compact whole-step cell requires its direct measurement"
        )
    for table in tables.values():
        ordered = sorted(zip(table["token_counts"], table["step_time_ms"]))
        table["token_counts"], table["step_time_ms"] = [list(values) for values in zip(*ordered)]
    return {
        "schema_version": 2,
        "cost_tables": tables,
        "engine_fingerprint": fingerprint,
        "engine_fingerprint_sha256": _canonical_sha256(fingerprint),
        "minimum_predicted_gain": minimum_predicted_gain,
        "measurements": sorted(
            measurements,
            key=lambda item: (
                item["rank_local_graph_batch_size"],
                item["rank_local_verifier_budget"],
            ),
        ),
    }


def runtime_cost_bytes(
    data: bytes,
    *,
    physical_k: int,
    graph_batch_sizes: list[int],
    minimum_predicted_gain: float = 0.01,
) -> tuple[bytes, dict[str, object]]:
    """Validate reuse or serialize measured complete-loop costs into existing schema.

    A raw export summary has a different timing/identity contract. Even if a
    caller toggles an eligibility flag, that cannot extend its measured scope.
    Already supplied schema-v2 artifacts retain their original provenance and
    values; validation is not a new GPU measurement or admission decision.
    """
    utilities = _source_module(
        "_dspark_cost_export_utilities", Path(__file__).with_name("verify_costs.py")
    )
    payload = utilities.parse_json(data)
    if not isinstance(payload, dict):
        raise TypeError("Costs must contain a JSON object")
    whole_step = payload.get("schema") == "verify-cost-measured-summary-v1"
    if whole_step:
        payload = _whole_step_payload(data, payload, minimum_predicted_gain=minimum_predicted_gain)
    elif payload.get("schema") == "verify-cost-raw-v1":
        raise ValueError(
            "Raw observations first require exact-rank reduction from the whole-step producer"
        )
    consumer_path = Path(__file__).resolve().parents[1] / "_torch/speculative/dspark_planner.py"
    consumer = _source_module("_dspark_cost_export_consumer", consumer_path)
    fingerprint = consumer.validate_sps_cost_table_payload(
        payload,
        graph_batch_sizes=graph_batch_sizes,
        max_draft_len=physical_k,
        validate_freshness=False,
    )
    # Use the same constructor as the file loader, including its bounded exact
    # candidate set and any already supplied drain metadata. No policy is added.
    consumer._build_exact_sps_cost_table(
        payload, max_draft_len=physical_k, validate_artifact_freshness=False
    )
    output = (
        json.dumps(payload, sort_keys=True, allow_nan=False).encode() + b"\n"
        if whole_step
        else data
    )
    return output, {
        "schema": "dspark-existing-cost-structural-check-v1",
        "source_artifact_sha256": hashlib.sha256(data).hexdigest(),
        "runtime_cost_artifact_sha256": hashlib.sha256(output).hexdigest(),
        "measured_timing_scope": WHOLE_STEP_SCOPE if whole_step else "original-supplied-artifact",
        "recorded_engine_fingerprint_sha256": payload["engine_fingerprint_sha256"],
        "recorded_source_head": fingerprint["source_head"],
        "physical_k": physical_k,
        "rank_local_graph_batch_sizes": graph_batch_sizes,
        "structural_validation_passed": True,
        "destination_freshness_checked": False,
        "new_measurements_created": False,
        "direct_whole_step_measurements_exported": whole_step,
        "task_accuracy_qualified": False,
        "performance_or_economic_admission_qualified": False,
        "warning": (
            "Supplied predictions retain their original source and GPU identity; "
            "their freshness and performance on the destination are unchecked."
        ),
    }


def check_cost_bytes(
    data: bytes, *, physical_k: int, graph_batch_sizes: list[int]
) -> dict[str, object]:
    """Return the existing artifact or complete-loop export's structural report."""
    return runtime_cost_bytes(data, physical_k=physical_k, graph_batch_sizes=graph_batch_sizes)[1]


def _write_new_bytes(path: Path, data: bytes) -> None:
    """Preserve every original byte; never overwrite or follow a final symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--physical-k", type=int, required=True)
    parser.add_argument("--graph-batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--minimum-predicted-gain",
        type=float,
        default=0.01,
        help="Policy margin for a newly exported whole-step table only.",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional new copy; original bytes are preserved."
    )
    args = parser.parse_args(argv)
    if args.input.is_symlink() or not args.input.is_file():
        raise ValueError("Input must be a regular, non-symlink cost artifact")
    data = args.input.read_bytes()
    output, report = runtime_cost_bytes(
        data,
        physical_k=args.physical_k,
        graph_batch_sizes=args.graph_batch_sizes,
        minimum_predicted_gain=args.minimum_predicted_gain,
    )
    if args.input.read_bytes() != data:
        raise ValueError("Original cost artifact changed during validation")
    if args.output is not None:
        _write_new_bytes(args.output, output)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
