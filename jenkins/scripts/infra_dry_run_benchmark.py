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

"""Run a small CUDA/NCCL smoke test and emit CI-friendly result artifacts."""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

NAME = "infra_dry_run"
MATRIX_SIZE = 64
JUNIT_FILE = "results-infra_dry_run.xml"
MANIFEST_FILE = "infra_dry_run_manifest.json"


class BenchmarkError(RuntimeError):
    """Raised after failure artifacts have been written."""


def _load_runtime_modules() -> tuple[Any, Any]:
    # Normal imports intentionally exercise the installed package and PyTorch.
    import tensorrt_llm
    import torch

    return tensorrt_llm, torch


def _rank_context(environ: Mapping[str, str]) -> dict[str, int]:
    try:
        rank = int(environ.get("RANK", "0"))
        local_rank = int(environ.get("LOCAL_RANK", "0"))
        world_size = int(environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise BenchmarkError("RANK, LOCAL_RANK, and WORLD_SIZE must be integers") from error
    if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
        raise BenchmarkError(
            f"invalid rank context: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _new_result(
    context: Mapping[str, int],
    stage: str | None,
    commit: str | None,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "name": NAME,
        "status": "failed",
        "product_tests_executed": 0,
        **context,
        "stage": stage or environ.get("STAGE_NAME") or environ.get("stageName") or "",
        "commit": commit or environ.get("GIT_COMMIT") or environ.get("gitlabCommit") or "",
        "tensorrt_llm_version": "unknown",
        "tensorrt_llm_module": "unknown",
    }


def _select_cuda_device(torch: Any, local_rank: int) -> str:
    if not torch.cuda.is_available():
        raise BenchmarkError("CUDA is required")
    device_count = int(torch.cuda.device_count())
    if local_rank >= device_count:
        raise BenchmarkError(
            f"LOCAL_RANK {local_rank} cannot select from {device_count} visible CUDA device(s)"
        )
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _cuda_matmul(torch: Any, local_rank: int, device: str) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    left = torch.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=torch.float16, device=device)
    right = torch.full((MATRIX_SIZE, MATRIX_SIZE), 0.25, dtype=torch.float16, device=device)
    output = torch.matmul(left, right)
    torch.cuda.synchronize(local_rank)
    if int(output.numel()) == 0:
        raise BenchmarkError("CUDA matrix multiplication returned an empty tensor")
    if not bool(torch.isfinite(output).all().item()):
        raise BenchmarkError("CUDA matrix multiplication returned non-finite values")
    checksum = float(output.float().sum().item())
    if not math.isfinite(checksum):
        raise BenchmarkError("CUDA matrix multiplication checksum is non-finite")
    return {
        "device": device,
        "matrix_size": MATRIX_SIZE,
        "dtype": "float16",
        "checksum": checksum,
    }


def _initialize_distributed(torch: Any, context: Mapping[str, int], timeout_seconds: int) -> bool:
    if context["world_size"] == 1:
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise BenchmarkError("torch.distributed with NCCL is required for WORLD_SIZE > 1")
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=context["rank"],
        world_size=context["world_size"],
        timeout=timedelta(seconds=timeout_seconds),
    )
    return True


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(result["rank"]),
        "world_size": int(result["world_size"]),
        "status": str(result["status"]),
        "checksum": result.get("cuda", {}).get("checksum"),
        "error": str(result.get("error", "")),
    }


def _gather_summaries(torch: Any, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    world_size = int(result["world_size"])
    local = torch.tensor(
        [
            float(result["rank"]),
            float(world_size),
            float(result["status"] == "passed"),
            float(result.get("cuda", {}).get("checksum", 0.0)),
        ],
        dtype=torch.float64,
        device=f"cuda:{result['local_rank']}",
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    return [
        {
            "rank": int(values[0]),
            "world_size": int(values[1]),
            "status": "passed" if bool(values[2]) else "failed",
            "checksum": float(values[3]),
            "error": "" if bool(values[2]) else "CUDA work failed on this rank",
        }
        for values in (item.cpu().tolist() for item in gathered)
    ]


def _validate(summaries: list[Mapping[str, Any]], world_size: int) -> list[str]:
    errors: list[str] = []
    ranks = [int(item["rank"]) for item in summaries]
    expected_ranks = list(range(world_size))
    if len(summaries) != world_size or sorted(ranks) != expected_ranks:
        errors.append(f"observed ranks {sorted(ranks)} do not match expected {expected_ranks}")
    if any(int(item["world_size"]) != world_size for item in summaries):
        errors.append("rank results contain a world-size mismatch")
    failed = [int(item["rank"]) for item in summaries if item["status"] != "passed"]
    if failed:
        errors.append(f"rank(s) {sorted(failed)} reported failure")
    try:
        checksums = [
            float(item["checksum"]) for item in summaries if item["status"] == "passed"
        ]
    except (TypeError, ValueError):
        errors.append("passed rank result is missing a valid CUDA checksum")
    else:
        if any(not math.isfinite(value) for value in checksums):
            errors.append("rank results contain a non-finite CUDA checksum")
        if checksums and any(
            not math.isclose(value, checksums[0], rel_tol=1e-6, abs_tol=1e-6)
            for value in checksums[1:]
        ):
            errors.append("rank results contain inconsistent CUDA checksums")
    return errors


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _write_reports(
    output_dir: Path,
    result: Mapping[str, Any],
    summaries: list[Mapping[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    world_size = int(result["world_size"])
    manifest: dict[str, Any] = {
        "name": NAME,
        "status": "failed" if errors else "passed",
        "product_tests_executed": 0,
        "stage": result["stage"],
        "commit": result["commit"],
        "world_size": world_size,
        "observed_ranks": sorted(int(item["rank"]) for item in summaries),
        "validation_errors": errors,
        "rank_results": summaries,
        "junit_file": JUNIT_FILE,
    }
    _write_json(output_dir / MANIFEST_FILE, manifest)

    by_rank = {int(item["rank"]): item for item in summaries}
    cases: list[tuple[str, str | None]] = []
    for rank in range(world_size):
        item = by_rank.get(rank)
        failure = None
        if item is None:
            failure = f"missing result for rank {rank}"
        elif item["status"] != "passed":
            failure = str(item.get("error") or f"rank {rank} failed")
        cases.append((f"{NAME}_rank_{rank}_cuda_matmul", failure))
    if world_size > 1:
        cases.append((f"{NAME}_nccl_collective", "; ".join(errors) or None))

    suite = ET.Element(
        "testsuite",
        {
            "name": NAME,
            "tests": str(len(cases)),
            "failures": str(sum(failure is not None for _, failure in cases)),
            "errors": "0",
            "skipped": "0",
            "time": "0",
        },
    )
    properties = ET.SubElement(suite, "properties")
    for name in ("product_tests_executed", "stage", "commit", "world_size"):
        ET.SubElement(properties, "property", {"name": name, "value": str(manifest[name])})
    for name, failure in cases:
        case = ET.SubElement(
            suite,
            "testcase",
            {"classname": NAME, "name": name, "time": "0"},
        )
        if failure:
            node = ET.SubElement(case, "failure", {"message": failure})
            node.text = failure

    root = ET.Element("testsuites")
    root.append(suite)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_dir / JUNIT_FILE, encoding="utf-8", xml_declaration=True)
    return manifest


def run_benchmark(
    output_dir: Path,
    *,
    stage: str | None = None,
    commit: str | None = None,
    timeout_seconds: int = 120,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    context = _rank_context(environment)
    result = _new_result(context, stage, commit, environment)
    summaries = [_summary(result)]
    torch = None
    distributed = False

    try:
        trtllm, torch = _load_runtime_modules()
        result["tensorrt_llm_version"] = str(getattr(trtllm, "__version__", "unknown"))
        result["tensorrt_llm_module"] = str(getattr(trtllm, "__file__", "unknown"))
        device = _select_cuda_device(torch, context["local_rank"])
        distributed = _initialize_distributed(torch, context, timeout_seconds)
        try:
            result["cuda"] = _cuda_matmul(torch, context["local_rank"], device)
            result["status"] = "passed"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        summaries = _gather_summaries(torch, result) if distributed else [_summary(result)]
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        summaries = [_summary(result)]
    finally:
        if distributed and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as error:
                result["status"] = "failed"
                result["error"] = f"distributed cleanup failed: {error}"
                summaries = [
                    item for item in summaries if int(item["rank"]) != context["rank"]
                ] + [_summary(result)]

    errors = _validate(summaries, context["world_size"])
    result["overall_status"] = "failed" if errors else "passed"
    _write_json(output_dir / f"{NAME}_rank_{context['rank']}.json", result)
    if context["rank"] == 0:
        _write_reports(output_dir, result, summaries, errors)
    if errors:
        raise BenchmarkError("; ".join(errors))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--stage")
    parser.add_argument("--commit")
    parser.add_argument("--distributed-timeout-seconds", type=int, default=120)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run_benchmark(
            args.output_dir,
            stage=args.stage,
            commit=args.commit,
            timeout_seconds=args.distributed_timeout_seconds,
        )
    except Exception as error:
        print(f"{NAME} failed: {error}", file=sys.stderr)
        return 1
    print(f"{NAME} passed on rank {result['rank']}/{result['world_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
