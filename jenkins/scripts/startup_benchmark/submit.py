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
"""Plan or submit serialized, exclusive-node startup jobs using SLURM/Pyxis."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument(
        "--repo", required=True, type=Path, help="Shared checkout containing runner.py"
    )
    parser.add_argument(
        "--models-root", required=True, type=Path, help="Shared checkpoint directory"
    )
    parser.add_argument("--output", required=True, type=Path, help="Fresh shared output directory")
    parser.add_argument(
        "--image", required=True, help="Prebuilt Pyxis image containing the runtime"
    )
    parser.add_argument("--partition", required=True)
    parser.add_argument("--account")
    parser.add_argument(
        "--time", help="Override per-case wall time; default derives from all trial timeouts"
    )
    parser.add_argument("--constraint", help="SLURM node feature constraint")
    parser.add_argument("--nodes", type=int, choices=[1], default=1)
    parser.add_argument("--gpus-per-node", type=int, help="Default: case TP * PP (maximum eight)")
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument(
        "--mount", action="append", default=[], help="Additional SRC:DST[:ro|rw] mount"
    )
    parser.add_argument("--cases", help="Comma-separated case names; default: non-optional cases")
    parser.add_argument("--variants", help="Comma-separated variant names; default: all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile", choices=["loader_isolation", "application_cold"], default="application_cold"
    )
    parser.add_argument(
        "--runtime-cache-seed", type=Path, help="Prepared cache root for loader_isolation"
    )
    parser.add_argument(
        "--keep-runtime-cache", action="store_true", help="Retain per-trial runtime caches"
    )
    parser.add_argument("--cache-reset-command", help="Explicit JSON argv passed to the runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--submit", action="store_true", help="Actually invoke sbatch; default is dry-run"
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="Only write scripts and a submission plan"
    )
    return parser


def _mounts(args: argparse.Namespace) -> str:
    paths = [args.repo, args.models_root, args.output]
    if args.runtime_cache_seed:
        paths.append(args.runtime_cache_seed)
    if any(":" in str(path) for path in paths):
        raise ValueError("Mount paths must not contain colons")
    mounts = [
        f"{args.repo}:{args.repo}:ro",
        f"{args.models_root}:{args.models_root}:ro",
        f"{args.output}:{args.output}:rw",
    ]
    destinations = {str(args.repo), str(args.models_root), str(args.output)}
    if len(destinations) != 3:
        raise ValueError("Repository, models, and output must have distinct mount paths")
    if args.runtime_cache_seed:
        if args.runtime_cache_seed == args.output:
            raise ValueError("Cache seed must not be the output directory")
        if str(args.runtime_cache_seed) not in destinations:
            mounts.append(f"{args.runtime_cache_seed}:{args.runtime_cache_seed}:ro")
            destinations.add(str(args.runtime_cache_seed))
    for mount in args.mount:
        parts = mount.split(":")
        if len(parts) not in (2, 3) or not all(Path(part).is_absolute() for part in parts[:2]):
            raise ValueError("--mount must be absolute SRC:DST[:ro|rw]")
        if len(parts) == 3 and parts[2] not in ("ro", "rw"):
            raise ValueError("--mount mode must be ro or rw")
        if parts[1] in destinations:
            raise ValueError(f"Duplicate mount destination: {parts[1]}")
        destinations.add(parts[1])
        mounts.append(mount if len(parts) == 3 else mount + ":ro")
    if any("," in mount or "\n" in mount or "\r" in mount for mount in mounts):
        raise ValueError("Mount paths must not contain commas or newlines")
    return ",".join(mounts)


def _script(args: argparse.Namespace, case: dict, variants: str) -> str:
    runner = args.repo / "jenkins/scripts/startup_benchmark/runner.py"
    command = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--cpu-bind=none",
        f"--container-image={args.image}",
        f"--container-mounts={_mounts(args)}",
        f"--container-workdir={args.output}",
        "env",
        f"LLM_MODELS_ROOT={args.models_root}",
        "python3",
        str(runner),
        "run",
        "--matrix",
        str(args.output / "matrix.json"),
        "--cases",
        case["name"],
        "--variants",
        variants,
        "--repeats",
        str(args.repeats),
        "--profile",
        args.profile,
        "--output",
        str(args.output / "results" / case["name"]),
        "--exclusive-node",
        "--runtime-image",
        args.image,
    ]
    if args.cache_reset_command:
        command.extend(["--cache-reset-command", args.cache_reset_command])
    if args.runtime_cache_seed:
        command.extend(["--runtime-cache-seed", str(args.runtime_cache_seed)])
    if args.keep_runtime_cache:
        command.append("--keep-runtime-cache")
    return "#!/bin/bash\nset -euo pipefail\n" + shlex.join(command) + "\n"


def _sbatch(
    args: argparse.Namespace, case: dict, script: Path, previous: str | None, variant_count: int
) -> list[str]:
    helper_budget = 300 if args.cache_reset_command else 0
    seconds = args.repeats * variant_count * (case["timeout_seconds"] + 90 + helper_budget) + 600
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    wall_time = args.time or f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--exclusive",
        "--mem=0",
        f"--gpus-per-node={args.gpus_per_node or case['tp'] * case['pp']}",
        f"--partition={args.partition}",
        f"--time={wall_time}",
        f"--job-name=startup-{case['name']}",
        f"--output={args.output}/slurm-{case['name']}-%j.out",
    ]
    for flag in ("account", "constraint", "cpus_per_task"):
        if value := getattr(args, flag):
            command.append(f"--{flag.replace('_', '-')}={value}")
    if previous:
        command.append(f"--dependency=afterany:{previous}")
    return command + [str(script)]


def _save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Write job artifacts, and submit only when explicitly requested."""
    from runner import load_matrix, select_names

    parser = _parser()
    args = parser.parse_args(argv)
    args.repo, args.models_root, args.output = (
        path.expanduser().resolve() for path in (args.repo, args.models_root, args.output)
    )
    if args.runtime_cache_seed:
        args.runtime_cache_seed = args.runtime_cache_seed.expanduser().resolve()
    if args.profile == "loader_isolation" and not args.runtime_cache_seed:
        parser.error("loader_isolation requires --runtime-cache-seed")
    if args.profile == "application_cold" and args.runtime_cache_seed:
        parser.error("application_cold cannot reuse a runtime cache seed")
    matrix = load_matrix(args.matrix)
    cases = select_names(matrix["cases"], args.cases)
    variants = ",".join(item["name"] for item in select_names(matrix["variants"], args.variants))
    if not cases or not variants:
        parser.error("Select at least one case and variant")
    if not (args.repo / "jenkins/scripts/startup_benchmark/runner.py").is_file():
        parser.error("--repo must contain jenkins/scripts/startup_benchmark/runner.py")
    if args.repeats < 1 or (args.cpus_per_task is not None and args.cpus_per_task < 1):
        parser.error("Repeats and CPUs per task must be positive")
    if args.cache_reset_command:
        reset = json.loads(args.cache_reset_command)
        if (
            not isinstance(reset, list)
            or not reset
            or not all(isinstance(arg, str) and arg for arg in reset)
        ):
            parser.error("--cache-reset-command must be a nonempty JSON argv list")
    _mounts(args)
    for case in cases:
        world = case["tp"] * case["pp"]
        gpus = args.gpus_per_node if args.gpus_per_node is not None else world
        if not 1 <= world <= gpus <= 8:
            parser.error(
                f"{case['name']}: only one node with 1–8 GPUs is supported; TP * PP must fit"
            )
    manifest_path = args.output / "submission.json"
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a fresh --output; existing artifacts found in {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental resubmission or destruction of prior evidence.
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        manifest_file.write('{"status": "preparing"}\n')
    (args.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": 1,
        "status": "submitting" if args.submit else "dry_run",
        "image_reference": args.image,
        "image_digest": None,
        "jobs": [],
        "repo": str(args.repo),
        "models_root": str(args.models_root),
        "profile": args.profile,
        "repeats": args.repeats,
        "variants": variants.split(","),
        "runtime_cache_seed": str(args.runtime_cache_seed) if args.runtime_cache_seed else None,
        "keep_runtime_cache": args.keep_runtime_cache,
        "collect_command": [
            "python3",
            str(args.repo / "jenkins/scripts/startup_benchmark/runner.py"),
            "collect",
            "--output",
            str(args.output),
        ],
    }
    if match := re.search(r"@(sha256:[0-9a-fA-F]{64})$", args.image):
        manifest["image_digest"] = match.group(1)
    for case in cases:
        script = args.output / f"{case['name']}.sh"
        script.write_text(_script(args, case, variants), encoding="utf-8")
        manifest["jobs"].append(
            {"case": case["name"], "script": str(script), "status": "planned", "job_id": None}
        )
    _save_manifest(manifest_path, manifest)
    previous = None
    for case, job in zip(cases, manifest["jobs"]):
        command = _sbatch(args, case, Path(job["script"]), previous, len(manifest["variants"]))
        job["command"] = command
        print(shlex.join(command))
        if not args.submit:
            previous = "<PREVIOUS_JOB_ID>"
            continue
        job["status"] = "submitting"
        _save_manifest(manifest_path, manifest)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            job.update(status="submission_unknown", error=str(error))
            manifest["status"] = "submission_failed"
            _save_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"Submission stopped; inspect {manifest_path} before retrying"
            ) from error
        job.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        if result.returncode or not re.fullmatch(r"[0-9]+(?:;[\w.-]+)?", result.stdout.strip()):
            job["status"] = "submission_failed" if result.returncode else "submission_unknown"
            manifest["status"] = "submission_failed"
            _save_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"Submission stopped; inspect {manifest_path}; existing jobs were not cancelled"
            )
        previous = result.stdout.strip().split(";")[0]
        job.update(status="submitted", job_id=previous)
        _save_manifest(manifest_path, manifest)
    if args.submit:
        manifest["status"] = "submitted"
    _save_manifest(manifest_path, manifest)
    print(f"Submission manifest: {manifest_path}")
    print("After jobs finish: " + shlex.join(manifest["collect_command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
