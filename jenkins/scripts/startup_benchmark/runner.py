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
"""Opt-in, single-node startup QA; no inference requests or product imports."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

if __package__:
    from .cache import (
        CacheVerificationError,
        checkpoint_manifest,
        discover_checkpoint_files,
        evict_and_verify,
    )
else:
    from cache import (
        CacheVerificationError,
        checkpoint_manifest,
        discover_checkpoint_files,
        evict_and_verify,
    )

POLICY_PATTERN = re.compile(
    r"Checkpoint I/O policy: requested=(?P<requested>[^,]+), "
    r"selected=(?P<selected>[^,]+), activated=(?P<activated>True|False), "
    r"effective=(?P<effective>[^,]+), fallback_reason=(?P<fallback_reason>.*)\."
)
CACHE_PATHS = {
    "CUDA_CACHE_PATH": "cuda",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "DG_JIT_CACHE_DIR": "deep_gemm",
    "HF_HOME": "huggingface",
    "HF_MODULES_CACHE": "huggingface_modules",
    "XDG_CACHE_HOME": "xdg",
    "TLLM_AUTOTUNER_CACHE_PATH": "autotuner.json",
}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metadata_fingerprint(directories: list[Path]) -> str:
    return fingerprint(
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in directories
            for path in sorted(directory.glob("*.json"))
        }
    )


def seed_fingerprint(seed: Path | None) -> str | None:
    if seed is None:
        return None
    entries = [
        (str(path.relative_to(seed)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(seed.rglob("*"))
        if path.is_file()
    ]
    if not entries:
        raise ValueError("A prepared runtime cache seed must contain files")
    return fingerprint(entries)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_matrix(path: Path) -> dict:
    """Validate QA orchestration fields without importing the runtime config schema."""
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or set(matrix) != {"version", "cases", "variants"}:
        raise ValueError("Matrix requires exactly version, cases and variants")
    if matrix["version"] != 1:
        raise ValueError("Unsupported matrix version")
    for kind in ("cases", "variants"):
        entries = matrix[kind]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{kind} must be a nonempty list")
        names = set()
        for entry in entries:
            allowed = {"name", "config"}
            if kind == "cases":
                allowed |= {
                    "model",
                    "size_group",
                    "tp",
                    "pp",
                    "ep",
                    "timeout_seconds",
                    "optional",
                    "checkpoint_dirs",
                    "notes",
                }
            if not isinstance(entry, dict) or set(entry) - allowed:
                raise ValueError(f"Unknown {kind} fields: {entry}")
            name = entry.get("name", "")
            if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name) or name in names:
                raise ValueError(f"Invalid or duplicate {kind} name: {name}")
            names.add(name)
            if not isinstance(entry.get("config"), dict):
                raise ValueError(f"{name}: config must be a mapping")
            if kind == "variants":
                if set(entry["config"]) != {"checkpoint_io_policy"} or not isinstance(
                    entry["config"]["checkpoint_io_policy"], str
                ):
                    raise ValueError(f"{name}: variants may change only checkpoint_io_policy")
                continue
            if not isinstance(entry.get("model"), str) or not entry["model"]:
                raise ValueError(f"{name}: local model path required")
            for field in ("tp", "pp", "ep", "timeout_seconds"):
                value = entry.get(field, 1 if field != "timeout_seconds" else 7200)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name}: {field} must be a positive integer")
                entry[field] = value
            if entry["tp"] * entry["pp"] > 8 or entry["tp"] % entry["ep"]:
                raise ValueError(f"{name}: only single-node, up-to-eight-rank TP/PP/EP supported")
            if "optional" in entry and type(entry["optional"]) is not bool:
                raise ValueError(f"{name}: optional must be boolean")
            if not isinstance(entry.get("checkpoint_dirs", []), list) or not all(
                isinstance(item, str) for item in entry.get("checkpoint_dirs", [])
            ):
                raise ValueError(f"{name}: checkpoint_dirs must be a list of paths")
    return matrix


def select_names(entries: list[dict], names: str | None) -> list[dict]:
    if names is None:
        return [entry for entry in entries if not entry.get("optional", False)]
    requested = names.split(",")
    lookup = {entry["name"]: entry for entry in entries}
    if len(set(requested)) != len(requested) or any(name not in lookup for name in requested):
        raise ValueError(f"Unknown or duplicate names: {names}; available: {', '.join(lookup)}")
    return [lookup[name] for name in requested]


def local_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"Unresolved model/cache path: {value}")
    path = Path(expanded).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Expected a local directory: {path}")
    return path


def runtime_cache_environment(root: Path, seed: Path | None) -> dict[str, str]:
    if seed is None:
        root.mkdir()
    else:
        shutil.copytree(seed, root)
    environment = os.environ.copy()
    # The single driver owns local rank spawning, not the outer srun MPI context.
    for name in tuple(environment):
        if name.startswith(("MPI", "OMPI", "PMI", "SLURM")) or name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "TRTLLM_FLASHINFER_WORKSPACE_MANAGED",
        ):
            environment.pop(name)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name, relative in CACHE_PATHS.items():
        path = root / relative
        if name != "TLLM_AUTOTUNER_CACHE_PATH":
            path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    return environment


def parse_policy(log: str, requested: str) -> dict:
    observations = [match.groupdict() for match in POLICY_PATTERN.finditer(log)]
    policy = {
        "requested": requested,
        "effective": "unknown",
        "activated": False,
        "complete": False,
        "scope": "server_reported",
        "observations": observations,
    }
    matching = [item for item in observations if item["requested"] == requested]
    if matching and len(matching) == len(observations):
        states = {
            (item["selected"], item["effective"], item["activated"], item["fallback_reason"])
            for item in matching
        }
        if len(states) == 1:
            selected, effective, activated, reason = states.pop()
            policy.update(
                selected=selected,
                effective=effective,
                activated=activated == "True",
                fallback_reason=reason,
                complete=True,
            )
    return policy


def extract_metrics(server_info: dict) -> dict:
    startup = server_info.get("startup_metrics", {})
    metrics = {}
    for section, prefix in (("model_loader", ""), ("draft_model_loader", "draft_model_")):
        for name, value in startup.get(section, {}).items():
            if (
                name.endswith("_seconds")
                and type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
            ):
                metrics[prefix + name] = value
        phases = [
            prefix + f"checkpoint_{phase}_seconds" for phase in ("preparation", "finalization")
        ]
        phases.insert(1, prefix + "weight_population_seconds")
        if all(name in metrics for name in phases):
            metrics[prefix + "checkpoint_pipeline_seconds"] = sum(metrics[name] for name in phases)
    return metrics


def wait_ready(
    process: subprocess.Popen, address_file: Path, started: float, timeout: int
) -> tuple[str, float]:
    # The address file is published before model initialization, not at readiness.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before ready: {process.returncode}")
        if address_file.exists():
            address = address_file.read_text().strip()
            if re.fullmatch(r"127\.0\.0\.1:\d+", address):
                try:
                    with opener.open(f"http://{address}/health", timeout=1) as response:
                        if response.status == 200:
                            return address, time.monotonic() - started
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
        time.sleep(0.1)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


def enable_subreaper() -> None:
    # Adopt double-forked MPI descendants so detached ranks cannot outlive a trial.
    if sys.platform != "linux":
        raise ValueError("Startup execution requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable benchmark child subreaper")


def child_pids(pid: int) -> list[int]:
    result = []
    paths = list(Path(f"/proc/{pid}/task").glob("*/children"))
    if pid == os.getpid() and not paths:
        raise RuntimeError("Cannot verify benchmark descendants through /proc")
    for children in paths:
        try:
            result.extend(int(value) for value in children.read_text().split())
        except FileNotFoundError:
            pass
    return result


def signal_children(sig: int) -> None:
    # Kill only verified direct children. Their descendants are adopted by this
    # subreaper and handled on the next pass; never signal a stale/reused PID.
    for pid in child_pids(os.getpid()):
        try:
            fd = os.pidfd_open(pid)
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[1]) == os.getpid():
                    signal.pidfd_send_signal(fd, sig)
            finally:
                os.close(fd)
        except (ProcessLookupError, FileNotFoundError):
            pass


def stop_server(process: subprocess.Popen | None) -> None:
    # No later cache reset is allowed until every adopted descendant has exited.
    signal_children(signal.SIGTERM)
    if process is not None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        signal_children(signal.SIGKILL)
        if process is not None:
            process.poll()
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        if not child_pids(os.getpid()):
            return
        time.sleep(0.1)
    raise RuntimeError("Unverified process teardown: aborting before another cache reset")


def runtime_identity(image: str) -> dict:
    runtime_commit = "unknown"
    try:
        version = importlib.metadata.version("tensorrt_llm")
        for link in importlib.metadata.metadata("tensorrt_llm").get_all("Project-URL", []):
            match = re.fullmatch(
                r"Source Commit, https://github.com/NVIDIA/TensorRT-LLM/commit/([0-9a-f]{40})", link
            )
            if match:
                runtime_commit = match[1]
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    repo = Path(__file__).resolve().parents[3]
    commit = (
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if shutil.which("git")
        else None
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu.returncode:
        raise RuntimeError(f"Cannot record GPU inventory: {gpu.stderr}")
    return {
        "runtime_image": image,
        "runtime_version": version,
        "runner_git_commit": commit.stdout.strip()
        if commit and commit.returncode == 0
        else "unknown",
        "runner_source_fingerprint": fingerprint(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        "runtime_git_commit": runtime_commit,
        "hostname": socket.gethostname(),
        "gpu_inventory": sorted(gpu.stdout.splitlines()),
        "prebuilt_flashinfer_cubin_dir": os.environ.get("FLASHINFER_CUBIN_DIR"),
    }


def run_trial(
    args: argparse.Namespace,
    case: dict,
    variant: dict,
    repetition: int,
    identity: dict,
    expected_manifest: list,
    files: list[Path],
    seed: Path | None,
) -> dict:
    trial = args.output / case["name"] / f"repeat_{repetition:02d}" / variant["name"]
    trial.mkdir(parents=True, exist_ok=False)
    config = {**case["config"], **variant["config"]}
    requested = config["checkpoint_io_policy"]
    result = {
        "schema_version": 1,
        "case": case["name"],
        "variant": variant["name"],
        "repetition": repetition,
        "profile": args.profile,
        "status": "failed",
        "identity": identity,
        "metrics": {},
        "cache": {},
        "policy": {"requested": requested, "effective": "unknown", "complete": False},
        "launch_to_ready_seconds": None,
        "error": None,
        "metric_scope": "model_loader_worker_rank_0",
        "readiness_poll_interval_seconds": 0.1,
    }
    process = None
    started = None
    log_path = trial / "server.log"
    try:
        model = local_path(case["model"])
        directories = [model] + [local_path(path) for path in case.get("checkpoint_dirs", [])]
        draft = (config.get("speculative_config") or {}).get("speculative_model_dir")
        if draft:
            config["speculative_config"] = {
                **config["speculative_config"],
                "speculative_model_dir": str(local_path(draft)),
            }
        config_path = trial / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        if seed_fingerprint(seed) != identity.get("runtime_cache_seed_fingerprint"):
            raise ValueError("Runtime cache seed changed between trials")
        environment = runtime_cache_environment(trial / "runtime_cache", seed)
        if seed_fingerprint(seed) != identity.get("runtime_cache_seed_fingerprint"):
            raise ValueError("Runtime cache seed changed while cloning")
        command = [
            "trtllm-serve",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--report_addr",
            str(trial / "server.addr"),
            "--config",
            str(config_path),
            "--tensor_parallel_size",
            str(case["tp"]),
            "--pipeline_parallel_size",
            str(case["pp"]),
            "--moe_expert_parallel_size",
            str(case["ep"]),
        ]
        result["command"] = command
        if checkpoint_manifest(discover_checkpoint_files(directories)) != expected_manifest:
            raise ValueError("Checkpoint changed between trials")
        expected_metadata = identity.get("checkpoint_metadata_fingerprint")
        if expected_metadata is not None and expected_metadata != metadata_fingerprint(directories):
            raise ValueError("Checkpoint config/index changed between trials")
        result["cache"] = evict_and_verify(
            files, exclusive=args.exclusive_node, reset_command=args.cache_reset_command
        )
        if child_pids(os.getpid()):
            raise RuntimeError("Cache-reset helper left running descendants")
        with log_path.open("w") as log:
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=trial,
                start_new_session=True,
            )
            address, duration = wait_ready(
                process, trial / "server.addr", started, case["timeout_seconds"]
            )
            result["launch_to_ready_seconds"] = duration
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"http://{address}/server_info", timeout=30) as response:
                info = json.load(response)
            write_json(trial / "server_info.json", info)
            result["metrics"] = extract_metrics(info)
            result["policy"] = parse_policy(log_path.read_text(errors="replace"), requested)
            policy = result["policy"]
            if (
                not result["metrics"].get("total_model_loading_seconds")
                or "checkpoint_pipeline_seconds" not in result["metrics"]
            ):
                raise ValueError("Required model-loader timings are missing")
            expected_policy = "rank_striped_read_ahead" if requested == "auto" else requested
            if (
                not policy["complete"]
                or policy["effective"] != expected_policy
                or policy["activated"] is not (requested != "native")
            ):
                raise ValueError(
                    "Requested loader did not activate, or effective-policy evidence is incomplete"
                )
            if checkpoint_manifest(discover_checkpoint_files(directories)) != expected_manifest or (
                expected_metadata is not None
                and expected_metadata != metadata_fingerprint(directories)
            ):
                raise ValueError("Checkpoint changed during startup")
            result["status"] = "passed"
    except CacheVerificationError as error:
        result.update(status="invalid", error=str(error), cache=error.evidence)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result.update(
            status="invalid" if result["launch_to_ready_seconds"] is not None else "failed",
            error=str(error),
        )
    finally:
        if started is not None:
            result["elapsed_seconds"] = time.monotonic() - started
        try:
            stop_server(process)
            if (
                not getattr(args, "keep_runtime_cache", False)
                and (trial / "runtime_cache").exists()
            ):
                shutil.rmtree(trial / "runtime_cache")
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            result.update(status="failed", cleanup_error=str(error))
            raise
        finally:
            write_json(trial / "result.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    if not args.exclusive_node:
        raise ValueError("--exclusive-node is required; obtain exclusive allocation first")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if any(
        name in os.environ
        for name in (
            "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR",
            "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY",
            "TLLM_SPAWN_PROXY_PROCESS",
        )
    ):
        raise ValueError("External TRT-LLM MPI sessions cannot be used for isolated startup trials")
    if any(
        int(os.environ.get(name, "1")) > 1
        for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_UNIV_SIZE")
    ):
        raise ValueError("Run one benchmark driver, not one driver per MPI rank")
    if args.profile == "loader_isolation" and args.runtime_cache_seed is None:
        raise ValueError("loader_isolation requires --runtime-cache-seed")
    if args.profile == "application_cold" and args.runtime_cache_seed is not None:
        raise ValueError("application_cold cannot reuse a runtime cache seed")
    matrix = load_matrix(args.matrix)
    cases = select_names(matrix["cases"], args.cases)
    variants = select_names(matrix["variants"], args.variants)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run_manifest.json").exists():
        raise ValueError("Use a fresh output directory for every campaign")
    seed = local_path(str(args.runtime_cache_seed)) if args.runtime_cache_seed else None
    if seed and (args.output == seed or seed in args.output.parents):
        raise ValueError("Output must not be inside the runtime cache seed")
    base_identity = runtime_identity(args.runtime_image)
    base_identity["runtime_cache_seed_fingerprint"] = seed_fingerprint(seed)
    base_identity["controlled_runtime_caches"] = CACHE_PATHS
    enable_subreaper()
    write_json(
        args.output / "run_manifest.json",
        {
            "cases": cases,
            "variants": variants,
            "repeats": args.repeats,
            "profile": args.profile,
            "identity": base_identity,
            "runtime_cache_seed": str(seed) if seed else None,
        },
    )
    failed = False
    for case in cases:
        case = dict(case)
        model = local_path(case["model"])
        directories = [model] + [local_path(path) for path in case.get("checkpoint_dirs", [])]
        case["model"] = str(model)
        case["checkpoint_dirs"] = [str(path) for path in directories[1:]]
        draft = (case["config"].get("speculative_config") or {}).get("speculative_model_dir")
        if draft:
            draft_path = local_path(draft)
            if draft_path not in directories:
                raise ValueError("External draft checkpoint must be listed in checkpoint_dirs")
            case["config"] = {
                **case["config"],
                "speculative_config": {
                    **case["config"]["speculative_config"],
                    "speculative_model_dir": str(draft_path),
                },
            }
        files = discover_checkpoint_files(directories)
        manifest = checkpoint_manifest(files)
        identity = {
            **base_identity,
            "checkpoint_fingerprint": fingerprint(manifest),
            "checkpoint_metadata_fingerprint": metadata_fingerprint(directories),
            "case_config_fingerprint": fingerprint(case),
            "checkpoint_bytes": sum(item["bytes"] for item in manifest),
            "model_path": str(model),
            "size_group": case.get("size_group"),
            "parallelism": {key: case[key] for key in ("tp", "pp", "ep")},
        }
        for repetition in range(args.repeats):
            # Counterbalance adjacent pairs without adding warmup launches.
            order = variants if repetition % 2 == 0 else list(reversed(variants))
            for variant in order:
                result = run_trial(args, case, variant, repetition, identity, manifest, files, seed)
                failed |= result["status"] != "passed"
    if __package__:
        from .results import collect_results
    else:
        from results import collect_results
    collect_results(args.output)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--matrix", type=Path, default=Path(__file__).with_name("matrix.yaml"))
    launch.add_argument("--cases")
    launch.add_argument("--variants")
    launch.add_argument("--repeats", type=int, default=3)
    launch.add_argument(
        "--profile", choices=("application_cold", "loader_isolation"), default="application_cold"
    )
    launch.add_argument("--runtime-cache-seed", type=Path)
    launch.add_argument(
        "--keep-runtime-cache",
        action="store_true",
        help="Retain potentially large caches for seed preparation",
    )
    launch.add_argument("--runtime-image", required=True)
    launch.add_argument("--exclusive-node", action="store_true")
    launch.add_argument("--cache-reset-command", type=json.loads)
    for command in (launch, commands.add_parser("collect")):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        from results import collect_results

        collect_results(args.output)
        return 0
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        if (args.output / "run_manifest.json").exists():
            write_json(args.output / "run_error.json", {"error": str(error)})
            from results import collect_results

            collect_results(args.output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
