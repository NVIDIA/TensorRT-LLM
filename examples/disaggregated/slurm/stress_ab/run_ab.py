# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run the pinned NVBUG 6649384 comparison inside an exclusive eight-B200 Slurm step."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from build_runtime import load_profile, runtime_identity, validate_source_profile

CASE = "test_disaggregated_stress_test[input8k-output1k-conc512-gpt_oss_120b_eagle_trtllm_stress]"
SELECTOR = f"disaggregated/test_disaggregated.py::{CASE}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _command(command: list[str], timeout: int = 300, **kwargs) -> str:
    return subprocess.check_output(command, text=True, timeout=timeout, **kwargs).strip()


def _environment(runtime: Path, harness: Path, models: Path, run_id: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONOPTIMIZE",
        "PYTEST_ADDOPTS",
    ):
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(runtime),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PATH=f"{runtime / 'bin'}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        VIRTUAL_ENV=sys.prefix,
        LLM_ROOT=str(harness),
        LLM_MODELS_ROOT=str(models),
        TLLM_DISAGG_STRESS_KEEP_LOGS="1",
        TLLM_AB_RUN_ID=run_id,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return env


def _validate_provenance(wheel: Path, provenance: dict, profile: dict | None = None) -> None:
    profile = profile or load_profile()
    _require(provenance.get("status") == "built", "baseline build did not finish successfully")
    _require(provenance.get("source_unchanged_after_build") is True, "source changed during build")
    _require(
        provenance.get("source_sha") == profile["control"],
        "wheel source SHA is not the pinned control",
    )
    _require(
        provenance.get("profile", load_profile()) == profile, "build and runtime profiles differ"
    )
    _require(provenance.get("clean_source") is True, "baseline build source was not clean")
    _require(provenance.get("wheel") == wheel.name, "provenance wheel filename mismatch")
    _require(provenance.get("wheel_sha256") == _sha(wheel), "wheel SHA256 mismatch")
    _require(
        bool(provenance.get("build_log") and provenance.get("build_command")),
        "missing build command/log provenance",
    )
    _require(
        bool(re.fullmatch(r"sha256:[0-9a-f]{64}", provenance.get("image_digest", ""))),
        "missing immutable container digest",
    )
    for field in ("image", "image_digest"):
        _require(
            os.environ.get(f"TLLM_AB_{field.upper()}") == provenance.get(field),
            f"running container {field} does not match build provenance",
        )


def _validate_dependencies(provenance: dict) -> dict:
    expected_python = provenance.get("runtime_python")
    expected_prefix = provenance.get("runtime_prefix")
    _require(sys.prefix != sys.base_prefix, "run the driver with the recorded baseline venv Python")
    _require(
        os.path.abspath(sys.executable) == expected_python,
        "driver interpreter differs from recorded runtime_python; preserve the venv symlink path",
    )
    _require(
        os.path.abspath(sys.prefix) == expected_prefix, "driver uses a different runtime_prefix"
    )
    actual = runtime_identity(Path(sys.executable))
    _require(bool(actual.get("runtime_distributions")), "no installed dependency inventory")
    for field, value in actual.items():
        _require(
            provenance.get(field) == value, f"baseline dependency environment changed: {field}"
        )
    entrypoints = provenance.get("runtime_entrypoints", {})
    _require(set(entrypoints) == {"aiperf", "lm_eval"}, "missing baseline CLI provenance")
    for name, metadata in entrypoints.items():
        script = Path(expected_prefix) / "bin" / name
        _require(
            metadata.get("path") == str(script) and metadata.get("sha256") == _sha(script),
            f"baseline CLI changed: {name}",
        )
    return actual


def _prepare_runtime(args: argparse.Namespace, provenance: dict) -> dict:
    _validate_provenance(args.wheel, provenance, args.profile)
    _require(
        shutil.disk_usage(args.output).free > args.wheel.stat().st_size * 4 + 20 * 1024**3,
        "insufficient free storage for the installed wheel and retained request/worker logs",
    )
    git = ["git", "-C", str(args.harness)]
    validate_source_profile(args.harness, args.profile)
    harness = _validate_harness(args.harness, args.profile)
    blobs = {
        arm: {
            relative: subprocess.check_output(git + ["show", f"{args.profile[arm]}:{relative}"])
            for relative in args.profile["runtime_files"]
        }
        for arm in ("control", "treatment")
    }
    base = args.output / "runtime" / "base"
    base.mkdir(parents=True)
    install = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--target",
        str(base),
        str(args.wheel),
    ]
    with (args.output / "pip.log").open("w") as log:
        subprocess.run(install, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    for relative, blob in blobs["control"].items():
        _require(
            (base / relative).read_bytes() == blob,
            f"wheel Python is not control source: {relative}",
        )
    _require(not list(base.rglob("*.pyc")), "wheel contains stale bytecode")
    binaries = {
        str(path.relative_to(base)): _sha(path)
        for path in base.rglob("*")
        if path.is_file() and (".so" in path.name or path.suffix in (".a", ".cubin", ".fatbin"))
    }
    _require(bool(binaries), "wheel has no compiled artifacts")
    for arm, files in blobs.items():
        _overlay_runtime(base, base.parent / arm, files, binaries)
    return {
        "provenance": provenance,
        "install_command": install,
        "binaries": binaries,
        "runtime_file_sha256": {
            arm: {relative: hashlib.sha256(blob).hexdigest() for relative, blob in files.items()}
            for arm, files in blobs.items()
        },
        **harness,
    }


def _overlay_runtime(base: Path, target: Path, files: dict, binaries: dict) -> None:
    shutil.copytree(base, target, copy_function=os.link)
    for relative, blob in files.items():
        destination = target / relative
        destination.unlink()  # Never write through a hardlink to the baseline.
        destination.write_bytes(blob)
    for relative, digest in binaries.items():
        _require(_sha(target / relative) == digest, f"compiled artifact differs: {relative}")
    _require((target / "bin/trtllm-serve").is_file(), "wheel lacks trtllm-serve entry point")


def _validate_harness(harness: Path, profile: dict) -> dict:
    git = ["git", "-C", str(harness)]
    head = _command(git + ["rev-parse", "HEAD"])
    if profile["harness_sha"]:
        _require(head == profile["harness_sha"], "test harness SHA differs from profile")
        _require(
            not _command(git + ["status", "--porcelain", "--untracked-files=all"]),
            "pinned test harness must be clean",
        )
    test = harness / "tests/integration/defs/disaggregated/test_disaggregated.py"
    tree = ast.parse(test.read_text())
    stress = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "test_disaggregated_stress_test"
    )
    configs = []
    for node in ast.walk(stress):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TestConfig"
        ):
            values = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
            if values.get("test_desc") == "gpt_oss_120b_eagle_trtllm_stress":
                configs.append(values)
    _require(
        len(configs) == 1 and configs[0].get("request_count") == profile["expected_requests"],
        "test request count differs from profile",
    )
    return {
        "harness_sha": head,
        "harness_diff": _command(git + ["diff", "HEAD"]),
        "harness_files": {
            relative: _sha(harness / relative)
            for relative in (
                "tests/integration/defs/disaggregated/test_disaggregated.py",
                "tests/integration/defs/conftest.py",
                "tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml",
                "tests/integration/lm_eval_configs/gsm8k_local.yaml",
                "requirements-dev.txt",
            )
        },
    }


def _gpu_inventory() -> list[list[str]]:
    _require(bool(os.environ.get("SLURM_JOB_ID")), "an existing Slurm allocation is required")
    _require(
        os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES")) == "1",
        "exactly one Slurm node is required",
    )
    _require(
        os.environ.get("SLURM_GPUS_ON_NODE") == "8",
        "allocation must provide eight GPUs on this node",
    )
    rows = [
        line.split(", ")
        for line in _command(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
    ]
    _require(
        len(rows) == 8 and all(len(row) == 4 and "B200" in row[1] for row in rows),
        "exactly eight B200 GPUs are required",
    )
    return rows


def _models(models_root: Path) -> dict:
    result = {}
    for name in ("gpt-oss-120b", "gpt-oss-120b-Eagle3"):
        directory = models_root / "gpt_oss" / name
        config = directory / "config.json"
        json.loads(config.read_text())
        indices = list(directory.glob("*.index.json"))
        shards = {
            directory / shard
            for index in indices
            for shard in json.loads(index.read_text()).get("weight_map", {}).values()
        }
        if not shards:
            shards = set(directory.glob("*.safetensors")) | set(
                directory.glob("pytorch_model*.bin")
            )
        _require(bool(shards), f"no model shards found: {directory}")
        for shard in shards:
            with shard.open("rb") as stream:
                _require(bool(stream.read(1)), f"empty model shard: {shard}")
        result[name] = {"config_sha256": _sha(config), "shards": sorted(map(str, shards))}
    return result


def _accuracy_inputs(args: argparse.Namespace) -> dict:
    env = _environment(args.output / "runtime/base", args.harness, args.models_root, "preflight")
    executables = {
        name: shutil.which(name, path=env["PATH"]) for name in ("aiperf", "lm_eval", "python3")
    }
    _require(all(executables.values()), "aiperf, lm_eval and python3 must already be installed")
    venv_bin = Path(sys.executable).parent
    _require(
        all(Path(path).parent == venv_bin for path in executables.values()),
        "aiperf, lm_eval and python3 must resolve in the recorded venv/bin",
    )
    template = args.harness / "tests/integration/lm_eval_configs/gsm8k_local.yaml"
    dataset = args.models_root / "datasets/openai/gsm8k/main/test-00000-of-00001.parquet"
    target = args.models_root / "gpt_oss/gpt-oss-120b"
    code = """
import importlib.metadata, json, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer
versions = {name: importlib.metadata.version(name) for name in ('lm_eval', 'aiperf')}
if versions != {'lm_eval': '0.4.10', 'aiperf': '0.8.0'}:
    raise ValueError('lm_eval==0.4.10 and aiperf==0.8.0 are required: ' + str(versions))
columns = pq.ParquetFile(sys.argv[2]).schema.names
if not {'question', 'answer'}.issubset(columns):
    raise ValueError('GSM8K parquet lacks question/answer columns')
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=True)
if not tokenizer.encode('A/B tokenizer preflight'):
    raise ValueError('tokenizer returned no tokens')
print('AB_INPUTS=' + json.dumps({'versions': versions, 'dataset_columns': columns}))
"""
    log = args.output / "inputs-preflight.log"
    try:
        text = _command(
            [sys.executable, "-c", code, str(target), str(dataset)],
            cwd=args.output,
            env=env,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        log.write_text(error.output)
        raise
    log.write_text(text + "\n")
    return {
        "executables": executables,
        "dataset": str(dataset),
        "dataset_sha256": _sha(dataset),
        "accuracy_template_sha256": _sha(template),
        "tokenizer_files": {
            str(path): _sha(path) for path in target.glob("*token*") if path.is_file()
        },
    }


def _cleanliness(run_id: str) -> dict:
    gpu = _command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], timeout=10
    )
    owned = []
    marker = f"TLLM_AB_RUN_ID={run_id}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == os.getuid() and marker in (
                entry / "environ"
            ).read_bytes().split(b"\0"):
                state = (entry / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state != "Z":
                    owned.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return {"clean": not gpu and not owned, "gpu_processes": gpu.splitlines(), "owned_pids": owned}


def _runtime_preflight(
    runtime: Path, expected_hashes: dict, env: dict, log: Path, dependency_versions: dict
) -> dict:
    code = """
import hashlib, importlib, importlib.metadata, json, pathlib, sys
import tensorrt_llm
import tensorrt_llm.bindings as bindings
files = {}
for relative in json.loads(sys.argv[1]):
    name = relative.removesuffix('.py').replace('/', '.').removesuffix('.__init__')
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    files[relative] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
print('AB_RUNTIME=' + json.dumps({'package': str(pathlib.Path(tensorrt_llm.__file__).resolve()),
 'runtime_files': files, 'bindings': str(pathlib.Path(bindings.__file__).resolve()),
 'versions': {name: importlib.metadata.version(name) for name in
 ('tensorrt-llm', 'torch', 'aiperf', 'lm_eval', 'nixl-cu13', 'pytest')}}))
"""
    try:
        text = _command(
            [sys.executable, "-c", code, json.dumps(list(expected_hashes))],
            cwd=runtime,
            env=env,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        log.write_text(error.output)
        raise
    log.write_text(text + "\n")
    data = json.loads(
        next(
            line.removeprefix("AB_RUNTIME=")
            for line in text.splitlines()
            if line.startswith("AB_RUNTIME=")
        )
    )
    for field in ("package", "bindings"):
        _require(
            Path(data[field]).is_relative_to(runtime), f"{field} imported outside selected runtime"
        )
    _require(set(data["runtime_files"]) == set(expected_hashes), "incomplete runtime file imports")
    for relative, expected in expected_hashes.items():
        actual = data["runtime_files"][relative]
        _require(
            Path(actual["path"]) == runtime / relative,
            f"{relative} imported outside selected runtime",
        )
        _require(
            actual["sha256"] == expected, f"imported Python differs from pinned source: {relative}"
        )
    for name, version in dependency_versions.items():
        _require(
            data["versions"][name] == version, f"runtime dependency differs from profile: {name}"
        )
    return data


def _accounting(path: Path, expected: int = 60000) -> dict:
    counts = {"records": 0, "valid": 0, "cancelled": 0, "errors": 0}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            _require(
                isinstance(record, dict) and ("metrics" in record or "error" in record),
                "unexpected per-request record schema",
            )
            error = record.get("error") or {}
            metadata = record.get("metadata") or {}
            _require(
                isinstance(error, dict) and isinstance(metadata, dict), "invalid record fields"
            )
            cancelled = (
                metadata.get("was_cancelled") is True
                or error.get("code") == 499
                or error.get("type") == "RequestCancellationError"
            )
            counts["records"] += 1
            counts["cancelled" if cancelled else "errors" if error else "valid"] += 1
    _require(
        counts["records"] == expected,
        f"expected {expected} request records, got {counts['records']}",
    )
    considered = counts["records"] - counts["cancelled"]
    _require(considered > 0, "all request records are cancelled")
    return {**counts, "error_rate": counts["errors"] / considered}


def _classify(trial: Path, rc: int, timed_out: bool, expected_requests: int = 60000) -> dict:
    _require(not timed_out, "trial timed out")
    _require(rc in (0, 1), f"pytest setup/interruption exit status {rc}")
    cases = ET.parse(trial / "junit.xml").findall(".//testcase")
    _require(len(cases) == 1 and cases[0].get("name") == CASE, "missing or extra selected tests")
    case = cases[0]
    _require(
        case.find("skipped") is None and case.find("error") is None, "test skipped or setup failed"
    )
    failed = case.find("failure") is not None
    _require((rc == 1) == failed, "pytest exit status and JUnit disagree")
    profiles = list((trial / "workspace").rglob("profile_export.jsonl"))
    _require(len(profiles) == 1, "missing or ambiguous per-request accounting")
    logs = {
        name: list((trial / "workspace").rglob(name))
        for name in ("worker_ctx_0.log", "worker_gen_0.log", "disagg_server.log")
    }
    _require(
        all(len(paths) == 1 and paths[0].stat().st_size for paths in logs.values()),
        "full context/generation/proxy logs were not retained",
    )
    accounting = _accounting(profiles[0], expected_requests)
    _require(
        failed or accounting["error_rate"] <= 0.05, "pytest passed despite excessive error rate"
    )
    return {
        "status": "fail" if failed else "pass",
        "accounting": accounting,
        "artifacts": {
            "profile": str(profiles[0]),
            **{name: str(paths[0]) for name, paths in logs.items()},
        },
    }


def _stop_group(process: subprocess.Popen) -> None:
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    process.wait(timeout=5)


def _run_trial(args: argparse.Namespace, arm: str, index: int, identity: dict, run_id: str) -> dict:
    trial = args.output / f"{index:02d}-{arm}"
    trial.mkdir()
    runtime = args.output / "runtime" / arm
    env = _environment(runtime, args.harness, args.models_root, run_id)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-vs",
        SELECTOR,
        "--workspace",
        str(trial / "workspace"),
        "--keep-workspace",
        "--junitxml",
        str(trial / "junit.xml"),
    ]
    result = {
        "arm": arm,
        "status": "invalid",
        "command": command,
        "started": time.time(),
        "pytest_returncode": None,
        "timed_out": False,
    }
    _save(trial / "result.json", result)
    try:
        result["dependencies"] = _validate_dependencies(identity["provenance"])
        before = _cleanliness(run_id)
        result["before"] = before
        _require(before["clean"], "live GPU workload or previous owned process before trial")
        result["runtime"] = _runtime_preflight(
            runtime,
            identity["runtime_file_sha256"][arm],
            env,
            trial / "preflight.log",
            args.profile["dependency_versions"],
        )
        with (trial / "pytest.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=args.harness / "tests/integration/defs",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["pid"] = process.pid
            _save(trial / "result.json", result)
            try:
                result["pytest_returncode"] = process.wait(timeout=args.trial_timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
            finally:
                _stop_group(process)
                if result["pytest_returncode"] is None:
                    result["pytest_returncode"] = process.returncode
        result.update(
            _classify(
                trial,
                result["pytest_returncode"],
                result["timed_out"],
                args.profile["expected_requests"],
            )
        )
    except (ValueError, OSError, ET.ParseError, subprocess.SubprocessError, StopIteration) as error:
        result.update(status="invalid", reason=str(error))
    finally:
        # Allow CUDA contexts to retire; never kill an unrelated GPU process.
        try:
            for _ in range(15):
                result["cleanup"] = _cleanliness(run_id)
                if result["cleanup"]["clean"]:
                    break
                time.sleep(1)
        except (OSError, subprocess.SubprocessError) as error:
            result["cleanup"] = {"clean": False, "error": str(error)}
        if not result["cleanup"]["clean"]:
            result.update(status="invalid", reason="live owned or GPU processes after trial")
        try:
            _validate_dependencies(identity["provenance"])
            result["dependency_environment_unchanged"] = True
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            result.update(
                status="invalid", reason=str(error), dependency_environment_unchanged=False
            )
        result["finished"] = time.time()
        _save(trial / "result.json", result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("wheel", "provenance", "output", "models-root", "harness"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--order", default="control,treatment,treatment,control")
    parser.add_argument("--trial-timeout", type=int, default=12600)
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args()
    args.profile = load_profile(args.profile)
    _require(sys.flags.optimize == 0, "driver must run without Python optimization")
    order = args.order.split(",")
    _require(order and all(arm in ("control", "treatment") for arm in order), "invalid arm order")
    _require(0 < args.trial_timeout <= 12600, "timeout must be between 1 and 12600 seconds")
    for field in ("wheel", "provenance", "output", "models_root", "harness"):
        setattr(args, field, getattr(args, field).resolve())
    if args.dry_plan:
        print(
            json.dumps(
                {
                    "selector": SELECTOR,
                    "order": order,
                    "control": args.profile["control"],
                    "treatment": args.profile["treatment"],
                    "profile": args.profile,
                    "timeout": args.trial_timeout,
                    "status": "plan_only_not_runtime_validation",
                },
                indent=2,
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    _require(not any(args.output.iterdir()), "output directory must be new or empty")
    run_id = str(uuid.uuid4())
    summary = {
        "schema_version": 1,
        "status": "invalid",
        "trials": [],
        "run_id": run_id,
        "order": order,
        "selector": SELECTOR,
        "control": args.profile["control"],
        "treatment": args.profile["treatment"],
        "profile": args.profile,
        "hostname": socket.gethostname(),
        "driver_sha256": _sha(Path(__file__)),
        "interpretation": "Observations only; two trials per arm cannot establish flake causality.",
    }
    _save(args.output / "summary.json", summary)
    try:
        provenance = json.loads(args.provenance.read_text())
        _validate_provenance(args.wheel, provenance, args.profile)
        summary["dependencies"] = _validate_dependencies(provenance)
        os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
        summary["gpus"] = _gpu_inventory()
        summary["models"] = _models(args.models_root)
        summary["accuracy_inputs"] = _accuracy_inputs(args)
        summary["identity"] = _prepare_runtime(args, provenance)
        summary["slurm"] = {
            name: os.environ.get(name)
            for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_NODELIST")
        }
        _save(args.output / "manifest.json", summary)
        for index, arm in enumerate(order, 1):
            trial = _run_trial(args, arm, index, summary["identity"], run_id)
            summary["trials"].append(trial)
            _save(args.output / "summary.json", summary)
            if not trial.get("cleanup", {}).get("clean"):
                break
        statuses = [trial["status"] for trial in summary["trials"]]
        summary["status"] = (
            "invalid"
            if "invalid" in statuses or len(statuses) != len(order)
            else "fail"
            if "fail" in statuses
            else "pass"
        )
    except (ValueError, OSError, subprocess.SubprocessError, StopIteration, SyntaxError) as error:
        summary["reason"] = str(error)
    finally:
        summary["finished"] = time.time()
        _save(args.output / "summary.json", summary)
    return {"pass": 0, "fail": 1, "invalid": 2}[summary["status"]]


if __name__ == "__main__":
    sys.exit(_main())
