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

import argparse
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

from build_runtime import runtime_identity

CONTROL = "0f2c3a95f9415045bdf06a7230759475692483b6"
TREATMENT = "3245fc3ecd76e2fb610f42f2422102e2430c28fe"
EXECUTOR = "tensorrt_llm/_torch/pyexecutor/py_executor.py"
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


def _validate_provenance(wheel: Path, provenance: dict) -> None:
    _require(provenance.get("status") == "built", "baseline build did not finish successfully")
    _require(provenance.get("source_unchanged_after_build") is True, "source changed during build")
    _require(provenance.get("source_sha") == CONTROL, "wheel source SHA is not the pinned control")
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
    _validate_provenance(args.wheel, provenance)
    _require(
        shutil.disk_usage(args.output).free > args.wheel.stat().st_size * 4 + 20 * 1024**3,
        "insufficient free storage for the installed wheel and retained request/worker logs",
    )
    git = ["git", "-C", str(args.harness)]
    changed = _command(git + ["diff", "--name-only", CONTROL, TREATMENT])
    _require(
        set(changed.splitlines())
        == {EXECUTOR, "tests/unittest/_torch/executor/test_disagg_inflight_cancel_gate.py"},
        "comparison has unexpected source changes",
    )
    blobs = {
        arm: subprocess.check_output(git + ["show", f"{sha}:{EXECUTOR}"])
        for arm, sha in (("control", CONTROL), ("treatment", TREATMENT))
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
    _require(
        (base / EXECUTOR).read_bytes() == blobs["control"], "wheel Python is not control source"
    )
    _require(not list(base.rglob("*.pyc")), "wheel contains stale bytecode")
    binaries = {
        str(path.relative_to(base)): _sha(path)
        for path in base.rglob("*")
        if path.is_file() and (".so" in path.name or path.suffix in (".a", ".cubin", ".fatbin"))
    }
    _require(bool(binaries), "wheel has no compiled artifacts")
    for arm, blob in blobs.items():
        target = base.parent / arm
        shutil.copytree(base, target, copy_function=os.link)
        destination = target / EXECUTOR
        destination.unlink()  # Other files are hardlinked; never write through to the baseline.
        destination.write_bytes(blob)
        for relative, digest in binaries.items():
            _require(
                _sha(target / relative) == digest, f"compiled artifact differs in {arm}: {relative}"
            )
        _require(
            (target / "bin" / "trtllm-serve").is_file(), "wheel lacks trtllm-serve entry point"
        )
    return {
        "provenance": provenance,
        "install_command": install,
        "binaries": binaries,
        "executor_sha256": {arm: hashlib.sha256(blob).hexdigest() for arm, blob in blobs.items()},
        "harness_sha": _command(git + ["rev-parse", "HEAD"]),
        "harness_diff": _command(git + ["diff", "HEAD"]),
        "harness_files": {
            str(path.relative_to(args.harness)): _sha(path)
            for path in [
                args.harness / "tests/integration/defs/disaggregated/test_disaggregated.py",
                args.harness / "tests/integration/defs/conftest.py",
            ]
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


def _runtime_preflight(runtime: Path, expected_hash: str, env: dict, log: Path) -> dict:
    code = """
import hashlib, importlib.metadata, json, pathlib
import tensorrt_llm
import tensorrt_llm._torch.pyexecutor.py_executor as executor
import tensorrt_llm.bindings as bindings
print('AB_RUNTIME=' + json.dumps({'package': str(pathlib.Path(tensorrt_llm.__file__).resolve()),
 'executor': str(pathlib.Path(executor.__file__).resolve()),
 'executor_sha256': hashlib.sha256(pathlib.Path(executor.__file__).read_bytes()).hexdigest(),
 'bindings': str(pathlib.Path(bindings.__file__).resolve()),
 'versions': {name: importlib.metadata.version(name) for name in
 ('tensorrt-llm', 'torch', 'aiperf', 'nixl-cu13', 'pytest')}}))
"""
    try:
        text = _command(
            [sys.executable, "-c", code], cwd=runtime, env=env, stderr=subprocess.STDOUT
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
    for field in ("package", "executor", "bindings"):
        _require(
            Path(data[field]).is_relative_to(runtime), f"{field} imported outside selected runtime"
        )
    _require(
        data["executor_sha256"] == expected_hash, "imported executor differs from pinned source"
    )
    _require(data["versions"]["aiperf"] == "0.8.0", "AIPerf must match the verified 0.8.0 schema")
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


def _classify(trial: Path, rc: int, timed_out: bool) -> dict:
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
    accounting = _accounting(profiles[0])
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
            runtime, identity["executor_sha256"][arm], env, trial / "preflight.log"
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
        result.update(_classify(trial, result["pytest_returncode"], result["timed_out"]))
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
    parser.add_argument("--order", default="control,treatment,treatment,control")
    parser.add_argument("--trial-timeout", type=int, default=12600)
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args()
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
                    "control": CONTROL,
                    "treatment": TREATMENT,
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
        "control": CONTROL,
        "treatment": TREATMENT,
        "hostname": socket.gethostname(),
        "driver_sha256": _sha(Path(__file__)),
        "interpretation": "Observations only; two trials per arm cannot establish flake causality.",
    }
    _save(args.output / "summary.json", summary)
    try:
        provenance = json.loads(args.provenance.read_text())
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
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        summary["reason"] = str(error)
    finally:
        summary["finished"] = time.time()
        _save(args.output / "summary.json", summary)
    return {"pass": 0, "fail": 1, "invalid": 2}[summary["status"]]


if __name__ == "__main__":
    sys.exit(_main())
