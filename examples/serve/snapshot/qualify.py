# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run isolated, real Snapshot startup experiments without Kubernetes or Dynamo.

This is a qualification instrument, not a production serving controller. Run it
only on an authorized Linux Docker host. Its privileged helper can access the
host runtime; workload containers remain nonprivileged and network-isolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from host_io import device_delta, sample_device
from private_client import validate_address

SNAPSHOT_REVISION = "4b1300a82c6676df020c201fa884cc602251e12e"
LABEL = "com.nvidia.trtllm.snapshot.attempt"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}")
HERE = Path(__file__).resolve().parent
REFERENCE_APP_SHA256 = "d61c3c4402bf98599b16cbe0247a70dc52d1deb0f7b3fbdfc58780f71f61f6aa"
REFERENCE_IMAGE = (
    "nvcr.io/nvidia/tensorrt-llm/release@sha256:"
    "16a103b8b1b682d287e8043fc674d23fa52d5b5f2127da913bf6c0643db3a073"
)
BROKER_IMAGE = (
    "ghcr.io/ai-dynamo/snapshot/pagebroker@sha256:"
    "bf75b039ca231d2fd81cf41291e1a11e26f3b6628788a8d22ce292327505eac6"
)


def _decode_output(output: str | bytes | None) -> str:
    """Normalize command output, including bytes retained by TimeoutExpired.

    Args:
        output: Complete or partial subprocess stream.

    Returns:
        Text suitable for diagnostic logs and JSON evidence.
    """
    return output.decode(errors="replace") if isinstance(output, bytes) else output or ""


def write_json(path: Path, value: Any) -> None:
    """Atomically publish an evidence record.

    Args:
        path: File under the experiment directory.
        value: JSON-compatible value.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def hash_file(path: Path) -> str:
    """Hash file contents without loading a model or artifact into memory.

    Args:
        path: Regular input file.

    Returns:
        SHA-256 content digest.
    """
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()


def inventory(root: Path) -> dict[str, Any]:
    """Inventory immutable files and distinguish known GPU device payloads.

    Args:
        root: Artifact or model directory.

    Returns:
        File hashes and logical/allocated sizes. Other bytes are not called CPU
        bytes: conventional CUDA checkpointing can store GPU payloads in them.

    Raises:
        ValueError: If a file is a symlink outside the inventory root.
    """
    files = []
    for path in sorted(root.rglob("*")):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Inventory escapes root: {path}")
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": stat.st_size,
                "allocated_bytes": stat.st_blocks * 512,
                "sha256": hash_file(path),
                "kind": "gpu_device_payload"
                if re.fullmatch(r"device-\d+\.bin", path.name)
                else "process_or_metadata",
            }
        )
    return {
        "files": files,
        "logical_bytes": sum(f["bytes"] for f in files),
        "allocated_bytes": sum(f["allocated_bytes"] for f in files),
        "gpu_device_payload_bytes": sum(
            f["bytes"] for f in files if f["kind"] == "gpu_device_payload"
        ),
        "cpu_only_payload_bytes": None,
        "cpu_only_payload_reason": (
            "The artifact format does not separate conventional CUDA offload from process images."
        ),
    }


def serve_config(tp: int) -> dict[str, Any]:
    """Build the bounded native V2 profile shared by cold/resident/compact runs.

    Args:
        tp: One or two MPI tensor-parallel ranks on one node.

    Returns:
        JSON-compatible serving YAML content.

    Raises:
        ValueError: If the requested profile has not been implemented.
    """
    if tp not in (1, 2):
        raise ValueError("Only single-node MPI TP1 and TP2 are experiment profiles")
    return {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": False,
        "num_postprocess_workers": 0,
        "enable_resource_governor": False,
        "max_beam_width": 1,
        "disable_overlap_scheduler": True,
        "allreduce_strategy": "NCCL",
        "torch_compile_config": None,
        "max_batch_size": 1,
        "max_seq_len": 512,
        "max_num_tokens": 1024,
        "enable_chunked_prefill": False,
        "kv_cache_config": {
            "use_kv_cache_manager_v2": True,
            "enable_block_reuse": False,
            "max_tokens": 4096,
            "host_cache_size": 0,
        },
        "cuda_graph_config": {"enable_padding": True, "batch_sizes": [1]},
        "sleep_config": {"restore_modes": {"kv_cache": "NONE"}},
    }


def validate_cohort(records: list[dict[str, Any]], attempt: str, tp: int) -> None:
    """Require one fresh observation from every participating rank.

    Args:
        records: Per-rank observations.
        attempt: Current operation nonce.
        tp: Required fixed cohort size.

    Raises:
        ValueError: If records are stale, duplicated or incomplete.
    """
    if (
        len(records) != tp
        or sorted(item["rank"] for item in records) != list(range(tp))
        or any(item["attempt"] != attempt for item in records)
    ):
        raise ValueError("Missing, duplicate or stale rank evidence")


def validate_reuse(
    captured: list[dict[str, Any]], counters: list[dict[str, Any]], attempt: str, tp: int
) -> None:
    """Require actual restored graph replay without repeating startup work.

    Args:
        captured: Positive per-rank counters from the warmed template.
        counters: Counters after private restored generation.
        attempt: Fresh restore attempt.
        tp: Required fixed cohort size.

    Raises:
        ValueError: If any rank lacks replay or repeats load/transform/capture.
    """
    validate_cohort(counters, attempt, tp)
    baseline = {item["rank"]: item["counts"] for item in captured}
    for record in counters:
        before, after = baseline[record["rank"]], record["counts"]
        for event in (
            "model_load",
            "model_reload",
            "weight_population",
            "post_load_walk",
            "staged_transform_walk",
            "graph_capture",
        ):
            if after.get(event, 0) != before.get(event, 0):
                raise ValueError(f"Rank {record['rank']} repeated {event}")
        if after.get("graph_replay", 0) <= before.get("graph_replay", 0):
            raise ValueError(f"Rank {record['rank']} has no restored graph replay")


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize completed matched trials without hiding failed attempts.

    Args:
        trials: All started trials, including failed and partial attempts.

    Returns:
        Counts and nearest-rank percentiles. Samples include observer overhead.
    """
    result: dict[str, Any] = {
        "attempted_trials": len(trials),
        "passed_trials": sum(trial["status"] == "PASS" for trial in trials),
        "percentile_method": "nearest-rank; instrumented wall time, no overhead subtraction",
        "policies": {},
    }
    for policy in ("resident", "compact"):
        samples: dict[str, list[float]] = {}
        for trial in trials:
            if trial.get("policy") != policy or trial["status"] != "PASS":
                continue
            for mode in ("cold", "restore", "native_cycle"):
                if mode in trial:
                    samples.setdefault(mode, []).append(
                        trial[mode]["first_correct_completion_seconds"]
                    )
            if "restore" in trial:
                samples.setdefault("cold_over_restore_ratio", []).append(
                    trial["cold"]["first_correct_completion_seconds"]
                    / trial["restore"]["first_correct_completion_seconds"]
                )
        result["policies"][policy] = {
            name: {
                "count": len(values),
                "p50": sorted(values)[math.ceil(len(values) * 0.5) - 1],
                "p95": sorted(values)[math.ceil(len(values) * 0.95) - 1],
            }
            for name, values in samples.items()
        }
    return result


class Experiment:
    """Own the bounded Docker resources for one qualification invocation."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize a new private output directory and immutable run identity.

        Args:
            args: Validated CLI arguments.
        """
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.run_id = uuid.uuid4().hex
        self.containers: dict[str, str] = {}
        self.images: dict[str, str] = {}
        self.report: dict[str, Any] = {
            "status": "RUNNING",
            "run": self.run_id,
            "snapshot_revision": SNAPSHOT_REVISION,
            "trials": [],
            "profiles": {
                "multi_node": "UNSUPPORTED_BY_SNAPSHOT_PIN",
                "external_kv_transfer": "BLOCKED_PENDING_RETIREMENT_AND_RECONNECT",
            },
        }
        for name in ("checkpoints", "pagebroker/control", "pagebroker/staging", "logs"):
            (self.root / name).mkdir(parents=True)

    def command(
        self,
        argv: list[str],
        *,
        timeout: int | None = None,
        data: str | None = None,
        log: Path | None = None,
    ) -> str:
        """Run a bounded command and retain diagnostic output.

        Args:
            argv: Exact executable and arguments; no shell expansion.
            timeout: Wall-clock command deadline.
            data: Optional stdin content.
            log: Optional persistent output file.

        Returns:
            Standard output.

        Raises:
            subprocess.CalledProcessError: On nonzero exit.
            subprocess.TimeoutExpired: On deadline expiry.
        """
        try:
            result = subprocess.run(
                argv,
                input=data,
                text=True,
                capture_output=True,
                timeout=timeout or self.args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            if log is not None:
                log.write_text(
                    _decode_output(error.output)
                    + "\n--- stderr ---\n"
                    + _decode_output(error.stderr)
                    + f"\n--- timed out after {error.timeout} seconds ---\n"
                )
            raise
        if log is not None:
            log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, argv, output=result.stdout, stderr=result.stderr
            )
        return result.stdout.strip()

    def image(self, reference: str) -> str:
        """Resolve a locally available immutable image without pulling a mutable tag.

        Args:
            reference: Immutable Docker image ID or repository digest.

        Returns:
            Locally verified image ID.

        Raises:
            ValueError: If the image input is mutable.
        """
        if not (IMAGE_ID.fullmatch(reference) or re.search(r"@sha256:[0-9a-f]{64}$", reference)):
            raise ValueError("Use immutable image IDs or repository digests, not tags")
        info = json.loads(self.command(["docker", "image", "inspect", reference]))[0]
        self.report.setdefault("images", {})[reference] = info
        return info["Id"]

    def launch(self, options: list[str], image: str, command: list[str], attempt: str) -> str:
        """Create and start an owned container with no automatic restart.

        Args:
            options: Explicit Docker options.
            image: Immutable image ID.
            command: Container process arguments.
            attempt: Ownership nonce used for cleanup.

        Returns:
            Full immutable container ID.
        """
        container = self.command(
            [
                "docker",
                "create",
                "--restart=no",
                "--label",
                f"{LABEL}={attempt}",
                *options,
                image,
                *command,
            ]
        )
        self.containers[container] = attempt
        self.command(["docker", "start", container])
        return container

    def cleanup(self, container: str) -> None:
        """Stop and remove only an immutable container owned by this invocation.

        Args:
            container: Previously created container ID.

        Raises:
            RuntimeError: If ownership changed or cleanup fails.
        """
        info = json.loads(self.command(["docker", "inspect", container], timeout=30))[0]
        if (
            info["Id"] != container
            or info["Config"]["Labels"].get(LABEL) != self.containers[container]
        ):
            raise RuntimeError("Cleanup ownership mismatch")
        try:
            logs = subprocess.run(
                ["docker", "logs", container], capture_output=True, text=True, timeout=30
            )
            (self.root / "logs" / f"{container}.log").write_text(logs.stdout + logs.stderr)
        finally:
            self.command(["docker", "rm", "--force", container], timeout=60)
            del self.containers[container]

    def wait(self, predicate: Callable[[], bool], description: str, container: str) -> float:
        """Wait for evidence while rejecting early container failure.

        Args:
            predicate: Evidence check without side effects.
            description: Timeout explanation.
            container: Owned workload whose liveness is required.

        Returns:
            Elapsed wall-clock seconds.

        Raises:
            RuntimeError: If the workload exits.
            TimeoutError: If evidence does not arrive.
        """
        start = time.monotonic()
        while time.monotonic() - start < self.args.timeout:
            if predicate():
                return time.monotonic() - start
            state = json.loads(
                self.command(
                    ["docker", "inspect", "--format", "{{json .State}}", container], timeout=30
                )
            )
            if not state["Running"]:
                raise RuntimeError(f"Workload exited while waiting for {description}: {state}")
            time.sleep(0.2)
        raise TimeoutError(description)

    def mounts(self) -> list[str]:
        """Build the host-helper mounts required by the pinned upstream executor.

        Returns:
            Docker volume options. Host views are read-only where possible.
        """
        return [
            "--volume",
            "/proc:/host/proc",
            "--volume",
            "/sys/fs/cgroup:/sys/fs/cgroup",
            "--volume",
            "/sys/kernel/security:/sys/kernel/security:ro",
            "--volume",
            "/var/run/docker.sock:/var/run/docker.sock",
            "--volume",
            f"{self.args.containerd_socket}:{self.args.containerd_socket}",
            "--volume",
            f"{self.docker_root}:{self.docker_root}:ro",
            "--volume",
            f"{self.root / 'checkpoints'}:/checkpoints",
            "--volume",
            f"{self.root / 'pagebroker'}:/pagebroker",
        ]

    def host(
        self, operation: str, container: str, attempt: str, artifact: str, trial_dir: Path
    ) -> dict[str, Any]:
        """Invoke the pinned host executor with an independent outer watchdog.

        Args:
            operation: Inspect, capture, or restore.
            container: Owned target container.
            attempt: Current immutable target ownership.
            artifact: Source artifact identity.
            trial_dir: Evidence destination.

        Returns:
            Upstream-backed host result and operation wall time.
        """
        request = {
            "operation": operation,
            "container": container,
            "attempt": attempt,
            "artifact": artifact,
            "image": self.images["workload"],
            "containerd_socket": self.args.containerd_socket,
            "timeout_seconds": self.args.timeout,
            "interpose": True,
        }
        helper = self.command(
            [
                "docker",
                "create",
                "--interactive",
                "--restart=no",
                "--label",
                f"{LABEL}={attempt}",
                "--privileged",
                "--pid=host",
                "--ipc=host",
                "--cgroupns=host",
                "--network=none",
                "--gpus",
                f'"device={self.gpus}"',
                "--env",
                f"NVIDIA_VISIBLE_DEVICES={self.gpus}",
                *self.mounts(),
                self.images["host"],
            ]
        )
        self.containers[helper] = attempt
        start = time.monotonic()
        io_before = sample_device(self.root / "checkpoints")
        try:
            output = self.command(
                ["docker", "start", "--attach", "--interactive", helper],
                timeout=self.args.timeout + 30,
                data=json.dumps(request),
                log=trial_dir / f"{operation}-host.log",
            )
            result = json.loads(output)
            if result["status"] != "PASS" or result["attempt"] != attempt:
                raise RuntimeError("Host operation returned invalid evidence")
            result["wall_seconds"] = time.monotonic() - start
            write_json(trial_dir / f"{operation}-host.json", result)
            return result
        finally:
            io_after = sample_device(self.root / "checkpoints")
            try:
                write_json(
                    trial_dir / f"{operation}-device-io.json", device_delta(io_before, io_after)
                )
            finally:
                self.cleanup(helper)

    def workload(self, control: Path, attempt: str, *, inert: bool = False) -> str:
        """Launch a network-isolated native MPI server or inert restore target.

        Args:
            control: Fresh per-attempt control mount.
            attempt: Ownership nonce.
            inert: Whether to launch only the restore placeholder.

        Returns:
            Owned container ID.
        """
        env = {
            "NVIDIA_VISIBLE_DEVICES": self.gpus,
            "TRTLLM_SNAPSHOT_EXPERIMENT": "1",
            "SNAPSHOT_CONTROL_DIR": "/snapshot-control",
            "PYTHONPATH": "/qualification",
            "HF_HUB_OFFLINE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "OMPI_MCA_pml": "ob1",
            "OMPI_MCA_btl": "tcp,self",
            "OMPI_MCA_btl_tcp_if_include": "lo",
            "OMPI_MCA_oob_tcp_if_include": "lo",
            "PRTE_MCA_oob_tcp_if_include": "lo",
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "TLLM_WORKER_USE_SINGLE_PROCESS": "0",
            "TRTLLM_NO_USAGE_STATS": "1",
            "TLLM_CUDA_GRAPH_STRICT_BUFFERS": "1",
            "TLLM_NCCL_SYMMETRIC_ZERO_COPY": "0",
            "TRTLLM_PROXY_BATCH_REQUESTS": "0",
        }
        options = [
            "--network=none",
            "--gpus",
            f'"device={self.gpus}"',
            "--shm-size=1g",
            "--volume",
            f"{control}:/snapshot-control",
            "--volume",
            f"{HERE}:/qualification:ro",
            "--volume",
            f"{self.args.model.resolve()}:/model:ro",
            "--volume",
            f"{self.root / 'interpose'}:/tmp/snapshot-cuda:ro",
            "--entrypoint",
            "/bin/sleep" if inert else "/tmp/snapshot-cuda/cuinterpose-launch",
        ]
        for key, value in env.items():
            options.extend(["--env", f"{key}={value}"])
        options.extend(["--cpus", "16", "--memory", "128g", "--memory-swap", "128g"])
        if self.args.reference_app:
            options.extend(
                [
                    "--volume",
                    f"{self.args.reference_app.resolve()}:/reference/app.py:ro",
                    "--env",
                    "TRTLLM_SNAPSHOT_EXPERIMENT=0",
                    "--env",
                    "TRTLLM_FREE_GPU_MEMORY_FRACTION=0.75",
                ]
            )
        command = (
            ["infinity"]
            if inert
            else [
                "--library",
                "/tmp/snapshot-cuda/libcuinterpose.so",
                "--",
                "trtllm-serve",
                "/model",
                "--config",
                "/snapshot-control/serve.yaml",
                "--num_serve_frontends",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--report_addr",
                "/snapshot-control/address",
            ]
        )
        if self.args.reference_app and not inert:
            command = [
                "--library",
                "/tmp/snapshot-cuda/libcuinterpose.so",
                "--",
                "python3",
                "/qualification/reference_workload.py",
            ]
        if self.args.native_only:
            options[options.index("--entrypoint") + 1] = "trtllm-serve"
            command = command[4:]
        return self.launch(options, self.images["workload"], command, attempt)

    def records(self, control: Path, prefix: str, attempt: str) -> list[dict[str, Any]]:
        """Read and validate a complete set of rank evidence.

        Args:
            control: Current control mount.
            prefix: Evidence kind.
            attempt: Expected fresh nonce.

        Returns:
            Validated observations in rank order.
        """
        records = [
            json.loads((control / f"{prefix}-{rank}.json").read_text())
            for rank in range(self.args.tp)
        ]
        validate_cohort(records, attempt, self.args.tp)
        return records

    def artifact_inventory(self, artifact: str) -> dict[str, Any]:
        """Hash root-owned CRIU files without changing their ownership or permissions.

        Args:
            artifact: Current source artifact ID.

        Returns:
            File inventory read inside a nonprivileged CPU-only container.
        """
        path = self.root / "checkpoints" / "artifacts" / artifact / "containers" / "main"
        reader = self.launch(
            [
                "--network=none",
                "--read-only",
                "--user",
                "0",
                "--env",
                "TRTLLM_SNAPSHOT_EXPERIMENT=0",
                "--mount",
                f"type=bind,src={path},dst=/artifact,readonly",
                "--volume",
                f"{HERE}:/qualification:ro",
                "--entrypoint",
                "python3",
            ],
            self.images["workload"],
            ["-S", "/qualification/inventory_artifact.py"],
            self.run_id,
        )
        try:
            code = self.command(["docker", "wait", reader])
            if code != "0":
                raise RuntimeError("Artifact inventory failed")
            return json.loads(self.command(["docker", "logs", reader]))
        finally:
            self.cleanup(reader)

    def negative_preflight(self) -> None:
        """Exercise ownership and stale-target rejection against the real host driver.

        Raises:
            RuntimeError: If a negative case succeeds, fails for the wrong reason,
                or mutates the destination before rejection.
        """
        for kind in ("wrong_owner", "stale_target"):
            attempt = uuid.uuid4().hex
            control = self.root / f"negative-{kind}"
            control.mkdir(mode=0o700)
            write_json(control / "experiment.json", {"attempt": attempt})
            target = self.workload(control, attempt, inert=True)
            record = {"case": kind, "attempt": attempt, "status": "RUNNING"}
            self.report.setdefault("negative_tests", []).append(record)
            try:
                if kind == "stale_target":
                    (control / "restore-complete").write_text("pid=2\n")
                operation = "inspect" if kind == "wrong_owner" else "restore"
                expected = "ownership" if kind == "wrong_owner" else "stale restore-complete"
                try:
                    self.host(
                        operation,
                        target,
                        uuid.uuid4().hex if kind == "wrong_owner" else attempt,
                        uuid.uuid4().hex,
                        control,
                    )
                except subprocess.CalledProcessError as error:
                    response = json.loads(error.output)
                    if response.get("status") != "FAIL" or expected not in response.get(
                        "error", ""
                    ):
                        raise RuntimeError(
                            f"{kind} failed for an unrelated reason: {response}"
                        ) from error
                    record["rejection"] = response
                else:
                    raise RuntimeError(f"Negative test {kind} unexpectedly succeeded")
                state = json.loads(
                    self.command(["docker", "inspect", "--format", "{{json .State}}", target])
                )
                if not state["Running"]:
                    raise RuntimeError("Preflight rejection mutated the target")
                record["status"] = "PASS"
            finally:
                self.cleanup(target)

    def generate(self, container: str, control: Path) -> dict[str, Any]:
        """Make a private deterministic completion through the actual HTTP server.

        Args:
            container: Isolated native server.
            control: Its active control directory.

        Returns:
            Validated response content, token usage and measured request time.
        """
        self.wait(lambda: (control / "address").is_file(), "HTTP bound address", container)
        output = self.command(
            ["docker", "exec", container, "python3", "-S", "/qualification/private_client.py"],
            timeout=120,
            log=control / "generation.log",
        )
        result = json.loads(output)
        write_json(control / "generation.json", result)
        return result

    def trial(self, index: int, policy: str) -> None:
        """Run a matched cold, source-capture and fresh restore experiment.

        Args:
            index: Repetition index.
            policy: Resident or compact KV treatment.
        """
        trial_dir = self.root / f"{index}-{policy}"
        trial_dir.mkdir()
        trial: dict[str, Any] = {"index": index, "policy": policy, "status": "RUNNING"}
        self.report["trials"].append(trial)
        template = uuid.uuid4().hex
        captured = []
        cold = None
        captured_address = None
        for mode in ("cold", "capture", "restore"):
            control = trial_dir / mode
            control.mkdir(mode=0o700)
            attempt = uuid.uuid4().hex
            config = {
                "mode": mode,
                "attempt": attempt,
                "template": template,
                "tp": self.args.tp,
                "policy": policy,
                "profile": f"dense-qwen3-mpi-tp{self.args.tp}",
                "timeout_seconds": self.args.timeout * 3,
            }
            write_json(control / "experiment.json", config)
            write_json(control / "serve.yaml", serve_config(self.args.tp))
            if mode == "restore":
                if captured_address is None:
                    raise RuntimeError("Capture did not publish an HTTP endpoint")
                (control / "address").write_text(captured_address + "\n")
            start = time.monotonic()
            container = self.workload(control, attempt, inert=mode == "restore")
            trial[mode] = {"attempt": attempt, "container": container, "status": "RUNNING"}
            try:
                if mode == "restore":
                    trial[mode]["host"] = self.host(mode, container, attempt, template, control)
                prefix = "capture" if mode == "capture" else "runtime"
                self.wait(
                    lambda: all(
                        (control / f"{prefix}-{rank}.json").exists() for rank in range(self.args.tp)
                    ),
                    f"all-rank {prefix}",
                    container,
                )
                observations = self.records(control, prefix, attempt)
                trial[mode]["rank_ready_seconds"] = time.monotonic() - start
                if mode == "capture":
                    captured = observations
                    captured_address = validate_address((control / "address").read_text())
                    (control / "ready-for-snapshot").touch()
                    trial[mode]["host"] = self.host(mode, container, attempt, template, control)
                    if not trial[mode]["host"]["source_terminated"]:
                        raise RuntimeError("Source was not terminated")
                    trial["artifact"] = self.artifact_inventory(template)
                    if not trial["artifact"]["files"]:
                        raise RuntimeError("Capture produced no artifact")
                    write_json(trial_dir / "artifact.json", trial["artifact"])
                else:
                    result = self.generate(container, control)
                    trial[mode]["first_correct_completion_seconds"] = time.monotonic() - start
                    if mode == "cold":
                        cold = result["completion"]
                    else:
                        if result["completion"] != cold:
                            raise RuntimeError(
                                "Restored completion differs from matched cold output"
                            )
                        validate_reuse(
                            captured,
                            self.records(control, "counters", attempt),
                            attempt,
                            self.args.tp,
                        )
                        self.records(control, "memory", attempt)
                trial[mode]["status"] = "PASS"
                write_json(self.root / "report.json", self.report)
            finally:
                self.cleanup(container)
        trial["status"] = "PASS"

    def reference_trial(self, index: int) -> None:
        """Reproduce the pinned upstream LLM/V1 workload with real host restore.

        Args:
            index: Repetition number. This is not native V2 qualification.
        """
        trial_dir = self.root / f"{index}-upstream-reference"
        trial_dir.mkdir()
        template = uuid.uuid4().hex
        trial: dict[str, Any] = {
            "kind": "upstream_llm_v1_reference",
            "index": index,
            "status": "RUNNING",
        }
        self.report["trials"].append(trial)
        for mode in ("capture", "restore"):
            attempt = uuid.uuid4().hex
            control = trial_dir / mode
            control.mkdir(mode=0o700)
            write_json(control / "experiment.json", {"attempt": attempt, "template": template})
            start = time.monotonic()
            container = self.workload(control, attempt, inert=mode == "restore")
            trial[mode] = {"attempt": attempt, "container": container}
            try:
                if mode == "capture":
                    self.wait(
                        lambda: (control / "ready-for-snapshot").exists(),
                        "upstream warmed source",
                        container,
                    )
                    trial[mode]["warm_seconds"] = time.monotonic() - start
                trial[mode]["host"] = self.host(mode, container, attempt, template, control)
                if mode == "restore":
                    self.wait(
                        lambda: (control / "trtllm-restore-ready").exists(),
                        "upstream restored generation",
                        container,
                    )
                    output = self.command(
                        [
                            "docker",
                            "exec",
                            container,
                            "python3",
                            "-S",
                            "/qualification/private_client.py",
                            "--reference",
                        ],
                        timeout=240,
                        log=control / "generation.log",
                    )
                    trial[mode]["semantic_validation"] = json.loads(output)
                    trial[mode]["first_correct_completion_seconds"] = time.monotonic() - start
                else:
                    if not trial[mode]["host"].get("source_terminated"):
                        raise RuntimeError("Upstream source survived capture")
                    trial["artifact"] = self.artifact_inventory(template)
                    if not trial["artifact"]["files"]:
                        raise RuntimeError("Capture produced no artifact")
                write_json(self.root / "report.json", self.report)
            finally:
                self.cleanup(container)
        trial["status"] = "PASS"

    def native_trial(self, index: int, policy: str) -> None:
        """Test native observers and KV cycles without any process checkpointing.

        Args:
            index: Repetition index.
            policy: Resident or compact native KV treatment.
        """
        trial = {
            "index": index,
            "policy": policy,
            "status": "RUNNING",
            "process_restore": "NOT_RUN",
        }
        self.report["trials"].append(trial)
        cold = None
        for mode in ("cold", "native_cycle"):
            control = self.root / f"{index}-{policy}-{mode}"
            control.mkdir(mode=0o700)
            attempt = uuid.uuid4().hex
            write_json(
                control / "experiment.json",
                {
                    "mode": mode,
                    "attempt": attempt,
                    "template": attempt,
                    "tp": self.args.tp,
                    "policy": policy,
                    "profile": f"dense-qwen3-mpi-tp{self.args.tp}",
                    "timeout_seconds": self.args.timeout,
                },
            )
            write_json(control / "serve.yaml", serve_config(self.args.tp))
            start = time.monotonic()
            container = self.workload(control, attempt)
            try:
                self.wait(
                    lambda: all(
                        (control / f"runtime-{rank}.json").is_file() for rank in range(self.args.tp)
                    ),
                    "native runtime",
                    container,
                )
                observations = self.records(control, "runtime", attempt)
                generated = self.generate(container, control)
                validate_reuse(
                    self.records(control, "capture", attempt),
                    self.records(control, "counters", attempt),
                    attempt,
                    self.args.tp,
                )
                if mode == "cold":
                    cold = generated["completion"]
                elif generated["completion"] != cold:
                    raise RuntimeError("Native KV cycle changed the matched completion")
                trial[mode] = {
                    "status": "PASS",
                    "ranks": observations,
                    "first_correct_completion_seconds": time.monotonic() - start,
                }
            finally:
                self.cleanup(container)
        trial["status"] = "PASS"

    def run_selected_profiles(self) -> None:
        """Execute the selected native-only or real Snapshot qualification mode."""
        names = ("workload",) if self.args.native_only else ("workload", "host", "broker")
        for name in names:
            self.images[name] = self.image(getattr(self.args, name + "_image"))
        self.report["experiment_sources"] = inventory(HERE)
        self.report["model"] = inventory(self.args.model.resolve())
        if not self.report["model"]["files"]:
            raise ValueError("Model directory is empty")
        if self.args.native_only:
            self.report["process_restore"] = "NOT_RUN"
            (self.root / "interpose").mkdir()
            for index in range(self.args.repeats):
                for policy in ("resident", "compact"):
                    self.native_trial(index, policy)
            self.report["status"] = "NATIVE_ONLY_PASS"
            return
        if self.image(BROKER_IMAGE) != self.images["broker"]:
            raise ValueError("PageBroker image differs from the pinned Snapshot build")
        if self.args.reference_app:
            if self.args.tp != 1 or hash_file(self.args.reference_app) != REFERENCE_APP_SHA256:
                raise ValueError("Reference requires the unchanged pinned app.py and TP1")
            if self.image(REFERENCE_IMAGE) != self.images["workload"]:
                raise ValueError("Reference requires the pinned upstream 1.3.0rc24 image")
            self.report["reference"] = {
                "app_sha256": REFERENCE_APP_SHA256,
                "scope": (
                    "Unchanged LLM/V1 application on standalone adapter; "
                    "not Kubernetes deployment reproduction"
                ),
                "model_binding": "read-only local model",
                "free_gpu_memory_fraction": 0.75,
                "cuinterpose": True,
            }
        info = json.loads(self.command(["docker", "image", "inspect", self.images["host"]]))[0]
        if info["Config"]["Labels"].get("com.nvidia.trtllm.snapshot.revision") != SNAPSHOT_REVISION:
            raise ValueError("Host driver source revision mismatch")
        payload = self.root / "interpose"
        payload.mkdir()
        export = self.command(
            ["docker", "create", "--label", f"{LABEL}={self.run_id}", self.images["host"]]
        )
        self.containers[export] = self.run_id
        for source in (
            "/usr/local/lib/snapshot/libcuinterpose.so",
            "/usr/local/lib/snapshot/libcuinterpose_core.so",
            "/usr/local/bin/cuinterpose-launch",
        ):
            self.command(["docker", "cp", f"{export}:{source}", str(payload / Path(source).name)])
        self.cleanup(export)
        self.report["interpose_payload"] = inventory(payload)
        broker = self.launch(
            [
                "--privileged",
                "--pid=host",
                "--ipc=host",
                "--cgroupns=host",
                "--network=none",
                "--gpus",
                f'"device={self.gpus}"',
                "--env",
                f"NVIDIA_VISIBLE_DEVICES={self.gpus}",
                "--volume",
                f"{self.root / 'pagebroker'}:/pagebroker",
                "--volume",
                f"{self.root / 'checkpoints'}:/checkpoints",
            ],
            self.images["broker"],
            [
                "/pagebroker/control/pagebroker.sock",
                "/pagebroker/staging",
                "/checkpoints",
                "--max-concurrent-requests",
                "8",
                "--custom-storage-buffer-count",
                "2",
                "--custom-storage-chunk-bytes",
                "67108864",
                "--custom-storage-max-pinned-bytes",
                "268435456",
            ],
            self.run_id,
        )
        self.wait(
            lambda: (self.root / "pagebroker/control/pagebroker.sock").exists(),
            "PageBroker socket",
            broker,
        )
        self.negative_preflight()
        for index in range(self.args.repeats):
            if self.args.reference_app:
                self.reference_trial(index)
            else:
                for policy in ("resident", "compact"):
                    self.trial(index, policy)
        self.report["status"] = "PASS"

    def run(self) -> int:
        """Run qualification and always preserve failures and owned-resource cleanup.

        Returns:
            Zero when the selected mode passes. Native-only mode never qualifies restore.
        """
        try:
            self.gpus = ",".join(self.args.gpu)
            self.docker_root = self.command(["docker", "info", "--format", "{{.DockerRootDir}}"])
            self.report["host"] = {
                "uname": os.uname()._asdict()
                if hasattr(os.uname(), "_asdict")
                else list(os.uname()),
                "gpu": self.command(
                    ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]
                ),
                "docker_root": self.docker_root,
                "selected_gpus": self.args.gpu,
            }
            self.run_selected_profiles()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            TimeoutError,
            subprocess.SubprocessError,
        ) as error:
            self.report["status"] = "FAIL"
            self.report["error"] = str(error)
            for records in (self.report["trials"], self.report.get("negative_tests", [])):
                for trial in records:
                    if trial["status"] == "RUNNING":
                        trial["status"] = "FAIL"
                        trial["error"] = str(error)
            if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                self.report["command_output"] = _decode_output(error.output)
                self.report["command_stderr"] = _decode_output(error.stderr)
        finally:
            for container in list(self.containers):
                try:
                    self.cleanup(container)
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    self.report.setdefault("cleanup_errors", []).append(
                        {"container": container, "error": str(error)}
                    )
                    self.report["status"] = "FAIL"
            self.report["summary"] = summarize(self.report["trials"])
            write_json(self.root / "report.json", self.report)
        return 0 if self.report["status"] in ("PASS", "NATIVE_ONLY_PASS") else 1


def main() -> int:
    """Parse explicit privilege and immutable-profile inputs.

    Returns:
        Qualification exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-privileged-host-driver", action="store_true")
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Nonprivileged cold/native KV-cycle checks; does not qualify Snapshot",
    )
    for name in ("workload", "host", "broker"):
        parser.add_argument(f"--{name}-image", required=name == "workload")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-app", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="append", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--containerd-socket",
        choices=("/run/docker/containerd/containerd.sock", "/run/containerd/containerd.sock"),
        default="/run/docker/containerd/containerd.sock",
    )
    args = parser.parse_args()
    if args.native_only:
        if (
            args.reference_app
            or args.allow_privileged_host_driver
            or args.host_image
            or args.broker_image
        ):
            parser.error("native-only mode excludes Snapshot/reference/privileged options")
    elif not args.allow_privileged_host_driver or not args.host_image or not args.broker_image:
        parser.error(
            "Restore requires explicit privileged-host authorization and pinned host/broker images"
        )
    if os.uname().sysname != "Linux":
        parser.error("Run on the authorized Linux Docker host, outside the workload MPI job")
    if len(set(args.gpu)) != args.tp or any(not GPU_UUID.fullmatch(gpu) for gpu in args.gpu):
        parser.error("Supply exactly one distinct allocated GPU UUID per rank")
    if not 1 <= args.repeats <= 20 or not 1 <= args.timeout <= 7200:
        parser.error("repeats must be 1..20 and timeout 1..7200")
    return Experiment(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
