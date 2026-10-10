# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental same-host native startup capture/restore, without Kubernetes."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import snapshot_probe as probe

SNAPSHOT_REVISION = "bc9d2161d2a7f203551e5b7e237defd90b1e9aa1"
_PROFILE_KEYS = {
    "model_revision",
    "weights_digest",
    "quantization",
    "layout",
    "image_digest",
    "trtllm_revision",
    "graph_buckets",
    "topology",
}


def file_digest(path: Path) -> str:
    """Hash a file without reading the entire artifact into memory.

    Args:
        path: Regular file to hash.

    Returns:
        SHA-256 content digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tree(root: int) -> dict[int, str]:
    """Observe one Linux process tree, including PID start times for safe cleanup.

    Args:
        root: Root process owned by this operation.

    Returns:
        PIDs mapped to kernel start times; zombies are excluded.
    """
    observed = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z":
                observed[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
    result = {}
    pending = [root]
    while pending:
        pid = pending.pop()
        if pid in observed and pid not in result:
            result[pid] = observed[pid][1]
            pending.extend(child for child, (parent, _) in observed.items() if parent == pid)
    return result


def terminate_tree(owned: dict[int, str]) -> None:
    """Kill only recorded processes whose kernel identity still matches.

    Args:
        owned: Process identities recorded by this coordinator.
    """
    for pid, identity in reversed(list(owned.items())):
        if process_tree(pid).get(pid) == identity:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_tool(argv: list[str], log: Path, deadline: float) -> str:
    """Execute one real host operation with retained diagnostics and a deadline.

    Args:
        argv: Executable and arguments, without shell interpretation.
        log: Append-only command log in the artifact directory.
        deadline: Absolute monotonic deadline.

    Returns:
        Captured standard output.

    Raises:
        RuntimeError: If the host operation fails.
    """
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=probe._remaining(deadline), check=False
    )
    with log.open("a") as output:
        output.write(json.dumps(argv) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(f"Host operation failed ({result.returncode}): {argv[0]}; see {log}")
    return result.stdout.strip()


def host_identity(bin_dir: Path, log: Path, deadline: float) -> dict[str, Any]:
    """Record strict same-host compatibility and require real checkpoint tools.

    Args:
        bin_dir: Matched Snapshot/CRIU tool directory.
        log: Command log.
        deadline: Operation deadline.

    Returns:
        Host, namespace, driver and executable identities.

    Raises:
        ValueError: If the platform or tools are unsuitable.
    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Snapshot startup prototype requires Linux x86_64")
    binaries = {
        name: file_digest((bin_dir / name).resolve()) for name in ("criu", "cuda-checkpoint-helper")
    }
    run_tool([str(bin_dir / "criu"), "check"], log, deadline)
    gpu = run_tool(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        log,
        deadline,
    )
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "namespaces": {
            name: os.readlink(f"/proc/self/ns/{name}") for name in ("pid", "mnt", "net", "user")
        },
        "uid": os.getuid(),
        "kernel": platform.release(),
        "gpus": gpu,
        "binaries": binaries,
        "snapshot_revision": SNAPSHOT_REVISION,
    }


def validate_profile(profile: dict[str, Any]) -> None:
    """Require explicit profile identity; never infer compatibility from health.

    Args:
        profile: User-supplied model/build/configuration identity.

    Raises:
        ValueError: If identity or topology is incomplete or unsupported.
    """
    if not isinstance(profile, dict) or not _PROFILE_KEYS.issubset(profile):
        raise ValueError(f"Profile requires {sorted(_PROFILE_KEYS)}")
    if any(profile[key] in (None, "", [], {}) for key in _PROFILE_KEYS):
        raise ValueError("Profile identity fields must be nonempty")
    if profile["topology"] != {"nodes": 1, "tp": 1, "pp": 1, "cp": 1}:
        raise ValueError("Phase 1 host adapter supports one aggregate GPU only")


def capture(args: argparse.Namespace) -> dict[str, Any]:
    """Capture a native clean-startup tree and publish only a complete artifact.

    Args:
        args: Parsed capture CLI arguments.

    Returns:
        Complete artifact manifest.
    """
    profile = probe._read_json(args.profile)
    validate_profile(profile)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or Path(command[0]).name != "trtllm-serve" or command.count("{address}") != 1:
        raise ValueError("Expected trtllm-serve argv with exactly one {address} argument")
    artifact = args.artifact.resolve()
    artifact.mkdir(mode=0o700)
    control = artifact / "control"
    control.mkdir(mode=0o700)
    images = artifact / "images"
    images.mkdir(mode=0o700)
    deadline = time.monotonic() + args.timeout
    started = time.monotonic()
    log = artifact / "capture.log"
    identity = host_identity(args.snapshot_bin, log, deadline)
    template_id = uuid.uuid4().hex
    probe._write_report(control / "template.json", {"template_id": template_id})
    env = dict(os.environ, TRTLLM_SNAPSHOT_DIR=str(control))
    command = [str(control / "address") if item == "{address}" else item for item in command]
    owned = {}
    with (artifact / "candidate.log").open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while not (control / "capture-0.json").exists():
                probe._check_process(process)
                probe._remaining(deadline)
                time.sleep(0.05)
            rank = probe._read_json(control / "capture-0.json")
            owned = process_tree(process.pid)
            if (
                rank["template_id"] != template_id
                or rank["world_size"] != 1
                or rank["pid"] not in owned
            ):
                raise ValueError("Capture rank is not in the owned native process tree")
            cuda_pids = run_tool(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], log, deadline
            )
            cuda_pids = sorted(set(int(pid) for pid in cuda_pids.splitlines()) & owned.keys())
            if rank["pid"] not in cuda_pids:
                raise ValueError("Warmed rank is not reported as a CUDA process")
            for action in ("lock", "checkpoint"):
                for pid in cuda_pids:
                    run_tool(
                        [
                            str(args.snapshot_bin / "cuda-checkpoint-helper"),
                            "--action",
                            action,
                            "--pid",
                            str(pid),
                        ],
                        log,
                        deadline,
                    )
            run_tool(
                [
                    str(args.snapshot_bin / "criu"),
                    "dump",
                    "--tree",
                    str(process.pid),
                    "--images-dir",
                    str(images),
                    "--log-file",
                    "dump.log",
                    "--shell-job",
                    "--tcp-established",
                ],
                log,
                deadline,
            )
            process.wait(timeout=probe._remaining(deadline))
            if any(process_tree(pid).get(pid) == token for pid, token in owned.items()):
                raise RuntimeError("Source tree survived capture")
            manifest = {
                "schema_version": 1,
                "template_id": template_id,
                "profile": profile,
                "profile_digest": probe._digest(profile),
                "host": identity,
                "command": command,
                "root_pid": process.pid,
                "pids": list(owned),
                "cuda_pids": cuda_pids,
                "ranks": [rank],
                "artifact_path": str(artifact),
                "capture_seconds": time.monotonic() - started,
                "images": {
                    str(path.relative_to(images)): file_digest(path)
                    for path in images.rglob("*")
                    if path.is_file()
                },
            }
            if "inventory.img" not in manifest["images"]:
                raise RuntimeError("CRIU did not produce an image inventory")
            probe._write_report(artifact / "manifest.json", manifest)
            return manifest
        finally:
            terminate_tree({**owned, **process_tree(process.pid)})
            process.wait(timeout=10)


def restore(args: argparse.Namespace) -> dict[str, Any]:
    """Restore, validate privately, then activate one exact same-host template.

    Args:
        args: Parsed restore CLI arguments.

    Returns:
        Trial evidence. Cold fallback is never attempted.
    """
    artifact = args.artifact.resolve()
    manifest = probe._read_json(artifact / "manifest.json")
    profile = probe._read_json(args.profile)
    validate_profile(profile)
    baseline = probe._read_json(args.baseline)
    requests = probe._read_json(args.requests)
    if (
        manifest.get("schema_version") != 1
        or manifest["profile"] != profile
        or manifest["profile_digest"] != probe._digest(profile)
        or manifest["artifact_path"] != str(artifact)
    ):
        raise ValueError("Incompatible or relocated Snapshot artifact")
    if (
        baseline.get("mode") != "cold"
        or baseline.get("generation_probe_status") != "PASS"
        or baseline.get("profile_digest") != probe._digest(profile)
        or baseline.get("requests_digest") != probe._digest(requests)
    ):
        raise ValueError("Need a passing cold probe with the same profile and requests")
    deadline = time.monotonic() + args.timeout
    session = uuid.uuid4().hex
    trial = artifact / f"restore-{session}"
    trial.mkdir(mode=0o700)
    log = trial / "host.log"
    if host_identity(args.snapshot_bin, log, deadline) != manifest["host"]:
        raise ValueError("Host, GPU, driver, namespace or Snapshot tools changed")
    images = artifact / "images"
    if {
        str(path.relative_to(images)): file_digest(path)
        for path in images.rglob("*")
        if path.is_file()
    } != manifest["images"]:
        raise ValueError("Snapshot image integrity check failed")
    if any(Path(f"/proc/{pid}").exists() for pid in manifest["pids"]):
        raise ValueError("Source PID is still occupied; refuse restore")
    control = artifact / "control"
    for path in [control / name for name in ("restore.json", "activate.json", "abort.json")]:
        path.unlink(missing_ok=True)
    for path in list(control.glob("memory-*.json")) + list(control.glob("runtime-*.json")):
        path.unlink()
    token = uuid.uuid4().hex
    headers = {"X-TRTLLM-Snapshot-Validation": token}
    report = {
        "session_id": session,
        "template_id": manifest["template_id"],
        "status": "FAIL",
        "timings_seconds": {},
        "outputs": [],
        "profile_digest": manifest["profile_digest"],
    }
    owned = {}
    started = time.monotonic()
    try:
        run_tool(
            [
                str(args.snapshot_bin / "criu"),
                "restore",
                "--images-dir",
                str(images),
                "--work-dir",
                str(trial),
                "--log-file",
                "restore.log",
                "--shell-job",
                "--tcp-established",
                "--restore-detached",
                "--pidfile",
                str(trial / "pid"),
            ],
            log,
            deadline,
        )
        root = int((trial / "pid").read_text())
        owned = process_tree(root)
        if root != manifest["root_pid"]:
            raise ValueError("Same-namespace restore changed the process identity")
        if set(owned) != set(manifest["pids"]):
            raise ValueError("Restored process cohort differs from the captured cohort")
        report["timings_seconds"]["cpu_restore"] = time.monotonic() - started
        for action in ("restore", "unlock"):
            for pid in manifest["cuda_pids"]:
                run_tool(
                    [
                        str(args.snapshot_bin / "cuda-checkpoint-helper"),
                        "--action",
                        action,
                        "--pid",
                        str(pid),
                    ],
                    log,
                    deadline,
                )
        report["timings_seconds"]["cuda_restored"] = time.monotonic() - started
        probe._write_report(
            control / "restore.json",
            {
                "session_id": session,
                "template_id": manifest["template_id"],
                "validation_token": token,
            },
        )
        while True:
            probe._remaining(deadline)
            if not process_tree(root):
                raise RuntimeError("Restored server exited before runtime validation")
            if (control / "runtime-0.json").exists():
                rank = probe._read_json(control / "runtime-0.json")
                if rank.get("session_id") != session:
                    raise ValueError("Stale runtime-ready acknowledgement")
                break
            time.sleep(0.05)
        report["timings_seconds"]["runtime_ready"] = time.monotonic() - started
        address = probe._read_address(control / "address")
        while True:
            try:
                status, _ = probe._request(
                    address, "/health", min(1, probe._remaining(deadline)), headers=headers
                )
                if status == 200:
                    break
            except (OSError, TimeoutError):
                probe._remaining(deadline)
            time.sleep(0.05)
        if probe._request(address, "/health", probe._remaining(deadline))[0] != 503:
            raise ValueError("Candidate admitted traffic before validation")
        for request in requests:
            status, body = probe._request(
                address, "/v1/completions", probe._remaining(deadline), request, headers
            )
            if status != 200:
                raise ValueError(f"Private generation returned HTTP {status}")
            report["outputs"].append(probe._completion(body))
        if report["outputs"] != baseline["outputs"]:
            raise ValueError("Restored completions differ from the cold baseline")
        report["timings_seconds"]["first_validated_responses"] = time.monotonic() - started
        probe._write_report(
            control / "activate.json",
            {"template_id": manifest["template_id"], "session_id": session},
        )
        if probe._request(address, "/health", probe._remaining(deadline))[0] != 200:
            raise ValueError("Serving admission did not activate")
        report.update(status="PASS", address=f"{address[0]}:{address[1]}")
        report["timings_seconds"]["serving_ready"] = time.monotonic() - started
        probe._write_report(trial / "report.json", report)
        print(json.dumps(report), flush=True)
        if args.serve:
            while process_tree(root):
                time.sleep(1)
        return report
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        probe._write_report(control / "abort.json", {"session_id": session})
        terminate_tree(owned)
        probe._write_report(trial / "report.json", report)


def main() -> None:
    """Parse the opt-in experimental CLI and serialize operations per artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("capture", "restore"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-bin", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--serve", action="store_true")
    args, args.command = parser.parse_known_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not args.snapshot_bin.is_absolute():
        parser.error("Need a positive timeout and absolute --snapshot-bin")
    if args.operation == "capture":
        print(json.dumps(capture(args), indent=2))
    else:
        if not args.baseline or not args.requests or args.command:
            parser.error("restore requires --baseline and --requests, and no command")
        with (args.artifact / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            restore(args)


if __name__ == "__main__":
    main()
