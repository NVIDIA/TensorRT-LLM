# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate a resident checkpoint from outside the worker and MPI failure domain."""

import argparse
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from engine_client import EngineClient, generate_tokens, validate_probe

SNAPSHOT_REVISION = "4b1300a82c6676df020c201fa884cc602251e12e"
DEFAULT_DEPLOYMENT = Path("/var/lib/trtllm-warp/deployment.json")


def _remaining(deadline: float) -> float:
    """Return the remaining monotonic budget, or raise TimeoutError on expiry."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("Checkpoint deadline expired")
    return left


class Host:
    """Use the bundled Docker adapter unless an operator selects another backend."""

    def __init__(self, executable: Path | None = None, *, deployment: Path | None = None) -> None:
        """Select deployment configuration without passing engine credentials to the adapter."""
        self.bundled = executable is None
        if executable is not None:
            if not executable.is_absolute() or deployment is not None:
                raise ValueError("Use an absolute host adapter path or a deployment, not both")
            self.command = [executable]
        else:
            config = deployment if deployment is not None else DEFAULT_DEPLOYMENT
            if not config.is_absolute():
                raise ValueError("Use an absolute deployment configuration path")
            self.command = [
                Path(sys.executable),
                "-E",
                "-s",
                Path(__file__).resolve().with_name("docker_adapter.py"),
                "--deployment",
                config,
            ]

    def engine(self, result: dict, expected: str, key: str) -> EngineClient:
        """Bind engine control to the host-verified instance, never a restored listener alone."""
        port = _port(result)
        if not self.bundled:
            return EngineClient(port, key)
        if not _digest(expected) or result.get("control_container") != expected:
            raise RuntimeError("The private control channel does not match the owned container")
        return EngineClient(port, key, container=expected)

    def call(
        self, action: str, attempt: str, template: str, deadline: float, **fields: Any
    ) -> dict:
        """Request one bounded host operation and verify its final receipt.

        Args:
            action: Operation in the local host-adapter contract.
            attempt: Fresh identifier for this capture or restore.
            template: Immutable template identifier.
            deadline: Local host monotonic deadline, shared with the adapter.
            **fields: Operation-specific inputs, never engine credentials.

        Returns:
            The correlated final result, not a progress marker.

        Raises:
            RuntimeError: The adapter failed or returned an unrelated receipt.
        """
        envelope = {"version": 1, "action": action, "attempt": attempt, "template": template}
        request = {**envelope, "deadline_monotonic": deadline, "input": fields}
        run = subprocess.run(
            self.command,
            input=json.dumps(request).encode(),
            stdout=subprocess.PIPE,
            env={"PATH": os.defpath},
            timeout=_remaining(deadline),
            check=True,
        )
        _remaining(deadline)
        receipt = json.loads(run.stdout)
        if (
            not isinstance(receipt, dict)
            or any(receipt.get(k) != v for k, v in envelope.items())
            or receipt.get("status") != "succeeded"
            or not isinstance(receipt.get("result"), dict)
        ):
            raise RuntimeError(f"No matching final host receipt for {action}")
        return receipt["result"]


def _private_dir(path: Path) -> None:
    """Reject a state directory shared with the workload or other local users."""
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("State directory must be owned by the operator and private (0700)")


def _sync_dir(directory: Path) -> None:
    """Persist directory entries before dispatching an external operation."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save(path: Path, record: dict) -> None:
    """Publish a complete, private record atomically and never replace an existing one."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(record, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temp, path)
            _sync_dir(path.parent)
        finally:
            temp.unlink()


def _start(state: Path, template: str, budget: float) -> tuple[Path, float]:
    """Check the operator state root, identifier and finite operation budget."""
    _private_dir(state)
    if not re.fullmatch("[0-9a-f]{32}", template):
        raise ValueError("Expected a 32-character lowercase hexadecimal template ID")
    if not math.isfinite(budget) or not 0 < budget <= 3600:
        raise ValueError("Budget must be positive and at most 3600 seconds")
    return state / template, time.monotonic() + budget


def _port(result: dict) -> int:
    """Return the port of an adapter-bound, private loopback control channel."""
    port = result.get("control_port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise RuntimeError("No valid control endpoint for the owned target")
    return port


def _digest(value: Any) -> bool:
    """Check a full immutable container ID or SHA-256 hex digest."""
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _check_driver(result: dict, action: str, attempt: str, template: str, container: str) -> None:
    """Require the selected Snapshot driver's final result for this owned container."""
    raw = result.get("snapshot")
    expected = {
        "snapshot_revision": SNAPSHOT_REVISION,
        "operation": action,
        "attempt": attempt,
        "artifact": template,
        "container": container,
        "status": "PASS",
    }
    if not isinstance(raw, dict) or any(raw.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Missing or stale Snapshot driver result")
    if action == "capture" and raw.get("source_terminated") is not True:
        raise RuntimeError("Snapshot driver did not confirm source termination")
    if action == "restore" and (
        raw.get("memory_ready") is not True
        or type(raw.get("restored_pid")) is not int
        or raw["restored_pid"] <= 0
    ):
        raise RuntimeError("Snapshot driver did not confirm process restore")


def _contain(host: Host, attempt: str, template: str) -> bool:
    """Return true only when the adapter confirms containment of this attempt."""
    try:
        result = host.call("discard", attempt, template, time.monotonic() + 30)
        if result.get("contained") is True:
            return True
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError):
        pass
    print(f"Host containment unconfirmed; reconcile attempt {attempt}", file=sys.stderr)
    return False


def capture(
    state: Path, template: str, source: str, probe: dict, host: Host, key: str, budget: float
) -> None:
    """Prepare a resident source and publish its manifest only after host capture.

    Args:
        state: Private operator-owned directory outside the captured job.
        template: New, single-use template ID.
        source: Full immutable ID of the pre-warmed source container.
        probe: Deterministic generation prompt and reference token IDs.
        host: Independently authorized host adapter.
        key: Source engine control credential.
        budget: Total operation deadline in seconds.
    """
    directory, deadline = _start(state, template, budget)
    if not _digest(source):
        raise ValueError("Expected a full immutable source container ID")
    validate_probe(probe)
    if directory.exists():
        raise FileExistsError(directory)
    # The source is already warm. Its immutable host ownership label was set at launch.
    attempt = template
    pending = directory / "pending-capture.json"
    dispatched = False
    published = False

    def before_dispatch() -> None:
        """Claim the template once, after read-only preflight and before any mutation."""
        directory.mkdir(mode=0o700)
        _sync_dir(state)
        _save(pending, {"attempt": attempt, "source": source})

    def on_send() -> None:
        nonlocal dispatched
        dispatched = True

    try:
        bound = host.call("bind_source", attempt, template, deadline, source=source)
        if bound.get("source") != source:
            raise RuntimeError("Control endpoint does not belong to the source")
        engine = host.engine(bound, source, key)
        evidence = engine.prepare(
            template, _remaining(deadline), probe, before_dispatch=before_dispatch, on_send=on_send
        )
        safe_until = evidence["capture_deadline"]
        if type(safe_until) not in (float, int) or not math.isfinite(safe_until):
            raise ValueError("Missing capture-safe deadline")
        deadline = min(deadline, safe_until)
        _remaining(deadline)
        identity = {
            k: evidence[k]
            for k in ("source_session_id", "template_id", "profile_digest", "cohort_digest")
        }
        proof = {"status_headers": engine.status_headers()} if host.bundled else {}
        result = host.call(
            "capture", attempt, template, deadline, source=source, identity=identity, **proof
        )
        _check_driver(result, "capture", attempt, template, source)
        digest = result.get("artifact_digest")
        if (
            result.get("source") != source
            or result.get("identity") != identity
            or result.get("source_terminated") is not True
            or not _digest(digest)
        ):
            raise RuntimeError("Host did not confirm a complete artifact and source termination")
        _remaining(deadline)
        _save(
            directory / "manifest.json",
            {
                "version": 1,
                "source": source,
                "identity": identity,
                "artifact_digest": digest,
                "probe": probe,
            },
        )
        published = True
        pending.unlink()
        _sync_dir(directory)
    finally:
        if dispatched and not published:
            _contain(host, attempt, template)


def restore(state: Path, template: str, host: Host, key: str, budget: float) -> str:
    """Restore an isolated candidate and publish it only after engine activation.

    Args:
        state: Private state directory used for capture.
        template: Published template ID.
        host: Independently authorized host adapter.
        key: Credential for the destination control channel.
        budget: Shared budget for restore, validation, activation and publication.

    Returns:
        The published target's immutable container ID.
    """
    directory, deadline = _start(state, template, budget)
    _private_dir(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if (directory / "pending-capture.json").exists():
        raise RuntimeError("Capture publication is unconfirmed; reconcile the source attempt")
    identity = manifest["identity"]
    source, digest = manifest["source"], manifest["artifact_digest"]
    if (
        type(manifest["version"]) is not int
        or manifest["version"] != 1
        or not isinstance(identity, dict)
        or set(identity) != {"source_session_id", "template_id", "profile_digest", "cohort_digest"}
        or identity["template_id"] != template
        or not isinstance(identity["source_session_id"], str)
        or not 0 < len(identity["source_session_id"]) <= 128
        or not _digest(identity["profile_digest"])
        or not _digest(identity["cohort_digest"])
        or not _digest(source)
        or not _digest(digest)
    ):
        raise ValueError("Invalid template manifest")
    validate_probe(manifest["probe"])
    attempt = secrets.token_hex(16)
    active = directory / "active-restore.json"
    # Also survives coordinator loss: an ambiguous publish must not start a second target.
    _save(active, {"attempt": attempt})
    journal = directory / attempt
    journal.mkdir(mode=0o700)
    _save(journal / "restore.json", {"attempt": attempt, "template": template})
    publishing = False
    retired = False
    try:
        result = host.call(
            "restore",
            attempt,
            template,
            deadline,
            source=source,
            identity=identity,
            artifact_digest=digest,
        )
        target = result.get("target")
        if (
            not _digest(target)
            or target == source
            or result.get("source") != source
            or result.get("artifact_digest") != digest
            or result.get("identity") != identity
            or result.get("isolated") is not True
            or result.get("authority_established") is not True
            or result.get("host_result_id") != attempt
        ):
            raise RuntimeError(
                "Host did not establish this isolated destination and restore result"
            )
        _check_driver(result, "restore", attempt, template, target)
        engine = host.engine(result, target, key)
        engine.validate_and_activate(
            identity, attempt, attempt, manifest["probe"], _remaining(deadline)
        )
        # A lost publish reply can mean the target is already serving. Never destroy it on timeout.
        _save(journal / "publishing.json", {"target": target})
        _remaining(deadline)
        publishing = True
        proof = {"status_headers": engine.status_headers()} if host.bundled else {}
        result = host.call("publish", attempt, template, deadline, target=target, **proof)
        if result.get("target") != target or result.get("published") is not True:
            raise RuntimeError("Host publication is unconfirmed")
        _save(journal / "published.json", {"target": target})
        retired = True
        return target
    finally:
        if publishing and not retired:
            print(
                f"Publication uncertain; reconcile host attempt {attempt} before retry",
                file=sys.stderr,
            )
        elif not retired:
            retired = _contain(host, attempt, template)
        if retired:
            active.unlink()
            _sync_dir(directory)


def generate(
    state: Path,
    template: str,
    target: str,
    host: Host,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    budget: float,
) -> list[int]:
    """Send one fresh inference request via the bundled backend's published local route."""
    directory, deadline = _start(state, template, budget)
    if not host.bundled or not _digest(target):
        raise ValueError("Generation requires the bundled adapter and a full destination ID")
    _private_dir(directory)
    for journal in directory.iterdir():
        if not re.fullmatch(r"[0-9a-f]{32}", journal.name):
            continue
        _private_dir(journal)
        published = journal / "published.json"
        if not published.is_file() or json.loads(published.read_bytes()) != {"target": target}:
            continue
        route = host.call("resolve", journal.name, template, deadline, target=target)
        if (
            route.get("published") is not True
            or route.get("target") != target
            or route.get("control_container") != target
        ):
            raise RuntimeError("Host route does not match the published destination")
        return generate_tokens(
            target, _port(route), model, prompt_token_ids, max_tokens, _remaining(deadline)
        )
    raise ValueError("No confirmed publication for this destination")


def main() -> None:
    """Run one opt-in CPU-side transaction; do not start a server or change normal startup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--deployment", type=Path, help=f"WARP host configuration (default: {DEFAULT_DEPLOYMENT})"
    )
    backend.add_argument("--host-adapter", type=Path, help="Override the bundled host adapter")
    parser.add_argument("--template", required=True)
    parser.add_argument("--budget", type=float, default=300)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("capture")
    source.add_argument("--source", required=True)
    source.add_argument("--probe", required=True, type=Path)
    commands.add_parser("restore")
    inference = commands.add_parser(
        "generate", help="Send a fresh request to a published local target"
    )
    inference.add_argument("--target", required=True)
    inference.add_argument("--model", required=True)
    inference.add_argument("--prompt-token-ids", required=True, type=int, nargs="+")
    inference.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()
    host = Host(args.host_adapter, deployment=args.deployment)
    if args.command == "generate":
        print(
            json.dumps(
                generate(
                    args.state,
                    args.template,
                    args.target,
                    host,
                    args.model,
                    args.prompt_token_ids,
                    args.max_tokens,
                    args.budget,
                )
            )
        )
        return
    key = os.environ.get("TRTLLM_RUNTIME_CONTROL_API_KEY")
    if not key:
        parser.error("Supply this endpoint's TRTLLM_RUNTIME_CONTROL_API_KEY")
    if args.command == "capture":
        capture(
            args.state,
            args.template,
            args.source,
            json.loads(args.probe.read_bytes()),
            host,
            key,
            args.budget,
        )
        print(args.template)
    else:
        print(restore(args.state, args.template, host, key, args.budget))


if __name__ == "__main__":
    main()
