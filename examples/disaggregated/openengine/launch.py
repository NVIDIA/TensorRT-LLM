# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch a local OpenEngine context/generation pair with automatic RankInfo keys."""

import argparse
import contextlib
import os
import secrets
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import yaml


def _write_configs(sources: list[Path], directory: Path) -> list[Path]:
    key = secrets.token_hex(32)
    paths = []
    for role, source in zip(("context", "generation"), sources):
        config = yaml.safe_load(source.read_text())
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError(f"{source}: configuration must be a YAML mapping")
        config["internal_request_auth_key"] = key
        config.setdefault("cache_transceiver_config", {"backend": "NIXL"})
        # These workers and their MPI children run on this host.
        if config.get("executor_backend", "mpi") != "mpi":
            raise ValueError("The local OpenEngine launcher requires executor_backend: mpi")
        destination = directory / f"{role}.yaml"
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            yaml.safe_dump(config, stream)
        paths.append(destination)
    return paths


def _stop_workers(workers: list[subprocess.Popen]) -> None:
    for worker in workers:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(worker.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    for worker in workers:
        with contextlib.suppress(subprocess.TimeoutExpired):
            worker.wait(timeout=max(0, deadline - time.monotonic()))
    # Also reap MPI descendants if their server exited before cleaning them up.
    for worker in workers:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(worker.pid, signal.SIGKILL)
        worker.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--context-config", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument(
        "--context-gpus", default="0", help="CUDA device IDs or UUIDs, comma-separated"
    )
    parser.add_argument(
        "--generation-gpus", default="1", help="CUDA device IDs or UUIDs, comma-separated"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--context-port", type=int, default=8001)
    parser.add_argument("--generation-port", type=int, default=8002)
    args = parser.parse_args(argv)
    devices = [
        [device.strip() for device in value.split(",")]
        for value in (args.context_gpus, args.generation_gpus)
    ]
    if any(not device for group in devices for device in group):
        parser.error("GPU lists must not contain empty device IDs")
    if len(set(devices[0] + devices[1])) != len(devices[0] + devices[1]):
        parser.error("Context and generation GPU lists must contain distinct devices")
    if args.context_port == args.generation_port or any(
        not 1 <= port <= 65535 for port in (args.context_port, args.generation_port)
    ):
        parser.error("Context and generation ports must be distinct and within 1..65535")

    stopped = threading.Event()
    stop_signal = 0

    def stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum
        stopped.set()

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with tempfile.TemporaryDirectory(prefix="trtllm-openengine-") as directory:
            paths = _write_configs([args.context_config, args.generation_config], Path(directory))
            workers = []
            try:
                for role, path, gpus, port in zip(
                    ("context", "generation"),
                    paths,
                    devices,
                    (args.context_port, args.generation_port),
                ):
                    if stopped.is_set():
                        break
                    command = [
                        "trtllm-serve",
                        args.model,
                        "--backend",
                        "pytorch",
                        "--grpc",
                        "--grpc-protocol",
                        "openengine",
                        "--host",
                        args.host,
                        "--port",
                        str(port),
                        "--config",
                        str(path),
                    ]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
                    workers.append(subprocess.Popen(command, env=env, start_new_session=True))
                    print(f"Starting {role} OpenEngine server at {args.host}:{port}", flush=True)
                while not stopped.wait(0.2):
                    for worker in workers:
                        status = worker.poll()
                        if status is not None:
                            return status if status > 0 else 1
                return 128 + stop_signal
            finally:
                _stop_workers(workers)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
