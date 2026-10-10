# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, opt-in clean-startup checkpoint protocol for the Snapshot prototype."""

import hmac
import json
import os
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def control_directory() -> Path | None:
    """Return the restricted control directory, or None for ordinary startup.

    Raises:
        ValueError: If the configured directory is not private to this user.
    """
    value = os.environ.get("TRTLLM_SNAPSHOT_DIR")
    if not value:
        return None
    path = Path(value)
    info = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("Snapshot control directory must be owned by this user and mode 0700")
    return path


def read_record(path: Path) -> dict[str, Any]:
    """Read one atomically published protocol record.

    Args:
        path: Record in the private control directory.

    Returns:
        A JSON object, or an empty object if not yet published.

    Raises:
        ValueError: If the record is not a JSON object.
    """
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return record


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Publish a record without exposing a partial JSON document.

    Args:
        path: Destination in a private directory.
        record: JSON-serializable protocol record.
    """
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as output:
        json.dump(record, output, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def validate_launch(host: str, args: dict, standalone: bool) -> None:
    """Reject profiles outside the native HTTP prototype before model loading.

    Args:
        host: HTTP bind address.
        args: Existing native LLM arguments.
        standalone: Whether discovery and disaggregated roles are absent.

    Raises:
        ValueError: If the prototype is enabled for an unsupported launch.
    """
    if control_directory() is None:
        return
    if (
        host != "127.0.0.1"
        or not standalone
        or args.get("num_serve_frontends", 1) != 1
        or args.get("orchestrator_type") is not None
        or args.get("backend") != "pytorch"
    ):
        raise ValueError(
            "Snapshot prototype requires standalone loopback HTTP, MPI and one frontend"
        )


def startup_checkpoint(executor: "PyExecutor") -> None:
    """Stop a clean, warmed executor before its request-processing threads start.

    Args:
        executor: Newly constructed PyTorch executor; never an active server.

    Raises:
        ValueError: If the profile is unsupported or restored evidence differs.
        RuntimeError: If the external coordinator aborts the operation.
    """
    directory = control_directory()
    if directory is None:
        return
    import torch

    args = executor.llm_args
    if (
        executor.dist.world_size != 1
        or executor.kv_cache_transceiver is not None
        or executor.kv_connector_manager is not None
        or executor.draft_model_engine is not None
        or executor.is_encoder_decoder
        or executor.dwdp_manager is not None
        or str(args.load_format).lower() in {"gms", "loadformat.gms"}
        or args.kv_cache_config.enable_block_reuse
        or args.kv_cache_config.host_cache_size
    ):
        raise ValueError(
            "Snapshot prototype requires one aggregate rank, resident KV and no block reuse/GMS"
        )
    if executor.worker_started or executor.active_requests or executor.previous_batch is not None:
        raise ValueError("Snapshot requires the clean startup boundary, not a live worker")
    runner = executor.model_engine.cuda_graph_runner
    if runner is None or not runner.graphs:
        raise ValueError("Snapshot prototype requires warmed CUDA graphs")
    torch.cuda.synchronize()
    template = read_record(directory / "template.json")
    if not template.get("template_id"):
        raise ValueError("Missing Snapshot template identity")
    evidence = {
        "pid": os.getpid(),
        "rank": executor.global_rank,
        "world_size": executor.dist.world_size,
        "graph_keys": sorted(map(str, runner.graphs)),
        "template_id": template["template_id"],
    }
    write_record(directory / f"capture-{executor.global_rank}.json", evidence)
    # The host owns the timeout. An absolute deadline saved in a template would
    # expire while that template is stored, before restored code can run.
    while True:
        if read_record(directory / "abort.json"):
            raise RuntimeError("Snapshot coordinator aborted startup")
        restored = read_record(directory / "restore.json")
        if restored:
            if (
                restored.get("template_id") != template["template_id"]
                or not restored.get("session_id")
                or not restored.get("validation_token")
            ):
                raise ValueError("Invalid Snapshot restore session")
            break
        time.sleep(0.05)
    torch.cuda.synchronize()
    write_record(
        directory / f"memory-{executor.global_rank}.json",
        {**evidence, "session_id": restored["session_id"]},
    )
    if sorted(map(str, runner.graphs)) != evidence["graph_keys"]:
        raise ValueError("Restored CUDA graph set changed")
    write_record(
        directory / f"runtime-{executor.global_rank}.json",
        {**evidence, "session_id": restored["session_id"]},
    )


class SnapshotAdmission:
    """Keep every HTTP route closed until validation or explicit activation."""

    def __init__(self, app: Any, directory: Path) -> None:
        """Wrap the native ASGI app.

        Args:
            app: Native serving application.
            directory: Private control directory shared with the coordinator.
        """
        self.app = app
        self.directory = directory

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """Reject unauthorized inference until the fresh session is activated.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callback.
            send: ASGI send callback.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        restored = read_record(self.directory / "restore.json")
        activation = read_record(self.directory / "activate.json")
        active = bool(restored.get("session_id")) and activation == {
            "session_id": restored["session_id"],
            "template_id": restored["template_id"],
        }
        supplied = dict(scope.get("headers", [])).get(b"x-trtllm-snapshot-validation", b"")
        expected = restored.get("validation_token", "").encode()
        validation = (
            bool(expected)
            and hmac.compare_digest(supplied, expected)
            and scope.get("path") in {"/health", "/v1/completions", "/server_info"}
        )
        if not read_record(self.directory / "abort.json") and (active or validation):
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"error":"Snapshot candidate is not serving-ready"}',
            }
        )
