# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import posixpath
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import IO, Any, Literal

import fsspec
from fsspec import AbstractFileSystem
from sshfs import SSHFileSystem

ExecutionMode = Literal["local", "remote"]
Side = Literal["control", "execution"]
DirectoryTarget = Literal["control", "execution", "both"]

EXECUTION_LAYOUT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExecutionLayout:
    """Identity and root paths for one workflow run."""

    schema_version: int
    mode: ExecutionMode
    run_id: str
    control_workspace: str
    run_root: str
    execution_workspace: str
    campaign_repo: str
    remote_host: str | None = None

    @classmethod
    def local(cls, *, control_workspace: Path, campaign_repo: str) -> ExecutionLayout:
        workspace = str(control_workspace.resolve())
        return cls(
            schema_version=EXECUTION_LAYOUT_SCHEMA_VERSION,
            mode="local",
            run_id=control_workspace.name,
            control_workspace=workspace,
            run_root=workspace,
            execution_workspace=workspace,
            campaign_repo=campaign_repo,
        )

    @classmethod
    def remote(
        cls,
        *,
        control_workspace: Path,
        remote_host: str,
        run_root: str,
        campaign_repo: str,
    ) -> ExecutionLayout:
        normalized_root = run_root.rstrip("/")
        if not normalized_root.startswith("/"):
            raise ValueError("remote execution run root must be an absolute POSIX path")
        return cls(
            schema_version=EXECUTION_LAYOUT_SCHEMA_VERSION,
            mode="remote",
            run_id=PurePosixPath(normalized_root).name,
            control_workspace=str(control_workspace.resolve()),
            run_root=normalized_root,
            execution_workspace=posixpath.join(normalized_root, "workspace"),
            campaign_repo=campaign_repo,
            remote_host=remote_host.strip(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionLayout:
        layout = cls(**data)
        if layout.schema_version != EXECUTION_LAYOUT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported execution layout version {layout.schema_version!r}; "
                f"expected {EXECUTION_LAYOUT_SCHEMA_VERSION}"
            )
        if layout.mode not in ("local", "remote"):
            raise ValueError(f"unsupported execution mode {layout.mode!r}")
        if layout.mode == "remote" and not layout.remote_host:
            raise ValueError("remote execution layout requires remote_host")
        return layout

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerfOptimizeLayout:
    """The single relative-path contract for perf-optimize artifacts."""

    task = "task.yaml"
    state = ".perf_optimize_state.json"
    progress = "progress.yaml"
    roadmap = "roadmap.yaml"
    sol_projection = "sol_projection.md"
    sol_work = "sol_work"
    reuse_analysis = "reused_analysis"
    report = "optimization_report.md"
    report_html = "optimization_report.html"
    tuning = "tuning"
    tuning_live = "tuning/extra_llm_api_options.yaml"
    tuning_accepted = "tuning/extra_llm_api_options.accepted.yaml"
    baseline = "baseline"
    baseline_report = "baseline/benchmark_results.md"
    rounds = "rounds"
    worktrees = "worktrees"
    final_verification = "final_verification"
    verification_report = "final_verification/verification_report.md"

    @staticmethod
    def _safe_item_id(item_id: str) -> str:
        import re

        return re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip("-.")[:48] or "item"

    def round_dir(self, round_no: int) -> str:
        return f"{self.rounds}/round_{round_no}"

    def analysis_dir(self, round_no: int) -> str:
        return f"{self.round_dir(round_no)}/analysis"

    def item_dir(self, round_no: int, item_index: int, item_id: str) -> str:
        safe = self._safe_item_id(item_id)
        return f"{self.round_dir(round_no)}/item_{item_index}_{safe}"

    def attempt_dir(
        self,
        round_no: int,
        item_index: int,
        item_id: str,
        attempt_no: int,
    ) -> str:
        return f"{self.item_dir(round_no, item_index, item_id)}/attempt_{attempt_no}"

    def item_progress(self, round_no: int, item_index: int, item_id: str) -> str:
        return f"{self.item_dir(round_no, item_index, item_id)}/progress.yaml"

    def item_result_dir(self, round_no: int, item_index: int, item_id: str) -> str:
        return f"{self.item_dir(round_no, item_index, item_id)}/result"

    def item_worktree(self, round_no: int, item_index: int, item_id: str) -> str:
        safe = self._safe_item_id(item_id)
        return f"{self.worktrees}/round_{round_no}/item_{item_index}_{safe}"

    def item_tuning_dir(self, round_no: int, item_index: int, item_id: str) -> str:
        return f"{self.item_dir(round_no, item_index, item_id)}/tuning"

    def item_tuning_live(self, round_no: int, item_index: int, item_id: str) -> str:
        return f"{self.item_tuning_dir(round_no, item_index, item_id)}/extra_llm_api_options.yaml"

    def item_tuning_accepted(self, round_no: int, item_index: int, item_id: str) -> str:
        return (
            f"{self.item_tuning_dir(round_no, item_index, item_id)}"
            "/extra_llm_api_options.accepted.yaml"
        )

    def integration_dir(self, round_no: int) -> str:
        return f"{self.round_dir(round_no)}/integration"

    def integration_worktree(self, round_no: int) -> str:
        return f"{self.worktrees}/round_{round_no}/integration"


class RunFileSystems:
    """Resolve one relative path contract over control and execution filesystems."""

    def __init__(
        self,
        *,
        layout: ExecutionLayout,
        control_fs: AbstractFileSystem,
        execution_fs: AbstractFileSystem,
    ) -> None:
        self.layout = layout
        self.control_fs = control_fs
        self.execution_fs = execution_fs
        self._execution_io_lock = RLock()

    @classmethod
    def from_layout(
        cls,
        layout: ExecutionLayout,
        *,
        known_hosts: str | Path | None = None,
        timeout: int = 20,
    ) -> RunFileSystems:
        control_fs = fsspec.filesystem("file")
        if layout.mode == "local":
            return cls(layout=layout, control_fs=control_fs, execution_fs=control_fs)

        hosts_path = Path(known_hosts or "~/.ssh/known_hosts").expanduser()
        if not hosts_path.is_file():
            raise FileNotFoundError(
                f"SSH known_hosts file does not exist: {hosts_path}; "
                "remote execution will not disable host-key verification"
            )
        execution_fs = SSHFileSystem(
            str(layout.remote_host),
            known_hosts=str(hosts_path),
            timeout=timeout,
        )
        return cls(layout=layout, control_fs=control_fs, execution_fs=execution_fs)

    @staticmethod
    def _relative(relative: str) -> str:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"execution path must be relative and may not contain '..': {relative}"
            )
        normalized = posixpath.normpath(str(path))
        return "" if normalized == "." else normalized

    def _fs(self, side: Side) -> AbstractFileSystem:
        return self.control_fs if side == "control" else self.execution_fs

    def _root(self, side: Side) -> str:
        if side == "control":
            return self.layout.control_workspace
        return self.layout.execution_workspace

    def _lock(self, *sides: Side):
        if self.layout.mode == "remote" and "execution" in sides:
            return self._execution_io_lock
        return nullcontext()

    def path(self, relative: str, *, on: Side) -> str:
        rel = self._relative(relative)
        root = self._root(on).rstrip("/")
        return root if not rel else posixpath.join(root, rel)

    def exists(self, relative: str, *, on: Side) -> bool:
        with self._lock(on):
            return bool(self._fs(on).exists(self.path(relative, on=on)))

    def makedirs(self, relative: str, *, on: DirectoryTarget) -> None:
        sides: tuple[Side, ...] = ("control", "execution") if on == "both" else (on,)
        seen: set[tuple[int, str]] = set()
        for side in sides:
            fs = self._fs(side)
            path = self.path(relative, on=side)
            target = (id(fs), path)
            if target in seen:
                continue
            seen.add(target)
            with self._lock(side):
                fs.makedirs(path, exist_ok=True)

    def read_text(self, relative: str, *, on: Side) -> str:
        with (
            self._lock(on),
            self._fs(on).open(self.path(relative, on=on), "rt", encoding="utf-8") as stream,
        ):
            return stream.read()

    def write_text(self, relative: str, content: str, *, on: Side) -> None:
        parent = posixpath.dirname(self._relative(relative))
        self.makedirs(parent, on=on)
        with (
            self._lock(on),
            self._fs(on).open(self.path(relative, on=on), "wt", encoding="utf-8") as stream,
        ):
            stream.write(content)

    def open(self, relative: str, mode: str, *, on: Side, **kwargs: Any) -> IO[Any]:
        return self._fs(on).open(self.path(relative, on=on), mode, **kwargs)

    def copy_file(
        self,
        source: str,
        destination: str,
        *,
        source_side: Side,
        destination_side: Side,
    ) -> None:
        parent = posixpath.dirname(self._relative(destination))
        self.makedirs(parent, on=destination_side)
        source_fs = self._fs(source_side)
        destination_fs = self._fs(destination_side)
        source_path = self.path(source, on=source_side)
        destination_path = self.path(destination, on=destination_side)
        if source_fs is destination_fs and source_path == destination_path:
            return
        with self._lock(source_side, destination_side):
            with source_fs.open(source_path, "rb") as src:
                with destination_fs.open(destination_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    def remove(self, relative: str, *, on: Side, recursive: bool = False) -> None:
        with self._lock(on):
            self._fs(on).rm(self.path(relative, on=on), recursive=recursive)

    def files(self, relative: str, *, on: Side) -> list[str]:
        """Return files below ``relative`` as workspace-relative paths."""
        root = self.path("", on=on).rstrip("/")
        directory = self.path(relative, on=on)
        with self._lock(on):
            paths = self._fs(on).find(directory, withdirs=False)
        prefix = root + "/"
        return sorted(path.removeprefix(prefix) for path in paths if path.startswith(prefix))

    def close(self) -> None:
        if self.execution_fs is not self.control_fs:
            close = getattr(self.execution_fs, "close", None)
            if callable(close):
                close()


def initialize_remote_execution(
    layout: ExecutionLayout,
    run_fs: RunFileSystems,
    perf_layout: PerfOptimizeLayout,
) -> None:
    """Create the fixed execution-side roots for a new remote run."""
    if layout.mode != "remote":
        return
    run_fs.makedirs("", on="execution")
    for relative in (
        perf_layout.tuning,
        perf_layout.baseline,
        perf_layout.rounds,
        perf_layout.worktrees,
        perf_layout.final_verification,
    ):
        run_fs.makedirs(relative, on="execution")


def sync_run_inputs_to_execution(
    layout: ExecutionLayout,
    run_fs: RunFileSystems,
    perf_layout: PerfOptimizeLayout,
) -> None:
    """Upload the normalized task and initial tuning files on a new remote run."""
    if layout.mode != "remote":
        return
    for relative in (
        perf_layout.task,
        perf_layout.tuning_live,
        perf_layout.tuning_accepted,
    ):
        run_fs.copy_file(
            relative,
            relative,
            source_side="control",
            destination_side="execution",
        )


def sync_benchmarker_results_to_control(
    layout: ExecutionLayout,
    run_fs: RunFileSystems,
    perf_layout: PerfOptimizeLayout,
) -> None:
    """Pull only baseline benchmark JSON files, preserving relative paths."""
    _sync_json_results_to_control(layout, run_fs, perf_layout.baseline)


def sync_qa_results_to_control(
    layout: ExecutionLayout,
    run_fs: RunFileSystems,
    perf_layout: PerfOptimizeLayout,
) -> None:
    """Pull only final-verification JSON files, preserving relative paths."""
    _sync_json_results_to_control(layout, run_fs, perf_layout.final_verification)


def _sync_json_results_to_control(
    layout: ExecutionLayout,
    run_fs: RunFileSystems,
    relative_directory: str,
) -> None:
    if layout.mode != "remote":
        return
    for relative in run_fs.files(relative_directory, on="execution"):
        if relative.endswith(".json"):
            run_fs.copy_file(
                relative,
                relative,
                source_side="execution",
                destination_side="control",
            )
