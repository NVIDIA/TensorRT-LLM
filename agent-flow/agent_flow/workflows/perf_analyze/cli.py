from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .prompts import build_perf_analyze_prompts
from .sol_methodology import resolve_sol_methodology
from .state import STATE_FILENAME
from .task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    load_and_validate_task_yaml,
    sol_enabled,
)
from .workflow import PerfAnalyzeWorkflow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a model with trtllm-serve, benchmark and profile it with "
        "benchmark_serving.py (nsys + torch profiler), and report the main "
        "performance bottleneck — via a benchmarker -> projector -> analyzer "
        "-> reporter pipeline (the projector derives an analytical "
        "speed-of-light ceiling per the internal-perf-sol-analysis skill "
        "unless task.yaml sets `sol.enabled: false`)."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task.yaml spec. Requires `checkpoint_path` and "
        "`trtllm_repo_path`; optional top-level `extra_llm_api_options` "
        "path, optional `benchmark` / `profile` blocks, an optional "
        "`slurm-environment` block, and an optional `sol` block "
        "(all fields optional: `enabled` gates the projector stage — on "
        "by default — and `gpu` names the GPU part for the SOL skill's "
        "peaks calculator). "
        "See task.example.yaml.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/perf-analyze"),
        help="Workspace directory for shared state files (task.yaml, "
        "benchmark_results.md, sol_projection.md, profile_findings.md, "
        "performance_report.md/.html, progress.yaml) and run artifacts "
        "(serve.log, result JSON, *.nsys-rep, torch_trace/).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files "
        f"({STATE_FILENAME}, benchmark_results.md, sol_projection.md, "
        "profile_findings.md, performance_report.md, "
        "performance_report.html, progress.yaml) and "
        "start fresh. Without this flag the workflow resumes from the "
        "checkpoint when one is present, and starts fresh otherwise.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        task_data = load_and_validate_task_yaml(args.task)
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    # Resolve the projector's methodology skill once, before the run, so
    # it is told to load a skill this session actually has. Skipped (free)
    # when the stage is off.
    methodology = resolve_sol_methodology(sol_enabled(task_data))
    note = methodology.console_note()
    if note:
        print(note, file=sys.stderr)
    prompts = build_perf_analyze_prompts(
        include_slurm_environment=has_slurm_environment(task_data),
        include_sol=sol_enabled(task_data),
        sol_methodology=methodology.name,
    )
    with PerfAnalyzeWorkflow(
        workspace=args.workspace,
        clean=args.clean,
        prompts=prompts,
        sol_methodology=methodology,
    ) as workflow:
        workflow.run(args.task)


if __name__ == "__main__":
    main()
