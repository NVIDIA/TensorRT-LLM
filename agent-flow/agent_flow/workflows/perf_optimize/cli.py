from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from agent_flow.workflows.perf_analyze.sol_methodology import resolve_sol_methodology

from .disagg import has_disagg
from .prompts import build_perf_optimize_prompts
from .sol_track import track_name
from .state import STATE_FILENAME
from .task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    kernel_coverage,
    load_and_validate_task_yaml,
    sol_enabled,
)
from .workflow import PerfOptimizeWorkflow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively optimize a trtllm-serve deployment: benchmark the "
        "baseline, profile and rank optimizations into roadmap.yaml, apply the "
        "top items one at a time (up to optimize.max_items_per_round per "
        "round), gate each change on code quality / functionality / measured "
        "perf (the evaluator approves, rejects, or pushes back each attempt, "
        "profiling every accept under nsys), run the full optimize.max_rounds "
        "budget unless the roadmap exhausts or the improvement target is met, "
        "verify the final state with one independent QA benchmark, and report "
        "expected-vs-measured gains — via a benchmarker -> [analyzer -> "
        "(optimizer <-> evaluator) x items] x rounds -> qa -> reporter loop. "
        "A one-shot SOL projector stage runs between the baseline and "
        "round 1 (sol_projection.md) unless task.yaml sets "
        "`sol.enabled: false`. "
        "--reuse-analysis seeds a fresh run from a previous perf-analyze / "
        "perf-optimize workspace so the campaign starts at the optimize stage."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task.yaml spec. Requires `checkpoint_path` and "
        "`trtllm_repo_path`; optional top-level `extra_llm_api_options` "
        "path, optional `benchmark` / `profile` / `optimize` / `accuracy` "
        "blocks, an optional `slurm-environment` block, and an optional "
        "`sol` block (all fields optional: `enabled` gates the one-shot "
        "SOL projector stage — on by default — and `gpu` names the GPU "
        "part for the SOL skill's peaks calculator). "
        "An optional `profile.kernel_coverage` block "
        "activates the per-kernel coverage contract: the analyzer's ncu "
        "dive covers every kernel above the share bar and answers "
        "faster?/fusible? per kernel in a schema-validated "
        "kernel_ledger.yaml each round. "
        "See task.example.yaml.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/perf-optimize"),
        help="Workspace directory for shared state (task.yaml, roadmap.yaml, "
        "sol_projection.md, baseline/, tuning/, rounds/, "
        "optimization_report.md/.html, progress.yaml) and run artifacts.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files/directories "
        f"({STATE_FILENAME}, sol_projection.md, roadmap.yaml, "
        "optimization_report.md/.html, "
        "progress.yaml, baseline/, rounds/, tuning/, sol_work/, "
        "reused_analysis/) and start fresh. The "
        "TRT-LLM checkout is not touched (abandoned perf-optimize/* branches "
        "are left for inspection). Without this flag the workflow resumes "
        "from the checkpoint when one is present, and starts fresh otherwise.",
    )
    parser.add_argument(
        "--reuse-analysis",
        default=None,
        metavar="DIR",
        help="Seed a fresh run from a previous perf-analyze workspace or "
        "perf-optimize campaign workspace instead of re-deriving its "
        "analysis: its baseline report (+ result JSONs), SOL projection "
        "(+ sol_work/), and newest profile findings (+ traces and "
        "kernel_ledger.yaml) are copied into this workspace, the "
        "benchmarker/projector stages are skipped, and round 1's analyzer "
        "runs plan-only — authoring roadmap.yaml from the imported evidence "
        "with no server, profiler, or benchmark. A source roadmap.yaml is "
        "kept aside as read-only prior art (reused_analysis/), never as this "
        "campaign's ledger. Whatever the source lacks is produced normally. "
        "Fresh runs only — ignored on resume.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override `optimize.max_rounds` from task.yaml on a fresh run "
        "(each round opens with an analyzer turn — a re-profile when the "
        "previous round accepted something, a replan otherwise — then "
        "applies up to `optimize.max_items_per_round` roadmap items one at "
        "a time). Ignored on resume — the checkpointed budget wins.",
    )
    _add_dry_run(parser)
    return parser.parse_args(argv)


def _run_disagg_sol(args) -> None:
    """The staged two-track path: fix the operating point, then optimize at it.

    Dispatched on the shape of `task.yaml` rather than on a flag, so one file
    describes the whole campaign and there is one place to look for what it
    will do. The single-track path below is untouched: a spec without a
    `disagg_sol` block reaches it exactly as before.
    """
    import json

    import yaml

    from . import spawn
    from .disagg_sol import DisaggSolError, supervise

    raw = yaml.safe_load(Path(args.task).read_text(encoding="utf-8")) or {}
    block = raw.get("disagg_sol") or {}
    root = Path(args.workspace)
    try:
        record = supervise(
            raw,
            sweeps={k: Path(v) for k, v in (block.get("sweeps") or {}).items()},
            repos={k: Path(v) for k, v in (block.get("repos") or {}).items()},
            workspace_root=root,
            label=block.get("label") or root.name,
            incumbent=block.get("incumbent"),
            dry_run=bool(getattr(args, "dry_run", False)),
            # Only when the spec says where the skill lives. Without it the
            # design stays an input and an unestablished one is refused --
            # which is the safer default for a step that costs an order of
            # magnitude more than the campaigns it enables.
            designer=(
                (lambda instruction: spawn.design(instruction, cwd=Path(block["config_repo"])))
                if block.get("config_repo")
                else None
            ),
        )
    except DisaggSolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(record, indent=2, default=str))


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "staged disagg only: select the operating point and write every "
            "campaign's task.yaml, then stop without starting them"
        ),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # Before schema validation: a staged spec is a different shape, and the
    # single-track schema would reject it for the fields it deliberately lacks.
    try:
        _raw = yaml.safe_load(Path(args.task).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"error: could not read {args.task}: {exc}", file=sys.stderr)
        sys.exit(2)
    if isinstance(_raw, dict) and "disagg_sol" in _raw:
        _run_disagg_sol(args)
        return
    try:
        task_data = load_and_validate_task_yaml(
            args.task,
            max_rounds_override=args.max_rounds,
        )
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if args.reuse_analysis is not None and not Path(args.reuse_analysis).expanduser().is_dir():
        print(
            f"error: --reuse-analysis source is not a directory: {args.reuse_analysis}",
            file=sys.stderr,
        )
        sys.exit(2)
    # Resolve the projector's methodology skill once, before the run, so
    # it is told to load a skill this session actually has. Skipped (free)
    # when the stage is off.
    methodology = resolve_sol_methodology(sol_enabled(task_data))
    note = methodology.console_note()
    if note:
        print(note, file=sys.stderr)
    prompts = build_perf_optimize_prompts(
        include_slurm_environment=has_slurm_environment(task_data),
        approaches=task_data["optimize"]["approaches"],
        include_sol=sol_enabled(task_data),
        kernel_coverage=kernel_coverage(task_data),
        sol_methodology=methodology.name,
        include_disagg=has_disagg(task_data),
        sol_track=track_name(task_data),
    )
    with PerfOptimizeWorkflow(
        workspace=args.workspace,
        clean=args.clean,
        prompts=prompts,
        max_rounds_override=args.max_rounds,
        reuse_analysis=args.reuse_analysis,
        sol_methodology=methodology,
    ) as workflow:
        workflow.run(args.task)


if __name__ == "__main__":
    main()
