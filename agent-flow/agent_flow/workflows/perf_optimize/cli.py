from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_flow.workflows.perf_analyze.sol_methodology import resolve_sol_methodology

from .disagg import has_disagg
from .prompts import build_perf_optimize_prompts
from .state import STATE_FILENAME
from .task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    kernel_coverage,
    load_and_validate_task_yaml,
    sol_enabled,
)
from .workflow import PerfOptimizeWorkflow, StageOutputsMissing


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively optimize a trtllm-serve deployment: benchmark the "
        "baseline, profile and rank optimizations into roadmap.yaml, apply the "
        "top items serially or concurrently in isolated worktrees (up to "
        "optimize.max_items_per_round per round), gate each candidate on "
        "code quality / functionality / measured "
        "perf (the evaluator approves, rejects, or pushes back each attempt, "
        "profiling candidate-ready states under nsys), directly accept serial "
        "candidates or integrate and benchmark a parallel batch, run the full optimize.max_rounds "
        "budget unless the roadmap exhausts or the improvement target is met, "
        "verify the final state with one independent QA benchmark, and report "
        "expected-vs-measured gains — via a benchmarker -> [analyzer -> "
        "(optimizer <-> evaluator) items -> optional integrator] x rounds "
        "-> qa -> reporter loop. "
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
        "eliminable?/faster?/fusible?/overlappable? per kernel in a "
        "schema-validated kernel_ledger.yaml each round. "
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
        "progress.yaml, baseline/, rounds/, worktrees/, tuning/, sol_work/, "
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
        "standing profile is stale, a replan otherwise — then evaluates up "
        "to `optimize.max_items_per_round` roadmap items per the configured "
        "`optimize.item_execution` mode). "
        "Ignored on resume — the checkpointed budget wins.",
    )
    parser.add_argument(
        "--stage-retries",
        type=int,
        default=1,
        metavar="N",
        help="How many extra turns to give a stage that returned without "
        "writing its required deliverable, counted PER checkpoint position so "
        "the budget resets whenever the campaign advances. The case this "
        "exists for is a role that submits a Slurm job, ends its turn meaning "
        "to collect the results next turn, and finds there is no next turn. "
        "0 restores the previous behaviour: fail the campaign and leave the "
        "re-run to a person. Only this failure is retried — a deliverable that "
        "exists but is invalid replays identically and is never retried.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
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
        remote_execution=task_data,
        campaign_name=args.workspace.resolve().name,
        approaches=task_data["optimize"]["approaches"],
        include_sol=sol_enabled(task_data),
        kernel_coverage=kernel_coverage(task_data),
        sol_methodology=methodology.name,
        include_disagg=has_disagg(task_data),
    )
    with PerfOptimizeWorkflow(
        workspace=args.workspace,
        clean=args.clean,
        prompts=prompts,
        max_rounds_override=args.max_rounds,
        reuse_analysis=args.reuse_analysis,
        sol_methodology=methodology,
    ) as workflow:
        _run_with_stage_retries(workflow, args.task, args.stage_retries)


def _run_with_stage_retries(workflow, task: str, budget: int) -> None:
    """Run the workflow, giving a stage that wrote nothing another turn.

    ``_require_stage_outputs`` raises before the checkpoint advances, and its
    own docstring already names the fix: "simply re-running the workflow
    retries the same stage". That instruction was addressed to a person. An
    unattended campaign has no person, so a role that ended its turn early
    killed hours of completed work -- and the recovery was a one-line command
    nobody was there to type.

    The budget is PER POSITION, not per run. A campaign is dozens of stages;
    one global counter would let three unrelated hiccups over eight rounds
    exhaust it, and would equally let a stage that can never succeed burn the
    whole budget while looking like progress. Keying on the checkpoint means
    the count resets the moment the campaign actually moves, and a stage that
    is genuinely stuck stops after ``budget`` tries at the same spot.

    Only :class:`StageOutputsMissing` is retried. Everything else propagates:
    a roadmap that fails schema validation, for instance, would replay the
    same invalid file and fail identically, so retrying it buys nothing and
    hides the real error behind repeated attempts.
    """
    seen: dict[tuple, int] = {}
    while True:
        try:
            workflow.run(task)
            return
        except StageOutputsMissing as exc:
            position = workflow.checkpoint_position()
            seen[position] = seen.get(position, 0) + 1
            if seen[position] > budget:
                raise
            print(
                f"retrying stage {exc.stage!r} "
                f"(attempt {seen[position] + 1} of {budget + 1}): "
                f"it left {', '.join(exc.missing)} unwritten",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
