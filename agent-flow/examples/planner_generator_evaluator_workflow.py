from __future__ import annotations

import argparse
from pathlib import Path

from agent_flow import (CLAUDE_CODE_DEFAULT_MODEL, CODEX_DEFAULT_MODEL,
                        AgentLayer, AgentLayerConfig, BackendConfig,
                        SessionConfig)

PLANNER_SYSTEM_PROMPT = """\
You are the Planner, the first agent in a three-agent coding workflow.

Your final response is forwarded directly to the Generator, so it must contain
the complete implementation plan.

Workspace artifacts:
- Write the plan you produce to `plan.md`.
- Append a short progress entry to `progress.md` after you finish.

Write the plan for the Generator and Evaluator, not for the end user. Focus on
features, constraints, behavior, and acceptance criteria. Do not prescribe file
layouts unless the task explicitly requires them.

No conversational filler. Jump straight into the plan.
"""

GENERATOR_SYSTEM_PROMPT = """\
You are the Generator, the second agent in a three-agent coding workflow.

The Planner's final response is forwarded directly to you in the user message.
Your own final response is forwarded directly to the Evaluator, so make it a
concise but complete implementation summary.

Workspace artifacts:
- `plan.md` contains the Planner's written plan for reference.
- `progress.md` is a shared log where you append an entry after you finish.

Use the forwarded plan as the source of truth, implement the work, validate it
by running the code, and append a progress entry describing what you changed,
what you verified, and any notable tradeoffs.

The Planner's directives take priority over raw Evaluator feedback if they
conflict. No conversational filler.
"""

EVALUATOR_SYSTEM_PROMPT = """\
You are the Evaluator, the final agent in a three-agent coding workflow.

The Planner's plan and the Generator's final response are forwarded directly to
you in the user message.

Workspace artifacts:
- `plan.md` contains the Planner's written plan for reference.
- `progress.md` contains the shared execution log where you append an entry
  after you finish.

Inspect the actual code, run the implementation, and append a progress entry
with concrete strengths, gaps, and a concise score grounded in observed
behavior.

Do not evaluate from code review alone. No conversational filler.
"""


class PlannerGeneratorEvaluatorWorkflow:
    """Small planner -> generator -> evaluator harness built on AgentLayer."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.plan_path = workspace / "plan.md"
        self.progress_path = workspace / "progress.md"

        # Use Codex for generation and Claude for planning/evaluation.
        self.planner = AgentLayer(
            AgentLayerConfig(
                name="planner",
                system_prompt=PLANNER_SYSTEM_PROMPT,
                backend=BackendConfig(kind="claude-code",
                                      model=CLAUDE_CODE_DEFAULT_MODEL),
                session=SessionConfig(mode="persistent"),
            ))
        self.generator = AgentLayer(
            AgentLayerConfig(
                name="generator",
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                backend=BackendConfig(kind="codex", model=CODEX_DEFAULT_MODEL),
                session=SessionConfig(mode="persistent"),
            ))
        self.evaluator = AgentLayer(
            AgentLayerConfig(
                name="evaluator",
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                backend=BackendConfig(kind="claude-code",
                                      model=CLAUDE_CODE_DEFAULT_MODEL),
                session=SessionConfig(mode="persistent"),
            ))

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text("", encoding="utf-8")
        self.progress_path.write_text("", encoding="utf-8")

    def run(self, task: str) -> None:
        plan = self.planner(
            f"Workspace: {self.workspace}\n\n"
            f"Task: {task}\n\n"
            f"Write your complete implementation plan to "
            f"`{self.plan_path}`.\n"
            "Return that same plan as your final response "
            "because it will be forwarded directly to the "
            "Generator.\n"
            f"Then append a progress entry to `{self.progress_path}` "
            f"summarizing what you planned.")

        generator_summary = self.generator(
            f"Workspace: {self.workspace}\n"
            "Iteration: 1\n\n"
            f"Task: {task}\n\n"
            "Use the Planner's final response below as the "
            "primary handoff input for this iteration.\n\n"
            "Planner final response:\n"
            f"{plan}\n\n"
            f"After completing your implementation, append a "
            f"progress entry to `{self.progress_path}` summarizing "
            f"what you implemented or changed this iteration.\n"
            "Return a concise implementation summary as your "
            "final response because it will be forwarded "
            "directly to the Evaluator.")

        self.evaluator(f"Workspace: {self.workspace}\n"
                       "Iteration: 1\n\n"
                       f"Task: {task}\n\n"
                       "Planner final response:\n"
                       f"{plan}\n\n"
                       "Generator final response:\n"
                       f"{generator_summary}\n\n"
                       f"After completing your evaluation, append a progress "
                       f"entry to `{self.progress_path}` with your evaluation "
                       f"summary.\n"
                       "Return your evaluation as the final response.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planner -> generator -> evaluator workflow example.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/planner-generator-evaluator-demo"),
        help="Workspace directory for plan/progress artifacts.",
    )
    parser.add_argument(
        "--task",
        default="Build a small Hello World script.",
        help="Task handed to the planner when --run is set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workflow = PlannerGeneratorEvaluatorWorkflow(args.workspace)
    workflow.run(args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
