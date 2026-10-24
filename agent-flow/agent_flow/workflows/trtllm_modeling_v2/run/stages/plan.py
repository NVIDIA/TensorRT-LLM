# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The survey and its paper review: what a `survey` or plan `review` task is asked.

A new run starts with the survey: phase 0 puts it in the graph before the
planner's first turn, so both start at once, beside preflight's substrate
gate. When it lands, the orchestrator adds one kernel task per vocabulary
queue row (``run/orchestrator.py``); the paper review is the planner's to
ask for, and nothing waits for it.
"""

from __future__ import annotations

from ...agents.prompts.plan_drafter import capture_lists_block
from .. import graph as graph_mod
from .. import orchestrator as orchestrator_mod

#: The id of the survey a new run starts with.
SURVEY_TASK_ID = "survey"


class PlanStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    #: A new run starts with the survey (:meth:`_start_with_the_survey`); off,
    #: the planner's first graph names it, as a test of the planner's graph alone wants.
    survey_first: bool = True

    def _start_with_the_survey(self) -> bool:
        """Put the survey in a new run's graph, beside the planner's first turn.

        A run with a graph already, or with a vocabulary, has had its survey.
        The task names the target's directory (``ref``), where the
        orchestrator places the target's own entries when it adds the kernel
        tasks. Returns whether the run was seeded.
        """
        if not self.survey_first or self.vocabulary_path.exists():
            return False
        survey = graph_mod.Task(
            id=SURVEY_TASK_ID,
            kind="survey",
            goal=(
                "map the checkpoint's computation steps onto the catalog: write plan.md and "
                "vocabulary.yaml"
            ),
            gpus=1,
            est_minutes=120,
            ref=str(getattr(self._preflight, "target_rel", "") or ""),
        )
        seeded = orchestrator_mod.seed(
            self.workspace,
            [survey],
            "the run starts with the survey",
            f"the run started with the survey (task `{SURVEY_TASK_ID}`), which runs beside "
            "this turn; keep it in your graph as it is. When it lands, the orchestrator adds "
            "one kernel task per vocabulary queue row (`k<row>-<entry>`) and they start at "
            "once. Write the first graph: the work that needs no queue row, such as the "
            "engine's configuration and cache work, starts now beside the survey",
        )
        if seeded:
            self.journal.event("phase 0", "the survey starts beside the planner's first turn")
        return seeded

    def _plan_drafter_prompt(self) -> str:
        """The survey task: map the checkpoint onto the catalog."""
        return (
            f"Workspace: {self.workspace}\n\n"
            f"Read `{self.task_path}` for the run's spec: the checkpoint, the "
            f"architecture, the parallel segment, the features and the accuracy anchor.\n\n"
            f"{self._surface_block}\n"
            f"{self._runtime_block}\n"
            f"{capture_lists_block(self.task_data)}\n"
            f"Write `{self.plan_path}` and `{self.vocabulary_path}` as your system "
            f"prompt specifies. The orchestrator parses `vocabulary.yaml` and refuses "
            f"a malformed one, so follow the schema exactly.\n\n"
            + self._report_instruction("plan_drafter")
        )

    def _plan_reviewer_prompt(self) -> str:
        """A paper review of the survey's plan and vocabulary."""
        return (
            f"Workspace: {self.workspace}\n\n"
            f"Read `{self.task_path}` (the spec), `{self.plan_path}` (the plan) and "
            f"`{self.vocabulary_path}` (the typed mapping and dispatch queue), and "
            f"`{self.progress_path}` for the PlanDrafter's summary.\n\n"
            "This is a paper review: do not build, run or test anything.\n\n"
            + self._report_instruction("plan_reviewer")
            + " `APPROVE` or `REJECT`."
        )
