import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._common import (
    DISAGG_CAMPAIGN,
    EXECUTION_SLURM_BOOTSTRAP,
    KERNEL_COVERAGE_REPORTER_GUIDANCE,
    SOL_ANALYZER_CONTEXT,
    SOL_OPTIMIZE_REPORTER_GUIDANCE,
    SOL_OPTIMIZER_CONTEXT,
    approach_restriction_note,
    kernel_coverage_analyzer_note,
)
from .analyzer import SYSTEM_PROMPT as ANALYZER_SYSTEM_PROMPT
from .benchmarker import SYSTEM_PROMPT as BENCHMARKER_SYSTEM_PROMPT
from .evaluator import SYSTEM_PROMPT as EVALUATOR_SYSTEM_PROMPT
from .optimizer import SYSTEM_PROMPT as OPTIMIZER_SYSTEM_PROMPT
from .projector import SYSTEM_PROMPT as PROJECTOR_SYSTEM_PROMPT
from .projector import build_projector_prompt
from .qa import SYSTEM_PROMPT as QA_SYSTEM_PROMPT
from .reporter import SYSTEM_PROMPT as REPORTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the seven agents in ``PerfOptimizeWorkflow``.

    Pass a custom bundle to ``PerfOptimizeWorkflow(..., prompts=...)``
    to swap or extend the default prompts; use ``with_extensions`` to
    derive a bundle that appends domain-specific guidance to the defaults.
    """

    benchmarker: str
    projector: str
    analyzer: str
    optimizer: str
    evaluator: str
    qa: str
    reporter: str

    def with_extensions(
        self,
        *,
        benchmarker: str = "",
        projector: str = "",
        analyzer: str = "",
        optimizer: str = "",
        evaluator: str = "",
        qa: str = "",
        reporter: str = "",
    ) -> "PromptBundle":
        """Return a new bundle with each non-empty extension appended.

        Empty / whitespace-only extensions leave the corresponding base
        prompt unchanged. Non-empty extensions are joined to the base with
        a single blank line separator.
        """

        def _append(base: str, extra: str) -> str:
            if not extra.strip():
                return base
            return base.rstrip() + "\n\n" + extra

        return PromptBundle(
            benchmarker=_append(self.benchmarker, benchmarker),
            projector=_append(self.projector, projector),
            analyzer=_append(self.analyzer, analyzer),
            optimizer=_append(self.optimizer, optimizer),
            evaluator=_append(self.evaluator, evaluator),
            qa=_append(self.qa, qa),
            reporter=_append(self.reporter, reporter),
        )


DEFAULT_PROMPTS = PromptBundle(
    benchmarker=BENCHMARKER_SYSTEM_PROMPT,
    projector=PROJECTOR_SYSTEM_PROMPT,
    analyzer=ANALYZER_SYSTEM_PROMPT,
    optimizer=OPTIMIZER_SYSTEM_PROMPT,
    evaluator=EVALUATOR_SYSTEM_PROMPT,
    qa=QA_SYSTEM_PROMPT,
    reporter=REPORTER_SYSTEM_PROMPT,
)


def build_perf_optimize_prompts(
    include_slurm_environment: bool = False,
    approaches: Sequence[str] | None = None,
    include_sol: bool = False,
    kernel_coverage: Mapping[str, Any] | None = None,
    sol_methodology: str = "full",
    include_disagg: bool = False,
) -> PromptBundle:
    """Return the workflow's prompt bundle, augmented per the task spec.

    When ``include_slurm_environment`` is True (the task spec carries a
    ``slurm-environment`` block), the Slurm container-bootstrap guidance
    is appended to every role that launches servers — all of them except
    the reporter, which only synthesizes existing artifacts, and the
    projector, which launches no servers either (under Slurm it runs on
    the login node and records the latency constants as unmeasured, per
    its own prompt).

    When ``approaches`` (``optimize.approaches`` from the task spec)
    restricts the run to a subset of the roadmap's approach values, the
    restriction note is appended to every role that plans, applies, or
    judges roadmap items — analyzer, optimizer, evaluator. (QA only
    verifies the final state, so the restriction does not concern it;
    the projector never touches roadmap items.)
    ``None`` or the full set leaves the prompts unchanged.

    When ``include_sol`` is True (the projector stage is enabled — the
    default, unless the task spec sets ``sol.enabled: false``), the
    SOL-consumption guidance is
    appended to the analyzer (rank roadmap items against the projected
    headroom and bound mix, and attribute any remaining gap before
    leaving the roadmap exhausted), the optimizer (aim each item's
    realization at the binding ceiling — context, never an expansion of
    the item), and the reporter (the "Projection vs Measured" section
    with its remaining-gap accountability breakdown). The projector's
    own SOL prompt is always in the bundle — the stage gate lives in
    the workflow. The evaluator and QA deliberately get no SOL context:
    their gates are measured-vs-measured with deterministic thresholds,
    and an analytical ceiling as context could anchor a fresh-eyes
    verdict on a model instead of the measurements. (The evaluator's
    projection-free contribution to gap accountability is the *Gap
    implication* line its negative verdicts always carry.)

    ``sol_methodology`` is ``"reduced"`` when this session has
    ``perf-analysis`` but not ``internal-perf-sol-analysis`` (resolved
    before the run by perf-analyze's
    ``sol_methodology.resolve_sol_methodology``, which this workflow
    shares); it appends the projector's fallback block and changes
    nothing else.

    When ``kernel_coverage`` is set (the validated
    ``profile.kernel_coverage`` block — the per-kernel coverage
    contract), the analyzer gets the coverage-driven ncu targeting, the
    two per-kernel questions (faster? fusible?), and the
    ``kernel_ledger.yaml`` contract with the task's bars interpolated;
    the reporter gets the "Kernel Coverage" accountability section. The
    other roles are unchanged — the ledger is authored by the analyzer
    and consumed by the reporter, with the orchestrator's deterministic
    validation in between.

    When ``include_disagg`` is True (the task spec carries a ``disagg``
    block), the disaggregated-serving section is appended to every role
    that launches or measures a server. It supersedes the single-server
    lifecycle, the tuning-config note and the profiling runs those roles
    otherwise follow, so it is composed last. The reporter and the
    projector are left alone: neither stands up a server, and the
    reporter reads the artifacts the others produced either way.

    Composing it here rather than carrying it unconditionally is what
    keeps the override unambiguous — a role either has the section and it
    applies, or it does not have it at all. The alternative (always
    present, gated on a sentence telling the agent to check
    ``task.yaml``) makes every aggregate campaign pay for it and turns a
    deployment-time fact into a per-turn inference the agent can get
    wrong.
    """
    bundle = DEFAULT_PROMPTS
    if sol_methodology != "full":
        bundle = dataclasses.replace(bundle, projector=build_projector_prompt(sol_methodology))
    restriction = approach_restriction_note(approaches) if approaches is not None else ""
    if restriction:
        bundle = bundle.with_extensions(
            analyzer=restriction,
            optimizer=restriction,
            evaluator=restriction,
        )
    if include_slurm_environment:
        bundle = bundle.with_extensions(
            benchmarker=EXECUTION_SLURM_BOOTSTRAP,
            analyzer=EXECUTION_SLURM_BOOTSTRAP,
            optimizer=EXECUTION_SLURM_BOOTSTRAP,
            evaluator=EXECUTION_SLURM_BOOTSTRAP,
            qa=EXECUTION_SLURM_BOOTSTRAP,
        )
    if include_sol:
        bundle = bundle.with_extensions(
            analyzer=SOL_ANALYZER_CONTEXT,
            optimizer=SOL_OPTIMIZER_CONTEXT,
            reporter=SOL_OPTIMIZE_REPORTER_GUIDANCE,
        )
    if include_disagg:
        bundle = bundle.with_extensions(
            benchmarker=DISAGG_CAMPAIGN,
            analyzer=DISAGG_CAMPAIGN,
            optimizer=DISAGG_CAMPAIGN,
            evaluator=DISAGG_CAMPAIGN,
            qa=DISAGG_CAMPAIGN,
        )
    if kernel_coverage is not None:
        bundle = bundle.with_extensions(
            analyzer=kernel_coverage_analyzer_note(
                float(kernel_coverage["min_share_pct"]),
                float(kernel_coverage["coverage_target_pct"]),
            ),
            reporter=KERNEL_COVERAGE_REPORTER_GUIDANCE,
        )
    return bundle


__all__ = [
    "ANALYZER_SYSTEM_PROMPT",
    "BENCHMARKER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "EVALUATOR_SYSTEM_PROMPT",
    "OPTIMIZER_SYSTEM_PROMPT",
    "PROJECTOR_SYSTEM_PROMPT",
    "PromptBundle",
    "QA_SYSTEM_PROMPT",
    "REPORTER_SYSTEM_PROMPT",
    "build_perf_optimize_prompts",
    "build_projector_prompt",
]
