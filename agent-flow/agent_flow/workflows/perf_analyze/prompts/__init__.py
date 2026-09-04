import dataclasses
from dataclasses import dataclass

from ._common import EXECUTION_SLURM_BOOTSTRAP, SOL_ANALYZER_CONTEXT, SOL_REPORTER_GUIDANCE
from .analyzer import SYSTEM_PROMPT as ANALYZER_SYSTEM_PROMPT
from .benchmarker import SYSTEM_PROMPT as BENCHMARKER_SYSTEM_PROMPT
from .projector import SYSTEM_PROMPT as PROJECTOR_SYSTEM_PROMPT
from .projector import build_projector_prompt
from .reporter import SYSTEM_PROMPT as REPORTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the four agents in ``PerfAnalyzeWorkflow``.

    Pass a custom bundle to ``PerfAnalyzeWorkflow(..., prompts=...)``
    to swap or extend the default prompts; use ``with_extensions`` to
    derive a bundle that appends domain-specific guidance to the defaults.
    """

    benchmarker: str
    projector: str
    analyzer: str
    reporter: str

    def with_extensions(
        self,
        *,
        benchmarker: str = "",
        projector: str = "",
        analyzer: str = "",
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
            reporter=_append(self.reporter, reporter),
        )


DEFAULT_PROMPTS = PromptBundle(
    benchmarker=BENCHMARKER_SYSTEM_PROMPT,
    projector=PROJECTOR_SYSTEM_PROMPT,
    analyzer=ANALYZER_SYSTEM_PROMPT,
    reporter=REPORTER_SYSTEM_PROMPT,
)


def build_perf_analyze_prompts(
    include_slurm_environment: bool = False,
    include_sol: bool = False,
    sol_methodology: str = "full",
) -> PromptBundle:
    """Return the workflow's prompt bundle, optionally augmented.

    When ``include_slurm_environment`` is True (the task spec carries a
    ``slurm-environment`` block), the Slurm container-bootstrap guidance is
    appended to the two roles that launch servers (benchmarker, analyzer).

    When ``include_sol`` is True (the projector stage is enabled — the
    default, unless the task spec sets ``sol.enabled: false``), the
    projection-consumption guidance is appended to the
    analyzer (context for hypothesis ranking plus the measured↔SOL
    correlation via ``sol_calc.py analyze``) and the reporter (required
    "Projection vs Measured" section + weighing rules). The projector's
    own prompt is always in the bundle — the stage gate lives in the
    workflow.

    ``sol_methodology`` is ``"reduced"`` when this session has
    ``perf-analysis`` but not ``internal-perf-sol-analysis`` (resolved
    before the run by ``sol_methodology.resolve_sol_methodology``); it
    appends the projector's fallback block and changes nothing else.
    """
    bundle = DEFAULT_PROMPTS
    if sol_methodology != "full":
        bundle = dataclasses.replace(bundle, projector=build_projector_prompt(sol_methodology))
    if include_slurm_environment:
        bundle = bundle.with_extensions(
            benchmarker=EXECUTION_SLURM_BOOTSTRAP,
            analyzer=EXECUTION_SLURM_BOOTSTRAP,
        )
    if include_sol:
        bundle = bundle.with_extensions(
            analyzer=SOL_ANALYZER_CONTEXT,
            reporter=SOL_REPORTER_GUIDANCE,
        )
    return bundle


__all__ = [
    "ANALYZER_SYSTEM_PROMPT",
    "BENCHMARKER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "PROJECTOR_SYSTEM_PROMPT",
    "PromptBundle",
    "REPORTER_SYSTEM_PROMPT",
    "build_perf_analyze_prompts",
    "build_projector_prompt",
]
