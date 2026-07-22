"""Tests for the modeling-bringup workflow prompts, CLI wiring, and task-schema validation."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.agent_team import workflow as _agent_team_workflow_module
from agent_flow.workflows.modeling_bringup import cli as _modeling_bringup_cli_module
from agent_flow.workflows.modeling_bringup import task_schema as _task_schema


def _load_agent_team_workflow_module():
    return _agent_team_workflow_module


def _load_modeling_bringup_cli_module():
    return _modeling_bringup_cli_module


def test_default_prompts_match_individual_constants():
    """``DEFAULT_PROMPTS`` must equal the per-agent module constants so the refactor keeps behavior."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert prompts.DEFAULT_PROMPTS.plan_drafter == prompts.PLAN_DRAFTER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.plan_reviewer == prompts.PLAN_REVIEWER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.coder == prompts.CODER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.reviewer == prompts.REVIEWER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.qa == prompts.QA_SYSTEM_PROMPT


def test_with_extensions_empty_returns_unchanged_bundle():
    """Empty / whitespace-only extensions must leave each base prompt byte-for-byte unchanged."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    extended = prompts.DEFAULT_PROMPTS.with_extensions(
        plan_drafter="",
        plan_reviewer="   \n  ",
        coder="",
        reviewer="",
        qa="\n",
    )
    assert extended.plan_drafter == prompts.DEFAULT_PROMPTS.plan_drafter
    assert extended.plan_reviewer == prompts.DEFAULT_PROMPTS.plan_reviewer
    assert extended.coder == prompts.DEFAULT_PROMPTS.coder
    assert extended.reviewer == prompts.DEFAULT_PROMPTS.reviewer
    assert extended.qa == prompts.DEFAULT_PROMPTS.qa


def test_with_extensions_appends_with_blank_line_separator():
    """Non-empty extensions are appended after a single blank line.

    The base is rstripped first so we never get more than one blank line.
    """
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    extended = prompts.DEFAULT_PROMPTS.with_extensions(
        coder="## Modeling-Bringup\nUse mixed precision.\n",
    )
    assert extended.coder.startswith(prompts.CODER_SYSTEM_PROMPT.rstrip())
    assert extended.coder.endswith("\n\n## Modeling-Bringup\nUse mixed precision.\n")
    # No triple newlines anywhere in the joined prompt.
    assert "\n\n\n" not in extended.coder
    # Other agents are untouched when only ``coder`` is extended.
    assert extended.qa == prompts.DEFAULT_PROMPTS.qa


def test_prompt_bundle_is_frozen():
    """Bundle must be immutable so callers can't accidentally mutate the shared default."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    with pytest.raises(Exception):  # FrozenInstanceError subclass of Exception
        prompts.DEFAULT_PROMPTS.coder = "mutated"  # type: ignore[misc]


def test_workflow_uses_default_prompts_when_arg_omitted(tmp_path):
    """Backwards compatibility: omitting ``prompts`` keeps every agent on the original prompts.

    The original prompts are the ``agent_team`` system prompts.
    """
    module = _load_agent_team_workflow_module()
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    workflow = module.AgentTeamWorkflow(workspace=tmp_path)
    try:
        assert workflow.plan_drafter.config.system_prompt == prompts.PLAN_DRAFTER_SYSTEM_PROMPT
        assert workflow.plan_reviewer.config.system_prompt == prompts.PLAN_REVIEWER_SYSTEM_PROMPT
        assert workflow.coder.config.system_prompt == prompts.CODER_SYSTEM_PROMPT
        assert workflow.reviewer.config.system_prompt == prompts.REVIEWER_SYSTEM_PROMPT
        assert workflow.qa.config.system_prompt == prompts.QA_SYSTEM_PROMPT
    finally:
        workflow.close()


def test_workflow_propagates_custom_prompt_bundle(tmp_path):
    """A custom bundle must reach every agent's ``system_prompt``.

    This includes the agents recreated by ``_reset_coder`` / ``_reset_reviewer``.
    """
    module = _load_agent_team_workflow_module()
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    custom = prompts.PromptBundle(
        plan_drafter="PD-SYS",
        plan_reviewer="PR-SYS",
        coder="CODER-SYS",
        reviewer="REVIEWER-SYS",
        qa="QA-SYS",
    )
    workflow = module.AgentTeamWorkflow(workspace=tmp_path, prompts=custom)
    try:
        assert workflow.plan_drafter.config.system_prompt == "PD-SYS"
        assert workflow.plan_reviewer.config.system_prompt == "PR-SYS"
        assert workflow.coder.config.system_prompt == "CODER-SYS"
        assert workflow.reviewer.config.system_prompt == "REVIEWER-SYS"
        assert workflow.qa.config.system_prompt == "QA-SYS"
        # Reset paths must rebuild from the same bundle, not the module
        # defaults.
        workflow._reset_coder()
        workflow._reset_reviewer()
        assert workflow.coder.config.system_prompt == "CODER-SYS"
        assert workflow.reviewer.config.system_prompt == "REVIEWER-SYS"
    finally:
        workflow.close()


def test_modeling_bringup_prompts_match_default_when_extras_empty(monkeypatch):
    """With every ``*_extra.SYSTEM_PROMPT_EXTENSION`` cleared, the modeling bundle matches the default.

    It must be identical to ``DEFAULT_PROMPTS`` so the wrapper itself adds nothing beyond what
    each ``*_extra.py`` contributes.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    for name in (
        "plan_drafter_extra",
        "plan_reviewer_extra",
        "coder_extra",
        "reviewer_extra",
        "qa_extra",
    ):
        module = importlib.import_module(f"agent_flow.workflows.modeling_bringup.prompts.{name}")
        monkeypatch.setattr(module, "SYSTEM_PROMPT_EXTENSION", "")
    importlib.reload(mb_prompts)
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert mb_prompts.MODELING_BRINGUP_PROMPTS == prompts.DEFAULT_PROMPTS


def test_modeling_bringup_prompts_extend_default_for_every_agent():
    """The populated extras must add modeling-bringup guidance to every agent's prompt.

    The base prompt is kept as a prefix.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS
    base = prompts.DEFAULT_PROMPTS
    for name in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        extended = getattr(bundle, name)
        baseline = getattr(base, name)
        assert extended != baseline, f"{name} extension is empty"
        assert extended.startswith(baseline.rstrip()), (
            f"{name} prompt does not start with the base prompt"
        )
        assert "\n\n\n" not in extended, f"{name} prompt has a triple newline from joining"


def test_modeling_bringup_coder_prompt_includes_accuracy_gate_framework():
    """Coder must see the accuracy-gate rules so it can choose the right next step.

    This matters before Reviewer or QA rejects a low-score run.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    coder_prompt = mb_prompts.MODELING_BRINGUP_PROMPTS.coder

    assert "## Accuracy-gate framework" in coder_prompt
    assert "Run baseline first" in coder_prompt
    assert (
        "If the baseline score is below the configured\n"
        "  threshold, keep fixing baseline before running the enabled "
        "configuration."
    ) in coder_prompt


def test_modeling_bringup_coder_prompt_includes_validation_policy_context():
    """Coder must see the strict policy blocks that define what each evidence label has to prove.

    The evidence-label glossary alone is not enough.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    coder_prompt = mb_prompts.MODELING_BRINGUP_PROMPTS.coder

    for snippet in (
        "## Design-review policy",
        "## Attention-validation policy",
        "## Full-model-stage scope",
        "## MoE-validation policy",
        "`temperature=0`, `top_k=1`, no sampling",
        "generate >=32 tokens per prompt for at least 5 fixed",
        "compare per-step logits",
        "Both `source_logit_replay` and `generation_parity` must each "
        "cover the CUDA\n  graph matrix",
    ):
        assert snippet in coder_prompt, (
            f"Coder prompt missing validation-policy context {snippet!r}"
        )


def test_modeling_bringup_prompts_include_accuracy_debugging_methodology():
    """Every modeling-bringup role must share the small failing-prompt loop.

    The loop is for debugging unexpectedly low accuracy.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    required_snippets = (
        "### Accuracy debugging methodology",
        "Export output results from the reference path",
        "Find the `wrong` results",
        "`bad prompts`",
        "per-layer comparison",
        "smaller reproductions",
        "until they are correct",
        "rerun the LLM API smoke, then the accuracy canary, then the full",
        "configured dataset for both baseline and enabled",
    )
    for role in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        prompt = getattr(bundle, role)
        for snippet in required_snippets:
            assert snippet in prompt, (
                f"{role} prompt missing accuracy debugging snippet {snippet!r}"
            )


def test_modeling_bringup_prompts_define_evidence_labels_as_prompt_contract():
    """The modeling-bringup evidence names are prompt-level labels in agent-flow, not built-in tests.

    Every role needs that framing so terms like ``source_logit_replay`` do not look like missing
    local functions.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    required_snippets = (
        "## Modeling-bringup evidence labels",
        "they are **not** built-in functions",
        "automatic controller checks",
        "`reference_tier` and `validation_tier` are the two schema metadata fields",
        "`reference_tier` must be one of `static`, `minimal_golden`, "
        "`reduced_source`,\nor `real_source`",
        "`validation_tier` must be one of `static`, `unit`,\n`integration`, or `real_runtime`",
        "`source_activation_replay`",
        "`source_logit_replay`",
        "`generation_parity`",
        "`real_runtime`",
        "writing only `validation_tier=real_runtime` is not enough",
        "`accuracy_canary`",
        "`cuda_graph_hard_path`",
        "`reference_tier` and `validation_tier`",
        "typed fields with\n  the accepted values listed above",
    )
    for role in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        prompt = getattr(bundle, role)
        for snippet in required_snippets:
            assert snippet in prompt, (
                f"{role} prompt missing evidence-label clarification {snippet!r}"
            )

    assert "test types named in the" not in bundle.plan_drafter
    assert "test types named in the" not in bundle.plan_reviewer
    assert "schema enums, or automatic controller checks" not in bundle.coder
    assert "evidence labels defined" in bundle.plan_drafter
    assert "evidence labels defined" in bundle.plan_reviewer


def test_modeling_bringup_plan_drafter_uses_underscored_tier_fields():
    """PlanDrafter must use the same metadata field spelling that PlanReviewer checks.

    PlanReviewer checks this spelling in acceptance criteria.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    plan_prompt = mb_prompts.MODELING_BRINGUP_PROMPTS.plan_drafter

    assert "**reference_tier**" in plan_prompt
    assert "**validation_tier**" in plan_prompt
    assert "**reference tier**" not in plan_prompt
    assert "**validation tier**" not in plan_prompt


def test_modeling_bringup_cli_module_imports():
    """The CLI module must expose ``main`` and the extended prompt bundle."""
    module = _load_modeling_bringup_cli_module()
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert callable(module.main)
    assert isinstance(module.MODELING_BRINGUP_PROMPTS, prompts.PromptBundle)


def test_modeling_bringup_cli_forwards_argv_verbatim():
    """Regression: the wrapper must forward ``argv`` to the generic agent-team ``main`` unchanged.

    Only a task-scoped prompt bundle is injected. A prior bug parsed each flag manually and
    dropped ``--feedback`` from the rebuilt constructor call, silently discarding user-supplied
    guidance. Delegating with raw ``argv`` keeps every flag (current and future) forwarded by
    construction.

    Inspect the wrapper's source AST: the only call must be
    ``_team_main(argv, prompts=prompts)``.
    """
    import ast
    import inspect

    source = inspect.getsource(_modeling_bringup_cli_module)
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_team_main"
    ]
    assert calls, "wrapper must call _team_main"
    call = calls[0]

    # Positional argv → not unpacked, not reconstructed.
    assert len(call.args) == 1, "wrapper must forward exactly one positional argument (argv)"
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "argv", (
        "wrapper must pass the raw `argv` parameter through, not a "
        "manually reconstructed list — that's how --feedback got dropped "
        "in the previous bug class"
    )

    # Prompts injection is the only kwarg.
    kw_names = sorted(kw.arg for kw in call.keywords)
    assert kw_names == ["prompts"], f"wrapper must only inject prompts=..., got keywords {kw_names}"
    prompts_kw = call.keywords[0]
    assert isinstance(prompts_kw.value, ast.Name) and prompts_kw.value.id == "prompts", (
        "prompts kwarg must pass the task-scoped prompt bundle, "
        "not a reconstructed argv list or a static Slurm bundle"
    )


def test_persistent_deviation_handling_in_modeling_bringup_reviewer():
    """Regression guard: the modeling-bringup Reviewer prompt must include persistent-deviation handling.

    The workflow has no re-plan stage, so a documented deviation that satisfies the acceptance
    criteria is the artifact's permanent shape and should be APPROVEd, not REJECTed iteration after
    iteration. In the gemma-4-exp7 worklog the same plan deviation was re-cited across 14 iterations
    without ever being explicitly accepted.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    assert "Persistent-deviation handling" in bundle.reviewer, (
        "Reviewer prompt must include persistent-deviation handling guidance"
    )
    assert "No re-plan stage" in bundle.reviewer, (
        "Reviewer prompt must spell out that the workflow has no "
        "re-plan stage so deviations satisfying criteria should be "
        "APPROVEd"
    )


def test_no_replan_stage_framing_in_generic_reviewer():
    """The generic agent_team Reviewer prompt must reflect the workflow's actual structure.

    No re-plan stage exists, so the Reviewer's anchor is acceptance-criteria.md and documented
    deviations satisfying criteria must be APPROVEd.
    """
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert "No re-plan stage" in prompts.REVIEWER_SYSTEM_PROMPT
    assert "Persistent-deviation handling" in prompts.REVIEWER_SYSTEM_PROMPT


def test_coder_prompt_references_no_replan_stage():
    """The Coder prompt should tell the Coder not to re-paste accepted deviations every iteration.

    A deviation is the artifact's permanent shape, not a replan request.
    """
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert "no re-plan stage" in prompts.CODER_SYSTEM_PROMPT
    assert "iteration noise" in prompts.CODER_SYSTEM_PROMPT


def test_status_done_todo_rubric_in_coder_and_reviewer_prompts():
    """status.md's ``Done / TODO`` section is the rolling progress summary read at a glance.

    The next agent and the human read it at a glance. Both the Coder (who produces the first draft
    each iteration) and the Reviewer (who adjusts it post-review) must carry the rubric in their
    populated modeling-bringup prompts.

    QA is excluded by design — its verdict must be grounded in
    ``task.yaml`` and ``acceptance-criteria.md`` alone, so the rubric
    must NOT leak into the QA prompt.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    for role in ("coder", "reviewer"):
        prompt = getattr(bundle, role)
        # Section heading the rubric mandates.
        assert "## status.md `Done / TODO` section" in prompt, (
            f"{role} prompt missing the Done/TODO rubric heading"
        )
        # The two required subsection headings inside status.md.
        assert "### Done" in prompt, f"{role} prompt missing the Done subsection heading"
        assert "### TODO" in prompt, f"{role} prompt missing the TODO subsection heading"
        # The "every acceptance-criteria item appears in exactly one"
        # rule — the load-bearing invariant that keeps the section
        # honest across iterations.
        assert "exactly one of `Done` or `TODO`" in prompt, (
            f"{role} prompt missing the exactly-one-of-Done-or-TODO invariant"
        )
        # Executed-evidence requirement for the Done list — otherwise
        # the section becomes wishful thinking.
        assert "executed evidence" in prompt, (
            f"{role} prompt missing the executed-evidence rule for the Done list"
        )
        # TODO requires a named blocker + planned next step.
        assert "TBD" in prompt, f"{role} prompt does not forbid `TBD` placeholders in the TODO list"
        # The Coder-drafts / Reviewer-adjusts handoff is what makes
        # this work across the two agents — verify both names appear
        # in the rubric narrative.
        assert "Coder" in prompt and "Reviewer" in prompt, (
            f"{role} prompt does not name both Coder and Reviewer in the rubric"
        )

    # QA must not carry the rubric: QA is stateless and is forbidden
    # from reading status.md.
    assert "## status.md `Done / TODO` section" not in bundle.qa, (
        "QA prompt unexpectedly carries the status.md rubric; "
        "status.md is not in QA's source-of-truth set"
    )


def test_hf_reference_golden_policy_in_planner_and_coder_prompts():
    """The golden-generate cross-check guidance reaches exactly two roles.

    It must reach the PlanDrafter (so the plan/criteria require the
    self-check) and the Coder (so it is implemented and committed as a
    fixture), and only those two roles.

    Scope is deliberate: the policy is wired into ``plan_drafter_extra``
    and ``coder_extra`` only, so it must NOT leak into the reviewer, QA,
    or plan-reviewer prompts.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    heading = "## Hand-written HF reference — golden-generate cross-check"
    for role in ("plan_drafter", "coder"):
        prompt = getattr(bundle, role)
        assert heading in prompt, f"{role} prompt missing the golden-generate cross-check heading"
        # The load-bearing instruction: commit a native-generate golden
        # fixture and assert token-for-token equality.
        assert "checked-in golden fixture" in prompt, (
            f"{role} prompt missing the committed golden-fixture rule"
        )
        assert "token-for-token" in prompt, (
            f"{role} prompt missing the token-for-token equality assertion"
        )
        # The hard constraint that protects the runtime's pinned deps.
        assert "Never mutate the repo's pinned `transformers`" in prompt, (
            f"{role} prompt missing the do-not-mutate-pinned-transformers constraint"
        )

    # Out-of-scope roles must not carry the policy.
    for role in ("plan_reviewer", "reviewer", "qa"):
        assert heading not in getattr(bundle, role), (
            f"{role} prompt unexpectedly carries the golden-generate "
            f"cross-check policy; it is scoped to planner + coder only"
        )


def test_qa_prompt_has_final_report_contract():
    """The modeling-bringup QA prompt must surface the final-report contract.

    The contract covers the artifact path, both terminal statuses, the section skeleton, the
    required dimensions (accuracy / performance vs HF / feature coverage / parallelism), the
    backend-selection and trtllm-repo-changes implementation sections, and the ``Not measured``
    fallback rule. Each piece is a load-bearing part of the closure artifact the user reads after
    a run.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    qa_prompt = mb_prompts.MODELING_BRINGUP_PROMPTS.qa

    # Artifact location and overwrite semantics.
    assert "<workspace>/final-report.md" in qa_prompt
    assert "Overwrite" in qa_prompt or "overwrite" in qa_prompt

    # Both terminal statuses are spelled out.
    assert "Status: ACCEPT" in qa_prompt
    assert "Status: INCOMPLETE" in qa_prompt

    # Top-level skeleton: two parts with the headings used by the
    # template. Diff-stability across bring-ups depends on these exact
    # headings making it into the prompt.
    assert "Part 1 — Acceptance-criteria status" in qa_prompt
    assert "Part 2 — Implementation overview" in qa_prompt

    # Part 1 required subsections.
    assert "Per-criterion outcome" in qa_prompt
    assert "Accuracy" in qa_prompt
    assert "Performance vs HuggingFace" in qa_prompt
    assert "Feature coverage" in qa_prompt
    assert "Parallelism coverage" in qa_prompt

    # Feature-coverage rows: CUDA graph (with hard-path qualifier),
    # overlap scheduler, chunked prefill, KVCacheManagerV2.
    assert "CUDA graph (hard-path)" in qa_prompt
    assert "Overlap scheduler" in qa_prompt
    assert "Chunked prefill" in qa_prompt
    assert "KVCacheManagerV2" in qa_prompt

    # Parallelism-coverage rows: TP / PP / DP / EP must all appear.
    for strategy in ("| TP |", "| PP |", "| DP |", "| EP |"):
        assert strategy in qa_prompt, f"parallelism row {strategy!r} missing from QA prompt"

    # Part 2 required subsections.
    assert "New model structure" in qa_prompt
    assert "Backend selection per module" in qa_prompt
    assert "TensorRT-LLM repo changes" in qa_prompt
    # The Backend table must list at minimum attention, MoE, and the
    # KV-cache manager — those are the backend-bearing modules.
    assert "| Attention |" in qa_prompt
    assert "| MoE (if present) |" in qa_prompt
    assert "| KV-cache manager |" in qa_prompt
    # Repo changes are sourced from observed git state, not memory.
    assert "git diff --name-status" in qa_prompt

    # Fallback rule for non-applicable signals.
    assert "Not measured" in qa_prompt
    # And the negative-instruction half: blank / N/A / TBD are not
    # acceptable substitutes (prevents QA from silently omitting a
    # required cell).
    for forbidden in ("blank", "N/A", "TBD"):
        assert forbidden in qa_prompt, (
            f"QA prompt does not forbid {forbidden!r} as a non-measurement placeholder"
        )

    # Ordering: the report write must happen before
    # ``append_qa_progress`` so an interrupt between the two does not
    # lose the closure artifact.
    write_idx = qa_prompt.find("Write the file **before** you call `append_qa_progress`")
    assert write_idx != -1, (
        "QA prompt must explicitly order the report write before append_qa_progress"
    )


def test_modeling_bringup_default_prompts_omit_slurm_guidance():
    """The module-level modeling-bringup bundle is for local tasks.

    Slurm/container guidance is injected only for validated task specs that contain
    ``slurm-environment``.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    for name in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        prompt = getattr(bundle, name)
        assert "Slurm container bootstrap" not in prompt
        assert "slurm-environment" not in prompt

    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "test_command.md" not in prompt


def test_modeling_bringup_prompts_require_srun_for_slurm():
    """Coder, Reviewer, and QA extensions must mandate cached commands run from the Slurm login node.

    The cached ``test_command.md`` commands must be runnable there — i.e. they spell out the
    ``srun`` / login-node requirement, not just "Slurm".
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(include_slurm_environment=True)
    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "srun" in prompt
        assert "login node" in prompt


def test_modeling_bringup_test_command_md_is_slurm_only():
    """Coder, Reviewer, and QA extensions must gate ``test_command.md`` on a Slurm environment.

    On local (non-Slurm) hosts the file is not created or maintained at all.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(include_slurm_environment=True)
    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "Slurm-only" in prompt
        # The gate must explicitly disable the mechanism on non-Slurm
        # hosts, not just call out that Slurm commands need srun.
        # Allow line wraps between the verbs and the file name by
        # collapsing whitespace before substring-matching.
        flat = " ".join(prompt.split())
        assert "do not create, read, or maintain `test_command.md`" in flat
        assert "non-Slurm" in prompt


def test_modeling_bringup_prompts_prefer_specialist_command_help():
    """The modeling-bringup extensions should encourage specialist help for `trtllm-*` drafting.

    They must do so without naming a plugin users may not have installed.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(include_slurm_environment=True)
    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "test-command specialist" in prompt
        assert "trtllm-agent-toolkit" not in prompt
        assert "test_command.md" in prompt


def test_modeling_bringup_prompts_require_minimal_test_command_cache():
    """The modeling-bringup extensions must mandate a minimal ``test_command.md``.

    It must hold only currently-passing commands, no inline supersede prose / archive section /
    top-of-file narrative, and a structured per-entry template that ties each command back to the
    acceptance criteria it verifies.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(include_slurm_environment=True)
    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        # Delete-on-failure / on-supersede rather than archive.
        assert "delete the entire entry" in prompt
        assert "progress.yaml" in prompt
        # Per-entry structured template fields.
        assert "criteria:" in prompt
        assert "verified:" in prompt
        assert "outputs:" in prompt
        # No top-of-file supersede narrative.
        assert "No top-of-file narrative" in prompt


def test_modeling_bringup_prompts_include_container_bootstrap():
    """A Slurm-enabled modeling-bringup prompt bundle must carry the container bootstrap.

    It must also use the ``slurm-environment`` YAML fields.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(include_slurm_environment=True)
    expected_lines = (
        "Slurm container bootstrap for TensorRT-LLM bring-up",
        "./scripts/build_wheel.py --trt_root /usr/local/tensorrt",
        "pip install -e .[devel]",
        "curl -fsSL https://claude.ai/install.sh | bash",
        "pip install claude-agent-sdk",
        "slurm-environment",
        "slurm_partition",
        "docker_image",
        "same path inside the container",
        "cd <trtllm_repo_path>",
        "--partition=<slurm_partition>",
        "--container-image=<docker_image>",
        "--container-mounts=<trtllm_repo_path>:<trtllm_repo_path>",
    )
    for name in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        prompt = getattr(bundle, name)
        for needle in expected_lines:
            assert needle in prompt, (
                f"{name} prompt is missing container-bootstrap line: {needle!r}"
            )
        assert '-a "90;100-real"' not in prompt
        assert "--qos" not in prompt


def test_modeling_bringup_workflow_uses_extended_bundle(tmp_path, monkeypatch):
    """Override one agent's extension and verify only that agent's prompt changes.

    The others keep their normally-populated extensions.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    baseline_bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    coder_extra = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts.coder_extra"
    )
    monkeypatch.setattr(coder_extra, "SYSTEM_PROMPT_EXTENSION", "## MB\nExtra coder guidance.\n")
    importlib.reload(mb_prompts)

    workflow_module = _load_agent_team_workflow_module()
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        prompts=mb_prompts.MODELING_BRINGUP_PROMPTS,
    )
    try:
        assert workflow.coder.config.system_prompt.endswith("\n\n## MB\nExtra coder guidance.\n")
        assert workflow.coder.config.system_prompt.startswith(prompts.CODER_SYSTEM_PROMPT.rstrip())
        # Agents whose extras were not monkeypatched keep their normally-
        # populated bundle prompts.
        assert workflow.qa.config.system_prompt == baseline_bundle.qa
        assert workflow.plan_drafter.config.system_prompt == baseline_bundle.plan_drafter
    finally:
        workflow.close()
        # Reset the override so subsequent tests see the populated bundle.
        monkeypatch.undo()
        importlib.reload(mb_prompts)


def _print_baseline_context_table(
    variant: str,
    rows: list[tuple[str, str, float, int | None, int | None]],
    unavailable: list[str],
) -> None:
    """Print an overview table of each role's baseline ("start") context usage.

    Visible with ``pytest -s``. Purely informational — the caller's
    assertions still enforce the 20% budget.
    """
    header = ("role", "backend", "context%", "tokens / window")
    table_rows = [
        (name, backend, f"{percentage:.1f}%", f"{tokens} / {window}")
        for name, backend, percentage, tokens, window in rows
    ]
    columns = list(zip(header, *table_rows)) if table_rows else [(h,) for h in header]
    widths = [max(len(str(cell)) for cell in col) for col in columns]

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(cells))

    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(f"\n=== modeling-bringup baseline context usage ({variant}) ===")
    print(_fmt(header))
    print(rule)
    for row in table_rows:
        print(_fmt(row))
    if not table_rows:
        print("(no live backend reported context usage)")
    if unavailable:
        print(f"skipped: {'; '.join(unavailable)}")


@pytest.mark.parametrize(
    "include_slurm_environment",
    [False, True],
    ids=["local", "slurm"],
)
def test_modeling_bringup_claude_agents_baseline_context_under_20_percent(
    tmp_path, include_slurm_environment
):
    """Each Claude-backed modeling-bringup agent's pre-input context must stay < 20%.

    The modeling-bringup bundle layers substantial domain guidance
    (source boundary, validation policies, accuracy gate, evidence
    labels, and — for Slurm tasks — the container bootstrap) on top of
    the generic ``agent_team`` prompts. This guards that the larger
    prompt, together with the Claude Code preset, built-in tools, MCP
    tool schemas, and memory the agent carries *before any user input*,
    still leaves most of the context window free for real work. The Slurm
    variant carries the largest prompts, so it is the worst case for
    bloat — both variants are exercised.

    Mirrors ``test_claude_agents_baseline_context_under_20_percent`` in
    the agent_team suite, but builds the workflow with the
    modeling-bringup bundle. Uses the live local ``/context`` control
    request via ``fetch_baseline_context_usage`` (no model call), so it
    runs for real wherever the ``claude`` CLI is installed and skips
    cleanly when the backend cannot be reached. A baseline at or above
    20% fails; an unreachable backend skips.
    """
    variant = "slurm" if include_slurm_environment else "local"
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(
        include_slurm_environment=include_slurm_environment
    )
    workflow_module = _load_agent_team_workflow_module()
    workflow = workflow_module.AgentTeamWorkflow(workspace=tmp_path, prompts=bundle)
    try:
        # Only Claude Code-backed agents report a pre-input /context
        # breakdown; the codex-backed plan_drafter / reviewer return None.
        claude_agents = {
            name: agent
            for name, agent in (
                ("coder", workflow.coder),
                ("plan_reviewer", workflow.plan_reviewer),
                ("qa", workflow.qa),
            )
            if agent.config.backend.kind == "claude-code"
        }
        assert claude_agents, "expected at least one Claude Code-backed modeling-bringup agent"

        unavailable: list[str] = []
        rows: list[tuple[str, str, float, int | None, int | None]] = []
        for name, agent in claude_agents.items():
            backend_kind = agent.config.backend.kind
            try:
                usage = agent.fetch_baseline_context_usage()
            except Exception as exc:  # CLI missing / backend unreachable
                unavailable.append(f"{name} ({variant}): {exc}")
                continue
            if usage is None or usage.context_percentage is None:
                unavailable.append(f"{name} ({variant}): no pre-input context usage")
                continue
            rows.append(
                (
                    name,
                    backend_kind,
                    usage.context_percentage,
                    usage.context_tokens,
                    usage.context_window,
                )
            )

        # Overview of where each role starts before any real work.
        _print_baseline_context_table(variant, rows, unavailable)

        for name, backend_kind, percentage, tokens, window in rows:
            assert percentage < 20.0, (
                f"{name} ({variant}) pre-input context {percentage:.1f}% "
                f"exceeds the 20% budget "
                f"({tokens}/{window} tokens)"
            )
        if not rows:
            pytest.skip("no live backend reported context usage: " + "; ".join(unavailable))
    finally:
        workflow.close()


def _valid_task_payload(tmp_path) -> dict:
    """Return a fully-populated payload whose three required paths exist on disk under ``tmp_path``."""
    ref = tmp_path / "modeling.py"
    ref.write_text("# stub\n", encoding="utf-8")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    repo = tmp_path / "trtllm-repo"
    repo.mkdir()
    return {
        "reference_code_path": str(ref),
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "completion_criteria": ["GSM8K > 95"],
        "implements_tips": ["Use KVCacheManagerV2"],
    }


def _write_task_yaml(tmp_path, payload) -> Path:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_modeling_bringup_cli_omits_slurm_prompts_without_slurm_environment(tmp_path, monkeypatch):
    """The CLI should pass a local-host prompt bundle for specs without ``slurm-environment``."""
    payload = _valid_task_payload(tmp_path)
    task_path = _write_task_yaml(tmp_path, payload)
    argv = ["--task", str(task_path), "--workspace", str(tmp_path / "work")]
    captured = {}

    def fake_team_main(forwarded_argv, *, prompts):
        captured["argv"] = forwarded_argv
        captured["prompts"] = prompts

    monkeypatch.setattr(_modeling_bringup_cli_module, "_team_main", fake_team_main)

    _modeling_bringup_cli_module.main(argv)

    assert captured["argv"] == argv
    assert "Slurm container bootstrap" not in captured["prompts"].coder
    assert "test_command.md" not in captured["prompts"].coder


def test_modeling_bringup_cli_includes_slurm_prompts_with_slurm_environment(tmp_path, monkeypatch):
    """The ``slurm-environment`` field is the switch for Slurm prompt injection."""
    payload = _valid_task_payload(tmp_path)
    payload["slurm-environment"] = {
        "slurm_partition": "gb200",
        "docker_image": "/containers/trtllm.sqsh",
    }
    task_path = _write_task_yaml(tmp_path, payload)
    argv = ["--task", str(task_path), "--workspace", str(tmp_path / "work")]
    captured = {}

    def fake_team_main(forwarded_argv, *, prompts):
        captured["argv"] = forwarded_argv
        captured["prompts"] = prompts

    monkeypatch.setattr(_modeling_bringup_cli_module, "_team_main", fake_team_main)

    _modeling_bringup_cli_module.main(argv)

    assert captured["argv"] == argv
    assert "Slurm container bootstrap" in captured["prompts"].coder
    assert "test_command.md" in captured["prompts"].coder
    assert "slurm_partition" in captured["prompts"].plan_drafter


def test_task_schema_accepts_valid_payload(tmp_path):
    """A complete, well-typed YAML loads cleanly and round-trips its fields.

    Extra keys are included too, so downstream agents see the full spec.
    """
    payload = _valid_task_payload(tmp_path)
    payload["extra_user_key"] = "preserved"
    path = _write_task_yaml(tmp_path, payload)

    data = _task_schema.load_and_validate_task_yaml(path)

    assert data["reference_code_path"] == payload["reference_code_path"]
    assert data["checkpoint_path"] == payload["checkpoint_path"]
    assert data["trtllm_repo_path"] == payload["trtllm_repo_path"]
    assert data["completion_criteria"] == ["GSM8K > 95"]
    assert data["implements_tips"] == ["Use KVCacheManagerV2"]
    assert data["extra_user_key"] == "preserved"
    assert not _task_schema.has_slurm_environment(data)


def test_task_schema_accepts_slurm_environment(tmp_path):
    """A Slurm task must name the partition and container image.

    The image path is not forced to exist on the local host.
    """
    payload = _valid_task_payload(tmp_path)
    payload["slurm-environment"] = {
        "slurm_partition": "gb200",
        "docker_image": "/containers/trtllm.sqsh",
    }
    path = _write_task_yaml(tmp_path, payload)

    data = _task_schema.load_and_validate_task_yaml(path)

    assert _task_schema.has_slurm_environment(data)
    assert data["slurm-environment"] == {
        "slurm_partition": "gb200",
        "docker_image": "/containers/trtllm.sqsh",
    }


@pytest.mark.parametrize(
    ("slurm_environment", "expected"),
    [
        ("not a mapping", "must be a mapping"),
        ({}, "slurm_partition"),
        ({"slurm_partition": "gb200"}, "docker_image"),
        ({"slurm_partition": " ", "docker_image": "/containers/trtllm.sqsh"}, "slurm_partition"),
        ({"slurm_partition": "gb200", "docker_image": 42}, "docker_image"),
    ],
)
def test_task_schema_rejects_invalid_slurm_environment(tmp_path, slurm_environment, expected):
    """If present, ``slurm-environment`` must contain the two fields the prompt builder relies on."""
    payload = _valid_task_payload(tmp_path)
    payload["slurm-environment"] = slurm_environment
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert expected in str(exc.value)


@pytest.mark.parametrize(
    "missing_field",
    ["reference_code_path", "checkpoint_path", "trtllm_repo_path"],
)
def test_task_schema_rejects_missing_required_field(tmp_path, missing_field):
    """Each required path field, when omitted, must be named in the error message."""
    payload = _valid_task_payload(tmp_path)
    del payload[missing_field]
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert missing_field in str(exc.value)


def test_task_schema_batches_all_missing_required_fields(tmp_path):
    """A single load reports every missing required field at once.

    The user does not have to fix them one at a time.
    """
    payload = _valid_task_payload(tmp_path)
    del payload["reference_code_path"]
    del payload["checkpoint_path"]
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    msg = str(exc.value)
    assert "reference_code_path" in msg
    assert "checkpoint_path" in msg


def test_task_schema_rejects_non_existent_path(tmp_path):
    """A required path that does not exist on disk is reported with the offending value.

    This lets the user see what they need to fix.
    """
    payload = _valid_task_payload(tmp_path)
    payload["checkpoint_path"] = "/path/that/should/not/exist/anywhere/xyz"
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert "checkpoint_path" in str(exc.value)
    assert "/path/that/should/not/exist/anywhere/xyz" in str(exc.value)


def test_task_schema_rejects_empty_string_required_field(tmp_path):
    """Whitespace-only / empty strings count as missing.

    The path probe would silently match the current directory, so reject up front.
    """
    payload = _valid_task_payload(tmp_path)
    payload["trtllm_repo_path"] = "   "
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert "trtllm_repo_path" in str(exc.value)


@pytest.mark.parametrize("field", ["completion_criteria", "implements_tips"])
def test_task_schema_defaults_optional_lists_to_empty(tmp_path, field):
    """Missing optional list fields default to ``[]`` so callers can rely on the key being present."""
    payload = _valid_task_payload(tmp_path)
    del payload[field]
    path = _write_task_yaml(tmp_path, payload)

    data = _task_schema.load_and_validate_task_yaml(path)
    assert data[field] == []


@pytest.mark.parametrize("field", ["completion_criteria", "implements_tips"])
def test_task_schema_rejects_non_list_optional_field(tmp_path, field):
    """If the user supplies an optional field, it must be a list of strings.

    A scalar or dict is a type error.
    """
    payload = _valid_task_payload(tmp_path)
    payload[field] = "not a list"
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert field in str(exc.value)


@pytest.mark.parametrize("field", ["completion_criteria", "implements_tips"])
def test_task_schema_rejects_non_string_list_items(tmp_path, field):
    """Items inside the optional lists must be strings, not numbers or dicts."""
    payload = _valid_task_payload(tmp_path)
    payload[field] = ["valid", 42]
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert field in str(exc.value)


def test_task_schema_accepts_empty_optional_lists(tmp_path):
    """An explicitly-empty list is the same as the field being missing."""
    payload = _valid_task_payload(tmp_path)
    payload["completion_criteria"] = []
    payload["implements_tips"] = []
    path = _write_task_yaml(tmp_path, payload)

    data = _task_schema.load_and_validate_task_yaml(path)
    assert data["completion_criteria"] == []
    assert data["implements_tips"] == []


def test_task_schema_rejects_malformed_yaml(tmp_path):
    """A file that is not valid YAML is rejected with the file path so the user can find it."""
    path = tmp_path / "broken.yaml"
    path.write_text("reference_code_path: [unclosed\n", encoding="utf-8")

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert "not valid YAML" in str(exc.value)
    assert str(path) in str(exc.value)


def test_task_schema_rejects_non_mapping_top_level(tmp_path):
    """A YAML scalar or list at the top level is not a usable task spec — must be a mapping."""
    path = tmp_path / "scalar.yaml"
    path.write_text("just-a-string\n", encoding="utf-8")

    with pytest.raises(_task_schema.TaskSchemaError, match="mapping"):
        _task_schema.load_and_validate_task_yaml(path)


def test_task_schema_rejects_missing_file(tmp_path):
    """A non-existent path is rejected with the path quoted in the error so the user can see the typo."""
    missing = tmp_path / "absent.yaml"

    with pytest.raises(_task_schema.TaskSchemaError, match="not found"):
        _task_schema.load_and_validate_task_yaml(missing)


# -------------------------------------------------------------- Stage/Goal
#
# The Stage/Goal control-flow protocol lives entirely in the
# modeling-bringup prompt extensions plus a single ``Replan mode: ...``
# line injected into the Reviewer's per-turn user prompt. It is gated on
# --replan-on-qa: ``build_modeling_bringup_prompts(replan_on_qa=True)``
# appends each ``*_extra.STAGE_GOAL_EXTENSION``; without the flag the
# agents run on the base flat-plan prompts. The tests below confirm each
# extension carries its load-bearing fragments. They are intentionally
# substring-style: schema *correctness* is exercised by end-to-end runs
# (manual or in CI), but the prompt scaffolding must stay intact across
# refactors of the extension files.

_STAGE_GOAL_HEADINGS = (
    "## Stage/Goal plan schema",
    "## Replan lock matrix",
    "## Replan decision mapping in Stage/Goal mode",
    "## Stage/Goal schema enforcement",
    "## Stage/Goal protocol — working a single Goal per turn",
    "## Stage/Goal state machine",
    "## Stage/Goal-aware verification scope",
)


def test_stage_goal_blocks_absent_without_replan_on_qa():
    """Stage/Goal is replan-only: the default bundle carries none of it.

    Without ``--replan-on-qa`` the agents run on the base flat-plan
    prompts, so no role's prompt may contain a Stage/Goal block.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    for bundle in (
        mb_prompts.MODELING_BRINGUP_PROMPTS,
        mb_prompts.build_modeling_bringup_prompts(replan_on_qa=False),
    ):
        for role in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
            prompt = getattr(bundle, role)
            for heading in _STAGE_GOAL_HEADINGS:
                assert heading not in prompt, (
                    f"{role} prompt leaks {heading!r} without replan-on-qa"
                )


def test_modeling_bringup_cli_gates_stage_goal_prompts_on_replan_flag(tmp_path, monkeypatch):
    """The CLI threads ``--replan-on-qa`` into the prompt build.

    Stage/Goal blocks reach the agents only when the flag is on.
    """
    payload = _valid_task_payload(tmp_path)
    task_path = _write_task_yaml(tmp_path, payload)
    captured = {}

    def fake_team_main(forwarded_argv, *, prompts):
        captured["prompts"] = prompts

    monkeypatch.setattr(_modeling_bringup_cli_module, "_team_main", fake_team_main)

    base_argv = ["--task", str(task_path), "--workspace", str(tmp_path / "work")]
    _modeling_bringup_cli_module.main(base_argv)
    assert "## Stage/Goal plan schema" not in captured["prompts"].plan_drafter
    assert "## Stage/Goal protocol — working a single Goal per turn" not in (
        captured["prompts"].coder
    )

    _modeling_bringup_cli_module.main(base_argv + ["--replan-on-qa"])
    assert "## Stage/Goal plan schema" in captured["prompts"].plan_drafter
    assert "## Stage/Goal protocol — working a single Goal per turn" in captured["prompts"].coder
    assert "## Stage/Goal state machine" in captured["prompts"].reviewer
    assert "## Stage/Goal-aware verification scope" in captured["prompts"].qa
    assert "## Stage/Goal schema enforcement" in captured["prompts"].plan_reviewer


def test_plan_drafter_extra_carries_stage_goal_schema():
    """PlanDrafter must learn the two-level Stage/Goal hierarchy.

    The matching ``acceptance-criteria.md`` partitioning comes with it.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    plan = mb_prompts.build_modeling_bringup_prompts(replan_on_qa=True).plan_drafter

    assert "## Stage/Goal plan schema" in plan
    assert "### Stage 1:" in plan
    assert "Exit criterion:" in plan
    assert "Goal 1.1:" in plan
    assert "## Stage 1 — accuracy convergence" in plan
    # Lock matrix is the contract for replan turns.
    assert "## Replan lock matrix" in plan
    assert "CLOSED" in plan and "IN_PROGRESS" in plan and "PENDING" in plan
    # Decision mapping is Stage-aware in replan mode.
    assert "## Replan decision mapping in Stage/Goal mode" in plan
    # QA REJECT remediation: insert a gap-fix Stage immediately
    # after the failing CLOSED Stage; downstream PENDING Stages
    # get renumbered mechanically AND may be content-revised
    # under PlanDrafter's judgement (with per-edit justification).
    # Demoting a CLOSED Stage back to IN_PROGRESS is forbidden.
    assert "insert a new gap-fix Stage immediately after the" in plan
    assert "Do not demote, reopen, or otherwise mutate Stage N" in plan
    assert "CLOSED Stages keep their original numbers" in plan
    assert "(N+1).1" in plan
    # PENDING-Stage content revision is allowed under judgement,
    # not silently auto-applied — the lock matrix permits it but
    # each edit must be justified.
    assert "mandatory and mechanical" in plan
    assert "optional and\n  judgement-driven" in plan
    # The old "demote the Stage back to IN_PROGRESS" and the
    # tail-append fallback for gap-fix Stages must not reappear.
    assert "Demote the Stage header" not in plan
    assert "append a new gap-fix Stage at the tail" not in plan


def test_plan_reviewer_extra_enforces_stage_goal_layout():
    """PlanReviewer must enforce the Stage/Goal layout.

    It REJECTs a plan that doesn't use the layout, and auto-REJECTs
    CLOSED-Stage edits during replan review.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    plan_reviewer = mb_prompts.build_modeling_bringup_prompts(replan_on_qa=True).plan_reviewer

    assert "## Stage/Goal schema enforcement" in plan_reviewer
    assert "### Stage <N>: <label>" in plan_reviewer
    assert "## Stage <N> — <label>" in plan_reviewer
    assert "## Replan-review lock-matrix enforcement" in plan_reviewer
    assert "CLOSED Stages are fully locked" in plan_reviewer
    # New auto-REJECT rules aligned with the gap-fix-Stage remediation:
    # CLOSED Stages are immutable (no demotion, no row reset, no
    # renumber); gap-fix Stage must be inserted immediately after the
    # failing CLOSED Stage with downstream PENDING Stages renumbered;
    # PENDING content revisions during a gap-fix turn must each be
    # justified in the PlanDrafter's summary.
    assert "demoted back to `— IN_PROGRESS`" in plan_reviewer
    assert "immediately after the failing CLOSED Stage" in plan_reviewer
    assert "PENDING Stages were **not** renumbered" in plan_reviewer
    assert "permanently pinned" in plan_reviewer
    assert "per-edit justification" in plan_reviewer
    assert "silent relaxations or silent Goal" in plan_reviewer


def test_coder_extra_carries_single_goal_protocol():
    """Coder must learn to work a single ``[Doing]`` goal per turn.

    The ``BLOCKER:`` line is reserved for when it is truly stuck.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    coder = mb_prompts.build_modeling_bringup_prompts(replan_on_qa=True).coder

    assert "## Stage/Goal protocol — working a single Goal per turn" in coder
    assert "## Stages & Goals" in coder
    assert "[Doing]" in coder
    # State-immutability rule: Coder doesn't bump iterations or flip
    # state — only Reviewer does.
    assert "Reviewer maintains the count" in coder
    # The BLOCKER line is the only worker-side Failed-eligible signal.
    assert "BLOCKER:" in coder
    assert "Do not write `BLOCKER:` casually" in coder
    # No minimum-iterations floor: the Coder may earn a [Failed]
    # close on iteration 1 if BLOCKER is genuine and Reviewer
    # independently agrees. The 5-iter gate is removed.
    assert "no minimum-iterations floor" in coder
    assert "iterations >= 5" not in coder
    assert "at least five iterations" not in coder


def test_reviewer_extra_carries_stage_state_machine():
    """Reviewer must own the Stage/Goal state machine.

    That covers the counter, the Failed-trigger conjunction, the
    mode-aware APPROVE gate, and the mandatory ``Stage closed: Stage
    <N>`` line on APPROVE.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    reviewer = mb_prompts.build_modeling_bringup_prompts(replan_on_qa=True).reviewer

    assert "## Stage/Goal state machine" in reviewer
    # Five-value internal decision.
    assert "iterations=N+1" in reviewer
    assert "(iterations=0)" in reviewer
    # Failed hard conjunction needs both: Coder's BLOCKER: line, and
    # Reviewer's independent confirmation. There is NO
    # minimum-iterations floor — a Goal can be marked [Failed] on
    # iteration 1 if both conditions hold.
    assert "BLOCKER:" in reviewer
    assert "independently confirm" in reviewer
    assert "no minimum-iterations floor" in reviewer
    assert "iterations >= 5" not in reviewer
    assert "fewer than 5 iterations" not in reviewer
    # QA-REJECT bookkeeping must teach the gap-fix-Stage protocol, not
    # the old demote-and-reopen rollback that contradicts the
    # PlanDrafter's lock matrix (CLOSED Stages are immutable).
    assert "gap-fix Stage\nimmediately after the failing CLOSED Stage" in reviewer
    assert "never demotes" in reviewer
    assert "demotes a `— CLOSED (pending QA)` Stage back" not in reviewer
    assert "reopens the failing Goals" not in reviewer
    # A single unreachable acceptance item under the active Goal is
    # sufficient for [Failed]; remaining items in the same Goal need
    # NOT be independently exhausted, because a Goal cannot close
    # while any of its items is unreachable. The old "no untried
    # approach is visible" framing required exhausting every item,
    # which deadlocks Goals whose unreachable items co-exist with
    # achievable items (e.g. Step3.7 Stage 2 Goal 2.1, where
    # generation_parity was reachable but full GSM8K was not).
    assert "at least one acceptance item" in reviewer
    assert "One unreachable item is sufficient" in reviewer
    assert "remaining items" in reviewer
    # Old global-exhaustion phrasing must not reappear: it conflates
    # "Goal unreachable" with "every item exhausted" and is what
    # caused the Step3.7 iter-11 deadlock.
    assert "no untried approach is visible" not in reviewer
    # Mode-aware APPROVE gate reads the injected Replan mode line.
    assert "Replan mode: enabled" in reviewer
    assert "Replan mode: disabled" in reviewer
    # The mandatory APPROVE summary line that QA scopes on.
    assert "Stage closed: Stage <N>" in reviewer
    # Failed-closure closure-mode suffix: lets QA distinguish "every
    # criterion passes" from "Goal unreachable, replan required" so
    # QA's REJECT framing is accurate instead of falsely accusing
    # the Reviewer of a contract violation. This is the Step3.7
    # iter-19/iter-21 protocol-conflict fix.
    assert ("Stage closed: Stage <N> (via Goal <X.Y> [Failed]; replan required)") in reviewer
    assert "**mandatory** whenever closure is via\n`[Failed]`" in reviewer
    assert "ordinary `[Done]`-closure write the bare label" in reviewer
    # Stage closure is gated on the Reviewer's endorsement of the
    # Coder's terminal conclusion, not on acceptance items being
    # checked. Specifically: a properly Failed last Goal must advance
    # the state machine just like a Done last Goal, and the old
    # "acceptance items gate Stage closure" framing must NOT reappear.
    assert "terminal endorsement equivalent to `[Done]`" in reviewer
    assert "no longer a Stage-closure blocker" in reviewer
    assert "owns the unmet-item handling" in reviewer
    assert "gate the Stage's closure" not in reviewer


def test_qa_extra_carries_stage_scoping_rule():
    """QA's Stage scoping rule must spell out its two key behaviors.

    Those are the narrow progress.yaml carve-out and the unified ``last
    Stage → verify whole file`` behavior.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    qa = mb_prompts.build_modeling_bringup_prompts(replan_on_qa=True).qa

    assert "## Stage/Goal-aware verification scope" in qa
    # The narrow exception to "do not read intermediate artifacts".
    assert "Narrow exception" in qa
    assert "progress.yaml" in qa
    # The exact label format QA parses.
    assert "Stage closed: Stage <N>" in qa
    # Unified behavior: last Stage → verify entire file; intermediate
    # → verify only the matching subsection.
    assert "verify only that subsection" in qa
    assert "verify the entire" in qa
    # Malformed/missing label → immediate REJECT.
    assert "REJECT this turn immediately" in qa
    # Failed-closure closure-mode suffix recognition: when present,
    # QA must reframe its REJECT as "Goal unreachable; routing to
    # PlanDrafter" instead of falsely accusing the Reviewer of a
    # contract violation. This is the Step3.7 iter-19/iter-21
    # Reviewer/QA protocol-conflict fix.
    assert "(via Goal <X.Y> [Failed]; replan required)" in qa
    assert "Failed-closure suffix" in qa
    assert "Failed (replan required)" in qa
    # Done-closure regression label — yapf wraps after "Failed" so
    # the literal substring carries the indent that follows.
    assert "Failed\n       (regression)" in qa
    assert "routing to PlanDrafter for gap-fix\n       Stage" in qa
    assert "Do NOT call this a Reviewer contract violation" in qa
    # Done-closure framing for genuine regressions stays sharp.
    assert (
        "criterion regressed: Reviewer's claimed Done-closure\n"
        "       does not hold under independent rerun"
    ) in qa


def test_reference_ladder_guidance_in_planner_only():
    """Accuracy-debug principle 1: plan the three-tier reference ladder.

    The PlanDrafter must lay out native ``from_pretrained`` baseline →
    aligned pure-PyTorch module reference → TensorRT-LLM parity so a
    later parity failure has an already-trusted target. This is
    planning guidance — it scopes the Goals — so it belongs to the
    PlanDrafter, not the Coder.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    pd = bundle.plan_drafter
    assert (
        "## Accuracy debugging — build a verified reference ladder before TensorRT-LLM parity"
    ) in pd, "PlanDrafter prompt missing the reference-ladder heading"
    # The native baseline is the ground-truth tier of the ladder.
    assert "AutoModelForCausalLM.from_pretrained" in pd, (
        "PlanDrafter prompt does not name the native from_pretrained baseline"
    )
    assert "3-5 fixed smoke-test prompts" in pd, (
        "PlanDrafter prompt does not pin 3-5 fixed smoke-test prompts"
    )
    # The middle tier: a pure-PyTorch module reference aligned to the
    # native baseline before any TRT-LLM parity is attempted.
    assert "Pure-PyTorch module reference" in pd, (
        "PlanDrafter prompt missing the pure-PyTorch module reference tier"
    )

    # Scoping: this is PlanDrafter-only guidance (principle 1). The
    # native-baseline marker must not leak into the Coder prompt.
    assert "AutoModelForCausalLM.from_pretrained" not in bundle.coder, (
        "reference-ladder guidance leaked into the Coder prompt; it is "
        "PlanDrafter-only planning guidance"
    )


def test_parity_vs_dataset_guidance_in_planner_only():
    """Accuracy-debug principle 2: read parity and dataset accuracy together.

    The PlanDrafter must tell the plan that a loose-but-stable parity
    number caused by benign numerical jitter is not by itself a defect
    when dataset accuracy is reasonable, so dataset testing need not
    wait for a perfectly-tight parity result.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    pd = bundle.plan_drafter
    assert ("## Accuracy debugging — read parity and dataset accuracy together") in pd, (
        "PlanDrafter prompt missing the parity-vs-dataset heading"
    )
    assert "benign numerical jitter" in pd, (
        "PlanDrafter prompt does not name benign numerical jitter as a "
        "reason a parity number can look insufficient"
    )
    assert "parity **and** dataset accuracy" in pd, (
        "PlanDrafter prompt does not require judging parity and dataset accuracy together"
    )

    # Regression guard: the contradictory "run configured accuracy gates
    # only after ... parity tests pass" ordering rule was removed from
    # the accuracy-gate framework because it is wrong — dataset-accuracy
    # testing must not be gated on parity passing first.
    assert "accuracy gates only after" not in pd, (
        "the deleted 'accuracy gates only after ... parity tests pass' "
        "ordering rule is back in the PlanDrafter prompt; it contradicts "
        "reading parity and dataset accuracy together"
    )

    # Scoping: principle 2 is PlanDrafter-only.
    assert "read parity and dataset accuracy" not in bundle.coder, (
        "parity-vs-dataset guidance leaked into the Coder prompt; it is "
        "PlanDrafter-only planning guidance"
    )


def test_gsm8k_reference_config_in_planner_and_coder():
    """Accuracy-debug principle 3: planner and coder share the gsm8k policy.

    The gsm8k reference-aligned config policy (fixed 100-sample subset,
    matched PyTorch-reference / TensorRT-LLM config, long-enough
    max_seq_length, matched tokenizer / apply_chat_template,
    thinking-mode trial, batch_size 8/16) is a runtime convention both
    the planner and the coder act on, so it must appear in BOTH the
    PlanDrafter and the Coder prompts.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    for role in ("plan_drafter", "coder"):
        prompt = getattr(bundle, role)
        assert "## gsm8k accuracy run — reference-aligned config" in prompt, (
            f"{role} prompt missing the gsm8k reference-config heading"
        )
        # Fixed 100-sample subset for comparable scores.
        assert "100-sample subset" in prompt, (
            f"{role} prompt does not pin a fixed 100-sample subset"
        )
        # max_seq_length long enough (2048) so answers are not truncated.
        assert "max_seq_length" in prompt and "2048" in prompt, (
            f"{role} prompt does not call out a long-enough max_seq_length (2048)"
        )
        # Matched tokenizer / chat-template rendering across both paths.
        assert "apply_chat_template" in prompt, (
            f"{role} prompt does not mention apply_chat_template rendering"
        )
        # Thinking-mode trial.
        assert "Thinking mode" in prompt, f"{role} prompt does not suggest trying thinking mode"
        # Throughput-friendly batch size.
        assert "`batch_size` 8 or 16" in prompt, (
            f"{role} prompt does not suggest batch_size 8 or 16"
        )


def test_accuracy_gap_teacher_forcing_in_planner_and_coder():
    """Accuracy-debug principle 5: localize gsm8k gaps with teacher forcing.

    When the TensorRT-LLM gsm8k score trails the PyTorch reference, the
    gap is localized by (a) picking the discriminating cases —
    reference-correct / TRT-LLM-wrong — and (b) comparing with teacher
    forcing (feed the reference's token to both paths so a single
    divergence does not fork the rest of the sequence out of
    comparability). Both the planner (who must budget a Goal for it)
    and the coder (who runs it) act on this, so it must appear in BOTH
    prompts.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    for role in ("plan_drafter", "coder"):
        prompt = getattr(bundle, role)
        assert ("## Debugging a TensorRT-LLM vs PyTorch-reference accuracy gap") in prompt, (
            f"{role} prompt missing the accuracy-gap debugging heading"
        )
        # Case selection: the reference-correct / TRT-LLM-wrong samples
        # are the ones that drive the gap.
        assert "Pick the discriminating cases" in prompt, (
            f"{role} prompt does not tell you to pick the discriminating cases"
        )
        assert "PyTorch reference is correct but TensorRT-LLM is wrong" in prompt, (
            f"{role} prompt does not scope analysis to reference-correct / TRT-LLM-wrong samples"
        )
        # Teacher forcing is the comparison method.
        assert "teacher forcing" in prompt, (
            f"{role} prompt does not name teacher forcing as the comparison method"
        )
        # The mechanism: feed the reference token to both paths so a
        # divergence does not fork the sequence.
        assert "feed the reference's chosen token to both" in prompt, (
            f"{role} prompt does not describe feeding the reference token "
            f"to both paths to keep the prefix aligned"
        )
        assert "fork at the first differing token" in prompt, (
            f"{role} prompt does not explain why free-running comparison "
            f"forks and loses comparability"
        )


def test_multimodal_text_first_in_planner_only():
    """Accuracy-debug principle 4: bring up the text path first.

    For a multimodal model the PlanDrafter must sequence the
    TensorRT-LLM bring-up so the text (language-model) path is brought
    up and validated through the gsm8k accuracy gate *before* the
    multimodal path is debugged.
    """
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    pd = bundle.plan_drafter
    assert "## Multimodal models — bring up the text path first" in pd, (
        "PlanDrafter prompt missing the multimodal text-first heading"
    )
    assert "text (language-model) path" in pd, (
        "PlanDrafter prompt does not name the text path as the first multimodal phase"
    )
    assert "text path has passed its accuracy gate" in pd, (
        "PlanDrafter prompt does not gate the multimodal phase on the "
        "text path passing its accuracy gate"
    )

    # Scoping: principle 4 is PlanDrafter-only.
    assert "bring up the text path first" not in bundle.coder, (
        "multimodal text-first guidance leaked into the Coder prompt; "
        "it is PlanDrafter-only planning guidance"
    )
