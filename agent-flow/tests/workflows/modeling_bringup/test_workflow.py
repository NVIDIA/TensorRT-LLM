from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.agent_team import \
    workflow as _agent_team_workflow_module
from agent_flow.workflows.modeling_bringup import \
    cli as _modeling_bringup_cli_module
from agent_flow.workflows.modeling_bringup import task_schema as _task_schema


def _load_agent_team_workflow_module():
    return _agent_team_workflow_module


def _load_modeling_bringup_cli_module():
    return _modeling_bringup_cli_module


def test_default_prompts_match_individual_constants():
    """``DEFAULT_PROMPTS`` must equal the per-agent module constants so the
    refactor keeps existing behavior."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert prompts.DEFAULT_PROMPTS.plan_drafter == prompts.PLAN_DRAFTER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.plan_reviewer == prompts.PLAN_REVIEWER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.coder == prompts.CODER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.reviewer == prompts.REVIEWER_SYSTEM_PROMPT
    assert prompts.DEFAULT_PROMPTS.qa == prompts.QA_SYSTEM_PROMPT


def test_with_extensions_empty_returns_unchanged_bundle():
    """Empty / whitespace-only extensions must leave each base prompt
    byte-for-byte unchanged."""
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
    """Non-empty extensions are appended after a single blank line; the base
    is rstripped first so we never get more than one blank line."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    extended = prompts.DEFAULT_PROMPTS.with_extensions(
        coder="## Modeling-Bringup\nUse mixed precision.\n", )
    assert extended.coder.startswith(prompts.CODER_SYSTEM_PROMPT.rstrip())
    assert extended.coder.endswith(
        "\n\n## Modeling-Bringup\nUse mixed precision.\n")
    # No triple newlines anywhere in the joined prompt.
    assert "\n\n\n" not in extended.coder
    # Other agents are untouched when only ``coder`` is extended.
    assert extended.qa == prompts.DEFAULT_PROMPTS.qa


def test_prompt_bundle_is_frozen():
    """Bundle must be immutable so callers can't accidentally mutate the
    shared default."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    with pytest.raises(Exception):  # FrozenInstanceError subclass of Exception
        prompts.DEFAULT_PROMPTS.coder = "mutated"  # type: ignore[misc]


def test_workflow_uses_default_prompts_when_arg_omitted(tmp_path):
    """Backwards compatibility: omitting ``prompts`` must keep every agent
    on the original ``agent_team`` system prompts."""
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
    """A custom bundle must reach every agent's ``system_prompt``, including
    the agents recreated by ``_reset_coder`` / ``_reset_reviewer``."""
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
    """With every ``*_extra.SYSTEM_PROMPT_EXTENSION`` cleared, the modeling
    bundle must be identical to ``DEFAULT_PROMPTS`` so the wrapper itself
    adds nothing beyond what each ``*_extra.py`` contributes."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    for name in ("plan_drafter_extra", "plan_reviewer_extra", "coder_extra",
                 "reviewer_extra", "qa_extra"):
        module = importlib.import_module(
            f"agent_flow.workflows.modeling_bringup.prompts.{name}")
        monkeypatch.setattr(module, "SYSTEM_PROMPT_EXTENSION", "")
    importlib.reload(mb_prompts)
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert mb_prompts.MODELING_BRINGUP_PROMPTS == prompts.DEFAULT_PROMPTS


def test_modeling_bringup_prompts_extend_default_for_every_agent():
    """The populated extras must add modeling-bringup guidance to every
    agent's prompt, while keeping the base prompt as a prefix."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS
    base = prompts.DEFAULT_PROMPTS
    for name in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        extended = getattr(bundle, name)
        baseline = getattr(base, name)
        assert extended != baseline, f"{name} extension is empty"
        assert extended.startswith(baseline.rstrip()), (
            f"{name} prompt does not start with the base prompt")
        assert "\n\n\n" not in extended, (
            f"{name} prompt has a triple newline from joining")


def test_modeling_bringup_coder_prompt_includes_accuracy_gate_framework():
    """Coder must see the accuracy-gate rules so it can choose the right
    next step before Reviewer or QA rejects a low-score run."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    coder_prompt = mb_prompts.MODELING_BRINGUP_PROMPTS.coder

    assert "## Accuracy-gate framework" in coder_prompt
    assert "Run baseline first" in coder_prompt
    assert ("If the baseline score is below the configured\n"
            "  threshold, keep fixing baseline before running the enabled "
            "configuration.") in coder_prompt


def test_modeling_bringup_coder_prompt_includes_validation_policy_context():
    """Coder must see the strict policy blocks that define what each
    evidence label has to prove, not only the evidence-label glossary."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
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
            f"Coder prompt missing validation-policy context {snippet!r}")


def test_modeling_bringup_prompts_include_accuracy_debugging_methodology():
    """Every modeling-bringup role must share the small failing-prompt loop
    for debugging unexpectedly low accuracy."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
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
                f"{role} prompt missing accuracy debugging snippet "
                f"{snippet!r}")


def test_modeling_bringup_prompts_define_evidence_labels_as_prompt_contract():
    """The modeling-bringup evidence names are prompt-level labels in
    agent-flow, not built-in tests. Every role needs that framing so terms like
    ``source_logit_replay`` do not look like missing local functions."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    required_snippets = (
        "## Modeling-bringup evidence labels",
        "they are **not** built-in functions",
        "automatic controller checks",
        "`reference_tier` and `validation_tier` are the two schema metadata "
        "fields",
        "`reference_tier` must be one of `static`, `minimal_golden`, "
        "`reduced_source`,\nor `real_source`",
        "`validation_tier` must be one of `static`, `unit`,\n`integration`, "
        "or `real_runtime`",
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
                f"{role} prompt missing evidence-label clarification "
                f"{snippet!r}")

    assert "test types named in the" not in bundle.plan_drafter
    assert "test types named in the" not in bundle.plan_reviewer
    assert "schema enums, or automatic controller checks" not in bundle.coder
    assert "evidence labels defined" in bundle.plan_drafter
    assert "evidence labels defined" in bundle.plan_reviewer


def test_modeling_bringup_plan_drafter_uses_underscored_tier_fields():
    """PlanDrafter must use the same metadata field spelling that
    PlanReviewer checks in acceptance criteria."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
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
    """Regression: the wrapper must forward ``argv`` to the generic
    agent-team ``main`` unchanged — only injecting a task-scoped prompt
    bundle. A prior bug parsed each flag manually and dropped
    ``--feedback`` from the rebuilt constructor call, silently discarding
    user-supplied guidance. Delegating with raw ``argv`` keeps every flag
    (current and future) forwarded by construction.

    Inspect the wrapper's source AST: the only call must be
    ``_team_main(argv, prompts=prompts)``."""
    import ast
    import inspect

    source = inspect.getsource(_modeling_bringup_cli_module)
    tree = ast.parse(source)

    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "_team_main"
    ]
    assert calls, "wrapper must call _team_main"
    call = calls[0]

    # Positional argv → not unpacked, not reconstructed.
    assert len(call.args) == 1, (
        "wrapper must forward exactly one positional argument (argv)")
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "argv", (
        "wrapper must pass the raw `argv` parameter through, not a "
        "manually reconstructed list — that's how --feedback got dropped "
        "in the previous bug class")

    # Prompts injection is the only kwarg.
    kw_names = sorted(kw.arg for kw in call.keywords)
    assert kw_names == [
        "prompts"
    ], (f"wrapper must only inject prompts=..., got keywords {kw_names}")
    prompts_kw = call.keywords[0]
    assert (isinstance(prompts_kw.value, ast.Name)
            and prompts_kw.value.id == "prompts"), (
                "prompts kwarg must pass the task-scoped prompt bundle, "
                "not a reconstructed argv list or a static Slurm bundle")


def test_persistent_deviation_handling_in_modeling_bringup_reviewer():
    """Regression guard: the modeling-bringup Reviewer prompt must
    include the persistent-deviation handling block. The workflow has
    no re-plan stage, so a documented deviation that satisfies the
    acceptance criteria is the artifact's permanent shape and should
    be APPROVEd, not REJECTed iteration after iteration. In the
    gemma-4-exp7 worklog the same plan deviation was re-cited across
    14 iterations without ever being explicitly accepted."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    assert "Persistent-deviation handling" in bundle.reviewer, (
        "Reviewer prompt must include persistent-deviation handling "
        "guidance")
    assert "No re-plan stage" in bundle.reviewer, (
        "Reviewer prompt must spell out that the workflow has no "
        "re-plan stage so deviations satisfying criteria should be "
        "APPROVEd")


def test_no_replan_stage_framing_in_generic_reviewer():
    """The generic agent_team Reviewer prompt must reflect the
    workflow's actual structure: no re-plan stage exists, so the
    Reviewer's anchor is acceptance-criteria.md and documented
    deviations satisfying criteria must be APPROVEd."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert "No re-plan stage" in prompts.REVIEWER_SYSTEM_PROMPT
    assert "Persistent-deviation handling" in prompts.REVIEWER_SYSTEM_PROMPT


def test_coder_prompt_references_no_replan_stage():
    """The Coder prompt should tell the Coder not to re-paste accepted
    deviations every iteration — a deviation is the artifact's
    permanent shape, not a replan request."""
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    assert "no re-plan stage" in prompts.CODER_SYSTEM_PROMPT
    assert "iteration noise" in prompts.CODER_SYSTEM_PROMPT


def test_status_done_todo_rubric_in_coder_and_reviewer_prompts():
    """status.md's ``Done / TODO`` section is the rolling progress
    summary the next agent and the human read at a glance. Both the
    Coder (who produces the first draft each iteration) and the
    Reviewer (who adjusts it post-review) must carry the rubric in
    their populated modeling-bringup prompts.

    QA is excluded by design — its verdict must be grounded in
    ``task.yaml`` and ``acceptance-criteria.md`` alone, so the rubric
    must NOT leak into the QA prompt."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    for role in ("coder", "reviewer"):
        prompt = getattr(bundle, role)
        # Section heading the rubric mandates.
        assert "## status.md `Done / TODO` section" in prompt, (
            f"{role} prompt missing the Done/TODO rubric heading")
        # The two required subsection headings inside status.md.
        assert "### Done" in prompt, (
            f"{role} prompt missing the Done subsection heading")
        assert "### TODO" in prompt, (
            f"{role} prompt missing the TODO subsection heading")
        # The "every acceptance-criteria item appears in exactly one"
        # rule — the load-bearing invariant that keeps the section
        # honest across iterations.
        assert "exactly one of `Done` or `TODO`" in prompt, (
            f"{role} prompt missing the exactly-one-of-Done-or-TODO "
            f"invariant")
        # Executed-evidence requirement for the Done list — otherwise
        # the section becomes wishful thinking.
        assert "executed evidence" in prompt, (
            f"{role} prompt missing the executed-evidence rule for "
            f"the Done list")
        # TODO requires a named blocker + planned next step.
        assert "TBD" in prompt, (
            f"{role} prompt does not forbid `TBD` placeholders in "
            f"the TODO list")
        # The Coder-drafts / Reviewer-adjusts handoff is what makes
        # this work across the two agents — verify both names appear
        # in the rubric narrative.
        assert "Coder" in prompt and "Reviewer" in prompt, (
            f"{role} prompt does not name both Coder and Reviewer in "
            f"the rubric")

    # QA must not carry the rubric: QA is stateless and is forbidden
    # from reading status.md.
    assert "## status.md `Done / TODO` section" not in bundle.qa, (
        "QA prompt unexpectedly carries the status.md rubric; "
        "status.md is not in QA's source-of-truth set")


def test_qa_prompt_has_final_report_contract():
    """The modeling-bringup QA prompt must surface the final-report
    contract: the artifact path, both terminal statuses, the section
    skeleton, the required dimensions (accuracy / performance vs HF /
    feature coverage / parallelism), the backend-selection and
    trtllm-repo-changes implementation sections, and the
    ``Not measured`` fallback rule. Each piece is a load-bearing part
    of the closure artifact the user reads after a run."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
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
        assert strategy in qa_prompt, (
            f"parallelism row {strategy!r} missing from QA prompt")

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
            f"QA prompt does not forbid {forbidden!r} as a "
            f"non-measurement placeholder")

    # Ordering: the report write must happen before
    # ``append_qa_progress`` so an interrupt between the two does not
    # lose the closure artifact.
    write_idx = qa_prompt.find("Write the file **before** you call "
                               "`append_qa_progress`")
    assert write_idx != -1, (
        "QA prompt must explicitly order the report write before "
        "append_qa_progress")


def test_modeling_bringup_default_prompts_omit_slurm_guidance():
    """The module-level modeling-bringup bundle is for local tasks.
    Slurm/container guidance is injected only for validated task specs
    that contain ``slurm-environment``."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    for name in ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"):
        prompt = getattr(bundle, name)
        assert "Slurm container bootstrap" not in prompt
        assert "slurm-environment" not in prompt

    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "test_command.md" not in prompt


def test_modeling_bringup_prompts_require_srun_for_slurm():
    """Coder, Reviewer, and QA extensions for modeling-bringup must all
    mandate that cached test_command.md commands be runnable from the
    Slurm login node — i.e. they spell out the ``srun`` / login-node
    requirement, not just "Slurm"."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(
        include_slurm_environment=True)
    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "srun" in prompt
        assert "login node" in prompt


def test_modeling_bringup_test_command_md_is_slurm_only():
    """Coder, Reviewer, and QA extensions must gate ``test_command.md``
    on a Slurm environment: on local (non-Slurm) hosts the file is
    not created or maintained at all."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(
        include_slurm_environment=True)
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
    """The modeling-bringup extensions should encourage specialist help
    for `trtllm-*` command drafting without naming a plugin users may not
    have installed."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(
        include_slurm_environment=True)
    for prompt in (bundle.coder, bundle.reviewer, bundle.qa):
        assert "test-command specialist" in prompt
        assert "trtllm-agent-toolkit" not in prompt
        assert "test_command.md" in prompt


def test_modeling_bringup_prompts_require_minimal_test_command_cache():
    """The modeling-bringup extensions must mandate a minimal
    ``test_command.md``: only currently-passing commands, no inline
    supersede prose / archive section / top-of-file narrative, and a
    structured per-entry template that ties each command back to the
    acceptance criteria it verifies."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(
        include_slurm_environment=True)
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
    """A Slurm-enabled modeling-bringup prompt bundle must carry the
    container bootstrap and use the ``slurm-environment`` YAML fields."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    bundle = mb_prompts.build_modeling_bringup_prompts(
        include_slurm_environment=True)
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
                f"{name} prompt is missing container-bootstrap line: "
                f"{needle!r}")
        assert '-a "90;100-real"' not in prompt
        assert "--qos" not in prompt


def test_modeling_bringup_workflow_uses_extended_bundle(tmp_path, monkeypatch):
    """Override one agent's extension and verify only that agent's prompt
    changes; the others keep their normally-populated extensions."""
    mb_prompts = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    baseline_bundle = mb_prompts.MODELING_BRINGUP_PROMPTS

    coder_extra = importlib.import_module(
        "agent_flow.workflows.modeling_bringup.prompts.coder_extra")
    monkeypatch.setattr(coder_extra, "SYSTEM_PROMPT_EXTENSION",
                        "## MB\nExtra coder guidance.\n")
    importlib.reload(mb_prompts)

    workflow_module = _load_agent_team_workflow_module()
    prompts = importlib.import_module("agent_flow.workflows.agent_team.prompts")
    workflow = workflow_module.AgentTeamWorkflow(
        workspace=tmp_path,
        prompts=mb_prompts.MODELING_BRINGUP_PROMPTS,
    )
    try:
        assert workflow.coder.config.system_prompt.endswith(
            "\n\n## MB\nExtra coder guidance.\n")
        assert workflow.coder.config.system_prompt.startswith(
            prompts.CODER_SYSTEM_PROMPT.rstrip())
        # Agents whose extras were not monkeypatched keep their normally-
        # populated bundle prompts.
        assert workflow.qa.config.system_prompt == baseline_bundle.qa
        assert workflow.plan_drafter.config.system_prompt == baseline_bundle.plan_drafter
    finally:
        workflow.close()
        # Reset the override so subsequent tests see the populated bundle.
        monkeypatch.undo()
        importlib.reload(mb_prompts)


def _valid_task_payload(tmp_path) -> dict:
    """Return a fully-populated payload whose three required paths exist
    on disk under ``tmp_path``."""
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


def test_modeling_bringup_cli_omits_slurm_prompts_without_slurm_environment(
        tmp_path, monkeypatch):
    """The CLI should pass a local-host prompt bundle for task specs that
    do not contain ``slurm-environment``."""
    payload = _valid_task_payload(tmp_path)
    task_path = _write_task_yaml(tmp_path, payload)
    argv = ["--task", str(task_path), "--workspace", str(tmp_path / "work")]
    captured = {}

    def fake_team_main(forwarded_argv, *, prompts):
        captured["argv"] = forwarded_argv
        captured["prompts"] = prompts

    monkeypatch.setattr(_modeling_bringup_cli_module, "_team_main",
                        fake_team_main)

    _modeling_bringup_cli_module.main(argv)

    assert captured["argv"] == argv
    assert "Slurm container bootstrap" not in captured["prompts"].coder
    assert "test_command.md" not in captured["prompts"].coder


def test_modeling_bringup_cli_includes_slurm_prompts_with_slurm_environment(
        tmp_path, monkeypatch):
    """The ``slurm-environment`` field is the switch for Slurm prompt
    injection."""
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

    monkeypatch.setattr(_modeling_bringup_cli_module, "_team_main",
                        fake_team_main)

    _modeling_bringup_cli_module.main(argv)

    assert captured["argv"] == argv
    assert "Slurm container bootstrap" in captured["prompts"].coder
    assert "test_command.md" in captured["prompts"].coder
    assert "slurm_partition" in captured["prompts"].plan_drafter


def test_task_schema_accepts_valid_payload(tmp_path):
    """A complete, well-typed YAML loads cleanly and round-trips its
    fields (including extra keys) so downstream agents see the full
    spec."""
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
    """A Slurm task must name the partition and container image without
    forcing the image path to exist on the local host."""
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
        ({
            "slurm_partition": "gb200"
        }, "docker_image"),
        ({
            "slurm_partition": " ",
            "docker_image": "/containers/trtllm.sqsh"
        }, "slurm_partition"),
        ({
            "slurm_partition": "gb200",
            "docker_image": 42
        }, "docker_image"),
    ],
)
def test_task_schema_rejects_invalid_slurm_environment(tmp_path,
                                                       slurm_environment,
                                                       expected):
    """If present, ``slurm-environment`` must contain the two fields the
    prompt builder relies on."""
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
    """Each required path field, when omitted, must be named in the
    error message."""
    payload = _valid_task_payload(tmp_path)
    del payload[missing_field]
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert missing_field in str(exc.value)


def test_task_schema_batches_all_missing_required_fields(tmp_path):
    """A single load reports every missing required field at once so the
    user does not have to fix them one at a time."""
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
    """A required path that does not exist on disk is reported with the
    offending value so the user can see what they need to fix."""
    payload = _valid_task_payload(tmp_path)
    payload["checkpoint_path"] = "/path/that/should/not/exist/anywhere/xyz"
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert "checkpoint_path" in str(exc.value)
    assert "/path/that/should/not/exist/anywhere/xyz" in str(exc.value)


def test_task_schema_rejects_empty_string_required_field(tmp_path):
    """Whitespace-only / empty strings count as missing — the path probe
    would silently match the current directory, so reject up front."""
    payload = _valid_task_payload(tmp_path)
    payload["trtllm_repo_path"] = "   "
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert "trtllm_repo_path" in str(exc.value)


@pytest.mark.parametrize("field", ["completion_criteria", "implements_tips"])
def test_task_schema_defaults_optional_lists_to_empty(tmp_path, field):
    """Missing optional list fields default to ``[]`` so callers can rely
    on the key being present."""
    payload = _valid_task_payload(tmp_path)
    del payload[field]
    path = _write_task_yaml(tmp_path, payload)

    data = _task_schema.load_and_validate_task_yaml(path)
    assert data[field] == []


@pytest.mark.parametrize("field", ["completion_criteria", "implements_tips"])
def test_task_schema_rejects_non_list_optional_field(tmp_path, field):
    """If the user supplies an optional field, it must be a list of
    strings — a scalar or dict is a type error."""
    payload = _valid_task_payload(tmp_path)
    payload[field] = "not a list"
    path = _write_task_yaml(tmp_path, payload)

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert field in str(exc.value)


@pytest.mark.parametrize("field", ["completion_criteria", "implements_tips"])
def test_task_schema_rejects_non_string_list_items(tmp_path, field):
    """Items inside the optional lists must be strings, not numbers or
    dicts."""
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
    """A file that is not valid YAML is rejected with the file path so
    the user can find it."""
    path = tmp_path / "broken.yaml"
    path.write_text("reference_code_path: [unclosed\n", encoding="utf-8")

    with pytest.raises(_task_schema.TaskSchemaError) as exc:
        _task_schema.load_and_validate_task_yaml(path)
    assert "not valid YAML" in str(exc.value)
    assert str(path) in str(exc.value)


def test_task_schema_rejects_non_mapping_top_level(tmp_path):
    """A YAML scalar or list at the top level is not a usable task
    spec — must be a mapping."""
    path = tmp_path / "scalar.yaml"
    path.write_text("just-a-string\n", encoding="utf-8")

    with pytest.raises(_task_schema.TaskSchemaError, match="mapping"):
        _task_schema.load_and_validate_task_yaml(path)


def test_task_schema_rejects_missing_file(tmp_path):
    """A non-existent path is rejected with the path quoted in the
    error so the user can see the typo."""
    missing = tmp_path / "absent.yaml"

    with pytest.raises(_task_schema.TaskSchemaError, match="not found"):
        _task_schema.load_and_validate_task_yaml(missing)
