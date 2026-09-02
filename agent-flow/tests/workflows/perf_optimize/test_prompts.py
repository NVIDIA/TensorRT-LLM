"""Tests that the perf-optimize role prompts carry their contracts.

The measuring roles must drive ``benchmark_serving.py`` / ``nsys`` from
the same canonical templates as perf-analyze; the mutating roles must
carry the git discipline and the acceptance gate; and every role that
touches ``roadmap.yaml`` must carry the roadmap contract (including the
orchestrator-owns-lifecycle rule).
"""

from __future__ import annotations

import re

from agent_flow.workflows.perf_analyze.prompts._common import (
    SOL_METHODOLOGY_FALLBACK as _ANALYZE_METHODOLOGY_FALLBACK,
)
from agent_flow.workflows.perf_optimize.prompts import (
    ANALYZER_SYSTEM_PROMPT,
    BENCHMARKER_SYSTEM_PROMPT,
    DEFAULT_PROMPTS,
    EVALUATOR_SYSTEM_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT,
    PROJECTOR_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    build_perf_optimize_prompts,
    build_projector_prompt,
)
from agent_flow.workflows.perf_optimize.prompts._common import (
    CASEBOOK_APPLY,
    EXPECTATION_GATE,
    GIT_DISCIPLINE,
    KERNEL_COVERAGE_REPORTER_GUIDANCE,
    KERNEL_REUSE,
    MEASUREMENT_PROTOCOL,
    OPTIMIZE_HTML_COMPANION,
    PROFILE_FINDINGS_CONTRACT,
    ROADMAP_SPEC,
    SOL_ANALYZER_CONTEXT,
    SOL_METHODOLOGY_FALLBACK,
    SOL_OPTIMIZE_REPORTER_GUIDANCE,
    SOL_OPTIMIZER_CONTEXT,
    TUNING_CONFIG_NOTE,
    approach_restriction_note,
    kernel_coverage_analyzer_note,
)

_ALL_PROMPTS = {
    "benchmarker": BENCHMARKER_SYSTEM_PROMPT,
    "projector": PROJECTOR_SYSTEM_PROMPT,
    "analyzer": ANALYZER_SYSTEM_PROMPT,
    "optimizer": OPTIMIZER_SYSTEM_PROMPT,
    "evaluator": EVALUATOR_SYSTEM_PROMPT,
    "qa": QA_SYSTEM_PROMPT,
    "reporter": REPORTER_SYSTEM_PROMPT,
}

# Roles that run the canonical benchmark_serving.py command themselves.
_MEASURING = ("benchmarker", "analyzer", "evaluator", "qa")


def _norm(text: str) -> str:
    """Collapse whitespace so substring assertions survive line-wrapping."""
    return re.sub(r"\s+", " ", text)


# Canonical ``benchmark_serving.py`` flags every measuring role must carry.
_BENCHMARK_CANONICAL_FLAGS = (
    "--tokenizer",
    "--trust-remote-code",
    "--random-ids",
    "--tokenize-on-client",
    "--ignore-eos",
    "--no-test-input",
    "--percentile-metrics",
)

# Canonical ``nsys profile`` flags the analyzer must carry.
_NSYS_CANONICAL_FLAGS = (
    "-t 'cuda,nvtx,python-gil'",
    "-c cudaProfilerApi",
    "--cuda-graph-trace node",
    "TLLM_NVTX_DEBUG=1",
    "--trace-fork-before-exec=true",
)


def test_measuring_roles_carry_canonical_benchmark_flags():
    for role in _MEASURING:
        for flag in _BENCHMARK_CANONICAL_FLAGS:
            assert flag in _ALL_PROMPTS[role], (role, flag)
        assert "do not improvise" in _ALL_PROMPTS[role], role


def test_analyzer_carries_canonical_nsys_flags():
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in ANALYZER_SYSTEM_PROMPT, flag
    # The safety flag that keeps nsys from SIGTERMing the engine.
    assert "--capture-range-end=stop" in ANALYZER_SYSTEM_PROMPT
    # Knob verification (verify before asserting) came along with the
    # profiling reference blocks.
    assert "TLLM_PROFILE_START_STOP" in ANALYZER_SYSTEM_PROMPT
    assert "Verify the profiling knobs first" in ANALYZER_SYSTEM_PROMPT


def test_analyzer_carries_the_ncu_deep_dive():
    # The shared Run C: a bounded per-kernel ncu capture of the top nsys
    # kernels, interpreted with the perf-nsight-compute-analysis skill.
    for flag in (
        "--target-processes all",
        "--profile-from-start off",
        "--section SpeedOfLight",
        "--launch-count",
    ):
        assert flag in ANALYZER_SYSTEM_PROMPT, flag
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "perf-nsight-compute-analysis" in prompt
    assert "trtllm-agent-toolkit:perf-nsight-compute-analysis" in prompt
    # The findings carry the dedicated section, degrading honestly.
    assert "## ncu kernel analysis" in prompt
    assert "ncu unavailable" in prompt


def test_analyzer_grounds_roadmap_items_across_the_analyses():
    # Expected gains draw on all three analyses — the nsys share, the
    # targeted kernel's ncu bound class, and the SOL correlation's gap
    # rows (when the sol block ran) — not the timeline alone.
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "Draw the evidence from all three analyses" in prompt
    assert "bound class" in prompt
    # The three-pillar synthesis rule arrives with the shared contract.
    assert "three evidence pillars" in prompt


def test_no_prompt_references_removed_builtin_tools():
    # The agents run on the CLI's ``default`` toolset, which no longer
    # includes ``Grep``/``Glob``; instructing them makes the agent call a
    # nonexistent tool. Every role prompt must avoid the tool names.
    for name, prompt in _ALL_PROMPTS.items():
        for tool in ("Grep", "Glob"):
            assert not re.search(rf"\b{tool}\b", prompt), (name, tool)


# ------------------------------------------------------------------- casebook


def test_benchmarker_and_analyzer_load_casebook_read_only():
    for role in ("benchmarker", "analyzer"):
        prompt = _ALL_PROMPTS[role]
        assert "perf-optimization-casebook" in prompt, role
        assert "`Skill` tool" in prompt, role
        assert "Ground your analysis in the optimization casebook" in prompt, role


def test_optimizer_gets_the_actionable_casebook_variant():
    assert "Apply from the optimization casebook" in OPTIMIZER_SYSTEM_PROMPT
    assert "perf-optimization-casebook" in OPTIMIZER_SYSTEM_PROMPT
    # The read-only stance would contradict the optimizer's job.
    assert "Ground your analysis in the optimization casebook" not in OPTIMIZER_SYSTEM_PROMPT
    block = _norm(CASEBOOK_APPLY)
    assert "how-to-apply" in block
    assert "rollback" in block
    # Still no hard dependency on the toolkit.
    assert "not available in this environment" in block


# -------------------------------------------------------------- git discipline


def test_mutating_roles_carry_git_discipline():
    for role in ("optimizer", "evaluator"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "dedicated optimization branch" in prompt, role
        assert "import tensorrt_llm" in prompt, role
        assert "`git push`" in prompt, role
    block = _norm(GIT_DISCIPLINE)
    # The orchestrator owns commits and reverts; agents never mutate git.
    assert "Never run `git commit`" in block
    assert "orchestrator owns all git state" in block
    assert "only the current roadmap item's changes" in block
    assert "active runtime checkout from the turn instructions" in block
    assert "prepend that exact checkout to `PYTHONPATH`" in block
    assert "may differ from the active runtime checkout" in block
    assert "an editable install" not in block


def test_code_edits_never_reference_run_internals():
    # A committed comment like "See the opt-008 bench in the perf
    # workspace" is meaningless to TRT-LLM readers: the discipline block
    # bans run-internal references in source, and the evaluator's
    # code-quality axis gates on it.
    block = _norm(GIT_DISCIPLINE)
    assert "stand on its own" in block
    assert "roadmap item ids (`opt-008`)" in block
    assert "the provenance story belongs in `optimization_summary.md`" in block
    gate = _norm(EXPECTATION_GATE)
    assert "no comments or names that reference this run's internals" in gate


# ---------------------------------------------------------------- kernel reuse


def test_kernel_work_roles_prefer_existing_kernels():
    # A correct fusion realized as a hand-written kernel that flashinfer
    # (or TRT-LLM itself) already ships is still the wrong change — the
    # whole plan → apply → gate chain must carry the reuse rule.
    for role in ("analyzer", "optimizer", "evaluator"):
        prompt = _ALL_PROMPTS[role]
        assert "Prefer existing kernels over writing new ones" in prompt, role
    block = _norm(KERNEL_REUSE)
    # Search order: the checkout first, then flashinfer, then any other
    # provider already integrated; a new kernel is the last resort.
    assert "The TRT-LLM checkout itself" in block
    assert "flashinfer" in block
    assert "already integrated" in block
    # Planning and implementing both record the search that came up empty.
    assert "what you searched" in block
    # The preference is conditional: an empty search makes a new kernel the
    # encouraged realization, never a dropped item — the analyzer still
    # plans it, the optimizer falls back to writing instead of recording a
    # no-change blocker, and the evaluator judges it on the normal axes.
    assert "writing a new kernel is the encouraged" in block
    assert "fall back to writing the kernel rather than recording a no-change blocker" in block
    assert "the new kernel is a legitimate realization" in block
    # The evaluator enforces reuse on the code-quality axis, gain or not.
    assert "never passes the code-quality axis, whatever gain it measures" in block
    assert "PUSH_BACK with `reason_category: code_quality`" in block
    gate = _norm(EXPECTATION_GATE)
    assert "adds no hand-written kernel" in gate
    # The measuring-only and synthesis roles never touch kernels.
    assert "Prefer existing kernels" not in BENCHMARKER_SYSTEM_PROMPT
    assert "Prefer existing kernels" not in REPORTER_SYSTEM_PROMPT


# ---------------------------------------------------------------- roadmap spec


def test_roadmap_touching_roles_carry_the_contract():
    for role in ("analyzer", "optimizer", "evaluator", "qa", "reporter"):
        prompt = _ALL_PROMPTS[role]
        assert "The roadmap contract (`roadmap.yaml`)" in prompt, role
        assert "List order is priority order" in prompt, role
    # The benchmarker runs before the roadmap exists.
    assert "The roadmap contract" not in BENCHMARKER_SYSTEM_PROMPT


def test_roadmap_contract_pins_ownership():
    for role in ("analyzer", "optimizer"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "The **orchestrator** owns every lifecycle field" in prompt, role


# ------------------------------------------------------------ acceptance gate


def test_evaluator_carries_the_expectation_gate():
    gate = _norm(EXPECTATION_GATE)
    assert "accept_fraction × expected_gain_pct" in gate
    assert "noise_floor_pct" in gate
    # Gains accumulate: the reference is the last accepted measurement.
    assert "last ACCEPTED measurement" in gate
    assert "never the original baseline" in gate
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "accept_fraction" in prompt
    for category in ("code_quality", "functionality", "perf_shortfall"):
        assert category in prompt, category


def test_expectation_gate_is_three_way():
    gate = _norm(EXPECTATION_GATE)
    # PUSH_BACK = winnable with a concrete fix; REJECT = broken premise,
    # terminal (saving the retries' benchmarks); final attempt coerces.
    assert "PUSH_BACK" in gate
    assert "REJECT" in gate
    assert "premise is broken" in gate
    assert "no retry would help" in gate
    assert "the orchestrator treats it as REJECT" in gate
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "`APPROVE` | `REJECT` | `PUSH_BACK`" in prompt


def test_evaluator_carries_the_accept_evidence_procedure():
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "Accept-evidence capture (APPROVE only)" in prompt
    # The capture is diagnostic and never contaminates the measurement.
    assert "diagnostic, never a measurement" in prompt
    assert "fresh relaunch" in prompt
    assert "never a reason to flip the verdict" in prompt
    # Mechanism verification is the point of the capture.
    assert "claimed mechanism is visible" in prompt
    # The canonical nsys wrap ships with the prompt so the evaluator
    # never improvises profiler flags.
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in EVALUATOR_SYSTEM_PROMPT, flag
    # And the report widens beyond the target metric.
    assert "full-metric diff" in prompt
    assert "Kernel evidence" in prompt


def test_expectation_gate_carries_the_pareto_rule():
    gate = _norm(EXPECTATION_GATE)
    # Curve mode: mean over per-point gains vs the same-concurrency
    # current_best.curve entry, plus the no-regress condition.
    assert "Pareto gate" in gate
    assert "mean_gain_pct = arithmetic mean of gain_i" in gate
    assert "every gain_i >= -regression_bar" in gate
    # The bar defaults to the noise floor when no budget is declared.
    assert "else noise_floor_pct" in gate
    assert "current_best.curve" in gate
    # Degraded fallback when an earlier accept carried no curve.
    assert "carries no `curve`" in gate


def test_expectation_gate_carries_focus_scoring():
    gate = _norm(EXPECTATION_GATE)
    # The scored subset narrows the mean, never the no-regress veto.
    assert "optimize.focus_concurrencies" in gate
    assert "scored points" in gate
    assert "no-regress condition covers **every** point" in gate
    # The ledger fields follow the scored mean.
    assert "the **scored** mean" in gate
    # Roadmap-touching roles learn the ledger semantics from the contract.
    spec = _norm(ROADMAP_SPEC)
    assert "Focus scoring" in spec
    assert "mean over **only those points**" in spec
    # The measurement protocol's aggregation rule names the subset too.
    assert "optimize.focus_concurrencies" in _norm(MEASUREMENT_PROTOCOL)


def test_expectation_gate_carries_the_regression_budget():
    gate = _norm(EXPECTATION_GATE)
    # The budget is owner-declared, never assumed, and defaults strict.
    assert "optimize.max_regression_pct" in gate
    assert "regression_bar" in gate
    assert "never yours to assume" in gate
    # Used budgets must be surfaced, not buried in the mean.
    assert "name that point" in gate
    reporter = _norm(REPORTER_SYSTEM_PROMPT)
    assert "regression budget is headline material" in reporter


def test_analyzer_carries_the_dormant_capability_sweep():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # Profiling is blind to levers that never run; round 1 must sweep
    # for them in the checkpoint config, serving config, and gated code.
    assert "Dormant-capability sweep" in prompt
    assert "mtp_num_hidden_layers" in prompt
    assert "speculative_config" in prompt
    assert 'grep -n "environ"' in prompt
    assert "## Dormant capabilities" in prompt
    # Dormant levers cannot have trace evidence — dismissing them for
    # lacking it is exactly the failure the sweep exists to prevent.
    assert 'Never dismiss for "no trace evidence"' in prompt


def test_reporter_carries_the_durable_facts_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Durable facts for the next campaign" in prompt
    # The three tags, each demanding a citation.
    assert "`[dead]`" in prompt
    assert "`[alive]`" in prompt
    assert "`[env]`" in prompt
    assert "Durable facts are evidence, not opinion" in prompt


def test_measuring_roles_carry_the_measurement_protocol():
    protocol = _norm(MEASUREMENT_PROTOCOL)
    assert "positive = improvement" in protocol
    assert "output_throughput" in protocol
    # Curve mode: one run per point over one server launch, per-point
    # result dirs, and the worked Pareto example.
    assert "one run per `benchmark.concurrency` point" in protocol
    assert "concurrency_<c>" in protocol
    assert "Curve worked example" in protocol
    assert "mean = +3.24%" in protocol
    for role in ("benchmarker", "evaluator", "qa"):
        assert "Measurement protocol" in _ALL_PROMPTS[role], role
        assert "one run per `benchmark.concurrency` point" in _norm(_ALL_PROMPTS[role]), role


def test_measuring_roles_carry_the_derived_metrics_reference():
    for role in ("benchmarker", "evaluator", "qa"):
        prompt = _norm(_ALL_PROMPTS[role])
        assert "1000 / mean_tpot_ms" in prompt, role
        assert "output_throughput / num_gpus" in prompt, role
        assert "curve summary table" in prompt, role


# --------------------------------------------------------------- tuning config


def test_server_roles_carry_the_tuning_config_supersede_note():
    note = _norm(TUNING_CONFIG_NOTE)
    assert "supersedes" in note
    assert "**always** passes" in note
    assert "turn instructions name the exact **active tuning config**" in note
    assert "supersedes shorthand references" in note
    assert "<workspace>/tuning/extra_llm_api_options.yaml" not in note
    for role in ("benchmarker", "analyzer", "optimizer", "evaluator", "qa"):
        assert "The active tuning config" in _ALL_PROMPTS[role], role
    # Only the optimizer may edit the live file; the snapshot is
    # orchestrator-managed.
    assert "accepted config snapshot" in note
    assert "Never edit" in note


# ------------------------------------------------------------------------- qa


def test_qa_prompt_is_a_decisionless_final_verification():
    prompt = _norm(QA_SYSTEM_PROMPT)
    # QA runs once and verifies; the orchestrator owns the loop.
    assert "final verification" in prompt
    assert "You do not decide whether the campaign continues" in prompt
    assert "CONTINUE" not in prompt
    assert "Final-profile" not in prompt
    # Accuracy runs only when task.yaml configures it.
    assert "only if `task.yaml` has an `accuracy` block" in prompt
    assert "accuracy: not configured" in prompt
    # Fresh-eyes isolation.
    assert "Do **not** read the evaluator's" in prompt


# -------------------------------------------------------------------- reporter


def test_reporter_reports_expected_vs_measured_and_future_work():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Expected vs measured" in prompt
    assert "Failed Attempts" in prompt
    assert "Remaining Roadmap" in prompt
    # The headline is the final verification's independent number, not
    # the evaluator chain's.
    assert "The headline number is the final verification's" in prompt
    assert "Final Verification" in prompt
    assert "verification_report.md" in prompt
    assert "optimization_report.html" in REPORTER_SYSTEM_PROMPT


def test_reporter_never_launches_servers():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Do **not** launch servers" in prompt


def test_reporter_carries_the_trajectory_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Optimization Trajectory" in prompt
    # The path is reconstructed from the structured trail in the order
    # applied — the roadmap's listing order is priority, not chronology —
    # and gaps stay gaps.
    assert "measured_value" in prompt
    assert "never interpolated" in prompt
    # The markdown table is the data the HTML line chart plots.
    assert "line chart" in prompt


def test_reporter_carries_the_kernel_comparison():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Kernel-Level Comparison" in prompt
    # Grounded in the analyzer rounds' nsys artifacts...
    assert "cuda_gpu_kern_sum" in prompt
    assert "nsys_stats.txt" in prompt
    # ...with honest provenance: what each profile covers, and no
    # fabricated "after" data when only round 1 was profiled.
    assert "which accepted items were in effect" in prompt
    assert "closing analyzer round may have profiled the final accepted state" in prompt
    assert "capture directory your driving instructions name as freshest" in prompt
    assert "no post-optimization profile exists" in prompt


def test_html_companion_charts_are_self_contained():
    block = _norm(OPTIMIZE_HTML_COMPANION)
    assert "Trajectory line chart" in block
    assert "Kernel before/after bars" in block
    # No chart library: data embedded inline, rendered to inline SVG.
    assert "no chart library" in block
    assert "inline SVG" in block
    # Charts never diverge from the tables they sit above.
    assert "the table is the source of truth" in block
    # New sections slot into the HTML body in markdown order.
    assert "Baseline, Optimization Trajectory, Pareto Improvement — curve mode only" in block
    assert "Kernel-Level Comparison, Failed Attempts" in block


def test_html_companion_carries_the_pareto_chart():
    block = _norm(OPTIMIZE_HTML_COMPANION)
    assert "Pareto improvement chart" in block
    assert "x = tok/s/user, y = tok/s/gpu" in block
    # Exactly two series — baseline vs final — with labeled points, and
    # the chart disappears rather than plotting invented curves.
    assert "baseline curve vs the final curve" in block
    assert "`c=<n>`" in block
    assert "Omit the chart and the section in scalar mode" in block


def test_reporter_carries_the_pareto_improvement_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Pareto Improvement" in prompt
    assert "omit this section entirely in scalar mode" in prompt
    # Provenance: baseline from the roadmap curve, final from QA's curve.
    assert "`baseline.curve`" in prompt
    # Curve-mode headline aggregates as the mean across points.
    assert "mean across concurrency points" in prompt
    # Rigor: per-point values trace to recorded curves.
    assert "Pareto values trace to recorded curves" in prompt


# -------------------------------------------------------------- SOL projector
# The projection methodology is the internal-perf-sol-analysis skill
# (peaks from its calculator, latency constants measured when a GPU is
# reachable, the α-β-u ceiling arithmetic shown in the report); the
# model architecture comes from the checkpoint's config.json. It runs
# once per campaign, against the perf-optimize baseline artifacts, and
# its guidance addresses the Analyzer and Reporter.


def test_projector_prompt_carries_no_dlsim_traces():
    prompt = PROJECTOR_SYSTEM_PROMPT
    # dlsim is gone entirely — no checkout cross-check, no paths, no
    # MCP tools, no execution-path names.
    assert "dlsim" not in prompt.lower()
    assert "python/lwdlm" not in prompt
    # The structural quantities come from the checkpoint's config.json.
    assert "config.json" in prompt


def test_projector_prompt_builds_on_sol_skill():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # The methodology is the SOL skill, loaded via the Skill tool — with
    # the fully-qualified name so a plugin-namespaced install resolves,
    # and graceful degradation when the skill is not installed.
    assert "internal-perf-sol-analysis" in prompt
    assert "trtllm-agent-toolkit:internal-perf-sol-analysis" in prompt
    assert "`Skill` tool" in prompt
    assert "not available in this environment" in prompt


# --------------------------------------------------------------------------- #
# The fallback block is single-sourced in perf-analyze's ``_common`` and
# re-exported here, so the two workflows cannot drift.
# --------------------------------------------------------------------------- #


def test_projection_setup_template_states_no_methodology_as_fact():
    """The template is copied verbatim into `sol_projection.md`.

    A hardcoded `Method:` / `Peaks file:` line makes the projector assert
    the full methodology even when it ran the fallback — a false
    provenance claim in the one artifact whose job is to disclose it, and
    one the Analyzer then follows to a peaks file nobody wrote.
    """
    for label, prompt in (
        ("full", PROJECTOR_SYSTEM_PROMPT),
        ("reduced", build_projector_prompt("reduced")),
    ):
        assert "- Method: <" in prompt, label
        assert "- Peaks file: <" in prompt, label
        assert "- Method: internal-perf-sol-analysis" not in prompt, label
        assert "- Peaks file: sol_work/peaks.json" not in prompt, label
        # The environment without a calculator has something to write.
        assert "not written: no peaks" in prompt, label


def test_full_methodology_leaves_the_projector_prompt_untouched():
    assert SOL_METHODOLOGY_FALLBACK is _ANALYZE_METHODOLOGY_FALLBACK
    assert build_projector_prompt() == PROJECTOR_SYSTEM_PROMPT
    assert build_projector_prompt("nonsense") == PROJECTOR_SYSTEM_PROMPT


def test_reduced_methodology_appends_the_fallback_block_and_nothing_else():
    bundle = build_perf_optimize_prompts(include_sol=True, sol_methodology="reduced")
    assert bundle.projector == PROJECTOR_SYSTEM_PROMPT + SOL_METHODOLOGY_FALLBACK
    full = build_perf_optimize_prompts(include_sol=True)
    assert bundle.analyzer == full.analyzer
    assert bundle.optimizer == full.optimizer
    assert bundle.reporter == full.reporter


def test_projector_prompt_resolves_peaks_and_latencies_via_skill():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # Peaks come from the skill's calculator. The "resolve, never
    # recall" rule is the skill's own — the loaded skill states it, so
    # the prompt carries only what the skill cannot know: which part
    # name to resolve.
    assert "sol_calc.py peaks --part" in prompt
    assert "part-name hint" in prompt
    # Latency constants: measured here when a GPU is reachable,
    # recorded as unmeasured (never guessed) when one is not — this
    # stage may run on a login node, which the skill cannot know.
    assert "measure_channels.py" in prompt
    assert "do **not** guess" in prompt
    assert "unmeasured" in prompt


def test_projector_prompt_never_fabricates_measured_inputs():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # ``sol_calc.py analyze`` correlates measured per-op times; no
    # profiling stage has run yet, so there are none — and script inputs
    # are never invented to force a run.
    assert "never fabricate an input" in prompt
    assert "measured_ms" in prompt


def test_projector_prompt_speaks_skill_vocabulary():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    for term in ("% of SOL", "MFU", "MBU", "gap-to-SOL", "α-β-u"):
        assert term in prompt, term
    assert "compute / memory / launch" in prompt
    # The ceiling models kernel execution + per-launch latency only — a
    # gap beyond it points at host/scheduling costs it does not price.
    assert "kernel execution plus per-launch latency only" in prompt
    assert "request queueing" in prompt


def test_projector_prompt_names_internal_knowledge_and_keeps_it_consultative():
    prompt = PROJECTOR_SYSTEM_PROMPT
    assert "internal-glean-search" in prompt
    assert "internal-glean-specialist" in prompt
    # No site-specific URL is baked into the prompt.
    assert "http://" not in prompt
    assert "https://" not in prompt
    normed = _norm(prompt)
    assert "if that skill/subagent exists" in normed
    assert "consultative" in normed
    assert "reproducible from the arithmetic" in normed


def test_projector_prompt_degrades_honestly():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "Projection unavailable" in prompt
    assert "never fabricate" in prompt


def test_projector_prompt_template_sections():
    for header in (
        "## Projection setup",
        "## Projected SOL ceiling",
        "## Measured vs SOL",
        "## Headroom & bound mix",
        "## Guidance for optimization",
        "## Caveats",
    ):
        assert header in PROJECTOR_SYSTEM_PROMPT, header


def test_projector_prompt_targets_the_optimize_pipeline():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # Once per campaign, against perf-optimize's artifact layout: the
    # baseline lives under baseline/ and the parallel mapping comes from
    # the live tuning config, not the task-level extra_llm_api_options.
    assert "once per campaign" in prompt
    assert "baseline/benchmark_results.md" in prompt
    assert "tuning/extra_llm_api_options.yaml" in prompt
    # Guidance addresses this workflow's consumers — the Analyzer owns
    # the roadmap; there is no Profiler stage here.
    assert "Analyzer" in prompt
    assert "the Analyzer owns `roadmap.yaml`" in prompt
    assert "expected_gain_pct" in prompt
    assert "Profiler" not in prompt
    # Later stages' files are off-limits.
    assert "do not touch them" in prompt
    # Curve mode: the ceiling is derived per configured point.
    assert "once per concurrency point" in prompt
    assert "point by point" in prompt


def test_sol_analyzer_context_is_context_not_evidence():
    block = _norm(SOL_ANALYZER_CONTEXT)
    assert "context, not evidence" in block
    assert "% of SOL" in block
    # The projection bounds roadmap expectations but never outranks the
    # trace, stays valid across rounds, and degrades honestly.
    assert "Sanity-bound `expected_gain_pct`" in block
    assert "outranks the projection" in block
    assert "never re-derive it" in block
    assert "missing or declares itself unavailable" in block
    assert "never present a SOL number as a measured one" in block


def test_sol_analyzer_context_forbids_silent_exhaustion():
    # A campaign must never end with projected headroom that is neither
    # attacked nor accounted for: an exhausted roadmap owes the
    # remaining-gap attribution.
    block = _norm(SOL_ANALYZER_CONTEXT)
    assert "No silent exhaustion" in block
    assert "## Remaining-gap attribution" in block
    assert "no actionable pending item" in block
    # Every gap part gets an item or an evidence-backed infeasibility
    # reason — citing artifacts, never hunches.
    assert "new roadmap item" in block
    assert "evidence-backed reason it cannot be closed in this campaign" in block
    assert "cite the artifact, not a hunch" in block
    # The unexplained bucket stays visible, and the evaluator's verdict
    # lines are named as an evidence source.
    assert "unexplained" in block
    assert "never absorbed into the other buckets" in block
    assert "Gap implication" in block


def test_sol_analyzer_context_correlates_per_round_with_the_skill_calculator():
    block = _norm(SOL_ANALYZER_CONTEXT)
    # The correlation is the skill's calculator over structural facts,
    # joined against the projector's persisted peaks file.
    assert "sol_calc.py analyze" in block
    assert "regions.json" in block
    assert "sol_work/peaks.json" in block
    assert "never invent params or `measured_ms` rows" in block
    # The joined table lands in the findings' dedicated section, with
    # honest degradation when a precondition fails.
    assert "## SOL correlation (measured vs ceiling)" in block
    assert "Correlation unavailable" in block
    # Optimize-specific placement and cadence: per-round artifacts, one
    # campaign-level peaks file, a fresh join every profiling round.
    assert "this round's `analysis/` directory" in block
    assert "Re-run the correlation **every round that profiles**" in block
    # …and not on the rounds that produce no measured rows to join.
    assert "A replan-only round produced no new measured rows" in block


def test_projector_prompt_persists_peaks_for_the_analyzer():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "Persist the machine-readable peaks file" in prompt
    assert "sol_work/peaks.json" in prompt
    # And the required-output template records the path — as the
    # placeholder it is, so a run without a calculator does not assert a
    # file it never wrote (see the template guard above).
    assert "Peaks file: <sol_work/peaks.json" in prompt


def test_analyzer_composes_the_shared_findings_contract():
    # perf-optimize's analyzer is perf-analyze's analyzer plus the
    # roadmap machinery: the findings report follows the same shared
    # contract (including the reserved SOL correlation section).
    assert PROFILE_FINDINGS_CONTRACT in ANALYZER_SYSTEM_PROMPT
    assert "## SOL correlation (measured vs ceiling)" in _norm(PROFILE_FINDINGS_CONTRACT)


def test_sol_optimizer_context_aims_at_the_binding_ceiling_without_scope_creep():
    block = _norm(SOL_OPTIMIZER_CONTEXT)
    # Context, not spec: the item outranks the projection, and the
    # projection never grows the change.
    assert "context, not spec" in block
    assert "Aim the implementation at the binding ceiling" in block
    assert "The projection never expands the item" in block
    assert "not yours to chase" in block
    assert "outrank the projection" in block
    # The claimed mechanism becomes checkable downstream.
    assert "SOL alignment:" in block
    assert "Mapping to the roadmap item" in block
    # Honest degradation when the projection is absent.
    assert "missing or declares itself unavailable" in block


def test_sol_reporter_guidance_carries_remaining_gap_accountability():
    block = _norm(SOL_OPTIMIZE_REPORTER_GUIDANCE)
    assert "Remaining-gap accountability" in block
    # The four exhaustive verdicts.
    for verdict in ("`closed`", "`infeasible: <constraint>`", "`untried`", "`unexplained`"):
        assert verdict in block, verdict
    # Verdicts trace to artifacts; fabricated justifications are worse
    # than an honest unexplained bucket. The analyzer's per-op
    # correlation table is a named evidence source for the gap parts.
    assert "Every accountability verdict traces to an artifact" in block
    assert "Gap implication" in block
    assert "SOL correlation" in block
    assert "worse than reporting it unexplained" in block
    # A zero-accept campaign still owes the breakdown.
    assert "accepted nothing must still fill the accountability" in block


def test_evaluator_negative_verdicts_carry_gap_implication():
    # PUSH_BACK/REJECT evidence feeds the analyzer's re-planning and the
    # report's remaining-gap attribution — without any SOL exposure.
    prompt = _norm(EVALUATOR_SYSTEM_PROMPT)
    assert "Gap implication:" in prompt
    for tag in (
        "mechanism-already-present",
        "mechanism-inapplicable",
        "applied-but-no-gain",
        "blocked-by-constraint",
    ):
        assert tag in prompt, tag
    # Judged from the evaluator's own evidence, and recorded in the
    # progress entry too.
    assert "judged from your own evidence" in prompt
    assert "include the `Gap implication` line" in prompt


def test_sol_reporter_guidance_carries_the_headroom_story():
    block = _norm(SOL_OPTIMIZE_REPORTER_GUIDANCE)
    assert "## Projection vs Measured" in block
    # Placement inside the optimize report.
    assert 'between "Final Verification" and "Config & Code Diff Summary"' in block
    # The optimize-flavored table: baseline vs final % of SOL.
    assert "Baseline % of SOL" in block
    assert "Final % of SOL" in block
    assert "final % of SOL − baseline % of SOL" in block
    # The final side falls back to the ledger, and a no-accept campaign
    # captured no headroom.
    assert "`current_best`" in block
    assert "captured none of the projected headroom" in block
    # Honesty rules.
    assert "Projection unavailable" in block
    assert "never fabricate" in block


def test_sol_bundle_extends_analyzer_optimizer_and_reporter_only():
    base = build_perf_optimize_prompts(include_sol=False)
    sol = build_perf_optimize_prompts(include_sol=True)
    assert "SOL projection as context" in sol.analyzer
    assert "SOL projection as context" in sol.optimizer
    assert "Projection vs Measured" in sol.reporter
    assert "SOL projection as context" not in base.analyzer
    assert "SOL projection as context" not in base.optimizer
    assert "Projection vs Measured (this task has a `sol` block)" not in base.reporter
    # Everything else — including the projector's own prompt, which is
    # always in the bundle (the stage gate lives in the workflow) — is
    # unchanged.
    for role in ("benchmarker", "projector", "evaluator", "qa"):
        assert getattr(sol, role) == getattr(base, role), role


def test_sol_bundle_composes_with_slurm_and_restriction():
    bundle = build_perf_optimize_prompts(
        include_slurm_environment=True, approaches=["config"], include_sol=True
    )
    assert "SOL projection as context" in bundle.analyzer
    assert "SOL projection as context" in bundle.optimizer
    assert "Approach restriction (`optimize.approaches`)" in bundle.analyzer
    assert "slurm-environment" in bundle.analyzer
    assert "Projection vs Measured" in bundle.reporter
    # The evaluator and QA judge on measurements alone — no SOL context.
    assert "SOL projection" not in bundle.evaluator
    assert "SOL projection" not in bundle.qa


def test_html_companion_overlays_the_sol_projected_curve():
    block = _norm(OPTIMIZE_HTML_COMPANION)
    assert "SOL-projected" in block
    assert "third polyline" in block
    # The overlay is honest: omitted, never approximated, without data.
    assert "omit the overlay, never approximate it" in block


# ----------------------------------------------------------------------- slurm


def test_slurm_bundle_augments_all_server_roles_but_not_reporter():
    base = build_perf_optimize_prompts(include_slurm_environment=False)
    slurm = build_perf_optimize_prompts(include_slurm_environment=True)
    for role in ("benchmarker", "analyzer", "optimizer", "evaluator", "qa"):
        assert "slurm-environment" in getattr(slurm, role), role
        assert "slurm-environment" not in getattr(base, role), role
    # The reporter never launches a server, so it is unchanged — and so
    # is the projector (no server work; under Slurm it runs on the login
    # node and records the latency constants as unmeasured).
    assert slurm.reporter == base.reporter
    assert slurm.projector == base.projector
    assert "slurm-environment" not in slurm.projector


def test_slurm_bundle_preserves_canonical_templates():
    slurm = build_perf_optimize_prompts(include_slurm_environment=True)
    for role in _MEASURING:
        for flag in _BENCHMARK_CANONICAL_FLAGS:
            assert flag in getattr(slurm, role), (role, flag)
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in slurm.analyzer, flag


# --------------------------------------------------------- approach restriction


def test_approach_restriction_note_only_built_for_real_restrictions():
    assert approach_restriction_note(("config", "code")) == ""
    assert approach_restriction_note(()) == ""  # nothing allowed = nonsense, no note
    code_only = _norm(approach_restriction_note(("code",)))
    # The disallowed side's deterministic guard is spelled out...
    assert "tuning/extra_llm_api_options.yaml" in code_only
    assert "auto-rejects the attempt without any evaluation" in code_only
    # ... including the defaults-in-source loophole.
    assert "the same violation in disguise" in code_only
    config_only = _norm(approach_restriction_note(("config",)))
    assert "git status --porcelain" in config_only
    assert "read-only for every role" in config_only


def test_restricted_bundle_augments_planning_and_gating_roles_only():
    base = build_perf_optimize_prompts()
    restricted = build_perf_optimize_prompts(approaches=["code"])
    marker = "Approach restriction (`optimize.approaches`)"
    for role in ("analyzer", "optimizer", "evaluator"):
        assert marker in getattr(restricted, role), role
        assert marker not in getattr(base, role), role
    # The benchmarker measures, qa verifies the final state, and the
    # reporter synthesizes — none plans, applies, or judges items.
    assert restricted.benchmarker == base.benchmarker
    assert restricted.qa == base.qa
    assert restricted.reporter == base.reporter


def test_full_approaches_list_leaves_bundle_unchanged():
    assert build_perf_optimize_prompts(approaches=["config", "code"]) == DEFAULT_PROMPTS
    assert build_perf_optimize_prompts(approaches=None) == DEFAULT_PROMPTS


def test_restriction_composes_with_slurm_augmentation():
    bundle = build_perf_optimize_prompts(include_slurm_environment=True, approaches=["code"])
    assert "Approach restriction (`optimize.approaches`)" in bundle.optimizer
    assert "slurm-environment" in bundle.optimizer
    for flag in _BENCHMARK_CANONICAL_FLAGS:
        assert flag in bundle.evaluator, flag


# ------------------------------------------------------ per-kernel coverage


def _coverage_bundle():
    return build_perf_optimize_prompts(
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0}
    )


def test_kernel_coverage_note_interpolates_the_task_bars():
    block = _norm(kernel_coverage_analyzer_note(0.75, 92.0))
    assert "0.75%" in block
    assert "92.0%" in block
    # The contract supersedes Run C's bounded top-kernel targeting.
    assert "supersedes Run C's target selection" in block


def test_kernel_coverage_note_poses_both_questions_per_kernel():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "can it be made faster?" in block
    assert "can it be fused with its neighbors?" in block
    # Each answer is an item or an evidence-backed dismissal — recorded
    # in the schema-validated ledger, with the abort consequence named.
    assert "kernel_ledger.yaml" in block
    assert "aborts the stage" in block
    assert "disposition: item" in block
    assert "disposition: dismissed" in block


def test_kernel_coverage_note_grounds_fusion_in_observed_adjacency():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    assert "observed adjacency, not guesses" in block
    assert "fusion.neighbors" in block
    # The recurring legitimate dismissals are named so verdicts stay
    # evidence-tagged rather than free-form.
    for tag in (
        "at-sol-floor",
        "below-materiality",
        "multi-consumer-pinned",
        "already-fused",
        "phase-boundary",
        "needs-rebuild",
    ):
        assert tag in block, tag
    # Materiality is judged on the whole fusible chain, not one kernel.
    assert "whole chain" in block


def test_kernel_coverage_needs_rebuild_requires_ruling_out_a_replacement():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    # "The incumbent ships compiled" alone does not dismiss a kernel: a
    # written-from-scratch replacement routed from Python must also be
    # ruled out, otherwise the answer is an item that swaps the call
    # site and lands only if the new kernel measures faster.
    assert "reroute the Python call site" in block
    assert "the replacement path is also ruled out" in block
    assert "swap the call site" in block
    # The same bar governs fusion cells that blame a compiled neighbor.
    assert "replacing the incumbent plus its glue" in block


def test_kernel_coverage_note_bounds_the_capture():
    block = _norm(kernel_coverage_analyzer_note(0.5, 95.0))
    # Multi-pass with re-filtering on missing stems, bounded passes, and
    # the honest degrade for kernels no pass reached.
    assert "3 passes" in block
    assert "still-missing stems" in block
    assert "server_ncu_pass<k>.ncu-rep" in block
    assert 'ncu: "unavailable: <reason>"' in block
    # Unactionable below-noise-floor items are not a valid answer.
    assert "below-materiality` dismissal wearing an item costume" in block


def test_kernel_coverage_bundle_extends_analyzer_and_reporter_only():
    base = build_perf_optimize_prompts()
    coverage = _coverage_bundle()
    assert "Per-kernel coverage contract" in coverage.analyzer
    assert "## Kernel Coverage" in coverage.reporter
    assert "Per-kernel coverage contract" not in base.analyzer
    assert "## Kernel Coverage" not in base.reporter
    for role in ("benchmarker", "projector", "optimizer", "evaluator", "qa"):
        assert getattr(coverage, role) == getattr(base, role), role


def test_kernel_coverage_off_leaves_bundle_unchanged():
    assert build_perf_optimize_prompts(kernel_coverage=None) == DEFAULT_PROMPTS


def test_kernel_coverage_reporter_section_slots_after_kernel_comparison():
    block = _norm(KERNEL_COVERAGE_REPORTER_GUIDANCE)
    assert 'between "Kernel-Level Comparison" and "Failed Attempts"' in block
    # Dispositions resolve to campaign outcomes and the untried tail is
    # itemized, never buried.
    assert "pending at campaign end" in block
    assert "untried tail" in block
    # Honest degrade when the final ledger is missing.
    assert "Kernel coverage ledger unavailable" in block


def test_kernel_coverage_composes_with_sol_slurm_and_restriction():
    bundle = build_perf_optimize_prompts(
        include_slurm_environment=True,
        approaches=["code"],
        include_sol=True,
        kernel_coverage={"min_share_pct": 0.5, "coverage_target_pct": 95.0},
    )
    assert "Per-kernel coverage contract" in bundle.analyzer
    assert "SOL projection as context" in bundle.analyzer
    assert "Approach restriction (`optimize.approaches`)" in bundle.analyzer
    assert "slurm-environment" in bundle.analyzer
    assert "## Kernel Coverage" in bundle.reporter
    assert "Projection vs Measured" in bundle.reporter


def test_measuring_roles_inherit_the_server_identity_checks():
    """Every role that launches a server must carry the stale-server guards.

    perf-optimize is where a stale server does the most damage: each
    `approach: config` item rewrites the *same*
    `tuning/extra_llm_api_options.yaml` against the *same* checkpoint, so a
    survivor from the previous round answers on :8000 under a matching
    model name and the evaluator's measured gain — the accept/reject gate
    — silently scores the wrong config.
    """
    for name, prompt in (
        ("benchmarker", BENCHMARKER_SYSTEM_PROMPT),
        ("optimizer", OPTIMIZER_SYSTEM_PROMPT),
        ("evaluator", EVALUATOR_SYSTEM_PROMPT),
        ("qa", QA_SYSTEM_PROMPT),
    ):
        text = _norm(prompt)
        assert "port 8000 already in use" in text, f"{name} lost the port precheck"
        assert "owns_port" in text, f"{name} lost the listener-ownership check"
        assert "not owned by PID" in text, f"{name} lost the identity failure path"
