"""Tests that the role prompts carry the canonical command templates.

The perf-analyze serving roles must drive ``benchmark_serving.py`` and
``nsys profile`` from a fixed command template rather than improvising
flags per run, so these tests pin the flags that template guarantees.
"""

from __future__ import annotations

import re

from agent_flow.workflows.perf_analyze.prompts import (
    ANALYZER_SYSTEM_PROMPT,
    BENCHMARKER_SYSTEM_PROMPT,
    PROJECTOR_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    build_perf_analyze_prompts,
    build_projector_prompt,
)
from agent_flow.workflows.perf_analyze.prompts._common import (
    BOTTLENECK_TAXONOMY,
    CASEBOOK_CONSULTATION,
    HTML_COMPANION,
    PROFILE_FINDINGS_CONTRACT,
    SERVER_LIFECYCLE,
    SOL_ANALYZER_CONTEXT,
    SOL_CORRELATION_METHOD,
    SOL_METHODOLOGY_FALLBACK,
    SOL_REPORTER_GUIDANCE,
)


def _norm(text: str) -> str:
    """Collapse whitespace so substring assertions survive line-wrapping."""
    return re.sub(r"\s+", " ", text)


# Built-in tools the Claude Code CLI dropped from its default toolset
# (``--tools default``): the agents run on that preset, so a prompt that
# instructs one of these makes the agent call a tool that resolves to
# "No such tool available" before it falls back to shell ``grep``. Prompts
# must steer to ``grep``/``rg`` via ``Bash`` instead. Matched on a word
# boundary so lowercase shell ``grep`` and words like "Global" don't trip.
_REMOVED_BUILTIN_TOOLS = ("Grep", "Glob")


def _removed_tool_refs(prompt: str) -> list[str]:
    return [name for name in _REMOVED_BUILTIN_TOOLS if re.search(rf"\b{name}\b", prompt)]


# Canonical ``benchmark_serving.py`` flags both serving roles must carry
# (the analyzer replays the benchmarker's load, so it inherits them too).
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

# Canonical ``ncu`` flags the analyzer's Run C must carry.
_NCU_CANONICAL_FLAGS = (
    "--target-processes all",
    "--profile-from-start off",
    "--section SpeedOfLight",
    "--launch-count",
    "--kernel-name",
)


def test_benchmarker_prompt_has_canonical_benchmark_flags():
    for flag in _BENCHMARK_CANONICAL_FLAGS:
        assert flag in BENCHMARKER_SYSTEM_PROMPT, flag


def test_analyzer_prompt_has_canonical_benchmark_flags():
    for flag in _BENCHMARK_CANONICAL_FLAGS:
        assert flag in ANALYZER_SYSTEM_PROMPT, flag


def test_analyzer_prompt_has_canonical_nsys_flags():
    for flag in _NSYS_CANONICAL_FLAGS:
        assert flag in ANALYZER_SYSTEM_PROMPT, flag


def test_analyzer_keeps_capture_range_end_stop_safety_flag():
    # The template omits it, but the automated run must keep it so nsys
    # does not SIGTERM the server at the window's end (default
    # ``stop-shutdown``), which crashes the engine and yields no report.
    assert "--capture-range-end=stop" in ANALYZER_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# ncu deep dive (Run C): a bounded per-kernel profile of the top nsys kernels
# over the same iteration window, interpreted with the
# perf-nsight-compute-analysis skill; the findings carry a dedicated section
# and the ranked hypotheses synthesize nsys + ncu + SOL correlation.
# --------------------------------------------------------------------------- #


def test_analyzer_prompt_has_canonical_ncu_flags():
    for flag in _NCU_CANONICAL_FLAGS:
        assert flag in ANALYZER_SYSTEM_PROMPT, flag
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "do not improvise the ncu flags" in prompt


def test_analyzer_loads_ncu_analysis_skill_and_degrades():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The skill is the capture + interpretation methodology, with the
    # fully-qualified fallback for plugin-namespaced installs.
    assert "perf-nsight-compute-analysis" in prompt
    assert "trtllm-agent-toolkit:perf-nsight-compute-analysis" in prompt
    # A missing tool / permission never blocks the run and never yields
    # fabricated metrics — the section degrades to a one-liner.
    assert "ERR_NVGPUCTRPERM" in prompt
    assert "ncu unavailable" in prompt


def test_ncu_run_targets_top_nsys_kernels_with_bounded_capture():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # Targets come from Run A's top-kernel table — never a blind sweep.
    assert "targets the top kernels Run A surfaced" in prompt
    # Kernel replay serializes the GPU: the replayed benchmark's client
    # numbers must never be reported as performance results.
    assert "not measurements" in prompt


def test_findings_contract_carries_the_ncu_section():
    block = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "## ncu kernel analysis" in block
    # Per-kernel classification per the skill's thresholds, degrading
    # honestly when ncu did not run.
    assert "bound class" in block
    assert "ncu unavailable" in block


def test_findings_contract_synthesizes_the_three_analyses():
    block = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "three evidence pillars" in block
    for pillar in ("nsys timeline", "ncu kernel analysis", "SOL correlation"):
        assert pillar in block, pillar
    # A pillar that did not run is named as missing, never skipped.
    assert "never silently skipped" in block


def test_reporter_grounds_recommendations_in_all_three_analyses():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Ground every recommendation in the three analyses" in prompt
    assert "ncu kernel analysis" in prompt
    # The fix must match the targeted kernel's measured bound class.
    assert "bound class" in prompt


def test_prompts_tell_agents_not_to_improvise_flags():
    # Both serving roles are steered to the canonical template rather than
    # figuring the command out on their own.
    assert "do not improvise" in BENCHMARKER_SYSTEM_PROMPT
    assert "do not improvise" in ANALYZER_SYSTEM_PROMPT


def test_no_prompt_references_removed_builtin_tools():
    # The agents run on the CLI's ``default`` toolset, which no longer
    # includes ``Grep``/``Glob``; instructing them makes the agent call a
    # nonexistent tool. Every role prompt must avoid the tool names.
    for name, prompt in (
        ("benchmarker", BENCHMARKER_SYSTEM_PROMPT),
        ("projector", PROJECTOR_SYSTEM_PROMPT),
        ("analyzer", ANALYZER_SYSTEM_PROMPT),
        ("reporter", REPORTER_SYSTEM_PROMPT),
    ):
        assert not _removed_tool_refs(prompt), (name, _removed_tool_refs(prompt))


def test_analyzer_prompt_steers_source_search_to_bash_grep():
    # The knob-verification step must search the checkout with shell
    # ``grep``/``rg`` via ``Bash`` (not the removed ``Grep`` tool).
    assert "grep" in ANALYZER_SYSTEM_PROMPT
    assert "`Bash`" in ANALYZER_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Bottleneck taxonomy: kernel-launch vs host-prep overhead must be split, and
# the CUDA-graph prescription must not be a blanket claim over the whole bucket
# (CUDA graphs collapse the launch storm inside the model forward but do NOT
# remove host input-prep like ``_prepare_inputs`` that runs outside the graph).
# --------------------------------------------------------------------------- #


def test_taxonomy_splits_launch_from_host_prep_overhead():
    taxonomy = _norm(BOTTLENECK_TAXONOMY)
    assert "Kernel-launch overhead" in taxonomy
    assert "Host-prep" in taxonomy


def test_taxonomy_drops_blanket_cuda_graph_prescription():
    # The old text ended the whole bucket with "Often fixable with CUDA
    # graphs / overlap scheduler", wrongly implying graphs fix host prep too.
    assert "Often fixable with CUDA graphs" not in BOTTLENECK_TAXONOMY


def test_taxonomy_warns_cuda_graphs_do_not_remove_host_prep():
    taxonomy = _norm(BOTTLENECK_TAXONOMY)
    assert "CUDA graphs do not remove this" in taxonomy
    # The host-prep sub-cause is anchored to a concrete named phase.
    assert "_prepare_inputs" in taxonomy


def test_reporter_ranks_recommendations_by_bottleneck_share():
    # Recommendations must be ranked by how much of the measured dominant
    # cost each fix removes, not by ease of implementation.
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "share of the measured bottleneck" in prompt
    assert "#1 recommendation must attack" in prompt


def test_reporter_warns_cuda_graphs_are_not_the_top_host_prep_fix():
    # A cheaper config fix (CUDA graphs) that only touches a smaller
    # component must not be ranked #1 when host prep is the dominant cost.
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "remove host input-prep" in prompt
    assert "does not belong at #1" in prompt
    # And the rigor rules reinforce impact-based ranking.
    assert "Rank recommendations by impact on the dominant cost" in prompt


def test_slurm_bundle_preserves_canonical_templates():
    # Slurm augmentation only appends container-bootstrap prose, so the
    # canonical templates must survive in the augmented serving prompts.
    slurm = build_perf_analyze_prompts(include_slurm_environment=True)
    for flag in _BENCHMARK_CANONICAL_FLAGS:
        assert flag in slurm.benchmarker, flag
    for flag in (*_BENCHMARK_CANONICAL_FLAGS, *_NSYS_CANONICAL_FLAGS, *_NCU_CANONICAL_FLAGS):
        assert flag in slurm.analyzer, flag


# --------------------------------------------------------------------------- #
# SOL projector: the projection methodology is the internal-perf-sol-analysis
# skill (peaks from its calculator, latency constants measured when a GPU is
# reachable, the α-β-u ceiling arithmetic shown in the report); the model
# architecture comes from the checkpoint's config.json. Internal knowledge
# is consultative reference only, and the role degrades to an honest
# "projection unavailable" file.
# --------------------------------------------------------------------------- #


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
# ``internal-perf-sol-analysis`` carries the ``internal-`` prefix, so
# open-source builds of the trtllm-agent-toolkit plugin strip it while keeping
# ``perf-analysis``. Which one this session has is resolved in Python before
# the stage runs (``sol_methodology.resolve_sol_methodology``); all it changes
# is whether the projector's prompt carries the fallback block.
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
    """The environment that has the skill gets exactly the prompt it always did."""
    assert build_projector_prompt() == PROJECTOR_SYSTEM_PROMPT
    assert build_projector_prompt("full") == PROJECTOR_SYSTEM_PROMPT
    # An unrecognised value must never silently downgrade the stage.
    assert build_projector_prompt("nonsense") == PROJECTOR_SYSTEM_PROMPT
    assert build_perf_analyze_prompts(include_sol=True) == build_perf_analyze_prompts(
        include_sol=True, sol_methodology="nonsense"
    )


def test_reduced_methodology_appends_the_fallback_block_and_nothing_else():
    reduced = build_projector_prompt("reduced")
    assert reduced == PROJECTOR_SYSTEM_PROMPT + SOL_METHODOLOGY_FALLBACK
    bundle = build_perf_analyze_prompts(include_sol=True, sol_methodology="reduced")
    assert bundle.projector.endswith(SOL_METHODOLOGY_FALLBACK)
    # Only the projector's brief changes — every other role keeps its prompt.
    full = build_perf_analyze_prompts(include_sol=True)
    assert bundle.analyzer == full.analyzer
    assert bundle.reporter == full.reporter
    assert bundle.benchmarker == full.benchmarker


def test_fallback_block_names_perf_analysis_and_withholds_the_peaks_file():
    """The one deliverable the fallback cannot produce is the peaks file.

    ``sol_calc.py`` ships with the missing skill, so nothing downstream
    reads it and a hand-made file would later be mistaken for calculator
    output.
    """
    block = _norm(SOL_METHODOLOGY_FALLBACK)
    assert "`perf-analysis`" in block
    assert "Skip `sol_work/peaks.json`" in block
    # It degrades the projection; it never invents one.
    assert "not calculator-resolved" in block
    assert "Never fabricate." in block


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


def test_projector_prompt_derives_sol_ceiling_via_skill_model():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "speed-of-light" in prompt
    assert "α-β-u" in prompt
    # The inline roofline arithmetic and the derated proj column are
    # gone — the skill is the method, SOL-only.
    assert "max(t_mem, t_math)" not in prompt
    assert "proj (realistic)" not in prompt
    assert "derate" not in prompt.lower()
    # The arithmetic must still be reproducible from the report.
    assert "numbers substituted" in prompt


def test_projector_prompt_never_fabricates_measured_inputs():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    # ``sol_calc.py analyze`` correlates measured per-op times; the
    # Analyzer has not run yet, so there are none — and script inputs
    # are never invented to force a run.
    assert "never fabricate an input" in prompt
    assert "measured_ms" in prompt


def test_projector_prompt_speaks_skill_vocabulary():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    for term in ("% of SOL", "MFU", "MBU", "gap-to-SOL"):
        assert term in prompt, term
    # The skill's bound taxonomy names which ceiling binds.
    assert "compute / memory / launch" in prompt
    # The ceiling models kernel execution + per-launch latency only — a
    # gap beyond it points at host/scheduling costs it does not price.
    assert "kernel execution plus per-launch latency only" in prompt
    assert "request queueing" in prompt


def test_projector_prompt_names_the_internal_knowledge_route():
    prompt = PROJECTOR_SYSTEM_PROMPT
    assert "internal-glean-search" in prompt
    assert "internal-glean-specialist" in prompt
    # Named as optional -- the session may not have either.
    assert "if that skill/subagent exists" in _norm(prompt)


def test_projector_prompt_ships_no_hosted_endpoint():
    """No site-specific URL is baked into the prompt."""
    prompt = PROJECTOR_SYSTEM_PROMPT
    assert "http://" not in prompt
    assert "https://" not in prompt


def test_projector_prompt_keeps_internal_knowledge_consultative():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "consultative" in prompt
    # Projected numbers must be reproducible from written-down arithmetic.
    assert "reproducible from the arithmetic" in prompt


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


def test_sol_bundle_extends_analyzer_and_reporter_only():
    base = build_perf_analyze_prompts(include_sol=False)
    sol = build_perf_analyze_prompts(include_sol=True)
    assert "SOL projection as context" in sol.analyzer
    assert "Projection vs Measured" in sol.reporter
    # The projection guidance never leaks into the other roles or the
    # un-augmented bundle.
    assert sol.benchmarker == base.benchmarker
    assert sol.projector == base.projector
    assert "SOL projection as context" not in base.analyzer
    assert "Projection vs Measured" not in base.reporter


def test_sol_reporter_guidance_weighs_and_degrades():
    block = _norm(SOL_REPORTER_GUIDANCE)
    # The projection must be weighed in the verdict and recommendations.
    assert "Main Bottleneck" in block
    assert "Recommendations" in block
    # The section speaks the skill's vocabulary.
    assert "% of SOL" in block
    # Measured evidence wins conflicts; unavailability is stated honestly.
    assert "measured evidence wins" in block
    assert "Projection unavailable" in block
    assert "never fabricate" in block


def test_sol_analyzer_context_is_context_not_evidence():
    block = _norm(SOL_ANALYZER_CONTEXT)
    assert "context, not evidence" in block
    assert "outranks the projection" in block
    # Hypothesis ranking keys off the skill's metrics.
    assert "% of SOL" in block


# --------------------------------------------------------------------------- #
# Measured↔SOL correlation: the projector persists a machine-readable peaks
# file (sol_work/peaks.json) and the sol-gated analyzer joins its fresh
# per-op measurements against it with the skill's ``sol_calc.py analyze``,
# reporting the joined per-op table in a dedicated findings section — with
# structural facts only, never invented inputs.
# --------------------------------------------------------------------------- #


def test_projector_prompt_persists_peaks_for_the_analyzer():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "Persist the machine-readable peaks file" in prompt
    assert "sol_work/peaks.json" in prompt
    # And the required-output template records the path — as the
    # placeholder it is, so a run without a calculator does not assert a
    # file it never wrote (see the template guard above).
    assert "Peaks file: <sol_work/peaks.json" in prompt


def test_findings_contract_reserves_the_sol_correlation_section():
    block = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "## SOL correlation (measured vs ceiling)" in block
    # Reserved, not required: without a sol block the section is omitted.
    assert "Omit the section entirely otherwise" in block


def test_analyzer_composes_shared_findings_contract_and_taxonomy():
    # The perf-analyze analyzer is perf-optimize's analyzer minus the
    # roadmap: both compose the same findings contract and taxonomy.
    assert "Required findings structure" in ANALYZER_SYSTEM_PROMPT
    assert "## Bottleneck taxonomy" in ANALYZER_SYSTEM_PROMPT
    # The verdict still belongs to the Reporter.
    assert "do not** issue the final verdict" in ANALYZER_SYSTEM_PROMPT


def test_sol_correlation_runs_the_skill_calculator_on_structural_facts():
    block = _norm(SOL_CORRELATION_METHOD)
    # The join is the skill's calculator over the projector's peaks. The
    # `regions.json` schema and the region-key contract are the skill's
    # own and are not restated — the block points at them and adds only
    # what the skill cannot know about this stage.
    assert "sol_calc.py analyze" in block
    assert "regions.json" in block
    assert "sol_work/peaks.json" in block
    assert "what the skill cannot know about this stage" in block
    # Latency constants: merged by the projector or measured here (a GPU
    # is reachable at this stage by construction).
    assert "measure_channels.py" in block
    assert "reachable here by construction" in block
    # Structural facts only; unmappable regions roll into `other`.
    assert "never invent params or `measured_ms` rows" in block
    assert "check-recipe" in block
    # The joined table lands in the findings section, degrading honestly.
    assert "## SOL correlation (measured vs ceiling)" in block
    assert "Correlation unavailable" in block


def test_sol_analyzer_context_carries_correlation_and_workspace_paths():
    block = _norm(SOL_ANALYZER_CONTEXT)
    assert "sol_calc.py analyze" in block
    # perf-analyze placement: artifacts sit next to the projector's peaks.
    assert "under `<workspace>/sol_work/`" in block


def test_correlation_is_gated_on_the_sol_block():
    base = build_perf_analyze_prompts(include_sol=False)
    sol = build_perf_analyze_prompts(include_sol=True)
    assert "sol_calc.py analyze" not in base.analyzer
    assert "sol_calc.py analyze" in sol.analyzer


def test_sol_reporter_guidance_lifts_the_correlation_table():
    block = _norm(SOL_REPORTER_GUIDANCE)
    assert "SOL correlation (measured vs ceiling)" in block
    assert "per-op table" in block
    # Absence degrades honestly rather than substituting.
    assert "unavailable" in block


def test_slurm_and_sol_bundles_compose():
    both = build_perf_analyze_prompts(include_slurm_environment=True, include_sol=True)
    # Slurm bootstrap and canonical templates survive in the serving roles.
    assert "slurm-environment" in both.benchmarker
    for flag in (*_BENCHMARK_CANONICAL_FLAGS, *_NSYS_CANONICAL_FLAGS, *_NCU_CANONICAL_FLAGS):
        assert flag in both.analyzer, flag
    # Both SOL extensions land on top.
    assert "SOL projection as context" in both.analyzer
    assert "Projection vs Measured" in both.reporter
    # The projector stays un-augmented (no server work; under Slurm it
    # runs on the login node and notes unmeasured latency constants).
    assert "slurm-environment" not in both.projector


# --------------------------------------------------------------------------- #
# Optimization casebook: both serving roles must proactively load the
# ``perf-optimization-casebook`` skill as read-only reference so their
# performance analysis is grounded in known TRT-LLM precedents.
# --------------------------------------------------------------------------- #


def test_serving_prompts_load_optimization_casebook():
    # Both roles that analyze the TRT-LLM run must be told to load the
    # casebook skill — and via the ``Skill`` tool, not merely mention it.
    for name, prompt in (
        ("benchmarker", BENCHMARKER_SYSTEM_PROMPT),
        ("analyzer", ANALYZER_SYSTEM_PROMPT),
    ):
        assert "perf-optimization-casebook" in prompt, name
        assert "`Skill` tool" in prompt, name


def test_reporter_prompt_does_not_own_casebook_consultation():
    # The user scoped this to the two serving roles; the reporter is
    # unchanged (it does not load servers or the casebook consultation
    # block). Guards against the shared block leaking into the reporter.
    assert "Ground your analysis in the optimization casebook" not in REPORTER_SYSTEM_PROMPT


def test_casebook_consultation_is_read_only_and_degrades_gracefully():
    block = _norm(CASEBOOK_CONSULTATION)
    # Consulted as read-only reference, never applied at this stage.
    assert "read-only reference material only" in block
    assert "apply optimizations" in block  # part of the "do not apply ..." constraint
    # Names the fully-qualified skill so a plugin-namespaced install resolves.
    assert "trtllm-agent-toolkit:perf-optimization-casebook" in block
    # No hard dependency: a missing skill must not block the run.
    assert "not available in this environment" in block


def test_slurm_prompts_keep_casebook_consultation():
    # Slurm augmentation appends to the base prompt, so the casebook
    # consultation must survive in both augmented serving prompts.
    slurm = build_perf_analyze_prompts(include_slurm_environment=True)
    for prompt in (slurm.benchmarker, slurm.analyzer):
        assert "perf-optimization-casebook" in prompt


# --------------------------------------------------------------------------- #
# HTML companion chart (ported from perf-optimize's report charts): the
# companion renders the top-kernels table as self-contained inline-SVG bars,
# degrading honestly when the profile produced no top-kernels table.
# --------------------------------------------------------------------------- #


def test_html_companion_chart_is_self_contained():
    block = _norm(HTML_COMPANION)
    assert "Top-kernel share bars" in block
    # No chart library: data embedded inline, rendered to inline SVG.
    assert "no chart library" in block
    assert "inline SVG" in block
    # The chart never diverges from the table it plots.
    assert "the table is the source of truth" in block
    assert "charts that plot exactly the numbers in the tables" in block


def test_html_companion_chart_degrades_without_nsys_table():
    # nsys can be skipped (profile.methods, missing knob) — the chart is
    # then omitted, never charted from invented numbers.
    block = _norm(HTML_COMPANION)
    assert "omit the chart" in block


def test_reporter_prompt_carries_the_chart_contract():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Top-kernel share bars" in prompt


# --------------------------------------------------------------------------- #
# Pareto-curve mode: one benchmark run per concurrency point, per-point
# result dirs, largest-point profiling, per-point projection, and the
# measured Pareto curve in the report + HTML companion.
# --------------------------------------------------------------------------- #


def test_serving_roles_carry_the_one_run_per_point_rule():
    for role, prompt in (
        ("benchmarker", BENCHMARKER_SYSTEM_PROMPT),
        ("analyzer", ANALYZER_SYSTEM_PROMPT),
    ):
        normed = _norm(prompt)
        assert "One run per concurrency point" in normed, role
        assert "concurrency_<c>" in normed, role
        # One server launch for the whole sweep, points ascending.
        assert "sequentially in ascending order" in normed, role


def test_benchmarker_carries_the_derived_metrics_reference():
    prompt = _norm(BENCHMARKER_SYSTEM_PROMPT)
    assert "1000 / mean_tpot_ms" in prompt
    assert "output_throughput / num_gpus" in prompt
    assert "curve summary table" in prompt
    # num_gpus provenance is recorded next to the metrics.
    assert "num_gpus" in prompt


def test_analyzer_profiles_the_largest_concurrency_point():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "the largest concurrency" in prompt
    assert "Do not profile the other points" in prompt
    assert "Profiled concurrency point" in prompt


def test_projector_projects_per_point_in_curve_mode():
    prompt = _norm(PROJECTOR_SYSTEM_PROMPT)
    assert "once per concurrency point" in prompt
    # The measured-vs-projected tables pair up point by point.
    assert "point by point" in prompt


def test_html_companion_carries_the_pareto_curve_chart():
    block = _norm(HTML_COMPANION)
    assert "Pareto curve" in block
    assert "x = tok/s/user, y = tok/s/gpu" in block
    assert "`c=<n>`" in block
    # Scalar runs or a missing table drop the chart and the section.
    assert "In scalar mode, or when the curve summary table is absent" in block


def test_reporter_carries_the_pareto_curve_section():
    prompt = _norm(REPORTER_SYSTEM_PROMPT)
    assert "Pareto Curve" in prompt
    assert "omit this section entirely in scalar mode" in prompt


# --------------------------------------------------------------------------- #
# Server identity: the recipe must prove the server on :8000 is *ours*
# --------------------------------------------------------------------------- #
#
# `trtllm-serve` is launched `setsid`-detached so it survives across the
# agent's separate `Bash` calls — which also means it survives a Ctrl-C,
# because the workflow's KeyboardInterrupt handler re-raises without tearing
# it down. The port is a fixed constant (`task_schema.SERVE_PORT`), so on
# resume the freshly launched server dies with "address already in use"
# while the *stale* one answers the health poll. The old recipe polled
# `curl /health` before `kill -0 $PID`, so it accepted that answer on
# iteration 1 and benchmarked a server running an older config — or a
# different checkpoint — with no crash and no warning.


def test_lifecycle_asserts_the_port_is_free_before_launching():
    """A stale detached server must be caught before it can be measured."""
    block = _norm(SERVER_LIFECYCLE)
    assert "sport = :8000" in block
    assert "port 8000 already in use" in block
    # Sidestepping onto a free port would silently decouple the server from
    # the benchmark/profiling commands, which all target :8000.
    assert "Do **not** work around a busy port by picking another one" in block


def test_lifecycle_checks_liveness_before_health():
    """`kill -0` must precede the curl, not follow it.

    Ordering alone is necessary but not sufficient (the doomed new server
    is briefly alive while the stale one answers) — hence the ownership
    check below — but a health-first poll can never observe the exit at
    all, so the order is pinned too.
    """
    block = _norm(SERVER_LIFECYCLE)
    liveness = block.index('kill -0 "$PID"')
    health = block.index("curl -fsS http://127.0.0.1:8000/health >/dev/null")
    assert liveness < health, "the readiness poll must check liveness before /health"


def test_lifecycle_verifies_the_listener_belongs_to_our_process_group():
    """READY requires the :8000 listener to be in the recorded PID's group.

    `setsid` makes the server a process-group leader (PGID == PID) and its
    workers inherit that PGID, so comparing the listener's PGID to the
    recorded PID is an exact identity check — a foreign server fails it.
    """
    block = _norm(SERVER_LIFECYCLE)
    assert "owns_port" in block
    assert "ps -o pgid=" in block
    assert "not owned by PID" in block
    # An unresolvable owner must fail closed, never be assumed to be ours.
    assert "could not resolve — do NOT assume ours" in block
    assert 'treat "unverified" as "not ours"' in block


def test_lifecycle_confirms_the_port_freed_after_teardown():
    block = _norm(SERVER_LIFECYCLE)
    assert "that :8000 is free again" in block
