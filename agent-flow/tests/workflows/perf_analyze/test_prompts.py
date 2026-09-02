"""Tests that the role prompts carry the canonical command templates.

The perf-analyze serving roles must drive ``benchmark_serving.py`` and
``nsys profile`` from a fixed command template rather than improvising
flags per run, so these tests pin the flags that template guarantees.
"""

from __future__ import annotations

import json
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
    TRTLLM_TAXONOMY_PATH,
    profile_ranks_note,
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

# Canonical Run A2 flags: the utilization pass (``--gpu-metrics-*``) and the
# call-stack pass (backtraces). Both are separate captures from the timing
# pass, so a prompt that carries the flags but folds them into Run A would
# silently perturb every timing number the findings rest on.
_NSYS_UTILIZATION_FLAGS = (
    "--gpu-metrics-devices=all",
    "--gpu-metrics-frequency=100000",
)
_NSYS_CALL_STACK_FLAGS = (
    "--python-backtrace",
    "--python-sampling=true",
    "--cudabacktrace=kernel:5000,sync:10000",
)

# Canonical ``ncu`` flags the analyzer's Run B must carry.
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


def test_analyzer_prompt_has_run_a2_utilization_and_call_stack_flags():
    # Without ``--gpu-metrics-*`` a trace ranks kernels by cost and calls it
    # headroom; without backtraces every kernel is a mangled name with no
    # owner. Both passes are pinned so they cannot quietly fall out.
    for flag in _NSYS_UTILIZATION_FLAGS + _NSYS_CALL_STACK_FLAGS:
        assert flag in ANALYZER_SYSTEM_PROMPT, flag
    assert "## Run A2" in ANALYZER_SYSTEM_PROMPT


def test_run_a2_passes_are_separate_captures_from_the_timing_pass():
    # Metric sampling and backtraces perturb the timeline, so they must land
    # in their own captures — the timing numbers stay Run A's.
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "server_nsys_metrics" in prompt
    assert "server_nsys_stacks" in prompt
    assert "additional** captures" in prompt


def test_run_a2_degrades_gracefully_instead_of_fabricating():
    # ERR_NVGPUCTRPERM is the expected portability failure of the metrics
    # pass; the prompt must send the agent on rather than into a permission
    # fight, and must never let it invent the numbers.
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "ERR_NVGPUCTRPERM" in prompt
    assert "NVreg_RestrictProfilingToAdminUsers" in prompt
    assert "additive, never blocking" in prompt


def test_run_a2_call_stack_pass_states_the_cuda_graph_limit():
    # One cudaGraphLaunch covers thousands of kernels, so a backtrace on it
    # names the launch site and not the operator inside the graph. A prompt
    # that omits this invites graph-launch stacks reported as call sites.
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "cudaGraphLaunch" in prompt
    assert "graphId IS NOT NULL" in prompt


def test_findings_contract_carries_the_run_a2_evidence():
    # The evidence lands inside the existing ``nsys timeline`` section, so
    # every "Profiling setup / nsys timeline / ..." enumeration elsewhere
    # stays valid.
    contract = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "gpu metrics unavailable" in contract
    assert "call stacks unavailable" in contract
    assert "bounding resource" in contract


def test_analyzer_keeps_capture_range_end_stop_safety_flag():
    # The template omits it, but the automated run must keep it so nsys
    # does not SIGTERM the server at the window's end (default
    # ``stop-shutdown``), which crashes the engine and yields no report.
    assert "--capture-range-end=stop" in ANALYZER_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# nsys timeline decomposition (Run A step 5): the analyzer does not read the
# trace by hand — it exports a .sqlite and runs the perf-nsight-system-analysis
# skill's pipeline, whose vocabulary the findings must then use.
# --------------------------------------------------------------------------- #


def test_analyzer_loads_nsys_analysis_skill_and_degrades():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # Named with the fully-qualified fallback for plugin-namespaced installs,
    # exactly as the ncu methodology skill is.
    assert "perf-nsight-system-analysis" in prompt
    assert "trtllm-agent-toolkit:perf-nsight-system-analysis" in prompt
    # It is proactive: the trace is already captured, so the pipeline costs
    # no extra server launch and runs whenever nsys runs.
    assert "costs no extra server launch" in prompt
    assert "without waiting to be asked" in prompt
    # A missing skill or a failed pipeline never blocks the run and never
    # yields a fabricated split — the section degrades to a one-liner.
    assert "timeline analysis unavailable" in prompt


def test_nsys_analysis_pipeline_is_driven_from_the_sqlite_export():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The pipeline reads a .sqlite, not the .nsys-rep next to it.
    assert "nsys export --type sqlite" in prompt
    assert "scripts/run_all.py" in prompt
    assert "--taxonomy" in prompt
    assert "--out <workspace>/nsys_analysis" in prompt


def test_nsys_analysis_resolves_the_skills_ask_the_user_steps_autonomously():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The skill tells a human operator to pick representatives and anchors;
    # a campaign has nobody to ask, so the prompt resolves both itself.
    assert "no user to ask" in prompt
    assert "--representative" in prompt
    assert "--anchor" in prompt


def test_findings_contract_carries_the_timeline_decomposition():
    block = _norm(PROFILE_FINDINGS_CONTRACT)
    # The skill's vocabulary, verbatim — "compute-absent", never
    # "compute idle", and the three causes it splits into.
    assert "compute-absent" in block
    assert 'never "compute idle"' in block
    for bucket in ("launch-starved", "blocking", "dependency-stalled"):
        assert bucket in block, bucket
    # Iteration counts back every table, and the section degrades honestly.
    assert "`n=` iterations" in block
    assert "timeline analysis unavailable" in block


# --------------------------------------------------------------------------- #
# ncu deep dive (Run B): a bounded per-kernel profile of the top nsys kernels
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


def test_remote_execution_prompt_is_short_and_task_specific():
    task = {
        "checkpoint_path": "/models/gemma",
        "slurm-environment": {
            "slurm_partition": "batch",
            "docker_image": "/images/trtllm.sqsh",
            "cluster_ssh": "user@login",
            "remote_run_root": "/scratch/runs/gemma-analyze",
            "account": "acct",
            "qos": "short",
        },
    }
    bundle = build_perf_analyze_prompts(
        include_slurm_environment=True,
        remote_execution=task,
        campaign_name="gemma-analyze",
    )
    for role in ("benchmarker", "projector", "analyzer"):
        prompt = getattr(bundle, role)
        compact = _norm(prompt)
        assert "SSH target: user@login" in prompt
        assert "Remote run root: /scratch/runs/gemma-analyze" in prompt
        assert "Container image: /images/trtllm.sqsh" in prompt
        assert "Model checkpoint: /models/gemma" in prompt
        assert "partition=batch, account=acct, qos=short" in prompt
        assert "rsync the required inputs and changed source" in compact
        assert "Pull the role's required outputs and failure logs back" in compact
    assert "SSH target:" not in bundle.reporter


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


# --------------------------------------------------------------------------- #
# The nsys-timeline pipeline's own artifacts: the taxonomy it classifies with,
# and the products the findings (and, downstream, the roadmap) are built from.
# Running `run_all.py` is not the same as consuming what it wrote.
# --------------------------------------------------------------------------- #


def test_timeline_pipeline_uses_the_trtllm_taxonomy_not_the_skill_template():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The skill's template is shaped for training frameworks and matches
    # almost nothing in a decode step; copying it verbatim leaves the hot
    # kernels uncategorized and the Step 5 mode decision meaningless.
    assert f"cp {TRTLLM_TAXONOMY_PATH}" in prompt
    # Named only to be ruled out as the source, never as the thing to copy.
    assert "not the skill's `references/taxonomy_template.json`" in prompt
    assert "shaped for training frameworks" in prompt


def test_trtllm_taxonomy_asset_is_loadable_and_keeps_the_reserved_categories():
    data = json.loads(TRTLLM_TAXONOMY_PATH.read_text(encoding="utf-8"))
    overall = data["Overall"]
    # `gemm` / `mha` are the pipeline's hard-coded Step 4 anchors and
    # `nccl` its collective class; renaming one silently empties a view.
    assert {"gemm", "mha", "nccl"} <= set(overall)
    # The exact-name overlay the correlation would merge into.
    assert data["ExactNames"] == {}
    for category, pattern in overall.items():
        re.compile(pattern)  # a broken regex fails the whole pipeline
        assert pattern.strip(), category


def test_trtllm_taxonomy_classifies_real_trtllm_kernel_names():
    overall = json.loads(TRTLLM_TAXONOMY_PATH.read_text(encoding="utf-8"))["Overall"]
    compiled = [(name, re.compile(rx)) for name, rx in overall.items()]

    def classify(kernel: str) -> str:
        return next((name for name, rx in compiled if rx.search(kernel)), "uncategorized")

    # First-match-wins ordering has to survive edits: a collective that
    # carries `moe` in its name is comm, and a grouped MoE matmul is a
    # GEMM rather than MoE plumbing.
    assert classify("ncclDevKernel_AllReduce_Sum_bf16_RING_LL") == "nccl"
    assert classify("moeA2ADispatchKernel") == "nccl"
    assert classify("nvjet_tst_128x128_64x6_1x1_h_bz_coopA") == "gemm"
    assert classify("void tensorrt_llm::kernels::moe_gemm::moeGemmKernel") == "gemm"
    assert classify("fmha_v2_flash_attention_fp16_128_64_sm90_kernel") == "mha"
    assert classify("xqa_kernel_dt_fp16_d128") == "mha"
    assert classify("flashinfer::BatchDecodeWithPagedKVCacheKernel") == "mha"
    assert classify("void tensorrt_llm::kernels::generalRmsNorm") == "norm"
    assert classify("finalizeMoeRoutingKernel") == "moe"
    assert classify("applyBiasRopeUpdateKVCache") == "rope"
    assert classify("triton_poi_fused_add_mul_0") == "triton"


def test_analyzer_verifies_the_taxonomy_before_quoting_a_category():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "Verify the taxonomy before quoting a single category number" in prompt
    # The three buckets that say how much of the classification is real.
    assert "classified_by" in prompt
    assert "uncategorized_above_threshold" in prompt
    # Iterating is free — the pipeline re-reads files, it does not re-profile.
    assert "re-reads files only, no server, no GPU" in prompt
    # The reserved names are load-bearing, not stylistic.
    assert "hard-codes them as the Step 4 anchors" in prompt


def test_analyzer_reads_the_per_op_breakdown_and_its_mode():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # Step 5 decides between two shapes and writes a different file each
    # way; the busy rungs size the prize, this names the target.
    assert "fused_share_of_residual_pct" in prompt
    assert "module_slicing_recommended" in prompt
    assert "opgroup.json" in prompt
    assert "module_slice.json" in prompt
    assert "window_labels" in prompt
    assert "names *what to optimize*" in prompt


def test_analyzer_authors_the_skills_items_json_handoff():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The skill's machine-readable opportunity list, authored from its own
    # numbers — the artifact downstream stages key coverage on.
    assert "items.json" in prompt
    assert "magnitudeMs" in prompt
    assert "boundingResource" in prompt
    assert 'Write `{"items": []}` when the analysis genuinely found nothing' in prompt
    # An opportunity omitted reads as one that does not exist.
    assert "reads as one that does not exist" in prompt


def test_analyzer_reconciles_the_pipelines_own_invariants():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "iter_ms ≈ device_busy_ms + device_idle_ms" in prompt
    assert "launch_starved + blocking + dependency_stalled ≈ compute_absent" in prompt
    # The usual cause, named so it can be found rather than rounded away.
    assert "a sum was used where a union belongs" in prompt


def test_ncu_targets_come_from_the_decomposition_not_the_kernel_sum():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # ~40 launches per server relaunch makes this the round's most
    # expensive choice; kern_sum sums overlapping streams over the whole
    # capture, the decomposition is a union clipped to the iteration.
    assert "Pick the targets from the timeline decomposition, not the kernel sum" in prompt
    assert "matched_kernels" in prompt
    assert "sum across overlapping streams over the whole capture" in prompt
    # Still the honest fallback when the pipeline could not run.
    assert "if nsys ran but the skill's pipeline did not, rank on `kern_sum`" in prompt


def test_findings_contract_owes_the_classification_and_the_step5_mode():
    contract = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "a category number whose taxonomy was not verified is not a finding" in contract
    assert "which Step 5 mode ran" in contract
    # Both reconciliations are reported as pass/fail, not assumed.
    assert "stated as pass/fail" in contract


# --------------------------------------------------------------------------- #
# Multi-rank capture and the rank-jitter step. Where the nsys wrap goes decides
# whether a multi-GPU run yields any model kernels at all, and several ranks
# are what buy the straggler verdict.
# --------------------------------------------------------------------------- #


def test_multi_gpu_names_the_spawn_limit_rather_than_the_wrong_env_var():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The real constraint: a bare trtllm-serve spawns its workers, and nsys
    # does not follow spawned processes.
    assert "MPI.COMM_SELF.Spawn" in prompt
    assert "does not follow them" in prompt
    # TLLM_PROFILE_LOG_RANKS selects which ranks print the step log; it is
    # not a capture knob, and treating it as one wastes an allocation.
    assert "selects which ranks print the step log line" in prompt
    # The window needs no per-rank handling — every rank arms on the same
    # iteration counter.
    assert "arms `cudaProfilerStart/Stop` on the same iteration" in prompt


def test_per_rank_wrap_goes_inside_the_launcher_and_only_on_listed_ranks():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "the wrap goes **inside** the step, once per task" in prompt
    assert "SLURM_PROCID" in prompt
    assert "trtllm-llmapi-launch" in prompt
    # Wrapping every rank makes the straggler verdict describe nsys.
    assert "makes the straggler verdict describe nsys rather than the model" in prompt


def test_several_ranks_take_a_survey_pass_then_a_representative_pass():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    # The skill refuses to pick representatives; exit 2 on the first pass is
    # the expected outcome, not a failure to retry around.
    assert "no `--representative`" in prompt
    assert "stops with exit 2, which is the expected outcome" in prompt
    assert "--representative 0 --representative 4" in prompt
    assert "--part stage-0=0,1,2,3" in prompt
    # Parts come from measured fingerprint groups, not an assumed layout.
    assert "group index shared by ranks with identical kernel fingerprints" in prompt
    assert "rather than from a rank-layout convention you assumed" in prompt
    # Rank ids are the pairing key.
    assert "never renumbered" in prompt


def test_jitter_step_is_read_with_its_verdict():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "part-<name>/jitter.json" in prompt
    assert "mean_jitter_wait_ms_per_iter" in prompt
    assert "imbalance_operator" in prompt
    # The spread alone is not actionable — pinned and rotating need
    # opposite fixes.
    assert "always with `straggler.verdict` beside the spread, never without" in prompt
    assert "need opposite fixes" in prompt
    # Lateness needs a shared clock; a one-rank part has no jitter at all.
    assert "only with its `floor_ms` beside it" in prompt
    assert "A part holding one rank has a `null` `jitter_cost`" in prompt


def test_jitter_wait_dominant_comm_is_read_as_imbalance():
    prompt = _norm(ANALYZER_SYSTEM_PROMPT)
    assert "is imbalance, not the network" in prompt
    assert "where the cost *appears*, not where it is *caused*" in prompt


def test_findings_contract_owes_the_straggler_verdict():
    contract = _norm(PROFILE_FINDINGS_CONTRACT)
    assert "Rank jitter" in contract
    assert "straggler verdict" in contract
    assert "the spread is never reported without it" in contract
    # One captured rank makes no imbalance claim at all.
    assert "make no imbalance claim" in contract


def test_profile_ranks_note_states_the_duty_for_each_shape():
    single = _norm(profile_ranks_note([0]))
    assert "rank 0 only" in single
    assert "rank-jitter step does not apply" in single

    several = _norm(profile_ranks_note([0, 4]))
    assert "ranks 0, 4" in several
    assert "one trace per rank" in several
    assert "only these ranks are wrapped" in several
    # Degrades where the topology cannot deliver per-rank traces.
    assert "spawn-launched `trtllm-serve` cannot" in several
    assert "make no imbalance claim" in several
