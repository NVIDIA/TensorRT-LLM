# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""That the gates are REACHED — not that they work, which is test_gates.py's job.

This file exists because `run_acceptance_gate` had three passing unit tests
and no call site anywhere in the orchestrator. Full coverage of a function's
own behaviour says nothing about whether anything calls it, and a gate nobody
calls gates nothing. Everything here asserts reachability and argument
routing.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import gates, workflow
from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow


def _call_block(source: str, symbol: str, lines: int = 14) -> str:
    """The `lines` source lines starting at the call to `symbol`.

    Sliced by line count rather than by a literal closing-paren sequence.
    The gate calls moved one indentation level in when they were looped over
    the member set, and an anchor keyed on indentation asserts formatting
    where what is being checked is argument routing.
    """
    rows = source.splitlines()
    start = next(i for i, row in enumerate(rows) if symbol in row)
    return "\n".join(rows[start : start + lines])


# ------------------------------------------------------- reachability


def test_the_acceptance_gate_is_called_by_the_unit_loop():
    """The hole this file was written for.

    `grep -rn run_acceptance_gate` used to find a definition, an export and
    three tests — and no caller.
    """
    source = inspect.getsource(workflow.StaircaseOnboardWorkflow._assemble_one_unit)
    assert "run_acceptance_gate" in source


def test_core_is_not_acceptance_gated():
    """Core has no draft path, so the comparison would be 1.0 against 1.0.

    A gate that passes whatever the code did is worse than no gate: it
    appears in the evidence.
    """
    assert "core" not in workflow.ACCEPTANCE_GATED_UNITS


def test_every_speculative_feature_is_acceptance_gated():
    """The acceptance gate's whole membership is the distribution-preserving features.

    If a new speculator lands in FEATURE_SKILLS and not here, its unit gets
    boot and accuracy only — both of which are blind to a miscomputed draft
    path by construction.
    """
    from agent_flow.workflows.staircase_onboard.task_schema import FEATURE_SKILLS

    speculative = {
        name for name, skill in FEATURE_SKILLS.items() if skill == "staircase-modeling-speculator"
    }
    assert speculative <= workflow.ACCEPTANCE_GATED_UNITS


def test_the_accuracy_gate_is_called_with_the_unit():
    """Without it the mtp unit is measured under core's LLM-API YAML.

    `resolve_llm_api_options` exists precisely to prevent that and was being
    handed `unit=None` by the only caller in the package.
    """
    source = inspect.getsource(workflow.StaircaseOnboardWorkflow._assemble_one_unit)
    assert "unit=unit" in _call_block(source, "run_accuracy_gate")


def test_the_anchor_is_resolved_per_unit():
    source = inspect.getsource(workflow.StaircaseOnboardWorkflow._assemble_one_unit)
    assert "self._reference_score(unit)" in source


# ------------------------------------------------------- per-unit resolution


_ACC = {
    "benchmark": {"default": "gsm8k", "multimodal": "mmmu"},
    "reference_yaml": {"default": "refs/gsm8k.yaml", "multimodal": "refs/mmmu.yaml"},
    "llm_api_options": {"default": "/core.yaml", "mtp": "/mtp.yaml"},
    "discriminators": {
        "default": {"quant_algo": "MIXED_PRECISION"},
        "mtp": {"quant_algo": "MIXED_PRECISION", "spec_dec_algo": "MTP"},
    },
}


@pytest.mark.parametrize(
    "unit,expected",
    [("core", "gsm8k"), ("mtp", "gsm8k"), ("multimodal", "mmmu"), (None, "gsm8k")],
)
def test_benchmark_resolves_per_unit(unit, expected):
    assert gates.resolve_per_unit(_ACC, "benchmark", unit) == expected


def test_reference_yaml_resolves_per_unit():
    """A multimodal unit anchors in a different FILE, not just a different row."""
    assert gates.resolve_per_unit(_ACC, "reference_yaml", "multimodal") == "refs/mmmu.yaml"
    assert gates.resolve_per_unit(_ACC, "reference_yaml", "core") == "refs/gsm8k.yaml"


def test_a_scalar_still_means_the_whole_run():
    """Nothing that already worked may change shape."""
    assert gates.resolve_per_unit({"benchmark": "mmlu"}, "benchmark", "mtp") == "mmlu"


def test_discriminators_resolve_per_unit():
    assert gates.resolve_discriminators(_ACC, "mtp") == {
        "quant_algo": "MIXED_PRECISION",
        "spec_dec_algo": "MTP",
    }
    assert gates.resolve_discriminators(_ACC, "core") == {"quant_algo": "MIXED_PRECISION"}


def test_a_flat_discriminator_mapping_applies_to_every_unit():
    flat = {"discriminators": {"quant_algo": "NVFP4"}}
    assert gates.resolve_discriminators(flat, "mtp") == {"quant_algo": "NVFP4"}
    assert gates.resolve_discriminators(flat, None) == {"quant_algo": "NVFP4"}


def test_no_discriminators_means_none_rather_than_empty():
    """`{}` and `None` are different questions for load_reference_score.

    `{}` asks for the row with no tags at all; `None` means the caller did
    not express an opinion. Conflating them is how a MIXED_PRECISION
    checkpoint anchors to the bf16 number.
    """
    assert gates.resolve_discriminators({}, "core") is None


# ------------------------------------------------------- the anchor itself


def _reference_file(tmp_path):
    path = tmp_path / "gsm8k.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "Qwen/Model": [
                    {"accuracy": 95.72},
                    {"quant_algo": "MIXED_PRECISION", "accuracy": 96.36},
                    {"quant_algo": "MIXED_PRECISION", "spec_dec_algo": "MTP", "accuracy": 95.57},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_an_untagged_lookup_picks_the_bf16_row_which_is_why_unit_matters(tmp_path):
    """Documents the defect rather than the fix: this is what used to happen."""
    assert gates.load_reference_score(_reference_file(tmp_path), "Qwen/Model") == 95.72


def test_resolve_reference_uses_the_units_discriminators(tmp_path):
    task = {
        "trtllm_repo_path": str(tmp_path),
        "accuracy": {
            "reference_yaml": "gsm8k.yaml",
            "discriminators": _ACC["discriminators"],
        },
    }
    _reference_file(tmp_path)
    assert gates.resolve_reference(task, model_key="Qwen/Model", unit="core") == 96.36
    assert gates.resolve_reference(task, model_key="Qwen/Model", unit="mtp") == 95.57


def test_a_model_with_no_untagged_row_needs_discriminators(tmp_path):
    """The other half of the defect, and the more expensive half.

    `Qwen/Qwen3.8-27B` has only tagged rows, so an untagged lookup does not
    pick the wrong number — it raises, at the END of an assembly.
    """
    path = tmp_path / "r.yaml"
    path.write_text(
        yaml.safe_dump({"M": [{"quant_algo": "MIXED_PRECISION", "accuracy": 95.53}]}),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="no entry matching"):
        gates.load_reference_score(path, "M")
    assert gates.load_reference_score(path, "M", discriminators={"quant_algo": "MIXED_PRECISION"})


# ------------------------------------------- the re-draft has to say what was wrong


def test_a_schema_rejection_puts_the_validator_output_in_the_redraft_prompt(tmp_path):
    """Otherwise the drafter is told to read a review that never happened.

    A malformed `vocabulary.yaml` is rejected BEFORE the reviewer runs, so
    `progress.yaml` holds no reviewer entry. The re-draft prompt used to say
    "fetch the PlanReviewer's REJECT and address every item it names" — and
    there was nothing there. The drafter would re-emit the same file and, at
    the default one-round cap, end the run with `plan-failed`.
    """
    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "vocabulary.yaml").write_text(
        "verdict: assemblable\nsteps:\n- step: norm\n  state: GAP\n", encoding="utf-8"
    )
    flow = StaircaseOnboardWorkflow(workspace=workspace)
    prompt = flow._plan_drafter_prompt(2)

    assert "schema check" in prompt
    assert "PlanReviewer's REJECT" not in prompt
    # The validator's own words, not a paraphrase.
    assert "tier" in prompt


def test_a_first_round_prompt_is_not_a_redraft(tmp_path):
    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    flow = StaircaseOnboardWorkflow(workspace=workspace)
    assert "re-draft" not in flow._plan_drafter_prompt(1)


# ------------------------------------------------- concurrent smiths and devices


def _smith_workflow(tmp_path, gpus_per_node):
    import yaml as _yaml

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    flow = StaircaseOnboardWorkflow(workspace=workspace)
    flow.task_data = {
        "trtllm_repo_path": str(tmp_path),
        "arch": "sm_103",
        "environment": {"gpus_per_node": gpus_per_node},
    }
    flow._kernel_sources_block = ""
    del _yaml
    return flow


def _item(n):
    from agent_flow.workflows.staircase_onboard.vocabulary import QueueItem

    return QueueItem(n=n, mode="onboard", target="norm/x")


def test_concurrent_smiths_are_spread_across_the_nodes_devices(tmp_path):
    """The launch protocol hands every role `CUDA_VISIBLE_DEVICES=0`.

    At ten parallel smiths that is ten processes on one device, and the
    symptom is `Executor creation failed due to insufficient GPU memory` on a
    284 GB card — which reads as a target that allocates too much. Measured
    on this host with four concurrent loads.
    """
    flow = _smith_workflow(tmp_path, 4)
    assigned = {
        n: [
            line
            for line in flow._smith_prompt(_item(n)).splitlines()
            if "CUDA_VISIBLE_DEVICES" in line
        ]
        for n in range(1, 9)
    }
    got = [row[0].split("CUDA_VISIBLE_DEVICES=")[1].split("`")[0] for row in assigned.values()]
    assert got == ["1", "2", "3", "0", "1", "2", "3", "0"]


def test_a_single_device_node_says_nothing_extra(tmp_path):
    """No assignment to make; a contradictory instruction is worse than none."""
    flow = _smith_workflow(tmp_path, 1)
    assert "CUDA_VISIBLE_DEVICES" not in flow._smith_prompt(_item(3))


# ------------------------------------------- assembly turns and their decisions


def test_the_assembler_decision_path_tracks_the_turn_not_the_previous_phase(tmp_path):
    """`current_iteration` was last set by the SMITH phase, to a batch index.

    Turn 2 of an assembly was therefore told to write `assembler-1.yaml` —
    the path turn 1 had already used. Either the re-draft overwrites its
    predecessor's decision, or the role notices the contradiction and writes
    the right file anyway, which is what happened here.
    """
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    source = inspect.getsource(StaircaseOnboardWorkflow._assemble_one_unit)
    head = source[: source.index("self._dispatch(")]
    assert "self._progress_ctx.current_iteration = state.unit_turns" in head


def test_a_blocked_decision_from_a_previous_process_is_handled_before_redispatching(tmp_path):
    """A resume re-enters the loop at the top and never acts on it otherwise.

    The gap handler runs immediately after a dispatch, so a BLOCKED decision
    written by a process that then died is invisible: the resumed run
    dispatches the assembler again and pays a full turn to re-derive an
    answer already on disk. Measured — this run paid exactly that.
    """
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    source = inspect.getsource(StaircaseOnboardWorkflow._assemble_one_unit)
    head = source[: source.index("self._dispatch(")]
    assert "_handle_blocked_assembler" in head


def test_several_vocabulary_gaps_dispatch_several_smiths(tmp_path):
    """`missing_entry` is one string and an assembly turn finds several gaps.

    All three of this run's gaps were the same class, found in one pass —
    which is how assembly gaps come, because the turn sees the whole forward
    before it stops. Dispatching the concatenation sends one smith after a
    target that does not exist.
    """
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    source = inspect.getsource(StaircaseOnboardWorkflow._handle_blocked_assembler)
    assert "re.split" in source
    # Concurrently: gaps found in one turn have no dependency on each other.
    assert "ThreadPoolExecutor" in source


# ------------------------------------------------- the runtime block, per unit


def _task(runtime):
    return {"runtime": runtime, "trtllm_repo_path": "/repo", "parallel": "tp1"}


_PER_UNIT_RUNTIME = {
    "default": {
        "cuda_graph": True,
        "chunked_prefill": True,
        "overlap_scheduler": True,
        "kv_cache_reuse": True,
        "extra_flags": ["--max_seq_len=8192"],
    },
    "multimodal": {"kv_cache_reuse": False},
}


def test_a_flat_runtime_block_still_means_the_whole_run():
    flat = {"cuda_graph": True, "kv_cache_reuse": True, "extra_flags": []}
    assert gates.resolve_runtime(_task(flat), "multimodal") == flat


def test_a_per_unit_block_is_layered_over_default():
    """A unit states only what differs, so `default` supplies the rest."""
    merged = gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "multimodal")
    assert merged["kv_cache_reuse"] is False
    assert merged["cuda_graph"] is True
    assert merged["extra_flags"] == ["--max_seq_len=8192"]


def test_a_unit_with_no_block_of_its_own_gets_default():
    assert gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "core")["kv_cache_reuse"] is True


def test_the_units_really_do_disagree_in_the_emitted_flags():
    """This is the whole point, and it is not hypothetical.

    `core` and `mtp` were gated with block reuse ON and passed at 96.25 and
    96.10; the multimodal path requires it OFF. Block reuse is one of the two
    switches spelled as a NEGATION, so "on" emits nothing and "off" emits
    `--disable_kv_cache_reuse` — which is exactly the translation a reader
    gets backwards.
    """
    core = gates.runtime_flags(gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "core"))
    mm = gates.runtime_flags(gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "multimodal"))
    assert "--disable_kv_cache_reuse" not in core
    assert "--disable_kv_cache_reuse" in mm


def test_the_boot_gate_takes_a_unit_because_it_cannot_read_the_options_file():
    """`quickstart_advanced.py` has no `--extra_llm_api_options`.

    So the gate that runs first and most often is the only one that cannot
    see a unit's declared configuration through `llm_api_options`, and the
    runtime block is its sole channel.
    """
    import inspect

    assert "unit" in inspect.signature(gates.run_boot_gate).parameters
    source = inspect.getsource(gates.run_boot_gate)
    assert "resolve_runtime(task_data, unit)" in source

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    loop = inspect.getsource(StaircaseOnboardWorkflow._assemble_one_unit)
    assert "unit=unit" in _call_block(loop, "run_boot_gate")


def test_a_resume_only_auto_handles_a_vocabulary_gap():
    """Any other blocked reason already ended the run once.

    Handling it again just ends it again — measured: a `knowledge_gap` block
    written at a session cut-off re-closed the run within seconds of every
    resume, before the newly-extended session could dispatch anything. An
    operator who resumed deliberately has said the condition should be
    revisited, so the other reasons fall through to a fresh turn.
    """
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    source = inspect.getsource(StaircaseOnboardWorkflow._assemble_one_unit)
    head = source[: source.index("self._dispatch(")]
    assert 'prior_reason == "vocabulary_gap"' in head


# ------------------------------------------- the auditor's runtime-value dump


class _FakeGateLog:
    def __init__(self, results):
        self.results = results

    def render(self):
        return ""


class _Res:
    def __init__(self, name, stdout_tail):
        self.name = name
        self.stdout_tail = stdout_tail


class _DumpWorkflow:
    """Just enough of the orchestrator to exercise `_auditor_dump`."""

    RUNTIME_VALUE_MARKER = StaircaseOnboardWorkflow.RUNTIME_VALUE_MARKER
    _auditor_dump = StaircaseOnboardWorkflow._auditor_dump

    def __init__(self, results, spool=None):
        self.gate_log = _FakeGateLog(results)
        self._spool = spool

    def _gate_env(self):
        return SimpleNamespace(spool=self._spool)


_MARKER = StaircaseOnboardWorkflow.RUNTIME_VALUE_MARKER


def test_every_gate_with_a_dump_is_included_not_just_boot():
    """The boot gate drives two short prompts and records warm-up shapes.

    Both of this project's auditor findings were visible only in an accuracy
    gate's dump, so passing `boot.stdout_tail` alone -- which is what this did
    for its whole life -- hands the auditor the one dump that cannot show them.
    """
    wf = _DumpWorkflow(
        [
            _Res("boot", f"{_MARKER} ===\n  RV ('batch', 8192)\n"),
            _Res("accuracy", f"{_MARKER} ===\n  RV ('vision', 12544)\n"),
        ]
    )
    dump = wf._auditor_dump()
    assert "`boot` gate" in dump and "`accuracy` gate" in dump
    assert "('vision', 12544)" in dump


def test_a_gate_with_no_dump_contributes_nothing():
    """It falls through to the directions, which legitimately mention `boot`.

    So this asserts on the section header, not on the word: a gate whose log
    never carried a dump must not be rendered as though it had one.
    """
    wf = _DumpWorkflow([_Res("boot", "just some log output\n")])
    dump = wf._auditor_dump()
    assert "`boot` gate" not in dump
    assert "just some log output" not in dump


def test_with_no_dump_at_all_the_auditor_is_told_where_to_look():
    """`gate_log` is in-memory, so a run RESUMED into deliver has no history.

    That is the normal way to reach this stage after a repair, and it is
    exactly when the auditor most needs the dump. Handing it an empty section
    is what made two auditors recover the logs by hand.
    """
    wf = _DumpWorkflow([], spool="/spool/run")
    dump = wf._auditor_dump()
    assert "/spool/run/logs/*.log" in dump
    assert _MARKER in dump
    assert "union" in dump
    # and it must not read as "the gates did not run"
    assert "not a sign the gates did not run" in dump


def test_without_a_spool_it_still_names_the_marker():
    wf = _DumpWorkflow([], spool=None)
    dump = wf._auditor_dump()
    assert _MARKER in dump
    assert "gate logs" in dump


def test_the_directions_prefer_the_accuracy_gates():
    """A boot dump is not a substitute, and the prompt has to say why."""
    wf = _DumpWorkflow([], spool="/spool/run")
    assert "accuracy gates" in wf._auditor_dump()


# -------------------------------------------- adding a unit to a delivered run


class _GrowthWorkflow:
    """Just enough of the orchestrator to exercise `_grown_units`."""

    _grown_units = StaircaseOnboardWorkflow._grown_units

    def __init__(self, features):
        self.task_data = {"features": list(features)}


def test_a_spec_that_grew_names_the_new_units_in_order():
    wf = _GrowthWorkflow(["mtp", "multimodal", "host_offload", "disagg"])
    state = SimpleNamespace(units_done=["core", "mtp", "multimodal"])
    assert wf._grown_units(state) == ["host_offload", "disagg"]


def test_a_spec_that_did_not_grow_names_nothing():
    wf = _GrowthWorkflow(["mtp"])
    state = SimpleNamespace(units_done=["core", "mtp"])
    assert wf._grown_units(state) == []


def test_only_a_delivered_run_re_enters_on_growth():
    """A failed run is ALSO missing units from `units_done`.

    Re-entering there would re-dispatch paid agent turns with nothing having
    changed -- exactly what REMEDIABLE_VERDICTS exists to prevent. So the
    growth branch is guarded on the verdict being `delivered`, and this
    asserts the guard is in the source rather than implied by it.
    """
    src = inspect.getsource(StaircaseOnboardWorkflow._init_state)
    assert 'state.verdict == "delivered" and self._grown_units(state)' in src


def test_growth_re_enters_at_assemble_and_clears_the_audit():
    """The new unit has to be gated, and then re-delivered and re-audited.

    Leaving `audit_clean` set would carry the previous unit's audit over a
    target that has since changed.
    """
    src = inspect.getsource(StaircaseOnboardWorkflow._init_state)
    growth = src[src.index('state.verdict == "delivered"') :]
    assert "state.stage = STAGE_ASSEMBLE" in growth
    assert "state.audit_clean = False" in growth


# ------------------------------- decisions must travel on every transport


def test_an_external_transport_always_uses_decision_files():
    """The MCP tools need a session; an external transport has none.

    Without this the orchestrator neither tells the role where to write its
    decision nor collects one, and `_run_deliver_phase` reads `progress.yaml`
    and gets whatever was there before -- the PREVIOUS run's verdict.

    Measured: three audit passes in a row had their verdicts discarded this
    way, including a `CLEAN` that was read back as the `UNCOVERED` from two
    passes earlier. A loop like that cannot converge and looks exactly like a
    gate that keeps failing.
    """
    src = inspect.getsource(StaircaseOnboardWorkflow.__init__)
    line = next(ln for ln in src.splitlines() if "self._decision_files =" in ln)
    assert "agent_transport is not None" in line


def test_the_enterprise_mcp_reason_is_not_dropped():
    """The two reasons are independent; either alone requires decision files."""
    src = inspect.getsource(StaircaseOnboardWorkflow.__init__)
    line = next(ln for ln in src.splitlines() if "self._decision_files =" in ln)
    assert "ENTERPRISE_MCP_CONFIG.exists()" in line
    assert " or " in line


# ------------------------------------------- witnesses for the new feature kinds


def test_every_feature_whose_standing_gates_are_blind_has_a_witness():
    """A new witnessed feature in FEATURE_SKILLS and not here gets boot and accuracy only."""
    from agent_flow.workflows.staircase_onboard.task_schema import FEATURE_SKILLS

    witnessed_skills = {
        "staircase-modeling-host-offload",
        "staircase-modeling-disagg",
        "staircase-modeling-bounded-replay",
        "staircase-modeling-kv-quant",
    }
    needing = {name for name, skill in FEATURE_SKILLS.items() if skill in witnessed_skills}
    assert needing == set(workflow.WITNESS_GATED_UNITS)
    assert workflow.WITNESS_GATED_UNITS["decoder_replay"] == "replay"
    assert workflow.WITNESS_GATED_UNITS["encoder_replay"] == "replay"
    assert workflow.WITNESS_GATED_UNITS["kv_quant"] == "kv_quant"


def test_the_witness_dispatch_reaches_the_new_witnesses():
    source = inspect.getsource(StaircaseOnboardWorkflow._run_witness)
    assert "run_replay_witness" in source
    assert "run_kv_quant_witness" in source
    # The replay workload is declared, so it must travel from the block.
    for key in ("window", "prompt_tokens", "prefix_tokens", "reuse_alignment"):
        assert f'"{key}"' in source


@pytest.mark.parametrize(
    "unit,kind,block,gap",
    [
        ("decoder_replay", "replay", {"contrast_env": {"X": "0"}, "window": 128}, "prompt_tokens"),
        ("encoder_replay", "replay", {"contrast_env": {"X": "0"}, "prefix_tokens": 9}, "window"),
        ("kv_quant", "kv_quant", {"contrast_env": {"X": "bf16"}}, "expected_bytes_per_token"),
        ("kv_quant", "kv_quant", {"expected_bytes_per_token": 890}, "contrast_env"),
        ("host_offload", "residency", {"expected_mib": 1}, "contrast_env"),
        ("disagg", "equivalence", None, "a block at all"),
        # Without its own launcher leg B is leg A again (F021a).
        ("disagg", "equivalence", {"contrast_env": {"TRTLLM_DISAGG": "1"}}, "contrast_command"),
        ("disagg", "equivalence", {"contrast_command": "/leg_b.py"}, "contrast_env"),
    ],
)
def test_an_unconfigured_witness_names_what_is_missing(unit, kind, block, gap):
    assert gap in StaircaseOnboardWorkflow._witness_block_gap(unit, kind, block)


@pytest.mark.parametrize(
    "unit,kind,block",
    [
        ("decoder_replay", "replay", {"contrast_env": {"X": "0"}, "window": 1, "prompt_tokens": 9}),
        ("kv_quant", "kv_quant", {"expected_bytes_per_token": 890, "exact": False}),
        ("host_offload", "residency", {"contrast_env": {"X": "0"}}),
        ("disagg", "equivalence", {"contrast_env": {"X": "1"}, "contrast_command": "/b.py"}),
    ],
)
def test_a_configured_witness_has_no_gap(unit, kind, block):
    """A lossy kv_quant drops the identity leg, and with it the contrast."""
    assert StaircaseOnboardWorkflow._witness_block_gap(unit, kind, block) is None


def test_the_disagg_block_reaches_the_equivalence_witness(tmp_path, monkeypatch):
    """The declared workload, the per-leg timeout and a workdir leg B can read."""
    from agent_flow.workflows.staircase_onboard import witness
    from agent_flow.workflows.staircase_onboard.environment import RunEnvironment

    from .test_workflow import _workflow

    seen: dict = {}

    def fake(environment, **kwargs):
        seen.update(kwargs)
        return gates.GateResult(name="equivalence", passed=True, detail="ok")

    monkeypatch.setattr(witness, "run_equivalence_witness", fake)
    wf = _workflow(tmp_path)
    wf.task_data = {
        "features": ["disagg"],
        "accuracy": {
            "witness": {
                "disagg": {
                    "contrast_env": {"TRTLLM_DISAGG": 1},
                    "contrast_command": "/onboard/leg_b.py",
                    "prompt_tokens": [131, 2050],
                    "timeout": 2700,
                    "legs_env": {"TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY": 0},
                }
            }
        },
    }
    wf.environment = RunEnvironment(kind="local", python="/py", world_size=4)
    wf._gate_environment = wf.environment
    assert wf._run_witness("disagg", "tep4", None).passed
    assert seen["contrast_command"] == "/onboard/leg_b.py"
    assert seen["contrast_env"] == {"TRTLLM_DISAGG": "1"}
    assert seen["legs_env"] == {"TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY": "0"}
    assert seen["prompt_tokens"] == [131, 2050]
    assert seen["timeout"] == 2700
    assert seen["member"] == "tep4"
    assert seen["workdir"] == wf.workspace / "witness" / "disagg"


def test_a_witness_without_a_timeout_keeps_the_boot_budget(tmp_path, monkeypatch):
    from agent_flow.workflows.staircase_onboard import witness
    from agent_flow.workflows.staircase_onboard.environment import RunEnvironment

    from .test_workflow import _workflow

    seen: dict = {}

    def fake(environment, **kwargs):
        seen.update(kwargs)
        return gates.GateResult(name="residency", passed=True, detail="ok")

    monkeypatch.setattr(witness, "run_residency_witness", fake)
    wf = _workflow(tmp_path)
    wf.task_data = {
        "features": ["host_offload"],
        "accuracy": {"witness": {"host_offload": {"contrast_env": {"X": "0"}, "expected_mib": 1}}},
    }
    wf.environment = RunEnvironment(kind="local", python="/py", world_size=1)
    wf._gate_environment = wf.environment
    wf._run_witness("host_offload", "tp1", None)
    assert seen["timeout"] == gates.BOOT_TIMEOUT_S


# ------------------------------- the batch limit reaches every token witness


@pytest.mark.parametrize(
    "unit,function,block",
    [
        (
            "kv_quant",
            "run_kv_quant_witness",
            {"contrast_env": {"X": "bf16"}, "expected_bytes_per_token": 890},
        ),
        (
            "disagg",
            "run_equivalence_witness",
            {"contrast_env": {"X": "1"}, "contrast_command": "/b.py"},
        ),
        ("host_offload", "run_residency_witness", {"contrast_env": {"X": "0"}, "expected_mib": 1}),
    ],
)
@pytest.mark.parametrize("declared", [None, 1])
def test_the_batch_limit_travels_from_the_block(
    tmp_path, monkeypatch, unit, function, block, declared
):
    """F036: absent it is None, so a block written before the key runs as it did."""
    from agent_flow.workflows.staircase_onboard import witness
    from agent_flow.workflows.staircase_onboard.environment import RunEnvironment

    from .test_workflow import _workflow

    seen: dict = {}

    def fake(environment, **kwargs):
        seen.update(kwargs)
        return gates.GateResult(name=unit, passed=True, detail="ok")

    monkeypatch.setattr(witness, function, fake)
    wf = _workflow(tmp_path)
    if declared is not None:
        block = block | {"max_batch_size": declared}
    wf.task_data = {"features": [unit], "accuracy": {"witness": {unit: block}}}
    wf.environment = RunEnvironment(kind="local", python="/py", world_size=1)
    wf._gate_environment = wf.environment
    wf._run_witness(unit, "tp1", None)
    assert seen["max_batch_size"] == declared


def test_a_replay_witness_is_not_handed_a_batch_limit():
    """It counts work, not tokens, and takes no such argument: passing one would raise."""
    from agent_flow.workflows.staircase_onboard import witness

    source = inspect.getsource(StaircaseOnboardWorkflow._run_witness)
    assert "schedule" not in _call_block(source, "run_replay_witness", lines=12)
    assert "max_batch_size" not in inspect.signature(witness.run_replay_witness).parameters


def test_a_failed_witness_leaves_both_legs_in_the_journal(tmp_path, monkeypatch):
    """The one-line summary cannot carry the excerpt; the journal and console do.

    Generated text can hold what rich reads as markup (`[/INST]`), which
    must print as text rather than raise.
    """
    from agent_flow.workflows.staircase_onboard import witness
    from agent_flow.workflows.staircase_onboard.environment import RunEnvironment

    from .test_workflow import _workflow

    tail = "continuation #0: first differs at token 3\n    native: '[/INST] yes'\n    bf16  : 'no'"

    def fake(environment, **kwargs):
        return gates.GateResult(
            name="kv_quant", passed=False, detail="native storage changed '[/x]'", stdout_tail=tail
        )

    monkeypatch.setattr(witness, "run_kv_quant_witness", fake)
    wf = _workflow(tmp_path)
    wf.task_data = {
        "features": ["kv_quant"],
        "accuracy": {
            "witness": {
                "kv_quant": {"contrast_env": {"X": "bf16"}, "expected_bytes_per_token": 890}
            }
        },
    }
    wf.environment = RunEnvironment(kind="local", python="/py", world_size=1)
    wf._gate_environment = wf.environment
    assert not wf._run_witness("kv_quant", "tp1", None).passed
    journal = wf.journal.path.read_text(encoding="utf-8")
    assert "witness `kv_quant` failed: evidence" in journal
    assert "native: '[/INST] yes'" in journal
