# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime map, held to the tree it maps.

The index is only worth consulting if a wrong entry is a loud failure. These
checks are what make that true, and they are in two groups:

* **the pointers resolve** — every path and every symbol, against the actual
  checkout. This is the whole premise: a map of paths survives a refactor
  only while someone checks that it did.
* **the map stays a map** — no code, no signatures, no line numbers. That
  restraint is the design, and a rule nothing enforces is a rule that decays.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import (
    ALWAYS_RELEVANT,
    INDEX_PATH,
    RuntimeArea,
    RuntimeIndexError,
    areas_for_demands,
    format_areas,
    load_index,
    unmatched_demands,
)
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import read_checkpoint_surface

REPO_ROOT = Path(__file__).resolve().parents[4]
in_checkout = pytest.mark.skipif(
    not (REPO_ROOT / "tensorrt_llm" / "_torch").is_dir(),
    reason="not inside a TensorRT-LLM checkout; nothing to resolve against",
)


@pytest.fixture(scope="module")
def index():
    return load_index()


# ------------------------------------------------------------- the pointers


@in_checkout
def test_every_path_resolves(index):
    """A pointer that has moved is a failure, not a stale sentence."""
    missing = [
        f"{area.name}: {path}"
        for area in index
        for path in area.pointers
        if not (REPO_ROOT / path).exists()
    ]
    assert not missing, "runtime_index.yaml points at paths that do not exist:\n  " + "\n  ".join(
        missing
    )


@in_checkout
def test_every_symbol_resolves(index):
    """The other half of "name the symbol, not the line".

    Searched inside the area's own `code` paths rather than the whole tree:
    a symbol that exists somewhere else entirely is not evidence that this
    entry points anywhere useful.
    """
    missing = []
    for area in index:
        haystack = ""
        for path in area.code:
            target = REPO_ROOT / path
            if target.is_dir():
                for source in target.rglob("*.py"):
                    haystack += source.read_text(encoding="utf-8", errors="ignore")
            elif target.is_file():
                haystack += target.read_text(encoding="utf-8", errors="ignore")
        for symbol in area.symbols:
            if symbol not in haystack:
                missing.append(f"{area.name}: {symbol} not found under {list(area.code)}")
    assert not missing, "runtime_index.yaml names symbols that do not resolve:\n  " + "\n  ".join(
        missing
    )


@in_checkout
def test_each_area_anchors_on_at_least_one_defined_symbol(index):
    """Substring presence is not enough.

    `test_every_symbol_resolves` concatenates the area's files and asks
    whether the name appears anywhere. An entry can pass that while pointing
    nowhere useful: its symbol may match only inside *comments* or as a
    substring of a different class, only on import lines, or only because
    files under the area import it from elsewhere.

    So every area must anchor on at least one symbol actually *defined*
    under its own `code` paths. Not all of them --- a config field is a
    legitimate symbol and is defined elsewhere --- but at least one, or the
    entry is not pointing at an implementation.
    """
    import re as _re

    unanchored = []
    for area in index:
        if not area.symbols:
            continue
        anchored = False
        for path in area.code:
            target = REPO_ROOT / path
            sources = (
                target.rglob("*.py") if target.is_dir() else [target] if target.is_file() else []
            )
            for source in sources:
                text = source.read_text(encoding="utf-8", errors="ignore")
                for symbol in area.symbols:
                    pattern = (
                        rf"^\s*(?:class|def)\s+{_re.escape(symbol)}\b|^{_re.escape(symbol)}\s*[:=]"
                    )
                    if _re.search(pattern, text, _re.M):
                        anchored = True
                        break
                if anchored:
                    break
            if anchored:
                break
        if not anchored:
            unanchored.append(
                f"{area.name}: none of {list(area.symbols)} is DEFINED under {list(area.code)}"
            )
    assert not unanchored, (
        "an area whose symbols only appear in comments or imports points nowhere:\n  "
        + "\n  ".join(unanchored)
    )


# --------------------------------------------------------- it stays a map


def test_the_index_carries_no_code(index):
    """No fences, no signatures. It says where to look, not what is there."""
    text = INDEX_PATH.read_text(encoding="utf-8")
    assert "```" not in text, "runtime_index.yaml contains a code block"
    # `def foo(` / `class Foo(` are the shapes that creep in first.
    offenders = re.findall(r"\b(?:def|class)\s+\w+\s*\(", text)
    assert not offenders, f"runtime_index.yaml is transcribing code: {offenders}"


def test_the_index_cites_no_line_numbers(index):
    """Same rule the catalog and the skills are held to."""
    from .test_skills import BARE_LINE_REFERENCE, LINE_CITATION

    text = INDEX_PATH.read_text(encoding="utf-8")
    offenders = LINE_CITATION.findall(text) + BARE_LINE_REFERENCE.findall(text)
    assert not offenders, f"runtime_index.yaml cites code by line: {offenders}"


@pytest.mark.parametrize("area_name", [a for a in load_index().areas])
def test_each_summary_stays_high_level(index, area_name):
    """A long summary is a summary that has started transcribing.

    The cap is arbitrary and the point is that there is one: an entry that
    needs more than a short paragraph is describing the code instead of
    locating it.
    """
    words = len(index.get(area_name).summary.split())
    assert 10 <= words <= 70, f"{area_name}: summary is {words} words"


# ----------------------------------------------------------------- schema


def test_an_unknown_field_is_rejected(tmp_path):
    bad = tmp_path / "i.yaml"
    bad.write_text("x:\n  summary: s\n  code: [tensorrt_llm]\n  wat: 1\n", encoding="utf-8")
    with pytest.raises(RuntimeIndexError, match="unknown field"):
        load_index(bad)


def test_a_missing_required_field_is_rejected(tmp_path):
    bad = tmp_path / "i.yaml"
    bad.write_text("x:\n  summary: s\n", encoding="utf-8")
    with pytest.raises(RuntimeIndexError, match="missing required field 'code'"):
        load_index(bad)


def test_an_empty_index_is_rejected(tmp_path):
    bad = tmp_path / "i.yaml"
    bad.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeIndexError):
        load_index(bad)


# ------------------------------------------------- the join to the survey


def _surface(**config):
    base = dict(
        architectures=["XForCausalLM"],
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )
    return read_checkpoint_surface(base | config).demand_names


def test_a_dense_model_is_not_handed_the_exotic_areas(index):
    """The reason this is a query and not a document.

    Loading mamba, KDA, mHC and disaggregation for a dense Llama costs
    context and offers the agent detail it has no business using.
    """
    names = {a.name for a in areas_for_demands(_surface(), index=index)}
    assert {"attention", "mlp", "normalization", "quantization"} <= names
    assert not names & {"mamba", "gated_deltanet", "kda", "moe", "mla", "speculation"}


def test_a_hybrid_recurrent_model_gets_the_recurrent_areas(index):
    names = {
        a.name
        for a in areas_for_demands(
            _surface(
                mamba_d_state=128, conv_kernel=4, layer_types=["linear_attention", "full_attention"]
            ),
            index=index,
        )
    }
    assert {"mamba", "gated_deltanet"} <= names
    assert "moe" not in names


def test_an_mla_moe_model_gets_those_and_not_the_recurrent_ones(index):
    names = {
        a.name
        for a in areas_for_demands(
            _surface(
                kv_lora_rank=512, q_lora_rank=1536, n_routed_experts=256, num_experts_per_tok=8
            ),
            index=index,
        )
    }
    assert {"mla", "moe"} <= names
    assert not names & {"mamba", "gated_deltanet"}


def test_the_always_relevant_areas_are_always_there(index):
    """A target always sits in a topology and is always captured or not."""
    names = {a.name for a in areas_for_demands([], index=index)}
    assert set(ALWAYS_RELEVANT) <= names
    for name in ALWAYS_RELEVANT:
        assert index.get(name) is not None, name


def test_areas_come_back_in_index_order(index):
    """Two runs on one checkpoint must read the same document."""
    demands = _surface(n_routed_experts=8, num_experts_per_tok=2)
    first = [a.name for a in areas_for_demands(demands, index=index)]
    assert first == [a.name for a in areas_for_demands(reversed(list(demands)), index=index)]


def test_every_demand_the_survey_can_emit_is_accounted_for(index):
    """The drift guard between `survey.py` and this index.

    A demand no area claims is not automatically wrong — some are genuinely
    informational. But it must be *visible*, because the alternative is a
    checkpoint feature that silently maps to no engine knowledge at all.
    """
    emitted = set()
    for config in (
        dict(kv_lora_rank=512, q_lora_rank=1536),
        dict(n_routed_experts=8, num_experts_per_tok=2, mlp_only_layers=[0]),
        dict(mamba_d_state=128, conv_kernel=4, layer_types=["a", "b"]),
        dict(sliding_window=4096, attn_logit_softcapping=50.0, sinks=4),
        dict(num_nextn_predict_layers=1, head_dim=256, use_qk_norm=True),
        dict(rope_scaling={"rope_type": "yarn"}, partial_rotary_factor=0.25),
        dict(quantization_config={"quant_algo": "NVFP4"}),
    ):
        emitted.update(_surface(**config))

    unclaimed = set(unmatched_demands(emitted, index=index))
    # Known-informational: these describe the stack rather than name an area.
    expected_unclaimed = {"stack.hybrid"}
    surprising = unclaimed - expected_unclaimed
    assert not surprising, (
        "survey.py emits demands no runtime area claims — either add the area "
        f"or mark the demand informational: {sorted(surprising)}"
    )


# --------------------------------------------------------------- rendering


def test_the_rendered_block_hands_over_pointers_not_content(index):
    block = format_areas(areas_for_demands(_surface(), index=index))
    assert "Read the source" in block
    assert "no line numbers" in block
    assert "tests/unittest/_torch" in block  # the test pointer is half the value


def test_unmatched_demands_are_surfaced_as_a_signal():
    block = format_areas(
        (RuntimeArea(name="x", summary="s", code=("tensorrt_llm",)),),
        unmatched=("something.nobody.claims",),
    )
    assert "Demands no area claims" in block
    assert "something.nobody.claims" in block


def test_an_empty_selection_renders_nothing():
    assert format_areas(()) == ""


# ------------------------------------------------- reaching the plan phase


def test_the_map_reaches_the_plan_drafter(tmp_path, monkeypatch):
    """A map nobody reads is a catalogue.

    And it must be *narrowed*: the drafter for a dense checkpoint should not
    be handed the recurrent areas.
    """
    import json

    import yaml

    from .test_delivery import _build_repo
    from .test_workflow import Script, _install, _workflow, passing_shell, unit_tasks, write_graph

    repo = _build_repo(tmp_path)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    checkpoint.joinpath("config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-5,
            }
        )
    )
    task = tmp_path / "task.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(checkpoint),
                "arch": "sm_103",
                "parallel": "tp1",
                "trtllm_repo_path": str(repo),
                "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
                "environment": {
                    "kind": "local",
                    "python": "/opt/venv/bin/python",
                    "env": {
                        "TMPDIR": "/scratch/t",
                        "CLAUDE_CODE_TMPDIR": "/scratch/t",
                    },
                },
                "accuracy": {
                    "benchmark": "gsm8k",
                    "reference_yaml": "r.yaml",
                    "model_key": "openai/gpt-oss-120b",
                    "tolerance": 5.0,
                },
            }
        ),
        encoding="utf-8",
    )

    def planner(turn, prompt):
        if turn == 1:
            core = {**unit_tasks(tmp_path / "ws")[0], "depends_on": ["survey"]}
            write_graph(prompt, [{"id": "survey", "kind": "survey", "goal": "map it"}, core])
            return {"decision": "GRAPH_UPDATED", "summary": "survey"}
        return {"decision": "ESCALATE", "summary": "stop here"}

    workflow = _workflow(tmp_path, shell_runner=passing_shell())
    calls = _install(monkeypatch, workflow, Script(None, planner=planner))
    with workflow:
        workflow.run(task)

    prompt = next(p for name, p in calls if name == "plan_drafter")
    assert "## Runtime map" in prompt
    assert "tensorrt_llm/_torch/attention" in prompt
    assert "tests/unittest/_torch/attention" in prompt
    # Narrowed, not a catalogue.
    assert "kimi_kda" not in prompt
    assert "modules/mamba" not in prompt


def test_the_three_writing_roles_are_told_the_map_protocol():
    """Only the roles that read the engine to write code need it.

    `SOURCE_BOUNDARY` binds plan_drafter, smith and assembler — the three
    that derive a target. A reviewer handed a "go read the engine" protocol
    is a reviewer that will go read the engine instead of the diff.
    """
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS

    for role in ("plan_drafter", "smith", "assembler"):
        text = getattr(DEFAULT_PROMPTS, role)
        assert "runtime map" in text, role
        assert "Take the location from" in text, role
    for role in ("reviewer", "plan_reviewer"):
        assert "runtime map" not in getattr(DEFAULT_PROMPTS, role), role


# ---------------------------------------------------------------- gating


def test_every_area_declares_how_it_is_proven(index):
    """Required, because "configured" is not "verified".

    A feature can be enabled and never witnessed. An area that cannot say how
    it is proven is an area that will be claimed on the strength of a gate
    that cannot see it.
    """
    from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import GATES

    for area in index:
        assert area.gating is not None, f"{area.name} declares no gating"
        assert area.gating.gate in GATES, f"{area.name}: gate={area.gating.gate!r}"
        assert len(area.gating.signal.split()) >= 5, f"{area.name}: signal is too thin"
        assert len(area.gating.blind.split()) >= 3, f"{area.name}: blind is too thin"


def test_the_areas_needing_a_witness_are_the_ones_we_know_are_unverified(index):
    """A cross-check against what the gates can see, not a tautology.

    The overlap scheduler, chunked prefill and block reuse pass every standard
    gate unwitnessed. They live in `scheduling` and `kv_cache`, and both must
    say `none` --- if either ever claims a standard gate covers it, that claim
    is the bug.
    """
    assert index.get("scheduling").gating.needs_a_witness
    assert index.get("kv_cache").gating.needs_a_witness
    # And the areas the gates genuinely do cover must not claim otherwise.
    assert index.get("attention").gating.gate == "accuracy"
    assert index.get("speculation").gating.gate == "acceptance"


def test_speculation_records_that_both_ordinary_gates_are_blind(index):
    """The single most expensive blind spot in the workflow."""
    blind = index.get("speculation").gating.blind
    assert "Boot and accuracy" in blind
    assert "distribution" in blind


def test_gating_reaches_the_rendered_block(index):
    """A blind spot nobody is shown is a blind spot nobody avoids."""
    block = format_areas(areas_for_demands(_surface(num_nextn_predict_layers=1), index=index))
    assert "**proven by**" in block
    assert "**blind spot**" in block
    assert "acceptance gate" in block


def test_a_bad_gate_name_is_rejected(tmp_path):
    bad = tmp_path / "i.yaml"
    bad.write_text(
        "x:\n  summary: s\n  code: [tensorrt_llm]\n"
        "  gating: {gate: sometimes, signal: a b c d e, blind: a b c}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeIndexError, match="must be one of"):
        load_index(bad)


def test_incomplete_gating_is_rejected(tmp_path):
    bad = tmp_path / "i.yaml"
    bad.write_text(
        "x:\n  summary: s\n  code: [tensorrt_llm]\n  gating: {gate: boot}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeIndexError, match="missing 'signal'"):
        load_index(bad)


# ------------------------------------------- the features must have a home


def test_every_feature_the_schema_accepts_has_a_runtime_area(index):
    """`features: [host_offload]` must resolve to somewhere an agent can read.

    A feature with no area, or one spelled differently from its area, leaves
    the agent nothing to read for it.
    """
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import FEATURE_SKILLS

    # The speculative family all live in one area, and `disagg` is the
    # feature-list spelling of the `disaggregation` area. Native KV storage
    # and encoder replay are pool and prefix-reuse questions, so `kv_cache`;
    # decoder replay truncates a sliding window, so `sparse_attention`.
    aliases = {
        "mtp": "speculation",
        "eagle3": "speculation",
        "dflash": "speculation",
        "dspark": "speculation",
        "disagg": "disaggregation",
        "kv_quant": "kv_cache",
        "encoder_replay": "kv_cache",
        "decoder_replay": "sparse_attention",
    }
    missing = [
        feature for feature in FEATURE_SKILLS if index.get(aliases.get(feature, feature)) is None
    ]
    assert not missing, f"features with no runtime area: {missing}"


# --------------------------------------------------------------- overview


def test_the_overview_names_only_areas_that_exist(index):
    """A flow stage pointing at a phantom area is a map with a dead end."""
    unknown = [
        f"{stage.stage}: {name}"
        for stage in index.overview.flow
        for name in stage.areas
        if index.get(name) is None
    ]
    assert not unknown, f"_overview.flow names areas that do not exist: {unknown}"


def test_every_area_is_placed_somewhere_in_the_flow(index):
    """An area in no stage is an area nobody knows who owns."""
    placed = {name for stage in index.overview.flow for name in stage.areas}
    orphans = sorted(set(index.areas) - placed)
    assert not orphans, f"areas absent from _overview.flow: {orphans}"


def test_the_forward_stage_is_where_a_target_lives(index):
    """The one stage a modeling_v2 target replaces."""
    forward = next(s for s in index.overview.flow if "forward" in s.stage)
    assert {"attention", "mlp", "normalization"} <= set(forward.areas)
    # The things a target must NOT try to do.
    assert "scheduling" not in forward.areas
    assert "kv_cache" not in forward.areas
    assert "speculation" not in forward.areas


def test_the_overview_leads_the_rendered_block(index):
    block = format_areas(areas_for_demands(_surface(), index=index), overview=index.overview)
    assert block.index("How the engine fits together") < block.index("## Runtime map")
    assert "THE TARGET" in block


def test_a_reserved_key_that_is_not_the_overview_is_rejected(tmp_path):
    bad = tmp_path / "i.yaml"
    bad.write_text("_wat:\n  a: 1\n", encoding="utf-8")
    with pytest.raises(RuntimeIndexError, match="unknown reserved key"):
        load_index(bad)


# ------------------------------------------------------------ jit/autotune


def test_jit_is_always_relevant(index):
    """Nothing in a config.json says "affected by tactic selection".

    Every model is, so it cannot hang off a demand.
    """
    from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import ALWAYS_RELEVANT

    assert "jit_autotune" in ALWAYS_RELEVANT
    assert "jit_autotune" in {a.name for a in areas_for_demands([], index=index)}


def test_jit_records_that_a_cold_receipt_describes_a_different_path(index):
    """The systemic consequence, stated where a smith taking a receipt sees it.

    A receipt taken in a cold process certifies tactics that serving --- which
    warms the autotuner during startup --- will not select.
    """
    blind = index.get("jit_autotune").gating.blind
    assert "cold" in blind and "receipt" in blind
    signal = index.get("jit_autotune").gating.signal
    assert "enable_autotuner" in signal
    assert "_run_autotuner_warmup" in signal
