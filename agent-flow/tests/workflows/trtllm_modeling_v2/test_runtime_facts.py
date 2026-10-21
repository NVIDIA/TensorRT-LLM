# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime facts: the engines of passed gates, per unit and member, in ``runtime_facts.yaml``."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import logit_agreement as la
from agent_flow.workflows.trtllm_modeling_v2.checks.gates import EngineAttestation
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run import runtime_facts as rf
from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

_PADDED = [1, 2, 4, 8, 16, 24, 32]


def _engine(
    batches=_PADDED,
    slots=(897,),
    *,
    captured=True,
    padded=True,
    kv=524288,
    free=12.5,
    spec=None,
    tp=1,
    ep=1,
    dp=False,
    leg="",
    reported=True,
) -> dict:
    """One engine's record, as a gate's ``GateResult.engine`` carries it."""
    config = {
        "cuda_graph": {"batch_sizes": list(batches), "enable_padding": padded},
        "speculative": spec,
        "tp": tp,
        "ep": ep,
        "attention_dp": dp,
        "max_batch_size": max(batches) if batches else 1,
        "max_num_tokens": 8192,
    }
    if captured:
        config["cuda_graph_captured"] = list(batches)
    return EngineAttestation(
        leg=leg,
        reported=reported,
        config=config if reported else {},
        recurrent_slots=tuple(slots),
        kv_max_tokens=kv if reported else None,
        devices=({"device": "0", "free_gib": free, "total_gib": 79.1},) if reported else (),
    ).record()


def _record(ws, unit="core", member="tp1", gate="boot", engines=None, marker="m1", current="m1"):
    return rf.record(
        ws,
        unit=unit,
        member=member,
        gate=gate,
        engines=engines if engines is not None else [_engine()],
        tree="v1-1-gabc",
        marker=marker,
        current=current,
    )


# ------------------------------------------------------------- one engine


def test_an_engine_record_gives_the_numbers_it_carries():
    numbers = rf.engine_numbers(
        _engine(slots=(3360,) * 4, spec={"algorithm": "MTP", "max_draft_len": 3}, tp=4, ep=4)
    )
    assert numbers == {
        "cuda_graph_batches": _PADDED,
        "cuda_graph_padded": True,
        "recurrent_slots": [3360] * 4,
        "kv_tokens": 524288,
        "free_gib": 12.5,
        "total_gib": 79.1,
        "speculative": "MTP depth 3",
        "tp": 4,
        "ep": 4,
        "max_batch_size": 32,
        "max_num_tokens": 8192,
        "attention_dp": False,
    }


def test_the_captured_batches_win_over_the_configured_ones():
    record = _engine()
    record["attested"]["cuda_graph_captured"] = [1, 2, 4]
    assert rf.engine_numbers(record)["cuda_graph_batches"] == [1, 2, 4]
    del record["attested"]["cuda_graph_captured"]
    assert rf.engine_numbers(record)["cuda_graph_batches"] == _PADDED


def test_an_unattested_engine_still_gives_its_pools_and_nothing_it_did_not_read():
    """An accuracy gate's engine prints no record; its bind lines still count."""
    assert rf.engine_numbers(_engine(reported=False, slots=(11061,))) == {
        "recurrent_slots": [11061]
    }
    assert rf.engine_numbers(_engine(reported=False, slots=())) == {}


# --------------------------------------------------------------- the file


def test_a_passed_gate_writes_the_members_facts_and_the_envelope(tmp_path):
    assert _record(tmp_path) == []
    _record(tmp_path, member="tep4", engines=[_engine(list(range(1, 33)), (3360,) * 4, tp=4, ep=4)])
    _record(
        tmp_path,
        unit="mtp",
        gate="acceptance",
        engines=[
            _engine([1, 2, 3], (700,), padded=False, spec={"algorithm": "MTP", "max_draft_len": 3})
        ],
    )
    data = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())
    assert set(data["members"]) == {"core@tp1", "core@tep4", "mtp@tp1"}
    facts = data["members"]["core@tp1"]["facts"]
    assert facts["cuda_graph_batches"] == _PADDED and facts["cuda_graph_padding"] == "padded"
    assert facts["recurrent_slots_per_rank"] == 897 and facts["engines"] == ["boot"]
    assert facts["stale"] is False and facts["tree"] == "v1-1-gabc"
    envelope = data["envelope"]
    assert envelope["recurrent_slots_per_rank"] == {"most": 3360, "at": "core@tep4"}
    assert envelope["cuda_graph_batches"] == list(range(1, 33))
    assert envelope["cuda_graph_unpadded_at"] == ["mtp@tp1"]
    assert envelope["speculative"] == ["MTP depth 3"]
    assert envelope["ep"] == {"most": 4, "at": "core@tep4"}
    assert envelope["stale"] == []


def test_a_members_facts_span_its_gates_and_legs(tmp_path):
    _record(tmp_path)
    _record(
        tmp_path,
        gate="envelope_boot",
        engines=[_engine([1, 2, 256], (1200,), leg="envelope"), _engine(leg="require")],
    )
    member = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())["members"]["core@tp1"]
    assert [(e["gate"], e.get("leg", "")) for e in member["engines"]] == [
        ("boot", ""),
        ("envelope_boot", "envelope"),
        ("envelope_boot", "require"),
    ]
    facts = member["facts"]
    assert facts["recurrent_slots_per_rank"] == 1200 and facts["max_batch_size"] == 256
    assert facts["cuda_graph_batches"] == sorted({*_PADDED, 256})
    assert facts["engines"] == ["boot", "envelope_boot/envelope", "envelope_boot/require"]


def test_a_member_measured_only_on_an_earlier_target_is_marked_stale(tmp_path):
    _record(tmp_path, member="tp1", marker="m1", current="m1")
    _record(tmp_path, member="tp2", marker="m2", current="m2")
    data = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())
    assert data["marker"] == "m2"
    assert data["members"]["core@tp1"]["facts"]["stale"] is True
    assert data["members"]["core@tp2"]["facts"]["stale"] is False
    assert data["envelope"]["stale"] == ["core@tp1"]
    # The gate measured the target before an integration moved it: its own engine is stale.
    _record(tmp_path, member="tp2", gate="logits", marker="m2", current="m3")
    data = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())
    assert data["members"]["core@tp2"]["facts"]["stale"] is True


def test_the_engines_on_the_current_target_are_the_members_facts(tmp_path):
    _record(tmp_path, gate="envelope_boot", engines=[_engine([1, 256], (1200,))], marker="m1")
    _record(tmp_path, gate="boot", marker="m2", current="m2")
    facts = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())["members"]["core@tp1"][
        "facts"
    ]
    assert facts["engines"] == ["boot"] and facts["recurrent_slots_per_rank"] == 897


def test_a_change_is_a_gates_own_numbers_moving_and_free_memory_is_not_one(tmp_path):
    _record(tmp_path)
    assert _record(tmp_path, engines=[_engine(free=3.0, kv=100)]) == []
    changes = _record(tmp_path, engines=[_engine([*_PADDED, 64], (1200,), padded=False)])
    assert changes == [
        "core:boot@tp1: CUDA-graph batches [1, 2, 4, 8, 16, 24, 32] -> [1, 2, 4, 8, 16, 24, 32, 64]; "
        "CUDA-graph padding padded -> unpadded; recurrent slots per rank 897 -> 1200; "
        "max_batch_size 32 -> 64"
    ]
    # Another gate's first pass is not a change, whatever the first one measured.
    assert _record(tmp_path, gate="acceptance", engines=[_engine(slots=(5,))]) == []
    data = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())
    [kept] = data["changes"]
    assert kept.endswith(changes[0])


def test_the_change_history_is_bounded(tmp_path):
    for slots in range(1, rf.MAX_CHANGES + 6):
        _record(tmp_path, engines=[_engine(slots=(slots,))])
    data = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())
    assert len(data["changes"]) == rf.MAX_CHANGES
    assert data["changes"][-1].endswith(f"{rf.MAX_CHANGES + 4} -> {rf.MAX_CHANGES + 5}")


def test_a_gate_with_no_numbers_writes_nothing(tmp_path):
    assert _record(tmp_path, engines=[_engine(reported=False, slots=())]) == []
    assert not (tmp_path / rf.FACTS_FILENAME).exists()


def test_a_file_that_cannot_be_read_is_started_again(tmp_path):
    (tmp_path / rf.FACTS_FILENAME).write_text("members: [not, a, mapping]\n")
    _record(tmp_path)
    assert set(yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())["members"]) == {
        "core@tp1"
    }


def test_number_lists_are_written_on_one_line(tmp_path):
    _record(tmp_path)
    text = (tmp_path / rf.FACTS_FILENAME).read_text()
    assert "cuda_graph_batches: [1, 2, 4, 8, 16, 24, 32]" in text


# ------------------------------------------------------------- the prompt


def test_without_facts_a_prompt_carries_no_block(tmp_path):
    assert rf.render(tmp_path) == ""


def test_the_block_names_the_file_the_envelope_and_each_member(tmp_path):
    _record(tmp_path)
    _record(
        tmp_path,
        unit="mtp",
        gate="acceptance",
        engines=[
            _engine([1, 2, 3], (700,), padded=False, spec={"algorithm": "MTP", "max_draft_len": 3})
        ],
        marker="m2",
        current="m2",
    )
    block = rf.render(tmp_path, unit="mtp")
    assert block.startswith(rf.HEADING)
    assert f"`{tmp_path / rf.FACTS_FILENAME}`" in block
    assert "Size and certify against these measured values, not estimates." in block
    assert "1 of 2 members were last measured on an earlier state of the target." in block
    assert "- CUDA-graph batches captured: 8 sizes [1-4, 8, 16, 24, 32]; unpadded at mtp@tp1" in (
        block
    )
    assert "- recurrent slots per rank: up to 897 (core@tp1)" in block
    assert "- least memory free once up: 12.50 of 79.10 GiB (core@tp1)" in block
    members = block.split("Per member:")[1].strip().splitlines()
    assert members[0].startswith("- mtp@tp1: 3 CUDA-graph batches [1-3], unpadded;")
    assert "speculation MTP depth 3" in members[0]
    assert (
        members[1].startswith("- core@tp1: 7 CUDA-graph batches")
        and "no speculation" in (members[1])
    )
    assert members[1].endswith("STALE: measured on an earlier state of the target")


def test_the_block_is_bounded_and_lists_changes_only_when_asked(tmp_path):
    for n in range(1, 5):
        _record(tmp_path, member=f"tp{n}")
    _record(tmp_path, member="tp1", engines=[_engine(slots=(1200,))])
    block = rf.render(tmp_path, members=2)
    assert "- and 2 more members in the file" in block
    assert "Measured changes" not in block
    planner = rf.render(tmp_path, changes=5)
    assert "Measured changes, newest last:" in planner
    assert "core:boot@tp1: recurrent slots per rank 897 -> 1200" in planner


# --------------------------------------------------------- the gate-pass hook


class _Stage(PlannedStage):
    """What a contract check uses of the workflow, with a scripted gate and target."""

    def __init__(self, workspace, markers):
        self.workspace = workspace
        self.markers = list(markers)
        self.events: list[tuple[str, str]] = []
        self.gate_log = SimpleNamespace(record=lambda result: None)
        self.journal = SimpleNamespace(event=lambda kind, text: self.events.append((kind, text)))

    def _target_marker(self):
        return self.markers.pop(0) if len(self.markers) > 1 else self.markers[0]

    def _spec_for(self, unit):
        return {"accuracy": {}}, []

    def _member_env(self, member):
        return "env"

    def _member_runner(self, member):
        return "runner"

    def _gate_until_verdict(self, run, log):
        return run()


def _verdict(passed=True, engines=None):
    return SimpleNamespace(
        name=la.GATE_NAME,
        member="tp1",
        passed=passed,
        tree_state="v1-2-gdef",
        engine=tuple(engines if engines is not None else [_engine()]),
        summary=lambda: "[PASS] logits@tp1",
    )


def _check():
    return contract_mod.Check(unit="core", gate=la.GATE_NAME, member="tp1", tier="inner")


def test_a_passed_check_records_its_engines_under_the_target_it_started_on(tmp_path, monkeypatch):
    monkeypatch.setattr(la, "run_logit_gate", lambda env, **kw: _verdict())
    stage = _Stage(tmp_path, ["m1", "m2"])
    assert stage._run_contract_check(_check(), log=None).passed
    data = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())
    [engine] = data["members"]["core@tp1"]["engines"]
    assert engine["gate"] == la.GATE_NAME and engine["marker"] == "m1"
    assert engine["tree"] == "v1-2-gdef"
    # The target moved while the gate ran: what it measured is not the target now.
    assert data["marker"] == "m2" and data["members"]["core@tp1"]["facts"]["stale"] is True


def test_a_failed_check_or_one_with_no_engine_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(la, "run_logit_gate", lambda env, **kw: _verdict(passed=False))
    _Stage(tmp_path, ["m1"])._run_contract_check(_check(), log=None)
    monkeypatch.setattr(la, "run_logit_gate", lambda env, **kw: _verdict(engines=[]))
    _Stage(tmp_path, ["m1"])._run_contract_check(_check(), log=None)
    assert not (tmp_path / rf.FACTS_FILENAME).exists()


def test_a_measured_change_is_journaled(tmp_path, monkeypatch):
    stage = _Stage(tmp_path, ["m1"])
    monkeypatch.setattr(la, "run_logit_gate", lambda env, **kw: _verdict())
    stage._run_contract_check(_check(), log=None)
    moved = _verdict(engines=[_engine(slots=(1200,))])
    monkeypatch.setattr(la, "run_logit_gate", lambda env, **kw: moved)
    stage._run_contract_check(_check(), log=None)
    assert (
        "runtime facts changed",
        f"core:{la.GATE_NAME}@tp1: recurrent slots per rank 897 -> 1200",
    ) in stage.events


def test_a_witness_check_records_its_engines_too(tmp_path):
    class Witnessed(_Stage):
        def _run_witness(self, unit, member, log):
            return _verdict(engines=[_engine(leg="A"), _engine(leg="B")])

    check = contract_mod.Check(unit="core", gate="equivalence", member="tp1", tier="proof")
    Witnessed(tmp_path, ["m1"])._run_contract_check(check, log=None)
    member = yaml.safe_load((tmp_path / rf.FACTS_FILENAME).read_text())["members"]["core@tp1"]
    assert member["facts"]["engines"] == ["equivalence/A", "equivalence/B"]
