# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""That the gates are REACHED — not that they work, which is test_gates.py's job.

Full coverage of a gate function's own behaviour says nothing about whether
anything calls it, and a gate nobody calls gates nothing. Everything here
asserts reachability and argument routing.
"""

from __future__ import annotations

import inspect

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.run import workflow
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow


def _call_block(source: str, symbol: str, lines: int = 14) -> str:
    """The `lines` source lines starting at the call to `symbol`.

    Sliced by line count rather than by a literal closing-paren sequence: an
    anchor keyed on indentation asserts formatting, where what is being
    checked is argument routing.
    """
    rows = source.splitlines()
    start = next(i for i, row in enumerate(rows) if symbol in row)
    return "\n".join(rows[start : start + lines])


# ------------------------------------------------------- reachability


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
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import FEATURE_SKILLS

    speculative = {
        name for name, skill in FEATURE_SKILLS.items() if skill == "trtllm-modeling-v2-speculator"
    }
    assert speculative <= workflow.ACCEPTANCE_GATED_UNITS


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
    """A scalar value applies to every unit."""
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
    """An untagged lookup returns the bf16 row, wrong for a MIXED_PRECISION checkpoint."""
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
    """A model with only tagged rows makes an untagged lookup raise.

    It does not pick a wrong number; it fails, and at the END of an assembly,
    which is the most expensive place to find out.
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


# ------------------------------------------------- concurrent smiths and devices


def _smith_workflow(tmp_path, gpus_per_node, kind=None):
    import yaml as _yaml

    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workspace = tmp_path / "ws"
    workspace.mkdir()
    flow = TrtllmModelingV2Workflow(workspace=workspace)
    flow.task_data = {
        "trtllm_repo_path": str(tmp_path),
        "arch": "sm_103",
        "environment": {"gpus_per_node": gpus_per_node},
    }
    if kind is not None:
        flow.task_data["environment"]["kind"] = kind
    flow._kernel_sources_block = ""
    del _yaml
    return flow


def _item(n):
    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem

    return QueueItem(n=n, mode="onboard", target="norm/x")


def test_concurrent_smiths_are_spread_across_the_nodes_devices(tmp_path):
    """The launch protocol hands every role `CUDA_VISIBLE_DEVICES=0`.

    Parallel smiths would then share one device, and the symptom is
    `Executor creation failed due to insufficient GPU memory` even on a large
    card — which reads as a target that allocates too much. Each dispatch is
    therefore assigned a device of its own.
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


@pytest.mark.parametrize("kind", ["session_spool"])
def test_a_scheduler_owned_node_gets_no_device_assignment(tmp_path, kind):
    """The scheduler hands out the devices; an index here contradicts the protocol.

    A spool exports `CUDA_VISIBLE_DEVICES` per job and its protocol says never
    to set it; under Slurm the allocation is the claim, which an index would
    mask.
    """
    flow = _smith_workflow(tmp_path, 4, kind=kind)
    for n in range(1, 9):
        assert "CUDA_VISIBLE_DEVICES" not in flow._smith_prompt(_item(n))


def test_a_local_assignment_overrides_the_claimed_devices_not_device_0(tmp_path):
    """The local protocol names the devices preflight claimed, which need not be 0."""
    flow = _smith_workflow(tmp_path, 4, kind="local")
    prompt = flow._smith_prompt(_item(2))
    assert "**Use `CUDA_VISIBLE_DEVICES=2` for this dispatch**" in prompt
    assert "not the devices in the launch protocol above" in prompt


# ------------------------------------------- where the gates write their files


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
    """Per-unit runtime blocks exist so that units can disagree on flags.

    A text unit can gate with block reuse ON while the multimodal path
    requires it OFF. Block reuse is one of the two switches spelled as a
    NEGATION, so "on" emits nothing and "off" emits
    `--disable_kv_cache_reuse` — which is exactly the translation a reader
    gets backwards.
    """
    core = gates.runtime_flags(gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "core"))
    mm = gates.runtime_flags(gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "multimodal"))
    assert "--disable_kv_cache_reuse" not in core
    assert "--disable_kv_cache_reuse" in mm


# ------------------------------------------- witnesses for the new feature kinds


def test_every_feature_whose_standing_gates_are_blind_has_a_witness():
    """A new witnessed feature in FEATURE_SKILLS and not here gets boot and accuracy only."""
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import FEATURE_SKILLS

    witnessed_skills = {
        "trtllm-modeling-v2-host-offload",
        "trtllm-modeling-v2-disagg",
        "trtllm-modeling-v2-bounded-replay",
        "trtllm-modeling-v2-kv-quant",
    }
    needing = {name for name, skill in FEATURE_SKILLS.items() if skill in witnessed_skills}
    assert needing == set(workflow.WITNESS_GATED_UNITS)
    assert workflow.WITNESS_GATED_UNITS["decoder_replay"] == "replay"
    assert workflow.WITNESS_GATED_UNITS["encoder_replay"] == "replay"
    assert workflow.WITNESS_GATED_UNITS["kv_quant"] == "kv_quant"


def test_the_witness_dispatch_reaches_the_new_witnesses():
    """The unit loop and a standalone run share one dispatch: `witness.run_declared_witness`."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness

    assert "run_declared_witness" in inspect.getsource(TrtllmModelingV2Workflow._run_witness)
    assert "_run_declared_witness" in inspect.getsource(witness.run_declared_witness)
    source = inspect.getsource(witness._run_declared_witness)
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
        # The run's own launcher needs the environment that sets its legs apart.
        ("disagg", "equivalence", {"contrast_command": "/leg_b.py"}, "contrast_env"),
    ],
)
def test_an_unconfigured_witness_names_what_is_missing(unit, kind, block, gap):
    assert gap in TrtllmModelingV2Workflow._witness_block_gap(unit, kind, block)


@pytest.mark.parametrize(
    "unit,kind,block",
    [
        ("decoder_replay", "replay", {"contrast_env": {"X": "0"}, "window": 1, "prompt_tokens": 9}),
        ("kv_quant", "kv_quant", {"expected_bytes_per_token": 890, "exact": False}),
        ("host_offload", "residency", {"contrast_env": {"X": "0"}}),
        ("disagg", "equivalence", {"contrast_env": {"X": "1"}, "contrast_command": "/b.py"}),
        # Without a launcher of its own, leg B is the flow's (`disagg_leg_b.py`).
        ("disagg", "equivalence", {}),
        ("disagg", "equivalence", {"contrast_env": {"TRTLLM_DISAGG": "1"}}),
    ],
)
def test_a_configured_witness_has_no_gap(unit, kind, block):
    """A lossy kv_quant drops the identity leg, and with it the contrast."""
    assert TrtllmModelingV2Workflow._witness_block_gap(unit, kind, block) is None


def test_the_disagg_block_reaches_the_equivalence_witness(tmp_path, monkeypatch):
    """The declared workload, the per-leg timeout and a workdir leg B can read."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment

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
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment

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
    """Absent from the block it is None, leaving the runtime block's batch limit in force."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment

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
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness

    source = inspect.getsource(witness._run_declared_witness)
    assert "shaped" not in _call_block(source, "return run_replay_witness", lines=10)
    assert "max_batch_size" not in inspect.signature(witness.run_replay_witness).parameters


def test_a_failed_witness_leaves_both_legs_in_the_journal(tmp_path, monkeypatch):
    """The one-line summary cannot carry the excerpt; the journal and console do.

    Generated text can hold what rich reads as markup (`[/INST]`), which
    must print as text rather than raise.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment

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
