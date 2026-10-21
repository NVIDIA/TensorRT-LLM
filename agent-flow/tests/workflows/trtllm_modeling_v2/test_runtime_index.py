# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime map: the index loads, joins to the survey's demands, and reaches the surveyor."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import (
    RuntimeArea,
    areas_for_demands,
    format_areas,
    load_index,
    unmatched_demands,
)
from agent_flow.workflows.trtllm_modeling_v2.onboard.survey import read_checkpoint_surface


@pytest.fixture(scope="module")
def index():
    return load_index()


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
    names = {a.name for a in areas_for_demands(_surface(), index=index)}
    assert {"attention", "mlp", "normalization", "quantization"} <= names
    assert not names & {"mamba", "gated_deltanet", "kda", "moe", "mla", "speculation"}


def test_unmatched_demands_are_surfaced_as_a_signal(index):
    demands = ["attention.shape", "something.nobody.claims"]
    assert unmatched_demands(demands, index=index) == ("something.nobody.claims",)
    block = format_areas(
        (RuntimeArea(name="x", summary="s", code=("tensorrt_llm",)),),
        unmatched=("something.nobody.claims",),
    )
    assert "Demands no area claims" in block
    assert "something.nobody.claims" in block


def test_the_overview_leads_the_rendered_block(index):
    block = format_areas(areas_for_demands(_surface(), index=index), overview=index.overview)
    assert block.index("How the engine fits together") < block.index("## Runtime map")
    assert "THE TARGET" in block


def test_the_map_reaches_the_surveyor(tmp_path, monkeypatch):
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

    prompt = next(p for name, p in calls if name == "surveyor")
    assert "## Runtime map" in prompt
    assert "tensorrt_llm/_torch/attention" in prompt
    assert "tests/unittest/_torch/attention" in prompt
    # Narrowed, not a catalogue.
    assert "kimi_kda" not in prompt
    assert "modules/mamba" not in prompt
