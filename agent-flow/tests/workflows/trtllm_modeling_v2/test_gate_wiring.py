# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""That the gates are reached with the unit's own arguments."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import workflow
from agent_flow.workflows.trtllm_modeling_v2.run.execution import kernel_task as worker
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import ExecutionStage


def test_resolve_reference_uses_the_units_discriminators(tmp_path):
    task = {
        "trtllm_repo_path": str(tmp_path),
        "accuracy": {
            "reference_yaml": "gsm8k.yaml",
            "discriminators": {
                "default": {"quant_algo": "MIXED_PRECISION"},
                "mtp": {"quant_algo": "MIXED_PRECISION", "spec_dec_algo": "MTP"},
            },
        },
    }
    references = [
        {"accuracy": 95.72},
        {"quant_algo": "MIXED_PRECISION", "accuracy": 96.36},
        {"quant_algo": "MIXED_PRECISION", "spec_dec_algo": "MTP", "accuracy": 95.57},
    ]
    (tmp_path / "gsm8k.yaml").write_text(yaml.safe_dump({"Qwen/Model": references}))
    assert accuracy_checks.resolve_reference(task, model_key="Qwen/Model", unit="core") == 96.36
    assert accuracy_checks.resolve_reference(task, model_key="Qwen/Model", unit="mtp") == 95.57


def test_concurrent_kernel_writers_are_spread_across_the_nodes_devices(tmp_path):
    """The launch protocol hands every role `CUDA_VISIBLE_DEVICES=0`."""
    (tmp_path / "ws").mkdir()
    flow = workflow.TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    flow.task_data = {
        "trtllm_repo_path": str(tmp_path),
        "arch": "sm_103",
        "environment": {"gpus_per_node": 4},
    }
    got = []
    for n in range(1, 9):
        prompt = worker._kernel_writer_prompt(
            ExecutionStage(flow), QueueItem(n=n, mode="onboard", target="norm/x")
        )
        line = next(line for line in prompt.splitlines() if "CUDA_VISIBLE_DEVICES" in line)
        got.append(line.split("CUDA_VISIBLE_DEVICES=")[1].split("`")[0])
    assert got == ["1", "2", "3", "0", "1", "2", "3", "0"]


def test_the_units_really_do_disagree_in_the_api_configuration():
    runtime = {
        "default": {
            "cuda_graph": True,
            "chunked_prefill": True,
            "overlap_scheduler": True,
            "kv_cache_reuse": True,
            "extra_flags": ["--max_seq_len=8192"],
        },
        "multimodal": {"kv_cache_reuse": False},
    }
    task = {
        "runtime": runtime,
        "trtllm_repo_path": "/repo",
        "parallel": "tp1",
        "checkpoint_path": "/model",
    }

    def config(unit):
        return check_config.check_configuration(
            task,
            RunEnvironment(kind="local", python="python"),
            unit=unit,
            features=[],
            max_tokens=1,
            member="tp1",
        )["llm"]

    assert config("core")["kv_cache_config"]["enable_block_reuse"]
    assert not config("multimodal")["kv_cache_config"]["enable_block_reuse"]


def test_the_planner_starts_at_the_cheapest_member_one_engine_fits(tmp_path):
    import json

    from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c

    (tmp_path / "ws").mkdir()
    flow = workflow.TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    flow.task_data = {"trtllm_repo_path": str(tmp_path), "environment": {}}
    contract = c.Contract(
        checks=tuple(c.Check("core", "boot", m, c.TIER_INNER) for m in ("tp1", "tp2", "tp4"))
    )
    assert ExecutionStage(flow)._start_member(contract).startswith("`tp1`: the cheapest member")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    index = {"metadata": {"total_size": 344 * 2**30}}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    flow.task_data.update(checkpoint_path=str(checkpoint))
    flow.task_data["environment"]["gpu_memory_gib"] = 277
    # 344 GiB of weights: two 277-GiB devices hold one engine, one does not.
    assert ExecutionStage(flow)._start_member(contract).startswith("`tp2`: one engine")
