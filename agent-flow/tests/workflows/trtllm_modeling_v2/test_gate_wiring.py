# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""That the gates are reached with the unit's own arguments."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.run import workflow


def test_every_speculative_feature_is_acceptance_gated():
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import SPECULATORS

    assert SPECULATORS <= workflow.ACCEPTANCE_GATED_UNITS


def test_every_feature_whose_standing_gates_are_blind_has_a_witness():
    """A new witnessed feature in FEATURE_FAMILIES and not here gets boot and accuracy only."""
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import FEATURE_FAMILIES

    witnessed = {"host_offload", "disagg", "bounded_replay", "kv_quant"}
    needing = {name for name, family in FEATURE_FAMILIES.items() if family in witnessed}
    assert needing == set(workflow.WITNESS_GATED_UNITS)
    assert workflow.WITNESS_GATED_UNITS["decoder_replay"] == "replay"
    assert workflow.WITNESS_GATED_UNITS["encoder_replay"] == "replay"
    assert workflow.WITNESS_GATED_UNITS["kv_quant"] == "kv_quant"


_ACC = {
    "benchmark": {"default": "gsm8k", "multimodal": "mmmu"},
    "reference_yaml": {"default": "refs/gsm8k.yaml", "multimodal": "refs/mmmu.yaml"},
    "llm_api_options": {"default": "/core.yaml", "mtp": "/mtp.yaml"},
    "discriminators": {
        "default": {"quant_algo": "MIXED_PRECISION"},
        "mtp": {"quant_algo": "MIXED_PRECISION", "spec_dec_algo": "MTP"},
    },
}


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


def _kernel_writer_workflow(tmp_path, gpus_per_node):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    flow = workflow.TrtllmModelingV2Workflow(workspace=workspace)
    flow.task_data = {
        "trtllm_repo_path": str(tmp_path),
        "arch": "sm_103",
        "environment": {"gpus_per_node": gpus_per_node},
    }
    flow._kernel_sources_block = ""
    return flow


def _item(n):
    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem

    return QueueItem(n=n, mode="onboard", target="norm/x")


def test_concurrent_kernel_writers_are_spread_across_the_nodes_devices(tmp_path):
    """The launch protocol hands every role `CUDA_VISIBLE_DEVICES=0`."""
    flow = _kernel_writer_workflow(tmp_path, 4)
    assigned = {
        n: [
            line
            for line in flow._kernel_writer_prompt(_item(n)).splitlines()
            if "CUDA_VISIBLE_DEVICES" in line
        ]
        for n in range(1, 9)
    }
    got = [row[0].split("CUDA_VISIBLE_DEVICES=")[1].split("`")[0] for row in assigned.values()]
    assert got == ["1", "2", "3", "0", "1", "2", "3", "0"]


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


def test_the_units_really_do_disagree_in_the_emitted_flags():
    """Per-unit runtime blocks exist so that units can disagree on flags."""
    core = gates.runtime_flags(gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "core"))
    mm = gates.runtime_flags(gates.resolve_runtime(_task(_PER_UNIT_RUNTIME), "multimodal"))
    assert "--disable_kv_cache_reuse" not in core
    assert "--disable_kv_cache_reuse" in mm
