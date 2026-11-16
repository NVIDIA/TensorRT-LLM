# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake advice (what the lint held as opinions) and the survey's vocabulary checks."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as vocab
from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art
from agent_flow.workflows.trtllm_modeling_v2.spec import advice, intake

# Where this fixture's engine keeps things: no anchor knows these places. Each file is
# found by what it defines, so any other layout would do as well.
FMHA_HEADER = "native/attn/fmha_kernels.h"
KV_DTYPES_SOURCE = "engine/executor/loading.py"


_CONFIG = {
    "architectures": ["ToyForCausalLM"],
    "text_config": {
        "head_dim": 192,
        "num_attention_heads": 8,
        "hidden_size": 1536,
        "sparse_config": {
            "attention_impl": "sparse_gqa",
            "topk": 512,
            "compression_method": "region_block_compress",
            "sparse_indexer_num_heads": 4,
        },
    },
}

_DEP4 = {
    "tensor_parallel_size": 4,
    "moe_expert_parallel_size": 4,
    "enable_attention_dp": True,
    "cuda_graph_config": {"enable_padding": True},
    "max_seq_len": 9216,
    "max_batch_size": 32,
}
_GOOD = {
    "checkpoint_path": None,
    "trtllm_repo_path": None,
    "arch": "sm_103",
    "agents": {"defaults": {"backend": "claude-code", "model": "m", "reasoning_effort": "high"}},
    # Each feature proven once, on its own deployment, against the first one's score.
    "deployments": [
        {"name": "dep4", "options": dict(_DEP4)},
        {
            "name": "kv_fp8",
            "options": {**_DEP4, "kv_cache_config": {"dtype": "fp8"}},
            "relative_to": "dep4",
        },
        {
            "name": "chunked",
            "options": {**_DEP4, "enable_chunked_prefill": True, "max_num_tokens": 512},
            "relative_to": "dep4",
        },
    ],
    "environment": {"kind": "local", "gpus_per_node": 4},
    "accuracy": {"benchmark": "gsm8k", "model_key": "toy"},
    "notes": "Architecture: toy. Prior art in this tree worth surveying: the QSA sparse ops.",
}


def _said(task, **context) -> str:
    """Every line of advice ``task`` gets, joined."""
    found = advice.advise(task, **context)
    return "\n".join((*found.warnings, *found.constraints))


def test_a_placeholder_anchor_is_named_with_the_score_it_reads_under(tmp_path):
    reference = tmp_path / "gsm8k.yaml"
    reference.write_text(yaml.safe_dump({"toy": [{"accuracy": 50.0}]}))
    task = copy.deepcopy(_GOOD)
    task["accuracy"].update(reference_yaml=str(reference))
    said = _said(task)
    assert "anchor reads as a placeholder: `dep4` on `gsm8k` is held to 50, under the 60" in said
    assert "change-requirement" in said
    reference.write_text(yaml.safe_dump({"toy": [{"accuracy": 93.1}]}))
    assert "placeholder" not in _said(task)
    # With no row at all the benchmark will be blocked before its load, and that is said.
    reference.write_text(yaml.safe_dump({"other": [{"accuracy": 93.1}]}))
    assert "no accuracy anchor: `dep4` on `gsm8k`" in _said(task)


def _tree(tmp_path):
    repo = tmp_path / "repo"
    ops = (
        repo / "tensorrt_llm/_torch/_experimental/modeling_v2/models/other/other_tp1/ops/attention"
    )
    ops.mkdir(parents=True)
    (ops / "qsa_paged_sparse_gqa.py").write_text("class A: ...\n")
    (ops / "rms_norm_rows.py").write_text("class B: ...\n")
    backend = repo / "engine/attn/sparse/qsa"
    backend.mkdir(parents=True)
    (backend / "kernels.py").write_text(
        "import triton\n\n\n@triton.jit\ndef triton_qsa_paged_sparse_gqa(q):\n    pass\n\n\n"
        "def helper():\n    pass\n"
    )
    header = repo / FMHA_HEADER
    header.parent.mkdir(parents=True)
    header.write_text('return {false, "HeadDimV must be 32, 64, 80, 128, or a multiple of 128"};\n')
    return repo


def _checkpoint(tmp_path, gib=None):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps(_CONFIG))
    if gib is not None:
        index = {"metadata": {"total_size": int(gib * 2**30)}, "weight_map": {}}
        (ckpt / "model.safetensors.index.json").write_text(json.dumps(index))
    return ckpt


def test_intake_lint_prints_advice_and_fails_only_a_spec_that_does_not_load(tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path)
    repo = _tree(tmp_path)
    task = copy.deepcopy(_GOOD)
    task.update(checkpoint_path=str(checkpoint), trtllm_repo_path=str(repo))
    task["environment"].update(kind="local", python="/usr/bin/python3")
    # Opinions that would have stood as findings: scored eager, on a placeholder anchor.
    task["deployments"][0]["options"]["cuda_graph_config"] = None
    task["deployments"][0]["benchmarks"] = [{"name": "gsm8k", "anchor": 10, "source": "none yet"}]
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    assert advice.main(["--task", str(path), "--report", str(tmp_path / "r.md")]) == 0
    report = (tmp_path / "r.md").read_text()
    assert report.startswith("# Intake advice")
    assert "anchor reads as a placeholder" in report
    # How many model loads the spec implies, where a person reads it before a run: one
    # session per deployment, and what adds to it.
    names = [d["name"] for d in task["deployments"]]
    line = next(line for line in report.splitlines() if line.startswith("Model loads"))
    assert line.startswith(f"Model loads the spec implies: {len(names)} at least (")
    assert all(f"`{name}` 1" in line for name in names)
    assert "The final pass measures a deployment again, one session more, only where" in line
    # A disaggregated pair is two role instances: its one session loads the model twice.
    pair = {
        "deployments": [
            {"name": "core", "options": {}},
            {"name": "pd", "ctx": {"options": {}}, "gen": {"options": {}}},
        ]
    }
    assert advice.loads_note(pair).startswith(
        "Model loads the spec implies: 3 at least (`core` 1, `pd` 2)"
    )
    assert advice.loads_note({}) == ""
    # With more than one deployment: how to reach a first milestone early. One line, and
    # nothing is refused for it; a spec on a baseline, or of one deployment, gets none.
    said = advice.milestone_note(pair)
    assert said.startswith(
        "To reach a first milestone early: run the first deployment (`core`) alone"
    )
    assert "then the rest (`pd`) in a second run" in said and "`onboarded: <ref>`" in said
    assert advice.milestone_note({**pair, "onboarded": "refs/mv2/x/onboarded"}) == ""
    # The second spec lists no first deployment, so none of its own can be judged against
    # it: the line says what to state instead, where the spec has such a deployment.
    assert "relative_to" not in said
    held = copy.deepcopy(pair)
    held["deployments"][1]["relative_to"] = "core"
    assert advice.milestone_note(held).endswith(
        "In the second spec `pd` cannot be `relative_to: core`, which is not measured "
        "there: state the score the first run measured for `core` as the benchmark's "
        "`anchor`, with its `source`."
    )
    assert advice.milestone_note({"deployments": pair["deployments"][:1]}) == ""
    assert len(names) > 1 and "To reach a first milestone early" in report
    found = advice.advise(pair)
    assert found.plan == (found.loads, found.milestone) and all(found.plan)
    capsys.readouterr()
    # A spec that cannot be read as one is the only thing that fails it.
    task["members"] = ["tp4"]
    path.write_text(yaml.safe_dump(task))
    assert intake.main(["lint", "--task", str(path)]) == 1
    assert "'members' is no longer read" in capsys.readouterr().out
    assert intake.main(["--help"]) == 0


def test_what_a_run_cannot_start_without_is_said(tmp_path):
    repo = _tree(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    task = copy.deepcopy(_GOOD)
    assert "agents:" not in _said(task, checkpoint=checkpoint, repo=repo)
    del task["agents"]
    assert "agents:" in _said(task, checkpoint=checkpoint, repo=repo)


def test_a_kv_dtype_the_engine_does_not_name_is_said(tmp_path):
    repo = _tree(tmp_path)
    loader = repo / KV_DTYPES_SOURCE
    loader.parent.mkdir(parents=True)
    loader.write_text('_KV_CACHE_MAP = {}\n_VALID_KV_CACHE_DTYPES = ("fp8", "nvfp4", "auto")\n')
    task = copy.deepcopy(_GOOD)
    task["deployments"].append(
        {"name": "kv_bf16", "options": {"kv_cache_config": {"dtype": "bfloat16"}}}
    )
    checkpoint = _checkpoint(tmp_path)
    said = _said(task, checkpoint=checkpoint, repo=repo)
    assert "kv_cache_config.dtype 'bfloat16' (kv_bf16) is not one the engine names" in said
    assert f"{KV_DTYPES_SOURCE}:2" in said
    task["deployments"][-1]["options"]["kv_cache_config"]["dtype"] = "auto"
    assert "KV dtype" not in _said(task, checkpoint=checkpoint, repo=repo)
    # A tree that does not say what it accepts is not guessed at.
    assert prior_art.engine_kv_dtypes(tmp_path / "elsewhere") is None


def test_the_first_deployment_holds_one_engine_of_the_checkpoint(tmp_path):
    task = copy.deepcopy(_GOOD)
    task["environment"]["gpu_memory_gib"] = 80
    # 300 GiB over 60 GiB a device (75% of 80): four devices cannot hold the weights.
    checkpoint = _checkpoint(tmp_path, 300)
    task["checkpoint_path"] = str(checkpoint)
    said = _said(task, checkpoint=checkpoint)
    assert "first deployment too narrow: `dep4` cannot hold the weights: 300 GiB on 4" in said
    # Unknown device memory: no estimate.
    del task["environment"]["gpu_memory_gib"]
    assert "too narrow" not in _said(task, checkpoint=checkpoint)


def test_a_vocabulary_is_held_to_what_code_needs_and_every_problem_is_said_at_once():
    """Which entry, which mode, what it waits for: the rest is the surveyor's to word."""
    doc = {
        "queue": [
            {"n": 1, "mode": "guess", "target": ""},
            {"n": 1, "mode": "verify", "target": "gemm/cublas_mm", "depends_on": [7]},
            {"n": 3, "mode": "onboard", "target": "ops/norm/rms"},
            {"n": 4, "mode": "onboard", "target": "ops/norm/rms"},
        ],
    }
    with pytest.raises(vocab.VocabularyError) as excinfo:
        vocab.parse_vocabulary(doc)
    message = str(excinfo.value)
    for said in ("queue[0].mode is 'guess'", "queue[0].target must name exactly one catalog entry",
                 "queue[1].n = 1 is a duplicate", "depends on 7, which is not in the queue",
                 "queue[3] is a second row on 'ops/norm/rms'"):  # fmt: skip
        assert said in message, said
    # A survey's own reasoning is held to no form: a verdict, a map of the steps with any
    # words for their state, a sourcing rung, a hundred cells, a verify row with no surface.
    free = {
        "verdict": "maybe",
        "steps": [{"step": "", "state": "NOPE", "tier": "tier 2"}],
        "queue": [
            {"n": 1, "mode": "verify", "target": "gemm/cublas_mm", "tier": 0,
             "cells": [{"m": m} for m in range(100)]},
            {"n": 2, "mode": "verify", "target": "gemm/cublas_mm", "site": "second_site"},
        ],
    }  # fmt: skip
    first, second = vocab.parse_vocabulary(free)
    assert (first.target, first.mode, first.variant) == (
        "gemm/cublas_mm", "verify", "ops/gemm/cublas_mm__r1",
    )  # fmt: skip
    # Two rows on one shared entry are two variants: no file is written twice.
    assert second.variant == "ops/gemm/cublas_mm__second_site"
    assert vocab.parse_vocabulary({}) == ()
