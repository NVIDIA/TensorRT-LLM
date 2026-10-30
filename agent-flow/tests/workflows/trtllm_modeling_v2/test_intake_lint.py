# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake lint and prior art: what a spec costs before launch, and what the tree already has."""

from __future__ import annotations

import copy
import json

import yaml

from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art, tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.spec import intake, lint

FMHA_HEADER = tree_anchors.ANCHORS["fmha_head_dims"].hint
STOCK_MODELS_DIR = tree_anchors.ANCHORS["builtin_models"].hint
STOCK_ARCH_INDEX = tree_anchors.ANCHORS["builtin_model_index"].hint
KV_DTYPES_SOURCE = tree_anchors.ANCHORS["kv_cache_dtypes"].hint

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

_GOOD = {
    "checkpoint_path": None,
    "trtllm_repo_path": None,
    "arch": "sm_103",
    "parallel": "unified",
    "members": {"default": ["tp4", "tep4", "dep4"]},
    "features": ["disagg"],
    "runtime": {
        "cuda_graph": True,
        "chunked_prefill": False,
        "kv_cache_reuse": False,
        "extra_flags": ["--max_seq_len=9216", "--max_batch_size=128", "--max_num_tokens=8192"],
    },
    # Each feature proven once, at the member the request validates fully (dep4).
    "variants": {
        "kv_fp8": {
            "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
            "members": ["dep4"],
            "gates": ["accuracy"],
        },
        "chunked": {
            "runtime": {"extra_flags": ["--max_num_tokens=512"]},
            "members": ["dep4"],
            "gates": ["accuracy"],
        },
    },
    "environment": {"kind": "session_spool", "gpus_per_node": 4},
    "accuracy": {"benchmark": "gsm8k"},
    "perf": {"enabled": True, "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]}},
    # tp4 / tep4 are text-only: core alone. disagg's test runs once, at dep4.
    "coverage": {"accept": ["kv_fp8@tp4", "kv_fp8@tep4", "chunked@tp4", "chunked@tep4"]},
    "notes": "Architecture: toy. Prior art in this tree worth surveying: the QSA sparse ops.",
}


def _slow():
    task = copy.deepcopy(_GOOD)
    task["runtime"]["cuda_graph"] = False
    task["runtime"]["chunked_prefill"] = True
    task["runtime"]["extra_flags"] = [
        "--max_seq_len=16384",
        "--max_batch_size=32",
        "--max_num_tokens=2048",
    ]
    task["variants"] = {
        f"{dtype}_c{c}_r{r}": {
            "runtime": {"chunked_prefill": bool(c), "kv_cache_reuse": bool(r)},
            "llm_api_options": {"kv_cache_config": {"dtype": dtype}},
        }
        for dtype in ("fp8", "nvfp4")
        for c in (0, 1)
        for r in (0, 1)
    }
    task["accuracy"] = {
        "benchmark": "gsm8k",
        "benchmark_args": ["--apply_chat_template"],
        "envelope": {"enabled": True},
        "disagg": {"test": "tests/accuracy/test_disagg.py::TestToy::test_auto_dtype", "devices": 8},
    }
    task["notes"] = "Never inspect any preexisting implementation, including sibling checkouts."
    return task


def _rules(report):
    return {f.rule for f in report.findings}


def _finding(rule, task, **context):
    """The finding of ``rule`` the lint makes of ``task``, or ``None``."""
    return next((f for f in lint.lint_task(task, **context).findings if f.rule == rule), None)


def test_a_batch_and_a_concurrency_are_both_global_totals():
    task = copy.deepcopy(_GOOD)
    task["members"] = {"default": ["dep4"]}
    task["runtime"]["extra_flags"] = ["--max_seq_len=9216", "--max_batch_size=32"]
    # Attention DP splits both alike: 32 in all is below 128 in all, whatever the ranks.
    assert "runtime-batch-below-concurrency" in _rules(lint.lint_task(task))
    task["runtime"]["extra_flags"] = ["--max_seq_len=9216", "--max_batch_size=128"]
    assert "runtime-batch-below-concurrency" not in _rules(lint.lint_task(task))


def test_an_accuracy_check_never_runs_eager():
    task = copy.deepcopy(_GOOD)
    task["variants"]["kv_fp8"]["runtime"] = {"cuda_graph": False}
    task["variants"]["chunked"]["llm_api_options"] = {"cuda_graph_config": None}
    finding = _finding("accuracy-eager", task)
    assert finding.severity == lint.BLOCK
    assert "without CUDA graphs for kv_fp8, chunked:" in finding.message
    # A block stands whatever the notes accept.
    task["notes"] += "\nintake-accept: accuracy-eager -- capture is a planned task"
    assert "accuracy-eager" in {f.rule for f in lint.lint_task(task).open()}
    # Off for the run: every unit and variant scored eager, and a smoke unit, which scores no
    # benchmark, is a warn; disagg's test configures its own engines.
    task = copy.deepcopy(_GOOD)
    task["runtime"]["cuda_graph"] = False
    task["features"] = ["disagg", "multimodal"]
    task["profile"] = {"multimodal": "smoke"}
    findings = {f.rule: f for f in lint.lint_task(task).findings}
    assert "for core, kv_fp8, chunked:" in findings["accuracy-eager"].message
    assert "the checks of multimodal run" in findings["runtime-cuda-graph-off"].message
    assert not {"accuracy-eager", "runtime-cuda-graph-off"} & _rules(lint.lint_task(_GOOD))


def test_a_placeholder_anchor_is_named_with_the_floor_it_implies(tmp_path):
    reference = tmp_path / "gsm8k.yaml"
    reference.write_text(yaml.safe_dump({"toy": [{"accuracy": 50.0}]}))
    task = copy.deepcopy(_GOOD)
    task["accuracy"].update(reference_yaml=str(reference), model_key="toy")
    assert _finding("accuracy-placeholder-anchor", task)
    reference.write_text(yaml.safe_dump({"toy": [{"accuracy": 93.1}]}))
    assert _finding("accuracy-placeholder-anchor", task) is None
    reference.write_text("# placeholder\n" + yaml.safe_dump({"toy": [{"accuracy": 95.0}]}))
    assert "95 -> floor 93.80" in _finding("accuracy-placeholder-anchor", task).message


def _tree(tmp_path):
    repo = tmp_path / "repo"
    ops = (
        repo / "tensorrt_llm/_torch/_experimental/modeling_v2/models/other/other_tp1/ops/attention"
    )
    ops.mkdir(parents=True)
    (ops / "qsa_paged_sparse_gqa.py").write_text("class A: ...\n")
    (ops / "rms_norm_rows.py").write_text("class B: ...\n")
    backend = repo / "tensorrt_llm/_torch/attention/backends/sparse/qsa"
    backend.mkdir(parents=True)
    (backend / "kernels.py").write_text(
        "def triton_qsa_paged_sparse_gqa(q):\n    pass\n\ndef helper():\n    pass\n"
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


def test_intake_lint_cli_exits_one_while_a_finding_stands(tmp_path, capsys):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(_slow()))
    assert lint.main(["--task", str(path), "--report", str(tmp_path / "r.md")]) == 1
    assert (tmp_path / "r.md").read_text().startswith("# Intake lint")
    assert intake.main(["--help"]) == 0


def test_the_environment_can_start_the_run(tmp_path):
    repo = _tree(tmp_path)
    zoo = repo / STOCK_MODELS_DIR
    zoo.mkdir(parents=True)
    (zoo / "modeling_other.py").write_text(
        '@register_auto_model("OtherForCausalLM")\nclass M: ...\n'
    )
    checkpoint = _checkpoint(tmp_path)
    task = copy.deepcopy(_GOOD)
    task["environment"].update(
        container_image="/images/x.sqsh",
        mounts=[f"{tmp_path}:/work"],
        python="/work/venv/bin/python",
    )
    rules = _rules(lint.lint_task(task, checkpoint=checkpoint, repo=repo))
    assert {"environment-python-missing", "substrate-checkpoint-missing"} <= rules
    # The image's own interpreter is not on this host to check; a proxy answers the gate.
    task["environment"].update(python="/usr/bin/python3", substrate_checkpoint=str(checkpoint))
    rules = _rules(lint.lint_task(task, checkpoint=checkpoint, repo=repo))
    assert not {"environment-python-missing", "substrate-checkpoint-missing"} & rules
    # The lazy zoo's index registers architectures too.
    (repo / STOCK_ARCH_INDEX).write_text(
        'MODEL_ARCH_TO_MODULE = {"ToyForCausalLM": "modeling_toy"}\n'
    )
    assert prior_art.stock_supports(repo, _CONFIG) is True


def test_a_kv_dtype_the_engine_refuses_is_a_block(tmp_path):
    repo = _tree(tmp_path)
    loader = repo / KV_DTYPES_SOURCE
    loader.parent.mkdir(parents=True)
    loader.write_text('_KV_CACHE_MAP = {}\n_VALID_KV_CACHE_DTYPES = ("fp8", "nvfp4", "auto")\n')
    task = copy.deepcopy(_GOOD)
    task["variants"]["kv_bf16"] = {"llm_api_options": {"kv_cache_config": {"dtype": "bfloat16"}}}
    finding = _finding("kv-cache-dtype", task, repo=repo)
    assert finding.severity == lint.BLOCK
    assert "'bfloat16' (kv_bf16)" in finding.message
    assert f"{KV_DTYPES_SOURCE}:2" in finding.message
    task["variants"]["kv_bf16"]["llm_api_options"]["kv_cache_config"]["dtype"] = "auto"
    assert "kv-cache-dtype" not in _rules(lint.lint_task(task, repo=repo))
    # A tree that does not say what it accepts is not guessed at.
    assert prior_art.engine_kv_dtypes(tmp_path / "elsewhere") is None


def test_a_feature_is_not_proven_again_under_another_axis():
    task = copy.deepcopy(_GOOD)
    task["features"] = ["disagg", "mtp"]
    task["members"]["mtp"] = ["dep4"]
    task["variants"]["mtp_kv_fp8"] = {
        "unit": "mtp",
        "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
    }
    finding = _finding("feature-combination", task)
    assert "mtp_kv_fp8 (mtp x kv dtype: " in finding.message
    assert "core proves the axis in kv_fp8" in finding.message
    # A variant on a feature unit that changes none of the storage or scheduling axes is the
    # feature's own configuration, not a combination.
    task["variants"]["mtp_kv_fp8"] = {"unit": "mtp", "env": {"TRTLLM_MTP_EAGER": "1"}}
    assert "feature-combination" not in _rules(lint.lint_task(task))


def test_every_member_holds_one_engine_of_the_checkpoint(tmp_path):
    task = copy.deepcopy(_GOOD)
    task["environment"]["gpu_memory_gib"] = 80
    # 300 GiB over 60 GiB a device (75% of 80): one engine needs 8 devices.
    finding = _finding("member-too-narrow", task, checkpoint=_checkpoint(tmp_path, 300))
    assert finding.message.startswith("dep4, tep4, tp4 hold one engine on fewer than 8")
    # Unknown device memory: no estimate.
    del task["environment"]["gpu_memory_gib"]
    assert "member-too-narrow" not in _rules(lint.lint_task(task, checkpoint=tmp_path / "ckpt"))
