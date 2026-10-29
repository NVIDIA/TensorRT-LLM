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
    "members": {"default": ["tp4", "tep4", "dep4"], "disagg": ["dep2"]},
    "features": ["disagg"],
    "runtime": {
        "cuda_graph": True,
        "chunked_prefill": True,
        "kv_cache_reuse": True,
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
    # dep2 holds disagg alone; the wider disagg members are checked outside the flow, and tp4 /
    # tep4 are text-only: core alone.
    "coverage": {
        "accept": [
            "core@dep2",
            "kv_fp8@dep2",
            "kv_fp8@tp4",
            "kv_fp8@tep4",
            "chunked@dep2",
            "chunked@tp4",
            "chunked@tep4",
            "disagg@tp4",
            "disagg@tep4",
            "disagg@dep4",
        ]
    },
    "notes": "Architecture: toy. Prior art in this tree worth surveying: the QSA sparse ops.",
}


def _slow():
    task = copy.deepcopy(_GOOD)
    task["members"]["disagg"] = ["dep4"]
    task["runtime"]["cuda_graph"] = False
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
    }
    task["notes"] = "Never inspect any preexisting implementation, including sibling checkouts."
    return task


def _rules(report):
    return {f.rule for f in report.findings}


def test_a_spec_built_for_serving_passes():
    report = lint.lint_task(_GOOD)
    assert not report.open(), report.render()


def test_slow_and_stalling_choices_are_named():
    rules = _rules(lint.lint_task(_slow()))
    assert {
        "accuracy-eager",
        "runtime-batch-below-concurrency",
        "runtime-max-seq-len-oversized",
        "runtime-small-chunk",
        "disagg-two-nodes",
        "variants-cross-product",
        "accuracy-chat-template-short-output",
        "envelope-open-ended-probes",
        "notes-missing-brief",
        "notes-broad-embargo",
    } <= rules


def test_attention_dp_splits_the_batch_across_ranks():
    task = copy.deepcopy(_GOOD)
    task["members"] = {"default": ["dep4"]}
    task["runtime"]["extra_flags"] = ["--max_seq_len=9216", "--max_batch_size=32"]
    assert "runtime-batch-below-concurrency" not in _rules(lint.lint_task(task))
    task["members"] = {"default": ["tp4"]}
    assert "runtime-batch-below-concurrency" in _rules(lint.lint_task(task))


def test_notes_accept_a_finding_with_its_reason():
    task = _slow()
    task["notes"] += "\nintake-accept: runtime-small-chunk -- the request serves short chunks"
    report = lint.lint_task(task)
    assert "runtime-small-chunk" in _rules(report)
    assert "runtime-small-chunk" not in {f.rule for f in report.open()}


def test_an_accuracy_check_never_runs_eager():
    task = copy.deepcopy(_GOOD)
    task["variants"]["kv_fp8"]["runtime"] = {"cuda_graph": False}
    task["variants"]["chunked"]["llm_api_options"] = {"cuda_graph_config": None}
    finding = next(f for f in lint.lint_task(task).findings if f.rule == "accuracy-eager")
    assert finding.severity == lint.BLOCK
    assert "without CUDA graphs for kv_fp8, chunked:" in finding.message
    # A block stands whatever the notes accept.
    task["notes"] += "\nintake-accept: accuracy-eager -- capture is a planned task"
    assert "accuracy-eager" in {f.rule for f in lint.lint_task(task).open()}
    # Off for the run: every unit and variant scored eager, and disagg's witness, which scores
    # no benchmark, is a warn.
    task = copy.deepcopy(_GOOD)
    task["runtime"]["cuda_graph"] = False
    findings = {f.rule: f for f in lint.lint_task(task).findings}
    assert "for core, kv_fp8, chunked:" in findings["accuracy-eager"].message
    assert "the checks of disagg run" in findings["runtime-cuda-graph-off"].message
    assert not {"accuracy-eager", "runtime-cuda-graph-off"} & _rules(lint.lint_task(_GOOD))


def test_a_placeholder_anchor_is_named(tmp_path):
    reference = tmp_path / "gsm8k.yaml"
    reference.write_text(yaml.safe_dump({"toy": [{"accuracy": 50.0}]}))
    task = copy.deepcopy(_GOOD)
    task["accuracy"].update(reference_yaml=str(reference), model_key="toy")
    assert "accuracy-placeholder-anchor" in _rules(lint.lint_task(task))
    reference.write_text(yaml.safe_dump({"toy": [{"accuracy": 93.1}]}))
    assert "accuracy-placeholder-anchor" not in _rules(lint.lint_task(task))


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


def test_prior_art_ranks_what_shares_the_checkpoints_vocabulary(tmp_path):
    repo = _tree(tmp_path)
    found = prior_art.find_prior_art(repo, _CONFIG)
    paths = [c.path for c in found.candidates]
    assert any(p.endswith("ops/attention/qsa_paged_sparse_gqa.py") for p in paths)
    assert any(p.endswith("backends/sparse/qsa/kernels.py") for p in paths)
    assert not any(p.endswith("rms_norm_rows.py") for p in paths)
    kernels = next(c for c in found.candidates if c.path.endswith("kernels.py"))
    assert kernels.names == ("triton_qsa_paged_sparse_gqa",)


def test_engine_constraints_read_the_fmha_rule_from_the_tree(tmp_path):
    repo = _tree(tmp_path)
    notes = prior_art.engine_constraints(_CONFIG, repo)
    assert len(notes) == 1 and "head dim 192" in notes[0] and "256" in notes[0]
    assert f"{FMHA_HEADER}:1" in notes[0]
    fits = {"text_config": {"head_dim": 128}}
    assert prior_art.engine_constraints(fits, repo) == []


def test_lint_reports_constraints_and_prior_art_from_the_checkpoint(tmp_path):
    repo = _tree(tmp_path)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_CONFIG))
    report = lint.lint_task(_GOOD, checkpoint=checkpoint, repo=repo)
    assert "engine-constraint" in _rules(report)
    text = report.render()
    assert "## Prior art in this tree" in text and "qsa_paged_sparse_gqa" in text


def test_intake_lint_cli_exits_one_while_a_finding_stands(tmp_path, capsys):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(_slow()))
    assert lint.main(["--task", str(path), "--report", str(tmp_path / "r.md")]) == 1
    assert (tmp_path / "r.md").read_text().startswith("# Intake lint")
    assert intake.main(["--help"]) == 0


def test_notes_stay_the_runs_brief():
    task = copy.deepcopy(_GOOD)
    task["notes"] += " x" * lint.NOTES_BRIEF_CHARS
    assert "notes-long" in _rules(lint.lint_task(task))
    task = copy.deepcopy(_GOOD)
    task["notes"] = (
        "INTAKE-ONLY HANDOFF. No build or Slurm job is allowed here.\n\n" + task["notes"]
    )
    assert "notes-intake-process" in _rules(lint.lint_task(task))
    # An accepted finding may explain itself in the intake's own words.
    task = copy.deepcopy(_GOOD)
    task["notes"] += "\n\nintake-accept: engine-constraint -- this intake chose padding."
    assert "notes-intake-process" not in _rules(lint.lint_task(task))


def test_a_placeholder_names_the_floor_it_implies(tmp_path):
    reference = tmp_path / "anchor.yaml"
    reference.write_text("# placeholder\n" + yaml.safe_dump({"toy": [{"accuracy": 95.0}]}))
    task = copy.deepcopy(_GOOD)
    task["accuracy"].update(reference_yaml=str(reference), model_key="toy")
    finding = next(
        f for f in lint.lint_task(task).findings if f.rule == "accuracy-placeholder-anchor"
    )
    assert "95 -> floor 93.80" in finding.message


def test_a_witness_launcher_reads_only_what_the_spec_sets(tmp_path):
    (tmp_path / "leg_b.py").write_text(
        "import os\ndriver = os.environ.get('TOY_DRIVER')\nos.environ['MV2FLOW_WITNESS_MEMBER']\n"
    )
    task = copy.deepcopy(_GOOD)
    block = {"contrast_command": "leg_b.py --fast", "contrast_env": {"TOY_LEG": "1"}}
    task["accuracy"]["witness"] = {"enabled": True, "disagg": block}
    findings = lint.lint_task(task, base=tmp_path).findings
    unset = next(f for f in findings if f.rule == "witness-contrast-env")
    assert "TOY_DRIVER" in unset.message and "MV2FLOW_" not in unset.message
    block["contrast_env"]["TOY_DRIVER"] = "/path/to/driver.py"
    assert "witness-contrast-env" not in _rules(lint.lint_task(task, base=tmp_path))
    block["contrast_command"] = "not_written_yet.py"
    pending = lint.lint_task(task, base=tmp_path)
    assert "witness-contrast-pending" in _rules(pending) and not pending.open()


def test_the_environment_can_start_the_run(tmp_path):
    repo = _tree(tmp_path)
    zoo = repo / STOCK_MODELS_DIR
    zoo.mkdir(parents=True)
    (zoo / "modeling_other.py").write_text(
        '@register_auto_model("OtherForCausalLM")\nclass M: ...\n'
    )
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_CONFIG))
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


def test_the_top_prior_art_leads_are_named_in_notes(tmp_path):
    repo = _tree(tmp_path)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_CONFIG))
    task = copy.deepcopy(_GOOD)
    unnamed = next(
        f
        for f in lint.lint_task(task, checkpoint=checkpoint, repo=repo).findings
        if f.rule == "notes-prior-art-unnamed"
    )
    assert "qsa/kernels.py" in unnamed.message
    task["notes"] += (
        " qsa/kernels.py: paged sparse GQA. attention/qsa_paged_sparse_gqa.py: the op around it."
    )
    rules = _rules(lint.lint_task(task, checkpoint=checkpoint, repo=repo))
    assert "notes-prior-art-unnamed" not in rules


def test_rules_are_a_registry():
    def toy_rule(ctx):
        if ctx.task.get("features"):
            yield lint.Finding("toy-feature", lint.INFO, "seen")

    report = lint.lint_task(_GOOD, rules=(toy_rule,))
    assert [f.rule for f in report.findings] == ["toy-feature"]
    assert lint.runtime_rules in lint.RULES and prior_art.fmha_head_dims in prior_art.ENGINE_RULES


def test_planned_sessions_are_not_a_schema_block():
    error = (
        "task.yaml failed trtllm-modeling-v2 schema validation:\n"
        "  - 'environment.spool' does not exist: /s. Start the compute session first.\n"
        "  - missing required field 'arch'"
    )
    problems, spools = lint._planned_sessions(error)
    assert problems == ["missing required field 'arch'"]
    assert spools and spools[0].startswith("'environment.spool' does not exist")


def test_a_kv_dtype_the_engine_refuses_is_a_block(tmp_path):
    repo = _tree(tmp_path)
    loader = repo / KV_DTYPES_SOURCE
    loader.parent.mkdir(parents=True)
    loader.write_text('_KV_CACHE_MAP = {}\n_VALID_KV_CACHE_DTYPES = ("fp8", "nvfp4", "auto")\n')
    task = copy.deepcopy(_GOOD)
    task["variants"]["kv_bf16"] = {"llm_api_options": {"kv_cache_config": {"dtype": "bfloat16"}}}
    finding = next(
        f for f in lint.lint_task(task, repo=repo).findings if f.rule == "kv-cache-dtype"
    )
    assert finding.severity == lint.BLOCK
    assert "'bfloat16' (kv_bf16)" in finding.message
    assert f"{KV_DTYPES_SOURCE}:2" in finding.message
    task["variants"]["kv_bf16"]["llm_api_options"]["kv_cache_config"]["dtype"] = "auto"
    assert "kv-cache-dtype" not in _rules(lint.lint_task(task, repo=repo))
    # A tree that does not say what it accepts is not guessed at.
    assert prior_art.engine_kv_dtypes(tmp_path / "elsewhere") is None


def test_a_requested_cell_no_check_covers_is_a_block_before_preflight_finds_it():
    task = copy.deepcopy(_GOOD)
    task["coverage"]["accept"].remove("disagg@dep4")
    finding = next(f for f in lint.lint_task(task).findings if f.rule == "coverage-gap")
    assert finding.severity == lint.BLOCK and "disagg@dep4" in finding.message
    assert "disagg@tp4" not in finding.message


def test_a_variant_that_boots_boots_at_every_member_of_its_unit():
    task = copy.deepcopy(_GOOD)
    for name in ("kv_fp8", "chunked"):
        del task["variants"][name]["gates"]
    finding = next(f for f in lint.lint_task(task).findings if f.rule == "variant-boot-everywhere")
    # Each variant boots at its own member and at every other member of core.
    assert "kv_fp8 (3)" in finding.message and "chunked (3)" in finding.message
    assert "6 boot checks" in finding.message
    assert "variant-boot-everywhere" not in _rules(lint.lint_task(_GOOD))


def test_a_feature_is_proven_at_one_member():
    task = copy.deepcopy(_GOOD)
    task["features"] = ["disagg", "multimodal"]
    finding = next(f for f in lint.lint_task(task).findings if f.rule == "feature-members")
    assert "multimodal at tep4, dep4, tp4 (6 checks)" in finding.message
    # disagg's one witness member is already one.
    assert "disagg" not in finding.message
    task["members"]["multimodal"] = ["dep4"]
    assert "feature-members" not in _rules(lint.lint_task(task))
    task["members"]["multimodal"] = ["tp4", "dep4"]
    task["profile"] = {"multimodal": "certified"}
    assert "feature-members" not in _rules(lint.lint_task(task))


def test_a_feature_is_not_proven_again_under_another_axis():
    task = copy.deepcopy(_GOOD)
    task["variants"]["disagg_kv_fp8"] = {
        "unit": "disagg",
        "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
    }
    finding = next(f for f in lint.lint_task(task).findings if f.rule == "feature-combination")
    assert "disagg_kv_fp8 (disagg x kv dtype: 1 check; core proves the axis in kv_fp8)" in (
        finding.message
    )
    # A variant on a feature unit that changes none of the storage or scheduling axes is the
    # feature's own configuration, not a combination.
    task["variants"]["disagg_kv_fp8"] = {"unit": "disagg", "env": {"TRTLLM_KV_TRANSFER": "1"}}
    assert "feature-combination" not in _rules(lint.lint_task(task))


def test_one_variant_crossing_two_axes_is_named():
    task = copy.deepcopy(_GOOD)
    task["variants"]["fp8_chunked"] = {
        "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
        "runtime": {"extra_flags": ["--max_num_tokens=512"]},
        "members": ["dep4"],
        "gates": ["accuracy"],
    }
    finding = next(f for f in lint.lint_task(task).findings if f.rule == "variants-cross-product")
    assert "1 variant combines" in finding.message and "fp8_chunked" in finding.message


def test_boot_boundaries_say_where_they_run_unless_certified():
    task = copy.deepcopy(_GOOD)
    task["accuracy"]["envelope"] = {"enabled": True, "probes": ["1, 2, 3, 4, 5, 6, 7,"]}
    report = lint.lint_task(task)
    finding = next(f for f in report.findings if f.rule == "envelope-on")
    # One member of each unit the boot gates; disagg's witness boots none.
    assert finding.severity == lint.INFO and "the boot at core@dep4 also runs" in finding.message
    assert not report.open()
    task["profile"] = "certified"
    assert "envelope-on" not in _rules(lint.lint_task(task))


def test_the_report_shows_where_the_checks_go():
    report = lint.lint_task(_GOOD)
    assert report.contract_lines[0].startswith("9 required checks")
    rows = report.contract_lines[2:]
    assert "- `core`: boot 3 (" in rows[0] and "accuracy 3 (" in rows[0]
    assert any(r.startswith("- `kv_fp8`: accuracy 1 (dep4)") for r in rows)
    assert any(r.startswith("- `chunked`: accuracy 1 (dep4)") for r in rows)
