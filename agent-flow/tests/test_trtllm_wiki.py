# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TensorRT-LLM wiki, from extraction to the workflow's use of it.

Extraction from a small checkout, lookups, the pack cache and the CLI; and the
facts the wiki and the ModelingV2 workflow rely on, held to this checkout.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.trtllm_wiki import accuracy, cli, config, env, features, modeling_v2, pack, recipes
from agent_flow.trtllm_wiki import tests as test_lists
from agent_flow.trtllm_wiki.changes import diff_packs
from agent_flow.trtllm_wiki.query import Wiki
from agent_flow.workflows.trtllm_modeling_v2 import cli as flow_cli
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

_CHECKOUT = Path(__file__).resolve().parents[2]
_MV2 = modeling_v2.MODELING_V2_REL

_SCHEMA = {
    "title": "TensorRT-LLM 9.9 trtllm-serve --config",
    "properties": {
        "max_num_tokens": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 8192},
        "env_overrides": {"anyOf": [{"type": "object"}, {"type": "null"}], "status": "prototype"},
        "kv_cache_config": {"$ref": "#/$defs/KvCacheConfig"},
        "speculative_config": {
            "anyOf": [
                {
                    "discriminator": {
                        "mapping": {"MTP": "#/$defs/Mtp", "DFlash": "#/$defs/DFlash"},
                        "propertyName": "decoding_type",
                    },
                    "oneOf": [{"$ref": "#/$defs/Mtp"}, {"$ref": "#/$defs/DFlash"}],
                },
                {"type": "null"},
            ]
        },
    },
    "$defs": {
        "KvCacheConfig": {
            "properties": {
                "enable_block_reuse": {"type": "boolean", "default": True},
                "dtype": {"type": "string", "default": "auto"},
            }
        },
        "Mtp": {
            "properties": {"decoding_type": {"const": "MTP"}, "max_draft_len": {"type": "integer"}}
        },
        "DFlash": {
            "properties": {
                "decoding_type": {"const": "DFlash"},
                "speculative_model": {"type": "string"},
            }
        },
    },
}

_OP = """
from ..._op import Arch, OpWrapper

class {name}(OpWrapper):
    ARCHS = frozenset({{Arch.SM_100}})
    CELLS = ()

    def __call__(self, x):
        torch.ops.trtllm.{call}(x)
        native_ops.fused_step(x)

    def reference(self, x):
        return x
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checkout(root: Path) -> Path:
    """A checkout holding one of every source the wiki reads."""
    _write(root / config.SERVE_SCHEMA_REL, json.dumps(_SCHEMA))
    family = root / _MV2 / "models" / "toy_vl"
    _write(
        family / "routing.py", '@routes("ToyForConditionalGeneration")\ndef route(c, a, cc): ...\n'
    )
    target = family / "toy_fp8__sm_100__tp4"
    _write(
        target / "modeling.py",
        "from ....catalog.norm.rms import RmsNorm\nfrom .ops.moe.router import Router\n\n"
        "class ModelingV2ToyFp8Sm100Tp4: ...\n",
    )
    _write(target / "ops" / "moe" / "router.py", _OP.format(name="Router", call="toy_router"))
    _write(root / _MV2 / "catalog" / "norm" / "rms.py", _OP.format(name="RmsNorm", call="rms_norm"))
    _write(
        root / modeling_v2.MODELING_V2_TESTS_REL / "norm" / "test_modeling_v2_rms.py",
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms import RmsNorm\n",
    )
    _write(
        root / modeling_v2.MV2_ACCURACY_TESTS_REL / "test_toy_vl.py",
        'class TestToy:\n    MODEL_NAME = "org/Toy"\n\n    def test_gsm8k(self): ...\n',
    )
    lists = {
        "l0_b200": [
            {
                "condition": {
                    "ranges": {"system_gpu_count": {"gte": 4, "lte": 4}},
                    "wildcards": {"gpu": ["*b200*"]},
                    "terms": {"stage": "post_merge", "backend": "pytorch"},
                },
                "tests": [
                    "accuracy/modeling_v2/test_toy_vl.py::TestToy::test_gsm8k TIMEOUT (90)",
                    "unittest/_torch/modeling_v2/norm",
                    "unittest/_torch/models/test_toy_v32_encoding.py",
                    "accuracy/test_param.py::TestP::test_a",
                    "accuracy/test_param.py::TestP::test_b[tp4]",
                    'unittest/_torch/kv -k "not slow" --ignore=unittest/_torch/kv/test_skipped.py',
                ],
            }
        ]
    }
    _write(root / test_lists.TEST_DB_REL / "l0_b200.yml", yaml.safe_dump(lists))
    _write(
        root / test_lists.WAIVES_REL,
        "unittest/_torch/modeling_v2/norm/test_modeling_v2_rms.py SKIP (bug)\n"
        "full:B200/accuracy/test_param.py::TestP::test_a SKIP (gpu bug)\n",
    )
    _write(
        root / config.LLM_ARGS_REL,
        '_TORCH_LLMARGS_REMOVED_ARG_CHECKS: Dict[str, Callable] = {"sampler_type": check}\n',
    )
    _write(
        root / accuracy.ACCURACY_CORE_REL,
        'class GSM8K(AccuracyTask):\n    DATASET = "gsm8k"\n    DATASET_DIR = f"{llm_models_root()}/datasets/gsm8k"\n'
        "    ALPHA = 0.05\n    NUM_SAMPLES = 1319\n",
    )
    _write(
        root / accuracy.REFERENCES_REL / "gsm8k.yaml",
        yaml.safe_dump({"org/Toy": [{"accuracy": 91.0}]}),
    )
    _write(
        root / "examples/configs/database/toy/b200/8k1k_conc64.yaml",
        yaml.safe_dump({"tensor_parallel_size": 4}),
    )
    _write(
        root / recipes.LOOKUP_REL,
        yaml.safe_dump(
            [
                {
                    "model": f"org/Toy-{c}",
                    "arch": "ToyForConditionalGeneration",
                    "gpu": "B200",
                    "num_gpus": 4,
                    "isl": 8192,
                    "osl": 1024,
                    "concurrency": c,
                    "config_path": "examples/configs/database/toy/b200/8k1k_conc64.yaml",
                }
                for c in (4, 64)
            ]
        ),
    )
    _write(
        root / "tensorrt_llm/_torch/knob.py",
        'import os\nx = os.environ.get("TLLM_TOY_KNOB", "0")\n',
    )
    return root


def test_the_schema_flattens_to_dotted_fields_with_union_members_merged(tmp_path):
    fields = {f["path"]: f for f in config.extract_config_fields(_checkout(tmp_path))["fields"]}
    assert fields["kv_cache_config.enable_block_reuse"]["default"] is True
    assert fields["max_num_tokens"]["type"] == "integer|null"
    spec = fields["speculative_config.decoding_type"]
    assert spec["choices"] == ["MTP", "DFlash"] and spec["variants"] == ["MTP", "DFlash"]
    assert fields["speculative_config.speculative_model"]["variants"] == ["DFlash"]
    options = {
        "kv_cache_config": {"enable_block_reuse": False, "enable_blok_reuse": True},
        "env_overrides": {"ANY_NAME": "1"},
        "max_tokens": 1,
    }
    unknown = config.unknown_fields(fields.values(), options)
    assert unknown == ["kv_cache_config.enable_blok_reuse", "max_tokens"]
    # A field the LLM API removed but still takes is not unknown.
    removed = config.extract_config_fields(tmp_path)["removed"]
    assert removed == ["sampler_type"]
    assert not config.unknown_fields(fields.values(), {"sampler_type": "auto"}, removed=removed)
    assert config.suggest(fields.values(), unknown[0]) == "kv_cache_config.enable_block_reuse"


def test_families_targets_and_ops_are_read_from_source(tmp_path):
    repo = _checkout(tmp_path)
    (family,) = modeling_v2.extract_families(repo)
    assert family["family"] == "toy_vl" and family["architectures"] == [
        "ToyForConditionalGeneration"
    ]
    (target,) = family["targets"]
    assert (target["checkpoint"], target["sm"], target["parallel"]) == ("toy_fp8", "sm_100", "tp4")
    assert family["accuracy_tests"]["classes"] == [
        {"name": "TestToy", "line": 1, "tests": ["test_gsm8k"], "model": "org/Toy"}
    ]
    ops = {op["name"]: op for op in modeling_v2.extract_ops(repo)}
    assert (ops["Router"]["scope"], ops["Router"]["target"], ops["Router"]["category"]) == (
        "target",
        "toy_fp8__sm_100__tp4",
        "moe",
    )
    assert ops["RmsNorm"]["calls"] == ["torch.ops.trtllm.rms_norm", "native:fused_step"]
    assert ops["RmsNorm"]["archs"] == ["sm_100"] and ops["RmsNorm"]["has_cells"]
    assert ops["RmsNorm"]["tests"] == [
        f"{modeling_v2.MODELING_V2_TESTS_REL}/norm/test_modeling_v2_rms.py"
    ]
    assert ops["Router"]["tests"] == []
    # The target that imports an operation uses it, wherever the operation lives.
    used = [f"{_MV2}/models/toy_vl/toy_fp8__sm_100__tp4"]
    assert ops["Router"]["used_by"] == used and ops["RmsNorm"]["used_by"] == used


def test_a_list_runs_a_test_by_node_file_or_directory_never_by_a_similar_name(tmp_path):
    repo = _checkout(tmp_path)
    entries = test_lists.extract_test_lists(repo)
    accuracy_entry = entries[0]
    assert accuracy_entry["timeout_min"] == 90 and accuracy_entry["gpu_count"] == {
        "gte": 4,
        "lte": 4,
    }

    def listed(test: str) -> bool:
        return any(test_lists.lists(e["path"], e["node"], test, e["ignore"]) for e in entries)

    assert listed("tests/integration/defs/accuracy/modeling_v2/test_toy_vl.py")
    assert listed("accuracy/modeling_v2/test_toy_vl.py::TestToy::test_gsm8k")
    assert not listed("accuracy/modeling_v2/test_toy_vl.py::TestToy::test_mmlu")
    assert listed("unittest/_torch/modeling_v2/norm/test_modeling_v2_rms.py::test_cells")
    assert not listed("tests/unittest/_torch/models/test_toy_v3.py")
    # Every parametrization of a listed test; a listed parametrization of an asked test.
    assert listed("accuracy/test_param.py::TestP::test_a[tp8]")
    assert listed("accuracy/test_param.py::TestP::test_b")
    assert not listed("accuracy/test_param.py::TestP::test_ab")
    # A directory entry runs what it does not --ignore, and records its -k selection.
    assert listed("unittest/_torch/kv/test_pool.py")
    assert not listed("unittest/_torch/kv/test_skipped.py")
    assert entries[-1]["select"] == "not slow"
    plain, by_gpu = test_lists.extract_waives(repo)
    assert plain["reason"] == "SKIP (bug)" and not plain["gpu"]
    assert (by_gpu["gpu"], by_gpu["path"], by_gpu["node"]) == (
        "B200",
        "tests/integration/defs/accuracy/test_param.py",
        "TestP::test_a",
    )


def test_accuracy_recipes_and_environment_are_indexed(tmp_path):
    repo = _checkout(tmp_path)
    (task,) = accuracy.extract_accuracy_tasks(repo)
    assert task["dataset_dir"] == "$LLM_MODELS_ROOT/datasets/gsm8k" and task["num_samples"] == 1319
    assert accuracy.extract_references(repo)["gsm8k"]["org/Toy"] == [{"accuracy": 91.0}]
    ranked = recipes.rank_recipes(
        recipes.extract_recipes(repo), text="toy", gpu="b200", concurrency=48
    )
    assert [r["concurrency"] for r in ranked] == [64, 4] and ranked[0]["tp"] == 4
    assert env.extract_env_vars(repo) == [
        {"name": "TLLM_TOY_KNOB", "reads": ["tensorrt_llm/_torch/knob.py:2"]}
    ]


def _commit(repo: Path) -> str:
    if not (repo / ".git").exists():
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_a_pack_is_reused_until_the_checkout_changes_and_the_cli_answers_from_it(tmp_path, capsys):
    repo = _checkout(tmp_path / "repo")
    _commit(repo)
    out = tmp_path / "ws" / "wiki"
    manifest, rebuilt = pack.ensure(repo, out)
    assert rebuilt and manifest["counts"]["families"] == 1 and "ci_stages" in manifest["empty"]
    assert not pack.ensure(repo, out)[1]
    (repo / config.SERVE_SCHEMA_REL).write_text(json.dumps({**_SCHEMA, "title": "edited"}))
    assert pack.ensure(repo, out)[1]
    # An untracked family is a change of the facts; a file outside the sources is not.
    _write(
        repo / _MV2 / "models" / "new_family" / "routing.py",
        '@routes("NewForCausalLM")\ndef route(): ...\n',
    )
    assert pack.ensure(repo, out)[1] and Wiki(out).family_for("NewForCausalLM") == "new_family"
    _write(repo / "experiments" / "run" / "journal.md", "a run directory inside the checkout\n")
    assert not pack.ensure(repo, out)[1]

    wiki = Wiki(out)
    assert wiki.family_for("ToyForConditionalGeneration") == "toy_vl"
    page = wiki.feature("block_reuse")
    assert page["feature"] == "reuse"
    assert any(f.get("missing") for f in page["fields"])  # the toy schema lacks most of them
    assert cli.main(["--pack", str(out), "ci", "accuracy/modeling_v2/test_toy_vl.py"]) == 0
    assert "l0_b200 [post_merge pytorch *b200* 4-4 GPU]" in capsys.readouterr().out
    assert cli.main(["--pack", str(out), "family", "NoSuchForCausalLM"]) == cli.NOT_FOUND
    assert cli.main(["--pack", str(out), "ci", "accuracy/not_listed.py"]) == cli.NOT_FOUND
    assert "not in any test list" in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing:
        cli.main(["--pack", str(tmp_path / "nothing"), "family", "toy_vl"])
    assert missing.value.code == 1
    options = tmp_path / "options.yaml"
    options.write_text(yaml.safe_dump({"kv_cache_config": {"dtyp": "fp8"}}))
    assert cli.main(["--pack", str(out), "validate", str(options)]) == cli.NOT_FOUND
    assert "not a field: kv_cache_config.dtyp" in capsys.readouterr().out
    # The workflow's own entry points: its preflight step and `trtllm-modeling-v2 wiki`.
    workflow = SimpleNamespace(task_data={"trtllm_repo_path": str(repo)}, workspace=tmp_path / "ws")
    assert "reused" in bootstrap._take_wiki(workflow)
    assert flow_cli._wiki(["family", "toy_vl", "--workspace", str(tmp_path / "ws")]) == 0
    assert "toy_fp8__sm_100__tp4" in capsys.readouterr().out


def test_what_changed_since_a_commit_is_scoped_to_the_target_and_written_when_the_run_resumes(
    tmp_path, capsys
):
    repo = _checkout(tmp_path / "repo")
    before = _commit(repo)
    workspace = tmp_path / "ws"
    workflow = SimpleNamespace(task_data={"trtllm_repo_path": str(repo)}, workspace=workspace)
    target = f"{_MV2}/models/toy_vl/toy_fp8__sm_100__tp4"
    assert "built in" in bootstrap._take_wiki(workflow, target)

    schema = json.loads((repo / config.SERVE_SCHEMA_REL).read_text())
    schema["properties"]["max_seq_len"] = {"type": "integer"}
    (repo / config.SERVE_SCHEMA_REL).write_text(json.dumps(schema))
    _write(
        repo / _MV2 / "models/toy_vl/toy_fp8__sm_100__tp4/ops/moe/gate.py",
        _OP.format(name="Gate", call="gate"),
    )
    other = repo / _MV2 / "models" / "other"
    _write(other / "routing.py", '@routes("OtherForCausalLM")\ndef route(): ...\n')
    _write(other / "ops" / "mix.py", _OP.format(name="Mix", call="mix"))
    _write(
        repo / accuracy.REFERENCES_REL / "gsm8k.yaml",
        yaml.safe_dump({"org/Toy": [{"accuracy": 92.5}]}),
    )
    after = _commit(repo)

    old = tmp_path / "old"
    assert pack.build_at(repo, before, old)["commit"] == before
    assert (
        pack.build_at(repo, before[:10], old)["build_s"] == Wiki(old).manifest["build_s"]
    )  # cached
    new = tmp_path / "new"
    pack.ensure(repo, new)
    scoped = diff_packs(Wiki(old), Wiki(new), target="toy_vl/toy_fp8__sm_100__tp4")
    assert [f["path"] for f in scoped["fields"]["added"]] == ["max_seq_len"]
    assert [op["name"] for op in scoped["ops"]["added"]] == ["Gate"]
    assert scoped["families"] == []  # the other family is out of scope
    assert scoped["references"] == [
        {
            "benchmark": "gsm8k",
            "model": "org/Toy",
            "old": [{"accuracy": 91.0}],
            "new": [{"accuracy": 92.5}],
        }
    ]
    whole = diff_packs(Wiki(old), Wiki(new))
    assert {op["name"] for op in whole["ops"]["added"]} == {"Gate", "Mix"}
    assert whole["families"] == [{"family": "other", "status": "added"}]
    with pytest.raises(ValueError, match="no target"):
        diff_packs(Wiki(old), Wiki(new), target="toy_vl/missing")

    argv = ["--pack", str(new), "changes", "--since", before, "--target", "toy_fp8__sm_100__tp4"]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert (
        f"changes {before[:12]} -> {after[:12]}" in text and "+ Gate" in text and "Mix" not in text
    )
    assert (tmp_path / f"new-at-{before[:12]}" / "manifest.json").is_file()

    # A resumed run on the moved checkout keeps the previous pack and says what changed.
    note = bootstrap._take_wiki(workflow, target)
    assert "the checkout moved" in note and "`wiki/changes.md`" in note
    assert Wiki(workspace / "wiki-previous").manifest["commit"] == before
    assert "+ Gate" in (workspace / "wiki" / "changes.md").read_text()
    assert "moved" not in bootstrap._take_wiki(workflow, target)


@pytest.mark.skipif(
    not (_CHECKOUT / config.SERVE_SCHEMA_REL).is_file(), reason="not inside a TensorRT-LLM checkout"
)
def test_this_checkout_still_has_what_the_wiki_and_the_workflow_rely_on():
    """A renamed field, family layout or list format fails here, not quietly in a run."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.check_config import SPECULATIVE_ALGORITHMS

    extracted = config.extract_config_fields(_CHECKOUT)
    known = {f["path"] for f in extracted["fields"]}
    missing = {
        f"{key}: {path}"
        for key, spec in features.FEATURES.items()
        for path in spec.fields
        if path not in known
    }
    assert not missing, f"feature fields gone from {config.SERVE_SCHEMA_REL}: {sorted(missing)}"
    assert set(SPECULATIVE_ALGORITHMS.values()) <= set(extracted["decoding_types"])
    families = modeling_v2.extract_families(_CHECKOUT)
    assert families and all(f["architectures"] and f["targets"] for f in families)
    entries = test_lists.extract_test_lists(_CHECKOUT)
    for family in families:
        accuracy_test = family["accuracy_tests"]["path"]
        assert accuracy_test and any(
            test_lists.lists(e["path"], e["node"], accuracy_test) for e in entries
        )
    gsm8k = next(t for t in accuracy.extract_accuracy_tasks(_CHECKOUT) if t["task"] == "GSM8K")
    assert (
        gsm8k["num_samples"]
        and gsm8k["alpha"]
        and "gsm8k" in accuracy.extract_references(_CHECKOUT)
    )
