# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 task schema behavior and regression checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.run.contract import compile_contract
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments, derive, shorthand, task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

from .conftest import SCRATCH


def _valid_spec(tmp_path) -> dict:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    return {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "agents": {"defaults": {"backend": "claude-code", "model": "test-model"}},
        "environment": {
            "kind": "local",
            "python": "/usr/bin/python3",
            "env": {"TRTLLM_MODELING_V2": "require", "TMPDIR": "/scratch/tmp"},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
        },
    }


def _write(tmp_path, spec) -> str:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(path)


def test_every_problem_is_batched_into_one_message(tmp_path):
    spec = _valid_spec(tmp_path)
    del spec["arch"]
    del spec["environment"]
    spec["checkpoint_path"] = "/definitely/not/here"
    # A disabled gate's fields are still validated.
    spec["accuracy"]["serve"] = {"enabled": False, "max_tokens": True, "prompt": " "}
    spec["accuracy"]["score_pattern"] = "accuracy: [0-9.]+"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    for expected in (
        "arch",
        "environment",
        "non-existent path",
        "accuracy.serve.max_tokens",
        "accuracy.serve.prompt",
        "'accuracy.score_pattern' needs a group around the score",
    ):
        assert expected in message


def test_a_misspelled_key_is_refused_with_its_spelling_at_any_depth(tmp_path):
    # A misspelled `features` must not silently build core only.
    spec = _valid_spec(tmp_path) | {"feature": ["mtp"]}
    spec["accuracy"]["benchmrk"] = "x"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'feature'" in str(excinfo.value)
    assert "did you mean 'features'" in str(excinfo.value)
    assert "did you mean 'accuracy.benchmark'" in str(excinfo.value)


def test_source_restriction_is_not_a_supported_production_task_option(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["source_embargo"] = []
    with pytest.raises(task_schema.TaskSchemaError, match="unknown key 'source_embargo'"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    # The flow has no tuning stage: a perf block is refused with why.
    del spec["source_embargo"]
    spec["perf"] = {"enabled": True}
    with pytest.raises(task_schema.TaskSchemaError, match="'perf' is not read: the flow has no"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def _checkpoint(root: Path) -> Path:
    path = root / "newmodel"
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["NewModelForCausalLM"],
        "_name_or_path": "org/newmodel",
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_experts": 16,
        "max_position_embeddings": 32768,
        "num_nextn_predict_layers": 1,
        "vision_config": {"model_type": "vit"},
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


def _site(root: Path, **changes) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    site = {
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "device": {"arch": "sm_100"},
        "defaults": {"members": ["tp1", "tep4", "dep4"]},
        **changes,
    }
    path = root / "site.yaml"
    path.write_text(yaml.safe_dump(site))
    return path


def _derive(root: Path, **site) -> tuple[Path, derive.Derivation]:
    site_path = _site(root, **site)
    derivation = derive.derive_spec(
        _checkpoint(root), yaml.safe_load(site_path.read_text()), site_path=site_path
    )
    return derive.write_derivation(root / "ws", derivation), derivation


def _engine(root: Path, name: str) -> dict:
    return yaml.safe_load((root / "ws" / derive.ENGINES_DIRNAME / f"{name}.yaml").read_text())


def test_derive_writes_the_first_deployment_only_and_says_what_it_left_open(tmp_path):
    agents = {"defaults": {"backend": "claude-code", "model": "m"}}
    path, derivation = _derive(tmp_path, agents=agents, defaults={"members": ["tep4", "dep4"]})
    written = yaml.safe_load(path.read_text())
    # The smallest spec: one deployment, the site's first member, its options beside the spec.
    assert written["deployments"] == [{"name": "tep4", "options": "engines/tep4.yaml"}]
    assert not {"members", "features", "variants", "runtime", "parallel"} & set(written)
    # The checkpoint's towers and drafter are the operator's to add: the first serves text.
    assert written["scope"] == "text_only" and written["accuracy"] == {"model_key": "org/newmodel"}
    assert derivation.questions == []
    core = _engine(tmp_path, "tep4")
    assert (core["max_seq_len"], core["max_batch_size"]) == (5120, 64)
    assert core["tensor_parallel_size"] == 4 and core["speculative_config"] is None
    # Core's switches: graphs and the overlap scheduler on, reuse and chunked prefill off.
    assert core["kv_cache_config"]["enable_block_reuse"] is False
    assert core["enable_chunked_prefill"] is False and core["cuda_graph_config"] == {}
    spec = load_and_validate_task_yaml(path)
    [first] = deployments.resolve(spec)
    assert first.core and first.name == "tep4" and first.inputs == ("text",)
    # No benchmark states an empty mapping, and nothing of another deployment is in this one.
    assert "benchmarks" not in written["deployments"][0]
    assert first.benchmarks[0].discriminators in (None, {})

    # What it cannot know is left open, one line each, and the spec is written all the same.
    checkpoint = _checkpoint(tmp_path)
    config = json.loads((checkpoint / "config.json").read_text())
    del config["_name_or_path"]
    config["quantization_config"] = {"quant_algo": "NVFP4"}
    config["max_position_embeddings"] = 2048
    (checkpoint / "config.json").write_text(json.dumps(config))
    site = yaml.safe_load(_site(tmp_path, device={}).read_text())
    derivation = derive.derive_spec(checkpoint, site)
    assert [q.partition(":")[0] for q in derivation.questions] == [
        "arch",
        "agents.defaults.model",
        "accuracy.model_key",
    ]
    assert "arch" not in derivation.spec and "accuracy" not in derivation.spec
    assert "leave `accuracy` out to onboard on boot only" in derivation.questions[2]
    # A default workload is fitted to what the model reaches.
    assert (derivation.deployment, derivation.options["max_seq_len"]) == ("tp1", 2048)
    # A quantized checkpoint is anchored to its own row, not the unquantized model's.
    config["_name_or_path"] = "org/newmodel"
    (checkpoint / "config.json").write_text(json.dumps(config))
    spec = derive.derive_spec(checkpoint, site).spec
    assert spec["accuracy"]["discriminators"] == {"quant_algo": "NVFP4"}
    with pytest.raises(derive.DeriveError, match="must state `trtllm_repo_path`"):
        derive.derive_spec(checkpoint, {"environment": {}})


def test_the_shorthand_is_read_once_at_load_and_a_loaded_spec_states_its_deployments(
    tmp_path, monkeypatch
):
    spec = _valid_spec(tmp_path) | {
        "parallel": "unified",
        "members": ["tp1", "tep2"],
        "features": ["mtp"],
        "variants": {"kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}},
        "runtime": {"extra_flags": ["--max_batch_size=32"]},
    }
    spec["accuracy"] |= {
        "model_key": {"default": "org/toy", "mtp": "org/toy-mtp"},
        "max_seq_len": 4096,
    }
    asked = shorthand.expand(spec)
    loaded = load_and_validate_task_yaml(_write(tmp_path, spec))
    # Below the loader there is one representation: the shorthand's keys are spent.
    assert not {"members", "features", "variants", "runtime"} & set(loaded)
    assert set(loaded["accuracy"]) == {"benchmark", "reference_yaml"}
    assert [d["name"] for d in loaded["deployments"]] == ["tp1", "tep2", "mtp", "kv_fp8"]
    monkeypatch.setattr(shorthand, "expand", lambda *a: pytest.fail("expanded below the loader"))
    # What the loaded spec states is what the shorthand asked for, to the last option.
    assert deployments.resolve(loaded) == asked
    found = {d.name: d for d in deployments.resolve(loaded)}
    assert list(found) == ["tp1", "tep2", "mtp", "kv_fp8"] and found["tp1"].core
    assert found["tp1"].server.options["max_batch_size"] == 32
    assert found["tp1"].server.options["max_seq_len"] == 4096
    assert found["kv_fp8"].server.options["kv_cache_config"]["dtype"] == "fp8"
    assert (found["mtp"].relative_to, found["tep2"].ranks) == ("tp1", 2)
    # What was stated per deployment is each benchmark's own.
    assert found["tp1"].benchmarks[0].model_key == "org/toy"
    assert found["mtp"].benchmarks[0].model_key == "org/toy-mtp"


@pytest.mark.parametrize("form", ["parallel", "members", "deployments"])
def test_accuracy_is_optional_on_every_form_of_the_spec(tmp_path, form):
    spec = _valid_spec(tmp_path)
    del spec["accuracy"]
    if form == "members":
        spec |= {"parallel": "unified", "members": ["tp1", "tep2"], "features": ["mtp"]}
    elif form == "deployments":
        del spec["parallel"]
        spec["deployments"] = [{"name": "tp1", "options": {}}, {"name": "kv", "options": {}}]
    loaded = load_and_validate_task_yaml(_write(tmp_path, spec))
    found = deployments.resolve(loaded)
    # Nothing scores them: each owes its boot, and the run onboards on the first one's.
    assert found[0].name == "tp1" and all(d.benchmarks == () for d in found)
    assert [c.gate for c in compile_contract(loaded).checks] == ["boot"] * len(found)


def test_a_key_this_version_no_longer_reads_is_dropped_with_one_notice_each(tmp_path):
    spec = _valid_spec(tmp_path)
    quiet: list[str] = []
    load_and_validate_task_yaml(_write(tmp_path, spec), quiet)
    assert quiet == []
    spec["accuracy"]["envelope"] = {"enabled": True, "probes": ["1, 2, 3,"]}
    spec["accuracy"]["reference_logits"] = {"path": "reference.json"}
    spec["commit_policy"] = {"allowed_paths": ["tensorrt_llm/"], "base_ref": "main"}
    notices: list[str] = []
    resolved = load_and_validate_task_yaml(_write(tmp_path, spec), notices)
    assert "envelope" not in resolved["accuracy"] and "reference_logits" not in resolved["accuracy"]
    assert len(notices) == 3
    assert notices[0].startswith("'accuracy.envelope' is ignored: a boot runs its workload only")
    assert notices[1].startswith("'accuracy.reference_logits' is ignored: ")
    # No path is refused by the spec any more: the key is accepted, said, and read no further.
    assert notices[2].startswith("'commit_policy.allowed_paths' is ignored: no path is refused")
    assert resolved["commit_policy"] == {"base_ref": "main"}
    # Without a list to fill, the spec loads the same, silently.
    assert load_and_validate_task_yaml(_write(tmp_path, spec)) == resolved


def test_each_deployment_owes_its_boot_its_benchmarks_and_its_tests_and_core_the_serve_gate(
    tmp_path,
):
    spec = _valid_spec(tmp_path) | {
        "parallel": "unified",
        "members": ["tep4", "dep4"],
        "features": ["mtp", "disagg"],
        "variants": {
            "mtp7": {
                "unit": "mtp",
                "gates": ["boot", "acceptance"],
                "llm_api_options": {"speculative_config": {"max_draft_len": 7}},
                "accuracy": {"acceptance": {"min_absolute": 2.0}},
            }
        },
    }
    spec["accuracy"]["acceptance"] = {"num_prompts": 32, "mtp": {"min_absolute": 1.5}}
    spec["accuracy"]["disagg"] = {"test": "tests/test_pd.py::TestM::test_hop"}
    notices: list[str] = []
    resolved = load_and_validate_task_yaml(_write(tmp_path, spec), notices)
    assert sorted(n.split(" is ignored")[0] for n in notices) == [
        "'accuracy.acceptance'",
        "'variants.mtp7.accuracy.acceptance'",
        "'variants.mtp7.gates'",
    ]
    assert any("judged by a native test" in n for n in notices)
    contract = compile_contract(resolved)
    assert [c.key for c in contract.checks] == [
        "tep4:boot",
        "tep4:accuracy[gsm8k]",
        "dep4:boot",
        "dep4:accuracy[gsm8k]",
        "mtp:boot",
        "mtp:accuracy[gsm8k]",
        "disagg:boot",
        "disagg:accuracy[gsm8k]",
        "disagg:tests",
        "mtp7:boot",
        "mtp7:accuracy[gsm8k]",
    ]
    assert any(
        "mtp: its drafter's acceptance is judged by no check here" in n for n in contract.notes
    )
    # The serve gate is core's, where the spec turns it on.
    resolved["accuracy"]["serve"] = {"enabled": True}
    more = {c.key for c in compile_contract(resolved).checks} - {c.key for c in contract.checks}
    assert more == {"tep4:serve"}
    # A spec with no accuracy block owes boots alone, and says so.
    bare = compile_contract({"deployments": [{"name": "only", "options": {}}]})
    assert [c.key for c in bare.checks] == ["only:boot"]
    assert bare.notes == ("only: it names no benchmark, so its boot alone judges it",)
