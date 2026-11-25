# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run's tests (``verification``): the flow's, changed by the spec, and the spec's own.

Schema, contract, preflight, the generated e2e test and how each kind of test runs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import e2e, user_tests
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.git import CommitPolicy, audit_commits
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec import verification as verification_mod

from ._helpers import e2e_answer

LONG = "tests/x/test_a.py::TestA::test_long[{member}]"
SPARSE = "sparse_topk_exceeded"
ROUTING = "tests/x/test_b.py"
PASS = '<testsuite><testcase classname="TestA" name="test_long"/></testsuite>'
FAMILY = "qwen3"
TARGET = "tensorrt_llm/_torch/modeling_v2/models/qwen3/qwen3_8b__sm_103__unified"


def _spec(tmp_path, tests=None, **extra) -> dict:
    checkpoint, repo = tmp_path / "models" / "Qwen3-8B", tmp_path / "repo"
    checkpoint.mkdir(parents=True, exist_ok=True)
    repo.mkdir(exist_ok=True)
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "unified",
        "members": ["tep4", "dep4"],
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "local",
            "python": "/usr/bin/python3",
            "env": {
                "TRTLLM_MODELING_V2": "require",
                "TMPDIR": "/scratch/tmp",
                "LLM_MODELS_ROOT": str(tmp_path / "models"),
            },
        },
        "accuracy": {
            "reference_yaml": "x.yaml",
            "model_key": "Qwen/Qwen3-8B",
        },
    }
    if tests is not None:
        spec["verification"] = tests
    spec.update(extra)
    return spec


def _load(tmp_path, spec) -> dict:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_schema.load_and_validate_task_yaml(path)


def _tests():
    return {
        "core": {
            "long_ctx": {
                "purpose": "a 32k-token prompt is answered as at 1k",
                "pytest": LONG,
                "members": "all",
                "stage": "delivery",
            },
            "routing": {
                "purpose": "the router picks the reference's experts",
                "pytest": ROUTING,
                "devices": 2,
            },
        }
    }


def _contract(data) -> dict[str, contract_mod.Check]:
    return {c.key: c for c in contract_mod.compile_contract(data, world_size=4).checks}


# ----------------------------------------------------------------------------- the contract


def test_the_flow_derives_its_tests_and_the_spec_adds_its_own(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    contract = contract_mod.compile_contract(data, world_size=4)
    checks = {c.key: c for c in contract.checks}
    # The flow's own: one e2e test per member, its boot prompts and benchmark on one engine.
    for member in ("tep4", "dep4"):
        e2e_check = checks[f"core:e2e@{member}"]
        assert e2e_check.stage == contract_mod.STAGE_LANDING and e2e_check.scores
        assert e2e_check.origin == verification_mod.ORIGIN_FLOW
        assert e2e_check.runner == verification_mod.RUN_DATASETS
    # The spec's own: a whole-model test at every member at its stage, a module test once.
    assert checks["core:long_ctx@tep4"].stage == contract_mod.STAGE_DELIVERY
    assert checks["core:long_ctx@dep4"].origin == verification_mod.ORIGIN_SPEC
    assert checks["core:routing@tp2"].stage == contract_mod.STAGE_LANDING
    assert {c.key for c in contract.by_stage(contract_mod.STAGE_DELIVERY)} >= {
        "core:long_ctx@tep4",
        "core:long_ctx@dep4",
    }
    assert any("`long_ctx`, the spec's own" in note for note in contract.notes)
    assert "| from |" in contract.render() and "| spec |" in contract.render()
    assert all(c.inputs for c in contract.checks)
    tests = {(t.feature, t.name): t for t in contract_mod.run_tests(data, world_size=4)}
    assert tests[("core", "e2e")].datasets == ("generate", "gsm8k")


def test_an_entry_named_like_a_flow_test_changes_it_and_enabled_false_drops_it(tmp_path):
    default = _contract(_load(tmp_path, _spec(tmp_path)))
    tests = {"core": {"e2e": {"datasets": ["generate", "gsm8k", "mmlu"], "stage": "delivery"}}}
    changed = _contract(_load(tmp_path, _spec(tmp_path, tests)))
    check = changed["core:e2e@tep4"]
    assert check.stage == contract_mod.STAGE_DELIVERY
    assert check.origin == verification_mod.ORIGIN_CHANGED
    # What the test runs is part of what its check reads: the change runs it again.
    assert check.inputs != default["core:e2e@tep4"].inputs
    # Brought by a test of the tree instead, the e2e check runs that test.
    tests = {"core": {"e2e": {"pytest": LONG}}}
    swapped = _contract(_load(tmp_path, _spec(tmp_path, tests)))["core:e2e@tep4"]
    assert swapped.runner == verification_mod.RUN_PYTEST and not swapped.scores
    # Turned off, it is not owed.
    off = _contract(_load(tmp_path, _spec(tmp_path, {"core": {"e2e": {"enabled": False}}})))
    assert not any(key.startswith("core:e2e") for key in off)


@pytest.mark.parametrize(
    "tests, says",
    [
        # A flow test the profile does not derive is not there to change or turn off.
        ({"core": {"serve": {"stage": "delivery"}}}, "changes a test the flow does not derive"),
        ({"core": {"serve": {"enabled": False}}}, "turns off a test the flow does not derive"),
        # trtllm-serve sends the boot prompts only.
        ({"core": {"e2e": {"serve": True}}}, "its datasets are [generate]"),
        # A test's own settings need what they apply to.
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "config": {"max_seq_len": 8}}}},
            "this test has none",
        ),
        (
            {
                "core": {
                    "t": {
                        "purpose": "p",
                        "datasets": ["generate"],
                        "serve": True,
                        "extra_checks": [SPARSE],
                    }
                }
            },  # fmt: skip
            "this test runs none there",
        ),
        (
            {
                "core": {
                    "t": {
                        "purpose": "p",
                        "datasets": ["gsm8k"],
                        "extra_checks": ["gsm8k"],
                    }
                }
            },  # fmt: skip
            "named like its datasets",
        ),
        ({"core": {"e2e": {"datasets": ["generate"], "evaluator_kwargs": {"x": 1}}}}, "runs none"),
    ],
)
def test_an_entry_that_finds_no_flow_test_or_cannot_run_is_refused_by_the_contract(
    tmp_path, tests, says
):
    data = _load(tmp_path, _spec(tmp_path, tests))
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        contract_mod.compile_contract(data, world_size=4)
    assert says in str(excinfo.value)


def test_a_spec_test_is_owed_whatever_the_profile(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests(), profile="bringup"))
    keys = set(_contract(data))
    assert {"core:long_ctx@tep4", "core:long_ctx@dep4", "core:routing@tp2"} <= keys
    # The bringup profile gates the flow's own at the cheapest member only.
    assert [k for k in keys if k.startswith("core:e2e")] == ["core:e2e@tep4"]


def test_a_changed_entry_changes_the_check_and_its_purpose_does_not(tmp_path):
    tests = _tests()
    tests["core"]["long_ctx"]["members"] = ["tep4"]
    first = _contract(_load(tmp_path, _spec(tmp_path, tests)))
    assert "core:long_ctx@tep4" in first and "core:long_ctx@dep4" not in first
    tests["core"]["long_ctx"]["purpose"] = "another purpose"
    reworded = _contract(_load(tmp_path, _spec(tmp_path, tests)))
    assert reworded["core:long_ctx@tep4"].inputs == first["core:long_ctx@tep4"].inputs
    tests["core"]["long_ctx"]["timeout"] = 1200
    second = _contract(_load(tmp_path, _spec(tmp_path, tests)))
    assert first["core:long_ctx@tep4"].inputs != second["core:long_ctx@tep4"].inputs
    # The flow's own checks keep their verdicts.
    assert first["core:e2e@tep4"].inputs == second["core:e2e@tep4"].inputs


def test_the_bring_up_holds_the_flows_own_engine_checks_only(tmp_path):
    tests = _tests()
    tests["core"]["routing"]["stage"] = "landing"
    tests["core"]["extra"] = {"purpose": "MMLU too", "datasets": ["mmlu"], "members": ["tep4"]}
    data = _load(tmp_path, _spec(tmp_path, tests))
    contract = contract_mod.compile_contract(data, world_size=4)
    owed = contract_mod.bringup_checks(contract)
    assert [c.key for c in owed] == ["core:e2e@tep4"]


def test_the_same_name_under_two_features_is_two_tests(tmp_path):
    spec = _spec(tmp_path, {"core": {"t": {"purpose": "p", "pytest": ROUTING}}})
    spec["features"] = ["mtp"]
    spec["verification"]["mtp"] = {"t": {"purpose": "p", "pytest": ROUTING}}
    keys = set(_contract(_load(tmp_path, spec)))
    assert {"core:t@tp1", "mtp:t@tp1"} <= keys


# ----------------------------------------------------------------------------- the schema


@pytest.mark.parametrize(
    "tests, says",
    [
        ({"nope": {"t": {"purpose": "p", "pytest": ROUTING}}}, "names no feature"),
        ({"core": {"t": {"pytest": ROUTING}}}, "purpose"),
        ({"core": {"t": {"purpose": "p"}}}, "must say how it runs"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "datasets": ["gsm8k"]}}}, "both"),
        ({"core": {"t": {"purpose": "p", "datasets": ["lambada"]}}}, "not in the catalog"),
        ({"core": {"t": {"purpose": "p", "datasets": ["gsm8k", "gsm8k"]}}}, "twice"),
        ({"core": {"t": {"purpose": "p", "datasets": ["gsm8k"], "devices": 1}}}, "module test"),
        ({"core": {"t": {"purpose": "p", "datasets": ["gsm8k"], "serve": True}}}, "[generate]"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "serve": True}}}, "trtllm-serve"),
        ({"core": {"e2e": {"enabled": "no"}}}, "enabled"),
        ({"core": {"t": {"purpose": "p", "pytest": "/abs/test_x.py"}}}, "repository-relative"),
        ({"core": {"t": {"purpose": "p", "pytest": "tests/x/README.md"}}}, "repository-relative"),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "members": ["tp1"], "devices": 1}}},
            "both 'members'",
        ),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "stage": "nightly"}}}, "stage"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "members": ["xx4"]}}}, "members"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "members": ["unified"]}}}, "concrete"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "devices": 0}}}, "devices"),
        ({"core": {"t": {"purpose": "p", "pytest": LONG}}}, "{member}"),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "env": {"MV2FLOW_MODEL": "x"}}}},
            "set by the flow",
        ),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "run": "pytest"}}},
            "not a key of a test",
        ),
        ({"core": {"t:x": {"purpose": "p", "pytest": ROUTING}}}, "name"),
        (
            {
                "core": {
                    "t": {
                        "purpose": "p",
                        "datasets": ["gsm8k"],
                        "config": {"tensor_parallel_size": 2},
                    }
                }
            },
            "decided by the member and the feature",
        ),
        (
            {
                "core": {
                    "t": {
                        "purpose": "p",
                        "datasets": ["gsm8k"],
                        "config": {"speculative_config": {}},
                    }
                }
            },
            "a configuration the run delivers is a variant",
        ),
        ({"core": {"t": {"purpose": "p", "datasets": ["gsm8k"], "config": {}}}}, "LLM-API options"),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "evaluator_kwargs": {"x": 1}}}},
            "a pytest test runs none",
        ),
        # An extra check is named by its function, takes no arguments, and judges datasets.
        (
            {
                "core": {
                    "t": {
                        "purpose": "p",
                        "datasets": ["gsm8k"],
                        "extra_checks": {SPARSE: {"tokens": 8}},
                    }
                }
            },  # fmt: skip
            "nonempty list of extra checks' names",
        ),
        ({"core": {"t": {"purpose": "p", "datasets": ["gsm8k"], "extra_checks": []}}}, "nonempty"),
        (
            {
                "core": {
                    "t": {
                        "purpose": "p",
                        "datasets": ["gsm8k"],
                        "extra_checks": ["tests/x/c.py::f"],
                    }
                }
            },  # fmt: skip
            "named by its function",
        ),
        (
            {"core": {"t": {"purpose": "p", "datasets": ["gsm8k"], "extra_checks": ["_helper"]}}},
            "named by its function",
        ),
        (
            {
                "core": {
                    "t": {"purpose": "p", "datasets": ["gsm8k"], "extra_checks": [SPARSE, SPARSE]}
                }
            },
            "twice",
        ),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "extra_checks": [SPARSE]}}},
            "asserts in its own code",
        ),
        (
            {
                "core": {"t": {"purpose": "p", "pytest": ROUTING}},
                "default": {"t": {"purpose": "p", "pytest": ROUTING}},
            },
            "names no feature",
        ),
        ({"tests": {"core": {"u": {"purpose": "p", "pytest": ROUTING}}}}, "names no feature"),
    ],
)
def test_an_entry_the_schema_refuses(tmp_path, tests, says):
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        _load(tmp_path, _spec(tmp_path, tests))
    assert says in str(excinfo.value)


def test_an_entry_of_a_flow_tests_name_needs_neither_purpose_nor_a_way_to_run(tmp_path):
    _load(tmp_path, _spec(tmp_path, {"core": {"e2e": {"stage": "delivery"}}}))
    _load(tmp_path, _spec(tmp_path, {"core": {"boot": {"enabled": False}}}))


# ----------------------------------------------------------------------------- preflight


def _tree(repo: Path) -> None:
    (repo / "tests" / "x").mkdir(parents=True, exist_ok=True)
    (repo / "tests" / "x" / "test_a.py").write_text(
        "class TestA:\n    def test_long(self):\n        pass\n", encoding="utf-8"
    )
    (repo / "tests" / "x" / "test_b.py").write_text("def test_route():\n    pass\n")
    runner = repo / e2e.RUNNER_MODULE
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("", encoding="utf-8")


def _row(repo: Path, dataset: str, score: float, model: str = "Qwen/Qwen3-8B") -> None:
    path = repo / e2e.REFERENCES_DIR / f"{dataset}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({model: [{"accuracy": score}]}), encoding="utf-8")


def test_preflight_holds_each_test_to_what_the_tree_holds(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    repo = Path(data["trtllm_repo_path"])
    # A tree with no runner of the generated e2e test predates it.
    with pytest.raises(preflight.PreflightError, match="this tree predates it"):
        preflight.check_tests(data, 4)
    _tree(repo)
    notes = preflight.check_tests(data, 4)
    assert any("`core.long_ctx`" in note and "delivery" in note for note in notes)
    tests = _tests()
    tests["core"]["long_ctx"]["pytest"] = "tests/x/test_a.py::TestA::test_gone[{member}]"
    data = _load(tmp_path, _spec(tmp_path, tests))
    with pytest.raises(preflight.SpecInconsistencyError, match="defines no `TestA::test_gone`"):
        preflight.check_tests(data, 4)
    # A benchmark beyond the feature's own needs a references row for the model.
    data = _load(tmp_path, _spec(tmp_path, {"core": {"e2e": {"datasets": ["gsm8k", "mmlu"]}}}))
    with pytest.raises(preflight.SpecInconsistencyError, match="mmlu has no references row"):
        preflight.check_tests(data, 4)
    _row(repo, "mmlu", 70.0)
    assert any("mmlu references row 70" in note for note in preflight.check_tests(data, 4))


def _hooks(repo: Path) -> tuple[str, str]:
    """A tree with a shared extra-checks file and the target's own; their paths."""
    target_file, shared_file = e2e.extra_check_files(FAMILY, TARGET)
    for path, text in (
        (shared_file, "def shared(result):\n    return ''\n\n\ndef both(result):\n    return ''\n\n\n"
                      "def takes_args(result, *, tokens):\n    return ''\n"),
        (target_file, "def own(result):\n    return ''\n\n\ndef both(result):\n    return ''\n"),
    ):  # fmt: skip
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    return target_file, shared_file


def test_an_extra_check_is_found_by_name_in_the_targets_file_then_the_shared_one(tmp_path):
    target_file, shared_file = _hooks(tmp_path)
    found, problems = e2e.resolve_extra_checks(
        tmp_path, FAMILY, TARGET, ["own", "shared", "both", "takes_args", "gone"]
    )
    assert found == {"own": f"{target_file}::own", "shared": f"{shared_file}::shared"}
    assert any("both is defined in" in p and "one name, one check" in p for p in problems)
    assert any("takes_args is not f(result)" in p for p in problems)
    assert any("gone is defined in none of" in p for p in problems)


def test_preflight_holds_an_extra_check_to_what_the_tree_defines(tmp_path):
    tests = {"core": {"sparse": {"purpose": "p", "datasets": ["gsm8k"], "extra_checks": ["gone"]}}}
    data = _load(tmp_path, _spec(tmp_path, tests))
    repo = Path(data["trtllm_repo_path"])
    _tree(repo)
    target_file, _ = _hooks(repo)
    with pytest.raises(
        preflight.SpecInconsistencyError, match="extra check gone is defined in none"
    ):
        preflight.check_tests(data, 4, family=FAMILY, target_rel=TARGET)
    tests["core"]["sparse"]["extra_checks"] = ["own"]
    data = _load(tmp_path, _spec(tmp_path, tests))
    notes = preflight.check_tests(data, 4, family=FAMILY, target_rel=TARGET)
    assert f"test `core.sparse`: extra check {target_file}::own" in notes


def test_the_trees_shared_extra_checks_are_found_by_name():
    repo = Path(__file__).resolve().parents[4]
    found, problems = e2e.resolve_extra_checks(repo, "no_family", "x/no_target", [SPARSE])
    assert found == {SPARSE: f"{e2e.SHARED_EXTRA_CHECKS}::{SPARSE}"}, problems


def test_a_speculators_benchmark_needs_the_row_its_engine_is_matched_to(tmp_path):
    spec = _spec(tmp_path, features=["mtp"], members=["tep4"])
    spec["accuracy"]["reference_yaml"] = f"{e2e.REFERENCES_DIR}/gsm8k.yaml"
    data = _load(tmp_path, spec)
    repo = Path(data["trtllm_repo_path"])
    _row(repo, "gsm8k", 80.0)  # a row without spec_dec_algo, and no other
    assert accuracy_checks.reference_discriminators(data, "core") is None
    assert accuracy_checks.reference_discriminators(data, "mtp") == {"spec_dec_algo": "MTP"}
    # accuracy_core matches the engine's spec_dec_algo, so preflight does too.
    with pytest.raises(preflight.PreflightError, match="unit `mtp`") as excinfo:
        preflight.check_accuracy_anchors(data)
    assert "The row needs spec_dec_algo: MTP" in str(excinfo.value)
    rows = [{"accuracy": 80.0}, {"spec_dec_algo": "MTP", "accuracy": 79.0}]
    (repo / e2e.REFERENCES_DIR / "gsm8k.yaml").write_text(yaml.safe_dump({"Qwen/Qwen3-8B": rows}))
    preflight.check_accuracy_anchors(data)
    # The check reads the same row its runner is judged by.
    runner = CheckRunner(_workflow(tmp_path, data))
    assert runner._reference_row("mtp", data, "gsm8k") == 79.0
    assert runner._reference_row("core", data, "gsm8k") == 80.0
    # A spec that names another spec_dec_algo than the engine runs is refused.
    spec["accuracy"]["discriminators"] = {"mtp": {"spec_dec_algo": "Eagle3"}}
    with pytest.raises(preflight.PreflightError, match="but its engine runs 'MTP'"):
        preflight.check_accuracy_anchors(_load(tmp_path, spec))


def test_verification_defined_follows_a_base_class_of_the_tree(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "base.py").write_text(
        "class Base:\n    def test_x(self):\n        pass\n"
    )
    (tmp_path / "tests" / "test_c.py").write_text(
        "from tests.base import Base\n\nclass TestC(Base):\n    pass\n"
    )
    there, _ = verification_mod.defined(tmp_path, "tests/test_c.py::TestC::test_x[tep4]")
    assert there is True
    there, said = verification_mod.defined(tmp_path, "tests/test_c.py::TestC::test_y")
    assert there is False and "test_y" in said


# ----------------------------------------------------------------------------- the e2e test


def _environment(world_size: int = 4) -> RunEnvironment:
    return RunEnvironment(kind="local", python="python3", world_size=world_size)


def _test(data, feature: str, name: str) -> verification_mod.Test:
    return next(
        t
        for t in contract_mod.run_tests(data, world_size=4)
        if (t.feature, t.name) == (feature, name)
    )


def test_a_case_is_the_features_configuration_at_the_member_and_its_datasets(tmp_path):
    spec = _spec(tmp_path, features=["mtp"])
    spec["accuracy"]["acceptance"] = {"mtp": {"min_absolute": 1.8, "floor_source": "ref drafter"}}
    spec["accuracy"]["acceptance"]["prompts"] = ["Count to ten."] * 4
    spec["accuracy"]["evaluator_kwargs"] = {"apply_chat_template": True}
    data = _load(tmp_path, spec)
    case = e2e.build_case(_environment(), data, _test(data, "mtp", "e2e"), "dep4", ["mtp"])
    assert case["id"] == "mtp-e2e-dep4" and case["devices"] == 4
    llm = case["llm"]
    assert llm["modeling_v2"] == "require" and "model" not in llm
    assert llm["enable_attention_dp"] and llm["speculative_config"]["decoding_type"] == "MTP"
    # A path under the models root is written for CI's own root.
    assert llm["speculative_config"]["speculative_model"] == "${LLM_MODELS_ROOT}/Qwen3-8B"
    generate, gsm8k, acceptance = case["datasets"]
    assert generate["kind"] == "generate" and generate["prompts"] and not generate["reuse_probe"]
    assert gsm8k == {
        "name": "gsm8k",
        "kind": "benchmark",
        "task": "GSM8K",
        "extra_acc_spec": None,
        "evaluator_kwargs": {"apply_chat_template": True},
    }
    assert acceptance["floor"] == 1.8 and acceptance["floor_source"] == "ref drafter"
    assert acceptance["workload"]["prompts"] == ["Count to ten."] * 4
    document = e2e.build_cases(data, "Qwen/Qwen3-8B", [case])
    assert document["model_dir"] == "${LLM_MODELS_ROOT}/Qwen3-8B" and document["version"] == 1


def test_a_tests_own_settings_lay_over_its_features_and_change_what_it_reads(tmp_path):
    sparse = {
        "purpose": "past the TopK budget the indexer runs and LongBench v2 holds",
        "datasets": ["longbench_v2"],
        "config": {
            "max_seq_len": 131072,
            "enable_chunked_prefill": True,
            "kv_cache_config": {"free_gpu_memory_fraction": 0.5},
        },
        "evaluator_kwargs": {"num_fewshot": 20},
        "extra_checks": [SPARSE],
    }
    spec = _spec(tmp_path, {"core": {"sparse": sparse}})
    spec["accuracy"].update(max_seq_len=5120, evaluator_kwargs={"apply_chat_template": True})
    data = _load(tmp_path, spec)
    test = _test(data, "core", "sparse")
    assert test.config["max_seq_len"] == 131072 and test.extra_checks == (SPARSE,)
    with pytest.raises(ValueError, match="were not looked up in the tree"):
        e2e.build_case(_environment(), data, test, "tep4", [])
    node = f"{e2e.SHARED_EXTRA_CHECKS}::{SPARSE}"
    case = e2e.build_case(_environment(), data, test, "tep4", [], {SPARSE: node})
    llm = case["llm"]
    # The test's options are the engine's last word, over accuracy.max_seq_len too; a nested
    # block keeps the feature's other keys.
    assert llm["max_seq_len"] == 131072 and llm["enable_chunked_prefill"] is True
    assert llm["kv_cache_config"] == {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.5}
    assert llm["tensor_parallel_size"] == 4 and llm["modeling_v2"] == "require"
    (longbench,) = case["datasets"]
    assert longbench["evaluator_kwargs"] == {"apply_chat_template": True, "num_fewshot": 20}
    assert case["extra_checks"] == [{"check": node}]
    # The flow's e2e, which sets none of them, is built as before.
    e2e_case = e2e.build_case(_environment(), data, _test(data, "core", "e2e"), "tep4", [])
    assert e2e_case["llm"]["max_seq_len"] == 5120
    assert "extra_checks" not in e2e_case
    # A test's settings are part of what its check reads.
    plain = _contract(_load(tmp_path, _spec(tmp_path, {"core": {"sparse": {
        k: v for k, v in sparse.items() if k != "config"}}})))  # fmt: skip
    assert plain["core:sparse@tep4"].inputs != _contract(data)["core:sparse@tep4"].inputs


def test_a_whole_model_pytest_test_is_handed_its_own_config_too(tmp_path):
    tests = {"core": {"long": {"purpose": "p", "pytest": LONG, "members": ["tep4"],
                               "config": {"max_seq_len": 65536}}}}  # fmt: skip
    data = _load(tmp_path, _spec(tmp_path, tests))
    work = tmp_path / "gate"
    handed = user_tests.handover(
        _environment(), data, _test(data, "core", "long"), member="tep4", features=[], workdir=work
    )
    options = yaml.safe_load(Path(handed[verification_mod.LLM_OPTIONS_ENV]).read_text())
    assert options["max_seq_len"] == 65536 and options["tensor_parallel_size"] == 4


def test_a_test_another_measures_on_the_same_engine_is_answered_by_it(tmp_path):
    tests = {
        "mtp": {
            "mtp_gsm8k": {"purpose": "p", "datasets": ["gsm8k"], "stage": "delivery"},
            "longer": {"purpose": "p", "datasets": ["gsm8k"], "config": {"max_seq_len": 9000}},
        },
        "core": {"again": {"purpose": "p", "datasets": ["generate", "gsm8k"]}},
    }
    contract = contract_mod.compile_contract(
        _load(tmp_path, _spec(tmp_path, tests, features=["mtp"], members=["tep4"])), world_size=4
    )
    checks = {c.key: c for c in contract.checks}
    # The same feature, member and settings, its datasets among the e2e's: one run answers both.
    assert checks["mtp:mtp_gsm8k@tep4"].answered_by == "mtp:e2e@tep4"
    assert any(n.startswith("mtp:mtp_gsm8k@tep4: answered by mtp:e2e@tep4") for n in contract.notes)
    # Other settings measure something else.
    assert checks["mtp:longer@tep4"].answered_by == ""
    # Of two that measure the same, the first answers the second, and is answered by none.
    assert checks["core:again@tep4"].answered_by == "core:e2e@tep4"
    assert checks["core:e2e@tep4"].answered_by == "" and checks["mtp:e2e@tep4"].answered_by == ""
    written = {f"{c['unit']}:{c['gate']}@{c['member']}": c for c in contract.as_dict()["checks"]}
    assert written["mtp:mtp_gsm8k@tep4"]["answered_by"] == "mtp:e2e@tep4"
    assert "answered_by" not in written["mtp:e2e@tep4"]


def test_a_speculators_e2e_screens_its_bases_at_the_same_member(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, features=["mtp"]))
    contract = contract_mod.compile_contract(data, world_size=4)
    checks = {c.key: c for c in contract.checks}
    for member in ("tep4", "dep4"):
        assert checks[f"core:e2e@{member}"].screened_by == f"mtp:e2e@{member}"
        assert checks[f"mtp:e2e@{member}"].screened_by == ""
    assert any(n.startswith("core:e2e: at a landing that names it") for n in contract.notes)
    # Not where the two measure differently.
    tests = {"core": {"e2e": {"config": {"max_seq_len": 9000}}}}
    changed = _contract(_load(tmp_path, _spec(tmp_path, tests, features=["mtp"])))
    assert changed["core:e2e@tep4"].screened_by == ""
    # Not for a speculator whose acceptance is relaxed: its answers are not the target's.
    options = tmp_path / "mtp.yaml"
    options.write_text(yaml.safe_dump(
        {"speculative_config": {"use_relaxed_acceptance_for_thinking": True}}))  # fmt: skip
    spec = _spec(tmp_path, features=["mtp"])
    spec["accuracy"]["llm_api_options"] = {"mtp": str(options)}
    relaxed = _contract(_load(tmp_path, spec))
    assert relaxed["core:e2e@tep4"].screened_by == ""


def test_a_relative_variant_is_judged_against_its_unit_only_where_both_measure_the_same(
    tmp_path,
):
    spec = _spec(tmp_path, members=["tep4"])
    spec["variants"] = {"kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}}
    spec["verification"] = {
        "kv_fp8": {
            "long": {"purpose": "p", "datasets": ["mmlu"]},
            "shots": {
                "purpose": "p",
                "datasets": ["gsm8k"],
                "evaluator_kwargs": {"num_fewshot": 20},
            },
        }
    }
    checks = _contract(_load(tmp_path, spec))
    assert checks["kv_fp8:e2e@tep4"].after == "core:e2e@tep4"
    # Another benchmark, or the same under another protocol, has its own reference.
    assert checks["kv_fp8:long@tep4"].after == "" and checks["kv_fp8:shots@tep4"].after == ""


def test_the_generated_files_are_written_once_and_committed_as_the_run(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    identity = {
        "GIT_AUTHOR_NAME": "Run",
        "GIT_AUTHOR_EMAIL": "run@example.com",
        "GIT_COMMITTER_NAME": "Run",
        "GIT_COMMITTER_EMAIL": "run@example.com",
    }
    document = {"version": 1, "model_name": "m", "model_dir": "/m", "cases": []}
    changed = e2e.write_generated(repo, FAMILY, TARGET, document)
    test_path, cases_path = e2e.generated_paths(FAMILY, TARGET)
    assert set(changed) == {
        test_path,
        cases_path,
        f"{e2e.target_dir(FAMILY, TARGET)}/__init__.py",
        f"{e2e.ACCURACY_ROOT}/{FAMILY}/__init__.py",
    }
    text = (repo / test_path).read_text()
    assert "class TestModelingV2Qwen38bSm103Unified(LlmapiAccuracyTestHarness)" in text
    assert "from ..._e2e import load_cases, run_case" in text
    sha = e2e.commit_generated(repo, changed, TARGET, identity)
    assert sha
    policy = CommitPolicy(identity=None)
    assert audit_commits(repo, "HEAD", policy) == []  # signed off as its author
    # Written again from the same spec, nothing changes and nothing is committed.
    assert e2e.write_generated(repo, FAMILY, TARGET, document) == []
    assert e2e.commit_generated(repo, [], TARGET, identity) == ""
    # A task's own commit may not change them.
    guarded = CommitPolicy(generated_paths=(test_path, cases_path))
    assert "generated by the run" in guarded.path_problem(cases_path)
    assert guarded.path_problem(f"{e2e.target_dir(FAMILY, TARGET)}/test_accuracy.py") == ""
    # Nor the extra checks that judge the run, the target's own or the shared ones.
    judged = CommitPolicy(protected_paths=e2e.extra_check_files(FAMILY, TARGET))
    for path in e2e.extra_check_files(FAMILY, TARGET):
        assert "does not change what it is judged by" in judged.path_problem(path)


def _workflow(tmp_path, data):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = _environment()
    workflow.task_data = data
    workflow._contract = contract_mod.compile_contract(data, world_size=4)
    workflow._preflight = SimpleNamespace(family=FAMILY, target_rel=TARGET)
    return workflow


def test_a_check_runs_its_case_and_a_placeholder_row_becomes_its_first_score(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    _row(Path(data["trtllm_repo_path"]), "gsm8k", 10.0)
    workflow = _workflow(tmp_path, data)
    check = _contract(data)["core:e2e@tep4"]
    seen = []

    def shell(command, env, timeout):
        seen.append(command)
        return e2e_answer(command, score=88.0)

    first = CheckRunner(workflow)._gate_call(check)(shell)
    assert first.passed and first.name == "e2e" and first.measured == 88.0, first.detail
    node = e2e.node_id(FAMILY, TARGET, "core", "e2e", "tep4")
    assert node.split("/")[-1] in seen[0] and e2e.RESULTS_ENV in seen[0]
    assert e2e.REFERENCES_ENV not in seen[0]  # the placeholder row judges the first run
    assert "gsm8k passed (88" in first.detail
    second = CheckRunner(workflow)._gate_call(check)(shell)
    assert '{"gsm8k": 88.0}' in seen[1] and "replaced the placeholder 10" in second.detail


def test_a_relative_variant_is_judged_against_its_units_score_on_the_same_target(tmp_path):
    spec = _spec(tmp_path, members=["tep4"])
    spec["variants"] = {"kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}}
    data = _load(tmp_path, spec)
    _row(Path(data["trtllm_repo_path"]), "gsm8k", 85.0)
    workflow = _workflow(tmp_path, data)
    checks = _contract(data)
    check = checks["kv_fp8:e2e@tep4"]
    assert check.after == "core:e2e@tep4"
    seen = []

    def shell(command, env, timeout):
        seen.append(command)
        return e2e_answer(command, score=90.0 if "core-e2e" in command else 89.0)

    # With no verdict of the unit's on this target, the unit's case runs here first.
    result = CheckRunner(workflow)._gate_call(check)(shell)
    assert len(seen) == 2 and "core-e2e-tep4" in seen[0] and "kv_fp8-e2e-tep4" in seen[1]
    assert '{"gsm8k": 90.0}' in seen[1]
    assert result.passed and "judged against core:e2e@tep4's score 90.00" in result.detail


def test_a_failing_dataset_fails_the_check_and_says_which(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    workflow = _workflow(tmp_path, data)
    check = _contract(data)["core:e2e@tep4"]
    result = CheckRunner(workflow)._gate_call(check)(
        lambda command, env, timeout: e2e_answer(command, score=40.0, passed=False)
    )
    assert not result.passed and "gsm8k FAILED (40" in result.detail
    assert result.floor == 81.7 and "accuracy_core" in result.floor_rule


def test_a_check_with_no_result_lines_says_what_pytest_said(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    workflow = _workflow(tmp_path, data)
    check = _contract(data)["core:e2e@tep4"]
    result = CheckRunner(workflow)._gate_call(check)(
        lambda command, env, timeout: CommandResult(returncode=4, stdout="ERROR collecting")
    )
    assert not result.passed and "the e2e test exited 4" in result.detail


# ----------------------------------------------------------------------------- pytest tests


def _report(workdir: Path, test: str, unit: str, member: str) -> Path:
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in test)
    return workdir / f"{stem}.{unit}.{member}.run.xml"


def test_a_whole_model_test_is_handed_its_members_configuration(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    test = _test(data, "core", "long_ctx")
    work = tmp_path / "gate"
    seen = []

    def runner(command, env, timeout):
        seen.append((command, timeout))
        _report(work, "long_ctx", "core", "tep4").write_text(PASS)
        return CommandResult(returncode=0)

    result = user_tests.run_user_test(
        _environment(),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tep4",
        features=[],
    )
    assert result.passed, result.detail
    command, timeout = seen[0]
    assert (
        "test_a.py::TestA::test_long[tep4]" in command
        and timeout == verification_mod.DEFAULT_TIMEOUT_S
    )
    for name in verification_mod.HANDOVER_ENV:
        assert name in command
    options = yaml.safe_load(next(work.glob("test.long_ctx.core.tep4.llm_api.yaml")).read_text())
    assert options["tensor_parallel_size"] == 4 and options["moe_expert_parallel_size"] == 4
    assert options["modeling_v2"] == "require" and "model" not in options


def test_a_module_test_is_handed_its_feature_only(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    test = _test(data, "core", "routing")
    work = tmp_path / "gate"

    def runner(command, env, timeout):
        assert "MV2FLOW_FEATURE" in command and "MV2FLOW_LLM_OPTIONS" not in command
        _report(work, "routing", "core", "tp2").write_text(PASS)
        return CommandResult(returncode=0)

    result = user_tests.run_user_test(
        _environment(2),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tp2",
    )
    assert result.passed, result.detail


@pytest.mark.parametrize(
    "xml, code, says",
    [
        (
            '<testsuite><testcase name="t"><skipped message="needs 4 GPUs"/></testcase></testsuite>',
            0,
            "skipped",
        ),
        ("<testsuite></testsuite>", 0, "selected no test"),
        (PASS, 3, "exited 3"),
        (None, 0, "no test report"),
    ],
)
def test_a_pytest_test_that_tested_nothing_fails(tmp_path, xml, code, says):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    test = _test(data, "core", "routing")
    work = tmp_path / "gate"

    def runner(command, env, timeout):
        if xml is not None:
            _report(work, "routing", "core", "tp2").write_text(xml)
        return CommandResult(returncode=code)

    result = user_tests.run_user_test(
        _environment(2),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tp2",
    )
    assert not result.passed and says in result.detail


def test_every_node_of_a_test_runs_and_each_must_pass(tmp_path):
    tests = _tests()
    tests["core"]["routing"]["pytest"] = [ROUTING, "tests/x/test_c.py"]
    data = _load(tmp_path, _spec(tmp_path, tests))
    test = _test(data, "core", "routing")
    work = tmp_path / "gate"
    ran = []

    def runner(command, env, timeout):
        part = len(ran)
        ran.append(command)
        xml = (
            PASS if part == 0 else "<testsuite><testcase name='t'><failure/></testcase></testsuite>"
        )
        _report(work, "routing", f"core.part{part}", "tp2").write_text(xml)
        return CommandResult(returncode=0)

    result = user_tests.run_user_test(
        _environment(2),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tp2",
    )
    assert len(ran) == 2 and not result.passed and "2/2 pytest jobs" in result.detail


def test_the_check_runner_runs_a_pytest_test_as_its_check(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    workflow = _workflow(tmp_path, data)
    check = _contract(data)["core:long_ctx@tep4"]
    gates = workflow.workspace / "gates" / "core"

    def shell(command, env, timeout):
        assert "test_long[tep4]" in command and "MV2FLOW_MEMBER" in command
        _report(gates, "long_ctx", "core", "tep4").write_text(PASS)
        return CommandResult(returncode=0)

    result = CheckRunner(workflow)._gate_call(check)(shell)
    assert result.passed and result.name == "long_ctx", result.detail


def test_what_a_test_reads_counts_as_a_check_input(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import gate_inputs, gate_relevant

    data = _load(tmp_path, _spec(tmp_path, _tests()))
    repo = Path(data["trtllm_repo_path"])
    _tree(repo)
    e2e.write_generated(repo, FAMILY, TARGET, {"version": 1, "cases": []})
    keep = gate_inputs(data, data["trtllm_repo_path"])
    # Under tests/, which no check reads otherwise: a change to one changes what a test measures.
    assert {"tests/x/test_a.py", "tests/x/test_b.py", e2e.RUNNER_MODULE} <= keep
    assert set(e2e.generated_paths(FAMILY, TARGET)) <= keep
    assert f"{e2e.REFERENCES_DIR}/gsm8k.yaml" in keep
    assert gate_relevant("tests/x/test_a.py", keep=keep)
    assert not gate_relevant("tests/x/test_other.py", keep=keep)


def test_the_gate_command_runs_one_check_by_hand(tmp_path, monkeypatch, capsys):
    import re

    from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli

    spec = _spec(tmp_path, _tests())
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(gate_cli, "target_identity", lambda task_data: (FAMILY, TARGET))
    seen = []

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            seen.append(command)
            if "test_e2e" in command:
                return e2e_answer(command)
            report = re.search(r"--junitxml=([^\s']+)", command)
            Path(report.group(1)).write_text(PASS, encoding="utf-8")
            return CommandResult(returncode=0)

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    common = [
        "--task",
        str(path),
        "--tree",
        str(tmp_path / "repo"),
        "--workdir",
        str(tmp_path / "w"),
    ]
    assert gate_cli.main(["long_ctx", "--member", "tep4", *common]) == 0
    assert "test_long[tep4]" in seen[-1] and "MV2FLOW_LLM_OPTIONS" in seen[-1]
    assert gate_cli.main(["routing", *common]) == 0
    assert "MV2FLOW_LLM_OPTIONS" not in seen[-1]
    # The e2e test needs the generated file in the tree measured.
    assert gate_cli.main(["e2e", *common]) == 2
    assert "the run writes the target's e2e test" in capsys.readouterr().err
    e2e.write_generated(tmp_path / "repo", FAMILY, TARGET, {"version": 1, "cases": []})
    assert gate_cli.main(["e2e", "--reference", "gsm8k=80", *common]) == 0
    assert "core-e2e-tep4" in seen[-1] and json.dumps({"gsm8k": 80.0}) in seen[-1]
    # A debugging run's cut of the check: the boot prompts alone.
    assert gate_cli.main(["e2e", "--datasets", "generate", *common]) == 0
    assert f"{e2e.DATASETS_ENV}=generate" in seen[-1]
    assert gate_cli.main(["e2e", "--datasets", "mmlu", *common]) == 2
    assert "--datasets mmlu: core's e2e runs ['generate', 'gsm8k']" in capsys.readouterr().err
    assert gate_cli.main(["nope", *common]) == 2
    assert "core has no test 'nope'" in capsys.readouterr().err


def test_the_run_metrics_name_each_check_by_its_test(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.records import metrics
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract

    status = contract.ContractStatus.load(tmp_path)
    for name, member, passed in (
        ("e2e", "tep4", True),
        ("e2e", "tep4", False),
        ("long_ctx.v2", "tp1", True),
        ("serve_one_rank", "unified", True),
    ):
        result = SimpleNamespace(
            name=name, member=member, passed=passed, detail="", duration_s=900.0
        )
        status.record("core", result)

    checks = metrics.collect(tmp_path)["checks"]
    assert sorted(checks) == ["e2e", "long_ctx.v2", "serve_one_rank"]
    assert checks["e2e"]["runs"] == 2 and checks["e2e"]["failed"] == 1
    # A member's width is its devices; one where the member does not say.
    assert checks["e2e"]["gpu_hours"] == 2.0
    assert checks["long_ctx.v2"]["gpu_hours"] == 0.25
    assert checks["serve_one_rank"]["gpu_hours"] == 0.25


def test_the_run_metrics_say_what_the_gpus_ran_besides_the_checks(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.records import metrics

    def job(owner, duration, *, devices=("0",), rc=0, submitted="2026-11-22T15:00:00Z"):
        return {
            "owner": owner,
            "duration_s": duration,
            "devices": list(devices),
            "rc": rc,
            "submitted": submitted,
            "started": "2026-11-22T15:02:00Z",
        }

    records = [
        job("gate", 3600),
        job("gate-e2e", 1800, devices=("0", "1")),
        job("", 360, rc=1),
        job("", 360),
        job("preflight-cpu", 10),
        job("", 3600, submitted="2026-11-22T13:00:00Z"),  # another run's, before this one opened
    ]
    since = metrics._epoch("2026-11-22T14:00:00Z")
    assert metrics.gpu_jobs(records, since) == {
        "checks": {"jobs": 1, "failed": 0, "gpu_hours": 1.0},
        "flow_other": {"jobs": 1, "failed": 0, "gpu_hours": 0.003},
        "worker": {"jobs": 2, "failed": 1, "gpu_hours": 0.2},
        "worker_gates": {"jobs": 1, "failed": 0, "gpu_hours": 1.0},
        "longest_wait_min": 2.0,
    }
    # A local run has no spool: nothing is said.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "task.yaml").write_text(yaml.safe_dump({"environment": {"kind": "local"}}))
    assert metrics._gpu_jobs(workspace, None) == {}
    spool = tmp_path / "spool"
    (spool / "done").mkdir(parents=True)
    (spool / "done" / "a.json").write_text(json.dumps(job("", 360)))
    (workspace / "task.yaml").write_text(
        yaml.safe_dump({"environment": {"kind": "session_spool", "spool": str(spool)}})
    )
    assert metrics._gpu_jobs(workspace, None)["worker"]["jobs"] == 1


def test_a_failed_pytest_check_quotes_pytests_own_reason(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks import pytest_node

    data = _load(tmp_path, _spec(tmp_path))
    workflow = _workflow(tmp_path, data)
    check = _contract(data)["core:e2e@tep4"]
    output = (
        "E       NoTarget: qwen3 is routed but its forward is not assembled yet\n"
        "=========================== short test summary info ============================\n"
        "FAILED accuracy/modeling_v2/q/t/test_e2e.py::TestX::test_e2e[core-e2e-tep4] - "
        "NoTarget: qwen3 is routed but its forward is not assembled yet\n"
    )
    seen = []

    def shell(command, env, timeout):
        seen.append(command)
        return CommandResult(returncode=1, stdout=output, stderr="Traceback ...\n" * 50)

    result = CheckRunner(workflow)._gate_call(check)(shell)
    assert "-rsfE" in seen[0]
    assert not result.passed
    assert result.detail.startswith("the e2e test exited 1: FAILED ")
    assert "forward is not assembled yet" in result.detail
    assert pytest_node.failure_summary("1 passed in 2s\n") == ""


def test_a_check_reads_its_own_results_not_a_workers_run_of_the_same_test(tmp_path, monkeypatch):
    from itertools import count

    from agent_flow.workflows.trtllm_modeling_v2.checks import pytest_node

    tokens = count()
    monkeypatch.setattr(pytest_node, "run_token", lambda: f"t{next(tokens)}")
    data = _load(tmp_path, _spec(tmp_path))
    test = _test(data, "core", "e2e")
    work = tmp_path / "gates" / "core"
    node = e2e.node_id(FAMILY, TARGET, "core", "e2e", "tep4")

    def run(runner):
        return e2e.run_e2e(
            _environment(), task_data=data, test=test, member="tep4", node=node,
            workdir=work, runner=runner,
        )  # fmt: skip

    # A worker's run of the benchmark alone, in the same directory, leaves its lines behind.
    worker, _ = run(lambda command, env, timeout: e2e_answer(command, score=89.4))
    assert worker.measured == 89.4
    # The run's own check fails before any dataset ran: it reports no score of anyone's.
    check, lines = run(
        lambda command, env, timeout: CommandResult(returncode=1, stdout="FAILED x - NoTarget\n")
    )
    assert not check.passed and lines == [] and check.measured is None
    assert "NoTarget" in check.detail and "89.4" not in check.detail


# ----------------------------------------------------------------------------- landing budget


def test_every_catalog_dataset_has_a_positive_reference_timing():
    assert set(verification_mod.DATASET_REF_MINUTES) == set(verification_mod.DATASETS)
    assert all(m > 0 for m in verification_mod.DATASET_REF_MINUTES.values())
    assert verification_mod.HEAVY_DATASETS <= set(verification_mod.BENCHMARKS)
    assert verification_mod.LONG_OUTPUT_BENCHMARKS <= verification_mod.HEAVY_DATASETS


def test_estimate_run_minutes_scales_thinking_without_double_counting_long_cot():
    gsm = verification_mod.Test(
        name="e2e",
        feature="core",
        purpose="p",
        runner=verification_mod.RUN_DATASETS,
        datasets=("generate", "gsm8k"),
    )
    gpqa = verification_mod.Test(
        name="gpqa",
        feature="core",
        purpose="p",
        runner=verification_mod.RUN_DATASETS,
        datasets=("gpqa_diamond",),
    )
    serve_test = verification_mod.Test(
        name="serve",
        feature="core",
        purpose="p",
        runner=verification_mod.RUN_SERVE,
        datasets=("generate",),
    )
    pytest_test = verification_mod.Test(
        name="routing",
        feature="core",
        purpose="p",
        runner=verification_mod.RUN_PYTEST,
        nodes=(ROUTING,),
        devices=1,
    )
    thinking = {"chat_template_kwargs": {"enable_thinking": True}}
    no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}

    assert verification_mod.estimate_run_minutes(gsm) == 6.5
    assert verification_mod.estimate_run_minutes(gsm, evaluator_kwargs=no_thinking) == 6.5
    assert verification_mod.estimate_run_minutes(gsm, evaluator_kwargs=thinking) == 30.5
    # gpqa_diamond already budgets 32k output tokens; thinking does not multiply it again.
    assert verification_mod.estimate_run_minutes(gpqa, evaluator_kwargs=thinking) == 32.0
    assert verification_mod.estimate_run_minutes(serve_test) == 0.5
    assert verification_mod.estimate_run_minutes(pytest_test) is None


def test_landing_budget_summarizes_stages_and_warns_on_heavy_or_over_budget(tmp_path):
    # Lean landing: 2 e2e (generate+gsm8k = 6.5m each) + 1 pytest (unestimated); 2 delivery checks.
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    contract = contract_mod.compile_contract(data, world_size=4)
    budget = contract_mod.landing_budget(data, contract, world_size=4)
    assert budget.landing_checks == 3
    assert budget.delivery_checks == 2
    assert budget.estimated_minutes == 13.0
    assert budget.unestimated_checks == 1
    assert budget.warnings == ()
    assert "3 landing (~13 min est. run time" in budget.summary_line()
    assert "+ 1 unestimated" in budget.summary_line()
    assert "2 delivery" in budget.summary_line()

    # Adding gpqa_diamond at landing on both members triggers heavy + over-budget warnings.
    heavy_tests = {
        "core": {
            "reasoning": {
                "purpose": "hard science reasoning holds",
                "datasets": ["gpqa_diamond"],
                "members": "all",
            }
        }
    }
    heavy_data = _load(tmp_path, _spec(tmp_path, heavy_tests))
    heavy_contract = contract_mod.compile_contract(heavy_data, world_size=4)
    heavy_budget = contract_mod.landing_budget(heavy_data, heavy_contract, world_size=4)
    assert heavy_budget.estimated_minutes == 77.0
    assert any(
        "heavy benchmark(s) at landing" in w and "gpqa_diamond" in w for w in heavy_budget.warnings
    )
    assert any("past the 60 min landing budget" in w for w in heavy_budget.warnings)

    # Moving it to delivery clears both warnings and drops it from the landing estimate.
    heavy_tests["core"]["reasoning"]["stage"] = "delivery"
    moved_data = _load(tmp_path, _spec(tmp_path, heavy_tests))
    moved_contract = contract_mod.compile_contract(moved_data, world_size=4)
    moved_budget = contract_mod.landing_budget(moved_data, moved_contract, world_size=4)
    assert moved_budget.estimated_minutes == 13.0
    assert moved_budget.delivery_checks == 2
    assert moved_budget.warnings == ()


def test_landing_budget_warns_on_thinking_mode_and_skips_answered_checks(tmp_path):
    # A subset check answered by core:e2e is not double-counted; enable_thinking warns & scales.
    spec = _spec(
        tmp_path,
        {
            "core": {
                "dup_gsm8k": {
                    "purpose": "same gsm8k measurement as e2e",
                    "datasets": ["gsm8k"],
                    "members": "tep4",
                }
            }
        },
    )
    spec["accuracy"]["evaluator_kwargs"] = {"chat_template_kwargs": {"enable_thinking": True}}
    data = _load(tmp_path, spec)
    contract = contract_mod.compile_contract(data, world_size=4)
    checks = {c.key: c for c in contract.checks}
    assert checks["core:dup_gsm8k@tep4"].answered_by == "core:e2e@tep4"

    budget = contract_mod.landing_budget(data, contract, world_size=4)
    # Two e2e checks × (0.5 + 6.0 * 5) = 61.0 min; dup_gsm8k is answered_by and not double-counted.
    assert budget.estimated_minutes == 61.0
    assert any("enable_thinking: true" in w for w in budget.warnings)
    assert any("past the 60 min landing budget" in w for w in budget.warnings)


def test_bootstrap_prints_and_journals_landing_budget(tmp_path):
    from rich.console import Console

    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    tests = {
        "core": {
            "reasoning": {
                "purpose": "hard science reasoning holds",
                "datasets": ["gpqa_diamond"],
                "members": "tep4",
            }
        }
    }
    data = _load(tmp_path, _spec(tmp_path, tests))
    workflow = _workflow(tmp_path, data)
    workflow._gate_env = lambda: _environment(world_size=4)
    sections: list[tuple[str, str]] = []
    workflow.journal = SimpleNamespace(section=lambda title, body: sections.append((title, body)))
    console = Console(record=True, width=200)

    bootstrap._note_coverage(workflow, console)
    bootstrap._write_contract(workflow)

    printed = console.export_text()
    assert "3 landing (~45 min est. run time" in printed
    assert "heavy benchmark(s) at landing" in printed
    contract_section = next(body for title, body in sections if title.startswith("contract:"))
    assert "stages: 3 landing" in contract_section
    assert "- warning: heavy benchmark(s) at landing" in contract_section
