# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 task schema behavior and regression checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.checks.boot import BootSpec
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments, derive, task_schema
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
        "trtllm_repo_path": str(repo),
        "deployments": [{"name": "tp1", "options": {"max_batch_size": 8}}],
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
    spec["accuracy"]["envelope"] = {"max_tokens": 16}
    spec["accuracy"]["score_pattern"] = "accuracy: [0-9.]+"
    spec["deployments"].append({"name": "tp1", "options": "absent.yaml"})
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    for expected in (
        "arch",
        "environment",
        "non-existent path",
        "'accuracy.envelope' is no longer read: nothing: the boundary phases left the flow",
        "'accuracy.score_pattern' needs a group around the score",
        "'deployments.tp1' is named twice",
        "'deployments.tp1.options' does not exist",
    ):
        assert expected in message


def test_a_misspelled_key_is_refused_with_its_spelling_at_any_depth(tmp_path):
    # A misspelled `deployments` must not silently bring up nothing.
    spec = _valid_spec(tmp_path) | {"deployment": []}
    spec["accuracy"]["toleranace"] = "x"
    # A key of a check that no longer runs is no key: nothing reads a stock checkpoint.
    spec["environment"]["substrate_checkpoint"] = str(tmp_path)
    spec["deployments"][0]["benchmakrs"] = []
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'deployment'" in str(excinfo.value)
    assert "did you mean 'deployments'" in str(excinfo.value)
    assert "did you mean 'accuracy.tolerance'" in str(excinfo.value)
    assert "did you mean 'deployments[0].benchmarks'" in str(excinfo.value)
    assert "unknown key 'environment.substrate_checkpoint'" in str(excinfo.value)


def test_a_value_of_the_wrong_type_is_a_schema_error_and_never_a_crash(tmp_path):
    spec = _valid_spec(tmp_path)
    for key, value in (
        ("inputs", ["text", {"image": "file"}]),
        ("probes", "reuse"),
        ("tests", {"a": 1}),
        ("benchmarks", [{"name": ["gsm8k"]}]),
        ("env", ["A=1"]),
        ("timeout", "soon"),
    ):
        stated = dict(spec["deployments"][0]) | {key: value}
        with pytest.raises(task_schema.TaskSchemaError, match=f"deployments.*{key}"):
            task_schema.load_and_validate_task_yaml(
                _write(tmp_path, spec | {"deployments": [stated]})
            )


def test_a_perf_block_is_refused_with_why_the_flow_has_no_tuning_stage(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["perf"] = {"enabled": True}
    with pytest.raises(task_schema.TaskSchemaError, match="'perf' is not read: the flow has no"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def test_a_spec_states_deployments_and_the_older_forms_say_what_replaced_them(tmp_path):
    spec = _valid_spec(tmp_path)
    stated = spec.pop("deployments")
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "missing required field 'deployments'" in str(excinfo.value)
    assert "intake derive --checkpoint" in str(excinfo.value)
    spec["deployments"] = stated
    older = {
        "parallel": "unified",
        "members": ["tp1", "tep4"],
        "features": ["mtp"],
        "runtime": {"cuda_graph": True},
        "variants": {"kv_fp8": {"unit": "core"}},
        "profile": "bringup",
        "coverage": {"accept": ["mtp@tp1"]},
        "scope": "text_only",
    }
    spec.update(older)
    spec["accuracy"].update(
        llm_api_options="x.yaml", disagg={"test": "t.py::T::t"}, serve={"enabled": True}
    )
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    lines = str(excinfo.value).splitlines()
    for key in older:
        said = [line for line in lines if line.startswith(f"  - '{key}' is no longer read: ")]
        assert len(said) == 1, key
    assert any("'members' is no longer read: one deployment per configuration" in x for x in lines)
    assert any("'accuracy.llm_api_options' is no longer read: each deployment" in x for x in lines)
    assert any("'accuracy.disagg' is no longer read: a deployment with `ctx`" in x for x in lines)
    assert any("'accuracy.serve' is no longer read" in x for x in lines)
    # One line each: none is also reported as an unknown key.
    assert not any("unknown key" in line for line in lines)


def test_what_a_spec_deploys_is_read_from_its_deployments(tmp_path):
    (tmp_path / "engines").mkdir()
    (tmp_path / "engines" / "tp2.yaml").write_text(yaml.safe_dump({"tensor_parallel_size": 2}))
    moves = {"cache_transceiver_config": {"backend": "DEFAULT"}}
    spec = _valid_spec(tmp_path)
    spec["target"] = {"parallel": "tp2"}
    spec["deployments"] = [
        {"name": "tp2", "options": "engines/tp2.yaml"},
        {
            "name": "pd",
            "ctx": {"options": moves},
            "gen": {"options": {"tensor_parallel_size": 2, **moves}, "instances": 2},
            "relative_to": "tp2",
        },
    ]
    data = load_and_validate_task_yaml(_write(tmp_path, spec))
    # An options file beside the spec is named by its absolute path from then on.
    assert data["deployments"][0]["options"] == str(tmp_path / "engines" / "tp2.yaml")
    first, pd = deployments.resolve(data)
    assert first.server.options == {"tensor_parallel_size": 2} and pd.ranks == 5
    # The words the run's records use: a unit is a deployment, its member the shape.
    assert task_schema.assembly_units(data) == ["tp2", "pd"]
    assert task_schema.widest_world_size(data) == 5
    assert task_schema.derive_slug(data) == "ckpt-sm_103-tp2"
    # A spec with no `accuracy` block loads: each deployment is judged by its session.
    del spec["accuracy"]
    bare = load_and_validate_task_yaml(_write(tmp_path, spec))
    assert all(d.benchmarks == () for d in deployments.resolve(bare))
    # A resolved spec, written back, loads to itself.
    assert load_and_validate_task_yaml(_write(tmp_path, bare)) == bare


def test_opinions_about_a_spec_are_advice_and_do_not_stop_its_load(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.spec import advice

    spec = _valid_spec(tmp_path)
    # On an anchor that reads as a placeholder, with no notes.
    spec["deployments"] = [
        {
            "name": "tp1",
            "options": {},
            "benchmarks": [{"name": "gsm8k", "anchor": 10, "source": "a placeholder"}],
        },
        {"name": "other", "options": {}, "benchmarks": [{"name": "mmlu"}]},
    ]
    data = load_and_validate_task_yaml(_write(tmp_path, spec))
    said = "\n".join(advice.spec_warnings(data))
    assert "anchor reads as a placeholder: `tp1` on `gsm8k` is held to 10" in said
    assert "no accuracy anchor: `other` on `mmlu`" in said and "blocked before its load" in said
    report = advice.advise(data)
    assert report.summary().endswith("none stops the run")
    assert "- `tp1` (first): aggregated, 1 rank(s); one session" in report.render()


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
    (root / "spool").mkdir(exist_ok=True)
    site = {
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "spool": str(root / "spool"),
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "device": {"arch": "sm_100"},
        "defaults": {"members": ["tp1", "tep4", "dep4"]},
        **changes,
    }
    path = root / "site.yaml"
    path.write_text(yaml.safe_dump(site))
    return path


def _derive(root: Path, goal: str | None = None, **site) -> Path:
    return cli._derive_spec(
        _checkpoint(root), _site(root, **site), root / "ws", goal=goal, clean=False
    )


def _options(spec: dict, name: str, role: str | None = None) -> dict:
    entry = next(d for d in spec["deployments"] if d["name"] == name)
    return yaml.safe_load(Path((entry[role] if role else entry)["options"]).read_text())


def test_the_goal_overrides_what_the_checkpoint_and_the_site_say(tmp_path):
    goal = (
        "-mtp multimodal:smoke tp1 dep4 isl=1024 osl=256 arch=sm_103 disagg@tp1/dep4 kv=bf16/fp8 "
        "concurrency=8/256"
    )
    site = dict(
        device={"arch": "sm_100", "gpus_per_node": 4},
        defaults={"disagg": {"env": {"UCX_TLS": "^ib"}, "timeout": 5400}},
    )
    path = _derive(tmp_path, goal, **site)
    spec = load_and_validate_task_yaml(path)
    assert spec["arch"] == "sm_103"
    # One deployment per layout, then each feature on its own, then each extra KV dtype.
    found = deployments.resolve(spec)
    assert [d.name for d in found] == ["tp1", "dep4", "multimodal", "pd", "kv_fp8"]
    assert [d.shape for d in found] == ["tp1", "dep4", "tp1", "pd5", "tp1"]
    by_name = {d.name: d for d in found}
    # The drafter the checkpoint carries is left out: the goal says so.
    assert not any(d.drafts for d in found)
    # A smoke feature is judged by its session alone; the others by the benchmark.
    assert by_name["multimodal"].inputs == ("text", "image")
    assert by_name["multimodal"].benchmarks == ()
    assert [b.name for b in by_name["tp1"].benchmarks] == ["gsm8k"]
    # The CI files are owed while one deployment lists a benchmark.
    from agent_flow.workflows.trtllm_modeling_v2.run.validation.completion import ci_waiver

    assert ci_waiver(spec) == ""
    boot_only = {**spec, "deployments": [{"name": "tp1", "options": {}, "benchmarks": []}]}
    assert ci_waiver(boot_only) == "not owed: no deployment of the spec lists a benchmark"
    # disagg where the goal puts it: context at tp1, generation at dep4, each stating what
    # moves the cache, judged against the first deployment's score, with the site's environment.
    pd = by_name["pd"]
    assert (pd.ctx.ranks, pd.gen.ranks, pd.relative_to) == (1, 4, "tp1")
    assert pd.env == {"UCX_TLS": "^ib"} and pd.timeout == 5400
    assert _options(spec, "pd", "gen")["cache_transceiver_config"] == {"backend": "DEFAULT"}
    assert _options(spec, "pd", "gen")["enable_attention_dp"] is True
    # The first KV dtype is every deployment's; each other one a deployment of its own.
    assert "dtype" not in _options(spec, "tp1")["kv_cache_config"]
    assert _options(spec, "kv_fp8")["kv_cache_config"]["dtype"] == "fp8"
    assert by_name["kv_fp8"].relative_to == "tp1"
    assert _options(spec, "tp1")["max_seq_len"] == 1280
    # The batch holds the goal's concurrency, whatever the default says; under attention
    # data parallel it is each rank's share.
    assert _options(spec, "tp1")["max_batch_size"] == 256
    assert _options(spec, "dep4")["max_batch_size"] == 64
    assert _options(spec, "dep4")["cuda_graph_config"]["max_batch_size"] == 64


def test_a_goal_that_names_its_features_gets_those_and_what_derive_cannot_know_is_asked(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    config = json.loads((checkpoint / "config.json").read_text())
    del config["_name_or_path"]
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(derive.DeriveError) as excinfo:
        cli._derive_spec(
            checkpoint, _site(tmp_path), tmp_path / "ws", goal="multimodal", clean=False
        )
    assert "accuracy.model_key" in str(excinfo.value)
    spec = yaml.safe_load((tmp_path / "ws" / derive.SPEC_FILENAME).read_text())
    # The checkpoint's drafter is left out: the goal names its features and not it.
    assert [d["name"] for d in spec["deployments"]] == ["tp1", "tep4", "dep4", "multimodal"]
    assert "features (left out)" in (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    with pytest.raises(derive.DeriveError, match="kv= is named once"):
        derive.parse_goal("kv=fp8 kv=nvfp4")


def test_core_keeps_reuse_and_chunking_off_and_each_named_one_is_its_own_deployment(tmp_path):
    spec = load_and_validate_task_yaml(
        _derive(
            tmp_path, "tp1 chunked_prefill prefix_reuse kv=bf16/fp8 chunk=256 concurrency=1/128"
        )
    )
    core = _options(spec, "tp1")
    assert core["cuda_graph_config"] == {"enable_padding": True, "max_batch_size": 128}
    assert "disable_overlap_scheduler" not in core
    assert core["kv_cache_config"]["enable_block_reuse"] is False
    assert core["enable_chunked_prefill"] is False and "max_num_tokens" not in core
    found = {d.name: d for d in deployments.resolve(spec)}
    # The checkpoint's drafter and tower are not brought up: the goal names neither. A
    # first run is one deployment, and the report says the word that adds each.
    assert list(found) == ["tp1", "kv_fp8", "chunked", "reuse"]
    chunked = _options(spec, "chunked")
    assert (chunked["enable_chunked_prefill"], chunked["max_num_tokens"]) == (True, 256)
    assert _options(spec, "reuse")["kv_cache_config"]["enable_block_reuse"] is True
    # Each changes one non-core feature, is judged against the first deployment's own
    # score, and its session proves the feature it turns on.
    for name in ("kv_fp8", "chunked", "reuse"):
        assert found[name].relative_to == "tp1"
    assert found["chunked"].asked() == ("chunk",) and found["reuse"].asked() == ("reuse",)
    assert found["tp1"].asked() == ()
    report = (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    assert (
        "the checkpoint carries MTP draft layers: `mtp` in --goal adds their deployment" in report
    )
    assert "the checkpoint carries a tower: `multimodal` in --goal adds its deployment" in report
    named = load_and_validate_task_yaml(_derive(tmp_path / "named", "tp1 mtp multimodal"))
    assert [d["name"] for d in named["deployments"]] == ["tp1", "mtp", "multimodal"]
    assert _options(named, "mtp")["speculative_config"] == {
        "decoding_type": "MTP",
        "max_draft_len": 1,
    }
    # The batch the spec carries, raised to the concurrency, not the default it started from.
    assert "| `workload.max_batch_size` | concurrency | `128` |" in report
    with pytest.raises(derive.DeriveError):
        derive.parse_goal("chunked_prefill:smoke")


def test_a_goal_that_says_mtp_gets_the_checkpoints_own_drafter_and_the_report_says_so(tmp_path):
    """A checkpoint with DSpark draft layers was given ordinary MTP at depth 1, in silence."""
    from agent_flow.workflows.trtllm_modeling_v2.spec import advice

    def checkpoint(root):
        path = _checkpoint(root)
        config = json.loads((path / "config.json").read_text())
        config.pop("num_nextn_predict_layers")
        # As a multimodal checkpoint states them: under the language model's own block.
        config["text_config"] = {"num_nextn_predict_layers": 3, "dspark_block_size": 5,
                                 "dspark_num_stages": 2}  # fmt: skip
        (path / "config.json").write_text(json.dumps(config))
        return path

    def derived(root, goal):
        path = cli._derive_spec(checkpoint(root), _site(root), root / "ws", goal=goal, clean=False)
        return load_and_validate_task_yaml(path), path

    spec, path = derived(tmp_path / "said", "tp1 mtp text_only model_key=org/newmodel")
    names = [d["name"] for d in spec["deployments"]]
    assert "dspark" in names and "mtp" not in names
    drafter = _options(spec, "dspark")["speculative_config"]
    assert drafter == {"decoding_type": "DSpark", "max_draft_len": 5}
    report = " ".join((path.parent / derive.REPORT_FILENAME).read_text().split())
    assert "the goal says `mtp`, and the checkpoint's own draft layers are DSpark's" in report
    # A goal that names its own drafter keeps its word.
    spec, _ = derived(tmp_path / "own", "tp1 mtp drafter=/models/d text_only model_key=org/m")
    assert "mtp" in [d["name"] for d in spec["deployments"]]
    # The advice reads the draft layers where the checkpoint states them.
    data, _ = derived(tmp_path / "advice", "tp1 dspark text_only model_key=org/newmodel")
    config = json.loads((Path(data["checkpoint_path"]) / "config.json").read_text())
    assert not [n for n in advice.checkpoint_notes(data, config) if "no draft layers" in n]


def test_a_disaggregated_pair_is_as_the_goal_says_and_one_wider_than_a_node_is_said(tmp_path):
    """`disagg` with no layout puts the first layout on both sides: eight devices on a node of four."""

    def pair(root, goal, **site):
        path = cli._derive_spec(
            _checkpoint(root), _site(root, **site), root / "ws", goal=goal, clean=False
        )
        spec = load_and_validate_task_yaml(path)
        report = " ".join((path.parent / derive.REPORT_FILENAME).read_text().split())
        ranks = [_options(spec, "pd", role)["tensor_parallel_size"] for role in ("ctx", "gen")]
        return ranks, report

    goal = "tp4 disagg text_only -spec model_key=org/newmodel"
    four = {"device": {"arch": "sm_100", "gpus_per_node": 4}}
    # The site states a node of four: the pair is written as the goal's words say, and the
    # report says it is wider than a node and which word states the two layouts.
    for root, said in (("unsaid", goal), ("wide", goal.replace("disagg", "disagg@tp4/tp4"))):
        ranks, report = pair(tmp_path / root, said, **four)
        assert ranks == [4, 4] and "pd (wider than a node)" in report and "8 devices" in report
        assert "its session runs as one job of whole nodes" in report
        assert "`disagg@<ctx>/<gen>` in --goal names the two layouts" in report
    # A pair that fits, and a site that states no node, change nothing and say nothing.
    ranks, report = pair(tmp_path / "fits", goal.replace("disagg", "disagg@tp1/tp3"), **four)
    assert ranks == [1, 3] and "pd (" not in report
    ranks, report = pair(tmp_path / "unstated", goal)
    assert ranks == [4, 4] and "pd (" not in report


def test_a_goal_word_derive_does_not_know_is_refused_with_the_words_it_knows():
    with pytest.raises(derive.DeriveError) as stopped:
        derive.parse_goal("tp4 adp4 multimodality")
    first, second, known = str(stopped.value).splitlines()
    assert first == "goal: 'adp4' is not a feature, a member or key=value"
    assert second == "goal: 'multimodality' is not a feature, a member or key=value"
    # Said once, whatever the number of unknown words: an agent picks the word it meant.
    assert known.startswith("goal: the words a goal knows are ")
    assert all(word in known for word in ("multimodal", "disagg", "prefix_reuse", "dep<N>"))


def test_derive_gives_a_session_a_limit_to_start_with_where_one_holder_admits_no_more(tmp_path):
    """The spec as written starts: half of what a holder admits, said to be a starting value."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.checks import session
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pool as pool_mod
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import scheduler
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight

    # Holders with no stated end: nothing is set, as before.
    spec = yaml.safe_load(_derive(tmp_path / "open", "tp1").read_text())
    assert all("timeout" not in d for d in spec["deployments"])
    # Holders of four hours: a session of the default checks (boot and one benchmark of
    # four hours) cannot finish in one.
    slurm = {
        "account": "<account>",
        "partition": "<partition>",
        "time": "04:00:00",
        "gpus_per_node": 4,
    }
    environment = {
        "python": "/opt/venv/bin/python",
        "spool": str(tmp_path / "spool"),
        "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        "slurm": slurm,
    }
    (tmp_path / "spool").mkdir()
    # No question: the first thing a user of a new model does is derive and start.
    path = _derive(tmp_path / "short", "tp1", environment=environment)
    spec = yaml.safe_load(path.read_text())
    (first,) = [d for d in spec["deployments"] if d["name"] == "tp1"]
    # The longest one holder admits is 13,380 s (four hours less a holder's start, the
    # margin, the teardown). Derive wrote that once: such a session started only in a
    # holder's first five minutes and, hung, cost the holder. It writes half.
    assert first["timeout"] == 6660
    wall = first["timeout"] + session.TEARDOWN_S
    # (Preflight reads the options a deployment names: here they are given in place.)
    asked = {**spec, "deployments": [{**first, "options": {}}]}
    assert "tp1 7260s" in preflight.check_gate_budgets(asked)[-1]
    # The scheduler admits it well into a holder, where the longest waits for the next one.
    watcher = SimpleNamespace(config=SimpleNamespace(ends_at=1000.0 + 4 * 3600))
    job = SimpleNamespace(holds_gpus=True, wall_s=wall)
    assert scheduler.Scheduler._fit_hold(watcher, job, 1000.0 + 7000) is None
    # One rule for "fits a holder", and every reader of it agrees: the longest is admitted
    # once the holder's start is over, and nothing longer.
    longest = 13380 + session.TEARDOWN_S
    assert longest <= pool_mod.fit_wall_s(4 * 3600) < longest + 60
    started = 1000.0 + pool_mod.START_SLACK_S
    job.wall_s = longest
    assert scheduler.Scheduler._fit_hold(watcher, job, started) is None
    assert "waits for a holder" in scheduler.Scheduler._fit_hold(watcher, job, started + 60).reason
    job.wall_s = longest + 60
    assert (
        "waits for a holder that ends later"
        in scheduler.Scheduler._fit_hold(watcher, job, started).reason
    )
    # And preflight refuses what an earlier flow derived, with the limit that does fit.
    asked["deployments"][0]["timeout"] = 13680
    with pytest.raises(preflight.PreflightError, match="at most 13380s"):
        preflight.check_gate_budgets(asked)
    # Not a question (the spec as written starts): listed among what the spec left
    # unstated, as a starting value, with where the real one is read.
    report = " ".join((path.parent / derive.REPORT_FILENAME).read_text().split())
    assert "## Open questions (0)" in report and "deployments.tp1.timeout" in report
    assert (
        "a starting value: half of what one holder of the site admits (13380s of its 14400s"
        in report
    )
    assert "The line the first session prints ends with its seconds: put that, with room" in report


def test_a_model_the_engine_registers_a_tokenizer_for_is_served_with_it(tmp_path):
    """Found in the engine's checkout by search as derive runs: the flow keeps no table of models."""
    repo = tmp_path / "repo"
    (repo / "engine" / "text").mkdir(parents=True)
    (repo / "engine" / "text" / "loading.py").write_text(
        'TOKENIZER_ALIASES = {\n    "newfam": "engine.text.newfam.NewfamTokenizer",\n}\n'
    )
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "x"]):  # fmt: skip
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    assert derive.registered_tokenizer(repo, "newfam") == "newfam"
    # Another model's alias is not this one's, and a checkout with no registry names none.
    assert derive.registered_tokenizer(repo, "newfam_v2") is None
    assert derive.registered_tokenizer(tmp_path, "newfam") is None
    checkpoint = _checkpoint(tmp_path)
    config = json.loads((checkpoint / "config.json").read_text())
    (checkpoint / "config.json").write_text(json.dumps({**config, "model_type": "newfam"}))
    derived = cli._derive_spec(
        checkpoint, _site(tmp_path), tmp_path / "ws", goal="tp1", clean=False
    )
    spec = load_and_validate_task_yaml(derived)
    assert _options(spec, "tp1")["custom_tokenizer"] == "newfam"
    report = (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    assert "custom_tokenizer" in report and "the engine registers one" in report


def test_derive_writes_the_boot_rule_and_names_the_engines_recipes_for_a_person(tmp_path):
    """What is certain is written; an architecture's own options are open, with where they are."""
    # The spec guide's own rule: one fact and one arithmetic problem, each checked.
    boot = yaml.safe_load(_derive(tmp_path / "plain", "tp1").read_text())["accuracy"]["boot"]
    assert [p["keywords_any"] for p in boot["prompts"]] == [["Paris"], ["72"]]
    assert boot["max_tokens"] == 64
    # A boot block written by hand that states no budget has the same room.
    assert BootSpec.from_mapping({"prompts": ["2 + 2 ="]}, "by hand").max_tokens == 64
    # The engine's files named for the checkpoint's model type, found by search as
    # derive runs: named in an open question and read by nobody but the person.
    repo = tmp_path / "repo"
    for name in (
        "tests/integration/defs/newfam_serving.py",
        "examples/models/core/newfam/README.md",
        "examples/models/core/other/README.md",
        "engine/models/newfam.py",
    ):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "x"]):  # fmt: skip
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    recipes = ["examples/models/core/newfam/README.md", "tests/integration/defs/newfam_serving.py"]
    assert derive.engine_recipes(repo, "newfam") == recipes
    assert derive.engine_recipes(repo, None) == [] == derive.engine_recipes(tmp_path, "newfam")
    checkpoint = _checkpoint(tmp_path)
    config = json.loads((checkpoint / "config.json").read_text())
    (checkpoint / "config.json").write_text(json.dumps({**config, "model_type": "newfam"}))
    with pytest.raises(derive.DeriveError) as open_:
        cli._derive_spec(checkpoint, _site(tmp_path), tmp_path / "ws", goal="tp1", clean=False)
    said = " ".join(str(open_.value).split())
    assert "leaves 1 question(s) open" in said and "- engine options: " in said
    assert f"files named for `newfam`: {', '.join(recipes)}. Add to each options file" in said


def test_a_site_file_that_is_not_there_is_one_line_that_names_the_option(tmp_path, capsys):
    """A stranger's first typo ended in a traceback; a missing checkpoint had one useful line."""
    from agent_flow.workflows.trtllm_modeling_v2.spec import intake

    argv = ["derive", "--checkpoint", str(_checkpoint(tmp_path)), "--goal", "tp1",
            "--out", str(tmp_path / "out")]  # fmt: skip
    assert intake.main([*argv, "--site", str(tmp_path / "not-yet.yaml")]) == 2
    (said,) = capsys.readouterr().err.splitlines()
    assert said.startswith(f"error: --site {tmp_path / 'not-yet.yaml'}: no site file can be read")
    assert "No such file" in said and not (tmp_path / "out").exists()
    # One that is no YAML says so the same way; the run's own route reads it through the same door.
    (tmp_path / "broken.yaml").write_text("environment: [unclosed\n")
    assert intake.main([*argv, "--site", str(tmp_path / "broken.yaml")]) == 2
    assert capsys.readouterr().err.startswith(f"error: --site {tmp_path / 'broken.yaml'}: no site")
    with pytest.raises(derive.DeriveError, match="--site .*not-yet.yaml: no site file"):
        cli._derive_spec(_checkpoint(tmp_path), tmp_path / "not-yet.yaml", tmp_path / "ws",
                         goal="tp1", clean=False)  # fmt: skip


def test_the_report_says_what_running_the_command_again_does_to_an_edit(tmp_path, capsys):
    """A stranger followed "run the same command" and lost the options they had edited."""
    from agent_flow.workflows.trtllm_modeling_v2.spec import intake

    out, site = tmp_path / "out", _site(tmp_path)
    argv = ["derive", "--checkpoint", str(_checkpoint(tmp_path)), "--site", str(site),
            "--goal", "tp1", "--out", str(out)]  # fmt: skip
    assert intake.main(argv) == 0 and "written anew" not in capsys.readouterr().out
    report = (out / derive.REPORT_FILENAME).read_text()
    # `intake derive` has no --clean and keeps nothing: its report says so.
    assert derive.REWRITTEN in report and "--clean" not in report
    options = out / derive.ENGINES_DIRNAME / "tp1.yaml"
    options.write_text(options.read_text() + "max_batch_size: 7\n")
    assert intake.main(argv) == 0
    assert f"{out} holds a spec derived before" in capsys.readouterr().out
    assert yaml.safe_load(options.read_text())["max_batch_size"] != 7
    # The command that derives and runs keeps the spec it finds, and its report says that.
    kept = cli._derive_spec(_checkpoint(tmp_path), site, tmp_path / "ws", goal="tp1", clean=False)
    assert derive.KEPT in (kept.parent / derive.REPORT_FILENAME).read_text()
    kept.write_text(kept.read_text() + "notes: mine\n")
    again = cli._derive_spec(_checkpoint(tmp_path), site, tmp_path / "ws", goal="tp1", clean=False)
    assert "notes: mine" in again.read_text()


def test_what_is_open_for_a_user_with_one_host_is_what_that_user_must_write(tmp_path, capsys):
    """A stranger was told to replace the environment with a cluster's, then met each missing value at a start."""
    from agent_flow.workflows.trtllm_modeling_v2.spec import intake

    out = tmp_path / "out"
    (tmp_path / "repo").mkdir()
    argv = ["derive", "--checkpoint", str(_checkpoint(tmp_path)), "--repo", str(tmp_path / "repo"),
            "--goal", "tp1 -mtp text_only arch=sm_100 model_key=org/newmodel", "--out", str(out)]  # fmt: skip
    assert intake.main(argv) == 0
    said = capsys.readouterr().out
    opened = [line for line in said.splitlines() if line.startswith("  open: ")]
    # The interpreter and the scratch variables, each once, and nothing that is not this user's.
    assert [line.split(":")[1].strip() for line in opened] == [
        "environment.python",
        "environment.env",
    ]
    assert "set TMPDIR and CLAUDE_CODE_TMPDIR to a scratch directory" in opened[1]
    assert "this one host" in said
    assert "environment.python" in (out / derive.REPORT_FILENAME).read_text()
    # A site that states them leaves nothing open about the environment.
    assert intake.main([*argv[:3], "--site", str(_site(tmp_path)), *argv[5:]]) == 0
    assert "open: environment" not in capsys.readouterr().out


def test_a_feature_derive_has_no_options_for_is_written_and_asked_about(tmp_path):
    with pytest.raises(derive.DeriveError, match="host_offload: its deployment `host_offload`"):
        _derive(tmp_path, "tp1 host_offload")
    spec = yaml.safe_load((tmp_path / "ws" / derive.SPEC_FILENAME).read_text())
    assert [d["name"] for d in spec["deployments"]] == ["tp1", "host_offload"]


def test_quantized_core_has_one_kv_dtype_source(tmp_path):
    spec = load_and_validate_task_yaml(_derive(tmp_path, "tp1 kv=fp8/nvfp4"))
    assert _options(spec, "tp1")["kv_cache_config"]["dtype"] == "fp8"
    assert _options(spec, "kv_nvfp4")["kv_cache_config"]["dtype"] == "nvfp4"
