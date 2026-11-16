# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One session per deployment: servers loaded once, every check a client, one line each.

The servers are stand-ins (``conftest.site``): what is under test is the session's own
order, placement, evidence and judgment, not an engine.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import session
from agent_flow.workflows.trtllm_modeling_v2.checks.check_runtime import (
    default_shell_runner,
    in_tree,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import session_driver
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

from .conftest import engine_definitions

MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}
BOOT = {"prompts": [{"prompt": "The capital of France is", "keywords_any": ["Paris"]}]}


@pytest.fixture
def task(tmp_path, site):
    """A spec's data for sessions on this host: a tree, a checkpoint, a declared anchor."""
    tree = tmp_path / "tree"
    (tree / "tensorrt_llm").mkdir(parents=True)
    engine_definitions(tree)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(tree),
        "environment": {"kind": "local", "python": sys.executable, "gpus_per_node": 8},
        "accuracy": {"boot": BOOT},
    }


def run(task, block, name=None, **more):
    data = {**task, "deployments": block}
    deployment = deployments.named(data, name)
    environment = RunEnvironment(kind="local", python=sys.executable)
    runner = in_tree(default_shell_runner, data["trtllm_repo_path"])
    workdir = data["trtllm_repo_path"] + "/../sessions"
    return session.run_session(
        deployment, environment, task_data=data, workdir=workdir, runner=runner, **more
    )


def lines(result):
    return {line.name: line for line in result.lines}


SCORED = [{"name": "gsm8k", "anchor": 41.77, "source": "measured on the stock path"}]


def test_an_aggregated_deployment_is_one_load_and_every_check_has_its_line(task):
    options = {"max_num_tokens": 64, "kv_cache_config": {"enable_block_reuse": True}}
    result = run(task, [{"name": "tp1", "options": options, "benchmarks": SCORED}])
    assert result.result.passed, result.render()
    assert result.loads == 1
    assert [line.name for line in result.lines] == [
        "devices",
        "attest",
        "boot",
        "api",
        "accuracy[gsm8k]",
        "reuse",
    ]
    found = lines(result)
    assert all(line.passed and line.member == "tp1" for line in result.lines)
    assert "tp1" in found["boot"].detail and "accuracy.boot" in found["boot"].detail
    assert found["accuracy[gsm8k]"].measured == 41.70
    assert "through the endpoint" in found["accuracy[gsm8k]"].detail
    assert "448 cached tokens" in found["reuse"].detail
    assert "1 model load(s) for 1 role instance(s)" in result.result.detail
    # The order is part of what was measured: scoring before the probe that fills the cache.
    assert "order: devices, attest, boot, api, accuracy[gsm8k], reuse" in result.result.detail
    assert result.result.name == "deployment" and result.result.engine[0]["status"] == "match"


def test_two_ranks_that_print_at_once_are_each_read_whole_from_a_file_of_their_own(
    task, monkeypatch, tmp_path
):
    """On GPUs two ranks of one server printed at once: one line of 16,526 characters."""
    block = [{"name": "tp2", "options": {"tensor_parallel_size": 2}}]
    monkeypatch.setenv("FAKE_MERGED_RANKS", "1")
    result = run(task, block)
    assert result.result.passed, result.render()
    assert "2 rank(s) of 1 server(s)" in lines(result)["attest"].detail
    directory = tmp_path / "sessions" / "tp2"
    # The log the two ranks share holds one broken line, which names neither rank.
    log = (directory / "server.log").read_text()
    assert log.count("MV2FLOW_RANK_ENGINE") == 2
    assert session_driver.wio.parse_marked(log, "MV2FLOW_RANK_ENGINE") == []
    # Each rank's own file holds its record whole.
    for rank in (0, 1):
        (kept,) = (directory / f"rank-engine.server.{rank}.jsonl").read_text().splitlines()
        assert json.loads(kept)["rank"] == rank
    # A rank that only printed is then unknown, as it was on GPUs: unknown is not a pass.
    monkeypatch.setenv("FAKE_NO_RANK_FILES", "1")
    result = run(task, block)
    attested = lines(result)["attest"]
    assert not attested.passed
    assert "rank 0: it printed no record; rank 1: it printed no record" in attested.detail
    # A session started again in the same place reads nothing of the one before.
    assert not list(directory.glob("rank-engine.*"))


def test_unequal_roles_on_one_node_get_disjoint_devices_and_one_load_each(task, tmp_path):
    block = [
        {
            "name": "pd",
            "ctx": {"options": MOVES},
            "gen": {"options": {"tensor_parallel_size": 2, **MOVES}, "instances": 2},
            "benchmarks": SCORED,
        }
    ]
    result = run(task, block)
    assert result.result.passed, result.render()
    assert result.loads == 3 and result.result.member == "pd5"
    found = lines(result)
    assert [line.name for line in result.lines] == [
        "devices",
        "attest",
        "boot",
        "api",
        "hop",
        "accuracy[gsm8k]",
    ]
    assert "ctx0 on" in found["devices"].detail
    for given in (": 0;", ": 1,2;", ": 3,4"):
        assert given in found["devices"].detail + ";"
    assert "5 rank(s) of 3 server(s)" in found["attest"].detail
    attested = [r for r in found["attest"].engine if "status" in r]
    assert {r["instance"]: r["devices"] for r in attested} == {
        "ctx0": ("0",),
        "gen0": ("1", "2"),
        "gen1": ("3", "4"),
    }
    assert found["hop"].detail == (
        "generation received the context's state for 1 of 1 request(s): each response's "
        "`X-TRTLLM-Start-End-Time` names the transfer interval of `gen` (`kv-start` to "
        "`kv-end`), as `tensorrt_llm/web/request_times.py` of the measured tree defines it"
    )
    assert "through the hop" in found["accuracy[gsm8k]"].detail
    assert "3 model load(s) for 3 role instance(s)" in result.result.detail
    # Each server was told its role, and the proxy was given every address.
    directory = tmp_path / "sessions" / "pd"
    started = json.loads((directory / "address.gen1.started.json").read_text())
    assert started["argv"][started["argv"].index("--server_role") + 1] == "generation"
    assert started["env"]["CUDA_VISIBLE_DEVICES"] == "3,4"
    # Its ranks name their role instance, which is what the job's stated ranks are counted by.
    assert started["env"]["MV2FLOW_ROLE"] == "gen1" and started["env"]["MV2FLOW_INSTANCE"] == "1"
    assert started["env"]["MV2FLOW_SERVER_ROLE"] == "gen"
    assert session.stated_ranks(deployments.named({"deployments": block})) == "ctx0:1,gen0:2,gen1:2"
    import yaml

    proxy = yaml.safe_load((directory / "proxy.yaml").read_text())
    assert proxy["context_servers"]["num_instances"] == 1
    assert proxy["generation_servers"]["num_instances"] == 2
    # What the session turned on to read the hop is in the options it wrote, not the spec's.
    written = yaml.safe_load((directory / "options.gen.yaml").read_text())
    assert written["return_perf_metrics"] is True and written["modeling_v2"] == "require"


PD = [{"name": "pd", "ctx": {"options": MOVES}, "gen": {"options": MOVES}}]


def test_answers_that_name_no_transfer_leave_the_hop_not_verified_and_fail_nothing(
    task, monkeypatch
):
    monkeypatch.setenv("FAKE_NO_HOP", "1")
    result = run(task, PD)
    hop = lines(result)["hop"]
    assert hop.passed and hop.unverified.startswith("1 of 1 response(s) name no transfer interval")
    assert hop.summary().startswith("[NOT VERIFIED] hop@pd2: not verified: 1 of 1 response(s)")
    assert "1 request(s) were answered through the proxy" in hop.detail
    # The line says which observation the engine's answer lacks, and what it did name.
    assert (
        "lacks the generation side's start or end of the transfer (it names disagg-start, "
        "ctx-start, ctx-end, gen-start, gen-end)" in hop.unverified
    )
    # The deployment is not failed for it, and its own line says what was not verified.
    assert result.result.passed, result.render()
    assert result.result.detail.startswith("4 check(s) passed; 1 not verified: hop; 2 model load")
    assert result.result.unverified.startswith("hop: 1 of 1 response(s)")
    # Nor when the proxy returns no metrics at all, though the tree defines the header.
    monkeypatch.setenv("FAKE_NO_METRICS", "1")
    hop = lines(run(task, PD))["hop"]
    assert hop.passed and hop.unverified.startswith(
        "no response carried `X-TRTLLM-Start-End-Time`, which "
        "`tensorrt_llm/web/request_times.py` of the measured tree defines for a request sent "
        "with `X-TRTLLM-return-metrics: 1`"
    )


def test_the_means_of_observing_a_transfer_are_found_by_their_definitions(tmp_path):
    means = session_driver.hop_means
    assert means(None)["means"] == means(tmp_path / "absent")["means"] == "none"
    engine_definitions(tmp_path)
    assert means(tmp_path) == {
        "means": "headers",
        "request_header": "X-TRTLLM-return-metrics",
        "header": "X-TRTLLM-Start-End-Time",
        "begin": "kv-start",
        "end": "kv-end",
        "source": "tensorrt_llm/web/request_times.py",
    }
    # An engine that renames what it sends, or moves the file, is followed, not assumed.
    source = tmp_path / "tensorrt_llm" / "web" / "request_times.py"
    moved = source.with_name("moved.py")
    moved.write_text(
        source.read_text()
        .replace("X-TRTLLM-Start-End-Time", "X-Engine-Times")
        .replace('"kv-start"', '"state-begin"')
        .replace('"kv-end"', '"state-end"')
    )
    source.unlink()
    found = means(tmp_path)
    assert (found["header"], found["begin"], found["end"], found["source"]) == (
        "X-Engine-Times",
        "state-begin",
        "state-end",
        "tensorrt_llm/web/moved.py",
    )
    # One that no longer names a transfer in what it sends offers no way.
    moved.write_text('RETURN_METRICS_HEADER = "X-TRTLLM-return-metrics"  # kv_cache_transfer\n')
    none = means(tmp_path)
    assert none["means"] == "none" and "1 file(s) name a state transfer" in none["why"]
    intervals = session.transfer_intervals
    header = "disagg-start;ts=1.0, ctx-start;ts=1.1, gen-kv-start;ts=1.2, gen-kv-end;ts=1.4"
    assert intervals(header, "kv-start", "kv-end") == {"gen": (1.2, 1.4)}
    # As a proxy on GPUs answered: the context names the start of its send and no end; only
    # the generation side has an interval.
    served = (
        "disagg-start;ts=3800703.185538000, disagg-end;ts=3800703.278289000, "
        "ctx-start;ts=3800703.186547000, ctx-end;ts=3800703.205365000, "
        "ctx-kv-start;ts=3800703.203773000, gen-start;ts=3800703.207083500, "
        "gen-end;ts=3800703.277075500, gen-kv-start;ts=3800703.208232500, "
        "gen-kv-end;ts=3800703.208956501"
    )
    assert intervals(served, "kv-start", "kv-end") == {"gen": (3800703.2082325, 3800703.208956501)}
    # A start with no end, an end before its start and a zero are no interval.
    assert intervals("gen-kv-start;ts=1.2", "kv-start", "kv-end") == {}
    assert intervals("gen-kv-start;ts=2, gen-kv-end;ts=1", "kv-start", "kv-end") == {}
    assert intervals("gen-kv-start;ts=0, gen-kv-end;ts=0", "kv-start", "kv-end") == {}
    assert intervals(None, "kv-start", "kv-end") == {}


def test_a_rank_that_ran_other_options_fails_attestation_and_the_rest_is_still_said(
    task, monkeypatch
):
    monkeypatch.setenv("FAKE_DTYPE", "fp8")
    options = {"kv_cache_config": {"dtype": "auto"}, "max_batch_size": 8}
    result = run(task, [{"name": "tp1", "options": options}])
    found = lines(result)
    # `auto` leaves the choice to the engine: what it chose is shown, and is no mismatch.
    assert found["attest"].passed
    assert "declared auto, ran fp8" in " ".join(found["attest"].engine[0]["resolved"])
    monkeypatch.setenv("FAKE_DTYPE", "nvfp4")
    result = run(task, [{"name": "tp1", "options": {"kv_cache_config": {"dtype": "fp8"}}}])
    found = lines(result)
    assert not found["attest"].passed
    assert "kv_cache_config.dtype is 'nvfp4', declared 'fp8'" in found["attest"].detail
    assert found["boot"].passed and not result.result.passed
    assert "1 of 4 check(s) failed: attest" in result.result.detail


def test_a_benchmark_without_an_anchor_is_blocked_before_the_load_and_the_rest_goes_on(task):
    block = [{"name": "tp1", "options": {}, "benchmarks": [{"name": "gsm8k"}]}]
    result = run(task, block)
    found = lines(result)
    blocked = found["accuracy[gsm8k]"]
    assert blocked.blocked and not blocked.passed and "no anchor" in blocked.detail
    assert blocked.summary().startswith("[BLOCKED] accuracy[gsm8k]@tp1")
    assert found["boot"].passed and found["attest"].passed
    assert not result.result.passed and result.result.blocked
    assert "1 check(s) blocked before the load: accuracy[gsm8k]" in result.result.detail
    # The evaluator was never started.
    assert "accuracy[gsm8k]" not in json.dumps(
        json.loads((task_dir(task) / "tp1" / "session.json").read_text())["steps"]
    )


def task_dir(task):
    return Path(task["trtllm_repo_path"]).parent / "sessions"


NO_DATA = "Couldn't reach 'openai/gsm8k' on the Hub (OfflineModeIsEnabled)"


def test_a_benchmark_whose_evaluator_cannot_load_its_data_is_blocked_before_any_server(
    task, monkeypatch
):
    monkeypatch.setenv("FAKE_NO_DATA", NO_DATA)
    result = run(task, [{"name": "tp1", "options": {}, "benchmarks": SCORED}])
    found = lines(result)
    # The job asked the evaluator's own loader on its node, and the line is its answer.
    assert found["accuracy[gsm8k]"].blocked.endswith(NO_DATA)
    assert found["boot"].passed and found["attest"].passed and found["api"].passed
    assert "1 check(s) blocked before the load: accuracy[gsm8k]" in result.result.detail
    # It was asked before any server was started, and the evaluator scored nothing.
    said = (task_dir(task) / "tp1" / "session.out").read_text()
    assert said.index('"blocked"') < said.index(session_driver.SERVER_MARKER)
    assert not (task_dir(task) / "tp1" / "accuracy.gsm8k.log").exists()
    # A benchmark left to `trtllm-eval` builds the same evaluator after a load of its own:
    # asked here the same, it has no such load.
    own = [{**SCORED[0], "through": "eval"}]
    result = run(task, [{"name": "tp1", "options": {}, "benchmarks": own}])
    assert lines(result)["accuracy[gsm8k]"].blocked.endswith(NO_DATA)
    assert result.own_loads == () and lines(result)["boot"].passed


def test_a_score_under_its_floor_fails_its_line_alone(task, monkeypatch):
    monkeypatch.setenv("FAKE_SCORE", "30.00")
    result = run(task, [{"name": "tp1", "options": {}, "benchmarks": SCORED}])
    found = lines(result)
    assert not found["accuracy[gsm8k]"].passed and found["accuracy[gsm8k]"].measured == 30.0
    assert found["boot"].passed and not result.result.passed


def test_a_benchmark_the_endpoint_cannot_carry_is_left_for_a_load_of_its_own(task):
    benchmarks = [
        {"name": "mmlu", "anchor": 70.0, "source": "measured"},
        {"name": "gsm8k", "anchor": 41.77, "source": "measured", "through": "eval"},
    ]
    result = run(task, [{"name": "tp1", "options": {}, "benchmarks": benchmarks}])
    assert result.result.passed, result.render()
    assert [p.name for p in result.own_loads] == ["accuracy[mmlu]", "accuracy[gsm8k]"]
    assert "accuracy[mmlu]" not in lines(result)
    # Through a hop there is no such load: an aggregated engine never crosses it.
    pd = [{"name": "pd", "ctx": {"options": MOVES}, "gen": {"options": MOVES}}]
    pd[0]["benchmarks"] = benchmarks[:1]
    blocked = lines(run(task, pd))["accuracy[mmlu]"]
    assert blocked.blocked and "never crosses the hop" in blocked.detail


def test_an_image_is_sent_when_the_deployment_takes_images(task, tmp_path, monkeypatch):
    image = tmp_path / "tree" / "tests" / "integration" / "test_input_files" / "merlion.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG fake")
    block = [{"name": "vl", "options": {}, "inputs": ["text", "image"]}]
    result = run(task, block)
    assert lines(result)["image"].passed, result.render()
    monkeypatch.setenv("FAKE_NO_IMAGES", "1")
    refused = lines(run(task, block))["image"]
    assert not refused.passed and "this model takes no image" in refused.detail


def test_an_image_the_tree_holds_as_an_lfs_pointer_is_read_from_the_checkouts_store(task, tmp_path):
    content = b"\x89PNG the photograph"
    oid = hashlib.sha256(content).hexdigest()
    tree, store = tmp_path / "tree", tmp_path / "checkout" / ".git"
    image = tree / "tests" / "integration" / "test_input_files" / "merlion.png"
    image.parent.mkdir(parents=True)
    image.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 19\n")
    # As a pin's tree: a git directory of its own, which names the checkout's.
    (tree / ".git").mkdir()
    (tree / ".git" / "commondir").write_text(f"{store}\n")
    block = [{"name": "vl", "options": {}, "inputs": ["text", "image"]}]
    # A checkout cloned without git-lfs has no such object: the line says what gets past it.
    lacking = lines(run(task, block))["image"]
    assert not lacking.passed and "is not in the LFS store of the checkout" in lacking.detail
    assert "`images` of a prompt in `accuracy.boot.prompts`" in lacking.detail
    # A checkout made with git-lfs keeps it: the image is sent, read from there.
    held = store / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    held.parent.mkdir(parents=True)
    held.write_bytes(content)
    assert lines(run(task, block))["image"].passed


def test_a_long_prompt_is_served_in_chunks_when_the_options_chunk(task):
    options = {"enable_chunked_prefill": True, "max_num_tokens": 64, "max_seq_len": 512}
    result = run(task, [{"name": "chunked", "options": options}])
    chunk = lines(result)["chunk"]
    assert chunk.passed and "a prompt of 129 tokens, past max_num_tokens 64" in chunk.detail
    # Asked of an engine that does not chunk, it is blocked before the load.
    blocked = lines(run(task, [{"name": "plain", "options": {}, "probes": ["chunk"]}]))["chunk"]
    assert blocked.blocked and "do not turn chunked prefill on" in blocked.detail


def test_a_repeated_prompt_that_resumes_nothing_fails_the_reuse_probe(task, monkeypatch):
    monkeypatch.setenv("FAKE_REUSE", "0")
    result = run(task, [{"name": "tp1", "options": {}, "probes": ["reuse"]}])
    reuse = lines(result)["reuse"]
    assert not reuse.passed and "the engine reused nothing" in reuse.detail


def test_a_drafters_acceptance_is_read_from_the_sessions_answers(task, monkeypatch):
    drafting = {"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    accuracy = {"acceptance": {"prompts": [f"Question {n}" for n in range(4)], "min_prompts": 4}}
    task = {**task, "accuracy": {**task["accuracy"], **accuracy}}
    block = [{"name": "mtp", "options": drafting}]
    monkeypatch.setenv("FAKE_ACCEPTED", "30")
    result = run(task, block)
    line = lines(result)["acceptance"]
    assert line.passed and line.measured == pytest.approx(51 / 21)
    assert "read from the session's answers" in line.detail
    assert "held to 1.0 alone" in " ".join(line.warnings)
    assert result.loads == 1
    monkeypatch.setenv("FAKE_ACCEPTED", "0")
    rejected = lines(run(task, block))["acceptance"]
    assert not rejected.passed and "every draft was rejected" in rejected.detail
    # A declared floor is the reference a drafter is held to; no other engine is loaded.
    task["accuracy"]["acceptance"]["mtp"] = {"measured_reference": 3.0, "floor_source": "stock"}
    monkeypatch.setenv("FAKE_ACCEPTED", "30")
    low = lines(run(task, block))["acceptance"]
    assert not low.passed and "under the declared floor 2.8500 (stock)" in low.detail


def test_a_server_that_dies_at_start_says_its_error_line_and_every_check_has_a_line(
    task, monkeypatch
):
    monkeypatch.setenv("FAKE_SERVE_EXIT", "1")
    result = run(task, [{"name": "tp1", "options": {}, "benchmarks": SCORED}])
    assert not result.result.passed
    assert "the deployment did not start" in result.result.detail
    assert "Failed to create NIXL backend" in result.result.detail
    found = lines(result)
    assert found["devices"].passed
    assert not found["boot"].passed and "did not start" in found["boot"].detail
    assert result.loads == 1


def test_a_deployment_wider_than_one_node_is_refused_where_no_step_crosses_nodes(task):
    block = [{"name": "wide", "options": {"tensor_parallel_size": 16}}]
    # This one host has no second node, and its environment says no way to run a step on one.
    result = run(task, block)
    assert not result.result.passed and result.result.blocked and result.lines == ()
    assert "`wide` holds 16 devices and one node has 8" in result.result.detail
    assert "which needs environment.slurm. the flow starts engines" in result.result.detail
    wide = deployments.named({"deployments": block})
    with pytest.raises(session.WiderThanOneNode):
        session.place({**task, "deployments": block}, wide)
    # With a way to run a step it is placed: a job of whole nodes, each rank on a device.
    slurm = {"environment": {"slurm": {"gpus_per_node": 8}}, "deployments": block}
    placed = session.place(slurm, wide)
    assert (placed.gpus, placed.nodes) == (16, 2)
    assert [(slot.node, len(slot.devices)) for slot in placed.layout["server"]] == [(0, 8), (1, 8)]
    # `trtllm-eval` loads on one node: across nodes such a benchmark is blocked before the load.
    across = [{**block[0], "benchmarks": [{**SCORED[0], "through": "eval"}]}]
    planned = session.plan(
        {**slurm, "checkpoint_path": task["checkpoint_path"], "deployments": across},
        deployments.named({"deployments": across}),
        directory=Path(task["trtllm_repo_path"]),
    )
    blocked = next(check for check in planned if check.name == "accuracy[gsm8k]")
    assert blocked.blocked.startswith(
        "`trtllm-eval` loads its engine on one node, and `wide` holds 16"
    )
    # One that fits a node stays one job on one node, with no layout of its own.
    narrow = deployments.named(
        {"deployments": [{"name": "tp4", "options": {"tensor_parallel_size": 4}}]}
    )
    assert session.place(slurm, narrow) == session.Placement(gpus=4, wall=placed.wall)


def test_the_job_asks_for_every_role_instances_devices_and_the_sessions_wall(task):
    block = [
        {
            "name": "pd",
            "ctx": {"options": MOVES},
            "gen": {"options": {"tensor_parallel_size": 2, **MOVES}},
            "timeout": 900,
        }
    ]
    placement = session.place(
        {**task, "deployments": block}, deployments.named({"deployments": block})
    )
    assert placement.gpus == 3 and placement.nodes == 1
    assert placement.wall == 900 + session.TEARDOWN_S


def test_two_servers_never_share_a_device():
    plan = session_driver.plan_devices(
        [{"name": "ctx0", "ranks": 1}, {"name": "gen0", "ranks": 2}], ["4", "5", "6", "7"]
    )
    assert plan == {"ctx0": ["4"], "gen0": ["5", "6"]}
    assert session_driver.overlap(plan) == ""
    assert "given to both ctx0 and gen0" in session_driver.overlap({"ctx0": ["4"], "gen0": ["4"]})
    with pytest.raises(ValueError, match="holds 2 device\\(s\\).*need 3"):
        session_driver.plan_devices(
            [{"name": "ctx0", "ranks": 1}, {"name": "gen0", "ranks": 2}], ["0", "1"]
        )
