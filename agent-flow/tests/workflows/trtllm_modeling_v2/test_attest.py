# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attestation: a deployment's options against what every rank's engine ran."""

from __future__ import annotations

import enum
import json
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import attest
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import engine_facts, workload_io
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Server

TARGET_CLASS = "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.model.GptOssModel"
OPTIONS = {
    "tensor_parallel_size": 2,
    "enable_attention_dp": True,
    "max_batch_size": 64,
    "cuda_graph_config": {},
    "speculative_config": None,
    "kv_cache_config": {"dtype": "nvfp4", "free_gpu_memory_fraction": 0.6},
    "a_dtype_option_added_tomorrow": "fp8",
}
#: What the engine resolved: every stated key, with the defaults it filled in around them.
RESOLVED = {
    **OPTIONS,
    "cuda_graph_config": {"batch_sizes": [1, 2], "enable_padding": True},
    "kv_cache_config": {"dtype": "nvfp4", "free_gpu_memory_fraction": 0.6, "tokens_per_block": 32},
    "max_seq_len": None,
}


def rank_lines(
    options: dict, /, *, tree: str = "/repo", ranks: int | None = None, **changes
) -> str:
    """The record each rank of an engine running ``options`` from ``tree`` prints."""
    count = ranks if ranks is not None else int(options.get("tensor_parallel_size") or 1)
    lines = []
    for rank in range(count):
        record = {
            "rank": rank,
            "options": options,
            "target": TARGET_CLASS,
            "origin": f"{tree}/tensorrt_llm/__init__.py",
            **changes,
        }
        lines.append(f"{workload_io.RANK_ENGINE_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


def test_declared_equals_observed_on_every_rank_with_no_option_known_by_name():
    found = attest.attest(rank_lines(RESOLVED), OPTIONS, tree="/repo")
    assert found.status == attest.MATCH and found.failure() is None
    assert (found.ranks, found.keys) == (2, 8)
    assert "modeling_v2" in found.target and found.origin == "/repo/tensorrt_llm/__init__.py"


@pytest.mark.parametrize(
    "resolved,said",
    [
        ({"max_batch_size": 32}, "rank 0: max_batch_size is 32, declared 64"),
        ({"kv_cache_config": {"dtype": "fp8"}}, "kv_cache_config.dtype is 'fp8', declared 'nvfp4'"),
        ({"cuda_graph_config": None}, "cuda_graph_config is None, declared {}"),
        (
            {"speculative_config": {"decoding_type": "MTP"}},
            "speculative_config is {'decoding_type'",
        ),
        ({"a_dtype_option_added_tomorrow": "auto"}, "a_dtype_option_added_tomorrow is 'auto'"),
        # Text is compared as written: a path or a name that differs by case is another one.
        ({"a_dtype_option_added_tomorrow": "FP8"}, "a_dtype_option_added_tomorrow is 'FP8'"),
    ],
)
def test_a_key_the_engine_resolved_otherwise_fails_the_check(resolved, said):
    found = attest.attest(rank_lines({**RESOLVED, **resolved}), OPTIONS)
    assert found.status == attest.MISMATCH and said in found.failure()


def test_a_value_that_leaves_the_choice_to_the_engine_records_what_each_rank_ran():
    declared = {
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {"dtype": "Auto"},
        "max_batch_size": 4,
    }
    ran = {
        "cache_transceiver_config": {"backend": "NIXL", "max_tokens_in_buffer": 4096},
        "kv_cache_config": {"dtype": "fp8"},
        "max_batch_size": 4,
    }
    found = attest.attest(rank_lines(ran), declared)
    assert found.status == attest.MATCH and found.failure() is None
    assert found.resolved == (
        "cache_transceiver_config.backend: declared DEFAULT, ran NIXL",
        "kv_cache_config.dtype: declared Auto, ran fp8",
    )
    assert "target " in attest.describe([found.record()])
    assert attest.describe([found.record()]).endswith(
        "; cache_transceiver_config.backend: declared DEFAULT, ran NIXL; "
        "kv_cache_config.dtype: declared Auto, ran fp8]"
    )
    # Every other value is still compared exactly, and a block the engine has off is no choice.
    assert attest.attest(rank_lines({**ran, "max_batch_size": 8}), declared).status == (
        attest.MISMATCH
    )
    off = attest.attest(rank_lines({**ran, "cache_transceiver_config": None}), declared)
    assert "cache_transceiver_config.backend is None, declared 'DEFAULT'" in off.failure()
    named = {**declared, "cache_transceiver_config": {"backend": "UCX"}}
    assert "is 'NIXL', declared 'UCX'" in attest.attest(rank_lines(ran), named).failure()
    almost = attest.attest(rank_lines({"backend": "NIXL"}), {"backend": "defaults"})
    assert almost.status == attest.MISMATCH and not almost.resolved


def test_a_missing_rank_or_key_reads_unknown_and_never_passes():
    one_rank = attest.attest(rank_lines(RESOLVED, ranks=1), OPTIONS)
    assert one_rank.status == attest.UNKNOWN
    assert "rank 1: it printed no record" in one_rank.failure()
    assert "unknown is not a pass" in one_rank.failure()
    no_key = {k: v for k, v in RESOLVED.items() if k != "enable_attention_dp"}
    assert "rank 0: enable_attention_dp is not in the engine's configuration" in (
        attest.attest(rank_lines(no_key), OPTIONS).failure()
    )
    unread = attest.attest(rank_lines(RESOLVED, options=None, error="AttributeError: x"), OPTIONS)
    assert "rank 0: AttributeError: x" in unread.failure()
    assert attest.attest("", OPTIONS).status == attest.UNKNOWN


def test_the_model_class_and_the_import_origin_are_attested_on_every_rank():
    stock = rank_lines(RESOLVED, target="tensorrt_llm._torch.models.modeling_qwen3.Qwen3")
    assert "not a Modeling V2 target" in attest.attest(stock, OPTIONS).failure()
    # A control run on the stock path is not held to a Modeling V2 class.
    assert attest.attest(stock, OPTIONS, target=False).status == attest.MATCH
    assert (
        "the model class it built"
        in attest.attest(rank_lines(RESOLVED, target=None), OPTIONS).failure()
    )
    elsewhere = attest.attest(rank_lines(RESOLVED, tree="/opt/other"), OPTIONS, tree="/repo")
    assert "imported tensorrt_llm from /opt/other/tensorrt_llm/__init__.py" in elsewhere.failure()


def test_a_value_inside_a_list_and_every_rank_of_every_parallel_size_are_held():
    inputs = [{"name": "token_type_ids", "dtype": "int32"}]
    declared = {"cuda_graph_config": {"extra_model_inputs": inputs}}
    other = {"cuda_graph_config": {"extra_model_inputs": [{**inputs[0], "dtype": "float32"}]}}
    assert attest.attest(rank_lines(declared), declared).status == attest.MATCH
    assert attest.attest(rank_lines(other), declared).status == attest.MISMATCH
    # An engine's ranks are its tensor, pipeline and context parallel sizes, multiplied.
    wide = {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
    assert Server(options=wide).ranks == 4
    half = attest.attest(rank_lines(wide, ranks=2), wide)
    assert (half.status, half.ranks) == (attest.UNKNOWN, 4)
    assert attest.attest(rank_lines(wide, ranks=4), wide).status == attest.MATCH
    # The model class is in the Modeling V2 package, not one whose name contains it.
    lookalike = rank_lines({}, target="tensorrt_llm._torch.models.not_modeling_v2.Stock")
    assert attest.attest(lookalike, {}).status == attest.MISMATCH


def test_a_repeated_prompt_must_come_back_with_cached_tokens():
    # One repeat; with attention DP a request may land on any rank, so one more per rank.
    assert attest.reuse_sends({"tensor_parallel_size": 4}) == 2
    assert attest.reuse_sends({"tensor_parallel_size": 4, "enable_attention_dp": True}) == 5
    assert attest.reuse_shortfall([0, 448]) is None
    assert attest.reuse_shortfall([0, 0, 0, 448, 0]) is None
    assert "the engine reused nothing" in attest.reuse_shortfall([0, 0])
    assert "unknown is not a pass" in attest.reuse_shortfall([0, None])
    assert "the engine reused nothing" in attest.reuse_shortfall([0, True])  # no count of tokens


def test_each_server_of_a_session_is_attested_by_its_own_records():
    ctx = rank_lines({"max_batch_size": 4}, instance="ctx0", devices=["0"], host="node1")
    gen = rank_lines(
        {"tensor_parallel_size": 2, "max_batch_size": 8},
        instance="gen0",
        devices=["1", "2"],
        host="node1",
    )
    output = ctx + gen
    found = attest.attest(
        output,
        {"tensor_parallel_size": 2, "max_batch_size": 8},
        role="gen",
        instance="gen0",
        devices=["1", "2"],
    )
    assert found.status == attest.MATCH and found.ranks == 2
    assert found.hosts == ("node1",) and found.devices == ("1", "2")
    assert "gen0: match" in attest.describe([found.record()])
    # The context server's record never stands in for a generation rank.
    alone = attest.attest(ctx, {"tensor_parallel_size": 2}, role="gen", instance="gen0")
    assert alone.status == attest.UNKNOWN and "the gen0 server" in alone.failure()
    # A rank shown other devices than its server was given ran somewhere else.
    moved = attest.attest(output, {"max_batch_size": 4}, role="ctx", instance="ctx0", devices=["3"])
    assert "was shown devices 0, not the 3 the session gave its server" in moved.failure()
    unsaid = attest.attest(
        rank_lines({}, instance="ctx0"), {}, role="ctx", instance="ctx0", devices=["0"]
    )
    assert "the devices it was shown" in unsaid.failure()


def test_two_servers_of_one_node_never_share_a_device():
    records = [
        {"instance": "ctx0", "host": "node1", "devices": ["0"]},
        {"instance": "gen0", "host": "node1", "devices": ["1", "2"]},
    ]
    assert attest.devices_problem(records, {"ctx0": 1, "gen0": 2}) == ""
    shared = [records[0], {"instance": "gen0", "host": "node1", "devices": ["0", "1"]}]
    assert "gen0 and ctx0 were both given device 0 of node1" in attest.devices_problem(
        shared, {"ctx0": 1, "gen0": 2}
    )
    # The same device id on another host is another device.
    apart = [records[0], {"instance": "gen0", "host": "node2", "devices": ["0", "1"]}]
    assert attest.devices_problem(apart, {"ctx0": 1, "gen0": 2}) == ""
    assert "2 rank(s) need 2 device(s)" in attest.devices_problem(
        [records[0], {"instance": "gen0", "host": "node1", "devices": ["1"]}],
        {"ctx0": 1, "gen0": 2},
    )
    assert "said nothing of its devices" in attest.devices_problem(records[:1], {"gen0": 2})


def test_a_rank_says_every_field_of_its_configuration_and_where_it_runs(monkeypatch, capsys):
    class Dtype(enum.Enum):
        FP8 = "fp8"

    class Field:
        def __init__(self, alias=None):
            self.alias, self.validation_alias = alias, None

    class Cache:
        model_fields = {"dtype": Field(), "enable_block_reuse": Field("block_reuse")}
        dtype, enable_block_reuse = Dtype.FP8, True

    class Args:
        model_fields = {"max_batch_size": Field(), "kv_cache_config": Field()}
        max_batch_size, kv_cache_config = 8, Cache()

    assert engine_facts.plain_config(Args()) == {
        "max_batch_size": 8,
        "kv_cache_config": {"dtype": "fp8", "enable_block_reuse": True, "block_reuse": True},
    }
    monkeypatch.setitem(
        __import__("sys").modules,
        "tensorrt_llm",
        SimpleNamespace(__file__="/repo/tensorrt_llm/__init__.py"),
    )
    worker = SimpleNamespace(llm_args=Args(), engine=None)
    environ = {"CUDA_VISIBLE_DEVICES": "2,3", "MV2FLOW_ROLE": "gen", "MV2FLOW_INSTANCE": "1"}
    seen = engine_facts.observe(worker, environ)
    assert seen["options"]["kv_cache_config"]["dtype"] == "fp8"
    assert seen["origin"] == "/repo/tensorrt_llm/__init__.py" and seen["target"] is None
    assert (seen["devices"], seen["role"], seen["instance"]) == (["2", "3"], "gen", "1")
    # As the hook prints it once the engine is up: what a session attests, and the rank. A
    # worker that cannot be read says so, which a session reads as a rank it could not attest.
    monkeypatch.delenv(workload_io.RANK_RECORDS_ENV, raising=False)
    worker.rank = 1
    engine_facts.emit_rank(worker)
    engine_facts.emit_rank(SimpleNamespace(rank=2))
    first, second = workload_io.parse_marked(
        capsys.readouterr().out, workload_io.RANK_ENGINE_MARKER
    )
    assert first["rank"] == 1 and first["options"]["max_batch_size"] == 8 and "error" not in first
    assert set(second) == {"rank", "error"} and "AttributeError" in second["error"]


def test_each_rank_keeps_its_records_in_a_file_of_its_own(tmp_path):
    wio = engine_facts.wio
    environ = {wio.RANK_RECORDS_ENV: str(tmp_path), "MV2FLOW_ROLE": "gen0"}
    for number in (0, 1):
        for turn in range(200):
            engine_facts.keep_rank_record({"rank": number, "turn": turn}, environ)
    kept = wio.kept_records(tmp_path, "gen0")
    assert len(kept) == 400
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "rank-engine.gen0.0.jsonl",
        "rank-engine.gen0.1.jsonl",
    ]
    # Another role instance's ranks are not this one's, and no directory means no file.
    assert wio.kept_records(tmp_path, "ctx0") == []
    engine_facts.keep_rank_record({"rank": 0}, {"MV2FLOW_ROLE": "gen0"})
    assert len(wio.kept_records(tmp_path, "gen0")) == 400
    # A rank's worker that cannot be read still says so, in its file as on its line.
    environ = {wio.RANK_RECORDS_ENV: str(tmp_path), "MV2FLOW_ROLE": "server"}
    engine_facts.keep_rank_record({"rank": 3, "error": "AttributeError: no engine"}, environ)
    assert wio.kept_records(tmp_path, "server") == [
        {"rank": 3, "error": "AttributeError: no engine"}
    ]
