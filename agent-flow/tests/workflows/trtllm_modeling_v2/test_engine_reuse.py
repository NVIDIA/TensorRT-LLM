# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block reuse is attested by what the engine reused, at every member.

The engine attestation recorded that block reuse was configured, not that the
engine reused anything: the accuracy logs showed reuse on and no hit counts,
and at every multi-rank member the boot gate attested 11 of 12 fields, block
reuse NOT ATTESTED, because rank 0's worker runs in the launcher's leader
process and the harness read only ``llm.args``. These tests hold the harness
to reading the prompt tokens every response says the engine resumed from its
KV cache, and a probe that sends one prompt twice; the attestation to taking
block reuse from them where no worker was read; and the boot, acceptance and
accuracy gates of a unit that asks for reuse to fail when the engine reused
nothing although it had a prefix to reuse. Engine output is canned or comes
from a fake ``tensorrt_llm`` package; no GPU is used.
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import (
    engine_facts,
    engine_reuse,
    eval_shim,
    workload_io,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

from .test_engine_attestation import (
    _BINDS,
    _FAKE_TRTLLM,
    _OPTIONS,
    _SNAPSHOTS,
    _declared,
    _engine_config,
    _engine_line,
    _env,
    _records,
    _task,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

_POLICY = {**_SNAPSHOTS, "additional_snapshot_offsets_from_end": []}


def _reuse_line(**fields) -> str:
    record = {
        "requests": 3,
        "reported": 3,
        "hits": 0,
        "reused_tokens": 0,
        "resumed_at": {},
        "min_resume": 256,
        "resumable": 0,
        "concurrency": 1,
        "probe": {"tokens": 320, "first": 0, "reused": 256},
    }
    record.update(fields)
    return f"{workload_io.REUSE_MARKER} {json.dumps(record)}\n"


# ------------------------------------------------------------ the harness


def test_each_prompt_is_matched_against_every_earlier_one():
    prompts = [(1, 2, 3, 4), (1, 2, 3, 9), (5, 6), None, (1, 2, 3, 4, 7), (5, 6, 1)]
    assert engine_reuse.shared_prefixes(prompts) == [0, 3, 0, None, 4, 2]
    assert engine_reuse.common_prefix((1, 2, 3), (1, 2, 3)) == 3
    assert engine_reuse.common_prefix((), (1,)) == 0


@pytest.mark.parametrize(
    "policy,block,resume,probe",
    [
        (_POLICY, 32, 256, 320),
        ({"periodic_snapshot_interval": 2048}, 32, 2048, 2112),
        (None, 32, 32, 96),
        ({"periodic_snapshot_interval": 0}, 64, 64, 192),
        ({"additional_snapshot_offsets_from_end": [128]}, 32, None, 192),
    ],
)
def test_the_engine_resumes_at_its_first_snapshot_or_one_block(policy, block, resume, probe):
    assert engine_reuse.min_resume(policy, block) == resume
    assert engine_reuse.probe_tokens(policy, block) == probe


def _output(ids, cached):
    return types.SimpleNamespace(prompt_token_ids=list(ids), cached_tokens=cached)


def test_the_ledger_counts_what_was_resumed_and_what_could_have_been():
    prefix = list(range(300))
    ledger = engine_reuse.ReuseLedger()
    ledger.add(_output(prefix + [1], 0))
    ledger.add(_output(prefix + [2], 256))
    ledger.add(_output(prefix[:100] + [3], 0))
    ledger.add(types.SimpleNamespace(prompt_token_ids=[7, 8]))
    later = _output(prefix + [4], 256)
    ledger.track(later)
    summary = ledger.summary(min_resume=256, concurrency=1, probe={"tokens": 320, "reused": 256})
    assert summary == {
        "requests": 5,
        "reported": 4,
        "hits": 2,
        "reused_tokens": 512,
        "resumed_at": {"256": 2},
        "min_resume": 256,
        "resumable": 2,
        "concurrency": 1,
        "probe": {"tokens": 320, "reused": 256},
    }
    assert ledger.summary(min_resume=None, concurrency=None)["resumable"] is None


def test_a_tracked_request_is_read_when_the_record_is_made():
    """The shims hand each request over as it is submitted; it finishes later."""
    ledger = engine_reuse.ReuseLedger()
    handle = _output([1, 2], None)
    ledger.track(handle)
    handle.cached_tokens = 32
    assert ledger.entries() == [((1, 2), 32)]


def test_the_probe_sends_one_prompt_twice_and_reads_the_second(monkeypatch):
    module = types.ModuleType("tensorrt_llm.scheduling_params")

    class SchedulingParams:
        def __init__(self, **fields):
            self.fields = fields

    module.SchedulingParams = SchedulingParams
    monkeypatch.setitem(sys.modules, "tensorrt_llm", types.ModuleType("tensorrt_llm"))
    monkeypatch.setitem(sys.modules, "tensorrt_llm.scheduling_params", module)
    sent: list = []

    class Handle:
        def __init__(self, output):
            self.output = output

        def result(self):
            return self.output

    class Engine:
        tokenizer = types.SimpleNamespace(encode=lambda text, add_special_tokens: [5, 6, 7])

        def generate_async(self, inputs, params, **kwargs):
            sent.append((inputs, kwargs))
            return Handle(_output(inputs["prompt_token_ids"], 256 if len(sent) > 1 else 0))

    record = engine_reuse.run_probe(Engine(), "params", 8, max_seq_len=64, attention_dp=True)
    assert record == {"tokens": 8, "first": 0, "reused": 256}
    assert [s[0]["prompt_token_ids"] for s in sent] == [[5, 6, 7, 5, 6, 7, 5, 6]] * 2
    assert all(
        s[1]["scheduling_params"].fields == {"attention_dp_rank": 0, "attention_dp_relax": False}
        for s in sent
    ), "both on the attention-DP rank whose cache holds the first"
    sent.clear()
    assert engine_reuse.run_probe(Engine(), "params", 8)["reused"] == 256
    assert sent[0][1] == {}
    too_long = engine_reuse.run_probe(Engine(), "params", 64, max_seq_len=64)
    assert too_long["skipped"].startswith("64 prompt tokens do not fit max_seq_len 64")
    unchunked = engine_reuse.run_probe(Engine(), "params", 64, max_prompt=32)
    assert unchunked["skipped"] == (
        "64 prompt tokens exceed max_num_tokens 32 without chunked prefill"
    )


def test_the_probe_prompt_is_prose_cut_to_length():
    def plain(text):
        return [ord(c) for c in text[:5]]

    assert (
        engine_reuse.probe_prompt(types.SimpleNamespace(encode=plain), 12)
        == ([ord(c) for c in engine_reuse.PROBE_TEXT[:5]] * 3)[:12]
    )
    fallback = engine_reuse.probe_prompt(None, 200)
    assert len(fallback) == 200 and min(fallback) >= 1000


def test_the_granularity_is_the_engines():
    state = types.SimpleNamespace(**_SNAPSHOTS, additional_snapshot_offsets_from_end=[])
    kv = types.SimpleNamespace(mamba_state_config=state, tokens_per_block=16)
    llm = types.SimpleNamespace(args=types.SimpleNamespace(kv_cache_config=kv))
    assert engine_reuse.engine_granularity(llm) == (_POLICY, 16)
    llm.get_kv_cache_capacity = lambda: {"tokensPerBlock": 64, "maxNumTokens": 6400}
    assert engine_reuse.engine_granularity(llm) == (_POLICY, 64), "the executor's own block"
    assert engine_reuse.engine_granularity(types.SimpleNamespace()) == (None, 32)


def test_a_record_that_cannot_be_made_is_reported(capsys):
    record = engine_reuse.emit(engine_reuse.ReuseLedger(), ("not", "a granularity"), concurrency=1)
    assert record["error"].startswith(("TypeError", "ValueError", "AttributeError"))
    [line] = capsys.readouterr().out.splitlines()
    assert gates.parse_reuse_records(line) == [record]


def test_without_a_worker_the_kv_pool_comes_from_rank_0s_executor():
    llm = types.SimpleNamespace(
        args=types.SimpleNamespace(),
        get_kv_cache_capacity=lambda: {"tokensPerBlock": 32, "maxNumTokens": 1932480},
    )
    record = engine_facts.collect(llm, None, environ={})
    assert record["worker"] is False and record["pools"] == {"kv_max_tokens": 1932480}

    def unreachable():
        raise RuntimeError("RPC client not initialized")

    assert engine_facts.kv_capacity(types.SimpleNamespace(get_kv_cache_capacity=unreachable)) == {}


# ------------------------------------------------------------ the attestation


def test_block_reuse_is_attested_by_the_probe_where_no_worker_was_read(tmp_path):
    """The multi-rank case: the record is llm.args, with block reuse left out."""
    config = _engine_config()
    del config["block_reuse"]
    head = _engine_line(config, worker=False)
    on = gates.attest_engine(head + _BINDS + _reuse_line(), _declared(tmp_path))
    assert not on.not_attested and not on.mismatches
    assert on.config["block_reuse"] is True and on.sources["block_reuse"] == gates.PROBE_SOURCE
    record = on.record()
    assert record["reuse"]["probe"] == {"tokens": 320, "first": 0, "reused": 256}
    assert record["reuse"]["recurrent"] is True and record["reuse"]["snapshot_restores"] == 0
    assert "reuse probe resumed at 256 of 320 (snapshot restored)" in gates.describe_engine(
        [record]
    )
    assert gates.describe_engine([record]).startswith(" [engine 12/12 fields as declared;")

    off = gates.attest_engine(
        head + _reuse_line(probe={"tokens": 320, "first": 0, "reused": 0}), _declared(tmp_path)
    )
    assert off.config["block_reuse"] is False
    assert "block_reuse: declared on, engine off (read from reuse probe)" in off.mismatches[0]


def test_a_request_the_engine_resumed_attests_reuse_and_none_resumed_does_not(tmp_path):
    config = _engine_config()
    del config["block_reuse"]
    head = _engine_line(config, worker=False)
    hit = _reuse_line(hits=11, reused_tokens=8448, resumed_at={"768": 11}, probe=None)
    attested = gates.attest_engine(head + hit, _declared(tmp_path))
    assert attested.sources["block_reuse"] == gates.CACHED_TOKENS_SOURCE
    assert attested.reuse["snapshot_restores"] == 11
    none = gates.attest_engine(head + _reuse_line(probe=None), _declared(tmp_path))
    assert none.not_attested == ("block_reuse",), "no reuse may only mean nothing to reuse"


def test_the_worker_reading_stands_beside_the_probe(tmp_path):
    attested = gates.attest_engine(
        _engine_line() + _reuse_line(probe={"tokens": 320, "first": 0, "reused": 0}),
        _declared(tmp_path),
    )
    assert attested.sources["block_reuse"] == "worker" and not attested.mismatches
    assert gates.reuse_shortfall(attested, asks=True), "configured on, and reused nothing"


def test_a_shortfall_is_judged_only_where_there_was_something_to_reuse(tmp_path):
    declared = _declared(tmp_path)

    def shortfall(asks=True, **fields):
        return gates.reuse_shortfall(
            gates.attest_engine(_engine_line() + _reuse_line(**fields), declared), asks=asks
        )

    probe = shortfall(probe={"tokens": 320, "first": 0, "reused": 0})
    assert probe.startswith(
        "the unit asks for KV-cache block reuse and the engine reused nothing: the reuse probe "
        "sent one 320-token prompt twice, the second after the first had finished, and the "
        "engine computed the second from token 0 (cached_tokens 0 on its response), where an "
        "engine that reuses resumes it at token 256 or later"
    )
    assert "kv_cache_config.mamba_state_config" in probe
    assert shortfall() is None, "the probe resumed"
    assert shortfall(asks=False, probe={"tokens": 320, "reused": 0}) is None
    assert shortfall(reported=0, probe={"tokens": 320, "reused": 0}) is None, "not reported"
    assert shortfall(probe={"tokens": 320, "skipped": "too long"}) is None

    workload = dict(requests=1319, reported=1319, resumable=1318, probe=None)
    many = shortfall(concurrency=128, **workload)
    assert many.startswith(
        "the unit asks for KV-cache block reuse and the engine reused nothing: 1318 of 1319 "
        "requests shared more than 256 prompt tokens with an earlier request, more than the "
        "128 the engine runs at once, and every response says it was computed from token 0"
    )
    assert shortfall(concurrency=2048, **workload) is None, "all may have run side by side"
    assert shortfall(concurrency=None, **workload) is None
    assert shortfall(concurrency=128, hits=1, **workload) is None


def test_the_summary_says_what_was_reused(tmp_path):
    def described(line: str, *, binds: str = _BINDS) -> str:
        return gates.describe_engine(
            [gates.attest_engine(_engine_line() + binds + line, _declared(tmp_path)).record()]
        )

    gsm8k = _reuse_line(
        requests=1319,
        reported=1319,
        hits=1318,
        reused_tokens=1012224,
        resumed_at={"768": 1318},
        resumable=1318,
        concurrency=128,
        probe=None,
    )
    assert (
        "; reuse 1318/1319 requests, 1012224 tokens (recurrent state restored at 768 x1318);"
        in (described(gsm8k))
    )
    one = _reuse_line(hits=1, reused_tokens=64, resumed_at={"64": 1}, probe=None)
    assert "reuse 1/3 requests, 64 tokens (recurrent state restored at 64 x1)" in (
        described(one, binds="")
    ), "the engine's snapshot policy says it keeps recurrent state"
    no_policy = _engine_config(
        mamba_snapshot={
            **_POLICY,
            "periodic_snapshot_interval": 0,
            "additional_snapshot_offsets_from_start": [],
        }
    )
    plain = gates.attest_engine(_engine_line(no_policy) + one, _declared(tmp_path)).record()
    assert "reuse 1/3 requests, 64 tokens (resumed at 64 x1)" in gates.describe_engine([plain])
    quiet = described(_reuse_line())
    assert "reuse 0/3" not in quiet and "reuse probe resumed at 256 of 320" in quiet
    assert "reuse not reported by the engine" in described(_reuse_line(reported=0, probe=None))
    assert "reuse probe not read (too long)" in described(
        _reuse_line(probe={"tokens": 320, "skipped": "too long"})
    )
    assert "reuse not read (KeyError: x)" in described(
        f'{workload_io.REUSE_MARKER} {{"error": "KeyError: x"}}\n'
    )
    old = gates.attest_engine(_engine_line(), _declared(tmp_path)).record()
    assert old["reuse"] == {} and "reuse" not in gates.describe_engine([old])


# ------------------------------------------------------------ the gates


def _runner(*outputs: str):
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=outputs[min(len(seen), len(outputs)) - 1])

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_the_boot_and_acceptance_drivers_send_the_probe(tmp_path):
    task = _task(tmp_path)
    gates.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_runner(_engine_line() + _records("Paris", "light")),
    )  # fmt: skip
    spec = json.loads((tmp_path / "boot.core.tp1.json").read_text())
    assert spec["reuse_probe"] is True
    plain = gates.write_driver_spec(tmp_path / "plain.json", task_data=task, argv=[])
    assert "reuse_probe" not in json.loads(plain.read_text()), "other drivers' specs as before"


def test_a_boot_gate_fails_when_the_engine_reused_nothing(tmp_path):
    task = _task(tmp_path)
    nothing = _reuse_line(probe={"tokens": 320, "first": 0, "reused": 0})
    result = gates.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_runner(_engine_line() + _records("Paris", "light") + nothing),
    )  # fmt: skip
    assert not result.passed
    assert result.detail.startswith("the unit asks for KV-cache block reuse and the engine")
    assert result.engine[0]["reuse"]["probe"]["reused"] == 0

    task["runtime"]["core"] = {"kv_cache_reuse": False}
    off = gates.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_runner(_engine_line(_engine_config(block_reuse=False)) + _records("Paris", "light")
                       + nothing),
    )  # fmt: skip
    assert off.passed, "a unit that turns reuse off asks for none"


def test_an_acceptance_leg_that_reused_nothing_fails_the_gate(tmp_path):
    task = _task(tmp_path, features=["mtp"])
    options = {**_OPTIONS, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    mtp = _engine_config(speculative={"algorithm": "MTP", "max_draft_len": 3})
    nothing = _reuse_line(probe={"tokens": 320, "first": 0, "reused": 0})
    run = _runner(_engine_line(mtp) + nothing)
    result = gates.run_acceptance_gate(
        _env(), task_data=task, runner=run, unit="core", member="tp1", workdir=tmp_path
    )
    assert len(run.seen) == 1 and not result.passed
    assert result.detail.startswith(
        "the unit asks for KV-cache block reuse and the engine of the require leg reused nothing"
    )
    spec = json.loads((tmp_path / "acceptance.core.tp1.json").read_text())
    assert spec["reuse_probe"] is True


def _accuracy(tmp_path, stdout: str, **task_changes) -> gates.GateResult:
    task = _task(tmp_path, **task_changes)
    task["accuracy"].update({"floor": {"rule": "tolerance"}, "tolerance": 5.0})
    return gates.run_accuracy_gate(
        _env(), task_data=task, reference=90.0, unit="core", workdir=tmp_path,
        runner=_runner(stdout + "lm-eval gsm8k average accuracy: 90.00\n"),
    )  # fmt: skip


def test_the_accuracy_gate_fails_when_the_benchmark_reused_nothing(tmp_path):
    workload = dict(requests=1319, reported=1319, resumable=1318, concurrency=128, probe=None)
    config = _engine_config()
    del config["block_reuse"]
    head = _engine_line(config, worker=False)
    nothing = _accuracy(tmp_path, head + _reuse_line(**workload))
    assert not nothing.passed and nothing.measured is None
    assert nothing.detail.startswith(
        "the unit asks for KV-cache block reuse and the engine reused nothing: 1318 of 1319"
    )
    reused = _accuracy(
        tmp_path, head + _reuse_line(hits=1318, reused_tokens=1012224, resumed_at={"768": 1318},
                                     **workload)
    )  # fmt: skip
    assert reused.passed, reused.detail
    [record] = reused.engine
    assert record["sources"]["block_reuse"] == gates.CACHED_TOKENS_SOURCE
    assert record["reuse"]["snapshot_restores"] == 1318
    assert "[engine 1/1 fields as declared; reuse 1318/1319 requests, 1012224 tokens" in (
        reused.summary()
    )


def test_the_accuracy_gate_holds_block_reuse_to_the_options_file(tmp_path):
    off = _accuracy(tmp_path, _engine_line(_engine_config(block_reuse=False)))
    assert not off.passed
    assert "block_reuse: declared on, engine off (read from worker)" in off.detail
    options = {**_OPTIONS, "kv_cache_config": {**_OPTIONS["kv_cache_config"],
                                                "enable_block_reuse": False}}  # fmt: skip
    path = tmp_path / "off.llm_api.yaml"
    path.write_text(yaml.safe_dump(options))
    declared_off = _accuracy(
        tmp_path,
        _engine_line(_engine_config(block_reuse=False)) + _reuse_line(probe=None),
        accuracy={"benchmark": "gsm8k", "llm_api_options": {"core": str(path)}},
    )
    assert declared_off.passed, declared_off.detail
    assert gates.llm_api_block_reuse({}) is True
    assert gates.llm_api_block_reuse({"kv_cache_config": {"enable_block_reuse": False}}) is False


def test_the_eval_command_keeps_trtllm_evals_argv():
    environment = RunEnvironment(kind="local", python="/py", world_size=4)
    command = gates.eval_harness_command(environment, ["trtllm-eval", '--model="/m"', "gsm8k"])
    assert command.endswith(f'/py {gates.EVAL_SHIM} trtllm-eval --model="/m" gsm8k')
    assert "mpirun -n 4" in command and "trtllm-llmapi-launch /py" in command


# ------------------------------------------- end to end, through the harness

_FAKE_LLM = """
    import os
    import types

    from tensorrt_llm.executor.base_worker import BaseWorker


    class MambaStateConfig:
        def __init__(self, periodic_snapshot_interval=0,
                     additional_snapshot_offsets_from_start=(),
                     additional_snapshot_offsets_from_end=()):
            self.periodic_snapshot_interval = periodic_snapshot_interval
            self.additional_snapshot_offsets_from_start = list(
                additional_snapshot_offsets_from_start)
            self.additional_snapshot_offsets_from_end = list(
                additional_snapshot_offsets_from_end)


    class KvCacheConfig:
        def __init__(self, **fields):
            self._stated = set(fields)
            self.enable_block_reuse = fields.get("enable_block_reuse", True)
            self.dtype = fields.get("dtype", "auto")
            self.mamba_ssm_cache_dtype = fields.get("mamba_ssm_cache_dtype", "auto")
            self.free_gpu_memory_fraction = fields.get("free_gpu_memory_fraction", 0.9)
            self.tokens_per_block = fields.get("tokens_per_block", 32)
            state = fields.get("mamba_state_config") or {}
            if isinstance(state, dict):
                state = MambaStateConfig(**state)
            self.mamba_state_config = state
            self.max_gpu_total_bytes = 0

        @property
        def model_fields_set(self):
            return self._stated


    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]


    class RequestOutput:
        # As the LLM API's: the handle `generate_async` returns is the output itself.

        def __init__(self, ids, cached):
            self.prompt_token_ids = ids
            self.cached_tokens = cached
            self.outputs = [types.SimpleNamespace(
                token_ids=[11, 12], text=" Paris", finish_reason="length")]

        def __iter__(self):
            yield self

        def result(self):
            return self


    class LLM:
        # Resumes a prompt where an earlier one left a recurrent-state snapshot, as a
        # hybrid engine does; FAKE_REUSE=0 never resumes, FAKE_WORKER=0 sets its worker
        # up in another process.

        def __init__(self, model=None, **kw):
            kv = kw.get("kv_cache_config") or KvCacheConfig()
            if isinstance(kv, dict):
                kv = KvCacheConfig(**kv)
            parallel = types.SimpleNamespace(
                tp_size=kw.get("tensor_parallel_size", 1), pp_size=1,
                moe_ep_size=kw.get("moe_expert_parallel_size") or -1, moe_tp_size=-1,
                enable_attention_dp=kw.get("enable_attention_dp", False))
            self.args = types.SimpleNamespace(
                modeling_v2=kw.get("modeling_v2", "off"), kv_cache_config=kv,
                cuda_graph_config=kw.get("cuda_graph_config"), speculative_config=None,
                max_num_tokens=kw.get("max_num_tokens", 8192),
                max_batch_size=kw.get("max_batch_size", 2048),
                max_seq_len=kw.get("max_seq_len"),
                enable_chunked_prefill=kw.get("enable_chunked_prefill", False),
                disable_overlap_scheduler=False, parallel_config=parallel)
            self.tokenizer = Tokenizer()
            self.seen = []
            if os.environ.get("FAKE_WORKER", "1") == "1":
                BaseWorker(self.args).setup_engine()

        def get_kv_cache_capacity(self):
            return {"maxNumBlocks": 100, "tokensPerBlock": 32, "maxNumTokens": 3200}

        def _resumed(self, ids):
            kv = self.args.kv_cache_config
            if os.environ.get("FAKE_REUSE", "1") != "1" or not kv.enable_block_reuse:
                return 0
            shared = 0
            for earlier in self.seen:
                n = 0
                while n < min(len(ids), len(earlier)) and ids[n] == earlier[n]:
                    n += 1
                shared = max(shared, n)
            limit = min(shared, len(ids) - 1)
            state = kv.mamba_state_config
            points = [o for o in state.additional_snapshot_offsets_from_start if o <= limit]
            if state.periodic_snapshot_interval:
                points += list(range(state.periodic_snapshot_interval, limit + 1,
                                     state.periodic_snapshot_interval))
            return max(points, default=0)

        def generate_async(self, inputs, params=None, streaming=False, **kwargs):
            if isinstance(inputs, dict):
                ids = list(inputs["prompt_token_ids"])
            else:
                ids = self.tokenizer.encode(inputs)
            cached = self._resumed(ids)
            self.seen.append(ids)
            return RequestOutput(ids, cached)

        def shutdown(self):
            pass
"""

_FAKE_QUICKSTART = """
    import argparse
    import types

    from tensorrt_llm import LLM, KvCacheConfig


    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--max_tokens", type=int, default=64)
        parser.add_argument("--tp_size", type=int, default=1)
        parser.add_argument("--moe_ep_size", type=int, default=-1)
        parser.add_argument("--enable_attention_dp", action="store_true")
        parser.add_argument("--max_seq_len", type=int)
        parser.add_argument("--max_batch_size", type=int, default=2048)
        parser.add_argument("--max_num_tokens", type=int, default=8192)
        parser.add_argument("--kv_cache_fraction", type=float, default=0.9)
        parser.add_argument("--mamba_ssm_cache_dtype", default="auto")
        parser.add_argument("--kv_cache_dtype", default="auto")
        parser.add_argument("--disable_kv_cache_reuse", action="store_true")
        parser.add_argument("--enable_chunked_prefill", action="store_true")
        parser.add_argument("--use_cuda_graph", action="store_true")
        parser.add_argument("--cuda_graph_padding_enabled", action="store_true")
        parser.add_argument("--cuda_graph_batch_sizes", type=int, nargs="+")
        return parser.parse_args()


    def setup_llm(args):
        kv = KvCacheConfig(
            enable_block_reuse=not args.disable_kv_cache_reuse,
            free_gpu_memory_fraction=args.kv_cache_fraction,
            dtype=args.kv_cache_dtype,
            mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        )
        graphs = None
        if args.use_cuda_graph:
            graphs = types.SimpleNamespace(
                batch_sizes=sorted(args.cuda_graph_batch_sizes or [1]),
                enable_padding=args.cuda_graph_padding_enabled)
        llm = LLM(
            model=args.model_dir, kv_cache_config=kv, cuda_graph_config=graphs,
            tensor_parallel_size=args.tp_size, moe_expert_parallel_size=args.moe_ep_size,
            enable_attention_dp=args.enable_attention_dp, max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size, max_num_tokens=args.max_num_tokens,
            enable_chunked_prefill=args.enable_chunked_prefill)
        return llm, types.SimpleNamespace(max_tokens=args.max_tokens, ignore_eos=False)
"""

#: A ``trtllm-eval`` that sends twelve few-shot prompts sharing an 840-token prefix.
_FAKE_EVAL = """
    import sys

    import yaml

    from tensorrt_llm import LLM as PyTorchLLM

    SHOTS = "Question: a farmer sows three fields with wheat. Answer: 3. " * 14


    def main():
        options = {}
        for word in sys.argv[1:]:
            if word.startswith("--extra_llm_api_options="):
                with open(word.split("=", 1)[1]) as handle:
                    options = yaml.safe_load(handle) or {}
        llm = PyTorchLLM(
            model="m", kv_cache_config=options.get("kv_cache_config"),
            max_batch_size=options.get("max_batch_size", 2048),
            max_num_tokens=options.get("max_num_tokens", 8192),
            enable_chunked_prefill=options.get("enable_chunked_prefill", False),
            modeling_v2=options.get("modeling_v2", "off"))
        handles = [llm.generate_async(SHOTS + f"Question {n}: how many?") for n in range(12)]
        for handle in handles:
            handle.result()
        llm.shutdown()
        print("lm-eval gsm8k average accuracy: 90.00")
        sys.exit(0)
"""


def _fake_site(tmp_path: Path, *, eval_module: str = _FAKE_EVAL) -> tuple[Path, Path]:
    """``(site, repo)``: a fake ``tensorrt_llm`` whose engine reuses, and a tree with its quickstart."""
    site = tmp_path / "site"
    files = {
        **_FAKE_TRTLLM,
        "tensorrt_llm/__init__.py": _FAKE_LLM,
        "tensorrt_llm/commands/__init__.py": "",
        "tensorrt_llm/commands/eval.py": eval_module,
    }
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    repo = tmp_path / "repo"
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(_FAKE_QUICKSTART), encoding="utf-8")
    return site, repo


def _real(site: Path, **env) -> RunEnvironment:
    return RunEnvironment(
        kind="local", python=sys.executable, world_size=1, env={"PYTHONPATH": str(site), **env}
    )


def _real_boot(tmp_path: Path, **env) -> gates.GateResult:
    site, repo = _fake_site(tmp_path)
    task = _task(tmp_path, trtllm_repo_path=str(repo))
    return gates.run_boot_gate(
        _real(site, **env), task_data=task, unit="core", member="tp1", workdir=tmp_path / "gates"
    )


def test_a_real_boot_attests_reuse_from_the_probe_without_a_worker(tmp_path):
    """Rank 0's worker in another process, as under trtllm-llmapi-launch: 12/12 all the same."""
    result = _real_boot(tmp_path, FAKE_WORKER="0")
    assert result.passed, f"{result.detail}\n{result.stdout_tail}"
    [record] = result.engine
    assert record["sources"]["block_reuse"] == gates.PROBE_SOURCE and not record["not_attested"]
    assert record["reuse"]["probe"] == {"tokens": 320, "first": 0, "reused": 256}
    assert record["pools"]["kv_max_tokens"] == 3200, "from rank 0's executor"
    assert (
        "[engine 12/12 fields as declared; KV 3200 tokens; reuse probe resumed at 256 of 320 "
        in (result.summary())
    )


def test_a_real_boot_fails_on_an_engine_that_reused_nothing(tmp_path):
    for worker in ("1", "0"):
        result = _real_boot(tmp_path / worker, FAKE_WORKER=worker, FAKE_REUSE="0")
        assert not result.passed, result.stdout_tail
        assert "the engine reused nothing: the reuse probe sent one 320-token prompt" in (
            result.detail
        )


def test_the_shim_reports_what_the_quickstart_reused(tmp_path):
    """The quickstart runs each prompt once together and once alone: the second resumes."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import witness

    site, repo = _fake_site(tmp_path)
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    script.write_text(
        script.read_text()
        + textwrap.dedent(
            """
            def main():
                args = parse_arguments()
                llm, params = setup_llm(args)
                prompt = "one two three four " * 80
                llm.generate_async(prompt, params)
                llm.generate_async(prompt, params, streaming=True)
            """
        )
    )
    import subprocess

    done = subprocess.run(
        [sys.executable, str(witness.QUICKSTART_SHIM), str(script), "--model_dir=/ckpt",
         "--max_batch_size=8"],
        capture_output=True, text=True, timeout=120, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site),
             gates.KV_CACHE_CONFIG_ENV: json.dumps({"mamba_state_config": _SNAPSHOTS})},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    [reuse] = gates.parse_reuse_records(done.stdout)
    assert (reuse["requests"], reuse["hits"], reuse["resumed_at"]) == (2, 1, {"768": 1})
    assert (reuse["min_resume"], reuse["resumable"], reuse["concurrency"]) == (256, 1, 8)


def _real_accuracy(tmp_path: Path, *, eval_module: str = _FAKE_EVAL, **env) -> gates.GateResult:
    site, repo = _fake_site(tmp_path, eval_module=eval_module)
    task = _task(tmp_path, trtllm_repo_path=str(repo))
    options = {**_OPTIONS, "max_batch_size": 4}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    task["accuracy"].update({"floor": {"rule": "tolerance"}, "tolerance": 5.0})
    return gates.run_accuracy_gate(
        _real(site, **env), task_data=task, reference=90.0, unit="core", member="tp1",
        workdir=tmp_path / "gates",
    )  # fmt: skip


def test_a_real_accuracy_gate_counts_the_restores_through_the_eval_shim(tmp_path):
    result = _real_accuracy(tmp_path)
    assert result.passed, f"{result.detail}\n{result.stdout_tail}"
    [record] = result.engine
    # The gates' rank hook had the worker print its own record beside the shim's.
    assert record["reported"] and record["sources"]["block_reuse"] == gates.RANKS_SOURCE
    reuse = record["reuse"]
    assert (reuse["requests"], reuse["hits"], reuse["resumable"]) == (12, 11, 11)
    assert reuse["resumed_at"] == {"768": 11} and reuse["snapshot_restores"] == 11
    assert reuse["concurrency"] == 4
    assert "reuse 11/12 requests, 8448 tokens (recurrent state restored at 768 x11)" in (
        result.summary()
    )


def test_a_real_accuracy_gate_fails_when_no_few_shot_prompt_resumed(tmp_path):
    result = _real_accuracy(tmp_path, FAKE_WORKER="0", FAKE_REUSE="0")
    assert not result.passed, result.stdout_tail
    assert result.detail.startswith(
        "the unit asks for KV-cache block reuse and the engine reused nothing: 11 of 12 "
        "requests shared more than 256 prompt tokens with an earlier request, more than the "
        "4 the engine runs at once"
    )


def test_an_eval_with_no_llm_class_to_wrap_runs_as_is(tmp_path):
    bare = _FAKE_EVAL.replace(" as PyTorchLLM", "").replace("PyTorchLLM(", "LLM(")
    result = _real_accuracy(tmp_path, eval_module=bare)
    assert result.passed, result.stdout_tail
    # The shim prints nothing of the engine; its worker, through the rank hook, still does.
    assert not gates.parse_engine_records(result.stdout_tail)
    assert result.engine[0]["reported"] and result.engine[0]["ranks"] == [0]
    assert not gates.parse_reuse_records(result.stdout_tail)


def test_the_eval_shim_insists_on_the_program_it_stands_in_for():
    with pytest.raises(SystemExit, match="usage: eval_shim.py trtllm-eval"):
        eval_shim.main(["trtllm-bench", "--model", "m"])


# ----------------------------------------------------- the engine's own code


def test_the_names_the_reuse_reading_relies_on_are_the_engines():
    """Read from the tree's own source, so a rename fails here and not in a gate."""
    executor = REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "py_executor.py"
    result = REPO_ROOT / "tensorrt_llm" / "executor" / "result.py"
    llm = REPO_ROOT / "tensorrt_llm" / "llmapi" / "llm.py"
    evaluate = REPO_ROOT / "tensorrt_llm" / "commands" / "eval.py"
    schedule = REPO_ROOT / "tensorrt_llm" / "scheduling_params.py"
    if not all(p.is_file() for p in (executor, result, llm, evaluate, schedule)):
        pytest.skip("the TensorRT-LLM sources are not in this checkout")
    assert "response.result.cached_tokens = request.cached_tokens" in executor.read_text()
    assert "self.cached_tokens = getattr(response_result, 'cached_tokens', 0)" in (
        result.read_text()
    )
    assert "def get_kv_cache_capacity(self) -> dict:" in llm.read_text()
    assert "from .. import LLM as PyTorchLLM" in evaluate.read_text()
    text = schedule.read_text()
    assert "attention_dp_rank: Optional[int]" in text and "attention_dp_relax: bool" in text
