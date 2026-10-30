# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The geometry probe measures the serving geometry before any kernel is certified.

Kernels certified against estimated shapes were certified again once gates
showed the real pool sizes, page views, chunk splits, capture lists and
verify depths. These tests hold the probe to booting one engine per declared
width, reading what it serves into ``runtime_facts.yaml`` and every prompt's
block, falling back to the stand-in checkpoint only for what the
configuration decides, standing on a recorded probe of the same inputs, and
never stopping the run. The harness runs over a fake ``LLM``; engine output
is canned; no GPU is used.
"""

from __future__ import annotations

import json
import textwrap
import threading
import time
import types
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import geometry_probe, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import geometry
from agent_flow.workflows.trtllm_modeling_v2.run import runtime_facts as rf

_PADDED = [1, 2, 4, 8, 16, 32]


def _task(tmp_path: Path, **extra) -> dict:
    core = tmp_path / "core.yaml"
    core.write_text(
        yaml.safe_dump(
            {
                "max_num_tokens": 4096,
                "kv_cache_config": {"mamba_state_config": {"periodic_snapshot_interval": 1024}},
            }
        ),
        encoding="utf-8",
    )
    mtp = tmp_path / "mtp.yaml"
    mtp.write_text(yaml.safe_dump({"speculative_config": {"max_draft_len": 3}}), encoding="utf-8")
    task = {
        "checkpoint_path": "/ckpt/newmodel",
        "parallel": "unified",
        "members": ["tp1", "tp4", "tep4", "dep4"],
        "trtllm_repo_path": str(tmp_path / "repo"),
        "arch": "sm_103",
        "features": ["mtp"],
        "runtime": {"default": {"cuda_graph": True, "chunked_prefill": True}},
        "accuracy": {
            "benchmark": "gsm8k",
            "llm_api_options": {"core": str(core), "mtp": str(mtp)},
        },
        "environment": {"kind": "local"},
    }
    task.update(extra)
    return task


def _env(world_size: int = 1) -> RunEnvironment:
    return RunEnvironment(kind="local", python="/py", world_size=world_size)


def _engine_output(*, batches=_PADDED, tp=1, ep=1, dp=False, kv=262144, spec=None) -> str:
    config = {
        "cuda_graph": {"batch_sizes": list(batches), "enable_padding": True},
        "cuda_graph_captured": list(batches),
        "speculative": spec,
        "tp": tp,
        "ep": ep,
        "attention_dp": dp,
        "max_batch_size": 64,
        "max_num_tokens": 4096,
    }
    lines = [
        f"{workload_io.ENGINE_MARKER} "
        + json.dumps(
            {
                "config": config,
                "sources": {k: "worker" for k in config},
                "worker": True,
                "pools": {"kv_max_tokens": kv},
                "memory": {
                    "source": "nvml",
                    "devices": [{"device": "0", "free_mib": 9000, "total_mib": 81000}],
                },
            }
        ),
        f"{geometry.GEOMETRY_MARKER} "
        + json.dumps(
            {
                "page": {
                    "tokens_per_block": 32,
                    "blocks": kv // 32,
                    "max_tokens": kv,
                    "max_seq_len": 16384,
                    "max_num_tokens": 4096,
                    "max_blocks_per_seq": 512,
                    "snapshot_policy": {"periodic_snapshot_interval": 1024},
                },
                "chunks": {"2049": [1024, 1024, 1], "8193": [4096, 4096, 1]},
                "errors": {},
            }
        ),
    ]
    return "\n".join(lines) + "\n"


class _Runner:
    """Answers each leg's launch by the member its argv names; records how many ran at once."""

    def __init__(self, answers=None, *, unsupported_for=()):
        self.commands: list[str] = []
        self.answers = answers or {}
        self.unsupported_for = set(unsupported_for)
        self.active = 0
        self.most = 0
        self._lock = threading.Lock()

    def __call__(self, command, cwd, timeout):
        with self._lock:
            self.commands.append(command)
            self.active += 1
            self.most = max(self.most, self.active)
        time.sleep(0.01)
        try:
            for checkpoint in self.unsupported_for:
                if f"--model_dir={checkpoint}" in command:
                    return CommandResult(
                        returncode=1,
                        stderr="RuntimeError: Unknown model architecture: NewModelForCausalLM\n",
                    )
            for needle, output in self.answers.items():
                if needle in command:
                    return CommandResult(returncode=0, stdout=output)
            return CommandResult(returncode=0, stdout=_engine_output())
        finally:
            with self._lock:
                self.active -= 1


# ------------------------------------------------------------- the harness


def test_the_marker_is_the_harnesses():
    assert geometry.GEOMETRY_MARKER == geometry_probe.GEOMETRY_MARKER
    assert geometry.PROBE_MAX_TOKENS == geometry_probe.PROBE_MAX_TOKENS


def test_the_chunks_are_each_iterations_context_tokens_in_order():
    stats = [
        json.dumps({"iter": 3, "inflightBatchingStats": {"numCtxTokens": 0}}),
        {"iter": 2, "inflightBatchingStats": {"numCtxTokens": 1}},
        {"iter": 0, "inflightBatchingStats": {"numCtxTokens": 1024}},
        {"iter": 1, "inflight_batching_stats": {"num_ctx_tokens": 1024}},
    ]
    assert geometry_probe.chunks_of(stats) == [1024, 1024, 1]
    assert geometry_probe.context_tokens({"iter": 0}) is None


def test_probe_lengths_leave_room_to_generate():
    assert geometry_probe.probe_lengths([8193, 257, 257, 0, 4095], 4096) == [257]
    assert geometry_probe.probe_lengths([5, 3], None) == [3, 5]


class _FakeLLM:
    """What the harness reads of an engine: its args, capacity, stats and generate."""

    def __init__(self, chunk: int = 1024, **kwargs):
        self.kwargs = kwargs
        self.chunk = chunk
        self.pending: list[dict] = []
        self.args = types.SimpleNamespace(
            max_seq_len=16384,
            max_num_tokens=4096,
            max_batch_size=64,
            enable_chunked_prefill=True,
            kv_cache_config=types.SimpleNamespace(
                mamba_state_config=types.SimpleNamespace(
                    periodic_snapshot_interval=1024,
                    additional_snapshot_offsets_from_start=[],
                    additional_snapshot_offsets_from_end=[],
                )
            ),
        )
        self.tokenizer = types.SimpleNamespace(vocab_size=32000)
        self.shut = False

    def get_kv_cache_capacity(self):
        return {"maxNumBlocks": 8192, "tokensPerBlock": 32, "maxNumTokens": 262144}

    def generate(self, prompts, params):
        length = len(prompts[0]["prompt_token_ids"])
        done, step = 0, 0
        while done < length:
            size = min(self.chunk, length - done)
            self.pending.append({"iter": step, "inflightBatchingStats": {"numCtxTokens": size}})
            done, step = done + size, step + 1
        self.pending.append({"iter": step, "inflightBatchingStats": {"numCtxTokens": 0}})

    def get_stats(self, timeout=2):
        out, self.pending = self.pending, []
        return out

    def shutdown(self):
        self.shut = True


def test_the_geometry_record_holds_the_page_view_and_the_chunks():
    base = types.SimpleNamespace(max_tokens=8, ignore_eos=False)
    record = geometry_probe.geometry(_FakeLLM(), base, {"chunk_probes": [2049, 40000]})

    assert record["page"]["tokens_per_block"] == 32
    assert record["page"]["max_blocks_per_seq"] == 512
    assert record["page"]["snapshot_policy"]["periodic_snapshot_interval"] == 1024
    assert record["chunks"] == {"2049": [1024, 1024, 1]}, "a length past max_seq_len is dropped"


def test_the_harness_builds_the_quickstarts_engine_with_iteration_statistics(tmp_path, capsys):
    """End to end, as a leg runs it: the tree's quickstart, statistics on, both records out."""
    built: list[_FakeLLM] = []
    quickstart = tmp_path / "quickstart_advanced.py"
    quickstart.write_text(
        textwrap.dedent(
            """
            import types
            LLM = None

            def parse_arguments():
                return types.SimpleNamespace()

            def setup_llm(args):
                return LLM(model="m"), types.SimpleNamespace(max_tokens=8, ignore_eos=False)
            """
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"quickstart": str(quickstart), "quickstart_argv": [], "chunk_probes": [2049]}),
        encoding="utf-8",
    )
    original = geometry_probe.engine_driver.load_quickstart

    def load(path):
        module = original(path)

        def make(**kwargs):
            llm = _FakeLLM(**kwargs)
            built.append(llm)
            return llm

        module.LLM = make
        return module

    geometry_probe.engine_driver.load_quickstart = load
    try:
        assert geometry_probe.main(["--spec", str(spec)]) == 0
    finally:
        geometry_probe.engine_driver.load_quickstart = original

    out = capsys.readouterr().out
    assert built and built[0].kwargs.get("enable_iter_perf_stats") is True
    assert built[0].shut
    assert workload_io.parse_marked(out, workload_io.ENGINE_MARKER)
    record = workload_io.parse_marked(out, geometry_probe.GEOMETRY_MARKER)[-1]
    assert record["chunks"] == {"2049": [1024, 1024, 1]}


# ------------------------------------------------------------- the legs


def test_the_legs_are_the_narrowest_the_widest_attention_dp_and_speculation(tmp_path):
    legs = geometry.plan_legs(_task(tmp_path), 4)
    assert [leg.key for leg in legs] == ["core@tp1", "core@tep4", "core@dep4", "mtp@tp1"]
    assert legs[-1].features == ("mtp",)


def test_a_concrete_segment_is_one_leg(tmp_path):
    task = _task(tmp_path, parallel="tp2", features=[])
    task.pop("members")
    assert [leg.key for leg in geometry.plan_legs(task, 2)] == ["core@tp2"]


def test_probe_lengths_cross_the_token_budget_and_the_snapshot_interval(tmp_path):
    lengths = geometry.probe_lengths(_task(tmp_path), "core")
    assert {4097, 1025}.issubset(lengths), "one past the budget and past the interval"
    assert len(lengths) <= geometry.MAX_PROBE_LENGTHS


def test_the_declared_verify_depths_are_each_speculative_units(tmp_path):
    assert geometry.declared_verify_depths(_task(tmp_path)) == {"mtp": 3}


def test_a_leg_boots_the_stock_path_at_its_member_with_the_gates_argv(tmp_path):
    leg = geometry.ProbeLeg("tep4")
    command, argv = geometry.leg_command(
        _task(tmp_path), _env(4), leg, mode=gates.MODE_OFF, workdir=tmp_path / "w"
    )
    assert str(geometry.GEOMETRY_PROBE) in command
    assert "TRTLLM_MODELING_V2=off" in command
    assert "--tp_size=4" in argv and "--moe_ep_size=4" in argv
    spec = json.loads((tmp_path / "w" / "geometry.core.tep4.json").read_text())
    assert 4097 in spec["chunk_probes"]


# ------------------------------------------------------------- the probe


def _probe(tmp_path, task, runner, **kwargs):
    return geometry.probe_geometry(
        task,
        workspace=tmp_path / "ws",
        env_for=lambda member: _env(int(member[-1])),
        runner_for=lambda member: runner,
        world_size=4,
        **kwargs,
    )


def test_the_probe_records_each_leg_in_the_runtime_facts(tmp_path):
    runner = _Runner()
    section = _probe(tmp_path, _task(tmp_path), runner)

    assert len(runner.commands) == 4
    assert runner.most == 1, "a local run boots one leg at a time"
    leg = section["legs"]["core@tp1"]
    assert leg["status"] == geometry.STATUS_MEASURED
    assert leg["numbers"]["cuda_graph_batches"] == _PADDED
    assert leg["numbers"]["kv_tokens"] == 262144
    assert leg["page"]["tokens_per_block"] == 32
    assert leg["chunks"]["2049"] == [1024, 1024, 1]
    assert section["verify_depths"] == {"mtp": 3}
    assert rf.geometry(tmp_path / "ws")["legs"].keys() == section["legs"].keys()


def test_a_scheduler_boots_the_legs_at_once(tmp_path):
    runner = _Runner()
    task = _task(tmp_path, environment={"kind": "session_spool"})
    _probe(tmp_path, task, runner)
    assert runner.most > 1


def test_every_prompt_carries_the_geometry_and_the_values_a_row_states(tmp_path):
    _probe(tmp_path, _task(tmp_path), _Runner())
    block = rf.render(tmp_path / "ws")

    assert rf.GEOMETRY_HEADING in block
    assert "- core@tep4:" in block and "KV pages of 32 tokens" in block
    assert "2049 -> [1024, 1024, 1]" in block
    assert "mtp 3 (decode q_len 4)" in block
    values = rf.geometry_values(rf.geometry(tmp_path / "ws"))
    assert values["decode_q_len"] == [1, 4]
    assert values["kv_tokens_per_block"] == [32]
    assert values["context_chunk_tokens"] == [1, 1024, 4096]


def test_an_unknown_architecture_falls_back_to_the_stand_in_for_the_configuration(tmp_path):
    task = _task(tmp_path, environment={"kind": "local", "substrate_checkpoint": "/ckpt/stand_in"})
    runner = _Runner(unsupported_for=("/ckpt/newmodel",))
    section = _probe(tmp_path, task, runner)

    leg = section["legs"]["core@tp1"]
    assert leg["status"] == geometry.STATUS_MEASURED and leg["source"] == geometry.SOURCE_PROXY
    assert leg["numbers"]["cuda_graph_batches"] == _PADDED, "the configuration transfers"
    assert "kv_tokens" not in leg["numbers"], "another model's pool does not"
    assert "blocks" not in leg["page"] and "chunks" not in leg
    assert "on the stand-in checkpoint" in rf.render(tmp_path / "ws")


def test_without_a_stand_in_an_unknown_architecture_is_not_measured_and_never_a_stop(tmp_path):
    runner = _Runner(unsupported_for=("/ckpt/newmodel",))
    section = _probe(tmp_path, _task(tmp_path), runner)

    assert {leg["status"] for leg in section["legs"].values()} == {geometry.STATUS_UNSUPPORTED}
    assert "not measured" in rf.render(tmp_path / "ws")
    assert "0 of 4 leg(s) measured" in geometry.summary(section)


def test_a_probe_of_the_same_inputs_stands_and_a_changed_runtime_block_does_not(tmp_path):
    task = _task(tmp_path)
    legs = geometry.plan_legs(task, 4)
    _probe(tmp_path, task, _Runner())
    digest = geometry.inputs_digest(task, gates.MODE_OFF, legs)
    assert geometry.recorded(tmp_path / "ws", digest)

    changed = _task(tmp_path, runtime={"default": {"cuda_graph": False}})
    assert not geometry.recorded(
        tmp_path / "ws", geometry.inputs_digest(changed, gates.MODE_OFF, legs)
    )


def test_a_gates_first_pass_is_held_against_the_probe(tmp_path):
    """The probe cannot hide a difference: the first gate that measures another list says so."""
    ws = tmp_path / "ws"
    _probe(tmp_path, _task(tmp_path), _Runner())
    engine = gates.attest_engine(_engine_output(batches=[1, 2, 4]), None).record()

    changes = rf.record(ws, unit="core", member="tp1", gate="boot", engines=[engine], marker="m")

    assert changes and "differs from the geometry probe" in changes[0]
    assert "CUDA-graph batches" in changes[0]
    assert rf.geometry(ws), "a gate's record leaves the probe's section in place"


def test_the_probe_leaves_the_gates_members_in_place(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    engine = gates.attest_engine(_engine_output(), None).record()
    rf.record(ws, unit="core", member="tp1", gate="boot", engines=[engine], marker="m")
    _probe(tmp_path, _task(tmp_path), _Runner())
    assert "core@tp1" in rf.load(ws)["members"]


# ------------------------------------------------------------- beside the run


class _Journal:
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def event(self, what, detail=""):
        self.events.append((what, detail))


def _stage(tmp_path, runner, *, gate: threading.Event | None = None):
    def run(command, cwd, timeout):
        if gate is not None:
            gate.wait(5)
        return runner(command, cwd, timeout)

    return types.SimpleNamespace(
        task_data=_task(tmp_path),
        workspace=tmp_path / "ws",
        journal=_Journal(),
        _gate_env=lambda: _env(4),
        _member_env=lambda member: _env(int(member[-1])),
        _member_runner=lambda member: run,
    )


def test_a_smith_dispatch_waits_for_a_probe_still_running(tmp_path):
    release = threading.Event()
    stage = _stage(tmp_path, _Runner(), gate=release)
    probe = geometry.start_probe(stage)
    assert probe is not None and stage._geometry_probe is probe

    threading.Timer(0.2, release.set).start()
    geometry.await_probe(stage, timeout=10)

    assert probe.done and probe.section["legs"]
    assert any(e[0] == "smith dispatch waited for the geometry probe" for e in stage.journal.events)
    assert any(e[0] == "geometry probe" for e in stage.journal.events)


def test_a_resumed_run_stands_on_the_recorded_probe(tmp_path):
    stage = _stage(tmp_path, _Runner())
    geometry.start_probe(stage).wait(10)
    again = _stage(tmp_path, _Runner())

    assert geometry.start_probe(again) is None
    assert again.journal.events[-1][0] == "geometry probe not run"


def test_a_probe_after_a_failed_substrate_gate_does_not_boot(tmp_path):
    runner = _Runner()
    stage = _stage(tmp_path, runner)
    stage._await_substrate = lambda: None
    stage._substrate_failure = "the substrate gate failed"
    geometry.start_probe(stage).wait(10)
    assert not runner.commands
