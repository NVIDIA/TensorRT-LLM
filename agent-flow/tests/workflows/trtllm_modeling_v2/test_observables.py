# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each check proves what it claims: its feature's evidence is observed and judged by its oracle."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, observables, serve_gate
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import RESOLVE_HOOK_DIR
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.sessions import prepend_to_inherited
from agent_flow.workflows.trtllm_modeling_v2.spec import variants

from .test_acceptance_gate import _driver_output
from .test_engine_attestation import (
    _boot_runner,
    _engine_config,
    _engine_line,
    _env,
    _records,
    _task,
)


def _rank_line(rank: int, config: dict | None = None) -> str:
    record = {"rank": rank, "config": config or _engine_config(), "pools": {}, "errors": {}}
    return f"{workload_io.RANK_ENGINE_MARKER} {json.dumps(record)}\n"


def _variant_task(tmp_path: Path, **entries) -> dict:
    task = _task(tmp_path, variants=entries)
    return variants.variant_spec(
        task, next(iter(entries)), world_size=1, options_dir=tmp_path / "variants"
    )


# ------------------------------------------------------------- every rank


def test_every_rank_is_held_to_the_declared_dtype():
    declared = {"kv_cache_dtype": "fp8", "tp": 4}
    asked = _engine_config(kv_cache_dtype="fp8", kv_cache_quant="FP8", tp=4)
    # Read from llm.args, the request agrees with itself whatever the ranks run.
    alone = gates.attest_engine(_engine_line(asked, worker=False), declared)
    assert alone.mismatches == () and alone.sources["kv_cache_dtype"] == "llm.args"
    fell_back = dict(asked, kv_cache_quant=None)
    ranks = "".join(_rank_line(r, fell_back if r == 2 else asked) for r in range(4))
    attested = gates.attest_engine(_engine_line(asked, worker=False) + ranks, declared)
    assert attested.mismatches == (
        "kv_cache_dtype: declared fp8, engine asked for fp8 and resolved None on rank(s) 2 "
        "(read from every rank)",
    )
    assert attested.sources["kv_cache_dtype"] == gates.RANKS_SOURCE
    assert attested.ranks == (0, 1, 2, 3) and attested.world == 4


def test_the_setting_a_check_is_about_must_be_observed_on_every_rank():
    declared = {"kv_cache_dtype": "fp8", "tp": 4}
    asked = _engine_config(kv_cache_dtype="fp8", kv_cache_quant="FP8", tp=4)
    alone = gates.attest_engine(_engine_line(asked, worker=False), declared)
    [failure] = observables.unobserved(alone, ("kv_cache_dtype",))
    assert "read only from llm.args, which repeats the request" in failure
    half = gates.attest_engine(_rank_line(0, asked) + _rank_line(1, asked), declared)
    [failure] = observables.unobserved(half, ("kv_cache_dtype",))
    assert "rank(s) 2, 3 of 4 printed no record" in failure
    every = "".join(_rank_line(r, asked) for r in range(4))
    assert observables.unobserved(gates.attest_engine(every, declared), ("kv_cache_dtype",)) == []
    # A field the check is not about is recorded, never required.
    assert observables.unobserved(alone, ()) == []


def test_a_variant_is_about_what_its_overlay_sets(tmp_path):
    task = _task(
        tmp_path,
        variants={
            "kv_fp8": {
                "runtime": {"extra_flags": ["--kv_cache_dtype=fp8"]},
                "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
            },
            "chunked": {
                "runtime": {"extra_flags": ["--max_num_tokens=512"]},
                "llm_api_options": {"max_num_tokens": 512, "enable_chunked_prefill": True},
            },
            "mtp2": {
                "unit": "core",
                "llm_api_options": {"speculative_config": {"max_draft_len": 2}},
            },
            "odd": {"env": {"SOME_SWITCH": "1"}},
        },
    )
    assert observables.subject_fields(task, "kv_fp8") == ("kv_cache_dtype",)
    assert observables.subject_fields(task, "chunked") == ("max_num_tokens", "chunked_prefill")
    assert observables.subject_fields(task, "mtp2") == ("speculative",)
    assert observables.subject_fields(task, "odd") == (observables.UNKNOWN,)
    assert observables.subject_fields(task, "core") == ()
    assert observables.lossless(observables.subject_fields(task, "chunked"))
    assert not observables.lossless(observables.subject_fields(task, "kv_fp8"))
    assert not observables.lossless(observables.subject_fields(task, "odd"))


def test_a_lossless_variant_is_held_to_the_reference_greedy_tokens(tmp_path):
    logits = {"reference": str(tmp_path / "ref.json.gz")}
    task = _task(
        tmp_path,
        variants={
            "chunked": {"llm_api_options": {"max_num_tokens": 512}},
            "kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}},
        },
    )
    task["accuracy"]["logits"] = logits
    assert gates.GreedySetting.from_task(task, "chunked") is not None
    assert gates.GreedySetting.from_task(task, "kv_fp8") is None


def test_a_cache_dtype_variant_fails_its_boot_where_no_rank_reports_the_dtype(tmp_path):
    spec = _variant_task(
        tmp_path,
        kv_fp8={
            "runtime": {"extra_flags": ["--kv_cache_dtype=fp8"]},
            "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
        },
    )
    engine = _engine_config(kv_cache_dtype="fp8", kv_cache_quant="FP8")
    asked = _engine_line(engine, worker=False) + _records("Paris", "light")
    result = gates.run_boot_gate(
        _env(), task_data=spec, unit="kv_fp8", member="tp1", workdir=tmp_path,
        runner=_boot_runner(asked),
    )  # fmt: skip
    assert not result.passed
    assert result.detail.startswith("the check cannot show the setting it exists to prove")
    observed = gates.run_boot_gate(
        _env(), task_data=spec, unit="kv_fp8", member="tp1", workdir=tmp_path,
        runner=_boot_runner(asked + _rank_line(0, engine)),
    )  # fmt: skip
    assert observed.passed, observed.detail


# ------------------------------------------------------------- chunked prefill


def test_a_chunked_check_needs_a_prompt_the_engine_had_to_split():
    resumed = {"prompt_tokens": 600, "cached_tokens": 200}
    assert "longest computed 400" in observables.chunk_shortfall([resumed, None], 512)
    assert observables.chunk_shortfall([{"prompt_tokens": 1025, "cached_tokens": 0}], 512) is None
    assert observables.chunk_probe_tokens(512, None) == 1025
    assert observables.chunk_probe_tokens(512, 600) == 598
    assert observables.chunk_probe_tokens(512, 500) is None


def test_a_chunked_variant_boot_sends_the_chunk_probe_and_fails_without_chunks(tmp_path):
    spec = _variant_task(
        tmp_path,
        chunked={
            "runtime": {"extra_flags": ["--max_num_tokens=512", "--max_seq_len=9216"]},
            "llm_api_options": {"max_num_tokens": 512},
        },
    )
    engine = _engine_config(max_num_tokens=512)
    head = _engine_line(engine) + _rank_line(0, engine) + _records("Paris", "light")
    short = f"{workload_io.RECORD_MARKER} " + json.dumps(
        {"index": 2, "text": "x", "prompt_tokens": 300, "cached_tokens": 0}
    )
    runner = _boot_runner(head + short + "\n")
    result = gates.run_boot_gate(
        _env(), task_data=spec, unit="chunked", member="tp1", workdir=tmp_path, runner=runner
    )
    assert not result.passed and "never split a context into chunks" in result.detail
    written = json.loads((tmp_path / "boot.chunked.tp1.json").read_text())
    assert written["requests"][-1] == {"id": "chunk", "probe_tokens": 1025, "max_tokens": 1}
    split = f"{workload_io.RECORD_MARKER} " + json.dumps(
        {"index": 2, "text": "x", "prompt_tokens": 1025, "cached_tokens": 0}
    )
    result = gates.run_boot_gate(
        _env(), task_data=spec, unit="chunked", member="tp1", workdir=tmp_path,
        runner=_boot_runner(head + split + "\n"),
    )  # fmt: skip
    assert result.passed, result.detail
    assert "served in chunks" in result.detail


# ------------------------------------------------------------- speculative decoding


def _depth_runner(deep, shallow, depth):
    def run(command, env=None, timeout=0):
        if "TRTLLM_MODELING_V2=off" in command:
            return CommandResult(returncode=1, stderr="Unknown model architecture: X")
        if f".depth{depth - 1}.json" in command:
            return CommandResult(returncode=0, stdout=_driver_output(shallow))
        return CommandResult(returncode=0, stdout=_driver_output(deep))

    return run


def _speculator_task(tmp_path: Path, depth: int) -> dict:
    options = tmp_path / "mtp.llm_api.yaml"
    options.write_text(yaml.safe_dump({"speculative_config": {"max_draft_len": depth}}))
    return {
        "checkpoint_path": "/ckpt/M",
        "parallel": "tep4",
        "trtllm_repo_path": "/repo",
        "features": ["mtp"],
        "accuracy": {"benchmark": "gsm8k", "llm_api_options": str(options)},
    }


def test_each_draft_depth_must_beat_the_one_below_it_without_a_stock_speculator(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        gates, "inspect_tree", lambda task_data, allow_dirty=None: gates.TreeState("t")
    )
    task = _speculator_task(tmp_path, depth=3)
    shallow = [2.2 + 0.3 * math.sin(i) for i in range(32)]
    flat = gates.run_acceptance_gate(
        _env(), task_data=task, workdir=tmp_path, runner=_depth_runner(shallow, shallow, 3)
    )
    assert not flat.passed
    assert "depth 3 accepts no more than depth 2" in flat.detail
    deeper = [r + 0.3 + 0.05 * math.cos(i) for i, r in enumerate(shallow)]
    rising = gates.run_acceptance_gate(
        _env(), task_data=task, workdir=tmp_path, runner=_depth_runner(deeper, shallow, 3)
    )
    assert rising.passed, rising.detail
    assert "depth 3's last draft position is accepted" in rising.detail


# ------------------------------------------------------------- serve


def _answers(*texts: str) -> str:
    lines = [
        f"{workload_io.RECORD_MARKER} "
        + json.dumps({"index": n, "status": 200, "text": t, "finish_reason": "length"})
        for n, t in enumerate(texts)
    ]
    server = {"healthy": True, "boot_s": 9.0}
    return "\n".join(lines) + f"\n{serve_gate.serve_driver.SERVER_MARKER} {json.dumps(server)}\n"


def test_serve_answers_are_held_to_the_boot_workload(tmp_path):
    task = {"checkpoint_path": "/ckpt", "trtllm_repo_path": str(tmp_path)}
    image = gates.BootPrompt(
        prompt="What is shown?", keywords_any=("lion", "statue"), images=("a.png",)
    )
    boot = gates.BootSpec(
        prompts=("The capital of France is",), keywords=("paris",), items=(image,)
    )
    requests = serve_gate.serve_requests(task, serve_gate.ServeSpec(), boot)
    assert [r["id"] for r in requests] == ["text", "boot0", "image"]
    assert requests[1]["boot"] == ["prompt", 0] and requests[2]["boot"] == ["item", 0]
    passed, detail, _ = serve_gate.judge(_answers("x", " Paris.", "A lion."), requests, 0, boot)
    assert passed, detail
    passed, detail, _ = serve_gate.judge(_answers("x", "zzz zzz", "qq qq"), requests, 0, boot)
    assert not passed
    assert "missing 2 keyword(s)" in detail and "'paris'" in detail and "lion" in detail
    # A workload with nothing to judge adds nothing to ask.
    bare = serve_gate.serve_requests(task, serve_gate.ServeSpec(), gates.BootSpec())
    assert [r["id"] for r in bare] == ["text"]


# ------------------------------------------------------------- the rank hook


_FAKE_WORKER = """
import types


class BaseWorker:
    rank = 3
    llm_args = types.SimpleNamespace(
        kv_cache_config=types.SimpleNamespace(dtype="fp8", enable_block_reuse=True)
    )

    def setup_engine(self):
        print("engine set up", flush=True)
"""


def test_the_rank_hook_has_every_worker_print_its_engine(tmp_path):
    site = tmp_path / "site"
    (site / "tensorrt_llm" / "executor").mkdir(parents=True)
    (site / "tensorrt_llm" / "__init__.py").write_text("")
    (site / "tensorrt_llm" / "executor" / "__init__.py").write_text("")
    (site / "tensorrt_llm" / "executor" / "base_worker.py").write_text(_FAKE_WORKER)
    later = tmp_path / "later"
    later.mkdir()
    (later / "sitecustomize.py").write_text("import sys\nsys.stderr.write('chained\\n')\n")
    script = "import tensorrt_llm.executor.base_worker as b\nb.BaseWorker().setup_engine()\n"
    # Both hooks of this kind on the path: each hands on forward, and the last one runs once.
    path = os.pathsep.join([str(gates.RANK_HOOK_DIR), str(RESOLVE_HOOK_DIR), str(site), str(later)])
    base = {"PATH": "/usr/bin:/bin", "PYTHONPATH": path}
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        env={**base, gates.RANK_FACTS_ENV: "1"}, cwd=tmp_path, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stderr.count("chained") == 1
    assert done.stdout.index("engine set up") < done.stdout.index(gates.RANK_ENGINE_MARKER)
    [record] = gates.parse_rank_records(done.stdout).values()
    assert record["rank"] == 3 and record["config"]["kv_cache_dtype"] == "fp8"
    assert record["config"]["block_reuse"] is True
    off = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        env=base, cwd=tmp_path, check=False,
    )  # fmt: skip
    assert off.returncode == 0 and gates.RANK_ENGINE_MARKER not in off.stdout
    assert off.stderr.count("chained") == 1


def test_the_hook_goes_first_on_pythonpath_and_leaves_no_empty_entry():
    env = gates._with_mode(_env(PYTHONPATH="/declared"), gates.MODE_REQUIRE)
    assert env.env["PYTHONPATH"] == os.pathsep.join([str(gates.RANK_HOOK_DIR), "/declared"])
    assert env.env[gates.RANK_FACTS_ENV] == "1"
    assigned = prepend_to_inherited("PYTHONPATH", "/hook")
    assert assigned == '/hook"${PYTHONPATH:+:$PYTHONPATH}"'
    script = textwrap.dedent(
        f"""
        unset PYTHONPATH; echo "[{assigned}]"
        PYTHONPATH=/inherited; echo "[{assigned}]"
        """
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["[/hook]", "[/hook:/inherited]"]
