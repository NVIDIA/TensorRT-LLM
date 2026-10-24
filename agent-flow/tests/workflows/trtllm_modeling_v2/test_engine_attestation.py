# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every gate attests the engine it ran, and records its pools and free memory.

A gate states a configuration and the engine can run another: the worker
turned block reuse off in every quickstart gate of a hybrid model while each
gate passed, the CUDA-graph lists of two launch paths differed, a boot beside
memory a cancelled job held sized half its pool, and MTP depth 7 at tp1 ran
with 0.5 GiB free until it ran out. These tests hold the harness to reading
the configuration back from the engine once it is up, the gates to comparing
it with what they declared and failing on a difference, and the verdict to
carrying the pools, the free memory and a LOW MARGIN warning. Engine output
is canned or comes from a fake ``tensorrt_llm`` package; no GPU is used.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import envelope_boot, gates, witness
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import engine_facts, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    RUNTIME_MARGIN_DEFAULTS,
    TaskSchemaError,
    _validate_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

#: A hybrid target's snapshot policy and runtime block, as a run's `task.yaml` states them.
_SNAPSHOTS = {
    "periodic_snapshot_interval": 2048,
    "additional_snapshot_offsets_from_start": [256, 512, 768],
}
_OPTIONS = {
    "kv_cache_config": {
        "free_gpu_memory_fraction": 0.6,
        "enable_block_reuse": True,
        "mamba_state_config": _SNAPSHOTS,
        "use_kv_cache_manager_v2": True,
    },
    "enable_chunked_prefill": True,
    "max_num_tokens": 8192,
    "cuda_graph_config": {"enable_padding": True, "max_batch_size": 128},
}
_RUNTIME = {
    "cuda_graph": True,
    "chunked_prefill": True,
    "overlap_scheduler": True,
    "kv_cache_reuse": True,
    "extra_flags": [
        "--max_seq_len=9216",
        "--max_batch_size=128",
        "--max_num_tokens=8192",
        "--kv_cache_fraction=0.6",
        "--mamba_ssm_cache_dtype=auto",
    ],
}
#: The 19 padded batches `CudaGraphConfig(enable_padding=True, max_batch_size=128)` captures.
_PADDED = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]

#: Four ranks' bind lines, as a tp4 boot logs them.
_BINDS = "".join(
    f"modeling_v2 newmodel: bound MambaHybridCacheManagerV2 #{n}: {slots} recurrent slots, "
    f"32 states per slot; SSM torch.float32 ({slots}, 4, 32, 64)\n"
    for n, slots in ((1, 96),) * 4 + ((2, 4096),) * 4
)
_REUSE_WARNING = (
    "[TRT-LLM] [W] [_torch] Disabling KV cache block reuse for the hybrid Mamba model "
    "because no Mamba state snapshot policy is configured."
)


def _task(tmp_path: Path, **extra) -> dict:
    options = tmp_path / "core.llm_api.yaml"
    options.write_text(yaml.safe_dump(_OPTIONS), encoding="utf-8")
    task = {
        "checkpoint_path": "/ckpt/model",
        "parallel": "unified",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": [],
        "runtime": {"default": dict(_RUNTIME)},
        "accuracy": {"benchmark": "gsm8k", "llm_api_options": {"core": str(options)}},
    }
    task.update(extra)
    return task


def _env(world_size: int = 1, **env) -> RunEnvironment:
    return RunEnvironment(kind="local", python="/py", world_size=world_size, env=dict(env))


def _words(task: dict, member: str = "tp1", unit: str = "core") -> list[str]:
    return gates.quickstart_words(
        task, _env(), unit=unit, features=None, max_tokens=32, member=member
    )


def _engine_config(**changes) -> dict:
    """What a correctly built core@tp1 engine reports."""
    config = {
        "modeling_v2": "require",
        "block_reuse": True,
        "mamba_snapshot": {
            **_SNAPSHOTS,
            "additional_snapshot_offsets_from_end": [],
        },
        "kv_cache_dtype": "auto",
        "kv_cache_quant": None,
        "mamba_ssm_cache_dtype": "auto",
        "mamba_ssm_cache_dtype_resolved": "float32",
        "cuda_graph": {"batch_sizes": list(_PADDED), "enable_padding": True},
        "cuda_graph_captured": list(_PADDED),
        "speculative": None,
        "max_num_tokens": 8192,
        "max_batch_size": 128,
        "max_seq_len": 9216,
        "chunked_prefill": True,
        "overlap_scheduler": True,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "attention_dp": False,
    }
    config.update(changes)
    return config


def _engine_line(
    config: dict | None = None,
    *,
    worker: bool = True,
    free_mib: float = 40960.0,
    total_mib: float = 81920.0,
    devices: int = 1,
    argv: list[str] | None = None,
    kv_max_tokens: int | None = 262144,
) -> str:
    config = _engine_config() if config is None else config
    record = {
        "argv": argv or [],
        "worker": worker,
        "observer": True,
        "config": config,
        "sources": {k: "worker" if worker else "llm.args" for k in config},
        "pools": {"kv_max_tokens": kv_max_tokens} if worker else {},
        "memory": {
            "source": "nvml",
            "devices": [
                {"device": str(d), "free_mib": free_mib, "total_mib": total_mib}
                for d in range(devices)
            ],
        },
        "errors": {},
    }
    return f"{workload_io.ENGINE_MARKER} {json.dumps(record)}\n"


def _records(*texts: str) -> str:
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


# --------------------------------------------------------- what is declared


def test_the_declared_engine_is_what_the_resolvers_put_in_the_argv(tmp_path):
    task = _task(tmp_path)
    declared = gates.engine_declaration(task, "core", _words(task), mode="require")
    assert declared == {
        "modeling_v2": "require",
        "block_reuse": True,
        "mamba_snapshot": {**_SNAPSHOTS, "additional_snapshot_offsets_from_end": []},
        "mamba_ssm_cache_dtype": "auto",
        "cuda_graph": {"batch_sizes": _PADDED, "enable_padding": True},
        "speculative": None,
        "max_num_tokens": 8192,
        "max_batch_size": 128,
        "chunked_prefill": True,
        "tp": 1,
        "ep": 1,
        "attention_dp": False,
    }
    # No flag and no options key: the KV dtype is the engine's to report, not judged.
    assert "kv_cache_dtype" not in declared


def test_the_member_and_the_speculator_are_declared(tmp_path):
    task = _task(tmp_path, features=["mtp"])
    options = {**_OPTIONS, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    declared = gates.engine_declaration(task, "core", _words(task, "dep4"), mode="off")
    assert declared["speculative"] == {"algorithm": "MTP", "max_draft_len": 3}
    assert (declared["tp"], declared["ep"], declared["attention_dp"]) == (4, 4, True)
    assert declared["modeling_v2"] == "off"
    tep = gates.engine_declaration(task, "core", _words(task, "tep4"), mode=None)
    assert (tep["tp"], tep["ep"], tep["attention_dp"]) == (4, 4, False)
    assert "modeling_v2" not in tep
    assert gates.engine_declaration(task, "core", _words(task, "tp4"), mode=None)["ep"] == 1


def test_a_dtype_the_argv_does_not_state_is_the_options_files():
    """The quickstart path does not carry it, so an engine without it differs from the unit."""
    options = {"kv_cache_config": {"dtype": "fp8", "mamba_ssm_cache_dtype": "float32"}}
    declared = gates.declared_engine(["--tp_size=1"], options=options)
    assert declared["kv_cache_dtype"] == "fp8" and declared["mamba_ssm_cache_dtype"] == "float32"
    flagged = gates.declared_engine(
        ["--kv_cache_dtype", "nvfp4", "--mamba_ssm_cache_dtype=auto"], options=options
    )
    assert flagged["kv_cache_dtype"] == "nvfp4" and flagged["mamba_ssm_cache_dtype"] == "auto"


def test_the_declared_snapshot_policy_reads_the_keys_as_the_engine_does():
    assert gates.declared_snapshot_policy({}) == {
        "periodic_snapshot_interval": 0,
        "additional_snapshot_offsets_from_start": [],
        "additional_snapshot_offsets_from_end": [],
    }
    legacy = gates.declared_snapshot_policy({"mamba_state_cache_interval": 512})
    assert legacy["periodic_snapshot_interval"] == 512
    conversation = gates.declared_snapshot_policy(
        {"mamba_state_config": _SNAPSHOTS, "block_reuse_config": {"policy": "per_conversation"}}
    )
    assert conversation["periodic_snapshot_interval"] == 0
    assert conversation["additional_snapshot_offsets_from_start"] == [256, 512, 768]


def test_reuse_turned_off_by_the_runtime_block_is_declared_off(tmp_path):
    task = _task(tmp_path)
    task["runtime"]["core"] = {"kv_cache_reuse": False}
    declared = gates.engine_declaration(task, "core", _words(task), mode="require")
    assert declared["block_reuse"] is False


# ------------------------------------------------------------ the comparison


def _declared(tmp_path, **changes) -> dict:
    task = _task(tmp_path)
    return {**gates.engine_declaration(task, "core", _words(task), mode="require"), **changes}


def test_an_engine_that_ran_the_declaration_attests_every_field(tmp_path):
    attested = gates.attest_engine(_engine_line() + _BINDS, _declared(tmp_path))
    assert attested.reported and not attested.mismatches and not attested.not_attested
    assert attested.recurrent_slots == (4096,) * 4
    assert attested.kv_max_tokens == 262144
    assert attested.devices == ({"device": "0", "free_gib": 40.0, "total_gib": 80.0},)
    record = attested.record()
    assert json.loads(json.dumps(record)) == record, "the record is JSON as it stands"
    assert record["pools"]["recurrent_slots"] == [4096] * 4


@pytest.mark.parametrize(
    "changes,named",
    [
        ({"block_reuse": False}, "block_reuse: declared on, engine off (read from worker)"),
        (
            {
                "cuda_graph": {
                    "batch_sizes": list(range(1, 32)) + [32, 64, 128],
                    "enable_padding": False,
                }
            },
            "cuda_graph: declared 19 batches [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, "
            "88, 96, 104, 112, 120, 128], padded, engine 34 batches [1-32, 64, 128], unpadded",
        ),
        ({"max_num_tokens": 4096}, "max_num_tokens: declared 8192, engine 4096"),
        ({"modeling_v2": "off"}, "modeling_v2: declared require, engine off"),
        ({"ep": 4}, "ep: declared 1, engine 4"),
        (
            {
                "mamba_snapshot": {
                    **_SNAPSHOTS,
                    "periodic_snapshot_interval": 0,
                    "additional_snapshot_offsets_from_end": [],
                }
            },
            "mamba_snapshot: declared interval 2048",
        ),
        (
            {"mamba_ssm_cache_dtype": "bfloat16"},
            "mamba_ssm_cache_dtype: declared auto, engine bfloat16",
        ),
    ],
)
def test_a_field_the_engine_ran_differently_is_named(tmp_path, changes, named):
    attested = gates.attest_engine(_engine_line(_engine_config(**changes)), _declared(tmp_path))
    assert len(attested.mismatches) == 1 and named in attested.mismatches[0], attested.mismatches
    assert "did not run the declared configuration" in attested.difference()


def test_a_cache_dtype_is_held_to_what_the_worker_resolved(tmp_path):
    declared = _declared(tmp_path, kv_cache_dtype="nvfp4", mamba_ssm_cache_dtype="float32")
    fell_back = _engine_config(
        kv_cache_dtype="nvfp4",
        kv_cache_quant="FP8",
        mamba_ssm_cache_dtype="float32",
        mamba_ssm_cache_dtype_resolved="bfloat16",
    )
    mismatches = gates.attest_engine(_engine_line(fell_back), declared).mismatches
    assert (
        "kv_cache_dtype: declared nvfp4, engine asked for nvfp4 and resolved FP8" in mismatches[0]
    )
    assert (
        "mamba_ssm_cache_dtype: declared float32, engine asked for float32 and resolved"
        in (mismatches[1])
    )
    native = _engine_config(
        kv_cache_dtype="nvfp4", kv_cache_quant="NVFP4", mamba_ssm_cache_dtype="fp32",
        mamba_ssm_cache_dtype_resolved="float32",
    )  # fmt: skip
    assert not gates.attest_engine(_engine_line(native), declared).mismatches


def test_the_speculator_compares_by_algorithm_and_depth(tmp_path):
    declared = _declared(tmp_path, speculative={"algorithm": "DSPARK", "max_draft_len": 5})
    same = _engine_config(speculative={"algorithm": "DSpark", "max_draft_len": 5})
    assert not gates.attest_engine(_engine_line(same), declared).mismatches
    shallow = _engine_config(speculative={"algorithm": "DSpark", "max_draft_len": 1})
    [line] = gates.attest_engine(_engine_line(shallow), declared).mismatches
    assert "declared DSPARK at draft length 5, engine DSpark at draft length 1" in line


def test_a_field_the_record_lacks_is_not_attested_never_agreed(tmp_path):
    config = _engine_config()
    del config["block_reuse"]
    attested = gates.attest_engine(_engine_line(config, worker=False), _declared(tmp_path))
    assert attested.not_attested == ("block_reuse",) and not attested.mismatches
    assert "another process" in attested.note
    # The log is the one place every rank says it turned reuse off.
    logged = gates.attest_engine(
        _engine_line(config, worker=False) + _REUSE_WARNING, _declared(tmp_path)
    )
    assert logged.sources["block_reuse"] == gates.LOG_SOURCE
    assert "block_reuse: declared on, engine off (read from engine log)" in logged.mismatches[0]


def test_without_a_record_every_declared_field_is_not_attested(tmp_path):
    attested = gates.attest_engine(_records("x") + _BINDS, _declared(tmp_path))
    assert not attested.reported and not attested.mismatches
    assert set(attested.not_attested) == set(_declared(tmp_path))
    assert attested.recurrent_slots == (4096,) * 4
    broken = f'{workload_io.ENGINE_MARKER} {{"error": "KeyError: x"}}\n'
    assert "could not read the engine: KeyError: x" in gates.attest_engine(broken, {}).note


def test_pools_are_read_per_rank_from_the_log():
    assert gates.parse_recurrent_slots(_BINDS) == [4096] * 4
    assert gates.parse_recurrent_slots("no binds\n") == []
    # One rank sized its pool beside memory a cancelled job held.
    uneven = _BINDS.replace("#2: 4096", "#2: 642", 1)
    assert gates.parse_recurrent_slots(uneven) == [642, 4096, 4096, 4096]
    kv = (
        "[TensorRT-LLM][INFO] [MemUsageChange] Allocated 29.11 GiB for max tokens in paged KV "
        "cache (524288).\n"
    ) * 2
    assert gates.parse_kv_allocations(kv) == [524288, 524288]


# ------------------------------------------------------------------ margin


def test_the_margin_is_the_larger_of_the_two_floors():
    margin = gates.MemoryMargin()
    assert (margin.min_free_gib, margin.min_free_fraction) == (2.0, 0.02)
    assert margin.threshold_gib(80.0) == 2.0
    assert margin.threshold_gib(276.62) == pytest.approx(5.5324)


def test_low_free_memory_is_a_warning_on_a_passing_verdict(tmp_path):
    records = [
        gates.attest_engine(
            _engine_line(free_mib=1024.0 * 23.97, total_mib=1024.0 * 276.62), {}
        ).record(),
        gates.attest_engine(
            _engine_line(free_mib=1024.0 * 0.515, total_mib=1024.0 * 276.62), {}, leg="off"
        ).record(),
    ]
    note = gates.MemoryMargin().note(records)
    assert note.startswith("LOW MARGIN: 0.52 GiB free on device 0 in the off leg once the engine")
    assert "under the 5.53 GiB threshold" in note
    assert gates.MemoryMargin().note(records[:1]) is None
    assert gates.MemoryMargin(min_free_gib=0, min_free_fraction=0).note(records) is None


def test_the_margin_is_set_per_unit_in_the_runtime_block(tmp_path):
    task = _task(tmp_path)
    assert gates.resolve_memory_margin(task, "core") == gates.MemoryMargin()
    task["runtime"]["default"]["min_free_memory_gib"] = 4
    task["runtime"]["mtp"] = {"min_free_memory_fraction": 0.1}
    assert gates.resolve_memory_margin(task, "core") == gates.MemoryMargin(4.0, 0.02)
    assert gates.resolve_memory_margin(task, "mtp") == gates.MemoryMargin(4.0, 0.1)


def test_the_schema_validates_the_margin_keys_and_never_fills_them():
    block = {"cuda_graph": True, "min_free_memory_gib": 3, "min_free_memory_fraction": 0.05}
    errors: list[str] = []
    _validate_runtime(block, errors)
    assert not errors and block["min_free_memory_gib"] == 3
    plain = {"cuda_graph": True}
    _validate_runtime(plain, errors)
    assert not errors and not set(RUNTIME_MARGIN_DEFAULTS) & set(plain)
    for bad in ({"min_free_memory_gib": -1}, {"min_free_memory_gib": "2"},
                {"min_free_memory_fraction": 1.5}, {"min_free_memory_gib": True}):  # fmt: skip
        errors = []
        _validate_runtime({"default": {**bad}, "core": {}}, errors)
        assert errors and next(iter(bad)) in errors[0], bad
    # Scalars, so a flat block is not mistaken for the per-unit form.
    flat = {"kv_cache_reuse": True, "min_free_memory_fraction": 0.05}
    assert gates.resolve_runtime({"runtime": flat}, "core") is flat
    with pytest.raises(TaskSchemaError):
        _validate_runtime(["not", "a", "mapping"], [])


# --------------------------------------------------------------- the verdict


def test_the_summary_carries_the_engine_and_an_old_verdict_reads_as_before(tmp_path):
    old = gates.GateResult(name="boot", passed=True, detail="ok", duration_s=3)
    assert old.engine == () and "[engine" not in old.summary()
    record = gates.attest_engine(_engine_line() + _BINDS, _declared(tmp_path)).record()
    result = gates.GateResult(name="boot", passed=True, detail="ok", member="tp4", engine=(record,))
    assert hash(result) == hash(replace(result, engine=())), "a verdict still hashes"
    assert (
        "[engine 12/12 fields as declared; recurrent slots 4096 x4; KV 262144 tokens; "
        "min free 40.00 of 80.00 GiB]" in result.summary()
    )
    bare = gates.attest_engine(_BINDS, None, absent=gates.EVAL_NOT_ATTESTED).record()
    assert gates.describe_engine([bare]) == (
        " [engine configuration not attested; recurrent slots 4096 x4]"
    )
    missing = gates.attest_engine("", _declared(tmp_path), leg="off").record()
    assert gates.describe_engine([record, missing]).startswith(" [engine 12/12 fields as declared")
    assert "| off: NOT ATTESTED (the run printed no MV2FLOW_ENGINE record)" in (
        gates.describe_engine([record, missing])
    )


def _boot_runner(*outputs: str):
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=outputs[min(len(seen), len(outputs)) - 1])

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_a_boot_gate_fails_on_an_engine_that_ran_another_configuration(tmp_path):
    task = _task(tmp_path)
    good = gates.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_boot_runner(_engine_line() + _BINDS[:0] + _records("Paris", "light")),
    )  # fmt: skip
    assert good.passed, good.detail
    [record] = good.engine
    assert record["reported"] and not record["mismatches"] and not good.warnings
    assert "12/12 fields as declared" in good.summary()

    # Block reuse off with no warning in the log: only the attestation sees it.
    silent = _engine_line(_engine_config(block_reuse=False)) + _records("Paris", "light")
    bad = gates.run_boot_gate(
        _env(), task_data=task, unit="core", member="tp1", workdir=tmp_path,
        runner=_boot_runner(silent),
    )  # fmt: skip
    assert not bad.passed
    assert bad.detail.startswith("the engine did not run the declared configuration -- ")
    assert "block_reuse: declared on, engine off (read from worker)" in bad.detail
    assert bad.engine[0]["mismatches"]


def test_a_boot_gate_passes_with_little_memory_left_and_says_so(tmp_path):
    tight = _engine_line(free_mib=527.0, total_mib=283258.0) + _records("Paris", "light")
    result = gates.run_boot_gate(
        _env(), task_data=_task(tmp_path), unit="core", member="tp1", workdir=tmp_path,
        runner=_boot_runner(tight),
    )  # fmt: skip
    assert result.passed
    assert result.warnings and result.warnings[0].startswith("LOW MARGIN: 0.51 GiB free")
    assert "[WARNING: LOW MARGIN" in result.summary()


def test_a_boot_gate_without_a_record_passes_unattested(tmp_path):
    """No record is reported as such; it is never read as agreement."""
    result = gates.run_boot_gate(
        _env(), task_data=_task(tmp_path), unit="core", member="tp1", workdir=tmp_path,
        runner=_boot_runner(_records("Paris", "light")),
    )  # fmt: skip
    assert result.passed and not result.engine[0]["reported"]
    assert "NOT ATTESTED (the run printed no MV2FLOW_ENGINE record)" in result.summary()


def test_the_accuracy_gate_records_pools_and_says_when_its_engine_is_not_attested(tmp_path):
    """Its engine runs under the eval shim; a run that printed no record says so."""
    task = _task(tmp_path)
    task["accuracy"].update({"floor": {"rule": "tolerance"}, "tolerance": 5.0})
    score = "lm-eval gsm8k average accuracy: 90.00\n"
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(0, stdout=_BINDS + score)

    result = gates.run_accuracy_gate(
        _env(), task_data=task, reference=90.0, unit="core", runner=run
    )
    assert result.passed
    assert str(gates.EVAL_SHIM) in seen[0] and " trtllm-eval --model=" in seen[0]
    assert result.engine[0]["note"] == gates.EVAL_NOT_ATTESTED
    assert result.engine[0]["declared"] == {"block_reuse": True}
    assert (
        "[engine NOT ATTESTED (the evaluation printed no engine record); recurrent slots 4096 x4]"
        in result.summary()
    )


def test_the_acceptance_gate_holds_each_leg_to_its_own_mode(tmp_path):
    task = _task(tmp_path, features=["mtp"])
    options = {**_OPTIONS, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    mtp = {"algorithm": "MTP", "max_draft_len": 3}
    legs = [
        _engine_line(_engine_config(speculative=mtp)),
        _engine_line(_engine_config(speculative=mtp, modeling_v2="require")),
    ]
    run = _boot_runner(*legs)
    result = gates.run_acceptance_gate(
        _env(), task_data=task, runner=run, unit="core", member="tp1", workdir=tmp_path
    )
    assert len(run.seen) == 2 and not result.passed
    assert result.detail.startswith("the engine of the off leg did not run the declared")
    assert "modeling_v2: declared off, engine require" in result.detail
    assert [r["leg"] for r in result.engine] == ["require", "off"]


def test_the_envelope_boot_attests_both_legs(tmp_path):
    task = _task(tmp_path)
    calls: list[int] = []

    def run(command, env=None, timeout=0):
        spec = json.loads(Path(re.search(r"--spec (\S+)", command).group(1)).read_text())
        calls.append(1)
        lines = [
            json.dumps({"index": n, "id": r["id"], "token_ids": [7, 8, 9, 10],
                        "prompt_tokens": len(r.get("token_ids") or ()) or 5})
            for n, r in enumerate(spec["requests"])
        ]  # fmt: skip
        config = _engine_config(cuda_graph=None) if len(calls) == 2 else _engine_config()
        body = "".join(f"{workload_io.RECORD_MARKER} {line}\n" for line in lines)
        return CommandResult(0, stdout=_engine_line(config) + body)

    result = envelope_boot.run_envelope_boot(
        _env(), task_data=task, runner=run, unit="core", member="tp1", workdir=tmp_path
    )
    assert not result.passed and len(calls) == 2
    assert result.detail.startswith("the engine of the envelope leg did not run")
    assert "cuda_graph: declared 19 batches" in result.detail and "engine no CUDA graphs" in (
        result.detail
    )
    assert [r["leg"] for r in result.engine] == ["reference", "envelope"]


def _witness_task(tmp_path: Path) -> dict:
    task = _task(tmp_path, features=["disagg"])
    task["accuracy"]["llm_api_options"] = {"disagg": task["accuracy"]["llm_api_options"]["core"]}
    task["runtime"]["disagg"] = {}
    task["accuracy"]["witness"] = {
        "disagg": {
            "contrast_env": {"MV2FLOW_DISAGG_LEG": "B"},
            "contrast_command": "/run/disagg_leg_b.py",
            "min_generated_tokens": 1,
        }
    }
    return task


def test_a_witness_attests_its_shim_legs_and_not_the_launcher(tmp_path):
    task = _witness_task(tmp_path)
    continuations = "[0] Prompt: 'p'..., Generated text: ' A'...\n[0] Witness token ids: [1, 2]\n"
    argv = gates.quickstart_words(
        task, _env(), unit="disagg", features=None, max_tokens=256, member="tp1"
    )
    argv = [w for w in argv if not w.startswith("--max_tokens")]

    def answer(config: dict):
        seen: list[str] = []

        def run(command, env=None, timeout=0):
            seen.append(command)
            if f"{witness.LEG_B_ABLATE_ENV}={witness.ABLATE_STATE}" in command:
                # Leg B's self-test: the state it received destroyed, the output moves.
                moved = continuations.replace("[1, 2]", "[3]")
                return CommandResult(0, stdout=moved + "Witness ablated: conv, ssm\n")
            head = _engine_line(config, argv=argv) if len(seen) == 1 else ""
            return CommandResult(0, stdout=head + continuations)

        run.seen = seen  # type: ignore[attr-defined]
        return run

    def witness_with(config: dict):
        run = answer(config)
        result = witness.run_declared_witness(
            _env(**{gates.MODELING_V2_ENV: "require"}),
            unit="disagg", member="tp1", task_data=task, runner=run, workdir=tmp_path / "w",
        )  # fmt: skip
        return run, result

    run, result = witness_with(_engine_config())
    assert result.passed, result.detail
    assert len(run.seen) == 3 and str(witness.QUICKSTART_SHIM) in run.seen[0]
    assert [r["leg"] for r in result.engine] == ["aggregated"], "the launcher leg is not attested"
    assert not result.engine[0]["mismatches"] and not result.engine[0]["not_attested"]

    _, wrong = witness_with(_engine_config(chunked_prefill=False))
    assert not wrong.passed
    assert "the engine of the aggregated leg did not run the declared configuration" in (
        wrong.detail
    )
    assert "chunked_prefill: declared on, engine off" in wrong.detail


def test_a_witness_names_its_legs_and_warns_on_the_tightest(tmp_path):
    task = _task(tmp_path, features=["host_offload"])
    task["accuracy"]["llm_api_options"] = {
        "host_offload": task["accuracy"]["llm_api_options"]["core"]
    }
    task["accuracy"]["witness"] = {
        "host_offload": {
            "contrast_env": {"TARGET_OFFLOAD": "0"},
            "expected_mib": 100,
            "min_generated_tokens": 1,
        }
    }
    argv = gates.quickstart_words(
        task, _env(), unit="host_offload", features=None, max_tokens=64, member="tp1"
    )
    legs = iter(((1000, 40960.0), (1200, 512.0)))

    def run(command, env=None, timeout=0):
        resident_mib, free_mib = next(legs)
        return CommandResult(
            0,
            stdout=_engine_line(argv=argv, free_mib=free_mib)
            + f"{witness.DEVICE_MEMORY_MARKER} {resident_mib}\n"
            + "[0] Prompt: 'p'..., Generated text: ' A'...\n[0] Witness token ids: [1, 2]\n",
        )

    result = witness.run_declared_witness(
        _env(**{gates.MODELING_V2_ENV: "require"}),
        unit="host_offload", member="tp1", task_data=task, runner=run, workdir=tmp_path / "w",
    )  # fmt: skip
    assert result.passed, result.detail
    assert [r["leg"] for r in result.engine] == ["offloaded", "resident"]
    assert all(not r["mismatches"] and not r["not_attested"] for r in result.engine)
    [note] = result.warnings
    assert note.startswith("LOW MARGIN: 0.50 GiB free on device 0 in the resident leg")


# ------------------------------------------------------------ the harness


def _args(**changes) -> types.SimpleNamespace:
    """A ``TorchLlmArgs`` as the reader sees it."""
    state = types.SimpleNamespace(
        periodic_snapshot_interval=2048,
        additional_snapshot_offsets_from_start=[256, 512, 768],
        additional_snapshot_offsets_from_end=[],
    )
    kv = types.SimpleNamespace(
        enable_block_reuse=True,
        mamba_state_config=state,
        dtype="auto",
        mamba_ssm_cache_dtype="auto",
        max_gpu_total_bytes=12 << 30,
    )
    parallel = types.SimpleNamespace(
        tp_size=4, pp_size=1, moe_ep_size=-1, moe_tp_size=-1, enable_attention_dp=False
    )
    fields = dict(
        modeling_v2="require",
        kv_cache_config=kv,
        cuda_graph_config=types.SimpleNamespace(batch_sizes=[8, 1, 2, 4], enable_padding=True),
        speculative_config=types.SimpleNamespace(decoding_type="MTP", max_draft_len=3),
        max_num_tokens=8192,
        max_batch_size=128,
        max_seq_len=9216,
        enable_chunked_prefill=True,
        disable_overlap_scheduler=False,
        parallel_config=parallel,
    )
    fields.update(changes)
    return types.SimpleNamespace(**fields)


def test_the_harness_reads_every_attested_field_from_the_args():
    config, errors = engine_facts.read_args(_args())
    assert not errors
    assert set(gates.ATTESTED_FIELDS) <= set(config)
    assert config["cuda_graph"] == {"batch_sizes": [1, 2, 4, 8], "enable_padding": True}
    assert config["speculative"] == {"algorithm": "MTP", "max_draft_len": 3}
    assert config["mamba_snapshot"] == {**_SNAPSHOTS, "additional_snapshot_offsets_from_end": []}
    assert (config["tp"], config["ep"], config["attention_dp"]) == (4, 1, False)
    stock = _args(cuda_graph_config=None, speculative_config=None)
    del stock.modeling_v2
    config, errors = engine_facts.read_args(stock)
    assert config["cuda_graph"] is None and config["speculative"] is None
    assert "modeling_v2" not in config and "AttributeError" in errors["modeling_v2"]


def test_the_expert_width_is_the_mappings():
    class Parallel(types.SimpleNamespace):
        def to_mapping(self):
            return types.SimpleNamespace(moe_ep_size=self.moe_ep_size * 2)

    assert engine_facts._expert_parallel(Parallel(moe_ep_size=2, moe_tp_size=-1)) == 4
    flat = types.SimpleNamespace(moe_ep_size=4, moe_tp_size=-1)
    assert engine_facts._expert_parallel(flat) == 4
    with pytest.raises(ValueError):
        engine_facts._expert_parallel(types.SimpleNamespace(moe_ep_size=-1, moe_tp_size=2))


def _worker(args: types.SimpleNamespace, *, reuse: bool = True) -> types.SimpleNamespace:
    stats = types.SimpleNamespace(max_num_blocks=1000, tokens_per_block=32)
    kv = types.SimpleNamespace(
        enable_block_reuse=reuse, tokens_per_block=32, get_kv_cache_stats=lambda: stats
    )
    quant = types.SimpleNamespace(kv_cache_quant_algo=None, mamba_ssm_cache_dtype="torch.float32")
    model = types.SimpleNamespace(model_config=types.SimpleNamespace(quant_config=quant))
    model_engine = types.SimpleNamespace(model=model, _cuda_graph_batch_sizes=[1, 2, 4])
    engine = types.SimpleNamespace(model_engine=model_engine, kv_cache_manager=kv)
    return types.SimpleNamespace(llm_args=args, engine=engine, rank=0)


def test_the_worker_is_read_from_its_engine():
    facts = engine_facts.read_worker(_worker(_args(), reuse=False))
    config = facts["config"]
    assert config["block_reuse"] is False, "the KV-cache manager's own value wins"
    assert config["kv_cache_quant"] is None
    assert config["mamba_ssm_cache_dtype_resolved"] == "float32"
    assert config["cuda_graph_captured"] == [1, 2, 4]
    assert facts["pools"] == {"kv_budget_bytes": 12 << 30, "kv_max_tokens": 32000}


def _observe(monkeypatch, base_worker: type) -> engine_facts.WorkerObserver:
    """An observer installed on ``base_worker`` as ``tensorrt_llm``'s ``BaseWorker``."""
    module = types.ModuleType("tensorrt_llm.executor.base_worker")
    module.BaseWorker = base_worker
    package = types.ModuleType("tensorrt_llm.executor")
    package.base_worker = module
    monkeypatch.setitem(sys.modules, "tensorrt_llm", types.ModuleType("tensorrt_llm"))
    monkeypatch.setitem(sys.modules, "tensorrt_llm.executor", package)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.executor.base_worker", module)
    observer = engine_facts.WorkerObserver()
    assert observer.install()
    return observer


def test_the_observer_reads_each_worker_as_it_sets_up(monkeypatch):
    calls: list[str] = []

    class BaseWorker:
        def setup_engine(self):
            calls.append("setup")
            self.llm_args.kv_cache_config.enable_block_reuse = False

    observer = _observe(monkeypatch, BaseWorker)
    worker = BaseWorker()
    worker.llm_args, worker.engine, worker.rank = _args(), None, 0
    worker.setup_engine()
    assert calls == ["setup"]
    [facts] = observer.workers
    assert facts["config"]["block_reuse"] is False, "read after the worker adjusted its args"


def test_a_worker_that_cannot_be_read_still_sets_up(monkeypatch):
    class BaseWorker:
        def setup_engine(self):
            return "engine up"

    observer = _observe(monkeypatch, BaseWorker)
    assert BaseWorker().setup_engine() == "engine up", "no llm_args to read, and no matter"
    assert not observer.workers and observer.failures[0].startswith("AttributeError")
    record = engine_facts.collect(types.SimpleNamespace(args=_args()), observer, environ={})
    assert record["worker"] is False and record["errors"]["worker"].startswith("AttributeError")
    line = f"{workload_io.ENGINE_MARKER} {json.dumps(record)}"
    assert "rank 0's worker could not be read: AttributeError" in gates.attest_engine(line, {}).note


def test_without_a_worker_here_the_worker_decided_fields_are_left_out():
    llm = types.SimpleNamespace(args=_args())
    record = engine_facts.collect(llm, None, argv=["--tp_size=4"], environ={})
    assert record["worker"] is False
    assert not set(engine_facts.WORKER_DECIDED) & set(record["config"])
    assert set(record["sources"].values()) == {engine_facts.SOURCE_LLM_ARGS}
    observer = engine_facts.WorkerObserver()
    observer.workers.append(engine_facts.read_worker(_worker(_args())))
    record = engine_facts.collect(llm, observer, environ={})
    assert record["worker"] is True and record["config"]["block_reuse"] is True
    assert set(record["sources"].values()) == {engine_facts.SOURCE_WORKER}
    assert record["pools"]["kv_max_tokens"] == 32000


def _fake_nvml(free_mib: float = 1000.0, total_mib: float = 80000.0) -> types.ModuleType:
    nvml = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    opened: list[str] = []
    nvml.NVMLError = NVMLError
    nvml.nvmlInit = lambda: opened.append("init")
    nvml.nvmlShutdown = lambda: opened.append("shutdown")
    nvml.nvmlDeviceGetCount = lambda: 8
    nvml.nvmlDeviceGetHandleByIndex = lambda i: f"index {i}"
    nvml.nvmlDeviceGetHandleByUUID = lambda u: f"uuid {u}"
    nvml.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(
        free=int(free_mib * (1 << 20)), total=int(total_mib * (1 << 20))
    )
    nvml.opened = opened
    return nvml


def test_free_memory_is_read_for_the_engines_devices(monkeypatch):
    nvml = _fake_nvml()
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    two = engine_facts.device_memory(2, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"})
    assert [d["device"] for d in two["devices"]] == ["4", "5"]
    assert two["devices"][0] == {"device": "4", "free_mib": 1000.0, "total_mib": 80000.0}
    assert nvml.opened == ["init", "shutdown"]
    unset = engine_facts.device_memory(3, {})
    assert [d["device"] for d in unset["devices"]] == ["0", "1", "2"]
    uuid = engine_facts.device_memory(1, {"CUDA_VISIBLE_DEVICES": "GPU-1234"})
    assert uuid["devices"][0]["device"] == "GPU-1234"

    def refuse():
        raise nvml.NVMLError("Driver Not Loaded")

    nvml.nvmlInit = refuse
    assert engine_facts.device_memory(1, {}) == {
        "source": None,
        "devices": [],
        "error": "nvmlInit: Driver Not Loaded",
    }


def test_the_record_drops_the_prompts_and_survives_a_failed_read(capsys):
    assert engine_facts.without_prompts(
        ["--tp_size=1", "--prompt", "one two", "three", "--max_tokens=4", "--prompt=x"]
    ) == ["--tp_size=1", "--max_tokens=4"]

    class Broken:
        @property
        def args(self):
            raise RuntimeError("boom")

    record = engine_facts.emit(Broken(), None)
    assert record == {"error": "RuntimeError: boom"}
    [line] = capsys.readouterr().out.splitlines()
    assert gates.parse_engine_records(line) == [record]


# ------------------------------------------- end to end, through the harness

_FAKE_TRTLLM = {
    "tensorrt_llm/__init__.py": "",
    "tensorrt_llm/executor/__init__.py": "",
    "tensorrt_llm/executor/base_worker.py": """
        import types


        class BaseWorker:
            def __init__(self, llm_args):
                self.llm_args = llm_args
                self.rank = 0
                self.engine = None

            def setup_engine(self):
                kv = self.llm_args.kv_cache_config
                state = kv.mamba_state_config
                # What the model loader does for a hybrid model, here without a word.
                if not (state.periodic_snapshot_interval
                        or state.additional_snapshot_offsets_from_start):
                    kv.enable_block_reuse = False
                kv.max_gpu_total_bytes = 1 << 30
                stats = types.SimpleNamespace(max_num_blocks=100, tokens_per_block=32)
                quant = types.SimpleNamespace(
                    kv_cache_quant_algo=None, mamba_ssm_cache_dtype="torch.float32")
                self.engine = types.SimpleNamespace(
                    model_engine=types.SimpleNamespace(
                        model=types.SimpleNamespace(
                            model_config=types.SimpleNamespace(quant_config=quant)),
                        _cuda_graph_batch_sizes=[1, 2],
                    ),
                    kv_cache_manager=types.SimpleNamespace(
                        enable_block_reuse=kv.enable_block_reuse,
                        tokens_per_block=32,
                        get_kv_cache_stats=lambda: stats,
                    ),
                )
    """,
    "pynvml.py": """
        import os
        import types


        class NVMLError(Exception):
            pass


        def nvmlInit():
            pass


        def nvmlShutdown():
            pass


        def nvmlDeviceGetCount():
            return 1


        def nvmlDeviceGetHandleByIndex(index):
            return index


        def nvmlDeviceGetMemoryInfo(handle):
            free = float(os.environ.get("FAKE_FREE_MIB", "40960")) * (1 << 20)
            return types.SimpleNamespace(free=int(free), total=80 << 30)
    """,
}

_FAKE_QUICKSTART = """
    import argparse
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
            self.mamba_ssm_cache_dtype = fields.get("mamba_ssm_cache_dtype", "bfloat16")
            self.free_gpu_memory_fraction = fields.get("free_gpu_memory_fraction", 0.9)
            state = fields.get("mamba_state_config") or {}
            if isinstance(state, dict):
                state = MambaStateConfig(**state)
            self.mamba_state_config = state
            self.max_gpu_total_bytes = 0

        @property
        def model_fields_set(self):
            return self._stated


    class LLM:
        def __init__(self, model, **kw):
            parallel = types.SimpleNamespace(
                tp_size=kw["tensor_parallel_size"], pp_size=1,
                moe_ep_size=kw["moe_expert_parallel_size"], moe_tp_size=-1,
                enable_attention_dp=kw["enable_attention_dp"])
            self.args = types.SimpleNamespace(
                modeling_v2=kw.get("modeling_v2", "off"),
                kv_cache_config=kw["kv_cache_config"],
                cuda_graph_config=kw["cuda_graph_config"],
                speculative_config=None,
                max_num_tokens=kw["max_num_tokens"],
                max_batch_size=kw["max_batch_size"],
                max_seq_len=kw["max_seq_len"],
                enable_chunked_prefill=kw["enable_chunked_prefill"],
                disable_overlap_scheduler=False,
                parallel_config=parallel,
            )
            # Single-process: rank 0's worker sets up in this process.
            BaseWorker(self.args).setup_engine()

        def generate_async(self, inputs, params, streaming=False):
            sequence = types.SimpleNamespace(
                token_ids=[11, 12], text=" Paris", finish_reason="length")
            return iter([types.SimpleNamespace(outputs=[sequence], prompt_token_ids=[1])])

        def shutdown(self):
            pass


    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--prompt", nargs="+", default=["The capital of France is"])
        parser.add_argument("--max_tokens", type=int, default=64)
        parser.add_argument("--tp_size", type=int, default=1)
        parser.add_argument("--moe_ep_size", type=int, default=-1)
        parser.add_argument("--enable_attention_dp", action="store_true")
        parser.add_argument("--max_seq_len", type=int)
        parser.add_argument("--max_batch_size", type=int, default=2048)
        parser.add_argument("--max_num_tokens", type=int, default=8192)
        parser.add_argument("--kv_cache_fraction", type=float, default=0.9)
        parser.add_argument("--mamba_ssm_cache_dtype", default="bfloat16")
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


    def main():
        args = parse_arguments()
        llm, _ = setup_llm(args)
        for n, prompt in enumerate(args.prompt):
            print(f"[{n}] Prompt: {prompt!r}, Generated text: ' Paris'...")
"""


def _fake_tree(tmp_path: Path) -> tuple[Path, Path]:
    """``(site, repo)``: a fake ``tensorrt_llm`` and ``pynvml``, and a tree with its quickstart."""
    site = tmp_path / "site"
    for name, text in _FAKE_TRTLLM.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    repo = tmp_path / "repo"
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(_FAKE_QUICKSTART), encoding="utf-8")
    return site, repo


def _run_boot(tmp_path: Path, task: dict, site: Path, **env) -> gates.GateResult:
    environment = RunEnvironment(
        kind="local", python=sys.executable, world_size=1, env={"PYTHONPATH": str(site), **env}
    )
    return gates.run_boot_gate(
        environment, task_data=task, unit="core", member="tp1", workdir=tmp_path / "gates"
    )


def test_a_real_harness_run_attests_its_engine(tmp_path):
    """The driver, the observer and the gate together, over a fake tree's quickstart."""
    site, repo = _fake_tree(tmp_path)
    task = _task(tmp_path, trtllm_repo_path=str(repo))
    result = _run_boot(tmp_path, task, site)
    assert result.passed, f"{result.detail}\n{result.stdout_tail}"
    [record] = result.engine
    assert record["reported"] and not record["mismatches"] and not record["not_attested"]
    # The rank hook on the gate's PYTHONPATH had the worker print its own record.
    assert record["sources"]["block_reuse"] == gates.RANKS_SOURCE and record["ranks"] == [0]
    assert record["attested"]["cuda_graph"] == {"batch_sizes": _PADDED, "enable_padding": True}
    assert record["pools"]["kv_max_tokens"] == 3200
    assert record["memory"]["devices"] == [{"device": "0", "free_gib": 40.0, "total_gib": 80.0}]
    assert not result.warnings

    tight = _run_boot(tmp_path, task, site, FAKE_FREE_MIB="1024")
    assert tight.passed and tight.warnings[0].startswith("LOW MARGIN: 1.00 GiB free on device 0")


def test_a_real_harness_run_catches_reuse_the_worker_turned_off_silently(tmp_path):
    """No snapshot policy handed over: the worker drops reuse and logs nothing."""
    site, repo = _fake_tree(tmp_path)
    task = _task(tmp_path, trtllm_repo_path=str(repo))
    options = {**_OPTIONS, "kv_cache_config": {"enable_block_reuse": True}}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    result = _run_boot(tmp_path, task, site)
    assert not result.passed
    assert "block_reuse: declared on, engine off on rank(s) 0 (read from every rank)" in (
        result.detail
    )
    assert "mamba_snapshot: declared interval 0" not in result.detail, "the declaration agrees"


def test_the_shim_prints_the_engine_record_without_the_prompts(tmp_path):
    site, repo = _fake_tree(tmp_path)
    script = repo / "examples" / "llm-api" / "quickstart_advanced.py"
    done = subprocess.run(
        [sys.executable, str(witness.QUICKSTART_SHIM), str(script), "--model_dir=/ckpt",
         "--tp_size=1", "--prompt", "one two", "three"],
        capture_output=True, text=True, timeout=120, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site), gates.MODELING_V2_ENV: "require"},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    [record] = gates.parse_engine_records(done.stdout)
    assert record["argv"] == ["--model_dir=/ckpt", "--tp_size=1"]
    assert record["worker"] is True and record["config"]["modeling_v2"] == "require"
    assert "[1] Prompt: 'three'" in done.stdout


# ----------------------------------------------------- the engine's own code


def test_the_names_the_harness_reads_are_the_engines():
    """Read from the tree's own source, so a rename fails here and not in a gate."""
    worker = REPO_ROOT / "tensorrt_llm" / "executor" / "base_worker.py"
    engine = REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "model_engine.py"
    kv = REPO_ROOT / "cpp" / "tensorrt_llm" / "batch_manager" / "kvCacheManager.cpp"
    if not (worker.is_file() and engine.is_file() and kv.is_file()):
        pytest.skip("the TensorRT-LLM sources are not in this checkout")
    source = worker.read_text(encoding="utf-8")
    assert "def setup_engine(self)" in source and "self.llm_args = llm_args" in source
    assert "self._cuda_graph_batch_sizes = " in engine.read_text(encoding="utf-8")
    assert "Allocated %0.2f GiB for max tokens in paged KV cache (%d)." in kv.read_text(
        encoding="utf-8"
    )
