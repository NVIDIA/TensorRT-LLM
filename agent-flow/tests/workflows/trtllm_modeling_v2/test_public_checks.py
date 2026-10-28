# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public entry boundaries that change a check's verdict or workload."""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import (
    check_config,
    runtime_evidence,
    witness_protocol,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import (
    engine_driver,
    reuse_probe,
    workload_io,
)
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight

from .test_gates import _TASK, _env


@pytest.mark.parametrize("driver", ["logit_driver.py", "serve_driver.py", "test_driver.py"])
def test_measured_drivers_start_by_path_without_agent_flow(driver, tmp_path):
    # -I -S hides the controller PYTHONPATH and editable-install hooks. Add only
    # installed dependencies (serve's benchmark client needs PyYAML), not agent_flow.
    path = Path(engine_driver.__file__).with_name(driver)
    script = (
        "import importlib.util, runpy, sys; "
        "sys.path.append(sys.argv.pop(1)); "
        "assert importlib.util.find_spec('agent_flow') is None; "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            sysconfig.get_path("purelib"),
            str(path),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_shared_options_preserve_nested_settings_and_refuse_ambiguous_inputs(tmp_path):
    path = tmp_path / "options.json"
    nested = {
        "kv_cache_config": {"mamba_state_config": {"periodic_snapshot_interval": 256}},
        "speculative_config": {
            "decoding_type": "DSpark",
            "max_draft_len": 3,
            "draft_len_schedule": {1: 2},
        },
        "moe_config": {"backend": "TRTLLM"},
    }
    path.write_text(json.dumps(nested))
    task = {**_TASK, "accuracy": {"llm_api_options": str(path)}}
    config = check_config.check_configuration(
        task, _env(), unit="dspark", features=["dspark"], max_tokens=8, member="tp1"
    )
    assert config["llm"]["speculative_config"]["decoding_type"] == "DSpark"
    assert config["llm"]["speculative_config"]["draft_len_schedule"] == {"1": 2}
    assert (
        config["llm"]["kv_cache_config"]["mamba_state_config"]
        == nested["kv_cache_config"]["mamba_state_config"]
    )
    assert config["sampling"]["temperature"] == 0.0
    task["features"] = ["dspark"]
    core = check_config.check_configuration(
        task, _env(), unit="core", features=None, max_tokens=8, member="tp1"
    )
    assert core["llm"]["speculative_config"] is None
    later = check_config.check_configuration(
        task, _env(), unit="dspark", features=None, max_tokens=8, member="tp1"
    )
    assert later["llm"]["speculative_config"]["decoding_type"] == "DSpark"
    task["runtime"] = {"extra_flags": ["--unknown-engine-option=1"]}
    with pytest.raises(preflight.PreflightError, match="runtime.extra_flags"):
        preflight.check_launch_settings(task)
    task["runtime"] = {"extra_flags": ["--spec_decode_algo=MTP"]}
    with pytest.raises(ValueError, match="conflicting check inputs"):
        check_config.check_configuration(
            task, _env(), unit="dspark", features=["dspark"], max_tokens=8, member="tp1"
        )


def test_public_entry_preserves_witness_warmup_and_all_sequences(monkeypatch, capsys):
    calls = []
    sequences = [
        SimpleNamespace(token_ids=[i], text=f"answer {i}", finish_reason="length") for i in (1, 2)
    ]
    output = SimpleNamespace(outputs=sequences, prompt_token_ids=[3], cached_tokens=0)

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))

        def generate(self, prompts, params):
            calls.append(("warmup", prompts))

        def generate_async(self, prompt, params, streaming=False):
            calls.append(("stream", prompt))
            return iter([output])

    monkeypatch.setitem(
        sys.modules, "tensorrt_llm", SimpleNamespace(LLM=LLM, SamplingParams=Params)
    )
    monkeypatch.setenv("TRTLLM_WITNESS_FULL_OUTPUT", "1")
    spec = {
        "llm": {"model": "/model", "kv_cache_config": {"enable_block_reuse": True}},
        "sampling": {"n": 2},
        "warmup": True,
        "witness": True,
        "requests": [{"id": "w0", "prompt": "hello", "max_tokens": 2}],
    }
    llm, params = engine_driver.build_llm(spec)
    engine_driver.run_workload(llm, params, spec)
    assert [c[0] for c in calls] == ["construct", "warmup", "stream"]
    assert calls[0][1]["kv_cache_config"] == spec["llm"]["kv_cache_config"]
    text = capsys.readouterr().out
    assert witness_protocol.parse_witness_token_ids(text) == ((1,), (2,))
    with pytest.raises(ValueError, match="one sequence"):
        engine_driver.build_llm({**spec, "witness": False})
    with pytest.raises(ValueError, match="normalized"):
        engine_driver.build_llm({**spec, "dataset": {"source": "offline"}})
    assert len(calls) == 3  # Unsupported workloads fail before allocating an engine.


def test_normalization_preserves_source_skip_template_order_and_image_requests(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"q": "skip", "tag": "x"},
                {},
                {"q": "first", "tag": "a"},
                {"q": "second", "tag": "b"},
            ]
        )
    )
    image = {
        "id": "image",
        "prompt": "look",
        "images": ["/local/image.png"],
        "render": "raw",
        "max_tokens": 3,
    }
    spec = {
        "requests": [image],
        "dataset": {
            "source": str(data),
            "text_field": "q",
            "skip": 1,
            "num": 2,
            "template": "{tag}: {text}",
        },
    }
    requests = workload_io.normalize_requests(spec)
    assert requests[0] == image
    assert [r["prompt"] for r in requests[1:]] == ["a: first", "b: second"]
    spec["dataset"]["skip"] = 99
    with pytest.raises(workload_io.DatasetError, match="no usable prompts"):
        workload_io.normalize_requests(spec)


def test_requested_inputs_and_partial_ranks_do_not_attest_actual_configuration():
    declared = {"tp": 2, "pp": 1, "block_reuse": True}
    output = f"{workload_io.ENGINE_MARKER} " + json.dumps({"config": {}, "requested": declared})
    attested = runtime_evidence.attest_engine(output, declared)
    assert set(attested.not_attested) == set(declared)
    output += f"\n{workload_io.RANK_ENGINE_MARKER} " + json.dumps({"rank": 0, "config": declared})
    attested = runtime_evidence.attest_engine(output, declared)
    assert set(attested.not_attested) == set(declared)
    assert "coverage UNKNOWN" in attested.note
    # An arbitrary evaluation with zero hits is not a controlled negative experiment.
    assert runtime_evidence.reuse_shortfall(attested, asks=True) is None
    unknown = runtime_evidence.EngineAttestation(
        leg="", reported=True, reuse={"probe": {"status": "UNKNOWN", "skipped": "no capacity"}}
    )
    assert "UNKNOWN" in runtime_evidence.reuse_shortfall(unknown, asks=True)


def test_repeat_probe_completes_before_resubmission_and_pins_attention_rank(monkeypatch):
    events = []
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.scheduling_params",
        SimpleNamespace(SchedulingParams=lambda **kw: kw),
    )

    class LLM:
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])

        def get_kv_cache_capacity(self):
            return {"tokensPerBlock": 4, "maxNumTokens": 128}

        def generate_async(self, prompt, params, **kwargs):
            events.append(("submit", kwargs))
            return SimpleNamespace(
                result=lambda: (events.append(("complete", {})) or SimpleNamespace(cached_tokens=4))
            )

    result = reuse_probe.probe(LLM(), SimpleNamespace(), {"enable_attention_dp": True})
    assert result["status"] == "OBSERVED" and result["reused"] == 4
    assert [e[0] for e in events] == ["submit", "complete", "submit", "complete"]
    assert events[0][1] == events[2][1]
    assert events[0][1]["scheduling_params"] == {
        "attention_dp_rank": 0,
        "attention_dp_relax": False,
    }
    assert reuse_probe.probe(SimpleNamespace(), SimpleNamespace(), {})["status"] == "UNKNOWN"
