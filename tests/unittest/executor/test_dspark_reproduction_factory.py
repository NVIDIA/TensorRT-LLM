# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run in the installed native runtime on CPU before a public GPU caller.

Actual TorchLlmArgs/LLM validation and factory signatures are exercised; only
costly model construction is intercepted. These do not prove native GPU math.
"""

import importlib.util
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tensorrt_llm._torch.pyexecutor import (
    dspark_observation_runtime,
    dspark_request_profile,
    py_executor_creator,
)
from tensorrt_llm._torch.speculative import dspark_reproduction
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", ["static", "dynamic", "collect"])
def test_complete_public_parser_actual_llm_validator_and_internal_factory(
    tmp_path, monkeypatch, mode
):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=CPU Control",
            "-c",
            "user.email=cpu-control@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "CPU",
        ],
        check=True,
        capture_output=True,
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"dspark_block_size": 3}', encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map": {"mtp.0.weight": "not-a-real-model.safetensors"}}', encoding="utf-8"
    )
    config = {
        "skip_tokenizer_init": True,
        "max_batch_size": 2,
        "max_seq_len": 128,
        "max_num_tokens": 16,
        "disable_overlap_scheduler": False,
        "cuda_graph_config": {"batch_sizes": [1, 2], "enable_padding": True},
        "speculative_config": {
            "decoding_type": "DSpark",
            "max_draft_len": 3,
            "speculative_model": str(model),
        },
    }
    config_path = tmp_path / "runtime.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        "\n".join(json.dumps({"prompt_token_ids": [4, 5]}) for _ in range(2)), encoding="utf-8"
    )
    cost = tmp_path / "structure-is-not-validated-at-argument-parser.json"
    cost.write_text("{}", encoding="utf-8")
    script = Path(dspark_reproduction.__file__).parents[2] / "tools/dspark_reproduction.py"
    spec = importlib.util.spec_from_file_location("_actual_public_endpoint_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []

    def prepare_only(args):
        llm_args, rows, contract, _ = dspark_reproduction._prepare(args, 1)
        assert isinstance(llm_args, TorchLlmArgs)
        assert llm_args.speculative_config.max_draft_len == 3
        assert llm_args.speculative_config.enable_confidence_scheduling == (mode == "dynamic")
        # Regression: internal executor_cls MUST fail the genuine public LLM
        # boundary, not be smuggled into public kwargs and discovered on a GPU.
        public_kwargs = {**config, "model": str(model), "backend": "pytorch"}
        LLM._validate_args_for_torch_backend(None, public_kwargs)
        with pytest.raises(ValueError, match="executor_cls"):
            LLM._validate_args_for_torch_backend(None, {**public_kwargs, "executor_cls": object})
        original = py_executor_creator.create_py_executor

        def capture_factory(**kwargs):
            # functools.wraps exposes the inner implementation by default;
            # bind the real callable, which owns its internal startup timer.
            outer_signature = inspect.signature(original, follow_wrapped=False)
            assert "_startup_timer" not in outer_signature.parameters
            outer_signature.bind(**kwargs)
            assert "executor_cls" not in kwargs
            assert kwargs["llm_args"] is llm_args
            assert isinstance(
                kwargs["runtime_constructors"], py_executor_creator._RuntimeConstructors
            )
            seen.append(kwargs)
            inner = py_executor_creator._create_py_executor_impl
            forwarded = []

            def capture_model_construction(**arguments):
                inspect.signature(inner, follow_wrapped=False).bind(**arguments)
                timer = arguments["_startup_timer"]
                assert isinstance(timer, py_executor_creator._StartupTimer)
                assert timer.name == "executor_creation" and timer.started > 0
                assert {
                    key: value for key, value in arguments.items() if key != "_startup_timer"
                } == {
                    "llm_args": llm_args,
                    "checkpoint_dir": kwargs["checkpoint_dir"],
                    "tokenizer": kwargs["tokenizer"],
                    "profiling_stage_data": None,
                    "resource_governor_queue": None,
                    "runtime_constructors": kwargs["runtime_constructors"],
                }
                forwarded.append(arguments)
                return "CPU constructor boundary only"

            # Execute the genuine decorated wrapper and real host-only timer;
            # intercept only the inner expensive model construction boundary.
            monkeypatch.setattr(
                py_executor_creator, "_create_py_executor_impl", capture_model_construction
            )
            result = original(**kwargs)
            assert len(forwarded) == 1
            return result

        monkeypatch.setattr(py_executor_creator, "create_py_executor", capture_factory)
        assert (
            dspark_observation_runtime.create_observation_executor(
                llm_args=llm_args,
                checkpoint_dir=str(model),
                tokenizer=None,
                collect_calibration=mode == "collect",
            )
            == "CPU constructor boundary only"
        )
        assert len(rows) == 2 and contract["batch_size"] == 2

    monkeypatch.setattr(dspark_reproduction, "run_native", prepare_only)
    args = [
        "run",
        "--mode",
        mode,
        "--model",
        str(model),
        "--config",
        str(config_path),
        "--inputs",
        str(inputs),
        "--source-root",
        str(root),
        "--physical-k",
        "3",
        "--batch-size",
        "2",
        "--output-tokens",
        "8",
        "--timeout",
        "3600",
        "--output",
        str(tmp_path / "private-output"),
    ]
    if mode == "dynamic":
        args += ["--cost-table", str(cost)]
    cli.main(args)
    assert len(seen) == 1
    # Keep the original setup (including MPI/device/loader handling), overriding
    # only its existing internal factory seam.
    assert dspark_reproduction.ObservationWorker.setup_engine is BaseWorker.setup_engine


@pytest.mark.cpu_only
def test_offline_head_evaluation_rejects_enabled_policy_before_factory(monkeypatch, tmp_path):
    def forbidden(**kwargs):
        raise AssertionError("must refuse before model construction")

    monkeypatch.setattr(py_executor_creator, "create_py_executor", forbidden)
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"mtp.0.weight": "not-a-real-model.safetensors"}}', encoding="utf-8"
    )
    spec = DSparkDecodingConfig(
        max_draft_len=3, speculative_model=str(tmp_path), enable_confidence_scheduling=True
    )
    with pytest.raises(ValueError, match="native full-K"):
        dspark_observation_runtime.create_observation_executor(
            llm_args=SimpleNamespace(speculative_config=spec), collect_calibration=True
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("resource_governor", [False, True])
def test_profile_factory_keeps_explicit_offline_constructors(
    monkeypatch: pytest.MonkeyPatch, resource_governor: bool
) -> None:
    profile = object()
    llm_args = SimpleNamespace(enable_resource_governor=resource_governor)
    seen = []
    result = object()

    def capture_factory(**kwargs):
        inspect.signature(py_executor_creator.create_py_executor, follow_wrapped=False).bind(
            **kwargs
        )
        seen.append(kwargs)
        return result

    monkeypatch.setattr(dspark_request_profile, "create_py_executor", capture_factory)
    if resource_governor:
        with pytest.raises(ValueError, match="resource-governor"):
            dspark_request_profile.create_profile_executor(llm_args=llm_args, profile=profile)
        assert not seen
        return

    assert (
        dspark_request_profile.create_profile_executor(
            llm_args=llm_args, profile=profile, checkpoint_dir="model", tokenizer=None
        )
        is result
    )
    assert len(seen) == 1
    constructors = seen[0]["runtime_constructors"]
    assert isinstance(constructors, py_executor_creator._RuntimeConstructors)
    assert constructors.model_engine.func is dspark_request_profile.OwnedDSparkProfileEngine
    assert constructors.model_engine.keywords == {"profile": profile}
    assert constructors.executor is dspark_request_profile.OwnedDSparkProfileExecutor
    assert constructors.kv_cache_creator is dspark_request_profile._ProfileKvCacheCreator


@pytest.mark.cpu_only
@pytest.mark.parametrize("active_profile", [False, True], ids=["native", "measured"])
@pytest.mark.parametrize("position_id_offsets", [None, object()], ids=["default", "offsets"])
def test_profile_replay_forwards_parent_position_offsets(
    monkeypatch, active_profile, position_id_offsets
):
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import DSparkCUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.dspark_profile_engine import _DSparkProfileGraphs

    # A newly added parent keyword must be explicitly supported, with the same
    # default, before the expensive model warmup reaches this override.
    parent_parameters = inspect.signature(DSparkCUDAGraphRunner.replay).parameters
    profile_parameters = inspect.signature(_DSparkProfileGraphs.replay).parameters
    parent_keywords = {
        name: parameter
        for name, parameter in parent_parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert "position_id_offsets" in parent_keywords
    assert parent_keywords.keys() <= profile_parameters.keys()
    for name, parameter in parent_keywords.items():
        assert profile_parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert profile_parameters[name].default == parameter.default
    assert profile_parameters["position_id_offsets"].default is None

    key, output, lease = (64, 3, False), object(), object()
    attention, speculative = object(), object()
    inputs = {
        "attn_metadata": attention,
        "spec_metadata": speculative,
        "optional_metadata": object(),
    }
    calls, events = [], []

    def replay(self, actual_key, current_inputs, *, position_id_offsets=None):
        calls.append((self, actual_key, current_inputs, position_id_offsets))
        events.append("replay")
        return output

    def stable(metadata, snapshot, *, context):
        assert metadata is attention or metadata is speculative
        assert context["rank"] == 0

    monkeypatch.setattr(DSparkCUDAGraphRunner, "replay", replay)
    monkeypatch.setattr(
        _DSparkProfileGraphs, "_assert_graph_tensor_ptrs_stable", staticmethod(stable)
    )
    runner = _DSparkProfileGraphs.__new__(_DSparkProfileGraphs)
    runner.active_profile_cell = (
        SimpleNamespace(graph_batch_size=64, verifier_budget=0) if active_profile else None
    )
    runner.config = SimpleNamespace(mapping=SimpleNamespace(rank=0))
    runner._profile = SimpleNamespace(max_samples=4, physical_k=3)
    runner._profile_records = []
    runner._profile_ptrs = {key: ({}, {})}
    runner.graph_metadata = {key: inputs}
    runner.active_request_ids = ("request",)
    runner._profile_pool = SimpleNamespace(
        begin=lambda: events.append("begin") or lease,
        end=lambda actual: events.append(("end", actual is lease)),
    )
    assert runner.replay(key, inputs, position_id_offsets=position_id_offsets) is output
    assert len(calls) == 1
    assert calls[0][0] is runner and calls[0][1] is key
    assert calls[0][2] is inputs and calls[0][3] is position_id_offsets
    assert events == (["begin", "replay", ("end", True)] if active_profile else ["replay"])
    assert len(runner._profile_records) == int(active_profile)


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True], ids=["native-collect", "dspark"])
@pytest.mark.parametrize("position_id_offsets", [None, object()], ids=["default", "offsets"])
def test_observation_replay_forwards_actual_parent_keyword_contract(
    tmp_path, monkeypatch, enabled, position_id_offsets
):
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
        CUDAGraphRunner,
        DSparkCUDAGraphRunner,
        KeyType,
    )
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"mtp.0.weight": "not-a-real-model.safetensors"}}', encoding="utf-8"
    )
    spec = DSparkDecodingConfig(
        max_draft_len=3,
        speculative_model=str(tmp_path),
        enable_confidence_scheduling=enabled,
    )
    captured = []

    def capture_factory(**kwargs):
        captured.append(kwargs["runtime_constructors"])
        return "CPU factory boundary only"

    monkeypatch.setattr(py_executor_creator, "create_py_executor", capture_factory)
    assert (
        dspark_observation_runtime.observation_factory(collect_calibration=not enabled)(
            llm_args=SimpleNamespace(speculative_config=spec), tokenizer=None
        )
        == "CPU factory boundary only"
    )
    engine_class = captured[0].model_engine
    monkeypatch.setattr(
        PyTorchModelEngine,
        "_initialize_decoder_runner",
        lambda self, runner_cls, runner_config=None, **kwargs: runner_cls,
    )
    engine = engine_class.__new__(engine_class)
    decoder_class = engine._initialize_decoder_runner(DecoderRunner)
    base = DSparkCUDAGraphRunner if enabled else CUDAGraphRunner
    parent_parameters = inspect.signature(base.replay).parameters
    monkeypatch.setattr(base, "__init__", lambda self, config: None)
    decoder = decoder_class.__new__(decoder_class)
    graph = decoder._create_cuda_graph_runner(SimpleNamespace())
    assert type(graph).__mro__[1] is base
    parameters = inspect.signature(type(graph).replay).parameters
    keywords = {
        name: parameter
        for name, parameter in parent_parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert "position_id_offsets" in keywords
    assert keywords.keys() <= parameters.keys()
    for name, parameter in keywords.items():
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default == parameter.default

    key = KeyType(64, 3, False, ragged_verify_bucket=64 if enabled else None)
    inputs = {"attn_metadata": object(), "spec_metadata": object(), "optional": object()}
    output, calls = object(), []

    def replay(self, actual_key, actual_inputs, *, position_id_offsets=None):
        calls.append((self, actual_key, actual_inputs, position_id_offsets))
        return output

    monkeypatch.setattr(base, "replay", replay)
    assert graph.replay(key, inputs, position_id_offsets=position_id_offsets) is output
    assert len(calls) == 1
    assert calls[0][0] is graph and calls[0][1] is key
    assert calls[0][2] is inputs and calls[0][3] is position_id_offsets
    assert graph.observed_routes == {
        "native_replays": int(not enabled),
        "compact_replays": int(enabled),
    }
    assert len(graph.observed_keys) == 1
    with pytest.raises(TypeError):
        graph.replay(key, inputs, unsupported_next_keyword=object())
