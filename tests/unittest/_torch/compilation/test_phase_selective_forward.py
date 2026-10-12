# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_pre_hopper

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import _PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm.llmapi import CudaGraphConfig, PrefillCudaGraphBackend, TorchCompileConfig


def _create_prefill_compiled_model() -> tuple[_PrefillCompiledModel, Mock, Mock]:
    eager_model = torch.nn.Module()
    compiled_model = torch.nn.Module()
    eager_forward = Mock(return_value="eager")
    compiled_forward = Mock(return_value="compiled")
    eager_model.forward = eager_forward
    compiled_model.forward = compiled_forward
    return (_PrefillCompiledModel(eager_model, compiled_model), eager_forward, compiled_forward)


def _create_compile_restricted_runner(
    model: _PrefillCompiledModel, enabled: bool = True
) -> DecoderRunner:
    runner = object.__new__(DecoderRunner)
    runner._config = SimpleNamespace(compile_only_piecewise_graphs=enabled)
    runner.model = SimpleNamespace(model=model)
    return runner


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
def test_engine_passes_compile_policy_to_decoder_runner(enabled: bool) -> None:
    engine = Mock()
    engine._compile_only_piecewise_graphs = enabled
    config = PyTorchModelEngine._decoder_runner_config(engine)
    runner_type = Mock()

    result = PyTorchModelEngine._initialize_decoder_runner(
        engine, runner_type, runner_config=config
    )

    assert config.compile_only_piecewise_graphs is enabled
    assert result is runner_type.return_value
    assert runner_type.call_args.args == (engine.model, config)
    assert "prefill_compiled_model" not in runner_type.call_args.kwargs
    assert "torch_compile_bypass" not in runner_type.call_args.kwargs


def test_prefill_compiled_model_uses_compiled_when_eligible() -> None:
    model, eager_forward, compiled_forward = _create_prefill_compiled_model()

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
        return_value=True,
    ):
        assert model("input") == "compiled"
    compiled_forward.assert_called_once_with("input")
    eager_forward.assert_not_called()


def test_prefill_compiled_model_bypass_is_restored() -> None:
    model, eager_forward, compiled_forward = _create_prefill_compiled_model()

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
        return_value=True,
    ):
        assert model() == "compiled"
        with model.bypass():
            assert model() == "eager"
        assert model() == "compiled"
    assert eager_forward.call_count == 1
    assert compiled_forward.call_count == 2


@pytest.mark.parametrize(
    ("can_run_graph", "prefill_graph_eligible", "expected"),
    [
        pytest.param(False, True, "compiled", id="pcg-eligible-prefill"),
        pytest.param(False, False, "eager", id="pcg-ineligible-prefill"),
        pytest.param(True, True, "eager", id="ordinary-generation-capture"),
        pytest.param(False, False, "eager", id="generation-graph-miss"),
    ],
)
def test_decoder_runner_selects_torch_compile_forward(
    can_run_graph: bool,
    prefill_graph_eligible: bool,
    expected: str,
) -> None:
    model, eager_forward, compiled_forward = _create_prefill_compiled_model()
    runner = _create_compile_restricted_runner(model)

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
        return_value=prefill_graph_eligible,
    ) as get_prefill_graph_flag:
        with runner._maybe_bypass_torch_compile(can_run_graph=can_run_graph):
            actual = model()

    assert actual == expected
    selected_forward = compiled_forward if expected == "compiled" else eager_forward
    other_forward = eager_forward if expected == "compiled" else compiled_forward
    selected_forward.assert_called_once_with()
    other_forward.assert_not_called()
    if can_run_graph:
        get_prefill_graph_flag.assert_not_called()
    else:
        get_prefill_graph_flag.assert_called_once_with()


def test_decoder_runner_explicit_bypass_skips_graph_eligibility() -> None:
    model, eager_forward, compiled_forward = _create_prefill_compiled_model()
    runner = _create_compile_restricted_runner(model)

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
        return_value=True,
    ) as get_prefill_graph_flag:
        with runner._maybe_bypass_torch_compile(bypass=True, can_run_graph=False):
            actual = model()

    assert actual == "eager"
    eager_forward.assert_called_once_with()
    compiled_forward.assert_not_called()
    get_prefill_graph_flag.assert_not_called()


def test_decoder_runner_bypass_requires_compile_only_option() -> None:
    model, eager_forward, compiled_forward = _create_prefill_compiled_model()
    runner = _create_compile_restricted_runner(model, enabled=False)

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
        return_value=True,
    ):
        with runner._maybe_bypass_torch_compile(bypass=True):
            actual = model()

    assert actual == "compiled"
    compiled_forward.assert_called_once_with()
    eager_forward.assert_not_called()


def test_decoder_runner_without_compile_restriction() -> None:
    runner = object.__new__(DecoderRunner)
    runner._config = SimpleNamespace(compile_only_piecewise_graphs=False)
    runner.model = torch.nn.Identity()
    with runner._maybe_bypass_torch_compile(bypass=True):
        pass


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("has_wrapper", [False, True])
def test_decoder_runner_validates_compiled_body_at_initialization(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, has_wrapper: bool
) -> None:
    model = object.__new__(DecoderModelForCausalLM)
    torch.nn.Module.__init__(model)
    wrapper, _, _ = _create_prefill_compiled_model()
    model.model = wrapper if has_wrapper else torch.nn.Identity()
    config = SimpleNamespace(compile_only_piecewise_graphs=enabled)
    for method in (
        "_allocate_decoder_buffers",
        "_init_decoder_state",
        "_initialize_cuda_graph_runner",
        "_initialize_breakable_cuda_graph_runner",
    ):
        monkeypatch.setattr(DecoderRunner, method, Mock())
    expected_error = (
        pytest.raises(AssertionError, match="_PrefillCompiledModel decoder body")
        if enabled and not has_wrapper
        else contextlib.nullcontext()
    )
    with expected_error:
        runner = DecoderRunner(
            model,
            config,
            input_processor=Mock(),
            model_caller=Mock(),
            mapping=Mock(),
            dist=None,
            moe_load_balancer=None,
            sparse_attention_config=None,
            torch_compile_backend=None,
            get_runtime_tokens_per_gen_step=Mock(),
            warmup_timer=Mock(),
            iter_states={},
            metrics={},
            kv_cache_manager_key=Mock(),
        )
        assert runner.model.model is model.model


def test_decoder_runner_nested_bypass_restores_after_exception() -> None:
    model, _, _ = _create_prefill_compiled_model()
    runner = _create_compile_restricted_runner(model)
    with runner._maybe_bypass_torch_compile(bypass=True):
        with pytest.raises(RuntimeError, match="forward failed"):
            with runner._maybe_bypass_torch_compile(bypass=True):
                raise RuntimeError("forward failed")
        assert model._bypass_active
    assert not model._bypass_active


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
def test_decoder_runner_warmup_bypasses_only_auxiliary_phases(monkeypatch, enabled: bool) -> None:
    """Exercise the relocated warmup orchestration with eager/compiled routing."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as runner_module
    from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer

    model, _, _ = _create_prefill_compiled_model()
    runner = _create_compile_restricted_runner(model, enabled=enabled)
    runner.model.config = SimpleNamespace(vocab_size=128)
    runner.mapping = SimpleNamespace(cp_size=1, has_cp_helix=lambda: False)
    runner.guided_decoder = None
    runner._metrics = {}
    runner._warmup_timer = _WarmupTimer(rank=0)
    runner.cuda_graph_runner = SimpleNamespace(
        allow_capture=contextlib.nullcontext, preallocate_padding_dummies=Mock()
    )
    runner._agree_warmup_flag = lambda value: value
    runner._agree_warmup_shapes = lambda shapes: shapes
    runner._get_full_general_warmup_requests = Mock(return_value=[(4, 0)])
    runner._get_max_shape_warmup_requests = Mock(return_value=[(8, 0)])
    runner.no_cuda_graph = contextlib.nullcontext
    observed = []

    def observe(phase):
        def forward(*args):
            observed.append((phase, model.use_compiled_forward()))

        return forward

    for method in (
        "_prewarm_engram_kernels",
        "_prewarm_csa2_sparse_indexer",
        "_prewarm_cute_dsl_indexer_q",
        "_ensure_dsa_attn_metadata_for_warmup",
        "_warmup_dg_paged_mqa_logits_metadata",
        "_warmup_cute_dsl_radix_topk",
        "_freeze_eager_workspace_floor",
    ):
        monkeypatch.setattr(runner, method, Mock())
    for method in (
        "_run_attention_warmup",
        "_run_autotuner_warmup",
        "_run_mamba_hybrid_warmup",
        "_general_warmup",
        "_run_cuda_graph_warmup",
    ):
        monkeypatch.setattr(runner, method, observe(method))
    monkeypatch.setattr(runner_module, "warmup_sampling_module", Mock())
    monkeypatch.setattr(runner_module, "log_mem_snapshot", Mock())
    monkeypatch.setattr(runner_module.AutoTuner, "get", Mock())
    monkeypatch.setattr(
        runner_module, "moe_a2a_steady_state_budget_for_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(runner_module, "gc", SimpleNamespace(collect=Mock()))
    with (
        patch("tensorrt_llm._torch.bolt_profiling.maybe_bolt_clear_counters", Mock()),
        patch(
            "tensorrt_llm._torch.custom_ops.torch_custom_ops.MoERunner.clear_all_workspaces", Mock()
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
            return_value=True,
        ),
    ):
        runner._warmup_scheduled(Mock(), SimpleNamespace(context_swa_layer_limit=None))

    assert observed == [
        ("_run_attention_warmup", not enabled),
        ("_general_warmup", True),
        ("_run_autotuner_warmup", not enabled),
        ("_run_mamba_hybrid_warmup", not enabled),
        ("_run_cuda_graph_warmup", True),
        ("_run_cuda_graph_warmup", True),
        ("_general_warmup", not enabled),
    ]
    assert not model._bypass_active


@skip_pre_hopper
@pytest.mark.timeout(1800)
def test_piecewise_cuda_graph_compile_restriction_stability() -> None:
    """Compare compiled and eager decode after compiled piecewise prefill."""
    models_root = llm_models_root()
    if models_root is None:
        pytest.skip("LLM_MODELS_ROOT is not available")
    model_path = models_root / "Qwen3/Qwen3-0.6B"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(
        max_tokens=8, end_id=-1, temperature=0, return_generation_logits=True
    )

    def run(compile_only_piecewise_graphs: bool) -> tuple[list[int], torch.Tensor]:
        torch_compile_config = TorchCompileConfig(
            enable_fullgraph=True,
            compile_only_piecewise_graphs=compile_only_piecewise_graphs,
        )
        with LLM(
            model_path,
            max_seq_len=128,
            max_num_tokens=32,
            max_batch_size=1,
            disable_overlap_scheduler=True,
            gather_generation_logits=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=1),
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
            prefill_capture_num_tokens=[32],
            torch_compile_config=torch_compile_config,
        ) as llm:
            output = llm.generate(prompts, sampling_params=sampling_params)[0].outputs[0]
            assert output.generation_logits is not None
            return list(output.token_ids), output.generation_logits.clone()

    compiled_tokens, compiled_logits = run(False)
    restricted_tokens, restricted_logits = run(True)

    assert restricted_tokens == compiled_tokens
    torch.testing.assert_close(restricted_logits, compiled_logits, rtol=1e-2, atol=1e-2)
