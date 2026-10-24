# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for warmup-cleanup behavior in PyTorchModelEngine.warmup().

Locks in that gc.collect() + torch.cuda.empty_cache() fire immediately after
_run_autotuner_warmup (step b) to release autotuner exploration leftovers.

The torch.cuda.empty_cache() after teardown_managers() in py_executor_creator
is covered end-to-end by integration tests rather than unit-tested here.
"""

import contextlib
import gc
import os
import sys
import unittest
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.pyexecutor.engine.model_call as model_call_module
import tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner as decoder_runner_module
import tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup as decoder_warmup_module
import tensorrt_llm._torch.pyexecutor.model_engine as model_engine_module
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, remove_weights
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import DecoderWarmup
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.speculative.utils import resolve_draft_len, update_draft_len
from tensorrt_llm._torch.utils import is_torch_compiling, torch_compiling
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    DraftTargetDecodingConfig,
    PARDDecodingConfig,
    PrefillCudaGraphBackend,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_kind", ["default", "m3", "m3_vl", "gdn", "mamba"])
@pytest.mark.parametrize("compile_enabled,piecewise", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("compile_only_piecewise_graphs", [False, True])
def test_pcg_fx_fallback_policy_is_model_specific(
    monkeypatch: pytest.MonkeyPatch,
    model_kind: str,
    compile_enabled: bool,
    piecewise: bool,
    compile_only_piecewise_graphs: bool,
) -> None:
    """Exercise the real constructor's routing gate without loading weights or CUDA."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        MiniMaxM3ForCausalLM,
        MiniMaxM3VLForConditionalGeneration,
    )
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHForCausalLM
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM
    from tensorrt_llm._torch.pyexecutor import _util

    model_cls = {
        "default": DecoderModelForCausalLM,
        "m3": MiniMaxM3ForCausalLM,
        "m3_vl": MiniMaxM3VLForConditionalGeneration,
        "gdn": Qwen3NextForCausalLM,
        "mamba": NemotronHForCausalLM,
    }[model_kind]
    assert model_cls.use_fx_for_pcg_fallback is (model_kind not in ("m3", "m3_vl"))
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    mapping = Mapping()
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(torch_dtype=torch.float32, hidden_size=4),
        mapping=mapping,
        sparse_attention_config=None,
    )
    eager = torch.nn.Linear(4, 4)
    model.model = eager
    compile_config = SimpleNamespace(
        enable_fullgraph=True,
        compile_only_piecewise_graphs=compile_only_piecewise_graphs,
        enable_inductor=False,
        enable_userbuffers=False,
        max_num_streams=1,
    )
    llm_args = SimpleNamespace(
        encode_only=False,
        mm_encoder_only=False,
        get_runtime_sizes=lambda: (1, 16, 64, 4),
        encoder_max_batch_size=None,
        encoder_max_num_tokens=None,
        enable_in_graph_sampling=False,
        enable_return_routed_experts=False,
        multimodal_config=SimpleNamespace(video_pruning_rate=None),
        checkpoint_format="HF",
        trust_remote_code=False,
        disable_overlap_scheduler=True,
        kv_cache_config=KvCacheConfig(),
        sleep_config=None,
        enable_layerwise_nvtx_marker=False,
        cuda_graph_config=None,
        torch_compile_config=compile_config if compile_enabled else None,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE if piecewise else None,
        prefill_capture_num_tokens=[4],
        allreduce_strategy="AUTO",
        attn_backend="TRTLLM",
        sparse_attention_config=None,
    )
    for name in (
        "should_enable_overlap_headroom",
        "resolved_kv_cache_manager_is_v2",
    ):
        monkeypatch.setattr(_util, name, Mock(return_value=False))
    monkeypatch.setattr(_util, "compute_max_num_sequences", Mock(return_value=4))
    for name in (
        "_configure_deep_gemm_pdl",
        "create_input_processor",
        "setup_mm_encoder_attn_metadata",
    ):
        monkeypatch.setattr(model_engine_module, name, Mock())
    monkeypatch.setattr(
        model_engine_module, "resolve_mrope_position_deltas_cache", lambda model: None
    )
    monkeypatch.setattr(
        model_engine_module, "is_hybrid_linear", lambda config: model_kind in ("gdn", "mamba")
    )
    monkeypatch.setattr(
        model_engine_module.MultimodalItemScheduler, "maybe_create", Mock(return_value=None)
    )
    monkeypatch.setattr(PyTorchModelEngine, "_validate_breakable_cuda_graph_compatibility", Mock())
    monkeypatch.setattr(PyTorchModelEngine, "_init_model_capacity", Mock())
    monkeypatch.setattr(PyTorchModelEngine, "__del__", lambda self: None)
    backend_factory = Mock()
    backend_factory.Streams = list
    monkeypatch.setattr(model_engine_module, "Backend", backend_factory)
    compiled = torch.nn.Identity()
    compile_model = Mock(return_value=compiled)
    monkeypatch.setattr(torch, "compile", compile_model)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 16)
    # Stop after the complete compilation block, before runtime cache allocation.
    monkeypatch.setattr(
        model_engine_module,
        "get_attention_backend",
        Mock(side_effect=RuntimeError("compile setup complete")),
    )
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    with torch_compiling(False), pytest.raises(RuntimeError, match="compile setup complete"):
        PyTorchModelEngine.__init__(
            engine,
            model_path="dummy",
            mapping=mapping,
            model=model,
            llm_args=llm_args,
            checkpoint_loader=Mock(),
        )

    expected_prefill_only = (
        compile_enabled
        and piecewise
        and (compile_only_piecewise_graphs or model_kind in ("m3", "m3_vl"))
    )
    assert engine._torch_compile_prefill_only is expected_prefill_only
    if not compile_enabled:
        compile_model.assert_not_called()
        assert model.model is eager
    else:
        compile_model.assert_called_once_with(
            eager, backend=engine._torch_compile_backend, fullgraph=True
        )
        assert backend_factory.call_args.args[0] is False  # Inductor stays disabled.
        if expected_prefill_only:
            assert isinstance(model.model, PrefillCompiledModel)
            assert model.model.eager_model is eager
            assert model.model.compiled_model is compiled
        else:
            assert model.model is compiled


@pytest.mark.cpu_only
@pytest.mark.parametrize("local_contexts", [0, 1])
def test_prefill_compile_uses_all_rank_prefill_decision(
    monkeypatch: pytest.MonkeyPatch,
    local_contexts: int,
) -> None:
    """Route using the global prefill decision, regardless of local contexts."""
    eager = torch.nn.Linear(4, 4)
    compiled = torch.nn.Module()
    compiled.shared = eager
    expected = torch.ones((2, 4))
    eager.forward = Mock(return_value=expected)
    compiled.forward = Mock(return_value=expected)
    router = PrefillCompiledModel(eager, compiled)
    assert router.weight is eager.weight
    assert list(router.parameters()) == list(eager.parameters())
    # Local decode-only ranks participate when another attention-DP rank
    # prefills; an over-ceiling local context batch uses eager instead.
    metadata = SimpleNamespace(num_contexts=local_contexts)
    for eligible in (True, False):
        monkeypatch.setattr(
            decoder_runner_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
        )
        assert router(expected, attn_metadata=metadata) is expected
        selected = compiled if eligible else eager
        selected.forward.assert_called_once_with(expected, attn_metadata=metadata)
    assert compiled.forward.call_count == eager.forward.call_count == 1


@pytest.mark.cpu_only
def test_prefill_compile_preserves_partial_weight_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload selected weights by original names into both model entry points."""
    import tensorrt_llm._torch.models.modeling_utils as modeling_utils

    eager = torch.nn.Sequential(OrderedDict(layers=torch.nn.Sequential(torch.nn.Linear(4, 4))))
    compiled = torch.compile(eager, backend="eager")
    model = DecoderModelForCausalLM.__new__(DecoderModelForCausalLM)
    torch.nn.Module.__init__(model)
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.model = PrefillCompiledModel(eager, compiled)
    mapper = HfWeightMapper()
    mapper._model = model
    loader = ModelLoader.__new__(ModelLoader)
    loader.weight_mapper = mapper
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")
    monkeypatch.setattr(modeling_utils, "local_mpi_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    weight = eager.layers[0].weight
    old_bias = eager.layers[0].bias.detach().clone()
    replacement = torch.full_like(weight, 3)
    for remove_duplicate in (False, True):
        names = dict(model.named_modules(remove_duplicate=remove_duplicate))
        assert names["model.layers.0"] is eager.layers[0]
        assert not any("eager_model" in name or "compiled_model" in name for name in names)
    assert list(model.model.children()) == [eager]
    assert list(model.model.state_dict()) == [
        "eager_model.layers.0.weight",
        "eager_model.layers.0.bias",
    ]
    apply_tensor = Mock(side_effect=lambda tensor: tensor)
    model.model._apply(apply_tensor)
    assert apply_tensor.call_count == 2
    loader.reload(model, {"model.layers.0.weight": replacement}, allow_partial_loading=True)

    assert eager.layers[0].weight is weight
    assert compiled.layers[0].weight is weight
    torch.testing.assert_close(weight, replacement)
    torch.testing.assert_close(eager.layers[0].bias, old_bias)
    inputs = torch.ones(2, 4)
    for eligible in (False, True):
        monkeypatch.setattr(
            decoder_runner_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
        )
        torch.testing.assert_close(model.model(inputs), inputs @ replacement.t() + old_bias)


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefill_only", [False, True])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_prefill_compile_scopes_whole_model_forward(
    monkeypatch: pytest.MonkeyPatch,
    prefill_only: bool,
    eligible: bool,
    bypass: bool,
    raises: bool,
) -> None:
    """Scope compile dispatch by phase independently of the model wrapper."""
    observed = []

    def forward(**kwargs: object) -> str:
        """Record compile state as a stand-in for a model-specific epilogue."""
        observed.append(is_torch_compiling())
        # This represents work after the transformer, such as Eagle3 drafting.
        if raises:
            raise RuntimeError("epilogue failure")
        observed.append(is_torch_compiling())
        return "done"

    eager_model = torch.nn.Identity()
    eager_model.model_config = SimpleNamespace(extra_attrs={})
    eager_model.forward = forward
    compiled_model = torch.nn.Identity()
    compiled_model.forward = forward
    prefill_compiled_model = PrefillCompiledModel(eager_model, compiled_model)
    model = prefill_compiled_model if prefill_only else eager_model
    engine = SimpleNamespace(
        _model_caller=ModelCaller(model, prefill_compile_only=prefill_only),
        _eager_workspace_reclaimer=None,
        _moe_workspace_reclaimer=None,
    )
    monkeypatch.setattr(model_call_module, "get_model_extra_attrs", lambda: {})
    monkeypatch.setattr(
        model_call_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
    )
    monkeypatch.setattr(model_call_module, "is_trace_enabled", lambda name: False)
    bypass_scope = (
        prefill_compiled_model.bypass() if prefill_only and bypass else contextlib.nullcontext()
    )
    with torch_compiling(True), bypass_scope:
        if raises:
            with pytest.raises(RuntimeError, match="epilogue failure"):
                DecoderRunner._model_forward(engine, is_dummy=False, attn_metadata=Mock())
        else:
            assert (
                DecoderRunner._model_forward(engine, is_dummy=False, attn_metadata=Mock()) == "done"
            )
        assert is_torch_compiling()
    expected = eligible if prefill_only else True
    assert observed == [expected] * (1 if raises else 2)


@pytest.mark.cpu_only
@pytest.mark.parametrize("compile_mode", ["eager", "all_batches", "prefill_only"])
@pytest.mark.parametrize("backend", [None, "auto", "flashinfer", "trtllm"])
def test_compiled_mxfp8_warmup_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    compile_mode: str,
    backend: str | None,
) -> None:
    """Tune only backends reached by real dispatch, preserving explicit choices."""
    import tensorrt_llm._torch.modules.linear as linear_module

    compile_enabled = compile_mode != "eager"
    prefill_only = compile_mode == "prefill_only"
    inputs = torch.zeros(2, 4)
    output = torch.zeros(2, 3)
    layer = SimpleNamespace(
        weight=torch.zeros(3, 4),
        weight_scale=torch.ones(4),
        _mxfp8_alpha=torch.ones(1, dtype=torch.float32),
        dtype=torch.float32,
    )
    native_gemm = Mock(return_value=output)
    flashinfer_gemm = Mock(return_value=output)
    monkeypatch.delenv("TRTLLM_MXFP8_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("TLLM_AUTOTUNER_CACHE_PATH", raising=False)
    if backend is not None:
        monkeypatch.setenv("TRTLLM_MXFP8_GEMM_BACKEND", backend)
    monkeypatch.setattr(linear_module, "_mxfp8_cutlass_op_available", lambda: True)
    monkeypatch.setattr(torch.ops.trtllm, "flashinfer_mm_mxfp8", flashinfer_gemm, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm, "mxfp8_quantize", Mock(return_value=(inputs, inputs)), raising=False
    )
    for name in ("mxfp8_mxfp8_gemm", "mxfp8_mxfp8_gemm_autotuned"):
        monkeypatch.setattr(torch.ops.trtllm, name, native_gemm, raising=False)
    flashinfer_tune = Mock(return_value=contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(autotune=flashinfer_tune))
    method = MXFP8LinearMethod()
    engine = SimpleNamespace(
        _warmup_timer=_WarmupTimer(rank=0),
        _config=SimpleNamespace(
            enable_autotuner=True,
            torch_compile_enabled=compile_enabled,
            torch_compile_prefill_only=prefill_only,
            max_num_tokens=16,
            max_batch_size=16,
            max_seq_len=2,
            original_max_draft_len=0,
            max_total_draft_tokens=0,
            is_spec_decode=False,
            max_draft_len=0,
            spec_config=None,
        ),
        _completed_autotuner_warmup_num_tokens=None,
        _eager_workspace_reclaimer=None,
        _moe_workspace_reclaimer=None,
        is_warmup=True,
        model=SimpleNamespace(
            modules=lambda: [
                SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                SimpleNamespace(quant_method=method),
            ],
            model_config=SimpleNamespace(extra_attrs={}),
            forward=lambda **kwargs: method.apply(layer, inputs, None),
        ),
        mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
        dist=object(),
        is_draft_model=False,
        guided_decoder=None,
        _graphs=SimpleNamespace(no_cuda_graph=lambda: contextlib.nullcontext(), enabled=True),
        _dummy_requests=SimpleNamespace(
            create_warmup_request=lambda resources, num_tokens, num_gen_requests: Mock(
                num_gen_requests=num_gen_requests
            ),
            release_batch_context=lambda batch, resources: contextlib.nullcontext(batch),
        ),
        _agreement=SimpleNamespace(
            agree_flag=lambda flag: flag,
            should_run_warmup_batch=Mock(return_value=True),
        ),
        _release_megamoe_profiling_scratch=Mock(),
        _forward_warmup=Mock(),
    )
    engine._model_caller = ModelCaller(engine.model, prefill_compile_only=prefill_only)
    cache = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
    resources = SimpleNamespace(
        get_resource_manager=lambda key: cache
        if key == ResourceManagerType.KV_CACHE_MANAGER
        else None
    )
    tuner = Mock(profiling_cache={})
    monkeypatch.setattr(decoder_warmup_module.AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(
        decoder_warmup_module, "autotune", lambda **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(MXFP8GemmRunner, "sync_all_tactic_caches", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(decoder_warmup_module, "clear_memory_buffers", lambda: None)
    monkeypatch.setattr(model_call_module, "get_model_extra_attrs", lambda: {})
    monkeypatch.setattr(model_call_module, "is_trace_enabled", lambda name: False)

    def forward(batch: Mock, *args: object, **kwargs: object) -> torch.Tensor:
        # Stand in for input preparation, but use the real model-call compile scope
        # and linear dispatch for each prefill/generation warmup batch.
        monkeypatch.setattr(
            model_call_module,
            "get_per_request_prefill_cuda_graph_flag",
            lambda: batch.num_gen_requests == 0,
        )
        return DecoderRunner._model_forward(engine, is_dummy=True, attn_metadata=batch)

    engine._forward_warmup.side_effect = forward
    with torch_compiling(compile_enabled):
        DecoderWarmup._run_autotuner_warmup(engine, resources)
        assert is_torch_compiling() is compile_enabled

    expected_backend = backend or ("trtllm" if compile_mode == "all_batches" else "auto")
    flashinfer_expected = expected_backend == "flashinfer" or (
        expected_backend == "auto" and compile_mode != "all_batches"
    )
    assert method.backend == expected_backend
    assert method._native_autotuned
    assert method._flashinfer_autotuned == flashinfer_expected
    assert flashinfer_tune.call_count == int(flashinfer_expected)
    assert engine._forward_warmup.call_count == (4 if flashinfer_expected else 2)
    assert engine._forward_warmup.call_args.kwargs == {
        "enable_spec_decode": False,
        "runtime_draft_len": 0,
    }
    expected_flashinfer_calls = (
        (4 if expected_backend == "flashinfer" else (1 if prefill_only else 2))
        if flashinfer_expected
        else 0
    )
    assert flashinfer_gemm.call_count == expected_flashinfer_calls
    assert native_gemm.call_count == (engine._forward_warmup.call_count - expected_flashinfer_calls)
    assert os.environ.get("TRTLLM_MXFP8_GEMM_BACKEND") == backend


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("completed_num_tokens", "peers_skip", "expect_skip"),
    [
        (16, True, True),
        # A peer whose free KV capacity gives another token shape still runs
        # the warmup, so this rank must run it too.
        (16, False, False),
        (8, True, False),
        (None, True, False),
    ],
)
def test_autotuner_warmup_skips_only_completed_token_shape(
    monkeypatch: pytest.MonkeyPatch,
    completed_num_tokens: int | None,
    peers_skip: bool,
    expect_skip: bool,
) -> None:
    """A repeated warmup with an already tuned token shape must not re-run tuning."""
    engine = SimpleNamespace(
        _config=SimpleNamespace(
            enable_autotuner=True,
            torch_compile_enabled=False,
            torch_compile_prefill_only=False,
            max_num_tokens=16,
            max_batch_size=16,
            max_seq_len=2,
            original_max_draft_len=0,
            max_total_draft_tokens=0,
            is_spec_decode=False,
            max_draft_len=0,
            spec_config=None,
        ),
        _completed_autotuner_warmup_num_tokens=completed_num_tokens,
        _warmup_timer=_WarmupTimer(rank=0),
        model=SimpleNamespace(modules=lambda: []),
        mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
        dist=object(),
        guided_decoder=None,
        _graphs=SimpleNamespace(no_cuda_graph=lambda: contextlib.nullcontext(), enabled=False),
        _dummy_requests=SimpleNamespace(
            create_warmup_request=Mock(return_value=object()),
            release_batch_context=lambda batch, resources: contextlib.nullcontext(batch),
        ),
        _agreement=SimpleNamespace(
            agree_flag=Mock(side_effect=lambda flag: flag and peers_skip),
            should_run_warmup_batch=Mock(return_value=True),
        ),
        _release_megamoe_profiling_scratch=Mock(),
        _forward_warmup=Mock(),
    )
    cache = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
    resources = SimpleNamespace(
        get_resource_manager=lambda key: cache
        if key == ResourceManagerType.KV_CACHE_MANAGER
        else None
    )
    tuner = Mock(profiling_cache={})
    get_tuner = Mock(return_value=tuner)
    monkeypatch.delenv("TLLM_AUTOTUNER_CACHE_PATH", raising=False)
    monkeypatch.setattr(decoder_warmup_module.AutoTuner, "get", get_tuner)
    monkeypatch.setattr(
        decoder_warmup_module, "autotune", lambda **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    clear_buffers = Mock()
    monkeypatch.setattr(decoder_warmup_module, "clear_memory_buffers", clear_buffers)

    DecoderWarmup._run_autotuner_warmup(engine, resources)

    # Every rank votes with its local decision, skip or not.
    engine._agreement.agree_flag.assert_called_once_with(completed_num_tokens == 16)
    if expect_skip:
        get_tuner.assert_not_called()
        engine._dummy_requests.create_warmup_request.assert_not_called()
        engine._forward_warmup.assert_not_called()
    else:
        assert engine._forward_warmup.call_count == 2
    # Buffers cached by earlier warmup phases are released on both paths.
    clear_buffers.assert_called_once_with()
    assert engine._completed_autotuner_warmup_num_tokens == 16


@pytest.mark.parametrize("config_cls", [DraftTargetDecodingConfig, PARDDecodingConfig])
def test_draft_length_resolution_resyncs_dynamic_buffers(config_cls):
    config = config_cls(max_draft_len=3, speculative_model="dummy", draft_len_schedule={1: 3, 4: 2})
    engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=3,
        max_total_draft_tokens=config.tokens_per_gen_step - 1,
    )
    requests = [
        SimpleNamespace(
            py_draft_tokens=[7, 8, 9] + [0] * (config.tokens_per_gen_step - 4),
            py_needs_onehot_draft_probs=False,
        )
        for _ in range(2)
    ]
    batch = SimpleNamespace(batch_size=2, generation_requests=requests)

    def resolve(**kwargs):
        return resolve_draft_len(config, batch, max_draft_len=3, static_draft_len=3, **kwargs)

    # Normal iteration selects K=2.
    assert update_draft_len(engine, batch) == 2
    assert all(request.py_draft_tokens[:2] == [7, 8] for request in requests)
    # Each explicit warmup shape resynchronizes the buffers and returns its
    # length, even when the batch size is unchanged.
    for draft_len in (0, 3, 1):
        assert resolve(draft_len=draft_len) == draft_len
        assert all(
            len(request.py_draft_tokens) == config.get_runtime_tokens_per_gen_step(draft_len) - 1
            for request in requests
        )
    assert resolve() == 2
    assert resolve(speculation_permanently_disabled=True) == 0
    assert all(request.py_draft_tokens == [] for request in requests)


def test_static_draft_length_resolution_preserves_proposals():
    request = SimpleNamespace(py_draft_tokens=[7], py_needs_onehot_draft_probs=False)
    batch = SimpleNamespace(batch_size=1, generation_requests=[request])

    assert resolve_draft_len(None, batch, max_draft_len=3, static_draft_len=5) == 5
    assert request.py_draft_tokens == [7]
    assert not request.py_needs_onehot_draft_probs


# Minimal fixtures mirroring sibling test_pytorch_model_engine.py — duplicated
# rather than imported to keep this file self-contained and avoid sibling-test
# import fragility.
@dataclass
class _Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class _DummyModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=_Config(torch_dtype=dtype))

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs):
        # Never actually called in these tests (the warmup helpers that would
        # invoke forward are all patched), but must exist for engine init.
        batch_size = kwargs["input_ids"].size(0)
        return {"logits": torch.randn((batch_size, 10), device="cuda")}


class _DummyModelEngine(PyTorchModelEngine):
    def __init__(self, llm_args: TorchLlmArgs, dtype: torch.dtype):
        mapping = Mapping(
            world_size=tensorrt_llm.mpi_world_size(),
            tp_size=tensorrt_llm.mpi_world_size(),
            rank=tensorrt_llm.mpi_rank(),
        )
        super().__init__(
            model_path="dummy",
            mapping=mapping,
            model=_DummyModel(dtype),
            llm_args=llm_args,
        )


# KVCM2 requires a current CUDA context on whatever thread destroys it. A manager
# left for the cyclic GC is freed on whichever thread runs the collection -- under
# pytest's watchdog that thread has no context, and the first driver call in
# teardown fails with CUDA error 201, poisoning KVCM2 for the rest of the process.
# Releasing them here keeps teardown on the test's own thread.
_KV_CACHE_MANAGERS_TO_SHUTDOWN = []


@pytest.fixture(autouse=True)
def _shutdown_kv_cache_managers():
    yield
    while _KV_CACHE_MANAGERS_TO_SHUTDOWN:
        _KV_CACHE_MANAGERS_TO_SHUTDOWN.pop().shutdown()


def _build_engine_and_resource_manager():
    tokens_per_block = 1
    max_tokens = 258
    num_layers = 1
    batch_size = 13
    llm_args = TorchLlmArgs(
        model="dummy",
        max_batch_size=batch_size,
        max_num_tokens=max_tokens,
        kv_cache_config=KvCacheConfig(max_tokens=max_tokens, use_kv_cache_manager_v2=True),
        cuda_graph_config=CudaGraphConfig(
            enable_padding=True, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]
        ),
    )
    model_engine = _DummyModelEngine(llm_args, torch.half)
    kv_cache_manager = KVCacheManagerV2(
        llm_args.kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=model_engine.model.config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_tokens,
        max_batch_size=batch_size,
        max_num_tokens=max_tokens,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    _KV_CACHE_MANAGERS_TO_SHUTDOWN.append(kv_cache_manager)
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
    return model_engine, resource_manager


@pytest.mark.parametrize("config_cls", [DraftTargetDecodingConfig, PARDDecodingConfig])
@pytest.mark.parametrize(
    "draft_len", [None, 0, 3, 1], ids=["general", "cuda_graph_0", "cuda_graph_3", "cuda_graph_1"]
)
def test_warmup_builders_resynchronize_stale_draft_buffers(
    config_cls: type[DecodingBaseConfig], draft_len: int | None
) -> None:
    config = config_cls(max_draft_len=3, speculative_model="dummy", draft_len_schedule={1: 3, 4: 2})
    engine, resource_manager = _build_engine_and_resource_manager()
    runner = engine._runner
    runner._config = replace(
        runner._config,
        spec_config=config,
        max_draft_len=config.max_draft_len,
        max_total_draft_tokens=config.tokens_per_gen_step - 1,
        max_draft_loop_tokens=config.tokens_per_gen_step - 1,
    )
    runner._get_runtime_tokens_per_gen_step = config.get_runtime_tokens_per_gen_step
    requests = runner._dummy_requests
    requests._config = runner._config
    requests.get_runtime_tokens_per_gen_step = runner._get_runtime_tokens_per_gen_step
    # Profiling a batch of two leaves K=2; every requested warmup shape below
    # differs from that value and must synchronize the requests.
    batch_size = 2

    if draft_len is None:
        expected_draft_len = runner._config.max_draft_len
        warmup_request = requests.create_warmup_request(
            resource_manager,
            num_tokens=batch_size * config.tokens_per_gen_step,
            num_gen_requests=batch_size,
        )
    else:
        expected_draft_len = draft_len
        warmup_request = requests.create_cuda_graph_warmup_request(
            resource_manager, batch_size, draft_len, max_seq_len=16
        )

    with runner._dummy_requests.release_batch_context(warmup_request, resource_manager) as batch:
        assert batch is not None
        assert len(batch.generation_requests) == batch_size
        expected_buffer_width = config.get_runtime_tokens_per_gen_step(expected_draft_len) - 1
        assert all(
            len(request.py_draft_tokens) == expected_buffer_width
            for request in batch.generation_requests
        )


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_generation_capture_passes_local_speculation_state(fails: bool) -> None:
    engine, resource_manager = _build_engine_and_resource_manager()
    assert not engine.is_spec_decode
    runner = engine._runner
    batches = {}

    def create_request(resources, batch_size, draft_len, *args, **kwargs):
        return batches.setdefault(draft_len, ScheduledRequests())

    forward_warmup = Mock(side_effect=RuntimeError("capture failure") if fails else None)
    with (
        patch.object(runner._graphs, "_get_graphs_to_capture", return_value=[(1, 0), (1, 2)]),
        patch.object(
            runner._dummy_requests, "create_cuda_graph_warmup_request", side_effect=create_request
        ),
        patch.object(
            runner._dummy_requests,
            "release_batch_context",
            side_effect=lambda request, resources: contextlib.nullcontext(request),
        ),
        patch.object(runner._graphs, "_forward_warmup", forward_warmup),
    ):
        if fails:
            with pytest.raises(RuntimeError, match="capture failure"):
                runner._graphs._capture_generation_cuda_graphs(resource_manager)
        else:
            runner._graphs._capture_generation_cuda_graphs(resource_manager)

    expected_calls = [
        call(batches[2], resource_manager, enable_spec_decode=True, runtime_draft_len=2)
    ]
    if not fails:
        expected_calls.append(
            call(batches[0], resource_manager, enable_spec_decode=False, runtime_draft_len=0)
        )
    assert forward_warmup.call_args_list == expected_calls
    assert runner._graphs.force_lora_graph_for_capture is None


@pytest.mark.parametrize("phase", ["general", "attention", "mamba", "prefill"])
def test_warmup_phases_pass_configured_speculation_state(phase: str) -> None:
    engine, resource_manager = _build_engine_and_resource_manager()
    runner = engine._runner
    spec_config = DraftTargetDecodingConfig(max_draft_len=3, speculative_model="dummy")
    batch = ScheduledRequests()
    forward_warmup = Mock()
    kv_cache_manager = resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)

    with contextlib.ExitStack() as stack:
        spec_decode_config = replace(runner._config, spec_config=spec_config, is_spec_decode=True)
        for owner in (runner._warmup, runner._graphs):
            stack.enter_context(patch.object(owner, "_config", spec_decode_config))
        stack.enter_context(
            patch.object(runner._dummy_requests, "create_warmup_request", Mock(return_value=batch))
        )
        stack.enter_context(
            patch.object(
                runner._dummy_requests,
                "release_batch_context",
                side_effect=lambda request, resources: contextlib.nullcontext(request),
            )
        )
        for owner in (runner._warmup, runner._graphs):
            stack.enter_context(patch.object(owner, "_forward_warmup", forward_warmup))
        if phase == "general":
            runner._warmup._general_warmup_impl(resource_manager, [(4, 0)])
        elif phase == "attention":
            runner._warmup._run_attention_warmup(resource_manager)
        elif phase == "mamba":
            stack.enter_context(
                patch.object(
                    decoder_warmup_module, "MambaHybridCacheManager", type(kv_cache_manager)
                )
            )
            stack.enter_context(
                patch.object(kv_cache_manager, "get_num_available_tokens", return_value=8)
            )
            stack.enter_context(
                patch.object(
                    runner._warmup,
                    "_config",
                    replace(runner._warmup._config, enable_autotuner=False),
                )
            )
            runner._warmup._run_mamba_hybrid_warmup(resource_manager)
        else:
            stack.enter_context(
                patch.object(
                    runner._graphs,
                    "_config",
                    replace(
                        runner._graphs._config,
                        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
                        prefill_cuda_graph_num_tokens=[4],
                    ),
                )
            )
            runner._graphs._capture_prefill_cuda_graphs(resource_manager)

    jit_warmup = {"trtllm_gen_jit_warmup": True} if phase == "attention" else {}
    assert forward_warmup.call_count > 0
    assert (
        forward_warmup.call_args_list
        == [
            call(
                batch, resource_manager, enable_spec_decode=True, runtime_draft_len=3, **jit_warmup
            )
        ]
        * forward_warmup.call_count
    )


class _Tracker:
    """Records method-call order (and each call's arguments) via mock side_effects."""

    def __init__(self):
        self.calls = []
        self.call_args = {}

    def __call__(self, name):
        def _wrapped(*args, **kwargs):
            self.calls.append(name)
            self.call_args.setdefault(name, []).append((args, kwargs))

        return _wrapped


def _run_warmup_tracked(
    model_engine, resource_manager, *, force_helix_cp=False, capture_logs=False, tracker=None
):
    """Patch warmup stages and cleanup operations, then run warmup.

    Optionally force helix CP and capture logs. Pass ``tracker`` to inspect the
    recorded call arguments. Returns (call_order_list, log_records_or_None).
    """
    if tracker is None:
        tracker = _Tracker()
    # Engines left by earlier tests are cyclic; finalize them before tracking.
    gc.collect()
    # Observe this module's explicit cleanup without counting unrelated
    # destructors triggered by gc.collect() or lazy imports.
    tracked_cuda = SimpleNamespace(**vars(torch.cuda))
    tracked_cuda.empty_cache = tracker("empty_cache")
    tracked_torch = SimpleNamespace(**vars(torch))
    tracked_torch.cuda = tracked_cuda
    helix_ctx = (
        patch.object(model_engine.mapping, "has_cp_helix", return_value=True)
        if force_helix_cp
        else contextlib.nullcontext()
    )

    with (
        helix_ctx,
        patch.object(
            model_engine._runner._warmup, "_general_warmup", side_effect=tracker("general_warmup")
        ),
        patch.object(
            model_engine._runner._warmup, "_run_autotuner_warmup", side_effect=tracker("autotuner")
        ),
        patch.object(
            model_engine._runner._graphs,
            "_capture_generation_cuda_graphs",
            side_effect=tracker("generation_cuda_graph"),
        ),
        patch.object(
            model_engine._runner._graphs,
            "_capture_additional_cuda_graphs",
            side_effect=tracker("additional_cuda_graph"),
        ),
        patch.object(
            model_engine._runner._graphs,
            "_capture_prefill_cuda_graphs",
            side_effect=tracker("prefill_cuda_graph"),
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.warmup_sampling_module",
            side_effect=tracker("sampling_warmup"),
        ),
        patch.object(decoder_warmup_module, "torch", tracked_torch),
        patch(
            "tensorrt_llm._torch.custom_ops.torch_custom_ops.MoERunner.clear_all_workspaces",
            side_effect=tracker("moe_clear"),
        ),
    ):
        if capture_logs:
            with _capture_tllm_logs() as logs:
                model_engine.warmup(resource_manager)
            return tracker.calls, logs
        model_engine.warmup(resource_manager)
        return tracker.calls, None


@contextlib.contextmanager
def _capture_tllm_logs():
    """Capture logger.info calls emitted from the decoder runner module.

    tensorrt_llm.logger.logger is a custom Singleton (not a stdlib
    logging.Logger) and does not route through stdlib logging by default,
    so a logging.Handler attached to logging.getLogger("tensorrt_llm")
    sees nothing. Patch the logger.info bound on the decoder runner module
    directly so we observe exactly the messages warmup() emits.
    """
    records = []

    def _record(*msg):
        records.append(" ".join(str(m) for m in msg))

    with patch.object(decoder_warmup_module.logger, "info", side_effect=_record):
        yield records


@pytest.mark.cpu_only
@pytest.mark.parametrize("schedule", [{4: 4, 8: 2, 16: 1}, {4: 2}, {16: 4}, {4: 0}])
def test_dynamic_warmup_graphs_match_replay_schedule(schedule: dict[int, int]) -> None:
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import (
        compute_dynamic_draft_len_mapping,
    )
    from tensorrt_llm._torch.speculative.utils import get_draft_len_for_batch_size

    batch_sizes = [1, 2, 4, 8, 16]
    runner = object.__new__(DecoderCudaGraphs)
    mode = SimpleNamespace(support_dynamic_draft_len=lambda: True)
    runner._config = SimpleNamespace(
        spec_config=SimpleNamespace(draft_len_schedule=schedule, spec_dec_mode=mode),
        cuda_graph_batch_sizes=batch_sizes,
        original_max_draft_len=max(schedule.values()),
    )
    runner._dynamic_draft_len_mapping = compute_dynamic_draft_len_mapping(runner._config)
    graphs = runner._get_graphs_to_capture(batch_sizes)
    replay = object.__new__(CUDAGraphRunner)
    replay.supported_batch_sizes = batch_sizes
    replay.dynamic_draft_len_mapping = runner._dynamic_draft_len_mapping

    reachable_graphs = set()
    for batch_size in range(1, max(batch_sizes) + 1):
        draft_len = get_draft_len_for_batch_size(schedule, batch_size, max(schedule.values()))
        padded_size = replay._round_up_batch_size_with_draft_len(batch_size, draft_len)
        assert padded_size >= batch_size
        reachable_graphs.add((padded_size, draft_len))

    assert set(graphs) == reachable_graphs
    assert len(graphs) == len(reachable_graphs)


class TestWarmupCleanup(unittest.TestCase):
    """Lock in warmup-cleanup behavior introduced by PR #14609 (Plan B)."""

    @pytest.mark.cpu_only
    def test_no_kv_cache_warmup_delegates_runner_lifecycle(self):
        """Verify no-KV-cache warmup delegates warmup and graph capture to the runner."""
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.llm_args = SimpleNamespace(enable_autotuner=False)
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine._metrics = {}
        model_engine._warmup_depth = 0
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None

        model_engine.warmup(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup(resource_manager)],
        )

    @pytest.mark.cpu_only
    def test_no_kv_cache_warmup_rejects_allocated_kv_cache(self):
        """Verify a no-KV-cache runner rejects an allocated KV cache before warmup."""
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.llm_args = SimpleNamespace(enable_autotuner=False)
        model_engine._metrics = {}
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine._warmup_depth = 0
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        runner = model_engine._runner
        runner._validate_resources = NoKVCacheRunner._validate_resources.__get__(runner)
        runner.warmup.side_effect = NoKVCacheRunner.warmup.__get__(runner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = object()

        with self.assertRaisesRegex(
            AssertionError,
            "no-KV-cache runner was initialized, but a KV cache manager was allocated",
        ):
            model_engine.warmup(resource_manager)

        model_engine._runner.warmup.assert_called_once_with(resource_manager)

    @pytest.mark.cpu_only
    def test_decoder_runner_warmup_sampling_and_kv_cache_cleanup(self) -> None:
        """Verify sampling warmup coverage and KV-cache cleanup timing within total warmup."""
        runner = object.__new__(DecoderRunner)
        warmup = runner._warmup = object.__new__(DecoderWarmup)
        warmup._warmup_timer = _WarmupTimer(rank=0)
        warmup.model = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
        warmup._config = SimpleNamespace(
            dtype=torch.float16, cuda_graph_batch_sizes=[1, 4], enable_in_graph_sampling=False
        )
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine._metrics = warmup._metrics = {}
        model_engine.llm_args = SimpleNamespace(enable_autotuner=False)
        model_engine.moe_load_balancer = None
        model_engine._warmup_depth = 0
        model_engine._runner = runner
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None
        events = []
        phase = warmup._warmup_timer.phase

        @contextlib.contextmanager
        def record_metric_scope(
            name: str, *, metrics: dict[str, float], metric_name: str
        ) -> Iterator[None]:
            events.append(("enter", metric_name))
            with phase(name, metrics=metrics, metric_name=metric_name):
                yield
            events.append(("exit", metric_name))

        with (
            patch.object(
                warmup._warmup_timer,
                "phase",
                side_effect=record_metric_scope,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.warmup_sampling_module",
                side_effect=lambda: events.append("sampling"),
            ) as warmup_sampling,
            _capture_tllm_logs() as logs,
        ):
            for enabled in (False, True):
                with self.subTest(enable_in_graph_sampling=enabled):
                    events.clear()
                    warmup_sampling.reset_mock()
                    warmup._config.enable_in_graph_sampling = enabled
                    runner._eager_workspace_reclaimer = object()
                    model_engine._warmup_impl(resource_manager)
                    self.assertEqual(
                        events,
                        [
                            ("enter", "sampling_warmup_seconds"),
                            "sampling",
                            ("exit", "sampling_warmup_seconds"),
                        ],
                    )
                    self.assertIsNone(runner._eager_workspace_reclaimer)
                    warmup_sampling.assert_called_once_with()

        self.assertTrue(
            any("Skipping warm up as no KV Cache manager allocated." in log for log in logs)
        )
        self.assertGreaterEqual(model_engine.metrics["sampling_warmup_seconds"], 0)

        kv_cache_manager = Mock()
        kv_cache_manager.check_invalid_values_in_kv_cache.return_value = False
        resource_manager.get_resource_manager.return_value = kv_cache_manager
        model_engine._metrics.clear()
        with (
            patch.object(model_engine, "_warmup_impl") as warmup_impl,
            patch("time.perf_counter", side_effect=[0.0, 2.0, 5.0, 7.0]),
        ):
            model_engine.warmup(resource_manager)

        warmup_impl.assert_called_once_with(resource_manager)
        kv_cache_manager.check_invalid_values_in_kv_cache.assert_called_once_with(
            fill_with_zero=True
        )
        self.assertEqual(
            model_engine.metrics,
            {"kv_cache_cleanup_seconds": 3.0, "total_warmup_seconds": 7.0},
        )

    def test_encoder_decoder_encoder_warmup_delegates_runner_lifecycle(self):
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=True))
        model_engine.moe_load_balancer = None
        model_engine._warmup_depth = 0
        model_engine._runner = Mock(spec=EncoderDecoderRunner)
        warmup_depths = []
        model_engine._runner.warmup_encoder.side_effect = lambda _: warmup_depths.append(
            model_engine._warmup_depth
        )
        resource_manager = object()

        model_engine.warmup_encoder(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup_encoder(resource_manager)],
        )
        self.assertEqual(warmup_depths, [1])
        self.assertEqual(model_engine._warmup_depth, 0)

    @pytest.mark.cpu_only
    def test_cuda_graph_metrics_exclude_piecewise_stages(self) -> None:
        """Verify generation graph timing excludes prefill and includes LoRA warmup cleanup."""
        graphs = DecoderCudaGraphs(
            cuda_graph_runner=SimpleNamespace(enabled=True, is_warmup_only=False),
            breakable_cuda_graph_runner=None,
            config=SimpleNamespace(
                enable_autotuner=True, prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE
            ),
            mapping=None,
            model=SimpleNamespace(modules=lambda: []),
            sparse_attention_config=None,
            metrics={},
            dummy_requests=None,
            agreement=None,
            metadata=None,
            get_runtime_tokens_per_gen_step=None,
            dynamic_draft_len_mapping=None,
            lora=SimpleNamespace(cuda_graph_manager=object()),
            forward_warmup=None,
            torch_compile_bypass=contextlib.nullcontext,
            context_promotion_supported=True,
        )
        resource_manager = ResourceManager({})
        events = []

        @contextlib.contextmanager
        def record_metric_scope(metric_name: str, _metrics: dict[str, float]) -> Iterator[None]:
            events.append(("enter", metric_name))
            try:
                yield
            finally:
                events.append(("exit", metric_name))

        autotuner = SimpleNamespace(
            cache_pp_recv=lambda: events.append(("lora", "cache_pp_recv")),
            cache_pp_send=lambda: events.append(("lora", "cache_pp_send")),
            clean_pp_flag=lambda: events.append(("lora", "clean_pp_flag")),
        )
        lora_cleanup_events = [
            ("lora", "cache_pp_recv"),
            ("lora", "cache_pp_send"),
            ("lora", "clean_pp_flag"),
            ("exit", "lora_autotune"),
            ("exit", "gen_cuda_graph_warmup_seconds"),
        ]

        with (
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs.autotune",
                side_effect=lambda **_: record_metric_scope("lora_autotune", {}),
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs.AutoTuner.get",
                return_value=autotuner,
            ),
            patch(
                "tensorrt_llm._torch.modules.linear.flashinfer_mxfp8_decode_graph_capture",
                side_effect=contextlib.nullcontext,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs.timing_metric",
                side_effect=record_metric_scope,
            ),
            patch.object(
                graphs,
                "_capture_generation_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "generation")),
            ) as generation,
            patch.object(
                graphs,
                "_capture_additional_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "additional")),
            ),
            patch.object(
                graphs,
                "_capture_prefill_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "piecewise")),
            ) as piecewise,
        ):
            graphs.run_cuda_graph_warmup(resource_manager)

            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_capture_seconds"),
                    ("stage", "generation"),
                    ("stage", "additional"),
                    ("exit", "gen_cuda_graph_capture_seconds"),
                    ("stage", "piecewise"),
                ],
            )

            events.clear()
            piecewise.reset_mock()
            graphs.cuda_graph_runner.is_warmup_only = True
            graphs.run_cuda_graph_warmup(resource_manager)

            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    ("stage", "generation"),
                    ("stage", "additional"),
                    *lora_cleanup_events,
                ],
            )
            piecewise.assert_not_called()

            events.clear()
            generation.side_effect = RuntimeError("warmup failed")
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                graphs.run_cuda_graph_warmup(resource_manager)
            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    *lora_cleanup_events,
                ],
            )
            piecewise.assert_not_called()

            events.clear()
            generation.reset_mock()
            graphs.cuda_graph_runner.enabled = False
            graphs._config.prefill_cuda_graph_backend = PrefillCudaGraphBackend.DISABLED
            graphs.run_cuda_graph_warmup(resource_manager)
            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    *lora_cleanup_events,
                ],
            )
            generation.assert_not_called()
            piecewise.assert_not_called()

    def test_empty_cache_fires_immediately_after_autotuner(self):
        """Change 1 placement: empty_cache must be the call right after
        _run_autotuner_warmup."""
        model_engine, resource_manager = _build_engine_and_resource_manager()
        calls, _ = _run_warmup_tracked(model_engine, resource_manager)

        self.assertIn("autotuner", calls)
        autotuner_idx = calls.index("autotuner")
        self.assertLess(
            autotuner_idx + 1,
            len(calls),
            f"Expected something after autotuner; got {calls}",
        )
        self.assertEqual(
            calls[autotuner_idx + 1],
            "empty_cache",
            f"Expected empty_cache right after autotuner; got {calls}",
        )

    def test_empty_cache_count_under_default(self):
        """Default warmup should call empty_cache exactly twice:
        once at the end of step (a) (pre-existing) and once after step (b)
        (Change 1)."""
        model_engine, resource_manager = _build_engine_and_resource_manager()
        calls, _ = _run_warmup_tracked(model_engine, resource_manager)
        self.assertEqual(
            calls.count("empty_cache"),
            2,
            f"Expected exactly 2 empty_cache calls; got order={calls}",
        )

    def test_helix_cp_runs_only_the_generation_autotuner_warmup(self):
        """With Helix CP, can_run_general_warmup is False and the context
        warmups (general, Mamba hybrid) are skipped; the autotuner warmup still
        runs, on generation requests, followed by one empty_cache."""
        model_engine, resource_manager = _build_engine_and_resource_manager()
        tracker = _Tracker()
        calls, _ = _run_warmup_tracked(
            model_engine, resource_manager, force_helix_cp=True, tracker=tracker
        )
        self.assertEqual(
            tracker.call_args["autotuner"],
            [((resource_manager,), {"generation_only": True})],
            "Helix CP must tune on generation requests only",
        )
        self.assertNotIn("general_warmup", calls, f"Helix CP should skip general; got {calls}")
        self.assertIn("autotuner", calls, f"Helix CP should run the autotuner; got {calls}")
        self.assertEqual(
            calls.count("empty_cache"),
            1,
            f"Helix CP should clean up once, after the autotuner; got {calls}",
        )
        self.assertEqual(calls[calls.index("autotuner") + 1], "empty_cache")

    @pytest.mark.cpu_only
    def test_generation_only_autotuner_warmup_tunes_one_full_generation_batch(self):
        """Helix CP tunes on max_batch_size generation requests (bounded by
        max_num_tokens), without the context batch."""
        tuner = SimpleNamespace(
            setup_distributed_state=Mock(),
            cache_pp_recv=Mock(),
            cache_pp_send=Mock(),
            clean_pp_flag=Mock(),
            profiling_cache={},
            print_profiling_cache=Mock(),
        )
        get_num_available_tokens = Mock(return_value=4096)
        engine = SimpleNamespace(
            _warmup_timer=_WarmupTimer(rank=0),
            _config=SimpleNamespace(
                enable_autotuner=True,
                torch_compile_enabled=False,
                max_num_tokens=100,
                max_batch_size=32,
                max_seq_len=1024,
                original_max_draft_len=3,
                max_total_draft_tokens=3,
                is_spec_decode=True,
                max_draft_len=3,
                spec_config=None,
            ),
            _graphs=SimpleNamespace(no_cuda_graph=lambda: contextlib.nullcontext(), enabled=False),
            model=SimpleNamespace(modules=lambda: []),
            mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
            dist=object(),
            guided_decoder=None,
            _dummy_requests=SimpleNamespace(
                create_warmup_request=Mock(return_value=object()),
                release_batch_context=Mock(return_value=contextlib.nullcontext(object())),
            ),
            _agreement=SimpleNamespace(
                agree_flag=lambda flag: flag,
                should_run_warmup_batch=Mock(return_value=True),
            ),
            _release_megamoe_profiling_scratch=Mock(),
            _forward_warmup=Mock(),
            _completed_autotuner_warmup_num_tokens=None,
        )
        kv_cache_manager = SimpleNamespace(get_num_available_tokens=get_num_available_tokens)
        resource_manager = SimpleNamespace(
            get_resource_manager=lambda key: kv_cache_manager
            if key == ResourceManagerType.KV_CACHE_MANAGER
            else None
        )

        with (
            patch.object(decoder_warmup_module.AutoTuner, "get", return_value=tuner),
            patch.object(decoder_warmup_module, "autotune", return_value=contextlib.nullcontext()),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch.object(decoder_warmup_module, "clear_memory_buffers"),
        ):
            DecoderWarmup._run_autotuner_warmup(engine, resource_manager, generation_only=True)

        # 100 tokens fit 25 requests of 1 + 3 tokens, below max_batch_size.
        engine._dummy_requests.create_warmup_request.assert_called_once_with(
            resource_manager, 100, 25
        )
        engine._forward_warmup.assert_called_once()
        get_num_available_tokens.assert_not_called()
        self.assertEqual(engine._completed_autotuner_warmup_num_tokens, 100)

    @pytest.mark.cpu_only
    def test_flashinfer_mxfp8_respects_disabled_global_autotuner(self):
        """The global autotuner switch also disables automatic FlashInfer tuning."""
        calls = []

        @contextlib.contextmanager
        def flashinfer_autotune():
            calls.append("flashinfer_autotune_enter")
            yield
            calls.append("flashinfer_autotune_exit")

        flashinfer_module = ModuleType("flashinfer")
        flashinfer_module.mm_mxfp8 = Mock()
        flashinfer_module.autotune = Mock(side_effect=flashinfer_autotune)

        with (
            patch.dict(
                sys.modules,
                {
                    "flashinfer": flashinfer_module,
                },
            ),
            patch(
                "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available",
                return_value=True,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TRTLLM_MXFP8_GEMM_BACKEND", None)
            os.environ.pop("TLLM_AUTOTUNER_CACHE_PATH", None)
            method = MXFP8LinearMethod()
            self.assertEqual(method.backend, "trtllm")

            engine = SimpleNamespace(
                _warmup_timer=_WarmupTimer(rank=0),
                _config=SimpleNamespace(enable_autotuner=False, torch_compile_enabled=False),
                _graphs=SimpleNamespace(enabled=True),
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
            )
            DecoderWarmup._run_autotuner_warmup(engine, Mock())

        self.assertEqual(calls, [])
        self.assertEqual(method.backend, "trtllm")
        self.assertFalse(method.use_native_autotuner)
        self.assertFalse(method._flashinfer_autotuned)
        flashinfer_module.autotune.assert_not_called()

    @pytest.mark.cpu_only
    def test_mxfp8_native_and_flashinfer_use_separate_warmup_passes(self):
        """Native and FlashInfer backends each receive an isolated tuning forward."""
        calls = []

        @contextlib.contextmanager
        def trtllm_autotune(**kwargs):
            self.assertIsNone(kwargs["cache_path"])
            calls.append("trtllm_autotune_enter")
            yield
            calls.append("trtllm_autotune_exit")

        @contextlib.contextmanager
        def flashinfer_autotune():
            calls.append("flashinfer_autotune_enter")
            yield
            calls.append("flashinfer_autotune_exit")

        flashinfer_module = ModuleType("flashinfer")
        flashinfer_module.mm_mxfp8 = Mock()
        flashinfer_module.autotune = Mock(side_effect=flashinfer_autotune)

        tuner = SimpleNamespace(
            setup_distributed_state=Mock(),
            cache_pp_recv=Mock(),
            cache_pp_send=Mock(),
            clean_pp_flag=Mock(),
            profiling_cache={},
            print_profiling_cache=Mock(),
        )

        with (
            patch.dict(sys.modules, {"flashinfer": flashinfer_module}),
            patch(
                "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available",
                return_value=True,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TRTLLM_MXFP8_GEMM_BACKEND", None)
            os.environ.pop("TLLM_AUTOTUNER_CACHE_PATH", None)
            method = MXFP8LinearMethod()
            self.assertFalse(method.needs_native_autotune)

            engine = SimpleNamespace(
                _warmup_timer=_WarmupTimer(rank=0),
                _config=SimpleNamespace(
                    enable_autotuner=True,
                    torch_compile_enabled=False,
                    max_num_tokens=16,
                    max_batch_size=16,
                    max_seq_len=2,
                    original_max_draft_len=0,
                    max_total_draft_tokens=0,
                    is_spec_decode=False,
                    max_draft_len=0,
                    spec_config=None,
                ),
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
                mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
                dist=object(),
                guided_decoder=None,
                _graphs=SimpleNamespace(
                    no_cuda_graph=lambda: contextlib.nullcontext(), enabled=True
                ),
                _dummy_requests=SimpleNamespace(
                    create_warmup_request=Mock(return_value=object()),
                    release_batch_context=Mock(
                        side_effect=[
                            contextlib.nullcontext(object()),
                            contextlib.nullcontext(object()),
                            contextlib.nullcontext(object()),
                            contextlib.nullcontext(object()),
                        ]
                    ),
                ),
                _completed_autotuner_warmup_num_tokens=None,
                _agreement=SimpleNamespace(
                    agree_flag=lambda flag: flag,
                    should_run_warmup_batch=Mock(return_value=True),
                ),
                _release_megamoe_profiling_scratch=Mock(),
                _forward_warmup=Mock(side_effect=lambda *args, **kwargs: calls.append("forward")),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: kv_cache_manager
                if key == ResourceManagerType.KV_CACHE_MANAGER
                else None
            )

            with (
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.AutoTuner.get",
                    return_value=tuner,
                ),
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.autotune",
                    side_effect=trtllm_autotune,
                ),
                patch.object(MXFP8GemmRunner, "sync_all_tactic_caches") as sync_tactics,
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.empty_cache"),
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.clear_memory_buffers"
                ),
            ):
                DecoderWarmup._run_autotuner_warmup(engine, resource_manager)

        self.assertEqual(
            calls,
            [
                "trtllm_autotune_enter",
                "forward",
                "forward",
                "trtllm_autotune_exit",
                "flashinfer_autotune_enter",
                "forward",
                "forward",
                "flashinfer_autotune_exit",
            ],
        )
        self.assertTrue(method._native_autotuned)
        self.assertFalse(method.needs_native_autotune)
        self.assertEqual(method.backend, "auto")
        self.assertTrue(method._flashinfer_autotuned)
        sync_tactics.assert_called_once_with(tuner)
        self.assertEqual(tuner.setup_distributed_state.call_count, 1)
        tuner.setup_distributed_state.assert_called_with(engine.mapping, engine.dist)

    @pytest.mark.cpu_only
    def test_native_mxfp8_falls_back_after_missing_warmup_batch(self):
        """A missing startup batch latches native MXFP8 to the default tactic."""
        calls = []

        @contextlib.contextmanager
        def trtllm_autotune(**kwargs):
            self.assertIsNone(kwargs["cache_path"])
            calls.append("autotune_enter")
            yield
            calls.append("autotune_exit")

        @contextlib.contextmanager
        def flashinfer_autotune():
            calls.append("flashinfer_autotune_enter")
            yield
            calls.append("flashinfer_autotune_exit")

        flashinfer_module = ModuleType("flashinfer")
        flashinfer_module.mm_mxfp8 = Mock()
        flashinfer_module.autotune = Mock(side_effect=flashinfer_autotune)

        tuner = SimpleNamespace(
            setup_distributed_state=Mock(),
            profiling_cache={},
            print_profiling_cache=Mock(),
        )

        with (
            patch.dict(sys.modules, {"flashinfer": flashinfer_module}),
            patch(
                "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available",
                return_value=True,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TRTLLM_MXFP8_GEMM_BACKEND", None)
            os.environ.pop("TLLM_AUTOTUNER_CACHE_PATH", None)
            method = MXFP8LinearMethod()
            engine = SimpleNamespace(
                _warmup_timer=_WarmupTimer(rank=0),
                _config=SimpleNamespace(
                    enable_autotuner=True,
                    torch_compile_enabled=False,
                    max_num_tokens=16,
                    max_batch_size=16,
                    max_seq_len=2,
                    original_max_draft_len=0,
                    max_total_draft_tokens=0,
                    is_spec_decode=False,
                    max_draft_len=0,
                    spec_config=None,
                ),
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
                mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
                dist=object(),
                guided_decoder=None,
                _graphs=SimpleNamespace(
                    no_cuda_graph=lambda: contextlib.nullcontext(), enabled=True
                ),
                _dummy_requests=SimpleNamespace(
                    create_warmup_request=Mock(return_value=object()),
                    release_batch_context=Mock(
                        side_effect=[
                            contextlib.nullcontext(None),
                            contextlib.nullcontext(None),
                            contextlib.nullcontext(None),
                            contextlib.nullcontext(None),
                        ]
                    ),
                ),
                _completed_autotuner_warmup_num_tokens=None,
                _agreement=SimpleNamespace(
                    agree_flag=lambda flag: flag,
                    should_run_warmup_batch=Mock(return_value=False),
                ),
                _release_megamoe_profiling_scratch=Mock(),
                _forward_warmup=Mock(),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: kv_cache_manager
                if key == ResourceManagerType.KV_CACHE_MANAGER
                else None
            )

            with (
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.AutoTuner.get",
                    return_value=tuner,
                ),
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.autotune",
                    side_effect=trtllm_autotune,
                ),
                patch.object(MXFP8GemmRunner, "sync_all_tactic_caches") as sync_tactics,
                patch("torch.cuda.empty_cache"),
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.clear_memory_buffers"
                ),
            ):
                DecoderWarmup._run_autotuner_warmup(engine, resource_manager)

        self.assertEqual(
            calls,
            [
                "autotune_enter",
                "autotune_exit",
                "flashinfer_autotune_enter",
                "flashinfer_autotune_exit",
            ],
        )
        self.assertFalse(method._native_autotuned)
        self.assertFalse(method.use_native_autotuner)
        self.assertFalse(method.needs_native_autotune)
        self.assertEqual(method.backend, "trtllm")
        self.assertFalse(method._flashinfer_autotuned)
        sync_tactics.assert_not_called()
        engine._forward_warmup.assert_not_called()

    @pytest.mark.cpu_only
    def test_flashinfer_mxfp8_rank_mismatch_falls_back_before_warmup(self):
        """TP and PP ranks agree on fallback before the tuning forward."""
        flashinfer_module = ModuleType("flashinfer")
        flashinfer_module.mm_mxfp8 = Mock()
        flashinfer_module.autotune = Mock(return_value=contextlib.nullcontext())
        tuner = SimpleNamespace(
            setup_distributed_state=Mock(),
            cache_pp_recv=Mock(),
            cache_pp_send=Mock(),
            clean_pp_flag=Mock(),
            profiling_cache={},
            print_profiling_cache=Mock(),
        )
        dist = SimpleNamespace(
            tp_allgather=Mock(return_value=[1, 1]),
            pp_allgather=Mock(return_value=[[1, 1], [1, 0]]),
        )

        with (
            patch.dict(sys.modules, {"flashinfer": flashinfer_module}),
            patch(
                "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available",
                return_value=True,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TRTLLM_MXFP8_GEMM_BACKEND", None)
            method = MXFP8LinearMethod()
            engine = SimpleNamespace(
                _warmup_timer=_WarmupTimer(rank=0),
                _config=SimpleNamespace(
                    enable_autotuner=True,
                    torch_compile_enabled=False,
                    max_num_tokens=16,
                    max_batch_size=16,
                    max_seq_len=2,
                    original_max_draft_len=0,
                    max_total_draft_tokens=0,
                    is_spec_decode=False,
                    max_draft_len=0,
                    spec_config=None,
                ),
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
                mapping=SimpleNamespace(tp_size=2, has_pp=lambda: True),
                dist=dist,
                guided_decoder=None,
                _graphs=SimpleNamespace(
                    no_cuda_graph=lambda: contextlib.nullcontext(), enabled=True
                ),
                _dummy_requests=SimpleNamespace(
                    create_warmup_request=Mock(return_value=object()),
                    release_batch_context=Mock(return_value=contextlib.nullcontext(object())),
                ),
                _completed_autotuner_warmup_num_tokens=None,
                _agreement=SimpleNamespace(
                    agree_flag=lambda flag: flag,
                    should_run_warmup_batch=Mock(return_value=True),
                ),
                _release_megamoe_profiling_scratch=Mock(),
                _forward_warmup=Mock(),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: kv_cache_manager
                if key == ResourceManagerType.KV_CACHE_MANAGER
                else None
            )

            with (
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.AutoTuner.get",
                    return_value=tuner,
                ),
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.autotune",
                    return_value=contextlib.nullcontext(),
                ),
                patch.object(MXFP8GemmRunner, "sync_all_tactic_caches") as sync_tactics,
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.empty_cache"),
                patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup.clear_memory_buffers"
                ),
            ):
                DecoderWarmup._run_autotuner_warmup(engine, resource_manager)

        dist.tp_allgather.assert_called_once_with(1)
        dist.pp_allgather.assert_called_once_with([1, 1])
        self.assertEqual(method.backend, "trtllm")
        self.assertTrue(method._native_autotuned)
        self.assertFalse(method._flashinfer_autotuned)
        sync_tactics.assert_called_once_with(tuner)
        flashinfer_module.autotune.assert_not_called()
        self.assertEqual(engine._forward_warmup.call_count, 1)

    @pytest.mark.cpu_only
    def test_native_mxfp8_respects_disabled_global_autotuner(self):
        """Avoid native MXFP8 warmup when the global autotuner is disabled."""
        with (
            patch(
                "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available",
                return_value=True,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TRTLLM_MXFP8_GEMM_BACKEND", None)
            method = MXFP8LinearMethod()
            engine = SimpleNamespace(
                _warmup_timer=_WarmupTimer(rank=0),
                _config=SimpleNamespace(enable_autotuner=False, torch_compile_enabled=False),
                _graphs=SimpleNamespace(enabled=False),
                model=SimpleNamespace(modules=lambda: [SimpleNamespace(quant_method=method)]),
            )

            DecoderWarmup._run_autotuner_warmup(engine, Mock())

        self.assertFalse(method.use_native_autotuner)
        self.assertFalse(method.needs_native_autotune)


def _indexer_q_prewarm_owner(modules: list, tp_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(modules=lambda: iter(modules)),
        mapping=SimpleNamespace(tp_size=tp_size),
        dist=Mock(),
    )


def _deepseek_v4_indexer(
    q_lora_rank: int = 256, with_wq_b: bool = True, fused_enabled: bool = True
) -> Mock:
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4Indexer

    indexer = Mock(spec=DeepseekV4Indexer)
    indexer.wq_b = (
        SimpleNamespace(weight=torch.empty((64, q_lora_rank), dtype=torch.bfloat16))
        if with_wq_b
        else None
    )
    indexer._is_fused_project_mxfp4_enabled.return_value = fused_enabled
    return indexer


def _other_pp_stage_indexer() -> Mock:
    """Layers owned by another pipeline stage keep wq_b but lose its weight."""
    indexer = _deepseek_v4_indexer()
    indexer.wq_b = torch.nn.Linear(256, 64, bias=False)
    remove_weights(indexer.wq_b)
    return indexer


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp_size", [1, 2])
def test_indexer_q_prewarm_runs_every_fallback_tactic(tp_size: int) -> None:
    indexer = _deepseek_v4_indexer(q_lora_rank=256)
    owner = _indexer_q_prewarm_owner([torch.nn.Linear(2, 2), indexer], tp_size)
    events = []
    indexer._project_and_quantize_q.side_effect = lambda *args: events.append("compile")
    owner.dist.tp_allgather.side_effect = lambda *args: events.append("barrier")

    with patch.object(torch.cuda, "synchronize", side_effect=lambda: events.append("synchronize")):
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    calls = indexer._project_and_quantize_q.call_args_list
    assert [c.args[0].shape for c in calls] == [(4, 256), (8, 256), (16, 256)]
    assert [c.args[1].shape for c in calls] == [(4,), (8,), (16,)]
    # The fused path only accepts bf16 activations.
    assert all(c.args[0].dtype == torch.bfloat16 for c in calls)
    assert all(c.args[1].dtype == torch.int32 for c in calls)
    indexer._is_fused_project_mxfp4_enabled.assert_called_once_with(torch.bfloat16)
    # The barrier is only useful once every compile on this rank has finished.
    expected_barrier = ["barrier"] if tp_size > 1 else []
    assert events == ["compile"] * 3 + ["synchronize"] + expected_barrier
    if tp_size > 1:
        owner.dist.tp_allgather.assert_called_once_with(1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case", ["no_indexer", "indexer_without_wq_b", "indexer_of_other_pp_stage"]
)
def test_indexer_q_prewarm_is_noop_without_local_dsv4_indexer(case: str) -> None:
    if case == "indexer_of_other_pp_stage":
        indexer = _other_pp_stage_indexer()
    else:
        indexer = _deepseek_v4_indexer(with_wq_b=False)
    model_modules = [torch.nn.Linear(2, 2)]
    if case != "no_indexer":
        model_modules.append(indexer)
    owner = _indexer_q_prewarm_owner(model_modules, tp_size=2)

    with patch.object(torch.cuda, "synchronize") as synchronize:
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    indexer._project_and_quantize_q.assert_not_called()
    synchronize.assert_not_called()
    owner.dist.tp_allgather.assert_not_called()


@pytest.mark.cpu_only
def test_indexer_q_prewarm_without_fused_path_compiles_nothing_but_reaches_barrier() -> None:
    """The fused-path check is rank-local, so it must not decide the TP barrier."""
    indexer = _deepseek_v4_indexer(fused_enabled=False)
    owner = _indexer_q_prewarm_owner([indexer], tp_size=2)

    with (
        patch.object(torch.cuda, "synchronize") as synchronize,
        patch.object(decoder_warmup_module.logger, "info") as info,
    ):
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    indexer._project_and_quantize_q.assert_not_called()
    synchronize.assert_not_called()
    info.assert_not_called()
    owner.dist.tp_allgather.assert_called_once_with(1)


@pytest.mark.cpu_only
def test_indexer_q_prewarm_compiles_first_fused_indexer() -> None:
    unfused_indexer = _deepseek_v4_indexer(fused_enabled=False)
    fused_indexer = _deepseek_v4_indexer()
    later_fused_indexer = _deepseek_v4_indexer()
    owner = _indexer_q_prewarm_owner(
        [unfused_indexer, fused_indexer, later_fused_indexer], tp_size=2
    )

    with patch.object(torch.cuda, "synchronize"):
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    unfused_indexer._project_and_quantize_q.assert_not_called()
    assert fused_indexer._project_and_quantize_q.call_count == 3
    later_fused_indexer._project_and_quantize_q.assert_not_called()
    owner.dist.tp_allgather.assert_called_once_with(1)


@pytest.mark.cpu_only
def test_indexer_q_prewarm_failure_still_reaches_barrier() -> None:
    """A per-shape failure must not skip the TP barrier the other ranks wait on."""
    indexer = _deepseek_v4_indexer()
    indexer._project_and_quantize_q.side_effect = [RuntimeError("compile"), None, None]
    owner = _indexer_q_prewarm_owner([indexer], tp_size=2)

    with (
        patch.object(torch.cuda, "synchronize"),
        patch.object(decoder_warmup_module.logger, "warning") as warning,
    ):
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    assert indexer._project_and_quantize_q.call_count == 3
    warning.assert_called_once()
    owner.dist.tp_allgather.assert_called_once_with(1)


@pytest.mark.cpu_only
def test_indexer_q_prewarm_does_not_report_completion_when_every_compile_fails() -> None:
    indexer = _deepseek_v4_indexer()
    indexer._project_and_quantize_q.side_effect = RuntimeError("compile")
    owner = _indexer_q_prewarm_owner([indexer], tp_size=2)

    with (
        patch.object(torch.cuda, "synchronize"),
        patch.object(decoder_warmup_module.logger, "warning") as warning,
        patch.object(decoder_warmup_module.logger, "info") as info,
    ):
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    assert warning.call_count == 3
    info.assert_not_called()
    owner.dist.tp_allgather.assert_called_once_with(1)


@pytest.mark.cpu_only
def test_indexer_q_prewarm_skips_indexers_of_other_pp_stages() -> None:
    remote_indexer = _other_pp_stage_indexer()
    local_indexer = _deepseek_v4_indexer()
    owner = _indexer_q_prewarm_owner([remote_indexer, local_indexer], tp_size=2)

    with patch.object(torch.cuda, "synchronize"):
        DecoderWarmup._prewarm_cute_dsl_indexer_q(owner)

    remote_indexer._project_and_quantize_q.assert_not_called()
    assert local_indexer._project_and_quantize_q.call_count == 3
    owner.dist.tp_allgather.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
