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
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.pyexecutor.engine.model_call as model_call_module
import tensorrt_llm._torch.pyexecutor.model_engine as model_engine_module
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine, _PrefillCompiledModel
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
        "should_enable_adp_dummy_fixes",
        "should_enable_scheduler_aware_adp_dummy",
        "should_enable_non_overlap_adp_forward_intent",
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
            assert isinstance(model.model, _PrefillCompiledModel)
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
    router = _PrefillCompiledModel(eager, compiled)
    assert router.weight is eager.weight
    assert list(router.parameters()) == list(eager.parameters())
    # Local decode-only ranks participate when another attention-DP rank
    # prefills; an over-ceiling local context batch uses eager instead.
    metadata = SimpleNamespace(num_contexts=local_contexts)
    for eligible in (True, False):
        monkeypatch.setattr(
            model_engine_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
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
    model.model = _PrefillCompiledModel(eager, compiled)
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
            model_engine_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
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
    prefill_compiled_model = _PrefillCompiledModel(eager_model, compiled_model)
    model = prefill_compiled_model if prefill_only else eager_model
    engine = SimpleNamespace(
        _model_caller=ModelCaller(model, prefill_compile_only=prefill_only),
        _eager_workspace_reclaimer=None,
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
                PyTorchModelEngine.model_forward(engine, is_dummy=False, attn_metadata=Mock())
        else:
            assert (
                PyTorchModelEngine.model_forward(engine, is_dummy=False, attn_metadata=Mock())
                == "done"
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
        weight=torch.zeros(3, 4), weight_scale=torch.ones(4), dtype=torch.float32
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
        llm_args=SimpleNamespace(enable_autotuner=True),
        _torch_compile_enabled=compile_enabled,
        _torch_compile_prefill_only=prefill_only,
        _torch_compile_backend=None,
        _eager_workspace_reclaimer=None,
        _warmup_timer=_WarmupTimer(rank=0),
        cuda_graph_runner=SimpleNamespace(enabled=True),
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
        kv_cache_manager_key="kv_cache",
        max_num_tokens=16,
        batch_size=16,
        max_seq_len=2,
        original_max_draft_len=0,
        max_total_draft_tokens=0,
        is_draft_model=False,
        guided_decoder=None,
        no_cuda_graph=lambda: contextlib.nullcontext(),
        _create_warmup_request=lambda resources, num_tokens, num_gen_requests: Mock(
            num_gen_requests=num_gen_requests
        ),
        _release_batch_context=lambda batch, resources: contextlib.nullcontext(batch),
        _should_run_warmup_batch=Mock(return_value=True),
        _release_megamoe_profiling_scratch=Mock(),
        # The serving flag differs from the configuration; warmup follows the configuration.
        is_spec_decode=False,
        enable_spec_decode=True,
        spec_config=None,
        max_draft_len=0,
        _forward_warmup=Mock(),
    )
    engine._model_caller = ModelCaller(engine.model, prefill_compile_only=prefill_only)
    cache = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
    resources = SimpleNamespace(
        get_resource_manager=lambda key: cache if key == "kv_cache" else None
    )
    tuner = Mock(profiling_cache={})
    monkeypatch.setattr(model_engine_module.AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(model_engine_module, "autotune", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(MXFP8GemmRunner, "sync_all_tactic_caches", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(model_engine_module, "clear_memory_buffers", lambda: None)
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
        return PyTorchModelEngine.model_forward(engine, is_dummy=True, attn_metadata=batch)

    engine._forward_warmup.side_effect = forward
    with torch_compiling(compile_enabled):
        PyTorchModelEngine._run_autotuner_warmup(engine, resources)
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


@pytest.mark.parametrize("config_cls", [DraftTargetDecodingConfig, PARDDecodingConfig])
def test_draft_length_resolution_resyncs_dynamic_buffers(config_cls):
    config = config_cls(max_draft_len=3, speculative_model="dummy", draft_len_schedule={1: 3, 4: 2})
    engine = SimpleNamespace(
        spec_config=config,
        max_draft_len=3,
        max_total_draft_tokens=config.tokens_per_gen_step - 1,
        runtime_draft_len=3,
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

    # Normal iteration selects K=2 and stores it on the engine.
    update_draft_len(engine, batch)
    assert engine.runtime_draft_len == 2
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
    engine.spec_config = config
    engine.max_draft_len = config.max_draft_len
    engine.max_total_draft_tokens = config.tokens_per_gen_step - 1
    engine.max_draft_loop_tokens = engine.max_total_draft_tokens
    engine.get_runtime_tokens_per_gen_step = config.get_runtime_tokens_per_gen_step
    # Profiling a batch of two leaves K=2; every requested warmup shape below
    # differs from that value and must synchronize the requests, while the
    # engine keeps the serving value.
    engine.runtime_draft_len = 2
    batch_size = 2

    if draft_len is None:
        expected_draft_len = engine.max_draft_len
        warmup_request = engine._create_warmup_request(
            resource_manager,
            num_tokens=batch_size * config.tokens_per_gen_step,
            num_gen_requests=batch_size,
        )
    else:
        expected_draft_len = draft_len
        warmup_request = engine._create_cuda_graph_warmup_request(
            resource_manager, batch_size, draft_len, max_seq_len=16
        )

    with engine._release_batch_context(warmup_request, resource_manager) as batch:
        assert batch is not None
        assert len(batch.generation_requests) == batch_size
        assert engine.runtime_draft_len == 2
        expected_buffer_width = config.get_runtime_tokens_per_gen_step(expected_draft_len) - 1
        assert all(
            len(request.py_draft_tokens) == expected_buffer_width
            for request in batch.generation_requests
        )


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_generation_capture_passes_local_speculation_state(fails: bool) -> None:
    engine, resource_manager = _build_engine_and_resource_manager()
    assert not engine.is_spec_decode
    # Serving values that differ from every captured shape below.
    engine.enable_spec_decode = True
    engine.runtime_draft_len = 5
    batches = {}

    def create_request(resources, batch_size, draft_len, *args, **kwargs):
        return batches.setdefault(draft_len, ScheduledRequests())

    forward_warmup = Mock(side_effect=RuntimeError("capture failure") if fails else None)
    with (
        patch.object(engine, "_get_graphs_to_capture", return_value=[(1, 0), (1, 2)]),
        patch.object(engine, "_create_cuda_graph_warmup_request", side_effect=create_request),
        patch.object(
            engine,
            "_release_batch_context",
            side_effect=lambda request, resources: contextlib.nullcontext(request),
        ),
        patch.object(engine, "_forward_warmup", forward_warmup),
    ):
        if fails:
            with pytest.raises(RuntimeError, match="capture failure"):
                engine._capture_generation_cuda_graphs(resource_manager)
        else:
            engine._capture_generation_cuda_graphs(resource_manager)

    expected_calls = [
        call(batches[2], resource_manager, enable_spec_decode=True, runtime_draft_len=2)
    ]
    if not fails:
        expected_calls.append(
            call(batches[0], resource_manager, enable_spec_decode=False, runtime_draft_len=0)
        )
    assert forward_warmup.call_args_list == expected_calls
    assert (engine.enable_spec_decode, engine.runtime_draft_len) == (True, 5)
    assert engine._force_lora_graph_for_capture is None


@pytest.mark.parametrize("phase", ["general", "attention", "mamba", "prefill"])
def test_warmup_phases_pass_configured_speculation_state(phase: str) -> None:
    engine, resource_manager = _build_engine_and_resource_manager()
    assert not engine.is_spec_decode
    # Serving values that no warmup phase may read.
    engine.enable_spec_decode = True
    engine.runtime_draft_len = 5
    batch = ScheduledRequests()
    forward_warmup = Mock()
    kv_cache_manager = resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(engine, "_create_warmup_request", Mock(return_value=batch))
        )
        stack.enter_context(
            patch.object(
                engine,
                "_release_batch_context",
                side_effect=lambda request, resources: contextlib.nullcontext(request),
            )
        )
        stack.enter_context(patch.object(engine, "_forward_warmup", forward_warmup))
        if phase == "general":
            engine._general_warmup_impl(resource_manager, [(4, 0)])
        elif phase == "attention":
            engine._run_attention_warmup(resource_manager)
        elif phase == "mamba":
            stack.enter_context(
                patch.object(model_engine_module, "MambaHybridCacheManager", type(kv_cache_manager))
            )
            stack.enter_context(
                patch.object(kv_cache_manager, "get_num_available_tokens", return_value=8)
            )
            stack.enter_context(
                patch.object(engine, "llm_args", SimpleNamespace(enable_autotuner=False))
            )
            engine._run_mamba_hybrid_warmup(resource_manager)
        else:
            stack.enter_context(
                patch.object(
                    engine, "prefill_cuda_graph_backend", PrefillCudaGraphBackend.BREAKABLE
                )
            )
            stack.enter_context(patch.object(engine, "_prefill_cuda_graph_num_tokens", [4]))
            engine._capture_prefill_cuda_graphs(resource_manager)

    assert forward_warmup.call_count > 0
    assert (
        forward_warmup.call_args_list
        == [call(batch, resource_manager, enable_spec_decode=False, runtime_draft_len=0)]
        * forward_warmup.call_count
    )
    assert (engine.enable_spec_decode, engine.runtime_draft_len) == (True, 5)


class _Tracker:
    """Records method-call order via mock side_effects."""

    def __init__(self):
        self.calls = []

    def __call__(self, name):
        def _wrapped(*args, **kwargs):
            self.calls.append(name)

        return _wrapped


def _run_warmup_tracked(
    model_engine, resource_manager, *, force_helix_cp=False, capture_logs=False
):
    """Patch warmup stages and cleanup operations, then run warmup.

    Optionally force helix CP and capture logs. Returns
    (call_order_list, log_records_or_None).
    """
    tracker = _Tracker()
    # Engines left by earlier tests are cyclic; finalize them before tracking.
    gc.collect()
    helix_ctx = (
        patch.object(model_engine.mapping, "has_cp_helix", return_value=True)
        if force_helix_cp
        else contextlib.nullcontext()
    )

    with (
        helix_ctx,
        patch.object(model_engine, "_general_warmup", side_effect=tracker("general_warmup")),
        patch.object(model_engine, "_run_autotuner_warmup", side_effect=tracker("autotuner")),
        patch.object(
            model_engine,
            "_capture_generation_cuda_graphs",
            side_effect=tracker("generation_cuda_graph"),
        ),
        patch.object(
            model_engine,
            "_capture_mixed_encoder_decoder_cuda_graphs",
            side_effect=tracker("mixed_cuda_graph"),
        ),
        patch.object(
            model_engine,
            "_capture_prefill_cuda_graphs",
            side_effect=tracker("prefill_cuda_graph"),
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor.model_engine.warmup_sampling_module",
            side_effect=tracker("sampling_warmup"),
        ),
        patch("torch.cuda.empty_cache", side_effect=tracker("empty_cache")),
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
    """Capture logger.info calls emitted from the model_engine module.

    tensorrt_llm.logger.logger is a custom Singleton (not a stdlib
    logging.Logger) and does not route through stdlib logging by default,
    so a logging.Handler attached to logging.getLogger("tensorrt_llm")
    sees nothing. Patch the logger.info bound on the model_engine module
    directly so we observe exactly the messages warmup() emits.
    """
    from tensorrt_llm._torch.pyexecutor import model_engine as _me_mod

    records = []

    def _record(*msg):
        records.append(" ".join(str(m) for m in msg))

    with patch.object(_me_mod.logger, "info", side_effect=_record):
        yield records


class TestWarmupCleanup(unittest.TestCase):
    """Lock in warmup-cleanup behavior introduced by PR #14609 (Plan B)."""

    @pytest.mark.cpu_only
    def test_no_kv_cache_warmup_delegates_runner_lifecycle(self):
        """Verify no-KV-cache warmup delegates warmup and graph capture to the runner."""
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine._metrics = {}
        model_engine.is_warmup = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._fallback_to_engine = False
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None

        with patch(
            "tensorrt_llm._torch.pyexecutor.model_engine.warmup_sampling_module"
        ) as warmup_sampling:
            model_engine.warmup(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup(resource_manager)],
        )
        warmup_sampling.assert_not_called()

    @pytest.mark.cpu_only
    def test_no_kv_cache_warmup_rejects_allocated_kv_cache(self):
        """Verify a no-KV-cache runner rejects an allocated KV cache before warmup."""
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine._metrics = {}
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine.is_warmup = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._fallback_to_engine = False
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
    def test_legacy_warmup_sampling_and_kv_cache_cleanup(self) -> None:
        """Verify sampling warmup coverage and KV-cache cleanup timing within total warmup."""
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine._warmup_timer = _WarmupTimer(rank=0)
        model_engine._metrics = {}
        model_engine.moe_load_balancer = None
        model_engine.is_warmup = False
        model_engine.enable_in_graph_sampling = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._fallback_to_engine = True
        model_engine._runner = None
        model_engine.model = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
        model_engine.dtype = torch.float16
        model_engine._cuda_graph_batch_sizes = [1, 4]
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None
        events = []
        phase = model_engine._warmup_timer.phase

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
                model_engine._warmup_timer,
                "phase",
                side_effect=record_metric_scope,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.warmup_sampling_module",
                side_effect=lambda: events.append("sampling"),
            ) as warmup_sampling,
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.warmup_sample_from_logits_op",
                side_effect=lambda *args: events.append("in_graph_sampling"),
            ) as warmup_in_graph_sampling,
            _capture_tllm_logs() as logs,
        ):
            for enabled in (False, True):
                with self.subTest(enable_in_graph_sampling=enabled):
                    events.clear()
                    warmup_sampling.reset_mock()
                    warmup_in_graph_sampling.reset_mock()
                    model_engine.enable_in_graph_sampling = enabled
                    model_engine._eager_workspace_reclaimer = object()
                    model_engine._warmup_impl(resource_manager)
                    self.assertEqual(
                        events,
                        [("enter", "sampling_warmup_seconds"), "sampling"]
                        + (["in_graph_sampling"] if enabled else [])
                        + [("exit", "sampling_warmup_seconds")],
                    )
                    self.assertIsNone(model_engine._eager_workspace_reclaimer)
                    warmup_sampling.assert_called_once_with()
                    if enabled:
                        warmup_in_graph_sampling.assert_called_once_with(
                            128, torch.device("cuda"), torch.float16, [1, 4]
                        )
                    else:
                        warmup_in_graph_sampling.assert_not_called()

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
        model_engine.is_warmup = False
        model_engine._runner = Mock(spec=EncoderDecoderRunner)
        resource_manager = object()

        model_engine._warmup_encoder_cuda_graphs_enc_dec(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup(resource_manager)],
        )

    @pytest.mark.cpu_only
    def test_cuda_graph_metrics_exclude_piecewise_stages(self) -> None:
        """Verify generation graph timing excludes prefill and includes LoRA warmup cleanup."""
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.model = SimpleNamespace(modules=lambda: [])
        model_engine.llm_args = SimpleNamespace(enable_autotuner=True)
        model_engine._lora = SimpleNamespace(cuda_graph_manager=object())
        model_engine.cuda_graph_runner = SimpleNamespace(
            enabled=True,
            is_warmup_only=False,
        )
        model_engine.prefill_cuda_graph_backend = PrefillCudaGraphBackend.PIECEWISE
        model_engine._metrics = {}
        resource_manager = object()
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
                "tensorrt_llm._torch.pyexecutor.model_engine.autotune",
                side_effect=lambda **_: record_metric_scope("lora_autotune", {}),
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.AutoTuner.get",
                return_value=autotuner,
            ),
            patch(
                "tensorrt_llm._torch.modules.linear.flashinfer_mxfp8_decode_graph_capture",
                side_effect=contextlib.nullcontext,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.timing_metric",
                side_effect=record_metric_scope,
            ),
            patch.object(
                model_engine,
                "_capture_generation_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "generation")),
            ) as generation,
            patch.object(
                model_engine,
                "_capture_mixed_encoder_decoder_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "mixed")),
            ),
            patch.object(
                model_engine,
                "_capture_prefill_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "piecewise")),
            ) as piecewise,
        ):
            model_engine._run_cuda_graph_warmup(resource_manager)

            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_capture_seconds"),
                    ("stage", "generation"),
                    ("stage", "mixed"),
                    ("exit", "gen_cuda_graph_capture_seconds"),
                    ("stage", "piecewise"),
                ],
            )

            events.clear()
            piecewise.reset_mock()
            model_engine.cuda_graph_runner.is_warmup_only = True
            model_engine._run_cuda_graph_warmup(resource_manager)

            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    ("stage", "generation"),
                    ("stage", "mixed"),
                    *lora_cleanup_events,
                ],
            )
            piecewise.assert_not_called()

            events.clear()
            generation.side_effect = RuntimeError("warmup failed")
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                model_engine._run_cuda_graph_warmup(resource_manager)
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
            model_engine.cuda_graph_runner.enabled = False
            model_engine.prefill_cuda_graph_backend = PrefillCudaGraphBackend.DISABLED
            model_engine._run_cuda_graph_warmup(resource_manager)
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

    def test_step_b_cleanup_skipped_with_helix_cp(self):
        """With Helix CP, can_run_general_warmup is False AND step (b) is
        gated off -> no empty_cache calls inside warmup()."""
        model_engine, resource_manager = _build_engine_and_resource_manager()
        calls, _ = _run_warmup_tracked(model_engine, resource_manager, force_helix_cp=True)
        self.assertNotIn("autotuner", calls, f"Helix CP should skip autotuner; got {calls}")
        self.assertEqual(
            calls.count("empty_cache"),
            0,
            f"Helix CP should skip all warmup cleanup; got {calls}",
        )

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
                llm_args=SimpleNamespace(enable_autotuner=False),
                cuda_graph_runner=SimpleNamespace(enabled=True),
                _torch_compile_enabled=False,
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
            )
            PyTorchModelEngine._run_autotuner_warmup(engine, Mock())

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
                llm_args=SimpleNamespace(enable_autotuner=True),
                cuda_graph_runner=SimpleNamespace(enabled=True),
                _torch_compile_enabled=False,
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
                kv_cache_manager_key="kv_cache",
                max_num_tokens=16,
                batch_size=16,
                max_seq_len=2,
                original_max_draft_len=0,
                mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
                dist=object(),
                guided_decoder=None,
                max_total_draft_tokens=0,
                no_cuda_graph=lambda: contextlib.nullcontext(),
                _create_warmup_request=Mock(return_value=object()),
                _release_batch_context=Mock(
                    side_effect=[
                        contextlib.nullcontext(object()),
                        contextlib.nullcontext(object()),
                        contextlib.nullcontext(object()),
                        contextlib.nullcontext(object()),
                    ]
                ),
                _should_run_warmup_batch=Mock(return_value=True),
                _release_megamoe_profiling_scratch=Mock(),
                is_spec_decode=False,
                spec_config=None,
                max_draft_len=0,
                _forward_warmup=Mock(side_effect=lambda *args, **kwargs: calls.append("forward")),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: kv_cache_manager if key == "kv_cache" else None
            )

            with (
                patch(
                    "tensorrt_llm._torch.pyexecutor.model_engine.AutoTuner.get",
                    return_value=tuner,
                ),
                patch(
                    "tensorrt_llm._torch.pyexecutor.model_engine.autotune",
                    side_effect=trtllm_autotune,
                ),
                patch.object(MXFP8GemmRunner, "sync_all_tactic_caches") as sync_tactics,
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.empty_cache"),
                patch("tensorrt_llm._torch.pyexecutor.model_engine.clear_memory_buffers"),
            ):
                PyTorchModelEngine._run_autotuner_warmup(engine, resource_manager)

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
                llm_args=SimpleNamespace(enable_autotuner=True),
                cuda_graph_runner=SimpleNamespace(enabled=True),
                _torch_compile_enabled=False,
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
                kv_cache_manager_key="kv_cache",
                max_num_tokens=16,
                batch_size=16,
                max_seq_len=2,
                original_max_draft_len=0,
                mapping=SimpleNamespace(tp_size=1, has_pp=lambda: False),
                dist=object(),
                guided_decoder=None,
                max_total_draft_tokens=0,
                no_cuda_graph=lambda: contextlib.nullcontext(),
                _create_warmup_request=Mock(return_value=object()),
                _release_batch_context=Mock(
                    side_effect=[
                        contextlib.nullcontext(None),
                        contextlib.nullcontext(None),
                        contextlib.nullcontext(None),
                        contextlib.nullcontext(None),
                    ]
                ),
                _should_run_warmup_batch=Mock(return_value=False),
                _release_megamoe_profiling_scratch=Mock(),
                is_spec_decode=False,
                spec_config=None,
                max_draft_len=0,
                _forward_warmup=Mock(),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: kv_cache_manager if key == "kv_cache" else None
            )

            with (
                patch(
                    "tensorrt_llm._torch.pyexecutor.model_engine.AutoTuner.get",
                    return_value=tuner,
                ),
                patch(
                    "tensorrt_llm._torch.pyexecutor.model_engine.autotune",
                    side_effect=trtllm_autotune,
                ),
                patch.object(MXFP8GemmRunner, "sync_all_tactic_caches") as sync_tactics,
                patch("torch.cuda.empty_cache"),
                patch("tensorrt_llm._torch.pyexecutor.model_engine.clear_memory_buffers"),
            ):
                PyTorchModelEngine._run_autotuner_warmup(engine, resource_manager)

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
                llm_args=SimpleNamespace(enable_autotuner=True),
                cuda_graph_runner=SimpleNamespace(enabled=True),
                _torch_compile_enabled=False,
                model=SimpleNamespace(
                    modules=lambda: [
                        SimpleNamespace(_use_flashinfer_mxfp8_decode_graph_default=True),
                        SimpleNamespace(quant_method=method),
                    ]
                ),
                kv_cache_manager_key="kv_cache",
                max_num_tokens=16,
                batch_size=16,
                max_seq_len=2,
                original_max_draft_len=0,
                mapping=SimpleNamespace(tp_size=2, has_pp=lambda: True),
                dist=dist,
                guided_decoder=None,
                max_total_draft_tokens=0,
                no_cuda_graph=lambda: contextlib.nullcontext(),
                _create_warmup_request=Mock(return_value=object()),
                _release_batch_context=Mock(return_value=contextlib.nullcontext(object())),
                _should_run_warmup_batch=Mock(return_value=True),
                _release_megamoe_profiling_scratch=Mock(),
                is_spec_decode=False,
                spec_config=None,
                max_draft_len=0,
                _forward_warmup=Mock(),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: kv_cache_manager if key == "kv_cache" else None
            )

            with (
                patch(
                    "tensorrt_llm._torch.pyexecutor.model_engine.AutoTuner.get",
                    return_value=tuner,
                ),
                patch(
                    "tensorrt_llm._torch.pyexecutor.model_engine.autotune",
                    return_value=contextlib.nullcontext(),
                ),
                patch.object(MXFP8GemmRunner, "sync_all_tactic_caches") as sync_tactics,
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.empty_cache"),
                patch("tensorrt_llm._torch.pyexecutor.model_engine.clear_memory_buffers"),
            ):
                PyTorchModelEngine._run_autotuner_warmup(engine, resource_manager)

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
                llm_args=SimpleNamespace(enable_autotuner=False),
                cuda_graph_runner=SimpleNamespace(enabled=False),
                _torch_compile_enabled=False,
                model=SimpleNamespace(modules=lambda: [SimpleNamespace(quant_method=method)]),
            )

            PyTorchModelEngine._run_autotuner_warmup(engine, Mock())

        self.assertFalse(method.use_native_autotuner)
        self.assertFalse(method.needs_native_autotune)


if __name__ == "__main__":
    unittest.main()
