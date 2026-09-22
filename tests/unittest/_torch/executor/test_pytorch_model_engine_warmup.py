# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for warmup-cleanup behavior in PyTorchModelEngine.warmup().

Locks in that gc.collect() + torch.cuda.empty_cache() fire immediately after
_run_autotuner_warmup (step b) to release autotuner exploration leftovers.

The torch.cuda.empty_cache() after teardown_managers() in py_executor_creator
is covered end-to-end by integration tests rather than unit-tested here.
"""

import contextlib
import os
import sys
import unittest
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.pyexecutor.model_engine as model_engine_module
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, timing_metric
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine, _PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.speculative.utils import update_draft_len
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
def test_pcg_fx_fallback_policy_is_model_specific(
    monkeypatch: pytest.MonkeyPatch,
    model_kind: str,
    compile_enabled: bool,
    piecewise: bool,
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
        enable_fullgraph=True, enable_inductor=False, enable_userbuffers=False, max_num_streams=1
    )
    llm_args = SimpleNamespace(
        encode_only=False,
        mm_encoder_only=False,
        get_runtime_sizes=lambda: (1, 16, 64, 4),
        encoder_max_batch_size=None,
        encoder_max_num_tokens=None,
        enable_in_graph_sampling=False,
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

    expected_prefill_only = compile_enabled and piecewise and model_kind in ("m3", "m3_vl")
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
@pytest.mark.parametrize("raises", [False, True])
def test_prefill_compile_scopes_whole_model_forward(
    monkeypatch: pytest.MonkeyPatch,
    prefill_only: bool,
    eligible: bool,
    raises: bool,
) -> None:
    """Keep compile state active through model epilogues and restore it on exit."""
    observed = []

    def forward(**kwargs: object) -> str:
        """Record compile state as a stand-in for a model-specific epilogue."""
        observed.append(is_torch_compiling())
        # This represents work after the transformer, such as Eagle3 drafting.
        if raises:
            raise RuntimeError("epilogue failure")
        observed.append(is_torch_compiling())
        return "done"

    engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(extra_attrs={}), forward=forward),
        _torch_compile_backend=None,
        _torch_compile_prefill_only=prefill_only,
        _eager_workspace_reclaimer=None,
        is_warmup=False,
    )
    monkeypatch.setattr(model_engine_module, "get_model_extra_attrs", lambda: {})
    monkeypatch.setattr(
        model_engine_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
    )
    monkeypatch.setattr(model_engine_module, "is_trace_enabled", lambda name: False)
    with torch_compiling(True):
        if raises:
            with pytest.raises(RuntimeError, match="epilogue failure"):
                PyTorchModelEngine.model_forward(engine, attn_metadata=Mock())
        else:
            assert PyTorchModelEngine.model_forward(engine, attn_metadata=Mock()) == "done"
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
        is_warmup=True,
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
        forward=Mock(),
    )
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
    monkeypatch.setattr(model_engine_module, "get_model_extra_attrs", lambda: {})
    monkeypatch.setattr(model_engine_module, "is_trace_enabled", lambda name: False)

    def forward(batch: Mock, **kwargs: object) -> torch.Tensor:
        # Stand in for input preparation, but use the real engine compile scope
        # and linear dispatch for each prefill/generation warmup batch.
        monkeypatch.setattr(
            model_engine_module,
            "get_per_request_prefill_cuda_graph_flag",
            lambda: batch.num_gen_requests == 0,
        )
        return PyTorchModelEngine.model_forward(engine, attn_metadata=batch)

    engine.forward.side_effect = forward
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
    assert engine.forward.call_count == (4 if flashinfer_expected else 2)
    expected_flashinfer_calls = (
        (4 if expected_backend == "flashinfer" else (1 if prefill_only else 2))
        if flashinfer_expected
        else 0
    )
    assert flashinfer_gemm.call_count == expected_flashinfer_calls
    assert native_gemm.call_count == engine.forward.call_count - expected_flashinfer_calls
    assert os.environ.get("TRTLLM_MXFP8_GEMM_BACKEND") == backend


@pytest.mark.parametrize("config_cls", [DraftTargetDecodingConfig, PARDDecodingConfig])
def test_warmup_overrides_dynamic_draft_length(config_cls):
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

    # Normal iteration selects K=2. Each explicit warmup shape must then
    # update both the engine and buffers, even when the batch size is unchanged.
    update_draft_len(engine, batch)
    assert engine.runtime_draft_len == 2
    assert all(request.py_draft_tokens[:2] == [7, 8] for request in requests)
    for draft_len in (0, 3, 1):
        update_draft_len(engine, batch, draft_len=draft_len)
        assert engine.runtime_draft_len == draft_len
        assert all(
            len(request.py_draft_tokens) == config.get_runtime_tokens_per_gen_step(draft_len) - 1
            for request in requests
        )
    update_draft_len(engine, batch)
    assert engine.runtime_draft_len == 2


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
def test_warmup_builders_resynchronize_stale_draft_length(
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
    # differs from that value and must synchronize the real engine and requests.
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
        assert engine.runtime_draft_len == expected_draft_len
        expected_buffer_width = config.get_runtime_tokens_per_gen_step(expected_draft_len) - 1
        assert all(
            len(request.py_draft_tokens) == expected_buffer_width
            for request in batch.generation_requests
        )


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

    def test_no_kv_cache_warmup_delegates_runner_lifecycle(self):
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine._metrics = {}
        model_engine.is_warmup = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None

        with patch(
            "tensorrt_llm._torch.pyexecutor.model_engine.warmup_sampling_module"
        ) as warmup_sampling:
            model_engine.warmup(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup(resource_manager), call.capture_graphs(resource_manager)],
        )
        warmup_sampling.assert_not_called()

    def test_no_kv_cache_warmup_rejects_allocated_kv_cache(self):
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine._metrics = {}
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine.is_warmup = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = object()

        with self.assertRaisesRegex(
            AssertionError,
            "no-KV-cache runner was initialized, but a KV cache manager was allocated",
        ):
            model_engine.warmup(resource_manager)

        self.assertEqual(model_engine._runner.method_calls, [])

    @pytest.mark.cpu_only
    def test_legacy_warmup_skips_without_kv_cache(self) -> None:
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine._warmup_timer = _WarmupTimer(rank=0)
        model_engine._metrics = {}
        model_engine.moe_load_balancer = None
        model_engine.is_warmup = False
        model_engine.enable_in_graph_sampling = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._runner = None
        model_engine.model = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
        model_engine.dtype = torch.float16
        model_engine._cuda_graph_batch_sizes = [1, 4]
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None
        events = []

        @contextlib.contextmanager
        def record_metric_scope(name: str, metrics: dict[str, float]) -> Iterator[None]:
            events.append(("enter", name))
            with timing_metric(name, metrics):
                yield
            events.append(("exit", name))

        with (
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.timing_metric",
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
            [call.warmup(resource_manager), call.capture_graphs(resource_manager)],
        )

    @pytest.mark.cpu_only
    def test_cuda_graph_metrics_exclude_piecewise_stages(self) -> None:
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.model = SimpleNamespace(modules=lambda: [])
        model_engine.llm_args = SimpleNamespace(enable_autotuner=True)
        model_engine.cuda_graph_lora_manager = object()
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

    @pytest.mark.cpu_only
    def test_piecewise_cuda_graph_metrics_are_recorded_separately(self) -> None:
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.prefill_cuda_graph_backend = PrefillCudaGraphBackend.PIECEWISE
        model_engine._torch_compile_enabled = True
        model_engine._torch_compile_piecewise_cuda_graph = True
        model_engine._prefill_cuda_graph_num_tokens = [4, 8]
        model_engine._metrics = defaultdict(float)
        resource_manager = object()
        events = []
        scopes = []
        warmup_metric = "ctx_cuda_graph_warmup_seconds"
        capture_metric = "ctx_cuda_graph_capture_seconds"
        post_metric = "post_ctx_cuda_graph_capture_warmup_seconds"

        @contextlib.contextmanager
        def record_metric_scope(name: str, metrics: dict[str, float]) -> Iterator[None]:
            scopes.append(name)
            try:
                with timing_metric(name, metrics):
                    yield
            finally:
                scopes.pop()

        def forward(batch: tuple[int, bool], **kwargs: object) -> torch.Tensor:
            events.append((batch, tuple(scopes)))
            return torch.empty(1)

        with (
            patch.object(model_engine, "no_cuda_graph", side_effect=contextlib.nullcontext),
            patch.object(
                model_engine,
                "_create_warmup_request",
                side_effect=lambda rm, tokens, gen, least_requests=True: (tokens, least_requests),
            ),
            patch.object(
                model_engine,
                "_release_batch_context",
                side_effect=lambda batch, rm: contextlib.nullcontext(batch),
            ),
            patch.object(model_engine, "_assert_all_tp_ranks_have_warmup_batch"),
            patch.object(model_engine, "forward", side_effect=forward),
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.timing_metric",
                side_effect=record_metric_scope,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner.timing_metric",
                side_effect=record_metric_scope,
            ),
            patch("tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner.BreakableCUDAGraph"),
            patch(
                "tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner.make_weak_ref",
                side_effect=lambda value: value,
            ),
            patch("torch.cuda.Stream"),
            patch("torch.cuda.current_stream"),
            patch("torch.cuda.stream", side_effect=lambda stream: contextlib.nullcontext()),
            patch("torch.cuda.graph_pool_handle", return_value=(1, 2)),
            patch(
                "torch.cuda.synchronize", side_effect=lambda: events.append(("sync", tuple(scopes)))
            ),
            patch("torch.cuda.empty_cache"),
            patch("gc.collect"),
            patch("time.perf_counter", side_effect=iter(range(100))),
        ):
            for backend in (PrefillCudaGraphBackend.PIECEWISE, PrefillCudaGraphBackend.BREAKABLE):
                with self.subTest(backend=backend):
                    events.clear()
                    model_engine._metrics.clear()
                    model_engine.prefill_cuda_graph_backend = backend
                    model_engine._torch_compile_piecewise_cuda_graph = (
                        backend == PrefillCudaGraphBackend.PIECEWISE
                    )
                    model_engine.breakable_cuda_graph_runner = (
                        BreakableCUDAGraphRunner(torch.nn.Identity())
                        if backend == PrefillCudaGraphBackend.BREAKABLE
                        else None
                    )
                    model_engine._capture_prefill_cuda_graphs(resource_manager)
                    warmup_steps = 3 if backend == PrefillCudaGraphBackend.PIECEWISE else 2
                    warmup_scope = (
                        BreakableCUDAGraphRunner.CUDA_GRAPH_WARMUP_METRIC
                        if backend == PrefillCudaGraphBackend.BREAKABLE
                        else warmup_metric
                    )
                    capture_scope = (
                        BreakableCUDAGraphRunner.CUDA_GRAPH_CAPTURE_METRIC
                        if backend == PrefillCudaGraphBackend.BREAKABLE
                        else capture_metric
                    )
                    expected = []
                    for tokens in (8, 4):
                        expected.extend([((tokens, True), (warmup_scope,))] * warmup_steps)
                        expected.extend(
                            [
                                ("sync", (warmup_scope,)),
                                ((tokens, True), (capture_scope,)),
                                ("sync", (capture_scope,)),
                            ]
                        )
                    for tokens in (8, 4):
                        expected.extend(
                            [((tokens, False), (post_metric,)), ("sync", (post_metric,))]
                        )
                    self.assertEqual(events, expected)
                    self.assertEqual(
                        model_engine.metrics.keys(), {warmup_metric, capture_metric, post_metric}
                    )
                    self.assertTrue(all(value >= 0 for value in model_engine.metrics.values()))
                    self.assertEqual(model_engine.metrics[warmup_metric], 2.0)
                    self.assertEqual(model_engine.metrics[capture_metric], 2.0)
                    if backend == PrefillCudaGraphBackend.BREAKABLE:
                        runner = model_engine.breakable_cuda_graph_runner
                        self.assertEqual(runner.metrics, {warmup_scope: 1.0, capture_scope: 1.0})
                        self.assertIsNot(runner.metrics, model_engine.metrics)

                        runner.clear()
                        model_engine._metrics.clear()
                        with patch.object(
                            model_engine, "forward", side_effect=RuntimeError("warmup failed")
                        ):
                            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                                model_engine._capture_prefill_cuda_graphs(resource_manager)
                        self.assertEqual(runner.metrics, {warmup_scope: 1.0})
                        self.assertEqual(model_engine.metrics, {})
                        self.assertFalse(runner.is_warming_up)
                        self.assertFalse(runner.is_capturing)

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
                is_draft_model=False,
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
                forward=Mock(side_effect=lambda *args, **kwargs: calls.append("forward")),
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
                is_draft_model=False,
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
                forward=Mock(),
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
        engine.forward.assert_not_called()

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
                is_draft_model=False,
                guided_decoder=None,
                max_total_draft_tokens=0,
                no_cuda_graph=lambda: contextlib.nullcontext(),
                _create_warmup_request=Mock(return_value=object()),
                _release_batch_context=Mock(return_value=contextlib.nullcontext(object())),
                _should_run_warmup_batch=Mock(return_value=True),
                _release_megamoe_profiling_scratch=Mock(),
                forward=Mock(),
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
        self.assertEqual(engine.forward.call_count, 1)

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
