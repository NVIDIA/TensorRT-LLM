# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
import gc
import os
import sys
import unittest
from collections.abc import Iterator
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import Mock, call, patch

import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import (
    decoder_config,
    decoder_graphs,
    decoder_warmup,
)
from _torch.pyexecutor.engine_test_utils import (
    DummyModel,
    _agreement,
    _capture_tllm_logs,
    kv_cache_manager,
)

import tensorrt_llm._torch.pyexecutor.engine.model_call as model_call_module
import tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup as decoder_warmup_module
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner
from tensorrt_llm._torch.models.modeling_utils import remove_weights
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup as warmup_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import DecoderMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import DecoderWarmup
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.utils import is_torch_compiling, torch_compiling
from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, PrefillCudaGraphBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.config import DecoderRunnerConfig


@contextlib.contextmanager
def _warmup_and_resources() -> Iterator[tuple[DecoderWarmup, ResourceManager]]:
    config = decoder_config()
    model = DummyModel(torch.half)
    backend = Mock(enabled=True, is_warmup_only=False)
    backend.allow_capture.side_effect = contextlib.nullcontext
    graphs = decoder_graphs(config, model=model, cuda_graph_runner=backend)
    warmup = decoder_warmup(config, model=model, graphs=graphs)
    with kv_cache_manager(model=model, max_num_tokens=config.max_num_tokens) as manager:
        yield warmup, ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})


class _Tracker:
    """Records method-call order (and each call's arguments) via mock side_effects."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.call_args: dict[str, list[tuple[tuple[object, ...], dict[str, object]]]] = {}

    def __call__(self, name: str) -> Callable[..., None]:
        def _wrapped(*args: object, **kwargs: object) -> None:
            self.calls.append(name)
            self.call_args.setdefault(name, []).append((args, kwargs))

        return _wrapped


def _run_warmup_tracked(
    model_engine: DecoderWarmup,
    resource_manager: ResourceManager,
    *,
    force_helix_cp: bool = False,
    capture_logs: bool = False,
    tracker: _Tracker | None = None,
) -> tuple[list[str], list[str] | None]:
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
        patch.object(model_engine, "_general_warmup", side_effect=tracker("general_warmup")),
        patch.object(model_engine, "_run_autotuner_warmup", side_effect=tracker("autotuner")),
        patch.object(
            model_engine._graphs,
            "_capture_generation_cuda_graphs",
            side_effect=tracker("generation_cuda_graph"),
        ),
        patch.object(
            model_engine._graphs,
            "_capture_additional_cuda_graphs",
            side_effect=tracker("additional_cuda_graph"),
        ),
        patch.object(
            model_engine._graphs,
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
                model_engine.run(resource_manager)
            return tracker.calls, logs
        model_engine.run(resource_manager)
        return tracker.calls, None


def _indexer_q_prewarm_owner(
    modules: list[torch.nn.Module | SimpleNamespace | Mock], tp_size: int
) -> SimpleNamespace:
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


class _StandInMambaCacheManager:
    """Concrete stand-in for the abstract Mamba cache-manager type check."""

    def __init__(self, available_tokens: int = 8) -> None:
        self._available_tokens = available_tokens

    def get_num_available_tokens(self, **_kwargs: object) -> int:
        return self._available_tokens


def _engine(
    *, world_size: int = 1, dwdp_size: int = 0, config: DecoderRunnerConfig | None = None
) -> DecoderWarmup:
    agreement = _agreement(world_size=world_size, dwdp_size=dwdp_size)
    engine = decoder_warmup(
        config or decoder_config(), agreement=agreement, mapping=agreement.mapping
    )
    engine._reset_moe_alltoall_state = mock.Mock()
    return engine


@contextlib.contextmanager
def _released_batch(batch: object) -> Iterator[object]:
    yield batch


@contextlib.contextmanager
def _no_cuda_side_effects() -> Iterator[tuple[mock.Mock, mock.Mock]]:
    with (
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        yield empty_cache, synchronize


def _general_warmup_engine(*, world_size: int, dwdp_size: int) -> DecoderWarmup:
    engine = _engine(world_size=world_size, dwdp_size=dwdp_size)
    engine._warmup_timer = _WarmupTimer(rank=engine.dist.rank)
    batch = object()
    engine._dummy_requests = SimpleNamespace(
        create_warmup_request=mock.Mock(return_value=batch),
        release_batch_context=lambda *_a, **_kw: _released_batch(batch),
    )
    return engine


_KV_ALLOC_ERROR = "Can't allocate new blocks for window size 8"


def _mamba_engine(*, world_size: int = 1, dwdp_size: int = 0) -> tuple[DecoderWarmup, object]:
    engine = _engine(
        world_size=world_size,
        dwdp_size=dwdp_size,
        config=decoder_config(
            max_num_tokens=8, max_batch_size=4, max_seq_len=8, enable_autotuner=False
        ),
    )
    engine._graphs = SimpleNamespace(no_cuda_graph=contextlib.nullcontext)
    # The warmup resolves its chunk-alignment variant off the model's Mamba
    # metadata class; a model declaring none takes the ``Mamba2Metadata``
    # default, i.e. no alignment rewrite.
    engine.model = SimpleNamespace()
    batch = object()
    engine._dummy_requests = SimpleNamespace(
        release_batch_context=lambda *_a, **_kw: _released_batch(batch)
    )

    cache_manager = _StandInMambaCacheManager()
    resource_manager = SimpleNamespace(
        get_resource_manager=lambda key: (
            cache_manager if key == ResourceManagerType.KV_CACHE_MANAGER else None
        )
    )
    return engine, resource_manager


def _run_mamba_warmup(engine: DecoderWarmup, resource_manager: object) -> None:
    with (
        mock.patch.object(warmup_module, "MambaHybridCacheManager", _StandInMambaCacheManager),
        mock.patch.object(
            warmup_module.Mamba2Metadata,
            "force_initial_states_for_warmup",
            side_effect=contextlib.nullcontext,
        ),
        mock.patch.object(warmup_module, "clear_memory_buffers"),
    ):
        engine._run_mamba_hybrid_warmup(resource_manager)


class DecoderDsaWarmupTestCase(unittest.TestCase):
    def test_warmup_builds_dsa_attn_metadata_when_no_forward_ran(self) -> None:
        """The DSA top-K pre-compile hooks read attn_metadata; when every
        warmup forward was skipped it is built on demand so the engines still
        compile before serving."""
        from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata

        engine = object.__new__(DecoderWarmup)
        engine._config = SimpleNamespace(
            attention_backend=SimpleNamespace(Metadata=DSAtrtllmAttentionMetadata),
            original_max_draft_len=2,
            cuda_graph_batch_sizes=[8],
            spec_config=None,
        )
        engine._metadata = DecoderMetadata(
            config=engine._config,
            mapping=None,
            model=None,
            buffers=None,
            attention_runtime_features=None,
        )
        metadata = Mock(spec=DSAtrtllmAttentionMetadata)

        def build(
            kv_cache_manager: KVCacheManagerV2, draft_kv_cache_manager: KVCacheManagerV2 | None
        ) -> Mock:
            engine._metadata.attn_metadata = metadata
            return metadata

        engine._metadata.setup_attention = Mock(side_effect=build)
        kv_cache_manager = Mock()
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = kv_cache_manager

        engine._ensure_dsa_attn_metadata_for_warmup(resource_manager)
        engine._warmup_cute_dsl_radix_topk()

        engine._metadata.setup_attention.assert_called_once_with(kv_cache_manager, None)
        metadata.warmup_cute_dsl_radix_topk.assert_called_once_with(3)
        metadata.warmup_selfsampling_topk.assert_called_once_with(3, batch_sizes=[8])

    def test_warmup_does_not_build_metadata_for_non_dsa_backend(self) -> None:
        engine = object.__new__(DecoderWarmup)
        engine._config = SimpleNamespace(
            attention_backend=SimpleNamespace(Metadata=TrtllmAttentionMetadata)
        )
        engine._metadata = DecoderMetadata(
            config=engine._config,
            mapping=None,
            model=None,
            buffers=None,
            attention_runtime_features=None,
        )
        engine._metadata.setup_attention = Mock()

        engine._ensure_dsa_attn_metadata_for_warmup(Mock())
        engine._warmup_cute_dsl_radix_topk()

        engine._metadata.setup_attention.assert_not_called()


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
        runtime_hooks=DEFAULT_RUNTIME_HOOKS,
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
        get_resource_manager=lambda key: (
            cache if key == ResourceManagerType.KV_CACHE_MANAGER else None
        )
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
        runtime_hooks=DEFAULT_RUNTIME_HOOKS,
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
        get_resource_manager=lambda key: (
            cache if key == ResourceManagerType.KV_CACHE_MANAGER else None
        )
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


@pytest.mark.parametrize("phase", ["general", "attention", "mamba", "prefill"])
def test_warmup_phases_pass_configured_speculation_state(phase: str) -> None:
    spec = DraftTargetDecodingConfig(max_draft_len=3, speculative_model="dummy")
    config = decoder_config(
        spec_config=spec,
        enable_autotuner=phase != "mamba",
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE
        if phase == "prefill"
        else PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[4] if phase == "prefill" else [],
    )
    batch = ScheduledRequests()
    forward_warmup = Mock()
    requests = Mock()
    requests.create_warmup_request.return_value = batch
    requests.release_batch_context.side_effect = lambda request, resources: contextlib.nullcontext(
        request
    )
    warmup = decoder_warmup(config, dummy_requests=requests, forward_warmup=forward_warmup)
    kv_cache_manager = Mock()
    kv_cache_manager.get_num_available_tokens.return_value = 8
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = kv_cache_manager
    if phase == "general":
        warmup._general_warmup_impl(resource_manager, [(4, 0)])
    elif phase == "attention":
        warmup._run_attention_warmup(resource_manager)
    elif phase == "mamba":
        with patch.object(decoder_warmup_module, "MambaHybridCacheManager", type(kv_cache_manager)):
            warmup._run_mamba_hybrid_warmup(resource_manager)
    else:
        warmup._graphs._capture_prefill_cuda_graphs(resource_manager)
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


class TestWarmupCleanup(unittest.TestCase):
    def test_empty_cache_fires_immediately_after_autotuner(self) -> None:
        """Change 1 placement: empty_cache must be the call right after
        _run_autotuner_warmup."""
        model_engine, resource_manager = self.enterContext(_warmup_and_resources())
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

    def test_empty_cache_count_under_default(self) -> None:
        """Default warmup should call empty_cache exactly twice:
        once at the end of step (a) (pre-existing) and once after step (b)
        (Change 1)."""
        model_engine, resource_manager = self.enterContext(_warmup_and_resources())
        calls, _ = _run_warmup_tracked(model_engine, resource_manager)
        self.assertEqual(
            calls.count("empty_cache"),
            2,
            f"Expected exactly 2 empty_cache calls; got order={calls}",
        )

    def test_helix_cp_runs_only_the_generation_autotuner_warmup(self) -> None:
        """With Helix CP, can_run_general_warmup is False and the context
        warmups (general, Mamba hybrid) are skipped; the autotuner warmup still
        runs, on generation requests, followed by one empty_cache."""
        model_engine, resource_manager = self.enterContext(_warmup_and_resources())
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
    def test_generation_only_autotuner_warmup_tunes_one_full_generation_batch(self) -> None:
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
            get_resource_manager=lambda key: (
                kv_cache_manager if key == ResourceManagerType.KV_CACHE_MANAGER else None
            )
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
    def test_flashinfer_mxfp8_respects_disabled_global_autotuner(self) -> None:
        """The global autotuner switch also disables automatic FlashInfer tuning."""
        calls = []

        @contextlib.contextmanager
        def flashinfer_autotune() -> Iterator[None]:
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
    def test_mxfp8_native_and_flashinfer_use_separate_warmup_passes(self) -> None:
        """Native and FlashInfer backends each receive an isolated tuning forward."""
        calls = []

        @contextlib.contextmanager
        def trtllm_autotune(**kwargs: object) -> Iterator[None]:
            self.assertIsNone(kwargs["cache_path"])
            calls.append("trtllm_autotune_enter")
            yield
            calls.append("trtllm_autotune_exit")

        @contextlib.contextmanager
        def flashinfer_autotune() -> Iterator[None]:
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
                get_resource_manager=lambda key: (
                    kv_cache_manager if key == ResourceManagerType.KV_CACHE_MANAGER else None
                )
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
    def test_native_mxfp8_falls_back_after_missing_warmup_batch(self) -> None:
        """A missing startup batch latches native MXFP8 to the default tactic."""
        calls = []

        @contextlib.contextmanager
        def trtllm_autotune(**kwargs: object) -> Iterator[None]:
            self.assertIsNone(kwargs["cache_path"])
            calls.append("autotune_enter")
            yield
            calls.append("autotune_exit")

        @contextlib.contextmanager
        def flashinfer_autotune() -> Iterator[None]:
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
                get_resource_manager=lambda key: (
                    kv_cache_manager if key == ResourceManagerType.KV_CACHE_MANAGER else None
                )
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
    def test_flashinfer_mxfp8_rank_mismatch_falls_back_before_warmup(self) -> None:
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
                get_resource_manager=lambda key: (
                    kv_cache_manager if key == ResourceManagerType.KV_CACHE_MANAGER else None
                )
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
    def test_native_mxfp8_respects_disabled_global_autotuner(self) -> None:
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "world_size,dwdp_size,is_fatal",
    [(1, 0, False), (2, 0, True), (1, 4, True)],
)
def test_general_warmup_oom_policy(world_size: int, dwdp_size: int, is_fatal: bool) -> None:
    engine = _general_warmup_engine(world_size=world_size, dwdp_size=dwdp_size)
    error = torch.OutOfMemoryError("asymmetric OOM")
    engine._forward_warmup = mock.Mock(side_effect=error if is_fatal else [error, None])

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        if is_fatal:
            with pytest.raises(torch.OutOfMemoryError) as excinfo:
                engine._general_warmup_impl(object(), [(128, 0), (64, 0)])
            assert excinfo.value is error
        else:
            engine._general_warmup_impl(object(), [(128, 0), (64, 0)])

    if is_fatal:
        # The remaining shape is never attempted: this rank's peers are
        # already stuck in the failed forward's collectives.
        engine._forward_warmup.assert_called_once()
    else:
        assert engine._forward_warmup.call_count == 2
        # A retry after an OOM between dispatch() and combine() has to start
        # from a clean MoE all-to-all state.
        engine._reset_moe_alltoall_state.assert_called_once_with()
        empty_cache.assert_called_once_with()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "error,recoverable",
    [
        (torch.OutOfMemoryError("OOM"), True),
        (RuntimeError(_KV_ALLOC_ERROR), True),
        (RuntimeError("unexpected"), False),
    ],
)
def test_mamba_preforward_error_policy_when_alone(error: Exception, recoverable: bool) -> None:
    engine, resource_manager = _mamba_engine()
    engine._dummy_requests.create_warmup_request = mock.Mock(side_effect=error)
    engine._forward_warmup = mock.Mock()

    with _no_cuda_side_effects():
        if recoverable:
            _run_mamba_warmup(engine, resource_manager)
        else:
            with pytest.raises(RuntimeError, match="unexpected"):
                _run_mamba_warmup(engine, resource_manager)

    engine._forward_warmup.assert_not_called()
    # The failure predates dispatch(), so there is no half-finished MoE
    # all-to-all exchange to unwind.
    engine._reset_moe_alltoall_state.assert_not_called()


@pytest.mark.cpu_only
def test_mamba_midforward_runtime_error_recovers_when_alone() -> None:
    engine, resource_manager = _mamba_engine()
    engine._dummy_requests.create_warmup_request = mock.Mock(return_value=object())
    engine._forward_warmup = mock.Mock(side_effect=RuntimeError("mid-forward failure"))

    with _no_cuda_side_effects():
        _run_mamba_warmup(engine, resource_manager)

    # Every shape is attempted, and each failed forward has to leave the MoE
    # all-to-all state clean for the shape that follows it.
    attempted = engine._dummy_requests.create_warmup_request.call_count
    assert attempted == 2
    assert engine._forward_warmup.call_count == attempted
    assert engine._reset_moe_alltoall_state.call_count == attempted


@pytest.mark.cpu_only
@pytest.mark.parametrize("phase", ["pre-forward", "mid-forward"])
def test_mamba_error_is_fatal_when_distributed(phase: str) -> None:
    engine, resource_manager = _mamba_engine(world_size=2)
    error = RuntimeError(_KV_ALLOC_ERROR)
    if phase == "pre-forward":
        engine._dummy_requests.create_warmup_request = mock.Mock(side_effect=error)
        engine._forward_warmup = mock.Mock()
    else:
        engine._dummy_requests.create_warmup_request = mock.Mock(return_value=object())
        engine._forward_warmup = mock.Mock(side_effect=error)

    with _no_cuda_side_effects():
        with pytest.raises(RuntimeError) as excinfo:
            _run_mamba_warmup(engine, resource_manager)

    assert excinfo.value is error
    # The remaining shape is never attempted: recovering locally would leave
    # peers waiting in a forward this rank has abandoned.
    engine._dummy_requests.create_warmup_request.assert_called_once()


@pytest.mark.cpu_only
class TestSamplingWarmup(unittest.TestCase):
    def test_sampling_and_phase_metrics_without_kv_cache(self) -> None:
        """Verify sampling warmup coverage and KV-cache cleanup timing within total warmup."""
        runner = object.__new__(DecoderRunner)
        warmup = runner._warmup = decoder_warmup(
            decoder_config(cuda_graph_batch_sizes=[1, 4]),
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=128)),
        )
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
                    warmup._config = replace(warmup._config, enable_in_graph_sampling=enabled)
                    runner._eager_workspace_reclaimer = object()
                    runner.warmup(resource_manager)
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
        self.assertGreaterEqual(warmup._metrics["sampling_warmup_seconds"], 0)
