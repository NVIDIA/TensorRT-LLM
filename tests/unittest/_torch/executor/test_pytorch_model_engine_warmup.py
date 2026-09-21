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
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.speculative.utils import update_draft_len
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    DraftTargetDecodingConfig,
    PARDDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping


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
    """Patch the four warmup helpers + empty_cache + MoERunner.clear and run
    model_engine.warmup(). Optionally force helix CP and capture logs.
    Returns (call_order_list, log_records_or_None)."""
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
        patch.object(model_engine, "_run_cuda_graph_warmup", side_effect=tracker("cuda_graph")),
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

    def test_legacy_warmup_skips_without_kv_cache(self):
        model_engine = object.__new__(PyTorchModelEngine)
        model_engine.moe_load_balancer = None
        model_engine.is_warmup = False
        model_engine.enable_in_graph_sampling = False
        model_engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        model_engine._runner = None
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None

        with (
            patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.warmup_sampling_module"
            ) as warmup_sampling,
            _capture_tllm_logs() as logs,
        ):
            model_engine.warmup(resource_manager)

        self.assertTrue(
            any("Skipping warm up as no KV Cache manager allocated." in log for log in logs)
        )
        warmup_sampling.assert_called_once_with()

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
                llm_args=SimpleNamespace(enable_autotuner=False),
                cuda_graph_runner=SimpleNamespace(enabled=True),
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
                llm_args=SimpleNamespace(enable_autotuner=True),
                cuda_graph_runner=SimpleNamespace(enabled=True),
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
                llm_args=SimpleNamespace(enable_autotuner=True),
                cuda_graph_runner=SimpleNamespace(enabled=True),
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
                llm_args=SimpleNamespace(enable_autotuner=True),
                cuda_graph_runner=SimpleNamespace(enabled=True),
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

    def test_native_mxfp8_respects_disabled_global_autotuner(self):
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
                llm_args=SimpleNamespace(enable_autotuner=False),
                cuda_graph_runner=SimpleNamespace(enabled=False),
                model=SimpleNamespace(modules=lambda: [SimpleNamespace(quant_method=method)]),
            )

            PyTorchModelEngine._run_autotuner_warmup(engine, Mock())

        self.assertFalse(method.use_native_autotuner)
        self.assertFalse(method.needs_native_autotune)


if __name__ == "__main__":
    unittest.main()


@pytest.fixture
def warmup_timing(monkeypatch):
    """Exercise the real timing helpers without constructing a GPU engine."""
    from tensorrt_llm._torch.pyexecutor import model_engine

    engine = SimpleNamespace(
        mapping=SimpleNamespace(rank=3),
        _warmup_purpose="memory_profiling",
        _warmup_pass=1,
        _warmup_timings={},
        _WARMUP_SLOW_PHASE_SEC=PyTorchModelEngine._WARMUP_SLOW_PHASE_SEC,
    )
    clock = Mock()
    log = Mock()
    monkeypatch.setattr(model_engine.time, "perf_counter", clock)
    monkeypatch.setattr(model_engine, "logger", log)
    monkeypatch.setattr(model_engine.os, "getpid", lambda: 42)
    return engine, clock, log


@pytest.mark.cpu_only
def test_warmup_phase_repeated_names(warmup_timing):
    """Repeated names accumulate and the summary reports identity and percentages."""
    engine, clock, log = warmup_timing
    clock.side_effect = [0, 2, 3, 6]
    for _ in range(2):
        with PyTorchModelEngine._warmup_phase(engine, "attention"):
            pass
    assert engine._warmup_timings == {"attention": 5}
    PyTorchModelEngine._log_warmup_summary(engine, 10)
    log.info.assert_any_call(
        "[warmup][pid=42][rank=3][purpose=memory_profiling][pass=1] "
        "summary: total=10.0s | attention=5.0s (50%)"
    )
    log.warning.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_warmup_phase_failure_records_elapsed(warmup_timing, error_type):
    """Preserve elapsed time and the original exception on interrupted warmup."""
    engine, clock, log = warmup_timing
    clock.side_effect = [2, 7]
    error = error_type("warmup failed")
    with pytest.raises(error_type) as caught:
        with PyTorchModelEngine._warmup_phase(engine, "attention"):
            raise error
    assert caught.value is error
    assert engine._warmup_timings == {"attention": 5}
    log.warning.assert_called_once_with(
        "[warmup][pid=42][rank=3][purpose=memory_profiling][pass=1] attention: failed after 5.0s"
    )
    assert not any(": done" in c.args[0] for c in log.info.call_args_list)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "timings, expected",
    [
        ({}, "summary: total=0.0s (no phases ran)"),
        ({"attention": 0.0}, "summary: total=0.0s | attention=0.0s (0%)"),
    ],
)
def test_warmup_summary_empty_or_zero(warmup_timing, timings, expected):
    """Empty or zero-duration summaries remain finite and do not warn."""
    engine, _, log = warmup_timing
    engine._warmup_timings = timings
    PyTorchModelEngine._log_warmup_summary(engine, 0)
    assert log.info.call_args.args[0].endswith(expected)
    log.warning.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("over_threshold", [False, True])
def test_warmup_slow_phase_warning(warmup_timing, over_threshold):
    """Warn only when a phase strictly exceeds the slow-phase threshold."""
    engine, _, log = warmup_timing
    elapsed = engine._WARMUP_SLOW_PHASE_SEC + int(over_threshold)
    engine._warmup_timings = {"attention": elapsed}
    PyTorchModelEngine._log_warmup_summary(engine, elapsed)
    if over_threshold:
        log.warning.assert_called_once_with(
            "[warmup][pid=42][rank=3][purpose=memory_profiling][pass=1] "
            f"slow phases (>{engine._WARMUP_SLOW_PHASE_SEC:.0f}s): attention={elapsed:.1f}s"
        )
    else:
        log.warning.assert_not_called()
