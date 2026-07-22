"""Unit tests for warmup-cleanup behavior in PyTorchModelEngine.warmup().

Locks in that gc.collect() + torch.cuda.empty_cache() fire immediately after
_run_autotuner_warmup (step b) to release autotuner exploration leftovers.

The torch.cuda.empty_cache() after teardown_managers() in py_executor_creator
is covered end-to-end by integration tests rather than unit-tested here.
"""

import contextlib
import sys
import unittest
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import CudaGraphConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping


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
            model_path="dummy", mapping=mapping, model=_DummyModel(dtype), llm_args=llm_args
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
        cuda_graph_config=CudaGraphConfig(
            enable_padding=True, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]
        ),
    )
    model_engine = _DummyModelEngine(llm_args, torch.half)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=model_engine.model.config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_tokens,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
    return model_engine, resource_manager


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

    def test_empty_cache_fires_immediately_after_autotuner(self):
        """Change 1 placement: empty_cache must be the call right after
        _run_autotuner_warmup."""
        model_engine, resource_manager = _build_engine_and_resource_manager()
        calls, _ = _run_warmup_tracked(model_engine, resource_manager)

        self.assertIn("autotuner", calls)
        autotuner_idx = calls.index("autotuner")
        self.assertLess(
            autotuner_idx + 1, len(calls), f"Expected something after autotuner; got {calls}"
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
            calls.count("empty_cache"), 0, f"Helix CP should skip all warmup cleanup; got {calls}"
        )

    def test_flashinfer_mxfp8_autotunes_before_graph_capture(self):
        """An auto-enabled M3 linear tunes even when TRT autotuning is disabled."""
        calls = []

        @contextlib.contextmanager
        def flashinfer_autotune():
            calls.append("flashinfer_autotune_enter")
            yield
            calls.append("flashinfer_autotune_exit")

        flashinfer_module = ModuleType("flashinfer")
        flashinfer_module.mm_mxfp8 = Mock()
        flashinfer_autotuner_module = ModuleType("flashinfer.autotuner")
        flashinfer_autotuner_module.autotune = Mock(side_effect=flashinfer_autotune)
        flashinfer_module.autotuner = flashinfer_autotuner_module

        with (
            patch.dict(
                sys.modules,
                {
                    "flashinfer": flashinfer_module,
                    "flashinfer.autotuner": flashinfer_autotuner_module,
                },
            ),
            patch(
                "tensorrt_llm._torch.modules.linear._mxfp8_cutlass_op_available",
                return_value=True,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
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
                kv_cache_manager_key="kv_cache",
                max_num_tokens=16,
                batch_size=16,
                max_seq_len=2,
                original_max_draft_len=0,
                mapping=SimpleNamespace(tp_size=1),
                is_draft_model=False,
                no_cuda_graph=lambda: contextlib.nullcontext(),
                _create_warmup_request=Mock(return_value=object()),
                _release_batch_context=Mock(return_value=contextlib.nullcontext(object())),
                _assert_all_tp_ranks_have_warmup_batch=Mock(),
                forward=Mock(side_effect=lambda *args, **kwargs: calls.append("forward")),
            )
            kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: 16)
            resource_manager = SimpleNamespace(
                get_resource_manager=lambda key: (kv_cache_manager if key == "kv_cache" else None)
            )

            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.empty_cache"),
                patch("tensorrt_llm._torch.pyexecutor.model_engine.clear_memory_buffers"),
            ):
                PyTorchModelEngine._run_autotuner_warmup(engine, resource_manager)

        self.assertEqual(
            calls,
            ["flashinfer_autotune_enter", "forward", "flashinfer_autotune_exit"],
        )
        self.assertEqual(method.backend, "auto")
        self.assertTrue(method._flashinfer_autotuned)
        flashinfer_autotuner_module.autotune.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
