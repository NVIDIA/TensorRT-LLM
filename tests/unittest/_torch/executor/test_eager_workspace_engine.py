# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.workspace import EagerWorkspaceReclaimer, WorkspaceShrinkPolicy

_ENGINE_MODULE = "tensorrt_llm._torch.pyexecutor.model_engine"
pytestmark = pytest.mark.cpu_only


@dataclass
class _ScratchMetadata:
    workspace: torch.Tensor
    is_cuda_graph: bool = False
    workspace_reclaimable: bool = True
    cuda_graph_workspace: torch.Tensor | None = None


class TestWorkspaceShrinkPolicy(unittest.TestCase):
    def test_third_forward_uses_window_maximum_and_warmup_floor(self) -> None:
        policy = WorkspaceShrinkPolicy(64)
        self.assertIsNone(policy.finish_forward(1024, 1024, grew=True))
        self.assertIsNone(policy.finish_forward(32, 1024, grew=False))
        self.assertEqual(policy.remaining, 2)
        self.assertIsNone(policy.finish_forward(16, 1024, grew=False))
        self.assertEqual(policy.remaining, 1)
        self.assertEqual(policy.finish_forward(8, 1024, grew=False), 64)
        self.assertEqual(policy.remaining, 3)
        for _ in range(8):
            self.assertIsNone(policy.finish_forward(8, 64, grew=False))

        policy.finish_forward(256, 1024, grew=False)
        policy.finish_forward(128, 1024, grew=False)
        self.assertEqual(policy.finish_forward(32, 1024, grew=False), 256)

    def test_equal_and_growth_reset_counter(self) -> None:
        for required, capacity, grew in [(1024, 1024, False), (1536, 2048, True)]:
            with self.subTest(required=required):
                policy = WorkspaceShrinkPolicy(64)
                policy.finish_forward(32, 1024, grew=False)
                policy.finish_forward(32, 1024, grew=False)
                self.assertIsNone(policy.finish_forward(required, capacity, grew=grew))
                self.assertEqual(policy.remaining, 3)
                self.assertIsNone(policy.finish_forward(32, capacity, grew=False))

    def test_no_report_does_not_advance(self) -> None:
        policy = WorkspaceShrinkPolicy(64)
        policy.finish_forward(32, 1024, grew=False)
        self.assertIsNone(policy.finish_forward(0, 1024, grew=False))
        self.assertEqual(policy.remaining, 2)


class TestEagerWorkspaceReclaimer(unittest.TestCase):
    def setUp(self) -> None:
        # Test ownership with real CPU storage and mocked CUDA interfaces.
        self.metadata = _ScratchMetadata(torch.empty(4096, dtype=torch.uint8))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.current_device = stack.enter_context(
            patch("torch.cuda.current_device", return_value=0)
        )
        self.current_stream = stack.enter_context(
            patch("torch.cuda.current_stream", return_value=object())
        )
        stack.enter_context(patch.object(torch.Tensor, "record_stream"))
        self.reclaimer = EagerWorkspaceReclaimer(self.metadata)

    def report_forward(self, required: int) -> None:
        with self.reclaimer.forward(self.metadata):
            self.size_layer(required)

    def size_layer(self, required: int) -> None:
        if self.metadata.workspace.numel() < required:
            self.metadata.workspace.resize_(required)

    def test_layers_count_once_and_storage_is_replaced(self) -> None:
        with self.reclaimer.forward(self.metadata):
            self.assertEqual(self.metadata.workspace.numel(), 4096)
            self.metadata.workspace.resize_(16384)
        self.assertEqual(self.reclaimer.policy.remaining, 3)
        self.current_stream.assert_called_once_with(0)
        previous = weakref.ref(self.metadata.workspace)
        pointer = self.metadata.workspace.data_ptr()
        for remaining in (2, 1, 3):
            with self.reclaimer.forward(self.metadata):
                self.assertEqual(self.metadata.workspace.numel(), 0)
                for required in (8192, 4096):
                    self.size_layer(required)
                self.assertEqual(self.metadata.workspace.numel(), 8192)
                self.assertEqual(self.metadata.workspace.data_ptr(), pointer)
                self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
            self.assertEqual(self.reclaimer.policy.remaining, remaining)
        self.assertIsNone(previous())
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 8192)
        for _ in range(3):
            self.report_forward(2048)
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 4096)

    def test_failed_and_overlapping_forwards(self) -> None:
        self.metadata.workspace.resize_(16384)
        self.report_forward(4096)
        with self.reclaimer.forward(self.metadata):
            pass
        self.assertEqual(self.reclaimer.policy.remaining, 2)
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            with self.reclaimer.forward(self.metadata):
                raise RuntimeError("model failure")
        self.assertEqual(self.reclaimer.policy.remaining, 3)
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
        with self.reclaimer.forward(self.metadata):
            with self.assertRaisesRegex(RuntimeError, "Overlapping forwards"):
                with self.reclaimer.forward(self.metadata):
                    self.fail("Nested workspace ownership must not be accepted")

    def test_unsafe_paths_do_not_reclaim(self) -> None:
        self.metadata.workspace.resize_(16384)
        self.report_forward(4096)
        self.metadata.is_cuda_graph = True
        self.metadata.cuda_graph_workspace = torch.empty(4096, dtype=torch.uint8)
        graph_tensor = self.metadata.cuda_graph_workspace
        with self.reclaimer.forward(self.metadata):
            self.assertEqual(self.metadata.workspace.numel(), 4096)
        self.metadata.is_cuda_graph = False
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.reclaimer.forward(self.metadata):
                self.assertEqual(self.metadata.workspace.numel(), 4096)
        self.assertEqual(self.reclaimer.policy.remaining, 2)
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
        self.assertIs(self.metadata.cuda_graph_workspace, graph_tensor)
        self.report_forward(4096)
        with self.reclaimer.forward(self.metadata):
            self.size_layer(4096)
            self.metadata.workspace_reclaimable = False
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
        self.assertEqual(self.reclaimer.policy.remaining, 3)

        for multi_stream in (True, False):
            with self.subTest(multi_stream=multi_stream):
                self.metadata.workspace_reclaimable = True
                self.reclaimer = EagerWorkspaceReclaimer(self.metadata)
                with self.reclaimer.forward(self.metadata):
                    self.assertEqual(self.metadata.workspace.numel(), 4096)
                if not multi_stream:
                    self.current_device.return_value = 1
                    self.current_stream.return_value = object()
                with with_multi_stream(multi_stream), self.reclaimer.forward(self.metadata):
                    self.assertEqual(self.metadata.workspace.numel(), 4096)
                self.current_stream.assert_called_with(self.current_device.return_value)
                with self.reclaimer.forward(self.metadata):
                    self.assertEqual(self.metadata.workspace.numel(), 4096)


class TestEagerWorkspaceEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = object.__new__(PyTorchModelEngine)
        self.engine._eager_workspace_reclaimer = None
        self.engine.is_spec_decode = False
        self.engine.mapping = SimpleNamespace(cp_size=1)
        self.engine.sparse_attention_config = None
        self.engine._torch_compile_backend = None
        self.engine.breakable_cuda_graph_runner = None
        self.engine._is_warmup = False
        self.metadata = object.__new__(TrtllmAttentionMetadata)
        self.metadata.workspace = torch.empty(4096, dtype=torch.uint8)
        self.engine.attn_metadata = self.metadata
        self.engine.model = SimpleNamespace(
            model_config=SimpleNamespace(extra_attrs={}), forward=Mock(return_value=42)
        )
        reclaimer_patch = patch(f"{_ENGINE_MODULE}.EagerWorkspaceReclaimer", autospec=True)
        self.reclaimer_class = reclaimer_patch.start()
        self.addCleanup(reclaimer_patch.stop)

    def freeze(self, *, is_encoder_decoder: bool = False) -> None:
        with patch.object(
            self.engine, "_is_encoder_decoder_model", return_value=is_encoder_decoder
        ):
            self.engine._freeze_eager_workspace_floor()

    def call(self) -> int:
        with (
            patch(f"{_ENGINE_MODULE}.get_model_extra_attrs", return_value={}),
            patch(f"{_ENGINE_MODULE}.is_trace_enabled", return_value=False),
        ):
            return self.engine.model_forward(attn_metadata=self.metadata)

    def test_forward_scope_and_warmup_bypass(self) -> None:
        with patch.dict(os.environ, {"TRTLLM_RECLAIM_WORKSPACE": "0"}):
            self.freeze()
            self.assertIsNone(self.engine._eager_workspace_reclaimer)
            self.reclaimer_class.assert_not_called()
        with patch.dict(os.environ, {"TRTLLM_RECLAIM_WORKSPACE": "1"}):
            self.freeze()
        self.reclaimer_class.assert_called_once_with(self.metadata)
        self.assertIs(self.engine._eager_workspace_reclaimer, self.reclaimer_class.return_value)
        scope = self.reclaimer_class.return_value.forward
        self.engine._is_warmup = True
        self.assertEqual(self.call(), 42)
        scope.assert_not_called()
        self.engine._is_warmup = False

        def forward(**kwargs: object) -> int:
            scope.return_value.__enter__.assert_called_once()
            return 42

        self.engine.model.forward.side_effect = forward
        self.assertEqual(self.call(), 42)
        scope.assert_called_once_with(self.metadata)
        scope.return_value.__exit__.assert_called_once_with(None, None, None)

    def test_ineligible_modes_and_workspaces_do_not_create_reclaimer(self) -> None:
        self.freeze(is_encoder_decoder=True)
        self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.reclaimer_class.assert_not_called()
        for name, value in [
            ("is_spec_decode", True),
            ("_torch_compile_backend", object()),
            ("breakable_cuda_graph_runner", object()),
            ("sparse_attention_config", object()),
        ]:
            with self.subTest(mode=name), patch.object(self.engine, name, value):
                self.freeze()
                self.assertIsNone(self.engine._eager_workspace_reclaimer)
        with patch.object(self.engine.mapping, "cp_size", 2):
            self.freeze()
            self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.metadata.workspace_reclaimable = False
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.metadata.workspace_reclaimable = True
        self.metadata.workspace = torch.empty(0, dtype=torch.uint8)
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)


if __name__ == "__main__":
    unittest.main()
