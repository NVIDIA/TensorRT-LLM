# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.workspace import EagerWorkspaceReclaimer, WorkspaceShrinkPolicy
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream


@dataclass
class _ScratchMetadata:
    workspace: torch.Tensor
    is_cuda_graph: bool = False
    workspace_reclaimable: bool = True
    cuda_graph_workspace: torch.Tensor | None = None


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


if __name__ == "__main__":
    unittest.main()
