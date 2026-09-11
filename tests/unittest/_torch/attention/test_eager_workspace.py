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
    workspace_required_bytes: torch.Tensor | None = None
    cuda_graph_workspace: torch.Tensor | None = None


@pytest.mark.cpu_only
class TestWorkspaceShrinkPolicy(unittest.TestCase):
    def test_third_forward_and_warmup_floor(self) -> None:
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

    def test_window_maximum_not_last_requirement(self) -> None:
        policy = WorkspaceShrinkPolicy(64)
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

    def test_invalid_requirement_and_floor(self) -> None:
        with self.assertRaises(ValueError):
            WorkspaceShrinkPolicy(-1)
        for required, capacity in [(-1, 64), (128, 64), (8, 32)]:
            with self.subTest(required=required, capacity=capacity):
                with self.assertRaises(ValueError):
                    WorkspaceShrinkPolicy(64).finish_forward(required, capacity, grew=False)


@pytest.mark.cpu_only
class TestEagerWorkspaceReclaimer(unittest.TestCase):
    def setUp(self) -> None:
        # Test ownership with real CPU storage; CUDA ordering is tested below.
        self.metadata = _ScratchMetadata(torch.empty(4096, dtype=torch.uint8))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.current_stream = stack.enter_context(
            patch("torch.cuda.current_stream", return_value=object())
        )
        stack.enter_context(patch.object(torch.Tensor, "record_stream"))
        self.reclaimer = EagerWorkspaceReclaimer(self.metadata)

    def report_forward(self, required: int) -> None:
        with self.reclaimer.forward(self.metadata):
            self.metadata.workspace_required_bytes.fill_(required)

    def test_layers_count_once_and_storage_is_replaced(self) -> None:
        self.metadata.workspace.resize_(16384)
        previous = weakref.ref(self.metadata.workspace)
        for remaining in (2, 1, 3):
            with self.reclaimer.forward(self.metadata):
                for _ in range(2):
                    self.metadata.workspace_required_bytes.fill_(4096)
                self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
            self.assertEqual(self.reclaimer.policy.remaining, remaining)
        self.assertIsNone(previous())
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 4096)
        self.assertIsNone(self.metadata.workspace_required_bytes)

    def test_exception_resets_and_clears_report(self) -> None:
        self.metadata.workspace.resize_(16384)
        self.report_forward(4096)
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            with self.reclaimer.forward(self.metadata):
                raise RuntimeError("model failure")
        self.assertEqual(self.reclaimer.policy.remaining, 3)
        self.assertIsNone(self.metadata.workspace_required_bytes)

    def test_unsupported_backend_prevents_shrink(self) -> None:
        self.metadata.workspace.resize_(16384)
        self.report_forward(4096)
        self.report_forward(4096)
        with self.reclaimer.forward(self.metadata):
            self.metadata.workspace_required_bytes.fill_(4096)
            self.metadata.workspace_reclaimable = False
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
        self.assertEqual(self.reclaimer.policy.remaining, 3)

    def test_graph_and_capture_leave_counter_and_storage_unchanged(self) -> None:
        self.metadata.workspace.resize_(16384)
        self.report_forward(4096)
        self.metadata.is_cuda_graph = True
        graph_tensor = self.metadata.cuda_graph_workspace
        with self.reclaimer.forward(self.metadata):
            self.assertIsNone(self.metadata.workspace_required_bytes)
        self.metadata.is_cuda_graph = False
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.reclaimer.forward(self.metadata):
                self.assertIsNone(self.metadata.workspace_required_bytes)
        self.assertEqual(self.reclaimer.policy.remaining, 2)
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)
        self.assertIs(self.metadata.cuda_graph_workspace, graph_tensor)

    def test_multi_stream_disables_reclamation(self) -> None:
        with with_multi_stream(True):
            with self.reclaimer.forward(self.metadata):
                self.assertIsNone(self.metadata.workspace_required_bytes)
        with self.reclaimer.forward(self.metadata):
            self.assertIsNone(self.metadata.workspace_required_bytes)

    def test_changed_stream_disables_reclamation(self) -> None:
        self.report_forward(4096)
        self.current_stream.return_value = object()
        with self.reclaimer.forward(self.metadata):
            self.assertIsNone(self.metadata.workspace_required_bytes)

    def test_overlapping_forward_is_rejected(self) -> None:
        with self.reclaimer.forward(self.metadata):
            with self.assertRaisesRegex(RuntimeError, "Overlapping forwards"):
                with self.reclaimer.forward(self.metadata):
                    self.fail("Nested workspace ownership must not be accepted")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required; no architecture restriction")
class TestEagerWorkspaceCuda(unittest.TestCase):
    def test_free_before_allocate_preserves_pending_same_stream_work(self) -> None:
        mib = 1024**2
        metadata = _ScratchMetadata(torch.empty(4096, dtype=torch.uint8, device="cuda"))
        reclaimer = EagerWorkspaceReclaimer(metadata)
        metadata.workspace.resize_(4 * mib)
        for _ in range(2):
            with reclaimer.forward(metadata):
                metadata.workspace_required_bytes.fill_(4096)
        with reclaimer.forward(metadata):
            metadata.workspace.fill_(1)
            result = metadata.workspace.sum()
            metadata.workspace_required_bytes.fill_(4096)
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        self.assertLessEqual(torch.cuda.max_memory_allocated(), before)
        self.assertEqual(torch.cuda.memory_allocated(), before - 4 * mib + 4096)
        metadata.workspace.fill_(7)
        torch.cuda.synchronize()
        self.assertEqual(result.item(), 4 * mib)


if __name__ == "__main__":
    unittest.main()
