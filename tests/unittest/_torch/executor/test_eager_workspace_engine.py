# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

_ENGINE_MODULE = "tensorrt_llm._torch.pyexecutor.model_engine"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class TestEagerWorkspaceEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = object.__new__(PyTorchModelEngine)
        self.engine._eager_workspace_shrink_enabled = True
        self.engine._eager_workspace_reclaimer = None
        self.engine.is_spec_decode = False
        self.engine.mapping = SimpleNamespace(cp_size=1)
        self.engine.sparse_attention_config = None
        self.engine._torch_compile_backend = None
        self.engine.breakable_cuda_graph_runner = None
        self.engine._is_warmup = False
        self.metadata = TrtllmAttentionMetadata(max_num_requests=2, max_num_tokens=128)
        self.metadata.workspace = torch.empty(4096, dtype=torch.uint8, device="cuda")
        self.engine.attn_metadata = self.metadata
        self.engine.model = SimpleNamespace(
            model_config=SimpleNamespace(extra_attrs={}), forward=self.model_forward
        )

    def model_forward(self, **kwargs: object) -> int:
        for _ in range(80):
            report = self.metadata.workspace_required_bytes
            if report is not None:
                report.fill_(4096)
        return 42

    def freeze(self) -> None:
        with patch.object(self.engine, "_is_encoder_decoder_model", return_value=False):
            self.engine._freeze_eager_workspace_floor()

    def call(self) -> int:
        with (
            patch(f"{_ENGINE_MODULE}.get_model_extra_attrs", return_value={}),
            patch(f"{_ENGINE_MODULE}.is_trace_enabled", return_value=False),
        ):
            return self.engine.model_forward(attn_metadata=self.metadata)

    def test_counts_whole_model_and_bypasses_warmup(self) -> None:
        self.freeze()
        self.metadata.workspace.resize_(16384)
        self.engine._is_warmup = True
        for _ in range(4):
            self.assertEqual(self.call(), 42)
        self.assertEqual(self.engine._eager_workspace_reclaimer.policy.remaining, 3)
        self.engine._is_warmup = False
        for remaining in (2, 1, 3):
            self.assertEqual(self.call(), 42)
            self.assertEqual(self.engine._eager_workspace_reclaimer.policy.remaining, remaining)
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 4096)

    def test_disabled_flag_leaves_workspace_alone(self) -> None:
        self.engine._eager_workspace_shrink_enabled = False
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.metadata.workspace.resize_(16384)
        for _ in range(4):
            self.call()
        self.assertEqual(self.metadata.workspace.untyped_storage().nbytes(), 16384)

    def test_unsupported_modes_do_not_create_reclaimer(self) -> None:
        for name, value in [
            ("is_spec_decode", True),
            ("_torch_compile_backend", object()),
            ("breakable_cuda_graph_runner", object()),
            ("sparse_attention_config", object()),
        ]:
            with self.subTest(mode=name), patch.object(self.engine, name, value):
                self.freeze()
                self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.engine.mapping.cp_size = 2
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)

    def test_empty_or_nonreclaimable_warmup_is_not_enabled(self) -> None:
        self.metadata.workspace_reclaimable = False
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.metadata.workspace_reclaimable = True
        self.metadata.workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)


if __name__ == "__main__":
    unittest.main()
