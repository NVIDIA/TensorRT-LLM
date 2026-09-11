# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

_ENGINE_MODULE = "tensorrt_llm._torch.pyexecutor.model_engine"
pytestmark = pytest.mark.cpu_only


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
        self.metadata = object.__new__(TrtllmAttentionMetadata)
        self.metadata.workspace = torch.empty(4096, dtype=torch.uint8)
        self.engine.attn_metadata = self.metadata
        self.engine.model = SimpleNamespace(
            model_config=SimpleNamespace(extra_attrs={}), forward=Mock(return_value=42)
        )
        reclaimer_patch = patch(f"{_ENGINE_MODULE}.EagerWorkspaceReclaimer", autospec=True)
        self.reclaimer_class = reclaimer_patch.start()
        self.addCleanup(reclaimer_patch.stop)

    def freeze(self) -> None:
        with patch.object(self.engine, "_is_encoder_decoder_model", return_value=False):
            self.engine._freeze_eager_workspace_floor()

    def call(self) -> int:
        with (
            patch(f"{_ENGINE_MODULE}.get_model_extra_attrs", return_value={}),
            patch(f"{_ENGINE_MODULE}.is_trace_enabled", return_value=False),
        ):
            return self.engine.model_forward(attn_metadata=self.metadata)

    def test_forward_scope_and_warmup_bypass(self) -> None:
        self.freeze()
        self.reclaimer_class.assert_called_once_with(self.metadata)
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

    def test_disabled_flag_leaves_workspace_alone(self) -> None:
        self.engine._eager_workspace_shrink_enabled = False
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)
        self.assertEqual(self.call(), 42)
        self.reclaimer_class.assert_not_called()

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
        self.metadata.workspace = torch.empty(0, dtype=torch.uint8)
        self.freeze()
        self.assertIsNone(self.engine._eager_workspace_reclaimer)


if __name__ == "__main__":
    unittest.main()
