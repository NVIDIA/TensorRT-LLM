# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the attention-DP guard in _capture_generation_cuda_graphs.

Root cause:
  _capture_generation_cuda_graphs iterates batch sizes in REVERSE order (largest
  first, so smaller graphs can reuse the memory pool).  Under attention-DP (ADP)
  the KV-cache capacity can differ across TP ranks.  _create_cuda_graph_warmup_request
  returns None when a rank lacks KV-cache space for the requested batch size.

  Without the guard, some ranks silently `continue` while others enter forward()
  with tp_comm collectives → collective deadlock.  The same scenario in
  _general_warmup_impl and _run_autotuner_warmup is already protected by
  _assert_all_tp_ranks_have_warmup_batch; _capture_generation_cuda_graphs had the
  same gap until this fix.

Symptoms of the bug (before fix):
  - DEP + enable_attention_dp + CUDA-graph batch_sizes including BS ≥ 16
  - Process hangs at iter=0, run.log = 0 bytes (deadlock during C-level NCCL init)
  - Appears as hang(iter=0, stale=~259s) in retry_dep_cells.sh
  - Reproducible: every attempt on umb-b300-020 with BS=32 DEP MEGAMOE_DEEPGEMM

Triggered in B8 sweep: F11/F13/F15/F16 (DEP GVR=OFF cells) with
  cuda_graph_config.batch_sizes=[1,..,8,16,32] on 8xB300 umb-b300-020.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestCudaGraphCaptureAdpGuard(unittest.TestCase):
    """Verify _capture_generation_cuda_graphs handles asymmetric batch is None."""

    def _make_model_engine(self, tp_size: int, batch_none_on_ranks):
        """
        Build a minimal ModelEngine-like object whose
        _capture_generation_cuda_graphs is the real implementation but whose
        dependencies are mocked.

        batch_none_on_ranks: set of TP rank indices that should get batch=None.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine as ModelEngine

        engine = object.__new__(ModelEngine)

        # --- mapping mock ---
        mapping = MagicMock()
        mapping.tp_size = tp_size
        mapping.has_cp_helix.return_value = False
        engine.mapping = mapping

        # --- dist mock: tp_allgather returns flags from all ranks ---
        # The "current rank" in a unit test is effectively rank 0.
        # We simulate the collective result as if we collected has_batch flags
        # from all ranks, where ranks in batch_none_on_ranks return 0.
        dist = MagicMock()

        def tp_allgather(value):
            flags = []
            for r in range(tp_size):
                flags.append(0 if r in batch_none_on_ranks else value)
            return flags

        dist.tp_allgather.side_effect = tp_allgather
        engine.dist = dist

        # --- cuda_graph_runner mock ---
        cg_runner = MagicMock()
        cg_runner.enabled = True
        cg_runner.allow_capture.return_value.__enter__ = lambda s: s
        cg_runner.allow_capture.return_value.__exit__ = MagicMock(return_value=False)
        engine.cuda_graph_runner = cg_runner

        # --- other attrs ---
        engine.batch_size = 32
        engine.max_beam_width = 1
        engine.max_seq_len = 71216
        engine.max_draft_len = 0
        engine.original_max_draft_len = 0
        engine.sparse_attention_config = None
        engine._torch_compile_piecewise_cuda_graph = False
        engine._torch_compile_enabled = False
        engine.is_draft_model = False
        engine.spec_config = None
        engine.enable_spec_decode = False
        engine.is_spec_decode = False
        engine.runtime_draft_len = 0
        engine._cuda_graph_batch_sizes = [1, 2, 4, 8, 16, 32]

        return engine

    def test_guard_present_in_source(self):
        """
        Structural test: _capture_generation_cuda_graphs must call
        _assert_all_tp_ranks_have_warmup_batch.

        This test fails if the guard is accidentally removed.
        """
        import inspect

        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

        src = inspect.getsource(PyTorchModelEngine._capture_generation_cuda_graphs)
        self.assertIn(
            "_assert_all_tp_ranks_have_warmup_batch",
            src,
            "_capture_generation_cuda_graphs must call "
            "_assert_all_tp_ranks_have_warmup_batch to prevent "
            "ADP-asymmetric-batch deadlocks during CUDA graph capture.",
        )

    def test_asymmetric_batch_none_raises_not_hangs(self):
        """
        When some TP ranks return batch=None and others don't,
        _capture_generation_cuda_graphs must raise RuntimeError
        (same contract as _general_warmup_impl) rather than deadlocking.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine as ModelEngine

        engine = self._make_model_engine(tp_size=4, batch_none_on_ranks={1, 2})

        # Rank 0 gets a valid batch; ranks 1,2 would get None.
        # _assert_all_tp_ranks_have_warmup_batch will see mixed flags → RuntimeError.
        valid_batch = MagicMock()

        called_forward = []

        def fake_forward(batch, **kw):
            called_forward.append(batch)

        engine.forward = fake_forward

        # Patch helpers to isolate the function under test.
        with (
            patch.object(
                ModelEngine,
                "_get_graphs_to_capture",
                return_value=[(32, 0)],
            ),
            patch.object(
                ModelEngine,
                "_create_cuda_graph_warmup_request",
                return_value=MagicMock(),  # warmup_request object (non-None)
            ),
            patch.object(
                ModelEngine,
                "_release_batch_context",
            ) as mock_ctx,
            patch.object(
                ModelEngine,
                "_update_draft_inference_state_for_warmup",
            ),
        ):
            # Context manager yields valid_batch (rank 0 has a valid batch)
            mock_ctx.return_value.__enter__ = MagicMock(return_value=valid_batch)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            with self.assertRaises(RuntimeError) as cm:
                engine._capture_generation_cuda_graphs(resource_manager=MagicMock())

        self.assertIn(
            "collective",
            str(cm.exception).lower(),
            "Error message should mention collective deadlock risk.",
        )
        # forward() must NOT have been called — we should have raised before it.
        self.assertEqual(
            called_forward, [], "forward() must not be called when the guard raises RuntimeError."
        )

    def test_all_none_skips_gracefully(self):
        """
        When ALL ranks return batch=None (everyone agrees: not enough KV cache),
        the function must skip gracefully without raising or deadlocking.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine as ModelEngine

        # All 4 ranks will have None → tp_allgather returns [0, 0, 0, 0]
        engine = self._make_model_engine(tp_size=4, batch_none_on_ranks={0, 1, 2, 3})

        called_forward = []

        def fake_forward(batch, **kw):
            called_forward.append(batch)

        engine.forward = fake_forward

        with (
            patch.object(
                ModelEngine,
                "_get_graphs_to_capture",
                return_value=[(32, 0)],
            ),
            patch.object(
                ModelEngine,
                "_create_cuda_graph_warmup_request",
                return_value=MagicMock(),
            ),
            patch.object(
                ModelEngine,
                "_release_batch_context",
            ) as mock_ctx,
            patch.object(
                ModelEngine,
                "_update_draft_inference_state_for_warmup",
            ),
        ):
            # Yield None — this rank also has no KV cache
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            # Should not raise
            engine._capture_generation_cuda_graphs(resource_manager=MagicMock())

        self.assertEqual(called_forward, [], "forward() must not be called when batch is None.")

    def test_all_valid_proceeds_normally(self):
        """
        When all ranks have a valid batch, forward() is called normally.
        """
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine as ModelEngine

        engine = self._make_model_engine(tp_size=4, batch_none_on_ranks=set())

        valid_batch = MagicMock()
        called_forward = []

        def fake_forward(batch, **kw):
            called_forward.append(batch)

        engine.forward = fake_forward

        with (
            patch.object(
                ModelEngine,
                "_get_graphs_to_capture",
                return_value=[(8, 0)],
            ),
            patch.object(
                ModelEngine,
                "_create_cuda_graph_warmup_request",
                return_value=MagicMock(),
            ),
            patch.object(
                ModelEngine,
                "_release_batch_context",
            ) as mock_ctx,
            patch.object(
                ModelEngine,
                "_update_draft_inference_state_for_warmup",
            ),
            patch("torch.cuda.synchronize"),
        ):
            mock_ctx.return_value.__enter__ = MagicMock(return_value=valid_batch)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            engine._capture_generation_cuda_graphs(resource_manager=MagicMock())

        self.assertEqual(
            len(called_forward),
            1,
            "forward() must be called once when all ranks have a valid batch.",
        )


if __name__ == "__main__":
    unittest.main()
