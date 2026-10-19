# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL-EP integration contracts, independent of the library's kernel selection."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tensorrt_llm._torch.moe.fused_moe.communication.nccl_ep import NcclEP, NcclEPWindow
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import NcclEpContext
from tensorrt_llm.bindings.internal.thop import BufferKind


class TestNcclEPZeroCopy(unittest.TestCase):
    def setUp(self):
        strategy = object.__new__(NcclEP)
        strategy.ep_size = 2
        strategy.ep_rank = 0
        strategy.max_tokens_per_rank = 4
        strategy.max_recv_tokens = 8
        strategy.hidden_size = 128
        strategy.max_top_k = 1
        strategy.use_external_fp8 = False
        strategy.use_external_nvfp4 = False
        strategy.uses_internal_fp8_dispatch = False
        strategy.use_low_precision_combine = False
        strategy.mapping = SimpleNamespace(moe_ep_group=[0, 1])
        strategy._ctx = SimpleNamespace(
            window_buffers_supported=True,
            output_tokens_buf=torch.empty((2, 4, 128), dtype=torch.bfloat16),
            topk_idx_dtype=torch.int32,
            kernel_resets_recv_topk_idx=True,
            kernel_writes_global_ids=True,
            recv_topk_idx_buf=torch.zeros((2, 4, 1), dtype=torch.int32),
            recv_topk_weights_buf=torch.ones((2, 4, 1)),
            recv_topk_weights_nd=object(),
            recv_topk_idx_nd=object(),
            scales_nd=None,
            dispatch_outputs=object(),
            dispatch_layout_info=object(),
            dispatch_config=object(),
            get_stream=lambda: 42,
        )
        strategy._dispatch_state = {"num_tokens": 3}
        strategy._handle = Mock()
        self.strategy = strategy
        self.ep = SimpleNamespace(
            Tensor=lambda tensor, **kwargs: SimpleNamespace(tensor=tensor, **kwargs),
            DispatchInputs=SimpleNamespace,
            DispatchOutputs=SimpleNamespace,
            CombineInputs=SimpleNamespace,
            CombineOutputs=SimpleNamespace,
            CombineConfig=SimpleNamespace,
            CombineQuantizationRecipe=SimpleNamespace(NVFP4=1),
        )
        self.window_kind = int(BufferKind.NCCL_WINDOW)

    def allocator(self, tensor, kind=None, handle=123):
        return patch.object(
            torch.ops.trtllm,
            "allocate_output_with_nccl_window",
            return_value=(tensor, self.window_kind if kind is None else kind, handle),
            create=True,
        )

    def test_dispatch_reuses_context_storage_in_eager_and_capture(self):
        for capturing in (False, True):
            with self.subTest(capturing=capturing):
                with (
                    self.allocator(None) as allocate,
                    patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
                    patch.dict(sys.modules, {"nccl.ep": self.ep}),
                ):
                    result, *_ = self.strategy.dispatch(
                        torch.ones((3, 128), dtype=torch.bfloat16),
                        None,
                        torch.zeros((3, 1), dtype=torch.int32),
                        torch.ones((3, 1)),
                        [3, 3],
                    )
                allocate.assert_not_called()
                self.assertEqual(result.data_ptr(), self.strategy._ctx.output_tokens_buf.data_ptr())
                self.assertIs(
                    self.strategy._handle.dispatch.call_args.args[1],
                    self.strategy._ctx.dispatch_outputs,
                )

    def test_context_reservation_and_collective_readiness(self):
        for handle in (0, 123):
            with self.subTest(handle=handle):
                ctx = object.__new__(NcclEpContext)
                ctx.mapping = self.strategy.mapping
                ctx.output_tokens_buf = torch.empty((2, 4, 128), dtype=torch.bfloat16)
                fallback = ctx.output_tokens_buf
                ctx.dispatch_window = None
                ctx._ep_mpi_comm = Mock()
                tokens = torch.empty_like(fallback)
                reservation = Mock()
                reservation.window.return_value = handle
                reservation.tensor.return_value = tokens
                with (
                    patch.object(
                        torch.classes.trtllm, "NCCLWindowReservation", return_value=reservation
                    ) as reserve,
                    patch.object(torch.cuda, "synchronize") as synchronize,
                ):
                    ctx._reserve_dispatch_tokens()
                reserve.assert_called_once_with(fallback, [0, 1])
                reservation.release.assert_not_called()
                if handle:
                    synchronize.assert_called_once_with(fallback.device)
                    ctx._ep_mpi_comm.Barrier.assert_called_once()
                    self.assertIs(ctx.output_tokens_buf, tokens)
                    self.assertEqual(ctx.dispatch_window.handle, handle)
                else:
                    synchronize.assert_not_called()
                    ctx._ep_mpi_comm.Barrier.assert_not_called()
                    self.assertIs(ctx.output_tokens_buf, fallback)
                    self.assertIsNone(ctx.dispatch_window)

    def test_model_releases_only_its_nccl_ep_communications(self):
        from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

        comm = Mock(spec=NcclEP)
        unrelated = Mock()
        model = Mock()
        model.modules.return_value = [
            SimpleNamespace(comm=comm),
            SimpleNamespace(comm=comm),
            SimpleNamespace(comm=unrelated),
            SimpleNamespace(),
        ]
        PyTorchModelEngine._release_nccl_ep_communications(SimpleNamespace(model=model))
        comm.destroy.assert_called_once()
        unrelated.destroy.assert_not_called()
        PyTorchModelEngine._release_nccl_ep_communications(SimpleNamespace(model=None))

    def test_context_release_follows_collective_quiescence(self):
        ctx = object.__new__(NcclEpContext)
        ctx.output_tokens_buf = torch.empty(1)
        ctx.dispatch_window = SimpleNamespace(handle=123)
        ctx._dispatch_reservation = Mock()
        ctx._ep_mpi_comm = Mock()
        ctx.ep_group = None
        ctx.comm = None
        order = Mock()
        order.attach_mock(ctx._ep_mpi_comm.Barrier, "barrier")
        order.attach_mock(ctx._dispatch_reservation.release, "release")
        with patch.object(torch.cuda, "synchronize") as synchronize:
            order.attach_mock(synchronize, "synchronize")
            ctx.destroy()
            ctx.destroy()
        self.assertEqual(
            [call[0] for call in order.mock_calls], ["synchronize", "barrier", "release"]
        )
        self.assertIsNone(ctx._dispatch_reservation)

    def test_dispatch_skips_pool_when_windows_are_unsupported(self):
        self.strategy._ctx.window_buffers_supported = False
        with (
            self.allocator(None) as allocate,
            patch.dict(sys.modules, {"nccl.ep": self.ep}),
        ):
            result, *_ = self.strategy.dispatch(
                torch.ones((3, 128), dtype=torch.bfloat16),
                None,
                torch.zeros((3, 1), dtype=torch.int32),
                torch.ones((3, 1)),
                [3, 3],
            )
        allocate.assert_not_called()
        self.assertEqual(result.data_ptr(), self.strategy._ctx.output_tokens_buf.data_ptr())
        self.assertIs(
            self.strategy._handle.dispatch.call_args.args[1], self.strategy._ctx.dispatch_outputs
        )

    def test_quantized_dispatch_preserves_transport_storage_and_return_dtype(self):
        for quant in ("fp8", "nvfp4", "internal_fp8"):
            with self.subTest(quant=quant):
                self.strategy.use_external_fp8 = quant == "fp8"
                self.strategy.use_external_nvfp4 = quant == "nvfp4"
                self.strategy.uses_internal_fp8_dispatch = quant == "internal_fp8"
                width = 64 if quant == "fp8" else 32 if quant == "nvfp4" else 128
                transport_dtype = torch.float8_e4m3fn if quant == "internal_fp8" else torch.bfloat16
                ctx = self.strategy._ctx
                ctx.output_tokens_buf = torch.empty((2, 4, width), dtype=transport_dtype)
                ctx.scales_buf = torch.ones(
                    (2, 4, 1 if quant == "internal_fp8" else 8),
                    dtype=torch.float32 if quant == "internal_fp8" else torch.uint8,
                )
                tokens = (
                    torch.ones((3, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
                    if quant == "fp8"
                    else torch.ones((3, 64), dtype=torch.uint8)
                    if quant == "nvfp4"
                    else torch.ones((3, 128), dtype=torch.bfloat16)
                )
                scales = torch.ones((3, 8), dtype=torch.uint8) if quant == "nvfp4" else None
                ctx.dispatch_outputs = SimpleNamespace(
                    tokens=SimpleNamespace(
                        tensor=ctx.output_tokens_buf, window=SimpleNamespace(handle=123)
                    )
                )
                with patch.dict(sys.modules, {"nccl.ep": self.ep}):
                    result, _, _, _ = self.strategy.dispatch(
                        tokens,
                        scales,
                        torch.zeros((3, 1), dtype=torch.int32),
                        torch.ones((3, 1)),
                        [3, 3],
                    )
                desc = self.strategy._handle.dispatch.call_args.args[1].tokens
                self.assertEqual(desc.window.handle, 123)
                self.assertEqual(desc.tensor.dtype, transport_dtype)
                self.assertEqual(result.data_ptr(), ctx.output_tokens_buf.data_ptr())
                self.assertEqual(result.shape, (8, 64 if quant == "nvfp4" else 128))
                self.assertEqual(
                    result.dtype, torch.uint8 if quant == "nvfp4" else torch.float8_e4m3fn
                )

    def test_combine_only_attaches_window_to_exact_destination(self):
        pooled = torch.empty((8, 128), dtype=torch.bfloat16)
        for tensor, windowed in (
            (pooled, True),
            (pooled.view(2, 4, 128), True),
            (pooled.clone(), False),
            (pooled.t().contiguous().t(), False),
            (pooled.to(torch.float32), False),
        ):
            with self.subTest(shape=tensor.shape, windowed=windowed):
                self.strategy._dispatch_state = {"num_tokens": 3}
                window = NcclEPWindow(
                    123, pooled.data_ptr(), pooled.numel(), pooled.dtype, pooled.device
                )
                with patch.dict(sys.modules, {"nccl.ep": self.ep}):
                    self.strategy.combine(tensor, window=window)
                desc = self.strategy._handle.combine.call_args.args[0].tokens
                self.assertEqual(hasattr(desc, "window"), windowed)
                if windowed:
                    self.assertEqual(desc.window.handle, 123)
                    self.assertEqual(desc.window_offset, 0)
                self.assertEqual(self.strategy._dispatch_state, {})

    def test_combine_ignores_invalid_window_metadata(self):
        tensor = torch.empty((8, 128), dtype=torch.bfloat16)
        window = NcclEPWindow(123, tensor.data_ptr(), tensor.numel(), tensor.dtype, tensor.device)
        for invalid in (
            window._replace(handle=0),
            window._replace(numel=tensor.numel() - 1),
            window._replace(device=torch.device("meta")),
        ):
            with self.subTest(window=invalid):
                self.strategy._dispatch_state = {"num_tokens": 3}
                with patch.dict(sys.modules, {"nccl.ep": self.ep}):
                    self.strategy.combine(tensor, window=invalid)
                desc = self.strategy._handle.combine.call_args.args[0].tokens
                self.assertFalse(hasattr(desc, "window"))

    def test_combine_window_is_optional(self):
        with patch.dict(sys.modules, {"nccl.ep": self.ep}):
            self.strategy.combine(torch.empty((8, 128), dtype=torch.bfloat16))
        desc = self.strategy._handle.combine.call_args.args[0].tokens
        self.assertFalse(hasattr(desc, "window"))

    def test_combine_accepts_caller_window_without_helper(self):
        tensor = torch.empty((8, 128), dtype=torch.bfloat16)
        window = NcclEPWindow(456, tensor.data_ptr(), tensor.numel(), tensor.dtype, tensor.device)
        with patch.dict(sys.modules, {"nccl.ep": self.ep}):
            self.strategy.combine(tensor, window=window)
        desc = self.strategy._handle.combine.call_args.args[0].tokens
        self.assertIs(desc.window, window)

    def test_scheduler_allocates_destination_only_when_supported(self):
        pooled = torch.empty((8, 128), dtype=torch.bfloat16)
        scheduler = object.__new__(ExternalCommMoEScheduler)
        for supported in (False, True):
            for finalize in (False, True):
                for kind, handle in ((self.window_kind, 123), (0, 0)):
                    with self.subTest(supported=supported, finalize=finalize, kind=kind):
                        scheduler.moe = SimpleNamespace(
                            comm=self.strategy,
                            enable_alltoall=True,
                            backend=SimpleNamespace(
                                supports_moe_output_in_alltoall_workspace=lambda: supported
                            ),
                        )
                        with self.allocator(pooled, kind, handle) as allocate:
                            plan = scheduler._build_comm_plan([3, 3], torch.bfloat16, finalize)
                        expected = supported and finalize
                        self.assertEqual(allocate.call_count, int(expected))
                        if expected and handle:
                            self.assertIs(plan.moe_output, pooled)
                            self.assertEqual(plan.combine_window.handle, handle)
                        else:
                            self.assertIsNone(plan.moe_output)
                            self.assertIsNone(plan.combine_window)
                        self.assertNotIn("combine_window", self.strategy._dispatch_state)
                        self.assertFalse(plan.payload_in_workspace)

    def test_scheduler_skips_pool_for_unsupported_windows_or_output_dtype(self):
        scheduler = object.__new__(ExternalCommMoEScheduler)
        scheduler.moe = SimpleNamespace(
            comm=self.strategy,
            enable_alltoall=True,
            backend=SimpleNamespace(supports_moe_output_in_alltoall_workspace=lambda: True),
        )
        for supported, dtype in ((False, torch.bfloat16), (True, torch.float32)):
            with self.subTest(supported=supported, dtype=dtype):
                self.strategy._ctx.window_buffers_supported = supported
                with self.allocator(None) as allocate:
                    plan = scheduler._build_comm_plan([3, 3], dtype, True)
                allocate.assert_not_called()
                self.assertIsNone(plan.moe_output)
                self.assertIsNone(plan.combine_window)
