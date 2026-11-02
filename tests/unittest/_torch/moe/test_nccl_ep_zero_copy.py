# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL-EP integration contracts, independent of the library's kernel selection."""

import sys
import unittest
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe import nccl_ep_utils
from tensorrt_llm._torch.moe.fused_moe.communication.base import Communication
from tensorrt_llm._torch.moe.fused_moe.communication.nccl_ep import NcclEP
from tensorrt_llm._torch.moe.fused_moe.nccl_ep_utils import NcclEpContext
from tensorrt_llm.bindings.internal.thop import BufferKind

pytestmark = pytest.mark.cpu_only


class TestNcclEPZeroCopy(unittest.TestCase):
    def setUp(self) -> None:
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

    def allocator(
        self, tensor: torch.Tensor | None, kind: int | None = None, handle: int = 123
    ) -> AbstractContextManager[MagicMock]:
        return patch.object(
            torch.ops.trtllm,
            "allocate_output_with_nccl_window",
            return_value=(tensor, self.window_kind if kind is None else kind, handle),
            create=True,
        )

    def test_dispatch_reuses_context_storage(self) -> None:
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

    def test_context_reservation_and_collective_readiness(self) -> None:
        for handle in (0, 123):
            with self.subTest(handle=handle):
                ctx = object.__new__(NcclEpContext)
                ctx.mapping = self.strategy.mapping
                ctx.output_tokens_buf = torch.empty((2, 4, 128), dtype=torch.bfloat16)
                fallback = ctx.output_tokens_buf
                ctx.dispatch_window = None
                ctx.window_buffers_supported = True
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
                synchronize.assert_called_once_with(fallback.device)
                ctx._ep_mpi_comm.Barrier.assert_called_once()
                if handle:
                    self.assertIs(ctx.output_tokens_buf, tokens)
                    self.assertEqual(ctx.dispatch_window.handle, handle)
                else:
                    self.assertIs(ctx.output_tokens_buf, fallback)
                    self.assertIsNone(ctx.dispatch_window)

    def test_torch_context_reservation_readiness(self) -> None:
        for handle in (0, 123):
            with self.subTest(handle=handle):
                ctx = object.__new__(NcclEpContext)
                ctx.mapping = SimpleNamespace(moe_ep_group=[0, 1], moe_ep_group_pg=object())
                ctx._ep_mpi_comm = None
                ctx.dispatch_window = None
                ctx.window_buffers_supported = True
                ctx.output_tokens_buf = torch.empty(1)
                reservation = Mock()
                reservation.window.return_value = handle
                order = Mock()
                with (
                    patch.object(
                        torch.classes.trtllm, "NCCLWindowReservation", return_value=reservation
                    ),
                    patch.object(torch.cuda, "synchronize") as synchronize,
                    patch.object(torch.distributed, "barrier") as barrier,
                ):
                    order.attach_mock(synchronize, "synchronize")
                    order.attach_mock(barrier, "barrier")
                    order.attach_mock(reservation.tensor, "tensor")
                    ctx._reserve_dispatch_tokens()
                synchronize.assert_called_once_with(torch.device("cpu"))
                barrier.assert_called_once_with(group=ctx.mapping.moe_ep_group_pg)
                if handle:
                    self.assertEqual(
                        [c[0] for c in order.mock_calls], ["synchronize", "barrier", "tensor"]
                    )
                else:
                    self.assertEqual([c[0] for c in order.mock_calls], ["synchronize", "barrier"])
                reservation.release.assert_not_called()

    def test_torch_shutdown_checks_work_before_releasing_resources(self) -> None:
        ctx = object.__new__(NcclEpContext)
        ctx.mapping = SimpleNamespace(moe_ep_group_pg=object())
        ctx._ep_mpi_comm = None
        ctx.output_tokens_buf = torch.empty(1)
        ctx.dispatch_window = SimpleNamespace(handle=123)
        ctx._dispatch_reservation = Mock()
        ctx.ep_group = None
        ctx.comm = None
        order = Mock()
        with (
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.distributed, "barrier") as barrier,
        ):
            barrier.return_value.is_completed.return_value = True
            order.attach_mock(synchronize, "synchronize")
            order.attach_mock(barrier, "barrier")
            order.attach_mock(ctx._dispatch_reservation.release, "release")
            ctx.destroy()
            ctx.destroy()
        self.assertEqual(
            barrier.call_args_list, [call(group=ctx.mapping.moe_ep_group_pg, async_op=True)] * 2
        )
        self.assertEqual(
            [c[0] for c in order.mock_calls],
            [
                "synchronize",
                "barrier",
                "barrier().is_completed",
                "barrier().wait",
                "barrier",
                "barrier().is_completed",
                "barrier().wait",
                "release",
            ],
        )

    def test_torch_shutdown_timeout_or_error_retains_resources(self) -> None:
        for phase in (1, 2):
            for failure in ("timeout", "backend_error"):
                with self.subTest(phase=phase, failure=failure):
                    ctx = object.__new__(NcclEpContext)
                    ctx.mapping = SimpleNamespace(moe_ep_group_pg=object())
                    ctx._ep_mpi_comm = None
                    ctx.output_tokens_buf = torch.empty(1)
                    ctx._dispatch_reservation = Mock()
                    ctx.ep_group = Mock()
                    ctx._timeout_proxy = Mock()
                    work = Mock()
                    work.is_completed.return_value = failure == "backend_error"
                    if failure == "backend_error":
                        work.wait.side_effect = RuntimeError("process group failure")
                    completed = Mock()
                    completed.is_completed.return_value = True
                    self.strategy._ctx = ctx
                    self.strategy._handle = Mock()
                    times = [0, 61] if phase == 1 else [0, 0, 61]
                    with (
                        patch.object(nccl_ep_utils, "_abandoned_contexts", []),
                        patch.object(nccl_ep_utils, "unregister_moe_comm_timeout_proxy"),
                        patch.object(nccl_ep_utils.time, "monotonic", side_effect=times),
                        patch.object(torch.cuda, "synchronize"),
                        patch.object(
                            torch.distributed,
                            "barrier",
                            side_effect=([work] if phase == 1 else [completed, work]),
                        ) as barrier,
                    ):
                        self.strategy.destroy()
                        ctx.destroy()
                        self.assertEqual(nccl_ep_utils._abandoned_contexts, [ctx])
                    self.assertEqual(barrier.call_count, phase)
                    self.assertIs(ctx._shutdown_request, work)
                    self.assertIs(ctx._shutdown_group, ctx.mapping.moe_ep_group_pg)
                    self.assertEqual(ctx._abandoned_instances, [self.strategy])
                    ctx._dispatch_reservation.abandon.assert_called_once()
                    ctx._dispatch_reservation.release.assert_not_called()
                    ctx.ep_group.destroy.assert_not_called()
                    self.strategy._handle.destroy.assert_not_called()
                    if failure == "timeout":
                        work.wait.assert_not_called()

    def test_model_releases_only_its_communication_resources(self) -> None:
        from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine

        comm = Mock(spec=Communication)
        nccl_comm = Mock(spec=NcclEP)
        unrelated = Mock()
        model = Mock()
        model.modules.return_value = [
            SimpleNamespace(comm=comm),
            SimpleNamespace(comm=comm),
            SimpleNamespace(comm=nccl_comm),
            SimpleNamespace(comm=unrelated),
            SimpleNamespace(),
        ]
        ModelEngine.release_communication_resources(SimpleNamespace(model=model))
        comm.release_executor_resources.assert_called_once()
        nccl_comm.release_executor_resources.assert_called_once()
        unrelated.release_executor_resources.assert_not_called()
        ModelEngine.release_communication_resources(SimpleNamespace(model=None))

    def test_executor_release_is_opt_in(self) -> None:
        comm = Mock(spec=Communication)
        Communication.release_executor_resources(comm)
        comm.destroy.assert_not_called()
        with patch.object(self.strategy, "destroy") as destroy:
            self.strategy.release_executor_resources()
        destroy.assert_called_once_with()

    def test_context_release_follows_collective_quiescence(self) -> None:
        for window in (None, SimpleNamespace(handle=123)):
            with self.subTest(windowed=window is not None):
                ctx = object.__new__(NcclEpContext)
                ctx.output_tokens_buf = torch.empty(1)
                ctx.dispatch_window = window
                ctx._dispatch_reservation = Mock()
                ctx._ep_mpi_comm = Mock()
                ctx.ep_group = None
                ctx.comm = None
                order = Mock()
                ctx._ep_mpi_comm.Ibarrier.return_value.Test.return_value = True
                order.attach_mock(ctx._ep_mpi_comm.Ibarrier, "barrier")
                order.attach_mock(ctx._dispatch_reservation.release, "release")
                with patch.object(torch.cuda, "synchronize") as synchronize:
                    order.attach_mock(synchronize, "synchronize")
                    ctx.destroy()
                    ctx.destroy()
                self.assertEqual(
                    [entry[0] for entry in order.mock_calls],
                    [
                        "synchronize",
                        "barrier",
                        "barrier().Test",
                        "barrier",
                        "barrier().Test",
                        "release",
                    ],
                )
                self.assertIsNone(ctx._dispatch_reservation)

    def test_shutdown_timeout_retains_resources_and_pending_request(self) -> None:
        for phase in (1, 2):
            with self.subTest(phase=phase):
                ctx = object.__new__(NcclEpContext)
                ctx.output_tokens_buf = torch.empty(1)
                ctx.dispatch_window = SimpleNamespace(handle=123)
                ctx._dispatch_reservation = Mock()
                ctx._ep_mpi_comm = Mock()
                ctx.ep_group = Mock()
                ctx.comm = Mock()
                ctx._timeout_proxy = Mock()
                pending = Mock()
                pending.Test.return_value = False
                completed = Mock()
                completed.Test.return_value = True
                ctx._ep_mpi_comm.Ibarrier.side_effect = (
                    [pending] if phase == 1 else [completed, pending]
                )
                self.strategy._ctx = ctx
                handle = self.strategy._handle = Mock()
                with (
                    patch.object(nccl_ep_utils, "_abandoned_contexts", []),
                    patch.object(nccl_ep_utils, "unregister_moe_comm_timeout_proxy"),
                    patch.object(
                        nccl_ep_utils.time,
                        "monotonic",
                        side_effect=([0, 61] if phase == 1 else [0, 0, 61]),
                    ),
                    patch.object(torch.cuda, "synchronize") as synchronize,
                ):
                    self.strategy.destroy()
                    self.strategy.destroy()
                    ctx.destroy()
                    self.assertEqual(nccl_ep_utils._abandoned_contexts, [ctx])
                    with self.assertRaisesRegex(RuntimeError, "restart workers"):
                        nccl_ep_utils.get_nccl_ep_context(None, 1, 1, 1, 1)
                    with self.assertRaisesRegex(RuntimeError, "shutting down"):
                        self.strategy._get_context()
                synchronize.assert_called_once()
                self.assertIs(ctx._shutdown_request, pending)
                self.assertEqual(ctx._abandoned_instances, [self.strategy])
                self.assertEqual(ctx._ep_mpi_comm.Ibarrier.call_count, phase)
                ctx._ep_mpi_comm.Free.assert_not_called()
                pending.Cancel.assert_not_called()
                pending.Free.assert_not_called()
                ctx._dispatch_reservation.abandon.assert_called_once()
                ctx._dispatch_reservation.release.assert_not_called()
                ctx.ep_group.destroy.assert_not_called()
                ctx.comm.finalize.assert_not_called()
                ctx.comm.destroy.assert_not_called()
                handle.destroy.assert_not_called()

    def test_late_rank_times_out_in_second_phase(self) -> None:
        # Model MPI's retained participation: rank 0 returns from its timeout,
        # but its outstanding phase-1 request lets rank 1 finish that phase.
        participants = [set(), set()]
        contexts = []
        for rank in (0, 1):
            ctx = object.__new__(NcclEpContext)
            ctx.output_tokens_buf = torch.empty(1)
            ctx._dispatch_reservation = Mock()
            ctx.ep_group = Mock()
            ctx._timeout_proxy = Mock()
            ctx._ep_mpi_comm = Mock()
            contexts.append(ctx)

        def enter(rank: int, phase: int) -> Mock:
            participants[phase].add(rank)
            request = Mock()
            request.Test.side_effect = lambda: len(participants[phase]) == 2
            return request

        contexts[0]._ep_mpi_comm.Ibarrier.side_effect = lambda: enter(0, 0)
        phases = iter((0, 1))
        contexts[1]._ep_mpi_comm.Ibarrier.side_effect = lambda: enter(1, next(phases))
        with (
            patch.object(nccl_ep_utils, "_abandoned_contexts", []),
            patch.object(nccl_ep_utils, "unregister_moe_comm_timeout_proxy"),
            patch.object(torch.cuda, "synchronize"),
            patch.object(nccl_ep_utils.time, "monotonic", side_effect=[0, 61, 61, 61, 122]),
        ):
            for ctx in contexts:
                ctx.destroy()
                ctx.ep_group.destroy.assert_not_called()
                ctx._dispatch_reservation.release.assert_not_called()
                ctx._dispatch_reservation.abandon.assert_called_once()
            self.assertEqual(nccl_ep_utils._abandoned_contexts, contexts)
        self.assertEqual(participants, [{0, 1}, {1}])

    def test_shutdown_rendezvous_precedes_native_handle_destruction(self) -> None:
        ctx = Mock()
        ctx._prepare_shutdown.return_value = True
        self.strategy._ctx = ctx
        order = Mock()
        order.attach_mock(ctx._prepare_shutdown, "prepare")
        order.attach_mock(self.strategy._handle.destroy, "handle")
        with patch.object(nccl_ep_utils, "release_nccl_ep_context") as release:
            order.attach_mock(release, "release")
            self.strategy.destroy()
        self.assertEqual([c[0] for c in order.mock_calls], ["prepare", "handle", "release"])
        self.assertIsNone(self.strategy._ctx)

    @contextmanager
    def _context_constructor_environment(
        self,
    ) -> Iterator[tuple[SimpleNamespace, SimpleNamespace, MagicMock]]:
        allocate = torch.empty

        def cpu_empty(*args: object, **kwargs: object) -> torch.Tensor:
            return allocate(*args, **{**kwargs, "device": "cpu"})

        mapping = SimpleNamespace(moe_ep_size=2, moe_ep_rank=0, pp_rank=0)
        core = SimpleNamespace(Communicator=Mock(), get_unique_id=Mock())
        group_config = Mock(side_effect=SimpleNamespace)
        group_config.__dataclass_fields__ = {"zero_copy": None}
        ep = SimpleNamespace(
            Algorithm=SimpleNamespace(LOW_LATENCY=1),
            DispatchConfig=SimpleNamespace,
            DispatchOutputs=SimpleNamespace,
            DispatchQuantizationRecipe=SimpleNamespace(NONE=0, DS_FP8E3M4=1, FWD=2),
            Group=Mock(),
            GroupConfig=group_config,
            Layout=SimpleNamespace(RANK_MAJOR=SimpleNamespace(name="RANK_MAJOR")),
            LayoutInfo=SimpleNamespace,
            Tensor=Mock(
                side_effect=lambda tensor, **kwargs: SimpleNamespace(tensor=tensor, **kwargs)
            ),
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "nccl": SimpleNamespace(core=core, ep=ep),
                    "nccl.core": core,
                    "nccl.ep": ep,
                },
            ),
            patch("tensorrt_llm._utils.mpi_comm") as mpi_comm,
            patch.object(nccl_ep_utils, "_nccl_ep_supports_int32_topk_idx", return_value=True),
            patch.object(nccl_ep_utils, "nccl_ep_supports_version", return_value=True),
            patch.object(nccl_ep_utils, "register_moe_comm_timeout_proxy"),
            patch.object(nccl_ep_utils, "unregister_moe_comm_timeout_proxy"),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch, "empty", side_effect=cpu_empty) as empty,
        ):
            mpi_comm.return_value.Split.return_value.allreduce.side_effect = lambda failed: failed
            yield mapping, ep, empty

    def test_context_constructor_wraps_dispatch_output_with_reserved_window(self) -> None:
        for quant in ("none", "fp8", "nvfp4", "internal_fp8"):
            for window in (None, SimpleNamespace(handle=123)):
                with self.subTest(quant=quant, windowed=window is not None):
                    reservation = Mock()

                    def reserve(ctx: NcclEpContext) -> None:
                        ctx._dispatch_reservation = reservation
                        ctx.dispatch_window = window

                    with (
                        self._context_constructor_environment() as (mapping, ep, _),
                        patch.object(NcclEpContext, "_reserve_dispatch_tokens", reserve),
                    ):
                        ctx = NcclEpContext(
                            mapping,
                            num_experts=4,
                            max_tokens_per_rank=4,
                            hidden_size=512,
                            max_top_k=1,
                            use_internal_fp8_dispatch=quant == "internal_fp8",
                            external_fp8=quant == "fp8",
                            external_nvfp4=quant == "nvfp4",
                        )
                    reservation.release.assert_not_called()
                    reservation.abandon.assert_not_called()
                    output_calls = [
                        c for c in ep.Tensor.call_args_list if c.args[0] is ctx.output_tokens_buf
                    ]
                    self.assertEqual(len(output_calls), 1)
                    kwargs = output_calls[0].kwargs
                    if window is not None:
                        self.assertEqual(set(kwargs), {"window", "window_offset"})
                        self.assertIs(kwargs["window"], window)
                        self.assertEqual(kwargs["window_offset"], 0)
                    else:
                        self.assertEqual(kwargs, {})
                    self.assertIs(ctx.dispatch_outputs.tokens, ctx.output_tokens_nd)
                    self.assertIs(ctx.output_tokens_nd.tensor, ctx.output_tokens_buf)

    def test_context_construction_failure_abandons_reservation(self) -> None:
        for stage in ("reservation", "buffer", "descriptor", "outputs", "peer"):
            with self.subTest(stage=stage):
                failure = RuntimeError(f"construction failed: {stage}")
                reservation = Mock()
                captured = []

                def reserve(ctx: NcclEpContext) -> None:
                    captured.append(ctx)
                    ctx._dispatch_reservation = reservation
                    ctx.dispatch_window = SimpleNamespace(handle=123)
                    if stage == "reservation":
                        raise failure
                    if stage == "peer":
                        ctx._ep_mpi_comm.allreduce.side_effect = lambda failed: 1

                with (
                    self._context_constructor_environment() as (mapping, ep, empty),
                    patch.object(NcclEpContext, "_reserve_dispatch_tokens", reserve),
                    patch.object(nccl_ep_utils, "_abandoned_contexts", []) as abandoned,
                    patch.object(torch.cuda, "synchronize") as synchronize,
                ):
                    if stage == "buffer":
                        # The first allocation precedes the reservation; the next
                        # allocation exercises rollback after acquiring it.
                        empty.side_effect = [torch.zeros(1), failure]
                    elif stage == "descriptor":
                        ep.Tensor.side_effect = failure
                    elif stage == "outputs":
                        ep.DispatchOutputs = Mock(side_effect=failure)
                    with self.assertRaises(RuntimeError) as raised:
                        NcclEpContext(mapping, 4, 4, 512, 1)
                    if stage == "peer":
                        self.assertIn("construction failed on a peer rank", str(raised.exception))
                    else:
                        self.assertIs(raised.exception, failure)
                    ctx = captured[0]
                    if stage == "reservation":
                        ctx._ep_mpi_comm.allreduce.assert_not_called()
                    else:
                        ctx._ep_mpi_comm.allreduce.assert_called_once_with(int(stage != "peer"))
                    self.assertEqual(abandoned, [ctx])
                    self.assertIs(ctx._dispatch_reservation, reservation)
                    self.assertTrue(ctx._abandoned)
                    reservation.abandon.assert_called_once()
                    reservation.release.assert_not_called()
                    ctx.ep_group.destroy.assert_not_called()
                    ctx.comm.destroy.assert_not_called()
                    ctx._ep_mpi_comm.Free.assert_not_called()
                    synchronize.assert_not_called()

    def test_construction_status_exchange_mpi_and_torch(self) -> None:
        for uses_mpi in (False, True):
            for failure in (None, "local", "peer"):
                with self.subTest(uses_mpi=uses_mpi, failure=failure):
                    ctx = object.__new__(NcclEpContext)
                    ctx._ep_mpi_comm = Mock() if uses_mpi else None
                    ctx.mapping = SimpleNamespace(moe_ep_group_pg=object())
                    status = torch.zeros(1, dtype=torch.int32)
                    votes = []

                    def vote(value: int | torch.Tensor, **kwargs: object) -> int | None:
                        votes.append(int(value) if uses_mpi else int(value.item()))
                        failed = int(failure is not None)
                        if uses_mpi:
                            return failed
                        value.fill_(failed)

                    if uses_mpi:
                        ctx._ep_mpi_comm.allreduce.side_effect = vote
                    local_error = ValueError("local initialization failure")
                    with patch.object(torch.distributed, "all_reduce", side_effect=vote) as reduce:
                        try:
                            with ctx._construction_rendezvous(None if uses_mpi else status):
                                if failure == "local":
                                    raise local_error
                        except ValueError as exc:
                            self.assertEqual(failure, "local")
                            self.assertIs(exc, local_error)
                        except RuntimeError as exc:
                            self.assertEqual(failure, "peer")
                            self.assertIn("construction failed on a peer rank", str(exc))
                        else:
                            self.assertIsNone(failure)
                    self.assertEqual(votes, [int(failure == "local")])
                    if uses_mpi:
                        reduce.assert_not_called()
                    else:
                        self.assertIs(reduce.call_args.args[0], status)
                        self.assertEqual(
                            reduce.call_args.kwargs,
                            {
                                "op": torch.distributed.ReduceOp.MAX,
                                "group": ctx.mapping.moe_ep_group_pg,
                            },
                        )

    def test_construction_status_error_preserves_local_exception(self) -> None:
        for uses_mpi in (False, True):
            for local_failure in (False, True):
                with self.subTest(uses_mpi=uses_mpi, local_failure=local_failure):
                    ctx = object.__new__(NcclEpContext)
                    ctx._ep_mpi_comm = Mock() if uses_mpi else None
                    ctx.mapping = SimpleNamespace(moe_ep_group_pg=object())
                    status = torch.zeros(1, dtype=torch.int32)
                    exchange_error = RuntimeError("status exchange failed")
                    local_error = ValueError("local initialization failure")
                    if uses_mpi:
                        ctx._ep_mpi_comm.allreduce.side_effect = exchange_error
                    with patch.object(torch.distributed, "all_reduce", side_effect=exchange_error):
                        with self.assertRaises((ValueError, RuntimeError)) as raised:
                            with ctx._construction_rendezvous(None if uses_mpi else status):
                                if local_failure:
                                    raise local_error
                    self.assertIs(
                        raised.exception, local_error if local_failure else exchange_error
                    )

    def test_quantized_dispatch_preserves_transport_storage_and_return_dtype(self) -> None:
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
                self.assertEqual(result.data_ptr(), ctx.output_tokens_buf.data_ptr())
                self.assertEqual(result.shape, (8, 64 if quant == "nvfp4" else 128))
                self.assertEqual(
                    result.dtype, torch.uint8 if quant == "nvfp4" else torch.float8_e4m3fn
                )
