# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NCCL fault-tolerance hooks on AllGatherReduceScatter.

These tests intentionally mock the C++ custom op.  They verify the Python-side
contract (global-rank validation, survivor publication, and dispatch-size
bookkeeping) without requiring NCCL, MPI, or GPUs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import pytest

from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm._torch.distributed import nccl_fault_tolerance
from tensorrt_llm._torch.distributed import ops as distributed_ops
from tensorrt_llm._torch.modules import linear as linear_module
from tensorrt_llm._torch.modules.fused_moe.communication import (
    allgather_reducescatter as allgather_rs_module,
)
from tensorrt_llm._torch.modules.fused_moe.communication.allgather_reducescatter import (
    AllGatherReduceScatter,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode


@pytest.fixture(autouse=True)
def _enable_and_clear_nccl_survivor_membership(monkeypatch):
    monkeypatch.setattr(nccl_fault_tolerance, "NCCL_FAULT_TOLERANCE_ENABLED", True)
    monkeypatch.setattr(distributed_ops, "NCCL_FAULT_TOLERANCE_ENABLED", True)
    monkeypatch.setattr(torch_custom_ops, "NCCL_FAULT_TOLERANCE_ENABLED", True)
    monkeypatch.setattr(allgather_rs_module, "NCCL_FAULT_TOLERANCE_ENABLED", True)
    monkeypatch.setattr(linear_module, "NCCL_FAULT_TOLERANCE_ENABLED", True)
    nccl_fault_tolerance._reset_nccl_group_registry_for_tests()
    yield
    nccl_fault_tolerance._reset_nccl_group_registry_for_tests()


@dataclass
class _FakeMapping:
    """A PP slice whose EP-local ranks map to non-zero global ranks."""

    world_size: int = 8
    rank: int = 6
    tp_size: int = 4
    tp_rank: int = 2
    moe_ep_size: int = 4
    moe_ep_rank: int = 2
    enable_attention_dp: bool = False
    _group: list[int] = field(default_factory=lambda: [4, 5, 6, 7])
    _moe_group: list[int] | None = None

    @property
    def tp_group(self) -> list[int]:
        return list(self._group)

    @property
    def moe_ep_group(self) -> list[int]:
        return list(self._group if self._moe_group is None else self._moe_group)


def _patch_reinit_op(
    monkeypatch: pytest.MonkeyPatch,
    implementation: Callable[[list[int], list[int], int], None],
) -> None:
    monkeypatch.setattr(
        allgather_rs_module.torch.ops.trtllm,
        "nccl_comm_abort_and_reinit",
        implementation,
        raising=False,
    )


def _patch_async_error_op(
    monkeypatch: pytest.MonkeyPatch,
    implementation: Callable[[list[int]], str | None],
) -> None:
    monkeypatch.setattr(
        allgather_rs_module.torch.ops.trtllm,
        "nccl_comm_get_async_error",
        implementation,
        raising=False,
    )


def _assert_active_mapping(mapping, expected_group: list[int], expected_rank: int) -> None:
    """Assert the distributed op sees compact ranks for the rebuilt comm."""
    assert list(mapping.tp_group) == expected_group
    assert mapping.tp_size == len(expected_group)
    assert mapping.tp_rank == expected_rank


def test_abort_and_reinit_canonicalizes_global_ranks(monkeypatch):
    calls: list[tuple[list[int], list[int], int]] = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append(
            (list(group), list(active_group), rendezvous_id)
        ),
    )
    comm = AllGatherReduceScatter(_FakeMapping())

    # Input order must not affect NCCL rank assignment: the C++ op receives the
    # original TP-group order in the global/world-rank domain.
    comm.abort_and_reinit([7, 4, 6])

    assert calls == [([4, 5, 6, 7], [4, 6, 7], 1)]


def test_reconfiguration_rejects_static_sharded_allgather(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    observed = []

    def fake_allgather(input, group, rank, group_boxed, dim, sizes):
        observed.append((input, group, rank, group_boxed, dim, sizes))
        return "gathered"

    monkeypatch.setattr(distributed_ops, "_allgather", fake_allgather)

    mapping = _FakeMapping()
    comm = AllGatherReduceScatter(mapping)
    comm.abort_and_reinit([4, 6, 7])

    # Transport recovery alone cannot recreate a TP shard. A normal model-wide
    # call using the original Mapping must fail instead of returning fewer
    # output features. AllGatherReduceScatter uses its explicit survivor mapping below.
    with pytest.raises(RuntimeError, match="statically sharded|redistribute"):
        distributed_ops.allgather("input", mapping, dim=0, sizes=[5, 7, 11, 13])
    assert observed == []


def test_reconfiguration_rejects_static_sharded_helix_alltoall(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    AllGatherReduceScatter(_FakeMapping()).abort_and_reinit([4, 6, 7])
    calls = []

    def fake_alltoall(inputs, group, num_lists):
        calls.append((list(inputs), list(group), num_lists))
        return ["output"]

    monkeypatch.setattr(
        distributed_ops.torch.ops.trtllm,
        "alltoall_helix",
        fake_alltoall,
        raising=False,
    )

    inputs = [allgather_rs_module.torch.tensor([rank]) for rank in range(4)]
    with pytest.raises(RuntimeError, match="statically sharded|redistribute"):
        distributed_ops.alltoall_helix(inputs, [4, 5, 6, 7])
    assert calls == []


def test_reconfiguration_is_shared_and_idempotent_across_moe_layers(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append((list(group), list(active_group))),
    )
    first_layer = AllGatherReduceScatter(_FakeMapping())
    second_layer = AllGatherReduceScatter(_FakeMapping())

    first_layer.abort_and_reinit([4, 6, 7], generation=1)
    second_layer.abort_and_reinit([4, 6, 7], generation=1)

    assert calls == [([4, 5, 6, 7], [4, 6, 7])]
    _assert_active_mapping(second_layer._active_mapping, [4, 6, 7], 1)


def test_reconfiguration_transaction_serializes_same_target(monkeypatch):
    entered_native = threading.Event()
    release_native = threading.Event()
    calls = []

    def native_rebuild(group, active_group, rendezvous_id):
        calls.append((list(group), list(active_group)))
        entered_native.set()
        assert release_native.wait(timeout=5)

    _patch_reinit_op(monkeypatch, native_rebuild)
    first_layer = AllGatherReduceScatter(_FakeMapping())
    second_layer = AllGatherReduceScatter(_FakeMapping())
    errors = []

    def reconfigure(layer):
        try:
            layer.abort_and_reinit([4, 6, 7], generation=1)
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    first = threading.Thread(target=reconfigure, args=(first_layer,))
    second = threading.Thread(target=reconfigure, args=(second_layer,))
    first.start()
    assert entered_native.wait(timeout=5)
    second.start()
    release_native.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == [([4, 5, 6, 7], [4, 6, 7])]


def test_shared_generation_deduplicates_and_advances_same_membership_rebuild(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append(
            (list(group), list(active_group), rendezvous_id)
        ),
    )

    comm = AllGatherReduceScatter(_FakeMapping())
    comm.abort_and_reinit([4, 5, 6, 7], generation=1)
    comm.abort_and_reinit([4, 5, 6, 7], generation=1)
    comm.abort_and_reinit([4, 5, 6, 7], generation=2)

    assert calls == [
        ([4, 5, 6, 7], [4, 5, 6, 7], 3),
        ([4, 5, 6, 7], [4, 5, 6, 7], 4),
    ]


def test_same_membership_recovery_requires_shared_generation(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append((list(group), list(active_group))),
    )

    comm = AllGatherReduceScatter(_FakeMapping())
    with pytest.raises(ValueError, match="generation is required"):
        comm.abort_and_reinit([4, 5, 6, 7])

    assert calls == []


@pytest.mark.parametrize("generation", [-1, 1.5, (1 << 63) - 2])
def test_raw_recovery_generation_must_fit_reserved_torch_int_range(monkeypatch, generation):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append(
            (group, active_group, rendezvous_id)
        ),
    )

    with pytest.raises(ValueError, match="generation must be a nonnegative integer"):
        AllGatherReduceScatter(_FakeMapping()).abort_and_reinit([4, 6, 7], generation=generation)

    assert calls == []


def test_generation_rejects_conflicting_raw_recovery_targets(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append((list(group), list(active_group))),
    )

    comm = AllGatherReduceScatter(_FakeMapping())
    comm.abort_and_reinit([4, 6, 7], generation=9)
    with pytest.raises(RuntimeError, match="conflicting.*generation 9"):
        comm.abort_and_reinit([6, 7], generation=9)

    assert calls == [([4, 5, 6, 7], [4, 6, 7])]


def test_fault_tolerance_control_path_requires_startup_mode(monkeypatch):
    monkeypatch.setattr(allgather_rs_module, "NCCL_FAULT_TOLERANCE_ENABLED", False)
    calls = []
    _patch_reinit_op(
        monkeypatch, lambda group, active_group, rendezvous_id: calls.append((group, active_group))
    )

    comm = AllGatherReduceScatter(_FakeMapping())
    with pytest.raises(RuntimeError, match="TLLM_FAULT_TOLERANCE_MODE=1"):
        comm.abort_and_reinit([4, 6, 7])

    assert calls == []


def test_default_off_collective_skips_survivor_registry(monkeypatch):
    monkeypatch.setattr(distributed_ops, "NCCL_FAULT_TOLERANCE_ENABLED", False)
    monkeypatch.setattr(distributed_ops, "mpi_disabled", lambda: False)

    def unexpected_registry_lookup(group, operation):
        raise AssertionError("default-off allgather consulted survivor state")

    observed = []
    monkeypatch.setattr(
        distributed_ops, "assert_nccl_group_not_reconfigured", unexpected_registry_lookup
    )
    monkeypatch.setattr(
        distributed_ops,
        "_allgather",
        lambda input, group, rank, group_boxed, dim, sizes: observed.append(
            (group, rank, group_boxed, dim, sizes)
        )
        or "gathered",
    )

    class ReferenceMapping:
        tp_group = [4, 5, 6, 7]
        tp_rank = 2

    mapping = ReferenceMapping()
    result = distributed_ops.allgather("input", mapping, dim=0, sizes=[1, 2, 3, 4])

    assert result == "gathered"
    assert observed == [([4, 5, 6, 7], 2, None, 0, [1, 2, 3, 4])]
    assert observed[0][0] is mapping.tp_group


def test_ft_auto_allreduce_uses_only_watchdog_managed_nccl(monkeypatch):
    monkeypatch.setattr(distributed_ops, "mpi_disabled", lambda: False)
    calls = []

    def fake_allreduce(**kwargs):
        calls.append(kwargs)
        return [kwargs["input"]]

    monkeypatch.setattr(
        distributed_ops.torch.ops.trtllm, "allreduce", fake_allreduce, raising=False
    )
    monkeypatch.setattr(
        distributed_ops,
        "get_allreduce_workspace",
        lambda mapping: (_ for _ in ()).throw(
            AssertionError("FT AUTO allocated a custom-allreduce workspace")
        ),
    )
    monkeypatch.setattr(
        distributed_ops.MNNVLAllReduce,
        "is_mnnvl",
        lambda mapping, dtype: (_ for _ in ()).throw(
            AssertionError("FT AUTO probed the MNNVL allreduce backend")
        ),
    )

    allreduce = distributed_ops.AllReduce(
        _FakeMapping(),
        strategy=distributed_ops.AllReduceStrategy.AUTO,
        dtype=distributed_ops.torch.float16,
    )
    input_tensor = distributed_ops.torch.ones(2)
    output = allreduce(input_tensor)

    assert output is input_tensor
    assert allreduce.workspace is None
    assert allreduce.mnnvl_allreduce is None
    assert len(calls) == 1
    assert calls[0]["strategy"] == distributed_ops.AllReduceStrategy.NCCL


@pytest.mark.parametrize(
    "strategy",
    [
        distributed_ops.AllReduceStrategy.UB,
        distributed_ops.AllReduceStrategy.ONESHOT,
        distributed_ops.AllReduceStrategy.MNNVL,
        distributed_ops.AllReduceStrategy.NCCL_SYMMETRIC,
        distributed_ops.AllReduceStrategy.SYMM_MEM,
    ],
)
def test_ft_allreduce_rejects_custom_transport_strategies(monkeypatch, strategy):
    monkeypatch.setattr(distributed_ops, "mpi_disabled", lambda: False)
    with pytest.raises(RuntimeError, match="watchdog-managed NCCL allreduce"):
        distributed_ops.AllReduce(_FakeMapping(), strategy=strategy)


def test_ft_tunable_allreduce_exposes_only_nccl_tactics():
    runner = torch_custom_ops.AllReduceRunner(
        4,
        [4, 5, 6, 7],
        int(distributed_ops.AllReduceFusionOp.NONE),
        1e-6,
        False,
    )
    inputs = [distributed_ops.torch.ones(2)]
    assert runner.get_valid_tactics(inputs, torch_custom_ops.OptimizationProfile()) == [
        distributed_ops.AllReduceStrategy.NCCL.value,
    ]


def test_ft_custom_allreduce_helpers_fail_before_workspace_allocation(monkeypatch):
    monkeypatch.setattr(distributed_ops, "mpi_disabled", lambda: False)
    monkeypatch.setattr(
        distributed_ops,
        "get_allreduce_workspace",
        lambda mapping: (_ for _ in ()).throw(
            AssertionError("unsupported FT helper allocated a workspace")
        ),
    )

    with pytest.raises(RuntimeError, match="MoEAllReduce.*custom collective"):
        distributed_ops.MoEAllReduce(_FakeMapping())
    with pytest.raises(RuntimeError, match="MiniMaxAllReduceRMS.*custom collective"):
        distributed_ops.MiniMaxAllReduceRMS(_FakeMapping())
    monkeypatch.setattr(
        distributed_ops.torch.ops.trtllm,
        "userbuffers_allreduce_finalize",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unsupported FT userbuffers op was invoked")
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="userbuffers allreduce"):
        distributed_ops.userbuffers_allreduce_finalize(distributed_ops.torch.ones(1))


def test_ft_disables_fused_nvfp4_gemm_allreduce(monkeypatch):
    class QuantMode:
        @staticmethod
        def has_nvfp4():
            return True

    class QuantConfig:
        layer_quant_mode = QuantMode()

    monkeypatch.setattr(linear_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(linear_module, "get_sm_version", lambda: 100)
    monkeypatch.setenv("TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED", "1")
    nvls_queries = []
    monkeypatch.setattr(
        linear_module, "ipc_nvls_supported", lambda: nvls_queries.append(True) or True
    )

    linear = Linear(
        128,
        64,
        bias=False,
        dtype=distributed_ops.torch.float16,
        mapping=_FakeMapping(),
        tensor_parallel_mode=TensorParallelMode.ROW,
        quant_config=QuantConfig(),
        skip_create_weights_in_init=True,
        enable_gemm_allreduce_fusion=True,
    )

    assert not linear.use_fused_gemm_allreduce
    assert nvls_queries == []
    with pytest.raises(RuntimeError, match="fused GEMM allreduce"):
        linear.apply_linear_allreduce(None, None)


def test_active_mapping_and_sizes_are_reused_until_membership_changes(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append((list(group), list(active_group))),
    )
    comm = AllGatherReduceScatter(_FakeMapping())
    original_mapping = comm._active_mapping
    original_sizes = [5, 7, 11, 13]

    assert comm._active_sizes(original_sizes) is original_sizes
    assert comm._active_mapping is original_mapping

    comm.abort_and_reinit([4, 6, 7])
    survivor_mapping = comm._active_mapping
    assert comm._active_sizes(original_sizes) == [5, 11, 13]
    assert comm._active_mapping is survivor_mapping
    comm._refresh_active_mapping()
    assert comm._active_mapping is survivor_mapping


def test_process_group_backend_skips_raw_nccl_membership_refresh(monkeypatch):
    monkeypatch.setattr(allgather_rs_module, "mpi_disabled", lambda: True)
    comm = AllGatherReduceScatter(_FakeMapping())
    monkeypatch.setattr(
        allgather_rs_module,
        "resolve_nccl_group",
        lambda group: (_ for _ in ()).throw(
            AssertionError("ProcessGroup dispatch consulted raw NCCL state")
        ),
    )
    observed = []
    monkeypatch.setattr(
        allgather_rs_module,
        "allgather",
        lambda inputs, mapping, *, dim, sizes: observed.append((mapping, sizes)) or inputs,
    )
    sizes = [5, 7, 11, 13]

    comm.dispatch("hidden", None, "slots", None, sizes)

    assert observed == [(comm.mapping, sizes)]


def test_reconfiguration_rejects_static_sharded_allreduce(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    mapping = _FakeMapping()
    comm = AllGatherReduceScatter(mapping)
    comm.abort_and_reinit([4, 6, 7])

    calls = []

    def fake_allreduce(**kwargs):
        calls.append(kwargs)
        return ("reduced",)

    allreduce = object.__new__(distributed_ops.AllReduce)
    allreduce.mapping = mapping
    allreduce._tp_group_tuple = tuple(mapping.tp_group)
    allreduce.strategy = distributed_ops.AllReduceStrategy.AUTO
    allreduce.workspace = None
    allreduce.symm_mem_allreduce = None
    allreduce.mnnvl_allreduce = None
    allreduce._disable_mpi = False
    allreduce.all_reduce_op = fake_allreduce

    tensor = allgather_rs_module.torch.ones(1)
    with pytest.raises(RuntimeError, match="statically sharded|redistribute"):
        allreduce.forward(tensor)
    assert calls == []
    assert not allreduce.uses_nccl_symmetric_memory_window()


@pytest.mark.parametrize(
    "active_ranks",
    [
        pytest.param([], id="empty"),
        pytest.param([4, 6, 6], id="duplicate"),
        pytest.param([-1, 6], id="negative"),
        pytest.param([4, 6, 8], id="outside_tp_group"),
        pytest.param([4, 5, 7], id="world_rank_missing"),
    ],
)
def test_abort_and_reinit_rejects_invalid_active_ranks(monkeypatch, active_ranks):
    calls = []
    _patch_reinit_op(
        monkeypatch, lambda group, active_group, rendezvous_id: calls.append((group, active_group))
    )
    comm = AllGatherReduceScatter(_FakeMapping())

    with pytest.raises(ValueError):
        comm.abort_and_reinit(active_ranks)

    assert calls == []


def test_abort_and_reinit_supports_mixed_tp_and_ep_topology(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch, lambda group, active_group, rendezvous_id: calls.append((group, active_group))
    )
    mapping = _FakeMapping(
        rank=6,
        tp_rank=2,
        moe_ep_size=2,
        moe_ep_rank=1,
        _group=[4, 5, 6, 7],
        _moe_group=[4, 6],
    )
    comm = AllGatherReduceScatter(mapping)

    # Global ranks are unambiguous even when the TP communicator spans several
    # MoE-EP slices. The higher-level health coordinator performs its local to
    # global translation before entering this API.
    comm.abort_and_reinit([4, 6, 7])

    assert calls == [([4, 5, 6, 7], [4, 6, 7])]


def test_abort_and_reinit_rejects_process_group_backend(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch, lambda group, active_group, rendezvous_id: calls.append((group, active_group))
    )
    monkeypatch.setattr(allgather_rs_module, "mpi_disabled", lambda: True, raising=False)
    comm = AllGatherReduceScatter(_FakeMapping())

    # The raw communicator cache is not used in Ray/ProcessGroup mode, and the
    # active mapping would otherwise delegate to the stale tp_group_pg.
    with pytest.raises(NotImplementedError, match="ProcessGroup|MPI-disabled"):
        comm.abort_and_reinit([4, 6, 7])

    assert calls == []


def test_check_async_error_returns_cleanly_when_watchdog_is_healthy(monkeypatch):
    queried_groups = []
    _patch_async_error_op(
        monkeypatch,
        lambda group: queried_groups.append(list(group)),
    )
    comm = AllGatherReduceScatter(_FakeMapping())

    comm.check_async_error()

    assert queried_groups == [[4, 5, 6, 7]]


def test_check_async_error_rejects_process_group_backend(monkeypatch):
    queried_groups = []
    _patch_async_error_op(
        monkeypatch,
        lambda group: queried_groups.append(list(group)),
    )
    monkeypatch.setattr(allgather_rs_module, "mpi_disabled", lambda: True, raising=False)
    comm = AllGatherReduceScatter(_FakeMapping())

    with pytest.raises(NotImplementedError, match="ProcessGroup|watchdog"):
        comm.check_async_error()

    assert queried_groups == []


def test_check_async_error_raises_stable_nccl_error_for_active_group(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    queried_groups = []

    def get_async_error(group):
        queried_groups.append(list(group))
        return "ncclSystemError: peer disconnected"

    _patch_async_error_op(monkeypatch, get_async_error)
    comm = AllGatherReduceScatter(_FakeMapping())
    comm.abort_and_reinit([4, 6, 7])

    with pytest.raises(
        RuntimeError,
        match="NCCL error: communicator was aborted: ncclSystemError: peer disconnected",
    ):
        comm.check_async_error()

    assert queried_groups == [[4, 6, 7]]


def test_other_layer_queries_shared_active_group_after_recovery(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    queried_groups = []
    _patch_async_error_op(monkeypatch, lambda group: queried_groups.append(list(group)))
    first_layer = AllGatherReduceScatter(_FakeMapping())
    second_layer = AllGatherReduceScatter(_FakeMapping())

    first_layer.abort_and_reinit([4, 6, 7])
    second_layer.check_async_error()

    assert queried_groups == [[4, 6, 7]]
    _assert_active_mapping(second_layer._active_mapping, [4, 6, 7], 1)


def test_reconfigured_row_linear_fails_before_using_incomplete_tp_shards(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    AllGatherReduceScatter(_FakeMapping()).abort_and_reinit([6, 7])
    calls = []

    class QuantMethod:
        supports_nccl_symmetric_memory_window_output = False

    class FakeRowLinear:
        tp_mode = TensorParallelMode.ROW
        mapping = _FakeMapping(rank=6, tp_rank=2)
        _tp_group_tuple = tuple(mapping.tp_group)
        tp_rank = 2
        tp_size = 4
        use_fused_gemm_allreduce = True
        reduce_output = True
        bias = "bias"
        quant_method = QuantMethod()
        lora = None

        def apply_linear_allreduce(self, *args, **kwargs):
            raise AssertionError("stale fused GEMM-allreduce path was used")

        def _maybe_fuse_bias_into_allreduce(self, bias, all_reduce_params):
            return False

        def apply_linear(self, input, bias, lora_params, layer_idx):
            calls.append(("gemm", input, bias))
            return "gemm-output"

        def all_reduce(self, output, *, all_reduce_params):
            calls.append(("allreduce", output))
            return "reduced"

    with pytest.raises(RuntimeError, match="statically sharded|redistribute"):
        Linear.forward(FakeRowLinear(), "input")
    assert calls == []


def test_reconfigured_column_linear_without_gather_fails_before_gemm(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    AllGatherReduceScatter(_FakeMapping()).abort_and_reinit([6, 7])
    calls = []

    class FakeColumnLinear:
        tp_mode = TensorParallelMode.COLUMN
        mapping = _FakeMapping(rank=6, tp_rank=2)
        _tp_group_tuple = tuple(mapping.tp_group)
        tp_size = 4
        gather_output = False
        bias = None

        def apply_linear(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "partial-output"

    with pytest.raises(RuntimeError, match="statically sharded|redistribute"):
        Linear.forward(FakeColumnLinear(), "input")
    assert calls == []


def test_dispatch_and_combine_filter_sizes_to_active_ranks(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    gather_calls = []
    scatter_calls = []

    def fake_allgather(inputs, mapping, *, dim, sizes):
        gather_calls.append((inputs, mapping, dim, sizes))
        return inputs

    def fake_reducescatter(input_tensor, mapping, *, dim, sizes):
        scatter_calls.append((input_tensor, mapping, dim, sizes))
        return "combined"

    monkeypatch.setattr(allgather_rs_module, "allgather", fake_allgather)
    monkeypatch.setattr(allgather_rs_module, "reducescatter", fake_reducescatter)

    mapping = _FakeMapping()
    comm = AllGatherReduceScatter(mapping)
    comm.abort_and_reinit([4, 6, 7])

    inputs = ["hidden", None, "slots", "scales"]
    assert comm.dispatch(*inputs, all_rank_num_tokens=[5, 7, 11, 13]) == tuple(inputs)
    assert comm.combine("moe-output") == "combined"

    assert len(gather_calls) == 1
    gathered_inputs, gather_mapping, gather_dim, gather_sizes = gather_calls[0]
    assert gathered_inputs == inputs
    assert gather_dim == 0
    assert gather_sizes == [5, 11, 13]
    _assert_active_mapping(gather_mapping, [4, 6, 7], 1)

    assert len(scatter_calls) == 1
    scatter_input, scatter_mapping, scatter_dim, scatter_sizes = scatter_calls[0]
    assert scatter_input == "moe-output"
    assert scatter_dim == 0
    assert scatter_sizes == [5, 11, 13]
    _assert_active_mapping(scatter_mapping, [4, 6, 7], 1)

    # Reconfiguration does not mutate Mapping, but the shared raw-NCCL
    # membership registry reroutes other collectives that consume it.
    assert mapping.tp_group == [4, 5, 6, 7]
    assert mapping.tp_rank == 2


def test_dp_padding_keeps_sizes_none_after_reconfiguration(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    observed = []

    def fake_allgather(inputs, mapping, *, dim, sizes):
        observed.append((mapping, sizes))
        return inputs

    monkeypatch.setattr(allgather_rs_module, "allgather", fake_allgather)

    comm = AllGatherReduceScatter(_FakeMapping())
    comm.abort_and_reinit([4, 6, 7])
    comm.dispatch("hidden", None, "slots", None, [5, 7, 11, 13], use_dp_padding=True)

    assert len(observed) == 1
    active_mapping, sizes = observed[0]
    assert sizes is None
    _assert_active_mapping(active_mapping, [4, 6, 7], 1)


def test_dp_padding_refreshes_mapping_after_another_layer_recovers(monkeypatch):
    _patch_reinit_op(monkeypatch, lambda group, active_group, rendezvous_id: None)
    observed = []

    def fake_allgather(inputs, mapping, *, dim, sizes):
        observed.append((mapping, sizes))
        return inputs

    monkeypatch.setattr(allgather_rs_module, "allgather", fake_allgather)
    recovering_layer = AllGatherReduceScatter(_FakeMapping())
    stale_layer = AllGatherReduceScatter(_FakeMapping())

    recovering_layer.abort_and_reinit([4, 6, 7])
    stale_layer.dispatch("hidden", None, "slots", None, [5, 7, 11, 13], use_dp_padding=True)

    assert len(observed) == 1
    active_mapping, sizes = observed[0]
    assert sizes is None
    _assert_active_mapping(active_mapping, [4, 6, 7], 1)


def test_later_reconfiguration_replaces_active_rank_filter(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append((list(group), list(active_group))),
    )
    observed = []

    def fake_allgather(inputs, mapping, *, dim, sizes):
        observed.append((mapping, sizes))
        return inputs

    monkeypatch.setattr(allgather_rs_module, "allgather", fake_allgather)

    comm = AllGatherReduceScatter(_FakeMapping())
    comm.abort_and_reinit([4, 6, 7])
    comm.abort_and_reinit([6, 7])
    comm.dispatch("hidden", None, "slots", None, [5, 7, 11, 13])

    assert calls == [
        ([4, 5, 6, 7], [4, 6, 7]),
        ([4, 6, 7], [6, 7]),
    ]
    assert len(observed) == 1
    active_mapping, sizes = observed[0]
    assert sizes == [11, 13]
    _assert_active_mapping(active_mapping, [6, 7], 0)


def test_reconfiguration_cannot_reactivate_a_removed_rank(monkeypatch):
    calls = []
    _patch_reinit_op(
        monkeypatch,
        lambda group, active_group, rendezvous_id: calls.append((list(group), list(active_group))),
    )

    comm = AllGatherReduceScatter(_FakeMapping())
    comm.abort_and_reinit([4, 6, 7])

    with pytest.raises(ValueError, match="reactivate|subset"):
        comm.abort_and_reinit([4, 5, 6, 7])

    assert calls == [([4, 5, 6, 7], [4, 6, 7])]


def test_failed_reinit_does_not_commit_active_rank_filter(monkeypatch):
    def fail_reinit(group, active_group, rendezvous_id):
        raise RuntimeError("NCCL communicator rebuild failed")

    _patch_reinit_op(monkeypatch, fail_reinit)
    comm = AllGatherReduceScatter(_FakeMapping())
    old_local_ranks = comm._active_local_ranks
    old_global_group = comm._active_global_group
    old_mapping = comm._active_mapping

    with pytest.raises(RuntimeError, match="NCCL communicator rebuild failed"):
        comm.abort_and_reinit([4, 6, 7])

    # The native call aborts the old communicator before initializing its
    # replacement, so transport is terminal after this failure. The only
    # transactional guarantee is that Python membership metadata does not
    # claim the failed replacement was committed.
    assert comm._active_local_ranks is old_local_ranks
    assert comm._active_global_group is old_global_group
    assert comm._active_mapping is old_mapping
