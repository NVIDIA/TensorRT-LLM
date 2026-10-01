# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging.version import Version

from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard, nccl_ep_utils
from tensorrt_llm._torch.moe.fused_moe.communication import communication_factory
from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided as one_sided_module
from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_two_sided as two_sided_module
from tensorrt_llm._torch.moe.fused_moe.communication import (
    nvlink_two_sided_flashinfer as flashinfer_module,
)
from tensorrt_llm._torch.moe.fused_moe.communication.allgather_reducescatter import (
    AllGatherReduceScatter,
)
from tensorrt_llm._torch.moe.fused_moe.communication.nccl_ep import NcclEP


def _make_model_config(
    act_dtype: torch.dtype = torch.bfloat16,
    moe_max_num_tokens: int | None = 1024,
):
    mapping = SimpleNamespace(
        enable_attention_dp=True,
        dp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
        moe_ep_rank=0,
        has_cp_helix=Mock(return_value=False),
    )
    return SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(hidden_size=4096),
        torch_dtype=act_dtype,
        quant_config=None,
        max_num_tokens=1024,
        moe_max_num_tokens=moe_max_num_tokens,
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
        moe_load_balancer=None,
    )


def _strategy_unavailable(*args, **kwargs):
    raise RuntimeError("strategy unavailable")


def _mock_nccl_ep_v02(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model the v0.2 runtime contract required by these factory tests."""
    monkeypatch.setattr(communication_factory, "nccl_ep_supports_version", lambda _: True)
    monkeypatch.setattr(communication_factory, "nccl_ep_supports_group_timeout", lambda: True)


def test_create_strategy_rejects_invalid_comm_timeout_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection treats constructor errors as "unavailable", so validation must come first."""
    guard = moe_comm_timeout_guard.MoECommTimeoutGuard(
        environ={moe_comm_timeout_guard.SERVING_TIMEOUT_ENV: "soon"}
    )
    monkeypatch.setattr(moe_comm_timeout_guard, "_DEFAULT_GUARD", guard)
    one_sided = Mock(side_effect=AssertionError("backend construction reached"))
    monkeypatch.setattr(communication_factory, "NVLinkOneSided", one_sided)

    with pytest.raises(ValueError, match=moe_comm_timeout_guard.SERVING_TIMEOUT_ENV):
        communication_factory.CommunicationFactory.create_strategy(
            _make_model_config(),
            num_experts=32,
            num_slots=32,
            top_k=8,
            expert_size_per_partition=16,
            hidden_size=4096,
        )

    one_sided.assert_not_called()


@pytest.mark.parametrize("use_flashinfer", [False, True])
def test_forced_two_sided_rejects_helix_before_workspace_allocation(
    monkeypatch: pytest.MonkeyPatch,
    use_flashinfer: bool,
) -> None:
    model_config = _make_model_config()
    model_config.mapping.has_cp_helix = Mock(return_value=True)
    monkeypatch.setattr(
        communication_factory.NVLinkTwoSided,
        "is_platform_supported",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        communication_factory.NVLinkTwoSidedFlashinfer,
        "is_platform_supported",
        Mock(return_value=True),
    )
    native_initialize = Mock(side_effect=AssertionError("native allocation reached"))
    flashinfer_symbols = Mock(side_effect=AssertionError("FlashInfer allocation reached"))
    monkeypatch.setattr(two_sided_module.MnnvlMemory, "initialize", native_initialize)
    monkeypatch.setattr(flashinfer_module, "_flashinfer_mnnvl", flashinfer_symbols)

    with pytest.raises(ValueError, match="does not support Helix context parallelism"):
        communication_factory.CommunicationFactory._create_forced_method(
            "NVLINK_TWO_SIDED",
            model_config,
            num_experts=32,
            num_slots=32,
            top_k=8,
            expert_size_per_partition=16,
            payload_in_workspace=False,
            alltoall_result_do_sum=True,
            use_flashinfer=use_flashinfer,
            hidden_size=4096,
        )

    native_initialize.assert_not_called()
    flashinfer_symbols.assert_not_called()


def test_forced_one_sided_rejects_helix_before_workspace_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = _make_model_config()
    model_config.mapping.world_size = model_config.mapping.moe_ep_size
    model_config.mapping.has_cp_helix = Mock(return_value=True)
    monkeypatch.setattr(
        communication_factory.NVLinkOneSided,
        "is_platform_supported",
        Mock(return_value=True),
    )
    native_initialize = Mock(side_effect=AssertionError("native allocation reached"))
    monkeypatch.setattr(one_sided_module.MnnvlMemory, "initialize", native_initialize)

    with pytest.raises(ValueError, match="does not support Helix context parallelism"):
        communication_factory.CommunicationFactory._create_forced_method(
            "NVLINK_ONE_SIDED",
            model_config,
            num_experts=32,
            num_slots=32,
            top_k=8,
            expert_size_per_partition=16,
            payload_in_workspace=False,
            alltoall_result_do_sum=True,
            use_flashinfer=False,
            hidden_size=4096,
        )

    native_initialize.assert_not_called()


def test_auto_selection_rejects_unrepurposed_helix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = _make_model_config()
    model_config.mapping.world_size = model_config.mapping.moe_ep_size
    model_config.mapping.has_cp_helix = Mock(return_value=True)
    monkeypatch.setattr(
        communication_factory.NVLinkOneSided,
        "is_platform_supported",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        communication_factory.NVLinkTwoSided,
        "is_platform_supported",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        communication_factory.NVLinkTwoSidedFlashinfer,
        "is_platform_supported",
        Mock(return_value=True),
    )
    native_initialize = Mock(side_effect=AssertionError("native allocation reached"))
    flashinfer_symbols = Mock(side_effect=AssertionError("FlashInfer allocation reached"))
    monkeypatch.setattr(one_sided_module.MnnvlMemory, "initialize", native_initialize)
    monkeypatch.setattr(flashinfer_module, "_flashinfer_mnnvl", flashinfer_symbols)
    with pytest.raises(ValueError, match="repurpose_helix_cp_to_tp"):
        communication_factory.CommunicationFactory.create_strategy(
            model_config,
            num_experts=32,
            num_slots=32,
            top_k=8,
            expert_size_per_partition=16,
            hidden_size=4096,
        )

    native_initialize.assert_not_called()
    flashinfer_symbols.assert_not_called()


def _install_failing_nccl_module(monkeypatch: pytest.MonkeyPatch, error: BaseException):
    def fail_get_lib_version():
        raise error

    monkeypatch.setattr(nccl_ep_utils, "_nccl_ep_installed", None)
    fake_ep = SimpleNamespace(get_lib_version=fail_get_lib_version)
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(ep=fake_ep))
    monkeypatch.setitem(sys.modules, "nccl.ep", fake_ep)


def test_nccl_ep_installed_handles_runtime_probe_failure(monkeypatch: pytest.MonkeyPatch):
    _install_failing_nccl_module(monkeypatch, RuntimeError("missing libnccl_ep"))

    assert nccl_ep_utils.is_nccl_ep_installed() is False
    assert nccl_ep_utils._nccl_ep_installed is False


@pytest.mark.parametrize(
    ("installed_version", "supported"),
    [("0.1", False), ("0.2", True), ("0.2.1", True)],
)
def test_nccl_ep_version_gate(
    monkeypatch: pytest.MonkeyPatch,
    installed_version: str,
    supported: bool,
):
    """NCCL-EP v0.2 features must never silently enable on an older wheel."""
    fake_ep = SimpleNamespace(get_lib_version=lambda: Version(installed_version))
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(ep=fake_ep))
    monkeypatch.setitem(sys.modules, "nccl.ep", fake_ep)

    assert nccl_ep_utils.nccl_ep_supports_version("0.2") is supported


def test_nccl_ep_invalid_version_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    """A malformed version must allow CommunicationFactory to fall back."""
    fake_ep = SimpleNamespace(get_lib_version=lambda: "not-a-version")
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(ep=fake_ep))
    monkeypatch.setitem(sys.modules, "nccl.ep", fake_ep)

    assert nccl_ep_utils.get_nccl_ep_version() is None
    assert not nccl_ep_utils.nccl_ep_supports_version("0.2")

    quant_config = SimpleNamespace(layer_quant_mode=SimpleNamespace(has_any_quant=lambda **_: True))
    assert (
        communication_factory.CommunicationFactory._get_nccl_ep_unavailable_reason(
            torch.bfloat16, quant_config, 32, 4096, 1024, 1024, 8
        )
        == "NcclEP v0.1 does not support quantized MoE communication."
    )


@pytest.mark.parametrize(("has_setter", "supported"), [(False, False), (True, True)])
def test_nccl_ep_group_timeout_probe(
    monkeypatch: pytest.MonkeyPatch,
    has_setter: bool,
    supported: bool,
):
    methods = {"set_timeout_ns": lambda self, timeout_ns: None} if has_setter else {}
    fake_ep = SimpleNamespace(Group=type("Group", (), methods))
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(ep=fake_ep))
    monkeypatch.setitem(sys.modules, "nccl.ep", fake_ep)

    assert nccl_ep_utils.nccl_ep_supports_group_timeout() is supported


def test_nccl_ep_without_runtime_group_timeout_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    """A wheel without the carried nccl-extensions patch cannot follow warmup timeouts."""
    _mock_nccl_ep_v02(monkeypatch)
    monkeypatch.setattr(communication_factory, "is_nccl_ep_installed", lambda: True)
    monkeypatch.setattr(communication_factory, "nccl_ep_supports_group_timeout", lambda: False)

    reason = communication_factory.CommunicationFactory._get_nccl_ep_unavailable_reason(
        torch.bfloat16, None, 32, 4096, 1024, 1024, 8
    )

    assert reason is not None and "set_timeout_ns" in reason


class _FakeNcclEP:
    def __init__(
        self,
        mapping,
        num_slots,
        hidden_size,
        max_num_tokens,
        moe_max_num_tokens,
        top_k=8,
        quant_config=None,
        use_low_precision_combine=False,
    ):
        self.mapping = mapping
        self.num_slots = num_slots
        self.hidden_size = hidden_size
        self.max_num_tokens = max_num_tokens
        self.moe_max_num_tokens = moe_max_num_tokens
        self.top_k = top_k
        self.use_low_precision_combine = use_low_precision_combine


@pytest.mark.parametrize(
    ("act_dtype", "moe_max_num_tokens", "match"),
    [
        (torch.float16, 1024, "act_dtype=torch.bfloat16"),
    ],
)
def test_forced_nccl_ep_validates_preconditions(
    act_dtype: torch.dtype,
    moe_max_num_tokens: int | None,
    match: str,
):
    model_config = _make_model_config(act_dtype, moe_max_num_tokens)

    with pytest.raises(ValueError, match=match):
        communication_factory.CommunicationFactory._create_forced_method(
            "NCCL_EP",
            model_config,
            num_experts=32,
            num_slots=32,
            top_k=8,
            expert_size_per_partition=16,
            payload_in_workspace=False,
            alltoall_result_do_sum=True,
            use_flashinfer=False,
            hidden_size=4096,
        )


def test_forced_nccl_ep_does_not_apply_host_smem_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    """NCCL-EP v0.2 owns LL kernel and shared-memory selection."""
    _mock_nccl_ep_v02(monkeypatch)
    model_config = _make_model_config(torch.bfloat16, 1024)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: pytest.fail("NCCL-EP factory must not probe host SMEM limits"),
    )
    monkeypatch.setattr(communication_factory, "NcclEP", _FakeNcclEP)

    strategy = communication_factory.CommunicationFactory._create_forced_method(
        "NCCL_EP",
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        payload_in_workspace=False,
        alltoall_result_do_sum=True,
        use_flashinfer=False,
        hidden_size=4096,
    )

    assert isinstance(strategy, _FakeNcclEP)


def test_forced_nccl_ep_allows_missing_moe_max_num_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_nccl_ep_v02(monkeypatch)
    model_config = _make_model_config(torch.bfloat16, None)
    monkeypatch.setattr(communication_factory, "NcclEP", _FakeNcclEP)

    strategy = communication_factory.CommunicationFactory._create_forced_method(
        "NCCL_EP",
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        payload_in_workspace=False,
        alltoall_result_do_sum=True,
        use_flashinfer=False,
        hidden_size=4096,
    )

    assert isinstance(strategy, _FakeNcclEP)
    assert strategy.max_num_tokens == model_config.max_num_tokens
    assert strategy.moe_max_num_tokens is None


def test_nccl_ep_downgrades_unsupported_low_precision_combine(
    monkeypatch: pytest.MonkeyPatch,
):
    """Strategy selection may request combine quantization for BF16 workloads."""
    monkeypatch.setattr(nccl_ep_utils, "is_nccl_ep_installed", lambda: True)

    monkeypatch.setattr(nccl_ep_utils, "get_nccl_ep_context", Mock())

    strategy = NcclEP(
        mapping=SimpleNamespace(moe_ep_size=2, moe_ep_rank=0),
        num_slots=32,
        hidden_size=4096,
        use_low_precision_combine=True,
    )

    assert not strategy.use_low_precision_combine


def test_nccl_ep_window_refusal_returns_persistent_dispatch_buffer(
    monkeypatch: pytest.MonkeyPatch,
):
    """A rejected NCCL window must not replace the buffer written by dispatch."""

    class FakeTensor:
        def __init__(self, tensor, **_):
            self.tensor = tensor

    class FakeHandle:
        def dispatch(self, _, outputs, **__):
            assert outputs is context.dispatch_outputs
            context.output_tokens_buf.fill_(7)

    fake_ep = SimpleNamespace(
        DispatchInputs=lambda **kwargs: SimpleNamespace(**kwargs),
        DispatchOutputs=lambda **kwargs: SimpleNamespace(**kwargs),
        Tensor=FakeTensor,
    )
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(ep=fake_ep))
    monkeypatch.setitem(sys.modules, "nccl.ep", fake_ep)

    persistent_tokens = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    context = SimpleNamespace(
        get_stream=lambda: None,
        output_tokens_buf=persistent_tokens,
        recv_topk_idx_buf=torch.full((1, 1, 1), -1, dtype=torch.int32),
        recv_topk_weights_buf=torch.zeros(1, 1, 1),
        recv_topk_weights_nd=object(),
        recv_topk_idx_nd=object(),
        scales_nd=None,
        topk_idx_dtype=torch.int32,
        kernel_resets_recv_topk_idx=True,
        kernel_writes_global_ids=True,
        zerocopy_enabled=True,
        dispatch_outputs=SimpleNamespace(tokens=object()),
        layout=object(),
        dispatch_layout_info=object(),
        dispatch_config=object(),
    )
    comm = NcclEP.__new__(NcclEP)
    comm.mapping = SimpleNamespace(moe_ep_group=[0])
    comm.ep_size = comm.ep_rank = 1
    comm.num_local_experts = comm.max_top_k = 1
    comm.max_tokens_per_rank = comm.max_recv_tokens = 1
    comm.hidden_size = 4
    comm.use_fp8 = comm.uses_internal_fp8_dispatch = False
    comm.use_external_fp8 = comm.use_external_nvfp4 = False
    comm._dispatch_state = {}
    monkeypatch.setattr(comm, "_get_context", lambda: context)
    monkeypatch.setattr(comm, "_setup_handle", lambda *_: FakeHandle())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fallback_tokens = torch.zeros_like(persistent_tokens)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "allocate_output_with_nccl_window",
        lambda *_: (fallback_tokens, 0, 0),
    )

    recv_hs, *_ = comm.dispatch(
        torch.zeros(1, 4, dtype=torch.bfloat16),
        None,
        torch.zeros(1, 1, dtype=torch.int32),
        torch.ones(1, 1),
        [1],
    )

    assert torch.equal(recv_hs, torch.full_like(recv_hs, 7))


def test_forced_nccl_ep_forwards_low_precision_combine(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_nccl_ep_v02(monkeypatch)
    model_config = _make_model_config(torch.bfloat16, 1024)
    model_config.use_low_precision_moe_combine = True
    monkeypatch.setattr(communication_factory, "NcclEP", _FakeNcclEP)

    strategy = communication_factory.CommunicationFactory._create_forced_method(
        "NCCL_EP",
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        payload_in_workspace=False,
        alltoall_result_do_sum=True,
        use_flashinfer=False,
        hidden_size=4096,
    )

    assert isinstance(strategy, _FakeNcclEP)
    assert strategy.use_low_precision_combine


def test_auto_selection_uses_nccl_ep_with_missing_moe_max_num_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_nccl_ep_v02(monkeypatch)
    model_config = _make_model_config(torch.bfloat16, None)

    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")
    monkeypatch.setattr(communication_factory, "NcclEP", _FakeNcclEP)

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, _FakeNcclEP)
    assert strategy.max_num_tokens == model_config.max_num_tokens
    assert strategy.moe_max_num_tokens is None


@pytest.mark.parametrize(
    ("act_dtype", "moe_max_num_tokens"),
    [
        (torch.float16, 1024),
    ],
)
def test_auto_selection_skips_nccl_ep_when_preconditions_fail(
    monkeypatch: pytest.MonkeyPatch,
    act_dtype: torch.dtype,
    moe_max_num_tokens: int | None,
):
    model_config = _make_model_config(act_dtype, moe_max_num_tokens)

    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("NcclEP should not be constructed")

    monkeypatch.setattr(communication_factory, "NcclEP", fail_if_called)

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, AllGatherReduceScatter)


def test_auto_selection_falls_back_when_nccl_probe_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = _make_model_config(torch.bfloat16, None)
    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")
    _install_failing_nccl_module(monkeypatch, OSError("missing native NCCL EP library"))

    strategy = communication_factory.CommunicationFactory.create_strategy(
        model_config,
        num_experts=32,
        num_slots=32,
        top_k=8,
        expert_size_per_partition=16,
        hidden_size=4096,
    )

    assert isinstance(strategy, AllGatherReduceScatter)


def test_nccl_ep_context_init_rejects_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = object.__new__(NcclEP)
    strategy._ctx = object.__new__(nccl_ep_utils.NcclEpContext)
    strategy._ctx.comm = Mock()
    strategy._ctx.initialized = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="context must be initialized before CUDA graph capture"):
        strategy._get_context()


def test_nccl_ep_handle_init_rejects_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = object.__new__(NcclEP)
    strategy._handle = None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def fail_create_handle(*args, **kwargs):
        raise AssertionError("create_handle should not run during CUDA graph capture")

    ctx = SimpleNamespace(
        ep_group=SimpleNamespace(create_handle=fail_create_handle),
        layout=object(),
    )

    with pytest.raises(
        RuntimeError, match="dispatch handle must be initialized before CUDA graph capture"
    ):
        strategy._setup_handle(ctx, object(), 0)


@pytest.fixture
def nccl_preflight_runtime(monkeypatch: pytest.MonkeyPatch):
    """Native communicator doubles; constructing a group is forbidden during preflight."""
    import tensorrt_llm._utils as utils

    class Layout(IntEnum):
        RANK_MAJOR = 1

    class NCCLError(Exception):
        pass

    monkeypatch.setitem(sys.modules, "nccl.bindings.nccl", SimpleNamespace(NCCLError=NCCLError))
    gin_types = SimpleNamespace(NONE=0, GDAKI=1, PROXY=2)
    comm = Mock(n_lsa_teams=2, gin_type=gin_types.GDAKI)
    core = SimpleNamespace(
        NcclGinType=gin_types,
        get_unique_id=Mock(return_value="unique-id"),
        Communicator=SimpleNamespace(init=Mock(return_value=comm)),
    )
    group_create = Mock(side_effect=AssertionError("EP group created before warmup"))
    ep = SimpleNamespace(Layout=Layout, Group=SimpleNamespace(create=group_create))
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(core=core, ep=ep))
    monkeypatch.setitem(sys.modules, "nccl.core", core)
    monkeypatch.setitem(sys.modules, "nccl.ep", ep)
    mpi = Mock()
    mpi.Get_rank.return_value = 0
    mpi.Get_size.return_value = 2
    mpi.allgather.side_effect = lambda error: [error, None]
    monkeypatch.setattr(utils, "mpi_comm", Mock(return_value=Mock(Split=Mock(return_value=mpi))))
    monkeypatch.setattr(nccl_ep_utils, "_ep_group_cache", {})
    monkeypatch.setattr(nccl_ep_utils, "_ep_group_refcounts", {})
    monkeypatch.setattr(nccl_ep_utils, "is_nccl_ep_installed", lambda: True)
    monkeypatch.setattr(nccl_ep_utils, "nccl_ep_supports_version", lambda _: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    loader = Mock()
    monkeypatch.setattr(nccl_ep_utils.ctypes, "CDLL", loader)
    monkeypatch.setenv("NCCL_NET_PLUGIN", "spcx")
    monkeypatch.delenv("NCCL_GIN_PLUGIN", raising=False)
    monkeypatch.delenv("NCCL_GIN_GPUNETIO_PATH", raising=False)
    return SimpleNamespace(
        comm=comm,
        core=core,
        mpi=mpi,
        loader=loader,
        group_create=group_create,
        mapping=SimpleNamespace(moe_ep_size=2, moe_ep_rank=0, pp_rank=0),
    )


@pytest.mark.parametrize(
    "single_domain,gin_type,plugin",
    [
        (True, 1, "spcx"),
        (False, 2, "spcx"),
        (False, 1, ""),
    ],
)
def test_nccl_ep_preflight_does_not_require_external_gpunetio(
    nccl_preflight_runtime,
    monkeypatch: pytest.MonkeyPatch,
    single_domain: bool,
    gin_type: int,
    plugin: str,
) -> None:
    runtime = nccl_preflight_runtime
    runtime.comm.n_lsa_teams = 1 if single_domain else 2
    runtime.comm.gin_type = gin_type
    runtime.loader.side_effect = OSError("GPUNetIO is absent")
    monkeypatch.setenv("NCCL_NET_PLUGIN", plugin)

    nccl_ep_utils._check_gin_dependencies(runtime.comm, runtime.mpi)

    runtime.loader.assert_not_called()
    runtime.mpi.allgather.assert_called_once_with(None)


@pytest.mark.parametrize("path", [None, "/plugin/libdoca_gpunetio_host.so"])
def test_nccl_ep_preflight_loads_gpunetio(
    nccl_preflight_runtime,
    monkeypatch: pytest.MonkeyPatch,
    path: str | None,
) -> None:
    runtime = nccl_preflight_runtime
    if path is not None:
        monkeypatch.setenv("NCCL_GIN_GPUNETIO_PATH", path)

    nccl_ep_utils._check_gin_dependencies(runtime.comm, runtime.mpi)

    runtime.loader.assert_called_once_with(path or "libdoca_gpunetio_host.so")
    runtime.mpi.allgather.assert_called_once_with(None)


@pytest.mark.parametrize("remote_failure", [False, True])
def test_nccl_ep_preflight_failure_cleans_up_on_every_rank(
    nccl_preflight_runtime,
    remote_failure: bool,
) -> None:
    runtime = nccl_preflight_runtime
    if remote_failure:
        runtime.mpi.allgather.side_effect = lambda _: [None, "cannot load GPUNetIO"]
    else:
        runtime.loader.side_effect = OSError("cannot load GPUNetIO")

    with pytest.raises(RuntimeError, match="EP rank [01].*GPUNetIO"):
        nccl_ep_utils.get_nccl_ep_context(
            runtime.mapping,
            32,
            128,
            4096,
            8,
            defer_initialization=True,
        )

    runtime.comm.finalize.assert_called_once()
    runtime.comm.destroy.assert_called_once()
    runtime.mpi.Free.assert_called_once()
    runtime.group_create.assert_not_called()
    assert not nccl_ep_utils._ep_group_cache
    assert not nccl_ep_utils._ep_group_refcounts


def test_nccl_ep_preflight_rejects_missing_gin(nccl_preflight_runtime) -> None:
    runtime = nccl_preflight_runtime
    runtime.comm.gin_type = runtime.core.NcclGinType.NONE

    with pytest.raises(RuntimeError, match="full connectivity"):
        nccl_ep_utils._check_gin_dependencies(runtime.comm, runtime.mpi)

    runtime.loader.assert_not_called()


@pytest.mark.parametrize("quantization", ["bf16", "fp8", "nvfp4"])
def test_nccl_ep_preflight_shares_context_and_defers_buffers(
    nccl_preflight_runtime,
    monkeypatch: pytest.MonkeyPatch,
    quantization: str,
) -> None:
    runtime = nccl_preflight_runtime
    quant_config = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(
            has_fp8_qdq=lambda: quantization == "fp8",
            has_nvfp4=lambda: quantization == "nvfp4",
        )
    )
    allocation = Mock(side_effect=AssertionError("tensor allocation during model construction"))
    monkeypatch.setattr(torch, "empty", allocation)
    first = NcclEP(runtime.mapping, 32, 4096, quant_config=quant_config)
    second = NcclEP(runtime.mapping, 32, 4096, quant_config=quant_config)

    assert first._ctx is second._ctx
    assert first._ctx.external_fp8 == (quantization == "fp8")
    assert first._ctx.external_nvfp4 == (quantization == "nvfp4")
    runtime.core.Communicator.init.assert_called_once()
    runtime.mpi.allgather.assert_called_once()
    runtime.group_create.assert_not_called()
    allocation.assert_not_called()

    # Dropping one layer must not release the shared communicator.
    first.destroy()
    runtime.comm.destroy.assert_not_called()
    second.destroy()
    second.destroy()
    runtime.comm.finalize.assert_called_once()
    runtime.comm.destroy.assert_called_once()
    runtime.mpi.Free.assert_called_once()
    assert not nccl_ep_utils._ep_group_cache


def test_nccl_ep_warmup_reuses_preflight_communicator(
    nccl_preflight_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = nccl_preflight_runtime
    initialize_group = Mock()
    monkeypatch.setattr(nccl_ep_utils.NcclEpContext, "_initialize_group", initialize_group)
    strategy = NcclEP(runtime.mapping, 32, 4096)

    context = strategy._get_context()
    assert strategy._get_context() is context
    assert context.comm is runtime.comm
    assert context.initialized
    initialize_group.assert_called_once()
    runtime.core.Communicator.init.assert_called_once()
    strategy.destroy()


def test_nccl_ep_group_init_failure_releases_preflight_resources(
    nccl_preflight_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = nccl_preflight_runtime
    monkeypatch.setattr(
        nccl_ep_utils.NcclEpContext,
        "_initialize_group",
        Mock(side_effect=RuntimeError("group creation failed")),
    )

    with pytest.raises(RuntimeError, match="group creation failed"):
        nccl_ep_utils.get_nccl_ep_context(runtime.mapping, 32, 128, 4096, 8)

    runtime.comm.destroy.assert_called_once()
    runtime.mpi.Free.assert_called_once()
    assert not nccl_ep_utils._ep_group_cache


@pytest.mark.parametrize("forced", [False, True])
def test_nccl_ep_missing_gpunetio_falls_back_only_for_auto_selection(
    nccl_preflight_runtime,
    monkeypatch: pytest.MonkeyPatch,
    forced: bool,
) -> None:
    runtime = nccl_preflight_runtime
    runtime.loader.side_effect = OSError("cannot load GPUNetIO")
    _mock_nccl_ep_v02(monkeypatch)
    monkeypatch.setattr(communication_factory, "is_nccl_ep_installed", lambda: True)
    monkeypatch.setattr(communication_factory, "NVLinkOneSided", _strategy_unavailable)
    monkeypatch.setattr(communication_factory, "NVLinkTwoSided", _strategy_unavailable)
    monkeypatch.setenv("TRTLLM_CAN_USE_DEEP_EP", "0")
    model_config = _make_model_config()
    model_config.mapping.pp_rank = 0
    kwargs = dict(
        num_experts=32, num_slots=32, top_k=8, expert_size_per_partition=16, hidden_size=4096
    )

    if forced:
        with pytest.raises(RuntimeError, match="NCCL_GIN_GPUNETIO_PATH"):
            communication_factory.CommunicationFactory._create_forced_method(
                "NCCL_EP",
                model_config,
                payload_in_workspace=False,
                alltoall_result_do_sum=True,
                use_flashinfer=False,
                **kwargs,
            )
    else:
        strategy = communication_factory.CommunicationFactory.create_strategy(
            model_config, **kwargs
        )
        assert isinstance(strategy, AllGatherReduceScatter)
    runtime.group_create.assert_not_called()
    runtime.comm.destroy.assert_called_once()


@pytest.mark.parametrize("native_error", [False, True])
def test_nccl_ep_topology_query_failure_is_collective(
    nccl_preflight_runtime,
    native_error: bool,
) -> None:
    runtime = nccl_preflight_runtime
    from nccl.bindings.nccl import NCCLError

    error = (
        NCCLError("incompatible NCCL runtime") if native_error else AttributeError("missing API")
    )

    class FailedQuery:
        @property
        def n_lsa_teams(self) -> int:
            raise error

    comm = FailedQuery()
    with pytest.raises(RuntimeError, match="cannot query NCCL communicator topology"):
        nccl_ep_utils._check_gin_dependencies(comm, runtime.mpi)
    runtime.mpi.allgather.assert_called_once()
    runtime.loader.assert_not_called()


def test_nccl_ep_communicator_init_failure_releases_mpi(nccl_preflight_runtime) -> None:
    runtime = nccl_preflight_runtime
    runtime.core.Communicator.init.side_effect = RuntimeError("communicator init failed")
    with pytest.raises(RuntimeError, match="communicator init failed"):
        nccl_ep_utils.get_nccl_ep_context(runtime.mapping, 32, 128, 4096, 8)
    runtime.mpi.Free.assert_called_once()
    assert not nccl_ep_utils._ep_group_cache
