# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for explicit, executor-wide lifecycle activation before publication."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_context_first_profile import _page_table

from tensorrt_llm._torch.disaggregation import kv_cache_transceiver as factory
from tensorrt_llm._torch.disaggregation import transceiver
from tensorrt_llm._torch.disaggregation.native import context_first_profile
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def activation(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Use real profile selection with controlled collective/MPI observations."""
    for name in (
        transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV,
        transceiver._FP4_MLA_OWNERSHIP_BRIDGE_ENV,
        "TRTLLM_NIXL_KVCACHE_BACKEND",
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP",
        "TRTLLM_DISAGG_LAYERWISE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    monkeypatch.setattr(transceiver, "use_pure_python_transfer_agent", lambda: False)
    monkeypatch.setattr(context_first_profile, "use_pure_python_transfer_agent", lambda: False)
    monkeypatch.setattr(
        context_first_profile, "build_page_table_from_manager", lambda _: _page_table()
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.is_disagg = True
    manager.dtype = DataType.BF16
    manager.kv_cache_type = CacheTypeCpp.SELF
    manager.layer_specs = None
    manager.max_seq_len = 128
    mapping = Mapping(world_size=2, rank=0, tp_size=2)
    config = CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="PYTHON", kv_cache_bounce_size_mb=0
    )
    dist = Mock(allgather=Mock(side_effect=lambda value: [value, value]))
    communicator = Mock()
    qualify = Mock(return_value=communicator)
    warning = Mock()
    monkeypatch.setattr(transceiver, "_retirement_executor_comm", qualify)
    monkeypatch.setattr(transceiver.logger, "warning", warning)
    return SimpleNamespace(
        mapping=mapping,
        manager=manager,
        config=config,
        dist=dist,
        communicator=communicator,
        qualify=qualify,
        warning=warning,
    )


def _select(activation: SimpleNamespace) -> tuple:
    return transceiver._select_lifecycle_ownership(
        activation.mapping, activation.manager, activation.config, activation.dist
    )


@pytest.mark.parametrize(
    "setting,legacy_fp4,fp4_layout,expected",
    [
        (None, False, False, (False, False)),
        ("0", False, False, (False, False)),
        ("1", False, False, (False, True)),
        (None, True, True, (True, False)),
        ("1", False, True, (True, False)),
        ("1", True, True, (True, False)),
    ],
)
def test_lifecycle_activation_preserves_explicit_policy(
    monkeypatch: pytest.MonkeyPatch,
    activation: SimpleNamespace,
    setting: str | None,
    legacy_fp4: bool,
    fp4_layout: bool,
    expected: tuple[bool, bool],
) -> None:
    """No-retry alone stays inactive; either opt-in uses its real profile validator."""
    if setting is not None:
        monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, setting)
    monkeypatch.setenv(transceiver._FP4_MLA_OWNERSHIP_BRIDGE_ENV, str(int(legacy_fp4)))
    if fp4_layout:
        activation.mapping = Mapping(world_size=2, rank=0, tp_size=2, enable_attention_dp=True)
        activation.manager.dtype = DataType.NVFP4
        activation.manager.kv_cache_type = CacheTypeCpp.SELFKONLY

    fp4, context_first, communicator = _select(activation)

    assert (fp4, context_first) == expected
    activation.dist.allgather.assert_called_once_with((*expected, None))
    if any(expected):
        assert communicator is activation.communicator
        activation.qualify.assert_called_once_with(activation.mapping, warn=False)
        activation.warning.assert_called_once()
    else:
        assert communicator is None
        activation.qualify.assert_not_called()
        activation.warning.assert_not_called()


@pytest.mark.parametrize("setting,legacy_fp4", [("true", False), ("", False), ("0", True)])
def test_invalid_lifecycle_policy_participates_in_rank_agreement(
    monkeypatch: pytest.MonkeyPatch,
    activation: SimpleNamespace,
    setting: str,
    legacy_fp4: bool,
) -> None:
    monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, setting)
    monkeypatch.setenv(transceiver._FP4_MLA_OWNERSHIP_BRIDGE_ENV, str(int(legacy_fp4)))

    with pytest.raises(ValueError, match="KV lifecycle initialization rejected: rank 0:"):
        _select(activation)

    activation.dist.allgather.assert_called_once()
    fp4, context_first, failure = activation.dist.allgather.call_args.args[0]
    assert (fp4, context_first) == (False, False)
    assert failure
    activation.qualify.assert_not_called()
    activation.warning.assert_not_called()


@pytest.mark.parametrize("failure_kind", ["unsupported", "retry", "no_mpi", "legacy_fp4"])
def test_requested_protection_rejects_local_failure_collectively(
    monkeypatch: pytest.MonkeyPatch, activation: SimpleNamespace, failure_kind: str
) -> None:
    monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, "1")
    if failure_kind == "unsupported":
        activation.manager.is_disagg = False
    elif failure_kind == "retry":
        monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "0")
    elif failure_kind == "no_mpi":
        activation.qualify.side_effect = ValueError("non-MPI executor is not qualified")
    else:
        monkeypatch.delenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV)
        monkeypatch.setenv(transceiver._FP4_MLA_OWNERSHIP_BRIDGE_ENV, "1")

    with pytest.raises(ValueError, match="KV lifecycle initialization rejected"):
        _select(activation)

    activation.dist.allgather.assert_called_once()
    assert activation.dist.allgather.call_args.args[0][2]
    activation.warning.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "remote_selection",
    [(False, False, None), (True, False, None), (False, False, "bad peer config")],
)
def test_all_ranks_reject_inconsistent_activation_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    activation: SimpleNamespace,
    rank: int,
    remote_selection: tuple,
) -> None:
    """An active non-lead rank cannot escape agreement with an inactive lead rank."""
    monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, "1")
    activation.mapping = Mapping(world_size=2, rank=rank, tp_size=2)
    activation.dist.allgather.side_effect = lambda local: (
        [local, remote_selection] if rank == 0 else [remote_selection, local]
    )
    worker = Mock()
    monkeypatch.setattr(transceiver, "TransferWorker", worker)
    monkeypatch.setattr(transceiver.KvCacheTransceiverV2, "_check_compatible", lambda _: None)

    with pytest.raises(ValueError, match="KV lifecycle"):
        transceiver.KvCacheTransceiverV2(
            activation.mapping, activation.dist, activation.manager, activation.config
        )

    activation.dist.allgather.assert_called_once_with((False, True, None))
    worker.assert_not_called()
    activation.warning.assert_not_called()


def test_inactive_rank_still_joins_activation_agreement(activation: SimpleNamespace) -> None:
    activation.dist.allgather.side_effect = lambda local: [local, (False, True, None)]

    with pytest.raises(ValueError, match="must agree on every executor rank"):
        _select(activation)

    activation.dist.allgather.assert_called_once_with((False, False, None))
    activation.qualify.assert_not_called()
    activation.warning.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_termination_warning_follows_agreement_on_lead_rank_only(
    monkeypatch: pytest.MonkeyPatch, activation: SimpleNamespace, rank: int
) -> None:
    monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, "1")
    activation.mapping = Mapping(world_size=2, rank=rank, tp_size=2)

    def agree(local: tuple) -> list[tuple]:
        activation.warning.assert_not_called()
        return [local, local]

    activation.dist.allgather.side_effect = agree
    assert _select(activation) == (False, True, activation.communicator)
    assert activation.warning.call_count == (1 if rank == 0 else 0)
    if rank == 0:
        assert "entire MPI job" in activation.warning.call_args.args[0]


@pytest.mark.parametrize("runtime", [None, "CPP"])
def test_explicit_lifecycle_policy_cannot_be_ignored_by_factory(
    monkeypatch: pytest.MonkeyPatch, activation: SimpleNamespace, runtime: str | None
) -> None:
    """No transceiver and a C++ transceiver cannot silently ignore explicit protection."""
    monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, "1")
    constructor = Mock()
    monkeypatch.setattr(factory, "BindKvCacheTransceiver", constructor)
    config = (
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime=runtime)
        if runtime is not None
        else None
    )
    with pytest.raises(ValueError, match="requires an enabled Python/NIXL transceiver"):
        factory.create_kv_cache_transceiver(None, None, Mock(), None, config)
    constructor.assert_not_called()


@pytest.mark.parametrize("setting", [None, "0", "invalid"])
def test_legacy_factory_policy_is_unchanged_without_opt_in(
    monkeypatch: pytest.MonkeyPatch, activation: SimpleNamespace, setting: str | None
) -> None:
    if setting is not None:
        monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, setting)
    constructor = Mock()
    monkeypatch.setattr(factory, "BindKvCacheTransceiver", constructor)
    monkeypatch.setattr(factory, "is_disagg_inflight_cancel_enabled", lambda: False)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP")
    if setting == "invalid":
        with pytest.raises(ValueError, match="must be 0 or 1"):
            factory.create_kv_cache_transceiver(None, None, Mock(), None, config)
        constructor.assert_not_called()
    else:
        assert (
            factory.create_kv_cache_transceiver(None, None, Mock(), None, config)
            is constructor.return_value
        )
        assert factory.create_kv_cache_transceiver(None, None, Mock(), None, None) is None
        constructor.assert_called_once()


def test_factory_leaves_python_policy_failure_to_rank_agreement(
    monkeypatch: pytest.MonkeyPatch, activation: SimpleNamespace
) -> None:
    """An invalid Python rank must join the collective instead of failing in the factory."""
    monkeypatch.setenv(transceiver._KV_LIFECYCLE_ENFORCEMENT_ENV, "invalid")
    constructor = Mock(side_effect=lambda *args: _select(activation))
    monkeypatch.setattr(transceiver, "KvCacheTransceiverV2", constructor)
    monkeypatch.setattr(factory, "is_disagg_inflight_cancel_enabled", lambda: False)
    with pytest.raises(ValueError, match="KV lifecycle initialization rejected"):
        factory.create_kv_cache_transceiver(
            activation.mapping, activation.dist, activation.manager, None, activation.config
        )
    constructor.assert_called_once()
    activation.dist.allgather.assert_called_once()
