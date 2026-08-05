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

import threading
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from tensorrt_llm._torch.disaggregation.capability_negotiation import (
    LifecycleNegotiationError,
    negotiate_generation_safe_lifecycle,
)
from tensorrt_llm._torch.disaggregation.lifecycle import LifecycleCapability
from tensorrt_llm._torch.disaggregation.protocol import (
    PROTOCOL_V1_REQUIRED_CAPABILITIES,
    ProtocolVersion,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MixedMambaHybridCacheManager
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle


def _mapping(*, attention_dp: bool = False):
    return SimpleNamespace(
        world_size=8,
        tp_size=4,
        pp_size=2,
        cp_size=1,
        enable_attention_dp=attention_dp,
    )


def _v1_contract(*, attention_dp: bool = False, generation_first: bool = False):
    return KvCacheTransceiverV2._build_lifecycle_capabilities(
        _mapping(attention_dp=attention_dp),
        protocol_version=ProtocolVersion.GENERATION_SAFE,
        bounce_transfer_enabled=False,
        supports_generation_first=generation_first,
    )


def _advertisement(*, attention_dp: bool = False, generation_first: bool = False):
    return KvCacheTransceiverV2._make_lifecycle_advertisement(
        lifecycle_contract=_v1_contract(
            attention_dp=attention_dp,
            generation_first=generation_first,
        ),
        instance_id=str(uuid4()),
        mapping=_mapping(attention_dp=attention_dp),
    )


def test_lifecycle_protocol_defaults_to_qualified_v0(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_DISAGG_LIFECYCLE_PROTOCOL_VERSION", raising=False)

    version = KvCacheTransceiverV2._configured_lifecycle_protocol_version()
    contract = KvCacheTransceiverV2._build_lifecycle_capabilities(
        _mapping(),
        protocol_version=version,
        bounce_transfer_enabled=False,
        supports_generation_first=False,
    )

    assert version is ProtocolVersion.QUALIFIED_LEGACY
    assert contract.qualified_legacy_mode


def test_lifecycle_protocol_v1_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_LIFECYCLE_PROTOCOL_VERSION", "1")

    assert (
        KvCacheTransceiverV2._configured_lifecycle_protocol_version()
        is ProtocolVersion.GENERATION_SAFE
    )


def test_invalid_lifecycle_protocol_configuration_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_DISAGG_LIFECYCLE_PROTOCOL_VERSION", "latest")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        KvCacheTransceiverV2._configured_lifecycle_protocol_version()


def test_python_v1_advertises_only_implemented_topology_contract() -> None:
    contract = _v1_contract(
        attention_dp=True,
        generation_first=True,
    )
    capability_names = frozenset(capability.value for capability in contract.supported)

    assert PROTOCOL_V1_REQUIRED_CAPABILITIES <= capability_names
    assert {
        LifecycleCapability.DIRECT_TRANSFER.value,
        LifecycleCapability.TENSOR_PARALLEL.value,
        LifecycleCapability.PIPELINE_PARALLEL.value,
        LifecycleCapability.ATTENTION_DATA_PARALLEL.value,
    } <= capability_names
    assert LifecycleCapability.GENERATION_FIRST in contract.supported
    assert not contract.qualified_legacy_mode


def test_generation_first_requires_explicit_exact_aux_support() -> None:
    disabled = _v1_contract(generation_first=False)
    enabled = _v1_contract(generation_first=True)

    assert LifecycleCapability.GENERATION_FIRST not in disabled.supported
    assert LifecycleCapability.GENERATION_FIRST in enabled.supported


def test_protocol_v1_refuses_allocator_without_snapshot_lease() -> None:
    with pytest.raises(RuntimeError, match="snapshot_and_lease"):
        KvCacheTransceiverV2._validate_generation_safe_allocator(
            SimpleNamespace(),
            _v1_contract(),
        )


def test_protocol_v1_refuses_cpp_v2_without_lease_capability() -> None:
    manager = SimpleNamespace(
        snapshot_and_lease=Mock(),
        supports_allocation_generation_leases=False,
    )

    with pytest.raises(RuntimeError, match="TLLM_KV_CACHE_MANAGER_V2_BACKEND=python"):
        KvCacheTransceiverV2._validate_generation_safe_allocator(
            manager,
            _v1_contract(),
        )


def test_protocol_v1_refuses_mixed_mamba_without_state_slot_leases() -> None:
    manager = object.__new__(MixedMambaHybridCacheManager)

    with pytest.raises(RuntimeError, match="independent Mamba state slots"):
        KvCacheTransceiverV2._validate_generation_safe_allocator(
            manager,
            _v1_contract(),
        )


@pytest.mark.parametrize(
    ("method_name", "operation"),
    [
        ("respond_and_send_async", "source send"),
        ("request_and_receive_sync", "synchronous receive"),
        ("request_and_receive_async", "asynchronous receive"),
    ],
)
def test_protocol_v1_transfer_entries_reject_identity_less_requests(
    method_name: str,
    operation: str,
) -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._lifecycle_contract = _v1_contract(generation_first=True)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._transfer_worker = Mock()
    transceiver._validate_receive_publication_contract = Mock()
    request = SimpleNamespace(
        py_disagg_transfer_protocol_identity=None,
        py_disaggregated_params=None,
    )

    with pytest.raises(
        RuntimeError,
        match=f"protocol-v1 {operation} requires exact transfer identity",
    ):
        getattr(transceiver, method_name)(request)

    transceiver._validate_receive_publication_contract.assert_not_called()
    assert transceiver._transfer_worker.mock_calls == []


def test_server_params_publish_the_exact_runtime_contract() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = _mapping(attention_dp=True)
    transceiver._dp_rank = 2
    transceiver._context_info_endpoint = "tcp://context:9000"
    transceiver._instance_name = str(uuid4())
    transceiver._lifecycle_contract = _v1_contract(attention_dp=True)
    transceiver._lifecycle_advertisement = KvCacheTransceiverV2._make_lifecycle_advertisement(
        lifecycle_contract=transceiver._lifecycle_contract,
        instance_id=transceiver._instance_name,
        mapping=transceiver._mapping,
    )

    params = transceiver.get_disaggregated_params()

    advertisement = params["context_transceiver_lifecycle"]
    assert advertisement["instance_id"] == transceiver._instance_name
    assert advertisement["protocol_version"] == transceiver.capabilities().protocol_version
    assert advertisement["capabilities"] == sorted(
        capability.value for capability in transceiver.capabilities().supported
    )
    assert params["ctx_dp_rank"] is None


def test_qualified_v0_keeps_legacy_server_params_shape() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = _mapping()
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "tcp://context:9000"
    transceiver._bounce_transfer_enabled = False

    params = transceiver.get_disaggregated_params()

    assert "context_transceiver_lifecycle" not in params
    assert params == {
        "ctx_dp_rank": 0,
        "ctx_info_endpoint": ["tcp://context:9000"],
    }


def test_server_params_reject_contract_drift() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = _mapping()
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "tcp://context:9000"
    transceiver._instance_name = str(uuid4())
    transceiver._lifecycle_contract = _v1_contract()
    transceiver._lifecycle_advertisement = _advertisement()

    with pytest.raises(RuntimeError, match="diverged"):
        transceiver.get_disaggregated_params()


def test_rank_group_rejects_mixed_runtime_contracts_before_endpoint_exchange() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._lifecycle_advertisement = _advertisement()
    local_contract = transceiver._lifecycle_contract_fingerprint()
    transceiver._dist = Mock()
    transceiver._dist.allgather.return_value = [
        local_contract,
        (
            local_contract[0],
            int(ProtocolVersion.QUALIFIED_LEGACY),
            (),
            True,
        ),
    ]

    with pytest.raises(RuntimeError, match="inconsistent lifecycle contracts"):
        transceiver._exchange_rank_info()

    transceiver._dist.allgather.assert_called_once_with(local_contract)


def test_deterministic_attention_dp_context_first_negotiates() -> None:
    source = _advertisement(attention_dp=True)
    destination = _advertisement(attention_dp=True)

    negotiated = negotiate_generation_safe_lifecycle(
        source,
        destination,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        ctx_dp_rank=2,
    )

    assert LifecycleCapability.ATTENTION_DATA_PARALLEL.value in negotiated.required_capabilities


def test_unknown_writer_generation_first_adp_rejects_before_publication() -> None:
    source = _advertisement(attention_dp=True, generation_first=True)
    destination = _advertisement(attention_dp=True, generation_first=True)

    with pytest.raises(LifecycleNegotiationError, match="exact context writer rank"):
        negotiate_generation_safe_lifecycle(
            source,
            destination,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ctx_dp_rank=None,
        )

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = _mapping(attention_dp=True)
    transceiver._lifecycle_contract = _v1_contract(
        attention_dp=True,
        generation_first=True,
    )
    request = SimpleNamespace(
        py_disaggregated_params=SimpleNamespace(
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ctx_dp_rank=None,
        )
    )
    with pytest.raises(RuntimeError, match="exact context writer rank"):
        transceiver._validate_receive_publication_contract(request)
