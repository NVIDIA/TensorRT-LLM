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

from types import SimpleNamespace
from uuid import UUID

import msgpack
import numpy as np
import pytest

from tensorrt_llm import bindings
from tensorrt_llm._torch.disaggregation.lifecycle import (
    LifecycleCapability,
    TransceiverCapabilities,
)
from tensorrt_llm._torch.disaggregation.native import rank_info as rank_info_module
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.protocol import (
    PROTOCOL_V1_REQUIRED_CAPABILITIES,
    ProtocolIdentityError,
)


def test_rank_info_construction():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    assert ri.instance_name == "gen_0"
    assert ri.tp_size == 2
    assert ri.pp_size == 1
    assert ri.layer_num_per_pp == [32]
    assert ri.sender_endpoints == ["tcp://10.0.0.1:5000"]


def test_rank_info_msgpack_roundtrip():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.instance_name == ri.instance_name
    assert restored.tp_size == ri.tp_size
    assert restored.transfer_engine_info == ri.transfer_engine_info
    assert restored.aux_meta is None


def test_rank_info_roundtrip_preserves_endpoint_lifecycle_advertisement():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        endpoint_incarnation=UUID(int=7),
        lifecycle_protocol_version=1,
        lifecycle_capabilities=tuple(sorted(PROTOCOL_V1_REQUIRED_CAPABILITIES)),
        qualified_legacy_mode=False,
        sender_endpoint_incarnations=[UUID(int=8)],
    )

    restored = RankInfo.from_bytes(ri.to_bytes())

    assert restored.endpoint_incarnation == UUID(int=7)
    assert restored.lifecycle_protocol_version == 1
    assert frozenset(restored.lifecycle_capabilities) == PROTOCOL_V1_REQUIRED_CAPABILITIES
    assert not restored.qualified_legacy_mode
    assert restored.sender_endpoint_incarnations == [UUID(int=8)]


def test_rank_info_protocol_v0_keeps_legacy_wire_shape():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=[],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        endpoint_incarnation=UUID(int=7),
    )

    payload = ri.to_bytes()
    wire = msgpack.unpackb(payload, raw=False)
    restored = RankInfo.from_bytes(payload)

    assert (
        not {
            "endpoint_incarnation",
            "lifecycle_protocol_version",
            "lifecycle_capabilities",
            "qualified_legacy_mode",
            "sender_endpoint_incarnations",
        }
        & wire.keys()
    )
    assert restored.endpoint_incarnation is None
    assert restored.lifecycle_protocol_version == 0
    assert restored.qualified_legacy_mode


def test_rank_info_rejects_partial_protocol_v1_advertisement():
    with pytest.raises(ProtocolIdentityError, match="missing identity capabilities"):
        RankInfo(
            instance_name="gen_0",
            instance_rank=0,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            layer_num_per_pp=[32],
            sender_endpoints=[],
            server_endpoint="tcp://10.0.0.1:5000",
            self_endpoint="tcp://10.0.0.1:5001",
            transfer_engine_info=b"",
            endpoint_incarnation=UUID(int=7),
            lifecycle_protocol_version=1,
            lifecycle_capabilities=("ATTEMPT_IDENTITY",),
            qualified_legacy_mode=False,
        )


def test_rank_info_rejects_noncanonical_capability_tuple():
    with pytest.raises(ValueError, match="sorted tuple"):
        RankInfo(
            instance_name="gen_0",
            instance_rank=0,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            layer_num_per_pp=[32],
            sender_endpoints=[],
            server_endpoint="tcp://10.0.0.1:5000",
            self_endpoint="tcp://10.0.0.1:5001",
            transfer_engine_info=b"",
            lifecycle_capabilities=("TENSOR_PARALLEL", "DIRECT_TRANSFER"),
        )


def test_rank_info_revalidates_late_protocol_v1_advertisement():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        endpoint_incarnation=UUID(int=7),
    )
    ri.lifecycle_protocol_version = 1
    ri.qualified_legacy_mode = False
    ri.sender_endpoint_incarnations = [UUID(int=8)]

    with pytest.raises(ProtocolIdentityError, match="missing identity capabilities"):
        ri.to_bytes()


def test_rank_info_rejects_nil_sender_endpoint_incarnation():
    with pytest.raises(ValueError, match="non-nil"):
        RankInfo(
            instance_name="gen_0",
            instance_rank=0,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            layer_num_per_pp=[32],
            sender_endpoints=["tcp://10.0.0.1:5000"],
            server_endpoint="tcp://10.0.0.1:5000",
            self_endpoint="tcp://10.0.0.1:5001",
            transfer_engine_info=b"",
            endpoint_incarnation=UUID(int=7),
            sender_endpoint_incarnations=[UUID(int=0)],
        )


def test_rank_info_validates_exact_server_runtime_contract():
    supported = frozenset(
        LifecycleCapability(capability) for capability in PROTOCOL_V1_REQUIRED_CAPABILITIES
    )
    contract = TransceiverCapabilities(
        protocol_version=1,
        supported=supported,
        qualified_legacy_mode=False,
    )
    instance_id = "750799b7-9e99-4b31-a7f0-52d44a1a7906"
    ri = RankInfo(
        instance_name=instance_id,
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=[],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        endpoint_incarnation=UUID(int=7),
        lifecycle_protocol_version=1,
        lifecycle_capabilities=tuple(sorted(capability.value for capability in contract.supported)),
        qualified_legacy_mode=False,
    )

    ri.validate_runtime_lifecycle(contract, instance_id=instance_id)

    with pytest.raises(ProtocolIdentityError, match="instance_name"):
        ri.validate_runtime_lifecycle(
            contract,
            instance_id="47e78cb5-c395-43aa-a20f-6a66e89f6e38",
        )

    drifted_contract = TransceiverCapabilities(
        protocol_version=1,
        supported=contract.supported | {LifecycleCapability.DIRECT_TRANSFER},
        qualified_legacy_mode=False,
    )
    with pytest.raises(ProtocolIdentityError, match="does not match"):
        ri.validate_runtime_lifecycle(
            drifted_contract,
            instance_id=instance_id,
        )


def test_rank_info_roundtrip_with_aux_meta():
    meta = AuxBufferMeta(
        ptrs=np.array([0x4000, 0x5000], dtype=np.int64),
        size=np.array([1024, 2048], dtype=np.int64),
        item_sizes=np.array([64, 128], dtype=np.int64),
        device="cpu",
    )
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        aux_meta=meta,
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.aux_meta is not None
    np.testing.assert_array_equal(restored.aux_meta.ptrs, [0x4000, 0x5000])
    np.testing.assert_array_equal(restored.aux_meta.size, [1024, 2048])
    np.testing.assert_array_equal(restored.aux_meta.item_sizes, [64, 128])
    assert restored.aux_meta.device == "cpu"


def test_from_kv_cache_manager_uses_first_nonzero_kv_head_count(monkeypatch) -> None:
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    mapping = SimpleNamespace(
        rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=False,
    )
    manager = SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[0, 8, 0],
        pp_layers=[0, 1, 2],
        tokens_per_block=32,
        head_dim=128,
        dtype=bindings.DataType.HALF,
        kv_factor=2,
    )

    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert info.attention.kv_heads_per_rank == 8


def test_from_kv_cache_manager_preserves_attention_dp_on_attention_free_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    mapping = SimpleNamespace(
        rank=1,
        tp_size=2,
        tp_rank=1,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=True,
    )
    manager = SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[0, 0],
        pp_layers=[0, 1],
        tokens_per_block=32,
        head_dim=128,
        dtype=bindings.DataType.HALF,
        kv_factor=2,
    )

    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert info.attention is not None
    assert info.attention.kv_heads_per_rank == 0
    assert info.attention.enable_attention_dp
    assert MambaPolicy._mamba_tp(info) == (1, 0)


def test_from_kv_cache_manager_applies_v1_contract_before_publication(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    mapping = SimpleNamespace(
        rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=False,
    )
    manager = SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[8],
        pp_layers=[0],
        tokens_per_block=32,
        head_dim=128,
        dtype=bindings.DataType.HALF,
        kv_factor=2,
    )
    contract = TransceiverCapabilities(
        protocol_version=1,
        supported=frozenset(
            LifecycleCapability(capability) for capability in PROTOCOL_V1_REQUIRED_CAPABILITIES
        ),
        qualified_legacy_mode=False,
    )
    instance_id = "750799b7-9e99-4b31-a7f0-52d44a1a7906"

    info = RankInfo.from_kv_cache_manager(
        instance_id,
        manager,
        device_id=0,
        lifecycle_contract=contract,
    )

    info.validate_runtime_lifecycle(contract, instance_id=instance_id)
    assert info.endpoint_incarnation is not None
    assert info.endpoint_incarnation.int != 0
    assert frozenset(info.lifecycle_capabilities) == PROTOCOL_V1_REQUIRED_CAPABILITIES


@pytest.mark.parametrize(
    ("dtype", "expected_element_bytes", "expected_type"),
    [(bindings.DataType.NVFP4, 0.5, float), (bindings.DataType.HALF, 2, int)],
)
def test_rank_info_represents_cache_element_bytes(
    monkeypatch, dtype, expected_element_bytes, expected_type
):
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _manager: None)
    manager = SimpleNamespace(
        mapping=SimpleNamespace(
            rank=0,
            tp_size=2,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            cp_size=1,
            cp_rank=0,
            enable_attention_dp=False,
        ),
        pp_layers=[0],
        num_kv_heads_per_layer=[4],
        tokens_per_block=64,
        head_dim=128,
        dtype=dtype,
        kv_factor=2,
    )

    rank_info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert rank_info.attention.element_bytes == expected_element_bytes
    assert isinstance(rank_info.attention.element_bytes, expected_type)

    restored = RankInfo.from_bytes(rank_info.to_bytes())
    assert restored.attention.element_bytes == expected_element_bytes
    assert isinstance(restored.attention.element_bytes, expected_type)
