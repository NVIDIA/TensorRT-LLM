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

from dataclasses import asdict, dataclass, field
from typing import List, Optional
from uuid import UUID, uuid4

import msgpack

from tensorrt_llm._torch.disaggregation.lifecycle import (
    LifecycleCapability,
    TransceiverCapabilities,
)
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.protocol import (
    ProtocolIdentityError,
    ProtocolVersion,
    validate_protocol_advertisement,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes


@dataclass
class RankInfo:
    instance_name: str
    instance_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    layer_num_per_pp: List[int]
    sender_endpoints: List[str]
    self_endpoint: str
    transfer_engine_info: bytes

    dp_size: int = 1
    dp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    device_id: int = 0

    attention: Optional[AttentionInfo] = None
    aux_meta: Optional[AuxBufferMeta] = None
    page_table: Optional[KVCachePageTable] = None
    endpoint_incarnation: Optional[UUID] = None
    lifecycle_protocol_version: int = int(ProtocolVersion.QUALIFIED_LEGACY)
    lifecycle_capabilities: tuple[str, ...] = field(default_factory=tuple)
    qualified_legacy_mode: bool = True
    sender_endpoint_incarnations: List[UUID] = field(default_factory=list)

    def __post_init__(self) -> None:
        capabilities = tuple(self.lifecycle_capabilities)
        if any(
            not isinstance(capability, str) or not capability for capability in capabilities
        ) or capabilities != tuple(sorted(set(capabilities))):
            raise ValueError(
                "lifecycle_capabilities must be a sorted tuple of unique non-empty strings"
            )
        self.lifecycle_capabilities = capabilities
        self.sender_endpoint_incarnations = [
            incarnation if isinstance(incarnation, UUID) else UUID(str(incarnation))
            for incarnation in self.sender_endpoint_incarnations
        ]
        self._validate_lifecycle_advertisement()

    def _validate_lifecycle_advertisement(self) -> None:
        capabilities = tuple(self.lifecycle_capabilities)
        if any(
            not isinstance(capability, str) or not capability for capability in capabilities
        ) or capabilities != tuple(sorted(set(capabilities))):
            raise ValueError(
                "lifecycle_capabilities must be a sorted tuple of unique non-empty strings"
            )
        if self.sender_endpoint_incarnations and len(self.sender_endpoint_incarnations) != len(
            self.sender_endpoints
        ):
            raise ValueError("sender_endpoint_incarnations must align with sender_endpoints")
        if any(
            not isinstance(incarnation, UUID) or incarnation.int == 0
            for incarnation in self.sender_endpoint_incarnations
        ):
            raise ValueError("sender endpoint incarnations must be non-nil UUIDs")
        if self.lifecycle_protocol_version == ProtocolVersion.GENERATION_SAFE and len(
            self.sender_endpoint_incarnations
        ) != len(self.sender_endpoints):
            raise ValueError("protocol-v1 requires an incarnation for every sender endpoint")
        validate_protocol_advertisement(
            self.lifecycle_protocol_version,
            endpoint_incarnation=self.endpoint_incarnation,
            capabilities=frozenset(self.lifecycle_capabilities),
            qualified_legacy_mode=self.qualified_legacy_mode,
        )

    def validate_runtime_lifecycle(
        self,
        lifecycle_contract: TransceiverCapabilities,
        *,
        instance_id: str,
    ) -> None:
        """Require exact parity with the immutable server-level contract."""
        if self.instance_name != instance_id:
            raise ProtocolIdentityError(
                "rank lifecycle instance_name does not match the server instance_id"
            )
        expected_capabilities = tuple(
            sorted(capability.value for capability in lifecycle_contract.supported)
        )
        actual_contract = (
            int(self.lifecycle_protocol_version),
            self.lifecycle_capabilities,
            self.qualified_legacy_mode,
        )
        expected_contract = (
            int(lifecycle_contract.protocol_version),
            expected_capabilities,
            lifecycle_contract.qualified_legacy_mode,
        )
        if actual_contract != expected_contract:
            raise ProtocolIdentityError(
                "rank lifecycle contract does not match the server advertisement"
            )
        self._validate_lifecycle_advertisement()

    @property
    def tp_size_per_dp_group(self) -> int:
        if self.attention is None:
            return self.tp_size
        return self.tp_size // self.dp_size if self.attention.enable_attention_dp else self.tp_size

    def to_bytes(self) -> bytes:
        # RankInfo is populated incrementally after construction. Validate the
        # final values again so a late mutation cannot advertise partial v1.
        self._validate_lifecycle_advertisement()
        data = asdict(self)
        data["attention"] = self.attention.to_dict() if self.attention is not None else None
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        data["page_table"] = self.page_table.to_dict() if self.page_table is not None else None
        data["endpoint_incarnation"] = (
            self.endpoint_incarnation.bytes if self.endpoint_incarnation is not None else None
        )
        data["sender_endpoint_incarnations"] = [
            incarnation.bytes for incarnation in self.sender_endpoint_incarnations
        ]
        if self.lifecycle_protocol_version == ProtocolVersion.QUALIFIED_LEGACY:
            # Keep the qualified-v0 payload byte-schema compatible with peers
            # that predate lifecycle negotiation. Qualification is the
            # receiver's explicit default only for protocol-v0.
            for key in (
                "endpoint_incarnation",
                "lifecycle_protocol_version",
                "lifecycle_capabilities",
                "qualified_legacy_mode",
                "sender_endpoint_incarnations",
            ):
                data.pop(key)
        return msgpack.packb(data)

    @classmethod
    def from_kv_cache_manager(
        cls,
        instance_name: str,
        kv_cache_manager: KVCacheManager,
        device_id: int,
        aux_buffer_meta: Optional[AuxBufferMeta] = None,
        lifecycle_contract: Optional[TransceiverCapabilities] = None,
    ) -> "RankInfo":
        m = kv_cache_manager.mapping
        kvm = kv_cache_manager
        if lifecycle_contract is None:
            lifecycle_contract = TransceiverCapabilities(
                protocol_version=int(ProtocolVersion.QUALIFIED_LEGACY),
                qualified_legacy_mode=True,
            )
        if not isinstance(lifecycle_contract, TransceiverCapabilities):
            raise TypeError("lifecycle_contract must be TransceiverCapabilities")
        if any(
            not isinstance(capability, LifecycleCapability)
            for capability in lifecycle_contract.supported
        ):
            raise TypeError("lifecycle_contract.supported must contain LifecycleCapability values")
        enable_attention_dp = m.enable_attention_dp
        # Keep AttentionInfo on attention-free PP stages so it can still carry
        # the attention-DP topology used by Mamba transfers.  A zero head count
        # means that this rank has no local attention cache; AttentionPolicy
        # must not perform head-ratio arithmetic for such ranks.
        kv_heads_per_rank = next((h for h in kvm.num_kv_heads_per_layer if h > 0), 0)
        # Eight is the smallest element count guaranteed to occupy whole bytes
        # for every supported sub-byte cache dtype (including NVFP4).
        bytes_for_eight_elements = get_size_in_bytes(8, kvm.dtype)
        element_bytes = (
            bytes_for_eight_elements // 8
            if bytes_for_eight_elements % 8 == 0
            else bytes_for_eight_elements / 8
        )
        return cls(
            instance_name=instance_name,
            instance_rank=m.rank,
            tp_size=m.tp_size,
            tp_rank=m.tp_rank,
            pp_size=m.pp_size,
            pp_rank=m.pp_rank,
            dp_size=m.tp_size if enable_attention_dp else m.dp_size,
            dp_rank=m.tp_rank if enable_attention_dp else 0,
            cp_size=m.cp_size,
            cp_rank=m.cp_rank,
            device_id=device_id,
            layer_num_per_pp=[len(kvm.pp_layers)],
            sender_endpoints=[],
            self_endpoint="",
            transfer_engine_info=bytes(),
            attention=AttentionInfo(
                kv_heads_per_rank=kv_heads_per_rank,
                tokens_per_block=kvm.tokens_per_block,
                dims_per_head=kvm.head_dim,
                element_bytes=element_bytes,
                enable_attention_dp=enable_attention_dp,
                is_mla=kvm.kv_factor == 1,
            ),
            aux_meta=aux_buffer_meta,
            page_table=build_page_table_from_manager(kvm),
            endpoint_incarnation=uuid4(),
            lifecycle_protocol_version=int(lifecycle_contract.protocol_version),
            lifecycle_capabilities=tuple(
                sorted(capability.value for capability in lifecycle_contract.supported)
            ),
            qualified_legacy_mode=lifecycle_contract.qualified_legacy_mode,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("attention") is not None:
            unpacked["attention"] = AttentionInfo.from_dict(unpacked["attention"])
        if unpacked.get("page_table") is not None:
            unpacked["page_table"] = KVCachePageTable.from_dict(unpacked["page_table"])
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        if unpacked.get("endpoint_incarnation") is not None:
            incarnation = unpacked["endpoint_incarnation"]
            if not isinstance(incarnation, bytes) or len(incarnation) != 16:
                raise ValueError("endpoint_incarnation must be a 16-byte UUID")
            unpacked["endpoint_incarnation"] = UUID(bytes=incarnation)
        if "lifecycle_capabilities" in unpacked:
            unpacked["lifecycle_capabilities"] = tuple(unpacked["lifecycle_capabilities"])
        if "sender_endpoint_incarnations" in unpacked:
            incarnations = unpacked["sender_endpoint_incarnations"]
            if any(
                not isinstance(incarnation, bytes) or len(incarnation) != 16
                for incarnation in incarnations
            ):
                raise ValueError("sender_endpoint_incarnations must contain 16-byte UUIDs")
            unpacked["sender_endpoint_incarnations"] = [
                UUID(bytes=incarnation) for incarnation in incarnations
            ]
        return cls(**unpacked)
