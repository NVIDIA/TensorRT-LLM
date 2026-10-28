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

from dataclasses import dataclass, fields
from typing import List, Optional

import msgpack

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


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

    enable_attention_dp: bool = False
    aux_meta: Optional[AuxBufferMeta] = None
    page_table: Optional[KVCachePageTable] = None
    bounded_replay_on_generation: bool = False

    @property
    def tp_size_per_dp_group(self) -> int:
        return self.tp_size // self.dp_size if self.enable_attention_dp else self.tp_size

    def to_bytes(self) -> bytes:
        # Shallow, field by field: the page table converts itself (to_dict), and a deep copy would
        # fail on its page layout's read-only mappings.
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        data["page_table"] = self.page_table.to_dict() if self.page_table is not None else None
        return msgpack.packb(data)

    @classmethod
    def from_kv_cache_manager(
        cls,
        instance_name: str,
        kv_cache_manager: KVCacheManager,
        device_id: int,
        aux_buffer_meta: Optional[AuxBufferMeta] = None,
    ) -> "RankInfo":
        m = kv_cache_manager.mapping
        kvm = kv_cache_manager
        enable_attention_dp = m.enable_attention_dp
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
            enable_attention_dp=enable_attention_dp,
            aux_meta=aux_buffer_meta,
            page_table=build_page_table_from_manager(kvm),
            bounded_replay_on_generation=getattr(kvm, "bounded_replay_on_generation", False),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("page_table") is not None:
            unpacked["page_table"] = KVCachePageTable.from_dict(unpacked["page_table"])
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)
