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

from dataclasses import asdict, dataclass, replace
from typing import List, Optional

import msgpack

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.page_mapping import RankPages
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2


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

    aux_meta: Optional[AuxBufferMeta] = None
    pages: Optional[RankPages] = None
    bounded_replay_on_generation: bool = False

    @property
    def tp_size_per_dp_group(self) -> int:
        """Tensor-parallel ranks of one data-parallel group: one under attention data parallelism,
        as the rank's page layout says, all of them otherwise."""
        if self.pages is not None and self.pages.layout.topology.attention_dp:
            return self.tp_size // self.dp_size
        return self.tp_size

    def to_bytes(self) -> bytes:
        # Pages hold read-only mappings, which asdict cannot copy; they serialize themselves.
        data = asdict(replace(self, pages=None))
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        data["pages"] = self.pages.to_dict() if self.pages is not None else None
        return msgpack.packb(data)

    @classmethod
    def from_kv_cache_manager(
        cls,
        instance_name: str,
        kv_cache_manager: KVCacheManagerV2,
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
            aux_meta=aux_buffer_meta,
            pages=RankPages.of_manager(kvm),
            bounded_replay_on_generation=getattr(kvm, "bounded_replay_on_generation", False),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("pages") is not None:
            unpacked["pages"] = RankPages.from_dict(unpacked["pages"])
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)
