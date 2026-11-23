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

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import msgpack

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes


def unified_draft_layer_count(kv_cache_manager: KVCacheManager) -> int:
    """Draft layers a unified KV cache manager holds on this rank.

    Folded MTP/Eagle draft layers have no layer specs; they travel as ordinary
    layers and are not counted.
    """
    if getattr(kv_cache_manager, "layer_specs", None) is None:
        return 0
    return len(kv_cache_manager.draft_layer_ids)


def _draft_context_keep_tokens(kv_cache_manager: KVCacheManager) -> int:
    """The draft rows a window-bounded drafter context keeps per prompt (0: every row)."""
    keep = getattr(kv_cache_manager, "draft_context_keep_tokens", None)
    return keep if isinstance(keep, int) and keep > 0 else 0


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
    # Layers a unified KV cache manager owns for the drafter (layer specs with
    # a DRAFT owner). They ride the target's pages, so both peers must agree.
    num_draft_layers: int = 0
    # The draft rows a context worker writes below each prompt's end when its
    # drafter context is window-bounded (KVCacheManagerV2.draft_context_keep_tokens);
    # 0 writes every row. Sent only when nonzero.
    draft_context_keep_tokens: int = 0
    # Whether this worker's KV cache reuses committed blocks. Local only: a
    # generation worker checks it against a context peer's window, and it is
    # never sent.
    reuses_cached_blocks: bool = False

    @property
    def tp_size_per_dp_group(self) -> int:
        if self.attention is None:
            return self.tp_size
        return self.tp_size // self.dp_size if self.attention.enable_attention_dp else self.tp_size

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data.pop("reuses_cached_blocks")
        if not data["draft_context_keep_tokens"]:
            data.pop("draft_context_keep_tokens")
        data["attention"] = self.attention.to_dict() if self.attention is not None else None
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
            num_draft_layers=unified_draft_layer_count(kvm),
            draft_context_keep_tokens=_draft_context_keep_tokens(kvm),
            reuses_cached_blocks=getattr(kvm, "enable_block_reuse", False) is True,
        )

    def publish_instance_draft_layers(self, draft_layers_per_pp: Sequence[int]) -> None:
        """Advertise the draft layers of the whole pipeline instead of this rank's.

        A generation peer validates the draft-layer count against the first
        context rank only, and every context rank compares its own count with
        each generation rank's. With the draft layers on the last stage of a
        pipeline-parallel context instance, both comparisons have to see the
        instance-wide count on every context rank.

        The draft layers' global ids follow the target's, so the per-stage
        layer ranges built from ``layer_num_per_pp`` stay contiguous only when
        all draft layers sit on the last stage; any other placement is
        rejected.

        Args:
            draft_layers_per_pp: Draft layers held by each pipeline stage, in
                stage order (one entry per stage, as gathered over the
                pipeline group).
        """
        counts = [int(count) for count in draft_layers_per_pp]
        if len(counts) != self.pp_size:
            raise ValueError(
                f"RankInfo: {len(counts)} per-stage draft layer counts for pp_size={self.pp_size}"
            )
        if counts[self.pp_rank] != self.num_draft_layers:
            raise ValueError(
                f"RankInfo: stage {self.pp_rank} reports {counts[self.pp_rank]} draft layers "
                f"but holds {self.num_draft_layers}"
            )
        if any(counts[:-1]):
            raise ValueError(
                f"RankInfo: draft layers per pipeline stage are {counts}; the KV transfer needs "
                "all draft layers on the last stage, after the target layers"
            )
        self.num_draft_layers = sum(counts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("attention") is not None:
            unpacked["attention"] = AttentionInfo.from_dict(unpacked["attention"])
        if unpacked.get("page_table") is not None:
            unpacked["page_table"] = KVCachePageTable.from_dict(unpacked["page_table"])
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)
