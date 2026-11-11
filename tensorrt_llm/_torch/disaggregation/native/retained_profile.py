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
"""Typed retained-only ownership admission; ordinary profiles stay unchanged.

Local eligibility establishes transport/managed-layout preconditions. It is not
a model quality or deployment qualification claim. A physically owned retained
session additionally requires verified startup model identity and an independently
recomputed complete native-fragment certificate before it becomes visible.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.base.agent import use_pure_python_transfer_agent
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

from .peer import PeerRegistrar
from .retained_coverage import (
    VERSION,
    CacheSnapshot,
    FrozenWritePlan,
    _heads,
    _json,
    bind_destination_session,
    bind_source_session,
    certify_destination,
    logical_layout_identity,
)


@dataclass(frozen=True)
class RetainedLocalProfile:
    manager: object = field(repr=False, compare=False)
    model_identity: str
    page_table_digest: str
    topology: tuple[int, int, int, int, int]
    version: str = VERSION

    def check_manager(self, manager) -> None:
        mapping = manager.mapping
        topology = (
            mapping.tp_size,
            mapping.tp_rank,
            mapping.pp_size,
            mapping.cp_size,
            mapping.dp_size,
        )
        if (
            manager is not self.manager
            or topology != self.topology
            or hashlib.sha256(_json(build_page_table_from_manager(manager).to_dict())).hexdigest()
            != self.page_table_digest
        ):
            raise ValueError("retained local manager/registered page-table identity changed")


def qualify_retained_local(
    mapping, manager, transport_config, retained_config
) -> RetainedLocalProfile:
    """Validate actual local objects before the transceiver's all-rank MPI vote."""
    allowed = {
        ("tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2", "KVCacheManagerV2"),
        (
            "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager",
            "MiniMaxM3KVCacheManagerV2",
        ),
    }
    if (type(manager).__module__, type(manager).__name__) not in allowed:
        raise ValueError("retained ownership requires a specifically audited managed cache type")
    if (
        not manager.is_disagg
        or manager.kv_cache_type != CacheTypeCpp.SELF
        or manager.is_draft
        or manager.layer_specs is not None
        or manager.mixed_precision_low_dtype is not None
        or manager.get_disagg_ignored_roles()
        or mapping.enable_attention_dp
        or mapping.tp_size not in (1, 2, 4, 8)
        or mapping.pp_size != 1
        or mapping.cp_size != 1
        or mapping.dp_size != 1
        or mapping.world_size != mapping.tp_size
    ):
        raise ValueError("retained ownership requires complete managed PAGED state and PP=DP=CP=1")
    if type(manager).__name__ == "MiniMaxM3KVCacheManagerV2" and manager._index_v_buffers:
        raise ValueError("retained ownership does not transfer unmanaged M3 INDEX_VALUE")
    if (
        transport_config.backend != "NIXL"
        or transport_config.transceiver_runtime != "PYTHON"
        or transport_config.kv_transfer_timeout_ms is None
        or not math.isfinite(transport_config.kv_transfer_timeout_ms)
        or transport_config.kv_transfer_timeout_ms <= 0
        or transport_config.kv_cache_bounce_size_mb != 0
        or transport_config.enable_pipelined_transfer
        or use_pure_python_transfer_agent()
        or os.getenv("TRTLLM_DISAGG_NO_RETRY", "0") != "1"
        or os.getenv("TRTLLM_NIXL_KVCACHE_BACKEND", "UCX") != "UCX"
        or os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "0") == "1"
        or os.getenv("TRTLLM_DISAGG_LAYERWISE", "0") == "1"
    ):
        raise ValueError(
            "retained ownership requires direct native UCX/no-retry/monolithic transfer"
        )
    if not retained_config.model_identity or not retained_config.model_manifest:
        raise ValueError("retained ownership requires configured startup checkpoint verification")
    table = build_page_table_from_manager(manager)
    role_map = {}
    for desc in manager.impl.pool_group_descs:
        for variant in desc.slot_desc.variants:
            for pool_id, coalesced in enumerate(variant.coalesced_buffers):
                size = int(coalesced.single_buffer_size)
                for index, buffer in enumerate(coalesced.buffer_ids):
                    key = (
                        int(variant.layer_group_id),
                        pool_id,
                        int(buffer.layer_id),
                        index * size,
                        size,
                    )
                    if key in role_map:
                        raise ValueError("retained native storage repeats a buffer identity")
                    role_map[key] = str(buffer.role)
    allowed_roles = {
        str(Role.KEY),
        str(Role.VALUE),
        str(Role.KEY_BLOCK_SCALE),
        str(Role.VALUE_BLOCK_SCALE),
        str(Role.INDEX_KEY),
    }
    seen, per_layer = set(), {}
    for group_id, group in enumerate(table.layer_groups):
        if group.kind != CacheKind.PAGED:
            raise ValueError(
                "retained ownership rejects unqualified recurrent checkpoint semantics"
            )
        layers = {layer.local_layer_id: layer.global_layer_id for layer in group.local_layers}
        for view in group.pool_views:
            if (
                view.mapper_kind not in (MapperKind.INDEXED, MapperKind.NHD, MapperKind.REPLICATED)
                or view.ring_slot_bytes is not None
            ):
                raise ValueError("retained ownership rejects unqualified role mapper/ring geometry")
            for entry in view.buffer_entries:
                local, offset, size = (
                    int(entry[name]) for name in ("local_layer_id", "offset", "size")
                )
                key = (group_id, view.pool_idx, local, offset, size)
                role = role_map[key]
                global_layer = int(layers[local])
                if role not in allowed_roles or size <= 0:
                    raise ValueError("retained ownership has an unqualified or empty managed role")
                seen.add(key)
                per_layer.setdefault(global_layer, set()).add(role)
                if role == str(Role.INDEX_KEY):
                    if view.mapper_kind != MapperKind.REPLICATED:
                        raise ValueError("retained INDEX_KEY must use native replicated ownership")
                else:
                    if view.mapper_kind == MapperKind.REPLICATED or view.num_kv_heads is None:
                        raise ValueError("retained K/V/scales require actual native head geometry")
                    original = manager.num_kv_heads
                    total = original if type(original) is int else original[global_layer]
                    _heads(int(total), int(view.num_kv_heads), mapping.tp_size, mapping.tp_rank)
                    if size % (int(view.num_kv_heads) * table.tokens_per_block):
                        raise ValueError(
                            "retained packed-role head/token slicing is not byte aligned"
                        )
    if not seen or seen != set(role_map):
        raise ValueError("retained native metadata omits managed storage buffers")
    for roles in per_layer.values():
        if not {str(Role.KEY), str(Role.VALUE)} <= roles:
            raise ValueError("retained model layer lacks complete K/V state")
        if (str(Role.KEY_BLOCK_SCALE) in roles) != (str(Role.VALUE_BLOCK_SCALE) in roles):
            raise ValueError("retained model layer lacks a complete K/V scale pair")
    return RetainedLocalProfile(
        manager,
        retained_config.model_identity,
        hashlib.sha256(_json(table.to_dict())).hexdigest(),
        (mapping.tp_size, mapping.tp_rank, mapping.pp_size, mapping.cp_size, mapping.dp_size),
    )


@dataclass(frozen=True)
class RetainedAdmission:
    local_profile: RetainedLocalProfile
    direction: str
    local_snapshot: CacheSnapshot
    selected_plans: tuple[FrozenWritePlan, ...]
    sources: tuple[CacheSnapshot, ...]
    authorized_writers: frozenset[tuple[str, int]]
    certificates: tuple[str, ...]

    def validate(self, endpoint, request_id, params, prompt_len, direction):
        if (
            direction != self.direction
            or endpoint._retained_profile is not self.local_profile
            or request_id != self.local_snapshot.request_id
            or params.schedule_style != DisaggScheduleStyle.CONTEXT_FIRST
            or prompt_len != self.local_snapshot.token_end
            or any(
                plan.binding.claim_id != params.disagg_request_id for plan in self.selected_plans
            )
        ):
            raise ValueError("retained admission does not bind this native session")
        self.local_profile.check_manager(self.local_profile.manager)
        info = endpoint._registrar.self_rank_info
        if info.to_bytes() != self.local_snapshot.rank_info_bytes:
            raise ValueError("retained admission local native registration changed")

    def install(self, session):
        if self.direction == "send":
            bind_source_session(session, self.local_snapshot, self.selected_plans)
        else:
            sources = {
                (value.rank_info.instance_name, value.rank_info.instance_rank): value
                for value in self.sources
            }
            bind_destination_session(
                session, self.local_snapshot, self.selected_plans, sources, self.authorized_writers
            )


def make_retained_admission(
    worker,
    verified_identity,
    *,
    direction,
    local_snapshot,
    sources,
    destinations,
    plans,
    certificates,
) -> RetainedAdmission:
    """Recheck the complete trusted all-rank proof before native registration."""
    from tensorrt_llm._torch.pyexecutor.retained_identity import VerifiedRetainedIdentity

    profile = worker._config.retained_profile
    if not isinstance(profile, RetainedLocalProfile) or not isinstance(
        verified_identity, VerifiedRetainedIdentity
    ):
        raise ValueError(
            "retained admission requires actual local profile and verified startup identity"
        )
    if (
        direction not in ("send", "receive")
        or verified_identity.model_identity != profile.model_identity
    ):
        raise ValueError("retained admission startup identity/mode mismatch")
    profile.check_manager(worker._config.kv_cache_manager)
    sources, destinations = tuple(sources), tuple(destinations)
    rows, certificates = tuple(tuple(row) for row in plans), tuple(certificates)
    for cohort in (sources, destinations):
        infos = [value.rank_info for value in cohort]
        if (
            not infos
            or len({info.instance_name for info in infos}) != 1
            or {info.instance_rank for info in infos} != set(range(len(cohort)))
            or any(
                info.tp_size != len(cohort) or info.tp_rank != info.instance_rank for info in infos
            )
            or len({value.generation for value in cohort}) != 1
            or {value.model_identity for value in cohort} != {verified_identity.model_identity}
            or len({logical_layout_identity(value) for value in cohort}) != 1
        ):
            raise ValueError("retained admission lacks a complete coherent engine rank cohort")
    if (
        len(rows) != len(destinations)
        or len(certificates) != len(destinations)
        or logical_layout_identity(sources[0]) != logical_layout_identity(destinations[0])
    ):
        raise ValueError("retained admission source/destination semantic layout differs")
    source_map = {
        (value.rank_info.instance_name, value.rank_info.instance_rank): value for value in sources
    }
    certified, writers_by_rank = [], {}
    for destination in destinations:
        info = destination.rank_info
        registrar = PeerRegistrar(info, KVRegionExtractorV1(info.page_table))
        ranks = registrar.get_peer_overlap(sources[0].rank_info, 0).ranks
        writers = {(sources[0].rank_info.instance_name, rank) for rank in ranks}
        row = rows[info.instance_rank]
        certified.append(certify_destination(destination, row, source_map, writers))
        writers_by_rank[info.instance_rank] = frozenset(writers)
    if tuple(certified) != certificates:
        raise ValueError("retained admission differs from independent whole-cohort certificates")
    local = worker._rank_info
    if direction == "send":
        selected = tuple(
            plan
            for row in rows
            for plan in row
            if (plan.binding.source_instance, plan.binding.source_rank)
            == (local.instance_name, local.instance_rank)
        )
        allowed = frozenset()
        candidates = sources
    else:
        selected = rows[local.instance_rank]
        allowed = writers_by_rank[local.instance_rank]
        candidates = destinations
    if (
        not selected
        or local_snapshot.digest not in {value.digest for value in candidates}
        or local_snapshot.rank_info_bytes != local.to_bytes()
        or local_snapshot.model_identity != verified_identity.model_identity
    ):
        raise ValueError("retained admission does not bind the local engine/allocation")
    return RetainedAdmission(
        profile, direction, local_snapshot, selected, sources, allowed, certificates
    )
