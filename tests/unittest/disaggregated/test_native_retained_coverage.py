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
"""Allocated-CPU native mapper certificate tests; never GPU/profile qualification.

Uses actual main RankInfo, page descriptors, PeerRegistrar, extractors and
Sender._build_kv_write_meta with synthetic integer pointer metadata. No memory
at those addresses is accessed. Run against matching-main bindings plus candidate
retained_coverage.py/transfer.py; never run these on a mismatched rc2 wheel.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native import retained_coverage as coverage
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

pytestmark = pytest.mark.cpu_only


TPB, TOKENS = 4, 8
MODEL = "a" * 64


def snapshot(tp, rank, instance, *, layout="HND", global_heads=8):
    heads = max(1, global_heads // tp)
    role_specs = [
        (str(Role.KEY), TPB * heads * 8, layout),
        (str(Role.VALUE), TPB * heads * 8, layout),
        (str(Role.KEY_BLOCK_SCALE), TPB * heads * 2, layout),
        (str(Role.VALUE_BLOCK_SCALE), TPB * heads * 2, layout),
        (str(Role.INDEX_KEY), 32, "REPLICATED"),
    ]
    # Real V2-style coalescing by equal buffer size. At TP8, INDEX_KEY
    # shares a physical pool with K/V; separate views preserve role meaning.
    sizes = sorted({size for _, size, _ in role_specs})
    pools, views, templates = [], [], []
    for pool_id, size in enumerate(sizes):
        base = (
            (0x100000000 if instance == "src" else 0x1000000000)
            + rank * 0x10000000
            + pool_id * 0x100000
        )
        entries = []
        for layer in range(2):
            for role, role_size, kind in role_specs:
                if role_size == size:
                    entries.append((layer, len(entries) * size, role, kind))
        stride = len(entries) * size
        pools.append(PhysicalPool(base_address=base, slot_bytes=stride, num_slots=16))
        for kind in sorted({entry[3] for entry in entries}):
            selected = [entry for entry in entries if entry[3] == kind]
            view_id = len(views)
            mapper = (
                MapperKind.REPLICATED
                if kind == "REPLICATED"
                else MapperKind.NHD
                if kind == "NHD"
                else MapperKind.INDEXED
            )
            views.append(
                PoolView(
                    pool_idx=pool_id,
                    buffer_entries=np.array(
                        [(layer, offset, size) for layer, offset, _, _ in selected],
                        dtype=BUFFER_ENTRY_DTYPE,
                    ),
                    pool_role=frozenset(role for _, _, role, _ in selected),
                    mapper_kind=mapper,
                    num_kv_heads=None if kind == "REPLICATED" else heads,
                )
            )
            for layer, offset, role, _ in selected:
                total = 0 if kind == "REPLICATED" else global_heads
                start = (
                    0 if kind == "REPLICATED" else coverage._heads(global_heads, heads, tp, rank)
                )
                templates.append(
                    coverage.BufferTemplate(
                        0,
                        view_id,
                        layer,
                        role,
                        base,
                        stride,
                        16,
                        offset,
                        size,
                        kind,
                        0 if kind == "REPLICATED" else heads,
                        total,
                        start,
                        TPB,
                    )
                )
    table = KVCachePageTable(
        tokens_per_block=TPB,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                local_layers=[
                    LocalLayer(local_layer_id=layer, global_layer_id=layer) for layer in range(2)
                ],
                pool_views=views,
                kv_head_num_per_rank=heads,
                sliding_window_size=None,
            )
        ],
        pool_groups=[PhysicalPoolGroup(pools=pools)],
    )
    info = RankInfo(
        instance_name=instance,
        instance_rank=rank,
        tp_size=tp,
        tp_rank=rank,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=rank,
        layer_num_per_pp=[2],
        self_endpoint="synthetic-no-transport",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=heads,
            tokens_per_block=TPB,
            dims_per_head=4,
            element_bytes=2,
            enable_attention_dp=False,
            is_mla=False,
        ),
        page_table=table,
        aux_meta=None,
        sender_endpoints=[],
    )
    return coverage.CacheSnapshot(
        instance + "-generation",
        MODEL,
        10 if instance == "src" else 20,
        TOKENS,
        info.to_bytes(),
        ((1, 3),),
        tuple(templates),
    )


def binding(source, destination):
    a, b = source.rank_info, destination.rank_info
    return coverage.ClaimBinding(
        71,
        "prepared-import-nonce",
        MODEL,
        source.generation,
        a.instance_name,
        a.instance_rank,
        destination.generation,
        b.instance_name,
        b.instance_rank,
        TOKENS,
    )


def cohort(source_tp, destination_tp, *, destination_rank=0, layout="HND", global_heads=8):
    sources = {
        ("src", rank): snapshot(source_tp, rank, "src", layout=layout, global_heads=global_heads)
        for rank in range(source_tp)
    }
    destination = snapshot(
        destination_tp, destination_rank, "dst", layout=layout, global_heads=global_heads
    )
    info = destination.rank_info
    registrar = PeerRegistrar(info, KVRegionExtractorV1(info.page_table))
    ranks = registrar.get_peer_overlap(sources[("src", 0)].rank_info, 0).ranks
    writers = {("src", rank) for rank in ranks}
    plans = [
        coverage.preview_write_plan(
            binding(sources[writer], destination), sources[writer], destination
        )
        for writer in sorted(writers)
    ]
    return sources, destination, plans, writers


def meta(plan, *, src=None, dst=None, sizes=None):
    a, b, c = plan.arrays()
    destination = RankInfo.from_bytes(plan.destination_rank_info_bytes)
    return NS(
        unique_rid=plan.binding.claim_id,
        peer_rank=plan.binding.destination_rank,
        peer_name=plan.binding.destination_instance + str(plan.binding.destination_rank),
        peer_endpoint=destination.self_endpoint,
        dst_device_id=destination.device_id,
        slice_id=0,
        receiver_slice_id=0,
        is_last_slice=True,
        src_ptrs=a.copy() if src is None else np.array(src, dtype=np.int64),
        dst_ptrs=b.copy() if dst is None else np.array(dst, dtype=np.int64),
        sizes=c.copy() if sizes is None else np.array(sizes, dtype=np.int64),
    )


def region(snapshot_value, role):
    return next(
        item
        for item in coverage._regions(snapshot_value)
        if item.buffer.role == str(role) and item.buffer.global_layer == 0 and item.ordinal == 0
    )


class NativeFragmentCoverage(unittest.TestCase):
    def test_matching_and_asymmetric_actual_mapper_partitions_all_roles(self):
        for layout in ("HND", "NHD"):
            for src_tp, dst_tp in (
                (1, 1),
                (2, 2),
                (4, 4),
                (8, 8),
                (1, 2),
                (2, 1),
                (2, 4),
                (4, 2),
                (4, 8),
                (8, 4),
                (1, 8),
                (8, 1),
            ):
                for dst_rank in range(dst_tp):
                    with self.subTest(
                        layout=layout, src_tp=src_tp, dst_tp=dst_tp, dst_rank=dst_rank
                    ):
                        sources, destination, plans, writers = cohort(
                            src_tp, dst_tp, destination_rank=dst_rank, layout=layout
                        )
                        self.assertEqual(
                            64,
                            len(coverage.certify_destination(destination, plans, sources, writers)),
                        )
                        for plan in plans:
                            coverage.check_native_operation(plan, meta(plan))

    def test_replicated_heads_include_zero_byte_writer_without_duplicate_index_owner(self):
        sources, destination, plans, writers = cohort(8, 2, global_heads=2)
        self.assertTrue(any(plan.total_bytes == 0 for plan in plans))
        coverage.certify_destination(destination, plans, sources, writers)
        dormant = next(plan for plan in plans if plan.total_bytes == 0)
        source = sources[("src", dormant.binding.source_rank)]
        a, b = region(source, Role.INDEX_KEY), region(destination, Role.INDEX_KEY)
        bad = coverage.freeze_write_plan(
            dormant.binding,
            meta(dormant, src=[a.start], dst=[b.start], sizes=[a.end - a.start]),
            source,
            destination,
        )
        with self.assertRaisesRegex(ValueError, "hole or overlap"):
            coverage.certify_destination(
                destination, [bad if plan is dormant else plan for plan in plans], sources, writers
            )

    def test_equal_byte_compensation_missing_index_and_duplicate_main_is_rejected(self):
        sources, destination, plans, writers = cohort(8, 8)
        plan, source = plans[0], sources[("src", 0)]
        missing = region(destination, Role.INDEX_KEY)
        src_key, dst_key = region(source, Role.KEY), region(destination, Role.KEY)
        src, dst, sizes = [], [], []
        for a, b, size in zip(*plan.arrays()):
            a, b, size = int(a), int(b), int(size)
            left, right = max(b, missing.start), min(b + size, missing.end)
            if left >= right:
                src.append(a)
                dst.append(b)
                sizes.append(size)
            else:
                if left > b:
                    src.append(a)
                    dst.append(b)
                    sizes.append(left - b)
                if right < b + size:
                    src.append(a + right - b)
                    dst.append(right)
                    sizes.append(b + size - right)
        src.append(src_key.start)
        dst.append(dst_key.start)
        sizes.append(missing.end - missing.start)
        bad = coverage.freeze_write_plan(
            plan.binding, meta(plan, src=src, dst=dst, sizes=sizes), source, destination
        )
        self.assertEqual(plan.total_bytes, bad.total_bytes)
        with self.assertRaisesRegex(ValueError, "hole or overlap|incomplete"):
            coverage.certify_destination(destination, [bad], sources, writers)

    def test_scale_hole_swapped_roles_wrong_offset_and_truncated_pool_fail(self):
        sources, destination, plans, writers = cohort(1, 1)
        plan, source = plans[0], sources[("src", 0)]
        for role in (Role.KEY_BLOCK_SCALE, Role.INDEX_KEY):
            missing = region(destination, role)
            native = meta(plan)
            keep = np.array(
                [
                    not (int(dst) <= missing.start < int(dst) + int(size))
                    for dst, size in zip(native.dst_ptrs, native.sizes)
                ]
            )
            bad = coverage.freeze_write_plan(
                plan.binding,
                meta(
                    plan,
                    src=native.src_ptrs[keep],
                    dst=native.dst_ptrs[keep],
                    sizes=native.sizes[keep],
                ),
                source,
                destination,
            )
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "incomplete|hole"):
                coverage.certify_destination(destination, [bad], sources, writers)
        source_value, dest_key = region(source, Role.VALUE), region(destination, Role.KEY)
        for swapped in (
            source_value,
            region(source, Role.INDEX_KEY),
            region(source, Role.KEY_BLOCK_SCALE),
        ):
            with (
                self.subTest(swapped=swapped.buffer.role),
                self.assertRaisesRegex(ValueError, "swaps role"),
            ):
                coverage.freeze_write_plan(
                    plan.binding,
                    meta(plan, src=[swapped.start], dst=[dest_key.start], sizes=[1]),
                    source,
                    destination,
                )
        native = meta(plan)
        native.dst_ptrs[0] += 1
        with self.assertRaises(ValueError):
            coverage.freeze_write_plan(plan.binding, native, source, destination)
        broken = replace(
            destination, buffers=(replace(destination.buffers[0], size=1), *destination.buffers[1:])
        )
        with self.assertRaisesRegex(ValueError, "offset/size"):
            coverage.certify_destination(broken, plans, sources, writers)
        info = destination.rank_info
        info.page_table.pool_groups[0].pools[0].slot_bytes -= 1
        truncated = replace(destination, rank_info_bytes=info.to_bytes())
        with self.assertRaisesRegex(ValueError, "registered physical slot"):
            coverage.certify_destination(truncated, plans, sources, writers)

    def test_generation_claim_cohort_and_control_limits_fail_before_publication(self):
        sources, destination, plans, writers = cohort(2, 1)
        with self.assertRaisesRegex(ValueError, "full writer cohort"):
            coverage.certify_destination(destination, plans[:-1], sources, writers)
        mixed = [replace(plans[0], binding=replace(plans[0].binding, nonce="other")), *plans[1:]]
        with self.assertRaisesRegex(ValueError, "mixes claims"):
            coverage.certify_destination(destination, mixed, sources, writers)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            coverage.preview_write_plan(
                replace(plans[0].binding, source_generation="old-generation"),
                sources[("src", 0)],
                destination,
            )
        wire = destination.wire()
        wire["version"] = "old-peer"
        with self.assertRaisesRegex(ValueError, "protocol/version"):
            coverage.snapshot_from_trusted_wire(wire)
        oversized = replace(destination, generation="X" * (coverage.MAX_CONTROL_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "bounded control capacity"):
            oversized.wire()

    def test_frozen_source_arrays_and_each_completed_writer_are_bound(self):
        sources, destination, plans, writers = cohort(2, 1)
        by_rank = {plan.binding.source_rank: plan for plan in plans}
        for plan in plans:
            for values in plan.arrays():
                self.assertFalse(values.flags.writeable)
            coverage.check_completed_writer(
                by_rank, 71, plan.binding.source_rank, 0, True, plan.total_bytes
            )
            with self.assertRaisesRegex(ValueError, "completed writer"):
                coverage.check_completed_writer(
                    by_rank, 71, plan.binding.source_rank, 0, True, plan.total_bytes + 1
                )
            native = meta(plan)
            native.src_ptrs[0] += 1
            with self.assertRaisesRegex(ValueError, "actual native operation"):
                coverage.check_native_operation(plan, native)

    def test_actual_native_presubmit_hook_rejects_changed_fragments_before_descriptors(self):
        from tensorrt_llm._torch.disaggregation.native import transfer

        sources, destination, plans, writers = cohort(1, 1)
        source, plan = sources[("src", 0)], plans[0]
        info, peer = source.rank_info, destination.rank_info
        registrar = PeerRegistrar(info, KVRegionExtractorV1(info.page_table))
        registrar.register(peer.instance_name, peer.instance_rank, peer)
        operation = meta(plan)
        operation.meta_type = transfer.WriteMetaType.KV
        operation.task = NS(
            _retained_coverage_by_rank={peer.instance_rank: plan},
            _retained_native_registrar=registrar,
        )
        operation.dst_ptrs[0] += 1
        descriptors = Mock()
        with patch.object(transfer, "MemoryDescs", descriptors):
            with self.assertRaisesRegex(ValueError, "actual native operation"):
                transfer.Sender._make_agent_request(operation, device_id=info.device_id)
        descriptors.from_arrays_uniform_device.assert_not_called()
        changed = replace(peer, self_endpoint="different-peer-registration")
        with self.assertRaisesRegex(ValueError, "registration/device changed"):
            coverage.check_native_operation(
                plan, meta(plan), source_info=info, destination_info=changed
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
