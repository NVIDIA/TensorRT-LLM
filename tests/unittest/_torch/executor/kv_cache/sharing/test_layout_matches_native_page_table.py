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
"""``derive_layout`` against the page table native builds from the same live manager, for three
model shapes with small caches and no checkpoint: each transferred buffer's layer, pool, offset,
size and mapper, and each layer group's kind, window, heads, pools and roles against the unit
space of its life cycle; and each buffer's offset and size against the runtime's address and page
size of its layer and role."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.cache_kinds import BufferDecl
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import ShardDesc, derive_layout
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

# Both dense shapes: bf16, a 4096-token pool, sequences of up to 2048 tokens.
DENSE = dict(dtype=DataType.BF16, max_tokens=4096, max_seq_len=2048)
# TinyLlama-1.1B: 22 layers, 4 KV heads of 64. Rank 1 of TP2 holds heads 2 and 3.
TINYLLAMA_TP2_RANK1 = dict(
    num_layers=22, num_kv_heads=4, head_dim=64, mapping=Mapping(world_size=2, rank=1, tp_size=2)
)
# DeepSeek-V3-Lite: 30 MLA layers, one 576-wide latent per token (512 + 64 rope), K only.
DSV3_LITE_MLA = dict(
    kv_cache_type=CacheType.SELFKONLY,
    num_layers=30,
    num_kv_heads=1,
    head_dim=512 + 64,
    tokens_per_block=64,
)
# shape -> (keywords for conftest's ``real_manager``, or ``None`` for its ``hybrid_manager``: the
# Nemotron-Nano-9B-v2 shape with fewer layers; the shard every layer group names)
SHAPES = {
    "tinyllama_tp2_rank1": ({**DENSE, **TINYLLAMA_TP2_RANK1}, ShardDesc(2, 1)),
    "dsv3_lite_mla": ({**DENSE, **DSV3_LITE_MLA}, ShardDesc()),
    "nemotron_h_hybrid": (None, ShardDesc()),
}


def native_groups(page_table):
    """What native's page table says per unit space (one per layer group, indexed alike): each
    region as ``(layer, pool base, offset, size, declarations)``. Pools are named by base address;
    order-free parts are sorted."""
    out = []
    for space in page_table.spaces:
        pools = [
            (int(p.base_address), int(p.slot_bytes), int(p.num_slots))
            for p in (page_table.pools[i] for i in space.pools)
        ]
        state = space.unit == CacheKind.STATE
        rows = [
            (
                region.layer,
                int(page_table.pools[region.pool].base_address),
                region.offset,
                region.nbytes,
                region.buffers,
            )
            for region in space.regions
        ]
        out.append(
            {
                "kind": "state" if state else "attention",
                "window": None if state else space.params["window"],
                "layers": space.layers,
                "pools": pools,
                "rows": sorted(rows, key=lambda row: row[:4]),
            }
        )
    return out


def layout_groups(geometry):
    """The same per layer group from ``derive_layout``'s result, without local-only buffers (native
    skips ignored roles); transferred buffers no group claims are listed last. A group's pools are
    those of its device pool group its transferred buffers sit in, as native's unit space lists only
    the pools its regions use."""
    layout = geometry.layout
    out, claimed = [], set()
    for desc in layout.layer_groups:
        g = desc.pool_group
        device_pools = geometry.device_pools[g]
        assert layout.pool_groups[g] == tuple(p.slot_bytes for p in device_pools)
        base_of = {p.index: p.base for p in device_pools}
        rows = []
        for i, b in enumerate(layout.buffers):
            if b.layer not in desc.layers or b.pool[0] != g or not b.transfer:
                continue
            assert b.expansion == 1, "native reads no expansion"
            claimed.add(i)
            rows.append((b.layer, base_of[b.pool[1]], b.offset, b.size, b.decls))
        used = {base for _, base, _, _, _ in rows}
        pools = [(p.base, p.slot_bytes, p.num_slots) for p in device_pools if p.base in used]
        out.append(
            {
                "kind": desc.kind,
                "window": desc.window,
                "layers": sorted(desc.layers),
                "pools": pools,
                "rows": sorted(rows, key=lambda row: row[:4]),
            }
        )
    stray = [
        (b.layer, b.role) for i, b in enumerate(layout.buffers) if b.transfer and i not in claimed
    ]
    if stray:
        out.append({"unclaimed buffers": stray})
    return out


def lies(geometry):
    """Layout fields that each misstate one thing native reads."""
    layout = geometry.layout
    first = next(b for b in layout.buffers if b.transfer)

    def buffers(**change):
        return tuple(dataclasses.replace(b, **change) if b is first else b for b in layout.buffers)

    def first_group(**change):
        return tuple(
            dataclasses.replace(d, **change) if i == 0 else d
            for i, d in enumerate(layout.layer_groups)
        )

    # The first buffer split differently: replicated if it is sharded, sharded if replicated.
    domain = first.decls[0].domain
    other = (
        (BufferDecl(first.size, domain, (("byte", first.size),), "byte"),)
        if first.replicated
        else (BufferDecl(first.size, domain),)
    )
    out = {
        "offset": {"buffers": buffers(offset=first.offset + first.size)},
        "layer": {"buffers": buffers(layer=first.layer + 1000)},
        "split": {"buffers": buffers(decls=other)},
        "extra buffer": {
            "buffers": layout.buffers + (dataclasses.replace(first, layer=first.layer + 1000),)
        },
        "window": {"layer_groups": first_group(window=64)},
    }
    # The first layer group pointed at another device pool group, when the layout has one.
    mine = layout.layer_groups[0].pool_group
    others = sorted(g for g in layout.pool_groups if g != mine)
    if others:
        out["pool group"] = {"layer_groups": first_group(pool_group=others[0])}
    return out


@pytest.mark.parametrize("shape", list(SHAPES))
def test_layout_matches_the_page_table_native_builds(real_manager, hybrid_manager, shape):
    kwargs, shard = SHAPES[shape]
    with real_manager(**kwargs) if kwargs else hybrid_manager() as mgr:
        geometry = derive_layout(mgr)
        page_table = build_page_table_from_manager(mgr)
        native = native_groups(page_table)
        derived = layout_groups(geometry)

        layout = geometry.layout
        paged = [s for s in page_table.spaces if s.unit == CacheKind.PAGED]
        assert paged
        assert all(s.params["tokens_per_block"] == layout.tokens_per_block for s in paged)
        assert len(derived) == len(native), derived[len(native) :]
        assert all(group["rows"] for group in native)
        for lg, (ours, theirs) in enumerate(zip(derived, native)):
            for key in theirs:
                assert ours[key] == theirs[key], f"layer group {lg}: {key}"
        assert [d.shard for d in layout.layer_groups] == [shard] * len(native)
        assert geometry.shards == ((shard.count, shard.index),) * len(native)
        # The lender finds a group's pools through either; they must agree.
        assert tuple(d.pool_group for d in layout.layer_groups) == geometry.pool_group_of
        assert geometry.recurrent == tuple(g["kind"] == "state" for g in native)

        # The comparison is not vacuous: each misstated field shows.
        for name, change in lies(geometry).items():
            lie = dataclasses.replace(geometry, layout=dataclasses.replace(layout, **change))
            assert layout_groups(lie) != native, f"the {name} lie went unnoticed"

        # Every buffer sits where the runtime keeps its layer and role. The rows above carry no
        # role, so a layer's K and V trading places shows only here.
        internal = {
            region.layer: region.local_layer
            for space in page_table.spaces
            for region in space.regions
        }
        base = {(p.group, p.index): p.base for ps in geometry.device_pools.values() for p in ps}
        for b in layout.buffers:
            lid = internal[b.layer]
            address = int(mgr.impl.get_mem_pool_base_address(lid, b.role, PageIndexMode.SHARED))
            page = int(mgr.impl.get_page_stride(lid, b.role)) * b.expansion
            assert (address - base[b.pool], page) == (b.offset, b.size), (b.layer, b.role)
