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
"""Compare live lender layouts with native page tables across model and parallel layouts."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    MapperKind,
    ShardDesc,
    head_major,
    replicated,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    MemoryKind,
    StagingOptions,
    attach_lender,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools")

# The dense shapes: bf16, a 4096-token pool, sequences of up to 2048 tokens.
DENSE = dict(dtype=DataType.BF16, max_tokens=4096, max_seq_len=2048)
# TinyLlama-1.1B: 22 layers, 4 KV heads of 64.
TINYLLAMA = dict(num_layers=22, num_kv_heads=4, head_dim=64)
# Its heads on five layers, which two pipeline stages split unevenly.
FIVE_LAYERS = dict(num_layers=5, num_kv_heads=4, head_dim=64)
# DeepSeek-V3-Lite: 30 MLA layers, one 576-wide latent per token (512 + 64 rope), K only.
DSV3_LITE_MLA = dict(
    kv_cache_type=CacheType.SELFKONLY,
    num_layers=30,
    num_kv_heads=1,
    head_dim=512 + 64,
    tokens_per_block=64,
)


def tp(rank, size=2, **kwargs):
    return Mapping(world_size=size, rank=rank, tp_size=size, **kwargs)


def pp2(rank):
    return Mapping(world_size=2, rank=rank, pp_size=2)


def helix2(rank):
    return Mapping(world_size=2, rank=rank, cp_size=2, cp_config={"cp_type": CpType.HELIX})


def minimax_m3(kit, kv_dtype, implementation="msa"):
    """A four-layer MiniMax-M3 manager: layers 0 to 2 dense, layer 3 sparse with a bf16 index-K."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    return kit.managed(
        lambda: MiniMaxM3KVCacheManagerV2(
            kv_cache_config=KvCacheConfig(
                max_tokens=1024, dtype=kv_dtype, enable_block_reuse=True, host_cache_size=0
            ),
            kv_cache_type=CacheType.SELF,
            num_layers=4,
            num_kv_heads=2,
            head_dim=128,
            tokens_per_block=128,
            max_seq_len=512,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            dtype={"nvfp4": DataType.NVFP4, "fp8": DataType.FP8}[kv_dtype],
            vocab_size=32000,
            max_num_tokens=512,
            sparse_layer_ids=[3],
            disable_index_value_layer_ids=[3],
            sparse_index_dim=128,
            sparse_attention_config=SimpleNamespace(
                implementation=implementation, indexer_kv_dtype="bf16"
            ),
        )
    )


def csa2(f):
    """The six-layer CSA2 manager native's CSA2 transfer test builds, bounded replay off."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    f.monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    f.monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0")
    layout = CSA2Layout((0, 2, 2, 1, 1, 1), (1, 3), (1, 3, 4), candidate_source_layer_id=3)
    return f.kit.managed(
        lambda: CSA2CacheManager(
            KvCacheConfig(max_gpu_total_bytes=128 << 20, enable_block_reuse=False, dtype="fp8"),
            CacheType.SELFKONLY,
            num_layers=len(layout.compress_ratios),
            tokens_per_block=128,
            max_seq_len=512,
            max_batch_size=2,
            max_num_tokens=1024,
            vocab_size=129280,
            dtype=DataType.BF16,
            mapping=Mapping(),
            layout=layout,
            is_disagg=True,
        )
    )


def deepseek_v4_mtp(f):
    """The DeepSeek-V4 manager with one MTP layer, which it keeps as a virtual layer of its own."""
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec_config = MTPDecodingConfig(max_draft_len=1)
    spec_config.num_nextn_predict_layers = 1  # the checkpoint sets it at model load
    return f.dsv4(spec_config=spec_config)


def kinds(layout):
    return [desc.kind for desc in layout.layer_groups]


def mapper_of(b):
    """The native mapper kind a buffer's tensors amount to."""
    if not b.split:
        return "REPLICATED"
    (tensor, *rest) = b.tensors
    names = [a.name for a in tensor.axes]
    if "shard" in names:
        return "INDEXED"  # a rank's own bytes, its heads undeclared
    if rest or "head" not in names:
        return "SECTIONED"
    return "NHD" if names[0] == "token" else "INDEXED"


def sections_of(b):
    """A sectioned buffer's section sizes on this rank, as native's ``section_bytes``; None
    otherwise."""
    if mapper_of(b) != "SECTIONED":
        return None
    return tuple(t.axes[0].held for t in b.tensors)


def head_bytes_of(b):
    """Bytes of one head on this rank of a buffer split by head, as native's ``bytes_per_head``;
    None otherwise."""
    head = next((a for a in b.tensors[0].axes if a.name == "head"), None)
    if mapper_of(b) not in ("INDEXED", "NHD") or head is None:
        return None
    return b.size // head.held


def roles(layout):
    return {(b.role, mapper_of(b)) for b in layout.buffers}


def check_tp2(mgr, layout):
    heads = set(mgr.num_kv_heads_per_layer)
    assert (mgr.mapping.tp_size, mgr.mapping.tp_rank, heads) == (2, 1, {2}), (mgr.mapping, heads)


def check_tp4_replicated(mgr, layout):
    assert (mgr.mapping.tp_size, mgr.mapping.tp_rank) == (4, 3)
    assert set(mgr.num_kv_heads_per_layer) == {1}


def check_attention_dp(mgr, layout):
    heads = set(mgr.num_kv_heads_per_layer)
    assert mgr.mapping.enable_attention_dp and heads == {4}, (mgr.mapping, heads)


def check_stage(rank):
    def check(mgr, layout):
        held = len(mgr.pp_layers)
        assert mgr.mapping.pp_rank == rank and 0 < held < 5, (mgr.mapping, mgr.pp_layers)

    return check


def check_vswa(mgr, layout):
    windows = sorted(desc.kept_window or 0 for desc in layout.layer_groups)
    assert windows == [0, 64], f"layer group windows {windows}, not one full and one of 64"


def check_mla(tp_size):
    def check(mgr, layout):
        heads = set(mgr.num_kv_heads_per_layer)
        assert (mgr.kv_factor, mgr.mapping.tp_size, heads) == (1, tp_size, {1}), mgr.mapping

    return check


def check_helix(mgr, layout):
    assert mgr._has_cp_helix and (mgr._helix_cp_rank, mgr._helix_cp_size) == (1, 2), mgr.mapping
    # A page holds tokens_per_block tokens of each head's K and V, two bytes each (fp16).
    per_head = {head_bytes_of(b) for b in layout.buffers}
    assert per_head == {layout.tokens_per_page * mgr.head_dim * 2}, (per_head, mgr.head_dim)


def check_beams(mgr, layout):
    assert mgr.max_beam_width == 2, mgr.max_beam_width


def check_hybrid(tp_size):
    def check(mgr, layout):
        assert mgr.mapping.tp_size == tp_size, mgr.mapping
        assert set(kinds(layout)) == {"attention", "state"}, kinds(layout)

    return check


def check_deepseek_v4(mgr, layout):
    widths = {g: len(w) for g, w in layout.pool_slot_bytes.items()}
    assert mgr._layer_attn_to_layer_id and max(widths.values()) >= 2, widths


def check_deepseek_v4_mtp(mgr, layout):
    check_deepseek_v4(mgr, layout)
    model_layers = {model_layer for model_layer, _ in mgr._layer_attn_to_layer_id}
    assert mgr.max_draft_len == 1 and max(model_layers) == 4, (mgr.max_draft_len, model_layers)


def check_csa2(mgr, layout):
    assert mgr._physical_roles and mgr._decoder_replay_window == 0, mgr._decoder_replay_window
    assert any(desc.kept_window is not None for desc in layout.layer_groups), kinds(layout)


def check_gdn(tp_size, stage_kinds):
    def check(mgr, layout):
        assert mgr.mapping.tp_size == tp_size, mgr.mapping
        assert set(kinds(layout)) == stage_kinds, kinds(layout)
        sections = {sections_of(b) for b in layout.buffers if mapper_of(b) == "SECTIONED"}
        assert sections and all(len(s) == 3 and s[0] == s[1] < s[2] for s in sections), sections

    return check


def check_minimax_m3(scales, *, nhd=False):
    def check(mgr, layout):
        held = roles(layout)
        if nhd:
            assert any(mapper == "NHD" for _, mapper in held), sorted(held)
        assert ("index_key", "REPLICATED") in held, sorted(held)
        assert (("key_block_scale", "INDEXED") in held) == scales, sorted(held)

    return check


@dataclasses.dataclass(frozen=True)
class Case:
    """One rank's manager, what it must be, and the shard, heads and quantization it names."""

    build: Callable
    check: Callable
    shard: ShardDesc = ShardDesc()
    total_heads: int | None = None
    quant: str | None = None


CASES = {
    "tinyllama_tp2_rank1": Case(
        lambda f: f.real(**DENSE, **TINYLLAMA, mapping=tp(1)), check_tp2, ShardDesc(2, 1), 4
    ),
    "tp4_two_heads_rank3": Case(
        lambda f: f.real(**{**DENSE, **TINYLLAMA, "num_kv_heads": 2}, mapping=tp(3, size=4)),
        check_tp4_replicated,
        ShardDesc(2, 1),
        2,
    ),
    "minimax_m3_triton_fp8": Case(
        lambda f: minimax_m3(f.kit, "fp8", implementation="triton"),
        check_minimax_m3(scales=False, nhd=True),
        total_heads=2,
        quant="fp8",
    ),
    "attention_dp_rank1": Case(
        lambda f: f.real(**DENSE, **TINYLLAMA, mapping=tp(1, enable_attention_dp=True)),
        check_attention_dp,
        total_heads=4,
    ),
    "pp2_stage0": Case(
        lambda f: f.real(**DENSE, **FIVE_LAYERS, mapping=pp2(0)), check_stage(0), total_heads=4
    ),
    "pp2_stage1": Case(
        lambda f: f.real(**DENSE, **FIVE_LAYERS, mapping=pp2(1)), check_stage(1), total_heads=4
    ),
    "vswa": Case(
        lambda f: f.real(**DENSE, **FIVE_LAYERS, windows=[64, DENSE["max_seq_len"]]),
        check_vswa,
        total_heads=4,
    ),
    "dsv3_lite_mla": Case(lambda f: f.real(**DENSE, **DSV3_LITE_MLA), check_mla(1), total_heads=1),
    # One latent head, which every tensor-parallel rank holds whole.
    "dsv3_lite_mla_tp2_rank1": Case(
        lambda f: f.real(**DENSE, **DSV3_LITE_MLA, mapping=tp(1)), check_mla(2), total_heads=1
    ),
    "helix_rank1": Case(
        lambda f: f.real(mapping=helix2(1), enable_block_reuse=False), check_helix, total_heads=4
    ),
    "beam2": Case(
        lambda f: f.real(**DENSE, **TINYLLAMA, max_beam_width=2), check_beams, total_heads=4
    ),
    "nemotron_h_hybrid": Case(lambda f: f.hybrid(), check_hybrid(1), total_heads=8),
    "nemotron_h_hybrid_tp2_rank1": Case(
        lambda f: f.hybrid(mapping=tp(1)), check_hybrid(2), ShardDesc(2, 1), 8
    ),
    "deepseek_v4": Case(lambda f: f.dsv4(), check_deepseek_v4, quant="fp8"),
    "deepseek_v4_mtp": Case(deepseek_v4_mtp, check_deepseek_v4_mtp, quant="fp8"),
    "csa2_full_prefill": Case(csa2, check_csa2),
    "qwen3_next_gdn": Case(lambda f: f.gdn(), check_gdn(1, {"attention", "state"}), total_heads=2),
    "qwen3_next_gdn_tp2_rank1": Case(
        lambda f: f.gdn(mapping=tp(1)), check_gdn(2, {"attention", "state"}), ShardDesc(2, 1), 2
    ),
    # Layers 0 and 1 of "MMM*": a stage holding recurrent state alone.
    "qwen3_next_gdn_pp2_stage0": Case(
        lambda f: f.gdn(mapping=pp2(0)), check_gdn(1, {"state"}), total_heads=2
    ),
    "qwen3_next_gdn_pp2_stage1": Case(
        lambda f: f.gdn(mapping=pp2(1)), check_gdn(1, {"attention", "state"}), total_heads=2
    ),
    "minimax_m3_fp8": Case(
        lambda f: minimax_m3(f.kit, "fp8"),
        check_minimax_m3(scales=False),
        total_heads=2,
        quant="fp8",
    ),
    "minimax_m3_nvfp4": Case(
        lambda f: minimax_m3(f.kit, "nvfp4"),
        check_minimax_m3(scales=True),
        total_heads=2,
        quant="nvfp4",
    ),
}
MARKS = {
    "deepseek_v4": [skip_pre_blackwell],
    "deepseek_v4_mtp": [skip_pre_blackwell],
    "csa2_full_prefill": [skip_pre_blackwell],
    "minimax_m3_fp8": [skip_pre_hopper],
    "minimax_m3_triton_fp8": [skip_pre_hopper],
    "minimax_m3_nvfp4": [skip_pre_hopper],
}


def attach(mgr):
    return attach_lender(
        mgr, scope=b"page-table", staging=StagingOptions(int(mgr.tokens_per_block))
    )


def _as_list(values):
    return None if values is None else [int(v) for v in values]


def native_groups(page_table):
    """What native's page table says per layer group. Pools are named by base address, layers by
    the global ids of ``local_layers``; order-free parts are sorted."""
    out = []
    for lg in page_table.layer_groups:
        pools = [
            (int(p.base_address), int(p.slot_bytes), int(p.num_slots))
            for p in page_table.pool_groups[lg.pool_group_idx].pools
        ]
        global_of = {int(ll.local_layer_id): int(ll.global_layer_id) for ll in lg.local_layers}
        state = lg.kind == CacheKind.STATE
        rows, roles, geometry = [], {}, {}
        for pv in lg.pool_views:
            key = (pools[pv.pool_idx][0], MapperKind(int(pv.mapper_kind)).name)
            assert key not in roles, f"two views of pool and mapper {key}"
            roles[key] = sorted(pv.pool_role)
            if state:
                geometry[key] = (_as_list(pv.section_bytes), pv.bytes_per_head)
            for e in pv.buffer_entries:
                layer = global_of[int(e["local_layer_id"])]
                rows.append((layer, key[0], int(e["offset"]), int(e["size"]), key[1]))
        out.append(
            {
                "kind": "state" if state else "attention",
                "window": None if state else lg.sliding_window_size,
                "heads": None if state else int(lg.kv_head_num_per_rank),
                "layers": sorted(global_of.values()),
                "pools": pools,
                "rows": sorted(rows),
                "roles": roles,
                "state_geometry": geometry,
            }
        )
    return out


def public_groups(layout, regions):
    """Project public transferred buffers into native's group, pool and role representation."""
    out, claimed = [], set()
    for desc in layout.layer_groups:
        g = desc.pool_group
        pools = []
        for p in range(len(layout.pool_slot_bytes.get(g, ()))):
            region = regions[layout.device_region_index[(g, p)]]
            pools.append((region.address, region.slot_bytes, region.slot_count))
        state = desc.kind == "state"
        rows, roles, geometries, heads = [], defaultdict(list), defaultdict(set), set()
        whole = set()
        for i, b in enumerate(layout.buffers):
            if b.layer not in desc.layers or b.pool_group != g or not b.transferred:
                continue
            claimed.add(i)
            key = (pools[b.pool_index][0], mapper_of(b))
            rows.append((b.layer, key[0], b.slot_offset, b.size, key[1]))
            roles[key].append(b.role)
            if state:
                geometries[key].add((sections_of(b), head_bytes_of(b)))
            elif key[1] in ("INDEXED", "NHD"):
                heads.add(b.size // head_bytes_of(b))
            elif key[1] == "REPLICATED":
                whole.add(layout.total_kv_heads_by_layer.get(b.layer))
        heads = heads or whole
        assert all(len(v) == 1 for v in geometries.values()), f"roles disagree: {dict(geometries)}"
        out.append(
            {
                "kind": desc.kind,
                "window": desc.kept_window,
                "heads": None if state else (heads.pop() if len(heads) == 1 else sorted(heads)),
                "layers": sorted(desc.layers),
                "pools": pools,
                "rows": sorted(rows),
                "roles": {k: sorted(set(v)) for k, v in roles.items()},
                "state_geometry": {k: (_as_list(s), h) for k, ((s, h),) in geometries.items()},
            }
        )
    stray = [
        (b.layer, b.role)
        for i, b in enumerate(layout.buffers)
        if b.transferred and i not in claimed
    ]
    if stray:
        out.append({"unclaimed buffers": stray})
    return out


# RankTopology and Mapping name the same rank coordinates.
PLACE = ("tp_size", "tp_rank", "pp_size", "pp_rank", "cp_size", "cp_rank", "dp_size", "dp_rank")


def topology(layout):
    return tuple(getattr(layout.topology, name) for name in PLACE) + (layout.topology.attention_dp,)


def manager_topology(mgr):
    mapping = mgr.mapping
    coordinates = tuple(getattr(mapping, name) for name in PLACE[:6])
    dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
    return coordinates + (mapping.dp_size, dp_rank, mapping.enable_attention_dp)


def lies(layout):
    """Public layouts that each misstate one thing native reads."""
    first = next(b for b in layout.buffers if b.transferred)

    def buffers(**change):
        return tuple(dataclasses.replace(b, **change) if b is first else b for b in layout.buffers)

    def first_group(**change):
        return tuple(
            dataclasses.replace(d, **change) if i == 0 else d
            for i, d in enumerate(layout.layer_groups)
        )

    other = (
        head_major(first.size, 1, 1) if mapper_of(first) == "REPLICATED" else replicated(first.size)
    )
    window = 128 if layout.layer_groups[0].kept_window == 64 else 64
    out = {
        "offset": dict(buffers=buffers(slot_offset=first.slot_offset + first.size)),
        "layer": dict(buffers=buffers(layer=first.layer + 1000)),
        "mapper": dict(buffers=buffers(tensors=other)),
        "extra buffer": dict(
            buffers=layout.buffers + (dataclasses.replace(first, layer=first.layer + 1000),)
        ),
        # The read window follows the misstated kept window, which it may not exceed.
        "window": dict(layer_groups=first_group(kept_window=window, read_window=None)),
    }
    # The first layer group pointed at another device pool group, when the layout has one.
    mine = layout.layer_groups[0].pool_group
    others = sorted(g for g in layout.pool_slot_bytes if g != mine)
    if others:
        out["pool group"] = dict(layer_groups=first_group(pool_group=others[0]))
    # Two pools' regions traded, when there are two.
    pools = sorted(layout.device_region_index)
    if len(pools) >= 2:
        traded = dict(layout.device_region_index)
        traded[pools[0]], traded[pools[1]] = traded[pools[1]], traded[pools[0]]
        out["regions"] = dict(device_region_index=traded)
    return {name: dataclasses.replace(layout, **change) for name, change in out.items()}


@pytest.mark.parametrize("case", [pytest.param(name, marks=MARKS.get(name, [])) for name in CASES])
def test_the_public_layout_matches_what_native_builds(
    case, kit, real_manager, hybrid_manager, gdn_manager, deepseek_v4_manager, monkeypatch
):
    spec = CASES[case]
    fixtures = SimpleNamespace(
        real=real_manager,
        hybrid=hybrid_manager,
        gdn=gdn_manager,
        dsv4=deepseek_v4_manager,
        kit=kit,
        monkeypatch=monkeypatch,
    )
    with spec.build(fixtures) as mgr:
        lender = attach(mgr)
        layout, regions = lender.page_layout, lender.regions
        spec.check(mgr, layout)
        page_table = build_page_table_from_manager(mgr)
        native = native_groups(page_table)
        derived = public_groups(layout, regions)

        # Each pool's region is a device region of its slot width.
        for (g, p), position in layout.device_region_index.items():
            region = regions[position]
            assert region.memory_kind is MemoryKind.DEVICE, (g, p, region)
            assert region.slot_bytes == layout.pool_slot_bytes[g][p], (g, p, region)
        assert layout.tokens_per_page * layout.topology.cp_size == page_table.tokens_per_block
        assert len(derived) == len(native), derived[len(native) :]
        assert all(group["rows"] for group in native)
        for lg, (ours, theirs) in enumerate(zip(derived, native)):
            for key in theirs:
                assert ours[key] == theirs[key], f"layer group {lg}: {key}"
        assert [d.shard for d in layout.layer_groups] == [spec.shard] * len(native)
        assert all(b.expansion == 1 for b in layout.buffers if b.transferred), "native reads none"
        assert topology(layout) == manager_topology(mgr), (layout.topology, mgr.mapping)
        assert layout.dtype == mgr.dtype.name
        assert layout.quant_mode == spec.quant, layout.quant_mode
        if spec.total_heads is not None:
            expected = {
                int(layer.global_layer_id): spec.total_heads
                for group in page_table.layer_groups
                if group.kind != CacheKind.STATE
                for layer in group.local_layers
            }
            assert dict(layout.total_kv_heads_by_layer) == expected

        # The comparison is not vacuous: each misstated field shows.
        for name, lie in lies(layout).items():
            assert public_groups(lie, regions) != native, f"the {name} lie went unnoticed"

        # Every buffer sits where the runtime keeps its layer and role. Native's views list roles
        # per pool and mapper, so a layer's K and V trading places shows only here.
        internal = {
            int(ll.global_layer_id): int(ll.local_layer_id)
            for lg in page_table.layer_groups
            for ll in lg.local_layers
        }
        for b in layout.buffers:
            lid = internal[b.layer]
            address = int(mgr.impl.get_mem_pool_base_address(lid, b.role, PageIndexMode.SHARED))
            base = regions[layout.device_region_index[(b.pool_group, b.pool_index)]].address
            page = int(mgr.impl.get_page_stride(lid, b.role)) * b.expansion
            assert (address - base, page) == (b.slot_offset, b.size), (b.layer, b.role)
