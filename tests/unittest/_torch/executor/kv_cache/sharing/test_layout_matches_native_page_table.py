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
"""Live lender layouts are the pages native transfers, across model and parallel layouts, and
native's correspondence plans transfers between the lender layouts of two parallel layouts."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.page_mapping import (
    RankPages,
    buffer_runs,
    match_buffers,
    plan_transfers,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_builder import device_pools, layer_configs
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import BufferDesc, PageLayout, ShardDesc
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


def split_of(buffer: BufferDesc) -> str:
    """How a buffer's bytes split across ranks: REPLICATED, HND (heads outermost), NHD (heads
    within each token) or SECTIONED."""
    if all(segment.split_axis is None for segment in buffer.segments):
        return "REPLICATED"
    if all(segment.split_axis == "byte" for segment in buffer.segments):
        return "SECTIONED"
    (segment,) = buffer.segments
    return {("head",): "HND", ("token", "head"): "NHD"}[tuple(name for name, _ in segment.axes)]


def roles(layout):
    return {(b.role, split_of(b)) for b in layout.buffers}


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
    per_head = {b.segments[0].unit_bytes for b in layout.buffers}
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
        sections = {
            tuple(segment.nbytes for segment in b.segments)
            for b in layout.buffers
            if split_of(b) == "SECTIONED"
        }
        assert sections and all(len(s) == 3 and s[0] == s[1] < s[2] for s in sections), sections

    return check


def check_minimax_m3(scales, *, nhd=False):
    def check(mgr, layout):
        held = roles(layout)
        if nhd:
            assert any(split == "NHD" for _, split in held), sorted(held)
        assert ("index_key", "REPLICATED") in held, sorted(held)
        assert (("key_block_scale", "HND") in held) == scales, sorted(held)

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


def as_native_sends(mgr, layout: PageLayout) -> PageLayout:
    """``layout`` with the roles native leaves to a receive-time rebuild not transferred: the
    lender, which invokes no rebuild, transfers every role."""
    if not callable(getattr(mgr, "rebuild_disagg_imported_cache", None)):
        return layout
    untransferred = mgr.get_untransferred_roles()
    return dataclasses.replace(
        layout,
        buffers=tuple(
            dataclasses.replace(b, transferred=b.role not in untransferred) for b in layout.buffers
        ),
    )


def lent_pages(mgr, lender) -> RankPages:
    """The pages a lender's layout describes, without its region numbering."""
    layout = dataclasses.replace(
        lender.page_layout, device_region_index={}, staging_region_index={}
    )
    pools = tuple(p for group in device_pools(mgr).values() for p in group)
    return RankPages(as_native_sends(mgr, layout), pools)


# RankTopology and Mapping name the same rank coordinates.
PLACE = ("tp_size", "tp_rank", "pp_size", "pp_rank", "cp_size", "cp_rank", "dp_size", "dp_rank")


def topology(layout):
    return tuple(getattr(layout.topology, name) for name in PLACE) + (layout.topology.attention_dp,)


def manager_topology(mgr):
    mapping = mgr.mapping
    coordinates = tuple(getattr(mapping, name) for name in PLACE[:6])
    dp_size = mapping.tp_size if mapping.enable_attention_dp else mapping.dp_size
    dp_rank = mapping.tp_rank if mapping.enable_attention_dp else 0
    return coordinates + (dp_size, dp_rank, mapping.enable_attention_dp)


@pytest.mark.parametrize("case", [pytest.param(name, marks=MARKS.get(name, [])) for name in CASES])
def test_the_public_layout_is_the_page_layout_native_transfers(
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
        native = RankPages.of_manager(mgr)
        assert native.layout == lent_pages(mgr, lender).layout

        # Each pool native names is a device region of the lender, of its slots.
        pools = sorted((p.group, p.index) for p in native.pools)
        assert sorted(layout.device_region_index) == pools
        for pool in native.pools:
            region = regions[layout.device_region_index[(pool.group, pool.index)]]
            assert region.memory_kind is MemoryKind.DEVICE, (pool, region)
            assert (region.address, region.slot_bytes, region.slot_count) == (
                pool.base,
                pool.slot_bytes,
                pool.num_slots,
            ), (pool, region)
            assert pool.slot_bytes == layout.pool_slot_bytes[pool.group][pool.index], pool
        assert all(native.buffers(g) for g in range(native.num_layer_groups))
        assert [d.shard for d in layout.layer_groups] == [spec.shard] * len(layout.layer_groups)
        assert all(b.expansion == 1 for b in layout.buffers if b.transferred), "native reads none"
        assert topology(layout) == manager_topology(mgr), (layout.topology, mgr.mapping)
        assert layout.dtype == mgr.dtype.name
        assert layout.quant_mode == spec.quant, layout.quant_mode
        if spec.total_heads is not None:
            expected = {
                layer: spec.total_heads
                for desc in layout.layer_groups
                if desc.kind != "state"
                for layer in desc.layers
            }
            assert dict(layout.total_kv_heads_by_layer) == expected

        # Every attention group keeps its layers' configured window, and every buffer sits where
        # the runtime keeps its layer and role: a layer's K and V trading places shows here.
        internal = {global_layer: lid for lid, (global_layer, _) in mgr.get_layer_ids().items()}
        configs = layer_configs(mgr)
        for desc in layout.layer_groups:
            if desc.kind == "attention":
                window = configs[internal[desc.layers[0]]].window_size
                assert desc.kept_window == window, (desc, window)
        for b in layout.buffers:
            lid = internal[b.layer]
            address = int(mgr.impl.get_mem_pool_base_address(lid, b.role, PageIndexMode.SHARED))
            base = regions[layout.device_region_index[(b.pool_group, b.pool_index)]].address
            page = int(mgr.impl.get_page_stride(lid, b.role)) * b.expansion
            assert (address - base, page) == (b.slot_offset, b.size), (b.layer, b.role)


# name: (how to build the context rank, how to build the generation rank).
PAIRS = {
    "tinyllama_tp1_to_tp2_rank1": (
        lambda f: f.real(**DENSE, **TINYLLAMA),
        lambda f: f.real(**DENSE, **TINYLLAMA, mapping=tp(1)),
    ),
    "nemotron_h_hybrid_tp1_to_tp2_rank1": (
        lambda f: f.hybrid(),
        lambda f: f.hybrid(mapping=tp(1)),
    ),
    "qwen3_next_gdn_tp1_to_tp2_rank1": (
        lambda f: f.gdn(),
        lambda f: f.gdn(mapping=tp(1)),
    ),
}


def rank_info(name, mgr, lender) -> RankInfo:
    """Native's rank info of ``mgr``, carrying the pages its lender's layout describes."""
    info = RankInfo.from_kv_cache_manager(name, mgr, device_id=torch.cuda.current_device())
    return dataclasses.replace(info, pages=lent_pages(mgr, lender))


@pytest.mark.parametrize("pair", list(PAIRS))
def test_native_plans_transfers_between_the_lender_layouts_of_two_parallel_layouts(
    pair, kit, real_manager, hybrid_manager, gdn_manager, monkeypatch
):
    """A context rank holding every head sends each generation-rank buffer its whole share: the
    generation rank's half of each split buffer, a replicated buffer whole."""
    fixtures = SimpleNamespace(
        real=real_manager, hybrid=hybrid_manager, gdn=gdn_manager, kit=kit, monkeypatch=monkeypatch
    )
    build_context, build_generation = PAIRS[pair]
    with build_context(fixtures) as ctx_mgr, build_generation(fixtures) as gen_mgr:
        ctx = rank_info("ctx", ctx_mgr, attach(ctx_mgr))
        gen = rank_info("gen", gen_mgr, attach(gen_mgr))
        src_pages, dst_pages = ctx.pages, gen.pages
        matched = match_buffers(src_pages, dst_pages)
        sent = {
            (dst_pages.buffers(pg)[pi].layer, dst_pages.buffers(pg)[pi].role)
            for pg, pi in matched.values()
        }
        received = {
            (b.layer, b.role)
            for g in range(dst_pages.num_layer_groups)
            for b in dst_pages.buffers(g)
        }
        assert sent == received, received - sent
        for (g, i), (pg, pi) in matched.items():
            src, dst = src_pages.buffers(g)[i], dst_pages.buffers(pg)[pi]
            runs = buffer_runs(src, dst, src_pages.group(g).grid, ctx, gen)
            assert sum(n for _, _, n in runs) == dst.size, (src, dst, runs)
            for s, d, n in runs:
                assert src.slot_offset <= s and s + n <= src.slot_offset + src.size, (src, runs)
                assert dst.slot_offset <= d and d + n <= dst.slot_offset + dst.size, (dst, runs)
            if src.segments == dst.segments:
                assert runs == [(src.slot_offset, dst.slot_offset, src.size)], (src, dst, runs)
            elif len(src.segments) == 1 and src.segments[0].rows == 1:
                # Rank 1 of two holds the second half of the heads.
                assert runs == [(src.slot_offset + dst.size, dst.slot_offset, dst.size)], runs
        groups = plan_transfers(ctx, gen, src_pages, dst_pages)
        planned = sorted((group.self_layer_group, i) for group in groups for i in group.buffers)
        assert planned == sorted(matched), "a matched buffer is in no transfer group"
        assert all(not group.plan.is_empty for group in groups)

        def described(planned_groups):
            return [
                (g.self_layer_group, g.peer_layer_group, g.self_pool, g.peer_pool, g.buffers)
                + (g.grid, g.owner_factor, g.plan.runs)
                for g in planned_groups
            ]

        native = plan_transfers(
            ctx, gen, RankPages.of_manager(ctx_mgr), RankPages.of_manager(gen_mgr)
        )
        assert described(native) == described(groups)
