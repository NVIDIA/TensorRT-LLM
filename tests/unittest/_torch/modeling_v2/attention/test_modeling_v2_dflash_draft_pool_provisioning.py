# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DFlash drafter's noise block stays on its own request's draft pages.

At a generation step the drafter stores its noise block's K/V at positions
``[ctx_after, ctx_after + block)`` of the draft KV pool, ``ctx_after`` being the request's
draft context after the step's ``accepted`` proposals and its bonus token. With every
proposal accepted the block's last rows reach ``history + 2 * max_draft_len + 2``. The draft
``KVCacheManagerV2`` must hold those positions at that forward: a row past the request's
pages goes through a block-table entry the manager never allocated, which the default
(cpp) V2 offsets carry as 0 and the drafter therefore decodes to pool page 0 -- a live page
that another request (or this one) owns.

The test runs that step on a real draft ``KVCacheManagerV2`` (FP8, one layer, 32-token
pages) and on a target ``KVCacheManagerV2`` whose explicit layer specs also hold the
drafter's FP8 layer (one KV cache for the target and the drafter; its block table comes
from the target's block offsets): request A is created first and owns pool page 0, holding
committed drafter context; request B is a long prompt whose draft history ends just before
a page boundary.
B's step grows the manager exactly as the scheduler does (``try_allocate_generation``), the
draft block table is decoded by ``DFlashWorker``'s own refresh, and the block is stored by
the drafter's own FP8 store at full acceptance. A's page-0 codes and B's committed codes
must be unchanged and every block row must land inside B's allocation. Needs one CUDA
device; no model files.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import dflash_store_fp8_kv
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_block_kv_lookahead
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

KV_HEADS, HEAD_DIM, PAGE = 2, 128, 32
#: Proposals per step and the drafter's block width (the pending token plus K mask slots).
K = 7
BLOCK = K + 1
#: Draft history of request A: one partially filled page.
A_HISTORY = 20
A_ID, B_ID = 1, 2


def _spec() -> DFlashDecodingConfig:
    return DFlashDecodingConfig(max_draft_len=K, attention_backend="TRTLLM")


def _manager() -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=512 * PAGE, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=8192,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8,
        spec_config=_spec(),
        is_draft=True,
    )


#: The drafter's layer id in a manager that also holds the target's layer 0.
DRAFT_LAYER = 1


def _unified_manager(draft_window) -> KVCacheManagerV2:
    """One manager for a target layer and the drafter's layer, both FP8.

    ``draft_window`` None gives both layers one lifecycle, so their equal-size pages coalesce
    into one pool; 4096 gives the drafter its own sliding-window lifecycle and pool.
    """
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        buffer_roles=(Role.KEY, Role.VALUE),
    )
    draft = KVCacheLayerSpec(
        global_layer_id=DRAFT_LAYER,
        owner=KVCacheLayerOwner.DRAFT,
        dtype=DataType.FP8,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=draft_window,
    )
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=512 * PAGE, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=8192,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8,
        spec_config=_spec(),
        layer_specs=[target, draft],
    )


#: Draft pool layouts: a separate draft manager, or the target's manager holding the drafter's
#: layer in its own lifecycle (the Beam layout) or coalesced with the target's layer.
LAYOUTS = {
    "draft-manager": lambda: _manager(),
    "unified-own-lifecycle": lambda: _unified_manager(4096),
    "unified-coalesced": lambda: _unified_manager(None),
}


def _generation_request(mgr: KVCacheManagerV2, request_id: int, history: int):
    """A generation request at rest: ``history`` committed draft positions, no step yet."""
    (req,) = mgr.add_dummy_requests(
        [request_id],
        token_nums=[history],
        is_gen=True,
        materialize_history=True,
        max_num_draft_tokens=K,
        kv_reserve_draft_tokens=0,
    )
    return req


def _draft_pool_and_tables(mgr: KVCacheManagerV2):
    """The drafter's FP8 pool view and this step's draft block table, both as the DFlash
    worker binds and decodes them (from the target's block offsets when ``mgr`` also holds
    the target's layer)."""
    worker = DFlashWorker(_spec(), Mapping(world_size=1, tp_size=1, rank=0))
    worker._ctx_unified_cache = not mgr.is_draft
    pool = worker._managed_ctx_pool(
        mgr, 1, KV_HEADS, HEAD_DIM, torch.float8_e4m3fn, 2, required=True
    )
    assert worker._init_ctx_block_tables(mgr, pool, 2, 1, KV_HEADS, HEAD_DIM, PAGE)
    num_pools = getattr(mgr, "num_attention_op_pools", mgr.num_pools)
    offsets = torch.zeros(
        (num_pools, 2, 2, mgr.max_blocks_per_seq), dtype=torch.int32, device="cuda"
    )
    mgr.copy_batch_block_offsets(offsets, [A_ID, B_ID], 1, 0, 2)
    name = "draft_kv_cache_block_offsets" if mgr.is_draft else "kv_cache_block_offsets"
    assert worker._refresh_ctx_block_tables(types.SimpleNamespace(**{name: offsets}), 2)
    return pool[0], worker._ctx_block_tables.long()


def _store(cache: torch.Tensor, table_row: torch.Tensor, positions: torch.Tensor, seed: int):
    """Store random K/V rows at ``positions`` through one request's block table."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(positions.numel(), KV_HEADS, HEAD_DIM, device="cuda", generator=gen)
    v = torch.randn(positions.numel(), KV_HEADS, HEAD_DIM, device="cuda", generator=gen)
    pages = table_row.gather(0, torch.div(positions, PAGE, rounding_mode="floor"))
    dflash_store_fp8_kv(cache, pages, positions % PAGE, k, v, 1.0)
    return pages


def _codes(cache: torch.Tensor, table_row: torch.Tensor, positions: torch.Tensor):
    pages = table_row.gather(0, torch.div(positions, PAGE, rounding_mode="floor"))
    return cache.view(torch.uint8)[pages, :, :, positions % PAGE].clone()


@pytest.mark.parametrize(
    "b_history",
    [
        # A 4097-token prompt after full-acceptance steps: the step's capacity ends one
        # position before, or exactly at, a page boundary, so one or two block rows lie
        # on the next logical page.
        pytest.param(4113, id="capacity-ends-one-before-page"),
        pytest.param(4114, id="capacity-ends-at-page"),
        pytest.param(4146, id="capacity-ends-at-next-page"),
        # Control: the block's tail stays on a page the step already allocated.
        pytest.param(4120, id="control-inside-page"),
    ],
)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_full_acceptance_block_stays_on_its_own_pages(layout, b_history):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    mgr = LAYOUTS[layout]()
    try:
        req_a = _generation_request(mgr, A_ID, A_HISTORY)
        req_b = _generation_request(mgr, B_ID, b_history)
        assert req_a is not None and req_b is not None

        # The scheduler's growth of B for this step, exactly as a generation step gets it.
        assert mgr.try_allocate_generation(req_b)
        capacity = mgr.kv_cache_map[B_ID].capacity
        cache, tables = _draft_pool_and_tables(mgr)
        assert cache.dtype == torch.float8_e4m3fn
        a_row, b_row = tables[0], tables[1]
        if layout == "draft-manager":
            # The premise: A owns pool page 0 and holds committed drafter context on it.
            assert int(a_row[0]) == 0
        a_positions = torch.arange(A_HISTORY, device="cuda")
        _store(cache, a_row, a_positions, seed=1)
        b_committed = torch.arange(b_history - 2 * PAGE, b_history, device="cuda")
        _store(cache, b_row, b_committed, seed=2)
        page0_before = cache.view(torch.uint8)[0].clone()
        a_codes_before = _codes(cache, a_row, a_positions)
        b_codes_before = _codes(cache, b_row, b_committed)

        # Full acceptance: the block starts after K accepted proposals and the bonus token.
        ctx_after = b_history + K + 1
        block = torch.arange(ctx_after, ctx_after + BLOCK, device="cuda")
        written = _store(cache, b_row, block, seed=3)
        torch.cuda.synchronize()
        diff = cache.view(torch.uint8)[0] != page0_before  # [K/V, heads, slot, dim]
        changed_slots = diff.any(dim=3).any(dim=1).any(dim=0).nonzero().flatten().tolist()
        assert not changed_slots, (
            f"request A's committed drafter context on pool page 0 changed at slots "
            f"{changed_slots}; B's block rows {block.tolist()} went to pages {written.tolist()}"
        )
        assert torch.equal(_codes(cache, a_row, a_positions), a_codes_before), (
            "request A's committed drafter context changed"
        )
        assert torch.equal(_codes(cache, b_row, b_committed), b_codes_before), (
            "request B's committed drafter context changed"
        )
        b_pages = int(torch.div(capacity + PAGE - 1, PAGE, rounding_mode="floor"))
        owned = set(b_row[:b_pages].tolist())
        assert set(written.tolist()) <= owned, (
            f"block rows landed on pages {sorted(set(written.tolist()) - owned)} that B does "
            "not hold"
        )
        assert int(block[-1]) < capacity, (
            f"the draft pool holds {capacity} positions of B at this forward; its block "
            f"writes up to position {int(block[-1])}"
        )
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_draft_pool_reserves_the_block_lookahead(layout):
    from tensorrt_llm._torch.speculative.dflash import check_draft_pool_reserve

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    mgr = LAYOUTS[layout]()
    try:
        spec = _spec()
        assert mgr._kv_reserve_draft_tokens == (
            spec.max_total_draft_tokens + dflash_block_kv_lookahead(spec)
        )
        check_draft_pool_reserve(spec, BLOCK, mgr)
    finally:
        mgr.shutdown()


class _Manager(types.SimpleNamespace):
    pass


@pytest.mark.parametrize("reserve,ok", [(K, False), (K + 1, False), (K + 2, True)])
def test_binding_refuses_a_pool_without_the_block_lookahead(reserve, ok):
    from tensorrt_llm._torch.speculative.dflash import check_draft_pool_reserve

    worker = DFlashWorker(_spec(), Mapping(world_size=1, tp_size=1, rank=0))
    manager = _Manager(_kv_reserve_draft_tokens=reserve)
    if ok:
        check_draft_pool_reserve(worker.spec_config, BLOCK, manager)
    else:
        with pytest.raises(RuntimeError, match="noise block"):
            check_draft_pool_reserve(worker.spec_config, BLOCK, manager)
