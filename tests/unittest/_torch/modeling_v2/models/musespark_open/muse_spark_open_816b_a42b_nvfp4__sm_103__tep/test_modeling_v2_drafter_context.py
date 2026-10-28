# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target's DFlash drafter paths over each layout of the drafter's context.

* ``dflash_window_attention`` and ``dflash_window_attention_wide`` (at each device's split
  geometry) over pages are bitwise their dense calls, the appended block included.
* ``context_kv_fork``'s commit empties the dummy slot, as the worker's own commit does.
* The block decode over a unified target/draft ``KVCacheManagerV2`` is bitwise the dense one.
* Over pages ``compose`` plans every sm103 TEP4 fast path but the window compaction.
"""

import importlib.util
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_drafter_window_attention import (  # noqa: E501
    attention as wide_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    dflash_window_attention_wide as wide_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    WINDOW,
    dflash_window_attention,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.speculative import dflash_attention
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _randn(gen, *shape):
    return torch.randn(*shape, device="cuda", generator=gen).bfloat16()


@pytest.mark.parametrize("append", [False, True], ids=["read", "append"])
@pytest.mark.parametrize("sms", [None, *sorted(wide_ops.WIDE_GEOMETRY_BY_SMS)])
def test_the_window_ops_over_pages_are_their_dense_calls(sms, append, monkeypatch):
    """``sms``: the wide op at that device's split geometry, run on the device at hand; None:
    the single-request op. Eight requests, a TP4 rank's heads, the served window, PDL."""
    op = dflash_window_attention
    if sms is not None:
        monkeypatch.setattr(wide_ops, "_device_sms", lambda index: sms)
        op = wide_ops.dflash_window_attention_wide
    gen = torch.Generator(device="cuda").manual_seed(0)
    # Windows starting inside a page; blocks inside one, straddling two or starting one.
    ctx = [5_000, 13, 1_020, 2_047, 300, 64, 1, 3_333]
    batch, m, hq, hkv, d, page, capacity = len(ctx), 8, 16, 2, 128, 32, 5_024
    slots, per_slot = batch + 1, capacity // page
    k_arena, v_arena = _randn(gen, slots, capacity, hkv, d), _randn(gen, slots, capacity, hkv, d)
    q, new = _randn(gen, batch, m, hq, d) * 2, {}
    if append:
        new = dict(k_new=_randn(gen, batch * m, hkv, d), v_new=_randn(gen, batch * m, hkv, d))
    # The slots' columns on scattered pages of a [pages, K/V, Hkv, page, D] pool.
    table = torch.randperm(slots * per_slot + 3, device="cuda", generator=gen)
    table = table[: slots * per_slot].view(slots, per_slot).int()
    pool = _randn(gen, slots * per_slot + 3, 2, hkv, page, d)
    k_pages, v_pages = pool[:, 0].transpose(1, 2), pool[:, 1].transpose(1, 2)
    cols = torch.arange(capacity, device="cuda")
    rows = table[:, cols // page].long(), cols % page
    k_pages[rows], v_pages[rows] = k_arena, v_arena
    # Request b on slot slots - 1 - b, so its table row is not its batch index.
    seqlens = torch.tensor(ctx, dtype=torch.int32, device="cuda") + m
    call = (seqlens, (slots - 1 - torch.arange(batch, device="cuda")).int(), WINDOW, None, True)
    dense = op(q, k_arena, v_arena, *call, **new)
    with validating(op):
        paged = op(q, k_pages, v_pages, *call, page_table=table, **new)
    assert torch.equal(paged, dense)
    # The blocks went to the requests' pages as to their slots, and nothing else moved.
    assert torch.equal(k_pages[rows], k_arena) and torch.equal(v_pages[rows], v_arena)


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "sm107_windowed_commit"])
def test_the_fork_commit_empties_the_dummy_slot(windowed):
    """The dummy slot, which CUDA-graph padding rows share, is empty after the commit, the DFlash
    worker's rule: over the dense arena, and after the sm107 windowed commit's own launch."""
    worker = object.__new__(DFlashWorker)
    worker._ctx_k_buf = torch.zeros(4, 5, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    worker._ctx_v_buf = torch.zeros_like(worker._ctx_k_buf)
    worker._ctx_len = torch.tensor([5, 0, 0, 6], device="cuda")
    # A windowed arena without drafter_window_compact: each slot starts at a base.
    worker._ctx_base = torch.tensor([2, 0, 0, 3], device="cuda") if windowed else None
    worker._ctx_keep, worker._max_ctx, worker._dummy_slot = (48 if windowed else 0), 64, 3
    worker._ctx_paged, worker._ctx_block_tables = False, None
    fork = context_kv_fork.ContextKvFork(worker, None, context_kv_fork.POINT)
    main = torch.cuda.current_stream()
    launched = context_kv_fork.Launched(main, None, 2, 7, context_kv_fork.POINT, None)
    # A request on slot 0 accepts 3 rows, a padding row on the dummy slot 8.
    accepted = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    num_ctx = fork.commit(launched, torch.tensor([0, 3], device="cuda"), accepted)
    fork.join(launched)
    assert worker._ctx_len.tolist() == [8, 0, 0, 0]
    # The lengths the drafter attends: physical over the windowed arena.
    assert num_ctx.tolist() == ([6, 0] if windowed else [8, 0])


PAGE = 16
#: Six requests, so a block decode of them takes the wide op where it is installed: one past the
#: target's window, and request 33 a CUDA-graph padding dummy.
REQUEST_IDS = range(31, 37)
CAPACITIES, HISTORIES = (200, 40, 16, 64, 120, 32), (180, 24, 0, 40, 100, 9)


def _unified_manager(num_draft: int, nkv: int, head_dim: int) -> KVCacheManagerV2:
    """The requests in one cache of six FP8 target layers and BF16 draft ones, all windowed."""
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=nkv,
        head_dim=head_dim,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=4 * PAGE,
    )
    specs = [replace(target, global_layer_id=i) for i in range(6)]
    draft = replace(target, owner=KVCacheLayerOwner.DRAFT, dtype=DataType.BF16)
    specs += [replace(draft, global_layer_id=6 + i) for i in range(num_draft)]
    manager = KVCacheManagerV2(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, host_cache_size=0, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=6,
        num_kv_heads=nkv,
        head_dim=head_dim,
        tokens_per_block=PAGE,
        max_seq_len=256,
        max_batch_size=len(REQUEST_IDS),
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_specs=specs,
    )
    for request_id, capacity, history in zip(REQUEST_IDS, CAPACITIES, HISTORIES):
        cache = manager._create_kv_cache(request_id, None, None, is_dummy=request_id == 33)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(capacity, history)
    return manager


@pytest.mark.parametrize("wide", [False, True], ids=["single", "wide"])
def test_the_block_decode_over_a_unified_kv_cache_is_the_dense_arena_one(wide):
    """Over each draft layer's view of the manager's pool and the worker's one page table, the
    block decode is bitwise the VANILLA drafter's over dense arenas of the same context."""
    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    # The speculative tests' tiny DFlash 2 drafter, over pages and over dense arenas.
    path = Path(__file__).parents[4] / "speculative" / "hw_agnostic" / "test_dflash2_semantics.py"
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", path)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    drafters = []
    for backend in ("TRTLLM", "VANILLA"):
        config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
        drafter = DFlashForCausalLM(config, dflash_attention_backend=backend).to("cuda")
        drafter.load_weights(dict(sem._tiny_weights(True)))
        drafter._build_fused_kv_buffers()
        block_decode.install(drafter, True, True)
        if wide:
            wide_attention.install(drafter)
        drafters.append(drafter)
    paged, dense = drafters
    layers, nkv, head_dim = paged._num_attn_layers, paged._num_kv_heads, paged._head_dim
    manager = _unified_manager(layers, nkv, head_dim)
    try:
        num_seqs = len(REQUEST_IDS)
        shape = (manager.num_attention_op_pools, num_seqs, 2, manager.max_blocks_per_seq)
        offsets = torch.full(shape, -7, dtype=torch.int32, device="cuda")
        manager.copy_batch_block_offsets(offsets, list(REQUEST_IDS), 1, 0, num_seqs)
        worker = object.__new__(DFlashWorker)
        worker._ctx_unified_cache = True
        pool = worker._managed_ctx_pool(manager, layers, nkv, head_dim, torch.bfloat16, 2)
        assert worker._init_ctx_block_tables(manager, pool, num_seqs, layers, nkv, head_dim, PAGE)
        metadata = types.SimpleNamespace(kv_cache_block_offsets=offsets)
        assert worker._refresh_ctx_block_tables(metadata, num_seqs)
        table = worker._ctx_block_tables[:num_seqs]
        # Each layer's view spans the whole interleaved pool: fill every one before reading any.
        gen = torch.Generator(device="cuda").manual_seed(0)
        for view in pool:
            view.normal_(generator=gen)
        ctx = torch.tensor(HISTORIES, device="cuda")
        cols = torch.arange(int(ctx.max()) + sem.BLOCK, device="cuda")
        # Each layer's K and V planes as [pages, page, Hkv, D] rows; the arenas [B, L, C, Hkv, D].
        planes = [[view[:, half].transpose(1, 2) for view in pool] for half in (0, 1)]
        at = table.long()[:, cols // PAGE], cols % PAGE
        arenas = [torch.stack([plane[at] for plane in half], dim=1) for half in planes]
        block = ctx[:, None] + torch.arange(sem.BLOCK, device="cuda")
        step = dict(
            noise_embedding=_randn(gen, num_seqs, sem.BLOCK, sem.TINY["hidden_size"]),
            query_positions=block,
            num_ctx_per_req=ctx,
            ctx_cache_batch_idx=torch.arange(num_seqs, device="cuda"),
        )
        got = paged.dflash_forward(
            **step, ctx_k_cache=None, ctx_v_cache=None, ctx_kv_cache=pool, ctx_page_table=table
        )
        want = dense.dflash_forward(**step, ctx_k_cache=arenas[0], ctx_v_cache=arenas[1])
        assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
        # The block's K/V are in the requests' pages as in their arena slots.
        landed = table.long().gather(1, block // PAGE), block % PAGE
        rows = torch.arange(num_seqs, device="cuda")[:, None]
        for half, arena in zip(planes, arenas):
            for layer, plane in enumerate(half):
                assert torch.equal(plane[landed], arena[rows, layer, block])
    finally:
        manager.shutdown()


def test_a_paged_context_plans_every_fast_path_but_the_window_compaction(monkeypatch):
    """sm103 TEP4 with a TRTLLM drafter, TRTLLM_DFLASH_WINDOWED_CTX_ARENA set: a paged context is
    never windowed, so the window compaction is the one served path the plan leaves out."""
    monkeypatch.setattr(dispatch, "missing_preconditions", lambda *args: [])
    monkeypatch.setattr(dispatch, "agreed_union", lambda mapping, local: list(dict.fromkeys(local)))
    model_config = types.SimpleNamespace(
        mapping=types.SimpleNamespace(tp_size=4, gpus_per_node=4, tp_group=list(range(4))),
        spec_config=DFlashDecodingConfig(max_draft_len=7, speculative_model="/drafter"),
        quant_config=types.SimpleNamespace(quant_algo=QuantAlgo.NVFP4),
    )
    worker = types.SimpleNamespace(_dflash_attention_backend="TRTLLM", windowed_ctx_arena=True)
    plan = dispatch.compose(
        None, model_config, ModelingV2Mode.REQUIRE, modeling.TARGET_PATHS, (10, 3), worker, 64
    )
    served, _ = dispatch.capacity_gated(frozenset(modeling.TARGET_PATHS[103, 4].served), 64)
    served, _ = dispatch.yielded(served)
    assert served - plan.paths == {"drafter_window_compact"}
