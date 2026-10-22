# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target on a unified target/draft KV cache (KVCacheManagerV2 layer specs).

* The verify fold's and the context-K/V fork's written-out worker steps leave the DFlash
  dummy slot empty, as the built-in ``_advance_ctx_len`` does.
* The step-input stager writes per-layer and shared attention-op page tables exactly as the
  manager's own copy does, and the life cycles' SHARED-path tables over their readable
  columns, each from a snapshot of the host rows taken when it records.
"""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.verify_fold import (  # noqa: E501
    fold as verify_fold,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.runtime import (  # noqa: E501
    step_commit,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

PAGE = 16
WINDOW = 4 * PAGE
MAX_SEQ_LEN = 256
NUM_TARGET, NUM_DRAFT, NKV, HQ, HEAD_DIM = 6, 2, 2, 4, 64
BLOCK = 8
SENTINEL = -7
# (request id, capacity, history, dummy): past the window, inside it, a CUDA-graph padding dummy.
REQUESTS = ((31, 200, 180, False), (32, 40, 24, False), (33, 16, 0, True))
UNSCHEDULED = (34, 96, 80, False)

_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# --------------------------------------------------------------------------
# The written-out worker steps keep the dummy slot empty
# --------------------------------------------------------------------------
class _Drafter:
    """``project_target_hidden`` / ``precompute_context_kv`` of a drafter with 2 layers of 1 x 4 heads."""

    def project_target_hidden(self, hidden):
        return hidden

    def precompute_context_kv(self, projected, positions):
        k = (positions + 1).view(-1, 1, 1, 1).expand(-1, 2, 1, 4).to(projected.dtype).clone()
        return k, k + 0.5


def test_the_fold_store_empties_the_dummy_slot():
    worker = object.__new__(verify_fold.SpecLoopFoldDFlashWorker)
    # Slot 3 is the dummy slot, with a length a stale step left there.
    worker._ctx_len = torch.tensor([5, 0, 0, 6])
    worker._dummy_slot, worker._max_ctx, worker._ctx_base, worker._ctx_keep = 3, 32, None, 0
    worker._ctx_block_tables, worker._ctx_paged = None, False
    worker._ctx_k_buf = torch.zeros(4, 2, 40, 1, 4, dtype=torch.bfloat16)
    worker._ctx_v_buf = torch.zeros_like(worker._ctx_k_buf)
    slots, accepted = torch.tensor([0, 3]), torch.tensor([3, 8], dtype=torch.int32)
    offsets = torch.arange(8)
    worker._fold_store_accepted_context(
        _Drafter(),
        torch.randn(16, 4, dtype=torch.bfloat16),
        SimpleNamespace(num_ctx_tokens=0),
        2,
        8,
        7,
        slots,
        torch.arange(2),
        offsets,
        worker._ctx_len[slots].unsqueeze(1) + offsets,
        accepted,
        accepted.long(),
    )
    assert worker._ctx_len.tolist() == [8, 0, 0, 0]


@_needs_cuda
@pytest.mark.parametrize("windowed", [False, True])
def test_the_fork_commit_empties_the_dummy_slot(windowed):
    k_arena = torch.randn(4, 5, 64, 1, 128, device="cuda").bfloat16()
    worker = SimpleNamespace(
        _ctx_k_buf=k_arena,
        _ctx_v_buf=torch.randn_like(k_arena),
        _ctx_len=torch.tensor([5, 0, 0, 6], device="cuda"),
        # A windowed arena's slots start at a base; the dummy slot's is stale too.
        _ctx_base=torch.tensor([2, 0, 0, 3], device="cuda") if windowed else None,
        _ctx_keep=48 if windowed else 0,
        _max_ctx=64,
        _dummy_slot=3,
    )
    fork = context_kv_fork.ContextKvFork(worker, None, context_kv_fork.POINT)
    launched = context_kv_fork.Launched(
        torch.cuda.current_stream(), None, 2, 7, context_kv_fork.POINT, None
    )
    accepted = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    num_ctx = fork.commit(launched, torch.tensor([0, 3], device="cuda"), accepted)
    fork.join(launched)
    torch.cuda.synchronize()
    assert worker._ctx_len.tolist() == [8, 0, 0, 0]
    assert num_ctx.tolist() == [8, 0]
    if windowed:
        assert worker._ctx_base.tolist() == [2, 0, 0, 0]


# --------------------------------------------------------------------------
# A unified target/draft cache: FP8 target layers, BF16 draft layers (twice the page bytes)
# --------------------------------------------------------------------------
LAYOUTS = ("grouped", "per-layer", "shared")


def _make_manager(monkeypatch, layout: str) -> KVCacheManagerV2:
    """``grouped`` / ``per-layer``: the unified cache's attention-op tables; ``shared``: no draft layers."""
    torch.cuda.init()
    if layout == "per-layer":
        monkeypatch.setattr(KVCacheManagerV2, "_get_attention_op_pool_groups", lambda self: None)
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=WINDOW,
    )
    specs = [
        replace(target, global_layer_id=i, sliding_window_size=None if i % 3 == 2 else WINDOW)
        for i in range(NUM_TARGET)
    ]
    draft = replace(target, owner=KVCacheLayerOwner.DRAFT, dtype=DataType.BF16)
    if layout != "shared":
        specs += [replace(draft, global_layer_id=NUM_TARGET + i) for i in range(NUM_DRAFT)]
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_TARGET,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_specs=specs,
    )
    assert manager._use_per_layer_page_tables == (layout == "per-layer")
    assert (manager._attention_op_pool_groups is not None) == (layout == "grouped")
    for request_id, capacity, history, dummy in REQUESTS + (UNSCHEDULED,):
        cache = manager._create_kv_cache(request_id, None, None, is_dummy=dummy)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(capacity, history)
    return manager


def _request_ids():
    return [request_id for request_id, *_ in REQUESTS]


def _destination(manager):
    shape = (manager.num_attention_op_pools, len(REQUESTS) + 2, 2, manager.max_blocks_per_seq)
    return torch.full(shape, SENTINEL, dtype=torch.int32, device="cuda")


def _native_tables(manager):
    dst = _destination(manager)
    manager.copy_batch_block_offsets(dst, _request_ids(), 1, 0, len(REQUESTS))
    torch.cuda.synchronize()
    return dst


def _stage_tables(manager, dst):
    stager = runtime._PrepareStager(
        step_commit.StepStaging(), Counter(), SimpleNamespace(), SimpleNamespace(), None
    )
    assert stager.block_offsets(manager, dst, _request_ids(), 1, 0, len(REQUESTS)), stager.declined
    stager.end({})


@_needs_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_stager_writes_the_managers_tables(monkeypatch, layout: str):
    manager = _make_manager(monkeypatch, layout)
    try:
        expected = _native_tables(manager)
        window_pool = manager.impl.get_layer_group_id(0)
        row = manager.index_mapper.get_copy_index(_request_ids(), 0, 1, False).tolist()[0]
        live = manager.host_kv_cache_block_offsets[window_pool, row, 0].tolist()
        assert BAD_PAGE_INDEX in live[: max(i for i, p in enumerate(live) if p >= 0)]
        staged = _destination(manager)
        _stage_tables(manager, staged)
        torch.cuda.synchronize()
        if layout == "shared":
            # The SHARED path covers the columns below ceil(capacity / page) + 1 only.
            cols = max(-(-capacity // PAGE) + 1 for _, capacity, *_ in REQUESTS)
            assert torch.equal(staged[..., :cols], expected[..., :cols])
            assert (staged[..., cols:] == SENTINEL).all()
        else:
            assert torch.equal(staged, expected)
    finally:
        manager.shutdown()


@_needs_cuda
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_staged_tables_are_a_snapshot_of_the_host_table(monkeypatch, layout: str):
    """The next step's host-table writes must not reach this step's staged tables."""
    manager = _make_manager(monkeypatch, layout)
    try:
        expected = _destination(manager)
        _stage_tables(manager, expected)
        torch.cuda.synchronize()
        host = manager.host_kv_cache_block_offsets
        saved = host.clone()
        staged = _destination(manager)
        # Hold the stream so the staged launch is still pending when the host table changes.
        torch.cuda._sleep(1 << 30)
        _stage_tables(manager, staged)
        host.fill_(7)
        torch.cuda.synchronize()
        host.copy_(saved)
        assert torch.equal(staged, expected)
    finally:
        manager.shutdown()
