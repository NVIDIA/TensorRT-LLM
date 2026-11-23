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
"""Unified target + draft KV transfer from a PP4 or TP4 DFlash context instance to a TP4 one.

The context instance holds the 62 target layers as 15/16/16/15 over four pipeline stages and the
drafter's five KV layers (global ids 62-66) on the last stage only; the generation instance holds
all 67 layers on every TP rank. Target layers are FP8 with 16 KV heads of 64, draft layers BF16
with 8 KV heads of 128, and the draft layers share the target's sliding window, so they live in
the sliding layer group next to the target's sliding layers, in a pool of their own page size.

On a pipeline-parallel DFlash context server every context rank publishes the instance-wide draft-layer
count, so the generation ranks accept the context instance, and every generation rank fans in
from the four stages. After the transfer, each generation rank's pages of every layer, the draft
layers included, must equal byte for byte its head slice of the context stage that holds the
layer.

A TP4 context instance (a TEP4 context server) holds all 67 layers on every rank, with the same
per-rank head split as the generation instance, so each generation rank's pages must equal its
peer context rank's pages.

The draft layers are BF16, or FP8 as ``TRTLLM_DFLASH_FP8_DRAFT_KV=1`` stores them: then they
are the size of the target's buffers and keep a pool of their own through their coalescing
group, and their bytes move the same way. On every rank that holds draft layers, the attention
op's pool pointer, mapping offset and page-index scale of each layer must address that layer's
own pages, in the draft pool as in the target's.

This runs the threaded single-process NIXL harness (``kv_transfer_harness``): eight transceivers
on one GPU.
"""

from functools import partial

import kv_transfer_harness as transfer_harness
import pytest
import torch

from tensorrt_llm import Mapping
from tensorrt_llm._torch.disaggregation import pp_dflash_context
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TARGET_LAYERS = 62
DRAFT_LAYERS = 5
DRAFT_IDS = list(range(TARGET_LAYERS, TARGET_LAYERS + DRAFT_LAYERS))
PP_PARTITION = [15, 16, 16, 15]
TARGET_HEADS, TARGET_HEAD_DIM = 16, 64
DRAFT_HEADS, DRAFT_HEAD_DIM = 8, 128
# Below max_seq_len, so the sliding group is its own life cycle, and not below the longest
# request the harness sends plus its first generated token (383 + 1), so no page of a request
# leaves the window during the test.
WINDOW = 384


def _is_full(layer: int) -> bool:
    return layer < TARGET_LAYERS and layer % 4 == 1


def _stage_layers(stage: int, pp: int) -> tuple[list[int], list[int]]:
    """(target layers, draft layers) held by one pipeline stage."""
    if pp == 1:
        return list(range(TARGET_LAYERS)), list(DRAFT_IDS)
    assert pp == len(PP_PARTITION)
    start = sum(PP_PARTITION[:stage])
    target = list(range(start, start + PP_PARTITION[stage]))
    return target, list(DRAFT_IDS) if stage == pp - 1 else []


def _layer_specs(target, draft, tp, draft_dtype=DataType.BF16):
    specs = [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=KVCacheLayerOwner.TARGET,
            dtype=DataType.FP8,
            num_kv_heads=TARGET_HEADS // tp,
            head_dim=TARGET_HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=None if _is_full(layer) else WINDOW,
        )
        for layer in target
    ]
    specs += [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=KVCacheLayerOwner.DRAFT,
            dtype=draft_dtype,
            num_kv_heads=DRAFT_HEADS // tp,
            head_dim=DRAFT_HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=WINDOW,
        )
        for layer in draft
    ]
    return specs


def _create_managers(
    tp: int, pp: int, enable_dp: bool, draft_dtype=DataType.BF16
) -> list[KVCacheManagerV2]:
    assert not enable_dp
    managers = []
    for rank in range(tp * pp):
        target, draft = _stage_layers(rank // tp, pp)
        managers.append(
            KVCacheManagerV2(
                KvCacheConfig(
                    max_gpu_total_bytes=256 << 20,
                    host_cache_size=0,
                    enable_block_reuse=False,
                    enable_swa_scratch_reuse=False,
                    event_buffer_max_size=0,
                ),
                CacheTypeCpp.SELF,
                num_layers=TARGET_LAYERS,
                num_kv_heads=TARGET_HEADS,
                head_dim=TARGET_HEAD_DIM,
                tokens_per_block=transfer_harness.TOKENS_PER_BLOCK,
                max_seq_len=transfer_harness.MAX_SEQ_LEN,
                max_batch_size=transfer_harness.MAX_BATCH_SIZE,
                mapping=Mapping(world_size=tp * pp, rank=rank, tp_size=tp, pp_size=pp),
                dtype=DataType.FP8,
                layer_specs=_layer_specs(target, draft, tp, draft_dtype),
            )
        )
    return managers


def _page_bytes(manager: KVCacheManagerV2, layer: int) -> torch.Tensor:
    """The layer's pages as bytes, [page, K|V, head, token, byte] (head-major pages)."""
    return manager.get_buffers(layer, kv_layout="HND").view(torch.uint8)


def _first_head(manager: KVCacheManagerV2, layer: int) -> int:
    local_heads = manager.num_kv_heads_per_layer[manager.layer_offsets[layer]]
    return manager.mapping.tp_rank * local_heads


def _initialize(managers, _tp, *, seed_base: int = 0, fill_random: bool = True) -> None:
    del seed_base
    for manager in managers:
        for layer in manager.pp_layers:
            pages = _page_bytes(manager, layer)
            if not fill_random:
                pages.zero_()
                continue
            n_page, n_kv, n_head, n_token, n_byte = pages.shape
            index = {"device": pages.device, "dtype": torch.int32}
            page = torch.arange(n_page, **index)[:, None, None, None, None]
            kv = torch.arange(n_kv, **index)[None, :, None, None, None]
            head = (_first_head(manager, layer) + torch.arange(n_head, **index))[
                None, None, :, None, None
            ]
            token = torch.arange(n_token, **index)[None, None, None, :, None]
            byte = torch.arange(n_byte, **index)[None, None, None, None, :]
            values = (layer * 29 + page * 11 + kv * 13 + head * 19 + token * 3 + byte) % 251 + 1
            pages.copy_(values.to(torch.uint8))


def _check_attention_op_tables(manager: KVCacheManagerV2) -> None:
    """Each layer's attention-op pointer, offset and table scale address its own pages.

    The attention op reads page ``i`` of a layer at ``pointer + (table[i] + offset * 2) *
    stride`` with ``table[i] = slot * scale`` (plus the K/V distance for V); the layer's K or V
    page of a slot is at its own base plus ``slot * scale * stride``.
    """
    groups = manager._attention_op_pool_groups
    assert groups is not None and not manager._use_per_layer_page_tables
    pointers = manager.kv_cache_pool_pointers[:, 0].tolist()
    for local, (op_pool, offset) in enumerate(manager.kv_cache_pool_mapping.tolist()):
        stride = manager.impl.get_page_stride(local, Role.KEY)
        converter = manager.impl.get_page_index_converter(local, Role.KEY)
        group = groups[op_pool]
        assert group.pool_id == manager.impl.get_layer_group_id(local), local
        assert group.scale == converter.scale, local
        for role, lane in ((Role.KEY, 0), (Role.VALUE, group.kv_offset)):
            base = manager.impl.get_mem_pool_base_address(local, role, PageIndexMode.SHARED)
            assert pointers[op_pool] + (lane + offset * 2) * stride == base, (local, role)


def _valid_pages(manager: KVCacheManagerV2, request_id: int, layer: int) -> list[int]:
    return [i for i in manager.get_batch_cache_indices([request_id], layer)[0] if i >= 0]


def _verify(
    *,
    request_lengths,
    ctx_managers,
    gen_managers,
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    ctx_request_ids,
    gen_request_ids,
) -> None:
    assert (ctx_tp, ctx_pp) in ((1, 4), (4, 1)) and (gen_tp, gen_pp) == (4, 1)
    assert not ctx_enable_dp and not gen_enable_dp
    held = sorted({layer for m in ctx_managers for layer in m.pp_layers})
    assert held == list(range(TARGET_LAYERS + DRAFT_LAYERS))
    if ctx_pp == 4:
        assert [len(m.pp_layers) for m in ctx_managers] == [15, 16, 16, 20]
        assert set(DRAFT_IDS) <= set(ctx_managers[-1].pp_layers)
    else:
        assert all(len(m.pp_layers) == TARGET_LAYERS + DRAFT_LAYERS for m in ctx_managers)
    for manager in list(ctx_managers) + list(gen_managers):
        if set(DRAFT_IDS) <= set(manager.pp_layers):
            _check_attention_op_tables(manager)

    def owner(layer: int, gen: KVCacheManagerV2) -> KVCacheManagerV2:
        """The context rank that holds ``layer`` and the generation rank's first head."""
        first = _first_head(gen, layer)
        (ctx,) = [
            m
            for m in ctx_managers
            if layer in m.pp_layers
            and _first_head(m, layer)
            <= first
            < _first_head(m, layer) + m.num_kv_heads_per_layer[m.layer_offsets[layer]]
        ]
        return ctx

    for req_idx, length in enumerate(request_lengths):
        num_pages = -(-length // transfer_harness.TOKENS_PER_BLOCK)
        for gen in gen_managers:
            assert len(gen.pp_layers) == TARGET_LAYERS + DRAFT_LAYERS
            for layer in gen.pp_layers:
                ctx = owner(layer, gen)
                # The prompt's pages; a side may already hold one more for the first new token.
                gen_pages = _valid_pages(gen, gen_request_ids[req_idx], layer)[:num_pages]
                ctx_pages = _valid_pages(ctx, ctx_request_ids[req_idx], layer)[:num_pages]
                assert len(gen_pages) == len(ctx_pages) == num_pages, (layer, length)
                actual = _page_bytes(gen, layer)[gen_pages]
                first = _first_head(gen, layer) - _first_head(ctx, layer)
                expected = _page_bytes(ctx, layer)[ctx_pages][:, :, first : first + actual.shape[2]]
                assert torch.equal(actual, expected), (
                    f"request {req_idx} layer {layer} gen tp_rank {gen.mapping.tp_rank}"
                )
                if layer in DRAFT_IDS:
                    # The draft pages arrived: the generation side started from zeros.
                    assert bool(actual.ne(0).all())


def _request_draft_dtype(monkeypatch, draft_dtype) -> None:
    """FP8 draft layers get a pool of their own only with the FP8 draft KV knob set."""
    if draft_dtype == DataType.FP8:
        monkeypatch.setenv("TRTLLM_DFLASH_FP8_DRAFT_KV", "1")
    else:
        monkeypatch.delenv("TRTLLM_DFLASH_FP8_DRAFT_KV", raising=False)


@pytest.mark.cuda
@pytest.mark.timeout(300)
@pytest.mark.parametrize("draft_dtype", [DataType.BF16, DataType.FP8], ids=["bf16", "fp8"])
@pytest.mark.parametrize("update_before_transfer", [True, False], ids=["update_before", "after"])
def test_pp4_context_to_tp4_generation_draft_kv_transfer(
    monkeypatch, update_before_transfer, draft_dtype
):
    # What the model construction of the context server records.
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", True)
    _request_draft_dtype(monkeypatch, draft_dtype)
    transfer_harness.run_kv_transfer_test(
        ctx_tp=1,
        ctx_pp=4,
        gen_tp=4,
        gen_pp=1,
        ctx_enable_dp=False,
        gen_enable_dp=False,
        update_before_transfer=update_before_transfer,
        manager_factory=partial(_create_managers, draft_dtype=draft_dtype),
        init_fn=_initialize,
        verify_fn=_verify,
    )


@pytest.mark.cuda
@pytest.mark.timeout(300)
@pytest.mark.parametrize("draft_dtype", [DataType.BF16, DataType.FP8], ids=["bf16", "fp8"])
@pytest.mark.parametrize("update_before_transfer", [True, False], ids=["update_before", "after"])
def test_tp4_context_to_tp4_generation_draft_kv_transfer(
    monkeypatch, update_before_transfer, draft_dtype
):
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", False)
    _request_draft_dtype(monkeypatch, draft_dtype)
    transfer_harness.run_kv_transfer_test(
        ctx_tp=4,
        ctx_pp=1,
        gen_tp=4,
        gen_pp=1,
        ctx_enable_dp=False,
        gen_enable_dp=False,
        update_before_transfer=update_before_transfer,
        manager_factory=partial(_create_managers, draft_dtype=draft_dtype),
        init_fn=_initialize,
        verify_fn=_verify,
    )
