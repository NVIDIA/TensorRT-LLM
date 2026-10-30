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

"""FP4 MLA (NVFP4 KV cache) disaggregated KV transfer tests.

This drives the shared threaded NIXL harness (``kv_transfer_harness``) with
``Fp4MlaKVCacheManagerV2``. The transfer carries the native K, K-scale, and
BF16 HP-tail roles. The V-scale and packed-V sidecars are local-only: the
generation side must not receive them and rebuilds them from the imported K
and K scales, which must reproduce what the context side derives from the same
cache.
"""

import math
from collections.abc import Sequence
from types import SimpleNamespace

import kv_transfer_harness as transfer_harness
import pytest
import torch

from tensorrt_llm import Mapping
from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_MLA_ATTENTION_BACKEND_ENV,
    _cutedsl_backend_available,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig

NUM_LAYERS = 4
HEAD_DIM = 576
KV_LORA_RANK = 512
MAX_DRAFT_LEN = 3
MAX_TOKENS = 4096
TOKENS_PER_BLOCK = transfer_harness.TOKENS_PER_BLOCK


def _requires_fp4_mla_backend(backend: str) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FP4 MLA (NVFP4 KV cache) requires Blackwell or newer")
    if backend == "cutedsl" and (
        torch.cuda.get_device_capability() != (10, 7) or not _cutedsl_backend_available()
    ):
        pytest.skip("The CuTeDSL FP4 MLA backend requires SM107 and nvidia-cutlass-dsl")


def _create_managers(
    tp: int, pp: int, enable_dp: bool, *, speculative: bool = True
) -> list[Fp4MlaKVCacheManagerV2]:
    return [
        Fp4MlaKVCacheManagerV2(
            KvCacheConfig(
                max_tokens=MAX_TOKENS,
                dtype="nvfp4",
                enable_block_reuse=False,
                host_cache_size=0,
            ),
            CacheTypeCpp.SELFKONLY,
            num_layers=NUM_LAYERS,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=transfer_harness.MAX_SEQ_LEN,
            max_batch_size=transfer_harness.MAX_BATCH_SIZE,
            mapping=Mapping(
                world_size=tp * pp,
                rank=rank,
                tp_size=tp,
                pp_size=pp,
                enable_attention_dp=enable_dp,
            ),
            dtype=DataType.NVFP4,
            spec_config=MTPDecodingConfig(max_draft_len=MAX_DRAFT_LEN) if speculative else None,
            max_num_tokens=MAX_TOKENS,
            pretrained_config=SimpleNamespace(kv_lora_rank=KV_LORA_RANK),
            is_disagg=True,
        )
        for rank in range(tp * pp)
    ]


def _context_without_speculation() -> transfer_harness.ManagerFactory:
    """Build the context side without speculative decoding, the generation side with it.

    The harness creates the context managers first. Without draft tokens the
    context worker keeps a 16-slot HP ring, while the generation worker reserves
    ``MAX_DRAFT_LEN`` more slots for rewind.
    """
    calls = []

    def factory(tp: int, pp: int, enable_dp: bool) -> list[Fp4MlaKVCacheManagerV2]:
        calls.append((tp, pp, enable_dp))
        return _create_managers(tp, pp, enable_dp, speculative=len(calls) > 1)

    return factory


def _initialize_cache(
    managers: Sequence[Fp4MlaKVCacheManagerV2],
    _tp: int,
    seed_base: int = 0,
    fill_random: bool = True,
) -> None:
    """Fill every physical pool, including the local-only sidecars.

    Context sidecars get nonzero bytes so a transfer that leaked them into
    the generation side is visible. Bytes stay below 0x7F so no K scale is
    an FP8 NaN.
    """
    for manager in managers:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + manager.mapping.pp_rank)
        for pool in manager._iter_fp4_mla_physical_pool_views():
            pool_bytes = pool.view(torch.uint8)
            if fill_random:
                pool_bytes.copy_(
                    torch.randint(
                        1,
                        0x7F,
                        pool_bytes.shape,
                        dtype=torch.uint8,
                        device=pool_bytes.device,
                        generator=generator,
                    )
                )
            else:
                pool_bytes.zero_()
    torch.cuda.synchronize()


def _compact_layer(manager: Fp4MlaKVCacheManagerV2, layer_idx: int) -> int:
    return manager._fp4_mla_compact_layer_idx(manager.layer_offsets[layer_idx])


def _prompt_pages(
    manager: Fp4MlaKVCacheManagerV2, request_id: int, layer_idx: int, prompt_len: int
) -> list[int]:
    num_blocks = math.ceil(prompt_len / TOKENS_PER_BLOCK)
    pages = manager.get_batch_cache_indices([request_id], layer_idx)[0][:num_blocks]
    assert len(pages) == num_blocks and all(page >= 0 for page in pages), pages
    return pages


def _hp_pages(manager: Fp4MlaKVCacheManagerV2, request_id: int, prompt_len: int) -> list[int]:
    """HP-tail page per prompt block ordinal, negative outside the HP window."""
    num_blocks = math.ceil(prompt_len / TOKENS_PER_BLOCK)
    hp_pool_id = manager.get_fp4_mla_page_table_spec().hp_pool_id
    pages = manager._get_batch_cache_indices_by_pool_id(
        [request_id], pool_id=hp_pool_id, is_kv_aggregate=False
    )[0][:num_blocks]
    return list(pages) + [-1] * (num_blocks - len(pages))


def _v_scale_pages(
    manager: Fp4MlaKVCacheManagerV2, compact_layer: int, pages: list[int]
) -> torch.Tensor:
    return manager.get_mla_v_scale_pool().view(torch.uint8)[compact_layer][pages]


def _v_packed_pages(
    manager: Fp4MlaKVCacheManagerV2, compact_layer: int, pages: list[int]
) -> torch.Tensor | None:
    pool = manager.get_mla_v_packed_pool(compact_layer)
    if pool is None:
        return None
    return pool.view(-1, manager.mla_v_head_dim, TOKENS_PER_BLOCK // 2)[pages]


def _handles_request(manager, tp: int, enable_dp: bool, req_idx: int) -> bool:
    return not enable_dp or req_idx % tp == manager.mapping.tp_rank


def _find_ctx_manager(
    ctx_managers: Sequence[Fp4MlaKVCacheManagerV2],
    *,
    ctx_tp: int,
    ctx_enable_dp: bool,
    req_idx: int,
    layer_idx: int,
) -> Fp4MlaKVCacheManagerV2:
    for manager in ctx_managers:
        if layer_idx in manager.pp_layers and _handles_request(
            manager, ctx_tp, ctx_enable_dp, req_idx
        ):
            return manager
    raise AssertionError(f"missing context source for request={req_idx}, layer={layer_idx}")


def _verify_cache(
    *,
    request_lengths: list[int],
    ctx_managers: Sequence[Fp4MlaKVCacheManagerV2],
    gen_managers: Sequence[Fp4MlaKVCacheManagerV2],
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    ctx_request_ids: list[int],
    gen_request_ids: list[int],
) -> None:
    del ctx_pp, gen_pp
    torch.cuda.synchronize()
    # A generation worker with MTP also caches the draft layers, which a
    # context worker without speculation neither holds nor sends.
    ctx_layers = {layer_idx for manager in ctx_managers for layer_idx in manager.pp_layers}

    # Transferred roles arrive byte-exact; derived sidecars stay local.
    for req_idx, prompt_len in enumerate(request_lengths):
        for gen_manager in gen_managers:
            if not _handles_request(gen_manager, gen_tp, gen_enable_dp, req_idx):
                continue
            gen_hp_pages = _hp_pages(gen_manager, gen_request_ids[req_idx], prompt_len)
            for layer_idx in gen_manager.pp_layers:
                if layer_idx not in ctx_layers:
                    continue
                ctx_manager = _find_ctx_manager(
                    ctx_managers,
                    ctx_tp=ctx_tp,
                    ctx_enable_dp=ctx_enable_dp,
                    req_idx=req_idx,
                    layer_idx=layer_idx,
                )
                gen_pages = _prompt_pages(
                    gen_manager, gen_request_ids[req_idx], layer_idx, prompt_len
                )
                ctx_pages = _prompt_pages(
                    ctx_manager, ctx_request_ids[req_idx], layer_idx, prompt_len
                )
                gen_kv, gen_sf = gen_manager.get_fp4_mla_cache_buffers(layer_idx)
                ctx_kv, ctx_sf = ctx_manager.get_fp4_mla_cache_buffers(layer_idx)
                assert torch.equal(gen_kv[gen_pages], ctx_kv[ctx_pages])
                assert torch.equal(gen_sf[gen_pages], ctx_sf[ctx_pages])

                gen_layer = _compact_layer(gen_manager, layer_idx)
                ctx_layer = _compact_layer(ctx_manager, layer_idx)
                assert not _v_scale_pages(gen_manager, gen_layer, gen_pages).any()
                gen_v_packed = _v_packed_pages(gen_manager, gen_layer, gen_pages)
                if gen_v_packed is not None:
                    assert not gen_v_packed.any()

                ctx_hp_pages = _hp_pages(ctx_manager, ctx_request_ids[req_idx], prompt_len)
                # The HP window always covers the last prompt block.
                assert gen_hp_pages[-1] >= 0 and ctx_hp_pages[-1] >= 0
                # Each ring holds global position p at slot p % ring_size of the
                # page holding p; the two sides may size their rings
                # differently, so compare the positions both retain.
                gen_slots = gen_manager.fp4_mla_hp_pool_size
                ctx_slots = ctx_manager.fp4_mla_hp_pool_size
                gen_hp = gen_manager.get_fp4_mla_hp_pool().view(torch.uint8)
                ctx_hp = ctx_manager.get_fp4_mla_hp_pool().view(torch.uint8)
                gen_ring = gen_hp[gen_hp_pages[-1], gen_layer, 0].view(gen_slots, -1)
                ctx_ring = ctx_hp[ctx_hp_pages[-1], ctx_layer, 0].view(ctx_slots, -1)
                last_page_start = (prompt_len - 1) // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
                retained = range(
                    max(last_page_start, prompt_len - min(gen_slots, ctx_slots)), prompt_len
                )
                assert len(retained) > 0
                assert torch.equal(
                    gen_ring[[p % gen_slots for p in retained]],
                    ctx_ring[[p % ctx_slots for p in retained]],
                )

    # Rebuilding the sidecars from the imported K and K scales reproduces what
    # the context side derives from its own cache.
    for managers, tp, enable_dp, request_ids in (
        (ctx_managers, ctx_tp, ctx_enable_dp, ctx_request_ids),
        (gen_managers, gen_tp, gen_enable_dp, gen_request_ids),
    ):
        for manager in managers:
            manager.get_mla_v_scale_pool_base().zero_()
            v_packed_base = manager.get_mla_v_packed_pool_base()
            if v_packed_base is not None:
                v_packed_base.zero_()
            manager.rebuild_disagg_imported_cache(
                [
                    SimpleNamespace(py_request_id=request_ids[req_idx], prompt_len=prompt_len)
                    for req_idx, prompt_len in enumerate(request_lengths)
                    if _handles_request(manager, tp, enable_dp, req_idx)
                ]
            )
    torch.cuda.synchronize()

    for req_idx, prompt_len in enumerate(request_lengths):
        for gen_manager in gen_managers:
            if not _handles_request(gen_manager, gen_tp, gen_enable_dp, req_idx):
                continue
            for layer_idx in gen_manager.pp_layers:
                if layer_idx not in ctx_layers:
                    continue
                ctx_manager = _find_ctx_manager(
                    ctx_managers,
                    ctx_tp=ctx_tp,
                    ctx_enable_dp=ctx_enable_dp,
                    req_idx=req_idx,
                    layer_idx=layer_idx,
                )
                gen_pages = _prompt_pages(
                    gen_manager, gen_request_ids[req_idx], layer_idx, prompt_len
                )
                ctx_pages = _prompt_pages(
                    ctx_manager, ctx_request_ids[req_idx], layer_idx, prompt_len
                )
                gen_layer = _compact_layer(gen_manager, layer_idx)
                ctx_layer = _compact_layer(ctx_manager, layer_idx)
                gen_v_scale = _v_scale_pages(gen_manager, gen_layer, gen_pages)
                assert gen_v_scale.any()
                assert torch.equal(gen_v_scale, _v_scale_pages(ctx_manager, ctx_layer, ctx_pages))
                gen_v_packed = _v_packed_pages(gen_manager, gen_layer, gen_pages)
                ctx_v_packed = _v_packed_pages(ctx_manager, ctx_layer, ctx_pages)
                assert (gen_v_packed is None) == (ctx_v_packed is None)
                if gen_v_packed is not None:
                    assert gen_v_packed.any()
                    assert torch.equal(gen_v_packed, ctx_v_packed)


# Production pairs a DEP context with a larger DEP generation instance; the
# remaining cases cover MLA TP replication and per-PP-stage layer subsets.
TOPOLOGIES = [
    (1, 1, False, 1, 1, False, "tp1_to_tp1"),
    (2, 1, True, 2, 1, True, "dep2_to_dep2"),
    (2, 1, True, 4, 1, True, "dep2_to_dep4"),
    (1, 1, False, 2, 1, False, "tp1_to_tep2"),
    (1, 2, False, 1, 1, True, "tp1pp2_to_dep1"),
    (1, 1, True, 1, 2, False, "dep1_to_tp1pp2"),
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp",
    [topology[:6] for topology in TOPOLOGIES[:3]],
    ids=[topology[6] for topology in TOPOLOGIES[:3]],
)
def test_fp4_mla_kv_transfer_into_a_larger_hp_ring(
    monkeypatch,
    ctx_tp: int,
    ctx_pp: int,
    ctx_enable_dp: bool,
    gen_tp: int,
    gen_pp: int,
    gen_enable_dp: bool,
) -> None:
    """A context worker without speculation feeds a generation worker with it.

    The rings differ in size (16 vs ``16 + MAX_DRAFT_LEN`` slots), so the HP
    tail has to move position by position instead of as one region.
    """
    _requires_fp4_mla_backend("triton")
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    transfer_harness.run_kv_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        update_before_transfer=True,
        manager_factory=_context_without_speculation(),
        init_fn=_initialize_cache,
        verify_fn=_verify_cache,
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["triton", "cutedsl"])
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp",
    [topology[:6] for topology in TOPOLOGIES],
    ids=[topology[6] for topology in TOPOLOGIES],
)
@pytest.mark.parametrize(
    "update_before_transfer",
    [True, False],
    ids=["update_before", "update_after"],
)
def test_fp4_mla_kv_transfer(
    monkeypatch,
    backend: str,
    ctx_tp: int,
    ctx_pp: int,
    ctx_enable_dp: bool,
    gen_tp: int,
    gen_pp: int,
    gen_enable_dp: bool,
    update_before_transfer: bool,
) -> None:
    _requires_fp4_mla_backend(backend)
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    transfer_harness.run_kv_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        update_before_transfer=update_before_transfer,
        manager_factory=_create_managers,
        init_fn=_initialize_cache,
        verify_fn=_verify_cache,
    )
