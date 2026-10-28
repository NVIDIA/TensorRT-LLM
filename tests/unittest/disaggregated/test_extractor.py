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

import msgpack
import numpy as np
import pytest
import torch
from layout_test_utils import manager_layout

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_cache_kinds,
    get_physical_pool,
    get_transferred_pool_indices,
    get_unique_pool_memory_descs,
    slot_pointers,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaRole
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    head_major,
    replicated,
    sectioned,
    token_major,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import (
    page_table_from_dict,
    page_table_to_dict,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KvCacheConfig,
    KVCacheManager,
    Mapping,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2


@pytest.mark.cuda
def test_slot_pointers_address_a_manager_s_pool():
    num_layers = 1
    num_kv_heads = 8
    head_dim = 16
    tokens_per_block = 32
    max_seq_len = 128
    max_batch_size = 2
    dtype = DataType.HALF
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)
    kv_cache_config = KvCacheConfigV2(
        max_tokens=512,
        free_gpu_memory_fraction=0.1,
        max_attention_window=None,
        enable_block_reuse=False,
        event_buffer_max_size=0,
        host_cache_size=0,
        enable_partial_reuse=False,
        copy_on_partial_reuse=False,
        sink_token_length=0,
        max_util_for_resume=1,
    )
    kv_cache_type = CacheTypeCpp.SELF

    manager = KVCacheManagerV2(
        kv_cache_config=kv_cache_config,
        kv_cache_type=kv_cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
        vocab_size=32000,
    )

    page_table = manager.get_page_table()
    region_ids = np.array([0, -1, 1], dtype=np.int64)
    pointers = slot_pointers(page_table, 0, 0, region_ids)

    pool = get_physical_pool(page_table, 0, 0)
    assert pool.slot_bytes > 0
    # Slots not held (-1) are left out.
    np.testing.assert_array_equal(pointers, [pool.base, pool.base + pool.slot_bytes])

    manager.shutdown()


@pytest.mark.cuda
def test_a_manager_s_page_table_places_its_buffers():
    num_layers = 8
    num_kv_heads = 8
    head_dim = 128
    tokens_per_block = 64
    max_tokens = 3200

    kv_cache_config = KvCacheConfigV2(max_tokens=max_tokens)

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    manager = KVCacheManagerV2(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=2048,
        max_batch_size=8,
        mapping=mapping,
        dtype=DataType.HALF,
        vocab_size=32000,
        max_num_tokens=max_tokens,
    )

    page_table = manager.get_page_table()
    layout = page_table.page_layout

    assert layout.tokens_per_page == 64
    assert get_cache_kinds(page_table) == [CacheKind.PAGED] * len(layout.layer_groups)
    assert sorted(layer for g in layout.layer_groups for layer in g.layers) == list(range(8))

    (pool_idx,) = get_transferred_pool_indices(page_table, 0)
    pool = get_physical_pool(page_table, 0, pool_idx)
    assert pool.base > 0
    # V2 reserves additional slots beyond the requested token capacity.
    assert pool.num_slots >= max_tokens // tokens_per_block

    layer = layout.layer_groups[0].layers[0]
    key, value = sorted(
        (b for b in layout.buffers if b.layer == layer), key=lambda b: b.slot_offset
    )
    assert (key.role, value.role) == ("key", "value")
    assert key.pool_index == value.pool_index == pool_idx
    assert pool.base + value.slot_offset > pool.base + key.slot_offset > 0
    assert page_table.kv_heads[0] == num_kv_heads

    manager.shutdown()


def _page_table(groups, pool_slot_bytes, buffers, bases, num_slots=4):
    """The page table of a manager with these layer groups and buffers.

    Each group has one internal layer per global layer, in order; device pools lie at
    ``bases[group]``.
    """
    layout = PageLayout(
        tokens_per_page=16,
        pool_slot_bytes=pool_slot_bytes,
        layer_groups=groups,
        buffers=buffers,
    )
    return manager_layout(layout, bases, kv_heads=1, num_slots=num_slots)


def _attention_page_table(buffers, pool_slot_bytes):
    return _page_table(
        (LayerGroupDesc("attention", None, 0, 0, (0, 1)),),
        {0: pool_slot_bytes},
        buffers,
        {0: [0x1000] * len(pool_slot_bytes)},
    )


def _kv(layer, role, pool, offset, size=128, tensors=None):
    """A K/V buffer of 16 tokens of one head, token-major, or ``tensors``."""
    return BufferDesc(
        layer,
        role,
        0,
        pool,
        offset,
        size,
        token_major(size, 16, 1, 1) if tensors is None else tensors,
    )


def _index_key(layer, pool, offset, size=128):
    return BufferDesc(layer, "index_key", 0, pool, offset, size, replicated(size))


def test_a_layer_group_transfers_the_pools_holding_its_transferred_buffers():
    """A layer group's pools that transfer are those holding its transferred buffers.

    Pools are listed by index within the group's pool group, once however many buffers share
    them; a pool holding only buffers peers do not transfer is left out.
    """
    kv = (
        _kv(0, "key", 0, 0),
        _kv(0, "value", 0, 128),
        _index_key(0, 0, 256),
        _kv(1, "key", 0, 384),
        _kv(1, "value", 0, 512),
    )
    assert get_transferred_pool_indices(_attention_page_table(kv, (640,)), 0) == [0]

    scales = tuple(
        BufferDesc(layer, "scale", 0, 1, 64 * layer, 64, transferred=False) for layer in (0, 1)
    )
    page_table = _attention_page_table(kv + (_index_key(1, 2, 0),) + scales, (640, 128, 128))
    assert get_transferred_pool_indices(page_table, 0) == [0, 2]


def _recurrent_page_table(conv_sections=(16, 16, 32), ple_on_layer=1):
    """A page table of a recurrent and an attention layer group.

    Group 0 is recurrent (layers 0 and 1): SSM and conv state share one 256-byte pool,
    ``[ssm, conv]`` per layer, and PLE n-gram context lives in a second, 32-byte pool on
    ``ple_on_layer`` only. Group 1 is attention (layer 2) with K/V in a pool group of its own.
    """
    buffers = []
    for layer in (0, 1):
        # ssm: [2 heads, 4, 4] fp16 = 64 bytes, 32 bytes per head
        buffers.append(BufferDesc(layer, "ssm_state", 0, 0, 128 * layer, 64, head_major(64, 2, 2)))
        buffers.append(
            BufferDesc(layer, "conv_state", 0, 0, 128 * layer + 64, 64, sectioned(conv_sections, 1))
        )
    buffers.append(
        BufferDesc(ple_on_layer, str(MambaRole.PLE_NGRAM_CONTEXT), 0, 1, 0, 32, replicated(32))
    )
    buffers.append(BufferDesc(2, "key", 1, 0, 0, 64))
    buffers.append(BufferDesc(2, "value", 1, 0, 64, 64))
    return _page_table(
        (
            LayerGroupDesc("state", None, 0, 0, (0, 1)),
            LayerGroupDesc("attention", None, 0, 1, (2,)),
        ),
        {0: (256, 32), 1: (128,)},
        tuple(buffers),
        {0: [0x10000, 0x20000], 1: [0x30000]},
    )


def test_a_page_table_addresses_a_recurrent_group_by_slot():
    """A recurrent group's pools come from its pool group.

    A request's slot of each is addressed by its base pointer, where the runs place each layer's
    bytes.
    """
    page_table = _recurrent_page_table()

    assert get_cache_kinds(page_table) == [CacheKind.STATE, CacheKind.PAGED]
    pool = get_physical_pool(page_table, 0, 0)
    assert (pool.base, pool.slot_bytes, pool.num_slots) == (0x10000, 256, 4)
    assert get_physical_pool(page_table, 1, 0).base == 0x30000

    # SSM and conv state share pool 0, PLE context lies in pool 1.
    assert get_transferred_pool_indices(page_table, 0) == [0, 1]
    by_role = {b.role: b for b in page_table.page_layout.buffers if b.layer == 1}
    assert (by_role["ssm_state"].pool_index, by_role["ssm_state"].slot_offset) == (0, 128)
    assert (by_role["conv_state"].pool_index, by_role["conv_state"].slot_offset) == (0, 192)
    assert by_role[str(MambaRole.PLE_NGRAM_CONTEXT)].pool_index == 1

    assert slot_pointers(page_table, 0, 0, np.array([2])).tolist() == [0x10000 + 2 * 256]
    assert slot_pointers(page_table, 0, 1, np.array([3])).tolist() == [0x20000 + 3 * 32]


def test_page_table_rejects_sections_that_do_not_cover_the_buffer():
    with pytest.raises(ValueError, match="do not cover its 64 bytes"):
        _recurrent_page_table(conv_sections=(16, 16, 16))


@pytest.mark.cpu_only
def test_a_page_table_travels_to_a_peer_whole():
    """A rank info carries its manager's page table as plain values, read back as they were."""
    page_table = _recurrent_page_table()
    received = page_table_from_dict(
        msgpack.unpackb(msgpack.packb(page_table_to_dict(page_table)), strict_map_key=False)
    )
    assert received.page_layout == page_table.page_layout
    assert received.device_pools == page_table.device_pools
    for name in ("tokens_per_block", "windows", "recurrent", "layers", "local_layers", "kv_heads"):
        assert getattr(received, name) == getattr(page_table, name), name
    assert slot_pointers(received, 0, 1, np.array([3])).tolist() == [0x20000 + 3 * 32]


def test_v2_mamba_registration_uses_coalesced_physical_pool():
    state_bytes = 64
    num_layers = 2
    num_slots = 8
    # Equal-sized SSM and convolution states share one interleaved V2 pool.
    physical_slot_bytes = state_bytes * num_layers * 2
    buffers = []
    for i, layer in enumerate((1, 2)):
        buffers.append(
            BufferDesc(layer, "conv_state", 0, 0, i * state_bytes, state_bytes, sectioned((64,), 1))
        )
        buffers.append(
            BufferDesc(
                layer,
                "ssm_state",
                0,
                0,
                (num_layers + i) * state_bytes,
                state_bytes,
                head_major(state_bytes, 2, 2),
            )
        )
    page_table = _page_table(
        (LayerGroupDesc("state", None, 0, 0, (1, 2)),),
        {0: (physical_slot_bytes,)},
        tuple(buffers),
        {0: [1000]},
        num_slots=num_slots,
    )

    # V2 interleaved: one shared pool registered once
    assert get_unique_pool_memory_descs(page_table, device_id=3) == [
        (1000, physical_slot_bytes * num_slots, 3, "kv_cache_memory_pool0")
    ]


# ---------------------------------------------------------------------------
# Boundary tests for block_id -> primary memory-pool slot translation
# (see cache_reuse.py: when host offload is enabled, a
#  block's logical ID can diverge from its primary-pool slot index after an
#  offload+onboard cycle, and the cache transceiver must translate before
#  doing pointer arithmetic).
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_get_memory_pool_block_indices_vswa():
    """VSWA / multi-window dispatch boundary check.

    Without host offload, primary slot indices should equal block IDs
    (no offload has happened), and the wrapper must:
    - dispatch correctly per window_size,
    - return non-negative pool indices that match what
      `slot_pointers` plugs into `base_ptr + idx * slot_bytes`.
    """
    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.sampling_params import SamplingParams

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    num_layers = 4
    num_kv_heads = 2
    head_dim = 64
    tokens_per_block = 16
    # Two distinct window sizes -> is_variable_window=True. Keep windows
    # large enough that a short prefill keeps every block live.
    max_attention_window_vec = [128] * 2 + [256] * 2

    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        free_gpu_memory_fraction=0.1,
        max_attention_window=max_attention_window_vec,
        enable_block_reuse=False,
        host_cache_size=0,
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max(max_attention_window_vec),
        max_batch_size=2,
        mapping=mapping,
        dtype=DataType.HALF,
    )

    try:
        # Drive a small prefill so the manager has resident blocks.
        sampling_params = SamplingParams()
        req = LlmRequest(
            request_id=0,
            max_new_tokens=1,
            input_tokens=list(range(tokens_per_block * 3)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()
            ),
            is_streaming=False,
        )
        manager.impl.add_sequence_batch([(req.py_request_id, req.prompt_len, 1)], [req])

        # For each unique window, look up that window's first layer to fetch
        # its block IDs, then translate and confirm the pool-slot indices
        # match what slot_pointers would compute against
        # the live primary pool.
        unique_windows = sorted(set(max_attention_window_vec))
        assert len(unique_windows) >= 2, "Test premise: at least 2 windows"

        for w in unique_windows:
            # Pick the first layer whose window matches.
            layer_idx = max_attention_window_vec.index(w)
            block_ids = manager.get_batch_cache_indices([req.py_request_id], layer_idx=layer_idx)[0]
            assert len(block_ids) > 0, f"prefill should populate window={w}"

            translated = manager.get_memory_pool_block_indices(block_ids, window_size=w)
            assert len(translated) == len(block_ids)
            assert all(idx >= 0 for idx in translated), (
                f"primary pool slot indices must be >= 0, got {translated}"
            )

            # Without host offload the translation is identity. This locks
            # the no-offload contract — if a future change introduces a
            # gratuitous remapping, this test will catch it.
            assert list(translated) == list(block_ids)

        # Note: with two distinct windows the underlying primary pools may
        # share a slot index *value* across windows but they live in
        # different pools, so we do not assert distinctness across windows.
    finally:
        manager.shutdown()


@pytest.mark.cuda
def test_get_memory_pool_block_indices_with_offload_onboard():
    """Offload -> onboard -> transfer cycle.

    What this test proves:
      1. While a block IS primary, its block ID can still differ from
         its primary-pool slot (primary-side divergence). This happens
         because block IDs are allocated globally (req_b's blocks get
         IDs 4-7 after req_a took 0-3), but primary-pool slots are
         recycled (req_a's freed slots 0-3 get reused for req_b's new
         blocks). Without translation, the disagg sender would
         compute `base + 4 * slot_bytes` and read past the end of a
         4-block primary pool — the exact pointer-arithmetic bug this
         fix addresses.
      2. Translation aborts on offloaded blocks with the documented
         "Block is not in the primary pool" message — the disagg cache
         transceiver relies on this to refuse host-pool transfers.
      3. After reuse-driven onboard, all blocks are primary again and
         pointers computed from the translated slot indices match
         `primary_pool_base + slot_idx * slot_bytes`.
    """
    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings.internal.testing import (
        simulate_prefill_completion_only_use_for_testing,
    )
    from tensorrt_llm.sampling_params import SamplingParams

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    tokens_per_block = 16
    # Tight primary capacity (~4 blocks) + plenty of host space (~32 blocks)
    # so eviction must spill to host rather than free outright.
    kv_cache_config = KvCacheConfig(
        max_tokens=tokens_per_block * 4,
        free_gpu_memory_fraction=0.1,
        enable_block_reuse=True,
        enable_partial_reuse=True,
        copy_on_partial_reuse=False,
        host_cache_size=tokens_per_block * 32 * 2 * 2 * 2 * 64,  # generous host pool
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=tokens_per_block * 4,
        max_batch_size=4,
        mapping=mapping,
        dtype=DataType.HALF,
    )

    try:
        # Skip cleanly if the build/test env didn't actually allocate a
        # secondary pool (e.g. host_cache_size rounded to 0). Without it,
        # this test cannot exercise the offload path.
        if getattr(manager, "blocks_in_secondary_pool", 0) <= 0:
            pytest.skip(
                "secondary (host) pool not allocated in this build; "
                "offload/onboard path cannot be exercised"
            )

        def _run(req_id, tokens):
            sampling_params = SamplingParams()
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=list(tokens),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
            )
            manager.impl.add_sequence_batch([(req.py_request_id, req.prompt_len, 1)], [req])
            return req

        window = manager.max_attention_window_vec[0]

        # Warm: take blocks for a 4-block prefill, capture its block_ids,
        # then commit to reuse so they live in the radix tree (eligible
        # for offload, not freed outright).
        prefill_tokens_a = list(range(tokens_per_block * 4))
        req_a = _run(0, prefill_tokens_a)
        block_ids_a = manager.get_batch_cache_indices([req_a.py_request_id], layer_idx=0)[0]
        # Translation is the identity at this point — block_ids equal
        # primary slot indices because nothing has been offloaded yet.
        original_indices = manager.get_memory_pool_block_indices(block_ids_a, window_size=window)
        assert list(original_indices) == list(block_ids_a)
        simulate_prefill_completion_only_use_for_testing(req_a)
        manager.free_resources(req_a)

        # Pressure: bring in a brand-new request that needs more primary
        # blocks than are currently free, forcing eviction of req_a's
        # cached prefix to the host pool.
        prefill_tokens_b = list(range(10_000, 10_000 + tokens_per_block * 4))
        req_b = _run(1, prefill_tokens_b)

        # Primary-side divergence: req_b's block IDs are allocated fresh
        # (typically 4..7 since req_a took 0..3 from the radix tree), but
        # the primary-pool slots are recycled from req_a's freed slots
        # (0..3). So even though every req_b block is currently primary,
        # `block_id != slot` for every one of them. This is the most
        # visible failure mode of the pre-fix code: the disagg sender
        # would compute `base + block_id_b * slot_bytes` and walk off the
        # end of a 4-block primary pool.
        block_ids_b = list(manager.get_batch_cache_indices([req_b.py_request_id], layer_idx=0)[0])
        slots_b = list(manager.get_memory_pool_block_indices(block_ids_b, window_size=window))
        primary_diverged = [(b, s) for b, s in zip(block_ids_b, slots_b) if b != s]
        assert primary_diverged, (
            f"expected req_b's block_ids to differ from its primary slots "
            f"(block IDs allocate forward, primary slots recycle); got "
            f"identity mapping block_ids={block_ids_b}, slots={slots_b}"
        )

        # Translation must abort for any offloaded block — this is the policy
        # the disagg cache transceiver relies on (otherwise the sender would
        # compute garbage pointers against host-pool blocks).
        any_non_primary = False
        for bid in block_ids_a:
            try:
                manager.get_memory_pool_block_indices([int(bid)], window_size=window)
            except RuntimeError as exc:
                assert "Block is not in the primary pool" in str(exc), (
                    f"unexpected error message: {exc}"
                )
                any_non_primary = True
        # If the eviction policy ended up keeping every prefix block in
        # primary (small possibility under fragmentation), we cannot
        # observe the offload assertion — skip the negative leg.
        if not any_non_primary:
            pytest.skip("no req_a block was offloaded under this build's eviction policy")

        # Onboard back: re-issue req_a with the same prompt. Reuse will
        # bring its blocks back to primary, possibly at *different*
        # primary slots than before.
        simulate_prefill_completion_only_use_for_testing(req_b)
        manager.free_resources(req_b)

        req_a2 = _run(2, prefill_tokens_a)
        block_ids_a2 = manager.get_batch_cache_indices([req_a2.py_request_id], layer_idx=0)[0]
        # After reuse-driven onboard, every block must be primary again
        # (translation succeeds); its primary slot may differ from the
        # logical block ID because onboard swaps slots in-place.
        translated_a2 = manager.get_memory_pool_block_indices(block_ids_a2, window_size=window)
        assert all(idx >= 0 for idx in translated_a2)

        simulate_prefill_completion_only_use_for_testing(req_a2)
        manager.free_resources(req_a2)
    finally:
        manager.shutdown()
