# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for mixed-precision KV allocation, pool sizing, reuse and transfer."""

from contextlib import closing

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="Mixed-precision NVFP4 KV requires Blackwell",
)

_MIXED_DTYPE = {
    DataType.FP8: "mixed_nvfp4_fp8",
    DataType.BF16: "mixed_nvfp4_bfloat16",
    DataType.HALF: "mixed_nvfp4_float16",
}


def mixed_config(dtype=DataType.BF16, **overrides):
    """A reuse-enabled config storing history in NVFP4 with one leading and 65 recent tokens."""
    options = dict(
        max_tokens=65536,
        enable_block_reuse=True,
        block_reuse_config={"policy": "per_request"},
        dtype=_MIXED_DTYPE[dtype],
        mixed_precision_first_n=1,
        mixed_precision_last_n=65,
        pool_ratio=[5 / 6, 1 / 6],
    )
    options.update(overrides)
    return KvCacheConfig(**options)


@pytest.fixture
def make_manager():
    managers = []

    def create(config=None, **kwargs):
        options = dict(
            num_layers=1,
            num_kv_heads=2,
            head_dim=64,
            tokens_per_block=128,
            max_seq_len=1024,
            max_batch_size=2,
            mapping=Mapping(),
            dtype=DataType.BF16,
        )
        options.update(kwargs)
        config = config or mixed_config(options["dtype"])
        manager = KVCacheManagerV2(config, CacheType.SELF, **options)
        managers.append(manager)
        return manager

    yield create
    for manager in reversed(managers):
        manager.shutdown()


def pool_pages(manager, layer_idx=0):
    """Page counts of the low and high pools of a mixed-precision layer."""
    return tuple(
        manager.get_buffers(layer_idx, mixed_precision_pool=pool).shape[0]
        for pool in ("low", "high")
    )


@pytest.mark.parametrize("dtype", (DataType.FP8, DataType.BF16, DataType.HALF))
def test_extra_pool_uses_existing_dtype_layout(make_manager, dtype):
    manager = make_manager(dtype=dtype)
    assert manager.mixed_precision_low_dtype == DataType.NVFP4
    assert manager.mixed_precision_high_layers == {0: 1}
    assert manager.kv_cache_manager_py_config.initial_pool_ratio == pytest.approx([5 / 6, 1 / 6])
    for pool, storage_dtype, width in (
        ("low", torch.uint8, 40),
        ("high", binding_to_torch_dtype(dtype), 64),
    ):
        nhd = manager.get_buffers(0, "NHD", mixed_precision_pool=pool)
        hnd = manager.get_buffers(0, "HND", mixed_precision_pool=pool)
        assert nhd.data_ptr() == hnd.data_ptr()
        assert nhd.dtype == hnd.dtype == storage_dtype
        assert nhd.shape[1:] == (2, 128, 2, width)
        assert hnd.shape[1:] == (2, 2, 128, width)


def test_plain_dtype_keeps_native_manager(make_manager):
    manager = make_manager(config=KvCacheConfig(max_tokens=65536, enable_block_reuse=True))
    assert manager.mixed_precision_low_dtype is None
    assert manager.mixed_precision_high_layers == {}
    assert len(manager.kv_cache_manager_py_config.layers) == 1
    assert manager.get_buffers(0).dtype == torch.bfloat16


def test_mixed_precision_preserves_native_and_shared_kv_storage(make_manager):
    options = dict(
        num_layers=4,
        pretrained_config=type(
            "Config",
            (),
            dict(
                num_hidden_layers=4,
                num_kv_shared_layers=1,
                layer_types=[
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                    "full_attention",
                ],
            ),
        )(),
    )
    windows = [1024, 513, 1024, 1024]
    native = make_manager(
        config=KvCacheConfig(
            max_tokens=65536,
            enable_block_reuse=True,
            max_attention_window=windows,
            pool_ratio=[0.5, 0.5],
        ),
        **options,
    )
    # Layer groups: full attention (mixed-precision low), sliding window, first/last tokens
    # (mixed-precision high).
    mixed = make_manager(
        config=mixed_config(max_attention_window=windows, pool_ratio=[0.25, 0.5, 0.25]),
        **options,
    )
    # Layer 3 feeds the shared KV layer 4 and stays native, like the SWA layer 1.
    assert mixed.mixed_precision_high_layers == {0: 4}
    for layer in (1, 2, 3):
        original = native.kv_cache_manager_py_config.layers[layer]
        mixed_manager_layer = mixed.kv_cache_manager_py_config.layers[layer]
        assert mixed_manager_layer.sliding_window_size == original.sliding_window_size
        assert mixed_manager_layer.reuse_policy == original.reuse_policy
        assert [(b.role, b.size) for b in mixed_manager_layer.buffers] == [
            (b.role, b.size) for b in original.buffers
        ]
        assert mixed.get_buffers(layer).dtype == native.get_buffers(layer).dtype
        assert mixed.get_buffers(layer).shape[1:] == native.get_buffers(layer).shape[1:]


def test_mixed_precision_keeps_kv_quota(make_manager):
    # Attention stages the native-dtype copy from its own buffer pool, so the
    # manager hands the whole KV budget to the pools.
    budget = 64 << 20
    native = make_manager(config=KvCacheConfig(enable_block_reuse=True, max_gpu_total_bytes=budget))
    mixed = make_manager(
        config=mixed_config(max_tokens=None, max_gpu_total_bytes=budget, pool_ratio=[0.5, 0.5])
    )
    assert native.kv_cache_manager_py_config.cache_tiers[0].quota == budget
    assert mixed.kv_cache_manager_py_config.cache_tiers[0].quota == budget


def test_pool_ratio_must_cover_mixed_precision_high_group(make_manager):
    # The runtime counts the appended mixed-precision high group as a layer group.
    with pytest.raises(ValueError, match="layer groups"):
        make_manager(config=mixed_config(pool_ratio=[1.0]))


def test_automatic_pool_sizing_defaults_to_half_max_seq_len(make_manager):
    """Without avg_seq_len the pools are sized as for max_seq_len / 2, not the backend's 2048-token request."""
    options = dict(pool_ratio=None, max_tokens=1 << 20, max_gpu_total_bytes=1 << 30)
    implicit = make_manager(config=mixed_config(**options), max_batch_size=64)
    explicit = make_manager(config=mixed_config(avg_seq_len=512, **options), max_batch_size=64)
    typical = implicit.kv_cache_manager_py_config.typical_step
    assert typical is not None
    assert {desc.capacity for desc in typical.kv_caches[1:]} == {512}
    assert pool_pages(implicit) == pool_pages(explicit)


@pytest.mark.parametrize("policy,reused_tokens", [("per_request", 128), ("all_reusable", 512)])
def test_native_common_reuse_and_window_retention(make_manager, policy, reused_tokens):
    manager = make_manager(config=mixed_config(block_reuse_config={"policy": policy}))
    config = manager.kv_cache_manager_py_config
    assert [layer.reuse_policy for layer in config.layers] == [kv.ReusePolicy.REQUIRED] * 2
    assert (config.layers[1].num_sink_tokens, config.layers[1].sliding_window_size) == (1, 66)
    assert config.commit_min_snapshot == (policy == "per_request")
    native = manager.impl
    stream = torch.cuda.current_stream().cuda_stream
    tokens = list(range(640))
    high_group = native.get_layer_group_id(manager.mixed_precision_high_layers[0])
    with closing(native.create_kv_cache(None, [])) as source:
        assert source.resume(stream) and source.resize(len(tokens))
        source.commit(tokens)
        source.suspend()
        assert source.resume(stream, is_decoding=True)
        pages = list(source.get_base_page_indices(high_group))
        assert pages[0] >= 0 and pages[4] >= 0
        assert pages[1:4] == [kv.BAD_PAGE_INDEX] * 3
    with closing(native.create_kv_cache(None, tokens)) as full:
        assert full.num_committed_tokens == 640
        assert full.resume(stream)
    with closing(native.create_kv_cache(None, tokens[:512])) as shortened:
        # Per-request snapshots keep only the first page reusable because the
        # required high tail must be complete; all-reusable commits every page.
        assert shortened.num_committed_tokens == reused_tokens
        assert shortened.resume(stream)


def test_disagg_exports_both_pools_and_pairs_sink_tail(make_manager):
    import numpy as np

    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import global_layer_ids

    manager = make_manager()
    rank = RankInfo.from_kv_cache_manager("mixed", manager, device_id=0)
    table = rank.page_table
    assert [group.local_layers[0].global_layer_id for group in table.layer_groups] == [0, 1]
    # KV cache sharing names the layers the same way.
    assert global_layer_ids(manager, [0, 1]) == [0, 1]
    assert [len(group.pool_views) for group in table.layer_groups] == [1, 1]
    registrar = PeerRegistrar(rank, KVRegionExtractorV1(table))
    peer = RankInfo.from_bytes(rank.to_bytes())
    peer.instance_name = "decode"
    assert registrar.get_pool_mapping(peer) == {(0, 0): (0, 0), (1, 0): (1, 0)}
    source, destination = Sender._pair_ordinals(
        np.array([4, -1, -1, -1, 7]), np.array([2, -1, -1, -1, 6])
    )
    assert source.tolist() == [4, 7]
    assert destination.tolist() == [2, 6]


def test_mixed_precision_rejects_custom_managers():
    """A manager subclass must opt in to mixed-precision KV; others are refused up front."""

    class _CustomManager(KVCacheManagerV2):
        pass

    manager = object.__new__(_CustomManager)
    with pytest.raises(NotImplementedError, match="_CustomManager"):
        manager._init_mixed_precision(
            KvCacheConfig(dtype="mixed_nvfp4_fp8"), None, 1024, DataType.FP8
        )
    # A plain dtype never reaches the check.
    manager._init_mixed_precision(KvCacheConfig(), None, 1024, DataType.FP8)
    assert manager.mixed_precision_high_layers == {}

    # Per-model subclasses inherit the declaration of the manager they wrap.
    class _OptedIn(_CustomManager):
        _supports_mixed_precision_kv = True

    class _Wrapper(_OptedIn):
        pass

    assert object.__new__(_Wrapper)._supports_mixed_precision_kv

    # A manager whose pool already holds the mixed-precision low (NVFP4) history accepts any pool dtype.
    class _OwnPoolIsMixedPrecisionLow(_OptedIn):
        _own_pool_is_mixed_precision_low = True

    own_pool_low = object.__new__(_OwnPoolIsMixedPrecisionLow)
    own_pool_low._init_mixed_precision(
        KvCacheConfig(dtype="mixed_nvfp4_bfloat16"), None, 1024, DataType.FP8
    )
    assert own_pool_low.mixed_precision_high_dtype == DataType.BF16


def test_separate_mtp_draft_manager_goes_mixed(make_manager):
    """A separate MTP draft manager adds mixed-precision high layers like the target."""
    from types import SimpleNamespace

    from tensorrt_llm.llmapi import MTPDecodingConfig

    manager = make_manager(
        config=mixed_config(pool_ratio=None),
        num_layers=1,
        layer_mask=[False, False, True],
        spec_config=MTPDecodingConfig(num_nextn_predict_layers=1),
        pretrained_config=SimpleNamespace(num_hidden_layers=2),
        is_draft=True,
    )
    assert manager.pp_layers == [2]
    assert manager.mixed_precision_high_layers == {0: 1}


def test_deepseek_v4_mixed_disagg_names_every_layer():
    """Only compressed DeepSeek-V4 layers use mixed-precision KV, not the MTP layer.

    Disaggregated transfer gives each mixed-precision high layer its own layer id.
    """
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
    from tensorrt_llm.llmapi import MTPDecodingConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

    ratios = [1, 4, 128, 4, 1]
    manager = DeepseekV4CacheManager(
        kv_cache_config=mixed_config(DataType.FP8, pool_ratio=None, enable_block_reuse=False),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=4,
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=4096,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.NVFP4,
        vocab_size=1000,
        max_input_len=4096,
        max_num_tokens=8192,
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=ratios
        ),
        spec_config=MTPDecodingConfig(max_draft_len=1),
        pretrained_config=SimpleNamespace(num_hidden_layers=4),
    )
    try:
        # Compressed layers 1-3 get a mixed-precision high layer; layer 0 (ratio 1) and MTP layer 4
        # stay native.
        assert manager.pp_layers == [0, 1, 2, 3, 4]
        assert sorted(manager.mixed_precision_high_layers) == [1, 2, 3]
        rank = RankInfo.from_kv_cache_manager("dsv4_mixed", manager, device_id=0)
        ids = [
            layer.global_layer_id
            for group in rank.page_table.layer_groups
            for layer in group.local_layers
        ]
        assert len(ids) == len(set(ids))
        # The high layer of model layer L is 2L + 1; every other id is doubled, so even.
        high_ids = {2 * layer + 1 for layer in (1, 2, 3)}
        assert high_ids <= set(ids) and all(i % 2 == 0 for i in set(ids) - high_ids)
        # A peer with the same layout pairs every pool, mixed-precision high pools included.
        registrar = PeerRegistrar(rank, KVRegionExtractorV1(rank.page_table))
        peer = RankInfo.from_bytes(rank.to_bytes())
        peer.instance_name = "decode"
        mapping = registrar.get_pool_mapping(peer)
        assert all(source == destination for source, destination in mapping.items())
        high_layer_groups = {
            int(manager.impl.get_layer_group_id(t))
            for t in manager.mixed_precision_high_layers.values()
        }
        assert high_layer_groups <= {group for group, _ in mapping}
    finally:
        manager.shutdown()


def test_disagg_ids_do_not_depend_on_mixed_layers_of_a_rank(make_manager):
    """A pipeline rank without mixed-precision layers still names model layer L as 2L."""
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

    # Sliding-window layers stay native, so this rank holds no mixed-precision layer.
    manager = make_manager(
        config=mixed_config(pool_ratio=None, max_attention_window=[64]), num_layers=2
    )
    assert manager.mixed_precision_high_layers == {}
    table = RankInfo.from_kv_cache_manager("mixed", manager, device_id=0).page_table
    ids = [layer.global_layer_id for group in table.layer_groups for layer in group.local_layers]
    assert sorted(ids) == [0, 2]


@pytest.mark.cpu_only
def test_mixed_precision_support_of_each_manager():
    """Managers of supported models opt in; others refuse mixed-precision KV at construction."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
        Glm5NextCacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
        MiniMaxM3KVCacheManagerV2,
    )
    from tensorrt_llm._torch.modules.fla.cache_manager import Qwen35HybridCacheManager
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.modules.mamba.cache_manager import NemotronHybridCacheManager
    from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import Qwen4ExpHybridCacheManager
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
        MambaHybridCacheManagerV2,
    )

    supported = {
        KVCacheManagerV2,
        MambaHybridCacheManagerV2,
        Qwen35HybridCacheManager,
        KimiK3HybridCacheManager,
        NemotronHybridCacheManager,
        DSACacheManagerV2,
        DeepseekV4CacheManager,
    }
    unsupported = {
        Qwen4ExpHybridCacheManager,
        Glm5NextCacheManager,
        CSA2CacheManager,
        MiniMaxM3KVCacheManagerV2,
        Fp4MlaKVCacheManagerV2,
    }
    for cls in supported | unsupported:
        opted_in = cls is KVCacheManagerV2 or cls._supports_mixed_precision_kv
        assert opted_in == (cls in supported), cls


@pytest.mark.parametrize("backend", ["dense", "dsa"])
def test_target_manager_beside_a_separate_eagle3_drafter_builds(backend):
    """With a separate draft manager the target manager holds only model layers, all mixed."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
    from tensorrt_llm.llmapi import Eagle3DecodingConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    options = dict(
        kv_cache_config=mixed_config(DataType.FP8, pool_ratio=None, enable_block_reuse=False),
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=64,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        spec_config=Eagle3DecodingConfig(max_draft_len=1, speculative_model="draft-model"),
        layer_mask=[True, True],
        pretrained_config=SimpleNamespace(num_hidden_layers=2),
    )
    if backend == "dsa":
        manager = DSACacheManagerV2(
            kv_cache_type=CacheType.SELFKONLY,
            sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=128),
            dtype=DataType.NVFP4,
            **options,
        )
    else:
        manager = KVCacheManagerV2(kv_cache_type=CacheType.SELF, dtype=DataType.FP8, **options)
    try:
        assert manager.pp_layers == [0, 1]
        assert sorted(manager.mixed_precision_high_layers) == [0, 1]
    finally:
        manager.shutdown()


def test_dsa_prepare_fills_the_mixed_precision_high_page_table():
    """DSA prepare() writes each request's high-pool pages to the GPU page table, -1 elsewhere."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import (
        DSACacheManagerV2,
        DSAtrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    # index_topk below the request lengths, so prepare() does not skip the indexer.
    sparse_config = DeepSeekSparseAttentionConfig(index_head_dim=128, index_topk=256)
    manager = DSACacheManagerV2(
        mixed_config(
            DataType.FP8,
            mixed_precision_first_n=64,
            mixed_precision_last_n=64,
            pool_ratio=None,
            enable_block_reuse=False,
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=64,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.NVFP4,
        sparse_attention_config=sparse_config,
    )
    try:
        # Two decode requests with 299 and 289 cached tokens, each writing one new token.
        request_ids, cached = [0, 1], [299, 289]
        assert manager.add_dummy_requests(request_ids, [n + 1 for n in cached], is_gen=True)
        metadata = DSAtrtllmAttentionMetadata(
            max_num_requests=2,
            max_num_tokens=2,
            kv_cache_manager=manager,
            sparse_metadata_params=sparse_config.to_sparse_metadata_params(),
            seq_lens=torch.tensor([1, 1], dtype=torch.int),
            num_contexts=0,
            request_ids=request_ids,
            prompt_lens=cached,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
        )
        metadata.prepare()
        table = metadata.mixed_precision_high_block_table.cpu()
        expected = torch.full_like(table, -1)
        high_layer = manager.mixed_precision_high_layers[0]
        manager._fill_page_table(high_layer, request_ids, expected.numpy())
        assert table.tolist() == expected.tolist()
        # The high pool keeps page 0 (first 64 tokens) and pages 3-4 (last 64 and the new one).
        held = [[i for i, page in enumerate(row) if page != -1] for row in table.tolist()]
        assert held == [[0, 3, 4]] * 2
    finally:
        manager.shutdown()


def test_mixed_mtp_draft_manager_with_target_pool_ratio(make_manager):
    """A target pool_ratio sizes the target; the mixed MTP draft manager still builds."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor._util import _fit_draft_pool_ratio
    from tensorrt_llm.llmapi import MTPDecodingConfig

    config = mixed_config(pool_ratio=[0.75, 0.25], max_tokens=None, max_gpu_total_bytes=128 << 20)
    target = make_manager(config=config, num_layers=2)
    assert target.kv_cache_manager_py_config.initial_pool_ratio == pytest.approx([0.75, 0.25])
    draft_config = config.model_copy()
    _fit_draft_pool_ratio(draft_config)
    draft = make_manager(
        config=draft_config,
        num_layers=1,
        layer_mask=[False, False, True],
        spec_config=MTPDecodingConfig(num_nextn_predict_layers=1),
        pretrained_config=SimpleNamespace(num_hidden_layers=2),
        is_draft=True,
    )
    assert draft.mixed_precision_high_layers == {0: 1}


def test_mixed_precision_high_window_covers_every_draft_step(make_manager):
    """A window of last_n + 1 holds every position an MTP step reads or writes.

    Each step grows the capacity by 1 + draft tokens: its queries at positions history ..
    history + draft read the last_n tokens before them, and the draft loop writes up to
    history + draft. Then 0..draft drafts are accepted and the history advances.
    """
    import random

    draft, tokens_per_block, first_n, last_n = 3, 16, 4, 20
    manager = make_manager(
        config=mixed_config(
            pool_ratio=None,
            enable_block_reuse=False,
            mixed_precision_first_n=first_n,
            mixed_precision_last_n=last_n,
        ),
        tokens_per_block=tokens_per_block,
        max_seq_len=512,
    )
    assert manager.kv_cache_manager_py_config.layers[1].sliding_window_size == last_n + 1
    native = manager.impl
    stream = torch.cuda.current_stream().cuda_stream
    high_group = native.get_layer_group_id(manager.mixed_precision_high_layers[0])
    rng = random.Random(0)
    history = 100
    with closing(native.create_kv_cache(None, [])) as kv_cache:
        assert kv_cache.resume(stream) and kv_cache.resize(history)
        while history + 1 + draft < 400:
            assert kv_cache.resize(history + 1 + draft, history)
            pages = list(kv_cache.get_base_page_indices(high_group))
            first_read = (history - last_n) // tokens_per_block
            last_write = (history + draft) // tokens_per_block
            held = [
                pages[block] != kv.BAD_PAGE_INDEX for block in range(first_read, last_write + 1)
            ]
            assert all(held), (history, first_read, last_write, pages)
            assert pages[0] != kv.BAD_PAGE_INDEX  # the first_n sink
            history += 1 + rng.randint(0, draft)
        # The window is tight: pages well before the last_n tokens were released.
        assert pages[2] == kv.BAD_PAGE_INDEX
