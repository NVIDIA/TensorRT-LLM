# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for mixed-precision KV allocation, pool sizing, reuse and transfer."""

from contextlib import closing

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
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
    assert manager.low_precision_dtype == DataType.NVFP4
    assert manager.high_layers == {0: 1}
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
    assert manager.low_precision_dtype is None
    assert manager.high_layers == {}
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
    # Layer groups: full attention (low), sliding window, retained tokens (high).
    mixed = make_manager(
        config=mixed_config(max_attention_window=windows, pool_ratio=[0.25, 0.5, 0.25]),
        **options,
    )
    # Layer 3 feeds the shared KV layer 4 and stays native, like the SWA layer 1.
    assert mixed.high_layers == {0: 4}
    for layer in (1, 2, 3):
        original = native.kv_cache_manager_py_config.layers[layer]
        retained = mixed.kv_cache_manager_py_config.layers[layer]
        assert retained.sliding_window_size == original.sliding_window_size
        assert retained.reuse_policy == original.reuse_policy
        assert [(b.role, b.size) for b in retained.buffers] == [
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


def test_pool_ratio_must_cover_retained_group(make_manager):
    # The runtime counts the appended retained-token group as a layer group.
    with pytest.raises(ValueError, match="layer groups"):
        make_manager(config=mixed_config(pool_ratio=[1.0]))


@pytest.mark.parametrize("ratio", ([0.6, 0.4], [0.75, 0.25]))
def test_explicit_pool_ratio_orders_retained_group_last(make_manager, ratio):
    # A byte budget, so the ratio is not distorted by the max_tokens cap.
    manager = make_manager(
        config=mixed_config(pool_ratio=ratio, max_tokens=None, max_gpu_total_bytes=128 << 20)
    )
    assert manager.kv_cache_manager_py_config.initial_pool_ratio == pytest.approx(ratio)
    assert [list(group) for group in manager.impl.layer_grouping] == [[0], [1]]
    low_pages, high_pages = pool_pages(manager)
    low_bytes = low_pages * manager.get_layer_bytes_per_token(0, Role.ALL) * 128
    high_bytes = high_pages * manager.kv_factor * 128 * 2 * 64 * 2
    print(
        f"explicit ratio {ratio}: low {low_pages} pages / {low_bytes >> 20} MiB, "
        f"high {high_pages} pages / {high_bytes >> 20} MiB"
    )
    assert low_bytes / (low_bytes + high_bytes) == pytest.approx(ratio[0], abs=0.05)


@pytest.mark.parametrize("estimating", (False, True), ids=("final", "estimate"))
def test_automatic_pool_sizing_covers_retained_group(make_manager, estimating):
    # A budget large enough that the typical workload, not the allocation
    # floors, decides the split: 64 resident 768-token requests.
    manager = make_manager(
        config=mixed_config(
            pool_ratio=None, avg_seq_len=768, max_tokens=1 << 20, max_gpu_total_bytes=1 << 30
        ),
        max_batch_size=64,
        is_estimating_kv_cache=estimating,
    )
    assert manager.kv_cache_manager_py_config.initial_pool_ratio is None
    low_pages, high_pages = pool_pages(manager)
    low_bytes = low_pages * manager.get_layer_bytes_per_token(0, Role.ALL) * 128
    high_bytes = high_pages * manager.kv_factor * 128 * 2 * 64 * 2
    print(
        f"automatic sizing: low {low_pages} pages / {low_bytes >> 20} MiB, "
        f"high {high_pages} pages / {high_bytes >> 20} MiB"
    )
    # Every resident request needs 6 history pages but only the sink page and
    # the 66-token window (2-3 pages) in the retained group. Retained pages are
    # BF16 and 3.2x larger than NVFP4 history pages, so compare page counts.
    assert low_pages >= 64 * 6
    assert high_pages >= 64 * 2
    assert high_pages < low_pages


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
    high_group = native.get_layer_group_id(manager.high_layers[0])
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

    manager = make_manager()
    rank = RankInfo.from_kv_cache_manager("mixed", manager, device_id=0)
    table = rank.page_table
    assert [group.local_layers[0].global_layer_id for group in table.layer_groups] == [0, 1]
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
    """Hybrid and sparse managers own extra pools; mixed precision refuses them up front."""

    class _CustomManager(KVCacheManagerV2):
        pass

    manager = object.__new__(_CustomManager)
    with pytest.raises(NotImplementedError, match="_CustomManager"):
        manager._init_mixed_precision(
            KvCacheConfig(dtype="mixed_nvfp4_fp8"), None, 1024, DataType.FP8
        )
    # A plain dtype never reaches the check.
    manager._init_mixed_precision(KvCacheConfig(), None, 1024, DataType.FP8)
    assert manager.high_layers == {}
