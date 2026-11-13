# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for mixed-precision KV allocation, pool sizing, reuse and transfer."""

from contextlib import closing
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_staging import MixedPrecisionKVStaging
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

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


class _ColdPageProvider:
    """Cold-page codec provider stand-in: the manager reads only its first/last-N window."""

    config = SimpleNamespace(algorithm="quantization_for_cold_page")

    def __init__(self, first_n=1, last_n=65):
        self.mixed_precision_high_window = (first_n, last_n)
        self.calls = []

    def create_cold_page_codec(self, cache_config, **kwargs):
        self.calls.append(dict(cache_config=cache_config, **kwargs))
        return None


def cold_page_config(**overrides):
    """A reuse-enabled plain-dtype config; the first/last-N window comes from the provider."""
    options = dict(
        max_tokens=65536,
        enable_block_reuse=True,
        block_reuse_config={"policy": "per_request"},
        pool_ratio=[5 / 6, 1 / 6],
    )
    options.update(overrides)
    return KvCacheConfig(**options)


def pool_pages(manager, layer_idx=0):
    """Page counts of the low and high pools of a mixed-precision layer."""
    return tuple(
        manager.get_buffers(layer_idx, mixed_precision_pool=pool).shape[0]
        for pool in ("low", "high")
    )


def _request(request_id, tokens):
    """A request that prefills `tokens` and generates one token."""
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def _cache_prompt(manager, tokens):
    """Compute and commit `tokens` as request 1, then free it, so a later request can reuse them."""
    source = _request(1, tokens)
    assert manager.prepare_context(source) and manager.resize_context(source, len(tokens))
    source.move_to_next_context_chunk()
    manager.try_commit_blocks(source)
    manager.free_resources(source)


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
        assert source.resume(stream)
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


def test_reuse_last_n_from_low_restores_evicted_window(make_manager):
    """A prefix whose high-precision last-N pages were evicted is reused in full; resuming the
    cache decodes those pages from the low-precision history."""
    manager = make_manager(config=mixed_config(mixed_precision_reuse_last_n_from_low=True))
    block = manager.tokens_per_block
    staging = MixedPrecisionKVStaging(
        manager, Buffers(), rows=1, layout="NHD", pool_layout="NHD", tokens_per_block=block
    )
    pool, low, high = staging.pools[0]

    # The source computes 640 tokens whose K and V repeat on every page.
    tokens = list(range(640))
    source = _request(1, tokens)
    assert manager.prepare_context(source) and manager.resize_context(source, len(tokens))
    staging.update([1], [0], [len(tokens)], capture_graph=False)
    page_values = torch.randn(pool.shape[1:], device="cuda").to(pool.dtype)
    pool[: len(tokens) // block] = page_values
    staging.convert(0, writeback=True)
    source.move_to_next_context_chunk()
    manager.try_commit_blocks(source)
    manager.free_resources(source)

    # Per-request snapshots keep the high pages of the source's last window only, so the
    # window at 512 tokens is missing; REQUIRED would stop reuse at the first page.
    reader = _request(2, tokens[:512] + [9999])
    assert manager.prepare_context(reader)
    assert manager.kv_cache_map[2].num_committed_tokens == 512
    # Suspended before its first forward, the cache resumes without selecting again.
    manager.kv_cache_map[2].suspend()
    assert manager.prepare_context(reader) and manager.resize_context(reader, 1)
    low_pages, high_pages = (table[0] for table in manager.get_mixed_precision_page_tables([2], 0))
    # The sink page was claimed with the prefix, pages 1 and 2 left the window, 3 is fresh.
    assert high_pages[0] >= 0 and high_pages[1:3] == [kv.BAD_PAGE_INDEX] * 2
    assert high_pages[3] >= 0
    low_before = low[low_pages].clone()
    pool.zero_()

    staging.update([2], [512], [1], capture_graph=False)
    staging.convert(0)
    # The sink keeps its high-precision values. Pages 1 and 2 decode from the low pool, and the
    # resume left that same decoded window in the fresh high page 3.
    assert torch.equal(pool[0], page_values) and torch.equal(high[high_pages[0]], page_values)
    assert torch.equal(pool[3], pool[1]) and not torch.equal(pool[3], page_values)
    assert torch.equal(low[low_pages], low_before)
    manager.free_resources(reader)


@pytest.mark.parametrize("recompute,reused", [(False, 512), (True, 384)])
@pytest.mark.parametrize("window", ["mixed_dtype", "cold_page"])
def test_reuse_without_the_last_n(make_manager, window, recompute, reused):
    """A prefix whose high-precision last-N pages were evicted is reused in full, or with
    recompute only up to the block boundary before them. The resume restores the window: the
    cold-page window copies the layer's own pages, checked here."""
    options = dict(
        mixed_precision_reuse_last_n_from_low=True, mixed_precision_recompute_last_n=recompute
    )
    if window == "mixed_dtype":
        manager = make_manager(config=mixed_config(**options))
    else:
        manager = make_manager(
            config=cold_page_config(**options), cold_page_codec_provider=_ColdPageProvider()
        )
    assert [layer.reuse_policy for layer in manager.kv_cache_manager_py_config.layers] == [
        kv.ReusePolicy.REQUIRED,
        kv.ReusePolicy.SINK_REQUIRED_ONLY,
    ]
    tokens = list(range(640))
    _cache_prompt(manager, tokens)
    low, high = (manager.get_buffers(0, mixed_precision_pool=pool) for pool in ("low", "high"))
    for pool in (low, high):
        pool.view(torch.uint8).copy_(torch.randint_like(pool.view(torch.uint8), 0, 120))

    # Per-request snapshots keep the high pages of the source's last window only. The window at
    # 512 tokens is missing; recompute backs off by last_n = 65 tokens to the block boundary at
    # 384, where the window is missing too. Either way the window is the page before `reused`.
    reader = _request(2, tokens[:512] + [9999])
    assert manager.prepare_context(reader)
    assert manager.kv_cache_map[2].num_committed_tokens == reused
    if window == "cold_page":
        torch.cuda.synchronize()
        pages = torch.full((2, 1, 8), -1, dtype=torch.int32)
        manager.fill_mixed_precision_page_tables([2], 0, pages.numpy())
        page = reused // manager.tokens_per_block - 1
        assert torch.equal(
            high[pages[1, 0, page]].view(torch.uint8), low[pages[0, 0, page]].view(torch.uint8)
        )
    manager.free_resources(reader)


def test_recompute_paired_draft_pool_claims_the_targets_depth(make_manager):
    """A paired draft pool claims the depth the target settled on after backing off before its
    missing last-N window, and restores its own window there instead of backing off again."""
    draft = make_manager(
        config=mixed_config(
            mixed_precision_reuse_last_n_from_low=True, mixed_precision_recompute_last_n=True
        ),
        is_draft=True,
        joint_kv_cache_reuse=True,
    )

    tokens = list(range(640))
    source = _request(1, tokens)
    assert draft.prepare_context_cache(source) == 0
    assert draft.try_allocate_draft_context(source, len(tokens))
    source.move_to_next_context_chunk()
    draft.try_commit_blocks(source)
    draft.free_resources(source)

    # The target backed off from 512 to 384; the draft takes 384 too.
    reader = _request(2, tokens[:512] + [9999])
    assert draft.prepare_context_cache(reader, 384) == 384
    draft.free_resources(reader)


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


@pytest.mark.parametrize("dtype", (DataType.FP8, DataType.BF16))
def test_cold_page_window_keeps_history_pool_native(make_manager, dtype):
    """first/last-N from the cold-page codec adds the high group; the history pool stays native."""
    provider = _ColdPageProvider()
    manager = make_manager(
        config=cold_page_config(), dtype=dtype, cold_page_codec_provider=provider
    )
    assert manager.mixed_precision_low_dtype == dtype
    assert (manager.first_n, manager.last_n) == (1, 65)
    assert manager.mixed_precision_high_layers == {0: 1}
    # The codec never sees the high layer, so its pages migrate unchanged.
    assert [
        [int(layer.layer_id) for layer in call["cache_config"].layers] for call in provider.calls
    ] == [[0]]
    element_size = 1 if dtype == DataType.FP8 else 2
    low = manager.get_buffers(0, "NHD", mixed_precision_pool="low")
    high = manager.get_buffers(0, "NHD", mixed_precision_pool="high")
    assert low.dtype == torch.uint8 and low.shape[1:] == (2, 128, 2, 64 * element_size)
    assert high.dtype == binding_to_torch_dtype(dtype) and high.shape[1:] == (2, 128, 2, 64)
    # Native bytes per token: the cold-page codec sees an ordinary KV layer.
    assert manager.get_layer_bytes_per_token(0, Role.KEY) == 2 * 64 * element_size
    config = manager.kv_cache_manager_py_config
    assert config.initial_pool_ratio == pytest.approx([5 / 6, 1 / 6])
    assert (config.layers[1].num_sink_tokens, config.layers[1].sliding_window_size) == (1, 66)
    assert [buffer.size for buffer in config.layers[1].buffers] == [
        buffer.size for buffer in config.layers[0].buffers
    ]
    # A high page is a page of the layer's own pool with the layer's slot layout.
    assert manager.mixed_precision_high_layer_offsets == {0: 0}
    assert manager.mixed_precision_high_block_deltas == {0: 0}


def test_cold_page_window_high_pages_cross_the_host_tier_unchanged(make_manager):
    """With the real NVFP4 cold-page codec, a page the high layer holds comes back from the host
    tier bit-exact, while the layer's own page of the same tokens comes back quantized."""
    from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
        Nvfp4ColdPageQuantizationCompression,
    )
    from tensorrt_llm.bindings.internal import kv_cache_compression as native_kvcc
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig

    provider = Nvfp4ColdPageQuantizationCompression(
        ColdPageQuantizationCompressionConfig(mixed_precision_first_n=1, mixed_precision_last_n=65),
        pretrained_config=SimpleNamespace(model_type="qwen3"),
    )
    manager = make_manager(
        config=cold_page_config(max_tokens=4096, host_cache_size=64 << 20),
        cold_page_codec_provider=provider,
    )
    native = manager.impl
    stream = torch.cuda.current_stream().cuda_stream
    high_layer = manager.mixed_precision_high_layers[0]
    low = manager.get_buffers(0, "NHD", mixed_precision_pool="low").view(torch.bfloat16)
    high = manager.get_buffers(0, "NHD", mixed_precision_pool="high")

    def view_pages(cache, layer):
        """The cache's blocks as pages of the layer's pool view (-1 for a released block)."""
        scale = int(native.get_page_index_converter(layer, Role.KEY).scale)
        return [
            page * scale // manager.kv_factor if page != kv.BAD_PAGE_INDEX else -1
            for page in cache.get_base_page_indices(native.get_layer_group_id(layer))[:5]
        ]

    tokens = list(range(5 * 128))
    with (
        patch.object(
            native_kvcc, "nvfp4_cold_page_encode", wraps=native_kvcc.nvfp4_cold_page_encode
        ) as encode,
        patch.object(
            native_kvcc, "nvfp4_cold_page_decode", wraps=native_kvcc.nvfp4_cold_page_decode
        ) as decode,
        closing(native.create_kv_cache(None, [])) as source,
    ):
        assert source.resume(stream) and source.resize(len(tokens))
        torch.manual_seed(0)
        written = {}
        for layer, pool in ((0, low), (high_layer, high)):
            for ordinal, page in enumerate(view_pages(source, layer)):
                pool[page].normal_()
                written[layer, ordinal] = pool[page].clone()
        torch.cuda.synchronize()
        source.commit(tokens)
        source.suspend()
        # Fresh caches take the GPU pages one after another; the suspended cache's committed
        # pages are the least recently used, so they leave for the host tier first.
        pressure = []
        try:
            for _ in range(64):
                cache = native.create_kv_cache(None, [])
                pressure.append(cache)
                assert cache.resume(stream) and cache.resize(256)
                cache.suspend()
                if _introspection.active_page_stats(source)[0][0] == 0:
                    break
            torch.cuda.synchronize()
            # Five pages of the layer and the two the high layer kept, all on the host.
            assert _introspection.active_page_stats(source)[0][:2] == [0, 7]
        finally:
            for cache in pressure:
                cache.close()
        assert encode.call_count > 0 and decode.call_count == 0
        assert source.resume(stream)
        torch.cuda.synchronize()
        assert _introspection.active_page_stats(source)[0][1] == 0
        assert decode.call_count > 0
        # The high layer's pages never went through the codec, so the window's pages come back
        # bit-exact; the layer's own pages were quantized on the way out.
        high_pages = view_pages(source, high_layer)
        assert [page >= 0 for page in high_pages] == [True, False, False, False, True]
        for ordinal in (0, 4):
            assert torch.equal(high[high_pages[ordinal]], written[high_layer, ordinal])
        for ordinal, page in enumerate(view_pages(source, 0)):
            restored, original = low[page].float(), written[0, ordinal].float()
            assert not torch.equal(restored, original)
            assert (restored - original).abs().mean() < 0.5 * original.abs().mean()


def test_cold_page_mixed_precision_is_per_layer_and_skips_sliding_window_layers(make_manager):
    """First/last-N mixed precision is per layer: a sliding-window layer gets no high layer and
    stays an ordinary cold-page layer of the codec."""
    provider = _ColdPageProvider()
    manager = make_manager(
        config=cold_page_config(max_attention_window=[1024, 513], pool_ratio=[0.5, 0.25, 0.25]),
        num_layers=2,
        cold_page_codec_provider=provider,
    )
    # Only the full-attention layer 0 takes the first/last-N mixed precision; its high layer is
    # appended as layer 2.
    assert manager.mixed_precision_high_layers == {0: 2}
    config = manager.kv_cache_manager_py_config
    assert config.layers[1].sliding_window_size == 513
    assert (config.layers[2].num_sink_tokens, config.layers[2].sliding_window_size) == (1, 66)
    # The codec quantizes both model layers, the sliding-window layer included, never the high layer.
    assert [
        [int(layer.layer_id) for layer in call["cache_config"].layers] for call in provider.calls
    ] == [[0, 1]]


@pytest.mark.parametrize(
    "first_n, last_n, builds",
    [(0, 127, False), (0, 126, True), (128, 127, True)],
    ids=["same_retention", "one_token_less", "different_sink_pages"],
)
def test_cold_page_mixed_precision_rejects_a_sliding_window_with_the_high_layers_retention(
    make_manager, first_n, last_n, builds
):
    """A sliding-window layer that keeps as many tokens as the high layers (a window of
    mixed_precision_last_n + 1 tokens, the same sink pages) would share their page group, which the
    cold-page codec cannot split; the manager refuses that first/last-N combination and names the
    layer. One token less, or another sink page count, keeps the layer and the high layers apart."""
    options = dict(
        config=cold_page_config(max_attention_window=[128, 1024], pool_ratio=[0.5, 0.25, 0.25]),
        num_layers=2,
        cold_page_codec_provider=_ColdPageProvider(first_n, last_n),
    )
    if not builds:
        with pytest.raises(
            ValueError,
            match="cannot be used with this model: layer 0 has a sliding window of 128 tokens",
        ):
            make_manager(**options)
        return
    manager = make_manager(**options)
    assert manager.mixed_precision_high_layers == {1: 2}
    high_layer = manager.kv_cache_manager_py_config.layers[2]
    assert (high_layer.num_sink_tokens, high_layer.sliding_window_size) == (first_n, last_n + 1)


def test_cold_page_page_group_check_uses_the_high_layers_reuse_policy():
    """A sliding-window layer with the high layers' retention shares their page group, and is
    rejected, only under their reuse policy too."""
    manager = SimpleNamespace(tokens_per_block=128, first_n=1, last_n=127)
    layer = SimpleNamespace(
        layer_id=0, sliding_window_size=128, num_sink_tokens=1, reuse_policy=kv.ReusePolicy.REQUIRED
    )
    check = KVCacheManagerV2._check_high_layers_share_no_page_group
    check(manager, [layer], kv.ReusePolicy.SINK_REQUIRED_ONLY)
    layer.reuse_policy = kv.ReusePolicy.SINK_REQUIRED_ONLY
    with pytest.raises(ValueError, match="layer 0 has a sliding window of 128 tokens"):
        check(manager, [layer], kv.ReusePolicy.SINK_REQUIRED_ONLY)


def test_mixed_precision_warns_when_no_layer_takes_it(make_manager):
    """When every attention layer has a sliding window, neither way of asking for mixed precision
    reaches a layer; the manager says so instead of staying silent."""
    from tensorrt_llm.logger import logger

    with patch.object(logger, "warning") as warning:
        hot = make_manager(
            config=mixed_config(pool_ratio=None, max_attention_window=[64]), num_layers=2
        )
    assert hot.mixed_precision_high_layers == {}
    assert any("has no effect" in str(call.args[0]) for call in warning.call_args_list)
    with patch.object(logger, "warning") as warning:
        cold = make_manager(
            config=cold_page_config(pool_ratio=None, max_attention_window=[64]),
            num_layers=2,
            cold_page_codec_provider=_ColdPageProvider(),
        )
    assert cold.mixed_precision_high_layers == {}
    assert any("has no effect" in str(call.args[0]) for call in warning.call_args_list)


def test_cold_page_mixed_precision_rejects_a_pool_with_a_layer_that_takes_none(make_manager):
    """A layer whose KV cache another layer shares takes no mixed precision but sits in the pool of
    layers that do; dense attention would redirect its block offsets without copying its rows
    back, so the manager refuses the model and names the layer."""
    options = dict(
        num_layers=4,
        pretrained_config=type(
            "Config",
            (),
            dict(
                num_hidden_layers=4,
                num_kv_shared_layers=1,
                layer_types=["full_attention"] * 4,
            ),
        )(),
    )
    # Two layer groups: the full-attention layers (the shared source among them) and the high group.
    with pytest.raises(
        ValueError, match="takes no first/last-N mixed precision but shares KV cache pool"
    ):
        make_manager(
            config=cold_page_config(pool_ratio=[0.75, 0.25]),
            cold_page_codec_provider=_ColdPageProvider(),
            **options,
        )


def test_cold_page_window_skips_the_codec_while_estimating(make_manager):
    provider = _ColdPageProvider(first_n=0, last_n=65)
    manager = make_manager(
        config=cold_page_config(), cold_page_codec_provider=provider, is_estimating_kv_cache=True
    )
    assert manager.mixed_precision_high_layers == {0: 1}
    assert provider.calls == []


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


@pytest.mark.parametrize("drafter", ["mtp", "eagle3"])
def test_cold_page_window_appended_draft_layers(make_manager, drafter):
    """Every drafter layer appended to the target cache keeps its first/last tokens: its group's
    pages must keep the high layers' geometry for them to share the pool."""
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, MTPDecodingConfig

    spec_config = (
        MTPDecodingConfig(num_nextn_predict_layers=1)
        if drafter == "mtp"
        else Eagle3DecodingConfig(max_draft_len=1, speculative_model="draft-model")
    )
    provider = _ColdPageProvider()
    manager = make_manager(
        config=cold_page_config(pool_ratio=None),
        num_layers=2,
        spec_config=spec_config,
        cold_page_codec_provider=provider,
    )
    assert manager.pp_layers == [0, 1, 2]
    assert manager.mixed_precision_high_layers == {0: 3, 1: 4, 2: 5}
    assert manager.mixed_precision_high_block_deltas == {0: 0}
    # The codec sees the model's layers only, never a mixed-precision high layer.
    assert [
        [int(layer.layer_id) for layer in call["cache_config"].layers] for call in provider.calls
    ] == [[0, 1, 2]]


@pytest.mark.parametrize("drafter", ["mtp", "eagle3"])
def test_cold_page_window_separate_draft_manager(make_manager, drafter):
    """A separate MTP draft manager takes the cold-page window; another drafter keeps a plain cache.

    The target's pool_ratio is fitted like the mixed-precision draft manager's: dropped for the
    MTP manager with its mixed-precision high pool, collapsed to [1.0] for a plain one-group cache.
    """
    from tensorrt_llm._torch.pyexecutor._util import _fit_draft_pool_ratio
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, MTPDecodingConfig

    spec_config = (
        MTPDecodingConfig(num_nextn_predict_layers=1)
        if drafter == "mtp"
        else Eagle3DecodingConfig(max_draft_len=1, speculative_model="draft-model")
    )
    provider = _ColdPageProvider()
    config = cold_page_config()
    _fit_draft_pool_ratio(config, has_mixed_precision_high_layers=drafter == "mtp")
    assert config.pool_ratio == (None if drafter == "mtp" else [1.0])
    manager = make_manager(
        config=config,
        num_layers=1,
        layer_mask=[False, False, True],
        spec_config=spec_config,
        pretrained_config=SimpleNamespace(num_hidden_layers=2),
        is_draft=True,
        cold_page_codec_provider=provider,
    )
    assert manager.pp_layers == [2]
    assert manager.mixed_precision_high_layers == ({0: 1} if drafter == "mtp" else {})
    assert [call["is_draft"] for call in provider.calls] == [True]


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


@pytest.mark.parametrize("high_dtype", [DataType.FP8, DataType.BF16])
def test_reuse_last_n_from_low_on_sparse_managers(high_dtype):
    """A sparse manager restores an FP8 high pool from its NVFP4 pool; other high dtypes are
    rejected when the manager is built."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    def build():
        return DSACacheManagerV2(
            kv_cache_config=mixed_config(
                high_dtype, pool_ratio=None, mixed_precision_reuse_last_n_from_low=True
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=64,
            max_seq_len=1024,
            max_batch_size=2,
            mapping=Mapping(),
            dtype=DataType.NVFP4,
            sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=128),
            pretrained_config=SimpleNamespace(num_hidden_layers=1),
        )

    if high_dtype != DataType.FP8:
        with pytest.raises(NotImplementedError, match="DSACacheManagerV2"):
            build()
        return
    manager = build()
    try:
        for layer_id in manager.mixed_precision_high_layers.values():
            layer = manager.kv_cache_manager_py_config.layers[layer_id]
            assert layer.reuse_policy == kv.ReusePolicy.SINK_REQUIRED_ONLY
    finally:
        manager.shutdown()


def _nvfp4_pool(num_pool_tokens, row_width):
    """NVFP4 data and scale pools whose rows all decode differently."""
    codes = torch.arange(1, num_pool_tokens + 1, dtype=torch.uint8, device="cuda")
    data_pool = (codes | (codes << 4))[:, None].expand(-1, row_width // 2).contiguous()
    scale_pool = torch.full(
        (num_pool_tokens, row_width // 16), 2.0, dtype=torch.float8_e4m3fn, device="cuda"
    )
    return data_pool, scale_pool


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="NVFP4 MLA gather requires Blackwell",
)
@pytest.mark.parametrize(
    "backend,residual_dim", [("dsa", 64), ("deepseek_v4", 0), ("deepseek_v4", 64)]
)
def test_sparse_managers_decode_the_window_from_the_nvfp4_pool(backend, residual_dim):
    """A sparse manager decodes a fresh window from its NVFP4 pool into the FP8 high rows as the
    read decodes it; a compressed row needs all of its tokens, and other high rows stay."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import (
        _get_nvfp4_mla_kv_cache_amax,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2

    head_dim = 576 if backend == "dsa" else 512
    row_bytes = (head_dim + residual_dim) // 2
    data_pool, scale_pool = _nvfp4_pool(12, head_dim + residual_dim)
    pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
    pointers[0, 0] = torch.tensor([data_pool.data_ptr(), scale_pool.data_ptr()])
    # Two rows a page: DSA rows are tokens, DeepSeek-V4 rows compress 4 tokens of 8-token blocks.
    # The request's own pages are 2, 0, 3 and its high pages 0, 2, 3 (page 0 holds its sink).
    own_pages, high_pages, high_layer = [2, 0, 3], [0, 2, 3], 1
    high = torch.full((4, 1, 2, 1, head_dim), 448.0, dtype=torch.float8_e4m3fn, device="cuda")

    def fill(layer, request_ids, out, role=None):
        out[0, :3] = high_pages if layer == high_layer else own_pages

    manager = SimpleNamespace(
        tokens_per_block=2 if backend == "dsa" else 8,
        max_blocks_per_seq=3,
        head_dim=head_dim,
        mla_kv_cache_residual_dim=residual_dim,
        nvfp4_residual_dim=residual_dim,
        mixed_precision_high_layers={0: high_layer},
        _fill_page_table=fill,
        _get_mixed_precision_buffers=lambda *args: high,
        get_pool_block_indices=lambda *args, **kwargs: torch.tensor([own_pages]),
        get_primary_pool_page_index_params=lambda layer: (1, 0),
        get_unique_primary_pool=lambda: torch.empty(6, 1),
        kv_cache_pool_pointers=pointers,
        kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        pp_layers=[0],
        _compress_ratios={0: 4},
        compressed_block_sizes={0: 2},
        _mixed_precision_layer_page=lambda local: (5, None),
        get_compress_pool_buffers=lambda ratio: (data_pool, scale_pool),
        # This DeepSeek-V4 layer's rows start 2 rows into its ratio's pool.
        get_buffers=lambda *args: SimpleNamespace(data_ptr=lambda: 4096 + 2 * row_bytes),
        compress_pool_ptrs={4: 4096},
        impl=SimpleNamespace(
            get_buffer_group_id=lambda layer, role: layer,
            get_page_index_converter=lambda layer, role: SimpleNamespace(scale=1),
        ),
    )
    manager._distinct_page_tables = partial(KVCacheManagerV2._distinct_page_tables, manager)
    manager._fill_page_tables = partial(KVCacheManagerV2._fill_page_tables, manager)
    # The launch reads the per-layer plan, built here from the stubs above.
    restore = DSACacheManagerV2 if backend == "dsa" else DeepseekV4CacheManager
    manager._nvfp4_restore_plan = restore._nvfp4_restore_plan.func(manager)
    if backend == "dsa":
        DSACacheManagerV2._decode_mixed_precision_window(manager, 7, 2, 5)
        # Tokens 2, 3 -> pool rows 0, 1; token 4 -> pool row 6.
        pool_rows = [0, 1, 6]
    else:
        # Tokens 8-21 cover the whole compressed rows 2-4.
        DeepseekV4CacheManager._decode_mixed_precision_window(manager, 7, 8, 22)
        # Rows 2, 3 -> pool rows 2, 3; row 4 -> pool row 2 + 3 * 2 = 8.
        pool_rows = [2, 3, 8]
    scale = torch.tensor([_get_nvfp4_mla_kv_cache_amax() / (448.0 * 6.0)], device="cuda")
    read = torch.empty((3, 1, head_dim), dtype=torch.float8_e4m3fn, device="cuda")
    torch.ops.trtllm.nvfp4_mla_kv_cache_gather_direct(
        data_pool,
        scale_pool,
        torch.tensor(pool_rows, dtype=torch.int32, device="cuda").view(-1, 1),
        read,
        scale,
        residual_dim,
        data_pool.shape[0],
    )
    # Rows 2, 3 sit in high page 2 and row 4 in high page 3; the rest keep the sentinel.
    expected = torch.full_like(high, 448.0).view(torch.uint8)
    expected[[2, 2, 3], 0, [0, 1, 0]] = read.view(torch.uint8)
    assert torch.equal(high.view(torch.uint8), expected)


def test_deepseek_v4_cold_page_window_high_layers_share_the_compress_pools():
    """With the cold-page window a DeepSeek-V4 high layer copies its COMPRESS page into that
    ratio's pool, where the backend reads it by row."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

    ratios = [1, 4, 128]
    manager = DeepseekV4CacheManager(
        kv_cache_config=cold_page_config(pool_ratio=None),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=3,
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=4096,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.FP8,
        vocab_size=1000,
        max_input_len=4096,
        max_num_tokens=8192,
        sparse_attn_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=ratios
        ),
        pretrained_config=SimpleNamespace(num_hidden_layers=3),
        cold_page_codec_provider=_ColdPageProvider(),
    )
    try:
        # Only the compressed layers (ratios 4 and 128) get a high layer; the manager checked at
        # init that each sits in its ratio's COMPRESS pool at the compress layer's slot.
        assert sorted(manager.mixed_precision_high_layers) == [1, 2]
        assert sorted(manager.mixed_precision_high_layer_offsets) == [1, 2]
        assert set(manager.mixed_precision_high_block_deltas.values()) == {0}
        layers = manager.kv_cache_manager_py_config.layers
        for local, high_layer in manager.mixed_precision_high_layers.items():
            compress_layer = manager._layer_attn_to_layer_id[
                local, DeepseekV4AttentionType.COMPRESS
            ]
            assert [b.size for b in layers[high_layer].buffers] == [
                b.size for b in layers[compress_layer].buffers
            ]
    finally:
        manager.shutdown()


def test_deepseek_v4_cold_page_window_redirects_rows_into_the_high_layer():
    """The redirected compressed rows read the high layer's pages through the COMPRESS pool; a
    second layer of a ratio, and its high layer, sit at their own offset inside that pool's pages."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.mixed_precision_high_rows import (
        mixed_precision_high_row_indices,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

    ratios = [1, 4, 128, 4]
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=128, window_size=128, compress_ratios=ratios
    )
    manager = DeepseekV4CacheManager(
        kv_cache_config=cold_page_config(pool_ratio=None, enable_block_reuse=False),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=4,
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=4096,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.FP8,
        vocab_size=1000,
        max_input_len=4096,
        max_num_tokens=8192,
        sparse_attn_config=sparse_config,
        pretrained_config=SimpleNamespace(num_hidden_layers=4),
        # The high layer holds the first page and a 128-token window: the last two pages here.
        cold_page_codec_provider=_ColdPageProvider(first_n=128, last_n=127),
    )
    try:
        request_ids, cached = [0, 1], [600, 700]
        assert manager.add_dummy_requests(request_ids, [n + 1 for n in cached], is_gen=True)
        metadata = DeepseekV4TrtllmAttentionMetadata(
            max_num_requests=2,
            max_num_tokens=2,
            kv_cache_manager=manager,
            sparse_attention_config=sparse_config,
            seq_lens=torch.tensor([1, 1], dtype=torch.int),
            num_contexts=0,
            request_ids=request_ids,
            prompt_lens=cached,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
            mapping=Mapping(),
        )
        metadata.prepare()
        # The backend reads every layer of a ratio from that ratio's COMPRESS pool base, so the
        # rows of the second ratio-4 layer (3), and of its high layer, carry a page offset.
        pools = {
            ratio: manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
            for layer_idx, ratio in ((1, 4), (2, 128))
        }
        assert [pools[ratio].data_ptr() for ratio in pools] == [
            manager.compress_pool_ptrs[ratio] for ratio in pools
        ]
        high_offsets = manager.mixed_precision_high_layer_offsets
        assert high_offsets[manager.layer_offsets[3]] > high_offsets[manager.layer_offsets[1]]
        for layer_idx, ratio in ((1, 4), (2, 128), (3, 4)):
            local = manager.layer_offsets[layer_idx]
            rows_per_page = manager.compressed_block_sizes[layer_idx]
            high_table = metadata.mixed_precision_high_block_table[ratio].cpu()
            # Compressed rows of request 1: pages 0 (sink), 4 and 5 (window) are held, 1-3 are not.
            held_pages = [page for page, entry in enumerate(high_table[1].tolist()) if entry >= 0]
            assert held_pages == [0, 4, 5], held_pages
            positions = torch.arange(6 * rows_per_page, dtype=torch.int32, device="cuda")
            pool_rows = torch.full_like(positions, -7)
            rows = mixed_precision_high_row_indices(
                metadata,
                SimpleNamespace(layer_idx=layer_idx, compress_ratio=ratio),
                positions[None],
                pool_rows[None],
                torch.ones(1, dtype=torch.int32, device="cuda"),
            )[0]
            # Mark every held row through the high-pool view: value = row position, shifted per
            # layer so the two ratio-4 layers' pages tell apart.
            high = manager._get_mixed_precision_buffers(local, "NHD", "high")
            high = high.view(high.shape[0], rows_per_page, -1).view(torch.uint8)
            for position in range(6 * rows_per_page):
                page, row = divmod(position, rows_per_page)
                if page in held_pages:
                    high[int(high_table[1, page]), row].fill_((position + 97 * layer_idx) % 251)
            # The ratio's COMPRESS pool, read at the redirected rows, shows the markers.
            pool = pools[ratio].reshape(-1, pools[ratio].shape[-1]).view(torch.uint8)
            for position in range(6 * rows_per_page):
                page = position // rows_per_page
                row = int(rows[position])
                if page in held_pages:
                    assert torch.all(pool[row] == (position + 97 * layer_idx) % 251), (
                        layer_idx,
                        position,
                        row,
                    )
                else:
                    assert row == -7, (layer_idx, position, row)
        # Resuming a prefix without its window copies the window pages (4 and 5 of request 1,
        # after its sink page) from the COMPRESS pages the high layers mirror.
        for layer_idx in (1, 2):
            own = manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS).view(torch.uint8)
            own.copy_(torch.randint_like(own, 0, 120))
        cache = SimpleNamespace(
            num_committed_tokens=cached[1], cuda_stream=torch.cuda.current_stream().cuda_stream
        )
        manager._restore_last_n_window(1, cache)
        torch.cuda.synchronize()
        for layer_idx, ratio in ((1, 4), (2, 128)):
            own = manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS).view(torch.uint8)
            high = manager._get_mixed_precision_buffers(
                manager.layer_offsets[layer_idx], "NHD", "high"
            ).view(torch.uint8)
            own_pages = metadata.compress_block_tables[ratio][1].tolist()
            high_pages = metadata.mixed_precision_high_block_table[ratio][1].tolist()
            for page in (4, 5):
                assert torch.equal(
                    high[high_pages[page]].flatten(), own[own_pages[page]].flatten()
                ), (layer_idx, page)
    finally:
        manager.shutdown()


def test_dsa_cold_page_window_high_layers_share_the_pool():
    """On a plain DSA pool a mixed-precision high layer mirrors its layer's buffers, in one pool."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    manager = DSACacheManagerV2(
        kv_cache_config=cold_page_config(pool_ratio=None),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=64,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.FP8,
        sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=128),
        pretrained_config=SimpleNamespace(num_hidden_layers=2),
        cold_page_codec_provider=_ColdPageProvider(),
    )
    try:
        assert manager.mixed_precision_high_layers == {0: 2, 1: 3}
        layers = manager.kv_cache_manager_py_config.layers
        for local, high_layer in manager.mixed_precision_high_layers.items():
            assert [(b.role, b.size) for b in layers[high_layer].buffers] == [
                (b.role, b.size) for b in layers[local].buffers
            ]
        assert manager.get_unique_primary_pool().shape[1] == 2
        # Checked at init: each high layer sits in the K pool at its layer's slot.
        assert manager.mixed_precision_high_layer_offsets == {0: 0, 1: 1}
        assert manager.mixed_precision_high_block_deltas == {0: 0}
    finally:
        manager.shutdown()
