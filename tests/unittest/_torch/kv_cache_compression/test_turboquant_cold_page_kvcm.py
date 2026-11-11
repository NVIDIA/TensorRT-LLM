# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TurboQuant cold pages through a real KVCacheManagerV2.

KVCM encodes a Page when it migrates from the GPU to the Host or Disk tier and decodes it on the
way back; Host <-> Disk migration copies the encoded blob unchanged. Each test sends Pages cold
with a pressure request that needs every GPU slot, brings them back, and compares them with the
codec's own round trip of the same Pages, which they must match bit for bit.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _torch.attention.sparse.deepseek_v4 import test_deepseek_v4_cache_manager as dsv4
from transformers import PretrainedConfig
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.turboquant_quantization import (
    TurboQuantColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal import kv_cache_compression as native_kvcc
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GPU_LEVEL,
    AttentionLayerConfig,
    BufferConfig,
    CacheTier,
    _introspection,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TurboQuant needs a GPU")

TOKENS_PER_BLOCK = 32
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROPE_DIM = 32  # The partial-rotary model below rotates the leading quarter of every K head.
NUM_PAGES = 5  # Odd: the kernels may pack two 64-row Pages per tile and mask the spare half.
COLD_LEVEL = 1  # The first tier below the GPU: Host, or Disk when there is no Host tier.
ROLES = ("key", "value")
COMPRESS = DeepseekV4AttentionType.COMPRESS
INDEXER = DeepseekV4AttentionType.INDEXER_COMPRESS


def _provider(bits=4, mode="mse", skip_rope=False):
    pretrained_config = PretrainedConfig(partial_rotary_factor=ROPE_DIM / HEAD_DIM)
    pretrained_config.model_type = "qwen3_5"
    config = ColdPageQuantizationCompressionConfig(
        quant="turboquant",
        turboquant_bits=bits,
        turboquant_mode=mode,
        skip_rope_quantization=skip_rope,
    )
    return TurboQuantColdPageQuantizationCompression(config, pretrained_config=pretrained_config)


def _manager(
    provider,
    *,
    num_layers=NUM_LAYERS,
    gpu_bytes=2 << 20,
    host_bytes=16 << 20,
    disk_path=None,
    reuse=False,
    **kwargs,
):
    """A BF16 KVCM; one 2 MiB GPU granule holds 32 Pages of two layers."""
    kv_cache_config = KvCacheConfig(
        max_gpu_total_bytes=gpu_bytes,
        enable_block_reuse=reuse,
        host_cache_size=host_bytes,
        disk_cache_size=None if disk_path is None else 16 << 20,
        disk_cache_path=disk_path,
    )
    return KVCacheManagerV2(
        kv_cache_config,
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=8192,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=1 << 20,
        cold_page_codec_provider=provider,
        **kwargs,
    )


def _request(request_id, num_tokens, first_token=0):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(first_token, first_token + num_tokens)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _page_ids(manager, request, layer):
    ids = manager.get_batch_cache_indices([request.py_request_id], layer)[0]
    return torch.tensor(ids, device="cuda")


def _read(manager, request):
    """KV of a request's Pages as [Pages, layers, K/V, heads, tokens, head_dim]."""
    return torch.stack(
        [
            manager.get_buffers(layer, kv_layout="HND")[_page_ids(manager, request, layer)]
            for layer in range(manager.num_local_layers)
        ],
        1,
    )


def _prefill(manager, request_id=0, num_tokens=NUM_PAGES * TOKENS_PER_BLOCK):
    """Run a context request whose Pages hold random K/V with a few large channels."""
    request = _request(request_id, num_tokens)
    assert manager.prepare_context(request)
    assert manager.resize_context(request, num_tokens)
    generator = torch.Generator(device="cuda").manual_seed(request_id)
    kv = torch.randn(
        (-(-num_tokens // TOKENS_PER_BLOCK), manager.num_local_layers, 2, NUM_KV_HEADS)
        + (TOKENS_PER_BLOCK, HEAD_DIM),
        generator=generator,
        device="cuda",
    )
    kv[..., [5, 77, 100]] *= torch.tensor([20.0, 30.0, 8.0], device="cuda")
    kv = kv.to(torch.bfloat16)
    for layer in range(manager.num_local_layers):
        pool = manager.get_buffers(layer, kv_layout="HND")
        pool[_page_ids(manager, request, layer)] = kv[:, layer]
    request.context_current_position = num_tokens
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    manager.update_context_resources(batch)
    torch.cuda.synchronize()
    return request, kv


def _displace(manager, pressure_id, suspend=False):
    """Allocate every GPU slot to a new request, migrating all other unlocked Pages down a tier.

    The request is then suspended, keeping its Pages, or freed. Returns its number of Pages.
    """
    num_slots = manager.impl.get_storage_statistics(GPU_LEVEL)[0].total
    pressure = _request(pressure_id, num_slots * TOKENS_PER_BLOCK, first_token=1 << 19)
    assert manager.prepare_context(pressure)
    assert manager.resize_context(pressure, pressure.prompt_len)
    torch.cuda.synchronize()
    if suspend:
        manager.suspend_request(pressure)
    else:
        manager.free_resources(pressure)
    return num_slots


def _levels(manager, request):
    """Number of the request's Pages on each cache tier."""
    return _introspection.active_page_stats(manager.kv_cache_map[request.py_request_id])[0]


def _offload(manager, request, pressure_id=1):
    manager.suspend_request(request)
    _displace(manager, pressure_id)
    return _levels(manager, request)


def _restore(manager, request):
    assert manager.resume_request(request)
    torch.cuda.synchronize()
    assert not any(_levels(manager, request)[COLD_LEVEL:])


@contextmanager
def _codec_spies(provider):
    """Spy on the provider's encode and decode, which the native codec looks up per call."""
    with (
        patch.object(provider, "encode_cold_pages", wraps=provider.encode_cold_pages) as encode,
        patch.object(provider, "decode_cold_pages", wraps=provider.decode_cold_pages) as decode,
    ):
        yield encode, decode


def _num_pages(spy):
    return sum(call.args[4] for call in spy.call_args_list)


def _codec_round_trip(provider, kv):
    """Encode and decode [Pages, layers, ...] KV with the codec alone, into device memory."""
    num_pages, num_layers = kv.shape[:2]
    hot = torch.cat([kv, torch.empty_like(kv)])
    buffer_bytes = hot[0, 0, 0].nbytes
    cache_config = SimpleNamespace(
        tokens_per_block=TOKENS_PER_BLOCK,
        layers=[
            AttentionLayerConfig(
                layer_id=layer,
                buffers=[BufferConfig(role=role, size=buffer_bytes) for role in ROLES],
            )
            for layer in range(num_layers)
        ],
    )
    state = provider.build_codec_state(
        cache_config,
        runtime_dtype=DataType.BF16,
        pp_layers=list(range(num_layers)),
        num_kv_heads_per_layer=[NUM_KV_HEADS] * num_layers,
        head_dim_per_layer=[HEAD_DIM] * num_layers,
    )
    lifecycle = SimpleNamespace(
        layers={
            layer: {
                role: SimpleNamespace(
                    raw_base=hot[0, layer, kv_index].data_ptr(),
                    raw_slot_bytes=hot[0].nbytes,
                    raw_bytes=buffer_bytes,
                )
                for kv_index, role in enumerate(ROLES)
            }
            for layer in range(num_layers)
        }
    )
    (properties,) = provider.configure(state, [lifecycle])
    cold = torch.empty(num_pages * properties.cold_page_bytes, dtype=torch.uint8, device="cuda")
    pages = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    encode_pairs = torch.stack([pages, pages], 1)
    decode_pairs = torch.stack([pages + num_pages, pages], 1)
    stream = torch.cuda.current_stream().cuda_stream
    provider.encode_cold_pages(
        state, 0, cold.data_ptr(), encode_pairs.data_ptr(), num_pages, stream
    )
    provider.decode_cold_pages(
        state, 0, cold.data_ptr(), decode_pairs.data_ptr(), num_pages, stream
    )
    torch.cuda.synchronize()
    return hot[num_pages:]


def _by_token(kv):
    """[Pages, layers, K/V, heads, tokens, dim] -> [layers, K/V, heads, Pages * tokens, dim]."""
    return kv.permute(1, 2, 3, 0, 4, 5).flatten(3, 4)


def _nmse(restored, original):
    """Mean over vectors of ||restored - original||^2 / ||original||^2."""
    x, y = original.float(), restored.float()
    return float(((y - x).square().sum(-1) / x.square().sum(-1)).mean())


def _assert_restored(provider, restored, original, max_nmse, rope_dim=0, num_tokens=None):
    """Restored KV is the codec's round trip: lossy within max_nmse, the K RoPE span exact."""
    expected = _by_token(_codec_round_trip(provider, original))[..., :num_tokens, :]
    restored, original = (_by_token(kv)[..., :num_tokens, :] for kv in (restored, original))
    assert torch.equal(restored, expected)
    assert torch.equal(restored[:, 0, ..., :rope_dim], original[:, 0, ..., :rope_dim])
    for kv_index, start in enumerate((rope_dim, 0)):
        error = _nmse(restored[:, kv_index, ..., start:], original[:, kv_index, ..., start:])
        assert 0 < error < max_nmse, f"{ROLES[kv_index]} nMSE {error}"


@pytest.mark.parametrize(
    ("bits", "mode", "skip_rope", "max_nmse"),
    [
        (4, "mse", True, 0.02),
        (3, "mse", False, 0.05),
        (3.5, "mse", False, 0.03),
        (4, "prod", False, 0.08),
    ],
    ids=["mse4-skip-rope", "mse3", "mse3.5", "prod4"],
)
def test_host_tier_round_trip(bits, mode, skip_rope, max_nmse):
    provider = _provider(bits, mode, skip_rope)
    manager = _manager(provider)
    try:
        request, kv = _prefill(manager)
        with _codec_spies(provider) as (encode, decode):
            assert _offload(manager, request) == [0, NUM_PAGES]
            _restore(manager, request)
        assert _num_pages(encode) == _num_pages(decode) == NUM_PAGES
        hot_bytes, cold_bytes = (
            sum(manager.impl.get_storage_statistics(level)[0].slot_sizes)
            for level in (GPU_LEVEL, COLD_LEVEL)
        )
        assert cold_bytes < hot_bytes / 2
        _assert_restored(provider, _read(manager, request), kv, max_nmse, ROPE_DIM * skip_rope)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("host_bytes", [0, 2 << 20], ids=["disk", "host-then-disk"])
def test_disk_tier_round_trip(tmp_path, host_bytes):
    """Hot -> Disk encodes into pinned staging; Host -> Disk moves encoded Pages unchanged."""
    provider = _provider()
    # The 2 MiB Host tier holds fewer cold Pages than the 16 MiB GPU tier has slots.
    manager = _manager(provider, gpu_bytes=16 << 20, host_bytes=host_bytes, disk_path=str(tmp_path))
    try:
        request, kv = _prefill(manager)
        with _codec_spies(provider) as (encode, decode):
            manager.suspend_request(request)
            num_slots = _displace(manager, pressure_id=1, suspend=bool(host_bytes))
            if host_bytes:
                assert _levels(manager, request) == [0, NUM_PAGES, 0]
                # The suspended pressure request's Pages overflow the Host tier, which moves the
                # older Pages on to Disk.
                _displace(manager, pressure_id=2)
            assert _levels(manager, request)[-1] == NUM_PAGES
            # Every Page is encoded once: the Host -> Disk move does not touch the codec.
            assert _num_pages(encode) == NUM_PAGES + (num_slots if host_bytes else 0)
            _restore(manager, request)
        assert _num_pages(decode) == NUM_PAGES
        _assert_restored(provider, _read(manager, request), kv, 0.02)
    finally:
        manager.shutdown()


def test_block_reuse_restores_cold_pages():
    """A new request reusing a prompt decodes its committed blocks back from the Host tier."""
    provider = _provider()
    manager = _manager(provider, reuse=True)
    num_tokens = NUM_PAGES * TOKENS_PER_BLOCK + 17  # The last block is partial.
    try:
        first, kv = _prefill(manager, num_tokens=num_tokens)
        manager.free_resources(first)  # Its committed blocks stay reusable.
        with _codec_spies(provider) as (encode, decode):
            _displace(manager, pressure_id=1)
            host = manager.impl.get_storage_statistics(COLD_LEVEL)[0]
            assert host.total - host.free == NUM_PAGES + 1
            second = _request(2, num_tokens)
            assert manager.prepare_context(second)
            # Everything but the last prompt token is reused, partial block included.
            assert second.context_current_position == num_tokens - 1
            assert manager.resize_context(second, second.context_remaining_length)
            torch.cuda.synchronize()
        assert _num_pages(encode) == _num_pages(decode) == NUM_PAGES + 1
        _assert_restored(provider, _read(manager, second), kv, 0.02, num_tokens=num_tokens - 1)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("bits", [3, 4])
def test_repeated_migration_keeps_the_error_of_one_round_trip(bits):
    provider = _provider(bits)
    manager = _manager(provider)
    try:
        request, kv = _prefill(manager)
        errors = []
        for trip in range(3):
            assert _offload(manager, request, pressure_id=trip + 1) == [0, NUM_PAGES]
            _restore(manager, request)
            errors.append(_nmse(_read(manager, request), kv))
        assert errors[1:] == pytest.approx(errors[:1] * 2, rel=1e-3)
    finally:
        manager.shutdown()


def test_target_and_draft_managers_build_independent_codec_states():
    provider = _provider(skip_rope=True)
    with (
        patch.object(
            provider, "create_cold_page_codec", wraps=provider.create_cold_page_codec
        ) as create_codec,
        patch.object(
            native_kvcc,
            "create_python_cold_page_codec",
            wraps=native_kvcc.create_python_cold_page_codec,
        ) as create_native_codec,
    ):
        target = _manager(provider)
        draft = _manager(provider, num_layers=1, is_draft=True)
    try:
        assert [call.kwargs["is_draft"] for call in create_codec.call_args_list] == [False, True]
        target_state, draft_state = (call.args[1] for call in create_native_codec.call_args_list)
        assert (target_state.layer_ids, draft_state.layer_ids) == ((0, 1), (0,))
        # The target keeps K's RoPE span as is; draft K vectors are quantized whole.
        for state, key_start in ((target_state, ROPE_DIM), (draft_state, 0)):
            key, value = state.layer_layouts[0].buffers
            assert (key.role, key.quantized_range_start) == ("key", key_start)
            assert (value.role, value.quantized_range_start) == ("value", 0)
        # Configuring the draft codec left the target's state alone.
        request, kv = _prefill(target)
        assert _offload(target, request) == [0, NUM_PAGES]
        _restore(target, request)
        _assert_restored(provider, _read(target, request), kv, 0.02, ROPE_DIM)
    finally:
        draft.shutdown()
        target.shutdown()


@skip_pre_blackwell
def test_deepseek_v4_csa_hca_lifecycle_migrates_through_host():
    """CSA rows quantize 448 NoPE values; RoPE, the indexer cache, and HCA rows stay as is."""
    suite = dsv4.TestDeepseekV4CacheManager()
    tokens_per_block = suite.tokens_per_block
    prompt_len, pressure_len = 64 * tokens_per_block, 65 * tokens_per_block
    compress_ratios = [4, 128]
    provider = TurboQuantColdPageQuantizationCompression(
        ColdPageQuantizationCompressionConfig(quant="turboquant"),
        pretrained_config=SimpleNamespace(model_type="deepseek_v4"),
    )
    with (
        patch.object(
            native_kvcc,
            "create_python_cold_page_codec",
            wraps=native_kvcc.create_python_cold_page_codec,
        ) as create_codec,
        _codec_spies(provider) as (encode, decode),
    ):
        manager, sparse_attn_config = suite._create_deepseek_v4_cache_manager(
            tokens_per_block=tokens_per_block,
            max_batch_size=1,
            max_seq_len=pressure_len,
            compress_ratios=compress_ratios,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            indexer_k_dtype="fp4",
            enable_swa_scratch_reuse=False,
            cold_page_codec_provider=provider,
            host_cache_size=64 << 20,
        )
        try:
            # One codec owns the CSA and HCA layers that share a lifecycle.
            assert create_codec.call_count == 1
            state = create_codec.call_args.args[1]
            csa, hca = (manager._layer_attn_to_layer_id[layer, COMPRESS] for layer in (0, 1))
            assert set(state.layer_ids) == {csa, hca}
            quantized = {
                (layer_id, buffer.role): buffer.quantized_range_elements
                for layer_id in (csa, hca)
                for buffer in state.layer_layouts[layer_id].buffers
            }
            assert quantized == {
                (csa, COMPRESS.role): 448,
                (csa, INDEXER.role): 0,
                (hca, COMPRESS.role): 0,
            }
            rows = tokens_per_block // 4
            (metadata,) = state.lifecycle_metadata
            assert metadata.cold_page_bytes == (
                rows * 448 // 2  # 4-bit NoPE codes
                + rows * 2  # one BF16 scale per row
                + rows * 64 * 2  # BF16 RoPE
                + rows * (suite.index_head_dim // 2 + suite.index_head_dim // 32)  # FP4 indexer
                + tokens_per_block // 128 * suite.head_dim * 2  # BF16 HCA row
            )

            first = suite._create_request(request_id=0, prompt_len=prompt_len)
            expected = suite._create_random_cache(
                seq_len=prompt_len,
                head_dim=suite.head_dim,
                sparse_attn_config=sparse_attn_config,
                dtype=torch.bfloat16,
                compressor_dtype=torch.float32,
            )
            expected_csa = expected[0, COMPRESS][0]
            expected_csa[:, [5, 300]] *= 20
            assert manager.prepare_context(first)
            assert manager.resize_context(first, first.context_chunk_size)
            suite._write_request_prefill(first, prompt_len, manager, expected)
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [first]
            first.context_current_position = prompt_len
            first.add_new_token(prompt_len, 0)
            manager.update_context_resources(batch)
            manager.update_resources(batch)
            torch.cuda.synchronize()

            manager.suspend_request(first)
            # Shrink the GPU tier to its floors so that the pressure request displaces the first.
            gpu_quota = manager.impl.get_quota(GPU_LEVEL)
            grain = _introspection.tier_allocation_granularity(CacheTier.GPU_MEM, gpu_quota)
            while manager.impl.resize(GPU_LEVEL, gpu_quota - grain):
                gpu_quota -= grain
            pressure = suite._create_request(request_id=1, prompt_len=pressure_len)
            assert manager.prepare_context(pressure)
            assert manager.resize_context(pressure, pressure.context_chunk_size)
            torch.cuda.synchronize()
            assert _levels(manager, first)[COLD_LEVEL] > 0
            encoded = _num_pages(encode)
            assert encoded > 0 and all(call.args[1] == 0 for call in encode.call_args_list)

            manager.free_resources(pressure)
            assert manager.resume_request(first)
            torch.cuda.synchronize()
            assert _levels(manager, first)[COLD_LEVEL] == 0
            assert _num_pages(decode) == encoded
            actual = suite._read_request(first, prompt_len, manager, compress_ratios)
        finally:
            manager.shutdown()

    actual_csa = actual[0, COMPRESS][0]
    nope, restored_nope = expected_csa[:, :448], actual_csa[:, :448]
    # Exactly the rows of the Pages that went through the Host tier changed.
    migrated = (restored_nope != nope).any(-1)
    assert int(migrated.sum()) == encoded * rows
    assert _nmse(restored_nope[migrated], nope[migrated]) < 0.02
    assert torch.equal(actual_csa[:, 448:], expected_csa[:, 448:])
    for want, got in zip(expected[0, INDEXER], actual[0, INDEXER]):
        assert torch.equal(got, want)
    assert torch.equal(actual[1, COMPRESS][0], expected[1, COMPRESS][0])
