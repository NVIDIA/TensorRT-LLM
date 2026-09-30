# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context metadata preserves model IDs without accessing omitted decoder SWA."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [1, 128, 129, 1024, 8192])
@torch.inference_mode()
def test_context_metadata_omits_decoder_swa_and_preserves_boundary_global(length):
    layout = CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20,
        (2, 8, 14, 20),
        (2, 8, 14, 20),
        candidate_source_layer_id=20,
    )
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=512 << 20, enable_block_reuse=False, dtype="fp8"),
        CacheType.SELFKONLY,
        layout=layout,
        context_swa_layer_limit=20,
        num_layers=40,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=8320,
        max_batch_size=1,
        max_input_len=8192,
        max_num_tokens=8192,
        dtype=DataType.BF16,
        vocab_size=1024,
    )
    request = LlmRequest(
        request_id=71,
        max_new_tokens=1,
        input_tokens=[i % 1024 for i in range(length)],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    try:
        assert manager.prepare_context(request)
        assert manager.resize_context(request, length)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=8192, kv_cache_manager=manager
        )
        metadata.request_ids = [71]
        metadata.seq_lens = torch.tensor([length], dtype=torch.int32)
        metadata.num_contexts = 1
        metadata.prompt_lens = [length]
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
        metadata.prepare()
        initial_pointers = {}
        # Exercise normal publication and direct converter fallback on the same
        # live metadata, then its device-side address refresh.
        for direct in (False, True):
            if direct:
                metadata.prepare_csa2()
            metadata.on_update_kv_lens()
            metadata._ensure_swa_slots()
            assert set(metadata.csa2_swa_indices) == set(range(40))
            assert set(metadata._csa2_swa_page_tables) == set(range(20))
            assert set(metadata.csa2_global_page_tables) == {2, 8, 14, 20}
            positions = torch.arange(length, dtype=torch.int64, device="cuda")
            for layer in range(40):
                reads = metadata.csa2_swa_indices[layer]
                writes = metadata.csa2_swa_write_slots[layer]
                if direct:
                    assert initial_pointers[layer] == (reads.data_ptr(), writes.data_ptr())
                else:
                    initial_pointers[layer] = (reads.data_ptr(), writes.data_ptr())
                if layer >= 20:
                    assert torch.all(reads == -1)
                    assert torch.all(writes == -1)
                    assert reads.untyped_storage().nbytes() == 8
                else:
                    pages = manager.get_cache_indices(71, layer, CSA2CacheRole.SWA)
                    expected = torch.tensor(pages, device="cuda")[positions // 128] * 128
                    expected += positions % 128
                    torch.testing.assert_close(writes, expected)
                    torch.testing.assert_close(reads[:, -1], expected)
                ratio = layout.compress_ratios[layer]
                expected_visible = (
                    (positions + 1) // ratio if ratio else torch.zeros_like(positions)
                )
                torch.testing.assert_close(metadata.csa2_visible_lengths[layer], expected_visible)
            assert torch.all(metadata.csa2_main_write_slots[20] >= 0)
            assert manager.get_main_buffer(20).data_ptr() == manager.get_main_buffer(39).data_ptr()
        metadata.num_contexts = 0
        with pytest.raises(ValueError, match="context-only cache cannot prepare generation"):
            metadata.prepare_csa2()
    finally:
        manager.free_resources(request)
        manager.shutdown()
