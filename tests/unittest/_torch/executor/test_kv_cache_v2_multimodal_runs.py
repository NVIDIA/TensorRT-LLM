# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import gen_multi_modal_tokens


def test_augment_tokens_for_block_reuse_uses_exact_multimodal_runs():
    vocab_size = 1000
    hash_ints = [1, 2, 3, 4, 5, 6, 7, 8]
    digest = b"".join(v.to_bytes(4, "big", signed=True) for v in hash_ints)
    mm_tokens = gen_multi_modal_tokens(vocab_size, digest, 4)

    manager = SimpleNamespace(vocab_size=vocab_size)
    req = SimpleNamespace(
        multimodal_hashes=[hash_ints],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_seqlen=[0, 2],
        multimodal_run_positions=[1, 8],
        multimodal_run_lengths=[2, 2],
    )

    tokens = list(range(12))
    augmented = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req)
    assert augmented[1:3] == mm_tokens[0:2]
    assert augmented[8:10] == mm_tokens[2:4]
    assert augmented[3:8] == tokens[3:8]

    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=7, end=10)
    assert sliced == [tokens[7], mm_tokens[2], mm_tokens[3]]
