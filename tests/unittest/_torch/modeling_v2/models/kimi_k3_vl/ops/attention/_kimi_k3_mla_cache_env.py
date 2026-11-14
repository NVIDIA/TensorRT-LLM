# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Kimi K3 latent cache for the MLA op tests: one MLA layer's bf16 pool (one latent head of 576, page 64) from a real
KV cache manager, every request's rows written through the manager's own block ids, and read back the same way."""

from __future__ import annotations

from typing import Dict, List

import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures, KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

PAGE = 64
LATENT, ROPE = 512, 64
ROW = LATENT + ROPE


class LatentCache:
    def __init__(self, num_requests: int, max_seq_len: int) -> None:
        self.num_requests = num_requests
        self.manager = KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=num_requests * max_seq_len, enable_block_reuse=False, host_cache_size=0
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=ROW,
            tokens_per_block=PAGE,
            max_seq_len=max_seq_len,
            max_batch_size=num_requests,
            mapping=Mapping(),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        assert self.manager.add_dummy_requests(
            list(range(num_requests)), [max_seq_len] * num_requests
        )
        self.rows: Dict[int, torch.Tensor] = {}

    def pool(self) -> torch.Tensor:
        return self.manager.get_buffers(0)

    def blocks(self, request_id: int) -> List[int]:
        return list(self.manager.get_batch_cache_indices([request_id])[0])

    def fill(self, lengths: List[int], seed: int) -> None:
        """Random rows ``[0, lengths[r])`` for request r, written page by page through its block ids."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        pool = self.pool()
        for r, n in enumerate(lengths):
            rows = (torch.randn(n, ROW, generator=gen, device="cuda") * 0.5).bfloat16()
            blocks = self.blocks(r)
            for p in range(0, n, PAGE):
                pool[blocks[p // PAGE], 0, : min(PAGE, n - p), 0] = rows[p : p + PAGE]
            self.rows[r] = rows

    def metadata(
        self, seq_lens, cached, num_contexts, prompt_lens, chunked_prefill=False, chunk_size=16
    ):
        md = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(len(seq_lens))),
            max_num_requests=self.num_requests,
            num_contexts=num_contexts,
            prompt_lens=list(prompt_lens),
            max_num_tokens=8192,
            kv_cache_manager=self.manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
            mapping=Mapping(),
            enable_context_mla_with_cached_kv=True,
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=chunked_prefill, chunk_size=chunk_size
            ),
        )
        md.prepare()
        return md

    def shutdown(self) -> None:
        self.manager.shutdown()
