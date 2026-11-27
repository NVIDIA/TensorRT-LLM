# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ._engine import shortcut_text_engine


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
def test_deterministic_batch_invariance(model_root, precision, record_property):
    # Sixteen seeded random prompts, enough to fill the decode batch.
    prompts = []
    for index in range(16):
        rng = random.Random(225110 + index)
        prompts.append([rng.randrange(1024, 32000) for _ in range(1024)])
    with shortcut_text_engine(
        model_root / precision,
        env_overrides={"FORCE_DETERMINISTIC": "1"},
        max_batch_size=16,
        max_seq_len=4224,
        max_num_tokens=4096,
        enable_chunked_prefill=False,
        enable_iter_perf_stats=True,
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=16),
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True, enable_block_reuse=False),
    ) as llm:
        sampling = SamplingParams(
            temperature=0, top_k=1, max_tokens=32, min_tokens=32, ignore_eos=True, detokenize=False
        )
        expected = [list(llm.generate(p, sampling).outputs[0].token_ids) for p in prompts]
        together = llm.generate(prompts, sampling)
        peak = max(s["inflightBatchingStats"]["numGenRequests"] for s in llm.get_stats(timeout=2))
        assert peak == 16
        actual = [list(request.outputs[0].token_ids) for request in together]
        record_property("peak_decode_batch", peak)
        record_property("single_token_ids", expected)
        record_property("batched_token_ids", actual)
        for index, (alone, batched) in enumerate(zip(expected, actual, strict=True)):
            assert len(alone) == 32, f"Prompt {index} generated {len(alone)} tokens alone"
            assert batched == alone, f"Prompt {index}: single {alone}, batched {batched}"
