# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

from .._common import ShortcutTextAccuracy
from .._engine import shortcut_text_engine


class TestShortcutNvfp4(ShortcutTextAccuracy):
    MODEL_NAME = "nemotron_shortcut/nvfp4"

    def test_gsm8k(self):
        self.evaluate_gsm8k()


@pytest.mark.skip_less_device(4)
def test_mtp_greedy_matches_ordinary_decode(model_root):
    # With two draft tokens, each step verifies three rows.
    prompt = "Answer with the name of the city. The capital of France is"
    window = 2176

    def generate(speculative_config=None):
        with shortcut_text_engine(
            model_root / "nvfp4",
            max_batch_size=1,
            max_seq_len=window,
            max_num_tokens=window,
            speculative_config=speculative_config,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False, avg_seq_len=window, free_gpu_memory_fraction=0.5
            ),
        ) as llm:
            tokens = llm.tokenizer.encode(prompt, add_special_tokens=False)
            assert len(tokens) == 13
            sampling = SamplingParams(
                temperature=0,
                top_k=1,
                max_tokens=12,
                return_perf_metrics=speculative_config is not None,
            )
            return llm.generate(tokens, sampling).outputs[0]

    ordinary = generate()
    mtp = generate(MTPDecodingConfig(max_draft_len=2))
    assert mtp.request_perf_metrics.speculative_decoding.total_accepted_draft_tokens > 0
    assert len(ordinary.token_ids) >= 8
    assert mtp.token_ids == ordinary.token_ids
