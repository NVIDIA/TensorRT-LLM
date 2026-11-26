# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank functional checks for hybrid prefix state on GB300."""

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

from ._engine import shortcut_text_engine


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp2"])
def test_chunked_prefix_resume(model_root, precision, mtp, record_property):
    chunk_tokens = 256
    with shortcut_text_engine(
        model_root / precision,
        max_batch_size=1,
        max_num_tokens=chunk_tokens,
        max_seq_len=2048,
        enable_chunked_prefill=True,
        speculative_config=MTPDecodingConfig(max_draft_len=2) if mtp else None,
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True,
            enable_block_reuse=True,
            mamba_state_config={"additional_snapshot_offsets_from_end": [1]},
        ),
    ) as llm:
        sampling = SamplingParams(temperature=0, top_k=1, max_tokens=12, return_perf_metrics=mtp)
        # Distinct first tokens detect accidental reuse across requests. The
        # answer depends on information in the first chunk in each case.
        filler = " A page of unrelated notes." * 100
        answers = []
        accepted = 0
        for city in ("Paris", "Tokyo"):
            prompt = f"{city} is the city in this record.{filler}\nThe city in this record is"
            cold = llm.generate(prompt, sampling)
            warm = llm.generate(prompt, sampling)
            assert len(cold.prompt_token_ids) > chunk_tokens
            assert cold.cached_tokens < chunk_tokens
            assert warm.cached_tokens >= chunk_tokens
            assert list(warm.outputs[0].token_ids) == list(cold.outputs[0].token_ids)
            assert warm.outputs[0].token_ids
            answers.append(warm.outputs[0].text)
            record_property(f"{city}_prompt_tokens", len(cold.prompt_token_ids))
            record_property(f"{city}_cached_tokens", (cold.cached_tokens, warm.cached_tokens))
            record_property(f"{city}_continuation", list(warm.outputs[0].token_ids))
            if mtp:
                metrics = warm.outputs[0].request_perf_metrics.speculative_decoding
                record_property(
                    f"{city}_accepted_drafted",
                    (metrics.total_accepted_draft_tokens, metrics.total_draft_tokens),
                )
                assert metrics.total_draft_tokens > 0
                accepted += metrics.total_accepted_draft_tokens
        assert answers[0] != answers[1]
        if mtp:
            assert accepted > 0, "Neither warm request accepted any draft tokens"


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
def test_host_cache_round_trip(model_root, precision, record_property):
    with shortcut_text_engine(
        model_root / precision,
        max_batch_size=1,
        max_seq_len=2048,
        max_num_tokens=256,
        enable_chunked_prefill=True,
        enable_iter_perf_stats=True,
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True,
            enable_block_reuse=True,
            max_tokens=2048,
            host_cache_size=4 << 30,
            cold_tier_offload_min_priority=0,
            mamba_state_config={"additional_snapshot_offsets_from_end": [1]},
        ),
    ) as llm:
        sampling = SamplingParams(temperature=0, max_tokens=8)
        prompt = "Paris is the capital in this record." + " An unrelated note." * 160
        prompt += "\nName the capital in the record:"
        cold = llm.generate(prompt, sampling)
        llm.get_stats(timeout=2)
        offloaded = 0
        for word in ("blue", "red", "green", "yellow", "white", "orange", "purple", "gold"):
            llm.generate((word + " ") * 900, sampling)
            for iteration in llm.get_stats(timeout=2):
                for pool in iteration.get("kvCacheIterationStatsByPoolGroup", {}).values():
                    offloaded += pool["iterOffloadBytes"]
            if offloaded:
                break
        assert offloaded > 0, "No cached pages moved to the host under pressure"
        restored = llm.generate(prompt, sampling)
        onboarded = sum(
            pool["iterOnboardBytes"]
            for iteration in llm.get_stats(timeout=2)
            for pool in iteration.get("kvCacheIterationStatsByPoolGroup", {}).values()
        )
        assert onboarded > 0, "The repeated request did not read any host-resident pages"
        assert restored.cached_tokens > 0
        assert list(restored.outputs[0].token_ids) == list(cold.outputs[0].token_ids)
        record_property("offloaded_bytes", offloaded)
        record_property("onboarded_bytes", onboarded)
        record_property("restored_cache_tokens", restored.cached_tokens)
