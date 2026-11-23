# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""End-to-end check of Gemma3's default SWA scratch reuse.

With block reuse off, Gemma3ForCausalLM turns SWA scratch reuse on by default
under TRTLLM attention. A prefill chunk longer than gemma-3-1b-it's 512-token
window writes the blocks that leave the window inside that chunk to scratch
slots; later chunks and decode read the retained per-layer tables. Greedy
outputs must match the explicit opt-out token for token.

Run as:
    LLM_MODELS_ROOT=/path pytest \
      tests/integration/defs/accuracy/test_gemma3_swa_scratch_default.py
"""

from collections import Counter

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.attention.backends.trtllm import trtllm_supports_swa_scratch_reuse
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvcm_v2
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

from ..conftest import llm_models_root

# Longer than the 512-token window, so a chunk can hold blocks that leave it.
_MAX_NUM_TOKENS = 1024
_SAMPLING = SamplingParams(max_tokens=24, temperature=0.0, top_k=1)


def _prompt(num_lockers: int, target: int) -> str:
    lines = [f"Locker {i} holds the code {(i * 7919) % 10000:04d}." for i in range(num_lockers)]
    return "\n".join(lines) + f"\nQuestion: which code does locker {target} hold? Answer:"


# Roughly 2-3K tokens each, with different lengths so chunk boundaries differ.
_PROMPTS = [_prompt(n, t) for n, t in ((150, 3), (170, 41), (190, 97), (210, 160))]


def _count_page_table_paths(monkeypatch: pytest.MonkeyPatch) -> Counter:
    calls = Counter()
    scratch_metadata = kvcm_v2.KVCacheManagerV2._copy_scratch_metadata_to_device
    per_layer_copy = kvcm_v2.copy_per_layer_block_offsets_to_device

    def count_scratch_step(self, *args, **kwargs):
        calls["scratch_step"] += 1
        return scratch_metadata(self, *args, **kwargs)

    def count_per_layer_copy(*args, **kwargs):
        calls["per_layer_copy"] += 1
        return per_layer_copy(*args, **kwargs)

    monkeypatch.setattr(
        kvcm_v2.KVCacheManagerV2, "_copy_scratch_metadata_to_device", count_scratch_step
    )
    monkeypatch.setattr(kvcm_v2, "copy_per_layer_block_offsets_to_device", count_per_layer_copy)
    return calls


def _generate(*, opt_out: bool) -> list[list[int]]:
    kv_cache_config = {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.5}
    if opt_out:
        kv_cache_config["enable_swa_scratch_reuse"] = False
    with LLM(
        f"{llm_models_root()}/gemma/gemma-3-1b-it/",
        kv_cache_config=KvCacheConfig(**kv_cache_config),
        enable_chunked_prefill=True,
        max_num_tokens=_MAX_NUM_TOKENS,
        max_seq_len=4096,
    ) as llm:
        manager = llm._executor.engine.kv_cache_manager
        assert manager.enable_swa_scratch_reuse is not opt_out
        outputs = llm.generate(_PROMPTS, _SAMPLING)
    return [list(output.outputs[0].token_ids) for output in outputs]


class TestGemma3SwaScratchDefault:
    def test_default_scratch_matches_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not trtllm_supports_swa_scratch_reuse():
            pytest.skip("SWA scratch reuse needs SM90 or an SM100-family GPU.")
        # Keep the executor in-process so the counters see its page-table copies.
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        calls = _count_page_table_paths(monkeypatch)

        scratch = _generate(opt_out=False)
        assert calls["scratch_step"] > 0, "no prefill step ran with scratch slots"
        assert calls["per_layer_copy"] > 0, "no step used the per-layer copy kernel"

        calls.clear()
        reference = _generate(opt_out=True)
        assert calls["scratch_step"] == 0 and calls["per_layer_copy"] == 0

        assert all(len(tokens) > 0 for tokens in scratch)
        assert scratch == reference
