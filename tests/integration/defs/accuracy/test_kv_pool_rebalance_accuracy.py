# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Accuracy test for the KVCacheManagerV2 rebalance hook.

Verifies that forcing the V2 auto-tuner to fire mid-generation does not
change greedy-decode outputs.  Uses Gemma-3-1B with explicit VSWA so the
KV cache lands in >=2 pool groups and ``adjust()`` has real work to do
(a single pool group would make rebalance a no-op).

Run as:
    LLM_MODELS_ROOT=/path pytest \
      tests/integration/defs/accuracy/test_kv_pool_rebalance_accuracy.py
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

from ..conftest import llm_models_root, skip_pre_hopper

# --------------------------------------------------------------------------- #
# Ratio injection
# --------------------------------------------------------------------------- #


def _inject_pool_ratio_mismatch(llm: LLM, *, skew: float = 2.0) -> None:
    """Force the V2 auto-tuner to do real pool-resize work on the next rebalance call.

    Delegates to the backend-agnostic KVCacheManagerV2 introspection hook, which
    bypasses the sample-count / cooldown gates and perturbs the target GPU ratio
    past the auto-tuner's adjustment threshold. The hook requires a model with
    >=2 pool groups (e.g. Gemma-3-1B with VSWA) and raises otherwise, so a future
    model change can't silently turn this test into a no-op.
    """
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    executor = llm._executor.engine
    kv_cache_manager = executor.kv_cache_manager
    _introspection.force_rebalance_precondition(kv_cache_manager.impl, skew=skew)


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #

# A handful of prompts spanning short, medium, and long context lengths.
# The long prompt is intentionally repetitive so it occupies multiple KV
# blocks and creates enough pool pressure for rebalance to matter.
_PROMPTS = [
    "The capital of France is",
    "Write one sentence about transformers.",
    "List three prime numbers greater than 100:",
    "The quick brown fox jumps over the lazy dog. " * 40,
]

_SAMPLING = SamplingParams(max_tokens=64, temperature=0.0, top_k=1)


def _vswa_kv_cache_config(*, enable_rebalance: bool) -> KvCacheConfig:
    """V2 manager + explicit VSWA pattern that yields multiple pool groups.

    Gemma-3-1B has 5 sliding-window layers : 1 full-attention layer.
    """
    return KvCacheConfig(
        use_kv_cache_manager_v2=True,
        enable_kv_pool_rebalance=enable_rebalance,
        max_attention_window=[512, 512, 512, 512, 512, 32768],
        # Block reuse disabled per the standing Gemma3 WAR for non-
        # inclusive sliding window kernel support.
        enable_block_reuse=False,
        enable_partial_reuse=False,
        tokens_per_block=32,
        free_gpu_memory_fraction=0.6,
    )


def _generate_tokens(*, model_path: str, disable_overlap: bool, enable_rebalance: bool):
    """Run one LLM, return list[list[int]] of generated token ids.

    Note: the ratio-injection helper requires direct access to the
    in-process PyExecutor, so the test runs in single-process worker
    mode (``TLLM_WORKER_USE_SINGLE_PROCESS=1``).  The caller is
    responsible for setting that env var (via monkeypatch or otherwise)
    before invoking this helper.
    """
    with LLM(
        model_path,
        disable_overlap_scheduler=disable_overlap,
        kv_cache_config=_vswa_kv_cache_config(enable_rebalance=enable_rebalance),
    ) as llm:
        if enable_rebalance:
            _inject_pool_ratio_mismatch(llm)
        outputs = llm.generate(_PROMPTS, _SAMPLING)
        return [list(o.outputs[0].token_ids) for o in outputs]


@skip_pre_hopper
class TestKvPoolRebalanceAccuracy:
    """Token-exact greedy-decode equivalence under rebalance.

    Compares rebalance=off and rebalance=on with a forced mid-generation
    adjust().
    """

    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-3-1b-it/"

    @pytest.mark.parametrize("disable_overlap", [True, False], ids=["no_overlap", "overlap"])
    def test_rebalance_matches_baseline(self, disable_overlap, monkeypatch):
        # Keep the PyExecutor in-process so the ratio-injection helper
        # can reach .engine on the client side.
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

        baseline = _generate_tokens(
            model_path=self.MODEL_PATH, disable_overlap=disable_overlap, enable_rebalance=False
        )

        treated = _generate_tokens(
            model_path=self.MODEL_PATH, disable_overlap=disable_overlap, enable_rebalance=True
        )

        assert len(baseline) == len(treated) == len(_PROMPTS)
        for i, (b, t) in enumerate(zip(baseline, treated)):
            assert b == t, (
                f"prompt {i}: rebalance changed greedy-decode output\n"
                f"  baseline: {b[:16]}...\n"
                f"  treated:  {t[:16]}..."
            )
