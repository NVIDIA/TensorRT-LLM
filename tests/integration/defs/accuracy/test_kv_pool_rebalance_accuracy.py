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

    Also drops the executor's rebalance-check throttle to every iteration, so
    the test does not depend on how ``KV_POOL_REBALANCE_CHECK_INTERVAL`` compares
    to the number of iterations this short prompt set happens to run.  Raise that
    interval above the iteration count and the hook would never fire, leaving the
    token comparison below to pass vacuously; the ratio assertion in
    ``_generate_tokens`` is the backstop that would catch it.
    """
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    executor = llm._executor.engine
    executor._rebalance_check_interval = 1
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

# Top-2 raw logprobs per generated token, so a divergence between the arms can
# be classified as a near-tie flip or a real accuracy break (see
# _assert_tokens_match_or_near_tie).  LogprobMode.RAW (the default) computes
# them from the unprocessed logits, so top_k=1 does not truncate them.
_SAMPLING = SamplingParams(max_tokens=64, temperature=0.0, top_k=1, logprobs=2)

# Largest top-1/top-2 logprob gap, in nats, at which the two arms may pick
# different greedy tokens.  Greedy decode is only exact up to floating-point
# reassociation: anything that changes accumulation order (batch composition,
# kernel selection) moves bf16 logits by a few ulps, which is 0.125 per ulp for
# logits in [16, 32).  Candidates that close can swap without anything being
# wrong.  nvbugs/6838020 hit exactly that on B300 -- prompt 1 at the token after
# "...architecture that", where " excels" leads " revolutionized" by 0.02 in
# fp32 and by one ulp in bf16.  Corrupted KV moves logits far more than this,
# so a divergence at a confident position still fails.
_NEAR_TIE_LOGPROB_MARGIN = 0.5


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
    """Run one LLM, return a (token_ids, logprobs) pair per prompt.

    ``logprobs`` holds one ``{token_id: Logprob}`` dict per generated token,
    covering that position's top-2 candidates.

    Note: the ratio-injection helper requires direct access to the
    in-process PyExecutor, so the test runs in single-process worker
    mode (``TLLM_WORKER_USE_SINGLE_PROCESS=1``).  The caller is
    responsible for setting that env var (via monkeypatch or otherwise)
    before invoking this helper.
    """
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    with LLM(
        model_path,
        disable_overlap_scheduler=disable_overlap,
        kv_cache_config=_vswa_kv_cache_config(enable_rebalance=enable_rebalance),
    ) as llm:
        impl = llm._executor.engine.kv_cache_manager.impl
        if enable_rebalance:
            _inject_pool_ratio_mismatch(llm)
        ratio_before = list(_introspection.current_gpu_ratio(impl))
        outputs = llm.generate(_PROMPTS, _SAMPLING)
        ratio_after = list(_introspection.current_gpu_ratio(impl))

        # Guard against a vacuous pass.  Token equality between the rebalance
        # and no-rebalance arms proves nothing if adjust() never ran, and
        # nothing in the run logs at info level to tell us it did.  The pool
        # ratio moving is the observable signature that it happened.
        if enable_rebalance:
            assert ratio_after != ratio_before, (
                "rebalance never fired: GPU pool ratio unchanged at "
                f"{ratio_before}. The token comparison would pass vacuously. "
                "Check the executor's rebalance-check throttle and the V2 "
                "auto-tuner's sample-count / cooldown gates."
            )
        else:
            assert ratio_after == ratio_before, (
                "pool ratio moved with enable_kv_pool_rebalance=False "
                f"({ratio_before} -> {ratio_after}); the baseline arm is "
                "supposed to hold pool ratios fixed."
            )

        results = []
        for o in outputs:
            token_ids = list(o.outputs[0].token_ids)
            logprobs = list(o.outputs[0].logprobs)
            # The near-tie check indexes logprobs by token position, so a
            # missing or misaligned list would make it fail for the wrong reason.
            assert len(logprobs) == len(token_ids), (
                f"expected one logprobs entry per generated token, got "
                f"{len(logprobs)} for {len(token_ids)} tokens"
            )
            results.append((token_ids, logprobs))
        return results


def _assert_tokens_match_or_near_tie(prompt_idx: int, baseline, treated) -> None:
    """Assert both arms decode identically, apart from at most one near-tie flip.

    Up to their first divergence the outputs must match token for token.  The
    divergence passes only if, in *both* arms, the two diverging tokens are that
    position's top-2 candidates and are within ``_NEAR_TIE_LOGPROB_MARGIN`` of
    each other.  That means the arms chose opposite sides of a near-tie while
    still agreeing on the two leading candidates.  Tokens after the divergence
    are not compared, because from there each arm continues a different prefix.
    """
    b_tokens, b_logprobs = baseline
    t_tokens, t_logprobs = treated
    context = (
        f"prompt {prompt_idx}: rebalance changed greedy-decode output\n"
        f"  baseline: {b_tokens[:16]}...\n"
        f"  treated:  {t_tokens[:16]}..."
    )

    k = next((j for j, (b, t) in enumerate(zip(b_tokens, t_tokens)) if b != t), None)
    if k is None:
        # One output is a prefix of the other, so one arm stopped early.  The
        # stopping decision does not show up as a token we could look up in
        # the logprobs, so the near-tie check cannot vouch for it.
        assert len(b_tokens) == len(t_tokens), (
            f"{context}\n  outputs agree for {min(len(b_tokens), len(t_tokens))} "
            f"tokens, then one arm stops ({len(b_tokens)} vs {len(t_tokens)} tokens)"
        )
        return

    b_tok, t_tok = b_tokens[k], t_tokens[k]
    for arm, logprobs in (("baseline", b_logprobs), ("treated", t_logprobs)):
        top = logprobs[k]
        assert b_tok in top and t_tok in top, (
            f"{context}\n  diverged at token {k} ({b_tok} vs {t_tok}), but the "
            f"{arm} arm's top-2 there is {sorted(top)}: not a near-tie flip"
        )
        margin = abs(top[b_tok].logprob - top[t_tok].logprob)
        assert margin <= _NEAR_TIE_LOGPROB_MARGIN, (
            f"{context}\n  diverged at token {k} ({b_tok} vs {t_tok}) with a "
            f"{arm} top-2 logprob gap of {margin:.4f} > "
            f"{_NEAR_TIE_LOGPROB_MARGIN}: not a near-tie flip"
        )
    print(
        f"prompt {prompt_idx}: accepted near-tie flip at token {k} "
        f"({b_tok} vs {t_tok}); {k} tokens matched before it"
    )


@skip_pre_hopper
class TestKvPoolRebalanceAccuracy:
    """Greedy-decode equivalence under rebalance.

    Compares rebalance=off and rebalance=on with a forced mid-generation
    adjust().  Outputs must match token for token, except that one divergence
    at a near-tie is tolerated (see _assert_tokens_match_or_near_tie).
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
            _assert_tokens_match_or_near_tie(i, b, t)
