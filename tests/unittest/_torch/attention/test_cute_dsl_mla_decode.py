# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode-only MLA test for the Blackwell CuTe DSL MLA decode kernels.

This test validates the CuTe DSL MLA *decode* kernels added under
``tensorrt_llm/_torch/cute_dsl_kernels/blackwell/attention/mla`` and dispatched
through the ``cute_dsl_mla`` TRTLLM FMHA library
(``tensorrt_llm/_torch/attention_backend/fmha/cute_dsl.py``):

- FP8 path: ``torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell``
- FP16/BF16 path: ``torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell``

Only the generation (decode) steps are asserted for numerical correctness.
The context phase runs solely to populate the paged KV cache and to build the
reference latent cache (``skip_context_assert=True``).

Crucially, the test monkeypatches ``CuteDslMlaFmha._run_mla_decode`` to count
invocations and asserts the CuTe DSL decode path was actually taken on every
decode step.

Platform: Blackwell SM100 / SM103 only.
"""

import pytest
import torch

# Reuse the proven setup + reference machinery from the full MLA test.
# The attention test directory is added to sys.path by pytest (prepend import
# mode, no package __init__), so the sibling module is imported by bare name.
import test_attention_mla
from test_attention_mla import RopeConfig, Scenario, _run_test_for_backend

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

# DeepSeek-V3-like MLA geometry the CuTe DSL kernel targets (num_heads=128,
# latent_dim=512, rope_dim=64). Kept small along the batch/step axes so the
# decode-only test stays fast. Multi-token/MTP decode is handled by replaying
# the single-token CuTe DSL kernel once per intra-step query.
_DECODE_CONTEXT_LENGTHS = [
    [10, 12, 5],
    [100, 300, 20, 10],
]
_DECODE_NUM_STEPS = 4

# Multi-layer is the structural difference between this single-step test and the
# real DeepSeek-V3 E2E run (61 MLA layers). With ``num_layers == 1`` the dispatch
# only ever sees ``layer_idx == 0``, so the per-layer paged-KV resolution in
# ``CuteDslMlaFmha._run_mla_decode`` (the
# ``host_kv_cache_pool_mapping[layer_idx]`` / per-layer ``get_buffers`` /
# block-offset path) is never exercised. The E2E run produces correct output on
# the first generated token (which comes from the TRTLLM prefill) and then
# degenerates on every subsequent CuteDSL decode step, consistent with the
# decode kernel reading the wrong blocks for ``layer_idx > 0``. Parametrize over
# >1 layers so the unit test reproduces that real case.
_DECODE_NUM_LAYERS = [1, 2]


def _is_blackwell_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.skipif(
        not _is_blackwell_sm100(),
        reason="CuTe DSL MLA decode kernels require Blackwell SM100/SM103.",
    ),
    pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not available."),
]


# kernel name -> (activation dtype, kv cache dtype)
#
# NOTE: the float16 instance of the FP16 decode op
# (``cute_dsl_mla_decode_fp16_blackwell`` with dtype=float16 / fp16 KV cache)
# aborts the process (SIGABRT) on SM100 in this environment, so that exact
# dtype is excluded until the crash is root-caused. The bf16 instance uses the
# same op and is covered below for DeepSeek-V3 bf16 runs.
_KERNEL_DTYPES = {
    "fp8": (torch.bfloat16, torch.float8_e4m3fn),
    "bf16": (torch.bfloat16, torch.bfloat16),
}


def _build_rope_config(scenario: Scenario) -> RopeConfig:
    return RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )


@pytest.fixture
def cute_dsl_decode_counter(monkeypatch):
    """Count successful CuTe DSL MLA decode dispatches so the test fails on
    silent fallback to the TRTLLM backend.

    The increment happens only after the real dispatch returns. If the kernel
    raises or the registry selects the fallback FMHA library, the counter does
    not advance and the per-test assertion fails loudly instead of the broken
    kernel masquerading as a working one."""
    from tensorrt_llm._torch.attention_backend.fmha.cute_dsl import CuteDslMlaFmha

    monkeypatch.setenv("TLLM_FMHA_LIBS", "cute_dsl_mla,fallback")

    original = CuteDslMlaFmha._run_mla_decode
    counter = {"calls": 0}

    def _counting_dispatch(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        counter["calls"] += 1
        return result

    monkeypatch.setattr(CuteDslMlaFmha, "_run_mla_decode", _counting_dispatch)
    return counter


@pytest.mark.parametrize("kernel", list(_KERNEL_DTYPES))
@pytest.mark.parametrize(
    "context_sequence_lengths", _DECODE_CONTEXT_LENGTHS, ids=lambda x: f"ctx_lens={x}"
)
@pytest.mark.parametrize("generation_seq_len_q", [1, 4, 8], ids=lambda x: f"gen_seq_len_q={x}")
@pytest.mark.parametrize("num_layers", _DECODE_NUM_LAYERS, ids=lambda x: f"num_layers={x}")
def test_cute_dsl_mla_decode(
    kernel, context_sequence_lengths, generation_seq_len_q, num_layers, cute_dsl_decode_counter
):
    """Decode-only MLA validation for the Blackwell CuTe DSL kernels."""
    dtype, kv_cache_dtype = _KERNEL_DTYPES[kernel]

    scenario = Scenario(dtype=dtype, kv_cache_dtype=kv_cache_dtype, num_layers=num_layers)
    rope_config = _build_rope_config(scenario)

    _run_test_for_backend(
        "TRTLLM",
        num_heads=scenario.num_heads,
        num_kv_heads=scenario.num_kv_heads,
        num_layers=scenario.num_layers,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        rope_config=rope_config,
        kv_cache_tokens_per_block=scenario.kv_cache_tokens_per_block,
        device=torch.device("cuda"),
        dtype=scenario.dtype,
        kv_cache_dtype=scenario.kv_cache_dtype,
        context_sequence_lengths=context_sequence_lengths,
        generation_seq_len_q=generation_seq_len_q,
        num_generation_steps=_DECODE_NUM_STEPS,
        v2_kv_cache=True,
        skip_context_assert=True,
    )

    # The decode path must have actually run the CuTe DSL kernel (1 dispatch
    # per layer per decode step), not silently fallen back to TRTLLM.
    expected = scenario.num_layers * _DECODE_NUM_STEPS
    assert cute_dsl_decode_counter["calls"] == expected, (
        f"Expected {expected} CuTe DSL MLA decode dispatches, got "
        f"{cute_dsl_decode_counter['calls']} (silent TRTLLM fallback?)"
    )


# Fold-path (H < M_tile) validation. The DeepSeek-V3 E2E run with TP=8 shards
# the 128 attention heads to num_heads=16 per rank; the decode kernel then folds
# F = compute_fold_sq_ratio(num_heads=16, seq_len_q, m_tile=128) query tokens
# into the head dim so M_eff = 16*F. The default ``test_cute_dsl_mla_decode``
# above uses num_heads=128 (>= m_tile) so F is always 1 (no fold) -- it validates
# seq_len_q=8 *correctness* but NOT the H=16 fold code path that the real run
# actually takes. This test pins num_heads=16 so each seq_len_q exercises a
# distinct fold factor: sq=1->F=1, sq=2->F=2, sq=4->F=4, sq=8->F=8 (M_eff=128,
# 100% M-tile fill). Confirms the fold path is numerically correct and that the
# FMHA gate/can_implement actually engage CuteDSL at seq_len_q=8 / H=16 (the
# geometry that silently fell back to TRTLLM in the draft_len=7 E2E bench).
_FOLD_NUM_HEADS = 16


@pytest.mark.parametrize("kernel", list(_KERNEL_DTYPES))
@pytest.mark.parametrize(
    "context_sequence_lengths", _DECODE_CONTEXT_LENGTHS, ids=lambda x: f"ctx_lens={x}"
)
@pytest.mark.parametrize("generation_seq_len_q", [1, 2, 4, 8], ids=lambda x: f"gen_seq_len_q={x}")
@pytest.mark.parametrize("num_layers", _DECODE_NUM_LAYERS, ids=lambda x: f"num_layers={x}")
def test_cute_dsl_mla_decode_fold_sq(
    kernel, context_sequence_lengths, generation_seq_len_q, num_layers, cute_dsl_decode_counter
):
    """H=16 (TP=8 per-rank) fold-path decode validation for the CuTe DSL kernels."""
    dtype, kv_cache_dtype = _KERNEL_DTYPES[kernel]

    scenario = Scenario(
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        num_layers=num_layers,
        num_heads=_FOLD_NUM_HEADS,
        num_kv_heads=_FOLD_NUM_HEADS,
    )
    rope_config = _build_rope_config(scenario)

    _run_test_for_backend(
        "TRTLLM",
        num_heads=scenario.num_heads,
        num_kv_heads=scenario.num_kv_heads,
        num_layers=scenario.num_layers,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        rope_config=rope_config,
        kv_cache_tokens_per_block=scenario.kv_cache_tokens_per_block,
        device=torch.device("cuda"),
        dtype=scenario.dtype,
        kv_cache_dtype=scenario.kv_cache_dtype,
        context_sequence_lengths=context_sequence_lengths,
        generation_seq_len_q=generation_seq_len_q,
        num_generation_steps=_DECODE_NUM_STEPS,
        v2_kv_cache=True,
        skip_context_assert=True,
    )

    expected = scenario.num_layers * _DECODE_NUM_STEPS
    assert cute_dsl_decode_counter["calls"] == expected, (
        f"Expected {expected} CuTe DSL MLA decode dispatches, got "
        f"{cute_dsl_decode_counter['calls']} (silent TRTLLM fallback?)"
    )


# E2E-reproduction case: a SHORT prompt decoded for MANY steps. The real
# DeepSeek-V3 run feeds a ~6-token prompt and generates ~64 tokens; its output
# is correct on the first (prefill) token and then degenerates on every CuteDSL
# decode step. The parametrized ``test_cute_dsl_mla_decode`` above keeps decode
# to 4 steps and so, even with a long context, never allocates a fresh paged-KV
# block *during* generation. Here the KV length grows from a short context
# across the page_size==32 block boundaries (at lengths 32 and 64) *mid-decode*,
# allocating new blocks and extending the per-request page_table on the fly —
# the exact paged-KV path the short-decode test never exercises. fp8 +
# seq_len_q==1 only (the configuration that degenerates E2E).
_LONG_DECODE_CONTEXT_LENGTHS = [5, 8, 3, 11]
_LONG_DECODE_NUM_STEPS = 64


@pytest.mark.parametrize("kernel", list(_KERNEL_DTYPES))
@pytest.mark.parametrize("num_layers", _DECODE_NUM_LAYERS, ids=lambda x: f"num_layers={x}")
@pytest.mark.parametrize("v2_kv_cache", [True, False], ids=lambda x: f"v2_kv_cache={x}")
def test_cute_dsl_mla_decode_long_decode(v2_kv_cache, num_layers, kernel, cute_dsl_decode_counter):
    """Long decode-only MLA run that crosses paged-KV block boundaries mid-decode.

    Reproduction for the DeepSeek-V3 E2E degeneration: short prompt, long
    generation, ``seq_len_q == 1``. The real run uses the v1 ``KVCacheManager``
    (``use_kv_cache_manager_v2=False``); the rest of this file only exercised the
    v2 manager, so ``v2_kv_cache`` is parametrized here to cover the v1 paged-KV
    block-offset layout that the dispatch resolves in
    ``_dispatch_cute_dsl_mla_decode``.
    """
    dtype, kv_cache_dtype = _KERNEL_DTYPES[kernel]

    scenario = Scenario(dtype=dtype, kv_cache_dtype=kv_cache_dtype, num_layers=num_layers)
    rope_config = _build_rope_config(scenario)

    _run_test_for_backend(
        "CUTEDSL",
        num_heads=scenario.num_heads,
        num_kv_heads=scenario.num_kv_heads,
        num_layers=scenario.num_layers,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        rope_config=rope_config,
        kv_cache_tokens_per_block=scenario.kv_cache_tokens_per_block,
        device=torch.device("cuda"),
        dtype=scenario.dtype,
        kv_cache_dtype=scenario.kv_cache_dtype,
        context_sequence_lengths=_LONG_DECODE_CONTEXT_LENGTHS,
        generation_seq_len_q=1,
        num_generation_steps=_LONG_DECODE_NUM_STEPS,
        v2_kv_cache=v2_kv_cache,
        skip_context_assert=True,
    )

    expected = scenario.num_layers * _LONG_DECODE_NUM_STEPS
    assert cute_dsl_decode_counter["calls"] == expected, (
        f"Expected {expected} CuTe DSL MLA decode dispatches, got "
        f"{cute_dsl_decode_counter['calls']} (silent TRTLLM fallback?)"
    )


# Standalone-shape parity: run the SAME (num_heads, batch, KV, seq_q) geometries
# the standalone kernel benchmark uses (bench/cutedsl_mla + standalone_mla_*.md)
# through the full integration decode path, so the two are apples-to-apples.
#
# The standalone harness feeds the kernel a total KV length ``seq_len_k = KV``.
# The integration path instead scans ``cache_seqs = num_cached + seq_len_q``:
# every freshly-appended query token of this step is counted (see
# ``attention_backend/fmha/cute_dsl.py``). To make the effective KV identical we
# set the context length to ``KV - seq_len_q`` and take a SINGLE decode step, so
# the decode kernel scans exactly ``KV`` positions -- matching the standalone
# column instead of ``KV + seq_len_q``.
#
# num_layers is pinned to 1 (multi-layer paged-KV resolution is covered above)
# and the batch/KV grid is curated to keep the bf16 reference cost bounded while
# still sampling every batch magnitude, every KV length, both head counts
# (16 = TP=8 per-rank fold path, 128 = no fold), and both seq_q values.
_STANDALONE_SHAPES = [
    # (num_heads, batch, kv)
    (16, 1, 1024),
    (16, 2, 8192),
    (16, 8, 4096),
    (16, 32, 2048),
    (16, 64, 1024),
    (16, 256, 1024),
    (128, 1, 8192),
    (128, 4, 4096),
    (128, 8, 8192),
    (128, 16, 2048),
    (128, 64, 1024),
]


@pytest.mark.parametrize("kernel", list(_KERNEL_DTYPES))
@pytest.mark.parametrize("generation_seq_len_q", [1, 2], ids=lambda x: f"gen_seq_len_q={x}")
@pytest.mark.parametrize(
    "num_heads,batch,kv",
    _STANDALONE_SHAPES,
    ids=[f"h{h}_b{b}_kv{k}" for (h, b, k) in _STANDALONE_SHAPES],
)
def test_cute_dsl_mla_decode_standalone_shapes(
    num_heads, batch, kv, generation_seq_len_q, kernel, cute_dsl_decode_counter,
    monkeypatch,
):
    """Decode-path parity with the standalone kernel benchmark shapes.

    Effective KV equals the standalone ``KV`` column: the context length is
    ``KV - seq_len_q`` and a single decode step is taken, so the CuTe DSL kernel
    scans exactly ``KV`` positions.
    """
    seq_q = generation_seq_len_q
    if kv - seq_q <= 0:
        pytest.skip("KV too short for the requested seq_len_q.")
    dtype, kv_cache_dtype = _KERNEL_DTYPES[kernel]

    # ``_run_test_for_backend`` sizes the KV-cache pool from the module globals
    # ``max_context_sequence_length`` (default 1000, for tiny correctness tests)
    # and ``max_num_contexts`` (default 10), NOT from the actual shape. These
    # standalone shapes use ctx up to ~8k over up to 256 sequences, so bump both
    # to the real shape or the pool runs out ("Not enough pages in GPU memory").
    monkeypatch.setattr(test_attention_mla, "max_context_sequence_length",
                        max(kv, test_attention_mla.max_context_sequence_length))
    monkeypatch.setattr(test_attention_mla, "max_num_contexts",
                        max(batch, test_attention_mla.max_num_contexts))

    scenario = Scenario(
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        num_layers=1,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )
    rope_config = _build_rope_config(scenario)

    context_sequence_lengths = [kv - seq_q] * batch
    num_generation_steps = 1

    _run_test_for_backend(
        "TRTLLM",
        num_heads=scenario.num_heads,
        num_kv_heads=scenario.num_kv_heads,
        num_layers=scenario.num_layers,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        rope_config=rope_config,
        kv_cache_tokens_per_block=scenario.kv_cache_tokens_per_block,
        device=torch.device("cuda"),
        dtype=scenario.dtype,
        kv_cache_dtype=scenario.kv_cache_dtype,
        context_sequence_lengths=context_sequence_lengths,
        generation_seq_len_q=seq_q,
        num_generation_steps=num_generation_steps,
        v2_kv_cache=True,
        skip_context_assert=True,
    )

    expected = scenario.num_layers * num_generation_steps
    assert cute_dsl_decode_counter["calls"] == expected, (
        f"Expected {expected} CuTe DSL MLA decode dispatches, got "
        f"{cute_dsl_decode_counter['calls']} (silent TRTLLM fallback?)"
    )


# (batch, kv): batch=2 exercises the split_kv>1 workspace path, batch=64 the
# split_kv==1 many-tiles path -- both under is_persistent tactic profiling.
_AUTOTUNE_SHAPES = [(2, 2048), (64, 1024)]


@pytest.mark.parametrize("kernel", list(_KERNEL_DTYPES))
@pytest.mark.parametrize(
    "force_persistent",
    [None, "0", "1"],
    ids=lambda v: f"force_persistent={v}",
)
@pytest.mark.parametrize(
    "batch,kv", _AUTOTUNE_SHAPES, ids=[f"b{b}_kv{k}" for (b, k) in _AUTOTUNE_SHAPES]
)
def test_cute_dsl_mla_decode_autotuned(
    batch, kv, force_persistent, kernel, cute_dsl_decode_counter, monkeypatch,
):
    """Exercise the op AutoTuner's ``is_persistent`` tactic path end-to-end.

    Unlike the other decode tests (which never enter ``with autotune()`` and so
    only run ``default_tactic``), this warms the AutoTuner on the first decode
    step. That drives ``get_valid_tactics`` to enumerate both ``is_persistent``
    variants (via ``get_is_persistent_candidates``), the tuner profiles the
    ``(tiler, split_kv, is_persistent)`` 4-tuples and caches the winner, and the
    next step reuses the tuned tactic. Both variants are numerically identical
    (persistent is a scheduling/codegen choice, not a math change), so the
    assertion is that every profiled+selected variant compiles, runs, and stays
    correct with no silent fallback:

    - ``force_persistent=None`` -> tuner enumerates [True, False] and PICKS one.
    - ``force_persistent="1"/"0"`` -> ``TLLM_CUTE_DSL_FORCE_PERSISTENT`` pins the
      single candidate, so each variant is validated through the tuner in turn.
    """
    seq_q = 2
    if kv - seq_q <= 0:
        pytest.skip("KV too short for the requested seq_len_q.")
    dtype, kv_cache_dtype = _KERNEL_DTYPES[kernel]

    if force_persistent is not None:
        monkeypatch.setenv("TLLM_CUTE_DSL_FORCE_PERSISTENT", force_persistent)

    monkeypatch.setattr(test_attention_mla, "max_context_sequence_length",
                        max(kv, test_attention_mla.max_context_sequence_length))
    monkeypatch.setattr(test_attention_mla, "max_num_contexts",
                        max(batch, test_attention_mla.max_num_contexts))

    scenario = Scenario(
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        num_layers=1,
        num_heads=128,
        num_kv_heads=128,
    )
    rope_config = _build_rope_config(scenario)

    context_sequence_lengths = [kv - seq_q] * batch
    # >=2 decode steps: step 1 warms + caches the tactic under autotune, step 2
    # runs outside autotune and must reuse the cached tuned tactic.
    num_generation_steps = 2

    _run_test_for_backend(
        "TRTLLM",
        num_heads=scenario.num_heads,
        num_kv_heads=scenario.num_kv_heads,
        num_layers=scenario.num_layers,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        rope_config=rope_config,
        kv_cache_tokens_per_block=scenario.kv_cache_tokens_per_block,
        device=torch.device("cuda"),
        dtype=scenario.dtype,
        kv_cache_dtype=scenario.kv_cache_dtype,
        context_sequence_lengths=context_sequence_lengths,
        generation_seq_len_q=seq_q,
        num_generation_steps=num_generation_steps,
        v2_kv_cache=True,
        skip_context_assert=True,
        autotune_warmup=True,
    )

    expected = scenario.num_layers * num_generation_steps
    assert cute_dsl_decode_counter["calls"] == expected, (
        f"Expected {expected} CuTe DSL MLA decode dispatches, got "
        f"{cute_dsl_decode_counter['calls']} (silent TRTLLM fallback?)"
    )
