# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified, detailed test suite for the PyTorch attention backends.

Every case runs the VanillaAttention golden plus each supported backend
(TRTLLM, FlashInfer) through a real KVCacheManager and asserts they match.

Coverage spans head dims, GQA/MQA, dtypes, fp8/fp4 KV cache, mask types,
RoPE, sliding window, paged vs no-cache, and context/decode/mixed batches.

Sampling: the full product is large; ``TRTLLM_ATTN_TEST_SAMPLE`` controls how
many sweep cases run (``full`` for everything, else an int count; default 80).
"""

import itertools
import os
import random

import pytest
import torch
from attention_test_harness import BackendCase, generate_inputs, run_backend, run_case
from utils.util import getSMVersion

# A simple neox RoPE config used by rope-on cases.
_ROPE_NEOX = dict(dim=None, theta=10000.0, max_positions=8192, is_neox=True)


def _rope_for(head_dim: int):
    cfg = dict(_ROPE_NEOX)
    cfg["dim"] = head_dim
    return cfg


def _needs_skip(case: BackendCase) -> str | None:
    """Return an sm-gating skip reason for the case, if any."""
    if case.kv_dtype == "float8_e4m3fn" and getSMVersion() < 89:
        return "FP8 KV cache requires sm>=89"
    if case.kv_dtype == "nvfp4" and getSMVersion() < 100:
        return "NVFP4 KV cache requires sm>=100 (Blackwell)"
    return None


# ---------------------------------------------------------------------------
# Curated core cases: explicit, fast, cover the key dimensions individually.
# ---------------------------------------------------------------------------
def _core_cases():
    cases = {}

    def add(name, **kw):
        kw.setdefault("num_heads", 8)
        kw.setdefault("num_kv_heads", 2)
        kw.setdefault("head_dim", 128)
        cases[name] = BackendCase(**kw)

    # Pure prefill (context only).
    add("ctx_causal", seq_lens=[16, 24], num_cached_tokens=[0, 0], num_contexts=2)
    add("ctx_full", seq_lens=[16, 24], num_cached_tokens=[0, 0], num_contexts=2, causal=False)
    # Pure decode (generation only): 1 new token, multi-block cached prefix.
    add("decode", seq_lens=[1, 1, 1], num_cached_tokens=[20, 130, 8], num_contexts=0, page_size=64)
    # Mixed in-flight batch: contexts + generations together.
    add(
        "mixed_ifb",
        seq_lens=[16, 1, 1],
        num_cached_tokens=[0, 30, 70],
        num_contexts=1,
        page_size=64,
    )
    # GQA / MQA / MHA.
    add(
        "mha",
        num_heads=8,
        num_kv_heads=8,
        seq_lens=[12, 1],
        num_cached_tokens=[0, 40],
        num_contexts=1,
    )
    add(
        "mqa",
        num_heads=16,
        num_kv_heads=1,
        seq_lens=[12, 1],
        num_cached_tokens=[0, 40],
        num_contexts=1,
    )
    # head_dim variations.
    add("hd64", head_dim=64, seq_lens=[16, 1], num_cached_tokens=[0, 40], num_contexts=1)
    # bf16.
    add("bf16", dtype="bfloat16", seq_lens=[16, 1], num_cached_tokens=[0, 40], num_contexts=1)
    # FP8 KV cache: context (prefill) exercises TRTLLM + FlashInfer; the mixed
    # case below validates FlashInfer fp8 decode (TRTLLM fp8 decode w/ manual
    # prefill is a documented limitation, skipped via the capability matrix).
    add(
        "fp8_ctx",
        kv_dtype="float8_e4m3fn",
        seq_lens=[16, 24],
        num_cached_tokens=[0, 0],
        num_contexts=2,
    )
    add(
        "fp8",
        kv_dtype="float8_e4m3fn",
        seq_lens=[16, 1, 1],
        num_cached_tokens=[0, 40, 8],
        num_contexts=1,
    )
    # Sliding window.
    add("sliding", seq_lens=[1, 1], num_cached_tokens=[100, 130], num_contexts=0, sliding_window=32)
    # RoPE on (non-fused; harness applies rope, all backends compare).
    add(
        "rope_ctx", seq_lens=[24, 16], num_cached_tokens=[0, 0], num_contexts=2, rope=_rope_for(128)
    )
    add(
        "rope_decode",
        seq_lens=[1, 1],
        num_cached_tokens=[40, 90],
        num_contexts=0,
        rope=_rope_for(128),
    )
    # No KV cache (ragged prefill).
    add(
        "no_cache_causal",
        cache="none",
        seq_lens=[16, 24, 5],
        num_cached_tokens=[0, 0, 0],
        num_contexts=3,
    )
    add(
        "no_cache_full",
        cache="none",
        seq_lens=[16, 24, 5],
        num_cached_tokens=[0, 0, 0],
        num_contexts=3,
        causal=False,
    )
    # KVCacheManagerV2.
    add(
        "decode_v2",
        seq_lens=[1, 1],
        num_cached_tokens=[20, 50],
        num_contexts=0,
        use_kv_cache_manager_v2=True,
    )
    # Small page size (more blocks per sequence).
    add("page16", seq_lens=[1], num_cached_tokens=[40], num_contexts=0, page_size=16)
    return cases


CORE_CASES = _core_cases()


@pytest.mark.parametrize("name", list(CORE_CASES), ids=lambda n: n)
def test_attention_backend_core(name):
    case = CORE_CASES[name]
    skip = _needs_skip(case)
    if skip:
        pytest.skip(skip)
    run_case(case)


# ---------------------------------------------------------------------------
# Sampled product sweep for breadth.
# ---------------------------------------------------------------------------
_SEQ_MIXES = {
    "context": dict(seq_lens=[16, 24], num_cached_tokens=[0, 0], num_contexts=2),
    "decode": dict(seq_lens=[1, 1, 1], num_cached_tokens=[20, 70, 8], num_contexts=0),
    "mixed": dict(seq_lens=[16, 1, 1], num_cached_tokens=[0, 30, 70], num_contexts=1),
}


def _build_sweep_cases():
    cases = []
    grid = itertools.product(
        [(8, 8), (8, 2), (16, 1)],  # (num_heads, num_kv_heads)
        [64, 128],  # head_dim
        ["float16", "bfloat16"],  # dtype
        ["same", "float8_e4m3fn"],  # kv_dtype
        [16, 64],  # page_size
        ["causal", "full", "sliding"],  # mask
        [None, "neox"],  # rope
        list(_SEQ_MIXES),  # seq mix
    )
    for heads, hd, dtype, kvd, page, mask, rope, mix in grid:
        layout = _SEQ_MIXES[mix]
        # Coherence filters.
        if mask == "full" and mix != "context":
            continue  # non-causal only meaningful for prefill here
        num_heads, num_kv_heads = heads
        max_total = max(c + s for c, s in zip(layout["num_cached_tokens"], layout["seq_lens"]))
        sliding = 32 if mask == "sliding" else None
        if sliding is not None and sliding >= max_total:
            continue
        cases.append(
            BackendCase(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=hd,
                dtype=dtype,
                kv_dtype=None if kvd == "same" else kvd,
                causal=(mask != "full"),
                sliding_window=sliding,
                page_size=page,
                rope=_rope_for(hd) if rope == "neox" else None,
                **layout,
            )
        )
    return cases


def _sampled_sweep_cases():
    cases = _build_sweep_cases()
    sample = os.environ.get("TRTLLM_ATTN_TEST_SAMPLE", "80")
    if sample.lower() == "full":
        return cases
    n = min(int(sample), len(cases))
    return random.Random(42).sample(cases, n)


_SWEEP_CASES = _sampled_sweep_cases()


def _sweep_id(case: BackendCase) -> str:
    mask = "full" if not case.causal else ("sliding" if case.sliding_window else "causal")
    rope = "rope" if case.rope else "norope"
    kvd = case.kv_dtype or "samekv"
    return (
        f"h{case.num_heads}kv{case.num_kv_heads}_d{case.head_dim}_"
        f"{case.dtype}_{kvd}_p{case.page_size}_{mask}_{rope}_"
        f"nq{case.nnz_q}_nc{sum(case.num_cached_tokens)}"
    )


@pytest.mark.parametrize("case", _SWEEP_CASES, ids=_sweep_id)
def test_attention_backend_sweep(case):
    skip = _needs_skip(case)
    if skip:
        pytest.skip(skip)
    run_case(case)


# ---------------------------------------------------------------------------
# Split-consistency: a mixed batch run whole must equal the same requests run
# in sub-batches (ports the unique self-consistency check from the existing
# flashinfer/vanilla tests). Catches batch-indexing bugs a golden comparison
# may miss.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM", "FLASHINFER"])
def test_split_consistency(backend):
    from capability_matrix import unsupported_reason

    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

    if backend == "FLASHINFER" and not IS_FLASHINFER_AVAILABLE:
        pytest.skip("flashinfer not available")

    # Two context + two generation requests.
    full = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[16, 24, 1, 1],
        num_cached_tokens=[0, 0, 30, 50],
        num_contexts=2,
    )
    if unsupported_reason(backend, full) is not None:
        pytest.skip(unsupported_reason(backend, full))

    inputs = generate_inputs(full, seed=0)
    out_full = run_backend(full, backend, inputs, kv_dtype=full.compute_dtype)

    # Run request 0 (ctx) + request 2 (gen) as a sub-batch with the SAME inputs.
    # The per-request slices of the packed tensors must reproduce the full run.
    # We validate request 0's context-token outputs are stable under batching.
    sub = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[16, 1],
        num_cached_tokens=[0, 30],
        num_contexts=1,
    )
    # Slice inputs for requests {0, 2}: q tokens [0:16] (ctx0) + [40:41] (gen2).
    q = torch.cat([inputs["q"][0:16], inputs["q"][40:41]], dim=0)
    nk = torch.cat([inputs["new_k"][0:16], inputs["new_k"][40:41]], dim=0)
    nv = torch.cat([inputs["new_v"][0:16], inputs["new_v"][40:41]], dim=0)
    sub_inputs = dict(
        q=q,
        new_k=nk,
        new_v=nv,
        cached_k=[inputs["cached_k"][0], inputs["cached_k"][2]],
        cached_v=[inputs["cached_v"][0], inputs["cached_v"][2]],
    )
    out_sub = run_backend(sub, backend, sub_inputs, kv_dtype=sub.compute_dtype)

    # ctx0 occupies rows [0:16] in both runs; gen2 is row 40 (full) / 16 (sub).
    torch.testing.assert_close(out_sub[0:16], out_full[0:16], atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(out_sub[16:17], out_full[40:41], atol=1e-2, rtol=1e-3)
