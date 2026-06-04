# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-backend feature-support matrix for the unified attention test suite.

Every backend-under-test is compared against the VanillaAttention golden, but
backends support different feature subsets. A case that exercises a feature a
backend lacks is skipped (with a clear reason) rather than failing.

Values are seeded from the backends' ``support_*`` classmethods and known
asserts (e.g. ``TrtllmAttention`` rejects cross-attention) and are sanity-checked
by :func:`verify_against_backends` in the synthetic suite.
"""

from typing import Optional

# Feature keys:
#   paged          - reads a paged KV cache (vs a single contiguous block)
#   fp8_kv         - FP8 (e4m3) KV cache
#   fp4_kv         - NVFP4 KV cache (Blackwell only)
#   sliding_window - sliding-window attention via attention_window_size
#   no_cache       - ragged/prefill forward with kv_cache_manager=None
#   sparse         - sparse-attention forward plumbing (degenerate regime here)
#   mla            - multi-head latent attention
BACKEND_CAPS = {
    "TRTLLM": dict(
        paged=True,
        fp8_kv=True,
        fp4_kv=True,
        sliding_window=True,
        no_cache=True,
        sparse=True,
        mla=True,
    ),
    "FLASHINFER": dict(
        paged=True,
        fp8_kv=True,
        fp4_kv=False,
        sliding_window=True,
        no_cache=True,
        sparse=False,
        mla=True,
    ),
    "VANILLA": dict(
        paged=False,
        fp8_kv=True,
        fp4_kv=False,
        sliding_window=True,
        no_cache=True,
        sparse=False,
        mla=False,
    ),
}


def required_features(case) -> set:
    """Return the set of capability keys a case requires of any backend."""
    feats = set()
    # ``case.kv_dtype`` is a string (or None to mirror the compute dtype).
    kv_dtype = getattr(case, "kv_dtype", None)
    if kv_dtype == "float8_e4m3fn":
        feats.add("fp8_kv")
    elif kv_dtype == "nvfp4":
        feats.add("fp4_kv")
    if getattr(case, "sliding_window", None):
        feats.add("sliding_window")
    if getattr(case, "cache", "paged") == "none":
        feats.add("no_cache")
    if getattr(case, "sparse", "off") != "off":
        feats.add("sparse")
    if getattr(case, "is_mla", False):
        feats.add("mla")
    return feats


def unsupported_reason(backend: str, case) -> Optional[str]:
    """Return why ``backend`` cannot run ``case``, or ``None`` if it can.

    Usable outside pytest (e.g. by the minimizer).
    """
    caps = BACKEND_CAPS[backend]
    for feat in required_features(case):
        if not caps.get(feat, False):
            return f"{backend} does not support feature '{feat}'"

    # Known limitation: TRTLLM's XQA fp8 decode kernel produces NaN when reading
    # a *manually* prefilled fp8 KV cache (the scale/layout state it expects is
    # only set up by the kernel's own append path, not by a test-side prefill).
    # fp8 prefill/context works and is still exercised. Generation-phase fp8
    # cases skip TRTLLM and validate FlashInfer fp8 decode instead.
    if backend == "TRTLLM" and getattr(case, "kv_dtype", None) == "float8_e4m3fn":
        has_generation = case.num_contexts < len(case.seq_lens)
        if has_generation:
            return (
                "TRTLLM fp8 KV decode with a manual cache prefill is "
                "unsupported (XQA NaN); fp8 context is covered separately"
            )
    return None


def skip_if_unsupported(backend: str, case) -> None:
    """``pytest.skip`` if ``backend`` cannot run ``case``."""
    reason = unsupported_reason(backend, case)
    if reason is not None:
        import pytest

        pytest.skip(reason)
