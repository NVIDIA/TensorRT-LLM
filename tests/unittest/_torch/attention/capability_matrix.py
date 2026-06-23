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

from utils.util import getSMVersion

# Feature keys:
#   paged          - reads a paged KV cache (vs a single contiguous block)
#   fp8_kv         - FP8 (e4m3) KV cache
#   fp4_kv         - NVFP4 KV cache (Blackwell only)
#   sliding_window - sliding-window attention via attention_window_size
#   no_cache       - ragged/prefill forward with kv_cache_manager=None
#   sparse         - sparse-attention forward plumbing (degenerate regime here)
#   mla            - multi-head latent attention
#   cross_attn     - cross-attention (encoder-decoder; seq_lens_kv != seq_lens)
#   kv_layouts     - supported paged-cache block layouts ("NHD" / "HND")
BACKEND_CAPS = {
    # NOTE on fp4_kv=False: NVFP4 KV cache cannot be exercised in this standalone
    # backend harness. (1) The NVFP4 attention op aborts without the per-tensor /
    # block scale tensors the Attention module feeds it from its projection
    # layers (qkv_proj.kv_scales), which a bare backend lacks. (2) The packed
    # e2m1 + swizzled block-scale cache cannot be manually prefilled
    # (get_buffers cannot even reshape the half-size packed pool). fp4 coverage
    # belongs at the model level via the capture hook on a real fp4 model run.
    "TRTLLM": dict(
        paged=True,
        fp8_kv=True,
        fp4_kv=False,
        sliding_window=True,
        no_cache=True,
        sparse=True,
        mla=True,
        cross_attn=True,
        kv_layouts=("HND",),  # paged FMHA/XQA is fixed to head-major
    ),
    "FLASHINFER": dict(
        paged=True,
        fp8_kv=True,
        fp4_kv=False,
        sliding_window=True,
        no_cache=True,
        sparse=False,
        mla=True,
        cross_attn=True,
        kv_layouts=("NHD", "HND"),  # selectable via metadata.kv_layout
    ),
    "VANILLA": dict(
        paged=False,
        fp8_kv=True,
        fp4_kv=False,
        sliding_window=True,
        no_cache=True,
        sparse=False,
        mla=False,
        cross_attn=True,
        kv_layouts=("NHD",),  # reads the NHD get_buffers view
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
    if getattr(case, "is_cross", False):
        feats.add("cross_attn")
    return feats


def unsupported_reason(backend: str, case) -> Optional[str]:
    """Return why ``backend`` cannot run ``case``, or ``None`` if it can.

    Usable outside pytest (e.g. by the minimizer).
    """
    # Hardware dtype gating (arch-level; applies to every backend under test).
    # The Vanilla golden runs in compute dtype so it is unaffected.
    sm = getSMVersion()
    kv_dtype = getattr(case, "kv_dtype", None)
    if kv_dtype == "float8_e4m3fn" and sm < 89:
        return f"FP8 KV cache requires sm>=89 (have sm{sm})"
    if kv_dtype == "nvfp4" and sm < 100:
        return f"NVFP4 KV cache requires sm>=100/Blackwell (have sm{sm})"

    caps = BACKEND_CAPS[backend]
    for feat in required_features(case):
        if not caps.get(feat, False):
            return f"{backend} does not support feature '{feat}'"

    # KV-cache block layout: a case may request a specific layout (NHD/HND). A
    # backend that cannot store the cache that way is skipped (e.g. TRTLLM is
    # head-major HND only). The Vanilla golden always runs in its native NHD and
    # is not gated here (run_case drives it directly, not via this check).
    kv_layout = getattr(case, "kv_layout", None)
    if kv_layout is not None and kv_layout not in caps.get("kv_layouts", ()):
        return f"{backend} does not support kv_layout '{kv_layout}'"

    # TRTLLM's fp8 generation (XQA) path computes in fp8 and needs the model's
    # real KV-dequant + output scale state (fed by the Attention module's
    # projection layers). A bare standalone backend lacks it: supplying a unit
    # output scale lets the kernel run, but the MHA path then produces garbage
    # (~1e5 abs error) while only GQA happens to tolerate it. fp8 *context*
    # (pure prefill) is exercised; FlashInfer validates fp8 decode on every arch.
    if backend == "TRTLLM" and getattr(case, "kv_dtype", None) == "float8_e4m3fn":
        has_generation = case.num_contexts < len(case.seq_lens)
        if has_generation:
            return (
                "TRTLLM fp8 KV generation (XQA) needs the model's fp8 scale "
                "state, absent in the standalone backend; fp8 context is covered "
                "and FlashInfer validates fp8 decode"
            )

    # TRTLLM fuses RoPE for MLA (support_fused_rope), so its absorbed-generation
    # invocation differs (RoPE applied inside mla_rope_generation, with cu_seqlens
    # / scheduler buffers). This unified harness exercises the *non-fusion* MLA
    # path shared by Vanilla (golden) and FlashInfer; TRTLLM MLA is validated in
    # test_attention_mla.py instead.
    if backend == "TRTLLM" and getattr(case, "is_mla", False):
        return (
            "TRTLLM MLA uses fused RoPE (distinct invocation); validated via "
            "test_attention_mla.py. This harness checks the non-fusion MLA path "
            "(Vanilla golden vs FlashInfer)"
        )

    # TRTLLM cross-attention works through the standard plumbing (prepare()
    # derives the cross kv_lens from seq_lens_kv; the backend builds cross_kv from
    # k/v). The aligned q_len == kv_len case is exercised on TRTLLM. The
    # q_len != kv_len case is numerically correct in isolation (verified vs the
    # golden to ~2e-3) but exhibits a cross-case state-dependent mismatch when run
    # after other cross cases in the same process (a TRTLLM cross-path global not
    # reset between fresh backend instances), so it is validated via
    # FlashInfer/Vanilla in the suite.
    if (
        backend == "TRTLLM"
        and getattr(case, "is_cross", False)
        and list(case.seq_lens) != list(case.seq_lens_kv)
    ):
        return (
            "TRTLLM cross q_len != kv_len is correct standalone (~2e-3) but "
            "flaky across cross cases in-process; validated via FlashInfer/Vanilla"
        )
    return None
