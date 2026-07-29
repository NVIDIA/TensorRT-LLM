# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-backend feature-support matrix for the unified attention test suite.

Every backend-under-test is compared against the VanillaAttention golden, but
backends support different feature subsets. A case that exercises a feature a
backend lacks is skipped (with a clear reason) rather than failing.

Values are seeded from the backends' ``support_*`` classmethods and known
backend-specific limitations that this standalone harness must skip.
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
#   cross_attn     - cross-attention (encoder-decoder)
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
        mla=True,
        cross_attn=True,
        kv_layouts=("NHD",),  # reads the NHD get_buffers view
    ),
}

_FLASHINFER_PAGED_UNSUPPORTED_HEAD_DIMS = (96, 512)

_TRTLLM_PAGED_UNSUPPORTED_HEAD_DIMS = (512,)
_TRTLLM_BLACKWELL_PAGED_UNSUPPORTED_HEAD_DIMS = (96,)


def required_features(case) -> set:
    """Return the set of capability keys a case requires of any backend."""
    feats = set()
    # ``case.kv_dtype`` is a string (or None to mirror the compute dtype).
    kv_dtype = getattr(case, "kv_dtype", None)
    if kv_dtype == "fp8":
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
    if kv_dtype == "fp8" and sm < 89:
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

    # FlashInfer's standard paged kernels reject these head dimensions. Keep
    # the skip list explicit so newly supported shapes are not hidden by a
    # broad dispatch-set predicate.
    if (
        backend == "FLASHINFER"
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
        and case.head_dim in _FLASHINFER_PAGED_UNSUPPORTED_HEAD_DIMS
    ):
        return f"FLASHINFER paged kernels do not support head_dim {case.head_dim}"

    # TRTLLM's standard paged FMHA/MMHA kernels in this build do not cover the
    # Gemma4 head_dim 512 path.
    if (
        backend == "TRTLLM"
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
        and case.head_dim in _TRTLLM_PAGED_UNSUPPORTED_HEAD_DIMS
    ):
        return f"TRTLLM paged attention does not support head_dim {case.head_dim}"

    # TRTLLM's Blackwell paged fallback path aborts for the Phi-3 head_dim 96
    # shape. Hopper covers that context config, but Blackwell must skip it
    # before entering the fused op.
    if (
        backend == "TRTLLM"
        and sm >= 100
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
        and case.head_dim in _TRTLLM_BLACKWELL_PAGED_UNSUPPORTED_HEAD_DIMS
    ):
        return f"TRTLLM Blackwell paged attention does not support head_dim {case.head_dim}"

    # TRTLLM's Blackwell no-cache fallback mismatches the Vanilla golden for the
    # Qwen2-VL vision tower's head_dim 80 workload; other no-cache head dims in
    # this sweep still pass on Blackwell.
    if (
        backend == "TRTLLM"
        and sm >= 100
        and getattr(case, "cache", "paged") == "none"
        and not getattr(case, "is_mla", False)
        and case.head_dim == 80
    ):
        return "TRTLLM Blackwell no-cache fallback is unstable for head_dim 80"

    return None
