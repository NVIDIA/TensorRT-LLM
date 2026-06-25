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
_FLASHINFER_PAGED_APPEND_OVER_1024_THREADS = (
    # bf16/fp16 append launches 16 threads/head for head_dim=128. With 128 KV
    # heads, that exceeds CUDA's 1024 threads/block launch limit.
    (128, 128, "bfloat16"),
    (128, 128, "float16"),
)

_TRTLLM_PAGED_UNSUPPORTED_HEAD_DIMS = (512,)
_TRTLLM_BLACKWELL_PAGED_UNSUPPORTED_HEAD_DIMS = (96,)
_TRTLLM_FAST_BUILD_GEN_UNSUPPORTED_HEAD_DIMS = (96,)
_TRTLLM_BLACKWELL_SLIDING_DECODE_UNSTABLE = (
    # Gemma3-27B local layers and Gemma4-31B sliding layers.
    (32, 16, 128, 1024),
    (32, 16, 256, 1024),
)


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

    # FlashInfer's standard paged path has shape-specific kernel limits in this
    # suite. Keep the skip list explicit so newly supported shapes are not
    # hidden by a broad dispatch-set predicate.
    if (
        backend == "FLASHINFER"
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
    ):
        kv_dtype = getattr(case, "kv_dtype", None) or getattr(case, "dtype", None)
        if case.head_dim in _FLASHINFER_PAGED_UNSUPPORTED_HEAD_DIMS:
            return f"FLASHINFER paged attention is unstable for head_dim {case.head_dim}"
        if (
            case.head_dim,
            case.num_kv_heads,
            kv_dtype,
        ) in _FLASHINFER_PAGED_APPEND_OVER_1024_THREADS:
            return (
                "FLASHINFER paged KV append exceeds CUDA's 1024 threads/block "
                f"limit for head_dim={case.head_dim}, "
                f"num_kv_heads={case.num_kv_heads}, kv_dtype={kv_dtype}"
            )

    # TRTLLM's standard paged FMHA/MMHA kernels in this build do not cover the
    # Gemma4 head_dim 512 path.
    if (
        backend == "TRTLLM"
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
        and case.head_dim in _TRTLLM_PAGED_UNSUPPORTED_HEAD_DIMS
    ):
        return f"TRTLLM paged attention is unstable for head_dim {case.head_dim}"

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
        return f"TRTLLM Blackwell paged attention is unstable for head_dim {case.head_dim}"

    # TRTLLM's Blackwell pure-decode sliding-window path is numerically unstable
    # for the listed Gemma GQA shapes. Mixed batches still use a different path
    # and match the golden.
    if (
        backend == "TRTLLM"
        and sm >= 100
        and getattr(case, "sliding_window", None) is not None
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
        and case.num_contexts == 0
        and (
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.sliding_window,
        )
        in _TRTLLM_BLACKWELL_SLIDING_DECODE_UNSTABLE
    ):
        return (
            "TRTLLM Blackwell sliding-window pure decode is unstable for "
            f"num_heads={case.num_heads}, num_kv_heads={case.num_kv_heads}, "
            f"head_dim={case.head_dim}, window={case.sliding_window}"
        )

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

    # TRTLLM MLA context is validated by test_attention_mla.py through the
    # production MLA module path. This standalone backend harness feeds already
    # up-projected random K/V tensors, which does not match the TRTLLM MLA
    # context op contract and produces invalid output, while Vanilla/FlashInfer
    # can still validate the up-projected context math and cache append.
    if (
        backend == "TRTLLM"
        and getattr(case, "is_mla", False)
        and getattr(case, "num_contexts", 0) == len(getattr(case, "seq_lens", ()))
    ):
        return (
            "TRTLLM MLA context is covered by test_attention_mla.py via the "
            "production MLA module path; standalone up-projected backend "
            "inputs are validated with Vanilla/FlashInfer"
        )

    # On Blackwell, TRTLLM-Gen is the supported MLA generation path. The current
    # kernel set explicitly lacks the DeepSeek-family decode shape at page_size
    # 32, and the generic fallback then tries to JIT an unsupported generated
    # kernel. FlashInfer/Vanilla still validate these standalone MLA gen cases.
    if (
        backend == "TRTLLM"
        and sm >= 100
        and getattr(case, "is_mla", False)
        and getattr(case, "num_contexts", 0) == 0
    ):
        head_dim_qk = getattr(case, "kv_lora_rank", 0) + getattr(case, "qk_rope_head_dim", 0)
        head_dim_v = getattr(case, "kv_lora_rank", 0)
        if (head_dim_qk, head_dim_v, getattr(case, "page_size", None)) == (
            576,
            512,
            32,
        ):
            return (
                "TRTLLM-Gen Blackwell MLA generation is missing the decode "
                "kernel for headDimQk=576, headDimV=512, page_size=32"
            )

    # The TRTLLM MMHA generation kernel is fast-built in this environment and
    # omits the Phi-3 head_dim 96 generation shape. Context FMHA covers that
    # config only on some architectures, so skip cases that include generation.
    if (
        backend == "TRTLLM"
        and getattr(case, "cache", "paged") != "none"
        and not getattr(case, "is_mla", False)
        and case.num_contexts < len(case.seq_lens)
        and case.head_dim in _TRTLLM_FAST_BUILD_GEN_UNSUPPORTED_HEAD_DIMS
    ):
        return f"TRTLLM fast-build MMHA generation is missing head_dim {case.head_dim}"

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
