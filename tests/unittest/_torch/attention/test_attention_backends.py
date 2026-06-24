# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified, detailed test suite for the PyTorch attention backends.

Every case runs the VanillaAttention golden plus each supported backend
(TRTLLM, FlashInfer) through a real KVCacheManager and asserts they match.

The breadth sweep is **model-derived**: it enumerates the distinct attention
configurations actually used by the supported models (``model_configs.py``), so
each case maps to a real workload. The orthogonal dimensions are bounded:

* default cross on every cacheable config: phase {ctx, dec, mix} x
  precision {bf16, fp8-KV} x KV-manager {v1, v2}, at page_size=32, layout=HND.
* the non-default dimension values (page_size=64, layout=NHD, dtype=fp16) are
  exercised on a small representative set (GQA / MHA / MQA) to avoid a full
  cross blow-up -- page/manager/layout are mostly head-config-independent. For
  single-KV-head configs, NHD duplicates HND and is canonicalized to HND.

Gen-only batches are also replayed through a captured CUDA graph; backends that
cannot serve a case (dtype/layout/feature) are skipped via the capability matrix
(``unsupported_reason``), which also gates sm-dependent dtypes.
"""

import pytest
import torch
from attention_test_harness import (
    BACKENDS_UNDER_TEST,
    BackendCase,
    generate_inputs,
    run_backend,
    run_case,
)
from model_configs import MODEL_CONFIGS, ModelAttnConfig

# Precision variants as (dtype, kv_dtype): bf16/fp16 are compute-only; fp8 is an
# fp8 KV cache with bf16 compute.
_BF16 = ("bfloat16", None)
_FP16 = ("float16", None)
_FP8 = ("bfloat16", "float8_e4m3fn")

# Core dimensions crossed on EVERY config; non-default values (page=64, NHD,
# fp16) are added only for the representative configs below.
_CORE_PRECISIONS, _CORE_LAYOUTS, _CORE_PAGES = [_BF16, _FP8], ["HND"], [32]
_EXTRA_PRECISIONS = [_FP16, _BF16, _FP8]
_EXTRA_LAYOUTS, _EXTRA_PAGES = ["HND", "NHD"], [32, 64]
_EXTRA_CONFIG_IDS = ("llama3_gqa", "mha_32x32_hd128", "mqa_synthetic")

# Standard self-attention batch phases.
_PHASES = {
    "ctx": dict(seq_lens=[16, 24, 5], num_cached_tokens=[0, 0, 0], num_contexts=3),
    "gen": dict(seq_lens=[1, 1, 1], num_cached_tokens=[20, 130, 8], num_contexts=0),
    "mix": dict(seq_lens=[16, 1, 1], num_cached_tokens=[0, 30, 70], num_contexts=1),
}


def _rope_dict(cfg: ModelAttnConfig):
    if cfg.rope is None:
        return None
    return dict(dim=cfg.head_dim, theta=10000.0, max_positions=8192, is_neox=(cfg.rope == "neox"))


def _common(cfg: ModelAttnConfig) -> dict:
    common = dict(
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        rope=_rope_dict(cfg),
        causal=cfg.mask in ("causal", "sliding"),
        sliding_window=cfg.sliding_window if cfg.mask == "sliding" else None,
    )
    if cfg.is_mla:
        common.update(
            is_mla=True,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
        )
    return common


def _prec_tag(dtype, kvd) -> str:
    return "fp8" if kvd is not None else ("bf16" if dtype == "bfloat16" else "fp16")


def _expand(cfg: ModelAttnConfig, precisions, kv_layouts, page_sizes):
    """Yield ``(id, BackendCase)`` for one config over the given dimensions.

    ``precisions`` / ``kv_layouts`` / ``page_sizes`` (plus KV-manager v1/v2) are
    the outer loops, shared by the mla-context, mla-generation and standard
    cases. Used to generate both the core slice (default dims on every config)
    and the feature slice (non-default dims on representative configs); callers
    dedup by id. Dimensions that don't apply to a case type are filtered:
    no-cache encoders take only the compute-dtype precisions; MLA uses bf16
    latent cache; cross skips fp8. Single-KV-head configs use only HND because
    NHD is physically equivalent and would skip TRTLLM coverage.
    """
    common = _common(cfg)

    # Bidirectional, KV-cache-free DiT / encoder workloads: only compute dtype.
    if cfg.no_cache:
        for dtype, kvd in precisions:
            if kvd is not None:
                continue  # no KV cache to quantize
            yield (
                f"{cfg.id}-{_prec_tag(dtype, kvd)}",
                BackendCase(
                    cache="none",
                    seq_lens=[16, 24, 5],
                    num_cached_tokens=[0, 0, 0],
                    num_contexts=3,
                    dtype=dtype,
                    **common,
                ),
            )
        return

    # Filter dims to those meaningful for this case type.
    if cfg.is_mla:
        precisions = [(d, k) for d, k in precisions if k is None]  # bf16 latent
    elif cfg.is_cross:
        precisions = [(d, k) for d, k in precisions if k is None]  # no fp8 cross
    if cfg.num_kv_heads == 1:
        kv_layouts = ["HND"]

    for dtype, kvd in precisions:
        for layout in kv_layouts:
            for page in page_sizes:
                for v2 in (False, True):
                    tag = f"{_prec_tag(dtype, kvd)}-{layout}-p{page}-{'v2' if v2 else 'v1'}"
                    base = dict(
                        page_size=page,
                        kv_layout=layout,
                        dtype=dtype,
                        kv_dtype=kvd,
                        use_kv_cache_manager_v2=v2,
                        **common,
                    )
                    if cfg.is_cross:
                        yield (
                            f"{cfg.id}-same_kv_{tag}",
                            BackendCase(
                                seq_lens_kv=_PHASES["ctx"]["seq_lens"], **_PHASES["ctx"], **base
                            ),
                        )
                        yield (
                            f"{cfg.id}-diff_kv_{tag}",
                            BackendCase(
                                seq_lens_kv=[
                                    x * y
                                    for x, y in zip(
                                        _PHASES["ctx"]["seq_lens"], [2, 3, 6], strict=True
                                    )
                                ],
                                **_PHASES["ctx"],
                                **base,
                            ),
                        )
                    elif cfg.is_mla and cfg.mla_context:
                        yield f"{cfg.id}-ctx-{tag}", BackendCase(**_PHASES["ctx"], **base)
                    elif cfg.is_mla:
                        yield f"{cfg.id}-gen-{tag}", BackendCase(**_PHASES["gen"], **base)
                    else:
                        for phase_name, phase in _PHASES.items():
                            yield f"{cfg.id}-{phase_name}-{tag}", BackendCase(**phase, **base)


def _model_cases():
    cases = {}
    cfg_by_id = {cfg.id: cfg for cfg in MODEL_CONFIGS}
    # Core slice on every config, then the feature slice (non-default page /
    # layout / precision) on the representative configs.
    for cfg in MODEL_CONFIGS:
        for case_id, case in _expand(cfg, _CORE_PRECISIONS, _CORE_LAYOUTS, _CORE_PAGES):
            cases[case_id] = case
    for cfg_id in _EXTRA_CONFIG_IDS:
        for case_id, case in _expand(
            cfg_by_id[cfg_id], _EXTRA_PRECISIONS, _EXTRA_LAYOUTS, _EXTRA_PAGES
        ):
            cases[case_id] = case
    return cases


MODEL_CASES = _model_cases()


@pytest.mark.parametrize("name", list(MODEL_CASES), ids=lambda n: n)
def test_attention_backend_model(name):
    run_case(MODEL_CASES[name])


# ---------------------------------------------------------------------------
# Split-consistency: a mixed batch run whole must equal the same requests run
# in sub-batches (ports the unique self-consistency check from the existing
# flashinfer/vanilla tests). Catches batch-indexing bugs a golden comparison
# may miss.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["VANILLA", *BACKENDS_UNDER_TEST])
def test_split_consistency(backend):
    from capability_matrix import unsupported_reason

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
