# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified, detailed test suite for the PyTorch attention backends.

Every case runs the VanillaAttention golden plus each supported backend
(TRTLLM, FlashInfer) through a real KVCacheManager and asserts they match.

The breadth sweep is **model-derived**: it enumerates the distinct attention
configurations actually used by the supported models (``model_attn_config.py``), so
each case maps to a real workload. The orthogonal dimensions are bounded:

* default cross on every cacheable config: phase {ctx, gen, mix} x
  precision {bf16, fp8-KV} x KV-manager {v1, v2}, at page_size=32, layout=HND.
  T5/BART encoder-decoder cross-attention uses V2 only.
* the non-default dimension values (page_size=64, layout=NHD, dtype=fp16) are
  exercised on a small representative set (GQA / MHA / MQA) to avoid a full
  cross blow-up -- page/manager/layout are mostly head-config-independent. For
  single-KV-head configs, NHD duplicates HND and is canonicalized to HND.

Gen-only batches are also replayed through a captured CUDA graph; backends that
cannot serve a case (dtype/layout/feature) are skipped via the capability matrix
(``unsupported_reason``), which also gates sm-dependent dtypes.
"""

import backend_case
import pytest
import torch
from backend_capability import unsupported_reason
from backend_case import (
    BACKENDS_UNDER_TEST,
    BF16_ATOL,
    BF16_RTOL,
    FP8_ATOL,
    BackendCase,
    generate_inputs,
    generate_mla_context_inputs,
    run_backend,
    run_case,
)
from model_attn_config import MODEL_CONFIGS, ModelAttnConfig
from utils.util import getSMVersion

from tensorrt_llm._utils import get_sm_version, is_sm_120f

# Precision variants as (dtype, kv_dtype): bf16/fp16 are compute-only; fp8 is an
# fp8 KV cache with bf16 compute.
_BF16 = ("bfloat16", None)
_FP16 = ("float16", None)
_FP8 = ("bfloat16", "fp8")

# Core dimensions crossed on EVERY config; non-default values (page=64, NHD,
# fp16) are added only for the representative configs below.
_CORE_PRECISIONS, _CORE_LAYOUTS, _CORE_PAGES = [_BF16, _FP8], ["HND"], [32]
_EXTRA_PRECISIONS = [_FP16, _BF16, _FP8]
_EXTRA_LAYOUTS, _EXTRA_PAGES = ["HND", "NHD"], [32, 64]
_EXTRA_CONFIG_IDS = (
    "llama3_8b_gqa",
    "llama2_7b_decilm_7b_mha",
    "gemma3_1b_mqa_hd256",
)

_NON_SLIDING_PHASE_WINDOW = 1024


def get_long_seq_len(window: int) -> int:
    return window + 17


def _phases(long_len: int, gen_len: int) -> dict:
    """The ctx/gen/mix batch shapes, parameterized by context and generation len."""
    return {
        "ctx": dict(
            seq_lens=[long_len, 73, 41],
            num_cached_tokens=[0, 0, 0],
            num_contexts=3,
        ),
        "gen": dict(
            seq_lens=[gen_len] * 3,
            num_cached_tokens=[long_len, 73, 41],
            num_contexts=0,
        ),
        "mix": dict(
            seq_lens=[long_len, gen_len, gen_len],
            num_cached_tokens=[0, long_len, 73],
            num_contexts=1,
        ),
    }


def _phases_from_window(window: int) -> dict:
    return _phases(get_long_seq_len(window), gen_len=1)


# Standard self-attention batch phases. Non-sliding cases use a nominal window
# only to choose non-tiny, non-power-of-two lengths; the backend still receives
# sliding_window=None.
_PHASES = _phases_from_window(_NON_SLIDING_PHASE_WINDOW)


# Model-agnostic sparse sweep dimensions, shared by every sparse config.
_SPARSE_COMPUTE_DTYPE = "bfloat16"
_SPARSE_KV_LAYOUT = "HND"
_SPARSE_PAGE_SIZE = 64


def _phases_for(cfg: ModelAttnConfig) -> dict:
    if cfg.sparse_attention_config is not None:
        return _phases(cfg.sparse_topk + 32, gen_len=1)
    if cfg.mask != "sliding":
        return _PHASES

    assert cfg.sliding_window is not None
    return _phases_from_window(cfg.sliding_window)


def _phases_to_run(cfg: ModelAttnConfig, available_phases: dict) -> tuple[str, ...]:
    return tuple(available_phases) if cfg.phases is None else cfg.phases


def _rope_dict(cfg: ModelAttnConfig):
    if cfg.rope is None:
        return None
    max_positions = 8192
    if cfg.sliding_window is not None:
        max_positions = max(max_positions, get_long_seq_len(cfg.sliding_window) + 1)
    return dict(
        dim=cfg.head_dim, theta=10000.0, max_positions=max_positions, is_neox=(cfg.rope == "neox")
    )


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
            hidden_size=cfg.hidden_size,
        )
    if cfg.sparse_attention_config is not None:
        common.update(sparse_attention_config=cfg.sparse_attention_config)
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
    phases = _phases_for(cfg)

    # Sparse cases use one model-agnostic sweep (bf16 latent cache, fixed layout/
    # page, both v1 and v2 KV cache managers). Every backend runs its own
    # indexer over shared weights, so a case covers selection and execution
    # together; the phase lengths (top-k + 32) keep every long row genuinely sparse.
    if cfg.sparse_attention_config is not None:
        for v2 in (False, True):
            manager = "v2" if v2 else "v1"
            tag = (
                f"{_prec_tag(_SPARSE_COMPUTE_DTYPE, None)}-{_SPARSE_KV_LAYOUT}"
                f"-p{_SPARSE_PAGE_SIZE}-{manager}"
            )
            for phase_name in ("ctx", "gen", "mix"):
                yield (
                    f"{cfg.id}-{phase_name}-{tag}",
                    BackendCase(
                        page_size=_SPARSE_PAGE_SIZE,
                        kv_layout=_SPARSE_KV_LAYOUT,
                        dtype=_SPARSE_COMPUTE_DTYPE,
                        use_kv_cache_manager_v2=v2,
                        **phases[phase_name],
                        **common,
                    ),
                )
        return

    # Bidirectional, KV-cache-free DiT / encoder workloads: only compute dtype.
    if cfg.no_cache:
        for dtype, kvd in precisions:
            if kvd is not None:
                continue  # no KV cache to quantize
            yield (
                f"{cfg.id}-{_prec_tag(dtype, kvd)}",
                BackendCase(
                    cache="none",
                    **phases["ctx"],
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
                    if cfg.is_cross and cfg.id != "mllama_gqa_cross" and not v2:
                        continue
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
                                seq_lens_kv=phases["ctx"]["seq_lens"], **phases["ctx"], **base
                            ),
                        )
                        yield (
                            f"{cfg.id}-diff_kv_{tag}",
                            BackendCase(
                                seq_lens_kv=[
                                    x * y
                                    for x, y in zip(
                                        phases["ctx"]["seq_lens"], [2, 3, 6], strict=True
                                    )
                                ],
                                **phases["ctx"],
                                **base,
                            ),
                        )
                    else:
                        for phase_name in _phases_to_run(cfg, phases):
                            yield (
                                f"{cfg.id}-{phase_name}-{tag}",
                                BackendCase(**phases[phase_name], **base),
                            )


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
def test_attention_backend(name):
    case = MODEL_CASES[name]
    if case.is_sparse and getSMVersion() < 100:
        # The backend matrix includes the TRTLLM trtllm-gen sparse FMHA, which
        # is Blackwell-only. The Vanilla reference itself remains usable below
        # SM100 through the module's FlashMLA dispatch.
        pytest.skip(f"DSA requires sm>=100/Blackwell (have sm{getSMVersion()})")
    run_case(case)


def _mla_context_v_stride_case() -> BackendCase:
    case = next(
        (
            case
            for name, case in MODEL_CASES.items()
            if name.startswith("hunyuan_mla_ctx-ctx-")
            and case.dtype == "bfloat16"
            and case.kv_dtype is None
        ),
        None,
    )
    if case is None:
        pytest.skip("no bf16 hunyuan_mla_ctx case")
    if get_sm_version() < 90:
        pytest.skip("requires SM90 or newer context FMHA")
    if reason := unsupported_reason("TRTLLM", case):
        pytest.skip(reason)
    return case


@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8_compute"]
)
def test_mla_context_v_token_stride_matches_both_input_layouts(kv_dtype: torch.dtype) -> None:
    """Check separate-V layouts through BF16 FMHA and the FP8 context input quantizer."""
    case = _mla_context_v_stride_case()
    fp8_context = kv_dtype == torch.float8_e4m3fn
    if fp8_context and not (get_sm_version() in (90, 100, 103, 107) or is_sm_120f()):
        pytest.skip("requires a GPU supporting FP8 context MLA")
    inputs = generate_mla_context_inputs(case)
    assert inputs["v"].stride(0) > inputs["v"].size(1)
    expected = run_backend(case, "VANILLA", inputs, kv_dtype=case.compute_dtype, kv_layout="NHD")
    outputs = []

    for make_contiguous in (False, True):
        current_inputs = dict(inputs)
        if make_contiguous:
            current_inputs["v"] = current_inputs["v"].contiguous()
        actual = run_backend(
            case, "TRTLLM", current_inputs, kv_dtype=kv_dtype, kv_layout=case.kv_layout
        )
        atol = BF16_ATOL + FP8_ATOL if fp8_context else BF16_ATOL
        torch.testing.assert_close(actual, expected, atol=atol, rtol=BF16_RTOL)
        outputs.append(actual)

    torch.testing.assert_close(outputs[0], outputs[1], atol=BF16_ATOL, rtol=BF16_RTOL)


@pytest.mark.parametrize(
    ("layout", "error"),
    [
        ("feature_stride", "contiguous V features within each token"),
        ("token_stride", "16-byte aligned V token stride"),
        ("pointer", "16-byte aligned V pointer"),
    ],
)
def test_mla_context_v_layout_rejected(layout: str, error: str) -> None:
    case = _mla_context_v_stride_case()
    inputs = generate_mla_context_inputs(case)
    v = inputs["v"]
    if layout == "feature_stride":
        invalid_v = torch.empty(v.size(0), v.size(1) * 2, dtype=v.dtype, device="cuda")[:, ::2]
    elif layout == "token_stride":
        invalid_v = torch.empty_strided(v.shape, (v.size(1) + 1, 1), dtype=v.dtype, device="cuda")
    else:
        storage = torch.empty(v.numel() + 1, dtype=v.dtype, device="cuda")
        invalid_v = storage[1:].view(v.shape)
    invalid_v.copy_(v)
    with pytest.raises(RuntimeError, match=error):
        run_backend(
            case,
            "TRTLLM",
            dict(inputs, v=invalid_v),
            kv_dtype=case.compute_dtype,
            kv_layout=case.kv_layout,
        )


# ---------------------------------------------------------------------------
# Backend-specific regressions that require a non-default implementation path
# or workload shape and therefore do not belong in the cross-backend model
# matrix above.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="Kimi K3 TRTLLM-Gen MLA regression requires SM100 or SM103",
)
@pytest.mark.parametrize("page_size", [32, 64], ids=["p32", "p64"])
@pytest.mark.parametrize("seq_len_q", [1, 3], ids=["q1", "q3"])
def test_trtllm_mla_kimi_k3_h96(
    monkeypatch: pytest.MonkeyPatch,
    page_size: int,
    seq_len_q: int,
) -> None:
    """Verify K3 H96 MLA generation numerics in eager and CUDA Graph execution."""
    batch_size = 32
    case = BackendCase(
        num_heads=96,
        num_kv_heads=1,
        head_dim=192,
        seq_lens=[seq_len_q] * batch_size,
        num_cached_tokens=[31] * batch_size,
        num_contexts=0,
        dtype="bfloat16",
        page_size=page_size,
        kv_layout="HND",
        is_mla=True,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    monkeypatch.setattr(backend_case, "BACKENDS_UNDER_TEST", ("TRTLLM",))
    monkeypatch.setenv("TLLM_FMHA_LIBS", "flashinfer_trtllm_gen,fallback")
    backend_case.run_case(case)


# ---------------------------------------------------------------------------
# Split-consistency: a mixed batch run whole must equal the same requests run
# in sub-batches (ports the unique self-consistency check from the existing
# flashinfer/vanilla tests). Catches batch-indexing bugs a golden comparison
# may miss.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["VANILLA", *BACKENDS_UNDER_TEST])
def test_split_consistency(backend):
    from backend_capability import unsupported_reason

    # Two context + two generation requests.
    ctx_lens = _PHASES["ctx"]["seq_lens"][:2]
    gen_cached_lens = _PHASES["gen"]["num_cached_tokens"][:2]
    kv_layout = "NHD" if backend == "VANILLA" else "HND"
    full = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[*ctx_lens, 1, 1],
        num_cached_tokens=[0, 0, *gen_cached_lens],
        num_contexts=2,
        kv_layout=kv_layout,
    )
    if unsupported_reason(backend, full) is not None:
        pytest.skip(unsupported_reason(backend, full))

    inputs = generate_inputs(full, seed=0)
    out_full = run_backend(full, backend, inputs, kv_dtype=full.compute_dtype, kv_layout=kv_layout)

    # Run request 0 (ctx) + request 2 (gen) as a sub-batch with the SAME inputs.
    # The per-request slices of the packed tensors must reproduce the full run.
    q_offsets = [0, *torch.tensor(full.seq_lens).cumsum(0).tolist()]
    sub = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[ctx_lens[0], 1],
        num_cached_tokens=[0, gen_cached_lens[0]],
        num_contexts=1,
        kv_layout=kv_layout,
    )
    # Slice inputs for requests {0, 2}: context request 0 plus generation
    # request 0 after the two contexts.
    q = torch.cat(
        [inputs["q"][q_offsets[0] : q_offsets[1]], inputs["q"][q_offsets[2] : q_offsets[3]]],
        dim=0,
    )
    nk = torch.cat(
        [
            inputs["new_k"][q_offsets[0] : q_offsets[1]],
            inputs["new_k"][q_offsets[2] : q_offsets[3]],
        ],
        dim=0,
    )
    nv = torch.cat(
        [
            inputs["new_v"][q_offsets[0] : q_offsets[1]],
            inputs["new_v"][q_offsets[2] : q_offsets[3]],
        ],
        dim=0,
    )
    sub_inputs = dict(
        q=q,
        new_k=nk,
        new_v=nv,
        cached_k=[inputs["cached_k"][0], inputs["cached_k"][2]],
        cached_v=[inputs["cached_v"][0], inputs["cached_v"][2]],
    )
    out_sub = run_backend(sub, backend, sub_inputs, kv_dtype=sub.compute_dtype, kv_layout=kv_layout)

    ctx0_end = ctx_lens[0]
    gen_full_start = q_offsets[2]
    gen_sub_start = ctx0_end
    torch.testing.assert_close(out_sub[:ctx0_end], out_full[:ctx0_end], atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(
        out_sub[gen_sub_start : gen_sub_start + 1],
        out_full[gen_full_start : gen_full_start + 1],
        atol=1e-2,
        rtol=1e-3,
    )
