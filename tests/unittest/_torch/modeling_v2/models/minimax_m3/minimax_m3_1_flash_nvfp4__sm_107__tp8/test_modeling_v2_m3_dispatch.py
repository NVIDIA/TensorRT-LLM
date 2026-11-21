# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 (Rubin, two trays) decode target's fast-path table (``modeling.py``: the row ranges and the selection
functions): its row ranges against the catalog cells that certify them, and the selection for every decode row count.

Every range's covered shapes are listed in ``ENABLED``; a range not listed covers nothing.

CPU only: nothing here launches a kernel.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling as dispatch,
)

ROWS = range(1, dispatch.DECODE_STEP_MAX + 2)  # every decode row count and the first one past it


@pytest.fixture(autouse=True)
def _policy():
    """Every test starts outside require mode with nothing logged, and leaves it so."""
    dispatch.set_require(False)
    dispatch.reset_logged()
    yield
    dispatch.set_require(False)
    dispatch.reset_logged()


@pytest.mark.parametrize("fast_path", dispatch.FAST_PATHS, ids=lambda r: r.path)
def test_every_covered_shape_is_a_certified_cell(fast_path) -> None:
    uncertified = fast_path.keys - fast_path.certified()
    assert not uncertified, (
        f"{fast_path.path} covers {sorted(uncertified, key=str)} but "
        f"{type(fast_path.entry).__name__}.CELLS certifies none of them: add a cell or narrow the range"
    )


# ------------------------------------------------------------------ the cells turned on


def test_the_moe_engine_takes_every_decode_step() -> None:
    """The m3_moe engine (TP8 width 384, shared expert folded in) at 1..8 rows, unless killed; its sum
    pushes into the multicast boundary all-reduce at the engine's rows while that all-reduce is on."""
    expected = set() if dispatch.KILL_MOE_ENGINE else set(range(1, 9))
    assert set(dispatch.MOE_ENGINE.keys) == expected
    for rows in ROWS:
        want = dispatch.MOE_ENGINE.path if rows in expected else None
        assert dispatch.moe_path(rows) == want, rows
    push_rows = dispatch.MOE_ENGINE_MAX_TOKENS if dispatch.MC_AR_TP2 else 0
    assert dispatch.MOE_PUSH_MAX_TOKENS == push_rows
    assert set(dispatch.MOE_PUSH.keys) == set(range(1, push_rows + 1))


# ------------------------------------------------------------------ on: the sparse layers' block selection


#: Every TP8 per-rank MXFP8 projection (N, K): sparse-layer QKV + index, dense QKV, o_proj, shared gate_up / down,
#: dense gate_up / down.
TP8_PROJECTIONS = (
    (1536, 6144),
    (1280, 6144),
    (6144, 1024),
    (768, 6144),
    (6144, 384),
    (3072, 6144),
    (6144, 1536),
)
# R1VR8: the step bound is 32 rows; the TP2-lineage paths keep their 8 certified rows;
# on an MXFP8 activation the GEMV serves the QKV and both gate_up projections up to 32 rows and the o_proj up to 16
# (the downs, bf16 input: 8). TRTLLM_M31_DISABLE_DECODE_LAYER_WIDE_ROWS=1: every range is 8 rows.
_STEP_MAX = 32 if dispatch.R1VR8 else 8
_DECODE_ROWS = range(1, 8 + 1)  # the TP2-lineage rows
_STEP_ROWS = range(1, _STEP_MAX + 1)
_GEMV_MAX = dict.fromkeys(TP8_PROJECTIONS, 8)
# Past the decode bound (VR_GEMV48 / LINEAR_CUBLASLT): the generation steps of 33..64 rows.
_WIDE_ROWS = range(max(33, dispatch.DECODE_MAX_TOKENS + 1), 64 + 1)
_GEMV48_ROWS = range(max(33, dispatch.DECODE_MAX_TOKENS + 1), 48 + 1)
_GEMV48_SHAPES = (
    (1536, 6144),
    (1280, 6144),
    (768, 6144),
)  # sparse / dense-attention QKV, shared gate_up
_CUBLASLT_SHAPES = (
    (1536, 6144),
    (6144, 1024),
    (768, 6144),
    (6144, 384),
    (3072, 6144),
    (6144, 1536),
)
if dispatch.R1VR8:
    _GEMV_MAX.update(
        {(1536, 6144): 32, (1280, 6144): 32, (768, 6144): 32, (3072, 6144): 32, (6144, 1024): 16}
    )

#: Fast paths turned on, each with the shapes it covers (the rest of the skeleton stays off).
ENABLED = {
    # Sampled (temperature + top-p) verify of 1-2 requests at draft length 3 (the shard kernel).
    "m3_sampled_accept": frozenset({(3, 1), (3, 2)}),
    # Greedy verify (shard argmax + accept) and the fused draft loop of 1-2 requests at draft length 3,
    # and that loop's draft-input rows (1 / 2 later-pass rows, 4 / 8 first-pass rows).
    "m3_shard_accept": frozenset({(3, 1), (3, 2)}),
    "m3_draft_step": frozenset({(3, 1), (3, 2)}),
    "m3_draft_input": frozenset({1, 2, 4, 8}),
    # The drafter's BF16 linears on FlashInfer's low-M kernels (TP8 port) at the certified b1 rows.
    "m3_draft_lowm": frozenset(
        {
            ((1280, 12288), 1),
            ((1280, 12288), 4),
            ((6144, 1024), 1),
            ((6144, 1024), 4),
            ((4608, 6144), 1),
            ((4608, 6144), 4),
            ((6144, 2304), 1),
            ((6144, 2304), 4),
            ((6144, 18432), 1),
        }
    ),
    # The fused first drafter inputs of 1-2 fused-verify requests, the capture norm and the chunk norm at
    # their 4 / 8 verify rows (own ranges and kills since the TP2 fast-path audit).
    "m3_fused_first_inputs": frozenset({1, 2}),
    "m3_capture_norm": frozenset({4, 8}),
    "m3_chunk_norm": frozenset({4, 8}),
    # MSA: the one-launch long-context sparse decode at the TP8 heads (pre-wait up to its rows, plain above), the
    # identity-selection producer, and the sparse layers' NVFP4 QKV + indexer producer at the TP8 heads.
    "m3_sparse_decode_lc8": frozenset({"plain", "prewait"}),
    "m3_producer": frozenset({"nvfp4_tp8"}),
    "sparse_decode_producer": frozenset({"multi_query", "per_row"}),
    # MoE: the m3_moe engine (TP8 width 384, shared expert folded in) at 1..8 rows, unless killed.
    "m3_moe_engine": frozenset() if dispatch.KILL_MOE_ENGINE else frozenset(range(1, 9)),
    # The TF32 tensor-core router of 9..16-row pure generation steps on the generic layer (above the engine's rows).
    "m3_router_tc": frozenset(range(9, (48 if dispatch.ROUTER48 else 16) + 1)),
    # The index scorer at 1-2 requests, the long-row select at every key width (129..9216 blocks).
    "m3_index_score": frozenset({1, 2}),
    "long_select": frozenset({2, 4, 9}),
    "long_select_dsl": frozenset({4, 11, 24}),
    # The dense layers' full-context decode of 1-2 requests, 1 + 3 verify rows or one draft-pass row.
    "m3_fullctx_decode": frozenset({(4, 1), (1, 1), (4, 2), (1, 2)}),
    # The decode MXFP8 GEMVs at every TP8 projection and decode row, on an MXFP8 and a bf16 activation.
    "m3_mx_gemv": frozenset((nk, m) for nk in TP8_PROJECTIONS for m in range(1, _GEMV_MAX[nk] + 1))
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    "m3_mx_gemv_bf16": frozenset((nk, m) for nk in TP8_PROJECTIONS for m in _DECODE_ROWS)
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    # Past the decode bound, the GEMV (VR_GEMV48) on the QKV and shared gate_up projections at 33..48 rows and
    # cuBLASLt's MXFP8 GEMM (LINEAR_CUBLASLT) on its six projections at every other row up to 64.
    "m3_mx_gemv48": frozenset((nk, m) for nk in _GEMV48_SHAPES for m in _GEMV48_ROWS)
    if (dispatch.MX_GEMV_TP2 and dispatch.VR_GEMV48)
    else frozenset(),
    "m3_mxfp8_cublaslt": frozenset(
        (nk, m)
        for nk in _CUBLASLT_SHAPES
        for m in _WIDE_ROWS
        if not (
            dispatch.MX_GEMV_TP2
            and dispatch.VR_GEMV48
            and nk in _GEMV48_SHAPES
            and m in _GEMV48_ROWS
        )
    )
    if dispatch.LINEAR_CUBLASLT
    else frozenset(),
    # The multicast boundary all-reduce, the o_proj's push into it, the fused embedding + first RMSNorm.
    "mc_allreduce_norm": frozenset(_STEP_ROWS) if dispatch.MC_AR_TP2 else frozenset(),
    "attn_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.ATTN_PUSH_TP2)
    else frozenset(),
    "dense_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.DENSE_PUSH_TP2)
    else frozenset(),
    # MoE: the engine's push into the multicast boundary all-reduce, at the engine's rows.
    "moe_push": frozenset(range(1, dispatch.MOE_PUSH_MAX_TOKENS + 1)),
    "embed_firstnorm": frozenset(_DECODE_ROWS) if dispatch.EMBED_FIRSTNORM_TP2 else frozenset(),
    # The fused draft loop's all-reduce boundaries (1..8 rows) on the multicast all-reduce (opts 16).
    "m3_draft_mc_allreduce_norm": frozenset(range(1, dispatch.DRAFT_MC_AR_MAX_ROWS + 1)),
    # The Eagle3 drafter layer's generation passes of 1-2 requests on the same recipes.
    "m3_drafter_fullctx": frozenset({(4, 1), (1, 1), (4, 2), (1, 2)}),
    # The FSEL select: one key, default ON (TRTLLM_M31_DISABLE_TP8_FSEL=1 turns it off).
    "tp8_fsel": frozenset({"fsel"}) if dispatch.FSEL_TOPK else frozenset(),
}


def test_every_enabled_fast_path_is_registered() -> None:
    """A path listed in ``ENABLED`` that ``FAST_PATHS`` no longer registers would never be checked below."""
    missing = set(ENABLED) - {r.path for r in dispatch.FAST_PATHS}
    assert not missing, f"ENABLED lists paths FAST_PATHS does not register: {sorted(missing)}"


# ------------------------------------------------------------------ landed: the decode MXFP8 GEMVs


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("nk", TP8_PROJECTIONS, ids=str)
def test_every_projection_takes_the_gemv_at_every_decode_row(nk, rows) -> None:
    """MXFP8 activation (a producer quantized it) -> the GEMV; bf16 -> the GEMV quantizing in its prologue (staged:
    the stock decode-graph GEMMs while MX_GEMV_TP2 is off)."""
    on = dispatch.MX_GEMV_TP2
    assert dispatch.linear_path(*nk, rows, prequant=True, low_m_gemm=True) == (
        dispatch.MX_GEMV_PATH if on else dispatch.PREQUANT_GEMM_PATH
    )
    assert dispatch.linear_path(*nk, rows, prequant=False, low_m_gemm=True) == (
        dispatch.MX_GEMV_BF16_PATH if on else dispatch.GENERIC_GEMM_PATH
    )


def test_a_miss_raises_only_under_require() -> None:
    dispatch.missed("some_path", 1, "a precondition")
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="some_path"):
        dispatch.missed("some_path", 1, "a precondition")


# ------------------------------------------------------- landed: the multicast boundary all-reduce


# ------------------------------------------------------------------ landed: the fused embedding + first RMSNorm


# ------------------------------------------------------- landed: MX-TAC (pinned decode-graph MXFP8 GEMM tactic)


# ------------------------------------------------------------------ the drafter's low-M linears (TP8 port)


def _runtime(fused: bool, indexer: str = "fp8", kv: str = "fp8"):
    """A model config with the runtime fields ``_validate_runtime`` reads."""
    sparse = SimpleNamespace(
        algorithm="minimax_m3",
        implementation="msa",
        fuse_qkv_index_projection=fused,
        indexer_kv_dtype=indexer,
    )
    mode = SimpleNamespace(
        has_fp8_kv_cache=lambda: kv == "fp8", has_fp4_kv_cache=lambda: kv == "nvfp4"
    )
    return SimpleNamespace(
        sparse_attention_config=sparse, quant_config=SimpleNamespace(quant_mode=mode)
    )


@pytest.mark.parametrize("fused", [True, False], ids=["five_way", "unfused"])
def test_the_target_takes_both_projection_layouts(fused) -> None:
    """The packed five-way projection and the separate QKV + MXFP8 index projections
    (fuse_qkv_index_projection=False) are both this target's; a BF16 indexer or a BF16 KV cache is not."""
    dispatch._validate_runtime(_runtime(fused))
    dispatch._validate_runtime(_runtime(fused, kv="nvfp4"))
    with pytest.raises(ValueError, match="indexer_kv_dtype='fp8'"):
        dispatch._validate_runtime(_runtime(fused, indexer="bf16"))
    with pytest.raises(ValueError, match="FP8 or NVFP4 KV cache"):
        dispatch._validate_runtime(_runtime(fused, kv="bf16"))
