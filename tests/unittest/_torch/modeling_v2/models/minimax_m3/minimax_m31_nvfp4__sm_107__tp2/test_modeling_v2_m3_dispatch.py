# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The M3.1 tp2 (Rubin) decode target's fast-path table (``modeling.py``: the row ranges and the selection functions):
its row ranges against the catalog cells that certify them, the selection for every decode row count, and the phase
split, at the TP2 per-rank geometry (32 query / 2 KV / 2 index heads, expert width 1536).

The plans are the M3 tp2 row buckets (tiny 8 / b16 16 / b48 48) with the TP4 base's one setting each; MoE16 is off
(the TP4 kernel is untested at width 1536), ROUTER48 takes the tensor-core router at 9..48 rows, the
multicast all-reduce + norm runs at world 2 without the one-shot cluster option. The ones not yet driven on SM 10.7
(the long-row select, the capped dense decode split, the early drafter fc, the MC finalize) cover nothing. The dense
layers' full-context decode keeps its TP4 build (M3.1 has no dense attention layers).

CPU only: nothing here launches a kernel.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    m3_report,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    triton_sparse_decode as tsd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm import (  # noqa: E501
    m3_mx_gemv,
)

ROWS = range(1, dispatch.DECODE_STEP_MAX + 2)  # every decode row count and the first one past it

# The TP2 per-rank MXFP8 projections (N, K): 32 query heads and 2 KV heads, expert width 1536 per rank.
TP2_PROJECTIONS = (
    m3_mx_gemv.QKV_SPARSE,
    m3_mx_gemv.QKV,
    m3_mx_gemv.O_PROJ,
    m3_mx_gemv.SHARED_GATE_UP,
    m3_mx_gemv.SHARED_DOWN,
    m3_mx_gemv.DENSE_GATE_UP,
    m3_mx_gemv.DENSE_DOWN,
)


@pytest.fixture(autouse=True)
def _policy():
    """Every test starts outside require mode with nothing logged or counted, and leaves it so."""
    dispatch.set_require(False)
    dispatch.reset_logged()
    m3_report.reset_work()
    yield
    dispatch.set_require(False)
    dispatch.reset_logged()
    m3_report.reset_work()


@pytest.mark.parametrize("fast_path", dispatch.FAST_PATHS, ids=lambda r: r.path)
def test_every_covered_shape_is_a_certified_cell(fast_path) -> None:
    uncertified = fast_path.keys - fast_path.certified()
    assert not uncertified, (
        f"{fast_path.path} covers {sorted(uncertified, key=str)} but "
        f"{type(fast_path.entry).__name__}.CELLS certifies none of them: add a cell or narrow the range"
    )


def test_the_fast_path_names_are_unique() -> None:
    names = [r.path for r in dispatch.FAST_PATHS]
    assert len(names) == len(set(names))


def test_the_decode_target_keeps_the_tp2_row_bounds() -> None:
    """Up to 48 decode-layer rows (the TP4 base's bound; R1VR8's kill restores the TP2-lineage 8), up to 64 with
    VR_BANDS (off with the R1VR8 kill); a wider step runs the generic layers."""
    assert dispatch.DECODE_MAX_TOKENS == (48 if dispatch.R1VR8 else dispatch.TP2_ROWS)
    assert dispatch.TP2_ROWS == 8
    assert dispatch.DECODE_STEP_MAX == (
        448 if dispatch.TP2WIDE_DECODE else 64 if dispatch.VR_BANDS else dispatch.DECODE_MAX_TOKENS
    )
    assert not dispatch.VR_BANDS or dispatch.R1VR8
    assert [dispatch.is_decode_step(n) for n in ROWS] == [True] * dispatch.DECODE_STEP_MAX + [False]


# ------------------------------------------------------------------ the paths still off on SM 10.7

OFF_ON_SM107 = (
    dispatch.DENSE_DECODE_SPLIT,
    dispatch.LONG_SELECT,
) + (() if dispatch.TP2W_MC_FINALIZE else (dispatch.MC_FINALIZE,))


@pytest.mark.parametrize("fast_path", OFF_ON_SM107, ids=lambda r: r.path)
def test_paths_without_an_sm107_build_cover_nothing(fast_path) -> None:
    assert not fast_path.keys, f"{fast_path.path} covers {sorted(fast_path.keys, key=str)}"


def test_decode_boundaries_follow_the_multicast_all_reduce_kill() -> None:
    """MC_AR_TP4 moves every decode boundary (and the o_proj / MoE pushes into it) onto the SM 10.7 TP4 multicast
    all-reduce; off, the built-in all-reduce runs and nothing pushes."""
    on = dispatch.MC_AR_TP4
    for plan in dispatch.DECODE_PLANS:
        want = (False, False, False) if plan.name in ("wide", "wide448") else (on, on, on)
        assert (plan.attn_mc_ar, plan.moe_mc_ar, plan.qkv_prequant) == want, plan
    assert dispatch.TP2_BUCKETS == tuple(b for b in (8, 16, 48) if b <= dispatch.DECODE_MAX_TOKENS)
    assert tuple(p.name for p in dispatch.DECODE_PLANS) == ("tiny", "b16", "b48")[
        : len(dispatch.TP2_BUCKETS)
    ] + (("wide",) if dispatch.VR_BANDS else ()) + (("wide448",) if dispatch.TP2WIDE_DECODE else ())
    assert tuple(p.max_tokens for p in dispatch.DECODE_PLANS) == dispatch.TP2_BUCKETS + (
        (64,) if dispatch.VR_BANDS else ()
    ) + ((448,) if dispatch.TP2WIDE_DECODE else ())
    # World 2: the TP4 op's row kernel takes world 4 only; the one-shot cluster option is off at every row count.
    assert dispatch.MC_AR_CLUSTER_MAX_ROWS == 0
    assert bool(dispatch.MC_ALLREDUCE.keys) == on
    assert dispatch.DRAFT_MC_AR.keys == (frozenset(range(1, 9)) if on else frozenset())
    assert dispatch.ATTN_PUSH_MAX_TOKENS == (8 if on else 0)
    assert dispatch.DENSE_PUSH_MAX_TOKENS == (8 if on and dispatch.DENSE_PUSH_TP4 else 0)
    assert dispatch.MOE_PUSH_MAX_TOKENS == (dispatch.MOE_ENGINE_MAX_TOKENS if on else 0)
    assert dispatch.GENERIC_MC_AR_MAX_TOKENS == 0
    assert dispatch.GENERIC_MC_AR_FOLDED_MAX_TOKENS == 0


def test_the_dense_decode_split_is_the_built_in_one() -> None:
    assert dict(dispatch.DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH) == {}


# ------------------------------------------------------------------ on: the dense layers' full-context decode


def test_the_dense_layers_keep_the_fullctx_decode_tables() -> None:
    """One or two requests, 1 + 3 verify rows or one draft-pass row, each at its own pinned recipe."""
    assert dispatch.FULLCTX_DECODE_ENABLED
    assert dispatch.FULLCTX_DECODE.keys == {(4, 1), (1, 1), (4, 2), (1, 2)}
    assert set(dispatch.FULLCTX_DECODE_RECIPES) == set(dispatch.FULLCTX_DECODE.keys)
    for recipe in dispatch.FULLCTX_DECODE_RECIPES.values():
        assert 2 <= recipe["splits"] <= 128  # the standalone reducer's range


def test_the_fullctx_plans_keep_their_tp4_build() -> None:
    """The dense-attention kernel is the TP4 build, unchanged: M3.1 runs MSA on every layer (no dense attention), so
    no TP2 step reaches it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        m3_fullctx_decode as kernel,
    )

    assert (kernel.HEADS, kernel.KV_HEADS, kernel.HEAD_DIM) == (16, 1, 128)
    assert kernel.HEADS_PER_KV == 16


def test_the_fullctx_entry_cells_run_at_the_forward_plan_capacity() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention import (  # noqa: E501
        m3_fullctx_decode as entry,
    )

    assert entry.MAX_KV_LEN == dispatch.FULLCTX_DECODE_MAX_KV_LEN


def test_the_fullctx_capacity_covers_the_agentx_cache_bound() -> None:
    """The forward compares the cache's length bound (max_seq_len plus the speculative rows it reserves) with the plan
    capacity; at the AgentX max_seq_len of 1 << 20 with up to 8 draft rows the dense layers must still be taken."""
    assert dispatch.FULLCTX_DECODE_MAX_KV_LEN >= (1 << 20) + 8


def test_the_drafter_takes_the_fullctx_decode_at_the_same_shapes() -> None:
    assert dispatch.DRAFTER_FULLCTX_ENABLED
    assert dispatch.DRAFTER_FULLCTX.keys == dispatch.FULLCTX_DECODE.keys


def test_the_dense_fp8_qkv_and_kv_insert_kill_constants_default_on() -> None:
    assert dispatch.DENSE_QKV_FP8_VIA_BF16 is True
    assert dispatch.DENSE_KV_INSERT_FUSED is True
    assert dispatch.FULLCTX_DECODE_OWN_REDUCER is True
    assert dispatch.FULLCTX_DECODE_MXFP8 is True


def test_the_fullctx_taken_lines_log_without_raising_under_require() -> None:
    dispatch.set_require(True)
    dispatch.taken(dispatch.FULLCTX_DECODE.path, 4)
    dispatch.taken(dispatch.DENSE_KV_INSERT_PATH, 4)
    dispatch.taken(dispatch.FULLCTX_DECODE_MXFP8_PATH, 4)


# ------------------------------------------------------------------ the paths on


@pytest.mark.parametrize("rows", range(1, dispatch.DECODE_MAX_TOKENS + 1))
def test_the_moe_takes_the_engines_and_the_generation_router(rows) -> None:
    """1..8 rows the MoE engine (router in-kernel); 9..13 the MoE16 engine only with MOE16_TP2 (off by default: the
    generic MoE); the unfused MoE of a pure generation step takes the TF32 tensor-core router at 9..48 rows (ROUTER48;
    9..16 with its kill), the fp32 router else."""
    expected = (
        dispatch.MOE_ENGINE.path
        if rows <= 8
        else dispatch.MOE16_ENGINE.path
        if rows <= 13 and dispatch.MOE16_TP2
        else dispatch.MC_FINALIZE.path
        if dispatch.MC_FINALIZE_MIN_TOKENS <= rows <= 48 and dispatch.TP2W_MC_FINALIZE
        else None
    )
    assert dispatch.moe_path(rows) == expected
    assert dispatch.ROUTER_TC_GEN_MAX_ROWS == (48 if dispatch.ROUTER48 else 16)
    assert dispatch.ROUTER_TC_GEN.covers(rows) == (9 <= rows <= dispatch.ROUTER_TC_GEN_MAX_ROWS)


def test_moe16_is_off_at_the_tp2_width() -> None:
    assert dispatch.MOE16_TP2 is False
    assert dispatch.MOE16_MAX_TOKENS == 0
    assert not dispatch.MOE16_ENGINE.keys


def _router_core(cluster16: bool):
    import torch

    return SimpleNamespace(router_tc_cluster16=cluster16, w={"l3_router": torch.zeros(128, 64)})


@pytest.mark.parametrize("rows", [1, 8, 9, 13, 14, 16, 17, 24, 32, 48, 49])
@pytest.mark.parametrize("cluster16", [True, False])
def test_the_generation_router_takes_router_tc_only_on_its_rows(
    monkeypatch, rows, cluster16
) -> None:
    """The built-in gate's choice: router_tc on 9..48 bf16 rows (ROUTER48) when the device co-schedules 16-CTA clusters;
    the fp32 router otherwise. Context steps (``_router_logits``) never take it."""
    import torch

    calls = []
    monkeypatch.setattr(dispatch, "m3_router_tc", lambda *a: calls.append("tc") or [a[0]])
    monkeypatch.setattr(dispatch, "m3_fp32_linear", lambda *a: calls.append("fp32") or a[0])
    h = torch.zeros(rows, 64, dtype=torch.bfloat16)
    dispatch._decode_router_logits(_router_core(cluster16), 3, h)
    assert calls == ["tc" if cluster16 and 9 <= rows <= dispatch.ROUTER_TC_GEN_MAX_ROWS else "fp32"]
    # ROUTER48's work counter: the router rows past the TP4 router's 16.
    assert m3_report.work_count("ROUTER48") == (1 if cluster16 and 16 < rows <= 48 else 0)
    calls.clear()
    dispatch._router_logits(_router_core(cluster16), 3, h)
    assert calls == ["fp32"]


def test_the_core_selects_the_target_on_the_context_count_alone() -> None:
    """Every pure generation step is decode's, whatever its width; a step with context rows (a mixed one too) is
    prefill's."""
    core = SimpleNamespace(_prefill="prefill", _decode="decode")
    select = dispatch.MiniMaxM3Tp2ModelingV2._select_target
    assert select(core, SimpleNamespace(num_contexts=0)) == "decode"
    assert select(core, SimpleNamespace(num_contexts=1)) == "prefill"
    assert select(core, SimpleNamespace(num_contexts=7)) == "prefill"


@pytest.mark.parametrize("rows", range(1, dispatch.DECODE_MAX_TOKENS + 1))
def test_the_projections_take_the_gemvs_else_the_mxfp8_gemm(rows) -> None:
    for nk in TP2_PROJECTIONS:
        prequant = dispatch.linear_path(*nk, rows, prequant=True, low_m_gemm=True)
        assert prequant == (
            dispatch.MX_GEMV_PATH
            if rows in dispatch.MX_GEMV_ROWS[nk]
            else dispatch.PREQUANT_GEMM_PATH
        )
        bf16 = dispatch.linear_path(*nk, rows, prequant=False, low_m_gemm=True)
        assert bf16 == (
            dispatch.MX_GEMV_BF16_PATH
            if rows <= dispatch.MX_GEMV_BF16_MAX_TOKENS[nk]
            else dispatch.GENERIC_GEMM_PATH
        )


def test_the_attention_producer_and_scorer_are_on() -> None:
    assert dispatch.PRODUCER.covers("nvfp4_tp2")
    assert dispatch.INDEX_SCORE.keys == {"early", "prefetch"}
    assert dispatch.SPARSE_DECODE_PRODUCER.keys == {"per_row", "multi_query"}
    assert dispatch.EMBED_FIRSTNORM.keys == frozenset(range(1, 9))


def test_speculation_runs_the_fused_tail_with_the_bf16_draft_head() -> None:
    requests = dispatch.DECODE_MAX_TOKENS // (dispatch.SHARD_ACCEPT_DRAFT_LEN + 1)
    assert dispatch.SHARD_ACCEPT_MAX_REQUESTS == requests
    assert dispatch.SAMPLED_VERIFY_MAX_REQUESTS == requests
    assert dispatch.FIRST_DRAFTER_INPUTS_MAX_REQUESTS == requests
    assert dispatch.TARGET_HEAD_MAX_ROWS == 0  # the in-tree BF16 head (Eagle3 only)
    assert dispatch.FC_AFTER_HEAD_MAX_ROWS == 8
    assert dispatch.DRAFT_STEP_MAX_REQUESTS == 8
    assert dispatch.FC_EARLY_MAX_ROWS == 0
    assert not dispatch.DRAFT_HEAD_CODED


def test_a_miss_raises_only_under_require() -> None:
    dispatch.missed("some_path", 1, "a precondition")
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="some_path"):
        dispatch.missed("some_path", 1, "a precondition")


def test_taken_logs_without_raising_under_require() -> None:
    dispatch.set_require(True)
    dispatch.taken(dispatch.INDEX_SCORE.path, 1)
    dispatch.taken(dispatch.MOE_ENGINE.path, 8)


def test_taken_logs_once_per_path_and_rows(monkeypatch) -> None:
    lines = []
    monkeypatch.setattr(dispatch.logger, "info", lines.append)
    for _ in range(3):
        dispatch.taken(dispatch.MX_GEMV_PATH, 4, "[4992, 6144] mxfp8 in")
    dispatch.taken(dispatch.MX_GEMV_PATH, 8, "[4992, 6144] mxfp8 in")
    taken_lines = [line for line in lines if "taken at" in line]
    assert len(taken_lines) == 2 and all("m3_mx_gemv taken at" in line for line in taken_lines)
    # Gate e: each taken call is also counted (powers of two logged: 1, 2, 4).
    work = [line for line in lines if line.startswith("[M3 sm107 tp2] rank")]
    assert len(work) == 3 and "m3_mx_gemv engaged" in work[0] and "work 4 (host count" in work[-1]
    assert m3_report.work_count(dispatch.MX_GEMV_PATH) == 4


@pytest.mark.parametrize(
    "world, rows, kernel",
    [(2, 1, "8-CTA cluster kernel"), (2, 32, "8-CTA cluster kernel"), (4, 8, "row kernel")],
)
def test_the_mc_allreduce_line_carries_world_kernel_and_call_count(
    monkeypatch, world, rows, kernel
) -> None:
    """The multicast all-reduce + norm's engagement line names the world size, the kernel the TP4 op runs there (its
    row kernel at world 4 only: TP2 runs the 8-CTA cluster kernel) and this rank's running call count; logged once per
    (rows, detail), counted on every call."""
    lines = []
    monkeypatch.setattr(dispatch.logger, "info", lines.append)
    mapping = SimpleNamespace(tp_size=world)
    for _ in range(3):
        dispatch._mc_taken(mapping, rows, "x")
    taken_lines = [line for line in lines if "taken at" in line]
    assert taken_lines == [
        f"[M3 dispatch] rank {taken_lines[0].split()[3][:-1]}: mc_allreduce_norm taken at {rows} rows "
        f"(x, world {world}, {kernel}, call 1)"
    ]
    assert m3_report.work_count(dispatch.MC_ALLREDUCE.path) == 3


def test_the_sparse_decode_writes_the_mxfp8_handoff() -> None:
    """Every decode step (at most DECODE_MAX_TOKENS rows) can take the MXFP8 o_proj hand-off of the sparse decode."""
    assert dispatch.MERGE_MXFP8_HANDOFF is True
    assert tsd.MERGE_MXFP8_MIN_TOKENS <= 1 and dispatch.DECODE_MAX_TOKENS <= tsd.MergeMxfp8.MAX_ROWS


def test_the_decode_mxfp8_decision_is_pinned_to_cute_dsl() -> None:
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        FlashInferMXFP8GemmRunner,
        MXFP8QuantizeRunner,
    )

    assert dispatch.MX_TAC_TP4
    assert dispatch.MXFP8_DECODE_QUANT_TACTIC == MXFP8QuantizeRunner.CUTE_DSL
    assert dispatch.MXFP8_DECODE_GEMM_TACTIC == FlashInferMXFP8GemmRunner.CUTE_DSL


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


def test_tp2w_mc_finalize_takes_rows_past_the_moe_engines() -> None:
    """TP2W_MC_FINALIZE: the unfinalized MoE + the finalize in the multicast boundary from the first row past the MoE
    engines (TP2_MC_FINALIZE_SMALL: 9, 14 with MoE16 on; 33 with its kill) to 48 (the op's certified 48-token bucket);
    the engines' rows keep their path; the kill switch leaves no rows."""
    if dispatch.TP2W_MC_FINALIZE:
        lo = max(9, dispatch.MOE16_MAX_TOKENS + 1) if dispatch.TP2_MC_FINALIZE_SMALL else 33
        assert (dispatch.MC_FINALIZE_MIN_TOKENS, dispatch.MC_FINALIZE_MAX_TOKENS) == (lo, 48)
        assert not any(dispatch.MC_FINALIZE.covers(r) for r in range(1, lo))
        assert all(dispatch.MC_FINALIZE.covers(r) for r in range(lo, 49))
    else:
        assert not dispatch.MC_FINALIZE.keys
    src = open(dispatch.__file__).read()
    assert 'TP2W_MC_FINALIZE = not m31_disabled("TP2W_MC_FINALIZE")' in src


def test_tp2_mc_finalize_small_kill_switch(monkeypatch) -> None:
    """TRTLLM_M31_DISABLE_TP2_MC_FINALIZE_SMALL=1: the MC finalize back to the WIDE rows 33..48."""
    import importlib

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
        m31_switches,
    )

    assert dispatch.TP2_MC_FINALIZE_SMALL and dispatch.MC_FINALIZE_MIN_TOKENS == 9
    monkeypatch.setattr(m31_switches, "_DISABLED", frozenset({"TP2_MC_FINALIZE_SMALL"}))
    try:
        m = importlib.reload(dispatch)
        assert not m.TP2_MC_FINALIZE_SMALL
        assert (m.MC_FINALIZE_MIN_TOKENS, m.MC_FINALIZE_MAX_TOKENS) == (33, 48)
        assert not any(m.MC_FINALIZE.covers(r) for r in range(1, 33))
    finally:
        monkeypatch.undo()
        importlib.reload(dispatch)
