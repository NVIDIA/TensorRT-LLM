# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which k3_rubin optimizations run, per modeling_v2 target. Nothing here reads the environment.

Each target in ``routing.py`` is one deployment shape and runs one fixed set of optimizations: the set
the Kimi K3 Rubin AgentX recipes turn on for the Pareto points that shape serves
(the Kimi K3 Rubin AgentX disaggregated recipes; the points that run helix8 all set the same
values). These used to be environment switches; :data:`RETIRED_ENV` lists them.

``route()`` calls :func:`select` with the target before it imports the target's module, so every
k3_rubin module is imported after the selection and reads :func:`active` at import (module-level
constants) or when called. Before any selection :func:`active` is ``K3Features()``: every optimization
off, the plain code path (what an unset environment gave). Unit tests that import the modules
directly get that; they change a field with ``monkeypatch.setattr(features, "_ACTIVE",
dataclasses.replace(features.active(), <field>=<value>))`` (fields read when called) or by patching
the module constant (fields read at import).
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Dict, Optional


@dataclasses.dataclass(frozen=True)
class K3Features:
    """One target's settings. Each default is the value the code used when the variable was unset."""

    # text_model.py
    fuse_moe_residual: bool = False  # TRTLLM_K3_FUSE_MOE_RESIDUAL
    fp8w_lowm_gemm: bool = False  # TRTLLM_K3_FP8W_LOWM_GEMM
    fp8w_lowm_fused_a: bool = False  # TRTLLM_K3_FP8W_LOWM_FUSED_A
    helix_v_up_before_a2a: bool = False  # TRTLLM_HELIX_V_UP_BEFORE_A2A
    route_tail: str = "0"  # TRTLLM_K3_ROUTE_TAIL: 0, 1 or gemv
    route_tail_order: str = "router"  # TRTLLM_K3_ROUTE_TAIL_ORDER: 0, router or fork
    ld_fused: str = "0"  # TRTLLM_K3_LD_FUSED: 0, 1 or 2
    router_first: bool = False  # TRTLLM_K3_ROUTER_FIRST
    shared_sm: int = 0  # TRTLLM_K3_SHARED_SM
    fc12_prezero: bool = False  # TRTLLM_K3_FC12_PREZERO
    router_exact: str = "0"  # TRTLLM_K3_ROUTER_EXACT: 0, 1 or kda
    latent_ar_own_ws: bool = False  # TRTLLM_K3_LATENT_AR_OWN_WS
    latent_ar_own_ws_max_m: int = 32  # TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M
    latent_ar_oneshot_max_bytes: int = 0  # TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES
    latent_ar_cute: str = "0"  # TRTLLM_K3_LATENT_AR_CUTE: 0, 1 or late
    lu_mid_mxfp8: bool = False  # TRTLLM_K3_LU_MID_MXFP8
    latent_ar_row: bool = False  # TRTLLM_K3_LATENT_AR_ROW
    attn_res_pdl: str = "0"  # TRTLLM_K3_ATTN_RES_PDL: 0, 1 or 2
    attn_res_pdl_max_m: int = 64  # TRTLLM_K3_ATTN_RES_PDL_MAX_M
    attn_res_mxfp8: bool = False  # TRTLLM_K3_ATTN_RES_MXFP8
    ar_push_amid: bool = False  # TRTLLM_K3_AR_PUSH_AMID
    ar_finish_fused: bool = False  # TRTLLM_K3_AR_FINISH_FUSED
    ull_oproj_sandwich: bool = False  # TRTLLM_K3_ULL_OPROJ_SANDWICH
    ull_tail_sandwich: bool = False  # TRTLLM_K3_ULL_TAIL_SANDWICH
    ull2_coll_cp_tail: bool = False  # TRTLLM_K3_ULL2_COLL_CP_TAIL
    ull2_coll_ar_early_state: str = "0"  # TRTLLM_K3_ULL2_COLL_AR_EARLY_STATE: 0 or clear
    ull_hygiene: bool = False  # TRTLLM_K3_ULL_HYGIENE
    ull_tail1: bool = False  # TRTLLM_K3_ULL_TAIL1
    ull_shared_persist: int = 0  # TRTLLM_K3_ULL_SHARED_PERSIST
    mid_ar_push: bool = False  # TRTLLM_K3_MID_AR_PUSH
    mid_mxfp8_in: bool = False  # TRTLLM_K3_MID_MXFP8_IN
    rp_tail: bool = False  # TRTLLM_K3_RP_TAIL
    rp_tail_max_m: int = 64  # TRTLLM_K3_RP_TAIL_MAX_M
    ull_front: bool = False  # TRTLLM_K3_ULL_FRONT
    ull_front_early: bool = False  # TRTLLM_K3_ULL_FRONT_EARLY
    # comm/lamport.py
    ar_push: bool = False  # TRTLLM_K3_AR_PUSH
    # kernels/
    attn_res_early_trigger: bool = False  # TRTLLM_K3_ATTN_RES_EARLY_TRIGGER
    kda_qkvg_stage_ring: bool = False  # TRTLLM_K3_KDA_QKVG_STAGE_RING
    cute_carveout_fix: bool = False  # TRTLLM_K3_CUTE_CARVEOUT_FIX
    route_approx: bool = False  # TRTLLM_K3_ROUTE_APPROX
    kda_decode_early_trigger: bool = False  # TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER
    # kda/kimi_kda_mixer.py
    kda_prepad_prefill: bool = False  # TLLM_KDA_PREPAD_PREFILL
    kda_fused_conv_prefill: bool = False  # TRTLLM_KDA_FUSED_CONV_PREFILL
    kda_decode_fused_fab: bool = False  # TRTLLM_K3_KDA_DECODE_FUSED_FAB
    kda_l2_prefetch: bool = False  # TRTLLM_K3_KDA_L2_PREFETCH
    kda_qkvg_stage: bool = False  # TRTLLM_K3_KDA_QKVG_STAGE
    # mla/
    mla_ctx_triton_quant: bool = False  # TRTLLM_K3_MLA_CTX_TRITON_QUANT
    mla_ctx_fp8_kv_fused: bool = False  # TRTLLM_K3_MLA_CTX_FP8_KV_FUSED
    oproj_pc: bool = False  # TRTLLM_K3_OPROJ_PC
    ull2_mla_tail: bool = False  # TRTLLM_K3_ULL2_MLA_TAIL
    ull3_mla_late_tail_max_tokens: int = 16  # TRTLLM_K3_ULL3_MLA_LATE_TAIL_MAX_TOKENS
    mla_gate_side: bool = False  # TRTLLM_K3_MLA_GATE_SIDE
    mla_gate_side_max_m: int = 0  # TRTLLM_K3_MLA_GATE_SIDE_MAX_M
    mla_q_norm_pdl: bool = False  # TRTLLM_K3_MLA_Q_NORM_PDL
    ull3_mla_gate_at: str = "decode"  # TRTLLM_K3_ULL3_MLA_GATE_AT
    mla_q_fused: bool = False  # TRTLLM_K3_MLA_Q_FUSED
    # moe/k3_ull_experts.py
    ull_moetp: bool = False  # TRTLLM_K3_ULL_MOETP
    ull_experts_trigger: str = "end"  # TRTLLM_K3_ULL_EXPERTS_TRIGGER: end or wait
    ull_experts_ctas: str = ""  # TRTLLM_K3_ULL_EXPERTS_CTAS: "<N>" or "<n_tok>:<N>,..."
    # spec/
    dspark_ar_push: bool = False  # TLLM_DSPARK_AR_PUSH
    draft3_glue: bool = False  # TRTLLM_K3_DRAFT3_GLUE
    draft3_qhead: bool = False  # TRTLLM_K3_DRAFT3_QHEAD
    dspark_draft_gemm: bool = False  # TLLM_DSPARK_DRAFT_GEMM
    dspark_fused_markov: bool = False  # TLLM_DSPARK_FUSED_MARKOV
    ull2_draft_a2a: bool = False  # TRTLLM_K3_ULL2_DRAFT_A2A
    ull2_draft_tail: bool = False  # TRTLLM_K3_ULL2_DRAFT_TAIL
    ull2_draft_store: bool = False  # TRTLLM_K3_ULL2_DRAFT_STORE
    spec_tail_fusion: bool = False  # TLLM_SPEC_TAIL_FUSION
    dspark_tail_v2: bool = False  # TLLM_DSPARK_TAIL_V2
    dspark_ctx_side_stream: bool = False  # TLLM_DSPARK_CTX_SIDE_STREAM
    dspark_draft_fusion: bool = False  # TLLM_DSPARK_DRAFT_FUSION
    spec_vocab_parallel: bool = False  # TLLM_SPEC_VOCAB_PARALLEL
    dspark_markov_v2: str = "0"  # TLLM_DSPARK_MARKOV_V2: 0, 1 or 2
    dspark_markov_v3: bool = False  # TLLM_DSPARK_MARKOV_V3
    spec_vocab_parallel_sampling: bool = False  # TLLM_SPEC_VOCAB_PARALLEL_SAMPLING

    def changed(self) -> Dict[str, object]:
        """The fields that differ from the plain path, by name."""
        base = K3Features()
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(base, f.name)
        }


# One profile per routing target (the names are routing.py's ``parallel`` values).
PROFILES: Dict[str, K3Features] = {
    # Context workers of every point (DWDP, one rank per process).
    "tp1": K3Features(
        kda_fused_conv_prefill=True,
        kda_prepad_prefill=True,
        mla_ctx_fp8_kv_fused=True,
        mla_ctx_triton_quant=True,
    ),
    # Generation of the helix8 points.
    "helix8": K3Features(
        ar_finish_fused=True,
        ar_push=True,
        ar_push_amid=True,
        attn_res_early_trigger=True,
        attn_res_mxfp8=True,
        attn_res_pdl="2",
        attn_res_pdl_max_m=128,
        dspark_ar_push=True,
        dspark_ctx_side_stream=True,
        dspark_draft_fusion=True,
        dspark_fused_markov=True,
        dspark_markov_v2="2",
        dspark_markov_v3=True,
        dspark_tail_v2=True,
        fc12_prezero=True,
        fp8w_lowm_fused_a=True,
        fp8w_lowm_gemm=True,
        fuse_moe_residual=True,
        helix_v_up_before_a2a=True,
        kda_decode_early_trigger=True,
        kda_decode_fused_fab=True,
        kda_l2_prefetch=True,
        kda_qkvg_stage=True,
        latent_ar_cute="1",
        latent_ar_oneshot_max_bytes=6291456,
        latent_ar_own_ws=True,
        latent_ar_own_ws_max_m=128,
        latent_ar_row=True,
        ld_fused="1",
        lu_mid_mxfp8=True,
        mid_ar_push=True,
        mid_mxfp8_in=True,
        mla_gate_side=True,
        mla_gate_side_max_m=128,
        mla_q_fused=True,
        mla_q_norm_pdl=True,
        oproj_pc=True,
        route_approx=True,
        route_tail="1",
        route_tail_order="0",
        router_exact="kda",
        router_first=True,
        rp_tail=True,
        rp_tail_max_m=32,
        shared_sm=16,
        spec_tail_fusion=True,
        spec_vocab_parallel=True,
        spec_vocab_parallel_sampling=True,
    ),
    # Generation of the ULL points.
    "helix16_moetp4ep4": K3Features(
        ar_finish_fused=True,
        ar_push=True,
        ar_push_amid=True,
        attn_res_early_trigger=True,
        attn_res_mxfp8=True,
        attn_res_pdl="2",
        cute_carveout_fix=True,
        draft3_glue=True,
        draft3_qhead=True,
        dspark_ar_push=True,
        dspark_ctx_side_stream=True,
        dspark_draft_fusion=True,
        dspark_draft_gemm=True,
        dspark_fused_markov=True,
        dspark_markov_v2="2",
        dspark_markov_v3=True,
        dspark_tail_v2=True,
        fp8w_lowm_fused_a=True,
        fp8w_lowm_gemm=True,
        fuse_moe_residual=True,
        helix_v_up_before_a2a=True,
        kda_decode_early_trigger=True,
        kda_decode_fused_fab=True,
        kda_qkvg_stage=True,
        kda_qkvg_stage_ring=True,
        ld_fused="1",
        mla_gate_side=True,
        mla_q_fused=True,
        mla_q_norm_pdl=True,
        oproj_pc=True,
        route_tail="1",
        router_exact="kda",
        router_first=True,
        rp_tail=True,
        rp_tail_max_m=32,
        spec_tail_fusion=True,
        spec_vocab_parallel=True,
        spec_vocab_parallel_sampling=True,
        ull2_coll_ar_early_state="clear",
        ull2_coll_cp_tail=True,
        ull2_draft_a2a=True,
        ull2_draft_store=True,
        ull2_draft_tail=True,
        ull2_mla_tail=True,
        ull3_mla_gate_at="exchange:16,start",
        ull3_mla_late_tail_max_tokens=32,
        ull_experts_ctas="8:196,16:204,32:196",
        ull_experts_trigger="wait",
        ull_front=True,
        ull_front_early=True,
        ull_hygiene=True,
        ull_moetp=True,
        ull_oproj_sandwich=True,
        ull_shared_persist=16,
        ull_tail1=True,
        ull_tail_sandwich=True,
    ),
    # Generation of the dep16 point.
    "dep16": K3Features(
        attn_res_pdl="2",
        dspark_draft_fusion=True,
        dspark_fused_markov=True,
        fp8w_lowm_fused_a=True,
        fp8w_lowm_gemm=True,
        fuse_moe_residual=True,
        spec_tail_fusion=True,
    ),
}

_FIELD_ENV = {
    "fuse_moe_residual": "TRTLLM_K3_FUSE_MOE_RESIDUAL",
    "fp8w_lowm_gemm": "TRTLLM_K3_FP8W_LOWM_GEMM",
    "fp8w_lowm_fused_a": "TRTLLM_K3_FP8W_LOWM_FUSED_A",
    "helix_v_up_before_a2a": "TRTLLM_HELIX_V_UP_BEFORE_A2A",
    "route_tail": "TRTLLM_K3_ROUTE_TAIL",
    "route_tail_order": "TRTLLM_K3_ROUTE_TAIL_ORDER",
    "ld_fused": "TRTLLM_K3_LD_FUSED",
    "router_first": "TRTLLM_K3_ROUTER_FIRST",
    "shared_sm": "TRTLLM_K3_SHARED_SM",
    "fc12_prezero": "TRTLLM_K3_FC12_PREZERO",
    "router_exact": "TRTLLM_K3_ROUTER_EXACT",
    "latent_ar_own_ws": "TRTLLM_K3_LATENT_AR_OWN_WS",
    "latent_ar_own_ws_max_m": "TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M",
    "latent_ar_oneshot_max_bytes": "TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES",
    "latent_ar_cute": "TRTLLM_K3_LATENT_AR_CUTE",
    "lu_mid_mxfp8": "TRTLLM_K3_LU_MID_MXFP8",
    "latent_ar_row": "TRTLLM_K3_LATENT_AR_ROW",
    "attn_res_pdl": "TRTLLM_K3_ATTN_RES_PDL",
    "attn_res_pdl_max_m": "TRTLLM_K3_ATTN_RES_PDL_MAX_M",
    "attn_res_mxfp8": "TRTLLM_K3_ATTN_RES_MXFP8",
    "ar_push_amid": "TRTLLM_K3_AR_PUSH_AMID",
    "ar_finish_fused": "TRTLLM_K3_AR_FINISH_FUSED",
    "ull_oproj_sandwich": "TRTLLM_K3_ULL_OPROJ_SANDWICH",
    "ull_tail_sandwich": "TRTLLM_K3_ULL_TAIL_SANDWICH",
    "ull2_coll_cp_tail": "TRTLLM_K3_ULL2_COLL_CP_TAIL",
    "ull2_coll_ar_early_state": "TRTLLM_K3_ULL2_COLL_AR_EARLY_STATE",
    "ull_hygiene": "TRTLLM_K3_ULL_HYGIENE",
    "ull_tail1": "TRTLLM_K3_ULL_TAIL1",
    "ull_shared_persist": "TRTLLM_K3_ULL_SHARED_PERSIST",
    "mid_ar_push": "TRTLLM_K3_MID_AR_PUSH",
    "mid_mxfp8_in": "TRTLLM_K3_MID_MXFP8_IN",
    "rp_tail": "TRTLLM_K3_RP_TAIL",
    "rp_tail_max_m": "TRTLLM_K3_RP_TAIL_MAX_M",
    "ull_front": "TRTLLM_K3_ULL_FRONT",
    "ull_front_early": "TRTLLM_K3_ULL_FRONT_EARLY",
    "ar_push": "TRTLLM_K3_AR_PUSH",
    "attn_res_early_trigger": "TRTLLM_K3_ATTN_RES_EARLY_TRIGGER",
    "kda_qkvg_stage_ring": "TRTLLM_K3_KDA_QKVG_STAGE_RING",
    "cute_carveout_fix": "TRTLLM_K3_CUTE_CARVEOUT_FIX",
    "route_approx": "TRTLLM_K3_ROUTE_APPROX",
    "kda_decode_early_trigger": "TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER",
    "kda_prepad_prefill": "TLLM_KDA_PREPAD_PREFILL",
    "kda_fused_conv_prefill": "TRTLLM_KDA_FUSED_CONV_PREFILL",
    "kda_decode_fused_fab": "TRTLLM_K3_KDA_DECODE_FUSED_FAB",
    "kda_l2_prefetch": "TRTLLM_K3_KDA_L2_PREFETCH",
    "kda_qkvg_stage": "TRTLLM_K3_KDA_QKVG_STAGE",
    "mla_ctx_triton_quant": "TRTLLM_K3_MLA_CTX_TRITON_QUANT",
    "mla_ctx_fp8_kv_fused": "TRTLLM_K3_MLA_CTX_FP8_KV_FUSED",
    "oproj_pc": "TRTLLM_K3_OPROJ_PC",
    "ull2_mla_tail": "TRTLLM_K3_ULL2_MLA_TAIL",
    "ull3_mla_late_tail_max_tokens": "TRTLLM_K3_ULL3_MLA_LATE_TAIL_MAX_TOKENS",
    "mla_gate_side": "TRTLLM_K3_MLA_GATE_SIDE",
    "mla_gate_side_max_m": "TRTLLM_K3_MLA_GATE_SIDE_MAX_M",
    "mla_q_norm_pdl": "TRTLLM_K3_MLA_Q_NORM_PDL",
    "ull3_mla_gate_at": "TRTLLM_K3_ULL3_MLA_GATE_AT",
    "mla_q_fused": "TRTLLM_K3_MLA_Q_FUSED",
    "ull_moetp": "TRTLLM_K3_ULL_MOETP",
    "ull_experts_trigger": "TRTLLM_K3_ULL_EXPERTS_TRIGGER",
    "ull_experts_ctas": "TRTLLM_K3_ULL_EXPERTS_CTAS",
    "dspark_ar_push": "TLLM_DSPARK_AR_PUSH",
    "draft3_glue": "TRTLLM_K3_DRAFT3_GLUE",
    "draft3_qhead": "TRTLLM_K3_DRAFT3_QHEAD",
    "dspark_draft_gemm": "TLLM_DSPARK_DRAFT_GEMM",
    "dspark_fused_markov": "TLLM_DSPARK_FUSED_MARKOV",
    "ull2_draft_a2a": "TRTLLM_K3_ULL2_DRAFT_A2A",
    "ull2_draft_tail": "TRTLLM_K3_ULL2_DRAFT_TAIL",
    "ull2_draft_store": "TRTLLM_K3_ULL2_DRAFT_STORE",
    "spec_tail_fusion": "TLLM_SPEC_TAIL_FUSION",
    "dspark_tail_v2": "TLLM_DSPARK_TAIL_V2",
    "dspark_ctx_side_stream": "TLLM_DSPARK_CTX_SIDE_STREAM",
    "dspark_draft_fusion": "TLLM_DSPARK_DRAFT_FUSION",
    "spec_vocab_parallel": "TLLM_SPEC_VOCAB_PARALLEL",
    "dspark_markov_v2": "TLLM_DSPARK_MARKOV_V2",
    "dspark_markov_v3": "TLLM_DSPARK_MARKOV_V3",
    "spec_vocab_parallel_sampling": "TLLM_SPEC_VOCAB_PARALLEL_SAMPLING",
}

# Variables no recipe sets (or every recipe sets to the same value): the code now uses that value.
_FIXED_ENV = (
    "TLLM_K3_MLA_GEN_BACKEND",
    "TLLM_KDA_BUF_CACHE_ENTRIES",
    "TLLM_SPEC_VOCAB_PARALLEL_COMM",
    "TRTLLM_K3_AR_PUSH_ONE_SHOT_MAX_BYTES",
    "TRTLLM_K3_BF16_OVER_PREQUANTIZED_MAX_M",
    "TRTLLM_K3_FP4_MLA_SHARED_DECODE",
    "TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES",
    "TRTLLM_K3_HELIX_TAIL_CTAS",
    "TRTLLM_K3_HELIX_TAIL_GATE",
    "TRTLLM_K3_HELIX_TAIL_MB",
    "TRTLLM_K3_KDA_L2_PREFETCH_MAX_ROWS",
    "TRTLLM_K3_KDA_MTP_V2",
    "TRTLLM_K3_KDA_MX_PDL",
    "TRTLLM_K3_LATENT_AR_ROW_MAX_M",
    "TRTLLM_K3_LATENT_STREAM_PRIORITY",
    "TRTLLM_K3_LD_FUSED_MAX_M",
    "TRTLLM_K3_LD_FUSED_MIN_M",
    "TRTLLM_K3_LOWM_SF_SLOTS",
    "TRTLLM_K3_LU_MID_MAX_M",
    "TRTLLM_K3_MID_AR_PUSH_MAX_M",
    "TRTLLM_K3_MID_OPROJ_PC",
    "TRTLLM_K3_MLA_Q_FUSED_KV_PRIO",
    "TRTLLM_K3_MLA_Q_FUSED_MAX_TOKENS",
    "TRTLLM_K3_MLA_Q_FUSED_NORM",
    "TRTLLM_K3_ROUTER_FIRST_MAX_M",
    "TRTLLM_K3_ROUTE_GEMV_MAX_M",
    "TRTLLM_K3_ROUTE_TAIL_MAX_M",
    "TRTLLM_K3_ROUTE_TAIL_ORDER_MIN_ROWS",
    "TRTLLM_K3_ROUTING_STATS_CHUNK",
    "TRTLLM_K3_ROUTING_STATS_DIR",
    "TRTLLM_K3_ROUTING_STATS_EVERY",
    "TRTLLM_K3_ROUTING_STATS_STRIDE",
    "TRTLLM_K3_SHARED_STREAM_PRIORITY",
    "TRTLLM_K3_SHARED_TAIL_MAX_M",
    "TRTLLM_K3_SHARED_TAIL_MIN_M",
    "TRTLLM_K3_SHARED_TAIL_PRIO",
    "TRTLLM_K3_ULL_EXPERTS_COMBINE",
    "TRTLLM_K3_ULL_FRONT_MAX_M",
)

#: Environment variables k3_rubin no longer reads -> the field that took their place (None: fixed in the code).
RETIRED_ENV: Dict[str, Optional[str]] = {
    **{env: field for field, env in _FIELD_ENV.items()},
    **{env: None for env in _FIXED_ENV},
}

_ACTIVE = K3Features()
_SELECTED: Optional[str] = None
_PACKAGE = __name__.rpartition(".")[0]


def active() -> K3Features:
    """The selected target's settings (``K3Features()`` before :func:`select`)."""
    return _ACTIVE


def selected() -> Optional[str]:
    """The selected target, or None."""
    return _SELECTED


def select(target: str) -> K3Features:
    """Make ``target``'s profile the active one; once per process, before any other k3_rubin import.

    Raises if another target was selected, or if a k3_rubin module was imported first (its
    import-time settings would be the plain path's, not ``target``'s).
    """
    global _ACTIVE, _SELECTED
    if target not in PROFILES:
        raise KeyError(f"no Kimi K3 feature profile {target!r}; have {sorted(PROFILES)}")
    if _SELECTED is not None:
        if _SELECTED != target:
            raise RuntimeError(
                f"Kimi K3 features: {_SELECTED!r} is selected in this process, cannot select {target!r}"
            )
        return _ACTIVE
    early = sorted(m for m in sys.modules if m.startswith(_PACKAGE + ".") and m != __name__)
    if early:
        raise RuntimeError(
            f"Kimi K3 features: k3_rubin modules were imported before the {target!r} profile was "
            f"selected, with the plain path's settings: {', '.join(early)}"
        )
    _ACTIVE, _SELECTED = PROFILES[target], target

    from tensorrt_llm.logger import logger

    changed = _ACTIVE.changed()
    logger.info(
        f"Kimi K3 features: {target} ({len(changed)} set): "
        + ", ".join(f"{name}={value}" for name, value in sorted(changed.items()))
    )
    stale = sorted(name for name in RETIRED_ENV if name in os.environ)
    if stale:
        logger.warning(
            "Kimi K3 features: these environment variables are no longer read (the target's "
            f"profile in k3_rubin/features.py decides): {', '.join(stale)}"
        )
    return _ACTIVE
