# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which k3_rubin optimizations run in this target. Nothing here reads the environment.

The k3_rubin kernels this target vendors read :func:`active` at import (module-level constants) or
when called. This target selects no k3_rubin profile: :func:`active` is ``K3Features()``, every
optimization off, the plain code path its kernels were taken with. Unit tests change a field with
``monkeypatch.setattr(features, "_ACTIVE", dataclasses.replace(features.active(), <field>=<value>))``
(fields read when called) or by patching the module constant (fields read at import).
"""

from __future__ import annotations

import dataclasses
from typing import Dict


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


_ACTIVE = K3Features()


def active() -> K3Features:
    """The settings the vendored k3_rubin kernels read (``K3Features()``)."""
    return _ACTIVE
