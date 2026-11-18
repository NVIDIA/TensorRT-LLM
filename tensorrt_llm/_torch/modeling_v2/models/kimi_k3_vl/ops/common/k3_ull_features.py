# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which optimizations the Kimi K3 SM107 Helix CP16 ULL target runs. Nothing here reads the environment.

The target serves one deployment, so it has one profile (:data:`PROFILE`), fixed at import: every module
of the target reads :func:`active` (module-level constants or when called). Only optimizations the plain
path leaves off are listed in :class:`K3Features`; the others run wherever their code is reached.
:data:`RETIRED_ENV` lists the environment variables the target does not read. Unit tests change a field
with ``monkeypatch.setattr(features, "_ACTIVE", dataclasses.replace(features.active(), <field>=<value>))``
(fields read when called) or by patching the constant in the module that reads it (fields read at import).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass(frozen=True)
class K3Features:
    """One deployment's settings. Each default is the plain code path.

    The fields down to ``dspark_tail_v2`` are the served tree's ``k3_rubin/features.K3Features``
    (6904cd9672) in its order, without the ones only served code this target does not run read
    (the MoE and KDA module paths the MR8 chain replaces, the ULL front, the MXFP8 AttnRes of the
    module path) and the ones whose served branch is this target's only code (``attn_res_pdl``,
    ``draft3_glue``, ``dspark_draft_gemm``: on in the served profile, written as the code itself);
    the kernels copied from there (``kernels/SOURCE.json``) read them by these names.
    The fields after it are this target's own (no served field).
    """

    # MoE (text_model.py)
    fp8w_lowm_gemm: bool = False  # TRTLLM_K3_FP8W_LOWM_GEMM
    # Attention residual and the decoder layer's all-reduces (attn_res.py, mla.py, kernels/)
    attn_res_pdl_max_m: int = 64  # TRTLLM_K3_ATTN_RES_PDL_MAX_M
    attn_res_early_trigger: bool = False  # TRTLLM_K3_ATTN_RES_EARLY_TRIGGER
    cute_carveout_fix: bool = False  # TRTLLM_K3_CUTE_CARVEOUT_FIX
    ull_oproj_sandwich: bool = False  # TRTLLM_K3_ULL_OPROJ_SANDWICH
    ull_tail_sandwich: bool = False  # TRTLLM_K3_ULL_TAIL_SANDWICH
    ull2_coll_cp_tail: bool = False  # TRTLLM_K3_ULL2_COLL_CP_TAIL and TRTLLM_K3_ULL_TAIL1
    ull4_mla_plan_early: bool = False  # TRTLLM_K3_ULL4_MLA_PLAN_EARLY
    # MLA (mla.py)
    ull2_mla_tail: bool = False  # TRTLLM_K3_ULL2_MLA_TAIL
    mla_q_norm_pdl: bool = False  # TRTLLM_K3_MLA_Q_NORM_PDL
    ull3_mla_gate_at: str = "decode"  # TRTLLM_K3_ULL3_MLA_GATE_AT
    # DSpark (drafter.py, dflash.py)
    ull2_draft_a2a: bool = False  # TRTLLM_K3_ULL2_DRAFT_A2A and TRTLLM_K3_ULL2_DRAFT_TAIL
    ull2_draft_store: bool = False  # TRTLLM_K3_ULL2_DRAFT_STORE
    spec_tail_fusion: bool = False  # TLLM_SPEC_TAIL_FUSION
    dspark_tail_v2: bool = False  # TLLM_DSPARK_TAIL_V2

    # This target's own fields. The served tree read these as environment variables or decided
    # them in its runtime; each value in PROFILE must be what the measured run (1007804) did.
    # Token embedding and the speculative decode step's head (modeling.py)
    embedding_gather: bool = False  # TLLM_SPEC_HEAD_FUSION
    # Decode step inputs: one upload and one scatter at the graph replay (modeling.py)
    pack_step_inputs: bool = False  # TRTLLM_K3_ULL_STEP_PREP and TRTLLM_BATCHED_INPUT_COMMIT
    # FP4 MLA generation decode (ops/attention/_fp4_mla_decode.py, kernels/fp4_mla_packed_reduce.py)
    ull2_mla_decode_pdl: bool = False  # TRTLLM_K3_ULL2_MLA_DEC_PDL
    fp4_mla_page_plan: bool = True  # TRTLLM_FP4_MLA_PAGE_PLAN (=0: False)
    fp4_mla_reduce_pdl: bool = False  # TRTLLM_FP4_MLA_REDUCE_PDL
    fp4_mla_f16_partials: bool = False  # TRTLLM_FP4_MLA_F16_PARTIALS

    def changed(self) -> Dict[str, object]:
        """The fields that differ from the plain path, by name."""
        base = K3Features()
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(base, f.name)
        }


#: The one deployment of this target: Helix CP16 generation with the KDA + MoE layers on the MR8
#: chain (``modeling.py``). The served fields are the served ``PROFILES["helix16_moetp4ep4"]``
#: (6904cd9672) unchanged; the target's own fields follow.
PROFILE = K3Features(
    attn_res_early_trigger=True,
    cute_carveout_fix=True,
    dspark_tail_v2=True,
    fp8w_lowm_gemm=True,
    mla_q_norm_pdl=True,
    spec_tail_fusion=True,
    ull2_coll_cp_tail=True,
    ull2_draft_a2a=True,
    ull2_draft_store=True,
    ull2_mla_tail=True,
    ull3_mla_gate_at="exchange:16,start",
    ull4_mla_plan_early=True,
    ull_oproj_sandwich=True,
    ull_tail_sandwich=True,
    # The target's own fields, at the values of 1007804's GEN process environment (its srun --export:
    # TRTLLM_FP4_MLA_F16_PARTIALS=1, TRTLLM_FP4_MLA_REDUCE_PDL=1, TRTLLM_K3_ULL2_MLA_DEC_PDL=1,
    # TLLM_SPEC_HEAD_FUSION=1; TRTLLM_FP4_MLA_PAGE_PLAN unset). The served backend and runtime read
    # these from the environment; #21996's __default helix16_moetp4ep4 profile turns on the same four.
    # pack_step_inputs stands for served StepShell(step_prep=True), its input commit at the replay.
    embedding_gather=True,
    fp4_mla_f16_partials=True,
    fp4_mla_reduce_pdl=True,
    ull2_mla_decode_pdl=True,
    pack_step_inputs=True,
)

_FIELD_ENV = {
    "ull2_coll_cp_tail": "TRTLLM_K3_ULL2_COLL_CP_TAIL",
    "embedding_gather": "TLLM_SPEC_HEAD_FUSION",
    "pack_step_inputs": "TRTLLM_K3_ULL_STEP_PREP",
    "fp8w_lowm_gemm": "TRTLLM_K3_FP8W_LOWM_GEMM",
    "attn_res_pdl_max_m": "TRTLLM_K3_ATTN_RES_PDL_MAX_M",
    "attn_res_early_trigger": "TRTLLM_K3_ATTN_RES_EARLY_TRIGGER",
    "cute_carveout_fix": "TRTLLM_K3_CUTE_CARVEOUT_FIX",
    "ull_oproj_sandwich": "TRTLLM_K3_ULL_OPROJ_SANDWICH",
    "ull_tail_sandwich": "TRTLLM_K3_ULL_TAIL_SANDWICH",
    "ull4_mla_plan_early": "TRTLLM_K3_ULL4_MLA_PLAN_EARLY",
    "ull2_mla_tail": "TRTLLM_K3_ULL2_MLA_TAIL",
    "mla_q_norm_pdl": "TRTLLM_K3_MLA_Q_NORM_PDL",
    "ull3_mla_gate_at": "TRTLLM_K3_ULL3_MLA_GATE_AT",
    "ull2_mla_decode_pdl": "TRTLLM_K3_ULL2_MLA_DEC_PDL",
    "fp4_mla_page_plan": "TRTLLM_FP4_MLA_PAGE_PLAN",
    "fp4_mla_reduce_pdl": "TRTLLM_FP4_MLA_REDUCE_PDL",
    "fp4_mla_f16_partials": "TRTLLM_FP4_MLA_F16_PARTIALS",
    "ull2_draft_a2a": "TRTLLM_K3_ULL2_DRAFT_A2A",
    "ull2_draft_store": "TRTLLM_K3_ULL2_DRAFT_STORE",
    "spec_tail_fusion": "TLLM_SPEC_TAIL_FUSION",
    "dspark_tail_v2": "TLLM_DSPARK_TAIL_V2",
}

# Variables nothing reads: the code they switched runs wherever it is reached, or no deployment runs it.
_FIXED_ENV = (
    "TRTLLM_K3_ATTN_RES_PDL",
    "TRTLLM_K3_DRAFT3_GLUE",
    "TLLM_DSPARK_DRAFT_GEMM",
    "TRTLLM_K3_FUSE_MOE_RESIDUAL",
    "TRTLLM_K3_LD_FUSED",
    "TRTLLM_K3_SHARED_SM",
    "TRTLLM_K3_ROUTER_EXACT",
    "TRTLLM_K3_LATENT_AR_OWN_WS",
    "TRTLLM_K3_ATTN_RES_MXFP8",
    "TRTLLM_K3_ULL_HYGIENE",
    "TRTLLM_K3_ULL_FRONT",
    "TLLM_KDA_PREPAD_PREFILL",
    "TRTLLM_KDA_FUSED_CONV_PREFILL",
    "TRTLLM_K3_KDA_L2_PREFETCH",
    "KIMI_K3_ATTN_RES_TOPOLOGY",
    "KIMI_K3_FUSED_ATTN_RES",
    "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS",
    "TLLM_DSPARK_AR_PUSH",
    "TLLM_DSPARK_CTX_SIDE_STREAM",
    "TLLM_DSPARK_DRAFT_FUSION",
    "TLLM_DSPARK_FUSED_MARKOV",
    "TLLM_DSPARK_MARKOV_V2",
    "TLLM_DSPARK_MARKOV_V3",
    "TLLM_K3_MLA_GEN_BACKEND",
    "TLLM_KDA_BUF_CACHE_ENTRIES",
    "TLLM_SPEC_VOCAB_PARALLEL",
    "TLLM_SPEC_VOCAB_PARALLEL_COMM",
    "TLLM_SPEC_VOCAB_PARALLEL_SAMPLING",
    "TRTLLM_BATCHED_INPUT_COMMIT",
    "TRTLLM_BATCHED_INPUT_COMMIT_VERIFY",
    "TRTLLM_FP4_MLA_LPT_PLAN",
    "TRTLLM_FP4_MLA_PACKED_WAVE_PLAN",
    "TRTLLM_FP4_MLA_REDUCE_SPREAD_STATS",
    "TRTLLM_FP4_MLA_SPLIT_KV",
    "TRTLLM_HELIX_V_UP_BEFORE_A2A",
    "TRTLLM_K3_AR_FINISH_FUSED",
    "TRTLLM_K3_AR_PUSH",
    "TRTLLM_K3_AR_PUSH_AMID",
    "TRTLLM_K3_AR_PUSH_ONE_SHOT_MAX_BYTES",
    "TRTLLM_K3_ATTN_RES_SM_RESERVE",
    "TRTLLM_K3_BF16_OVER_PREQUANTIZED_MAX_M",
    "TRTLLM_K3_DRAFT3_QHEAD",
    "TRTLLM_K3_DRAFT3_REDUCE_UNROLL",
    "TRTLLM_K3_FC12_DIE_LAYOUT",
    "TRTLLM_K3_FC12_DIE_SCHED",
    "TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS",
    "TRTLLM_K3_FC12_PREZERO",
    "TRTLLM_K3_FC12_SCHED_LOOKAHEAD",
    "TRTLLM_K3_FC12_SM_RESERVE",
    "TRTLLM_K3_FP4_MLA_SHARED_DECODE",
    "TRTLLM_K3_FP8W_LOWM_FUSED_A",
    "TRTLLM_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES",
    "TRTLLM_K3_HELIX_TAIL_CTAS",
    "TRTLLM_K3_HELIX_TAIL_FUSED",
    "TRTLLM_K3_HELIX_TAIL_GATE",
    "TRTLLM_K3_HELIX_TAIL_MB",
    "TRTLLM_K3_KDA_DECODE_EARLY_TRIGGER",
    "TRTLLM_K3_KDA_DECODE_FUSED_FAB",
    "TRTLLM_K3_KDA_L2_PREFETCH_MAX_ROWS",
    "TRTLLM_K3_KDA_MTP_V2",
    "TRTLLM_K3_KDA_MX_PDL",
    "TRTLLM_K3_KDA_QKVG_STAGE",
    "TRTLLM_K3_KDA_QKVG_STAGE_RING",
    "TRTLLM_K3_LATENT_AR_CUTE",
    "TRTLLM_K3_LATENT_AR_ONESHOT_MAX_BYTES",
    "TRTLLM_K3_LATENT_AR_OWN_WS_MAX_M",
    "TRTLLM_K3_LATENT_AR_ROW",
    "TRTLLM_K3_LATENT_AR_ROW_MAX_M",
    "TRTLLM_K3_LATENT_STREAM_PRIORITY",
    "TRTLLM_K3_LD_FUSED_MAX_M",
    "TRTLLM_K3_LD_FUSED_MIN_M",
    "TRTLLM_K3_LOWM_SF_SLOTS",
    "TRTLLM_K3_LU_MID_MAX_M",
    "TRTLLM_K3_LU_MID_MXFP8",
    "TRTLLM_K3_MID_AR_PUSH",
    "TRTLLM_K3_MID_AR_PUSH_MAX_M",
    "TRTLLM_K3_MID_MXFP8_IN",
    "TRTLLM_K3_MID_OPROJ_PC",
    "TRTLLM_K3_MLA_CTX_FP8_KV_FUSED",
    "TRTLLM_K3_MLA_CTX_TRITON_QUANT",
    "TRTLLM_K3_MLA_GATE_SIDE",
    "TRTLLM_K3_MLA_GATE_SIDE_MAX_M",
    "TRTLLM_K3_MLA_Q_FUSED",
    "TRTLLM_K3_MLA_Q_FUSED_KV_PRIO",
    "TRTLLM_K3_MLA_Q_FUSED_MAX_TOKENS",
    "TRTLLM_K3_MLA_Q_FUSED_NORM",
    "TRTLLM_K3_OPROJ_PC",
    "TRTLLM_K3_ROUTER_FIRST",
    "TRTLLM_K3_ROUTER_FIRST_MAX_M",
    "TRTLLM_K3_ROUTE_APPROX",
    "TRTLLM_K3_ROUTE_GEMV_MAX_M",
    "TRTLLM_K3_ROUTE_TAIL",
    "TRTLLM_K3_ROUTE_TAIL_MAX_M",
    "TRTLLM_K3_ROUTE_TAIL_ORDER",
    "TRTLLM_K3_ROUTE_TAIL_ORDER_MIN_ROWS",
    "TRTLLM_K3_ROUTING_STATS_CHUNK",
    "TRTLLM_K3_ROUTING_STATS_DIR",
    "TRTLLM_K3_ROUTING_STATS_EVERY",
    "TRTLLM_K3_ROUTING_STATS_STRIDE",
    "TRTLLM_K3_RP_TAIL",
    "TRTLLM_K3_RP_TAIL_MAX_M",
    "TRTLLM_K3_SHARED_STREAM_PRIORITY",
    "TRTLLM_K3_SHARED_TAIL_MAX_M",
    "TRTLLM_K3_SHARED_TAIL_MIN_M",
    "TRTLLM_K3_SHARED_TAIL_PRIO",
    "TRTLLM_K3_ULL2_COLL_AR_EARLY_STATE",
    "TRTLLM_K3_ULL2_DRAFT_TAIL",
    "TRTLLM_K3_ULL3_MLA_LATE_TAIL_MAX_TOKENS",
    "TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER",
    "TRTLLM_K3_ULL_EXPERTS",
    "TRTLLM_K3_ULL_EXPERTS_COMBINE",
    "TRTLLM_K3_ULL_EXPERTS_CTAS",
    "TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR",
    "TRTLLM_K3_ULL_EXPERTS_TRIGGER",
    "TRTLLM_K3_ULL_FRONT_EARLY",
    "TRTLLM_K3_ULL_FRONT_MAX_M",
    "TRTLLM_K3_ULL_MOEBLK",
    "TRTLLM_K3_ULL_MOETP",
    "TRTLLM_K3_ULL_SHARED_PERSIST",
    "TRTLLM_K3_ULL_TAIL1",
)

# Variables read under another name -> that name.
_RENAMED_ENV = {
    "TRTLLM_MNNVL_AR_ONESHOT_MAX_BYTES": "TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES",
}

#: Environment variables the target does not read -> the field that replaces them (None: no field;
#: ``_RENAMED_ENV`` names the variable read instead).
RETIRED_ENV: Dict[str, Optional[str]] = {
    **{env: field for field, env in _FIELD_ENV.items()},
    **{env: None for env in _FIXED_ENV},
    **{env: None for env in _RENAMED_ENV},
}

_ACTIVE = PROFILE


def active() -> K3Features:
    """This target's settings (:data:`PROFILE`)."""
    return _ACTIVE
