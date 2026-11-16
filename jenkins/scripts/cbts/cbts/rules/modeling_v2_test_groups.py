# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 test ownership and cross-directory selection associations."""

M3_UNIT_DIR = "unittest/_torch/modeling_v2/models/minimax_m3"
_M3_SM107_TARGETS = {tp: f"{M3_UNIT_DIR}/minimax_m3_nvfp4__sm_107__tp{tp}" for tp in (2, 4, 8)}
_M3_SM107_ROUTER_TESTS = {
    tp: f"{M3_UNIT_DIR}/minimax_m3_nvfp4__sm_107__tp{tp}/ops/moe/test_modeling_v2_m3_router_tc.py"
    for tp in (2, 4, 8)
}
_M3_SM107_MOE_TEST_GROUPS = {
    tp: tuple(f"{_M3_SM107_TARGETS[tp]}/{relative}" for relative in relative_paths)
    + (_M3_SM107_ROUTER_TESTS[tp],)
    for tp, relative_paths in {
        2: (
            "ops/moe/test_modeling_v2_m3_moe_engine.py",
            "ops/moe/test_modeling_v2_m3_moe_epoch_wrap.py",
            "ops/m3_moe/test_modeling_v2_m3_moe_flat_gate.py",
            "ops/m3_moe/test_modeling_v2_m3_moe_sh_take.py",
            "ops/moe/test_modeling_v2_m3_moe_triton_finalize.py",
            "ops/moe/test_modeling_v2_m3_fp32_router_rows.py",
            "ops/moe/test_modeling_v2_m3_moe_finalize_rows.py",
        ),
        4: (
            "ops/moe/test_modeling_v2_m3_moe16_engine.py",
            "ops/moe/test_modeling_v2_m3_moe_engine.py",
            "ops/m3_moe/test_modeling_v2_m3_moe_flat_gate_tp4.py",
        ),
        8: ("ops/moe/test_modeling_v2_m3_moe_engine.py",),
    }.items()
}
_K3_TP16_EP1_UNIT_DIR = (
    "unittest/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"
)
_GLM_SM107_UNIT_DIR = (
    "unittest/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency"
)

# These additive associations cover component tests in both category/ and
# ops/category/. Function/class-scoped changes keep their precise AST anchors.
MODELING_V2_TEST_GROUPS: dict[str, tuple[str, ...]] = {
    "tests/unittest/_torch/modeling_v2/moe/_fp4_moe_test_utils.py": (
        "unittest/_torch/modeling_v2",
    ),
    f"tests/{_K3_TP16_EP1_UNIT_DIR}/ops/moe/_k3_moe_engines.py": (_K3_TP16_EP1_UNIT_DIR,),
    **{
        f"tests/{_GLM_SM107_UNIT_DIR}/{relative}": (
            f"{_GLM_SM107_UNIT_DIR}/gemm",
            f"{_GLM_SM107_UNIT_DIR}/ops/gemm/test_modeling_v2_router_gemm.py",
        )
        for relative in (
            "ops/gemm/test_modeling_v2_router_gemm.py",
            "gemm/test_modeling_v2_dense_mlp_small_m.py",
            "gemm/test_modeling_v2_draft_hidden_half.py",
            "gemm/test_modeling_v2_fused_a_gemm.py",
            "gemm/test_modeling_v2_indexer_q_tail.py",
            "gemm/test_modeling_v2_low_m_gemm.py",
            "gemm/test_modeling_v2_o_proj_publish.py",
            "gemm/test_modeling_v2_q_b_fp8_rope.py",
        )
    },
    **{
        f"tests/{M3_UNIT_DIR}/minimax_m3_nvfp4__sm_103__tp{tp}/{relative}": (
            f"{M3_UNIT_DIR}/minimax_m3_nvfp4__sm_103__tp{tp}/comm",
            f"{M3_UNIT_DIR}/minimax_m3_nvfp4__sm_103__tp{tp}/ops/comm",
        )
        for tp in (4, 8)
        for relative in (
            "comm/_rank_job.py",
            "comm/_m3_allreduce_op_matrix.py",
            "comm/test_modeling_v2_m3_allreduce.py",
            "ops/comm/_m3_mc_allreduce_norm_op_matrix.py",
            "ops/comm/test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py",
            "ops/comm/test_modeling_v2_m3_mc_finalize_allreduce_norm.py",
        )
    },
    **{
        f"tests/{target}/{relative}": (target,)
        for tp, target in _M3_SM107_TARGETS.items()
        for relative in (
            "comm/_rank_job.py",
            "comm/_m3_allreduce_op_matrix.py",
            "ops/comm/_m3_mc_allreduce_norm_op_matrix.py",
            *(("ops/comm/_m3_moe_push_op_matrix.py",) if tp == 2 else ()),
            *(("ops/moe/_m3_moe_push_op_matrix.py",) if tp == 8 else ()),
            *(
                (
                    "comm/_m3_capture_in_boundary_op_matrix.py",
                    "comm/_m3_draft_ar_norm_op_matrix.py",
                )
                if tp == 4
                else ()
            ),
            *(("comm/_sampled_accept_tp8.py",) if tp == 8 else ()),
        )
    },
    **{
        f"tests/{path}": group
        for tp, group in _M3_SM107_MOE_TEST_GROUPS.items()
        for path in (
            *group,
            f"{_M3_SM107_TARGETS[tp]}/ops/moe/_m3_moe_operands.py",
            f"{M3_UNIT_DIR}/minimax_m3_nvfp4__sm_107__tp{tp}/ops/moe/_m3_router.py",
        )
    },
}
