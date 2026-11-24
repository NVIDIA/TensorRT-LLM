# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused latent-MoE tail's instances per attention-TP size.

TP32 has a second, two-shot-only instance above its one-shot tail's 32 rows. Every other instance keeps what it
had before that instance existed: the TP4 / TP8 / TP16 tails and the TP32 one-shot tail keep their token
counts (capacity, dispatch bound, one-shot bound), their numerics flag, their up-projection tactics, and their
two-shot collective's all-rank wait and rank-order sums. The model builds the second instance only where
``fused_tail_twoshot_only_rows`` names rows (TP32), so the TP8 / TP16 dispatch is the default tail's alone.
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (
    moe_tail_latent as lmt,
)

# (min tokens, capacity, dispatch max, one-shot max) of the default instance per TP size, before the
# two-shot-only instance.
BASE_ROWS = {4: (1, 192, 128, 32), 8: (1, 192, 128, 32), 16: (1, 192, 192, 16), 32: (1, 32, 32, 32)}


@pytest.mark.parametrize("tp", sorted(BASE_ROWS))
def test_default_instance_rows_unchanged(tp: int) -> None:
    assert tuple(lmt.fused_tail_rows(tp, lmt.MAX_NUM_TOKENS)) == BASE_ROWS[tp]


def test_twoshot_only_instance_at_tp32_only() -> None:
    assert lmt.fused_tail_twoshot_only_rows(32) == (33, 192)
    for tp in (4, 8, 16):
        assert lmt.fused_tail_twoshot_only_rows(tp) is None
        with pytest.raises(ValueError):
            lmt.fused_tail_rows(tp, lmt.MAX_NUM_TOKENS, twoshot_only=True)
    wide = lmt.fused_tail_rows(32, lmt.MAX_NUM_TOKENS, twoshot_only=True)
    assert tuple(wide) == (33, 192, 192, 0)
    # The two TP32 instances take adjacent row ranges.
    assert wide.min_tokens == lmt.fused_tail_rows(32, lmt.MAX_NUM_TOKENS).dispatch_max + 1


def test_numerics_flags_unchanged() -> None:
    assert lmt.MNNVL_NORM_TP_SIZES == frozenset({8, 16})
    assert [lmt.fused_tail_mnnvl_norm(tp) for tp in (4, 8, 16, 32)] == [False, True, True, False]
    assert lmt.fused_tail_mnnvl_norm(32, twoshot_only=True)
    assert lmt._SPLITK_TOKEN_BUCKETS == (8, 16, 32, 64, 128)


def _base_tactic(sk, m: int, shard_dim: int):
    """``tactic_for_m`` as it was before the two-shot-only instance."""
    if shard_dim == 896 and 8 < m <= 64:
        return sk.SplitKTactic(64, 8 if m <= 16 else 16, 4 if m <= 32 else 2, 7)
    if m <= 32 and (shard_dim >= 896 or m <= 16):
        base = sk.SplitKTactic(64, next(n for n in (8, 16, 32) if n >= m), 4, 1)
        stages = min(sk._max_ab_stages_for(base, sk._SMEM_CAPACITY_BYTES), 7)
        return sk.SplitKTactic(base.mma_m, base.mma_n, base.split_k, stages)
    if shard_dim >= 896:
        return sk.SplitKTactic(64, 32, 2, 7)
    return sk.SplitKTactic(64, 16 if m <= 64 else 32, 4, 7)


def test_up_projection_tactics_unchanged() -> None:
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_tail_splitk_multicast_gemm as sk,
    )

    # TP4 / TP8 / TP16 shards at every row, the TP32 one-shot tail's rows.
    for shard, rows in ((1792, 192), (896, 192), (448, 192), (224, 32)):
        for m in range(1, rows + 1):
            assert sk.tactic_for_m(m, shard) == _base_tactic(sk, m, shard), (shard, m)
    # The two-shot-only instance accumulates the whole K (the unfused cuBLAS GEMM's split at its rows).
    for m in range(33, 193):
        assert sk.whole_k_tactic_for_m(m).split_k == 1


def test_twoshot_collective_sum_orders() -> None:
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_tail_twoshot_collective as tc,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_mnnvl_norm import (  # noqa: E501
        MnnvlNormPlan,
    )

    def plan(tp: int) -> MnnvlNormPlan:
        return MnnvlNormPlan(
            dim=3584,
            oneshot_max_rows=(1 << 20) // (3584 * tp * 2),
            oneshot_rank_chunk=tp if tp <= 8 else 8,
            twoshot_rank_chunk=tp if tp < 16 else 16,
            geometries=((192, 448, 1),),
        )

    shape = dict(latent_dim=3584, hidden_dim=7168)
    # Up to 16 ranks the kernel keeps its all-rank wait and its sums (the plan's for the routed rows, rank order
    # for the shared slices), whatever the plan.
    for tp in (4, 8, 16):
        for norm_plan in (None, plan(tp)):
            assert tc.rank_sum_chunks(tp_size=tp, min_m=1, norm_plan=norm_plan, **shape) == (tp, tp)
            kernel = tc.TwoShotAllReduceRMSNormReduceScatter(
                rank=0, tp_size=tp, max_m=192, norm_plan=norm_plan, **shape
            )
            assert kernel.poll_ranks == tp
    # 32 ranks: 16 ranks per wait; with the plan both sums in the MNNVL two-shot all-reduce's 16-rank chunks, which
    # needs every row two-shot.
    assert tc.rank_sum_chunks(tp_size=32, min_m=33, norm_plan=plan(32), **shape) == (16, 16)
    assert tc.rank_sum_chunks(tp_size=32, min_m=1, norm_plan=None, **shape) == (32, 32)
    with pytest.raises(ValueError):
        tc.rank_sum_chunks(tp_size=32, min_m=1, norm_plan=plan(32), **shape)
