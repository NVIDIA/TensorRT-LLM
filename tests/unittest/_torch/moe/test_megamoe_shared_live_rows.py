# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused shared expert's live-row bound, without a GPU.

The kernel bounds the shared expert's rows with the device twin of ``live_rows.live_row_count``
(``kernels/cutedsl_megamoe/helpers/live_rows.py``); the host twin there must agree with the
engine's rule on every word. The Blackwell kernel takes the bound only when built with
``shared_live_rows``, which keys its name and needs a fused shared expert.
"""

import pytest

from tensorrt_llm._torch.moe import live_rows

pytestmark = pytest.mark.cpu_only


def test_kernel_rule_is_the_engine_rule():
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.live_rows import (
        live_row_count_host,
    )

    for requests in (0, 1, 2, 3, 4, 8, 255, live_rows.MAX_REQUESTS):
        for real in sorted({0, min(1, requests), requests // 2, max(requests - 1, 0), requests}):
            word = live_rows.encode(real, requests)
            for rows in (0, 1, 3, 8, 32, 96, 1024, 8192):
                assert live_row_count_host(word, rows) == live_rows.live_row_count(word, rows), (
                    word,
                    rows,
                )
    # A word outside the rule (more real than requests) keeps every row, as the engine's does.
    word = (4 << live_rows.REQUEST_BITS) | 6
    assert live_row_count_host(word, 64) == live_rows.live_row_count(word, 64) == 64


def _blackwell_kernel(**impl_changes):
    pytest.importorskip("cutlass")
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import ImplDesc, ProblemDesc
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (  # noqa: E501
        BlockScaledSwapAbMegaMoeKernel,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import CombineFormat

    shared = impl_changes.pop("shared", True)
    problem = dict(
        expert_count=16,
        intermediate_gateup_size=512,
        hidden_size=1024,
        quant_kind="nvfp4",
        a_major_mode=OperandMajorMode.K,
        b_major_mode=OperandMajorMode.K,
        combine_format=CombineFormat.parse("bf16"),
        gate_up_clamp=None,
        world_size=4,
        topk=4,
        topk_index_dtype=cutlass.Int64,
        max_tokens_per_rank=128,
        apply_topk_at_fc1=True,
    )
    if shared:
        problem.update(shared_intermediate_size=1024, shared_gate_up_clamp=None)
    impl = dict(
        mma_tiler_mnk=(256, 128, 256),
        cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        schedule_policy=("grouped", 512),
        token_padding_block=64,
        sf_padding_block=128,
        work_id_mode="atomic_counter",
        fc2_use_bulk=False,
        epi_flag_batches=(2, 4),
        launch_cluster_count=64,
        token_in_flag_batch=1,
        token_back_mode="standalone_warps",
        reduce_topk_in_kernel=False,
    )
    impl.update(impl_changes)
    return BlockScaledSwapAbMegaMoeKernel(ProblemDesc(problem), ImplDesc(impl))


def test_shared_live_rows_keys_the_kernel():
    plain = _blackwell_kernel()
    bounded = _blackwell_kernel(shared_live_rows=True)
    assert "shared_live_rows" in type(plain).impl_desc_require()
    assert not plain.shared_live_rows and bounded.shared_live_rows
    assert bounded.name() == plain.name().replace(
        plain._shared_name_suffix(), bounded._shared_name_suffix()
    )
    assert bounded._shared_name_suffix() == plain._shared_name_suffix() + "_live_v1"
    assert bounded._shared_param_keys == plain._shared_param_keys + ("live_word",)
    with pytest.raises(ValueError, match="shared_live_rows"):
        _blackwell_kernel(shared=False, shared_live_rows=True)
