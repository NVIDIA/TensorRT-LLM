# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM Rubin MegaMoE runner builds its own gen-phase kernel on its phase_interleave tactics."""

import pytest
import torch

import tensorrt_llm  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="The high-throughput GLM target requires SM107",
)


@pytest.mark.parametrize("bucket", [32, 64, 128, 192])
def test_runner_builds_its_phase_interleave_genphase_kernel(
    monkeypatch: pytest.MonkeyPatch, bucket: int
) -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    if not megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        pytest.skip(str(megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON))
    from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__high_throughput.ops.moe import (  # noqa: E501
        megamoe_runner as glm_runner,
    )

    # Construction only; the occupancy query would need a CUDA context.
    monkeypatch.setattr(megamoe_op, "get_sm_version", lambda: 107)
    monkeypatch.setattr(megamoe_op, "_max_active_clusters", lambda size, *args: 184 // size)
    runner = glm_runner.RubinMegaMoERunner(
        local_rank=0,
        output_dtype=torch.bfloat16,
        world_size=4,
        num_topk=8,
        num_experts_per_rank=64,
        hidden_size=6144,
        intermediate_size_per_partition=2048,
        expand_intermediate_size_per_partition=4096,
        max_tokens_per_rank=bucket,
        apply_topk_in_fc1=True,
        in_kernel_fc2_reduce=megamoe_op._sm107_use_in_kernel_fc2_reduce(bucket, True),
    )
    tactic = runner._default_tactic(bucket)
    assert tactic == megamoe_op._unpack_tactic(glm_runner._GLM_GENPHASE_TACTICS[bucket])
    assert runner._build_kernel_once(tactic).schedule_mode == "phase_interleave"
    # The shared gen-phase table runs the decode scheduler, which GLM's kernel lacks.
    shared_tactic = megamoe_op._unpack_tactic(megamoe_op._SM107_GENPHASE_TACTICS[bucket])
    with pytest.raises(ValueError, match="no decode scheduler"):
        runner._build_kernel_once(shared_tactic)
    with pytest.raises(ValueError, match="does not support combine_dedup"):
        runner._build_kernel_once(megamoe_op._with_combine_dedup(tactic))
