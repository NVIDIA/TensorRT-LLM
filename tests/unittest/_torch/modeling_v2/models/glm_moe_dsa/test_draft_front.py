# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3),
    reason="draft_front requires an SM103 GPU",
)
def test_pack_draft_front_matches_flashinfer_rmsnorm() -> None:
    from flashinfer.norm import rmsnorm

    draft_front = import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa."
        "glm_5_2_nvfp4__sm_103__tp8.kernels.draft_front"
    )
    native_norm = import_module(f"{draft_front.__name__}.native_norm")

    device = torch.device("cuda", 0)
    reason = native_norm.flashinfer_path_reason(device)
    assert reason is None, reason
    generator = torch.Generator(device=device).manual_seed(123)
    hidden_size = 7168
    tp_size = 8
    eps = 1e-6
    enorm_weight = torch.randn(
        hidden_size, generator=generator, device=device, dtype=torch.bfloat16
    )
    hnorm_weight = torch.randn(
        hidden_size, generator=generator, device=device, dtype=torch.bfloat16
    )
    for rows in (1, 17, 33):
        embeds = torch.randn(
            (rows, hidden_size), generator=generator, device=device, dtype=torch.bfloat16
        )
        hidden_states = torch.randn(
            (rows, hidden_size), generator=generator, device=device, dtype=torch.bfloat16
        )
        expected = torch.cat(
            (rmsnorm(embeds, enorm_weight, eps), rmsnorm(hidden_states, hnorm_weight, eps)), dim=-1
        ).chunk(tp_size, dim=-1)
        for rank in range(tp_size):
            actual = draft_front.pack_draft_front(
                embeds,
                hidden_states,
                enorm_weight,
                eps,
                hnorm_weight,
                eps,
                tp_rank=rank,
                tp_size=tp_size,
            )
            assert actual is not None
            assert torch.equal(actual, expected[rank]), f"rows={rows}, tp_rank={rank}"
