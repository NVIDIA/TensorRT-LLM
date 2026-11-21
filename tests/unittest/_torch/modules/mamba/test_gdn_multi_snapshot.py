# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.gdn_multi_snapshot import (
    gather_conv_snapshot_windows,
    store_gdn_snapshots,
)
from tensorrt_llm._utils import is_flashinfer_gdn_prefill_supported_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_stores_checkpoints_and_convolution_windows_without_touching_live_slots(state_dtype):
    torch.manual_seed(739)
    count, width, conv_dim = 3, 3, 16
    slots_cpu = [6, 1, 9]
    endpoints = [64, 128, 197]
    slots = torch.tensor(slots_cpu, device="cuda", dtype=torch.int32)
    indices = torch.tensor(
        [i for end in endpoints for i in range(end - width, end)], device="cuda", dtype=torch.int64
    )
    mixed_qkv = torch.randn(220, conv_dim, device="cuda", dtype=torch.bfloat16)
    checkpoint_states = torch.randn(count, 2, 64, 64, device="cuda", dtype=torch.float32)
    ssm = torch.randn(12, 2, 64, 64, device="cuda", dtype=state_dtype)
    conv = torch.randn(12, conv_dim, width, device="cuda", dtype=torch.bfloat16)
    original_ssm, original_conv = ssm.clone(), conv.clone()

    windows = gather_conv_snapshot_windows(mixed_qkv, indices, count, width)
    store_gdn_snapshots(ssm, conv, checkpoint_states, windows, slots)

    for row, (slot, end) in enumerate(zip(slots_cpu, endpoints)):
        torch.testing.assert_close(
            ssm[slot], checkpoint_states[row].to(state_dtype), rtol=0, atol=0
        )
        torch.testing.assert_close(conv[slot], mixed_qkv[end - width : end].T, rtol=0, atol=0)
    untouched = [slot for slot in range(12) if slot not in slots_cpu]
    assert torch.equal(ssm[untouched], original_ssm[untouched])
    assert torch.equal(conv[untouched], original_conv[untouched])


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_flashinfer_gdn_prefill_supported_arch(),
    reason="FlashInfer GDN prefill requires SM90, SM100/SM103/SM107 or SM120/SM121",
)
def test_prefill_checkpoints_store_states_for_each_reusable_prefix(state_dtype):
    from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule

    torch.manual_seed(743)
    tokens, heads, dim, interval, width = 256, 2, 128, 64, 3
    q = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1
    ).bfloat16()
    v = (torch.randn(1, tokens, heads, dim, device="cuda") * 0.1).bfloat16()
    g = -0.05 * torch.rand(1, tokens, heads, device="cuda")
    beta = torch.rand(1, tokens, heads, device="cuda")
    ssm = (0.01 * torch.randn(8, heads, dim, dim, device="cuda")).to(state_dtype)
    live_before = ssm[0].clone()
    conv = torch.randn(8, 16, width, device="cuda", dtype=torch.bfloat16)
    mixed_qkv = torch.randn(tokens, 16, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([5, 2, 7], device="cuda", dtype=torch.int32)
    points = [64, 128, 192]
    indices = torch.tensor(
        [i for point in points for i in range(point - width, point)],
        device="cuda",
        dtype=torch.int64,
    )
    checkpoints = torch.empty(4, heads, dim, dim, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    checkpoint_cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    windows = gather_conv_snapshot_windows(mixed_qkv, indices, len(points), width)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=ssm[:1],
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=True,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=checkpoint_cu,
        checkpoint_every_n_tokens=interval,
    )
    store_gdn_snapshots(ssm, conv, checkpoints[: len(points)], windows, slots)

    for row, point in enumerate(points):
        _, expected = chunk_gated_delta_rule(
            q[:, :point],
            k[:, :point],
            v[:, :point],
            g[:, :point],
            beta[:, :point],
            initial_state=live_before.unsqueeze(0),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, point], device="cuda", dtype=torch.int32),
            use_cp=True,
        )
        torch.testing.assert_close(
            ssm[slots[row]], expected[0].to(state_dtype), rtol=1.6e-2, atol=1e-3
        )
        torch.testing.assert_close(
            conv[slots[row]], mixed_qkv[point - width : point].T, rtol=0, atol=0
        )
    assert torch.equal(ssm[0], live_before)
