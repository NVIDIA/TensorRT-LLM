# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.gdn_multi_snapshot import gather_conv_snapshot_windows
from tensorrt_llm._torch.modules.mamba.mamba_multi_snapshot import store_mamba_snapshots
from tensorrt_llm._torch.modules.mamba.ssd_state_passing import _state_passing_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_ssd_boundaries_and_conv_windows_match_independent_recurrence(state_dtype):
    torch.manual_seed(914)
    nchunks, heads, dim, width = 5, 2, 8, 128
    contributions = torch.randn(1, nchunks, heads, dim * dim, device="cuda") * 0.1
    log_decay = -torch.rand(1, heads, nchunks, width, device="cuda") * 0.2
    initial = torch.randn(1, heads, dim * dim, device="cuda") * 0.1
    chunk_states, _ = _state_passing_fwd(
        contributions, log_decay, initial_states=initial, chunk_size=width, out_dtype=state_dtype
    )
    raw_conv = torch.randn(nchunks * width, 16, device="cuda", dtype=torch.bfloat16)
    conv_width = 3
    points, slot_ids = [128, 256, 384], [5, 2, 7]
    conv_indices = torch.tensor(
        [i for point in points for i in range(point - conv_width, point)],
        device="cuda",
        dtype=torch.int64,
    )
    windows = gather_conv_snapshot_windows(raw_conv, conv_indices, len(points), conv_width)
    ssm = torch.randn(8, heads, dim, dim, device="cuda", dtype=state_dtype)
    conv = torch.randn(8, 16, conv_width, device="cuda", dtype=torch.bfloat16)
    before_ssm, before_conv = ssm.clone(), conv.clone()
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int64)
    checkpoints = (
        chunk_states[0]
        .view(nchunks, heads, dim, dim)
        .index_select(0, torch.tensor([1, 2, 3], device="cuda"))
    )
    store_mamba_snapshots(checkpoints, ssm, conv, windows, slots)

    state = initial.cpu()
    expected = []
    for i in range(3):
        decay = log_decay[:, :, i, -1].cpu().exp().unsqueeze(-1)
        state = decay * state + contributions[:, i].cpu()
        expected.append(state.view(heads, dim, dim).to(state_dtype))
    for row, (slot, point) in enumerate(zip(slot_ids, points)):
        torch.testing.assert_close(ssm[slot].cpu(), expected[row], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            conv[slot], raw_conv[point - conv_width : point].T, rtol=0, atol=0
        )
    untouched = [slot for slot in range(8) if slot not in slot_ids]
    assert torch.equal(ssm[untouched], before_ssm[untouched])
    assert torch.equal(conv[untouched], before_conv[untouched])


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_ssd_prefill_checkpoints_match_separate_prefix_prefills(state_dtype):
    from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

    torch.manual_seed(915)
    tokens, heads, dim, states, chunk = 512, 2, 32, 32, 128
    x = (0.1 * torch.randn(1, tokens, heads, dim, device="cuda")).bfloat16()
    dt = torch.randn(1, tokens, heads, device="cuda").bfloat16()
    a = -torch.rand(heads, device="cuda") - 1
    b = (0.1 * torch.randn(1, tokens, 1, states, device="cuda")).bfloat16()
    c = (0.1 * torch.randn(1, tokens, 1, states, device="cuda")).bfloat16()
    initial = (0.01 * torch.randn(1, heads, dim, states, device="cuda")).to(state_dtype)

    def scan(length, checkpoints=None):
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        count = (length + chunk - 1) // chunk
        return mamba_chunk_scan_combined(
            x[:, :length],
            dt[:, :length],
            a,
            b[:, :length],
            c[:, :length],
            chunk,
            initial_states=initial,
            seq_idx=torch.zeros(1, length, device="cuda", dtype=torch.int32),
            chunk_indices=torch.arange(count, device="cuda", dtype=torch.int32),
            chunk_offsets=torch.zeros(count, device="cuda", dtype=torch.int32),
            cu_seqlens=cu,
            dt_softplus=True,
            out=torch.empty_like(x[:, :length]),
            return_varlen_states=True,
            state_dtype=state_dtype,
            checkpoint_chunks=checkpoints,
        )

    _, checkpoints = scan(tokens, torch.tensor([1, 2, 3], device="cuda"))
    for row, point in enumerate((128, 256, 384)):
        expected = scan(point)
        torch.testing.assert_close(checkpoints[row], expected[0], rtol=1e-2, atol=2e-3)
