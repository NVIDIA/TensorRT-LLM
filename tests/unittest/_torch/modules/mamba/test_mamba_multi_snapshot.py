# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.gdn_multi_snapshot import gather_conv_snapshot_windows
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mamba.mamba_multi_snapshot import store_mamba_snapshots
from tensorrt_llm._torch.modules.mamba.ssd_state_passing import _state_passing_fwd
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.multi_snapshot import (
    PeriodicSnapshotBatch,
)

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


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_mamba_mixer_saves_selected_prefix_states(state_dtype):
    """The mixer saves nonadjacent checkpoints in the requested cache slots."""
    torch.manual_seed(916)
    tokens, heads, head_dim, d_state, groups, chunk = 512, 2, 32, 32, 1, 128
    inner, conv_dim, width = heads * head_dim, heads * head_dim + 2 * groups * d_state, 3
    projected = (0.1 * torch.randn(tokens, inner + conv_dim + heads, device="cuda")).bfloat16()
    initial_conv = (0.1 * torch.randn(8, conv_dim, width, device="cuda")).bfloat16()
    initial_ssm = (0.01 * torch.randn(8, heads, head_dim, d_state, device="cuda")).to(state_dtype)
    mixer = SimpleNamespace(
        layer_idx=0,
        tp_d_inner=inner,
        tp_conv_dim=conv_dim,
        tp_ngroups=groups,
        tp_nheads=heads,
        head_dim=head_dim,
        d_state=d_state,
        chunk_size=chunk,
        _token_major_conv=True,
        _mamba_ssm_cache_dtype=state_dtype,
        delta_softplus=True,
        A=-torch.rand(heads, device="cuda") - 1,
        D=torch.rand(heads, device="cuda"),
        dt_bias=torch.rand(heads, device="cuda"),
        conv1d=SimpleNamespace(
            weight=(0.1 * torch.randn(conv_dim, width + 1, device="cuda")).bfloat16(),
            bias=(0.1 * torch.randn(conv_dim, device="cuda")).bfloat16(),
        ),
    )
    points, pending_slots = [128, 384], [5, 2]
    snapshot = PeriodicSnapshotBatch(
        interval=chunk,
        checkpoint_count=4,
        checkpoint_cu_starts=torch.tensor([0, 4], device="cuda"),
        checkpoint_rows=torch.tensor([0, 2], device="cuda"),
        positions=torch.tensor(points, device="cuda"),
        conv_indices=torch.tensor(
            [i for point in points for i in range(point - width, point)], device="cuda"
        ),
        slots=torch.tensor(pending_slots, device="cuda"),
    )

    def run(length, periodic=None):
        conv, ssm = initial_conv.clone(), initial_ssm.clone()
        metadata = SimpleNamespace(
            state_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
            seq_idx=torch.zeros(1, length, device="cuda", dtype=torch.int32),
            has_initial_states=torch.ones(1, device="cuda", dtype=torch.bool),
            use_initial_states=True,
            chunk_indices=torch.arange(length // chunk, device="cuda", dtype=torch.int32),
            chunk_offsets=torch.zeros(length // chunk, device="cuda", dtype=torch.int32),
            periodic_snapshot=periodic,
        )
        layer_cache = SimpleNamespace(conv=conv, temporal=ssm)
        attention = SimpleNamespace(
            num_contexts=1,
            seq_lens=torch.tensor([length], device="cuda"),
            num_ctx_tokens=length,
            num_tokens=length,
            kv_cache_manager=SimpleNamespace(mamba_layer_cache=lambda _: layer_cache),
        )
        output = torch.empty(length, inner, device="cuda", dtype=torch.bfloat16)
        Mamba2Mixer.forward_core(
            mixer, projected[:length].clone(), attention, metadata, None, output
        )
        return output, conv, ssm

    expected_output, expected_conv, expected_ssm = run(tokens)
    actual_output, actual_conv, actual_ssm = run(tokens, snapshot)
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(actual_conv[0], expected_conv[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_ssm[0], expected_ssm[0], rtol=1e-2, atol=2e-3)
    for point, slot in zip(points, pending_slots):
        _, prefix_conv, prefix_ssm = run(point)
        torch.testing.assert_close(actual_conv[slot], prefix_conv[0], rtol=0, atol=0)
        torch.testing.assert_close(actual_ssm[slot], prefix_ssm[0], rtol=1e-2, atol=2e-3)
