# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

DIM = 128
WINDOW = 16
SCALE = DIM**-0.5
LOWER_BOUND = -5.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f() or not IS_CUTLASS_DSL_AVAILABLE,
    reason="KDA MTP replay requires CUTLASS DSL on datacenter Blackwell",
)


def _randn(shape, scale=1.0, shift=0.0):
    return torch.randn(shape, device="cuda", dtype=torch.float32) * scale + shift


def _conv4_silu(raw_x, conv_weight, conv_state, slots):
    old = conv_state.index_select(0, slots).float()
    sequence = torch.cat((old, raw_x.float().transpose(1, 2)), dim=-1)
    terms = [
        sequence[:, :, tap : tap + raw_x.shape[1]] * conv_weight[:, tap].float()[None, :, None]
        for tap in range(4)
    ]
    convolved = torch.stack(terms).sum(dim=0)
    return (convolved * torch.sigmoid(convolved)).transpose(1, 2).to(torch.bfloat16)


@torch.no_grad()
def _replay_reference(
    raw_x,
    conv_weight,
    conv_state,
    raw_g,
    raw_beta,
    checkpoint,
    state_indices,
    is_dummy,
    history_k,
    history_u,
    history_G,
    history_len,
    A_log,
    dt_bias,
):
    batch, width, _ = raw_x.shape
    heads = history_k.shape[1]
    slots = state_indices.long()
    packed_qkv = _conv4_silu(raw_x, conv_weight, conv_state, slots)
    section = heads * DIM
    query = packed_qkv[..., :section].view(batch, width, heads, DIM).float()
    key = packed_qkv[..., section : 2 * section].view_as(query).float()
    value = packed_qkv[..., 2 * section :].view_as(query).float()
    query_norm = (
        query * SCALE / torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6)
    ).to(torch.bfloat16)
    key_norm = (key / torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6)).to(
        torch.bfloat16
    )

    gate_input = raw_g.float() + dt_bias.view(1, 1, heads, DIM)
    gate = LOWER_BOUND * torch.sigmoid(torch.exp(A_log.float())[None, None, :, None] * gate_input)
    beta = torch.sigmoid(raw_beta.float())
    lengths = history_len.index_select(0, slots).long()
    selected_k = history_k.index_select(0, slots)[:, :, :WINDOW].float()
    selected_u = history_u.index_select(0, slots)[:, :, :WINDOW].float()
    selected_G = history_G.index_select(0, slots)[:, :, :WINDOW].float()
    positions = torch.arange(WINDOW, device=raw_x.device)[None, None, :, None]
    active = positions < lengths[:, None, None, None]
    previous_G = torch.zeros(batch, heads, DIM, device=raw_x.device)
    nonempty = lengths > 0
    if bool(nonempty.any()):
        rows = torch.where(nonempty)[0]
        previous_G[rows] = selected_G[rows, :, lengths[rows] - 1]
    decay = torch.where(active, torch.exp(previous_G[:, :, None] - selected_G), 0.0)
    state = checkpoint.index_select(0, slots).float()
    state *= torch.exp(previous_G)[:, :, None, :]
    state += torch.einsum("bhtv,bhtk->bhvk", selected_u, selected_k * decay)

    output = torch.empty(batch, width, heads, DIM, device=raw_x.device, dtype=torch.bfloat16)
    current_u = torch.empty_like(output)
    current_G = torch.empty(batch, width, heads, DIM, device=raw_x.device, dtype=torch.float32)
    running_G = previous_G
    for token in range(width):
        running_G = running_G + gate[:, token]
        current_G[:, token] = running_G
        state *= torch.exp(gate[:, token])[:, :, None, :]
        key_token = key_norm[:, token].float()
        state_key = torch.einsum("bhvk,bhk->bhv", state, key_token)
        update = beta[:, token, :, None] * (value[:, token] - state_key)
        state += update[:, :, :, None] * key_token[:, :, None]
        output[:, token] = torch.einsum("bhvk,bhk->bhv", state, query_norm[:, token].float()).to(
            torch.bfloat16
        )
        current_u[:, token] = update.to(torch.bfloat16)

    live_rows = torch.where(~is_dummy)[0]
    for row in live_rows.tolist():
        slot = int(slots[row].item())
        begin = int(lengths[row].item())
        end = begin + width
        history_k[slot, :, begin:end] = key_norm[row].transpose(0, 1)
        history_u[slot, :, begin:end] = current_u[row].transpose(0, 1)
        history_G[slot, :, begin:end] = current_G[row].transpose(0, 1)
    return output


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_kda_mtp_replay_matches_reference(state_dtype):
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_replay import kda_mtp_replay

    torch.manual_seed(7)
    batch, slots, width, heads = 3, 6, 3, 3
    channels = 3 * heads * DIM
    physical_width = 4 * heads * DIM
    raw_values = _randn((batch, width, channels), 0.09).to(torch.bfloat16)
    raw_x = torch.empty_strided(
        raw_values.shape,
        (width * physical_width, physical_width, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    raw_x.copy_(raw_values)
    conv_weight = _randn((channels, 4), 0.18).to(torch.bfloat16)
    conv_state = _randn((slots, channels, 3), 0.09).to(torch.bfloat16)
    raw_g = _randn((batch, width, heads, DIM), 0.85, -0.36).to(torch.bfloat16)
    raw_beta = _randn((batch, width, heads), 1.0, 2.7).to(torch.bfloat16)
    checkpoint = _randn((slots, heads, DIM, DIM), 0.012).to(state_dtype)
    capacity = WINDOW + width
    raw_history_k = _randn((slots, heads, capacity, DIM), 0.09)
    history_k = (
        raw_history_k
        / torch.sqrt(torch.sum(raw_history_k * raw_history_k, dim=-1, keepdim=True) + 1.0e-6)
    ).to(torch.bfloat16)
    history_u = _randn((slots, heads, capacity, DIM), 0.012).to(torch.bfloat16)
    history_G = torch.cumsum(-(_randn((slots, heads, capacity, DIM), 0.07).abs() + 0.01), dim=2)
    state_indices = torch.tensor([4, 1, 5], device="cuda", dtype=torch.int32)
    history_len = torch.tensor([2, 7, 4, 11, 1, 15], device="cuda", dtype=torch.int32)
    is_dummy = torch.tensor([False, True, False], device="cuda")
    A_log = _randn((heads,), 0.42, -0.21).clamp_(-0.75, 0.9)
    dt_bias = _randn((heads * DIM,), 1.4, -4.6).clamp_(-8.0, -1.25)

    reference_k = history_k.clone()
    reference_u = history_u.clone()
    reference_G = history_G.clone()
    reference_output = _replay_reference(
        raw_x,
        conv_weight,
        conv_state,
        raw_g,
        raw_beta,
        checkpoint,
        state_indices,
        is_dummy,
        reference_k,
        reference_u,
        reference_G,
        history_len,
        A_log,
        dt_bias,
    )
    output = torch.empty_like(reference_output)
    candidate_x = torch.full(
        (batch, width, channels), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    checkpoint_before = checkpoint.clone()
    conv_before = conv_state.clone()

    kda_mtp_replay(
        raw_x,
        conv_weight,
        conv_state,
        raw_g,
        raw_beta,
        checkpoint,
        state_indices,
        is_dummy,
        history_k,
        history_u,
        history_G,
        history_len,
        A_log,
        dt_bias,
        output,
        candidate_x,
    )

    live = ~is_dummy
    torch.testing.assert_close(output[live], reference_output[live], rtol=2e-2, atol=5e-3)
    torch.testing.assert_close(candidate_x[live], raw_x[live], rtol=0, atol=0)
    torch.testing.assert_close(history_k, reference_k, rtol=1e-2, atol=1.5e-3)
    torch.testing.assert_close(history_u, reference_u, rtol=2e-2, atol=5e-3)
    torch.testing.assert_close(history_G, reference_G, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(checkpoint, checkpoint_before, rtol=0, atol=0)
    torch.testing.assert_close(conv_state, conv_before, rtol=0, atol=0)


@torch.no_grad()
def _commit_reference(
    state_views,
    conv_views,
    candidate_x,
    history_k,
    history_u,
    history_G,
    history_len,
    replay_work_items,
    accepted_tokens,
    is_dummy,
):
    state_dtype = state_views[0].dtype
    for work_item in replay_work_items:
        position = int(work_item[0].item())
        slot = int(work_item[1].item())
        start = int(work_item[2].item())
        if bool(is_dummy[position].item()):
            continue
        accepted = int(accepted_tokens[position].item())
        if accepted:
            old_conv = torch.stack([view[slot] for view in conv_views])
            accepted_x = candidate_x[:, position, :accepted].transpose(1, 2)
            new_conv = torch.cat((old_conv, accepted_x), dim=-1)[..., -3:]
            for layer, view in enumerate(conv_views):
                view[slot].copy_(new_conv[layer])

        total = start + accepted
        if total <= WINDOW:
            history_len[slot] = total
            continue
        tail = total - WINDOW
        end_G = history_G[:, slot, :, WINDOW - 1].float()
        first_G = history_G[:, slot, :, :WINDOW].float()
        first_k = history_k[:, slot, :, :WINDOW].float()
        first_u = history_u[:, slot, :, :WINDOW].float()
        decay = torch.exp(end_G[:, :, None] - first_G)
        update = torch.einsum("lhtv,lhtk->lhvk", first_u, first_k * decay)
        checkpoint = torch.stack([view[slot] for view in state_views]).float()
        committed = checkpoint * torch.exp(end_G)[:, :, None] + update
        for layer, view in enumerate(state_views):
            view[slot].copy_(committed[layer].to(state_dtype))
        if tail:
            tail_k = history_k[:, slot, :, WINDOW : WINDOW + tail].clone()
            tail_u = history_u[:, slot, :, WINDOW : WINDOW + tail].clone()
            tail_G = history_G[:, slot, :, WINDOW : WINDOW + tail].clone()
            history_k[:, slot, :, :tail] = tail_k
            history_u[:, slot, :, :tail] = tail_u
            history_G[:, slot, :, :tail] = tail_G - end_G[:, :, None]
        history_len[slot] = tail


def _make_state_views(indirect, layers, slots, heads, channels, state_dtype):
    if indirect:
        states = [_randn((slots, heads, DIM, DIM), 0.012).to(state_dtype) for _ in range(layers)]
        conv = [_randn((slots, channels, 3), 0.09).to(torch.bfloat16) for _ in range(layers)]
    else:
        state_storage = _randn((slots, layers, heads, DIM, DIM), 0.012).to(state_dtype)
        conv_storage = _randn((slots, layers, channels, 3), 0.09).to(torch.bfloat16)
        states = [state_storage[:, layer] for layer in range(layers)]
        conv = [conv_storage[:, layer] for layer in range(layers)]
    state_descriptors = torch.tensor(
        [(state.data_ptr(), state.stride(0) * state.element_size()) for state in states],
        device="cuda",
        dtype=torch.int64,
    )
    conv_descriptors = torch.tensor(
        [(state.data_ptr(), state.stride(0) * state.element_size()) for state in conv],
        device="cuda",
        dtype=torch.int64,
    )
    return states, conv, state_descriptors, conv_descriptors


@pytest.mark.parametrize("indirect", [False, True], ids=["affine", "indirect"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_kda_mtp_commit_matches_reference(indirect, state_dtype):
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_commit import kda_mtp_commit

    torch.manual_seed(11)
    layers, slots, candidate_capacity, width, heads = 2, 5, 5, 3, 3
    channels = 3 * heads * DIM
    capacity = WINDOW + width
    states, conv, state_desc, conv_desc = _make_state_views(
        indirect, layers, slots, heads, channels, state_dtype
    )
    reference_states = [state.clone() for state in states]
    reference_conv = [state.clone() for state in conv]
    candidate_x = _randn((layers, candidate_capacity, width, channels), 0.09).to(torch.bfloat16)
    raw_history_k = _randn((layers, slots, heads, capacity, DIM), 0.09)
    history_k = (
        raw_history_k
        / torch.sqrt(torch.sum(raw_history_k * raw_history_k, dim=-1, keepdim=True) + 1.0e-6)
    ).to(torch.bfloat16)
    history_u = _randn((layers, slots, heads, capacity, DIM), 0.012).to(torch.bfloat16)
    history_G = torch.cumsum(
        -(_randn((layers, slots, heads, capacity, DIM), 0.07).abs() + 0.01), dim=3
    )
    accepted_storage = torch.tensor([-1, 3, 2, 1], device="cuda", dtype=torch.int32)
    accepted_tokens = accepted_storage[1:]
    dummy_storage = torch.tensor([True, False, False, True], device="cuda")
    is_dummy = dummy_storage[1:]
    assert accepted_tokens.data_ptr() % 16 != 0
    assert is_dummy.data_ptr() % 16 != 0
    starts = torch.tensor([15, 10, 16], device="cuda", dtype=torch.int32)
    positions = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    slot_for_position = torch.tensor([1, 3, 4], device="cuda", dtype=torch.int32)
    replay_work_items = torch.stack(
        (
            positions,
            slot_for_position[positions.long()],
            starts[positions.long()],
            torch.zeros_like(positions),
        ),
        dim=1,
    ).contiguous()
    history_len = torch.tensor([4, 15, 2, 10, 16], device="cuda", dtype=torch.int32)
    reference_k = history_k.clone()
    reference_u = history_u.clone()
    reference_G = history_G.clone()
    reference_len = history_len.clone()
    _commit_reference(
        reference_states,
        reference_conv,
        candidate_x,
        reference_k,
        reference_u,
        reference_G,
        reference_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
    )

    kda_mtp_commit(
        states[0],
        conv[0],
        candidate_x,
        history_k,
        history_u,
        history_G,
        history_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
        ssm_state_descriptors=state_desc,
        conv_state_descriptors=conv_desc,
    )

    state_tolerance = (2e-2, 8e-3) if state_dtype is torch.bfloat16 else (1e-2, 1e-3)
    for actual, reference in zip(states, reference_states):
        torch.testing.assert_close(
            actual, reference, rtol=state_tolerance[0], atol=state_tolerance[1]
        )
    for actual, reference in zip(conv, reference_conv):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(history_k, reference_k, rtol=0, atol=0)
    torch.testing.assert_close(history_u, reference_u, rtol=0, atol=0)
    torch.testing.assert_close(history_G, reference_G, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(history_len, reference_len, rtol=0, atol=0)
