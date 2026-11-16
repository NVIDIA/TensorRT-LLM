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
HEAD_STATE_CASES = [
    pytest.param(3, torch.bfloat16, id="kda-bf16"),
    pytest.param(3, torch.float32, id="kda-fp32"),
    pytest.param(16, torch.float32, id="glm-tp4-fp32"),
    pytest.param(64, torch.float32, id="glm-adp-fp32"),
]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f() or not IS_CUTLASS_DSL_AVAILABLE,
    reason="KDA MTP replay requires CUTLASS DSL on datacenter Blackwell",
)


def _randn(shape, scale=1.0, shift=0.0):
    return torch.randn(shape, device="cuda", dtype=torch.float32) * scale + shift


def _make_pool_history(layers, replay_slots, heads, capacity):
    """Per-layer K/U/G histories laid out like KV-cache pool pages.

    One page per replay slot holds every layer's planes, so replay slots are
    widely strided; the accepted lengths are column 0 of an int32 [R, 4] count
    row. Returns ([K], [U], [G], history_len) with [R, heads, capacity, 128]
    views per layer.
    """
    elements = heads * capacity * DIM
    page = layers * 4 * elements + 64  # K + U (BF16) + G (FP32) per layer
    pool = torch.zeros(replay_slots, page, device="cuda", dtype=torch.bfloat16)
    history_k, history_u, history_G = [], [], []
    offset = 0
    for _ in range(layers):
        shape = (replay_slots, heads, capacity, DIM)
        k = pool[:, offset : offset + elements].view(shape)
        offset += elements
        u = pool[:, offset : offset + elements].view(shape)
        offset += elements
        g = pool[:, offset : offset + 2 * elements].view(torch.float32).view(shape)
        offset += 2 * elements
        raw_k = _randn(shape, 0.09)
        k.copy_(raw_k / torch.sqrt(torch.sum(raw_k * raw_k, dim=-1, keepdim=True) + 1.0e-6))
        u.copy_(_randn(shape, 0.012))
        g.copy_(torch.cumsum(-(_randn(shape, 0.07).abs() + 0.01), dim=2))
        history_k.append(k)
        history_u.append(u)
        history_G.append(g)
    count = torch.zeros(replay_slots, 4, device="cuda", dtype=torch.int32)
    return history_k, history_u, history_G, count[:, 0]


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
    replay_indices,
):
    batch, width, _ = raw_x.shape
    heads = history_k.shape[1]
    slots = state_indices.long()
    replay_slots = replay_indices.long()
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
    lengths = history_len.index_select(0, replay_slots).long()
    selected_k = history_k.index_select(0, replay_slots)[:, :, :WINDOW].float()
    selected_u = history_u.index_select(0, replay_slots)[:, :, :WINDOW].float()
    selected_G = history_G.index_select(0, replay_slots)[:, :, :WINDOW].float()
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
        slot = int(replay_slots[row].item())
        begin = int(lengths[row].item())
        end = begin + width
        history_k[slot, :, begin:end] = key_norm[row].transpose(0, 1)
        history_u[slot, :, begin:end] = current_u[row].transpose(0, 1)
        history_G[slot, :, begin:end] = current_G[row].transpose(0, 1)
    return output


@pytest.mark.parametrize(("heads", "state_dtype"), HEAD_STATE_CASES)
@pytest.mark.parametrize(
    ("width", "strong_decay"),
    [(1, False), (4, False), (8, True)],
    ids=["single-token", "weak-decay", "strong-decay"],
)
@torch.no_grad()
def test_kda_mtp_replay_matches_reference(heads, state_dtype, width, strong_decay):
    """Live rows with partial and full histories, a dummy row, page-strided
    histories at replay slots apart from the state slots, and a strided
    history_len."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_replay import kda_mtp_replay

    torch.manual_seed(7)
    batch, slots, replay_slot_count = 3, 6, 7
    token_capacity = 8
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
    capacity = WINDOW + token_capacity
    (history_k,), (history_u,), (history_G,), history_len = _make_pool_history(
        1, replay_slot_count, heads, capacity
    )
    state_indices = torch.tensor([4, 1, 5], device="cuda", dtype=torch.int32)
    replay_indices = torch.tensor([6, 0, 2], device="cuda", dtype=torch.int32)
    history_len.copy_(torch.tensor([7, 2, 15, 11, 1, 3, 4], device="cuda", dtype=torch.int32))
    is_dummy = torch.tensor([False, True, False], device="cuda")
    A_log = _randn((heads,), 0.42, -0.21).clamp_(-0.75, 0.9)
    dt_bias = _randn((heads * DIM,), 1.4, -4.6).clamp_(-8.0, -1.25)
    if strong_decay:
        raw_g.fill_(10.0)
        A_log.zero_()
        dt_bias.zero_()
        history_G.copy_(
            LOWER_BOUND
            * torch.arange(1, capacity + 1, device="cuda", dtype=torch.float32).view(
                1, 1, capacity, 1
            )
        )
        history_len[replay_indices[~is_dummy].long()] = torch.tensor(
            [15, 16], device="cuda", dtype=torch.int32
        )

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
        replay_indices,
    )
    output = torch.empty_like(reference_output)
    candidate_x = torch.full(
        (batch, token_capacity, channels), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    checkpoint_before = checkpoint.clone()
    conv_before = conv_state.clone()
    history_len_before = history_len.clone()

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
        history_window=WINDOW,
        replay_indices=replay_indices,
    )

    live = ~is_dummy
    torch.testing.assert_close(output[live], reference_output[live], rtol=2e-2, atol=5e-3)
    torch.testing.assert_close(candidate_x[live, :width], raw_x[live], rtol=0, atol=0)
    assert torch.isnan(candidate_x[:, width:]).all()
    torch.testing.assert_close(history_k, reference_k, rtol=1e-2, atol=1.5e-3)
    torch.testing.assert_close(history_u, reference_u, rtol=2e-2, atol=5e-3)
    torch.testing.assert_close(history_G, reference_G, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(checkpoint, checkpoint_before, rtol=0, atol=0)
    torch.testing.assert_close(conv_state, conv_before, rtol=0, atol=0)
    torch.testing.assert_close(history_len, history_len_before, rtol=0, atol=0)


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
    history_k = torch.stack(history_k)
    history_u = torch.stack(history_u)
    history_G = torch.stack(history_G)
    for work_item in replay_work_items:
        position = int(work_item[0].item())
        slot = int(work_item[1].item())
        start = int(work_item[2].item())
        replay_slot = int(work_item[4].item())
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
            history_len[replay_slot] = total
            continue
        tail = total - WINDOW
        end_G = history_G[:, replay_slot, :, WINDOW - 1].float()
        first_G = history_G[:, replay_slot, :, :WINDOW].float()
        first_k = history_k[:, replay_slot, :, :WINDOW].float()
        first_u = history_u[:, replay_slot, :, :WINDOW].float()
        decay = torch.exp(end_G[:, :, None] - first_G)
        update = torch.einsum("lhtv,lhtk->lhvk", first_u, first_k * decay)
        checkpoint = torch.stack([view[slot] for view in state_views]).float()
        committed = checkpoint * torch.exp(end_G)[:, :, None] + update
        for layer, view in enumerate(state_views):
            view[slot].copy_(committed[layer].to(state_dtype))
        if tail:
            tail_k = history_k[:, replay_slot, :, WINDOW : WINDOW + tail].clone()
            tail_u = history_u[:, replay_slot, :, WINDOW : WINDOW + tail].clone()
            tail_G = history_G[:, replay_slot, :, WINDOW : WINDOW + tail].clone()
            history_k[:, replay_slot, :, :tail] = tail_k
            history_u[:, replay_slot, :, :tail] = tail_u
            history_G[:, replay_slot, :, :tail] = tail_G - end_G[:, :, None]
        history_len[replay_slot] = tail
    return history_k, history_u, history_G


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
@pytest.mark.parametrize(("heads", "state_dtype"), HEAD_STATE_CASES)
@torch.no_grad()
def test_kda_mtp_commit_matches_reference(indirect, heads, state_dtype):
    """Conv windows and checkpoints at the state slots; histories and their
    lengths at replay slots apart from them, read through page-strided views."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_commit import (
        build_kda_mtp_history_descriptors,
        kda_mtp_commit,
    )

    torch.manual_seed(11)
    layers, slots, replay_slot_count, candidate_capacity, width = 2, 5, 6, 5, 3
    channels = 3 * heads * DIM
    capacity = WINDOW + width
    states, conv, state_desc, conv_desc = _make_state_views(
        indirect, layers, slots, heads, channels, state_dtype
    )
    reference_states = [state.clone() for state in states]
    reference_conv = [state.clone() for state in conv]
    candidate_x = _randn((layers, candidate_capacity, width, channels), 0.09).to(torch.bfloat16)
    history_k, history_u, history_G, history_len = _make_pool_history(
        layers, replay_slot_count, heads, capacity
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
    replay_slot_for_position = torch.tensor([5, 0, 2], device="cuda", dtype=torch.int32)
    replay_work_items = torch.stack(
        (
            positions,
            slot_for_position[positions.long()],
            starts[positions.long()],
            torch.zeros_like(positions),
            replay_slot_for_position[positions.long()],
        ),
        dim=1,
    ).contiguous()
    history_len.copy_(torch.tensor([10, 4, 16, 2, 7, 15], device="cuda", dtype=torch.int32))
    reference_len = history_len.clone()
    reference_k, reference_u, reference_G = _commit_reference(
        reference_states,
        reference_conv,
        candidate_x,
        [view.clone() for view in history_k],
        [view.clone() for view in history_u],
        [view.clone() for view in history_G],
        reference_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
    )

    kda_mtp_commit(
        states[0],
        conv[0],
        candidate_x,
        build_kda_mtp_history_descriptors(history_k, history_u, history_G),
        history_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
        heads=heads,
        history_window=WINDOW,
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
    torch.testing.assert_close(torch.stack(history_k), reference_k, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(history_u), reference_u, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(history_G), reference_G, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(history_len, reference_len, rtol=0, atol=0)
