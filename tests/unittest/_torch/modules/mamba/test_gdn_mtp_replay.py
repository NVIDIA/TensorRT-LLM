# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

DIM = 128
WINDOW = 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f() or not IS_CUTLASS_DSL_AVAILABLE,
    reason="GDN MTP replay requires CUTLASS DSL on datacenter Blackwell",
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
    b,
    a,
    checkpoint,
    state_indices,
    is_dummy,
    history_u,
    history_k,
    history_G,
    history_len,
    A_log,
    dt_bias,
    replay_indices,
):
    batch, width, _ = raw_x.shape
    key_heads = history_k.shape[2]
    value_heads = history_u.shape[2]
    heads_per_key = value_heads // key_heads
    slots = state_indices.long()
    replay_slots = replay_indices.long()

    packed_qkv = _conv4_silu(raw_x, conv_weight, conv_state, slots)
    key_width = key_heads * DIM
    value_width = value_heads * DIM
    query = packed_qkv[..., :key_width].view(batch, width, key_heads, DIM)
    key = packed_qkv[..., key_width : 2 * key_width].view_as(query)
    value = packed_qkv[..., 2 * key_width : 2 * key_width + value_width].view(
        batch, width, value_heads, DIM
    )
    beta_input = b.reshape(batch, width, value_heads)
    decay_input = a.reshape(batch, width, value_heads)

    query_f32 = query.float()
    key_f32 = key.float()
    query_norm = (
        query_f32
        * (DIM**-0.5 / torch.sqrt(torch.sum(query_f32 * query_f32, dim=-1, keepdim=True) + 1e-6))
    ).to(torch.bfloat16)
    key_norm = (key_f32 / torch.sqrt(torch.sum(key_f32 * key_f32, dim=-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    )

    gate_input = decay_input.float() + dt_bias.float()[None, None, :]
    softplus = torch.where(
        gate_input <= 20.0,
        torch.log1p(torch.exp(gate_input)),
        gate_input,
    )
    gate = -torch.exp(A_log.float())[None, None, :] * softplus
    beta = torch.sigmoid(beta_input.float())

    lengths = history_len.index_select(0, replay_slots).long()
    selected_u = history_u[:, :WINDOW].index_select(0, replay_slots).float()
    selected_k = history_k[:, :WINDOW].index_select(0, replay_slots).float()
    selected_G = history_G[:, :, :WINDOW].index_select(0, replay_slots).float()
    key_for_value_head = torch.arange(value_heads, device=raw_x.device) // heads_per_key
    selected_k = selected_k[:, :, key_for_value_head, :]

    positions = torch.arange(WINDOW, device=raw_x.device)[None, None, :]
    active = positions < lengths[:, None, None]
    previous_G = torch.zeros(batch, value_heads, device=raw_x.device, dtype=torch.float32)
    nonempty = lengths > 0
    if bool(nonempty.any()):
        rows = torch.where(nonempty)[0]
        previous_G[rows] = selected_G[rows, :, lengths[rows] - 1]
    history_decay = torch.where(
        active,
        torch.exp(previous_G[:, :, None] - selected_G),
        0.0,
    )
    state = checkpoint.index_select(0, slots).float()
    state *= torch.exp(previous_G)[:, :, None, None]
    state += torch.einsum("btiv,btik,bit->bivk", selected_u, selected_k, history_decay)

    output = torch.empty(
        batch,
        width,
        value_heads,
        DIM,
        device=raw_x.device,
        dtype=torch.bfloat16,
    )
    current_u = torch.empty_like(output)
    current_G = torch.empty(batch, width, value_heads, device=raw_x.device, dtype=torch.float32)
    query_hv = query_norm[:, :, key_for_value_head].float()
    key_hv = key_norm[:, :, key_for_value_head].float()
    running_G = previous_G
    for token in range(width):
        running_G = running_G + gate[:, token]
        current_G[:, token] = running_G
        state *= torch.exp(gate[:, token])[:, :, None, None]
        state_key = torch.einsum("bhvk,bhk->bhv", state, key_hv[:, token])
        update = beta[:, token, :, None] * (value[:, token].float() - state_key)
        state += update[:, :, :, None] * key_hv[:, token, :, None, :]
        output[:, token] = torch.einsum("bhvk,bhk->bhv", state, query_hv[:, token]).to(
            torch.bfloat16
        )
        current_u[:, token] = update.to(torch.bfloat16)

    writable_rows = torch.where(~is_dummy)[0]
    writable_slots = replay_slots[writable_rows]
    writable_lengths = lengths[writable_rows]
    for token in range(width):
        write_positions = writable_lengths + token
        history_u[writable_slots, write_positions] = current_u[writable_rows, token]
        history_k[writable_slots, write_positions] = key_norm[writable_rows, token]
        history_G[writable_slots, :, write_positions] = current_G[writable_rows, token]
    return output, raw_x.contiguous()


def _gate_columns(qkvz_width: int, value_heads: int, padded: bool) -> tuple[int, int, int]:
    """Columns of b and a and the row width of the input projection.

    ``padded`` mirrors ``Qwen3NextGatedDeltaNet._refresh_fused_in_proj``: the
    ``[qkvz | b | pad | a | pad]`` layout starts each gate block at a 32-byte
    (16 BF16 column) boundary; otherwise ``b`` and ``a`` are adjacent.
    """
    align = 16 if padded else 1

    def _up(columns: int) -> int:
        return -(-columns // align) * align

    b_col = _up(qkvz_width)
    a_col = b_col + _up(value_heads)
    return b_col, a_col, a_col + _up(value_heads)


@pytest.mark.parametrize("layout", ["adjacent", "padded"])
@pytest.mark.parametrize("width", [1, 3, 4, 8])
@pytest.mark.parametrize("head_ratio", [2, 4, 8])
def test_gdn_mtp_replay_matches_reference(width: int, head_ratio: int, layout: str) -> None:
    """Live rows with empty, partial and full (16-row) histories, widths that need
    MMA padding, a dummy row, and replay slots apart from the state slots. The
    q|k|v, b and a inputs are column views of one projection, either with
    adjacent gate blocks or with the padded fused in_proj layout."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_gdn_mtp_replay import gdn_mtp_replay

    torch.manual_seed(7)
    batch, slots, replay_slot_count = 4, 5, 6
    key_heads = 1
    value_heads = key_heads * head_ratio
    channels = (2 * key_heads + value_heads) * DIM
    qkvz_width = channels + value_heads * DIM

    b_col, a_col, row_width = _gate_columns(qkvz_width, value_heads, layout == "padded")
    # Padding columns hold NaN so that any read of them corrupts the result.
    projected = torch.full(
        (batch * width, row_width), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    projected[:, :qkvz_width] = _randn((batch * width, qkvz_width), 0.09)
    projected[:, b_col : b_col + value_heads] = _randn((batch * width, value_heads), 0.5, -0.1)
    projected[:, a_col : a_col + value_heads] = _randn((batch * width, value_heads), 0.5, -0.1)
    b = projected[:, b_col : b_col + value_heads]
    a = projected[:, a_col : a_col + value_heads]
    raw_x_2d = projected[:, :channels]
    raw_x = raw_x_2d.as_strided(
        (batch, width, channels),
        (width * raw_x_2d.stride(0), raw_x_2d.stride(0), 1),
    )
    conv_weight = _randn((channels, 4), 0.18).to(torch.bfloat16)
    conv_state = _randn((slots, channels, 3), 0.09).to(torch.bfloat16)
    checkpoint = _randn((slots, value_heads, DIM, DIM), 0.012).to(torch.bfloat16)
    capacity = WINDOW + width
    history_u = _randn((replay_slot_count, capacity, value_heads, DIM), 0.012).to(torch.bfloat16)
    raw_history_k = _randn((replay_slot_count, capacity, key_heads, DIM), 0.09)
    history_k = (
        raw_history_k / torch.sqrt(raw_history_k.square().sum(dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    history_G = torch.cumsum(
        -(_randn((replay_slot_count, value_heads, capacity), 0.025).abs() + 0.002),
        dim=-1,
    )
    state_indices = torch.tensor([0, 2, 3, 1], device="cuda", dtype=torch.int32)
    replay_indices = torch.tensor([5, 0, 2, 3], device="cuda", dtype=torch.int32)
    # Live rows read histories of 0, 16 and 7 rows; the dummy row's slot holds 11.
    history_len = torch.tensor([16, 4, 7, 11, 1, 0], device="cuda", dtype=torch.int32)
    is_dummy = torch.tensor([False, False, False, True], device="cuda")
    A_log = _randn((value_heads,), 0.35, -0.25).clamp_(-1.0, 0.5)
    dt_bias = _randn((value_heads,), 0.65, -2.5).clamp_(-5.0, 0.5)

    reference_u = history_u.clone()
    reference_k = history_k.clone()
    reference_G = history_G.clone()
    reference_output, reference_candidates = _replay_reference(
        raw_x,
        conv_weight,
        conv_state,
        b,
        a,
        checkpoint,
        state_indices,
        is_dummy,
        reference_u,
        reference_k,
        reference_G,
        history_len,
        A_log,
        dt_bias,
        replay_indices,
    )
    output = torch.empty_like(reference_output)
    candidate_x = torch.full(
        (batch, width, channels), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    checkpoint_before = checkpoint.clone()
    conv_before = conv_state.clone()
    history_len_before = history_len.clone()

    gdn_mtp_replay(
        raw_x,
        conv_weight,
        conv_state,
        b,
        a,
        checkpoint,
        state_indices,
        is_dummy,
        history_u,
        history_k,
        history_G,
        history_len,
        A_log,
        dt_bias,
        output,
        candidate_x,
        replay_indices=replay_indices,
    )

    live = ~is_dummy
    torch.testing.assert_close(output[live], reference_output[live], rtol=2e-2, atol=5e-3)
    torch.testing.assert_close(candidate_x[live], reference_candidates[live], rtol=0, atol=0)
    torch.testing.assert_close(history_u, reference_u, rtol=2e-2, atol=5e-3)
    torch.testing.assert_close(history_k, reference_k, rtol=1e-2, atol=1.5e-3)
    torch.testing.assert_close(history_G, reference_G, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(checkpoint, checkpoint_before, rtol=0, atol=0)
    torch.testing.assert_close(conv_state, conv_before, rtol=0, atol=0)
    torch.testing.assert_close(history_len, history_len_before, rtol=0, atol=0)


@torch.no_grad()
def _commit_reference(
    state_views,
    conv_views,
    candidate_x,
    history_u,
    history_k,
    history_G,
    history_len,
    replay_work_items,
    accepted_tokens,
    is_dummy,
):
    value_heads = history_u.shape[-2]
    key_heads = history_k.shape[-2]
    key_for_value_head = torch.arange(value_heads, device=history_u.device) // (
        value_heads // key_heads
    )
    for work_item in replay_work_items:
        position = int(work_item[0].item())
        slot = int(work_item[1].item())
        start = int(work_item[2].item())
        replay_slot = int(work_item[4].item())
        if bool(is_dummy[position].item()):
            continue

        accepted = int(accepted_tokens[position].item())
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
        first_u = history_u[:, replay_slot, :WINDOW].float()
        first_k = history_k[:, replay_slot, :WINDOW].float()[:, :, key_for_value_head]
        update = torch.einsum(
            "ltiv,ltik,lit->livk",
            first_u,
            first_k,
            torch.exp(end_G[:, :, None] - first_G),
        )
        checkpoint = torch.stack([view[slot] for view in state_views]).float()
        committed = checkpoint * torch.exp(end_G)[:, :, None, None] + update
        for layer, view in enumerate(state_views):
            view[slot].copy_(committed[layer].to(torch.bfloat16))

        tail_u = history_u[:, replay_slot, WINDOW : WINDOW + tail].clone()
        tail_k = history_k[:, replay_slot, WINDOW : WINDOW + tail].clone()
        tail_G = history_G[:, replay_slot, :, WINDOW : WINDOW + tail].clone()
        history_u[:, replay_slot, :tail] = tail_u
        history_k[:, replay_slot, :tail] = tail_k
        history_G[:, replay_slot, :, :tail] = tail_G - end_G[:, :, None]
        history_len[replay_slot] = tail


def _make_state_storage(indirect, layers, slots, value_heads, channels):
    if indirect:
        states = [
            _randn((slots, value_heads, DIM, DIM), 0.012).to(torch.bfloat16) for _ in range(layers)
        ]
        conv = [_randn((slots, channels, 3), 0.09).to(torch.bfloat16) for _ in range(layers)]
        state_descriptors = torch.tensor(
            [(state.data_ptr(), state.stride(0)) for state in states],
            device="cuda",
            dtype=torch.int64,
        )
        conv_descriptors = torch.tensor(
            [(state.data_ptr(), state.stride(0)) for state in conv],
            device="cuda",
            dtype=torch.int64,
        )
        return states, conv, states[0], conv[0], state_descriptors, conv_descriptors

    state_storage = _randn((slots, layers, value_heads, DIM, DIM), 0.012).to(torch.bfloat16)
    conv_storage = _randn((slots, layers, channels, 3), 0.09).to(torch.bfloat16)
    states = [state_storage[:, layer] for layer in range(layers)]
    conv = [conv_storage[:, layer] for layer in range(layers)]
    return states, conv, state_storage, conv_storage, None, None


def _make_pool_history(layers, replay_slots, capacity, key_heads, value_heads):
    """Per-layer U/K/G histories laid out like KV-cache pool pages: one page per
    replay slot holds every layer's history, so replay slots are widely strided."""
    u_elements = capacity * value_heads * DIM
    k_elements = capacity * key_heads * DIM
    g_elements = 2 * value_heads * capacity  # FP32 in BF16 elements
    page = layers * (u_elements + k_elements + g_elements) + 64
    pool = torch.zeros(replay_slots, page, device="cuda", dtype=torch.bfloat16)
    history_u, history_k, history_G = [], [], []
    offset = 0
    for _ in range(layers):
        u = pool[:, offset : offset + u_elements].view(replay_slots, capacity, value_heads, DIM)
        offset += u_elements
        k = pool[:, offset : offset + k_elements].view(replay_slots, capacity, key_heads, DIM)
        offset += k_elements
        g = pool[:, offset : offset + g_elements].view(torch.float32)
        g = g.view(replay_slots, value_heads, capacity)
        offset += g_elements
        u.copy_(_randn(u.shape, 0.012))
        raw_k = _randn(k.shape, 0.09)
        k.copy_(raw_k / torch.sqrt(raw_k.square().sum(dim=-1, keepdim=True) + 1e-6))
        g.copy_(torch.cumsum(-(_randn(g.shape, 0.025).abs() + 0.002), dim=-1))
        history_u.append(u)
        history_k.append(k)
        history_G.append(g)
    return history_u, history_k, history_G


@pytest.mark.parametrize("indirect", [False, True], ids=["affine", "indirect"])
@pytest.mark.parametrize("head_ratio", [1, 4])
@pytest.mark.parametrize(
    "accepted_counts,starting_lengths",
    [((3, 2, 1), (15, 10, 16)), ((1, 1, 1), (16, 15, 0)), ((4, 4, 2), (16, 0, 9))],
    ids=["mtp3", "golden-only", "full-window"],
)
def test_gdn_mtp_commit_matches_reference(
    indirect: bool,
    head_ratio: int,
    accepted_counts: tuple[int, int, int],
    starting_lengths: tuple[int, int, int],
) -> None:
    """Conv windows and checkpoints at the state slots, histories and their
    lengths at replay slots apart from them, read through page-strided views."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_gdn_mtp_commit import (
        build_gdn_mtp_history_descriptors,
        gdn_mtp_commit,
    )

    torch.manual_seed(11)
    layers, slots, replay_slot_count, candidate_capacity, width = 2, 5, 6, 5, 4
    key_heads = 1
    value_heads = key_heads * head_ratio
    channels = (2 * key_heads + value_heads) * DIM
    capacity = WINDOW + width

    states, conv, state_arg, conv_arg, state_desc, conv_desc = _make_state_storage(
        indirect, layers, slots, value_heads, channels
    )
    reference_states = [state.clone() for state in states]
    reference_conv = [state.clone() for state in conv]
    candidate_x = _randn((layers, candidate_capacity, width, channels), 0.09).to(torch.bfloat16)
    history_u, history_k, history_G = _make_pool_history(
        layers, replay_slot_count, capacity, key_heads, value_heads
    )
    history_descriptors = build_gdn_mtp_history_descriptors(history_u, history_k, history_G)
    accepted_tokens = torch.tensor(accepted_counts, device="cuda", dtype=torch.int32)
    is_dummy = torch.tensor([False, False, True], device="cuda")
    starts = torch.tensor(starting_lengths, device="cuda", dtype=torch.int32)
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
    # The length is one column of a per-slot metadata row, as in the pool.
    replay_meta = torch.zeros(replay_slot_count, 4, device="cuda", dtype=torch.int32)
    history_len = replay_meta[:, 0]
    history_len.copy_(
        torch.tensor(
            [starting_lengths[1], 4, starting_lengths[2], 2, 3, starting_lengths[0]],
            device="cuda",
            dtype=torch.int32,
        )
    )

    reference_u = torch.stack([view.clone() for view in history_u])
    reference_k = torch.stack([view.clone() for view in history_k])
    reference_G = torch.stack([view.clone() for view in history_G])
    reference_len = history_len.clone()
    _commit_reference(
        reference_states,
        reference_conv,
        candidate_x,
        reference_u,
        reference_k,
        reference_G,
        reference_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
    )

    gdn_mtp_commit(
        state_arg,
        conv_arg,
        candidate_x,
        history_descriptors,
        history_len,
        replay_work_items,
        accepted_tokens,
        is_dummy,
        key_heads=key_heads,
        value_heads=value_heads,
        history_window=WINDOW,
        ssm_state_descriptors=state_desc,
        conv_state_descriptors=conv_desc,
    )

    for actual, reference in zip(states, reference_states):
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=8e-3)
    for actual, reference in zip(conv, reference_conv):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(history_u), reference_u, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(history_k), reference_k, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(history_G), reference_G, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(history_len, reference_len, rtol=0, atol=0)
    assert not replay_meta[:, 1:].any()
