# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLA parity tests for the fused KDA single-token decode kernel."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401

CONV_WIDTH = 4
HEAD_DIM = 128
NUM_CACHE_SLOTS = 514
OUTPUT_NORM_EPS = 1e-5


@dataclass(frozen=True)
class KdaInputs:
    num_cache_slots: int
    x_q: torch.Tensor
    x_k: torch.Tensor
    x_v: torch.Tensor
    w_q_t: torch.Tensor
    w_k_t: torch.Tensor
    w_v_t: torch.Tensor
    bias_q: torch.Tensor
    bias_k: torch.Tensor
    bias_v: torch.Tensor
    conv_state_q: torch.Tensor
    conv_state_k: torch.Tensor
    conv_state_v: torch.Tensor
    conv_state_packed: torch.Tensor | None
    conv_state_storage: torch.Tensor | None
    conv_state_slot_stride: int | None
    a_log: torch.Tensor
    g: torch.Tensor
    dt_bias: torch.Tensor
    beta: torch.Tensor
    output_norm_gate: torch.Tensor
    output_norm_weight: torch.Tensor
    state_storage: torch.Tensor
    state_indices: torch.Tensor | None
    cu_seqlens: torch.Tensor


def _state_view(
    storage: torch.Tensor,
    num_cache_slots: int,
    num_heads: int,
    head_dim: int,
    slot_gap: int | None,
) -> torch.Tensor:
    dense_slot_stride = num_heads * head_dim * head_dim
    slot_gap = 0 if slot_gap is None else slot_gap
    assert (dense_slot_stride + slot_gap) % 4 == 0, (
        "KDA state slots must remain aligned for float4 state accesses"
    )
    return storage.as_strided(
        (num_cache_slots, num_heads, head_dim, head_dim),
        (dense_slot_stride + slot_gap, head_dim * head_dim, head_dim, 1),
    )


def _make_inputs(
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    use_state_indices: bool,
    update_conv_cache: bool,
    state_slot_gap: int | None,
    seed: int,
) -> KdaInputs:
    torch.manual_seed(seed)
    projection_size = num_heads * head_dim
    num_cache_slots = NUM_CACHE_SLOTS if use_state_indices else batch_size
    conv_slots = num_cache_slots if update_conv_cache else batch_size

    if update_conv_cache:
        dense_conv_slot_stride = 3 * projection_size * (CONV_WIDTH - 1)
        conv_state_slot_stride = dense_conv_slot_stride + 64
        conv_state_storage = torch.randn(
            (conv_slots * conv_state_slot_stride,),
            device="cuda",
            dtype=torch.bfloat16,
        )
        conv_state_packed = conv_state_storage.as_strided(
            (conv_slots, 3 * projection_size, CONV_WIDTH - 1),
            (conv_state_slot_stride, CONV_WIDTH - 1, 1),
        )
        conv_state_q = conv_state_packed[:, :projection_size]
        conv_state_k = conv_state_packed[:, projection_size : 2 * projection_size]
        conv_state_v = conv_state_packed[:, 2 * projection_size :]
    else:
        conv_state_packed = None
        conv_state_storage = None
        conv_state_slot_stride = None
        conv_state_q, conv_state_k, conv_state_v = (
            torch.randn(
                (conv_slots, projection_size, CONV_WIDTH - 1),
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )

    dense_slot_stride = num_heads * head_dim * head_dim
    slot_gap = 0 if state_slot_gap is None else state_slot_gap
    state_storage = torch.randn(
        (num_cache_slots * (dense_slot_stride + slot_gap),),
        device="cuda",
        dtype=torch.float32,
    )
    state_indices = None
    if use_state_indices:
        state_indices = torch.randperm(
            num_cache_slots,
            device="cuda",
            dtype=torch.int32,
        )[:batch_size]

    return KdaInputs(
        num_cache_slots=num_cache_slots,
        x_q=torch.randn(
            (1, batch_size, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        x_k=torch.randn(
            (1, batch_size, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        x_v=torch.randn(
            (1, batch_size, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        w_q_t=torch.randn(
            (CONV_WIDTH, projection_size),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        w_k_t=torch.randn(
            (CONV_WIDTH, projection_size),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        w_v_t=torch.randn(
            (CONV_WIDTH, projection_size),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        bias_q=torch.randn(
            (projection_size,),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        bias_k=torch.randn(
            (projection_size,),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        bias_v=torch.randn(
            (projection_size,),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        conv_state_q=conv_state_q,
        conv_state_k=conv_state_k,
        conv_state_v=conv_state_v,
        conv_state_packed=conv_state_packed,
        conv_state_storage=conv_state_storage,
        conv_state_slot_stride=conv_state_slot_stride,
        a_log=torch.empty(num_heads, device="cuda", dtype=torch.float32).uniform_(1.0, 16.0).log_(),
        g=torch.randn(
            (1, batch_size, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        dt_bias=torch.empty(projection_size, device="cuda", dtype=torch.float32).uniform_(
            -4.0, -2.0
        ),
        beta=torch.randn(
            (1, batch_size, num_heads),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        output_norm_gate=torch.randn(
            (1, batch_size, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        output_norm_weight=torch.empty(head_dim, device="cuda", dtype=torch.float32).uniform_(
            0.5, 1.5
        ),
        state_storage=state_storage,
        state_indices=state_indices,
        cu_seqlens=torch.arange(batch_size + 1, device="cuda", dtype=torch.int32),
    )


def _conv_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[1]
    x_flat = x.transpose(0, 1).reshape(batch_size, num_heads * head_dim)
    window = torch.cat((conv_state.float(), x_flat.float().unsqueeze(-1)), dim=-1)
    output = bias.float() + (window * weight_t.transpose(0, 1).float()).sum(dim=-1)
    output = F.silu(output).reshape(batch_size, 1, num_heads, head_dim).to(torch.bfloat16)
    updated_state = torch.cat((conv_state[:, :, 1:], x_flat.unsqueeze(-1)), dim=-1)
    return output, updated_state


def _fla_reference(
    inputs: KdaInputs,
    initial_state: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    apply_output_norm: bool,
    update_conv_cache: bool,
    apply_beta_sigmoid: bool,
    gate_lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    from fla.ops.kda import fused_recurrent_kda

    if update_conv_cache:
        assert inputs.state_indices is not None
        state_indices = inputs.state_indices.long()
        conv_q = inputs.conv_state_q.index_select(0, state_indices)
        conv_k = inputs.conv_state_k.index_select(0, state_indices)
        conv_v = inputs.conv_state_v.index_select(0, state_indices)
    else:
        conv_q = inputs.conv_state_q
        conv_k = inputs.conv_state_k
        conv_v = inputs.conv_state_v

    q, updated_conv_q = _conv_reference(
        inputs.x_q,
        conv_q,
        inputs.w_q_t,
        inputs.bias_q,
        num_heads,
        head_dim,
    )
    k, updated_conv_k = _conv_reference(
        inputs.x_k,
        conv_k,
        inputs.w_k_t,
        inputs.bias_k,
        num_heads,
        head_dim,
    )
    v, updated_conv_v = _conv_reference(
        inputs.x_v,
        conv_v,
        inputs.w_v_t,
        inputs.bias_v,
        num_heads,
        head_dim,
    )

    output, final_state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=inputs.g.transpose(0, 1),
        beta=inputs.beta.transpose(0, 1).float(),
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=apply_beta_sigmoid,
        lower_bound=gate_lower_bound,
        state_v_first=True,
    )

    if apply_output_norm:
        output_float = output.float()
        rstd = torch.rsqrt(output_float.square().mean(dim=-1, keepdim=True) + OUTPUT_NORM_EPS)
        output = (
            output_float
            * rstd
            * inputs.output_norm_weight.view(1, 1, 1, head_dim)
            * torch.sigmoid(inputs.output_norm_gate.transpose(0, 1).float())
        )
    return (
        output.to(torch.bfloat16),
        final_state.float(),
        (updated_conv_q, updated_conv_k, updated_conv_v),
    )


def _assert_parity(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    min_cosine: float = 0.9999,
    max_relative_l2: float = 1e-2,
) -> None:
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = F.cosine_similarity(actual_float, expected_float, dim=0).item()
    relative_l2 = (
        (actual_float - expected_float).norm() / expected_float.norm().clamp_min(1e-12)
    ).item()
    max_abs = (actual_float - expected_float).abs().max().item()
    assert cosine > min_cosine, f"{name}: cosine={cosine:.8f}, max_abs={max_abs:.6g}"
    assert relative_l2 < max_relative_l2, (
        f"{name}: relative_l2={relative_l2:.8f}, max_abs={max_abs:.6g}"
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 17, 32])
@pytest.mark.parametrize("num_heads", [2, 3, 4, 6, 12, 96])
@pytest.mark.parametrize(
    (
        "use_state_indices,"
        "update_conv_cache,state_slot_gap,apply_output_norm,apply_beta_sigmoid,gate_lower_bound"
    ),
    [
        pytest.param(True, False, None, True, True, -5.0, id="indexed"),
        pytest.param(False, False, None, False, False, None, id="batch-local-softplus-decay"),
        pytest.param(True, True, 73728, True, True, -5.0, id="indexed-conv-strided"),
    ],
)
def test_kda_decode_matches_fla(
    batch_size: int,
    num_heads: int,
    use_state_indices: bool,
    update_conv_cache: bool,
    state_slot_gap: int | None,
    apply_output_norm: bool,
    apply_beta_sigmoid: bool,
    gate_lower_bound: float | None,
) -> None:
    head_dim = HEAD_DIM
    inputs = _make_inputs(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        use_state_indices=use_state_indices,
        update_conv_cache=update_conv_cache,
        state_slot_gap=state_slot_gap,
        seed=2026 + batch_size + num_heads,
    )
    actual_state = _state_view(
        inputs.state_storage,
        inputs.num_cache_slots,
        num_heads,
        head_dim,
        state_slot_gap,
    )
    state_gap = None
    state_gap_before = None
    if state_slot_gap is not None:
        dense_slot_stride = num_heads * head_dim * head_dim
        state_gap = inputs.state_storage.view(
            inputs.num_cache_slots, dense_slot_stride + state_slot_gap
        )[:, dense_slot_stride:]
        state_gap_before = state_gap.clone()

    if inputs.state_indices is None:
        initial_selected_state = actual_state.clone()
        state_before = None
    else:
        initial_selected_state = actual_state.index_select(0, inputs.state_indices.long()).clone()
        state_before = actual_state.clone()

    if update_conv_cache:
        assert inputs.conv_state_packed is not None
        assert inputs.conv_state_storage is not None
        assert inputs.conv_state_slot_stride is not None
        projection_size = num_heads * head_dim
        actual_conv_storage = inputs.conv_state_storage.clone()
        actual_conv_packed = actual_conv_storage.as_strided(
            inputs.conv_state_packed.shape,
            inputs.conv_state_packed.stride(),
        )
        actual_conv_q = actual_conv_packed[:, :projection_size]
        actual_conv_k = actual_conv_packed[:, projection_size : 2 * projection_size]
        actual_conv_v = actual_conv_packed[:, 2 * projection_size :]
        dense_conv_slot_stride = 3 * projection_size * (CONV_WIDTH - 1)
        actual_conv_gap = actual_conv_storage.view(
            inputs.num_cache_slots, inputs.conv_state_slot_stride
        )[:, dense_conv_slot_stride:]
        conv_gap_before = actual_conv_gap.clone()
    else:
        actual_conv_q = inputs.conv_state_q.clone()
        actual_conv_k = inputs.conv_state_k.clone()
        actual_conv_v = inputs.conv_state_v.clone()
        actual_conv_gap = conv_gap_before = None
    conv_before = (actual_conv_q.clone(), actual_conv_k.clone(), actual_conv_v.clone())
    if update_conv_cache:
        projection_size = num_heads * head_dim
        expected_conv_stride = (inputs.conv_state_slot_stride, CONV_WIDTH - 1, 1)
        for conv_state in (actual_conv_q, actual_conv_k, actual_conv_v):
            assert conv_state.stride() == expected_conv_stride

    expected_output, expected_state, expected_conv = _fla_reference(
        inputs,
        initial_selected_state,
        num_heads=num_heads,
        head_dim=head_dim,
        apply_output_norm=apply_output_norm,
        update_conv_cache=update_conv_cache,
        apply_beta_sigmoid=apply_beta_sigmoid,
        gate_lower_bound=gate_lower_bound,
    )

    # kda_decode is inplace-only: the caller supplies the output buffer and the
    # kernel writes into it (the op returns ``()``).
    actual_output = torch.empty(
        (batch_size, 1, num_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.ops.trtllm.kda_decode(
        inputs.x_q,
        inputs.x_k,
        inputs.x_v,
        inputs.w_q_t,
        inputs.w_k_t,
        inputs.w_v_t,
        inputs.bias_q,
        inputs.bias_k,
        inputs.bias_v,
        actual_conv_q,
        actual_conv_k,
        actual_conv_v,
        inputs.a_log,
        inputs.g,
        inputs.dt_bias,
        inputs.beta,
        inputs.output_norm_gate,
        inputs.output_norm_weight,
        inputs.state_indices,
        inputs.cu_seqlens,
        actual_state,
        apply_output_norm,
        update_conv_cache,
        gate_lower_bound is not None,
        apply_beta_sigmoid,
        0.0 if gate_lower_bound is None else gate_lower_bound,
        head_dim**-0.5,
        OUTPUT_NORM_EPS,
        actual_output,
    )

    _assert_parity("output", actual_output, expected_output)
    actual_selected_state = (
        actual_state
        if inputs.state_indices is None
        else actual_state.index_select(0, inputs.state_indices.long())
    )
    _assert_parity(
        "recurrent state",
        actual_selected_state,
        expected_state,
    )
    if state_before is not None:
        state_before.index_copy_(0, inputs.state_indices.long(), actual_selected_state)
        torch.testing.assert_close(actual_state, state_before, rtol=0, atol=0)
    if state_gap is not None:
        torch.testing.assert_close(state_gap, state_gap_before, rtol=0, atol=0)

    if update_conv_cache:
        assert inputs.state_indices is not None
        state_indices = inputs.state_indices.long()
        for actual, expected, before in zip(
            (actual_conv_q, actual_conv_k, actual_conv_v),
            expected_conv,
            conv_before,
            strict=True,
        ):
            expected_pool = before.clone(memory_format=torch.preserve_format)
            expected_pool.index_copy_(0, state_indices, expected)
            torch.testing.assert_close(actual, expected_pool, rtol=0, atol=0)
        torch.testing.assert_close(actual_conv_gap, conv_gap_before, rtol=0, atol=0)
    else:
        for actual, before in zip(
            (actual_conv_q, actual_conv_k, actual_conv_v),
            conv_before,
            strict=True,
        ):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
