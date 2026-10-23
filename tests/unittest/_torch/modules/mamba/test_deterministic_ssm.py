# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed Mamba2 recurrence: phase invariance and independent reference numerics."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.deterministic_ssm import deterministic_ssm
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    cu_seqlens_to_chunk_indices_offsets_triton,
)
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update


@pytest.mark.parametrize("dim,n", [(64, 64), (80, 128), (128, 128)])
@pytest.mark.parametrize("softplus", [False, True])
def test_deterministic_ssm_reference_and_split(dim, n, softplus):
    torch.manual_seed(23)
    heads, groups, length = 4, 2, 17
    xbc = (
        torch.randn(length, heads * dim + 2 * groups * n, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    dt = torch.randn(length, heads + 3, device="cuda", dtype=torch.bfloat16)[:, :heads] * 0.1
    a = -torch.rand(heads, device="cuda")
    d = torch.randn(heads, device="cuda")
    bias = torch.randn(heads, device="cuda") * 0.1
    initial = torch.randn(5, heads, dim, n, device="cuda") * 0.1
    indices = torch.tensor([3], device="cuda", dtype=torch.int32)
    has = torch.tensor([True], device="cuda")
    state = initial.clone()
    output = torch.empty(length, heads * dim, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    deterministic_ssm(xbc, dt, a, d, bias, state, cu, indices, has, output, groups, softplus)
    expected_state = initial[3].double()
    reference = []
    for t in range(length):
        x = xbc[t, : heads * dim].double().reshape(heads, dim)
        b, c = (
            xbc[t, heads * dim :]
            .double()
            .reshape(2, groups, n)
            .repeat_interleave(heads // groups, dim=1)
        )
        delta = dt[t].double() + bias.double()
        if softplus:
            delta = torch.nn.functional.softplus(delta)
        expected_state = (
            expected_state * torch.exp(delta * a.double())[:, None, None]
            + (b * delta[:, None])[:, None, :] * x[:, :, None]
        )
        reference.append((expected_state * c[:, None, :]).sum(-1) + x * d.double()[:, None])
    torch.testing.assert_close(
        output.float(), torch.stack(reference).flatten(1).float(), rtol=0.008, atol=5e-5
    )
    torch.testing.assert_close(state[3].double(), expected_state, rtol=1e-4, atol=2e-7)
    split_state = initial.clone()
    split_output = torch.empty_like(output)
    for t in range(length):
        deterministic_ssm(
            xbc[t : t + 1],
            dt[t : t + 1],
            a,
            d,
            bias,
            split_state,
            cu.new_tensor([0, 1]),
            indices,
            has,
            split_output[t : t + 1],
            groups,
            softplus,
        )
    torch.testing.assert_close(split_output, output, rtol=0, atol=0)
    torch.testing.assert_close(split_state, state, rtol=0, atol=0)
    torch.testing.assert_close(state[[0, 1, 2, 4]], initial[[0, 1, 2, 4]], rtol=0, atol=0)


def _make_mamba_core(dim, n):
    layer = Mamba2Mixer.__new__(Mamba2Mixer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.tp_nheads, layer.tp_ngroups = 4, 2
    layer.head_dim, layer.d_state, layer.chunk_size = dim, n, 128
    layer.tp_d_inner = 4 * dim
    layer.tp_conv_dim = 4 * dim + 4 * n
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(layer.tp_conv_dim, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
        bias=torch.randn(layer.tp_conv_dim, device="cuda", dtype=torch.bfloat16) * 0.1,
    )
    layer.A = -torch.rand(4, device="cuda")
    layer.D = torch.randn(4, device="cuda")
    layer.dt_bias = torch.randn(4, device="cuda") * 0.1
    layer.delta_softplus = True
    layer._mamba_ssm_cache_dtype = torch.float32
    layer._token_major_conv = True
    layer._use_flashinfer = False
    layer._stochastic_rounding_for_flashinfer = False
    layer.selective_state_update_func = selective_state_update
    layer.norm = SimpleNamespace(is_nvfp4=False)
    layer.cache_derived_state()
    layer.aux_steram = torch.cuda.Stream()
    layer.events = [torch.cuda.Event(), torch.cuda.Event()]
    return layer


def _run_mamba_core(layer, projected, conv, state, lengths, prefills, slots, initial, output=None):
    indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    metadata = SimpleNamespace(
        state_indices=indices,
        cu_seqlens=cu,
        _arange_buffer=torch.arange(len(slots) + 1, device="cuda", dtype=torch.int32),
        has_initial_states=torch.tensor(initial, device="cuda", dtype=torch.bool),
        use_initial_states=any(initial[:prefills]),
    )
    if prefills:
        metadata.seq_idx = torch.repeat_interleave(
            torch.arange(prefills, device="cuda", dtype=torch.int32),
            torch.tensor(lengths[:prefills], device="cuda"),
            output_size=sum(lengths[:prefills]),
        ).unsqueeze(0)
        metadata.chunk_indices, metadata.chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
            cu[: prefills + 1], 128, total_seqlens=sum(lengths[:prefills])
        )
    cache = SimpleNamespace(
        mamba_layer_cache=lambda _: SimpleNamespace(conv=conv, temporal=state),
        is_speculative=lambda: False,
    )
    attn = SimpleNamespace(
        num_contexts=prefills,
        seq_lens=indices,
        num_ctx_tokens=sum(lengths[:prefills]),
        num_tokens=sum(lengths),
        kv_cache_manager=cache,
    )
    if output is None:
        output = torch.empty(
            projected.shape[0], layer.tp_d_inner, device="cuda", dtype=torch.bfloat16
        )
    layer.forward_core(projected, attn, metadata, None, output)
    return output


@pytest.mark.parametrize("length", [7, 127, 129])
@pytest.mark.parametrize("initial", [False, True])
def test_mamba2_deterministic_prefill_matches_decode(monkeypatch, length, initial):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(31)
    layer = _make_mamba_core(64, 128)
    projected = (
        torch.randn(
            length, layer.tp_d_inner + layer.tp_conv_dim + 4, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    )
    conv = torch.randn(5, layer.tp_conv_dim, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(5, 4, 64, 128, device="cuda") * 0.1
    full_conv, full_state = conv.clone(), state.clone()
    full = _run_mamba_core(layer, projected, full_conv, full_state, [length], 1, [3], [initial])
    prefix = length // 2
    outputs = [_run_mamba_core(layer, projected[:prefix], conv, state, [prefix], 1, [3], [initial])]
    for t in range(prefix, length):
        outputs.append(
            _run_mamba_core(layer, projected[t : t + 1], conv, state, [1], 0, [3], [True])
        )
    torch.testing.assert_close(torch.cat(outputs), full, atol=0, rtol=0)
    torch.testing.assert_close(conv, full_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, full_state, atol=0, rtol=0)


def test_mamba2_deterministic_mixed_batch_and_padding(monkeypatch):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(32)
    layer = _make_mamba_core(80, 64)
    lengths, slots, initial = [7, 13, 1, 1], [4, 1, 3, 0], [False, True, True, True]
    projected = (
        torch.randn(
            sum(lengths) + 3,
            layer.tp_d_inner + layer.tp_conv_dim + 4,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    original = projected.clone()
    conv = torch.randn(6, layer.tp_conv_dim, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(6, 4, 80, 64, device="cuda") * 0.1
    mixed_conv, mixed_state = conv.clone(), state.clone()
    mixed = _run_mamba_core(layer, projected, mixed_conv, mixed_state, lengths, 2, slots, initial)
    start = 0
    outputs = []
    for i, length in enumerate(lengths):
        outputs.append(
            _run_mamba_core(
                layer,
                projected[start : start + length],
                conv,
                state,
                [length],
                int(i < 2),
                [slots[i]],
                [initial[i]],
            )
        )
        start += length
    torch.testing.assert_close(mixed[:start], torch.cat(outputs), rtol=0, atol=0)
    assert torch.count_nonzero(mixed[start:]) == 0
    torch.testing.assert_close(mixed_conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(mixed_state, state, rtol=0, atol=0)
    torch.testing.assert_close(projected, original, rtol=0, atol=0)


def test_deterministic_ssm_graph_updates_indexed_pool_and_masks_padding():
    torch.manual_seed(81)
    heads, groups, dim, n = 4, 2, 64, 64
    xbc = torch.randn(7, heads * dim + 2 * groups * n, device="cuda", dtype=torch.bfloat16) * 0.1
    dt = torch.randn(7, heads, device="cuda", dtype=torch.bfloat16) * 0.1
    a = -torch.rand(heads, device="cuda")
    d = torch.randn(heads, device="cuda")
    bias = torch.randn(heads, device="cuda")
    initial = torch.randn(5, heads, dim, n, device="cuda") * 0.1
    state = initial.clone()
    cu = torch.tensor([0, 3, 4, 7], device="cuda", dtype=torch.int32)
    indices = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
    has = torch.tensor([True, True, False], device="cuda")
    output = torch.empty(7, heads * dim, device="cuda", dtype=torch.bfloat16)

    def run(s, y):
        deterministic_ssm(xbc, dt, a, d, bias, s, cu, indices, has, y, groups, True)

    run(state, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(state, output)
    xbc.mul_(0.5)
    dt.add_(0.25)
    state.copy_(initial)
    indices[0] = 4
    expected_state = initial.clone()
    expected = torch.empty_like(output)
    run(expected_state, expected)
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(state[[1, 2, 3]], initial[[1, 2, 3]], atol=0, rtol=0)
    assert torch.count_nonzero(output[3]) == 0
