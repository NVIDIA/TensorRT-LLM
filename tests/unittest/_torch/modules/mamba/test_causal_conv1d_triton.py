# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness tests for the Triton causal conv1d kernels' conv_state update.

``_causal_conv1d_fwd_kernel`` and ``_causal_conv1d_update_kernel`` first read
the incoming ``conv_state`` with a 1-D (``[BLOCK_N]``) layout to seed the
convolution window, then overwrite those very same cells with a 2-D
(``[NP2_STATELEN, BLOCK_N]``) layout. The thread that writes a cell is not the
thread that read it, so the two phases need an explicit ``tl.debug_barrier()``
between them; otherwise correctness depends on whether the Triton compiler
happens to emit a ``bar.sync`` for the chosen launch configuration.

The tests therefore sweep several ``(num_warps, BLOCK_N, num_stages)``
combinations - some of which get no compiler-inserted barrier - and always feed
a non-zero initial state, since that is the only case where the stale-read is
observable. ``num_stages`` is swept because the prefill kernel's token loop is a
dynamic ``range``, so the pipeliner rewrites the code around the barrier; the
decode kernel only uses ``tl.static_range`` and is unaffected by it.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton causal conv1d kernels require CUDA"
)

# (num_warps, BLOCK_N, num_stages): the last two `(num_warps, BLOCK_N)` pairs get
# no compiler-inserted `bar.sync`, so they only produce correct results if the
# kernels barrier explicitly. The stage counts cover both the unpipelined case
# and one either side of the kernels' defaults.
LAUNCH_CONFIGS = [(4, 128, 3), (8, 32, 1), (16, 64, 2)]


def _conv1d_ref(x, conv_state, weight, bias, activation):
    """Reference for one convolution step over a dense batch.

    x: [batch, dim, seqlen], conv_state: [batch, dim, state_len]
    weight: [dim, width], bias: [dim] or None

    Returns (out [batch, dim, seqlen], new_conv_state [batch, dim, state_len]).
    """
    dim, width = weight.shape
    state_len = conv_state.shape[-1]
    padded = torch.cat([conv_state.float(), x.float()], dim=-1)
    # Keep only the `width - 1` history entries the convolution actually needs.
    windowed = padded[..., state_len - (width - 1) :]
    out = F.conv1d(
        windowed,
        weight.float().unsqueeze(1),
        bias.float() if bias is not None else None,
        groups=dim,
    )
    if activation:
        out = F.silu(out)
    return out, padded[..., -state_len:]


def _make_inputs(dim, width, dtype, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

    weight = rand(dim, width)
    bias = rand(dim)
    return weight, bias


def _tolerance(dtype):
    return {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}


@skip_unsupported
@pytest.mark.parametrize("num_warps, block_n, num_stages", LAUNCH_CONFIGS)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_preserves_initial_state(num_warps, block_n, num_stages, width, dtype):
    """The decode kernel must convolve against the *old* conv_state."""
    device = "cuda"
    batch, dim, seqlen = 4, 64, 1
    state_len = width - 1
    num_cache_lines = batch + 2

    weight, bias = _make_inputs(dim, width, dtype, device, seed=1234)
    generator = torch.Generator(device=device).manual_seed(4321)
    x = torch.randn(batch, seqlen, dim, dtype=dtype, device=device, generator=generator).transpose(
        1, 2
    )
    conv_state = torch.randn(
        num_cache_lines, dim, state_len, dtype=dtype, device=device, generator=generator
    )
    # Interleave the slots so the kernel exercises non-identity cache indices.
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)

    ref_out, ref_state = _conv1d_ref(
        x, conv_state[conv_state_indices.long()], weight, bias, activation="silu"
    )

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        _block_n=block_n,
        _num_warps=num_warps,
        _num_stages=num_stages,
    )

    tol = _tolerance(dtype)
    torch.testing.assert_close(out.float(), ref_out, **tol)
    torch.testing.assert_close(conv_state[conv_state_indices.long()].float(), ref_state, **tol)


@skip_unsupported
@pytest.mark.parametrize("num_warps, block_n, num_stages", LAUNCH_CONFIGS)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fwd_preserves_initial_state(num_warps, block_n, num_stages, width, dtype):
    """The varlen prefill kernel must convolve against the *old* conv_state.

    ``seq_lens`` mixes sequences longer and shorter than ``state_len`` so both
    conv_state update branches of the kernel are covered.
    """
    device = "cuda"
    dim = 64
    state_len = width - 1
    seq_lens = [7, 3, 1]
    batch = len(seq_lens)
    num_cache_lines = batch + 2

    weight, bias = _make_inputs(dim, width, dtype, device, seed=1234)
    generator = torch.Generator(device=device).manual_seed(4321)
    total_tokens = sum(seq_lens)
    # channel-last: x is (dim, cu_seqlen) with stride(0) == 1
    x = torch.randn(total_tokens, dim, dtype=dtype, device=device, generator=generator).transpose(
        0, 1
    )
    conv_state = torch.randn(
        num_cache_lines, dim, state_len, dtype=dtype, device=device, generator=generator
    )
    initial_conv_state = conv_state.clone()

    query_start_loc = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32, device=device
    )
    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    # Mix cached and fresh sequences: only the cached ones read conv_state first.
    has_initial_state = torch.tensor([True, False, True], device=device)

    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_state,
        query_start_loc,
        seq_lens,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        _block_n=block_n,
        _num_warps=num_warps,
        _num_stages=num_stages,
    )

    tol = _tolerance(dtype)
    for i, seq_len in enumerate(seq_lens):
        slot = int(cache_indices[i])
        start = int(query_start_loc[i])
        x_i = x[:, start : start + seq_len].unsqueeze(0)
        state_i = initial_conv_state[slot].unsqueeze(0)
        if not has_initial_state[i]:
            state_i = torch.zeros_like(state_i)

        ref_out, ref_state = _conv1d_ref(x_i, state_i, weight, bias, activation="silu")

        torch.testing.assert_close(
            out[:, start : start + seq_len].unsqueeze(0).float(), ref_out, **tol
        )
        torch.testing.assert_close(conv_state[slot].unsqueeze(0).float(), ref_state, **tol)
