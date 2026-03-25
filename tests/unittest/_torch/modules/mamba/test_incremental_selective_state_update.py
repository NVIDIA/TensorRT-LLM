# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from einops import rearrange, repeat

from utils.torch_ref import selective_state_update_ref  # noqa: F401 (pure-Python reference)

from tensorrt_llm._torch.modules.mamba.incremental_selective_state_update import \
    incremental_selective_state_update
from tensorrt_llm._torch.modules.mamba.selective_state_update import \
    selective_state_update

# ---------------------------------------------------------------------------
# Configs derived from NVIDIA-Nemotron-3-Super-120B-A12B Mamba2 parameters
# (nheads=128, headdim=64, d_state=128, ngroups=8) with TP split applied:
#   TP=8: nheads=16, ngroups=1   — primary production config
#   TP=4: nheads=32, ngroups=2   — exercises ngroups>1 (grouped B/C path)
# ---------------------------------------------------------------------------
_CONFIGS = [
    # (nheads, head_dim, d_state, ngroups)
    (16, 64, 128, 1),   # TP=8 production config
    (32, 64, 128, 2),   # TP=4, ngroups>1 (more heads than B/C groups)
]


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("paged_cache", [False, True],
                         ids=["no_cache_indices", "paged_cache"])
def test_incremental_selective_state_update(nheads, head_dim, d_state, ngroups,
                                            state_dtype, paged_cache):
    """
    Verify that:
      incremental_selective_state_update(state0, old_inputs, k, new_x, ...)
    produces the same output as:
      selective_state_update(state_after_k_old_tokens, new_x, ...)
    and writes state_after_k_old_tokens back to the state tensor.

    Precision note: the incremental kernel accumulates old-token replays in fp32
    registers before storing back to bf16 memory, and continues using that fp32
    register value for the new-token output — it does NOT reload from bf16.
    Therefore the reference intermediate states are captured in float32 to match.

    Parametrized over:
      - paged_cache=False: state_batch_indices is None (batch == cache slots)
      - paged_cache=True:  batch=2 scattered into a 4-slot cache (slots 1 and 3)
    """
    T = 6      # number of tokens in each step, draft length + 1
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16   # input activations are bf16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device,
                                           dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    # A: (nheads, head_dim, d_state) with stride(-2)=0, stride(-1)=0  [tie_hdim]
    A_base = -torch.rand(nheads, device=device) - 0.5   # float32, negative
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)

    # dt_bias: (nheads, head_dim) with stride(-1)=0  [tie_hdim]
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    # D: (nheads, head_dim) — pass to avoid a pre-existing bug in
    # selective_state_update when D=None (TypeError on `*(strides) if D is not None else 0`)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Initial SSM state (cache_size slots)
    state0 = torch.randn(cache_size, nheads, head_dim, d_state,
                         device=device, dtype=state_dtype)

    # Old inputs: T tokens per batch request
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)   # stride(-1)=0
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # -------------------------------------------------------------------
    # Capture intermediate SSM states using selective_state_update.
    # Precision: Even for bf16 ssm state, use float32 buffer so the captured
    # states match the fp32 register accumulation inside the incremental
    # kernel.  (The kernel loads bf16 state0, accumulates in fp32, stores back
    # to bf16.  A float32 buffer avoids an extra bf16 rounding that would cause
    # output mismatches.)  Caching requires state_batch_indices
    # (HAS_STATE_BATCH_INDICES gate).
    # -------------------------------------------------------------------
    states_buffer_f32 = torch.zeros(cache_size, T, nheads, head_dim, d_state,
                                    device=device, dtype=torch.float32)
    cache_idx_for_capture = (state_batch_indices if paged_cache else
                             torch.arange(batch, device=device, dtype=torch.int32))
    out1 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        state0.clone(),
        x1, dt1, A, B1, C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=T,
        out=out1,
        disable_state_update=True,
    )
    # states_buffer_f32[slot, k] = SSM state (fp32) after applying x1[..., k] to state0[slot]

    # -------------------------------------------------------------------
    # Build intermediate_update_inputs:
    # layout (cache_size, T, nheads*head_dim + nheads + ngroups*d_state)
    # -------------------------------------------------------------------
    old_x_flat = rearrange(x1, "b t h p -> b t (h p)")
    old_B_flat = rearrange(B1, "b t g n -> b t (g n)")
    interm_batch = torch.cat([old_x_flat, dt1_base, old_B_flat],
                             dim=-1).contiguous()
    if paged_cache:
        intermediate_update_inputs = torch.zeros(
            cache_size, T,
            nheads * head_dim + nheads + ngroups * d_state,
            device=device, dtype=dtype)
        intermediate_update_inputs[state_batch_indices] = interm_batch
    else:
        intermediate_update_inputs = interm_batch

    # -------------------------------------------------------------------
    # Main loop: test each k (number of old tokens replayed)
    # -------------------------------------------------------------------
    for k in range(T + 1):
        torch.manual_seed(k + 100)

        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        slots = state_batch_indices if paged_cache else slice(None)

        # Reference state in float32 (matches incremental kernel's fp32 register path)
        # For k=0: state0 loaded as fp32 (bf16→fp32 is lossless for bf16 values)
        # For k>0: fp32 state captured after k old tokens
        ref_state_f32 = state0.float().clone()
        if k > 0:
            ref_state_f32[slots] = states_buffer_f32[slots, k - 1]

        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state_f32,   # float32 starting state
            x2, dt2, A, B2, C2,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=(state_batch_indices if paged_cache else None),
            out=ref_out,
        )

        # Incremental kernel
        test_state = state0.clone()
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        incremental_selective_state_update(
            test_state,
            intermediate_update_inputs.clone(),
            prev_tokens,
            x=x2, dt=dt2, A=A, B=B2, C=C2,
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
        )

        # Output must match.  The incremental kernel uses bf16 tl.dot for the
        # output phase (C·state and CB·x), matching ssd_chunk_scan convention.
        # The reference uses scalar fp32 ops, so we allow bf16-level tolerance.
        torch.testing.assert_close(test_out, ref_out, rtol=2e-2, atol=5e-1,
                                   msg=f"Output mismatch at k={k}")

        # State in memory: incremental kernel stores (state after k old tokens) cast to
        # state_dtype.  Compare against the float32 reference cast to the same dtype.
        expected_state = (state0[slots] if k == 0
                          else states_buffer_f32[slots, k - 1].to(state_dtype))
        torch.testing.assert_close(test_state[slots], expected_state,
                                   rtol=1e-2, atol=1e-2,
                                   msg=f"State mismatch at k={k}")
