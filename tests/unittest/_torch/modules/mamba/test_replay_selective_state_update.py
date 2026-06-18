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
import torch.nn.functional as F
from einops import repeat

from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import (
    replay_selective_state_update,
)
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._utils import get_sm_version

# Philox stochastic rounding uses PTX cvt.rs.f16x2.f32 which requires sm >= 100.
_skip_pre_sm100 = pytest.mark.skipif(
    get_sm_version() < 100, reason="Philox stochastic rounding needs sm >= 100"
)

# Configs derived from NVIDIA-Nemotron-3-Super-120B-A12B Mamba2 parameters
# (nheads=128, headdim=64, d_state=128, ngroups=8) with TP split applied:
#   TP=8: nheads=16, ngroups=1   — primary production config
#   TP=4: nheads=32, ngroups=2   — exercises ngroups>1 (grouped B/C path)
_CONFIGS = [
    # (nheads, head_dim, d_state, ngroups)
    (16, 64, 128, 1),  # TP=8 production config
    (32, 64, 128, 2),  # TP=4, ngroups>1 (more heads than B/C groups)
]


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"])
@pytest.mark.parametrize(
    "T", [6, 10, 16, 27, 32, 55], ids=["T6", "T10", "T16", "T27", "T32", "T55"]
)
def test_replay_selective_state_update(
    nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, T
):
    """
    Verify that:
      replay_selective_state_update(state0, old_caches, k, new_x, ...)
    produces the same output as:
      selective_state_update(state_after_k_old_tokens, new_x, ...)
    and writes state_after_k_old_tokens back to the state tensor.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16  # input activations are bf16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    # A: (nheads, head_dim, d_state) with stride(-2)=0, stride(-1)=0  [tie_hdim]
    A_base = -torch.rand(nheads, device=device) - 0.5  # float32, negative
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)

    # dt_bias: (nheads, head_dim) with stride(-1)=0  [tie_hdim]
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    # D: (nheads, head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Initial SSM state (cache_size slots)
    state0 = torch.randn(cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype)

    # Old inputs: T tokens per batch request
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)  # stride(-1)=0
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # Capture intermediate SSM states using selective_state_update.
    states_buffer_f32 = torch.zeros(
        cache_size, T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    out1 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        state0.clone(),
        x1,
        dt1,
        A,
        B1,
        C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=T,
        out=out1,
        disable_state_update=True,
    )

    # Build cache tensors for the replay kernel.
    # old_x: (cache, T, nheads, dim) bf16 — single-buffered
    # old_B: (cache, 2, T, ngroups, dstate) bf16 — double-buffered
    # old_dt: (cache, 2, nheads, T) fp32 — double-buffered, T contiguous
    # old_dA_cumsum: (cache, 2, nheads, T) fp32 — double-buffered, T contiguous
    # cache_buf_idx: random 0s and 1s to verify indexing correctness
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Fill each slot's READ buffer (indexed by cache_buf_idx) with step 1's data.
    # The OTHER buffer has random garbage to catch indexing bugs.
    slots = state_batch_indices if paged_cache else slice(None)
    old_x[slots] = x1

    # Compute processed dt and dA_cumsum for step 1
    dt1 = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1, dim=1)

    # Write to each slot's read buffer based on its cache_buf_idx
    slot_indices = state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        batch_idx = i  # maps slot back to the batch index
        old_B[slot, buf] = B1[batch_idx]
        old_dt[slot, buf] = dt1[batch_idx].T  # (T, nheads) → (nheads, T)
        old_dA_cumsum[slot, buf] = dA_cumsum1[batch_idx].T  # (T, nheads) → (nheads, T)

    # Main loop: test each k (number of old tokens replayed)
    for k in range(T + 1):
        torch.manual_seed(k + 100)

        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        # Reference
        ref_state_f32 = state0.float().clone()
        if k > 0:
            ref_state_f32[slots] = states_buffer_f32[slots, k - 1]

        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state_f32,
            x2,
            dt2,
            A,
            B2,
            C2,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=(state_batch_indices if paged_cache else None),
            out=ref_out,
        )

        # Replay kernel
        test_state = state0.clone()
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        # cache_buf_idx stays at its random values — each slot reads from its own buffer

        replay_selective_state_update(
            test_state,
            old_x.clone(),
            old_B.clone(),
            old_dt.clone(),
            old_dA_cumsum.clone(),
            cache_buf_idx.clone(),
            prev_tokens,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
        )

        # Tolerance rationale: the replay kernel uses bf16 tl.dot for four
        # matmuls (dB_scaled @ old_x, C @ state, CB_scaled @ x, and C @ B in
        # precompute).  The reference (selective_state_update) and flashinfer
        # baseline use fp32 element-wise MACs.  The bf16 input casts lose the
        # dt_bias/A-derived bits that the baselines keep — per-element rounding,
        # not accumulating.  Prefill (ssd_chunk_scan) does identical bf16 tl.dot
        # casts, so we match prefill precision exactly.  Empirical: max ~1.0 at
        # T<=16, ~2.0 at T=32-55; mean ~0.014; <0.02% of elements exceed 0.5.
        # State dtype (fp16/bf16/fp32) doesn't shift the error — bf16 dot
        # inputs dominate, not state storage.
        torch.testing.assert_close(
            test_out, ref_out, rtol=2e-2, atol=1.0, msg=f"Output mismatch at k={k}"
        )

        expected_state = (
            state0[slots] if k == 0 else states_buffer_f32[slots, k - 1].to(state_dtype)
        )
        torch.testing.assert_close(
            test_state[slots], expected_state, rtol=2e-2, atol=1.0, msg=f"State mismatch at k={k}"
        )


@_skip_pre_sm100
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"])
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
def test_replay_selective_state_update_philox(nheads, head_dim, d_state, ngroups, paged_cache, T):
    """
    Verify that Philox stochastic rounding produces correct results.

    Runs our kernel twice with identical inputs: once without rounding
    (fp16 state, deterministic), once with rounding (fp16 state, Philox).
    The outputs should be nearly identical — stochastic rounding only
    perturbs the state by ±1 fp16 ULP, which barely affects output.
    Also verifies the state dtype remains fp16.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0 = torch.randn(cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype)

    # Cache tensors
    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    # New token inputs
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((cache_size,), T // 2, device=device, dtype=torch.int32)

    common_kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
    )

    # --- Run without rounding (deterministic fp16 state store) ---
    state_no_round = state0.clone()
    out_no_round = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_no_round,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_no_round,
        **common_kwargs,
    )

    # --- Run with Philox rounding ---
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rounded = state0.clone()
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Outputs should be nearly identical — rounding only perturbs the
    # post-replay state by ±1 ULP before the output phase reads it.
    torch.testing.assert_close(
        out_rounded, out_no_round, rtol=2e-2, atol=1.0, msg="Output diverged with Philox rounding"
    )

    # State should remain fp16
    assert state_rounded.dtype == torch.float16

    # States should differ by at most 1 fp16 ULP per element.
    # fp16 ULP depends on magnitude: up to 0.5 for values near 512.
    # Use rtol to account for magnitude-dependent ULP.
    slots = state_batch_indices if paged_cache else slice(None)
    torch.testing.assert_close(
        state_rounded[slots],
        state_no_round[slots],
        rtol=2e-3,
        atol=0.2,
        msg="State diverged with Philox rounding",
    )


@_skip_pre_sm100
def test_philox_rounding_unbiased():
    """
    Verify that Philox stochastic rounding is unbiased.

    Runs the replay kernel with fp32 state (capturing the true fp32
    post-replay state) and with fp16 state + Philox rounding.  Compares the
    rounding residual (fp16_state.float() - fp32_state) against deterministic
    rounding (fp32_state.to(fp16).float() - fp32_state).

    Deterministic round-to-nearest-even has a systematic positive bias on
    the residual.  Philox stochastic rounding should be unbiased: the mean
    residual should be near zero.

    Uses a large batch (16) for ~2M state elements — plenty of statistics.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 16, 6
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Use fp32 initial state so replay produces non-fp16-representable values
    state0 = torch.randn(batch, nheads, head_dim, d_state, device=device, dtype=torch.float32)

    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt_val = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((batch,), T, device=device, dtype=torch.int32)

    common_kwargs = dict(
        x=x,
        dt=dt_val,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    # 1. fp32 state — captures true post-replay state
    state_fp32 = state0.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_fp32,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_fp32,
        **common_kwargs,
    )

    # 2. fp16 state with Philox rounding
    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    state_rounded = state0.to(torch.float16).clone()
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    replay_selective_state_update(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Compute rounding residuals where fp32 state has non-zero values
    fp32_vals = state_fp32.flatten()
    stochastic_residual = state_rounded.float().flatten() - fp32_vals
    deterministic_residual = fp32_vals.to(torch.float16).float() - fp32_vals

    # Only consider elements where rounding matters (non-zero residual possible)
    nonzero_mask = deterministic_residual.abs() > 0
    num_nonzero = nonzero_mask.sum().item()
    assert num_nonzero > 1000, f"Too few roundable elements: {num_nonzero}"

    stochastic_mean = stochastic_residual[nonzero_mask].mean().item()
    deterministic_mean = deterministic_residual[nonzero_mask].mean().item()

    # Stochastic rounding should be less biased than deterministic.
    # With ~millions of elements, the stochastic mean should be very close to 0.
    # Deterministic round-to-nearest-even has a small but systematic bias.
    assert abs(stochastic_mean) < abs(deterministic_mean) or abs(stochastic_mean) < 1e-5, (
        f"Stochastic rounding appears biased: stochastic_mean={stochastic_mean:.6f}, "
        f"deterministic_mean={deterministic_mean:.6f}, n_elements={num_nonzero}"
    )


# HEADS_PER_BLOCK > 1 test.  The default heuristic only picks HPB > 1 at large
# total_heads (>= 256-512), which the main test with batch=2 never reaches.
# This test overrides _heads_per_block to exercise the two-loop structure in
# the precompute kernel (store-then-reload of per-head dt/dA_cumsum).
# Configs: (nheads=16, ngroups=1) and (nheads=32, ngroups=2) both have
# heads_per_group=16.  The heuristic caps HPB at min(2|4, hpg), so HPB=2, 4.
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
@pytest.mark.parametrize("heads_per_block", [2, 4], ids=["HPB2", "HPB4"])
@pytest.mark.parametrize("launch_with_pdl", [False, True], ids=["no_ext_pdl", "ext_pdl"])
@pytest.mark.parametrize("use_internal_pdl", [False, True], ids=["no_int_pdl", "int_pdl"])
@pytest.mark.parametrize("batch", [1, 2, 8, 16], ids=["B1", "B2", "B8", "B16"])
def test_replay_heads_per_block(
    nheads,
    head_dim,
    d_state,
    ngroups,
    state_dtype,
    T,
    heads_per_block,
    launch_with_pdl,
    use_internal_pdl,
    batch,
):
    """
    Verify replay_selective_state_update produces correct results when
    _heads_per_block > 1, exercising the precompute kernel's two-loop
    structure (store per-head dt/dA_cumsum in loop 1, reload in loop 2).
    """
    device = "cuda"
    dtype = torch.bfloat16

    if nheads % heads_per_block != 0:
        pytest.skip(f"nheads ({nheads}) not divisible by heads_per_block ({heads_per_block})")
    if heads_per_block > nheads // ngroups:
        pytest.skip(
            f"heads_per_block ({heads_per_block}) exceeds heads_per_group ({nheads // ngroups})"
        )

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    cache_size = batch
    state0 = torch.randn(cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype)

    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    states_buffer_f32 = torch.zeros(
        cache_size, T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = torch.arange(batch, device=device, dtype=torch.int32)
    out1 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        state0.clone(),
        x1,
        dt1,
        A,
        B1,
        C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=T,
        out=out1,
        disable_state_update=True,
    )

    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    old_x[:] = x1
    dt1 = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1, dim=1)

    for slot in range(cache_size):
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf] = B1[slot]
        old_dt[slot, buf] = dt1[slot].T
        old_dA_cumsum[slot, buf] = dA_cumsum1[slot].T

    k = T
    torch.manual_seed(123)

    x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    ref_state_f32 = state0.float().clone()
    ref_state_f32[:] = states_buffer_f32[:, k - 1]
    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32,
        x2,
        dt2,
        A,
        B2,
        C2,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=None,
        out=ref_out,
    )

    test_state = state0.clone()
    prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

    replay_selective_state_update(
        test_state,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        x=x2,
        dt=dt2,
        A=A,
        B=B2,
        C=C2,
        out=test_out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=None,
        _heads_per_block=heads_per_block,
        launch_with_pdl=launch_with_pdl,
        use_internal_pdl=use_internal_pdl,
    )

    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=2e-2,
        atol=1.0,
        msg=f"Output mismatch with HPB={heads_per_block}, T={T}, "
        f"nheads={nheads}, ngroups={ngroups}, state_dtype={state_dtype}",
    )

    expected_state = states_buffer_f32[:, k - 1].to(state_dtype)
    torch.testing.assert_close(
        test_state,
        expected_state,
        rtol=2e-2,
        atol=1.0,
        msg=f"State mismatch with HPB={heads_per_block}, T={T}, "
        f"nheads={nheads}, ngroups={ngroups}, state_dtype={state_dtype}",
    )


# HPB > 1 multi-step test.  Production chains decode steps; bugs in
# buffer ordering or stale cache values accumulate across steps and can
# be invisible in a single-step test.
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
@pytest.mark.parametrize("heads_per_block", [2, 4], ids=["HPB2", "HPB4"])
@pytest.mark.parametrize("paged_cache", [False, True], ids=["contig", "paged"])
def test_replay_heads_per_block_multistep(
    nheads, head_dim, d_state, ngroups, state_dtype, T, heads_per_block, paged_cache
):
    """
    Chain N decode steps with HPB > 1 and verify each step's output matches
    a fresh reference.  A bug that mixes up WRITE/READ buffers, writes wrong
    data to cache, or races in the two-loop structure would accumulate
    across steps.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    n_steps = 8

    if nheads % heads_per_block != 0:
        pytest.skip(f"nheads ({nheads}) not divisible by HPB ({heads_per_block})")
    if heads_per_block > nheads // ngroups:
        pytest.skip(f"HPB ({heads_per_block}) exceeds heads_per_group ({nheads // ngroups})")

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
        slots = state_batch_indices
    else:
        cache_size = batch
        state_batch_indices = None
        slots = slice(None)

    all_x = []
    all_dt = []
    all_B = []
    all_C = []
    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        all_x.append(torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype))
        dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        all_dt.append(repeat(dt_base, "b t h -> b t h p", p=head_dim))
        all_B.append(torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype))
        all_C.append(torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype))

    torch.manual_seed(999)
    state_init = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    ref_state = state_init.float().clone()
    ref_outs = []
    ref_slots = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    for step in range(n_steps):
        out_step = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state,
            all_x[step],
            all_dt[step],
            A,
            all_B[step],
            all_C[step],
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=ref_slots,
            out=out_step,
        )
        ref_outs.append(out_step)

    test_state = state_init.clone()
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.zeros(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.zeros(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.zeros(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    for step in range(n_steps):
        k = T if step > 0 else 0
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

        replay_selective_state_update(
            test_state,
            old_x,
            old_B,
            old_dt,
            old_dA_cumsum,
            cache_buf_idx,
            prev_tokens,
            x=all_x[step],
            dt=all_dt[step],
            A=A,
            B=all_B[step],
            C=all_C[step],
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            _heads_per_block=heads_per_block,
        )

        if paged_cache:
            cache_buf_idx[slots] = 1 - cache_buf_idx[slots]
        else:
            cache_buf_idx[:] = 1 - cache_buf_idx

        torch.testing.assert_close(
            test_out,
            ref_outs[step],
            rtol=2e-2,
            atol=2.0,
            msg=f"Output mismatch at step {step} with HPB={heads_per_block}, "
            f"T={T}, nheads={nheads}, ngroups={ngroups}, "
            f"state_dtype={state_dtype}, paged_cache={paged_cache}",
        )
