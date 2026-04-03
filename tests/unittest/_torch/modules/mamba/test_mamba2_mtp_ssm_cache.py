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

import pytest
import torch
import torch.nn.functional as F
from einops import repeat


def mamba2_mtp_ssm_cache_ref(
    state,
    x,
    dt,
    A,
    B,
    C,
    out,
    intermediate_states_buffer,
    cache_steps,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=-1,
    disable_state_update=True,
    retrieve_parent_token=None,
    intermediate_state_indices=None,
):
    """
    PyTorch reference implementation for selective_state_update_mtp_ssm_cache_trtllm.

    Arguments:
        state: (batch, nheads, head_dim, ssm_dim)
        x: (batch, cache_steps, nheads, head_dim)
        dt: (batch, cache_steps, nheads, head_dim) strided
        A: (nheads, head_dim, ssm_dim) strided
        B: (batch, cache_steps, ngroups, ssm_dim)
        C: (batch, cache_steps, ngroups, ssm_dim)
        out: (batch, cache_steps, nheads, head_dim)
        intermediate_states_buffer: (batch, cache_steps, nheads, head_dim, ssm_dim)
        cache_steps: int
        D: (nheads, head_dim) optional
        z: (batch, cache_steps, nheads, head_dim) optional
        dt_bias: (nheads, head_dim) optional
        retrieve_parent_token: (batch, cache_steps) int32 optional
        intermediate_state_indices: (batch,) int32 optional
    """
    batch = x.shape[0]
    nheads = state.shape[1]
    ssm_dim = state.shape[3]
    ngroups = B.shape[2]
    nheads_ngroups_ratio = nheads // ngroups

    state = state.clone().float()
    out = out.clone()

    for b in range(batch):
        if state_batch_indices is not None:
            state_idx = state_batch_indices[b].item()
            if state_idx == pad_slot_id:
                continue
        else:
            state_idx = b

        if intermediate_state_indices is not None:
            cache_idx = intermediate_state_indices[b].item()
        elif state_batch_indices is not None:
            cache_idx = state_batch_indices[b].item()
        else:
            cache_idx = b

        cur_state = state[state_idx].float()  # (nheads, head_dim, ssm_dim)

        for t in range(cache_steps):
            # Tree-based: restore state from parent's cached intermediate
            if retrieve_parent_token is not None and t > 0:
                parent_idx = retrieve_parent_token[b, t].item()
                if 0 <= parent_idx < cache_steps:
                    cur_state = intermediate_states_buffer[cache_idx, parent_idx].float()

            x_t = x[b, t].float()  # (nheads, head_dim)
            dt_t = dt[b, t].float()  # (nheads, head_dim)

            if dt_bias is not None:
                dt_t = dt_t + dt_bias.float()
            if dt_softplus:
                dt_t = F.softplus(dt_t)

            A_val = A.float()  # (nheads, head_dim, ssm_dim)

            dA = torch.exp(A_val * dt_t.unsqueeze(-1))  # (nheads, head_dim, ssm_dim)

            B_t = B[b, t].float()  # (ngroups, ssm_dim)
            C_t = C[b, t].float()  # (ngroups, ssm_dim)

            # Expand B and C from ngroups to nheads (contiguous: each group
            # repeated nheads_ngroups_ratio times, matching kernel's
            # head // nheads_ngroups_ratio mapping)
            B_t_expanded = (
                B_t.unsqueeze(1)
                .expand(ngroups, nheads_ngroups_ratio, ssm_dim)
                .reshape(nheads, ssm_dim)
            )  # (nheads, ssm_dim)

            C_t_expanded = (
                C_t.unsqueeze(1)
                .expand(ngroups, nheads_ngroups_ratio, ssm_dim)
                .reshape(nheads, ssm_dim)
            )  # (nheads, ssm_dim)

            dB = B_t_expanded.unsqueeze(1) * dt_t.unsqueeze(-1)  # (nheads, head_dim, ssm_dim)

            cur_state = cur_state * dA + dB * x_t.unsqueeze(-1)

            # Cache intermediate state
            intermediate_states_buffer[cache_idx, t] = cur_state.to(
                intermediate_states_buffer.dtype
            )

            # Output: sum(state * C, dim=-1)
            out_t = (cur_state * C_t_expanded.unsqueeze(1)).sum(dim=-1)  # (nheads, head_dim)

            if D is not None:
                out_t = out_t + x_t * D.float()
            if z is not None:
                z_t = z[b, t].float()
                out_t = out_t * z_t * torch.sigmoid(z_t)

            out[b, t] = out_t.to(out.dtype)

        if not disable_state_update:
            state[state_idx] = cur_state.to(state.dtype)

    return out, intermediate_states_buffer


try:
    from tensorrt_llm._torch.modules.mamba.selective_state_update import (
        selective_state_update_mtp_ssm_cache_trtllm,
    )

    _op_available = True
except (ImportError, RuntimeError):
    _op_available = False

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available,
    reason="Requires CUDA and trtllm.mamba2_mtp_ssm_cache_update op",
)


@skip_unsupported
class TestMamba2MTPSSMCache:
    """Tests for the mamba2_mtp_ssm_cache_update CUDA kernel."""

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("cache_steps", [4, 8])
    @pytest.mark.parametrize("nheads,ngroups", [(32, 2), (16, 4)])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("ssm_dim", [128, 256])
    def test_with_all_optional_tensors(
        self, batch_size, cache_steps, nheads, ngroups, head_dim, ssm_dim
    ):
        """Test CUDA kernel vs PyTorch reference with all optional tensors set."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.bfloat16
        weight_dtype = torch.float32
        std = 0.5

        # SSM state
        state = torch.randn(batch_size, nheads, head_dim, ssm_dim, device=device, dtype=dtype) * std
        state_ref = state.clone()

        # Input tensors
        x = torch.randn(batch_size, cache_steps, nheads, head_dim, device=device, dtype=dtype) * std

        # dt is strided: (batch, cache_steps, nheads, head_dim) with stride (1, 0) on last two dims
        dt_base = (
            torch.randn(batch_size, cache_steps, nheads, device=device, dtype=weight_dtype) * std
        )
        dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)

        # A is strided: (nheads, head_dim, ssm_dim) with stride (1, 0, 0) on all dims
        A_base = torch.randn(nheads, device=device, dtype=weight_dtype) * std
        A = repeat(A_base, "h -> h p n", p=head_dim, n=ssm_dim)

        B = torch.randn(batch_size, cache_steps, ngroups, ssm_dim, device=device, dtype=dtype) * std
        C = torch.randn(batch_size, cache_steps, ngroups, ssm_dim, device=device, dtype=dtype) * std

        # D is strided: (nheads, head_dim) with stride (1, 0)
        D_base = torch.randn(nheads, device=device, dtype=weight_dtype) * std
        D = repeat(D_base, "h -> h p", p=head_dim)

        # z
        z = torch.randn(batch_size, cache_steps, nheads, head_dim, device=device, dtype=dtype) * std

        # dt_bias is strided: (nheads, head_dim) with stride (1, 0)
        dt_bias_base = torch.randn(nheads, device=device, dtype=weight_dtype) * std
        dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

        # state_batch_indices
        state_batch_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

        # intermediate_state_indices
        intermediate_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

        # retrieve_parent_token: a tree structure
        # For each batch, step 0 has parent -1, step t>0 has parent (t-1)//2
        # to form a binary-tree-like structure.
        parent_list = [-1] + [(t - 1) // 2 for t in range(1, cache_steps)]
        retrieve_parent_token = torch.tensor(
            [parent_list] * batch_size, device=device, dtype=torch.int32
        )

        # Output and intermediate buffers
        out_kernel = torch.zeros(
            batch_size, cache_steps, nheads, head_dim, device=device, dtype=dtype
        )
        out_ref = out_kernel.clone()
        intermediate_kernel = torch.zeros(
            batch_size, cache_steps, nheads, head_dim, ssm_dim, device=device, dtype=dtype
        )
        intermediate_ref = intermediate_kernel.clone()

        # Run CUDA kernel
        selective_state_update_mtp_ssm_cache_trtllm(
            state,
            x,
            dt,
            A,
            B,
            C,
            out_kernel,
            intermediate_kernel,
            cache_steps,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            pad_slot_id=-1,
            disable_state_update=True,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
        )

        # Run PyTorch reference
        out_ref, intermediate_ref = mamba2_mtp_ssm_cache_ref(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            out_ref,
            intermediate_ref,
            cache_steps,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            pad_slot_id=-1,
            disable_state_update=True,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
        )

        # The CUDA kernel is compiled with --use_fast_math and stores
        # intermediate states in bfloat16, so rare outliers from fast_math
        # exp/softplus are expected. Use a two-tier check: most elements
        # must be close, and only a tiny fraction may exceed the tolerance.
        tight_atol = 1e-2

        out_diff = (out_kernel.float() - out_ref.float()).abs()
        out_mismatch_frac = (out_diff > tight_atol).float().mean().item()
        assert out_mismatch_frac < 1e-4, (
            f"Too many out mismatches: {out_mismatch_frac:.6f} "
            f"(> 0.01%), max diff = {out_diff.max().item():.4f}"
        )

        inter_diff = (intermediate_kernel.float() - intermediate_ref.float()).abs()
        mismatch_frac = (inter_diff > tight_atol).float().mean().item()
        assert mismatch_frac < 1e-4, (
            f"Too many intermediate_states mismatches: {mismatch_frac:.6f} "
            f"(> 0.01%), max diff = {inter_diff.max().item():.4f}"
        )
