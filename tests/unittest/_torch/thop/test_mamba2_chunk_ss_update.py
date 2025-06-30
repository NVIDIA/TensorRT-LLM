# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itertools import product

import pytest
import torch
from einops import rearrange, repeat
from utils.torch_ref import (selective_state_update_ref,
                             ssd_chunk_scan_combined_ref)

from tensorrt_llm._torch.modules.mamba.selective_state_update import \
    selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import \
    mamba_chunk_scan_combined
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.llmapi.utils import get_total_gpu_memory


@pytest.mark.parametrize(
    "dim, headdim, ngroups, dstate, req_type, dtype, batch_size, max_seq_len, has_z, remove_padding, paged_cache",
    # dim parametrization
    list(
        product([1024, 2048, 5120], [64], [1], [128], ['context', 'generation'],
                ['bfloat16'], [3], [16], [False], [True], [False])) +
    # headdim parametrization
    list(
        product([2048], [32, 64, 128, 256], [1], [128],
                ['context', 'generation'], ['bfloat16'], [3], [16], [False],
                [True], [False])) +
    # ngroups parametrization
    list(
        product([2048], [64], [1, 4], [128], ['context', 'generation'],
                ['bfloat16'], [3], [16], [False], [True], [False])) +
    # dstate parametrization
    list(
        product([2048], [64], [1], [64, 96, 128, 256],
                ['context', 'generation'], ['bfloat16'], [3], [16], [False],
                [True], [False])) +
    # dtype parametrization
    list(
        product([2048], [64], [1], [128], ['context', 'generation'],
                ['float16', 'bfloat16', 'float32'], [3], [16], [False], [True],
                [False])) +
    # batch_size parametrization
    list(
        product([2048], [64], [1], [128], ['context', 'generation'],
                ['bfloat16'], [1, 2, 8, 16], [16], [False], [True], [False])) +
    # max_seq_len parametrization
    list(
        product([2048], [64], [1], [128], ['context', 'generation'],
                ['bfloat16'], [3], [32, 64, 256, 2048, 16384], [False], [True],
                [False])) +
    # has_z parametrization
    list(
        product([2048], [64], [1], [128], ['context', 'generation'],
                ['bfloat16'], [3], [32], [True, False], [True], [False])) +
    # remove_padding parametrization
    list(
        product([2048], [64], [1], [128], ['context', 'generation'],
                ['bfloat16'], [3], [32], [False], [True, False], [False])) +
    # paged_cache parametrization (relevant for generation only)
    list(
        product([2048], [64], [1], [128], ['generation'], ['bfloat16'], [3],
                [32], [False], [False], [True, False])) +
    # long sequence test to cover the int overflow issue
    [
        pytest.param(
            2048,
            64,
            1,
            128,
            'context',
            'float16',
            2,
            131072,
            False,
            False,
            False,
            marks=pytest.mark.skipif(
                get_total_gpu_memory(0) < 68 * 1024**3,
                reason=
                "The long sequence test needs at least 68GB memory, skipping"))
    ])
def test_mamba2_chunk_scan_selective_state_update(dim, headdim, ngroups, dstate,
                                                  req_type, dtype, batch_size,
                                                  max_seq_len, has_z,
                                                  remove_padding, paged_cache):
    # configs
    device = "cuda"
    seq_len = max_seq_len if req_type == 'context' else 1
    long_context = max_seq_len >= 128 * 1024
    chunk_size = 256
    nheads = dim // headdim
    delta_softplus = True
    mean = 0.0
    if long_context:
        std_dev = 0.05
    elif dtype == "float32":
        std_dev = 0.5
    else:
        std_dev = 0.1

    torch_dtype = str_dtype_to_torch(dtype)

    # test data
    torch.random.manual_seed(0)
    if req_type == 'context' and remove_padding:
        last_token_ids = torch.randint(1,
                                       seq_len + 1, (batch_size, ),
                                       dtype=torch.int32,
                                       device=device)
        last_token_ids[0] = seq_len
        cu_seqlens = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(last_token_ids, dim=0, dtype=torch.int32)
        ],
                               dim=0)
        seq_idx = torch.repeat_interleave(
            torch.arange(len(last_token_ids), dtype=torch.int32, device=device),
            last_token_ids,
            output_size=cu_seqlens[-1]).unsqueeze(0)
        input_batch_size = 1
        input_seq_len = cu_seqlens[-1]
    else:
        input_batch_size = batch_size
        input_seq_len = seq_len
    state = torch.empty(batch_size,
                        nheads,
                        headdim,
                        dstate,
                        device=device,
                        dtype=torch_dtype)
    x = torch.empty(input_batch_size,
                    input_seq_len,
                    nheads,
                    headdim,
                    device=device,
                    dtype=torch_dtype)
    x.normal_(mean, std_dev)
    state.normal_(mean, std_dev)
    dt = torch.randn(input_batch_size,
                     input_seq_len,
                     nheads,
                     device=device,
                     dtype=torch_dtype)
    dt_bias = torch.rand(nheads, device=device) - 4.0
    A = -torch.rand(nheads, device=device) - 1.0
    B = torch.randn(input_batch_size,
                    input_seq_len,
                    ngroups,
                    dstate,
                    device=device,
                    dtype=torch_dtype)
    C = torch.randn_like(B)
    D = torch.randn(nheads, device=device)
    if has_z:
        z = torch.randn_like(x)

    if req_type == 'generation':
        # remove the seqlen dimension
        x = x.squeeze(1)
        B = B.squeeze(1)
        C = C.squeeze(1)
        z = z.squeeze(1) if has_z else None
        dt = dt.squeeze(1)

        # repeat for multiple heads
        A = repeat(A, "h -> h p n", p=headdim, n=dstate).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=headdim)
        dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
        D = repeat(D, "h -> h p", p=headdim)

    state_ref = rearrange(state, "... p n -> ... n p").detach().clone()
    x_ref = x.detach().clone()
    dt_ref = dt.detach().clone()
    dt_bias_ref = dt_bias.detach().clone()
    A_ref = (rearrange(A, "... p n -> ... n p")
             if A.ndim == 3 else A).detach().clone()
    B_ref = B.detach().clone()
    C_ref = C.detach().clone()
    D_ref = D.detach().clone()
    z_ref = z.detach().clone() if has_z else None

    if req_type == "context":
        out, ssm_state = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            z=z if has_z else None,
            dt_bias=dt_bias,
            seq_idx=seq_idx if remove_padding else None,
            cu_seqlens=cu_seqlens if remove_padding else None,
            dt_softplus=delta_softplus,
            return_final_states=not remove_padding,
            return_varlen_states=remove_padding,
        )

        if (ssm_state.shape[0] > 1 and ssm_state.dtype == torch.float32
                and torch_dtype != torch.float32):
            # In batched mode (i.e. - batch_dim>1), the ssm_states are created in float32 inside the kernel.
            # Batched mode isn't used when serving a model (we use cu_seqlens instead) so no point in changing
            # the kernel to control ssm_state dtype. Just cast to test dtype here
            ssm_state = ssm_state.to(dtype=torch_dtype)

        outputs = (out, ssm_state)

    else:
        if paged_cache:
            padded_batch_size = 2 * batch_size
            state_batch_indices = torch.randperm(padded_batch_size,
                                                 device=device,
                                                 dtype=torch.int32)[:batch_size]
            orig_state = state.detach().clone()
            state = torch.empty([padded_batch_size, nheads, headdim, dstate],
                                dtype=torch_dtype,
                                device=device)
            state[state_batch_indices] = orig_state
        else:
            state_batch_indices = None

        y = selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z if has_z else None,
            dt_bias=dt_bias,
            dt_softplus=delta_softplus,
            state_batch_indices=state_batch_indices,
        )
        outputs = (y, state[state_batch_indices]
                   if state_batch_indices is not None else state)

    # pytorch run
    if req_type == 'context':
        if remove_padding:
            out_ref = torch.zeros(input_batch_size,
                                  input_seq_len,
                                  nheads,
                                  headdim,
                                  device=device,
                                  dtype=torch_dtype)
            for i in range(batch_size):
                start = cu_seqlens[i]
                end = cu_seqlens[i + 1]
                part_out_ref, part_state_ref = ssd_chunk_scan_combined_ref(
                    x_ref[:, start:end, ...],
                    dt_ref[:, start:end, ...],
                    A_ref,
                    B_ref[:, start:end, ...],
                    C_ref[:, start:end, ...],
                    chunk_size,
                    D=D_ref,
                    z=z_ref[:, start:end, ...] if has_z else None,
                    dt_bias=dt_bias_ref,
                    dt_softplus=delta_softplus)
                out_ref[0, start:end, ...] = part_out_ref.squeeze(0)
                state_ref[i, ...] = part_state_ref.squeeze(0)
        elif long_context:
            out_ref = torch.zeros(batch_size,
                                  seq_len,
                                  nheads,
                                  headdim,
                                  device=device,
                                  dtype=torch_dtype)
            # to save memory
            for i in range(batch_size):
                part_out_ref, part_state_ref = ssd_chunk_scan_combined_ref(
                    x_ref[i:i + 1, ...],
                    dt_ref[i:i + 1, ...],
                    A_ref,
                    B_ref[i:i + 1, ...],
                    C_ref[i:i + 1, ...],
                    chunk_size,
                    D=D_ref,
                    z=z_ref[i:i + 1, ...] if has_z else None,
                    dt_bias=dt_bias_ref,
                    dt_softplus=delta_softplus)
                out_ref[i, ...] = part_out_ref.squeeze(0)
                state_ref[i, ...] = part_state_ref.squeeze(0)
        else:
            out_ref, state_ref = ssd_chunk_scan_combined_ref(
                x_ref,
                dt_ref,
                A_ref,
                B_ref,
                C_ref,
                chunk_size,
                D=D_ref,
                z=z_ref if has_z else None,
                dt_bias=dt_bias_ref,
                dt_softplus=delta_softplus)
    elif req_type == 'generation':
        out_ref = selective_state_update_ref(state_ref,
                                             x_ref,
                                             dt_ref,
                                             A_ref,
                                             B_ref,
                                             C_ref,
                                             D=D_ref,
                                             z=z_ref,
                                             dt_bias=dt_bias_ref,
                                             dt_softplus=delta_softplus)
    state_ref = rearrange(state_ref, "... n p-> ... p n")

    atol = {"float16": 2e-2, "float32": 1e-2, "bfloat16": 1e-1}

    torch.testing.assert_close(outputs[0], out_ref, rtol=1e-2, atol=atol[dtype])
    torch.testing.assert_close(outputs[1],
                               state_ref,
                               rtol=1e-2,
                               atol=atol[dtype])
