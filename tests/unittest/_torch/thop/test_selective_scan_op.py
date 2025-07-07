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
import unittest
from itertools import product

import pytest
import torch
from einops import rearrange, repeat
from parameterized import parameterized
from utils.torch_ref import (selective_scan_ref, selective_state_update_ref,
                             ssd_chunk_scan_combined_ref)
from utils.util import unittest_name_func

import tensorrt_llm


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(list(
        product([2048], [16], ['context', 'generation'],
                ['float16', 'float32', 'bfloat16'], [3], [16], [False, True])),
                          name_func=unittest_name_func)
    def test_selective_scan(self, dim, dstate, req_type, dtype, batch_size,
                            max_seq_len, remove_padding):

        # configs
        device = "cuda"
        seq_len = max_seq_len if req_type == 'context' else 1
        dt_rank = 160
        delta_softplus = True
        torch_dtype = getattr(torch, dtype)

        # not used in mamba_v1
        nheads = 0
        ngroups = 1
        chunk_size = 0

        # test data
        torch.random.manual_seed(0)
        if remove_padding:
            last_token_ids = torch.randint(1,
                                           seq_len + 1, (batch_size, ),
                                           dtype=torch.int32)
            host_context_lengths = last_token_ids.detach().clone().cpu()
            last_token_ids = torch.cumsum(last_token_ids,
                                          dim=0,
                                          dtype=torch.int32).to(device)
            total_num_tokens = last_token_ids[batch_size - 1]
        else:
            last_token_ids = torch.ones(
                (batch_size, ), dtype=torch.int32, device=device) * seq_len
            host_context_lengths = last_token_ids.detach().clone().cpu()
            total_num_tokens = batch_size * seq_len
        state = torch.randn(batch_size,
                            dstate,
                            dim,
                            device=device,
                            dtype=torch_dtype)
        x = torch.randn(total_num_tokens, dim, device=device, dtype=torch_dtype)
        dt = torch.randn(total_num_tokens,
                         dim,
                         device=device,
                         dtype=torch_dtype)
        dt_bias = torch.rand(dim, device=device) - 4.0
        A = -torch.rand(dstate, dim, device=device) - 1.0
        BC = torch.randn(total_num_tokens,
                         dt_rank + dstate * 2,
                         device=device,
                         dtype=torch_dtype)
        D = torch.randn(dim, device=device)
        z = torch.randn_like(x)
        host_request_types = torch.tensor([0 if req_type == 'context' else 1] *
                                          batch_size,
                                          dtype=torch.int32)
        if not remove_padding or req_type == 'generation':
            x = x.view(-1, seq_len, dim)
            dt = dt.view(-1, seq_len, dim)
            BC = BC.view(-1, seq_len, dt_rank + dstate * 2)
            z = z.view(-1, seq_len, dim)

        if remove_padding and req_type == "generation":
            output = torch.zeros(x.squeeze(1).shape,
                                 device=device,
                                 dtype=torch_dtype)
        else:
            output = torch.zeros(x.shape, device=device, dtype=torch_dtype)

        state_ref = state.detach().clone()
        x_ref = x.detach().clone()
        dt_ref = dt.detach().clone()
        dt_bias_ref = dt_bias.detach().clone()
        A_ref = A.detach().clone()
        B_ref = BC[..., dt_rank:dt_rank + dstate].detach().clone()
        C_ref = BC[..., dt_rank + dstate:].detach().clone()
        D_ref = D.detach().clone()
        z_ref = z.detach().clone()

        if remove_padding and req_type == "generation":
            x = x.squeeze(1)

        is_mamba2 = False
        slot_mapping = None
        is_paged_state = False

        outputs = torch.ops.trtllm.selective_scan(
            x,
            state,
            dt,
            dt_bias,
            A,
            BC,
            D,
            host_request_types,
            last_token_ids,
            z,
            host_context_lengths,
            slot_mapping,
            dim,
            dstate,
            nheads,
            ngroups,
            chunk_size,
            dt_rank,
            delta_softplus,
            remove_padding,
            is_mamba2,
            is_paged_state,
        )

        if remove_padding and req_type == "generation":
            out_ref = torch.zeros(output.unsqueeze(1).shape,
                                  device=device,
                                  dtype=torch_dtype)
        else:
            out_ref = torch.zeros(output.shape,
                                  device=device,
                                  dtype=torch_dtype)

        if req_type == 'context':
            # pytorch run
            if remove_padding:
                for i in range(batch_size):
                    start_id = 0 if i == 0 else last_token_ids[i - 1]
                    end_id = last_token_ids[i]
                    part_out_ref, part_state_ref = selective_scan_ref(
                        x_ref[start_id:end_id].unsqueeze(0),
                        dt_ref[start_id:end_id].unsqueeze(0),
                        A_ref,
                        B_ref[start_id:end_id].unsqueeze(0),
                        C_ref[start_id:end_id].unsqueeze(0),
                        D=D_ref,
                        z=z_ref[start_id:end_id].unsqueeze(0),
                        delta_bias=dt_bias_ref,
                        delta_softplus=True)
                    out_ref[start_id:end_id][:] = part_out_ref.squeeze(0)
                    state_ref[i][:][:] = part_state_ref.squeeze(0)
            else:
                out_ref, state_ref = selective_scan_ref(x_ref,
                                                        dt_ref,
                                                        A_ref,
                                                        B_ref,
                                                        C_ref,
                                                        D=D_ref,
                                                        z=z_ref,
                                                        delta_bias=dt_bias_ref,
                                                        delta_softplus=True)
        elif req_type == 'generation':
            # pytorch run

            out_ref = selective_state_update_ref(state_ref,
                                                 x_ref.squeeze(1),
                                                 dt_ref.squeeze(1),
                                                 A_ref,
                                                 B_ref.squeeze(1),
                                                 C_ref.squeeze(1),
                                                 D=D_ref,
                                                 z=z_ref.squeeze(1),
                                                 dt_bias=dt_bias_ref,
                                                 dt_softplus=True)
            out_ref = out_ref.unsqueeze(1)

        atol = {"float16": 5e-3, "float32": 2e-3, "bfloat16": 5e-2}

        if remove_padding and req_type == "generation":
            out_ref = out_ref.squeeze(1)

        torch.testing.assert_close(outputs[0],
                                   out_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
        torch.testing.assert_close(outputs[1],
                                   state_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])

    @parameterized.expand(
        # P=8x and H=2x
        list(
            product([160, 320, 640], [80], [1], [128], ['context'], ['float16'],
                    [1, 2, 8, 16], [16, 64, 256], [True], [True])) +
        # normal tests
        list(
            product([2048], [64], [1, 4], [128], ['context', 'generation'],
                    ['float32', 'float16', 'bfloat16'], [3], [16],
                    [True, False], [True, False])) +
        # arbitrary N generation tests
        list(
            product([2048], [64], [1, 4], [16, 32, 48, 64, 80, 96, 128, 256],
                    ['generation'], ['float32', 'float16'], [3], [16], [True],
                    [True])) +
        # long sequence tests to cover the int overflow issue
        list(
            product([5120], [64], [1], [128], ['context'], ['float16'], [2],
                    [131072], [True, False], [True, False])) +
        # P=8x and H=2x
        list(
            product([144], [72], [1], [64, 128, 256], ['context', 'generation'],
                    ['float16'], [16], [16384], [True, False], [True, False])),
        name_func=unittest_name_func)
    def test_selective_scan_v2(self, dim, headdim, ngroups, dstate, req_type,
                               dtype, batch_size, max_seq_len, has_z,
                               remove_padding):
        pytest.skip("https://nvbugs/5324258")
        if dtype == 'float32' and req_type == 'context':
            pytest.skip(
                "Mamba2 chunk scan kernel only support float16 and bfloat16")
        if max_seq_len >= 128 * 1024:
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if total_gpu_mem <= 68 * 1024**3:
                pytest.skip(
                    "The long sequence test needs at least 68GB memory, skipping"
                )

        # configs
        device = "cuda"
        seq_len = max_seq_len if req_type == 'context' else 1
        long_context = max_seq_len >= 128 * 1024
        chunk_size = 256
        nheads = dim // headdim
        nheads_pad0 = (nheads + 7) // 8 * 8 - nheads
        delta_softplus = True
        mean = 0.0
        if long_context:
            std_dev = 0.05
        elif dtype == "float32":
            std_dev = 0.5
        else:
            std_dev = 0.1

        torch_dtype = getattr(torch, dtype)

        # test data
        torch.random.manual_seed(0)
        if remove_padding:
            last_token_ids = torch.randint(1,
                                           seq_len + 1, (batch_size, ),
                                           dtype=torch.int32)
            last_token_ids[0] = seq_len
            host_context_lengths = last_token_ids.detach().clone().cpu()
            last_token_ids = torch.cumsum(last_token_ids,
                                          dim=0,
                                          dtype=torch.int32).to(device)
            total_num_tokens = last_token_ids[batch_size - 1]
        else:
            last_token_ids = torch.ones(
                (batch_size, ), dtype=torch.int32, device=device) * seq_len
            host_context_lengths = last_token_ids.detach().clone().cpu()
            total_num_tokens = batch_size * seq_len
        state = torch.empty(batch_size,
                            nheads,
                            dstate,
                            headdim,
                            device=device,
                            dtype=torch_dtype)
        x = torch.empty(total_num_tokens, dim, device=device, dtype=torch_dtype)
        x.normal_(mean, std_dev)
        state.normal_(mean, std_dev)
        dt = torch.randn(total_num_tokens,
                         nheads,
                         device=device,
                         dtype=torch_dtype)
        if nheads_pad0:
            dt_pad0 = torch.randn(total_num_tokens,
                                  nheads_pad0,
                                  device=device,
                                  dtype=torch_dtype)
        dt_bias = torch.rand(nheads, device=device) - 4.0
        A = -torch.rand(nheads, device=device) - 1.0
        BC = torch.randn(total_num_tokens,
                         ngroups * dstate * 2,
                         device=device,
                         dtype=torch_dtype)
        D = torch.randn(nheads, device=device)
        if has_z:
            z = torch.randn_like(x)
        host_request_types = torch.tensor([0 if req_type == 'context' else 1] *
                                          batch_size,
                                          dtype=torch.int32)
        if not remove_padding or req_type == 'generation':
            x = x.view(-1, seq_len, dim)
            dt = dt.view(-1, seq_len, nheads)
            if nheads_pad0:
                dt_pad0 = dt_pad0.view(-1, seq_len, nheads_pad0)
            BC = BC.view(-1, seq_len, ngroups * dstate * 2)
            if has_z:
                z = z.view(-1, seq_len, dim)
        xBC = torch.concat([x, BC], dim=-1).contiguous()
        if has_z:
            zxBCdt = torch.concat([z, torch.randn_like(xBC), dt] + ([
                dt_pad0,
            ] if nheads_pad0 else []),
                                  dim=-1).contiguous()
        else:
            zxBCdt = torch.concat([torch.randn_like(xBC), dt] + ([
                dt_pad0,
            ] if nheads_pad0 else []),
                                  dim=-1).contiguous()

        if remove_padding and req_type == "generation":
            xBC = xBC.squeeze(1)

        if remove_padding and req_type == "generation":
            output = torch.zeros(x.squeeze(1).shape,
                                 device=device,
                                 dtype=torch_dtype)
        else:
            output = torch.zeros(x.shape, device=device, dtype=torch_dtype)

        state_ref = state.detach().clone()
        x_ref = x.detach().clone()
        dt_ref = dt.detach().clone()
        dt_bias_ref = dt_bias.detach().clone()
        A_ref = A.detach().clone()
        B_ref = BC[..., 0:ngroups * dstate].detach().clone()
        C_ref = BC[..., ngroups * dstate:].detach().clone()
        D_ref = D.detach().clone()
        z_ref = z.detach().clone() if has_z else None

        is_mamba2 = True
        slot_mapping = None
        delta_rank = 0
        is_paged_state = False

        outputs = torch.ops.trtllm.selective_scan(
            xBC,
            state,
            zxBCdt,
            dt_bias,
            A,
            xBC,
            D,
            host_request_types,
            last_token_ids,
            zxBCdt if has_z else None,
            host_context_lengths,
            slot_mapping,
            dim,
            dstate,
            nheads,
            ngroups,
            chunk_size,
            delta_rank,
            delta_softplus,
            remove_padding,
            is_mamba2,
            is_paged_state,
        )

        if remove_padding and req_type == "generation":
            out_ref = torch.zeros(output.unsqueeze(1).shape,
                                  device=device,
                                  dtype=torch_dtype)
        else:
            out_ref = torch.zeros(output.shape,
                                  device=device,
                                  dtype=torch_dtype)

        # pytorch run
        if req_type == 'context':
            if remove_padding:
                for i in range(batch_size):
                    start = 0 if i == 0 else last_token_ids[i - 1]
                    end = last_token_ids[i]
                    x_reshaped = rearrange(x_ref[start:end].unsqueeze(0),
                                           "b l (h p) -> b l h p",
                                           p=headdim)
                    B_ref_reshaped = rearrange(B_ref[start:end].unsqueeze(0),
                                               "b l (g n) -> b l g n",
                                               g=ngroups)
                    C_ref_reshaped = rearrange(C_ref[start:end].unsqueeze(0),
                                               "b l (g n) -> b l g n",
                                               g=ngroups)
                    z_ref_reshaped = rearrange(z_ref[start:end].unsqueeze(0),
                                               "b l (h p) -> b l h p",
                                               p=headdim) if has_z else None
                    part_out_ref, part_state_ref = ssd_chunk_scan_combined_ref(
                        x_reshaped,
                        dt_ref[start:end].unsqueeze(0),
                        A_ref,
                        B_ref_reshaped,
                        C_ref_reshaped,
                        chunk_size,
                        D=D_ref,
                        z=z_ref_reshaped,
                        dt_bias=dt_bias_ref,
                        dt_softplus=delta_softplus)
                    part_out_ref = rearrange(part_out_ref,
                                             "b l h p -> b l (h p)")
                    out_ref[
                        start:end,
                    ] = part_out_ref.squeeze(0)
                    state_ref[
                        i,
                    ] = part_state_ref.squeeze(0)
            elif long_context:
                # to save memory
                for i in range(batch_size):
                    x_reshaped = rearrange(x_ref[
                        i:i + 1,
                    ],
                                           "b l (h p) -> b l h p",
                                           p=headdim)
                    B_ref_reshaped = rearrange(B_ref[
                        i:i + 1,
                    ],
                                               "b l (g n) -> b l g n",
                                               g=ngroups)
                    C_ref_reshaped = rearrange(C_ref[
                        i:i + 1,
                    ],
                                               "b l (g n) -> b l g n",
                                               g=ngroups)
                    z_ref_reshaped = rearrange(
                        z_ref[
                            i:i + 1,
                        ], "b l (h p) -> b l h p", p=headdim) if has_z else None
                    part_out_ref, part_state_ref = ssd_chunk_scan_combined_ref(
                        x_reshaped,
                        dt_ref[
                            i:i + 1,
                        ],
                        A_ref,
                        B_ref_reshaped,
                        C_ref_reshaped,
                        chunk_size,
                        D=D_ref,
                        z=z_ref_reshaped,
                        dt_bias=dt_bias_ref,
                        dt_softplus=delta_softplus)
                    part_out_ref = rearrange(part_out_ref,
                                             "b l h p -> b l (h p)")
                    out_ref[
                        i,
                    ] = part_out_ref.squeeze(0)
                    state_ref[
                        i,
                    ] = part_state_ref.squeeze(0)
            else:
                x_reshaped = rearrange(x_ref, "b l (h p) -> b l h p", p=headdim)
                B_ref_reshaped = rearrange(B_ref,
                                           "b l (g n) -> b l g n",
                                           g=ngroups)
                C_ref_reshaped = rearrange(C_ref,
                                           "b l (g n) -> b l g n",
                                           g=ngroups)
                z_ref_reshaped = rearrange(
                    z_ref, "b l (h p) -> b l h p", p=headdim) if has_z else None
                out_ref, state_ref = ssd_chunk_scan_combined_ref(
                    x_reshaped,
                    dt_ref,
                    A_ref,
                    B_ref_reshaped,
                    C_ref_reshaped,
                    chunk_size,
                    D=D_ref,
                    z=z_ref_reshaped,
                    dt_bias=dt_bias_ref,
                    dt_softplus=delta_softplus)
                out_ref = rearrange(out_ref, "b l h p -> b l (h p)")
        elif req_type == 'generation':
            A_ref = repeat(A_ref, "h -> h n p", p=headdim,
                           n=dstate).to(dtype=torch.float32)
            dt_ref = repeat(dt_ref.squeeze(1), "b h -> b h p", p=headdim)
            dt_bias_ref = repeat(dt_bias_ref, "h -> h p", p=headdim)
            D_ref = repeat(D_ref, "h -> h p", p=headdim)
            B_ref = rearrange(B_ref.squeeze(1), "b (g n) -> b g n", g=ngroups)
            C_ref = rearrange(C_ref.squeeze(1), "b (g n) -> b g n", g=ngroups)
            x_reshaped = rearrange(x_ref.squeeze(1),
                                   "b (h p) -> b h p",
                                   p=headdim)
            if has_z:
                z_ref = rearrange(z_ref.squeeze(1),
                                  "b (h p) -> b h p",
                                  p=headdim)
            out_ref = selective_state_update_ref(state_ref,
                                                 x_reshaped,
                                                 dt_ref,
                                                 A_ref,
                                                 B_ref,
                                                 C_ref,
                                                 D=D_ref,
                                                 z=z_ref,
                                                 dt_bias=dt_bias_ref,
                                                 dt_softplus=delta_softplus)
            out_ref = rearrange(out_ref, "b h p -> b (h p)").unsqueeze(1)

        atol = {"float16": 2e-2, "float32": 2e-3, "bfloat16": 1e-1}

        if remove_padding and req_type == "generation":
            out_ref = out_ref.squeeze(1)

        torch.testing.assert_close(outputs[0],
                                   out_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
        torch.testing.assert_close(outputs[1],
                                   state_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
