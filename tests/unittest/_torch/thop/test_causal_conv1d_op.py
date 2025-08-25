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

import random
from itertools import product

import pytest
import torch
from utils.torch_ref import mamba_conv1d_ref

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.llmapi.utils import get_total_gpu_memory


@pytest.mark.parametrize(
    "dim, dconv, req_type, dtype, batch_size, max_seq_len, remove_padding, apply_silu, paged_cache, use_initial_state",
    list(
        product([2048], [4], ['context', 'generation'],
                ['float16', 'float32', 'bfloat16'], [5], [16], [False, True],
                [False, True], [False, True], [False])) +
    # test with initial state
    list(
        product([2048], [4], ['context'], ['bfloat16'], [5], [16],
                [False, True], [False], [False, True], [True])) +
    # long sequence tests to cover the int overflow issue
    list(
        map(
            lambda x: pytest.param(
                *x,
                marks=pytest.mark.
                skipif(get_total_gpu_memory(0) < 33 * 1024**3,
                       reason=
                       "The long sequence test needs at least 33GB memory, skipping"
                       )),
            product([5376], [4], ['context'], ['float16', 'bfloat16'], [2],
                    [131072], [False, True], [False, True], [False], [False]))))
@pytest.mark.high_cuda_memory
def test_causal_conv1d(dim, dconv, req_type, dtype, batch_size, max_seq_len,
                       remove_padding, apply_silu, paged_cache,
                       use_initial_state):
    device = "cuda"
    seq_len = max_seq_len if req_type == "context" else 1
    mean = 0.0
    std_dev = 1.0 if dtype == "float32" else 0.5
    torch_dtype = str_dtype_to_torch(dtype)

    # test data
    torch.random.manual_seed(0)

    query_start_loc = None
    if remove_padding and req_type == "context":
        last_token_ids = torch.tensor(
            [0] + [random.randint(1, seq_len) for _ in range(batch_size)],
            dtype=torch.int32).to(device)
        last_token_ids[1] = seq_len
        host_context_lengths = last_token_ids[1:].detach().clone().cpu()
        query_start_loc = torch.cumsum(last_token_ids, dim=0,
                                       dtype=torch.int32).to(device)
    else:
        host_context_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32) * seq_len

    if req_type == "context" and not use_initial_state:
        conv_state = torch.zeros([batch_size, dim, dconv - 1],
                                 dtype=torch_dtype,
                                 device=device)
    else:
        conv_state = torch.randn(batch_size,
                                 dim,
                                 dconv - 1,
                                 dtype=torch_dtype,
                                 device=device)
        conv_state.normal_(mean, std_dev)

    conv_weight = torch.randn([dim, 1, dconv], dtype=torch_dtype, device=device)

    conv_bias = torch.randn([dim], dtype=torch_dtype, device=device)

    x = torch.empty(batch_size, dim, seq_len, device=device, dtype=torch_dtype)
    x.normal_(mean, std_dev)

    if req_type == "context" and remove_padding:
        x_batches = []
        for b in range(batch_size):
            x_batches.append(x[b, :, :host_context_lengths[b]])
            x_in_out = torch.cat(x_batches, dim=1)
    else:
        x_in_out = x.detach().clone()

    if paged_cache:
        padded_batch_size = 2 * batch_size
        cache_indices = torch.randperm(padded_batch_size,
                                       device=device,
                                       dtype=torch.int32)[:batch_size]
        conv_state_in_out = torch.empty([padded_batch_size, dim, dconv - 1],
                                        dtype=torch_dtype,
                                        device=device)
        conv_state_in_out[cache_indices] = conv_state.detach().clone()
    else:
        cache_indices = None
        conv_state_in_out = conv_state.detach().clone()

    conv_weight_input = conv_weight.squeeze(1).contiguous()

    if req_type == "context":
        has_initial_state = None if not use_initial_state else torch.ones(
            batch_size, device=device, dtype=torch.bool)

        torch.ops.trtllm.causal_conv1d_fwd(
            x_in_out,
            conv_weight_input,
            conv_bias,
            conv_state_in_out,
            query_start_loc,
            cache_indices,
            has_initial_state,
            apply_silu,
            PAD_SLOT_ID,
        )
        outputs = (x_in_out, conv_state_in_out[cache_indices]
                   if cache_indices is not None else conv_state_in_out)

    else:
        conv_state_indices = cache_indices
        cache_seqlens = None

        torch.ops.trtllm.causal_conv1d_update(
            x_in_out,
            conv_state_in_out,
            conv_weight_input,
            conv_bias,
            apply_silu,
            cache_seqlens,
            conv_state_indices,
            PAD_SLOT_ID,
        )
        outputs = (x_in_out, conv_state_in_out[cache_indices]
                   if cache_indices is not None else conv_state_in_out)

    out_ref = torch.zeros_like(x)
    conv_state_ref = torch.zeros_like(conv_state)

    for b in range(batch_size):
        (
            out_ref[b:b + 1, :, :host_context_lengths[b].item()],
            conv_state_ref[b:b + 1, :, :],
        ) = mamba_conv1d_ref(
            x[b:b + 1, :, :host_context_lengths[b].item()],
            conv_state[b:b + 1, :, :],
            conv_weight,
            conv_bias,
            apply_silu,
        )

    if remove_padding and req_type == "context":
        out_ref_batches = []
        for b in range(batch_size):
            out_ref_batches.append(out_ref[b, :, :host_context_lengths[b]])
        out_ref = torch.cat(out_ref_batches, dim=1)

    atol = {"float16": 1e-2, "float32": 2e-3, "bfloat16": 1e-1}

    torch.testing.assert_close(outputs[0], out_ref, rtol=1e-2, atol=atol[dtype])
    torch.testing.assert_close(outputs[1],
                               conv_state_ref,
                               rtol=1e-2,
                               atol=atol[dtype])
