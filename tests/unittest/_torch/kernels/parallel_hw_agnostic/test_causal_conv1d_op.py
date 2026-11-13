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
        product(
            [2048],
            [4],
            ["context", "generation"],
            ["float16", "float32", "bfloat16"],
            [5],
            [16],
            [False, True],
            [False, True],
            [False, True],
            [False],
        )
    )
    +
    # test with initial state
    list(
        product(
            [2048],
            [4],
            ["context"],
            ["bfloat16"],
            [5],
            [16],
            [False, True],
            [False],
            [False, True],
            [True],
        )
    )
    +
    # long sequence tests to cover the int overflow issue
    list(
        map(
            lambda x: pytest.param(
                *x,
                marks=pytest.mark.skipif(
                    get_total_gpu_memory(0) < 33 * 1024**3,
                    reason="The long sequence test needs at least 33GB memory, skipping",
                ),
            ),
            product(
                [5376],
                [4],
                ["context"],
                ["float16", "bfloat16"],
                [2],
                [131072],
                [False, True],
                [False, True],
                [False],
                [False],
            ),
        )
    ),
)
@pytest.mark.high_cuda_memory
def test_causal_conv1d(
    dim,
    dconv,
    req_type,
    dtype,
    batch_size,
    max_seq_len,
    remove_padding,
    apply_silu,
    paged_cache,
    use_initial_state,
):
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
            [0] + [random.randint(1, seq_len) for _ in range(batch_size)], dtype=torch.int32
        ).to(device)
        last_token_ids[1] = seq_len
        host_context_lengths = last_token_ids[1:].detach().clone().cpu()
        query_start_loc = torch.cumsum(last_token_ids, dim=0, dtype=torch.int32).to(device)
    else:
        host_context_lengths = torch.ones((batch_size,), dtype=torch.int32) * seq_len

    if req_type == "context" and not use_initial_state:
        conv_state = torch.zeros([batch_size, dim, dconv - 1], dtype=torch_dtype, device=device)
    else:
        conv_state = torch.randn(batch_size, dim, dconv - 1, dtype=torch_dtype, device=device)
        conv_state.normal_(mean, std_dev)

    conv_weight = torch.randn([dim, 1, dconv], dtype=torch_dtype, device=device)

    conv_bias = torch.randn([dim], dtype=torch_dtype, device=device)

    x = torch.empty(batch_size, dim, seq_len, device=device, dtype=torch_dtype)
    x.normal_(mean, std_dev)

    if req_type == "context" and remove_padding:
        x_batches = []
        for b in range(batch_size):
            x_batches.append(x[b, :, : host_context_lengths[b]])
        x_in_out = torch.cat(x_batches, dim=1)
    else:
        x_in_out = x.detach().clone()

    if paged_cache:
        padded_batch_size = 2 * batch_size
        cache_indices = torch.randperm(padded_batch_size, device=device, dtype=torch.int32)[
            :batch_size
        ]
        conv_state_in_out = torch.empty(
            [padded_batch_size, dim, dconv - 1], dtype=torch_dtype, device=device
        )
        conv_state_in_out[cache_indices] = conv_state.detach().clone()
    else:
        cache_indices = None
        conv_state_in_out = conv_state.detach().clone()

    conv_weight_input = conv_weight.squeeze(1).contiguous()

    if req_type == "context":
        has_initial_state = (
            None
            if not use_initial_state
            else torch.ones(batch_size, device=device, dtype=torch.bool)
        )

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
        outputs = (
            x_in_out,
            conv_state_in_out[cache_indices] if cache_indices is not None else conv_state_in_out,
        )

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
        outputs = (
            x_in_out,
            conv_state_in_out[cache_indices] if cache_indices is not None else conv_state_in_out,
        )

    out_ref = torch.zeros_like(x)
    conv_state_ref = torch.zeros_like(conv_state)

    for b in range(batch_size):
        (
            out_ref[b : b + 1, :, : host_context_lengths[b].item()],
            conv_state_ref[b : b + 1, :, :],
        ) = mamba_conv1d_ref(
            x[b : b + 1, :, : host_context_lengths[b].item()],
            conv_state[b : b + 1, :, :],
            conv_weight,
            conv_bias,
            apply_silu,
        )

    if remove_padding and req_type == "context":
        out_ref_batches = []
        for b in range(batch_size):
            out_ref_batches.append(out_ref[b, :, : host_context_lengths[b]])
        out_ref = torch.cat(out_ref_batches, dim=1)

    atol = {"float16": 1e-2, "float32": 2e-3, "bfloat16": 1e-1}

    torch.testing.assert_close(outputs[0], out_ref, rtol=1e-2, atol=atol[dtype])
    torch.testing.assert_close(outputs[1], conv_state_ref, rtol=1e-2, atol=atol[dtype])


# A conv-state pool past 2^32 elements: Qwen3.8-Flash-Next's 10240 conv channels of width 4,
# viewed out of a slot page of 1105920 elements (the stride of a dep4 rank's recurrent pool).
# Slot 3884 is the first whose offset, 3884 * 1105920, passes 2^32; formed in 32 bits it wraps
# to element 425984, inside slot 0's page.
_BIG_DIM, _BIG_WIDTH, _BIG_STRIDE, _BIG_SLOT = 10240, 4, 1105920, 3884
_BIG_WRAP = _BIG_SLOT * _BIG_STRIDE - 2**32


@pytest.fixture(scope="module")
def big_pool_storage():
    """One bf16 buffer reaching slot `_BIG_SLOT`'s last element (8.6 GB), shared by the cases."""
    if get_total_gpu_memory(0) < 16 * 1024**3:
        pytest.skip("needs 16 GiB of device memory")
    numel = _BIG_SLOT * _BIG_STRIDE + _BIG_DIM * (_BIG_WIDTH - 1)
    storage = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
    yield storage
    del storage
    torch.cuda.empty_cache()


@pytest.mark.high_cuda_memory
@pytest.mark.parametrize(
    "op, seq_len",
    [("update", 1), ("update", 3), ("fwd", 16)],
    ids=["update_seqlen1", "update_seqlen3", "fwd_seqlen16"],
)
def test_conv_state_slot_past_32_bit_offset(big_pool_storage, op, seq_len):
    """A slot past 2^32 elements of the pool is read and written at its own address.

    Each kernel that indexes the pool by slot runs over two slots, `_BIG_SLOT` and 7: the
    single-token decode kernel (seqlen 1), the general update kernel (seqlen 3, an MTP verify
    step) and the channel-major forward kernel with an initial state. The result and the
    states must equal the same call on a compact copy of the two slots, bitwise, and the
    element range the slot's offset wraps to in 32 bits must be untouched.
    """
    torch.manual_seed(0)
    state_len = _BIG_WIDTH - 1
    pool = big_pool_storage.as_strided(
        (_BIG_SLOT + 1, _BIG_DIM, state_len), (_BIG_STRIDE, state_len, 1)
    )
    slots = torch.tensor([_BIG_SLOT, 7], dtype=torch.int32, device="cuda")
    pool[slots.long()] = torch.randn(2, _BIG_DIM, state_len, device="cuda").to(torch.bfloat16)
    wrap = big_pool_storage[_BIG_WRAP : _BIG_WRAP + _BIG_DIM * state_len]
    wrap.zero_()
    weight = torch.randn(_BIG_DIM, _BIG_WIDTH, device="cuda").to(torch.bfloat16)
    bias = torch.randn(_BIG_DIM, device="cuda").to(torch.bfloat16)
    x = torch.randn(2, _BIG_DIM, seq_len, device="cuda").to(torch.bfloat16)
    compact = pool[slots.long()].clone()
    x_ref = x.clone()

    if op == "update":
        torch.ops.trtllm.causal_conv1d_update(x, pool, weight, bias, True, None, slots, PAD_SLOT_ID)
        torch.ops.trtllm.causal_conv1d_update(
            x_ref, compact, weight, bias, True, None, None, PAD_SLOT_ID
        )
    else:
        initial = torch.ones(2, dtype=torch.bool, device="cuda")
        torch.ops.trtllm.causal_conv1d_fwd(
            x, weight, bias, pool, None, slots, initial, True, PAD_SLOT_ID
        )
        torch.ops.trtllm.causal_conv1d_fwd(
            x_ref, weight, bias, compact, None, None, initial, True, PAD_SLOT_ID
        )

    assert torch.equal(x, x_ref)
    assert torch.equal(pool[slots.long()], compact)
    assert not wrap.any(), "a write landed where the slot's offset wraps in 32 bits"


@pytest.mark.high_cuda_memory
def test_conv_state_batch_stride_past_32_bits_is_refused(big_pool_storage):
    """The kernels store strides in 32 bits: a pool whose slot stride is 2^32 is refused."""
    pool = big_pool_storage.as_strided((2, _BIG_DIM, _BIG_WIDTH - 1), (2**32, _BIG_WIDTH - 1, 1))
    slots = torch.tensor([1], dtype=torch.int32, device="cuda")
    weight = torch.randn(_BIG_DIM, _BIG_WIDTH, device="cuda").to(torch.bfloat16)
    x = torch.randn(1, _BIG_DIM, 1, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="32-bit strides"):
        torch.ops.trtllm.causal_conv1d_update(x, pool, weight, None, True, None, slots, PAD_SLOT_ID)
    with pytest.raises(RuntimeError, match="32-bit strides"):
        torch.ops.trtllm.causal_conv1d_fwd(
            x, weight, None, pool, None, slots, None, True, PAD_SLOT_ID
        )
