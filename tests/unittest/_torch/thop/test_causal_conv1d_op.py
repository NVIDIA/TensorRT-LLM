import random
import unittest
from itertools import product

import pytest
import torch
from parameterized import parameterized
from utils.torch_ref import mamba_conv1d_ref
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm._torch.modules.mamba.causal_conv1d import PAD_SLOT_ID


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        list(
            product([2048], [4], ['context', 'generation'],
                    ['float16', 'float32', 'bfloat16'], [5], [16],
                    [False, True], [False, True])) +
        # long sequence tests to cover the int overflow issue
        list(
            product([5376], [4], ['context'], ['float16', 'bfloat16'], [2],
                    [131072], [False, True], [False, True])),
        name_func=unittest_name_func)
    def test_causal_conv1d(self, dim, dconv, req_type, dtype, batch_size,
                           max_seq_len, remove_padding, apply_silu):
        if max_seq_len == 131072:
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if total_gpu_mem <= 33 * 1024**3:
                pytest.skip(
                    "The long sequence test needs at least 33GB memory, skipping"
                )

        device = "cuda"
        seq_len = max_seq_len if req_type == "context" else 1
        mean = 0.0
        std_dev = 1.0 if dtype == "float32" else 0.5
        torch_dtype = getattr(torch, dtype)

        # test data
        torch.random.manual_seed(0)

        query_start_loc = None
        if remove_padding and req_type == "context":
            last_token_ids = torch.tensor(
                [0] + [random.randint(1, seq_len) for _ in range(batch_size)],
                dtype=torch.int32).to(device)
            last_token_ids[1] = seq_len
            host_context_lengths = last_token_ids[1:].detach().clone().cpu()
            query_start_loc = torch.cumsum(last_token_ids,
                                           dim=0,
                                           dtype=torch.int32).to(device)
        else:
            host_context_lengths = torch.ones(
                (batch_size, ), dtype=torch.int32) * seq_len

        if req_type == "context":
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

        conv_weight = torch.randn([dim, 1, dconv],
                                  dtype=torch_dtype,
                                  device=device)

        conv_bias = torch.randn([dim], dtype=torch_dtype, device=device)

        x = torch.empty(batch_size,
                        dim,
                        seq_len,
                        device=device,
                        dtype=torch_dtype)
        x.normal_(mean, std_dev)

        if req_type == "context" and remove_padding:
            x_batches = []
            for b in range(batch_size):
                x_batches.append(x[b, :, :host_context_lengths[b]])
                x_in_out = torch.cat(x_batches, dim=1)
        else:
            x_in_out = x.detach().clone()

        conv_state_in_out = conv_state.detach().clone()
        conv_weight_input = conv_weight.squeeze(1).contiguous()

        if req_type == "context":
            cache_indices = None
            has_initial_state = None

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
            outputs = (x_in_out, conv_state_in_out)

        else:
            cache_seqlens = None
            conv_state_indices = None

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
            outputs = (x_in_out, conv_state_in_out)

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

        torch.testing.assert_close(outputs[0],
                                   out_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
        torch.testing.assert_close(outputs[1],
                                   conv_state_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
