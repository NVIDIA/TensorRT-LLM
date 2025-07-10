import unittest
from itertools import product

import pytest
import torch
from parameterized import parameterized
from utils.torch_ref import mamba_conv1d_ref
from utils.util import unittest_name_func

import tensorrt_llm


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        list(
            product([2048], [4], ['context', 'generation'],
                    ['float16', 'float32', 'bfloat16'], [5], [16], [0, 64],
                    [False, True], [False, True])) +
        # long sequence tests to cover the int overflow issue
        list(
            product([5376], [4], ['context'], ['float16', 'bfloat16'], [2],
                    [131072], [10240], [False, True], [False, True])),
        name_func=unittest_name_func)
    @pytest.mark.high_cuda_memory
    def test_mamba_conv1d(self, dim, dconv, req_type, dtype, batch_size,
                          max_seq_len, stride_size, remove_padding, apply_silu):
        if max_seq_len == 131072:
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if total_gpu_mem <= 33 * 1024**3:
                pytest.skip(
                    "The long sequence test needs at least 33GB memory, skipping"
                )

        device = "cuda"
        seq_len = max_seq_len if req_type == "context" else 1
        with_stride = stride_size > 0
        pre_stride = stride_size
        post_stride = 64 if with_stride else 0
        mean = 0.0
        std_dev = 1.0 if dtype == "float32" else 0.5
        torch_dtype = getattr(torch, dtype)

        # test data
        last_token_ids_input = None
        torch.random.manual_seed(0)
        if remove_padding and req_type == "context":
            last_token_ids = torch.randint(1,
                                           seq_len + 1, (batch_size, ),
                                           dtype=torch.int32)
            last_token_ids[0] = seq_len
            host_context_lengths = last_token_ids.detach().clone().cpu()
            last_token_ids_input = torch.cumsum(last_token_ids,
                                                dim=0,
                                                dtype=torch.int32).to(device)
        else:
            last_token_ids = (torch.ones(
                (batch_size, ), dtype=torch.int32, device=device) * seq_len)
            host_context_lengths = last_token_ids.detach().clone().cpu()
            last_token_ids_input = last_token_ids

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

        host_request_types = torch.tensor([0 if req_type == "context" else 1] *
                                          batch_size,
                                          dtype=torch.int32)
        x = torch.empty(batch_size,
                        dim,
                        seq_len,
                        device=device,
                        dtype=torch_dtype)
        x.normal_(mean, std_dev)

        x_input = x.detach().permute(0, 2, 1).contiguous()
        if remove_padding and req_type == "context":
            x_batches = []
            for b in range(batch_size):
                x_batches.append(x_input[b, :last_token_ids[b], :])
            x_input = torch.cat(x_batches, dim=0)

        if with_stride:
            base_shape = [
                x_input.shape[i] for i in range(len(x_input.shape) - 1)
            ]
            pad_pre_shape = base_shape + [pre_stride]
            pad_post_shape = base_shape + [post_stride]
            pad_pre = torch.randn(pad_pre_shape,
                                  device=device,
                                  dtype=torch_dtype)
            pad_post = torch.randn(pad_post_shape,
                                   device=device,
                                   dtype=torch_dtype)
            x_input = torch.cat([pad_pre, x_input, pad_post],
                                dim=-1).contiguous()

        conv_state_input = conv_state.permute(0, 2, 1).contiguous()
        conv_weight_input = conv_weight.permute(1, 2, 0).contiguous()

        is_paged_state = False
        slot_mapping = None

        if remove_padding and req_type == "generation":
            x_input = x_input.squeeze(1)

        outputs = torch.ops.trtllm.mamba_conv1d(
            x_input,
            conv_weight_input,
            conv_bias,
            conv_state_input,
            host_request_types,
            last_token_ids_input,
            host_context_lengths,
            slot_mapping,
            dim,
            dconv,
            pre_stride,
            post_stride,
            remove_padding,
            apply_silu,
            is_paged_state,
        )

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

        conv_state_ref = conv_state_ref.permute(0, 2, 1).contiguous()
        out_ref = out_ref.permute(0, 2, 1).contiguous()

        if remove_padding and req_type == "context":
            out_ref_batches = []
            for b in range(batch_size):
                out_ref_batches.append(out_ref[b, :host_context_lengths[b], :])
            out_ref = torch.cat(out_ref_batches, dim=0)

        if remove_padding and req_type == "generation":
            out_ref = out_ref.squeeze(1)

        atol = {"float16": 1e-2, "float32": 2e-3, "bfloat16": 1e-1}

        torch.testing.assert_close(outputs[0],
                                   out_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
        torch.testing.assert_close(outputs[1],
                                   conv_state_ref,
                                   rtol=1e-2,
                                   atol=atol[dtype])
