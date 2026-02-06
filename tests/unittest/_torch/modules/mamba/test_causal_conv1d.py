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

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID


def mamba_conv1d_ref(x, past_conv_state, conv_weight, conv_bias, apply_silu):
    """
    Reference implementation for causal conv1d.

    Arguments:
        x: [batch_size, dim, seq_len]
        past_conv_state: [batch_size, dim, dconv-1]
        conv_weight: [dim, 1, dconv]
        conv_bias: [dim]
    Output:
        y: [batch_size, dim, seq_len]
        present_conv_state: [batch_size, dim, dconv-1]
    """
    assert x.dim() == 3
    assert past_conv_state.dim() == 3
    assert conv_weight.dim() == 3
    assert conv_bias.dim() == 1
    batch_size, dim, seq_len = x.shape
    assert past_conv_state.shape[0] == batch_size
    assert past_conv_state.shape[1] == dim
    dconv = past_conv_state.shape[2] + 1
    assert conv_weight.shape[0] == dim
    assert conv_weight.shape[1] == 1
    assert conv_weight.shape[2] == dconv
    assert conv_weight.shape[0] == dim

    padded_x = torch.cat([past_conv_state, x], dim=2)
    present_conv_state = padded_x[:, :, -(dconv - 1) :]
    x_conv = F.conv1d(padded_x, conv_weight, bias=conv_bias, groups=dim)

    y = F.silu(x_conv) if apply_silu else x_conv
    return y, present_conv_state


def trtllm_causal_conv1d_available():
    """Check if trtllm.causal_conv1d_fwd is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "causal_conv1d_fwd")


skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or not trtllm_causal_conv1d_available(),
    reason="Requires CUDA and trtllm.causal_conv1d_fwd op",
)


@skip_unsupported
class TestCausalConv1d:
    """Tests for the causal_conv1d CUDA kernel."""

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
    @pytest.mark.parametrize("apply_silu", [True, False])
    @pytest.mark.parametrize("dim", [256, 512, 1024, 2048])
    def test_basic_correctness(self, dtype, apply_silu, dim):
        """Test basic correctness against reference implementation."""
        torch.manual_seed(42)
        device = "cuda"
        torch_dtype = getattr(torch, dtype)

        batch_size = 4
        seq_len = 32
        dconv = 4
        std_dev = 0.5
        x = torch.randn(batch_size, dim, seq_len, dtype=torch_dtype, device=device)
        x = x * std_dev
        conv_state = torch.zeros(batch_size, dim, dconv - 1, dtype=torch_dtype, device=device)
        conv_weight = torch.randn(dim, 1, dconv, dtype=torch_dtype, device=device)
        conv_bias = torch.randn(dim, dtype=torch_dtype, device=device)
        x_kernel = x.clone()
        conv_state_kernel = conv_state.clone()

        conv_weight_input = conv_weight.squeeze(1).contiguous()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_kernel,
            conv_weight_input,
            conv_bias,
            conv_state_kernel,
            None,  # query_start_loc
            None,  # cache_indices
            None,  # has_initial_state
            apply_silu,
            PAD_SLOT_ID,
        )
        out_ref, conv_state_ref = mamba_conv1d_ref(
            x, conv_state, conv_weight, conv_bias, apply_silu
        )

        torch.testing.assert_close(x_kernel, out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(conv_state_kernel, conv_state_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_various_batch_sizes(self, batch_size):
        """Test with various batch sizes."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.bfloat16
        dim = 1024
        seq_len = 64
        dconv = 4
        apply_silu = True

        x = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device) * 0.5
        conv_state = torch.zeros(batch_size, dim, dconv - 1, dtype=dtype, device=device)
        conv_weight = torch.randn(dim, 1, dconv, dtype=dtype, device=device)
        conv_bias = torch.randn(dim, dtype=dtype, device=device)
        x_kernel = x.clone()
        conv_state_kernel = conv_state.clone()

        conv_weight_input = conv_weight.squeeze(1).contiguous()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_kernel,
            conv_weight_input,
            conv_bias,
            conv_state_kernel,
            None,
            None,
            None,
            apply_silu,
            PAD_SLOT_ID,
        )
        out_ref, conv_state_ref = mamba_conv1d_ref(
            x, conv_state, conv_weight, conv_bias, apply_silu
        )

        torch.testing.assert_close(x_kernel, out_ref, rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(conv_state_kernel, conv_state_ref, rtol=1e-2, atol=1e-1)

    @pytest.mark.parametrize("dconv", [2, 3, 4])
    def test_various_kernel_widths(self, dconv):
        """Test with different convolution kernel widths."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.bfloat16

        batch_size = 4
        dim = 1024
        seq_len = 64
        apply_silu = True
        x = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device) * 0.5
        conv_state = torch.zeros(batch_size, dim, dconv - 1, dtype=dtype, device=device)
        conv_weight = torch.randn(dim, 1, dconv, dtype=dtype, device=device)
        conv_bias = torch.randn(dim, dtype=dtype, device=device)
        x_kernel = x.clone()
        conv_state_kernel = conv_state.clone()

        conv_weight_input = conv_weight.squeeze(1).contiguous()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_kernel,
            conv_weight_input,
            conv_bias,
            conv_state_kernel,
            None,
            None,
            None,
            apply_silu,
            PAD_SLOT_ID,
        )
        out_ref, conv_state_ref = mamba_conv1d_ref(
            x, conv_state, conv_weight, conv_bias, apply_silu
        )

        torch.testing.assert_close(x_kernel, out_ref, rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(conv_state_kernel, conv_state_ref, rtol=1e-2, atol=1e-1)

    def test_with_initial_state(self):
        """Test with non-zero initial conv state."""
        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.bfloat16

        batch_size = 4
        dim = 1024
        seq_len = 32
        dconv = 4
        apply_silu = True

        x = torch.randn(batch_size, dim, seq_len, dtype=dtype, device=device) * 0.5
        # Non-zero initial state
        conv_state = torch.randn(batch_size, dim, dconv - 1, dtype=dtype, device=device)
        conv_state = conv_state * 0.5
        conv_weight = torch.randn(dim, 1, dconv, dtype=dtype, device=device)
        conv_bias = torch.randn(dim, dtype=dtype, device=device)
        conv_state_kernel = conv_state.clone()
        # Need to tell the kernel about initial state
        has_initial_state = torch.ones(batch_size, dtype=torch.bool, device=device)
        query_start_loc = torch.tensor(
            [0] + [seq_len * (i + 1) for i in range(batch_size)],
            dtype=torch.int32,
            device=device,
        )
        # Reshape for varlen format
        x_varlen = x.transpose(1, 2).reshape(-1, dim).T.contiguous()

        conv_weight_input = conv_weight.squeeze(1).contiguous()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_varlen,
            conv_weight_input,
            conv_bias,
            conv_state_kernel,
            query_start_loc,
            None,  # cache_indices
            has_initial_state,
            apply_silu,
            PAD_SLOT_ID,
        )

        out_ref_list = []
        conv_state_ref_list = []
        for b in range(batch_size):
            out_b, state_b = mamba_conv1d_ref(
                x[b : b + 1],
                conv_state[b : b + 1],
                conv_weight,
                conv_bias,
                apply_silu,
            )
            out_ref_list.append(out_b)
            conv_state_ref_list.append(state_b)
        out_ref = torch.cat(out_ref_list, dim=0)
        conv_state_ref = torch.cat(conv_state_ref_list, dim=0)
        x_kernel_reshaped = (
            x_varlen.T.reshape(batch_size, seq_len, dim).transpose(1, 2).contiguous()
        )

        torch.testing.assert_close(x_kernel_reshaped, out_ref, rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(conv_state_kernel, conv_state_ref, rtol=1e-2, atol=1e-1)
