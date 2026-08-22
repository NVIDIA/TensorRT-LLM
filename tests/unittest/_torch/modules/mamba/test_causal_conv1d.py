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


@skip_unsupported
class TestCausalConv1dChannelLast:
    """Tests for the channel-last (token-major) causal_conv1d forward kernel.

    The existing channel-major kernel is the reference: both are dispatched from the same op,
    so any difference is the channel-last kernel's own.
    """

    @staticmethod
    def _case(seed, total_tokens, dim, dconv, seq_lens, dtype, row_pitch=None, pad_first=False):
        device = "cuda"
        g = torch.Generator(device=device).manual_seed(seed)
        pitch = row_pitch if row_pitch is not None else dim
        # A wider row pitch models a column slice of a fused projection.
        base = (torch.randn(total_tokens, pitch, generator=g, device=device) * 0.5).to(dtype)
        x_tok_major = base[:, :dim]

        weight = (torch.randn(dim, dconv, generator=g, device=device) * 0.3).to(dtype)
        bias = (torch.randn(dim, generator=g, device=device) * 0.1).to(dtype)
        query_start_loc = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32, device=device
        )
        batch = len(seq_lens)
        num_cache_lines = batch + 2
        cache_indices = torch.arange(batch, dtype=torch.int32, device=device) + 2
        if pad_first:
            cache_indices[0] = PAD_SLOT_ID
        has_initial_state = torch.zeros(batch, dtype=torch.bool, device=device)
        has_initial_state[::2] = True
        conv_states = (
            torch.randn(num_cache_lines, dim, dconv - 1, generator=g, device=device) * 0.5
        ).to(dtype)
        return (
            x_tok_major,
            weight,
            bias,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_states,
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("apply_silu", [True, False])
    @pytest.mark.parametrize("dconv", [2, 3, 4])
    @pytest.mark.parametrize(
        "dim,seq_lens,row_pitch",
        [
            (2048, [512], None),  # single-sequence prefill, 16B-aligned fast path
            (2048, [200, 1, 3, 308], None),  # ragged varlen
            (2048, [1, 2, 1, 3], None),  # sequences shorter than dconv - 1
            (2054, [512], None),  # dim not a multiple of the vector width -> scalar path
            (2048, [512], 5120),  # channel slice of a wider projection
            (2048, [512], 5121),  # unaligned row pitch -> scalar path
        ],
    )
    def test_matches_channel_major(self, dtype, apply_silu, dconv, dim, seq_lens, row_pitch):
        """Channel-last output and conv_states must match the channel-major kernel."""
        x_tok_major, weight, bias, qsl, cache_idx, has_init, conv_states = self._case(
            seed=1234,
            total_tokens=sum(seq_lens),
            dim=dim,
            dconv=dconv,
            seq_lens=seq_lens,
            dtype=dtype,
            row_pitch=row_pitch,
        )

        # Reference: materialise the channel-major copy the old path required.
        x_ref = x_tok_major.t().contiguous()
        cs_ref = conv_states.clone()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_ref, weight, bias, cs_ref, qsl, cache_idx, has_init, apply_silu, PAD_SLOT_ID
        )

        # Channel-last: pass a transposed view, write into a token-major buffer.
        out_tok_major = x_tok_major.contiguous().clone()
        cs_new = conv_states.clone()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_tok_major.t(),
            weight,
            bias,
            cs_new,
            qsl,
            cache_idx,
            has_init,
            apply_silu,
            PAD_SLOT_ID,
            out_tok_major.t(),
        )

        torch.testing.assert_close(out_tok_major, x_ref.t(), rtol=1e-2, atol=1e-2)
        # conv_states are copied verbatim out of x, so they must be exact.
        torch.testing.assert_close(cs_new, cs_ref, rtol=0, atol=0)

    def test_pad_slot_is_skipped(self):
        """Sequences mapped to PAD_SLOT_ID must leave out and conv_states untouched."""
        seq_lens = [200, 312]
        x_tok_major, weight, bias, qsl, cache_idx, has_init, conv_states = self._case(
            seed=7,
            total_tokens=sum(seq_lens),
            dim=1024,
            dconv=4,
            seq_lens=seq_lens,
            dtype=torch.bfloat16,
            pad_first=True,
        )
        sentinel = torch.full_like(x_tok_major, 3.0)
        out = sentinel.contiguous().clone()
        cs_new = conv_states.clone()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_tok_major.t(),
            weight,
            bias,
            cs_new,
            qsl,
            cache_idx,
            has_init,
            True,
            PAD_SLOT_ID,
            out.t(),
        )
        torch.testing.assert_close(out[: seq_lens[0]], sentinel[: seq_lens[0]], rtol=0, atol=0)

        # Only the padded sequence is skipped. Sequence 1 is real, so its own
        # cache line must be rewritten -- asserting the whole tensor is
        # unchanged would also pass if the kernel did nothing at all.
        live = cache_idx[1].item()
        untouched = [i for i in range(conv_states.shape[0]) if i != live]
        torch.testing.assert_close(cs_new[untouched], conv_states[untouched], rtol=0, atol=0)
        assert not torch.equal(cs_new[live], conv_states[live])

    def test_rejects_inplace_and_mismatched_out_layout(self):
        """The chunked channel-last kernel reads a halo, so it cannot alias out onto x."""
        seq_lens = [512]
        x_tok_major, weight, bias, qsl, cache_idx, has_init, conv_states = self._case(
            seed=3,
            total_tokens=sum(seq_lens),
            dim=1024,
            dconv=4,
            seq_lens=seq_lens,
            dtype=torch.bfloat16,
        )
        x_cl = x_tok_major.t()
        with pytest.raises(RuntimeError, match="cannot run in-place"):
            torch.ops.trtllm.causal_conv1d_fwd(
                x_cl,
                weight,
                bias,
                conv_states.clone(),
                qsl,
                cache_idx,
                has_init,
                True,
                PAD_SLOT_ID,
                x_cl,
            )
        channel_major_out = torch.empty_like(x_cl.contiguous())
        with pytest.raises(RuntimeError, match="channel-last out"):
            torch.ops.trtllm.causal_conv1d_fwd(
                x_cl,
                weight,
                bias,
                conv_states.clone(),
                qsl,
                cache_idx,
                has_init,
                True,
                PAD_SLOT_ID,
                channel_major_out,
            )

    def test_python_wrapper_allocates_channel_last_out(self):
        """causal_conv1d_fn must return a token-major result for a token-major input."""
        from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn

        seq_lens = [512]
        x_tok_major, weight, bias, qsl, cache_idx, has_init, conv_states = self._case(
            seed=11,
            total_tokens=sum(seq_lens),
            dim=1024,
            dconv=4,
            seq_lens=seq_lens,
            dtype=torch.bfloat16,
        )
        out = causal_conv1d_fn(
            x_tok_major.t(),
            weight,
            bias,
            query_start_loc=qsl,
            cache_indices=cache_idx,
            has_initial_state=has_init,
            conv_states=conv_states.clone(),
            activation="silu",
            pad_slot_id=PAD_SLOT_ID,
        )
        assert out.stride(0) == 1 and out.stride(1) == x_tok_major.shape[1]
        assert out.data_ptr() != x_tok_major.data_ptr()

        x_ref = x_tok_major.t().contiguous()
        torch.ops.trtllm.causal_conv1d_fwd(
            x_ref,
            weight,
            bias,
            conv_states.clone(),
            qsl,
            cache_idx,
            has_init,
            True,
            PAD_SLOT_ID,
        )
        torch.testing.assert_close(out, x_ref, rtol=1e-2, atol=1e-2)
