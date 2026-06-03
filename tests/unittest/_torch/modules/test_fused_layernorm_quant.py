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
"""Unit tests for fused LayerNorm + (optional AdaLN modulation) + NVFP4 quant.

The kernel under test is `torch.ops.trtllm.fused_layernorm_quantize`, used by
the Wan 2.2 transformer block's norm1/norm2/norm3 in the static NVFP4 fast
path. We exercise three configurations of the same kernel:

  - plain LN (no affine, no modulation)        -> all four optional args None
  - LN with learned affine (norm2 in Wan)      -> ln_weight + ln_bias provided
  - AdaLN modulation (norm1, norm3 in Wan)     -> scale_msa + shift_msa provided

For each case we compare the kernel against a high-precision PyTorch reference
fed into the separate `fp4_quantize` op. A small fraction of values land exactly
on a quantization boundary and can flip between fused and separate paths due to
FMA-vs-mul-add ordering -- we use the same >=99% match-rate tolerance as the
existing relu2/gelu_tanh tests.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.unittest.utils.util import getSMVersion


def fused_layernorm_quantize_available():
    """Return True iff the ``trtllm::fused_layernorm_quantize`` op is registered."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_layernorm_quantize")


def fp4_quantize_available():
    """Return True iff the reference ``trtllm::fp4_quantize`` op is registered."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fp4_quantize")


skip_unless_fused_layernorm_and_fp4 = pytest.mark.skipif(
    getSMVersion() < 100
    or not fused_layernorm_quantize_available()
    or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm fused_layernorm_quantize + fp4_quantize ops",
)


def _quantize_reference(
    x: torch.Tensor,
    ln_weight,
    ln_bias,
    scale_msa,
    shift_msa,
    seq_len_per_batch: int,
    eps: float,
    sf_vec_size: int,
):
    """Compute the high-precision PyTorch reference for the fused kernel.

    Runs PyTorch LN -> optional affine -> optional modulation -> cast to input
    dtype -> separate ``fp4_quantize``. Returns ``(fp4, sf, sf_scale)``.
    """
    m, n = x.shape
    x_fp32 = x.float()

    if ln_weight is not None:
        normed = F.layer_norm(
            x_fp32,
            (n,),
            weight=ln_weight.float(),
            bias=ln_bias.float(),
            eps=eps,
        )
    else:
        normed = F.layer_norm(x_fp32, (n,), weight=None, bias=None, eps=eps)

    if scale_msa is not None:
        # scale_msa / shift_msa are [B, N]; row r uses batch r / seq_len_per_batch.
        # Match the kernel's behavior of casting modulation to input dtype.
        s = scale_msa.to(x.dtype)
        sh = shift_msa.to(x.dtype)
        b = s.shape[0]
        s_exp = s.unsqueeze(1).expand(b, seq_len_per_batch, n).reshape(m, n)
        sh_exp = sh.unsqueeze(1).expand(b, seq_len_per_batch, n).reshape(m, n)
        # Cast normed to input dtype before applying modulation to mirror the
        # fused kernel, which reads bf16/fp16 scale_msa/shift_msa.
        normed = normed.to(x.dtype).float()
        normed = normed * (1 + s_exp.float()) + sh_exp.float()

    normed_dt = normed.to(x.dtype)

    sf_scale = (normed_dt.abs().amax().float() / (6.0 * 448.0)).view(1).to(x.device)
    fp4_ref, sf_ref = torch.ops.trtllm.fp4_quantize(
        normed_dt,
        sf_scale,
        sf_vec_size,
        False,  # use_ue8m0
        True,  # is_sf_swizzled_layout
    )
    return fp4_ref, sf_ref, sf_scale


@skip_unless_fused_layernorm_and_fp4
@pytest.mark.parametrize("m,n", [(32, 5120), (128, 5120)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_layernorm_quantize_plain_ln(m, n, dtype):
    """Test the LN-only path (no affine, no modulation).

    Mirrors what the kernel must do for a vanilla ``LayerNorm`` followed by
    a downstream NVFP4 ``Linear``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, n, dtype=dtype, device=device) * 0.5

    fp4_ref, sf_ref, sf_scale = _quantize_reference(x, None, None, None, None, 1, 1e-6, 16)

    fp4_fused, sf_fused = torch.ops.trtllm.fused_layernorm_quantize(
        x.contiguous(),
        None,
        None,
        None,
        None,
        sf_scale,
        1,
        1e-6,
        16,
    )

    assert fp4_fused.shape == (m, n // 2)
    assert sf_fused.shape == sf_ref.shape

    match_rate = (fp4_fused == fp4_ref).float().mean().item()
    assert match_rate >= 0.99, (
        f"plain_ln match rate {match_rate:.4f} < 0.99 for ({m}, {n}), {dtype}"
    )


@skip_unless_fused_layernorm_and_fp4
@pytest.mark.parametrize("m,n", [(32, 5120), (128, 5120)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_layernorm_quantize_ln_affine(m, n, dtype):
    """Test the LN-with-affine path (learned weight and bias).

    This is the configuration used by Wan 2.2's ``norm2`` (cross-attention
    input).
    """
    torch.manual_seed(1)
    device = torch.device("cuda")
    x = torch.randn(m, n, dtype=dtype, device=device) * 0.5
    ln_w = torch.randn(n, dtype=dtype, device=device) * 0.1 + 1.0
    ln_b = torch.randn(n, dtype=dtype, device=device) * 0.01

    fp4_ref, sf_ref, sf_scale = _quantize_reference(x, ln_w, ln_b, None, None, 1, 1e-6, 16)

    fp4_fused, sf_fused = torch.ops.trtllm.fused_layernorm_quantize(
        x.contiguous(),
        ln_w.contiguous(),
        ln_b.contiguous(),
        None,
        None,
        sf_scale,
        1,
        1e-6,
        16,
    )

    assert fp4_fused.shape == (m, n // 2)
    assert sf_fused.shape == sf_ref.shape

    match_rate = (fp4_fused == fp4_ref).float().mean().item()
    assert match_rate >= 0.99, (
        f"ln_affine match rate {match_rate:.4f} < 0.99 for ({m}, {n}), {dtype}"
    )


@skip_unless_fused_layernorm_and_fp4
@pytest.mark.parametrize(
    "b,s,n",
    [
        (1, 32, 5120),  # smallest case
        (2, 64, 5120),  # multi-batch broadcast
        (4, 16, 5120),  # exercises non-trivial seq_len_per_batch
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_layernorm_quantize_adaln(b, s, n, dtype):
    """Test the AdaLN modulation path: ``LN(x) * (1 + scale_msa) + shift_msa``.

    ``scale_msa`` / ``shift_msa`` broadcast across each batch's S sequence
    positions. This is the configuration used by Wan 2.2's ``norm1`` and
    ``norm3``.
    """
    torch.manual_seed(2)
    device = torch.device("cuda")
    m = b * s
    x = torch.randn(m, n, dtype=dtype, device=device) * 0.5
    scale_msa = torch.randn(b, n, dtype=dtype, device=device) * 0.1
    shift_msa = torch.randn(b, n, dtype=dtype, device=device) * 0.05

    fp4_ref, sf_ref, sf_scale = _quantize_reference(
        x, None, None, scale_msa, shift_msa, s, 1e-6, 16
    )

    fp4_fused, sf_fused = torch.ops.trtllm.fused_layernorm_quantize(
        x.contiguous(),
        None,
        None,
        scale_msa.contiguous(),
        shift_msa.contiguous(),
        sf_scale,
        s,
        1e-6,
        16,
    )

    assert fp4_fused.shape == (m, n // 2)
    assert sf_fused.shape == sf_ref.shape

    match_rate = (fp4_fused == fp4_ref).float().mean().item()
    assert match_rate >= 0.99, (
        f"adaln match rate {match_rate:.4f} < 0.99 for B={b} S={s} N={n}, {dtype}"
    )


def test_layernorm_falls_back_on_per_token_modulation():
    """Verify per-token timestep modulation falls back to the unfused path.

    Per-token timestep produces ``scale_msa.shape == [B, S, N]`` with S>1.
    The fused kernel only supports one modulation vector per batch (its
    ``batch_idx = row / seq_len_per_batch`` indexing cannot represent S
    distinct modulation vectors per batch), so the Python wrapper must
    detect this case and route to ``_forward_unfused`` rather than raising
    a shape ValueError or invoking the fused kernel with mis-sized inputs.

    Runs on any device because it exercises only the Python dispatch path.
    """
    from tensorrt_llm._torch.modules.layer_norm import LayerNorm
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor

    b, s, n = 2, 3, 5120
    ln = LayerNorm(hidden_size=n, eps=1e-6, quantize_type="nvfp4")
    # Force the fast-path gate open even off Blackwell so the fallback
    # decision is exercised in CI on any GPU/CPU host.
    ln.is_nvfp4 = True
    ln.nvfp4_scale = torch.tensor([1.0])

    x = torch.randn(b, s, n)
    scale_msa = torch.randn(b, s, n) * 0.1
    shift_msa = torch.randn(b, s, n) * 0.05

    out = ln.forward(x, scale_msa=scale_msa, shift_msa=shift_msa, seq_len_per_batch=s)

    assert not isinstance(out, Fp4QuantizedTensor), (
        "per-token modulation must fall back to the FP32 unfused path "
        "instead of taking the fused NVFP4 kernel"
    )
    assert out.shape == x.shape
