# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Quality test: fused RMSNorm+NVFP4-quantize (modules/fused_ops/
rmsnorm_fp4_quant, backed by flashinfer's CuTe-DSL ``rmsnorm_fp4quant``) vs
the unfused reference chain
(trtllm::flashinfer_rmsnorm -> trtllm::fp4_quantize).

The fused kernel quantizes the fp32 norm result directly, without the
intermediate bf16 round the unfused chain performs when materializing the
normed tensor, so its outputs are near- but not byte-identical (~1.4% of
payload nibbles / ~2% of block scales differ by one code step at serving
shapes). The tests therefore compare dequantized values, not bytes:

- the fused path's quantization error against the fp32 norm must match the
  unfused chain's (both are roundings of the same quantity);
- dequantization reads the scale factors at the swizzled offsets of
  get_sf_out_offset_128x4, so any layout or scale-convention bug shows up
  as a gross error, not a tolerance miss;
- a small byte-mismatch bound documents the near-parity property itself.
"""

import importlib
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_ops.gelu_tanh_mul_fp4_quant import sf_swizzled_offsets
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import (
    add_rmsnorm_fp4_quant,
    add_rmsnorm_fp4_quant_available,
    rmsnorm_fp4_quant,
    rmsnorm_fp4_quant_available,
)

pytestmark = pytest.mark.skipif(
    not rmsnorm_fp4_quant_available(),
    reason="flashinfer CuTe-DSL rmsnorm_fp4quant unavailable",
)

_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_LUT = _E2M1_LUT + [-v for v in _E2M1_LUT]
_fusion_module = importlib.import_module("tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant")


def _reference(x, w, eps, gs):
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, eps)
    q, sf = torch.ops.trtllm.fp4_quantize(n, gs, 16, False)
    return q.view(torch.uint8), sf.view(torch.uint8)


def _norm_fp32(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _block_scales(sf, m, h):
    """Block scales [m, h//16] read at the swizzled offsets (as fp32)."""
    valid = sf_swizzled_offsets(m, h // 16, sf.device)
    return sf.view(-1)[valid].view(torch.float8_e4m3fn).float().view(m, h // 16)


def _dequant(q, sf, m, h, gs):
    """Dequantize payload + swizzled scales to fp32 [m, h]."""
    lut = torch.tensor(_E2M1_LUT, device=q.device)
    vals = torch.empty((m, h), device=q.device, dtype=torch.float32)
    vals[:, 0::2] = lut[(q & 0xF).long()]
    vals[:, 1::2] = lut[(q >> 4).long()]
    return vals * _block_scales(sf, m, h).repeat_interleave(16, dim=1) / gs


def _check(x, w, eps, gs):
    fq, fsf = rmsnorm_fp4_quant(x, w, eps, gs)
    rq, rsf = _reference(x.contiguous(), w, eps, gs)
    assert fq.shape == rq.shape and fsf.numel() == rsf.numel()
    assert fq.dtype == torch.uint8 and fsf.dtype == torch.uint8

    m, h = x.shape
    n32 = _norm_fp32(x.contiguous(), w, eps)
    err_fused = (_dequant(fq, fsf, m, h, gs) - n32).abs().mean().item()
    err_ref = (_dequant(rq, rsf, m, h, gs) - n32).abs().mean().item()
    # Both outputs are NVFP4 roundings of the same fp32 norm; the fused
    # path must not be a worse quantizer than the unfused chain (measured
    # equal to ~0.1% at serving shapes; the slack covers rounding-path
    # differences on small inputs).
    assert err_fused <= err_ref * 1.05 + 1e-6, f"{err_fused=} vs {err_ref=}"
    # Near-parity: the two paths differ only where a rounding boundary is
    # crossed (~1.4% of bytes at serving shapes). Blocks whose e4m3 scale
    # underflows to zero are excluded: their payload never contributes to
    # the dequantized value and the implementations fill it differently.
    live = (_block_scales(rsf, m, h) != 0) & (_block_scales(fsf, m, h) != 0)
    live_bytes = live.repeat_interleave(8, dim=1)  # 8 payload bytes / block
    if live_bytes.any():
        mm = (fq != rq)[live_bytes].float().mean().item()
        assert mm < 5e-2, f"payload byte mismatch fraction {mm}"


# 5376 is the Gemma4-31B hidden size; 228 the pinned decode batch; 7700 the
# profiled serving prefill size; 333 exercises padded tail rows.
@pytest.mark.parametrize("n_tokens", [1, 7, 228, 333, 7700])
@pytest.mark.parametrize("hidden", [5376, 512])
def test_rmsnorm_fp4_quant_quality(n_tokens, hidden):
    torch.manual_seed(1234)
    x = (
        torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        * torch.rand((n_tokens, 1), dtype=torch.bfloat16, device="cuda")
        * 3
    )
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6)
    gs = (448.0 * 6.0 / n.abs().max().float()).reshape(1)
    _check(x, w, 1e-6, gs)


@pytest.mark.parametrize("gs_value", [1e6, 1.0, 1e-6])
def test_rmsnorm_fp4_quant_extreme_scales(gs_value):
    """Saturating / degenerate global scales must behave like the unfused op."""
    torch.manual_seed(7)
    x = torch.randn((333, 5376), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([gs_value], dtype=torch.float32, device="cuda")
    _check(x, w, 1e-6, gs)


def test_rmsnorm_fp4_quant_strided_rows():
    """Row-strided input (a view of a wider buffer)."""
    torch.manual_seed(3)
    n, h = 65, 5376
    buf = torch.randn((n, h + 512), dtype=torch.bfloat16, device="cuda")
    x = buf[:, :h]
    w = torch.rand((h,), dtype=torch.bfloat16, device="cuda") + 0.5
    normed = torch.ops.trtllm.flashinfer_rmsnorm(x.contiguous(), w, 1e-6)
    gs = (448.0 * 6.0 / normed.abs().max().float()).reshape(1)
    _check(x, w, 1e-6, gs)


def test_rmsnorm_fp4_quant_zero_row():
    """An all-zero row must dequantize to exactly zero (zero block scales)."""
    torch.manual_seed(5)
    x = torch.randn((9, 5376), dtype=torch.bfloat16, device="cuda")
    x[4] = 0
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([100.0], dtype=torch.float32, device="cuda")
    fq, fsf = rmsnorm_fp4_quant(x, w, 1e-6, gs)
    dq = _dequant(fq, fsf, 9, 5376, gs)
    assert dq[4].abs().max().item() == 0.0
    _check(x, w, 1e-6, gs)


def test_rmsnorm_fp4_quant_cuda_graph():
    """The kernel must be CUDA-graph capturable and replay deterministically
    (the serving decode path replays it inside captured graphs)."""
    torch.manual_seed(11)
    x = torch.randn((228, 5376), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([100.0], dtype=torch.float32, device="cuda")

    eager_q, eager_sf = rmsnorm_fp4_quant(x, w, 1e-6, gs)  # also JIT warmup
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_q, graph_sf = rmsnorm_fp4_quant(x, w, 1e-6, gs)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_q, eager_q)
    valid = sf_swizzled_offsets(228, 5376 // 16, x.device)
    assert torch.equal(graph_sf.view(-1)[valid], eager_sf.view(-1)[valid])


@pytest.mark.skipif(
    not add_rmsnorm_fp4_quant_available(),
    reason="flashinfer CuTe-DSL add_rmsnorm_fp4quant unavailable",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 228])
@pytest.mark.parametrize("padded_hidden_size", [2880, 3072])
def test_add_rmsnorm_fp4_quant_gpt_oss(num_tokens, padded_hidden_size):
    """GPT-OSS keeps a logical 2880-wide norm and pads only packed outputs."""
    from tensorrt_llm._torch.custom_ops import flashinfer_fused_add_rmsnorm

    torch.manual_seed(19)
    hidden_size = 2880
    eps = 1e-5
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.rand((hidden_size,), dtype=torch.bfloat16, device="cuda") + 0.5
    global_scale = torch.tensor([100.0], dtype=torch.float32, device="cuda")

    x_ref = x.clone()
    residual_ref = residual.clone()
    flashinfer_fused_add_rmsnorm(x_ref, residual_ref, weight, eps)
    padded_ref = F.pad(x_ref, (0, padded_hidden_size - hidden_size))
    fp4_ref, sf_ref = torch.ops.trtllm.fp4_quantize(padded_ref, global_scale, 16, False, False)
    sf_ref = sf_ref.view(num_tokens, -1)

    fp4, sf, norm_out = add_rmsnorm_fp4_quant(
        x,
        residual,
        weight,
        eps,
        global_scale,
        padded_hidden_size,
    )

    assert fp4.shape == fp4_ref.shape
    assert sf.shape == sf_ref.shape
    assert fp4.dtype == torch.uint8
    assert sf.dtype == torch.uint8
    torch.testing.assert_close(residual, residual_ref, rtol=0, atol=0)
    torch.testing.assert_close(norm_out, x_ref, rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(fp4.view(torch.uint8)[:, hidden_size // 2 :]) == 0
    assert torch.count_nonzero(sf.view(torch.uint8)[:, hidden_size // 16 :]) == 0

    fused_dequant = _dequant_linear(fp4, sf, num_tokens, padded_hidden_size, global_scale)
    reference_dequant = _dequant_linear(
        fp4_ref, sf_ref, num_tokens, padded_hidden_size, global_scale
    )
    target = padded_ref.float()
    fused_error = (fused_dequant - target).abs().mean().item()
    reference_error = (reference_dequant - target).abs().mean().item()
    assert fused_error <= reference_error * 1.05 + 1e-6


def _dequant_linear(q, sf, m, h, gs):
    """Dequantize an NVFP4 tensor with row-major E4M3 block scales."""
    q = q.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, device=q.device)
    values = torch.empty((m, h), device=q.device, dtype=torch.float32)
    values[:, 0::2] = lut[(q & 0xF).long()]
    values[:, 1::2] = lut[(q >> 4).long()]
    scales = sf.view(torch.float8_e4m3fn).float().view(m, h // 16)
    return values * scales.repeat_interleave(16, dim=1) / gs


@pytest.mark.skipif(
    not add_rmsnorm_fp4_quant_available(),
    reason="flashinfer CuTe-DSL add_rmsnorm_fp4quant unavailable",
)
def test_add_rmsnorm_fp4_quant_cuda_graph():
    torch.manual_seed(23)
    x = torch.randn((32, 2880), dtype=torch.bfloat16, device="cuda")
    residual_source = torch.randn_like(x)
    weight = torch.rand((2880,), dtype=torch.bfloat16, device="cuda") + 0.5
    global_scale = torch.tensor([100.0], dtype=torch.float32, device="cuda")

    add_rmsnorm_fp4_quant(x, residual_source.clone(), weight, 1e-5, global_scale, 3072)
    expected_residual = residual_source.clone()
    expected = add_rmsnorm_fp4_quant(x, expected_residual, weight, 1e-5, global_scale, 3072)
    torch.cuda.synchronize()

    graph_residual = residual_source.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_residual.copy_(residual_source)
        actual = add_rmsnorm_fp4_quant(x, graph_residual, weight, 1e-5, global_scale, 3072)
    assert actual[0].dtype == torch.uint8
    assert actual[1].dtype == torch.uint8
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_residual, expected_residual)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.cpu_only
@pytest.mark.parametrize("supports_padding", [False, True])
@pytest.mark.parametrize("padded_hidden_size", [2880, 3072])
def test_add_rmsnorm_fp4_quant_padding_dispatch(
    monkeypatch: pytest.MonkeyPatch, supports_padding: bool, padded_hidden_size: int
) -> None:
    x = torch.zeros((2, 2880), dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    weight = torch.ones((2880,), dtype=x.dtype)
    scale = torch.ones(1, dtype=torch.float32)
    direct = supports_padding and padded_hidden_size != x.shape[1]
    output_width = padded_hidden_size if direct else x.shape[1]
    fp4 = torch.full((2, output_width // 2), 0x21, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    sf = torch.full((2, output_width // 16), 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn)
    fused = Mock(return_value=(fp4, sf))
    pad = Mock(return_value=(fp4, sf))
    monkeypatch.setattr(_fusion_module, "_add_rmsnorm_fp4quant", fused)
    monkeypatch.setattr(_fusion_module, "_ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING", supports_padding)
    monkeypatch.setattr(
        _fusion_module, "signature", Mock(side_effect=AssertionError("capability must be cached"))
    )
    monkeypatch.setattr(torch.ops.trtllm, "pad_nvfp4_activation", pad)

    actual = add_rmsnorm_fp4_quant(x, residual, weight, 1e-5, scale, padded_hidden_size)

    fused.assert_called_once()
    if direct:
        assert fused.call_args.kwargs["output_hidden_size"] == padded_hidden_size
    else:
        assert "output_hidden_size" not in fused.call_args.kwargs
    if padded_hidden_size != x.shape[1] and not direct:
        pad.assert_called_once_with(fp4, sf, padded_hidden_size)
    else:
        pad.assert_not_called()
    # No numeric cast or copy at the MoE boundary, and no widening of the router input.
    for output, source in zip(actual[:2], (fp4, sf)):
        assert output.dtype == torch.uint8
        assert output.data_ptr() == source.data_ptr()
        assert torch.equal(output, source.view(torch.uint8))
    assert actual[2] is fused.call_args.kwargs["y_out"]
    assert actual[2].shape == x.shape


@pytest.mark.cpu_only
def test_add_rmsnorm_fp4_quant_does_not_retry_after_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.ones((1, 16), dtype=torch.bfloat16)
    residual = torch.zeros_like(x)

    def fail_after_add(*args: object, **kwargs: object) -> None:
        residual.add_(x)
        raise RuntimeError("fused kernel failed")

    fused = Mock(side_effect=fail_after_add)
    pad = Mock()
    monkeypatch.setattr(_fusion_module, "_add_rmsnorm_fp4quant", fused)
    monkeypatch.setattr(_fusion_module, "_ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING", True)
    monkeypatch.setattr(torch.ops.trtllm, "pad_nvfp4_activation", pad)

    with pytest.raises(RuntimeError, match="fused kernel failed"):
        add_rmsnorm_fp4_quant(x, residual, x[0], 1e-5, torch.ones(1), 32)
    fused.assert_called_once()
    pad.assert_not_called()
    assert torch.equal(residual, x)


@pytest.mark.skipif(
    not _fusion_module._ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING,
    reason="FlashInfer direct padded output unavailable",
)
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 228, 7700])
@pytest.mark.parametrize("hidden_size,padded_hidden_size", [(2880, 2880), (2880, 3072), (512, 544)])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_add_rmsnorm_fp4_quant_direct_padding_matches_legacy(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    hidden_size: int,
    padded_hidden_size: int,
    use_cuda_graph: bool,
) -> None:
    """Padding must not alter any byte from the corrected compact-output kernel."""
    torch.manual_seed(31)
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    source = torch.randn_like(x)
    weight = torch.rand((hidden_size,), dtype=x.dtype, device=x.device) + 0.5
    scale = torch.tensor([100.0], dtype=torch.float32, device=x.device)
    legacy_residual = source.clone()
    monkeypatch.setattr(_fusion_module, "_ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING", False)
    expected = add_rmsnorm_fp4_quant(x, legacy_residual, weight, 1e-5, scale, padded_hidden_size)
    monkeypatch.setattr(_fusion_module, "_ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING", True)
    pad = Mock(side_effect=AssertionError("direct output must not launch separate padding"))
    monkeypatch.setattr(torch.ops.trtllm, "pad_nvfp4_activation", pad)
    residual = source.clone()

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual.copy_(source)
        return add_rmsnorm_fp4_quant(x, residual, weight, 1e-5, scale, padded_hidden_size)

    actual = run()
    torch.cuda.synchronize()
    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
    for _ in range(3):
        if graph is not None:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(residual, legacy_residual)
        for output, reference in zip(actual, expected):
            assert torch.equal(output, reference)
        assert actual[0].dtype == actual[1].dtype == torch.uint8
        assert actual[0].shape == (num_tokens, padded_hidden_size // 2)
        assert actual[1].shape == (num_tokens, padded_hidden_size // 16)
        assert actual[2].shape == x.shape
        assert torch.count_nonzero(actual[0][:, hidden_size // 2 :]) == 0
        assert torch.count_nonzero(actual[1][:, hidden_size // 16 :]) == 0
    pad.assert_not_called()


if __name__ == "__main__":
    test_rmsnorm_fp4_quant_quality(7700, 5376)
    test_rmsnorm_fp4_quant_quality(228, 5376)
    test_rmsnorm_fp4_quant_extreme_scales(1e6)
    test_rmsnorm_fp4_quant_strided_rows()
    test_rmsnorm_fp4_quant_zero_row()
    test_rmsnorm_fp4_quant_cuda_graph()
    print("ALL QUALITY CHECKS PASSED")
