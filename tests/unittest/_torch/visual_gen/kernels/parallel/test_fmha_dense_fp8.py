# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm::fmha_dense_fp8: dense FP8 context attention through the trtllm-gen FMHA runner."""

import math

import pytest
import torch
import torch.nn.functional as F
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)
from tensorrt_llm._utils import get_sm_version

pytestmark = skip_pre_blackwell

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
LOG2E = math.log2(math.e)
HEAD_DIM = 128


def _supported(num_heads, out_dtype, fp16_softmax):
    return torch.ops.trtllm.fmha_dense_fp8_is_supported(
        HEAD_DIM, num_heads, out_dtype, fp16_softmax
    )


def _require(num_heads, out_dtype, fp16_softmax):
    if not _supported(num_heads, out_dtype, fp16_softmax):
        pytest.skip(f"no dense FP8 kernel on SM{get_sm_version()} for this configuration")


def _cu_seqlens(batch, seq_len):
    return torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda")


def _reference(q, k, v, batch, seq_len):
    """SDPA in FP32 on [T, H, D] tensors that are already dequantized."""

    def bshd(x):
        return x.float().view(batch, seq_len, -1, HEAD_DIM).transpose(1, 2)

    out = F.scaled_dot_product_attention(bshd(q), bshd(k), bshd(v))
    return out.transpose(1, 2).reshape(batch * seq_len, -1, HEAD_DIM)


def _inputs(batch, seq_len, num_heads, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch * seq_len, num_heads, HEAD_DIM)
    return [torch.randn(shape, device="cuda", generator=g) for _ in range(3)]


def _quant(x, scale):
    return (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _min_cosine(out_dtype):
    """FP8 output adds one e4m3 rounding of the result."""
    return 0.995 if out_dtype == E4M3 else 0.999


@pytest.mark.parametrize("batch,seq_len", [(1, 1560), (2, 1560), (1, 4096)])
@pytest.mark.parametrize("num_heads", [10, 40])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, E4M3])
def test_bf16_softmax_matches_sdpa(batch, seq_len, num_heads, out_dtype):
    """Per-tensor FP8 Q/K/V with the softmax scale applied by the kernel (BF16 softmax)."""
    _require(num_heads, out_dtype, fp16_softmax=False)
    q, k, v = _inputs(batch, seq_len, num_heads)
    qs, ks, vs = (t.abs().amax() / E4M3_MAX for t in (q, k, v))
    q8, k8, v8 = _quant(q, qs), _quant(k, ks), _quant(v, vs)

    softmax_scale = HEAD_DIM**-0.5
    bmm1 = torch.stack([softmax_scale * qs * ks, softmax_scale * qs * ks * LOG2E]).float()
    out_scale = torch.tensor(4.0, device="cuda") if out_dtype == E4M3 else None
    scale_out = (vs / out_scale if out_scale is not None else vs).reshape(1).float()

    out = torch.ops.trtllm.fmha_dense_fp8(
        q8, k8, v8, _cu_seqlens(batch, seq_len), seq_len, bmm1, scale_out, False, out_dtype
    )
    assert out.dtype == out_dtype and out.shape == q8.shape
    result = out.float() * (out_scale if out_scale is not None else 1.0)
    ref = _reference(q8.float() * qs, k8.float() * ks, v8.float() * vs, batch, seq_len)
    assert _cosine(result, ref) > _min_cosine(out_dtype)


@pytest.mark.parametrize("num_heads", [10, 40])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, E4M3])
def test_fp16_softmax_folded_scale_matches_sdpa(num_heads, out_dtype):
    """FP16 softmax with the softmax scale folded into the Q/K quantization (unit scaleSoftmaxLog2)."""
    _require(num_heads, out_dtype, fp16_softmax=True)
    batch, seq_len = 2, 1560
    q, k, v = _inputs(batch, seq_len, num_heads, seed=1)
    qk_mul = math.sqrt(HEAD_DIM**-0.5 * LOG2E)
    vs = v.abs().amax() / E4M3_MAX
    q8, k8, v8 = _quant(q * qk_mul, 1.0), _quant(k * qk_mul, 1.0), _quant(v, vs)

    out_scale = torch.tensor(4.0, device="cuda") if out_dtype == E4M3 else None
    scale_out = (vs / out_scale if out_scale is not None else vs).reshape(1).float()

    out = torch.ops.trtllm.fmha_dense_fp8(
        q8, k8, v8, _cu_seqlens(batch, seq_len), seq_len, None, scale_out, True, out_dtype
    )
    result = out.float() * (out_scale if out_scale is not None else 1.0)
    ref = _reference(q8.float() / qk_mul, k8.float() / qk_mul, v8.float() * vs, batch, seq_len)
    assert _cosine(result, ref) > _min_cosine(out_dtype)


def test_fp16_softmax_uses_a_unit_softmax_scale():
    """With fp16_softmax the op runs the kernels with scaleSoftmaxLog2 = 1 whatever scale_bmm1 holds."""
    _require(10, torch.bfloat16, fp16_softmax=True)
    num_heads, batch, seq_len = 10, 1, 1560
    q, k, v = (_quant(t * 0.5, 1.0) for t in _inputs(batch, seq_len, num_heads, seed=2))
    cu, scale_out = _cu_seqlens(batch, seq_len), torch.ones(1, device="cuda")

    def run(scale_bmm1):
        return torch.ops.trtllm.fmha_dense_fp8(
            q, k, v, cu, seq_len, scale_bmm1, scale_out, True, torch.bfloat16
        )

    out = run(None)
    assert torch.equal(out, run(torch.tensor([1.0 / LOG2E, 2.0], device="cuda")))

    # softmax(exp2(q.k)): the SDPA reference divides by sqrt(D), so its inputs are rescaled.
    mul = (HEAD_DIM**0.5 / LOG2E) ** 0.5
    ref = _reference(q.float() * mul, k.float() * mul, v, batch, seq_len)
    assert _cosine(out, ref) > 0.999


@pytest.mark.parametrize("fp16_softmax", [False, True])
def test_variable_seqlens_match_sdpa(fp16_softmax):
    """Packed sequences of different lengths; each one attends only to itself."""
    _require(10, torch.bfloat16, fp16_softmax)
    num_heads, seq_lens = 10, [700, 1560, 129]
    cu = torch.tensor([0] + seq_lens, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    q, k, v = _inputs(1, sum(seq_lens), num_heads, seed=4)
    vs = v.abs().amax() / E4M3_MAX
    v8 = _quant(v, vs)
    if fp16_softmax:
        qk_mul = math.sqrt(HEAD_DIM**-0.5 * LOG2E)
        q8, k8 = _quant(q * qk_mul, 1.0), _quant(k * qk_mul, 1.0)
        q_deq, k_deq, bmm1 = q8.float() / qk_mul, k8.float() / qk_mul, None
    else:
        qs, ks = q.abs().amax() / E4M3_MAX, k.abs().amax() / E4M3_MAX
        q8, k8 = _quant(q, qs), _quant(k, ks)
        q_deq, k_deq = q8.float() * qs, k8.float() * ks
        softmax_scale = HEAD_DIM**-0.5
        bmm1 = torch.stack([softmax_scale * qs * ks, softmax_scale * qs * ks * LOG2E]).float()

    out = torch.ops.trtllm.fmha_dense_fp8(
        q8, k8, v8, cu, max(seq_lens), bmm1, vs.reshape(1).float(), fp16_softmax, torch.bfloat16
    )
    v_deq = v8.float() * vs
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        ref = _reference(q_deq[start:end], k_deq[start:end], v_deq[start:end], 1, end - start)
        assert _cosine(out[start:end], ref) > _min_cosine(torch.bfloat16)


@pytest.mark.parametrize("fp16_softmax", [False, True])
def test_cuda_graph_replay(fp16_softmax):
    """Scales come from device memory, so a captured graph follows new inputs and scales."""
    _require(10, torch.bfloat16, fp16_softmax)
    num_heads, batch, seq_len = 10, 1, 1560
    q8, k8, v8 = (_quant(t * 0.5, 1.0) for t in _inputs(batch, seq_len, num_heads, seed=5))
    cu = _cu_seqlens(batch, seq_len)
    bmm1 = torch.tensor([HEAD_DIM**-0.5, HEAD_DIM**-0.5 * LOG2E], device="cuda")
    scale_out = torch.ones(1, device="cuda")

    def run():
        return torch.ops.trtllm.fmha_dense_fp8(
            q8, k8, v8, cu, seq_len, bmm1, scale_out, fp16_softmax, torch.bfloat16
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # warm up: loads the cubins outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    for seed in (6, 7):
        new_q, new_k, new_v = (
            _quant(t * 0.5, 1.0) for t in _inputs(batch, seq_len, num_heads, seed)
        )
        q8.copy_(new_q), k8.copy_(new_k), v8.copy_(new_v)
        scale_out.fill_(0.5 * seed)
        graph.replay()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)


def test_rejects_bad_max_seqlen():
    _require(10, torch.bfloat16, fp16_softmax=False)
    q8, k8, v8 = (_quant(t, 1.0) for t in _inputs(1, 256, 10, seed=8))
    bmm1 = torch.tensor([HEAD_DIM**-0.5, HEAD_DIM**-0.5 * LOG2E], device="cuda")
    for max_seqlen in (0, 257):
        with pytest.raises(RuntimeError, match="max_seqlen"):
            torch.ops.trtllm.fmha_dense_fp8(
                q8, k8, v8, _cu_seqlens(1, 256), max_seqlen, bmm1, torch.ones(1, device="cuda"),
                False, torch.bfloat16,
            )  # fmt: skip


def test_is_supported_probe():
    assert not _supported(40, torch.float16, fp16_softmax=False)
    for fp16_softmax in (False, True):
        assert isinstance(_supported(40, torch.bfloat16, fp16_softmax), bool)
    if get_sm_version() == 107:
        for out_dtype in (torch.bfloat16, E4M3):
            assert _supported(40, out_dtype, fp16_softmax=True)
            assert _supported(40, out_dtype, fp16_softmax=False)


def test_fake_under_compile():
    _require(10, torch.bfloat16, fp16_softmax=False)
    batch, seq_len = 1, 256
    q8, k8, v8 = (_quant(t, 1.0) for t in _inputs(batch, seq_len, 10, seed=3))
    cu = _cu_seqlens(batch, seq_len)
    bmm1 = torch.tensor([HEAD_DIM**-0.5, HEAD_DIM**-0.5 * LOG2E], device="cuda")
    scale_out = torch.ones(1, device="cuda")

    def fn(q, k, v):
        return torch.ops.trtllm.fmha_dense_fp8(
            q, k, v, cu, seq_len, bmm1, scale_out, False, torch.bfloat16
        )

    eager = fn(q8, k8, v8)
    compiled = torch.compile(fn, fullgraph=True)(q8, k8, v8)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
