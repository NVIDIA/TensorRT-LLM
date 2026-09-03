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
"""Tests for the two-input SwiGLU used by split gate/up projections.

`silu_and_mul` requires gate and up adjacent in one tensor. Models that keep
gate and up as separate projections -- static-FP8 Cosmos3 does, to preserve each
projection's calibrated scale -- have no such tensor, and concatenating one is a
large GPU copy. `silu_and_mul_2in` consumes the two tensors directly.

Every case asserts *bit-exact* equality with the fused op rather than a
tolerance: the two kernels perform the same arithmetic in the same order and
accumulate in fp32, so any difference is a defect rather than drift.
"""

import pytest
import torch

from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    _SILU_AND_MUL_2IN_LAUNCH_PARAMS,
    _silu_and_mul_2in_launch_params,
)
from tensorrt_llm._torch.modules.swiglu import swiglu, swiglu_2in

# (M, intermediate). The large case is the Cosmos3 Nano default T2V request:
# 720x1280 x 189 frames with CFG gives M = 88320.
SHAPES = [(8, 4), (13, 257), (1760, 12288), (88320, 12288)]
DTYPES = [torch.bfloat16, torch.float16]


def _reference(gate, up, **kwargs):
    """The fused path, fed an explicitly concatenated tensor.

    silu_and_mul is rank-2 only, so higher-rank inputs are flattened for the
    comparison and the result restored to the original shape.
    """
    flat_gate = gate.reshape(-1, gate.shape[-1])
    flat_up = up.reshape(-1, up.shape[-1])
    out = swiglu(torch.cat([flat_gate, flat_up], dim=-1), **kwargs)
    return out.reshape(gate.shape)


def _capture(fn):
    """Capture fn() after warming up on a side stream.

    Capture is sensitive to allocator state left by earlier tests, so the
    documented warmup protocol is used rather than a single default-stream call.
    """
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return out


def _pair(m, n, dtype, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(m, n, device="cuda", dtype=dtype),
        torch.randn(m, n, device="cuda", dtype=dtype),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m, n", SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16"])
def test_matches_fused_silu_and_mul(m, n, dtype):
    gate, up = _pair(m, n, dtype)
    expected = _reference(gate, up)
    actual = swiglu_2in(gate, up)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("swiglu_limit", [1.0, 7.0])
def test_matches_fused_with_swiglu_limit(swiglu_limit):
    """The limit clamps gate and up differently, so it must be plumbed through."""
    gate, up = _pair(256, 512, torch.bfloat16)
    # Scale up so values actually exceed the limit and the clamp is exercised.
    gate, up = gate * 10.0, up * 10.0

    expected = _reference(gate, up, swiglu_limit=swiglu_limit)
    actual = swiglu_2in(gate, up, swiglu_limit=swiglu_limit)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "swiglu_alpha, swiglu_beta",
    [(1.702, 1.0), (1.702, 0.0), (1.0, 1.0)],
    ids=["swigluoai", "alpha_only", "beta_only"],
)
def test_matches_fused_with_alpha_beta(swiglu_alpha, swiglu_beta):
    """alpha gains inside the sigmoid, beta offsets up; both must be plumbed.

    Defaulting them drops the kernel to the alpha=1, beta=0 special case, which
    is numerically wrong rather than merely unsupported -- and silently so.
    """
    gate, up = _pair(256, 512, torch.bfloat16)

    expected = _reference(gate, up, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
    actual = swiglu_2in(gate, up, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_alpha_beta_defaults_match_plain_swiglu():
    """Omitting alpha/beta must stay bit-identical to plain silu_and_mul."""
    gate, up = _pair(256, 512, torch.bfloat16)

    assert torch.equal(
        swiglu_2in(gate, up), swiglu_2in(gate, up, swiglu_alpha=1.0, swiglu_beta=0.0)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m, n", [(256, 512), (1760, 12288)], ids=lambda v: str(v))
def test_fp8_output_matches_fused(m, n):
    """With an FP8 down_proj, GatedMLP asks SwiGLU to emit FP8 directly.

    The activation then carries down_proj's input quantization, so the two-input
    form has to reproduce it exactly rather than returning BF16.
    """
    gate, up = _pair(m, n, torch.bfloat16)
    scale = torch.tensor(1e-2, device="cuda", dtype=torch.float32)
    kwargs = dict(quant_scale=scale, quant_type=torch.float8_e4m3fn)

    expected = _reference(gate, up, **kwargs)
    actual = swiglu_2in(gate, up, **kwargs)

    assert actual.dtype == torch.float8_e4m3fn
    assert torch.equal(actual.float(), expected.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16_out", "fp8_out"])
def test_torch_compile_fullgraph(quantized):
    """Exercises register_fake: a wrong meta impl breaks tracing, not eager.

    The Cosmos3 E2E helper runs with compilation disabled, so nothing else in
    this feature's test surface would catch it.
    """
    gate, up = _pair(256, 512, torch.bfloat16)
    scale = torch.tensor(1e-2, device="cuda", dtype=torch.float32)

    def fn(g, u):
        if quantized:
            return swiglu_2in(g, u, quant_scale=scale, quant_type=torch.float8_e4m3fn)
        return swiglu_2in(g, u)

    eager = fn(gate, up)
    compiled = torch.compile(fn, fullgraph=True)(gate, up)

    assert compiled.shape == eager.shape
    assert compiled.dtype == eager.dtype
    assert torch.equal(compiled.float(), eager.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_capture():
    gate, up = _pair(256, 512, torch.bfloat16)
    captured = _capture(lambda: swiglu_2in(gate, up))
    assert torch.equal(captured, swiglu_2in(gate, up))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rejects_mismatched_inputs():
    gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(Exception, match="same shape"):
        swiglu_2in(gate, torch.randn(8, 32, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(Exception, match="same dtype"):
        swiglu_2in(gate, torch.randn(8, 16, device="cuda", dtype=torch.float16))
    with pytest.raises(Exception, match="same device"):
        swiglu_2in(gate, torch.randn(8, 16, dtype=torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rejects_non_contiguous():
    """The kernel indexes flat runs, so a strided view must be rejected.

    Accepting one produced silently wrong output rather than an error: a (32, 2)
    view with inner stride 2 differed from the reference by 2.7 in BF16 and 272
    with FP8 output.
    """
    gate = torch.randn(32, 4, device="cuda", dtype=torch.bfloat16)[:, ::2]
    up = torch.randn(32, 4, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not gate.is_contiguous()

    with pytest.raises(Exception, match="contiguous"):
        swiglu_2in(gate, up)
    # Contiguous in one operand only is still rejected.
    with pytest.raises(Exception, match="contiguous"):
        swiglu_2in(gate.contiguous(), up)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16_out", "fp8_out"])
def test_opcheck(quantized):
    """Full torch.library contract: schema, fake tensor, autograd registration."""
    gate, up = _pair(64, 128, torch.bfloat16)
    kwargs = {}
    if quantized:
        kwargs = dict(
            scale=torch.tensor(1e-2, device="cuda", dtype=torch.float32), dtype=torch.float8_e4m3fn
        )
    torch.library.opcheck(torch.ops.trtllm.silu_and_mul_2in, (gate, up), kwargs)


@pytest.mark.parametrize("sm_version", sorted(_SILU_AND_MUL_2IN_LAUNCH_PARAMS) + [90, 120])
def test_launch_params_are_valid_launch_configs(sm_version):
    """Guards the tuning table against edits that would not launch.

    Triton needs a power-of-two block, and threads (num_warps * 32) must stay
    within the 1024-per-block limit -- easy to violate when hand-editing tuned
    values. Covers every tuned SM version plus untuned ones that take the
    fallback row, since an untuned architecture must still launch.
    """
    for out_dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        block_elements, num_warps = _silu_and_mul_2in_launch_params(sm_version, out_dtype)
        assert block_elements & (block_elements - 1) == 0, (
            f"sm_{sm_version} {out_dtype}: block_elements {block_elements} is not a power of two"
        )
        assert 1 <= num_warps <= 32
        assert num_warps * 32 <= 1024, (
            f"sm_{sm_version} {out_dtype}: {num_warps} warps exceeds the per-block thread limit"
        )
        assert block_elements % (num_warps * 32) == 0, (
            f"sm_{sm_version} {out_dtype}: {block_elements} elements do not divide evenly "
            f"across {num_warps * 32} threads"
        )


def test_sm103_shares_the_sm100_row():
    """Pins the documented measurement: the sm_103 sweep landed within 0.4% of
    the sm_100 configuration, so it must not silently diverge."""
    for out_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        assert _silu_and_mul_2in_launch_params(103, out_dtype) == _silu_and_mul_2in_launch_params(
            100, out_dtype
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "shape", [(2, 8, 64), (4, 256, 512), (2, 3, 4, 16)], ids=["rank3_small", "rank3_model", "rank4"]
)
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16_out", "fp8_out"])
def test_rank_n_inputs(shape, quantized):
    """The kernel walks a flat run, so any matching contiguous shape is valid.

    Model activations are rank-3 [batch, seq, hidden]. An earlier revision
    asserted rank 2 in the op while its fake accepted rank 3, so compiled
    execution traced cleanly and then failed.
    """
    torch.manual_seed(0)
    gate = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    kwargs = {}
    if quantized:
        kwargs = dict(
            quant_scale=torch.tensor(1e-2, device="cuda", dtype=torch.float32),
            quant_type=torch.float8_e4m3fn,
        )

    actual = swiglu_2in(gate, up, **kwargs)
    assert actual.shape == gate.shape
    assert torch.equal(actual.float(), _reference(gate, up, **kwargs).float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rank3_compile_and_cuda_graph():
    """Guards the fake/eager rank agreement that previously diverged."""
    gate, up = _pair(2 * 8, 64, torch.bfloat16)
    gate, up = gate.reshape(2, 8, 64), up.reshape(2, 8, 64)

    eager = swiglu_2in(gate, up)
    compiled = torch.compile(lambda g, u: swiglu_2in(g, u), fullgraph=True)(gate, up)
    assert compiled.shape == eager.shape
    assert torch.equal(compiled, eager)

    captured = _capture(lambda: swiglu_2in(gate, up))
    assert torch.equal(captured, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_opcheck_rank3():
    gate, up = _pair(2 * 8, 64, torch.bfloat16)
    torch.library.opcheck(
        torch.ops.trtllm.silu_and_mul_2in, (gate.reshape(2, 8, 64), up.reshape(2, 8, 64))
    )
