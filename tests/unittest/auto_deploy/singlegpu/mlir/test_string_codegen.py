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

"""Tests for string-based Triton kernel generation."""

import pytest

xdsl = pytest.importorskip("xdsl")

import torch  # noqa: E402
from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    TensorType,
)
from xdsl.ir import Block  # noqa: E402

from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdMul,
    AdReduceMean,
    AdRsqrt,
    AdSplat,
)
from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (  # noqa: E402
    FusibleSubgraph,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generated_add_mul_correctness():
    """Generated add+mul kernel matches torch reference."""
    t = TensorType(BFloat16Type(), [2, 128])
    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)
    z = block.insert_arg(t, 2)

    add = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add)
    mul = AdMul.build(operands=[add.output, z], result_types=[t])
    block.add_op(mul)

    sg = FusibleSubgraph(
        ops=[add, mul],
        inputs=[x, y, z],
        outputs=[mul.output],
    )

    from tensorrt_llm._torch.auto_deploy.mlir.codegen.triton_emitter import (
        generate_kernel_from_subgraph,
    )

    kernel_fn = generate_kernel_from_subgraph(sg)
    assert callable(kernel_fn)

    # Test on GPU
    xt = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    yt = torch.randn_like(xt)
    zt = torch.randn_like(xt)
    result = kernel_fn(xt, yt, zt)
    ref = (xt + yt) * zt
    torch.testing.assert_close(result[0], ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generated_rmsnorm_correctness():
    """Generated decomposed rmsnorm kernel matches torch reference."""
    hidden = 128
    t = TensorType(BFloat16Type(), [2, hidden])
    t_scalar = TensorType(BFloat16Type(), [2, 1])
    tw = TensorType(BFloat16Type(), [hidden])

    block = Block()
    x = block.insert_arg(t, 0)
    w = block.insert_arg(tw, 1)

    # Build ops: mul(x,x) -> reduce_mean -> splat(eps) -> add(var,eps) -> rsqrt
    #   -> mul(x,inv) -> mul(normed,w)
    sq = AdMul.build(operands=[x, x], result_types=[t])
    block.add_op(sq)

    var = AdReduceMean.build(
        operands=[sq.output],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "keepdim": IntegerAttr(1, IntegerType(1)),
        },
        result_types=[t_scalar],
    )
    block.add_op(var)

    eps_t = AdSplat.build(
        attributes={"value": FloatAttr(1e-5, Float64Type())},
        result_types=[t_scalar],
    )
    block.add_op(eps_t)

    var_eps = AdAdd.build(operands=[var.output, eps_t.output], result_types=[t_scalar])
    block.add_op(var_eps)

    inv = AdRsqrt.build(operands=[var_eps.output], result_types=[t_scalar])
    block.add_op(inv)

    normed = AdMul.build(operands=[x, inv.output], result_types=[t])
    block.add_op(normed)

    result_op = AdMul.build(operands=[normed.output, w], result_types=[t])
    block.add_op(result_op)

    sg = FusibleSubgraph(
        ops=[sq, var, eps_t, var_eps, inv, normed, result_op],
        inputs=[x, w],
        outputs=[result_op.output],
    )

    from tensorrt_llm._torch.auto_deploy.mlir.codegen.triton_emitter import (
        generate_kernel_from_subgraph,
    )

    kernel_fn = generate_kernel_from_subgraph(sg)

    # Compare against torch reference rmsnorm
    xt = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)
    wt = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    result = kernel_fn(xt, wt)

    # Torch reference
    x_f32 = xt.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + 1e-5)
    ref = (x_f32 * inv_rms).to(torch.bfloat16) * wt

    torch.testing.assert_close(result[0], ref, atol=1e-2, rtol=1e-2)


def test_kernel_cache_reuse():
    """Same subgraph structure produces the same hash."""
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.kernel_cache import KernelCache

    cache = KernelCache()

    # Build same subgraph structure twice
    t = TensorType(BFloat16Type(), [2, 128])
    block1 = Block()
    x1 = block1.insert_arg(t, 0)
    y1 = block1.insert_arg(t, 1)
    add1 = AdAdd.build(operands=[x1, y1], result_types=[t])
    block1.add_op(add1)
    mul1 = AdMul.build(operands=[add1.output, x1], result_types=[t])
    block1.add_op(mul1)
    sg1 = FusibleSubgraph(ops=[add1, mul1], inputs=[x1, y1], outputs=[mul1.output])

    block2 = Block()
    x2 = block2.insert_arg(t, 0)
    y2 = block2.insert_arg(t, 1)
    add2 = AdAdd.build(operands=[x2, y2], result_types=[t])
    block2.add_op(add2)
    mul2 = AdMul.build(operands=[add2.output, x2], result_types=[t])
    block2.add_op(mul2)
    sg2 = FusibleSubgraph(ops=[add2, mul2], inputs=[x2, y2], outputs=[mul2.output])

    h1 = KernelCache.hash_subgraph(sg1)
    h2 = KernelCache.hash_subgraph(sg2)
    assert h1 == h2, "Same subgraph structure should produce same hash"

    # Verify cache put/get
    cache.put(h1, lambda: None)
    assert cache.get(h2) is not None
