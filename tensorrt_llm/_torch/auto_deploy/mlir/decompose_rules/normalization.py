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

"""Decomposition rules for normalization ops.

Decomposes ``ad.rmsnorm`` into a sequence of primitive ops:
    x^2 -> mean(dim=-1) -> +eps -> rsqrt -> x*inv -> *weight
"""

from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType

from ..decompose import decomposition
from ..dialect import AdAdd, AdMul, AdReduceMean, AdRMSNorm, AdRsqrt, AdSplat


@decomposition(AdRMSNorm)
def decompose_rmsnorm(op: AdRMSNorm):
    """Decompose ``ad.rmsnorm(x, w, eps)`` into primitives.

    Steps:
        1. sq = x * x
        2. var = reduce_mean(sq, dim=-1, keepdim=True)
        3. eps_tensor = splat(eps)
        4. var_eps = var + eps_tensor
        5. inv = rsqrt(var_eps)
        6. normed = x * inv  (broadcast)
        7. result = normed * weight  (broadcast)
    """
    x = op.input
    w = op.weight
    eps_val = op.eps
    t = op.output.type

    # Build reduced shape: same as t but last dim = 1
    shape = t.get_shape()
    reduced_shape = [s if i < len(shape) - 1 else 1 for i, s in enumerate(shape)]
    t_scalar = TensorType(t.element_type, reduced_shape)

    ops = []

    # sq = x * x
    sq = AdMul.build(operands=[x, x], result_types=[t])
    ops.append(sq)

    # var = reduce_mean(sq, dim=-1, keepdim=True)
    var = AdReduceMean.build(
        operands=[sq.output],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "keepdim": IntegerAttr(1, IntegerType(1)),
        },
        result_types=[t_scalar],
    )
    ops.append(var)

    # eps_tensor = splat(eps)
    eps_t = AdSplat.build(attributes={"value": eps_val}, result_types=[t_scalar])
    ops.append(eps_t)

    # var_eps = var + eps
    var_eps = AdAdd.build(operands=[var.output, eps_t.output], result_types=[t_scalar])
    ops.append(var_eps)

    # inv = rsqrt(var_eps)
    inv = AdRsqrt.build(operands=[var_eps.output], result_types=[t_scalar])
    ops.append(inv)

    # normed = x * inv  (broadcast)
    normed = AdMul.build(operands=[x, inv.output], result_types=[t])
    ops.append(normed)

    # result = normed * weight  (broadcast)
    result = AdMul.build(operands=[normed.output, w], result_types=[t])
    ops.append(result)

    return ops, result.output
