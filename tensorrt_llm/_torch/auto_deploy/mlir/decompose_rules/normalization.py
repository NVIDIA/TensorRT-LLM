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

Decomposes ``ad.rmsnorm`` and ``ad.gated_rmsnorm`` into sequences of
primitive ops suitable for elementwise fusion.

Standard RMSNorm:
    x^2 -> mean(dim=-1) -> +eps -> rsqrt -> x*inv -> *weight

Gated RMSNorm (norm_before_gate=False, Nemotron Nano default):
    silu(gate) -> x*silu_gate -> sq -> mean(dim=-1, group_size=G) ->
    +eps -> rsqrt -> gated_x*inv -> *weight
"""

from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType

from ..decompose import decomposition
from ..dialect import (
    AdAdd,
    AdGatedRMSNorm,
    AdMul,
    AdReduceMean,
    AdRMSNorm,
    AdRsqrt,
    AdSilu,
    AdSplat,
)


def _build_rmsnorm_primitives(x, w, eps_val, t, group_size=0):
    """Build the core RMSNorm primitive ops: sq -> mean -> +eps -> rsqrt -> *inv -> *weight.

    Args:
        x: Input SSAValue.
        w: Weight SSAValue.
        eps_val: Epsilon FloatAttr.
        t: Output TensorType.
        group_size: Group size for grouped normalization (0 = full dim).

    Returns:
        Tuple of (ops_list, result_op) where result_op is the final mul.
    """
    shape = t.get_shape()

    if group_size > 0:
        # Grouped mode: reduced shape has last dim = ngroups (one scalar per group)
        hidden = shape[-1]
        ngroups = hidden // group_size if hidden > 0 else -1
        reduced_shape = [s if i < len(shape) - 1 else ngroups for i, s in enumerate(shape)]
    else:
        # Standard mode: reduced shape has last dim = 1
        reduced_shape = [s if i < len(shape) - 1 else 1 for i, s in enumerate(shape)]

    t_scalar = TensorType(t.element_type, reduced_shape)

    ops = []

    # sq = x * x
    sq = AdMul.build(operands=[x, x], result_types=[t])
    ops.append(sq)

    # var = reduce_mean(sq, dim=-1, keepdim=True, group_size=group_size)
    var = AdReduceMean.build(
        operands=[sq.output],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "keepdim": IntegerAttr(1, IntegerType(1)),
            "group_size": IntegerAttr(group_size, IntegerType(64)),
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

    # normed = x * inv  (broadcast within group)
    normed = AdMul.build(operands=[x, inv.output], result_types=[t])
    ops.append(normed)

    # result = normed * weight  (broadcast)
    result = AdMul.build(operands=[normed.output, w], result_types=[t])
    ops.append(result)

    return ops, result


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
    ops, result = _build_rmsnorm_primitives(op.input, op.weight, op.eps, op.output.type)
    return ops, result.output


@decomposition(AdGatedRMSNorm)
def decompose_gated_rmsnorm(op: AdGatedRMSNorm):
    """Decompose ``ad.gated_rmsnorm(x, w, gate, eps, group_size, norm_before_gate)``.

    For ``norm_before_gate=False`` (Nemotron Nano default):
        1. silu_gate = silu(gate)
        2. gated_x = x * silu_gate
        3-9. rmsnorm(gated_x, weight, eps, group_size)

    For ``norm_before_gate=True``:
        1-7. normed = rmsnorm(x, weight, eps, group_size)
        8. silu_gate = silu(gate)
        9. result = normed * silu_gate

    """
    # TODO: remove this guard once 2D grid is validated E2E
    group_size = op.group_size.value.data
    if group_size > 0:
        return None

    x = op.input
    w = op.weight
    gate = op.gate
    eps_val = op.eps
    group_size = op.group_size.value.data
    norm_before_gate = bool(op.norm_before_gate.value.data)
    t = op.output.type

    ops = []

    if not norm_before_gate:
        # Gate before norm: silu(gate) * x, then rmsnorm
        silu_gate = AdSilu.build(operands=[gate], result_types=[t])
        ops.append(silu_gate)

        gated_x = AdMul.build(operands=[x, silu_gate.output], result_types=[t])
        ops.append(gated_x)

        norm_ops, result = _build_rmsnorm_primitives(gated_x.output, w, eps_val, t, group_size)
        ops.extend(norm_ops)
    else:
        # Norm before gate: rmsnorm(x), then * silu(gate)
        norm_ops, norm_result = _build_rmsnorm_primitives(x, w, eps_val, t, group_size)
        ops.extend(norm_ops)

        silu_gate = AdSilu.build(operands=[gate], result_types=[t])
        ops.append(silu_gate)

        result = AdMul.build(operands=[norm_result.output, silu_gate.output], result_types=[t])
        ops.append(result)

    return ops, result.output
