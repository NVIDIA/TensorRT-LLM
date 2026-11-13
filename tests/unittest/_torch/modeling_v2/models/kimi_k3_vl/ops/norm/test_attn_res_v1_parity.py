# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's attn_res ops give the same bits as the built-in model's shared ops they were copied from.

The shared ops (torch.ops.trtllm.attn_res_*, cpp/tensorrt_llm/thop/attnResOp.cpp) stay in the wheel for the built-in
Kimi K3 model, so each OpWrapper's cells run through both and every output must match bit for bit:

- attn_res_fwd (k3_attn_res_fwd) against trtllm::attn_res_fwd: output, rsigma, probs and logits;
- attn_res_rmsnorm_fwd (k3_attn_res_rmsnorm_fwd, early trigger) against trtllm::attn_res_rmsnorm_fwd;
- attn_res_add_rmsnorm_fwd (k3_attn_res_add_rmsnorm_fwd, early trigger) against trtllm::attn_res_add_rmsnorm_fwd;
- attn_res_add_rmsnorm_persistent_fwd against trtllm::attn_res_add_rmsnorm_persistent_fwd, with and without the
  addend and with sm_reserve.

The early trigger only moves when a PDL dependent may launch; no dependent runs here, so it cannot change the bits.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers the shared torch.ops.trtllm.* comparison ops)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_add_rmsnorm_fwd import (
    attn_res_add_rmsnorm_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_add_rmsnorm_persistent_fwd import (  # noqa: E501
    attn_res_add_rmsnorm_persistent_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_fwd import (
    attn_res_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_rmsnorm_fwd import (
    attn_res_rmsnorm_fwd,
)

__extra_import_path__ = ["."]
from _native_attn_res_inputs import bits, make_inputs

assert torch.cuda.is_available(), "the attn_res ops require a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)


def _assert_same_bits(family, builtin, cell) -> None:
    assert len(family) == len(builtin), cell.why
    for index, (f, b) in enumerate(zip(family, builtin)):
        assert f.shape == b.shape and f.dtype == b.dtype, f"{cell.why}: output {index}"
        assert torch.equal(bits(f), bits(b)), f"{cell.why}: output {index} differs"


def _ids(op):
    return [c.why[:48] for c in op.CELLS]


@pytest.mark.parametrize("cell", attn_res_fwd.CELLS, ids=_ids(attn_res_fwd))
def test_attn_res_fwd(cell) -> None:
    x = make_inputs(cell)
    args = (
        x["layer_residual"],
        x["block_residual"],
        x["res_weight"],
        x["rms_weight"],
        cell.spec["eps"],
    )
    _assert_same_bits(attn_res_fwd(*args), torch.ops.trtllm.attn_res_fwd(*args), cell)


@pytest.mark.parametrize("cell", attn_res_rmsnorm_fwd.CELLS, ids=_ids(attn_res_rmsnorm_fwd))
def test_attn_res_rmsnorm_fwd(cell) -> None:
    x = make_inputs(cell)
    args = (
        x["layer_residual"],
        x["block_residual"],
        x["res_weight"],
        x["rms_weight"],
        x["output_rms_weight"],
        cell.spec["eps"],
        cell.spec["output_eps"],
    )
    _assert_same_bits(
        (attn_res_rmsnorm_fwd(*args),), (torch.ops.trtllm.attn_res_rmsnorm_fwd(*args),), cell
    )


@pytest.mark.parametrize("cell", attn_res_add_rmsnorm_fwd.CELLS, ids=_ids(attn_res_add_rmsnorm_fwd))
def test_attn_res_add_rmsnorm_fwd(cell) -> None:
    x = make_inputs(cell)
    args = (
        x["layer_residual"],
        x["addend"],
        x["block_residual"],
        x["res_weight"],
        x["rms_weight"],
        x["output_rms_weight"],
        cell.spec["eps"],
        cell.spec["output_eps"],
    )
    _assert_same_bits(
        attn_res_add_rmsnorm_fwd(*args), torch.ops.trtllm.attn_res_add_rmsnorm_fwd(*args), cell
    )


@pytest.mark.parametrize(
    "cell", attn_res_add_rmsnorm_persistent_fwd.CELLS, ids=_ids(attn_res_add_rmsnorm_persistent_fwd)
)
def test_attn_res_add_rmsnorm_persistent_fwd(cell) -> None:
    x = make_inputs(cell)
    args = (
        x["layer_residual"],
        x["addend"] if cell.spec["add"] else None,
        x["block_residual"],
        x["res_weight"],
        x["rms_weight"],
        x["output_rms_weight"],
        cell.spec["eps"],
        cell.spec["output_eps"],
        cell.spec["sm_reserve"],
    )
    _assert_same_bits(
        attn_res_add_rmsnorm_persistent_fwd(*args),
        torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(*args),
        cell,
    )
