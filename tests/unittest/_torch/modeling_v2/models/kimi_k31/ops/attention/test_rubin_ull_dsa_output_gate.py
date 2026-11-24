# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_output_gate entry: every cell bitwise PyTorch's two BF16 kernels,
written into the caller's buffer, and the target's call site (``dsa.sigmoid_gate_into``) the same."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_output_gate import (  # noqa: E501
    dsa_output_gate as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the ULL target runs on SM10x",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

_TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
WIDTH = 6 * 128  # this rank's 6 heads of 128 values


def _kernel():
    import importlib

    gate = importlib.import_module(
        _TARGET.rpartition(".")[0] + ".kernels.attention.rubin_ull_mla_output_gate"
    )
    return gate._sigmoid_gate_mul_kernel, gate._BLOCK


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    import importlib

    kernel, block = _kernel()
    rows, scale = cell.spec["rows"], cell.spec["scale"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(rows, WIDTH, generator=g, device="cuda").bfloat16()
    gate = (torch.randn(rows, WIDTH, generator=g, device="cuda") * scale).bfloat16()
    if scale > 10:
        # Gates in the denormal-sigmoid band and past the saturation points.
        gate.view(-1)[:6] = torch.tensor([-95.0, -100.0, -88.0, -120.0, 20.0, 120.0]).bfloat16()
    out = torch.full_like(x, float("nan"))
    with validating(op):
        got = op(kernel, block, x, gate, out)
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr()
    ref = op.reference(kernel, block, x, gate, out)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    # The target's call site is the same launch.
    dsa = importlib.import_module(_TARGET + ".modeling")
    again = torch.full_like(x, float("nan"))
    dsa.sigmoid_gate_into(x, gate, again)
    torch.cuda.synchronize()
    assert torch.equal(again.view(torch.int16), out.view(torch.int16))


def test_is_valid_refuses_what_the_kernel_would_read_wrongly() -> None:
    kernel, block = _kernel()
    x = torch.zeros(8, WIDTH, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="one shape"):
        op.is_valid(kernel, block, x, x[:4], x)
    with pytest.raises(ValueError, match="one shape"):
        op.is_valid(kernel, block, x, x.t(), x.t())
    with pytest.raises(ValueError, match="one shape"):
        op.is_valid(kernel, block, x, x.float(), x)
