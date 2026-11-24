# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's situ_and_mul entry: every cell against the FP64 activation, within
one BF16 rounding of each side plus the libdevice tanh's few FP32 ulps."""

import subprocess
import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.rubin_ull_situ_and_mul import (  # noqa: E501
    situ_and_mul as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def test_entry_registers_its_op() -> None:
    """A fresh interpreter that imports only this entry has torch.ops.trtllm.situ_and_mul, as the
    helix16 generation target does (it has no other import of tensorrt_llm._torch.modules.situ)."""
    code = (
        "import torch\n"
        "import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation"
        ".rubin_ull_situ_and_mul\n"
        "assert hasattr(torch.ops.trtllm, 'situ_and_mul')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # Gates and ups across the activation's curve and past the soft cap's knee.
    x = (torch.randn(spec["rows"], 2 * spec["d"], generator=g, device="cuda") * 8).to(
        torch.bfloat16
    )
    with validating(op):
        out = op(x, spec["beta"], spec["linear_beta"])
    torch.cuda.synchronize()
    ref = op.reference(x, spec["beta"], spec["linear_beta"])
    assert out.shape == ref.shape and out.dtype == ref.dtype
    err = (out.double() - ref.double()).abs()
    band = 2.0**-7 * ref.double().abs() + 2.0**-120
    assert bool((err <= band).all()), f"max err / band {(err / band).max().item():.3f}"
