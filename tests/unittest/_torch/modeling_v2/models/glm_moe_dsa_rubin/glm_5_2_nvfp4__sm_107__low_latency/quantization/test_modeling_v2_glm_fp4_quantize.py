# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification for the GLM-5.2 routed-expert NVFP4 input."""

import zlib
from collections.abc import Callable

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize as shared_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.quantization.fp4_quantize import (  # noqa: E501
    fp4_quantize as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 NVFP4 input is certified on sm_107"
)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack [M, K/2] bytes into [M, K] codes, with element 2i in the low nibble."""
    rows, half = packed.shape
    codes = torch.empty(rows, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell: Cell, certifying: Callable[[OpWrapper], None]) -> None:  # noqa: F811 — pytest resolves the fixture
    """Scales are exact; packed codes are exact outside the shared rounding-tie window."""
    certifying(op)
    generator = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(cell.spec["shape"], generator=generator, device="cuda").to(cell.spec["dtype"])
    global_scale = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    vec, swizzled = cell.spec["vec"], cell.spec["swizzled"]

    packed, scales = op(x, global_scale, vec, False, swizzled)
    ref_packed, ref_scales = op.reference(x, global_scale, vec, False, swizzled)
    assert torch.equal(scales, ref_scales), "block scales are exact; a mismatch is a defect"

    tie = shared_op.near_tie(shared_op.scaled(x, global_scale, vec))
    codes, ref_codes = _unpack(packed), _unpack(ref_packed)
    assert torch.equal(codes[~tie], ref_codes[~tie]), "codes differ away from a rounding tie"
    # The shared NVFP4 certificate bounds the tie window to 1% of the elements.
    assert tie.sum().item() <= 0.01 * tie.numel(), (
        f"{tie.sum().item()} of {tie.numel()} elements sit in the tie window; "
        "a window this wide is no longer a statement about rounding"
    )
