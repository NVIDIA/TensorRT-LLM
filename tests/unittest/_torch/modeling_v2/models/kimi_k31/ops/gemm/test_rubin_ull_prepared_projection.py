# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's prepared_projection entry: MR8's split-K KDA in-projection (BF16),
prepared as ``FlatKda`` prepares it, against the FP64 product."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_mr8 import (  # noqa: E501
    module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    weights as layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_prepared_projection import (  # noqa: E501
    prepared_projection as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_prepared_projection import (  # noqa: E501
    projection_weight,
)

__extra_import_path__ = ["../../../..", "."]
from _bf16_band import assert_bf16_gemm_close  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="MR8's split-K projection needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

PITCH, HIDDEN, TOKENS = layers.kda_pitch(6), 7168, 8


@pytest.fixture(scope="module")
def checkpoint():
    """E4M3 codes and UE8M0 bytes of one packed KDA in-projection, exactly BF16 when dequantized."""
    g = torch.Generator(device="cuda").manual_seed(3)
    codes = (torch.randn(PITCH, HIDDEN, generator=g, device="cuda") * 32).to(torch.float8_e4m3fn)
    raw = torch.randint(110, 116, (PITCH, HIDDEN // 32), generator=g, device="cuda")
    return codes, raw.to(torch.uint8)


def _projection(checkpoint, rows: int):
    codes, raw = checkpoint
    weight = layers.dequantize_mxfp8(codes, raw)
    providers = module("runtime.providers")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    policies = providers.projection_policies(16, PITCH, [rows], sms, TOKENS)
    return providers.PreparedProjection(weight, policies).prepare()


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, checkpoint) -> None:
    rows = cell.spec["rows"]
    projection = _projection(checkpoint, rows)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    out = torch.full((rows, PITCH), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        result = op(projection, x, out)
    torch.cuda.synchronize()
    assert result is out
    weight = projection_weight(projection)
    assert_bf16_gemm_close(out, op.reference(projection, x, out), x, weight)
