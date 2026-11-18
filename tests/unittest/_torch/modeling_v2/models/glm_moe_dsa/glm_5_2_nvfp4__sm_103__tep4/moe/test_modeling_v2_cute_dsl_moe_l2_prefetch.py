# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_l2_prefetch entry: the kernel selects the
experts the reference ranks, and writes nothing."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.moe import (  # noqa: E501
    moe_decode_l2_prefetch as kernel,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_l2_prefetch as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_moe_l2_prefetch


@pytest.fixture(scope="module")
def operands():
    """One layer's FC1 operands in the grouped GEMM's layout (bytes only matter for size)."""
    w = torch.randint(0, 256, (64, 4096, 3072), dtype=torch.uint8, device="cuda")
    s = torch.randint(0, 256, (64, 32, 96, 512), dtype=torch.uint8, device="cuda")
    return w, s


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device())


def _hist(active: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.zeros(64, dtype=torch.int32, device="cuda")
    if active:
        idx = torch.randperm(64, generator=g, device="cuda")[:active]
        h[idx] = torch.randint(1, 9, (active,), generator=g, device="cuda", dtype=torch.int32)
    return h


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, operands) -> None:
    w, s = operands
    hist = _hist(cell.spec["active"], zlib.crc32(cell.why.encode()) % 2**31)
    before = (hist.clone(), w[:2].clone(), s[:2].clone())
    with validating(op):
        assert op(hist, w, s, compiled) is None
    torch.cuda.synchronize()
    assert (
        torch.equal(hist, before[0])
        and torch.equal(w[:2], before[1])
        and torch.equal(s[:2], before[2])
    )
    # The selection the kernel makes, through its debug form: the reference's ranking.
    sel = torch.full((kernel.MAX_TOP_E,), -7, dtype=torch.int32, device="cuda")
    kernel.run_prefetch(hist, w, s, top_e=entry.TOP_E, ntiles=entry.NTILES, ctas=entry.CTAS,
                        sel_out=sel)  # fmt: skip
    torch.cuda.synchronize()
    want = entry.reference_selection(hist, entry.TOP_E)
    assert sel[: len(want)].tolist() == want, (sel.tolist(), want)


def test_is_valid_refuses_a_wrong_histogram(compiled, operands) -> None:
    w, s = operands
    with pytest.raises(AssertionError, match="expert_hist"):
        with validating(op):
            op(torch.zeros(32, dtype=torch.int32, device="cuda"), w, s, compiled)
