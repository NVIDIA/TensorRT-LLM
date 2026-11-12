# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_merge entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_moe_merge runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_merge import (  # noqa: E402, E501
    HIDDEN,
    CompiledMerge,
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_merge import (  # noqa: E402, E501
    cute_dsl_moe_merge as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

#: The compiled add serves every row count; built once a session.
_HANDLES: dict[int, CompiledMerge] = {}


def _compiled() -> CompiledMerge:
    device = torch.cuda.current_device()
    if device not in _HANDLES:
        _HANDLES[device] = build(device)
    return _HANDLES[device]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Bit-exact: both sides do the same arithmetic on each element.

    The kernel widens each bf16 pair to fp32 (exact), adds them in fp32 with
    round-to-nearest-even, and narrows the sum to bf16 with
    round-to-nearest-even. The reference is `(routed.float() +
    shared.float()).to(bfloat16)`: the same three steps, element by element,
    with no accumulation order to differ in. So every output bit has to match,
    and the comparison is `torch.equal`. A wrong element pairing, a dropped
    operand or an unwritten block differs by O(1).

    `routed` is the output, written in place, so it cannot start as NaN. The
    reference is taken from a copy of the original; an element left unwritten
    keeps `routed` alone, which equals `routed + shared` only where `shared` is
    below half an ulp of `routed`.
    """
    rows = cell.spec["rows"]
    compiled = _compiled()
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    routed = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    shared = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    routed_before, shared_before = routed.clone(), shared.clone()
    with validating(op):
        assert op(routed, shared, compiled) is routed, "the op writes `routed` and returns it"
    assert torch.equal(shared, shared_before), "the op wrote `shared`"
    assert torch.equal(routed, op.reference(routed_before, shared, compiled)), (
        "the sum differs from the fp32 sum rounded once to bf16"
    )


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def test_guard_refuses_operands_of_different_shapes() -> None:
    """The kernel adds `routed.numel()` elements from each pointer: a shorter
    `shared` is read past its end, a longer one is silently cut."""
    routed = torch.zeros(65, HIDDEN, dtype=torch.bfloat16, device="cuda")
    shared = torch.zeros(66, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="shared must be bf16"):
        with validating(op):
            op(routed, shared, CompiledMerge(0, _unreachable))


def test_guard_refuses_partially_overlapping_operands() -> None:
    """Two views of one buffer one row apart: each CTA reads `shared` elements
    another CTA may already have overwritten as `routed`."""
    buf = torch.zeros(66, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="may alias exactly but not partially overlap"):
        with validating(op):
            op(buf[1:], buf[:-1], CompiledMerge(0, _unreachable))
