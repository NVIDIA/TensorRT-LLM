# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mtp_moe_experts entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL grouped GEMMs are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    TOP_K,
    allocate_maps,
    reference_maps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_mtp_moe_experts import (  # noqa: E402, E501
    CompiledMtpExperts,
    allocate_c,
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_mtp_moe_experts import (  # noqa: E402, E501
    cute_dsl_mtp_moe_experts as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, INTER, EXPERTS, GLOBAL_EXPERTS = 6144, 2048, 64, 256
#: The target's capacity, the token budget.
MAX_ROWS = 8192

_STATE: dict = {}


def _state():
    """The weights, the per-slot buffers and the compiled pair, once per session. The
    weights are scaled so each GEMM's sums are O(1)."""
    if not _STATE:
        g = torch.Generator(device="cuda").manual_seed(20261)
        bf = {"dtype": torch.bfloat16, "device": "cuda"}
        fc1_w = torch.empty(EXPERTS, 2 * INTER, HIDDEN, **bf)
        fc2_w = torch.empty(EXPERTS, HIDDEN, INTER, **bf)
        for e in range(EXPERTS):
            fc1_w[e] = torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * 0.02
            fc2_w[e] = torch.randn(HIDDEN, INTER, generator=g, device="cuda") * 0.03
        _STATE.update(
            fc1_w=fc1_w,
            fc2_w=fc2_w,
            x=torch.zeros(MAX_ROWS, HIDDEN, **bf),
            out=torch.zeros(MAX_ROWS, HIDDEN, **bf),
            w=torch.zeros(MAX_ROWS, TOP_K, dtype=torch.float32, device="cuda"),
            maps=allocate_maps(MAX_ROWS, "cuda"),
            c=allocate_c(MAX_ROWS, INTER, "cuda"),
        )
        _STATE["compiled"] = build(
            x=_STATE["x"],
            c=_STATE["c"],
            out=_STATE["out"],
            topk_w=_STATE["w"],
            maps=_STATE["maps"],
            fc1_w=fc1_w,
            fc2_w=fc2_w,
            max_rows=MAX_ROWS,
        )
    return _STATE


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides round the SwiGLU output to bf16 once (FC1 writes bf16) and accumulate
    the GEMMs in fp32; they differ in two places.

    1. FC1's fp32 sums differ from the reference's in order (~5e-6 relative) and its
       sigmoid is approximate, so about one intermediate in a thousand rounds to the
       neighbouring bf16 value: one bf16 ulp on one of 2048 terms, far below the band.
    2. FC2 adds each of a token's (up to eight) contributions into the bf16 output with
       atomics, one rounding per add in a varying order, against one rounding of the
       fp32 sum: up to ~4.5 ulp of the row scale at the worst element, ~0.8 ulp RMS.

    Hence 16 ulp element-wise and 2 ulp relative RMS. A wrong gate/up pairing, a missing
    expert or a row gathered from the wrong token is O(1).
    """
    assert_within_ulp(out, ref, element_ulp=16.0, rms_ulp=2.0)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    rows, offset = cell.spec["rows"], cell.spec["offset"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    s = _state()
    x = s["x"][:rows]
    x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda"))
    # Top-8 of 256 with about half the winners in this rank's window, skewed so some
    # experts take several 128-row tiles at 8192 rows.
    bias = torch.randn(GLOBAL_EXPERTS, generator=g, device="cuda") * 0.5
    bias[offset : offset + EXPERTS] += 1.0
    choice = torch.randn(rows, GLOBAL_EXPERTS, generator=g, device="cuda") * 0.3 + bias
    ids = torch.topk(choice, TOP_K, dim=-1).indices.to(torch.int32)
    raw = torch.rand(rows, TOP_K, generator=g, device="cuda") + 0.1
    topk_w = s["w"][:rows]
    topk_w.copy_(raw / raw.sum(-1, keepdim=True) * 2.5)
    maps = s["maps"]
    ref_maps = reference_maps(ids, offset)
    for k, t in maps.items():
        t.fill_(-7)
        t[: ref_maps[k].numel()].copy_(ref_maps[k])
    out = s["out"][:rows]
    out.zero_()  # the fused top-k's zero plane, or `zero_()` on a context step
    initial = out.clone()
    with validating(op):
        result = op(x, maps, topk_w, s["fc1_w"], s["fc2_w"], out, s["c"], s["compiled"])
    assert result is out, "the op accumulates into `out` and returns it"
    ref = op.reference(x, maps, topk_w, s["fc1_w"], s["fc2_w"], initial, s["c"], s["compiled"])
    _compare(out, ref)


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _fake(max_rows: int) -> CompiledMtpExperts:
    bias = torch.zeros(1, device="cuda")
    return CompiledMtpExperts(_unreachable, _unreachable, bias, HIDDEN, INTER, EXPERTS, max_rows)


_EMPTY: dict = {}


def _args(rows: int):
    """Operands with the right shapes and no content: the guards read only metadata."""
    bf = {"dtype": torch.bfloat16, "device": "cuda"}
    if not _EMPTY:
        _EMPTY["fc1_w"] = torch.empty(EXPERTS, 2 * INTER, HIDDEN, **bf)
        _EMPTY["fc2_w"] = torch.empty(EXPERTS, HIDDEN, INTER, **bf)
    return dict(
        x=torch.zeros(rows, HIDDEN, **bf),
        maps=allocate_maps(max(rows, 128), "cuda"),
        topk_w=torch.zeros(rows, TOP_K, device="cuda"),
        fc1_w=_EMPTY["fc1_w"],
        fc2_w=_EMPTY["fc2_w"],
        out=torch.zeros(rows, HIDDEN, **bf),
        c=allocate_c(max(rows, 128), INTER, "cuda"),
    )


def test_guard_refuses_rows_past_the_compiled_capacity() -> None:
    with pytest.raises(AssertionError, match="compiled for up to 64 rows"):
        with validating(op):
            op(**_args(128), compiled=_fake(64))


def test_guard_refuses_a_strided_output() -> None:
    """FC2 adds through a raw pointer: a column slice would be added to as dense rows."""
    args = _args(8)
    wide = torch.zeros(8, HIDDEN + 64, dtype=torch.bfloat16, device="cuda")
    args["out"] = wide[:, :HIDDEN]
    with pytest.raises(AssertionError, match="out must be contiguous"):
        with validating(op):
            op(**args, compiled=_fake(MAX_ROWS))


def test_guard_refuses_too_few_routing_weights() -> None:
    args = _args(8)
    args["topk_w"] = torch.zeros(4, TOP_K, device="cuda")
    with pytest.raises(AssertionError, match="rows \\* 8 weights"):
        with validating(op):
            op(**args, compiled=_fake(MAX_ROWS))
