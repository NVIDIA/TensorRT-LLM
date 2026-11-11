# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_topk_routing entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL top-k kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    MAP_KEYS,
    TOP_K,
    allocate_maps,
    canonical_maps,
    check_maps,
    reference_maps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    PREFILL_MAX_ROWS,
    CompiledTopkRouting,
    allocate_decode_outputs,
    allocate_prefill_scratch,
    build_decode,
    build_prefill,
    selection_margin,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

EXPERTS, HIDDEN = 256, 6144
ROUTED_SCALE = 2.5
#: Rows whose 8th and 9th selection scores sit closer than this (relative) may pick
#: either expert by the last bit of the sigmoid; see `selection_margin`.
TIE_MARGIN = 2.0**-20
#: Weights: the kernel normalizes in fp32 (a five-step butterfly sum of eight values,
#: one multiply, one divide) where the reference does it in fp64, and both take the
#: sigmoid's tanh form in fp32. At most ~6 fp32 roundings separate them, 2**-21 relative;
#: the band is 2**-19. A wrong normalization or a biased weight misses by 1e-2.
WEIGHT_RTOL = 2.0**-19

_COMPILED: dict = {}


def _compiled(spec) -> CompiledTopkRouting:
    rows = spec["rows"]
    if rows > 64:
        key = "prefill"
        if key not in _COMPILED:
            _COMPILED[key] = build_prefill(torch.cuda.current_device())
        return _COMPILED[key]
    key = (rows, spec["splits"], spec["signal_gg1"], spec.get("fuse_zero", False))
    if key not in _COMPILED:
        _COMPILED[key] = build_decode(
            rows,
            spec["splits"],
            torch.cuda.current_device(),
            signal_gg1=spec["signal_gg1"],
            fuse_zero=spec.get("fuse_zero", False),
        )
    return _COMPILED[key]


def _inputs(spec, g):
    """Logits as each router writes them (`[rows, 256]`, or four K partials), and a bias
    wide enough to skew selection so experts take several tiles at the large counts."""
    rows = spec["rows"]
    if rows > 64:
        logits = torch.randn(rows, EXPERTS, generator=g, device="cuda")
    elif spec["splits"] == 1:
        logits = torch.randn(rows, EXPERTS, generator=g, device="cuda")
    else:
        logits = torch.randn(rows, spec["splits"], EXPERTS, generator=g, device="cuda") * 0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    return logits.contiguous(), bias


def _outputs(rows: int):
    if rows <= 64:
        out = allocate_decode_outputs(rows, "cuda")
    else:
        # Sized for the token budget, as the target's slot buffers are.
        out = {
            "topk_weights": torch.empty(PREFILL_MAX_ROWS, TOP_K, device="cuda"),
            "topk_indices": torch.empty(PREFILL_MAX_ROWS * TOP_K, dtype=torch.int32, device="cuda"),
            **allocate_prefill_scratch("cuda"),
            **allocate_maps(PREFILL_MAX_ROWS, "cuda"),
        }
    for t in out.values():
        t.fill_(-7)  # an entry the kernel had to write and did not reads as -7
    return out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Ids as sets per row (a row's eight winners in any order), weights matched by id
    within `WEIGHT_RTOL`, and the maps exact in canonical form.

    Selection is exact except where the reference's 8th and 9th scores are within
    `TIE_MARGIN`: there the kernel's sigmoid and torch's may differ in the last bit,
    which no reference fixes. Those rows are exempt and must be rare. The maps are
    checked against `reference_maps` of the kernel's own ids (and against the op's
    reference outright when no row was exempt), with `check_maps` on the inverse.
    """
    spec = cell.spec
    rows, offset = spec["rows"], spec["offset"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    logits, bias = _inputs(spec, g)
    outputs = _outputs(rows)
    zero = None
    if spec.get("fuse_zero"):
        zero = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    compiled = _compiled(spec)
    with validating(op):
        weights, ids, maps = op(logits, bias, offset, ROUTED_SCALE, outputs, zero, compiled)
    ref_w, ref_ids, ref_maps = op.reference(
        logits, bias, offset, ROUTED_SCALE, outputs, zero, compiled
    )

    exempt = selection_margin(logits, bias) < TIE_MARGIN
    assert int(exempt.sum()) <= max(1, rows // 1000), f"{int(exempt.sum())} near-tie rows"
    keep = ~exempt
    got_ids, order = torch.sort(ids, dim=-1)
    want_ids, want_order = torch.sort(ref_ids, dim=-1)
    assert torch.equal(got_ids[keep], want_ids[keep]), "selected experts differ"
    got_w = torch.gather(weights, 1, order)
    want_w = torch.gather(ref_w, 1, want_order)
    torch.testing.assert_close(got_w[keep], want_w[keep], rtol=WEIGHT_RTOL, atol=0.0)

    check_maps(maps, ids, offset)
    got = canonical_maps(maps, rows * TOP_K)
    own = reference_maps(ids, offset)
    for k in MAP_KEYS:
        assert torch.equal(got[k], own[k]), f"{k} does not describe the kernel's own ids"
        if not bool(exempt.any()):
            assert torch.equal(got[k], ref_maps[k]), f"{k} differs from the reference"
    if zero is not None:
        assert bool((zero == 0).all()), "the fused zero plane was not cleared"


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _fake(kind: str, rows, splits: int = 1, fuse_zero: bool = False) -> CompiledTopkRouting:
    return CompiledTopkRouting(kind, rows, splits, False, fuse_zero, _unreachable)


def test_guard_refuses_another_row_count() -> None:
    """A decode handle is exact-row; its outputs are sized for those rows."""
    logits = torch.zeros(5, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="compiled for 4 rows"):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(5, "cuda"), None, _fake("decode", 4))


def test_guard_refuses_partials_for_a_one_plane_handle() -> None:
    """Four K partials read as one plane would sum a quarter of the product."""
    logits = torch.zeros(8, 4, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="logit plane"):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(8, "cuda"), None, _fake("decode", 8))


def test_guard_refuses_outputs_sized_for_other_rows() -> None:
    logits = torch.zeros(4, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="for 4 rows"):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(2, "cuda"), None, _fake("decode", 4))


def test_guard_refuses_a_bf16_bias() -> None:
    logits = torch.zeros(4, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(4, "cuda"), None, _fake("decode", 4))


def test_guard_refuses_a_zero_plane_for_the_prefill_kernel() -> None:
    logits = torch.zeros(65, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    zero = torch.zeros(65, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="no zero plane"):
        with validating(op):
            op(logits, bias, 0, 1.0, _outputs(65), zero, _fake("prefill", None))
