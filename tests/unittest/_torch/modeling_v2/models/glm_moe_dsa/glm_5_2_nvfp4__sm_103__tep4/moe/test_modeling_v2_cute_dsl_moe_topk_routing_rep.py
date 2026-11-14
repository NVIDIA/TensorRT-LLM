# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_topk_routing_rep entry: selection and
weights equal the push top-k's, each rank's maps equal the slot reference, and across the
four EP ranks every (token, k) is mapped exactly once."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL top-k kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    MAP_KEYS,
    TOP_K,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    allocate_decode_outputs,
    build_decode,
    cute_dsl_moe_topk_routing,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_rep import (  # noqa: E402, E501
    allocate_outputs,
    canonical_slot_maps,
    reference_slot_maps,
    slot_ids,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_rep import (  # noqa: E402, E501
    build_decode as build_rep,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_rep import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing_rep as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

EXPERTS = 256
ROUTED_SCALE = 2.5
_COMPILED: dict = {}


def _rep(spec):
    key = (
        spec["rows"],
        spec["splits"],
        spec["nrep"],
        spec["signal_gg1"],
        spec.get("expert_hist", False),
    )
    if key not in _COMPILED:
        _COMPILED[key] = build_rep(
            spec["rows"], spec["splits"], spec["nrep"], torch.cuda.current_device(),
            signal_gg1=spec["signal_gg1"], expert_hist=spec.get("expert_hist", False),
        )  # fmt: skip
    return _COMPILED[key]


def _push(spec):
    key = ("push", spec["rows"], spec["splits"])
    if key not in _COMPILED:
        _COMPILED[key] = build_decode(
            spec["rows"], spec["splits"], torch.cuda.current_device(), signal_gg1=False
        )
    return _COMPILED[key]


def _case(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, splits = spec["rows"], spec["splits"]
    logits = torch.randn(rows, splits, EXPERTS, generator=g, device="cuda").contiguous() * 0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    # The served push top-k's ids pick the frequency hot list.
    out = allocate_decode_outputs(rows, "cuda")
    _, ids, _ = cute_dsl_moe_topk_routing(logits, bias, 0, ROUTED_SCALE, out, None, _push(spec))
    torch.cuda.synchronize()
    nrep = spec["nrep"]
    if spec["hot"] == "freq":
        cnt = torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS)
        hot = torch.argsort(cnt, descending=True, stable=True)[:nrep].tolist()
    elif spec["hot"] == "rand":
        hot = torch.randperm(EXPERTS, generator=g, device="cuda")[:nrep].tolist()
    else:
        hot = [-1] * nrep
    hot_t = torch.tensor(hot, dtype=torch.int32, device="cuda")
    return logits, bias, hot_t, (out["topk_weights"].clone(), ids.clone())


def _run(spec, logits, bias, hot_t, offset):
    outputs = allocate_outputs(spec["rows"], spec["nrep"], "cuda")
    for t in outputs.values():
        t.fill_(-7)
    hist = (
        torch.full((64,), -1, dtype=torch.int32, device="cuda") if spec.get("expert_hist") else None
    )
    with validating(op):
        weights, ids, maps = op(
            logits, bias, hot_t, offset, ROUTED_SCALE, outputs, _rep(spec), expert_hist=hist
        )
    torch.cuda.synchronize()
    return weights, ids, maps, hist


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    rows, nrep, offset = spec["rows"], spec["nrep"], spec["offset"]
    logits, bias, hot_t, (push_w, push_ids) = _case(spec, zlib.crc32(cell.why.encode()) % 2**31)
    weights, ids, maps, hist = _run(spec, logits, bias, hot_t, offset)
    # Selection and weights: the push top-k's, bit for bit (same arithmetic, same order).
    assert torch.equal(ids, push_ids), "selected experts differ from the push top-k"
    assert torch.equal(weights.reshape(-1), push_w.reshape(-1)[: rows * TOP_K])
    # This rank's maps against the slot reference of the kernel's own ids.
    nslots = 64 + nrep
    got = canonical_slot_maps(maps, rows * TOP_K, nslots)
    want = reference_slot_maps(slot_ids(ids, hot_t.tolist(), offset), nslots)
    for k in MAP_KEYS:
        assert torch.equal(got[k], want[k]), f"{k} differs from the slot reference"
    if hist is not None:
        local = ids[(ids >= offset) & (ids < offset + 64)].long() - offset
        assert torch.equal(hist, torch.bincount(local, minlength=64).to(torch.int32)), (
            "expert_hist is not the step's home-expert counts"
        )
    # The four ranks together: every (token, k) mapped exactly once, and with an unused
    # hot list each rank's maps are the home mapping.
    cover = torch.zeros(rows * TOP_K, dtype=torch.int32, device="cuda")
    for off in (0, 64, 128, 192):
        _, _, m, _ = _run(spec, logits, bias, hot_t, off)
        cover += (m["expanded_idx_to_permuted_idx"][: rows * TOP_K] >= 0).to(torch.int32)
        if spec["hot"] == "none":
            tiles = int(m["num_non_exiting_tiles"][0])
            assert bool((m["tile_idx_to_expert_idx"][:tiles] < 64).all())
    assert bool((cover == 1).all()), f"{int((cover != 1).sum())} (token, k) pairs not mapped once"


def test_guard_refuses_a_short_hot_list() -> None:
    spec = op.CELLS[0].spec
    logits = torch.zeros(spec["rows"], spec["splits"], EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="hot_ids"):
        with validating(op):
            op(logits, bias, torch.full((4,), -1, dtype=torch.int32, device="cuda"), 0, 1.0,
               allocate_outputs(spec["rows"], spec["nrep"], "cuda"), _rep(spec))  # fmt: skip


def test_guard_refuses_home_sized_outputs() -> None:
    """Maps sized for 64 tiles are written past their end by a replica slot."""
    spec = op.CELLS[0].spec
    logits = torch.zeros(spec["rows"], spec["splits"], EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    hot = torch.full((spec["nrep"],), -1, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="replica slots"):
        with validating(op):
            op(logits, bias, hot, 0, 1.0, allocate_decode_outputs(spec["rows"], "cuda"), _rep(spec))
