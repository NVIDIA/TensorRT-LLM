# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_prefill_topk_routing_rep entry: selection
and weights equal the prefill top-k's, each rank's maps equal the water-fill slot reference,
across the four EP ranks every (token, k) is mapped exactly once, and on a skewed step the
ranks' tile loads are levelled."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL top-k kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_prefill_topk_routing_rep import (  # noqa: E402, E501
    allocate_scratch,
    build_prefill,
    slot_ids,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_prefill_topk_routing_rep import (  # noqa: E402, E501
    cute_dsl_moe_prefill_topk_routing_rep as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    MAP_KEYS,
    TOP_K,
    allocate_maps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    PREFILL_MAX_ROWS,
    allocate_prefill_scratch,
    cute_dsl_moe_topk_routing,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    build_prefill as build_served,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_rep import (  # noqa: E402, E501
    canonical_slot_maps,
    reference_slot_maps,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

EXPERTS = 256
ROUTED_SCALE = 2.5
_COMPILED: dict = {}


def _rep(nrep: int):
    if nrep not in _COMPILED:
        _COMPILED[nrep] = build_prefill(nrep, torch.cuda.current_device())
    return _COMPILED[nrep]


def _served():
    if "served" not in _COMPILED:
        _COMPILED["served"] = build_served(torch.cuda.current_device())
    return _COMPILED["served"]


def _outputs(rows: int, nrep: int | None) -> dict:
    """One output set at the token budget's capacity, poisoned; `nrep` None: the served
    prefill top-k's set."""
    i32 = {"dtype": torch.int32, "device": "cuda"}
    out = {
        "topk_weights": torch.zeros(rows * TOP_K, dtype=torch.float32, device="cuda"),
        "topk_indices": torch.full((rows * TOP_K,), -7, **i32),
        **(allocate_prefill_scratch("cuda") if nrep is None else allocate_scratch(nrep, "cuda")),
        **allocate_maps(PREFILL_MAX_ROWS, "cuda", replicas=nrep or 0),
    }
    for k in MAP_KEYS:
        out[k].fill_(-7)
    return out


def _case(spec, seed):
    """Logits, bias, the hot list and the served prefill top-k's `(weights, ids, maps)`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, nrep = spec["rows"], spec["nrep"]
    logits = torch.randn(rows, 1, EXPERTS, generator=g, device="cuda")
    if spec.get("skew"):
        # Rank 1's experts busier, four of them very busy: the skew the replicas level.
        logits[:, :, 64:128] += 0.6
        logits[:, :, 70:74] += 2.0
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.05
    served = _outputs(rows, None)
    w, ids, maps = cute_dsl_moe_topk_routing(
        logits, bias, spec["offset"], ROUTED_SCALE, served, None, _served()
    )
    torch.cuda.synchronize()
    if spec["hot"] == "freq":
        cnt = torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS)
        hot = torch.argsort(cnt, descending=True, stable=True)[:nrep].tolist()
    elif spec["hot"] == "rand":
        hot = torch.randperm(EXPERTS, generator=g, device="cuda")[:nrep].tolist()
        hot[-1] = -1
    else:
        hot = [-1] * nrep
    hot_t = torch.tensor(hot, dtype=torch.int32, device="cuda")
    return logits, bias, hot_t, (w.clone(), ids.clone(), maps)


def _run(spec, logits, bias, hot_t, offset):
    outputs = _outputs(spec["rows"], spec["nrep"])
    with validating(op):
        weights, ids, maps = op(
            logits, bias, hot_t, offset, ROUTED_SCALE, outputs, _rep(spec["nrep"])
        )
    torch.cuda.synchronize()
    return weights, ids, maps


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    rows, nrep, offset = spec["rows"], spec["nrep"], spec["offset"]
    logits, bias, hot_t, (ref_w, ref_ids, _) = _case(spec, zlib.crc32(cell.why.encode()) % 2**31)
    weights, ids, maps = _run(spec, logits, bias, hot_t, offset)
    # Selection and weights: the served prefill top-k's, bit for bit.
    assert torch.equal(ids, ref_ids), "selected experts differ from the prefill top-k"
    assert torch.equal(weights.view(torch.int32), ref_w.view(torch.int32))
    # This rank's maps against the water-fill slot reference of the kernel's own ids.
    nslots = 64 + nrep
    got = canonical_slot_maps(maps, rows * TOP_K, nslots)
    want = reference_slot_maps(slot_ids(ids, hot_t.tolist(), offset), nslots)
    for k in MAP_KEYS:
        assert torch.equal(got[k], want[k]), f"{k} differs from the slot reference"
    # The four ranks together: every (token, k) mapped exactly once; with an unused hot
    # list each rank's maps are the home mapping (the served kernel's tiles).
    cover = torch.zeros(rows * TOP_K, dtype=torch.int32, device="cuda")
    tiles, home = [], []
    for off in (0, 64, 128, 192):
        _, _, m = _run(spec, logits, bias, hot_t, off)
        cover += (m["expanded_idx_to_permuted_idx"][: rows * TOP_K] >= 0).to(torch.int32)
        tiles.append(int(m["num_non_exiting_tiles"][0]))
        served = _outputs(rows, None)
        cute_dsl_moe_topk_routing(logits, bias, off, ROUTED_SCALE, served, None, _served())
        torch.cuda.synchronize()
        home.append(int(served["num_non_exiting_tiles"][0]))
    assert bool((cover == 1).all()), f"{int((cover != 1).sum())} (token, k) pairs not mapped once"
    if spec["hot"] == "none":
        assert tiles == home, f"unused hot list: tiles {tiles} != the home mapping's {home}"
    if spec.get("skew"):
        # Levelled: the busiest rank's tile count drops below the home mapping's.
        assert max(tiles) < max(home), f"tiles {tiles} not levelled against home {home}"


def test_guard_refuses_a_short_hot_list() -> None:
    logits = torch.zeros(513, 1, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="hot_ids"):
        with validating(op):
            op(logits, bias, torch.full((4,), -1, dtype=torch.int32, device="cuda"), 0, 1.0,
               _outputs(513, 8), _rep(8))  # fmt: skip


def test_guard_refuses_home_sized_maps() -> None:
    """Maps sized without the replica slots' tiles are written past their end."""
    logits = torch.zeros(PREFILL_MAX_ROWS, 1, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    outputs = _outputs(PREFILL_MAX_ROWS, 8)
    outputs.update(allocate_maps(PREFILL_MAX_ROWS, "cuda"))
    with pytest.raises(AssertionError, match="replica slots write"):
        with validating(op):
            op(logits, bias, torch.full((8,), -1, dtype=torch.int32, device="cuda"), 0, 1.0,
               outputs, _rep(8))  # fmt: skip


def test_guard_refuses_the_four_warp_rows() -> None:
    """65..128 rows select on the 4-warp histogram, for which no replica kernel is built:
    every rank keeps the home mapping there."""
    logits = torch.zeros(128, 1, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="serves 129"):
        with validating(op):
            op(logits, bias, torch.full((8,), -1, dtype=torch.int32, device="cuda"), 0, 1.0,
               _outputs(128, 8), _rep(8))  # fmt: skip
