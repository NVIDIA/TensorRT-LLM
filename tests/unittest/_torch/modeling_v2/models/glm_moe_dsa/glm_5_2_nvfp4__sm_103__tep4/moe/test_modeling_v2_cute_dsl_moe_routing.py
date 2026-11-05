# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_routing entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL routing kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    MAP_KEYS,
    ROW_COUNTS,
    TOP_K,
    CompiledRouting,
    allocate_maps,
    allocate_workspace,
    build,
    canonical_maps,
    check_maps,
    pad_expert_id,
    schedule_bucket,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    cute_dsl_moe_routing as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

EXPERTS = 256

_COMPILED: dict[int, CompiledRouting] = {}


def _compiled(rows: int) -> CompiledRouting:
    """One compile per schedule bucket per session; the target builds all of `ROW_COUNTS`
    at once, which is the same functions keyed the same way."""
    bucket = schedule_bucket(rows)
    if bucket not in _COMPILED:
        _COMPILED[bucket] = build([rows], torch.cuda.current_device())
    return _COMPILED[bucket]


def _ids(rows: int, pad: int, offset: int, g: torch.Generator) -> torch.Tensor:
    """Distinct top-8 global ids per row, skewed by a bias so some local experts take
    several 128-row tiles at the larger row counts, as the checkpoint's correction bias
    does; the last `pad` rows carry `pad_expert_id`, as the target pads them."""
    bias = torch.randn(EXPERTS, generator=g, device="cuda")
    choice = torch.randn(rows, EXPERTS, generator=g, device="cuda") * 0.3 + bias
    ids = torch.topk(choice, TOP_K, dim=-1).indices.to(torch.int32)
    if pad:
        ids[rows - pad :] = pad_expert_id(offset)
    return ids.contiguous()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """The maps are integers, so the gate is exact -- in canonical form.

    The kernels claim a row inside an expert's tiles by atomics, so the order of an
    expert's rows is unspecified; `canonical_maps` re-sorts each expert's rows and
    rebuilds both index maps from the forward map, and `check_maps` proves the kernel's
    own inverse map agrees with its forward map. Tile list, limits and counts compare
    as written. A padding row must vanish: no tile, inverse -1.
    """
    rows, pad, offset = cell.spec["rows"], cell.spec.get("pad", 0), cell.spec["offset"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    ids = _ids(rows, pad, offset, g)
    # The target stages the ids into a flat slot buffer before the call.
    staged = torch.empty(rows * TOP_K, dtype=torch.int32, device="cuda")
    staged.copy_(ids.reshape(-1))
    maps = allocate_maps(rows, "cuda")
    for t in maps.values():
        t.fill_(-7)  # an entry the kernel had to write and did not reads as -7
    counts, barrier = allocate_workspace("cuda")
    compiled = _compiled(rows)
    with validating(op):
        result = op(staged, offset, maps, counts, barrier, compiled)
    assert result is maps, "the op writes `maps` and returns it"
    ref = op.reference(staged, offset, maps, counts, barrier, compiled)

    check_maps(maps, staged, offset)
    got = canonical_maps(maps, rows * TOP_K)
    for k in MAP_KEYS:
        assert torch.equal(got[k], ref[k]), f"{k} differs from the reference"
    if pad:
        pad_rows = torch.arange((rows - pad) * TOP_K, rows * TOP_K, device="cuda")
        assert bool((maps["expanded_idx_to_permuted_idx"][pad_rows] == -1).all())


def test_row_counts_cover_every_reachable_bucket() -> None:
    """One compiled dict serves the trunk and the MTP layer; a bucket missing from it
    would be a KeyError on the hot path. Reachable: the trunk's decode widths padded to
    a power of two, its context steps padded to a power of two below 513 and exact
    above, and the MTP layer's context steps padded to a power of two."""
    built = {schedule_bucket(r) for r in ROW_COUNTS}

    def pow2(n: int) -> int:
        return 1 << max(0, (n - 1).bit_length())

    reachable = {schedule_bucket(pow2(r)) for r in range(1, 513)}
    reachable |= {schedule_bucket(r) for r in range(513, 8193)}
    reachable |= {schedule_bucket(pow2(r)) for r in range(65, 8193)}
    assert reachable <= built, f"buckets {sorted(reachable - built)} have no compiled schedule"


def _ok(rows: int = 8):
    staged = torch.zeros(rows * TOP_K, dtype=torch.int32, device="cuda")
    counts, barrier = allocate_workspace("cuda")
    return staged, allocate_maps(rows, "cuda"), counts, barrier


def _unreachable_compiled() -> CompiledRouting:
    def unreachable(*args):
        raise AssertionError("the guard should have refused before the kernel ran")

    return CompiledRouting(fns={schedule_bucket(r): unreachable for r in ROW_COUNTS})


def test_guard_refuses_maps_too_small_for_the_rows() -> None:
    """Every extent is runtime: a map set sized for 4 rows is written past its end."""
    staged, _, counts, barrier = _ok(64)
    small = allocate_maps(4, "cuda")
    with pytest.raises(AssertionError, match="rows can write"):
        with validating(op):
            op(staged, 0, small, counts, barrier, _unreachable_compiled())


def test_guard_refuses_strided_ids() -> None:
    """A strided view is read as a dense buffer."""
    _, maps, counts, barrier = _ok(8)
    wide = torch.zeros(8, 2 * TOP_K, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="contiguous int32"):
        with validating(op):
            op(wide[:, :TOP_K], 0, maps, counts, barrier, _unreachable_compiled())


def test_guard_refuses_an_offset_that_is_not_a_window() -> None:
    staged, maps, counts, barrier = _ok(8)
    with pytest.raises(AssertionError, match="one EP4 window"):
        with validating(op):
            op(staged, 32, maps, counts, barrier, _unreachable_compiled())


def test_guard_refuses_a_bucket_that_was_not_built() -> None:
    """The op never compiles: a missing bucket is the caller's build list's bug."""
    staged, maps, counts, barrier = _ok(8)
    with pytest.raises(AssertionError, match="no compiled schedule"):
        with validating(op):
            op(staged, 0, maps, counts, barrier, CompiledRouting(fns={}))
