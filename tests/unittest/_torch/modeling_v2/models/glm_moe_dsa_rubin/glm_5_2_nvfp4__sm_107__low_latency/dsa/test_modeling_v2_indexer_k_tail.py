# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM indexer_k_tail entry.

The K cache is real: an INDEX_KEY pool of 64-token pages ([pages, 64, 1, 132] uint8) filled with a
sentinel, a random block-table row for the request and the metadata's slot mappings. The cache is
checked in two ways: bitwise against the op's own K outputs scattered by the reference's byte
placement (so every byte the op wrote is where it belongs, and nothing else moved), and through
those outputs against the reference arithmetic.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa import (  # noqa: E501
    _indexer_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa.indexer_k_tail import (  # noqa: E501
    indexer_k_tail as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
import _indexer_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "indexer_k_tail requires a CUDA device"


def _build(spec, seed: int) -> dict:
    """The op's arguments for one cell. The whole pool allocation rides along as ``_pool``."""
    gen = torch.Generator().manual_seed(seed)
    rows, d = spec["rows"], spec["head_dim"]
    positions = torch.arange(spec["first_position"], spec["first_position"] + rows)
    if spec["k_layout"] == "fused_a":
        fused = torch.randn(rows, spec["fused_a_width"], generator=gen).to(torch.bfloat16).cuda()
        indexer_k = fused[:, spec["k_col"] : spec["k_col"] + d]
        assert not indexer_k.is_contiguous()
    else:
        indexer_k = torch.randn(rows, d, generator=gen).to(torch.bfloat16).cuda()
    ln_weight = (1.0 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16).cuda()
    ln_bias = (0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16).cuda()
    cache, whole = inputs.k_cache_pool(spec["pages"], spec["pool_layout"])
    row = inputs.block_row(positions, spec["pages"], gen)
    slot_fp8, slot_scale = inputs.slot_mappings(positions, row)
    if spec["negative_slot"]:
        slot_fp8[2] = -1
    return dict(
        _pool=whole,
        indexer_k=indexer_k,
        positions=positions.to(torch.int32).cuda(),
        cos_sin=inputs.cos_sin_table(int(positions[-1]) + 1, spec["rope_dim"]),
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        eps=spec["eps"],
        rope_dim=spec["rope_dim"],
        interleave=spec["interleave"],
        use_ue8m0=spec["use_ue8m0"],
        k_fp8=torch.empty(rows, d, dtype=torch.float8_e4m3fn, device="cuda"),
        k_scale=torch.empty(rows, 1, dtype=torch.float32, device="cuda"),
        k_cache=cache,
        slot_mapping_fp8=slot_fp8,
        slot_mapping_scale=slot_scale,
        num_cache_tokens=spec["cache_rows"],
        use_pdl=spec["use_pdl"],
        launch_dependents=spec["launch_dependents"],
        prewait=spec["prewait"],
    )


def _check(args: dict, out, ref, cache_before: torch.Tensor, whole_before=None, whole=None):
    k8, k_scale, cache = out
    k8_ref, k_scale_ref, _ = ref
    # The arithmetic: within one E4M3 step of the reference, almost every code exact.
    inputs.assert_quantized_close(k8, k_scale, k8_ref, k_scale_ref)
    # The placement: the op's own rows, scattered by the reference rule into the cache as it was,
    # are the cache byte for byte.
    expected = cache_before.clone()
    _indexer_reference.write_cache_rows(
        expected,
        k8,
        k_scale,
        args["slot_mapping_fp8"],
        args["slot_mapping_scale"],
        args["num_cache_tokens"],
    )
    assert torch.equal(cache.view(torch.uint8), expected.view(torch.uint8)), "cache bytes differ"
    if whole is not None:
        # The other layers' pages of an interleaved pool are untouched.
        assert torch.equal(whole[:, 0], whole_before[:, 0])
        assert torch.equal(whole[:, 2], whole_before[:, 2])


def _op_args(args: dict) -> dict:
    return {k: v for k, v in args.items() if not k.startswith("_")}


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    whole = args.pop("_pool") if cell.spec["pool_layout"] == "interleaved" else None
    args = _op_args(args)
    cache_before = args["k_cache"].clone()
    whole_before = whole.clone() if whole is not None else None
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert out[0] is args["k_fp8"] and out[1] is args["k_scale"] and out[2] is args["k_cache"]
    _check(args, out, ref, cache_before, whole_before, whole)


def test_rows_past_num_cache_tokens_stay_out_of_the_cache():
    """Rows at and past ``num_cache_tokens`` get K outputs and no cache bytes."""
    spec = dict(op.CELLS[0].spec, cache_rows=2)
    args = _op_args(_build(spec, seed=11))
    before = args["k_cache"].clone()
    op(**args)
    torch.cuda.synchronize()
    changed = (args["k_cache"].view(torch.uint8) != before.view(torch.uint8)).sum().item()
    # Two rows: 128 FP8 bytes and 4 scale bytes each, at most (a value may equal the sentinel).
    assert 0 < changed <= 2 * (128 + 4)


def test_cuda_graph_replay_follows_new_inputs():
    """The decode path is captured: a replay reads the K rows, positions and slots in place."""
    spec = op.CELLS[0].spec
    args = _op_args(_build(spec, seed=21))
    op(**args)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)
    # Positions inside the captured cos/sin table, rows of another request map.
    fresh = _build(dict(spec, first_position=spec["first_position"] - 8), seed=22)
    args["indexer_k"].copy_(fresh["indexer_k"])
    args["positions"].copy_(fresh["positions"])
    args["slot_mapping_fp8"].copy_(fresh["slot_mapping_fp8"])
    args["slot_mapping_scale"].copy_(fresh["slot_mapping_scale"])
    cache_before = args["k_cache"].clone()
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _check(args, (args["k_fp8"], args["k_scale"], args["k_cache"]), ref, cache_before)


def test_guard_refuses_a_wider_cos_sin_table():
    args = _op_args(_build(op.CELLS[0].spec, seed=32))
    wide = torch.cat([args["cos_sin"], args["cos_sin"]], dim=1)
    with pytest.raises(AssertionError, match="cos_sin rows"):
        with validating(op):
            op(**dict(args, cos_sin=wide))


def test_guard_refuses_a_position_past_the_table():
    args = _op_args(_build(op.CELLS[0].spec, seed=33))
    positions = args["positions"].clone()
    positions[-1] = args["cos_sin"].shape[0]
    with pytest.raises(AssertionError, match="past the cos/sin table"):
        with validating(op):
            op(**dict(args, positions=positions))


def test_guard_refuses_a_strided_k_output():
    args = _op_args(_build(op.CELLS[0].spec, seed=34))
    wide = torch.empty(args["k_fp8"].shape[0], 256, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(**dict(args, k_fp8=wide[:, :128]))


def test_guard_refuses_a_slot_past_the_cache():
    args = _op_args(_build(op.CELLS[0].spec, seed=35))
    slots = args["slot_mapping_fp8"].clone()
    slots[0] = args["k_cache"].numel() - 64
    with pytest.raises(AssertionError, match="writes past the cache"):
        with validating(op):
            op(**dict(args, slot_mapping_fp8=slots))


def test_guard_refuses_more_cache_rows_than_k_rows():
    args = _op_args(_build(op.CELLS[0].spec, seed=36))
    with pytest.raises(AssertionError, match="num_cache_tokens"):
        with validating(op):
            op(**dict(args, num_cache_tokens=args["indexer_k"].shape[0] + 1))
