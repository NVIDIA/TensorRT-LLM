# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM indexer_prologue entry.

Q is the wq_b output ([N, 32 * 128] bf16); K and the weights are column views of a fused-A output
row block, as the target hands them over. The cache is the engine's INDEX_KEY layout (see
``_indexer_inputs``); its bytes are checked against the op's own K rows scattered by the reference
rule, the outputs against the reference arithmetic.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa import (  # noqa: E501
    _indexer_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa.indexer_prologue import (  # noqa: E501
    indexer_prologue as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
import _indexer_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "indexer_prologue requires a CUDA device"


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    rows, heads, d = spec["rows"], spec["n_heads"], spec["head_dim"]
    positions = torch.arange(spec["first_position"], spec["first_position"] + rows)
    q = torch.randn(rows, heads * d, generator=gen).to(torch.bfloat16).cuda()
    if spec["k_layout"] == "fused_a":
        fused = torch.randn(rows, spec["fused_a_width"], generator=gen).to(torch.bfloat16).cuda()
        indexer_k = fused[:, spec["k_col"] : spec["k_col"] + d]
        weights = fused[:, spec["k_col"] + d : spec["k_col"] + d + heads]
    else:
        indexer_k = torch.randn(rows, d, generator=gen).to(torch.bfloat16).cuda()
        weights = torch.randn(rows, heads, generator=gen).to(torch.bfloat16).cuda()
    cache = slot_fp8 = slot_scale = None
    if spec["write_cache"]:
        cache, _ = inputs.k_cache_pool(spec["pages"], spec["pool_layout"])
        row = inputs.block_row(positions, spec["pages"], gen)
        slot_fp8, slot_scale = inputs.slot_mappings(positions, row)
    return dict(
        indexer_k=indexer_k,
        weights=weights,
        q=q.view(rows, heads, d),
        positions=positions.to(torch.int32).cuda(),
        cos_sin=inputs.cos_sin_table(int(positions[-1]) + 1, spec["rope_dim"]),
        ln_weight=(1.0 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16).cuda(),
        ln_bias=(0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16).cuda(),
        eps=spec["eps"],
        weight_scale=spec["weight_scale"],
        rope_dim=spec["rope_dim"],
        interleave=spec["interleave"],
        use_ue8m0=spec["use_ue8m0"],
        k_cache=cache,
        slot_mapping_fp8=slot_fp8,
        slot_mapping_scale=slot_scale,
        num_cache_tokens=spec["cache_rows"] if spec["write_cache"] else 0,
        use_pdl=spec["use_pdl"],
        launch_dependents=spec["launch_dependents"],
        prewait=spec["prewait"],
    )


def _check(args, out, ref, cache_before):
    q8, k8, k_scale, w, q_scale = out
    q8_ref, k8_ref, k_scale_ref, w_ref, q_scale_ref, _ = ref
    rows, heads, d = args["q"].shape
    assert q8.shape == (rows, heads, d) and q_scale.shape == (rows, heads, 1)
    assert w.shape == (rows, heads) and w.dtype == torch.float32
    inputs.assert_quantized_close(q8, q_scale, q8_ref, q_scale_ref)
    inputs.assert_quantized_close(k8, k_scale, k8_ref, k_scale_ref)
    # weights = (w * q_scale) * weight_scale, two fp32 roundings: dividing the Q scale back out
    # leaves the input weight times the constant within two ulps whichever scale either side
    # picked.
    torch.testing.assert_close(
        w / q_scale[..., 0],
        w_ref / q_scale_ref[..., 0],
        rtol=2 * 2.0**-23,
        atol=0.0,
    )
    if args["k_cache"] is not None:
        expected = cache_before.clone()
        _indexer_reference.write_cache_rows(
            expected,
            k8,
            k_scale,
            args["slot_mapping_fp8"],
            args["slot_mapping_scale"],
            args["num_cache_tokens"],
        )
        assert torch.equal(args["k_cache"], expected), "cache bytes differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    cache_before = args["k_cache"].clone() if args["k_cache"] is not None else None
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    _check(args, out, ref, cache_before)


def test_q_part_matches_the_k_tail_launch_of_the_same_kernel():
    """The K part does not depend on whether the Q part is compiled in: the whole prologue and the
    K tail write the same K bytes for the same inputs."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa.indexer_k_tail import (  # noqa: E501
        indexer_k_tail,
    )

    args = _build(op.CELLS[0].spec, seed=41)
    _, k8, k_scale, _, _ = op(**dict(args, k_cache=None, num_cache_tokens=0))
    tail_k8 = torch.empty_like(k8)
    tail_scale = torch.empty_like(k_scale)
    indexer_k_tail(
        indexer_k=args["indexer_k"],
        positions=args["positions"],
        cos_sin=args["cos_sin"],
        ln_weight=args["ln_weight"],
        ln_bias=args["ln_bias"],
        eps=args["eps"],
        rope_dim=args["rope_dim"],
        interleave=args["interleave"],
        use_ue8m0=args["use_ue8m0"],
        k_fp8=tail_k8,
        k_scale=tail_scale,
        k_cache=args["k_cache"],
        slot_mapping_fp8=args["slot_mapping_fp8"],
        slot_mapping_scale=args["slot_mapping_scale"],
        num_cache_tokens=args["num_cache_tokens"],
        use_pdl=args["use_pdl"],
        launch_dependents=args["launch_dependents"],
        prewait=args["prewait"],
    )
    torch.cuda.synchronize()
    assert torch.equal(k8.view(torch.uint8), tail_k8.view(torch.uint8))
    assert torch.equal(k_scale, tail_scale)


def test_guard_refuses_weights_of_another_head_count():
    args = _build(op.CELLS[0].spec, seed=42)
    with pytest.raises(AssertionError, match="weights must be"):
        with validating(op):
            op(**dict(args, weights=args["weights"][:, :16]))


def test_guard_refuses_a_cache_write_without_slots():
    args = _build(op.CELLS[0].spec, seed=43)
    with pytest.raises(AssertionError, match="both slot mappings"):
        with validating(op):
            op(**dict(args, slot_mapping_scale=None))


def test_guard_refuses_a_non_unit_inner_stride_q():
    args = _build(op.CELLS[0].spec, seed=44)
    rows, heads, d = args["q"].shape
    strided = torch.empty(rows, heads, 2 * d, dtype=torch.bfloat16, device="cuda")[..., ::2]
    with pytest.raises(AssertionError, match="unit inner stride"):
        with validating(op):
            op(**dict(args, q=strided))
