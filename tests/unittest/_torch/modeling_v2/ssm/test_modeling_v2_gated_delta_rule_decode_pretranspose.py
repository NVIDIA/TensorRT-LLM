# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_delta_rule_decode_pretranspose catalog entry (FlashInfer fp32-pool decode)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.gated_delta_rule_decode_pretranspose import (  # noqa: E501
    QWEN38_GEOMETRY,
    _scattered,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.gated_delta_rule_decode_pretranspose import (  # noqa: E501
    gated_delta_rule_decode_pretranspose as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gated_delta_rule_decode_pretranspose requires a CUDA device"

DEV = "cuda"
_PAD = 64


def _compare(out, ref) -> None:
    """`(o, pool)` against the fp64 reference rounded once, at torch's default
    dtype-aware band. Measured on the microbench inputs: o max abs 1.2e-4 at
    |o| <= 6.2e-2 (identical to the Triton op's), state max abs 2.6e-7. The
    pool is compared whole, so slots the call did not name are gated too."""
    (o, pool), (ref_o, ref_pool) = out, ref
    assert o.dtype == ref_o.dtype and o.shape == ref_o.shape, (
        (o.dtype, o.shape),
        (ref_o.dtype, ref_o.shape),
    )
    assert torch.isfinite(o).all()
    torch.testing.assert_close(o, ref_o)
    assert pool.shape == ref_pool.shape and pool.dtype == ref_pool.dtype
    torch.testing.assert_close(pool, ref_pool)


def _build(spec, seed):
    """`__call__`'s arguments (by name) plus the buffers some layouts view."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, HV, K, V = spec["geometry"]
    m, slots = spec["m"], spec["slots"]
    mixed = torch.randn(m, 2 * H * K + HV * V, generator=g, device=DEV, dtype=torch.bfloat16)
    qv, kv, vv = torch.split(mixed, [H * K, H * K, HV * V], 1)
    q, k, v = (
        torch.reshape(qv, [m, 1, H, K]),
        torch.reshape(kv, [m, 1, H, K]),
        torch.reshape(vv, [m, 1, HV, V]),
    )
    if spec.get("qkv", "views") == "contiguous":
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    a = torch.randn(m, HV, generator=g, device=DEV, dtype=torch.bfloat16).view(m, 1, HV)
    b = torch.randn(m, HV, generator=g, device=DEV, dtype=torch.bfloat16).view(m, 1, HV)
    A_log = torch.empty(HV, device=DEV).uniform_(-5.5, 1.0, generator=g)
    dt_bias = torch.randn(HV, generator=g, device=DEV) * 4.0
    layout = spec.get("pool", "contiguous")
    extras = {}
    if layout == "padded":
        backing = torch.randn(slots, HV * V * K + _PAD, generator=g, device=DEV) * 0.05
        pool = backing[:, : HV * V * K].view(slots, HV, V, K)
        extras["backing"] = backing
    elif layout == "layer_view":
        cache = torch.randn(3, slots, HV, V, K, generator=g, device=DEV) * 0.05
        pool = cache[1]
        extras["cache"] = cache
    else:
        pool = torch.randn(slots, HV, V, K, generator=g, device=DEV) * 0.05
    indices = spec.get("indices") or _scattered(m, slots)
    indices = torch.tensor(indices, dtype=torch.int32, device=DEV)
    output = None
    if spec.get("output", "buffer") == "buffer":
        output = torch.full((m, 1, HV, V), -7.0, device=DEV, dtype=torch.bfloat16)
    args = dict(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=None,
        output=output,
        use_qk_l2norm=True,
        initial_state=pool,
        initial_state_indices=indices,
        output_state_indices=None,
    )
    return args, extras


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    pool = args["initial_state"]
    pool_before = pool.clone()
    cache_before = extras["cache"].clone() if "cache" in extras else None
    slot_numel = pool[0].numel()
    pad_before = extras["backing"][:, slot_numel:].clone() if "backing" in extras else None
    ref = op.reference(**args)

    o, returned_pool = op(**args)

    assert returned_pool.data_ptr() == pool.data_ptr(), "the pool is advanced in place"
    _compare((o, pool), ref)
    if args["output"] is not None:
        assert o.data_ptr() == args["output"].data_ptr(), "output= must be written in place"
    named = {s for s in args["initial_state_indices"].tolist() if s >= 0}
    unnamed = [s for s in range(pool.shape[0]) if s not in named]
    assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
    for slot in named:
        assert not torch.equal(pool[slot], pool_before[slot]), slot
    if cache_before is not None:
        for other in (0, 2):
            assert torch.equal(extras["cache"][other], cache_before[other]), other
    if pad_before is not None:
        assert torch.equal(extras["backing"][:, slot_numel:], pad_before), "slot padding written"


def test_repeated_calls_are_bitwise() -> None:
    """Two identical calls from the same start give identical output and state."""
    results = []
    for _ in range(2):
        args, _ = _build(dict(geometry=QWEN38_GEOMETRY, m=8, slots=40), seed=5)
        o, pool = op(**args)
        results.append((o.clone(), pool.clone()))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])


def test_matches_in_tree_triton_op() -> None:
    """Same inputs through the in-tree Triton decode op this entry stands in
    for: output and state agree at the default band (each is within it of fp64)."""
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as triton_op,
    )

    H, HV, K, V = QWEN38_GEOMETRY
    m = 32
    args, _ = _build(dict(geometry=QWEN38_GEOMETRY, m=m, slots=461), seed=11)
    pool_t = args["initial_state"].clone()
    cu = torch.arange(m + 1, device=DEV, dtype=torch.int64)
    o_t = triton_op(
        A_log=args["A_log"],
        a=args["a"].view(m, HV),
        dt_bias=args["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=torch.reshape(args["q"], [1, m, H, K]),
        k=torch.reshape(args["k"], [1, m, H, K]),
        v=torch.reshape(args["v"], [1, m, HV, V]),
        b=args["b"].view(m, HV),
        initial_state_source=pool_t,
        initial_state_indices=args["initial_state_indices"],
        scale=None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        output=None,
    )
    o, pool = op(**args)
    torch.testing.assert_close(o.view(1, m, HV, V), o_t)
    torch.testing.assert_close(pool, pool_t)
