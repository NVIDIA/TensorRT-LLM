# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_chunk_gated_delta_rule entry (FlashInfer GDN prefill)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.flashinfer_chunk_gated_delta_rule import (  # noqa: E501
    QWEN38_GEOMETRY,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.flashinfer_chunk_gated_delta_rule import (  # noqa: E501
    flashinfer_chunk_gated_delta_rule as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_chunk_gated_delta_rule requires a CUDA device"

DEV = "cuda"
#: The band the Triton chunk op is certified at (rtol bf16 default; atol 2e-3
#: on `o`, 2e-2 on states), derived for a chunked WY reformulation with bf16
#: operands. This op is the same class of sum: measured against fp64 at
#: 1 x 8192 tokens, o max abs 1.26e-3 and state max abs 8.6e-3 (Triton op:
#: 1.13e-3 / 7.7e-3), inside it.
RTOL, ATOL_OUTPUT, ATOL_STATE = 1.6e-2, 2e-3, 2e-2


def _compare(out: tuple, ref: tuple) -> None:
    """`(o, final_state, pool)` against `reference`'s triple; `None` must match."""
    for got, want, atol in zip(out, ref, (ATOL_OUTPUT, ATOL_STATE, ATOL_STATE)):
        assert (got is None) == (want is None), (got is None, want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape, (
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), want.float(), rtol=RTOL, atol=atol)


def _build(spec, seed):
    """Keyword arguments in the domain a GDN post-conv step produces:
    unit-norm q/k (unless the cell feeds raw ones), g < 0 in log space, beta a sigmoid."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = spec["geometry"]
    seqlens = spec["seqlens"]
    total = sum(seqlens)
    q = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    k = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    if spec.get("normalize_qk", True):
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q, k = q.bfloat16(), k.bfloat16()
    v = torch.randn(
        1, total, num_v_heads, head_v_dim, generator=gen, device=DEV, dtype=torch.bfloat16
    )
    a_log = torch.log(torch.rand(num_v_heads, generator=gen, device=DEV) * 15 + 1)
    x = torch.randn(1, total, num_v_heads, generator=gen, device=DEV) + torch.randn(
        num_v_heads, generator=gen, device=DEV
    )
    g = (-torch.exp(a_log) * torch.nn.functional.softplus(x)).float()
    beta = torch.rand(1, total, num_v_heads, generator=gen, device=DEV).sigmoid().float()
    pool = (
        torch.randn(spec["slots"], num_v_heads, head_v_dim, head_k_dim, generator=gen, device=DEV)
        * 0.1
    ).to(spec.get("pool_dtype", torch.float32))
    cu = torch.tensor([0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.long, device=DEV)
    indices = torch.tensor(spec["indices"], dtype=torch.int32, device=DEV)
    output = None
    if spec.get("output") == "buffer":
        output = torch.full(
            (1, total, num_v_heads, head_v_dim), -7.0, device=DEV, dtype=torch.bfloat16
        )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=pool,
        initial_state_indices=indices,
        inplace_indexed_state_update=spec.get("inplace", True),
        output_final_state=spec.get("output_final_state", False),
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=spec.get("l2norm", False),
        output=output,
        state_workspace=None,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    pool = args["initial_state"]
    pool_before = pool.clone()
    ref = op.reference(**args)

    o, final_state = op(**args)

    _compare((o, final_state, pool), ref)
    if args["output"] is not None:
        assert o.data_ptr() == args["output"].data_ptr(), "output= must be written in place"
    named = set(args["initial_state_indices"].tolist())
    unnamed = [s for s in range(pool.shape[0]) if s not in named]
    assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
    if not args["inplace_indexed_state_update"]:
        assert torch.equal(pool, pool_before), "pool written without in-place update"


def test_repeated_calls_are_bitwise() -> None:
    """Two identical calls from the same start give identical output and state."""
    results = []
    for _ in range(2):
        args = _build(
            dict(geometry=QWEN38_GEOMETRY, seqlens=(200, 37, 91), slots=6, indices=(5, 0, 2)),
            seed=7,
        )
        o, _ = op(**args)
        results.append((o.clone(), args["initial_state"].clone()))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])


def test_matches_in_tree_triton_chunk_op() -> None:
    """Same arguments through the vendored Triton chunk op this entry stands in
    for: o and the written slots agree at the band. The Triton op's cold
    indexed call is spent on scratch first."""
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as triton_op

    def run_triton(a, pool):
        return triton_op(
            q=a["q"],
            k=a["k"],
            v=a["v"],
            g=a["g"],
            beta=a["beta"],
            scale=None,
            initial_state=pool,
            initial_state_indices=a["initial_state_indices"],
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=a["cu_seqlens"],
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            output=None,
        )

    warm = _build(dict(geometry=QWEN38_GEOMETRY, seqlens=(64,), slots=1, indices=(0,)), seed=1)
    run_triton(warm, warm["initial_state"])
    args = _build(
        dict(geometry=QWEN38_GEOMETRY, seqlens=(1024, 37, 1), slots=461, indices=(460, 0, 17)),
        seed=3,
    )
    pool_t = args["initial_state"].clone()
    o_t, _ = run_triton(args, pool_t)
    o, _ = op(
        args["q"],
        args["k"],
        args["v"],
        args["g"],
        args["beta"],
        None,
        args["initial_state"],
        args["initial_state_indices"],
        True,
        False,
        args["cu_seqlens"],
        False,
        False,
        None,
    )
    torch.testing.assert_close(o.float(), o_t.float(), rtol=RTOL, atol=ATOL_OUTPUT)
    torch.testing.assert_close(args["initial_state"], pool_t, rtol=RTOL, atol=ATOL_STATE)
