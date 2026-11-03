# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_chunk_gated_delta_rule entry (FlashInfer GDN prefill)."""

import zlib

import pytest
import torch

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
    unit-norm q/k, g < 0 in log space, beta a sigmoid."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = spec["geometry"]
    seqlens = spec["seqlens"]
    total = sum(seqlens)
    q = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
    k = torch.randn(1, total, num_k_heads, head_k_dim, generator=gen, device=DEV)
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
    extra = {}
    if "initial" in spec:
        extra["has_initial_state"] = torch.tensor(spec["initial"], dtype=torch.bool, device=DEV)
    if spec.get("linear_gate"):
        extra["linear_gate"] = True
        g = torch.exp(g)
    return dict(
        **extra,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=pool,
        initial_state_indices=indices,
        inplace_indexed_state_update=True,
        output_final_state=False,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
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


_CARRIED = next(c for c in op.CELLS if "initial" in c.spec)


def test_has_initial_state_is_the_reset_then_the_plain_call() -> None:
    """The flags replace `reset_recurrent_state_rows` before the call: zeroing the fresh
    slots and running the plain in-place call gives the same output and pool, bitwise; a
    linear gate that is `torch.exp` of the plain call's log gate changes nothing."""
    args = _build(_CARRIED.spec, seed=5)
    flags = args.pop("has_initial_state")
    linear_gate = args.pop("linear_gate", False)
    pool = args["initial_state"]
    fresh = args["initial_state_indices"][~flags].long()
    pool[fresh] = torch.randn_like(pool[fresh])  # stale rows the flags must not read
    plain = dict(args, initial_state=pool.clone(), output=args["output"].clone())
    plain["initial_state"][fresh] = 0.0
    if linear_gate:
        plain["g"] = torch.log(args["g"])
        # The linear gate of the target call is torch.exp of the log gate, bitwise.
        args["g"] = torch.exp(plain["g"])

    o_plain, _ = op(**plain)
    o, final_state = op(**args, has_initial_state=flags, linear_gate=linear_gate)

    assert final_state is None
    assert torch.equal(o, o_plain)
    assert torch.equal(pool, plain["initial_state"])
