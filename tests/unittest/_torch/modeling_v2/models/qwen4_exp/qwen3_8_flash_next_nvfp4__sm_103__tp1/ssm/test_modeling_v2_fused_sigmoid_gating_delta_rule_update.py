# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_gating_delta_rule_update entry.

Everything here certifies the vendored Triton branch: the target's decode call -- the
rows form (no `cu_seqlens`) over a bf16 pool -- and the packed call over an fp32 pool that
the GDN block test uses as its reference. `test_rows_form_is_batch_invariant` shows the
property the bf16 call takes the rows form for.
"""

import zlib
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_sigmoid_gating_delta_rule_update import (  # noqa: E501
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_sigmoid_gating_delta_rule_update import (  # noqa: E501
    fused_sigmoid_gating_delta_rule_update as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_gating_delta_rule_update requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, Optional[torch.Tensor]],
    ref: tuple[torch.Tensor, Optional[torch.Tensor]],
) -> None:
    """`(o, pool)`: the returned output with the pool as the call left it.

    Both at torch's default dtype-aware band: the kernel and the reference
    accumulate the same fp32 sequence and round once (bf16 `o`: measured
    3.05e-05 max abs at N=32, 461 slots; fp32 pool 2.4e-07). Over a bf16 pool the
    state is rounded once on store, as the reference rounds it, so a stored value
    can sit one bf16 step (2^-8 relative) from the reference's, inside the band. Five wrong
    variants (a/b swapped, norm dropped, pool `(V, K)` transposed, initial
    state ignored, q-head order reversed) land 1.7e-2 to 4.5 away, three
    orders of magnitude outside. The pool is compared whole, so the
    unnamed slots are gated too (the reference leaves them bit-identical).
    """
    (o, pool), (ref_o, ref_pool) = out, ref
    assert o.dtype == ref_o.dtype and o.shape == ref_o.shape, (
        (o.dtype, o.shape),
        (ref_o.dtype, ref_o.shape),
    )
    assert torch.isfinite(o).all()
    torch.testing.assert_close(o, ref_o)
    assert (pool is None) == (ref_pool is None)
    if pool is not None:
        assert pool.dtype == ref_pool.dtype and pool.shape == ref_pool.shape
        torch.testing.assert_close(pool, ref_pool)


DEV = "cuda"
NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 16, 48, 128, 128
MIXED_QKV_WIDTH = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240
#: The target's one GDN input-projection output: [q|k|v | z | a | b], 16480 wide.
IN_PROJ_A = MIXED_QKV_WIDTH + NUM_V_HEADS * HEAD_V_DIM  # 16384
IN_PROJ_WIDTH = IN_PROJ_A + 2 * NUM_V_HEADS
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0


def _rows_form(mixed, a, b):
    """The target's bf16 decode operands: q/k/v `[N, 1, heads, 128]` views of the packed
    `[N, 10240]` block, reshaped whole so the size-1 token axis carries the row stride, and
    a/b `[N, 1, 48]`."""
    n = mixed.shape[0]
    qkv = mixed.view(n, 1, MIXED_QKV_WIDTH // HEAD_K_DIM, HEAD_K_DIM)
    q, k, v = torch.split(qkv, [NUM_K_HEADS, NUM_K_HEADS, NUM_V_HEADS], 2)
    return q, k, v, a.reshape(n, 1, NUM_V_HEADS), b.reshape(n, 1, NUM_V_HEADS)


def _build(spec, seed):
    """`__call__`'s 15 positional arguments, plus the packed block the rows form
    views. q/k/v are views over one packed `[T, 10240]` conv output, exactly as
    the decode call reshapes its column splits."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    seq_lens = spec["seq_lens"]
    total = sum(seq_lens)
    if spec["ab"] == "in_proj":
        proj = torch.randn(total, IN_PROJ_WIDTH, generator=g, device=DEV, dtype=torch.bfloat16)
        mixed = proj[:, :MIXED_QKV_WIDTH]
    else:
        mixed = torch.randn(total, MIXED_QKV_WIDTH, generator=g, device=DEV, dtype=torch.bfloat16)
    kd = NUM_K_HEADS * HEAD_K_DIM
    q = mixed[:, :kd].view(1, total, NUM_K_HEADS, HEAD_K_DIM)
    k = mixed[:, kd : 2 * kd].view(1, total, NUM_K_HEADS, HEAD_K_DIM)
    v = mixed[:, 2 * kd :].view(1, total, NUM_V_HEADS, HEAD_V_DIM)
    if spec["ab"] == "in_proj":
        a = proj[:, IN_PROJ_A : IN_PROJ_A + NUM_V_HEADS]
        b = proj[:, IN_PROJ_A + NUM_V_HEADS :]
        assert mixed.stride(0) == a.stride(0) == b.stride(0) == IN_PROJ_WIDTH
    else:
        a = torch.randn(total, NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(total, NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
    A_log = torch.randn(NUM_V_HEADS, generator=g, device=DEV)
    dt_bias = torch.randn(NUM_V_HEADS, generator=g, device=DEV)

    state_shape = (spec["slots"], NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM)
    pool = torch.randn(state_shape, generator=g, device=DEV) * 0.05
    pool = pool.to(spec.get("pool_dtype", torch.float32))
    indices = spec.get("indices") or tuple(range(len(seq_lens)))
    indices = torch.tensor(indices, dtype=torch.int32, device=DEV)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int64, device=DEV
    )
    if spec.get("layout", "varlen") == "rows":
        q, k, v, a, b = _rows_form(mixed, a, b)
        cu_seqlens = None
    args = [
        A_log,
        a,
        dt_bias,
        SOFTPLUS_BETA,
        SOFTPLUS_THRESHOLD,
        q,
        k,
        v,
        b,
        pool,
        indices,
        None,
        True,
        cu_seqlens,
        None,
    ]
    return args, {"mixed": mixed}


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args, _ = _build(spec, seed=zlib.crc32(cell.why.encode()))
    pool = args[9]
    pool_before = pool.clone()
    ref = op.reference(*args)

    out = op(*args)

    _compare((out, pool), ref)
    # The whole-pool compare is the lock on unnamed slots; make it bitwise.
    named = {s for s in args[10].tolist() if s >= 0}
    unnamed = [s for s in range(pool.shape[0]) if s not in named]
    assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
    if spec["slots"] == ENGINE_SLOTS:
        for slot in named:
            assert not torch.equal(pool[slot], pool_before[slot]), slot


#: Batch sizes either side of every FlashInfer dispatch boundary at 48 value heads (its
#: kernel and tile change at 2|3, 6|7, 10|11 and 21|22) and the engine's CUDA-graph batch
#: sizes up to 32.
_BATCHES = (1, 2, 3, 10, 11, 16, 21, 22, 32)
_FIXED_SLOT = 301
_FIXED_ROW_SLOTS = 512


def test_rows_form_is_batch_invariant() -> None:
    """The property the target's bf16 decode call relies on: a row's output and state are
    bitwise independent of the batch it rides in, at batch sizes either side of every
    FlashInfer dispatch boundary and at both ends of the batch. One fixed row (activations
    and slot 301 of a bf16 pool) sits at position 0 and N-1 of random N-row batches."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    cpu = torch.Generator().manual_seed(3)
    pool0 = (
        torch.randn(
            _FIXED_ROW_SLOTS, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, generator=gen, device=DEV
        )
        * 0.05
    ).to(torch.bfloat16)
    row = torch.randn(MIXED_QKV_WIDTH, generator=gen, device=DEV, dtype=torch.bfloat16)
    row_a, row_b = (
        torch.randn(NUM_V_HEADS, generator=gen, device=DEV, dtype=torch.bfloat16) for _ in range(2)
    )
    A_log = torch.randn(NUM_V_HEADS, generator=gen, device=DEV)
    dt_bias = torch.randn(NUM_V_HEADS, generator=gen, device=DEV)
    first = None
    for n in _BATCHES:
        for pos in sorted({0, n - 1}):
            g = torch.Generator(device=DEV).manual_seed(1000 * n + pos)
            mixed = torch.randn(n, MIXED_QKV_WIDTH, generator=g, device=DEV, dtype=torch.bfloat16)
            a = torch.randn(n, NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
            b = torch.randn(n, NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
            mixed[pos], a[pos], b[pos] = row, row_a, row_b
            others = [s for s in torch.randperm(_FIXED_ROW_SLOTS, generator=cpu).tolist()]
            others = [s for s in others if s != _FIXED_SLOT][:n]
            others[pos] = _FIXED_SLOT
            slots = torch.tensor(others, dtype=torch.int32, device=DEV)
            pool = pool0.clone()
            q, k, v, a_in, b_in = _rows_form(mixed, a, b)
            out = op(A_log, a_in, dt_bias, 1.0, 20.0, q, k, v, b_in, pool, slots, None, True, None)
            got = (out.reshape(n, NUM_V_HEADS, HEAD_V_DIM)[pos].clone(), pool[_FIXED_SLOT].clone())
            first = first or got
            assert torch.equal(got[0], first[0]) and torch.equal(got[1], first[1]), (
                f"the fixed row differs at batch {n}, position {pos}"
            )


def test_guard_refuses_rows_views_whose_token_stride_is_not_the_row_stride() -> None:
    """Unguarded, q/k/v reshaped to `[N, 1, heads, 128]` from column slices of the packed
    block carry `stride(1) = heads * 128`: without `cu_seqlens` the kernel reads row n at
    `n * stride(1)`, from other rows' columns."""
    n, kd = 8, NUM_K_HEADS * HEAD_K_DIM
    spec = dict(
        seq_lens=(1,) * n, slots=64, ab="separate", layout="rows", pool_dtype=torch.bfloat16
    )
    args, extras = _build(spec, seed=600)
    mixed = extras["mixed"]
    pool_before = args[9].clone()
    ref = op.reference(*args)
    bad = list(args)
    bad[5] = mixed[:, :kd].reshape(n, 1, NUM_K_HEADS, HEAD_K_DIM)
    bad[6] = mixed[:, kd : 2 * kd].reshape(n, 1, NUM_K_HEADS, HEAD_K_DIM)
    bad[7] = mixed[:, 2 * kd :].reshape(n, 1, NUM_V_HEADS, HEAD_V_DIM)
    assert all(bad[i].stride(1) != bad[i].stride(0) for i in (5, 6, 7))
    for i in (5, 6, 7):
        assert torch.equal(bad[i], args[i]), "the same values, other strides"
    got = op(*bad)
    with pytest.raises(AssertionError):
        _compare((got, bad[9]), ref)
    guarded = list(bad)
    guarded[9] = pool_before
    with pytest.raises(AssertionError, match=r"stride\(0\) must equal"):
        with validating(op):
            op(*guarded)
