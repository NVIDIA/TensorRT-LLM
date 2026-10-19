# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_gating_delta_rule_update entry.

Everything here certifies the vendored Triton branch. The target's two decode calls
keep the FlashInfer branch out -- the packed call over an fp32 pool, and the rows form
(no `cu_seqlens`) over a bf16 one -- which `test_certified_branch_is_the_vendored_triton_kernel`
proves. `test_rows_form_is_batch_invariant` shows the property the bf16 call takes the rows
form for.
"""

import zlib
from typing import Optional

import pytest
import torch

import tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent as _impl
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
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0
_LAYERS = 3
_TAIL = 5


def _rows_form(mixed, a, b):
    """The target's bf16 decode operands: q/k/v `[N, 1, heads, 128]` views of the packed
    `[N, 10240]` block, reshaped whole so the size-1 token axis carries the row stride, and
    a/b `[N, 1, 48]`."""
    n = mixed.shape[0]
    qkv = mixed.view(n, 1, MIXED_QKV_WIDTH // HEAD_K_DIM, HEAD_K_DIM)
    q, k, v = torch.split(qkv, [NUM_K_HEADS, NUM_K_HEADS, NUM_V_HEADS], 2)
    return q, k, v, a.reshape(n, 1, NUM_V_HEADS), b.reshape(n, 1, NUM_V_HEADS)


def _build(spec, seed):
    """`__call__`'s 15 positional arguments, plus the backing buffers some
    layouts view. q/k/v are views over one packed `[T, 10240]` conv output,
    exactly as the decode call reshapes its column splits."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    seq_lens = spec["seq_lens"]
    total = sum(seq_lens)
    layout = spec.get("layout", "varlen")
    mixed = torch.randn(total, MIXED_QKV_WIDTH, generator=g, device=DEV, dtype=torch.bfloat16)
    kd = NUM_K_HEADS * HEAD_K_DIM
    q = mixed[:, :kd].view(1, total, NUM_K_HEADS, HEAD_K_DIM)
    k = mixed[:, kd : 2 * kd].view(1, total, NUM_K_HEADS, HEAD_K_DIM)
    v = mixed[:, 2 * kd :].view(1, total, NUM_V_HEADS, HEAD_V_DIM)
    if spec.get("normalize_qk", False):
        for t in (q, k):
            tf = t.float()
            t.copy_(tf / (tf.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6))
    if spec.get("ab", "packed") == "separate":
        a = torch.randn(total, NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(total, NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
    else:
        ba = torch.randn(total, 2 * NUM_V_HEADS, generator=g, device=DEV, dtype=torch.bfloat16)
        b, a = ba[:, :NUM_V_HEADS], ba[:, NUM_V_HEADS:]
    A_log = torch.randn(NUM_V_HEADS, generator=g, device=DEV)
    dt_bias = torch.randn(NUM_V_HEADS, generator=g, device=DEV)

    extras = {"mixed": mixed}
    slots = spec["slots"]
    pool = None
    if slots is not None:
        lead = (_LAYERS, slots) if layout == "layer_view" else (slots,)
        state_shape = (*lead, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM)
        pool = torch.randn(state_shape, generator=g, device=DEV) * 0.05
        pool = pool.to(spec.get("pool_dtype", torch.float32))
        if layout == "layer_view":
            extras["cache"] = pool
            pool = pool[1]
    indices = spec.get("indices") or tuple(range(len(seq_lens)))
    indices = torch.tensor(indices, dtype=torch.int32, device=DEV)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int64, device=DEV
    )
    if layout == "batched":
        n, t = len(seq_lens), seq_lens[0]
        q, k = (x.reshape(n, t, NUM_K_HEADS, HEAD_K_DIM).contiguous() for x in (q, k))
        v = v.reshape(n, t, NUM_V_HEADS, HEAD_V_DIM).contiguous()
        a, b = (x.reshape(n, t, NUM_V_HEADS).contiguous() for x in (a, b))
        cu_seqlens = None
    elif layout == "rows":
        q, k, v, a, b = _rows_form(mixed, a, b)
        cu_seqlens = None
    output = None
    if spec.get("output") == "prefix":
        extras["buf"] = torch.full(
            (1, total + _TAIL, NUM_V_HEADS, HEAD_V_DIM), -7.0, device=DEV, dtype=torch.bfloat16
        )
        output = extras["buf"][:, :total]
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
        spec.get("l2norm", True),
        cu_seqlens,
        output,
    ]
    return args, extras


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args, extras = _build(spec, seed=zlib.crc32(cell.why.encode()))
    pool = args[9]
    pool_before = None if pool is None else pool.clone()
    cache_before = extras["cache"].clone() if "cache" in extras else None
    tail_before = extras["buf"][:, -_TAIL:].clone() if "buf" in extras else None
    ref = op.reference(*args)

    out = op(*args)

    _compare((out, pool), ref)
    if args[14] is not None:
        assert out.data_ptr() == args[14].data_ptr(), "output= must be written in place"
        assert torch.equal(extras["buf"][:, -_TAIL:], tail_before), "buffer tail written"
    if cache_before is not None:
        for other in (0, 2):
            assert torch.equal(extras["cache"][other], cache_before[other]), other
    if pool is not None:
        # The whole-pool compare is the lock on unnamed slots; make it bitwise.
        named = {s for s in args[10].tolist() if s >= 0}
        unnamed = [s for s in range(pool.shape[0]) if s not in named]
        assert torch.equal(pool[unnamed], pool_before[unnamed]), "unnamed slots written"
        if spec["slots"] == ENGINE_SLOTS:
            for slot in named:
                assert not torch.equal(pool[slot], pool_before[slot]), slot


def test_certified_branch_is_the_vendored_triton_kernel() -> None:
    """Both target decode calls keep the FlashInfer arm out, armed both ways: its entry
    point is replaced by one that raises; the packed call over an fp32 pool and the rows
    form over a bf16 pool complete, and the packed call over a bf16 pool in the same
    process does raise -- so the replaced entry point is known to be reachable."""

    def _boom(*args, **kwargs):
        raise AssertionError("FlashInfer arm taken")

    original = _impl._flashinfer_gdn_decode
    try:
        _impl._flashinfer_gdn_decode = _boom
        args, _ = _build(dict(seq_lens=(1,) * 4, slots=24), seed=100)
        assert not _impl._can_use_flashinfer_gdn_decode(args[9], HEAD_K_DIM, HEAD_V_DIM, 4, 4)
        assert op(*args).shape == (1, 4, NUM_V_HEADS, HEAD_V_DIM)

        rows = dict(seq_lens=(1,) * 32, slots=64, ab="separate", pool_dtype=torch.bfloat16)
        args, _ = _build(dict(rows, layout="rows"), seed=102)
        assert op(*args).shape == (32, 1, NUM_V_HEADS, HEAD_V_DIM)

        args, _ = _build(dict(seq_lens=(1,) * 4, slots=24, pool_dtype=torch.bfloat16), seed=101)
        assert _impl._can_use_flashinfer_gdn_decode(args[9], HEAD_K_DIM, HEAD_V_DIM, 4, 4)
        with pytest.raises(AssertionError, match="FlashInfer arm taken"):
            op(*args)
    finally:
        _impl._flashinfer_gdn_decode = original


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


def test_rows_form_is_the_packed_triton_call_bitwise() -> None:
    """At either pool dtype the rows form computes what the packed Triton call computes,
    bit for bit -- output and pool. The packed call is forced onto the Triton kernel for
    the bf16 comparison."""
    for pool_dtype in (torch.float32, torch.bfloat16):
        for n in (1, 8, 32):
            spec = dict(seq_lens=(1,) * n, slots=64, ab="separate", pool_dtype=pool_dtype)
            packed, _ = _build(spec, seed=300 + n)
            rows, _ = _build(dict(spec, layout="rows"), seed=300 + n)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(_impl, "_can_use_flashinfer_gdn_decode", lambda *a, **k: False)
                o_packed = op(*packed)
            o_rows = op(*rows)
            assert torch.equal(o_rows.reshape(o_packed.shape), o_packed), (pool_dtype, n)
            assert torch.equal(rows[9], packed[9]), (pool_dtype, n)


@pytest.mark.parametrize("n", (1, 16, 32))
def test_rows_form_cuda_graph_replay_is_bitwise_eager(n) -> None:
    """The bf16 rows form captured at batch `n` and replayed on new activations, slots and
    pool content, against the same call run eagerly. The kernel takes the batch as a
    constexpr, so each batch size compiles on its eager call, before the capture."""
    spec = dict(
        seq_lens=(1,) * n,
        slots=ENGINE_SLOTS,
        ab="separate",
        layout="rows",
        pool_dtype=torch.bfloat16,
    )
    args, extras = _build(spec, seed=400 + n)
    static = list(args)
    for _ in range(2):
        op(*static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = op(*static)
    for trial in range(3):
        slots = tuple(7 * trial + 13 * i for i in range(n))
        fresh, fresh_extras = _build(dict(spec, indices=slots), seed=500 + 10 * n + trial)
        for position in (0, 1, 2, 8, 9, 10):
            static[position].copy_(fresh[position])
        extras["mixed"].copy_(fresh_extras["mixed"])
        pool_before = static[9].clone()
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_pool = out_static.clone(), static[9].clone()
        eager_args = list(fresh)
        eager_args[9] = pool_before
        eager = op(*eager_args)
        assert torch.equal(replayed, eager), (n, trial)
        assert torch.equal(replayed_pool, pool_before), (n, trial)


def test_default_scale_and_repeated_calls_are_bitwise() -> None:
    """`scale=None` equals an explicit `K ** -0.5` bitwise, and two identical
    calls give identical output and state."""
    args, _ = _build(dict(seq_lens=(1,) * 4, slots=24), seed=95)
    pool = args[9]
    results = []
    for scale in (None, None, HEAD_K_DIM**-0.5):
        call = list(args)
        call[9], call[11] = pool.clone(), scale
        results.append((op(*call), call[9]))
    for o, p in results[1:]:
        assert torch.equal(o, results[0][0]) and torch.equal(p, results[0][1])


def test_guard_refuses_a_noncontiguous_output_buffer() -> None:
    """Unguarded, the kernel writes `output` with packed arithmetic: a last-dim
    slice of a doubly wide buffer is filled with misplaced values."""
    args, _ = _build(dict(seq_lens=(1,) * 4, slots=24), seed=97)
    pool_before = args[9].clone()
    ref = op.reference(*args)
    wide = torch.zeros(1, 4, NUM_V_HEADS, 2 * HEAD_V_DIM, device=DEV, dtype=torch.bfloat16)
    unguarded = list(args)
    unguarded[14] = wide[..., :HEAD_V_DIM]
    got = op(*unguarded)
    with pytest.raises(AssertionError):
        _compare((got, args[9]), ref)
    assert torch.count_nonzero(wide[..., HEAD_V_DIM:]) > 0, "the hazard this guard covers is gone"

    strided = torch.zeros(1, 8, NUM_V_HEADS, HEAD_V_DIM, device=DEV, dtype=torch.bfloat16)
    for bad in (wide[..., :HEAD_V_DIM], strided[:, ::2]):
        guarded = list(args)
        guarded[9], guarded[14] = pool_before.clone(), bad
        with pytest.raises(AssertionError, match="output must be contiguous"):
            with validating(op):
                op(*guarded)


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


def test_wrong_variants_land_far_outside_the_gate() -> None:
    """The default band separates right from wrong by > 100x the correct gap."""
    args, _ = _build(dict(seq_lens=(1,) * 4, slots=24), seed=98)
    pool_before = args[9].clone()
    ref = op.reference(*args)
    got = op(*args)
    _compare((got, args[9]), ref)
    gap = (got.float() - ref[0].float()).abs().max().item()

    def ref_with(**overrides):
        names = ("A_log", "a", "dt_bias", "q", "b", "pool", "l2norm")
        positions = (0, 1, 2, 5, 8, 9, 12)
        call = list(args)
        call[9] = pool_before
        for name, position in zip(names, positions):
            if name in overrides:
                call[position] = overrides[name]
        return op.reference(*call)[0]

    q = args[5]
    flipped = q[:, :, torch.arange(NUM_K_HEADS, device=DEV).flip(0), :]
    wrong = {
        "a and b swapped": ref_with(a=args[8], b=args[1]),
        "l2norm omitted": ref_with(l2norm=False),
        "pool (V, K) transposed": ref_with(pool=pool_before.transpose(2, 3).contiguous()),
        "initial state ignored": ref_with(pool=torch.zeros_like(pool_before)),
        "q head order reversed": ref_with(q=flipped),
    }
    for name, variant in wrong.items():
        distance = (got.float() - variant.float()).abs().max().item()
        assert distance > 100 * gap, f"{name}: {distance} vs correct {gap}"
        with pytest.raises(AssertionError):
            _compare((got, args[9]), (variant, ref[1]))
