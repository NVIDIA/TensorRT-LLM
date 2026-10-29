# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_sparse_mla_prefill entry.

The pool is synthetic: `[pages, 3 layers, 32 tokens, 576]` e4m3, addressed by
global rows `page * 96 + layer * 32 + token` as the target's are. Request `b`
owns a random set of pages for its cached prefix and its new rows. Its row at
absolute position `pos` has the key window `[ks_b, ks_b + pos + 1)` and
selects that causal prefix (all of it below top_k, a random 2048 of it
above), in random order, then -1s. The top-scoring row sits at the last valid
slot, so a kernel that reads its valid prefix one short drops the key that
dominates the softmax.

The dense-causal cells use the same pool through a `[B, pages]` block table:
row `i` of request `b` sees positions `0 .. cached_b + i`, at
`table[b, pos // 32] * 96 + layer * 32 + pos % 32`.
"""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_sparse_mla_prefill as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_sparse_mla_prefill
HEADS, LATENT, V_DIM, TOP_K = entry.NUM_HEADS, entry.LATENT_DIM, entry.KV_LORA_RANK, entry.TOP_K
PAGE = 32
LAYERS = 3
#: The layer every cell's rows address.
LAYER = 1
#: The scale arrays the target passes: natural 1/16, base-2 1/16 * log2(e), output 1.
BMM1 = (0.0625, 0.0625 * math.log2(math.e))
BMM2 = (1.0,)
#: Operand spread. Scores have std SIGMA**2 * sqrt(576) / 16 = 3.4, so the
#: softmax over 2048 keys puts tens of percent on its top key: peaked enough
#: that dropping one key shows, flat enough that many keys contribute.
SIGMA = 1.5
#: Spare entries past `total_q` in the bounds buffers, as in the target's
#: graph-stable capacity allocations. Filled with garbage.
BOUNDS_SLACK = 7

#: e4m3 rounds to nearest with 3 mantissa bits: at most 2**-4 of the value.
P_ROUND = 2.0**-4
#: Smallest normal e4m3 is 2**-6; below it the spacing is 2**-9, so a
#: probability scaled by 448 moves by at most 2**-10 / 448 absolute.
P_SUBNORMAL = 2.0**-10 / 448.0


def _band(q, pool, rows, scale):
    """`(A, Q2, S)` per output element, in fp64, from the operands.

    `A = sum_r p_r |v_r|`, `Q2 = sum_r p_r**2 v_r**2`, and `S` bounds the
    subnormal-probability error, `P_SUBNORMAL * max_r p_r * sum_r |v_r|`
    (the max undoes the normalization: unnormalized, the top key is 1).
    """
    qd = q.view(torch.uint8).view(torch.float8_e4m3fn).float().double()
    kv = pool.reshape(-1).view(torch.uint8).reshape(-1, LATENT)
    shape = (qd.shape[0], HEADS, V_DIM)
    a, q2, s_ = (torch.zeros(shape, dtype=torch.float64, device=q.device) for _ in range(3))
    for lo in range(0, qd.shape[0], 8):
        hi = min(lo + 8, qd.shape[0])
        r = rows[lo:hi]
        valid = r >= 0
        k = kv[r.clamp(min=0)].view(torch.float8_e4m3fn).float().double()
        s = torch.einsum("thd,tkd->thk", qd[lo:hi], k) * scale
        p = torch.softmax(s.masked_fill(~valid[:, None, :], float("-inf")), -1).nan_to_num(0.0)
        v = k[..., :V_DIM].abs() * valid[..., None]
        a[lo:hi] = torch.einsum("thk,tkd->thd", p, v)
        q2[lo:hi] = torch.einsum("thk,tkd->thd", p * p, v * v)
        s_[lo:hi] = P_SUBNORMAL * p.amax(-1, keepdim=True) * v.sum(1)[:, None, :]
    return a, q2, s_


def _compare(out, ref, band) -> None:
    """Per element `(2**-4 + 2**-8 + 2**-12) A + S`; overall RMS `2 * 2**-4 sqrt(mean Q2)`
    plus the same deterministic terms.

    Kernel and reference take the same fp8 operands and both form exact fp32
    products, so they differ where the kernel rounds: each probability is
    scaled by 448 and rounded to e4m3 before the PV product (at most 2**-4 of
    itself, or `P_SUBNORMAL` absolute when subnormal), while the softmax sum
    keeps the unrounded fp32 value. That moves an output by at most
    `2**-4 * sum_r p_r |v_r| = 2**-4 A`. The bf16 output rounding of kernel
    and reference (2**-9 each) adds `2**-8 A`, and fp32 scores and exp2 well
    under `2**-12 A`. The RMS gate is the statistical form of the same
    rounding: independent zero-mean errors of at most 2**-4 give an expected
    square of at most `2**-8 Q2`, gated with a factor 2 on the root. A kernel
    reading a wrong row, a wrong prefix or a wrong scale misses by O(A).
    """
    a, q2, s = band
    assert bool(torch.isfinite(out).all()), "an output element is not finite (unwritten?)"
    err = out.double().reshape(a.shape) - ref.double().reshape(a.shape)
    bound = (P_ROUND + 2.0**-8 + 2.0**-12) * a + s
    over = err.abs() > bound
    assert not bool(over.any()), (
        f"{int(over.sum())} elements outside the band; worst |err| / bound = "
        f"{float((err.abs() / bound.clamp_min(1e-30)).max()):.3f}"
    )
    rms = float(err.pow(2).mean().sqrt())
    rms_bound = float(
        2 * P_ROUND * q2.mean().sqrt()
        + (2.0**-8 + 2.0**-12) * a.pow(2).mean().sqrt()
        + s.pow(2).mean().sqrt()
    )
    assert rms <= rms_bound, f"RMS error {rms:.3e} exceeds {rms_bound:.3e}"


def _build(spec, seed):
    """`(args, rows)`: the op's arguments, and the global rows the indices name."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    new, cached = spec["new"], spec["cached"]
    totals = [c + n for c, n in zip(cached, new)]
    pages = [math.ceil(n / PAGE) for n in totals]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu)
    pool = (torch.randn(num_pages, LAYERS, PAGE, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    total_q = sum(new)
    q = (torch.randn(total_q, HEADS, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    kv = pool.reshape(-1).view(torch.uint8).reshape(-1, LATENT)
    rows = torch.full((total_q, TOP_K), -1, dtype=torch.int64)
    ks = torch.full((total_q + BOUNDS_SLACK,), -12345, dtype=torch.int32)
    ke = torch.full((total_q + BOUNDS_SLACK,), 99999, dtype=torch.int32)
    used, row, k_start = 0, 0, 0
    for b in range(len(new)):
        table = order[used : used + pages[b]]
        used += pages[b]
        for i in range(new[b]):
            causal = cached[b] + i + 1
            valid = min(causal, TOP_K)
            pos = torch.randperm(causal, generator=cpu)[:valid]
            sel = table[pos // PAGE] * (LAYERS * PAGE) + LAYER * PAGE + pos % PAGE
            # The top-scoring key, in any head, goes to the last valid slot.
            k = kv[sel.cuda()].view(torch.float8_e4m3fn).float()
            best = int((q[row].float() @ k.t()).amax(0).argmax())
            sel[[best, valid - 1]] = sel[[valid - 1, best]]
            rows[row, :valid] = sel
            ks[row] = k_start
            ke[row] = k_start + causal
            row += 1
        k_start += totals[b]
    rows = rows.cuda()
    args = dict(
        quant_q_buffer=q,
        kv_pool_view=pool,
        sparse_attn_indices=rows.to(torch.int32),
        output=torch.full(
            (total_q, HEADS * V_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        cu_seqlen_ks=ks.cuda(),
        cu_seqlen_ke=ke.cuda(),
        mla_bmm1_scale=torch.tensor(BMM1, dtype=torch.float32, device="cuda"),
        mla_bmm2_scale=torch.tensor(BMM2, dtype=torch.float32, device="cuda"),
    )
    return args, rows


def _build_dense(spec, seed, layer=LAYER):
    """`(args, rows)` for the dense-causal mode, and the index image it reads (the
    rows the index mode would be handed for the same selection)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    new, cached = spec["new"], spec["cached"]
    pages = [math.ceil((c + n) / PAGE) for c, n in zip(cached, new)]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu)
    pool = (torch.randn(num_pages, LAYERS, PAGE, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    total_q = sum(new)
    q = (torch.randn(total_q, HEADS, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    table = torch.full((len(new), max(pages) + 1), -1, dtype=torch.int32)
    rows = torch.full((total_q, TOP_K), -1, dtype=torch.int64)
    req, used, row = [], 0, 0
    for b in range(len(new)):
        table[b, : pages[b]] = order[used : used + pages[b]]
        used += pages[b]
        for i in range(new[b]):
            pos = torch.arange(min(cached[b] + i + 1, TOP_K))
            rows[row, : len(pos)] = (
                table[b, pos // PAGE].long() * (LAYERS * PAGE) + layer * PAGE + pos % PAGE
            )
            req.append(b)
            row += 1
    new_t = torch.tensor(new, dtype=torch.int32)
    cached_t = torch.tensor(cached, dtype=torch.int32, device="cuda") if any(cached) else None
    ks, ke = entry.dense_causal_bounds(new_t.cuda(), cached_t, total_q)
    args = dict(
        quant_q_buffer=q,
        kv_pool_view=pool,
        sparse_attn_indices=None,
        output=torch.full(
            (total_q, HEADS * V_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        mla_bmm1_scale=torch.tensor(BMM1, dtype=torch.float32, device="cuda"),
        mla_bmm2_scale=torch.tensor(BMM2, dtype=torch.float32, device="cuda"),
        block_table=table.cuda(),
        # Slack past total_q, as the engine's per-token buffer has.
        req_idx=torch.tensor(req + [0] * BOUNDS_SLACK, dtype=torch.int32, device="cuda"),
        stride_factor=LAYERS * PAGE,
        layer_offset=layer * PAGE,
        dense_causal_groups=entry.dense_causal_groups(new_t).cuda(),
    )
    return args, rows.cuda()


@pytest.fixture(scope="module")
def compiled():
    """The handle the target builds at load: the -1-tail-safe kernel."""
    return entry.compile(device_index=torch.cuda.current_device())


@pytest.fixture(scope="module")
def compiled_dc():
    """The dense-causal handle the target builds at load."""
    return entry.compile(device_index=torch.cuda.current_device(), dense_causal=True)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, compiled_dc, certifying) -> None:  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if not cell.spec.get("dense_causal"):
        args, rows = _build(cell.spec, seed)
        out = op(**args, compiled=compiled)
        torch.cuda.synchronize()
        ref = op.reference(**args, compiled=compiled)
        _compare(out, ref, _band(args["quant_q_buffer"], args["kv_pool_view"], rows, BMM1[0]))
        return
    args, rows = _build_dense(cell.spec, seed)
    handle = compiled_dc
    if cell.spec.get("prepared"):
        # The target's step: bound on layer 0 with that layer's output, then layer
        # LAYER launched through the same binding with its own output and offset.
        first = dict(args, output=torch.empty_like(args["output"]), layer_offset=0)
        handle = entry.prepare_dense_causal(compiled_dc, *(first[k] for k in _PREPARE_ARGS))
        op(**first, compiled=handle)
    out = op(**args, compiled=handle)
    torch.cuda.synchronize()
    ref = op.reference(**args, compiled=handle)
    # The reference's own window rows against the ones built here.
    index_args = dict(args, sparse_attn_indices=rows.to(torch.int32))
    assert torch.equal(ref, op.reference(**index_args, compiled=compiled))
    _compare(out, ref, _band(args["quant_q_buffer"], args["kv_pool_view"], rows, BMM1[0]))


_PREPARE_ARGS = (
    "quant_q_buffer", "kv_pool_view", "output", "cu_seqlen_ks", "cu_seqlen_ke",
    "mla_bmm1_scale", "mla_bmm2_scale", "block_table", "req_idx", "stride_factor",
    "layer_offset", "dense_causal_groups",
)  # fmt: skip


def test_dense_causal_equals_the_index_mode(compiled, compiled_dc) -> None:
    """The kernel's own claim, which the target's switch between the two modes
    rests on: on the dense causal index image with `ks = 0`, the two modes write
    the same bits."""
    args, rows = _build_dense(dict(new=[37, 300, 131], cached=[0, 0, 64]), seed=11)
    dense = op(**args, compiled=compiled_dc).clone()
    index = {k: v for k, v in args.items() if k not in _DENSE_ONLY}
    index["sparse_attn_indices"] = rows.to(torch.int32)
    index["output"] = torch.empty_like(args["output"])
    sparse = op(**index, compiled=compiled)
    torch.cuda.synchronize()
    assert torch.equal(dense, sparse)


_DENSE_ONLY = ("block_table", "req_idx", "stride_factor", "layer_offset", "dense_causal_groups")


def _handle(all_valid: bool = False) -> "entry.CompiledSparseMlaPrefill":
    """A handle with no kernel: `is_valid` reads only its validity mode."""
    return entry.CompiledSparseMlaPrefill(
        device_index=torch.cuda.current_device(), all_sparse_indices_valid=all_valid, kernel=None
    )


_GUARD_SPEC = dict(new=[5, 4], cached=[0, 2046])


def test_is_valid_admits_a_well_formed_call() -> None:
    args, _ = _build(_GUARD_SPEC, 1)
    op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_hole_in_the_valid_prefix() -> None:
    """The kernel gathers the -1 from a substitute row and mixes it in."""
    args, _ = _build(_GUARD_SPEC, 2)
    args["sparse_attn_indices"][3, 1] = -1
    with pytest.raises(AssertionError, match="inside a row's valid prefix"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_row_in_the_tail() -> None:
    """A key window one short of the selection: the kernel masks the last
    selected row and returns a softmax over the rest."""
    args, _ = _build(_GUARD_SPEC, 3)
    args["cu_seqlen_ke"][2] -= 1
    with pytest.raises(AssertionError, match="-1 tail"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_scale_pair_the_kernel_reads_half_of() -> None:
    """`[0]` says 1/16; the kernel applies `[1]`, which says 1/8."""
    args, _ = _build(_GUARD_SPEC, 4)
    args["mla_bmm1_scale"][1] *= 2
    with pytest.raises(AssertionError, match="base-2"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_short_rows_under_the_all_valid_build() -> None:
    """That build reads all 2048 entries of every row, -1s included."""
    args, _ = _build(_GUARD_SPEC, 5)
    with pytest.raises(AssertionError, match="all_sparse_indices_valid"):
        op.is_valid(**args, compiled=_handle(all_valid=True))


def _dense_handle() -> "entry.CompiledSparseMlaPrefill":
    return entry.CompiledSparseMlaPrefill(
        device_index=torch.cuda.current_device(),
        all_sparse_indices_valid=False,
        kernel=None,
        dense_causal=True,
    )


_DENSE_GUARD_SPEC = dict(new=[6, 9], cached=[40, 0])


def test_is_valid_admits_a_well_formed_dense_causal_call() -> None:
    args, _ = _build_dense(_DENSE_GUARD_SPEC, 6)
    op.is_valid(**args, compiled=_dense_handle())


def test_is_valid_refuses_dense_causal_windows_that_do_not_start_at_zero() -> None:
    """The kernel never reads `ks`: a window meant to start later is read from 0."""
    args, _ = _build_dense(_DENSE_GUARD_SPEC, 7)
    args["cu_seqlen_ks"][3] = 1
    with pytest.raises(AssertionError, match="cu_seqlen_ks must be 0"):
        op.is_valid(**args, compiled=_dense_handle())


def test_is_valid_refuses_a_group_across_two_requests() -> None:
    """A work item reads one request's KV for all of its rows."""
    args, _ = _build_dense(_DENSE_GUARD_SPEC, 8)
    groups = torch.tensor([[0, 4], [4, 4], [8, 4], [12, 3]], dtype=torch.int32, device="cuda")
    args["dense_causal_groups"] = groups
    with pytest.raises(AssertionError, match="spans two requests"):
        op.is_valid(**args, compiled=_dense_handle())


def test_is_valid_refuses_a_window_past_the_block_table() -> None:
    """A position whose block is -1 is outside the kernel's contract."""
    args, _ = _build_dense(_DENSE_GUARD_SPEC, 9)
    args["block_table"][0, 1] = -1
    with pytest.raises(AssertionError, match="-1 block inside a row's window"):
        op.is_valid(**args, compiled=_dense_handle())


def test_is_valid_refuses_a_prepared_step_handed_other_bounds(compiled_dc) -> None:
    """A prepared launch re-binds only output and layer offset; other bounds passed
    in are silently replaced by the step's."""
    args, _ = _build_dense(_DENSE_GUARD_SPEC, 10)
    handle = entry.prepare_dense_causal(compiled_dc, *(args[k] for k in _PREPARE_ARGS))
    args["cu_seqlen_ke"] = args["cu_seqlen_ke"].clone()
    with pytest.raises(AssertionError, match="cu_seqlen_ke is not the tensor"):
        op.is_valid(**args, compiled=handle)
