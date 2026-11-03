# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_sparse_mla_decode entry.

The pool is synthetic: `[pages, 3 layers, 32 tokens, 576]` e4m3, which is the
row layout the target's global rows address (`page * 96 + layer * 32 +
token`). Each sequence owns a random set of pages, and each query row selects
its causal prefix (all of it below top_k, a random 2048 of it above), in
random order, then -1s. The row whose score is highest sits at the last valid
slot, so a kernel that reads its valid prefix one short drops the key that
dominates the softmax.

The handle is built the way the target builds it, for every batch up to a
bound, and the bound is the largest batch any cell drives. A `gmem_split` cell
takes the smallest batch whose kernel pick on this device is that split with
the global-memory reducer; the device's SM count and cluster occupancy decide
it, so it is found here rather than written into the cell.
"""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_sparse_mla_decode as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_sparse_mla_decode
HEADS, LATENT, V_DIM, TOP_K = entry.NUM_HEADS, entry.LATENT_DIM, entry.KV_LORA_RANK, entry.TOP_K
PAGE = entry.LAYER_ROW_STRIDE
LAYERS = 3
#: Operand spread. Scores have std SIGMA**2 * sqrt(576) / 16 = 3.4, so the
#: softmax over 2048 keys puts tens of percent on its top key: peaked enough
#: that dropping one key shows, flat enough that many keys contribute.
SIGMA = 1.5
SCALES = (entry.SOFTMAX_SCALE, entry.SOFTMAX_SCALE * math.log2(math.e))

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
    `2**-4 * sum_r p_r |v_r| = 2**-4 A`. Two bf16 roundings (a split-K partial
    and the output, 2**-9 each) add `2**-8 A`, and fp32 scores and exp2 well
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
    p, lens, layer, kind = spec["p"], spec["lens"], spec["layer"], spec["index_kind"]
    pages = [math.ceil(n / PAGE) for n in lens]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu)
    pool = (torch.randn(num_pages, LAYERS, PAGE, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    total_q = len(lens) * p
    q = (torch.randn(total_q, HEADS, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    kv = pool.reshape(-1).view(torch.uint8).reshape(-1, LATENT)
    rows = torch.full((total_q, TOP_K), -1, dtype=torch.int64)
    used = 0
    for b, n in enumerate(lens):
        table = order[used : used + pages[b]]
        used += pages[b]
        for t in range(p):
            causal = n - p + t + 1
            valid = min(max(causal, 0), TOP_K)
            pos = torch.randperm(causal, generator=cpu)[:valid]
            sel = table[pos // PAGE] * (LAYERS * PAGE) + layer * PAGE + pos % PAGE
            # The top-scoring key, in any head, goes to the last valid slot.
            k = kv[sel.cuda()].view(torch.float8_e4m3fn).float()
            best = int((q[b * p + t].float() @ k.t()).amax(0).argmax())
            sel[[best, valid - 1]] = sel[[valid - 1, best]]
            rows[b * p + t, :valid] = sel
    rows = rows.cuda()
    shift = layer * PAGE if kind == entry.INDEX_KIND_BASE_WITH_LAYER_OFFSET else 0
    indices = torch.where(rows >= 0, rows - shift, rows).to(torch.int32)
    args = dict(
        quant_q_buffer=q,
        kv_pool_view=pool,
        sparse_attn_indices=indices,
        output=torch.full(
            (total_q, HEADS * V_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        cu_q_seqlens=torch.arange(len(lens) + 1, dtype=torch.int32, device="cuda") * HEADS * p,
        sequence_length=torch.tensor(lens, dtype=torch.int32, device="cuda"),
        mla_bmm1_scale=torch.tensor(SCALES, dtype=torch.float32, device="cuda"),
        mla_bmm2_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        index_kind=kind,
        local_layer_idx=layer if shift else None,
    )
    return args, rows


def _gmem_batch(split: int) -> int | None:
    """The smallest batch whose kernel pick on this device is `split` CTAs with
    the global-memory reducer, or None when every batch of that split fits in
    one occupancy wave. A split above 1 needs at least `2 * max_sq * batch`
    SMs, so the search stops at the SM count."""
    dev = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for batch in range(1, sms + 1):
        if entry.split_k_variant(batch, device_index=dev) == (split, True):
            return batch
    return None


def _resolve(spec) -> dict | None:
    """`spec` with its lengths: a `gmem_split` cell gets the batch this device
    selects that reducer at, lengths crossing top_k both ways. None when the
    device never selects it."""
    if "lens" in spec:
        return dict(spec)
    batch = _gmem_batch(spec["gmem_split"])
    if batch is None:
        return None
    return dict(spec, lens=[spec["p"] + (347 * i) % 3000 for i in range(batch)])


@pytest.fixture(scope="module")
def compiled():
    """The handle as the target builds it, both index kinds, for every batch up
    to the largest any cell drives."""
    bound = max(len(r["lens"]) for c in op.CELLS if (r := _resolve(c.spec)) is not None)
    return entry.compile(max_batch_size=bound, device_index=torch.cuda.current_device())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    spec = _resolve(cell.spec)
    if spec is None:
        # Before arming: an armed entry that never runs fails its teardown.
        pytest.skip(
            f"no batch selects the {cell.spec['gmem_split']}-CTA split with the "
            "global-memory reducer on this device; the target never reaches it here"
        )
    certifying(op)
    args, rows = _build(spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args, compiled=compiled)
    torch.cuda.synchronize()
    ref = op.reference(**args, compiled=compiled)
    _compare(out, ref, _band(args["quant_q_buffer"], args["kv_pool_view"], rows, SCALES[0]))


def test_handle_covers_every_batch_up_to_a_large_bound() -> None:
    """The target builds its handle for the engine's max_batch_size, whatever
    it is: every batch up to that bound must find its pick in the handle, for
    both index kinds, and the compile count must stay bounded. Picks the cells
    already built come from the compile cache."""
    dev = torch.cuda.current_device()
    bound = 2048
    handle = entry.compile(max_batch_size=bound, device_index=dev)
    built = {(k.num_ctas_kv, k.use_gmem_reduction, k.index_kind) for k in handle.kernels}
    kinds = (entry.INDEX_KIND_GLOBAL, entry.INDEX_KIND_BASE_WITH_LAYER_OFFSET)
    for batch in range(1, bound + 1):
        variant = entry.split_k_variant(batch, device_index=dev)
        for kind in kinds:
            assert (*variant, kind) in built, (batch, variant, kind)
    # Splits 8, 4 and 2, each with or without the global-memory reducer, and 1.
    assert len(built) <= 7 * len(kinds), sorted(built)


def test_a_batch_above_the_handle_raises_rather_than_compiling() -> None:
    """Outside its handle the op raises: a compile on the serving path could
    land inside a CUDA-graph capture."""
    small = entry.compile(max_batch_size=2, device_index=torch.cuda.current_device())
    args, _ = _build(dict(p=2, lens=[40, 3000, 900], index_kind="global", layer=0), 7)
    with pytest.raises(ValueError, match="no kernel compiled for 3 sequences"):
        op(**args, compiled=small)


def _handle() -> "entry.CompiledSparseMlaDecode":
    """A handle with no kernels: `is_valid` reads only its `max_sq`."""
    return entry.CompiledSparseMlaDecode(
        device_index=torch.cuda.current_device(),
        max_batch_size=32,
        max_sq=entry.MAX_SQ,
        all_sparse_indices_valid=False,
        kernels={},
    )


_GUARD_SPEC = dict(p=2, lens=[40, 3000], index_kind="global", layer=0)


def test_is_valid_admits_a_well_formed_call() -> None:
    args, _ = _build(_GUARD_SPEC, 1)
    op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_hole_in_the_valid_prefix() -> None:
    """The kernel gathers the -1 as a pool row and mixes it into the softmax."""
    args, _ = _build(_GUARD_SPEC, 2)
    args["sparse_attn_indices"][0, 3] = -1
    with pytest.raises(AssertionError, match="inside a query's valid prefix"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_row_in_the_tail() -> None:
    """A sequence_length one short of the selection: the kernel masks the
    last selected row and returns a softmax over the rest."""
    args, _ = _build(_GUARD_SPEC, 3)
    args["sequence_length"][0] -= 1
    with pytest.raises(AssertionError, match="-1 tail"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_scale_the_kernel_ignores() -> None:
    """q_scaling 2 halves the softmax scale in the array; the kernel keeps 1/16."""
    args, _ = _build(_GUARD_SPEC, 4)
    args["mla_bmm1_scale"] = args["mla_bmm1_scale"] / 2
    with pytest.raises(AssertionError, match="compiled softmax scale"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_more_rows_than_max_sq() -> None:
    """Three query rows for one sequence and one for the other: the aggregate
    check passes, and the kernel skips the third row."""
    args, _ = _build(_GUARD_SPEC, 5)
    args["cu_q_seqlens"] = torch.tensor([0, 3 * HEADS, 4 * HEADS], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="max_sq"):
        op.is_valid(**args, compiled=_handle())


def test_is_valid_refuses_a_row_past_the_pool() -> None:
    args, _ = _build(_GUARD_SPEC, 6)
    pool_rows = args["kv_pool_view"].numel() // LATENT
    args["sparse_attn_indices"][1, 0] = pool_rows
    with pytest.raises(AssertionError, match="past the pool"):
        op.is_valid(**args, compiled=_handle())
