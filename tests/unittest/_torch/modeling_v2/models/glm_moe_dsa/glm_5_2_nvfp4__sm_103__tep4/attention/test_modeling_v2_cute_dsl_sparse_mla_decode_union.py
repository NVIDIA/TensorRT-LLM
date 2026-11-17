# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_sparse_mla_decode_union entry (MTP3 / MTP5).

The pool, query and band are the per-row entry's (`test_modeling_v2_cute_dsl_sparse_mla_decode`):
the same synthetic `[pages, 3 layers, 32 tokens, 576]` e4m3 pool, each query row a random 2048
of its causal prefix with the top-scoring key at the last valid slot, and the same fp64 band.
Here the rows are passed as request-local positions with the sequences' block table, as the
indexer hands them to the target, and every output is gated against the per-row op's fp64
reference over the pool rows those positions name.
"""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_sparse_mla_decode as row_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_sparse_mla_decode_union as entry,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name
from test_modeling_v2_cute_dsl_sparse_mla_decode import (  # noqa: E402,F401 — fixture by name
    HEADS,
    LATENT,
    LAYERS,
    PAGE,
    SCALES,
    SIGMA,
    TOP_K,
    V_DIM,
    _assert_discriminative,
    _band,
    _compare,
    compiled,
    op,
)

union_op = entry.cute_dsl_sparse_mla_decode_union


def _build_union(spec, seed):
    """`(args, rows)` for the union op: `_build`'s pool, query and selections (each row a
    random 2048 of its causal prefix, top key last), passed as request-local positions with
    the sequences' block table; `rows` the global pool rows they name, -1 past the prefix.
    `widths` (optional) gives each sequence's row count (a runtime partial batch)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    p, lens, layer = spec["p"], spec["lens"], spec["layer"]
    widths = spec.get("widths", [p] * len(lens))
    pages = [max(math.ceil(n / PAGE), 1) for n in lens]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu)
    pool = (torch.randn(num_pages, LAYERS, PAGE, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    total_q = sum(widths)
    q = (torch.randn(total_q, HEADS, LATENT, generator=g, device="cuda") * SIGMA).to(
        torch.float8_e4m3fn
    )
    kv = pool.reshape(-1).view(torch.uint8).reshape(-1, LATENT)
    positions = torch.full((total_q, TOP_K), -1, dtype=torch.int32)
    rows = torch.full((total_q, TOP_K), -1, dtype=torch.int64)
    max_blocks = max(pages) + 2
    table = torch.full((len(lens), max_blocks), -1, dtype=torch.int32)
    req_idx = torch.empty(total_q, dtype=torch.int32)
    used = qi = 0
    for b, (n, w) in enumerate(zip(lens, widths)):
        tab = order[used : used + pages[b]]
        used += pages[b]
        table[b, : pages[b]] = tab.to(torch.int32)
        for t in range(w):
            req_idx[qi] = b
            causal = n - w + t + 1
            valid = min(max(causal, 0), TOP_K)
            if valid > 0:
                pos = torch.randperm(causal, generator=cpu)[:valid]
                sel = tab[pos // PAGE] * (LAYERS * PAGE) + layer * PAGE + pos % PAGE
                k = kv[sel.cuda()].view(torch.float8_e4m3fn).float()
                best = int((q[qi].float() @ k.t()).amax(0).argmax())
                pos[[best, valid - 1]] = pos[[valid - 1, best]]
                sel[[best, valid - 1]] = sel[[valid - 1, best]]
                positions[qi, :valid] = pos.to(torch.int32)
                rows[qi, :valid] = sel
            qi += 1
    cu = torch.tensor([0] + torch.tensor(widths).cumsum(0).tolist(), dtype=torch.int32) * HEADS
    args = dict(
        quant_q_buffer=q,
        kv_pool_view=pool,
        topk_positions=positions.cuda(),
        block_table=table.cuda(),
        req_idx=req_idx.cuda(),
        output=torch.full(
            (total_q, HEADS * V_DIM), float("nan"), dtype=torch.bfloat16, device="cuda"
        ),
        cu_q_seqlens=cu.cuda(),
        sequence_length=torch.tensor(lens, dtype=torch.int32, device="cuda"),
        layer_id=layer,
        stride_factor=LAYERS * PAGE,
    )
    return args, rows.cuda()


@pytest.fixture(scope="module")
def compiled_union():
    """Union handles by `max_sq` with a plan for every batch up to the largest any union cell
    of that width drives, served by the target or not (the kernel answers every batch; which
    ones the target sends it is a speed choice, `union_batches`)."""
    bounds: dict[int, int] = {}
    for c in union_op.CELLS:
        bounds[c.spec["p"]] = max(bounds.get(c.spec["p"], 0), len(c.spec["lens"]))
    dev = torch.cuda.current_device()
    return {
        w: entry.compile_union(
            max_batch_size=b, device_index=dev, max_sq=w, batches=range(1, b + 1)
        )
        for w, b in bounds.items()
    }


@pytest.mark.parametrize("cell", union_op.CELLS, ids=[c.why[:44] for c in union_op.CELLS])
def test_union_certified_cells(cell, compiled_union, certifying) -> None:  # noqa: F811
    certifying(union_op)
    spec = cell.spec
    handle = compiled_union[spec["p"]]
    args, rows = _build_union(spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = union_op(**args, compiled=handle)
    torch.cuda.synchronize()
    ref = union_op.reference(**args, compiled=handle)
    band = _band(args["quant_q_buffer"], args["kv_pool_view"], rows, SCALES[0])
    # Rows with no selection at all (a sequence shorter than its row's causal offset) have
    # no key to drop: certify the rows that have one.
    has_key = (rows >= 0).any(-1)
    a, q2, s_, d = band
    _assert_discriminative((a[has_key], q2[has_key], s_[has_key], d[has_key]))
    _compare(out, ref, band)


#: MTP3 / MTP5 at 32 sequences (the census batch; MTP3's plan keeps per-thread sums), MTP3 at
#: 24 (4-row groups), and MTP5 at 3 (the widest cluster split).
_UNION_REPLAY = {
    "mtp3": dict(p=4, lens=[4 + 271 * i for i in range(32)], layer=2),
    "mtp3_b24": dict(p=4, lens=[2 + 359 * i for i in range(24)], layer=1),
    "mtp5": dict(p=6, lens=[6 + 263 * i for i in range(32)], layer=0),
    "mtp5_b3": dict(p=6, lens=[9, 2050, 3333], layer=1),
}


@pytest.mark.parametrize("name", list(_UNION_REPLAY))
def test_union_graph_replay_is_bitwise_eager(name, compiled_union) -> None:
    """Captured in the decode graph: a replay writes the eager call's bytes, and a replay on new
    inputs copied into the same buffers writes that call's bytes."""
    spec = _UNION_REPLAY[name]
    handle = compiled_union[spec["p"]]
    args, _ = _build_union(spec, 41)
    fresh, _ = _build_union(spec, 42)
    eager = union_op(**args, compiled=handle).clone()
    eager_fresh = union_op(**fresh, compiled=handle).clone()
    again = union_op(**args, compiled=handle).clone()
    assert torch.equal(again.view(torch.int16), eager.view(torch.int16)), "not deterministic"
    args["output"].fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        union_op(**args, compiled=handle)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args["output"].view(torch.int16), eager.view(torch.int16))
    for key in (
        "quant_q_buffer",
        "kv_pool_view",
        "topk_positions",
        "block_table",
        "req_idx",
        "sequence_length",
    ):
        args[key].copy_(fresh[key])
    args["output"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args["output"].view(torch.int16), eager_fresh.view(torch.int16))


@pytest.mark.parametrize("max_sq", [4, 6], ids=["mtp3", "mtp5"])
def test_union_handle_covers_every_batch_up_to_a_large_bound(max_sq) -> None:
    """Every batch the union kernel serves on this device (`union_batches`) up to the engine's
    bound finds its compiled pick, and no other batch has one (it takes the per-row kernel);
    the picks keep every group's cluster in one occupancy wave and use at most 8 chunks."""
    dev = torch.cuda.current_device()
    handle = entry.compile_union(max_batch_size=256, device_index=dev, max_sq=max_sq)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    served = entry.union_batches(dev, max_sq, 256)
    sums = entry.UNION_THREAD_SUMS.get((sms, max_sq), frozenset())
    assert set(handle.plans) == set(served)
    for batch in sorted(served):
        rg, chunks, thread_sums = handle.plans[batch]
        assert (rg, chunks, thread_sums) == entry.union_plan(batch, device_index=dev, max_sq=max_sq)
        assert thread_sums == (batch in sums)
        groups = (max_sq + rg - 1) // rg
        assert 1 <= chunks <= 8 and (chunks == 1 or batch * groups * chunks <= sms), (
            batch,
            rg,
            chunks,
        )
        assert rg != 4 or chunks >= 2, (batch, rg, chunks)
    assert len(set(handle.plans.values())) <= 10


def test_union_serves_the_measured_batches_on_b300() -> None:
    """On a 148-SM B300 the target sends MTP5 at 13..15 and 24..32 sequences and MTP3 at
    19..26 and 30..32 to the union kernel (`UNION_SERVED`); the census batches, MTP5 and MTP3
    at 32, take 3-row groups on 2-chunk clusters and 2-row groups on 2-chunk clusters with
    per-thread sums; MTP3 at 23..26 4-row groups on 5-chunk clusters with per-thread sums;
    MTP5 at 14 and MTP3 at 21 2-row groups on 3-chunk ones, as before."""
    dev = torch.cuda.current_device()
    if torch.cuda.get_device_properties(dev).multi_processor_count != 148:
        pytest.skip("the served batches were measured on a 148-SM B300")
    assert entry.union_batches(dev, 6, 32) == frozenset([13, 14, 15, *range(24, 33)])
    assert entry.union_batches(dev, 4, 32) == frozenset([*range(19, 27), *range(30, 33)])
    assert entry.union_plan(32, device_index=dev, max_sq=6) == (3, 2, False)
    assert entry.union_plan(14, device_index=dev, max_sq=6) == (2, 3, False)
    assert entry.union_plan(21, device_index=dev, max_sq=4) == (2, 3, False)
    assert entry.union_plan(32, device_index=dev, max_sq=4) == (2, 2, True)
    assert entry.union_plan(24, device_index=dev, max_sq=4) == (4, 5, True)
    assert entry.union_plan(23, device_index=dev, max_sq=4) == (4, 5, True)
    # The MTP3 CUDA-graph batch 16 keeps the per-row kernel (one wave of split-K 2 there).
    assert 16 not in entry.union_batches(dev, 4, 32)


#: Rows per group x cluster chunks the kernel compiles, every one: (2 | 3 | 4) x 1..8, except 4
#: rows on 1 chunk (refused: SMEM); per-thread sums at a few of them.
_UNION_KERNEL_SPLITS = [
    *((rg, c, False) for rg in (2, 3, 4) for c in range(1, 9) if (rg, c) != (4, 1)),
    (2, 1, True),
    (2, 2, True),
    (2, 8, True),
    (3, 2, True),
    (4, 2, True),
    (4, 5, True),
    (4, 8, True),
]


@pytest.mark.parametrize(
    "rg,chunks,thread_sums",
    _UNION_KERNEL_SPLITS,
    ids=[f"rg{r}c{c}{'t' if t else ''}" for r, c, t in _UNION_KERNEL_SPLITS],
)
def test_union_kernel_every_group_width_and_split(rg, chunks, thread_sums) -> None:
    """The kernel itself at every rows-per-group / chunk split it takes, whatever the target
    serves: lengths 1..6, 2045..2053, 6000 and past the position map, within the fp64 band
    with a discriminative reference; MTP3 width for 2 and 4 rows, MTP5 for 3."""
    p = 6 if rg == 3 else 4
    lens = [1, 2, 3, 4, 5, 6, *range(2045, 2054), 6000, entry.UNION_MASK_CAP + 37, 3000]
    spec = dict(p=p, lens=lens, layer=1)
    args, rows = _build_union(spec, 1000 * rg + 10 * chunks + thread_sums)
    entry._union_kernel.prepare_union_runner(
        args["quant_q_buffer"],
        args["kv_pool_view"],
        args["topk_positions"],
        args["block_table"],
        args["req_idx"],
        args["output"],
        args["cu_q_seqlens"],
        args["sequence_length"],
        layer_offset=spec["layer"] * PAGE,
        stride_factor=args["stride_factor"],
        max_sq=p,
        rows_per_group=rg,
        num_chunks=chunks,
        thread_sums=thread_sums,
    )()
    torch.cuda.synchronize()
    ref = union_op.reference(**args, compiled=None)
    band = _band(args["quant_q_buffer"], args["kv_pool_view"], rows, SCALES[0])
    has_key = (rows >= 0).any(-1)
    a, q2, s_, d = band
    _assert_discriminative((a[has_key], q2[has_key], s_[has_key], d[has_key]))
    _compare(args["output"], ref, band)


def test_union_kernel_refuses_four_row_groups_on_one_chunk() -> None:
    with pytest.raises(ValueError, match="at least 2 chunks"):
        entry._union_kernel.UnionSpecialization(rows_per_group=4, num_chunks=1).validate()


def test_union_launcher_refuses_padded_cu_q_seqlens_and_strided_output() -> None:
    """The grid's z extent comes from `cu_q_seqlens`, so a padded one would read
    `sequence_length` past its end; a non-contiguous `output` would be written through a copy."""
    args, _ = _build_union(dict(p=4, lens=[3, 2050], layer=0), 5)
    names = ("quant_q_buffer", "kv_pool_view", "topk_positions", "block_table", "req_idx")
    head = [args[n] for n in names]
    out, cu, lens = args["output"], args["cu_q_seqlens"], args["sequence_length"]
    kw = dict(
        layer_offset=0,
        stride_factor=args["stride_factor"],
        max_sq=4,
        rows_per_group=2,
        num_chunks=2,
    )
    padded = torch.cat([cu, cu[-1:]])
    with pytest.raises(ValueError, match="cu_q_seqlens must have 3 entries"):
        (
            call_quant_q_buffer,
            call_kv_pool_view,
            call_topk_positions,
            call_block_table,
            call_req_ids,
        ) = head
        entry._union_kernel.prepare_union_runner(
            call_quant_q_buffer,
            call_kv_pool_view,
            call_topk_positions,
            call_block_table,
            call_req_ids,
            out,
            padded,
            lens,
            **kw,
        )
    strided = torch.empty(out.shape[::-1], dtype=out.dtype, device=out.device).t()
    with pytest.raises(ValueError, match="output must be contiguous"):
        (
            call_quant_q_buffer,
            call_kv_pool_view,
            call_topk_positions,
            call_block_table,
            call_req_ids,
        ) = head
        entry._union_kernel.prepare_union_runner(
            call_quant_q_buffer,
            call_kv_pool_view,
            call_topk_positions,
            call_block_table,
            call_req_ids,
            strided,
            cu,
            lens,
            **kw,
        )


@pytest.mark.parametrize("batch", list(range(12, 33)) + [1, 3])
def test_union_mtp3_every_batch(batch, compiled_union) -> None:
    """MTP3 at 1, 3 and 12..32 sequences on the handle's plan for that batch (the served ones
    as served), lengths cycling through 1..6, 2045..2053 and 6000, the first sequence a partial
    one of 2 rows; within the fp64 band."""
    cyc = [1, 2, 3, 4, 5, 6, *range(2045, 2054), 6000]
    lens = [cyc[(7 * i) % len(cyc)] for i in range(batch)]
    widths = [2] + [4] * (batch - 1)
    spec = dict(p=4, lens=lens, widths=widths, layer=batch % 3)
    args, rows = _build_union(spec, 77 + batch)
    out = union_op(**args, compiled=compiled_union[4])
    torch.cuda.synchronize()
    ref = union_op.reference(**args, compiled=compiled_union[4])
    band = _band(args["quant_q_buffer"], args["kv_pool_view"], rows, SCALES[0])
    has_key = (rows >= 0).any(-1)
    a, q2, s_, d = band
    _assert_discriminative((a[has_key], q2[has_key], s_[has_key], d[has_key]))
    _compare(out, ref, band)


_UNION_VS_PER_ROW = {
    "mtp5_b32": dict(p=6, lens=[2850 + 7 * i for i in range(32)], layer=1),
    "mtp3_b32": dict(p=4, lens=[2850 + 7 * i for i in range(32)], layer=2),
    "mtp3_b24": dict(p=4, lens=[1500 + 107 * i for i in range(24)], layer=0),
}


@pytest.mark.parametrize("name", list(_UNION_VS_PER_ROW))
def test_union_matches_the_per_row_kernel_within_the_band(name, compiled_union, compiled) -> None:  # noqa: F811
    """The same selections through both paths: each within the band of the fp64 reference
    (the union kernel sums the rows in another order, so not bitwise)."""
    spec = _UNION_VS_PER_ROW[name]
    w = spec["p"]
    args, rows = _build_union(spec, 5)
    out_u = union_op(**args, compiled=compiled_union[w]).clone()
    per_row = dict(
        quant_q_buffer=args["quant_q_buffer"],
        kv_pool_view=args["kv_pool_view"],
        sparse_attn_indices=rows.to(torch.int32),
        output=torch.full_like(args["output"], float("nan")),
        cu_q_seqlens=args["cu_q_seqlens"],
        sequence_length=args["sequence_length"],
        mla_bmm1_scale=torch.tensor(SCALES, dtype=torch.float32, device="cuda"),
        mla_bmm2_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        index_kind=row_entry.INDEX_KIND_GLOBAL,
        local_layer_idx=None,
    )
    out_w = op(**per_row, compiled=compiled[w]).clone()
    band = _band(args["quant_q_buffer"], args["kv_pool_view"], rows, SCALES[0])
    ref = union_op.reference(**args, compiled=compiled_union[w])
    _compare(out_u, ref, band)
    _compare(out_w, ref, band)


_UNION_GUARD = dict(p=4, lens=[40, 3000], layer=0)


def _union_handle() -> "entry.CompiledSparseMlaDecodeUnion":
    return entry.CompiledSparseMlaDecodeUnion(
        device_index=torch.cuda.current_device(), max_batch_size=32, max_sq=4, plans={}
    )


def test_union_is_valid_admits_a_well_formed_call() -> None:
    args, _ = _build_union(_UNION_GUARD, 1)
    union_op.is_valid(**args, compiled=_union_handle())


def test_union_is_valid_refuses_a_repeated_position() -> None:
    """The union counts a position once; the per-row reference would count it twice."""
    args, _ = _build_union(_UNION_GUARD, 2)
    args["topk_positions"][1, 5] = args["topk_positions"][1, 6]
    with pytest.raises(AssertionError, match="repeated position"):
        union_op.is_valid(**args, compiled=_union_handle())


def test_union_is_valid_refuses_a_position_in_the_tail() -> None:
    """A sequence_length one short of the selection: the kernel ignores the row's last entry."""
    args, _ = _build_union(_UNION_GUARD, 3)
    args["sequence_length"][0] -= 1
    with pytest.raises(AssertionError, match="after the row's valid prefix"):
        union_op.is_valid(**args, compiled=_union_handle())


def test_union_is_valid_refuses_a_position_past_the_sequence() -> None:
    args, _ = _build_union(_UNION_GUARD, 4)
    args["topk_positions"][2, 0] = int(args["sequence_length"][0])
    with pytest.raises(AssertionError, match="outside"):
        union_op.is_valid(**args, compiled=_union_handle())
