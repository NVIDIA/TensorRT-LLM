# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_kv_store_nvfp4 entry.

The paged state is real: one NVFP4 pool as the QSA cache manager serves it -- e2m1
data `[pages, 2, Hkv, tpb, D/2]` and e4m3 block scales `[pages, 2, Hkv, tpb, D/16]`,
K and V the two role planes of each -- and the engine's `[slots, width]` table, a
row per request slot, more slots than requests, live rows' pages drawn from a
seeded permutation, `-1` past each allocation. Both pools are pre-filled with a
sentinel so a stray write shows.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_kv_store_nvfp4 import (  # noqa: E501
    qsa_paged_kv_store_nvfp4 as op,
)
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_kv_store_nvfp4 requires a CUDA device"

DEV = "cuda"
DATA_FILL, SCALE_FILL = 0xA5, 0x5A
QUANT_MODE_NVFP4_KV = 8192


def _compare(out, ref) -> None:
    """Bitwise over all four planes: the store is a pure function of the row's
    bits, and `reference` performs the same rounding steps."""
    names = ("k_cache", "v_cache", "k_scale_cache", "v_scale_cache")
    for name, got, want in zip(names, out, ref):
        assert got.dtype == want.dtype and got.shape == want.shape, (
            name,
            got.shape,
            want.shape,
        )
        differ = (got != want).nonzero()
        if differ.numel():
            first = tuple(differ[0].tolist())
            raise AssertionError(
                f"{name}: {differ.shape[0]} bytes differ from the reference, first at "
                f"{first}: {int(got[first]):#04x} vs {int(want[first]):#04x}"
            )


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


def _fill_rows(spec, k, v, gen):
    """Row values: random with per-block magnitudes over 2^-10..2^5, plus edge blocks
    (zero, -0.0, an outlier) when the cell asks for them."""
    hkv, head_dim = spec["hkv"], spec["head_dim"]
    rows = k.shape[0]
    for t in (k, v):
        base = torch.randn(rows, hkv, head_dim // 16, 16, generator=gen)
        exponent = torch.randint(-10, 6, (rows, hkv, head_dim // 16, 1), generator=gen)
        t.copy_((base * torch.pow(2.0, exponent.float())).reshape(t.shape).to(DEV))
    if spec["values"] == "edge":
        k[0, 0, :16] = 0.0
        v[0, 1, 16:32] = -0.0
        k[1, 1, 32:48] = torch.tensor([-0.0] * 8 + [0.0] * 8)
        k[2, 0, 5] = 900.0
        v[3, 0, 200] = -900.0
        k[4, 1, 64:80] = 1e-30  # far below the smallest e4m3 scale
    else:
        assert spec["values"] == "randn"


def _pool(spec):
    """The allocation the cell's planes view: data and scales of `slot_pages` role
    pages per slot, every byte its sentinel."""
    pages, slot_pages = spec["pages"], spec["slot_pages"]
    hkv, tpb, head_dim = spec["hkv"], spec["tokens_per_block"], spec["head_dim"]
    data = torch.full(
        (pages, slot_pages, hkv, tpb, head_dim // 2), DATA_FILL, dtype=torch.uint8, device=DEV
    )
    scales = torch.full(
        (pages, slot_pages, hkv, tpb, head_dim // 16), SCALE_FILL, dtype=torch.uint8, device=DEV
    )
    return data, scales


def _planes(args):
    return (args["k_cache"], args["v_cache"], args["k_scale_cache"], args["v_scale_cache"])


def _reset(planes) -> None:
    for plane, fill in zip(planes, (DATA_FILL, DATA_FILL, SCALE_FILL, SCALE_FILL)):
        plane.fill_(fill)


def _assert_other_roles_untouched(spec, pool) -> None:
    """Every role page of a slot but the call's K and V still holds its sentinel."""
    data, scales = pool
    for role in range(spec["slot_pages"]):
        if role // 2 != spec["layer"]:
            assert bool((data[:, role] == DATA_FILL).all()), f"data role {role} moved"
            assert bool((scales[:, role] == SCALE_FILL).all()), f"scale role {role} moved"


def _build(spec, seed: int, pool=None):
    """Real pool, table and rows for one cell; returns the op's kwargs, the pool,
    and the packed QKV rows the K/V slices view. `pool` reuses an allocation
    from `_pool(spec)`."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim = spec["tokens_per_block"], spec["hkv"], spec["head_dim"]
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    used, allocated_cols = 0, []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)

    rows = len(positions)
    qkv = torch.zeros(
        rows, spec["heads_q"] + 2 * hkv, head_dim, dtype=spec["row_dtype"], device=DEV
    )
    k = qkv[:, spec["heads_q"] : spec["heads_q"] + hkv]
    v = qkv[:, spec["heads_q"] + hkv :]
    qkv[:, : spec["heads_q"]] = torch.randn(rows, spec["heads_q"], head_dim, generator=gen).to(
        device=DEV, dtype=spec["row_dtype"]
    )
    _fill_rows(spec, k, v, gen)

    layer = spec["layer"]
    data, scales = _pool(spec) if pool is None else pool
    args = dict(
        k=k,
        v=v,
        k_cache=data[:, 2 * layer],
        v_cache=data[:, 2 * layer + 1],
        k_scale_cache=scales[:, 2 * layer],
        v_scale_cache=scales[:, 2 * layer + 1],
        request_indices=torch.tensor(requests, dtype=spec["request_dtype"], device=DEV),
        logical_positions=torch.tensor(positions, dtype=spec["position_dtype"], device=DEV),
        block_table=table.to(DEV),
        tokens_per_block=tpb,
        kv_scale_orig_quant=torch.tensor(
            [1.0, spec["global_k"], spec["global_v"]], dtype=torch.float32, device=DEV
        ),
    )
    return args, (data, scales), qkv


def _snapshot(pool):
    return tuple(t.clone() for t in pool)


def _pools_equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _graph_replay(spec, seed: int, graph_batches) -> None:
    """Each G: capture one step of the first G requests, replay the next step.

    The next step lives in the captured buffers: new k/v rows, new pages in the
    table, and every request's positions advanced by its rows, the way a decode
    step moves. The replay must equal the reference and an eager call on the
    same inputs, bit for bit. One pool serves the whole sweep, as the engine's does.
    """
    pool = _pool(spec)
    steps = {n for _, n in spec["segments"]}
    assert len(steps) == 1, "a graph cell's requests all carry the same row count"
    step = steps.pop()
    for g in graph_batches:
        sub = {**spec, "segments": spec["segments"][:g]}
        args, _, _ = _build(sub, seed=seed + g, pool=pool)
        following, _, _ = _build(sub, seed=seed + 1000 + g, pool=pool)
        planes = _planes(args)
        positions = args["logical_positions"]
        next_positions = positions.clone()
        positions.sub_(step)
        assert int(positions.min()) >= 0

        def call():
            op(**args)

        for _ in range(3):  # compile the Triton kernel before capture
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()

        for name in ("k", "v", "block_table"):
            args[name].copy_(following[name])
        positions.copy_(next_positions)
        _reset(planes)
        op(**args)
        eager = tuple(plane.clone() for plane in planes)
        _reset(planes)
        ref = op.reference(**args)
        graph.replay()
        torch.cuda.synchronize()
        _compare(planes, ref)
        assert all(torch.equal(p, e) for p, e in zip(planes, eager)), (
            f"G={g} x {step} rows: replay differs from eager"
        )
        del graph
    _assert_other_roles_untouched(spec, pool)


def _thop_append(qkv, pool, table, requests, new, cached, num_contexts, p, scale_orig_quant):
    """The attention op's own append over the same rows (e4m3 output: the append
    does not depend on the output type)."""
    data, scales = pool
    tpb = 32
    hq, hkv, d = 24, 2, 256
    ns = len(requests)
    width = table.shape[1]
    offsets = torch.zeros(1, ns, 2, width, dtype=torch.int32, device=DEV)
    for row, r in enumerate(requests):
        live = table[r].clamp_min(0)
        offsets[0, row, 0] = 2 * live
        offsets[0, row, 1] = 2 * live + 1
    pointers = torch.zeros(1, 2, 2, dtype=torch.int64)
    pointers[0, 0, 0], pointers[0, 0, 1] = data.data_ptr(), scales.data_ptr()
    kv = [c + n for c, n in zip(cached, new)]
    ctx = [n if i < num_contexts else kv[i] for i, n in enumerate(new)]
    scale_quant_orig = scale_orig_quant.reciprocal()
    rows = qkv.reshape(qkv.shape[0], -1)
    FallbackFmha.attention(
        q=rows, k=None, v=None,
        output=torch.empty(rows.shape[0], hq * d, dtype=torch.float8_e4m3fn, device=DEV),
        output_sf=None, workspace=torch.empty(0, dtype=torch.int8, device=DEV),
        sequence_length=torch.tensor(kv, dtype=torch.int32, device=DEV),
        host_past_key_value_lengths=torch.tensor(kv, dtype=torch.int32),
        host_total_kv_lens=torch.tensor(
            [sum(kv[:num_contexts]), sum(kv[num_contexts:])], dtype=torch.int32
        ),
        context_lengths=torch.tensor(ctx, dtype=torch.int32, device=DEV),
        host_context_lengths=torch.tensor(ctx, dtype=torch.int32),
        host_request_types=torch.tensor(
            [0 if i < num_contexts else 1 for i in range(ns)], dtype=torch.int32
        ),
        max_context_q_len_override=None, kv_cache_block_offsets=offsets,
        host_kv_cache_pool_pointers=pointers,
        host_kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        cache_indirection=None, kv_scale_orig_quant=scale_orig_quant, kv_scale_quant_orig=scale_quant_orig,
        out_scale=torch.ones(1, dtype=torch.float32, device=DEV), rotary_inv_freq=None,
        rotary_cos_sin=None, latent_cache=None, q_pe=None, block_ids_per_seq=None,
        attention_sinks=None, is_fused_qkv=True, update_kv_cache=True,
        predicted_tokens_per_seq=p, local_layer_idx=0, num_heads=hq, num_kv_heads=hkv,
        head_size=d, tokens_per_block=tpb, max_num_requests=ns,
        max_context_length=width * tpb, max_seq_len=width * tpb,
        attention_window_size=width * tpb, beam_width=1, mask_type=1,
        quant_mode=QUANT_MODE_NVFP4_KV, q_scaling=1.0, position_embedding_type=0, rope_dim=0,
        rope_base=10000.0, rope_scale_type=0, rope_scale=1.0, rope_short_m_scale=1.0,
        rope_long_m_scale=1.0, rope_max_positions=1024, rope_original_max_positions=1024,
        use_paged_context_fmha=True, attention_input_type=0, is_mla_enable=False,
        chunked_prefill_buffer_batch_size=1, q_lora_rank=None, kv_lora_rank=None,
        qk_nope_head_dim=None, qk_rope_head_dim=None, v_head_dim=None, rope_append=None,
        mrope_rotary_cos_sin=None, mrope_position_deltas=None, helix_position_offsets=None,
        helix_is_inactive_rank=None, attention_chunk_size=None, softmax_stats_tensor=None,
        is_spec_decoding_enabled=False, use_spec_decoding=False, is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None, spec_decoding_position_offsets=None,
        spec_decoding_packed_mask=None, spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None, spec_bl_tree_first_sparse_mask_offset_kv=None,
        sparse_kv_indices=None, sparse_kv_offsets=None, sparse_attn_indices=None,
        sparse_attn_offsets=None, sparse_attn_indices_block_size=0, num_contexts=num_contexts,
        num_ctx_tokens=sum(new[:num_contexts]),
    )  # fmt: skip
    torch.cuda.synchronize()


def _thop_equality(spec, seed: int) -> None:
    """Steps of a context, a chunked continuation, decode P=1 and MTP verify P=3 over
    three requests, each written once by the attention op's append into one pool and
    once by this op into another, at the cell's second-level scales: the two pools stay
    bitwise equal, and this op's output equals its reference at every step."""
    base, ours, _ = _build(spec, seed=seed)
    theirs = _snapshot(ours)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    table = base["block_table"]
    cached = [0, 0, 0]
    for new, nc, p in (
        ([70, 33, 5], 3, 1),
        ([41, 64, 27], 3, 1),
        ([1, 1, 1], 0, 1),
        ([3] * 3, 0, 3),
    ):
        rows = sum(new)
        qkv = torch.randn(rows, 28, 256, generator=gen)
        qkv *= torch.pow(2.0, torch.randint(-6, 4, (rows, 28, 1), generator=gen).float())
        qkv = qkv.to(device=DEV, dtype=torch.bfloat16)
        requests = [r for r, n in enumerate(new) for _ in range(n)]
        positions = [c + i for c, n in zip(cached, new) for i in range(n)]
        _thop_append(qkv, theirs, table, [0, 1, 2], new, cached, nc, p, base["kv_scale_orig_quant"])
        args = {
            **base,
            "k": qkv[:, 24:26],
            "v": qkv[:, 26:28],
            "request_indices": torch.tensor(requests, dtype=torch.int32, device=DEV),
            "logical_positions": torch.tensor(positions, dtype=torch.int64, device=DEV),
        }
        ref = op.reference(**args)
        out = op(**args)
        _compare(out, ref)
        for plane, (a, b) in enumerate(zip(ours, theirs)):
            assert torch.equal(a, b), (
                f"step {new}: {('data', 'scales')[plane]} differ from the attention op's "
                f"append in {int((a != b).sum())} bytes"
            )
        cached = [c + n for c, n in zip(cached, new)]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode())
    if cell.spec["thop_append"]:
        _thop_equality(cell.spec, seed=seed)
        return
    args, pool, _ = _build(cell.spec, seed=seed)
    before = _snapshot(pool)
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["k_cache"] and out[3] is args["v_scale_cache"]
    _compare(out, ref)
    wrote = not _pools_equal(pool, before)
    assert wrote == bool(cell.spec["segments"]), (
        "a cell with rows must write, an empty one must not"
    )
    _assert_other_roles_untouched(cell.spec, pool)


def test_cuda_graph_replay_matches_eager() -> None:
    """Decode MTP under a graph: 8 requests x 8 rows in the prefill cell's trunk layout,
    captured once, replayed on the next step's rows, pages and positions."""
    prefill = next(c.spec for c in op.CELLS if c.why.startswith("prefill:"))
    spec = {**prefill, "segments": tuple((29 + 61 * i, 8) for i in range(8)), "pages": 256}
    _graph_replay(spec, seed=8, graph_batches=(8,))


def _decode_args(seed: int, **overrides):
    return _build({**op.CELLS[0].spec, **overrides}, seed=seed)


def test_op_rejects_strided_request_indices() -> None:
    """The callable's own loud error -- why the entry carries no `is_valid` -- for the
    metadata a flat-pointer kernel would otherwise read at the wrong rows."""
    args, _, _ = _decode_args(seed=5)
    strided = torch.stack([args["request_indices"]] * 2, 1)[:, 0]
    with pytest.raises(ValueError, match="request_indices must be contiguous"):
        with validating(op):
            op(**{**args, "request_indices": strided})
