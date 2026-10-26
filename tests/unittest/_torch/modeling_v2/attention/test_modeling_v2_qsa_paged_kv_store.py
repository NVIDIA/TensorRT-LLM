# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_kv_store catalog entry.

The paged state is real: one `[pages, slot_pages, Hkv, tpb, D]` HND pool whose
role pages `2 * layer` and `2 * layer + 1` are the K and V caches (2 pages per
slot for the qwen4_exp cells; 46 for Step-5, the K/V of its 23 sparse layers in
one slot of the full-attention pool), and the engine's `[slots, width]` table --
a row per request slot, each live row's written pages drawn from a seeded
permutation of the pool, its earlier columns either unique pages too or shared
pool pages (a reused prefix), `-1` past its allocation, and the rows past the
batch holding a previous step's plausible page ids. Every pool byte starts as a
sentinel, so a stray write shows.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_kv_store import (  # noqa: E501
    qsa_paged_kv_store as op,
)
from tensorrt_llm.bindings.internal import thop

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_kv_store requires a CUDA device"

DEV = "cuda"
FILL = -2.0
F8 = torch.float8_e4m3fn
QUANT_MODE_FP8_KV = 128


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.view(as_int)


def _fill_bits(dtype: torch.dtype) -> int:
    return int(_bits(torch.tensor([FILL]).to(dtype))[0])


def _compare(out, ref) -> None:
    """Bitwise over both role planes, compared as raw bits.

    A store is a move with one conversion, and `reference` performs the
    same conversion, so there is no rounding to spend a tolerance on. With
    `_other_pages_untouched` this is also the stray-write check: the planes
    and the slot's other role pages are the whole pool allocation.
    """
    for name, got, want in zip(("k_cache", "v_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        if not torch.equal(_bits(got), _bits(want)):
            differ = _bits(got) != _bits(want)
            first = tuple(differ.nonzero()[0].tolist())
            raise AssertionError(
                f"{name}: {int(differ.sum())} elements differ from the reference, first at {first}"
            )


def _other_pages_untouched(spec, pool) -> None:
    """Every role page of a slot but the call's K and V still holds the sentinel."""
    fill = _fill_bits(pool.dtype)
    for page in range(spec["slot_pages"]):
        if page // 2 != spec["layer"]:
            assert bool((_bits(pool[:, page]) == fill).all()), f"role page {page} of a slot moved"


def _pool(spec) -> torch.Tensor:
    pool = torch.empty(
        (
            spec["pages"],
            spec["slot_pages"],
            spec["hkv"],
            spec["tokens_per_block"],
            spec["head_dim"],
        ),
        dtype=spec["cache_dtype"],
        device=DEV,
    )
    _bits(pool).fill_(_fill_bits(spec["cache_dtype"]))
    return pool


def _hole(spec, table, allocated_cols):
    """One row that resolves to no slot: its (request, position)."""
    assert spec["hole"] == "unallocated_page", spec["hole"]
    assert int(table[0, allocated_cols[0]]) == -1
    return 0, allocated_cols[0] * spec["tokens_per_block"]


def _table(spec, gen):
    """`[slots, width]` int32 page table and each request's allocated column count."""
    tpb, slots, width, pages = (
        spec["tokens_per_block"],
        spec["slots"],
        spec["width"],
        spec["pages"],
    )
    segments = spec["segments"]
    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    used, allocated_cols, written = 0, [], []
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        first = 0 if spec["prefix"] == "unique" else start // tpb
        table[r] = -1
        table[r, first:cols] = order[used : used + cols - first]
        written += [(r, c) for c in range(start // tpb, cols)]
        used += cols - first
        allocated_cols.append(cols)
    assert used <= pages and len(segments) <= slots, "pool or table too small for the batch"
    if spec["include_last_page"]:
        # The last column written takes the pool's last slot (order[used - 1] was its page).
        last = pages - 1
        j = int((order == last).nonzero()[0, 0])
        if j >= used:
            order[j], order[used - 1] = order[used - 1].clone(), last
            r, c = written[-1]
            table[r, c] = last
        assert bool((table[[r for r, _ in written], [c for _, c in written]] == last).any())
    if spec["prefix"] == "shared":
        # Prefix columns hold pages this step does not write, shared across requests.
        spare = order[used:]
        for r, (start, _) in enumerate(segments):
            first = start // tpb
            if first:
                pick = torch.randint(0, spare.numel(), (first,), generator=gen)
                table[r, :first] = spare[pick]
    return table, allocated_cols


def _rows(spec, rows: int, seed: int, gen) -> torch.Tensor:
    """`[rows, heads_q + 2 * hkv + row_pad_heads, head_dim]` rows, zero past `real_dim`."""
    heads = spec["heads_q"] + 2 * spec["hkv"] + spec["row_pad_heads"]
    shape = (rows, heads, spec["head_dim"])
    if not spec["scaled_values"]:
        values = torch.randn(*shape, generator=gen, dtype=torch.float32)
        return values.to(device=DEV, dtype=spec["row_dtype"])
    vgen = torch.Generator(device=DEV).manual_seed(seed)
    values = torch.randn(*shape, generator=vgen, device=DEV)
    values *= torch.pow(2.0, torch.randint(-6, 4, (rows, heads, 1), generator=vgen, device=DEV))
    if spec["outlier_fraction"]:
        hit = torch.rand(*shape, generator=vgen, device=DEV) < spec["outlier_fraction"]
        hit[:, :, spec["real_dim"] :] = False
        big = 440.0 + 200.0 * torch.rand(*shape, generator=vgen, device=DEV)
        values = torch.where(hit, torch.sign(values) * big, values)
    values[:, :, spec["real_dim"] :] = 0.0
    return values.to(spec["row_dtype"])


def _build(spec, seed: int):
    """Real pool, table and rows for one cell; returns the op's kwargs and the pool."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]
    table, allocated_cols = _table(spec, gen)
    if spec["hole"] is not None:
        request, position = _hole(spec, table, allocated_cols)
        requests.append(request)
        positions.append(position)

    hq, hkv = spec["heads_q"], spec["hkv"]
    qkv = _rows(spec, len(positions), seed, gen)
    k = qkv[:, hq : hq + hkv]
    v = qkv[:, hq + hkv : hq + 2 * hkv]
    if spec["row_layout"] == "padded":
        k, v = k.contiguous(), v.contiguous()
    else:
        assert spec["row_layout"] == "packed", spec["row_layout"]

    pool = _pool(spec)
    layer = spec["layer"]
    args = dict(
        k=k,
        v=v,
        k_cache=pool[:, 2 * layer],
        v_cache=pool[:, 2 * layer + 1],
        request_indices=torch.tensor(requests, dtype=spec["request_dtype"], device=DEV),
        logical_positions=torch.tensor(positions, dtype=spec["position_dtype"], device=DEV),
        block_table=table.to(DEV),
        tokens_per_block=spec["tokens_per_block"],
    )
    return args, pool


def _wrote(args) -> bool:
    fill = _fill_bits(args["k_cache"].dtype)
    return not all(bool((_bits(args[name]) == fill).all()) for name in ("k_cache", "v_cache"))


_CELL_PARAMS = [
    pytest.param(c, id=c.why[:44], marks=(pytest.mark.large,) if c.spec["large"] else ())
    for c in op.CELLS
]


@pytest.mark.parametrize("cell", _CELL_PARAMS)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, pool = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["k_cache"] and out[1] is args["v_cache"]
    _compare(out, ref)
    _other_pages_untouched(cell.spec, pool)
    assert _wrote(args) == bool(cell.spec["segments"]), (
        "a cell with rows must write, an empty one must not"
    )
    if cell.spec["outlier_fraction"] and cell.spec["cache_dtype"] == F8:
        # The saturating case was reached: some stored value is +-448 (0x7e / 0xfe).
        real = cell.spec["real_dim"]
        assert any(bool(((_bits(t)[..., :real] & 0x7F) == 0x7E).any()) for t in out), (
            "no value reached the e4m3 saturation bound"
        )


def _decode_args(seed: int, **overrides):
    """The decode cell's layout over a bf16 pool, without its skipped row."""
    spec = {**op.CELLS[0].spec, "hole": None}
    return _build({**spec, **overrides}, seed=seed)


#: The cell of 32 MTP7 verify blocks (256 rows): the segments, pool and table it runs over.
_MTP7_32 = next(c.spec for c in op.CELLS if c.why.startswith("fp8 decode MTP7: 32 requests"))
_MTP7_BATCH = {key: _MTP7_32[key] for key in ("segments", "pages", "slots")}


def test_cuda_graph_replay_follows_updated_positions() -> None:
    """Captured once at 32 MTP7 verify blocks, replayed against new positions written
    into the same buffer."""
    args, pool = _decode_args(seed=9, **_MTP7_BATCH)
    positions = args["logical_positions"]
    saved = positions.clone()
    positions.zero_()

    def call():
        op(**args)

    for _ in range(3):  # compile the Triton kernel before capture
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    _bits(pool).fill_(_fill_bits(pool.dtype))
    positions.copy_(saved)
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare((args["k_cache"], args["v_cache"]), ref)


def test_cuda_graph_replay_step5_decode_128() -> None:
    """Step-5's 128-row decode (dp4, bf16 pool, 46-page slots, packed rows), captured
    once and replayed after the next step's positions, request indices, table and
    rows are written into the captured buffers."""
    spec = next(c.spec for c in op.CELLS if c.why.startswith("Step-5 dp4 decode 128 rows"))
    args, pool = _build(spec, seed=11)
    shifted = tuple(((start + 1) % 9216, n) for start, n in spec["segments"])
    nxt, _ = _build({**spec, "segments": shifted, "pages": 64 + spec["pages"] // 8}, seed=12)

    def call():
        op(**args)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):  # compile the Triton kernel before capture
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    for name in ("k", "v", "request_indices", "logical_positions", "block_table"):
        args[name].copy_(nxt[name])
    _bits(pool).fill_(_fill_bits(pool.dtype))
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare((args["k_cache"], args["v_cache"]), ref)
    _other_pages_untouched(spec, pool)
    assert _wrote(args)


def test_guard_refuses_strided_row_metadata() -> None:
    """`is_valid`'s one guard, after the hazard it exists for.

    The kernel receives strides for k, v, both caches and the table, but reads
    the two metadata vectors as flat unit-stride buffers. A column of a 2-wide
    buffer is read through its base pointer: accepted, and rows land elsewhere.
    """
    args, pool = _decode_args(seed=10)
    requests = args["request_indices"]
    rows = requests.numel()
    wide_requests = torch.full((rows, 2), rows - 1, dtype=requests.dtype, device=DEV)
    wide_requests[:, 0] = requests
    ref = op.reference(**args)
    strided = {**args, "request_indices": wide_requests[:, 0]}

    pool.fill_(FILL)
    out = op(**strided)  # unguarded: accepted
    assert not all(torch.equal(_bits(got), _bits(want)) for got, want in zip(out, ref)), (
        "strided metadata was expected to place rows elsewhere"
    )

    pool.fill_(FILL)
    before = pool.clone()
    with pytest.raises(AssertionError, match="must be a contiguous 1-D tensor"):
        with validating(op):
            op(**strided)
    assert torch.equal(_bits(pool), _bits(before)), "a refused call still wrote"


# ----------------------------------------------------------------------------
# The bytes the attention op's own append writes
# ----------------------------------------------------------------------------

#: Step-5 slot layout for the cross-check: the 23 sparse layers' K/V pages per slot,
#: the call on the last layer's pair.
_THOP_SLOT_PAGES, _THOP_LAYER, _THOP_TPB, _THOP_D = 46, 22, 32, 256


def _thop_append(qkv, pool, table, new, cached, num_contexts, hq, hkv, cache_dtype) -> None:
    """thop `attention` with `update_kv_cache=True` over the same rows, addressing
    the pool the way the engine does for this layout: the K lane of the block
    offsets holds `slot * slot_pages`, V one page after, and the pool mapping's
    layer offset moves both by `2 * layer` pages."""
    ns, width = len(new), table.shape[1]
    offsets = torch.zeros(1, ns, 2, width, dtype=torch.int32, device=DEV)
    for r in range(ns):
        live = table[r].clamp_min(0)
        offsets[0, r, 0] = _THOP_SLOT_PAGES * live
        offsets[0, r, 1] = _THOP_SLOT_PAGES * live + 1
    pointers = torch.zeros(1, 2, dtype=torch.int64)
    pointers[0, 0] = pool.data_ptr()
    mapping = torch.tensor([[0, _THOP_LAYER]], dtype=torch.int32)
    kv = [c + n for c, n in zip(cached, new)]
    ctx = [n if i < num_contexts else kv[i] for i, n in enumerate(new)]
    fp8 = cache_dtype == F8
    scale = torch.ones(1, dtype=torch.float32, device=DEV) if fp8 else None
    rows = qkv.reshape(qkv.shape[0], -1)
    thop.attention(
        q=rows, k=None, v=None,
        output=torch.empty(rows.shape[0], hq * _THOP_D, dtype=torch.bfloat16, device=DEV),
        output_sf=None, workspace_=torch.empty(0, dtype=torch.int8, device=DEV),
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
        host_kv_cache_pool_pointers=pointers, host_kv_cache_pool_mapping=mapping,
        cache_indirection=None, kv_scale_orig_quant=scale, kv_scale_quant_orig=scale,
        out_scale=None, rotary_inv_freq=None,
        rotary_cos_sin=None, latent_cache=None, q_pe=None, block_ids_per_seq=None,
        attention_sinks=None, is_fused_qkv=True, update_kv_cache=True,
        predicted_tokens_per_seq=1, local_layer_idx=0, num_heads=hq, num_kv_heads=hkv,
        head_size=_THOP_D, tokens_per_block=_THOP_TPB, max_num_requests=ns,
        max_context_length=width * _THOP_TPB, max_seq_len=width * _THOP_TPB,
        attention_window_size=width * _THOP_TPB, beam_width=1, mask_type=1,
        quant_mode=QUANT_MODE_FP8_KV if fp8 else 0, q_scaling=0.8660254,
        position_embedding_type=0, rope_dim=0,
        rope_base=10000.0, rope_scale_type=0, rope_scale=1.0, rope_short_m_scale=1.0,
        rope_long_m_scale=1.0, rope_max_positions=1024, rope_original_max_positions=1024,
        use_paged_context_fmha=True, attention_input_type=0, is_mla_enable=False,
        chunked_prefill_buffer_batch_size=1, q_lora_rank=None, kv_lora_rank=None,
        qk_nope_head_dim=None, qk_rope_head_dim=None, v_head_dim=None, rope_append=None,
        mrope_rotary_cos_sin=None, mrope_position_deltas=None, helix_position_offsets=None,
        helix_is_inactive_rank=None, attention_chunk_size=None, softmax_stats_tensor=None,
        is_spec_decoding_enabled=False, use_spec_decoding=False, is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None, spec_decoding_position_offsets_for_cpp=None,
        spec_decoding_packed_mask=None, spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None, spec_bl_tree_first_sparse_mask_offset_kv=None,
        sparse_kv_indices=None, sparse_kv_offsets=None, sparse_attn_indices=None,
        sparse_attn_offsets=None, sparse_attn_indices_block_size=0, num_contexts=num_contexts,
        num_ctx_tokens=sum(new[:num_contexts]),
    )  # fmt: skip
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "hkv,cache_dtype",
    [(4, torch.bfloat16), (4, F8), (1, torch.bfloat16), (1, F8)],
    ids=["dp4-bf16", "dp4-fp8", "tp4-bf16", "tp4-fp8"],
)
def test_matches_the_attention_ops_append(hkv: int, cache_dtype: torch.dtype) -> None:
    """A context step, a chunked continuation and two decode steps over three
    requests, each written once by thop's append into one pool and once by this op
    into another, at Step-5's heads (16 query heads per KV head, 256 wide with 192
    real), the 46-page slot layout, and 1% of values past 448: on every slot a row
    names, the two pools hold the same bytes, and this op equals its reference at
    every step.

    Outside the named slots thop's bf16 context append also zeroes the V plane past
    a context request's last token in its last page; this op leaves those slots as
    they were, which the reference comparison pins for this op only.
    """
    hq = 16 * hkv
    gen = torch.Generator(device="cpu").manual_seed(31 * hkv + (cache_dtype == F8))
    spec = dict(
        pages=64,
        slot_pages=_THOP_SLOT_PAGES,
        layer=_THOP_LAYER,
        hkv=hkv,
        tokens_per_block=_THOP_TPB,
        head_dim=_THOP_D,
        cache_dtype=cache_dtype,
        heads_q=hq,
        row_pad_heads=0,
        row_dtype=torch.bfloat16,
        scaled_values=True,
        outlier_fraction=0.01,
        real_dim=192,
    )
    width = 292
    order = torch.randperm(spec["pages"], generator=gen).to(torch.int32)
    table = torch.full((3, width), -1, dtype=torch.int32)
    for r in range(3):
        table[r, :8] = order[8 * r : 8 * r + 8]
    table = table.to(DEV)
    ours = _pool(spec)
    theirs = ours.clone()
    k_plane, v_plane = 2 * _THOP_LAYER, 2 * _THOP_LAYER + 1
    named = torch.zeros(spec["pages"], _THOP_TPB, dtype=torch.bool, device=DEV)
    cached = [0, 0, 0]
    for step, (new, num_contexts) in enumerate(
        (([70, 33, 5], 3), ([41, 64, 27], 3), ([1, 1, 1], 0), ([1, 1, 1], 0))
    ):
        qkv = _rows(spec, sum(new), seed=1000 * step + hkv, gen=gen)
        requests = [r for r, n in enumerate(new) for _ in range(n)]
        positions = [c + i for c, n in zip(cached, new) for i in range(n)]
        _thop_append(qkv, theirs, table, new, cached, num_contexts, hq, hkv, cache_dtype)
        args = dict(
            k=qkv[:, hq : hq + hkv],
            v=qkv[:, hq + hkv :],
            k_cache=ours[:, k_plane],
            v_cache=ours[:, v_plane],
            request_indices=torch.tensor(requests, dtype=torch.int32, device=DEV),
            logical_positions=torch.tensor(positions, dtype=torch.int64, device=DEV),
            block_table=table,
            tokens_per_block=_THOP_TPB,
        )
        ref = op.reference(**args)
        out = op(**args)
        _compare(out, ref)

        position = args["logical_positions"]
        pages = table[args["request_indices"].long(), position // _THOP_TPB].long()
        named[pages, position % _THOP_TPB] = True
        for plane in (k_plane, v_plane):
            got = _bits(ours[:, plane]).permute(0, 2, 1, 3)[named]
            want = _bits(theirs[:, plane]).permute(0, 2, 1, 3)[named]
            assert torch.equal(got, want), (
                f"step {new}: role page {plane} differs from the attention op's append in "
                f"{int((got != want).sum())} of {got.numel()} named elements"
            )
        cached = [c + n for c, n in zip(cached, new)]
    _other_pages_untouched(spec, ours)
    if cache_dtype == F8:
        saturated = (_bits(ours[:, k_plane : v_plane + 1]) & 0x7F) == 0x7E
        assert bool(saturated.any()), "no value reached the e4m3 saturation bound"
