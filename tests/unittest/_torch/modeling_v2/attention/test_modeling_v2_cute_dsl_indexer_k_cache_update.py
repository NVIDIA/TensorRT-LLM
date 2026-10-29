# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_indexer_k_cache_update catalog entry."""

import math
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_k_cache_update import (  # noqa: E501
    NORM_EPS,
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_k_cache_update import (  # noqa: E501
    cute_dsl_indexer_k_cache_update as op,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _compute_slot_mappings
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), (
    "the CuTe DSL kernel runs on a GPU; this test cannot run without one"
)

HEAD_DIM, ROPE_DIM = 128, 64
#: Page size and per-token bytes of the indexer K pool (128 codes + one fp32 scale).
PAGE, TOKEN_BYTES = 32, 132
#: The layer of the pool a cell writes unless its spec names one: neither the
#: first nor the last, so a stray write on either side shows.
LAYER = 3
POISON = 0xA7
#: Mistral Large 4's `rope_parameters` (interleaved cells): YaRN with no magnitude
#: scaling, and LeRoPE's per-frequency exp(alpha); its max_seq_len bounds the table.
#: These are the FP8 checkpoint's factor and the FP8 target's max_seq_len; a
#: cell's `yarn_factor` and `max_seq_len` replace them (the NVFP4 checkpoint's
#: factor is 2048, its target's max_seq_len 32768, or 1048576 at its own context: the
#: `_ML4_1M_TP4` and `_ML4_1M` cells).
ML4_THETA, ML4_FACTOR, ML4_ORIGINAL_MAX, ML4_BETA_FAST, ML4_BETA_SLOW = 1.0e4, 512.0, 512, 32.0, 1.0
ML4_MAX_SEQ_LEN = 9216
#: The engine's token budget per step at the 1M envelope (the chunk size); a
#: manager at max_seq_len 32768 gets the same.
MAX_NUM_TOKENS = 32768
#: Tables up to this many rows are evaluated on the host; longer ones (the
#: 1048576-row tables, 2^26 cos and sin) in fp64 on the device. A row depends
#: only on its index. tp4's `table_dtype` cells evaluate theirs on the host at
#: every length, as tp4's test did (`_table`).
HOST_TABLE_ROWS = 32768
#: Mistral Large 4's index_sharing (params.json's DSA sharing), one letter per layer: layers 0..16
#: run their own indexer, then each 4 layers are 2 'shared' and 2 'full', and
#: layer 71 is full. Pool row r of the indexer K cache is the r-th full layer.
ML4_INDEX_PATTERN = "F" * 17 + "SSFF" * 13 + "SSF"
ML4_FULL_LAYERS = [i for i, role in enumerate(ML4_INDEX_PATTERN) if role == "F"]
assert len(ML4_INDEX_PATTERN) == 72 and len(ML4_FULL_LAYERS) == 44
GIB = 2**30

_COMPILED: dict = {}


def _compiled(is_neox: bool, cache_contiguous: bool):
    key = (bool(is_neox), bool(cache_contiguous))
    if key not in _COMPILED:
        _COMPILED[key] = build(
            torch.cuda.current_device(), is_neox=is_neox, cache_contiguous=cache_contiguous
        )
    return _COMPILED[key]


def _as_table(angles: torch.Tensor) -> torch.Tensor:
    """`[positions, cos(32) | sin(32)]` in fp32, viewed `[positions, 2, 32]` as the op reads it."""
    table = torch.cat([angles.cos(), angles.sin()], dim=1).float().cuda().contiguous()
    return table.view(-1, 2, ROPE_DIM // 2)


def _table_device(positions: int) -> str:
    return "cpu" if positions <= HOST_TABLE_ROWS else "cuda"


def _yarn_table(
    positions: int, gen: torch.Generator, factor: float = ML4_FACTOR, device: str | None = None
) -> torch.Tensor:
    """Mistral Large 4's indexer table: YaRN frequencies, each scaled by a LeRoPE exp(alpha).

    The YaRN ramp runs between the correction dims of beta_fast and beta_slow
    and is taken on the unscaled frequencies; alpha is a checkpoint tensor, so a
    seeded draw stands in for it. The angles and their cos and sin are taken on
    `device`, by default `_table_device(positions)`.
    """
    half = ROPE_DIM // 2
    pos_freqs = ML4_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float64) / ROPE_DIM)
    lerope = torch.exp(0.1 * torch.randn(half, generator=gen, dtype=torch.float64))
    extrapolation, interpolation = lerope / pos_freqs, lerope / (factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        return (ROPE_DIM * math.log(ML4_ORIGINAL_MAX / (rotations * 2 * math.pi))) / (
            2 * math.log(ML4_THETA)
        )

    low = max(math.floor(correction_dim(ML4_BETA_FAST)), 0)
    high = min(math.ceil(correction_dim(ML4_BETA_SLOW)), ROPE_DIM - 1)
    ramp = ((torch.arange(half, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    inv = (interpolation * ramp + extrapolation * (1 - ramp)).to(device or _table_device(positions))
    t = torch.arange(positions, dtype=torch.float64, device=inv.device)
    return _as_table(t[:, None] * inv[None, :])


def _identity_table(positions: int, device: str | None = None) -> torch.Tensor:
    """A NoPE layer's table: cos 1, sin 0 at every position."""
    dev = device or _table_device(positions)
    return _as_table(torch.zeros(positions, ROPE_DIM // 2, dtype=torch.float64, device=dev))


def _keys_and_norm(spec, tokens: int, g: torch.Generator):
    """`(indexer_k, norm_weight, norm_bias)`: bf16 keys (misaligned when the spec says so)."""
    keys = torch.randn(tokens, HEAD_DIM, generator=g, device="cuda") * 3.0 + 0.5
    if spec["misalign"]:
        storage = torch.empty(tokens * HEAD_DIM + 1, dtype=torch.bfloat16, device="cuda")
        indexer_k = storage[1:].view(tokens, HEAD_DIM)
        indexer_k.copy_(keys)
        assert indexer_k.data_ptr() % 8 != 0
    else:
        indexer_k = keys.to(torch.bfloat16)
    norm_weight = 1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")
    norm_bias = 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")
    return indexer_k, norm_weight, norm_bias


def _table(spec, cpu: torch.Generator):
    """`(table, rows)`: the cell's rope table and its row count, the positions it serves.

    `table_dtype="bfloat16"` rounds the values to bf16 and keeps them in fp32, as
    the NVFP4 target holds its tables (the op reads fp32). Only tp4's cells carry
    it; their table is evaluated on the host at every length, as tp4's test did,
    and every other cell's on `_table_device`'s device, as tep4's and dep4's did
    (the host up to `HOST_TABLE_ROWS`, as the mainline's).
    """
    max_pos = spec.get("max_seq_len", ML4_MAX_SEQ_LEN)
    device = "cpu" if "table_dtype" in spec else None
    if spec["table"] == "yarn":
        table = _yarn_table(max_pos, cpu, spec.get("yarn_factor", ML4_FACTOR), device)
    else:
        table = _identity_table(max_pos, device)
    if spec.get("table_dtype") == "bfloat16":
        table = table.to(torch.bfloat16).float()
    return table, max_pos


def _positions(spec) -> torch.Tensor:
    """A step's positions from `requests` and `cached`: request i's rows at
    `cached[i]..cached[i] + requests[i] - 1`, in request order. (tep4's and dep4's
    `_request_positions`, the same function under another name.)"""
    new = list(spec["requests"])
    cached = list(spec.get("cached", (0,) * len(new)))
    assert sum(new) == spec["tokens"], (sum(new), spec["tokens"])
    return torch.cat([torch.arange(c, c + n) for c, n in zip(cached, new)])


def _build(spec, seed):
    """`(args, pool, layer)` for one cell; `args` is the op's positional argument list.

    Without a `layout` the pool is `[blocks, layers, page * 132]` uint8,
    block-major over layers as the V1 DSA cache manager allocates the indexer K
    cache, and the cache is one layer's `[blocks, page, 1, 132]` view of it. With
    `layout="v2"` the pool is the same bytes, flat, and the cache is the view
    `DSACacheManagerV2` hands out (measured in the `manager_v2` cells): the
    contiguous `[blocks * layers - layer, page, 1, 132]` from the layer's first
    page, block b's page at `b * layers` in it. Each token gets a distinct
    (block, slot) and the TRT-LLM slot pair: codes at `page * page_size * 132 +
    slot * 128`, the scale at `page * page_size * 132 + page_size * 128 + slot *
    4`. Padding rows get -1. The slot mappings have `slot_rows` entries
    (default: one per token); past the pool's `blocks * page` slots the entries
    are -1, a tail no row reads. `far_end` puts the tokens in the pool's last
    blocks: ascending in V1's layout, and in V2's in the order V2 hands them out,
    a request's first page in the highest block. With `requests` the positions
    are each request's cached..cached+new-1 (`_positions`); without, 0..tokens-1
    above 64 tokens and random below.
    """
    assert spec.get("layout") in (None, "v2"), spec.get("layout")
    v2 = spec.get("layout") == "v2"
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    tokens, blocks, layers = spec["tokens"], spec["blocks"], spec["pool_layers"]
    layer = spec.get("layer", min(LAYER, layers - 1))
    if v2:
        page_bytes = PAGE * TOKEN_BYTES
        pool = torch.full((blocks * layers * page_bytes,), POISON, dtype=torch.uint8, device="cuda")
        k_cache = pool[layer * page_bytes :].view(blocks * layers - layer, PAGE, 1, TOKEN_BYTES)
    else:
        pool = torch.full(
            (blocks, layers, PAGE * TOKEN_BYTES), POISON, dtype=torch.uint8, device="cuda"
        )
        k_cache = pool[:, layer, :].view(blocks, PAGE, 1, TOKEN_BYTES)

    indexer_k, norm_weight, norm_bias = _keys_and_norm(spec, tokens, g)
    table, max_pos = _table(spec, cpu)
    if "requests" in spec:
        pos = _positions(spec).cuda()
    elif tokens > 64:
        pos = torch.arange(tokens, device="cuda")
    else:
        pos = torch.randint(0, max_pos, (tokens,), generator=g, device="cuda")
    if spec.get("top_position"):
        pos[0] = max_pos - 1

    slot_rows = spec.get("slot_rows", tokens)
    drawn = min(slot_rows, blocks * PAGE)
    assert drawn >= tokens, "the pool has fewer slots than the cell has rows"
    if spec.get("far_end") and v2:
        n = torch.arange(drawn)
        flat = (blocks - 1 - n // PAGE) * PAGE + n % PAGE
    elif spec.get("far_end"):
        flat = torch.arange(blocks * PAGE - drawn, blocks * PAGE)
    else:
        flat = torch.randperm(blocks * PAGE, generator=cpu)[:drawn]
    block, slot = flat // PAGE, flat % PAGE
    page = block * layers if v2 else block
    fp8_slot = page * PAGE * TOKEN_BYTES + slot * HEAD_DIM
    scale_slot = page * PAGE * TOKEN_BYTES + PAGE * HEAD_DIM + slot * 4
    if drawn < slot_rows:
        tail = torch.full((slot_rows - drawn,), -1, dtype=fp8_slot.dtype)
        fp8_slot, scale_slot = torch.cat([fp8_slot, tail]), torch.cat([scale_slot, tail])
    if spec["pad_rows"]:
        fp8_slot[tokens - spec["pad_rows"] : tokens] = -1
        scale_slot[tokens - spec["pad_rows"] : tokens] = -1
    args = [
        indexer_k,
        norm_weight,
        norm_bias,
        pos.to(torch.int32),
        table,
        k_cache,
        fp8_slot.cuda(),
        scale_slot.cuda(),
        tokens,
    ]
    return args, pool, layer


def _dsa_manager_v2(blocks: int, max_seq_len: int, max_num_tokens: int) -> DSACacheManagerV2:
    """The manager the engine builds for a Mistral Large 4 target by default
    (`use_kv_cache_manager_v2` defaults to True): `DSACacheManagerV2` over the
    72-layer fp8 latent pool beside the 44-row indexer pool, with a quota of
    `blocks` pages (`max_tokens`), at the target's max_seq_len and the engine's
    max_num_tokens (equal at 32768; 32768 at 1048576)."""
    return DSACacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_tokens=blocks * PAGE, host_cache_size=0
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=72,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=PAGE,
        max_seq_len=max_seq_len,
        max_batch_size=16,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=HEAD_DIM),
        pretrained_config=SimpleNamespace(
            num_hidden_layers=72,
            qk_rope_head_dim=ROPE_DIM,
            index_topk_pattern=list(ML4_INDEX_PATTERN),
        ),
        max_num_tokens=max_num_tokens,
    )


def _build_manager(spec, seed):
    """`(args, manager, pool, head)` for a `manager_v2` cell.

    A real `DSACacheManagerV2` holds one request per `requests` entry (that
    many new tokens after its `cached` ones); the cache is the view
    `get_indexer_k_cache_buffers(model_layer)` hands out and the slot mappings
    are the engine's own arithmetic (`_compute_slot_mappings` over the block ids
    times the manager's page scale), as the DSA metadata builds them. The
    manager runs at the spec's `max_num_tokens` (tp4's 1M cells name 32768) or,
    when it names none, at `min(max_seq_len, MAX_NUM_TOKENS)`: 32768 for every
    cell, as each member's test built it. The context count passed with the
    block ids is each member's own: in a cell naming `max_num_tokens` (tp4's), a
    request with no cached tokens or more than one new one is a context request,
    as tp4's test counted; in every other cell, a request with no cached tokens,
    as the mainline's, tep4's and dep4's tests counted (the two agree on the
    mainline's cells). `pool` is the first indexer row's view, which spans the
    whole indexer pool and is poisoned first; `head` is the byte the cell's view
    starts at in it. The view's geometry is asserted to be the one the `v2` cells
    rebuild. The pool is allocated outside torch's caching allocator, so what
    torch still caches from earlier cells is returned to the device first.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    new = list(spec["requests"])
    cached = list(spec.get("cached", (0,) * len(new)))
    pos = _positions(spec)
    max_num_tokens = spec.get("max_num_tokens", min(spec["max_seq_len"], MAX_NUM_TOKENS))
    torch.cuda.empty_cache()
    manager = _dsa_manager_v2(spec["blocks"], spec["max_seq_len"], max_num_tokens)
    try:
        assert sum(manager.indexer_k_cache_local_layer_mask) == spec["pool_layers"]
        assert manager.indexer_k_cache_page_scale == spec["pool_layers"]
        ids = list(range(len(new)))
        added = manager.add_dummy_requests(
            ids, [c + n for c, n in zip(cached, new)], is_gen=False, prepare_resource=True
        )
        assert added is not None, "the pool cannot hold the cell's requests"
        if "max_num_tokens" in spec:
            contexts = sum(1 for c, n in zip(cached, new) if c == 0 or n > 1)
        else:
            contexts = sum(1 for c in cached if c == 0)
        blocks = manager.get_pool_block_indices(len(ids), request_ids=ids, num_contexts=contexts)
        page_table = blocks.long() * manager.indexer_k_cache_page_scale
        req = torch.repeat_interleave(torch.arange(len(new)), torch.tensor(new))
        fp8_slot, scale_slot = _compute_slot_mappings(pos, page_table, req, HEAD_DIM, PAGE, 128)
        pool = manager.get_indexer_k_cache_buffers(ML4_FULL_LAYERS[0])
        pool.fill_(POISON)
        k_cache = manager.get_indexer_k_cache_buffers(spec["model_layer"])
        row = ML4_FULL_LAYERS.index(spec["model_layer"])
        head = row * PAGE * TOKEN_BYTES
        assert k_cache.is_contiguous(), k_cache.stride()
        assert tuple(k_cache.shape) == (pool.shape[0] - row, PAGE, 1, TOKEN_BYTES), k_cache.shape
        assert k_cache.data_ptr() - pool.data_ptr() == head
        print(
            f"\n[manager] quota {spec['blocks']} pages, {manager.num_blocks} slots, block table "
            f"{tuple(blocks.shape)}, max_seq_len {manager.max_seq_len}, max_num_tokens "
            f"{max_num_tokens}, {contexts} contexts; view {tuple(k_cache.shape)}, "
            f"{k_cache.numel()} bytes, slots {int(fp8_slot.min())}..{int(fp8_slot.max())}"
        )
        indexer_k, norm_weight, norm_bias = _keys_and_norm(spec, spec["tokens"], g)
        table, _ = _table(spec, cpu)
    except BaseException:
        manager.shutdown()
        raise
    args = [
        indexer_k,
        norm_weight,
        norm_bias,
        pos.to(device="cuda", dtype=torch.int32),
        table,
        k_cache,
        fp8_slot.cuda(),
        scale_slot.cuda(),
        spec["tokens"],
    ]
    return args, manager, pool, head


def _byte_index(fp8_slot, scale_slot, rows: slice):
    """Flat cache-byte indices of the codes and the scales the live rows in `rows` write."""
    f, s = fp8_slot[rows], scale_slot[rows]
    live = f >= 0
    code_at = (f[live].unsqueeze(1) + torch.arange(HEAD_DIM, device="cuda")).flatten()
    scale_at = (s[live].unsqueeze(1) + torch.arange(4, device="cuda")).flatten()
    return code_at, scale_at


def _compare(args, expected, is_neox: bool) -> None:
    """Bytes the op must not write: exact. Scales: exact. Codes: within the
    band `_code_excess` derives, and rarely off at all.

    `args` is the call's argument list; `expected` the reference's cache image.

    The kernel normalises in fp32 with its own summation order and an
    approximate rsqrt; the reference normalises in fp64. The two round a
    normed element to the other bf16 neighbour about once in 10**4 values
    (measured: 2 to 9 codes of 4194304 differ in a 32768-row cell), so at most
    2**-10 of the codes (at least one) may differ. Where one does, the
    difference is bounded per code (`_code_excess`); rope carries a moved
    neighbour into the rotated pair, and a pair that cancels turns it into
    several e4m3 steps at the small result (measured: a code 3 steps off, at
    x13 * cos + x12 * sin = 0.68897 - 0.70631, reproduced by moving x12 one
    bf16 ulp). The scale is a power of two from the row's amax; one ulp on the
    amax only moves it when the amax sits on a power-of-two boundary, which
    these inputs do not hit, so scales are exact.

    The bytes are compared a slice at a time and only the differing ones are
    kept, so a view of a whole pool costs no full-size mask.
    """
    stats = _code_excess(args, expected, is_neox)
    print(f"\n[codes] {stats}")
    assert stats["scales_equal"], "a row scale differs"
    assert stats["outside_slots"] == 0, (
        f"the op wrote outside its slots ({stats['outside_slots']} bytes)"
    )
    assert stats["outside_band"] == 0, (
        f"{stats['outside_band']} codes are outside the band, the worst at "
        f"{stats['worst']:.3g} times it"
    )
    assert stats["differ"] <= stats["allowed"], (
        f"{stats['differ']} of {stats['codes']} codes differ; at most {stats['allowed']} may"
    )


def _ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    """One bf16 unit in the last place at |x| (fp64), the smallest normal's below it."""
    _, exp = torch.frexp(x.abs().clamp(min=2.0**-126))
    return torch.ldexp(torch.ones_like(x), exp - 8)


def _step_e4m3(v: torch.Tensor) -> torch.Tensor:
    """The e4m3 spacing at |v| (fp64, in units of the row scale): 2**-9 below 2**-6."""
    _, exp = torch.frexp(v.abs().clamp(min=2.0**-6))
    return torch.ldexp(torch.ones_like(v), exp - 4)


def _code_excess(args, expected, is_neox: bool) -> dict:
    """How the written cache departs from the reference's, and how far.

    For a code that differs, at row t and column c, the band in the cache's
    values (the e4m3 value times the row scale s) is

        |x_c'| u_c + |x_p'| u_p  +  ulp_bf16(|r| + that)  +  s * step_e4m3(|v| / s)

    for a rotated column (partner p, the table's cos and sin at the row's
    position as the weights `|x'|` stand for), and `u_c + s * step` for an
    unrotated one. `u` is one bf16 ulp of the normed element plus 2**-20 of
    its LayerNorm terms, the most the kernel's fp32 LayerNorm moves it from
    the reference's fp64 one (a few fp32 ulps, which cross at most one bf16
    rounding, plus their absolute size where the terms cancel); the rotation
    carries each by its weight |cos| or |sin|, plus 2**-22 of the products for
    the kernel's fp32 arithmetic; the rotated value `r` and the quantized one
    `v` may then each round to the other neighbour, one bf16 ulp and one e4m3
    step. The normed elements, the rotation and its magnitude are the test's
    fp64 evaluation of the inputs; the reference being checked is the entry's.
    """
    k_cache, fp8_slot, scale_slot, n = args[5], args[6], args[7], int(args[8])
    got = k_cache.contiguous().view(-1)
    want = expected.view(-1)
    code_at, scale_at = _byte_index(fp8_slot, scale_slot, slice(0, n))
    off_at = _differing_bytes(got, want)
    in_codes = torch.isin(off_at, code_at)
    stats = dict(
        codes=code_at.numel(),
        allowed=max(1, code_at.numel() >> 10),
        scales_equal=torch.equal(got[scale_at], want[scale_at]),
        outside_slots=int((~in_codes).sum()),
        differ=int(in_codes.sum()),
        outside_band=0,
        worst=0.0,
    )
    off_at = off_at[in_codes]
    if off_at.numel() == 0:
        return stats
    f = fp8_slot[:n]
    live = torch.nonzero(f >= 0).flatten()
    order = torch.argsort(f[live])
    rows = live[order[torch.searchsorted(f[live][order], off_at, right=True) - 1]]
    cols = off_at - f[rows]
    s = want[scale_slot[rows].unsqueeze(1) + torch.arange(4, device="cuda")]
    s = s.contiguous().view(torch.float32).flatten().double()

    x = args[0][:n].double()
    mean = x.mean(dim=-1, keepdim=True)
    centered = (x - mean) * torch.rsqrt((x - mean).pow(2).mean(dim=-1, keepdim=True) + NORM_EPS)
    terms = (centered * args[1].double()).abs() + args[2].double().abs()
    normed = (centered * args[1].double() + args[2].double()).float().to(torch.bfloat16).double()
    u = _ulp_bf16(normed) + 2.0**-20 * terms

    move = u[rows, cols]
    rotated = cols < ROPE_DIM
    if bool(rotated.any()):
        rc = cols[rotated]
        partner = (rc + ROPE_DIM // 2) % ROPE_DIM if is_neox else rc ^ 1
        col = rc % (ROPE_DIM // 2) if is_neox else rc // 2
        rr = rows[rotated]
        pos = args[3][rr].long()
        cos = args[4][pos, 0, col].double().abs()
        sin = args[4][pos, 1, col].double().abs()
        xc, xp = normed[rr, rc].abs(), normed[rr, partner].abs()
        carried = cos * u[rr, rc] + sin * u[rr, partner] + 2.0**-22 * (xc * cos + xp * sin)
        first = rc < ROPE_DIM // 2 if is_neox else rc % 2 == 0
        sign = torch.where(first, -1.0, 1.0).double()
        r = (
            normed[rr, rc] * args[4][pos, 0, col].double()
            + sign * normed[rr, partner] * args[4][pos, 1, col].double()
        ).abs()
        move = move.clone()
        move[rotated] = carried + _ulp_bf16(r + carried)
    a = got[off_at].view(torch.float8_e4m3fn).double() * s
    b = want[off_at].view(torch.float8_e4m3fn).double() * s
    band = move + s * _step_e4m3(torch.maximum(a.abs(), b.abs()) / s)
    ratio = (a - b).abs() / band
    stats["outside_band"] = int((ratio > 1).sum())
    stats["worst"] = float(ratio.max())
    return stats


def _differing_bytes(got: torch.Tensor, want: torch.Tensor, chunk: int = 1 << 28) -> torch.Tensor:
    """Flat indices at which two equally long byte vectors differ."""
    found = [
        torch.nonzero(got[s : s + chunk] != want[s : s + chunk]).flatten() + s
        for s in range(0, got.numel(), chunk)
    ]
    return torch.cat(found) if found else torch.empty(0, dtype=torch.long, device=got.device)


def _assert_other_layers_untouched(pool, before, layer) -> None:
    assert torch.equal(pool[:, :layer], before[:, :layer]), "a lower layer was written"
    assert torch.equal(pool[:, layer + 1 :], before[:, layer + 1 :]), "a higher layer was written"


def _assert_head_untouched(pool, head: int) -> None:
    """A V2 view runs from its layer's first page to the pool's end, so the other
    layers' pages after it are in the view (`_compare` holds them); the bytes
    before it are not."""
    assert bool((pool.view(-1)[:head] == POISON).all()), "the op wrote below the layer's view"


def _assert_replay_matches_eager(spec, compiled, seed) -> None:
    """Capture the call at this batch, replay it on new data, and require the eager bytes.

    A capture freezes the launcher, the grid and every pointer; what changes
    between replays is the data behind them. So the static inputs take a second
    set of keys, weights, positions, table and slots, the pool goes back to
    poison, and the replay must write bitwise what an eager call on that data
    writes into a pool of its own. `validate=False`, as in every capture.
    """
    static, pool, _ = _build(spec, seed)
    fresh, _, _ = _build(spec, seed + 1)
    eager, eager_pool, _ = _build(spec, seed + 1)
    kwargs = dict(is_neox=spec["is_neox"], compiled=compiled, validate=False)
    op(*static, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        op(*static, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    for i in (0, 1, 2, 3, 4, 6, 7):
        static[i].copy_(fresh[i])
    pool.fill_(POISON)
    graph.replay()
    op(*eager, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(pool, eager_pool), "the replay wrote other bytes than the eager call"


#: The steps a `boot_capture` cell's graph runs, in order: the capture batch, then replays.
BOOT_STEPS = ("warmup", "live", "idle", "drawn")


def _boot_inputs(spec, base, step: str, seed: int) -> dict:
    """One step's data through a `boot_capture` cell's graph: `{arg index: tensor}` for the
    keys, positions and both slot mappings.

    `warmup` is the engine's capture batch: row 0 the longest dummy's new key at position
    max_seq_len - 2 (its KV length max_seq_len - 1), rows 1.. one-key fillers at position 0,
    each in a page of its own. `live` is attention DP's one live row at max_seq_len - 1 ahead of
    the cell's `pad_rows` padding rows, and `idle` an idle rank's one-key dummy (position 0)
    ahead of them; the padding rows are one request repeated, so they carry one key, position
    0 and one slot pair. `drawn` is a batch of live rows at drawn positions. A row's slots are
    `_compute_slot_mappings`'s: its page (a block of its own times the 44 rows) and its
    position's place in the page. The mappings keep `base`'s length (`slot_rows`); the entries
    past the batch are `_build`'s, and the op must not read them.
    """
    rows, top = spec["tokens"], spec["max_seq_len"] - 1
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    keys = (torch.randn(rows, HEAD_DIM, generator=g, device="cuda") * 3.0 + 0.5).to(torch.bfloat16)
    block = torch.randperm(spec["blocks"], generator=cpu)[:rows]
    if step == "drawn":
        pos = torch.randint(0, top + 1, (rows,), generator=cpu)
    else:
        pos = torch.zeros(rows, dtype=torch.long)
        pos[0] = {"warmup": top - 1, "live": top, "idle": 0}[step]
    pad = spec["pad_rows"] if step in ("live", "idle") else 0
    if pad:
        keys[rows - pad :] = keys[rows - pad]
        block[rows - pad :] = block[rows - pad]
    page = block * spec["pool_layers"] * PAGE * TOKEN_BYTES
    fp8_slot, scale_slot = base[6].clone(), base[7].clone()
    fp8_slot[:rows] = (page + (pos % PAGE) * HEAD_DIM).cuda()
    scale_slot[:rows] = (page + PAGE * HEAD_DIM + (pos % PAGE) * 4).cuda()
    return {0: keys, 3: pos.to(device="cuda", dtype=torch.int32), 6: fp8_slot, 7: scale_slot}


def _drive_boot_capture(spec, seed, certifying) -> None:  # noqa: F811
    """One `boot_capture` cell: a long-family decode graph as the dep4 engine captures and
    replays it (`_boot_inputs`).

    The warm-up batch is called once and captured (`validate=False`, as the engine does);
    the graph is then replayed on each later step's data into a pool poisoned again, and each
    replay must write bitwise what an eager call on that data writes into a pool of its own.
    Armed, every step's data, the warm-up batch first, is the eager, validated call gated
    against the reference. The table and the norm weights stay as captured, as the engine's
    do.
    """
    static, pool, layer = _build(dict(spec, pad_rows=0), seed)
    eager, eager_pool, _ = _build(dict(spec, pad_rows=0), seed)
    compiled = _compiled(spec["is_neox"], static[5].is_contiguous())
    kwargs = dict(is_neox=spec["is_neox"], compiled=compiled, validate=False)
    steps = {s: _boot_inputs(spec, static, s, seed + 1 + i) for i, s in enumerate(BOOT_STEPS)}

    def load(args, step):
        for i, t in steps[step].items():
            args[i].copy_(t)

    load(static, "warmup")
    op(*static, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        op(*static, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    for step in BOOT_STEPS[1:]:
        load(static, step)
        pool.fill_(POISON)
        graph.replay()
        load(eager, step)
        eager_pool.fill_(POISON)
        op(*eager, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(pool, eager_pool), f"the {step} replay wrote other bytes than eager"
    certifying(op)
    for step in BOOT_STEPS:
        load(static, step)
        pool.fill_(POISON)
        expected = op.reference(*static, is_neox=spec["is_neox"], compiled=compiled)
        out = op(*static, is_neox=spec["is_neox"], compiled=compiled, validate=True)
        assert out is static[5], "the op writes the caller's cache and returns it"
        _assert_head_untouched(pool, layer * PAGE * TOKEN_BYTES)
        print(f"\n[boot {step}] positions {static[3].tolist()}")
        _compare(static, expected, spec["is_neox"])


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _drive_manager_cell(spec, seed, certifying) -> None:  # noqa: F811
    """One `manager_v2` cell: the eager, validated call on a real manager's view.

    The manager's views must be gone before it releases its pool, and a failed
    check's traceback holds them in its frames; so a failure is carried out of
    the block as its message alone and raised once the manager is shut down.
    The device memory in use with the manager up is printed: its pool is
    allocated outside torch's caching allocator.
    """
    args, manager, pool, head = _build_manager(spec, seed)
    failure = None
    try:
        compiled = _compiled(spec["is_neox"], args[5].is_contiguous())
        certifying(op)
        expected = op.reference(*args, is_neox=spec["is_neox"], compiled=compiled)
        out = op(*args, is_neox=spec["is_neox"], compiled=compiled, validate=True)
        assert out is args[5], "the op writes the caller's cache and returns it"
        _assert_head_untouched(pool, head)
        _compare(args, expected, spec["is_neox"])
        free, total = torch.cuda.mem_get_info()
        print(f"[device] {(total - free) / GIB:.2f} GiB in use of {total / GIB:.2f}")
    except Exception as exc:  # noqa: BLE001 -- re-raised below, without its frames
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        torch.cuda.synchronize()
        args = pool = expected = out = None
        manager.shutdown()
    assert failure is None, failure


def _drive_position_sweep(spec, seed, certifying) -> None:  # noqa: F811
    """One `position_sweep` cell: the cell's keys and slots, run once per chunk of
    `tokens` positions across `[lo, hi)`, so every table row there is read at its
    own position. Each call is the eager, validated one on a pool poisoned
    again, gated against the reference like any cell."""
    args, pool, layer = _build(spec, seed)
    compiled = _compiled(spec["is_neox"], args[5].is_contiguous())
    certifying(op)
    lo, hi = spec["position_sweep"]
    ramp = torch.arange(spec["tokens"], dtype=torch.int32, device="cuda")
    for start in range(lo, hi, spec["tokens"]):
        pool.fill_(POISON)
        args[3] = ramp + start
        expected = op.reference(*args, is_neox=spec["is_neox"], compiled=compiled)
        out = op(*args, is_neox=spec["is_neox"], compiled=compiled, validate=True)
        assert out is args[5], "the op writes the caller's cache and returns it"
        _assert_head_untouched(pool, layer * PAGE * TOKEN_BYTES)
        _compare(args, expected, spec["is_neox"])
    assert start + spec["tokens"] == hi, "the sweep stopped short of its last position"


def _cell_ids(cells) -> list[str]:
    """A test id per cell: the first 44 characters of its `why`, as each source's test
    named it. Where two cells' ids would be one (tp4's and tep4's 1M decode-graph cells
    share a why, or its first 44 characters, and differ in the table: tp4's carries
    `table_dtype`), the cell carrying `table_dtype` gets it appended; every other id
    stays as its source's test had it."""
    short = [c.why[:44] for c in cells]
    count = {i: short.count(i) for i in short}
    ids = [
        f"{i} table_dtype={c.spec['table_dtype']}"
        if count[i] > 1 and "table_dtype" in c.spec
        else i
        for i, c in zip(short, cells)
    ]
    assert len(set(ids)) == len(ids), [i for i in ids if ids.count(i) > 1]
    return ids


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=_cell_ids(op.CELLS),
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    spec = cell.spec
    if spec.get("layout") == "manager_v2":
        _drive_manager_cell(spec, _seed(cell), certifying)
        return
    if spec.get("position_sweep"):
        _drive_position_sweep(spec, _seed(cell), certifying)
        return
    if spec.get("boot_capture"):
        _drive_boot_capture(spec, _seed(cell), certifying)
        return
    args, pool, layer = _build(spec, seed=_seed(cell))
    k_cache = args[5]
    compiled = _compiled(spec["is_neox"], k_cache.is_contiguous())
    if spec.get("graph"):
        # Before arming: the guard reads positions and slots back to the host,
        # which a capture forbids.
        _assert_replay_matches_eager(spec, compiled, _seed(cell))
    certifying(op)
    # The reference reads the bytes it does not write, so it runs first.
    expected = op.reference(*args, is_neox=spec["is_neox"], compiled=compiled)
    v2 = spec.get("layout") == "v2"
    pool_before = None if v2 else pool.clone()
    # The target's first, eager call validates on the device; later ones do not.
    out = op(*args, is_neox=spec["is_neox"], compiled=compiled, validate=True)
    assert out is k_cache, "the op writes the caller's cache and returns it"
    if v2:
        _assert_head_untouched(pool, layer * PAGE * TOKEN_BYTES)
    else:
        _assert_other_layers_untouched(pool, pool_before, layer)
    _compare(args, expected, spec["is_neox"])


def _cell_32768_v2():
    """The non-large 32768-row cell on a rebuilt V2 view at max_seq_len 32768 (slots past 2^31
    and 2^32 bytes)."""
    return next(
        c
        for c in op.CELLS
        if c.spec.get("layout") == "v2"
        and c.spec["tokens"] == 32768
        and c.spec.get("max_seq_len") == 32768
        and not c.spec.get("large")
    )


def _cell_1m_chunk_v2():
    """tp4's 1M-table 32768-row chunk on a rebuilt V2 view: positions 1015808..1048575 over
    the bf16-rounded table (`table_dtype`, which tep4's and dep4's cells do not carry)."""
    return next(
        c
        for c in op.CELLS
        if c.spec.get("layout") == "v2"
        and c.spec.get("max_seq_len") == 1048576
        and c.spec.get("requests") == (32768,)
        and c.spec["table"] == "yarn"
        and c.spec.get("table_dtype") == "bfloat16"
    )


def _cell_1m_v2():
    """The 1M envelope's last chunk on a rebuilt V2 view: positions 1015808..1048575 (tep4's
    and dep4's cell, over the fp32 table: tp4's, with `table_dtype`, is `_cell_1m_chunk_v2`)."""
    return next(
        c
        for c in op.CELLS
        if c.spec.get("layout") == "v2"
        and c.spec.get("max_seq_len") == 1048576
        and c.spec["table"] == "yarn"
        and c.spec.get("cached") == (1015808,)
        and "table_dtype" not in c.spec
    )


@pytest.mark.parametrize(
    "tokens, prefixes",
    [(9216, (1024, 1025)), (32768, (1024, 1025, 9216, 16385, 32767))],
    ids=["9216", "32768"],
)
def test_rows_do_not_depend_on_the_launch_shape(tokens, prefixes) -> None:
    """The launcher and the grid change no byte a row writes.

    One input of a full-step cell -- the FP8 target's 9216-row interleaved cell
    on the strided view, or the NVFP4 target's 32768-row cell on a V2 view (the
    contiguous launchers) -- run at shorter num_tokens: 1024 (2 rows per CTA),
    1025 (8 rows per CTA) and, at 32768, also 9216, 16385 and 32767. The rows a
    shorter run writes are bitwise the full run's, and the rows at and past its
    num_tokens stay unwritten. With the bucket-edge cells and the maximum
    measured against the reference, every row count between them writes what
    they measured.
    """
    if tokens == 9216:
        cell = next(c for c in op.CELLS if c.spec["tokens"] == 9216 and c.spec["table"] == "yarn")
    else:
        cell = _cell_32768_v2()
    full, _, _ = _build(cell.spec, _seed(cell))
    compiled = _compiled(False, full[5].is_contiguous())
    op(*full, is_neox=False, compiled=compiled)
    want = full[5].contiguous().view(-1)
    for n in prefixes:
        args, _, _ = _build(cell.spec, _seed(cell))
        args[8] = n
        op(*args, is_neox=False, compiled=compiled)
        got = args[5].contiguous().view(-1)
        head = torch.cat(_byte_index(args[6], args[7], slice(0, n)))
        tail = torch.cat(_byte_index(args[6], args[7], slice(n, None)))
        assert torch.equal(got[head], want[head]), f"num_tokens {n} wrote other bytes"
        assert bool((got[tail] == POISON).all()), f"num_tokens {n} wrote rows past it"
        args = got = None


@pytest.mark.parametrize("layout", [None, "v2"], ids=["interleaved", "interleaved-v2"])
def test_a_capturable_call_matches_the_validated_one(layout) -> None:
    """`validate=False` (every call inside a CUDA graph) writes the same bytes."""
    spec = dict(
        tokens=16,
        pool_layers=4,
        blocks=8,
        pad_rows=2,
        misalign=False,
        is_neox=False,
        table="yarn",
        layout=layout,
    )
    args, _, _ = _build(spec, seed=41)
    again, _, _ = _build(spec, seed=41)
    compiled = _compiled(False, args[5].is_contiguous())
    op(*args, is_neox=False, compiled=compiled, validate=True)
    op(*again, is_neox=False, compiled=compiled, validate=False)
    assert torch.equal(args[5], again[5])


@pytest.mark.parametrize(
    "wrong",
    [
        "offsets wrapped at 2^32",
        "positions one back",
        "rotation reversed",
        "1M positions one back",
        "1M positions wrapped at 2^15",
    ],
)
def test_the_gate_sees_a_wrapped_offset_or_a_wrong_rotation(wrong) -> None:
    """The 32768-row V2 cell's gate is not blind to what a 32-bit offset, an
    off-by-one position or a reversed rotation would do, nor the 1M chunk cell's
    to a position one back or one read through a 32768-row table.

    The op runs on that cell's input with every slot taken modulo 2^32 (what an
    unsigned 32-bit byte offset would write; the rows past 2^32 bytes land
    inside the view, among other rows), with every position one back (the
    first row stays at 0), or with the table's sines negated; on the 1M chunk
    (positions 1015808..1048575) with every position one back, or modulo 2^15
    (the rows a 32768-row table would hand out cyclically). `_compare`, against
    the reference of the true input, must refuse each; what it measured is
    printed.
    """
    cell = _cell_1m_v2() if wrong.startswith("1M") else _cell_32768_v2()
    args, _, _ = _build(cell.spec, _seed(cell))
    compiled = _compiled(False, args[5].is_contiguous())
    expected = op.reference(*args, is_neox=False, compiled=compiled)
    call = list(args)
    if wrong == "offsets wrapped at 2^32":
        moved = int((args[6] >= 2**32).sum())
        assert moved > 0, "no slot of the cell lies past 2^32 bytes"
        call[6], call[7] = args[6] % 2**32, args[7] % 2**32
    elif wrong.endswith("positions one back"):
        call[3] = (args[3] - 1).clamp(min=0)
    elif wrong == "1M positions wrapped at 2^15":
        assert int(args[3].min()) >= 2**15, "the cell has a position below 2^15"
        call[3] = args[3] % 2**15
    else:
        call[4] = args[4].clone()
        call[4][:, 1].neg_()
    op(*call, is_neox=False, compiled=compiled)
    print(f"\n[{wrong}] {_code_excess(args, expected, is_neox=False)}")
    with pytest.raises(AssertionError, match="scale differs|outside its slots|outside the band"):
        _compare(args, expected, is_neox=False)


@pytest.mark.parametrize("wrong", ["positions one back", "positions wrapped at 2^15"])
def test_the_gate_sees_a_far_position_read_from_the_wrong_row(wrong) -> None:
    """The 1M chunk cell's gate is not blind to a key roped at the wrong row.

    The op runs on that cell's input (positions 1015808..1048575 over the
    bf16-rounded 1048576-row table) with every position one back, or taken
    modulo 2^15 (the row a position wrapped at the 32768-row envelope would
    read). `_compare`, against the reference of the true input, must refuse
    each; what it measured is printed.
    """
    cell = _cell_1m_chunk_v2()
    args, _, _ = _build(cell.spec, _seed(cell))
    compiled = _compiled(False, args[5].is_contiguous())
    expected = op.reference(*args, is_neox=False, compiled=compiled)
    call = list(args)
    if wrong == "positions one back":
        call[3] = args[3] - 1
    else:
        call[3] = args[3] % 2**15
    op(*call, is_neox=False, compiled=compiled)
    print(f"\n[{wrong}] {_code_excess(args, expected, is_neox=False)}")
    with pytest.raises(AssertionError, match="scale differs|outside its slots|outside the band"):
        _compare(args, expected, is_neox=False)


_SMALL = dict(
    tokens=8, pool_layers=4, blocks=8, pad_rows=0, misalign=False, is_neox=False, table="yarn"
)


def _small():
    return _build(_SMALL, seed=43)


def test_guard_refuses_a_rope_mode_the_launchers_were_not_built_for() -> None:
    args, _, _ = _small()
    neox = _compiled(True, False)
    expected = op.reference(*args, is_neox=False, compiled=neox)
    # Unguarded, launchers built for NeoX rotate interleaved keys the NeoX way:
    # well-formed rows whose rope codes are mostly wrong.
    op(*args, is_neox=False, compiled=neox)
    rope_at = (args[6].unsqueeze(1) + torch.arange(ROPE_DIM, device="cuda")).flatten()
    wrong = (args[5].contiguous().view(-1)[rope_at] != expected.view(-1)[rope_at]).float().mean()
    assert float(wrong) > 0.5, f"only {float(wrong):.2f} of the rope codes moved"
    with pytest.raises(AssertionError, match="compiled for is_neox"):
        with validating(op):
            op(*_small()[0], is_neox=False, compiled=neox)


def test_guard_refuses_contiguous_addressing_on_a_strided_view() -> None:
    args, pool, layer = _small()
    assert not args[5].is_contiguous()
    dense = _compiled(False, True)
    # Unguarded, the contiguous launchers address the layer's view as if dense,
    # and the rows of later blocks land in other layers of the same pool.
    before = pool.clone()
    op(*args, is_neox=False, compiled=dense)
    assert not torch.equal(pool[:, :layer], before[:, :layer]), "no other layer was written"
    args, _, _ = _small()
    with pytest.raises(AssertionError, match="strided view"):
        with validating(op):
            op(*args, is_neox=False, compiled=dense)


def test_guard_refuses_a_position_past_the_rope_table() -> None:
    # Unguarded, the kernel reads past the table: not driven here.
    args, _, _ = _small()
    args[3] = args[3].clone()
    args[3][0] = args[4].shape[0]
    with pytest.raises(AssertionError, match="rope table"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(False, False))


def test_guard_refuses_a_misaligned_slot() -> None:
    # Unguarded, the kernel stores a 32-bit word at an unaligned byte: not driven here.
    args, _, _ = _small()
    args[6] = args[6].clone()
    args[6][1] += 2
    with pytest.raises(AssertionError, match="4-byte aligned"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(False, False))


def test_guard_refuses_slots_that_disagree_on_validity() -> None:
    args, _, _ = _small()
    args[7] = args[7].clone()
    args[7][2] = -1
    # Unguarded, a row with a code slot and no scale slot is silently skipped.
    op(*args, is_neox=False, compiled=_compiled(False, False))
    row_codes = args[6][2] + torch.arange(HEAD_DIM, device="cuda")
    assert bool((args[5].contiguous().view(-1)[row_codes] == POISON).all())
    args, _, _ = _small()
    args[7] = args[7].clone()
    args[7][2] = -1
    with pytest.raises(AssertionError, match="disagree"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(False, False))


def _strided_keys(args):
    """Mistral Large 4's keys as the projection leaves them: the first 128 columns of `[T, 160]`."""
    kw = torch.zeros(args[0].shape[0], HEAD_DIM + 32, dtype=torch.bfloat16, device="cuda")
    kw[:, :HEAD_DIM] = args[0]
    args[0] = kw[:, :HEAD_DIM]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_strided_keys, r"Mismatched indexer_k\.strides\[0\]"),
        (lambda a: a.__setitem__(1, a[1].to(torch.bfloat16)), r"argument #1 .*dtype=float32"),
        (lambda a: a.__setitem__(2, a[2].to(torch.bfloat16)), r"argument #2 .*dtype=float32"),
        (lambda a: a.__setitem__(4, a[4].to(torch.bfloat16)), r"argument #4 .*dtype=float32"),
        (lambda a: a.__setitem__(3, a[3].long()), r"argument #3 .*dtype=int32"),
        (lambda a: a.__setitem__(6, a[6].int()), r"argument #6 .*dtype=int64"),
        (lambda a: a.__setitem__(7, a[7].int()), r"argument #7 .*dtype=int64"),
        (lambda a: a.__setitem__(3, torch.cat([a[3], a[3]])), r"Mismatched position_ids\.shape"),
        (lambda a: a.__setitem__(6, a[6][:4]), r"Mismatched slot_mapping_fp8\.shape"),
    ],
    ids=[
        "strided keys",
        "bf16 norm weight",
        "bf16 norm bias",
        "bf16 table",
        "int64 positions",
        "int32 code slots",
        "int32 scale slots",
        "positions longer than keys",
        "slots shorter than keys",
    ],
)
def test_the_launcher_refuses_what_it_was_not_compiled_for(mutate, message) -> None:
    """The note's refusals, pinned: raised before any launch, with `validate` off.

    The pool is unchanged afterwards, so nothing ran.
    """
    args, pool, _ = _small()
    before = pool.clone()
    mutate(args)
    with pytest.raises(ValueError, match=message):
        op(*args, is_neox=False, compiled=_compiled(False, False), validate=False)
    torch.cuda.synchronize()
    assert torch.equal(pool, before), "a refused call wrote the pool"
