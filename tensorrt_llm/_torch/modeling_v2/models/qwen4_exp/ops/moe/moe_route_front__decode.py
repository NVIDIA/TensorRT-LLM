# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode MoE block's routing, pre-routed tables and NVFP4 activation in one launch.

After the router GEMM, the trtllm-gen NVFP4 runner normally quantizes its input (`fp4_quantize`)
and routes it itself (a histogram and a scan, or a cluster kernel). The route front
(`kernels/blackwell/moe_route_front`, thread-block clusters) does all three in one launch over a
step's gathered rows: the top-10 of the 512 router logits with the Renormalize weights, the
runner's pre-routed tables for an expert window (expanded -> permuted index, permuted index ->
token, CTA -> local expert, CTA -> row limit, CTA count, padded rows), and the NVFP4 codes with
linear e4m3 block scales at the layer's global scale. The runner then enters pre-routed
(`run_moe_prerouted`) at the tactic whose tile the tables were built for.

Up to `LEAN_MAX_ROWS` rows the lean form runs (one cheaper count exchange; it releases its
dependent, the runner's FC1, at its start, whose own wait still holds it until this grid
completes); past them clusters of 16 and the default release, or with `early=True` the release
right after the launch's wait, which is then its first statement (the same bits; only FC1's
launch moves).

`fast2=True` / `1` (past 512 rows, where a warp routes two tokens) routes each of a warp's two
tokens on the kernel's one-token fast path instead of both on the full per-lane sort: the same
bits. `fast2=2` also loads the second token's logits before the first token is routed.

`route_front_only(..., ids_in=True)` (past `LEAN_MAX_ROWS`) takes rows their owners already
routed (`IDS_ROW_SLOTS` int16 a row: the experts, then their weight bits, which the attention-DP
packed gather forms with the kernel's own `route_one_token`) and builds the same tables.

Bitwise: the experts and weights are `renorm_moe_routing_op`'s, the codes and scales
`fp4_quantize`'s; the tables match the runner's own routing up to the row order inside an
expert's tile, which no output depends on.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)

#: The routing geometry the kernel is written for: 512 experts, top-10.
NUM_EXPERTS = 512
TOP_K = 10
#: Rows a launch takes: clusters of 8 to 256 rows (the lean form), clusters of 16 to 1024.
MAX_ROWS = 1024
LEAN_MAX_ROWS = 256
_SF_VEC = 16

_COMPILED: dict = {}


def max_num_tiles(rows: int, top_k: int, num_local: int, tile: int) -> int:
    """The runner's CTA bound of a pre-routed problem (`getMaxNumCgasInBatchDim`)."""
    remaining = rows * top_k
    filled = min(num_local, remaining)
    remaining -= filled
    return filled + (remaining // tile if remaining > 0 else 0)


def table_sizes(rows: int, top_k: int, num_local: int, tile: int) -> list[int]:
    """int32 sizes of the six pre-routed tables in one flat buffer, in `run_moe_prerouted`'s
    order: expanded -> permuted index, permuted index -> token (one tile of slack), CTA -> local
    expert, CTA -> row limit, CTA count, padded row count."""
    ctas = max_num_tiles(rows, top_k, num_local, tile)
    return [rows * top_k, (ctas + 1) * tile, ctas, ctas, 1, 1]


def table_views(flat: torch.Tensor, rows: int, top_k: int, num_local: int, tile: int) -> list:
    """The six tables of `flat` (`table_sizes` order) as int32 views."""
    sizes = table_sizes(rows, top_k, num_local, tile)
    return list(torch.split(flat[: sum(sizes)], sizes))


def _threads(rows: int, hidden: int, num_local: int) -> int:
    """Threads a CTA of the lean form: the fewest of 256 / 512 / 1024 that keep one token a
    routing warp, one thread a local expert and one quantization pass at 7 clusters."""
    tokens_per_cta = -(-rows // 8)
    for threads in (256, 512, 1024):
        if (
            32 * tokens_per_cta <= threads
            and num_local <= threads
            and rows * (hidden // 16) <= 7 * 8 * threads
        ):
            return threads
    return 1024


#: Rows from which a warp routes two tokens (clusters of 16 CTAs of 32 warps): `fast2` changes
#: the launch only past them.
FAST2_MIN_ROWS = 513
#: int16 slots of a routed row (`ids_in`): the 10 experts, their 10 weights' bits, 4 zeros.
IDS_ROW_SLOTS = 24


def _key(
    rows, hidden, row_stride, num_local, tile, fast2, route_only=False, ids_in=False, early=False
) -> tuple:
    form = int(fast2) if rows >= FAST2_MIN_ROWS else 0
    if form not in (0, 1, 2):
        raise ValueError(f"fast2={fast2!r}: expected 0 / False, 1 / True or 2")
    pdl = bool(get_env_enable_pdl())
    # `early` moves only the release past the lean rows, and only with PDL.
    early = bool(early) and pdl and rows > LEAN_MAX_ROWS
    return (
        # form 1 keys as `True`, as before form 2 existed
        (rows, hidden, row_stride, num_local, tile, pdl, form if form != 1 else True)
        + (("route",) if route_only else ())
        + (("ids",) if ids_in else ())
        + (("early",) if early else ())
    )


def compile_front(
    rows: int,
    hidden: int,
    row_stride: int,
    num_local: int,
    tile: int,
    fast2: bool | int = False,
    route_only: bool = False,
    ids_in: bool = False,
    early: bool = False,
):
    """Compile (once) the launch of `rows` gathered rows of `hidden` (row stride `row_stride`)
    for a window of `num_local` experts at tile `tile` (`fast2`: two tokens a warp on the
    one-token fast path, `2` with the second token's logits loaded first; `route_only`: the
    routing cluster alone, `route_front_only`; `ids_in`: with it, over routed rows past
    `LEAN_MAX_ROWS`; `early`: past them, the release right after the wait); False where the
    kernel does not take the shape. Outside CUDA-graph capture only; `__call__` then finds it."""
    key = _key(rows, hidden, row_stride, num_local, tile, fast2, route_only, ids_in, early)
    if key in _COMPILED:
        return _COMPILED[key] is not None
    assert not torch.cuda.is_current_stream_capturing(), "compile outside graph capture"
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.kernels.blackwell.moe_route_front import (
        ROUTE_CLUSTER,
        ROUTE_CLUSTER_WIDE,
        ROUTE_THREADS,
        MoeRouteFrontKernel,
    )

    lean = rows <= LEAN_MAX_ROWS
    config = dict(
        num_rows=rows,
        k_extent=hidden,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local=num_local,
        tile=tile,
        a_stride=row_stride,
        cluster=ROUTE_CLUSTER if lean else ROUTE_CLUSTER_WIDE,
        quant_clusters=7 if lean else 3,
        threads=_threads(rows, hidden, num_local) if lean else ROUTE_THREADS,
        lean=lean,
    )
    if (
        not 1 <= rows <= MAX_ROWS
        or (ids_in and not route_only)
        or not MoeRouteFrontKernel.supports(ids_in=ids_in, **config)
    ):
        _COMPILED[key] = None
        return False
    kernel = MoeRouteFrontKernel(
        use_pdl=key[5],
        fast2=int(key[6]),
        route_only=route_only,
        ids_in=ids_in,
        early="early" in key,
        **config,
    )
    z = cutlass.Int64(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _COMPILED[key] = cute.compile(kernel, z, z, z, z, z, z, z, z, cutlass.Int32(0), stream)
    return True


def _cell(
    rows: int,
    offset: int,
    tile: int,
    why: str,
    num_local: int = 128,
    fast2: bool | int = False,
    early: bool = False,
) -> Cell:
    spec = dict(rows=rows, hidden=2560, local_offset=offset, num_local=num_local, tile=tile)
    if fast2:
        spec["fast2"] = True if fast2 is True else int(fast2)
    if early:
        spec["early"] = True
    return Cell(why=why, spec=spec)


class MoeRouteFront(OpWrapper):
    """`xq, xsf, ids, wts = moe_route_front(logits, x, global_scale, tables, tile, local_offset,
    num_local)`: over `x` `[T, H]` bf16 (`T` = the gathered rows, `compile_front`d) and its
    router logits `[T, 512]` bf16 (contiguous), the NVFP4 codes `[T, H / 2]` uint8 with linear
    block scales `[T, H / 16]` uint8 at `global_scale`, the top-10 experts `[T, 10]` int32 and
    their Renormalize weights `[T, 10]` bf16, and the pre-routed tables of the window
    `[local_offset, local_offset + num_local)` written into `tables` (int32, at least
    `table_sizes` long; `table_views` reads them) for tile `tile`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        _cell(48, 0, 8, "dep4 MTP-3: 3 requests a rank gathered (48 rows), window 0, tile 8"),
        _cell(96, 128, 16, "dep4 MTP-3: 6 requests a rank (96 rows), window 1, tile 16"),
        _cell(256, 384, 32, "dep4 MTP-3: 16 a rank (256 rows, the lean form's last), window 3"),
        _cell(512, 256, 32, "dep4 MTP-3: 32 a rank (512 rows, clusters of 16), window 2"),
        _cell(1024, 0, 64, "dep4 MTP-3: 64 a rank (1024 rows, the launch's last), tile 64"),
        _cell(4, 384, 8, "tep4 MTP-3: 1 request (4 rows, the rank's own), window 3, tile 8"),
        _cell(16, 128, 8, "tep4 MTP-3: 4 requests (16 rows, the rank's own), window 1, tile 8"),
        _cell(96, 448, 16, "dep8 MTP-3: 3 requests a rank (96 rows), window 7 of 64", 64),
        _cell(448, 64, 32, "dep8 MTP-3: 14 a rank (448 rows, clusters of 16), window 1", 64),
        _cell(896, 320, 64, "dep8 MTP-3: 28 a rank (896 rows), window 5, tile 64", 64),
        _cell(544, 128, 32, "dep8 fast2: 17 a rank (544 rows), window 2, tile 32", 64, True),
        _cell(768, 448, 32, "dep8 fast2: 24 a rank (768 rows), window 7, tile 32", 64, True),
        _cell(1024, 0, 64, "fast2: 1024 rows, 64 tokens a CTA (two mask words)", 64, True),
        _cell(640, 256, 32, "dep4 fast2=2: 40 a rank (640 rows), window 2, early", 128, 2, True),
        _cell(
            832,
            384,
            64,
            "dep4 fast2=2: 52 a rank (832 rows, the measured dep4 cap), window 3, early",
            128,
            2,
            True,
        ),
        _cell(1024, 0, 64, "fast2=2: 1024 rows, 64 tokens a CTA, window 0, early", 128, 2, True),
        _cell(768, 448, 32, "dep8 fast2: 24 a rank (768 rows), window 7, early", 64, True, True),
        _cell(
            448,
            64,
            32,
            "dep8: 14 a rank (448 rows, one token a warp), window 1, early",
            64,
            False,
            True,
        ),
    )

    note = """
    Source: `kernels/blackwell/moe_route_front` (CuTe DSL clusters; lean to 256 rows), compiled
    per shape by `compile_front` outside capture. Certified for 128-expert windows of the
    512-expert / top-10 layer at hidden 2560 on GB300 (dep4's gathered rows, tep4's own 4-16
    rows), tiles 8 to 64; written for 64-expert windows (dep8's gathered rows, the last three
    cells). Not certified: other routing geometries.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        tables: torch.Tensor,
        tile: int,
        local_offset: int,
        num_local: int,
        fast2: bool | int = False,
        early: bool = False,
    ):
        import cuda.bindings.driver as cuda
        import cutlass

        rows, hidden = x.shape
        key = _key(rows, hidden, x.stride(0), num_local, tile, fast2, early=early)
        fn = _COMPILED.get(key)
        assert fn is not None, f"no route front compiled for {key}: `compile_front` first"
        dev = x.device
        xq = torch.empty([rows, hidden // 2], dtype=torch.uint8, device=dev)
        xsf = torch.empty([rows, hidden // _SF_VEC], dtype=torch.uint8, device=dev)
        ids = torch.empty([rows, TOP_K], dtype=torch.int32, device=dev)
        wts = torch.empty([rows, TOP_K], dtype=torch.bfloat16, device=dev)
        p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
        fn(
            p64(logits),
            p64(x),
            p64(global_scale),
            p64(xq),
            p64(xsf),
            p64(ids),
            p64(wts),
            p64(tables),
            cutlass.Int32(local_offset),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return xq, xsf, ids, wts

    def reference(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        tables: torch.Tensor,
        tile: int,
        local_offset: int,
        num_local: int,
        fast2: bool | int = False,
        early: bool = False,
    ):
        """The composed path it replaces, without the tables (they are checked by their meaning,
        `check_tables`): `fp4_quantize` with linear scales, `renorm_moe_routing_op`. `fast2` and
        `early` move no value."""
        xq, xsf = fp4_quantize(x, global_scale, _SF_VEC, False, False)
        ids, wts = renorm_moe_routing_op(logits, TOP_K, torch.bfloat16)
        return xq, xsf.reshape(x.shape[0], -1), ids, wts

    def is_valid(
        self,
        logits,
        x,
        global_scale,
        tables,
        tile,
        local_offset,
        num_local,
        fast2=False,
        early=False,
    ) -> None:
        assert int(fast2) in (0, 1, 2), "fast2: 0 / False, 1 / True or 2"
        rows, hidden = x.shape
        assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.data_ptr() % 16 == 0
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous()
        assert logits.shape == (rows, NUM_EXPERTS) and 1 <= rows <= MAX_ROWS
        assert 0 <= local_offset and local_offset + num_local <= NUM_EXPERTS
        assert tables.dtype == torch.int32 and tables.is_contiguous()
        assert tables.numel() >= sum(table_sizes(rows, TOP_K, num_local, tile))


def route_front_only(
    logits: torch.Tensor,
    tables: torch.Tensor,
    tile: int,
    local_offset: int,
    num_local: int,
    hidden: int,
    fast2: bool | int = False,
    ids_in: bool = False,
    early: bool = False,
):
    """`moe_route_front`'s routing alone, for rows whose NVFP4 form the caller has already (the
    attention-DP packed gather): `ids, wts` and the tables written into `tables`, bitwise the full
    launch's over the same logits (the same routing cluster; its quantization clusters are not
    launched). With `ids_in`, `logits` is the rows' routing instead, `[rows, IDS_ROW_SLOTS]` int16
    (contiguous) formed from their logits by the kernel's `route_one_token`: the same `ids, wts`
    and tables. `early`: the launch compiled with the release right after its wait.
    `compile_front(rows, hidden, hidden, num_local, tile, fast2, True, ids_in, early)` first.
    Not an entry of its own: `moe_route_front`'s test holds it to that launch."""
    import cuda.bindings.driver as cuda
    import cutlass

    rows = logits.shape[0]
    if ids_in:
        # the kernel reads 16-bit slots at a fixed row pitch: a logits tensor here reads garbage
        assert logits.dtype is torch.int16 and logits.is_contiguous(), logits.dtype
        assert tuple(logits.shape) == (rows, IDS_ROW_SLOTS), tuple(logits.shape)
    fn = _COMPILED.get(_key(rows, hidden, hidden, num_local, tile, fast2, True, ids_in, early))
    assert fn is not None, f"no routing-only front compiled for {rows} rows: `compile_front`"
    dev = logits.device
    ids = torch.empty([rows, TOP_K], dtype=torch.int32, device=dev)
    wts = torch.empty([rows, TOP_K], dtype=torch.bfloat16, device=dev)
    z = cutlass.Int64(0)
    p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
    fn(
        p64(logits),
        z,
        z,
        z,
        z,
        p64(ids),
        p64(wts),
        p64(tables),
        cutlass.Int32(local_offset),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return ids, wts


def routed_rows(ids: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """The `ids_in` form of rows routed to experts `ids` `[R, 10]` int32 with weights `wts`
    `[R, 10]` bf16: `[R, IDS_ROW_SLOTS]` int16."""
    out = torch.zeros([ids.shape[0], IDS_ROW_SLOTS], dtype=torch.int16, device=ids.device)
    out[:, :TOP_K] = ids.to(torch.int16)
    out[:, TOP_K : 2 * TOP_K] = wts.contiguous().view(torch.int16)
    return out


def check_tables(views, ids: torch.Tensor, local_offset: int, num_local: int, tile: int) -> str:
    """What is wrong with the pre-routed tables `views` (`table_views`) of the experts `ids`,
    or "": every (row, slot) of the window has a permuted index whose token is the row and whose
    CTA serves its expert, every other slot -1; each expert's slots fill whole tiles from its
    first CTA; the CTA count and the padded rows match."""
    e2p, p2t, cta_expert, cta_limit, n_ctas, n_padded = (v.cpu() for v in views)
    ids = ids.cpu().to(torch.int64)
    rows = ids.shape[0]
    local = ids - local_offset
    inside = (local >= 0) & (local < num_local)
    counts = torch.bincount(local[inside], minlength=num_local)
    tiles = (counts + tile - 1) // tile
    if int(n_ctas[0]) != int(tiles.sum()) or int(n_padded[0]) != int(tiles.sum()) * tile:
        return f"CTA count {int(n_ctas[0])} / padded {int(n_padded[0])}, want {int(tiles.sum())}"
    e2p = e2p.view(rows, -1).to(torch.int64)
    if bool((e2p[~inside] != -1).any()):
        return "a slot outside the window has a permuted index"
    perm = e2p[inside]
    if bool((perm < 0).any()) or perm.unique().numel() != perm.numel():
        return "the window's slots do not have distinct permuted indices"
    if bool((perm >= int(n_padded[0])).any()):
        return "a permuted index sits past the padded rows (a CTA the runner does not launch)"
    tokens = torch.arange(rows).unsqueeze(1).expand_as(e2p)[inside]
    if not torch.equal(p2t[perm].to(torch.int64), tokens):
        return "a permuted index names another token"
    if not torch.equal(cta_expert[perm // tile].to(torch.int64), local[inside]):
        return "a permuted index sits in another expert's CTA"
    if bool((perm >= cta_limit[perm // tile].to(torch.int64)).any()):
        return "a permuted index sits past its CTA's row limit"
    return ""


moe_route_front = MoeRouteFront()
