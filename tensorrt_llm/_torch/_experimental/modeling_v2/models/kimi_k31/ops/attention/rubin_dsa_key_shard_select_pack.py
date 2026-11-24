# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's key-sharded decode Top-K in one launch of helix16's select / pack / merge.

helix8's chain (``rubin_dsa_key_shard.py``) is the CuTe local Top-``local_k``, the Triton pack
(flags) and merge, then the remap of the kept local tokens to KV pool rows. This entry gets the
same rows from helix16's ``KeyShardSelectPack`` (``kernels/attention/rubin_ull_dsa_index_select_pack.py``
and the exchange kernels it launches) over helix8's workspace:

- the exact local Top-``local_k`` of each row by key (score desc, local index asc on the
  order-preserving bits of the FP32 logit);
- every candidate pushed to every rank as an LL pair tagged with the call's epoch (no flag, no
  system-scope fence);
- the exact global Top-K of the received keys with the ``ties_exact`` certificate, a rank whose
  unsent keys could win pushing its exact local Top-K for the others to re-select from;
- this rank's winners written as the KV pool rows the remap makes of them (``KvRows``), with the
  kept counts and the combine mask ``(bounds == 0) | (count == 0)``.

The selected set is a total order's Top-K (keys are unique), so the rows, counts and mask are
helix8's chain's word for word; the caller still checks that on the device before using it
(helix8's ``dsa_ks_select_pack``).

Workspace: helix8's key-shard buffer sized for the engine merge layout (``KeyShardTopKMerge``: the
receive slots, flags and local logits of helix8's layout, then the LL region,
``select_bytes_needed``), so the chain and this entry read the local logits where the logits kernel
wrote them (``rubin_dsa_key_shard_select_ws``).

Cluster: the local select runs on ``select_cluster(rows, SMs, cap)`` CTAs per row, sized per step
(one CTA per SM for the step's rows: with ``cap`` 4, 4 up to 53 rows, 3 up to 70, 2 up to 106 on 212
SMs), not by the workspace's row capacity; ``precompile`` compiles every size a workspace's row
counts take. One CTA per row is the one-kernel exchange (``KeyShardExchangeKernel``); more is the
cluster select / push grid and a merge grid (``KeyShardClusterExchangeKernel``, split kernels).

KV rows: a kept key whose page is past the block table's columns has no KV pool row; helix8's remap
writes ``-1`` for it and counts it, the select / pack drops it (``kv_keep``). The served rows never
hold one (their lengths are at most the workspace width, the table's columns x 64); ``is_valid``
states it.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_dsa_key_shard import KeyShardWorkspace

#: Tokens per page of the MLA and indexer caches the KV-row arithmetic assumes.
PAGE_TOKENS = 64
#: CTAs per row of the local select at most (helix16's ``local_topk.CLUSTER``).
CLUSTER_MAX = 4


def select_cluster(rows: int, sms: int, cap: int = CLUSTER_MAX) -> int:
    """CTAs per row of the local select for a step of ``rows`` rows: ``min(cap, sms // rows)``, at
    least 1, so the step's clusters fit one CTA per SM. The selected keys and their order do not
    depend on it (``kernels/attention/rubin_ull_dsa_index_local_topk.py``)."""
    return max(1, min(cap, sms // max(rows, 1)))


def select_bytes_needed(rows_max: int, top_k: int, world: int, local_width: int) -> int:
    """Bytes of one rank's key-shard buffer (helix8's ``local_k``) that also holds the LL region of
    the select / pack: at least ``key_shard_bytes_needed``'s."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import KeyShardTopKMerge

    from ...kernels.attention.rubin_dsa_key_shard import key_shard_local_k

    return KeyShardTopKMerge.bytes_needed(rows_max, top_k, world, key_shard_local_k(), local_width)


def key_shard_kv_rows(
    block_table: torch.Tensor, req_idx: torch.Tensor, stride_rows: int, row_offset: int
):
    """The remap's arguments as the select / pack takes them (``KvRows``): KV pool row of
    request-local token ``t`` of row ``r`` = ``block_table[req_idx[r], t // 64] * stride_rows +
    row_offset + t % 64``, ``-1`` past the table's columns or on a negative entry.
    ``block_table`` int32 ``[requests, pages]`` (unit column stride), ``req_idx`` int32 ``[>= rows]``."""
    from ...kernels.attention.rubin_ull_dsa_index import KvRows

    return KvRows(block_table, req_idx, stride_rows, row_offset)


@functools.cache
def _runner_class():
    """``KeyShardSelectPack`` with the cluster of the exchange's local select sized per step."""
    from ...kernels.attention import rubin_ull_dsa_index_local_topk as local_topk
    from ...kernels.attention.rubin_ull_dsa_index_select_pack import KeyShardSelectPack

    class KeyShardSelectPackRunner(KeyShardSelectPack):
        """helix16's runner over helix8's workspace: no threshold publication (helix16's DSA stack
        only), and ``cluster_for(rows)`` CTAs per row on every call (at most ``cluster_max``)."""

        def __init__(self, merge, cluster_max: int = CLUSTER_MAX):
            super().__init__(merge)
            self.publish, self.pub = False, None
            props = torch.cuda.get_device_properties(self.device)
            self.sms = props.multi_processor_count
            splits = (
                1 < local_topk.CLUSTER <= local_topk.CLUSTER_MAX
                and props.major >= 9
                and merge.local_width > self.share_keys
            )
            self.cluster_cap = min(local_topk.CLUSTER, CLUSTER_MAX, cluster_max) if splits else 1

        def cluster_for(self, rows: int) -> int:
            return select_cluster(rows, self.sms, self.cluster_cap)

        def precompile(self, rows_max: int, masked: bool, bounds64: bool) -> Tuple[int, ...]:
            """Compile the KV-row exchange of every cluster size of 1..``rows_max`` rows (eager,
            before any capture runs it); returns the sizes."""
            sizes = tuple(sorted({self.cluster_for(r) for r in range(1, rows_max + 1)}))
            for cluster in sizes:
                if cluster > 1:
                    self._compile_cluster(bounds64, masked, cluster, False, False, True, False)
                else:
                    self._compile_exchange(bounds64, masked, True)
            return sizes

        def __call__(self, row_lens, rows, out=None, counts=None, bounds=None, mask_out=None,
                     exchange=False, tops=False, kv_rows=None):  # fmt: skip
            self.cluster = self.cluster_for(rows)
            super().__call__(row_lens, rows, out, counts, bounds, mask_out, exchange=exchange,
                             tops=tops, kv_rows=kv_rows)  # fmt: skip

    return KeyShardSelectPackRunner


def rubin_dsa_key_shard_select_ws(
    ws: KeyShardWorkspace, cluster_max: int = CLUSTER_MAX
) -> Optional[Any]:
    """The select / pack runner over helix8's key-shard workspace ``ws`` (its buffers at least
    ``select_bytes_needed``), its per-step cluster at most ``cluster_max`` CTAs per row, or None
    where helix16's kernels do not take the workspace's shape (the group size, ``local_k``,
    ``top_k``, the buffers' alignment). Not collective: views of the buffers and a compile (the
    plain select / pack); call it outside CUDA graph capture."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import KeyShardTopKMerge

    merge = KeyShardTopKMerge(
        list(ws.buffers), ws.rank, ws.rows_max, ws.top_k, ws.local_width, local_k=ws.local_k,
        max_spin=ws.max_spin,
    )  # fmt: skip
    assert merge.local_logits.data_ptr() == ws.local_logits.data_ptr(), "one local logits buffer"
    assert merge.local_width == ws.local_width
    runner_cls = _runner_class()
    if not runner_cls.supported(merge):
        return None
    runner = runner_cls(merge, cluster_max)
    return runner if runner.supports_exchange() else None


def _reference_rows(logits_by_rank, row_lens_by_rank, rank: int, top_k: int, kv_rows):
    from .rubin_dsa_key_shard import _exact_shares

    shares = _exact_shares(logits_by_rank, row_lens_by_rank, top_k)[rank]
    rows = len(shares)
    device = logits_by_rank[0].device
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=device)
    counts = torch.zeros(rows, dtype=torch.int32, device=device)
    for r, mine in enumerate(shares):
        mine = mine.long()
        if kv_rows is not None:
            table = kv_rows.block_table[int(kv_rows.req_idx[r])].long().to(mine.device)
            page = mine // PAGE_TOKENS
            inside = page < table.shape[0]
            entry = torch.where(inside, table[page.clamp(max=table.shape[0] - 1)], -1)
            mine = torch.where(
                inside & (entry >= 0),
                entry * kv_rows.stride_rows + kv_rows.row_offset + mine % PAGE_TOKENS,
                -1,
            )
        out[r, : mine.numel()] = mine.to(torch.int32)
        counts[r] = mine.numel()
    return out, counts


class RubinDsaKeyShardSelectPack(OpWrapper):
    """This rank's share of each row's exact global Top-K from ``runner.merge.local_logits[:rows]``
    (this rank's local scores of the row, ``row_lens[:rows]`` int32 valid columns each), in one
    launch of helix16's select / pack / LL exchange / merge (module doc).

    Writes ``out`` int32 ``[rows, top_k]``: the kept local tokens in ascending local position as KV
    pool rows (``kv_rows``; local token indices without it), then -1; ``counts`` int32 ``[>= rows]``
    their number; with ``bounds`` (int32 or int64 ``[>= rows]``, each row's rank-local KV bound)
    ``mask_out`` bool ``[>= rows]`` the combine mask ``(bounds == 0) | (count == 0)``. Returns None.

    Collective over the workspace's group: every rank makes the same call with the same ``rows``.
    """

    ARCHS = frozenset({Arch.SM_107})

    # helix8's group emulated on one GPU (every rank's workspace on it, every rank on its own
    # stream): the select and merge CTAs of every rank and row must be resident at once. ``sms``
    # stands in for the SM count of the cluster rule: one entry per run, so 4 rows on 212 / 12 / 8 /
    # 4 SMs take clusters of 4 / 3 / 2 / 1. Every cell is also compared with helix8's chain (local
    # Top-K, pack, merge, remap) on the same inputs.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8's served shape: 8 ranks, 16.5k local keys (132k context), every cluster size",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(16000, 16512),
                top_k=2048,
                kind="randn",
                sms=(212, 12, 8, 4),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
        Cell(
            why="FP8 indexer scores (sums of w * relu(q . k) of E4M3 values): exact ties",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(16000, 16512),
                top_k=2048,
                kind="fp8",
                sms=(212, 12),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
        Cell(
            why="scores rounded to 1/4: very many ties at the 512 and 2048 boundaries",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(16000, 16512),
                top_k=2048,
                kind="coarse",
                sms=(212, 12, 4),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
        Cell(
            why="one rank holds most of every row's Top-K: the short rank's rescue",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(16000, 16512),
                top_k=2048,
                kind="skew",
                sms=(212, 4),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
        Cell(
            why="long rows (33k local keys, 264k context): the shares exceed a chunk",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(32000, 33024),
                top_k=2048,
                kind="randn",
                sms=(212, 12),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
        Cell(
            why="short rows and empty rows (fewer keys than the Top-K), clusters 4 and 1",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(0, 300),
                top_k=2048,
                kind="randn",
                sms=(212, 4),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
        Cell(
            why="local token indices, no bounds: counts only",
            spec=dict(
                world=8,
                rows=4,
                row_lens=(16000, 16512),
                top_k=2048,
                kind="randn",
                sms=(212,),
                kv_rows=False,
                bounds=False,
            ),
        ),  # fmt: skip
        Cell(
            why="4 ranks, 8 rows, all scores equal",
            spec=dict(
                world=4,
                rows=8,
                row_lens=(4000, 8256),
                top_k=2048,
                kind="zeros",
                sms=(212,),
                kv_rows=True,
                bounds=True,
            ),
        ),  # fmt: skip
    )
    note = """
    Exact: helix8's chain gives the same words on every cell (the test compares both on the same
    inputs, over two calls per workspace so both LL parities run). The LL polls have a deadline of
    ``runner.merge.max_spin`` microseconds; a timed-out poll counts in ``runner.merge.state[3]``
    and traps (``kernels/common/rubin_ull_spin_trap.py``, sites 171-174). Rows with a short rank
    count in ``runner.merge.state[2]``. Not certified here: more than 4 rows at 8 ranks on one GPU
    (the residency of the emulated group); world 8 at 24 / 54 / 66 rows is measured by the bench.
    """

    def __call__(
        self,
        runner,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor,
        counts: torch.Tensor,
        bounds: Optional[torch.Tensor],
        mask_out: Optional[torch.Tensor],
        kv_rows,
    ) -> None:
        runner(row_lens, rows, out, counts, bounds, mask_out, exchange=True, tops=False,
               kv_rows=kv_rows)  # fmt: skip

    def reference(
        self,
        runner,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor,
        counts: torch.Tensor,
        bounds: Optional[torch.Tensor],
        mask_out: Optional[torch.Tensor],
        kv_rows,
        *,
        group: Sequence[Tuple[torch.Tensor, torch.Tensor]] = (),
    ):
        """A collective: ``group`` holds every rank's ``(local_logits [rows, >= len], row_lens)`` in
        rank order. Returns this rank's ``(out, counts, mask)`` (``mask`` None without
        ``mask_out``); ``out`` and ``counts`` are not read."""
        rank, top_k = runner.merge.rank, runner.merge.top_k
        logits = [lg[:rows].float().cpu() for lg, _ in group]
        lens = [ln[:rows].cpu() for _, ln in group]
        want, sizes = _reference_rows(logits, lens, rank, top_k, kv_rows)
        mask = None
        if mask_out is not None:
            mask = (bounds[:rows].cpu() == 0) | (sizes == 0)
        return want, sizes, mask

    def is_valid(
        self,
        runner,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor,
        counts: torch.Tensor,
        bounds: Optional[torch.Tensor],
        mask_out: Optional[torch.Tensor],
        kv_rows,
    ) -> None:
        merge = runner.merge
        # Rows past the workspace land in another row's LL slots.
        assert 0 < rows <= merge.rows_max, "more rows than the workspace holds"
        assert int(row_lens[:rows].max()) <= merge.local_width, "a row runs past the local logits"
        assert (mask_out is None) == (bounds is None), "mask_out goes with bounds"
        assert kv_rows is None or kv_rows.req_idx.shape[0] >= rows, (
            "fewer request indices than rows"
        )
        # A kept key past the table's pages: the remap writes -1 and counts it, this drops it.
        pages = -(-int(row_lens[:rows].max()) // PAGE_TOKENS)
        assert kv_rows is None or pages <= kv_rows.block_table.shape[1], (
            "a row's keys run past the block table's pages"
        )


rubin_dsa_key_shard_select_pack = RubinDsaKeyShardSelectPack()


# --- the first-use device check of the caller (helix8's dsa_ks_select_pack) -----------------------


def key_shard_agree(ws: KeyShardWorkspace, ok: bool) -> bool:
    """Whether ``ok`` holds on every rank of ``ws``'s group (host collective over the workspace's
    communicator; a workspace without one, e.g. a group emulated on one GPU, answers ``ok``)."""
    if ws.owner is None:
        return bool(ok)
    from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

    return _mnnvl_workspace_all_succeeded(ws.owner[1], bool(ok))


def _coarse(x: torch.Tensor) -> torch.Tensor:
    """FP32 values with 3 mantissa bits left: runs of exactly equal scores."""
    return (x.contiguous().view(torch.int32) & ~0xFFFFF).view(torch.float32)


def key_shard_check_cases(
    real: torch.Tensor, row_lens: torch.Tensor, rank: int, world: int, width: int
) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    """The check's inputs, one at a time: ``(name, local logits [rows, width], row lengths)``.

    - ``step``: the step's own logits and row lengths;
    - ``ties``: the same with 3 mantissa bits left (ties at the 512 and 2048 boundaries);
    - ``fallback``: ``ties`` with every row ``r`` of rank ``r % world`` raised by 2^24, so that rank
      holds the row's Top-K (the merges' exact fallback / rescue);
    - ``long``: every row ``width`` keys long, normal scores rounded to 1/4 (a fixed seed per rank):
      the longest rows the workspace takes, with many ties.
    """
    rows = real.shape[0]
    lens = row_lens[:rows]
    yield "step", real, lens
    ties = _coarse(real)
    yield "ties", ties, lens
    favored = (torch.arange(rows, device=real.device) % world) == rank
    yield "fallback", torch.where(favored[:, None], ties + 2.0**24, ties), lens
    del ties
    gen = torch.Generator(device=real.device).manual_seed(7919 + rank)
    long_logits = torch.round(torch.randn(real.shape, generator=gen, device=real.device) * 4) / 4
    yield "long", long_logits, torch.full_like(lens, width)


def _difference(name: str, want: Sequence, got: Sequence) -> str:
    for label, a, b in zip(("KV rows", "counts", "mask"), want, got):
        if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
            return f"{label} differ on the {name} logits"
    return ""


def key_shard_select_check(
    ws: KeyShardWorkspace,
    rows: int,
    width: int,
    row_lens: torch.Tensor,
    today: Callable[[torch.Tensor], Sequence[Optional[torch.Tensor]]],
    select: Callable[[torch.Tensor], Sequence[Optional[torch.Tensor]]],
    counters: Sequence[torch.Tensor] = (),
    agree: Callable[[KeyShardWorkspace, bool], bool] = key_shard_agree,
) -> str:
    """The first-use device check of the select / pack against today's chain at ``rows`` rows.

    ``ws.local_logits[:rows, :width]`` holds the step's logits. For each of
    ``key_shard_check_cases`` (the step's logits first), written into the local logits: ``today(lens)``
    and ``select(lens)`` (each a collective of the group, returning ``(KV rows [rows, top_k],
    counts [rows], mask [rows] or None)``) compared word for word; every rank finishes a case before
    any rank writes the next case's logits (a peer's fallback reads them). The step's logits are
    restored at the end. ``counters``: state words whose index 3 counts timed-out polls.
    ``agree``: the group's verdict (``key_shard_agree``), also its barrier between cases.

    Collective, eager (outside capture). Returns "" when every case matched on every rank, else
    this rank's first difference, or "a peer's check failed"."""
    logits = ws.local_logits[:rows, :width]
    cuda = logits.is_cuda
    if cuda:
        torch.cuda.synchronize()
    real = logits.clone()
    timeouts0 = [int(c[3]) for c in counters]
    why = ""
    for name, case_logits, lens in key_shard_check_cases(real, row_lens, ws.rank, ws.world, width):
        logits.copy_(case_logits)
        want = today(lens)
        got = select(lens)
        if cuda:
            torch.cuda.synchronize()
        if not why:
            why = _difference(name, [w[:rows] if w is not None else None for w in want],
                              [g[:rows] if g is not None else None for g in got])  # fmt: skip
        agree(ws, True)
    logits.copy_(real)
    if cuda:
        torch.cuda.synchronize()
    timeouts = sum(int(c[3]) - t for c, t in zip(counters, timeouts0))
    if not why and timeouts:
        why = f"{timeouts} exchange polls timed out"
    if not agree(ws, not why) and not why:
        why = "a peer's check failed"
    return why


__all__: List[str] = [
    "CLUSTER_MAX",
    "PAGE_TOKENS",
    "RubinDsaKeyShardSelectPack",
    "key_shard_agree",
    "key_shard_check_cases",
    "key_shard_kv_rows",
    "key_shard_select_check",
    "rubin_dsa_key_shard_select_pack",
    "rubin_dsa_key_shard_select_ws",
    "select_bytes_needed",
    "select_cluster",
]
