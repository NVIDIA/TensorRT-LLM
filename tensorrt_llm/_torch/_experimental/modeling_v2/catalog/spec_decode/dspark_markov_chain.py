# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft head's vanilla-Markov chain over vocab-sharded draft logits, in one launch per pass.

For every request b and block position k (greedy): ``bias = bf16(w2_shard . w1[prev])``, ``value = bf16(base[b, k]
+ bias)``, ``token[b, k]`` = the first maximum of ``value`` over the whole vocabulary (every rank's shard), ``prev =
token[b, k]``; ``prev`` for k = 0 is the anchor token. The in-tree chain gathers the full-vocabulary logits and runs
K launches of GEMV + add + argmax on every rank; this is one launch per rank, exchanging one (value, index) entry per
CTA and position over a multicast NVLink buffer.

Shapes: the kernel splits a shard over an even number of co-resident CTAs, each owning a multiple of 64 rows (at most
``SMEM_ROW_BUDGET``). A shard that does not split so is padded: ``padded_rows`` gives the padded row count; the
caller allocates the draft logits with that last dimension (``pad_logits_buffer``, tail filled with -inf, never a
maximum) and pads ``w2_shard`` once at load (``pad_w2_shard``, zero rows).

Compiled on the first call per shape, which must run outside CUDA-graph capture (``precompile``). The exchange
workspace is collective on its first use: every rank of the TP group calls ``workspace_args`` at the same point.

Variants: batch-1 calls can take ``dspark_markov_chain_kernel_b1`` (``enable_batch1_variant``) and, on top of it, its
LL flag-in-payload exchange ``dspark_markov_chain_kernel_b1_ll`` (``enable_ll_exchange``); all are bitwise the same
tokens. Every rank must run the same variant for a call (the switches are process-wide; set them the same way on
every rank).
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._op import Arch, Cell, OpWrapper

MAX_BLOCK = 8  # capacity: fixed: the workspace's positions per call
MAX_GRID = 152  # capacity: fixed: the workspace's CTAs per rank (all co-resident)
CERTIFIED_BATCH = (
    8  # the largest batch the cells drive (the kernel takes up to the kernel module's MAX_BATCH)
)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_workspaces: Dict[object, dict] = {}
_modules: Dict[str, object] = {}
_dummy: Dict[torch.device, torch.Tensor] = {}
_const_inputs: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


_batch1_variant = [
    False
]  # S4B8: the batch-1 calls take dspark_markov_chain_kernel_b1 (see enable_batch1_variant)


_ll_exchange = [
    False
]  # MARKOV_LL: the batch-1 calls take dspark_markov_chain_kernel_b1_ll (see enable_ll_exchange)


_cached_inputs = [
    False
]  # TAILBK_SPEC: the kernel's read-only accepted inputs come from a per-shape cache (see enable_cached_inputs)
_cached_taken = [False]  # a call took its accepted inputs from the cache (see cached_inputs_taken)


def _kernel_module(batch1: bool = False, ll: bool = False):
    """The CuTe DSL kernel beside this file (or, for ``batch1``, its batch-1 variant; with ``ll`` too, the batch-1
    variant with the LL exchange), imported on first use (it pulls in cutlass)."""
    name = ("kernel_b1_ll" if ll else "kernel_b1") if batch1 else "kernel"
    mod = _modules.get(name)
    if mod is None:
        mod = _modules[name] = importlib.import_module(f".dspark_markov_chain_{name}", __package__)
    return mod


_fp32_sum = [
    True
]  # the in-tree chain's arithmetic: argmax on the unrounded fp32 sum (see enable_fp32_sum)


def enable_fp32_sum(enabled: bool = True) -> bool:
    """Compare base + bias as the unrounded fp32 sum, as the in-tree chain does: its draft logits are fp32
    (LogitsProcessor's .float()) and it widens the bf16 bias, so its argmax sees fp32 sums. Off: round the sum to
    bf16 first (a bf16-logits reference). On by default; set before the first call of a shape."""
    _fp32_sum[0] = bool(enabled)
    return _fp32_sum[0]


def enable_batch1_variant(enabled: bool = True) -> bool:
    """Run batch-1 calls on dspark_markov_chain_kernel_b1: the same chain with the position loop's base loads issued
    before the W1 wait and the exchange polled in one first pass (bitwise the same tokens; faster at one request,
    slower at more, so only batch 1 takes it). Off by default; set before the first call of a shape."""
    _batch1_variant[0] = bool(enabled)
    return _batch1_variant[0]


def enable_ll_exchange(enabled: bool = True) -> bool:
    """Run the batch-1 variant's calls on dspark_markov_chain_kernel_b1_ll: the same chain with the per-position
    exchange as one 16-byte {value, tag, index, tag} multicast store per CTA (no fence, no flag word) polled and
    reduced by one warp (no CTA barrier after the poll); bitwise the same tokens. Only batch-1 calls with the batch-1
    variant on take it. Off by default; set it the same way on every rank, before the first call of a shape."""
    _ll_exchange[0] = bool(enabled)
    return _ll_exchange[0]


def enable_cached_inputs(enabled: bool = True) -> bool:
    """Take the kernel's read-only accepted inputs (``num_accepted`` ones, ``accepted_rows`` arange, ``accepted``
    zeros; they only feed ``next_new``, which the op discards) from a cache filled on the first call per (device,
    batch, block) outside CUDA-graph capture, instead of three fill/arange launches per call. The kernels never write
    them, so the tokens are bitwise the same. Off by default; set it before the first call of a shape."""
    _cached_inputs[0] = bool(enabled)
    return _cached_inputs[0]


def cached_inputs_taken() -> bool:
    """Whether a call since process start took its accepted inputs from the cache (the engagement of
    ``enable_cached_inputs``; a cache fill alone does not count)."""
    return _cached_taken[0]


def _accepted_inputs(
    batch: int, block: int, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(num_accepted ones [B], accepted_rows arange [B], accepted zeros [B, block + 1]), int32: cached under
    ``enable_cached_inputs`` (filled outside capture only; a miss under capture allocates afresh, uncached)."""
    key = (device, batch, block)
    inputs = _const_inputs.get(key) if _cached_inputs[0] else None
    if inputs is not None:
        _cached_taken[0] = True
    else:
        inputs = (
            torch.ones(batch, dtype=torch.int32, device=device),
            torch.arange(batch, dtype=torch.int32, device=device),
            torch.zeros(batch, block + 1, dtype=torch.int32, device=device),
        )
        if _cached_inputs[0] and not torch.cuda.is_current_stream_capturing():
            _const_inputs[key] = inputs
    return inputs


def ll_exchange_engaged(batch: int) -> bool:
    """Whether a call of ``batch`` requests runs the LL exchange kernel under the current switches."""
    return _batch1_variant[0] and _ll_exchange[0] and int(batch) == 1


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def pick_grid(shard: int, block: int, batch: int, device=None) -> int:
    """CTAs for a (padded) shard: 128 when supported, else the largest supported count up to the SM count; 0 if
    none."""
    kern = _kernel_module()
    limit = min(MAX_GRID, _sm_count(device if device is not None else torch.cuda.current_device()))
    if 128 <= limit and kern.supports(shard, 128, block, batch):
        return 128
    for grid in range(limit, 0, -1):
        if kern.supports(shard, grid, block, batch):
            return grid
    return 0


def padded_rows(shard: int, max_batch: int, device=None) -> int:
    """The smallest row count >= ``shard`` that the kernel splits for every batch <= ``max_batch`` and block <=
    MAX_BLOCK; 0 if none fits the shared-memory budget."""
    kern = _kernel_module()
    step = kern.WARPS * kern.BLOCK_ROWS
    rows = -(-shard // step) * step
    limit = rows + MAX_GRID * step
    while rows <= limit:
        if all(pick_grid(rows, MAX_BLOCK, b, device) > 0 for b in (1, max_batch)):
            return rows
        rows += step
    return 0


def pad_w2_shard(w2_shard: torch.Tensor, rows: int) -> torch.Tensor:
    """``w2_shard`` [S, R] bf16 with zero rows appended up to ``rows`` (done once at load)."""
    out = w2_shard.new_zeros(rows, w2_shard.shape[1])
    out[: w2_shard.shape[0]].copy_(w2_shard)
    return out


def pad_logits_buffer(
    batch: int, block: int, rows: int, device, dtype=torch.bfloat16
) -> torch.Tensor:
    """A [batch, block, rows] draft-logits buffer whose padding columns hold -inf (the draft head writes [..., :S])."""
    return torch.full((batch, block, rows), float("-inf"), dtype=dtype, device=device)


def workspace_args(mapping, group=None) -> dict:
    """This TP group's exchange workspace as the call's ``ws_*``, ``rank``, ``slots`` and ``buf_words`` arguments.

    Three rotating Lamport buffers (every word 0x80000000 when armed) and, after them, the LL variant's two regions
    of the same size (zero-filled: tag 0 is never a call's tag) behind one multicast mapping, and their flag words
    (the Lamport variants' [0..3], the LL variant's [4..7]); built once per (mapping, group) and reused. Collective
    on first use, outside CUDA-graph capture: every rank of the TP group calls it at the same point. ``group``
    overrides the communicator (a torch.distributed group when MPI is disabled).
    """
    key = (mapping, id(group) if group is not None else None)
    ws = _workspaces.get(key)
    if ws is not None:
        return ws["args"]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "dspark_markov_chain: the workspace must be allocated outside CUDA-graph capture"
        )
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    kern = _kernel_module()
    ll = _kernel_module(batch1=True, ll=True)
    world = mapping.tp_size
    buf_words = kern.buffer_words(MAX_BLOCK, kern.MAX_BATCH, world, MAX_GRID)
    comm = group if group is not None else _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    error: Optional[Exception] = None
    try:
        lamport_words = kern.BUFFERS * buf_words
        words = ll.ll_workspace_words(buf_words)
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        with torch.inference_mode():
            uc[:lamport_words].fill_(kern.EMPTY_WORD)
            uc[lamport_words:].zero_()
            flags = torch.zeros(
                max(kern.FLAG_WORDS, ll.LL_FLAG_WORDS), dtype=torch.int32, device=uc.device
            )
        torch.cuda.synchronize()
        args = dict(
            ws_uc=uc,
            ws_mc=mc,
            ws_flags=flags,
            rank=mapping.tp_rank,
            slots=world,
            buf_words=buf_words,
        )
        ws = dict(handle=handle, comm=comm, args=args)
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has armed it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError(
            "dspark_markov_chain Lamport buffers failed on at least one rank"
        ) from error
    _workspaces[key] = ws
    return ws["args"]


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


def supports(base: torch.Tensor, w1: torch.Tensor, w2_shard: torch.Tensor) -> bool:
    """Whether the kernel takes these operands (dtypes, ranks, the padded shard's split); a caller that gets False
    runs the in-tree chain."""
    kern = _kernel_module()
    if base.dim() != 3 or base.dtype != torch.bfloat16 or not base.is_cuda:
        return False
    batch, block, rows = base.shape
    return (
        w1.dtype == torch.bfloat16
        and w2_shard.dtype == torch.bfloat16
        and w1.dim() == 2
        and w1.shape[1] == kern.MARKOV_RANK
        and tuple(w2_shard.shape) == (rows, kern.MARKOV_RANK)
        and 0 < block <= MAX_BLOCK
        and 0 < batch <= kern.MAX_BATCH
        and pick_grid(rows, block, batch, base.device) > 0
    )


def _launch(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    kv_lens: Optional[torch.Tensor] = None,
    rewind: Optional[torch.Tensor] = None,
    rewind_first: int = 0,
) -> torch.Tensor:
    """One chain launch; with ``kv_lens`` and ``rewind``, also ``kv_lens[rewind_first + i] = max(kv_lens[...] -
    rewind[i], 0)`` for every i (after the kernel's grid-dependency wait)."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    kern = _kernel_module()
    if not supports(base, w1, w2_shard):
        raise ValueError(f"dspark_markov_chain: unsupported operands base {tuple(base.shape)} {base.dtype}, "
                         f"w2_shard {tuple(w2_shard.shape)}")  # fmt: skip
    if first_prev.dtype != torch.int64 or first_prev.numel() != base.shape[0]:
        raise ValueError("dspark_markov_chain: first_prev must be int64 [B]")
    batch, block, rows = base.shape
    batch1 = _batch1_variant[0] and batch == 1
    ll = batch1 and _ll_exchange[0]
    grid = pick_grid(rows, block, batch, base.device)
    if kern.buffer_words(block, batch, slots, grid) > buf_words:
        raise ValueError("dspark_markov_chain: the exchange does not fit the workspace buffers")
    if ll:
        ll_kern = _kernel_module(batch1=True, ll=True)
        if (
            ll_kern.ll_region_words(block, slots, grid) > buf_words
            or ws_uc.numel() < ll_kern.ll_workspace_words(buf_words)
            or ws_flags.numel() < ll_kern.LL_FLAG_WORDS
        ):
            raise ValueError("dspark_markov_chain: the LL exchange does not fit the workspace")
    device = base.device
    tokens = torch.empty(batch, block, dtype=torch.int32, device=device)
    next_new = torch.empty(batch, block + 1, dtype=torch.int32, device=device)
    dummy = _dummy.get(device)
    if dummy is None:
        # The kernel's optional outputs and inputs (corrected logits, accepted rows, KV-length rewind) are unused
        # here: one zero word stands for each, and an empty rewind leaves it untouched.
        dummy = _dummy[device] = torch.zeros(4, dtype=torch.int32, device=device)
    rewind_count = 0 if rewind is None else int(rewind.numel())
    if rewind_count:
        if not supports_rewind(kv_lens, rewind, rewind_first):
            raise ValueError(
                "dspark_markov_chain: unsupported KV-length rewind (see supports_rewind())"
            )
    else:
        kv_lens, rewind = dummy, dummy
    nacc, rows_idx, acc = _accepted_inputs(batch, block, device)
    args = (
        _arg(base.contiguous().view(-1).view(torch.int16)),
        _arg(first_prev.contiguous().view(-1), align=8),
        _arg(w1.contiguous().view(-1).view(torch.int32)),
        _arg(w2_shard.contiguous().view(-1).view(torch.int32)),
        _arg(dummy.view(torch.float32)),
        _arg(tokens.view(-1)),
        _arg(next_new.view(-1)),
        _arg(acc.view(-1), align=4),
        _arg(nacc, align=4),
        _arg(rows_idx, align=4),
        _arg(kv_lens, align=4),
        _arg(rewind, align=4),
        _arg(ws_uc.view(-1)),
        _arg(ws_mc.view(-1)),
        _arg(ws_flags.view(-1)),
    )
    scalars = (
        int(w1.shape[0]),
        int(shard_offset),
        int(rank),
        int(block + 1),
        int(rewind_first),
        rewind_count,
    )
    use_pdl = get_env_enable_pdl()
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    consts = (grid, rows // grid, rows, block, batch, slots, 1, buf_words, True, _fp32_sum[0])
    key = consts + (use_pdl, batch1, ll)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::dspark_markov_chain must run once per shape outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    _kernel_module(batch1, ll).k3_markov, *args, *scalars, *consts, use_pdl, stream
                )
    fn(*args, *scalars, stream)
    return tokens


@torch.library.custom_op("trtllm::dspark_markov_chain", mutates_args=())
def _dspark_markov_chain(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
) -> torch.Tensor:
    return _launch(
        base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words
    )


@_dspark_markov_chain.register_fake
def _(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words):
    return base.new_empty((base.shape[0], base.shape[1]), dtype=torch.int32)


@torch.library.custom_op("trtllm::dspark_markov_chain_rewind", mutates_args=("kv_lens",))
def _dspark_markov_chain_rewind(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    rewind_first: int,
) -> torch.Tensor:
    return _launch(
        base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words,
        kv_lens, rewind, rewind_first,
    )  # fmt: skip


@_dspark_markov_chain_rewind.register_fake
def _(base, first_prev, w1, w2_shard, shard_offset, ws_uc, ws_mc, ws_flags, rank, slots, buf_words, kv_lens,
      rewind, rewind_first):  # fmt: skip
    return base.new_empty((base.shape[0], base.shape[1]), dtype=torch.int32)


def supports_rewind(
    kv_lens: Optional[torch.Tensor], rewind: Optional[torch.Tensor], rewind_first: int
) -> bool:
    """Whether a chain call can also apply the KV-length rewind ``kv_lens[first + i] = max(kv_lens[first + i] -
    rewind[i], 0)``: int32 contiguous CUDA operands on one device, 1..``kernel MAX_BATCH`` amounts, in range. A caller
    that gets False applies the rewind itself."""
    if kv_lens is None or rewind is None:
        return False
    n = rewind.numel()
    return (
        kv_lens.is_cuda
        and rewind.is_cuda
        and kv_lens.device == rewind.device
        and kv_lens.dtype == torch.int32
        and rewind.dtype == torch.int32
        and kv_lens.is_contiguous()
        and rewind.is_contiguous()
        and 0 < n <= _kernel_module().MAX_BATCH
        and 0 <= int(rewind_first)
        and int(rewind_first) + n <= kv_lens.numel()
    )


def dspark_markov_chain_rewind(
    base: torch.Tensor,
    first_prev: torch.Tensor,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    rewind_first: int,
    *,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    slots: int,
    buf_words: int,
) -> torch.Tensor:
    """``dspark_markov_chain``'s tokens (the same launch, bitwise the same tokens) plus the target's KV-length rewind
    after the draft forward in the same launch (TAILBK_SPEC site H): ``kv_lens[rewind_first + i] =
    max(kv_lens[rewind_first + i] - rewind[i], 0)``, applied after the kernel's grid-dependency wait (every reader of
    ``kv_lens`` in the draft forward finished). Check ``supports_rewind`` first; the call is a collective as
    ``dspark_markov_chain``'s, and each rank rewinds its own ``kv_lens``. The kernel module's compile is shared with
    ``dspark_markov_chain`` (the rewind's operands are runtime arguments)."""
    return torch.ops.trtllm.dspark_markov_chain_rewind(
        base, first_prev, w1, w2_shard, int(shard_offset), ws_uc, ws_mc, ws_flags, int(rank), int(slots),
        int(buf_words), kv_lens, rewind, int(rewind_first),
    )  # fmt: skip


def precompile(
    mapping,
    rows: int,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    shard_offset: int,
    blocks: Tuple[int, ...],
    max_batch: int,
    group=None,
) -> None:
    """Compile every (batch <= ``max_batch``, block in ``blocks``) shape outside capture (collective: every rank)."""
    ws = workspace_args(mapping, group)
    for block in blocks:
        for batch in range(1, max_batch + 1):
            base = pad_logits_buffer(batch, block, rows, w2_shard.device)
            prev = torch.zeros(batch, dtype=torch.long, device=w2_shard.device)
            dspark_markov_chain(base, prev, w1, w2_shard, shard_offset, **ws)
    torch.cuda.synchronize()


class DsparkMarkovChain(OpWrapper):
    """Greedy draft tokens [B, K] (int32) of the DSpark vanilla-Markov head, from this rank's padded vocab shard
    of the block logits.

    ``base`` [B, K, rows] bf16: this rank's shard of the draft head's block logits (columns ``[0, shard)``), the
    padding columns -inf. ``first_prev`` [B] int64: the anchor tokens. ``w1`` [V, 256] bf16: ``markov_w1``, whole.
    ``w2_shard`` [rows, 256] bf16: this rank's rows of ``markov_w2``, zero-padded. ``shard_offset``: the vocabulary
    index of this rank's first column. ``ws_*``, ``rank``, ``slots``, ``buf_words``: ``workspace_args``'s.

    Every rank returns the same tokens. The call is a collective: every rank of the group calls it with the same
    (B, K), in the same order.
    """

    ARCHS = frozenset({Arch.SM_103})

    # MiniMax-M3.1's DSpark drafter: vocabulary 200064, Markov rank 256, K = 3 and 7 draft positions, the decode
    # batches the M3.1 targets serve with it (1..8). TP4 is one GB300 tray, TP8 two.
    _M31 = dict(vocab=200064, batches=tuple(range(1, CERTIFIED_BATCH + 1)))

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="M3.1 TP4, K = 3: the 50016-row shard, padded to 50688",
            spec=dict(world=4, block=3, **_M31),
        ),
        Cell(
            why="M3.1 TP4, K = 7: seven chained exchanges per call",
            spec=dict(world=4, block=7, **_M31),
        ),
        Cell(
            why="a shard that splits without padding (32768 / 4 = 8192 rows)",
            spec=dict(world=4, block=3, vocab=32768, batches=(1, 8)),
        ),
        Cell(
            why="M3.1 TP8 over two trays, K = 3: the 25008-row shard, padded to 25088; fabric-handle multicast",
            spec=dict(world=8, block=3, **_M31),
        ),
        Cell(why="M3.1 TP8 over two trays, K = 7", spec=dict(world=8, block=7, **_M31)),
    )

    note = """
    Greedy only, and only the vanilla Markov head: no d2t map, no candidate selector, no sampling. A caller with
    any of those runs the in-tree chain.

    The kernel takes batches up to its module's MAX_BATCH (16) and blocks up to MAX_BLOCK (8); the cells certify
    batches 1..8. Measured: at 16 requests the one-launch chain is 0.93x the in-tree chain's time, so the M3.1
    targets serve batches up to 8 only.

    The kernel also computes next_new_tokens, the accepted-row bookkeeping and the target's KV-length rewind of
    the K3 fused verify; this entry passes none of them (one dummy word each) and returns only the tokens.
    """

    def __call__(
        self,
        base: torch.Tensor,
        first_prev: torch.Tensor,
        w1: torch.Tensor,
        w2_shard: torch.Tensor,
        shard_offset: int,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        slots: int,
        buf_words: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.dspark_markov_chain(
            base,
            first_prev,
            w1,
            w2_shard,
            int(shard_offset),
            ws_uc,
            ws_mc,
            ws_flags,
            int(rank),
            int(slots),
            int(buf_words),
        )

    def reference(
        self,
        base_in_rank_order: Sequence[torch.Tensor],
        first_prev: torch.Tensor,
        w1: torch.Tensor,
        w2_shard_in_rank_order: Sequence[torch.Tensor],
        shard: int,
    ) -> torch.Tensor:
        """The DSpark worker's in-tree chain on the gathered full-vocabulary logits: per position ``bias =
        F.linear(w1[prev], w2)`` in bf16, ``logits + bias`` in the logits' dtype, ``torch.argmax`` (first maximum).
        With fp32_sum (default) the logits are widened to fp32 first, as LogitsProcessor's ``.float()`` hands them
        to the in-tree chain, so the bias is widened and the sum is not rounded; without it, bf16 logits + bf16 bias.

        Not more accurate than the in-tree chain: the tokens must equal its tokens bit for bit, so this is that
        chain. The documented collective exception to mirroring ``__call__``: it takes every rank's ``base`` and
        ``w2_shard`` in rank order, and ``shard``, the unpadded columns of each.
        """
        full = torch.cat([b[..., :shard] for b in base_in_rank_order], dim=-1)
        if _fp32_sum[0]:
            full = full.float()
        w2 = torch.cat([w[:shard] for w in w2_shard_in_rank_order], dim=0)
        prev = first_prev
        tokens = []
        for k in range(full.shape[1]):
            bias = F.linear(w1[prev], w2).to(full.dtype)
            prev = torch.argmax(full[:, k] + bias, dim=-1)
            tokens.append(prev)
        return torch.stack(tokens, 1).to(torch.int32)

    def is_valid(
        self,
        base: torch.Tensor,
        first_prev: torch.Tensor,
        w1: torch.Tensor,
        w2_shard: torch.Tensor,
        shard_offset: int,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        slots: int,
        buf_words: int,
    ) -> None:
        # The kernel takes every column of `base` as a candidate and offsets its index by `shard_offset`: a padding
        # column that is not -inf can win and return an index past this rank's shard, and a wrong offset returns
        # another rank's token ids. Neither is checked on the way to the kernel.
        vocab = w1.shape[0]
        assert vocab % slots == 0, (vocab, slots)
        shard = vocab // slots
        assert shard_offset == rank * shard, (shard_offset, rank, shard)
        assert base.shape[-1] >= shard, (base.shape, shard)
        assert bool(torch.isneginf(base[..., shard:]).all()), (
            "the padding columns of base must hold -inf"
        )
        # An anchor outside the vocabulary indexes `w1` past its end inside the kernel.
        assert bool(((first_prev >= 0) & (first_prev < vocab)).all()), (
            "first_prev outside [0, vocab)"
        )


dspark_markov_chain = DsparkMarkovChain()
