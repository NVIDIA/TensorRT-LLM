# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's layer-0 QKV(G) projection as a per-token table (the ``layer0_qkv_table`` fast path).

On a decode step the target's layer 0 starts with the fused pre (``kernels/pf_pre``): per row,
the embedding gather, the embedding norm, layer 0's gemma pre-QKV norm (BF16, the attention input) and
the exact FP32 widening (the residual stream). Layer 0's fused QKV(G) projection then multiplies the
attention input by the rank's ``[2304, 8192]`` weight with the dense-GEMM plan's kernel for the step's
row count (``kernels/dense_gemms``). Every one of these is per row, so a row of the projection is a
function of that row's token id alone, provided the GEMM's row ``i`` does not depend on the other rows
of its call (``row_independent``; checked at install).

This module builds, after the target's weights and fast paths are in place, a table of the projection's
row for every vocab id (plus one entry for an id outside the table, whose embedding row the fused pre
reads as zeros) by running the installed statements -- ``fused_pre`` and the plan's 8-row kernel -- on
calls of 8 distinct ids, under inference mode. Per decode step of 8 rows (``DecodeTarget``):

* a Triton gather writes the step's ``[8, 2304]`` projection from the table on the main stream, and
  layer 0's QKV dispatcher returns it instead of running the GEMM (``DenseGemmDispatcher.substitute``);
* the fused pre still writes the FP32 residual (and the unused attention input), on a side stream
  forked before the gather (``NamedFork`` ``tq_pre``), and the fork is joined by layer 0's attention
  output projection's dispatcher, before that GEMM, i.e. before the post-attention hop reads the
  residual.

Every value is a bit copy of what the built-in chain computes for that id: value-identical by
construction, certified by the install-time checks (``check``: eager calls of random 8-id batches,
and CUDA-graph replays of the gather against the live chain on fresh ids) and the tests.

Ownership and order:
* The table has one writer, the install's build, and is read-only afterwards.
* Per step the gather is the single writer of layer 0's QKV(G) output (in the built-in chain, the GEMM);
  it waits (``griddepcontrol.wait``) before its first read of the ids and releases its dependent after
  the wait. Its reader, the sliding-window split, reads the operand after its own wait.
* The fused pre is the single writer of the residual; its outputs are allocated on the main stream
  before the fork, so their blocks belong to the main stream, which reads them only after the join.
  The fork starts after the main stream's work so far (the ids are written by then) and the fused pre
  waits before its first read, as in the built-in chain. Layer 0's ``o_proj`` dispatcher joins the fork
  before its GEMM, the post-attention hop (the first reader of the residual) waits on that GEMM, so the
  residual is complete before it is read. A forward hook on layer 0 raises if the step's fork or
  substitute is still outstanding after the layer, so a path that bypassed either cannot go unnoticed.
* The table is a function of the loaded weights and the embedding table; ``install`` keys it on those
  tensors (identity and address) and rebuilds when a load hook sees other tensors.

Memory: (vocab + 1) x 2304 BF16 per rank, allocated in the target's post-load hook, i.e. before the
KV cache is sized.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm.logger import logger

#: ``1``: the table installed, the fused pre forked beside the gather (variant 1); ``0`` off. The
#: shell installs it only with the ``layer0_qkv_table`` fast path (``_install_single_request``).
TQ_TARGET_L0 = "1"
#: Random 8-id batches the install-time eager check runs the live chain on (besides the edge ids and
#: ``CHECK_IDS``); each batch also checks the permuted order.
CHECK_BATCHES = 512
#: CUDA-graph replays of the install-time replay check (a fresh random batch each).
CHECK_REPLAYS = 300
#: Optional file of whitespace-separated token ids the check also covers ("" = none).
CHECK_IDS = ""

#: The decode-step row count the table is built and used for (K + 1 with K = 7).
ROWS = 8
#: Columns per gather program.
BLOCK = 256
#: Name of the fused pre's fork in the dense-GEMM plan.
FORK_NAME = "tq_pre"

#: The mode of an installed table: "" variant 1, the fast path. The other two keep every value and
#: separate the table's effect from its side effects (attribution): ``hold`` installs the table, its
#: checks, the fork stream and the layer-0 guard exactly as variant 1 does (same allocations in the
#: same order) but never arms a step, so every step runs the built-in chain (fused pre, then layer 0's
#: QKV GEMM) on the memory map of variant 1; ``inline`` arms every step with the fused pre on the
#: current stream ahead of the gather instead of on the fork's stream (no fork, no join).
SP_TQ_ATTR = ""
_SP_TQ_ATTR_MODES = ("", "hold", "inline")
#: Attribution only ("0" = none): a device allocation of this many MiB, never read or written, held
#: from this install point on (with or without the table), so it is in place before the KV cache is
#: sized; it moves every later allocation's address (and shrinks the KV pool by its size) and
#: changes nothing else.
SP_PAD_MIB = "0"


def enabled() -> bool:
    """Whether the table is installed (variant 1, ``TQ_TARGET_L0``)."""
    return TQ_TARGET_L0 == "1"


def attribution_mode() -> str:
    """``SP_TQ_ATTR`` ("" = variant 1, "hold", "inline").

    Raises:
        ValueError: on any other value.
    """
    if SP_TQ_ATTR not in _SP_TQ_ATTR_MODES:
        raise ValueError(
            f"[musespark_open] SP_TQ_ATTR={SP_TQ_ATTR!r} is not one of {_SP_TQ_ATTR_MODES}"
        )
    return SP_TQ_ATTR


@triton.jit
def _tq_gather_kernel(
    ids_ptr,
    table_ptr,
    out_ptr,
    vocab,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr = False,
):
    # Program (row, chunk): BLOCK columns of the row's projection, moved as their 16 bits.
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if LAUNCH_WITH_PDL:
        # Before any global read: the ids are a previous kernel's output.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.load(ids_ptr + row).to(tl.int64)
    # Entry ``vocab`` holds the chain of an id outside the table.
    entry = tl.where((token >= 0) & (token < vocab), token, vocab)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    in_row = cols < N
    bits = tl.load(table_ptr + entry * N + cols, mask=in_row, other=0)
    tl.store(out_ptr + row.to(tl.int64) * N + cols, bits, mask=in_row)


def gather(table: torch.Tensor, ids: torch.Tensor, pdl: bool = False) -> torch.Tensor:
    """``[rows, N]``: row ``i`` is ``table[ids[i]]``, or ``table[V]`` for an id outside ``[0, V)``.

    ``table`` is ``[V + 1, N]`` BF16 (contiguous); ``ids`` 1-D int32 / int64 (contiguous)."""
    rows = ids.numel()
    n = int(table.shape[1])
    out = torch.empty((rows, n), dtype=table.dtype, device=table.device)
    _tq_gather_kernel[(rows, triton.cdiv(n, BLOCK))](
        ids,
        table.view(torch.int16),
        out.view(torch.int16),
        int(table.shape[0]) - 1,
        N=n,
        BLOCK=BLOCK,
        **({"LAUNCH_WITH_PDL": True, "launch_pdl": True} if pdl else {}),
        num_warps=4,
    )
    return out


def row_independent(product: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> List[str]:
    """What breaks row independence of ``product`` (an ``[8, K] -> [8, N]`` map) on ``x``, or ``[]``.

    Checks, bit for bit, that a permutation of the rows permutes the output rows, and that a row's
    output does not change when every other row of the call is replaced."""
    problems = []
    base = product(x)
    generator = torch.Generator(device="cpu").manual_seed(20261101)
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    permuted = product(x[perm].contiguous())
    if not torch.equal(permuted.view(torch.int16), base[perm].view(torch.int16)):
        problems.append("a permutation of the rows does not permute the output rows")
    for row in range(x.shape[0]):
        other = torch.randn_like(x, dtype=torch.float32).to(x.dtype)
        other[row] = x[row]
        if not torch.equal(product(other)[row].view(torch.int16), base[row].view(torch.int16)):
            problems.append(f"row {row} changes with the other rows of its call")
    return problems


class TargetL0Table:
    """The table and the per-step gather that stand in for layer 0's QKV(G) GEMM.

    ``table`` ``[V + 1, N]`` BF16; entry ``V`` is an id outside the embedding table. ``embedding`` is
    the replicated embedding table the rows were built from.
    """

    def __init__(
        self, table: torch.Tensor, embedding: torch.Tensor, kernel: str, key: tuple
    ) -> None:
        self.table = table
        self.embedding = embedding
        self.kernel = kernel
        self.key = key
        self.vocab = int(table.shape[0]) - 1
        self.n = int(table.shape[1])
        #: Steps taken (a test counts them).
        self.calls = 0
        #: Set while a step is armed (between ``front`` and layer 0's end).
        self.armed = False

    @property
    def nbytes(self) -> int:
        return self.table.numel() * self.table.element_size()

    def takes(self, embedding: Optional[torch.Tensor], input_ids: Optional[torch.Tensor]) -> bool:
        """Whether a decode step of ``input_ids`` can take the table (host-side, rank-uniform)."""
        return (
            isinstance(input_ids, torch.Tensor)
            and embedding is self.embedding
            and input_ids.is_cuda
            and input_ids.dtype in (torch.int32, torch.int64)
            and input_ids.numel() == ROWS
            and input_ids.is_contiguous()
            and not torch.is_grad_enabled()
        )

    def __call__(self, ids: torch.Tensor, pdl: bool = False) -> torch.Tensor:
        return gather(self.table, ids.reshape(-1), pdl)


def _qkv_site(core: nn.Module):
    """Layer 0's QKV dispatcher (``kernels/dense_gemms``), or None."""
    linear = core.layers[0].self_attn.qkv_proj
    return linear.__dict__.get("_low_m_dispatcher")


def _o_proj_site(core: nn.Module):
    linear = core.layers[0].self_attn.o_proj
    return linear.__dict__.get("_low_m_dispatcher")


def live_chain(core: nn.Module, pdl: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
    """The built-in chain from 8 ids to layer 0's QKV(G) output: ``fused_pre`` then the plan's 8-row kernel."""
    from ...ops.gemm.dense_gemm import dense_gemm
    from ..pf_pre import pre as _pf_pre

    dispatcher = _qkv_site(core)
    plan = dispatcher.site.plan
    kernel = plan.kernels["qkv"][ROWS - 1]
    weight = core.layers[0].self_attn.qkv_proj.weight

    def chain(ids: torch.Tensor) -> torch.Tensor:
        attn, _ = _pf_pre.fused_pre(core, ids, pdl=pdl)
        return dense_gemm(attn, weight, kernel, plan.enable_pdl and pdl)

    return chain


def build(
    chain: Callable[[torch.Tensor], torch.Tensor], vocab: int, n: int, device
) -> torch.Tensor:
    """``[vocab + 1, n]``: ``chain`` on the ids ``0 .. vocab - 1`` in calls of 8 consecutive ids, and on
    ``-1`` (entry ``vocab``). Under inference mode."""
    table = torch.empty((vocab + 1, n), dtype=torch.bfloat16, device=device)
    ids = torch.arange(vocab + ROWS, dtype=torch.int64, device=device)
    ids[vocab:] = -1
    with torch.inference_mode():
        for begin in range(0, vocab + 1, ROWS):
            rows = chain(ids[begin : begin + ROWS])
            end = min(begin + ROWS, vocab + 1)
            table[begin:end] = rows[: end - begin]
    return table


def _check_ids(vocab: int, device) -> torch.Tensor:
    """The ids the eager check covers: edges, ``CHECK_BATCHES`` random 8-id batches and ``CHECK_IDS``."""
    generator = torch.Generator(device="cpu").manual_seed(20261102)
    parts = [
        torch.tensor([0, 1, vocab - 1, vocab, -1, vocab + 7, 2, 3], dtype=torch.int64),
        torch.randint(0, vocab, (CHECK_BATCHES * ROWS,), generator=generator, dtype=torch.int64),
    ]
    if CHECK_IDS:
        with open(CHECK_IDS) as listed:
            values = [int(v) for v in listed.read().split()]
        parts.append(torch.tensor(values, dtype=torch.int64))
    ids = torch.cat(parts)
    pad = (-ids.numel()) % ROWS
    if pad:
        ids = torch.cat([ids, ids[:pad]])
    return ids.to(device)


def check(
    chain: Callable[[torch.Tensor], torch.Tensor], tq: TargetL0Table, ids: torch.Tensor
) -> Tuple[int, List[int]]:
    """``(ids checked, ids whose row differs)``: the gather vs ``chain`` on ``ids`` in calls of 8, in
    the given and in a reversed order, bit for bit (at most 16 differing ids listed)."""
    bad: List[int] = []
    with torch.inference_mode():
        for begin in range(0, ids.numel(), ROWS):
            batch = ids[begin : begin + ROWS].contiguous()
            for order in (batch, batch.flip(0).contiguous()):
                live = chain(order)
                got = tq(order)
                rows = (live.view(torch.int16) != got.view(torch.int16)).any(dim=1)
                if bool(rows.any()):
                    for i in torch.nonzero(rows).flatten().tolist():
                        if len(bad) < 16:
                            bad.append(int(order[i]))
    return int(ids.numel()), bad


def check_replays(
    chain: Callable[[torch.Tensor], torch.Tensor], tq: TargetL0Table, replays: int, pdl: bool
) -> int:
    """Differing values over ``replays`` CUDA-graph replays of the gather against ``chain`` (one graph,
    both on one static ids buffer refilled with a fresh random batch before each replay)."""
    device = tq.table.device
    ids = torch.zeros(ROWS, dtype=torch.int32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(20261103)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.inference_mode(), torch.cuda.stream(stream):
        chain(ids)
        tq(ids, pdl)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            live = chain(ids)
            got = tq(ids, pdl)
            diff = (live.view(torch.int16) != got.view(torch.int16)).sum()
        total = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(replays):
            ids.copy_(torch.randint(0, tq.vocab, (ROWS,), generator=generator, dtype=torch.int32))
            graph.replay()
            total += diff
    torch.cuda.current_stream(device).wait_stream(stream)
    return int(total.item())


def unsupported_reason(core: nn.Module) -> Optional[str]:
    """Why the table cannot stand in for this target's layer-0 front, or None (rank-uniform)."""
    from ..pf_pre import pre as _pf_pre
    from ..replicated_embedding.embedding import table_of

    if not _pf_pre.FUSED_PRE:
        return "the fused pre is off"
    reason = _pf_pre.unsupported_reason(core)
    if reason is not None:
        return f"the fused pre does not apply: {reason}"
    if table_of(getattr(core, "embed_tokens", None)) is None:
        return "the token embedding is not the replicated table"
    qkv, o_proj = _qkv_site(core), _o_proj_site(core)
    if qkv is None or o_proj is None:
        return "layer 0's projections are not on the dense-GEMM dispatchers (dense_gemms)"
    if qkv.site.plan is not o_proj.site.plan:
        return "layer 0's QKV and o_proj dispatchers do not share a plan"
    megakernel = getattr(core, "_megakernel", None)
    if megakernel is not None and megakernel.replaces(0):
        return "the megakernel runs layer 0"
    weight = core.layers[0].self_attn.qkv_proj.weight
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
        return "layer 0's QKV weight is not a contiguous 2-D BF16 tensor"
    return None


def _key(core: nn.Module, embedding: torch.Tensor) -> tuple:
    layer = core.layers[0]
    tensors = (
        embedding,
        core.embed_norm.weight,
        layer.pre_qkv_proj_norm.weight,
        layer.self_attn.qkv_proj.weight,
    )
    return tuple((id(t), t.data_ptr()) for t in tensors)


class _LayerZeroGuard:
    """Forward hook on layer 0: an armed step must have used its substitute and joined its fork."""

    def __init__(self, tq: TargetL0Table, fork) -> None:
        self.tq = tq
        self.fork = fork

    def __call__(self, module, args, output):
        tq = self.tq
        if not tq.armed:
            return output
        tq.armed = False
        dispatcher = _qkv_site(module_core(module))
        if dispatcher is not None and dispatcher.substitute is not None:
            dispatcher.substitute = None
            raise RuntimeError(
                "[musespark_open] target table: layer 0 ran without its QKV dispatcher taking the gathered rows"
            )
        if self.fork.pending:
            raise RuntimeError(
                "[musespark_open] target table: layer 0 ended before its o_proj dispatcher joined the fused pre's fork"
            )
        return output


def module_core(layer: nn.Module) -> nn.Module:
    return layer.__dict__["_tq_core"]


def front(core: nn.Module, input_ids: torch.Tensor, pdl: bool):
    """``(layer0_input, hidden_states)`` of an armed step, or None when the step cannot take the
    table.

    Forks the fused pre (into outputs allocated on the current stream), launches the gather on the
    current stream and hands its rows to layer 0's QKV dispatcher."""
    from ..pf_pre import pre as _pf_pre
    from ..replicated_embedding.embedding import table_of

    tq: Optional[TargetL0Table] = core.__dict__.get("_tq_target_l0")
    if tq is None or not tq.takes(table_of(core.embed_tokens), input_ids):
        return None
    mode = attribution_mode()
    if mode == "hold":
        # Attribution: the table stays resident, the step runs the built-in chain.
        return None
    fork = core.__dict__["_tq_fork"]
    dispatcher = _qkv_site(core)
    ids = input_ids.reshape(-1)
    device = ids.device
    attn = torch.empty((ROWS, _pf_pre.HIDDEN), dtype=torch.bfloat16, device=device)
    residual = torch.empty((ROWS, _pf_pre.HIDDEN), dtype=torch.float32, device=device)
    if mode == "inline":
        # Attribution: the fused pre on the current stream, then the gather (nothing to join).
        _pf_pre.fused_pre(core, ids, pdl, out=(attn, residual))
    elif not fork.fork(lambda: _pf_pre.fused_pre(core, ids, pdl, out=(attn, residual)), device):
        logger.warning_once(
            "[musespark_open] target_l0 table: a decode step inside a capture found no fork stream; it runs "
            "the served chain",
            key="musespark_open.tq_target_l0.skipped",
        )
        return None
    dispatcher.substitute = tq(ids, pdl)
    tq.armed = True
    tq.calls += 1
    return attn, residual


def install(core: nn.Module) -> Optional[TargetL0Table]:
    """Build (or keep) the table and attach it as ``_tq_target_l0``. Rank-local; idempotent.

    Raises:
        RuntimeError: if the install-time checks find a row that is not the live chain's.
    """
    from ..replicated_embedding.embedding import table_of

    reason = unsupported_reason(core)
    if reason is not None:
        logger.warning(f"[musespark_open] target table not installed: {reason}")
        return None
    embedding = table_of(core.embed_tokens)
    key = _key(core, embedding)
    held: Optional[TargetL0Table] = core.__dict__.get("_tq_target_l0")
    if held is not None and held.key == key:
        return held
    core.__dict__.pop("_tq_target_l0", None)
    device = embedding.device
    weight = core.layers[0].self_attn.qkv_proj.weight
    n = int(weight.shape[0])
    vocab = int(embedding.shape[0])
    plan = _qkv_site(core).site.plan
    kernel = plan.kernels["qkv"][ROWS - 1]
    torch.cuda.synchronize(device)
    free_before = torch.cuda.mem_get_info(device)[0]
    start = time.monotonic()
    chain = live_chain(core, pdl=False)
    with torch.inference_mode():
        x = torch.randn((ROWS, int(weight.shape[1])), device=device).to(torch.bfloat16)
        from ...ops.gemm.dense_gemm import dense_gemm

        problems = row_independent(lambda a: dense_gemm(a, weight, kernel, False), x)
    if problems:
        raise RuntimeError(
            f"[musespark_open] target table: the 8-row QKV kernel {kernel} is not row-independent: "
            + "; ".join(problems)
        )
    table = build(chain, vocab, n, device)
    torch.cuda.synchronize(device)
    built = time.monotonic() - start
    tq = TargetL0Table(table, embedding, kernel, key)
    checked, bad = check(chain, tq, _check_ids(vocab, device))
    replay_diff = check_replays(live_chain(core, pdl=True), tq, CHECK_REPLAYS, pdl=True)
    if bad or replay_diff:
        raise RuntimeError(
            f"[musespark_open] target table: the table differs from the live chain (ids {bad}, "
            f"{replay_diff} values over {CHECK_REPLAYS} graph replays)"
        )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_after = torch.cuda.mem_get_info(device)[0]
    fork = core.__dict__.get("_tq_fork")
    if fork is None:
        fork = plan.named_forks.get(FORK_NAME) or plan.add_fork(FORK_NAME, "o_proj")
        core.__dict__["_tq_fork"] = fork
    if fork.stream is None:
        # Make the fork's stream now (outside any capture), so a decode graph captured before any
        # eager decode step still forks the fused pre instead of skipping the table.
        fork.fork(lambda: None, device)
        fork.join()
    layer = core.layers[0]
    if "_tq_core" not in layer.__dict__:
        layer.__dict__["_tq_core"] = core
        layer.register_forward_hook(_LayerZeroGuard(tq, fork))
    else:
        for hook in layer._forward_hooks.values():
            if isinstance(hook, _LayerZeroGuard):
                hook.tq = tq
    core.__dict__["_tq_target_l0"] = tq
    gib = float(1 << 30)
    mode = attribution_mode()
    steps = {
        "": "fused pre forked, joined at layer 0's o_proj",
        "hold": "ATTRIBUTION hold (SP_TQ_ATTR): table resident, no step armed",
        "inline": "ATTRIBUTION inline (SP_TQ_ATTR): fused pre on the main stream before the gather",
    }[mode]
    logger.info(
        f"[musespark_open] target table: table [{vocab + 1}, {n}] {tq.nbytes / gib:.3f} GiB, device free "
        f"{free_before / gib:.2f} -> {free_after / gib:.2f} GiB; built in {built:.1f} s with {kernel} "
        f"(row-independent); check: {checked} ids x 2 orders and {CHECK_REPLAYS} graph replays bitwise "
        f"the live chain; {steps}"
    )
    return tq


def pad_mib() -> int:
    """``SP_PAD_MIB`` as a non-negative integer (0 = no pad).

    Raises:
        ValueError: on a value that is not a non-negative integer.
    """
    if not SP_PAD_MIB.isdigit():
        raise ValueError(
            f"[musespark_open] SP_PAD_MIB={SP_PAD_MIB!r} is not a non-negative integer"
        )
    return int(SP_PAD_MIB)


def install_pad(core: nn.Module) -> Optional[torch.Tensor]:
    """Hold ``pad_mib()`` MiB of device memory on ``core`` as ``_sp_pad``; idempotent. None when 0."""
    mib = pad_mib()
    if mib == 0:
        return None
    held = core.__dict__.get("_sp_pad")
    if held is not None and held.numel() == mib << 20:
        return held
    device = next(core.parameters()).device
    pad = torch.empty(mib << 20, dtype=torch.uint8, device=device)
    core.__dict__["_sp_pad"] = pad
    logger.info(
        f"[musespark_open] SP_PAD_MIB pad: {mib} MiB held before the KV cache is sized at {pad.data_ptr():#x} "
        "(attribution only; never read or written)"
    )
    return pad


def install_served(core: nn.Module) -> None:
    """``install`` when ``enabled()``; then the attribution pad, when ``SP_PAD_MIB`` holds one."""
    if enabled():
        install(core)
    install_pad(core)
