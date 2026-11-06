# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash 2 drafter's layer-0 front as a per-token table (part of the ``drafter_folds`` fast path).

On a decode step of one request the drafter's block is ``K + 1 = 8`` noise rows: the embedding of the
bonus token, then seven rows of the mask token's embedding (``verify_fold``'s noise gather). Layer 0 of
the block decode (``drafter_block_decode.windowed_block_decode``) runs on those rows only -- the target's
context enters later, through the window attention's context K/V -- so its front is a function of the
bonus id alone:

* ``input_layernorm`` is per row;
* ``attention_convs[0].prepare``: the coefficient projection is per row, and the two-tap block
  convolution mixes row ``t`` with row ``t - 1`` of the same block (masked at the block's first row), so
  row 0 depends on the bonus row, row 1 on the bonus row and the mask row, rows 2-7 on mask rows only;
  ``finish``'s coefficients (side 1, read by the layer's attention hop) are row ``t``'s own;
* ``qkv_proj`` is per row.

So rows 0-1 of the pre-RoPE ``qkv`` and row 0 of the side-1 coefficients depend on the bonus id; the
other rows are constants. This module builds, after the drafter's weights and fast paths are in place,
a table of those rows for every vocab id (plus one entry for an id outside the table, whose noise row
the gather writes as +0.0) by running the installed statements -- the same modules, at M = 8, on
``[v, MASK x 7]``, under inference mode -- and stores the constant rows once. Per step, one Triton
launch in the place of the verify fold prepare's noise gather (``replicated_embedding_gather_twice``,
``drafter_preamble``) writes the
noise block, its residual copy, all 8 ``qkv`` rows and all 8 side-1 coefficient rows (the QK norm +
RoPE rewrites ``qkv`` in place, so every row is rewritten every step). The drafter forward then skips
layer 0's input norm, coefficient GEMV, prepare convolution and ``qkv`` GEMV. Every value is a bit copy
of what the module chain computes for that id: value-identical by construction, certified by the
install-time check (``check``) and the tests.

Ownership and order:
* The tables have one writer, the install's build, and are read-only afterwards.
* Per step the gather is the single writer of layer 0's ``qkv`` rows and side-1 coefficient rows (in
  the module chain those are the two GEMVs); like the gather it replaces it waits
  (``griddepcontrol.wait``) before its first read of the ids and releases its dependent after the wait.
  Its readers -- layer 0's QK norm + RoPE (in place, after its own wait), the window attention's split
  (``k_new`` / ``v_new`` after its wait) and the layer's attention hop (the coefficients after its
  wait) -- follow it in stream order behind kernels that each wait before they complete; the next
  step's gather follows this step's readers in stream order. No flag, no spin.
* The tables are a function of the loaded weights and the embedding table. ``install`` keys them on
  those tensors (identity and address) and rebuilds when a load hook sees other tensors.

Memory: (vocab + 1) x (2 x qkv width + side-1 width) BF16 per rank; at TP8 (qkv 1280, side 1 1024) and
202,048 ids, 1.45 GB, allocated during the drafter load, i.e. before the KV cache is sized.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm.logger import logger

from ...ops.embedding.replicated_gather import BLOCK as GATHER_BLOCK
from ...ops.embedding.replicated_gather import replicated_embedding_gather

#: How the table is built: ``graph`` (CUDA graphs of ``BUILD_CHUNK`` chains; ``build`` also takes ``eager``).
BUILD_MODE = "graph"
#: Chains per captured build graph.
BUILD_CHUNK = 256
#: Random ids the install-time check runs the live chain on (besides the edge ids and ``CHECK_IDS``).
CHECK_RANDOM = 4096
#: A file of whitespace-separated token ids the check also covers, or empty for none.
CHECK_IDS = ""

#: The block width the table is built for (K + 1 with K = 7).
BLOCK_SIZE = 8
#: Leading block rows of ``qkv`` that depend on the bonus id (row 0 its own, row 1 through the conv tap).
QKV_ROWS = 2
#: Leading block rows of the side-1 coefficients that depend on the bonus id.
COEF_ROWS = 1
#: Largest call (ids) the gather takes: 64 requests of 8 rows.
MAX_ROWS = 512


@triton.jit
def _tt_gather_kernel(
    table_ptr,
    ids_ptr,
    noise_ptr,
    copy_ptr,
    qkv_tab_ptr,
    qkv_const_ptr,
    qkv_ptr,
    coef_tab_ptr,
    coef_const_ptr,
    coef_ptr,
    vocab,
    HIDDEN: tl.constexpr,
    NQ: tl.constexpr,
    NC: tl.constexpr,
    COEF_ROW: tl.constexpr,
    COEF_OFF: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    QKV_DYN: tl.constexpr,
    COEF_DYN: tl.constexpr,
    NOISE_CHUNKS: tl.constexpr,
    QKV_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr = False,
):
    # Program (row, chunk): chunks [0, NOISE_CHUNKS) are ``_replicated_embedding_gather2_kernel``'s
    # body for this row (noise block and residual copy); the next QKV_CHUNKS write this row's qkv
    # columns, the last one its side-1 coefficients. Every value is moved as its 16 bits.
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if LAUNCH_WITH_PDL:
        # Before any global read: the ids are the previous kernel's output.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    lanes = tl.arange(0, BLOCK)
    if chunk < NOISE_CHUNKS:
        token = tl.load(ids_ptr + row).to(tl.int64)
        valid = (token >= 0) & (token < vocab)
        cols = chunk * BLOCK + lanes
        in_row = cols < HIDDEN
        source = tl.where(valid, token, 0) * HIDDEN
        bits = tl.load(table_ptr + source + cols, mask=in_row & valid, other=0)
        magnitude = bits.to(tl.int32) & 0x7FFF
        bits = tl.where(magnitude == 0, tl.zeros_like(bits), bits)
        tl.store(noise_ptr + row.to(tl.int64) * HIDDEN + cols, bits, mask=in_row)
        tl.store(copy_ptr + row.to(tl.int64) * HIDDEN + cols, bits, mask=in_row)
    else:
        pos = row % BLOCK_ROWS
        bonus = tl.load(ids_ptr + (row - pos)).to(tl.int64)
        # Entry ``vocab`` holds the chain of an id outside the table (a +0.0 noise row).
        entry = tl.where((bonus >= 0) & (bonus < vocab), bonus, vocab)
        part = chunk - NOISE_CHUNKS
        if part < QKV_CHUNKS:
            cols = part * BLOCK + lanes
            in_row = cols < NQ
            if pos < QKV_DYN:
                bits = tl.load(
                    qkv_tab_ptr + (entry * QKV_DYN + pos) * NQ + cols, mask=in_row, other=0
                )
            else:
                bits = tl.load(qkv_const_ptr + pos * NQ + cols, mask=in_row, other=0)
            tl.store(qkv_ptr + row.to(tl.int64) * NQ + cols, bits, mask=in_row)
        else:
            cols = (part - QKV_CHUNKS) * BLOCK + lanes
            in_row = cols < NC
            if pos < COEF_DYN:
                bits = tl.load(coef_tab_ptr + entry * NC + cols, mask=in_row, other=0)
            else:
                bits = tl.load(coef_const_ptr + pos * NC + cols, mask=in_row, other=0)
            tl.store(coef_ptr + row.to(tl.int64) * COEF_ROW + COEF_OFF + cols, bits, mask=in_row)


def layer0_front(model: nn.Module, noise_rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(qkv, coefficients)`` of the installed layer-0 front on ``noise_rows`` (``[rows, hidden]``).

    The statements of ``windowed_block_decode``'s layer 0 before the QK norm + RoPE: the input norm
    (the ``kernels/ox_entry`` entry-trigger norm when installed and taking the rows, else
    ``input_layernorm``), then
    ``attention_convs[0].prepare`` and ``qkv_proj``. ``coefficients`` is ``prepare``'s side-1 view
    (``[rows, taps, groups]``). Call under inference mode, as the forward runs.
    """
    layer = model.model.layers[0]
    entry_norm = model.__dict__.get("_ox_entry_norm")
    if entry_norm is not None and entry_norm.takes(noise_rows):
        normed = entry_norm(noise_rows)
    else:
        normed = layer.input_layernorm(noise_rows)
    convolved, coefficients = model.attention_convs[0].prepare(normed, BLOCK_SIZE)
    qkv = layer.self_attn.qkv_proj(convolved)
    return qkv, coefficients


def noise_ids(bonus: torch.Tensor, mask_id: int) -> torch.Tensor:
    """``[n, 8]`` int64 blocks ``[bonus[i], MASK x 7]`` for the ids in ``bonus`` (``[n]``)."""
    ids = torch.full(
        (bonus.numel(), BLOCK_SIZE), int(mask_id), dtype=torch.int64, device=bonus.device
    )
    ids[:, 0] = bonus.to(torch.int64)
    return ids


class DrafterL0Table:
    """The tables and the per-step gather that stands in for the drafter's layer-0 front.

    ``qkv_tab`` ``[V + 1, 2, nq]`` and ``coef_tab`` ``[V + 1, nc]`` hold the bonus-dependent rows of
    every id (entry ``V``: an id outside the table); ``qkv_const`` ``[8, nq]`` / ``coef_const``
    ``[8, nc]`` the constant rows (rows below 2 / 1 unused).
    """

    def __init__(
        self,
        qkv_tab: torch.Tensor,
        coef_tab: torch.Tensor,
        qkv_const: torch.Tensor,
        coef_const: torch.Tensor,
        table: torch.Tensor,
        mask_id: int,
        taps: int,
        groups: int,
        key: tuple,
    ) -> None:
        self.qkv_tab = qkv_tab
        self.coef_tab = coef_tab
        self.qkv_const = qkv_const
        self.coef_const = coef_const
        self.table = table
        self.mask_id = int(mask_id)
        self.taps = int(taps)
        self.groups = int(groups)
        self.nq = int(qkv_tab.shape[-1])
        self.nc = int(coef_tab.shape[-1])
        self.key = key
        #: Calls taken (a test counts them).
        self.calls = 0

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.qkv_tab, self.coef_tab, self.qkv_const, self.coef_const)
        )

    def takes(self, table: torch.Tensor, ids: torch.Tensor, mask_id: Optional[int]) -> bool:
        """Whether a step's noise gather of ``ids`` from ``table`` can be this gather (host-side, rank-uniform).

        The table must be the one the rows were built from, the ids ``[g, 8]`` contiguous int64 blocks
        (the fold's noise ids: the bonus id, then the mask id the fold was given, ``mask_id``).
        """
        return (
            mask_id is not None
            and int(mask_id) == self.mask_id
            and table is self.table
            and ids.dtype == torch.int64
            and ids.dim() == 2
            and ids.shape[1] == BLOCK_SIZE
            and ids.is_contiguous()
            and ids.device == table.device
            and 1 <= ids.numel() <= MAX_ROWS
        )

    def __call__(self, table: torch.Tensor, ids: torch.Tensor, pdl: bool = False):
        """``(noise, residual, (qkv, coefficients))`` for ``takes`` inputs.

        ``noise`` / ``residual``: ``replicated_embedding_gather_twice(table, ids)``'s outputs
        (``[g, 8, hidden]``). ``qkv``: ``[g * 8, nq]``, layer 0's pre-RoPE ``qkv_proj`` output.
        ``coefficients``: the side-1 view ``[g * 8, taps, groups]`` of a ``[g * 8, 2, taps, groups]``
        buffer, as ``prepare`` hands ``finish``'s coefficients to the hop.
        """
        vocab, hidden = table.shape
        rows = ids.numel()
        device = table.device
        noise = torch.empty((*ids.shape, hidden), dtype=table.dtype, device=device)
        residual = torch.empty_like(noise)
        qkv = torch.empty((rows, self.nq), dtype=self.qkv_tab.dtype, device=device)
        coef = torch.empty((rows, 2 * self.nc), dtype=self.coef_tab.dtype, device=device)
        noise_chunks = triton.cdiv(hidden, GATHER_BLOCK)
        qkv_chunks = triton.cdiv(self.nq, GATHER_BLOCK)
        coef_chunks = triton.cdiv(self.nc, GATHER_BLOCK)
        _tt_gather_kernel[(rows, noise_chunks + qkv_chunks + coef_chunks)](
            table.view(torch.int16),
            ids.reshape(-1),
            noise.view(torch.int16),
            residual.view(torch.int16),
            self.qkv_tab.view(torch.int16),
            self.qkv_const.view(torch.int16),
            qkv.view(torch.int16),
            self.coef_tab.view(torch.int16),
            self.coef_const.view(torch.int16),
            coef.view(torch.int16),
            vocab,
            HIDDEN=hidden,
            NQ=self.nq,
            NC=self.nc,
            COEF_ROW=2 * self.nc,
            COEF_OFF=self.nc,
            BLOCK_ROWS=BLOCK_SIZE,
            QKV_DYN=QKV_ROWS,
            COEF_DYN=COEF_ROWS,
            NOISE_CHUNKS=noise_chunks,
            QKV_CHUNKS=qkv_chunks,
            BLOCK=GATHER_BLOCK,
            **({"LAUNCH_WITH_PDL": True, "launch_pdl": True} if pdl else {}),
            num_warps=4,
        )
        self.calls += 1
        side1 = coef.view(rows, 2, self.taps, self.groups)[:, 1]
        return noise, residual, (qkv, side1)


def _key(model: nn.Module, table: torch.Tensor, mask_id: int) -> tuple:
    """The tensors the tables are a function of: identity and address of each, and the mask id."""
    layer = model.model.layers[0]
    tensors = (
        table,
        layer.input_layernorm.weight,
        model.attention_convs[0].kernel_projection.weight,
        model.attention_convs[0].base_kernel,
        layer.self_attn.qkv_proj.weight,
    )
    return tuple((id(t), t.data_ptr()) for t in tensors) + (int(mask_id),)


def unsupported_reason(model: Optional[nn.Module], table: Optional[torch.Tensor]) -> Optional[str]:
    """Why the table cannot stand in for this drafter's layer-0 front, or None (rank-uniform)."""
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    from ..drafter_block_decode import block_decode as _block_decode

    if not isinstance(model, DFlashForCausalLM):
        return f"the drafter is {type(model).__name__}, not a DFlashForCausalLM"
    bound = model.__dict__.get("dflash_forward")
    if getattr(bound, "__func__", None) is not _block_decode.windowed_block_decode:
        return "the drafter forward is not drafter_block_decode's windowed_block_decode"
    if not getattr(model, "has_block_conv", False):
        return "the drafter has no DFlash 2 block convolutions"
    if getattr(model, "block_size", None) != BLOCK_SIZE:
        return f"the drafter block is {getattr(model, 'block_size', None)} rows, not {BLOCK_SIZE}"
    if table is None:
        return "the token embedding is not the replicated table (replicated_embedding)"
    if table.dtype != torch.bfloat16 or table.dim() != 2 or not table.is_contiguous():
        return "the embedding table is not a contiguous [vocab, hidden] BF16 tensor"
    attn = model.model.layers[0].self_attn
    weight = getattr(attn.qkv_proj, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return "layer 0's qkv_proj has no BF16 weight"
    if weight.shape[0] != attn.q_size + 2 * attn.kv_size:
        return "layer 0's qkv_proj width is not q + 2 kv"
    conv = model.attention_convs[0]
    if conv.kernel_projection.weight.dtype != torch.bfloat16:
        return "layer 0's coefficient projection is not BF16"
    return None


def _mask_id(llm: nn.Module, model: nn.Module) -> int:
    """The mask id the verify fold writes into the noise ids (its own resolution order)."""
    worker = getattr(llm, "spec_worker", None)
    resolve = getattr(worker, "_fold_mask_token_id", None)
    if resolve is not None:
        return int(resolve(model))
    return int(model.mask_token_id)


def _empty_tables(model: nn.Module, vocab: int, device):
    attn = model.model.layers[0].self_attn
    conv = model.attention_convs[0]
    nq = attn.q_size + 2 * attn.kv_size
    nc = conv.taps * conv.num_groups
    qkv_tab = torch.empty((vocab + 1, QKV_ROWS, nq), dtype=torch.bfloat16, device=device)
    coef_tab = torch.empty((vocab + 1, nc), dtype=torch.bfloat16, device=device)
    return qkv_tab, coef_tab, nq, nc, conv.taps, conv.num_groups


def _entry_ids(vocab: int, device) -> torch.Tensor:
    """Bonus id of each table entry: ``0 .. vocab - 1``, then ``-1`` (an id outside the table)."""
    ids = torch.arange(vocab + 1, dtype=torch.int64, device=device)
    ids[vocab] = -1
    return ids


def _build_eager(model, table, mask_id, qkv_tab, coef_tab, entries) -> None:
    hidden = table.shape[1]
    blocks = noise_ids(entries, mask_id)
    for index in range(entries.numel()):
        noise = replicated_embedding_gather(table, blocks[index]).view(BLOCK_SIZE, hidden)
        qkv, coefficients = layer0_front(model, noise)
        qkv_tab[index].copy_(qkv[:QKV_ROWS])
        coef_tab[index].copy_(coefficients[0].reshape(-1))


def _build_graph(model, table, mask_id, qkv_tab, coef_tab, entries, chunk: int) -> None:
    hidden = table.shape[1]
    device = table.device
    chunk = max(1, min(chunk, entries.numel()))
    ids_buf = noise_ids(entries[:chunk], mask_id)
    stage_qkv = torch.empty((chunk,) + tuple(qkv_tab.shape[1:]), dtype=qkv_tab.dtype, device=device)
    stage_coef = torch.empty((chunk, coef_tab.shape[1]), dtype=coef_tab.dtype, device=device)

    def chains():
        for i in range(chunk):
            noise = replicated_embedding_gather(table, ids_buf[i]).view(BLOCK_SIZE, hidden)
            qkv, coefficients = layer0_front(model, noise)
            stage_qkv[i].copy_(qkv[:QKV_ROWS])
            stage_coef[i].copy_(coefficients[0].reshape(-1))

    # Eager pass first: every kernel is compiled outside a capture.
    chains()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chains()
    total = entries.numel()
    for base in range(0, total, chunk):
        n = min(chunk, total - base)
        ids_buf[:n, 0].copy_(entries[base : base + n])
        graph.replay()
        qkv_tab[base : base + n].copy_(stage_qkv[:n])
        coef_tab[base : base + n].copy_(stage_coef[:n])
    torch.cuda.synchronize(device)
    del graph


def build(
    model: nn.Module,
    table: torch.Tensor,
    mask_id: int,
    mode: str = BUILD_MODE,
    chunk: int = BUILD_CHUNK,
    vocab: Optional[int] = None,
) -> DrafterL0Table:
    """Run the installed layer-0 front for every id of ``table`` and return the tables.

    ``vocab`` limits the build to the first ``vocab`` ids (tests); the gather then treats every
    larger id as outside the table, as the noise gather does for ids past the table.
    """
    vocab = table.shape[0] if vocab is None else int(vocab)
    if vocab != table.shape[0]:
        table = table[:vocab]
    device = table.device
    with torch.inference_mode():
        qkv_tab, coef_tab, nq, nc, taps, groups = _empty_tables(model, vocab, device)
        # The constant rows: rows 2-7 of qkv and rows 1-7 of the coefficients of any block.
        probe = replicated_embedding_gather(
            table, noise_ids(torch.zeros(1, dtype=torch.int64, device=device), mask_id)[0]
        ).view(BLOCK_SIZE, table.shape[1])
        qkv, coefficients = layer0_front(model, probe)
        qkv_const = torch.zeros((BLOCK_SIZE, nq), dtype=qkv.dtype, device=device)
        coef_const = torch.zeros((BLOCK_SIZE, nc), dtype=coefficients.dtype, device=device)
        qkv_const[QKV_ROWS:].copy_(qkv[QKV_ROWS:])
        coef_const[COEF_ROWS:].copy_(coefficients[COEF_ROWS:].reshape(BLOCK_SIZE - COEF_ROWS, nc))
        entries = _entry_ids(vocab, device)
        if mode == "eager":
            _build_eager(model, table, mask_id, qkv_tab, coef_tab, entries)
        elif mode == "graph":
            _build_graph(model, table, mask_id, qkv_tab, coef_tab, entries, chunk)
        else:
            raise ValueError(f"the table build mode must be graph or eager, got {mode!r}")
    return DrafterL0Table(
        qkv_tab,
        coef_tab,
        qkv_const,
        coef_const,
        table,
        mask_id,
        taps,
        groups,
        _key(model, table, mask_id),
    )


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())


def check(
    model: nn.Module, tt: DrafterL0Table, table: torch.Tensor, ids: torch.Tensor, pdl: bool = False
) -> Tuple[int, list]:
    """The live chain vs the table gather on the blocks of ``ids`` (bonus ids); ``(checked, mismatches)``.

    For every id: the gather's noise block and residual copy vs ``replicated_embedding_gather``, and its
    8 ``qkv`` rows and 8 side-1 coefficient rows vs ``layer0_front`` on that noise block, bit for bit.
    """
    bad = []
    hidden = table.shape[1]
    blocks = noise_ids(ids.to(table.device), tt.mask_id)
    with torch.inference_mode():
        for index in range(blocks.shape[0]):
            block = blocks[index : index + 1]
            noise, residual, (qkv, side1) = tt(table, block, pdl)
            want_noise = replicated_embedding_gather(table, block)
            want_qkv, want_coef = layer0_front(model, want_noise.view(BLOCK_SIZE, hidden))
            diffs = (
                _bits_differ(noise, want_noise),
                _bits_differ(residual, want_noise),
                _bits_differ(qkv, want_qkv),
                _bits_differ(side1, want_coef),
            )
            if any(diffs):
                bad.append((int(block[0, 0]), diffs))
                if len(bad) >= 16:
                    break
    return blocks.shape[0], bad


def _check_ids(vocab: int, mask_id: int, device) -> torch.Tensor:
    """Edge ids, ``CHECK_RANDOM`` random ids (the same on every rank) and the ids in ``CHECK_IDS``."""
    edges = [0, 1, vocab - 1, vocab, -1, mask_id]
    generator = torch.Generator().manual_seed(20261101)
    random = torch.randint(0, vocab, (CHECK_RANDOM,), generator=generator).tolist()
    listed = []
    if CHECK_IDS:
        with open(CHECK_IDS) as handle:
            listed = [int(token) for token in handle.read().split()]
    return torch.tensor(edges + random + listed, dtype=torch.int64, device=device), len(listed)


def install(llm: nn.Module) -> Optional[DrafterL0Table]:
    """Build (or keep) the drafter's layer-0 table and attach it as ``_tt_drafter_l0``. Rank-local.

    Idempotent: the shell installs from every load hook; the build runs again only when the weights
    or the embedding table it reads are other tensors than the last build's.

    Raises:
        ValueError: if the drafter does not take the table path (``unsupported_reason``) or the
            install-time check finds a row that differs from the live chain.
    """
    from ..replicated_embedding.embedding import table_of

    model = getattr(llm, "draft_model", None)
    embed = getattr(
        getattr(getattr(model, "draft_model_full", None), "model", None), "embed_tokens", None
    )
    table = table_of(embed)
    reason = unsupported_reason(model, table)
    if reason is not None:
        raise ValueError(
            f"[musespark_open] drafter layer-0 table (drafter_folds) cannot run here: {reason}"
        )
    mask_id = _mask_id(llm, model)
    existing = model.__dict__.get("_tt_drafter_l0")
    if isinstance(existing, DrafterL0Table) and existing.key == _key(model, table, mask_id):
        return existing
    model.__dict__.pop("_tt_drafter_l0", None)
    device = table.device
    torch.cuda.synchronize(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    start = time.perf_counter()
    tt = build(model, table, mask_id)
    torch.cuda.synchronize(device)
    built = time.perf_counter() - start
    ids, listed = _check_ids(table.shape[0], mask_id, device)
    checked, bad = check(model, tt, table, ids)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info(device)
    if bad:
        raise ValueError(
            f"[musespark_open] drafter layer-0 table: the table differs from the live chain for {len(bad)} of "
            f"{checked} checked ids (id, (noise, residual, qkv, coefficient) bits differing): {bad[:8]}"
        )
    model.__dict__["_tt_drafter_l0"] = tt
    logger.info(
        f"[musespark_open] drafter layer-0 table: layer 0's input norm, coefficient GEMV, prepare conv and "
        f"qkv GEMV served from a per-token table ({table.shape[0]} ids + 1, qkv {tt.nq} x {QKV_ROWS} "
        f"rows + side-1 coefficients {tt.nc}; {tt.nbytes / 2**30:.3f} GiB, device free "
        f"{free_before / 2**30:.2f} -> {free_after / 2**30:.2f} GiB; built in {built:.1f} s "
        f"({BUILD_MODE}); check: {checked} ids ({listed} listed) bitwise the live chain)"
    )
    return tt
