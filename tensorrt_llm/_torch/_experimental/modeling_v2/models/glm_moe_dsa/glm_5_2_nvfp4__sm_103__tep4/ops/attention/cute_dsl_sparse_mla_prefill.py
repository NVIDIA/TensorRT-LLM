# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context sparse MLA over the fp8 latent pool, top-k 2048, on the sm_103 CuTe DSL kernel:
over an index tensor, or dense causal through the block table."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_fmha_sparse_mla_prefill_h16_qk576_v512_topk2048_fp8_bf16 as _kernel,
)

TOP_K = _kernel.TOPK
NUM_HEADS = _kernel.NUM_HEADS
KV_LORA_RANK = _kernel.L
LATENT_DIM = _kernel.KV_STRIDE
#: The paged block size the dense-causal mode's address arithmetic is built for.
DENSE_CAUSAL_BLOCK = 32
#: Query tokens per dense-causal work item.
DENSE_CAUSAL_GROUP = _kernel.G_TOK
#: `(new_lens, cached_lens | None, total_q) -> (ks, ke)`: the dense-causal key windows,
#: int32 on the device, `ks = 0` and `ke = cached + i + 1` for new row `i` of a request.
#: Both lengths are int32 on the device; reads their sum back to the host.
dense_causal_bounds = _kernel.dense_causal_bounds
#: The dense-causal work items `[G, 2] = (q0, n_tok)` from the host int32 new-row
#: lengths: up to four consecutive rows of one request each, request-major.
dense_causal_groups = _kernel.dense_causal_groups


@dataclass(frozen=True)
class CompiledSparseMlaPrefill:
    """The one kernel `compile` built and the mode it was built for."""

    device_index: int
    all_sparse_indices_valid: bool
    kernel: object
    dense_causal: bool = False


@dataclass(frozen=True)
class PreparedDenseCausal:
    """A dense-causal launch bound to one context step's operands.

    `runner` is the kernel's validated, address-bound launcher for the step's
    first layer. The op re-launches its compiled function for each later layer
    with only `output` and `layer_offset` re-bound, the two operands that change
    between layers; everything else is a step constant.
    """

    compiled: CompiledSparseMlaPrefill
    runner: object


#: The dense-causal launch ABI the re-bind writes into: output at 5, the layer
#: offset last of 13 (q, pool, indices, ks, ke, out, bmm1, bmm2, req_idx,
#: block_table, groups, stride, layer_offset).
_ARG_OUTPUT, _ARG_LAYER_OFFSET, _NUM_ARGS = 5, 12, 13


def _warm(compiled: object, all_valid: bool, dense_causal: bool, device_index: int) -> None:
    """Launch `compiled` once on eight rows of one request and synchronize: the first
    call of a compiled function binds and loads its module, which must not happen
    inside a served step. The dense-causal arm reads the rows through a one-entry
    block table, the index arm through the equivalent index tensor (all row 0 for
    the all-valid build, which reads every entry)."""
    dev = torch.device("cuda", device_index)
    rows = 8
    q = torch.zeros((rows, NUM_HEADS, LATENT_DIM), dtype=torch.float8_e4m3fn, device=dev)
    pool = torch.zeros(
        (2 * DENSE_CAUSAL_BLOCK, 1, LATENT_DIM), dtype=torch.float8_e4m3fn, device=dev
    )
    ke = torch.arange(1, rows + 1, dtype=torch.int32, device=dev)
    common = dict(
        cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32, device=dev),
        cu_seqlen_ke=ke,
        mla_bmm1_scale=torch.tensor(
            (0.0625, 0.0625 * math.log2(math.e)), dtype=torch.float32, device=dev
        ),
        mla_bmm2_scale=torch.ones(1, dtype=torch.float32, device=dev),
    )
    out = torch.empty((rows, NUM_HEADS, KV_LORA_RANK), dtype=torch.bfloat16, device=dev)
    if dense_causal:
        runner = _kernel.prepare_runner(
            q, pool, None, out, None, None, **common,
            block_table=torch.zeros((1, 1), dtype=torch.int32, device=dev),
            req_idx=torch.zeros(rows, dtype=torch.int32, device=dev),
            stride_factor=DENSE_CAUSAL_BLOCK,
            layer_offset=0,
            dense_causal_groups=dense_causal_groups(torch.tensor([rows], dtype=torch.int32)),
        )  # fmt: skip
    else:
        slots = torch.arange(TOP_K, dtype=torch.int32, device=dev)[None, :].expand(rows, -1)
        indices = (
            torch.zeros_like(slots)
            if all_valid
            else torch.where(slots < ke[:, None], slots, torch.full_like(slots, -1))
        )
        runner = _kernel.prepare_runner(
            q, pool, indices.contiguous(), out, None, None, **common,
            all_sparse_indices_valid=all_valid,
        )  # fmt: skip
    dataclasses.replace(runner, compiled=compiled)()
    torch.cuda.synchronize(dev)


def compile(  # noqa: A001 -- the kernel module's own name for this step
    *, device_index: int, all_sparse_indices_valid: bool = False, dense_causal: bool = False
) -> CompiledSparseMlaPrefill:
    """Compile the kernel; return the handle the op takes as `compiled`.

    Every shape is symbolic (query count, pool rows, bounds capacity), so one
    compile covers every context step. `dense_causal` builds the block-table,
    per-token-mask mode instead of the index mode (never with
    `all_sparse_indices_valid`). Call once at load. Costs one CuTe DSL compile
    and one launch on eight zero rows.
    """
    kernel = _kernel.compile(
        all_sparse_indices_valid=bool(all_sparse_indices_valid),
        dense_block_table=bool(dense_causal),
        dense_causal=bool(dense_causal),
        device_index=int(device_index),
    )
    _warm(kernel, bool(all_sparse_indices_valid), bool(dense_causal), int(device_index))
    return CompiledSparseMlaPrefill(
        device_index=int(device_index),
        all_sparse_indices_valid=bool(all_sparse_indices_valid),
        kernel=kernel,
        dense_causal=bool(dense_causal),
    )


def prepare_dense_causal(
    compiled: CompiledSparseMlaPrefill,
    quant_q_buffer: torch.Tensor,
    kv_pool_view: torch.Tensor,
    output: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    block_table: torch.Tensor,
    req_idx: torch.Tensor,
    stride_factor: int,
    layer_offset: int,
    dense_causal_groups: torch.Tensor,
) -> PreparedDenseCausal:
    """Validate and bind one context step's dense-causal operands, on its first layer.

    The op takes the result as `compiled` for every layer of the step, with that
    layer's `output` and `layer_offset`. Host work only (some sixty contract
    checks); no compile, no sync. Raises on a handle not built for dense causal or
    on any operand the kernel rejects.
    """
    if not compiled.dense_causal:
        raise ValueError("prepare_dense_causal needs a handle compiled with dense_causal=True")
    runner = _kernel.prepare_runner(
        quant_q_buffer, kv_pool_view, None, output, None, None,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        block_table=block_table,
        req_idx=req_idx,
        stride_factor=int(stride_factor),
        layer_offset=int(layer_offset),
        dense_causal_groups=dense_causal_groups,
    )  # fmt: skip
    if len(runner.args) != _NUM_ARGS or not isinstance(runner.args[_ARG_LAYER_OFFSET], int):
        raise RuntimeError(f"the kernel's launch ABI changed: {len(runner.args)} arguments")
    return PreparedDenseCausal(
        compiled=compiled, runner=dataclasses.replace(runner, compiled=compiled.kernel)
    )


def _pool_bytes(kv_pool_view: torch.Tensor) -> torch.Tensor:
    """The pool as `[rows, 576]` e4m3 bytes, whatever its own shape. Bytes, so
    that rows can be gathered; view the gathered rows as e4m3."""
    return kv_pool_view.reshape(-1).view(torch.uint8).reshape(-1, LATENT_DIM)


def _dense_rows(cu_seqlen_ke, block_table, req_idx, stride_factor, layer_offset, total_q):
    """`[total_q, 2048]` int64 pool rows the dense-causal mode reads: query `q` takes
    positions `0 .. min(ke[q], 2048) - 1` of request `req_idx[q]`, at
    `block_table[req, pos // 32] * stride_factor + pos % 32 + layer_offset`; -1 past
    them. A block past the table's width reads as -1."""
    pos = torch.arange(TOP_K, device=block_table.device)[None, :]
    n = cu_seqlen_ke[:total_q].to(torch.int64).clamp(0, TOP_K)[:, None]
    blk = pos // DENSE_CAUSAL_BLOCK
    width = block_table.shape[1]
    table = block_table.to(torch.int64)[req_idx[:total_q].to(torch.int64)]
    page = torch.gather(table, 1, blk.clamp(max=width - 1).expand(total_q, -1))
    rows = page * int(stride_factor) + pos % DENSE_CAUSAL_BLOCK + int(layer_offset)
    keep = (pos < n) & (blk < width) & (page >= 0)
    return torch.where(keep, rows, torch.full_like(rows, -1))


def _check_dense_causal(
    quant_q_buffer, kv_pool_view, sparse_attn_indices, output, cu_seqlen_ks, cu_seqlen_ke,
    mla_bmm1_scale, mla_bmm2_scale, compiled, block_table, req_idx, stride_factor,
    layer_offset, dense_causal_groups,
) -> None:  # fmt: skip
    """The dense-causal inputs the kernel takes and answers wrongly."""
    total_q = int(quant_q_buffer.shape[0])
    assert sparse_attn_indices is None, "the dense-causal mode reads no index tensor"
    assert block_table is not None and req_idx is not None and stride_factor is not None, (
        "the dense-causal mode needs block_table, req_idx and stride_factor"
    )
    assert dense_causal_groups is not None, "the dense-causal mode needs its group table"
    # The kernel never reads `ks`: every window starts at position 0.
    assert bool((cu_seqlen_ks[:total_q] == 0).all()), (
        "cu_seqlen_ks must be 0 for every row; the dense-causal kernel does not read it"
    )
    # Every position a row sees must be a real block: an in-window -1, or a block
    # past the table's width, is outside the kernel's contract.
    ke = cu_seqlen_ke[:total_q].cpu().long().clamp(0, TOP_K)
    req = req_idx[:total_q].cpu().long()
    table = block_table.cpu().long()
    need = (ke + DENSE_CAUSAL_BLOCK - 1) // DENSE_CAUSAL_BLOCK
    assert int(need.max()) <= table.shape[1], "a row's window runs past the block table"
    cols = torch.arange(table.shape[1])[None, :]
    assert bool(((table[req] >= 0) | (cols >= need[:, None])).all()), (
        "a -1 block inside a row's window"
    )
    pool_rows = kv_pool_view.numel() * kv_pool_view.element_size() // LATENT_DIM
    top = (
        table[req].clamp(min=0).max(1).values * int(stride_factor)
        + DENSE_CAUSAL_BLOCK
        - 1
        + int(layer_offset)
    )
    assert int(top.max()) < pool_rows, f"a row past the pool's {pool_rows} rows"
    # Each work item is up to four consecutive rows of one request; together they
    # cover every query row once. A group across two requests reads the first's KV.
    groups = dense_causal_groups.cpu().long()
    q0, n_tok = groups[:, 0], groups[:, 1]
    assert bool(((n_tok >= 1) & (n_tok <= DENSE_CAUSAL_GROUP)).all()), "a group of 0 or >4 rows"
    covered = torch.zeros(total_q, dtype=torch.long)
    for a, n in zip(q0.tolist(), n_tok.tolist()):
        assert a >= 0 and a + n <= total_q, "a group past total_q"
        assert bool((req[a : a + n] == req[a]).all()), "a group spans two requests"
        covered[a : a + n] += 1
    assert bool((covered == 1).all()), "the groups do not cover every query row exactly once"
    if isinstance(compiled, PreparedDenseCausal):
        # The launch re-binds only `output` and `layer_offset`: any other operand
        # that is not the one prepared is silently replaced by it.
        bound = compiled.runner.args
        for name, given, at in (
            ("quant_q_buffer", quant_q_buffer, 0),
            ("kv_pool_view", kv_pool_view, 1),
            ("cu_seqlen_ks", cu_seqlen_ks, 3),
            ("cu_seqlen_ke", cu_seqlen_ke, 4),
            ("mla_bmm1_scale", mla_bmm1_scale, 6),
            ("mla_bmm2_scale", mla_bmm2_scale, 7),
            ("req_idx", req_idx, 8),
            ("block_table", block_table, 9),
        ):
            assert given.data_ptr() == bound[at].data_ptr(), (
                f"{name} is not the tensor this step was prepared with"
            )
        assert torch.equal(groups, bound[10].cpu().long()), "dense_causal_groups changed"
        assert int(stride_factor) == bound[11], "stride_factor changed"
        # The kernel's own checks of `output` ran on the first layer's only.
        assert (
            output.dtype == torch.bfloat16
            and output.is_contiguous()
            and output.numel() == total_q * NUM_HEADS * KV_LORA_RANK
            and output.data_ptr() % 32 == 0
        ), "output must be contiguous, 32-byte aligned bf16 of total_q * 16 * 512"


class CuteDslSparseMlaPrefill(OpWrapper):
    """Absorbed sparse MLA for the context rows, into `output`.

    `quant_q_buffer` is `[T, 16, 576]` e4m3 (`[q_nope_absorbed | q_rope]`),
    `kv_pool_view` the fp8 latent pool (any contiguous shape, read as
    `[rows, 576]`), `sparse_attn_indices` `[T, 2048]` int32 global pool rows,
    `output` `T * 16 * 512` bf16 (any contiguous shape). `cu_seqlen_ks` /
    `cu_seqlen_ke` are the per-row key windows the indexer used, at least `T`
    long. For query row `q` with selected rows `R` (its non-negative entries):

        output[q, h] = softmax_r(s * q[q, h] . kv[r]) @ kv[r, :512] * bmm2,  r in R

    with `s = mla_bmm1_scale[0]` and `bmm2 = mla_bmm2_scale[0]`; a row with no
    selected rows is zero. Returns `output`.

    Dense-causal mode (a handle compiled with `dense_causal=True`, or one
    `prepare_dense_causal` bound for the step): `sparse_attn_indices` is None and
    `R` is the causal window itself, positions `0 .. min(ke[q], 2048) - 1` of
    request `req_idx[q]`, at pool rows
    `block_table[req, pos // 32] * stride_factor + pos % 32 + layer_offset`
    (`ks` is 0). `dense_causal_groups` (`dense_causal_groups(...)`) splits the
    rows into work items of up to four consecutive rows of one request. The
    target takes it on a step where the engine skipped the indexer for the
    context rows, so the dense window is the selection.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this for the context rows of every prefill step, with
    # the index rows already global pool rows (the cached prefix included)
    # and its constant scales (1/16 natural, 1/16 * log2(e) base 2, output 1).
    # The cells vary what a context step varies: how many requests, where the
    # key window starts, and whether a chunk's rows cross top_k so that some
    # rows select 2048 of more keys and others have -1 tails. The dense-causal
    # cells cover what that mode adds: request tails that leave a partial group,
    # several requests, a cached prefix, and the per-layer re-bind of a step's
    # prepared launch.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request of 100 rows: every row below top_k, -1 tails of every "
            "length from 1948 to 2047",
            spec=dict(new=[100], cached=[0]),
        ),
        Cell(
            why="three requests in one call, the last with a cached prefix: key windows "
            "start past 0, and 466 rows recycle the persistent CTAs",
            spec=dict(new=[37, 300, 129], cached=[0, 0, 64]),
        ),
        Cell(
            why="a chunk past 2048: 120 new rows after a 1990-token prefix, so the "
            "row count crosses top_k mid-chunk, next to a short request",
            spec=dict(new=[64, 120], cached=[0, 1990]),
        ),
        Cell(
            why="dense causal, one request of 102 rows: a two-row tail group, windows "
            "ending mid-block",
            spec=dict(new=[102], cached=[0], dense_causal=True),
        ),
        Cell(
            why="dense causal, three requests with 1- and 3-row tail groups, the last "
            "after a 64-row cached prefix",
            spec=dict(new=[37, 300, 131], cached=[0, 0, 64], dense_causal=True),
        ),
        Cell(
            why="dense causal through a prepared step: first layer bound, a second "
            "layer re-bound to its own output and layer offset",
            spec=dict(new=[5, 70], cached=[33, 0], dense_causal=True, prepared=True),
        ),
    )

    note = """
    The target compiles both kernels at load (the -1-tail-safe index mode, and
    the dense-causal mode) and calls the op for every context row of a step;
    there is no stock fallback once the rope/KV append has run. The legacy
    `cu_q_seqlens` / `sequence_length` arguments of the kernel are not device
    inputs and the target passes None for both, so this op does not take them.

    The dense-causal mode runs only on a step where the engine skipped the
    indexer for the context rows (every context sequence within top_k) and
    the KV block size is 32; any other step takes the index mode. The target
    prepares one launch per step on the first layer (`prepare_dense_causal`)
    and re-binds output and layer offset for the others.

    The kernel reads `mla_bmm1_scale[1]` (the base-2 softmax scale) and
    `mla_bmm2_scale[0]`, never `mla_bmm1_scale[0]`; `is_valid` holds `[1]` to
    `[0] * log2(e)`. An unprepared call is validated by the kernel (host
    checks, no device sync); it rejects loudly a non-sm_103 device,
    misalignment, an output overlapping any input, and short bounds buffers.
    A prepared call is not: `is_valid` holds its operands to the bound ones.

    Not certified: `all_sparse_indices_valid=True`, or a step above the cells'
    468 rows. The kernel's query count is symbolic, so a longer step runs the
    same code with more CTA recycling.
    """

    def __call__(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        sparse_attn_indices: torch.Tensor | None,
        output: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        compiled: CompiledSparseMlaPrefill | PreparedDenseCausal,
        block_table: torch.Tensor | None = None,
        req_idx: torch.Tensor | None = None,
        stride_factor: int | None = None,
        layer_offset: int = 0,
        dense_causal_groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(compiled, PreparedDenseCausal):
            # Re-launch the step's bound function with this layer's output and
            # layer offset; every other operand is the step's.
            runner = compiled.runner
            bound = runner.args
            _kernel.PreparedMlaPrefillRunner(
                compiled=runner.compiled,
                args=(
                    *bound[:_ARG_OUTPUT],
                    output.reshape(int(bound[0].shape[0]), NUM_HEADS, KV_LORA_RANK),
                    *bound[_ARG_OUTPUT + 1 : _ARG_LAYER_OFFSET],
                    int(layer_offset),
                ),
                device_index=runner.device_index,
            )()
            return output
        if quant_q_buffer.device.index != compiled.device_index:
            raise ValueError(
                f"the kernel was compiled for device {compiled.device_index}, not "
                f"{quant_q_buffer.device}"
            )
        # Preparing looks up the kernel by its mode and device: the handle's own
        # key, so it finds the cached build rather than compiling. The mode is
        # the handle's, never the arguments': other operands would select a
        # mode nothing compiled at load.
        if compiled.dense_causal:
            if dense_causal_groups is None or block_table is None:
                raise ValueError("a dense-causal handle needs block_table and dense_causal_groups")
            runner = _kernel.prepare_runner(
                quant_q_buffer, kv_pool_view, None, output, None, None,
                cu_seqlen_ks=cu_seqlen_ks,
                cu_seqlen_ke=cu_seqlen_ke,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                block_table=block_table,
                req_idx=req_idx,
                stride_factor=stride_factor,
                layer_offset=int(layer_offset),
                dense_causal_groups=dense_causal_groups,
            )  # fmt: skip
        else:
            if block_table is not None or dense_causal_groups is not None:
                raise ValueError("an index-mode handle takes no block table or group table")
            runner = _kernel.prepare_runner(
                quant_q_buffer,
                kv_pool_view,
                sparse_attn_indices,
                output,
                None,
                None,
                cu_seqlen_ks=cu_seqlen_ks,
                cu_seqlen_ke=cu_seqlen_ke,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                all_sparse_indices_valid=compiled.all_sparse_indices_valid,
            )
        dataclasses.replace(runner, compiled=compiled.kernel)()
        return output

    def reference(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        sparse_attn_indices: torch.Tensor | None,
        output: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        compiled: CompiledSparseMlaPrefill | PreparedDenseCausal,
        block_table: torch.Tensor | None = None,
        req_idx: torch.Tensor | None = None,
        stride_factor: int | None = None,
        layer_offset: int = 0,
        dense_causal_groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Softmax attention over each query's non-negative rows, in fp64,
        rounded to bf16 once; a new tensor shaped like `output`.

        Index mode: `cu_seqlen_ks` / `cu_seqlen_ke` and the handle are not read:
        which rows count is said by the indices alone, and `is_valid` holds the
        kernel's positional reading of them to the same set. Dense-causal mode
        (no indices): the rows are the causal window through the block table
        (`cu_seqlen_ks`, the groups and the handle are not read).
        """
        scale = float(mla_bmm1_scale[0])
        out_scale = float(mla_bmm2_scale[0])
        q = quant_q_buffer.view(torch.uint8).view(torch.float8_e4m3fn).float().double()
        if sparse_attn_indices is None:
            rows = _dense_rows(
                cu_seqlen_ke, block_table, req_idx, stride_factor, layer_offset, q.shape[0]
            )
        else:
            rows = sparse_attn_indices.to(torch.int64)
        kv = _pool_bytes(kv_pool_view)
        out = torch.zeros(
            q.shape[0], q.shape[1], KV_LORA_RANK, dtype=torch.float64, device=q.device
        )
        for lo in range(0, q.shape[0], 8):
            hi = min(lo + 8, q.shape[0])
            r = rows[lo:hi]
            valid = r >= 0
            k = kv[r.clamp(min=0)].view(torch.float8_e4m3fn).float().double()
            s = torch.einsum("thd,tkd->thk", q[lo:hi], k) * scale
            s = s.masked_fill(~valid[:, None, :], float("-inf"))
            p = torch.softmax(s, dim=-1).nan_to_num(0.0)
            out[lo:hi] = torch.einsum("thk,tkd->thd", p, k[..., :KV_LORA_RANK])
        return (out * out_scale).to(torch.bfloat16).reshape(output.shape)

    def is_valid(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        sparse_attn_indices: torch.Tensor | None,
        output: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        compiled: CompiledSparseMlaPrefill | PreparedDenseCausal,
        block_table: torch.Tensor | None = None,
        req_idx: torch.Tensor | None = None,
        stride_factor: int | None = None,
        layer_offset: int = 0,
        dense_causal_groups: torch.Tensor | None = None,
    ) -> None:
        """The inputs the kernel takes and answers wrongly. Reads every index
        back to the host, which is why it is never on the serving path."""
        # The kernel loads only the base-2 scale `[1]`. A `[0]` that disagrees
        # describes a softmax it does not compute.
        s0, s1 = (float(x) for x in mla_bmm1_scale.tolist())
        assert math.isclose(s1, s0 * math.log2(math.e), rel_tol=1e-6), (
            f"mla_bmm1_scale[1] ({s1}) must be [0] * log2(e) ({s0 * math.log2(math.e)}); "
            "the kernel applies the base-2 scale [1] and never reads [0]"
        )
        handle = compiled.compiled if isinstance(compiled, PreparedDenseCausal) else compiled
        if handle.dense_causal:
            _check_dense_causal(
                quant_q_buffer, kv_pool_view, sparse_attn_indices, output, cu_seqlen_ks,
                cu_seqlen_ke, mla_bmm1_scale, mla_bmm2_scale, compiled, block_table, req_idx,
                stride_factor, layer_offset, dense_causal_groups,
            )  # fmt: skip
            return
        total_q = int(quant_q_buffer.shape[0])
        # The kernel does not look at which entries are -1: it reads the first
        # `clamp(ke - ks, 0, 2048)` entries of row `q` as rows and masks the
        # rest. A -1 inside that prefix is gathered (from a substitute row); a
        # row after it is dropped. Both return a plausible softmax.
        window = cu_seqlen_ke[:total_q].cpu().long() - cu_seqlen_ks[:total_q].cpu().long()
        valid = window.clamp(0, TOP_K)
        rows = sparse_attn_indices.to(torch.int64).cpu()
        prefix = torch.arange(TOP_K)[None, :] < valid[:, None]
        pool_rows = kv_pool_view.numel() * kv_pool_view.element_size() // LATENT_DIM
        assert bool(((rows >= 0) | ~prefix).all()), (
            "a -1 inside a row's valid prefix (the first clamp(ke - ks, 0, 2048) entries); "
            "the kernel gathers it as a row"
        )
        assert bool(((rows < pool_rows) | ~prefix).all()), (
            f"an index in a valid prefix is past the pool's {pool_rows} rows"
        )
        assert bool(((rows == -1) | prefix).all()), (
            "a row after a query's valid prefix, in the -1 tail; the kernel masks it, so "
            "the key window and the selection disagree"
        )
        # The all-valid build drops the tail mask and reads all 2048 entries.
        if handle.all_sparse_indices_valid:
            assert bool((valid == TOP_K).all()), (
                "the all_sparse_indices_valid build reads 2048 entries of every row; a "
                "row with a shorter window attends to its -1 tail"
            )


cute_dsl_sparse_mla_prefill = CuteDslSparseMlaPrefill()
