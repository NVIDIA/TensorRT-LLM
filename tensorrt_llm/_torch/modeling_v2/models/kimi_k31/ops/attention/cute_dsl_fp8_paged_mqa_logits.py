# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode / verify logits over the paged FP8 index-key cache (CuTe DSL), into a caller buffer.

The registered op ``torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits`` returns a view of a shared arena buffer
that its next call overwrites. This entry launches the same compiled kernel (the op's own runner and kernel
cache) into an ``out`` the caller owns, so the output buffer is an explicit argument and the call allocates
nothing.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.mqa_logits_util import (
    is_uneven_split,
    legal_atoms,
    validate_atom_split,
)
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as _dsl_ops
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Keys per split of the kernel (two math warpgroups of 128 keys): the logits row stride is a multiple of it.
SPLIT_KV = 256
#: The kernel's key tile; the page size is separate (``kv_cache.shape[1]``).
_COMPUTE_BLOCK_KV = 128
#: The widest MMA N tile (``atom * num_heads``) the kernel builds.
_MAX_N_TILE = 256


def logits_row_stride(max_context_len: int) -> int:
    """Row stride of the logits buffer the kernel writes: ``max_context_len`` rounded up to whole splits."""
    return -(-max_context_len // SPLIT_KV) * SPLIT_KV


def _spec(batch, next_n, context_lens, **overrides) -> dict:
    """``context_lens``: (low, high) of the per-request key counts, drawn per cell."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        num_next_n_atoms=1,
        atom=None,
        num_heads=32,
        head_dim=128,
        tokens_per_block=64,
        context_lens=tuple(context_lens),
        max_context_len=32768,
        page_stride_pad=0,
        data="int",
    )
    spec.update(overrides)
    return spec


class CuteDslFp8PagedMqaLogits(OpWrapper):
    """``out[b * n + t, j] = k_scale[j] * sum_h weights[b * n + t, h] * relu(q[b, t, h] . k[j])``
    for the keys ``j <= context_lens[b] - n + t`` of request ``b`` (``n`` = ``q.shape[1]``).

    Key ``j`` of request ``b`` is token ``j % P`` of page ``block_table[b, j // P]`` of ``kv_cache``
    ([pages, P, 1, 132] uint8: per page P rows of 128 E4M3 bytes, then P fp32 scales). ``q`` E4M3
    [B, n, 32, 128]; ``weights`` fp32 [B * n, 32]; ``context_lens`` int32 [B]; ``schedule_meta`` the
    DeepGEMM schedule (``get_paged_mqa_logits_metadata`` with block_kv 64) over the ``B *
    num_next_n_atoms`` atom rows, each with its own causal key count. With ``num_next_n_atoms > 1`` the
    kernel splits each request's ``n`` rows into that many atoms (``atom`` = the tile width of an uneven
    split). ``out`` is an fp32 [B * n, max_context_len] view with unit column stride; the kernel
    writes it and returns it. Columns past a row's causal end hold unspecified values; nothing past
    column max_context_len is written.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="plain decode, one request at the top-k edge (2049 keys)",
            spec=_spec(1, 1, (2049, 2049)),
        ),
        Cell(
            why="plain decode, one request below the top-k (300 keys): fewer keys than one split",
            spec=_spec(1, 1, (300, 300)),
        ),
        Cell(
            why="plain decode, 8 requests of 1.5k-10k keys, 64-token pages",
            spec=_spec(8, 1, (1500, 10000)),
        ),
        Cell(
            why="plain decode, 32-token pages",
            spec=_spec(4, 1, (3000, 9000), tokens_per_block=32),
        ),
        Cell(
            why="plain decode, 128-token pages",
            spec=_spec(4, 1, (3000, 9000), tokens_per_block=128),
        ),
        Cell(
            why="plain decode, 64 requests: multi-wave schedule",
            spec=_spec(64, 1, (2000, 20000)),
        ),
        Cell(
            why="plain decode, 256 requests (the largest decode batch) of short contexts",
            spec=_spec(256, 1, (1, 4096), max_context_len=8192),
        ),
        Cell(
            why="DFlash2 w5 / w8 at small batch: the verify rows expanded to next_n=1 rows "
            "(B=1 x 5 rows)",
            spec=_spec(5, 1, (4096, 4096)),
        ),
        Cell(
            why="DFlash2 w8 at large batch: [B*2, 4] atom split of the 8 verify rows",
            spec=_spec(32, 4, (8000, 16000)),
        ),
        Cell(
            why="DFlash2 w5 at large batch: in-kernel uneven split, 2 atoms of tile 3",
            spec=_spec(64, 5, (12000, 16384), num_next_n_atoms=2, atom=3),
        ),
        Cell(
            why="DFlash2 w8 at large batch: in-kernel even split, 2 atoms of 4",
            spec=_spec(16, 8, (5000, 9000), num_next_n_atoms=2),
        ),
        Cell(
            why="native next_n=2 tile (w8 expanded 4 x 2 when the waves tie)",
            spec=_spec(4, 2, (3000, 5000)),
        ),
        Cell(
            why="padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(4, 1, (3000, 5000), page_stride_pad=512),
        ),
        Cell(
            why="real-valued UE8M0-scaled keys and weights: accumulation order visible",
            spec=_spec(4, 1, (3000, 6000), data="randn"),
        ),
        Cell(
            why="real-valued data on a DFlash2 w5 uneven split",
            spec=_spec(8, 5, (3000, 6000), num_next_n_atoms=2, atom=3, data="randn"),
        ),
        Cell(
            why="native next_n=8 tile (N 256, one UMMA stage): the w8 verify rows read the keys once",
            spec=_spec(4, 8, (8000, 16000)),
        ),
        Cell(
            why="native next_n=7 tile (N 224)",
            spec=_spec(8, 7, (5000, 9000)),
        ),
        Cell(
            why="native next_n=6 tile (N 192), 12 requests",
            spec=_spec(12, 6, (5000, 9000)),
        ),
        Cell(
            why="real-valued data on a native next_n=8 tile",
            spec=_spec(4, 8, (3000, 6000), data="randn"),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this from the DSA indexer for every generation row (decode and
    the DFlash2 verify): the default path (use_cute_dsl_paged_mqa_logits, default True) on the SM100
    family. The indexer metadata picks the presentation of the verify rows: native n <= 4, a
    caller-side reshape [B * f, n / f] with the expanded block table and key counts, or the
    in-kernel split (num_next_n_atoms, atom) with a schedule over the staggered atom rows. The kernel
    also builds native tiles up to n = 8 (N = 32 n <= 256, one UMMA stage; the n = 6 / 7 / 8 cells):
    they read each request's keys once, and at the K3.1 bar shapes (6-8 verify rows, 130k-190k keys)
    they beat the 2-atom presentations on the same bits. 32 index heads x 128, not split over
    attention TP; max_context_len is the indexer's max sequence length.

    `out` is caller-owned and persistent: a target allocates one fp32 buffer per engine (stable
    across CUDA-graph replays) and passes its [:B * n, :max_context_len] view; the registered op
    used a shared arena buffer with row stride logits_row_stride(max_context_len) for the same
    purpose. A row stride of exactly max_context_len (10000, not a multiple of 256) was measured
    too: the causal logits are unchanged and the slack columns of a padded buffer stay untouched,
    so the rounding is not needed. The kernel writes whole 256-key splits with disjoint CTA
    stores and no atomics: deterministic, and bit-identical to the registered op on the same inputs. Compiles one kernel
    per (page size, heads, n, atom split, SM count) on first use, through the registered op's
    runner and kernel cache.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        out: torch.Tensor,
    ) -> torch.Tensor:
        runner = _dsl_ops.CuteDSLPagedMQALogitsRunner
        batch, next_n, num_heads, head_dim = q.shape
        atom = validate_atom_split(
            next_n, num_heads, num_next_n_atoms, atom, legal_atoms(num_heads, _MAX_N_TILE, 8)
        )
        atom_rows = batch * num_next_n_atoms
        if is_uneven_split(next_n, num_next_n_atoms, atom):
            rows = batch * next_n * num_heads
            q_3d = q.reshape(-1).as_strided((rows, head_dim, 1), (head_dim, 1, rows * head_dim))
            w_2d = weights.reshape(-1)
        else:
            n_tile = atom * num_heads
            q_3d = q.reshape(-1).as_strided(
                (n_tile, head_dim, atom_rows), (head_dim, 1, n_tile * head_dim)
            )
            w_2d = weights.reshape(-1).as_strided((n_tile, atom_rows), (1, n_tile))
        num_pages, page = kv_cache.shape[0], kv_cache.shape[1]
        num_sms = _dsl_ops._get_num_sms()
        f32 = torch.float32
        key = (_COMPUTE_BLOCK_KV, page, num_heads, head_dim, next_n, num_sms, 1, f32, f32, f32)
        key = key + (num_next_n_atoms, atom)
        if key not in runner.kernel_cache:
            runner._compile(*key)
        runner.kernel_cache[key](
            kv_cache.reshape(num_pages, -1),
            q_3d.view(torch.uint8),
            w_2d,
            out,
            block_table,
            context_lens,
            schedule_meta,
            num_pages,
            atom_rows,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        out: torch.Tensor,
    ) -> torch.Tensor:
        """fp64 evaluation over each row's causal keys, -inf elsewhere, into a new tensor.

        ``schedule_meta``, ``num_next_n_atoms`` and ``atom`` only partition the work and are ignored;
        ``out`` is left untouched.
        """
        batch, next_n, num_heads, head_dim = q.shape
        page = kv_cache.shape[1]
        logits = torch.full(
            (batch * next_n, max_context_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        for b in range(batch):
            ctx = int(context_lens[b])
            pages = block_table[b, : -(-ctx // page)].long()
            raw = kv_cache[pages].reshape(pages.shape[0], -1)
            keys = raw[:, : page * head_dim].contiguous().view(torch.float8_e4m3fn)
            scales = raw[:, page * head_dim : page * (head_dim + 4)].contiguous()
            scales = scales.view(torch.float32)
            k = keys.reshape(-1, head_dim)[:ctx].double() * scales.reshape(-1)[:ctx, None].double()
            for t in range(next_n):
                row = b * next_n + t
                end = ctx - next_n + t + 1
                if end <= 0:
                    continue
                score = (q[b, t].double() @ k[:end].t()).relu()
                logits[row, :end] = (score * weights[row].double()[:, None]).sum(0).float()
        return logits

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        out: torch.Tensor,
    ) -> None:
        # The kernel takes the page bytes as one flat row per page; a page view whose 132-byte
        # rows are not packed is read at the wrong offsets.
        assert kv_cache.stride(1) == kv_cache.shape[3] and kv_cache.stride(3) == 1, (
            "kv_cache rows within a page must be packed"
        )
        assert int(context_lens.max()) <= max_context_len, (
            "a context longer than max_context_len writes past the logits row"
        )
        # q and weights are re-strided from their base pointer: a non-packed view is read as
        # another request's rows.
        assert q.is_contiguous() and weights.is_contiguous(), "q and weights must be packed"
        assert weights.shape == (q.shape[0] * q.shape[1], q.shape[2]), (
            "weights must hold one row per query row"
        )
        # The kernel addresses rows through out's row stride and never checks the row count: a
        # buffer with fewer rows than query rows is written past its end.
        assert out.dtype == torch.float32 and out.stride(1) == 1, (
            "out must be fp32 with unit column stride"
        )
        assert out.shape[0] >= q.shape[0] * q.shape[1] and out.shape[1] == max_context_len, (
            f"out must be [>= B * n, max_context_len], got {tuple(out.shape)}"
        )


cute_dsl_fp8_paged_mqa_logits = CuteDslFp8PagedMqaLogits()


_PDL_KERNELS: dict = {}


def _compiled_pdl(page: int, num_heads: int, head_dim: int, next_n: int, num_sms: int):
    """The same kernel as the registered op's runner builds for one native atom, compiled with
    programmatic dependent launch (its TMA warps wait before the first predecessor-produced read;
    the math warps release the dependents after their last logits store)."""
    key = (page, num_heads, head_dim, next_n, num_sms)
    compiled = _PDL_KERNELS.get(key)
    if compiled is not None:
        return compiled
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.fp8_paged_mqa_logits import (
        FP8MQALogitsKernel,
    )

    n_tile = next_n * num_heads
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (cute.sym_int(), page * (head_dim + 4)), stride=(cute.sym_int64(), 1)
    )
    rows = cute.sym_int()
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n_tile, head_dim, rows), stride_order=(1, 0, 2)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (n_tile, rows), stride_order=(0, 1)
    )
    batch = cute.sym_int()
    logits_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch, cute.sym_int()), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (batch,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(), 2), stride_order=(1, 0)
    )
    kernel = FP8MQALogitsKernel(
        block_kv=_COMPUTE_BLOCK_KV,
        phys_block_kv=page,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_next_n_atoms=1,
        atom=None,
        num_sms=num_sms,
        num_epi_subtiles=1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        pdl=True,
    )
    compiled = cute.compile(
        kernel,
        kv_fake,
        q_fake,
        w_fake,
        logits_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _PDL_KERNELS[key] = compiled
    return compiled


class CuteDslFp8PagedMqaLogitsPdl(CuteDslFp8PagedMqaLogits):
    """``cute_dsl_fp8_paged_mqa_logits`` for a native tile (``num_next_n_atoms == 1``), launched
    with programmatic dependent launch: the same kernel and the same logits, its launch and
    pipeline setup overlapping the producer of q / weights / the key cache. ``num_next_n_atoms``
    and ``atom`` must be 1 and None."""

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=c.why + " (PDL)", spec=c.spec)
        for c in CuteDslFp8PagedMqaLogits.CELLS
        if c.spec["num_next_n_atoms"] == 1 and c.spec["atom"] is None
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).
    The K3.1 perf targets call this in place of cute_dsl_fp8_paged_mqa_logits when the metadata
    picks a native tile (dsl_expand_factor 1, dsl_kernel_atoms 1; with kernel_atoms 1..8 that is
    every bar shape) and the predecessor is the fused decode prologue (which releases its
    dependents early). Same bits as the non-PDL entry; -0.7..-1.0 us per DSA layer in the
    GB300 decode chain. Compiles one kernel per (page size, heads, n, SM count) on first use,
    outside CUDA-graph capture.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        out: torch.Tensor,
    ) -> torch.Tensor:
        batch, next_n, num_heads, head_dim = q.shape
        n_tile = next_n * num_heads
        q_3d = q.reshape(-1).as_strided((n_tile, head_dim, batch), (head_dim, 1, n_tile * head_dim))
        w_2d = weights.reshape(-1).as_strided((n_tile, batch), (1, n_tile))
        num_pages, page = kv_cache.shape[0], kv_cache.shape[1]
        compiled = _compiled_pdl(page, num_heads, head_dim, next_n, _dsl_ops._get_num_sms())
        compiled(
            kv_cache.reshape(num_pages, -1),
            q_3d.view(torch.uint8),
            w_2d,
            out,
            block_table,
            context_lens,
            schedule_meta,
            num_pages,
            batch,
        )
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_next_n_atoms: int,
        atom: Optional[int],
        out: torch.Tensor,
    ) -> None:
        assert num_next_n_atoms == 1 and atom is None, "native tiles only"
        assert q.shape[1] * q.shape[2] <= _MAX_N_TILE, "the N tile is at most 256"
        super().is_valid(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            num_next_n_atoms,
            atom,
            out,
        )


cute_dsl_fp8_paged_mqa_logits_pdl = CuteDslFp8PagedMqaLogitsPdl()
