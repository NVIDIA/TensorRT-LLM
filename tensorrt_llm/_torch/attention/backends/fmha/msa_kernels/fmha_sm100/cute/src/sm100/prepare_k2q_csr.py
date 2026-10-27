# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Sparse k2q CSR builder for SM100.

Thin dispatcher that calls the CUDA C++ kernel pipeline in
``src.sm100.build_k2q_csr``. Supports ``topK in {4, 8, 16, 32}`` and
``blk_kv == 128`` only — other shapes raise ``ValueError`` rather than
silently falling back to a torch-reference path.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.sm100.prepare_scheduler import SparseAttentionSchedule, SPARSE_SCHEDULE_MODEL


_SUPPORTED_TOPK = (4, 8, 16, 32)
_SUPPORTED_BLK_KV = 128


_SCHEDULE_ORDERS = ("stock", "lpt")
_LPT_LOGGED = False


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the torch path below needs no Triton
    triton = None

_LPT_MAX_ROWS = 16384  # one Triton program sorts up to this many scheduler rows; more falls back to the torch sort

if triton is not None:

    @triton.jit
    def _lpt_sort_rows_kernel(md_ptr, out_ptr, wc_ptr, n_rows, qcap, s_c, s_h, s_r, s_q, BLOCK: tl.constexpr):
        """One program: packed key (q_count desc | head | row | q_begin | row index), sort, gather the 6 columns of
        each row's source into out (out-of-place: no load / store race). Keys are unique (row index in the low bits)."""
        offs = tl.arange(0, BLOCK)
        m = offs < n_rows
        live = offs < tl.load(wc_ptr)
        base = md_ptr + offs * 6
        hd = tl.load(base + 0, mask=m, other=0).to(tl.int64)
        rw = tl.load(base + 1, mask=m, other=0).to(tl.int64)
        qb = tl.load(base + 2, mask=m, other=0).to(tl.int64)
        qc = tl.load(base + 3, mask=m, other=0).to(tl.int64)
        o64 = offs.to(tl.int64)
        key = ((qcap - qc) << s_c) | (hd << s_h) | (rw << s_r) | (qb << s_q) | o64
        key = tl.where(live, key, (tl.full([BLOCK], 1, tl.int64) << 62) | o64)
        key = tl.where(m, key, (tl.full([BLOCK], 1, tl.int64) << 62) | (tl.full([BLOCK], 1, tl.int64) << 61) | o64)
        key = tl.sort(key)
        src = (key & ((1 << s_q) - 1)).to(tl.int32)
        src_base = md_ptr + src * 6
        c0 = tl.load(src_base + 0, mask=m)
        c1 = tl.load(src_base + 1, mask=m)
        c2 = tl.load(src_base + 2, mask=m)
        c3 = tl.load(src_base + 3, mask=m)
        c4 = tl.load(src_base + 4, mask=m)
        c5 = tl.load(src_base + 5, mask=m)
        obase = out_ptr + offs * 6
        tl.store(obase + 0, c0, mask=m)
        tl.store(obase + 1, c1, mask=m)
        tl.store(obase + 2, c2, mask=m)
        tl.store(obase + 3, c3, mask=m)
        tl.store(obase + 4, c4, mask=m)
        tl.store(obase + 5, c5, mask=m)


def _lpt_ext():
    """The build_k2q_csr CUDA extension when it carries run_lpt_reorder (else None: Triton / torch fallbacks)."""
    try:
        from src.sm100.build_k2q_csr import _ext
    except Exception:  # pragma: no cover
        return None
    return _ext if hasattr(_ext, "run_lpt_reorder") else None


def _lpt_reorder(schedule, *, q_cap: int, q_bound: int, total_rows: int, head_kv: int) -> None:
    """Heaviest-work-item-first (LPT) issue order for the sparse forward schedule, on device (no host sync).

    The forward launches one CTA per scheduler_metadata row in row order; a heavy (up to target_q_per_cta queries)
    row issued late is the grid's tail. Live rows (row < work_count, a device tensor) are re-sorted by q_count
    (column 3) descending, ties broken by the row's own content -- head_kv (0), row_linear (1), q_begin (2), all
    ascending -- so the order is a function of the schedule's rows, not of the order the builder emitted them in
    (deterministic across processes and graph replays). Unused capacity rows sort last. It runs AFTER the split-slot
    pass has assigned every partial's slot, so it changes when an item runs, never which slot it writes: no output bit
    changes. Preferred: the build_k2q_csr extension's run_lpt_reorder (one CUDA launch from C++, no Python ops);
    else one Triton launch + one copy; else the torch sort below.
    Capturable: no .item().
    """
    global _LPT_LOGGED
    md = schedule.scheduler_metadata
    n = md.shape[0]
    bq = max(int(q_bound), 1).bit_length()           # q_begin < the k2q list length bound (nnz_upper_bound)
    br = max(int(total_rows), 1).bit_length()        # row_linear < total_rows
    bh = max(int(head_kv), 1).bit_length()           # head_kv < head_kv
    bc = max(int(q_cap), 1).bit_length() + 1         # q_count <= q_cap
    block = 1 << max(n - 1, 1).bit_length()
    bi = block.bit_length()                          # row index < block
    fits = bc + bh + br + bq + bi <= 61
    s_q = bi
    s_r = s_q + bq
    s_h = s_r + br
    s_c = s_h + bh
    ext = _lpt_ext()
    if fits and ext is not None and md.is_cuda and md.is_contiguous() and ext.run_lpt_reorder(
        md, schedule.work_count, (1 << bc) - 1, s_c, s_h, s_r, s_q
    ):
        pass  # one launch from C++ (+ a D2D copy): the in-builder path
    elif triton is not None and md.is_cuda and md.is_contiguous() and block <= _LPT_MAX_ROWS and fits:
        s_q = bi
        s_r = s_q + bq
        s_h = s_r + br
        s_c = s_h + bh
        out = torch.empty_like(md)
        _lpt_sort_rows_kernel[(1,)](md, out, schedule.work_count, n, (1 << bc) - 1, s_c, s_h, s_r, s_q,
                                    BLOCK=block, num_warps=16 if block >= 8192 else 8)
        md.copy_(out)
    else:
        if bq + br + bh + bc > 62:
            raise ValueError(f"LPT key needs {bq + br + bh + bc} bits (> 62): q {q_bound}, rows {total_rows}, heads {head_kv}")
        live = torch.arange(n, device=md.device, dtype=torch.int32) < schedule.work_count
        m = md.to(torch.int64)
        key = ((((1 << bc) - 1 - m[:, 3]) << (bh + br + bq)) | (m[:, 0] << (br + bq)) | (m[:, 1] << bq) | m[:, 2])
        key = torch.where(live, key, torch.full_like(key, (1 << 62)))
        md.copy_(md.index_select(0, torch.sort(key).indices))
    if not _LPT_LOGGED:
        _LPT_LOGGED = True
        print("[fmha_sm100] build_k2q_csr: LPT schedule order (heaviest work item first) in effect", flush=True)


_K2Q_FUSE_LOGGED = False


def _k2q_fused_build(ext, args, *, schedule, q_cap: int, q_bound: int, total_rows: int, head_kv: int) -> bool:
    """MiniMax-M3.1 K2Q_FUSE: the build and the LPT order in one C++ call (1 memset + 3 launches, bitwise the
    run_build_k2q_csr_with_schedule + run_lpt_reorder outputs). False (nothing launched) when the extension lacks the
    entry or the call is outside its shapes; the caller then runs the two-call path. Kill:
    TRTLLM_M31_DISABLE_TP2CTX_K2Q_FUSE=1 (read by the extension)."""
    global _K2Q_FUSE_LOGGED
    fused = getattr(ext, "run_build_k2q_csr_lpt_fused", None) if ext is not None else None
    if fused is None:
        return False
    n = schedule.scheduler_metadata.shape[0]
    bq = max(int(q_bound), 1).bit_length()
    br = max(int(total_rows), 1).bit_length()
    bh = max(int(head_kv), 1).bit_length()
    bc = max(int(q_cap), 1).bit_length() + 1
    bi = (1 << max(n - 1, 1).bit_length()).bit_length()
    if bc + bh + br + bq + bi > 61:
        return False
    s_q = bi
    s_r = s_q + bq
    s_h = s_r + br
    s_c = s_h + bh
    if not fused(*args, (1 << bc) - 1, s_c, s_h, s_r, s_q):
        return False
    if not _K2Q_FUSE_LOGGED:
        _K2Q_FUSE_LOGGED = True
        print("[fmha_sm100] build_k2q_csr: K2Q_FUSE taken (build + LPT order: 1 memset + 3 launches)", flush=True)
    return True


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


class SparseK2qCsrBuilderSm100:
    """Build the k2q CSR reverse index for sparse attention on SM100.

    The public API matches the historical CUTE DSL builder so callers
    (``sparse_index_utils.build_k2q_csr``, attention kernels) need no
    changes. Internally the kernel pipeline runs five CUDA C++ kernels:
    ``build_row_map`` -> ``hist`` -> ``row_prefix`` -> ``tile_prefix_smem``
    -> ``scatter`` (5 kernels + 2 ``cudaMemsetAsync``).
    """

    def __init__(self) -> None:
        # No persistent state — the JIT-compiled extension is loaded
        # lazily by ``src.sm100.build_k2q_csr`` on first call.
        self._run = None
        self._run_with_schedule = None

    def _ensure_loaded(self) -> None:
        if self._run is None:
            from src.sm100.build_k2q_csr import (
                run_build_k2q_csr,
                run_build_k2q_csr_with_schedule,
            )
            self._run = run_build_k2q_csr
            self._run_with_schedule = run_build_k2q_csr_with_schedule

    def __call__(
        self,
        q2k_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        total_k: int,
        blk_kv: int = 128,
        max_seqlen_k: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        total_rows: Optional[int] = None,
        qhead_per_kv: int = 1,
        return_schedule: bool = False,
        schedule_order: str = "stock",
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, SparseAttentionSchedule]:
        # schedule_order (opt-in, default "stock" = the built order): "lpt" issues the heaviest work items first
        # (_lpt_reorder); it needs return_schedule.
        if schedule_order not in _SCHEDULE_ORDERS or (schedule_order != "stock" and not return_schedule):
            raise ValueError(f"schedule_order must be one of {_SCHEDULE_ORDERS} ('lpt' needs return_schedule)")
        # ---- Validation ----------------------------------------------------
        if blk_kv != _SUPPORTED_BLK_KV:
            raise ValueError(
                f"SparseK2qCsrBuilderSm100 only supports blk_kv == "
                f"{_SUPPORTED_BLK_KV}, got {blk_kv}"
            )
        if q2k_indices.dtype != torch.int32:
            raise TypeError(
                f"q2k_indices must be torch.int32, got {q2k_indices.dtype}"
            )
        if q2k_indices.ndim != 3:
            raise ValueError(
                f"q2k_indices must be rank-3 [head_kv, total_q, topK], "
                f"got shape {tuple(q2k_indices.shape)}"
            )
        if not q2k_indices.is_contiguous():
            raise ValueError("q2k_indices must be contiguous")
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens_q and cu_seqlens_k must be torch.int32")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must share shape [B + 1]"
            )
        if not (q2k_indices.is_cuda and cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
            raise ValueError("all inputs must be CUDA tensors")
        if (
            q2k_indices.device != cu_seqlens_q.device
            or q2k_indices.device != cu_seqlens_k.device
        ):
            raise ValueError("all inputs must share a device")
        if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous")

        total_k = int(total_k)
        if total_k < 0:
            raise ValueError(f"total_k must be non-negative, got {total_k}")

        head_kv, total_q, topk = (int(v) for v in q2k_indices.shape)
        if topk not in _SUPPORTED_TOPK:
            raise ValueError(
                f"SparseK2qCsrBuilderSm100 only supports topK in "
                f"{_SUPPORTED_TOPK}, got {topk}"
            )

        batch = int(cu_seqlens_q.shape[0] - 1)
        if batch < 0:
            raise ValueError("cu_seqlens tensors must have shape [B + 1]")
        if return_schedule and max_seqlen_k is None:
            raise ValueError("build_k2q_csr requires max_seqlen_k when return_schedule=True")
        max_k_tokens = int(max_seqlen_k) if max_seqlen_k is not None else total_k
        max_kv_blocks = _ceil_div(max(max_k_tokens, blk_kv), blk_kv)
        if total_rows is not None:
            total_rows = int(total_rows)
        elif total_k % blk_kv == 0:
            total_rows = total_k // blk_kv
        else:
            total_rows = _ceil_div(total_k + batch * (blk_kv - 1), blk_kv)
        if total_rows < 0:
            raise ValueError(f"total_rows must be non-negative, got {total_rows}")
        total_rows = max(total_rows, 0)
        nnz_upper_bound = total_q * topk
        qhead_per_kv = int(qhead_per_kv)
        if qhead_per_kv <= 0:
            raise ValueError(f"qhead_per_kv must be positive, got {qhead_per_kv}")
        if return_schedule:
            if max_seqlen_q is None:
                raise ValueError("build_k2q_csr requires max_seqlen_q when return_schedule=True")
            max_seqlen_q = int(max_seqlen_q)

        # ---- Output tensors ------------------------------------------------
        device = q2k_indices.device
        k2q_row_ptr = torch.empty(
            (head_kv, total_rows + 1), dtype=torch.int32, device=device,
        )
        k2q_q_indices = torch.empty(
            (head_kv, nnz_upper_bound), dtype=torch.int32, device=device,
        )
        schedule = None
        if return_schedule:
            target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
                total_q=total_q,
                topk=topk,
                blk_kv=blk_kv,
                head_kv=head_kv,
                qhead_per_kv=qhead_per_kv,
                device=device,
            )
            scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
                total_rows=total_rows,
                total_q=total_q,
                topk=topk,
                head_kv=head_kv,
                target_q_per_cta=target_q_per_cta,
            )
            scheduler_metadata = torch.empty(
                (scheduler_metadata_capacity, 6), dtype=torch.int32, device=device
            )
            work_count = torch.empty((1,), dtype=torch.int32, device=device)
            qsplit_indices = torch.empty_like(k2q_q_indices)
            split_counts = torch.empty(
                (total_q, head_kv), dtype=torch.int32, device=device
            )
            schedule = SparseAttentionSchedule(
                enabled=True,
                scheduler_metadata=scheduler_metadata,
                work_count=work_count,
                qsplit_indices=qsplit_indices,
                split_counts=split_counts,
                target_q_per_cta=target_q_per_cta,
            )

        # Empty workload short-circuit (the CUDA path also handles this,
        # but doing it here saves a JIT load for trivial calls).
        if total_rows == 0 or total_q == 0 or head_kv == 0 or topk == 0:
            k2q_row_ptr.zero_()
            k2q_q_indices.fill_(-1)
            if schedule is not None:
                schedule.work_count.zero_()
                schedule.split_counts.zero_()
                return k2q_row_ptr, k2q_q_indices, schedule
            return k2q_row_ptr, k2q_q_indices

        self._ensure_loaded()
        with torch.cuda.nvtx.range("SparseK2qCsr_Pipeline"):
            if schedule is None:
                self._run(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                )
            else:
                args = (
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.qsplit_indices,
                    schedule.split_counts,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    schedule.target_q_per_cta,
                    schedule.work_capacity,
                    max_seqlen_q,
                )
                if schedule_order == "lpt" and _k2q_fused_build(
                    _lpt_ext(), args, schedule=schedule, q_cap=schedule.target_q_per_cta, q_bound=nnz_upper_bound,
                    total_rows=total_rows, head_kv=head_kv,
                ):
                    return k2q_row_ptr, k2q_q_indices, schedule
                self._run_with_schedule(
                    q2k_indices,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    k2q_row_ptr,
                    k2q_q_indices,
                    schedule.scheduler_metadata,
                    schedule.work_count,
                    schedule.qsplit_indices,
                    schedule.split_counts,
                    topk,
                    blk_kv,
                    total_rows,
                    max_kv_blocks,
                    schedule.target_q_per_cta,
                    schedule.work_capacity,
                    max_seqlen_q,
                )
                if schedule_order == "lpt":
                    _lpt_reorder(schedule, q_cap=schedule.target_q_per_cta, q_bound=nnz_upper_bound,
                                 total_rows=total_rows, head_kv=head_kv)
        if schedule is not None:
            return k2q_row_ptr, k2q_q_indices, schedule
        return k2q_row_ptr, k2q_q_indices
