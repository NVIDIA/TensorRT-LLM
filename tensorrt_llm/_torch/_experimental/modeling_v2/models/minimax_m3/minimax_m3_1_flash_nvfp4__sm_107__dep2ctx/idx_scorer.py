# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_IDX_SCORER (CTX, exact, default on) on the sm_107 dep2ctx target: the prefill indexer scores from the Q8KV8
prefill scorer (MiniMax msa_dev 25c1447, vendored by MSA) at the generic F8F6F4 MMA (K = 32), in place of the
fmha_sm100 OnlyScore proxy pass and its single-CTA plan.

The scorer's raw page maxima equal the proxy's on every historical page bit for bit, and the top-16 tables
the select builds from them are identical: it never writes a row's local page, which the select forces anyway
(register() requires local_blocks >= 1).

Registered on every layer's MSA indexer ``prefill_scorer_hook`` when the MSA backend exposes one (it calls the hook on
pure-prefill steps ahead of any read of the step's proxy plan); a backend without the hook keeps the proxy pass and
register() registers nothing. ``accepts`` tells whether a step's host lengths admit the scorer and ``prewarm``
compiles or loads the scorer's kernels before the server is ready. Per step:

- the lengths are read once (the same host length objects, which prepare() stages anew every step, skip the
  ``.tolist()`` and the eligibility checks on the later layers);
- the plan reads only the lengths and is kept for the next step with equal lengths; its device metadata is staged
  through pinned memory without a host sync;
- the block table is gathered from the page table once per step (the same page table object with the same length
  objects: within a step prepare() is the only writer of either) and again on every new step;
- the scores go to the select as a zero-copy view of the scorer's row-major [T * H, pitch] buffer, [H, pages, T] with
  unit block stride, which the tile select's row-major op reads (tile_select.select_rows, registered as the indexer's
  ``prefill_scorer_selector``, so only these steps reach it; DEP2CTX_TSEL_ROWS); b22's amax / permute / contiguous
  copy (two full fp32 matrices per layer) remains only for more than one index head per KV head;
- a decline releases the kept plan (its [T * H, pitch] score buffer), so it is not alive beside the proxy's.

Declines (the proxy runs) under CUDA-graph capture, for non-FP8 operands, for a page / head size other than 128, for an
index-Q that is not contiguous or not 16-byte aligned, for rows whose causal offset is not bottom-right aligned, past
8192 pages per row, and when the scorer is not the exact K = 32 build. Kill switch:
``TRTLLM_M31_DISABLE_DEP2CTX_IDX_SCORER=1`` (``TRTLLM_M31_DISABLE_DEP2CTX_TSEL_ROWS=1`` turns it off too).
Registration: ``IDX_SCORER_REGISTERED:`` (vs
``IDX_SCORER_OFF:``); engagement (a step taken): ``IDX_SCORER_ON:`` and the family's ``MSA_PREFILL_SCORER taken``.

DEP2CTX_IDX_SCORER_K64 (not bitwise, off by default): ``TRTLLM_M31_ENABLE_DEP2CTX_IDX_SCORER_K64=1`` also accepts the
scorer built with the sm_107 FP8 MMA at K = 64 (``TRTLLM_M31_IDX_SCORER_MMA_K=64``, part of its compile key): page
maxima may differ by 1-3 ulp, so a top-16 table may differ.
"""

import functools
import importlib
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels.msa_utils import (
    install_msa_cutlass_compatibility,
)
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

from . import tile_select
from .m31_switches import m31_disabled, m31_enabled

IDX_SCORER = not m31_disabled("DEP2CTX_IDX_SCORER")
# Not bitwise: an opt-in (see the header).
IDX_SCORER_K64 = m31_enabled("DEP2CTX_IDX_SCORER_K64")
# The vendored scorer's interface module (MSA names the path).
SCORER_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.msa_kernels."
    "fmha_sm100_q8kv8.q8_indexer_interface"
)
PAGE = 128  # capacity: fixed: the index-K cache page size the scorer is built for
HEAD_DIM = 128  # capacity: fixed: the index head size the scorer is built for
MAX_PAGES = 8192  # capacity: fixed: the scorer's block-table width bound (it raises past it)
Q_ALIGN = 16  # capacity: fixed: the scorer's TMA alignment of index-Q (it raises below it)

_STATE: dict = {}


def _log_once(key: str, msg: str) -> None:
    if key not in _STATE:
        _STATE[key] = True
        logger.info(msg)


def _release_plan() -> None:
    """Drop the kept plan (its [T * H, pitch] score buffer and block table); the next taken step plans anew."""
    _STATE["wrapper"] = _STATE["key"] = _STATE["gather"] = _STATE["table_src"] = None


def _decline(reason: str) -> None:
    """Release the kept plan (the proxy scores this step, with its own buffer) and report the reason once per process
    (the indexer counts every decline)."""
    _release_plan()
    key = f"declined_{reason}"
    if key not in _STATE:
        _STATE[key] = True
        logger.warning(f"[M3.1 dep2ctx] IDX_SCORER declined: {reason}; the proxy scores this step")


def _module():
    """The vendored interface module when it is importable and exact (K = 32 on sm_107; K = 64 with the opt-in), else
    None. Decided once."""
    if "module" in _STATE:
        return _STATE["module"]
    mod = None
    try:
        # The scorer is a CuTe DSL kernel: the image's cutlass DSL needs the MSA compatibility shim first.
        install_msa_cutlass_compatibility()
        mod = importlib.import_module(
            "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.msa_kernels.fmha_sm100_q8kv8.q8_indexer_interface"
        )
        prefill = mod.Q8KV8PrefillIndexerSm100
        mma_k = prefill.mma_inst_k((10, 7))
        exact = not prefill.uses_sm107_fp8_mma((10, 7)) and mma_k == 32
        _STATE["mma_k"] = mma_k
        if not exact and not (IDX_SCORER_K64 and mma_k == 64):
            _log_once(
                "not_exact",
                f"[M3.1 dep2ctx] IDX_SCORER_OFF: the scorer is not the exact K=32 build (K={mma_k}; K=64 needs "
                "TRTLLM_M31_ENABLE_DEP2CTX_IDX_SCORER_K64=1) (proxy kept)",
            )
            mod = None
        elif not exact:
            _log_once(
                "k64",
                "[M3.1 dep2ctx] IDX_SCORER K=64 (DEP2CTX_IDX_SCORER_K64, NOT bitwise: page maxima may differ by "
                "1-3 ulp)",
            )
    except (ImportError, AttributeError) as err:
        _log_once(
            "no_module", f"[M3.1 dep2ctx] IDX_SCORER_OFF: {SCORER_MODULE} unavailable ({err})"
        )
        mod = None
    _STATE["module"] = mod
    return mod


def _to_device(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``host`` on ``device`` without a host sync: a pageable copy waits for the stream to drain, which costs the
    host its run-ahead (a re-plan would sync several times per layer). Pinned and
    non-blocking (maybe_pin_memory: the repo's pinning policy); the caching host allocator keeps the pinned block
    until the copy has run."""
    if device.type != "cuda":
        return host.to(device)
    return maybe_pin_memory(host).to(device, non_blocking=True)


def _step_lengths(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu):
    """(qo list, kv list, plan key, decline reason or None) for this step's host lengths, computed on the first layer
    and kept for the same three objects (prepare() stages new ones every step; held here, so their ids are not reused
    while cached)."""
    cached = _STATE.get("lens")
    if (
        cached is not None
        and cached[0] is qo_lens_cpu
        and cached[1] is kv_lens_cpu
        and cached[2] is qo_offset_cpu
    ):
        return cached[3]
    qo = [int(x) for x in qo_lens_cpu.tolist()]
    kv = [int(x) for x in kv_lens_cpu.tolist()]
    reason = None
    if not qo or min(qo) <= 0:
        reason = "an empty row"
    elif qo_offset_cpu is not None and [int(x) for x in qo_offset_cpu.tolist()] != [
        k - q for k, q in zip(kv, qo)
    ]:
        reason = "rows not aligned bottom-right"  # the scorer aligns queries at kv_len - qo_len + i
    elif max(-(-k // PAGE) for k in kv) > MAX_PAGES:
        reason = f"more than {MAX_PAGES} pages in a row"
    info = (qo, kv, (tuple(qo), tuple(kv)), reason)
    _STATE["lens"] = (qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, info)
    return info


def _wrapper(mod, num_heads: int, key: tuple, qo, kv, kv_indices: torch.Tensor, step):
    """The planned wrapper for these lengths (the plan reads only the lengths; reused for the same ``key``), with its
    block table gathered from ``kv_indices`` once per step (``step`` is the step's length-objects tuple): the scorer
    kernel reads it at run time."""
    w = _STATE.get("wrapper")
    if w is not None and _STATE.get("key") == key:
        src = _STATE.get("table_src")
        if src is not None and src[0] is kv_indices and src[1] is step:
            return w  # same step, same page table object: its pages are already gathered
        gather, valid, block_table = _STATE["gather"]
        block_table.copy_(
            torch.where(
                valid,
                kv_indices[gather],
                torch.zeros((), dtype=kv_indices.dtype, device=valid.device),
            )
        )
        _STATE["table_src"] = (kv_indices, step)
        return w
    # Free the old plan's [T * H, pitch] score buffer before the new plan allocates its own (peak memory): drop the
    # local reference too.
    w = None
    _release_plan()
    device = kv_indices.device
    pages = [-(-k // PAGE) for k in kv]
    max_pages = max(pages)
    starts, acc = [], 0
    for n in pages:
        starts.append(acc)
        acc += n
    cols = torch.arange(max_pages)
    gather = torch.tensor(starts)[:, None] + cols[None, :]
    valid = cols[None, :] < torch.tensor(pages)[:, None]
    gather = _to_device(torch.where(valid, gather, torch.zeros_like(gather)), device)
    valid = _to_device(valid, device)
    # A normal tensor, so the in-place re-gather works in and out of inference mode.
    with torch.inference_mode(False):
        block_table = torch.empty(gather.shape, dtype=torch.int32, device=device)
    block_table.copy_(
        torch.where(
            valid, kv_indices[gather], torch.zeros((), dtype=kv_indices.dtype, device=device)
        )
    )
    cu_q = _to_device(
        torch.tensor([0] + list(torch.tensor(qo).cumsum(0).tolist()), dtype=torch.int32), device
    )
    seq_lens = _to_device(torch.tensor(kv, dtype=torch.int32), device)
    w = mod.BatchPrefillIndexerQ8KV8Wrapper(num_heads=num_heads)
    w.plan(
        cu_q,
        seq_lens,
        block_table,
        total_q=int(sum(qo)),
        max_seqlen_q=max(qo),
        max_seqlen_k=max(kv),
    )
    _STATE["wrapper"], _STATE["key"], _STATE["gather"] = w, key, (gather, valid, block_table)
    _STATE["table_src"] = (kv_indices, step)
    return w


def score(
    idx_q,
    idx_k_paged,
    qo_lens_cpu,
    kv_lens_cpu,
    qo_offset_cpu,
    kv_indices,
    *,
    num_kv_heads: int = 1,
) -> Optional[torch.Tensor]:
    """The per-KV-head prefill block scores [num_kv_heads, n_blocks, total_q] fp32 (the amax over each KV head's
    contiguous group of this rank's index heads, as _group_max_reduce), or None to keep the proxy. With one index head
    per KV head (dep2ctx: 4 and 4 per rank) a zero-copy view of the scorer's buffer: strides (pitch, 1, H * pitch)."""
    if torch.cuda.is_current_stream_capturing():
        return None
    if num_kv_heads <= 0 or idx_q.dim() != 3 or idx_q.shape[1] % num_kv_heads:
        return None
    if (
        idx_q.dtype != torch.float8_e4m3fn
        or idx_k_paged.dtype != torch.float8_e4m3fn
        or idx_q.shape[-1] != HEAD_DIM
        or idx_k_paged.dim() != 4
        or idx_k_paged.shape[1] != 1
        or idx_k_paged.shape[2] != PAGE
        or idx_k_paged.shape[3] != HEAD_DIM
    ):
        return None
    if not idx_q.is_contiguous() or idx_q.data_ptr() % Q_ALIGN:
        _decline(
            "index-Q not contiguous or not 16-byte aligned"
        )  # the scorer would raise, not decline
        return None
    step = _step_lengths(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu)
    qo, kv, lens_key, reason = step
    if reason is None and sum(qo) != idx_q.shape[0]:
        reason = "the lengths do not cover index-Q"
    if reason is not None:
        _decline(reason)
        return None
    mod = _module()
    if mod is None:
        return None
    num_heads = int(idx_q.shape[1])
    w = _wrapper(mod, num_heads, (*lens_key, num_heads), qo, kv, kv_indices, step)
    scores = w.run_scores(
        idx_q, idx_k_paged.view(idx_k_paged.shape[0], PAGE, HEAD_DIM)
    )  # [T * H, pages], row t * H + h
    total_q = idx_q.shape[0]
    _log_once(
        f"taken_{num_heads}",
        f"[M3.1 dep2ctx] IDX_SCORER_ON: "
        f"{'exact K=32' if _STATE.get('mma_k', 32) == 32 else 'K=64 (NOT bitwise)'} prefill scorer "
        f"({total_q} rows x {num_heads} index heads)",
    )
    # Index head h belongs to KV head h // group (contiguous groups, as _group_max_reduce): dep2ctx has 4 index heads
    # and 4 KV heads per rank (group 1), so an amax over all the rank's heads would merge KV heads.
    group = num_heads // num_kv_heads
    if group == 1:
        # Zero copy: [T, H, pages] -> [H, pages, T], unit block stride; tile_select's row-major op reads it as is.
        return scores.view(total_q, num_heads, -1).permute(1, 2, 0)
    return scores.view(total_q, num_kv_heads, group, -1).amax(dim=2).permute(1, 2, 0).contiguous()


def accepts(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu) -> bool:
    """msa_indexer's host-length predicate for prepare(): False for a step ``score`` declines on its lengths alone
    (an empty row, rows off bottom-right, past 8192 pages). Computed once per step's length objects and kept, so the
    forward's layers reuse it."""
    return _step_lengths(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu)[3] is None


def prewarm(num_heads: int, device: Optional[torch.device] = None) -> None:
    """Compile or load the scorer's kernels (plan reset, plan build, scorer: their keys do not depend on the shapes)
    with one small step on ``device`` (default: the current CUDA device), before the server is ready. Leaves the kept
    plan alone."""
    mod = _module()
    if mod is None:
        return
    if device is None:
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
    qo, kv = PAGE, 2 * PAGE  # two pages: every row scores its historical page 0
    w = mod.BatchPrefillIndexerQ8KV8Wrapper(num_heads=int(num_heads))
    w.plan(
        torch.tensor([0, qo], dtype=torch.int32, device=device),
        torch.tensor([kv], dtype=torch.int32, device=device),
        torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        total_q=qo,
        max_seqlen_q=qo,
        max_seqlen_k=kv,
    )
    w.run_scores(
        torch.zeros((qo, int(num_heads), HEAD_DIM), dtype=torch.float8_e4m3fn, device=device),
        torch.zeros((2, PAGE, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device),
    )
    logger.info(
        f"[M3.1 dep2ctx] IDX_SCORER prewarm: scorer kernels ready ({int(num_heads)} index heads)"
    )


def _off_reason(model) -> Optional[str]:
    """Why the scorer must not register on this model, or None."""
    if not IDX_SCORER:
        return "TRTLLM_M31_DISABLE_DEP2CTX_IDX_SCORER=1"
    if not tile_select.TSEL or tile_select._rows_op() is None:
        # Without the row-major tile select the zero-copy view would go to the one-CTA-per-row built-in select.
        return "the row-major tile select is off or not built (DEP2CTX_TSEL / DEP2CTX_TSEL_ROWS)"
    for layer in model.model.layers:
        indexer = getattr(layer.self_attn.attn, "indexer", None)
        if indexer is not None and int(indexer.config.local_blocks) < 1:
            # The scorer leaves a row's local page unwritten; only a forced local page keeps that cell unread.
            return f"local_blocks {int(indexer.config.local_blocks)} < 1"
    if _module() is None:
        return "the exact scorer module is unavailable"
    return None


def register(model) -> int:
    """Register ``score`` (bound to the indexer's per-rank KV head count) on every layer's MSA indexer that has an
    unset scorer hook, with ``tile_select.select_rows`` as its ``prefill_scorer_selector`` when that is unset; the
    number registered."""
    reason = _off_reason(model)
    if reason is not None:
        logger.info(f"[M3.1 dep2ctx] IDX_SCORER_OFF: {reason}; the proxy scores")
        return 0
    registered = 0
    for layer in model.model.layers:
        indexer = getattr(layer.self_attn.attn, "indexer", None)  # only the MSA backend has one
        if indexer is not None and getattr(indexer, "prefill_scorer_hook", False) is None:
            indexer.prefill_scorer_hook = functools.partial(
                score, num_kv_heads=int(indexer.config.num_kv_heads)
            )
            if getattr(indexer, "prefill_scorer_selector", False) is None:
                indexer.prefill_scorer_selector = tile_select.select_rows
            registered += 1
    if not registered:
        logger.info(
            "[M3.1 dep2ctx] IDX_SCORER_OFF: no MSA indexer exposes prefill_scorer_hook; the proxy scores"
        )
        return 0
    logger.info(f"[M3.1 dep2ctx] IDX_SCORER_REGISTERED: prefill scorer on {registered} layers")
    return registered
