# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_IDX_SCORER (CTX, exact) on the sm_107 tp2ctx target: the prefill indexer scores from the Q8KV8 prefill scorer
(MiniMax msa_dev 25c1447, vendored by MSA) at the generic F8F6F4 MMA (K = 32), in place of the fmha_sm100 OnlyScore
proxy pass and its single-CTA plan.

The scorer's raw page maxima equal the proxy's on every historical page bit for bit (CTX 840485), and the top-16 tables
the select builds from them are identical (840579): it never writes a row's local page, which the select forces anyway.
Plan + kernel per call on sm_107 (840558): A2K 316.7 -> 63.5 us, A8K 1196 -> 116, a 16K chunk 2311 -> 189.

Registered on every layer's MSA indexer ``prefill_scorer_hook``. The plan (it reads only the host lengths) is reused by
the next call with the same lengths; every call re-gathers the planned block table in place from the page table it is
handed, so a page table rewritten in place (also an inference tensor, which has no version counter) is always seen.

Declines (the proxy runs) under CUDA-graph capture, for non-FP8 operands, for a
page / head size other than 128, for rows whose causal offset is not bottom-right aligned, and when the scorer is not
the exact K = 32 build. Opt-in: ``TRTLLM_M31_ENABLE_TP2CTX_IDX_SCORER=1``; ``TRTLLM_M31_DISABLE_TP2CTX_IDX_SCORER=1``
wins.
"""

import functools
import importlib
import os
from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    _install_msa_cutlass_compatibility,
)
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled, m31_enabled

IDX_SCORER = m31_enabled("TP2CTX_IDX_SCORER") and not m31_disabled("TP2CTX_IDX_SCORER")
# The vendored scorer's interface module (MSA names the path; overridable for tests).
SCORER_MODULE = os.environ.get(
    "TRTLLM_M31_IDX_SCORER_MODULE",
    "tensorrt_llm._torch.attention.backends.fmha.msa_kernels.fmha_sm100_q8kv8.q8_indexer_interface",
)
PAGE = 128  # capacity: fixed: the index-K cache page size the scorer is built for
HEAD_DIM = 128  # capacity: fixed: the index head size the scorer is built for

_STATE: dict = {}


def _log_once(key: str, msg: str) -> None:
    if key not in _STATE:
        _STATE[key] = True
        logger.info(msg)


def _module():
    """The vendored interface module when it is importable and exact (K = 32 on sm_107), else None. Decided once."""
    if "module" in _STATE:
        return _STATE["module"]
    mod = None
    try:
        # The scorer is a CuTe DSL kernel: the image's cutlass DSL needs the MSA compatibility shim first.
        _install_msa_cutlass_compatibility()
        mod = importlib.import_module(SCORER_MODULE)
        prefill = mod.Q8KV8PrefillIndexerSm100
        if prefill.uses_sm107_fp8_mma((10, 7)) or prefill.mma_inst_k((10, 7)) != 32:
            _log_once(
                "not_exact",
                "[M3.1 tp2ctx] IDX_SCORER_OFF: the scorer is not the exact K=32 build (proxy kept)",
            )
            mod = None
    except (ImportError, AttributeError) as err:
        _log_once("no_module", f"[M3.1 tp2ctx] IDX_SCORER_OFF: {SCORER_MODULE} unavailable ({err})")
        mod = None
    _STATE["module"] = mod
    return mod


def _wrapper(mod, num_heads: int, key: tuple, qo, kv, kv_indices: torch.Tensor):
    """The planned wrapper for these lengths (the plan reads only the lengths; reused for the same ``key``), with its
    block table re-gathered in place from ``kv_indices`` on every call (the scorer kernel reads it at run time)."""
    w = _STATE.get("wrapper")
    if w is not None and _STATE.get("key") == key:
        gather, valid, block_table = _STATE["gather"]
        block_table.copy_(
            torch.where(
                valid,
                kv_indices[gather],
                torch.zeros((), dtype=kv_indices.dtype, device=valid.device),
            )
        )
        return w
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
    gather = torch.where(valid, gather, torch.zeros_like(gather)).to(device)
    valid = valid.to(device)
    # A normal tensor, so the in-place re-gather works in and out of inference mode.
    with torch.inference_mode(False):
        block_table = torch.empty(gather.shape, dtype=torch.int32, device=device)
    block_table.copy_(
        torch.where(
            valid, kv_indices[gather], torch.zeros((), dtype=kv_indices.dtype, device=device)
        )
    )
    cu_q = torch.tensor(
        [0] + list(torch.tensor(qo).cumsum(0).tolist()), dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(kv, dtype=torch.int32, device=device)
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
    contiguous group of this rank's index heads, as _group_max_reduce), or None to keep the proxy."""
    if torch.cuda.is_current_stream_capturing():
        return None
    if num_kv_heads <= 0 or idx_q.dim() != 3 or idx_q.shape[1] % num_kv_heads:
        return None
    if (
        idx_q.dtype != torch.float8_e4m3fn
        or idx_k_paged.dtype != torch.float8_e4m3fn
        or idx_q.dim() != 3
        or idx_q.shape[-1] != HEAD_DIM
        or idx_k_paged.dim() != 4
        or idx_k_paged.shape[1] != 1
        or idx_k_paged.shape[2] != PAGE
        or idx_k_paged.shape[3] != HEAD_DIM
    ):
        return None
    qo = [int(x) for x in qo_lens_cpu.tolist()]
    kv = [int(x) for x in kv_lens_cpu.tolist()]
    if not qo or sum(qo) != idx_q.shape[0] or min(qo) <= 0:
        return None
    if qo_offset_cpu is not None and [int(x) for x in qo_offset_cpu.tolist()] != [
        k - q for k, q in zip(kv, qo)
    ]:
        return None  # the scorer aligns queries bottom-right (position = kv_len - qo_len + i)
    mod = _module()
    if mod is None:
        return None
    num_heads = int(idx_q.shape[1])
    # The plan reads only the lengths; the block table is re-gathered from kv_indices on every call, so an in-place
    # rewrite of the page table (an inference tensor has no version counter) is always seen without a re-plan.
    key = (tuple(qo), tuple(kv), num_heads)
    w = _wrapper(mod, num_heads, key, qo, kv, kv_indices)
    scores = w.run_scores(
        idx_q, idx_k_paged.view(idx_k_paged.shape[0], PAGE, HEAD_DIM)
    )  # [T * H, pages]
    total_q = idx_q.shape[0]
    _log_once(
        f"taken_{num_heads}",
        f"[M3.1 tp2ctx] IDX_SCORER_ON: exact K=32 prefill scorer ({total_q} rows x {num_heads} index heads)",
    )
    # Index head h belongs to KV head h // group (contiguous groups, as _group_max_reduce): tp2ctx has 2 index heads and
    # 2 KV heads per rank (group 1), so an amax over all the rank's heads would merge two KV heads (gate 841031).
    group = num_heads // num_kv_heads
    return scores.view(total_q, num_kv_heads, group, -1).amax(dim=2).permute(1, 2, 0).contiguous()


def register(model) -> int:
    """Register ``score`` (bound to the indexer's per-rank KV head count) on every layer's MSA indexer that has an
    unset scorer hook; the number registered."""
    if not IDX_SCORER:
        logger.info(
            "[M3.1 tp2ctx] IDX_SCORER_OFF: not enabled (TRTLLM_M31_ENABLE_TP2CTX_IDX_SCORER); the proxy scores"
        )
        return 0
    registered = 0
    for layer in model.model.layers:
        indexer = getattr(layer.self_attn.attn, "indexer", None)  # only the MSA backend has one
        if indexer is not None and getattr(indexer, "prefill_scorer_hook", False) is None:
            indexer.prefill_scorer_hook = functools.partial(
                score, num_kv_heads=int(indexer.config.num_kv_heads)
            )
            registered += 1
    if not registered:
        logger.info(
            "[M3.1 tp2ctx] IDX_SCORER_OFF: no MSA indexer exposes prefill_scorer_hook; the proxy scores"
        )
    return registered
