# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""FMHA varlen attention API: plan + run for SM100.

fmha_sm100_plan and fmha_sm100.

Doc is REPO/docs/fmha_sm100_api.md
"""

import math
from typing import Optional, Tuple, Union

__all__ = [
    "fmha_sm100_plan", "fmha_sm100", "sparse_topk_select",
    "_fmha_sm100_plan", "_fmha_sm100",
]

import numpy as np
import os as _ss_os0
_SS_VERIFY_ZERO = _ss_os0.environ.get("TLLM_FMHA_PLAN_SS_VERIFY", "0") == "1"
import torch

from .jit import _dlpack_dtype_code, _PACK_FACTORS, get_fmha_variant, get_reduction_module, get_plan_fn, get_sparse_topk_module
from .sparse_fmha_adapter import sparse_fmha, sparse_fmha_plan


_np_staging = np.empty(4096 * 1024, dtype=np.int32)
_np_staging_offset = 0

def _reset_np_staging():
    global _np_staging_offset
    _np_staging_offset = 0

def _plan_buf_from_list(data, device):
    global _np_staging, _np_staging_offset
    n = len(data)
    end = _np_staging_offset + n
    if end > _np_staging.shape[0]:
        _np_staging = np.empty(max(end, _np_staging.shape[0] * 2), dtype=np.int32)
        _np_staging_offset = 0
        end = n
    _np_staging[_np_staging_offset:end] = data
    buf = torch.empty(n, dtype=torch.int32, device=device)
    buf.copy_(torch.from_numpy(_np_staging[_np_staging_offset:end]), non_blocking=True)
    _np_staging_offset = end
    return buf

from enum import IntEnum, auto

class _BuffTag(IntEnum):
    fmha_sm100_cutlass_workspace = auto()

    packed_work_range = auto()
    packed_work_info = auto()
    kv_tile_begin_indices = auto()
    kv_tile_end_indices = auto()
    kv_split_indices = auto()
    plan_cost = auto()
    num_kv_splits_per_row = auto()
    workspace_lse = auto()

    workspace_o = auto()

    sparse_topk_workspace = auto()

    Total = auto()


_workspace_cache = [[None] * _BuffTag.Total for _ in range(16)]
# _workspace_cache_per_plan = []

def _new_ws_cache():
    pass
    # global _workspace_cache
    # if len(_workspace_cache) == 0:
    #     print("new")
    #     _workspace_cache_per_plan.append([[None] * _BuffTag.Total for _ in range(16)])

def _alloc_workspace_buf(tag, size, device, dtype):
    global _workspace_cache
    device_id = torch.device(device).index
    buf = _workspace_cache[device_id][tag]
    if buf is not None and buf.shape[0] >= size:
        return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    _workspace_cache[device_id][tag] = buf
    return buf

def _get_workspace_buf(tag, device):
    global _workspace_cache
    device_id = torch.device(device).index
    return _workspace_cache[device_id][tag]

def _alloc_perplan_buf(tag, size, device, dtype):
    if _SS_VERIFY_ZERO:
        return torch.zeros(size, dtype=dtype, device=device)  # verify mode only: unwritten tails compare equal
    # global _workspace_cache
    # device_id = torch.device(device).index
    # buf = _workspace_cache[-1][device_id][tag]
    # if buf is not None and buf.shape[0] >= size:
    #     return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    # _workspace_cache[-1][device_id][tag] = buf
    return buf

_NUM_CTA = None
def _get_num_cta(device):
    global _NUM_CTA
    if _NUM_CTA is None:
        _NUM_CTA = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_CTA

def _compute_pack_factor(max_qo_len, num_qo_heads, num_kv_heads):
    if num_kv_heads == -1:
        return 1
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1

def _to_device(t, d):
    return t.to(d, non_blocking=True) if t is not None else None

def _prefill_qlen_threshold(sparse):
    return 32 if sparse else 128

def _validate_fmha_inputs(
    q, k, v, qo_segment_lens, kv_segment_lens,
    num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
    batch_size, qo_total_len, is_paged,
    kv_indices, kv_block_indexes, qo_offset, page_size,
    pack_factor=1,
    packed_work_range=None, packed_work_info=None,
    num_kv_splits=1,
    kv_tile_begin_indices=None, kv_tile_end_indices=None, kv_split_indices=None,
    qo_segment_offsets=None, kv_page_indptr=None,
):
    assert q.dim() == 3, f"q must be [total_qo_len, num_qo_heads, head_dim], got {q.shape}"
    assert num_qo_heads % num_kv_heads == 0, (
        f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    assert qo_segment_lens.dim() == 1, f"qo_segment_lens must be 1D, got {qo_segment_lens.shape}"
    assert kv_segment_lens.dim() == 1, f"kv_segment_lens must be 1D, got {kv_segment_lens.shape}"
    assert qo_segment_lens.shape[0] == batch_size
    assert kv_segment_lens.shape[0] == batch_size

    qo_lens_cpu = qo_segment_lens.cpu()
    kv_lens_cpu = kv_segment_lens.cpu()

    expected_qo_total = qo_total_len * pack_factor if pack_factor > 1 else qo_total_len
    assert qo_lens_cpu.sum().item() == expected_qo_total, (
        f"sum(qo_segment_lens)={qo_lens_cpu.sum().item()} != expected={expected_qo_total} "
        f"(q.shape[0]={qo_total_len}, pack_factor={pack_factor})"
    )

    if is_paged:
        assert k.dim() == 4, f"paged K must be [total_pages, H_kv, page_size, D], got {k.shape}"
        assert v.dim() == 4, f"paged V must be [total_pages, H_kv, page_size, D], got {v.shape}"
        total_pages = k.shape[0]
        assert k.shape[1] == num_kv_heads
        assert k.shape[2] == page_size
        assert v.shape[0] == total_pages
        assert v.shape[1] == num_kv_heads
        assert v.shape[2] == page_size
    else:
        total_kv_len = k.shape[0]
        assert kv_lens_cpu.sum().item() == total_kv_len, (
            f"sum(kv_segment_lens)={kv_lens_cpu.sum().item()} != k.shape[0]={total_kv_len}"
        )

    if qo_offset is not None:
        assert qo_offset.dim() == 1 and qo_offset.shape[0] == batch_size, (
            f"qo_offset must be [batch_size={batch_size}], got {qo_offset.shape}"
        )
        # off_cpu = qo_offset.cpu()
        # assert (off_cpu >= 0).all(), (
        #     f"qo_offset must be non-negative, got min={off_cpu.min().item()}"
        # )
        # unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        # max_offsets = kv_lens_cpu - unpacked_qo_lens
        # violations = off_cpu > max_offsets
        # if violations.any():
        #     b = violations.nonzero()[0].item()
        #     assert False, (
        #         f"qo_offset[{b}]={off_cpu[b].item()} > kv_len-qo_len="
        #         f"{kv_lens_cpu[b].item()}-{qo_lens_cpu[b].item()}={max_offsets[b].item()}"
        #     )

    if kv_indices is not None and kv_block_indexes is None:
        assert is_paged, "kv_indices provided but K/V are not paged (4D)"
        kvi_cpu = kv_indices.cpu()
        total_pages_in_table = kvi_cpu.shape[0]
        total_pages_needed = sum(
            (kv_lens_cpu[b].item() + page_size - 1) // page_size for b in range(batch_size)
        )
        assert total_pages_in_table == total_pages_needed, (
            f"kv_indices length {total_pages_in_table} != "
            f"sum(ceil(kv_segment_lens/page_size))={total_pages_needed}"
        )
        total_pages = k.shape[0]
        assert (kvi_cpu >= 0).all() and (kvi_cpu < total_pages).all(), (
            f"kv_indices has values outside [0, {total_pages}): "
            f"min={kvi_cpu.min().item()}, max={kvi_cpu.max().item()}"
        )

    if kv_block_indexes is not None:
        assert is_paged, "sparse mode requires paged KV (kv_indices)"
        assert kv_indices is not None, "sparse mode requires kv_indices"
        assert kv_block_indexes.dim() == 3, (
            f"kv_block_indexes must be [total_qo_len, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        )
        assert kv_block_indexes.shape[0] == qo_total_len, (
            f"kv_block_indexes must be [total_qo_len={qo_total_len}, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        )
        assert kv_block_indexes.shape[1] == num_kv_heads

        bi_cpu = kv_block_indexes.cpu()
        valid_mask = (bi_cpu != -1)
        assert ((bi_cpu >= 0) | (bi_cpu == -1)).all(), (
            "kv_block_indexes must contain only non-negative indices or -1"
        )
        assert (valid_mask[:, :, :-1] >= valid_mask[:, :, 1:]).all(), (
            "kv_block_indexes: -1 padding must be at the end"
        )
        unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        batch_of_qtoken = torch.repeat_interleave(
            torch.arange(batch_size), unpacked_qo_lens)
        kv_lens_per_qtoken = kv_lens_cpu[batch_of_qtoken]
        num_pages = ((kv_lens_per_qtoken + page_size - 1) // page_size).view(-1, 1, 1)
        assert (bi_cpu[valid_mask] < num_pages.expand_as(bi_cpu)[valid_mask]).all(), (
            "kv_block_indexes: index out of range for batch page count"
        )
        both_valid = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        if both_valid.any():
            assert (bi_cpu[:, :, 1:][both_valid] > bi_cpu[:, :, :-1][both_valid]).all(), (
                "kv_block_indexes: must be strictly ascending"
            )

    if qo_segment_offsets is not None:
        off_cpu = qo_segment_offsets.cpu()
        assert off_cpu.shape[0] == batch_size + 1, (
            f"qo_segment_offsets must have {batch_size + 1} elements, got {off_cpu.shape[0]}"
        )
        assert off_cpu[0].item() == 0, f"qo_segment_offsets[0]={off_cpu[0].item()} != 0"
        assert off_cpu[-1].item() == expected_qo_total, (
            f"qo_segment_offsets[-1]={off_cpu[-1].item()} != expected={expected_qo_total}"
        )
        diffs = off_cpu[1:] - off_cpu[:-1]
        assert (diffs >= 0).all(), "qo_segment_offsets must be non-decreasing"
        assert (diffs == qo_lens_cpu).all(), "qo_segment_offsets must be cumsum of qo_segment_lens"

    if kv_page_indptr is not None:
        ip_cpu = kv_page_indptr.cpu()
        assert ip_cpu.shape[0] == batch_size + 1, (
            f"kv_page_indptr must have {batch_size + 1} elements, got {ip_cpu.shape[0]}"
        )
        assert ip_cpu[0].item() == 0, f"kv_page_indptr[0]={ip_cpu[0].item()} != 0"
        assert (ip_cpu[1:] >= ip_cpu[:-1]).all(), "kv_page_indptr must be non-decreasing"
        if kv_indices is not None:
            assert ip_cpu[-1].item() <= kv_indices.shape[0], (
                f"kv_page_indptr[-1]={ip_cpu[-1].item()} > kv_indices.size={kv_indices.shape[0]}"
            )

    if packed_work_range is not None and packed_work_info is not None:
        import math
        pwr_cpu = packed_work_range.cpu()
        pwi_cpu = packed_work_info.cpu()
        num_ctas = pwr_cpu.shape[0]
        max_work_idx = pwi_cpu.shape[0]

        packed_num_heads = num_qo_heads // pack_factor if pack_factor > 1 else num_qo_heads
        qo_tile_size = 128 if int(qo_lens_cpu.max()) <= 128 else 256
        max_tiles_per_batch = [(int(l) + qo_tile_size - 1) // qo_tile_size for l in qo_lens_cpu]

        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            assert start <= end, (
                f"packed_work_range[{cta}]: start={start} > end={end}"
            )
            assert end <= max_work_idx, (
                f"packed_work_range[{cta}]: end={end} > packed_work_info.size={max_work_idx}"
            )
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qo_tile = (packed >> 32) & 0xFFFFFFFF
                assert bi < batch_size, (
                    f"packed_work_info[{wi}]: batch_idx={bi} >= batch_size={batch_size}"
                )
                assert hi < packed_num_heads, (
                    f"packed_work_info[{wi}]: head_idx={hi} >= packed_num_heads={packed_num_heads}"
                )
                assert qo_tile < max_tiles_per_batch[bi], (
                    f"packed_work_info[{wi}]: qo_tile={qo_tile} >= max_tiles={max_tiles_per_batch[bi]} "
                    f"for batch {bi} (qo_len={int(qo_lens_cpu[bi])})"
                )

        # Verify completeness: every (batch, head, qo_tile) must appear exactly once
        # (for num_kv_splits=1) or exactly num_splits times (for split-KV, each with
        # a different split index covering the full KV range).
        expected = set()
        for bi in range(batch_size):
            for hi in range(packed_num_heads):
                for qt in range(max_tiles_per_batch[bi]):
                    expected.add((bi, hi, qt))

        seen = {}
        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qt = (packed >> 32) & 0xFFFFFFFF
                key = (bi, hi, qt)
                seen[key] = seen.get(key, 0) + 1

        missing = expected - set(seen.keys())
        assert not missing, (
            f"plan missing {len(missing)} work items, e.g. {list(missing)[:5]}"
        )

        if num_kv_splits <= 1:
            duplicates = {k: v for k, v in seen.items() if v > 1}
            assert not duplicates, (
                f"plan has {len(duplicates)} duplicated work items (nosplit), e.g. {list(duplicates.items())[:5]}"
            )

        if num_kv_splits > 1 and kv_tile_begin_indices is not None:
            tb_cpu = kv_tile_begin_indices.cpu()
            te_cpu = kv_tile_end_indices.cpu()
            sp_cpu = kv_split_indices.cpu()

            from collections import defaultdict
            tile_splits = defaultdict(list)

            for cta in range(num_ctas):
                r = int(pwr_cpu[cta].item())
                start = r & 0xFFFFFFFF
                end = (r >> 32) & 0xFFFFFFFF
                for wi in range(start, end):
                    tb, te = int(tb_cpu[wi].item()), int(te_cpu[wi].item())
                    assert tb <= te, (
                        f"kv_tile_begin[{wi}]={tb} > kv_tile_end[{wi}]={te}"
                    )
                    sp = int(sp_cpu[wi].item())
                    assert 0 <= sp < num_kv_splits, (
                        f"kv_split_indices[{wi}]={sp} out of range [0, {num_kv_splits})"
                    )
                    packed = int(pwi_cpu[wi].item())
                    key = (packed & 0xFFFF, (packed >> 16) & 0xFFFF,
                           (packed >> 32) & 0xFFFFFFFF)
                    tile_splits[key].append((tb, te, sp))

            is_sparse = kv_block_indexes is not None
            kv_tile_size = 256 if qo_tile_size == 128 else 128
            for key, splits in tile_splits.items():
                bi, hi, qt = key
                if is_sparse:
                    kv_block_num = kv_block_indexes.shape[2]
                    kl = kv_block_num * page_size
                    off_q = kl - int(qo_lens_cpu[bi])
                else:
                    kl = int(kv_lens_cpu[bi])
                    off_q = int(qo_offset[bi].cpu()) if qo_offset is not None else (kl - int(qo_lens_cpu[bi]))
                packed_q_end = (qt + 1) * qo_tile_size
                q_end = (packed_q_end - 1) // pack_factor + 1 if pack_factor > 1 else packed_q_end
                eff_kv = min(q_end + off_q, kl)
                expected_iters = max(0, (eff_kv + kv_tile_size - 1) // kv_tile_size)
                splits_sorted = sorted(splits, key=lambda x: x[0])
                if expected_iters > 0:
                    assert splits_sorted[0][0] == 0, (
                        f"tile {key}: first split begins at {splits_sorted[0][0]}, expected 0"
                    )
                    assert splits_sorted[-1][1] == expected_iters, (
                        f"tile {key}: last split ends at {splits_sorted[-1][1]}, expected {expected_iters}"
                    )
                for i in range(1, len(splits_sorted)):
                    prev_end = splits_sorted[i - 1][1]
                    curr_begin = splits_sorted[i][0]
                    assert prev_end == curr_begin, (
                        f"tile {key}: gap/overlap at split boundary: "
                        f"prev_end={prev_end}, curr_begin={curr_begin}"
                    )
                sub_ids = sorted(s[2] for s in splits_sorted)
                expected_ids = list(range(len(splits_sorted)))
                assert sub_ids == expected_ids, (
                    f"tile {key}: sub_ids {sub_ids} not contiguous 0..{len(splits_sorted)-1}"
                )


def _expand_for_per_token_sparse(qo_lens, kv_lens, qo_offset, page_size, pack_factor=1):
    B = len(qo_lens)
    total_q = sum(qo_lens)

    if qo_offset is None:
        qo_offset = [kv_lens[i] - qo_lens[i] for i in range(B)]

    kv_page_indptr = [0]
    acc = 0
    for k in kv_lens:
        acc += (k + page_size - 1) // page_size
        kv_page_indptr.append(acc)

    expanded_qo_lens = [pack_factor] * total_q
    expanded_kv_lens = [0] * total_q
    expanded_qo_offset = [0] * total_q
    expanded_kv_page_indptr = [0] * (total_q + 1)

    idx = 0
    for b in range(B):
        kv_len_b = kv_lens[b]
        qo_offset_b = qo_offset[b]
        kv_page_indptr_b = kv_page_indptr[b]
        for j in range(qo_lens[b]):
            expanded_kv_lens[idx] = kv_len_b
            expanded_qo_offset[idx] = qo_offset_b + j
            expanded_kv_page_indptr[idx] = kv_page_indptr_b
            idx += 1

    expanded_kv_page_indptr[total_q] = kv_page_indptr[B]

    return expanded_qo_lens, expanded_kv_lens, expanded_qo_offset, expanded_kv_page_indptr

class PlanInfo(dict):
    """Execution plan returned by the internal dense FMHA planner.

    Users normally receive the public tuple returned by ``fmha_sm100_plan`` and
    pass it unchanged to ``fmha_sm100``.  The dictionary stores CUDA worklists,
    sequence metadata, split-KV workspaces, and cached buffers owned by the
    plan.
    """

    def __del__(self):
        pass
        # print("del")
        # _workspace_cache_per_plan.append(self["_ws_cache"])

def _make_plan_info(
    packed_work_range, packed_work_info,
    kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
    num_kv_splits, workspace_o, workspace_lse,
    max_qo_len, predicted_speedup, num_kv_splits_per_row,
    qo_segment_offsets, kv_segment_offsets, kv_page_indptr, max_k_tiles,
    qo_segment_lens, kv_segment_lens, qo_offset,
    pack_factor, orig_num_qo_heads,
    qo_len_uniform, cute_workspace_buffer, qo_tile_size=None
):
    # ws = _workspace_cache_per_plan.pop()
    # print("pop")
    return PlanInfo({
        # "_ws_cache" : ws,
        "packed_work_range": packed_work_range,
        "packed_work_info": packed_work_info,
        "kv_tile_begin_indices": kv_tile_begin_indices,
        "kv_tile_end_indices": kv_tile_end_indices,
        "kv_split_indices": kv_split_indices,
        "num_kv_splits": num_kv_splits,
        "workspace_o": workspace_o,
        "workspace_lse": workspace_lse,
        "max_qo_len": max_qo_len,
        "predicted_speedup": predicted_speedup,
        "num_kv_splits_per_row": num_kv_splits_per_row,
        "qo_segment_offsets": qo_segment_offsets,
        "kv_segment_offsets": kv_segment_offsets,
        "kv_page_indptr": kv_page_indptr,
        "max_k_tiles": max_k_tiles,
        "qo_segment_lens": qo_segment_lens,
        "kv_segment_lens": kv_segment_lens,
        "qo_offset": qo_offset,
        "pack_factor": pack_factor,
        "orig_num_qo_heads": orig_num_qo_heads,
        "qo_len_uniform": qo_len_uniform,
        "cute_workspace_buffer": cute_workspace_buffer,
        "MM-SA-Nv":False,
        **({"qo_tile_size": qo_tile_size} if qo_tile_size is not None else {}),
    })


# ---- MiniMax-M3 (SM107 only): DEFERRED split decision (exact). The adaptive plan's split decision (plan_cost[0] = split,
# [1] = predicted speedup) is read back with plan_cost.tolist() in prepare(); its single-CTA plan kernel waits for an SM
# under the previous step's kernels, so that read stalls the prepare thread (~1 ms per ctx-bearing step) and costs the
# overlap scheduler its run-ahead. Deferred: the same plan kernel, its 2 floats copied to a pinned per-plan buffer with
# an event, and the split-dependent plan fields filled at their first read -- the attention launch that consumes the plan,
# an eager point, by which time the event has normally completed. The fields are then the ones the immediate read gives.
# Never resolved under stream capture (asserted). Kill: TLLM_FMHA_PLAN_DEFER_READ=0. TLLM_FMHA_PLAN_DECISION_LOG=<path>
# appends one line per adaptive plan (split, speedup, max_kv_splits, total_qo) on either path, for the trace gate.
_PLAN_DEFER_READ = _ss_os0.environ.get("TLLM_FMHA_PLAN_DEFER_READ", "1") != "0"
_PLAN_DECISION_LOG = _ss_os0.environ.get("TLLM_FMHA_PLAN_DECISION_LOG")
_SPLIT_KEYS = frozenset((
    "kv_tile_begin_indices", "kv_tile_end_indices", "kv_split_indices", "num_kv_splits", "workspace_o",
    "workspace_lse", "predicted_speedup", "num_kv_splits_per_row",
))


def _log_plan_decision(split, speedup, max_kv_splits, total_qo_len, path):
    if _PLAN_DECISION_LOG:
        with open(_PLAN_DECISION_LOG, "a") as f:
            f.write(f"pid={_ss_os0.getpid()} {path} split={int(bool(split))} speedup={speedup!r} "
                    f"max_kv_splits={max_kv_splits} total_qo={total_qo_len}\n")


class _DeferredSplitPlanInfo(PlanInfo):
    """A PlanInfo whose split-dependent fields are filled from the plan's cost read at their first access."""

    def _resolve(self):
        lazy = self.__dict__.pop("_lazy", None)
        if lazy is None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fmha_sm100: a deferred plan decision was first read under CUDA graph capture")
        lazy["event"].synchronize()
        split_flag, speedup = lazy["host"].tolist()
        do_split = split_flag > 0
        _log_plan_decision(do_split, speedup, lazy["max_kv_splits"], lazy["total_qo_len"], "deferred")
        put = dict.__setitem__
        if do_split:
            put(self, "workspace_o", _alloc_workspace_buf(
                _BuffTag.workspace_o, lazy["total_qo_len"] * lazy["max_kv_splits"] * lazy["num_qo_heads"] * 128,
                lazy["device"], dtype=torch.bfloat16))
            put(self, "num_kv_splits", lazy["max_kv_splits"])
        else:
            for key in ("kv_tile_begin_indices", "kv_tile_end_indices", "kv_split_indices",
                        "num_kv_splits_per_row", "workspace_o", "workspace_lse"):
                put(self, key, None)
            put(self, "num_kv_splits", 1)
        put(self, "predicted_speedup", speedup)

    def __getitem__(self, key):
        if key in _SPLIT_KEYS and "_lazy" in self.__dict__:
            self._resolve()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in _SPLIT_KEYS and "_lazy" in self.__dict__:
            self._resolve()
        return dict.get(self, key, default)

    def items(self):
        # fmha_sm100_plan's mixed-batch path clones a sub-plan through items(): resolve first (that path keeps the
        # immediate read). values() stays unresolved: the side-stream wrapper only record_stream()s the tensors.
        self._resolve()
        return dict.items(self)


def _call_plan(qo_segment_offsets, qo_segment_lens, kv_segment_lens,
               packed_work_range, packed_work_info,
               qo_tile_size, kv_tile_size, num_qo_heads, num_ctas, causal,
               qo_offset, num_kv_splits,
               kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
               chunk_size, out_max_sm_cost, num_kv_splits_per_row, workspace_lse, lse_total_size,
               pack_factor, cuda_stream=None,
               ):

    plan_module = get_plan_fn()

    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    plan_module.plan(
        qo_segment_offsets, qo_segment_lens, kv_segment_lens,
        packed_work_range, packed_work_info,
        qo_tile_size, kv_tile_size, num_qo_heads, num_ctas, causal,
        qo_offset, num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        chunk_size, out_max_sm_cost, num_kv_splits_per_row,
        cuda_stream,
        workspace_lse, lse_total_size,
        pack_factor,
    )

def _fmha_sm100_plan_impl(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    usable_SM_count: int = -1,
    causal: bool = True,
    sparse_kernel_mode: str = 'auto',
    use_fp8_kvcache: bool = False,
    maxscore_qo_tile: int = -1,
    device = None,
    stream = None
):
    # maxscore_qo_tile (opt-in, default -1 = the tile rule below): 128 plans a long-q max-score pass (the M3.1 tp2ctx
    # prefill indexer proxy) with the tile_q 128 x tile_kv 256 variant, the only split-capable tile, instead of the
    # tile_q 256 the rule picks for q > 128 (which also forces num_kv_splits = 1). A max score is per (row, kv block),
    # so neither the tile nor the split changes one bit of it. Dense planner only; q <= 128 is unaffected (the rule
    # already picks 128 there).
    if maxscore_qo_tile != -1 and (maxscore_qo_tile != 128 or not output_maxscore or kv_block_num > 0):
        raise ValueError("maxscore_qo_tile supports only 128, with output_maxscore, on the dense planner")
    device = torch.cuda.current_device() if device is None else device
    _reset_np_staging()

    qo_lens = qo_segment_lens.tolist()
    max_qo_len_orig = max(qo_lens) if qo_lens else 0

    if kv_block_num > 0 and (sparse_kernel_mode == 'prefill' or (sparse_kernel_mode == 'auto' and max_qo_len_orig > _prefill_qlen_threshold(True))):
        # print("Nv-Prefill")
        # sparse_fmha_plan reads these lengths on the host and stages its own
        # device copies, so hand them through unchanged.
        return sparse_fmha_plan(qo_segment_lens=qo_segment_lens, kv_segment_lens=kv_segment_lens,
                num_qo_heads=num_qo_heads, causal=causal, qo_offset=qo_offset,
                num_kv_splits=num_kv_splits, page_size=page_size, output_maxscore=output_maxscore,
                kv_block_num=kv_block_num, num_kv_heads=num_kv_heads, usable_SM_count=usable_SM_count,
                use_fp8_kvcache=use_fp8_kvcache)

    _new_ws_cache()

    cute_workspace_buffer = _alloc_workspace_buf(_BuffTag.fmha_sm100_cutlass_workspace, 32 * 1024 * 1024, device, torch.uint8)

    cuda_stream = stream

    num_ctas = _get_num_cta(device)
    if usable_SM_count > 0:
        num_ctas = min(usable_SM_count, num_ctas)

    orig_num_qo_heads = num_qo_heads
    pack_factor = _compute_pack_factor(max_qo_len_orig, num_qo_heads, num_kv_heads)
    qo_len_uniform = len(qo_lens) > 0 and min(qo_lens) == max_qo_len_orig
    if pack_factor > 1:
        num_qo_heads = num_qo_heads // pack_factor

    kv_lens = kv_segment_lens.tolist()
    kv_page_indptr_list = None
    if kv_block_num > 0 and page_size > 0:
        total_q = sum(qo_lens)
        total_q_packed = total_q * pack_factor if pack_factor > 1 else total_q
        assert total_q_packed * num_qo_heads <= 65536
        qo_offset_in = qo_offset.tolist() if qo_offset is not None else None
        qo_lens, kv_lens, qo_offset, kv_page_indptr_list = \
            _expand_for_per_token_sparse(qo_lens, kv_lens, qo_offset_in, page_size, pack_factor)
    elif kv_block_num > 0 and page_size <= 0:
        print("[Error] Sparse mode must be used together with paged kv!")
    else:
        if pack_factor > 1:
            qo_lens = [q * pack_factor for q in qo_lens]
        if page_size > 0:
            acc = 0
            kv_page_indptr_list = [0]
            for k in kv_lens:
                acc += (k + page_size - 1) // page_size
                kv_page_indptr_list.append(acc)

    acc = 0
    qo_offsets = [0]
    for q in qo_lens:
        acc += q
        qo_offsets.append(acc)
    acc = 0
    kv_offsets = [0]
    for k in kv_lens:
        acc += k
        kv_offsets.append(acc)

    max_kv_len = max(kv_lens)
    qo_offset_list = qo_offset if isinstance(qo_offset, list) else qo_offset.tolist()
    if not causal:
        qo_offset_list = [max_kv_len] * len(kv_lens)

    qo_segment_offsets = _plan_buf_from_list(qo_offsets, device)
    kv_segment_offsets = _plan_buf_from_list(kv_offsets, device)
    qo_offset = _plan_buf_from_list(qo_offset_list, device)
    kv_segment_lens = _plan_buf_from_list(kv_lens, device)
    kv_page_indptr = _plan_buf_from_list(kv_page_indptr_list, device) if kv_page_indptr_list is not None else None

    plan_kv_lens_list = kv_lens
    plan_qo_offset = qo_offset
    if kv_block_num > 0 and page_size > 0:
        plan_kv_lens_list = [kv_block_num * page_size] * len(kv_lens)
        plan_qo_offset = None

    max_qo_len = max(qo_lens)
    qo_tile_size = 128 if max_qo_len <= 128 else 256
    _mq_tile = None
    if maxscore_qo_tile == 128:
        qo_tile_size = _mq_tile = 128
    kv_tile_size = 256 if qo_tile_size == 128 else 128

    total_qo_len = qo_offsets[-1]

    max_k_tiles = math.ceil(math.ceil(max_kv_len / 128) / 128) * 128 if output_maxscore else -1
    maxscore_elems = num_qo_heads * max_k_tiles * total_qo_len
    if output_maxscore and maxscore_elems > (1 << 31):
        print(f"Too huge setting to output maxscore!")
        max_k_tiles = -1

    if num_kv_splits < 1 and qo_tile_size == 128:
        group_iters = []
        batch_size = len(qo_lens)
        for b_idx in range(batch_size):
            ql_b, kl_b = qo_lens[b_idx], plan_kv_lens_list[b_idx]
            off_q = kl_b - ql_b
            for t in range(math.ceil(ql_b / qo_tile_size)):
                packed_q_end = (t + 1) * qo_tile_size
                q_end = (packed_q_end - 1) // pack_factor + 1 if pack_factor > 1 else packed_q_end
                if causal and q_end + off_q <= 0:
                    continue
                eff_kv = min(q_end + off_q, kl_b) if causal else kl_b
                group_iters.append(math.ceil(max(eff_kv, 0) / kv_tile_size))

        if not group_iters:
            num_kv_splits = 1
        elif len(group_iters) * num_qo_heads > 4096:
            # Too many tiles for split-KV smem — fall back to nosplit greedy
            num_kv_splits = 1
        else:
            total_iters = sum(group_iters) * num_qo_heads
            avg_iters = max(2, (total_iters + num_ctas - 1) // num_ctas + 3)
            chunk_size = avg_iters
            max_pieces = max(math.ceil(g / chunk_size) for g in group_iters)
            max_kv_splits = min(2 * max(max_pieces, 1), 64)

            max_work_items = 131072 * max_kv_splits
            
            packed_work_range = _alloc_perplan_buf(_BuffTag.packed_work_range, num_ctas, device, torch.int64)
            packed_work_info = _alloc_perplan_buf(_BuffTag.packed_work_info, max_work_items, device, torch.int64)
            kv_tile_begin_indices = _alloc_perplan_buf(_BuffTag.kv_tile_begin_indices, max_work_items, device, torch.int32)
            kv_tile_end_indices = _alloc_perplan_buf(_BuffTag.kv_tile_end_indices, max_work_items, device, torch.int32)
            kv_split_indices = _alloc_perplan_buf(_BuffTag.kv_split_indices, max_work_items, device, torch.int32)
            num_kv_splits_per_row = _alloc_perplan_buf(_BuffTag.num_kv_splits_per_row, total_qo_len, device, torch.int32)
            defer = _PLAN_DEFER_READ and _ss_is_sm107(device) and not torch.cuda.is_current_stream_capturing()
            # A deferred read owns its plan_cost (the shared one may be rewritten by a later plan on another stream).
            plan_cost = (torch.empty(2, dtype=torch.float32, device=device) if defer
                         else _alloc_workspace_buf(_BuffTag.plan_cost, 2, device, dtype=torch.float32))

            lse_total_size = max_kv_splits * total_qo_len * num_qo_heads
            workspace_lse = _alloc_perplan_buf(_BuffTag.workspace_lse, lse_total_size, device, torch.float32)

            qo_segment_lens_gpu = _plan_buf_from_list(qo_lens, device)
            plan_kv_lens_gpu = _plan_buf_from_list(plan_kv_lens_list, device)
            _call_plan(qo_segment_offsets,
                qo_segment_lens_gpu, plan_kv_lens_gpu,
                packed_work_range, packed_work_info,
                qo_tile_size, kv_tile_size,
                num_qo_heads, num_ctas, causal, plan_qo_offset,
                -max_kv_splits,
                kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
                chunk_size, plan_cost, num_kv_splits_per_row, workspace_lse, lse_total_size,
                pack_factor, cuda_stream)

            if defer:
                host_cost = torch.empty(2, dtype=torch.float32, pin_memory=True)
                host_cost.copy_(plan_cost, non_blocking=True)
                cost_ready = torch.cuda.Event()
                cost_ready.record()
                info = _make_plan_info(
                    packed_work_range=packed_work_range, packed_work_info=packed_work_info,
                    kv_tile_begin_indices=kv_tile_begin_indices, kv_tile_end_indices=kv_tile_end_indices,
                    kv_split_indices=kv_split_indices, num_kv_splits=max_kv_splits, workspace_o=None,
                    workspace_lse=workspace_lse, max_qo_len=max_qo_len, predicted_speedup=None,
                    num_kv_splits_per_row=num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets,
                    kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
                    qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
                    pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
                    qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer, qo_tile_size=_mq_tile)
                deferred = _DeferredSplitPlanInfo(info)
                # plan_cost is kept alive until the read resolves (the copy reads it on this stream).
                deferred.__dict__["_lazy"] = dict(event=cost_ready, host=host_cost, plan_cost=plan_cost,
                                                  max_kv_splits=max_kv_splits, total_qo_len=total_qo_len,
                                                  num_qo_heads=num_qo_heads, device=device)
                return deferred

            _tl0 = _ss_time.perf_counter()
            _cost = plan_cost.tolist()
            _log_plan_decision(_cost[0] > 0, _cost[1], max_kv_splits, total_qo_len, "immediate")
            _SS_STATS["tl"].append((_ss_time.perf_counter() - _tl0) * 1e3)
            do_split = _cost[0] > 0
            predicted_speedup = _cost[1]

            if do_split:
                workspace_o = _alloc_workspace_buf(_BuffTag.workspace_o, total_qo_len * max_kv_splits * num_qo_heads * 128, device, dtype=torch.bfloat16)
            else:
                kv_tile_begin_indices = None
                kv_tile_end_indices = None
                kv_split_indices = None
                num_kv_splits_per_row = None
                workspace_o = None
                workspace_lse = None
                max_kv_splits = 1

            return _make_plan_info(
                packed_work_range=packed_work_range, packed_work_info=packed_work_info,
                kv_tile_begin_indices=kv_tile_begin_indices, kv_tile_end_indices=kv_tile_end_indices, 
                kv_split_indices=kv_split_indices, num_kv_splits=max_kv_splits, workspace_o=workspace_o, 
                workspace_lse=workspace_lse, max_qo_len=max_qo_len, predicted_speedup=predicted_speedup, 
                num_kv_splits_per_row=num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets, 
                kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
                qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
                pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
                qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer, qo_tile_size=_mq_tile)

        num_kv_splits = 1
    elif num_kv_splits < 1:
        num_kv_splits = 1
    elif qo_tile_size == 256:
        num_kv_splits = 1

    packed_work_range = _alloc_perplan_buf(_BuffTag.packed_work_range, num_ctas, device, torch.int64)
    max_work_items = 131072 * max(num_kv_splits, 1)
    packed_work_info = _alloc_perplan_buf(_BuffTag.packed_work_info, max_work_items, device, torch.int64)
    if num_kv_splits > 1:
        kv_tile_begin_indices = _alloc_perplan_buf(_BuffTag.kv_tile_begin_indices, max_work_items, device, torch.int32)
        kv_tile_end_indices = _alloc_perplan_buf(_BuffTag.kv_tile_end_indices, max_work_items, device, torch.int32)
        kv_split_indices = _alloc_perplan_buf(_BuffTag.kv_split_indices, max_work_items, device, torch.int32)
        num_kv_splits_per_row = _alloc_perplan_buf(_BuffTag.num_kv_splits_per_row, total_qo_len, device, torch.int32)
        workspace_o = _alloc_workspace_buf(_BuffTag.workspace_o, total_qo_len * num_kv_splits * num_qo_heads * 128, device, dtype=torch.bfloat16)

        lse_total_size = num_kv_splits * total_qo_len * num_qo_heads
        workspace_lse = _alloc_perplan_buf(_BuffTag.workspace_lse, lse_total_size, device, torch.float32)
    else:
        kv_tile_begin_indices = None
        kv_tile_end_indices = None
        kv_split_indices = None
        num_kv_splits_per_row = None
        workspace_o = None
        
        lse_total_size = 0
        workspace_lse = None

    qo_segment_lens_gpu = _plan_buf_from_list(qo_lens, device)
    plan_kv_lens_gpu = _plan_buf_from_list(plan_kv_lens_list, device)
    _call_plan(qo_segment_offsets, qo_segment_lens_gpu, plan_kv_lens_gpu,
        packed_work_range, packed_work_info,
        qo_tile_size, kv_tile_size,
        num_qo_heads, num_ctas, causal, plan_qo_offset,
        num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        0, None, num_kv_splits_per_row, workspace_lse, lse_total_size,
        pack_factor, cuda_stream)

    return _make_plan_info(
        packed_work_range=packed_work_range, packed_work_info=packed_work_info,
        kv_tile_begin_indices=kv_tile_begin_indices, kv_tile_end_indices=kv_tile_end_indices,
        kv_split_indices=kv_split_indices, num_kv_splits=num_kv_splits, workspace_o=workspace_o,
        workspace_lse=workspace_lse, max_qo_len=max_qo_len, predicted_speedup=1.0,
        num_kv_splits_per_row=num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
        qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
        pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
        qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer, qo_tile_size=_mq_tile)




# ---- TTFT/CTX: side-stream plan (exact). The adaptive plan reads plan_cost back to the host (.tolist()); on the current
# stream that read waits for every kernel queued before it (the previous prefill chunk), so the host can no longer run
# ahead. All plan inputs come from host lists and every plan buffer is fresh, so the whole plan can run on a dedicated
# HIGH-priority stream: the read then waits only for the plan stream's own copies + the tiny plan kernel. The consumer
# stream (caller's stream, else current) then waits device-side on the plan stream, and every returned tensor is
# record_stream()'d for it. Same kernel, same inputs -> same plan. Kill: TLLM_FMHA_PLAN_SIDE_STREAM=0.
import inspect as _ss_inspect
import os as _ss_os
import sys as _ss_sys
import time as _ss_time

_SS_ON = _ss_os.environ.get("TLLM_FMHA_PLAN_SIDE_STREAM", "1") != "0"
_SS_STREAMS = {}
_SS_SIG = _ss_inspect.signature(_fmha_sm100_plan_impl)
_SS_STATS = {"n": 0, "waits": [], "logged": False, "vn": 0, "vbad": 0, "tl": []}
_SS_VERIFY = _ss_os.environ.get("TLLM_FMHA_PLAN_SS_VERIFY", "0") == "1"  # gate arm only (implies a main-stream sync)


_SS_CC = {}


def _ss_index(device):
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def _ss_is_sm107(device):
    idx = _ss_index(device)
    cc = _SS_CC.get(idx)
    if cc is None:
        cc = _SS_CC[idx] = torch.cuda.get_device_capability(idx)
    return cc == (10, 7)


def _ss_stream(device):
    idx = _ss_index(device)
    st = _SS_STREAMS.get(idx)
    if st is None:
        lo, hi = torch.cuda.Stream.priority_range()
        st = _SS_STREAMS[idx] = torch.cuda.Stream(device=idx, priority=min(lo, hi))
    return st


def _ss_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _ss_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _ss_tensors(v)


def _ss_same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
                and a.dtype == b.dtype and torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_ss_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_ss_same(x, y) for x, y in zip(a, b))
    return a == b


def _ss_verify(side_out, ref_out):
    st = _SS_STATS
    st["vn"] += 1
    ok = True
    if isinstance(side_out, dict) and isinstance(ref_out, dict):
        for k in side_out:
            if k in ("cute_workspace_buffer", "workspace_o"):
                continue  # scratch buffers the plan never writes
            if not _ss_same(side_out.get(k), ref_out.get(k)):
                ok = False
                _ss_sys.stderr.write(f"[fmha_sm100] SS VERIFY MISMATCH key={k}\n")
    else:
        ok = _ss_same(side_out, ref_out)
    if not ok:
        st["vbad"] += 1
    if st["vn"] % 200 == 0 or not ok:
        _ss_sys.stderr.write(f"[fmha_sm100] SS VERIFY calls {st['vn']} mismatches {st['vbad']}\n")


def _ss_eligible(b):
    a = b.arguments
    if not _ss_is_sm107(a.get("device")):
        return False  # validated on SM107 only: SM100 / SM103 keep the stock current-stream plan
    for name in ("qo_segment_lens", "kv_segment_lens", "qo_offset"):
        v = a.get(name)
        if isinstance(v, torch.Tensor) and v.is_cuda:
            return False  # produced on a CUDA stream: reading it on the plan stream would race
    if torch.cuda.is_current_stream_capturing():
        return False
    kv_block_num = a.get("kv_block_num", -1)
    mode = a.get("sparse_kernel_mode", "auto")
    lens = a.get("qo_segment_lens")
    mq = int(lens.max()) if isinstance(lens, torch.Tensor) and lens.numel() else 0
    if kv_block_num > 0 and (mode == "prefill" or (mode == "auto" and mq > _prefill_qlen_threshold(True))):
        return False  # sparse_fmha_plan branch: not audited for the side stream
    return True


def _ss_tl_report(tag):
    st = _SS_STATS
    st["n"] += 1
    if st["n"] % 50 == 0 and st["tl"]:
        t = sorted(st["tl"]); st["tl"] = []
        _ss_sys.stderr.write(f"[fmha_sm100] {tag}: last {len(t)} plan_cost.tolist() host waits ms p50 "
                             f"{t[len(t)//2]:.3f} p99 {t[min(len(t)-1, int(0.99*len(t)))]:.3f} max {t[-1]:.3f}\n")


def _fmha_sm100_plan(*args, **kwargs):
    if not _SS_ON:
        out = _fmha_sm100_plan_impl(*args, **kwargs)
        _ss_tl_report("stock-stream plan (kill switch)")
        return out
    b = _SS_SIG.bind(*args, **kwargs)
    b.apply_defaults()
    if not _ss_eligible(b):
        _SS_STATS["tl"].clear()  # never reported on this path (e.g. off SM107): keep the sample list bounded
        return _fmha_sm100_plan_impl(*args, **kwargs)
    dev = b.arguments.get("device")
    caller = b.arguments.get("stream")
    consumer = torch.cuda.ExternalStream(caller) if isinstance(caller, int) and caller else torch.cuda.current_stream()
    side = _ss_stream(dev)
    b.arguments["stream"] = side.cuda_stream
    t0 = _ss_time.perf_counter()
    with torch.cuda.stream(side):
        out = _fmha_sm100_plan_impl(*b.args, **b.kwargs)
    dt = (_ss_time.perf_counter() - t0) * 1e3
    consumer.wait_stream(side)
    for t in _ss_tensors(out):
        if t.is_cuda:
            t.record_stream(consumer)
    if _SS_VERIFY:
        _ss_verify(out, _fmha_sm100_plan_impl(*args, **kwargs))
    st = _SS_STATS
    st["n"] += 1
    st["waits"].append(dt)
    if not st["logged"]:
        st["logged"] = True
        _ss_sys.stderr.write("[fmha_sm100] plan decision on side stream taken (high priority)\n")
    if st["n"] % 50 == 0:
        w = sorted(st["waits"]); st["waits"] = []
        t = sorted(st["tl"]); st["tl"] = []
        if t:
            _ss_sys.stderr.write(f"[fmha_sm100] side-stream plan: last {len(t)} plan_cost.tolist() host waits ms p50 "
                                 f"{t[len(t)//2]:.3f} p99 {t[min(len(t)-1, int(0.99*len(t)))]:.3f} max {t[-1]:.3f}\n")
        _ss_sys.stderr.write(f"[fmha_sm100] side-stream plan: last {len(w)} calls, plan wall ms (upper bound of the plan_cost read wait) p50 {w[len(w)//2]:.3f} "
                             f"p99 {w[min(len(w)-1, int(0.99*len(w)))]:.3f} max {w[-1]:.3f}\n")
    return out

def _fmha_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    o_scale: Optional[float] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    check_input_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    if plan_info["MM-SA-Nv"]:
        return sparse_fmha(q=q, k=k, v=v, plan_info=plan_info, out=out, max_score=max_score, 
                sm_scale=sm_scale, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale, o_scale=o_scale,
                kv_indices=kv_indices, output_maxscore=output_maxscore, output_o=output_o, q_offset_override=q_offset_override,
                kv_block_indexes=kv_block_indexes, check_input_valid=check_input_valid)


    nnz_qo, num_qo_heads, head_dim_qk = q.shape
    if kv_indices is None:
        nnz_kv, num_kv_heads, head_dim_vo = v.shape
        is_paged = False
    else:
        nnz_kv, num_kv_heads, page_size, head_dim_vo = v.shape
        is_paged = True

    qo_total_len = nnz_qo

    packed_work_range = plan_info["packed_work_range"]
    packed_work_info = plan_info["packed_work_info"]
    kv_tile_begin_indices = plan_info["kv_tile_begin_indices"]
    kv_tile_end_indices = plan_info["kv_tile_end_indices"]
    kv_split_indices = plan_info["kv_split_indices"]
    num_kv_splits = plan_info["num_kv_splits"]
    workspace_o = plan_info["workspace_o"]
    workspace_lse = plan_info["workspace_lse"]
    plan_max_qo_len = plan_info["max_qo_len"]
    num_kv_splits_per_row = plan_info["num_kv_splits_per_row"]
    qo_segment_offsets = plan_info["qo_segment_offsets"]
    kv_segment_offsets = plan_info["kv_segment_offsets"]
    kv_page_indptr = plan_info["kv_page_indptr"]
    max_k_tiles = plan_info["max_k_tiles"]
    qo_segment_lens = plan_info["qo_segment_lens"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    qo_offset = plan_info["qo_offset"] if q_offset_override is None else q_offset_override
    pack_factor = plan_info["pack_factor"]
    orig_num_qo_heads = plan_info["orig_num_qo_heads"]
    qo_len_uniform = plan_info["qo_len_uniform"]
    workspace_buffer = plan_info["cute_workspace_buffer"]

    batch_size = qo_segment_lens.shape[0]

    if isinstance(qo_offset, int):
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)
    elif qo_offset is not None:
        assert qo_offset.device == qo_segment_lens.device

    if check_input_valid:
        _validate_fmha_inputs(
            q, k, v, qo_segment_lens, kv_segment_lens,
            num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
            batch_size, qo_total_len, is_paged,
            kv_indices, kv_block_indexes, qo_offset,
            page_size if is_paged else 0,
            pack_factor=pack_factor,
            packed_work_range=packed_work_range,
            packed_work_info=packed_work_info,
            num_kv_splits=num_kv_splits,
            kv_tile_begin_indices=kv_tile_begin_indices,
            kv_tile_end_indices=kv_tile_end_indices,
            kv_split_indices=kv_split_indices,
            qo_segment_offsets=qo_segment_offsets,
            kv_page_indptr=kv_page_indptr,
        )

    if pack_factor > 1 and orig_num_qo_heads is not None:
        num_qo_heads = orig_num_qo_heads // pack_factor
        qo_total_len = nnz_qo * pack_factor

    max_qo_len = plan_max_qo_len
    qo_tile_size = plan_info.get("qo_tile_size") or (128 if max_qo_len <= 128 else 256)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
    if q_scale is None:
        q_scale = 1.0
    if k_scale is None:
        k_scale = 1.0
    if v_scale is None:
        v_scale = 1.0
    if o_scale is None:
        o_scale = 1.0

    assert output_o or output_maxscore

    use_split_kv = (num_kv_splits > 1 and workspace_o is not None)

    if not output_maxscore or max_k_tiles == -1:
        max_score = None
        output_o = True
    elif max_score is None and max_k_tiles > 0:
        max_score = torch.full(
            (orig_num_qo_heads, max_k_tiles, nnz_qo),
            -float("inf"), dtype=torch.float32, device=q.device)

    if not output_o:
        out = None
    elif out is None:
        out_dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype
        out = torch.empty(
            nnz_qo,
            orig_num_qo_heads, head_dim_vo,
            device=q.device, dtype=out_dtype,
        )

    # Determine variant dispatch parameters
    dtype_code = _dlpack_dtype_code(q.dtype)

    kv_block_indexes_ptr_exists = kv_block_indexes is not None
    max_score_exists = max_score is not None
    o_exists = out is not None
    if kv_block_indexes_ptr_exists:
        sparse_mode = 0
    elif max_score_exists and o_exists:
        sparse_mode = 1
    elif max_score_exists:
        sparse_mode = 2
    else:
        sparse_mode = 3

    variant_page_size = (k.shape[2] if is_paged else -1)

    variant_module = get_fmha_variant(
        dtype_code, qo_tile_size, (max_qo_len <= 64),
        sparse_mode, variant_page_size, use_split_kv, pack_factor)

    variant_module.run(
        workspace_buffer,
        q, k, v,
        qo_segment_lens, kv_segment_lens,
        qo_segment_offsets, kv_segment_offsets,
        packed_work_range, packed_work_info,
        out,
        sm_scale, q_scale, k_scale, v_scale, o_scale,
        max_qo_len,
        qo_offset,
        num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        workspace_o, workspace_lse,
        num_kv_splits_per_row,
        qo_tile_size,
        kv_indices, kv_page_indptr,
        max_score,
        max_k_tiles,
        kv_block_indexes,
        pack_factor,
        bool(qo_len_uniform),
        torch.cuda.current_stream().cuda_stream,
    )

    # Split-KV reduction
    if use_split_kv and out is not None:
        log2_e = math.log2(math.exp(1.0))
        scale_softmax_log2 = float(q_scale * k_scale * sm_scale) * log2_e
        inv_scale_o = float(o_scale)

        reduction_module = get_reduction_module()
        reduction_module.reduction(
            workspace_o,
            out,
            workspace_lse,
            num_kv_splits_per_row,
            scale_softmax_log2, inv_scale_o,
            num_kv_splits, qo_total_len, num_qo_heads, head_dim_vo,
            num_qo_heads * head_dim_vo, head_dim_vo,
            num_qo_heads * head_dim_vo, head_dim_vo,
            orig_num_qo_heads,
            num_kv_heads,
            pack_factor,
            torch.cuda.current_stream().cuda_stream,
        )

    return out, max_score


def _mixed_batch_split(qo_segment_lens, split_threshold, *, enable, sparse):
    """Row index splitting a mixed prefill/decode batch, or 0 for no split.

    Returns ``split`` such that rows ``[0, split)`` and ``[split, batch)`` are
    each uniformly prefill or uniformly decode. Both halves must be contiguous
    row ranges because that is all the run path can slice.

    Returns 0 when the batch is uniform, when the two classes interleave, or
    when splitting is disabled.

    Both orderings are recognised:

    * **Decode-first** (prefill rows form a suffix) is this module's original
      convention and is honoured for every plan flavour, so existing callers
      keep their exact behaviour.
    * **Prefill-first** (prefill rows form a prefix), which is what
      TensorRT-LLM produces, is honoured only for ``sparse`` plans. The dense
      flavour already handles a mixed batch in one call, whereas the sparse
      MM-SA-Nv prefill kernel has no mixed path and would otherwise drag every
      decode row through it. Confining it to sparse plans also leaves the proxy
      and dense plans byte-identical to their pre-split behaviour, which
      matters because the proxy's max-score output drives top-k block
      selection.
    """
    batch_size = int(qo_segment_lens.shape[0])
    if not enable or batch_size < 2:
        return 0
    is_prefill = qo_segment_lens > split_threshold
    n_prefill = int(is_prefill.sum())
    if n_prefill == 0 or n_prefill == batch_size:
        return 0
    if bool(is_prefill[batch_size - n_prefill:].all()):
        return batch_size - n_prefill
    if sparse and bool(is_prefill[:n_prefill].all()):
        return n_prefill
    # Interleaved: neither half would be uniform, so plan the batch as one.
    return 0


def _cache_split_boundaries(sub_lo):
    """Precompute the boundaries the run path slices a split batch at.

    Both are properties of the low sub-plan alone, so they are constant for the
    whole step. Reading them at plan time keeps the device-to-host read off the
    per-layer path, where a sparse model would otherwise repeat it once per
    layer.
    """
    offsets = sub_lo.get("qo_segment_offsets")
    if offsets is not None:
        sub_lo["_split_nnz"] = int(offsets[-1].item()) // sub_lo.get("pack_factor", 1)
    if "kv_page_indptr" in sub_lo:
        sub_lo["_split_kv_pages"] = int(sub_lo["kv_page_indptr"][-1].item())
    elif "total_rows" in sub_lo:
        sub_lo["_split_kv_pages"] = int(sub_lo["total_rows"])


def fmha_sm100_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *args,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    split_prefill_decode = True,
    **kwargs
):
    """Build a reusable execution plan for ``fmha_sm100``.

    The plan is shape-dependent and can be reused across layers or repeated
    calls that share the same sequence lengths, head counts, page size, sparse
    mode, and output mode.  Planning may run CUDA kernels and allocates
    workspaces, so it should be done outside tight per-layer loops when
    possible.

    Parameters
    ----------
    qo_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request Q/O lengths.
        CPU tensors are accepted for the dense planner; sparse prefill planning
        moves them to CUDA internally.
    kv_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request KV lengths.
    *args
        Positional arguments forwarded to the internal planner.  In normal
        usage this is ``num_qo_heads`` and optionally ``num_kv_heads``.
    qo_offset : int or torch.Tensor, optional
        Per-request causal offset.  If omitted, defaults to
        ``kv_segment_lens - qo_segment_lens`` for bottom-right causal masking.
        A tensor must have shape ``[batch_size]``.
    split_prefill_decode : bool, optional
        If True, a mixed batch whose decode and prefill requests form two
        contiguous groups is split into two sub-plans.  Either group may come
        first; see ``_mixed_batch_split`` for which orderings apply to which
        plan flavours.  A batch that interleaves the two classes is planned as
        one.
    **kwargs
        Planner options forwarded to ``_fmha_sm100_plan``.  Common options are
        ``num_kv_heads``, ``num_kv_splits``, ``page_size``,
        ``output_maxscore``, ``kv_block_num``, ``usable_SM_count``, ``causal``,
        ``sparse_kernel_mode``, ``use_fp8_kvcache``, ``device``, and ``stream``.

    Returns
    -------
    tuple
        ``(has_mixed_prefill, split, batch_size, sub_plan_lo, sub_plan_hi)``,
        where ``sub_plan_lo`` describes batch rows ``[0, split)`` and
        ``sub_plan_hi`` describes rows ``[split, batch_size)``.  Which of the
        two is the prefill half depends on the caller's ordering, so consumers
        must test a sub-plan's own tags rather than its position.
        ``sub_plan_hi`` is None when the batch was not split.  Pass this tuple
        unchanged as ``plan_info`` to ``fmha_sm100``.
    """

    # assert qo_segment_lens.device.type == 'cpu' \
    #         and kv_segment_lens.device.type == 'cpu'
    # assert qo_offset is None or isinstance(qo_offset, int) or qo_offset.device.type == 'cpu'

    if qo_offset is None:
        qo_offset = kv_segment_lens - qo_segment_lens
    elif isinstance(qo_offset, int):
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)

    batch_size = qo_segment_lens.shape[0]
    sparse = kwargs.get("kv_block_num", -1) > 0
    split_threshold = _prefill_qlen_threshold(sparse)
    split = _mixed_batch_split(
        qo_segment_lens, split_threshold, enable=split_prefill_decode, sparse=sparse
    )
    if split > 0:
        sub_lo = _fmha_sm100_plan(qo_segment_lens[:split], kv_segment_lens[:split], *args,
                                    qo_offset=qo_offset[:split], **kwargs)
        sub_lo = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sub_lo.items()}
        sub_hi = _fmha_sm100_plan(qo_segment_lens[split:], kv_segment_lens[split:], *args,
                                    qo_offset=qo_offset[split:], **kwargs)
        _cache_split_boundaries(sub_lo)
        return (True, split, batch_size, sub_lo, sub_hi)
    else:
        plan = _fmha_sm100_plan(qo_segment_lens, kv_segment_lens, *args,
                                    qo_offset=qo_offset, **kwargs)
        return (False, 0, batch_size, plan, None)

def fmha_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run dense, paged, or sparse SM100 FMHA using a precomputed plan.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_qo_len, num_qo_heads, head_dim]``.  Supported dtypes are
        ``torch.bfloat16`` and ``torch.float8_e4m3fn``.  ``head_dim`` must be
        128.
    k : torch.Tensor
        Dense layout ``[total_kv_len, num_kv_heads, head_dim]`` or paged layout
        ``[total_pages, num_kv_heads, page_size, head_dim]``.
    v : torch.Tensor
        Same layout as ``k``.  The output head dimension follows ``v.shape[-1]``.
    plan_info : tuple
        Return value from ``fmha_sm100_plan`` for the same lengths, head layout,
        page size, and sparse/output mode.
    kv_indices : torch.Tensor, optional
        Paged-KV physical page table, flattened across the batch.  Required when
        ``k`` and ``v`` use paged layout.  Shape is ``[sum_pages]`` and dtype is
        int32.
    kv_block_indexes : torch.Tensor, optional
        Sparse KV block indices from ``sparse_topk_select``.  Shape
        ``[total_qo_len, num_kv_heads or num_qo_heads, kv_block_num]``, dtype
        int32, ascending per row with ``-1`` padding at the tail.
    q_offset_override : int or torch.Tensor, optional
        Runtime causal-offset override.  Tensor form must have shape
        ``[batch_size]`` on the same CUDA device as the plan metadata.  The
        override must stay within the causal range visible to the original
        plan.
    out : torch.Tensor, optional
        Preallocated output buffer with shape
        ``[total_qo_len, num_qo_heads, head_dim_v]``.
    max_score : torch.Tensor, optional
        Preallocated per-KV-tile score buffer with shape
        ``[num_qo_heads, max_k_tiles, total_qo_len]`` and dtype float32.
    **kwargs
        Runtime options forwarded to the kernel runner.  Common options are
        ``sm_scale``, ``q_scale``, ``k_scale``, ``v_scale``, ``o_scale``,
        ``output_maxscore``, ``output_o``, and ``check_input_valid``.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(out, max_score)``.  Either item may be ``None`` if the corresponding
        output was disabled.  When the plan was split into two sub-plans, each
        half writes its own rows of ``out`` directly; if no ``out`` was supplied
        the halves are concatenated back into the original batch order instead.
    """
    has_mixed_prefill, split, batch_size, decode, prefill = plan_info
    if not has_mixed_prefill:
        return _fmha_sm100(q, k, v, decode, out=out, max_score=max_score, kv_indices=kv_indices,kv_block_indexes=kv_block_indexes, q_offset_override=q_offset_override, **kwargs)
    else:
        # The `decode` / `prefill` names are historical: these are really the
        # sub-plans for batch rows [0, split) and [split, batch_size), and which
        # one holds the prefill requests depends on the caller's ordering. None
        # of the slicing below needs to know, because every boundary is derived
        # from the low sub-plan's own offsets.
        decode_nnz = decode.get("_split_nnz")
        if decode_nnz is None:
            decode_pack = decode.get("pack_factor", 1)
            decode_nnz = decode["qo_segment_offsets"][-1].item() // decode_pack
        is_paged = kv_indices is not None
        nnz_qo = q.shape[0]
        num_qo_heads = q.shape[1]

        q_decode = q[:decode_nnz]
        q_prefill = q[decode_nnz:]

        if is_paged:
            k_decode, v_decode = k, v
            k_prefill, v_prefill = k, v
            kv_page_split = decode.get("_split_kv_pages")
            if kv_page_split is None:
                if "kv_page_indptr" in decode:
                    kv_page_split = decode["kv_page_indptr"][-1].item()
                else:
                    kv_page_split = decode["total_rows"]
            decode_kv_indices = kv_indices[:kv_page_split]
            prefill_kv_indices = kv_indices[kv_page_split:]
        else:
            if "kv_segment_offsets" in decode:
                decode_kv_nnz = decode["kv_segment_offsets"][-1].item()
            else:
                decode_kv_nnz = decode["cu_seqlens_k"][-1].item()
            k_decode, k_prefill = k[:decode_kv_nnz], k[decode_kv_nnz:]
            v_decode, v_prefill = v[:decode_kv_nnz], v[decode_kv_nnz:]
            decode_kv_indices = None
            prefill_kv_indices = None

        decode_block_idx = kv_block_indexes[:decode_nnz] if kv_block_indexes is not None else None
        prefill_block_idx = kv_block_indexes[decode_nnz:] if kv_block_indexes is not None else None
        if isinstance(q_offset_override, int):
            decode_qo_offset = torch.full((split,), q_offset_override, dtype=torch.int32, device=q.device)
            prefill_qo_offset = torch.full((batch_size - split,), q_offset_override, dtype=torch.int32, device=q.device)
        elif q_offset_override is not None:
            decode_qo_offset = q_offset_override[:split]
            prefill_qo_offset = q_offset_override[split:]
        else:
            decode_qo_offset = None
            prefill_qo_offset = None

        # Let each half write straight into its own rows of the caller's buffer.
        # Row views of a contiguous [nnz, H, D] output are themselves
        # contiguous, which is what the kernels require, so this removes a
        # full-size concatenate plus copy from every call.
        write_in_place = out is not None and out.is_contiguous()

        # ---- Run kernels ----
        decode_out, decode_ms = _fmha_sm100(
            q_decode, k_decode, v_decode, decode,
            out=out[:decode_nnz] if write_in_place else None, max_score=None,
            kv_indices=decode_kv_indices, kv_block_indexes=decode_block_idx,
            q_offset_override=decode_qo_offset,
            **kwargs)
        prefill_out, prefill_ms = _fmha_sm100(
            q_prefill, k_prefill, v_prefill, prefill,
            out=out[decode_nnz:] if write_in_place else None, max_score=None,
            kv_indices=prefill_kv_indices, kv_block_indexes=prefill_block_idx,
            q_offset_override=prefill_qo_offset,
            **kwargs)

        # ---- Merge out ----
        if write_in_place:
            combined_out = out
        elif decode_out is not None and prefill_out is not None:
            combined_out = torch.cat([decode_out, prefill_out], dim=0)
        else:
            combined_out = None

        if out is not None and combined_out is not None and combined_out is not out:
            out.copy_(combined_out)

        # ---- Merge max_score ----
        if decode_ms is not None and prefill_ms is not None:
            d_kt, p_kt = decode_ms.shape[1], prefill_ms.shape[1]
            max_kt = max(d_kt, p_kt)
            combined_ms = torch.full(
                (num_qo_heads, max_kt, nnz_qo),
                -float("inf"), dtype=torch.float32, device=q.device)
            combined_ms[:, :d_kt, :decode_nnz] = decode_ms
            combined_ms[:, :p_kt, decode_nnz:] = prefill_ms
        else:
            combined_ms = decode_ms if decode_ms is not None else prefill_ms

        if max_score is not None and combined_ms is not None:
            max_score.copy_(combined_ms)

        return (out if out is not None else combined_out,
                max_score if max_score is not None else combined_ms)


# ============================================================================
# Sparse TopK Select
# ============================================================================


def sparse_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    r"""Select top-k KV-tile indices per (qo_head, token) row from the FMHA max-score tensor.

    Designed for the MQA proxy-KV sparse attention path where the dense pass uses
    ``num_kv_heads_dense=1``, so ``max_score.shape[0] == num_kv_heads_real`` and
    each head row is processed independently (no GQA reduction inside this call).

    Parameters
    ----------
    max_score : torch.Tensor
        Shape ``(num_qo_heads, max_k_tiles, total_qo_len)``, contiguous, float32.
        Slots beyond the actual KV tile count must be pre-filled with ``-inf``
        (fmha_sm100 does this automatically via ``torch.full``).
    topk : int
        Must be exactly 16.
    num_valid_pages : int, optional
        Actual number of KV pages in the page table, i.e. ``ceil(kv_len / page_size)``.
        ``max_k_tiles`` is round-up-aligned and always >= ``num_valid_pages``.
        The kernel may select tile indices in ``[num_valid_pages, max_k_tiles-1]``
        (all-``-inf`` padding tiles). Passing ``num_valid_pages`` replaces those
        out-of-range indices with ``-1`` and sorts them to the tail, matching the
        sparse FMHA kernel's kv_block_indexes contract.
        **Strongly recommended**: omitting this allows OOB page-table accesses in
        the sparse attention pass.
    force_begin_blocks : int
        Number of KV blocks at the beginning of the sequence (indices 0..N-1) to
        always include in the top-k result, regardless of their scores.  Useful
        for sink tokens.  Default 0.
    force_end_blocks : int
        Number of KV blocks at the end of the valid sequence (indices
        nvp-N..nvp-1, closest to the current query) to always include.  Useful
        for local-window attention.  Default 0.

    Returns
    -------
    torch.Tensor
        Shape ``(total_qo_len, num_qo_heads, topk)``, int32, ascending by tile index.
        Out-of-range entries (if any) are ``-1`` at the tail.
    """

    assert max_score.dtype == torch.float32, f"max_score must be float32, got {max_score.dtype}"
    assert max_score.dim() == 3, f"max_score must be 3D, got {max_score.shape}"
    assert max_score.is_contiguous(), "max_score must be contiguous"
    assert topk == 16, f"topk must be 16, got {topk}"

    num_qo_heads, max_k_tiles, total_qo_len = max_score.shape

    # v2.3 kernel only supports the insertion-sort path (K < 12288).
    assert max_k_tiles < 12288, (
        f"max_k_tiles={max_k_tiles} >= 12288: v2.3 kernel only supports K < 12288 "
        f"(radix-sort path not yet implemented). kv_len must be < {12288 * 128} tokens."
    )

    if num_valid_pages is not None:
        assert 0 < num_valid_pages <= max_k_tiles, (
            f"num_valid_pages={num_valid_pages} must be in (0, max_k_tiles={max_k_tiles}]"
        )
        nvp_arg = int(num_valid_pages)
    else:
        # v2.5_oob_clamp_in_kernel: kernel takes a unified num_valid_pages arg.
        # When the caller doesn't supply one, pass max_k_tiles so the in-kernel
        # `idx >= num_valid_pages` check never triggers (idx is always in
        # [0, max_k_tiles)).
        nvp_arg = int(max_k_tiles)

    assert force_begin_blocks >= 0 and force_end_blocks >= 0, (
        f"force_begin_blocks={force_begin_blocks} and force_end_blocks={force_end_blocks} "
        f"must be non-negative"
    )
    assert force_begin_blocks + force_end_blocks <= topk, (
        f"force_begin_blocks({force_begin_blocks}) + force_end_blocks({force_end_blocks}) "
        f"= {force_begin_blocks + force_end_blocks} exceeds topk={topk}"
    )

    # Workspace = transpose_buf only: (num_qo_heads, max_k_tiles, total_qo_len) fp32.
    workspace_size = num_qo_heads * max_k_tiles * total_qo_len  # int32 elements

    workspace_buffer = _alloc_workspace_buf(_BuffTag.sparse_topk_workspace, workspace_size, max_score.device, torch.int32)
    
    if output is not None:
        output_indices = output
    else:
        output_indices = torch.empty(
            total_qo_len, num_qo_heads, topk,
            dtype=torch.int32, device=max_score.device,
        )

    module = get_sparse_topk_module()
    # MQA dense pass: num_kv_heads_dense=1, so h_r=num_qo_heads.
    # The kernel only uses num_qo_heads = h_r * num_kv_heads as a product;
    # passing num_kv_heads=1 is equivalent to any other valid factorisation.
    #
    # v2.5_oob_clamp_in_kernel: OOB clamp is folded into the kernel — the prior
    # post-process torch.where + sort + torch.where chain (~84-101 us / call)
    # is replaced by passing num_valid_pages directly to the kernel.
    module.sparse_topk_select(
        max_score, output_indices, workspace_buffer,
        topk,
        nvp_arg,
        int(force_begin_blocks),
        int(force_end_blocks),
        torch.cuda.current_stream().cuda_stream,
    )

    return output_indices
