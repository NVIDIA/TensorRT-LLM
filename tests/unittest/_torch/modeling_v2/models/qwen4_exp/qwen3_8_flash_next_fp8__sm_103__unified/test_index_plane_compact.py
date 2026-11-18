# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact QSA index pages (`index_plane_layout`) against plain ones: the same steps (chunked
prefill of odd lengths across page edges, then verify windows) written both ways give bitwise the
same compressed keys, coordinates and index scores (family Triton kernels and both CuTe scorers)."""

import random
from importlib import import_module

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.qsa import index_plane_layout as ipl
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (  # noqa: E501
    qsa_paged_index_scores,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_prefill_compress import (  # noqa: E501
    qsa_prefill_compress,
)

U = "tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified"
store_mod = import_module(U + ".ops.attention.qsa_paged_index_store")
compact_mod = import_module(U + ".ops.attention.qsa_index_compact_plane")
cute_pre = import_module(U + ".ops.attention.qsa_index_scores_cute__prefill")
cute_dec = import_module(U + ".ops.attention.qsa_index_scores_groups_cute__decode")

DEV = "cuda"
TPB, CR, D = 32, 4, 128
ROWS_C = ipl.plane_rows(TPB, CR, compact=True)


def test_row_map_windows_are_distinct():
    """Any 5 consecutive positions map to distinct rows; a group's raw rows survive the next
    group; slot_rows agrees with row_of."""
    assert ROWS_C == 14
    for start in range(0, 200):
        rows = [ipl.row_of(p, TPB, CR, True) for p in range(start, start + CR + 1)]
        assert len(set(rows)) == len(rows), (start, rows)
        assert all(0 <= r < ROWS_C for r in rows)
    slots = torch.arange(TPB)
    mapped = ipl.slot_rows(slots, TPB, CR, ROWS_C)
    assert mapped.tolist() == [ipl.row_of(int(s), TPB, CR, True) for s in slots]
    assert ipl.slot_rows(slots, TPB, CR, TPB).tolist() == slots.tolist()


def _run(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    nreq, width, pages = 6, 96, 6 * 96 + 8
    perm = torch.randperm(pages - 8, device=DEV).to(torch.int32) + 4
    table = perm[: nreq * width].reshape(nreq, width).contiguous()
    plain = torch.zeros(pages, TPB, 1, D, dtype=torch.bfloat16, device=DEV)
    comp = torch.zeros(pages, ROWS_C, 1, D, dtype=torch.bfloat16, device=DEV)
    pos_p = torch.zeros(pages, TPB, 3, dtype=torch.int32, device=DEV)
    pos_c = torch.zeros(pages, TPB, 3, dtype=torch.int32, device=DEV)
    knorm = torch.randn(D, device=DEV).to(torch.bfloat16)
    cos_sin = torch.randn(width * TPB + 64, D // 2, device=DEV, dtype=torch.float32)
    length = [0] * nreq
    steps = 0
    # a schedule: prefill chunks of odd sizes, then verify windows of 3 / 4 rows
    plan = []
    for _ in range(8):
        plan.append([random.choice([1, 2, 3, 5, 7, 13, 33, 100, 257, 517]) for _ in range(nreq)])
    for _ in range(12):
        plan.append([random.choice([3, 4]) for _ in range(nreq)])
    bad = 0
    for chunk in plan:
        req, logical = [], []
        for r, n in enumerate(chunk):
            req += [r] * n
            logical += list(range(length[r], length[r] + n))
        rows = len(req)
        req_t = torch.tensor(req, dtype=torch.int32, device=DEV)
        log_t = torch.tensor(logical, dtype=torch.int64, device=DEV)
        coords = torch.stack([log_t, log_t, log_t], 1).contiguous()
        if steps % 2:
            # As the engine passes them: the transpose of a `[3, capacity]` position-id buffer,
            # at an offset (aligned or not), the axis stride the capacity (a multiple of 16 or
            # not).
            offset = (steps // 2) % 4
            capacity = -(-(rows + offset) // 16) * 16 if steps % 4 == 1 else rows + offset + 5
            ids = torch.zeros(3, capacity, dtype=torch.int32, device=DEV)
            ids[:, offset : offset + rows] = coords.transpose(0, 1)
            coords = ids[:, offset : offset + rows].transpose(0, 1)
        keys = torch.randn(rows, 1, D, device=DEV).to(torch.bfloat16)
        store_mod.qsa_paged_index_store(keys, coords, plain, pos_p, req_t, log_t, table, TPB)
        qsa_prefill_compress(
            logical_positions=log_t, request_indices=req_t, block_table=table, index_cache=plain,
            position_cache=pos_p, k_norm_weight=knorm, cos_sin=cos_sin, eps=1e-6,
            tokens_per_block=TPB, compress_ratio=CR, mrope_section=None,
        )  # fmt: skip
        compact_mod.qsa_index_compact_plane(
            keys, coords, comp, pos_c, req_t, log_t, table, TPB, CR, knorm, cos_sin, 1e-6, None
        )
        for r, n in enumerate(chunk):
            length[r] += n
        torch.cuda.synchronize()
        steps += 1
        variants = tuple(
            len(k.device_caches[torch.cuda.current_device()][0])
            for k in (compact_mod._compress_kernel, compact_mod._store_kernel)
        )
        if steps == 2:
            first_variants = variants
        elif steps > 2 and variants != first_variants:
            bad += 1
            print(
                f"step {steps}: {rows} rows compiled a new variant {first_variants} -> {variants}"
            )
        # every completed group's compressed key
        for r in range(nreq):
            groups = length[r] // CR
            for b in range(groups):
                lp = (b * CR + CR - 1) // TPB
                page = int(table[r, lp])
                a = plain[page, (b * CR + CR - 1) % TPB, 0]
                c = comp[page, b % (TPB // CR), 0]
                if not torch.equal(a.view(torch.int16), c.view(torch.int16)):
                    bad += 1
                    if bad < 5:
                        print(f"step {steps} req {r} group {b}: compressed keys differ")
        if not torch.equal(pos_p, pos_c):
            bad += 1
            print(f"step {steps}: position planes differ")
        # scorers on a query row per request at its last position
        q = torch.randn(nreq, 4, D, device=DEV).to(torch.bfloat16)
        qpos = torch.tensor([length[r] - 1 for r in range(nreq)], dtype=torch.int64, device=DEV)
        qreq = torch.arange(nreq, dtype=torch.int32, device=DEV)
        sa = qsa_paged_index_scores(
            q=q, index_cache=plain, block_table=table, query_positions=qpos, request_indices=qreq,
            tokens_per_block=TPB, compress_ratio=CR, only_visible_blocks=True, context_rows=False,
        )  # fmt: skip
        sc = qsa_paged_index_scores(
            q=q, index_cache=comp, block_table=table, query_positions=qpos, request_indices=qreq,
            tokens_per_block=TPB, compress_ratio=CR, only_visible_blocks=True, context_rows=False,
        )  # fmt: skip
        vis = (qpos + 1) // CR
        inside = torch.arange(sa.shape[1], device=DEV)[None, :] < vis[:, None]
        if not torch.equal(sa[inside], sc[inside]):
            bad += 1
            print(f"step {steps}: family decode scores differ")
        if cute_pre.supports(q, plain, TPB, CR) and cute_pre.supports(q, comp, TPB, CR):
            pa = cute_pre.qsa_index_scores_prefill(q, plain, table, qpos, qreq, TPB, CR)
            pc = cute_pre.qsa_index_scores_prefill(q, comp, table, qpos, qreq, TPB, CR)
            if not torch.equal(pa[inside], pc[inside]):
                bad += 1
                print(f"step {steps}: CuTe context scores differ")
        # verify groups: 3 rows a request at the last three positions
        if min(length) >= 3:
            q3 = torch.randn(nreq * 3, 4, D, device=DEV).to(torch.bfloat16)
            p3 = torch.tensor(
                [length[r] - 3 + t for r in range(nreq) for t in range(3)], dtype=torch.int64,
                device=DEV,
            )  # fmt: skip
            r3 = torch.arange(nreq, dtype=torch.int32, device=DEV).repeat_interleave(3)
            ga = cute_dec.index_scores_groups(q3, plain, table, p3, r3, TPB, CR, False, 3)
            gc = cute_dec.index_scores_groups(q3, comp, table, p3, r3, TPB, CR, False, 3)
            v3 = (p3 + 1) // CR
            in3 = torch.arange(ga.shape[1], device=DEV)[None, :] < v3[:, None]
            if not torch.equal(ga[in3], gc[in3]):
                bad += 1
                print(f"step {steps}: CuTe verify-group scores differ")
    return bad


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compact_plane_is_bitwise_the_plain_plane(seed):
    assert _run(seed) == 0
