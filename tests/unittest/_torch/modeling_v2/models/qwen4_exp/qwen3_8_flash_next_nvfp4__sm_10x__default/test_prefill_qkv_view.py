# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prefill sparse attention of a context-only step reads and writes in place.

Over an unscaled (bf16 or fp8) pool a sparse context-only step reads its query heads through
the fused projection's rows (`PrefillTarget._attention_core`) instead of a compact `[q | k |
v]` copy, and a row-tile launch writes its rows of `attn_out` itself (`rowtile_out`) instead of
a fresh tensor copied afterwards. Both kernels take q's and the output's strides, so the
answer is bitwise the compact copy's, on the row-tile pass and on the per-row kernel below its
row threshold.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp import archs
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

# The row-tile pass and the per-row kernel run on the GPUs the family serves (their `ARCHS`).
_served_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not archs.served(torch.cuda.get_device_capability()),
    reason=f"the sparse attention kernels run on {archs.served_names()}",
)
DEV = "cuda"

TPB, D, RATIO, TOPK = 32, 256, 4, 2048
QH, HKV = 6, 1
# The fused projection row past q | k | v: the gate and the index columns (any width that
# keeps rows 16-byte aligned).
EXTRA = QH * D + 768
# (cached, new) tokens a context request.
SEGMENTS = ((0, 300), (900, 200), (1400, 100))


def _step(pool_dtype, seed: int = 11):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    kv_lens = [s + n for s, n in SEGMENTS]
    pages = [math.ceil(k / TPB) for k in kv_lens]
    pool_pages = sum(pages) + 5
    shape = (pool_pages, HKV, TPB, D)
    k = (torch.randn(shape, generator=gen, device=DEV) * 0.5).to(pool_dtype)
    v = (torch.randn(shape, generator=gen, device=DEV) * 0.5).to(pool_dtype)
    slots = 2 * len(SEGMENTS)
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))
    slot_of = slot_of[: len(SEGMENTS)].to(DEV, torch.int32)
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, max(pages)), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), pages):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    new = torch.tensor([n for _, n in SEGMENTS], dtype=torch.long, device=DEV)
    logical = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in SEGMENTS])
    rows = logical.numel()
    column = torch.arange(TOPK + RATIO - 1, device=DEV)[None, :]
    selected = torch.where(column <= logical[:, None], column, -1).to(torch.int32)
    qkv_width = (QH + 2 * HKV) * D
    proj = torch.randn(rows, qkv_width + EXTRA, generator=gen, device=DEV).to(torch.bfloat16)
    core = SimpleNamespace(
        heads_q=QH,
        heads_kv=HKV,
        head_dim=D,
        qkv_width=qkv_width,
        q_width=QH * D,
        softmax_scale=D**-0.5,
        compress_ratio=RATIO,
    )
    return dict(
        core=core,
        kv=SimpleNamespace(k=k, v=v, k_scale=None, v_scale=None),
        block_table=block_table,
        req_idx=slot_of.repeat_interleave(new),
        proj=proj,
        selected=selected,
        logical=logical.to(torch.int32),
        rows=rows,
    )


def _attend(step, qkv, attn_out, in_place):
    """`_attention_core`'s context launch over all rows, as the target runs it."""
    core = step["core"]
    rows = step["rows"]
    direct = attn_out.view(rows, core.heads_q, core.head_dim) if in_place else None
    out = M._sparse_gqa(
        core,
        step["kv"],
        M._packed_q(core, qkv),
        step["block_table"],
        step["selected"],
        step["req_idx"],
        TPB,
        step["logical"],
        None,
        rowtile_out=direct,
    )
    if direct is None or out.data_ptr() != direct.data_ptr():
        attn_out.copy_(torch.reshape(out, [rows, core.q_width]))
    return out


@_served_gpu
@pytest.mark.parametrize("pool_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("min_rows", [64, 1 << 20], ids=["row_tile", "per_row"])
def test_q_view_and_in_place_output_are_bitwise_the_compact_copy(
    monkeypatch, pool_dtype, min_rows
) -> None:
    monkeypatch.setattr(M, "_ROWTILE_MIN_ROWS", min_rows)
    step = _step(pool_dtype)
    core, proj, rows = step["core"], step["proj"], step["rows"]
    compact = proj[:, : core.qkv_width].clone()
    want = torch.full((rows, core.q_width), float("nan"), dtype=torch.bfloat16, device=DEV)
    _attend(step, compact, want, in_place=False)

    view = proj[:, : core.qkv_width]
    assert view.data_ptr() == proj.data_ptr() and view.stride(0) == proj.stride(0)
    got = torch.full_like(want, float("nan"))
    out = _attend(step, view, got, in_place=True)

    # Both start as NaN and the inputs are finite with a resident token a row: a row left
    # unwritten, or a NaN both launches produce, fails here rather than matching bitwise.
    assert torch.isfinite(want).all() and torch.isfinite(got).all()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # The row-tile pass wrote `attn_out` itself; the per-row kernel returns its own tensor.
    assert (out.data_ptr() == got.data_ptr()) == (min_rows == 64)


@_served_gpu
def test_the_row_tile_entry_refuses_an_out_it_cannot_write(monkeypatch) -> None:
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.attention.qsa_sparse_gqa_rowtile__prefill import (  # noqa: E501
        qsa_sparse_gqa_rowtile as op,
    )

    step = _step(torch.bfloat16)
    core, rows = step["core"], step["rows"]
    args = dict(
        q=M._packed_q(core, step["proj"][:, : core.qkv_width]),
        k_cache=step["kv"].k,
        v_cache=step["kv"].v,
        block_table=step["block_table"],
        selected_tokens=step["selected"],
        request_indices=step["req_idx"],
        tokens_per_block=TPB,
        softmax_scale=core.softmax_scale,
        query_positions=step["logical"],
        compress_ratio=RATIO,
    )
    strided = torch.empty(rows, core.heads_q, 2 * D, dtype=torch.bfloat16, device=DEV)[..., :D]
    with pytest.raises(ValueError, match="contiguous"):
        op(**args, out=strided)
    with pytest.raises(ValueError, match="contiguous"):
        op(**args, out=torch.empty(rows, core.heads_q, D, dtype=torch.float32, device=DEV))
