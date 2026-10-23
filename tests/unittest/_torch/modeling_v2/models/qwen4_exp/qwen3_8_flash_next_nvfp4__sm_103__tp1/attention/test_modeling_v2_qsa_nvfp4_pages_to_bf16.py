# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_nvfp4_pages_to_bf16 entry.

The paged state is real: one NVFP4 pool as the QSA cache manager serves it -- e2m1 data
`[pages, role pages, Hkv, tpb, D/2]` and e4m3 block scales `[pages, role pages, Hkv, tpb,
D/16]`, the layer's K and V two of the role planes -- and the engine's `[slots, width]`
table, the listed requests on seeded distinct slots, their pages a seeded permutation,
`-1` past each sequence. Scratch and table are pre-filled with sentinels, so a missing or
a stray write shows. Every cell is bitwise `reference`, and the scratch is what the NVFP4
sparse GQA kernels dequantize.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4 as _gqa,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_nvfp4_pages_to_bf16 import (  # noqa: E501
    qsa_nvfp4_pages_to_bf16 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_nvfp4_pages_to_bf16 requires a CUDA device"

DEV = "cuda"
SCRATCH_FILL = 0x7FC1  # a bf16 NaN
TABLE_FILL = -7


def _build(spec, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d = spec["tokens_per_block"], spec["kv_heads"], spec["head_dim"]
    counts = [math.ceil(n / tpb) for n in spec["kv_lens"]]
    needed = sum(counts) + 7
    pool_pages = needed + 5
    role_pages, layer = spec["slot_pages"], spec["layer"]
    data = torch.randint(
        0, 256, (pool_pages, role_pages, kv, tpb, d // 2), generator=gen, device=DEV
    ).to(torch.uint8)
    scales = torch.randint(
        0x08, 0x31, (pool_pages, role_pages, kv, tpb, d // 16), generator=gen, device=DEV
    ).to(torch.uint8)
    width, slots_total = spec["width"], spec["slots"]
    table = torch.full((slots_total, width), -1, dtype=torch.int32, device=DEV)
    slots = torch.randperm(slots_total, generator=torch.Generator().manual_seed(seed))
    slots = slots[: len(counts)]
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    cursor = 0
    for slot, count in zip(slots.tolist(), counts):
        table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    if spec["holes"]:
        first = slots[0].item()
        table[first, 1] = -1  # unmapped inside the sequence
        table[first, 2] = pool_pages + 3  # past the pool
    counts_t = torch.tensor(counts, dtype=torch.int64, device=DEV)
    listed = int(counts_t.sum())
    out_shape = (listed, kv, tpb, d)
    args = dict(
        k_cache=data[:, 2 * layer],
        v_cache=data[:, 2 * layer + 1],
        k_scale_cache=scales[:, 2 * layer],
        v_scale_cache=scales[:, 2 * layer + 1],
        kv_scale_quant_orig=torch.tensor(
            [1.0, spec["global_k"], spec["global_v"]], dtype=torch.float32, device=DEV
        ),
        block_table=table,
        slots=slots.to(DEV, torch.int32),
        counts=counts_t,
        offsets=torch.cumsum(counts_t, 0) - counts_t,
        table=torch.full_like(table, TABLE_FILL),
        k_out=torch.full(out_shape, SCRATCH_FILL, dtype=torch.int16, device=DEV).view(
            torch.bfloat16
        ),
        v_out=torch.full(out_shape, SCRATCH_FILL, dtype=torch.int16, device=DEV).view(
            torch.bfloat16
        ),
    )
    return args


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.bfloat16:
        a, b = a.view(torch.int16), b.view(torch.int16)
    return a.shape == b.shape and torch.equal(a, b)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- fixture by name
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()))
    ref = op.reference(**{k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()})
    out = op(**args)
    for name, got, want in zip(("k_out", "v_out", "table"), out, ref):
        assert _bitwise(got, want), f"{name} differs from the reference"
    # Every listed page was written, and only the listed requests' table entries.
    assert not bool(out[0].isnan().any()) and not bool(out[1].isnan().any())
    listed_rows = torch.zeros_like(args["block_table"], dtype=torch.bool)
    for slot, count in zip(args["slots"].tolist(), args["counts"].tolist()):
        listed_rows[slot, :count] = True
    assert bool((out[2][~listed_rows] == TABLE_FILL).all())


def test_scratch_is_what_the_sparse_gqa_kernels_read() -> None:
    """The scratch holds `_dequantize_tokens`' values (the NVFP4 sparse GQA entry's own
    dequantize, fp32 rounded once to bf16) for every listed page."""
    spec = op.CELLS[4].spec
    args = _build({**spec, "global_k": 0.37, "global_v": 2.5}, 11)
    k_out, v_out, table = op(**args)
    tpb = spec["tokens_per_block"]
    for slot, count, base in zip(
        args["slots"].tolist(), args["counts"].tolist(), args["offsets"].tolist()
    ):
        pages = args["block_table"][slot, :count].long()
        tokens = torch.arange(tpb, device=DEV)[None, :].expand(count, -1)
        for got, data, scales, index, grouped in (
            (k_out, args["k_cache"], args["k_scale_cache"], 1, False),
            (v_out, args["v_cache"], args["v_scale_cache"], 2, True),
        ):
            want = _gqa._dequantize_tokens(
                data,
                scales,
                pages[:, None].expand(-1, tpb),
                tokens,
                args["kv_scale_quant_orig"][index],
                grouped,
            )  # [count, tpb, H, D]
            want = want.transpose(1, 2).to(torch.bfloat16)
            assert _bitwise(got[base : base + count], want)
        assert torch.equal(
            table[slot, :count], torch.arange(base, base + count, device=DEV, dtype=torch.int32)
        )


def test_listing_past_the_scratch_is_dropped() -> None:
    """A request whose pages run past `listed` writes only those below it."""
    spec = op.CELLS[2].spec
    args = _build(spec, 5)
    short = args["k_out"].shape[0] - 3
    args["k_out"], args["v_out"] = args["k_out"][:short], args["v_out"][:short]
    ref = op.reference(**{k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()})
    out = op(**args)
    for got, want in zip(out, ref):
        assert _bitwise(got, want)


def test_guards() -> None:
    args = _build(op.CELLS[0].spec, 3)
    with validating(op), pytest.raises(AssertionError):
        op(**{**args, "k_out": args["k_out"].float()})
    with validating(op), pytest.raises(AssertionError):
        op(**{**args, "block_table": args["block_table"].long()})
    with validating(op), pytest.raises(AssertionError):
        op(**{**args, "table": args["table"][:, :-1]})
