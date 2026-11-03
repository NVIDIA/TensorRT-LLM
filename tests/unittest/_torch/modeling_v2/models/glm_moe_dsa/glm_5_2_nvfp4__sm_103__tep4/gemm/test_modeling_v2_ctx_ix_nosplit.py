# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the context indexer projection without its column split
(TRTLLM_GLM52_CTX_IX_NOSPLIT, `GlmMoeDsaTep4ModelingV2._indexer_project`).

The split path:
1. the prefill GEMM writes the fp32 `[T, 160]` product;
2. one launch splits it into bf16 keys `[T, 128]` and packed fp32 head weights `[T, 32]`;
3. the K write (`cute_dsl_indexer_k_cache_update`) and the q side
   (`cute_dsl_indexer_q_side_fused`) read those.

Without the split, the K write's `f32_input` launchers and the q side's pitch-160 launcher read
their columns of the GEMM output in place; the K write rounds the keys to bf16 on load, as the
split does. Every K-cache byte (codes, scales and the bytes around them) and every q-side output
byte must equal the split path's, at every context row count the prefill handle serves.
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernels need an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_indexer_k_cache_update as k_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_indexer_q_side_fused as q_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_indexer_pre_proj_split as proj_entry,
)

HIDDEN, KEY, HEADS, HEAD_DIM = 6144, 128, 32, 128
PITCH = KEY + HEADS
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
THETA = 8.0e6
PAGE, TOKEN_BYTES, LAYERS, LAYER = 32, 132, 4, 2
POISON = 0xA7
#: The target's rope mode.
IS_NEOX = False
#: Context row counts (the prefill handle serves 65..8192; at most 64 rows the decode
#: handles run): partial and full 128-row tiles, odd counts, the served step sizes.
ROWS = [65, 77, 127, 128, 129, 1099, 3001, 4097, 7293, 7513, 7999, 8191, 8192]

_H: dict = {}


def _handles(weight):
    if not _H:
        dev = torch.cuda.current_device()
        _H["proj"] = proj_entry.build_prefill(weight)
        _H["k"] = k_entry.build(dev, is_neox=IS_NEOX, cache_contiguous=False, f32_input=True)
        _H["q32"] = q_entry.build(dev, is_neox=IS_NEOX)
        _H["q160"] = q_entry.build(dev, is_neox=IS_NEOX, wq_pitch=PITCH)
    return _H


def _rope_k(positions: int) -> torch.Tensor:
    half = 32
    inv = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) * 2.0 / 64))
    ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).float().cuda().contiguous()


def _case(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    # Hidden states and weights of realistic scale; the products span several binades, so
    # the bf16 rounding of the keys is exercised.
    x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    weight = (torch.randn(PITCH, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(
        torch.bfloat16
    )
    kn_w = 1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")
    kn_b = 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")
    max_pos = 16384
    pos = (torch.arange(rows, device="cuda") + int(torch.randint(0, 4096, (1,), generator=cpu))).to(
        torch.int32
    )
    table = _rope_k(max_pos)
    blocks = (rows + PAGE - 1) // PAGE + 3
    flat = torch.randperm(blocks * PAGE, generator=cpu)[:rows]
    block, slot = flat // PAGE, flat % PAGE
    fp8_slot = (block * PAGE * TOKEN_BYTES + slot * HEAD_DIM).cuda()
    scale_slot = (block * PAGE * TOKEN_BYTES + PAGE * HEAD_DIM + slot * 4).cuda()
    q = (torch.randn(rows, HEADS, HEAD_DIM, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    return x, weight, kn_w, kn_b, pos, table, blocks, fp8_slot, scale_slot, q


def _run(case, nosplit: bool):
    x, weight, kn_w, kn_b, pos, table, blocks, fp8_slot, scale_slot, q = case
    h = _handles(weight)
    rows = int(x.shape[0])
    if nosplit:
        kb, wq = proj_entry.project_unsplit(x, weight, h["proj"])
        assert kb.dtype == torch.float32 and kb.stride() == (PITCH, 1)
        qside = h["q160"]
    else:
        kb = torch.empty(rows, KEY, dtype=torch.bfloat16, device="cuda")
        wq = torch.empty(rows, HEADS, dtype=torch.float32, device="cuda")
        proj_entry.cute_dsl_indexer_pre_proj_split(x, weight, kb, wq, h["proj"])
        qside = h["q32"]
    pool = torch.full((blocks, LAYERS, PAGE * TOKEN_BYTES), POISON, dtype=torch.uint8).cuda()
    k_cache = pool[:, LAYER, :].view(blocks, PAGE, 1, TOKEN_BYTES)
    k_entry.cute_dsl_indexer_k_cache_update(
        kb, kn_w, kn_b, pos, table.view(-1, 2, 32), k_cache, fp8_slot, scale_slot, rows,
        is_neox=IS_NEOX, compiled=h["k"],
    )  # fmt: skip
    q_fp8 = torch.full((rows, HEADS, HEAD_DIM), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    weights = torch.full((rows, HEADS), float("nan"), device="cuda")
    q_entry.cute_dsl_indexer_q_side_fused(
        q, pos, table, wq, WEIGHT_SCALE, q_fp8, weights, is_neox=IS_NEOX, compiled=qside
    )
    torch.cuda.synchronize()
    return pool, q_fp8.view(torch.uint8), weights


@pytest.mark.parametrize("rows", ROWS)
def test_unsplit_reads_give_the_split_bytes(rows) -> None:
    case = _case(rows, zlib.crc32(str(rows).encode()) % 2**31)
    want_pool, want_q, want_w = _run(case, nosplit=False)
    got_pool, got_q, got_w = _run(case, nosplit=True)
    # Non-degenerate: every row's codes were written, and the weights are finite and nonzero.
    assert int((want_pool != POISON).sum()) >= rows * HEAD_DIM // 2, "few K-cache bytes written"
    assert bool(torch.isfinite(want_w).all()) and bool((want_w != 0).any()), "degenerate weights"
    assert torch.equal(got_pool, want_pool), (
        f"rows {rows}: {int((got_pool != want_pool).sum())} K-cache bytes differ from the split path"
    )
    assert torch.equal(got_q, want_q), (
        f"rows {rows}: {int((got_q != want_q).sum())} q_fp8 bytes differ from the split path"
    )
    assert torch.equal(got_w.view(torch.int32), want_w.view(torch.int32)), (
        f"rows {rows}: {int((got_w != want_w).sum())} head weights differ from the split path"
    )


if __name__ == "__main__":  # the serving container has no pytest
    for r in ROWS:
        test_unsplit_reads_give_the_split_bytes(r)
        print(f"OK rows={r}", flush=True)
    print("IX_NOSPLIT_TEST PASS")
