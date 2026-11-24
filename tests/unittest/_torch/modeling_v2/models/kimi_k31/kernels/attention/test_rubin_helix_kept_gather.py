# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kept-entry remap + NVFP4 -> E4M3 gather of helix DSA generation rows (family kernel copy).

The kernel's conversions against the C++ gather on every dequantization input of both NVFP4 latent
layouts (mla_kv_cache_residual_dim 64 and 0), and its CUDA graph replay. The helix-shaped rows are
the entry's certified cells (ops/attention/test_rubin_helix_kept_gather_op.py), gated against the
two-step path of the C++ ops.
"""

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

import tensorrt_llm  # noqa: E402,F401  (loads the C++ custom ops)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (  # noqa: E402
    rubin_helix_kept_gather as kg,
)
from tensorrt_llm._utils import get_sm_version  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the Rubin family kernels are certified on sm_107",
)

HEAD = kg.HEAD_DIM
TPB = 64
TOPK = 2048
DEQUANT = 100.0 / (448.0 * 6.0)  # TRTLLM_NVFP4_MLA_KV_CACHE_AMAX unset (100)
RESIDUALS = [pytest.param(64, id="residual64"), pytest.param(0, id="residual0")]


# --------------------------------------------------------------------------------------------- helpers
def _host_pointers(data, scale):
    pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
    pointers[0, 0, 0] = data.data_ptr()
    pointers[0, 0, 1] = scale.data_ptr()
    mapping = torch.zeros((1, 2), dtype=torch.int32)
    return pointers, mapping


def _kept(
    topk,
    lens,
    req,
    block_table,
    stride_factor,
    layer_offset,
    data,
    scale,
    dequant,
    num_pool_tokens,
    residual,
):
    rows = topk.shape[0]
    out = torch.zeros((rows, topk.shape[1], HEAD), dtype=torch.float8_e4m3fn, device="cuda")
    compact = torch.full_like(topk, 12345)
    kg.helix_kept_gather(
        topk,
        lens,
        req,
        block_table,
        TPB,
        stride_factor,
        layer_offset,
        data.view(-1),
        scale.view(-1),
        dequant,
        num_pool_tokens,
        out,
        compact,
        residual_dim=residual,
    )
    return out, compact


def _random_pools(num_pool_tokens, gen, residual):
    data_bytes, scale_bytes = kg.nvfp4_row_bytes(residual)
    data = torch.randint(
        0, 256, (num_pool_tokens, data_bytes), dtype=torch.uint8, device="cuda", generator=gen
    )
    # Finite non-negative E4M3 block scales (0x00..0x7E), the writer's range.
    scale = torch.randint(
        0, 0x7F, (num_pool_tokens, scale_bytes), dtype=torch.uint8, device="cuda", generator=gen
    )
    return data, scale


def _helix_case(rows, num_req, local_len, lens_list, gen, num_layers=3, layer=1):
    """Helix-shaped rows: each row keeps ``len`` distinct ascending local positions first, then -1."""
    pages_per_req = (local_len + TPB - 1) // TPB
    num_blocks = num_req * pages_per_req + 16
    num_pool_tokens = num_blocks * num_layers * TPB
    perm = torch.randperm(num_blocks, generator=gen, device="cuda").to(torch.int32)
    block_table = perm[: num_req * pages_per_req].view(num_req, pages_per_req).contiguous()
    topk = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    for r, n in enumerate(lens_list):
        n = min(n, local_len)
        if n > 0:
            pos = torch.randperm(local_len, generator=gen, device="cuda")[:n].sort().values
            topk[r, :n] = pos.to(torch.int32)
    lens = torch.tensor(lens_list, dtype=torch.int32, device="cuda")
    req = (torch.arange(rows, device="cuda") * num_req // rows).to(torch.int32)
    stride_factor = num_layers * TPB
    return topk, lens, req, block_table, stride_factor, layer, num_pool_tokens


def _every_two_level_input(finite):
    """Residual-64 rows holding every NoPE (code, finite scale byte) and every RoPE (main code, residual code, main
    scale, residual scale) combination."""
    nf = finite.numel()  # 254
    codes = torch.arange(16, dtype=torch.int32)
    # RoPE: one 16-value group per (sm, sr, shift): main code j, residual code (j + shift) % 16.
    sm = finite.repeat_interleave(nf * 16)
    sr = finite.repeat_interleave(16).repeat(nf)
    shift = torch.arange(16, dtype=torch.int32).repeat(nf * nf)
    groups = sm.numel()  # 254 * 254 * 16
    rows = (groups + 3) // 4
    main = codes[None, :].expand(groups, 16)
    res = (codes[None, :] + shift[:, None]) % 16
    pad = rows * 4 - groups
    if pad:
        main = torch.cat([main, main[:pad]])
        res = torch.cat([res, res[:pad]])
        sm = torch.cat([sm, sm[:pad]])
        sr = torch.cat([sr, sr[:pad]])
    main_bytes = (main[:, 0::2] | (main[:, 1::2] << 4)).to(torch.uint8)  # [groups, 8]
    res_bytes = (res[:, 0::2] | (res[:, 1::2] << 4)).to(torch.uint8)
    rope = torch.cat([main_bytes, res_bytes], dim=1).view(rows, 64)
    # NoPE: 32 groups per row, every code once per group, scale bytes cycling through the finite set.
    nope_codes = codes.repeat(32)
    nope = (
        (nope_codes[0::2] | (nope_codes[1::2] << 4)).to(torch.uint8).repeat(rows, 1)
    )  # [rows, 256]
    nope_scale = finite[(torch.arange(rows * 32) % nf)].view(rows, 32).to(torch.uint8)
    rope_scale = torch.stack([sm, sr], dim=1).view(rows, 8).to(torch.uint8)
    return torch.cat([nope, rope], dim=1), torch.cat([nope_scale, rope_scale], dim=1)


def _every_one_level_input(finite):
    """Residual-0 rows holding every (data byte, finite scale byte) pair at each of the 36 group positions: row r,
    group g has scale finite[(r + g) % 254] and data bytes (8 * (r // 254) + 8 * g + j) % 256, j < 8, so the 32 rows
    that give group g a scale hold all 256 byte values there once (every pair of E2M1 codes)."""
    nf = finite.numel()
    rows = nf * 32
    r = torch.arange(rows, dtype=torch.int64)
    g = torch.arange(36, dtype=torch.int64)
    j = torch.arange(8, dtype=torch.int64)
    data = (8 * (r // nf)[:, None, None] + 8 * g[None, :, None] + j[None, None, :]) % 256
    scale = finite[(r[:, None] + g[None, :]) % nf]
    return data.to(torch.uint8).view(rows, 288), scale.to(torch.uint8)


@pytest.mark.parametrize("residual", RESIDUALS)
def test_dequant_matches_cpp_on_every_input(residual):
    """Every dequantization input of the layout, against the C++ op.

    Residual 64: the RoPE part takes the evaluation order nvcc emitted for the C++ kernel (fma(main, A, residual * B);
    another order gives other E4M3 bytes for about 6e-4 of the inputs). Residual 0: all 36 groups are one-level (the
    C++ kernel converts them like NoPE). Both must be exact."""
    finite = torch.tensor([b for b in range(256) if (b & 0x7F) != 0x7F], dtype=torch.int32)
    if residual == 64:
        data, scale = _every_two_level_input(finite)
    else:
        data, scale = _every_one_level_input(finite)
    data, scale = data.cuda().contiguous(), scale.cuda().contiguous()
    rows = data.shape[0]
    assert (data.shape[1], scale.shape[1]) == kg.nvfp4_row_bytes(residual)

    num_tokens = rows
    cols = TOPK
    padded = (num_tokens + cols - 1) // cols * cols
    glob = torch.full((padded,), -1, dtype=torch.int32, device="cuda")
    glob[:num_tokens] = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    glob = glob.view(-1, cols)
    dequant = torch.tensor([DEQUANT], dtype=torch.float32, device="cuda")
    pointers, mapping = _host_pointers(data, scale)
    ref = torch.zeros((glob.shape[0], cols, HEAD), dtype=torch.float8_e4m3fn, device="cuda")
    compact = torch.empty_like(glob)
    torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
        pointers, mapping, glob, ref, compact, dequant, 0, residual, num_tokens
    )
    # Identity remap: one request whose block b is page b, page stride TPB, layer 0.
    pages = padded // TPB
    block_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, pages)
    req = torch.zeros((glob.shape[0],), dtype=torch.int32, device="cuda")
    lens = torch.full((glob.shape[0],), cols, dtype=torch.int32, device="cuda")
    ref_bytes = ref.view(torch.uint8).view(-1, HEAD)[:num_tokens]
    out, comp = _kept(
        glob, lens, req, block_table, TPB, 0, data, scale, dequant, num_tokens, residual=residual
    )
    assert torch.equal(comp, compact)
    got = out.view(torch.uint8).view(-1, HEAD)[:num_tokens]
    mismatch = (
        int((got[:, :512] != ref_bytes[:, :512]).sum()),
        int((got[:, 512:] != ref_bytes[:, 512:]).sum()),
    )
    assert mismatch == (0, 0), (
        f"residual {residual}: (NoPE, RoPE) mismatching bytes {mismatch} over {num_tokens * 512} NoPE and "
        f"{num_tokens * 64} RoPE values"
    )


def test_cuda_graph_replay_reads_device_row_lengths():
    """A captured launch replays with new row lengths read on the device (residual 0, the served layout)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    rows = 64
    lens_a = [
        int(v) for v in torch.randint(100, 400, (rows,), generator=gen, device="cuda").tolist()
    ]
    lens_b = [int(v) for v in torch.randint(0, 520, (rows,), generator=gen, device="cuda").tolist()]
    topk, lens, req, block_table, stride_factor, layer, num_pool_tokens = _helix_case(
        rows, 8, 16384, lens_a, gen
    )
    data, scale = _random_pools(num_pool_tokens, gen, 0)
    dequant = torch.tensor([DEQUANT], dtype=torch.float32, device="cuda")
    out = torch.zeros((rows, TOPK, HEAD), dtype=torch.float8_e4m3fn, device="cuda")
    compact = torch.empty_like(topk)

    def run():
        kg.helix_kept_gather(
            topk,
            lens,
            req,
            block_table,
            TPB,
            stride_factor,
            layer,
            data.view(-1),
            scale.view(-1),
            dequant,
            num_pool_tokens,
            out,
            compact,
            residual_dim=0,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lens_list in (lens_b, lens_a):
        lens.copy_(torch.tensor(lens_list, dtype=torch.int32, device="cuda"))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        exp_out, exp_compact = _kept(
            topk,
            lens,
            req,
            block_table,
            stride_factor,
            layer,
            data,
            scale,
            dequant,
            num_pool_tokens,
            residual=0,
        )
        assert torch.equal(compact, exp_compact)
        assert torch.equal(out.view(torch.uint8), exp_out.view(torch.uint8))
