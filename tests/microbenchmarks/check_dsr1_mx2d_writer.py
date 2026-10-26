# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Bit-exact check of the MXFP4-2D ("MX2D") FP4 MLA KV writers.

Writes prompts with the production context writer, appends generated tokens
one step at a time with the production generation writer (including steps
whose new token raises a 32x32 block's scale, which re-quantizes the block
from its HP-row shadow), and compares every written latent-prefix code and
K-scale byte with a direct torch quantization of the BF16 latents:

    s = clamp(2^(floor(log2 amax_32x32) - 2), 2^-15, 4),  code = E2M1(x / s),
    K-scale byte (both 16-dim columns of the 32-dim group) = E4M3(s * 64).

Run with TRTLLM_FP4_MLA_MXFP4_2D=1 (the script sets it before import)::

    python check_dsr1_mx2d_writer.py --seq-lens 1 17 31 32 33 100 127 128 129 300 --steps 70
"""

import argparse
import os
import sys

os.environ["TRTLLM_FP4_MLA_MXFP4_2D"] = "1"
os.environ.setdefault("TRTLLM_FP4_MLA_ATTENTION_BACKEND", "cutedsl")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_dsr1_mla_decode as bench  # noqa: E402

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code, ties up, saturating at 6; +0 for zero results."""
    m = y.abs().clamp(max=6.0)
    small = torch.floor(m * 2.0 + 0.5) * 0.5
    mid = torch.floor(m + 0.5)
    big = torch.where(m >= 5.0, torch.full_like(m, 6.0), torch.full_like(m, 4.0))
    r = torch.where(m < 2.0, small, torch.where(m < 4.0, mid, big))
    lut = _E2M1.to(y.device)
    mag = (r[..., None] == lut).float().argmax(dim=-1)
    sign = ((y < 0) & (mag > 0)).to(torch.int64) * 8
    return (mag + sign).to(torch.uint8)


def _block_scale(amax: torch.Tensor) -> torch.Tensor:
    _, e = torch.frexp(amax)  # amax = m * 2^e, m in [0.5, 1): floor(log2 amax) = e - 1
    e = torch.where(amax > 0, e - 1, torch.full_like(e, -127))
    e = (e - 2).clamp(min=-15, max=2)
    return torch.ldexp(torch.ones_like(amax), e)


def _swz(row: torch.Tensor, col: torch.Tensor, sf_per_token: int) -> torch.Tensor:
    padded = ((sf_per_token + 3) // 4) * 4
    return (
        col % 4
        + (col // 4) * (4 * 128)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * padded)
    )


def check_sequence(kv_cache, sf_cache, pages, latent_rows, kv_gs):
    """Compare one sequence's written prefix with the reference; return #mismatches."""
    n = latent_rows.shape[0]
    sf_per_token = sf_cache.shape[-1] if sf_cache.ndim > 1 else None
    bad = 0
    x = latent_rows[:, :512].float()
    for b0 in range(0, n, 32):
        blk = x[b0 : min(b0 + 32, n)].reshape(-1, 16, 32)  # [t, groups, 32]
        amax = blk.abs().amax(dim=(0, 2))  # [16]
        s = _block_scale(amax)
        codes = _e2m1_codes(blk / s[None, :, None])  # [t, 16, 32]
        for t in range(blk.shape[0]):
            pos = b0 + t
            page = pages[pos // 128]
            row = pos % 128
            raw = kv_cache[page, 0, row, 0, :256]
            got = torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(16, 32)
            bad += int((got != codes[t]).sum().item())
            cols = torch.arange(32, device=x.device)
            offs = _swz(torch.full_like(cols, row), cols, sf_per_token)
            got_sf = sf_cache[page].reshape(-1).view(torch.float8_e4m3fn)[offs].float()
            want_sf = (s * kv_gs).repeat_interleave(2)
            bad += int((got_sf != want_sf).sum().item())
    return bad


def main():
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    p = argparse.ArgumentParser()
    p.add_argument(
        "--seq-lens", type=int, nargs="+", default=[1, 17, 31, 32, 33, 100, 127, 128, 129, 300]
    )
    p.add_argument("--steps", type=int, default=70)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    assert fp4._fp4_mla_mxfp4_2d_enabled() and fp4.FP4_MLA_KV_GLOBAL_SCALE == 64.0
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    ctx_lens = list(args.seq_lens)
    cap_lens = [L + args.steps for L in ctx_lens]
    B = len(ctx_lens)
    # Allocate for the final lengths, write only the prompts.
    manager, metadata, q = bench.build_fp4_case(
        cap_lens, args.heads, "cutedsl", keep_latent=False, skip_fill=True
    )
    try:
        assert getattr(manager, "fp4_mla_mxfp4_2d", False)
        assert (
            manager.get_mla_v_scale_pool() is None and manager.get_mla_v_packed_pool_base() is None
        )
        kv_cache, sf_cache = manager.get_fp4_mla_cache_buffers(0)
        stride = metadata.fp4_mla_page_table_stride
        pages = metadata.paged_kv_indices.view(B, stride).cpu().tolist()
        latents = [
            (torch.randn(L, bench.HEAD_DIM_QK, device=device) * 0.5).to(torch.bfloat16)
            for L in cap_lens
        ]
        # Context: every prompt token.
        lens_t = torch.tensor(ctx_lens, dtype=torch.int64, device=device)
        metadata.num_contexts = B
        metadata.num_ctx_tokens = int(sum(ctx_lens))
        metadata.num_context_blocks = B * stride
        metadata.num_generation_blocks = 0
        metadata.batch_indices = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32, device=device), lens_t
        )
        metadata.positions = torch.cat(
            [torch.arange(L, dtype=torch.int32, device=device) for L in ctx_lens]
        )
        metadata.kv_lens_cuda_runtime = lens_t.to(torch.int32)
        metadata.prompt_lens_cuda_runtime = lens_t.to(torch.int32)
        metadata.prompt_lens_cpu_runtime = torch.tensor(ctx_lens, dtype=torch.int32)
        ctx_latent = torch.cat([lat[:L] for lat, L in zip(latents, ctx_lens)]).contiguous()
        fp4.scatter_fp4_mla_kv_cache(
            metadata,
            ctx_latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=512,
        )
        torch.cuda.synchronize()
        bad = sum(
            check_sequence(kv_cache, sf_cache, pages[i], latents[i][: ctx_lens[i]], 64.0)
            for i in range(B)
        )
        print(f"MX2D_WRITER context: mismatches={bad}", flush=True)
        total_bad = bad
        # Generation: one token per sequence per step.
        metadata.num_contexts = 0
        metadata.num_ctx_tokens = 0
        metadata.num_context_blocks = 0
        metadata.num_generation_blocks = B * stride
        metadata.prompt_lens_cuda_runtime = torch.ones(B, dtype=torch.int32, device=device)
        metadata.prompt_lens_cpu_runtime = torch.ones(B, dtype=torch.int32)
        metadata.batch_indices = torch.arange(B, dtype=torch.int32, device=device)
        cos_sin = torch.zeros(max(cap_lens) + 1, 64, 2, dtype=torch.float32, device=device)
        cos_sin[..., 0] = 1.0
        qg = (torch.randn(B, args.heads, bench.HEAD_DIM_QK, device=device) * 0.25).to(
            torch.bfloat16
        )
        for step in range(args.steps):
            cur = [L + step for L in ctx_lens]
            if step % 7 == 3:
                # A large token raises its block's scale: the block is re-quantized
                # from the shadow of its earlier tokens.
                for i in range(B):
                    latents[i][cur[i], :512] *= 8.0
            kv = torch.tensor([c + 1 for c in cur], dtype=torch.int32, device=device)
            metadata.kv_lens_cuda_runtime = kv
            metadata.positions = torch.tensor(cur, dtype=torch.int32, device=device)
            metadata.fp4_mla_generation_lengths_num_tokens = -1
            new = torch.stack([latents[i][cur[i]] for i in range(B)]).contiguous()
            fp4.scatter_fp4_mla_kv_cache(
                metadata,
                new,
                layer_idx=0,
                token_offset=0,
                phase="generation",
                local_layer=0,
                v_head_dim=512,
                rotary_cos_sin=cos_sin,
                q_pe=qg[..., 512:],
                q_rope_out=qg[..., 512:],
                q_quant_input=qg,
            )
        torch.cuda.synchronize()
        bad = sum(
            check_sequence(kv_cache, sf_cache, pages[i], latents[i][: cap_lens[i]], 64.0)
            for i in range(B)
        )
        print(f"MX2D_WRITER generation({args.steps} steps): mismatches={bad}", flush=True)
        total_bad += bad
        print(f"MX2D_WRITER_DONE ok={total_bad == 0}", flush=True)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
