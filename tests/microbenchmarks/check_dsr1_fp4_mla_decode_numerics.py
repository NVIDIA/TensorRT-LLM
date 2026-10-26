# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Numerics check for the DS-R1 FP4 MLA decode backends (Triton vs CuTeDSL).

One FP4 MLA cache is written by the production context writer (built with the
``cutedsl`` storage policy, which is a superset of Triton's), then both decode
backends run on it and are compared against two FP32 references:

* ``exact``: softmax(Q_q . K_q^T) . V_q on the *dequantized* FP4 Q/K/V with the
  K-RoPE residual terms, i.e. the kernels' math without P quantization;
* ``bf16``:  softmax(Q . K^T) . V on the original BF16 Q and latent (what the
  FP8/BF16 model would compute from the same activations).

Usage::

    python check_dsr1_fp4_mla_decode_numerics.py --seq-lens 1 17 129 1000 3000 \
        --heads 128 --backends triton cutedsl
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_dsr1_mla_decode as bench  # noqa: E402

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _dequant_fp4_swizzled(fp4_bytes, sf, *, logical_dim, sf_per_token, global_scale, block=16):
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=fp4_bytes.device)
    b = fp4_bytes.view(torch.uint8)[:, : logical_dim // 2]
    rows = b.shape[0]
    codes = torch.stack((b & 0x0F, b >> 4), dim=-1).reshape(rows, logical_dim)
    vals = lut[codes.long()].reshape(rows, sf_per_token, block)
    row = torch.arange(rows, device=b.device)[:, None]
    col = torch.arange(sf_per_token, device=b.device)[None, :]
    padded = ((sf_per_token + 3) // 4) * 4
    off = (
        col % 4
        + (col // 4) * (4 * 128)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * padded)
    )
    scales = sf.view(torch.float8_e4m3fn).reshape(-1).float()[off]
    return (vals * scales[..., None] / global_scale).reshape(rows, logical_dim)


def _references(metadata, q, seq_lens, sm_scale):
    """Return (exact_ref, bf16_ref) as [Nq, H, 512] float32."""
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(0)
    storage_dim = kv_cache.shape[-1] * 2
    kv_gs = float(metadata._fp4_mla_kv_global_scale.item())
    q_gs = metadata._fp4_mla_q_global_scale
    head_dim = bench.HEAD_DIM_QK
    res = fp4.FP4_MLA_Q_RESIDUAL_DIM
    prefix = head_dim - res
    n, heads, _ = q.shape
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
        q.reshape(-1, head_dim), q_gs, res, is_act=True
    )
    q_logical = head_dim + res
    qd = _dequant_fp4_swizzled(
        q_fp4,
        q_sf,
        logical_dim=q_logical,
        sf_per_token=q_logical // 16,
        global_scale=float(q_gs.item()),
    ).reshape(n, heads, q_logical)
    q_tail = qd[..., prefix:].reshape(n, heads, res // 16, 2, 16)
    q_main = q_tail[..., 0, :].reshape(n, heads, res)
    q_res = q_tail[..., 1, :].reshape(n, heads, res)
    q_exact = torch.cat((qd[..., :prefix], q_main, q_res, q_main), dim=-1)

    stride = metadata.fp4_mla_page_table_stride
    pages = metadata.paged_kv_indices.view(-1, stride)
    latent = metadata._bench_latent.float()
    starts = np.concatenate(([0], np.cumsum(seq_lens)[:-1]))
    exact_out, bf16_out = [], []
    for i, kv_len in enumerate(seq_lens):
        npg = (kv_len + 127) // 128
        ids = pages[i, :npg]
        fp4_rows = kv_cache[ids, 0, :, 0, :].reshape(npg * 128, -1)
        sf_rows = sf_cache[ids]
        deq = torch.cat(
            [
                _dequant_fp4_swizzled(
                    fp4_rows[p * 128 : (p + 1) * 128],
                    sf_rows[p],
                    logical_dim=storage_dim,
                    sf_per_token=storage_dim // 16,
                    global_scale=kv_gs,
                )
                for p in range(npg)
            ]
        )[:kv_len]
        k_main = deq[:, prefix:head_dim]
        k_res = deq[:, head_dim : head_dim + res]
        k_exact = torch.cat((deq[:, :prefix], k_main, k_main, k_res), dim=-1)
        v_exact = deq[:, : bench.KV_LORA_RANK]
        p = torch.softmax(q_exact[i] @ k_exact.T * sm_scale, dim=-1)
        exact_out.append(p @ v_exact)
        lat = latent[starts[i] : starts[i] + kv_len]
        p = torch.softmax(q[i].float() @ lat.T * sm_scale, dim=-1)
        bf16_out.append(p @ lat[:, : bench.KV_LORA_RANK])
    return torch.stack(exact_out), torch.stack(bf16_out)


def _err(out, ref):
    o, r = out.float().reshape(-1), ref.float().reshape(-1)
    rel = (torch.linalg.vector_norm(o - r) / torch.linalg.vector_norm(r)).item()
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    return {
        "max_abs": (o - r).abs().max().item(),
        "rel_l2": rel,
        "cos": cos,
        "finite": bool(torch.isfinite(o).all().item()),
    }


def run_case(seq_lens, heads, backends, seed, save_prefix=None):
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    build_backend = "cutedsl" if "cutedsl" in backends else backends[0]
    manager, metadata, q = bench.build_fp4_case(
        seq_lens, heads, build_backend, seed=seed, keep_latent=True
    )
    try:
        batch = len(seq_lens)
        metadata.fp4_mla_generation_kv_lens.copy_(metadata.kv_lens_cuda_runtime)
        metadata.fp4_mla_generation_append_lens.copy_(metadata.prompt_lens_cuda_runtime)
        metadata.fp4_mla_generation_lengths_num_tokens = batch
        metadata.fp4_mla_generation_lengths_num_seqs = batch
        metadata.fp4_mla_generation_lengths_num_contexts = 0
        q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
            q.view(-1, q.shape[-1]),
            metadata._fp4_mla_q_global_scale,
            fp4.FP4_MLA_Q_RESIDUAL_DIM,
            is_act=True,
        )
        q_sf = q_sf.view(torch.float8_e4m3fn).reshape(-1)
        outs = {}
        for be in backends:
            os.environ[fp4.FP4_MLA_ATTENTION_BACKEND_ENV] = be
            out = torch.full(
                (q.shape[0], heads, bench.KV_LORA_RANK),
                float("nan"),
                dtype=torch.bfloat16,
                device=q.device,
            )
            fp4.run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q=q,
                output=out,
                sm_scale=bench.SM_SCALE,
                kv_lora_rank=bench.KV_LORA_RANK,
                qk_rope_head_dim=bench.QK_ROPE_HEAD_DIM,
                prequantized_q=q_fp4,
                prequantized_q_sf=q_sf,
                q_batch_capacity=q.shape[0],
            )
            torch.cuda.synchronize()
            outs[be] = out
        if save_prefix:
            torch.save(
                {be: o.cpu() for be, o in outs.items()},
                f"{save_prefix}_h{heads}_b{batch}_s{seed}.pt",
            )
        exact, ref = _references(metadata, q, seq_lens, bench.SM_SCALE)
        rows = []
        for be, out in outs.items():
            rows.append((be, "exact", _err(out, exact)))
            rows.append((be, "bf16", _err(out, ref)))
        rows.append(("exact_fp32", "bf16", _err(exact, ref)))
        if len(outs) == 2:
            a, b = list(outs)
            rows.append((a, b, _err(outs[a], outs[b])))
        # Per-sequence worst case vs the bf16 reference.
        for be, out in outs.items():
            worst = max(_err(out[i], ref[i])["rel_l2"] for i in range(batch))
            rows.append((be, "bf16_worst_seq_rel_l2", {"rel_l2": worst}))
        return rows
    finally:
        manager.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1, 17, 127, 128, 129, 1000, 3000])
    p.add_argument(
        "--random-batch",
        type=int,
        default=0,
        help="add a case of this many dataset-like random lengths",
    )
    p.add_argument("--isl-osl", default=None)
    p.add_argument("--heads", type=int, nargs="+", default=[128])
    p.add_argument("--backends", nargs="+", default=["triton", "cutedsl"])
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--save-prefix", default=None, help="save raw outputs for bitwise A/B")
    p.add_argument("--compare-saved", nargs=2, default=None)
    args = p.parse_args()
    if args.compare_saved:
        a, b = (torch.load(x) for x in args.compare_saved)
        for be in a:
            same = torch.equal(a[be].view(torch.int16), b[be].view(torch.int16))
            print(
                f"BITWISE {be}: {'identical' if same else 'DIFFERENT'} "
                f"max_abs={(a[be].float() - b[be].float()).abs().max().item():.3g}"
            )
        return
    cases = [list(args.seq_lens)]
    if args.random_batch:
        isl_osl = bench.load_isl_osl(args.isl_osl) if args.isl_osl else None
        dist = "dataset_u" if isl_osl is not None else "uniform:1-4000"
        cases.append(bench.sample_kv_lens(dist, args.random_batch, 7, isl_osl, 8192))
    ok = True
    for heads in args.heads:
        for seq_lens in cases:
            label = (
                f"heads={heads} B={len(seq_lens)} kv[min/mean/max]="
                f"{min(seq_lens)}/{int(np.mean(seq_lens))}/{max(seq_lens)}"
            )
            try:
                rows = run_case(seq_lens, heads, args.backends, args.seed, args.save_prefix)
            except Exception as exc:  # report and continue
                import traceback

                traceback.print_exc()
                print(f"NUMERICS {label} ERROR {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                ok = False
                continue
            for a, b, e in rows:
                fields = " ".join(
                    f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()
                )
                print(f"NUMERICS {label} {a} vs {b}: {fields}", flush=True)
                if e.get("finite") is False:
                    ok = False
    print("NUMERICS_DONE ok=" + str(ok), flush=True)


if __name__ == "__main__":
    main()
