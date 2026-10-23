# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A/B latency of the QSA sparse core: TRT-LLM Triton vs FlashInfer PrimTS.

Qwen3.8-Flash-Next geometry (D256, 4-token KV blocks, top-k 512 blocks) over an
HND paged cache. Triton time includes the block-to-token expansion the Triton
kernel needs; PrimTS consumes block IDs directly. Both outputs are checked
against the Torch reference on sampled rows before timing.

    python bench_qsa_sparse_core.py --mode prefill --ctx 32768 --chunk 4096
    python bench_qsa_sparse_core.py --mode decode --ctx 32768 --batch 64 --sq 4
    python bench_qsa_sparse_core.py --preset quick --out results.md
"""

import argparse
import json
import math
import statistics
import sys
import time
from types import SimpleNamespace

import torch

HEAD_DIM = 256
KV_BLOCK = 4
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // KV_BLOCK
PREFILL_GROUP = 4
FP8_MAX = 448.0

QUICK_PRESET = [
    dict(mode="prefill", ctx=32768, chunk=4096, batch=1, sq=1),
    dict(mode="decode", ctx=32768, chunk=0, batch=8, sq=1),
    dict(mode="decode", ctx=32768, chunk=0, batch=64, sq=1),
    dict(mode="decode", ctx=32768, chunk=0, batch=64, sq=4),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    p.add_argument("--tp", type=int, choices=(1, 2), default=2, help="1: Hq/Hkv 24/2, 2: 12/1")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=4096, help="prefill query rows per request")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--sq", type=int, default=1, help="decode query tokens per request (1 or MTP+1)")
    p.add_argument("--kv-dtype", choices=("bf16", "fp8"), default="bf16")
    p.add_argument("--page", type=int, default=32)
    p.add_argument("--pattern", choices=("random", "sorted"), default="random")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--l2-mb", type=int, default=256)
    p.add_argument("--check-rows", type=int, default=8)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--skip-triton", action="store_true")
    p.add_argument("--skip-prims", action="store_true")
    p.add_argument("--preset", choices=("quick",), default=None)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--out", default=None, help="markdown file to append rows to")
    p.add_argument("--json", default=None, help="json file to append rows to")
    return p.parse_args()


def heads_for_tp(tp):
    return (24, 2) if tp == 1 else (12, 1)


def make_cache(num_requests, ctx, hkv, page, kv_dtype, device, gen):
    pages_per_req = math.ceil(ctx / page)
    num_pages = num_requests * pages_per_req
    k = (
        torch.randn(
            num_pages, hkv, page, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.25
    )
    v = (
        torch.randn(
            num_pages, hkv, page, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.25
    )
    perm = torch.randperm(num_pages, device=device, generator=gen).to(torch.int32)
    block_table = perm.view(num_requests, pages_per_req).contiguous()
    return k.to(kv_dtype), v.to(kv_dtype), block_table


def make_block_ids(positions, pattern, device, gen):
    rows = positions.numel()
    visible = (positions + 1) // KV_BLOCK
    max_vis = int(visible.max())
    keys = torch.rand(rows, max_vis, device=device, generator=gen)
    col = torch.arange(max_vis, device=device)
    keys.masked_fill_(col[None, :] >= visible[:, None], 2.0)
    order = keys.argsort(dim=1)[:, :BLOCK_TOPK]
    count = visible.clamp(max=BLOCK_TOPK)
    valid = torch.arange(BLOCK_TOPK, device=device)[None, :] < count[:, None]
    ids = order.to(torch.int32)
    if pattern == "sorted":
        ids = (
            torch.where(valid, ids, torch.full_like(ids, torch.iinfo(torch.int32).max))
            .sort(dim=1)
            .values
        )
    return torch.where(valid, ids, torch.full_like(ids, -1)).contiguous()


def build_case(args, device, gen):
    hq, hkv = heads_for_tp(args.tp)
    kv_dtype = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float8_e4m3fn
    if args.mode == "prefill":
        rows_per_req = args.chunk
    else:
        rows_per_req = args.sq
    B = args.batch
    rows = B * rows_per_req
    q = torch.randn(rows, hq, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=gen) * 0.25
    k, v, block_table = make_cache(B, args.ctx, hkv, args.page, kv_dtype, device, gen)
    req = torch.arange(B, dtype=torch.int32, device=device).repeat_interleave(rows_per_req)
    pos = torch.arange(args.ctx - rows_per_req, args.ctx, dtype=torch.int64, device=device).repeat(
        B
    )
    seq_lens = torch.full((rows,), args.ctx, dtype=torch.int32, device=device)
    ids = make_block_ids(pos, args.pattern, device, gen)
    scale = HEAD_DIM**-0.5
    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        block_table=block_table,
        req=req,
        pos=pos,
        seq_lens=seq_lens,
        ids=ids,
        scale=scale,
        hq=hq,
        hkv=hkv,
        rows=rows,
        rows_per_req=rows_per_req,
        B=B,
        kv_dtype=kv_dtype,
    )


def triton_runner(case, args):
    from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
        triton_expand_qsa_block_indices,
        triton_qsa_paged_sparse_gqa,
    )

    def run():
        tokens = triton_expand_qsa_block_indices(
            case.ids, case.pos, case.seq_lens, compress_ratio=KV_BLOCK, token_topk=TOKEN_TOPK
        )
        return triton_qsa_paged_sparse_gqa(
            q=case.q,
            k_cache=case.k,
            v_cache=case.v,
            block_table=case.block_table,
            selected_tokens=tokens,
            request_indices=case.req,
            tokens_per_block=args.page,
            softmax_scale=case.scale,
            query_positions=case.pos,
            compress_ratio=KV_BLOCK,
        )

    return run


def prims_runner(case, args, device):
    from tensorrt_llm._torch.attention.backends.prims_ts import (
        QTokenKvBlockSparsePagedTSWrapper,
        get_q_token_kv_block_sparse_workspace_size,
        make_q_token_kv_block_sparse_qo_indptr,
    )

    q = case.q
    sm_scale = case.scale
    if case.kv_dtype == torch.float8_e4m3fn:
        q_scale = (q.float().abs().amax() / FP8_MAX).clamp_min(1e-8).item()
        q = (q.float() / q_scale).to(torch.float8_e4m3fn)
        sm_scale = case.scale * q_scale
    out = torch.empty(case.rows, case.hq, HEAD_DIM, dtype=torch.bfloat16, device=device)
    packed = args.mode == "prefill"
    if packed:
        group = PREFILL_GROUP
        starts = torch.arange(0, case.rows + 1, case.rows_per_req, dtype=torch.int32)
        qo_indptr = make_q_token_kv_block_sparse_qo_indptr(
            starts, case.rows, group_size=group, device=device
        )
        routes = qo_indptr.numel() - 1
        q_view, out_view, split_kv = q, out, False
    else:
        group = case.rows_per_req
        qo_indptr = None
        routes = case.B
        q_view = q.view(case.B, 1, group, case.hq, HEAD_DIM)
        out_view = out.view_as(q_view)
        split_kv = True
    ws_bytes = get_q_token_kv_block_sparse_workspace_size(
        q_view,
        case.k,
        case.block_table,
        block_topk=BLOCK_TOPK,
        max_seq_len_kv=args.ctx,
        o_data_type=out.dtype,
        qo_indptr=qo_indptr,
        seq_len_q=group if packed else None,
        kv_block_size=KV_BLOCK,
        split_kv=split_kv,
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        routes,
        group,
        case.hq,
        case.hkv,
        HEAD_DIM,
        KV_BLOCK,
        args.page,
        BLOCK_TOPK,
        args.ctx,
        device=device,
        workspace_buffer=workspace,
        use_packed_q=packed,
        split_kv=split_kv,
        q_data_type=q.dtype,
        kv_data_type=case.k.dtype,
        o_data_type=out.dtype,
    )

    def run():
        wrapper.run(
            q_view,
            (case.k, case.v),
            case.block_table,
            case.ids,
            case.req,
            case.pos,
            qo_indptr=qo_indptr,
            sm_scale=sm_scale,
            out=out_view,
        )
        return out

    return run, dict(group_size=group, routes=routes, workspace_mb=ws_bytes / 2**20)


def reference(case, args, rows_idx):
    from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
        triton_expand_qsa_block_indices,
    )
    from tensorrt_llm._torch.attention.backends.sparse.qsa.module import qsa_sparse_gqa_reference

    tokens = triton_expand_qsa_block_indices(
        case.ids[rows_idx],
        case.pos[rows_idx],
        case.seq_lens[rows_idx],
        compress_ratio=KV_BLOCK,
        token_topk=TOKEN_TOPK,
    )
    meta = SimpleNamespace(
        qsa_block_table=case.block_table,
        kv_cache_manager=SimpleNamespace(tokens_per_block=args.page),
    )
    return qsa_sparse_gqa_reference(
        q=case.q[rows_idx],
        k_cache=case.k.to(torch.bfloat16),
        v_cache=case.v.to(torch.bfloat16),
        selected_tokens=tokens,
        request_indices=case.req[rows_idx],
        metadata=meta,
        softmax_scale=case.scale,
    )


def check(name, out, ref, rows_idx, tol):
    got = out[rows_idx].float()
    err = (got - ref.float()).abs().max().item()
    ok = torch.allclose(got, ref.float(), atol=tol, rtol=tol)
    print(f"  [{name}] max_abs_err={err:.4e} {'OK' if ok else 'MISMATCH'} (tol {tol})")
    return ok, err


def time_fn(fn, args, device):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if args.no_graph:
        launch = fn
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        launch = graph.replay
    scrub = torch.empty(args.l2_mb << 20, dtype=torch.uint8, device=device)
    samples = []
    for i in range(args.warmup + args.iters):
        scrub.random_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        if i >= args.warmup:
            samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def summarize(samples):
    s = sorted(samples)
    return dict(median_us=statistics.median(s), p95_us=s[int(0.95 * (len(s) - 1))], n=len(s))


def run_case(args, device):
    gen = torch.Generator(device=device).manual_seed(args.seed)
    case = build_case(args, device, gen)
    hq, hkv = case.hq, case.hkv
    label = (
        f"{args.mode} tp{args.tp} Hq/Hkv={hq}/{hkv} ctx={args.ctx} "
        + (f"chunk={args.chunk} " if args.mode == "prefill" else f"bs={args.batch} sq={args.sq} ")
        + f"kv={args.kv_dtype} page={args.page} pattern={args.pattern}"
    )
    print(f"== {label}")
    rows_idx = torch.linspace(
        0, case.rows - 1, steps=min(args.check_rows, case.rows), device=device
    ).long()
    ref = reference(case, args, rows_idx)
    tol = 2e-2 if args.kv_dtype == "bf16" else 5e-2
    row = dict(
        label=label,
        mode=args.mode,
        tp=args.tp,
        ctx=args.ctx,
        chunk=args.chunk,
        batch=args.batch,
        sq=args.sq,
        kv_dtype=args.kv_dtype,
        page=args.page,
        pattern=args.pattern,
        rows=case.rows,
    )
    runners = []
    if not args.skip_triton:
        t0 = time.time()
        fn = triton_runner(case, args)
        ok, err = check("triton", fn(), ref, rows_idx, tol)
        row.update(triton_ok=ok, triton_err=err)
        runners.append(("triton", fn))
        print(f"  triton ready in {time.time() - t0:.1f}s")
    if not args.skip_prims:
        t0 = time.time()
        fn, info = prims_runner(case, args, device)
        ok, err = check("prims_ts", fn(), ref, rows_idx, tol)
        row.update(prims_ok=ok, prims_err=err, **info)
        runners.append(("prims_ts", fn))
        print(
            f"  prims_ts ready in {time.time() - t0:.1f}s (G={info['group_size']}, routes={info['routes']})"
        )
    samples = {name: [] for name, _ in runners}
    for order in (runners, list(reversed(runners))):
        for name, fn in order:
            samples[name] += time_fn(fn, args, device)
    for name in samples:
        row[name] = summarize(samples[name])
        print(
            f"  {name:8s} median {row[name]['median_us']:9.1f} us  p95 {row[name]['p95_us']:9.1f} us"
        )
    if "triton" in row and "prims_ts" in row:
        row["speedup"] = row["triton"]["median_us"] / row["prims_ts"]["median_us"]
        print(f"  speedup (triton/prims_ts) = {row['speedup']:.3f}x")
    return row


def md_row(r):
    t = f"{r['triton']['median_us']:.1f}" if "triton" in r else "-"
    p = f"{r['prims_ts']['median_us']:.1f}" if "prims_ts" in r else "-"
    s = f"{r['speedup']:.2f}x" if "speedup" in r else "-"
    okc = "".join(
        f"{n}:{'ok' if r.get(k) else 'BAD'} "
        for n, k in (("T", "triton_ok"), ("P", "prims_ok"))
        if k in r
    )
    return f"| {r['label']} | {r['rows']} | {t} | {p} | {s} | {okc.strip()} |"


def main():
    args = parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    print("device:", torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    cases = []
    if args.preset == "quick":
        for over in QUICK_PRESET:
            a = argparse.Namespace(**vars(args))
            for k, v in over.items():
                setattr(a, k, v)
            cases.append(a)
    else:
        cases.append(args)
    rows = []
    for a in cases:
        try:
            rows.append(run_case(a, device))
        except Exception as exc:  # keep going so one failing config does not hide the rest
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append(
                dict(
                    label=f"{a.mode} ctx={a.ctx} bs={a.batch} sq={a.sq} chunk={a.chunk}",
                    rows=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        torch.cuda.empty_cache()
    header = "| case | rows | Triton us | PrimTS us | Triton/PrimTS | check |\n|---|---:|---:|---:|---:|---|"
    print(header)
    for r in rows:
        print(
            md_row(r)
            if "error" not in r
            else f"| {r['label']} | - | error: {r['error'][:80]} | | | |"
        )
    if args.out:
        with open(args.out, "a") as f:
            f.write(
                f"\n### {torch.cuda.get_device_name(device)} {time.strftime('%Y-%m-%d %H:%M')}\n{header}\n"
            )
            for r in rows:
                f.write(
                    (md_row(r) if "error" not in r else f"| {r['label']} | - | error | | | |")
                    + "\n"
                )
    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    bad = [r["label"] for r in rows if "error" in r or not (r["triton_ok"] and r["prims_ok"])]
    if bad:
        print("FAILED cases:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
