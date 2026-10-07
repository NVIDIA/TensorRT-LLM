# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One layer of causal-rollout attention at Cosmos3-Nano geometry, four ways.

    trtllm       TRTLLM backend over CausalKVCacheManager: one fused paged trtllm-gen call
    cudnn_paged  cuDNN backend over the same cache: run-copy of the new K/V + paged SDPA
    slab         one contiguous K/V buffer per layer: in-place write + cuDNN dense SDPA
    naive        materialised torch.cat of all K/V + SDPA (the VANILLA path shipping today)

Timing is CUPTI kernel timestamps on CUDA-graph replays with an L2 flush before
each iteration; ``cupti.finalize`` runs exactly once per process. Every variant
is cross-checked against an fp32 dense reference before anything is timed, so a
silent kernel fallback (one that drops the cached prefix) fails here rather than
being measured.

    python tests/microbenchmarks/bench_kv_causal_attention.py --prompt-len 512   # a denoising step
    python tests/microbenchmarks/bench_kv_causal_attention.py --prompt-len 512 --causal-block-size 394   # clean pass
    python tests/microbenchmarks/bench_kv_causal_attention.py --chunk-cycle --layers 36   # a chunk + commit

The paged variants read exactly the model's window (the prompt, ``window``
tokens before each block, the earlier blocks, the block) and are checked
against that; the dense variants read the whole resident history, stale tokens
included, and are checked against that.
"""

from __future__ import annotations

import argparse
import bisect
import statistics
import sys
import time
from functools import partial

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import CausalKVCacheManager

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 32, 8, 128
TOKENS_PER_FRAME, FRAMES_PER_CHUNK = 394, 4
CHUNK = TOKENS_PER_FRAME * FRAMES_PER_CHUNK
DTYPE = torch.bfloat16
DEV = torch.device("cuda")


# ------------------------------------------------------------------ CUPTI timing


class CuptiTimer:
    def __init__(
        self, iters: int, warmup: int, l2_flush: bool, use_graph: bool, dump: bool = False
    ) -> None:
        from cupti import cupti

        self.cupti, self.iters, self.warmup = cupti, iters, warmup
        self.use_graph = use_graph
        self.dump = dump
        self._l2 = (
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV) if l2_flush else None
        )

    def _flush(self) -> None:
        if self._l2 is not None:
            self._l2.fill_(0)

    def time(self, run_fn, tag: str) -> dict:
        cupti = self.cupti
        run_fn()
        torch.cuda.synchronize()

        if self.use_graph:
            g_reset = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_reset):
                self._flush()
            g_run = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_run):
                run_fn()
            torch.cuda.synchronize()
            reset, run = g_reset.replay, g_run.replay
        else:
            reset, run = self._flush, run_fn

        for _ in range(self.warmup):
            reset()
            run()
        torch.cuda.synchronize()

        launches, kernels = [], []

        def on_buffer_requested():
            return 8 * 1024 * 1024, 0

        def on_buffer_completed(_launches, _kernels, activities):
            for a in activities:
                if a.kind in (
                    cupti.ActivityKind.CONCURRENT_KERNEL,
                    cupti.ActivityKind.MEMCPY,
                    cupti.ActivityKind.MEMSET,
                ):
                    name = a.name if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else None
                    _kernels.append((a.start, a.end, a.correlation_id, name))
                elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                    _launches.append((a.start, a.end, a.correlation_id))

        kinds = [
            cupti.ActivityKind.RUNTIME,
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.DRIVER,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        ]
        for kind in kinds:
            cupti.activity_enable(kind)
        cupti.activity_register_callbacks(
            on_buffer_requested, partial(on_buffer_completed, launches, kernels)
        )

        stamps = []
        for _ in range(self.iters):
            reset()
            torch.cuda.synchronize()
            t0 = cupti.get_timestamp()
            run()
            torch.cuda.synchronize()
            stamps.append((t0, cupti.get_timestamp()))

        cupti.activity_flush_all(0)
        for kind in kinds:
            cupti.activity_disable(kind)

        by_corr: dict[int, list] = {}
        for k in kernels:
            by_corr.setdefault(k[2], []).append(k)
        launches.sort(key=lambda x: x[0])
        starts = [x[0] for x in launches]

        us, busy, counts = [], [], []
        for idx, (t0, t1) in enumerate(stamps):
            lo, hi = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
            ks = [k for i in range(lo, hi) for k in by_corr.get(launches[i][2], [])]
            if not ks:
                raise RuntimeError(f"{tag}: no kernel activity recorded for iteration {idx}")
            t_start = min(k[0] for k in ks)
            us.append((max(k[1] for k in ks) - t_start) / 1e3)
            busy.append(sum(k[1] - k[0] for k in ks) / 1e3)
            counts.append(sum(1 for k in ks if k[3] is not None))
            if self.dump and idx == 0:
                for k in sorted(ks, key=lambda r: r[0]):
                    print(
                        f"  k: {tag:6s} +{(k[0] - t_start) / 1e3:7.1f}us  {(k[1] - k[0]) / 1e3:7.1f}us  "
                        f"{k[3] or '<memcpy/memset>'}"[:150]
                    )
        us.sort()
        return {
            "median_us": statistics.median(us),
            "min_us": us[0],
            "p90_us": us[int(0.9 * (len(us) - 1))],
            "busy_us": statistics.median(busy),  # kernels' own durations, no gaps
            "kernels": statistics.median(counts),
        }


# ------------------------------------------------------------------ variants


def build_cache(
    prompt_len: int,
    window_tokens: int,
    history_chunks: int,
    tokens_per_page: int,
    gen,
    num_layers: int = 1,
):
    """Open a cache, write and pin the prompt, commit ``history_chunks`` chunks.

    Returns the manager plus dense copies of the prompt and of every committed
    history token, oldest first (layer 0's values; other layers get the same).
    """
    mgr = CausalKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        tokens_per_page=tokens_per_page,
        fixed_capacity=max(prompt_len, 1),
        window_tokens=window_tokens,
        chunk_tokens=CHUNK,
        causal_block_sizes=(CHUNK, TOKENS_PER_FRAME),
    )
    mgr.open(pin_tokens=prompt_len)
    kp = torch.randn(prompt_len, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    vp = torch.randn_like(kp)
    for layer in range(num_layers):
        mgr.write_range(layer, 0, kp, vp)
    if prompt_len:
        mgr.commit(prompt_len)

    hist_k, hist_v = [], []
    for _ in range(history_chunks):
        k = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
        v = torch.randn_like(k)
        for layer in range(num_layers):
            mgr.write_range(layer, mgr.past_tokens, k, v)
        mgr.commit()
        hist_k.append(k)
        hist_v.append(v)
    return mgr, kp, vp, torch.cat(hist_k), torch.cat(hist_v)


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], enable_gqa=True
    )[0].transpose(0, 1)


def exact_reference(q, kp, vp, hk, hv, k, v, lo, hi, window):
    """Reference for block ``[lo, hi)`` of the chunk.

    Over the prompt, the ``window`` keys before the block and the chunk up to
    ``hi``; ``hk``/``hv`` are every committed history token.
    """
    keys, values = torch.cat((kp, hk, k[:hi])), torch.cat((vp, hv, v[:hi]))
    pos = torch.arange(keys.shape[0], device=keys.device)
    visible = (pos < kp.shape[0]) | (pos >= kp.shape[0] + hk.shape[0] + lo - window)
    return sdpa(q[lo:hi].float(), keys[visible].float(), values[visible].float())


def chunk_cycle(args, gen) -> None:
    """One chunk's worth of work across ``--layers`` layers.

    Four denoising forwards and one clean pass per layer under CUDA graphs, then
    one commit, with the commit's GPU span (CUPTI) and host time measured
    separately.
    """
    window = args.window_frames * TOKENS_PER_FRAME
    layers = args.layers
    mgr, kp, vp, hk, hv = build_cache(
        args.prompt_len, window, args.history_chunks, args.tokens_per_page, gen, layers
    )
    q = torch.randn(CHUNK, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    k = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    v = torch.randn_like(k)
    q4, k4, v4 = q[None], k[None], v[None]
    if args.backend == "trtllm":
        state: dict = {}
        attns = [
            TrtllmAttention(
                layer_idx=i,
                num_heads=NUM_HEADS,
                head_dim=HEAD_DIM,
                num_kv_heads=NUM_KV_HEADS,
                dtype=DTYPE,
                max_seq_len=mgr.capacity,
                attention_metadata_state=state,
            )
            for i in range(layers)
        ]

        def forward(block):
            for attn in attns:
                attn.forward(
                    q4, k4, v4, batch_size=1, seq_len=CHUNK, kv_cache=mgr, causal_block_size=block
                )
    else:
        attns = [
            CuDNNAttention(
                layer_idx=i,
                num_heads=NUM_HEADS,
                head_dim=HEAD_DIM,
                num_kv_heads=NUM_KV_HEADS,
                dtype=DTYPE,
            )
            for i in range(layers)
        ]

        def forward(block):
            for attn in attns:
                attn.forward(q4, k4, v4, kv_cache=mgr, causal_block_size=block)

    print(
        f"chunk cycle: {layers} layers, backend {args.backend}, prompt {args.prompt_len}, "
        f"history {mgr.history_tokens} [{max(0, mgr.history_tokens - window)} stale], "
        f"chunk {CHUNK}, clean pass in {CHUNK // TOKENS_PER_FRAME} blocks of {TOKENS_PER_FRAME}"
    )
    # Correctness first, on the last layer (every layer holds the same values): a
    # denoising forward and the clean pass against the exact window.
    for block in (CHUNK, TOKENS_PER_FRAME):
        out = (
            attns[-1]
            .forward(q4, k4, v4, batch_size=1, seq_len=CHUNK, kv_cache=mgr, causal_block_size=block)
            .reshape(CHUNK, NUM_HEADS, HEAD_DIM)
        )
        ref = torch.cat(
            [
                exact_reference(q, kp, vp, hk, hv, k, v, lo, lo + block, window)
                for lo in range(0, CHUNK, block)
            ]
        )
        err = (out.float() - ref).abs().max().item()
        name = "denoising" if block == CHUNK else "clean pass"
        print(f"check {name:10s} max|err| vs fp32 exact window = {err:.4f}")
        if err > 2e-2:
            raise SystemExit(f"chunk cycle: {name} deviates from the exact window by {err}")

    timer = CuptiTimer(args.iters, args.warmup, not args.no_l2_flush, not args.no_graph)
    denoise = timer.time(lambda: forward(None), "denoise")
    clean = timer.time(lambda: forward(TOKENS_PER_FRAME), "clean")

    # Commit: GPU span by CUPTI (eager, no graph): first kernel start to last kernel
    # end, so it includes launch gaps between the kernels; host time by the wall clock.
    # Each commit at steady state drops pages, so the rotation and the private-page
    # copies for both blockings are exercised every time.
    def refresh_and_commit():
        mgr.commit()
        if args.backend == "trtllm":
            for n, size in ((1, CHUNK), (CHUNK // TOKENS_PER_FRAME, TOKENS_PER_FRAME)):
                attns[0].metadata.prepare_with_kv_cache(mgr, n, size)

    commit_dev = CuptiTimer(args.iters, args.warmup, False, False).time(
        refresh_and_commit, "commit"
    )
    host = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        refresh_and_commit()
        torch.cuda.synchronize()
        host.append((time.perf_counter() - t0) * 1e3)
    host_ms = statistics.median(host)

    forwards = 4 * denoise["median_us"] + clean["median_us"]
    total = forwards + commit_dev["busy_us"]
    print(f"\n{'per chunk, all layers':34s} {'median':>10s}")
    print(f"{'4 denoising forwards':34s} {4 * denoise['median_us'] / 1e3:9.2f} ms")
    print(f"{'1 clean pass':34s} {clean['median_us'] / 1e3:9.2f} ms")
    print(
        f"{'commit, kernels (sum)':34s} {commit_dev['busy_us'] / 1e3:9.2f} ms  "
        f"({commit_dev['kernels']:.0f} kernels)"
    )
    print(f"{'commit, first to last kernel':34s} {commit_dev['median_us'] / 1e3:9.2f} ms")
    print(f"{'commit, host wall (incl. above)':34s} {host_ms:9.2f} ms")
    print(
        f"{'total device per chunk':34s} {total / 1e3:9.2f} ms  "
        f"(commit share {100 * commit_dev['busy_us'] / total:.2f}%)"
    )
    print(
        f"{'per layer per forward':34s} denoise {denoise['median_us'] / layers:.1f} us, "
        f"clean {clean['median_us'] / layers:.1f} us"
    )
    mgr.shutdown()
    timer.cupti.finalize()  # exactly once per process


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["trtllm", "cudnn_paged", "slab", "naive", "all"],
        default="all",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--tokens-per-page", type=int, default=32, help="cache page size")
    ap.add_argument("--window-frames", type=int, default=96)
    ap.add_argument(
        "--history-chunks",
        type=int,
        default=28,
        help="chunks committed before timing; > window/chunk so the table has rotated",
    )
    ap.add_argument(
        "--causal-block-size",
        type=int,
        default=0,
        help="cut the chunk into causal blocks causal across each other (394 = the clean "
        "pass, one block per frame); 0 = one causal block (a denoising step)",
    )
    ap.add_argument("--no-l2-flush", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dump-kernels", action="store_true", help="print kernel names for iteration 0"
    )
    ap.add_argument(
        "--chunk-cycle",
        action="store_true",
        help="time one chunk across --layers layers: 4 denoising forwards, the clean "
        "pass, and the commit with its bookkeeping",
    )
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--backend", choices=["cudnn_paged", "trtllm"], default="cudnn_paged")
    args = ap.parse_args()
    if args.chunk_cycle:
        chunk_cycle(args, torch.Generator(device=DEV).manual_seed(args.seed))
        return
    causal_block_size = args.causal_block_size or CHUNK
    if CHUNK % causal_block_size:
        raise SystemExit(f"--causal-block-size must divide the chunk of {CHUNK} tokens")
    num_causal_blocks = CHUNK // causal_block_size
    blocks = [
        (i * causal_block_size, (i + 1) * causal_block_size) for i in range(num_causal_blocks)
    ]

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    window = args.window_frames * TOKENS_PER_FRAME
    mgr, kp, vp, k_hist, v_hist = build_cache(
        args.prompt_len, window, args.history_chunks, args.tokens_per_page, gen
    )
    start = mgr.past_tokens
    seq_len = start + CHUNK

    q = torch.randn(CHUNK, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    k = torch.randn(CHUNK, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE, generator=gen)
    v = torch.randn_like(k)
    print(
        f"geometry: q={CHUNK} keys={seq_len} (prompt {args.prompt_len} + history "
        f"{mgr.history_tokens} [{max(0, mgr.history_tokens - window)} stale] + chunk {CHUNK}) "
        f"in {num_causal_blocks} "
        f"causal block(s) of {causal_block_size}; heads={NUM_HEADS}/{NUM_KV_HEADS} d={HEAD_DIM} "
        f"page={mgr.tokens_per_page} bf16"
    )

    trtllm_attn = TrtllmAttention(
        layer_idx=0,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        dtype=DTYPE,
        max_seq_len=mgr.capacity,
        attention_metadata_state={},
    )
    cudnn_attn = CuDNNAttention(
        layer_idx=0, num_heads=NUM_HEADS, head_dim=HEAD_DIM, num_kv_heads=NUM_KV_HEADS, dtype=DTYPE
    )
    q4, k4, v4 = q[None], k[None], v[None]
    block_arg = causal_block_size if num_causal_blocks > 1 else None

    def run_trtllm():
        return trtllm_attn.forward(
            q4, k4, v4, batch_size=1, seq_len=CHUNK, kv_cache=mgr, causal_block_size=block_arg
        )

    def run_cudnn_paged():
        return cudnn_attn.forward(q4, k4, v4, kv_cache=mgr, causal_block_size=block_arg)

    # Dense paths have no per-row key count, so causal blocks cost them one launch each:
    # that is the per-frame clean pass as the reference runs it.
    def per_block(attend):
        if num_causal_blocks == 1:
            return attend(q, seq_len)
        return torch.cat([attend(q[lo:hi], start + hi) for lo, hi in blocks])

    # Slab: what a contiguous per-layer buffer would cost. Prompt and the resident
    # history (stale tokens included: a slab has no per-block start) are already in
    # place; a call writes the new tokens and attends over a slice.
    slab_k = torch.empty(mgr.capacity, NUM_KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE)
    slab_v = torch.empty_like(slab_k)
    n_res = mgr.history_tokens
    resident_k, resident_v = torch.cat((kp, k_hist[-n_res:])), torch.cat((vp, v_hist[-n_res:]))
    slab_k[:start].copy_(resident_k)
    slab_v[:start].copy_(resident_v)

    def run_slab():
        slab_k[start:seq_len].copy_(k)
        slab_v[start:seq_len].copy_(v)
        return per_block(lambda qq, n: sdpa(qq, slab_k[:n], slab_v[:n]))

    def run_naive():
        return per_block(
            lambda qq, n: sdpa(
                qq, torch.cat((resident_k, k[: n - start])), torch.cat((resident_v, v[: n - start]))
            )
        )

    variants = {
        "trtllm": run_trtllm,
        "cudnn_paged": run_cudnn_paged,
        "slab": run_slab,
        "naive": run_naive,
    }
    chosen = list(variants) if args.variant == "all" else [args.variant]
    # Cross-check before timing: a silent fallback that ignores the prefix, or a
    # causal block that sees the causal blocks after it, shows up here.
    all_k, all_v = torch.cat((resident_k, k)).float(), torch.cat((resident_v, v)).float()
    ref_resident = torch.cat(
        [sdpa(q[lo:hi].float(), all_k[: start + hi], all_v[: start + hi]) for lo, hi in blocks]
    )
    ref_exact = torch.cat(
        [exact_reference(q, kp, vp, k_hist, v_hist, k, v, lo, hi, window) for lo, hi in blocks]
    )
    chunk_only = sdpa(q.float(), k.float(), v.float())
    for name in chosen:
        out = variants[name]().reshape(CHUNK, NUM_HEADS, HEAD_DIM).float()
        torch.cuda.synchronize()
        exact = name in ("trtllm", "cudnn_paged")
        err = (out - (ref_exact if exact else ref_resident)).abs().max().item()
        err_chunk = (out - chunk_only).abs().max().item()
        print(
            f"check {name:12s} max|err| vs fp32 {'exact window' if exact else 'resident'} = "
            f"{err:.4f}   vs new-only = {err_chunk:.4f}"
        )
        if err > 2e-2 or err_chunk < 1e-2:
            raise SystemExit(f"{name}: wrong result -- not benchmarking a broken path")

    timer = CuptiTimer(
        args.iters, args.warmup, not args.no_l2_flush, not args.no_graph, args.dump_kernels
    )
    rows = []
    for name in chosen:
        try:
            r = timer.time(variants[name], name)
        except RuntimeError as e:
            if args.no_graph or "capture" not in str(e).lower():
                raise
            print(f"{name}: graph capture failed ({str(e)[:60]}); timing eager", file=sys.stderr)
            r = CuptiTimer(
                args.iters, args.warmup, not args.no_l2_flush, False, args.dump_kernels
            ).time(variants[name], name)
        rows.append((name, r))

    print(f"\n{'variant':12s} {'median us':>10s} {'min us':>9s} {'p90 us':>9s} {'kernels':>8s}")
    for name, r in rows:
        print(
            f"{name:12s} {r['median_us']:10.1f} {r['min_us']:9.1f} {r['p90_us']:9.1f} {r['kernels']:8.0f}"
        )
    mgr.shutdown()
    timer.cupti.finalize()  # exactly once per process


if __name__ == "__main__":
    main()
