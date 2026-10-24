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
"""One layer of causal-rollout attention, four ways.

    trtllm         TRTLLM backend over BlockCausalKVCacheManager: one fused paged trtllm-gen call
    cudnn_paged    cuDNN backend over the same cache: run-copy of the new K/V + paged SDPA
    sdpa_resident  one contiguous K/V buffer per layer: in-place write + torch SDPA
                   (PyTorch picks the kernel) over the resident history
    naive          materialised torch.cat of all K/V + SDPA (the VANILLA path)

The geometry defaults to ``--preset cosmos3-sim``: Cosmos3-Nano-Sim-Bimanual as the
cosmos-framework example serves it (64 latent frames of K/V: 8 sink frames pinned after
the prompt, 55 recent frames, the current frame; 2-frame chunks of 394 tokens; 32/8
heads of 128). Every geometry flag overrides the preset.

Timing is CUPTI kernel timestamps on CUDA-graph replays with an L2 flush before
each iteration; ``cupti.finalize`` runs exactly once per process. Every variant
is cross-checked against an fp32 dense reference before anything is timed, so a
silent kernel fallback (one that drops the cached prefix) fails here rather than
being measured.

    python tests/microbenchmarks/bench_block_causal_attention.py   # a denoising step
    python tests/microbenchmarks/bench_block_causal_attention.py --causal-block-size 394   # clean pass
    python tests/microbenchmarks/bench_block_causal_attention.py --chunk-cycle --layers 36   # a chunk + commit
    python tests/microbenchmarks/bench_block_causal_attention.py --ulysses 2 --json out.json

The paged variants read exactly the model's window (the pinned prefix, ``window``
tokens before each causal block, the earlier causal blocks, the causal block) and are
checked against that; the dense variants read the whole resident history, stale tokens
included, and are checked against that. Where the two answers differ measurably, a
variant must also be clearly closer to its own reference than to the other one, so a
paged path that reads stale tokens cannot pass as exact.
"""

from __future__ import annotations

import argparse
import bisect
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.cache import BlockCausalKVCacheManager, RolloutSpec

# The fp32 reference shared with the unit tests lives in the unittest tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unittest"))
from utils.block_causal_reference import exact_reference  # noqa: E402

DEV = torch.device("cuda")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

PRESETS = {
    "cosmos3-sim": dict(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_frame=394,
        frames_per_chunk=2,
        window_frames=55,
        sink_frames=8,
        prompt_len=512,
        tokens_per_page=32,
        dtype="bf16",
    ),
}

# A wrong answer this far from the right one is a broken path, not rounding.
MAX_ERR = 2e-2
# The exact and resident references must differ by this much before the check below
# asks a variant to be closer to its own reference than to the other one.
MIN_SEPARATION = 5e-3


@dataclass(frozen=True)
class Geometry:
    """Per-rank shape of one attention layer and the rollout around it."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    tokens_per_frame: int
    frames_per_chunk: int
    window_frames: int
    sink_frames: int
    prompt_len: int
    tokens_per_page: int
    dtype: str

    @property
    def chunk(self) -> int:
        return self.tokens_per_frame * self.frames_per_chunk

    @property
    def window(self) -> int:
        return self.window_frames * self.tokens_per_frame

    @property
    def pinned(self) -> int:
        """The prompt plus the sink frames: the fixed region."""
        return self.prompt_len + self.sink_frames * self.tokens_per_frame

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]


# ------------------------------------------------------------------ CUPTI timing


class CuptiTimer:
    def __init__(
        self, iters: int, warmup: int, l2_flush: bool, use_graph: bool, dump: bool = False
    ) -> None:
        from cupti import cupti

        self.cupti, self.iters, self.warmup = cupti, iters, warmup
        self.use_graph = use_graph
        self.dump = dump
        l2_bytes = 2 * torch.cuda.get_device_properties(DEV).L2_cache_size
        self._l2 = torch.empty(l2_bytes, dtype=torch.uint8, device=DEV) if l2_flush else None

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

        us, busy, counts, names = [], [], [], []
        for idx, (t0, t1) in enumerate(stamps):
            lo, hi = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
            ks = [k for i in range(lo, hi) for k in by_corr.get(launches[i][2], [])]
            if not ks:
                raise RuntimeError(f"{tag}: no kernel activity recorded for iteration {idx}")
            t_start = min(k[0] for k in ks)
            us.append((max(k[1] for k in ks) - t_start) / 1e3)
            busy.append(sum(k[1] - k[0] for k in ks) / 1e3)
            counts.append(sum(1 for k in ks if k[3] is not None))
            if idx == 0:
                names = [k[3] for k in sorted(ks, key=lambda r: r[0]) if k[3] is not None]
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
            "kernel_names": names,
            "graph": self.use_graph,
        }


# ------------------------------------------------------------------ variants


def rand_kv(g: Geometry, n: int, gen) -> torch.Tensor:
    return torch.randn(
        n, g.num_kv_heads, g.head_dim, device=DEV, dtype=g.torch_dtype, generator=gen
    )


def build_cache(g: Geometry, history_chunks: int, gen, num_layers: int = 1):
    """Open a cache, pin the prompt and the sink frames, commit ``history_chunks`` chunks.

    Returns the manager plus dense copies of the pinned prefix and of every committed
    history token, oldest first (layer 0's values; other layers get the same).
    """
    spec = RolloutSpec(
        window_tokens=g.window,
        max_staged_tokens=g.chunk,
        causal_blocks={g.chunk: 1, g.tokens_per_frame: g.frames_per_chunk},
        max_pin_tokens=g.pinned,
    )
    mgr = BlockCausalKVCacheManager(
        num_layers=num_layers,
        num_kv_heads=g.num_kv_heads,
        head_dim=g.head_dim,
        dtype=g.torch_dtype,
        tokens_per_page=g.tokens_per_page,
        pool_tokens=BlockCausalKVCacheManager.pool_tokens_for(
            spec, tokens_per_page=g.tokens_per_page
        ),
    )
    mgr.open(spec, pin_tokens=g.pinned)
    kp = rand_kv(g, g.pinned, gen)
    vp = torch.randn_like(kp)
    for layer in range(num_layers):
        mgr.write_range(layer, 0, kp, vp)
    if g.pinned:
        mgr.commit(g.pinned)

    hist_k, hist_v = [], []
    for _ in range(history_chunks):
        k = rand_kv(g, g.chunk, gen)
        v = torch.randn_like(k)
        for layer in range(num_layers):
            mgr.write_range(layer, mgr.staging_offset, k, v)
        mgr.commit(mgr.max_staged_tokens)
        hist_k.append(k)
        hist_v.append(v)
    return mgr, kp, vp, torch.cat(hist_k), torch.cat(hist_v)


def new_tokens(g: Geometry, gen, fused: bool):
    """This forward's q, k, v ``[chunk, heads, head_dim]``.

    With ``fused`` they are slices of one ``[chunk, H + 2*H_kv, head_dim]`` buffer, as a
    fused qkv_proj makes them.
    """
    width = g.num_heads + 2 * g.num_kv_heads
    if fused:
        qkv = torch.randn(
            g.chunk, width, g.head_dim, device=DEV, dtype=g.torch_dtype, generator=gen
        )
        q = qkv[:, : g.num_heads]
        k = qkv[:, g.num_heads : g.num_heads + g.num_kv_heads]
        v = qkv[:, g.num_heads + g.num_kv_heads :]
        return q, k, v
    q = torch.randn(
        g.chunk, g.num_heads, g.head_dim, device=DEV, dtype=g.torch_dtype, generator=gen
    )
    k = rand_kv(g, g.chunk, gen)
    return q, k, torch.randn_like(k)


def pool_bytes(mgr: BlockCausalKVCacheManager, num_layers: int) -> int:
    return sum(
        mgr.kv_buffer(i).numel() * mgr.kv_buffer(i).element_size() for i in range(num_layers)
    )


def check(name: str, out, own_ref, other_ref, chunk_only, own_label: str) -> dict:
    """Raise unless ``out`` is the answer its reference says it is.

    It must be within ``MAX_ERR`` of its own reference, differ from the new-tokens-only
    answer, and, where the two references differ measurably, be clearly closer to its
    own one than to the other.
    """
    err = (out - own_ref).abs().max().item()
    err_other = (out - other_ref).abs().max().item()
    separation = (own_ref - other_ref).abs().max().item()
    err_chunk = (out - chunk_only).abs().max().item()
    print(
        f"check {name:14s} max|err| vs fp32 {own_label} = {err:.4f}   vs the other "
        f"reference = {err_other:.4f} (references differ by {separation:.4f})   "
        f"vs new-only = {err_chunk:.4f}"
    )
    if err > MAX_ERR or err_chunk < 1e-2:
        raise SystemExit(f"{name}: wrong result -- not benchmarking a broken path")
    if separation > MIN_SEPARATION and not err_other > 2 * err:
        raise SystemExit(
            f"{name}: as close to the {'resident' if own_label == 'exact window' else 'exact'} "
            f"answer ({err_other:.4f}) as to its own ({err:.4f}) -- wrong keys read"
        )
    return {"max_err": err, "max_err_other_reference": err_other, "separation": separation}


def environment() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    props = torch.cuda.get_device_properties(DEV)
    return {
        "gpu": props.name,
        "sm": f"{props.major}{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "host": platform.node(),
        "git_sha": sha,
    }


def write_json(path: str, args, g: Geometry, records: list) -> None:
    with open(path, "w") as f:
        json.dump(
            {"args": vars(args), "geometry": asdict(g), "env": environment(), "records": records},
            f,
            indent=2,
        )
    print(f"wrote {path}")


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], enable_gqa=True
    )[0].transpose(0, 1)


def make_attn(backend: str, g: Geometry, layer_idx: int, mgr, state: dict):
    if backend == "trtllm":
        return TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=g.num_heads,
            head_dim=g.head_dim,
            num_kv_heads=g.num_kv_heads,
            dtype=g.torch_dtype,
            max_seq_len=mgr.capacity,
            attention_metadata_state=state,
        )
    return CuDNNAttention(
        layer_idx=layer_idx,
        num_heads=g.num_heads,
        head_dim=g.head_dim,
        num_kv_heads=g.num_kv_heads,
        dtype=g.torch_dtype,
    )


def chunk_cycle(args, g: Geometry, gen) -> None:
    """One chunk's worth of work across ``--layers`` layers.

    ``--denoise-steps`` denoising forwards and one clean pass per layer under CUDA
    graphs, then one commit, with the commit's GPU span (CUPTI) and host time measured
    separately.
    """
    layers, chunk, tpf = args.layers, g.chunk, g.tokens_per_frame
    mgr, kp, vp, hk, hv = build_cache(g, args.history_chunks, gen, layers)
    q, k, v = new_tokens(g, gen, args.fused_qkv)
    q4, k4, v4 = q[None], k[None], v[None]
    state: dict = {}
    attns = [make_attn(args.backend, g, i, mgr, state) for i in range(layers)]

    def forward_layer(attn, causal_block_size):
        plan = mgr.plan(chunk, causal_block_size)
        if args.backend == "trtllm":
            return attn.forward(q4, k4, v4, batch_size=1, seq_len=chunk, kv=plan)
        return attn.forward(q4, k4, v4, kv=plan)

    def forward(causal_block_size):
        for attn in attns:
            forward_layer(attn, causal_block_size)

    stale = max(0, mgr.history_tokens - g.window)
    print(
        f"chunk cycle: {layers} layers, backend {args.backend}, pinned {g.pinned} "
        f"(prompt {g.prompt_len} + {g.sink_frames} sink frames), history "
        f"{mgr.history_tokens} [{stale} stale], chunk {chunk}, clean pass in "
        f"{g.frames_per_chunk} causal blocks of {tpf}"
    )
    # Correctness first, on the last layer (every layer holds the same values): a
    # denoising forward and the clean pass against the exact window.
    for causal_block_size in (chunk, tpf):
        out = forward_layer(attns[-1], causal_block_size).reshape(chunk, g.num_heads, g.head_dim)
        ref = torch.cat(
            [
                exact_reference(
                    q.float(), kp, vp, hk, hv, k, v, lo, lo + causal_block_size, g.window
                )
                for lo in range(0, chunk, causal_block_size)
            ]
        )
        err = (out.float() - ref).abs().max().item()
        name = "denoising" if causal_block_size == chunk else "clean pass"
        print(f"check {name:10s} max|err| vs fp32 exact window = {err:.4f}")
        if err > MAX_ERR:
            raise SystemExit(f"chunk cycle: {name} deviates from the exact window by {err}")

    timer = CuptiTimer(args.iters, args.warmup, not args.no_l2_flush, not args.no_graph)
    denoise = timer.time(lambda: forward(None), "denoise")
    clean = timer.time(lambda: forward(tpf), "clean")

    # Commit: GPU span by CUPTI (eager, no graph): first kernel start to last kernel
    # end, so it includes launch gaps between the kernels; host time by the wall clock.
    # Each commit at steady state drops pages, so the rotation and the private-page
    # copies for both blockings are exercised every time. Nothing else runs between a
    # commit and the next forward: the kernels read the cache's tensors directly.
    def commit():
        mgr.commit(mgr.max_staged_tokens)

    commit_dev = CuptiTimer(args.iters, args.warmup, False, False).time(commit, "commit")
    host = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        commit()
        torch.cuda.synchronize()
        host.append((time.perf_counter() - t0) * 1e3)
    host_ms = statistics.median(host)

    # Elapsed spans throughout: the forwards' medians and the commit's first-to-last
    # kernel span, so the total is wall time on the device, gaps included.
    steps = args.denoise_steps
    forwards = steps * denoise["median_us"] + clean["median_us"]
    total = forwards + commit_dev["median_us"]
    print(f"\n{'per chunk, all layers':34s} {'median':>10s}")
    print(f"{f'{steps} denoising forwards':34s} {steps * denoise['median_us'] / 1e3:9.2f} ms")
    print(f"{'1 clean pass':34s} {clean['median_us'] / 1e3:9.2f} ms")
    print(
        f"{'commit, kernels (sum)':34s} {commit_dev['busy_us'] / 1e3:9.2f} ms  "
        f"({commit_dev['kernels']:.0f} kernels)"
    )
    print(f"{'commit, first to last kernel':34s} {commit_dev['median_us'] / 1e3:9.2f} ms")
    print(f"{'commit, host wall (incl. above)':34s} {host_ms:9.2f} ms")
    print(
        f"{'total device per chunk':34s} {total / 1e3:9.2f} ms  "
        f"(commit share {100 * commit_dev['median_us'] / total:.2f}%)"
    )
    print(
        f"{'per layer per forward':34s} denoise {denoise['median_us'] / layers:.1f} us, "
        f"clean {clean['median_us'] / layers:.1f} us"
    )
    if args.json:
        common = {"backend": args.backend, "layers": layers, "pool_bytes": pool_bytes(mgr, layers)}
        records = [
            {**common, "variant": "denoise", **denoise},
            {**common, "variant": "clean", **clean},
            {**common, "variant": "commit", **commit_dev, "host_ms": host_ms},
        ]
        write_json(args.json, args, g, records)
    mgr.shutdown()
    timer.cupti.finalize()  # exactly once per process


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["trtllm", "cudnn_paged", "sdpa_resident", "naive", "all"],
        default="all",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="cosmos3-sim")
    geometry = ap.add_argument_group("geometry (each overrides the preset)")
    geometry.add_argument("--num-heads", type=int, help="query heads of the whole model")
    geometry.add_argument("--num-kv-heads", type=int, help="K/V heads of the whole model")
    geometry.add_argument("--head-dim", type=int)
    geometry.add_argument("--tokens-per-frame", type=int)
    geometry.add_argument("--frames-per-chunk", type=int)
    geometry.add_argument("--window-frames", type=int, help="recent history frames")
    geometry.add_argument("--sink-frames", type=int, help="frames pinned after the prompt")
    geometry.add_argument("--prompt-len", type=int, help="prompt tokens")
    geometry.add_argument("--tokens-per-page", type=int, help="cache page size")
    geometry.add_argument("--dtype", choices=sorted(DTYPES))
    geometry.add_argument(
        "--ulysses", type=int, default=1, help="sequence-parallel ranks: heads per rank"
    )
    ap.add_argument(
        "--history-chunks",
        type=int,
        default=40,
        help="chunks committed before timing; > window/chunk so the table has rotated",
    )
    ap.add_argument(
        "--causal-block-size",
        type=int,
        default=0,
        help="tokens per causal block: tokens-per-frame (the clean pass, one causal block "
        "per frame) or 0 = one causal block of the whole chunk (a denoising step)",
    )
    ap.add_argument(
        "--fused-qkv",
        action="store_true",
        help="slice q, k, v from one buffer, as the model's fused qkv_proj does",
    )
    ap.add_argument("--no-l2-flush", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dump-kernels", action="store_true", help="print kernel names for iteration 0"
    )
    ap.add_argument("--json", metavar="PATH", help="write one record per variant to PATH")
    ap.add_argument(
        "--chunk-cycle",
        action="store_true",
        help="time one chunk across --layers layers: --denoise-steps denoising forwards, "
        "the clean pass, and the commit with its bookkeeping",
    )
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--denoise-steps", type=int, default=4)
    ap.add_argument("--backend", choices=["cudnn_paged", "trtllm"], default="cudnn_paged")
    return ap.parse_args()


def geometry_from(args) -> Geometry:
    fields = dict(PRESETS[args.preset])
    for name in fields:
        if getattr(args, name) is not None:
            fields[name] = getattr(args, name)
    for name in ("num_heads", "num_kv_heads"):
        if fields[name] % args.ulysses:
            raise SystemExit(f"--ulysses {args.ulysses} does not divide {name} {fields[name]}")
        fields[name] //= args.ulysses
    return Geometry(**fields)


def main() -> None:
    args = parse_args()
    g = geometry_from(args)
    if args.chunk_cycle:
        chunk_cycle(args, g, torch.Generator(device=DEV).manual_seed(args.seed))
        return
    chunk, dtype = g.chunk, g.torch_dtype
    causal_block_size = args.causal_block_size or chunk
    if causal_block_size not in (chunk, g.tokens_per_frame):
        raise SystemExit(
            f"--causal-block-size must be {g.tokens_per_frame} (one frame) or 0 (the chunk)"
        )
    num_causal_blocks = chunk // causal_block_size
    causal_blocks = [
        (i * causal_block_size, (i + 1) * causal_block_size) for i in range(num_causal_blocks)
    ]

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    mgr, kp, vp, k_hist, v_hist = build_cache(g, args.history_chunks, gen)
    start = mgr.staging_offset
    seq_len = start + chunk

    q, k, v = new_tokens(g, gen, args.fused_qkv)
    stale = max(0, mgr.history_tokens - g.window)
    print(
        f"geometry: q={chunk} keys={seq_len} (pinned {g.pinned} = prompt {g.prompt_len} + "
        f"{g.sink_frames} sink frames; history {mgr.history_tokens} [{stale} stale]; chunk "
        f"{chunk}) in {num_causal_blocks} causal block(s) of {causal_block_size}; "
        f"heads={g.num_heads}/{g.num_kv_heads} per rank d={g.head_dim} "
        f"page={mgr.tokens_per_page} {g.dtype}{' fused qkv' if args.fused_qkv else ''}"
    )
    if not stale:
        print(
            "note: no stale tokens resident, so exact and resident answers coincide and the "
            "check cannot tell them apart; pick a window that is not a page multiple",
            file=sys.stderr,
        )

    trtllm_attn = make_attn("trtllm", g, 0, mgr, {})
    cudnn_attn = make_attn("cudnn_paged", g, 0, mgr, {})
    q4, k4, v4 = q[None], k[None], v[None]
    plan = mgr.plan(chunk, causal_block_size)

    def run_trtllm():
        return trtllm_attn.forward(q4, k4, v4, batch_size=1, seq_len=chunk, kv=plan)

    def run_cudnn_paged():
        return cudnn_attn.forward(q4, k4, v4, kv=plan)

    # Dense paths have no per-causal-block key count, so causal blocks cost them one launch each:
    # that is the per-frame clean pass as the reference runs it.
    def per_block(attend):
        if num_causal_blocks == 1:
            return attend(q, seq_len)
        return torch.cat([attend(q[lo:hi], start + hi) for lo, hi in causal_blocks])

    # A contiguous per-layer buffer: the pinned prefix and the resident history (stale
    # tokens included: it has no per-causal-block start) are already in place; a call
    # writes the new tokens and attends over a slice.
    slab_k = torch.empty(mgr.capacity, g.num_kv_heads, g.head_dim, device=DEV, dtype=dtype)
    slab_v = torch.empty_like(slab_k)
    n_res = mgr.history_tokens
    resident_k, resident_v = torch.cat((kp, k_hist[-n_res:])), torch.cat((vp, v_hist[-n_res:]))
    slab_k[:start].copy_(resident_k)
    slab_v[:start].copy_(resident_v)

    def run_sdpa_resident():
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
        "sdpa_resident": run_sdpa_resident,
        "naive": run_naive,
    }
    chosen = list(variants) if args.variant == "all" else [args.variant]
    # Cross-check before timing: a silent fallback that ignores the prefix, a causal
    # block that sees the causal blocks after it, or a paged path that reads stale
    # tokens shows up here.
    all_k, all_v = torch.cat((resident_k, k)).float(), torch.cat((resident_v, v)).float()
    ref_resident = torch.cat(
        [
            sdpa(q[lo:hi].float(), all_k[: start + hi], all_v[: start + hi])
            for lo, hi in causal_blocks
        ]
    )
    ref_exact = torch.cat(
        [
            exact_reference(q.float(), kp, vp, k_hist, v_hist, k, v, lo, hi, g.window)
            for lo, hi in causal_blocks
        ]
    )
    chunk_only = sdpa(q.float(), k.float(), v.float())
    checks = {}
    for name in chosen:
        out = variants[name]().reshape(chunk, g.num_heads, g.head_dim).float()
        torch.cuda.synchronize()
        if name in ("trtllm", "cudnn_paged"):
            checks[name] = check(name, out, ref_exact, ref_resident, chunk_only, "exact window")
        else:
            checks[name] = check(name, out, ref_resident, ref_exact, chunk_only, "resident")

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

    print(f"\n{'variant':14s} {'median us':>10s} {'min us':>9s} {'p90 us':>9s} {'kernels':>8s}")
    for name, r in rows:
        print(
            f"{name:14s} {r['median_us']:10.1f} {r['min_us']:9.1f} {r['p90_us']:9.1f} "
            f"{r['kernels']:8.0f}{'' if r['graph'] else '  (eager)'}"
        )
    if args.json:
        pool = pool_bytes(mgr, 1)
        records = [
            {
                "variant": name,
                "causal_block_size": causal_block_size,
                "pool_bytes": pool if name in ("trtllm", "cudnn_paged") else None,
                **checks[name],
                **r,
            }
            for name, r in rows
        ]
        write_json(args.json, args, g, records)
    mgr.shutdown()
    timer.cupti.finalize()  # exactly once per process


if __name__ == "__main__":
    main()
