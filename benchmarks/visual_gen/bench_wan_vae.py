# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan VAE end-to-end benchmark: the native WanVAE against WinoVae.

Decode and encode at the Cosmos3 shapes, with random weights (seeded) or a checkpoint, random
inputs or a real clip. CUDA-graph replay timed with CUPTI kernel timestamps, L2 flushed between
replays; accuracy against the native model run in fp32. Writes one JSON per run.

    python benchmarks/visual_gen/bench_wan_vae.py --label wino --out-dir /tmp/wan_vae
        --shapes t2v_720p_189f --winovae-opts fused_norm=1,fused_residual=1,...
    python benchmarks/visual_gen/bench_wan_vae.py ... --world-size 2 --split-dim width
"""

from __future__ import annotations

import argparse
import atexit
import bisect
import datetime as dt
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

# Wan2.2-TI2V-5B VAE, as shipped in Cosmos3 checkpoints' vae/config.json.
COSMOS_VAE_CONFIG: dict[str, Any] = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
}

# Cosmos3 defaults (tensorrt_llm/_torch/visual_gen/models/cosmos3/defaults.py):
# T2I 1024x1024, 720p video 189 frames, Edge 480p video 121 frames.
SHAPES: dict[str, tuple[int, int, int]] = {
    "t2i_1024": (1, 1024, 1024),
    "t2v_720p_189f": (189, 720, 1280),
    # 720p over the frame count of the sample clips (real-input accuracy runs).
    "t2v_720p_121f": (121, 720, 1280),
    "edge_480p_121f": (121, 480, 832),
    # Cosmos3-Nano-Sim-Bimanual streaming decode: 901 pixel frames = 226 latent frames,
    # decoded 2 latent frames (8 pixel frames) per step after a 1-frame cold start.
    "bimanual_480x832_901f": (901, 480, 832),
    # Its conditioning input: one 480x832 image encoded once per episode.
    "bimanual_cond_480x832": (1, 480, 832),
}
# Decoder temporal chunk size (latent frames per call) a shape is decoded with when the
# command line does not override it.
SHAPE_DECODE_CHUNK: dict[str, int] = {"bimanual_480x832_901f": 2}

_L2_FLUSH_BYTES = 128 * 1024 * 1024
_l2_flush: torch.Tensor | None = None

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("attention", re.compile(r"fmha|flash|sdpa|attention", re.I)),
    (
        "conv",
        re.compile(r"cudnn|conv|gemm|cutlass|implicit|xmma|nhwc|nchw|wgrad|dgrad|sm\d+_", re.I),
    ),
    (
        "copy",
        re.compile(r"CatArray|copy|pad|index|gather|scatter|transpose|permute|contiguous", re.I),
    ),
    (
        "norm_act",
        re.compile(
            r"reduce|norm|silu|sigmoid|Mul|Add|Div|Sub|elementwise|unrolled|vectorized|clamp|exp|upsample|nearest",
            re.I,
        ),
    ),
]


def _classify(name: str | None) -> str:
    if name is None:
        return "memcpy_memset"
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(name):
            return category
    return "other"


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _gpu_signature() -> tuple[str, str]:
    full = torch.cuda.get_device_name(0)
    short = full.replace("NVIDIA", "").strip().replace(" ", "_")
    return full, short


def _flush_l2() -> None:
    global _l2_flush
    if _l2_flush is None:
        _l2_flush = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    _l2_flush.fill_(0.0)
    torch.cuda.synchronize()


def _latent_shape(T: int, H: int, W: int) -> tuple[int, int, int, int, int]:
    return (1, COSMOS_VAE_CONFIG["z_dim"], (T - 1) // 4 + 1, H // 16, W // 16)


def _pixel_shape(T: int, H: int, W: int) -> tuple[int, int, int, int, int]:
    return (1, 3, T, H, W)


def _real_pixels(video: str | None, image: str | None, T: int, H: int, W: int) -> torch.Tensor:
    """Real pixels as ``[1, 3, T, H, W]`` fp32 in [-1, 1] on the GPU.

    The first T frames of ``video`` (or ``image`` when T == 1 and one is given), resized to
    H x W with antialiasing.
    """
    import numpy as np

    if T == 1 and image:
        from PIL import Image

        frames = [np.asarray(Image.open(image).convert("RGB"))]
    else:
        if not video:
            raise ValueError(f"a {T}-frame shape needs --video")
        import av

        frames = []
        with av.open(video) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == T:
                    break
        if len(frames) < T:
            raise ValueError(f"{video} has {len(frames)} frames, the shape needs {T}")
    x = torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1  # T,3,h,w
    if x.shape[-2:] != (H, W):
        x = torch.nn.functional.interpolate(
            x, size=(H, W), mode="bilinear", antialias=True, align_corners=False
        )
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _reference_state_dict(seed: int) -> dict[str, torch.Tensor]:
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE

    torch.manual_seed(seed)
    model = WanVAE(COSMOS_VAE_CONFIG)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _build(
    backend: str,
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype,
    winovae_opts: dict[str, Any],
) -> torch.nn.Module:
    if backend == "native":
        from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE

        model = WanVAE(COSMOS_VAE_CONFIG)
    elif backend == "winovae":
        from tensorrt_llm._torch.visual_gen.models.wan.wino_vae import WinoVae

        model = WinoVae(COSMOS_VAE_CONFIG, winovae_opts)
    else:
        raise ValueError(backend)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device="cuda", dtype=dtype).eval().requires_grad_(False)


def _init_parallel(world_size: int, split_dim: str):
    """Join the torchrun process group and return ``(rank, wrapper)``.

    ``wrapper`` turns a single-GPU VAE into its spatially sharded ParallelVAE; without
    parallelism the result is ``(0, None)``.
    """
    if world_size <= 1:
        return 0, None
    import torch.distributed as dist

    from tensorrt_llm._torch.visual_gen.modules.vae.parallel_vae_interface import ParallelVAEFactory

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != world_size:
        raise RuntimeError(
            f"torchrun started {dist.get_world_size()} ranks, --world-size is {world_size}"
        )
    ranks = list(range(world_size))
    pg = dist.new_group(ranks, use_local_synchronization=False)
    adj_groups = [
        dist.new_group([ranks[i], ranks[i + 1]], use_local_synchronization=False)
        for i in range(world_size - 1)
    ]

    def wrap(model: torch.nn.Module) -> torch.nn.Module:
        return ParallelVAEFactory.from_vae(model, split_dim, pg, adj_groups).eval()

    return dist.get_rank(), wrap


def _make_op(
    model: torch.nn.Module, op: str, inp: torch.Tensor, decode_chunk: int
) -> Callable[[], torch.Tensor]:
    if op == "decode":
        return lambda: model.decode(inp, return_dict=False, temporal_chunk_size=decode_chunk)[0]
    if op == "encode":
        return lambda: model.encode(inp, return_dict=False)[0].mode()
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _try_capture(run_fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph | None:
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.inference_mode(), torch.cuda.graph(graph):
            run_fn()
        torch.cuda.synchronize()
        return graph
    except RuntimeError as exc:  # capture-incompatible path: fall back to eager replay
        torch.cuda.synchronize()
        print(f"    [graph] capture failed, timing eager: {exc}", file=sys.stderr)
        return None


TOP_KERNELS = 15  # kernels kept in each timing record, largest first


def _time_cupti(
    run_fn: Callable[[], Any], warmup: int, iters: int, use_graph: bool
) -> dict[str, Any]:
    from cupti import cupti

    with torch.inference_mode():
        run_fn()
    torch.cuda.synchronize()

    graph = _try_capture(run_fn) if use_graph else None
    replay = graph.replay if graph is not None else run_fn

    with torch.inference_mode():
        for _ in range(warmup):
            _flush_l2()
            replay()
        torch.cuda.synchronize()

    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[int, int, int, str | None]] = []

    def _buf_requested():
        return 8 * 1024 * 1024, 0

    def _buf_completed(activities):
        for a in activities:
            if a.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                name = a.name if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else None
                kernels.append((a.start, a.end, a.correlation_id, name))
            elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((a.start, a.end, a.correlation_id))

    kinds = [
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ]
    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(_buf_requested, _buf_completed)

    windows: list[tuple[int, int]] = []
    with torch.inference_mode():
        for _ in range(iters):
            _flush_l2()
            t0 = cupti.get_timestamp()
            replay()
            torch.cuda.synchronize()
            t1 = cupti.get_timestamp()
            windows.append((t0, t1))

    cupti.activity_flush_all(0)
    for kind in kinds:
        cupti.activity_disable(kind)
    # No cupti.finalize() here: once per process, at exit.

    corr_to_kernels: dict[int, list[tuple[int, int, int, str | None]]] = {}
    for k in kernels:
        corr_to_kernels.setdefault(k[2], []).append(k)
    sorted_launches = sorted(launches, key=lambda e: e[0])
    launch_starts = [e[0] for e in sorted_launches]
    kernels_by_start = sorted(kernels, key=lambda k: k[0])
    kernel_starts = [k[0] for k in kernels_by_start]

    span_us: list[float] = []
    busy_us: list[float] = []
    wall_us: list[float] = []
    n_kernels: list[int] = []
    category_ns: dict[str, int] = {}
    kernel_ns: dict[str, int] = {}
    for idx, (t0, t1) in enumerate(windows):
        li = bisect.bisect_left(launch_starts, t0)
        ri = bisect.bisect_right(launch_starts, t1)
        corr_ids = {sorted_launches[i][2] for i in range(li, ri)}
        iter_kernels: list[tuple[int, int, int, str | None]] = []
        for cid in corr_ids:
            iter_kernels.extend(corr_to_kernels.get(cid, []))
        if not iter_kernels:
            # Fall back to the timestamp window (same clock as the CUPTI records).
            ki = bisect.bisect_left(kernel_starts, t0)
            kj = bisect.bisect_right(kernel_starts, t1)
            iter_kernels = [k for k in kernels_by_start[ki:kj] if k[1] <= t1]
        if not iter_kernels:
            raise RuntimeError(f"CUPTI recorded no kernels for iteration {idx}")
        min_start = min(k[0] for k in iter_kernels)
        max_end = max(k[1] for k in iter_kernels)
        span_us.append((max_end - min_start) / 1e3)
        busy_us.append(sum(k[1] - k[0] for k in iter_kernels) / 1e3)
        wall_us.append((t1 - t0) / 1e3)
        n_kernels.append(len(iter_kernels))
        for k in iter_kernels:
            d = k[1] - k[0]
            category_ns[_classify(k[3])] = category_ns.get(_classify(k[3]), 0) + d
            short = (k[3] or "<memcpy/memset>")[:160]
            kernel_ns[short] = kernel_ns.get(short, 0) + d

    def _stats(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
        return {
            "median_us": statistics.median(values),
            "p90_us": p90,
            "min_us": ordered[0],
            "max_us": ordered[-1],
        }

    top = sorted(kernel_ns.items(), key=lambda kv: -kv[1])[:TOP_KERNELS]
    return {
        "graph": graph is not None,
        "iters": iters,
        "span": _stats(span_us),
        "busy": _stats(busy_us),
        "wall": _stats(wall_us),
        "kernels_per_iter": statistics.median(n_kernels),
        "category_us_per_iter": {k: v / 1e3 / iters for k, v in sorted(category_ns.items())},
        "top_kernels_us_per_iter": [{"name": n, "us": v / 1e3 / iters} for n, v in top],
    }


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.float()
    e = expected.float()
    diff = a - e
    return {
        "rel_l2": (diff.norm() / e.norm().clamp_min(1e-12)).item(),
        "max_abs": diff.abs().max().item(),
        "rms": diff.pow(2).mean().sqrt().item(),
        "nan_count": int(torch.isnan(a).sum().item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_winovae_opts(text: str) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, _, value = item.partition("=")
        if not value:
            opts[key] = True
        elif value.lower() in ("1", "true", "yes", "0", "false", "no"):
            opts[key] = value.lower() in ("1", "true", "yes")
        elif value.isdigit():
            opts[key] = int(value)
        else:
            opts[key] = value
    return opts


def _clip_span(
    model: torch.nn.Module,
    latents: torch.Tensor,
    n_frames: int,
    decode_chunk: int,
    warmup: int,
    iters: int,
    use_graph: bool,
) -> tuple[float, bool]:
    """Median GPU span (ms) and graph flag of one whole-clip decode of ``n_frames`` latents.

    Timed separately with its own warmup and capture attempt; the flag says whether the
    timed iterations ran as a CUDA graph.
    """
    sub = latents[:, :, :n_frames].contiguous(memory_format=torch.channels_last_3d)
    fn = _make_op(model, "decode", sub, decode_chunk)
    t = _time_cupti(fn, warmup, iters, use_graph=use_graph)
    return t["span"]["median_us"] / 1e3, bool(t["graph"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", required=True, help="run name, e.g. baseline, fused_norm")
    parser.add_argument(
        "--backends",
        default="native,winovae",
        help="native runs first so winovae gets a vs_native accuracy row",
    )
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--ops", default="decode,encode")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--ref-dtype",
        default="float32",
        choices=["float32", "none"],
        help="dtype of the native model used as the accuracy reference; none skips",
    )
    parser.add_argument("--winovae-opts", default="", help="comma list, e.g. fused_norm=1")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="VAE directory (diffusion_pytorch_model.safetensors); default: random weights "
        "from --seed",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="video file for the encode input of multi-frame shapes (frames resized to the "
        "shape; needs at least T frames); the decode input is then the fp32 reference's "
        "encoding of those frames",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="image file for single-frame shapes (default: the video's first frame)",
    )
    parser.add_argument(
        "--decode-chunk",
        type=int,
        default=0,
        help="temporal_chunk_size for decode (0: per-shape default, 1 unless the shape says "
        "otherwise)",
    )
    parser.add_argument(
        "--span-diffs",
        action="store_true",
        help="also time whole-clip decodes of 1, 1 + chunk and 1 + 2 * chunk latent frames and "
        "report the differences of their median GPU spans (each timed separately, with graph "
        "capture attempted unless disabled; not a measurement of an incremental streaming step)",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--top-kernels", type=int, default=15, help="kernels kept per timing record, largest first"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="GPUs for the spatially sharded parallel VAE; re-executes itself under torchrun "
        "when > 1",
    )
    parser.add_argument(
        "--split-dim",
        default="height",
        choices=["height", "width"],
        help="shard dimension of the parallel VAE when --world-size > 1",
    )
    return parser.parse_args()


def _relaunch_under_torchrun(world_size: int) -> None:
    """Re-execute this script under torchrun with ``world_size`` ranks; forwards the exit code."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    sys.exit(subprocess.call(cmd))


def _load_state_dict(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """The checkpoint's weights, or the seeded random reference weights."""
    if args.checkpoint:
        from safetensors.torch import load_file

        return load_file(str(Path(args.checkpoint) / "diffusion_pytorch_model.safetensors"))
    return _reference_state_dict(args.seed)


def _build_inputs(
    args: argparse.Namespace, shape_name: str, state_dict: dict[str, torch.Tensor], dtype
) -> dict[str, torch.Tensor]:
    """The ``{"decode": latents, "encode": pixels}`` inputs of one shape, in ``dtype``.

    Real frames when a video or image is given (the decode input is then the fp32 native
    model's encoding of them), else seeded random tensors.
    """
    T, H, W = SHAPES[shape_name]
    if args.video or args.image:
        pixels = _real_pixels(args.video, args.image, T, H, W)
        model = _build("native", state_dict, torch.float32, {})
        with torch.inference_mode():
            latents = model.encode(pixels.float(), return_dict=False)[0].mode()
        del model
        torch.cuda.empty_cache()
        return {"decode": latents.to(dtype), "encode": pixels.to(dtype)}
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    return {
        "decode": torch.randn(_latent_shape(T, H, W), generator=gen, device="cuda").to(dtype),
        "encode": (torch.rand(_pixel_shape(T, H, W), generator=gen, device="cuda") * 2 - 1).to(
            dtype
        ),
    }


def _fp32_reference(
    state_dict: dict[str, torch.Tensor], op: str, inp: torch.Tensor, decode_chunk: int
) -> torch.Tensor:
    """The native model's fp32 output for ``op`` on ``inp``, on the CPU."""
    model = _build("native", state_dict, torch.float32, {})
    with torch.inference_mode():
        reference = _make_op(model, op, inp.float(), decode_chunk)().float().cpu()
    del model
    torch.cuda.empty_cache()
    return reference


def _run_backend(
    backend: str,
    *,
    args: argparse.Namespace,
    state_dict: dict[str, torch.Tensor],
    dtype,
    winovae_opts: dict[str, Any],
    parallel,
    shape_name: str,
    op: str,
    inp: torch.Tensor,
    decode_chunk: int,
    reference: torch.Tensor | None,
    native_out: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Build ``backend``, run ``op`` once for accuracy (unless skipped), time it, print one line.

    Returns ``(result row, the output on the CPU or None when accuracy was skipped)``.
    """
    t_start = time.time()
    model = _build(backend, state_dict, dtype, winovae_opts if backend == "winovae" else {})
    if parallel is not None:
        model = parallel(model)
    run_fn = _make_op(model, op, inp, decode_chunk)

    row: dict[str, Any] = {
        "backend": backend,
        "shape": shape_name,
        "op": op,
        "input_shape": list(inp.shape),
        "decode_chunk": decode_chunk,
    }
    if parallel is not None:
        torch.distributed.barrier()
    out_cpu: torch.Tensor | None = None
    if not args.skip_accuracy:
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = run_fn()
        torch.cuda.synchronize()
        row["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        row["output_shape"] = list(out.shape)
        out_cpu = out.float().cpu()
        if reference is not None:
            row["vs_ref_fp32"] = _metrics(out_cpu, reference)
        if backend != "native" and native_out is not None:
            row["vs_native"] = _metrics(out_cpu, native_out)
        del out

    row["timing"] = _time_cupti(run_fn, args.warmup, args.iters, use_graph=not args.no_graph)
    if args.span_diffs and op == "decode" and inp.shape[2] >= 1 + 2 * decode_chunk:
        frames = [1, 1 + decode_chunk, 1 + 2 * decode_chunk]
        timed = [
            _clip_span(model, inp, n, decode_chunk, args.warmup, args.iters, not args.no_graph)
            for n in frames
        ]
        spans = [t for t, _ in timed]
        row["span_diffs"] = {
            "chunk_latent_frames": decode_chunk,
            "latent_frames": frames,
            "span_ms": spans,
            "graph": [g for _, g in timed],
            "diff_ms": [spans[1] - spans[0], spans[2] - spans[1]],
        }
        print(
            "    whole-clip decode spans for "
            + ", ".join(f"{n} latent frames: {t:.2f} ms" for n, t in zip(frames, spans))
            + f"; differences {spans[1] - spans[0]:.2f} / {spans[2] - spans[1]:.2f} ms",
            flush=True,
        )
    row["setup_s"] = time.time() - t_start

    t = row["timing"]
    acc = row.get("vs_ref_fp32", {})
    print(
        f"  {backend:9s} graph={str(t['graph']):5s} "
        f"span med {t['span']['median_us'] / 1e3:9.2f} ms "
        f"p90 {t['span']['p90_us'] / 1e3:9.2f} ms  "
        f"wall med {t['wall']['median_us'] / 1e3:9.2f} ms  "
        f"kernels/iter {t['kernels_per_iter']:.0f}  "
        + (f"rel_l2 {acc['rel_l2']:.3e} max_abs {acc['max_abs']:.3e}" if acc else ""),
        flush=True,
    )
    del model, run_fn
    torch.cuda.empty_cache()
    return row, out_cpu


def main() -> None:
    args = _parse_args()
    global TOP_KERNELS
    TOP_KERNELS = args.top_kernels

    if args.world_size > 1 and "RANK" not in os.environ:
        _relaunch_under_torchrun(args.world_size)
    dist_rank, parallel = _init_parallel(args.world_size, args.split_dim)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    dtype = getattr(torch, args.dtype)
    backends = args.backends.split(",")
    shapes = args.shapes.split(",")
    ops = args.ops.split(",")
    winovae_opts = _parse_winovae_opts(args.winovae_opts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _finalize():
        from cupti import cupti

        cupti.finalize()

    atexit.register(_finalize)

    repo = Path(__file__).resolve().parents[2]
    gpu_full, gpu_short = _gpu_signature()
    quiet = dist_rank != 0  # ranks other than 0 compute but do not report
    if quiet:
        sys.stdout = open(os.devnull, "w")

    meta = {
        "label": args.label,
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo),
        "gpu": gpu_full,
        "gpu_short": gpu_short,
        "world_size": args.world_size,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "dtype": args.dtype,
        "winovae_opts": winovae_opts,
        "decode_chunk": args.decode_chunk or "per-shape",
        "iters": args.iters,
        "warmup": args.warmup,
        "cudnn_benchmark": args.cudnn_benchmark,
        "graph_requested": not args.no_graph,
        "argv": sys.argv[1:],
    }
    print(json.dumps(meta, indent=1))

    state_dict = _load_state_dict(args)
    if args.checkpoint:
        meta["checkpoint"] = args.checkpoint
    use_reference = not args.skip_accuracy and args.ref_dtype != "none"
    if args.video or args.image:
        if not use_reference:
            raise ValueError(
                "real inputs need the fp32 reference (their decode input is its encoding)"
            )
        meta["video"], meta["image"] = args.video, args.image
    results: list[dict[str, Any]] = []

    for shape_name in shapes:
        inputs = _build_inputs(args, shape_name, state_dict, dtype)
        decode_chunk = args.decode_chunk or SHAPE_DECODE_CHUNK.get(shape_name, 1)
        for op in ops:
            if op not in inputs:
                raise ValueError(op)
            inp = inputs[op]
            print(
                f"\n=== {shape_name} {op} input={tuple(inp.shape)} chunk={decode_chunk} ===",
                flush=True,
            )
            reference = (
                _fp32_reference(state_dict, op, inp, decode_chunk) if use_reference else None
            )
            native_out: torch.Tensor | None = None
            for backend in backends:
                row, out_cpu = _run_backend(
                    backend,
                    args=args,
                    state_dict=state_dict,
                    dtype=dtype,
                    winovae_opts=winovae_opts,
                    parallel=parallel,
                    shape_name=shape_name,
                    op=op,
                    inp=inp,
                    decode_chunk=decode_chunk,
                    reference=reference,
                    native_out=native_out,
                )
                if backend == "native":
                    native_out = out_cpu
                results.append(row)
            del reference, native_out
        del inputs
        torch.cuda.empty_cache()

    if not quiet:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"run_{stamp}_{args.label}_{gpu_short}_ws{args.world_size}.json"
        out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"\nwrote {out_path}")
    if parallel is not None:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
