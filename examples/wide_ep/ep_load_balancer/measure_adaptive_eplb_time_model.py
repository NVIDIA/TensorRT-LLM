# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate adaptive EPLB's MegaMoE CuteDSL decode time model: one ``scale`` per CUDA graph bucket.

Random slot-row designs per bucket (``local batch x (1 + draft tokens)`` tokens per GPU) run as
bench_moe forced routing patterns (slots as experts), timed as the slowest rank's median launch.
Per bucket, scale = the slope of the times on ``decode_time_model``'s times at scale 1 (an intercept
per bucket); it is accepted when the scale of every fourth design, held out, agrees with the others'
within 3 standard errors (bootstrap over the designs). Every check that can fail runs before the
bucket's first launch and ends in a vote of all ranks; rank 0 writes once every rank computed the
same file, and a failure after a launch aborts every rank. GPU: ``PYTHONPATH=tests/microbenchmarks
mpirun -np <EP> python3 <file> --out D [--local-batches B ..] [--rates BW PEAK]`` writes
``D/calibration.json`` (``MoeAdaptiveLoadBalancerConfig.time_model_file``) and every design's times
in ``D/times.json``; ``--dry-run EP`` (CPU) fits the model's own times plus noise.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

KEYS = ("per_active_slot_us", "per_extra_tile_us", "per_row_us", "hot_slot_extra_tile_us")
WARMUP, ITERS, REPS = 2, 10, 3  # graph replays before timing, launches per graph, passes
DESIGNS, SLOTS, SYNTHETIC_SCALE = 128, 16, 0.9  # per local batch; per GPU (DSV4's 512 at EP32)


def make_designs(g, n, rng):
    """``n`` designs ``[ep, slots]``: random active slots, rows per GPU, concentration, hot slots."""
    total, cap, out = g.ep * g.T * g.K, g.ep * g.T, []  # a slot takes a token at most once
    while len(out) < n:
        hot = np.zeros((g.ep, g.M))
        for r in rng.choice(g.ep, rng.integers(0, g.ep // 4 + 2), replace=False):
            hot[r, 0] = min(cap, g.tile * rng.integers(1, cap // g.tile + 2)) - rng.integers(64)
        w = rng.lognormal(0, rng.uniform(0, 1.5), hot.shape) * (hot == 0)
        w *= (np.arange(g.M) < rng.integers(1, g.M + 1, (g.ep, 1))) * rng.lognormal(0, 0.7)
        rest = int(total - hot.sum())
        x = hot + rng.multinomial(max(rest, 0), (w / w.sum()).ravel()).reshape(w.shape)
        if rest >= 0 and x.max() <= cap:
            out.append(x.astype(np.int64))
    return np.stack(out)


def pattern(x):
    """bench_moe routing pattern of design ``x``: each GPU sends every GPU 1 / ep of its rows."""
    ep, rows = len(x), x.sum(1)
    dispatch = np.tile(rows // ep, (ep, 1))
    owed = np.repeat(np.arange(ep), rows % ep)  # the remainders, dealt to the sources in turn
    np.add.at(dispatch, (np.arange(len(owed)) % ep, owed), 1)
    out = dict(ep_size=ep, experts_per_rank=x.shape[1], expert_histogram=x.tolist())
    return dict(out, slot_dispatch_matrix=dispatch.tolist())


def model_terms(xs, sizes, expert_bytes, device, rates):
    """Unit costs at scale 1 of ``rates`` (bandwidth, peak, scale) and ``[N, ep, 4]`` unit-cost terms."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import TimeModel
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl as Mega

    costs = Mega.decode_time_model(SimpleNamespace(**sizes), expert_bytes, device, rates)
    flat = xs.reshape(-1, xs.shape[-1]).astype(float)
    units = [TimeModel(*w, scale=1.0, tile_rows=costs["tile_rows"]) for w in np.eye(4)]
    terms = np.stack([m.layer_time_us(flat, 1) for m in units], -1).reshape(*xs.shape[:2], 4)
    return costs, terms


def fit_bucket(p, y, rng, draws=1000):
    """``(scale, its SE, z)`` of a bucket's designs.

    Scale is the slope of ``y`` on ``p`` with an intercept and SE its bootstrap over the designs; z
    is the slope of every fourth design, held out, minus the others' in standard errors of that
    difference (the same bootstrap).
    """
    idx = np.arange(len(p))
    out, fit = idx[idx % 4 == 3], idx[idx % 4 != 3]

    def slope(i):
        return np.polyfit(p[i], y[i], 1)[0]

    def draw(i):
        return slope(rng.choice(i, len(i)))

    boot = np.array([(draw(idx), draw(out) - draw(fit)) for _ in range(draws)])
    return slope(idx), boot[:, 0].std(ddof=1), (slope(out) - slope(fit)) / boot[:, 1].std(ddof=1)


def routes(path, g, src, device):
    """bench_moe's forced routes of rank ``src``; its rows per slot and tokens taking a slot twice."""
    import bench_moe as bm
    import torch

    spec = bm.RoutingControlSpec(routing_mode="forced", routing_pattern_file=str(path))
    plan = bm._build_routing_plan(spec, g.ep * g.T, g.ep, g.K, g.ep * g.M, g.ep, True)
    ids, scales = bm._materialize_selected_experts_for_rank(
        plan, src, g.K, g.M, g.ep, device, torch.bfloat16
    )
    s = np.sort(ids.cpu().numpy(), 1)
    count = np.bincount(s.ravel(), minlength=g.ep * g.M)
    return (ids, scales), np.append(count, (s[:, 1:] == s[:, :-1]).sum())


def time_designs(g, m, paths, exact, rank, gather, hosts):
    """GPU: ``[REPS, N]`` us of the designs routed exactly, and the voted calibration identity.

    bench_moe's MegaMoE CuteDSL layer times the designs; every rank votes for the identity before
    the first launch.
    """
    import bench_moe as bm
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer as ab

    bf16, cuda, tokens = torch.bfloat16, torch.device("cuda"), [g.T] * g.ep
    spec = (m.hidden_size, m.intermediate_size, "NVFP4", "RENORMALIZE")
    model = bm.ModelSpec("custom", g.ep * g.M, g.K, *spec)
    config = bm.ConfigSpec(backend="MEGAMOE_CUTEDSL", parallel_mode="DEP", cuda_graph=True)
    mapping = bm._build_mapping_from_config(config, g.ep)
    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()
    flags = dict(
        use_cuda_graph=True, use_low_precision_moe_combine=False, enable_perfect_router=False
    )
    kinds = dict(dtype=bf16, routing_logits_dtype=bf16, device=cuda, max_num_tokens=g.T)
    moe = bm._build_moe_module(
        model=model, config=config, mapping=mapping, moe_backend=config.backend, **flags, **kinds
    )[0]
    x, logits = bm._make_inputs(g.T, m.hidden_size, g.ep * g.M, bf16, bf16, cuda, seed=rank)
    backend = getattr(moe, "backend", moe)
    identity = ab.calibration_identity(backend, g.ep, hosts, [g.T])
    if len({json.dumps(v, sort_keys=True) for v in gather(identity)}) != 1:
        raise SystemExit(f"bucket {g.T}: the ranks' identities differ; nothing launched")
    y = np.full((REPS, len(paths)), np.nan)
    for rep in range(REPS):  # the same order on every rank: the kernel is collective
        for d in np.random.default_rng([rep, g.T]).permutation(np.flatnonzero(exact)):
            ids_scales = routes(paths[d], g, rank, cuda)[0]
            with bm._maybe_install_routing_control_patch(moe, *ids_scales, active=True):
                if np.isnan(y).all():
                    bm._run_autotune(moe, x, logits, tokens, False)
                t = bm._time_moe_forward_cuda_graph(
                    moe, x, logits, tokens, warmup=WARMUP, iters=ITERS
                )
            y[rep, d] = 1e3 * np.median(
                bm._slowest_rank_iter_maxes(bm._gather_per_iteration_times(t[0]))
            )
    moe.destroy()
    return y, identity


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--draft-tokens", type=int, default=3, help="per request per forward (MTP-3)")
    ap.add_argument(
        "--rates", type=float, nargs=2, help="HBM bytes/s and dense FP4 FLOP/s of a GPU"
    )
    ap.add_argument("--dry-run", type=int, metavar="EP", help="CPU, no launches: the model's times")
    a = ap.parse_args()
    import bench_moe as bm
    import torch

    import tensorrt_llm
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer as ab
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl as Mega
    from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size

    m = bm.BUILT_IN_MODELS["deepseek_v4_pro"]
    ep, rank = (a.dry_run, 0) if a.dry_run else (mpi_world_size(), mpi_rank())
    gather, hosts = (lambda v: [v]), 1
    if not a.dry_run:
        gather = mpi_allgather
        bm._set_device_from_local_rank()
        why, hosts = ab.nvlink_domain_error(SimpleNamespace(allgather=gather))
        if why:
            raise SystemExit(f"nothing launched: {why}")
    device = "GB300" if a.dry_run else torch.cuda.get_device_name()
    entry = next((v for key, v in Mega._DECODE_DEVICES.items() if key in device), None)
    rates = (*a.rates, 1.0) if a.rates else entry
    if rates is None:
        raise SystemExit(f"no built-in rates for {device!r}: pass --rates")
    sizes = dict(hidden_size=m.hidden_size, intermediate_size_per_partition=m.intermediate_size)
    sizes.update(expand_intermediate_size_per_partition=2 * m.intermediate_size)  # gated FC1
    expert_bytes = m.hidden_size * 3 * m.intermediate_size * 9 // 16  # NVFP4, an FP8 scale per 16
    costs = model_terms(np.zeros((1, 1, 1)), sizes, expert_bytes, device, rates)[0]
    buckets, times, identity, tactics = [], {}, {}, {}
    for lb in a.local_batches:  # every rank draws the same designs
        T = lb * (1 + a.draft_tokens)
        g = SimpleNamespace(ep=ep, M=SLOTS, T=T, K=m.top_k, tile=costs["tile_rows"])
        rng = np.random.default_rng([ep, lb])
        xs = make_designs(g, DESIGNS, rng)
        paths = [a.out / f"lb{lb}" / f"{d:03d}.json" for d in range(len(xs))]
        for path, x in zip(paths, xs) if rank == 0 else ():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pattern(x)))
        if not a.dry_run:
            mpi_barrier()
        sources, on = (range(ep), "cpu") if a.dry_run else ([rank], "cuda")
        counts = np.stack([sum(routes(path, g, src, on)[1] for src in sources) for path in paths])
        counts = counts if a.dry_run else np.sum(mpi_allgather(counts), 0)
        exact = (counts[:, :-1] == xs.reshape(len(xs), -1)).all(1) & (counts[:, -1] == 0)
        terms = model_terms(xs[exact], sizes, expert_bytes, device, rates)[1]
        p = (terms @ [costs[k] for k in KEYS]).max(1)
        if a.dry_run:
            y = lb + SYNTHETIC_SCALE * p + rng.normal(0.0, 0.5, (REPS, len(p)))
            identity = dict(device=device, ep_size=ep, hosts=1, tactics={str(T): "dry-run"})
        else:
            y, identity = time_designs(g, m, paths, exact, rank, gather, hosts)
            y = y[:, exact]
        tactics.update(identity["tactics"])
        k, se, z = fit_bucket(p, y.mean(0), np.random.default_rng([ep, lb, 1]))
        noise = np.sqrt(np.var(y, 0, ddof=1).mean() / REPS)
        i, j = np.triu_indices(len(p), 1)
        big = np.abs(y.mean(0)[i] - y.mean(0)[j]) > 3 * np.sqrt(2) * noise
        conc = np.mean(np.sign(y.mean(0)[i] - y.mean(0)[j]) == np.sign(p[i] - p[j]), where=big)
        rmse = np.std(y.mean(0) - k * p)
        buckets.append(dict(tokens=T, scale=round(float(k), 4), se=round(float(se), 4)))
        buckets[-1].update(holdout_z=round(float(z), 2), designs=int(exact.sum()))
        buckets[-1].update(rmse_us=round(float(rmse), 2), concordance=round(float(conc), 3))
        times[str(T)] = dict(
            exact=exact.tolist(), model_us=p.round(3).tolist(), us=y.round(3).tolist()
        )
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gather(time.time())[0]))
    measured = dict(trtllm_version=tensorrt_llm.__version__, utc=utc)
    measured.update(designs=DESIGNS, slots=SLOTS, reps=REPS, draft_tokens=a.draft_tokens)
    doc = dict(zip(("schema", "version"), ab.CALIBRATION), identity=dict(identity, tactics=tactics))
    doc.update(rates=dict(hbm_bytes_per_s=rates[0], fp4_dense_flops_per_s=rates[1]))
    doc.update(buckets=buckets, measurement=measured)
    text = json.dumps(doc, indent=1, sort_keys=True)
    refused = [b["tokens"] for b in buckets if not abs(b["holdout_z"]) <= 3.0]
    if len(set(gather(hashlib.sha256(text.encode()).hexdigest()))) != 1:
        refused.append("the ranks computed different files")
    if rank == 0:
        (a.out / "times.json").write_text(json.dumps(times))
        for lb, bucket in zip(a.local_batches, buckets):
            print(f"lb {lb}: {json.dumps(bucket)}")
        if refused:
            print(f"refused, no calibration written: {refused}")
        else:
            (a.out / "calibration.json.tmp").write_text(text)
            os.replace(a.out / "calibration.json.tmp", a.out / "calibration.json")
            scales = {b["tokens"]: b["scale"] for b in buckets}
            print(f"scale per bucket {json.dumps(scales)} in {a.out / 'calibration.json'}")
    return 1 if refused else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        if "--dry-run" not in sys.argv:  # a rank failing after a launch leaves the others waiting
            from mpi4py import MPI

            traceback.print_exc()
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
        raise
