# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure ``scale`` of adaptive EPLB's MegaMoE CuteDSL decode time model on this GPU and backend.

Random slot-row designs per local batch run as bench_moe forced routing patterns (slots as its
experts), timed as the slowest rank's median launch. Fit: time ~ intercept per batch + scale x the
model time of ``decode_time_model`` at scale 1; per batch it prints the RMSE and the share of pairs
differing by over 3 standard errors that the model orders alike. A new GPU first gets a datasheet
entry with scale 1. GPU: ``PYTHONPATH=tests/microbenchmarks mpirun -np <EP> python3 <file> --out D``;
``--dry-run EP`` (CPU) fits the model's own times and must recover ``SYNTHETIC_SCALE``.
"""

import argparse
import json
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


def model_terms(xs, sizes, expert_bytes, device):
    """Unit costs, ``[N, ep, 4]`` unit-cost terms and ``(key, entry)`` of the device's model."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import TimeModel
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl as Mega

    table = Mega._DECODE_DEVICES
    costs = Mega.decode_time_model(
        SimpleNamespace(_DECODE_DEVICES=table, **sizes), expert_bytes, device
    )
    flat = xs.reshape(-1, xs.shape[-1]).astype(float)
    units = [TimeModel(*w, scale=1.0, tile_rows=costs["tile_rows"]) for w in np.eye(4)]
    terms = np.stack([m.layer_time_us(flat, 1) for m in units], -1).reshape(*xs.shape[:2], 4)
    key = next(key for key in table if key in device)
    return costs, terms, (key, table[key])


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


def time_designs(g, m, paths, exact, rank):
    """GPU: ``[REPS, N]`` us of the designs routed exactly, by bench_moe's MegaMoE CuteDSL layer."""
    import bench_moe as bm
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner

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
    return y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-batches", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--dry-run", type=int, metavar="EP", help="CPU, no launches: the model's times")
    a = ap.parse_args()
    import bench_moe as bm
    import torch

    from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size

    m = bm.BUILT_IN_MODELS["deepseek_v4_pro"]
    ep, rank = (a.dry_run, 0) if a.dry_run else (mpi_world_size(), mpi_rank())
    if not a.dry_run:
        bm._set_device_from_local_rank()
    device = "GB300" if a.dry_run else torch.cuda.get_device_name()
    sizes = dict(hidden_size=m.hidden_size, intermediate_size_per_partition=m.intermediate_size)
    sizes.update(expand_intermediate_size_per_partition=2 * m.intermediate_size)  # gated FC1
    expert_bytes = m.hidden_size * 3 * m.intermediate_size * 9 // 16  # NVFP4, an FP8 scale per 16
    costs, _, (key, entry) = model_terms(np.zeros((1, 1, 1)), sizes, expert_bytes, device)
    batches = []
    for lb in a.local_batches:  # every rank draws the same designs
        # T: 4 tokens per request per forward (MTP-3)
        g = SimpleNamespace(ep=ep, M=SLOTS, T=4 * lb, K=m.top_k, tile=costs["tile_rows"])
        xs = make_designs(g, DESIGNS, np.random.default_rng([ep, lb]))
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
        terms = model_terms(xs[exact], sizes, expert_bytes, device)[1]
        p = (terms @ [costs[k] for k in KEYS]).max(1)
        if a.dry_run:
            y = np.tile(lb + SYNTHETIC_SCALE * p, (REPS, 1))
        else:
            y = time_designs(g, m, paths, exact, rank)[:, exact]
        batches.append((lb, p, y.mean(0), np.sqrt(np.var(y, 0, ddof=1).mean() / REPS), len(xs)))
    k = sum((p - p.mean()) @ y for _, p, y, _, _ in batches)
    k /= sum((p - p.mean()) @ p for _, p, _, _, _ in batches)
    lines = [
        f"scale {k:.4f} (shipped {costs['scale']})",
        f'entry: "{key}": {entry[:2] + (round(float(k), 3),)},',
    ]
    for lb, p, y, se, n in batches:
        i, j = np.triu_indices(len(y), 1)
        big = np.abs(y[i] - y[j]) > 3 * np.sqrt(2) * se
        conc = np.mean(np.sign(y[i] - y[j]) == np.sign(p[i] - p[j]), where=big)
        lines.append(
            f"lb {lb}: {len(y)} of {n} designs routed exactly, rmse {np.std(y - k * p):.2f} us, "
            f"concordance {conc:.3f} ({big.sum()} pairs)"
        )
    if rank == 0:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
