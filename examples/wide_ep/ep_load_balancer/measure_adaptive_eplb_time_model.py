# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate adaptive EPLB's MegaMoE CuteDSL decode time model per CUDA graph bucket.

Random slot-row designs per bucket (``local batch x (1 + draft tokens)`` tokens per GPU) run as
bench_moe forced routing patterns (slots as experts), timed as the slowest rank's median launch.
Per bucket, ``per_active_slot_us`` is the read cost of a slot with rows that the designs show
(:func:`read_cost`) and scale the slope of the times on ``decode_time_model``'s times at scale 1 at
that cost (an intercept per bucket); the bucket is accepted when the scale of every fourth design,
held out, agrees with the others' within 3 standard errors (bootstrap over the designs).
``route_us`` is what the route op adds to a layer once its route table holds a replica
(:func:`time_fixed`). Every check that can fail runs before the bucket's first launch and ends in a
vote of all ranks; rank 0 writes once every rank computed the same file, and a failure after a
launch aborts every rank. GPU: ``PYTHONPATH=tests/microbenchmarks mpirun -np <EP> python3 <file>
--out D [--local-batches B ..] [--rates BW PEAK]`` writes ``D/calibration.json``
(``MoeAdaptiveLoadBalancerConfig.time_model_file``), every design's times in ``D/times.json`` and
the fixed costs of the balancer's ops in ``D/fixed.json``; ``--dry-run EP`` (CPU) fits the model's
own times plus noise and builds the route tables without a launch.
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
# Per local batch, designs whose rows sit on the home slots alone: with REPS passes, 24 paired
# differences, a standard error of about 0.15 us at the 0.5 us noise of one median launch.
FIXED_DESIGNS = 8
# What time_fixed times per home-only design, in us: the layer of SLOTS slots alone, then with the
# route op on the home table, on a table with replicas, the statistics op alone, the staging op, and
# the layer of the home slots alone (the deployment without adaptive EPLB).
FIXED_PARTS = ("layer", "route_home", "route_replicas", "record", "stage", "layer_home_slots")


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


def fit_read(terms, costs, y):
    """The read cost ``theta >= 0`` of a slot with rows that fits the designs best.

    The fit is ``y ~ a + k * max_g(theta * slots_g + rest_g)``, the other unit costs held: a linear
    fit of ``a``, ``k * theta`` and ``k`` on every design's busiest GPU, repeated until the busiest
    GPUs repeat.
    """
    slots, rest = terms[..., 0], terms[..., 1:] @ costs[1:]
    theta, seen, rows = costs[0], set(), np.arange(len(y))
    while True:
        busiest = np.argmax(theta * slots + rest, 1)
        if busiest.tobytes() in seen:
            return theta
        seen.add(busiest.tobytes())
        a = np.stack([np.ones(len(y)), slots[rows, busiest], rest[rows, busiest]], 1)
        _, k_theta, k = np.linalg.lstsq(a, y, rcond=None)[0]
        if not k > 0.0:  # no time grows with the rows: the designs show no read cost
            return costs[0]
        theta = max(k_theta, 0.0) / k


def read_cost(terms, costs, y):
    """``(theta, held-out RMSE at theta, at the model's own read cost)`` of a bucket's designs.

    ``theta`` is fitted on all designs when, fitted on three in four, it predicts the fourth better
    than the model's own read cost (``costs[0]``) does at a scale fitted alike; else it is the
    model's own.
    """
    held = np.arange(len(y)) % 4 == 3

    def held_rmse(theta):
        p = (terms @ np.r_[theta, costs[1:]]).max(1)
        k, a = np.polyfit(p[~held], y[~held], 1)
        return float(np.sqrt(np.mean((y[held] - a - k * p[held]) ** 2)))

    fitted, own = held_rmse(fit_read(terms[~held], costs, y[~held])), held_rmse(costs[0])
    return (fit_read(terms, costs, y) if fitted < own else costs[0]), fitted, own


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


def build_layer(g, m, rank, max_tokens=None):
    """bench_moe's MegaMoE CuteDSL layer of ``g.ep * g.M`` slots as experts and its inputs.

    The layer's MegaMoE buckets are those of ``max_tokens`` tokens per GPU (default: the bucket's own).
    """
    import bench_moe as bm
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner

    bf16, cuda = torch.bfloat16, torch.device("cuda")
    spec = (m.hidden_size, m.intermediate_size, "NVFP4", "RENORMALIZE")
    model = bm.ModelSpec("custom", g.ep * g.M, g.K, *spec)
    config = bm.ConfigSpec(backend="MEGAMOE_CUTEDSL", parallel_mode="DEP", cuda_graph=True)
    mapping = bm._build_mapping_from_config(config, g.ep)
    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()
    flags = dict(
        use_cuda_graph=True, use_low_precision_moe_combine=False, enable_perfect_router=False
    )
    kinds = dict(
        dtype=bf16, routing_logits_dtype=bf16, device=cuda, max_num_tokens=max_tokens or g.T
    )
    moe = bm._build_moe_module(
        model=model, config=config, mapping=mapping, moe_backend=config.backend, **flags, **kinds
    )[0]
    x, logits = bm._make_inputs(g.T, m.hidden_size, g.ep * g.M, bf16, bf16, cuda, seed=rank)
    return moe, (x, logits, [g.T] * g.ep)


def time_launch(moe, inputs, ids_scales, extra=None, tune=False):
    """Microseconds of one forward of ``moe`` on forced routes, the slowest rank's median launch.

    ``extra()`` launches where a load balancer routes, once the MoE scheduler named the bucket.
    """
    import bench_moe as bm

    backend = getattr(moe, "backend", moe)
    hint = backend.set_adaptive_launch_tokens
    if extra is not None:
        backend.set_adaptive_launch_tokens = lambda tokens: (hint(tokens), extra())
    try:
        with bm._maybe_install_routing_control_patch(moe, *ids_scales, active=True):
            if tune:
                bm._run_autotune(moe, *inputs, False)
            t = bm._time_moe_forward_cuda_graph(moe, *inputs, warmup=WARMUP, iters=ITERS)
    finally:
        vars(backend).pop("set_adaptive_launch_tokens", None)
    return 1e3 * np.median(bm._slowest_rank_iter_maxes(bm._gather_per_iteration_times(t[0])))


def time_designs(g, m, paths, exact, rank, gather, hosts):
    """GPU: ``[REPS, N]`` us of the designs routed exactly and the voted calibration identity.

    Every rank votes for the calibration identity before the first launch.
    """
    import torch

    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer as ab

    moe, inputs = build_layer(g, m, rank)
    identity = ab.calibration_identity(getattr(moe, "backend", moe), g.ep, hosts, [g.T])
    if len({json.dumps(v, sort_keys=True) for v in gather(identity)}) != 1:
        raise SystemExit(f"bucket {g.T}: the ranks' identities differ; nothing launched")
    y = np.full((REPS, len(paths)), np.nan)
    for rep in range(REPS):  # the same order on every rank: the kernel is collective
        for d in np.random.default_rng([rep, g.T]).permutation(np.flatnonzero(exact)):
            ids_scales = routes(paths[d], g, rank, torch.device("cuda"))[0]
            y[rep, d] = time_launch(moe, inputs, ids_scales, tune=np.isnan(y).all())
    moe.destroy()
    return y, identity


def route_tables(x, homes):
    """``(experts, home table, table with replicas)`` of home-only design ``x`` ``[ep, SLOTS]``.

    ``experts`` is the expert of every home slot (-1 elsewhere); the tables are the packed route
    tables of the home layout and of the layout planned on ``x``'s rows.
    """
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import policy

    ep, slots = x.shape
    geometry = policy.ExpertGeometry(ep * homes, ep * slots, ep)
    home = geometry.home_layout()
    load = x[:, :homes].reshape(-1).astype(float)
    plan = policy.plan_layout(load, geometry, home, home, policy.active_slots(geometry))
    tables = [
        policy.build_route_tables(t, None, ep * homes, ep * slots) for t in (home, plan.layout)
    ]
    experts = np.full(ep * slots, -1)
    experts[home >= 0] = home[home >= 0]
    return experts, *(np.array(t.packed(), dtype=np.int32) for t in tables)


def balancer_launches(backend, rank, ids_scales, experts, tables, stats):
    """The launches :func:`time_fixed` adds to a layer's forward, in FIXED_PARTS order.

    None, the route op on each of ``tables`` writing into the layer's route staging as a load
    balancer does, the statistics op, and the staging op of the layer without a load balancer.
    """
    import torch

    from tensorrt_llm._torch.moe.live_rows import current_word

    ids, scales = ids_scales[0].int(), ids_scales[1].float()
    ops, word = torch.ops.trtllm, current_word(experts.device)

    def route(table):
        staging = backend.route_staging(*experts.shape)
        if staging is None:
            out, weights = torch.empty_like(experts, dtype=backend.slot_id_dtype), (None, None)
        else:
            out, weights = staging.topk_idx, (scales, staging.topk_weights)
        ops.adaptive_eplb_route_into(experts, *table, rank, out, *weights, *stats, True, word)

    def stage():  # a launch that reads its inputs in place stages nothing
        staging = backend.route_staging(*ids.shape)
        if staging is not None:
            weights = staging.topk_weights[:, : ids.shape[1]]
            ops.megamoe_stage_routes(ids, scales, staging.topk_idx, weights)

    record = (lambda: ops.adaptive_eplb_record(experts, *stats, True, word),)
    return (None, *(lambda t=t: route(t) for t in tables), *record, stage)


def time_fixed(g, m, rank, max_tokens, xs, paths, own_paths):
    """GPU: ``[REPS, D, FIXED_PARTS]`` us of home-only designs ``xs`` ``[D, ep, SLOTS]``.

    Both layers (SLOTS slots, then the home slots alone on ``own_paths``) take the MegaMoE buckets of
    the deployment's ``max_tokens``, so the route op and the staging op see the deployment's staging.
    """
    import torch

    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import statistics_buffers

    cuda, homes = torch.device("cuda"), m.num_experts // g.ep
    moe, inputs = build_layer(g, m, rank, max_tokens)
    out = np.full((REPS, len(xs), len(FIXED_PARTS)), np.nan)
    buffers = statistics_buffers(1, m.num_experts, cuda)
    enabled = torch.ones(1, dtype=torch.int32, device=cuda)
    stats = (buffers["rows"][0], buffers["forwards"][0], enabled)
    split = [m.num_experts, m.num_experts, g.ep * g.M, g.ep * g.M]
    for d, x in enumerate(xs):
        ids_scales = routes(paths[d], g, rank, cuda)[0]
        experts, *tables = (torch.from_numpy(t).to(cuda) for t in route_tables(x, homes))
        experts = experts[ids_scales[0].long()].int()
        tables = [t.split(split) for t in tables]
        backend = getattr(moe, "backend", moe)
        launches = balancer_launches(backend, rank, ids_scales, experts, tables, stats)
        for rep in range(REPS):
            for i, extra in enumerate(launches):
                out[rep, d, i] = time_launch(
                    moe, inputs, ids_scales, extra, tune=rep == d == i == 0
                )
    moe.destroy()
    home_only = SimpleNamespace(**dict(vars(g), M=homes))
    moe, inputs = build_layer(home_only, m, rank, max_tokens)
    for rep in range(REPS):
        for d, path in enumerate(own_paths):
            ids_scales = routes(path, home_only, rank, cuda)[0]
            out[rep, d, -1] = time_launch(moe, inputs, ids_scales, tune=rep == d == 0)
    moe.destroy()
    return out


def time_on_deployment(g, m, rank, max_tokens, paths):
    """GPU: ``[REPS, len(paths)]`` us of designs on a layer of the deployment's MegaMoE buckets."""
    import torch

    moe, inputs = build_layer(g, m, rank, max_tokens)
    y = np.full((REPS, len(paths)), np.nan)
    for rep in range(REPS):  # the same order on every rank: the kernel is collective
        for d in np.random.default_rng([rep, g.T, 2]).permutation(len(paths)):
            ids_scales = routes(paths[d], g, rank, torch.device("cuda"))[0]
            y[rep, d] = time_launch(moe, inputs, ids_scales, tune=np.isnan(y).all())
    moe.destroy()
    return y


def deployment_check(terms, costs, y_bucket, y_deploy):
    """The read cost and scale the same designs give on the bucket's own layer and on the deployment's.

    ``y_bucket`` and ``y_deploy`` are their mean times (us); the paired difference is reported with them.
    """
    out = {}
    for name, y in (("bucket", y_bucket), ("deployment", y_deploy)):
        theta = fit_read(terms, costs, y)
        out[name] = dict(per_active_slot_us=round(float(theta), 4))
        p = (terms @ np.r_[theta, costs[1:]]).max(1)
        out[name].update(scale=round(float(np.polyfit(p, y, 1)[0]), 4))
    diff = y_deploy - y_bucket
    sd = diff.std(ddof=1) / np.sqrt(len(diff))
    return dict(
        out, designs=len(diff), mean_diff_us=[round(float(diff.mean()), 3), round(float(sd), 3)]
    )


def fixed_costs(t):
    """Per-layer costs ``(mean, SE)`` in us of :func:`time_fixed`'s times ``t``.

    Means and standard errors are over the paired designs and passes.
    """
    c = {name: t[..., i] for i, name in enumerate(FIXED_PARTS)}
    parts = dict(
        route_us=c["route_replicas"] - c["route_home"],
        route_home_us=c["route_home"] - c["layer"],
        record_us=c["record"] - c["layer"],
        stage_us=c["stage"] - c["layer"],
        extra_slots_us=c["layer"] - c["layer_home_slots"],
    )
    # On, every layer at home, over the layer without adaptive EPLB; and the part of it a launch
    # without the route op on a layer of the home slots alone would save (it still records).
    parts["on_at_home_us"] = parts["route_home_us"] - parts["stage_us"] + parts["extra_slots_us"]
    parts["saved_without_route_us"] = parts["on_at_home_us"] - parts["record_us"]
    n = t[..., 0].size
    return {k: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))) for k, v in parts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--draft-tokens", type=int, default=3, help="per request per forward (MTP-3)")
    ap.add_argument(
        "--max-num-tokens", type=int, default=2048, help="the deployment's: its MegaMoE buckets"
    )
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
    unit = np.array([costs[k] for k in KEYS])
    buckets, times, identity, tactics, fixed = [], {}, {}, {}, {}
    for lb in a.local_batches:  # every rank draws the same designs
        T = lb * (1 + a.draft_tokens)
        g = SimpleNamespace(ep=ep, M=SLOTS, T=T, K=m.top_k, tile=costs["tile_rows"])
        rng = np.random.default_rng([ep, lb])
        xs = make_designs(g, DESIGNS, rng)
        homes = SimpleNamespace(**dict(vars(g), M=m.num_experts // ep))
        own = make_designs(homes, FIXED_DESIGNS, rng)
        padded = np.concatenate([own, np.zeros((*own.shape[:2], SLOTS - homes.M), int)], 2)
        paths = [a.out / f"lb{lb}" / f"{d:03d}.json" for d in range(len(xs))]
        home_paths = [a.out / f"lb{lb}" / f"home{d:02d}.json" for d in range(len(own))]
        own_paths = [a.out / f"lb{lb}" / f"own{d:02d}.json" for d in range(len(own))]
        for path, x in (
            zip(paths + home_paths + own_paths, [*xs, *padded, *own]) if rank == 0 else ()
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pattern(x)))
        if not a.dry_run:
            mpi_barrier()
        sources, on = (range(ep), "cpu") if a.dry_run else ([rank], "cuda")
        counts = np.stack([sum(routes(path, g, src, on)[1] for src in sources) for path in paths])
        counts = counts if a.dry_run else np.sum(mpi_allgather(counts), 0)
        exact = (counts[:, :-1] == xs.reshape(len(xs), -1)).all(1) & (counts[:, -1] == 0)
        terms = model_terms(xs[exact], sizes, expert_bytes, device, rates)[1]
        if a.dry_run:
            y = lb + SYNTHETIC_SCALE * (terms @ unit).max(1)
            y = y + rng.normal(0.0, 0.5, (REPS, len(y)))
            identity = dict(device=device, ep_size=ep, hosts=1, tactics={str(T): "dry-run"})
            for x in padded:  # the tables and experts the GPU run launches with, checked
                experts, home, _ = route_tables(x, homes.M)
                slots = home[2 * m.num_experts :][: m.num_experts]
                assert (slots[experts[experts >= 0]] == np.flatnonzero(experts >= 0)).all()
            part = None
            deployed = y.mean(0)[3::4] + 1.0  # the control's designs, a microsecond slower
        else:
            y, identity = time_designs(g, m, paths, exact, rank, gather, hosts)
            y = y[:, exact]
            times_fixed = time_fixed(g, m, rank, a.max_num_tokens, padded, home_paths, own_paths)
            part = fixed_costs(times_fixed)
            deployed = None
            # The deployment pads this bucket's launches to a larger MegaMoE bucket: time a fourth of the
            # designs there too, to see what the bucket's own layer misses.
            if next(b for b in Mega._resolve_maxt_buckets(a.max_num_tokens) + [T] if T <= b) != T:
                on = [paths[d] for d in np.flatnonzero(exact)[3::4]]
                deployed = time_on_deployment(g, m, rank, a.max_num_tokens, on).mean(0)
        if deployed is not None:  # the designs of the control: every fourth of those routed exactly
            check = deployment_check(terms[3::4], unit, y.mean(0)[3::4], deployed)
            fixed.setdefault(str(T), {})["deployment_buckets"] = check
        tactics.update(identity["tactics"])
        theta, held_fit, held_own = read_cost(terms, unit, y.mean(0))
        p = (terms @ np.r_[theta, unit[1:]]).max(1)
        k, se, z = fit_bucket(p, y.mean(0), np.random.default_rng([ep, lb, 1]))
        noise = np.sqrt(np.var(y, 0, ddof=1).mean() / REPS)
        i, j = np.triu_indices(len(p), 1)
        big = np.abs(y.mean(0)[i] - y.mean(0)[j]) > 3 * np.sqrt(2) * noise
        conc = np.mean(np.sign(y.mean(0)[i] - y.mean(0)[j]) == np.sign(p[i] - p[j]), where=big)
        rmse = np.std(y.mean(0) - k * p)
        buckets.append(dict(tokens=T, scale=round(float(k), 4), se=round(float(se), 4)))
        buckets[-1].update(holdout_z=round(float(z), 2), designs=int(exact.sum()))
        buckets[-1].update(rmse_us=round(float(rmse), 2), concordance=round(float(conc), 3))
        buckets[-1].update(per_active_slot_us=round(float(theta), 4))
        buckets[-1].update(read_holdout_rmse_us=[round(held_fit, 2), round(held_own, 2)])
        if part is not None:  # a negative mean is a cost below the noise: none
            buckets[-1].update(route_us=round(max(part["route_us"][0], 0.0), 4))
            fixed.setdefault(str(T), {}).update(
                {k: [round(v, 4) for v in ms] for k, ms in part.items()}
            )
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
        (a.out / "fixed.json").write_text(json.dumps(fixed, indent=1, sort_keys=True))
        for lb, bucket in zip(a.local_batches, buckets):
            print(f"lb {lb}: {json.dumps(bucket)}")
            print(
                f"lb {lb} fixed (mean, SE) us per layer: {json.dumps(fixed.get(str(bucket['tokens'])))}"
            )
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
