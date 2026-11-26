# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration of the backend's decode time model on the GPUs it prices.

Per decode bucket (tokens per GPU per forward), random slot-row designs run as forced routes
through a live MoE layer. ``per_active_slot_us`` is the read cost of a slot with rows that the
designs show (:func:`read_cost`) and ``scale`` the slope of the times on the model's times at that
cost; the bucket holds when the scale of every fourth design, held out, agrees with the others'
within ``holdout_z`` standard errors (:func:`fit_bucket`). ``route_us`` is what the route op adds to
a layer once its route table holds a replica (:func:`fixed_costs`). Every check that can fail runs
before the first launch and ends in a vote; the ranks time in one order and fit the same times, so
they compute the same calibration, which they vote on before rank 0 caches it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pynvml
import torch

import tensorrt_llm
from tensorrt_llm._torch.autotuner import AutoTuner

from . import policy

CALIBRATION = ("tensorrt_llm.adaptive_eplb.decode_time_model", 1)
"""Schema and version of a calibration file of the decode time model."""
KEYS = ("per_active_slot_us", "per_extra_tile_us", "per_row_us", "hot_slot_extra_tile_us")
# What the fixed costs time per home-only design: the layer alone, then with the route op on the
# home table and on a table with replicas.
FIXED_PARTS = ("layer", "route_home", "route_replicas")
DIGEST_WORDS = 1 << 22
"""int: 32-bit words of a tensor :func:`snapshot` digests at a time: one chunk at a time is widened
to int64 (32 MiB, and as much for its square); a tensor of at most 16 MiB is copied whole instead."""


@dataclass(frozen=True)
class Settings:
    """The calibration's ``MoeAdaptiveLoadBalancerConfig.calibration_*`` fields."""

    designs: int
    fixed_designs: int
    reps: int
    iters: int
    warmup: int
    holdout_z: float

    @classmethod
    def of(cls, config: object) -> Settings:
        return cls(**{f.name: getattr(config, f"calibration_{f.name}") for f in fields(cls)})


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cache_path(config: object, identity: dict) -> Path | str:
    """The cache file of a run of ``identity``: ``calibration_cache_dir``, else ``adaptive_eplb``
    under ``TRTLLM_CACHE_DIR``; or why the run has none: neither is set, or NVML cannot tell the
    driver, without which the key could name another driver's calibration."""
    root = config.calibration_cache_dir or os.environ.get("TRTLLM_CACHE_DIR")
    if not root:
        return "no cache directory (calibration_cache_dir or TRTLLM_CACHE_DIR)"
    try:
        versions = platform_versions()
    except Exception as error:  # noqa: BLE001 - uncached, the run still calibrates
        return f"the GPU driver version is unreadable ({type(error).__name__}: {error})"
    key = dict(identity, settings=asdict(Settings.of(config)), schema=CALIBRATION)
    wheels = importlib.metadata.distributions(name="tensorrt_llm")  # the build: its source commit
    build = [u for w in wheels for u in w.metadata.get_all("Project-URL") or [] if "/commit/" in u]
    lib = Path(tensorrt_llm.__file__).parent / "libs" / "libth_common.so"  # and its kernels' bytes,
    build.append(file_sha256(lib) if lib.is_file() else None)  # also when replaced under one wheel
    fit = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # this measurement and its fit
    key.update(trtllm_version=tensorrt_llm.__version__, build=build, fit=fit, **versions)
    folder = Path(os.path.expanduser(root))
    folder = folder if config.calibration_cache_dir else folder / "adaptive_eplb"
    return folder / f"{digest(key)}.json"


def file_sha256(path: Path) -> str:
    """The sha256 of the file at ``path``, read 1 MiB at a time (0.2 s for a 234 MB library)."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def platform_versions() -> dict:
    """What changes the kernels' times or the designs outside the run's identity: the GPU driver and
    its CUDA version (NVML), the CUDA runtime, numpy (its generator draws the designs and keeps no
    stream across versions)."""
    pynvml.nvmlInit()
    try:
        driver, cuda = pynvml.nvmlSystemGetDriverVersion(), pynvml.nvmlSystemGetCudaDriverVersion()
    finally:
        pynvml.nvmlShutdown()
    driver = driver.decode() if isinstance(driver, bytes) else driver
    return dict(driver=driver, cuda=cuda, cuda_runtime=torch.version.cuda, numpy=np.__version__)


def read_cache(path: Path | str) -> str | None:
    try:
        return path.read_text() if isinstance(path, Path) else None
    except OSError:
        return None


def write_cache(path: Path | str, text: str) -> str | None:
    """Write ``text`` at ``path``; why it stays in memory only, if it does."""
    if isinstance(path, str):  # why the run has no cache file
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        part.write_text(text)
        os.replace(part, path)
    except OSError as error:
        return f"{path} is not writable ({type(error).__name__}: {error})"
    return None


def evict(written: Path, keep: int) -> list[str]:
    """Delete all but the ``keep`` newest calibrations beside ``written`` (by write time; never
    ``written``), each with its ``.times.json`` first; 0 keeps all. A ``.times.json`` without its
    calibration (no bucket held) is an entry of its own. Only ``<sha256>.json`` files of this schema
    and times of its layout count, so a shared directory keeps everything else; a file another
    process deleted first is no failure. Why a file stayed, never an exception: it loses only space."""
    if keep <= 0:
        return []
    entries, failed = [], []
    try:
        for path in written.parent.glob("*.json"):
            name = re.fullmatch(r"([0-9a-f]{64})(\.times)?\.json", path.name)
            times = bool(name and name[2])  # alone: one beside its calibration goes with it
            if not name or path == written or times and path.with_name(f"{name[1]}.json").exists():
                continue
            try:
                doc = json.loads(path.read_text())
                rows = doc.values() if times and isinstance(doc, dict) else [doc]
                if all(
                    isinstance(r, dict)
                    and ("us" in r if times else r.get("schema") == CALIBRATION[0])
                    for r in rows
                ):
                    entries.append((path.stat().st_mtime_ns, path.name, path, not times))
            except (OSError, ValueError):  # gone, or not ours
                continue
    except OSError as error:
        return [f"{written.parent} ({type(error).__name__}: {error})"]
    for *_, path, whole in sorted(entries, reverse=True)[keep - 1 :]:
        for part in (path.with_suffix(".times.json"), path) if whole else (path,):
            try:
                part.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                failed.append(f"{part} ({type(error).__name__}: {error})")
    return failed


def make_designs(g: SimpleNamespace, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` designs ``[ep, slots]``: random active slots, rows per GPU, concentration, hot slots."""
    total, cap, out = g.ep * g.T * g.K, g.ep * g.T, []  # a slot takes a token at most once
    while len(out) < n:
        hot = np.zeros((g.ep, g.M))
        for r in rng.choice(g.ep, rng.integers(0, g.ep // 4 + 2), replace=False):
            hot[r, 0] = min(cap, g.tile * rng.integers(1, cap // g.tile + 2)) - rng.integers(64)
        hot = np.maximum(hot, 0)  # a bucket of fewer rows per GPU than the jitter
        w = rng.lognormal(0, rng.uniform(0, 1.5), hot.shape) * (hot == 0)
        w *= (np.arange(g.M) < rng.integers(1, g.M + 1, (g.ep, 1))) * rng.lognormal(0, 0.7)
        rest = int(total - hot.sum())
        x = hot + rng.multinomial(max(rest, 0), (w / w.sum()).ravel()).reshape(w.shape)
        if rest >= 0 and x.max() <= cap:
            out.append(x.astype(np.int64))
    return np.stack(out)


def forced_routes(x: np.ndarray, rank: int, top_k: int) -> np.ndarray:
    """Rank ``rank``'s ``[tokens, top_k]`` global slot ids that route design ``x`` ``[ep, slots]``.

    The rows of all slots in slot order are dealt to the ranks in turn, so a rank sends every GPU
    about 1 / ep of its rows and takes at most ``ceil(rows / ep) <= tokens`` rows of a slot; written
    column by column, a token's ``top_k`` slots are distinct.
    """
    rows = np.repeat(np.arange(x.size), x.ravel())[rank :: len(x)]
    return np.ascontiguousarray(rows.reshape(top_k, -1).T)


def model_terms(xs: np.ndarray, tile_rows: int) -> np.ndarray:
    """``[N, ep, 4]`` unit-cost terms of designs ``xs`` ``[N, ep, slots]``, in :data:`KEYS` order."""
    units = [policy.TimeModel(*w, scale=1.0, tile_rows=tile_rows) for w in np.eye(4)]
    flat = xs.reshape(-1, xs.shape[-1]).astype(float)
    return np.stack([m.layer_time_us(flat, 1) for m in units], -1).reshape(*xs.shape[:2], 4)


def fit_read(terms: np.ndarray, costs: np.ndarray, y: np.ndarray) -> float:
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
        if not k > 0.0:  # no row cost resolved: theta, a slot read in rows (a row costs 1 at the
            return costs[0] or float(rest.max())  # unit rates), at most the designs' largest rest
        theta = max(k_theta, 0.0) / k


def read_cost(terms: np.ndarray, costs: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``(theta, held-out RMSE at theta, at the prior read cost)`` of a bucket's designs.

    ``theta`` is fitted on all designs when, fitted on three in four, it predicts the fourth better
    than the prior read cost (``costs[0]``) does at a scale fitted alike; else it is the prior.
    """
    held = np.arange(len(y)) % 4 == 3

    def held_rmse(theta):
        p = (terms @ np.r_[theta, costs[1:]]).max(1)
        k, a = np.polyfit(p[~held], y[~held], 1)
        return float(np.sqrt(np.mean((y[held] - a - k * p[held]) ** 2)))

    fitted, own = held_rmse(fit_read(terms[~held], costs, y[~held])), held_rmse(costs[0])
    return (fit_read(terms, costs, y) if fitted < own else costs[0]), fitted, own


def fit_bucket(p: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int = 1000) -> tuple:
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


def route_tables(x: np.ndarray, homes: int) -> list[np.ndarray]:
    """``[home layout, packed home table, packed table with replicas]`` of home-only design ``x``
    ``[ep, slots]``; the replicas are those a plan on ``x``'s rows places."""
    ep, slots = x.shape
    geometry = policy.ExpertGeometry(ep * homes, ep * slots, ep)
    home = geometry.home_layout()
    load = x[:, :homes].reshape(-1).astype(float)
    plan = policy.plan_layout(load, geometry, home, home, policy.active_slots(geometry))
    tables = [
        policy.build_route_tables(t, None, ep * homes, ep * slots).packed()
        for t in (home, plan.layout)
    ]
    return [home, *(np.array(t, dtype=np.int32) for t in tables)]


def fixed_costs(t: np.ndarray) -> dict[str, tuple[float, float]]:
    """Per-layer costs ``(mean, SE)`` in us of the fixed parts' times ``t`` ``[reps, D, parts]``."""
    c = {name: t[..., i] for i, name in enumerate(FIXED_PARTS)}
    parts = dict(route_us=c["route_replicas"] - c["route_home"])
    parts.update(route_home_us=c["route_home"] - c["layer"])
    n = t[..., 0].size
    return {k: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))) for k, v in parts.items()}


class LiveTimer:
    """Times designs on a live layer (``backend.time_decode_steps``) with forced routes; each pass
    takes the slowest rank's time of every launch, so all ranks hold the same times."""

    def __init__(self, backend: object, rank: int, num_experts: int, settings: Settings, gather):
        self.backend, self.rank, self.num_experts = backend, rank, num_experts
        self.settings, self.gather, self.host_us = settings, gather, 0.0
        self.device = torch.device("cuda", torch.cuda.current_device())

    def refusal(self) -> str | None:
        """Why the layer cannot launch on forced routes now, if it cannot."""
        if AutoTuner.get().is_tuning_mode or torch.cuda.is_current_stream_capturing():
            return "the autotuner is tuning or a CUDA graph is being captured"
        return None

    def _ids(self, x: np.ndarray, top_k: int) -> torch.Tensor:
        ids = torch.from_numpy(forced_routes(x, self.rank, top_k))
        return ids.to(self.device, self.backend.slot_id_dtype)

    def _passes(self, tokens: int, steps: list[Callable[[], torch.Tensor]]) -> np.ndarray:
        s = self.settings
        y = np.empty((s.reps, len(steps)))
        for rep in range(s.reps):  # the same order on every rank: the kernel is collective
            order = np.random.default_rng([rep, tokens]).permutation(len(steps))
            us, host = self.backend.time_decode_steps(
                tokens, [steps[i] for i in order], warmup=s.warmup, iters=s.iters
            )
            y[rep, order] = np.max(np.array(self.gather(us)), 0)
            self.host_us = max(self.host_us, *self.gather(host))
        return y

    def designs(self, tokens: int, top_k: int, xs: np.ndarray) -> np.ndarray:
        """``[reps, N]`` us of designs ``xs``."""
        steps = [lambda ids=self._ids(x, top_k): ids for x in xs]
        return self._passes(tokens, steps)

    def fixed(self, tokens: int, top_k: int, xs: np.ndarray) -> np.ndarray:
        """``[reps, D, FIXED_PARTS]`` us of home-only designs ``xs``, the route op writing the
        layer's route staging as the balancer's route does, into statistics of its own."""
        homes, steps = self.num_experts // len(xs[0]), []
        rows = policy.statistics_buffers(1, self.num_experts, self.device)
        on = torch.ones(1, dtype=torch.int32, device=self.device)
        stats = (rows["rows"][0], rows["forwards"][0], on)
        weights = torch.full((tokens, top_k), 1.0 / top_k, device=self.device)
        for x in xs:
            ids = self._ids(x, top_k)
            home, *tables = (torch.from_numpy(t).to(self.device) for t in route_tables(x, homes))
            experts = home[ids.long()].int()
            steps.append(lambda ids=ids: ids)
            for table in tables:
                split = table.split([self.num_experts] * 2 + [home.numel()] * 2)
                steps.append(lambda t=split, e=experts: self._route(e, t, weights, stats))
        y = self._passes(tokens, steps)
        return y.reshape(len(y), len(xs), len(FIXED_PARTS))

    def _route(self, experts, table, weights, stats) -> torch.Tensor:
        staging = self.backend.route_staging(*experts.shape, experts.device)
        if staging is None:
            out = experts.new_empty(experts.shape, dtype=self.backend.slot_id_dtype)
            pair, word = (None, None), None
        else:
            out, pair = staging.topk_idx, (weights, staging.topk_weights)
            word = staging.live_row_word
        route = torch.ops.trtllm.adaptive_eplb_route_into
        route(experts, *table, self.rank, out, *pair, *stats, True, word)
        return out[: experts.shape[0]]


def calibrate(
    backend: object,
    tokens: list[int],
    *,
    identity: dict,
    expert_bytes: int,
    num_experts: int,
    num_slots: int,
    rank: int,
    gather: Callable[[object], list],
    settings: Settings,
    timer: object = None,
    rates: tuple[float, float] | None = None,
) -> tuple[str | None, str | None, dict]:
    """Calibrate ``backend``'s decode time model for every decode bucket. Collective over EP.

    Args:
        backend: A live MoE backend of the model (``AdaptiveEplbBackend``).
        tokens: Tokens per GPU per forward of every decode bucket.
        identity: What the calibration holds for (``balancer.calibration_identity``).
        expert_bytes: Bytes of one expert's weights.
        num_experts: Routed experts of a layer.
        num_slots: Expert slots of a layer over the EP group.
        rank: This rank's index in the EP group.
        gather: Every rank's value of its argument, by rank.
        settings: Designs, passes, launches and the hold-out bound.
        timer: What times the designs (default :class:`LiveTimer`).
        rates: Bytes/s and FLOP/s to price the model at, the read cost a prior at them; by default
            the rates of one microsecond per slot read and per row, without a prior read cost.

    Returns:
        ``(calibration file text, None)`` that every rank computed alike, or ``(None, why)``; and
        every bucket's design times and model times.
    """
    start, ep, top_k = time.perf_counter(), int(identity["ep_size"]), int(identity["top_k"])
    tokens = sorted({int(t) for t in tokens})
    tiles = {t: identity["tile_rows"].get(str(t)) for t in tokens}  # each bucket's tactic's
    unknown = [t for t, tile in tiles.items() if not tile]
    why = f"no launch resolved the tactic of buckets {unknown}" if unknown else None
    timer = timer or LiveTimer(backend, rank, num_experts, settings, gather)
    why = getattr(timer, "refusal", lambda: None)() or why
    prior = np.r_[float(rates is not None), 1, 1, 1]
    if rates is None:  # one microsecond per slot read and per row: the fit's units
        probe = backend.decode_time_model(expert_bytes, (1e6, 1e6), tile_rows=1)
        rates = (1e6 * probe["per_active_slot_us"], 1e6 * probe["per_row_us"])
    plans = {}
    for t in tokens if why is None else ():  # every rank draws the same designs
        g = SimpleNamespace(ep=ep, M=num_slots // ep, T=t, K=top_k, tile=tiles[t])
        rng = np.random.default_rng([ep, t])
        xs = make_designs(g, settings.designs, rng)
        homes = SimpleNamespace(**dict(vars(g), M=num_experts // ep))
        own = make_designs(homes, settings.fixed_designs, rng)
        plans[t] = xs, np.concatenate([own, np.zeros((*own.shape[:2], g.M - homes.M), int)], 2)
    votes = gather((why, digest(identity), digest({t: x.tolist() for t, (x, _) in plans.items()})))
    why = next((w for w, *_ in votes if w), None)
    if why is None and len({v[1:] for v in votes}) > 1:
        why = "the EP ranks resolve different decode kernels or designs"
    if why is not None:
        return None, why, {}
    buckets, times = [], {}
    for t, (xs, own) in plans.items():
        y = timer.designs(t, top_k, xs)
        route = fixed_costs(timer.fixed(t, top_k, own))["route_us"]
        terms = model_terms(xs, tiles[t])
        fields = backend.decode_time_model(expert_bytes, rates, tile_rows=tiles[t])
        unit = np.array([fields[k] for k in KEYS]) * prior
        theta, held_fit, held_own = read_cost(terms, unit, ym := np.median(y, 0))
        p = (terms @ np.r_[theta, unit[1:]]).max(1)
        k, se, z = fit_bucket(p, ym, np.random.default_rng([ep, t, 1]))
        rmse = np.std(ym - k * p)
        buckets.append(dict(tokens=t, scale=float(k), se=float(se), holdout_z=round(float(z), 2)))
        fine = abs(z) <= settings.holdout_z and k > 0 and np.isfinite([k, theta]).all()
        buckets[-1].update(held=bool(fine))  # else no trusted model there
        buckets[-1].update(designs=len(xs), rmse_us=round(float(rmse), 2), tile_rows=tiles[t])
        buckets[-1].update(median_us=round(float(np.median(y)), 2))
        buckets[-1].update(per_active_slot_us=float(theta), route_us=max(route[0], 0.0))
        buckets[-1].update(read_holdout_rmse_us=[round(held_fit, 2), round(held_own, 2)])
        times[str(t)] = dict(model_us=p.round(3).tolist(), us=y.round(3).tolist())
    measured = dict(asdict(settings), trtllm_version=tensorrt_llm.__version__, slots=num_slots)
    measured.update(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(gather(time.time())[0])))
    measured.update(seconds=round(max(gather(time.perf_counter() - start)), 1))
    measured.update(host_us_per_launch=round(float(getattr(timer, "host_us", 0.0)), 1))
    doc = dict(zip(("schema", "version"), CALIBRATION), identity=identity, buckets=buckets)
    doc.update(rates=dict(hbm_bytes_per_s=rates[0], fp4_dense_flops_per_s=rates[1]))
    text = json.dumps(dict(doc, measurement=measured), indent=1, sort_keys=True)
    text = gather(text)[0]  # rank 0's fit: every rank holds the same times, not always the same fit
    held = any(b["held"] for b in json.loads(text)["buckets"])
    why = None if held else f"no bucket's held-out designs agree within {settings.holdout_z} SE"
    return text, why, times


def snapshot(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per tensor, a copy, or for a large one the sums and sums of squares of its 32-bit words by
    :data:`DIGEST_WORDS`: what :func:`unchanged` compares."""
    out = []
    for tensor in tensors:
        flat = tensor.detach().reshape(-1)
        if flat.numel() * flat.element_size() <= 4 * DIGEST_WORDS:
            out.append(flat.clone())
            continue
        sums = []
        for chunk in flat.view(torch.int32).split(DIGEST_WORDS):  # one chunk widened at a time
            words = chunk.long()
            sums.append(torch.stack((words.sum(), words.square().sum())))
        out.append(torch.stack(sums))
    return out


def unchanged(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return len(before) == len(after) and all(map(torch.equal, before, after))
