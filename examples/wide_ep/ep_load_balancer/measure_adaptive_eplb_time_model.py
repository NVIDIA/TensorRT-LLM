# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate adaptive EPLB's MegaMoE CuteDSL decode time model per CUDA graph bucket, offline.

The serving start-up does the same by default (``MoeAdaptiveLoadBalancerConfig.time_model``
``auto``); this tool writes the calibration of a deployment without the model, for
``time_model: file``. It builds bench_moe's NVFP4 MegaMoE CuteDSL layer of a built-in model
(``--model``, the geometry: hidden and intermediate sizes, experts, top-k) with the deployment's
slots (slots as experts, the deployment's MegaMoE buckets), and runs the package's calibration on it
(``tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.calibrate``): per bucket (``local batch x (1 +
draft tokens)`` tokens per GPU) random slot-row designs as forced routes, the read cost, the scale
and its hold-out check, and the route op's cost of a replica. GPU: ``PYTHONPATH=tests/microbenchmarks
mpirun -np <EP> python3 <file> --out D --num-slots S [--model M] [--local-batches B ..]
[--rates BW PEAK]``
writes ``D/calibration.json`` (``time_model_file``) and every design's times in ``D/times.json``;
``--dry-run EP --rates BW PEAK`` (CPU) fits the model's own times plus noise without a launch.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SYNTHETIC_SCALE = 0.9  # the dry run's times over the model's


class SyntheticTimer:
    """The dry run's times: the model's at ``unit`` costs times SYNTHETIC_SCALE, plus noise."""

    def __init__(self, unit, tile_rows, reps):
        self.unit, self.tile_rows, self.reps = unit, tile_rows, reps
        self.rng = np.random.default_rng(0)

    def designs(self, tokens, top_k, xs):
        from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import calibrate

        y = tokens + SYNTHETIC_SCALE * (calibrate.model_terms(xs, self.tile_rows) @ self.unit).max(
            1
        )
        return y + self.rng.normal(0.0, 0.5, (self.reps, len(y)))

    def fixed(self, tokens, top_k, xs):
        return self.rng.normal(0.0, 0.5, (self.reps, len(xs), 3)).cumsum(-1)


def build_layer(m, ep, slots, max_tokens, rank):
    """bench_moe's MegaMoE CuteDSL layer of ``ep * slots`` slots as experts.

    Its MegaMoE buckets are those of ``max_tokens`` tokens per GPU, the deployment's.
    """
    import bench_moe as bm
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner

    bf16, cuda = torch.bfloat16, torch.device("cuda")
    spec = (m.hidden_size, m.intermediate_size, "NVFP4", "RENORMALIZE")
    model = bm.ModelSpec("custom", ep * slots, m.top_k, *spec)
    config = bm.ConfigSpec(backend="MEGAMOE_CUTEDSL", parallel_mode="DEP", cuda_graph=True)
    mapping = bm._build_mapping_from_config(config, ep)
    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()
    flags = dict(
        use_cuda_graph=True, use_low_precision_moe_combine=False, enable_perfect_router=False
    )
    kinds = dict(dtype=bf16, routing_logits_dtype=bf16, device=cuda, max_num_tokens=max_tokens)
    moe = bm._build_moe_module(
        model=model, config=config, mapping=mapping, moe_backend=config.backend, **flags, **kinds
    )[0]
    return moe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--draft-tokens", type=int, default=3, help="per request per forward (MTP-3)")
    ap.add_argument(
        "--max-num-tokens", type=int, default=2048, help="the deployment's: its MegaMoE buckets"
    )
    ap.add_argument(
        "--rates", type=float, nargs=2, help="HBM bytes/s and dense FP4 FLOP/s: a prior read cost"
    )
    ap.add_argument("--dry-run", type=int, metavar="EP", help="CPU, no launches: the model's times")
    ap.add_argument(
        "--num-slots", type=int, required=True, help="the deployment's moe_load_balancer num_slots"
    )
    ap.add_argument("--tile-rows", type=int, default=256, help="the dry run's MMA tile M")
    import bench_moe as bm

    ap.add_argument("--model", default="deepseek_v4_pro", choices=sorted(bm.BUILT_IN_MODELS))
    a = ap.parse_args()

    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer as ab
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import calibrate
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl as Mega
    from tensorrt_llm._utils import mpi_allgather, mpi_rank, mpi_world_size
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

    m = bm.BUILT_IN_MODELS[a.model]
    ep, rank = (a.dry_run, 0) if a.dry_run else (mpi_world_size(), mpi_rank())
    if a.num_slots % ep or a.num_slots < m.num_experts:
        raise SystemExit(
            f"--num-slots {a.num_slots}: not at least the experts and a multiple of {ep}"
        )
    settings = calibrate.Settings.of(MoeAdaptiveLoadBalancerConfig(num_slots=a.num_slots))
    tokens = [lb * (1 + a.draft_tokens) for lb in a.local_batches]
    # One expert's NVFP4 weights (gated FC1 and FC2, an FP8 scale per 16); the layer's planes below.
    expert_bytes = m.hidden_size * 3 * m.intermediate_size * 9 // 16
    sizes = dict(hidden_size=m.hidden_size, intermediate_size_per_partition=m.intermediate_size)
    sizes.update(expand_intermediate_size_per_partition=2 * m.intermediate_size)  # gated FC1
    if a.dry_run:
        if not a.rates:
            raise SystemExit("--dry-run prices its synthetic times at --rates")
        backend = SimpleNamespace(**sizes)
        backend.decode_time_model = lambda *args, **kw: Mega.decode_time_model(backend, *args, **kw)
        unit = backend.decode_time_model(expert_bytes, a.rates, tile_rows=a.tile_rows)
        unit = np.array([unit[k] for k in calibrate.KEYS])
        timer, gather = SyntheticTimer(unit, a.tile_rows, settings.reps), (lambda v: [v])
        identity = dict(device="dry-run", ep_size=ep, hosts=1, num_slots=a.num_slots, top_k=m.top_k)
        identity.update(tactics={str(t): "dry-run" for t in tokens})
        identity.update(tile_rows={str(t): a.tile_rows for t in tokens})
    else:
        bm._set_device_from_local_rank()
        timer, gather = None, mpi_allgather
        why, hosts = ab.nvlink_domain_error(SimpleNamespace(allgather=gather))
        if why:
            raise SystemExit(f"nothing launched: {why}")
        moe = build_layer(m, ep, a.num_slots // ep, a.max_num_tokens, rank)
        backend = getattr(moe, "backend", moe)
        planes = [backend.get_parameter(name) for name, _ in backend.slot_weight_planes()]
        expert_bytes = sum(p.stride(0) * p.element_size() for p in planes)  # a slot's bytes
        identity = ab.calibration_identity(backend, ep, hosts, tokens, a.num_slots)
    text, why, times = calibrate.calibrate(
        backend,
        tokens,
        identity=identity,
        expert_bytes=expert_bytes,
        num_experts=m.num_experts,
        num_slots=a.num_slots,
        rank=rank,
        gather=gather,
        settings=settings,
        timer=timer,
        rates=tuple(a.rates) if a.rates else None,
    )
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "times.json").write_text(json.dumps(times))
        if why:
            print(f"refused, no calibration written: {why}")
        else:
            doc = json.loads(text)
            doc["measurement"]["draft_tokens"] = a.draft_tokens
            (a.out / "calibration.json.tmp").write_text(json.dumps(doc, indent=1, sort_keys=True))
            os.replace(a.out / "calibration.json.tmp", a.out / "calibration.json")
            for bucket in doc["buckets"]:
                print(json.dumps(bucket))
            print(f"written: {a.out / 'calibration.json'}")
    return 1 if why else 0


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
