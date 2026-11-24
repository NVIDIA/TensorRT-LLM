# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep (and dep16) context target's routed experts (``_moe_routed``: route, one-sided all-to-all, CuteDSL
FC1 / FC2 finalize, all-to-all back) under per-layer dynamic EPLB, on four SM107 GPUs of one
NVLink domain (EP4 over the Ulysses CP group), with the K3.1 expert shapes (896 experts, top-16,
NVFP4 3584 x 3072) and 8 helper slots per rank (928 slots: 224 resident experts + 8 helper slots).

Per call, against the same layer without a load balancer (the same weights, the same routing):

* the routed rows agree: with the BF16 combine to BF16 rounding (a pick a helper slot serves joins
  another rank's partial sum), with the FP8 combine within the wire's rounding (each rank's
  partial rounded to E4M3, half an ulp is 1/16 of its binade: 2 x 1/16 of the largest output, the
  band of ``test_rubin_moe_a2a.py``);
* every physical route is a slot of its expert: a resident slot its own expert, a helper slot of
  this rank a byte-exact copy of the routed expert's six live planes; the resident rows never
  change;
* on skewed routings the busiest rank's routed rows drop.

Over even rows, skewed routings, uneven rows and an idle rank (padded as ``_moe_dep_ctx`` pads).
``--bench`` (module run) also times both layers per case; ``--dep16`` runs dep16's
``_moe_routed`` (the same function over its own module) instead of cep's; ``--bf16`` combines in
BF16 instead of FP8; ``--every k`` replans every k-th call (per-layer-resident helpers)
and reuses the placement between (``--resident``: per-layer-resident helpers at k = 1), each case
then running a replanning and a reusing call.

Requirements: four SM107 GPUs with fabric (MNNVL) memory and the dynamic EPLB kernels in the
build; ``CUDA_SCALE_LAUNCH_QUEUES=4x`` in the workers' environment. Under srun the MPI pool hangs:
run the module itself on four ranks (``srun -n 4 --mpi=pmix python3 <this file>``).
"""

import functools
import importlib
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _SM107, reason="certified on SM107 only"),
]

WORLD = 4
NUM_EXPERTS, TOP_K, LATENT, INTER = 896, 16, 3584, 3072
HELPERS = 8
CAP = 2048  # rows a rank routes per call at most (the planner's and the all-to-all's capacity)
_FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
#: The context targets whose ``_moe_routed`` runs the CuteDSL experts behind the all-to-all.
_MODELING = {t: f"{_FAM}.kimi_k31_pro_nvfp4__sm_107__{t}.modeling" for t in ("cep", "dep16")}
_WORKER_ENV = {
    "CUDA_SCALE_LAUNCH_QUEUES": "4x",
    "TRTLLM_MOE_COMM_TIMEOUT_SEC": "60",
    "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "120",
}
_FIRST_RANK = tuple(range(TOP_K))
_SPREAD = tuple(range(0, NUM_EXPERTS, 56))
#: (every rank's rows, hot experts).
_CASES = (
    ((256, 256, 256, 256), ()),
    ((1024, 1024, 1024, 1024), _FIRST_RANK),
    ((7, 300, 33, 2048), _SPREAD),
    ((512, 0, 128, 64), _FIRST_RANK),  # an idle rank, padded
    ((2048, 2048, 2048, 2048), _FIRST_RANK),
)


def _planes(rank: int) -> dict:
    """This rank's 224 experts in the cep weight layout (``weights.py``): FC1 rows 16 gate / 16 up,
    E2M1 pairs as uint8, 128x4-swizzled E4M3 block scales, per-expert FP32 alphas."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops", "moe"))
    import _rubin_nvfp4_moe_inputs as moe_inputs

    home = NUM_EXPERTS // WORLD
    layer = moe_inputs.layer(moe_inputs.experts(home, LATENT, INTER, 1000 + rank), LATENT)
    return dict(
        fc1_w=layer["fc1_w"].view(torch.uint8).contiguous(),
        fc1_s=layer["fc1_s"].contiguous(),
        fc2_w=layer["fc2_w"].view(torch.uint8).contiguous(),
        fc2_s=layer["fc2_s"].contiguous(),
        fc31_alpha=layer["fc1_alpha"].float().contiguous(),
        fc2_alpha=layer["fc2_alpha"].float().contiguous(),
        fc31_in_scale=layer["g1"].float().reshape(1),
        fc2_in_scale=layer["g2"].float().reshape(1),
    )


def _row_sums(plane: torch.Tensor) -> torch.Tensor:
    """A position-weighted checksum of every slot row of ``plane`` (byte-exact copies agree)."""
    words = plane.reshape(plane.shape[0], -1).view(torch.uint8).to(torch.int64)
    position = torch.arange(1, words.shape[1] + 1, dtype=torch.int64, device=plane.device)
    return (words * position).sum(dim=1)


def _checksums(L, rows: slice) -> torch.Tensor:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_replica_plan import (  # noqa: E501
        LIVE_PLANES,
    )

    return torch.stack([_row_sums(getattr(L, n)[rows]) for n in LIVE_PLANES], dim=1).cpu()


def _moe_w(modeling, planes: dict, **routing):
    fields = {f: None for f in modeling._MoeW._fields}
    fields.update(
        router_bias=torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda"),
        **{k: v for k, v in planes.items()},
        **routing,
    )
    return modeling._MoeW(**fields)


def _inputs(rows: int, runtime_max: int, seed: int, hot):
    """The rank's latent rows and logits, padded to ``runtime_max`` zero rows on an idle rank."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = rows if rows else runtime_max
    x = (torch.randn(n, LATENT, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    logits = torch.randn(n, NUM_EXPERTS, generator=g, device="cuda")
    if hot:
        logits[:, list(hot)] += 4.0
    if not rows:
        x.zero_()
        logits.zero_()
    return x, logits


def _rank(
    target: str = "cep",
    bench: bool = False,
    verbose: bool = False,
    low_precision: bool = True,
    every: int = 1,
    resident: bool = False,
) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
        PerLayerDynamicMoeLoadBalancer,
    )
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
    from tensorrt_llm.mapping import CpType, Mapping

    comm = mpi_comm()
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    MnnvlMemory.initialize()
    if not all(comm.allgather(MnnvlMemory.supports_mnnvl())):
        return dict(rank=rank, skip="per-layer dynamic EPLB needs fabric (MNNVL) memory")
    modeling = importlib.import_module(_MODELING[target])
    replica = importlib.import_module(f"{_FAM}.ops.moe.rubin_moe_replica_plan")
    a2a = importlib.import_module(f"{_FAM}.ops.moe.rubin_moe_a2a")

    mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=1,
        cp_size=WORLD,
        cp_config={"cp_type": CpType.ULYSSES},
        moe_tp_size=1,
        moe_ep_size=WORLD,
    )
    assert mapping.moe_ep_rank == rank, (mapping.moe_ep_rank, rank)
    config = MoePerLayerDynamicLoadBalancerConfig(
        num_slots=NUM_EXPERTS + WORLD * HELPERS,
        auxiliary_sms=8,
        helper_slot_mode="per_layer_resident" if every > 1 or resident else "shared_banks",
    )
    config.setup(ep_rank=rank, ep_size=WORLD)
    manager = PerLayerDynamicMoeLoadBalancer(
        ep_rank=rank, ep_size=WORLD, config=config, mapping=mapping
    )
    errors, notes = [], []
    main = torch.cuda.Stream()
    try:
        with torch.cuda.stream(main):
            home = NUM_EXPERTS // WORLD
            (handle,) = replica.replica_layers(manager, NUM_EXPERTS, TOP_K, 1)
            ref_planes = _planes(rank)
            live = {
                k: torch.nn.Parameter(v.clone(), requires_grad=False) for k, v in ref_planes.items()
            }
            owner = replica.bind_replica_planes(
                handle,
                live,
                mapping=mapping,
                layer_idx=1,
                max_tokens_per_rank=CAP,
                hidden=LATENT,
                intermediate=INTER,
            )
            slots = int(handle.compute_slots_per_rank)
            if slots != home + HELPERS or live["fc1_w"].shape[0] != slots:
                errors.append(f"compute slots {slots}, weights {live['fc1_w'].shape[0]} rows")
            ws = a2a.rubin_moe_a2a_workspace(
                mapping,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                max_num_tokens_per_rank=CAP,
                hidden_size=LATENT,
                dtype=torch.bfloat16,
                use_low_precision_combine=low_precision,
                force_cft=None,
            )
            captured = {}

            def route(ids, *args, _real=handle.route, **kwargs):
                physical = _real(ids, *args, **kwargs)
                captured["routes"] = (ids, physical)
                return physical

            handle.route = route
            cadence = router = None
            if every > 1:
                placements = replica.ReplicaPlacements(manager, mapping, 1, HELPERS)
                cadence = replica.ReplicaCadence(every)
                router = replica.ReplicaRouter(handle, cadence, placements, 0)

                def plan(ids, _real=router.plan):
                    physical, p = _real(ids)
                    captured["routes"] = (ids, physical)
                    return physical, p

                router.plan = plan
            eplb = _moe_w(
                modeling,
                {k: live[k].data if k in live else v for k, v in ref_planes.items()},
                expert_offset=rank * slots,
                num_experts=slots * WORLD,
                plan=functools.partial(replica.rubin_moe_replica_route, handle)
                if router is None
                else router.plan,
                finish=handle.finish_replica_plan if router is None else router.finish,
                record=modeling._moe_no_record,
            )
            plain = _moe_w(
                modeling,
                ref_planes,
                expert_offset=rank * home,
                num_experts=NUM_EXPERTS,
                plan=modeling._moe_resident_plan,
                finish=modeling._moe_no_finish,
                record=modeling._moe_no_record,
            )
            moe_comm = modeling._MoeComm(
                latent_stream=torch.cuda.Stream(),
                memset_stream=torch.cuda.Stream(),
                **{f: torch.cuda.Event() for f in modeling._MoeComm._fields[2:]},
            )
            rt_on = SimpleNamespace(moe_comm=moe_comm, a2a=ws._replace(num_experts=slots * WORLD))
            rt_off = SimpleNamespace(moe_comm=moe_comm, a2a=ws)
            resident_sums = _checksums(eplb, slice(0, home))
            table = torch.cat(comm.allgather(resident_sums))
        main.synchronize()
        for c, (case, hot) in enumerate(_CASES):
            runtime_max = max(case)
            x, logits = _inputs(case[rank], runtime_max, 31 * c + rank, hot)
            st = SimpleNamespace(cuda_graph=False, all_rank_num_tokens=list(case))
            if cadence is not None:
                cadence.forwards = 0
            for again in range(1 if cadence is None else 2):
                if cadence is not None:
                    cadence.begin()  # the first call replans, the second reuses its placement
                with torch.cuda.stream(main):
                    with MoeLoadBalancerIterContext(manager):
                        got = modeling._moe_routed(eplb, x, logits, runtime_max, st, rt_on)
                    want = modeling._moe_routed(plain, x, logits, runtime_max, st, rt_off)
                main.synchronize()
                tag = f"case {c} {case}" + ("" if cadence is None else f" call {again}")
                delta = got.float() - want.float()
                if got.shape != want.shape or not bool(torch.isfinite(got.float()).all()):
                    errors.append(
                        f"{tag}: output {tuple(got.shape)} vs {tuple(want.shape)} or not finite"
                    )
                elif want.numel():
                    norm = want.float().norm().clamp_min(1e-6)
                    rel = float(delta.norm() / norm)
                    peak = float(delta.abs().max() / want.float().abs().max().clamp_min(1e-6))
                    same = bool(torch.equal(got, want))
                    notes.append(f"{tag}: rel err {rel:.1e} peak {peak:.1e} bitwise {same}")
                    bounds = (0.08, 2.0 / 16) if low_precision else (1e-2, 5e-2)
                    if rel > bounds[0] or peak > bounds[1]:
                        errors.append(f"{tag}: relative error {rel:.2e} (peak {peak:.2e})")
                if not torch.equal(_checksums(eplb, slice(0, home)), resident_sums):
                    errors.append(f"{tag}: resident rows changed")
                ids, physical = captured["routes"]
                routes = comm.allgather((ids.cpu().long(), physical.cpu().long()))
                helper = _checksums(eplb, slice(home, slots))
                every_logical, every_physical = [], []
                for source, (logical, phys) in enumerate(routes):
                    owner_rank, local = phys // slots, phys % slots
                    resident = local < home
                    if not torch.equal(logical[resident], (owner_rank * home + local)[resident]):
                        errors.append(
                            f"{tag}: rank {source}: a resident slot serves another expert"
                        )
                    mine = (owner_rank == rank) & ~resident
                    for expert, row in zip(logical[mine].tolist(), (local[mine] - home).tolist()):
                        if not torch.equal(helper[row], table[expert]):
                            errors.append(
                                f"{tag}: helper {row} of rank {rank} is not expert {expert}"
                            )
                            break
                    if case[source]:
                        every_logical.append(logical.flatten())
                        every_physical.append(phys.flatten())
                before = torch.bincount(torch.cat(every_logical) // home, minlength=WORLD)
                after = torch.bincount(torch.cat(every_physical) // slots, minlength=WORLD)
                mean = float(before.sum()) / WORLD
                notes.append(
                    f"{tag}: rank loads {before.tolist()} -> {after.tolist()}, max/mean "
                    f"{float(before.max()) / mean:.3f} -> {float(after.max()) / mean:.3f}"
                )
                if int(after.max()) > int(before.max()):
                    errors.append(f"{tag}: the busiest rank got busier")
                if hot and int(after.max()) >= int(before.max()):
                    errors.append(f"{tag}: no rebalancing")
            if bench:
                times = {}
                for name, L, rt, ctx in (
                    ("off", plain, rt_off, None),
                    ("on", eplb, rt_on, manager),
                ):
                    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
                    with torch.cuda.stream(main):
                        for i in range(25):
                            if i == 5:
                                comm.Barrier()
                                start.record()
                            if cadence is not None and name == "on":
                                cadence.begin()
                            with MoeLoadBalancerIterContext(ctx):
                                modeling._moe_routed(L, x, logits, runtime_max, st, rt)
                        stop.record()
                    main.synchronize()
                    times[name] = start.elapsed_time(stop) / 20 * 1000
                all_times = comm.allgather(times)
                notes.append(
                    f"{tag}: _moe_routed us/call per rank off "
                    f"{[round(t['off']) for t in all_times]} on {[round(t['on']) for t in all_times]}"
                )
            if verbose and rank == 0:
                print(f"rank 0: {notes[-1]}", flush=True)
        del owner
    except Exception:
        if verbose:
            print(f"rank {rank}: {traceback.format_exc()}", flush=True)
        raise
    finally:
        torch.cuda.synchronize()
        if not manager.shutdown():
            errors.append("the EPLB manager did not shut down")
    failures = [f"rank {r}: {e}" for r, items in enumerate(comm.allgather(errors)) for e in items]
    return dict(rank=rank, errors=failures, notes=notes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="four GPUs")
@pytest.mark.parametrize("low_precision", [False, True], ids=["bf16-combine", "fp8-combine"])
@pytest.mark.parametrize("target", sorted(_MODELING))
def test_context_routed_experts_under_per_layer_dynamic_eplb(target, low_precision) -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with MPIPoolExecutor(max_workers=WORLD, env=_WORKER_ENV) as pool:
        results = list(pool.map(lambda _: _rank(target, low_precision=low_precision), range(WORLD)))
    skipped = [r["skip"] for r in results if "skip" in r]
    if skipped:
        pytest.skip(skipped[0])
    assert not results[0]["errors"], "\n".join(results[0]["errors"])


if __name__ == "__main__":
    import faulthandler

    if os.environ.get("CUDA_SCALE_LAUNCH_QUEUES") != "4x":
        raise SystemExit("export CUDA_SCALE_LAUNCH_QUEUES=4x before the launch")
    faulthandler.dump_traceback_later(1200, exit=True)
    target = "dep16" if "--dep16" in sys.argv else "cep"
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    result = _rank(
        target,
        bench="--bench" in sys.argv,
        verbose=True,
        low_precision="--bf16" not in sys.argv,
        every=every,
        resident="--resident" in sys.argv,
    )
    if result["rank"] == 0:
        if "skip" in result:
            print(f"CEP_MOE_EPLB SKIP {result['skip']}")
        else:
            print("\n".join(f"rank 0: {n}" for n in result["notes"]))
            if result["errors"]:
                print("CEP_MOE_EPLB FAIL\n" + "\n".join(result["errors"]))
            else:
                print("CEP_MOE_EPLB PASS")
