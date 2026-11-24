# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep target's routed experts under per-layer dynamic EPLB, on four GPUs of one NVLink domain (attention DP over
EP4, the one-sided all-to-all), with the K3.1 expert shapes (896 experts, top-16, MXFP4 3584 x 3072) and 8 helper
slots per rank (928 slots: 224 resident experts + 8 helper slots per rank).

Per call, against the same layer without a load balancer (same packed weights):

* the output agrees to BF16 rounding (a pick served by a helper slot is summed on another rank);
* every physical route of every rank is a slot of the right expert: a resident slot its own expert, a helper slot of
  this rank a byte-exact copy of the routed expert's four runner-layout tensors (checksums of every expert's resident
  rows), and the resident rows never change;
* on skewed routings the busiest rank's routed rows drop;

over uneven rows, an idle rank (DP padding), the TRTLLM-Gen runner's rows (8192 dispatched rows) and a call split into
chunks by the MoE token budget (every chunk plans and releases its own generation). Two layers alternate the two shared
helper banks. The shared-expert overlap runs once inside the first plan of a call, the work queued behind the dispatch
(``run_after_next_dispatch``) runs once per call, and the load record (``ops/moe/replica_balance``) flushes its
windows over the EP group.

Requirements: four SM100-family GPUs with fabric (MNNVL) memory and the dynamic EPLB kernels in the build. The workers
start with ``CUDA_SCALE_LAUNCH_QUEUES=4x``. Under srun the MPI pool hangs: run the module itself on four ranks instead
(``srun -n 4 --mpi=pmix python3 <this file>``, with ``CUDA_SCALE_LAUNCH_QUEUES=4x`` exported).
"""

import importlib
import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

pytestmark = pytest.mark.threadleak(enabled=False)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
NUM_EXPERTS, TOP_K, HIDDEN, INTER = 896, 16, 3584, 3072
HELPERS = 8
CAPS = (4.0, 25.0)
_WORKER_ENV = {
    "CUDA_SCALE_LAUNCH_QUEUES": "4x",
    # A stuck cross-rank wait then fails within a minute instead of five.
    "TRTLLM_MOE_COMM_TIMEOUT_SEC": "60",
    "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "120",
}
_FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_MODELING = f"{_FAM}.kimi_k31_pro_mxfp4__sm_103__dep.modeling"
_PLANES = ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale")

# Hot experts: some of the first rank's, of the last rank's, and spread over the ranks.
_FIRST_RANK = tuple(range(TOP_K))
_LAST_RANK = tuple(range(NUM_EXPERTS - TOP_K, NUM_EXPERTS))
_SPREAD = tuple(range(0, NUM_EXPERTS, 56))
#: (every rank's rows, hot experts, MoE token budget per call (None: max_num_tokens x ranks)).
_CASES = (
    ((64, 64, 64, 64), (), None),
    ((300, 300, 300, 300), _FIRST_RANK, None),
    ((7, 0, 33, 64), _LAST_RANK, None),  # an idle rank
    ((1024, 0, 512, 1000), _SPREAD, None),
    ((2048, 2048, 2048, 2048), _FIRST_RANK, None),  # 8192 dispatched rows: the runner
    ((4096, 0, 1500, 5), _FIRST_RANK, 4096),  # four chunks, empty chunks on two ranks
)


def _init_worker(custom_paths):
    for custom_path in custom_paths:
        if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
            sys.path.append(custom_path)


@pytest.fixture(scope="module")
def executor():
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the dep target's MoE kernels and the dynamic EPLB kernels: sm_100 / sm_103")
    with MPIPoolExecutor(
        initializer=_init_worker, initargs=(sys.path,), max_workers=WORLD, env=_WORKER_ENV
    ) as pool:
        yield pool


def _expert(layer: int, e: int) -> dict:
    """Checkpoint tensors of expert ``e`` of ``layer`` (deterministic): packed MXFP4 codes and UE8M0 scales."""
    g = torch.Generator(device="cuda").manual_seed(100_000 * layer + e)

    def codes(*shape):
        return torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.uint8)

    def scales(*shape):
        return torch.randint(117, 125, shape, generator=g, device="cuda", dtype=torch.uint8)

    return dict(
        w1_weight=codes(INTER, HIDDEN // 2),
        w1_weight_scale=scales(INTER, HIDDEN // 32),
        w3_weight=codes(INTER, HIDDEN // 2),
        w3_weight_scale=scales(INTER, HIDDEN // 32),
        w2_weight=codes(HIDDEN, INTER // 2),
        w2_weight_scale=scales(HIDDEN, INTER // 32),
    )


def _bias() -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(7)
    return (torch.randn(NUM_EXPERTS, generator=g, device="cuda") * 0.1).float().contiguous()


def _model_config(mapping, budget, load_balancer):
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig

    pretrained = PretrainedConfig()
    pretrained.torch_dtype = torch.bfloat16
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        max_num_tokens=4096,
        moe_max_num_tokens=budget,
        moe_backend="TRTLLM",
        moe_load_balancer=load_balancer,
    )


def _experts(cfg, layer: int, bias: torch.Tensor, manager=None):
    """The dep target's routed experts of ``layer`` with its packed weights, bound (with ``manager``) to the EPLB
    live-weight arena by the post-load hook."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.mxfp4_expert_pack import (
        load_packed_mxfp4_expert,
    )
    from tensorrt_llm._torch.utils import AuxStreamType

    module = importlib.import_module(_MODELING)
    experts = module.K31TargetRoutedExperts(
        cfg,
        cfg.mapping,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        gate_cap=CAPS[0],
        linear_cap=CAPS[1],
        layer_idx=layer,
        e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
        aux_stream_dict={t: torch.cuda.Stream() for t in AuxStreamType},
        load_balancer=manager,
    )
    for slot, e in enumerate(experts.initial_local_expert_ids):
        load_packed_mxfp4_expert(
            *(getattr(experts, name) for name in _PLANES),
            slot,
            experts._packed_mxfp4_loaded_slots,
            intermediate_size=INTER,
            moe_tp_size=1,
            moe_tp_rank=0,
            **_expert(layer, e),
        )
    experts.post_load_weights()
    return experts


def _row_sums(plane: torch.Tensor) -> torch.Tensor:
    """A position-weighted int64 checksum of every slot row of ``plane`` (byte-exact copies agree)."""
    words = plane.reshape(plane.shape[0], -1).view(torch.int64)
    position = torch.arange(1, words.shape[1] + 1, dtype=torch.int64, device=plane.device)
    return torch.stack([(row * position).sum() for row in words])


def _checksums(experts, rows: slice) -> torch.Tensor:
    """``[rows, planes]`` checksums of the given slot rows of every live-weight plane."""
    return torch.stack([_row_sums(getattr(experts, name)[rows]) for name in _PLANES], dim=1).cpu()


def _inputs(rows: int, seed: int, hot):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    logits = torch.randn(rows, NUM_EXPERTS, generator=g, device="cuda") * 2.0
    if hot:
        logits[:, list(hot)] += 4.0
    return x, logits


def _output_errors(tag: str, got: torch.Tensor, want: torch.Tensor, notes: list) -> list:
    """The EPLB layer's output against the layer without a load balancer: the same up to BF16 rounding (a pick a
    helper slot serves joins another rank's partial sum). Notes the relative error."""
    if got.shape != want.shape or not bool(torch.isfinite(got.float()).all()):
        return [f"{tag}: output {tuple(got.shape)} vs {tuple(want.shape)} or not finite"]
    if got.numel() == 0:
        return []
    delta = got.float() - want.float()
    rel = float(delta.norm() / want.float().norm().clamp_min(1e-6))
    peak = float(delta.abs().max() / want.float().abs().max().clamp_min(1e-6))
    notes.append(f"{tag}: output relative error {rel:.1e} (peak {peak:.1e})")
    if rel > 1e-2 or peak > 5e-2:
        return [f"{tag}: relative error {rel:.2e} (peak {peak:.2e})"]
    return []


def _route_errors(comm, rank, experts, table, captured, skewed, notes, tag):
    """Check the last plan of an EPLB layer from every rank's routes and this rank's helper rows; notes the rank
    loads and the routes on helper slots."""
    errors = []
    home = experts.expert_size_per_partition
    slots = experts.compute_slot_count
    ids, physical = captured
    routes = comm.allgather((ids.cpu().long(), physical.cpu().long()))
    helper = _checksums(experts, slice(home, slots))
    every_logical, every_physical = [], []
    for source, (logical, phys) in enumerate(routes):
        if phys.shape != logical.shape:
            errors.append(
                f"rank {source}: route shapes {tuple(phys.shape)} vs {tuple(logical.shape)}"
            )
            continue
        if not bool(((phys >= 0) & (phys < WORLD * slots)).all()):
            errors.append(f"rank {source}: a physical slot out of range")
            continue
        owner, local = phys // slots, phys % slots
        resident = local < home
        if not torch.equal(logical[resident], (owner * home + local)[resident]):
            errors.append(f"rank {source}: a resident slot serves another expert")
        mine = (owner == rank) & ~resident
        for expert, row in zip(logical[mine].tolist(), (local[mine] - home).tolist()):
            if not torch.equal(helper[row], table[expert]):
                errors.append(
                    f"rank {source}: helper slot {row} of rank {rank} does not hold expert {expert}"
                )
                break
        every_logical.append(logical.flatten())
        every_physical.append(phys.flatten())
    if every_logical:
        physical = torch.cat(every_physical)
        before = torch.bincount(torch.cat(every_logical) // home, minlength=WORLD)
        after = torch.bincount(physical // slots, minlength=WORLD)
        loads = f"rank loads {before.tolist()} -> {after.tolist()}"
        notes.append(
            f"{tag}: {loads}, {int((physical % slots >= home).sum())} routes on helper slots"
        )
        if int(after.max()) > int(before.max()):
            errors.append(f"the busiest rank got busier: {loads}")
        if skewed and int(after.max()) >= int(before.max()):
            errors.append(f"no rebalancing: {loads}")
    return errors


def _rank_eplb(verbose: bool = False) -> dict:
    """One rank's run of every case; returns ``dict(rank, errors)`` (or ``skip``). ``verbose``: print each case's
    errors on this rank as they are found."""
    import tensorrt_llm
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
    from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import run_after_next_dispatch
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
        PerLayerDynamicMoeLoadBalancer,
    )
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping

    comm = mpi_comm()
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    MnnvlMemory.initialize()
    if not all(comm.allgather(MnnvlMemory.supports_mnnvl())):
        return dict(
            rank=rank, skip="per-layer dynamic EPLB needs fabric (MNNVL) memory on every GPU"
        )
    replica_balance = importlib.import_module(f"{_FAM}.ops.moe.replica_balance")

    mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=WORLD,
        moe_tp_size=1,
        moe_ep_size=WORLD,
        enable_attention_dp=True,
    )
    config = MoePerLayerDynamicLoadBalancerConfig(
        num_slots=NUM_EXPERTS + WORLD * HELPERS, auxiliary_sms=8, helper_slot_mode="shared_banks"
    )
    config.setup(ep_rank=rank, ep_size=WORLD)
    manager = PerLayerDynamicMoeLoadBalancer(
        ep_rank=rank, ep_size=WORLD, config=config, mapping=mapping
    )
    bias = _bias()
    errors = []
    # MAIN, as in the executor: one stream of lower priority than the planner's copy stream.
    main = torch.cuda.Stream()
    try:
        with torch.device("cuda"), torch.cuda.stream(main):
            layers = {}
            for budget in (None, 4096):
                cfg = _model_config(mapping, budget, config)
                plain = _model_config(mapping, budget, None)
                for layer in (1, 2) if budget is None else (3,):
                    eplb = _experts(cfg, layer, bias, manager)
                    ref = _experts(plain, layer, bias)
                    layers.setdefault(budget, []).append((layer, eplb, ref))
            main.synchronize()
        num_layers = len(manager.single_layer_load_balancers)
        log = replica_balance.ReplicaBalanceLog(
            ep_rank=rank,
            ep_size=WORLD,
            home_experts=NUM_EXPERTS // WORLD,
            compute_slots=NUM_EXPERTS // WORLD + HELPERS,
            num_layers=num_layers,
            top_k=TOP_K,
            ep_comm=lambda: manager.get_ep_comm(None, mapping),
            every=1,
            window=2,
        )
        tables, captured, notes = {}, {}, []
        for budget, items in layers.items():
            for layer, eplb, ref in items:
                home = eplb.expert_size_per_partition
                if tuple(eplb.w3_w1_weight.shape[:1]) != (home + HELPERS,):
                    errors.append(f"layer {layer}: weights hold {eplb.w3_w1_weight.shape[0]} slots")
                # Every expert's resident checksums, and this rank's: the copies never write resident rows.
                mine = _checksums(eplb, slice(0, home))
                tables[layer] = (torch.cat(comm.allgather(mine)), mine)
                eplb.balance_log = log
                handle = eplb.layer_load_balancer

                def route(ids, *args, _handle=handle, _real=handle.route, **kwargs):
                    physical = _real(ids, *args, **kwargs)
                    captured[_handle.layer_idx] = (ids, physical)
                    return physical

                handle.route = route
        for c, (case, hot, budget) in enumerate(_CASES):
            reported = len(errors)
            x, logits = _inputs(case[rank], seed=31 * c + rank, hot=hot)
            overlapped, after_dispatch = [], []
            with torch.cuda.stream(main):
                main.wait_stream(torch.cuda.default_stream())
                with MoeLoadBalancerIterContext(manager):
                    log.begin_forward()
                    outputs = []
                    for layer, eplb, ref in layers[budget]:
                        run_after_next_dispatch(lambda: after_dispatch.append(1))
                        with eplb.replica_plan_overlap(lambda: overlapped.append(1)):
                            got = eplb(x, logits, all_rank_num_tokens=list(case))
                        outputs.append(
                            (layer, eplb, got, ref(x, logits, all_rank_num_tokens=list(case)))
                        )
                    log.end_forward()
            main.synchronize()
            n = len(layers[budget])
            if overlapped != [1] * n or len(after_dispatch) < n:
                errors.append(
                    f"case {c}: overlap ran {len(overlapped)}, after-dispatch {len(after_dispatch)} times"
                )
            for layer, eplb, got, want in outputs:
                tag = f"case {c} {case} layer {layer}"
                errors += _output_errors(tag, got, want, notes)
                table, mine = tables[layer]
                if not torch.equal(
                    _checksums(eplb, slice(0, eplb.expert_size_per_partition)), mine
                ):
                    errors.append(f"{tag}: resident rows changed")
                if budget is None:
                    errors += [
                        f"{tag}: {e}"
                        for e in _route_errors(
                            comm,
                            rank,
                            eplb,
                            table,
                            captured[eplb.layer_load_balancer.layer_idx],
                            bool(hot),
                            notes,
                            tag,
                        )
                    ]
            if verbose:
                for error in errors[reported:]:
                    print(f"rank {rank}: {error}", flush=True)
        # Six forwards in windows of two: the flushes ran at the starts of forwards 3 and 5.
        if rank == 0 and not (0 < log.totals["max_after"] <= log.totals["max_before"]):
            errors.append(f"load record totals {log.totals}")
        if verbose and rank == 0:
            notes.append(f"load record totals {log.totals}")
            print("\n".join(f"rank 0: {note}" for note in notes), flush=True)
        if manager.iter_id != len(_CASES):
            errors.append(f"{manager.iter_id} load-balancer iterations, expected {len(_CASES)}")
    except Exception:
        if verbose:
            print(f"rank {rank}: {traceback.format_exc()}", flush=True)
        raise
    finally:
        torch.cuda.synchronize()
        if not manager.shutdown():
            errors.append("the EPLB manager did not shut down")
    failures = [f"rank {r}: {e}" for r, items in enumerate(comm.allgather(errors)) for e in items]
    return dict(rank=rank, errors=failures)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="four GPUs")
def test_per_layer_dynamic_eplb_routed_experts(executor) -> None:
    results = list(executor.map(lambda _: _rank_eplb(), range(WORLD)))
    skipped = [r["skip"] for r in results if "skip" in r]
    if skipped:
        assert len(skipped) == WORLD, "inconsistent platform support across ranks"
        pytest.skip(skipped[0])
    assert sorted(r["rank"] for r in results) == list(range(WORLD))
    assert not results[0]["errors"], "\n".join(results[0]["errors"])


if __name__ == "__main__":
    import faulthandler

    if os.environ.get("CUDA_SCALE_LAUNCH_QUEUES") != "4x":
        raise SystemExit("export CUDA_SCALE_LAUNCH_QUEUES=4x before the launch")
    faulthandler.dump_traceback_later(1200, exit=True)
    result = _rank_eplb(verbose=True)
    if result["rank"] == 0:
        if "skip" in result:
            print(f"PER_LAYER_DYNAMIC_EPLB SKIP {result['skip']}")
        elif result["errors"]:
            print("PER_LAYER_DYNAMIC_EPLB FAIL\n" + "\n".join(result["errors"]))
        else:
            print("PER_LAYER_DYNAMIC_EPLB PASS")
