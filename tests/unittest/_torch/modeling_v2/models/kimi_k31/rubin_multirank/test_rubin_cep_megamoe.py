# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep context target's routed experts under MegaMoE (``_moe_megamoe``) against the all-to-all
around the CuteDSL experts (``_moe_routed_chunks``), on cep rows, over the EP group of the run:
four SM107 GPUs of one tray (EP4), or eight of two trays (EP8), with the K3.1 expert shapes (896
experts, top-16, NVFP4 3584 x 3072) and the same weights and routing on both paths.

cep rows: every rank routes the same row count (``all_rank_num_tokens`` is ``(rows,) * N``), from
one row to a cep16 step of 131072 tokens (8192 rows a rank), and past the top bucket (two MegaMoE
launches, two all-to-all calls). Per case, against the CuteDSL path:

* with the BF16 combine on both, the routed rows agree to BF16 rounding of the two kernels'
  different summation orders (relative error 1e-2, peak 5e-2 of the largest output);
* against the FP8 combine (``use_low_precision_combine``) within the wire's rounding (each
  rank's partial rounded to E4M3: the band of ``test_rubin_moe_a2a.py``).

``--bench`` (module run) also times both paths per case (``us/call``, the slowest rank).

Requirements: SM107 GPUs with fabric (MNNVL) memory, one NVLink domain; the EP group is the whole
world. Under srun the MPI pool hangs: run the module itself on the ranks
(``srun -n 4 --mpi=pmix python3 <this file>``, or ``-N 2 --ntasks-per-node 4`` for EP8).
"""

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

NUM_EXPERTS, TOP_K, LATENT, INTER = 896, 16, 3584, 3072
#: Rows a rank routes per launch at most: a cep16 step of 131072 tokens (the MegaMoE top bucket
#: and the all-to-all's capacity).
CAP = 8192
_FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_MODELING = f"{_FAM}.kimi_k31_pro_nvfp4__sm_107__cep.modeling"
_WORKER_ENV = {"TRTLLM_MOE_COMM_TIMEOUT_SEC": "60", "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "300"}
_SPREAD = tuple(range(0, NUM_EXPERTS, 56))
#: (rows every rank routes, hot experts): one row, a short step, buckets of both kernels, a cep16
#: step at the 131072-token budget, skewed routing, and a step past the top bucket.
_CASES = (
    (1, ()),
    (96, ()),
    (1024, ()),
    (4096, ()),
    (CAP, ()),
    (CAP, _SPREAD),
    (CAP + 1808, ()),
)


def _planes(rank: int, local: int) -> dict:
    """This rank's experts in the cep weight layout (``weights.py``): FC1 rows 16 gate / 16 up,
    E2M1 pairs as uint8, 128x4-swizzled E4M3 block scales, per-expert FP32 alphas, MegaMoE's FC2
    input scale per expert."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops", "moe"))
    import _rubin_nvfp4_moe_inputs as moe_inputs

    layer = moe_inputs.layer(moe_inputs.experts(local, LATENT, INTER, 1000 + rank), LATENT)
    g2 = layer["g2"].float().reshape(1)
    return dict(
        fc1_w=layer["fc1_w"].view(torch.uint8).contiguous(),
        fc1_s=layer["fc1_s"].contiguous(),
        fc2_w=layer["fc2_w"].view(torch.uint8).contiguous(),
        fc2_s=layer["fc2_s"].contiguous(),
        fc31_alpha=layer["fc1_alpha"].float().contiguous(),
        fc2_alpha=layer["fc2_alpha"].float().contiguous(),
        fc31_in_scale=layer["g1"].float().reshape(1),
        fc2_in_scale=g2,
        fc1_norm_const=g2.expand(local).contiguous(),
    )


def _moe_w(modeling, planes: dict, rank: int, local: int):
    fields = {f: None for f in modeling._MoeW._fields}
    fields.update(
        router_bias=torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda"),
        expert_offset=rank * local,
        num_experts=NUM_EXPERTS,
        plan=modeling._moe_resident_plan,
        finish=modeling._moe_no_finish,
        record=modeling._moe_no_record,
        **planes,
    )
    return modeling._MoeW(**fields)


def _inputs(rows: int, seed: int, hot):
    """The rank's latent rows and router logits."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, LATENT, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    logits = torch.randn(rows, NUM_EXPERTS, generator=g, device="cuda") * 2
    if hot:
        logits[:, list(hot)] += 4.0
    return x, logits


def _time(fn, comm, main, iters=10, warm=3) -> float:
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.cuda.stream(main):
        for i in range(warm + iters):
            if i == warm:
                comm.Barrier()
                start.record()
            fn()
        stop.record()
    main.synchronize()
    return start.elapsed_time(stop) / iters * 1000


def _rank(bench: bool = False, verbose: bool = False) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._utils import mpi_comm
    from tensorrt_llm.mapping import CpType, Mapping

    comm = mpi_comm()
    rank, world = tensorrt_llm.mpi_rank(), comm.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    MnnvlMemory.initialize()
    if not all(comm.allgather(MnnvlMemory.supports_mnnvl())):
        return dict(rank=rank, skip="the all-to-all and MegaMoE need fabric (MNNVL) memory")
    modeling = importlib.import_module(_MODELING)
    a2a = importlib.import_module(f"{_FAM}.ops.moe.rubin_moe_a2a")
    mm = importlib.import_module(f"{_FAM}.ops.moe.rubin_megamoe")
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=1,
        cp_size=world,
        cp_config={"cp_type": CpType.ULYSSES},
        moe_tp_size=1,
        moe_ep_size=world,
    )
    assert mapping.moe_ep_rank == rank, (mapping.moe_ep_rank, rank)
    local = NUM_EXPERTS // world
    errors, notes = [], []
    main = torch.cuda.Stream()
    try:
        with torch.cuda.stream(main):
            L = _moe_w(modeling, _planes(rank, local), rank, local)
            mws = mm.rubin_megamoe_workspace(
                mm.megamoe_ep_group(mapping),
                rank,
                world,
                local,
                LATENT,
                INTER,
                CAP,
                CAP,
                torch.device("cuda"),
            )
            moe_comm = modeling._MoeComm(
                latent_stream=torch.cuda.Stream(),
                memset_stream=torch.cuda.Stream(),
                **{f: torch.cuda.Event() for f in modeling._MoeComm._fields[2:]},
            )
            workspaces = {
                low: a2a.rubin_moe_a2a_workspace(
                    mapping,
                    num_experts=NUM_EXPERTS,
                    top_k=TOP_K,
                    max_num_tokens_per_rank=CAP,
                    hidden_size=LATENT,
                    dtype=torch.bfloat16,
                    use_low_precision_combine=low,
                    force_cft=None,
                )
                for low in (False, True)
            }
            rt_mm = SimpleNamespace(megamoe=mws, moe_rows=CAP)
            rt_cd = {
                low: SimpleNamespace(moe_comm=moe_comm, a2a=ws, moe_rows=CAP)
                for low, ws in workspaces.items()
            }
        main.synchronize()
        for c, (rows, hot) in enumerate(_CASES):
            x, logits = _inputs(rows, 31 * c + rank, hot)
            st = SimpleNamespace(cuda_graph=False, all_rank_num_tokens=[rows] * world)
            with torch.cuda.stream(main):
                got = modeling._moe_megamoe(L, x, logits, rows, st, rt_mm).clone()
                want = {}
                for low in (False, True):
                    want[low] = modeling._moe_routed_chunks(L, x, logits, rows, st, rt_cd[low])
            main.synchronize()
            tag = f"case {c} rows {rows}" + (" skewed" if hot else "")
            for low, bounds in ((False, (1e-2, 5e-2)), (True, (0.08, 2.0 / 16))):
                ref = want[low]
                if got.shape != ref.shape or not bool(torch.isfinite(got.float()).all()):
                    errors.append(f"{tag}: MegaMoE {tuple(got.shape)} vs {tuple(ref.shape)}")
                    continue
                delta = got.float() - ref.float()
                rel = float(delta.norm() / ref.float().norm().clamp_min(1e-6))
                peak = float(delta.abs().max() / ref.float().abs().max().clamp_min(1e-6))
                combine = "fp8" if low else "bf16"
                notes.append(f"{tag} vs CuteDSL ({combine} combine): rel {rel:.1e} peak {peak:.1e}")
                if rel > bounds[0] or peak > bounds[1]:
                    errors.append(f"{tag} ({combine} combine): rel {rel:.2e}, peak {peak:.2e}")
            if bench:
                times = dict(
                    megamoe=_time(
                        lambda: modeling._moe_megamoe(L, x, logits, rows, st, rt_mm), comm, main
                    ),
                    cutedsl=_time(
                        lambda: modeling._moe_routed_chunks(L, x, logits, rows, st, rt_cd[True]),
                        comm,
                        main,
                    ),
                )
                every = comm.allgather(times)
                notes.append(
                    f"{tag}: us/call slowest rank MegaMoE "
                    f"{max(t['megamoe'] for t in every):.0f} CuteDSL (fp8 combine) "
                    f"{max(t['cutedsl'] for t in every):.0f}"
                )
            if verbose and rank == 0:
                print(f"rank 0: {notes[-1]}", flush=True)
    except Exception:
        if verbose:
            print(f"rank {rank}: {traceback.format_exc()}", flush=True)
        raise
    finally:
        torch.cuda.synchronize()
    failures = [f"rank {r}: {e}" for r, items in enumerate(comm.allgather(errors)) for e in items]
    return dict(rank=rank, errors=failures, notes=notes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="four GPUs")
def test_cep_routed_experts_megamoe_against_cutedsl() -> None:
    from mpi4py.futures import MPIPoolExecutor

    world = 4
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} GPUs")
    with MPIPoolExecutor(max_workers=world, env=_WORKER_ENV) as pool:
        results = list(pool.map(lambda _: _rank(), range(world)))
    skipped = [r["skip"] for r in results if "skip" in r]
    if skipped:
        pytest.skip(skipped[0])
    assert not results[0]["errors"], "\n".join(results[0]["errors"])


if __name__ == "__main__":
    import faulthandler

    faulthandler.dump_traceback_later(1800, exit=True)
    for key, value in _WORKER_ENV.items():
        os.environ.setdefault(key, value)
    result = _rank(bench="--bench" in sys.argv, verbose=True)
    if result["rank"] == 0:
        if "skip" in result:
            print(f"CEP_MEGAMOE SKIP {result['skip']}")
        else:
            print("\n".join(f"rank 0: {n}" for n in result["notes"]))
            if result["errors"]:
                print("CEP_MEGAMOE FAIL\n" + "\n".join(result["errors"]))
            else:
                print("CEP_MEGAMOE PASS")
