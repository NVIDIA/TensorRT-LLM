# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the tier-3 decode MoE chain against the built-in one (TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

One MoE layer of synthetic NVFP4 experts (the rank's 32 and its ring
predecessor's, stacked as ``replicated_experts`` stacks them), fed from every rank's
pool slice and router logits, finalized over the group. Four chains per step:

* ``ring_buddy``: the built-in pool gather + flashinfer RMSNorm/NVFP4, the built-in
  routing kernel, ``ring_buddy_assign``'s balanced placement, TRTLLM-gen's
  stacked expert call, the built-in fused finalize all-reduce;
* ``+ persistent_experts``: the same with the persistent expert kernel;
* ``+ finalize_transport``: the same with the transport finalize reading the
  persistent kernel's slot map;
* ``+ moe_prologue`` (the tier-3 chain): the prologue's routed pool kernel (logit
  exchange, routing and placement inside it) feeding the persistent kernel,
  finalized by ``finalize_transport``.

Every chain's finalized block must equal the ``ring_buddy`` chain's bit for bit, and the routed
kernel's expert input, routing and placement must equal the built-in chain's.
The logits are constructed: random, a third of the experts tied, every logit
within 3 ulps of one value, zero selection keys, and infinite / huge values; the
row count goes through 1..8. Eagerly and replayed from one CUDA graph per chain
with the inputs changed in place, one rank delayed before each step.
"""

import time

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

_T = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)

WORLD = 8
EXPERTS, BLOCK, TOP_K = 256, 32, 8
POOL_COLUMNS = 512
HIDDEN = 4096
INTER = 4096
SF_VEC = 16
#: The pre-expert norm epsilon, the routing renormalization epsilon, the post-expert norm epsilon.
EPS, RENORM_EPS, FIN_EPS = 1e-5, 1e-15, 1e-6
KINDS = ("random", "ties", "ulp", "zeros", "extremes")
ROWS = (8, 1, 5, 8, 3, 7, 2, 8, 4, 6)


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("needs sm_103 or sm_107")
    return comm, rank


def _block(seed: int, block: int, device) -> dict:
    """Expert block ``block`` (32 synthetic NVFP4 experts); every rank builds any block alike."""
    gen = torch.Generator(device=device).manual_seed(seed * 64 + block)
    n = BLOCK
    w1 = torch.randint(0, 256, (n, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=device,
                       generator=gen)  # fmt: skip
    s1 = torch.exp2(torch.randint(-2, 1, (n, 2 * INTER, HIDDEN // SF_VEC), device=device,
                                  generator=gen).float()).to(torch.float8_e4m3fn)  # fmt: skip
    w2 = torch.randint(0, 256, (n, HIDDEN, INTER // 2), dtype=torch.uint8, device=device,
                       generator=gen)  # fmt: skip
    s2 = torch.exp2(torch.randint(-2, 1, (n, HIDDEN, INTER // SF_VEC), device=device,
                                  generator=gen).float()).to(torch.float8_e4m3fn)  # fmt: skip
    ids = torch.arange(block * n, (block + 1) * n, device=device, dtype=torch.float32)
    alpha1 = (1.0 + ids / EXPERTS) * 2.0**-6
    alpha2 = (1.0 + (EXPERTS - ids) / EXPERTS) * 2.0**-6
    return dict(w1=w1, s1=s1, w2=w2, s2=s2, scale_c=alpha1 * 0.5, alpha1=alpha1, alpha2=alpha2)


def _logits(seed: int, rows: int, kind: str) -> torch.Tensor:
    """``[rows, 256]`` FP32 logits every rank builds alike (constructions of the routed op test)."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, EXPERTS, generator=gen) * 2
    if kind == "ties":
        base[:, ::3] = 1.25
    elif kind == "ulp":
        steps = torch.randint(-3, 4, (rows, EXPERTS), generator=gen).float()
        base = 0.75 + steps * 2.0**-24
    elif kind == "zeros":
        base[:, 10:40] = 0.0
    elif kind == "extremes":
        base[0, 5] = float("inf")
        base[min(1, rows - 1), 7] = float("-inf")
        base[:, 100:104] = torch.tensor([1e30, -1e30, 88.0, -88.0])
    return base.float()


def _bias(seed: int, kind: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(10_000 + seed)
    bias = torch.randn(EXPERTS, generator=gen) * 0.05
    if kind == "ties":
        bias[::3] = 0.0
    elif kind == "ulp":
        bias.zero_()
    elif kind == "zeros":
        bias[10:40] = -0.5
    return bias.float()


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    size = tensor.element_size()
    return tensor.view({4: torch.int32, 2: torch.int16}.get(size, torch.uint8))


@pytest.fixture(scope="module")
def env():
    """Workspaces (collective), the stacked expert block and the persistent kernel."""
    comm, rank = _world()
    import importlib

    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenRoutingMethod
    from tensorrt_llm._torch.models.musespark_open_finalize_ar import FinalizeIpcWorkspace
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    R = importlib.import_module(_T + ".ops.comm.pool_gather_route")
    inbox_mod = importlib.import_module(_T + ".ops.comm.projection_inbox")
    fat = importlib.import_module(_T + ".ops.moe.finalize_allreduce_transport")
    pe = importlib.import_module(_T + ".ops.moe.persistent_experts")
    rba = importlib.import_module(_T + ".ops.moe.ring_buddy_assign")
    if not (R.available() and fat.HAS_CUTE_DSL):
        pytest.skip("needs the CuTe DSL and flashinfer's RMSNormFP4QuantKernel")
    device = torch.device("cuda", torch.cuda.current_device())
    mapping = Mapping(
        world_size=WORLD, rank=rank, gpus_per_node=4, tp_size=WORLD, moe_ep_size=WORLD
    )
    e = type("Env", (), {})()
    e.comm, e.rank, e.R, e.fat, e.pe, e.rba = comm, rank, R, fat, pe, rba
    e.gather = ProjectionGatherWorkspace(mapping)
    e.inbox = inbox_mod.ProjectionInbox(mapping, row_words=R.ROW_WORDS)
    e.finalize = FinalizeIpcWorkspace(mapping, use_mnnvl=True)
    e.transport = fat.FinalizeTransportWorkspace(
        FinalizeIpcWorkspace(mapping, use_mnnvl=True), mapping, fat.DEFAULT
    )
    own = _block(7, rank, device)
    predecessor = _block(7, (rank - 1) % WORLD, device)
    e.stacked = {k: torch.cat([own[k], predecessor[k]]).contiguous() for k in own}
    del own, predecessor
    pe.persistent_experts.prepare(
        e.stacked["w1"], e.stacked["s1"], e.stacked["w2"], e.stacked["s2"]
    )
    gen = torch.Generator(device=device).manual_seed(77)
    e.norm = (torch.randn(HIDDEN, device=device, generator=gen) * 0.1).to(torch.bfloat16)
    # Static inputs: the step's values are copied into these in place.
    e.local = torch.zeros(8, POOL_COLUMNS, device=device, dtype=torch.bfloat16)
    e.logits = torch.zeros(8, EXPERTS, device=device)
    e.logits_local = torch.zeros(8, BLOCK, device=device)
    e.bias = torch.zeros(EXPERTS, device=device)
    e.gain = torch.ones(HIDDEN, device=device, dtype=torch.bfloat16)
    e.global_scale = torch.tensor([224.0], device=device)
    e.routing = MuseSparkOpenRoutingMethod(
        top_k=TOP_K,
        num_experts=EXPERTS,
        callable_e_score_correction_bias=lambda: e.bias,
        renorm_eps=RENORM_EPS,
    )
    torch.cuda.synchronize()
    comm.Barrier()
    yield e
    torch.cuda.synchronize()
    comm.Barrier()
    e.finalize.close()
    e.transport.legacy.close()


def _expert_call(e, fp4, sf, ids, weights_bf16):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
        fp4_block_scale_moe_runner as moe_op,
    )

    s = e.stacked
    return moe_op(
        None, None, fp4, sf, s["w1"], s["s1"], None, None, None, None, s["w2"], s["s2"], None,
        s["scale_c"], s["alpha1"], s["alpha2"], EXPERTS, TOP_K, None, None, INTER, 0, 2 * BLOCK,
        None, 1, False, 0, topk_weights=weights_bf16, topk_ids=ids, output=None,
        tune_max_num_tokens=8192, use_dp=False,
    )  # fmt: skip


def _persistent(e, fp4, sf, ids, weights_bf16):
    s = e.stacked
    return e.pe.persistent_experts(
        fp4, sf, ids, weights_bf16, s["w1"], s["s1"], s["w2"], s["s2"],
        s["scale_c"], s["alpha1"], s["alpha2"],
    )  # fmt: skip


def _chains(e, rows: int) -> dict:
    """Every chain on the first ``rows`` rows of the static inputs; returns their outputs."""
    from flashinfer.norm import rmsnorm_fp4quant

    local, logits = e.local[:rows], e.logits[:rows]
    gathered = e.gather.gather(local)
    fp4, sf = rmsnorm_fp4quant(
        gathered,
        e.gain,
        eps=EPS,
        global_scale=e.global_scale,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=False,
    )
    fp4, sf = fp4.view(torch.uint8), sf.view(torch.uint8)
    x_sf = sf.reshape(-1).view(torch.float8_e4m3fn)
    e.routing.clear_routing_memo()
    selected, weights = e.routing.apply(logits)
    ids, weights_bf16, _ = e.rba.ring_buddy_assign(selected, weights, e.rank, True)
    out = {"fp4": fp4, "sf": sf, "selected": selected, "weights": weights, "ids": ids,
           "weights_bf16": weights_bf16}  # fmt: skip
    ring_buddy_on = _expert_call(e, fp4, x_sf, ids, weights_bf16)
    out["ring_buddy"] = e.finalize.finalize(
        ring_buddy_on[0], ring_buddy_on[2], weights, e.norm, FIN_EPS, rows
    )
    persistent_experts_on = _persistent(e, fp4, x_sf, ids, weights_bf16)
    out["ring_buddy_persistent_experts"] = e.finalize.finalize(
        persistent_experts_on[0], persistent_experts_on[2], weights, e.norm, FIN_EPS, rows
    )
    out["ring_buddy_persistent_experts_finalize_transport"] = e.transport.finalize(
        persistent_experts_on[0], persistent_experts_on[2], weights, e.norm, FIN_EPS, rows
    )
    routed = e.R.pool_gather_route(
        local,
        e.logits_local[:rows],
        e.inbox,
        e.gain,
        e.global_scale,
        EPS,
        e.bias,
        RENORM_EPS,
        e.R.PLACE_BALANCE,
        e.R.LOCAL_EXPERTS,
        return_gathered=True,
    )
    r_sf = routed.scale.reshape(-1).view(torch.float8_e4m3fn)
    moe_prologue_on = _persistent(e, routed.fp4, r_sf, routed.local_ids, routed.weights_bf16)
    out["tier3"] = e.transport.finalize(
        moe_prologue_on[0], moe_prologue_on[2], routed.weights, e.norm, FIN_EPS, rows
    )
    out.update(
        r_fp4=routed.fp4,
        r_sf=routed.scale,
        r_selected=routed.selected,
        r_weights=routed.weights,
        r_ids=routed.local_ids,
        r_weights_bf16=routed.weights_bf16,
    )
    return out


def _set_inputs(e, seed: int, rows: int, kind: str) -> None:
    gen = torch.Generator().manual_seed(seed * WORLD + e.rank)
    scale = (0.05, 1.0, 20.0)[seed % 3]
    local = (torch.randn(rows, POOL_COLUMNS, generator=gen) * scale).to(torch.bfloat16)
    logits = _logits(seed, rows, kind)
    e.local[:rows].copy_(local)
    e.logits[:rows].copy_(logits)
    e.logits_local[:rows].copy_(logits[:, BLOCK * e.rank : BLOCK * (e.rank + 1)])
    e.bias.copy_(_bias(seed, kind))
    e.gain.copy_((1 + 0.1 * torch.randn(HIDDEN, generator=torch.Generator().manual_seed(seed)))
                 .to(torch.bfloat16))  # fmt: skip
    e.global_scale.fill_(448.0 * 6.0 / (12.0 * scale))


_PAIRS = (
    ("ring_buddy_persistent_experts", "ring_buddy"),
    ("ring_buddy_persistent_experts_finalize_transport", "ring_buddy"),
    ("tier3", "ring_buddy"),
    ("r_fp4", "fp4"),
    ("r_sf", "sf"),
    ("r_selected", "selected"),
    ("r_weights", "weights"),
    ("r_ids", "ids"),
    ("r_weights_bf16", "weights_bf16"),
)


def _problems(out: dict, what: str) -> list:
    found = []
    for got, want in _PAIRS:
        a, b = out[got], out[want]
        if a.shape != b.shape:
            found.append(f"{what}: {got} {tuple(a.shape)} vs {want} {tuple(b.shape)}")
            continue
        differ = int((_bits(a) != _bits(b)).sum())
        if differ:
            found.append(f"{what}: {differ}/{a.numel()} of {got} differ from {want}")
    return found


def _agree(comm, found: list, what: str) -> None:
    everyone = comm.allgather(found[:4])
    bad = {rank: f for rank, f in enumerate(everyone) if f}
    assert not bad, f"{what}: {bad}"


@pytest.mark.parametrize("kind", KINDS)
@torch.inference_mode()
def test_eager_chains_are_bitwise_the_builtin_chain(env, kind):
    e = env
    found = []
    for step, rows in enumerate(ROWS):
        seed = 1000 * len(kind) + step
        _set_inputs(e, seed, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == step % WORLD:
            time.sleep(0.002)
        out = _chains(e, rows)
        torch.cuda.synchronize()
        found += _problems(out, f"eager {kind} step {step} rows={rows}")
    _agree(e.comm, found, f"eager {kind}")


@pytest.mark.parametrize("rows", (1, 3, 8))
@torch.inference_mode()
def test_replayed_chains_are_bitwise_the_builtin_chain(env, rows):
    """One CUDA graph of every chain per row count, replayed with new inputs and a delayed rank."""
    e = env
    _set_inputs(e, 1, rows, "random")
    _chains(e, rows)
    torch.cuda.synchronize()
    e.comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _chains(e, rows)
    torch.cuda.synchronize()
    e.comm.Barrier()
    found = []
    for replay in range(2 * len(KINDS)):
        kind = KINDS[replay % len(KINDS)]
        _set_inputs(e, 7000 + 10 * rows + replay, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == replay % WORLD:
            time.sleep(0.002)
        graph.replay()
        torch.cuda.synchronize()
        found += _problems(out, f"graph rows={rows} replay {replay} {kind}")
    _agree(e.comm, found, f"graph rows={rows}")
