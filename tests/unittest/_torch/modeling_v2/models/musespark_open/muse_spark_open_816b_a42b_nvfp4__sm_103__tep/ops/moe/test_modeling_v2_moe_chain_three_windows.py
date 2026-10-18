# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the ``three_window_placement`` decode MoE chain (TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

One MoE layer of synthetic NVFP4 experts: every rank stacks its own block and
its two ring predecessors' (``[own | b-1 | b-2]``, 96 experts; the first 64 are
the two-window stack). Per step, from every rank's pool slice and router logits:

* ``owner``: the prologue's routed pool kernel with owner-only placement, the
  persistent kernel over the 64-stack, the ``finalize_transport`` finalize -- the
  built-in grouping of expert rows into rank partials;
* ``tier3``: the same with the balanced two-window placement (the tier-3 chain);
* ``three_windows``: the routed kernel with ``PLACE_BALANCE3``, the persistent
  kernel over the 96-stack, the ``finalize_transport`` finalize;
* the reference: ``ring_windows.place`` of the built-in routing's selection
  (host), TRTLLM-gen's stacked call over the 96 experts, the built-in fused
  finalize all-reduce -- another producer and another finalize kernel for the
  same placement.

Gated: the three-window chain's expert input, selected ids and FP32 weights are
the built-in chain's bit for bit (placement does not change routing: expert-flip
rate 0); its local ids and stats are ``ring_windows.place``'s; its finalized block
is the reference's bit for bit and the same on every rank. Gated against the
unrounded reference (every rank's FP32 partial from the unfused
``post_expert_norm_combine``, summed in float64): the relative RMS error of
``owner`` (the built-in grouping), ``tier3`` and ``three_windows`` in BF16 ulp;
the three-window mean must be at most ``ERROR_RATIO`` times the two-window one
(the grouping it replaces), and at most ``ERROR_RATIO`` times the built-in
grouping's wherever the two-window one is. Logits as the tier-3 chain test
(random, ties, ulp near-ties, zero keys, extremes), rows 1..8, one rank delayed
per step; then CUDA-graph replays of the three chains with the inputs changed in
place.
"""

import json
import os
import statistics
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
EPS, RENORM_EPS, FIN_EPS = 1e-5, 1e-15, 1e-6
KINDS = ("random", "ties", "ulp", "zeros", "extremes")
ROWS = (8, 1, 5, 8, 3, 7, 2, 8, 4, 6, 8, 8)
#: The accuracy gate: mean error vs the unrounded sum at most this times the grouping compared against.
ERROR_RATIO = 1.10
ULP = 2.0**-8


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
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
    """Workspaces (collective), the 96-expert stack and its 64 prefix, the persistent kernels."""
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
    rw = importlib.import_module(_T + ".kernels.replicated_experts.windows")
    if not (R.available() and fat.HAS_CUTE_DSL):
        pytest.skip("needs the CuTe DSL and flashinfer's RMSNormFP4QuantKernel")
    device = torch.device("cuda", torch.cuda.current_device())
    mapping = Mapping(
        world_size=WORLD, rank=rank, gpus_per_node=4, tp_size=WORLD, moe_ep_size=WORLD
    )
    e = type("Env", (), {})()
    e.comm, e.rank, e.R, e.fat, e.pe, e.rw = comm, rank, R, fat, pe, rw
    e.gather = ProjectionGatherWorkspace(mapping)
    e.inbox = inbox_mod.ProjectionInbox(mapping, row_words=R.ROW_WORDS)
    e.finalize = FinalizeIpcWorkspace(mapping, use_mnnvl=True)
    e.transport = fat.FinalizeTransportWorkspace(
        FinalizeIpcWorkspace(mapping, use_mnnvl=True), mapping, fat.DEFAULT
    )
    blocks = [_block(7, (rank - back) % WORLD, device) for back in range(3)]
    e.stack96 = {k: torch.cat([b[k] for b in blocks]).contiguous() for k in blocks[0]}
    del blocks
    e.stack64 = {k: v[: 2 * BLOCK] for k, v in e.stack96.items()}
    for stack in (e.stack64, e.stack96):
        pe.persistent_experts.prepare(stack["w1"], stack["s1"], stack["w2"], stack["s2"])
    gen = torch.Generator(device=device).manual_seed(77)
    e.norm = (torch.randn(HIDDEN, device=device, generator=gen) * 0.1).to(torch.bfloat16)
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


def _persistent(e, stack, fp4, sf, ids, weights_bf16):
    s = stack
    return e.pe.persistent_experts(
        fp4, sf, ids, weights_bf16, s["w1"], s["s1"], s["w2"], s["s2"],
        s["scale_c"], s["alpha1"], s["alpha2"],
    )  # fmt: skip


def _trtllm_gen(e, stack, fp4, sf, ids, weights_bf16, num_local):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
        fp4_block_scale_moe_runner as moe_op,
    )

    s = stack
    return moe_op(
        None, None, fp4, sf, s["w1"], s["s1"], None, None, None, None, s["w2"], s["s2"], None,
        s["scale_c"], s["alpha1"], s["alpha2"], EXPERTS, TOP_K, None, None, INTER, 0, num_local,
        None, 1, False, 0, topk_weights=weights_bf16, topk_ids=ids, output=None,
        tune_max_num_tokens=8192, use_dp=False,
    )  # fmt: skip


def _routed(e, rows, place):
    return e.R.pool_gather_route(
        e.local[:rows],
        e.logits_local[:rows],
        e.inbox,
        e.gain,
        e.global_scale,
        EPS,
        e.bias,
        RENORM_EPS,
        place,
        e.R.LOCAL_EXPERTS,
        return_gathered=True,
    )


def _served(e, rows: int) -> dict:
    """The three chains (owner, tier3, three_windows): graph-capturable."""
    out = {}
    for name, place, stack in (
        ("owner", e.R.PLACE_OWNER, e.stack64),
        ("tier3", e.R.PLACE_BALANCE, e.stack64),
        ("three_windows", e.R.PLACE_BALANCE3, e.stack96),
    ):
        routed = _routed(e, rows, place)
        sf = routed.scale.reshape(-1).view(torch.float8_e4m3fn)
        ex = _persistent(e, stack, routed.fp4, sf, routed.local_ids, routed.weights_bf16)
        out[name] = e.transport.finalize(ex[0], ex[2], routed.weights, e.norm, FIN_EPS, rows)
        out[f"{name}_rows"], out[f"{name}_index"] = ex[0], ex[2]
        out[f"{name}_routed"] = routed
    return out


def _builtin_routing(e, rows: int):
    from flashinfer.norm import rmsnorm_fp4quant

    gathered = e.gather.gather(e.local[:rows])
    fp4, sf = rmsnorm_fp4quant(
        gathered,
        e.gain,
        eps=EPS,
        global_scale=e.global_scale,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=False,
    )
    e.routing.clear_routing_memo()
    selected, weights = e.routing.apply(e.logits[:rows])
    return fp4.view(torch.uint8), sf.view(torch.uint8), selected, weights


def _reference_chain(e, rows: int, fp4, sf, selected, weights):
    """``ring_windows.place`` (host), TRTLLM-gen over the 96 experts, the built-in finalize."""
    flat, stats = e.rw.place(selected.reshape(-1).tolist(), e.rank, 3)
    ids = torch.tensor(flat, dtype=torch.int32, device=selected.device).reshape(selected.shape)
    stats = torch.tensor(stats, dtype=torch.int32, device=selected.device)
    x_sf = sf.reshape(-1).view(torch.float8_e4m3fn)
    ex = _trtllm_gen(e, e.stack96, fp4, x_sf, ids, weights.to(torch.bfloat16), 3 * BLOCK)
    block = e.finalize.finalize(ex[0], ex[2], weights, e.norm, FIN_EPS, rows)
    return ids, stats, block


def _exact(e, rows_t, index, weights, rows: int) -> torch.Tensor:
    """The block with no BF16 rounding of the rank partials (float64 sum of FP32 partials)."""
    from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

    partial = post_expert_norm_combine(
        rows_t, index.reshape(-1), weights, e.norm, FIN_EPS,
        num_tokens=rows, top_k=TOP_K, hidden_size=HIDDEN,
    ).float()  # fmt: skip
    gathered = e.comm.allgather(partial.double().cpu())
    return sum(gathered).to(partial.device)


def _rel_rms_ulp(exact: torch.Tensor, block: torch.Tensor) -> float:
    diff = block.double() - exact
    return (diff.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt().clamp_min(1e-30)).item() / ULP


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


def _differ(a, b) -> int:
    if a.shape != b.shape:
        return a.numel() + b.numel()
    return int((_bits(a) != _bits(b)).sum())


def _check_step(e, out: dict, rows: int, what: str, errors: dict) -> list:
    found = []
    fp4, sf, selected, weights = _builtin_routing(e, rows)
    ids, stats, block = _reference_chain(e, rows, fp4, sf, selected, weights)
    torch.cuda.synchronize()
    routed = out["three_windows_routed"]
    for got, want, name in (
        (routed.fp4, fp4, "expert input"),
        (routed.scale, sf, "expert input scales"),
        (routed.selected, selected, "selected ids (expert flips)"),
        (routed.weights, weights, "routing weights"),
        (out["tier3_routed"].selected, selected, "tier3 selected ids"),
        (routed.local_ids, ids, "local ids vs ring_windows.place"),
        (routed.stats, stats, "stats vs ring_windows.place"),
        (routed.weights_bf16, weights.to(torch.bfloat16), "BF16 weights"),
        (out["three_windows"], block, "three-window block vs TRTLLM-gen + built-in finalize"),
    ):
        n = _differ(got, want)
        if n:
            found.append(f"{what}: {n} elements of {name} differ")
    everyone = e.comm.allgather(out["three_windows"].cpu())
    if not all(torch.equal(v, everyone[0]) for v in everyone):
        found.append(f"{what}: ranks hold different three-window blocks")
    exact = _exact(e, out["three_windows_rows"], out["three_windows_index"], weights, rows)
    for name in ("owner", "tier3", "three_windows"):
        errors.setdefault(name, []).append(_rel_rms_ulp(exact, out[name]))
    after = stats[WORLD:].tolist()
    errors.setdefault("max_after_three_windows", []).append(max(after))
    errors.setdefault("max_after_tier3", []).append(max(out["tier3_routed"].stats[WORLD:].tolist()))
    return found


def _agree(comm, found: list, what: str) -> None:
    everyone = comm.allgather(found[:4])
    bad = {rank: f for rank, f in enumerate(everyone) if f}
    assert not bad, f"{what}: {bad}"


def _report(e, errors: dict, what: str) -> None:
    if e.rank == 0:
        summary = {k: round(statistics.fmean(v), 4) for k, v in errors.items()}
        summary["worst"] = {k: round(max(v), 4) for k, v in errors.items()}
        print(f"\n[three_windows chain] {what}: {json.dumps(summary)}")
        path = os.environ.get("MODELING_V2_THREE_WINDOWS_CHAIN_OUT")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(dict(what=what, **summary)) + "\n")
    owner = statistics.fmean(errors["owner"])
    tier3 = statistics.fmean(errors["tier3"])
    three_windows_on = statistics.fmean(errors["three_windows"])
    # No worse than the two-window grouping it replaces ...
    assert three_windows_on <= ERROR_RATIO * tier3 + 1e-9, (what, errors)
    # ... and no new exceedance of the built-in grouping's: where the two-window grouping
    # is within ERROR_RATIO of it, the three-window one is too. (On constructions that pile
    # every selected expert into one block the built-in grouping rounds a single rank
    # partial and the two-window grouping exceeds ERROR_RATIO of it as well.)
    if tier3 <= ERROR_RATIO * owner:
        assert three_windows_on <= ERROR_RATIO * owner + 1e-9, (what, errors)


@pytest.mark.parametrize("kind", KINDS)
@torch.inference_mode()
def test_eager_three_windows_chain(env, kind):
    e = env
    found, errors = [], {}
    for step, rows in enumerate(ROWS):
        seed = 1000 * len(kind) + step
        _set_inputs(e, seed, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == step % WORLD:
            time.sleep(0.002)
        out = _served(e, rows)
        torch.cuda.synchronize()
        found += _check_step(e, out, rows, f"eager {kind} step {step} rows={rows}", errors)
    _agree(e.comm, found, f"eager {kind}")
    _report(e, errors, f"eager {kind}")


@pytest.mark.parametrize("rows", (1, 3, 8))
@torch.inference_mode()
def test_replayed_three_windows_chain(env, rows):
    """One CUDA graph of the three chains per row count, replayed with new inputs."""
    e = env
    _set_inputs(e, 1, rows, "random")
    _served(e, rows)
    torch.cuda.synchronize()
    e.comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _served(e, rows)
    torch.cuda.synchronize()
    e.comm.Barrier()
    found, errors = [], {}
    for replay in range(2 * len(KINDS)):
        kind = KINDS[replay % len(KINDS)]
        _set_inputs(e, 7000 + 10 * rows + replay, rows, kind)
        torch.cuda.synchronize()
        e.comm.Barrier()
        if e.rank == replay % WORLD:
            time.sleep(0.002)
        graph.replay()
        torch.cuda.synchronize()
        found += _check_step(e, out, rows, f"graph rows={rows} replay {replay} {kind}", errors)
    _agree(e.comm, found, f"graph rows={rows}")
    _report(e, errors, f"graph rows={rows}")
