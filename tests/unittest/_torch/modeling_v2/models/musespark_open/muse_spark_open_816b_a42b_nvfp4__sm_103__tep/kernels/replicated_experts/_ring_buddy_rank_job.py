# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``replicated_experts`` across the real tep8 group: 8 ranks on two NVLink-connected nodes.

One process per GPU, launched by the cluster's MPI (``srun --mpi=pmix -N2
--ntasks-per-node=4``); not a pytest module. Each rank builds the same
synthetic NVFP4 layers (deterministic per layer seed), keeps its own 32-expert
block, obtains its ring predecessor's block through ``ring_buddy``'s
replication path (the NCCL all-gather the target uses at load), and runs the
MoE block twice per routing -- the built-in way (own block, global ids) and
the ring-buddy way (stacked block, ``ring_buddy_assign`` ids) -- each followed by
the real fused finalize all-reduce (``FinalizeIpcWorkspace``, fabric memory).

``--check``: replication is bitwise the predecessor's block; owner-only
placement is bitwise the built-in output; balanced placement is inside the
declared error budget; every rank holds the same output.

``--bench``: per-call time of the MoE block on the critical path (expert op
+ fused finalize all-reduce; every call synchronizes the group), timed as
CUDA-graph replays over many routings and rotating layers so expert weights
stream cold. Routings come from a K=7 routing census (per-rank expert counts
of verify steps, ``--census``) or from a real router (``--router-ckpt``:
the checkpoint's router and expert bias of one layer, random hidden states).

``--persistent-experts-on`` adds the ``persistent_experts`` arms: the ring-buddy
placement with the persistent expert kernel (``ops/moe/persistent_experts.py``)
instead of the stacked TRTLLM-gen call, inline and pre-placed. ``--check`` then
also requires its finalized block to be bitwise the ring-buddy block (balanced)
and the built-in block (owner-only). ``--rounds N`` times every arm N times in a
rotating order (one graph per arm, re-replayed each round) and records each round.

``--finalize-transport-on`` also finalizes through the ``finalize_transport``
kernel (``FinalizeTransportWorkspace`` wrapping a third built-in workspace of the
group): ``--check`` then requires every block it finalizes -- the built-in call's,
the ring-buddy call's and (with ``--persistent-experts-on``) the persistent kernel's,
balanced and owner-only -- to be bitwise the built-in kernel's block of the same
expert call, and ``--bench`` adds the ``finalize_transport`` arms.

``--three-windows-on`` (with ``--persistent-experts-on``) adds the
``three_window_placement`` arms: every rank also replicates its second ring
predecessor's block (``ring_buddy._ring_block``, checked bitwise) into a
96-expert stack, places each routing with ``ring_windows.place`` (host, ahead of
the graph: in the model the placement runs inside the prologue's routed pool
kernel) and runs the persistent kernel over the 96 experts. ``--check`` requires
its finalized block to be the same on every rank and bitwise the block of
TRTLLM-gen's 96-expert call with the same ids, and records the relative RMS error
of the built-in, two-window and three-window blocks against the unrounded sum
(every rank's FP32 partial, summed in float64); ``--bench`` adds the pre-placed
arms ``ring_buddy_persistent_experts_pre`` / ``three_windows_persistent_experts_pre``
(and their ``_finalize_transport_pre`` twins with ``--finalize-transport-on``),
``ring_buddy_persistent_experts_w96_pre`` (the two-window placement, its ids of
experts held elsewhere moved past the third window, through the 96-expert stack
and kernel: the window's own cost; checked bitwise the two-window block), and the
expert kernel alone (``exp_ring_buddy``, ``exp_ring_buddy_w96``,
``exp_three_windows``: no finalize, every rank's own time, summarized as mean and
max over ranks); the census statistics add the three-window maximum.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import torch

HIDDEN = 4096
INTER = 4096
SV = 16
EPS = 1e-6
ELEMENT_ULP, RMS_ULP = 8.0, 1.0


def _log(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def _layer(seed, device):
    """One synthetic 256-expert NVFP4 layer (bytes in arbitrary layout; see the moe test)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
        EXPERTS,
    )

    gen = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randint(0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8,
                       device=device, generator=gen)  # fmt: skip
    s1 = torch.exp2(torch.randint(-2, 1, (EXPERTS, 2 * INTER, HIDDEN // SV), device=device,
                                  generator=gen).float()).to(torch.float8_e4m3fn)  # fmt: skip
    w2 = torch.randint(0, 256, (EXPERTS, HIDDEN, INTER // 2), dtype=torch.uint8, device=device,
                       generator=gen)  # fmt: skip
    s2 = torch.exp2(torch.randint(-2, 1, (EXPERTS, HIDDEN, INTER // SV), device=device,
                                  generator=gen).float()).to(torch.float8_e4m3fn)  # fmt: skip
    ids = torch.arange(EXPERTS, device=device, dtype=torch.float32)
    alpha1 = (1.0 + ids / EXPERTS) * 2.0**-6
    return dict(w1=w1, s1=s1, w2=w2, s2=s2, scale_c=alpha1 * 0.5, alpha1=alpha1,
                alpha2=(1.0 + (EXPERTS - ids) / EXPERTS) * 2.0**-6)  # fmt: skip


def _census_routing(loads, rows, rng, block, topk):
    """Top-k rows whose distinct experts reproduce one per-rank census exactly."""
    chosen = []
    for b, count in enumerate(loads):
        chosen += rng.sample(range(block * b, block * (b + 1)), count)
    rng.shuffle(chosen)
    out = [[] for _ in range(rows)]
    for j, e in enumerate(chosen):
        out[j % rows].append(e)
    for row in out:
        assert len(row) <= topk, "census needs more than rows*topk slots"
        while len(row) < topk:
            c = rng.choice(chosen)
            if c not in row:
                row.append(c)
    return out


def _router_routings(ckpt, layer, count, rows, seed, device):
    """Routing of random hidden states through one checkpoint layer's real router."""
    from safetensors import safe_open

    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    text = cfg.get("text_config", cfg)
    # The open export ships the router's metap multiplier as output_multiplier
    # (configs/musespark_open.py checks the two agree); eps is the config default.
    multiplier = float(text["output_multiplier"])
    eps = float(text.get("moe", {}).get("router_score_division_eps", 1e-15))
    prefix = f"model.language_model.layers.{layer}.mlp.gate"
    tensors = {}
    for name in ("weight", "e_score_correction_bias"):
        key = f"{prefix}.{name}"
        with safe_open(os.path.join(ckpt, index[key]), framework="pt", device="cpu") as f:
            tensors[name] = f.get_tensor(key).to(device=device, dtype=torch.float32)
    weight = tensors["weight"]
    if weight.shape[0] != 256:
        weight = weight.t().contiguous()
    bias = tensors["e_score_correction_bias"]
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for _ in range(count):
        x = torch.randn(rows, weight.shape[1], device=device, generator=gen)
        scores = torch.sigmoid((x @ weight.t()) * multiplier)
        top = torch.topk(scores + bias, 8, dim=-1).indices
        picked = scores.gather(1, top)
        out.append((top.to(torch.int32), picked / (picked.sum(-1, keepdim=True) + eps)))
    return out, dict(multiplier=multiplier, eps=eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--census", default=None)
    parser.add_argument("--router-ckpt", default=None)
    parser.add_argument("--router-layers", default="5,30")
    parser.add_argument("--layers", type=int, default=6, help="distinct synthetic layers to rotate")
    parser.add_argument("--samples", type=int, default=96, help="routings per captured graph")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--persistent-experts-on", action="store_true", help="add the persistent-kernel arms"
    )
    parser.add_argument("--rounds", type=int, default=1, help="timing rounds, rotating arm order")
    parser.add_argument(
        "--finalize-transport-on",
        action="store_true",
        help="also finalize by the finalize_transport kernel",
    )
    parser.add_argument("--three-windows-on", action="store_true", help="add the three-window arms")
    args = parser.parse_args()
    assert not args.three_windows_on or args.persistent_experts_on, (
        "--three-windows-on runs the persistent kernel: pass --persistent-experts-on too"
    )

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    assert world == 8, f"tep8 needs 8 ranks, got {world}"
    local = int(os.environ.get("SLURM_LOCALID", rank % 4))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
        fp4_block_scale_moe_runner as moe_op,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
        BLOCK,
        EXPERTS,
        RANKS,
        TOP_K,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
        ring_buddy_assign as assign,
    )
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.models.musespark_open_finalize_ar import FinalizeIpcWorkspace
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=8, rank=rank, gpus_per_node=4, tp_size=8, moe_ep_size=8)
    workspace = FinalizeIpcWorkspace(mapping, use_mnnvl=True)
    transport_workspace = None
    if args.finalize_transport_on:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
            finalize_allreduce_transport as fat,
        )

        transport_workspace = fat.FinalizeTransportWorkspace(
            FinalizeIpcWorkspace(mapping, use_mnnvl=True), mapping, fat.DEFAULT
        )
    rows = args.rows
    result = dict(rank=rank, rows=rows, host=os.uname().nodename)

    # ---- layers: own block from the deterministic layer, predecessor via replication
    own_layers, stacked_layers, norm_weights = [], [], []
    stacked3_layers = []
    t0 = time.time()
    replication_ok = True
    for li in range(args.layers):
        full = _layer(1000 + li, device)
        own = {k: v[BLOCK * rank : BLOCK * (rank + 1)].clone() for k, v in full.items()}
        prev = (rank - 1) % RANKS
        expected_prev = {k: v[BLOCK * prev : BLOCK * (prev + 1)] for k, v in full.items()}
        replica = {k: ring_buddy._predecessor_block(v, mapping) for k, v in own.items()}
        for k in own:
            if not torch.equal(
                replica[k].view(torch.uint8), expected_prev[k].contiguous().view(torch.uint8)
            ):
                replication_ok = False
        if args.three_windows_on:
            prev2 = (rank - 2) % RANKS
            expected_prev2 = {k: v[BLOCK * prev2 : BLOCK * (prev2 + 1)] for k, v in full.items()}
            replica2 = {k: ring_buddy._ring_block(v, mapping, 2) for k, v in own.items()}
            for k in own:
                if not torch.equal(
                    replica2[k].view(torch.uint8), expected_prev2[k].contiguous().view(torch.uint8)
                ):
                    replication_ok = False
            stacked3 = {k: torch.cat([own[k], replica[k], replica2[k]]).contiguous() for k in own}
            stacked = {k: v[: 2 * BLOCK] for k, v in stacked3.items()}
            stacked3_layers.append(stacked3)
            del expected_prev2, replica2
        else:
            stacked = {k: torch.cat([own[k], replica[k]]).contiguous() for k in own}
        del full, expected_prev, replica
        torch.cuda.empty_cache()
        own_layers.append(own)
        stacked_layers.append(stacked)
        g = torch.Generator(device=device).manual_seed(77 + li)
        norm_weights.append(
            (torch.randn(HIDDEN, device=device, generator=g) * 0.1).to(torch.bfloat16)
        )
    torch.cuda.synchronize()
    result["replication_bitwise"] = replication_ok
    result["setup_s"] = round(time.time() - t0, 1)
    all_ok = comm.allgather(replication_ok)
    _log(rank, f"replication bitwise on every rank: {all_ok}; setup {result['setup_s']} s")

    g = torch.Generator(device=device).manual_seed(5)
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device=device, generator=g)
    x_sf = torch.exp2(torch.randint(-2, 1, (rows * HIDDEN // SV,), device=device, generator=g)
                      .float()).to(torch.float8_e4m3fn)  # fmt: skip

    def expert_call(weights, ids, w_bf16, offset, num_local):
        return moe_op(
            None, None, x, x_sf,
            weights["w1"], weights["s1"], None, None, None, None,
            weights["w2"], weights["s2"], None,
            weights["scale_c"], weights["alpha1"], weights["alpha2"],
            EXPERTS, TOP_K, None, None, INTER, offset, num_local, None, 1, False, 0,
            topk_weights=w_bf16, topk_ids=ids, output=None, tune_max_num_tokens=8192, use_dp=False,
        )  # fmt: skip

    def built_in(li, selected, weights, ws=None):
        out = expert_call(own_layers[li], selected, weights.to(torch.bfloat16), BLOCK * rank, BLOCK)
        return (ws or workspace).finalize(out[0], out[2], weights, norm_weights[li], EPS, rows)

    def ring_buddy_on(li, selected, weights, balance=True, ws=None):
        ids, w_bf16, _ = assign(selected, weights, rank, balance)
        out = expert_call(stacked_layers[li], ids, w_bf16, 0, 2 * BLOCK)
        return (ws or workspace).finalize(out[0], out[2], weights, norm_weights[li], EPS, rows)

    persistent = None
    if args.persistent_experts_on:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.persistent_experts import (  # noqa: E501
            persistent_experts as persistent,
        )

        persistent.prepare(
            stacked_layers[0]["w1"], stacked_layers[0]["s1"],
            stacked_layers[0]["w2"], stacked_layers[0]["s2"],
        )  # fmt: skip
        if args.three_windows_on:
            persistent.prepare(
                stacked3_layers[0]["w1"], stacked3_layers[0]["s1"],
                stacked3_layers[0]["w2"], stacked3_layers[0]["s2"],
            )  # fmt: skip

    def persistent_experts_call(weights, ids, w_bf16):
        return persistent(
            x, x_sf, ids, w_bf16,
            weights["w1"], weights["s1"], weights["w2"], weights["s2"],
            weights["scale_c"], weights["alpha1"], weights["alpha2"],
        )  # fmt: skip

    def persistent_experts_on(li, selected, weights, balance=True, ws=None):
        ids, w_bf16, _ = assign(selected, weights, rank, balance)
        out = persistent_experts_call(stacked_layers[li], ids, w_bf16)
        return (ws or workspace).finalize(out[0], out[2], weights, norm_weights[li], EPS, rows)

    windows = None
    if args.three_windows_on:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
            windows,
        )

    def place3(selected):
        """``ring_windows.place`` for this rank: (ids [M, 8] int32, stats [16] int32)."""
        flat, stats = windows.place(selected.reshape(-1).tolist(), rank, 3)
        ids = torch.tensor(flat, dtype=torch.int32, device=device).reshape(selected.shape)
        return ids, torch.tensor(stats, dtype=torch.int32, device=device)

    def three_windows_on(li, selected, weights, ws=None, ids=None, kernel="persistent"):
        if ids is None:
            ids = place3(selected)[0]
        w_bf16 = weights.to(torch.bfloat16)
        if kernel == "persistent":
            out = persistent_experts_call(stacked3_layers[li], ids, w_bf16)
        else:
            out = expert_call(stacked3_layers[li], ids, w_bf16, 0, 3 * BLOCK)
        return (ws or workspace).finalize(out[0], out[2], weights, norm_weights[li], EPS, rows)

    def exact_block(li, selected, weights):
        """The block with no BF16 rounding of the rank partials (float64 sum of FP32 partials)."""
        from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

        out = expert_call(own_layers[li], selected, weights.to(torch.bfloat16), BLOCK * rank, BLOCK)
        partial = post_expert_norm_combine(
            out[0], out[2], weights, norm_weights[li], EPS,
            num_tokens=rows, top_k=TOP_K, hidden_size=HIDDEN,
        ).double()  # fmt: skip
        return sum(comm.allgather(partial.cpu())).to(device)

    def rel_rms_ulp(exact, block):
        diff = block.double() - exact
        return (diff.pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()).item() / 2**-8

    # ---- routings (identical on every rank)
    sources = {}
    if args.census:
        census = json.load(open(args.census))["census"]
        rng = random.Random(42)
        picks = rng.sample(census, min(len(census), 4 * args.samples))
        gw = torch.Generator(device=device).manual_seed(9)
        routings = []
        for item in picks:
            ids = _census_routing(item["loads"], rows, rng, BLOCK, TOP_K)
            w = torch.softmax(torch.randn(rows, TOP_K, device=device, generator=gw), -1)
            routings.append((torch.tensor(ids, dtype=torch.int32, device=device), w))
        sources["trace_census"] = routings
    if args.router_ckpt:
        for layer in [int(v) for v in args.router_layers.split(",") if v]:
            routings, meta = _router_routings(args.router_ckpt, layer, 2 * args.samples, rows,
                                              1234 + layer, device)  # fmt: skip
            sources[f"router_l{layer}"] = routings
            result[f"router_l{layer}_meta"] = meta
    if not sources:
        rng = random.Random(1)
        sources["uniform"] = [
            (torch.tensor([rng.sample(range(EXPERTS), TOP_K) for _ in range(rows)],
                          dtype=torch.int32, device=device),
             torch.softmax(torch.randn(rows, TOP_K, device=device), -1))
            for _ in range(2 * args.samples)
        ]  # fmt: skip

    # Census statistics of each source (distinct experts, max-rank count before/after placement).
    for name, routings in sources.items():
        before, after, distinct = [], [], []
        for sel, w in routings:
            _, _, stats = assign(sel, w, rank, True)
            s = stats.tolist()
            before.append(max(s[:RANKS]))
            after.append(max(s[RANKS:]))
            distinct.append(sum(s[:RANKS]))
        result[f"{name}_census"] = dict(
            distinct_mean=statistics.fmean(distinct), max_before=statistics.fmean(before),
            max_after=statistics.fmean(after))  # fmt: skip
        _log(rank, f"{name}: distinct {statistics.fmean(distinct):.2f}, max-rank experts "
                   f"{statistics.fmean(before):.2f} -> {statistics.fmean(after):.2f}")  # fmt: skip
        if args.three_windows_on:
            after3 = [max(place3(sel)[1].tolist()[RANKS:]) for sel, _ in routings]
            result[f"{name}_census"]["max_after_three_windows"] = statistics.fmean(after3)
            _log(rank, f"{name}: three windows: max-rank experts "
                       f"{statistics.fmean(after3):.3f}")  # fmt: skip

    # Autotune both expert-call shapes once (local 32 and local 64), as warmup does.
    sel0, w0 = next(iter(sources.values()))[0]
    with autotune():
        built_in(0, sel0, w0)
        ring_buddy_on(0, sel0, w0)
        if args.three_windows_on:
            three_windows_on(0, sel0, w0, kernel="trtllm_gen")
    torch.cuda.synchronize()
    comm.Barrier()

    # ---- correctness
    if args.check:
        worst_rms, worst_elem, differing, total = 0.0, 0.0, 0, 0
        owner_only_equal = True
        persistent_experts_equal_ring_buddy = persistent_experts_owner_equal = True
        finalize_transport_equal = {
            "built_in": True,
            "ring_buddy": True,
            "ring_buddy_owner": True,
            "persistent_experts": True,
            "persistent_experts_owner": True,
        }
        consistent = True
        three_windows_checks = dict(
            three_windows_bitwise_trtllm_gen_96=True,
            three_windows_every_rank_same_output=True,
            three_windows_finalize_transport_bitwise_builtin_finalize=True,
        )
        three_windows_errors = {}
        for name, routings in sources.items():
            for i, (sel, w) in enumerate(routings[: args.samples]):
                li = i % args.layers
                ref = built_in(li, sel, w)
                same = ring_buddy_on(li, sel, w, balance=False)
                bal = ring_buddy_on(li, sel, w, balance=True)
                torch.cuda.synchronize()
                owner_only_equal &= bool(torch.equal(same, ref))
                if args.persistent_experts_on:
                    persistent_experts_bal = persistent_experts_on(li, sel, w, balance=True)
                    persistent_experts_same = persistent_experts_on(li, sel, w, balance=False)
                    torch.cuda.synchronize()
                    persistent_experts_equal_ring_buddy &= bool(
                        torch.equal(persistent_experts_bal, bal)
                    )
                    persistent_experts_owner_equal &= bool(
                        torch.equal(persistent_experts_same, ref)
                    )
                if transport_workspace is not None:
                    # The same expert calls, finalized by the finalize_transport kernel.
                    t = transport_workspace
                    finalize_transport_equal["built_in"] &= bool(
                        torch.equal(built_in(li, sel, w, ws=t), ref)
                    )
                    finalize_transport_equal["ring_buddy_owner"] &= bool(
                        torch.equal(ring_buddy_on(li, sel, w, balance=False, ws=t), ref)
                    )
                    finalize_transport_equal["ring_buddy"] &= bool(
                        torch.equal(ring_buddy_on(li, sel, w, ws=t), bal)
                    )
                    if args.persistent_experts_on:
                        finalize_transport_equal["persistent_experts"] &= bool(
                            torch.equal(persistent_experts_on(li, sel, w, ws=t), bal)
                        )
                        finalize_transport_equal["persistent_experts_owner"] &= bool(
                            torch.equal(persistent_experts_on(li, sel, w, balance=False, ws=t), ref)
                        )
                    torch.cuda.synchronize()
                if args.three_windows_on:
                    three_windows_out = three_windows_on(li, sel, w)
                    three_windows_gen = three_windows_on(li, sel, w, kernel="trtllm_gen")
                    torch.cuda.synchronize()
                    three_windows_checks["three_windows_bitwise_trtllm_gen_96"] &= bool(
                        torch.equal(three_windows_out, three_windows_gen)
                    )
                    three_windows_outs = comm.allgather(three_windows_out.cpu())
                    three_windows_checks["three_windows_every_rank_same_output"] &= all(
                        torch.equal(o, three_windows_outs[0]) for o in three_windows_outs
                    )
                    if transport_workspace is not None:
                        three_windows_checks[
                            "three_windows_finalize_transport_bitwise_builtin_finalize"
                        ] &= bool(
                            torch.equal(
                                three_windows_on(li, sel, w, ws=transport_workspace),
                                three_windows_out,
                            )
                        )
                    exact = exact_block(li, sel, w)
                    for arm, block in (
                        ("built_in", ref),
                        ("ring_buddy", bal),
                        ("three_windows", three_windows_out),
                    ):
                        three_windows_errors.setdefault(f"{name}:{arm}", []).append(
                            rel_rms_ulp(exact, block)
                        )
                diff = bal.float() - ref.float()
                rms = (diff.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item() / 2**-8
                row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
                elem = (diff.abs() / row).max().item() / 2**-8
                worst_rms, worst_elem = max(worst_rms, rms), max(worst_elem, elem)
                differing += int((diff != 0).sum().item())
                total += diff.numel()
                outs = comm.allgather(bal.cpu())
                consistent &= all(torch.equal(o, outs[0]) for o in outs)
        result["check"] = dict(
            owner_only_bitwise=owner_only_equal, every_rank_same_output=consistent,
            balanced_worst_rel_rms_ulp=worst_rms, balanced_worst_element_ulp=worst_elem,
            balanced_fraction_elements_differ=differing / max(total, 1),
            within_budget=worst_rms <= RMS_ULP and worst_elem <= ELEMENT_ULP)  # fmt: skip
        if args.persistent_experts_on:
            result["check"]["persistent_experts_bitwise_ring_buddy"] = (
                persistent_experts_equal_ring_buddy
            )
            result["check"]["persistent_experts_owner_bitwise_built_in"] = (
                persistent_experts_owner_equal
            )
        if transport_workspace is not None:
            result["check"]["finalize_transport_bitwise_builtin_finalize"] = (
                finalize_transport_equal
            )
        if args.three_windows_on:
            result["check"].update(three_windows_checks)
            result["check"]["error_vs_unrounded_ulp"] = {
                k: dict(mean=statistics.fmean(v), worst=max(v), n=len(v))
                for k, v in three_windows_errors.items()
            }
        verdicts = comm.allgather(result["check"])
        _log(rank, "check per rank:", json.dumps(verdicts, indent=None))

    # ---- microbenchmark
    if args.bench:
        bench = {}
        static_sel = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        static_w = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
        for name, routings in sources.items():
            sel_all = torch.stack([s for s, _ in routings[: args.samples]])
            w_all = torch.stack([w for _, w in routings[: args.samples]])
            n = sel_all.shape[0]
            pre_ids, pre_wbf = [], []
            for i in range(n):
                ids, wb, _ = assign(sel_all[i], w_all[i], rank, True)
                pre_ids.append(ids)
                pre_wbf.append(wb)
            pre3_ids = [place3(sel_all[i])[0] for i in range(n)] if args.three_windows_on else []
            # Two-window ids for the 96-expert window: the placement marks experts held elsewhere
            # 64 + (g mod 192), which would land in the third window; move them past it.
            pre_ids_w96 = [
                torch.where(
                    ids >= 2 * BLOCK, 3 * BLOCK + (ids - 2 * BLOCK) % (EXPERTS - 3 * BLOCK), ids
                )
                for ids in pre_ids
            ]
            arms = {}

            def arm_built_in(i):
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                return built_in(i % args.layers, static_sel, static_w)

            def arm_ring_buddy(i):
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                return ring_buddy_on(i % args.layers, static_sel, static_w)

            def arm_ring_buddy_preplaced(i):
                # The placement precomputed: the in-model case, where the
                # assignment kernel runs in main-stream slack before the join.
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                out = expert_call(
                    stacked_layers[i % args.layers], pre_ids[i], pre_wbf[i], 0, 2 * BLOCK
                )
                return workspace.finalize(
                    out[0], out[2], static_w, norm_weights[i % args.layers], EPS, rows
                )

            def arm_persistent_experts(i):
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                return persistent_experts_on(i % args.layers, static_sel, static_w)

            def arm_persistent_experts_preplaced(i):
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                out = persistent_experts_call(
                    stacked_layers[i % args.layers], pre_ids[i], pre_wbf[i]
                )
                return workspace.finalize(
                    out[0], out[2], static_w, norm_weights[i % args.layers], EPS, rows
                )

            def arm_ring_buddy_finalize_transport(i):
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                return ring_buddy_on(i % args.layers, static_sel, static_w, ws=transport_workspace)

            def arm_persistent_experts_finalize_transport(i):
                static_sel.copy_(sel_all[i])
                static_w.copy_(w_all[i])
                return persistent_experts_on(
                    i % args.layers, static_sel, static_w, ws=transport_workspace
                )

            def preplaced(ids_list, stacks, ws=None):
                def arm(i):
                    static_sel.copy_(sel_all[i])
                    static_w.copy_(w_all[i])
                    out = persistent_experts_call(stacks[i % args.layers], ids_list[i], pre_wbf[i])
                    return (ws or workspace).finalize(
                        out[0], out[2], static_w, norm_weights[i % args.layers], EPS, rows
                    )

                return arm

            if args.three_windows_on:
                # Two-window ids through the 96-expert window: the same finalized block, every sample.
                same = True
                for i in range(n):
                    narrow = preplaced(pre_ids, stacked_layers)(i).clone()
                    same &= bool(torch.equal(narrow, preplaced(pre_ids_w96, stacked3_layers)(i)))
                _log(
                    rank,
                    f"bench {name}: two-window ids through the 96-expert window = the two-window block: {same}",
                )
                result.setdefault("check", {})[f"{name}:ring_buddy_w96_bitwise_ring_buddy"] = same

            def expert_only(ids_list, stacks):
                # The expert kernel alone, no finalize: each rank's own time.
                def arm(i):
                    return persistent_experts_call(stacks[i % args.layers], ids_list[i], pre_wbf[i])

                return arm

            arm_fns = [("built_in", arm_built_in), ("ring_buddy", arm_ring_buddy),
                       ("ring_buddy_preplaced", arm_ring_buddy_preplaced)]  # fmt: skip
            if args.three_windows_on:
                arm_fns += [
                    ("ring_buddy_persistent_experts_pre", preplaced(pre_ids, stacked_layers)),
                    ("three_windows_persistent_experts_pre", preplaced(pre3_ids, stacked3_layers)),
                    # The two-window placement through the 96-expert kernel and stack: the
                    # window's own cost, same work and same bits as ring_buddy_persistent_experts_pre.
                    (
                        "ring_buddy_persistent_experts_w96_pre",
                        preplaced(pre_ids_w96, stacked3_layers),
                    ),
                    ("exp_ring_buddy", expert_only(pre_ids, stacked_layers)),
                    ("exp_ring_buddy_w96", expert_only(pre_ids_w96, stacked3_layers)),
                    ("exp_three_windows", expert_only(pre3_ids, stacked3_layers)),
                ]
                if transport_workspace is not None:
                    arm_fns += [
                        (
                            "ring_buddy_persistent_experts_finalize_transport_pre",
                            preplaced(pre_ids, stacked_layers, transport_workspace),
                        ),
                        (
                            "three_windows_persistent_experts_finalize_transport_pre",
                            preplaced(pre3_ids, stacked3_layers, transport_workspace),
                        ),
                        (
                            "ring_buddy_persistent_experts_finalize_transport_w96_pre",
                            preplaced(pre_ids_w96, stacked3_layers, transport_workspace),
                        ),
                    ]
            if args.persistent_experts_on:
                arm_fns += [
                    ("ring_buddy_persistent_experts", arm_persistent_experts),
                    ("ring_buddy_persistent_experts_preplaced", arm_persistent_experts_preplaced),
                ]
            if transport_workspace is not None:
                arm_fns += [("ring_buddy_finalize_transport", arm_ring_buddy_finalize_transport)]
                if args.persistent_experts_on:
                    arm_fns += [
                        (
                            "ring_buddy_persistent_experts_finalize_transport",
                            arm_persistent_experts_finalize_transport,
                        )
                    ]
            graphs = {}
            for arm_name, fn in arm_fns:
                for i in range(3):
                    fn(i)
                torch.cuda.synchronize()
                comm.Barrier()
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph, stream=stream):
                        for i in range(n):
                            fn(i)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                graphs[arm_name] = graph
            per_round = {arm_name: [] for arm_name, _ in arm_fns}
            names = [arm_name for arm_name, _ in arm_fns]
            for r in range(args.rounds):
                # Rotate the arm order every round so drift and warm-up spread over every arm.
                order = names[r % len(names) :] + names[: r % len(names)]
                if r % 2:
                    order.reverse()
                for arm_name in order:
                    graph = graphs[arm_name]
                    times = []
                    for _ in range(7):
                        comm.Barrier()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end) * 1000.0 / n)
                    per_round[arm_name].append(sorted(times)[len(times) // 2])
                    comm.Barrier()
            for arm_name in names:
                arms[arm_name] = statistics.median(per_round[arm_name])
            graphs.clear()
            comm.Barrier()
            if args.three_windows_on:
                # Expert-only arms run without a cross-rank sync: summarize every rank's time.
                everyone = comm.allgather({k: v for k, v in arms.items() if k.startswith("exp_")})
                spread = {
                    k: dict(
                        mean=statistics.fmean(e[k] for e in everyone),
                        max=max(e[k] for e in everyone),
                        per_rank=[round(e[k], 3) for e in everyone],
                    )
                    for k in everyone[0]
                }
                bench[f"{name}_expert_only"] = spread
                _log(rank, f"bench {name} expert kernel alone (us per call, over ranks): " +
                     ", ".join(f"{k} mean {v['mean']:.2f} max {v['max']:.2f}"
                               for k, v in spread.items()))  # fmt: skip
            bench[name] = arms
            bench[f"{name}_rounds"] = per_round
            _log(rank, f"bench {name} (us per MoE block, n={n}, median of {args.rounds} rounds): " +
                 ", ".join(f"{k} {v:.2f}" for k, v in arms.items()))  # fmt: skip
            if args.rounds > 1:
                _log(rank, f"bench {name} rounds: " + json.dumps(
                    {k: [round(v, 2) for v in vs] for k, vs in per_round.items()}))  # fmt: skip
        result["bench"] = bench

    everything = comm.gather(result, root=0)
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(everything, f, indent=1)
    comm.Barrier()
    workspace.close()
    if transport_workspace is not None:
        transport_workspace.legacy.close()


if __name__ == "__main__":
    main()
