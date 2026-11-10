# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-path collectives (``decode_comm.py``) of the Kimi K3 target ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``
against its built-in path, and its route A MoE decode path (``decode_moe.py``), at W ranks of one GB200 tray (default
4), at the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_decode_comm_tp16_moetp4ep4_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _kimi_k3_decode_comm_tp16_moetp4ep4_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of checks inside one W-rank job, sharing the target's collective state. The
collected entry point is ``test_modeling_v2_kimi_k3_decode_comm_tp16_moetp4ep4_op_matrix.py``.

The decode_comm checks are ``_kimi_k3_decode_comm_checks.py`` of the family test directory, all of them at every step
of its ``ALL_STEPS``, the speculative ones included. This file adds the checks of route A's MoE decode path:
  * decode MoE (first, outside inference mode with autograd on, as post_load_weights runs): the MoE decode path built
    from a MoE layer's own modules (the target's gate, whose parameters require grad, the latent projections, the
    stock RMSNorm and shared GatedMLP; zeroed routed experts) at this TP, every kernel warmed up, then a 3-token step
    and a wide 16-token step against the shared expert's forward.
  * decode MoE push: the MoE decode path with random MXFP4 routed experts (route A's layout: 224 local experts of
    intermediate 768, every rank its own EP shard) on the TP group's latent exchange, its latent all-reduce as the
    routed experts' push form plus comm/k3_latent_reduce (``push=True``) against the routed experts' all-reduce
    (``push=False``), bit for bit: one call at every M 1..8, both tails; 3 layers x 12 steps with M dipping and growing
    back, wide steps and one step without the push in between, a random rank late on every pushing call; a 2-layer step
    captured in a CUDA graph and replayed with rewritten inputs; and, at every M 1..8, a 3-layer step captured and
    replayed 16 times (48 push + reduce calls, the exchange's two halves rotating at every layer position). After all
    of it the exchange is empty and its call count has advanced by one per pushing call.
"""

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "comm"))
import _kimi_k3_decode_comm_checks as dc  # noqa: E402
import _lockstep as ls  # noqa: E402
from _kimi_k3_decode_comm_checks import LATENT, MOE_LAYERS, TAIL_ACT, H, _gen, _normal  # noqa: E402

DEADLINE_S = dc.DEADLINE_S


# The MoE decode path at this check's TP: a moetp4ep4 rank's routed experts (intermediate columns, experts), and a
# shared expert whose rank slice is a TP16 rank's (TAIL_ACT = 384 activation columns) at every world.
I_TP, E_LOCAL, NUM_EXPERTS = 768, 224, 896
SITU_CAPS = (4.0, 25.0)
MOE_TOL = 3e-2  # the front's fused shared gate_up + SiTU against the shared expert's own GEMMs


def _decode_moe_layer():
    """A MoE layer with the parts the decode path reads, from the modules KimiK3MoERuntime builds: the target's
    KimiK3MoEGate, nn.Linear latent projections, the stock RMSNorm and GatedMLP (at this check's TP), each with its
    own parameters (the gate's and the latent projections' require grad, as in the model), and zeroed routed-expert
    buffers in TRTLLM-Gen's W4A8_MXFP4_MXFP8 layout, so the routed experts add nothing."""
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm._torch.modules.situ import SituAndMul
    from tensorrt_llm.functional import AllReduceStrategy

    cfg = SimpleNamespace(
        num_experts_per_token=16,
        num_experts=NUM_EXPERTS,
        routed_scaling_factor=2.827,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        hidden_size=H,
    )
    with torch.device("cuda"):
        gate = dc.T.KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16)
        down = nn.Linear(H, LATENT, bias=False, dtype=torch.bfloat16)
        up = nn.Linear(LATENT, H, bias=False, dtype=torch.bfloat16)
        norm = RMSNorm(hidden_size=LATENT, eps=1e-5, dtype=torch.bfloat16)
        shared = GatedMLP(
            hidden_size=H,
            intermediate_size=TAIL_ACT * dc.R.world,
            bias=False,
            activation=SituAndMul(
                beta=SITU_CAPS[0], linear_beta=SITU_CAPS[1], use_fused_activation=True
            ),
            dtype=torch.bfloat16,
            config=ModelConfig(mapping=dc.R.mapping, allreduce_strategy=AllReduceStrategy.MNNVL),
            reduce_output=True,
            layer_idx=MOE_LAYERS[0],
            is_shared_expert=True,
        )
    g = _gen(5000)  # replicated
    gr = _gen(5100 + dc.R.rank)  # this rank's shared expert slices
    with torch.no_grad():
        gate.weight.copy_(_normal(g, gate.weight.shape, 0.02))
        gate.e_score_correction_bias.copy_(
            0.05 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
        )
        down.weight.copy_(_normal(g, down.weight.shape, 0.02))
        up.weight.copy_(_normal(g, up.weight.shape, 0.02))
        norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
        shared.gate_up_proj.weight.copy_(_normal(gr, shared.gate_up_proj.weight.shape, 0.02))
        shared.down_proj.weight.copy_(_normal(gr, shared.down_proj.weight.shape, 0.02))

    def zeros(*shape):
        return nn.Parameter(
            torch.zeros(shape, dtype=torch.uint8, device="cuda"), requires_grad=False
        )

    backend = SimpleNamespace(
        w3_w1_weight=zeros(E_LOCAL, 2 * I_TP, LATENT // 2),
        w3_w1_weight_scale=zeros(E_LOCAL, 2 * I_TP, LATENT // 32),
        w2_weight=zeros(E_LOCAL, LATENT, I_TP // 2),
        w2_weight_scale=zeros(E_LOCAL, LATENT, I_TP // 32),
        expert_size_per_partition=E_LOCAL,
        slot_start=dc.R.rank * E_LOCAL % NUM_EXPERTS,
    )
    all_reduce = AllReduce(
        mapping=dc.R.mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
    )
    return SimpleNamespace(
        num_experts=NUM_EXPERTS,
        top_k=cfg.num_experts_per_token,
        moe_hidden_size=LATENT,
        hidden_size=H,
        gate=gate,
        routed_expert_down_proj=down,
        routed_expert_up_proj=up,
        routed_expert_norm=norm,
        shared_experts=shared,
        routed_experts=SimpleNamespace(backend=backend, all_reduce=all_reduce),
        _situ_betas=SITU_CAPS,
        moe_main_event=torch.cuda.Event(),
        moe_shared_event=torch.cuda.Event(),
        shared_expert_stream=torch.cuda.Stream(),
    )


def check_decode_moe_from_model_parameters():
    """The MoE decode path built as post_load_weights builds it, outside inference mode with autograd on, from a MoE
    layer's own modules (_decode_moe_layer): the shared state, the folded latent norm, the layer's decode weights and
    one call of every kernel (the front, both k3_moe builds, k3_route_quant). Then a step of 3 tokens (the replicated
    tail) and a wide step of 16 (this rank's share, reduced here) against the shared expert's forward: the routed
    experts are zeros, so the two agree."""
    if dc.R.world not in (4, 8, 16):
        if dc.R.rank == 0:
            print(
                f"[rank 0] decode MoE skipped at world {dc.R.world}: the front runs 4, 8 or 16 ranks",
                flush=True,
            )
        return
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    dm = dc.T._decode_moe
    moe = _decode_moe_layer()
    assert moe.gate.e_score_correction_bias.requires_grad
    mnnvl = dc.T._decode_comm.MnnvlWorkspace.create(
        dc.R.mapping, dc.T._decode_comm.MNNVL_BUFFER_BYTES
    )
    device = torch.device("cuda", torch.cuda.current_device())
    state = dm.K3DecodeMoe.create(dc.R.mapping, device, I_TP, E_LOCAL, mnnvl)
    dm.fold_latent_norm(moe)
    layer = dm.K3DecodeMoeLayer.create(moe, state, dc.R.rank, dc.R.world)
    layer.warm_up(moe)
    for rows in (3, 16):
        x = _normal(_gen(5200 + rows), (rows, H), 0.5)
        y = layer.forward(moe, x, None, partial_tail=False)
        if rows > dm.MAX_TOKENS:  # a wide step returns this rank's share
            y = moe.routed_experts.all_reduce(y)
        want = moe.shared_experts(x)
        torch.cuda.synchronize()
        err = ls.rel_err(y, want)
        if dc.R.rank == 0:
            print(f"[rank 0] decode MoE, {rows} tokens: vs the shared expert {err:.2e}", flush=True)
        assert torch.isfinite(y.float()).all() and err <= MOE_TOL, (rows, err)
        assert dc.R.same_on_ranks(y), (rows, "ranks differ")


PUSH_LAYERS = 3
PUSH_STEPS = (8, 3, 1, 8, 16, 5, 2, 8, 1, 16, 7, 8)  # tokens per step; 16: a wide step
ONE_SHOT_STEP = 5  # the sequence's step that keeps the routed experts' all-reduce
RING_REPLAYS = 16
EXCHANGE_EMPTY = -(2**31)  # an exchange word no push has written


def _rand_mxfp4(rows, k, gen):
    """Random checkpoint-format MXFP4 (packed [rows, k / 2], one E8M0 per 32 k), as test_modeling_v2_k3_moe.py draws
    it."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k))
    exps = torch.randint(
        base, base + 6, (rows, k // 32), dtype=torch.uint8, device="cuda", generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


def _load_experts(backend, seed):
    """Random MXFP4 routed experts through the TRTLLM-Gen W4A8_MXFP4_MXFP8 loader into ``backend``'s buffers, at
    route A's layout: intermediate 768 per rank (moe_tp 4), 224 local experts."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    i_full = I_TP * 4
    module = SimpleNamespace(
        tp_size=4,
        tp_rank=0,
        scaling_vector_size=32,
        intermediate_size=i_full,
        intermediate_size_per_partition=I_TP,
        hidden_size=LATENT,
    )
    gen = _gen(seed)
    with torch.no_grad():
        for e in range(E_LOCAL):
            w1, w1s = _rand_mxfp4(i_full, LATENT, gen)  # gate
            w3, w3s = _rand_mxfp4(i_full, LATENT, gen)  # up
            w2, w2s = _rand_mxfp4(LATENT, i_full, gen)  # down
            method.load_expert_w3_w1_weight(module, w1, w3, backend.w3_w1_weight[e])
            method.load_expert_w2_weight(module, w2, backend.w2_weight[e])
            method.load_expert_w3_w1_weight_scale_mxfp4(
                module, w1s, w3s, backend.w3_w1_weight_scale[e]
            )
            method.load_expert_w2_weight_scale_mxfp4(module, w2s, backend.w2_weight_scale[e])
    torch.cuda.synchronize()


def _outs(out):
    """A decode MoE call's tensors: a PendingTail's latent and shared activation, else the output."""
    return (out.latent, out.act) if isinstance(out, dc.T._decode_comm.PendingTail) else (out,)


def _same_bits(got, want) -> bool:
    return all(
        g.shape == w.shape
        and torch.equal(g.contiguous().view(torch.int16), w.contiguous().view(torch.int16))
        for g, w in zip(got, want)
    )


def check_decode_moe_push():
    """The decode MoE's latent all-reduce as the routed experts' push form plus comm/k3_latent_reduce, against the
    routed experts' all-reduce, bit for bit (see the module docstring)."""
    if dc.R.world not in (4, 8, 16):
        if dc.R.rank == 0:
            print(f"[rank 0] decode MoE push skipped at world {dc.R.world}", flush=True)
        return
    dm = dc.T._decode_moe
    moe = _decode_moe_layer()
    _load_experts(moe.routed_experts.backend, 6000 + dc.R.rank)
    device = torch.device("cuda", torch.cuda.current_device())
    state = dm.K3DecodeMoe.create(dc.R.mapping, device, I_TP, E_LOCAL, dc.COMM.mnnvl)
    ex = state.exchange
    assert ex is not None, "no latent exchange at a TP size it takes"
    dm.fold_latent_norm(moe)
    layers = [
        dm.K3DecodeMoeLayer.create(moe, state, dc.R.rank, dc.R.world) for _ in range(PUSH_LAYERS)
    ]
    layers[0].warm_up(moe)
    dc.R.barrier()
    count0 = int(ex.flags[0].item())
    pushes = 0

    def call(layer, x, push, tail):
        nonlocal pushes
        out = _outs(layer.forward(moe, x, None, partial_tail=tail, push=push))
        pushes += int(push and x.shape[0] <= dm.MAX_TOKENS)
        return out

    # One call at every M, both tails.
    for m in range(1, dm.MAX_TOKENS + 1):
        x = _normal(_gen(6100 + m), (m, H), 0.5)
        for tail in (False, True):
            want = call(layers[0], x, False, tail)
            got = call(layers[0], x, True, tail)
            torch.cuda.synchronize()
            assert _same_bits(got, want), (m, tail, "push != one-shot")
            assert got[0].float().abs().sum().item() > 0, (m, tail, "a zero output proves nothing")
            # The latent and the replicated output are the same on every rank; the shared activation is the rank's.
            assert dc.R.same_on_ranks(got[0]), (m, tail, "ranks differ")

    # A sequence of steps over every layer, both paths; a random rank late on every pushing call.
    late = random.Random(11)
    runs = {}
    for push in (False, True):
        outs = []
        for i, m in enumerate(PUSH_STEPS):
            x = _normal(_gen(6200 + i), (m, H), 0.5)
            for j, layer in enumerate(layers):
                pushing = push and i != ONE_SHOT_STEP
                dc.R.late(late.randrange(dc.R.world) if pushing else None)
                outs.append(call(layer, x, pushing, tail=j % 2 == 0))
        torch.cuda.synchronize()
        runs[push] = outs
    assert all(_same_bits(g, w) for g, w in zip(runs[True], runs[False])), (
        "sequence: push != one-shot"
    )

    # A 2-layer step at 8 tokens in a CUDA graph, replayed with rewritten inputs, against eager calls without the push.
    x_static = _normal(_gen(6300), (dm.MAX_TOKENS, H), 0.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = [
            _outs(layer.forward(moe, x_static, None, partial_tail=j == 0, push=True))
            for j, layer in enumerate(layers[:2])
        ]
    for rep in range(3):
        x_static.copy_(_normal(_gen(6400 + rep), (dm.MAX_TOKENS, H), 0.5))
        graph.replay()
        pushes += 2
        torch.cuda.synchronize()
        got = [tuple(t.clone() for t in outs) for outs in captured]
        want = [call(layer, x_static, False, j == 0) for j, layer in enumerate(layers[:2])]
        torch.cuda.synchronize()
        assert all(_same_bits(g, w) for g, w in zip(got, want)), (
            rep,
            "graph replay != eager one-shot",
        )
    del graph, captured

    # Ring reuse: at every M, a 3-layer step captured once and replayed RING_REPLAYS times.
    for m in range(1, dm.MAX_TOKENS + 1):
        x_static = _normal(_gen(6500 + m), (m, H), 0.5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = [
                _outs(layer.forward(moe, x_static, None, partial_tail=j == 1, push=True))
                for j, layer in enumerate(layers)
            ]
        for rep in range(RING_REPLAYS):
            x_static.copy_(_normal(_gen(6600 + 64 * m + rep), (m, H), 0.5))
            dc.R.late(late.randrange(dc.R.world))
            graph.replay()
            pushes += PUSH_LAYERS
            torch.cuda.synchronize()
            got = [tuple(t.clone() for t in outs) for outs in captured]
            want = [call(layer, x_static, False, j == 1) for j, layer in enumerate(layers)]
            torch.cuda.synchronize()
            assert all(_same_bits(g, w) for g, w in zip(got, want)), (
                m,
                rep,
                "ring replay != eager one-shot",
            )
        del graph, captured

    # The exchange is empty, and its count advanced by one per pushing call since the warm-up.
    dc.R.barrier()
    assert bool((ex.uc == EXCHANGE_EMPTY).all().item()), "exchange words left after the last reduce"
    assert int(ex.flags[2].item()) == 0, "arrival word not cleared"
    assert int(ex.flags[0].item()) == count0 + pushes, (int(ex.flags[0].item()), count0, pushes)
    if dc.R.rank == 0:
        print(
            f"[rank 0] decode MoE push: {pushes} push + reduce calls, all equal to the one-shot",
            flush=True,
        )


CHECKS = [
    dc.check_takes_oproj,
    dc.check_decode_one_shot,
    dc.check_fused_vs_builtin,
    dc.check_sandwich_vs_mnnvl_bitwise,
    dc.check_graph_capture_and_replay,
    dc.check_moe_tail_deferral,
    dc.check_moe_tail_tap,
    dc.check_moe_tail_graph,
    check_decode_moe_push,
]


def _run_one_rank(args) -> int:
    dc.start_rank(args)
    import importlib

    modeling = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling"
    )
    return dc.run_matrix(
        modeling,
        CHECKS,
        steps=[name for name, _, _ in dc.ALL_STEPS],
        first=[check_decode_moe_from_model_parameters],
    )


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
