# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE decode path of the Kimi K3 target ``kimi_k3_mxfp4__sm_100__tp16_moetp16ep1`` (``decode_moe.py``: every
expert on each rank, the routed experts pushing into the latent exchange) at W ranks of one GB200 tray (default 4),
at the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_route_b_decode_moe_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _kimi_k3_route_b_decode_moe_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of checks inside one W-rank job, sharing the decode path's collective state.
The collected entry point is ``test_modeling_v2_kimi_k3_route_b_decode_moe_op_matrix.py``.

Every rank builds MoE layers the way the target's ``post_load_weights`` does (``K3DecodeMoe.create``,
``fold_latent_norm``, ``K3DecodeMoeLayer.create``, ``warm_up``) from stand-in layers holding what the path reads: the
target's KimiK3MoEGate, nn.Linear latent projections, the stock RMSNorm and shared GatedMLP at this TP, and the routed
experts of a TP16 rank (all 896, the 192-wide intermediate slice zero-padded to 256 by TRT-LLM's loader, random
MXFP4). The reference of a step is the same front with the returned-partial ``k3_moe`` of its ``K3MoeState`` and the
routed experts' MNNVL all-reduce.

Checks:
  * build: outside inference mode with autograd on, as post_load_weights runs (the gate's parameters require grad).
  * engines (eager steps, which return the partials to the all-reduce): for 1..8 tokens, the latent the path hands on
    (its PendingTail) against the reference: bit for bit at 3..8 tokens, within ENGINE_TOL at 1 and 2 (k3_moe_m1 /
    k3_moe_m2 against k3_moe, which round the FC1 sums apart); the engine each count ran on (the engines' epochs);
    the replicated output against the reference tail on the same latent; the routing against the noaux_tc arithmetic
    in torch (the same top-16 experts, the weights within ROUTE_TOL).
  * push: three layers on one state over steps of 1, 2, 8, 3, 1, 5, 2 and 7 tokens, each engine's push and
    k3_latent_reduce against the same engine's returned partial and the all-reduce, bit for bit; the exchange's
    halves alternate across every engine and layer.
  * graph: three layers of 2, 8 and 1 tokens captured once pushing (``push=True``, as a pushing step's
    ``DecodeStep.latent_push`` has the MoE runtime pass it) and replayed
    with rewritten inputs: each replay equal to eager steps (which do not push), bit for bit.
  * Every output bitwise equal across the ranks.
"""

import functools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the Kimi K3 target's MoE decode path requires CUDA devices"

DEADLINE_S = 1500
H = 7168
LATENT = 3584
NUM_EXPERTS = 896
# A TP16 rank's 192-wide intermediate slice, zero-padded to whole tiles (256) by the loader.
I_TP, I_PAD, MOE_TP = 192, 256, 16
SV = 32
SHARED_PER_RANK = 384
SITU_CAPS = (4.0, 25.0)
ENGINE_TOL = 8e-3  # max |err| / max |ref| of the latent, k3_moe_m1 / k3_moe_m2 against k3_moe
ROUTE_TOL = 2e-2  # the bf16 routing weights against the fp32 torch routing
SEQUENCE = (1, 2, 8, 3, 1, 5, 2, 7)
GRAPH_ROWS = (2, 8, 1)

R = None
T = None  # the target module
DM = None  # its decode_moe module


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0, offset=0.0):
    return (offset + scale * torch.randn(shape, generator=g, device="cuda")).bfloat16()


def _rand_mxfp4(rows, k, k_full, gen):
    """Random checkpoint-format MXFP4: packed [rows, k / 2] (low nibble = even k), E8M0 per 32 k, scaled so a
    k_full-long dot product lands near std 3."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k_full))
    exps = torch.randint(
        base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


@functools.lru_cache(maxsize=None)
def _experts(seed: int):
    """A TP16 rank's 896 experts through W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod's loader (the buffers the engines read):
    the 192-wide shard generated as rank 0 of tensors that hold exactly it, sliced and padded to 256 by the loader."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=MOE_TP, tp_rank=0, scaling_vector_size=SV, intermediate_size=I_TP * MOE_TP,
                             intermediate_size_per_partition=I_TP, hidden_size=LATENT)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    w31 = torch.zeros(NUM_EXPERTS, 2 * I_PAD, LATENT // 2, **kw)
    w31s = torch.zeros(NUM_EXPERTS, 2 * I_PAD, LATENT // SV, **kw)
    w2 = torch.zeros(NUM_EXPERTS, LATENT, I_PAD // 2, **kw)
    w2s = torch.zeros(NUM_EXPERTS, LATENT, I_PAD // SV, **kw)
    gen = _gen(seed)
    for e in range(NUM_EXPERTS):
        gate, gate_s = _rand_mxfp4(I_TP, LATENT, LATENT, gen)
        up, up_s = _rand_mxfp4(I_TP, LATENT, LATENT, gen)
        down, down_s = _rand_mxfp4(LATENT, I_TP, I_TP * MOE_TP, gen)
        method.load_expert_w3_w1_weight(module, gate, up, w31[e])
        method.load_expert_w2_weight(module, down, w2[e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, gate_s, up_s, w31s[e])
        method.load_expert_w2_weight_scale_mxfp4(module, down_s, w2s[e])
    torch.cuda.synchronize()
    return w31, w31s, w2, w2s


def _moe_layer(seed: int, experts):
    """A MoE layer with what the decode path reads, from the modules KimiK3MoERuntime builds (the gate's and the latent
    projections' parameters require grad, as in the model), and ``experts`` (every expert on this rank)."""
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
        gate = T.KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16)
        down = nn.Linear(H, LATENT, bias=False, dtype=torch.bfloat16)
        up = nn.Linear(LATENT, H, bias=False, dtype=torch.bfloat16)
        norm = RMSNorm(hidden_size=LATENT, eps=1e-5, dtype=torch.bfloat16)
        shared = GatedMLP(
            hidden_size=H,
            intermediate_size=SHARED_PER_RANK * R.world,
            bias=False,
            activation=SituAndMul(
                beta=SITU_CAPS[0], linear_beta=SITU_CAPS[1], use_fused_activation=True
            ),
            dtype=torch.bfloat16,
            config=ModelConfig(mapping=R.mapping, allreduce_strategy=AllReduceStrategy.MNNVL),
            reduce_output=True,
            layer_idx=seed,
            is_shared_expert=True,
        )
    g = _gen(5000 + seed)  # replicated
    gr = _gen(5100 + seed * 16 + R.rank)  # this rank's shared expert slices
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
    backend = SimpleNamespace(
        w3_w1_weight=experts[0],
        w3_w1_weight_scale=experts[1],
        w2_weight=experts[2],
        w2_weight_scale=experts[3],
        expert_size_per_partition=NUM_EXPERTS,
        intermediate_size_per_partition=I_TP,
        quant_method=SimpleNamespace(intermediate_size_per_partition_lean=I_TP),
        slot_start=0,
    )
    all_reduce = AllReduce(
        mapping=R.mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
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
        _reduce_routed_output=True,
        moe_main_event=torch.cuda.Event(),
        moe_shared_event=torch.cuda.Event(),
        shared_expert_stream=torch.cuda.Stream(),
    )


BUILT = None


def check_build():
    """The decode path built as post_load_weights builds it, outside inference mode with autograd on: the shared state
    (collective: the head workspace, the latent exchange; the engines' push builds compile), three MoE layers' folded
    latent norms and decode weights, and one call of every kernel (warm_up: the front, each engine's push, the
    reduce)."""
    global BUILT
    if R.world not in (4, 8, 16):
        raise AssertionError(
            f"the front and the latent exchange run 4, 8 or 16 ranks, not {R.world}"
        )
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    experts = _experts(7)
    moes = [_moe_layer(seed, experts) for seed in (22, 23, 24)]
    assert moes[0].gate.e_score_correction_bias.requires_grad
    mnnvl = T._decode_comm.MnnvlWorkspace.create(R.mapping, T._decode_comm.MNNVL_BUFFER_BYTES)
    device = torch.device("cuda", torch.cuda.current_device())
    backend = moes[0].routed_experts.backend
    state = DM.K3DecodeMoe.create(
        R.mapping,
        device,
        backend.w3_w1_weight.shape[1] // 2,
        backend.expert_size_per_partition,
        mnnvl,
        i_logical=backend.quant_method.intermediate_size_per_partition_lean,
    )
    assert state.wide is None and state.exchange is not None
    assert state.m1.push_compiled(R.world) and state.m2.push_compiled(R.world)
    layers = []
    for moe in moes:
        DM.fold_latent_norm(moe)
        layers.append(DM.K3DecodeMoeLayer.create(moe, state, R.rank, R.world))
    layers[0].warm_up(moes[0])
    BUILT = SimpleNamespace(state=state, moes=moes, layers=layers)
    if R.rank == 0:
        print(f"[rank 0] built 3 MoE layers on 896 local experts at world {R.world}", flush=True)


def _x(seed, rows):
    """The MoE input of a step: the same rows on every rank, as the decode path receives them."""
    return _normal(_gen(9000 + seed), (rows, H), 0.5)


def _reference(layer, moe, x):
    """The front, the returned-partial k3_moe of the layer's K3MoeState, the routed experts' MNNVL all-reduce:
    (latent [rows, 3584], shared activation, routing ids, routing weights)."""
    ids, weights, x_fp8, x_sf, shared_act = layer._front(moe, x)
    routed = DM.k3_moe(x_fp8, x_sf, ids, weights, 0, layer.small)
    return moe.routed_experts.all_reduce(routed), shared_act, ids, weights


def _wired(layer, moe, x, push=False):
    pending = layer.forward(moe, x, None, partial_tail=True, push=push)
    assert isinstance(pending, T._decode_comm.PendingTail), type(pending)
    return pending


def _engine_epochs():
    st = BUILT.state
    return st.m1.epochs.clone(), st.m2.epochs.clone()


def _routing_ref(moe, x):
    """The noaux_tc arithmetic in fp32 torch: sigmoid scores, the top 16 of scores + bias, the weights renormalized and
    scaled."""
    logits = x.float() @ moe.gate.weight.float().t()
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + moe.gate.e_score_correction_bias.float(), 16, dim=-1).indices
    picked = scores.gather(1, ids)
    weights = picked / picked.sum(-1, keepdim=True) * moe.gate.routed_scaling_factor
    return ids, weights


def _compare(name, got, want, exact, tol=ENGINE_TOL):
    err = ls.rel_err(got, want)
    same = bool(
        torch.equal(got.contiguous().view(torch.int16), want.contiguous().view(torch.int16))
    )
    ok = same if exact else err <= tol
    assert torch.isfinite(got.float()).all(), name
    assert ok, f"{name}: bitwise {same}, rel err {err:.3e}"
    assert R.same_on_ranks(got), f"{name}: ranks differ"
    return same, err


def check_engines():
    """Every token count 1..8 on layer 0: the latent against the reference (bit for bit from 3 tokens), the engine it
    ran on, the replicated output against the reference tail on the same latent, and the routing."""
    layer, moe = BUILT.layers[0], BUILT.moes[0]
    for rows in range(1, DM.MAX_TOKENS + 1):
        x = _x(rows, rows)
        m1_before, m2_before = _engine_epochs()
        pending = _wired(layer, moe, x)
        m1_after, m2_after = _engine_epochs()
        ran = ("k3_moe_m1" if not torch.equal(m1_before, m1_after) else "") + (
            "k3_moe_m2" if not torch.equal(m2_before, m2_after) else ""
        )
        want_engine = {1: "k3_moe_m1", 2: "k3_moe_m2"}.get(rows, "")
        assert ran == want_engine, (rows, ran, want_engine)
        latent, shared_act, ids, weights = _reference(layer, moe, x)
        same, err = _compare(f"latent {rows}", pending.latent, latent, exact=rows > 2)
        assert torch.equal(pending.act, shared_act), rows
        # The replicated tail: the latent RMS on the folded up projection's fp32 output, plus the shared expert.
        y = layer.forward(moe, x, None, partial_tail=False)
        shared = moe.shared_experts.down_proj(shared_act, layer_idx=moe.shared_experts.layer_idx)
        up = DM._gemv(None, "moe_up", latent, moe.routed_expert_up_proj.weight, out_fp32=True)
        scale = torch.rsqrt(latent.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        y_ref = (up * scale + shared.float()).bfloat16()
        _compare(f"output {rows}", y, y_ref, exact=rows > 2)
        # The front's routing against the noaux_tc arithmetic in torch.
        ref_ids, ref_weights = _routing_ref(moe, x)
        order = torch.argsort(ids.long(), dim=-1)
        ref_order = torch.argsort(ref_ids, dim=-1)
        assert torch.equal(ids.long().gather(1, order), ref_ids.gather(1, ref_order)), (
            rows,
            "routing ids",
        )
        route_err = ls.rel_err(weights.float().gather(1, order), ref_weights.gather(1, ref_order))
        assert route_err <= ROUTE_TOL, (rows, route_err)
        torch.cuda.synchronize()
        if R.rank == 0:
            print(f"[rank 0] {rows} tokens on {ran or 'k3_moe'}: latent bitwise {same} (rel err {err:.2e}), "
                  f"routing weights {route_err:.2e}", flush=True)  # fmt: skip


def check_push():
    """Three layers on one state over steps of SEQUENCE tokens: each layer's engine pushing into the latent exchange,
    then k3_latent_reduce, against the same engine's returned partial and the routed experts' all-reduce, bit for
    bit."""
    for step, rows in enumerate(SEQUENCE):
        for i, (layer, moe) in enumerate(zip(BUILT.layers, BUILT.moes)):
            x = _x(100 + 10 * step + i, rows)
            ids, weights, x_fp8, x_sf, _ = layer._front(moe, x)
            layer._routed(x_fp8, x_sf, ids, weights, 0, push=True)
            pushed = DM.k3_latent_reduce(rows, BUILT.state.exchange)
            returned = moe.routed_experts.all_reduce(layer._routed(x_fp8, x_sf, ids, weights, 0))
            _compare(f"push step {step} layer {i} ({rows} tokens)", pushed, returned, exact=True)
    torch.cuda.synchronize()
    if R.rank == 0:
        print(
            f"[rank 0] push: {len(SEQUENCE)} steps x 3 layers equal to the all-reduce", flush=True
        )


def check_graph():
    """Layers 0, 1, 2 at GRAPH_ROWS tokens captured in one CUDA graph pushing (the engines push and k3_latent_reduce
    sums, as on a pushing step), replayed three times with rewritten inputs: each replay's latents and shared
    activations equal eager steps on the same inputs (the returned partials and the all-reduce), bit for bit."""
    xs = [_x(300 + i, rows) for i, rows in enumerate(GRAPH_ROWS)]
    chain = list(zip(BUILT.layers, BUILT.moes, xs))
    for layer, moe, x in chain:  # eager: no push
        _wired(layer, moe, x)
    torch.cuda.synchronize()
    R.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [_wired(layer, moe, x, push=True) for layer, moe, x in chain]
    for replay in range(3):
        for i, x in enumerate(xs):
            x.copy_(_x(400 + 10 * replay + i, x.shape[0]))
        R.barrier()
        graph.replay()
        torch.cuda.synchronize()
        captured = [(p.latent.clone(), p.act.clone()) for p in outs]
        R.barrier()
        eager = [_wired(layer, moe, x) for layer, moe, x in chain]
        torch.cuda.synchronize()
        for i, ((latent, act), e) in enumerate(zip(captured, eager)):
            _compare(f"graph replay {replay} layer {i}", latent, e.latent, exact=True)
            # This rank's shared expert slice: its own activation, not one equal across the ranks.
            assert torch.equal(act.view(torch.int16), e.act.view(torch.int16)), (replay, i)
    if R.rank == 0:
        print("[rank 0] graph: 3 replays (pushing) equal to eager steps", flush=True)


def _run_one_rank(args) -> int:
    global R, T, DM
    R = ls.Rank(args)
    import importlib

    T = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.modeling"
    )
    DM = T._decode_moe
    # First, before anything exists as an inference tensor: the decode MoE as post_load_weights builds it.
    code = ls.run_checks(R, [check_build])
    if code:
        return code
    with torch.inference_mode():
        code = ls.run_checks(R, [check_engines, check_push, check_graph])
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
