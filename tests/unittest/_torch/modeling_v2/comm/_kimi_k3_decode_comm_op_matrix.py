# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's decode-path collectives (``decode_comm.py`` of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``)
against its built-in path, at W ranks of one GB200 tray (default 4), at the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_decode_comm_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _kimi_k3_decode_comm_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of checks inside one W-rank job, sharing the target's collective state. The
collected entry point is ``test_modeling_v2_kimi_k3_decode_comm_op_matrix.py``.

Every rank builds the target's own decoder layers (KimiLinearDecoderLayer): KDA layer 12 (a snapshot layer, so the
post-attention step has no prefix sum), KDA layer 13 and MLA layer 15 with dense MLPs, and the MoE layers 22 (KDA),
23 (MLA) and 24 (a KDA snapshot layer). Their attention modules are stand-ins that keep the interface the layer calls:
``will_run_decode_branch`` (KDA: any step decode_step classifies; MLA: a decode step only),
``forward(..., reduce_output, project_output)``, a real bf16 o_proj [7168, 768] and the real stock AllReduce over
MNNVL; the core they hand over is set by the check. A dense layer's MLP is a recorder that returns zeros, so the layer
returns the post-attention ``updated`` and the recorder holds ``normed``. A MoE layer's MoE is a stand-in that hands
on a PendingTail of the TP16 per-rank shapes (latent [T, 3584], shared activation [T, 384], tail weight
[7168, 256 + 384]), or returns that tail reduced in torch and the stock all-reduce.

Checks:
  * decode MoE (first, outside inference mode with autograd on, as post_load_weights runs): the MoE decode path built
    from a MoE layer's own modules (the target's gate, whose parameters require grad, the latent projections, the
    stock RMSNorm and shared GatedMLP; zeroed routed experts) at this TP, every kernel warmed up, then a 3-token step
    and a wide 16-token step against the shared expert's forward.
  * takes_oproj: the TP16 o_proj shape only, bias-free, within the kernel's candidate count.
  * use_decode_one_shot: every stock MNNVL all-reduce takes the decode ceiling.
  * fused vs built-in: for each dense layer and step (decode of 1 / 3 / 8 tokens, DSpark 1 x 6, a 5-token prefill,
    12 / 16 unclassified tokens, a wide 2 x 8 step), the layer with the target's K3DecodeComm against the same layer
    without it: ``updated`` and ``normed`` within TOL, and the path the call took (the attention's reduce_output /
    project_output).
  * sandwich vs the MNNVL entry on exact payloads (products and sums exact in fp32): bit for bit.
  * graph: three layers chained, captured at an 8-token decode step (sandwich) and a 12-token unclassified step (MNNVL
    entry), replayed with rewritten inputs: each replay equal to an eager run, bit for bit.
  * MoE tail deferral: layer 22 hands its tail to layer 23 or 24, whose pre-attention step runs it as
    comm/k3_sandwich_tail (layer 24's kernel stores the prefix sum into the bank row it takes), against the same layers
    with the tail reduced in torch and added before the built-in pre-attention step: the consumer's attention input,
    its outputs and the bank within TOL; the deferred chain captured and replayed equals eager bit for bit.
  * Every output bitwise equal across the ranks.

Every rank draws the replicated tensors (prefix sum, snapshot bank, norms) from one seed and its own o_proj, core and
tail from a rank seed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the Kimi K3 target's decode collectives require CUDA devices"

DEADLINE_S = 1500
H = 7168
K_IN = 768
LATENT = 3584
LAT_SLICE = 224  # a TP16 rank's latent columns
TAIL_LAT = 256  # the tail weight's latent part, zero-padded
TAIL_ACT = 384  # a TP16 rank's shared activation columns
MOE_LAYERS = (22, 23, 24)  # KDA, MLA, KDA (a snapshot layer)
SNAPSHOTS = 2  # valid bank rows before a non-snapshot layer
BANK = 8
TOL = 2e-2  # normed and updated (fused vs built-in): max |err| / max |ref|
LAYERS = (12, 13, 15)

R = None
T = None  # the target module
STATS = {
    "bitwise_updated": 0,
    "bitwise_normed": 0,
    "cases": 0,
    "max_err_normed": 0.0,
    "max_err_updated": 0.0,
}


class _Attention(nn.Module):
    """Stand-in for the target's attention module (see the module docstring)."""

    def __init__(self, mapping, strategy, decode_only: bool):
        super().__init__()
        from tensorrt_llm._torch.distributed import AllReduce

        self.o_proj = nn.Linear(K_IN, H, bias=False, dtype=torch.bfloat16, device="cuda")
        self._o_allreduce = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
        self.decode_only = decode_only
        self.core = None
        self.calls = []

    def will_run_decode_branch(self, attn_metadata, step) -> bool:
        return step is not None and (step.decode or not self.decode_only)

    def forward(
        self, hidden_states, attn_metadata, step=None, reduce_output=True, project_output=True
    ):
        self.calls.append((reduce_output, project_output))
        self.last_input = hidden_states.clone()
        core = self.core
        assert core.shape[0] == hidden_states.shape[0], (core.shape, hidden_states.shape)
        if not project_output:
            if not self.will_run_decode_branch(attn_metadata, step):
                raise ValueError("project_output=False off the decode branch")
            return core
        out = self.o_proj(core)
        return self._o_allreduce(out) if reduce_output else out


class _KDA(_Attention):
    def __init__(
        self,
        cfg,
        layer_idx,
        mapping=None,
        allreduce_strategy=None,
        aux_stream=None,
        model_config=None,
    ):
        super().__init__(mapping, allreduce_strategy, decode_only=False)


class _MLARuntime(nn.Module):
    def __init__(self, cfg, layer_idx, model_config, aux_stream_dict):
        super().__init__()
        self.mixer = _Attention(
            model_config.mapping, model_config.allreduce_strategy, decode_only=True
        )
        self._o_allreduce = self.mixer._o_allreduce

    def will_run_decode_branch(self, attn_metadata, step) -> bool:
        return self.mixer.will_run_decode_branch(attn_metadata, step)

    def forward(
        self, hidden_states, attn_metadata, step=None, reduce_output=True, project_output=True
    ):
        return self.mixer(hidden_states, attn_metadata, step, reduce_output, project_output)


class _Moe(nn.Module):
    """Stand-in for KimiK3MoERuntime. With ``partial_tail`` it hands on ``pending`` (a PendingTail set by the check);
    otherwise it returns that tail reduced in torch and the stock all-reduce (zeros without one) and records its input.
    The reference tail: ``[RMSNorm(latent)[:, lo:lo + 224] | act] @ weight.T``, the RMS applied to the fp32 product
    of the latent slice, rounded to bf16 per rank, then summed over the ranks."""

    def __init__(self, model_config, cfg, layer_idx, aux_stream_dict):
        super().__init__()
        from tensorrt_llm._torch.distributed import AllReduce

        self.all_reduce = AllReduce(mapping=model_config.mapping, strategy=model_config.allreduce_strategy,
                                    dtype=torch.bfloat16)  # fmt: skip
        self.pending = None
        self.last_input = None

    def tail_rp_eligible(self, hidden_states, step) -> bool:
        return self.pending is not None and step is not None and hidden_states.shape[0] <= 8

    def forward(self, hidden_states, all_rank_num_tokens=None, partial_tail=False, step=None):
        self.last_input = hidden_states.clone()
        p = self.pending
        if partial_tail:
            return p
        if p is None:
            return torch.zeros_like(hidden_states)
        lat, w = p.latent.float(), p.weight.float()
        rs = torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + p.lat_eps)
        y = (lat[:, p.lo : p.lo + LAT_SLICE] @ w[:, :LAT_SLICE].t()) * rs + p.act.float() @ w[
            :, TAIL_LAT:
        ].t()
        return self.all_reduce(y.bfloat16())


def _attention(layer) -> _Attention:
    return layer.linear_attn if layer.is_kda else layer.self_attn.mixer


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _normal(g, shape, scale=1.0, offset=0.0):
    return (offset + scale * torch.randn(shape, generator=g, device="cuda")).bfloat16()


class _Recorder:
    """The layer's MLP: records its input (``normed``) and returns zeros, so the layer returns ``updated``."""

    def __init__(self):
        self.normed = None

    def __call__(self, hidden_states, step):
        self.normed = hidden_states.clone()
        return torch.zeros_like(hidden_states)


def build_layers():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.functional import AllReduceStrategy

    T.K3DecodeKDA = _KDA
    T.KimiMLARuntime = _MLARuntime
    T.KimiK3MoERuntime = _Moe
    cfg = SimpleNamespace(
        hidden_size=H,
        num_experts=None,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        intermediate_size=2112 * R.world,
        activation_situ_beta=1.0,
        activation_situ_linear_beta=8.0,
        rms_norm_eps=1e-5,
        attn_res_block_size=12,
        num_hidden_layers=93,
        linear_attn_config=dict(
            kda_layers=[i for i in range(1, 94) if i % 4],
            full_attn_layers=[i for i in range(1, 94) if i % 4 == 0],
        ),
    )
    model_config = ModelConfig(mapping=R.mapping, allreduce_strategy=AllReduceStrategy.MNNVL)
    stream = torch.cuda.Stream()
    streams = {kind: stream for kind in AuxStreamType}
    moe_cfg = SimpleNamespace(**{**vars(cfg), "num_experts": 896})
    layers = {}
    with torch.device("cuda"):
        for idx in LAYERS + MOE_LAYERS:
            layer = T.KimiLinearDecoderLayer(
                model_config, moe_cfg if idx in MOE_LAYERS else cfg, idx, streams
            )
            assert layer._mnnvl_allreduce() is not None, (
                f"layer {idx}: the attention all-reduce is not MNNVL"
            )
            assert layer.is_moe == (idx in MOE_LAYERS), idx
            if not layer.is_moe:
                layer.recorder = _Recorder()
                layer._dense_mlp = layer.recorder
            g = _gen(1000 + idx)  # replicated
            with torch.no_grad():
                for norm in (
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.self_attention_res_norm,
                    layer.mlp_res_norm,
                ):
                    norm.weight.copy_(_normal(g, norm.weight.shape, 0.1, 1.0))
                for proj in (layer.self_attention_res_proj, layer.mlp_res_proj):
                    proj.weight.copy_(_normal(g, proj.weight.shape, 0.05))
                # This rank's o_proj slice: exact payloads, so products and their sums are exact in fp32.
                gr = _gen(2000 + 100 * idx + R.rank)
                _attention(layer).o_proj.weight.copy_(ls.exact_bf16(gr, (H, K_IN), -4, 5, 1 / 16))
            layers[idx] = layer
    return layers


def make_comm(layers):
    takes = {
        idx: T._decode_comm.K3DecodeComm.takes_oproj(layer._o_proj(), BANK)
        for idx, layer in layers.items()
    }
    assert all(takes.values()), takes
    oproj = layers[LAYERS[0]]._o_proj().weight
    comm = T._decode_comm.K3DecodeComm.create(R.mapping, oproj)
    comm.compile_tail(
        LATENT, TAIL_ACT, torch.zeros(H, TAIL_LAT + TAIL_ACT, dtype=torch.bfloat16, device="cuda")
    )
    return comm


class Case:
    """One layer call's inputs: the prefix sum and bank (replicated) and this rank's core."""

    def __init__(self, seed, idx, tokens):
        g = _gen(seed)
        self.x = _normal(g, (tokens, H), 1.0)
        self.bank = _normal(g, (BANK, tokens, H), 1.0)
        self.snapshots = 0 if idx % 12 == 0 else SNAPSHOTS
        gr = _gen(seed * 31 + 7 + R.rank)
        self.core = ls.exact_bf16(gr, (tokens, K_IN), -4, 5, 1 / 8)


def run(layer, case, step, comm, sandwich=True):
    """``(updated, normed, calls)`` of one layer call on clones of the case's tensors."""
    attention = _attention(layer)
    attention.core = case.core
    attention.calls.clear()
    layer.decode_comm = comm
    layer.sandwich_oproj = sandwich
    bank = case.bank.clone()
    updated, _ = layer(case.x.clone(), bank, case.snapshots, SimpleNamespace(), step=step)
    torch.cuda.synchronize()
    return updated.clone(), layer.recorder.normed, list(attention.calls)


def _err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def STEPS():
    DS = T.DecodeStep
    return [
        ("decode1", 1, DS(1, 1, 1)),
        ("decode3", 3, DS(3, 3, 1)),
        ("dspark6", 6, DS(6, 1, 6)),
        ("decode8", 8, DS(8, 8, 1)),
        ("prefill5", 5, DS(5)),  # small, not a decode step
        ("unclassified12", 12, None),
        ("unclassified16", 16, None),
        ("wide16", 16, DS(16, 2, 8)),
    ]


def expected_call(layer, step):
    """(reduce_output, project_output) the layer should pass on ``step`` with the comm."""
    if step is not None and step.wide:
        return (
            False,
            True,
        )  # the partial, for the wide step's all-reduce ceiling (wide_all_reduce)
    if step is not None and step.num_tokens <= 8 and (step.decode or layer.is_kda):
        return (True, False)  # the sandwich: the core
    return (False, True)  # the MNNVL entry: the unreduced partial


def check_decode_one_shot():
    """use_decode_one_shot raises every stock MNNVL all-reduce's ceiling; the check then restores the stock one, so
    the built-in reference keeps main's choice."""
    mods = [m.mnnvl_allreduce for layer in LAYERS_BUILT.values() for m in layer.modules()
            if getattr(m, "mnnvl_allreduce", None) is not None]  # fmt: skip
    assert len(mods) >= 2 * len(LAYERS), len(mods)  # attention + dense MLP all-reduce per layer
    stock = {id(m): m.one_shot_max_bytes for m in mods}
    for layer in LAYERS_BUILT.values():
        T._decode_comm.use_decode_one_shot(layer)
    assert all(m.one_shot_max_bytes == T._decode_comm.DECODE_AR_ONE_SHOT_MAX_BYTES for m in mods)
    for m in mods:
        m.one_shot_max_bytes = stock[id(m)]


def check_takes_oproj():
    K3DecodeComm = T._decode_comm.K3DecodeComm
    good = nn.Linear(K_IN, H, bias=False, dtype=torch.bfloat16, device="cuda")
    wide = nn.Linear(1024, H, bias=False, dtype=torch.bfloat16, device="cuda")
    biased = nn.Linear(K_IN, H, bias=True, dtype=torch.bfloat16, device="cuda")
    fp32 = nn.Linear(K_IN, H, bias=False, dtype=torch.float32, device="cuda")
    assert K3DecodeComm.takes_oproj(good, BANK)
    assert not K3DecodeComm.takes_oproj(wide, BANK)
    assert not K3DecodeComm.takes_oproj(biased, BANK)
    assert not K3DecodeComm.takes_oproj(fp32, BANK)
    assert not K3DecodeComm.takes_oproj(good, 9), (
        "9 snapshots + the update exceed the kernel's 9 candidates"
    )


def check_fused_vs_builtin():
    seed = 1
    for idx in LAYERS:
        layer = LAYERS_BUILT[idx]
        for name, tokens, step in STEPS():
            seed += 1
            case = Case(seed, idx, tokens)
            ref_u, ref_n, ref_calls = run(layer, case, step, None)
            got_u, got_n, got_calls = run(layer, case, step, COMM)
            assert ref_calls == [(True, True)], (idx, name, ref_calls)
            assert got_calls == [expected_call(layer, step)], (idx, name, got_calls)
            eu, en = _err(got_u, ref_u), _err(got_n, ref_n)
            same_u, same_n = torch.equal(got_u, ref_u), torch.equal(got_n, ref_n)
            STATS["cases"] += 1
            STATS["bitwise_updated"] += same_u
            STATS["bitwise_normed"] += same_n
            STATS["max_err_updated"] = max(STATS["max_err_updated"], eu)
            STATS["max_err_normed"] = max(STATS["max_err_normed"], en)
            if R.rank == 0:
                print(
                    f"[rank 0] layer {idx} {name}: path {got_calls[0]} updated err {eu:.2e} bitwise {same_u}, "
                    f"normed err {en:.2e} bitwise {same_n}",
                    flush=True,
                )
            assert torch.isfinite(got_n.float()).all() and torch.isfinite(got_u.float()).all()
            assert eu < TOL and en < TOL, (idx, name, eu, en)
            assert R.same_on_ranks(got_u, got_n), (idx, name, "ranks differ")
            if step is not None and step.wide:
                # The same all-reduce (one-shot at W <= 4 under either ceiling) and the same epilogue.
                assert same_u and same_n, (idx, name, "a wide step keeps the built-in arithmetic")


def check_sandwich_vs_mnnvl_bitwise():
    seed = 500
    for idx in LAYERS:
        layer = LAYERS_BUILT[idx]
        for name, tokens, step in STEPS():
            if expected_call(layer, step) != (True, False):
                continue
            seed += 1
            case = Case(seed, idx, tokens)
            sw_u, sw_n, sw_calls = run(layer, case, step, COMM, sandwich=True)
            mn_u, mn_n, mn_calls = run(layer, case, step, COMM, sandwich=False)
            assert sw_calls == [(True, False)] and mn_calls == [(False, True)], (sw_calls, mn_calls)
            assert torch.equal(sw_u, mn_u), (idx, name, "updated", _err(sw_u, mn_u))
            assert torch.equal(sw_n, mn_n), (idx, name, "normed", _err(sw_n, mn_n))
            if R.rank == 0:
                print(f"[rank 0] layer {idx} {name}: sandwich == mnnvl bit for bit", flush=True)


def _chain(cases, step):
    """The three layers in order, each layer's input the previous one's output (the first takes cases[0].x)."""
    x = cases[0].x_buf
    outs = []
    for idx, case in zip(LAYERS, cases):
        layer = LAYERS_BUILT[idx]
        _attention(layer).core = case.core_buf
        layer.decode_comm = COMM
        layer.sandwich_oproj = True
        x, _ = layer(x, case.bank_buf, case.snapshots, SimpleNamespace(), step=step)
        outs.append((x, layer.recorder.normed))
    return outs


def check_graph_capture_and_replay():
    for name, tokens, step in (("decode8", 8, T.DecodeStep(8, 8, 1)), ("unclassified12", 12, None)):
        cases = [Case(900 + i, idx, tokens) for i, idx in enumerate(LAYERS)]
        for case in cases:
            case.x_buf, case.bank_buf, case.core_buf = (
                case.x.clone(),
                case.bank.clone(),
                case.core.clone(),
            )
        # Warm-up eagerly (the stock all-reduce sizes its workspace on its first call of a size), then capture.
        _chain(cases, step)
        torch.cuda.synchronize()
        R.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = _chain(cases, step)
        for replay in range(3):
            fresh = [Case(950 + 10 * replay + i, idx, tokens) for i, idx in enumerate(LAYERS)]
            for case, new in zip(cases, fresh):
                case.x_buf.copy_(new.x)
                case.bank_buf.copy_(new.bank)
                case.core_buf.copy_(new.core)
            graph.replay()
            torch.cuda.synchronize()
            got = [(u.clone(), n.clone()) for u, n in outs]
            # The eager reference on the same inputs (fresh buffers: the snapshot layer writes its bank row).
            for case, new in zip(cases, fresh):
                case.x_buf.copy_(new.x)
                case.bank_buf.copy_(new.bank)
                case.core_buf.copy_(new.core)
            want = [(u.clone(), n.clone()) for u, n in _chain(cases, step)]
            torch.cuda.synchronize()
            for (gu, gn), (wu, wn) in zip(got, want):
                assert torch.equal(gu, wu) and torch.equal(gn, wn), (name, replay)
            assert R.same_on_ranks(*[t for pair in got for t in pair]), (
                name,
                replay,
                "ranks differ",
            )
        if R.rank == 0:
            print(f"[rank 0] graph {name}: 3 replays == eager bit for bit", flush=True)
        del graph


def _pending(seed, tokens):
    """A PendingTail of the TP16 per-rank shapes: the reduced latent replicated, this rank's shared activation and
    tail weight (its latent padding columns zero), this rank's first latent column."""
    g = _gen(seed)
    latent = _normal(g, (tokens, LATENT), 1.0)
    gr = _gen(seed * 17 + 3 + R.rank)
    act = _normal(gr, (tokens, TAIL_ACT), 0.5)
    weight = _normal(gr, (H, TAIL_LAT + TAIL_ACT), 0.03)
    weight[:, LAT_SLICE:TAIL_LAT] = 0
    return T._decode_comm.PendingTail(latent, act, weight.contiguous(), R.rank * LAT_SLICE, 1e-5)


def _tail_chain(producer, consumer, x, bank, step, pending, defer, consumer_core, producer_core):
    """The producer layer (its MoE handing ``pending`` on with ``defer``, else adding its reduced tail), then the
    consumer layer. Returns the consumer's attention input, its returned prefix sum, its MoE input and the bank."""
    _attention(producer).core = producer_core
    _attention(consumer).core = consumer_core
    producer.block_sparse_moe.pending = pending
    consumer.block_sparse_moe.pending = None
    for layer in (producer, consumer):
        layer.decode_comm = COMM
        layer.sandwich_oproj = True
    snapshots = SNAPSHOTS
    out = producer(x, bank, snapshots, SimpleNamespace(), step=step, defer_moe_tail=defer)
    if defer:
        prefix, snapshots, partial = out
        assert partial is pending
    else:
        (prefix, snapshots), partial = out, None
    prefix_c, snapshots_c = consumer(
        prefix, bank, snapshots, SimpleNamespace(), step=step, pending_moe_partial=partial
    )
    return (
        _attention(consumer).last_input,
        prefix_c,
        consumer.block_sparse_moe.last_input,
        bank[:snapshots_c],
    )


def check_moe_tail_deferral():
    seed = 3000
    for consumer_idx in (23, 24):
        producer, consumer = LAYERS_BUILT[22], LAYERS_BUILT[consumer_idx]
        for tokens in (1, 3, 8):
            seed += 1
            step = T.DecodeStep(tokens, tokens, 1)
            case = Case(seed, 22, tokens)
            gr = _gen(seed * 13 + 5 + R.rank)
            core_c = ls.exact_bf16(gr, (tokens, K_IN), -4, 5, 1 / 8)
            pending = _pending(seed, tokens)
            got = _tail_chain(producer, consumer, case.x.clone(), case.bank.clone(), step, pending, True, core_c,
                              case.core)  # fmt: skip
            got = [t.clone() for t in got]
            want = _tail_chain(producer, consumer, case.x.clone(), case.bank.clone(), step, pending, False, core_c,
                               case.core)  # fmt: skip
            torch.cuda.synchronize()
            names = ("attention input", "prefix sum", "MoE input", "bank")
            errs = [_err(a, b) for a, b in zip(got, want)]
            if R.rank == 0:
                print(
                    f"[rank 0] tail {22}->{consumer_idx} T={tokens}: "
                    + ", ".join(f"{n} {e:.2e}" for n, e in zip(names, errs)),
                    flush=True,
                )
            assert all(torch.isfinite(t.float()).all() for t in got)
            assert all(e < TOL for e in errs), (consumer_idx, tokens, dict(zip(names, errs)))
            assert R.same_on_ranks(*got), (consumer_idx, tokens, "ranks differ")


def check_moe_tail_graph():
    producer, consumer = LAYERS_BUILT[22], LAYERS_BUILT[24]
    tokens = 8
    step = T.DecodeStep(tokens, tokens, 1)
    case = Case(4000, 22, tokens)
    x, bank = case.x.clone(), case.bank.clone()
    core_p, core_c = case.core.clone(), case.core.clone()
    pending = _pending(4000, tokens)

    def chain():
        return _tail_chain(producer, consumer, x, bank, step, pending, True, core_c, core_p)

    chain()  # eager warm-up
    torch.cuda.synchronize()
    R.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    for replay in range(3):
        fresh_case = Case(4100 + replay, 22, tokens)
        fresh = _pending(4100 + replay, tokens)

        def load():
            x.copy_(fresh_case.x)
            bank.copy_(fresh_case.bank)
            core_p.copy_(fresh_case.core)
            core_c.copy_(fresh_case.core.flip(0))
            for a, b in zip(pending[:3], fresh[:3]):
                a.copy_(b)

        load()
        graph.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in outs]
        load()
        want = [t.clone() for t in chain()]
        torch.cuda.synchronize()
        for a, b in zip(got, want):
            assert torch.equal(a, b), replay
        assert R.same_on_ranks(*got), (replay, "ranks differ")
    if R.rank == 0:
        print("[rank 0] graph tail 22->24 T=8: 3 replays == eager bit for bit", flush=True)
    del graph


# The MoE decode path at this check's TP: a moetp4ep4 rank's routed experts (intermediate columns, experts), and a
# shared expert whose rank slice at TP 4 is a TP16 rank's (384 activation columns).
I_TP, E_LOCAL, NUM_EXPERTS = 768, 224, 896
SHARED = 384 * 4
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
        gate = T.KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16)
        down = nn.Linear(H, LATENT, bias=False, dtype=torch.bfloat16)
        up = nn.Linear(LATENT, H, bias=False, dtype=torch.bfloat16)
        norm = RMSNorm(hidden_size=LATENT, eps=1e-5, dtype=torch.bfloat16)
        shared = GatedMLP(
            hidden_size=H,
            intermediate_size=SHARED,
            bias=False,
            activation=SituAndMul(
                beta=SITU_CAPS[0], linear_beta=SITU_CAPS[1], use_fused_activation=True
            ),
            dtype=torch.bfloat16,
            config=ModelConfig(mapping=R.mapping, allreduce_strategy=AllReduceStrategy.MNNVL),
            reduce_output=True,
            layer_idx=MOE_LAYERS[0],
            is_shared_expert=True,
        )
    g = _gen(5000)  # replicated
    gr = _gen(5100 + R.rank)  # this rank's shared expert slices
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
        slot_start=R.rank * E_LOCAL % NUM_EXPERTS,
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
    if R.world not in (4, 8, 16):
        if R.rank == 0:
            print(
                f"[rank 0] decode MoE skipped at world {R.world}: the front runs 4, 8 or 16 ranks",
                flush=True,
            )
        return
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    dm = T._decode_moe
    moe = _decode_moe_layer()
    assert moe.gate.e_score_correction_bias.requires_grad
    mnnvl = T._decode_comm.MnnvlWorkspace.create(R.mapping, T._decode_comm.MNNVL_BUFFER_BYTES)
    device = torch.device("cuda", torch.cuda.current_device())
    state = dm.K3DecodeMoe.create(R.mapping, device, I_TP, E_LOCAL, mnnvl)
    dm.fold_latent_norm(moe)
    layer = dm.K3DecodeMoeLayer.create(moe, state, R.rank, R.world)
    layer.warm_up(moe)
    for rows in (3, 16):
        x = _normal(_gen(5200 + rows), (rows, H), 0.5)
        y = layer.forward(moe, x, None, partial_tail=False)
        if rows > dm.MAX_TOKENS:  # a wide step returns this rank's share
            y = moe.routed_experts.all_reduce(y)
        want = moe.shared_experts(x)
        torch.cuda.synchronize()
        err = ls.rel_err(y, want)
        if R.rank == 0:
            print(f"[rank 0] decode MoE, {rows} tokens: vs the shared expert {err:.2e}", flush=True)
        assert torch.isfinite(y.float()).all() and err <= MOE_TOL, (rows, err)
        assert R.same_on_ranks(y), (rows, "ranks differ")


CHECKS = [
    check_takes_oproj,
    check_decode_one_shot,
    check_fused_vs_builtin,
    check_sandwich_vs_mnnvl_bitwise,
    check_graph_capture_and_replay,
    check_moe_tail_deferral,
    check_moe_tail_graph,
]

LAYERS_BUILT = None
COMM = None


def _run_one_rank(args) -> int:
    global R, T, LAYERS_BUILT, COMM
    R = ls.Rank(args)
    import importlib

    T = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling"
    )
    # First, before anything exists as an inference tensor: the decode MoE as post_load_weights builds it.
    code = ls.run_checks(R, [check_decode_moe_from_model_parameters])
    with torch.inference_mode():
        LAYERS_BUILT = build_layers()
        COMM = make_comm(LAYERS_BUILT)
        code = ls.run_checks(R, CHECKS) or code
    if R.rank == 0:
        print(f"[rank 0] world {R.world}; {STATS}", flush=True)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
