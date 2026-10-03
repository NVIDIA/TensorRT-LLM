# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The push builds of the Kimi K3 decode routed experts: each rank's routed partial stored into a latent exchange for
trtllm::k3_latent_reduce instead of returned. One process per GPU over the TP group of this run (4 on one GB200 tray).
Every rank holds its own TP16 experts (all 896, the rank's 192-wide intermediate slice zero-padded to 256 by TRT-LLM's
loader; random checkpoint-format MXFP4, a seed per rank) and runs the same tokens.

push         trtllm::k3_moe_m1 at 1 and 2 tokens and trtllm::k3_moe_m2 at 2 (Layer.push), on tokens routed by
             trtllm::k3_route_quant;
k3_moe_push  trtllm::k3_moe's push build at 1, 3 and 8 tokens, after trtllm::k3_route_quant and after
             trtllm::k3_moe_front: K3MoeLayer.push into the run's exchange, the moe/k3_moe entry's k3_moe_push into
             the 16-slot one. The front's head and shared experts are sharded over the run's ranks; its shared
             activation must be the same bits in every call of a set.
k3_moe_push_route_a
             the same on route A's experts (tp16_moetp4ep4): a 768-wide intermediate slice of 224 local experts at
             offset (rank % 4) x 224, so the ranks of one tray hold 4 different expert sets.

Per op, token count and routing set, against the plain call (Layer.__call__, K3MoeLayer.__call__ after the same
producer), whose partial must be nonzero:
  exact    the partial pushed into the group's exchange (one slot per rank): the reduce equals MNNVLAllReduce's
           one-shot of the plain partials bit for bit, on every rank and run to run (two more push + reduce pairs);
           afterwards the exchange is empty, its call count is +1 and the arrival word 0;
  exact16  the same into a 16-slot exchange, every rank's partial in 4 slots (TP16's receive side on one tray): the
           reduce equals the one-shot's order over 16 slots (fp32 sums of 8 slots, added in order, then bf16);
  wrap     all of it again with both exchanges' call counts starting at 2^31 - 2, so the int32 count wraps to -2^31
           during the sets and the halves keep alternating.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_moe_push.py [push k3_moe_push]
"""

import functools
import hashlib
import math
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the test is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _trtllm():
    """``torch.ops.trtllm``, named only in a nested function: the pool gets this module's functions by value, and
    cloudpickle cannot pickle a function whose own code names ``torch.ops`` (it adds ``sys.modules["torch.ops"]`` to
    the function's state)."""

    def namespace():
        return torch.ops.trtllm

    return namespace()


WORLD = 4
H, NUM_EXPERTS, SV = 3584, 896, 32
# TP16: a rank's 192-wide intermediate slice, zero-padded to whole tiles (256) by the loader.
I_TP, I_PAD, MOE_TP = 192, 256, 16
# Route A (tp16_moetp4ep4): a rank's 768-wide slice (whole tiles) of 224 local experts.
I_TP_A, E_LOCAL_A, MOE_TP_A = 768, 224, 4
HIDDEN, SHARED_INTER = 7168, 6144  # the front's input width; two shared experts of 3072
GATE_CAP, LINEAR_CAP = 4.0, 25.0  # the SiTU caps
RSF = 2.827
SETS = 4
COUNT_STARTS = (0, 2**31 - 2)
EMPTY_WORD = -(2**31)
ENGINES = (("k3_moe_m1", 1), ("k3_moe_m1", 2), ("k3_moe_m2", 2))  # (engine, tokens per call)
# What routes and quantizes K3MoeLayer.push's tokens.
K3_MOE_PRODUCERS = ("k3_route_quant", "k3_moe_front")
K3_MOE_TOKENS = (1, 3, 8)
K3_MOE_SETS = 2


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return torch.cuda.get_device_capability() == (10, 0)


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _supported(), reason=f"needs {WORLD} SM100 GPUs with MNNVL and mpi4py"),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(_bits(t).cpu().numpy().tobytes()).hexdigest()


def _rand_mxfp4(rows, k, k_full, gen):
    """Random checkpoint-format MXFP4: packed [rows, k / 2] (low nibble = even k), E8M0 per 32 k, scaled so a
    k_full-long dot product lands near std 3."""
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k_full))
    exps = torch.randint(
        base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


# _experts' buffers per seed and layout. A dict, not functools.lru_cache: the pool's workers would get an lru_cache
# wrapper by reference, from a module they cannot import.
_EXPERTS = {}


def _experts(seed: int, route_a: bool = False):
    """This rank's experts through W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod's loader: the buffers the engines read, the
    rank's intermediate slice generated alone. TP16: all 896 experts, the 192-wide slice as rank 0 of tensors that
    hold exactly it (the loader slices it, then pads it to 256); route A: 224 experts, the 768-wide slice whole tiles,
    which the loader takes as one shard. Built once per seed and layout."""
    if (seed, route_a) in _EXPERTS:
        return _EXPERTS[seed, route_a]
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    if route_a:
        i_tp, i_pad, shards, moe_tp, experts = I_TP_A, I_TP_A, 1, MOE_TP_A, E_LOCAL_A
    else:
        i_tp, i_pad, shards, moe_tp, experts = I_TP, I_PAD, MOE_TP, MOE_TP, NUM_EXPERTS
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=shards, tp_rank=0, scaling_vector_size=SV, intermediate_size=i_tp * shards,
                             intermediate_size_per_partition=i_tp, hidden_size=H)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    w31 = torch.zeros(experts, 2 * i_pad, H // 2, **kw)
    w31s = torch.zeros(experts, 2 * i_pad, H // SV, **kw)
    w2 = torch.zeros(experts, H, i_pad // 2, **kw)
    w2s = torch.zeros(experts, H, i_pad // SV, **kw)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for e in range(experts):
        gate, gate_s = _rand_mxfp4(i_tp, H, H, gen)
        up, up_s = _rand_mxfp4(i_tp, H, H, gen)
        down, down_s = _rand_mxfp4(H, i_tp, i_tp * moe_tp, gen)
        method.load_expert_w3_w1_weight(module, gate, up, w31[e])
        method.load_expert_w2_weight(module, down, w2[e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, gate_s, up_s, w31s[e])
        method.load_expert_w2_weight_scale_mxfp4(module, down_s, w2s[e])
    torch.cuda.synchronize()
    _EXPERTS[seed, route_a] = w31, w31s, w2, w2s
    return _EXPERTS[seed, route_a]


def _tokens(m: int, seed: int):
    """The same tokens on every rank: hidden rows, fp32 router logits and the routing bias, quantized and routed by
    trtllm::k3_route_quant as the engines' callers do."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, generator=gen, device="cuda").bfloat16()
    logits = torch.randn(m, NUM_EXPERTS, generator=gen, device="cuda")
    bias = (torch.randn(NUM_EXPERTS, generator=gen, device="cuda") * 0.05).float()
    ids, weights, x_fp8, x_sf = _trtllm().k3_route_quant(logits, bias, x, RSF, True)
    return x_fp8, x_sf, ids, weights


class _Exchange:
    """A latent exchange over this run's ranks with ``slots`` slots: K3LatentExchange's buffers (int32 words
    [2][8][slots][1792] behind one multicast mapping, every word 0x80000000; int32 flags[4]) at any slot count, its
    call count starting at ``count``."""

    def __init__(self, ctx, slots: int, count: int):
        from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import k3_latent_reduce as kernel
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
        )

        words = kernel.buffer_words(slots)
        fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or ctx.mapping.is_multi_node()
        self.handle = _make_mnnvl_mcast_buffer(
            _get_mnnvl_workspace_comm(ctx.mapping), words * 4, ctx.mapping, fabric
        )
        self.uc = self.handle.get_uc_buffer(ctx.rank, (words,), torch.int32, 0)
        self.mc = self.handle.get_mc_buffer((words,), torch.int32, 0)
        self.uc.fill_(EMPTY_WORD)
        self.flags = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.flags[0] = count
        self.slots, self.count, self.rank = slots, count, ctx.rank
        torch.cuda.synchronize()
        ctx.comm.Barrier()

    def reduce(self, m: int) -> torch.Tensor:
        out = _trtllm().k3_latent_reduce(self.uc, self.flags, m, 0)
        self.count = (self.count + 1 + 2**31) % 2**32 - 2**31  # int32 two's complement
        return out

    def state_ok(self, ctx) -> bool:
        """Every rank's last reduce done and nothing of the next call pushed yet: the whole buffer is empty, the call
        count advanced, the arrival word cleared."""
        torch.cuda.synchronize()
        ctx.comm.Barrier()
        flags = self.flags.tolist()
        ok = bool((self.uc == EMPTY_WORD).all().item()) and flags[0] == self.count and flags[2] == 0
        ctx.comm.Barrier()
        return ok


def _order16(rows, copies):
    """The one-shot's order over len(rows) x copies slots (slot s holds rank s // copies's row): fp32 sums of 8 slots
    from slot 0, the sums added in order, then bf16 (round to nearest even)."""
    slots = [rows[s // copies].float() for s in range(len(rows) * copies)]
    total = torch.zeros_like(slots[0])
    for first in range(0, len(slots), 8):
        chunk = torch.zeros_like(slots[0])
        for s in slots[first : first + 8]:
            chunk = chunk + s
        total = total + chunk
    return total.bfloat16()


def _context():
    os.environ.setdefault("TRTLLM_FORCE_MNNVL_AR", "1")
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import (
        latent_op,  # noqa: F401  (registers the reduce)
    )
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op as _rq  # noqa: F401
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    return SimpleNamespace(comm=comm, rank=rank, world=world, mapping=mapping,
                           mnnvl=MNNVLAllReduce(mapping, torch.bfloat16))  # fmt: skip


def _allreduce(ctx, y: torch.Tensor) -> torch.Tensor:
    """MNNVLAllReduce sent one-shot (the order the reduce reproduces)."""
    from tensorrt_llm._torch.distributed import AllReduceParams

    return ctx.mnnvl(
        y, AllReduceParams(), one_shot_max_bytes=y.numel() * ctx.world * y.element_size()
    )


def _row(ctx, op, case, m, y, ref, got, state, rows, copies16, got16, state16, **extra):
    """One result row; ``ok`` when every check holds on every rank."""
    row = dict(op=op, case=case, M=m, nonzero=bool((y != 0).any().item()), exact=_same(got[0], ref),
               det=all(_same(g, got[0]) for g in got[1:]),
               ranks_agree=len(set(ctx.comm.allgather(_digest(got[0])))) == 1, state=state,
               exact16=_same(got16, _order16(rows, copies16)), state16=state16, **extra)  # fmt: skip
    row["ok"] = all(
        ctx.comm.allgather(all(v for k, v in row.items() if k not in ("op", "case", "M")))
    )
    return row


def _layer(name: str, m: int, weights):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    device = torch.device("cuda")
    if name == "k3_moe_m1":
        state = op.K3MoeM1State(device, I_TP, I_PAD, NUM_EXPERTS, num_tokens=m)
    else:
        state = op.K3MoeM2State(device, I_TP, I_PAD, NUM_EXPERTS)
    return state.layer(*weights)


def check_push(ctx):
    results = []
    weights = _experts(20260928 + ctx.rank)
    # Each state compiles its builds once.
    layers = {(name, m): _layer(name, m, weights) for name, m in ENGINES}
    copies16 = 16 // ctx.world
    for start in COUNT_STARTS:
        ex, ex16 = _Exchange(ctx, ctx.world, start), _Exchange(ctx, 16, start)
        for (name, m), layer in layers.items():
            for si in range(SETS):
                x_fp8, x_sf, ids, w = _tokens(m, 1000 + 10 * si + m)
                y = layer(x_fp8, x_sf, ids, w, 0)
                ref = _allreduce(ctx, y)
                got = []
                for _ in range(3):
                    layer.push(x_fp8, x_sf, ids, w, 0, ex.mc, ex.flags, ctx.rank)
                    got.append(ex.reduce(m))
                state = ex.state_ok(ctx)
                rows = [t.cuda() for t in ctx.comm.allgather(y.cpu())]
                layer.push(x_fp8, x_sf, ids, w, 0, ex16.mc, ex16.flags, ctx.rank, copies16)
                got16 = ex16.reduce(m)
                state16 = ex16.state_ok(ctx)
                results.append(_row(ctx, name, f"set{si}_count{start}", m, y, ref, got, state, rows, copies16,
                                    got16, state16))  # fmt: skip
    return results


def _front(ctx):
    """The MoE front's weights on this rank: its head rows (latent-down, then router) and the shared experts' gate_up
    rows, sharded over the run's ranks, and the head all-gather's workspace."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import K3MoeHeadWorkspace

    wl, we, inter = H // ctx.world, NUM_EXPERTS // ctx.world, SHARED_INTER // ctx.world
    gen = torch.Generator(device="cuda").manual_seed(111 + ctx.rank)
    head = (torch.randn(wl + we, HIDDEN, device="cuda", generator=gen) * 0.02).bfloat16()
    head[wl:] *= 8.0  # router rows: logits of a few units
    gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=gen) * 0.02).bfloat16()
    assert front_op.weight_supported(
        ctx.world, inter, HIDDEN, torch.device("cuda", torch.cuda.current_device())
    )
    fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or ctx.mapping.is_multi_node()
    return SimpleNamespace(weight=front_op.front_weight(head, gate_up), inter=inter,
                           head=K3MoeHeadWorkspace.create(ctx.mapping, fabric_handle=fabric))  # fmt: skip


def _k3_moe_layer(weights, route_a: bool = False):
    """This rank's experts as a K3MoeLayer of the 8-token build; a call with an exchange takes its push build."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import K3MoeState

    device = torch.device("cuda", torch.cuda.current_device())
    if route_a:
        return K3MoeState(device, I_TP_A, E_LOCAL_A).layer(*weights)
    return K3MoeState(device, I_PAD, NUM_EXPERTS).layer(*weights)


def _k3_moe_inputs(producer, m, seed):
    """The producer's inputs: the latent and router logits (k3_route_quant), or the MoE input (k3_moe_front), with the
    routing bias."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bias = (torch.randn(NUM_EXPERTS, generator=gen, device="cuda") * 0.05).float()
    if producer == "k3_route_quant":
        x = torch.randn(m, H, generator=gen, device="cuda").bfloat16()
        return x, torch.randn(m, NUM_EXPERTS, generator=gen, device="cuda"), bias
    return torch.randn(m, HIDDEN, generator=gen, device="cuda").bfloat16(), bias


def _k3_moe_routed(producer, inputs, front):
    """(MXFP8 rows, their scales, ids, weights, shared activation or None) from the producer, in K3MoeLayer's
    argument order. k3_moe_front is collective over the run's ranks (the head all-gather on ``front.head``)."""
    if producer == "k3_route_quant":
        x, logits, bias = inputs
        ids, weights, x_fp8, x_sf = _trtllm().k3_route_quant(logits, bias, x, RSF, True)
        return x_fp8, x_sf, ids, weights, None
    x, bias = inputs
    head = front.head
    ids, weights, x_fp8, x_sf, shared = _trtllm().k3_moe_front(
        x, front.weight, bias, RSF, front.inter, GATE_CAP, LINEAR_CAP, head.uc, head.mc, head.flags, head.rank,
        head.world_size,
    )  # fmt: skip
    return x_fp8, x_sf, ids, weights, shared


def check_k3_moe_push(ctx, route_a: bool = False):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe import k3_moe_push

    results = []
    front = _front(ctx)
    seed = 20260928 + ctx.rank
    layer = _k3_moe_layer(_experts(seed, route_a), route_a)
    # Route A's experts of this rank: global ids [offset, offset + 224), the rank's EP index being rank % 4.
    offset = ctx.rank * E_LOCAL_A % NUM_EXPERTS if route_a else 0
    layout = ", route A" if route_a else ""
    copies16 = 16 // ctx.world
    for start in COUNT_STARTS:
        ex, ex16 = _Exchange(ctx, ctx.world, start), _Exchange(ctx, 16, start)
        for producer in K3_MOE_PRODUCERS:
            for m in K3_MOE_TOKENS:
                for si in range(K3_MOE_SETS):
                    inputs = _k3_moe_inputs(producer, m, 2000 + 10 * si + m)
                    *routed, shared = _k3_moe_routed(producer, inputs, front)
                    y = layer(*routed, offset)
                    ref = _allreduce(ctx, y)
                    got, pushed_shared = [], []
                    for _ in range(3):
                        *routed, pushed = _k3_moe_routed(producer, inputs, front)
                        layer.push(*routed, offset, ex, ctx.rank)
                        pushed_shared.append(pushed)
                        got.append(ex.reduce(m))
                    state = ex.state_ok(ctx)
                    rows = [t.cuda() for t in ctx.comm.allgather(y.cpu())]
                    # TP16's receive side: this rank's partial in slots 4 r .. 4 r + 3, one push each.
                    for c in range(copies16):
                        *routed, pushed = _k3_moe_routed(producer, inputs, front)
                        k3_moe_push(*routed, offset, layer, ex16, ctx.rank * copies16 + c)
                        pushed_shared.append(pushed)
                    got16 = ex16.reduce(m)
                    state16 = ex16.state_ok(ctx)
                    shared_eq = shared is None or all(_same(s, shared) for s in pushed_shared)
                    results.append(_row(ctx, f"K3MoeLayer.push after {producer}{layout}", f"set{si}_count{start}", m,
                                        y, ref, got, state, rows, copies16, got16, state16,
                                        shared_eq=shared_eq))  # fmt: skip
    return results


def check_sequences(ctx):
    """The push form of the moe/k3_moe_m1 entry (catalog wrappers, one token) on one created state and one exchange,
    each push followed by one reduce, against MNNVLAllReduce's one-shot of the plain partials:
      steps     12 steps of 3 layers (the experts in 3 orders), the tokens changing every step, a random rank 5 ms
                late at every call;
      replay    one step captured and replayed 4 times with rewritten inputs, an eager push + reduce between replays;
      swapped   the negative control: rank 0 pushes two token sets in the other order. Nothing raises or hangs, but
                every rank's two sums are wrong (each reduce sums rank 0's partial of the other set); the next pair,
                in the same order on every rank, is right again."""
    import random
    import time

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe_m1 import (
        k3_moe_m1,
        k3_moe_m1_push,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op

    device = torch.device("cuda", torch.cuda.current_device())
    base = _experts(20260928 + ctx.rank)
    weights = [
        tuple(t.roll(li, dims=0).contiguous() for t in base) if li else base for li in range(3)
    ]
    state = op.K3MoeM1State.create(
        device, I_TP, I_PAD, NUM_EXPERTS, num_tokens=1, push=((ctx.world, 1),)
    )
    plain_state = op.K3MoeM1State.create(device, I_TP, I_PAD, NUM_EXPERTS, num_tokens=1)
    layers = [state.layer(*w) for w in weights]
    plain_layers = [plain_state.layer(*w) for w in weights]
    sets = [_tokens(1, 3000 + s) for s in range(5)]
    refs = {(li, si): _allreduce(ctx, k3_moe_m1(*sets[si], 0, plain_layers[li]))
            for li in range(3) for si in range(5)}  # fmt: skip
    ex = _Exchange(ctx, ctx.world, 0)
    late = random.Random(7)  # the same draws on every rank
    results = []

    def push_reduce(li, tokens):
        k3_moe_m1_push(*tokens, 0, layers[li], ex)
        return ex.reduce(1)

    good = True
    for step in range(12):
        for li in range(3):
            if late.randrange(ctx.world) == ctx.rank:
                time.sleep(0.005)
            good &= _same(push_reduce(li, sets[step % 4]), refs[li, step % 4])
    results.append(dict(op="k3_moe_m1_push", case="steps", M=1, exact=good, state=ex.state_ok(ctx)))

    static = [tuple(t.clone() for t in sets[0]) for _ in range(3)]
    graph = torch.cuda.CUDAGraph()
    ctx.comm.Barrier()
    with torch.cuda.graph(graph):
        outs = [push_reduce(li, static[li]) for li in range(3)]
    ex.count = (ex.count - 3 + 2**31) % 2**32 - 2**31  # capture launched nothing
    good = True
    for rep in range(4):
        for li in range(3):
            for dst, src in zip(static[li], sets[(rep + li) % 4]):
                dst.copy_(src)
        ctx.comm.Barrier()
        graph.replay()
        ex.count = (ex.count + 3 + 2**31) % 2**32 - 2**31
        good &= all(_same(outs[li], refs[li, (rep + li) % 4]) for li in range(3))
        good &= _same(push_reduce(rep % 3, sets[4]), refs[rep % 3, 4])
    results.append(
        dict(op="k3_moe_m1_push", case="replay", M=1, exact=good, state=ex.state_ok(ctx))
    )
    del graph

    first, second = (sets[1], sets[0]) if ctx.rank == 0 else (sets[0], sets[1])
    got0, got1 = push_reduce(0, first), push_reduce(0, second)
    wrong = [(got0 != refs[0, 0]).float().mean().item(), (got1 != refs[0, 1]).float().mean().item()]
    after = _same(push_reduce(0, sets[2]), refs[0, 2])
    results.append(dict(op="k3_moe_m1_push", case="swapped", M=1, detected=min(wrong) > 0.5, after=after,
                        state=ex.state_ok(ctx)))  # fmt: skip
    for row in results:
        row["ok"] = all(
            ctx.comm.allgather(all(v for k, v in row.items() if k not in ("op", "case", "M")))
        )
    return results


CHECKS = {
    "push": check_push,
    "k3_moe_push": check_k3_moe_push,
    "k3_moe_push_route_a": functools.partial(check_k3_moe_push, route_a=True),
    "sequences": check_sequences,
}


def _run_checks(names):
    try:
        ctx = _context()
        with torch.inference_mode():
            return [row for name in names for row in CHECKS[name](ctx)]
    except Exception:
        traceback.print_exc()
        raise


def _report(rows):
    for row in rows:
        fields = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("op", "case", "M"))
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("check", list(CHECKS))
def test_k3_moe_push(mpi_pool_executor, check):
    per_rank = list(mpi_pool_executor.map(_run_checks, [[check]] * WORLD))
    _report(per_rank[0])
    assert all(row["ok"] for rows in per_rank for row in rows)


def main() -> int:
    names = sys.argv[1:] or list(CHECKS)
    rows = _run_checks(names)
    if MPI.COMM_WORLD.Get_rank() == 0:
        _report(rows)
        print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
