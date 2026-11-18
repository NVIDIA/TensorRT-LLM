# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's generation step on a 4-layer Kimi K3, eager against CUDA graph replay, bitwise.

16 ranks (GB300: 4 nodes x 4 GPUs, or Rubin), one MPI world (``_ull_helix16_harness``). The model
is the target's text backbone (``KimiK3Helix16UllModelingV2``) with the checkpoint's config
(``$K3_CKPT``) cut to 4 layers, every kind of layer and seam once:

* layer 0: KDA (core on the chain, ``run_solo``) + o_proj all-reduce + the dense MLP kernels;
* layer 1: the chain's entry (our mix finishes layer 0's all-reduce), KDA + MoE on the chain;
* layer 2: KDA + MoE on the chain, its consumer storing the MLA layer's input (MXFP8 at <= 16
  rows);
* layer 3: MLA + its MoE on the chain, consumed by the output AttnRes + final norm. Its KV cache
  is FP8 (model-wide FP8 KV): the target decodes an FP4 cache (CuTe DSL, SM107 only) or an FP8
  cache (the CuTe DSL FP8 decode of the deployment's layers 91 / 92, SM103 and SM107), and nothing
  else. Under the deployment's NVFP4 KV a model's only MLA layer stays NVFP4, which needs the FP4
  MLA cache manager that ``MlaCache`` does not build.

The NVFP4 routed experts run their Rubin engines forced off Rubin (``nv_force``). Weights: the
modules' random (``fill``), the chain's from the synthetic checkpoint of the chain's tests
(896 experts; ``num_attn_res_blocks`` of the output mix set to the 4-layer model's one bank row).
KDA state: a stand-in of the cache manager's checkpoints (``_kda_checkpoints`` views over plain
device buffers). Each step: 1, 2 or 4 requests x 8 tokens, DSpark taps of layers 0, 2 and 3.

Checks: the output rows and the taps of a CUDA graph replay equal the eager step's, from the same
KV cache, KDA checkpoints and input.

Run (ocijhb, 4 GB300 nodes)::

    K3_CKPT=<Kimi K3 NVFP4 checkpoint> srun -N4 --ntasks-per-node=4 --mpi=pmix \\
        python -m pytest -q -p no:cacheprovider \\
        tests/unittest/_torch/modeling_v2/models/kimi_k3_vl/\\
kimi_k3_nvfp4__sm_107__helix16_moetp16ep1/test_k3_ull_small_model_graph.py
"""

import dataclasses
from importlib import import_module
from types import SimpleNamespace as NS

import _synthetic_checkpoint as synth
import _ull_helix16_harness as h
import pytest
import torch

from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = [pytest.mark.threadleak(enabled=False)]

LAYERS = 4
Q = 8
TAPS = (0, 2, 3)


class _KdaState:
    """The chain's KDA checkpoint state over plain device buffers: per KDA layer one
    ``kda_ckpt`` buffer ``[slots, payload]`` (``_kda_checkpoints`` layout, BF16 states), random
    contents, every slot's accepted count 0 (checkpoint 0 is the step's start)."""

    def __init__(self, layers, slots, heads, device):
        ck = import_module(h.T.rpartition(".")[0] + ".ops.cache.k3_ull_kda_checkpoints")
        layout = ck.kda_checkpoint_layout(Q - 1, heads, 128, 3, torch.bfloat16, torch.bfloat16)
        gen = torch.Generator(device=device).manual_seed(7)
        self.raw, self.views = {}, {}
        for i in layers:
            raw = torch.zeros((slots, layout.nbytes), dtype=torch.uint8, device=device)
            views = ck.kda_checkpoint_views(raw, layout)
            views.ckpt.copy_(torch.randn(views.ckpt.shape, device=device, generator=gen) * 0.05)
            views.vconv.copy_(torch.randn(views.vconv.shape, device=device, generator=gen) * 0.5)
            self.raw[i], self.views[i] = raw, views
        first = self.views[layers[0]]
        self.rows = ck.checkpoint_position_rows(first, device)
        self.slot_units = first.slot_units
        self.accepted = torch.zeros(slots, dtype=torch.int32, device=device)
        self._ck = ck
        self._saved = None

    def caches(self, layer_idx):
        v = self.views[layer_idx]
        return [v.conv_flat, v.state_flat]

    def indices(self, state_indices, slots, accepted):
        n = slots.shape[0]
        self._ck.checkpoint_indices(
            state_indices[:n], self.accepted, self.rows, self.slot_units, slots, accepted
        )

    def snapshot(self):
        self._saved = {i: r.clone() for i, r in self.raw.items()}

    def restore(self):
        for i, r in self._saved.items():
            self.raw[i].copy_(r)


class _Taps:
    """``spec_metadata`` of the taps: ``maybe_capture_hidden_states`` copies into fixed rows."""

    def __init__(self, rows, hidden):
        self._capture_layer_set = set(TAPS)
        self.rows = {
            i: torch.zeros((rows, hidden), dtype=torch.bfloat16, device="cuda") for i in TAPS
        }

    def maybe_capture_hidden_states(self, layer_idx, value, residual):
        self.rows[layer_idx].copy_(value)


def _model(rank):
    modeling = import_module(f"{h.T}.modeling")
    fwr = import_module(f"{h.T}.modeling")
    cw = import_module(h.T.rpartition(".")[0] + ".ops.chain.k3_ull_chain")
    model_config, cfg, mapping_cp = h.text_model_config(
        rank, num_layers=LAYERS, kv_cache_quant_algo=QuantAlgo.FP8
    )
    model = modeling.KimiK3Helix16UllModelingV2(model_config).cuda()
    h.fill(model, seed=rank)
    l0 = model.layers[0]
    l0.mlp.down.data[:, l0.mlp.inter :].zero_()
    fwr._swap_linear_to_fp8_weight_read(l0.linear_attn, "o_proj", (type(l0.linear_attn.o_proj),))
    h.post_load(l0.linear_attn.o_proj)
    for layer in model.mla_layers:
        h.finalize_mla(layer.self_attn, h.T)

    ckpt = synth.synthetic_checkpoint(range(LAYERS))
    weights = cw.load_chain_weights(
        ckpt, synth.text_config(), rank, torch.device("cuda"), layers=list(range(LAYERS))
    )
    # The 4-layer model's bank has one row: the output mix reads that one.
    weights = dataclasses.replace(
        weights,
        output=dataclasses.replace(
            weights.output, num_attn_res_blocks=model.num_attn_res_snapshots
        ),
    )
    for layer in model.layers:
        layer.attn_res = weights.layers[layer.layer_idx].attn_res
    model.chain_weights = weights
    return model, cfg, mapping_cp


@pytest.fixture(scope="module")
def built():
    comm, rank = h.world()
    model, cfg, mapping_cp = _model(rank)
    heads = cfg.linear_attn_config["num_heads"] // h.WORLD
    kda = _KdaState([i for i, k in enumerate(model.kinds) if k != 2], 5, heads, "cuda")
    model.build_chain(None, kda_state=kda)
    ok, why = model.chain.effect_ok()
    assert ok, why
    return comm, rank, model, cfg, mapping_cp, kda


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_step_graph_replay_matches_eager(built, batch):
    comm, rank, model, cfg, mapping_cp, kda = built
    rows = batch * Q
    cache = h.MlaCache(
        mapping_cp,
        batch=batch,
        q=Q,
        ctx_len=16 * 128,
        num_layers=LAYERS,
        dtype=torch.float8_e4m3fn,
    )
    md = cache.metadata
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    md.mamba_metadata = NS(generation_state_indices=slots, state_indices=slots)
    gen = torch.Generator(device="cuda").manual_seed(99)
    ids = torch.randint(0, cfg.vocab_size, (rows,), device="cuda", generator=gen)
    taps = _Taps(rows, cfg.hidden_size)
    cache.snapshot()
    kda.snapshot()

    def step():
        cache.restore()
        kda.restore()
        return model(md, input_ids=ids, spec_metadata=taps)

    want = step().clone()
    want_taps = {i: t.clone() for i, t in taps.rows.items()}
    bad = [
        n
        for n, t in (("out", want), *((f"tap{i}", t) for i, t in want_taps.items()))
        if not h.finite(t)
    ]
    every = comm.allgather(bad)
    assert not any(every), "the eager step is not finite: " + "; ".join(
        f"rank {r}: {b}" for r, b in enumerate(every) if b
    )
    again = step().clone()
    ok = h.same_bits(want, again)
    assert h.all_ranks(comm, ok), f"rank {rank}: two eager steps differ"

    graph = torch.cuda.CUDAGraph()
    comm.Barrier()
    with torch.cuda.graph(graph):
        out = step()
    for t in taps.rows.values():
        t.zero_()
    graph.replay()
    torch.cuda.synchronize()
    # Which of the step's outputs differ, on which ranks: the taps follow layers 0, 2 and 3, so
    # the first differing one names the layer.
    pairs = {"out": (want, out), **{f"tap{i}": (want_taps[i], taps.rows[i]) for i in TAPS}}
    bad = {
        name: int((h.bits(a) != h.bits(b)).sum()) if a.shape == b.shape else -1
        for name, (a, b) in pairs.items()
        if not h.same_bits(a, b)
    }
    every = comm.allgather(bad)
    assert not any(every), (
        "the graph replay differs from the eager step (differing elements): "
        + "; ".join(f"rank {r}: {b}" for r, b in enumerate(every) if b)
    )
