# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: ``persistent_experts`` on real weights in the composed target (TP8) -- bit for bit
the ``replicated_experts`` model.

Run like ``test_modeling_v2_musespark_open_decode_forward.py`` (eight MPI ranks on
two NVLink-connected four-GPU GB300 nodes, the checkpoint in ``MUSESPARK_REFERENCE_CKPT``
and the deployment's switches); anywhere else every test skips. That
module's harness is reused (loaded by path, its own tests not collected here)
with this module's layers, arms and entry points.

The text model is built over checkpoint layers 0, 5, 30, 59, 60 and 61: the
two BF16-expert layers (0 and 61, which keep the ring-buddy FlashInfer call) and
four NVFP4 MoE layers, among them 5 and 30, whose real experts, block scales
and per-expert scalars the persistent kernel runs on. It returns every routed FC2 row
bitwise and only changes the row map the fused finalize all-reduce reads
(``expert_rows.py``), so:

* ``ring_buddy,persistent_experts`` equals ``ring_buddy``, and the tier-1 target set with
  ``persistent_experts`` equals it without -- every decode output
  (final and both outputs of every layer), the greedy token of every row and
  the KV cache bytes of every layer, eager and replayed from a CUDA graph, at
  1..8 rows (the persistent kernel) and at 9 and 16 rows (it leaves those to the stacked
  TRTLLM-gen call); ``persistent_experts`` runs on exactly the four NVFP4 layers
  at 1..8 rows and never above.
* Per call, on each NVFP4 layer's real ring-buddy stack: many routings of the
  layer's real router (independent rows, repeated rows, a router input scaled
  down until the logits nearly tie) placed by ``ring_buddy_assign`` (balanced),
  with expert inputs from the layer's own pre-expert norm + NVFP4 quantization,
  through the stacked TRTLLM-gen call and through the persistent kernel: every row the
  finalize selects (``expert_rows.gather_routed_rows``) is bitwise equal, and
  so is the built-in fused finalize all-reduce's output.
* The tier-2 target set (tier 1 plus ``early_release``, ``persistent_experts`` and
  ``stale_poll``) equals tier 1.
"""

import importlib.util
import json
import os
import random
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.persistent_experts import (  # noqa: E501
    rows as expert_rows,
)


def _load_decode_forward_harness():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "test_modeling_v2_musespark_open_decode_forward.py",
    )
    spec = importlib.util.spec_from_file_location("_decode_forward_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_decode_forward_harness()

#: Checkpoint layers (BF16 experts at 0 and 61; NVFP4 at 5, 30, 59, 60).
CKPT_LAYERS = (0, 5, 30, 59, 60, 61)
NVFP4_LAYERS = 4
#: The tier-1 target-side set.
TARGET = "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch"
ARMS = (
    ("builtin", None, None),
    ("ring_buddy", "ring_buddy", None),
    ("ring_buddy_persistent_experts", "ring_buddy,persistent_experts", "ring_buddy"),
    ("target", TARGET, None),
    ("target_persistent_experts", TARGET + ",persistent_experts", "target"),
    ("target4", TARGET + ",early_release,persistent_experts,stale_poll", "target"),
)
ENTRIES = {
    "ring_buddy_assign": (set(c3._plans.LOADED), len(CKPT_LAYERS), 16),
    "persistent_experts": ({"persistent_experts"}, NVFP4_LAYERS, 8),
    "attention_norm_gate": ({"early_release"}, len(CKPT_LAYERS), 8),
}
#: Calls per NVFP4 layer per rank in the per-call comparison.
CALLS = int(os.environ.get("MODELING_V2_PERSISTENT_EXPERTS_CALL_LOG") or 1000)

c3.CKPT_LAYERS = CKPT_LAYERS
c3.ARMS = ARMS
c3.ENTRIES = ENTRIES
c3.COMBINED = {}
c3.DETERMINISM = ("target_persistent_experts",)

_PER_CALL: dict = {}


@pytest.fixture(scope="module")
def env():
    """Every arm's model copy, the LM head shard and the real embedding rows (collective)."""
    comm, rank, checkpoint = c3._world()
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
        triggers as pdl_triggers,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
        persistent_experts as pe_mod,
    )
    from tensorrt_llm.mapping import Mapping

    e = types.SimpleNamespace(comm=comm, rank=rank, checkpoint=checkpoint, sink=[], models={})
    e.mapping = Mapping(
        world_size=c3.WORLD,
        rank=rank,
        gpus_per_node=4,
        tp_size=c3.WORLD,
        moe_ep_size=c3.WORLD,
        moe_tp_size=1,
    )
    weights, open_format = c3._checkpoint_weights(checkpoint)
    patched = [
        (ring_buddy, "ring_buddy_assign", "ring_buddy_assign"),
        (pe_mod, "persistent_experts", "persistent_experts"),
        (pdl_triggers._norm_gate, "attention_norm_gate", "attention_norm_gate"),
    ]
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patched]
    for module, attr, name in patched:
        setattr(module, attr, c3._Counted(name, getattr(module, attr)))
    try:
        e.layout = {}
        for arm, selection, _ in ARMS:
            e.models[arm], config = c3._build_arm(e, weights, open_format, selection)
            e.layout[arm] = [
                (
                    hasattr(layer.feed_forward, "_replicated_experts"),
                    getattr(getattr(layer.feed_forward, "_replicated_experts", None), "kind", None),
                )
                for layer in e.models[arm].layers
            ]
            comm.Barrier()
        e.pretrained = config.pretrained_config
        e.free_gib = torch.cuda.mem_get_info()[0] / 2**30
        embed = weights["model.embed_tokens.weight"]
        generator = torch.Generator().manual_seed(2026)
        tokens = torch.randint(0, embed.get_shape()[0], (c3.TOKEN_POOL,), generator=generator)
        e.table = torch.cat([embed[t : t + 1] for t in tokens.tolist()]).to("cuda", torch.bfloat16)
        shard = c3.VOCAB // c3.WORLD
        e.vocab_begin = rank * shard
        e.head = weights["lm_head.weight"][e.vocab_begin : e.vocab_begin + shard].to(
            "cuda", torch.float32
        )
        torch.cuda.synchronize()
        comm.Barrier()
        yield e
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        for module, attr, original in originals:
            setattr(module, attr, original)


def test_the_arms_have_the_expected_layout(env):
    """Every MoE layer of every arm with ``ring_buddy`` is installed; NVFP4 layers 1-4, BF16 0 and 5."""
    e = env
    wrong = []
    kinds = ["bf16", "nvfp4", "nvfp4", "nvfp4", "nvfp4", "bf16"]
    for arm, selection, _ in ARMS:
        ring_buddy_on = bool(c3._selection(selection) & c3._plans.LOADED)
        for index, (replicated, kind) in enumerate(e.layout[arm]):
            if replicated != ring_buddy_on or (ring_buddy_on and kind != kinds[index]):
                wrong.append(f"{arm} layer {index}: replicated={replicated} kind={kind}")
    assert not wrong, wrong


@pytest.mark.parametrize("rows", c3.ROWS)
def test_persistent_experts_is_the_ring_buddy_model_in_eager_decode(env, rows):
    c3._compare_arms(env, rows, "eager")


@pytest.mark.parametrize("rows", c3.ROWS)
def test_persistent_experts_is_the_ring_buddy_model_in_cuda_graphs(env, rows):
    c3._compare_arms(env, rows, "graph")


def _routings(moe, rows: int, kind: str, gen: torch.Generator):
    """(selected, weights) of the layer's real router on one synthetic hidden state."""
    hidden = torch.randn(rows, 8192, generator=gen)
    if kind == "repeat":
        hidden = hidden[:1].expand(rows, -1).clone()
    elif kind == "pairs":
        hidden = hidden[torch.arange(rows) // 2]
    elif kind == "near_tie":
        hidden = hidden * 1e-3
    logits = moe.router_logits(hidden.to("cuda", torch.bfloat16))
    return moe.experts.routing_method.apply(logits)


@torch.inference_mode()
def test_per_call_bitwise_on_the_real_stacks(env):
    """Each NVFP4 layer's real stack, ``CALLS`` routings per rank: the persistent kernel's rows and the finalized
    block are bitwise the stacked TRTLLM-gen call's (collective)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
        ring_buddy_assign,
    )

    e = env
    model = e.models["ring_buddy_persistent_experts"]
    rng = random.Random(17)
    gen = torch.Generator().manual_seed(17)
    kinds = ("independent", "repeat", "pairs", "near_tie")
    report = {
        "calls": 0,
        "slots": 0,
        "rows_differ": 0,
        "blocks_differ": 0,
        "e_hist": {},
    }
    for index, layer in enumerate(model.layers):
        moe = layer.feed_forward
        state = getattr(moe, "_replicated_experts", None)
        if state is None or state.kind != "nvfp4":
            continue
        backend = ring_buddy._backend(moe)
        routing = moe.experts.routing_method
        stacked = ring_buddy._nvfp4_call(backend, routing)
        persistent = ring_buddy._nvfp4_call(
            backend, routing, persistent=True, stacks=state.by_window
        )
        workspace = getattr(moe, "_finalize_ar_workspace", None)
        assert workspace is not None, "the fused finalize all-reduce is off"
        for call in range(CALLS):
            rows = 1 + call % 8
            kind = kinds[rng.randrange(len(kinds))]
            selected, weights = _routings(moe, rows, kind, gen)
            routing.clear_routing_memo()
            local_ids, weights_bf16, _ = ring_buddy_assign(selected, weights, e.rank, True)
            pooled = torch.randn(rows, 4096, generator=gen) * (0.25 + 4 * rng.random())
            x = moe.prepare_expert_input(pooled.to("cuda", torch.bfloat16))
            ours = persistent(state.stacks, x, local_ids, weights_bf16)
            theirs = stacked(state.stacks, x, local_ids, weights_bf16)
            assert ours is not None and theirs is not None, "an expert call fell back"
            a, valid_a = expert_rows.gather_routed_rows(ours[0], ours[2], rows, 4096)
            b, valid_b = expert_rows.gather_routed_rows(theirs[0], theirs[2], rows, 4096)
            assert torch.equal(valid_a, valid_b)
            differ = int((a.view(torch.int32) != b.view(torch.int32)).any(-1).sum())
            norm = moe.post_expert_norm_weight
            eps = moe.post_expert_norm_eps
            block_a = workspace.finalize(ours[0], ours[2], weights, norm, eps, rows)
            block_b = workspace.finalize(theirs[0], theirs[2], weights, norm, eps, rows)
            active = len({int(v) for v in local_ids.reshape(-1).tolist() if 0 <= int(v) < 64})
            report["e_hist"][active] = report["e_hist"].get(active, 0) + 1
            report["calls"] += 1
            report["slots"] += int(valid_a.sum())
            report["rows_differ"] += differ
            report["blocks_differ"] += int(not torch.equal(block_a, block_b))
    everyone = e.comm.allgather(report)
    _PER_CALL["per_rank"] = everyone
    bad = {r: rep for r, rep in enumerate(everyone) if rep["rows_differ"] or rep["blocks_differ"]}
    assert all(rep["calls"] for rep in everyone), "no NVFP4 layer was compared"
    assert not bad, f"persistent_experts differs from the stacked TRTLLM-gen call: {bad}"


def test_write_the_result_table(env):
    """Rank 0 writes the per-arm, per-width table and the per-call summary when asked to."""
    e = env
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and e.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        table = {
            f"{mode}:{arm}@{rows}": result
            for (mode, arm, rows), result in sorted(c3._RESULTS.items())
        }
        table["layout"] = e.layout
        table["free_gib_after_build"] = e.free_gib
        table["per_call"] = _PER_CALL
        with open(os.path.join(out_dir, "compose_persistent_experts.json"), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True, default=str)
