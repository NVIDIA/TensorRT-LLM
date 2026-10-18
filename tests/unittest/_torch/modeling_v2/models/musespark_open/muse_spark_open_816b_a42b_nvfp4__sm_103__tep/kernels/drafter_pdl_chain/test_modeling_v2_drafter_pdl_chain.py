# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the drafter PDL chain (``pdl_conv`` / ``pdl_merge`` / ``pdl_silu``), published drafter, TP8.

Run with eight MPI ranks (one per GPU) on two NVLink-connected four-GPU GB300
nodes, with ``MUSESPARK_REFERENCE_DRAFTER=<Muse-Spark-Open-DFlash2 checkpoint>``
and ``TRTLLM_ENABLE_PDL=1`` set on every rank, running
``python -m pytest <this file>``.

Anywhere else every test skips.

The drafter is built TP8 with its real weights and the drafter fast paths
``drafter_block_conv``, ``drafter_block_decode``, ``drafter_fixed_gemms`` and
``drafter_hops``. One
request's arena holds a context of real-looking K/V, and each step runs the
block decode of ``K + 1 = 8`` rows on fresh noise. Every arm -- the base,
``pdl_conv``, ``pdl_merge``, ``pdl_silu`` and all three -- must give the base's bits: the
block decode's output and the block's K/V columns in the arena. Eagerly,
under CUDA-graph replay (2,000 replays per arm, new noise each, compared on
the device, a random rank delayed before random replays), with eager
forwards between replays. Each arm's captured graph is checked for the
programmatic edges the arm adds (into and out of the convolution and the
silu copy).

The draft head (``pdl_head`` / ``selector_fork``) runs after the block decode as
the DFlash worker runs it: the built-in slot-id lines on the draft head's worker
(or ``pdl_head``'s row view), the draft head's logits processor (``selector_fork``
forks the selector
projection there), this rank's LM-head shard GEMM (flashinfer split-K with
programmatic dependent launch; a seeded random shard, since the target's head
is not in the drafter checkpoint), then ``selector_draft_tokens`` with the real
selector and the MNNVL key all-reduce across the eight ranks. Arms base,
``pdl_head``, ``selector_fork``, both, and the full chain (all five launches) must
give the base's
shard logits and draft tokens, eagerly and over 2,000 graph replays with a
random rank delayed.

``MODELING_V2_TIMING=1`` also times the decode graph per arm (alternating
arm order, 4 rounds, median and minimum per rank, max over ranks): the
in-model drafter chain with the real predecessors and successors, 5 layers of
distinct cold weights (1.4 GB per rank) in one CUDA graph.
"""

import os
import statistics

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

__extra_import_path__ = ["../.."]
from _graph_edges import _check, _kernel_name, programmatic_edges  # noqa: E402

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")


def _world():
    try:
        from tensorrt_llm._utils import mpi_comm

        comm = mpi_comm()
        return comm, comm.Get_size(), comm.Get_rank()
    except Exception:  # noqa: BLE001 - no MPI launch means no group
        return None, 1, 0


COMM, WORLD, RANK = _world()

pytestmark = pytest.mark.skipif(
    WORLD != 8
    or not _DRAFTER_DIR
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="needs 8 MPI ranks on sm_103 or sm_107 and MUSESPARK_REFERENCE_DRAFTER",
)

CTX = 3_000
BLOCK = 8
SLOT = 1
REPLAYS = 2_000
ARMS = ("base", "pdl_conv", "pdl_merge", "pdl_silu", "conv_merge_silu")
FLAGS = {
    "base": (),
    "pdl_conv": ("c",),
    "pdl_merge": ("m",),
    "pdl_silu": ("q",),
    "conv_merge_silu": ("c", "m", "q"),
}


def _agree(ok: bool) -> bool:
    return COMM.allreduce(int(ok)) == WORLD


@pytest.fixture(scope="module")
def env():
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
        block_decode as draft,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_hops import (  # noqa: E501
        hops as drafter_hops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_pdl_chain import (  # noqa: E501
        chain as drafter_pdl,
    )
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
    from tensorrt_llm.mapping import Mapping

    assert get_env_enable_pdl(), "run with TRTLLM_ENABLE_PDL=1 (the chain needs PDL)"
    local = int(os.environ.get("SLURM_LOCALID", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=4, tp_size=WORLD)
    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda", local):
        model = DFlashForCausalLM(
            ModelConfig(pretrained_config=pretrained, mapping=mapping, attn_backend="TRTLLM")
        )
    model.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert model.has_block_conv and model.dflash_attention_backend == "VANILLA"
    draft.install(model, True, True)
    drafter_worker.install(model, None, mapping, False, False, True)
    drafter_hops.install(model)
    # The drafter's own three launches; the head's two need the worker (below).
    reason = (
        drafter_pdl._conv_reason(model)
        or drafter_pdl._merge_reason(model)
        or drafter_pdl._mlp_reason(model)
    )

    g = torch.Generator(device="cuda").manual_seed(1234)
    captured = (
        torch.randn(
            CTX, pretrained.hidden_size * len(model.target_layer_ids), generator=g, device="cuda"
        )
        * 2.0
    ).to(torch.bfloat16)
    with torch.inference_mode():
        k, v = model.precompute_context_kv(
            model.project_target_hidden(captured), torch.arange(CTX, device="cuda")
        )
    shape = (2, model._num_attn_layers, CTX + 256, model._num_kv_heads, model._head_dim)
    state = dict(
        model=model,
        mapping=mapping,
        drafter_pdl=drafter_pdl,
        reason=reason,
        pool_k=torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
        pool_v=torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
        noise=torch.empty(1, BLOCK, pretrained.hidden_size, dtype=torch.bfloat16, device="cuda"),
        positions=torch.arange(CTX, CTX + BLOCK, device="cuda").unsqueeze(0),
        num_ctx=torch.tensor([CTX], device="cuda"),
        batch_idx=torch.tensor([SLOT], device="cuda"),
        hidden=pretrained.hidden_size,
    )
    state["pool_k"][SLOT, :, :CTX] = k.permute(1, 0, 2, 3)
    state["pool_v"][SLOT, :, :CTX] = v.permute(1, 0, 2, 3)
    COMM.Barrier()
    return state


def _set_arm(env, arm):
    _set_flags(env, FLAGS[arm])


def _set_flags(env, flags):
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    model = env["model"]
    pdl = env["drafter_pdl"]
    for conv in list(model.attention_convs) + list(model.mlp_convs):
        conv.__dict__["_pdl_conv"] = "c" in flags
    model.__dict__["_pdl_merge"] = "m" in flags
    for layer in model.model.layers:
        layer.mlp.__class__ = pdl.EntryTriggerGatedMLP if "q" in flags else GatedMLP


def _forward(env):
    return env["model"].dflash_forward(
        noise_embedding=env["noise"],
        query_positions=env["positions"],
        num_ctx_per_req=env["num_ctx"],
        ctx_k_cache=env["pool_k"],
        ctx_v_cache=env["pool_v"],
        ctx_cache_batch_idx=env["batch_idx"],
    )


def _block_kv(env):
    return (
        env["pool_k"][SLOT, :, CTX : CTX + BLOCK].clone(),
        env["pool_v"][SLOT, :, CTX : CTX + BLOCK].clone(),
    )


def _noise(env, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, BLOCK, env["hidden"], generator=g, device="cuda").to(torch.bfloat16)


def _eager(env, arm, seed):
    _set_arm(env, arm)
    env["noise"].copy_(_noise(env, seed))
    env["pool_k"][SLOT, :, CTX : CTX + BLOCK].fill_(float("nan"))
    env["pool_v"][SLOT, :, CTX : CTX + BLOCK].fill_(float("nan"))
    hop = env["model"]._drafter_hop
    calls = hop.calls
    out = _forward(env).clone()
    return (out, *_block_kv(env)), hop.calls - calls


def _bits_differing(a, b) -> int:
    return sum(int((x.view(torch.int16) != y.view(torch.int16)).sum()) for x, y in zip(a, b))


def test_the_served_drafter_takes_every_chain_launch(env):
    assert env["reason"] is None, env["reason"]


@torch.inference_mode()
def test_every_arm_is_the_base_bit_for_bit_in_eager_decode(env):
    bad = []
    for seed in range(12):
        want, hops = _eager(env, "base", seed)
        if hops != 10:
            bad.append(f"base seed {seed}: {hops} fused hops")
        for arm in ARMS[1:]:
            got, hops = _eager(env, arm, seed)
            if hops != 10:
                bad.append(f"{arm} seed {seed}: {hops} fused hops, not 10")
            diff = _bits_differing(got, want)
            if diff:
                bad.append(f"{arm} seed {seed}: {diff} values differ")
    torch.cuda.synchronize()
    _set_arm(env, "base")
    assert _agree(not bad), f"rank {RANK}: {bad[:6]}"


def _capture(env, arm, keep=False):
    _set_arm(env, arm)
    _forward(env)
    torch.cuda.synchronize()
    COMM.Barrier()
    graph = torch.cuda.CUDAGraph(keep_graph=keep)
    with torch.cuda.graph(graph):
        out = _forward(env)
    return graph, out


@torch.inference_mode()
def test_each_arm_adds_its_programmatic_edges(env):
    """Edges into and out of the convolution and the silu copy, per captured arm."""
    found = {}
    for arm in ARMS:
        graph, _ = _capture(env, arm, keep=True)
        edges = programmatic_edges(graph)
        found[arm] = {
            "into_conv": sum(c for (s, t, _), c in edges.items() if "block_conv" in t),
            "conv_out": sum(c for (s, t, _), c in edges.items() if "block_conv" in s),
            "into_silu": sum(c for (s, t, _), c in edges.items() if "silu_and_mul_pdl" in t),
            "silu_out": sum(c for (s, t, _), c in edges.items() if "silu_and_mul_pdl" in s),
            "merge_out": sum(c for (s, t, _), c in edges.items() if "merge" in s),
            "norm_out": sum(c for (s, t, _), c in edges.items() if "norm" in s.lower()),
        }
        del graph
    torch.cuda.synchronize()
    _set_arm(env, "base")
    if RANK == 0:
        print("\nprogrammatic edges per arm:", found)
    bad = []
    for arm in ARMS:
        f, flags = found[arm], FLAGS[arm]
        conv_edges = 10 if "c" in flags else 0
        silu_edges = 5 if "q" in flags else 0
        if f["into_conv"] != conv_edges or f["conv_out"] < conv_edges:
            bad.append(f"{arm}: conv edges {f}")
        if f["into_silu"] != silu_edges or f["silu_out"] != silu_edges:
            bad.append(f"{arm}: silu edges {f}")
        if f["merge_out"] != 5:
            bad.append(f"{arm}: merge edges {f}")
        if f["norm_out"] < 1:
            bad.append(f"{arm}: no programmatic edge out of a norm {f}")
    assert _agree(not bad), f"rank {RANK}: {bad}"


@torch.inference_mode()
def test_graph_replays_of_every_arm_match_the_base_over_2000_steps(env):
    graphs, outs = {}, {}
    for arm in ARMS:
        graphs[arm], outs[arm] = _capture(env, arm)
    _set_arm(env, "base")
    g = torch.Generator(device="cuda").manual_seed(99)
    pool = torch.randn(64, 1, BLOCK, env["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    mismatches = {arm: torch.zeros((), dtype=torch.int64, device="cuda") for arm in ARMS[1:]}
    host = torch.Generator().manual_seed(RANK * 7 + 1)
    shared = torch.Generator().manual_seed(5)
    for step in range(REPLAYS):
        env["noise"].copy_(pool[step % 64])
        env["noise"][0, step % BLOCK, step % env["hidden"]] += float(step % 17)
        graphs["base"].replay()
        want = (outs["base"].clone(), *_block_kv(env))
        for arm in ARMS[1:]:
            late = int(torch.randint(0, 2 * WORLD, (1,), generator=shared))
            if late == RANK and step % 5 == 0:
                torch.cuda._sleep(int(torch.randint(1_000, 400_000, (1,), generator=host)))
            graphs[arm].replay()
            got = (outs[arm], *_block_kv(env))
            for a, b in zip(got, want):
                mismatches[arm] += (a.view(torch.int16) != b.view(torch.int16)).sum()
        if step % 250 == 249:
            for arm in ARMS[1:]:
                _eager(env, arm, 10_000 + step)
    torch.cuda.synchronize()
    _set_arm(env, "base")
    counts = {arm: int(m) for arm, m in mismatches.items()}
    assert _agree(all(c == 0 for c in counts.values())), f"rank {RANK}: {counts}"


# --------------------------------------------------------------------------
# The draft head (pdl_head / selector_fork) behind the block decode, eight ranks
# --------------------------------------------------------------------------
K = BLOCK - 1
HEAD_ARMS = ("base", "pdl_head", "selector_fork", "head_and_fork", "full_chain")
HEAD_FLAGS = {
    "base": (),
    "pdl_head": ("h",),
    "selector_fork": ("s",),
    "head_and_fork": ("h", "s"),
    "full_chain": ("c", "m", "q", "h", "s"),
}


@pytest.fixture(scope="module")
def head(env):
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        default_tactic,
        run_splitk_dense,
    )

    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    model, pdl, hidden = env["model"], env["drafter_pdl"], env["hidden"]
    selector = model.candidate_selector
    vocab = int(selector.predecessor_codebook.shape[0])
    shard = vocab // WORLD
    g = torch.Generator(device="cuda").manual_seed(4100 + RANK)
    weight = (torch.randn(shard, hidden, generator=g, device="cuda") / hidden**0.5).to(
        torch.bfloat16
    )
    out = torch.empty(K, shard, dtype=torch.bfloat16, device="cuda")
    tactic = default_tactic(K, shard, hidden)

    class ShardHead:
        """This rank's LM-head shard as ``gathers_column_shards`` admits it; split-K with PDL."""

        tp_mode = TensorParallelMode.COLUMN
        gather_output = True
        tp_size = WORLD
        padding_size = 0
        gather_output_sizes = None
        lora = None
        enable_lm_head_tp_in_adp = False
        bias = None

        def __init__(self):
            self.weight = weight

        def apply_linear(self, x, bias):
            return run_splitk_dense(x, weight.t(), None, out, True, tactic)

    lm_head = ShardHead()
    assert draft_head.gathers_column_shards(lm_head)
    projection = selector.hidden_projection
    assert type(projection) in (torch.nn.Linear, pdl.ForkedSelectorProjection)
    projection.__class__ = pdl.ForkedSelectorProjection
    if projection.__dict__.get("_selector_fork") is None:
        projection._selector_fork = pdl.SelectorFork(projection)
    workers = {}
    for flag in (False, True):
        worker = object.__new__(draft_head.ShardedDraftHeadDFlashWorker)
        worker.__dict__["_pdl_head"] = flag
        workers[flag] = worker
    allreduce = draft_head.key_allreduce(env["mapping"])
    anchor = torch.randint(
        0, vocab, (1,), generator=torch.Generator(device="cuda").manual_seed(11), device="cuda"
    )
    COMM.Barrier()
    return dict(
        draft_head=draft_head,
        selector=selector,
        lm_head=lm_head,
        processor=draft_head.ShardDraftLogitsProcessor(torch.nn.Identity()),
        fork=projection._selector_fork,
        workers=workers,
        allreduce=allreduce,
        anchor=anchor,
    )


def _head_step(env, head, arm):
    """The block decode, then the served worker's draft-head lines: (shard logits, draft tokens)."""
    flags = HEAD_FLAGS[arm]
    hidden = env["hidden"]
    hidden_states_out = _forward(env).reshape(-1, hidden)
    worker = head["workers"]["h" in flags]
    gen_gather_ids = worker._draft_slot_ids(None, 1, BLOCK, K)
    gen_gather_ids = gen_gather_ids.clamp(max=hidden_states_out.shape[0] - 1)
    rows = hidden_states_out[gen_gather_ids]
    processor = head["processor"]
    processor.__dict__["_selector_fork"] = head["fork"] if "s" in flags else None
    logits = processor(rows, head["lm_head"], None, True)
    tokens = head["draft_head"].selector_draft_tokens(
        logits.view(1, K, -1),
        rows.reshape(1, K, hidden),
        head["anchor"],
        head["selector"],
        env["mapping"],
        head["allreduce"],
    )
    return logits, tokens


def _head_arm(env, arm):
    _set_flags(env, tuple(f for f in HEAD_FLAGS[arm] if f in ("c", "m", "q")))


def _same(a, b) -> bool:
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


@torch.inference_mode()
def test_draft_head_arms_are_the_base_bit_for_bit_in_eager_steps(env, head):
    fork = head["fork"]
    launches, taken, forked_steps = fork.launches, fork.taken, 0
    bad = []
    for seed in range(12):
        env["noise"].copy_(_noise(env, seed))
        _head_arm(env, "base")
        want = [t.clone() for t in _head_step(env, head, "base")]
        for arm in HEAD_ARMS[1:]:
            _head_arm(env, arm)
            logits, tokens = _head_step(env, head, arm)
            forked_steps += "s" in HEAD_FLAGS[arm]
            if not _same(logits, want[0]):
                bad.append(f"{arm} seed {seed}: shard logits differ")
            if not _same(tokens, want[1]):
                bad.append(f"{arm} seed {seed}: draft tokens differ")
    torch.cuda.synchronize()
    _set_arm(env, "base")
    if (fork.launches - launches, fork.taken - taken) != (forked_steps, forked_steps):
        bad.append(
            f"fork launched {fork.launches - launches} / taken {fork.taken - taken}, "
            f"expected {forked_steps}"
        )
    if RANK == 0:
        exchange = "MNNVL all-reduce" if head["allreduce"] is not None else "all-gather"
        print(f"\ndraft head: key exchange by {exchange}; draft tokens {want[1].tolist()}")
    assert _agree(not bad), f"rank {RANK}: {bad[:6]}"


def _head_capture(env, head, arm, keep=False):
    _head_arm(env, arm)
    _head_step(env, head, arm)
    torch.cuda.synchronize()
    COMM.Barrier()
    graph = torch.cuda.CUDAGraph(keep_graph=keep)
    with torch.cuda.graph(graph):
        logits, tokens = _head_step(env, head, arm)
    return graph, logits, tokens


def _graph_census(graph):
    """(kernel-name counts, nodes with more than one dependent) of a kept, unreplayed graph."""
    import collections

    from cuda.bindings import driver

    handle = driver.CUgraph(graph.raw_cuda_graph())
    count = int(_check(driver.cuGraphGetNodes(handle, 0))[-1])
    nodes = _check(driver.cuGraphGetNodes(handle, count))[0]
    names = {}
    kernels = collections.Counter(
        name for name in (_kernel_name(node, names) for node in nodes) if not name.startswith("<")
    )
    get = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    total = int(_check(get(handle, 0))[-1])
    sources = _check(get(handle, total))[0]
    fan_out = collections.Counter(int(source) for source in sources)
    return kernels, sum(1 for degree in fan_out.values() if degree > 1)


@torch.inference_mode()
def test_draft_head_graphs_drop_the_gather_and_fork_the_projection(env, head):
    """pdl_head: the clamp and the gather kernels are gone; selector_fork: the same kernels plus one fork."""
    census, edges = {}, {}
    for arm in ("base", "pdl_head", "selector_fork"):
        graph, _, _ = _head_capture(env, head, arm, keep=True)
        census[arm] = _graph_census(graph)
        edges[arm] = programmatic_edges(graph)
        del graph
    torch.cuda.synchronize()
    _set_arm(env, "base")
    (base_kernels, base_forks), (h_kernels, h_forks), (s_kernels, s_forks) = (
        census["base"],
        census["pdl_head"],
        census["selector_fork"],
    )
    dropped = base_kernels - h_kernels
    if RANK == 0:
        print(
            f"\npdl_head drops {dict(dropped)}; fan-out nodes base {base_forks}, selector_fork {s_forks}"
        )
        added = {k: v for k, v in edges["pdl_head"].items() if edges["base"].get(k, 0) < v}
        print("pdl_head programmatic edges not in the base:", added)
    bad = []
    if sum(dropped.values()) != 2 or h_kernels - base_kernels:
        bad.append(
            f"pdl_head kernels: dropped {dict(dropped)}, added {dict(h_kernels - base_kernels)}"
        )
    if s_kernels != base_kernels:
        bad.append("selector_fork changed the captured kernels")
    if s_forks != base_forks + 1 or h_forks != base_forks:
        bad.append(f"fan-out nodes base {base_forks}, pdl_head {h_forks}, selector_fork {s_forks}")
    assert _agree(not bad), f"rank {RANK}: {bad}"


@torch.inference_mode()
def test_draft_head_graph_replays_match_the_base_over_2000_steps(env, head):
    graphs, outs = {}, {}
    for arm in HEAD_ARMS:
        graph, logits, tokens = _head_capture(env, head, arm)
        graphs[arm], outs[arm] = graph, (logits, tokens)
    _set_arm(env, "base")
    g = torch.Generator(device="cuda").manual_seed(98)
    pool = torch.randn(64, 1, BLOCK, env["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    mismatches = {arm: torch.zeros((), dtype=torch.int64, device="cuda") for arm in HEAD_ARMS[1:]}
    host = torch.Generator().manual_seed(RANK * 7 + 3)
    shared = torch.Generator().manual_seed(6)
    for step in range(REPLAYS):
        env["noise"].copy_(pool[step % 64])
        env["noise"][0, step % BLOCK, step % env["hidden"]] += float(step % 17)
        graphs["base"].replay()
        want = [t.clone() for t in outs["base"]]
        for arm in HEAD_ARMS[1:]:
            late = int(torch.randint(0, 2 * WORLD, (1,), generator=shared))
            if late == RANK and step % 5 == 0:
                torch.cuda._sleep(int(torch.randint(1_000, 400_000, (1,), generator=host)))
            graphs[arm].replay()
            logits, tokens = outs[arm]
            mismatches[arm] += (logits.view(torch.int16) != want[0].view(torch.int16)).sum()
            mismatches[arm] += (tokens != want[1]).sum()
        if step % 250 == 249:
            for arm in HEAD_ARMS:
                _head_arm(env, arm)
                _head_step(env, head, arm)
            _set_arm(env, "base")
    torch.cuda.synchronize()
    counts = {arm: int(m) for arm, m in mismatches.items()}
    assert _agree(all(c == 0 for c in counts.values())), f"rank {RANK}: {counts}"


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_drafter_chain(env):
    """The block decode (5 layers, 10 hop sites) as one CUDA graph per arm; us per forward."""
    graphs = {}
    for arm in ARMS:
        graphs[arm], _ = _capture(env, arm)
        for _ in range(20):
            graphs[arm].replay()
    torch.cuda.synchronize()
    samples = {arm: [] for arm in ARMS}
    for rnd in range(4):
        order = ARMS if rnd % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            for _ in range(5):
                COMM.Barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) * 1000.0 / 50)
    result = {}
    for arm in ARMS:
        result[arm] = COMM.allreduce(statistics.median(samples[arm]), op=MPI.MAX)
        result[arm + "_min"] = COMM.allreduce(min(samples[arm]), op=MPI.MAX)
        result[arm + "_rounds"] = [
            COMM.allreduce(statistics.median(samples[arm][5 * r : 5 * r + 5]), op=MPI.MAX)
            for r in range(4)
        ]
    if RANK == 0:
        print("\ndrafter block decode (5 layers), us per forward, max over ranks:")
        for arm in ARMS:
            rounds = " ".join(f"{x:7.2f}" for x in result[arm + "_rounds"])
            print(
                f"  {arm:8s} median {result[arm]:8.2f}  min {result[arm + '_min']:8.2f}  "
                f"delta vs base {result[arm] - result['base']:+7.2f}  rounds {rounds}"
            )
    _set_arm(env, "base")
