# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's DSpark drafter (``K3DSparkDrafter`` of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``) on one GPU,
at one TP16 rank's drafter shapes: 6 query heads and 1 KV head of 64, hidden 7168, MLP width 896, context pages of
64 rows, two layers.

* A decode block of every split ``attention/k3_drafter_attn_qknorm`` certifies runs the drafter entries (the
  attention once per layer) and matches the stock block forward (``DFlashForCausalLM.dflash_forward`` on the same
  module and context), with the torch GEMMs and with the decode GEMV sites (which take the blocks of at most 8 rows).
  The entries leave the cache untouched.
* A split the entry does not certify runs the stock forward, bit for bit, without the entry.
* Under CUDA-graph capture, the entries take a block only once its attention compile key has run eagerly.
* Negative control: a weight changed between the two forwards fails the comparison.
* With ``ctx_rows_start`` the entries read the batch's page-table rows as a view of a larger table (the manager's,
  rows from the first gen request on), not a gathered copy, and give the gather's result bit for bit.

On the TP group's collective state (``use_decode_comm``), here a group of one rank whose collectives are torch
stand-ins counted per call (the ops themselves are certified by their multi-GPU op matrices):

* The context projection runs on the split ``fc`` (on one rank, the whole weight): up to a decode step's rows through
  ``comm/mnnvl_fusion_allreduce`` with ``hidden_norm``, more rows through the drafter's TP all-reduce, and matches the
  replicated projection.
* A decode block of every certified split runs its residual adds and RMSNorms in its all-reduces and matches the
  stock block forward: up to 8 rows ``comm/k3_sandwich_plain`` for o_proj (and, on the decode GEMV sites, its
  SiLU-and-mul form for the down projection), else ``comm/mnnvl_fusion_allreduce``, two per layer, the last with the
  final norm. Both sandwich forms compiled with the state.
* The stock all-reduces and norms run without the collective state, where the workspace does not hold the block's
  rows, or where the layers' norms do not take the fused form; a sandwich form that did not compile gives way to
  ``comm/mnnvl_fusion_allreduce``.
* A fused block captured in a CUDA graph makes the same collective calls and replays the eager result.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    decode_comm,
    decode_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as target,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="the K3 drafter entries run on sm_100 only",
)

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM, LAYERS, PAGE = 7168, 896, 6, 1, 64, 2, 64
VOCAB, RANK = 1024, 16
# Context lengths per request: inside a page, on a page boundary, across pages, empty.
CTX = (37, 64, 130, 0, 5, 200, 63, 1)
REL_L2 = 1e-2
SITES = ("drafter_qkv", "drafter_o", "drafter_gate_up", "drafter_down")


def _config():
    return Qwen3Config.from_dict(
        dict(
            architectures=["Qwen3ForCausalLM"],
            model_type="qwen3",
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_hidden_layers=LAYERS,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            torch_dtype="bfloat16",
            tie_word_embeddings=False,
            markov_rank=RANK,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            dflash_config={"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]},
        )
    )


def _weights(seed=11):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {
        "fc.weight": rnd(HIDDEN, 2 * HIDDEN),
        "hidden_norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_head.proj.weight": rnd(1, HIDDEN + RANK),
        "confidence_head.proj.bias": rnd(1),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.k_proj.weight"] = rnd(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.v_proj.weight"] = rnd(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = rnd(HEAD_DIM, scale=0.05) + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(HEAD_DIM, scale=0.05) + 1.0
        w[p + "input_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTER)
    return w


def _load_drafter():
    model_config = ModelConfig(pretrained_config=_config(), attn_backend="TRTLLM")
    module = target.K3DSparkDrafter(model_config, dflash_attention_backend="TRTLLM").to("cuda")
    module.load_weights(_weights())
    assert module._k3_layers_take(), "the test drafter must be one the drafter entries take"
    return module


@pytest.fixture(scope="module")
def drafter():
    return _load_drafter()


# The TP group's collective state for a group of one rank: the workspaces' sizes are what the drafter reads; their
# collectives are the stand-ins below.
ONE_RANK_COMM = decode_comm.K3DecodeComm(
    mnnvl=SimpleNamespace(world_size=1, buffer_bytes=decode_comm.MNNVL_BUFFER_BYTES),
    sandwich=SimpleNamespace(world_size=1),
)
DECODE_ROWS = target.MAX_REQUESTS * target.MAX_TOKENS_PER_REQUEST


def _rms_norm(x, weight, eps):
    """``RMSNorm(x) * weight`` in fp32, rounded once to bf16."""
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(
        torch.bfloat16
    )


def _fusion_allreduce_one_rank(
    input, workspace, one_shot_max_bytes, residual=None, norm_weight=None, eps=None
):
    """``comm/mnnvl_fusion_allreduce`` over one rank: the sum is the input, then the residual add and the RMSNorm."""
    assert workspace is ONE_RANK_COMM.mnnvl
    assert one_shot_max_bytes == decode_comm.DECODE_AR_ONE_SHOT_MAX_BYTES
    assert input.dtype == residual.dtype == norm_weight.dtype == torch.bfloat16
    assert input.is_contiguous() and residual.is_contiguous() and input.shape == residual.shape
    updated = (input.float() + residual.float()).to(torch.bfloat16)
    return _rms_norm(updated, norm_weight, eps), updated


def _sandwich_plain_one_rank(x, weight, residual, norm_weight, eps, workspace, swiglu=False):
    """``comm/k3_sandwich_plain`` over one rank: the projection (of ``silu(gate) * up`` with ``swiglu``), then the
    residual add and the RMSNorm."""
    assert workspace is ONE_RANK_COMM.sandwich
    k = weight.shape[1]
    assert (
        x.shape == (residual.shape[0], 2 * k if swiglu else k)
        and residual.shape[1] == weight.shape[0]
    )
    assert x.is_contiguous() and residual.is_contiguous() and isinstance(eps, float)
    a = x.float()
    if swiglu:
        a = (torch.nn.functional.silu(a[:, :k]) * a[:, k:]).to(torch.bfloat16).float()
    partial = (a @ weight.float().T).to(torch.bfloat16)
    updated = (residual.float() + partial.float()).to(torch.bfloat16)
    return _rms_norm(updated, norm_weight, eps), updated


def _one_rank_collectives(monkeypatch, calls):
    """The drafter's collectives replaced by their one-rank stand-ins, each call recorded as (op, rows) or, for the
    sandwich, (op, rows, swiglu)."""

    def fusion(input, *args, **kwargs):
        calls.append(("mnnvl_fusion_allreduce", input.shape[0]))
        return _fusion_allreduce_one_rank(input, *args, **kwargs)

    def sandwich(x, *args, swiglu=False):
        calls.append(("k3_sandwich_plain", x.shape[0], swiglu))
        return _sandwich_plain_one_rank(x, *args, swiglu=swiglu)

    monkeypatch.setattr(decode_comm, "mnnvl_fusion_allreduce", fusion)
    monkeypatch.setattr(decode_comm, "k3_sandwich_plain", sandwich)


@pytest.fixture(scope="module")
def fused_drafter():
    """The drafter on the one-rank collective state: its fc split over one rank, both sandwich forms compiled (their
    compile calls run on the stand-in)."""
    with pytest.MonkeyPatch.context() as mp:
        _one_rank_collectives(mp, [])
        module = _load_drafter()
        module.use_decode_comm(ONE_RANK_COMM)
    return module


@pytest.fixture
def collectives(monkeypatch):
    """The one-rank stand-ins of the drafter's collectives, counted: the list of (op, rows) calls."""
    calls = []
    _one_rank_collectives(monkeypatch, calls)
    return calls


@pytest.fixture(scope="module")
def gemvs():
    return decode_gemv.K3DecodeGemvs.create(None, sites=SITES)


def _block(batch, block, seed):
    """A block's inputs: noise rows, query positions after each request's context, and a paged context cache
    (random rows, the requests' pages scattered) with its page table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor(CTX[:batch], dtype=torch.int32, device="cuda")
    width = math.ceil((int(ctx.max()) + block) / PAGE) + 1
    pages = batch * width + 3
    order = torch.randperm(pages, generator=g, device="cuda")[: batch * width]
    caches = [
        (torch.randn(pages, 2, KV_HEADS, PAGE, HEAD_DIM, generator=g, device="cuda")).to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    return dict(
        noise_embedding=torch.randn(batch, block, HIDDEN, generator=g, device="cuda").to(
            torch.bfloat16
        ),
        query_positions=(ctx.long()[:, None] + torch.arange(block, device="cuda")),
        num_ctx_per_req=ctx,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, dtype=torch.int32, device="cuda"),
        ctx_kv_cache=caches,
        ctx_page_table=order.view(batch, width).to(torch.int32),
    )


def _copy(inputs):
    out = dict(inputs)
    out["ctx_kv_cache"] = [c.clone() for c in inputs["ctx_kv_cache"]]
    return out


def _rel_l2(y, ref):
    return ((y.double() - ref.double()).norm() / ref.double().norm()).item()


@pytest.fixture
def attn_calls(monkeypatch):
    """Counts the drafter's calls of the attention entry."""
    calls = []
    entry = target.k3_drafter_attn_qknorm

    def counted(*args):
        calls.append(args[0].shape[0])
        return entry(*args)

    monkeypatch.setattr(target, "k3_drafter_attn_qknorm", counted)
    return calls


@pytest.mark.parametrize("use_gemvs", [False, True], ids=["torch", "gemv"])
@pytest.mark.parametrize("split", sorted(target.DRAFTER_ATTN_SPLITS))
def test_block_matches_the_stock_forward(
    drafter, gemvs, attn_calls, collectives, monkeypatch, use_gemvs, split
):
    batch, block = split
    projected = []
    project = gemvs.project

    def recorded(site, x, weight):
        y = project(site, x, weight)
        projected.append((site, y is not None))
        return y

    monkeypatch.setattr(gemvs, "project", recorded)
    drafter.decode_gemvs = gemvs if use_gemvs else None
    inputs = _block(batch, block, seed=batch * 10 + block)
    stock_inputs = _copy(inputs)
    before = [c.clone() for c in inputs["ctx_kv_cache"]]
    out = drafter.dflash_forward(**inputs)
    ref = DFlashForCausalLM.dflash_forward(drafter, **stock_inputs)
    torch.cuda.synchronize()
    assert attn_calls == [batch * block] * LAYERS
    assert out.shape == ref.shape == (batch * block, HIDDEN)
    err = _rel_l2(out, ref)
    assert err <= REL_L2, err
    assert all(torch.equal(c, b) for c, b in zip(inputs["ctx_kv_cache"], before))
    # Without the TP group's collective state the module all-reduces and the stock norms run.
    assert collectives == []
    if use_gemvs:
        # At most 8 rows every site takes its projection; above, each declines and its module runs (the down
        # projection's site is not asked once gate / up's declined).
        takes = batch * block <= decode_gemv.MAX_ROWS
        asked = SITES if takes else SITES[:3]
        assert sorted(projected) == sorted(
            (site, takes) for site in asked for _ in range(LAYERS)
        ), projected
    else:
        assert projected == []


def test_an_uncertified_split_runs_the_stock_forward(drafter, attn_calls):
    drafter.decode_gemvs = None
    batch, block = 4, 4
    assert (batch, block) not in target.DRAFTER_ATTN_SPLITS
    inputs = _block(batch, block, seed=44)
    stock_inputs = _copy(inputs)
    out = drafter.dflash_forward(**inputs)
    ref = DFlashForCausalLM.dflash_forward(drafter, **stock_inputs)
    assert attn_calls == []
    assert torch.equal(out, ref)


def test_capture_takes_only_compiled_keys(drafter):
    inputs = _block(1, 8, seed=18)
    args = (inputs["noise_embedding"], inputs["ctx_kv_cache"], inputs["ctx_page_table"])
    drafter._k3_attn_ran.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refused = drafter._k3_block_keys(*args)
    assert refused is None
    drafter.dflash_forward(**inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        taken = drafter._k3_block_keys(*args)
    assert taken is not None and taken <= drafter._k3_attn_ran


def test_negative_control_a_changed_weight_fails(drafter):
    drafter.decode_gemvs = None
    inputs = _block(1, 8, seed=81)
    stock_inputs = _copy(inputs)
    weight = drafter.model.layers[0].self_attn.o_proj.weight
    saved = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.add_(torch.randn_like(weight) * 0.02)
        out = drafter.dflash_forward(**inputs)
    finally:
        with torch.no_grad():
            weight.copy_(saved)
    ref = DFlashForCausalLM.dflash_forward(drafter, **stock_inputs)
    assert _rel_l2(out, ref) > REL_L2


@pytest.mark.parametrize("split", [(1, 7), (3, 1), (8, 7)])
def test_page_table_rows_as_a_view(drafter, gemvs, monkeypatch, split):
    batch, block = split
    start = 2
    tables = []
    entry = target.k3_drafter_attn_qknorm

    def recorded(*args):
        tables.append(args[7])  # the page table
        return entry(*args)

    monkeypatch.setattr(target, "k3_drafter_attn_qknorm", recorded)
    drafter.decode_gemvs = gemvs
    inputs = _block(batch, block, seed=batch * 10 + block + 2)
    # The batch's rows sit at [start, start + batch) of a larger table; the other rows hold no page.
    table = inputs["ctx_page_table"]
    padded = torch.full((start + batch + 1, table.shape[1]), -1, dtype=torch.int32, device="cuda")
    padded[start : start + batch] = table
    inputs["ctx_page_table"] = padded
    inputs["ctx_cache_batch_idx"] = torch.arange(
        start, start + batch, dtype=torch.long, device="cuda"
    )
    gathered = drafter.dflash_forward(**_copy(inputs))
    viewed = drafter.dflash_forward(**inputs, ctx_rows_start=start)
    torch.cuda.synchronize()
    assert torch.equal(viewed, gathered)
    assert len(tables) == 2 * LAYERS and all(torch.equal(t, table) for t in tables)
    # The gather's rows are a copy; the view's are the table's own rows.
    assert all(t.data_ptr() != padded[start].data_ptr() for t in tables[:LAYERS])
    assert all(t.data_ptr() == padded[start].data_ptr() for t in tables[LAYERS:])


@pytest.mark.parametrize("rows", [1, 8, DECODE_ROWS, 200])
def test_split_fc_matches_the_replicated_projection(drafter, fused_drafter, collectives, rows):
    """On one rank the block is the whole fc. Up to a decode step's rows the projection's all-reduce applies
    hidden_norm; above, the drafter's TP all-reduce (the identity on one rank) runs and then hidden_norm."""
    fc = fused_drafter.fc
    assert isinstance(fc, target.K3FcSlice) and (fc.start, fc.end) == (0, 2 * HIDDEN)
    assert torch.equal(fc.weight, drafter.fc.weight)
    g = torch.Generator(device="cuda").manual_seed(rows)
    features = torch.randn(rows, 2 * HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    out = fused_drafter.project_target_hidden(features)
    ref = drafter.project_target_hidden(features)
    torch.cuda.synchronize()
    assert out.shape == ref.shape == (rows, HIDDEN)
    err = _rel_l2(out, ref)
    assert err <= REL_L2, err
    assert collectives == ([("mnnvl_fusion_allreduce", rows)] if rows <= DECODE_ROWS else [])


def test_fused_drafter_compiled_both_sandwich_forms(fused_drafter):
    assert fused_drafter.decode_comm is ONE_RANK_COMM
    assert fused_drafter._k3_norms_fuse
    assert fused_drafter._k3_sandwich_forms == {"o_proj", "down"}


def _fused_calls(rows, use_gemvs):
    """One layer's collective calls on the fused path: up to 8 rows the o_proj sandwich, and the down projection's
    sandwich where the gate / up site produced its input; else the projection then the fused all-reduce."""
    if rows > decode_gemv.MAX_ROWS:
        return [("mnnvl_fusion_allreduce", rows)] * 2
    down = ("k3_sandwich_plain", rows, True) if use_gemvs else ("mnnvl_fusion_allreduce", rows)
    return [("k3_sandwich_plain", rows, False), down]


@pytest.mark.parametrize("use_gemvs", [False, True], ids=["torch", "gemv"])
@pytest.mark.parametrize("split", sorted(target.DRAFTER_ATTN_SPLITS))
def test_fused_block_matches_the_stock_forward(
    fused_drafter, gemvs, attn_calls, collectives, use_gemvs, split
):
    batch, block = split
    rows = batch * block
    fused_drafter.decode_gemvs = gemvs if use_gemvs else None
    inputs = _block(batch, block, seed=batch * 10 + block + 1)
    stock_inputs = _copy(inputs)
    before = [c.clone() for c in inputs["ctx_kv_cache"]]
    noise = inputs["noise_embedding"].clone()
    out = fused_drafter.dflash_forward(**inputs)
    calls = list(collectives)
    ref = DFlashForCausalLM.dflash_forward(fused_drafter, **stock_inputs)
    torch.cuda.synchronize()
    assert attn_calls == [rows] * LAYERS
    assert calls == _fused_calls(rows, use_gemvs) * LAYERS
    assert out.shape == ref.shape == (rows, HIDDEN)
    err = _rel_l2(out, ref)
    assert err <= REL_L2, err
    # The block's input served as the first residual, read only; the cache is untouched.
    assert torch.equal(inputs["noise_embedding"], noise)
    assert all(torch.equal(c, b) for c, b in zip(inputs["ctx_kv_cache"], before))


def _stock_norms_case(fused_drafter, monkeypatch, case):
    if case == "workspace":
        small = decode_comm.K3DecodeComm(
            mnnvl=SimpleNamespace(world_size=1, buffer_bytes=1024),
            sandwich=ONE_RANK_COMM.sandwich,
        )
        monkeypatch.setattr(fused_drafter, "decode_comm", small)
    else:
        monkeypatch.setattr(fused_drafter, "_k3_norms_fuse", False)


@pytest.mark.parametrize("case", ["workspace", "norms"])
def test_fused_norms_fall_back_to_the_stock_norms(
    fused_drafter, gemvs, attn_calls, collectives, monkeypatch, case
):
    """A block whose rows the MNNVL workspace does not hold, or layers whose norms the fused all-reduces do not
    reproduce, keep the module all-reduces and the stock norms."""
    _stock_norms_case(fused_drafter, monkeypatch, case)
    fused_drafter.decode_gemvs = gemvs
    inputs = _block(1, 7, seed=17)
    stock_inputs = _copy(inputs)
    out = fused_drafter.dflash_forward(**inputs)
    ref = DFlashForCausalLM.dflash_forward(fused_drafter, **stock_inputs)
    torch.cuda.synchronize()
    assert collectives == []
    assert attn_calls == [7] * LAYERS
    err = _rel_l2(out, ref)
    assert err <= REL_L2, err


def test_uncompiled_sandwich_gives_way_to_the_fused_all_reduce(
    fused_drafter, gemvs, collectives, monkeypatch
):
    monkeypatch.setattr(fused_drafter, "_k3_sandwich_forms", frozenset({"down"}))
    fused_drafter.decode_gemvs = gemvs
    inputs = _block(1, 8, seed=88)
    stock_inputs = _copy(inputs)
    out = fused_drafter.dflash_forward(**inputs)
    ref = DFlashForCausalLM.dflash_forward(fused_drafter, **stock_inputs)
    torch.cuda.synchronize()
    assert collectives == [("mnnvl_fusion_allreduce", 8), ("k3_sandwich_plain", 8, True)] * LAYERS
    err = _rel_l2(out, ref)
    assert err <= REL_L2, err


@pytest.mark.parametrize("split", [(1, 7), (2, 7)])
def test_fused_block_replays_under_capture(fused_drafter, gemvs, collectives, split):
    """Eager first (the attention compiles for the block's key), then captured and replayed."""
    fused_drafter.decode_gemvs = gemvs
    inputs = _block(*split, seed=70 + split[0])
    eager = fused_drafter.dflash_forward(**_copy(inputs))
    torch.cuda.synchronize()
    eager_calls = list(collectives)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_drafter.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    assert eager_calls == _fused_calls(split[0] * split[1], True) * LAYERS
    assert collectives[len(eager_calls) :] == eager_calls
    err = _rel_l2(captured, eager)
    assert err <= 1e-3, err
