# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's DSpark drafter (``K3DSparkDrafter`` of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``) on one GPU,
against Modeling V1's stock GQA DSpark drafter (``modeling_dspark.GQADSparkForCausalLM``), both loaded from the same
draft checkpoint, at one TP16 rank's shapes, two layers, 64-row context pages.

Two drafter configs:

* ``entries``: 6 query heads and 1 KV head of 64, plain NeoX RoPE, the layout the drafter entries take.
* ``radixark``: 4 / 1 heads of 64 with YaRN RoPE (RadixArk-Kimi-K3-DSpark's at TP16), which the entries never take.

Bit for bit against the stock drafter (the same kernels in the same order):

* The weight load: every tensor (this rank's projections, norms, ``fc`` / ``hidden_norm``, the Markov and confidence
  heads, the trained mask row).
* The RoPE cos / sin cache, the fused context K / V weights and ``precompute_context_kv``.
* The replicated ``project_target_hidden``.
* The generic block on TRTLLM (uncertified splits of the ``entries`` drafter, every split of the ``radixark``
  drafter): the output and the block's K / V the call leaves in the paged cache.
* ``attention_backend`` VANILLA (flash-attn over the worker's private arena): the block of both drafters, every split,
  the output and the block's K / V left in the arena; the context K / V.

The drafter entries and the TP group's fused collectives (one-rank stand-ins, counted) run kernels the stock drafter
does not: against its block forward within a bound, with the calls each path makes checked, the split context
projection within a bound of the replicated one; the cache untouched; CUDA-graph capture taking only compiled keys;
captured blocks replaying the eager result; a changed weight failing the comparison; ``ctx_rows_start`` reading
page-table rows as a view.
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
    drafter as target,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the K3 drafter entries run on sm_100 and sm_103",
)

HIDDEN, INTER, HEAD_DIM, LAYERS, PAGE = 7168, 896, 64, 2, 64
VOCAB, RANK = 1024, 16
CONFIGS = {"entries": (6, 1, None), "radixark": (4, 1, "yarn")}
# Context lengths per request: inside a page, on a page boundary, across pages, empty.
CTX = (37, 64, 130, 0, 5, 200, 63, 1)
REL_L2 = 1e-2
SITES = ("drafter_qkv", "drafter_o", "drafter_gate_up", "drafter_down")


def _config(name):
    heads, kv_heads, rope = CONFIGS[name]
    return Qwen3Config.from_dict(
        dict(
            architectures=["DSparkDraftModel"],
            model_type="qwen3",
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_hidden_layers=LAYERS,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=HEAD_DIM,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rope_scaling=(
                None
                if rope is None
                else {
                    "rope_type": "yarn",
                    "factor": 16.0,
                    "original_max_position_embeddings": 256,
                }
            ),
            attention_bias=False,
            torch_dtype="bfloat16",
            tie_word_embeddings=False,
            markov_rank=RANK,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            dflash_config={"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]},
        )
    )


def _weights(name, seed=11):
    heads, kv_heads, _ = CONFIGS[name]
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {
        "fc.weight": rnd(HIDDEN, 2 * HIDDEN),
        "hidden_norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "embed_tokens.weight": rnd(VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_head.proj.weight": rnd(1, HIDDEN + RANK),
        "confidence_head.proj.bias": rnd(1),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(heads * HEAD_DIM, HIDDEN)
        w[p + "self_attn.k_proj.weight"] = rnd(kv_heads * HEAD_DIM, HIDDEN)
        w[p + "self_attn.v_proj.weight"] = rnd(kv_heads * HEAD_DIM, HIDDEN)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, heads * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = rnd(HEAD_DIM, scale=0.05) + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(HEAD_DIM, scale=0.05) + 1.0
        w[p + "input_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTER)
    return w


def _load(cls, name, backend="TRTLLM"):
    model_config = ModelConfig(pretrained_config=_config(name), attn_backend="TRTLLM")
    module = cls(model_config, dflash_attention_backend=backend).to("cuda")
    module.load_weights(dict(_weights(name)))
    return module


def _pair(name, backend="TRTLLM"):
    return SimpleNamespace(
        new=_load(target.K3DSparkDrafter, name, backend),
        old=_load(GQADSparkForCausalLM, name, backend),
    )


@pytest.fixture(scope="module")
def drafters():
    """Both drafters of both configs."""
    pairs = {name: _pair(name) for name in CONFIGS}
    assert pairs["entries"].new._k3_layers_take()
    assert not pairs["radixark"].new._k3_layers_take()
    return pairs


@pytest.fixture(scope="module")
def drafter(drafters):
    """The new ``entries`` drafter (the decode entries take it)."""
    return drafters["entries"].new


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
    """The drafters' collectives replaced by their one-rank stand-ins, each call recorded as (op, rows) or, for the
    sandwich, (op, rows, swiglu)."""

    def fusion(input, *args, **kwargs):
        calls.append(("mnnvl_fusion_allreduce", input.shape[0]))
        return _fusion_allreduce_one_rank(input, *args, **kwargs)

    def sandwich(x, *args, swiglu=False):
        calls.append(("k3_sandwich_plain", x.shape[0], swiglu))
        return _sandwich_plain_one_rank(x, *args, swiglu=swiglu)

    monkeypatch.setattr(decode_comm, "mnnvl_fusion_allreduce", fusion)
    monkeypatch.setattr(decode_comm, "k3_sandwich_plain", sandwich)


class _OneRankTPAllReduce(torch.nn.Module):
    """The TP all-reduce of a group of one rank: the identity."""

    def forward(self, input, all_reduce_params=None):
        return input

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return False


def _one_rank_tp_all_reduce(new, old):
    """Every layer's o_proj and down projection reduce over the one-rank group, as they reduce over the TP group on
    more ranks (one rank builds them without an all-reduce). The fused path keys on that all-reduce."""
    for layer in new.model.layers:
        layer.self_attn.o_proj.all_reduce = _OneRankTPAllReduce()
        layer.mlp.down_proj.all_reduce = _OneRankTPAllReduce()
    for layer in old.model.layers:
        o_proj = layer.self_attn.o_proj
        o_proj.reduce_output = True
        o_proj.all_reduce = _OneRankTPAllReduce()


@pytest.fixture(scope="module")
def fused():
    """Both ``entries`` drafters on the one-rank collective state: fc split over one rank, both sandwich forms
    compiled (their compile calls run on the stand-in)."""
    with pytest.MonkeyPatch.context() as mp:
        _one_rank_collectives(mp, [])
        pair = _pair("entries")
        _one_rank_tp_all_reduce(pair.new, pair.old)
        pair.new.use_decode_comm(ONE_RANK_COMM)
    return pair


@pytest.fixture
def collectives(monkeypatch):
    """The one-rank stand-ins of the drafters' collectives, counted: the list of (op, rows) calls."""
    calls = []
    _one_rank_collectives(monkeypatch, calls)
    return calls


@pytest.fixture(scope="module")
def gemvs():
    return decode_gemv.K3DecodeGemvs.create(None, sites=SITES)


def _block(batch, block, seed, kv_heads=1):
    """A block's inputs: noise rows, query positions after each request's context, and a paged context cache
    (random rows, the requests' pages scattered) with its page table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor([CTX[i % len(CTX)] for i in range(batch)], dtype=torch.int32, device="cuda")
    width = math.ceil((int(ctx.max()) + block) / PAGE) + 1
    pages = batch * width + 3
    order = torch.randperm(pages, generator=g, device="cuda")[: batch * width]
    caches = [
        (torch.randn(pages, 2, kv_heads, PAGE, HEAD_DIM, generator=g, device="cuda")).to(
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


def _assert_bitwise(new, old, what):
    assert new.shape == old.shape and new.dtype == old.dtype, f"{what}: {new.shape} vs {old.shape}"
    if not torch.equal(new, old):
        diff = (new.float() - old.float()).abs()
        raise AssertionError(
            f"{what}: {int((new != old).sum())} of {new.numel()} differ, max |diff| {float(diff.max())}"
        )


@pytest.fixture
def attn_calls(monkeypatch):
    """Counts the new drafter's calls of the attention entry."""
    calls = []
    entry = target.k3_drafter_attn_qknorm

    def counted(*args):
        calls.append(args[0].shape[0])
        return entry(*args)

    monkeypatch.setattr(target, "k3_drafter_attn_qknorm", counted)
    return calls


# ----------------------------------------------------------------------------------------------------------------------
# The weight load and the context side.
# ----------------------------------------------------------------------------------------------------------------------

_PARAMS = (
    "self_attn.qkv_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_weight_load_matches_stock(drafters, name):
    new, old = drafters[name].new, drafters[name].old
    # Every alias: the stock drafter registers its backbone first as draft_model_full.model, the new one as model.
    old_params = dict(old.named_parameters(remove_duplicate=False))
    new_params = dict(new.named_parameters(remove_duplicate=False))
    for i in range(LAYERS):
        for param in _PARAMS:
            key = f"model.layers.{i}.{param}"
            _assert_bitwise(new_params[key], old_params[key], key)
    for key in ("model.norm.weight", "fc.weight", "hidden_norm.weight"):
        _assert_bitwise(new_params[key], old_params[key], key)
    for attr in ("markov_w1", "markov_w2", "confidence_proj_weight", "confidence_proj_bias"):
        _assert_bitwise(getattr(new, attr), getattr(old, attr), attr)
    _assert_bitwise(new.mask_token_embedding, old.mask_token_embedding, "mask_token_embedding")
    assert new.hidden_norm.eps == old.hidden_norm.eps
    for field in ("mask_token_id", "target_layer_ids", "block_size", "_dspark_shift_label"):
        assert getattr(new, field) == getattr(old, field), field
    assert new.has_markov_head and old.has_markov_head
    assert (new.logit_multiplier, new.logit_softcap) == (old.logit_multiplier, old.logit_softcap)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_rope_cache_and_context_kv_match_stock(drafters, name):
    new, old = drafters[name].new, drafters[name].old
    _assert_bitwise(new._get_cos_sin_cache(), old._get_cos_sin_cache(), "cos / sin cache")
    old._build_fused_kv_buffers()
    new._build_fused_kv_buffers()
    _assert_bitwise(new._fused_kv_weight, old._fused_kv_weight, "fused K / V weight")
    _assert_bitwise(new._k_norm_stacked, old._k_norm_stacked, "stacked k_norm")
    assert new._k_norm_eps == old._k_norm_eps
    assert new._fused_kv_bias is None and old._fused_kv_bias is None
    assert new._input_ln_eps is None and old._input_ln_eps is None
    assert new._is_neox and old._is_neox
    for field in ("_num_attn_layers", "_num_heads", "_num_kv_heads", "_head_dim", "_kv_factor"):
        assert getattr(new, field) == getattr(old, field), field
    assert new._use_fused_qk_norm_rope == old._use_fused_qk_norm_rope == (name == "entries")
    g = torch.Generator(device="cuda").manual_seed(3)
    for rows in (1, 7, 300):
        projected = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        positions = torch.randint(0, 4000, (rows,), generator=g, device="cuda")
        k_new, v_new = new.precompute_context_kv(projected, positions)
        k_old, v_old = old.precompute_context_kv(projected, positions)
        _assert_bitwise(k_new, k_old, f"context K, {rows} rows")
        _assert_bitwise(v_new, v_old, f"context V, {rows} rows")


@pytest.mark.parametrize("rows", [1, 8, DECODE_ROWS, 200])
def test_projection_matches_stock(drafters, fused, collectives, rows):
    """Replicated: the stock projection, bit for bit. Split over one rank: the fused all-reduce up to a decode step's
    rows, else the TP all-reduce, within a bound of the stock replicated projection (the stand-in norm rounds once)."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    features = torch.randn(rows, 2 * HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    pair = drafters["entries"]
    _assert_bitwise(
        pair.new.project_target_hidden(features),
        pair.old.project_target_hidden(features),
        "replicated projection",
    )
    assert collectives == []
    fc = fused.new.fc
    assert isinstance(fc, target.K3FcSlice) and (fc.start, fc.end) == (0, 2 * HIDDEN)
    out = fused.new.project_target_hidden(features)
    ref = fused.old.project_target_hidden(features)
    assert collectives == ([("mnnvl_fusion_allreduce", rows)] if rows <= DECODE_ROWS else [])
    assert _rel_l2(out, ref) <= REL_L2


def test_fused_drafter_compiled_both_sandwich_forms(fused):
    assert fused.new.decode_comm is ONE_RANK_COMM and fused.new._k3_norms_fuse
    assert fused.new._k3_sandwich_forms == {"o_proj", "down"}


def test_worker_fields(drafters):
    """Every name the speculative worker reads off a drafter, on the new drafter. The stock drafter may lack one the
    worker reads with a default (``getattr(draft_model, "lilicorr", None)``); where it has it, the values agree."""
    for pair in drafters.values():
        for name in target.WORKER_FIELDS:
            assert hasattr(pair.new, name), name
        for name in (
            "lilicorr",
            "is_dflash2",
            "has_block_conv",
            "has_candidate_selector",
            "_paged_ctx_cache",
        ):
            if hasattr(pair.old, name):
                assert getattr(pair.new, name) == getattr(pair.old, name), name
        assert pair.new.dflash_attention_backend == pair.old.dflash_attention_backend == "TRTLLM"
        assert pair.new.validate_block_attention_windows() == (
            pair.old.validate_block_attention_windows()
        )


# ----------------------------------------------------------------------------------------------------------------------
# The block forward.
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("use_gemvs", [False, True], ids=["torch", "gemv"])
@pytest.mark.parametrize("split", sorted(target.DRAFTER_ATTN_SPLITS))
def test_entries_within_the_stock_bound(
    drafters, gemvs, attn_calls, collectives, monkeypatch, use_gemvs, split
):
    """A certified split on the drafter entries (one attention launch per layer): within a bound of the stock block
    forward; the GEMV sites take at most 8 rows; the cache is untouched."""
    pair = drafters["entries"]
    batch, block = split
    projected = []
    project = gemvs.project

    def recorded(site, x, weight):
        y = project(site, x, weight)
        projected.append((site, y is not None))
        return y

    monkeypatch.setattr(gemvs, "project", recorded)
    pair.new.decode_gemvs = gemvs if use_gemvs else None
    inputs = _block(batch, block, seed=batch * 10 + block)
    stock_inputs = _copy(inputs)
    before = [c.clone() for c in inputs["ctx_kv_cache"]]
    out = pair.new.dflash_forward(**inputs)
    new_projected = list(projected)
    stock = pair.old.dflash_forward(**stock_inputs)
    torch.cuda.synchronize()
    assert attn_calls == [batch * block] * LAYERS
    assert _rel_l2(out, stock) <= REL_L2
    assert all(torch.equal(c, b) for c, b in zip(inputs["ctx_kv_cache"], before))
    assert collectives == []
    if use_gemvs:
        takes = batch * block <= decode_gemv.MAX_ROWS
        asked = SITES if takes else SITES[:3]
        assert sorted(new_projected) == sorted(
            (site, takes) for site in asked for _ in range(LAYERS)
        ), new_projected
    else:
        assert projected == []


_GENERIC_SPLITS = {
    # Splits DRAFTER_ATTN_SPLITS does not list (it lists every (r, 7) up to r = 8).
    "entries": [(4, 4), (3, 5), (5, 2), (12, 7)],
    "radixark": [(1, 1), (1, 7), (2, 7), (8, 7), (8, 8), (3, 4), (16, 7)],
}


@pytest.mark.parametrize(
    "name,split",
    [(name, split) for name, splits in _GENERIC_SPLITS.items() for split in splits],
    ids=lambda v: str(v),
)
def test_generic_block_matches_stock(drafters, attn_calls, name, split):
    """The generic block (a split the entries do not take, or a drafter they never take): the stock DFlash block
    forward, bit for bit, both in the output and in the block's K / V it leaves in the context pages."""
    pair = drafters[name]
    for d in (pair.new, pair.old):
        d.decode_gemvs = None
    batch, block = split
    kv_heads = CONFIGS[name][1]
    inputs = _block(batch, block, seed=batch * 10 + block + 5, kv_heads=kv_heads)
    old_inputs = _copy(inputs)
    out = pair.new.dflash_forward(**inputs)
    ref = pair.old.dflash_forward(**old_inputs)
    torch.cuda.synchronize()
    assert attn_calls == []
    _assert_bitwise(out, ref, f"{name} generic block {split}")
    for layer, (c_new, c_old) in enumerate(zip(inputs["ctx_kv_cache"], old_inputs["ctx_kv_cache"])):
        _assert_bitwise(c_new, c_old, f"{name} context pages of layer {layer} after {split}")


def test_generic_block_replays_under_capture(drafters):
    """The generic block captured after an eager call of its shape replays the eager result."""
    pair = drafters["radixark"]
    pair.new.decode_gemvs = None
    inputs = _block(2, 7, seed=91)
    eager = pair.new.dflash_forward(**_copy(inputs))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pair.new.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, eager, "replayed generic block")


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


def test_negative_control_a_changed_weight_fails(drafters):
    pair = drafters["entries"]
    for d in (pair.new, pair.old):
        d.decode_gemvs = None
    inputs = _block(1, 8, seed=81)
    old_inputs = _copy(inputs)
    weight = pair.new.model.layers[0].self_attn.o_proj.weight
    saved = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.add_(torch.randn_like(weight) * 0.02)
        out = pair.new.dflash_forward(**inputs)
    finally:
        with torch.no_grad():
            weight.copy_(saved)
    ref = pair.old.dflash_forward(**old_inputs)
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


def _fused_calls(rows, use_gemvs):
    """One layer's collective calls on the fused path: up to 8 rows the o_proj sandwich, and the down projection's
    sandwich where the gate / up site produced its input; else the projection then the fused all-reduce."""
    if rows > decode_gemv.MAX_ROWS:
        return [("mnnvl_fusion_allreduce", rows)] * 2
    down = ("k3_sandwich_plain", rows, True) if use_gemvs else ("mnnvl_fusion_allreduce", rows)
    return [("k3_sandwich_plain", rows, False), down]


@pytest.mark.parametrize("use_gemvs", [False, True], ids=["torch", "gemv"])
@pytest.mark.parametrize("split", sorted(target.DRAFTER_ATTN_SPLITS))
def test_fused_block_within_the_stock_bound(
    fused, gemvs, attn_calls, collectives, use_gemvs, split
):
    """A certified split on the entries with the residual adds and norms in the all-reduces: the expected collective
    calls, within a bound of the stock block forward."""
    batch, block = split
    rows = batch * block
    fused.new.decode_gemvs = gemvs if use_gemvs else None
    inputs = _block(batch, block, seed=batch * 10 + block + 1)
    stock_inputs = _copy(inputs)
    before = [c.clone() for c in inputs["ctx_kv_cache"]]
    noise = inputs["noise_embedding"].clone()
    out = fused.new.dflash_forward(**inputs)
    calls = list(collectives)
    stock = fused.old.dflash_forward(**stock_inputs)
    torch.cuda.synchronize()
    assert attn_calls == [rows] * LAYERS
    assert calls == _fused_calls(rows, use_gemvs) * LAYERS
    assert _rel_l2(out, stock) <= REL_L2
    # The block's input served as the first residual, read only; the cache is untouched.
    assert torch.equal(inputs["noise_embedding"], noise)
    assert all(torch.equal(c, b) for c, b in zip(inputs["ctx_kv_cache"], before))


@pytest.mark.parametrize("case", ["workspace", "norms"])
def test_fused_norms_fall_back_to_the_module_norms(fused, gemvs, collectives, monkeypatch, case):
    """A block whose rows the MNNVL workspace does not hold, or layers whose norms the fused all-reduces do not
    reproduce, keep the module all-reduces and norms (no fused call), within a bound of the stock block forward."""
    d = fused.new
    if case == "workspace":
        small = decode_comm.K3DecodeComm(
            mnnvl=SimpleNamespace(world_size=1, buffer_bytes=1024),
            sandwich=ONE_RANK_COMM.sandwich,
        )
        monkeypatch.setattr(d, "decode_comm", small)
    else:
        monkeypatch.setattr(d, "_k3_norms_fuse", False)
    d.decode_gemvs = gemvs
    inputs = _block(1, 7, seed=17)
    stock_inputs = _copy(inputs)
    out = d.dflash_forward(**inputs)
    stock = fused.old.dflash_forward(**stock_inputs)
    torch.cuda.synchronize()
    assert collectives == []
    assert _rel_l2(out, stock) <= REL_L2


def test_uncompiled_sandwich_gives_way_to_the_fused_all_reduce(
    fused, gemvs, collectives, monkeypatch
):
    monkeypatch.setattr(fused.new, "_k3_sandwich_forms", frozenset({"down"}))
    fused.new.decode_gemvs = gemvs
    inputs = _block(1, 8, seed=88)
    stock_inputs = _copy(inputs)
    out = fused.new.dflash_forward(**inputs)
    calls = list(collectives)
    stock = fused.old.dflash_forward(**stock_inputs)
    torch.cuda.synchronize()
    assert calls == [("mnnvl_fusion_allreduce", 8), ("k3_sandwich_plain", 8, True)] * LAYERS
    assert _rel_l2(out, stock) <= REL_L2


@pytest.mark.parametrize("split", [(1, 7), (2, 7)])
def test_fused_block_replays_under_capture(fused, gemvs, collectives, split):
    """Eager first (the attention compiles for the block's key), then captured and replayed."""
    fused.new.decode_gemvs = gemvs
    inputs = _block(*split, seed=70 + split[0])
    eager = fused.new.dflash_forward(**_copy(inputs))
    torch.cuda.synchronize()
    eager_calls = list(collectives)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused.new.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    assert eager_calls == _fused_calls(split[0] * split[1], True) * LAYERS
    assert collectives[len(eager_calls) :] == eager_calls
    err = _rel_l2(captured, eager)
    assert err <= 1e-3, err


# ----------------------------------------------------------------------------------------------------------------------
# attention_backend VANILLA: the worker's private context arena, flash-attn.
# ----------------------------------------------------------------------------------------------------------------------

_VANILLA_REASON = target._dense_attn_op.unavailability_reason()
_needs_flash_attn = pytest.mark.skipif(
    _VANILLA_REASON is not None, reason=f"VANILLA needs flash-attn: {_VANILLA_REASON}"
)


@pytest.fixture(scope="module")
def vanilla_drafters():
    """Both drafters of both configs on VANILLA."""
    return {name: _pair(name, "VANILLA") for name in CONFIGS}


def _arena_block(batch, block, seed, kv_heads=1):
    """A block's inputs on the private arena: noise rows, positions after each request's context, and the arena
    ([slots, layers, columns, KV, head_dim] K and V, random) with the requests' slots scattered."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor([CTX[i % len(CTX)] for i in range(batch)], dtype=torch.int32, device="cuda")
    slots = batch + 3
    columns = int(ctx.max()) + block + 5
    shape = (slots, LAYERS, columns, kv_heads, HEAD_DIM)
    return dict(
        noise_embedding=torch.randn(batch, block, HIDDEN, generator=g, device="cuda").to(
            torch.bfloat16
        ),
        query_positions=(ctx.long()[:, None] + torch.arange(block, device="cuda")),
        num_ctx_per_req=ctx,
        ctx_k_cache=torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16),
        ctx_v_cache=torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16),
        ctx_cache_batch_idx=torch.randperm(slots, generator=g, device="cuda")[:batch].to(
            torch.int32
        ),
        ctx_kv_cache=None,
        ctx_page_table=None,
    )


def _arena_copy(inputs):
    out = dict(inputs)
    out["ctx_k_cache"] = inputs["ctx_k_cache"].clone()
    out["ctx_v_cache"] = inputs["ctx_v_cache"].clone()
    return out


@_needs_flash_attn
def test_vanilla_drafters_resolve_like_stock(vanilla_drafters):
    for pair in vanilla_drafters.values():
        assert pair.new.dflash_attention_backend == pair.old.dflash_attention_backend == "VANILLA"
        assert (
            pair.new.validate_block_attention_windows()
            == pair.old.validate_block_attention_windows()
            == []
        )
        assert pair.new._paged_ctx_cache == pair.old._paged_ctx_cache


_VANILLA_SPLITS = {
    # The entries' certified splits too: they take TRTLLM only, so every VANILLA block is the generic one.
    "entries": [(1, 7), (8, 7), (2, 8), (4, 4), (12, 7)],
    "radixark": [(1, 1), (1, 7), (2, 7), (8, 7), (8, 8), (3, 4), (16, 7)],
}


@_needs_flash_attn
@pytest.mark.parametrize(
    "name,split",
    [(name, split) for name, splits in _VANILLA_SPLITS.items() for split in splits],
    ids=lambda v: str(v),
)
def test_vanilla_block_matches_stock(vanilla_drafters, attn_calls, name, split):
    """VANILLA: the stock DFlash block forward on its arena path, bit for bit, both in the output and in the
    block's K / V it leaves in the arena; the drafter entries never run."""
    pair = vanilla_drafters[name]
    batch, block = split
    inputs = _arena_block(batch, block, seed=batch * 10 + block + 7, kv_heads=CONFIGS[name][1])
    old_inputs = _arena_copy(inputs)
    out = pair.new.dflash_forward(**inputs)
    ref = pair.old.dflash_forward(**old_inputs)
    torch.cuda.synchronize()
    assert attn_calls == []
    _assert_bitwise(out, ref, f"{name} VANILLA block {split}")
    _assert_bitwise(
        inputs["ctx_k_cache"], old_inputs["ctx_k_cache"], f"{name} arena K after {split}"
    )
    _assert_bitwise(
        inputs["ctx_v_cache"], old_inputs["ctx_v_cache"], f"{name} arena V after {split}"
    )


@_needs_flash_attn
def test_vanilla_context_kv_matches_stock(vanilla_drafters):
    pair = vanilla_drafters["radixark"]
    g = torch.Generator(device="cuda").manual_seed(4)
    projected = torch.randn(9, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    positions = torch.randint(0, 4000, (9,), generator=g, device="cuda")
    for new, old in zip(
        pair.new.precompute_context_kv(projected, positions),
        pair.old.precompute_context_kv(projected, positions),
    ):
        _assert_bitwise(new, old, "VANILLA context K / V")


@_needs_flash_attn
def test_vanilla_block_replays_under_capture(vanilla_drafters):
    pair = vanilla_drafters["radixark"]
    inputs = _arena_block(2, 7, seed=93)
    eager = pair.new.dflash_forward(**_arena_copy(inputs))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = pair.new.dflash_forward(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(captured, eager, "replayed VANILLA block")


# RoPE settings a Qwen3-shaped drafter config can declare: the two drafters above, the other scalings the engine's
# table builder serves, and the transformers 5 spelling (``rope_parameters``, also per layer type).
_ROPE_SETTINGS = {
    "plain": dict(rope_scaling=None),
    "yarn": dict(
        rope_scaling={"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 256}
    ),
    "yarn-mscale": dict(
        rope_scaling={
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
    ),
    "linear": dict(rope_scaling={"type": "linear", "factor": 2.0}),
    "llama3": dict(
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    ),
    "rope-parameters": dict(
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 1024,
        }
    ),
    "per-layer-type": dict(
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 50000.0}}
    ),
}


@pytest.mark.parametrize("name", list(_ROPE_SETTINGS))
def test_draft_rope_matches_the_attention_backends_rope_params(name):
    """``DraftRope`` reads the drafter config as ``RopeParams.from_config`` reads it, and its table is the
    non-interleaved ``create_rope_const_params`` table, bit for bit."""
    from tensorrt_llm._torch.attention.backends.interface import RopeParams

    def config():
        cfg = _config("entries")
        for key in ("rope_scaling", "rope_parameters"):
            if hasattr(cfg, key):
                setattr(cfg, key, None)
        for key, value in _ROPE_SETTINGS[name].items():
            setattr(cfg, key, value)
        return cfg

    ours, theirs = target.DraftRope(config()), RopeParams.from_config(config())
    for field in (
        "dim",
        "theta",
        "alpha",
        "scale_type",
        "scale",
        "low_freq_factor",
        "high_freq_factor",
        "max_positions",
        "original_max_positions",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "short_factor",
        "long_factor",
        "max_seq_len",
        "duplicate_data",
    ):
        assert getattr(ours, field) == getattr(theirs, field), field
    table = ours.cos_sin()
    want = theirs.create_rope_const_params(interleave=False)[1]
    assert table.dtype == want.dtype and table.shape == want.shape
    assert torch.equal(table, want)
