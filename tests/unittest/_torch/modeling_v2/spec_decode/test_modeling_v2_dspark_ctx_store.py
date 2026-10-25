# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""catalog/spec_decode/dspark_ctx_store: the one-launch context store writes, bit for bit, what the stock path writes.

Stock = DFlashForCausalLM.precompute_context_kv, the accept mask and DFlashWorker._store_context_kv_paged (flashinfer's
paged append per layer), run on a tiny real drafter (2 layers, 2 kv heads of 128, k_norm), with full RoPE and with
the MiniMax-M3.1 drafter's partial RoPE (rotary_dim 64 of head_dim 128: dims 64..127 are not rotated). Both write into
caches filled with a sentinel, compared in an int16 view, for the private arena (one stacked tensor) and the bound pool
(a list of per-layer tensors), with masked rows, clamped columns and a placeholder page. Cells written more than once
(see _single_writer) are excluded: stock's own value there is a race.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_ctx_store import (
    ctx_store_supported,
    fold_k_norm_weight,
    fused_ctx_store,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.speculative.dflash import DFlashWorker


def _has_trtllm_gen():
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_sm_100f

    return is_sm_100f()


pytestmark = pytest.mark.skipif(
    not _has_trtllm_gen(), reason="the paged draft cache path needs an SM100-family GPU"
)

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB, PAGE = 64, 2, 4, 2, 128, 128, 512, 32


def _drafter(partial_rotary_factor=1.0):
    from transformers import Qwen3Config

    from tensorrt_llm._torch.model_config import ModelConfig

    cfg = Qwen3Config.from_dict(
        dict(
            architectures=["DFlashDraftModel"],
            model_type="qwen3",
            block_size=4,
            hidden_size=HIDDEN,
            num_hidden_layers=LAYERS,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            intermediate_size=INTER,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
            max_position_embeddings=8192,
            rope_theta=1e7,
            rope_scaling=None,
            partial_rotary_factor=partial_rotary_factor,
            attention_bias=False,
            torch_dtype="bfloat16",
            num_target_layers=4,
            tie_word_embeddings=False,
            dflash_config={"target_layer_ids": [0, 1], "mask_token_id": VOCAB - 2},
        )
    )
    model = DFlashForCausalLM(
        ModelConfig(pretrained_config=cfg, attn_backend="TRTLLM"), dflash_attention_backend="TRTLLM"
    ).to("cuda")
    g = torch.Generator().manual_seed(3)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    w = {
        "fc.weight": rnd(HIDDEN, HIDDEN * 2),
        "hidden_norm.weight": rnd(HIDDEN) + 1.0,
        "norm.weight": rnd(HIDDEN) + 1.0,
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.k_proj.weight"] = rnd(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.v_proj.weight"] = rnd(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = rnd(HEAD_DIM) + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(HEAD_DIM) * 4 + 1.0
        w[p + "input_layernorm.weight"] = rnd(HIDDEN) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTER)
    model.load_weights(w)
    model._build_fused_kv_buffers()
    assert model._get_cos_sin_cache().shape[-1] == int(HEAD_DIM * partial_rotary_factor)
    return model


def _case(n_req, kp1, seed, placeholder):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows_total, pages_per_row = n_req + 2, 6
    cap = pages_per_row * PAGE
    table = (
        torch.randperm(rows_total * pages_per_row, generator=g, device="cuda")
        .int()
        .view(rows_total, -1)
    )
    if placeholder:
        table[:, pages_per_row // 2 :] = (
            0  # past each request's allocation: page 0, which another request owns
        )
    rows = torch.arange(1, n_req + 1, device="cuda")
    ctx_len = torch.randint(0, cap, (n_req,), generator=g, device="cuda")
    ctx_len[0] = cap - 2  # the last request runs into the clamp
    off = torch.arange(kp1, device="cuda")
    cols = torch.minimum(ctx_len[:, None] + off, torch.tensor(cap - 1, device="cuda"))
    nacc = torch.randint(1, kp1 + 1, (n_req,), generator=g, device="cuda")
    proj = (torch.randn(n_req * kp1, HIDDEN, generator=g, device="cuda") * 2).bfloat16()
    return dict(
        table=table,
        rows=rows,
        pos=(ctx_len[:, None] + off).reshape(-1),
        cols=cols.reshape(-1),
        nacc=nacc,
        proj=proj,
        kp1=kp1,
        pages=rows_total * pages_per_row,
        rows_total=rows_total,
        ppr=pages_per_row,
    )


def _stock(model, c, kv_buf):
    """The worker's own code after the projection: precompute_context_kv, the mask, _store_context_kv_paged."""
    k, v = model.precompute_context_kv(c["proj"], c["pos"])
    mask = (torch.arange(c["kp1"], device="cuda")[None] < c["nacc"][:, None]).reshape(-1)
    mb = mask.view(-1, 1, 1, 1).to(k.dtype)
    k.mul_(mb)
    v.mul_(mb)
    worker = object.__new__(DFlashWorker)
    worker._lean_glue = False
    worker._ctx_paged_append = None
    worker._ctx_block_tables = c["table"]
    worker._ctx_block_indptr = torch.arange(
        0, (c["rows_total"] + 1) * c["ppr"], c["ppr"], device="cuda", dtype=torch.int32
    )
    worker._ctx_page_size = PAGE
    worker._ctx_kv_buf = kv_buf
    worker._ctx_kv_last_page_len = torch.full(
        (c["rows_total"],), PAGE, device="cuda", dtype=torch.int32
    )
    rows = c["rows"][:, None].expand(-1, c["kp1"]).reshape(-1)
    DFlashWorker._store_context_kv_paged(worker, k, v, rows, c["cols"])


def _caches(c, layout):
    full = torch.full(
        (LAYERS, c["pages"], 2, KV_HEADS, PAGE, HEAD_DIM), 7.0, device="cuda"
    ).bfloat16()
    return full if layout == "stacked" else [full[i].clone() for i in range(LAYERS)]


def _single_writer(c):
    """Cells exactly one row writes. A clamped column (several rows of one request) or a placeholder page (several
    requests) is written more than once; stock's winner there is a race between the append's thread blocks and no
    reader reads it (dflash_allocated_ctx_limit), so those cells are left out of the comparison."""
    rows = c["rows"].repeat_interleave(c["kp1"])
    key = c["table"][rows, c["cols"] // PAGE].long() * PAGE + c["cols"] % PAGE
    uniq, counts = torch.unique(key, return_counts=True)
    keep = torch.ones(
        LAYERS, c["pages"], 2, KV_HEADS, PAGE, HEAD_DIM, dtype=torch.bool, device="cuda"
    )
    for k in uniq[counts > 1].tolist():
        keep[:, k // PAGE, :, :, k % PAGE] = False
    return keep


def _as_int16(kv_buf):
    t = kv_buf if isinstance(kv_buf, torch.Tensor) else torch.stack(kv_buf)
    return t.view(torch.int16)


@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
@pytest.mark.parametrize("layout", ["stacked", "list"])
@pytest.mark.parametrize("n_req,kp1", [(1, 4), (3, 4), (2, 8)])
@pytest.mark.parametrize("placeholder", [False, True])
@pytest.mark.parametrize("columns", ["given", "from_counts", "from_max_col"])
@pytest.mark.parametrize("fold", [False, True])
def test_fused_store_is_the_stock_store_bit_for_bit(
    layout, n_req, kp1, placeholder, partial_rotary_factor, columns, fold
):
    """fold: TAILBK_SPEC F, the k_norm weight multiply inside the kernel (the drafter's weight is bf16, so it is
    taken); the stores must equal stock's either way."""
    model = _drafter(partial_rotary_factor)
    if fold:
        assert (
            model._k_norm_stacked.dtype == torch.bfloat16 and model._k_norm_stacked.is_contiguous()
        )
    fold_k_norm_weight(fold)
    try:
        _check_fused_store(model, layout, n_req, kp1, placeholder, columns)
    finally:
        fold_k_norm_weight(False)


def _check_fused_store(model, layout, n_req, kp1, placeholder, columns):
    for seed in range(4):
        c = _case(n_req, kp1, seed, placeholder)
        ref, got = _caches(c, layout), _caches(c, layout)
        assert ctx_store_supported(model, got)
        _stock(model, c, ref)
        # The case's columns are the positions clamped to the rows' pages (every row holds ppr pages), so the
        # kernel's own derivation (S10c-A2: from the page counts, or from the arena's last column) must match them.
        kwargs = {}
        if columns == "from_counts":
            kwargs["capacity_counts"] = torch.full(
                (c["rows_total"],), c["ppr"], dtype=torch.int64, device="cuda"
            )
        elif columns == "from_max_col":
            kwargs["max_col"] = c["ppr"] * PAGE - 1
        cols = c["cols"] if columns == "given" else None
        assert fused_ctx_store(
            model,
            c["proj"],
            c["pos"],
            c["nacc"],
            c["rows"],
            cols,
            c["table"],
            got,
            c["kp1"],
            **kwargs,
        )
        torch.cuda.synchronize()
        keep = _single_writer(c)
        assert torch.equal(_as_int16(ref)[keep], _as_int16(got)[keep]), f"seed {seed}"


def test_unsupported_drafters_keep_the_stock_path():
    model = _drafter()
    kv = torch.zeros(LAYERS, 4, 2, KV_HEADS, PAGE, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    assert ctx_store_supported(model, kv)
    assert not ctx_store_supported(model, kv.float())  # not a bf16 cache
    assert not ctx_store_supported(model, kv[:, :, :1])  # not a [K | V] HND page
    no_norm = SimpleNamespace(
        **{k: getattr(model, k) for k in ("_k_norm_eps", "_is_neox", "_num_kv_heads", "_head_dim")}
    )
    no_norm._k_norm_stacked = None
    assert not ctx_store_supported(no_norm, kv)
    assert not fused_ctx_store(no_norm, None, None, None, None, None, None, kv, 4)
    # Derived columns need exactly one bound: the page counts (int64) or the arena's last column.
    assert not fused_ctx_store(model, None, None, None, None, None, None, kv, 4)
    # A rotary dim the kernel cannot split into power-of-two halves (48 of 128) keeps the stock path.
    odd_rotary = SimpleNamespace(
        **{
            k: getattr(model, k)
            for k in ("_k_norm_eps", "_is_neox", "_num_kv_heads", "_head_dim", "_k_norm_stacked")
        }
    )
    odd_rotary._get_cos_sin_cache = lambda: torch.zeros(16, 48, device="cuda")
    assert not ctx_store_supported(odd_rotary, kv)
