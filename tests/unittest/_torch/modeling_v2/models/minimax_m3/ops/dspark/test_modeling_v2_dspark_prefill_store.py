# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/dspark/dspark_prefill_store (DEP2CTX_DSPARK_STORE): the per-chunk prefill store writes, bit for bit, what
DFlashWorker's stock prefill lines write (GPU, sm_107).

Stock = DFlashForCausalLM.precompute_context_kv(proj, arange(start, end)) + DFlashWorker._store_context_kv_paged(k, v,
full(row), arange(start, end)) (flashinfer append per layer), with the drafter's lean glue off and, where the engine has
it, on. Fused = PrefillStore -> fused_ctx_store -> family_ops().minimax_m31_dspark_ctx_store (generic build, given
columns, every row accepted), k_norm weight fold off and on. Both write into caches filled with a sentinel and are
compared WHOLE in an int16 view: a prefill chunk writes every (page, slot) once, so no cell is excluded.

Cells: chunk length 1 / 64 / 3500, slot-local start 0 / 31 / 8192 (continuation and reuse-start chunks),
page 64, the private arena (stacked, rows = slots), on the
M3.1 drafter's K / V geometry (hidden 6144, 5 layers, 4 kv heads of 128, partial RoPE 0.5: rotary 64 of 128; tiny
full-RoPE drafters are covered by test_modeling_v2_dspark_ctx_store.py).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import dspark_prefill_store as dps
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_store import (
    ctx_store_supported,
    fold_k_norm_weight,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the DSpark context store op runs on sm_107 only",
)

# The drafter's lean glue (DFlashWorker.enable_lean_glue / DFlashForCausalLM._dflash_lean_glue) and the per-chunk
# prefill hook (DFlashWorker.enable_fused_prefill_store) are engine hooks; cells that need them skip without them.
_HAS_LEAN_GLUE = hasattr(DFlashWorker, "enable_lean_glue")

M31 = dict(hidden=6144, layers=5, heads=8, kv_heads=4)  # the M3.1 DSpark drafter's K / V geometry
HEAD_DIM, INTER, VOCAB = 128, 128, 512
_MODELS = {}


def _drafter(geom, partial_rotary_factor):
    key = (tuple(sorted(geom.items())), partial_rotary_factor)
    if key in _MODELS:
        return _MODELS[key]
    from transformers import Qwen3Config

    from tensorrt_llm._torch.model_config import ModelConfig

    hidden, layers, heads, kv_heads = (
        geom["hidden"],
        geom["layers"],
        geom["heads"],
        geom["kv_heads"],
    )
    cfg = Qwen3Config.from_dict(
        dict(
            architectures=["DFlashDraftModel"],
            model_type="qwen3",
            block_size=4,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=HEAD_DIM,
            intermediate_size=INTER,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            vocab_size=VOCAB,
            max_position_embeddings=32768,
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
        "fc.weight": rnd(hidden, hidden * 2),
        "hidden_norm.weight": rnd(hidden) + 1.0,
        "norm.weight": rnd(hidden) + 1.0,
    }
    for i in range(layers):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(heads * HEAD_DIM, hidden)
        w[p + "self_attn.k_proj.weight"] = rnd(kv_heads * HEAD_DIM, hidden)
        w[p + "self_attn.v_proj.weight"] = rnd(kv_heads * HEAD_DIM, hidden)
        w[p + "self_attn.o_proj.weight"] = rnd(hidden, heads * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = rnd(HEAD_DIM) + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(HEAD_DIM) * 4 + 1.0
        w[p + "input_layernorm.weight"] = rnd(hidden) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(hidden) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, hidden)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, hidden)
        w[p + "mlp.down_proj.weight"] = rnd(hidden, INTER)
    model.load_weights(w)
    model._build_fused_kv_buffers()
    assert model._get_cos_sin_cache().shape[-1] == int(HEAD_DIM * partial_rotary_factor)
    _MODELS[key] = model
    return model


def _pool(model, rows, width, page, layout):
    """A sentinel-filled draft cache and a random dense block table (every row owns `width` distinct pages)."""
    L, nkv = model._num_attn_layers, model._num_kv_heads
    pages = rows * width
    full = torch.full((L, pages, 2, nkv, page, HEAD_DIM), 7.0, device="cuda").bfloat16()
    cache = full if layout == "stacked" else [full[i].clone() for i in range(L)]
    g = torch.Generator(device="cuda").manual_seed(rows * 131 + width)
    table = torch.randperm(pages, generator=g, device="cuda").int().view(rows, width).contiguous()
    return cache, table


def _stock_worker(model, cache, table, page, lean):
    """The worker fields _store_context_kv_paged reads, as _lazy_init_ctx_buffers / _init_ctx_block_tables set them
    (dense table, indptr = arange * width)."""
    w = object.__new__(DFlashWorker)
    w._lean_glue = lean
    w._ctx_paged_append = None
    w._ctx_block_tables = table
    rows, width = table.shape
    w._ctx_block_indptr = torch.arange(
        0, (rows + 1) * width, width, device="cuda", dtype=torch.int32
    )
    w._ctx_page_size = page
    w._ctx_kv_buf = cache
    w._ctx_kv_last_page_len = torch.full((rows,), page, device="cuda", dtype=torch.int32)
    return w


def _stock_store(model, proj, start, end, row, cache, table, page, lean):
    model._dflash_lean_glue = lean
    try:
        pos = torch.arange(start, end, dtype=torch.long, device="cuda")
        k, v = model.precompute_context_kv(proj, pos)
        rows = torch.full((end - start,), row, dtype=torch.long, device="cuda")
        _stock_worker(model, cache, table, page, lean)._store_context_kv_paged(k, v, rows, pos)
    finally:
        model._dflash_lean_glue = False


def _i16(cache):
    return (cache if isinstance(cache, torch.Tensor) else torch.stack(cache)).view(torch.int16)


@pytest.mark.parametrize("layout", ["stacked"])
@pytest.mark.parametrize("page", [64])
@pytest.mark.parametrize("start,n", [(0, 1), (31, 64), (8192, 3500)])
@pytest.mark.parametrize("fold", [True])
@pytest.mark.parametrize("lean", [False, True])
def test_prefill_store_is_the_stock_store_bit_for_bit(layout, page, start, n, fold, lean):
    if lean and not _HAS_LEAN_GLUE:
        pytest.skip("needs the engine's drafter lean glue hook (DFlashWorker.enable_lean_glue)")
    model = _drafter(M31, 0.5)
    rows, width = 3, -(-(start + n) // page) + 1
    ref, table = _pool(model, rows, width, page, layout)
    got, _ = _pool(model, rows, width, page, layout)
    assert ctx_store_supported(model, got)
    g = torch.Generator(device="cuda").manual_seed(start * 7 + n)
    hidden = model._fused_kv_weight.shape[1]
    proj = (torch.randn(n, hidden, generator=g, device="cuda") * 2).bfloat16()
    row = 1
    _stock_store(model, proj, start, start + n, row, ref, table, page, lean)
    fold_k_norm_weight(fold)
    try:
        store = dps.PrefillStore()
        assert store(model, proj, start, start + n, row, table, got)
    finally:
        fold_k_norm_weight(False)
    torch.cuda.synchronize()
    assert torch.equal(_i16(ref), _i16(got))
    assert not torch.equal(_i16(got), _i16(_pool(model, rows, width, page, layout)[0]))  # it wrote


# ----------------------------------------------------------------------------------------------------------------------
# The loop: DFlashWorker._store_prefill_context with the hook vs without, multi-step schedule
# ----------------------------------------------------------------------------------------------------------------------
