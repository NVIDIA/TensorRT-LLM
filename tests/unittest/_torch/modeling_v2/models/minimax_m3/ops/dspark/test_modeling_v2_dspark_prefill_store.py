# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/dspark/dspark_prefill_store (DEP2CTX_DSPARK_STORE): the per-chunk prefill store writes, bit for bit, what
DFlashWorker's stock prefill lines write (GPU, sm_107).

Stock = DFlashForCausalLM.precompute_context_kv(proj, arange(start, end)) + DFlashWorker._store_context_kv_paged(k, v,
full(row), arange(start, end)) (flashinfer append per layer), with the drafter's lean glue off and, where the engine has
it, on. Fused = PrefillStore -> fused_ctx_store -> family_ops().minimax_m31_dspark_ctx_store (generic build, given
columns, every row accepted), k_norm weight fold off and on. Both write into caches filled with a sentinel and are
compared WHOLE in an int16 view: a prefill chunk writes every (page, slot) once, so no cell is excluded.

Cells: chunk length 1 / 7 / 64 / 3500 / 16384, slot-local start 0 / 31 / 8192 (continuation and reuse-start chunks),
page 32 / 64, the bound pool (per-layer list, rows = batch positions) and the private arena (stacked, rows = slots),
full and partial RoPE (0.5: the M3.1 drafter's rotary 64 of 128), a tiny drafter and the M3.1 drafter's K / V geometry
(hidden 6144, 5 layers, 4 kv heads of 128). Then the loop: DFlashWorker._store_prefill_context over a multi-step
schedule with the hook installed vs not, on the stacked private arena, a per-layer arena and the bound pool, with and
without host positions (the latter takes the hooked loop's lazy first-position read); it needs the engine's
DFlashWorker.enable_fused_prefill_store hook.

Negative controls (the oracle must DETECT every one): an off-by-one start, a zero accept count, the rows of two
requests swapped, and the k_norm weight folded in the kernel on top of torch's multiply. The last two run their
unmutated twin too, so a detection is the mutation's and not the harness's.
"""

import collections
import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_store as dcs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_prefill_store as dps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_store import (
    ctx_store_supported,
    fold_k_norm_weight,
    fused_ctx_store,
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
_HAS_PREFILL_HOOK = hasattr(DFlashWorker, "enable_fused_prefill_store")

TINY = dict(hidden=64, layers=2, heads=4, kv_heads=2)
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


@pytest.mark.parametrize("geom", ["tiny", "m31"])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
@pytest.mark.parametrize("layout", ["stacked", "list"])
@pytest.mark.parametrize("page", [32, 64])
@pytest.mark.parametrize("start,n", [(0, 1), (0, 7), (31, 64), (0, 3500), (8192, 3500), (0, 16384)])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("lean", [False, True])
def test_prefill_store_is_the_stock_store_bit_for_bit(
    geom, partial_rotary_factor, layout, page, start, n, fold, lean
):
    if lean and not _HAS_LEAN_GLUE:
        pytest.skip("needs the engine's drafter lean glue hook (DFlashWorker.enable_lean_glue)")
    if geom == "m31" and partial_rotary_factor != 0.5:
        pytest.skip("the M3.1 drafter is partial-rotary 0.5")
    model = _drafter(TINY if geom == "tiny" else M31, partial_rotary_factor)
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


@pytest.mark.parametrize("mutation", ["start_plus_one", "zero_accepted"])
def test_oracle_sees_mutations(mutation):
    model = _drafter(TINY, 0.5)
    page, start, n, row = 32, 0, 64, 1
    ref, table = _pool(model, 3, 4, page, "list")
    got, _ = _pool(model, 3, 4, page, "list")
    g = torch.Generator(device="cuda").manual_seed(13)
    proj = (
        torch.randn(n, model._fused_kv_weight.shape[1], generator=g, device="cuda") * 2
    ).bfloat16()
    _stock_store(model, proj, start, start + n, row, ref, table, page, False)
    pos = torch.arange(start, start + n, device="cuda") + (1 if mutation == "start_plus_one" else 0)
    nacc = torch.tensor([0 if mutation == "zero_accepted" else 2**31 - 1], device="cuda")
    assert fused_ctx_store(
        model, proj, pos, nacc, torch.tensor([row], device="cuda"), pos, table, got, n
    )
    torch.cuda.synchronize()
    assert not torch.equal(_i16(ref), _i16(got))


@pytest.mark.parametrize("swap", [False, True])
def test_oracle_sees_swapped_request_rows(swap):
    """Two requests' chunks stored by the stock lines into rows 1 and 2; the fused store writes them into the same rows
    (equal) or into each other's rows (must differ)."""
    model = _drafter(TINY, 0.5)
    page, n = 32, 64
    ref, table = _pool(model, 3, 4, page, "list")
    got, _ = _pool(model, 3, 4, page, "list")
    g = torch.Generator(device="cuda").manual_seed(11)
    hidden = model._fused_kv_weight.shape[1]
    pa, pb = ((torch.randn(n, hidden, generator=g, device="cuda") * 2).bfloat16() for _ in range(2))
    _stock_store(model, pa, 0, n, 1, ref, table, page, False)
    _stock_store(model, pb, 0, n, 2, ref, table, page, False)
    store = dps.PrefillStore()
    assert store(model, pa, 0, n, 2 if swap else 1, table, got)
    assert store(model, pb, 0, n, 1 if swap else 2, table, got)
    torch.cuda.synchronize()
    assert torch.equal(_i16(ref), _i16(got)) is not swap


def _store_weight_twice(model, proj, start, end, row, table, cache, fold):
    """fused_ctx_store's lines with torch's k_norm weight multiply always applied, and the kernel's fold on top (fold
    True: the weight lands twice) or not (fold False: fused_ctx_store with the fold off)."""
    n = end - start
    L, nkv, hd = model._num_attn_layers, model._num_kv_heads, model._head_dim
    kv_flat = F.linear(proj, model._fused_kv_weight, model._fused_kv_bias)
    k = F.rms_norm(
        kv_flat.view(n, L, 2, nkv, hd)[:, :, 0].contiguous(), (hd,), eps=model._k_norm_eps
    )
    w = model._k_norm_stacked
    k = k * w.view(1, L, 1, hd)
    pos = torch.arange(start, end, dtype=torch.int64, device="cuda")
    cache_arg, ptr_list, stride_layer = dcs._cache_arg(cache)
    first = cache if isinstance(cache, torch.Tensor) else cache[0]
    family_ops().minimax_m31_dspark_ctx_store(
        kv_flat,
        k,
        w.contiguous(),
        model._get_cos_sin_cache(),
        pos,
        torch.tensor([2**31 - 1], dtype=torch.int64, device="cuda"),
        torch.tensor([row], dtype=torch.int64, device="cuda"),
        pos,
        pos,
        0,
        table,
        cache_arg,
        stride_layer,
        n,
        L,
        nkv,
        hd,
        first.shape[-2],
        ptr_list,
        False,
        False,
        fold,
    )


@pytest.mark.parametrize("fold_on_top", [False, True])
def test_oracle_sees_the_weight_folded_on_top_of_torchs_multiply(fold_on_top):
    model = _drafter(TINY, 0.5)
    page, start, n, row = 32, 0, 64, 1
    ref, table = _pool(model, 3, 4, page, "list")
    got, _ = _pool(model, 3, 4, page, "list")
    g = torch.Generator(device="cuda").manual_seed(17)
    proj = (
        torch.randn(n, model._fused_kv_weight.shape[1], generator=g, device="cuda") * 2
    ).bfloat16()
    _stock_store(model, proj, start, start + n, row, ref, table, page, False)
    _store_weight_twice(model, proj, start, start + n, row, table, got, fold_on_top)
    torch.cuda.synchronize()
    assert torch.equal(_i16(ref), _i16(got)) is not fold_on_top


# ----------------------------------------------------------------------------------------------------------------------
# The loop: DFlashWorker._store_prefill_context with the hook vs without, multi-step schedule
# ----------------------------------------------------------------------------------------------------------------------


def _loop_worker(model, bound, layout, slots, max_ctx, page):
    """The private arena is one stacked [L, pages, 2, nkv, page, hd] tensor (_lazy_init_ctx_buffers) with rows =
    slots; the bound pool is per-layer views with rows = batch positions. ``layout`` picks the cache form."""
    pages = max_ctx // page
    w = object.__new__(DFlashWorker)
    torch.nn.Module.__init__(w)
    w._ctx_paged = True
    w._max_ctx = max_ctx
    w._ctx_page_size = page
    w._ctx_reuse_addressable = False
    w._req_ctx_pos, w._parked_ctx_base = {}, collections.OrderedDict()
    w._req_to_slot, w._free_slots = {}, collections.deque(range(slots))
    w._ctx_len_host, w._ctx_base_host, w._ctx_stored_host = (
        [0] * slots,
        [0] * slots,
        [False] * slots,
    )
    w._ctx_len = torch.zeros(slots, dtype=torch.long, device="cuda")
    w._ctx_base_pages = torch.zeros(slots, dtype=torch.long, device="cuda")
    w._ctx_stored = torch.zeros(slots, dtype=torch.bool, device="cuda")
    L, nkv = model._num_attn_layers, model._num_kv_heads
    full = torch.full((L, slots * pages, 2, nkv, page, HEAD_DIM), 7.0, device="cuda").bfloat16()
    w._ctx_kv_buf = full if layout == "stacked" else [full[i].clone() for i in range(L)]
    perm = torch.randperm(slots * pages, generator=torch.Generator().manual_seed(5)).int().cuda()
    w._ctx_page_table = perm.view(slots, pages).contiguous()
    w._ctx_kv_indptr = torch.arange(0, slots * pages + 1, pages, dtype=torch.int32, device="cuda")
    w._ctx_block_tables = perm.flip(0).view(slots, pages).contiguous() if bound else None
    w._ctx_block_indptr = w._ctx_kv_indptr if bound else None
    w._ctx_block_counts = (
        torch.full((slots,), pages, dtype=torch.long, device="cuda") if bound else None
    )
    w._ctx_kv_last_page_len = torch.full((slots,), page, dtype=torch.int32, device="cuda")
    w._ctx_paged_append = None
    w._lean_glue = False
    w._fused_prefill_store = None
    return w


STEPS = [
    [(10, 0, 3500), (11, 0, 70)],
    [(10, 3500, 4096), (12, 2048, 300), (13, 0, 0)],
    [(14, 0, 16000), (11, 70, 1)],
    [(14, 16000, 500), (10, 7596, 33)],  # 14 overflows max_ctx 16384: stock skip
    [(11, 0, 9)],  # reused id: reset
]


def _run(w, model, steps, hidden, host_pos):
    for chunks in steps:
        n = sum(s for _, _, s in chunks)
        hs = (
            (torch.randn(n, hidden, generator=torch.Generator().manual_seed(n)) * 2)
            .bfloat16()
            .cuda()
        )
        pos = torch.cat([torch.arange(f, f + s, dtype=torch.int32) for _, f, s in chunks]).cuda()
        spec = types.SimpleNamespace(
            request_ids=[r for r, _, _ in chunks],
            host_position_ids=pos.cpu() if host_pos else None,
            get_hidden_states=lambda total, hs=hs: hs,
        )
        attn = types.SimpleNamespace(
            num_ctx_tokens=n,
            num_contexts=len(chunks),
            _seq_lens=torch.tensor([s for _, _, s in chunks]),
        )
        w._store_prefill_context(model, spec, attn, pos, n)
    torch.cuda.synchronize()


@pytest.mark.skipif(
    not _HAS_PREFILL_HOOK,
    reason="needs the engine's per-chunk prefill store hook (DFlashWorker.enable_fused_prefill_store)",
)
@pytest.mark.parametrize("host_pos", [True, False])
@pytest.mark.parametrize(
    "bound, layout",
    [(False, "stacked"), (False, "list"), (True, "list")],
    ids=["arena-stacked", "arena-list", "bound-list"],
)
def test_prefill_loop_with_the_hook_is_bitwise_the_stock_loop(bound, layout, host_pos, monkeypatch):
    """host_pos False takes the hooked loop's lazy first-position read (position_ids[offset].item())."""
    model = _drafter(M31, 0.5)
    hidden = model._fused_kv_weight.shape[1]
    # The projection is outside the change; monkeypatch restores the cached drafter for later tests.
    monkeypatch.setattr(model, "project_target_hidden", lambda x: x)
    ref = _loop_worker(model, bound, layout, slots=6, max_ctx=16384, page=32)
    got = _loop_worker(model, bound, layout, slots=6, max_ctx=16384, page=32)
    store = dps.PrefillStore()
    assert got.enable_fused_prefill_store(store)
    fold_k_norm_weight(True)
    try:
        _run(ref, model, STEPS, hidden, host_pos)
        _run(got, model, STEPS, hidden, host_pos)
    finally:
        fold_k_norm_weight(False)
    assert store.calls == store.taken > 0
    assert torch.equal(_i16(ref._ctx_kv_buf), _i16(got._ctx_kv_buf))
    assert torch.equal(ref._ctx_len, got._ctx_len) and ref._ctx_len_host == got._ctx_len_host
    assert torch.equal(ref._ctx_base_pages, got._ctx_base_pages)
    assert ref._req_to_slot == got._req_to_slot and ref._req_ctx_pos == got._req_ctx_pos
