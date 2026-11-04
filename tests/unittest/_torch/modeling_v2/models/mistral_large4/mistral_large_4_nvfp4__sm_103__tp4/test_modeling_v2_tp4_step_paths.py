# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Mistral-Large-4-1T-A52B NVFP4 tp4 target chooses per step, and the engine sizes it
chooses against. No path is selected by the serving configuration: each context step picks its
own from its sizes, and the two sizes that bound it are derived on the first step
(`_derive_first_step`) from the engine's quantities on the attention metadata alone
(max_num_tokens, max_num_requests, max_seq_len, and runtime_features: chunked prefill and block
reuse, as the executor builds them from the LLM-API options); both are 0 until then.

* The context path, by the step's context KV n_kv (every context request's cached and new rows):
  while n_kv fits the held workspace, the workspace path over its first n_kv rows; past it, the
  decode path, the context rows in sub-batches of at most `_CTX_DECODE_ROWS` with one FlashMLA
  scheduler per sub-batch size and no workspace. The held workspace is allocated once, on the
  first context step whichever path it takes, and every later step reuses its storage.
* Layer 35, the owner the NoPE layers 39, 43, ..., 71 attend over (`shared_source`, layer 35
  alone at every max_seq_len), by fit: while 2 n_kv fits the held rows it is gathered once into
  the n_kv rows kept beside the step's own (`kept`), where 39 and 43 read it without a gather;
  past that, with n_kv still within the held rows, every layer gathers its source into the
  first n_kv rows, 35 again for 39 and for 43.
* The held workspace's rows (`_ctx_workspace_rows`): the most context KV one step can attend
  (`_most_ctx_kv`) twice over, up to `_CTX_WORKSPACE_ROWS` (2**19). With neither chunked
  prefill nor block reuse every context row is new, so it is twice max_num_tokens: 65536 at the
  32768 serving config (max_num_tokens = max_seq_len = 32768), 32768 at the Pareto config
  (max_num_tokens 16384, max_seq_len 32768), 65536 at max_seq_len 131072 and max_num_tokens
  32768. With either, a step can attend each request's cached prefix up to max_seq_len, and it
  is mainline's never-stall worst case min(max_num_requests, max_num_tokens) * max_seq_len twice
  over, so 2**19: the 32768 config with block reuse (max_batch_size 16), the 1M config
  (max_seq_len 1048576, chunked prefill) and max_seq_len 131072 with chunked prefill. Block
  reuse is the runtime features' cache_reuse; neither size reads the KV cache manager.
* The indexer's prefill logits budget (`_prefill_logits_budget`): mainline's reachable
  MQA-logits tile, within `_PREFILL_LOGITS_CEILING` (2**28) and within max_seq_len / 64 rows of
  max_seq_len keys. 2**24 at max_seq_len 32768 (512-row tiles at 32768 keys), the 32768 and the
  Pareto config alike; 2**28 at 131072 (2048-row tiles) and at 1048576 (256-row tiles).
  TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET lowers it and cannot raise it. `_select_ctx` scores each
  block's rows in tiles of budget // keys.
* The indexer's scoring blocks (`_indexer_blocks`): an engine chunk is one block, as planned,
  while it holds at most max_seq_len keys, several requests included; one past max_seq_len
  keys is cut at its requests over the host prefix sums `step_args` carries on every context
  step. A block with no key fills its rows' selection slots with -1.
* The rope tables: `indexer` and `identity`, no `mla` table; `build_kernels` and
  `_derive_first_step` build them with no such option.

`_derive_first_step` runs on the core the target builds from the checkpoint's native config,
params.json, on the meta device (skipped where the checkpoint is absent under LLM_MODELS_ROOT);
everything else runs on host values, with recording stand-ins for the kernels the paths call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_workspace_bytes,
)

pytestmark = pytest.mark.cpu_only

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
#: The checkpoint's layers, the owner its NoPE layers 39, 43, ..., 71 attend over, each layer's
#: latent source, the KV owners, and the latent row's width.
LAYERS = 72
SHARED_SOURCE = 35
KV_SOURCE = [SHARED_SOURCE if i > SHARED_SOURCE and i % 4 == 3 else i for i in range(LAYERS)]
KV_OWNERS = [i for i in range(LAYERS) if KV_SOURCE[i] == i]
LAT_DIM = 576
#: The engine's max_batch_size, max_num_requests on the attention metadata.
MAX_BATCH = 16
#: The mainline knob that lowers the reachable MQA-logits tile.
ELEM_BUDGET_ENV = "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET"
#: Serving configs by what the target sizes from -- (max_num_tokens, max_seq_len, chunked
#: prefill, block reuse) at max_batch_size 16 -- with the held workspace's rows and the logits
#: budget they give.
CONFIGS = [
    pytest.param((32768, 32768, False, False), 65536, 1 << 24, id="32k"),
    pytest.param((16384, 32768, False, False), 32768, 1 << 24, id="pareto"),
    pytest.param((32768, 32768, False, True), 1 << 19, 1 << 24, id="32k_block_reuse"),
    pytest.param((32768, 1048576, True, False), 1 << 19, 1 << 28, id="1m"),
    pytest.param((32768, 131072, True, False), 1 << 19, 1 << 28, id="131072_chunked"),
    pytest.param((32768, 131072, False, False), 65536, 1 << 28, id="131072"),
]
#: The checkpoint's rope surface (its native config, adapted from params.json), what
#: `_rope_tables` reads of it.
ROPE_CFG = SimpleNamespace(
    rope_parameters=dict(
        rope_type="yarn",
        rope_theta=10000.0,
        factor=2048.0,
        original_max_position_embeddings=512,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale_all_dim=1,
        attention_factor=1.0,
    ),
    qk_rope_head_dim=64,
)
CPU = torch.device("cpu")


def _core(max_num_tokens: int, max_seq_len: int | None):
    """The core of rank 0 at four ranks, on the meta device, for an engine at `max_num_tokens`
    and `max_seq_len`."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm.mapping import Mapping

    root = llm_models_root()
    paths = [root / name for name in CHECKPOINTS] if root is not None else []
    path = next((p for p in paths if (p / "params.json").is_file()), None)
    if path is None:
        pytest.skip(f"none of {CHECKPOINTS} is under LLM_MODELS_ROOT ({root})")
    mapping = Mapping(
        world_size=4, rank=0, gpus_per_node=4, tp_size=4, pp_size=1, moe_ep_size=1, moe_tp_size=4
    )
    model_config = ModelConfig(
        pretrained_config=MistralConfigLoader().load_pretrained_config(str(path)),
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
    )
    with torch.device("meta"):
        return M.MistralLarge4Tp4ModelingV2(model_config)


def _engine(max_num_tokens: int, max_seq_len: int, chunked: bool, reuse: bool) -> SimpleNamespace:
    """What the target reads of the engine's attention metadata to size its buffers, with the
    runtime features as `create_py_executor` builds them (chunked prefill `chunked`, block reuse
    `reuse`, no drafter, the chunk max_num_tokens), and the KV cache manager surface the first
    step checks."""
    return SimpleNamespace(
        max_num_tokens=max_num_tokens,
        max_num_requests=MAX_BATCH,
        max_seq_len=max_seq_len,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=chunked,
            cache_reuse=reuse,
            has_speculative_draft_tokens=False,
            chunk_size=max_num_tokens,
        ),
        kv_cache_manager=SimpleNamespace(
            latent_layout=M._LATENT_LAYOUT,
            kv_source_layers=KV_SOURCE,
            latent_pool_row=lambda i: KV_OWNERS.index(KV_SOURCE[i]),
            kv_owner_layers=KV_OWNERS,
        ),
        tokens_per_block=32,
    )


def _no_device(monkeypatch) -> list:
    """Stand-ins for what `build_kernels` and `_derive_first_step` compile or read off a device.
    The rope tables' stand-in takes `_rope_tables`'s parameters and no other, so a call with an
    `mla` option fails; it records each table's rows."""
    built = []

    def tables(cfg, alpha, positions, dtype, device):
        built.append(positions)
        return SimpleNamespace(rows=positions)

    monkeypatch.setattr(M, "_rope_tables", tables)
    monkeypatch.setattr(M, "build_ix_q_side", lambda dev, is_neox: None)
    monkeypatch.setattr(M, "build_ix_k_cache", lambda dev, is_neox, cache_contiguous: None)
    monkeypatch.setattr(M, "build_ds_mla_decode", lambda dev, num_heads: None)
    monkeypatch.setattr(
        M, "build_ix_topk_decode", lambda dev, next_n: SimpleNamespace(next_n=next_n)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return built


def _cumsum(lens) -> torch.Tensor:
    return torch.tensor([0, *lens], dtype=torch.int64).cumsum(0)


# ---- the engine sizes ----


@pytest.mark.parametrize(
    "max_seq_len", [8192, 32768, 32769, 1048576, None], ids=lambda n: f"max_seq_len{n}"
)
def test_core_selects_nothing_by_max_seq_len(max_seq_len):
    """Built at any max_seq_len (None: the engine infers one, at least the checkpoint's
    max_position_embeddings, 1048576), the core is the same: layer 35 is its one shared source,
    and the logits budget and the held workspace's rows are 0 until the first step fixes them.
    Nothing of a static switch is left: no `long_context`, no threshold, no fixed budget."""
    core = _core(32768, max_seq_len)
    assert core.kv_source == KV_SOURCE
    assert [i for i, keep in enumerate(core.shared_source) if keep] == [SHARED_SOURCE]
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (0, 0)
    assert not hasattr(core, "long_context")
    # The fused attention glue at every max_seq_len, unless `_UNFUSED_GLUE`.
    assert core.unfused_glue is M._UNFUSED_GLUE
    # The native generation-row decode at every max_seq_len, unless `_GEN_ATTN_FLASH_MLA`.
    assert core.gen_attn_flash_mla is M._GEN_ATTN_FLASH_MLA
    for name in (
        "_LONG_CONTEXT_THRESHOLD",
        "_LONG_PREFILL_LOGITS_BUDGET",
        "_PREFILL_LOGITS_BUDGET",
    ):
        assert not hasattr(M, name), name


@pytest.mark.parametrize("engine, rows, budget", CONFIGS)
def test_ctx_workspace_rows_from_runtime_features(engine, rows, budget):
    """`_most_ctx_kv` is max_num_tokens with neither chunked prefill nor block reuse in the
    runtime features, else mainline's never-stall worst case min(max_num_requests,
    max_num_tokens) * max_seq_len; the held workspace has room for it twice, up to 2**19 rows."""
    max_num_tokens, max_seq_len, chunked, reuse = engine
    md = _engine(*engine)
    features = md.runtime_features
    assert (features.chunked_prefill, features.cache_reuse) == (chunked, reuse)
    most = min(MAX_BATCH, max_num_tokens) * max_seq_len if chunked or reuse else max_num_tokens
    assert M._most_ctx_kv(md) == most
    assert M._CTX_WORKSPACE_ROWS == 1 << 19
    assert M._ctx_workspace_rows(md) == min(2 * most, M._CTX_WORKSPACE_ROWS) == rows


@pytest.mark.parametrize(
    "engine, rows",
    [
        pytest.param((32768, 32768, False), (65536, 1 << 19), id="32k"),
        pytest.param((16384, 32768, False), (32768, 1 << 19), id="pareto"),
        pytest.param((32768, 1048576, True), (1 << 19, 1 << 19), id="1m"),
    ],
)
def test_block_reuse_read_from_the_runtime_features(engine, rows):
    """Block reuse is the attention metadata's `runtime_features.cache_reuse`, which the executor
    sets from the KvCacheConfig's enable_block_reuse (`create_py_executor`, which keeps the two in
    step where it turns reuse off); the sizes read the attention metadata alone, here one with no
    KV cache manager at all. Without chunked prefill, cache_reuse off holds twice max_num_tokens
    (65536 at the 32768 config, 32768 at the Pareto config) and on the worst case of 16 cached
    prefixes of max_seq_len, 2**19 rows; under chunked prefill (the 1M config) the worst case
    either way."""
    max_num_tokens, max_seq_len, chunked = engine
    for reuse, want in zip((False, True), rows):
        md = _engine(max_num_tokens, max_seq_len, chunked, reuse)
        assert md.runtime_features.cache_reuse is reuse
        metadata = SimpleNamespace(
            max_num_tokens=md.max_num_tokens,
            max_num_requests=md.max_num_requests,
            max_seq_len=md.max_seq_len,
            runtime_features=md.runtime_features,
        )
        most = MAX_BATCH * max_seq_len if chunked or reuse else max_num_tokens
        assert M._most_ctx_kv(metadata) == M._most_ctx_kv(md) == most
        assert M._ctx_workspace_rows(metadata) == M._ctx_workspace_rows(md) == want


#: (max_num_tokens, max_seq_len, budget, rows a tile of max_seq_len keys holds).
BUDGETS = [
    pytest.param(32768, 32768, 1 << 24, 512, id="32k"),
    pytest.param(16384, 32768, 1 << 24, 512, id="pareto"),
    pytest.param(32768, 131072, 1 << 28, 2048, id="131072"),
    pytest.param(32768, 1048576, 1 << 28, 256, id="1m"),
    # A step budget under max_seq_len / 64 tokens: mainline's reachable tile binds.
    pytest.param(256, 32768, 1 << 23, 256, id="reachable"),
]


@pytest.mark.parametrize("max_num_tokens, max_seq_len, budget, tile_rows", BUDGETS)
def test_prefill_logits_budget(monkeypatch, max_num_tokens, max_seq_len, budget, tile_rows):
    """The budget is the least of mainline's reachable tile (min(2**31, max_num_tokens x
    max_seq_len) logits), the ceiling 2**28, and max_seq_len / 64 rows of max_seq_len keys."""
    monkeypatch.delenv(ELEM_BUDGET_ENV, raising=False)
    reachable = get_indexer_mqa_logits_workspace_bytes(max_num_tokens, max_seq_len) // 4
    assert reachable == min(1 << 31, max_num_tokens * max_seq_len)
    got = M._prefill_logits_budget(max_num_tokens, max_seq_len)
    assert got == min(reachable, M._PREFILL_LOGITS_CEILING, max_seq_len * (max_seq_len >> 6))
    assert got == budget and got // max_seq_len == tile_rows


def test_prefill_logits_budget_ceiling_and_env_override(monkeypatch):
    """`_PREFILL_LOGITS_CEILING` is 2**28 logits, one 1 GiB fp32 tile. Under
    TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET 2**20 every config's budget is 2**20 (32-row tiles at
    32768 keys); set past the ceiling it raises nothing."""
    assert M._PREFILL_LOGITS_CEILING == 1 << 28
    configs = [(32768, 32768), (16384, 32768), (32768, 131072), (32768, 1048576)]
    monkeypatch.delenv(ELEM_BUDGET_ENV, raising=False)
    unset = [M._prefill_logits_budget(n, s) for n, s in configs]
    assert unset == [1 << 24, 1 << 24, 1 << 28, 1 << 28]
    monkeypatch.setenv(ELEM_BUDGET_ENV, str(1 << 20))
    assert [M._prefill_logits_budget(n, s) for n, s in configs] == [1 << 20] * 4
    assert M._prefill_logits_budget(32768, 32768) // 32768 == 32
    monkeypatch.setenv(ELEM_BUDGET_ENV, str(1 << 30))
    assert [M._prefill_logits_budget(n, s) for n, s in configs] == unset


@pytest.mark.parametrize("engine, rows, budget", CONFIGS)
def test_derive_first_step_sets_the_engine_sizes(monkeypatch, engine, rows, budget):
    """On the core built from the checkpoint's config at the engine's max_num_tokens and
    max_seq_len, `_derive_first_step` sets the logits budget and the held workspace's rows to
    the module helpers' values for the step's metadata (`build_kernels` built the rope tables at
    max_seq_len, and they are not built again). The target's first context step, one fresh
    request of max_num_tokens, then allocates the held workspace at those rows and keeps layer
    35's rows beside its own: the most context KV fits twice by construction."""
    monkeypatch.delenv(ELEM_BUDGET_ENV, raising=False)
    built = _no_device(monkeypatch)
    max_num_tokens, max_seq_len, _, _ = engine
    core = _core(max_num_tokens, max_seq_len)
    core.build_kernels(0)
    md = _engine(*engine)
    core._derive_first_step(md)
    assert built == [max_seq_len]
    assert core.prefill_logits_budget == M._prefill_logits_budget(max_num_tokens, max_seq_len)
    assert core.prefill_logits_budget == budget
    assert core.ctx_workspace_rows == M._ctx_workspace_rows(md) == rows
    assert core._derived_for is md.kv_cache_manager and core._max_ctx_len == max_seq_len

    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = core
    step = _host_step([(0, max_num_tokens)])
    ctx = target._ctx_workspace(step, max_num_tokens, CPU)
    assert target._ws_buf.shape == (rows, LAT_DIM) and target._ws_buf.dtype == torch.bfloat16
    assert ctx.ws_buf.shape == (max_num_tokens, LAT_DIM) and ctx.kept is not None


# ---- the rope tables ----


def test_rope_tables_hold_no_mla_table():
    """`_rope_tables` returns its rows, the indexer's `[cos | sin]` table and the NoPE indexer
    layers' identity table, nothing else, and takes no `mla` option; a longer table is the same
    table with more rows."""
    alpha = torch.linspace(-0.5, 0.5, 32)
    tables = M._rope_tables(ROPE_CFG, alpha, 300, torch.bfloat16, "cpu")
    assert set(vars(tables)) == {"rows", "indexer", "identity"}
    assert tables.rows == 300
    assert tables.indexer.shape == (300, 64) and tables.indexer.dtype == torch.float32
    assert torch.equal(tables.identity, torch.cat([torch.ones(300, 32), torch.zeros(300, 32)], 1))
    longer = M._rope_tables(ROPE_CFG, alpha, 600, torch.bfloat16, "cpu")
    assert torch.equal(longer.indexer[:300], tables.indexer)
    with pytest.raises(TypeError):
        M._rope_tables(ROPE_CFG, alpha, 300, torch.bfloat16, "cpu", with_mla=True)


def test_rope_tables_built_and_grown_with_no_mla_option(monkeypatch):
    """`build_kernels` builds the tables at the core's rows, and `_derive_first_step` grows them
    to the engine's max_seq_len, both through `_rope_tables`'s own parameters alone."""
    built = _no_device(monkeypatch)
    core = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=ROPE_CFG),
        w={"lerope_alpha": None, "final_norm": torch.empty(0, dtype=torch.bfloat16)},
        rope_rows=16,
        kv_source=KV_SOURCE,
        num_layers=LAYERS,
        _derived_for=None,
        _ix_k=None,
        unfused_glue=False,
        heads=16,
        gen_attn_flash_mla=False,
    )
    M.MistralLarge4Tp4ModelingV2.build_kernels(core, 0)
    md = _engine(64, 64, False, False)
    M.MistralLarge4Tp4ModelingV2._derive_first_step(core, md)
    assert built == [16, 64]
    assert core.rope_rows == 64 and core._rope.rows == 64


# ---- the generation rows' sparse decode


@pytest.mark.parametrize("max_seq_len", [8192, 32768, 32769, 1048576])
@pytest.mark.parametrize("flash_mla", [False, True], ids=["native", "flash_mla"])
def test_native_decode_built_at_every_max_seq_len(monkeypatch, max_seq_len, flash_mla):
    """`build_kernels` compiles cute_dsl_ds_mla_sparse_decode at the rank's heads for the
    generation rows whatever the engine's max_seq_len -- the 32K and the 1M configs alike -- and
    builds no native handle only under `_GEN_ATTN_FLASH_MLA` (the core's `gen_attn_flash_mla`),
    which keeps FlashMLA's 64-head decode. The shipped default is the native kernel."""
    assert M._GEN_ATTN_FLASH_MLA is False
    built = []
    monkeypatch.setattr(M, "_rope_tables", lambda *a: SimpleNamespace(rows=a[2]))
    monkeypatch.setattr(M, "build_ix_q_side", lambda dev, is_neox: None)
    monkeypatch.setattr(
        M, "build_ds_mla_decode", lambda dev, num_heads: built.append(num_heads) or "native"
    )
    core = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=ROPE_CFG),
        w={"lerope_alpha": None, "final_norm": torch.empty(0, dtype=torch.bfloat16)},
        rope_rows=max_seq_len,
        heads=16,
        unfused_glue=False,
        gen_attn_flash_mla=flash_mla,
    )
    M.MistralLarge4Tp4ModelingV2.build_kernels(core, 0)
    assert core._rope.rows == max_seq_len
    assert core._mla_decode == (None if flash_mla else "native")
    assert built == ([] if flash_mla else [16])


# ---- the decode top-k's selector


#: Engine max_seq_lens, the decode scorer's column count, and whether cute_dsl_radix_topk_decode
#: serves it (`_TOPK_DECODE_COLUMNS`, 4096..65536): either side of both ends, the 32K and the 1M
#: configs and 131072.
TOPK_WIDTHS = [
    (2048, False),
    (4095, False),
    (4096, True),
    (8192, True),
    (32768, True),
    (65536, True),
    (65537, False),
    (131072, False),
    (1048576, False),
]


def _topk_md(max_seq_len: int, chunked: bool, reuse: bool) -> SimpleNamespace:
    """A first step's metadata at `max_seq_len` (32768 tokens, 16 requests) on a fresh KV cache
    manager, with chunked prefill and block reuse as given."""
    return SimpleNamespace(
        max_num_tokens=32768,
        max_num_requests=16,
        max_seq_len=max_seq_len,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=chunked, cache_reuse=reuse, chunk_size=32768
        ),
        kv_cache_manager=SimpleNamespace(
            latent_layout=M._LATENT_LAYOUT,
            kv_source_layers=KV_SOURCE,
            latent_pool_row=lambda i: KV_OWNERS.index(KV_SOURCE[i]),
            kv_owner_layers=KV_OWNERS,
        ),
        tokens_per_block=32,
    )


@pytest.mark.parametrize("max_seq_len, serves", TOPK_WIDTHS, ids=[str(w) for w, _ in TOPK_WIDTHS])
def test_decode_topk_selector_by_the_scorer_width(monkeypatch, max_seq_len, serves):
    """`_derive_first_step` holds cute_dsl_radix_topk_decode's handle exactly when the decode
    scorer's column count, the engine's max_seq_len, is one the selector serves (4096..65536);
    otherwise none, and both decode call sites run the trtllm indexer_topk_decode op. Nothing
    else decides it: chunked prefill and block reuse, on or off, keep the one handle, built
    once. A replaced KV cache manager at a width on the other side of the range flips it, and
    `_TOPK_DECODE_CUTE_DSL` False builds none at any width."""
    assert M._TOPK_DECODE_CUTE_DSL is True
    assert M._TOPK_DECODE_COLUMNS == (4096, 65536)
    calls = []
    monkeypatch.setattr(M, "_rope_tables", lambda *a: SimpleNamespace(rows=a[2]))
    monkeypatch.setattr(M, "build_ix_k_cache", lambda dev, is_neox, cache_contiguous: None)
    monkeypatch.setattr(
        M, "build_ix_topk_decode", lambda dev, next_n: calls.append(next_n) or object()
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    core = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=ROPE_CFG),
        w={"lerope_alpha": None, "final_norm": torch.empty(0, dtype=torch.bfloat16)},
        rope_rows=16,
        kv_source=KV_SOURCE,
        num_layers=LAYERS,
        _derived_for=None,
        _ix_k=None,
        _ix_topk=None,
        _rope=SimpleNamespace(rows=16),
    )
    derive = M.MistralLarge4Tp4ModelingV2._derive_first_step
    handles = []
    for chunked, reuse in ((False, False), (True, False), (False, True), (True, True)):
        derive(core, _topk_md(max_seq_len, chunked, reuse))
        handles.append(core._ix_topk)
    if serves:
        assert calls == [1] and handles[0] is not None
        assert all(h is handles[0] for h in handles)
    else:
        assert calls == [] and handles == [None] * 4
    derive(core, _topk_md(1048576 if serves else 32768, False, False))
    assert (core._ix_topk is None) is serves
    monkeypatch.setattr(M, "_TOPK_DECODE_CUTE_DSL", False)
    derive(core, _topk_md(32768, False, False))
    assert core._ix_topk is None


# ---- the indexer's context selection ----


def _packed():
    """Two context requests of 40 cached + 8 new and 30 cached + 6 new tokens, packed by the
    engine into one prefill chunk of 14 rows over 84 keys: the rows' windows `[ks, ke)` in the
    chunk's keys, and the metadata's host prefix sums."""
    ks = torch.tensor([0] * 8 + [48] * 6, dtype=torch.int32)
    ke = torch.tensor([*range(41, 49), *range(79, 85)], dtype=torch.int32)
    chunk = SimpleNamespace(
        token_start=0,
        token_end=14,
        k_token_start=0,
        k_token_end=84,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    md = SimpleNamespace(
        enable_context_mla_with_cached_kv=True,
        host_ctx_uncached_token_indptr=torch.tensor([0, 8, 14]),
        host_ctx_kv_indptr=torch.tensor([0, 48, 84]),
    )
    return chunk, md


def _spans(blocks) -> list:
    return [(b.token_start, b.token_end, b.k_token_start, b.k_token_end) for b in blocks]


def test_ctx_offsets_device_and_host():
    """`_ctx_kv_offsets` hands over mainline's device offsets only; `_ctx_host_offsets` the host
    prefix sums of the new and of the total lengths: the engine's host indptrs under context MLA
    over cached KV, else the prefix sum of the host seq_lens for both."""
    md = SimpleNamespace(
        enable_context_mla_with_cached_kv=False,
        mla_prepare_ctx_cu_seqlens=lambda: torch.tensor([0, 8, 14], dtype=torch.int32),
        seq_lens=torch.tensor([8, 6, 1], dtype=torch.int32),
    )
    assert set(M._ctx_kv_offsets(md, 2)) == {
        "cu_ctx_cached",
        "cu_ctx_kv",
        "max_ctx_new",
        "ctx_cached_tokens",
    }
    assert M._ctx_host_offsets(md, 2) == dict(host_cu_ctx_new=[0, 8, 14], host_cu_ctx_kv=[0, 8, 14])
    _, md = _packed()
    assert M._ctx_host_offsets(md, 2) == dict(
        host_cu_ctx_new=[0, 8, 14], host_cu_ctx_kv=[0, 48, 84]
    )


def test_indexer_blocks_cut_only_past_max_seq_len():
    """The packed chunk of 84 keys: at max_seq_len 64 cut at its requests, each request's
    windows re-based to its own first key; at 84 and at 128 the very chunk, one block over both
    requests. With no chunk planned, one chunk over every context row, cut the same way: two
    fresh prompts of 40 and 30 tokens are one block within max_seq_len 128 and two past 64."""
    chunk, md = _packed()
    step = dict(num_ctx_tokens=14, num_ctx_kv_tokens=84, **M._ctx_host_offsets(md, 2))
    step.update(cu_seqlen_ks=chunk.cu_seqlen_ks, cu_seqlen_ke=chunk.cu_seqlen_ke)

    blocks = M._indexer_blocks([chunk], dict(step, max_seq_len=64))
    assert _spans(blocks) == [(0, 8, 0, 48), (8, 14, 48, 84)]
    assert blocks[0].cu_seqlen_ks.tolist() == [0] * 8
    assert blocks[0].cu_seqlen_ke.tolist() == list(range(41, 49))
    assert blocks[1].cu_seqlen_ks.tolist() == [0] * 6
    assert blocks[1].cu_seqlen_ke.tolist() == list(range(31, 37))
    for max_seq_len in (84, 128):
        (whole,) = M._indexer_blocks([chunk], dict(step, max_seq_len=max_seq_len))
        assert whole is chunk

    fresh = dict(
        num_ctx_tokens=70,
        num_ctx_kv_tokens=70,
        host_cu_ctx_new=[0, 40, 70],
        host_cu_ctx_kv=[0, 40, 70],
        cu_seqlen_ks=torch.tensor([0] * 40 + [40] * 30, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(1, 71, dtype=torch.int32),
    )
    (whole,) = M._indexer_blocks(None, dict(fresh, max_seq_len=128))
    assert _spans([whole]) == [(0, 70, 0, 70)]
    assert torch.equal(whole.cu_seqlen_ke, fresh["cu_seqlen_ke"])
    blocks = M._indexer_blocks(None, dict(fresh, max_seq_len=64))
    assert _spans(blocks) == [(0, 40, 0, 40), (40, 70, 40, 70)]
    assert blocks[1].cu_seqlen_ks.tolist() == [0] * 30
    assert blocks[1].cu_seqlen_ke.tolist() == list(range(1, 31))


def _step_md(**fields) -> SimpleNamespace:
    """The metadata `step_args` reads, for the packed step of `_packed` (two context requests
    over cached prefixes, one engine chunk), with `fields` overriding it."""
    chunk, host = _packed()
    i64 = dict(dtype=torch.int64)
    md = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            get_latent_pool=lambda: torch.zeros(2, 3, 32, 656, dtype=torch.uint8),
            get_indexer_k_cache_buffers=lambda i: None,
        ),
        num_seqs=2,
        num_tokens=14,
        kv_lens_cuda_runtime=None,
        kv_lens_runtime=None,
        prompt_lens_cpu_runtime=None,
        kv_cache_block_offsets=None,
        host_kv_cache_pool_pointers=None,
        host_kv_cache_pool_mapping=None,
        tokens_per_block=32,
        max_seq_len=64,
        block_table=None,
        req_idx_per_token=None,
        slot_mapping_fp8=None,
        slot_mapping_scale=None,
        skip_indexer_for_gen_reqs=False,
        enable_indexer_skip=False,
        indexer_k_cache_block_offsets=None,
        scheduler_metadata_buffer=None,
        radix_aux_indices=None,
        radix_aux_logits=None,
        num_contexts=2,
        num_ctx_tokens=14,
        seq_lens=torch.tensor([8, 6], dtype=torch.int32),
        skip_indexer_for_ctx_reqs=False,
        cu_seqlen_ks=chunk.cu_seqlen_ks,
        cu_seqlen_ke=chunk.cu_seqlen_ke,
        indexer_prefill_q_split=None,
        indexer_prefill_chunks=[chunk],
        num_ctx_kv_tokens=14,
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
        enable_context_mla_with_cached_kv=True,
        ctx_cached_token_indptr=torch.tensor([0, 40, 70], **i64),
        ctx_kv_indptr=torch.tensor([0, 48, 84], **i64),
        max_ctx_seq_len=8,
        num_ctx_cached_tokens=70,
        host_ctx_uncached_token_indptr=host.host_ctx_uncached_token_indptr,
        host_ctx_kv_indptr=host.host_ctx_kv_indptr,
    )
    for name, value in fields.items():
        setattr(md, name, value)
    return md


@pytest.mark.parametrize("max_seq_len, cut", [(64, True), (84, False), (128, False)])
def test_step_args_carries_the_host_prefix_sums(max_seq_len, cut):
    """`step_args` over the packed chunk of 84 keys carries the host prefix sums on every
    context step and cuts the chunk only past max_seq_len keys."""
    md = _step_md(max_seq_len=max_seq_len)
    step = _prefill_target(indexer_layers=[1]).step_args(md)
    assert step["ctx_cached_tokens"] == 70 and int(step["cu_ctx_kv"][-1]) == 84
    assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 48, 84])
    if cut:
        assert _spans(step["indexer_blocks"]) == [(0, 8, 0, 48), (8, 14, 48, 84)]
    else:
        (whole,) = step["indexer_blocks"]
        assert whole is md.indexer_prefill_chunks[0]


@pytest.mark.parametrize(
    "max_seq_len, spans",
    [(16, [(0, 14, 0, 14)]), (10, [(0, 8, 0, 8), (8, 14, 8, 14)])],
    ids=["within_max_seq_len", "past_max_seq_len"],
)
def test_step_args_host_prefix_sums_without_cached_kv(max_seq_len, spans):
    """Without context MLA over cached KV (neither chunked prefill nor block reuse) `step_args`
    takes the host prefix sums off the host seq_lens, and the whole step's block, two fresh
    prompts of 8 and 6 tokens planned by no chunk, is cut at its requests only past max_seq_len
    keys, the second request's windows re-based to its own first key."""
    md = _step_md(
        max_seq_len=max_seq_len,
        enable_context_mla_with_cached_kv=False,
        mla_prepare_ctx_cu_seqlens=lambda: torch.tensor([0, 8, 14], dtype=torch.int32),
        indexer_prefill_chunks=None,
        cu_seqlen_ks=torch.tensor([0] * 8 + [8] * 6, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(1, 15, dtype=torch.int32),
    )
    step = _prefill_target(indexer_layers=[1]).step_args(md)
    assert step["ctx_cached_tokens"] == 0 and step["cu_ctx_kv"].tolist() == [0, 8, 14]
    assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 8, 14])
    blocks = step["indexer_blocks"]
    assert _spans(blocks) == spans
    if len(blocks) == 2:
        assert blocks[1].cu_seqlen_ks.tolist() == [0] * 6
        assert blocks[1].cu_seqlen_ke.tolist() == list(range(1, 7))


def _recorded_kernels(monkeypatch) -> list:
    """Stand-ins for the context selection's kernels, recording each key gather's
    `(k_token_start, n_k)`, each scored tile's `(rows, keys)` and each top-k call's window
    ends. The gathered keys are zero-stride views: nothing of n_k x 128 is allocated. Each
    tile's split-KV schedule is a marker of the windows and keys it was built for, which its
    logits call must be handed (checked here), and the top-k's scratch a marker the target
    must hand every top-k call."""
    calls = []

    def gather(pool, fp8, scale, k_token_start, n_k, dim):
        calls.append(("gather", k_token_start, n_k))
        return torch.zeros(1, dim).expand(n_k, dim), torch.zeros(1, 1).expand(n_k, 1)

    def schedule(ks, ke, n_k, heads):
        return ("schedule", ks.tolist(), ke.tolist(), n_k, heads)

    def logits(q, kv, weights, ks, ke, schedule_meta=None):
        assert schedule_meta == ("schedule", ks.tolist(), ke.tolist(), kv[0].shape[0], q.shape[1])
        calls.append(("logits", q.shape[0], kv[0].shape[0]))
        return torch.empty(0)

    def topk(scores, ks, ke, out, k, workspace=None):
        assert workspace == "gvr-scratch"
        calls.append(("topk", ks.shape[0], int(ke[-1])))

    monkeypatch.setattr(M, "indexer_k_cache_gather_op", gather)
    monkeypatch.setattr(M, "mqa_logits_schedule", schedule)
    monkeypatch.setattr(M, "fp8_mqa_logits", logits)
    monkeypatch.setattr(M, "cute_dsl_gvr_topk_prefill", topk)
    monkeypatch.setattr(M, "gvr_workspace", lambda device: "gvr-scratch")
    monkeypatch.setattr(M, "gvr_warmup", lambda top_k, cols, elems, device, ws: 0)
    return calls


def _prefill_target(**core) -> M.PrefillTarget:
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    # The selection's top-k warm-up reads the engine's max_seq_len off the core.
    core.setdefault("_max_ctx_len", 1 << 20)
    target.core = SimpleNamespace(**core)
    return target


@pytest.mark.parametrize(
    "max_num_tokens, max_seq_len, env, rows, tiles",
    [
        pytest.param(32768, 32768, None, 1024, [512, 512], id="32k"),
        pytest.param(16384, 32768, None, 1024, [512, 512], id="pareto"),
        pytest.param(32768, 131072, None, 4096, [2048, 2048], id="131072"),
        pytest.param(32768, 1048576, None, 512, [256, 256], id="1m"),
        pytest.param(32768, 32768, 1 << 20, 64, [32, 32], id="env_2**20"),
    ],
)
def test_select_ctx_tiles_under_the_budget(
    monkeypatch, max_num_tokens, max_seq_len, env, rows, tiles
):
    """`rows` context rows over max_seq_len keys, scored in tiles of the engine's budget //
    keys: 512 rows at 32768 keys under 2**24, 2048 at 131072 and 256 at 1048576 under 2**28,
    32 at 32768 under TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET 2**20. The keys are gathered once,
    every tile is within the budget, and every row is selected once."""
    if env is None:
        monkeypatch.delenv(ELEM_BUDGET_ENV, raising=False)
    else:
        monkeypatch.setenv(ELEM_BUDGET_ENV, str(env))
    budget = M._prefill_logits_budget(max_num_tokens, max_seq_len)
    calls = _recorded_kernels(monkeypatch)
    keys = max_seq_len
    block = SimpleNamespace(
        token_start=0,
        token_end=rows,
        k_token_start=0,
        k_token_end=keys,
        cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(keys - rows + 1, keys + 1, dtype=torch.int32),
    )
    step = dict(
        indexer_blocks=[block], slot_mapping_fp8_fullkv=None, slot_mapping_scale_fullkv=None
    )
    target = _prefill_target(idx_dim=128, idx_topk=2048, prefill_logits_budget=budget)
    q, w, out = torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(rows, 1)
    target._select_ctx(step, None, q, w, out)
    assert [c for c in calls if c[0] == "gather"] == [("gather", 0, keys)]
    scored = [c[1:] for c in calls if c[0] == "logits"]
    assert [n for n, _ in scored] == tiles
    assert all(n * k <= budget for n, k in scored)
    assert sum(c[1] for c in calls if c[0] == "topk") == rows


def test_select_ctx_block_with_no_key(monkeypatch):
    """A block of rows with no key fills its rows' selection slots with -1 and leaves every
    other row's as they are; nothing is gathered, scored or selected for it."""
    calls = _recorded_kernels(monkeypatch)
    block = SimpleNamespace(
        token_start=2,
        token_end=5,
        k_token_start=7,
        k_token_end=7,
        cu_seqlen_ks=torch.zeros(3, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(3, dtype=torch.int32),
    )
    step = dict(
        indexer_blocks=[block], slot_mapping_fp8_fullkv=None, slot_mapping_scale_fullkv=None
    )
    target = _prefill_target(idx_dim=128, idx_topk=4, prefill_logits_budget=1 << 24)
    out = torch.full((6, 4), 9, dtype=torch.int32)
    target._select_ctx(step, None, torch.empty(6, 1), torch.empty(6, 1), out)
    assert calls == []
    assert out[:2].eq(9).all() and out[5:].eq(9).all()
    assert out[2:5].eq(-1).all()


# ---- the context workspace and the context attention ----


def _host_step(contexts) -> dict:
    """The step fields `_ctx_workspace` and `_attn_ctx` read, for context requests of
    (cached, new) tokens."""
    return dict(
        num_contexts=len(contexts),
        num_ctx_tokens=sum(n for _, n in contexts),
        ctx_cached_tokens=sum(c for c, _ in contexts),
        cu_ctx_kv=_cumsum([c + n for c, n in contexts]),
        latent_pool=None,
        latent_kv=None,
        block_table=None,
        tokens_per_block=32,
    )


def _attn_target(held: int) -> M.PrefillTarget:
    """A prefill target over what its context workspace and attention read off the core: the
    held workspace's rows, and `shared_source` as the core builds it, layer 35 alone (checked
    on the core itself by `test_core_selects_nothing_by_max_seq_len`)."""
    return _prefill_target(
        lat_dim=LAT_DIM,
        ctx_workspace_rows=held,
        shared_source=[i == SHARED_SOURCE for i in range(LAYERS)],
        softmax_scale=1.0,
        idx_topk=4,
        _pool_row=list(range(LAYERS)),
        _stride_factor=1,
    )


def test_context_path_by_the_steps_context_kv():
    """Held rows 64. Steps whose context KV (cached + new over the context requests) is 16, 32,
    40 and 64 take the workspace path, the held buffer's first n_kv rows, with the next n_kv
    kept beside them at 16 and 32 (2 n_kv within 64) and none at 40 and 64; one row past, the
    decode path, with no workspace. Every step reads the one buffer the first allocated."""
    target = _attn_target(64)
    assert target._ws_buf is None
    buf = None
    for contexts, kept in (
        ([(10, 6)], True),
        ([(20, 4), (0, 8)], True),
        ([(30, 10)], False),
        ([(60, 4)], False),
    ):
        step = _host_step(contexts)
        n_kv = step["ctx_cached_tokens"] + step["num_ctx_tokens"]
        ctx = target._ctx_workspace(step, step["num_ctx_tokens"], CPU)
        buf = target._ws_buf if buf is None else buf
        assert target._ws_buf is buf
        assert buf.shape == (64, LAT_DIM) and buf.dtype == torch.bfloat16
        assert ctx.ws_buf.shape == (n_kv, LAT_DIM) and ctx.ws_buf.data_ptr() == buf.data_ptr()
        kv = [c + n for c, n in contexts]
        assert ctx.kv_req.tolist() == [r for r, n in enumerate(kv) for _ in range(n)]
        assert ctx.kv_pos.tolist() == [p for n in kv for p in range(n)]
        assert ctx.ws_off.tolist() == _cumsum(kv)[:-1].tolist() and ctx.ws == {}
        if kept:
            assert ctx.kept.shape == (n_kv, LAT_DIM)
            assert ctx.kept.data_ptr() == buf[n_kv].data_ptr()
        else:
            assert ctx.kept is None

    ctx = target._ctx_workspace(_host_step([(60, 5)]), 5, CPU)
    assert ctx.ws_buf is None and not hasattr(ctx, "kv_req")
    assert ctx.sub_batches == [(0, 5)] and sorted(ctx.sched) == [5]
    assert target._ws_buf is buf


def test_held_workspace_allocated_on_a_first_step_past_it():
    """The held workspace is allocated on the first context step even when that step takes the
    decode path: 20000 new rows past 64 held, in sub-batches of 8192 with one scheduler per
    sub-batch size. The next step, within the held rows, reads that buffer."""
    target = _attn_target(64)
    ctx = target._ctx_workspace(_host_step([(0, 20000)]), 20000, CPU)
    buf = target._ws_buf
    assert buf.shape == (64, LAT_DIM) and buf.dtype == torch.bfloat16
    assert ctx.ws_buf is None
    assert M._CTX_DECODE_ROWS == 8192
    assert ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 20000)]
    assert sorted(ctx.sched) == [3616, 8192]
    assert not any(m.have_initialized for m in ctx.sched.values())
    ctx = target._ctx_workspace(_host_step([(4, 4)]), 4, CPU)
    assert target._ws_buf is buf and ctx.ws_buf.data_ptr() == buf.data_ptr()


@pytest.mark.parametrize("held, kept", [(64, True), (24, False)], ids=["fits_twice", "fits_once"])
def test_nope_source_kept_or_gathered_again_by_fit(monkeypatch, held, kept):
    """Layers 35, 39 (attending over 35's latent), 40 and 43 (over 35's) on two steps of two
    context requests of 10 and 6 KV rows (16). Held 64, so 2 x 16 fits: layer 35 is gathered
    once into the 16 rows kept beside the step's, which 39 and 43 read without a gather, and 40
    into the step's first 16. Held 24, so only 16 fits: every layer gathers its source into the
    step's first 16 rows, 35 again for 39 and for 43. Both steps run on the one held buffer."""
    gathers, prefills = [], []

    def gather(pool, row, kv_req, kv_pos, block_table, out):
        gathers.append((row, out))
        return out

    monkeypatch.setattr(M, "ds_mla_dequant_gather", gather)
    monkeypatch.setattr(
        M, "flash_mla_sparse_prefill", lambda q, ws, idx, scale, out: prefills.append(ws)
    )
    target = _attn_target(held)
    step = _host_step([(7, 3), (0, 6)])
    req_idx = torch.tensor([0] * 3 + [1] * 6)
    sel = torch.zeros(9, 4, dtype=torch.int32)
    buf = None
    for _ in range(2):
        gathers.clear()
        prefills.clear()
        ctx = target._ctx_workspace(step, 9, CPU)
        buf = target._ws_buf if buf is None else buf
        assert target._ws_buf is buf and buf.shape == (held, LAT_DIM)
        assert ctx.ws_buf.shape == (16, LAT_DIM) and ctx.ws_buf.data_ptr() == buf.data_ptr()
        assert ctx.ws == {}
        for layer in (SHARED_SOURCE, 39, 40, 43):
            L = SimpleNamespace(kv_source=KV_SOURCE[layer])
            target._attn_ctx(step, L, torch.empty(9, 1), torch.empty(9, 1), req_idx, sel, ctx)
        if kept:
            assert ctx.kept.shape == (16, LAT_DIM) and ctx.kept.data_ptr() == buf[16].data_ptr()
            assert [row for row, _ in gathers] == [SHARED_SOURCE, 40]
            assert gathers[0][1] is ctx.kept and gathers[1][1] is ctx.ws_buf
            assert ctx.ws == {SHARED_SOURCE: ctx.kept}
            assert [ws is ctx.kept for ws in prefills] == [True, True, False, True]
            assert prefills[2] is ctx.ws_buf
        else:
            assert ctx.kept is None
            assert [row for row, _ in gathers] == [SHARED_SOURCE, SHARED_SOURCE, 40, SHARED_SOURCE]
            assert all(out is ctx.ws_buf for _, out in gathers)
            assert all(ws is ctx.ws_buf for ws in prefills) and ctx.ws == {}


def test_decode_path_attends_off_the_pool_in_sub_batches(monkeypatch):
    """Past the held rows `_attn_ctx` gathers nothing: the selections become pool slots once
    per layer, at the layer's source row, and every sub-batch is one sparse decode call on its
    own size's scheduler; layer 39 reads layer 35's pool row."""
    converted, decoded = [], []

    def convert(req_idx, block_table, topk, tpb, k, stride, pool_row):
        converted.append(pool_row)
        return topk

    def decode(q, kv, slots, sched, scale, out):
        decoded.append((q.shape[0], slots.shape[0], sched))

    def no_gather(*args):
        raise AssertionError("the decode path gathers nothing")

    monkeypatch.setattr(M, "convert_req_index_to_global", convert)
    monkeypatch.setattr(M, "flash_mla_sparse_decode", decode)
    monkeypatch.setattr(M, "ds_mla_dequant_gather", no_gather)
    target = _attn_target(64)
    tc = 20000
    step = _host_step([(5, tc)])
    ctx = target._ctx_workspace(step, tc, CPU)
    assert ctx.ws_buf is None
    sel = torch.zeros(tc, 4, dtype=torch.int32)
    req_idx = torch.zeros(tc, dtype=torch.int64)
    for layer in (36, 39):
        decoded.clear()
        L = SimpleNamespace(kv_source=KV_SOURCE[layer])
        target._attn_ctx(step, L, torch.empty(tc, 1), torch.empty(tc, 1), req_idx, sel, ctx)
        assert decoded == [(n, n, ctx.sched[n]) for n in (8192, 8192, 3616)]
    assert converted == [36, SHARED_SOURCE]
