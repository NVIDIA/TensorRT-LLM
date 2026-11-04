# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-step paths of the Mistral-Large-4-1T-A52B NVFP4 tep4 target, and the sizes it takes from
the engine.

No path is selected by the serving configuration. The core reads nothing off max_seq_len when it
is built: `shared_source` is structural (layer 35 alone, the owner the NoPE layers 39, 43, ..., 71
attend over), and the indexer's logits budget and the held context workspace's rows are 0 until
the first step, when `_derive_first_step` sizes them from the engine's quantities on the attention
metadata (max_num_tokens, max_num_requests, max_seq_len and runtime_features):

* the logits budget, `_prefill_logits_budget(max_num_tokens, max_seq_len)`: mainline's reachable
  MQA-logits tile (`get_indexer_mqa_logits_workspace_bytes`, which
  TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET can only lower), within `_PREFILL_LOGITS_CEILING` (2**28)
  and within max_seq_len / 64 query rows at max_seq_len keys: 2**24 at max_seq_len 32768 (512-row
  tiles at 32768 keys), the ceiling from max_seq_len 131072 on (2048-row tiles at 131072 keys, 256
  at 1048576);
* the held workspace's rows, `_ctx_workspace_rows`: twice the most context KV one step can attend
  (`_most_ctx_kv`: max_num_tokens with neither chunked prefill nor block reuse, else
  min(max_num_requests, max_num_tokens) x max_seq_len), up to `_CTX_WORKSPACE_ROWS` (2**19):
  65536 for the 32768 serving config (max_num_tokens = max_seq_len = 32768, chunked prefill and
  block reuse off), 2**19 with either at 16 requests of max_seq_len 32768 or more.

Chunked prefill and block reuse are the metadata's runtime_features alone, which the executor
builds once for the service (`AttentionRuntimeFeatures(chunked_prefill=enable_chunked_prefill,
cache_reuse=kv_cache_config.enable_block_reuse)` in py_executor_creator, turned off again where
it disables either): the KV cache manager is not consulted for them, and a user's
fp8_context_mla_kv_len_cap is not read.

Each context step then picks its own path by its context KV n_kv, every context request's cached
and new rows (`PrefillTarget._ctx_workspace`): while n_kv fits the held rows, FlashMLA's sparse
prefill over those rows dequantized into the held workspace's first n_kv rows, layer 35's gather
kept in the next n_kv rows for its sharers when n_kv fits twice and gathered again by each sharer
when it fits once (`_attn_ctx`); past the held rows, the sparse decode over the pool in sub-batches
of at most 8192 context rows. The workspace is allocated once, on the first context step, and its
storage is the one every later step uses. The context indexer cuts an engine chunk at its requests
only when it holds more than max_seq_len keys, over the host prefix sums `step_args` carries on
every context step (`_ctx_host_offsets`), scores each block in tiles under the budget, and fills
the rows of a block with no key with -1 (`_select_ctx`). The rope tables hold no `mla` table.

No device: the core is built on the meta device from the checkpoint's native config, params.json
(skipped where it is absent under LLM_MODELS_ROOT); everything else runs on host values, with
recording stand-ins for the kernels the paths call.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures

pytestmark = pytest.mark.cpu_only

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
#: The checkpoint's layers, the owner its NoPE layers 39, 43, ..., 71 attend over, each layer's
#: latent source, the 63 KV owners and the latent row's width.
LAYERS = 72
SHARED_SOURCE = 35
KV_SOURCE = [SHARED_SOURCE if i > SHARED_SOURCE and i % 4 == 3 else i for i in range(LAYERS)]
OWNERS = [i for i in range(LAYERS) if KV_SOURCE[i] == i]
LAT_DIM = 576
#: Engine max_seq_lens a core is built at -- below, at and one past 32768, 1048576, and unset
#: (the engine infers one of at least the checkpoint's max_position_embeddings, 1048576) -- none
#: of which moves anything the core holds before its first step.
MAX_SEQ_LENS = [8192, 32768, 32769, 1048576, None]
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
#: Mainline's override of the reachable MQA-logits tile, in logits.
ENV_BUDGET = "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET"
#: Serving configs by their numbers, (max_num_tokens, max_seq_len, max_batch_size, chunked
#: prefill, block reuse), and what the target takes from each: the most context KV one step
#: attends (`_most_ctx_kv`), the held workspace's rows (`_ctx_workspace_rows`) and the logits
#: budget (`_prefill_logits_budget`).
CONFIGS = {
    "32k": ((32768, 32768, 16, False, False), (32768, 1 << 16, 1 << 24)),
    "16k_tokens-32k": ((16384, 32768, 16, False, False), (16384, 1 << 15, 1 << 24)),
    "32k-block_reuse": ((32768, 32768, 16, False, True), (1 << 19, 1 << 19, 1 << 24)),
    "1m-chunked": ((32768, 1048576, 16, True, False), (1 << 24, 1 << 19, 1 << 28)),
    "1m-chunked-block_reuse": ((32768, 1048576, 16, True, True), (1 << 24, 1 << 19, 1 << 28)),
    "131072-chunked": ((32768, 131072, 16, True, False), (1 << 21, 1 << 19, 1 << 28)),
    "131072-unchunked": ((32768, 131072, 16, False, False), (32768, 1 << 16, 1 << 28)),
}
#: The 32768 config with neither chunked prefill nor block reuse and the 1M config, by their
#: numbers as above.
PLAIN_32K, CHUNKED_1M = (CONFIGS[c][0] for c in ("32k", "1m-chunked"))


def _sizing(max_num_tokens, max_seq_len, max_batch_size, chunked, reuse, **fields):
    """The engine's quantities on the attention metadata, as `build_attention_metadata` fills
    them (max_num_requests is the engine's max_batch_size, the runtime features its chunked
    prefill and block reuse) and the KV cache manager states max_seq_len."""
    return SimpleNamespace(
        max_num_tokens=max_num_tokens,
        max_num_requests=max_batch_size,
        max_seq_len=max_seq_len,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=chunked, cache_reuse=reuse, chunk_size=max_num_tokens
        ),
        **fields,
    )


def _core(max_seq_len: int | None):
    """The core of rank 0 at four ranks, on the meta device, for an engine at `max_seq_len`."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm.mapping import Mapping

    root = llm_models_root()
    paths = [root / name for name in CHECKPOINTS] if root is not None else []
    path = next((p for p in paths if (p / "params.json").is_file()), None)
    if path is None:
        pytest.skip(f"none of {CHECKPOINTS} is under LLM_MODELS_ROOT ({root})")
    mapping = Mapping(
        world_size=4, rank=0, gpus_per_node=4, tp_size=4, pp_size=1, moe_ep_size=4, moe_tp_size=1
    )
    model_config = ModelConfig(
        pretrained_config=MistralConfigLoader().load_pretrained_config(str(path)),
        mapping=mapping,
        max_num_tokens=32768,
        max_seq_len=max_seq_len,
    )
    with torch.device("meta"):
        return M.MistralLarge4Tep4ModelingV2(model_config)


def _manager():
    """The fp8_ds_mla KV cache manager's surface `_derive_first_step` reads: its layout, the
    layers' latent sources and owner rows and the KV owners. Nothing else: it holds no
    KvCacheConfig, so a read of one fails."""
    return SimpleNamespace(
        latent_layout=M._LATENT_LAYOUT,
        kv_source_layers=KV_SOURCE,
        latent_pool_row=lambda i: OWNERS.index(KV_SOURCE[i]),
        kv_owner_layers=OWNERS,
    )


@pytest.fixture
def built(monkeypatch) -> list:
    """Stand-ins for what `build_kernels` and `_derive_first_step` build on a device. The rope
    tables are recorded by their rows through a stand-in with `_rope_tables`'s signature and no
    other parameter, so a call asking for any other table fails."""
    rows = []

    def tables(cfg, alpha, positions, dtype, device):
        rows.append(positions)
        return SimpleNamespace(rows=positions)

    monkeypatch.setattr(M, "_rope_tables", tables)
    monkeypatch.setattr(M, "build_ix_q_side", lambda dev, is_neox: None)
    monkeypatch.setattr(M, "build_ix_k_cache", lambda dev, is_neox, cache_contiguous: None)
    monkeypatch.setattr(M, "build_ds_mla_decode", lambda dev, num_heads: None)
    monkeypatch.setattr(
        M, "build_ix_topk_decode", lambda dev, next_n: SimpleNamespace(next_n=next_n)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return rows


# ---- the core: nothing selected off max_seq_len, its sizes from the first step


def test_module_holds_no_switch():
    """The switch and the budgets only it chose between are gone; what remains is fixed: the
    budget's ceiling (2**28 logits, 256 rows at 1048576 keys), the held workspace's ceiling
    (2**19 rows) and the decode path's sub-batch (8192 rows). `_indexer_blocks` takes the chunk
    plan and the step only, and `_rope_tables` builds no `mla` table on request."""
    removed = ("_LONG_CONTEXT_THRESHOLD", "_LONG_PREFILL_LOGITS_BUDGET", "_PREFILL_LOGITS_BUDGET")
    assert not [name for name in removed if hasattr(M, name)]
    assert M._PREFILL_LOGITS_CEILING == 1 << 28
    assert (M._CTX_WORKSPACE_ROWS, M._CTX_DECODE_ROWS) == (1 << 19, 8192)
    assert list(inspect.signature(M._indexer_blocks).parameters) == ["chunks", "step"]
    assert list(inspect.signature(M._rope_tables).parameters) == [
        "cfg",
        "alpha",
        "positions",
        "dtype",
        "device",
    ]


@pytest.mark.parametrize("max_seq_len", MAX_SEQ_LENS, ids=[f"max_seq_len{n}" for n in MAX_SEQ_LENS])
def test_core_reads_nothing_off_max_seq_len(max_seq_len):
    """Whatever the engine's max_seq_len, the core is built the same: no switch (`long_context`
    is gone), layer 35 the one shared source, and no logits budget or held workspace rows until
    the first step. The max_seq_len sets the rope tables' first size only."""
    core = _core(max_seq_len)
    assert not hasattr(core, "long_context")
    # The fused attention glue at every max_seq_len, unless `_UNFUSED_GLUE`.
    assert core.unfused_glue is M._UNFUSED_GLUE
    # The native generation-row decode at every max_seq_len, unless `_GEN_ATTN_FLASH_MLA`.
    assert core.gen_attn_flash_mla is M._GEN_ATTN_FLASH_MLA
    assert core.kv_source == KV_SOURCE
    assert [i for i, shared in enumerate(core.shared_source) if shared] == [SHARED_SOURCE]
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (0, 0)
    assert core.rope_rows == (max_seq_len or 1048576)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_first_step_sizes_the_core(monkeypatch, built, config):
    """On the core built from the checkpoint's config at the engine's max_seq_len: `build_kernels`
    builds the rope tables at max_seq_len rows with no `mla` table asked for, and
    `_derive_first_step`, handed the engine's metadata on the first step, sets the logits budget
    and the held workspace's rows from max_num_tokens, max_num_requests, max_seq_len and the
    runtime features -- 2**24 and 65536 at the 32768 serving config, 2**28 and 2**19 at the 1M
    config (max_seq_len 1048576, chunked prefill) -- without growing the tables."""
    monkeypatch.delenv(ENV_BUDGET, raising=False)
    sizes, (_, rows, budget) = CONFIGS[config]
    core = _core(sizes[1])
    core.build_kernels(0)
    assert built == [sizes[1]]
    core._derive_first_step(_sizing(*sizes, kv_cache_manager=_manager(), tokens_per_block=32))
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (budget, rows)
    assert core._max_ctx_len == sizes[1] and core._stride_factor == 63 * 32
    assert built == [sizes[1]]


def test_first_step_grows_the_tables_and_rederives_for_a_new_manager(monkeypatch, built):
    """On a stand-in core with what `build_kernels` and `_derive_first_step` read: the tables
    built at 16 rows grow to an engine max_seq_len of 64 on the first step, neither call asking
    for an `mla` table. The sizes are fixed for the KV cache manager they were derived with -- a
    later step's metadata on the same manager moves nothing -- and derived again for a replaced
    one."""
    monkeypatch.delenv(ENV_BUDGET, raising=False)
    core = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=ROPE_CFG),
        w={"lerope_alpha": None, "final_norm": torch.empty(0, dtype=torch.bfloat16)},
        rope_rows=16,
        kv_source=KV_SOURCE,
        num_layers=LAYERS,
        _derived_for=None,
        _ix_k=None,
        prefill_logits_budget=0,
        ctx_workspace_rows=0,
        unfused_glue=False,
        heads=16,
        gen_attn_flash_mla=False,
    )
    M.MistralLarge4Tep4ModelingV2.build_kernels(core, 0)
    mgr = _manager()
    small = (32768, 64, 16, False, False)
    M.MistralLarge4Tep4ModelingV2._derive_first_step(
        core, _sizing(*small, kv_cache_manager=mgr, tokens_per_block=32)
    )
    assert built == [16, 64] and core._rope.rows == 64
    # 64 rows of 64 keys: max_seq_len x (max_seq_len >> 6) binds.
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (64, 1 << 16)

    sizes, (_, rows, budget) = CONFIGS["1m-chunked"]
    M.MistralLarge4Tep4ModelingV2._derive_first_step(
        core, _sizing(*sizes, kv_cache_manager=mgr, tokens_per_block=32)
    )
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (64, 1 << 16)
    assert built == [16, 64]
    M.MistralLarge4Tep4ModelingV2._derive_first_step(
        core, _sizing(*sizes, kv_cache_manager=_manager(), tokens_per_block=32)
    )
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (budget, rows)
    assert built == [16, 64, 1048576]


def test_first_step_sizes_the_held_rows_from_the_runtime_features(monkeypatch, built):
    """On the core built from the checkpoint's config at max_seq_len 32768, each first step of
    the 32768 config (32768 tokens, 16 requests) on a new KV cache manager: the held rows follow
    that step's metadata's runtime features, derived again for each manager. Block reuse on: the
    never-stall worst case, 16 x 32768 rows of context KV, 2**19 held; off: every context row
    new, 65536 held; chunked prefill on without block reuse: 2**19 again. The logits budget,
    from max_num_tokens and max_seq_len alone, stays 2**24."""
    monkeypatch.delenv(ENV_BUDGET, raising=False)
    core = _core(32768)
    core.build_kernels(0)
    for chunked, reuse, rows in (
        (False, True, 1 << 19),
        (False, False, 1 << 16),
        (True, False, 1 << 19),
        (False, True, 1 << 19),
    ):
        mgr = _manager()
        md = _sizing(*PLAIN_32K[:3], chunked, reuse, kv_cache_manager=mgr, tokens_per_block=32)
        core._derive_first_step(md)
        assert core._derived_for is mgr
        assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (1 << 24, rows), (
            chunked,
            reuse,
        )
    assert built == [32768]


# ---- the sizes, from the engine's quantities


@pytest.mark.parametrize("config", list(CONFIGS))
def test_held_workspace_rows_from_the_runtime_features(config):
    """`_most_ctx_kv` and `_ctx_workspace_rows` on each config's metadata. With neither chunked
    prefill nor block reuse every context row of a step is new: max_num_tokens of KV, twice that
    held (65536 at 32768 tokens, whatever max_seq_len). Either one lets a step attend cached
    prefixes: min(max_num_requests, max_num_tokens) x max_seq_len, mainline's never-stall worst
    case, and the held rows reach their 2**19 ceiling (the 32768 config with block reuse at
    max_batch_size 16, the 1M config, 131072 with chunked prefill)."""
    sizes, (most, rows, _) = CONFIGS[config]
    md = _sizing(*sizes)
    assert M._most_ctx_kv(md) == most
    assert M._ctx_workspace_rows(md) == rows == min(2 * most, M._CTX_WORKSPACE_ROWS)


def test_block_reuse_is_read_from_the_runtime_features():
    """Block reuse is the metadata's `runtime_features.cache_reuse`, the executor's
    `kv_cache_config.enable_block_reuse` (turned off again where it disables reuse). On the
    32768 config (16 requests, chunked prefill off): on, the never-stall worst case, 16 x 32768
    of context KV and 2**19 held rows; off, max_num_tokens and 65536 rows. Under chunked prefill
    (the 1M config) the worst case holds either way. The metadata carries no KV cache manager
    here: nothing else is read for it."""
    for reuse, most in ((True, 1 << 19), (False, 32768)):
        md = _sizing(*PLAIN_32K[:4], reuse)
        assert not hasattr(md, "kv_cache_manager")
        assert md.runtime_features.cache_reuse is reuse
        assert M._most_ctx_kv(md) == most, reuse
        assert M._ctx_workspace_rows(md) == min(2 * most, 1 << 19)
        md = _sizing(*CHUNKED_1M[:4], reuse)
        assert M._most_ctx_kv(md) == 1 << 24 and M._ctx_workspace_rows(md) == 1 << 19
    # The same on the runtime features the executor flips in place when it disables reuse.
    md = _sizing(*PLAIN_32K[:4], True)
    md.runtime_features.cache_reuse = False
    assert M._most_ctx_kv(md) == 32768 and M._ctx_workspace_rows(md) == 1 << 16


@pytest.mark.parametrize(
    "max_num_tokens, max_seq_len, budget, rows",
    [
        pytest.param(32768, 32768, 1 << 24, 512, id="32k"),
        pytest.param(16384, 32768, 1 << 24, 512, id="16k_tokens-32k"),
        pytest.param(32768, 131072, 1 << 28, 2048, id="131072"),
        pytest.param(8192, 1048576, 1 << 28, 256, id="8k_tokens-1m"),
        pytest.param(32768, 1048576, 1 << 28, 256, id="1m"),
    ],
)
def test_logits_budget_from_the_engine(monkeypatch, max_num_tokens, max_seq_len, budget, rows):
    """`_prefill_logits_budget`: the reachable tile (max_num_tokens x max_seq_len, at most
    mainline's 2**31 default) within the 2**28 ceiling and within max_seq_len / 64 rows at
    max_seq_len keys. 2**24 at max_seq_len 32768, at 32768 or 16384 tokens: 512-row tiles of
    32768 keys. The ceiling from 131072 on: 2048 rows of 131072 keys, 256 of 1048576. A step of
    max_num_tokens rows against max_seq_len keys is scored in tiles of those rows, the last one
    short."""
    monkeypatch.delenv(ENV_BUDGET, raising=False)
    assert M._prefill_logits_budget(max_num_tokens, max_seq_len) == budget
    assert budget <= M._PREFILL_LOGITS_CEILING
    tiles = M._prefill_tiles(0, max_num_tokens, max_seq_len, budget)
    assert budget // max_seq_len == rows and len(tiles) == -(-max_num_tokens // rows)
    assert all(hi - lo == rows for lo, hi in tiles[:-1]) and tiles[-1][1] == max_num_tokens


def test_logits_budget_env_override_only_lowers(monkeypatch):
    """TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET lowers the reachable tile: at 2**20 every config's
    budget is 2**20 (32-row tiles at 32768 keys). Above the other bounds it raises none: at 2**34
    the 32768 config stays at 2**24 and the 1M config at the ceiling."""
    monkeypatch.setenv(ENV_BUDGET, str(1 << 20))
    for sizes in ((32768, 32768), (16384, 32768), (32768, 131072), (32768, 1048576)):
        assert M._prefill_logits_budget(*sizes) == 1 << 20, sizes
    assert M._prefill_tiles(0, 1024, 32768, 1 << 20)[0] == (0, 32)
    monkeypatch.setenv(ENV_BUDGET, str(1 << 34))
    assert M._prefill_logits_budget(32768, 32768) == 1 << 24
    assert M._prefill_logits_budget(32768, 1048576) == 1 << 28


def test_rope_tables_hold_no_mla_table():
    """`_rope_tables` builds the indexer's `[cos | sin]` rows and the NoPE layers' identity rows,
    nothing else: the MLA rope ops' duplicated layout, which no op reads, is not built, and
    asking for it is an error."""
    alpha = torch.linspace(-0.5, 0.5, 32)
    tables = M._rope_tables(ROPE_CFG, alpha, 300, torch.bfloat16, "cpu")
    assert sorted(vars(tables)) == ["identity", "indexer", "rows"]
    assert tables.indexer.shape == tables.identity.shape == (300, 64)
    with pytest.raises(TypeError):
        M._rope_tables(ROPE_CFG, alpha, 300, torch.bfloat16, "cpu", with_mla=True)


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
    M.MistralLarge4Tep4ModelingV2.build_kernels(core, 0)
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
            latent_pool_row=lambda i: OWNERS.index(KV_SOURCE[i]),
            kv_owner_layers=OWNERS,
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
    derive = M.MistralLarge4Tep4ModelingV2._derive_first_step
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


# ---- the indexer's context selection


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


def _spans(blocks) -> list[tuple[int, int, int, int]]:
    return [(b.token_start, b.token_end, b.k_token_start, b.k_token_end) for b in blocks]


def test_ctx_offsets():
    """`_ctx_kv_offsets` hands over the context rope/append op's device offsets; the host prefix
    sums of the new and the total lengths come from `_ctx_host_offsets`, off the host
    `seq_lens` without cached KV and off the metadata's host copies with it."""
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
    """A chunk of more than max_seq_len keys (84 past 64) is cut at its requests, each request's
    rows against its own keys with their windows re-based to its first key. A chunk within
    max_seq_len (84 at 84 and at 128) is passed through as the one block it is, though it holds
    two requests. With no chunk planned, one chunk over every context row, cut by the same
    rule."""
    chunk, md = _packed()
    step = dict(num_ctx_tokens=14, num_ctx_kv_tokens=84, **M._ctx_host_offsets(md, 2))
    step.update(cu_seqlen_ks=chunk.cu_seqlen_ks, cu_seqlen_ke=chunk.cu_seqlen_ke)

    for chunks in ([chunk], None):
        blocks = M._indexer_blocks(chunks, dict(step, max_seq_len=64))
        assert _spans(blocks) == [(0, 8, 0, 48), (8, 14, 48, 84)]
        assert blocks[0].cu_seqlen_ks.tolist() == [0] * 8
        assert blocks[0].cu_seqlen_ke.tolist() == list(range(41, 49))
        assert blocks[1].cu_seqlen_ks.tolist() == [0] * 6
        assert blocks[1].cu_seqlen_ke.tolist() == list(range(31, 37))

    for max_seq_len in (84, 128):
        (whole,) = M._indexer_blocks([chunk], dict(step, max_seq_len=max_seq_len))
        assert whole is chunk
        (whole,) = M._indexer_blocks(None, dict(step, max_seq_len=max_seq_len))
        assert _spans([whole]) == [(0, 14, 0, 84)]
        assert torch.equal(whole.cu_seqlen_ke, chunk.cu_seqlen_ke)


def _step_md(max_seq_len: int, cached_kv: bool):
    """A context step's metadata, every field `step_args` reads. With cached KV, the packed
    chunk of `_packed` and the host prefix sums; without, two prompts of 8 and 6 tokens the
    engine planned no chunk for."""
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
        max_seq_len=max_seq_len,
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
        indexer_prefill_q_split=None,
        num_ctx_kv_tokens=14,
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
        enable_context_mla_with_cached_kv=cached_kv,
    )
    if cached_kv:
        chunk, host = _packed()
        vars(md).update(
            cu_seqlen_ks=chunk.cu_seqlen_ks,
            cu_seqlen_ke=chunk.cu_seqlen_ke,
            indexer_prefill_chunks=[chunk],
            ctx_cached_token_indptr=torch.tensor([0, 40, 70], **i64),
            ctx_kv_indptr=torch.tensor([0, 48, 84], **i64),
            max_ctx_seq_len=8,
            num_ctx_cached_tokens=70,
            host_ctx_uncached_token_indptr=host.host_ctx_uncached_token_indptr,
            host_ctx_kv_indptr=host.host_ctx_kv_indptr,
        )
        return md
    vars(md).update(
        cu_seqlen_ks=torch.tensor([0] * 8 + [8] * 6, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([*range(1, 9), *range(9, 15)], dtype=torch.int32),
        indexer_prefill_chunks=None,
        mla_prepare_ctx_cu_seqlens=lambda: torch.tensor([0, 8, 14], dtype=torch.int32),
    )
    return md


def _prefill_target(**core) -> M.PrefillTarget:
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    # The selection's top-k warm-up reads the engine's max_seq_len off the core.
    core.setdefault("_max_ctx_len", 1 << 20)
    target.core = SimpleNamespace(**core)
    return target


def test_step_args_carry_the_host_prefix_sums():
    """`step_args` takes the host prefix sums on every context step and hands over the scoring
    blocks they cut. With cached KV (the packed chunk of two requests over 84 keys): cut at
    max_seq_len 64, the engine's chunk itself at 128. Without (two fresh prompts of 8 and 6
    tokens, no chunk planned, the sums off the host `seq_lens`): cut at max_seq_len 8, one block
    over the 14 rows at 32."""
    target = _prefill_target(indexer_layers=[1])

    step = target.step_args(_step_md(64, cached_kv=True))
    assert step["ctx_cached_tokens"] == 70 and int(step["cu_ctx_kv"][-1]) == 84
    assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 48, 84])
    assert _spans(step["indexer_blocks"]) == [(0, 8, 0, 48), (8, 14, 48, 84)]
    md = _step_md(128, cached_kv=True)
    step = target.step_args(md)
    assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 48, 84])
    (whole,) = step["indexer_blocks"]
    assert whole is md.indexer_prefill_chunks[0]

    step = target.step_args(_step_md(8, cached_kv=False))
    assert step["ctx_cached_tokens"] == 0 and step["cu_ctx_kv"].tolist() == [0, 8, 14]
    assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 8, 14])
    blocks = step["indexer_blocks"]
    assert _spans(blocks) == [(0, 8, 0, 8), (8, 14, 8, 14)]
    assert blocks[1].cu_seqlen_ks.tolist() == [0] * 6
    assert blocks[1].cu_seqlen_ke.tolist() == list(range(1, 7))
    step = target.step_args(_step_md(32, cached_kv=False))
    assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 8, 14])
    assert _spans(step["indexer_blocks"]) == [(0, 14, 0, 14)]


def _recorded_kernels(monkeypatch) -> list:
    """Stand-ins for the context selection's kernels, recording each key gather's
    `(k_token_start, n_k)`, each scored tile's rows and keys and each top-k call's windows.
    Each tile's split-KV schedule is a marker of the windows and keys it was built for, which
    its logits call must be handed (checked here), and the top-k's scratch a marker the target
    must hand every top-k call."""
    calls = []

    def gather(pool, fp8, scale, k_token_start, n_k, dim):
        calls.append(("gather", k_token_start, n_k))
        return torch.empty(n_k, dim), torch.empty(n_k, 1)

    def schedule(ks, ke, n_k, heads):
        return ("schedule", ks.tolist(), ke.tolist(), n_k, heads)

    def logits(q, kv, weights, ks, ke, schedule_meta=None):
        assert schedule_meta == ("schedule", ks.tolist(), ke.tolist(), kv[0].shape[0], q.shape[1])
        calls.append(("logits", q.shape[0], kv[0].shape[0]))
        return torch.empty(0)

    def topk(scores, ks, ke, out, k, workspace=None):
        assert workspace == "gvr-scratch"
        calls.append(("topk", ks.tolist(), ke.tolist()))

    monkeypatch.setattr(M, "indexer_k_cache_gather_op", gather)
    monkeypatch.setattr(M, "mqa_logits_schedule", schedule)
    monkeypatch.setattr(M, "fp8_mqa_logits", logits)
    monkeypatch.setattr(M, "cute_dsl_gvr_topk_prefill", topk)
    monkeypatch.setattr(M, "gvr_workspace", lambda device: "gvr-scratch")
    monkeypatch.setattr(M, "gvr_warmup", lambda top_k, cols, elems, device, ws: 0)
    return calls


@pytest.mark.parametrize(
    "max_num_tokens, max_seq_len, env_budget, tile",
    [
        pytest.param(32768, 32768, None, 512, id="32k"),
        pytest.param(16384, 32768, None, 512, id="16k_tokens-32k"),
        pytest.param(32768, 131072, None, 8192, id="131072"),
        pytest.param(32768, 1048576, None, 8192, id="1m"),
        pytest.param(32768, 32768, 1 << 20, 32, id="32k-env_2_20"),
    ],
)
def test_select_ctx_tiles_under_the_engines_budget(
    monkeypatch, max_num_tokens, max_seq_len, env_budget, tile
):
    """1024 context rows over 32768 keys, scored in tiles of budget / 32768 rows under the
    budget the core takes from the engine: two 512-row tiles under 2**24 (the 32768 configs),
    one under 2**28 (131072 and 1M), 32 tiles of 32 rows with the override at 2**20. The keys
    are gathered once for the block."""
    if env_budget is None:
        monkeypatch.delenv(ENV_BUDGET, raising=False)
    else:
        monkeypatch.setenv(ENV_BUDGET, str(env_budget))
    calls = _recorded_kernels(monkeypatch)
    rows, keys = 1024, 32768
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
    budget = M._prefill_logits_budget(max_num_tokens, max_seq_len)
    target = _prefill_target(idx_dim=128, idx_topk=2048, prefill_logits_budget=budget)
    q, w, out = torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(rows, 1)
    target._select_ctx(step, None, q, w, out)
    assert [c for c in calls if c[0] == "gather"] == [("gather", 0, keys)]
    tiles = [c[1] for c in calls if c[0] == "logits"]
    assert tiles == [min(tile, rows)] * -(-rows // tile)
    assert all(c[2] == keys for c in calls if c[0] == "logits")


def test_select_ctx_fills_a_block_with_no_key(monkeypatch):
    """A block with rows but no key (n_k = 0) gathers, scores and selects nothing: its rows'
    selection slots are filled with -1, which every attention kernel skips. A block with no row
    is skipped and writes nothing; a block with keys runs as ever."""
    calls = _recorded_kernels(monkeypatch)
    blocks = [
        SimpleNamespace(
            token_start=0,
            token_end=3,
            k_token_start=5,
            k_token_end=5,
            cu_seqlen_ks=torch.zeros(3, dtype=torch.int32),
            cu_seqlen_ke=torch.zeros(3, dtype=torch.int32),
        ),
        SimpleNamespace(
            token_start=3,
            token_end=3,
            k_token_start=0,
            k_token_end=8,
            cu_seqlen_ks=torch.zeros(0, dtype=torch.int32),
            cu_seqlen_ke=torch.zeros(0, dtype=torch.int32),
        ),
        SimpleNamespace(
            token_start=3,
            token_end=6,
            k_token_start=5,
            k_token_end=13,
            cu_seqlen_ks=torch.zeros(3, dtype=torch.int32),
            cu_seqlen_ke=torch.tensor([6, 7, 8], dtype=torch.int32),
        ),
    ]
    step = dict(indexer_blocks=blocks, slot_mapping_fp8_fullkv=None, slot_mapping_scale_fullkv=None)
    target = _prefill_target(idx_dim=128, idx_topk=4, prefill_logits_budget=1 << 24)
    out = torch.full((6, 4), 7, dtype=torch.int32)
    target._select_ctx(step, None, torch.empty(6, 1), torch.empty(6, 1), out)
    assert bool((out[:3] == -1).all()) and bool((out[3:] == 7).all())
    assert calls == [("gather", 5, 8), ("logits", 3, 8), ("topk", [0, 0, 0], [6, 7, 8])]


# ---- the context path, the held workspace and the context attention


def _attn_target(held: int) -> M.PrefillTarget:
    """A prefill target over what its context workspace and attention read off the core: the
    held workspace's rows as `_derive_first_step` sets them, and layer 35 the one shared source,
    as the core builds it (`test_core_reads_nothing_off_max_seq_len`)."""
    return _prefill_target(
        lat_dim=LAT_DIM,
        ctx_workspace_rows=held,
        shared_source=[i == SHARED_SOURCE for i in range(LAYERS)],
        softmax_scale=1.0,
        idx_topk=4,
        _pool_row=list(range(LAYERS)),
        _stride_factor=1,
    )


def _ctx_step(kv_lens: list[int], new: int) -> dict:
    """A context step's entries the context path reads: requests of `kv_lens` cached and new rows
    each, `new` of them this step's."""
    return dict(
        num_contexts=len(kv_lens),
        num_ctx_tokens=new,
        ctx_cached_tokens=sum(kv_lens) - new,
        cu_ctx_kv=torch.tensor([0, *kv_lens]).cumsum(0),
        latent_pool=None,
        latent_kv=None,
        block_table=None,
        tokens_per_block=32,
    )


def _row_ptr(buf: torch.Tensor, row: int) -> int:
    return buf.data_ptr() + row * buf.stride(0) * buf.element_size()


def test_context_path_by_the_steps_kv():
    """Steps on one target whose held workspace is 64 rows, by their context KV n_kv: 16 and 32
    fit twice (the workspace path over rows [0, n_kv), rows [n_kv, 2 n_kv) kept); 33 and 64 fit
    once (the workspace path, nothing kept); 65 does not fit (the decode path, the context rows
    in one sub-batch). The first step allocates the bf16 [64, 576] workspace, and every later
    step, either path, uses that very storage. Each request's rows start at its cumulative KV
    offset."""
    target = _attn_target(64)
    cpu = torch.device("cpu")
    first = target._ctx_workspace(_ctx_step([10, 6], 9), 9, cpu)
    buf = target._ws_buf
    assert buf.shape == (64, LAT_DIM) and buf.dtype == torch.bfloat16
    assert first.kv_req.tolist() == [0] * 10 + [1] * 6
    assert first.kv_pos.tolist() == [*range(10), *range(6)]
    assert first.ws_off.tolist() == [0, 10] and first.ws == {}
    for kv_lens, new, kept in (
        ([10, 6], 9, True),
        ([20, 12], 12, True),
        ([30, 3], 5, False),
        ([40, 24], 24, False),
    ):
        n_kv = sum(kv_lens)
        ctx = target._ctx_workspace(_ctx_step(kv_lens, new), new, cpu)
        assert target._ws_buf is buf
        assert ctx.ws_buf.data_ptr() == buf.data_ptr() and ctx.ws_buf.shape == (n_kv, LAT_DIM)
        assert ctx.kv_req.shape == ctx.kv_pos.shape == (n_kv,)
        if kept:
            assert ctx.kept.shape == (n_kv, LAT_DIM)
            assert ctx.kept.data_ptr() == _row_ptr(buf, n_kv)
        else:
            assert ctx.kept is None
    past = target._ctx_workspace(_ctx_step([40, 25], 25), 25, cpu)
    assert past.ws_buf is None and past.sub_batches == [(0, 25)] and sorted(past.sched) == [25]
    assert target._ws_buf is buf


def test_decode_path_sub_batches_and_its_first_step_allocates():
    """A first context step past the held rows (20000 new rows over 4096 cached, held 2**14)
    takes the decode path and still allocates the held workspace, so the KV-capacity estimate's
    peak holds it whichever path its step takes. The context rows run in sub-batches of 8192, one
    FlashMLA scheduler per sub-batch size."""
    target = _attn_target(1 << 14)
    ctx = target._ctx_workspace(_ctx_step([24096], 20000), 20000, torch.device("cpu"))
    assert target._ws_buf.shape == (1 << 14, LAT_DIM)
    assert ctx.ws_buf is None
    assert ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 20000)]
    assert sorted(ctx.sched) == [3616, 8192]


@pytest.mark.parametrize(
    "config, kv_lens, new, path",
    [
        # A full step of the 32768 config: n_kv 32768 fits its 65536 rows twice, exactly.
        pytest.param("32k", [32768], 32768, "kept", id="32k-full_step"),
        pytest.param("32k", [7, 33], 40, "kept", id="32k-two_prompts"),
        # The 1M config's first chunk, its 16th (n_kv 2**19), its 17th.
        pytest.param("1m-chunked", [32768], 32768, "kept", id="1m-first_chunk"),
        pytest.param("1m-chunked", [1 << 19], 32768, "workspace", id="1m-16th_chunk"),
        pytest.param("1m-chunked", [(1 << 19) + 32768], 32768, "decode", id="1m-17th_chunk"),
        # The 32768 config with block reuse: a full step over a cached prefix.
        pytest.param("32k-block_reuse", [32768] * 4, 32768, "kept", id="32k_reuse-4x32k"),
    ],
)
def test_serving_configs_steps(config, kv_lens, new, path):
    """Steps of the serving configs on a target holding the rows `_ctx_workspace_rows` gives the
    config: the 32768 config's 65536, the 1M config's and the 32768 config with block reuse's
    2**19. The path each step takes, and whether layer 35's gather is kept beside its rows."""
    sizes, (_, rows, _) = CONFIGS[config]
    assert M._ctx_workspace_rows(_sizing(*sizes)) == rows
    target = _attn_target(rows)
    ctx = target._ctx_workspace(_ctx_step(kv_lens, new), new, torch.device("cpu"))
    n_kv = sum(kv_lens)
    assert target._ws_buf.shape == (rows, LAT_DIM)
    if path == "decode":
        assert n_kv > rows and ctx.ws_buf is None
        assert ctx.sub_batches == [(lo, lo + 8192) for lo in range(0, new, 8192)]
        return
    assert n_kv <= rows and ctx.ws_buf.shape == (n_kv, LAT_DIM)
    assert ctx.ws_buf.data_ptr() == target._ws_buf.data_ptr()
    assert (ctx.kept is not None) == (path == "kept") == (2 * n_kv <= rows)


def _recorded_attention(monkeypatch) -> SimpleNamespace:
    """Stand-ins for the context attention's kernels: each dequantizing gather's source row and
    output buffer, each sparse prefill's workspace, each slot conversion's source row and each
    sparse decode call's rows and scheduler."""
    seen = SimpleNamespace(gathers=[], prefills=[], converts=[], decodes=[])

    def gather(pool, row, kv_req, kv_pos, block_table, out):
        seen.gathers.append((row, out))
        return out

    def convert(req_idx, block_table, sel, tokens_per_block, topk, stride, row):
        seen.converts.append(row)
        return torch.zeros(sel.shape, dtype=torch.int32)

    def decode(q, kv, slots, sched, scale, out):
        seen.decodes.append((q.shape[0], sched))

    monkeypatch.setattr(M, "ds_mla_dequant_gather", gather)
    monkeypatch.setattr(
        M, "flash_mla_sparse_prefill", lambda q, ws, idx, scale, out: seen.prefills.append(ws)
    )
    monkeypatch.setattr(M, "convert_req_index_to_global", convert)
    monkeypatch.setattr(M, "flash_mla_sparse_decode", decode)
    return seen


def _attend(target, step, ctx, tc: int) -> None:
    """Layers 35, 39 (attending over 35's latent), 40 and 43 (over 35's) on the step's context
    rows, as the forward runs them."""
    req_idx = torch.zeros(tc, dtype=torch.long)
    sel = torch.zeros(tc, 4, dtype=torch.int32)
    for layer in (SHARED_SOURCE, 39, 40, 43):
        L = SimpleNamespace(kv_source=KV_SOURCE[layer])
        target._attn_ctx(step, L, torch.empty(tc, 1), torch.empty(tc, 1), req_idx, sel, ctx)


def test_shared_source_kept_when_it_fits_twice_else_gathered_again(monkeypatch):
    """Two steps on one target holding 64 rows, each running layers 35, 39, 40 and 43. Context KV
    16 fits twice: layer 35's rows are gathered once, into the kept rows [16, 32), and remembered
    for the step; 39 and 43 read them without a gather of their own, and 40 gathers its own
    source into rows [0, 16). Context KV 40 fits once: nothing is kept, every layer gathers its
    source's rows into rows [0, 40), 35's again for 39 and for 43. Both steps use the one held
    buffer."""
    seen = _recorded_attention(monkeypatch)
    target = _attn_target(64)
    step = _ctx_step([10, 6], 9)
    ctx = target._ctx_workspace(step, 9, torch.device("cpu"))
    buf = target._ws_buf
    _attend(target, step, ctx, 9)
    assert [row for row, _ in seen.gathers] == [SHARED_SOURCE, 40]
    assert seen.gathers[0][1] is ctx.kept and seen.gathers[1][1] is ctx.ws_buf
    assert ctx.kept.data_ptr() == _row_ptr(buf, 16)
    assert ctx.ws == {SHARED_SOURCE: ctx.kept}
    assert [ws is ctx.kept for ws in seen.prefills] == [True, True, False, True]
    assert seen.prefills[2] is ctx.ws_buf

    seen.gathers.clear()
    seen.prefills.clear()
    step = _ctx_step([30, 10], 9)
    ctx = target._ctx_workspace(step, 9, torch.device("cpu"))
    assert target._ws_buf is buf and ctx.kept is None
    _attend(target, step, ctx, 9)
    assert [row for row, _ in seen.gathers] == [SHARED_SOURCE, SHARED_SOURCE, 40, SHARED_SOURCE]
    assert all(out is ctx.ws_buf for _, out in seen.gathers)
    assert ctx.ws_buf.data_ptr() == buf.data_ptr()
    assert all(ws is ctx.ws_buf for ws in seen.prefills) and ctx.ws == {}
    assert not seen.converts and not seen.decodes


def test_decode_path_attends_off_the_pool(monkeypatch):
    """A step past the held rows (9000 new rows over 9000 keys, held 64): no gather and no
    prefill; each layer converts its selections to pool slots of its source's owner row, then
    runs the sparse decode over its sub-batches of 8192 and 808 rows, each on that size's
    scheduler."""
    seen = _recorded_attention(monkeypatch)
    target = _attn_target(64)
    step = _ctx_step([9000], 9000)
    ctx = target._ctx_workspace(step, 9000, torch.device("cpu"))
    _attend(target, step, ctx, 9000)
    assert not seen.gathers and not seen.prefills
    assert seen.converts == [SHARED_SOURCE, SHARED_SOURCE, 40, SHARED_SOURCE]
    expect = [(8192, ctx.sched[8192]), (808, ctx.sched[808])] * 4
    assert [rows for rows, _ in seen.decodes] == [rows for rows, _ in expect]
    assert all(sched is want for (_, sched), (_, want) in zip(seen.decodes, expect))
