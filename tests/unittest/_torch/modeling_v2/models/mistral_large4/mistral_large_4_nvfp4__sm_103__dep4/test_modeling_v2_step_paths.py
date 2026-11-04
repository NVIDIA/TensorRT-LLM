# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Mistral-Large-4-1T-A52B NVFP4 dep4 target chooses per step, and the sizes it reads off
the engine to choose by.

No path is selected by the serving configuration. The core fixes its sizes on the first step the
engine hands over (`_derive_first_step`, again for a replaced KV cache manager) from the attention
metadata's max_num_tokens, max_num_requests, max_seq_len and runtime_features (chunked prefill,
block reuse: the executor builds them once from enable_chunked_prefill and the KvCacheConfig's
enable_block_reuse); every step then takes its paths from its own sizes:

* the held context workspace, bf16 `[core.ctx_workspace_rows, 576]`, allocated on the first
  context step whichever path it takes and held: `_ctx_workspace_rows` is twice the most
  context KV one step can attend (`_most_ctx_kv`: max_num_tokens with neither chunked prefill
  nor block reuse, else mainline's min(max_num_requests, max_num_tokens) x max_seq_len), up to
  `_CTX_WORKSPACE_ROWS` (2**19): 65536 rows at the 32K serving config (max_seq_len =
  max_num_tokens = 32768, both off), 2**19 with block reuse at max_batch_size 16 and at the 1M
  config (max_seq_len 1048576, chunked prefill);
* the context path, by the step's context KV n_kv (every context request's cached and new rows):
  the sparse prefill over the held workspace's first n_kv rows while n_kv fits the held rows,
  past them the sparse fp8 decode over the pool, every context row a one-row sequence, in
  sub-batches of at most `_CTX_DECODE_ROWS` (8192) with one scheduler per sub-batch size;
* layer 35's latent, which the NoPE layers 39, 43, ..., 71 attend over (`shared_source`, layer 35
  only, at every max_seq_len): when 2 n_kv fits the held rows it is gathered once into the n_kv
  rows after the step's own (`kept`) and 39 and 43 read it there without a gather; otherwise
  every layer gathers its source into the step's rows, 35's again for 39 and for 43;
* the indexer's logits budget, `_prefill_logits_budget(max_num_tokens, max_seq_len)`: mainline's
  reachable MQA-logits tile (TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET lowers it), within
  `_PREFILL_LOGITS_CEILING` (2**26 at dep4) and max_seq_len x max_seq_len / 64 -- 2**24 at
  max_seq_len 32768 (512-row tiles at 32768 keys), the ceiling at 131072 and at 1048576 -- with
  each tile's logits freed before the next tile's are allocated;
* the indexer's scoring blocks: an engine chunk of more than max_seq_len keys is cut at its
  requests over the host prefix sums `step_args` carries on every context step; a chunk within
  max_seq_len is one block, several requests or not; a block with no key fills its rows' slots
  with -1;
* the expert path's calls, keyed on chunked prefill alone: `_EXPERT_CALL_TOKENS[world]` (131072
  at four ranks, 32768 at eight) without it, block reuse or not, and
  `_EXPERT_CALL_TOKENS_CHUNKED[world]` (32768 at both) with it, at any max_seq_len; the MoE
  autotuner's largest bucket is the call, capped at every rank's max_num_tokens gathered;
* the rope tables: `rows`, `indexer` and `identity`, and no MLA table.

No device: the core is built on the meta device from the checkpoint's native config, params.json
(skipped where it is absent under LLM_MODELS_ROOT); everything else runs on host values, with
recording stand-ins for the kernels the paths call.
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest
import torch
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_workspace_bytes,
)

pytestmark = pytest.mark.cpu_only

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
#: The checkpoint's layers, the owner its NoPE layers 39, 43, ..., 71 attend over, each layer's
#: latent source, the 63 owners, and the latent row's width.
LAYERS = 72
SHARED_SOURCE = 35
KV_SOURCE = [SHARED_SOURCE if i > SHARED_SOURCE and i % 4 == 3 else i for i in range(LAYERS)]
OWNERS = [i for i in range(LAYERS) if KV_SOURCE[i] == i]
LAT_DIM = 576
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
_BUDGET_ENV = "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET"

#: Engines by name: (max_num_tokens, max_num_requests, max_seq_len, chunked prefill, block reuse).
#: The 32K serving config, it with block reuse, with chunked prefill and block reuse, and with
#: chunked prefill at one request per step, the Pareto config, the 1M config, and max_seq_len
#: 131072 with chunked prefill and with neither. Both flags are the attention metadata's runtime
#: features, as the executor builds them from enable_chunked_prefill and the KvCacheConfig's
#: enable_block_reuse.
ENGINES = {
    "32k": (32768, 16, 32768, False, False),
    "32k_block_reuse": (32768, 16, 32768, False, True),
    "32k_chunked_block_reuse": (32768, 16, 32768, True, True),
    "32k_chunked_one_request": (32768, 1, 32768, True, False),
    "pareto": (16384, 16, 32768, False, False),
    "1m": (32768, 16, 1048576, True, False),
    "131072_chunked": (32768, 16, 131072, True, False),
    "131072": (32768, 16, 131072, False, False),
}
#: What each engine sizes: (the most context KV one step attends, the held workspace's rows, the
#: indexer's logits budget). The expert path's calls follow the engine's chunked prefill flag
#: (`CALLS`), not the context KV: block reuse without it keeps `_EXPERT_CALL_TOKENS`, one request
#: per step under it, which attends at most max_seq_len = max_num_tokens, takes
#: `_EXPERT_CALL_TOKENS_CHUNKED`.
SIZES = {
    "32k": (32768, 65536, 1 << 24),
    "32k_block_reuse": (1 << 19, 1 << 19, 1 << 24),
    "32k_chunked_block_reuse": (1 << 19, 1 << 19, 1 << 24),
    "32k_chunked_one_request": (32768, 65536, 1 << 24),
    "pareto": (16384, 32768, 1 << 24),
    "1m": (1 << 24, 1 << 19, 1 << 26),
    "131072_chunked": (1 << 21, 1 << 19, 1 << 26),
    "131072": (32768, 65536, 1 << 26),
}
#: The expert path's call by (chunked prefill, world).
CALLS = {(False, 4): 131072, (False, 8): 32768, (True, 4): 32768, (True, 8): 32768}


@pytest.fixture(autouse=True)
def _default_logits_budget(monkeypatch):
    """Mainline's logits budget at its default: a TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET set in the
    environment would lower the budgets pinned here."""
    monkeypatch.delenv(_BUDGET_ENV, raising=False)


def _md(name: str, **extra) -> SimpleNamespace:
    """The attention metadata's engine quantities for engine `name`, as `_most_ctx_kv` and
    `_derive_first_step` read them, plus `extra`."""
    max_num_tokens, max_num_requests, max_seq_len, chunked, reuse = ENGINES[name]
    return SimpleNamespace(
        max_num_tokens=max_num_tokens,
        max_num_requests=max_num_requests,
        max_seq_len=max_seq_len,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=chunked, cache_reuse=reuse, chunk_size=max_num_tokens
        ),
        **extra,
    )


def _manager() -> SimpleNamespace:
    """A KV cache manager with what `_derive_first_step` reads of it: the fp8_ds_mla layout, the
    checkpoint's kv_source_layers and its 63 owners' pool rows. A new one each call: the core
    derives again for a replaced manager."""
    return SimpleNamespace(
        latent_layout=M._LATENT_LAYOUT,
        kv_source_layers=KV_SOURCE,
        latent_pool_row=lambda i: OWNERS.index(KV_SOURCE[i]),
        kv_owner_layers=OWNERS,
    )


def _first_step_md(name: str) -> SimpleNamespace:
    """A first step's metadata on engine `name`: its quantities, a fresh KV cache manager and the
    pool's 32-token pages."""
    return _md(name, kv_cache_manager=_manager(), tokens_per_block=32)


def _stub_builds(monkeypatch) -> list:
    """Stand-ins for the kernel builds and the rope tables, and a device index: `_rope_tables`
    takes exactly `(cfg, alpha, positions, dtype, device)` and records each table's rows."""
    built = []

    def tables(cfg, alpha, positions, dtype, device):
        built.append(positions)
        return SimpleNamespace(rows=positions)

    monkeypatch.setattr(M, "_rope_tables", tables)
    monkeypatch.setattr(M, "build_ix_q_side", lambda dev, is_neox: None)
    monkeypatch.setattr(M, "build_ix_k_cache", lambda dev, is_neox, cache_contiguous: None)
    monkeypatch.setattr(
        M, "build_ix_topk_decode", lambda dev, next_n: SimpleNamespace(next_n=next_n)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return built


def _core(max_seq_len: int | None = None, world: int = 4):
    """The core of rank 0 at `world` attention-DP ranks, on the meta device, built from the
    checkpoint's params.json, as the native format reads it, with the engine's max_seq_len."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
    from tensorrt_llm.mapping import Mapping

    root = llm_models_root()
    paths = [root / name for name in CHECKPOINTS] if root is not None else []
    path = next((p for p in paths if (p / "params.json").is_file()), None)
    if path is None:
        pytest.skip(f"none of {CHECKPOINTS} is under LLM_MODELS_ROOT ({root})")
    mapping = Mapping(
        world_size=world,
        rank=0,
        gpus_per_node=4,
        tp_size=world,
        pp_size=1,
        moe_ep_size=world,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    model_config = ModelConfig(
        pretrained_config=MistralConfigLoader().load_pretrained_config(str(path)),
        mapping=mapping,
        max_num_tokens=32768,
        max_seq_len=max_seq_len,
    )
    with torch.device("meta"):
        return M.MistralLarge4Dep4ModelingV2(model_config)


def _check_sizes(core, name: str, world: int) -> None:
    """The core's four engine sizes after `_derive_first_step` on engine `name`."""
    max_num_tokens, chunked = ENGINES[name][0], ENGINES[name][3]
    _, rows, budget = SIZES[name]
    call = CALLS[(chunked, world)]
    assert core.prefill_logits_budget == budget
    assert core.ctx_workspace_rows == rows
    assert core.expert_call_tokens == call
    assert core.expert_tune_tokens == min(call, world * max_num_tokens)


# ---- the engine's sizes ----


@pytest.mark.parametrize("name", list(ENGINES))
def test_workspace_rows_from_the_runtime_features(name):
    """`_most_ctx_kv` and `_ctx_workspace_rows` on each engine's metadata: with neither chunked
    prefill nor block reuse a step attends at most max_num_tokens of context KV (32768 at the 32K
    config and at 131072 without either, 16384 at the Pareto config), so the held workspace is
    twice that; with either, mainline's never-stall worst case min(max_num_requests,
    max_num_tokens) x max_seq_len (2**19 at 32K with block reuse at max_batch_size 16, chunked
    prefill or not, 2**24 at the 1M config, 2**21 at 131072 chunked, 32768 at one request per
    step), the workspace capped at 2**19 rows."""
    assert M._CTX_WORKSPACE_ROWS == 1 << 19
    most, rows, _ = SIZES[name]
    md = _md(name)
    assert M._most_ctx_kv(md) == most
    assert M._ctx_workspace_rows(md) == rows == min(2 * most, M._CTX_WORKSPACE_ROWS)


def test_block_reuse_is_read_from_the_runtime_features():
    """Block reuse is the attention metadata's runtime feature, `cache_reuse` (the executor builds
    it from the KvCacheConfig's enable_block_reuse): at the 32K config, cache_reuse on gives
    mainline's never-stall worst case, min(16, 32768) x 32768 = 2**19 rows of context KV and a
    held workspace of 2**19 rows, off max_num_tokens and 65536 rows; with chunked prefill the
    worst case either way. Nothing else is read: the metadata carries no KV cache manager."""
    for chunked, reuse, most in (
        (False, False, 32768),
        (False, True, 1 << 19),
        (True, False, 1 << 19),
        (True, True, 1 << 19),
    ):
        md = _md("32k")
        md.runtime_features = AttentionRuntimeFeatures(
            chunked_prefill=chunked, cache_reuse=reuse, chunk_size=32768
        )
        assert not hasattr(md, "kv_cache_manager")
        assert M._most_ctx_kv(md) == most, (chunked, reuse)
        assert M._ctx_workspace_rows(md) == min(2 * most, M._CTX_WORKSPACE_ROWS), (chunked, reuse)


@pytest.mark.parametrize(
    "max_num_tokens, max_seq_len, budget, tile",
    [
        (32768, 32768, 1 << 24, 512),
        (16384, 32768, 1 << 24, 512),
        (32768, 65536, 1 << 26, 1024),
        (32768, 131072, 1 << 26, 512),
        (32768, 1048576, 1 << 26, 64),
        (64, 32768, 1 << 21, 64),
    ],
    ids=["32k", "pareto", "65536", "131072", "1m", "64_tokens"],
)
def test_prefill_logits_budget(max_num_tokens, max_seq_len, budget, tile):
    """The budget is the least of mainline's reachable tile (its 2**31 default within max_num_tokens
    x max_seq_len), the ceiling 2**26 and max_seq_len x max_seq_len / 64: 2**24 at max_seq_len
    32768, so 512-row tiles at 32768 keys; the ceiling from 65536 on, 512-row tiles at 131072 keys
    and 64 at 1048576; the reachable tile where max_num_tokens binds it (64 rows at 64 tokens)."""
    assert M._PREFILL_LOGITS_CEILING == 1 << 26
    reachable = get_indexer_mqa_logits_workspace_bytes(max_num_tokens, max_seq_len) // 4
    got = M._prefill_logits_budget(max_num_tokens, max_seq_len)
    assert got == budget == min(reachable, 1 << 26, max_seq_len * max_seq_len // 64)
    assert got // max_seq_len == tile


def test_logits_budget_env_override_only_lowers(monkeypatch):
    """TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET lowers mainline's reachable tile and so the budget at
    every engine; set past the ceiling it raises nothing."""
    monkeypatch.setenv(_BUDGET_ENV, str(1 << 20))
    assert M._prefill_logits_budget(32768, 32768) == 1 << 20
    assert M._prefill_logits_budget(32768, 1048576) == 1 << 20
    monkeypatch.setenv(_BUDGET_ENV, str(1 << 40))
    assert M._prefill_logits_budget(32768, 32768) == 1 << 24
    assert M._prefill_logits_budget(32768, 1048576) == M._PREFILL_LOGITS_CEILING


@pytest.mark.parametrize("world", [4, 8], ids=["dep4", "dep8"])
@pytest.mark.parametrize("max_seq_len", [8192, 32768, 32769, 1048576, None])
def test_core_holds_no_engine_size_until_the_first_step(max_seq_len, world):
    """Built at any engine max_seq_len (None: the engine infers one), the core chooses nothing by
    it: its four engine sizes are 0 until `_derive_first_step`, and `shared_source` is
    structural, layer 35 alone, the owner the NoPE layers attend over."""
    core = _core(max_seq_len, world)
    assert not hasattr(core, "long_context")
    # The fused attention glue at every max_seq_len, unless `_UNFUSED_GLUE`.
    assert core.unfused_glue is M._UNFUSED_GLUE
    assert core.kv_source == KV_SOURCE
    assert [i for i, keep in enumerate(core.shared_source) if keep] == [SHARED_SOURCE]
    sizes = (core.prefill_logits_budget, core.ctx_workspace_rows)
    assert sizes + (core.expert_call_tokens, core.expert_tune_tokens) == (0, 0, 0, 0)
    assert M._EXPERT_CALL_TOKENS == {4: 131072, 8: 32768}
    assert M._EXPERT_CALL_TOKENS_CHUNKED == {4: 32768, 8: 32768}
    assert not hasattr(M, "_EXPERT_CALL_TOKENS_CACHED")


@pytest.mark.parametrize("world", [4, 8], ids=["dep4", "dep8"])
@pytest.mark.parametrize("name", list(ENGINES))
def test_first_step_sizes_the_core(monkeypatch, name, world):
    """`_derive_first_step` on the real core (meta device) sets the logits budget, the held
    workspace's rows and the expert path's call and autotuner bucket from the first step's
    metadata: at four ranks one call of 131072 without chunked prefill, block reuse or not (the
    autotuner's bucket every rank's max_num_tokens gathered: 65536 at the Pareto config), 32768
    with it; at eight ranks 32768 either way."""
    _stub_builds(monkeypatch)
    core = _core(world=world)
    core.build_kernels(0)
    core._derive_first_step(_first_step_md(name))
    _check_sizes(core, name, world)
    assert core._stride_factor == len(OWNERS) * 32 and core._max_ctx_len == ENGINES[name][2]


@pytest.mark.parametrize("world", [4, 8], ids=["dep4", "dep8"])
def test_expert_calls_keyed_on_chunked_prefill_alone(monkeypatch, world):
    """The expert path's call follows the engine's chunked prefill flag alone: without it
    `_EXPERT_CALL_TOKENS[world]` (131072 at four ranks, 32768 at eight), block reuse or not -- a
    32K step under block reuse, which may attend 2**19 rows of context KV (a held workspace of
    2**19 rows), still makes one call of every rank's 32768 rows at four -- and with it
    `_EXPERT_CALL_TOKENS_CHUNKED[world]` (32768 at both), at max_seq_len 32768, 131072 and
    1048576, block reuse or not, and at one request per step, whose context KV is
    max_num_tokens. The autotuner's bucket is the call within every rank's max_num_tokens
    gathered (65536 at the Pareto config's 16384 at four ranks). On the real core (meta device),
    each engine with a KV cache manager of its own."""
    _stub_builds(monkeypatch)
    core = _core(world=world)
    core.build_kernels(0)
    unchunked, chunked = M._EXPERT_CALL_TOKENS[world], M._EXPERT_CALL_TOKENS_CHUNKED[world]
    assert (unchunked, chunked) == ({4: 131072, 8: 32768}[world], 32768)
    # (engine, its runtime features' chunked prefill and block reuse)
    for name, chunked_prefill, block_reuse in (
        ("32k", False, False),
        ("32k_block_reuse", False, True),
        ("pareto", False, False),
        ("131072", False, False),
        ("32k_chunked_one_request", True, False),
        ("32k_chunked_block_reuse", True, True),
        ("131072_chunked", True, False),
        ("1m", True, False),
    ):
        md = _first_step_md(name)
        assert md.runtime_features.chunked_prefill is chunked_prefill
        assert md.runtime_features.cache_reuse is block_reuse
        call = chunked if chunked_prefill else unchunked
        core._derive_first_step(md)
        assert core.expert_call_tokens == call, name
        assert core.expert_tune_tokens == min(call, world * md.max_num_tokens), name
        assert core.ctx_workspace_rows == SIZES[name][1], name


def test_a_replaced_kv_cache_manager_sizes_the_core_again(monkeypatch):
    """The sizes are fixed once per KV cache manager: a later step on the same manager changes
    none of them, a replaced manager derives them again from its step's metadata."""
    _stub_builds(monkeypatch)
    core = _core()
    core.build_kernels(0)
    first = _first_step_md("32k")
    core._derive_first_step(first)
    _check_sizes(core, "32k", 4)
    core._derive_first_step(_md("1m", kv_cache_manager=first.kv_cache_manager, tokens_per_block=32))
    _check_sizes(core, "32k", 4)
    core._derive_first_step(_first_step_md("1m"))
    _check_sizes(core, "1m", 4)


def test_another_world_is_refused():
    """The expert call tables hold four and eight ranks only: two ranks are refused before
    anything is built, whatever the engine's max_seq_len."""
    for max_seq_len in (32768, 1048576):
        with pytest.raises(ValueError, match="serves 4 ranks on one node or 8"):
            _core(max_seq_len, 2)


# ---- the rope tables ----


def test_rope_tables_hold_no_mla_table():
    """`_rope_tables` builds `rows`, the indexer's `[cos | sin]` (which `ds_mla_rope_append`
    reads too) and the NoPE identity table, and takes no `with_mla`."""
    alpha = torch.linspace(-0.5, 0.5, 32)
    tables = M._rope_tables(ROPE_CFG, alpha, 300, torch.bfloat16, "cpu")
    assert set(vars(tables)) == {"rows", "indexer", "identity"}
    assert tables.rows == 300 and tables.indexer.shape == tables.identity.shape == (300, 64)
    with pytest.raises(TypeError):
        M._rope_tables(ROPE_CFG, alpha, 300, torch.bfloat16, "cpu", with_mla=True)


def test_build_kernels_and_first_step_build_the_rope_tables(monkeypatch):
    """`build_kernels` builds the tables at the core's first size and `_derive_first_step` grows
    them to the engine's max_seq_len, both through `(cfg, alpha, positions, dtype, device)`
    alone; a replaced manager at a shorter max_seq_len keeps them. On a core holding just what
    the two read, sized like any core."""
    built = _stub_builds(monkeypatch)
    core = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=ROPE_CFG),
        w={"lerope_alpha": None, "final_norm": torch.empty(0, dtype=torch.bfloat16)},
        rope_rows=16,
        kv_source=KV_SOURCE,
        num_layers=LAYERS,
        dp_group=[0, 1, 2, 3],
        _derived_for=None,
        _ix_k=None,
        unfused_glue=False,
    )
    M.MistralLarge4Dep4ModelingV2.build_kernels(core, 0)
    md = _first_step_md("1m")
    M.MistralLarge4Dep4ModelingV2._derive_first_step(core, md)
    assert built == [16, 1048576] and core._rope.rows == core.rope_rows == 1048576
    _check_sizes(core, "1m", 4)
    M.MistralLarge4Dep4ModelingV2._derive_first_step(core, _first_step_md("32k"))
    assert built == [16, 1048576] and core._rope.rows == 1048576
    _check_sizes(core, "32k", 4)


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
        dp_group=[0, 1, 2, 3],
        _derived_for=None,
        _ix_k=None,
        _ix_topk=None,
        _rope=SimpleNamespace(rows=16),
    )
    derive = M.MistralLarge4Dep4ModelingV2._derive_first_step
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


def test_ctx_offsets_device_and_host():
    """`_ctx_kv_offsets` hands over the device offsets the context workspace is laid out by;
    `_ctx_host_offsets` the host prefix sums the indexer's blocks are cut by: the engine's host
    indptrs under context MLA with cached KV, the prefix sum of `seq_lens` without it."""
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


def _recorded_kernels(monkeypatch) -> list:
    """Stand-ins for the context selection's kernels, recording each key gather's
    `(k_token_start, n_k)`, each scored tile's rows and keys and whether the previous tile's
    logits are still alive when it is scored, and each top-k call's windows. None writes the
    selection. Each tile's split-KV schedule is a marker of the windows and keys it was built
    for, which its logits call must be handed (checked here), and the top-k's scratch a
    marker the target must hand every top-k call."""
    calls = []
    previous = []

    def gather(pool, fp8, scale, k_token_start, n_k, dim):
        calls.append(("gather", k_token_start, n_k))
        return torch.empty(n_k, dim), torch.empty(n_k, 1)

    def schedule(ks, ke, n_k, heads):
        return ("schedule", ks.tolist(), ke.tolist(), n_k, heads)

    def logits(q, kv, weights, ks, ke, schedule_meta=None):
        assert schedule_meta == ("schedule", ks.tolist(), ke.tolist(), kv[0].shape[0], q.shape[1])
        alive = previous[-1]() is not None if previous else None
        calls.append(("logits", q.shape[0], kv[0].shape[0], alive))
        out = torch.empty(1)
        previous.append(weakref.ref(out))
        return out

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


def _prefill_target(**core) -> M.PrefillTarget:
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(**core)
    return target


def _select_step(chunks) -> dict:
    """What `_select_ctx` reads of a step over `_packed`'s two context requests (no decode
    rows), the metadata's host prefix sums included, as `step_args` always carries them."""
    chunk, md = _packed()
    return dict(
        indexer_prefill_chunks=chunks,
        num_ctx_kv_tokens=84,
        cu_seqlen_ks=chunk.cu_seqlen_ks,
        cu_seqlen_ke=chunk.cu_seqlen_ke,
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
        **M._ctx_host_offsets(md, 2),
    )


def _select_target(max_seq_len: int, budget: int = 1 << 24) -> M.PrefillTarget:
    return _prefill_target(
        idx_dim=128, idx_topk=4, prefill_logits_budget=budget, _max_ctx_len=max_seq_len
    )


@pytest.mark.parametrize("max_seq_len", [64, 84, 128], ids=["past", "at", "within"])
def test_select_ctx_cuts_a_chunk_only_past_max_seq_len(monkeypatch, max_seq_len):
    """The engine's chunk of two requests over 84 keys: past a max_seq_len of 64 it is cut at its
    requests, each request's keys gathered alone and its windows rebased to its first key; at
    max_seq_len 84 or 128 it is scored whole, one gather over its 84 keys and the chunk's own
    windows."""
    calls = _recorded_kernels(monkeypatch)
    chunk, _ = _packed()
    out = torch.empty(14, 4, dtype=torch.int32)
    _select_target(max_seq_len)._select_ctx(
        _select_step([chunk]), None, torch.empty(14, 1), torch.empty(14, 1), out, 14
    )
    gathers = [c[1:] for c in calls if c[0] == "gather"]
    windows = [c[1:] for c in calls if c[0] == "topk"]
    if max_seq_len < 84:
        assert gathers == [(0, 48), (48, 36)]
        assert windows == [([0] * 8, list(range(41, 49))), ([0] * 6, list(range(31, 37)))]
        return
    assert gathers == [(0, 84)]
    assert windows == [(chunk.cu_seqlen_ks.tolist(), chunk.cu_seqlen_ke.tolist())]


def test_select_ctx_fills_a_block_with_no_key(monkeypatch):
    """A chunk whose rows have no key leaves every slot of those rows -1, gathering and scoring
    nothing for them; the next chunk's rows are selected as usual."""
    calls = _recorded_kernels(monkeypatch)
    empty = SimpleNamespace(token_start=0, token_end=3, k_token_start=0, k_token_end=0)
    keyed = SimpleNamespace(
        token_start=3,
        token_end=6,
        k_token_start=0,
        k_token_end=10,
        cu_seqlen_ks=torch.zeros(3, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([8, 9, 10], dtype=torch.int32),
    )
    step = dict(
        indexer_prefill_chunks=[empty, keyed],
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
        host_cu_ctx_new=[0, 6],
        host_cu_ctx_kv=[0, 10],
    )
    out = torch.full((6, 4), 7, dtype=torch.int32)
    _select_target(64)._select_ctx(step, None, torch.empty(6, 1), torch.empty(6, 1), out, 6)
    assert out[:3].eq(-1).all() and out[3:].eq(7).all()
    assert [c[:3] for c in calls if c[0] != "topk"] == [("gather", 0, 10), ("logits", 3, 10)]


@pytest.mark.parametrize("engine", ["32k", "1m", "env_2**20"])
def test_select_ctx_tiles_under_the_engines_budget(monkeypatch, engine):
    """2049 context rows over 32768 keys, one unplanned chunk, in tiles of the budget over the
    keys: 512 rows at the 32K config's 2**24, 2048 at the 1M config's 2**26, 32 under
    TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET 2**20; each tile's logits freed before the next tile is
    scored."""
    if engine == "env_2**20":
        monkeypatch.setenv(_BUDGET_ENV, str(1 << 20))
    max_seq_len = {"32k": 32768, "1m": 1048576, "env_2**20": 32768}[engine]
    budget = M._prefill_logits_budget(32768, max_seq_len)
    tile = {"32k": 512, "1m": 2048, "env_2**20": 32}[engine]
    calls = _recorded_kernels(monkeypatch)
    rows, keys = 2049, 32768
    step = dict(
        indexer_prefill_chunks=None,
        num_ctx_kv_tokens=keys,
        cu_seqlen_ks=torch.zeros(rows, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(keys - rows + 1, keys + 1, dtype=torch.int32),
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
        host_cu_ctx_new=[0, rows],
        host_cu_ctx_kv=[0, keys],
    )
    target = _select_target(max_seq_len, budget)
    target._select_ctx(
        step, None, torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(rows, 1), rows
    )
    tiles = [c[1:] for c in calls if c[0] == "logits"]
    assert budget // keys == tile and calls[0] == ("gather", 0, keys)
    full, last = divmod(rows, tile)
    expected = [(tile, keys, False)] * full + [(last, keys, False)] * bool(last)
    assert tiles == [(tile, keys, None)] + expected[1:]


def _step_md(cached_kv: bool) -> SimpleNamespace:
    """A step's metadata of two context requests, as `PrefillTarget.step_args` reads it: under
    context MLA with cached KV `_packed`'s two (40 + 8 and 30 + 6 tokens, 84 keys), the engine's
    host indptrs beside its device ones; without it two fresh prompts of 8 and 6 tokens, whose
    host prefix sums are those of `seq_lens`."""
    i64 = dict(dtype=torch.int64)
    if cached_kv:
        chunk, host = _packed()
        phase = dict(
            num_ctx_kv_tokens=14,
            enable_context_mla_with_cached_kv=True,
            ctx_cached_token_indptr=torch.tensor([0, 40, 70], **i64),
            ctx_kv_indptr=torch.tensor([0, 48, 84], **i64),
            max_ctx_seq_len=8,
            num_ctx_cached_tokens=70,
            host_ctx_uncached_token_indptr=host.host_ctx_uncached_token_indptr,
            host_ctx_kv_indptr=host.host_ctx_kv_indptr,
        )
    else:
        chunk = SimpleNamespace(
            token_start=0,
            token_end=14,
            k_token_start=0,
            k_token_end=14,
            cu_seqlen_ks=torch.tensor([0] * 8 + [8] * 6, dtype=torch.int32),
            cu_seqlen_ke=torch.tensor([*range(1, 9), *range(9, 15)], dtype=torch.int32),
        )
        phase = dict(
            num_ctx_kv_tokens=14,
            enable_context_mla_with_cached_kv=False,
            mla_prepare_ctx_cu_seqlens=lambda: torch.tensor([0, 8, 14], dtype=torch.int32),
        )
    return SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            get_latent_pool=lambda: torch.zeros(2, 3, 32, 656, dtype=torch.uint8),
            get_indexer_k_cache_buffers=lambda i: None,
        ),
        num_seqs=2,
        num_tokens=14,
        all_rank_num_tokens=[14, 1, 1, 1],
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
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
        **phase,
    )


@pytest.mark.parametrize("cached_kv", [True, False], ids=["cached_kv", "fresh"])
def test_step_args_always_carries_the_host_prefix_sums(monkeypatch, cached_kv):
    """`step_args` hands over the host prefix sums on every context step, from the engine's host
    indptrs under context MLA with cached KV and from `seq_lens` without it, and the selection
    cuts the step's chunk at its requests by them once it holds more keys than max_seq_len:
    84 keys past 64, and 14 fresh keys past a max_seq_len of 10."""
    md = _step_md(cached_kv)
    target = _select_target(64 if cached_kv else 10)
    target.core.indexer_layers = [1]
    step = target.step_args(md)
    assert step["indexer_prefill_chunks"] is md.indexer_prefill_chunks
    if cached_kv:
        assert step["ctx_cached_tokens"] == 70 and int(step["cu_ctx_kv"][-1]) == 84
        assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 48, 84])
    else:
        assert step["ctx_cached_tokens"] == 0 and int(step["cu_ctx_kv"][-1]) == 14
        assert (step["host_cu_ctx_new"], step["host_cu_ctx_kv"]) == ([0, 8, 14], [0, 8, 14])
    calls = _recorded_kernels(monkeypatch)
    out = torch.empty(14, 4, dtype=torch.int32)
    target._select_ctx(step, None, torch.empty(14, 1), torch.empty(14, 1), out, 14)
    gathers = [c[1:] for c in calls if c[0] == "gather"]
    windows = [c[1:] for c in calls if c[0] == "topk"]
    assert gathers == ([(0, 48), (48, 36)] if cached_kv else [(0, 8), (8, 6)])
    if not cached_kv:
        assert windows == [([0] * 8, list(range(1, 9))), ([0] * 6, list(range(1, 7)))]


# ---- the context workspace and the context attention ----


def _attn_target(held: int) -> M.PrefillTarget:
    """A prefill target over what its context workspace and attention read off the core: a held
    workspace of `held` rows (`ctx_workspace_rows`, small here to reach each path cheaply), and
    `shared_source` as every core holds it (layer 35 alone,
    `test_core_holds_no_engine_size_until_the_first_step`)."""
    return _prefill_target(
        lat_dim=LAT_DIM,
        ctx_workspace_rows=held,
        shared_source=[i == SHARED_SOURCE for i in range(LAYERS)],
        softmax_scale=1.0,
        idx_topk=4,
        _pool_row=list(range(LAYERS)),
        _stride_factor=1,
    )


def _ctx_step(contexts) -> dict:
    """What `_ctx_workspace` and `_attn_ctx` read of a step of context requests of (cached, new)
    tokens each."""
    kv = [c + n for c, n in contexts]
    return dict(
        num_contexts=len(contexts),
        num_ctx_tokens=sum(n for _, n in contexts),
        ctx_cached_tokens=sum(c for c, _ in contexts),
        cu_ctx_kv=torch.tensor([0, *kv]).cumsum(0),
        latent_pool=None,
        latent_kv=None,
        block_table=None,
        tokens_per_block=32,
    )


def _rows_of(view: torch.Tensor, buf: torch.Tensor) -> tuple[int, int]:
    """The held buffer's rows `[lo, hi)` a view of it covers."""
    assert view.untyped_storage().data_ptr() == buf.untyped_storage().data_ptr()
    lo = (view.data_ptr() - buf.data_ptr()) // (LAT_DIM * buf.element_size())
    return lo, lo + view.shape[0]


@pytest.mark.parametrize(
    "contexts, path",
    [
        ([(7, 3), (0, 6)], "kept"),
        ([(14, 6)], "kept"),
        ([(15, 6)], "workspace"),
        ([(30, 10)], "workspace"),
        ([(31, 10)], "decode"),
    ],
    ids=["n_kv16", "n_kv20", "n_kv21", "n_kv40", "n_kv41"],
)
def test_context_path_by_n_kv(contexts, path):
    """Over a held workspace of 40 rows: a step's context KV of 16 or 20 rows fits it twice (the
    step's rows first, then as many kept), 21 or 40 once (no kept rows), 41 not at all (the
    decode path, every context row a one-row sequence, no workspace). The held workspace is the
    core's `ctx_workspace_rows` x 576 bf16, allocated on the step."""
    target = _attn_target(40)
    step = _ctx_step(contexts)
    tc = step["num_ctx_tokens"]
    n_kv = step["ctx_cached_tokens"] + tc
    ctx = target._ctx_workspace(step, tc, torch.device("cpu"))
    buf = target._ws_buf
    assert buf.shape == (40, LAT_DIM) and buf.dtype == torch.bfloat16
    if path == "decode":
        assert ctx.ws_buf is None and not hasattr(ctx, "kv_req")
        assert ctx.sub_batches == [(0, tc)] and sorted(ctx.sched) == [tc]
        return
    assert _rows_of(ctx.ws_buf, buf) == (0, n_kv) and ctx.ws == {}
    if path == "kept":
        assert _rows_of(ctx.kept, buf) == (n_kv, 2 * n_kv)
    else:
        assert ctx.kept is None
    kv = [c + n for c, n in contexts]
    assert ctx.kv_req.tolist() == [r for r, n in enumerate(kv) for _ in range(n)]
    assert ctx.kv_pos.tolist() == [p for n in kv for p in range(n)]
    assert ctx.ws_off.tolist() == [0, *torch.tensor(kv).cumsum(0).tolist()][:-1]


def test_decode_path_in_sub_batches(monkeypatch):
    """20000 context rows past the held rows: the selections become pool slots once, then the
    sparse decode runs in sub-batches of 8192, 8192 and 3616 rows on their sizes' schedulers, and
    nothing is gathered."""
    calls = []
    monkeypatch.setattr(
        M, "ds_mla_dequant_gather", lambda *args: pytest.fail("the decode path gathered")
    )
    monkeypatch.setattr(
        M,
        "convert_req_index_to_global",
        lambda req, table, sel, tpb, topk, stride, row: calls.append(("slots", row)) or sel,
    )
    monkeypatch.setattr(
        M,
        "flash_mla_sparse_decode",
        lambda q, kv, slots, sched, scale, out: calls.append(("decode", q.shape[0], sched)),
    )
    target = _attn_target(40)
    step = _ctx_step([(100, 20000)])
    ctx = target._ctx_workspace(step, 20000, torch.device("cpu"))
    assert ctx.ws_buf is None
    assert ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 20000)]
    assert sorted(ctx.sched) == [3616, 8192]
    L = SimpleNamespace(kv_source=KV_SOURCE[39])
    sel = torch.zeros(20000, 4, dtype=torch.int32)
    target._attn_ctx(
        step, L, torch.empty(20000, 1), torch.empty(20000, 1), torch.zeros(20000), sel, ctx
    )
    assert calls == [
        ("slots", SHARED_SOURCE),
        ("decode", 8192, ctx.sched[8192]),
        ("decode", 8192, ctx.sched[8192]),
        ("decode", 3616, ctx.sched[3616]),
    ]


@pytest.mark.parametrize("held, kept", [(40, True), (24, False)], ids=["fits_twice", "fits_once"])
def test_layer_35_kept_or_gathered_again_by_fit(monkeypatch, held, kept):
    """Layers 35, 39 (attending over 35's latent), 40 and 43 (over 35's) on one step of two
    context requests of 10 and 6 KV rows. Over 40 held rows the step fits twice: layer 35 is
    gathered once into the kept rows, which 39 and 43 read without a gather, and 40 into the
    step's own rows. Over 24 it fits once: every layer gathers its source's rows into the step's
    rows, 35's again for 39 and for 43, and nothing is kept."""
    gathers, prefills = [], []

    def gather(pool, row, kv_req, kv_pos, block_table, out):
        gathers.append((row, out))
        return out

    monkeypatch.setattr(M, "ds_mla_dequant_gather", gather)
    monkeypatch.setattr(
        M, "flash_mla_sparse_prefill", lambda q, ws, idx, scale, out: prefills.append(ws)
    )
    target = _attn_target(held)
    step = _ctx_step([(7, 3), (0, 6)])
    ctx = target._ctx_workspace(step, 9, torch.device("cpu"))
    buf = target._ws_buf
    assert _rows_of(ctx.ws_buf, buf) == (0, 16) and ctx.ws == {}
    req_idx = torch.tensor([0] * 3 + [1] * 6)
    sel = torch.zeros(9, 4, dtype=torch.int32)
    for layer in (SHARED_SOURCE, 39, 40, 43):
        L = SimpleNamespace(kv_source=KV_SOURCE[layer])
        target._attn_ctx(step, L, torch.empty(9, 1), torch.empty(9, 1), req_idx, sel, ctx)

    if not kept:
        assert ctx.kept is None
        assert [row for row, _ in gathers] == [35, 35, 40, 35]
        assert all(out is ctx.ws_buf for _, out in gathers)
        assert all(ws is ctx.ws_buf for ws in prefills) and ctx.ws == {}
        return
    assert _rows_of(ctx.kept, buf) == (16, 32)
    assert [row for row, _ in gathers] == [35, 40]
    assert gathers[0][1] is ctx.kept and gathers[1][1] is ctx.ws_buf
    assert ctx.ws == {SHARED_SOURCE: ctx.kept}
    assert [ws is ctx.kept for ws in prefills] == [True, True, False, True]
    assert prefills[2] is ctx.ws_buf


def test_workspace_is_allocated_once_and_held():
    """The held workspace is allocated on the first context step, here one on the decode path,
    and every later step's rows, kept rows included, are that buffer's: 16 KV rows (kept), 30
    (fits once), 41 (decode path) and 5 (kept) over 40 held rows."""
    target = _attn_target(40)
    assert target._ws_buf is None
    cpu = torch.device("cpu")
    ctx = target._ctx_workspace(_ctx_step([(35, 6)]), 6, cpu)
    buf = target._ws_buf
    assert ctx.ws_buf is None and buf.shape == (40, LAT_DIM)
    for contexts, rows, kept in (
        ([(10, 6)], (0, 16), (16, 32)),
        ([(20, 10)], (0, 30), None),
        ([(1, 40)], None, None),
        ([(0, 2), (0, 3)], (0, 5), (5, 10)),
    ):
        step = _ctx_step(contexts)
        ctx = target._ctx_workspace(step, step["num_ctx_tokens"], cpu)
        assert target._ws_buf is buf
        if rows is None:
            assert ctx.ws_buf is None
            continue
        assert _rows_of(ctx.ws_buf, buf) == rows
        assert (ctx.kept if kept is None else _rows_of(ctx.kept, buf)) == kept
