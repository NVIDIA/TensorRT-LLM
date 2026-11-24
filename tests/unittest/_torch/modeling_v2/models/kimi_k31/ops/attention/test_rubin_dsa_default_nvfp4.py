# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent-KV hooks of the default Rubin target's DSA metadata (CPU): with an NVFP4 cache the
generation gather's buffers exist from the metadata's creation at the engine's size, no step
allocates, and each step's gather plan holds what the engine DSA backend passes."""

import ast
import inspect
from pathlib import Path

import pytest
import torch

# CPU cases of a K3.1 Rubin helper. A GPU stage other than sm_107 collects this directory: skip
# there before any family import, so nothing of the family is imported on that stage.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip(
        "K3.1 Rubin (sm_107) helper; the CPU stage runs these cases", allow_module_level=True
    )

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E402
    rubin_dsa_default_metadata,
    rubin_dsa_metadata,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E402
    rubin_dsa_default_nvfp4 as nvfp4,
)
from tensorrt_llm._torch.attention.backends import interface  # noqa: E402
from tensorrt_llm._torch.attention.backends.sparse.dsa import (  # noqa: E402
    metadata as engine_metadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata  # noqa: E402
from tensorrt_llm.bindings import DataType  # noqa: E402

pytestmark = pytest.mark.cpu_only

EngineMetadata = engine_metadata.DSAtrtllmAttentionMetadata
Base = rubin_dsa_metadata.RubinDSAMetadata
Helix = rubin_dsa_metadata.RubinDSAHelixMetadata
# The hooks read no layout, so the dep16 / tep8 class stands for both. Which hook each class calls,
# and when, is test_rubin_dsa_default_metadata.py's check.
Default = rubin_dsa_default_metadata.RubinDsaDefaultMetadata
STATE = ("nvfp4_kv_cache", "nvfp4_gen_gather", "nvfp4_ctx_gather")

# The step tests' small shapes (real CPU buffers).
SMALL_TOPK, SMALL_HEAD = 4, 32
_MAX_REQUESTS, _MAX_TOKENS, _PAGES = 8, 64, 4


class _Manager:
    """What the hooks and the engine's allocation read off the family cache manager. Its unique
    primary pool is refused as the family manager refuses it when FP8 pages of the last MLA layers
    sit beside NVFP4 ones (``mixed``)."""

    def __init__(self, dtype, *, head_dim=SMALL_HEAD, mixed=True, tokens_per_block=64):
        self.dtype = dtype
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.mixed = mixed
        self.unique_pool_reads = 0

    def get_unique_primary_pool(self):
        self.unique_pool_reads += 1
        if self.mixed:
            raise ValueError("Mixed Kimi DSA storage requires a layer-specific primary pool.")
        # [slots, page scale, 1, elements]: NVFP4 codes, two per byte.
        return torch.zeros(4, 2, 1, self.tokens_per_block * self.head_dim // 2, dtype=torch.int8)


class _Allocations:
    """The bare objects' ``get_empty``: records each request and allocates on the CPU."""

    def __init__(self):
        self.calls = []
        self.frozen = False

    def __call__(self, buffers, tensor_shape, dtype, cache_name, capture_graph=False):
        if self.frozen:
            raise AssertionError(f"{cache_name} allocated after the metadata's creation")
        self.calls.append((cache_name, tuple(tensor_shape), dtype, capture_graph, buffers))
        return torch.empty(tuple(tensor_shape), dtype=dtype)

    def names(self):
        return [call[0] for call in self.calls]


def _set(md, **attrs):
    """Set fields of a bare object; an engine property (``num_tokens``, ``seq_lens``, ...) becomes a
    plain attribute of the object's own class."""
    cls = type(md).__mro__[1]
    for key, value in attrs.items():
        if isinstance(inspect.getattr_static(cls, key, None), property):
            setattr(type(md), key, value)
        else:
            setattr(md, key, value)


def _bare(cls, **attrs):
    """A metadata object without the engine's constructor, of its own subclass of ``cls``."""
    md = object.__new__(type(f"_Bare{cls.__name__}", (cls,), {}))
    _set(md, **attrs)
    return md


def _created(
    manager,
    *,
    sequences=4,
    draft=0,
    topk=SMALL_TOPK,
    graph=False,
    engine_buffers=True,
    is_cross=False,
    hook=True,
):
    """A metadata object as its creation leaves it: the engine DSA post-init's allocation of the
    generation gather's buffers (``engine_buffers``), then the latent-KV hook."""
    alloc = _Allocations()
    md = _bare(
        Default,
        kv_cache_manager=manager,
        max_num_sequences=sequences,
        max_draft_tokens=draft,
        num_sparse_topk=topk,
        is_cuda_graph=graph,
        cuda_graph_buffers=object() if graph else None,
        is_cross=is_cross,
        get_empty=alloc,
        nvfp4_mla_fp8_scratch=None,
        nvfp4_mla_compact_indices=None,
    )
    if engine_buffers:
        EngineMetadata._create_nvfp4_mla_generation_buffers(md, capture_graph=graph)
    if hook:
        md._nvfp4_post_init()
    return md, alloc


def _spec_update(monkeypatch, md, draft):
    """The engine's speculative-decoding update, which the runner calls before every prepare. Only
    the generation gather's part runs; the other spec-decoding buffers are recorded no-ops."""
    monkeypatch.setattr(
        TrtllmAttentionMetadata, "update_spec_dec_param", lambda self, *a, **k: None
    )
    for name in (
        "_create_kv_lens_2d_buffer",
        "create_expanded_buffers",
        "_create_helix_sparse_row_buffers",
        "_create_radix_aux_buffers",
    ):
        setattr(md, name, lambda *a, **k: None)
    _set(md, _radix_rows_per_sequence=1)
    type(md).update_spec_dec_param(
        md,
        batch_size=md.max_num_sequences,
        is_spec_decoding_enabled=draft > 0,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        max_draft_len=draft,
        max_total_draft_tokens=draft,
    )


def _prepare_step(md, seq_lens, cached, num_contexts, hook=True):
    """The fields the specialized preparation fills for a step, through the engine's own methods
    (KV indptrs, per-token request index), then the latent-KV hook (``hook``)."""
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    cached_lens = torch.tensor(cached, dtype=torch.int32)
    num_seqs = len(seq_lens)
    _set(
        md,
        num_contexts=num_contexts,
        num_generations=num_seqs - num_contexts,
        num_seqs=num_seqs,
        num_tokens=int(seq.sum()),
        num_ctx_tokens=int(seq[:num_contexts].sum()),
        seq_lens=seq,
    )
    for name in (
        "ctx_cached_token_indptr",
        "ctx_kv_indptr",
        "gen_cached_token_indptr",
        "gen_kv_indptr",
    ):
        setattr(md, name, torch.zeros(_MAX_REQUESTS + 1, dtype=torch.int64))
        setattr(md, f"host_{name}", torch.zeros(_MAX_REQUESTS + 1, dtype=torch.int64))
    md.req_idx_per_token = torch.full((_MAX_TOKENS,), -1, dtype=torch.int32)
    md.host_req_idx_per_token = torch.full((_MAX_TOKENS,), -1, dtype=torch.int32)
    # Each request's latent-pool slots (the engine's block table: -1 past the request's pages).
    slots = 100 + torch.arange(_MAX_REQUESTS * _PAGES, dtype=torch.int32).view(
        _MAX_REQUESTS, _PAGES
    )
    pages = (cached_lens + seq + 63) // 64
    slots[:num_seqs][torch.arange(_PAGES) >= pages.unsqueeze(1)] = -1
    md.block_table = slots
    md.prepare_for_mla_rope_append(cached_lens, cached_lens + seq)
    md.prepare_for_indices_conversion()
    if hook:
        md._nvfp4_prepare()


def _state(md):
    return {name: getattr(md, name, "unset") for name in STATE}


def _assigned_attributes(path: Path) -> set:
    tree = ast.parse(path.read_text())
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def test_the_hooks_set_new_attribute_names_only():
    """The post-init hook adds exactly its three attributes and ``prepare`` adds none. None of them
    is a class name or an attribute the engine or specialized metadata assigns, so the engine's own
    buffers keep their meaning."""
    md, _ = _created(_Manager(DataType.NVFP4), draft=1, hook=False)
    before = set(vars(md))
    md._nvfp4_post_init()
    assert set(vars(md)) - before == set(STATE)
    _prepare_step(md, [5, 2], [0, 9], 1, hook=False)
    before = set(vars(md))
    md._nvfp4_prepare()
    assert set(vars(md)) == before
    modules = (
        engine_metadata,
        rubin_dsa_metadata,
        interface,
        inspect.getmodule(TrtllmAttentionMetadata),
    )
    taken = set().union(*(_assigned_attributes(Path(inspect.getfile(m))) for m in modules))
    for klass in (Base, Helix, *EngineMetadata.__mro__):
        taken |= set(vars(klass))
    assert {"nvfp4_mla_fp8_scratch", "nvfp4_mla_compact_indices"} <= taken
    assert not set(STATE) & taken


# ----------------------------------------------------------------------------- no latent-KV state


@pytest.mark.parametrize("case", ["fp8", "no-cache", "cross"])
def test_an_fp8_cache_no_cache_or_a_cross_metadata_takes_no_latent_kv_state(case):
    manager = {
        "fp8": _Manager(DataType.FP8),
        "no-cache": None,
        "cross": _Manager(DataType.NVFP4),
    }[case]
    md, alloc = _created(manager, engine_buffers=False, is_cross=case == "cross", hook=False)
    alloc.frozen = True
    md._nvfp4_post_init()
    assert _state(md) == {
        "nvfp4_kv_cache": False,
        "nvfp4_gen_gather": None,
        "nvfp4_ctx_gather": None,
    }
    md._nvfp4_prepare()
    md._nvfp4_on_update_kv_lens()
    assert _state(md)["nvfp4_gen_gather"] is None and _state(md)["nvfp4_ctx_gather"] is None
    assert md.nvfp4_mla_fp8_scratch is None and not alloc.calls


def test_the_cache_test_is_the_engines():
    """The latent KV is NVFP4 exactly when the engine's DSA post-init allocates the buffers."""
    for dtype, nvfp4_cache in ((DataType.NVFP4, True), (DataType.FP8, False), (None, False)):
        manager = _Manager(dtype)
        assert nvfp4.nvfp4_latent_cache(manager) is nvfp4_cache
        md, _ = _created(manager, hook=False)
        assert (md.nvfp4_mla_fp8_scratch is not None) is nvfp4_cache
    assert nvfp4.nvfp4_latent_cache(None) is False


# ----------------------------------------------------------------------------- the scratch


def test_the_scratch_exists_before_the_kv_cache_estimate(monkeypatch):
    """The KV cache estimate measures a profiling run whose first forward creates the eager
    metadata, updates its draft length and prepares a step. The scratch is complete after the
    first two, before any step, and no step allocates: the estimate counts all of it."""
    sequences, draft = 4, 3
    md, alloc = _created(_Manager(DataType.NVFP4), sequences=sequences, draft=0)
    # Creation: the engine's size for the draft length the eager metadata has (none yet).
    assert md.nvfp4_kv_cache is True
    assert alloc.names() == ["nvfp4_mla_fp8_scratch", "nvfp4_mla_compact_indices"]
    assert tuple(md.nvfp4_mla_fp8_scratch.shape) == (sequences, SMALL_TOPK, SMALL_HEAD)
    # The runner's speculative-decoding update, before every prepare: the engine's own method.
    assert type(md).update_spec_dec_param is EngineMetadata.update_spec_dec_param
    _spec_update(monkeypatch, md, draft)
    rows = sequences * (1 + draft)
    scratch, compact = md.nvfp4_mla_fp8_scratch, md.nvfp4_mla_compact_indices
    assert tuple(scratch.shape) == (rows, SMALL_TOPK, SMALL_HEAD)
    assert tuple(compact.shape) == (rows, SMALL_TOPK)
    assert scratch.dtype == torch.float8_e4m3fn and compact.dtype == torch.int32
    # The profiling run's steps (a context step, then a full verify step) and the KV-length hook
    # allocate nothing and use the buffers the update left.
    alloc.frozen = True
    _prepare_step(md, [7, 5], [0, 0], 2)
    assert md.nvfp4_gen_gather is None and md.nvfp4_ctx_gather is not None
    _prepare_step(md, [1 + draft] * sequences, [30] * sequences, 0)
    md._nvfp4_on_update_kv_lens()
    gen = md.nvfp4_gen_gather
    assert gen.rows == rows and md.nvfp4_ctx_gather is None
    assert gen.out.data_ptr() == scratch.data_ptr()
    assert gen.compact_indices.data_ptr() == compact.data_ptr()
    assert md.nvfp4_mla_fp8_scratch is scratch and md.nvfp4_mla_compact_indices is compact


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_missing_buffers_are_allocated_at_creation_the_engines_way(graph):
    """If the engine left the buffers out, the post-init hook allocates them: the engine's names,
    size and buffer pool. Nothing else does."""
    md, alloc = _created(
        _Manager(DataType.NVFP4), sequences=2, draft=5, graph=graph, engine_buffers=False
    )
    rows = 2 * 6
    assert [(name, shape, dtype, g) for name, shape, dtype, g, _ in alloc.calls] == [
        ("nvfp4_mla_fp8_scratch", (rows, SMALL_TOPK, SMALL_HEAD), torch.float8_e4m3fn, graph),
        ("nvfp4_mla_compact_indices", (rows, SMALL_TOPK), torch.int32, graph),
    ]
    assert tuple(md.nvfp4_mla_fp8_scratch.shape) == (rows, SMALL_TOPK, SMALL_HEAD)


def _wrong(case, rows):
    fp8 = torch.float8_e4m3fn
    return {
        "short": (torch.empty(rows - 1, SMALL_TOPK, SMALL_HEAD, dtype=fp8), None),
        "topk": (torch.empty(rows, SMALL_TOPK + 1, SMALL_HEAD, dtype=fp8), None),
        "head_dim": (torch.empty(rows, SMALL_TOPK, SMALL_HEAD // 2, dtype=fp8), None),
        "bf16": (torch.empty(rows, SMALL_TOPK, SMALL_HEAD, dtype=torch.bfloat16), None),
        "strided": (torch.empty(SMALL_HEAD, SMALL_TOPK, rows, dtype=fp8).permute(2, 1, 0), None),
        "short_offsets": (None, torch.empty(rows - 1, SMALL_TOPK, dtype=torch.int32)),
        "int64_offsets": (None, torch.empty(rows, SMALL_TOPK, dtype=torch.int64)),
    }[case]


@pytest.mark.parametrize(
    "case", ["short", "topk", "head_dim", "bf16", "strided", "short_offsets", "int64_offsets"]
)
def test_buffers_of_another_size_or_type_are_refused_at_creation(case):
    md, alloc = _created(_Manager(DataType.NVFP4), draft=2, hook=False)
    scratch, compact = _wrong(case, md.nvfp4_mla_fp8_scratch.shape[0])
    if scratch is not None:
        md.nvfp4_mla_fp8_scratch = scratch
    if compact is not None:
        md.nvfp4_mla_compact_indices = compact
    alloc.frozen = True
    with pytest.raises(RuntimeError, match="generation gather's FP8 scratch"):
        md._nvfp4_post_init()


# ----------------------------------------------------------------------------- the gather plan


def test_a_mixed_step_gets_both_gathers():
    """Two context requests (the second chunk of one, a fresh one) and a verify request."""
    md, _ = _created(_Manager(DataType.NVFP4), sequences=4, draft=3)
    _prepare_step(md, [5, 3, 4], [4, 0, 20], 2)
    ctx, gen = md.nvfp4_ctx_gather, md.nvfp4_gen_gather
    assert isinstance(ctx, nvfp4.Nvfp4ContextGather)
    assert ctx.query_req_indices.tolist() == [0] * 5 + [1] * 3
    assert ctx.cu_kv_lengths.tolist() == [0, 9, 12]
    assert ctx.total_kv_tokens == 12 and type(ctx.total_kv_tokens) is int
    assert torch.equal(ctx.block_table, md.block_table[:2])
    assert ctx.block_table[0].tolist() == [100, -1, -1, -1]
    # The context gather's input types: int32 rows and slots, int64 offsets, all contiguous views.
    assert ctx.query_req_indices.dtype == torch.int32 and ctx.block_table.dtype == torch.int32
    assert ctx.cu_kv_lengths.dtype == torch.int64
    for mine, theirs in (
        (ctx.query_req_indices, md.req_idx_per_token),
        (ctx.block_table, md.block_table),
        (ctx.cu_kv_lengths, md.ctx_kv_indptr),
    ):
        assert mine.is_contiguous() and mine.data_ptr() == theirs.data_ptr()
    # The step's 4 generation rows are the leading rows of the 16-row stable buffers.
    assert gen.rows == 4
    assert gen.out.data_ptr() == md.nvfp4_mla_fp8_scratch.data_ptr()
    assert gen.compact_indices.data_ptr() == md.nvfp4_mla_compact_indices.data_ptr()


def test_a_step_past_the_buffers_is_refused_before_any_launch():
    """A metadata sized for one draft token cannot hold a verify step of three."""
    md, _ = _created(_Manager(DataType.NVFP4), sequences=4, draft=1)
    with pytest.raises(RuntimeError, match=r"16 generation rows exceed .* 8 rows"):
        _prepare_step(md, [4, 4, 4, 4], [10, 10, 10, 10], 0)


@pytest.mark.parametrize("defect", ["count", "zero"])
def test_a_context_step_with_disagreeing_or_no_kv_tokens_is_refused(defect):
    md, _ = _created(_Manager(DataType.NVFP4), draft=1)
    _prepare_step(md, [5, 3], [4, 0], 2)
    md.num_ctx_mla_kv_tokens = md.num_ctx_mla_kv_tokens + 1 if defect == "count" else 0
    with pytest.raises(RuntimeError, match="NVFP4 context gather"):
        md._nvfp4_prepare()


def test_the_context_inputs_are_what_the_engine_backend_passes():
    """With one NVFP4 pool the engine's per-step pool cache can run: its context-gather inputs are
    the plan's, tensor for tensor."""
    md, _ = _created(_Manager(DataType.NVFP4, mixed=False), sequences=4, draft=3)
    _prepare_step(md, [5, 3, 4], [4, 0, 20], 2)
    _set(md, _pool_cache_valid=False, _cached_kv_mgr_id=0)
    md._ensure_pool_view_cached()
    ctx = md.nvfp4_ctx_gather
    assert md._cached_pool_view is None  # NVFP4 codes have no [token, head_dim] view
    for mine, engine in (
        (ctx.query_req_indices, md._cached_req_idx_ctx),
        (ctx.block_table, md._cached_block_table_ctx),
    ):
        assert torch.equal(mine, engine) and mine.data_ptr() == engine.data_ptr()
    assert torch.equal(ctx.cu_kv_lengths, md.ctx_kv_indptr[: md.num_contexts + 1])
    assert ctx.total_kv_tokens == md.num_ctx_mla_kv_tokens


def test_the_context_inputs_need_no_unique_primary_pool():
    """FP8 pages of the last MLA layers beside NVFP4 ones: the engine's per-step pool cache cannot
    run, the plan never asks for the unique pool."""
    manager = _Manager(DataType.NVFP4, mixed=True)
    md, _ = _created(manager, sequences=4, draft=3)
    _prepare_step(md, [5, 3, 4], [4, 0, 20], 2)
    assert manager.unique_pool_reads == 0 and md.nvfp4_ctx_gather.total_kv_tokens == 12
    _set(md, _pool_cache_valid=False, _cached_kv_mgr_id=0)
    with pytest.raises(ValueError, match="layer-specific primary pool"):
        md._ensure_pool_view_cached()


# ----------------------------------------------------------------------------- the KV-length hook


def test_the_kv_length_hook_changes_nothing():
    md, alloc = _created(_Manager(DataType.NVFP4), sequences=4, draft=3)
    _prepare_step(md, [5, 4], [4, 30], 1)
    before = dict(vars(md))
    alloc.frozen = True
    md._nvfp4_on_update_kv_lens()
    after = vars(md)
    assert after.keys() == before.keys()
    assert all(after[name] is value for name, value in before.items())


_ALLOCATING = frozenset(
    {
        "get_empty",
        "empty",
        "zeros",
        "ones",
        "full",
        "empty_like",
        "zeros_like",
        "ones_like",
        "full_like",
        "new_empty",
        "new_zeros",
        "new_ones",
        "new_full",
        "tensor",
        "arange",
        "clone",
        "contiguous",
        "_create_nvfp4_mla_generation_buffers",
    }
)


def test_no_step_hook_allocates():
    """Allocation is the post-init hook's only: ``prepare`` and what it calls slice and read."""
    tree = ast.parse(inspect.getsource(nvfp4))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in (
        "_nvfp4_prepare",
        "_generation_gather",
        "_context_gather",
        "_nvfp4_on_update_kv_lens",
    ):
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
        }
        assert not called & _ALLOCATING, name
