# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MSA indexer's prefill context scorer and the step plans it makes dead (CPU).

- MSA_LAZY_DENSE_PLAN: prepare() defers the dense-layer plan to its first read, which builds it once with the eager
  build's arguments; the kill switch builds it eagerly; a first read under capture raises, naming the switch.
- MSA_LAZY_PROXY_PLAN: with a prefill scorer armed, a pure-prefill step's proxy plan (plain or segmented) is
  built on their first read, once, equal to the eager build; not armed, a mixed step, the kill switch, or lengths the
  scorer's accepts() rejects: eager (in prepare(), off the forward's critical path).
- prewarm_prefill_context_scorer runs the armed scorer's warm-up (the executor's MSA indexer prewarm calls it).
- MsaIndexer.select_prefill_hooked: the hook's scores reach prefill_scorer_selector first, then the select as is
  (prefill select), the proxy never runs; no other select ever calls prefill_scorer_selector;
  a decline returns None and is counted; a taken step drops the reused proxy buffer.
- run_indexer calls the hook on pure-prefill FP8 / NVFP4 steps only, before any read of the proxy plan or its page
  table; a decline falls through to the proxy path with the step's page table.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import (
    msa_backend,
    msa_indexer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
    nvfp4_index_k,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_backend import (
    MiniMaxM31MsaSparseAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_indexer import (
    MsaIndexer,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy

pytestmark = pytest.mark.cpu_only

PAGE = 128
I64_MAX = torch.iinfo(torch.int64).max


@pytest.fixture(autouse=True)
def _cpu_capture_answer(monkeypatch):
    """A CPU-only torch raises from torch.cuda.is_current_stream_capturing; a CUDA build answers False outside a
    capture, which is what these host-side tests need (tests that model a capture patch it themselves)."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.fixture(autouse=True)
def _disarmed():
    """Every test starts and ends with no prefill scorer armed and fresh counters."""
    msa_indexer.arm_prefill_context_scorer(0)
    counts = dict(msa_indexer._PREFILL_SCORER_COUNTS)
    yield
    msa_indexer.arm_prefill_context_scorer(0)
    msa_indexer._PREFILL_SCORER_COUNTS.update(counts)


# ---- step plans ----------------------------------------------------------------------------------------------------


def _plan_metadata(
    monkeypatch, *, num_contexts, qo_lens, kv_lens, policy=MsaRuntimePolicy(), segmented=None
):
    """A metadata with just the state _build_step_plans reads, over a recording fake fmha_sm100_plan (and a stand-in
    for segmented_proxy_plan: None plans unsplit)."""
    planned = []
    monkeypatch.setattr(msa_backend, "segmented_proxy_plan", segmented or (lambda *a, **kw: None))

    def fmha_sm100_plan(qo_lens_cpu, kv_lens_cpu, **kwargs):
        call = dict(kwargs, qo=qo_lens_cpu.tolist(), kv=kv_lens_cpu.tolist())
        planned.append(call)
        return call

    monkeypatch.setattr(
        msa_backend, "require_msa_module", lambda: SimpleNamespace(fmha_sm100_plan=fmha_sm100_plan)
    )
    monkeypatch.setattr(msa_backend, "_cache_device", lambda metadata: torch.device("cpu"))
    cls = MiniMaxM31MsaSparseAttention.Metadata
    metadata = cls.__new__(cls)
    metadata.mapping = None
    metadata._seq_lens = torch.tensor(qo_lens, dtype=torch.int32)
    metadata.num_contexts = num_contexts
    metadata.kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    metadata.is_cuda_graph = False
    metadata.kv_cache_manager = SimpleNamespace(tokens_per_block=PAGE, dtype=None)
    metadata._stage_host_lengths()
    metadata._set_decode_span()
    metadata._msa_fields_ready = True
    metadata._msa_params = SimpleNamespace(
        sharded_index_head_count=lambda mapping: 4,
        sharded_head_counts=lambda mapping: (16, 4),
        topk=16,
    )
    metadata.msa_policy = policy
    metadata.msa_kv_indices = torch.arange(64, dtype=torch.int32)
    return metadata, planned


def _same(a, b) -> bool:
    """Plans (kwargs dicts holding host tensors) equal value for value."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b


PREFILL = dict(num_contexts=2, qo_lens=(300, 700), kv_lens=(300 + 4096, 700 + 9000))
MIXED = dict(num_contexts=1, qo_lens=(700, 1, 1), kv_lens=(700 + 9000, 40, 33))


def test_dense_plan_is_built_on_its_first_read_with_the_eager_arguments(monkeypatch) -> None:
    monkeypatch.setattr(msa_backend, "_LAZY_DENSE_PLAN", False)
    eager, eager_planned = _plan_metadata(monkeypatch, **PREFILL)
    eager._build_step_plans()
    assert len(eager_planned) == 3  # proxy, sparse GQA, dense
    monkeypatch.setattr(msa_backend, "_LAZY_DENSE_PLAN", True)
    lazy, planned = _plan_metadata(monkeypatch, **PREFILL)
    lazy._build_step_plans()
    assert len(planned) == 2 and lazy._msa_prefill_dense_plan is None
    dense = lazy.msa_prefill_dense_plan
    assert len(planned) == 3 and _same(dense, eager.msa_prefill_dense_plan)
    assert eager.msa_prefill_dense_plan is eager_planned[2]
    assert "kv_block_num" not in dense and dense["use_fp8_kvcache"] is False
    assert lazy.msa_prefill_dense_plan is dense and len(planned) == 3  # once per step
    # The next step's prepare() drops it: a read then builds the new step's plan.
    lazy._build_step_plans()
    assert lazy._msa_prefill_dense_plan is None and len(planned) == 5
    lazy.msa_prefill_dense_plan
    assert len(planned) == 6


def test_dense_plan_first_read_under_capture_raises_naming_the_switch(monkeypatch) -> None:
    lazy, planned = _plan_metadata(monkeypatch, **PREFILL)
    lazy._build_step_plans()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="TRTLLM_M31_DISABLE_MSA_LAZY_DENSE_PLAN"):
        lazy.msa_prefill_dense_plan
    assert len(planned) == 2


def _proxy_view(metadata):
    return (metadata.msa_prefill_proxy_plan, metadata.msa_prefill_proxy_kv_indices)


@pytest.mark.parametrize("segmented", [False, True])
def test_armed_pure_prefill_builds_the_proxy_plans_on_first_read(monkeypatch, segmented):
    seg_kv = torch.arange(7, dtype=torch.int32)

    def segment(fmha, qo, off, **kw):
        return ({"segmented": qo.tolist()}, seg_kv) if segmented else None

    policy = SimpleNamespace(
        short_sequence_identity=False,
        kv_update=None,
        prefill_proxy_qo_tile=128,
        prefill_proxy_kv_splits=4,
        prefill_proxy_kv_splits_by_rows=None,
    )
    monkeypatch.setattr(msa_indexer, "_plan_takes_maxscore_tile", lambda plan: True)
    eager, eager_planned = _plan_metadata(monkeypatch, policy=policy, segmented=segment, **PREFILL)
    eager._build_step_plans()
    want = _proxy_view(eager)
    msa_indexer.arm_prefill_context_scorer(60)
    lazy, planned = _plan_metadata(monkeypatch, policy=policy, segmented=segment, **PREFILL)
    lazy._build_step_plans()
    # prepare() planned only the attention side: no proxy plan_kernel on a step the scorer may own.
    assert len(planned) == len(eager_planned) - (0 if segmented else 1)
    assert lazy._msa_prefill_proxy_plan is None and lazy._msa_prefill_proxy_plan_thunk is not None
    before = len(planned)
    got = _proxy_view(lazy)
    assert len(planned) == before + (0 if segmented else 1)
    assert _same(got[0], want[0])
    assert torch.equal(got[1], want[1])
    if segmented:
        assert got[1] is seg_kv
    else:
        assert got[0]["maxscore_qo_tile"] == 128 and got[0]["num_kv_splits"] == 4
        assert got[1] is lazy.msa_kv_indices
    assert _proxy_view(lazy)[0] is got[0] and len(planned) == before + (0 if segmented else 1)


@pytest.mark.parametrize(
    "case",
    ["not armed", "mixed step", "kill switch"],
)
def test_proxy_plan_stays_eager(monkeypatch, case) -> None:
    if case != "not armed":
        msa_indexer.arm_prefill_context_scorer(60)
    if case == "kill switch":
        monkeypatch.setattr(msa_backend, "_LAZY_PROXY_PLAN", False)
    metadata, planned = _plan_metadata(monkeypatch, **(MIXED if case == "mixed step" else PREFILL))
    metadata._build_step_plans()
    assert metadata._msa_prefill_proxy_plan_thunk is None
    assert metadata._msa_prefill_proxy_plan is not None and planned[0]["output_maxscore"]
    assert planned[0]["qo"] == [700 if case == "mixed step" else 300] + (
        [] if case == "mixed step" else [700]
    )


@pytest.mark.parametrize("ok", [True, False])
def test_armed_scorer_accepts_decides_lazy_or_eager(monkeypatch, ok) -> None:
    """A step the armed scorer declines on its host lengths is planned in prepare() (not on the forward's critical
    path, inside the first declining layer); accepts() sees the step's own length objects."""
    seen = []

    def accepts(qo, kv, off):
        seen.append((qo, kv, off))
        return ok

    msa_indexer.arm_prefill_context_scorer(60, accepts=accepts)
    metadata, planned = _plan_metadata(monkeypatch, **PREFILL)
    metadata._build_step_plans()
    assert len(seen) == 1
    assert seen[0][0] is metadata.msa_qo_lens_cpu and seen[0][1] is metadata.msa_kv_lens_cpu
    assert seen[0][2] is metadata.msa_qo_offset_cpu
    assert (metadata._msa_prefill_proxy_plan_thunk is not None) is ok
    assert (metadata._msa_prefill_proxy_plan is None) is ok
    assert any(call.get("output_maxscore") for call in planned) is (not ok)


def test_arm_records_accepts_and_prewarm_and_disarm_clears_them() -> None:
    warmed = []
    msa_indexer.arm_prefill_context_scorer(
        3, accepts=lambda *a: False, prewarm=lambda heads: warmed.append(heads)
    )
    assert not msa_indexer.prefill_context_scorer_accepts(None, None, None)
    assert msa_indexer.prewarm_prefill_context_scorer(4) and warmed == [4]
    msa_indexer.arm_prefill_context_scorer(0, accepts=lambda *a: True, prewarm=warmed.append)
    assert (
        msa_indexer._PREFILL_SCORER_ACCEPTS is None and msa_indexer._PREFILL_SCORER_PREWARM is None
    )
    assert not msa_indexer.prefill_context_scorer_accepts(None, None, None)
    assert not msa_indexer.prewarm_prefill_context_scorer(4) and warmed == [4]
    msa_indexer.arm_prefill_context_scorer(2)  # no predicate: every step, no warm-up
    assert msa_indexer.prefill_context_scorer_accepts(None, None, None)
    assert not msa_indexer.prewarm_prefill_context_scorer(4)


def test_the_executor_prewarm_runs_the_scorer_prewarm() -> None:
    """The scorer's kernels compile in the family MSA indexer prewarm the engine runs before ready
    (RuntimeHooks.sparse_attention.indexer_prewarm), not on the first long request: the warm-up's short prefills take
    the identity shortcut ahead of the hook."""
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import engine

    body = inspect.getsource(engine.indexer_prewarm)
    call = body.index("prewarm_prefill_context_scorer(num_index_heads)")
    assert (
        body.index("prewarm_split_proxy_variants(") < call < body.index("torch.cuda.synchronize()")
    )
    assert body.index("with torch.inference_mode():") < call


def test_an_unread_deferred_plan_never_outlives_its_step(monkeypatch) -> None:
    """Every layer's scorer took step 1, so its deferred plan was never built; step 2 (a mixed step, planned eagerly)
    must read its own plan, not step 1's."""
    msa_indexer.arm_prefill_context_scorer(60)
    metadata, planned = _plan_metadata(monkeypatch, **PREFILL)
    metadata._build_step_plans()
    assert metadata._msa_prefill_proxy_plan_thunk is not None
    metadata._seq_lens = torch.tensor(MIXED["qo_lens"], dtype=torch.int32)
    metadata.num_contexts = MIXED["num_contexts"]
    metadata.kv_lens = torch.tensor(MIXED["kv_lens"], dtype=torch.int32)
    metadata._stage_host_lengths()
    metadata._set_decode_span()
    metadata._build_step_plans()
    assert metadata._msa_prefill_proxy_plan_thunk is None
    assert metadata.msa_prefill_proxy_plan["qo"] == [700]


def test_proxy_plan_first_read_under_capture_raises_naming_the_switch(monkeypatch) -> None:
    msa_indexer.arm_prefill_context_scorer(60)
    lazy, _ = _plan_metadata(monkeypatch, **PREFILL)
    lazy._build_step_plans()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="TRTLLM_M31_DISABLE_MSA_LAZY_PROXY_PLAN"):
        lazy.msa_prefill_proxy_plan


# ---- MsaIndexer.select_prefill_hooked ------------------------------------------------------------------------------


def _config(num_kv_heads=4, num_index_heads=4, init_blocks=0, local_blocks=1):
    return MiniMaxM3SparseConfig(
        num_q_heads=16,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        num_index_heads=num_index_heads,
        sparse_index_dim=128,
        block_size=PAGE,
        topk=16,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


def _key(bits):
    bits = bits.to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, bits, ~bits & 0x7FFFFFFF)


def ref_select(
    max_score_kv, *, topk, n_valid_blocks, init_blocks, local_blocks, head_major_output=False
):
    """minimax_m3_select_blocks' contract as a host key sort (any strides): [total_q, heads, topk]."""
    heads, blocks, rows = max_score_kv.shape
    key = _key(max_score_kv.contiguous().view(torch.int32))
    raw = n_valid_blocks.to(torch.int64).view(1, 1, rows)
    b = torch.arange(blocks).view(1, blocks, 1)
    forced = _key(torch.tensor([1.0e30, 1.0e29]).view(torch.int32))
    key = torch.where(b < init_blocks, forced[0], key)
    key = torch.where(b >= (raw - local_blocks).clamp(min=0), forced[1], key)
    length = raw.clamp(0, blocks)
    ranked = torch.where(b < length, (key << 31) | b, I64_MAX)
    if blocks < topk:  # fewer blocks than picks: pad with empty ranks
        ranked = torch.cat([ranked, ranked.new_full((heads, topk - blocks, rows), I64_MAX)], dim=1)
    top = ranked.sort(dim=1).values[:, :topk, :]
    valid = (torch.arange(topk).view(1, topk, 1) < length.clamp(max=topk)) & (
        (top >> 31) < 0xFF800000
    )
    ids = torch.where(valid, top & 0x7FFFFFFF, I64_MAX).sort(dim=1).values
    table = torch.where(ids == I64_MAX, -1, ids).to(torch.int32)  # [heads, topk, rows]
    return table.permute(0, 2, 1).contiguous().permute(1, 0, 2)  # head-major backing


def _hook_inputs(qo=(300,), prefix=(4096,), heads=4):
    qo_cpu = torch.tensor(qo, dtype=torch.int32)
    kv_cpu = qo_cpu + torch.tensor(prefix, dtype=torch.int32)
    q = torch.zeros((sum(qo), heads, 128), dtype=torch.float8_e4m3fn)
    pages = int(sum(-(-int(k) // PAGE) for k in kv_cpu))
    k = torch.zeros((pages, 1, PAGE, 128), dtype=torch.float8_e4m3fn)
    return q, k, qo_cpu, kv_cpu, kv_cpu - qo_cpu, torch.arange(pages, dtype=torch.int32)


def test_hook_scores_reach_the_prefill_select_as_is(monkeypatch) -> None:
    ix = MsaIndexer(_config())
    q, k, qo, kv, off, kv_indices = _hook_inputs()
    view = torch.randn(300, 4, 40).permute(1, 2, 0)  # row-major [H, pages, T], unit block stride
    calls = []

    def hook(*args):
        calls.append(args)
        return view

    def proxy(*a, **kw):
        raise AssertionError("the proxy ran on a hooked step")

    selected = []
    table = torch.zeros(300, 4, 16, dtype=torch.int32)

    def select_kv(scores, n_valid, *, prefill=False, out=None):
        selected.append((scores, n_valid, prefill))
        return table

    monkeypatch.setattr(ix, "_proxy_scores", proxy)
    monkeypatch.setattr(ix, "_select_kv", select_kv)
    ix.prefill_scorer_hook = hook
    nvb = torch.ones(300, dtype=torch.int32)
    assert ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, nvb) is table
    assert len(calls) == 1 and all(
        a is b for a, b in zip(calls[0], (q, k, qo, kv, off, kv_indices))
    )
    assert len(calls[0]) == 6
    assert selected[0][0] is view and selected[0][1] is nvb and selected[0][2] is True
    assert msa_indexer._PREFILL_SCORER_COUNTS["taken"] >= 1


def test_the_prefill_scorer_selector_is_tried_first(monkeypatch) -> None:
    ix = MsaIndexer(_config(init_blocks=2, local_blocks=3))
    q, k, qo, kv, off, kv_indices = _hook_inputs()
    view = torch.randn(300, 4, 40).permute(1, 2, 0)
    ix.prefill_scorer_hook = lambda *a: view
    table = torch.zeros(300, 4, 16, dtype=torch.int32)
    picked, fallback = [], []
    monkeypatch.setattr(ix, "_select_kv", lambda *a, **kw: fallback.append(a) or table.clone())
    ix.prefill_scorer_selector = lambda *a: picked.append(a) or table
    nvb = torch.ones(300, dtype=torch.int32)
    assert ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, nvb) is table
    assert len(picked) == 1 and picked[0][0] is view and picked[0][1] is nvb
    assert picked[0][2:] == (2, 3) and fallback == []
    # It declines: the indexer's own select runs on the same scores.
    ix.prefill_scorer_selector = lambda *a: picked.append(a) and None
    assert ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, nvb) is not table
    assert len(fallback) == 1 and fallback[0][0] is view


def test_no_other_select_calls_the_prefill_scorer_selector(monkeypatch) -> None:
    ix = MsaIndexer(_config())
    ix.prefill_scorer_selector = lambda *a: pytest.fail(
        "a non-hooked select reached the scorer's selector"
    )
    ix.selector_hook = lambda *a: torch.zeros(300, 4, 16, dtype=torch.int32)
    scores = torch.randn(300, 4, 40).permute(1, 2, 0)
    nvb = torch.ones(300, dtype=torch.int32)
    for prefill in (False, True):
        assert ix._select_kv(scores, nvb, prefill=prefill) is not None
        assert ix._select(scores, nvb, prefill=prefill) is not None


def test_no_hook_or_no_lengths_returns_none() -> None:
    ix = MsaIndexer(_config())
    q, k, qo, kv, off, kv_indices = _hook_inputs()
    assert ix.prefill_scorer_hook is None
    assert ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, None) is None
    ix.prefill_scorer_hook = lambda *a: pytest.fail("called without lengths")
    assert ix.select_prefill_hooked(q, k, None, kv, off, kv_indices, None) is None


def test_decline_returns_none_counts_and_warns(monkeypatch) -> None:
    ix = MsaIndexer(_config())
    warnings = []
    monkeypatch.setattr(msa_indexer.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    monkeypatch.setitem(msa_indexer._PREFILL_SCORER_COUNTS, "declined", 0)
    ix.prefill_scorer_hook = lambda *a: None
    q, k, qo, kv, off, kv_indices = _hook_inputs()
    for _ in range(12):
        assert ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, None) is None
    assert msa_indexer._PREFILL_SCORER_COUNTS["declined"] == 12
    assert len(warnings) == 2 and all("MSA_PREFILL_SCORER declined" in w for w in warnings)


def test_taken_step_drops_the_reused_proxy_buffer(monkeypatch) -> None:
    ix = MsaIndexer(_config())
    monkeypatch.setattr(msa_indexer, "_proxy_ms_step", ("plan", 300, torch.empty(1)))
    monkeypatch.setattr(msa_indexer, "select_blocks_from_maxscore", ref_select)
    ix.prefill_scorer_hook = lambda *a: torch.randn(4, 40, 300)
    q, k, qo, kv, off, kv_indices = _hook_inputs()
    ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, None)
    assert msa_indexer._proxy_ms_step is None


def test_n_valid_blocks_default_is_the_per_token_count(monkeypatch) -> None:
    ix = MsaIndexer(_config())
    seen = []
    monkeypatch.setattr(ix, "_select_kv", lambda s, n, **kw: seen.append(n) or "t")
    ix.prefill_scorer_hook = lambda *a: torch.randn(4, 40, 300)
    q, k, qo, kv, off, kv_indices = _hook_inputs()
    ix.select_prefill_hooked(q, k, qo, kv, off, kv_indices, None)
    pos = torch.arange(4096, 4096 + 300)
    assert torch.equal(seen[0].to(torch.int64), (pos + PAGE) // PAGE)


# ---- run_indexer: where the hook is called -------------------------------------------------------------------------


class _SpyMetadata:
    """The metadata run_indexer reads, with spies on the three proxy-plan properties."""

    msa_max_blocks_per_seq = 0
    _msa_prefill_identity = None
    msa_worst_case_max_k_tiles = 8

    def __init__(
        self, qo, prefix, *, span=None, policy=MsaRuntimePolicy(), cache_dtype=torch.uint8
    ):
        self.reads = []
        self.msa_qo_lens_cpu = torch.tensor(qo, dtype=torch.int32)
        self.msa_kv_lens_cpu = self.msa_qo_lens_cpu + torch.tensor(prefix, dtype=torch.int32)
        self.msa_qo_offset_cpu = self.msa_kv_lens_cpu - self.msa_qo_lens_cpu
        self.num_seqs = len(qo)
        self.msa_decode_span = span
        self.msa_policy = policy
        pages = int(sum(-(-int(k) // PAGE) for k in self.msa_kv_lens_cpu))
        self.msa_kv_indices = torch.arange(pages, dtype=torch.int32).flip(0)
        self.msa_block_table = torch.arange(pages, dtype=torch.int32).view(1, -1)
        self.msa_seq_lens_cuda = self.msa_kv_lens_cpu.clone()
        self.msa_prefill_n_valid_blocks = torch.ones(sum(qo) + 8, dtype=torch.int32)
        self.msa_n_valid_blocks = self.msa_prefill_n_valid_blocks
        width = 128 if cache_dtype != torch.uint8 else 128 // 2 + 128 // 16
        self._cache = torch.zeros((pages, 1, PAGE, width), dtype=cache_dtype)

    def msa_idx_k_cache(self, layer_idx):
        return self._cache

    def msa_proxy_max_score_view(self, heads, tiles, tokens):
        return torch.zeros(heads, tiles, tokens)

    @property
    def msa_prefill_proxy_plan(self):
        self.reads.append("plan")
        return ("proxy-plan",)

    @property
    def msa_prefill_proxy_kv_indices(self):
        self.reads.append("kv_indices")
        return "proxy-kv-indices"


def _attention(indexer, kv_dtype):
    return SimpleNamespace(
        layer_idx=0,
        m3_config=SimpleNamespace(sparse_index_dim=128, num_index_heads=4, num_kv_heads=4),
        indexer=indexer,
        indexer_kv_dtype=kv_dtype,
    )


def _nvfp4_stubs(monkeypatch):
    compact_kv = torch.arange(3, dtype=torch.int32)
    gathered = torch.zeros((3, 1, PAGE, 128), dtype=torch.float8_e4m3fn)
    monkeypatch.setattr(
        nvfp4_index_k,
        "compact_step_pages",
        lambda bt, kvl, rows: (torch.arange(3), compact_kv, torch.zeros(1, 3, dtype=torch.int32)),
    )
    monkeypatch.setattr(nvfp4_index_k, "gather_fp8", lambda cache, ids, out: gathered)
    return compact_kv, gathered


def _hooked_indexer(result):
    ix = MsaIndexer(_config())
    ix.hook_calls = []
    ix.select_calls = []

    def hook(*args):
        ix.hook_calls.append(args)
        return result

    ix.prefill_scorer_hook = hook
    ix.select_blocks = lambda *a, **kw: ix.select_calls.append(kw) or "proxy-table"
    return ix


def test_nvfp4_pure_prefill_hook_reads_no_proxy_plan(monkeypatch) -> None:
    compact_kv, gathered = _nvfp4_stubs(monkeypatch)
    monkeypatch.setattr(msa_indexer, "select_blocks_from_maxscore", ref_select)
    ix = _hooked_indexer(torch.randn(4, 3, 200))
    meta = _SpyMetadata([200], [100])
    q = torch.zeros((200, 4 * 128), dtype=torch.float8_e4m3fn)
    table = MiniMaxM31MsaSparseAttention.run_indexer(_attention(ix, "nvfp4"), q, None, meta)
    assert table.shape == (200, 4, 16) and not ix.select_calls
    assert meta.reads == []  # neither the plan nor its page table was read
    args = ix.hook_calls[0]
    assert args[1] is gathered and args[5] is compact_kv and args[0].shape == (200, 4, 128)
    assert args[2] is meta.msa_qo_lens_cpu and args[3] is meta.msa_kv_lens_cpu


def test_nvfp4_decline_falls_through_with_the_step_page_table(monkeypatch) -> None:
    compact_kv, gathered = _nvfp4_stubs(monkeypatch)
    ix = _hooked_indexer(None)
    meta = _SpyMetadata([200], [100])
    q = torch.zeros((200, 4 * 128), dtype=torch.float8_e4m3fn)
    assert (
        MiniMaxM31MsaSparseAttention.run_indexer(_attention(ix, "nvfp4"), q, None, meta)
        == "proxy-table"
    )
    assert len(ix.hook_calls) == 1 and "plan" in meta.reads
    assert "kv_indices" not in meta.reads  # NVFP4 reads its own step table
    kw = ix.select_calls[0]
    assert kw["kv_indices"] is compact_kv and kw["proxy_plan"] == ("proxy-plan",)


def test_fp8_pure_prefill_hook_gets_the_flattened_page_table(monkeypatch) -> None:
    ix = _hooked_indexer(None)
    meta = _SpyMetadata([200], [100], cache_dtype=torch.float8_e4m3fn)
    q = torch.zeros((200, 4 * 128), dtype=torch.float8_e4m3fn)
    MiniMaxM31MsaSparseAttention.run_indexer(_attention(ix, "fp8"), q, None, meta)
    # Never the segmented proxy plan's page copy: the hook plans its own block table from the flat one.
    assert ix.hook_calls[0][5] is meta.msa_kv_indices
    assert ix.select_calls[0]["kv_indices"] == "proxy-kv-indices"


@pytest.mark.parametrize("case", ["mixed", "decode", "bf16"])
def test_hook_is_not_called_off_pure_prefill_fp8(monkeypatch, case) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_backend import (
        MsaDecodeSpan,
    )

    ix = _hooked_indexer(torch.randn(4, 3, 200))
    span = {
        "mixed": MsaDecodeSpan(row_first=1, query_len=1),
        "decode": MsaDecodeSpan(row_first=0, query_len=1),
    }.get(case)
    qo, prefix = (
        ([200, 1], [100, 300])
        if case == "mixed"
        else ([1], [300])
        if case == "decode"
        else ([200], [100])
    )
    policy = MsaRuntimePolicy()
    cache = torch.bfloat16 if case == "bf16" else torch.float8_e4m3fn
    meta = _SpyMetadata(qo, prefix, span=span, policy=policy, cache_dtype=cache)
    num_tokens = sum(qo)
    q = torch.zeros((num_tokens, 4 * 128), dtype=cache)
    idx_k = torch.zeros((num_tokens, 128), dtype=torch.bfloat16) if case == "bf16" else None
    if case == "bf16":
        meta.msa_write_idx_k = lambda layer_idx, k: None
    monkeypatch.setattr(msa_backend, "_log_decode_reader_taken", lambda *a: None)
    attention = _attention(ix, "bf16" if case == "bf16" else "fp8")
    MiniMaxM31MsaSparseAttention.run_indexer(attention, q, idx_k, meta, idx_k_prewritten=True)
    assert ix.hook_calls == [] and len(ix.select_calls) == 1


def test_the_hook_contract_is_documented_on_the_indexer() -> None:
    import inspect

    src = inspect.getsource(MsaIndexer)
    assert "self.prefill_scorer_hook" in src and "select_prefill_hooked" in src
    assert "self.prefill_scorer_selector" in src
    assert functools.reduce(
        lambda ok, name: ok and name in msa_indexer.__all__,
        (
            "arm_prefill_context_scorer",
            "prefill_context_scorer_accepts",
            "prefill_context_scorer_armed",
            "prewarm_prefill_context_scorer",
        ),
        True,
    )
