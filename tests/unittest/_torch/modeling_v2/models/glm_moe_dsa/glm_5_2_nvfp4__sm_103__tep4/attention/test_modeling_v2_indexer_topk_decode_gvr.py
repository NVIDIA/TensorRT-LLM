# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode top-k on the stock engine's self-sampling GVR engine, through the target's dispatch.

When the engine's sparse config sets `enable_heuristic_topk` (and the self-sampling GVR
prerequisites hold), the target's generation rows take the stock `TopK` module's GVR decode
(`_dec_topk_gvr_launch`) instead of our one-CTA-per-row kernel. The engine claims an exact
top-k set; here, through `_indexer_decode` with the scorer replaced by prepared logits:

- the set equals `torch.topk`'s on tie-free rows (each causal prefix a permutation of distinct
  integers) at 8k / 64k / 160k / 200k columns, 1..12 rows with next_n 1, 2 and 6, short rows
  (below 2048) and causal ends included; columns past each row's causal end are poisoned, so
  a read past it would be selected;
- every row has the sparse MLA decode's layout: `min(end, 2048)` distinct in-range entries,
  then -1, and a short row lists `0..end-1` in order;
- with ties at the k-th value every column above it is kept with the right count, and with
  `gvr_deterministic_topk` the set is exactly the lower-column tie break;
- a captured launch replays with new logits and longer KV to the eager answer.

The decision (`_dec_topk_gvr_choice`: ours by default, GVR with TRTLLM_GLM52_DEC_TOPK_GVR=1) is
pinned for the config on / off, the env opt-in and each missing prerequisite, from the engine's
real `DeepSeekSparseAttentionConfig`; the dispatch with recorders, including a capture whose
launcher is missing (that graph keeps ours).
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    modeling as M,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_topk_decode import (  # noqa: E402, E501
    select_rows,
)
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation  # noqa: E402
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig  # noqa: E402

TOP_K = 2048
POISON = 1.0e9
#: The engine's split-work scratch: `[rows, 10, top_k]` (dsa/metadata.py), read only by the
#: stock module's fall-through.
MAX_BLOCKS = 10
WIDTHS = (8192, 65536, 163840, 212992)


def _gvr(det: bool = False) -> TopK:
    return TopK(
        TOP_K,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        gvr_self_sampling=True,
        gvr_deterministic=det,
    )


def _lens(width: int, seqs: int, seed: int) -> list[int]:
    """Lengths across the width: the full width, short rows (below top-k), and between."""
    g = torch.Generator().manual_seed(seed)
    out = [width]
    for i in range(1, seqs):
        out.append(int(torch.randint(TOP_K // 2, width + 1, (1,), generator=g)) if i % 3 else 700)
    return out[:seqs]


def _end(lens, next_n: int, r: int) -> int:
    return lens[r // next_n] - next_n + r % next_n + 1


def _build(width: int, lens: list[int], next_n: int, seed: int, levels: int | None = None):
    """Logits as the DSL paged-MQA arena hands them over: a column slice of rows padded to 256
    columns, poisoned past each row's causal end."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = len(lens) * next_n
    stride = (width + 255) // 256 * 256
    arena = torch.full((rows, stride), POISON, dtype=torch.float32, device="cuda")
    for r in range(rows):
        end = _end(lens, next_n, r)
        if levels is None:
            vals = torch.randperm(end, generator=g, device="cuda").float()
        else:
            vals = torch.randint(0, levels, (end,), generator=g, device="cuda").float()
        arena[r, :end] = vals
    return arena[:, :width], torch.tensor(lens, dtype=torch.int32, device="cuda")


class _Logits:
    """Stands in for the paged scorer: returns the prepared logits."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, *a):
        return self.logits

    @staticmethod
    def split_metadata(*a):
        return (None, None, None)


def _md(seq_lens: torch.Tensor):
    rows_max = 64
    return SimpleNamespace(
        kv_lens_cuda_runtime=seq_lens,
        indexer_k_cache_block_offsets=torch.zeros(seq_lens.shape[0], 1, dtype=torch.int32),
        scheduler_metadata_buffer=None,
        radix_aux_indices=torch.empty(
            (rows_max, MAX_BLOCKS, TOP_K), dtype=torch.int32, device="cuda"
        ),
        radix_aux_logits=torch.empty(
            (rows_max, MAX_BLOCKS, TOP_K), dtype=torch.float32, device="cuda"
        ),
    )


class _Ours(str):
    """A stand-in decode top-k handle: compares as its name, and takes the one-CTA launcher."""

    def launcher_for(self, rows: int):
        return 1, None


def _core(width: int, gvr, built=()):
    """A core with only what `_indexer_decode` reads."""
    core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
    core.route_hits, core.route_misses = {}, {}
    core.idx_heads, core.idx_dim, core.idx_topk = 1, 1, TOP_K
    core._max_ctx_len = width
    # Not `Module.__init__`-ed, so a `TopK` cannot be registered as a submodule: set it plainly.
    object.__setattr__(core, "_dec_topk_gvr", gvr)
    core._dec_topk_gvr_live = gvr is not None
    core._ix_topk_dec_compiled = {n: _Ours(f"ours{n}") for n in built}
    return core


def _run(monkeypatch, core, logits, seq_lens, next_n, out=None):
    monkeypatch.setattr(M, "cute_dsl_fp8_paged_mqa_logits", _Logits(logits))
    rows = logits.shape[0]
    out = torch.full((rows, TOP_K), -7, dtype=torch.int32, device="cuda") if out is None else out
    q_fp8 = torch.zeros(rows, 1, 1)
    core._indexer_decode(_md(seq_lens), None, q_fp8, None, out, 0, seq_lens.shape[0], next_n)
    return out


def _check_layout(out, lens, next_n) -> None:
    """The sparse MLA decode's contract: `min(end, 2048)` in-range entries, then -1."""
    for r in range(out.shape[0]):
        end = _end(lens, next_n, r)
        n = min(end, TOP_K)
        row = out[r]
        assert bool(((row[:n] >= 0) & (row[:n] < end)).all()), r
        assert bool((row[n:] == -1).all()), r
        assert row[:n].unique().numel() == n, r
        if end <= TOP_K:
            assert torch.equal(row[:n].cpu(), torch.arange(n, dtype=torch.int32)), r


def _exact(logits, lens, next_n) -> torch.Tensor:
    """`torch.topk` per row over its causal prefix, sorted, -1 padded (tie-free rows)."""
    rows = logits.shape[0]
    out = torch.full((rows, TOP_K), -1, dtype=torch.int32, device="cuda")
    for r in range(rows):
        end = _end(lens, next_n, r)
        n = min(end, TOP_K)
        out[r, :n] = logits[r, :end].topk(n).indices.to(torch.int32)
    return out.sort(dim=1).values


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("next_n, seqs", [(1, 1), (1, 6), (1, 12), (2, 3), (2, 6), (6, 1), (6, 2)])
def test_gvr_exact_set_through_dispatch(monkeypatch, width, next_n, seqs) -> None:
    lens = _lens(width, seqs, seed=width + seqs)
    logits, seq_lens = _build(width, lens, next_n, seed=7 * width + next_n)
    core = _core(width, _gvr())
    out = _run(monkeypatch, core, logits, seq_lens, next_n)
    assert core.route_hits.get("ix_topk_dec_gvr") == 1, core.route_hits
    _check_layout(out, lens, next_n)
    assert torch.equal(out.sort(dim=1).values, _exact(logits, lens, next_n))


@pytest.mark.parametrize("width", (65536, 212992))
def test_gvr_ties_keep_everything_above_the_kth(monkeypatch, width) -> None:
    next_n, lens = 6, _lens(width, 2, seed=3)
    logits, seq_lens = _build(width, lens, next_n, seed=11, levels=64)
    out = _run(monkeypatch, _core(width, _gvr()), logits, seq_lens, next_n)
    _check_layout(out, lens, next_n)
    for r in range(logits.shape[0]):
        end = _end(lens, next_n, r)
        if end <= TOP_K:
            continue
        kth = logits[r, :end].topk(TOP_K).values[-1]
        above = int((logits[r, :end] > kth).sum())
        vals = logits[r, out[r].long()]
        assert int((vals > kth).sum()) == above and bool((vals >= kth).all()), r


@pytest.mark.parametrize("width", (65536, 163840))
def test_gvr_deterministic_breaks_ties_to_the_lower_column(monkeypatch, width) -> None:
    next_n, lens = 6, _lens(width, 2, seed=4)
    logits, seq_lens = _build(width, lens, next_n, seed=19, levels=64)
    out = _run(monkeypatch, _core(width, _gvr(det=True)), logits, seq_lens, next_n)
    _check_layout(out, lens, next_n)
    rows = logits.shape[0]
    starts = torch.zeros(rows, dtype=torch.int64, device="cuda")
    ends = torch.tensor([_end(lens, next_n, r) for r in range(rows)], device="cuda")
    want = select_rows(logits.contiguous(), starts, ends, TOP_K)
    assert torch.equal(out.sort(dim=1).values, want.sort(dim=1).values)
    again = _run(monkeypatch, _core(width, _gvr(det=True)), logits, seq_lens, next_n)
    assert torch.equal(out, again)


def test_gvr_replays_in_a_graph(monkeypatch) -> None:
    """Warm as `_derive_first_step` does, capture, then replay with new logits and longer KV."""
    width, next_n, seqs = 163840, 6, 2
    core = _core(width, _gvr())
    core.mtp_widths = (2, next_n)
    core._warm_dec_topk_gvr(SimpleNamespace(max_num_requests=seqs, max_seq_len=width))
    lens = [100000, 3000]
    logits, seq_lens = _build(width, lens, next_n, seed=13)
    md = _md(seq_lens)
    monkeypatch.setattr(M, "cute_dsl_fp8_paged_mqa_logits", _Logits(logits))
    rows = logits.shape[0]
    out = torch.full((rows, TOP_K), -7, dtype=torch.int32, device="cuda")
    q_fp8 = torch.zeros(rows, 1, 1)

    def run():
        core._indexer_decode(md, None, q_fp8, None, out, 0, seqs, next_n)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    assert core.route_hits.get("ix_topk_dec_gvr") == 1, core.route_hits
    longer = [163840, 2600]
    fresh, fresh_lens = _build(width, longer, next_n, seed=17)
    logits.copy_(fresh)
    seq_lens.copy_(fresh_lens)
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    _check_layout(out, longer, next_n)
    assert torch.equal(out.sort(dim=1).values, _exact(fresh, longer, next_n))
    eager = _run(monkeypatch, _core(width, _gvr()), fresh, fresh_lens, next_n)
    assert torch.equal(out.sort(dim=1).values, eager.sort(dim=1).values)


def _params(**kw):
    """The engine's own sparse params for GLM-5.2's MTP layer index (`_engine_sparse_params`)."""
    cfg = SimpleNamespace(
        num_hidden_layers=78, index_n_heads=32, index_head_dim=128, index_topk=TOP_K
    )
    mc = SimpleNamespace(
        sparse_attention_config=DeepSeekSparseAttentionConfig(**kw), pretrained_config=cfg
    )
    return M._engine_sparse_params(mc, 0)


_ON = dict(enable_heuristic_topk=True, use_cute_dsl_topk=True, use_cute_dsl_paged_mqa_logits=True)


@pytest.mark.parametrize(
    "cfg, topk, sm, dsl, env, want",
    [
        # The GVR sparse config: ours (the cluster split on long engines) unless the env asks for GVR.
        (_ON, TOP_K, 103, True, None, None),
        (_ON, TOP_K, 103, True, "0", None),
        # TRTLLM_GLM52_DEC_TOPK_GVR=1: GVR, not deterministic / deterministic.
        (_ON, TOP_K, 103, True, "1", False),
        (dict(_ON, gvr_deterministic_topk=True), TOP_K, 103, True, "1", True),
        (dict(_ON, use_cute_dsl_topk=False), TOP_K, 100, True, "1", False),
        # Config off: ours, whatever the env.
        ({}, TOP_K, 103, True, "1", None),
        (dict(use_cute_dsl_topk=True), TOP_K, 103, True, "1", None),
        # A missing prerequisite: the temporal engine, no CuTe DSL, another arch, another k.
        (dict(_ON, use_self_sampling_topk=False), TOP_K, 103, True, "1", None),
        (_ON, TOP_K, 103, False, "1", None),
        (_ON, TOP_K, 90, True, "1", None),
        (_ON, TOP_K, 120, True, "1", None),
        (_ON, 1024, 103, True, "1", None),
    ],
)
def test_gvr_choice(cfg, topk, sm, dsl, env, want) -> None:
    got = M._dec_topk_gvr_choice(_params(**cfg), topk, sm_version=sm, dsl_available=dsl, env=env)
    assert got is want


def test_gvr_choice_without_sparse_config() -> None:
    mc = SimpleNamespace(sparse_attention_config=None, pretrained_config=SimpleNamespace())
    params = M._engine_sparse_params(mc, 0)
    assert params is None
    assert (
        M._dec_topk_gvr_choice(params, TOP_K, sm_version=103, dsl_available=True, env=None) is None
    )


class _Gvr:
    """Records GVR launches; raises like a launcher missing under capture when asked."""

    def __init__(self, calls, raise_missing=False):
        self.calls, self.raise_missing = calls, raise_missing

    def __call__(self, *a, **k):
        if self.raise_missing:
            raise RuntimeError("varlen launcher not compiled for this shape")
        self.calls.append(("gvr", k["next_n"], k["max_seq_len"]))


@pytest.mark.parametrize(
    "gvr, capturing, gen_p, want, counted",
    [
        ("on", False, 6, [("gvr", 6, 163840)], True),
        ("on", False, 1, [("gvr", 1, 163840)], True),
        ("off", False, 6, ["ours6"], False),
        # A launcher missing inside a capture: that graph keeps ours.
        ("missing", True, 6, ["ours6"], False),
    ],
)
def test_indexer_decode_dispatch(monkeypatch, gvr, capturing, gen_p, want, counted) -> None:
    calls = []
    monkeypatch.setattr(M, "cute_dsl_fp8_paged_mqa_logits", _Logits("logits"))
    monkeypatch.setattr(M, "indexer_topk_decode", lambda *a, **k: calls.append("stock"))
    monkeypatch.setattr(
        M, "cute_dsl_indexer_topk_decode", lambda *a, compiled, **k: calls.append(compiled)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    topk = None if gvr == "off" else _Gvr(calls, raise_missing=gvr == "missing")
    core = _core(163840, topk, built=(1, 2, 6))
    seqs = 2
    md = SimpleNamespace(
        kv_lens_cuda_runtime=torch.zeros(seqs, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.zeros(seqs, 1, dtype=torch.int32),
        scheduler_metadata_buffer=None,
        radix_aux_indices=None,
        radix_aux_logits=None,
    )
    q_fp8 = torch.zeros(seqs * gen_p, 1, 1)
    core._indexer_decode(md, None, q_fp8, None, None, 0, seqs, gen_p)
    assert calls == want, calls
    assert core.route_hits.get("ix_topk_dec_gvr", 0) == int(counted)
    assert core.route_hits.get("ix_topk_dec", 0) == int(not counted)


@pytest.mark.parametrize(
    "width, env, want",
    [
        # The realtext / accuracy engines (16384 / 8192 columns) keep ours; long engines take GVR.
        (8192, None, False),
        (16384, None, False),
        (16416, None, True),
        (65536, None, True),
        (172032, None, True),
        (212992, None, True),
        # TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE moves the bound.
        (16384, "8192", True),
        (172032, "200000", False),
    ],
)
def test_gvr_serves_by_width(monkeypatch, width, env, want) -> None:
    if env is None:
        monkeypatch.delenv("TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE", env)
    assert M._dec_topk_gvr_serves(width) is want


def test_config_on_narrow_engine_keeps_ours(monkeypatch) -> None:
    """GVR configured but not live (a width at or below the bound): ours, counted as ours."""
    calls = []
    monkeypatch.setattr(M, "cute_dsl_fp8_paged_mqa_logits", _Logits("logits"))
    monkeypatch.setattr(
        M, "cute_dsl_indexer_topk_decode", lambda *a, compiled, **k: calls.append(compiled)
    )
    core = _core(16384, _Gvr(calls), built=(6,))
    core._dec_topk_gvr_live = False
    md = SimpleNamespace(
        kv_lens_cuda_runtime=torch.zeros(2, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        scheduler_metadata_buffer=None,
        radix_aux_indices=None,
        radix_aux_logits=None,
    )
    core._indexer_decode(md, None, torch.zeros(12, 1, 1), None, None, 0, 2, 6)
    assert calls == ["ours6"]
    assert core.route_hits.get("ix_topk_dec") == 1 and "ix_topk_dec_gvr" not in core.route_hits


def test_missing_launcher_outside_capture_raises(monkeypatch) -> None:
    monkeypatch.setattr(M, "cute_dsl_fp8_paged_mqa_logits", _Logits("logits"))
    core = _core(163840, _Gvr([], raise_missing=True), built=(6,))
    md = SimpleNamespace(
        kv_lens_cuda_runtime=torch.zeros(1, dtype=torch.int32),
        indexer_k_cache_block_offsets=torch.zeros(1, 1, dtype=torch.int32),
        scheduler_metadata_buffer=None,
        radix_aux_indices=None,
        radix_aux_logits=None,
    )
    with pytest.raises(RuntimeError, match="not compiled"):
        core._indexer_decode(md, None, torch.zeros(6, 1, 1), None, None, 0, 1, 6)


def test_warm_covers_every_decode_row_count(monkeypatch) -> None:
    from tensorrt_llm._torch.kernels.blackwell.top_k import (
        gvr_topk_decode_self_sampling_host as gvr_host,
    )

    seen = []
    monkeypatch.setattr(gvr_host, "warmup_varlen", lambda k, msl, **kw: seen.append((k, msl, kw)))
    core = _core(172032, _gvr(det=True))
    core.mtp_widths = (2, 6)
    core._warm_dec_topk_gvr(SimpleNamespace(max_num_requests=3, max_seq_len=172000))
    assert [s[2]["next_n"] for s in seen] == [1, 2, 6]
    for k, msl, kw in seen:
        assert (k, msl) == (TOP_K, 172000)
        assert kw["row_stride"] == 172032 and kw["deterministic"] is True
        assert kw["num_rows_list"] == tuple(kw["next_n"] * b for b in (1, 2, 3))
