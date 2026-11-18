# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``fmk_context_fmha`` / ``fmk_pure_context_fmha``: the context rows of mixed steps and of
pure context steps on the multi-instance context FMHA.

Claims checked here:

* ``plan_context_fmha`` covers every key tile of every (request, 128-row block, head group)
  exactly once, cuts an item only before its first diagonal tile, writes the merge table of
  every cut item, and never uses more than ``max_slots`` workspace slots (CPU).
* At the tep4 rank shape (32 query / 4 KV heads, head dim 64, FP8 KV in 32-token pages of a
  ``KVCacheManagerV2``, pages scattered), on a mixed step of context requests and 9..12
  generation requests (one short request, several requests, a short chunk over a deep
  prefix, odd lengths, long-context mixes with up to 397k cached keys), with
  ``FmkContextPhaseFmha`` in place of ``ThopContextPhaseFmha``:

  - the context rows ran the multi-instance kernel (not the thop op);
  - every K / V byte of the cache after the step, and every generation row, is bitwise the
    thop op's (``ThopContextPhaseFmha`` on the same inputs);
  - the context rows are finite and their relative L2 error against an fp32 reference
    (E4M3 Q, the cached E4M3 K / V) is within 15 % of the thop op's own error.

* Under a stream capture the library declines (the thop op serves the call).
* Under ``MetaInitMode`` (the loader's model construction) the install-time setup still
  reserves a workspace with storage and runs its warm-up call.
* The kernel's launch on a fixed plan, captured in a CUDA graph and replayed with new
  inputs, writes the eager launch's bytes.
* Pure context steps (``FmkPureContextFmha`` just before the thop fallback, at the tep4 rank
  shape, on the context-server shapes: one 8k chunk at depth 0 and 61.7k, two and three
  requests up to 598k cached keys, odd lengths):

  - the manager selects it alone and the context rows ran the multi-instance kernel;
  - every K / V byte of the cache after the step is bitwise the thop op's (the fallback on
    the whole batch, the call before this path);
  - the rows are finite and their relative L2 error against an fp32 reference is within
    15 % of the thop op's own error;
  - a declined call runs the fallback on the manager's arguments: output and cache bitwise
    the thop op's;
  - with projection rows past the step's (a replayed breakable prefill graph's bucket) or a
    ``context_only`` input, it still runs and writes the unpadded mixed call's bytes;
  - a mixed step does not reach it (``fmk_context_fmha``'s library serves it);
  - ``install_pure_context`` puts it just before the fallback of the full-attention layers
    only, idempotently, sharing ``fmk_context_fmha``'s workspace and plan when installed.
"""

import math
import types

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.attention import (  # noqa: E501
    DecodeTrtllmGenFmha,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.fmk_context_fmha import (  # noqa: E501
    attention as fmkc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.mixed_step_gen_rows import (  # noqa: E501
    attention as mixed,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.attention import (  # noqa: E501
    context_fmha_fmk as fmk,
)
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

H, G, D, P = 32, 4, 64, 32
DEV = torch.device("cuda")


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


needs_sm107 = pytest.mark.skipif(_sm() != fmkc.SERVED_SM, reason="the kernel is written for sm107")


# --------------------------------------------------------------------------- planner


PLAN_MIXES = [
    [(166000, 8192)],
    [(560000, 8192)],
    [(0, 300)],
    [(5306, 1)],
    [(5007, 300), (777, 1), (33019, 4097)],
    [(63, 129), (64, 64), (1000, 255), (12345, 1000), (0, 1)],
    # long-context mixed steps with many generation requests
    [(128, 2976), (81777, 1292), (336256, 1937), (128, 1632)],
    [(160, 6016), (397518, 1870)],
    [(94141, 2969), (2272, 4924)],
    [(16768, 128), (474435, 2255), (128, 5471)],
    [(8064, 7936)],
    [(28064, 586)],
]


@pytest.mark.parametrize("mix", PLAN_MIXES, ids=lambda m: f"{len(m)}req-{m[0][0]}x{m[0][1]}")
@pytest.mark.parametrize("split", [True, False])
def test_planner_covers_every_tile_once(mix, split):
    q_lens = [q for _, q in mix]
    kv_lens = [c + q for c, q in mix]
    plan = fmk.plan_context_fmha(
        q_lens, kv_lens, num_ctas=212, split=split, device=torch.device("cpu")
    )
    items = fmk._items(q_lens, kv_lens, H, G, 4, 64)
    units = plan.units[: plan.num_units].tolist()
    cover = {}
    for b, m, hg, t0, t1, mode, slot, _ in units:
        cover.setdefault((b, m, hg), []).append((t0, t1, mode, slot))
    assert len(cover) == len(items)
    cut = 0
    for b, m, hg, t_end, first_masked in items:
        segs = sorted(cover[(b, m, hg)])
        assert segs[0][0] == 0 and segs[-1][1] == t_end
        assert all(x[1] == y[0] for x, y in zip(segs, segs[1:]))
        if len(segs) == 1:
            assert segs[0][2] == 0
            continue
        cut += 1
        assert all(s[2] == 1 for s in segs)
        assert all(s[1] <= first_masked for s in segs[:-1])
    off = plan.cta_off.tolist()
    assert off[0] == 0 and off[-1] == plan.num_units and off == sorted(off)
    assert (plan.merge_meta is None and cut == 0) or plan.merge_meta.shape[0] == cut
    assert plan.num_slots <= fmk.max_slots(212)
    if plan.merge_slots is not None:
        slots = plan.merge_slots.flatten().tolist()
        used = sorted(s for s in slots if s >= 0)
        assert used == list(range(plan.num_slots))


# --------------------------------------------------------------------------- mixed steps on sm107


def _fill_cache(mgr, request_ids, cached_k, cached_v):
    """Write each request's cached prefix (``[c, G, D]`` E4M3) into its pages (HND)."""
    buf = mgr.get_buffers(0, kv_layout="HND")  # [pages, 2, G, P, D]
    raw = buf.view(torch.uint8)
    raw.zero_()
    blocks = mgr.get_batch_cache_indices(list(request_ids), 0)
    for i, ids in enumerate(blocks):
        c = cached_k[i].shape[0]
        if c == 0:
            continue
        n = -(-c // P)
        pages = torch.tensor([b for b in ids if b != -1][:n], dtype=torch.long, device=DEV)
        for kv, src in ((0, cached_k[i]), (1, cached_v[i])):
            pad = torch.zeros(n * P, G, D, dtype=torch.uint8, device=DEV)
            pad[:c] = src.view(torch.uint8)
            raw[pages, kv] = pad.view(n, P, G, D).transpose(1, 2)
    return buf


def _cache_tokens(buf, page_ids, kv: int, n_pages: int):
    """``[n_pages * P, G, D]`` E4M3 tokens of one request's K (0) or V (1) pages."""
    pages = torch.tensor(page_ids[:n_pages], dtype=torch.long, device=DEV)
    raw = buf.view(torch.uint8)[pages, kv]  # [n, G, P, D]
    return raw.transpose(1, 2).reshape(n_pages * P, G, D).view(torch.float8_e4m3fn)


def _reference(q_e4m3, k_all, v_all, rows, heads, kv_len, q_len):
    """fp32 attention of the sampled rows / heads of one request (bottom-right causal)."""
    out = torch.empty(len(rows), len(heads), D, dtype=torch.float32, device=DEV)
    off = kv_len - q_len
    k = k_all[:kv_len].float()
    v = v_all[:kv_len].float()
    for a, r in enumerate(rows):
        lim = off + r + 1
        for b_, h in enumerate(heads):
            s = (k[:lim, h // (H // G)] @ q_e4m3[r, h].float()) / math.sqrt(D)
            p = torch.softmax(s, dim=0)
            out[a, b_] = p @ v[:lim, h // (H // G)]
    return out


class _Step:
    """One mixed step on a real TrtllmAttention and KVCacheManagerV2 (layer 0)."""

    def __init__(self, contexts, gens, seed):
        self.contexts = contexts
        self.seq_lens = [q for _, q in contexts] + [1] * len(gens)
        self.cached = [c for c, _ in contexts] + list(gens)
        self.num_contexts = len(contexts)
        self.token_nums = [c + s for c, s in zip(self.cached, self.seq_lens)]
        num_seqs = len(self.seq_lens)
        self.request_ids = list(range(num_seqs))
        pages_per_seq = max(-(-t // P) for t in self.token_nums)
        self.mgr = KVCacheManagerV2(
            KvCacheConfig(max_tokens=num_seqs * pages_per_seq * P),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=G,
            head_dim=D,
            tokens_per_block=P,
            max_seq_len=pages_per_seq * P,
            max_batch_size=num_seqs,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=torch_dtype_to_binding(torch.float8_e4m3fn),
        )
        self.mgr.add_dummy_requests(self.request_ids, self.token_nums)
        gen = torch.Generator(device=DEV).manual_seed(seed)
        self.cached_k = [
            torch.randn(c, G, D, generator=gen, device=DEV).to(torch.float8_e4m3fn)
            for c in self.cached
        ]
        self.cached_v = [
            torch.randn(c, G, D, generator=gen, device=DEV).to(torch.float8_e4m3fn)
            for c in self.cached
        ]
        tokens = sum(self.seq_lens)
        self.qkv = torch.randn(tokens, (H + 2 * G) * D, generator=gen, device=DEV).to(
            torch.bfloat16
        )
        self.buf = _fill_cache(self.mgr, self.request_ids, self.cached_k, self.cached_v)
        self.attn = create_attention(
            "TRTLLM",
            layer_idx=0,
            num_heads=H,
            head_dim=D,
            num_kv_heads=G,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
            q_scaling=1.0,
        )
        libs = self.attn._fmha_manager.fmha_libs
        libs.insert(0, DecodeTrtllmGenFmha(self.attn))
        index = next(i for i, f in enumerate(libs) if isinstance(f, FallbackFmha))
        self.thop_lib = mixed.ThopContextPhaseFmha(self.attn, libs[index])
        libs.insert(index, self.thop_lib)
        self.lib_index = index
        num_ctas = torch.cuda.get_device_properties(DEV).multi_processor_count
        workspace = fmk.FmkWorkspace()
        workspace.reserve(fmk.max_slots(num_ctas), fmkc.NUM_INST, DEV)
        self.fmk_lib = fmkc.FmkContextPhaseFmha(
            self.attn, libs[index + 1], fmkc.StepPlans(num_ctas), workspace
        )

    def metadata(self):
        md = get_attention_backend("TRTLLM").Metadata(
            num_contexts=self.num_contexts,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=self.cached),
            seq_lens=torch.tensor(self.seq_lens, dtype=torch.int),
            max_num_requests=len(self.seq_lens),
            max_num_tokens=max(8192, sum(self.seq_lens)),
            kv_cache_manager=self.mgr,
            request_ids=self.request_ids,
            prompt_lens=self.seq_lens[: self.num_contexts] + self.token_nums[self.num_contexts :],
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        md.prepare()
        return md

    def run(self, lib):
        libs = self.attn._fmha_manager.fmha_libs
        libs[self.lib_index] = lib
        self.attn._fmha_manager._cache.clear()
        md = self.metadata()
        assert mixed.takes_generation(md)
        out = self.attn.forward(
            self.qkv.clone(),
            None,
            None,
            md,
            forward_args=AttentionForwardArgs(
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_input_type=AttentionInputType.mixed,
            ),
        )
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        return out[: sum(self.seq_lens)].clone()

    def cache_bytes(self):
        return self.buf.view(torch.uint8).clone()

    def close(self):
        self.mgr.shutdown()


STEP_CASES = {
    "short": ([(0, 300)], [1500] * 9),
    "three": ([(5007, 300), (777, 1), (33019, 4097)], [2000] * 9),
    "deep_short_q": ([(100000, 1024)], [3000] * 10),
    "odd": ([(63, 129), (64, 64), (1000, 255), (12345, 1000)], [777] * 12),
    "served_mix_a": ([(128, 2976), (81777, 1292), (336256, 1937), (128, 1632)], [5000] * 9),
    "served_mix_b": ([(160, 6016), (397518, 1870)], [4000] * 9),
}


@needs_sm107
@pytest.mark.parametrize("name", list(STEP_CASES))
def test_mixed_step_context_rows(name):
    contexts, gens = STEP_CASES[name]
    step = _Step(contexts, gens, seed=7 + len(name))
    try:
        ctl = step.run(step.thop_lib)
        ctl_cache = step.cache_bytes()
        got = step.run(step.fmk_lib)
        got_cache = step.cache_bytes()
        assert step.fmk_lib.fmk_calls == 1 and step.fmk_lib.declined_calls == 0
        assert torch.equal(ctl_cache, got_cache), "K / V appended differently"
        ctx_rows = sum(q for _, q in contexts)
        assert torch.equal(ctl[ctx_rows:], got[ctx_rows:]), "generation rows differ"
        assert bool(torch.isfinite(got[:ctx_rows]).all())
        # fp32 reference of sampled rows of every request from the cache's own K / V
        blocks = step.mgr.get_batch_cache_indices(step.request_ids, 0)
        q_e4m3 = step.qkv[:ctx_rows, : H * D].view(ctx_rows, H, D).to(torch.float8_e4m3fn)
        heads = [0, 7, 8, 31]
        e_ctl = e_got = r2 = 0.0
        row0 = 0
        for b, (c, q) in enumerate(contexts):
            kv_len = c + q
            n = -(-kv_len // P)
            ids = [x for x in blocks[b] if x != -1]
            k_all = _cache_tokens(step.buf, ids, 0, n)
            v_all = _cache_tokens(step.buf, ids, 1, n)
            rows = sorted({0, q // 3, q // 2, q - 1, max(0, q - 2)})
            ref = _reference(q_e4m3[row0 : row0 + q], k_all, v_all, rows, heads, kv_len, q)
            idx = torch.tensor(rows, device=DEV) + row0
            a = ctl[idx].view(len(rows), H, D)[:, heads].float()
            g = got[idx].view(len(rows), H, D)[:, heads].float()
            e_ctl += ((a - ref) ** 2).sum().item()
            e_got += ((g - ref) ** 2).sum().item()
            r2 += (ref**2).sum().item()
            row0 += q
        rel_ctl, rel_got = math.sqrt(e_ctl / r2), math.sqrt(e_got / r2)
        diff = (got[:ctx_rows].float() - ctl[:ctx_rows].float()).abs()
        print(
            f"{name}: rel L2 vs fp32 thop {rel_ctl:.4e} fmk {rel_got:.4e}; "
            f"|fmk - thop| max {diff.max().item():.3e} mean {diff.mean().item():.3e}"
        )
        assert rel_got <= 1.15 * rel_ctl + 1e-3, (rel_got, rel_ctl)
    finally:
        step.close()


@needs_sm107
def test_declines_under_capture():
    step = _Step([(0, 300)], [1500] * 9, seed=3)
    try:
        md = step.metadata()
        params = type("P", (), {})()  # only the capture check is reached
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            reason = step.fmk_lib._decline(params, ((300,), (300,)))
        assert reason == "a stream capture is in progress"
        del md
    finally:
        step.close()


@needs_sm107
def test_prepare_under_meta_init():
    num_ctas = torch.cuda.get_device_properties(DEV).multi_processor_count
    with MetaInitMode():
        assert torch.empty(1, device=DEV).is_meta
        workspace = fmkc._prepare(DEV, H, G, num_ctas)
    torch.cuda.synchronize()
    assert not workspace.part_o.is_meta and not workspace.part_ml.is_meta
    assert workspace.nbytes > 0


@needs_sm107
def test_graph_replay_matches_eager():
    gen = torch.Generator(device=DEV).manual_seed(11)
    mix = [(5007, 300), (777, 1), (33019, 4097)]
    q_lens = [q for _, q in mix]
    kv_lens = [c + q for c, q in mix]
    pages = [-(-k // P) for k in kv_lens]
    total = sum(pages)
    pool = torch.randn(2 * total, G, P, D, generator=gen, device=DEV).to(torch.float8_e4m3fn)
    perm = torch.randperm(total, generator=gen, device=DEV).to(torch.int32)
    width = max(pages)
    table = torch.zeros(len(mix), 2, width, dtype=torch.int32, device=DEV)
    base = 0
    for b, n in enumerate(pages):
        table[b, 0, :n] = 2 * perm[base : base + n]
        table[b, 1, :n] = 2 * perm[base : base + n] + 1
        base += n
    cu_q = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV
    )
    kvl = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    scales = torch.tensor([math.log2(math.e) / math.sqrt(D), 1.0], dtype=torch.float32, device=DEV)
    num_ctas = torch.cuda.get_device_properties(DEV).multi_processor_count
    plan = fmk.plan_context_fmha(q_lens, kv_lens, num_ctas=num_ctas, device=DEV)
    assert plan.merge_meta is not None
    ws = fmk.FmkWorkspace()
    ws.reserve(fmk.max_slots(num_ctas), 4, DEV)
    q = torch.empty(sum(q_lens), H, D, dtype=torch.float8_e4m3fn, device=DEV)
    out = torch.empty(sum(q_lens), H, D, dtype=torch.bfloat16, device=DEV)

    def launch():
        fmk.paged_context_fmha_fmk(
            q, pool, table, cu_q, kvl, plan, scales[0:1], scales[1:2], out, workspace=ws
        )

    q.copy_(torch.randn(q.shape, generator=gen, device=DEV).to(torch.float8_e4m3fn))
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(2):
        q.copy_(torch.randn(q.shape, generator=gen, device=DEV).to(torch.float8_e4m3fn))
        out.fill_(float("nan"))
        launch()
        eager = out.clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


# --------------------------------------------------------------------------- pure context steps


def test_takes_pure_context():
    md = types.SimpleNamespace
    assert fmkc.takes_pure_context(md(num_contexts=2, num_generations=0, beam_width=1))
    assert not fmkc.takes_pure_context(md(num_contexts=2, num_generations=9, beam_width=1))
    assert not fmkc.takes_pure_context(md(num_contexts=0, num_generations=0, beam_width=1))
    assert not fmkc.takes_pure_context(md(num_contexts=1, num_generations=0, beam_width=2))


class _PureStep(_Step):
    """A pure context step (no generation request) on the libraries a context server's
    full-attention layer holds, plus an ``FmkPureContextFmha`` placed by ``run``."""

    def __init__(self, contexts, seed):
        super().__init__(contexts, [], seed)
        self.fallback = self.attn._fmha_manager.fmha_libs[self.lib_index + 1]
        assert isinstance(self.fallback, FallbackFmha)
        self.pure_lib = fmkc.FmkPureContextFmha(
            self.attn, self.fallback, self.fmk_lib._plans, self.fmk_lib._workspace
        )

    def run(self, lib, pad_rows=0, input_type=AttentionInputType.mixed):
        """The step with ``lib`` just before the fallback, or with none (the thop op).

        ``pad_rows`` rows of noise follow the step's rows in the projection, as in a replayed
        breakable prefill graph's bucket."""
        libs = self.attn._fmha_manager.fmha_libs
        libs[:] = [f for f in libs if not isinstance(f, fmkc.FmkPureContextFmha)]
        if lib is not None:
            libs.insert(libs.index(self.fallback), lib)
        self.attn._fmha_manager._cache.clear()
        md = self.metadata()
        assert fmkc.takes_pure_context(md)
        qkv = self.qkv.clone()
        if pad_rows:
            noise = torch.randn(pad_rows, qkv.shape[1], device=DEV).to(qkv.dtype)
            qkv = torch.cat([qkv, noise])
        out = self.attn.forward(
            qkv,
            None,
            None,
            md,
            forward_args=AttentionForwardArgs(
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_input_type=input_type,
            ),
        )
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        return out[: sum(self.seq_lens)].clone()


PURE_CASES = {
    "single_shallow": [(0, 8192)],
    "single_deep": [(61700, 8192)],
    "two": [(35232, 5208), (128, 2976)],
    "three_deep": [(15872, 6240), (83416, 970), (598262, 970)],
    "odd": [(63, 129), (64, 64), (1000, 255), (12345, 1000)],
}


@needs_sm107
@pytest.mark.parametrize("name", list(PURE_CASES))
def test_pure_context_step_rows(name):
    contexts = PURE_CASES[name]
    step = _PureStep(contexts, seed=17 + len(name))
    try:
        ctl = step.run(None)
        ctl_cache = step.cache_bytes()
        got = step.run(step.pure_lib)
        got_cache = step.cache_bytes()
        assert step.pure_lib.fmk_calls == 1 and step.pure_lib.declined_calls == 0
        assert step.thop_lib.context_calls == 0 and step.fmk_lib.fmk_calls == 0
        assert torch.equal(ctl_cache, got_cache), "K / V appended differently"
        rows = sum(q for _, q in contexts)
        assert bool(torch.isfinite(got).all())
        blocks = step.mgr.get_batch_cache_indices(step.request_ids, 0)
        q_e4m3 = step.qkv[:rows, : H * D].view(rows, H, D).to(torch.float8_e4m3fn)
        heads = [0, 7, 8, 31]
        e_ctl = e_got = r2 = 0.0
        row0 = 0
        for b, (c, q) in enumerate(contexts):
            kv_len = c + q
            n = -(-kv_len // P)
            ids = [x for x in blocks[b] if x != -1]
            k_all = _cache_tokens(step.buf, ids, 0, n)
            v_all = _cache_tokens(step.buf, ids, 1, n)
            sampled = sorted({0, q // 3, q // 2, q - 1, max(0, q - 2)})
            ref = _reference(q_e4m3[row0 : row0 + q], k_all, v_all, sampled, heads, kv_len, q)
            idx = torch.tensor(sampled, device=DEV) + row0
            a = ctl[idx].view(len(sampled), H, D)[:, heads].float()
            g = got[idx].view(len(sampled), H, D)[:, heads].float()
            e_ctl += ((a - ref) ** 2).sum().item()
            e_got += ((g - ref) ** 2).sum().item()
            r2 += (ref**2).sum().item()
            row0 += q
        rel_ctl, rel_got = math.sqrt(e_ctl / r2), math.sqrt(e_got / r2)
        diff = (got.float() - ctl.float()).abs()
        print(
            f"{name}: rel L2 vs fp32 thop {rel_ctl:.4e} fmk {rel_got:.4e}; "
            f"|fmk - thop| max {diff.max().item():.3e} mean {diff.mean().item():.3e}"
        )
        assert rel_got <= 1.15 * rel_ctl + 1e-3, (rel_got, rel_ctl)
    finally:
        step.close()


@needs_sm107
def test_pure_context_decline_runs_the_old_call(monkeypatch):
    step = _PureStep([(5007, 300), (33019, 4097)], seed=5)
    try:
        ctl = step.run(None)
        ctl_cache = step.cache_bytes()
        monkeypatch.setattr(step.pure_lib, "_decline", lambda params, lengths: "forced")
        got = step.run(step.pure_lib)
        assert step.pure_lib.declined_calls == 1 and step.pure_lib.fmk_calls == 0
        assert step.pure_lib.context_calls == 1
        assert torch.equal(ctl, got)
        assert torch.equal(ctl_cache, step.cache_bytes())
    finally:
        step.close()


@needs_sm107
def test_a_mixed_step_does_not_reach_the_pure_context_library():
    step = _Step([(5007, 300), (33019, 4097)], [2000] * 9, seed=9)
    try:
        libs = step.attn._fmha_manager.fmha_libs
        fallback = libs[step.lib_index + 1]
        pure = fmkc.FmkPureContextFmha(
            step.attn, fallback, step.fmk_lib._plans, step.fmk_lib._workspace
        )
        libs.insert(step.lib_index + 1, pure)
        step.run(step.fmk_lib)
        assert step.fmk_lib.fmk_calls == 1
        assert pure.context_calls == 0 and pure.fmk_calls == 0
    finally:
        step.close()


def _model_of(*attentions):
    layers = [types.SimpleNamespace(self_attn=a) for a in attentions]
    return types.SimpleNamespace(layers=layers)


@needs_sm107
def test_install_pure_context():
    step = _PureStep([(0, 300)], seed=13)
    try:
        full = types.SimpleNamespace(is_sliding=False, layer_idx=0, attn=step.attn)
        sliding = types.SimpleNamespace(is_sliding=True, layer_idx=1, attn=None)
        model = _model_of(full, sliding)
        with pytest.raises(ValueError, match="written for sm107"):
            fmkc.install_pure_context(model, (10, 3))
        libs = step.attn._fmha_manager.fmha_libs
        libs[step.lib_index] = step.fmk_lib  # fmk_context_fmha installed on this layer
        assert fmkc.install_pure_context(model, (10, 7)) == 1
        pure = fmkc.pure_context_library(step.attn)
        assert pure is not None and libs.index(pure) == libs.index(step.fallback) - 1
        assert pure._workspace is step.fmk_lib._workspace and pure._plans is step.fmk_lib._plans
        assert fmkc.context_library(step.attn) is step.fmk_lib
        before = list(libs)
        assert fmkc.install_pure_context(model, (10, 7)) == 1
        assert libs == before
        got = step.run(pure)
        assert pure.fmk_calls == 1 and bool(torch.isfinite(got).all())
    finally:
        step.close()


@needs_sm107
@pytest.mark.parametrize(
    "pad_rows, input_type",
    [
        (0, AttentionInputType.context_only),
        (61, AttentionInputType.mixed),
        (61, AttentionInputType.context_only),
    ],
    ids=["context_only", "padded_mixed", "padded_context_only"],
)
def test_pure_context_step_padded_rows_and_context_only(pad_rows, input_type):
    """A context server's step as a replayed breakable prefill graph hands it (projection rows
    past the step's, either input type the library takes): the multi-instance kernel serves it
    with no decline, and the step's rows and the K / V cache are bitwise those of the unpadded
    mixed call."""
    step = _PureStep([(35232, 5208), (128, 2976)], seed=29)
    try:
        ref = step.run(step.pure_lib)
        ref_cache = step.cache_bytes()
        got = step.run(step.pure_lib, pad_rows=pad_rows, input_type=input_type)
        assert step.pure_lib.fmk_calls == 2 and step.pure_lib.declined_calls == 0
        assert torch.equal(ref, got)
        assert torch.equal(ref_cache, step.cache_bytes())
    finally:
        step.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
