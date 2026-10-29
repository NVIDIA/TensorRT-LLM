# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer TRTLLM-Gen FMHA at 128-token pages on SM107 (Rubin).

On SM107 the FallbackFmha (thop) has no paged-context kernels for 128-token pages,
so the MiniMax-M3 KV managers list 128 as a TRTLLM-Gen extra page size and
FlashInferTrtllmGenFmha serves those steps (and only those). These tests run the
real forward at the MiniMax-M3 Eagle3 draft layer shape (16 q : 1 kv head,
head dim 128, BF16 Q / output) with BF16 and FP8 KV and check, for every step:

- FlashInfer served it (run_context / run_generation) and FallbackFmha did not;
- the output is within a tolerance of an FP32 reference built from the pool
  contents (FlashInfer is precision-equivalent to the thop P32 kernels, not
  bitwise, so there is no bit-exact oracle);

over fresh, cached-prefix and chunked contexts (each chunk also matches the
same rows run unchunked and leaves the same K / V), generation (q_len 1),
linear Eagle3 verify (q_len 1 + 3) and mixed context + generation batches.

The masked tail of a request's last page is read by the kernel: NaN bytes there
turn the rows that attend to that page into NaN (measured on SM107 for both this
route and the thop kernels). MiniMax-M3 guards against it by zeroing its KV pools
at allocation (test_minimax_m3_kv_pool_zero_init.py);
the "nan" tail cases check that rows of requests without a tail stay exact.
"""

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha import fallback
from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi_mod
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_ON_SM107 = torch.cuda.is_available() and get_sm_version() == 107
pytestmark = pytest.mark.skipif(
    not _ON_SM107, reason="FlashInfer serves 128-token pages only on SM107 in this route"
)

# The MiniMax-M3 Eagle3 draft attention shape on one rank.
_HEADS, _KV_HEADS, _HEAD_DIM, _PAGE = 16, 1, 128, 128
_REF_RID = 100  # the unchunked run of request r is request _REF_RID + r
_KV_DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
_KV_IDS = ["bf16_kv", "fp8_kv"]

# Worst row's max |out - ref| / max |ref|. "float" is vs the FP32 reference (FP8 KV context
# also quantizes Q to E4M3, which the reference mirrors; FP8 KV generation keeps BF16 Q);
# "unchunked" is a chunk vs the same rows of the unchunked run (same cached K / V): 0 means
# bitwise equal. Each bound is 1.25x the worst case of a seed sweep at this 16:1 shape
# (Hecate 826693: 32 seeds per context case, 16 seeds x {zero, NaN} tails per generation
# case), rounded up. Per-case worst (BF16 / FP8 KV):
#   context rows: fresh 0.0048 / 0.045, cached prefixes 0.0053 / 0.047, chunked 0.0052 / 0.046,
#     mixed batches 0.0061 / 0.049;
#   generation rows: q_len 1 0.0054 / 0.094, verify 1+3 0.0057 / 0.116, mixed 0.0057 / 0.143.
#     FP8 grows with the past (q_len 1: 0.083 at 640 -> 0.143 at 165000); BF16 does not;
#   chunks vs unchunked: 0 for every seed and case.
_TOLERANCE = {
    torch.bfloat16: {"ctx": 7.7e-3, "gen": 7.2e-3, "unchunked": 0.0},
    torch.float8_e4m3fn: {"ctx": 6.2e-2, "gen": 1.8e-1, "unchunked": 0.0},
}
# A kernel that dropped the cached prefix would match attention over the chunk alone; in the
# sweep that error was >= 1.05, at least 4x above every "ctx" bound.
_DROPPED_PREFIX_MARGIN = 4

# (case id, steps): each step is a batch of (request id, cached tokens, new tokens) context rows, run in order.
_CACHED_CONTEXT_CASES = [
    ("fresh_512", [[(0, 0, 512)]]),
    ("prefix_1000_unaligned", [[(0, 0, 1000)], [(0, 1000, 700)]]),
    ("prefix_1024_aligned", [[(0, 0, 1024)], [(0, 1024, 300)]]),
    ("chunked_6096_at_2048", [[(0, 0, 2048)], [(0, 2048, 2048)], [(0, 4096, 2000)]]),
    ("prefix_beside_fresh", [[(0, 0, 384)], [(0, 384, 257), (1, 0, 129)]]),
]

_PASTS = [2047, 16383, 165000, 640]
# (case id, context rows [(cached, new)], generation rows [(past, q_len)]); contexts come first, as in the runtime.
_GENERATION_CASES = [
    ("gen_q1", [], [(p, 1) for p in _PASTS]),
    ("verify_1p3", [], [(p, 4) for p in _PASTS]),
    ("mixed_q1", [(0, 2048), (16384, 2048)], [(640, 1), (16383, 1), (165000, 1)]),
    ("mixed_1p3", [(0, 2048), (16384, 2048)], [(640, 4), (16383, 4), (165000, 4)]),
]


@pytest.fixture
def served(monkeypatch):
    """Count the steps each FMHA library ran."""
    counts = {"fi_ctx": 0, "fi_gen": 0, "fallback": 0}
    for cls, name, key in (
        (fi_mod.FlashInferTrtllmGenFmha, "run_context", "fi_ctx"),
        (fi_mod.FlashInferTrtllmGenFmha, "run_generation", "fi_gen"),
        (fallback.FallbackFmha, "forward", "fallback"),
    ):
        original = getattr(cls, name)

        def wrapped(self, *args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, wrapped)
    return counts


def _attention(kv_dtype: torch.dtype) -> TrtllmAttention:
    assert fi_mod.FlashInferTrtllmGenFmha.is_environment_available(), (
        "SM107 serves 128-token pages only through FlashInfer TRTLLM-Gen FMHA"
    )
    fp8 = kv_dtype == torch.float8_e4m3fn
    return TrtllmAttention(
        layer_idx=0,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        num_kv_heads=_KV_HEADS,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8) if fp8 else None,
    )


def _kv_manager(kv_dtype: torch.dtype, num_tokens: int) -> KVCacheManager:
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=num_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        tokens_per_block=_PAGE,
        max_seq_len=num_tokens,
        max_batch_size=8,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8 if kv_dtype == torch.float8_e4m3fn else DataType.BF16,
    )
    # As MiniMax-M3's KV managers list it: this is what admits 128-token pages to FlashInfer on SM107.
    manager.trtllm_gen_extra_tokens_per_block = frozenset({_PAGE})
    # Zero the pool, as MiniMax-M3's managers do at allocation (see the module docstring).
    _raw_pool(manager).zero_()
    return manager


def _raw_pool(manager) -> torch.Tensor:
    """The layer's pool as bytes: [pages, K/V, kv heads, page tokens, head dim bytes]."""
    return manager.get_buffers(0, kv_layout="HND").view(torch.uint8)


def _token_slots(manager, rid: int, length: int):
    """Pool (page, offset) of tokens 0..length-1 of request rid."""
    pages = manager.get_batch_cache_indices([rid], 0)[0]
    blocks = torch.tensor(pages, device="cuda", dtype=torch.long)
    tokens = torch.arange(length, device="cuda")
    return blocks[tokens // _PAGE], tokens % _PAGE


def _step(attn, manager, rows, qkv: torch.Tensor, num_contexts: int) -> torch.Tensor:
    """One step as the serving runner prepares it. rows = [(request id, cached, new)], contexts first.

    A context row's prompt length is its current chunk, with the prefix in num_cached_tokens_per_seq;
    a generation row's prompt length is its past.
    """
    metadata = TrtllmAttentionMetadata(
        max_num_requests=8,
        max_num_tokens=max(8192, qkv.shape[0] + 8),
        kv_cache_manager=manager,
        runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
        num_contexts=num_contexts,
        seq_lens=torch.tensor([n for _, _, n in rows], dtype=torch.int),
        request_ids=[rid for rid, _, _ in rows],
        prompt_lens=[n if i < num_contexts else c for i, (_, c, n) in enumerate(rows)],
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[c for _, c, _ in rows]
        ),
    )
    assert metadata.tokens_per_block == _PAGE
    metadata.prepare()
    out = attn.forward(qkv, None, None, metadata, attention_mask=PredefinedAttentionMask.CAUSAL)
    return out[0] if isinstance(out, tuple) else out


def _reference_from_pool(
    manager, kv_dtype, rid: int, q: torch.Tensor, cached: int, quantize_q: bool
) -> torch.Tensor:
    """FP32 causal attention of rows cached..cached+n over the K / V the pool holds for tokens 0..cached+n."""
    n = q.shape[0]
    pages, offsets = _token_slots(manager, rid, cached + n)
    kv = _raw_pool(manager)[pages, :, :, offsets, :].view(kv_dtype).float()  # [T, K/V, kv heads, D]
    group = _HEADS // _KV_HEADS
    k = kv[:, 0].repeat_interleave(group, dim=1)  # [T, heads, D]
    v = kv[:, 1].repeat_interleave(group, dim=1)
    if quantize_q:
        q = q.to(torch.float8_e4m3fn)
    q = q.float().view(n, _HEADS, _HEAD_DIM)
    scores = torch.einsum("nhd,thd->nht", q, k) * _HEAD_DIM**-0.5
    q_pos = cached + torch.arange(n, device="cuda")
    k_pos = torch.arange(cached + n, device="cuda")
    scores = scores.masked_fill(k_pos[None, None, :] > q_pos[:, None, None], float("-inf"))
    return torch.einsum("nht,thd->nhd", scores.softmax(-1), v).reshape(n, _HEADS * _HEAD_DIM)


def _float_reference(qkv: torch.Tensor, start: int, n: int, kv_from: int, fp8: bool):
    """Causal attention of rows start..start+n over keys kv_from..start+n of one request's QKV, in FP32."""
    q_size, kv_size = _HEADS * _HEAD_DIM, _KV_HEADS * _HEAD_DIM
    rows = qkv[kv_from : start + n]
    if fp8:
        rows = rows.to(torch.float8_e4m3fn)
    rows = rows.float()
    q = rows[start - kv_from :, :q_size].view(n, _HEADS, _HEAD_DIM).transpose(0, 1)
    k = rows[:, q_size : q_size + kv_size].view(-1, _KV_HEADS, _HEAD_DIM).transpose(0, 1)
    v = rows[:, q_size + kv_size :].view(-1, _KV_HEADS, _HEAD_DIM).transpose(0, 1)
    k = k.repeat_interleave(_HEADS // _KV_HEADS, dim=0)
    v = v.repeat_interleave(_HEADS // _KV_HEADS, dim=0)
    scores = q @ k.transpose(1, 2) / _HEAD_DIM**0.5
    q_pos = torch.arange(start, start + n, device=qkv.device)[:, None]
    k_pos = torch.arange(kv_from, start + n, device=qkv.device)[None, :]
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(0, 1).reshape(n, q_size)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    out, ref = out.float(), ref.float()
    scale = ref.abs().amax(dim=1).clamp_min(1e-6)
    return ((out - ref).abs().amax(dim=1) / scale).max().item()


def _cached_kv(manager, rid: int, length: int) -> torch.Tensor:
    pages, offsets = _token_slots(manager, rid, length)
    return _raw_pool(manager)[pages, :, :, offsets, :]


@pytest.mark.parametrize("kv_dtype", _KV_DTYPES, ids=_KV_IDS)
@pytest.mark.parametrize("case", _CACHED_CONTEXT_CASES, ids=[c[0] for c in _CACHED_CONTEXT_CASES])
def test_cached_context_matches_unchunked(served, case, kv_dtype) -> None:
    """FlashInfer attends to the cached prefix: each chunk's output matches the same rows of the request run
    in one unchunked step and an FP32 reference, and the chunks leave the same K / V in the cache."""
    name, steps = case
    fp8 = kv_dtype == torch.float8_e4m3fn
    tolerance = _TOLERANCE[kv_dtype]
    lengths = {}
    for step in steps:
        for rid, c, n in step:
            assert c == lengths.get(rid, 0), "each chunk starts at its request's cached length"
            lengths[rid] = c + n

    gen = torch.Generator(device="cuda").manual_seed(len(name))
    width = (_HEADS + 2 * _KV_HEADS) * _HEAD_DIM
    qkv = {
        rid: torch.randn(n, width, generator=gen, device="cuda").to(torch.bfloat16)
        for rid, n in lengths.items()
    }
    num_tokens = 2 * sum(-(-n // _PAGE) * _PAGE for n in lengths.values()) + 8 * _PAGE
    manager = _kv_manager(kv_dtype, num_tokens)
    try:
        attn = _attention(kv_dtype)
        rids = sorted(lengths)
        manager.add_dummy_requests(
            rids + [_REF_RID + r for r in rids], [lengths[r] for r in rids] * 2
        )
        # The unchunked run: every request as one fresh context.
        rows = [(_REF_RID + r, 0, lengths[r]) for r in rids]
        out = _step(attn, manager, rows, torch.cat([qkv[r] for r in rids]), len(rows))
        unchunked, first = {}, 0
        for r in rids:
            unchunked[r] = out[first : first + lengths[r]]
            first += lengths[r]
            error = _relative_error(unchunked[r], _float_reference(qkv[r], 0, lengths[r], 0, fp8))
            print(f"\n[{name}] request {r} unchunked vs FP32 reference: {error:.4f}")
            assert error <= tolerance["ctx"]

        for step in steps:
            out = _step(
                attn,
                manager,
                step,
                torch.cat([qkv[r][c : c + n] for r, c, n in step]),
                len(step),
            )
            first = 0
            for r, c, n in step:
                got = out[first : first + n]
                first += n
                assert torch.isfinite(got.float()).all()
                error = _relative_error(got, unchunked[r][c : c + n])
                print(f"[{name}] request {r} rows {c}..{c + n} vs unchunked: {error:.4f}")
                assert error <= tolerance["unchunked"]
                reference = _float_reference(qkv[r], c, n, 0, fp8)
                assert _relative_error(got, reference) <= tolerance["ctx"]
                if c:
                    # A kernel that dropped the cached prefix would match attention over the chunk alone.
                    dropped = _float_reference(qkv[r], c, n, c, fp8)
                    assert _relative_error(got, dropped) > _DROPPED_PREFIX_MARGIN * tolerance["ctx"]
        assert served["fi_ctx"] > 0 and served["fallback"] == 0, served
        torch.cuda.synchronize()
        for r in rids:
            assert torch.equal(
                _cached_kv(manager, r, lengths[r]), _cached_kv(manager, _REF_RID + r, lengths[r])
            ), f"request {r}: the chunks' K / V differ from the unchunked run's"
    finally:
        manager.shutdown()


@pytest.mark.parametrize("tail", ["zero", "nan"])
@pytest.mark.parametrize("kv_dtype", _KV_DTYPES, ids=_KV_IDS)
@pytest.mark.parametrize("case", _GENERATION_CASES, ids=[c[0] for c in _GENERATION_CASES])
def test_generation_verify_and_mixed_steps(served, case, kv_dtype, tail) -> None:
    """Generation (q_len 1), linear verify (1 + 3) and mixed context + generation steps over long cached
    contexts: FlashInfer serves every phase and each row is within tolerance of an FP32 reference from the
    pool contents. With NaN bytes in a last page's unwritten tail, rows of requests whose last page is full
    stay exact; the other requests' NaN rows are what the KV-pool zero-init prevents."""
    name, ctx, gen_rows = case
    fp8 = kv_dtype == torch.float8_e4m3fn
    tolerance = _TOLERANCE[kv_dtype]
    reqs = list(ctx) + list(gen_rows)  # (cached, new) per request
    num_contexts = len(ctx)
    totals = [c + n for c, n in reqs]
    num_tokens = sum(-(-t // _PAGE) * _PAGE for t in totals) + 8 * _PAGE
    manager = _kv_manager(kv_dtype, num_tokens)
    try:
        attn = _attention(kv_dtype)
        rids = list(range(len(reqs)))
        manager.add_dummy_requests(rids, totals)
        raw = _raw_pool(manager)
        gen = torch.Generator(device="cuda").manual_seed(5)
        for rid, (c, n) in zip(rids, reqs):
            if c:
                # Cached K / V written straight into the pool.
                kv = torch.randn(2, c, _KV_HEADS, _HEAD_DIM, generator=gen, device="cuda") * 2
                pages, offsets = _token_slots(manager, rid, c)
                for s in (0, 1):
                    raw[pages, s, :, offsets, :] = kv[s].to(kv_dtype).view(torch.uint8)
            if tail == "nan" and (c + n) % _PAGE:
                last_page = manager.get_batch_cache_indices([rid], 0)[0][(c + n - 1) // _PAGE]
                raw[last_page, :, :, (c + n) % _PAGE :, :] = 0xFF  # NaN in BF16 and E4M3
        width = (_HEADS + 2 * _KV_HEADS) * _HEAD_DIM
        qkv = torch.cat(
            [
                torch.randn(n, width, generator=gen, device="cuda").to(torch.bfloat16)
                for _, n in reqs
            ]
        )
        rows = [(rid, c, n) for rid, (c, n) in zip(rids, reqs)]
        out = _step(attn, manager, rows, qkv, num_contexts)
        torch.cuda.synchronize()

        assert served["fallback"] == 0, served
        assert (served["fi_ctx"] > 0) == (num_contexts > 0), served
        assert served["fi_gen"] > 0, served

        first = 0
        for i, (rid, (c, n)) in enumerate(zip(rids, reqs)):
            got = out[first : first + n].float()
            q = qkv[first : first + n, : _HEADS * _HEAD_DIM]
            first += n
            is_context = i < num_contexts
            has_nan_tail = tail == "nan" and (c + n) % _PAGE != 0
            nonfinite = int((~torch.isfinite(got)).any(dim=1).sum())
            if has_nan_tail:
                print(f"[{name}] request {rid} (tail NaN): {nonfinite}/{n} non-finite rows")
                continue
            assert nonfinite == 0, f"request {rid}: {nonfinite}/{n} non-finite rows"
            ref = _reference_from_pool(manager, kv_dtype, rid, q, c, quantize_q=fp8 and is_context)
            error = _relative_error(got, ref)
            phase = "ctx" if is_context else "gen"
            print(
                f"[{name}] request {rid} ({phase}, past {c}, q_len {n}) vs FP32 reference: {error:.4f}"
            )
            assert error <= tolerance[phase]
    finally:
        manager.shutdown()
