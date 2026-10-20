# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FallbackFmha's refusal of paged prefill at a KV manager's TRTLLM-Gen extra page sizes: kept on every SM but SM107,
where no TRTLLM-Gen FMHA exists and the thop paged-context kernel serves those pages (its kernel presence is
test_context_fmha_kernel_presence.py). The dispatch tests run on CPU: the check runs before any kernel. The SM107 tests
run the admitted path's forward at 128-token pages: fresh, cached-prefix and chunked contexts must match the same
sequences run unchunked, BF16 and FP8 KV."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha import fallback
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
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


class _PastRefusal(Exception):
    """Raised by the stubbed paged-context validation that follows the refusal check."""


def _fmha():
    fmha = object.__new__(fallback.FallbackFmha)

    def _validate(metadata):
        raise _PastRefusal()

    fmha._validate_paged_context_fmha = _validate
    return fmha


def _context_batch(tokens_per_block: int = 128, paged: bool = True):
    return SimpleNamespace(
        num_contexts=1,
        kv_cache_manager=SimpleNamespace(trtllm_gen_extra_tokens_per_block=frozenset({128})),
        use_paged_context_fmha=paged,
        tokens_per_block=tokens_per_block,
    )


@pytest.mark.parametrize("sm", [90, 100, 103, 120])
def test_extra_page_paged_prefill_is_refused_off_sm107(monkeypatch, sm) -> None:
    monkeypatch.setattr(fallback, "get_sm_version", lambda: sm)
    with pytest.raises(RuntimeError, match="cannot serve paged prefill with 128-token KV pages"):
        _fmha()._is_supported(None, None, None, _context_batch(), None, phase=FmhaPhase.CONTEXT)


def test_extra_page_paged_prefill_is_served_on_sm107(monkeypatch) -> None:
    monkeypatch.setattr(fallback, "get_sm_version", lambda: 107)
    with pytest.raises(_PastRefusal):
        _fmha()._is_supported(None, None, None, _context_batch(), None, phase=FmhaPhase.CONTEXT)


@pytest.mark.parametrize("sm", [100, 107])
def test_other_page_sizes_and_unpaged_prefill_pass_the_check(monkeypatch, sm) -> None:
    monkeypatch.setattr(fallback, "get_sm_version", lambda: sm)
    for batch in (_context_batch(tokens_per_block=32), _context_batch(paged=False)):
        with pytest.raises(_PastRefusal):
            _fmha()._is_supported(None, None, None, batch, None, phase=FmhaPhase.CONTEXT)


_ON_SM107 = torch.cuda.is_available() and get_sm_version() == 107
_HEADS, _KV_HEADS, _HEAD_DIM, _PAGE = 8, 2, 128, 128
_REF_RID = 100  # the unchunked run of request r is request _REF_RID + r

# (case id, steps): each step is a batch of (request id, cached tokens, new tokens) context rows, run in order.
_CACHED_CONTEXT_CASES = [
    ("fresh_512", [[(0, 0, 512)]]),
    ("prefix_1000_unaligned", [[(0, 0, 1000)], [(0, 1000, 700)]]),
    ("prefix_1024_aligned", [[(0, 0, 1024)], [(0, 1024, 300)]]),
    ("chunked_6096_at_2048", [[(0, 0, 2048)], [(0, 2048, 2048)], [(0, 4096, 2000)]]),
    ("prefix_beside_fresh", [[(0, 0, 384)], [(0, 384, 257), (1, 0, 129)]]),
]
# Worst row's max |out - ref| / max |ref| per KV dtype: chunked vs unchunked (same cached K / V), and unchunked vs a
# float reference (the FP8 KV kernel also quantizes Q and the softmax to FP8).
_TOLERANCE = {
    torch.bfloat16: {"unchunked": 1e-2, "float": 2e-2},
    torch.float8_e4m3fn: {"unchunked": 3e-2, "float": 1.5e-1},
}


def _kv_manager(kv_dtype: torch.dtype, num_tokens: int) -> KVCacheManager:
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=num_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        tokens_per_block=_PAGE,
        max_seq_len=8192,
        max_batch_size=8,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8 if kv_dtype == torch.float8_e4m3fn else DataType.BF16,
    )
    # As MiniMax-M3's KV managers list it: this is what the admission in FallbackFmha keys on.
    manager.trtllm_gen_extra_tokens_per_block = frozenset({_PAGE})
    # Zero the pool: a context step reads a request's whole last page, and stale bytes in its unwritten tail that
    # decode as FP8 NaN turn the masked P = 0 into NaN outputs.
    manager.get_buffers(0, kv_layout="HND").view(torch.uint8).zero_()
    return manager


def _context_step(attn, manager, rows, qkv: torch.Tensor) -> torch.Tensor:
    """One context-only step as the serving runner prepares it: a request's prompt length is its current chunk, with
    the prefix in ``num_cached_tokens_per_seq``."""
    metadata = TrtllmAttentionMetadata(
        max_num_requests=8,
        max_num_tokens=8192,
        kv_cache_manager=manager,
        runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
        num_contexts=len(rows),
        seq_lens=torch.tensor([n for _, _, n in rows], dtype=torch.int),
        request_ids=[rid for rid, _, _ in rows],
        prompt_lens=[n for _, _, n in rows],
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[c for _, c, _ in rows]
        ),
    )
    assert metadata.use_paged_context_fmha and metadata.tokens_per_block == _PAGE
    metadata.prepare()
    return attn.forward(qkv, None, None, metadata, attention_mask=PredefinedAttentionMask.CAUSAL)


def _cached_kv(manager, rid: int, length: int) -> torch.Tensor:
    pool = manager.get_buffers(0, kv_layout="HND")  # [pages, K/V, kv heads, page tokens, head dim]
    pages = manager.get_batch_cache_indices([rid], 0)[0][: -(-length // _PAGE)]
    kv = pool[torch.tensor(pages, device=pool.device)].permute(1, 2, 0, 3, 4)
    return kv.reshape(2, _KV_HEADS, -1, _HEAD_DIM)[:, :, :length].view(torch.uint8)


def _float_reference(
    qkv: torch.Tensor, start: int, n: int, kv_from: int, fp8: bool
) -> torch.Tensor:
    """Causal attention of rows start..start+n over keys kv_from..start+n, in float32."""
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


@pytest.mark.skipif(not _ON_SM107, reason="the SM107 thop paged-context path at 128-token pages")
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16_kv", "fp8_kv"]
)
@pytest.mark.parametrize("case", _CACHED_CONTEXT_CASES, ids=[c[0] for c in _CACHED_CONTEXT_CASES])
def test_sm107_cached_context_matches_unchunked(monkeypatch, case, kv_dtype) -> None:
    """The admitted path attends to the cached prefix: each chunk's output matches the same rows of the request run
    in one unchunked step, and the chunks leave the same K / V in the cache."""
    name, steps = case
    fp8 = kv_dtype == torch.float8_e4m3fn
    tolerance = _TOLERANCE[kv_dtype]
    lengths = {}
    for step in steps:
        for rid, c, n in step:
            assert c == lengths.get(rid, 0), "each chunk starts at its request's cached length"
            lengths[rid] = c + n
    served = []
    forward = fallback.FallbackFmha.forward
    monkeypatch.setattr(
        fallback.FallbackFmha,
        "forward",
        lambda self, q, k, v, metadata, args: served.append(metadata.num_contexts)
        or forward(self, q, k, v, metadata, args),
    )

    gen = torch.Generator(device="cuda").manual_seed(len(name))
    width = (_HEADS + 2 * _KV_HEADS) * _HEAD_DIM
    qkv = {
        rid: torch.randn(n, width, generator=gen, device="cuda").to(torch.bfloat16)
        for rid, n in lengths.items()
    }
    num_tokens = 2 * sum(-(-n // _PAGE) * _PAGE for n in lengths.values())
    manager = _kv_manager(kv_dtype, num_tokens)
    try:
        attn = TrtllmAttention(
            layer_idx=0,
            num_heads=_HEADS,
            head_dim=_HEAD_DIM,
            num_kv_heads=_KV_HEADS,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8) if fp8 else None,
        )
        rids = sorted(lengths)
        manager.add_dummy_requests(
            rids + [_REF_RID + r for r in rids], [lengths[r] for r in rids] * 2
        )
        # The unchunked run: every request as one fresh context.
        rows = [(_REF_RID + r, 0, lengths[r]) for r in rids]
        out = _context_step(attn, manager, rows, torch.cat([qkv[r] for r in rids]))
        unchunked, first = {}, 0
        for r in rids:
            unchunked[r] = out[first : first + lengths[r]]
            first += lengths[r]
            error = _relative_error(unchunked[r], _float_reference(qkv[r], 0, lengths[r], 0, fp8))
            print(f"\n[{name}] request {r} unchunked vs float reference: {error:.4f}")
            assert error <= tolerance["float"]

        for step in steps:
            out = _context_step(
                attn, manager, step, torch.cat([qkv[r][c : c + n] for r, c, n in step])
            )
            first = 0
            for r, c, n in step:
                got = out[first : first + n]
                first += n
                assert torch.isfinite(got.float()).all()
                error = _relative_error(got, unchunked[r][c : c + n])
                print(f"[{name}] request {r} rows {c}..{c + n} vs unchunked: {error:.4f}")
                assert error <= tolerance["unchunked"]
                if c:
                    # A kernel that dropped the cached prefix would match attention over the chunk alone.
                    dropped = _float_reference(qkv[r], c, n, c, fp8)
                    assert _relative_error(got, dropped) > 4 * tolerance["float"]
        assert served and all(served), "the context steps ran through FallbackFmha"
        torch.cuda.synchronize()
        for r in rids:
            assert torch.equal(
                _cached_kv(manager, r, lengths[r]), _cached_kv(manager, _REF_RID + r, lengths[r])
            ), f"request {r}: the chunks' K / V differ from the unchunked run's"
    finally:
        manager.shutdown()
