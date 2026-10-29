# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context workspace of the Mistral-Large-4-1T-A52B NVFP4 dep4 target: the KV rows its
dequantizing gather and prefill kernel address. Under KV block reuse a context request's KV
is its cached prefix plus its new tokens, so the workspace covers both, while the engine's
`num_ctx_kv_tokens` counts the new tokens only. Under chunked prefill every chunk after a
prompt's first is such a request, its earlier chunks the cached prefix.

The workspace is the first rows of one held bf16 [rows, 576] buffer, allocated on the first
context step, its rows sized from the engine's quantities (`_ctx_workspace_rows`): 65536 at the
32K serving config (max_seq_len = max_num_tokens = 32768, chunked prefill and block reuse off),
2**19 at the 1M config (max_seq_len 1048576, chunked prefill). A step whose context KV fits the
held rows twice has the rows after its own (`kept`) for layer 35's gather; a step whose context
KV is past the held rows takes the decode path, which has no workspace
(test_modeling_v2_ctx_bound.py drives both on a device). No device: the step is built from
host-side metadata."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata

pytestmark = pytest.mark.cpu_only

_LAT_DIM = 576
# The 1M config, per rank: max_seq_len, and the most tokens one step carries.
_MAX_SEQ_LEN = 1048576
_CHUNK = 32768
# The engines, per rank, as (max_seq_len, chunked prefill): the 32K serving config and the 1M
# config, both at max_num_tokens 32768, 16 requests and block reuse off.
_ENGINES = {"32k": (32768, False), "1m": (_MAX_SEQ_LEN, True)}


def _held_rows(engine: str) -> int:
    """The held workspace's rows `_derive_first_step` gives the core on `engine`: 65536 at the
    32K config (twice its max_num_tokens), 2**19 at the 1M config."""
    max_seq_len, chunked = _ENGINES[engine]
    md = SimpleNamespace(
        max_num_tokens=_CHUNK,
        max_num_requests=16,
        max_seq_len=max_seq_len,
        runtime_features=AttentionRuntimeFeatures(chunked_prefill=chunked, cache_reuse=False),
    )
    rows = M._ctx_workspace_rows(md)
    assert rows == {"32k": 65536, "1m": 1 << 19}[engine]
    return rows


def _target(engine: str) -> M.PrefillTarget:
    """A prefill target holding just what its context workspace reads off the core on `engine`."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(lat_dim=_LAT_DIM, ctx_workspace_rows=_held_rows(engine))
    return target


def _cumsum(lens):
    return torch.tensor([0] + lens, dtype=torch.int64).cumsum(0)


def _check_decode_path(ws, tc):
    """A step past the held workspace: no workspace, the context rows in sub-batches of at most
    8192, one scheduler per sub-batch size."""
    assert ws.ws_buf is None
    assert ws.sub_batches == [(lo, min(lo + 8192, tc)) for lo in range(0, tc, 8192)]
    assert sorted(ws.sched) == sorted({hi - lo for lo, hi in ws.sub_batches})


def _metadata(cached, new):
    """The fields `_ctx_kv_offsets` reads, as the engine fills them for context requests with
    `cached` reused (or earlier-chunk) and `new` scheduled tokens each."""
    if any(cached):
        return SimpleNamespace(
            enable_context_mla_with_cached_kv=True,
            ctx_cached_token_indptr=_cumsum(cached),
            ctx_kv_indptr=_cumsum([c + n for c, n in zip(cached, new)]),
            max_ctx_seq_len=max(new),
            num_ctx_cached_tokens=sum(cached),
        )
    return SimpleNamespace(
        enable_context_mla_with_cached_kv=False,
        mla_prepare_ctx_cu_seqlens=lambda: _cumsum(new).to(torch.int32),
        seq_lens=torch.tensor(new, dtype=torch.int32),
    )


def _check_workspace_path(target, ws, n_kv: int) -> None:
    """A step on the held workspace: its first `n_kv` rows, and the `n_kv` after them kept for
    layer 35's gather when the context KV fits twice."""
    held = target._ws_buf.shape[0]
    assert ws.ws_buf.shape == (n_kv, _LAT_DIM) and ws.ws_buf.dtype == torch.bfloat16
    assert ws.ws_buf.data_ptr() == target._ws_buf.data_ptr() and ws.ws == {}
    if 2 * n_kv <= held:
        assert ws.kept.shape == (n_kv, _LAT_DIM)
        assert ws.kept.data_ptr() == ws.ws_buf.data_ptr() + n_kv * _LAT_DIM * 2
    else:
        assert ws.kept is None


@pytest.mark.parametrize(
    "engine, cached, new",
    [
        pytest.param("32k", [0, 0], [7, 33], id="32k-no_reuse"),
        pytest.param("32k", [64, 0, 96], [5, 40, 1], id="32k-mixed_prefixes"),
        pytest.param("32k", [32], [1], id="32k-one_new_token"),
        pytest.param("32k", [0], [_CHUNK], id="32k-whole_step"),
        pytest.param("1m", [0, 0], [7, 33], id="1m-no_reuse"),
        pytest.param("1m", [64, 0, 96], [5, 40, 1], id="1m-mixed_prefixes"),
        pytest.param("1m", [32], [1], id="1m-one_new_token"),
        # A 1048559-token prompt's last chunk at the 1M config (31 chunks of 32768 cached),
        # beside a first chunk of another prompt on the same rank.
        pytest.param("1m", [31 * _CHUNK, 0], [32751, 9], id="1m-chunk_over_1m_prefix"),
        # Two 5-token last chunks over 983040-token prefixes, packed into one indexer chunk of
        # 1966090 keys, past max_seq_len.
        pytest.param("1m", [983040, 983040], [5, 5], id="1m-packed_tails"),
    ],
)
def test_workspace_covers_the_cached_prefix_and_the_new_tokens(engine, cached, new):
    """The step's workspace is its context KV, cached and new, in the held buffer of the engine's
    rows; past them, the decode path."""
    step = dict(
        num_contexts=len(new),
        num_ctx_tokens=sum(new),
        num_ctx_kv_tokens=sum(new),
        **M._ctx_kv_offsets(_metadata(cached, new), len(new)),
    )
    target = _target(engine)
    ws = target._ctx_workspace(step, sum(new), torch.device("cpu"))

    kv_lens = [c + n for c, n in zip(cached, new)]
    held = target.core.ctx_workspace_rows
    assert target._ws_buf.shape == (held, _LAT_DIM) and target._ws_buf.dtype == torch.bfloat16
    if sum(kv_lens) > held:
        _check_decode_path(ws, sum(new))
        return
    _check_workspace_path(target, ws, sum(kv_lens))
    assert ws.kv_req.tolist() == [r for r, n in enumerate(kv_lens) for _ in range(n)]
    assert ws.kv_pos.tolist() == [p for n in kv_lens for p in range(n)]
    assert ws.ws_off.tolist() == _cumsum(kv_lens)[:-1].tolist()


def _engine_step(contexts, decode_kv_lens=()):
    """One dep4 rank's step as the engine's DSA metadata prepares it under chunked prefill:
    `contexts` lists each context request's (cached prefix, new tokens), `decode_kv_lens` each
    generation row's KV length. `DSAtrtllmAttentionMetadata.prepare` takes the KV lengths as
    cached + new (`cached_token_lens + seq_lens_kv`) and hands both to its own
    `prepare_for_mla_rope_append`, which fills the cached count and the cached and total prefix
    sums run here; the new tokens' host prefix sum is the base metadata's. The metadata holds
    what that method reads, its buffers at this step's size."""
    num_contexts, num_generations = len(contexts), len(decode_kv_lens)
    seq_lens = [new for _, new in contexts] + [1] * num_generations
    cached = [prefix for prefix, _ in contexts] + [kv_len - 1 for kv_len in decode_kv_lens]
    num_seqs = num_contexts + num_generations
    i64 = dict(dtype=torch.int64)
    md = SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_seqs=num_seqs,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        enable_context_mla_with_cached_kv=True,
        host_ctx_uncached_token_indptr=torch.tensor(
            [0, *itertools.accumulate(seq_lens[:num_contexts])], **i64
        ),
        **{
            name: torch.zeros(num_seqs + 1, **i64)
            for name in (
                "host_ctx_cached_token_indptr",
                "ctx_cached_token_indptr",
                "host_ctx_kv_indptr",
                "ctx_kv_indptr",
                "host_gen_cached_token_indptr",
                "gen_cached_token_indptr",
                "host_gen_kv_indptr",
                "gen_kv_indptr",
            )
        },
    )
    cached_token_lens = torch.tensor(cached, dtype=torch.int)
    kv_lens = cached_token_lens + md.seq_lens
    DSAtrtllmAttentionMetadata.prepare_for_mla_rope_append(md, cached_token_lens, kv_lens)
    return md


@pytest.mark.parametrize(
    "contexts, decode_kv_lens",
    [
        # A 1048559-token prompt's first, second and last chunks, the last with 15 decode rows
        # (up to max_seq_len keys) behind it.
        ([(0, _CHUNK)], ()),
        ([(_CHUNK, _CHUNK)], ()),
        ([(31 * _CHUNK, 32751)], [_MAX_SEQ_LEN - 1, 2, 100000] + [4096] * 12),
        # Its last chunk beside another prompt's first, and two packed 5-token tails.
        ([(31 * _CHUNK, 32751), (0, 9)], ()),
        ([(983040, 5), (983040, 5)], [700000]),
    ],
    ids=["first_chunk", "second_chunk", "last_chunk_with_decode", "last_and_first", "packed_tails"],
)
def test_n_kv_is_the_engines_cached_plus_new_total(contexts, decode_kv_lens):
    """`_ctx_workspace` sizes the workspace `ctx_cached_tokens + num_ctx_tokens`: on every step
    at the 1M config that is the context requests' cached + new total the engine itself sums
    (`num_ctx_mla_kv_tokens`, the last entry of `ctx_kv_indptr`), whatever the generation rows
    behind them hold; on every chunk after a prompt's first it is more than the step's new
    tokens. The host prefix sums (`_ctx_host_offsets`) end at the same total and at the new
    tokens'."""
    md = _engine_step(contexts, decode_kv_lens)
    num_contexts = len(contexts)
    new = [n for _, n in contexts]
    step = dict(
        num_contexts=num_contexts,
        num_ctx_tokens=sum(new),
        **M._ctx_kv_offsets(md, num_contexts),
        **M._ctx_host_offsets(md, num_contexts),
    )
    total = sum(prefix + n for prefix, n in contexts)
    assert step["ctx_cached_tokens"] == sum(prefix for prefix, _ in contexts)
    assert step["ctx_cached_tokens"] + step["num_ctx_tokens"] == total
    assert total == md.num_ctx_mla_kv_tokens == int(md.host_ctx_kv_indptr[num_contexts])
    assert int(step["cu_ctx_kv"][-1]) == total
    assert step["host_cu_ctx_kv"][-1] == total and step["host_cu_ctx_new"][-1] == sum(new)
    if any(prefix for prefix, _ in contexts):
        assert total > step["num_ctx_tokens"]

    target = _target("1m")
    ws = target._ctx_workspace(step, sum(new), torch.device("cpu"))
    assert target._ws_buf.shape == (M._CTX_WORKSPACE_ROWS, _LAT_DIM)
    if total > M._CTX_WORKSPACE_ROWS:
        _check_decode_path(ws, sum(new))
        return
    _check_workspace_path(target, ws, total)
    assert ws.kv_req.shape == ws.kv_pos.shape == (total,)
    starts = [0, *itertools.accumulate(prefix + n for prefix, n in contexts)][:-1]
    assert ws.ws_off.tolist() == starts
    for r, (prefix, n) in enumerate(contexts):
        lo, hi = starts[r], starts[r] + prefix + n
        assert int(ws.kv_req[lo]) == int(ws.kv_req[hi - 1]) == r
        assert (int(ws.kv_pos[lo]), int(ws.kv_pos[hi - 1])) == (0, prefix + n - 1)
