# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context workspace of the Mistral-Large-4-1T-A52B NVFP4 tep4 target: the KV rows its
dequantizing gather and prefill kernel address. Under KV block reuse a context request's KV
is its cached prefix plus its new tokens, so the workspace covers both, while the engine's
`num_ctx_kv_tokens` counts the new tokens only. Under chunked prefill every chunk after a
prompt's first is such a request, its earlier chunks the cached prefix.

The workspace is the first KV rows of one held bf16 [rows, 576] buffer, allocated on the first
context step at the rows `_ctx_workspace_rows` takes from the engine's attention metadata (chunked
prefill and block reuse from its runtime features): 65536 at the 32768 serving config
(max_seq_len = max_num_tokens = 32768, chunked prefill and block reuse off), 2**19 at that config
with block reuse and at the 1M config (max_seq_len 1048576, chunked prefill). A step whose KV
fits the buffer twice also keeps the next KV rows for layer 35's gather; a step whose KV is past
the buffer takes the decode path, in sub-batches of at most 8192 context rows. The latent is
replicated, so every rank builds the same workspace at tep4 (four ranks) and at tep8 (eight
ranks over two nodes of four). No device: the step is built from host-side metadata."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures

pytestmark = pytest.mark.cpu_only

_LAT_DIM = 576


def _cumsum(lens):
    return torch.tensor([0] + lens, dtype=torch.int64).cumsum(0)


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


#: The held workspace's rows, from the engine's metadata (`_ctx_workspace_rows`, chunked prefill
#: and block reuse from its runtime features): the 32768 config's (max_num_tokens 32768, no
#: chunked prefill or block reuse), the same with block reuse at max_batch_size 16, and the 1M
#: config's (max_seq_len 1048576, chunked prefill).
_HELD = {
    "32k": ((32768, 32768, False, False), 1 << 16),
    "32k_reuse": ((32768, 32768, False, True), 1 << 19),
    "1m": ((32768, 1048576, True, False), 1 << 19),
}


def _held_rows(config: str) -> int:
    (max_num_tokens, max_seq_len, chunked, reuse), rows = _HELD[config]
    md = SimpleNamespace(
        max_num_tokens=max_num_tokens,
        max_num_requests=16,
        max_seq_len=max_seq_len,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=chunked, cache_reuse=reuse, chunk_size=max_num_tokens
        ),
    )
    assert M._ctx_workspace_rows(md) == rows
    return rows


@pytest.mark.parametrize(
    "config, cached, new",
    [
        pytest.param("32k", [0, 0], [7, 33], id="32k-no_reuse"),
        # A full 32768-token step: its KV fits the 65536 held rows twice, exactly.
        pytest.param("32k", [0], [32768], id="32k-full_step"),
        pytest.param("32k_reuse", [64, 0, 96], [5, 40, 1], id="32k_reuse-mixed_prefixes"),
        pytest.param("32k_reuse", [32], [1], id="32k_reuse-one_new_token"),
        pytest.param("1m", [0, 0], [7, 33], id="1m-no_reuse"),
        pytest.param("1m", [64, 0, 96], [5, 40, 1], id="1m-mixed_prefixes"),
        pytest.param("1m", [32], [1], id="1m-one_new_token"),
        # The 16th chunk of a 1M prompt: KV 2**19, the held rows once, nothing kept.
        pytest.param("1m", [15 * 32768], [32768], id="1m-chunk_at_the_held_rows"),
        # A 1M prompt's last chunk (31 chunks of 32768 cached), beside a first chunk of another
        # prompt: past the held rows.
        pytest.param("1m", [31 * 32768, 0], [32751, 9], id="1m-chunk_over_1m_prefix"),
    ],
)
def test_workspace_covers_the_cached_prefix_and_the_new_tokens(config, cached, new):
    step = dict(
        num_contexts=len(new),
        num_ctx_tokens=sum(new),
        num_ctx_kv_tokens=sum(new),
        **M._ctx_kv_offsets(_metadata(cached, new), len(new)),
    )
    held = _held_rows(config)
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(lat_dim=_LAT_DIM, ctx_workspace_rows=held)
    ws = target._ctx_workspace(step, sum(new), torch.device("cpu"))

    kv_lens = [c + n for c, n in zip(cached, new)]
    n_kv = sum(kv_lens)
    assert target._ws_buf.shape == (held, _LAT_DIM) and target._ws_buf.dtype == torch.bfloat16
    if n_kv > held:
        assert ws.ws_buf is None
        assert ws.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, sum(new))]
        return
    buf = target._ws_buf
    assert ws.ws_buf.data_ptr() == buf.data_ptr()
    assert ws.ws_buf.shape == (n_kv, _LAT_DIM)
    if 2 * n_kv <= held:
        assert ws.kept.shape == (n_kv, _LAT_DIM)
        assert ws.kept.data_ptr() == buf.data_ptr() + n_kv * _LAT_DIM * buf.element_size()
    else:
        assert ws.kept is None
    assert ws.ws == {}
    assert ws.kv_req.tolist() == [r for r, n in enumerate(kv_lens) for _ in range(n)]
    assert ws.kv_pos.tolist() == [p for n in kv_lens for p in range(n)]
    assert ws.ws_off.tolist() == _cumsum(kv_lens)[:-1].tolist()
