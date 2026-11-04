# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context workspace of the Mistral-Large-4-1T-A52B NVFP4 tp4 target: the KV rows its
dequantizing gather and prefill kernel address. Under KV block reuse a context request's KV
is its cached prefix plus its new tokens, so the workspace covers both, while the engine's
`num_ctx_kv_tokens` counts the new tokens only. The workspace is the first rows of the bf16
workspace the target holds, `[core.ctx_workspace_rows, 576]`, allocated on the first context
step; its rows are sized from the engine's quantities (`_ctx_workspace_rows`), checked at two
of them (ids `32k` and `1m`): the 32768 serving config (max_num_tokens = max_seq_len = 32768,
neither chunked prefill nor block reuse), 65536 rows, and the 1M config (max_seq_len 1048576,
chunked prefill), 2**19. The steps here fit either twice, so the next rows are kept beside them
for the shared owner. No device: the step is built from host-side metadata."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

_LAT_DIM = 576
#: The engine quantities the held workspace is sized from: (max_num_tokens, max_seq_len,
#: chunked prefill, block reuse) at max_batch_size 16, and the rows they give.
_CONFIGS = [
    pytest.param((32768, 32768, False, False), 65536, id="32k"),
    pytest.param((32768, 1048576, True, False), 1 << 19, id="1m"),
]


def _cumsum(lens):
    return torch.tensor([0] + lens, dtype=torch.int64).cumsum(0)


def _metadata(cached, new):
    """The fields `_ctx_kv_offsets` reads, as the engine fills them for context requests with
    `cached` reused and `new` scheduled tokens each."""
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


def _held_rows(max_num_tokens, max_seq_len, chunked, reuse) -> int:
    """`_ctx_workspace_rows` for an engine at these quantities, off the attention metadata
    fields it reads (block reuse is the runtime features' cache_reuse)."""
    return M._ctx_workspace_rows(
        SimpleNamespace(
            max_num_tokens=max_num_tokens,
            max_num_requests=16,
            max_seq_len=max_seq_len,
            runtime_features=SimpleNamespace(chunked_prefill=chunked, cache_reuse=reuse),
        )
    )


@pytest.mark.parametrize("engine, rows", _CONFIGS)
@pytest.mark.parametrize(
    "cached, new",
    [
        ([0, 0], [7, 33]),
        ([64, 0, 96], [5, 40, 1]),
        ([32], [1]),
    ],
    ids=["no_reuse", "mixed_prefixes", "one_new_token"],
)
def test_workspace_covers_the_cached_prefix_and_the_new_tokens(cached, new, engine, rows):
    step = dict(
        num_contexts=len(new),
        num_ctx_tokens=sum(new),
        num_ctx_kv_tokens=sum(new),
        **M._ctx_kv_offsets(_metadata(cached, new), len(new)),
    )
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(lat_dim=_LAT_DIM, ctx_workspace_rows=_held_rows(*engine))
    assert target.core.ctx_workspace_rows == rows
    ws = target._ctx_workspace(step, sum(new), torch.device("cpu"))

    kv_lens = [c + n for c, n in zip(cached, new)]
    n_kv = sum(kv_lens)
    assert ws.ws_buf.shape == (n_kv, _LAT_DIM)
    assert ws.kv_req.tolist() == [r for r, n in enumerate(kv_lens) for _ in range(n)]
    assert ws.kv_pos.tolist() == [p for n in kv_lens for p in range(n)]
    assert ws.ws_off.tolist() == _cumsum(kv_lens)[:-1].tolist()
    # The workspaces the step keeps by owner layer: none before any layer attends.
    assert ws.ws == {}
    # The first rows of the held workspace, and the next as many kept beside them.
    assert target._ws_buf.shape == (rows, _LAT_DIM)
    assert target._ws_buf.dtype == torch.bfloat16
    assert ws.ws_buf.data_ptr() == target._ws_buf.data_ptr()
    assert ws.kept.shape == (n_kv, _LAT_DIM)
    assert ws.kept.data_ptr() == target._ws_buf[n_kv].data_ptr()
