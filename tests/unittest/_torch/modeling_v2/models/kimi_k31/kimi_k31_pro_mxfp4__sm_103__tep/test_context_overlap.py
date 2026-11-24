# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 worker's side-stream drafter context projection (``K31DFlashWorker._k31_context_begin`` /
``_k31_context_join``) on one GPU: the joined latents equal the serial projection bit for bit, eager and replayed
from a CUDA graph; a projection is joined once; nothing is issued with ``TLLM_K31_DFLASH2_CTX_OVERLAP=0`` or outside
multi-stream execution; the one-launch prepare joins a pending projection before it falls back to the torch path
(which projects, and all-gathers, on the current stream). The collective order on a TP group is covered by the
4-rank replica under compute-sanitizer and by the in-model identity gates."""

import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep import (
    modeling,
)
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, CAPTURES, LATENT, LAYERS = 1024, 3, 576, 5


class _Drafter:
    """The drafter methods the context projection calls, on plain torch ops."""

    context_kv_only = False

    def __init__(self, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.fc_weight = (
            torch.randn(H, CAPTURES * H, generator=g, device="cuda") * 0.02
        ).bfloat16()
        self.latent_weight = (
            torch.randn(LAYERS * LATENT, H, generator=g, device="cuda") * 0.02
        ).bfloat16()
        self._fused_kv_weight = self.latent_weight
        self.model = types.SimpleNamespace(embed_tokens=None)

    def project_target_hidden(self, hidden_states):
        x = F.linear(hidden_states, self.fc_weight).float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()

    def k31_context_latents(self, projected):
        n = projected.shape[0]
        return F.linear(projected, self.latent_weight).view(n, LAYERS, LATENT)


def _step(num_gens: int, width: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    captured = torch.randn(
        num_gens * width + 3, CAPTURES * H, generator=g, device="cuda"
    ).bfloat16()
    spec = types.SimpleNamespace(
        runtime_draft_len=width - 1, get_hidden_states=lambda n: captured[:n] if n else None
    )
    attn = types.SimpleNamespace(num_seqs=num_gens, num_contexts=0, num_ctx_tokens=0)
    return spec, attn, captured


def _worker():
    worker = modeling.K31DFlashWorker.__new__(modeling.K31DFlashWorker)
    worker._ctx_kv_buf = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    worker._ctx_page_table = None
    worker._k31_fused_prepare_serves = lambda *args: True
    return worker


def _serial(drafter, captured, rows):
    return drafter.k31_context_latents(drafter.project_target_hidden(captured[:rows]))


@pytest.mark.parametrize("num_gens", [1, 2, 3, 8])
def test_joined_latents_equal_serial(num_gens, monkeypatch) -> None:
    monkeypatch.delenv("TLLM_K31_DFLASH2_CTX_OVERLAP", raising=False)
    width = 5
    drafter = _Drafter(seed=num_gens)
    spec, attn, captured = _step(num_gens, width, seed=7 + num_gens)
    worker = _worker()
    ref = _serial(drafter, captured, num_gens * width)
    with with_multi_stream(True):
        worker._k31_context_begin(drafter, attn, spec, num_gens * width)
        busy = torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")
        latents = worker._k31_context_join()
    assert latents is not None, "the projection was not issued on the side stream"
    assert worker._k31_context_join() is None, "a projection is joined once"
    torch.cuda.synchronize()
    assert torch.equal(latents, ref)
    del busy


def test_cuda_graph_replay(monkeypatch) -> None:
    monkeypatch.delenv("TLLM_K31_DFLASH2_CTX_OVERLAP", raising=False)
    num_gens, width = 3, 5
    drafter = _Drafter(seed=1)
    spec, attn, captured = _step(num_gens, width, seed=2)
    worker = _worker()
    out = {}
    with with_multi_stream(True):
        worker._k31_context_begin(drafter, attn, spec, num_gens * width)
        worker._k31_context_join()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            worker._k31_context_begin(drafter, attn, spec, num_gens * width)
            out["latents"] = worker._k31_context_join() * 2
        torch.cuda.current_stream().wait_stream(stream)
    g = torch.Generator(device="cuda").manual_seed(3)
    for _ in range(3):
        captured.copy_(torch.randn(captured.shape, generator=g, device="cuda").bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out["latents"], _serial(drafter, captured, num_gens * width) * 2)


def test_nothing_issued_when_off(monkeypatch) -> None:
    drafter = _Drafter(seed=4)
    spec, attn, _ = _step(2, 5, seed=5)
    worker = _worker()
    monkeypatch.setenv("TLLM_K31_DFLASH2_CTX_OVERLAP", "0")
    with with_multi_stream(True):
        worker._k31_context_begin(drafter, attn, spec, 10)
    assert worker._k31_context_join() is None
    monkeypatch.delenv("TLLM_K31_DFLASH2_CTX_OVERLAP")
    with with_multi_stream(False):
        worker._k31_context_begin(drafter, attn, spec, 10)
    assert worker._k31_context_join() is None


def test_fused_prepare_fallback_joins_first(monkeypatch) -> None:
    """A step the one-launch prepare declines (here: int64 counts) joins the pending projection before returning to
    the torch path."""
    monkeypatch.delenv("TLLM_K31_DFLASH2_CTX_OVERLAP", raising=False)
    drafter = _Drafter(seed=6)
    spec, attn, _ = _step(2, 5, seed=8)
    worker = _worker()
    with with_multi_stream(True):
        worker._k31_context_begin(drafter, attn, spec, 10)
        assert worker._k31_ctx_pending is not None
        accepted = torch.zeros(2, 5, dtype=torch.int32, device="cuda")
        counts = torch.ones(2, dtype=torch.int64, device="cuda")
        assert (
            worker._k31_fused_gen_inputs(accepted, counts, attn, spec, drafter, 10, None, 0, None)
            is None
        )
    assert worker._k31_ctx_pending is None
