# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical parity of the VisualGen VSA backends.

A Wan-style self-attention module (fused QKV projection, QK RMSNorm, RoPE) is run
with the CuTeDSL and TRTLLM VSA backends on a ragged latent and compared with
dense attention through the module's own projections, and with each other.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from utils.util import isSM100Family

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa import (
    VSAMetadataBuilder,
    set_vsa_forward_context,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.vsa.backend import VSACuTeDSLAttention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode, apply_rotary_emb
from tensorrt_llm.visual_gen.args import AttentionConfig, VideoSparseAttentionConfig

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA needs CUDA")
_REQUIRES_SM100 = pytest.mark.skipif(
    not isSM100Family(),
    reason="CuTe DSL and PrimTS block-sparse parity requires SM100 or SM103",
)
_BACKENDS = pytest.mark.parametrize("backend", ["CUTEDSL", "TRTLLM"])

# A ragged latent exercises VSA padding and token-mask lowering on both fine stages.
_LATENT_SHAPE = (9, 9, 9)
_SEQ_LEN = _LATENT_SHAPE[0] * _LATENT_SHAPE[1] * _LATENT_SHAPE[2]
_NUM_HEADS = 4
_HEAD_DIM = 128
_HIDDEN_SIZE = _NUM_HEADS * _HEAD_DIM


def _make_attention(backend: str, sparsity: float, *, seed: int) -> Attention:
    """A Wan self-attention module on CUDA whose weights come from ``seed``."""

    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=_HIDDEN_SIZE,
            num_attention_heads=_NUM_HEADS,
            attention_head_dim=_HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(
            backend=backend,
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=sparsity),
        ),
        skip_create_weights_in_init=False,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    attention = Attention(_HIDDEN_SIZE, _NUM_HEADS, qkv_mode=QKVMode.FUSE_QKV, config=config)
    attention = attention.to(device=torch.device("cuda"), dtype=torch.bfloat16).eval()
    # Fail loudly if the VSA path silently fell back to the VANILLA backend.
    assert attention.attn_backend == backend, (
        f"Expected {backend} VSA backend, got {attention.attn_backend!r}"
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for name, parameter in sorted(attention.named_parameters()):
        values = torch.randn(parameter.shape, generator=generator, device="cuda")
        if "norm" in name:
            values = 1.0 + 0.05 * values
        else:
            values = 0.05 * values
        parameter.data.copy_(values.to(parameter.dtype))
    return attention


def _vsa_metadata(sparsity: float):
    return VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=_LATENT_SHAPE,
        patch_size=(1, 1, 1),
        vsa_sparsity=sparsity,
        device=torch.device("cuda"),
    )


def _rope(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables with the full head_dim in the ``[1, S, 1, D]`` layout the module uses."""

    device = torch.device("cuda")
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, head_dim, device=device) * (-torch.log(torch.tensor(10000.0)) / head_dim)
    )
    freqs_cos = torch.cos(position * div_term).unsqueeze(0).unsqueeze(2)
    freqs_sin = torch.sin(position * div_term).unsqueeze(0).unsqueeze(2)
    return freqs_cos, freqs_sin


def _dense_reference(attention: Attention, hidden_states: torch.Tensor, freqs) -> torch.Tensor:
    """Dense self-attention through the module's own projections, norms and RoPE."""

    batch_size, seq_len = hidden_states.shape[:2]
    q, k, v = attention.get_qkv(hidden_states, None)
    q, k = attention.apply_qk_norm(q, k)
    q = apply_rotary_emb(q.view(batch_size, seq_len, _NUM_HEADS, _HEAD_DIM), *freqs)
    k = apply_rotary_emb(k.view(batch_size, seq_len, _NUM_HEADS, _HEAD_DIM), *freqs)
    v = v.view(batch_size, seq_len, _NUM_HEADS, _HEAD_DIM)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return attention.to_out[0](out.transpose(1, 2).flatten(2))


def _run(attention: Attention, hidden_states: torch.Tensor, freqs, sparsity: float) -> torch.Tensor:
    """One VSA forward with the coarse gate closed, so the output is the fine branch only."""

    with torch.no_grad(), set_vsa_forward_context(_vsa_metadata(sparsity)):
        return attention(hidden_states, freqs=freqs, gate_compress=torch.zeros_like(hidden_states))


def _inputs(seed: int, batch_size: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        batch_size, _SEQ_LEN, _HIDDEN_SIZE, generator=generator, device="cuda", dtype=torch.bfloat16
    )


@_REQUIRES_CUDA
@_BACKENDS
def test_vsa_backends_match_dense_at_sparsity_zero(backend: str) -> None:
    """VSA at sparsity 0 with the coarse gate closed selects every cube, so the fine branch
    is dense attention and must match the naive SDPA reference modulo bf16 rounding."""

    attention = _make_attention(backend, sparsity=0.0, seed=42)
    hidden_states = _inputs(seed=42, batch_size=2)
    freqs = _rope(_SEQ_LEN, _HEAD_DIM)

    out_vsa = _run(attention, hidden_states, freqs, sparsity=0.0)
    with torch.no_grad():
        out_dense = _dense_reference(attention, hidden_states, freqs)

    assert out_vsa.shape == out_dense.shape
    torch.testing.assert_close(out_vsa, out_dense, rtol=1e-2, atol=1e-2)


@_REQUIRES_SM100
def test_vsa_backends_match_each_other_on_ragged_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """CuTeDSL and TRTLLM implement the same sparse VSA fine-stage semantics.

    Both modules carry the same weights, both must run their sparse fine stage
    (the CuTe kernel and the PrimTS block-sparse FMHA respectively), and both
    outputs stay in the same neighbourhood of dense attention.
    """

    sparsity = 0.5
    attentions = {
        backend: _make_attention(backend, sparsity=sparsity, seed=0)
        for backend in ("CUTEDSL", "TRTLLM")
    }
    hidden_states = _inputs(seed=0, batch_size=1)
    freqs = _rope(_SEQ_LEN, _HEAD_DIM)

    sparse_fine_executed = {"CUTEDSL": 0, "TRTLLM": 0}
    cute_execute = VSACuTeDSLAttention._execute_sparse_fine
    primts_forward = PrimsTSBlockSparseFmha.forward

    def counted_cute(self, inputs):
        sparse_fine_executed["CUTEDSL"] += 1
        return cute_execute(self, inputs)

    def counted_primts(*args, **kwargs):
        sparse_fine_executed["TRTLLM"] += 1
        return primts_forward(*args, **kwargs)

    monkeypatch.setattr(VSACuTeDSLAttention, "_execute_sparse_fine", counted_cute)
    monkeypatch.setattr(PrimsTSBlockSparseFmha, "forward", counted_primts)

    outputs = {
        backend: _run(attention, hidden_states, freqs, sparsity)
        for backend, attention in attentions.items()
    }
    with torch.no_grad():
        dense = _dense_reference(attentions["TRTLLM"], hidden_states, freqs)

    assert sparse_fine_executed == {"CUTEDSL": 1, "TRTLLM": 1}
    for backend, output in outputs.items():
        assert torch.isfinite(output).all(), backend
        # Random weights spread attention mass evenly, so dropping half of the cubes
        # moves the output well away from dense (cosine similarity 0.85 measured on
        # B200); the bound only guards against a broken fine stage.
        cos_sim = F.cosine_similarity(
            output.float().reshape(-1), dense.float().reshape(-1), dim=0
        ).item()
        assert cos_sim >= 0.65, f"{backend} VSA at 50% sparsity drifted from dense: {cos_sim:.4f}"
    torch.testing.assert_close(outputs["CUTEDSL"], outputs["TRTLLM"], rtol=1e-2, atol=1e-2)


@_REQUIRES_SM100
def test_vsa_trtllm_cuda_graph_replays_live_routes() -> None:
    """Captured VSA recomputes routes when graph-stable input storage changes."""

    attention = _make_attention("TRTLLM", sparsity=0.5, seed=17)
    freqs = _rope(_SEQ_LEN, _HEAD_DIM)
    static_hidden = _inputs(seed=17, batch_size=1)
    static_gate = torch.zeros_like(static_hidden)
    metadata = _vsa_metadata(0.5)

    for _ in range(2):
        with torch.no_grad(), set_vsa_forward_context(metadata):
            attention(static_hidden, freqs=freqs, gate_compress=static_gate)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad(), set_vsa_forward_context(metadata):
        graph_output = attention(static_hidden, freqs=freqs, gate_compress=static_gate)

    initial_output = graph_output.clone()
    live_hidden = torch.randn_like(static_hidden)
    static_hidden.copy_(live_hidden)
    graph.replay()
    replay_output = graph_output.clone()
    with torch.no_grad(), set_vsa_forward_context(metadata):
        eager_output = attention(live_hidden, freqs=freqs, gate_compress=static_gate)

    assert not torch.equal(initial_output, replay_output)
    torch.testing.assert_close(replay_output, eager_output, rtol=1e-2, atol=1e-2)
