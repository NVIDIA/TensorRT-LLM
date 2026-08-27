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
"""Sol-Attn correctness tests: backend dispatch, config guards, step-context
lifecycle, dense_layers/dense_steps guards.

Mirrors test_attention_cute_dsl_vsa.py's structure and scope for its sibling
sparse-attention algorithm. GPU kernel-vs-dense numerical equivalence (the
analogue of VSA's test_cute_kernel_matches_dense_at_full_topk) is not yet
covered here -- see the TODO on test_cute_kernel_matches_dense_placeholder
below for what it needs and why it's deferred, not just missing.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn import (
    SolAttnAttention,
    SolAttnStepContext,
    _parse_dense_layers,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen.args import AttentionConfig, SolAttnAttentionConfig


@pytest.fixture(autouse=True)
def _reset_sol_attn_step_context():
    """SolAttnStepContext is process-wide mutable state; isolate every test."""
    SolAttnStepContext.reset()
    yield
    SolAttnStepContext.reset()


def test_cute_dsl_factory_dispatches_dense_and_sol_attn() -> None:
    dense_config = AttentionConfig(backend="CUTEDSL")
    dense_attention = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=dense_config,
    )

    sparse_config = SolAttnAttentionConfig(tau=2.0, dense_steps=10)
    sol_attn_config = AttentionConfig(backend="CUTEDSL", sparse_attention_config=sparse_config)
    sol_attn_attention = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=sol_attn_config,
    )

    assert isinstance(dense_attention, CuTeDSLAttention)
    assert isinstance(sol_attn_attention, SolAttnAttention)
    assert sol_attn_attention.tau == 2.0
    assert sol_attn_attention.dense_steps == 10


def _make_config(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    backend: str,
    sol_attn_tau: "float | None" = None,
) -> DiffusionModelConfig:
    """Minimal DiffusionModelConfig for one Attention module."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=1e-6,
    )
    sparse_attention_config = (
        SolAttnAttentionConfig(tau=sol_attn_tau) if sol_attn_tau is not None else None
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(backend=backend, sparse_attention_config=sparse_attention_config),
        skip_create_weights_in_init=False,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    return config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Sol-Attn needs CUDA")
def test_sol_attn_falls_back_to_vanilla_for_cross_attention():
    """Cross-attention (SEPARATE_QKV) falls back to VANILLA -- Sol-Attn is self-attn only."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = _make_config(
        hidden_size=64, num_heads=4, head_dim=16, backend="CUTEDSL", sol_attn_tau=1.0
    )
    cross_attn = (
        Attention(64, 4, qkv_mode=QKVMode.SEPARATE_QKV, config=cfg)
        .to(device=device, dtype=dtype)
        .eval()
    )
    assert cross_attn.attn_backend == "VANILLA", (
        f"Sol-Attn on cross-attention should fall back to VANILLA, got {cross_attn.attn_backend!r}"
    )


def test_sol_attn_with_context_parallelism_raises():
    """Sol-Attn + Attention2D/Ring must error at construction (needs the full sequence per rank)."""
    pretrained_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        attention_head_dim=16,
        eps=1e-6,
    )
    cfg = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=SolAttnAttentionConfig(tau=1.0),
        ),
        skip_create_weights_in_init=False,
    )
    cfg.visual_gen_mapping = SimpleNamespace(
        ring_size=1,
        ring_group=None,
        ulysses_size=1,
        ulysses_group=None,
        attn2d_row_size=2,
        attn2d_col_size=2,
        attn2d_row_group=None,
        attn2d_col_group=None,
        cp_size=4,
    )
    with pytest.raises(ValueError, match="incompatible with context parallelism"):
        Attention(64, 4, qkv_mode=QKVMode.FUSE_QKV, config=cfg)


def test_sol_attn_rejects_gqa_mqa():
    """Sol-Attn is MHA-only; num_kv_heads != num_heads must fail fast at construction."""
    with pytest.raises(AssertionError, match="MHA-only"):
        SolAttnAttention(layer_idx=0, num_heads=8, head_dim=128, num_kv_heads=2)


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, frozenset()),
        ("", frozenset()),
        ("0", frozenset({0})),
        ("0,2,4", frozenset({0, 2, 4})),
        ("0-3", frozenset({0, 1, 2, 3})),
        ("0-1,5,7-8", frozenset({0, 1, 5, 7, 8})),
        (" 0 , 2 ", frozenset({0, 2})),
    ],
    ids=["none", "empty", "single", "list", "range", "mixed", "whitespace"],
)
def test_parse_dense_layers(spec, expected):
    assert _parse_dense_layers(spec) == expected


def test_sol_attn_step_context_lifecycle():
    """advance_step/reset/current_step/is_advancing behave as SolAttnAttention.forward relies on."""
    assert SolAttnStepContext.current_step() == -1
    assert not SolAttnStepContext.is_advancing()

    SolAttnStepContext.advance_step()
    assert SolAttnStepContext.current_step() == 0
    assert SolAttnStepContext.is_advancing()

    SolAttnStepContext.advance_step()
    assert SolAttnStepContext.current_step() == 1

    SolAttnStepContext.reset()
    assert SolAttnStepContext.current_step() == -1
    assert not SolAttnStepContext.is_advancing()


def test_sol_attn_dense_steps_without_advancing_raises():
    """dense_steps > 0 on a pipeline that never calls advance_step() must hard-error, not
    silently run dense forever (the failure mode SolAttnStepContext.forward() guards against)."""
    attn = SolAttnAttention(layer_idx=0, num_heads=2, head_dim=16)
    attn.dense_steps = 5
    q = k = v = torch.randn(1, 4, 2, 16)
    with pytest.raises(RuntimeError, match="never has"):
        attn.forward(q, k, v)


def test_sol_attn_dense_layers_guard_skips_kernel(monkeypatch):
    """A layer_idx in dense_layers must use the dense SDPA path and never invoke the kernel."""
    import tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn as sol_attn_mod

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("kernel must not be invoked for a dense_layers-forced layer")

    monkeypatch.setattr(sol_attn_mod, "_sol_attn_run", _fail_if_called)

    attn = SolAttnAttention(layer_idx=3, num_heads=2, head_dim=16)
    attn.dense_layers = frozenset({3})
    q = k = v = torch.randn(1, 4, 2, 16)
    out = attn.forward(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skip(
    reason=(
        "TODO(sol-attn): numerical equivalence vs dense SDPA at zero/near-zero routing "
        "(the analogue of VSA's test_cute_kernel_matches_dense_at_full_topk) needs (a) "
        "reading cute_dsl_kernels/blackwell/sol_attn/interface.py to find the exact "
        "tau/thresh_type combination that guarantees full (non-sparse) block routing, "
        "since Sol-Attn's routing is score-derived rather than a plain top-k like VSA's, "
        "and (b) real sm90/sm100 Blackwell hardware to calibrate rtol/atol, both out of "
        "scope for this pass. Real-world quality at production settings is already "
        "covered by the LPIPS-gated evidence in the A14B/5B sparse-attn comparison "
        "reports (docs/sparse-attn-solattn-vs-skipsoftmax/), which this test would "
        "complement, not replace."
    )
)
def test_cute_kernel_matches_dense_placeholder():
    pass
