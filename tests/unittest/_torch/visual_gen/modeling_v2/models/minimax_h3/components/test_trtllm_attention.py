# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.trtllm_attention import (
    H3TrtllmAttention,
)
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="H3 TRTLLM attention uses SM100 or SM103",
)


def test_h3_uneven_padded_prefix():
    torch.manual_seed(42)
    seq, valid, heads = 65, 37, 4
    q, k, v = [
        torch.randn(2, seq, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    v[:, valid:].fill_(100)
    attn = H3TrtllmAttention(num_heads=heads, head_dim=128, attention_metadata_state={})
    actual = attn.forward(
        q, k, v, batch_size=2, seq_len=seq, discard_padded_queries=True, valid_seq_len=valid
    )
    expected = (
        F.scaled_dot_product_attention(*[x[:, :valid].transpose(1, 2).float() for x in (q, k, v)])
        .transpose(1, 2)
        .flatten(2)
    )
    torch.testing.assert_close(actual[:, :valid].float(), expected, rtol=0.02, atol=0.02)
    assert torch.count_nonzero(actual[:, valid:]) == 0


def test_h3_sage_key_smoothing():
    torch.manual_seed(42)
    batch, seq, heads, dim = 1, 257, 4, 128
    q, k, v = [
        torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    # A common offset changes quantization error while preserving the ideal attention weights.
    k += 32 * torch.randn((1, 1, heads, dim), device=k.device, dtype=k.dtype)
    cfg = QuantAttentionConfig(
        qk_dtype="fp8",
        v_dtype="fp8",
        q_block_size=1,
        k_block_size=16,
        v_block_size=1,
        smooth_k=True,
    )
    args = dict(num_heads=heads, num_kv_heads=heads, head_dim=dim, dtype=q.dtype)
    inputs = dict(batch_size=batch, seq_len=seq)
    candidate = H3TrtllmAttention(
        **args,
        attention_metadata_state={},
        attention_config=AttentionConfig(backend="TRTLLM", quant_attention_config=cfg),
    )
    reference = TrtllmAttention(**args, attention_metadata_state={}, quant_attention_config=cfg)
    expected = reference.forward(q, k, v, **inputs)
    actual = candidate.forward(q, k, v, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference.quant_attention_config = cfg.model_copy(update={"smooth_k": False})
    assert not torch.equal(expected, reference.forward(q, k, v, **inputs))


def _attention_pair(config, num_heads=1, *, v1_sparse=None, v2_sparse=None):
    options = dict(num_heads=num_heads, head_dim=128, dtype=torch.bfloat16)
    v1 = TrtllmAttention(
        **options,
        quant_attention_config=config.quant_attention_config,
        sparse_params=v1_sparse,
        attention_metadata_state={},
    )
    v2 = H3TrtllmAttention(
        **options, attention_config=config, sparse_params=v2_sparse, attention_metadata_state={}
    )
    return v1, v2


def test_h3_primts_unavailable_keeps_native_sage_fallback(monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha
    from tensorrt_llm._torch.attention.backends.sparse.params import (
        BlockSparseForwardInputs,
        SparseBackendForwardArgs,
    )
    from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.prims_routed_attention import (
        PrimsRoutes,
    )
    from tensorrt_llm.bindings.internal import thop

    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    # The PrimTS availability probe requires this binding. Native context attention uses run_context.
    monkeypatch.delattr(thop, "trtllm_gen_generation_preprocess", raising=False)
    recipe = QuantAttentionConfig(
        qk_dtype="fp8",
        v_dtype="fp8",
        q_block_size=1,
        k_block_size=16,
        v_block_size=1,
        smooth_v=True,
    )
    v1, v2 = _attention_pair(AttentionConfig(backend="TRTLLM", quant_attention_config=recipe))
    assert not any(isinstance(fmha, PrimsTSFmha) for fmha in v1._fmha_manager.fmha_libs)
    rng = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = [
        torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16, generator=rng)
        for _ in range(3)
    ]
    inputs = dict(batch_size=1, seq_len=128)
    expected = v1.forward(q, k, v, **inputs)
    actual = v2.forward(q, k, v, **inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert v2.state["h3_thop"]
    routes = PrimsRoutes(
        q_block_size=128,
        kv_block_size=128,
        max_blocks_per_row=1,
        block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32, device=q.device),
        block_indices=torch.tensor([0], dtype=torch.int32, device=q.device),
    )
    with pytest.raises(RuntimeError, match="No TRT-LLM attention FMHA library"):
        v1.forward(
            q,
            k,
            v,
            **inputs,
            sparse_backend_args=SparseBackendForwardArgs(
                block_sparse_inputs=BlockSparseForwardInputs(**routes)
            ),
        )
    with pytest.raises(RuntimeError, match="available PrimTS block-sparse"):
        v2.forward(q, k, v, **inputs, routes=routes)


@pytest.mark.parametrize("qk_dtype", ["bf16", "fp8"])
def test_h3_per_tensor_fp8_dense_and_routed(monkeypatch, qk_dtype):
    from tensorrt_llm._torch.attention.backends.sparse.params import (
        BlockSparseForwardInputs,
        SparseBackendForwardArgs,
    )
    from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.prims_routed_attention import (
        PrimsRoutes,
    )

    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    recipe = QuantAttentionConfig(qk_dtype=qk_dtype, v_dtype="fp8")
    v1, v2 = _attention_pair(AttentionConfig(backend="TRTLLM", quant_attention_config=recipe))
    rng = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = [
        torch.randn(1, 256, 1, 128, device="cuda", dtype=torch.bfloat16, generator=rng)
        for _ in range(3)
    ]
    inputs = dict(batch_size=1, seq_len=256)
    expected = v1.forward(q, k, v, **inputs)
    torch.testing.assert_close(v2.forward(q, k, v, **inputs), expected, rtol=0, atol=0)
    routes = PrimsRoutes(
        q_block_size=128,
        kv_block_size=128,
        max_blocks_per_row=1,
        block_indptr=torch.tensor([[[0, 1, 2]]], dtype=torch.int32, device=q.device),
        block_indices=torch.tensor([0, 0], dtype=torch.int32, device=q.device),
    )
    v1_routes = SparseBackendForwardArgs(block_sparse_inputs=BlockSparseForwardInputs(**routes))
    routed = v1.forward(q, k, v, **inputs, sparse_backend_args=v1_routes)
    torch.testing.assert_close(v2.forward(q, k, v, **inputs, routes=routes), routed, rtol=0, atol=0)
    assert not torch.equal(routed, expected)


@pytest.mark.parametrize("qk_dtype", ["bf16", "fp8"])
def test_h3_vc_repair_matches_v1(monkeypatch, qk_dtype):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    recipe = QuantAttentionConfig(
        qk_dtype=qk_dtype, v_dtype="fp8", vc_attention=True, vc_repair_budget=0.25
    )
    v1, v2 = _attention_pair(AttentionConfig(backend="TRTLLM", quant_attention_config=recipe), 2)
    for length in (256, 265):
        q = torch.zeros(1, length, 2, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.ones_like(q)
        expected = v1.forward(q, q, v, batch_size=1, seq_len=length)
        actual = v2.forward(q, q, v, batch_size=1, seq_len=length)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, v.flatten(2), rtol=0, atol=0)


@pytest.mark.parametrize("skip_first", [False, True])
def test_h3_sage_smoothing_with_skip_softmax_phases(monkeypatch, skip_first):
    from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.skip_softmax_threshold import (
        H3SkipSoftmax,
    )
    from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig

    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts_block_sparse,fallback")
    recipe = QuantAttentionConfig(
        qk_dtype="fp8",
        v_dtype="fp8",
        q_block_size=1,
        k_block_size=16,
        v_block_size=1,
        smooth_v=True,
    )
    skip = SkipSoftmaxAttentionConfig(threshold_scale_factor=8.0, disabled_until_timestep=0.6)
    config = AttentionConfig(
        backend="TRTLLM", quant_attention_config=recipe, sparse_attention_config=skip
    )
    v1, v2 = _attention_pair(
        config, v1_sparse=skip.to_sparse_params(), v2_sparse=H3SkipSoftmax(8.0, 0.6)
    )
    q = torch.zeros(1, 512, 1, 128, dtype=torch.bfloat16, device="cuda")
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    q[..., 0] = 128**0.5
    k[:, 128:256, :, 0], k[:, 256:384, :, 0], k[:, 384:, :, 0] = -4, -5, -9
    v[:, 128:256], v[:, 256:384], v[:, 384:] = 1, 8, 64
    results = {}
    for active in (skip_first, not skip_first, skip_first):
        timestep = torch.tensor([0.4 if active else 0.8], device=q.device)
        inputs = dict(batch_size=1, seq_len=512, timestep=timestep)
        expected = v1.forward(q, k, v, **inputs)
        actual = v2.forward(q, k, v, **inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        results[active] = actual
    assert not torch.equal(results[True], results[False])


def test_sol_trtllm_predictor_and_dense_prefix():
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.sol_predictor import (
        sol_predictor,
    )
    from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.dense_attention import (
        create_h3_attention,
    )
    from tensorrt_llm.visual_gen.sparse_attention import SolAttentionConfig

    sparse = SolAttentionConfig(tau=0.5, thresh_type="exact", disabled_until_timestep=0.6)
    config = DiffusionModelConfig(
        attention=AttentionConfig(backend="TRTLLM", sparse_attention_config=sparse),
        attention_metadata_state={},
    )
    attention = create_h3_attention(
        config, num_heads=2, head_dim=128, layer_idx=1, module_name="attn"
    )
    rng = torch.Generator(device="cuda").manual_seed(39)
    q, k, v = [
        (torch.randint(-16, 17, (1, 321, 2, 128), device="cuda", generator=rng) / 16).bfloat16()
        for _ in range(3)
    ]
    assert attention._sol_routes(q, k, v, 0.6) is None
    routes = attention._sol_routes(q, k, v, 0.5)
    expected = sol_predictor.reference(q, k, v, tau=0.5, sm_scale=128**-0.5, thresh_type="exact")
    for actual, reference in zip(
        (routes["exact_block_bits"], routes["k_summary"], routes["v_summary"]),
        expected,
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
