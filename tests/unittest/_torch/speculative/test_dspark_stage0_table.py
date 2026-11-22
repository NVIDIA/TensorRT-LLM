# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark stage-0 token table: the gathered stage-0 attention inputs and entry coefficients equal,
bit for bit and in CUDA-graph replays, what an attention-TP rank's stage 0 computes for one
request without the table: draft block inputs, entry boundary, ``dspark_attention_forward_tp``."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100/SM103 with CuTe DSL required",
)

VOCAB, HIDDEN, HC, Q_RANK, HEADS, HEAD_DIM, ROPE, BLOCK = 3000, 5120, 4, 1280, 8, 512, 64, 5


def _fp8(gen, rows, cols):
    """E4M3 weights with linear UE8M0 scales (one per 32 inputs)."""
    weight = (torch.randn(rows, cols, device="cuda", generator=gen) * 64).to(torch.float8_e4m3fn)
    scale = torch.randint(112, 120, (rows, cols // 32), device="cuda", generator=gen)
    return weight, scale.to(torch.uint8)


def _weights(gen, coefficients=None):
    from tensorrt_llm._torch.models.dspark_stage0_table import Stage0Weights

    qkv, qkv_scale = _fp8(gen, Q_RANK + HEAD_DIM, HIDDEN)
    wq_b, wq_b_scale = _fp8(gen, HEADS * HEAD_DIM, Q_RANK)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, device="cuda", generator=gen)).bfloat16()

    return Stage0Weights(
        embed=(torch.randn(VOCAB, HIDDEN, device="cuda", generator=gen) * 0.05).bfloat16(),
        hc_mult=HC,
        entry_norm_w=norm(HIDDEN),
        entry_norm_eps=1e-6,
        qkv_weight=qkv,
        qkv_scale=qkv_scale,
        q_rank=Q_RANK,
        q_norm_w=norm(Q_RANK),
        kv_norm_w=norm(HEAD_DIM),
        eps=1e-6,
        wq_b_weight=wq_b,
        wq_b_scale=wq_b_scale,
        heads=HEADS,
        head_dim=HEAD_DIM,
        rope_dim=ROPE,
        coefficients=coefficients,
    )


class _AttentionInputs(Exception):
    """Stops ``dspark_attention_forward_tp`` once its attention inputs are formed."""


def _production_stage0(monkeypatch, w, hc, freqs_cis, noise, block, token, start):
    """``(q, draft block, step, entry coefficients)`` of a one-request step's stage 0 as the draft
    computes it without the table: block inputs, entry boundary, then the attention's inputs."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.kernels.blackwell.dspark.generation_prepare import (
        dspark_draft_block_inputs,
    )
    from tensorrt_llm._torch.models import modeling_dspark

    bonus = torch.tensor([token], dtype=torch.int64, device="cuda")
    start_pos = torch.tensor([start], dtype=torch.int64, device="cuda")
    x, _, (_, blk_freqs), pre_mix = dspark_draft_block_inputs(
        bonus,
        start_pos,
        w.embed,
        freqs_cis,
        block_size=block,
        noise_token_id=noise,
        hc_mult=HC,
        pre_mix=True,
    )
    _, pre, post, comb, layer_input, _ = hc.split_hc_lagged(
        None, x, None, None, pre_mix, w.entry_norm_w, w.entry_norm_eps
    )
    inverse = torch.view_as_real(blk_freqs)
    step = modeling_dspark.DSparkTPAttentionStep(
        main_rope_freqs=None,
        block_freqs=blk_freqs,
        positions=torch.arange(block, dtype=torch.int32, device="cuda"),
        rope_table=inverse.view(block, ROPE // 2, 2).transpose(1, 2).contiguous(),
        inverse_rope_freqs=inverse,
        block_rope_freqs=inverse.view(block, ROPE // 2, 2),
    )
    attn = SimpleNamespace(
        qk_head_dim=HEAD_DIM,
        num_heads_tp=HEADS,
        wq_b=SimpleNamespace(weight=w.wq_b_weight, _lowm_scale_lin=w.wq_b_scale),
        attn_sink=torch.zeros(HEADS, device="cuda"),
    )
    inputs = {}

    def attention(q, draft_block, *args, **kwargs):
        inputs.update(q=q, draft_block=draft_block)
        raise _AttentionInputs

    with monkeypatch.context() as patch, pytest.raises(_AttentionInputs):
        patch.setattr(modeling_dspark, "fused_dsv4_dspark_attention", attention)
        modeling_dspark.dspark_attention_forward_tp(
            layer_input.view(1, block, HIDDEN),
            layer_input,
            start_pos,
            None,
            bonus,
            None,
            step,
            attn=attn,
            qkv=(w.qkv_weight, w.qkv_scale, Q_RANK),
            wo_a_scale=None,
            wkv=None,
            q_norm_w=w.q_norm_w,
            kv_norm_w=w.kv_norm_w,
            eps=w.eps,
            softmax_scale=1.0,
            # Skips the window metadata and the main-KV frame write; stage 0's inputs read neither.
            ring_attention_metadata=(None,) * 4,
            main_kv=(bonus, start_pos),
        )
    return inputs["q"], inputs["draft_block"], step, (pre, post, comb)


# Block 8 fills in chunks of the small-M kernels' 64 rows.
@pytest.mark.parametrize("block", [BLOCK, 8])
@torch.inference_mode()
def test_stage0_table_gather_matches_production_stage0(monkeypatch, block):
    from tensorrt_llm._torch.models.dspark_stage0_table import (
        Stage0Table,
        check_stage0_table,
        fill_stage0_table,
        stage0_gather,
    )
    from tensorrt_llm._torch.models.modeling_dspark import precompute_dspark_freqs_cis
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

    gen = torch.Generator(device="cuda").manual_seed(7)
    hc = mHC(
        mult=HC,
        hidden_size=HIDDEN,
        sinkhorn_iters=20,
        dtype=torch.bfloat16,
        eps=1e-6,
        norm_eps=1e-6,
        sinkhorn_eps=1e-6,
        post_mult_value=2.0,
    ).cuda()
    hc.fn.copy_(torch.randn(hc.fn.shape, device="cuda", generator=gen) * 1e-3)
    hc.base.copy_(torch.randn(hc.base.shape, device="cuda", generator=gen) * 0.1)
    hc.scale.copy_(torch.randn(hc.scale.shape, device="cuda", generator=gen) * 0.1)
    hc.post_load_weights()
    w = _weights(gen, coefficients=hc.lagged_coefficients)
    freqs_cis = precompute_dspark_freqs_cis(ROPE, 4096, device="cuda")
    noise = VOCAB - 3
    table = Stage0Table(VOCAB, HEADS, HEAD_DIM, ROPE, "cuda", hc_mult=HC)
    fill_stage0_table(table, w, block)
    # Every bonus token of the sample (and the noise token) equals the live chain at random
    # positions, which also picks each RoPE contraction.
    assert check_stage0_table(table, w, freqs_cis, noise, block) and table.coeff_ready

    bonus = torch.zeros(1, dtype=torch.int64, device="cuda")
    block_freqs = torch.zeros(block, ROPE // 2, 2, device="cuda")

    def gather():
        return stage0_gather(table, bonus, block_freqs, noise, block)

    gather()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, draft_block, rope_table, coeffs = gather()
    for token, start in ((17, 300), (VOCAB - 1, 9), (-1, 4000), (noise, 77)):
        want_q, want_block, step, want_coeffs = _production_stage0(
            monkeypatch, w, hc, freqs_cis, noise, block, token, start
        )
        bonus.fill_(token)
        block_freqs.copy_(step.block_rope_freqs)
        graph.replay()
        assert torch.equal(q.view(torch.int16), want_q.reshape(q.shape).view(torch.int16))
        assert torch.equal(draft_block.view(torch.int16), want_block.view(torch.int16))
        assert torch.equal(rope_table, step.rope_table)
        for got, want in zip(coeffs, want_coeffs):
            assert torch.equal(
                got.reshape(-1).view(torch.int32), want.reshape(-1).view(torch.int32)
            )
