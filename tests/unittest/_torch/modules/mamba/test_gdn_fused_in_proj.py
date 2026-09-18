# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN mixer: one fused [qkvz | b | pad | a | pad] input projection GEMM.

The mixer keeps one weight for both column-parallel input projections. The
qkvz parameter aliases its leading row slab so in-place (re)loads land in the
fused buffer; the b and a gate rows are copied into two 32-byte-aligned column
blocks on every post_load_weights (checkpoint load, RL refit), so the decode
kernel never clones a misaligned gate slice. The forward runs a single GEMM
whose column slices feed the GDN kernels exactly like the two projections did."""
import pytest
import torch
from transformers import Qwen3NextConfig

import tensorrt_llm._torch.modules.mamba.gdn_mixer as gdn_mixer
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import copy_weight
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _make_mixer():
    config = Qwen3NextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        vocab_size=1024,
        torch_dtype=torch.bfloat16,
    )
    model_config = ModelConfig(pretrained_config=config)
    mixer = Qwen3NextGatedDeltaNet(model_config, aux_stream=torch.cuda.Stream(), layer_idx=0).cuda()
    torch.manual_seed(0)
    for lin in (mixer.in_proj_qkvz, mixer.in_proj_ba):
        copy_weight(lin.weight, torch.randn_like(lin.weight, dtype=torch.float32) * 0.02)
    mixer.post_load_weights()
    return mixer


def _projections(mixer, x):
    mixed_qkv, z, a, b = mixer._compute_tokenwise_inputs(x)
    return [t.float().clone() for t in (mixed_qkv, z, a, b)]


def _pair_path_projections(mixer, x, monkeypatch):
    """Reference: the two separate Linears (fused path switched off)."""
    with monkeypatch.context() as m:
        m.setattr(gdn_mixer, "_GDN_FUSED_IN_PROJ", False)
        mixer.cache_derived_state()
        assert mixer._in_proj_fused_weight is None
        ref = _projections(mixer, x)
    mixer.cache_derived_state()
    assert mixer._in_proj_fused_weight is not None
    return ref


def test_fused_projection_matches_the_two_linears(monkeypatch):
    mixer = _make_mixer()
    rows_q = mixer.in_proj_qkvz.weight.shape[0]
    n_gate = mixer.num_v_heads_per_tp
    fused = mixer._in_proj_fused_weight
    assert fused is not None and fused.shape[1] == mixer.hidden_size
    assert mixer.in_proj_qkvz.weight.data_ptr() == fused[:rows_q].data_ptr()
    # b and a live in 32-byte-aligned column blocks of the projection; the pad rows are zero
    b_col, a_col = mixer._in_proj_b_col, mixer._in_proj_a_col
    itemsize = fused.element_size()
    assert b_col >= rows_q and (b_col * itemsize) % 32 == 0 and (a_col * itemsize) % 32 == 0
    assert a_col >= b_col + n_gate and fused.shape[0] >= a_col + n_gate
    assert (fused.shape[0] * itemsize) % 32 == 0, "row stride keeps every token row aligned"
    assert torch.equal(fused[b_col:b_col + n_gate], mixer.in_proj_ba.weight[:n_gate])
    assert torch.equal(fused[a_col:a_col + n_gate], mixer.in_proj_ba.weight[n_gate:])
    pad = torch.ones(fused.shape[0], dtype=torch.bool, device=fused.device)
    pad[:rows_q] = False
    pad[b_col:b_col + n_gate] = False
    pad[a_col:a_col + n_gate] = False
    assert torch.count_nonzero(fused[pad]) == 0
    for num_tokens in (1, 8, 130):
        x = torch.randn(num_tokens, mixer.hidden_size, device="cuda", dtype=torch.bfloat16)
        ref = _pair_path_projections(mixer, x, monkeypatch)
        got = _projections(mixer, x)
        for name, r, g in zip(("mixed_qkv", "z", "a", "b"), ref, got):
            assert r.shape == g.shape, name
            torch.testing.assert_close(g, r, atol=2e-2, rtol=2e-2, msg=name)
        mixed_qkv, z, a, b = mixer._compute_tokenwise_inputs(x)
        assert a.data_ptr() % 32 == 0 and b.data_ptr() % 32 == 0, "gate slices aligned for the decode kernel"
        assert a[1:].data_ptr() % 32 == 0 if num_tokens > 1 else True
        assert a.stride(1) == 1 and b.stride(1) == 1


def test_in_place_weight_updates_reach_the_fused_weight():
    """qkvz: a refit that copies into the Linear parameter (copy_weight / NCCL
    update) is visible through the fused buffer without any refresh. ba: the
    gate rows are re-copied by post_load_weights, which every load and the RL
    refit (rlhf_utils.finalize_weight_update) run."""
    mixer = _make_mixer()
    rows_q = mixer.in_proj_qkvz.weight.shape[0]
    new_q = torch.randn_like(mixer.in_proj_qkvz.weight, dtype=torch.float32) * 0.05
    copy_weight(mixer.in_proj_qkvz.weight, new_q)
    assert torch.equal(mixer._in_proj_fused_weight[:rows_q], new_q.to(torch.bfloat16))
    new_ba = torch.randn_like(mixer.in_proj_ba.weight, dtype=torch.float32) * 0.05
    copy_weight(mixer.in_proj_ba.weight, new_ba)
    fused_before = mixer._in_proj_fused_weight
    mixer.post_load_weights()
    assert mixer._in_proj_fused_weight is fused_before, "same buffer: captured graphs keep a valid address"
    n = mixer.num_v_heads_per_tp
    b_col, a_col = mixer._in_proj_b_col, mixer._in_proj_a_col
    assert torch.equal(fused_before[b_col:b_col + n], new_ba[:n].to(torch.bfloat16))
    assert torch.equal(fused_before[a_col:a_col + n], new_ba[n:].to(torch.bfloat16))
    x = torch.randn(4, mixer.hidden_size, device="cuda", dtype=torch.bfloat16)
    _, _, a, b = mixer._compute_tokenwise_inputs(x)
    ref = torch.nn.functional.linear(x, new_ba.to(torch.bfloat16)).float()
    torch.testing.assert_close(b.float(), ref[:, :n], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(a.float(), ref[:, n:], atol=2e-2, rtol=2e-2)


def test_replaced_parameter_storage_is_re_aliased_on_post_load():
    mixer = _make_mixer()
    rows_q = mixer.in_proj_qkvz.weight.shape[0]
    fused_before = mixer._in_proj_fused_weight
    fresh = torch.randn_like(mixer.in_proj_qkvz.weight) * 0.03
    mixer.in_proj_qkvz.weight.data = fresh  # new storage, as a weight re-creation would do
    assert mixer.in_proj_qkvz.weight.data_ptr() != fused_before[:rows_q].data_ptr()
    mixer.post_load_weights()
    fused = mixer._in_proj_fused_weight
    assert fused is fused_before  # same buffer, so captured graphs keep a valid address
    assert torch.equal(fused[:rows_q], fresh)
    assert mixer.in_proj_qkvz.weight.data_ptr() == fused[:rows_q].data_ptr()


def test_ineligible_projections_fall_back_to_the_pair(monkeypatch):
    mixer = _make_mixer()
    mixer.in_proj_ba.bias = torch.nn.Parameter(
        torch.zeros(mixer.in_proj_ba.weight.shape[0], device="cuda", dtype=torch.bfloat16))
    mixer.cache_derived_state()
    assert mixer._in_proj_fused_weight is None
    x = torch.randn(3, mixer.hidden_size, device="cuda", dtype=torch.bfloat16)
    mixed_qkv, z, a, b = mixer._compute_tokenwise_inputs(x)
    assert mixed_qkv.shape == (3, mixer.conv_dim_per_tp) and b.shape == (3, mixer.num_v_heads_per_tp)
    mixer.in_proj_ba.bias = None
    mixer.cache_derived_state()
    assert mixer._in_proj_fused_weight is not None
    with monkeypatch.context() as m:
        m.setattr(gdn_mixer, "_GDN_FUSED_IN_PROJ", False)
        mixer.cache_derived_state()
        assert mixer._in_proj_fused_weight is None
