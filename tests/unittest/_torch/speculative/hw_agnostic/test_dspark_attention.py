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
"""Unit tests for the DSpark captured-context attention primitives (CPU).

The two primitives are validated against fully independent computations:
``dspark_sparse_attn`` vs (a) a per-element loop reference of the kernel formula
and (b) ``torch.nn.functional.scaled_dot_product_attention`` for the no-sink,
all-valid case; ``get_dspark_topk_idxs`` vs the reference index formula.
"""

import types
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm._torch.models.modeling_dspark as modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    DSv4DSparkDraftModel,
    _output_projection,
    _qkv_projections,
    apply_dspark_rotary,
    dspark_attention_forward,
    dspark_sparse_attn,
    get_dspark_topk_idxs,
    precompute_dspark_freqs_cis,
)
from tensorrt_llm._torch.modules.linear import Linear

# The captured-context attention primitives were folded into
# modeling_dspark; keep the historical alias so monkeypatch targets
# below read unchanged.
dspark_attention = modeling_dspark


def test_rmsnorm_rope_fallback_applies_weight_without_rmsnorm(monkeypatch):
    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_AVAILABLE", False)
    torch.manual_seed(11)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    weight = torch.randn(64, dtype=torch.bfloat16)
    freqs_cis = torch.empty(2, 3, 1, dtype=torch.complex64)

    actual = dspark_attention._rmsnorm_rope_batched(
        x,
        weight,
        1e-6,
        0,
        freqs_cis,
        apply_weight=True,
        apply_rmsnorm=False,
    )

    expected = (x.float() * weight.float()).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rmsnorm_rope_skips_frequency_view_on_unsupported_arch(monkeypatch):
    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(dspark_attention, "is_sm_100f", lambda: False)
    view_as_real = Mock(side_effect=AssertionError("frequency view should be skipped"))
    monkeypatch.setattr(torch, "view_as_real", view_as_real)
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)

    actual = dspark_attention._rmsnorm_rope_batched(
        x,
        torch.ones(64, dtype=torch.bfloat16),
        1e-6,
        0,
        torch.empty(2, 3, 1, dtype=torch.complex64),
        apply_weight=False,
        apply_rmsnorm=False,
    )

    assert actual is x
    view_as_real.assert_not_called()


def test_batched_attention_rejects_mismatched_window_size():
    unused_tensor = torch.empty(0)
    with pytest.raises(ValueError, match="does not match window_size"):
        dspark_attention.dspark_attention_forward_batched(
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 1),
            torch.zeros(1, dtype=torch.long),
            torch.empty(1, 2, 1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            wq_a=unused_tensor,
            q_norm_w=unused_tensor,
            wq_b=unused_tensor,
            wkv=unused_tensor,
            kv_norm_w=unused_tensor,
            wo_a=unused_tensor,
            wo_b=unused_tensor,
            attn_sink=unused_tensor,
            n_heads=1,
            head_dim=1,
            rope_head_dim=0,
            n_groups=1,
            o_lora_rank=1,
            window_size=3,
            eps=1e-6,
            softmax_scale=1.0,
            freqs_cis=unused_tensor,
        )


@pytest.mark.parametrize(
    "value,tp_size,enable_attention_dp,expected",
    [
        (None, 1, False, False),
        (None, 2, True, False),
        (None, 2, False, True),
        ("fp8", 2, False, True),
        ("bf16", 1, False, True),
    ],
)
def test_use_dspark_bf16_linear(monkeypatch, value, tp_size, enable_attention_dp, expected):
    if value is None:
        monkeypatch.delenv("TRTLLM_DSPARK_LINEAR_BACKEND", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DSPARK_LINEAR_BACKEND", value)

    mapping = types.SimpleNamespace(
        tp_size=tp_size,
        enable_attention_dp=enable_attention_dp,
    )
    assert modeling_dspark._use_dspark_bf16_linear(mapping) is expected


def test_use_dspark_bf16_linear_rejects_invalid_backend(monkeypatch):
    monkeypatch.setenv("TRTLLM_DSPARK_LINEAR_BACKEND", "auto")
    mapping = types.SimpleNamespace(tp_size=1, enable_attention_dp=False)

    with pytest.raises(ValueError, match="must be 'fp8' or 'bf16'"):
        modeling_dspark._use_dspark_bf16_linear(mapping)


def test_fused_qkv_projection_groups_rows_without_copy() -> None:
    batch, block, hidden = 2, 3, 4
    q_rank, kv_rank = 6, 2
    x = torch.arange(batch * block * hidden, dtype=torch.float32).view(batch, block, hidden)
    main_x = -torch.arange(batch * hidden, dtype=torch.float32).view(batch, 1, hidden)
    weight = torch.arange((q_rank + kv_rank) * hidden, dtype=torch.float32).view(
        q_rank + kv_rank, hidden
    )
    projected_inputs: list[torch.Tensor] = []

    def qkv_proj(rows: torch.Tensor) -> torch.Tensor:
        projected_inputs.append(rows.clone())
        return F.linear(rows, weight)

    q_latent, kv, main_kv = _qkv_projections(
        x,
        main_x,
        wq_a=None,
        wkv=None,
        q_norm_w=torch.empty(q_rank),
        qkv_proj=qkv_proj,
    )

    assert len(projected_inputs) == 1
    expected_rows = torch.cat((main_x.view(-1, hidden), x.view(-1, hidden)), dim=0)
    torch.testing.assert_close(projected_inputs[0], expected_rows)

    expected_main = F.linear(main_x, weight)
    expected_block = F.linear(x, weight)
    torch.testing.assert_close(q_latent, expected_block[..., :q_rank])
    torch.testing.assert_close(kv, expected_block[..., q_rank:])
    torch.testing.assert_close(main_kv, expected_main[..., q_rank:])

    for tensor in (q_latent, kv, main_kv):
        flat = tensor.view(-1, tensor.shape[-1])
        assert flat.stride(-1) == 1
        assert flat.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


def test_fused_fp8_input_producers_feed_prequantized_linears(monkeypatch) -> None:
    import tensorrt_llm._torch.models.modeling_dspark as dspark_attention

    batch, block, hidden = 2, 3, 4
    q_rank, kv_rank, q_out = 8, 2, 12
    x = torch.randn(batch, block, hidden)
    main_x = torch.randn(batch, 1, hidden)
    concat_input = (object(), object())
    norm_input = (object(), object())
    projection_inputs = []
    prevalidated_calls = []

    def prevalidated_projection(input, projection):
        prevalidated_calls.append((input, projection))
        return projection(input)

    monkeypatch.setattr(dspark_attention, "_prequantized_fp8_projection", prevalidated_projection)

    def concat_quant(first, second):
        assert first.shape == (batch, hidden)
        assert second.shape == (batch * block, hidden)
        return concat_input

    def qkv_proj(input):
        projection_inputs.append(input)
        return torch.randn(batch * (block + 1), q_rank + kv_rank)

    monkeypatch.setattr(dspark_attention, "cute_dsl_dspark_concat_fp8_quant", concat_quant)
    q_latent, _, _ = _qkv_projections(
        x,
        main_x,
        wq_a=None,
        wkv=None,
        q_norm_w=torch.empty(q_rank),
        qkv_proj=qkv_proj,
        use_fused_fp8_input=True,
    )
    assert projection_inputs == [concat_input]
    assert prevalidated_calls == [(concat_input, qkv_proj)]

    def rmsnorm_quant(input, weight, eps):
        assert input is q_latent
        assert weight.shape == (q_rank,)
        assert eps == 1e-6
        return norm_input

    def q_b_proj(input):
        projection_inputs.append(input)
        return torch.randn(batch * block, q_out)

    monkeypatch.setattr(dspark_attention, "cute_dsl_dspark_rmsnorm_fp8_quant", rmsnorm_quant)
    q = dspark_attention._q_projection(q_latent, torch.ones(q_rank), 1e-6, q_b_proj, True)
    assert projection_inputs[-1] == norm_input
    assert prevalidated_calls[-1] == (norm_input, q_b_proj)
    assert q.shape == (batch, block, q_out)


def test_output_projection_uses_fused_fp8_path(monkeypatch):
    import tensorrt_llm._torch.models.modeling_dspark as dspark_attention

    batch, block, groups, rank, group_dim, hidden = 2, 3, 2, 128, 16, 24
    o = torch.randn(batch, block, groups, group_dim, dtype=torch.bfloat16)
    o_a_proj = torch.empty(groups, rank, group_dim, dtype=torch.float8_e4m3fn)
    o_a_scale = object()
    calls = []
    prevalidated_calls = []

    def prevalidated_projection(input, projection):
        prevalidated_calls.append((input, projection))
        return projection(input)

    monkeypatch.setattr(dspark_attention, "_prequantized_fp8_projection", prevalidated_projection)

    class FakeOutputLinear:
        has_fp8_block_scales = True
        use_cute_dsl_blockscaling_mm = True
        disable_deep_gemm = False
        in_features = groups * rank
        weight = torch.empty(hidden, groups * rank, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(1, dtype=torch.uint8)

        def __call__(self, input):
            calls.append(input)
            return torch.zeros(batch * block, hidden, dtype=torch.bfloat16)

    def fake_quantize(input):
        assert input.shape == (batch * block, groups, group_dim)
        return input.permute(1, 0, 2).contiguous().to(torch.float8_e4m3fn), object()

    def fake_bmm_quantize(
        input,
        weight,
        input_scale,
        weight_scale,
        output,
        packed_scale,
        use_tvm_ffi=True,
        dspark_max_batch=0,
        dspark_rows_per_request=0,
    ):
        assert input.shape == (groups, batch * block, group_dim)
        assert weight is o_a_proj
        assert input_scale is not None
        assert weight_scale is o_a_scale
        assert output.shape == (groups, batch * block, rank)
        assert packed_scale.dtype == torch.uint8
        assert use_tvm_ffi is True
        assert dspark_max_batch == 0
        assert dspark_rows_per_request == 0
        output.zero_()
        packed_scale.zero_()

    monkeypatch.setattr(dspark_attention, "get_sm_version", lambda: 107)
    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_INTERNAL_AVAILABLE", True)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "fp8_batched_quantize_1x128_permute102",
        fake_quantize,
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_dspark_fp8_bmm_quantize_rubin_out",
        fake_bmm_quantize,
    )

    output = _output_projection(
        o,
        wo_a=None,
        wo_b=FakeOutputLinear(),
        n_groups=groups,
        o_lora_rank=rank,
        o_a_proj=o_a_proj,
        o_a_scale=o_a_scale,
        o_a_dequant=None,
    )

    assert output.shape == (batch, block, hidden)
    assert len(calls) == 1
    assert len(prevalidated_calls) == 1
    assert prevalidated_calls[0][0] is calls[0]
    fp8_output, packed_scale = calls[0]
    assert fp8_output.shape == (batch * block, groups * rank)
    assert fp8_output.dtype == torch.float8_e4m3fn
    assert packed_scale.dtype == torch.uint8


def test_fp8_output_projection_uses_prepared_attention_quantized_output(monkeypatch):
    import tensorrt_llm._torch.models.modeling_dspark as dspark_attention

    batch, block, hidden = 1, 2, 4
    heads, head_dim, rope_dim, groups, rank, window = 2, 4, 2, 2, 2, 4
    x = torch.zeros(batch, block, hidden, dtype=torch.bfloat16)
    main_x = torch.zeros(batch, 1, hidden, dtype=torch.bfloat16)
    start_pos = torch.tensor([3], dtype=torch.long)
    slots = torch.tensor([0], dtype=torch.long)
    valid_len = torch.tensor([3], dtype=torch.long)
    kv_cache = torch.zeros(batch, window, head_dim, dtype=torch.bfloat16)
    q_latent = torch.zeros(batch, block, 2, dtype=torch.bfloat16)
    block_kv = torch.zeros(batch, block, head_dim, dtype=torch.bfloat16)
    main_kv = torch.zeros(batch, 1, head_dim, dtype=torch.bfloat16)
    q_projected = torch.zeros(batch, block, heads * head_dim, dtype=torch.bfloat16)
    prepared_fp8 = object()
    prepared_scale = object()
    expected_output = torch.zeros(batch, block, hidden, dtype=torch.bfloat16)
    calls = []

    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(
        dspark_attention,
        "_qkv_projections",
        lambda *args, **kwargs: (q_latent, block_kv, main_kv),
    )
    monkeypatch.setattr(
        dspark_attention,
        "_q_projection",
        lambda *args, **kwargs: q_projected,
    )
    monkeypatch.setattr(
        dspark_attention,
        "_rmsnorm_rope_batched",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(
        dspark_attention,
        "is_cute_dsl_dspark_attention_prepared_supported",
        lambda *args: True,
        raising=False,
    )
    monkeypatch.setattr(
        dspark_attention,
        "is_fused_dspark_attention_preparation_supported",
        lambda *args: True,
        raising=False,
    )
    monkeypatch.setattr(
        dspark_attention,
        "cute_dsl_dspark_rmsnorm_rope_cache_write",
        lambda *args: (slots.to(torch.int32), start_pos.to(torch.int32)),
        raising=False,
    )
    monkeypatch.setattr(
        dspark_attention,
        "cute_dsl_dspark_rmsnorm_rope_block_page",
        lambda *args: torch.zeros(batch, 8, head_dim, dtype=torch.bfloat16),
        raising=False,
    )

    def prepared_attention_fp8(*args):
        calls.append("prepared_fp8_attention")
        assert args[-1] == groups
        return prepared_fp8, prepared_scale

    def output_projection(grouped_o_fp8, grouped_o_scale, **kwargs):
        calls.append("fp8_output_consumer")
        assert grouped_o_fp8 is prepared_fp8
        assert grouped_o_scale is prepared_scale
        assert kwargs["batch"] == batch
        assert kwargs["block"] == block
        return expected_output

    monkeypatch.setattr(
        dspark_attention,
        "cute_dsl_dspark_attention_prepared_fp8",
        prepared_attention_fp8,
        raising=False,
    )
    monkeypatch.setattr(
        dspark_attention,
        "cute_dsl_dspark_attention_prepared",
        lambda *args: pytest.fail("BF16 prepared attention should not run"),
        raising=False,
    )
    monkeypatch.setattr(
        dspark_attention,
        "_fused_fp8_output_projection_from_quantized",
        output_projection,
    )
    monkeypatch.setattr(
        dspark_attention,
        "_output_projection",
        lambda *args, **kwargs: pytest.fail("standalone output quantization should not run"),
    )

    actual = dspark_attention.dspark_attention_forward_batched(
        x,
        main_x,
        start_pos,
        kv_cache,
        slots,
        valid_len,
        wq_a=None,
        q_norm_w=torch.ones(2, dtype=torch.bfloat16),
        wq_b=object(),
        wkv=None,
        kv_norm_w=torch.ones(head_dim, dtype=torch.bfloat16),
        wo_a=None,
        wo_b=object(),
        attn_sink=torch.zeros(heads, dtype=torch.float32),
        n_heads=heads,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        n_groups=groups,
        o_lora_rank=rank,
        window_size=window,
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
        freqs_cis=torch.ones(16, rope_dim // 2, dtype=torch.complex64),
        persist=True,
        qkv_proj=object(),
        o_a_proj=torch.empty(groups, rank, heads * head_dim // groups, dtype=torch.float8_e4m3fn),
        o_a_scale=torch.ones(1),
        use_fused_output_projection=True,
    )

    assert actual is expected_output
    assert calls == ["prepared_fp8_attention", "fp8_output_consumer"]


def test_rope_table_is_cached_once_per_device():
    model = types.SimpleNamespace(
        _attn_params={"rope_head_dim": 16},
        _freqs_cap=64,
        _rope_theta=10000.0,
        _freqs_table_cache={},
    )

    first = DSv4DSparkDraftModel._dspark_freqs_table(model, torch.device("cpu"))
    second = DSv4DSparkDraftModel._dspark_freqs_table(model, torch.device("cpu"))

    assert first.data_ptr() == second.data_ptr()
    assert len(model._freqs_table_cache) == 1
    positions = torch.tensor([1, 17, 63])
    expected = precompute_dspark_freqs_cis(16, 64, rope_theta=10000.0)
    torch.testing.assert_close(first[positions], expected[positions])


def test_dspark_block_uses_stage_id_as_attention_layer_idx(monkeypatch):
    captured = {}

    def fake_decoder_layer_init(
        self,
        model_config,
        layer_idx,
        aux_stream_dict,
        attention_layer_idx=None,
        mapping_with_cp=None,
        disable_post_moe_fusion=False,
    ):
        torch.nn.Module.__init__(self)
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.layer_idx = layer_idx
        captured.update(
            layer_idx=layer_idx,
            attention_layer_idx=attention_layer_idx,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            disable_post_moe_fusion=disable_post_moe_fusion,
        )

    monkeypatch.setattr(
        modeling_dspark.DeepseekV4DecoderLayer,
        "__init__",
        fake_decoder_layer_init,
    )
    model_config = types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(vocab_size=128, hc_mult=2),
        spec_config=None,
    )

    block = modeling_dspark.DSv4DSparkBlock(
        model_config,
        layer_idx=10,
        aux_stream_dict={},
        stage_id=1,
        num_stages=3,
        num_capture_layers=0,
    )

    assert block.layer_idx == captured["layer_idx"] == 10
    assert captured["attention_layer_idx"] == block.stage_id == 1
    assert captured["disable_post_moe_fusion"] is True


def test_context_wkv_autotune_bucket_contract():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _gen_dspark_context_wkv_m_profiles,
        _map_dspark_context_wkv_m_profile,
    )

    assert [_map_dspark_context_wkv_m_profile(m) for m in (1, 32, 128, 129, 511, 1024)] == [
        128,
        128,
        128,
        256,
        512,
        1024,
    ]
    assert _gen_dspark_context_wkv_m_profiles(1) == (128,)
    assert _gen_dspark_context_wkv_m_profiles(128) == (128,)
    assert _gen_dspark_context_wkv_m_profiles(129) == (128, 256)
    assert _gen_dspark_context_wkv_m_profiles(1024) == (128, 256, 512, 1024)


def test_generation_autotune_bucket_contract_uses_dspark_rows():
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _get_dspark_generation_fp8_bmm_quant_tuning_config,
        _get_dspark_generation_mxfp8_tuning_config,
    )

    tuner = AutoTuner()
    block_config = _get_dspark_generation_mxfp8_tuning_config(5, 32)
    profiles = tuner._optimization_profiles(block_config, [torch.empty(5, 8)])
    assert [profile.get_opt_shapes() for profile in profiles] == [
        ((5, 8),),
        ((10, 8),),
        ((20, 8),),
        ((40, 8),),
        ((80, 8),),
        ((160, 8),),
    ]
    block_spec = block_config.dynamic_tensor_specs[0]
    assert [block_spec.map_to_tuning_buckets(m) for m in (5, 15, 25, 160)] == [
        5,
        20,
        40,
        160,
    ]

    qkv_config = _get_dspark_generation_mxfp8_tuning_config(6, 32)
    profiles = tuner._optimization_profiles(qkv_config, [torch.empty(6, 8)])
    assert [profile.get_opt_shapes()[0][0] for profile in profiles] == [
        6,
        12,
        24,
        48,
        96,
        192,
    ]

    oa_config = _get_dspark_generation_fp8_bmm_quant_tuning_config(5, 32)
    oa_spec = oa_config.dynamic_tensor_specs[0]
    assert oa_spec.gen_tuning_buckets == (
        5,
        6,
        10,
        12,
        20,
        24,
        40,
        48,
        80,
        96,
        160,
        192,
    )
    assert [oa_spec.map_to_tuning_buckets(m) for m in (5, 6, 15, 18, 160, 192)] == [
        5,
        6,
        20,
        20,
        160,
        192,
    ]


def test_model_configures_projection_owned_generation_buckets():
    class Projection:
        def __call__(self, x):
            return x

    qkv, q_b, o_b = Projection(), Projection(), Projection()
    model = types.SimpleNamespace(
        _use_bf16_linear=False,
        block_size=5,
        mtp_layers=[
            types.SimpleNamespace(_dspark_attn={"qkv_proj": qkv, "wq_b": q_b, "wo_b": o_b})
        ],
    )

    DSv4DSparkDraftModel.configure_generation_autotuner(model, max_batch=32)

    assert (
        qkv._dspark_generation_tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets[-1] == 192
    )
    assert (
        q_b._dspark_generation_tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets[-1] == 160
    )
    assert o_b._dspark_generation_tuning_config is q_b._dspark_generation_tuning_config
    assert o_b._dspark_generation_max_batch == 32
    assert o_b._dspark_generation_rows_per_request == 5


def test_stack_fp8_projection_output_rows_preserves_stage_order(monkeypatch):
    rows, output_rows, hidden = 256, 128, 128
    scale_cols = hidden // 32
    weights = [torch.full((rows, hidden), stage, dtype=torch.float8_e4m3fn) for stage in range(2)]
    unswizzled = [
        torch.arange(rows * scale_cols).view(rows, scale_cols) + stage * 10000 for stage in range(2)
    ]
    projections = [
        types.SimpleNamespace(weight=weight, weight_scale=object()) for weight in weights
    ]

    def fake_unswizzle(scale, scale_rows, cols, scaling_vector_size):
        assert (scale_rows, cols, scaling_vector_size) == (rows, hidden, 32)
        stage = next(
            stage
            for stage, projection in enumerate(projections)
            if scale is projection.weight_scale
        )
        return unswizzled[stage]

    def fake_swizzle(scale, scale_rows, cols, scaling_vector_size):
        assert (scale_rows, cols, scaling_vector_size) == (
            len(projections) * output_rows,
            hidden,
            32,
        )
        return scale.flatten()

    monkeypatch.setattr(modeling_dspark, "unswizzle_sf", fake_unswizzle)
    monkeypatch.setattr(modeling_dspark, "swizzle_sf", fake_swizzle)

    stacked_weight, stacked_scale = modeling_dspark._stack_fp8_projection_output_rows(
        projections, output_rows
    )

    assert stacked_weight.shape == (len(projections) * output_rows, hidden)
    torch.testing.assert_close(
        stacked_weight,
        torch.cat([weight[-output_rows:] for weight in weights]),
    )
    torch.testing.assert_close(
        stacked_scale,
        torch.cat([scale[-output_rows:] for scale in unswizzled]).flatten(),
    )


@pytest.mark.parametrize(
    "shape,output_rows",
    [
        ((256, 128), 64),
        ((192, 128), 128),
        ((256, 96), 128),
    ],
)
def test_stack_fp8_projection_output_rows_rejects_unaligned_geometry(shape, output_rows):
    projection = types.SimpleNamespace(
        weight=torch.empty(shape, dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(0),
    )

    with pytest.raises(ValueError, match="128"):
        modeling_dspark._stack_fp8_projection_output_rows([projection], output_rows)


def test_bind_attn_projections_reuses_loaded_mla_modules(monkeypatch):
    import tensorrt_llm._torch.models.modeling_dspark as dspark_attention

    qkv_proj = Mock()
    qkv_proj.in_features = 4
    q_b_proj = Mock()
    q_b_proj.in_features = 6
    o_b_proj = Mock()
    o_b_proj.has_fp8_block_scales = True
    o_b_proj.use_cute_dsl_blockscaling_mm = True
    o_b_proj.disable_deep_gemm = False
    o_b_proj.in_features = 16 * 128
    o_b_proj.weight = torch.empty(8, 16 * 128, dtype=torch.float8_e4m3fn)
    o_b_proj.weight_scale = torch.empty(1, dtype=torch.uint8)
    o_a_proj = torch.empty(16, 128, 8, dtype=torch.float8_e4m3fn)
    o_a_scale = torch.empty(16, 1, 1)
    o_a_dequant = object()
    q_norm = torch.ones(6)
    kv_norm = torch.ones(8)
    sink = torch.zeros(128)
    attn = types.SimpleNamespace(
        num_heads_tp=128,
        n_local_groups=16,
        q_a_layernorm=types.SimpleNamespace(weight=q_norm),
        q_b_proj=q_b_proj,
        kv_a_layernorm=types.SimpleNamespace(weight=kv_norm),
        o_b_proj=o_b_proj,
        mqa=types.SimpleNamespace(attn_sink=sink),
        kv_a_proj_with_mqa=qkv_proj,
        o_a_proj=o_a_proj,
        o_a_proj_scale=o_a_scale,
        o_a_proj_dequant=o_a_dequant,
    )
    stage = types.SimpleNamespace(self_attn=attn)
    model = types.SimpleNamespace(
        _attn_params={
            "n_heads": 128,
            "n_groups": 16,
            "head_dim": 8,
            "eps": 1e-6,
        },
        mtp_layers=[stage],
    )
    monkeypatch.setattr(dspark_attention, "get_sm_version", lambda: 107)
    monkeypatch.setattr(dspark_attention, "IS_CUTLASS_DSL_INTERNAL_AVAILABLE", True)

    DSv4DSparkDraftModel._bind_attn_projections(model)

    bound = stage._dspark_attn
    assert bound["wq_a"] is None
    assert bound["wkv"] is None
    assert bound["wo_a"] is None
    assert bound["qkv_proj"] is qkv_proj
    assert bound["wq_b"] is q_b_proj
    assert bound["wo_b"] is o_b_proj
    assert model._context_wkv_fp8 is None
    assert bound["o_a_proj"] is o_a_proj
    assert bound["o_a_scale"] is o_a_scale
    assert bound["o_a_dequant"] is o_a_dequant
    assert bound["use_fused_output_projection"] is True
    assert bound["use_fused_qkv_input_projection"] is False
    assert bound["use_fused_q_input_projection"] is False
    assert bound["q_norm_w"] is q_norm
    assert bound["kv_norm_w"] is kv_norm
    assert bound["attn_sink"] is sink

    attn.o_a_proj = torch.empty(16, 128, 8, dtype=torch.bfloat16)
    DSv4DSparkDraftModel._bind_attn_projections(model)
    assert stage._dspark_attn["use_fused_output_projection"] is False

    context_wkv = (object(), object())
    stack_projections = Mock(return_value=context_wkv)
    monkeypatch.setattr(
        modeling_dspark,
        "_can_use_fused_fp8_input_projection",
        lambda projection, _in_features: projection is qkv_proj,
    )
    monkeypatch.setattr(
        modeling_dspark,
        "_stack_fp8_projection_output_rows",
        stack_projections,
    )
    from tensorrt_llm._torch.custom_ops import dspark_rmsnorm_rope_custom_op as rmsnorm_rope_op

    monkeypatch.setattr(
        rmsnorm_rope_op,
        "precompile_dspark_fp8_input_fusions",
        Mock(),
    )

    DSv4DSparkDraftModel._bind_attn_projections(model)

    stack_projections.assert_called_once_with([qkv_proj], 8)
    assert model._context_wkv_fp8 is context_wkv


@pytest.mark.parametrize(
    "tp_size,enable_attention_dp,use_bf16",
    [
        (1, False, False),
        (2, True, False),
        (2, False, True),
    ],
)
def test_post_load_weights_selects_dspark_linear_backend(
    monkeypatch, tp_size, enable_attention_dp, use_bf16
):
    model = DSv4DSparkDraftModel.__new__(DSv4DSparkDraftModel)
    torch.nn.Module.__init__(model)
    monkeypatch.delenv("TRTLLM_DSPARK_LINEAR_BACKEND", raising=False)
    mapping = types.SimpleNamespace(
        tp_size=tp_size,
        enable_attention_dp=enable_attention_dp,
    )
    model._use_bf16_linear = modeling_dspark._use_dspark_bf16_linear(mapping)
    model.mtp_layers = []
    fp8_bind = Mock()
    bf16_bind = Mock()
    monkeypatch.setattr(DSv4DSparkDraftModel, "_bind_attn_projections", fp8_bind)
    monkeypatch.setattr(DSv4DSparkDraftModel, "_bind_bf16_attn_projections", bf16_bind)
    weights = {}

    model.post_load_weights()
    model.post_load_weights(weights)
    model.post_load_weights()

    if use_bf16:
        fp8_bind.assert_not_called()
        bf16_bind.assert_called_once_with(weights)
    else:
        fp8_bind.assert_called_once_with()
        bf16_bind.assert_not_called()


def test_post_load_weights_selects_rubin_cute_backend_before_transform(monkeypatch):
    model = DSv4DSparkDraftModel.__new__(DSv4DSparkDraftModel)
    torch.nn.Module.__init__(model)
    model._use_bf16_linear = False

    projection = Linear.__new__(Linear)
    torch.nn.Module.__init__(projection)
    projection.use_cute_dsl_blockscaling_mm = False
    projection.transform_weights = Mock()
    stage = types.SimpleNamespace(self_attn=types.SimpleNamespace(modules=lambda: [projection]))
    model.mtp_layers = [stage]
    model.modules = lambda: [projection]

    bind = Mock()
    monkeypatch.setattr(modeling_dspark, "get_sm_version", lambda: 107)
    monkeypatch.setattr(DSv4DSparkDraftModel, "_bind_attn_projections", bind)

    model.post_load_weights({})

    assert projection.use_cute_dsl_blockscaling_mm is True
    projection.transform_weights.assert_called_once_with()
    bind.assert_called_once_with()


def test_bind_bf16_attn_projections_disables_every_fp8_projection():
    model = DSv4DSparkDraftModel.__new__(DSv4DSparkDraftModel)
    torch.nn.Module.__init__(model)
    model._context_wkv_fp8 = object()
    stage = types.SimpleNamespace(input_layernorm=types.SimpleNamespace(weight=torch.ones(1)))
    model.mtp_layers = [stage]
    prefix = "mtp.0.attn."
    shapes = {
        "wq_a": (2, 4),
        "wq_b": (6, 2),
        "wkv": (3, 4),
        "wo_a": (8, 3),
        "wo_b": (4, 8),
    }
    weights = {
        f"{prefix}q_norm.weight": torch.ones(2),
        f"{prefix}kv_norm.weight": torch.ones(3),
        f"{prefix}attn_sink": torch.zeros(6),
    }
    for index, (name, shape) in enumerate(shapes.items(), start=1):
        weights[f"{prefix}{name}.weight"] = torch.full(shape, index, dtype=torch.float8_e4m3fn)
        weights[f"{prefix}{name}.scale"] = torch.full((1, 1), index + 0.5)

    model._bind_bf16_attn_projections(weights)

    bound = stage._dspark_attn
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        expected = model._block_dequant(
            weights[f"{prefix}{name}.weight"], weights[f"{prefix}{name}.scale"]
        )
        assert bound[name].dtype == torch.bfloat16
        torch.testing.assert_close(bound[name], expected)
    assert bound["qkv_proj"] is None
    assert bound["o_a_proj"] is None
    assert bound["o_a_scale"] is None
    assert bound["o_a_dequant"] is None
    assert bound["use_fused_output_projection"] is False
    assert bound["use_fused_qkv_input_projection"] is False
    assert bound["use_fused_q_input_projection"] is False
    assert model._context_wkv_fp8 is None


def test_context_wkv_fp8_quantizes_once_per_window_write(monkeypatch):
    num_stages, hidden, head_dim, window = 3, 4, 2, 8
    model = DSv4DSparkDraftModel.__new__(DSv4DSparkDraftModel)
    torch.nn.Module.__init__(model)
    model._attn_params = {
        "head_dim": head_dim,
        "rope_head_dim": 0,
        "window_size": window,
        "eps": 1e-6,
    }
    weights = [
        torch.arange(head_dim * hidden, dtype=torch.float32).view(head_dim, hidden) + stage_id
        for stage_id in range(num_stages)
    ]
    qkv_projections = [
        Mock(side_effect=AssertionError("full q_a+wkv projection is forbidden"))
        for _ in range(num_stages)
    ]
    stages = [
        types.SimpleNamespace(
            _dspark_attn={
                "qkv_proj": qkv_projections[stage_id],
                "kv_norm_w": torch.ones(head_dim),
                "wkv": weights[stage_id],
            },
        )
        for stage_id in range(num_stages)
    ]
    stages[0].main_proj = Mock(side_effect=lambda x: x)
    stages[0].main_norm = Mock(side_effect=lambda x: x)
    model.mtp_layers = stages
    model._dspark_freqs_table = lambda device: torch.empty(
        32, 1, dtype=torch.complex64, device=device
    )
    model._context_wkv_fp8 = (torch.cat(weights), object())

    quant_calls = []
    gemm_calls = []
    input_scale = object()

    def fake_quantize(rows):
        quant_calls.append(rows)
        return rows, input_scale

    def fake_gemm(rows, weight, scale, weight_scale):
        assert scale is input_scale
        gemm_calls.append((rows, weight, weight_scale))
        return F.linear(rows, weight)

    monkeypatch.setattr(
        torch.ops.trtllm,
        "fp8_quantize_1x128_packed_ue8m0",
        fake_quantize,
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_dspark_context_wkv_gemm_rubin",
        fake_gemm,
    )
    monkeypatch.setattr(modeling_dspark, "_rmsnorm", lambda value, *args: value)
    monkeypatch.setattr(modeling_dspark, "_rope_last_dims", lambda value, *args: value)
    monkeypatch.setattr(modeling_dspark, "_rope_last_dims_batched", lambda value, *args: value)

    main_hidden = torch.arange(2 * hidden, dtype=torch.float32).view(2, hidden)
    positions = torch.tensor([1, 2])
    stage_windows = torch.zeros(num_stages, window, head_dim)
    model.write_context_windows(main_hidden, positions, stage_windows)

    assert len(quant_calls) == 1
    assert len(gemm_calls) == 1
    for stage_id, weight in enumerate(weights):
        torch.testing.assert_close(
            stage_windows[stage_id, positions], F.linear(main_hidden, weight)
        )

    batched_hidden = torch.arange(2 * 2 * hidden, dtype=torch.float32).view(2, 2, hidden)
    batched_positions = torch.tensor([[1, 32], [3, 4]])
    request_slots = torch.tensor([0, 1])
    mask = torch.tensor([[True, False], [True, True]])
    kv_windows = torch.zeros(2, num_stages, window, head_dim)
    model.write_context_windows_batched(
        batched_hidden, batched_positions, request_slots, mask, kv_windows
    )

    assert len(quant_calls) == 2
    assert len(gemm_calls) == 2
    for request in range(2):
        for token in range(2):
            for stage_id, weight in enumerate(weights):
                expected = (
                    F.linear(batched_hidden[request, token], weight)
                    if mask[request, token]
                    else torch.zeros(head_dim)
                )
                torch.testing.assert_close(
                    kv_windows[request, stage_id, batched_positions[request, token] % window],
                    expected,
                )
    assert all(projection.call_count == 0 for projection in qkv_projections)

    expected_stage_windows = stage_windows.clone()
    expected_kv_windows = kv_windows.clone()
    model._context_wkv_fp8 = None
    for stage in stages:
        stage._dspark_attn["qkv_proj"] = None
    stage_windows.zero_()
    kv_windows.zero_()
    model.write_context_windows(main_hidden, positions, stage_windows)
    model.write_context_windows_batched(
        batched_hidden, batched_positions, request_slots, mask, kv_windows
    )
    torch.testing.assert_close(stage_windows, expected_stage_windows)
    torch.testing.assert_close(kv_windows, expected_kv_windows)
    assert len(quant_calls) == len(gemm_calls) == 2


@pytest.mark.parametrize("enable_fused_hc", [True, False])
def test_forward_stage_honors_enable_fused_hc(monkeypatch, enable_fused_hc):
    """The draft stage must honor the fused HC setting."""
    torch.manual_seed(71)
    num_requests, block_size, hc_mult, hidden_size = 1, 2, 2, 3
    h = torch.randn(num_requests, block_size, hc_mult, hidden_size)
    attention_input = torch.randn(num_requests, block_size, hidden_size)
    attention_output = torch.randn_like(attention_input)
    mid_residual = torch.randn_like(h)
    attention_post_mix = torch.randn(num_requests, block_size, hc_mult, 1)
    attention_comb_mix = torch.randn(num_requests, block_size, hc_mult, hc_mult)
    ffn_post_mix = torch.randn_like(attention_post_mix)
    ffn_comb_mix = torch.randn_like(attention_comb_mix)
    raw_ffn_input = torch.randn_like(attention_input)
    normed_ffn_input = torch.randn_like(attention_input)
    moe_output = torch.randn(num_requests * block_size, hidden_size)
    final_h = torch.randn_like(h)
    events = []

    def record(name, result):
        def call(*args, **kwargs):
            events.append(name)
            return result

        return call

    monkeypatch.setattr(
        modeling_dspark,
        "dspark_attention_forward",
        Mock(return_value=attention_output),
    )

    hc_attn = types.SimpleNamespace(
        pre_mapping=Mock(return_value=(attention_post_mix, attention_comb_mix, attention_input)),
        post_mapping=Mock(side_effect=record("attention_post", mid_residual)),
    )
    hc_ffn = types.SimpleNamespace(
        fused_hc=Mock(
            side_effect=record(
                "fused",
                (mid_residual, ffn_post_mix, ffn_comb_mix, normed_ffn_input),
            )
        ),
        pre_mapping=Mock(
            side_effect=record("ffn_pre", (ffn_post_mix, ffn_comb_mix, raw_ffn_input))
        ),
        post_mapping=Mock(side_effect=record("ffn_post", final_h)),
    )
    post_attention_layernorm = Mock(side_effect=record("ffn_norm", normed_ffn_input))
    post_attention_layernorm.weight = torch.ones(hidden_size)
    post_attention_layernorm.variance_epsilon = 1e-6
    stage = types.SimpleNamespace(
        enable_fused_hc=enable_fused_hc,
        hc_attn=hc_attn,
        hc_ffn=hc_ffn,
        input_layernorm=Mock(side_effect=lambda tensor: tensor),
        post_attention_layernorm=post_attention_layernorm,
        mlp=Mock(return_value=moe_output),
        _dspark_attn={},
    )
    model = types.SimpleNamespace(
        use_real_mla=False,
        _attn_params={"window_size": 2, "head_dim": 1},
        model_config=types.SimpleNamespace(
            mapping=types.SimpleNamespace(enable_attention_dp=False, tp_size=8)
        ),
    )

    actual = DSv4DSparkDraftModel._forward_stage(
        model,
        stage,
        h,
        torch.randn(num_requests, hidden_size),
        1,
        torch.empty(0),
        torch.zeros(num_requests, block_size, dtype=torch.long),
    )

    assert actual is final_h
    torch.testing.assert_close(
        stage.mlp.call_args.args[0],
        normed_ffn_input.reshape(num_requests * block_size, hidden_size),
    )
    # Non-attention-DP multi-GPU (enable_attention_dp=False, tp_size>1): the draft
    # MoE must all-reduce its TP-sharded output, mirroring the target MoE. The
    # attention-DP and single-GPU paths keep it disabled.
    assert stage.mlp.call_args.kwargs["final_all_reduce_params"].enable_allreduce is True
    if enable_fused_hc:
        hc_ffn.fused_hc.assert_called_once()
        hc_attn.post_mapping.assert_not_called()
        hc_ffn.pre_mapping.assert_not_called()
        fused_kwargs = hc_ffn.fused_hc.call_args.kwargs
        assert events == ["fused", "ffn_post"]
        post_attention_layernorm.assert_not_called()
        assert fused_kwargs["norm_weight"] is post_attention_layernorm.weight
        assert fused_kwargs["norm_eps"] == post_attention_layernorm.variance_epsilon
    else:
        assert events == ["attention_post", "ffn_pre", "ffn_norm", "ffn_post"]
        hc_ffn.fused_hc.assert_not_called()
        hc_attn.post_mapping.assert_called_once()
        hc_ffn.pre_mapping.assert_called_once_with(mid_residual)
        post_attention_layernorm.assert_called_once_with(raw_ffn_input)


def _ref_precompute_freqs_cis(dim, seqlen, base):
    """DeepSpec precompute_freqs_cis with original_seq_len == 0 (no YaRN)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _ref_apply_rotary_emb(x, freqs_cis, inverse=False):
    """DeepSpec apply_rotary_emb (returns a fresh tensor instead of in-place)."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        fc = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        fc = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


def _loop_reference(q, kv, attn_sink, topk_idxs, scale):
    """Obvious, slow per-(b,m,h) reference of the exact kernel math."""
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d, dtype=torch.float32)
    qf, kvf, sink = q.float(), kv.float(), attn_sink.float()
    for bi in range(b):
        for mi in range(m):
            idxs = topk_idxs[bi, mi].tolist()
            for hi in range(h):
                scores, vecs = [], []
                for j in idxs:
                    if j < 0:
                        continue
                    k = kvf[bi, j]
                    scores.append(torch.dot(qf[bi, mi, hi], k) * scale)
                    vecs.append(k)
                if not scores:
                    continue
                s = torch.stack(scores)
                smax = s.max()
                p = torch.exp(s - smax)
                denom = p.sum() + torch.exp(sink[hi] - smax)
                num = (p.unsqueeze(-1) * torch.stack(vecs)).sum(0)
                out[bi, mi, hi] = num / denom
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_attn_matches_loop_reference(seed):
    torch.manual_seed(seed)
    b, m, h, d, n, topk = 2, 5, 3, 16, 40, 12
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, n, d)
    attn_sink = torch.randn(h)
    idx = torch.stack(
        [torch.stack([torch.randperm(n)[:topk] for _ in range(m)]) for _ in range(b)]
    ).int()
    scale = d**-0.5
    got = dspark_sparse_attn(q, kv, attn_sink, idx, scale).float()
    ref = _loop_reference(q, kv, attn_sink, idx, scale)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_sparse_attn_no_sink_matches_sdpa():
    """With a -inf sink (no sink mass) and all-valid contiguous indices, the
    primitive must equal standard scaled-dot-product attention over the gathered
    KV — an independent implementation."""
    torch.manual_seed(0)
    b, m, h, d, topk = 2, 4, 5, 16, 9
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, topk, d)  # n == topk, attend to all
    attn_sink = torch.full((h,), float("-inf"))
    idx = torch.arange(topk).view(1, 1, -1).expand(b, m, topk).int()
    scale = d**-0.5
    got = dspark_sparse_attn(q, kv, attn_sink, idx, scale).float()

    # SDPA: q [b,h,m,d], k/v [b,h,topk,d] (broadcast the shared KV over heads).
    qh = q.permute(0, 2, 1, 3)
    kvh = kv.unsqueeze(1).expand(b, h, topk, d)
    ref = F.scaled_dot_product_attention(qh, kvh, kvh, scale=scale).permute(0, 2, 1, 3)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_sparse_attn_sink_reduces_mass():
    """A finite sink must strictly shrink the attention output magnitude vs an
    infinitely-negative (disabled) sink, because it adds denominator mass only."""
    torch.manual_seed(0)
    b, m, h, d, topk = 1, 2, 2, 16, 6
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, topk, d)
    idx = torch.arange(topk).view(1, 1, -1).expand(b, m, topk).int()
    scale = d**-0.5
    no_sink = dspark_sparse_attn(q, kv, torch.full((h,), float("-inf")), idx, scale)
    with_sink = dspark_sparse_attn(q, kv, torch.zeros(h), idx, scale)
    assert with_sink.abs().sum() < no_sink.abs().sum()


def test_sparse_attn_masked_indices_excluded():
    """An index of -1 must be excluded exactly (equiv. to dropping that column)."""
    torch.manual_seed(0)
    b, m, h, d = 1, 1, 2, 16
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, 5, d)
    sink = torch.full((h,), float("-inf"))
    scale = d**-0.5
    full = torch.tensor([[[0, 1, 2, 3]]]).int()
    masked = torch.tensor([[[0, 1, 2, -1]]]).int()
    drop3 = torch.tensor([[[0, 1, 2]]]).int()
    got_masked = dspark_sparse_attn(q, kv, sink, masked, scale)
    got_drop = dspark_sparse_attn(q, kv, sink, drop3, scale)
    torch.testing.assert_close(got_masked, got_drop, rtol=1e-5, atol=1e-5)
    # And masking genuinely changes the result vs attending to position 3.
    got_full = dspark_sparse_attn(q, kv, sink, full, scale)
    assert not torch.allclose(got_full, got_masked, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "start_pos,window,block", [(1, 128, 5), (3, 128, 5), (10, 4, 5), (200, 128, 6)]
)
def test_get_dspark_topk_idxs_matches_reference(start_pos, window, block):
    bsz = 3
    got = get_dspark_topk_idxs(window, bsz, block, start_pos)
    # Reference formula (DeepSpec get_dspark_topk_idxs).
    ctx = torch.arange(min(window, start_pos + 1))
    blk = window + torch.arange(block)
    ref_row = torch.cat([ctx, blk]).int()
    assert got.shape == (bsz, block, ref_row.numel())
    for bi in range(bsz):
        for mi in range(block):
            torch.testing.assert_close(got[bi, mi], ref_row)


def test_get_dspark_topk_idxs_requires_generation():
    with pytest.raises(AssertionError):
        get_dspark_topk_idxs(128, 1, 5, 0)


@pytest.mark.parametrize("rope_head_dim,seqlen", [(64, 16), (64, 1), (128, 8)])
def test_precompute_freqs_cis_matches_reference(rope_head_dim, seqlen):
    got = precompute_dspark_freqs_cis(rope_head_dim, seqlen, rope_theta=10000.0)
    ref = _ref_precompute_freqs_cis(rope_head_dim, seqlen, 10000.0)
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("ndim", [3, 4])
def test_apply_rotary_matches_reference(ndim):
    torch.manual_seed(0)
    b, s, h, rd = 2, 5, 4, 64
    x = torch.randn(b, s, h, rd) if ndim == 4 else torch.randn(b, s, rd)
    fc = precompute_dspark_freqs_cis(rd, s)
    got = apply_dspark_rotary(x, fc)
    ref = _ref_apply_rotary_emb(x, fc)
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("ndim", [3, 4])
def test_apply_rotary_inverse_roundtrip(ndim):
    """De-rotation (inverse) must undo the forward rotation (property test)."""
    torch.manual_seed(1)
    b, s, h, rd = 2, 6, 3, 64
    x = torch.randn(b, s, h, rd) if ndim == 4 else torch.randn(b, s, rd)
    fc = precompute_dspark_freqs_cis(rd, s)
    roundtrip = apply_dspark_rotary(apply_dspark_rotary(x, fc), fc, inverse=True)
    torch.testing.assert_close(roundtrip, x, rtol=1e-5, atol=1e-5)


def _make_attn_inputs(seed=0):
    """Small synthetic DSpark attention inputs/weights (CPU bf16)."""
    torch.manual_seed(seed)
    dim, n_heads, head_dim, rd = 12, 4, 8, 4
    q_lora, o_lora, n_groups = 6, 5, 2
    window, block, start_pos = 8, 3, 5
    b = 2
    g = dict(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rd,
        q_lora=q_lora,
        o_lora=o_lora,
        n_groups=n_groups,
        window=window,
        block=block,
        start_pos=start_pos,
        b=b,
        eps=1e-6,
        softmax_scale=head_dim**-0.5,
    )
    bf = torch.bfloat16
    g["x"] = torch.randn(b, block, dim, dtype=bf)
    g["main_x"] = torch.randn(b, 1, dim, dtype=bf)
    g["kv_cache0"] = torch.randn(b, window, head_dim, dtype=bf)
    g["wq_a"] = torch.randn(q_lora, dim, dtype=bf) * 0.1
    g["wq_b"] = torch.randn(n_heads * head_dim, q_lora, dtype=bf) * 0.1
    g["wkv"] = torch.randn(head_dim, dim, dtype=bf) * 0.1
    g["wo_a"] = torch.randn(n_groups * o_lora, n_heads * head_dim // n_groups, dtype=bf) * 0.1
    g["wo_b"] = torch.randn(dim, n_groups * o_lora, dtype=bf) * 0.1
    g["q_norm"] = torch.ones(q_lora)
    g["kv_norm"] = torch.ones(head_dim)
    g["attn_sink"] = torch.randn(n_heads)
    g["freqs"] = precompute_dspark_freqs_cis(rd, start_pos + 1 + block + 2)
    return g


def _run(g):
    return dspark_attention_forward(
        g["x"],
        g["main_x"],
        g["start_pos"],
        g["kv_cache0"],
        wq_a=g["wq_a"],
        q_norm_w=g["q_norm"],
        wq_b=g["wq_b"],
        wkv=g["wkv"],
        kv_norm_w=g["kv_norm"],
        wo_a=g["wo_a"],
        wo_b=g["wo_b"],
        attn_sink=g["attn_sink"],
        n_heads=g["n_heads"],
        head_dim=g["head_dim"],
        rope_head_dim=g["rope_head_dim"],
        n_groups=g["n_groups"],
        o_lora_rank=g["o_lora"],
        window_size=g["window"],
        eps=g["eps"],
        softmax_scale=g["softmax_scale"],
        freqs_cis=g["freqs"],
    )


def test_attention_forward_shape_and_determinism():
    g = _make_attn_inputs()
    o = _run(g)
    assert tuple(o.shape) == (g["b"], g["block"], g["dim"])
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o, _run(g))  # deterministic


def test_attention_forward_accepts_loaded_projection_modules(monkeypatch):
    """The production module path matches the tensor-weight reference path."""
    from tensorrt_llm._torch.modules import mla

    g = _make_attn_inputs(seed=13)
    o_a_proj = g["wo_a"].view(g["n_groups"], g["o_lora"], -1).to(torch.float8_e4m3fn)
    reference = dict(g)
    reference["wo_a"] = o_a_proj.to(g["wo_a"].dtype).flatten(0, 1)
    expected = _run(reference)

    def fake_blockscaled_bmm(
        mat1,
        mat2_fp8,
        mat2_scale,
        out,
        mat2_dequant=None,
        use_cute_dsl_blockscaling_bmm=False,
    ):
        del mat2_scale, mat2_dequant
        assert use_cute_dsl_blockscaling_bmm is True
        torch.bmm(
            mat1.transpose(0, 1),
            mat2_fp8.to(mat1.dtype).transpose(1, 2),
            out=out,
        )

    monkeypatch.setattr(mla, "fp8_block_scaling_bmm_out", fake_blockscaled_bmm)
    actual = dspark_attention_forward(
        g["x"],
        g["main_x"],
        g["start_pos"],
        g["kv_cache0"],
        wq_a=None,
        q_norm_w=g["q_norm"],
        wq_b=lambda x: F.linear(x, g["wq_b"]),
        wkv=None,
        kv_norm_w=g["kv_norm"],
        wo_a=None,
        wo_b=lambda x: F.linear(x, g["wo_b"]),
        attn_sink=g["attn_sink"],
        n_heads=g["n_heads"],
        head_dim=g["head_dim"],
        rope_head_dim=g["rope_head_dim"],
        n_groups=g["n_groups"],
        o_lora_rank=g["o_lora"],
        window_size=g["window"],
        eps=g["eps"],
        softmax_scale=g["softmax_scale"],
        freqs_cis=g["freqs"],
        qkv_proj=lambda x: torch.cat((F.linear(x, g["wq_a"]), F.linear(x, g["wkv"])), dim=-1),
        o_a_proj=o_a_proj,
        o_a_scale=torch.ones(g["n_groups"], 1, 1),
    )

    torch.testing.assert_close(actual, expected)


def test_attention_forward_does_not_mutate_kv_cache():
    """The rolling window write must be functional (cache cloned, not mutated)."""
    g = _make_attn_inputs()
    before = g["kv_cache0"].clone()
    _run(g)
    torch.testing.assert_close(g["kv_cache0"], before)
