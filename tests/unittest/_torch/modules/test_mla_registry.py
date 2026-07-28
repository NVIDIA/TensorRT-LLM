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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mla import MLA, create_mla_outputs_impl
from tensorrt_llm.functional import PositionEmbeddingType


class _FakeAttention(nn.Module):
    def support_fused_rope(self) -> bool:
        return True

    def update_quant_config(self, _quant_config: object) -> None:
        pass


def _make_mla(config: ModelConfig) -> MLA:
    position_embedding = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(dim=2, max_positions=8),
    )
    return MLA(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        predicted_tokens_per_seq=1,
        max_position_embeddings=8,
        bias=False,
        pos_embd_params=position_embedding,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=config,
        o_lora_rank=2,
    )


def test_duplicate_layer_ids_preserve_all_mla_registrations() -> None:
    target_config = ModelConfig(skip_create_weights_in_init=True)
    draft_config = ModelConfig(skip_create_weights_in_init=True)
    next_config = ModelConfig(skip_create_weights_in_init=True)
    draft_config.extra_attrs = target_config.extra_attrs
    next_config.extra_attrs = target_config.extra_attrs

    with patch(
        "tensorrt_llm._torch.modules.mla.create_attention",
        side_effect=lambda *args, **kwargs: _FakeAttention(),
    ):
        target_mla = _make_mla(target_config)
        draft_mla = _make_mla(draft_config)
        next_mla = _make_mla(next_config)

    assert target_mla.layer_idx == draft_mla.layer_idx == next_mla.layer_idx == 0
    assert target_mla.layer_idx_str == "0"
    assert draft_mla.layer_idx_str == "0_0"
    assert next_mla.layer_idx_str == "0_1"
    registry = target_config.extra_attrs["mla_layers"]
    assert registry["0"]() is target_mla
    assert registry["0_0"]() is draft_mla
    assert registry["0_1"]() is next_mla


def test_dsv4_epilogue_fusion_returns_final_output_inside_breakable_graph() -> None:
    metadata = SimpleNamespace(num_contexts=1, num_generations=1, num_tokens=5)
    mla_layer = Mock(spec=MLA)
    mla_layer._should_use_dsv4_epilogue_fusion.return_value = True
    mla_layer.create_output.return_value = torch.empty(8, 4, 2)
    hidden_states = torch.empty(8, 8)

    with (
        patch(
            "tensorrt_llm._torch.modules.mla._extract_mla_extra_attrs",
            return_value=(metadata, mla_layer),
        ),
        patch(
            "tensorrt_llm._torch.modules.mla.is_in_breakable_cuda_graph",
            return_value=True,
        ),
    ):
        output = create_mla_outputs_impl(hidden_states, "0")

    assert output is mla_layer.create_output.return_value
    mla_layer.create_output.assert_called_once_with(
        hidden_states,
        1,
        enable_dsv4_epilogue_fusion=True,
    )
    mla_layer._create_dsv4_epilogue_buffers.assert_not_called()


def test_create_mla_outputs_custom_op_returns_tensor() -> None:
    schema = torch.ops.trtllm.create_mla_outputs.default._schema
    assert [str(return_value.type) for return_value in schema.returns] == ["Tensor"]


def test_mla_custom_op_marks_only_final_output_mutable() -> None:
    schema = torch.ops.trtllm.mla_custom_op_inplace.default._schema
    mutated_args = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    assert mutated_args == ["output"]


def test_dsv4_epilogue_fusion_supports_mixed_batch() -> None:
    mla_layer = SimpleNamespace(
        _disable_dsv4_epilogue_fusion=False,
        is_deepseek_v4=True,
        mapping=SimpleNamespace(
            has_cp_helix=lambda: False,
            enable_attention_dp=True,
        ),
        num_heads=128,
        num_heads_tp=128,
        mqa=SimpleNamespace(
            sparse_params=object(),
            has_fp8_kv_cache=True,
        ),
        o_a_proj=SimpleNamespace(dtype=torch.float8_e4m3fn),
        kv_lora_rank=448,
        qk_rope_head_dim=64,
        qk_head_dim=512,
        v_head_dim=512,
        n_local_groups=8,
        inverse_rotary_emb=SimpleNamespace(is_neox=False),
    )

    with patch("tensorrt_llm._torch.modules.mla.is_sm_100f", return_value=True):
        assert MLA._should_use_dsv4_epilogue_fusion(mla_layer, 1, 1)


def test_dsv4_fusion_create_output_uses_bucket_token_count() -> None:
    mla_layer = SimpleNamespace(
        n_local_groups=4,
        o_lora_rank=3,
        dtype=torch.bfloat16,
    )
    hidden_states = torch.empty(8, 16)

    output = MLA.create_output(
        mla_layer,
        hidden_states,
        num_contexts=1,
        enable_dsv4_epilogue_fusion=True,
    )

    assert output.shape == (8, 4, 3)
    assert output.dtype == torch.bfloat16


def test_dsv4_fusion_o_proj_only_flattens_lora_output() -> None:
    projected = torch.randn(7, 5)
    mla_layer = SimpleNamespace(
        n_local_groups=4,
        o_lora_rank=3,
        o_b_proj=Mock(return_value=projected),
    )
    lora_o = torch.randn(7, 4, 3)

    output = MLA._deepseek_v4_o_proj(
        mla_layer,
        lora_o,
        enable_dsv4_epilogue_fusion=True,
    )

    assert output is projected
    mla_layer.o_b_proj.assert_called_once()
    torch.testing.assert_close(mla_layer.o_b_proj.call_args.args[0], lora_o.flatten(1))


def test_dsv4_epilogue_buffers_use_real_token_count() -> None:
    mla_layer = SimpleNamespace(
        n_local_groups=4,
        num_heads_tp=128,
        v_head_dim=512,
    )
    q = torch.empty(8, 16)

    fp8_o, output_sf = MLA._create_dsv4_epilogue_buffers(mla_layer, q, num_tokens=5)

    assert fp8_o.shape == (4, 5, 32 * 512)
    assert output_sf.shape == (4, 32 * 4, 8)


@pytest.mark.parametrize(
    "num_context_tokens,num_generation_tokens,bucket_tokens",
    [(5, 0, 8), (0, 3, 4), (5, 3, 12)],
)
def test_dsv4_epilogue_bmm_writes_only_phase_ranges(
    num_context_tokens: int,
    num_generation_tokens: int,
    bucket_tokens: int,
) -> None:
    groups = 2
    rank = 3
    output = torch.full((bucket_tokens, groups, rank), -1.0)
    mla_layer = SimpleNamespace(
        o_a_proj=torch.empty(0),
        o_a_proj_scale=torch.empty(0),
    )

    def fake_bmm(_attn_fp8, _weight, attn_scale, _weight_scale, phase_output):
        phase_output.fill_(attn_scale.item())

    with patch.object(
        torch.ops.trtllm,
        "cute_dsl_fp8_bmm_blackwell",
        side_effect=fake_bmm,
    ) as bmm:
        context_epilogue = None
        if num_context_tokens:
            context_epilogue = (
                torch.empty(groups, num_context_tokens, 4),
                torch.tensor(11.0),
            )
        generation_epilogue = None
        if num_generation_tokens:
            generation_epilogue = (
                torch.empty(groups, num_generation_tokens, 4),
                torch.tensor(22.0),
            )
        MLA._run_dsv4_o_lora_bmms(
            mla_layer,
            output,
            num_context_tokens,
            num_context_tokens + num_generation_tokens,
            context_epilogue,
            generation_epilogue,
        )

    assert bmm.call_count == bool(num_context_tokens) + bool(num_generation_tokens)
    if num_context_tokens:
        torch.testing.assert_close(
            output[:num_context_tokens], torch.full_like(output[:num_context_tokens], 11.0)
        )
    if num_generation_tokens:
        generation_end = num_context_tokens + num_generation_tokens
        torch.testing.assert_close(
            output[num_context_tokens:generation_end],
            torch.full_like(output[num_context_tokens:generation_end], 22.0),
        )
    real_tokens = num_context_tokens + num_generation_tokens
    torch.testing.assert_close(output[real_tokens:], torch.full_like(output[real_tokens:], -1.0))
