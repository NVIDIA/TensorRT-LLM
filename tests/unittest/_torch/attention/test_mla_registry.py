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

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
    _create_dsv4_epilogue_buffers,
    _run_dsv4_o_lora_bmms,
    prepare_sparse_attn_outputs,
    project_sparse_attn_output,
)
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.functional import PositionEmbeddingType


class _FakeAttention(nn.Module):
    def support_fused_rope(self) -> bool:
        return True

    def update_quant_config(self, _quant_config: object) -> None:
        pass


def _make_mla(config: ModelConfig, cls: type[MLA] = MLA) -> MLA:
    position_embedding = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(dim=2, max_positions=8),
    )
    return cls(
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


class _OutputGateStub:
    """Minimal ``MLA`` stand-in that records what the output-gate hook sees."""

    forward = MLA.forward

    def __init__(self, register_to_config: bool) -> None:
        self.mapping = SimpleNamespace(has_cp_helix=lambda: False, enable_attention_dp=False)
        self.layer_idx = 0
        self.layer_idx_str = "0"
        self.register_to_config = register_to_config
        self.attn_output = torch.zeros(2, 4)
        self.gate_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.projected: list[torch.Tensor] = []

    def _create_outputs(
        self, hidden_states: torch.Tensor, attn_metadata: object
    ) -> list[torch.Tensor]:
        return [self.attn_output]

    def forward_impl(
        self,
        position_ids: object,
        hidden_states: torch.Tensor,
        attn_metadata: object,
        attn_output: list[torch.Tensor],
        latent_cache_gen: object = None,
    ) -> None:
        attn_output[0].fill_(2.0)

    def _forward_custom_op(
        self,
        hidden_states: torch.Tensor,
        position_ids: object,
        attn_output: list[torch.Tensor],
        latent_cache_gen: object,
    ) -> None:
        attn_output[0].fill_(2.0)

    def _apply_output_gate(
        self, hidden_states: torch.Tensor, attn_output: torch.Tensor
    ) -> torch.Tensor:
        self.gate_calls.append((hidden_states, attn_output.clone()))
        return attn_output * 5.0

    def _project_output(
        self,
        attn_output: list[torch.Tensor],
        position_ids: object,
        attn_metadata: object,
        all_reduce_params: object,
    ) -> torch.Tensor:
        self.projected.append(attn_output[0])
        return attn_output[0]


def test_base_output_gate_is_identity() -> None:
    attn_output = torch.randn(2, 4)

    assert MLA._apply_output_gate(None, torch.randn(2, 4), attn_output) is attn_output


@pytest.mark.parametrize("register_to_config", [False, True])
def test_output_gate_runs_between_attention_and_output_projection(
    register_to_config: bool,
) -> None:
    mla_layer = _OutputGateStub(register_to_config)
    hidden_states = torch.randn(2, 4)
    attn_metadata = SimpleNamespace(num_contexts=0, num_tokens=2)

    with patch.object(torch.ops.trtllm, "create_mla_outputs", return_value=mla_layer.attn_output):
        output = mla_layer.forward(None, hidden_states, attn_metadata)

    # The hook receives the module input and the completed attention output.
    assert len(mla_layer.gate_calls) == 1
    gate_hidden_states, gate_attn_output = mla_layer.gate_calls[0]
    assert gate_hidden_states is hidden_states
    torch.testing.assert_close(gate_attn_output, torch.full((2, 4), 2.0))
    # Its result, not the raw attention output, is what gets projected.
    assert mla_layer.projected == [output]
    torch.testing.assert_close(output, torch.full((2, 4), 10.0))


def test_output_gate_override_rejected_with_sparse_hooks() -> None:
    class _GatedMLA(MLA):
        def _apply_output_gate(
            self, hidden_states: torch.Tensor, attn_output: torch.Tensor
        ) -> torch.Tensor:
            return attn_output * 2.0

    config = ModelConfig(skip_create_weights_in_init=True)
    with (
        patch(
            "tensorrt_llm._torch.attention.mla.create_attention",
            side_effect=lambda *args, **kwargs: _FakeAttention(),
        ),
        patch(
            "tensorrt_llm._torch.attention.mla.get_sparse_mla_hooks",
            return_value=Mock(),
        ),
        pytest.raises(NotImplementedError, match="_apply_output_gate"),
    ):
        _make_mla(config, cls=_GatedMLA)


def test_duplicate_layer_ids_preserve_all_mla_registrations() -> None:
    target_config = ModelConfig(skip_create_weights_in_init=True)
    draft_config = ModelConfig(skip_create_weights_in_init=True)
    next_config = ModelConfig(skip_create_weights_in_init=True)
    draft_config.extra_attrs = target_config.extra_attrs
    next_config.extra_attrs = target_config.extra_attrs

    with patch(
        "tensorrt_llm._torch.attention.mla.create_attention",
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


def _make_dsv4_epilogue_layer() -> SimpleNamespace:
    return SimpleNamespace(
        _disable_dsv4_epilogue_fusion=False,
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
        o_lora_rank=3,
        dtype=torch.bfloat16,
        inverse_rotary_emb=SimpleNamespace(is_neox=False),
        create_output=Mock(),
    )


def test_mla_custom_op_marks_only_final_output_mutable() -> None:
    schema = torch.ops.trtllm.mla_custom_op_inplace.default._schema
    mutated_args = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    assert mutated_args == ["output"]


def test_create_mla_outputs_custom_op_returns_tensor() -> None:
    schema = torch.ops.trtllm.create_mla_outputs.default._schema
    assert [str(return_value.type) for return_value in schema.returns] == ["Tensor"]


def test_dsv4_epilogue_fusion_supports_mixed_batch() -> None:
    mla_layer = _make_dsv4_epilogue_layer()
    metadata = SimpleNamespace(num_contexts=1, num_generations=1)
    hidden_states = torch.empty(8, 16)

    with patch(
        "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module.is_sm_100f",
        return_value=True,
    ):
        outputs = prepare_sparse_attn_outputs(mla_layer, hidden_states, metadata)

    assert len(outputs) == 1
    assert outputs[0].shape == (8, 8, 3)
    assert outputs[0].dtype == torch.bfloat16
    mla_layer.create_output.assert_not_called()


def test_dsv4_fusion_create_output_uses_bucket_token_count() -> None:
    mla_layer = _make_dsv4_epilogue_layer()
    metadata = SimpleNamespace(num_contexts=1, num_generations=0)
    hidden_states = torch.empty(8, 16)

    with patch(
        "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module.is_sm_100f",
        return_value=True,
    ):
        output = prepare_sparse_attn_outputs(mla_layer, hidden_states, metadata)[0]

    assert output.shape == (8, 8, 3)
    assert output.dtype == torch.bfloat16


def test_dsv4_fusion_o_proj_only_flattens_lora_output() -> None:
    projected = torch.randn(7, 5)
    mla_layer = SimpleNamespace(
        n_local_groups=4,
        o_lora_rank=3,
        o_b_proj=Mock(return_value=projected),
    )
    lora_o = torch.randn(7, 4, 3)

    output = project_sparse_attn_output(mla_layer, [lora_o])

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

    fp8_o, output_sf = _create_dsv4_epilogue_buffers(mla_layer, q, num_tokens=5)

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
        _run_dsv4_o_lora_bmms(
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
