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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.attention import mla as mla_module
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.locality_domain.policy import (
    LocalityDomainExecutionPlanner,
    LocalityDomainPolicy,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.functional import PositionEmbeddingType


class _FakeAttention(nn.Module):
    def support_fused_rope(self) -> bool:
        return True


class _RecordingLocalityDomainRuntime:
    def __init__(self) -> None:
        self.num_partitions = 2
        self.entered_partitions: list[int] = []
        self.exited_partitions: list[int] = []
        self.prepared_plans: list[object] = []

    @contextmanager
    def partition_weight_context(self, partition_id: int) -> Iterator[None]:
        self.entered_partitions.append(partition_id)
        yield
        self.exited_partitions.append(partition_id)

    def prepare_for_capture(self, plan: object) -> None:
        self.prepared_plans.append(plan)


def _make_bare_absorption_mla(
    *,
    policy_enabled: bool = True,
    use_cute_dsl_bf16_bmm: bool = True,
    is_deepseek_v4: bool = False,
    weight_dtype: torch.dtype = torch.bfloat16,
    qk_nope_head_dim: int = 128,
) -> tuple[MLA, _RecordingLocalityDomainRuntime]:
    mla = MLA.__new__(MLA)
    nn.Module.__init__(mla)
    mla.is_deepseek_v4 = is_deepseek_v4
    mla.use_cute_dsl_bf16_bmm = use_cute_dsl_bf16_bmm
    mla._locality_domain_policy = LocalityDomainPolicy(
        enabled=policy_enabled,
        allowed_ops=frozenset({"bf16_bmm"}),
    )
    runtime = _RecordingLocalityDomainRuntime()
    mla._locality_domain_runtime = runtime
    mla._locality_domain_k_b_proj_trans_shards = None
    mla._locality_domain_v_b_proj_shards = None
    mla._weights_transformed = False
    mla.kv_b_proj = SimpleNamespace(
        quant_config=None, partition_plan=SimpleNamespace(enabled=False)
    )
    mla.num_heads_tp = 2
    mla.num_heads_tp_cp = 2
    mla.qk_nope_head_dim = qk_nope_head_dim
    mla.kv_lora_rank = 512
    mla.v_head_dim = 128

    k_values = torch.arange(
        mla.num_heads_tp * mla.kv_lora_rank * mla.qk_nope_head_dim,
        dtype=torch.float32,
    ).view(mla.num_heads_tp, mla.kv_lora_rank, mla.qk_nope_head_dim)
    v_values = torch.arange(
        mla.num_heads_tp_cp * mla.v_head_dim * mla.kv_lora_rank,
        dtype=torch.float32,
    ).view(mla.num_heads_tp_cp, mla.v_head_dim, mla.kv_lora_rank)
    mla.k_b_proj_trans = nn.Parameter(k_values.to(weight_dtype), requires_grad=False)
    mla.v_b_proj = nn.Parameter(v_values.to(weight_dtype), requires_grad=False)
    return mla, runtime


def _make_mla_with_config(
    config: ModelConfig,
    q_lora_rank: int | None,
    *,
    enable_locality_domain_bf16_linear: bool = True,
) -> MLA:
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
        q_lora_rank=q_lora_rank,
        kv_lora_rank=4,
        predicted_tokens_per_seq=1,
        max_position_embeddings=8,
        bias=False,
        pos_embd_params=position_embedding,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=config,
        o_lora_rank=2,
        enable_locality_domain_bf16_linear=enable_locality_domain_bf16_linear,
    )


@pytest.mark.parametrize(
    ("q_lora_rank", "enable_locality_domain_bf16_linear"),
    [
        pytest.param(4, True, id="q-lora"),
        pytest.param(None, True, id="fused-q-projection"),
        pytest.param(4, False, id="explicit-opt-out"),
    ],
)
def test_mla_wires_locality_domain_policy_to_bf16_linear_projections(
    q_lora_rank: int | None,
    enable_locality_domain_bf16_linear: bool,
) -> None:
    allowed_ops = {"bf16_linear"}
    if enable_locality_domain_bf16_linear:
        allowed_ops.add("bf16_bmm")
    policy = LocalityDomainPolicy(enabled=True, allowed_ops=frozenset(allowed_ops))
    config = ModelConfig(
        skip_create_weights_in_init=True,
        use_cute_dsl_bf16_gemm=True,
        locality_domain_policy=policy,
    )

    with patch.object(mla_module, "create_attention", return_value=_FakeAttention()):
        mla = _make_mla_with_config(
            config,
            q_lora_rank,
            enable_locality_domain_bf16_linear=enable_locality_domain_bf16_linear,
        )

    assert mla._locality_domain_policy is policy
    for projection in (mla.q_b_proj, mla.kv_b_proj, mla.o_proj):
        assert projection.use_cute_dsl_bf16_gemm
        assert projection.enable_locality_domain_bf16_linear is enable_locality_domain_bf16_linear
        assert projection._locality_domain_policy is policy
    assert not mla.kv_a_proj_with_mqa.enable_locality_domain_bf16_linear
    if q_lora_rank is None:
        assert mla.q_b_proj is mla.q_proj


@pytest.mark.parametrize("partition_enabled", [True, False])
def test_bind_v_b_proj_weight_respects_bf16_linear_partition_plan(
    partition_enabled: bool,
) -> None:
    mla = MLA.__new__(MLA)
    nn.Module.__init__(mla)
    mla.kv_b_proj = SimpleNamespace(
        partition_plan=SimpleNamespace(
            enabled=partition_enabled,
            op_kind="bf16_linear" if partition_enabled else None,
        )
    )
    source = torch.arange(24, dtype=torch.bfloat16).view(4, 6)
    weight = source[1:]

    mla._bind_v_b_proj_weight(weight)

    torch.testing.assert_close(mla.v_b_proj, weight)
    aliases_weight = (
        mla.v_b_proj.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    )
    assert aliases_weight is not partition_enabled


def test_transform_builds_fresh_localized_absorption_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_calls = []

    def _plan_bf16_bmm(
        _planner: LocalityDomainExecutionPlanner,
        *,
        dtype: torch.dtype,
        use_cute_dsl_bf16_bmm: bool,
    ) -> SimpleNamespace:
        plan_calls.append((dtype, use_cute_dsl_bf16_bmm))
        return SimpleNamespace(enabled=True, num_partitions=2)

    monkeypatch.setattr(
        mla_module.LocalityDomainExecutionPlanner,
        "plan_bf16_bmm",
        _plan_bf16_bmm,
    )
    mla, runtime = _make_bare_absorption_mla()
    original_k = mla.k_b_proj_trans
    original_v = mla.v_b_proj

    mla.transform_weights()

    k_shards = mla._locality_domain_k_b_proj_trans_shards
    v_shards = mla._locality_domain_v_b_proj_shards
    assert k_shards is not None
    assert v_shards is not None
    assert [tuple(shard.shape) for shard in k_shards] == [
        (mla.num_heads_tp, 256, mla.qk_nope_head_dim),
        (mla.num_heads_tp, 256, mla.qk_nope_head_dim),
    ]
    assert [tuple(shard.shape) for shard in v_shards] == [
        (mla.num_heads_tp_cp, 64, mla.kv_lora_rank),
        (mla.num_heads_tp_cp, 64, mla.kv_lora_rank),
    ]
    torch.testing.assert_close(k_shards[0], original_k[:, :256])
    torch.testing.assert_close(k_shards[1], original_k[:, 256:])
    torch.testing.assert_close(v_shards[0], original_v[:, :64])
    torch.testing.assert_close(v_shards[1], original_v[:, 64:])
    assert all(shard.is_contiguous() for shard in (*k_shards, *v_shards))
    assert all(
        shard.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        for shard, source in (
            (k_shards[0], original_k),
            (k_shards[1], original_k),
            (v_shards[0], original_v),
            (v_shards[1], original_v),
        )
    )
    assert runtime.entered_partitions == [0, 1, 0, 1]
    assert runtime.exited_partitions == [0, 1, 0, 1]
    assert len(runtime.prepared_plans) == 1
    assert plan_calls == [(torch.bfloat16, True)]
    assert mla.k_b_proj_trans is original_k
    assert mla.v_b_proj is original_v

    # Repeated staged/legacy hooks must preserve capture-stable shard storage.
    mla.cache_derived_state()
    mla.post_load_weights()
    assert mla._locality_domain_k_b_proj_trans_shards is k_shards
    assert mla._locality_domain_v_b_proj_shards is v_shards
    assert len(runtime.prepared_plans) == 1

    old_k_shards = k_shards
    with torch.no_grad():
        original_k.fill_(3)
        original_v.fill_(5)
    # The checkpoint loader binds the new V weight after loading kv_b_proj.
    # It must invalidate the derived shards without a manual flag reset.
    mla._bind_v_b_proj_weight(original_v)
    mla.post_load_weights()

    rebuilt_k_shards = mla._locality_domain_k_b_proj_trans_shards
    rebuilt_v_shards = mla._locality_domain_v_b_proj_shards
    assert rebuilt_k_shards is not None
    assert rebuilt_v_shards is not None
    assert (
        rebuilt_k_shards[0].untyped_storage().data_ptr()
        != old_k_shards[0].untyped_storage().data_ptr()
    )
    torch.testing.assert_close(rebuilt_k_shards[0], original_k[:, :256])
    torch.testing.assert_close(rebuilt_v_shards[1], original_v[:, 64:])
    assert plan_calls == [(torch.bfloat16, True), (torch.bfloat16, True)]
    assert len(runtime.prepared_plans) == 2


@pytest.mark.parametrize(
    ("is_deepseek_v4", "qk_nope_head_dim"),
    [
        pytest.param(False, 127, id="unaligned-qk-nope-head-dim"),
        pytest.param(True, 128, id="deepseek-v4"),
    ],
)
def test_absorption_shards_reject_unsupported_weights_before_planning(
    monkeypatch: pytest.MonkeyPatch,
    is_deepseek_v4: bool,
    qk_nope_head_dim: int,
) -> None:
    def _unexpected_plan(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unsupported absorption weights must be rejected before planning")

    monkeypatch.setattr(
        mla_module.LocalityDomainExecutionPlanner,
        "plan_bf16_bmm",
        _unexpected_plan,
    )
    mla, runtime = _make_bare_absorption_mla(
        is_deepseek_v4=is_deepseek_v4,
        qk_nope_head_dim=qk_nope_head_dim,
    )

    if is_deepseek_v4:
        # DSV4 sparse hooks do not create absorption projection weights.
        del mla.k_b_proj_trans
        del mla.v_b_proj
    mla.transform_weights()

    assert mla._locality_domain_k_b_proj_trans_shards is None
    assert mla._locality_domain_v_b_proj_shards is None
    assert runtime.entered_partitions == []


def test_absorption_shards_forward_inputs_to_disabled_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _plan_bf16_bmm(
        planner: LocalityDomainExecutionPlanner,
        *,
        dtype: torch.dtype,
        use_cute_dsl_bf16_bmm: bool,
    ) -> SimpleNamespace:
        assert not planner.policy.enabled
        assert dtype == torch.float8_e4m3fn
        assert not use_cute_dsl_bf16_bmm
        return SimpleNamespace(enabled=False)

    monkeypatch.setattr(
        mla_module.LocalityDomainExecutionPlanner,
        "plan_bf16_bmm",
        _plan_bf16_bmm,
    )
    mla, runtime = _make_bare_absorption_mla(
        policy_enabled=False,
        use_cute_dsl_bf16_bmm=False,
        weight_dtype=torch.float8_e4m3fn,
    )

    mla.transform_weights()

    assert mla._locality_domain_k_b_proj_trans_shards is None
    assert mla._locality_domain_v_b_proj_shards is None
    assert runtime.entered_partitions == []


@pytest.mark.parametrize(
    ("use_cute", "sm_version", "with_shards", "expected_op"),
    [
        (True, 107, True, "locality_domain"),
        (True, 107, False, "rubin"),
        (True, 100, True, "blackwell"),
        (True, 103, True, "blackwell"),
        (True, 90, True, "bmm_out"),
        (False, 107, True, "bmm_out"),
    ],
)
def test_bf16_bmm_dispatch_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    use_cute: bool,
    sm_version: int,
    with_shards: bool,
    expected_op: str,
) -> None:
    mla = MLA.__new__(MLA)
    nn.Module.__init__(mla)
    mla.use_cute_dsl_bf16_bmm = use_cute
    monkeypatch.setattr(mla_module, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(mla_module, "is_sm_100f", lambda: 100 <= sm_version < 110)

    calls = []

    def _record(name: str) -> Callable[..., None]:
        def _op(*args: object) -> None:
            calls.append((name, args))

        return _op

    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_bmm_locality_domain_inplace_rubin",
        _record("locality_domain"),
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_bmm_rubin",
        _record("rubin"),
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_bmm_blackwell",
        _record("blackwell"),
    )
    monkeypatch.setattr(torch.ops.trtllm, "bmm_out", _record("bmm_out"))

    a = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    weight = torch.empty((2, 6, 4), dtype=torch.bfloat16)
    output = torch.empty((2, 3, 6), dtype=torch.bfloat16)
    shards = (weight[:, :3].clone(), weight[:, 3:].clone()) if with_shards else None

    mla._bmm_bf16_out(a, weight, weight.transpose(1, 2), output, shards)

    assert len(calls) == 1
    op_name, op_args = calls[0]
    assert op_name == expected_op
    assert op_args[0] is a
    assert op_args[-1] is output
    if expected_op == "locality_domain":
        assert shards is not None
        assert op_args[1] is shards[0]
        assert op_args[2] is shards[1]
    elif expected_op in ("rubin", "blackwell"):
        assert op_args[1] is weight
