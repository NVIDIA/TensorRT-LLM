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

"""
Test for DeepSeek-V4 output projection (_deepseek_v4_o_proj).
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import per_block_cast_to_fp8_e8m0, per_token_cast_to_fp8_e8m0
from utils.util import skip_pre_blackwell

import tensorrt_llm._torch.modules.mla as mla_module
from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, TunableRunner, autotune
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import weight_dequant
from tensorrt_llm._torch.models.modeling_deepseekv4 import (
    DeepseekV4ForCausalLM,
    _resolve_enable_fused_hc,
)
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..test_sparse_mla_forward import RopeConfig, _calc_diff, apply_rotary_emb, precompute_freqs_cis

FP8_O_PROJ_DIFF_TOL = 2e-3


@pytest.mark.parametrize(
    ("config_enabled", "env_value", "expected"),
    [
        (True, None, True),
        (False, None, False),
        (True, "0", False),
        (False, "1", True),
    ],
)
def test_dsv4_fused_oproj_requires_split_output_consumer(
    config_enabled: bool,
    env_value: str | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(enable_fused_hc=config_enabled)
    if env_value is None:
        monkeypatch.delenv("TRTLLM_MHC_ENABLE_FUSED_HC", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_MHC_ENABLE_FUSED_HC", env_value)
    monkeypatch.delenv("TRTLLM_DSV4_DISABLE_FUSED_OPROJ", raising=False)
    monkeypatch.setattr(mla_module, "is_sm_100f", lambda: True)
    monkeypatch.setattr(mla_module, "IS_CUTLASS_DSL_AVAILABLE", True)

    module = SimpleNamespace(
        allow_dsv4_split_output=_resolve_enable_fused_hc(config),
        n_local_groups=16,
        num_groups=16,
        o_a_proj=torch.empty((1,), device="meta", dtype=torch.float8_e4m3fn),
        o_b_proj=SimpleNamespace(
            tp_size=1,
            has_fp8_block_scales=True,
            use_cute_dsl_blockscaling_mm=False,
        ),
        dtype=torch.bfloat16,
    )

    assert MLA._should_use_fused_oproj(module) is expected


def test_dsv4_deep_gemm_ob_warmup_skips_cute_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(_should_use_fused_oproj=lambda: True)

    monkeypatch.delenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", raising=False)
    assert MLA._should_warmup_dsv4_deep_gemm_ob(module)

    monkeypatch.setenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", "1")
    assert not MLA._should_warmup_dsv4_deep_gemm_ob(module)


@pytest.mark.parametrize(
    ("num_tokens", "expected_split"),
    [(1, 4), (16, 4), (32, 2), (64, 2), (96, 2), (128, 2), (160, 1), (16384, 1)],
)
def test_select_dsv4_ob_split_k_auto_policy(
    num_tokens: int, expected_split: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRTLLM_DSV4_OB_SPLIT_K", raising=False)
    assert (
        cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner._select_num_splits(num_tokens)
        == expected_split
    )


def test_select_dsv4_ob_split_k_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", "4")
    assert cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner._select_num_splits(32) == 4
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", "3")
    with pytest.raises(ValueError, match="unsupported.*split-K factor"):
        cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner._select_num_splits(32)


def test_dsv4_fmha_epilogue_output_uses_fused_oproj() -> None:
    attn_fp8 = torch.empty((16, 4, 4096), device="meta", dtype=torch.float8_e4m3fn)
    attn_scale = torch.empty((16, 32, 4), device="meta")
    expected = torch.empty((8, 7168), device="meta", dtype=torch.bfloat16)
    calls = []
    module = SimpleNamespace(
        _should_use_fused_oproj=lambda: True,
        _fused_oa_ob_proj=lambda fp8, scale, num_tokens: calls.append((fp8, scale, num_tokens))
        or expected,
    )

    output = MLA._deepseek_v4_o_proj(module, (attn_fp8, attn_scale))

    assert output is expected
    assert calls == [(attn_fp8, attn_scale, 4)]


def test_dsv4_fmha_epilogue_output_uses_unsplit_fallback_without_fused_hc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens, num_groups, o_lora_rank, hidden_size = 4, 16, 1024, 7168
    attn_fp8 = torch.empty((num_groups, num_tokens, 4096), device="meta", dtype=torch.float8_e4m3fn)
    attn_scale = torch.empty((num_groups, 32, num_tokens), device="meta")
    calls = []

    class OutputProjection:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            calls.append(("ob", inputs.shape))
            return inputs.new_empty((inputs.shape[0], hidden_size), dtype=torch.bfloat16)

    module = SimpleNamespace(
        allow_dsv4_split_output=False,
        n_local_groups=num_groups,
        num_groups=num_groups,
        o_lora_rank=o_lora_rank,
        o_a_proj=torch.empty((1,), device="meta", dtype=torch.float8_e4m3fn),
        o_a_proj_scale=torch.empty((1,), device="meta"),
        o_b_proj=OutputProjection(),
        dtype=torch.bfloat16,
    )
    module.o_b_proj.tp_size = 1
    module.o_b_proj.has_fp8_block_scales = True
    module.o_b_proj.use_cute_dsl_blockscaling_mm = False
    module._should_use_fused_oproj = lambda: MLA._should_use_fused_oproj(module)

    def bmm_fp8out(*args) -> None:
        calls.append(("oa", args[-1].shape))

    monkeypatch.delenv("TRTLLM_DSV4_DISABLE_FUSED_OPROJ", raising=False)
    monkeypatch.setattr(mla_module, "is_sm_100f", lambda: True)
    monkeypatch.setattr(mla_module, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_fp8_bmm_blackwell", bmm_fp8out)

    output = MLA._deepseek_v4_o_proj(module, (attn_fp8, attn_scale))

    assert output.shape == (num_tokens, hidden_size)
    assert calls == [
        ("oa", torch.Size([num_groups, num_tokens, o_lora_rank])),
        ("ob", torch.Size([num_tokens, num_groups * o_lora_rank])),
    ]


def test_dsv4_ob_cute_tactics_are_runtime_tuned() -> None:
    runner_cls = cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner
    runner = runner_cls()

    assert issubclass(runner_cls, TunableRunner)
    assert not hasattr(runner_cls, "_get_tactic")
    for num_splits in (1, 2, 4):
        config = runner.get_tuning_config(num_splits)
        tactics = runner.get_valid_tactics([], OptimizationProfile(), num_splits=num_splits)
        assert config.use_cuda_graph
        assert config.exclude_from_cache
        assert len(tactics) > 1
        assert runner._get_fallback_tactic(num_splits) in tactics

    split1_tiles = {tactic[0] for tactic in runner_cls._SPLIT_K1_TACTICS}
    assert (256, 128) in split1_tiles
    assert (256, 144) in split1_tiles


@pytest.mark.parametrize(
    ("num_splits", "num_tokens", "expected_bucket"),
    [
        (4, 1, 1),
        (4, 3, 4),
        (4, 16, 16),
        (2, 17, 32),
        (2, 37, 64),
        (2, 128, 128),
        (1, 129, 192),
        (1, 193, 256),
        (1, 1025, 1280),
        (1, 16384, 16384),
    ],
)
def test_dsv4_ob_tuning_bucket_mapping(
    num_splits: int, num_tokens: int, expected_bucket: int
) -> None:
    config = cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner.get_tuning_config(num_splits)
    assert config.dynamic_tensor_specs[0].map_to_tuning_buckets(num_tokens) == expected_bucket


@pytest.mark.parametrize(
    ("num_splits", "max_num_tokens", "expected_buckets"),
    [
        (4, 1, (1, 2, 4, 8, 16)),
        (2, 32, (32, 64, 128)),
        (1, 193, (192, 256)),
    ],
)
def test_dsv4_ob_tuning_bucket_generation(
    num_splits: int, max_num_tokens: int, expected_buckets: tuple[int, ...]
) -> None:
    config = cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner.get_tuning_config(num_splits)
    assert config.dynamic_tensor_specs[0].gen_tuning_buckets(max_num_tokens) == expected_buckets


def test_dsv4_ob_cute_compile_key_excludes_runtime_shapes() -> None:
    runner = cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner
    tactic = runner._SPLIT_K1_TACTICS[0]
    key = runner._get_compile_key(16384, 1, *tactic, 74, True)

    assert key == runner._get_compile_key(16384, 1, *tactic, 74, True)
    assert key != runner._get_compile_key(16384, 1, *tactic, 80, True)


def test_dsv4_ob_tuning_input_hook_restores_packed_scale_layout() -> None:
    m, n, k, packed_k = 37, 128, 1024, 2
    inputs = [
        torch.empty((m, k), dtype=torch.float8_e4m3fn),
        torch.empty((m, packed_k), dtype=torch.int32),
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.empty((n, packed_k), dtype=torch.int32),
        torch.empty((2 * m, n), dtype=torch.bfloat16),
    ]

    prepared = cute_dsl_custom_ops._prepare_dsv4_ob_tuning_inputs(inputs)

    assert prepared[1].shape == (m, packed_k)
    assert prepared[1].stride() == (1, 40)
    assert prepared[1].untyped_storage().nbytes() == 40 * packed_k * 4
    assert prepared[4] is inputs[4]


def test_dsv4_deep_gemm_ob_warmup_uses_startup_autotuner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm import deep_gemm

    weight = torch.empty((128, 512), device="meta", dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((128, 1), device="meta", dtype=torch.int32)
    tuning_calls = []
    gemm_calls = []

    def choose_one(custom_op, runners, tuning_config, inputs):
        tuning_calls.append((custom_op, runners, tuning_config, inputs))

    def deep_gemm_nt(a, b, output, **kwargs):
        gemm_calls.append((a, b, output, kwargs))

    tuner = SimpleNamespace(is_tuning_mode=True, choose_one=choose_one)
    monkeypatch.setattr(AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", deep_gemm_nt)
    cute_dsl_custom_ops.warmup_dsv4_deep_gemm_ob(weight, weight_scale, torch.bfloat16, 37)

    assert len(tuning_calls) == 1
    custom_op, runners, tuning_config, inputs = tuning_calls[0]
    assert custom_op == "trtllm::dsv4_deep_gemm_ob::startup_warmup"
    runner = runners[0]
    assert isinstance(runner, cute_dsl_custom_ops.Dsv4DeepGemmObWarmupRunner)
    assert runner.get_valid_tactics([], OptimizationProfile()) == [0]
    assert tuning_config.exclude_from_cache
    assert tuning_config.inputs_pre_hook is cute_dsl_custom_ops._prepare_dsv4_ob_tuning_inputs
    assert tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets(8192) == tuple(
        range(8, 128, 8)
    ) + tuple(range(128, 8192, 128))
    assert inputs[0].shape == (37, 512)
    assert inputs[1].shape == (37, 1)
    assert inputs[1].stride() == (1, 40)
    assert inputs[1].untyped_storage().nbytes() == 40 * 4
    assert inputs[2] is weight
    assert inputs[3] is weight_scale
    assert inputs[4].shape == (37, 128)

    assert runner(inputs, tactic=0) is inputs[4]
    assert len(gemm_calls) == 1
    assert gemm_calls[0][0][0] is inputs[0]
    assert gemm_calls[0][0][1] is inputs[1]
    assert gemm_calls[0][1][0] is inputs[2]
    assert gemm_calls[0][1][1] is inputs[3]
    assert gemm_calls[0][2] is inputs[4]
    assert gemm_calls[0][3] == {"c": None, "disable_ue8m0_cast": False}


def test_dsv4_model_warmup_deduplicates_deep_gemm_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def make_attention(enabled: bool = True):
        return SimpleNamespace(
            _should_warmup_dsv4_deep_gemm_ob=lambda: enabled,
            o_b_proj=SimpleNamespace(
                weight=torch.empty((128, 512), device="meta", dtype=torch.float8_e4m3fn),
                weight_scale=torch.empty((1, 4), device="meta"),
            ),
            dtype=torch.bfloat16,
        )

    first = make_attention()
    duplicate = make_attention()
    disabled = make_attention(enabled=False)
    module = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=first),
                SimpleNamespace(self_attn=duplicate),
                SimpleNamespace(self_attn=disabled),
            ]
        )
    )
    calls = []

    def warmup(weight, weight_scale, output_dtype, max_num_tokens):
        calls.append((weight, weight_scale, output_dtype, max_num_tokens))

    monkeypatch.setattr(cute_dsl_custom_ops, "warmup_dsv4_deep_gemm_ob", warmup)
    DeepseekV4ForCausalLM.warmup_dsv4_fused_ob(module, 8192)

    assert len(calls) == 1
    assert calls[0][0] is first.o_b_proj.weight
    assert calls[0][1] is first.o_b_proj.weight_scale
    assert calls[0][2:] == (torch.bfloat16, 8192)


def _check_dsv4_oa_fp8out_tensor_contract() -> None:
    batch_size, m, n, k = 4, 3, 128, 128
    sf_m, sf_k, sf_n = 4, 1, 1
    packed_sf_n = 1
    a = torch.empty((batch_size, m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty((batch_size, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((batch_size, sf_k, sf_m), device="cuda", dtype=torch.float32)
    b_sf = torch.empty((batch_size, sf_n, sf_k), device="cuda", dtype=torch.float32)
    output = torch.empty((m, batch_size * n), device="cuda", dtype=torch.float8_e4m3fn)
    sf_padded = torch.empty_strided(
        (sf_m, packed_sf_n), (1, sf_m), device="cuda", dtype=torch.int32
    )
    sf_out = sf_padded[:m]
    inputs = [a, b, a_sf, b_sf, output, sf_out]
    runner = cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner

    assert runner._validate_fp8out_inputs(inputs) == (batch_size, m, n, k, sf_m, sf_n, sf_k)

    def replace(index: int, tensor: torch.Tensor) -> list[torch.Tensor]:
        invalid = inputs.copy()
        invalid[index] = tensor
        return invalid

    noncontiguous_a = torch.empty((batch_size, m, k + 1), device="cuda", dtype=torch.float8_e4m3fn)[
        ..., :k
    ]
    overlapping_output = torch.empty((m, 1), device="cuda", dtype=torch.float8_e4m3fn).expand(
        -1, batch_size * n
    )
    bad_sf_storage = torch.empty(sf_m * 2, device="cuda", dtype=torch.int32)
    bad_sf_shape = torch.as_strided(bad_sf_storage, (m, 2), (1, sf_m))
    invalid_cases = (
        (replace(0, torch.empty_like(a, device="cpu")), ValueError, "CUDA tensors"),
        (replace(0, a.to(torch.float16)), TypeError, "input must have dtype"),
        (replace(0, a.reshape(batch_size * m, k)), ValueError, "ranks"),
        (replace(0, a[..., :64].contiguous()), ValueError, "batch/K dimensions"),
        (replace(2, a_sf[..., :3]), ValueError, "input_scale must have shape"),
        (replace(0, noncontiguous_a), ValueError, "input must be contiguous"),
        (replace(4, overlapping_output), ValueError, "output_fp8 must be"),
        (replace(5, bad_sf_shape), ValueError, "sf_out must have shape"),
    )
    for invalid, error_type, match in invalid_cases:
        with pytest.raises(error_type, match=match):
            runner._validate_fp8out_inputs(invalid)


@skip_pre_blackwell
@pytest.mark.parametrize("m", [3, 37, 129])
def test_dsv4_oa_fp8out_clears_packed_scale_padding(m: int) -> None:
    if get_sm_version() // 10 != 10:
        pytest.skip("DSV4 O_a FP8 output requires the SM100 family")

    batch_size, n, k = 4, 128, 128
    sf_m = (m + 3) // 4 * 4
    a = torch.zeros((batch_size, m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros((batch_size, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.ones((batch_size, 1, sf_m), device="cuda")
    b_sf = torch.ones((batch_size, 1, 1), device="cuda")
    output = torch.empty((m, batch_size * n), device="cuda", dtype=torch.float8_e4m3fn)
    sf_padded = torch.empty_strided((sf_m, 1), (1, sf_m), device="cuda", dtype=torch.int32).fill_(
        0x7F7F7F7F
    )

    torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell_fp8out(a, b, a_sf, b_sf, output, sf_padded[:m])

    assert torch.count_nonzero(sf_padded[m:]).item() == 0


def _make_dsv4_ob_packed_scale(n: int, k: int) -> torch.Tensor:
    packed_k = k // 512
    storage = torch.empty((packed_k, n), device="meta", dtype=torch.int32)
    return torch.as_strided(storage, (n, packed_k), (1, n))


def _make_dsv4_ob_meta_case(
    m: int, *, packed_weight_scale: bool
) -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor]:
    n, k = 7168, 16384
    packed_k = k // 512
    weight_scale = (
        _make_dsv4_ob_packed_scale(n, k)
        if packed_weight_scale
        else torch.empty((n // 128, k // 128), device="meta")
    )
    module = SimpleNamespace(
        hidden_size=n,
        dtype=torch.bfloat16,
        o_b_proj=SimpleNamespace(
            weight=torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn),
            weight_scale=weight_scale,
        ),
    )
    activation = torch.empty((m, k), device="meta", dtype=torch.float8_e4m3fn)
    aligned_m = (m + 3) // 4 * 4
    activation_scale = torch.empty_strided(
        (m, packed_k), (1, aligned_m), device="meta", dtype=torch.int32
    )
    return module, activation, activation_scale


def test_dsv4_ob_split_k_one_uses_cute_dsl_and_caches_weight_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm import deep_gemm
    from tensorrt_llm.quantization.utils import fp8_utils

    m = 256
    module, activation, activation_scale = _make_dsv4_ob_meta_case(m, packed_weight_scale=False)
    n, k = module.hidden_size, activation.shape[1]
    transformed_scale = _make_dsv4_ob_packed_scale(n, k)
    calls = []
    transforms = []

    def fail_deep_gemm(*args, **kwargs):
        raise AssertionError("supported SK1 must not dispatch to DeepGEMM")

    def splitk_gemm(a, sfa, b, sfb):
        calls.append((a, sfa, b, sfb))
        return torch.empty((m, n), device=a.device, dtype=torch.bfloat16)

    def transform_weight_scale(scale, **kwargs):
        transforms.append((scale, kwargs))
        return transformed_scale

    monkeypatch.setenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", "1")
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", "1")
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", fail_deep_gemm)
    monkeypatch.setattr(fp8_utils, "transform_sf_into_required_layout", transform_weight_scale)
    monkeypatch.setattr(torch.ops.trtllm, "dsv4_fp8_splitk_gemm", splitk_gemm)
    output = MLA._fused_ob_gemm(module, activation, activation_scale, m)
    cached_output = MLA._fused_ob_gemm(module, activation, activation_scale, m)

    assert output.shape == (m, n)
    assert cached_output.shape == (m, n)
    assert output.device.type == "meta"
    assert len(transforms) == 1
    assert transforms[0][0] is module.o_b_proj.weight_scale
    assert module.o_b_proj._ob_wsf_int is transformed_scale
    assert len(calls) == 2
    assert calls[0][0] is activation
    assert calls[0][1] is activation_scale
    assert calls[0][2] is module.o_b_proj.weight
    assert calls[0][3] is transformed_scale


@pytest.mark.parametrize(("m", "expected_split"), [(1, 4), (32, 2), (64, 2), (128, 2), (160, 1)])
def test_dsv4_ob_auto_split_uses_cute_dsl(
    m: int, expected_split: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm import deep_gemm

    module, activation, activation_scale = _make_dsv4_ob_meta_case(m, packed_weight_scale=True)
    n = module.hidden_size
    calls = []

    def fail_deep_gemm(*args, **kwargs):
        raise AssertionError("automatic O_b must not dispatch to DeepGEMM")

    def splitk_gemm(a, sfa, b, sfb):
        calls.append((a, sfa, b, sfb))
        return torch.empty((expected_split * m, n), device=a.device, dtype=torch.bfloat16)

    monkeypatch.setenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", "1")
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", fail_deep_gemm)
    monkeypatch.setattr(torch.ops.trtllm, "dsv4_fp8_splitk_gemm", splitk_gemm)
    output = MLA._fused_ob_gemm(module, activation, activation_scale, m)

    assert output.shape == (expected_split * m, n)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("m", "enable_cute"),
    [(160, False), (0, True)],
    ids=["default", "zero-m"],
)
def test_dsv4_ob_uses_deep_gemm_fallback(
    m: int, enable_cute: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm import deep_gemm
    from tensorrt_llm.quantization.utils import fp8_utils

    module, activation, activation_scale = _make_dsv4_ob_meta_case(m, packed_weight_scale=False)
    n = module.hidden_size
    calls = []

    def deep_gemm_nt(inputs, weights, output, **kwargs):
        calls.append((inputs, weights, output, kwargs))

    def fail_cute(*args, **kwargs):
        raise AssertionError("disabled CuTe O_b must dispatch to DeepGEMM")

    def fail_transform(*args, **kwargs):
        raise AssertionError("DeepGEMM fallback must not transform the weight scale")

    def fail_autotuner(*args, **kwargs):
        raise AssertionError("DeepGEMM inference must not consult AutoTuner")

    if enable_cute:
        monkeypatch.setenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", "1")
    else:
        monkeypatch.delenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", raising=False)
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", deep_gemm_nt)
    monkeypatch.setattr(fp8_utils, "transform_sf_into_required_layout", fail_transform)
    monkeypatch.setattr(AutoTuner, "get", fail_autotuner)
    monkeypatch.setattr(torch.ops.trtllm, "dsv4_fp8_splitk_gemm", fail_cute)
    output = MLA._fused_ob_gemm(module, activation, activation_scale, m)

    assert output.shape == (m, n)
    assert output.device.type == "meta"
    assert len(calls) == 1
    assert calls[0][0] == (activation, activation_scale)
    assert calls[0][1] == (module.o_b_proj.weight, module.o_b_proj.weight_scale)
    assert calls[0][2] is output


@skip_pre_blackwell
@pytest.mark.parametrize(
    ("num_tokens", "num_splits"),
    [(1, 2), (2, 2), (32, 2), (64, 2), (128, 2), (256, 1), (512, 1), (64, 4)],
)
def test_dsv4_pro_fp8_splitk_gemm_partials(
    num_tokens: int, num_splits: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if get_sm_version() // 10 != 10:
        pytest.skip("dsv4_fp8_splitk_gemm requires the SM100 family")

    n, k = 7168, 16384
    aligned_m = (num_tokens + 3) // 4 * 4
    packed_k = k // 512
    a = torch.full((num_tokens, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.full((n, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)

    # Pack four unit UE8M0 scales per int32.
    sfa_storage = torch.full((aligned_m * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    sfb_storage = torch.full((n * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    sfa = torch.as_strided(sfa_storage, (num_tokens, packed_k), (1, aligned_m))
    sfb = torch.as_strided(sfb_storage, (n, packed_k), (1, n))
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", str(num_splits))
    output = torch.ops.trtllm.dsv4_fp8_splitk_gemm(a, sfa, b, sfb)
    partials = output.view(num_splits, num_tokens, n)
    expected_partial = (k // num_splits) * 0.03125 * 0.03125
    torch.testing.assert_close(
        partials.float(), torch.full_like(partials, expected_partial, dtype=torch.float32)
    )


@skip_pre_blackwell
def test_dsv4_pro_fp8_splitk_gemm_reuses_kernels_and_captures_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if get_sm_version() // 10 != 10:
        pytest.skip("dsv4_fp8_splitk_gemm requires the SM100 family")

    n, k = 7168, 16384
    packed_k = k // 512
    b = torch.full((n, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)
    sfb_storage = torch.full((n * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    sfb = torch.as_strided(sfb_storage, (n, packed_k), (1, n))
    runner = cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner
    tuner = AutoTuner.get()
    monkeypatch.delenv("TRTLLM_DSV4_OB_SPLIT_K", raising=False)
    monkeypatch.setattr(runner, "kernel_cache", {})
    tuner.clear_cache()
    tuner.stats.cache_misses = 0

    def run(num_tokens: int) -> torch.Tensor:
        aligned_m = (num_tokens + 3) // 4 * 4
        a = torch.full((num_tokens, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)
        sfa_storage = torch.full(
            (aligned_m * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32
        )
        sfa = torch.as_strided(sfa_storage, (num_tokens, packed_k), (1, aligned_m))
        return torch.ops.trtllm.dsv4_fp8_splitk_gemm(a, sfa, b, sfb)

    # Context-only engine warmup must autotune and compile every generation
    # output contract before capture begins.
    with autotune():
        run(256)
    compiled_kernels = len(runner.kernel_cache)
    assert compiled_kernels == sum(
        len(tactics)
        for tactics in (
            runner._SPLIT_K1_TACTICS,
            runner._SPLIT_K2_TACTICS,
            runner._SPLIT_K4_TACTICS,
        )
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("CuTe JIT occurred after AutoTuner warmup")

    monkeypatch.setattr(cute_dsl_custom_ops.cute, "compile", fail_compile)

    # Cached M=1/128/256 contracts must not compile or fall back at runtime.
    for num_tokens in (1, 37, 128, 256):
        output = run(num_tokens)
        num_splits = runner._select_num_splits(num_tokens)
        expected_partial = (k // num_splits) * 0.03125 * 0.03125
        torch.testing.assert_close(
            output[0, 0].float(), torch.tensor(expected_partial, device="cuda")
        )
    assert len(runner.kernel_cache) == compiled_kernels
    assert tuner.stats.cache_misses == 0

    # Capture an irregular shape after warmup and verify replay does not compile.
    num_tokens = 37
    aligned_m = (num_tokens + 3) // 4 * 4
    a = torch.full((num_tokens, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)
    sfa_storage = torch.full((aligned_m * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    sfa = torch.as_strided(sfa_storage, (num_tokens, packed_k), (1, aligned_m))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = torch.ops.trtllm.dsv4_fp8_splitk_gemm(a, sfa, b, sfb)
    graph.replay()
    torch.cuda.synchronize()
    expected_partial = (k // 2) * 0.03125 * 0.03125
    torch.testing.assert_close(
        graph_output[0, 0].float(), torch.tensor(expected_partial, device="cuda")
    )
    assert len(runner.kernel_cache) == compiled_kernels
    assert tuner.stats.cache_misses == 0


@skip_pre_blackwell
def test_dsv4_pro_fp8_splitk_gemm_packed_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    if get_sm_version() // 10 != 10:
        pytest.skip("dsv4_fp8_splitk_gemm requires the SM100 family")

    m, n, k, num_splits = 64, 128, 2048, 4
    k_blocks = k // 128
    torch.manual_seed(1234)
    a = (torch.randn((m, k), device="cuda") * 0.125).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.125).to(torch.float8_e4m3fn)
    exp_a = torch.randint(124, 131, (m, k_blocks), device="cuda")
    exp_b = torch.randint(124, 131, (n // 128, k_blocks), device="cuda").repeat_interleave(
        128, dim=0
    )

    def pack_scales(exponents: torch.Tensor, aligned_rows: int) -> torch.Tensor:
        rows, num_blocks = exponents.shape
        grouped = exponents.reshape(rows, num_blocks // 4, 4).to(torch.int32)
        packed = (
            grouped[..., 0]
            | (grouped[..., 1] << 8)
            | (grouped[..., 2] << 16)
            | (grouped[..., 3] << 24)
        )
        storage = torch.zeros((num_blocks // 4, aligned_rows), device="cuda", dtype=torch.int32)
        storage[:, :rows] = packed.transpose(0, 1)
        return torch.as_strided(storage, packed.shape, (1, aligned_rows))

    sfa = pack_scales(exp_a, m)
    sfb = pack_scales(exp_b, n)
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", str(num_splits))
    output = torch.ops.trtllm.dsv4_fp8_splitk_gemm(a, sfa, b, sfb)
    partials = output.view(num_splits, m, n)

    scale_a = torch.exp2(exp_a.float() - 127.0)
    scale_b = torch.exp2(exp_b.float() - 127.0)
    a_dequant = (a.float().reshape(m, k_blocks, 128) * scale_a[..., None]).reshape(m, k)
    b_dequant = (b.float().reshape(n, k_blocks, 128) * scale_b[..., None]).reshape(n, k)
    split_k = k // num_splits
    expected = torch.stack(
        [
            a_dequant[:, split * split_k : (split + 1) * split_k]
            @ b_dequant[:, split * split_k : (split + 1) * split_k].T
            for split in range(num_splits)
        ]
    )
    torch.testing.assert_close(partials.float(), expected, rtol=0.02, atol=0.05)


def _per_token_fp8_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Simulate the fused inverse-RoPE FP8 quantization consumed by o_a_proj."""
    original_shape = x.shape
    flattened_x = x.reshape(-1, original_shape[-1])
    fp8_x, scale = per_token_cast_to_fp8_e8m0(flattened_x)
    dequant_x = (fp8_x.view(flattened_x.shape[0], -1, 128).float() * scale.unsqueeze(-1)).view_as(
        flattened_x
    )
    return dequant_x.to(x.dtype).reshape(original_shape)


def calculate_reference_deepseek_v4_o_proj(
    attn_out_latent,
    o_a_proj,
    o_b_proj_weight,
    freqs_cis,
    n_local_groups,
    qk_nope_head_dim,
    qk_rope_head_dim,
    device,
    is_fp8: bool = False,
):
    """
    Reference implementation for DeepSeek-V4 output projection based on ref/model.py.

    Args:
        attn_out_latent: [num_tokens, num_heads, qk_head_dim] attention output
        o_a_proj: [n_local_groups, o_lora_rank, num_heads * qk_head_dim // n_groups]
        o_b_proj_weight: [hidden_size, n_groups * o_lora_rank]
        freqs_cis: [num_toknes, rope_head_dim / 2] rotary embeddings
        n_local_groups: Number of local output projection groups
        qk_nope_head_dim: Dimension of non-positional part
        qk_rope_head_dim: Dimension of positional part
        device: Device to run on
        is_fp8: Whether test fp8 or bf16

    Returns:
        output: [num_tokens, hidden_size] projected output
    """
    num_tokens = attn_out_latent.shape[0]

    # Apply RoPE to attn_out_pe
    attn_out_latent = attn_out_latent.unsqueeze(0)
    apply_rotary_emb(attn_out_latent[..., -qk_rope_head_dim:], freqs_cis, inverse=True)

    # Reshape for grouped projection
    attn_out_grouped = attn_out_latent.view(num_tokens, n_local_groups, -1)
    if is_fp8:
        attn_out_grouped = _per_token_fp8_quant_dequant(
            attn_out_grouped.transpose(0, 1).contiguous()
        ).transpose(0, 1)

    # Apply o_a_proj: einsum equivalent to bmm
    o_lora = torch.einsum("tgd,grd->tgr", attn_out_grouped, o_a_proj)

    # Flatten and apply o_b_proj, [num_tokens, n_local_groups * o_lora_rank]
    o_lora_flat = o_lora.flatten(1)
    if is_fp8:
        o_lora_flat = o_lora_flat.to(torch.float8_e4m3fn).to(torch.bfloat16)
    output = torch.nn.functional.linear(o_lora_flat, o_b_proj_weight)  # [num_tokens, hidden_size]

    return output


def _build_dsv4_o_proj_case(
    num_tokens: int,
    dtype_str: str,
    device: torch.device,
    use_cute_dsl_blockscaling_mm: bool | None = None,
) -> tuple[MLA, torch.Tensor, torch.Tensor, SimpleNamespace]:
    """Build an MLA module, inputs, and reference-path tensors for the DeepSeek-V4
    o_proj tests. The blockscaling flag selects the O_b backend without changing
    the generated tensors.

    Returns:
        (mla, attn_out_latent, position_ids, refs) where ``refs`` is a namespace
        carrying the dequantized weights / freqs_cis / dims the analytic
        reference path consumes.
    """
    dtype = torch.bfloat16

    # Model configuration matching the reference model
    num_heads = 64
    q_lora_rank = 1024
    kv_lora_rank = 448
    qk_nope_head_dim = 448
    qk_rope_head_dim = 64
    v_head_dim = 512
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_size = 4096
    max_position_embeddings = 65536
    o_lora_rank = 1024
    num_groups = 8
    n_local_groups = num_groups  # no TP in this test

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create RoPE config
    rope_config = RopeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 4,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        qk_rope_head_dim=qk_rope_head_dim,
        model_type="deepseek_v4",
    )

    # Setup model config with deepseek_v4 sparse attention
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
    )

    # Create sparse attention config for deepseek_v4
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
    )

    quant_config = QuantConfig()
    if dtype_str == "fp8":
        quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        quant_config.group_size = 128

    if use_cute_dsl_blockscaling_mm is None:
        use_cute_dsl_blockscaling_mm = dtype_str == "fp8"
    model_config = ModelConfig(
        mapping=mapping,
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_config,
        quant_config=quant_config,
        use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
    )

    # Setup positional embedding params
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    # Create MLA module with deepseek_v4 configuration
    mla = MLA(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        predicted_tokens_per_seq=1,
        max_position_embeddings=max_position_embeddings,
        bias=False,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=dtype,
        config=model_config,
        num_groups=num_groups,
        o_lora_rank=o_lora_rank,
    ).to(device)

    # Initialize weights
    nn_init_std = 0.02
    fp8_a_weight = fp8_a_scale = o_a_proj_bf16 = fp8_b_weight_dequant = dim = None
    with torch.no_grad():
        # Initialize o_a_proj weights
        if dtype_str == "bf16":
            mla.o_a_proj.data = (
                torch.randn(
                    n_local_groups,
                    o_lora_rank,
                    num_heads * qk_head_dim // num_groups,
                    dtype=dtype,
                    device=device,
                )
                * nn_init_std
            )
        elif dtype_str == "fp8":
            dim = num_heads * qk_head_dim // num_groups
            o_a_proj_bf16 = (
                torch.randn(n_local_groups, o_lora_rank, dim, dtype=torch.bfloat16, device=device)
                * nn_init_std
            )

            fp8_a_weight, fp8_a_scale = per_block_cast_to_fp8_e8m0(o_a_proj_bf16.reshape(-1, dim))
            fp8_a_weight = fp8_a_weight.reshape(n_local_groups, o_lora_rank, dim)
            mla.o_a_proj.data = fp8_a_weight
            mla.o_a_proj_scale.data = fp8_a_scale.reshape(
                n_local_groups, o_lora_rank // 128, dim // 128
            )
            # SM100 keeps only the quantized O_a weights.

        # Initialize o_b_proj weights
        if dtype_str == "bf16":
            mla.o_b_proj.weight.data = (
                torch.randn(hidden_size, num_groups * o_lora_rank, dtype=dtype, device=device)
                * nn_init_std
            )
        elif dtype_str == "fp8":
            # For FP8, properly quantize using fp8_quantize_1x128_sf_transpose
            o_b_proj_weight_bf16 = (
                torch.randn(
                    hidden_size, num_groups * o_lora_rank, dtype=torch.bfloat16, device=device
                )
                * nn_init_std
            )

            # Quantize the weight
            fp8_b_weight, fp8_b_scale = per_block_cast_to_fp8_e8m0(o_b_proj_weight_bf16)
            fp8_b_weight_dequant = weight_dequant(fp8_b_weight, fp8_b_scale).bfloat16()
            mla.o_b_proj.weight.data = fp8_b_weight
            mla.o_b_proj.weight_scale.data = fp8_b_scale
            if not use_cute_dsl_blockscaling_mm:
                # Match the post-load layout consumed by DeepGEMM.
                mla.o_b_proj.transform_weights()

    # Generate test inputs
    # Note: for deepseek_v4, kv_lora_rank equals qk_head_dim
    attn_out_latent = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Dequantize the weights actually consumed by the FP8 path.
    if dtype_str == "bf16":
        o_a_proj_ref = mla.o_a_proj.data
        o_b_proj_weight_ref = mla.o_b_proj.weight.data
    else:
        o_a_proj_ref = (
            weight_dequant(
                fp8_a_weight.reshape(-1, dim).contiguous(),
                fp8_a_scale.contiguous(),
            )
            .bfloat16()
            .reshape(o_a_proj_bf16.shape)
        )
        o_b_proj_weight_ref = fp8_b_weight_dequant

    freqs_cis = precompute_freqs_cis(
        qk_rope_head_dim,
        num_tokens,
        max_position_embeddings,
        rope_config.rope_theta,
        rope_config.rope_scaling["factor"],
        rope_config.rope_scaling["beta_fast"],
        rope_config.rope_scaling["beta_slow"],
    ).to(device)

    refs = SimpleNamespace(
        o_a_proj=o_a_proj_ref,
        o_b_proj_weight=o_b_proj_weight_ref,
        freqs_cis=freqs_cis,
        n_local_groups=n_local_groups,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    return mla, attn_out_latent, position_ids, refs


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("dtype_str", ["bf16", "fp8"])
def test_deepseek_v4_o_proj(num_tokens: int, dtype_str: str) -> None:
    """Test DeepSeek-V4 output projection (_deepseek_v4_o_proj)."""
    print(
        f"\n{'=' * 80}\nTesting: deepseek_v4_o_proj num_tokens={num_tokens} dtype={dtype_str}\n{'=' * 80}"
    )

    if dtype_str == "fp8" and get_sm_version() < 100:
        pytest.skip("FP8 is not supported on pre-Blackwell architectures")

    device = torch.device("cuda")

    mla, attn_out_latent, position_ids, refs = _build_dsv4_o_proj_case(
        num_tokens, dtype_str, device
    )

    # Preserve the input because inverse RoPE is in-place.
    output = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    reference_output = calculate_reference_deepseek_v4_o_proj(
        attn_out_latent=attn_out_latent,
        o_a_proj=refs.o_a_proj,
        o_b_proj_weight=refs.o_b_proj_weight,
        freqs_cis=refs.freqs_cis[0:num_tokens],
        n_local_groups=refs.n_local_groups,
        qk_nope_head_dim=refs.qk_nope_head_dim,
        qk_rope_head_dim=refs.qk_rope_head_dim,
        device=device,
        is_fp8=dtype_str == "fp8",
    )

    # Validate output shapes
    assert output.shape == reference_output.shape, (
        f"Shape mismatch: output {output.shape} vs reference {reference_output.shape}"
    )
    assert output.dtype == reference_output.dtype, (
        f"Dtype mismatch: output {output.dtype} vs reference {reference_output.dtype}"
    )
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert torch.isfinite(reference_output).all(), "Reference output contains non-finite values"

    # Compare results
    abs_error = (output - reference_output).abs()
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()

    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    if dtype_str == "fp8":
        diff = _calc_diff(output, reference_output)
        assert diff < FP8_O_PROJ_DIFF_TOL, f"{diff=}"
    else:
        torch.testing.assert_close(output, reference_output, rtol=0.1, atol=0.1)
        print(f"  ✓ Test passed for num_tokens={num_tokens}, dtype={dtype_str}\n")


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_tokens", [1, 16, 32, 128, 256])
def test_deepseek_v4_o_proj_fused_fp8_equivalence(
    num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fused O_a output must feed default DeepGEMM and opt-in CuTe O_b."""
    if get_sm_version() < 100:
        pytest.skip("fused DeepSeek-V4 FP8 O-projection requires Blackwell (SM100+)")

    device = torch.device("cuda")
    mla, attn_out_latent, position_ids, refs = _build_dsv4_o_proj_case(
        num_tokens, "fp8", device, use_cute_dsl_blockscaling_mm=False
    )
    # The standalone MLA fixture has no decoder consumer to advertise split
    # output support, so opt in explicitly for this fused-path test.
    mla.allow_dsv4_split_output = True
    monkeypatch.setattr(cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner, "kernel_cache", {})
    if num_tokens == 1:
        _check_dsv4_oa_fp8out_tensor_contract()

    # Ensure the fused path is eligible.
    assert mla.allow_dsv4_split_output
    assert mla.o_a_proj.dtype == torch.float8_e4m3fn
    assert mla.n_local_groups == mla.num_groups
    assert getattr(mla.o_b_proj, "tp_size", 1) == 1
    assert mla.o_b_proj.has_fp8_block_scales
    assert not getattr(mla.o_b_proj, "use_cute_dsl_blockscaling_mm", False)

    # Explicit kill switch retains the unfused fallback.
    monkeypatch.setenv("TRTLLM_DSV4_DISABLE_FUSED_OPROJ", "1")
    out_unfused = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    monkeypatch.delenv("TRTLLM_DSV4_DISABLE_FUSED_OPROJ", raising=False)
    # Keep the standalone CuTe result model-shaped.
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", "1")
    fused_outputs: dict[str, torch.Tensor] = {}
    monkeypatch.delenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", raising=False)
    fused_outputs["DeepGEMM"] = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)
    monkeypatch.setenv("TRTLLM_DSV4_ENABLE_CUTE_DSL_OB_PROJ", "1")
    fused_outputs["CuTe"] = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    # Analytic reference (same correctness bar as the unfused correctness test).
    reference_output = calculate_reference_deepseek_v4_o_proj(
        attn_out_latent=attn_out_latent.clone(),
        o_a_proj=refs.o_a_proj,
        o_b_proj_weight=refs.o_b_proj_weight,
        freqs_cis=refs.freqs_cis[0:num_tokens],
        n_local_groups=refs.n_local_groups,
        qk_nope_head_dim=refs.qk_nope_head_dim,
        qk_rope_head_dim=refs.qk_rope_head_dim,
        device=device,
        is_fp8=True,
    )

    for backend, output in fused_outputs.items():
        diff_vs_ref = _calc_diff(output, reference_output)
        diff_vs_unfused = _calc_diff(output, out_unfused)
        print(
            f"\n  num_tokens={num_tokens} backend={backend} "
            f"diff(ref)={diff_vs_ref:.3e} diff(unfused)={diff_vs_unfused:.3e}"
        )
        assert output.shape == out_unfused.shape
        assert output.dtype == out_unfused.dtype
        assert torch.isfinite(output).all()
        assert diff_vs_ref < FP8_O_PROJ_DIFF_TOL, f"{backend} {diff_vs_ref=}"
        assert diff_vs_unfused < 1e-3, f"{backend} {diff_vs_unfused=}"

    backend_diff = _calc_diff(fused_outputs["CuTe"], fused_outputs["DeepGEMM"])
    assert backend_diff < 1e-3, f"{backend_diff=}"

    expected_smem_epilogue = num_tokens <= 32
    expected_smem_row_iters = (num_tokens + 15) // 16 if expected_smem_epilogue else 1
    fp8out_keys = [
        key
        for key in cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache
        if key[0] == "fp8out"
        and key[-2] == expected_smem_row_iters
        and key[-1] == expected_smem_epilogue
    ]
    assert len(fp8out_keys) == 1

    if num_tokens == 16:
        compiled_gemm = cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache[
            fp8out_keys[0]
        ]
        out_cached = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)
        assert (
            cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache[fp8out_keys[0]]
            is compiled_gemm
        )
        torch.testing.assert_close(out_cached, fused_outputs["CuTe"], rtol=0, atol=0)
