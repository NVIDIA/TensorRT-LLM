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
"""CPU contracts of the MegaMoE CuteDSL combine format.

``MEGAMOE_COMBINE_FORMAT`` wins when set and applies to every capacity rung;
otherwise ``use_low_precision_moe_combine`` selects MXFP8 for the SM107 rungs of
at least 4096 tokens per rank and BF16 everywhere else. The backend is the real
class built on CPU, with the provider query and the symmetric provider mocked,
so these tests claim no kernel behavior.
"""

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as backend_module
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_ENV = "MEGAMOE_COMBINE_FORMAT"
_FLAG = "moe_config.use_low_precision_moe_combine"
_MXFP8 = "32e4m3xe8m0"
_NVFP4 = "16e2m1xbf16"
_SM107_RUNGS = [32, 64, 128, 192, 256, 384, 512, 1024, 4096, 8192]
_GLM_MEGAMOE = Path(__file__).resolve().parents[4] / (
    "tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/"
    "glm_5_2_nvfp4__sm_107__high_throughput/ops/moe/megamoe.py"
)


class _Bf16OnlyMegaMoE(MegaMoECuteDsl):
    _honors_low_precision_combine = False


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """SM107 unless a test sets ``env.sm``; records the INFO lines the backend logs once."""
    for name in (_ENV, "MEGAMOE_UGPU_LAYOUT", "MEGAMOE_TACTIC_AUTOTUNE"):
        monkeypatch.delenv(name, raising=False)
    state = SimpleNamespace(sm=107, info_once=Mock())
    monkeypatch.setattr(backend_module, "get_sm_version", lambda: state.sm)
    monkeypatch.setattr(backend_module.logger, "info_once", state.info_once)
    return state


def _backend(flag: bool, cls: type = MegaMoECuteDsl, max_num_tokens: int = 8192) -> MegaMoECuteDsl:
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
        use_low_precision_moe_combine=flag,
        max_num_tokens=max_num_tokens,
    )
    return cls(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
    )


def _formats(backend: MegaMoECuteDsl) -> dict:
    return {rung: backend._combine_format_for(rung) for rung in backend._maxt_buckets}


def _logged(env: SimpleNamespace) -> list:
    return [call.args[0] for call in env.info_once.call_args_list]


@pytest.mark.parametrize(
    "flag,env_format,expected,min_tokens,reason",
    [
        (False, None, "bf16", 0, "the default"),
        (True, None, _MXFP8, 4096, f"set by {_FLAG}"),
        (False, _MXFP8, _MXFP8, 0, f"set by {_ENV}"),
        (False, _NVFP4, _NVFP4, 0, f"set by {_ENV}"),
        (True, "bf16", "bf16", 0, f"set by {_ENV}, which overrides {_FLAG}"),
        (True, _NVFP4, _NVFP4, 0, f"set by {_ENV}, which overrides {_FLAG}"),
    ],
)
def test_env_wins_then_the_flag_selects_mxfp8_on_sm107(
    env, monkeypatch, flag, env_format, expected, min_tokens, reason
) -> None:
    if env_format is not None:
        monkeypatch.setenv(_ENV, env_format)
    backend = _backend(flag)
    assert backend.combine_format == expected
    assert backend._combine_format_min_tokens == min_tokens
    assert backend._combine_format_reason == reason
    assert any(message.endswith(f"({reason})") for message in _logged(env))


def test_the_flag_moves_only_rungs_from_4096_to_mxfp8(env) -> None:
    assert MegaMoECuteDsl._SM107_LOW_PRECISION_COMBINE_MIN_TOKENS == 4096
    backend = _backend(True)
    assert backend._maxt_buckets == _SM107_RUNGS
    assert _formats(backend) == {rung: _MXFP8 if rung >= 4096 else "bf16" for rung in _SM107_RUNGS}
    assert backend._quantized_combine_rungs() == [4096, 8192]
    assert (
        f"[MegaMoECuteDsl] combine format {_MXFP8} for capacity rungs [4096, 8192], "
        f"bf16 for {_SM107_RUNGS[:-2]}; {_MXFP8} starts at 4096 tokens per rank "
        f"(set by {_FLAG})"
    ) in _logged(env)
    # A ladder that stops below 4096 keeps BF16 throughout.
    small = _backend(True, max_num_tokens=2048)
    assert set(_formats(small).values()) == {"bf16"} and not small._quantized_combine_rungs()


@pytest.mark.parametrize("sm", [100, 103])
def test_the_flag_keeps_bf16_off_sm107_and_the_env_still_applies(env, monkeypatch, sm) -> None:
    env.sm = sm
    backend = _backend(True)
    assert set(_formats(backend).values()) == {"bf16"}
    assert backend._combine_format_reason == (
        f"{_FLAG} selects a quantized MegaMoE combine on SM107 only, not on SM{sm}"
    )
    assert any("bf16 for every capacity rung" in message for message in _logged(env))
    monkeypatch.setenv(_ENV, _MXFP8)
    assert set(_formats(_backend(True)).values()) == {_MXFP8}


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("env_format", ["", "bogus", "BF16"])
def test_invalid_env_is_rejected_with_or_without_the_flag(
    env, monkeypatch, env_format: str, flag: bool
) -> None:
    monkeypatch.setenv(_ENV, env_format)
    with pytest.raises(ValueError, match=f"{_ENV} must be one of"):
        _backend(flag)


def test_bf16_only_subclass_keeps_bf16_under_the_flag(env, monkeypatch) -> None:
    backend = _backend(True, _Bf16OnlyMegaMoE)
    assert set(_formats(backend).values()) == {"bf16"}
    assert "ignores " + _FLAG in backend._combine_format_reason
    # The environment variable still wins, as before this flag was honored.
    monkeypatch.setenv(_ENV, _MXFP8)
    assert set(_formats(_backend(True, _Bf16OnlyMegaMoE)).values()) == {_MXFP8}


def test_glm_rubin_megamoe_does_not_honor_the_flag() -> None:
    # Importing the GLM Rubin target loads its SM107 native library; read the source instead.
    text = _GLM_MEGAMOE.read_text()
    cls = next(
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.ClassDef) and node.name == "RubinMegaMoE"
    )
    values = {
        target.id: node.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    value = values["_honors_low_precision_combine"]
    assert isinstance(value, ast.Constant) and value.value is False
    # Its providers and launches take the format of their own rung.
    assert "combine_format=self.combine_format" not in text


@pytest.mark.parametrize(
    "flag,env_format,rungs",
    [(True, None, [4096, 8192]), (False, _MXFP8, _SM107_RUNGS)],
)
def test_ugpu_localization_refuses_any_quantized_rung(
    env, monkeypatch, flag, env_format, rungs
) -> None:
    if env_format is not None:
        monkeypatch.setenv(_ENV, env_format)
    backend = _backend(flag)
    backend.ep_size = 2
    with pytest.raises(ValueError, match="requires the BF16 combine, but the capacity rungs"):
        backend._configure_ugpu_localization("n_split")
    with pytest.raises(ValueError, match=re.escape(f"{rungs} combine in {_MXFP8}")):
        backend._configure_ugpu_localization("expert_half")
    assert backend._ugpu_weight_layout == "none"


def test_ugpu_localization_accepts_the_flag_on_an_all_bf16_ladder(env) -> None:
    backend = _backend(True, max_num_tokens=2048)
    backend.ep_size = 2
    backend._configure_ugpu_localization("n_split")
    assert backend._ugpu_weight_layout == "n_split"


@pytest.mark.parametrize(
    "flag,env_format,quantized",
    [(False, None, set()), (True, None, {4096, 8192}), (False, _MXFP8, set(_SM107_RUNGS))],
)
def test_each_rung_sizes_its_provider_for_its_own_format(
    env, monkeypatch, flag, env_format, quantized
) -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op_module

    if env_format is not None:
        monkeypatch.setenv(_ENV, env_format)
    query = Mock(return_value=(128, False))
    provider = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(op_module, "query_megamoe_kernel_properties", query, raising=False)
    monkeypatch.setattr(op_module, "get_megamoe_symm_provider", provider, raising=False)
    backend = _backend(flag)
    backend.apply_topk_in_fc1 = True
    backend.ep_size = 2
    backend._ep_pg = object()
    backend._alloc_symm_provider()
    assert query.call_count == provider.call_count == len(_SM107_RUNGS)
    for call in (*query.call_args_list, *provider.call_args_list):
        rung = call.kwargs["max_tokens_per_rank"]
        expected = _MXFP8 if rung in quantized else "bf16"
        assert call.kwargs["combine_format"] == expected
    # The in-kernel FC2 reduce (rung 32, top-k at FC1) follows its rung's format.
    reduced = {
        call.kwargs["max_tokens_per_rank"]
        for call in query.call_args_list
        if call.kwargs["in_kernel_fc2_reduce"]
    }
    assert reduced == {32} - quantized


def test_single_rank_staging_takes_its_rung_format(env, monkeypatch) -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op_module

    query = Mock(return_value=(128, False))
    monkeypatch.setattr(op_module, "query_megamoe_kernel_properties", query, raising=False)
    backend = _backend(True)
    for rung in (1024, 4096):
        backend._ensure_local_staging(
            top_k=2, hidden=512, device="cpu", output_dtype=torch.bfloat16, max_T=rung
        )
    assert [call.kwargs["combine_format"] for call in query.call_args_list] == ["bf16", _MXFP8]


@pytest.mark.parametrize(
    "flag,env_format,declined", [(False, None, False), (True, None, False), (False, _MXFP8, True)]
)
def test_fused_shared_expert_needs_the_bf16_combine(
    env, monkeypatch, flag, env_format, declined
) -> None:
    # Fusion is an SM100/SM103 feature, where the flag keeps BF16.
    env.sm = 100
    if env_format is not None:
        monkeypatch.setenv(_ENV, env_format)
    monkeypatch.setattr(MegaMoECuteDsl, "fused_shared_expert_count", classmethod(lambda cls, c: 1))
    warning_once = Mock()
    monkeypatch.setattr(backend_module.logger, "warning_once", warning_once)
    backend = _backend(flag)
    assert backend.num_fused_shared_expert == (0 if declined else 1)
    warned = any("needs the bf16 combine" in call.args[0] for call in warning_once.call_args_list)
    assert warned is declined
