# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for MUSESPARK_OPEN_FP8_PROJECTIONS (block-FP8 qkv/o/pool/unpool on context workers).

* The load-time quantizer: power-of-two (UE8M0) 128x128 block scales equal to
  ``2**ceil(log2(max(amax, 1e-4) / 448))``, so ``post_load_weights``' resmooth
  (``resmooth_to_fp8_e8m0``, whose arithmetic ``_resmooth`` restates on the CPU) keeps
  ``q * scale`` exactly; quantize-then-shard equals shard-then-quantize on 128-aligned
  shards; and the input refusals.
* The switch's parsing and its boot refusals, on stand-in configs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
    E4M3_MAX,
    FP8_AMAX_FLOOR,
    FP8_WEIGHT_BLOCK,
    _quantize_block_fp8_ue8m0,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    FP8_PROJECTION_NAMES,
    _check_fp8_projection_linear,
    _check_fp8_projections_supported,
    _resolve_fp8_projections,
)

pytestmark = pytest.mark.cpu_only

_ENV = "MUSESPARK_OPEN_FP8_PROJECTIONS"
_CPU = torch.device("cpu")
B = FP8_WEIGHT_BLOCK


def _blocks(t: torch.Tensor) -> torch.Tensor:
    rows, cols = t.shape
    return t.reshape(rows // B, B, cols // B, B)


def _block_amax(t: torch.Tensor) -> torch.Tensor:
    return _blocks(t.abs()).amax(dim=(1, 3))


def _dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (_blocks(q.float()) * scale[:, None, :, None]).reshape(q.shape)


def _resmooth(q: torch.Tensor, scale: torch.Tensor):
    """``resmooth_to_fp8_e8m0``'s per-block arithmetic: rederive each block's UE8M0 scale
    from ``max(|q * scale|, 1e-4)`` and requantize to E4M3."""
    w = _dequant(q, scale)
    amax = _block_amax(w).clamp_min(FP8_AMAX_FLOOR)
    new_scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    new_q = (_blocks(w) / new_scale[:, None, :, None]).reshape(w.shape).to(torch.float8_e4m3fn)
    return new_q, new_scale


def _reference_scale(weight: torch.Tensor) -> torch.Tensor:
    amax = _block_amax(weight.double()).clamp_min(FP8_AMAX_FLOOR)
    return torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX))).float()


def _weight(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=gen)
    # One block per row of blocks at a different magnitude, so the scales span
    # several powers of two.
    for i in range(rows // B):
        w[i * B : (i + 1) * B] *= 2.0 ** (i - 2)
    return w.to(torch.bfloat16)


class TestQuantizer:
    def test_scales_are_the_ue8m0_ceiling(self):
        w = _weight(4 * B, 3 * B)
        q, scale = _quantize_block_fp8_ue8m0(w, device=_CPU)
        assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
        assert scale.dtype == torch.float32 and scale.shape == (4, 3)
        mantissa, _ = torch.frexp(scale)
        assert torch.all(mantissa == 0.5), "every block scale is a power of two"
        torch.testing.assert_close(scale, _reference_scale(w), rtol=0, atol=0)

    def test_largest_code_of_a_block_is_in_upper_octave(self):
        w = _weight(2 * B, 2 * B, seed=1)
        q, _ = _quantize_block_fp8_ue8m0(w, device=_CPU)
        qmax = _block_amax(q.float())
        assert torch.all(qmax >= E4M3_MAX / 2) and torch.all(qmax <= E4M3_MAX)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_resmooth_keeps_every_value(self, seed):
        w = _weight(4 * B, 2 * B, seed=seed)
        q, scale = _quantize_block_fp8_ue8m0(w, device=_CPU)
        new_q, new_scale = _resmooth(q, scale)
        # resmooth keeps the scale or halves it (doubling q), both exact.
        assert torch.all((new_scale == scale) | (new_scale == scale / 2))
        torch.testing.assert_close(_dequant(new_q, new_scale), _dequant(q, scale), rtol=0, atol=0)

    def test_power_of_two_amax_lands_on_e4m3_max(self):
        # amax / 448 exactly a power of two: frexp's mantissa is 0.5, so the
        # ceiling is that power itself and the block's largest code is 448.
        w = torch.full((B, B), 0.5, dtype=torch.bfloat16)
        w[0, 0] = E4M3_MAX * 2.0**-4
        q, scale = _quantize_block_fp8_ue8m0(w, device=_CPU)
        assert scale.item() == 2.0**-4
        assert q.float().abs().max().item() == E4M3_MAX

    def test_all_zero_block_takes_the_floor_scale(self):
        w = torch.zeros(B, 2 * B, dtype=torch.bfloat16)
        w[:, B:] = 1.0
        q, scale = _quantize_block_fp8_ue8m0(w, device=_CPU)
        floor = torch.exp2(torch.ceil(torch.log2(torch.tensor(FP8_AMAX_FLOOR / E4M3_MAX))))
        assert scale[0, 0].item() == floor.item()
        assert torch.all(q[:, :B].float() == 0)
        new_q, new_scale = _resmooth(q, scale)
        torch.testing.assert_close(_dequant(new_q, new_scale), _dequant(q, scale), rtol=0, atol=0)

    def test_quantize_then_shard_equals_shard_then_quantize(self):
        w = _weight(4 * B, 4 * B, seed=3)
        q, scale = _quantize_block_fp8_ue8m0(w, device=_CPU)
        for rank in range(2):
            rows = slice(rank * 2 * B, (rank + 1) * 2 * B)
            q_shard, scale_shard = _quantize_block_fp8_ue8m0(w[rows], device=_CPU)
            assert torch.equal(q[rows].view(torch.uint8), q_shard.view(torch.uint8))
            assert torch.equal(scale[rank * 2 : (rank + 1) * 2], scale_shard)

    @pytest.mark.parametrize(
        "weight, match",
        [
            (torch.zeros(B, B, dtype=torch.float16), "come from BF16"),
            (torch.zeros(B, B, B, dtype=torch.bfloat16), "2-D"),
            (torch.zeros(B + 64, B, dtype=torch.bfloat16), "multiples of 128"),
            (torch.zeros(B, B - 64, dtype=torch.bfloat16), "multiples of 128"),
        ],
    )
    def test_refuses_what_it_cannot_quantize(self, weight, match):
        with pytest.raises(ValueError, match=match):
            _quantize_block_fp8_ue8m0(weight, device=_CPU)

    def test_refuses_a_non_finite_weight(self):
        w = torch.ones(B, B, dtype=torch.bfloat16)
        w[3, 5] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            _quantize_block_fp8_ue8m0(w, device=_CPU)


class TestResolve:
    @pytest.mark.parametrize("value", [None, "", "0", "off", "False", " no "])
    def test_off(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(_ENV, raising=False)
        else:
            monkeypatch.setenv(_ENV, value)
        assert _resolve_fp8_projections() == ()

    @pytest.mark.parametrize("value", ["1", "all", "ON", "true", "yes"])
    def test_all(self, monkeypatch, value):
        monkeypatch.setenv(_ENV, value)
        assert _resolve_fp8_projections() == FP8_PROJECTION_NAMES == ("qkv", "o", "pool", "unpool")

    @pytest.mark.parametrize(
        "value, names",
        [
            ("qkv", ("qkv",)),
            ("pool+qkv", ("qkv", "pool")),
            ("unpool + o", ("o", "unpool")),
            ("qkv+qkv", ("qkv",)),
        ],
    )
    def test_subset_in_canonical_order(self, monkeypatch, value, names):
        monkeypatch.setenv(_ENV, value)
        assert _resolve_fp8_projections() == names

    @pytest.mark.parametrize("value", ["qkv+gate", "qkv+", "+", "dense"])
    def test_unknown_or_empty_token(self, monkeypatch, value):
        monkeypatch.setenv(_ENV, value)
        with pytest.raises(ValueError, match="unknown projection"):
            _resolve_fp8_projections()


def _model_config(**overrides):
    config = dict(
        pretrained_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        modeling_v2_target=None,
        use_cute_dsl_blockscaling_mm=False,
        lora_config=None,
        max_num_tokens=8192,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


class TestSupported:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT", raising=False)

    @pytest.mark.parametrize("sm_version", [100, 103])
    def test_served_config(self, sm_version):
        _check_fp8_projections_supported(_model_config(), FP8_PROJECTION_NAMES, sm_version)

    @pytest.mark.parametrize(
        "overrides, sm_version, match",
        [
            (dict(modeling_v2_target=object), 103, "context-worker switch"),
            ({}, 90, "SM100-family"),
            ({}, 120, "SM100-family"),
            (
                dict(pretrained_config=SimpleNamespace(torch_dtype=torch.float16)),
                103,
                "BF16 activations",
            ),
            (dict(use_cute_dsl_blockscaling_mm=True), 103, "use_cute_dsl_blockscaling_mm"),
            (dict(lora_config=object()), 103, "LoRA"),
            (dict(max_num_tokens=8193), 103, "max_num_tokens <= 8192"),
        ],
    )
    def test_refusals(self, overrides, sm_version, match):
        with pytest.raises(ValueError, match=match):
            _check_fp8_projections_supported(
                _model_config(**overrides), FP8_PROJECTION_NAMES, sm_version
            )

    def test_ll_qkv_layout_refuses_qkv_only(self, monkeypatch):
        monkeypatch.setenv("MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT", "1")
        with pytest.raises(ValueError, match="MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT"):
            _check_fp8_projections_supported(_model_config(), ("qkv",), 103)
        _check_fp8_projections_supported(_model_config(), ("o", "pool", "unpool"), 103)


def _linear(**overrides):
    config = dict(
        quant_config=None,
        use_cute_dsl_blockscaling_mm=False,
        disable_deep_gemm=False,
        in_features=8192,
        out_features=4096,
        tp_sharding=None,
        fused_weight_shard_indices_mapping=None,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


class TestLinearCheck:
    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            dict(tp_sharding=(1024, 2048)),
            dict(tp_sharding={"q": (0, 2048), "k": (2048, 2304)}),
            dict(fused_weight_shard_indices_mapping={"q": (0, 2048), "k": (2048, 128)}),
        ],
    )
    def test_aligned(self, overrides):
        _check_fp8_projection_linear(0, "qkv", _linear(**overrides))

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(out_features=4096 + 64),
            dict(tp_sharding=(1024, 1088)),
            dict(fused_weight_shard_indices_mapping={"k": (2048, 64)}),
        ],
    )
    def test_misaligned_shard(self, overrides):
        with pytest.raises(ValueError, match="off the 128-row blocks"):
            _check_fp8_projection_linear(3, "qkv", _linear(**overrides))

    @pytest.mark.parametrize("flag", ["use_cute_dsl_blockscaling_mm", "disable_deep_gemm"])
    def test_deep_gemm_opt_out(self, flag):
        with pytest.raises(ValueError, match="opts out of DeepGEMM"):
            _check_fp8_projection_linear(0, "o", _linear(**{flag: True}))

    def test_already_quantized(self):
        quant_config = SimpleNamespace(
            quant_algo="NVFP4",
            layer_quant_mode=SimpleNamespace(has_any_quant=lambda exclude_kv_cache: True),
        )
        with pytest.raises(ValueError, match="already quantized"):
            _check_fp8_projection_linear(0, "pool", _linear(quant_config=quant_config))
