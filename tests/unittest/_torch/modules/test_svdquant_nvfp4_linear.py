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

"""NVFP4 Linear with a ModelOpt SVDQuant low-rank residual.

The exported weight is ``W - B @ A``; the layer must return
``nvfp4_gemm(x * pre_quant_scale) + ((x * pre_quant_scale) @ A^T) @ B^T``.
Each test loads the same packed weights twice, once with and once without the
residual tensors, and checks that the two outputs differ by exactly the
high-precision residual, so no FP4 reference implementation is needed.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="NVFP4 GEMM needs SM100 or newer",
)

K, RANK, M = 64, 4, 8
FP8_E4M3_MAX, E2M1_MAX = 448.0, 6.0


def _modelopt_shard(out_features: int, seed: int) -> dict[str, torch.Tensor]:
    """One ModelOpt NVFP4_SVD shard: packed weight, scales, pre_quant_scale, LoRA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(out_features, K, device="cuda", dtype=torch.bfloat16, generator=g)
    qweight, weight_scale, weight_scale_2 = quantize_nvfp4(weight)
    return {
        "weight": qweight,
        "weight_scale": weight_scale,
        "weight_scale_2": weight_scale_2,
        # amax / (448 * 6), the ModelOpt convention for the static activation scale
        "input_scale": torch.tensor(4.0 / (FP8_E4M3_MAX * E2M1_MAX), device="cuda"),
        "pre_quant_scale": torch.rand(K, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5,
        "svdquant_lora_a": torch.randn(RANK, K, device="cuda", dtype=torch.bfloat16, generator=g)
        * 0.1,
        "svdquant_lora_b": torch.randn(
            out_features, RANK, device="cuda", dtype=torch.bfloat16, generator=g
        )
        * 0.1,
    }


def _without_lora(shard: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in shard.items() if not k.startswith("svdquant_lora")}


_SHARD_KEYS = {
    WeightMode.VANILLA: None,
    WeightMode.FUSED_GATE_UP_LINEAR: ("gate", "up"),
    WeightMode.FUSED_QKV_LINEAR: ("q", "k", "v"),
}


def _make_linear(out_features: list[int], weight_mode: WeightMode) -> Linear:
    """A Linear in ``weight_mode`` whose fused output rows are ``out_features`` per shard."""
    mapping = None
    if _SHARD_KEYS[weight_mode] is not None:
        offsets = [sum(out_features[:i]) for i in range(len(out_features))]
        mapping = {
            key: (offset, size)
            for key, offset, size in zip(_SHARD_KEYS[weight_mode], offsets, out_features)
        }
    linear = Linear(
        K,
        sum(out_features),
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, pre_quant_scale=True),
        weights_loading_config=WeightsLoadingConfig(weight_mode=weight_mode),
        fused_weight_shard_indices_mapping=mapping,
        reduce_output=False,
    )
    return linear.cuda()


def _residual(x: torch.Tensor, shards: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    x_scaled = x * shards[0]["pre_quant_scale"]
    return torch.cat(
        [F.linear(F.linear(x_scaled, s["svdquant_lora_a"]), s["svdquant_lora_b"]) for s in shards],
        dim=-1,
    )


@pytest.mark.parametrize(
    ("weight_mode", "out_features"),
    [
        (WeightMode.VANILLA, [48]),
        (WeightMode.FUSED_GATE_UP_LINEAR, [32, 32]),
        (WeightMode.FUSED_QKV_LINEAR, [32, 16, 16]),
    ],
)
def test_svdquant_residual_adds_to_nvfp4_output(weight_mode, out_features) -> None:
    shards = [_modelopt_shard(n, seed) for seed, n in enumerate(out_features)]
    for shard in shards[1:]:
        shard["pre_quant_scale"] = shards[0]["pre_quant_scale"]
        shard["input_scale"] = shards[0]["input_scale"]
        shard["weight_scale_2"] = shards[0]["weight_scale_2"]
    total = sum(out_features)

    with_lora = _make_linear(out_features, weight_mode)
    with_lora.load_weights(shards)
    with_lora.post_load_weights()
    assert with_lora.svdquant_lora_a.shape == (RANK * len(shards), K)
    assert with_lora.svdquant_lora_b.shape == (total, RANK * len(shards))

    without_lora = _make_linear(out_features, weight_mode)
    without_lora.load_weights([_without_lora(s) for s in shards])
    without_lora.post_load_weights()
    assert without_lora.svdquant_lora_a is None

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    expected = without_lora(x) + _residual(x, shards)
    torch.testing.assert_close(with_lora(x), expected, atol=2e-2, rtol=2e-2)


def test_svdquant_residual_requires_every_fused_shard() -> None:
    shards = [_modelopt_shard(32, 0), _without_lora(_modelopt_shard(32, 1))]
    shards[1]["pre_quant_scale"] = shards[0]["pre_quant_scale"]
    linear = _make_linear([32, 32], WeightMode.FUSED_GATE_UP_LINEAR)
    with pytest.raises(ValueError, match="every fused shard"):
        linear.load_weights(shards)
