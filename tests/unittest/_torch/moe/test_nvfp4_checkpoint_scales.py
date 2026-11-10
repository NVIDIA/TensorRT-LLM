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
"""An NVFP4 MoE refuses checkpoint experts whose scales would not load.

The NVFP4 loaders read every scale with ``weights.get`` and zero-pad a block
scale that is narrower than its destination, so without a check an expert
stored in another format -- an MXFP4 expert has one E8M0 scale per 32 inputs
and no global scales -- would load without an error and compute garbage. These
tests build such experts on the CPU and check that both loading paths refuse
them while an NVFP4 expert still loads.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4CutlassFusedMoEMethod

pytestmark = pytest.mark.cpu_only

HIDDEN = 64
INTERMEDIATE = 32


class _RecordingMethod(NVFP4CutlassFusedMoEMethod):
    """Records the block scales it is handed instead of laying them out."""

    def __init__(self):
        super().__init__()
        self.loaded = []

    def load_expert_w3_w1_weight_scale_nvfp4(
        self, module, w1_weight_scale, w3_weight_scale, dst_w3_w1_weight_scale, expert_idx=-1
    ):
        self.loaded.append(("w3_w1", w1_weight_scale, w3_weight_scale))

    def load_expert_w2_weight_scale_nvfp4(self, module, w2_weight_scale, dst_w2_weight_scale):
        self.loaded.append(("w2", w2_weight_scale))


class _MxBlockMethod(_RecordingMethod):
    """A method whose checkpoint has 32-input blocks and no global scales."""

    checkpoint_has_global_scales = False


def _module(scaling_vector_size=16):
    return SimpleNamespace(
        layer_idx=61,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        scaling_vector_size=scaling_vector_size,
        is_gated_activation=True,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        _add_raw_shared_weights_for_unmap=lambda tensors: None,
    )


def _expert(expert_id=0, block=16, global_scales=True, drop=()):
    """One checkpoint expert: packed weights, block scales of `block` inputs."""
    weights = {}
    for proj, rows, cols in (
        ("w1", INTERMEDIATE, HIDDEN),
        ("w3", INTERMEDIATE, HIDDEN),
        ("w2", HIDDEN, INTERMEDIATE),
    ):
        prefix = f"{expert_id}.{proj}"
        weights[f"{prefix}.weight"] = torch.zeros(rows, cols // 2, dtype=torch.uint8)
        weights[f"{prefix}.weight_scale"] = torch.ones(rows, cols // block, dtype=torch.uint8)
        if global_scales:
            weights[f"{prefix}.weight_scale_2"] = torch.tensor(1.0)
            weights[f"{prefix}.input_scale"] = torch.tensor(1.0)
    for key in drop:
        del weights[key]
    return weights


def _load_block_scales(method, module, weights):
    tmp_weight_scale_2 = {}
    method.load_fp4_weight_block_scales(
        module,
        weights,
        [0],
        torch.empty(1, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.uint8),
        torch.empty(1, HIDDEN, INTERMEDIATE // 16, dtype=torch.uint8),
        tmp_weight_scale_2,
    )
    return tmp_weight_scale_2


def test_nvfp4_expert_loads():
    method = _RecordingMethod()
    tmp_weight_scale_2 = _load_block_scales(method, _module(), _expert())
    assert [entry[0] for entry in method.loaded] == ["w3_w1", "w2"]
    assert sorted(tmp_weight_scale_2[0]) == ["w1", "w2", "w3"]


def test_mxfp4_expert_is_refused():
    # The MTP block of an NVFP4 DeepSeek-V4 export: 32-input E8M0 block
    # scales and no global scales, configured as an NVFP4 layer.
    method = _RecordingMethod()
    with pytest.raises(ValueError, match=r"MoE layer 61: 0\.w1\.weight_scale .* need 4 columns"):
        _load_block_scales(method, _module(), _expert(block=32, global_scales=False))
    assert method.loaded == []


@pytest.mark.parametrize("key", ["0.w1.weight_scale", "0.w3.weight_scale", "0.w2.weight_scale"])
def test_missing_block_scale_is_refused(key):
    method = _RecordingMethod()
    with pytest.raises(ValueError, match=f"has no {key}; an NVFP4 expert needs one block scale"):
        _load_block_scales(method, _module(), _expert(drop=(key,)))
    assert method.loaded == []


@pytest.mark.parametrize("key", ["0.w1.weight_scale_2", "0.w2.weight_scale_2"])
def test_missing_global_weight_scale_is_refused(key):
    method = _RecordingMethod()
    with pytest.raises(ValueError, match=f"has no {key}; an NVFP4 expert needs its global"):
        _load_block_scales(method, _module(), _expert(drop=(key,)))


def test_input_scale_is_not_required():
    # Activations quantized dynamically need no checkpoint input_scale.
    method = _RecordingMethod()
    _load_block_scales(
        method,
        _module(),
        _expert(drop=("0.w1.input_scale", "0.w3.input_scale", "0.w2.input_scale")),
    )
    assert len(method.loaded) == 2


def test_method_with_wider_blocks_and_no_global_scales_loads_its_format():
    # The check follows the method's own block width and global-scale
    # contract, so a method declared for 32-input blocks without global scales
    # still loads such an expert.
    method = _MxBlockMethod()
    _load_block_scales(
        method, _module(scaling_vector_size=32), _expert(block=32, global_scales=False)
    )
    assert [entry[0] for entry in method.loaded] == ["w3_w1", "w2"]


def test_expert_absent_from_a_partial_reload_is_not_judged():
    method = _RecordingMethod()
    weights = {k: v for k, v in _expert().items() if not k.endswith(".weight")}
    weights.pop("0.w2.weight_scale")
    _load_block_scales(method, _module(), weights)


def test_streamed_expert_with_narrow_block_scale_is_refused():
    method = _RecordingMethod()
    module = _module()
    module.expert_size_per_partition = 1
    module.initial_local_expert_ids = [0]
    module._streamed_expert_slots = set()
    expert = _expert(block=32, global_scales=False)
    kwargs = {
        f"{proj}_{kind}": expert.get(f"0.{proj}.{kind}", torch.tensor(1.0))
        for proj in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    with pytest.raises(ValueError, match=r"w1_weight_scale has shape \(32, 2\)"):
        method.load_streaming_nvfp4_expert(module, global_expert_id=0, local_slot_id=0, **kwargs)
    assert module._streamed_expert_slots == set()
