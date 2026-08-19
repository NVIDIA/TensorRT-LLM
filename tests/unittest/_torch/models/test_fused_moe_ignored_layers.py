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
"""A producer's per-expert ignore rule must de-quantize the fused MoE.

The fused module has no per-expert child, so a rule naming an expert's weights
(``...experts.<i>.up_proj``) only matches synthesized candidates. The pass must
probe every expert index (not just expert 0), strip a ``.backend`` wrapper, and
warn rather than silently round a partial list to all-or-nothing.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.modules.fused_moe import MoE
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS = 8
BASE = "model.layers.3.mlp.experts"


class _FakeMoE(MoE):
    """Minimal MoE stand-in exposing only what the exclusion pass reads."""

    def __init__(self, num_experts: int) -> None:
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
        self._weights_created = True


def _attach(root: nn.Module, path: str, leaf: nn.Module) -> None:
    """Register ``leaf`` under ``root`` at a dotted path, so ``named_modules``
    yields ``path`` (plain nn.Modules stand in for the intermediate wrappers)."""
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = parent._modules.get(part)
        if child is None:
            child = nn.Module()
            parent.add_module(part, child)
        parent = child
    parent.add_module(parts[-1], leaf)


def _apply_exclusions(moe_path: str, exclude_modules: list[str]) -> _FakeMoE:
    moe = _FakeMoE(NUM_EXPERTS)
    model = nn.Module()
    _attach(model, moe_path, moe)
    model.model_config = SimpleNamespace(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES, exclude_modules=list(exclude_modules)
        )
    )
    modeling_utils.DecoderModelForCausalLM.apply_quant_config_exclude_modules(model)
    return moe


def test_exact_nonzero_expert_index_dequantizes() -> None:
    """Regression: an exact rule for a nonzero expert must de-quantize the
    module. The old code only probed expert 0, so such rules never matched."""
    last = NUM_EXPERTS - 1
    moe = _apply_exclusions(BASE, [f"{BASE}.{last}.up_proj", "lm_head"])

    assert moe.quant_config.quant_algo is None
    assert moe._weights_created is False


def test_backend_wrapped_moe_is_dequantized() -> None:
    """ConfigurableMoE wraps the weight-owning backend; the ``.backend`` suffix
    must be stripped so the producer's expert rule reaches the backend."""
    moe = _apply_exclusions(f"{BASE}.backend", [f"{BASE}.5.down_proj", "lm_head"])

    assert moe.quant_config.quant_algo is None
    assert moe._weights_created is False


def test_partial_expert_list_warns_and_dequantizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fused module shares one quant config, so a partial list cannot be
    honoured per-expert: warn, and treat the whole module as excluded."""
    warnings: list[str] = []
    monkeypatch.setattr(modeling_utils.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    exclude = [f"{BASE}.{e}.up_proj" for e in range(3)] + ["lm_head"]
    moe = _apply_exclusions(BASE, exclude)

    assert moe.quant_config.quant_algo is None
    assert any("fused module" in w for w in warnings)
