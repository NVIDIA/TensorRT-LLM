# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm.mapping import Mapping


def test_timing_metric_accumulates_and_records_failures(monkeypatch) -> None:
    perf_counter_values = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(modeling_utils.time, "perf_counter", lambda: next(perf_counter_values))
    metrics = {}

    with modeling_utils.timing_metric("load_seconds", metrics):
        pass

    with pytest.raises(RuntimeError, match="load failed"):
        with modeling_utils.timing_metric("load_seconds", metrics):
            raise RuntimeError("load failed")

    assert metrics["load_seconds"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.layers.0.mlp", "model.layers.0.mlp"),
        ("model._orig_mod.layers.0.mlp", "model.layers.0.mlp"),
        ("_orig_mod.model.layers.0.mlp", "model.layers.0.mlp"),
        ("model._orig_mod", "model"),
        ("_orig_mod", ""),
        # Only whole components are the wrapper, never a substring.
        ("model.layers._orig_mod_x.mlp", "model.layers._orig_mod_x.mlp"),
    ],
)
def test_strip_torch_compile_wrapper(name, expected) -> None:
    assert modeling_utils.strip_torch_compile_wrapper(name) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the module walk pins the CUDA device")
def test_load_weights_impl_v2_matches_through_torch_compile_wrapper(monkeypatch) -> None:
    """A refit into a compiled model loads every module, wrapper or not.

    ``torch.compile`` inserts ``_orig_mod`` into the paths below the compiled
    scope; checkpoint keys never carry it. Before the loader stripped it, an
    ``allow_partial_loading=True`` reload matched nothing under the wrapper
    and silently kept the old weights.
    """

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config = ModelConfig(
                pretrained_config=PretrainedConfig(tie_word_embeddings=False), mapping=Mapping()
            )
            self.config = self.model_config.pretrained_config
            self.model = _Inner()
            self.lm_head = nn.Linear(4, 2, bias=False)

    stub = _Stub()
    # Compile the inner scope the way PyTorchModelEngine does for a
    # DecoderModelForCausalLM; the wrapper is only walked, never run.
    stub.model = torch.compile(stub.model, backend="eager")
    assert any("_orig_mod" in n for n, _ in stub.named_parameters())

    weights = {
        "model.layers.0.weight": torch.full((4, 4), 2.0),
        "model.layers.0.bias": torch.full((4,), 3.0),
        "lm_head.weight": torch.full((2, 4), 5.0),
    }
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")
    mapper = HfWeightMapper()
    mapper.init_model_and_config(stub, stub.model_config)
    modeling_utils._load_weights_impl_v2(stub, weights, mapper, allow_partial_loading=True)

    inner = stub.model._orig_mod
    torch.testing.assert_close(inner.layers[0].weight.detach(), weights["model.layers.0.weight"])
    torch.testing.assert_close(inner.layers[0].bias.detach(), weights["model.layers.0.bias"])
    torch.testing.assert_close(stub.lm_head.weight.detach(), weights["lm_head.weight"])
