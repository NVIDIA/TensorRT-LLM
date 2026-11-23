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
"""A real DFlash drafter dropped from a MuseSparkOpen pipeline stage, on one GPU.

The CPU tests (``test_musespark_open_pp_dflash_context.py``) run the decision and
the hooks over stand-in module trees. Here the drafter is the real
``DFlashForCausalLM``:

* the speculative tests' tiny DFlash 2 drafter, built on the device, dropped the
  way a non-last stage of a pipelined context server drops it: every byte it
  allocated is returned, and the stage's draft-weight load of its checkpoint
  reads nothing and allocates nothing; on the last stage the same load builds
  the whole drafter;
* the tiny drafter of a stage of a PP4 context server with the ``pp_partition``
  ``[15, 16, 16, 15]``: built with the stage's mapping and the released drafter's
  layer count its own stage hook cannot split it; built with
  ``pipeline_free_drafter_mapping`` it keeps every layer;
* with ``LLM_MODELS_ROOT`` holding the released drafter, the full-size drafter
  built as the model loader builds it (meta parameters, except the norm weights
  ``MetaInitMode`` leaves real) and dropped: the device tables it allocated at
  construction are returned too.
"""

import gc
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_DRAFTER_DIR_NAME = "Muse-Spark-Open-DFlash2"
_PARTITION = [15, 16, 16, 15]
_RELEASED_DRAFT_LAYERS = 5


def _pp4_stage_mapping(rank):
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=4, rank=rank, gpus_per_node=4, pp_size=4, pp_partition=_PARTITION)


def _semantics():
    """The speculative tests' tiny DFlash 2 drafter config and weights."""
    path = Path(__file__).parents[1] / "speculative" / "hw_agnostic" / "test_dflash2_semantics.py"
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allocated():
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def _stage(drafter, removed):
    """A decoder stage that holds ``drafter``, as far as ``load_draft_weights`` reads it."""
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenForCausalLM
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
    from tensorrt_llm.mapping import Mapping

    stage = MuseSparkOpenForCausalLM.__new__(MuseSparkOpenForCausalLM)
    nn.Module.__init__(stage)
    stage.model_config = SimpleNamespace(mapping=Mapping(world_size=4, rank=0, pp_size=4))
    stage.draft_model = drafter
    stage.spec_config = SimpleNamespace(
        spec_dec_mode=SpeculativeDecodingMode.DFLASH, uses_replacement_heads=False
    )
    vocab, hidden = drafter.config.vocab_size, drafter.config.hidden_size
    stage.model = nn.Module()
    stage.model.embed_tokens = nn.Embedding(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    stage.lm_head = nn.Linear(hidden, vocab, bias=False, device="cuda", dtype=torch.bfloat16)
    stage._pp_stage_drafter_removed = removed
    return stage


def test_a_dropped_tiny_drafter_returns_its_memory_and_skips_its_checkpoint():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    sem = _semantics()
    weights = {key: value.cuda() for key, value in sem._tiny_weights(True).items()}
    base = _allocated()
    config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(config, dflash_attention_backend="TRTLLM").to("cuda")
    assert _allocated() > base

    builtin._drop_pipeline_stage_drafter(drafter, 0, keep=[])
    assert _allocated() == base
    assert sum(param.numel() for param in drafter.parameters()) == 0
    with pytest.raises(RuntimeError, match="holds no drafter"):
        drafter()

    stage = _stage(drafter, removed=True)
    loaded_base = _allocated()
    stage.load_draft_weights(weights, weight_mapper=None)
    assert _allocated() == loaded_base
    assert getattr(drafter, "fc", None) is None
    assert sum(param.numel() for param in drafter.parameters()) == 0


def test_the_last_stage_loads_the_tiny_drafter_as_before():
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    sem = _semantics()
    weights = {key: value.cuda() for key, value in sem._tiny_weights(True).items()}
    config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(config, dflash_attention_backend="TRTLLM").to("cuda")
    stage = _stage(drafter, removed=False)
    stage.load_draft_weights(weights, weight_mapper=None)
    assert torch.equal(drafter.fc.weight, weights["fc.weight"])
    assert drafter.lm_head is stage.lm_head
    assert drafter.draft_model_full.model.embed_tokens is stage.model.embed_tokens


@pytest.mark.parametrize("rank", [0, 3])
def test_the_tiny_drafter_of_a_pp4_context_stage_is_built_whole(rank):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    sem = _semantics()
    stage = _pp4_stage_mapping(rank)
    # With the released drafter's layer count the generic stage hook gets past its
    # "enough layers for the pipeline" check and fails on the target's partition.
    released_depth = sem._tiny_config(True)
    released_depth.num_hidden_layers = _RELEASED_DRAFT_LAYERS
    released_depth.layer_types = ["sliding_attention"] * _RELEASED_DRAFT_LAYERS
    with pytest.raises(ValueError, match=f"does not match num_layers={_RELEASED_DRAFT_LAYERS}"):
        DFlashForCausalLM(
            ModelConfig(pretrained_config=released_depth, attn_backend="TRTLLM", mapping=stage),
            dflash_attention_backend="TRTLLM",
        )
    pretrained = sem._tiny_config(True)
    config = ModelConfig(
        pretrained_config=pretrained,
        attn_backend="TRTLLM",
        mapping=builtin.pipeline_free_drafter_mapping(stage),
    )
    drafter = DFlashForCausalLM(config, dflash_attention_backend="TRTLLM").to("cuda")
    layers = drafter.model.layers
    assert len(layers) == pretrained.num_hidden_layers
    for layer in layers:
        assert sum(param.numel() for param in layer.parameters()) > 0
        assert not getattr(layer, "_weights_removed", False)
    weights = {key: value.cuda() for key, value in sem._tiny_weights(True).items()}
    stage_model = _stage(drafter, removed=False)
    stage_model.load_draft_weights(weights, weight_mapper=None)
    assert torch.equal(drafter.fc.weight, weights["fc.weight"])


def _released_drafter_dir():
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        return None
    path = os.path.join(root, _DRAFTER_DIR_NAME)
    return path if os.path.isfile(os.path.join(path, "config.json")) else None


@pytest.mark.skipif(_released_drafter_dir() is None, reason="released drafter checkpoint not found")
@pytest.mark.parametrize("layout", ["one-gpu", "pp4-stage-0"])
def test_a_dropped_released_drafter_returns_what_its_construction_allocated(layout):
    """Meta parameters cost nothing; the RoPE tables built at construction are device memory.

    ``MetaInitMode`` moves only ``empty`` allocations to ``meta``; the norm weights,
    built with ``torch.ones``, stay real, as they do in the model loader. Every
    other parameter, which is all of the drafter's size, is on ``meta``.

    ``pp4-stage-0`` builds it as stage 0 of a PP4 context server does
    (``pipeline_free_drafter_mapping``).
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm._torch.utils import model_extra_attrs
    from tensorrt_llm.mapping import Mapping

    mapping = (
        Mapping()
        if layout == "one-gpu"
        else builtin.pipeline_free_drafter_mapping(_pp4_stage_mapping(0))
    )
    config = ModelConfig.from_pretrained(
        _released_drafter_dir(), trust_remote_code=True, attn_backend="TRTLLM", mapping=mapping
    )
    base = _allocated()
    with model_extra_attrs(config.extra_attrs), MetaInitMode():
        drafter = DFlashForCausalLM(config, dflash_attention_backend="TRTLLM")
    norm_weights = {
        f"{name}.{param_name}"
        for name, module in drafter.named_modules()
        if isinstance(module, RMSNorm)
        for param_name, _ in module.named_parameters(recurse=False)
    }
    real = sorted(name for name, param in drafter.named_parameters() if param.device.type != "meta")
    assert set(real) <= norm_weights, sorted(set(real) - norm_weights)
    meta_params = sum(
        param.numel() for param in drafter.parameters() if param.device.type == "meta"
    )
    assert meta_params * 2 / 2**30 > 8.5
    built = _allocated()
    builtin._drop_pipeline_stage_drafter(drafter, 0, keep=[])
    assert _allocated() == base, (base, built)
    assert sum(param.numel() for param in drafter.parameters()) == 0
