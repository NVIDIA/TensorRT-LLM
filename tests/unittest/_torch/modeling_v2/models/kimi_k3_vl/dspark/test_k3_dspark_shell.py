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
"""``KimiK3SpecForCausalLM``, the family's speculative-decoding base, at construction: it builds the
drafter's config with the target's execution layout, the drafter and the worker with the target's
constants, binds the worker to both models, appends both to the epilogue, declares the decode
prologue as a runtime hook, and refuses every speculative mode but one-engine standalone DSpark. The
drafter checkpoint config and the drafter itself are stand-ins here (the drafter is covered by its
parity tests); the target is a two-layer stand-in."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark import shell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.drafter import (
    K3DSparkPath,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.worker import (
    K3DSparkWorker,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

HIDDEN, VOCAB = 64, 256


class _Mode:
    def __init__(self, dspark=True, one_engine=True):
        self._dspark, self._one_engine = dspark, one_engine

    def use_one_engine(self):
        return self._one_engine

    def is_dspark(self):
        return self._dspark

    def is_dflash(self):
        return False

    def is_parallel_draft(self):
        return True

    def __repr__(self):
        return f"_Mode(dspark={self._dspark})"


def _spec_config(**mode):
    return SimpleNamespace(
        spec_dec_mode=_Mode(**mode),
        speculative_model="/drafter",
        draft_is_embedded_in_target=False,
        attention_backend="AUTO",
        max_draft_len=7,
        moe_backend=None,
        _use_shared_kv_cache=True,
        enable_penalty=False,
    )


class _StubDrafter(nn.Module):
    dflash_attention_backend = "CUTEDSL"
    shift_label = True
    has_markov_head = True
    _helix_mapping = None

    def __init__(self, draft_config, spec_config, path, helix_mapping, context_kv_only):
        super().__init__()
        self.args = dict(
            draft_config=draft_config,
            spec_config=spec_config,
            path=path,
            helix_mapping=helix_mapping,
            context_kv_only=context_kv_only,
        )
        self.logits_processor = None


class _Target(nn.Module):
    def forward(self, input_ids=None, **kwargs):
        return torch.zeros(input_ids.shape[0], HIDDEN)


def _target_config(spec_config, kv_algo=QuantAlgo.NVFP4):
    cfg = PretrainedConfig(hidden_size=HIDDEN, vocab_size=VOCAB, torch_dtype=torch.bfloat16)
    config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        quant_config=QuantConfig(kv_cache_quant_algo=kv_algo),
        spec_config=spec_config,
        max_num_tokens=128,
        max_seq_len=4096,
    )
    config.extra_attrs["probe"] = 1
    return config


@pytest.fixture
def stubs(monkeypatch):
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        cfg = PretrainedConfig(hidden_size=HIDDEN, vocab_size=VOCAB, torch_dtype=torch.bfloat16)
        return ModelConfig(pretrained_config=cfg, mapping=kwargs["mapping"])

    def build(draft_config, spec_config, *, path, helix_mapping=None, context_kv_only=False):
        return _StubDrafter(draft_config, spec_config, path, helix_mapping, context_kv_only)

    monkeypatch.setattr(ModelConfig, "from_pretrained", staticmethod(from_pretrained))
    monkeypatch.setattr(shell, "build_k3_dspark_drafter", build)
    return calls


def _prologue(*args, **kwargs):
    return None


class _Ctx(shell.KimiK3SpecForCausalLM):
    dspark_path = K3DSparkPath()
    dspark_context_kv_only = True


class _Helix(shell.KimiK3SpecForCausalLM):
    dspark_path = K3DSparkPath(spec_tail_fusion=True, tail_v2=True)
    spec_decode_prologue = staticmethod(_prologue)


@pytest.mark.parametrize("cls", [_Ctx, _Helix])
def test_builds_the_drafter_and_the_worker(stubs, cls):
    spec_config = _spec_config()
    config = _target_config(spec_config)
    model = cls(_Target(), config, hidden_size=HIDDEN, vocab_size=VOCAB)

    ((path, kwargs),) = stubs
    assert path == "/drafter" and kwargs["spec_config"] is None
    assert kwargs["mapping"].to_dict() == config.mapping.mapping_for_full_rows().to_dict()
    assert kwargs["max_seq_len"] == 4096 and kwargs["max_num_tokens"] == 128
    draft_config = model.draft_config
    assert draft_config.extra_attrs is config.extra_attrs
    assert draft_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4

    drafter, worker = model.draft_model, model.spec_worker
    assert isinstance(drafter, _StubDrafter) and isinstance(worker, K3DSparkWorker)
    assert drafter.args["draft_config"] is draft_config
    assert drafter.args["path"] is cls.dspark_path and worker._path is cls.dspark_path
    assert drafter.args["context_kv_only"] is cls.dspark_context_kv_only
    assert drafter.args["helix_mapping"] is None
    assert drafter.logits_processor is model.logits_processor
    assert model.epilogue[-2:] == [drafter, worker]
    assert worker.plain_logits_processor_type is LogitsProcessor
    assert worker._draft_model_ref() is drafter and worker._target_model_ref() is model
    assert model.use_separate_draft_kv_cache is False
    assert model.layer_idx == -1

    hooks = model.runtime_hooks
    if cls.spec_decode_prologue is None:
        assert hooks is None
    else:
        assert hooks.decode_prologue is _prologue


def test_the_target_helix_mapping_reaches_the_drafter(stubs):
    spec_config = _spec_config()
    config = _target_config(spec_config)
    helix = Mapping(world_size=1, rank=0)
    object.__setattr__(config, "_helix_mapping_with_cp", helix)
    model = _Helix(_Target(), config, hidden_size=HIDDEN, vocab_size=VOCAB)
    assert model.draft_model.args["helix_mapping"] is helix


def test_a_draft_kv_override_reaches_the_draft_config(stubs):
    config = _target_config(_spec_config())
    config.extra_attrs["draft_kv_cache_quant_algo_override"] = QuantAlgo.FP8.value
    model = _Ctx(_Target(), config, hidden_size=HIDDEN, vocab_size=VOCAB)
    assert model.draft_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


@pytest.mark.parametrize("mode", [dict(dspark=False), dict(one_engine=False)])
def test_other_speculative_modes_are_refused(stubs, mode):
    with pytest.raises(ValueError, match="one-engine DSpark"):
        _Ctx(_Target(), _target_config(_spec_config(**mode)), hidden_size=HIDDEN, vocab_size=VOCAB)
    assert stubs == []


def test_without_speculation_nothing_is_built(stubs):
    model = _Ctx(_Target(), _target_config(None), hidden_size=HIDDEN, vocab_size=VOCAB)
    assert model.draft_model is None and model.spec_worker is None and model.draft_config is None
    assert stubs == []
