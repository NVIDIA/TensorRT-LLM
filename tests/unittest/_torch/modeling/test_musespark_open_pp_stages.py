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
"""Pipeline stages of the MuseSparkOpen decoder, on CPU.

The stage hook (``MuseSparkOpenModel.__pp_init__``), the stage-local forward
and the loader's stage filter run here for real, around stub decoder layers
that keep the one property of the real layer the stage boundary depends on: a
layer hands its successor either nothing (the successor norms the residual
itself) or a precomputed attention input from the fused boundary, and the two
differ by a rounding. Stages run one after another in one process, with the
point-to-point send / receive replaced by a queue, so a pipelined forward can be
compared with the one-stage forward bit for bit.

The stage hook, forward and loader run here directly, and the construction-time
refusals have their own tests below.

The 4-GPU comparison of a synthetic checkpoint served at PP4 and PP1 is in
``tests/unittest/_torch/multi_gpu_modeling/test_musespark_open_pp.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.configs.musespark_open import FULL_ATTENTION, MuseSparkOpenConfig
from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.checkpoints.hf import musespark_open_weight_mapper as mapper
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenForCausalLM,
    MuseSparkOpenModel,
    pipeline_parallel_refusal,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm.mapping import Mapping

HIDDEN, VOCAB, LAYERS = 16, 64, 8
DTYPE = torch.bfloat16


# --------------------------------------------------------------------------
# Stage partition on the released 62-layer layout
# --------------------------------------------------------------------------
def _stage_ranges(num_layers, pp_size, partition):
    ranges = []
    for rank in range(pp_size):
        mapping = Mapping(
            world_size=pp_size,
            rank=rank,
            gpus_per_node=pp_size,
            pp_size=pp_size,
            pp_partition=partition,
        )
        ranges.append(mapping.pp_layers(num_layers))
    return ranges


@pytest.mark.parametrize("partition", [[15, 16, 17, 14], None], ids=["15-16-17-14", "default"])
def test_every_stage_holds_four_of_the_sixteen_full_attention_layers(partition):
    """Equal full-attention KV per stage: the PP4 KV pool is then MIN-synced at no loss.

    The 16 full-attention (NoPE) layers sit at l = 1 mod 4; the sliding-window
    layers' KV is bounded by the 2048-key window. A split that put five full
    layers on one stage would make that stage's per-token KV 25 % larger than
    the others', and the stage pools are sized to the smallest.
    """
    config = MuseSparkOpenConfig(num_hidden_layers=62)
    ranges = _stage_ranges(62, 4, partition)
    assert [layer for stage in ranges for layer in stage] == list(range(62))
    for stage in ranges:
        assert stage == list(range(stage[0], stage[-1] + 1))
        full = [layer for layer in stage if config.layer_types[layer] == FULL_ATTENTION]
        assert len(full) == 4, (stage, full)
    # The checkpoint's BF16-expert layers (0 and 61) land on the edge stages.
    assert 0 in ranges[0] and 61 in ranges[-1]
    if partition is not None:
        assert [len(stage) for stage in ranges] == partition


# --------------------------------------------------------------------------
# Stub decoder: real stage hook and forward, stub layers
# --------------------------------------------------------------------------
class _StubNorm(nn.Module):
    """RMSNorm stand-in that runs on CPU in BF16 (no flashinfer) and can be skipped."""

    def __init__(self, hidden, use_gemma=True, has_weights=True):
        super().__init__()
        self.use_gemma = use_gemma
        self.variance_epsilon = 1e-6
        if has_weights:
            self.weight = nn.Parameter(torch.randn(hidden, dtype=DTYPE) * 0.1, requires_grad=False)
        else:
            self.weight = None

    def _gain(self):
        if self.weight is None:
            return 1.0
        return self.weight.float() + 1.0 if self.use_gemma else self.weight.float()

    def forward(self, hidden_states):
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (x * self._gain()).to(DTYPE)

    def fused(self, residual):
        """The fused boundary's path: norm the FP32 residual, round once."""
        return self.forward(residual)

    def skip_forward(self, hidden_states):
        return hidden_states


class _StubEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB, HIDDEN, dtype=DTYPE), requires_grad=False)

    def forward(self, input_ids):
        return self.weight[input_ids]

    def skip_forward(self, input_ids):
        return torch.empty(input_ids.shape + (HIDDEN,), dtype=DTYPE)


class _StubLayer(DecoderLayer):
    """One residual update per layer, with the MuseSparkOpen boundary contract.

    The unfused path norms ``hidden_states.to(bf16)``; the fused path norms the
    FP32 residual. They differ by a BF16 rounding, so a stage boundary that
    dropped the precomputed input and re-normed on the next stage would not
    reproduce the one-stage forward.
    """

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.pre_qkv_proj_norm = _StubNorm(HIDDEN)
        self.proj = nn.Parameter(
            torch.randn(HIDDEN, HIDDEN, dtype=DTYPE) * 0.3, requires_grad=False
        )
        self.calls = []

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "precomputed": precomputed_attn_input is not None,
                "next_layer_norm": next_layer_norm,
                "kwargs": sorted(kwargs),
            }
        )
        if precomputed_attn_input is not None:
            attn_input = precomputed_attn_input
        else:
            attn_input = self.pre_qkv_proj_norm(hidden_states.to(DTYPE))
        hidden_states = hidden_states + torch.tanh(attn_input.float() @ self.proj.float())
        if next_layer_norm is None:
            return hidden_states
        return hidden_states, next_layer_norm.fused(hidden_states)


def _model_config(mapping, num_layers=LAYERS):
    pretrained = SimpleNamespace(
        hidden_size=HIDDEN, torch_dtype=DTYPE, num_hidden_layers=num_layers
    )
    return SimpleNamespace(mapping=mapping, pretrained_config=pretrained, spec_config=None)


def _stub_model(mapping, num_layers=LAYERS):
    """A MuseSparkOpenModel whose stage hook and forward are the real ones."""
    torch.manual_seed(1234)
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    DecoderModel.__init__(model, _model_config(mapping, num_layers))
    model.embed_tokens = _StubEmbedding()
    model.embed_norm = _StubNorm(HIDDEN, use_gemma=False, has_weights=False)
    model.m_emb = 1.0
    model.layers = nn.ModuleList(_StubLayer(index) for index in range(num_layers))
    model.norm = _StubNorm(HIDDEN, use_gemma=False)
    model._prefill_sp_enabled = False
    model.dtype = DTYPE
    model.__pp_init__()
    return model


def _mapping(rank, pp_size=4, partition=None):
    return Mapping(
        world_size=pp_size,
        rank=rank,
        gpus_per_node=max(pp_size, 1),
        pp_size=pp_size,
        pp_partition=partition,
    )


class _Wire:
    """The point-to-point boundary as a FIFO; records what crossed it."""

    def __init__(self):
        self.queue = []
        self.sent = []

    def send(self, tensors):
        self.sent.append([(t.shape, t.dtype) for t in tensors])
        self.queue.append([t.clone() for t in tensors])

    def recv(self, tensors):
        payload = self.queue.pop(0)
        assert [(t.shape, t.dtype) for t in tensors] == [(t.shape, t.dtype) for t in payload]
        for buffer, value in zip(tensors, payload):
            buffer.copy_(value)


def _run_pipeline(monkeypatch, partition, input_ids, pp_size=4):
    wire = _Wire()
    monkeypatch.setattr(modeling_musespark_open, "pp_send_tensors", wire.send)
    monkeypatch.setattr(modeling_musespark_open, "pp_recv_tensors", wire.recv)
    stages = [_stub_model(_mapping(rank, pp_size, partition)) for rank in range(pp_size)]
    outputs = [
        stage(attn_metadata=None, input_ids=input_ids, position_ids=None, marker=True)
        for stage in stages
    ]
    assert not wire.queue
    return stages, outputs, wire


@pytest.mark.parametrize("partition", [None, [1, 2, 3, 2], [3, 1, 1, 3]])
def test_pipelined_forward_equals_the_one_stage_forward_bit_for_bit(monkeypatch, partition):
    input_ids = torch.tensor([3, 17, 5, 63, 0, 9, 41])
    reference = _stub_model(Mapping())(attn_metadata=None, input_ids=input_ids, position_ids=None)
    stages, outputs, _ = _run_pipeline(monkeypatch, partition, input_ids)
    assert outputs[-1].dtype == DTYPE
    assert torch.equal(outputs[-1], reference)
    # Every layer ran exactly once across the stages, on its owner, with the
    # caller's keyword arguments passed through.
    for layer_idx in range(LAYERS):
        calls = [len(stage.layers[layer_idx].calls) for stage in stages]
        owner = next(i for i, stage in enumerate(stages) if layer_idx in range(*stage._pp_stage))
        assert calls == [1 if i == owner else 0 for i in range(len(stages))]
        assert stages[owner].layers[layer_idx].calls[0]["kwargs"] == ["marker"]


def test_the_stage_boundary_carries_the_fp32_residual_and_the_bf16_attention_input(
    monkeypatch,
):
    input_ids = torch.arange(11)
    stages, outputs, wire = _run_pipeline(monkeypatch, [1, 2, 3, 2], input_ids)
    expected = [(torch.Size([11, HIDDEN]), torch.float32), (torch.Size([11, HIDDEN]), DTYPE)]
    assert wire.sent == [expected] * 3
    for stage, output in zip(stages[:-1], outputs[:-1]):
        # A non-last stage returns its FP32 residual; the LM head that reads it
        # is skipped there.
        assert output.dtype == torch.float32 and output.shape == (11, HIDDEN)


def test_norm_fusion_runs_through_the_boundary_and_ends_at_the_last_layer(monkeypatch):
    stages, _, _ = _run_pipeline(monkeypatch, [1, 2, 3, 2], torch.arange(5))
    for rank, stage in enumerate(stages):
        first, end = stage._pp_stage
        last_call = stage.layers[end - 1].calls[0]
        if end < LAYERS:
            # The stage's last layer fuses the next stage's first norm, whose
            # weight this stage keeps while the rest of that layer is dropped.
            boundary = stage.layers[end]
            assert last_call["next_layer_norm"] is boundary.pre_qkv_proj_norm
            assert boundary.pre_qkv_proj_norm.weight is not None
            assert "weight" in boundary.pre_qkv_proj_norm._parameters
            assert boundary._weights_removed and "proj" not in boundary._parameters
        else:
            assert last_call["next_layer_norm"] is None
        # A non-first stage's first layer consumes the received input; stage 0's
        # first layer norms the embedding itself.
        assert stage.layers[first].calls[0]["precomputed"] == (rank > 0)
        for layer_idx in range(LAYERS):
            if layer_idx < first or layer_idx > end:
                assert "weight" not in stage.layers[layer_idx].pre_qkv_proj_norm._parameters


def test_the_first_stage_owns_the_embedding_and_the_last_the_final_norm():
    partition = [15, 16, 17, 14]
    for rank in range(4):
        stage = _stub_model(_mapping(rank, 4, partition), num_layers=62)
        assert ("weight" in stage.embed_tokens._parameters) == (rank == 0)
        assert ("weight" in stage.norm._parameters) == (rank == 3)
        assert stage._pp_stage == (sum(partition[:rank]), sum(partition[: rank + 1]))


def test_the_last_stage_owns_the_lm_head():
    """The causal-LM hook drops the LM head off non-last stages before the model's."""
    for rank in range(4):
        causal_lm = MuseSparkOpenForCausalLM.__new__(MuseSparkOpenForCausalLM)
        nn.Module.__init__(causal_lm)
        mapping = _mapping(rank)
        causal_lm.model_config = _model_config(mapping)
        torch.manual_seed(0)
        causal_lm.model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
        DecoderModel.__init__(causal_lm.model, causal_lm.model_config)
        causal_lm.model.embed_tokens = _StubEmbedding()
        causal_lm.model.layers = nn.ModuleList(_StubLayer(i) for i in range(LAYERS))
        causal_lm.model.norm = _StubNorm(HIDDEN, use_gemma=False)
        causal_lm.lm_head = _StubEmbedding()
        causal_lm.prologue, causal_lm.epilogue = [], [causal_lm.lm_head]
        causal_lm.__pp_init__()
        assert ("weight" in causal_lm.lm_head._parameters) == (rank == 3)
        assert causal_lm.model._pp_stage is not None


def test_one_stage_keeps_every_module():
    model = _stub_model(Mapping())
    assert model._pp_stage is None
    assert all("proj" in layer._parameters for layer in model.layers)
    assert "weight" in model.embed_tokens._parameters and "weight" in model.norm._parameters


def test_a_stage_without_layers_is_refused():
    with pytest.raises(ValueError, match="non-empty contiguous slice"):
        _stub_model(_mapping(1, 4, [4, 0, 2, 2]))


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def _refusal_config(pp_size=4, cp_size=1, attention_dp=False, spec=None, layers=62):
    mapping = SimpleNamespace(pp_size=pp_size, cp_size=cp_size, enable_attention_dp=attention_dp)
    return SimpleNamespace(
        mapping=mapping,
        spec_config=spec,
        pretrained_config=SimpleNamespace(num_hidden_layers=layers),
    )


def test_pipeline_parallel_refusal():
    assert pipeline_parallel_refusal(_refusal_config(pp_size=1, spec=object())) is None
    assert pipeline_parallel_refusal(_refusal_config()) is None
    assert "speculative" in pipeline_parallel_refusal(_refusal_config(spec=object()))
    assert "context parallelism" in pipeline_parallel_refusal(_refusal_config(cp_size=2))
    assert "attention data" in pipeline_parallel_refusal(_refusal_config(attention_dp=True))
    assert "cannot fill" in pipeline_parallel_refusal(_refusal_config(layers=3))


def _decoder_config(spec=None):
    return SimpleNamespace(
        quant_config=None,
        pretrained_config=SimpleNamespace(num_hidden_layers=LAYERS),
        mapping=_mapping(0),
        spec_config=spec,
    )


def test_the_decoder_admits_pipeline_stages():
    """pp_size > 1 needs no extra switch: construction goes straight to the
    stage refusals."""
    with pytest.raises(ValueError, match="speculative decoding is not supported"):
        modeling_musespark_open.MuseSparkOpenForCausalLM(_decoder_config(spec=object()))


def test_the_omni_wrapper_refuses_the_towers_under_pipeline_parallelism(monkeypatch):
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.models import modeling_musespark_open_omni as omni

    built = []
    monkeypatch.setattr(omni, "MuseSparkOpenVisionEncoder", lambda cfg: built.append("tower"))

    class _Decoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg.pretrained_config
            built.append("decoder")

    monkeypatch.setattr(omni, "MuseSparkOpenForCausalLM", _Decoder)

    def model_config(disable):
        return SimpleNamespace(
            pretrained_config=PretrainedConfig(),
            disable_mm_encoder=disable,
            mapping=_mapping(0),
        )

    with pytest.raises(ValueError, match="text only"):
        omni.MuseSparkOpenOmniForConditionalGeneration(model_config(False))
    assert built == []
    wrapper = omni.MuseSparkOpenOmniForConditionalGeneration(model_config(True))
    assert wrapper.mm_encoder is None and built == ["decoder"]


# --------------------------------------------------------------------------
# Loader stage filter
# --------------------------------------------------------------------------
def _checkpoint_keys(num_layers):
    keys = mapper.missing_in_range_keys([], num_layers, 256, [0, num_layers - 1])
    # One out-of-range layer and one perception key, as a truncated checkpoint carries.
    return keys + [f"model.layers.{num_layers}.add_attn.gate", "vision_model.ln_pre.weight"]


def _load(monkeypatch, stage_model, num_layers=LAYERS):
    loaded = {"layers": [], "linear": [], "params": []}

    def load_layer(model, layer, layer_idx, weights, accounting, open_format=False):
        loaded["layers"].append(layer_idx)
        prefix = f"model.layers.{layer_idx}."
        accounting.consumed.update(key for key in weights if key.startswith(prefix))

    monkeypatch.setattr(mapper, "load_layer", load_layer)
    monkeypatch.setattr(mapper, "_load_linear", lambda module, w: loaded["linear"].append(module))
    monkeypatch.setattr(mapper, "_copy_param", lambda param, v: loaded["params"].append(param))
    weights = {key: torch.zeros(1) for key in _checkpoint_keys(num_layers)}
    lm_head = nn.Linear(1, 1)
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=num_layers), model=stage_model, lm_head=lm_head
    )
    summary = mapper.load_musespark_open_weights(model, weights)
    return loaded, summary, lm_head


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_the_loader_reads_only_the_stage_tensors(monkeypatch, rank):
    partition = [1, 2, 3, 2]
    stage = _stub_model(_mapping(rank, 4, partition))
    first, end = stage._pp_stage
    loaded, summary, lm_head = _load(monkeypatch, stage)

    assert loaded["layers"] == list(range(first, end))
    assert (stage.embed_tokens in loaded["linear"]) == (rank == 0)
    assert (lm_head in loaded["linear"]) == (rank == 3)
    # The last stage loads the final norm; every other stage loads the next
    # stage's first pre-QKV norm instead, for the fused boundary.
    if rank == 3:
        assert len(loaded["params"]) == 1 and loaded["params"][0] is stage.norm.weight
    else:
        assert len(loaded["params"]) == 1
        assert loaded["params"][0] is stage.layers[end].pre_qkv_proj_norm.weight
    # Every text key is either this stage's or accounted to another stage; the
    # loader's zero-unaccounted check passed.
    per_layer = len(mapper.missing_in_range_keys([], 1, 256, [0])) - 3
    other_layers = LAYERS - (end - first)
    expected_other = sum(
        per_layer
        + (0 if layer in (0, LAYERS - 1) else 6)  # the NVFP4 scales of quantized layers
        - (1 if layer == end else 0)  # the boundary norm this stage loads
        for layer in range(LAYERS)
        if not first <= layer < end
    )
    expected_other += (rank != 0) + 2 * (rank != 3)
    assert other_layers > 0
    assert summary["ignored_other_stage"] == expected_other
    assert summary["ignored_out_of_range"] == 1


def test_one_stage_loads_everything_and_keeps_the_summary_unchanged(monkeypatch):
    stage = _stub_model(Mapping())
    loaded, summary, lm_head = _load(monkeypatch, stage)
    assert loaded["layers"] == list(range(LAYERS))
    assert stage.embed_tokens in loaded["linear"] and lm_head in loaded["linear"]
    assert "ignored_other_stage" not in summary
    assert set(summary) == {
        "consumed",
        "ignored_out_of_range",
        "ignored_perception",
        "ignored_other",
        "ignored_vision_unused",
        "ignored_duplicate_vision",
        "ignored_stale_fused_qkv",
        "ignored_audio",
    }
