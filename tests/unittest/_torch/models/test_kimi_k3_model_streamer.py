# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the Kimi K3 ``plan_model_streamer_load()`` methods.

`run_model_streamer()` and the streaming helpers in ``models/checkpoints/model_streamer/model_loading.py``
have their own tests in ``tests/unittest/_torch/executor/``. These tests build
the Kimi K3 plans and check what they name and how they load it, running them
through `run_model_streamer()` with a fake streamer where the load matters.
"""

import itertools
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.model_streamer import model_loading as model_streamer
from tensorrt_llm._torch.models.checkpoints.model_streamer.model_loading import ModelStreamerPlan


def _expert_layer(key: str) -> int:
    return int(key.split(".layers.")[1].split(".")[0])


def _multimodal_wrapper(llm, mm_encoder, *, disable_mm_encoder=False):
    """Return an object that runs the Kimi K3 VL loading methods on fake parts."""
    from tensorrt_llm._torch.models.modeling_kimi_k3_vl import KimiK3ForConditionalGeneration

    class Wrapper:
        _LANG_PREFIX = KimiK3ForConditionalGeneration._LANG_PREFIX
        _ensure_mm_encoder = KimiK3ForConditionalGeneration._ensure_mm_encoder
        load_weights = KimiK3ForConditionalGeneration.load_weights
        plan_model_streamer_load = KimiK3ForConditionalGeneration.plan_model_streamer_load

    model = Wrapper()
    model.llm = llm
    model.mm_encoder = mm_encoder
    model.model_config = SimpleNamespace(disable_mm_encoder=disable_mm_encoder)
    return model


def test_multimodal_wrapper_adds_vision_group(monkeypatch):
    from tensorrt_llm._torch.models import modeling_kimi_k25

    text_group = ({"language_model.weight"}, lambda tensors: None)

    class TextModel:
        def plan_model_streamer_load(self, checkpoint_keys):
            self.planned_keys = checkpoint_keys
            return ModelStreamerPlan(groups=[text_group])

    class VisionModel:
        def load_weights(self, weights):
            self.loaded_weights = weights

    monkeypatch.setattr(modeling_kimi_k25, "_has_meta_tensors", lambda model: False)
    model = _multimodal_wrapper(TextModel(), VisionModel())
    checkpoint_keys = {"language_model.weight", "vision_tower.weight", "mm_projector.weight"}

    plan = model.plan_model_streamer_load(checkpoint_keys)

    assert model.llm.planned_keys == checkpoint_keys
    # One plan, so the text and vision tensors stream in one session.
    assert plan.groups[0] is text_group
    [(vision_keys, load_vision)] = plan.groups[1:]
    assert vision_keys == {"vision_tower.weight", "mm_projector.weight"}
    assert load_vision == model.mm_encoder.load_weights


def test_multimodal_wrapper_text_only_skips_vision():
    text_plan = ModelStreamerPlan(groups=[({"language_model.model.weight"}, lambda tensors: None)])

    class TextModel:
        def plan_model_streamer_load(self, checkpoint_keys):
            return text_plan

        def load_weights(self, weights):
            self.loaded_weights = weights

    text = TextModel()
    model = _multimodal_wrapper(text, None, disable_mm_encoder=True)
    plan = model.plan_model_streamer_load({"language_model.model.weight", "vision_tower.weight"})
    assert plan is text_plan
    assert len(plan.groups) == 1
    assert model.mm_encoder is None

    model.load_weights(
        {
            "language_model.model.weight": torch.tensor([1]),
            "vision_tower.weight": torch.tensor([2]),
        }
    )
    assert set(text.loaded_weights) == {"model.weight"}
    assert model.mm_encoder is None


@pytest.mark.parametrize("needs_layer_finalize", [True, False])
def test_streamer_expert_loading_matches_regular_loader(monkeypatch, needs_layer_finalize):
    pytest.importorskip("fla")
    from tensorrt_llm._torch.models.modeling_kimi_k3_loader import _EXPERT_LAYERS_PER_BATCH
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    prefix = "language_model."
    kinds = ("weight", "weight_scale")
    weight_names = ("w1", "w2", "w3")
    num_layers = 2 * _EXPERT_LAYERS_PER_BATCH + 1
    # Expert 1 is owned by another rank; local slot 0 holds expert 2.
    local_expert_ids = (2, 0)
    generator = torch.Generator().manual_seed(0)
    checkpoint = {
        f"{prefix}model.layers.{layer}.block_sparse_moe.experts.{expert}.{weight}.{kind}": (
            torch.randn(2, generator=generator)
        )
        for layer in range(num_layers)
        for expert in range(3)
        for weight in weight_names
        for kind in kinds
    }

    def build():
        """Return a model harness with fake expert backends that record every load."""
        events = []
        loads = {}

        def load_expert(backend, base, expert_idx, slot, get_tensor):
            loads[(base, expert_idx, slot)] = {
                f"{weight}.{kind}": get_tensor(f"{base}.{expert_idx}.{weight}.{kind}").clone()
                for weight in weight_names
                for kind in kinds
            }
            backend.loaded_slots.add(slot)

        spec = SimpleNamespace(
            kinds=kinds,
            loader=load_expert,
            loaded_slots_attr="loaded_slots",
            needs_layer_finalize=needs_layer_finalize,
        )

        class Backend:
            def __init__(self, layer):
                self.layer = layer
                self.loaded_slots = set()
                self.expert_size_per_partition = len(local_expert_ids)
                self.quant_method = SimpleNamespace(
                    prepare_streaming_expert_load=lambda backend: events.append(
                        ("prepare", backend.layer)
                    )
                )

            def process_weights_after_loading(self):
                events.append(("finalize", self.layer))

        class Harness:
            plan_model_streamer_load = KimiLinearForCausalLM.plan_model_streamer_load
            checkpoint_name_plan = KimiLinearForCausalLM.checkpoint_name_plan
            _load_expert_slices = KimiLinearForCausalLM._load_expert_slices

            def __init__(self):
                self.model = SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            is_moe=True,
                            is_kda=True,
                            block_sparse_moe=SimpleNamespace(
                                local_expert_ids=list(local_expert_ids),
                                expert_ckpt_spec=spec,
                                routed_experts=SimpleNamespace(backend=Backend(layer)),
                            ),
                        )
                        for layer in range(num_layers)
                    ]
                )
                self.model_config = SimpleNamespace(
                    mapping=SimpleNamespace(rank=0, world_size=1),
                    pretrained_config=SimpleNamespace(_name_or_path=None),
                )

            def parameters(self):
                return iter([torch.empty(0)])

            def _trunk_parameters(self):
                return {}

            def _attention_fp8_linears(self):
                return []

            def _finalize_weight_load(self, num_params, num_moe_layers):
                pass

        return Harness(), events, loads

    regular, regular_events, regular_loads = build()
    _, _, expert_jobs = regular.checkpoint_name_plan(prefix)
    regular._load_expert_slices(dict(checkpoint), expert_jobs)

    streamed, streamed_events, streamed_loads = build()
    streamed_batches = []

    def stream(weight_map, batches, consume, *, distributed):
        assert not distributed
        for batch in batches:
            streamed_batches.append(set(batch))
            # Interleave layers, so every layer in a batch starts before any finishes.
            by_layer = defaultdict(list)
            for name in sorted(batch):
                by_layer[_expert_layer(name)].append(name)
            for names in itertools.zip_longest(*by_layer.values()):
                for name in names:
                    if name is not None:
                        consume(name, checkpoint[name])
        return {}

    monkeypatch.setattr(model_streamer, "stream_batched_tensors", stream)
    plan = streamed.plan_model_streamer_load(set(checkpoint))
    model_streamer.run_model_streamer(
        plan, dict.fromkeys(checkpoint, "model.safetensors"), rank=0, world_size=1
    )

    assert streamed_loads.keys() == regular_loads.keys()
    assert len(regular_loads) == num_layers * len(local_expert_ids)
    for job, tensors in regular_loads.items():
        assert streamed_loads[job].keys() == tensors.keys()
        for name, value in tensors.items():
            assert torch.equal(streamed_loads[job][name], value), (job, name)

    expected_events = (
        sorted((event, layer) for layer in range(num_layers) for event in ("finalize", "prepare"))
        if needs_layer_finalize
        else []
    )
    assert sorted(regular_events) == expected_events
    assert sorted(streamed_events) == expected_events
    assert [len({_expert_layer(name) for name in batch}) for batch in streamed_batches] == [
        _EXPERT_LAYERS_PER_BATCH,
        _EXPERT_LAYERS_PER_BATCH,
        1,
    ]
    prepared, peak = set(), 0
    for event, layer in streamed_events:
        if event == "prepare":
            prepared.add(layer)
        else:
            prepared.discard(layer)
        peak = max(peak, len(prepared))
    assert peak == (_EXPERT_LAYERS_PER_BATCH if needs_layer_finalize else 0)
