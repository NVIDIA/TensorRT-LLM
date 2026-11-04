# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU engine admission and routing contracts; no model allocation."""

from enum import Enum
from types import SimpleNamespace as NS

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.beam import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (
    ModelingV2BeamFullModelNvfp4Sm103Tp4 as Target,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "ids,end_id,lora,valid",
    [
        ([0, 9], None, None, True),
        ([9], -1, None, True),
        ([0], 9, None, True),
        ([-1], None, None, False),
        ([10], None, None, False),
        ([0], -2, None, False),
        ([0], 10, None, False),
        ([0], None, 0, False),
    ],
)
def test_request_rejected_before_batch_execution(ids, end_id, lora, valid):
    target = NS(model=NS(vocab_size=10))
    request = NS(
        py_end_id=end_id,
        lora_task_id=lora,
        check_token_id_range=lambda n: all(0 <= token < n for token in ids),
    )
    if valid:
        Target.validate_request(target, request)
    else:
        with pytest.raises(ValueError):
            Target.validate_request(target, request)


def test_default_weight_loader_is_refused_explicitly():
    with pytest.raises(ValueError, match="BeamNvfp4CheckpointLoader"):
        Target.load_weights(NS(), {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("cuda_graph_config", object()),
        ("disable_overlap_scheduler", False),
        ("load_format", "dummy"),
        ("max_beam_width", 2),
        ("attn_backend", "FLASHINFER"),
        ("lora_config", object()),
        ("kv_connector_config", object()),
        ("cache_transceiver_config", object()),
        ("enable_return_routed_experts", True),
        ("max_num_tokens", 0),
        ("max_num_tokens", 8193),
    ],
)
def test_unsupported_engine_option_is_refused(field, value):
    assert routing.engine_refusal(NS(**{field: value})) is not None


def test_engine_reports_all_refusals_and_accepts_auto_enum():
    load_format = Enum("LoadFormat", "AUTO DUMMY", start=0)
    assert routing.engine_refusal(NS(load_format=load_format.AUTO, max_num_tokens=8192)) is None
    refusal = routing.engine_refusal(NS(load_format=load_format.DUMMY, max_beam_width=2))
    assert "load_format" in refusal and "max_beam_width" in refusal


def test_route_requires_loader_before_reading_checkpoint(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        layout as nvfp4_layout,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
        weights,
    )

    model_config = NS(
        pretrained_config=NS(
            num_hidden_layers=52,
            hidden_size=5120,
            num_experts=256,
            quantization_config={"quant_algo": "NVFP4"},
        ),
        mapping=NS(
            world_size=4,
            tp_size=4,
            pp_size=1,
            moe_tp_size=4,
            moe_ep_size=1,
            enable_attention_dp=False,
        ),
        spec_config=None,
    )
    monkeypatch.setattr(routing, "engine_kv_cache_quant_algo", lambda *args: None)
    reads = []
    monkeypatch.setattr(nvfp4_layout, "check_checkpoint", lambda path: reads.append(path))
    for bad_loader in (None, object()):
        with pytest.raises(NoTarget, match="checkpoint_loader=BeamNvfp4CheckpointLoader"):
            routing.route(model_config, NS(checkpoint_loader=bad_loader), (10, 3))
    assert reads == []
    loader_instance = object.__new__(weights.BeamNvfp4CheckpointLoader)
    assert routing.route(model_config, NS(checkpoint_loader=loader_instance), (10, 3)) is Target
    assert routing.route(model_config, NS(checkpoint_format="beam_nvfp4"), (10, 3)) is Target
    assert reads == [None, None]
    # A per-instance loader overrides the named format in ModelLoader.
    with pytest.raises(NoTarget, match="checkpoint_loader=BeamNvfp4CheckpointLoader"):
        routing.route(
            model_config, NS(checkpoint_format="beam_nvfp4", checkpoint_loader=object()), (10, 3)
        )
    assert reads == [None, None]
