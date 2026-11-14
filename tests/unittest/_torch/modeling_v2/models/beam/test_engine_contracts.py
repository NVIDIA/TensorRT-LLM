# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission and routing contracts for the eager TP4 target; no model allocation."""

import builtins
from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.beam import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    layout as nvfp4_layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (
    MAX_NUM_TOKENS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (
    ModelingV2BeamFullModelNvfp4Sm103Tp4 as Target,
)
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM

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


@pytest.fixture
def route_setup(monkeypatch: pytest.MonkeyPatch) -> tuple[NS, NS, list]:
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
        quant_config=NS(quant_algo="NVFP4", kv_cache_quant_algo=None),
        spec_config=None,
    )
    llm_args = NS(
        checkpoint_format="beam_nvfp4",
        kv_cache_config=NS(dtype="auto"),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        max_num_tokens=MAX_NUM_TOKENS,
    )
    reads = []
    monkeypatch.setattr(nvfp4_layout, "check_checkpoint", lambda path: reads.append(path))
    return model_config, llm_args, reads


def test_route_requires_loader_before_reading_checkpoint(route_setup) -> None:
    model_config, llm_args, reads = route_setup
    llm_args.checkpoint_format = None
    for bad_loader in (None, object()):
        llm_args.checkpoint_loader = bad_loader
        with pytest.raises(NoTarget, match="checkpoint_loader=BeamNvfp4CheckpointLoader"):
            routing.route(model_config, llm_args, (10, 3))
    assert reads == []
    llm_args.checkpoint_loader = object.__new__(weights.BeamNvfp4CheckpointLoader)
    assert routing.route(model_config, llm_args, (10, 3)) is Target
    llm_args.checkpoint_loader = None
    llm_args.checkpoint_format = "beam_nvfp4"
    assert routing.route(model_config, llm_args, (10, 3)) is Target
    assert reads == [None, None]
    # A per-instance loader overrides the named format in ModelLoader.
    llm_args.checkpoint_loader = object()
    with pytest.raises(NoTarget, match="checkpoint_loader=BeamNvfp4CheckpointLoader"):
        routing.route(model_config, llm_args, (10, 3))
    assert reads == [None, None]


@pytest.mark.parametrize("pretrained_quant", [None, {}, NS()], ids=["none", "dict", "object"])
@pytest.mark.parametrize(
    "fallback",
    ["NVFP4", Enum("QuantAlgo", {"NVFP4": "NVFP4"}).NVFP4],
    ids=["string", "enum"],
)
def test_route_selects_nvfp4_from_model_quant_config(
    route_setup, pretrained_quant, fallback
) -> None:
    model_config, llm_args, reads = route_setup
    model_config.pretrained_config.quantization_config = pretrained_quant
    model_config.quant_config.quant_algo = fallback
    assert routing.route(model_config, llm_args, (10, 3)) is Target
    assert reads == [None]


@pytest.mark.parametrize("pretrained_quant", ["NVFP4", "FP8"])
def test_pretrained_quantization_takes_precedence_over_fallback(
    route_setup, pretrained_quant
) -> None:
    model_config, llm_args, reads = route_setup
    model_config.pretrained_config.quantization_config = {"quant_algo": pretrained_quant}
    model_config.quant_config.quant_algo = "FP8" if pretrained_quant == "NVFP4" else "NVFP4"
    if pretrained_quant == "NVFP4":
        assert routing.route(model_config, llm_args, (10, 3)) is Target
        assert reads == [None]
    else:
        with pytest.raises(NoTarget, match="quantization FP8"):
            routing.route(model_config, llm_args, (10, 3))
        assert reads == []


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("cuda_graph_config", object(), "CUDA graphs are not supported by this eager Beam target"),
        ("torch_compile_config", object(), "torch_compile_config must be None"),
        ("prefill_cuda_graph_backend", "piecewise", "prefill_cuda_graph_backend must be disabled"),
        ("prefill_cuda_graph_backend", "breakable", "prefill_cuda_graph_backend must be disabled"),
        (
            "prefill_cuda_graph_backend",
            Enum("PrefillBackend", {"BREAKABLE": "breakable"}).BREAKABLE,
            "prefill_cuda_graph_backend must be disabled",
        ),
        (
            "disable_overlap_scheduler",
            False,
            "the overlap scheduler is not supported by this eager Beam target",
        ),
        ("load_format", "dummy", "load_format must be auto"),
        ("max_beam_width", 2, "max_beam_width must be 1"),
        ("max_num_tokens", None, "max_num_tokens must be explicitly set"),
        ("max_num_tokens", MAX_NUM_TOKENS + 1, f"max_num_tokens {MAX_NUM_TOKENS + 1} is outside"),
    ],
)
def test_eager_target_route_refuses_unsupported_engine_options(
    route_setup, field, value, message
) -> None:
    model_config, llm_args, reads = route_setup
    # Graph/overlap guards must also run when the token budget is unspecified.
    llm_args.max_num_tokens = None
    setattr(llm_args, field, value)
    assert message in routing.engine_refusal(llm_args)
    with pytest.raises(NoTarget, match=message):
        routing.route(model_config, llm_args, (10, 3))
    assert reads == []


@pytest.mark.parametrize(
    "backend", ["disabled", Enum("PrefillBackend", {"DISABLED": "disabled"}).DISABLED]
)
def test_eager_route_accepts_disabled_prefill_graphs(route_setup, backend) -> None:
    config, args, reads = route_setup
    args.prefill_cuda_graph_backend = backend
    args.torch_compile_config = None
    assert routing.route(config, args, (10, 3)) is Target
    assert reads == [None]


@pytest.mark.parametrize("builtin", [None, object])
def test_auto_refusal_explains_missing_fallback(route_setup, monkeypatch, builtin) -> None:
    config, args, reads = route_setup
    config.pretrained_config.architectures = ["BeamForCausalLM"]
    args.modeling_v2 = "auto"
    args.cuda_graph_config = object()
    args.disable_overlap_scheduler = False
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    monkeypatch.setattr(AutoModelForCausalLM, "_resolve_class", lambda config: builtin)
    log = Mock()
    monkeypatch.setattr(_router_index, "logger", log)
    assert _router_index.modeling_v2_resolve(config, args) is None
    if builtin is None:
        message = log.warning.call_args.args[0]
        assert "no built-in implementation" in message
        assert "CUDA graphs" in message and "overlap scheduler" in message
        log.info.assert_not_called()
    else:
        log.warning.assert_not_called()
        assert "using the built-in implementation" in log.info.call_args.args[0]
    assert reads == []


def test_wrong_hardware_refusal_does_not_import_target_model(route_setup, monkeypatch) -> None:
    config, args, reads = route_setup
    original_import = builtins.__import__

    def guarded_import(name, *positional, **kwargs):
        if "full_model_nvfp4__sm_103__tp4.model" in name:
            pytest.fail(f"Target model imported before hardware admission: {name}")
        return original_import(name, *positional, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(NoTarget, match="compute capability"):
        routing.route(config, args, (9, 0))
    assert reads == []


@pytest.mark.parametrize("diagnostics", [False, True])
def test_checkpoint_digests_follow_diagnostic_mode(diagnostics, tmp_path) -> None:
    checkpoint = str(tmp_path)
    record = {"loaded": True}
    load = Mock(return_value=record)
    target = NS(
        checkpoint_dir=checkpoint,
        model=NS(diagnostics=diagnostics, load=load),
        load_record=None,
    )
    assert Target.load_checkpoint(target, checkpoint) is record
    load.assert_called_once_with(checkpoint, record_digests=diagnostics)
    assert target.load_record is record


@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("chunked,reuse", [(False, False), (True, False), (False, True)])
def test_route_checks_paged_context_page_size(route_setup, page_size, chunked, reuse) -> None:
    config, args, reads = route_setup
    args.enable_chunked_prefill = chunked
    args.kv_cache_config = NS(dtype="auto", tokens_per_block=page_size, enable_block_reuse=reuse)
    if page_size != 32 and (chunked or reuse):
        with pytest.raises(NoTarget, match="kv_cache_config.tokens_per_block must be 32"):
            routing.route(config, args, (10, 3))
        assert reads == []
    else:
        assert routing.route(config, args, (10, 3)) is Target
        assert reads == [None]
