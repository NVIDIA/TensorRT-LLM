# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Beam NVFP4 target's channel parser and family routing (generated inputs)."""

import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget, routing_module
from tensorrt_llm.llmapi.reasoning_parser import BeamHarmonyTextParser, ReasoningParserFactory

pytestmark = pytest.mark.cpu_only

#: The parser's registry key and the channel delimiters it consumes.
PARSER_KEY = "beam_harmony_text"
TAGS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)

#: (reply text after the generation prefix, expected content, expected reasoning_content)
REPLIES = {
    "analysis_then_final": (
        "<|channel|>analysis<|message|>Add the two numbers.<|end|><|start|>assistant"
        "<|channel|>final<|message|>2 + 3 = 5.<|return|>",
        "2 + 3 = 5.",
        "Add the two numbers.",
    ),
    "final_only": ("<|channel|>final<|message|>Paris.<|return|>", "Paris.", ""),
    "no_headers": ("Plain text without channels.", "Plain text without channels.", ""),
    "stopped_in_analysis": ("<|channel|>analysis<|message|>Still thinking", "", "Still thinking"),
    "stopped_in_final": (
        "<|channel|>analysis<|message|>a<|end|><|start|>assistant<|channel|>final<|message|>Par",
        "Par",
        "a",
    ),
    "multiline_bodies": (
        "<|channel|>analysis<|message|>Line 1\n\nLine 2 <b>x</b><|end|><|start|>assistant"
        "<|channel|>final<|message|>\\(12 - 7 = 5\\)\n\nANSWER: 5<|return|>",
        "\\(12 - 7 = 5\\)\n\nANSWER: 5",
        "Line 1\n\nLine 2 <b>x</b>",
    ),
    "literal_angle_brackets": (
        "<|channel|>final<|message|>a <| b and <|x|><|return|>",
        "a <| b and <|x|>",
        "",
    ),
    "tool_channel_ignored": (
        "<|channel|>commentary to=functions.f <|constrain|>json<|message|>{}<|call|>",
        "",
        "",
    ),
    "terminated_by_call": ("<|channel|>final<|message|>done<|call|>ignored", "done", ""),
}


def _stream(text, pieces):
    parser = BeamHarmonyTextParser()
    content = reasoning = ""
    for piece in pieces:
        delta = parser.parse_delta(piece)
        content += delta.content
        reasoning += delta.reasoning_content
    tail = parser.finish()
    return content + tail.content, reasoning + tail.reasoning_content


@pytest.mark.parametrize("name", sorted(REPLIES))
def test_whole_reply(name):
    text, content, reasoning = REPLIES[name]
    result = BeamHarmonyTextParser().parse(text)
    assert (result.content, result.reasoning_content) == (content, reasoning)
    assert not any(tag in result.content or tag in result.reasoning_content for tag in TAGS)


@pytest.mark.parametrize("name", sorted(REPLIES))
def test_every_fragmentation_agrees_with_the_whole_reply(name):
    text, content, reasoning = REPLIES[name]
    assert _stream(text, list(text)) == (content, reasoning)
    for cut in range(len(text) + 1):
        assert _stream(text, [text[:cut], text[cut:]]) == (content, reasoning), cut
    for size in (2, 3, 5, 7, 11):
        pieces = [text[i : i + size] for i in range(0, len(text), size)]
        assert _stream(text, pieces) == (content, reasoning), size


def test_parse_is_self_contained_and_repeatable():
    text, content, reasoning = REPLIES["analysis_then_final"]
    parser = BeamHarmonyTextParser()
    parser.parse_delta("<|channel|>analysis<|message|>left over")
    for _ in range(2):
        result = parser.parse(text)
        assert (result.content, result.reasoning_content) == (content, reasoning)


def test_registered_under_its_key():
    parser = ReasoningParserFactory.create_reasoning_parser(PARSER_KEY)
    assert isinstance(parser, BeamHarmonyTextParser)
    assert ReasoningParserFactory.needs_raw_special_tokens(PARSER_KEY)


#: Generated checkpoint directories (config files only) the routing tests point at.
_FILES = {}


def checkpoint_config() -> dict:
    """The supported checkpoint's config.json, as the target's layout states it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.layout_bf16 import (  # noqa: E501
        SUPPORTED_CONFIG,
    )

    full = {5, 11, 17, 23, 29, 35, 41, 47, 51}
    config = dict(SUPPORTED_CONFIG, architectures=["BeamForCausalLM"])
    config["layer_types"] = [
        "full_attention" if i in full else "sliding_attention" for i in range(52)
    ]
    config["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * 49
    config["rope_parameters"] = {
        "full_attention": None,
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    }
    return config


def quant_config() -> dict:
    """A ModelOpt NVFP4 recipe excluding every higher-precision Linear module."""
    patterns = ["lm_head", "model.embed_tokens", "model.layers.0*", "model.layers.1.*"]
    patterns.append("model.layers.2.*")
    for layer in range(3, 52):
        patterns += [f"model.layers.{layer}.mlp.gate", f"model.layers.{layer}.self_attn*"]
    return {
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": patterns,
        }
    }


def drafter_config(**dflash_overrides) -> dict:
    """The final DFlash 2 drafter's config.json fields the target's contract reads."""
    dflash = dict(
        target_layer_ids=[4, 16, 28, 40, 46],
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
        final_logit_softcapping=30,
    )
    dflash.update(dflash_overrides)
    return dict(hidden_size=5120, vocab_size=151936, draft_vocab_size=151936, dflash_config=dflash)


def write_checkpoint(directory, config=None, quant=None) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(checkpoint_config() if config is None else config)
    )
    if quant is not False:
        (directory / "hf_quant_config.json").write_text(
            json.dumps(quant_config() if quant is None else quant)
        )
    return str(directory)


@pytest.fixture(autouse=True, scope="module")
def _checkpoint_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("beam_checkpoint")
    _FILES["target"] = write_checkpoint(root / "target")
    (root / "drafter").mkdir()
    (root / "drafter" / "config.json").write_text(json.dumps(drafter_config()))
    _FILES["drafter"] = str(root / "drafter")
    yield
    _FILES.clear()


def _beam_config(**overrides):
    fields = dict(
        architectures=["BeamForCausalLM"],
        num_hidden_layers=52,
        hidden_size=5120,
        num_experts=256,
        quantization_config={"quant_algo": "NVFP4"},
        name_or_path=_FILES["target"],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mapping(**overrides):
    fields = dict(
        world_size=4, tp_size=4, pp_size=1, moe_tp_size=4, moe_ep_size=1, enable_attention_dp=False
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_route_claims_only_the_supported_deployment():
    from tensorrt_llm.quantization.mode import QuantAlgo

    route = routing_module("BeamForCausalLM").route
    target = route(_beam_config(), _mapping(), (10, 3), None, None)
    assert target.__name__ == "ModelingV2BeamFullModelNvfp4Sm103Tp4"
    assert route(_beam_config(), _mapping(), (10, 3), None, QuantAlgo.FP8) is target
    assert route(_beam_config(), _mapping(), (10, 3), _dflash(), QuantAlgo.FP8) is target
    declined = [
        (_beam_config(), _mapping(), (10, 0), None, None),
        (_beam_config(num_hidden_layers=6), _mapping(), (10, 3), None, None),
        (_beam_config(quantization_config=None), _mapping(), (10, 3), None, None),
        (_beam_config(), _mapping(tp_size=2, pp_size=2, moe_tp_size=2), (10, 3), None, None),
        (_beam_config(), _mapping(moe_tp_size=1, moe_ep_size=4), (10, 3), None, None),
        (_beam_config(), _mapping(enable_attention_dp=True), (10, 3), None, None),
        (_beam_config(), _mapping(), (10, 3), object(), None),
        (_beam_config(), _mapping(), (10, 3), None, QuantAlgo.NVFP4),
        (_beam_config(), _mapping(), (10, 3), None, "NVFP4"),
        (_beam_config(), _mapping(), (10, 3), None, "fp8_ds_mla"),
    ]
    for args in declined:
        with pytest.raises(NoTarget, match="KV cache|compute capability"):
            route(*args)


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "checkpoint_kv,runtime_kv,supported",
    [
        (None, "auto", True),
        (None, "fp8", True),
        (None, "nvfp4", False),
        ("FP8", "auto", True),
        (None, "fp8_ds_mla", False),
    ],
)
def test_engine_resolver_honors_effective_kv_format(
    monkeypatch, mode, checkpoint_kv, runtime_kv, supported
):
    """The engine's complete routing call accepts BF16 and FP8 KV and refuses other formats."""
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    config = SimpleNamespace(
        pretrained_config=_beam_config(),
        mapping=_mapping(),
        spec_config=None,
        quant_config=SimpleNamespace(kv_cache_quant_algo=checkpoint_kv),
    )
    args = SimpleNamespace(modeling_v2=mode, kv_cache_config=SimpleNamespace(dtype=runtime_kv))
    if supported:
        target = _router_index.modeling_v2_resolve(config, args)
        assert target.__name__ == "ModelingV2BeamFullModelNvfp4Sm103Tp4"
    elif mode == "auto":
        assert _router_index.modeling_v2_resolve(config, args) is None
    else:
        with pytest.raises(ValueError, match="KV cache"):
            _router_index.modeling_v2_resolve(config, args)


@pytest.mark.parametrize("max_num_tokens", [0, -1, 8193, 16384])
def test_route_refuses_an_unsupported_token_budget_before_reading_the_checkpoint(
    monkeypatch, max_num_tokens
):
    """The target computes at most 8192 tokens per engine step: a larger (or empty) budget is
    refused by selection, before the checkpoint's files are read, and ``require`` fails with it."""
    routing = routing_module("BeamForCausalLM")

    def unexpected_read(*args):
        pytest.fail("an unsupported budget must be refused before reading the checkpoint")

    monkeypatch.setattr(routing, "checkpoint_refusal", unexpected_read)
    args = SimpleNamespace(disable_overlap_scheduler=True, max_num_tokens=max_num_tokens)
    reason = rf"max_num_tokens {max_num_tokens} is outside \[1, 8192\]"
    with pytest.raises(NoTarget, match=reason):
        routing.route(_beam_config(), _mapping(), (10, 3), _dflash(), None, llm_args=args)
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    config = SimpleNamespace(
        pretrained_config=_beam_config(),
        mapping=_mapping(),
        spec_config=None,
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
    )
    args = SimpleNamespace(
        modeling_v2="require",
        kv_cache_config=SimpleNamespace(dtype="fp8"),
        max_num_tokens=max_num_tokens,
    )
    with pytest.raises(ValueError, match=reason):
        _router_index.modeling_v2_resolve(config, args)


@pytest.mark.parametrize("max_num_tokens", [1, 8192])
def test_route_accepts_token_budget_boundaries(max_num_tokens):
    route = routing_module("BeamForCausalLM").route
    args = SimpleNamespace(disable_overlap_scheduler=True, max_num_tokens=max_num_tokens)
    target = route(_beam_config(), _mapping(), (10, 3), _dflash(), None, llm_args=args)
    assert target.__name__ == "ModelingV2BeamFullModelNvfp4Sm103Tp4"


def test_load_pretrained_config_preserves_beam_rope(tmp_path):
    from tensorrt_llm._torch.models.modeling_beam import BeamConfig
    from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

    rope = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": None,
    }
    config = {
        "model_type": "beam",
        "num_hidden_layers": 6,
        "torch_dtype": "bfloat16",
        "rope_parameters": rope,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_pretrained_config(str(tmp_path))
    assert isinstance(loaded, BeamConfig)
    assert loaded.name_or_path == str(tmp_path)
    assert loaded.torch_dtype == torch.bfloat16
    assert loaded.rope_parameters == rope


# ---------------------------------------------------------------------------------------------
# DFlash 2 speculative decoding: routing and the drafter contract
# ---------------------------------------------------------------------------------------------


def _dflash(**overrides):
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

    fields = dict(max_draft_len=7, attention_backend="TRTLLM", speculative_model=_FILES["drafter"])
    fields.update(overrides)
    return DFlashDecodingConfig(**fields)


def _mtp():
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    return MTPDecodingConfig(max_draft_len=3)


def test_route_claims_the_dflash_deployment():
    route = routing_module("BeamForCausalLM").route
    target = route(_beam_config(), _mapping(), (10, 3), _dflash(), None)
    assert target.__name__ == "ModelingV2BeamFullModelNvfp4Sm103Tp4"
    args = SimpleNamespace(disable_overlap_scheduler=True)
    assert route(_beam_config(), _mapping(), (10, 3), _dflash(), None, llm_args=args) is target


@pytest.mark.parametrize(
    "spec,args,reason",
    [
        (lambda: _dflash(attention_backend="VANILLA"), None, "only the TRTLLM backend reads"),
        (lambda: _dflash(attention_backend="FA4"), None, "only the TRTLLM backend reads"),
        (lambda: _dflash(max_draft_len=15), None, "max_draft_len 15, not 7"),
        (lambda: _dflash(max_draft_len=3), None, "max_draft_len 3, not 7"),
        (lambda: _dflash(draft_len_schedule={1: 7, 4: 3}), None, "draft_len_schedule"),
        (lambda: _dflash(use_rejection_sampling=True), None, "rejection sampling"),
        (lambda: _dflash(speculative_model=None), None, "drafter checkpoint"),
        (
            lambda: _dflash(),
            SimpleNamespace(disable_overlap_scheduler=False),
            "overlap scheduler",
        ),
        (lambda: _mtp(), None, "is not DFlash"),
    ],
    ids=["vanilla", "fa4", "k15", "k3", "schedule", "rejection", "no-drafter", "overlap", "mtp"],
)
def test_route_refuses_other_speculation_clearly(spec, args, reason):
    route = routing_module("BeamForCausalLM").route
    with pytest.raises(NoTarget, match=reason):
        route(_beam_config(), _mapping(), (10, 3), spec(), None, llm_args=args)


_TARGET_CONFIG = {"hidden_size": 5120, "vocab_size": 151936}


def _check_drafter_config():
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.layout import (  # noqa: E501
        check_drafter_config,
    )

    return check_drafter_config


def test_drafter_contract_accepts_the_dflash2_drafter():
    _check_drafter_config()(drafter_config(), _TARGET_CONFIG, [4, 16, 28, 40, 46])


@pytest.mark.parametrize(
    "config,layers,reason",
    [
        (drafter_config(), [3, 15, 27, 39, 45], "target_layer_ids"),
        (drafter_config(target_layer_ids=[4, 16, 28, 40, 52]), [4, 16, 28, 40, 52], "0..51"),
        (drafter_config(target_layer_ids=[16, 4]), [16, 4], "ascending"),
        (drafter_config(final_logit_softcapping=None), [4, 16, 28, 40, 46], "softcapping"),
        (drafter_config(selector_rank=0), [4, 16, 28, 40, 46], "selector_rank"),
    ],
    ids=["shifted", "out-of-range", "unordered", "no-softcap", "no-selector"],
)
def test_drafter_contract_refuses_a_mismatch(config, layers, reason):
    with pytest.raises(ValueError, match=reason):
        _check_drafter_config()(config, _TARGET_CONFIG, layers)


def test_drafter_contract_refuses_another_vocabulary_or_width():
    with pytest.raises(ValueError, match="draft_vocab_size 32000"):
        _check_drafter_config()(
            {**drafter_config(), "draft_vocab_size": 32000}, _TARGET_CONFIG, [4, 16, 28, 40, 46]
        )
    with pytest.raises(ValueError, match="hidden_size 4096"):
        _check_drafter_config()(
            {**drafter_config(), "hidden_size": 4096}, _TARGET_CONFIG, [4, 16, 28, 40, 46]
        )


# ---------------------------------------------------------------------------------------------
# Selection and construction share one configuration validator
# ---------------------------------------------------------------------------------------------


def _refused_checkpoints() -> list:
    """(id, config.json, hf_quant_config.json or False for none) the target cannot compute."""
    out = []
    for field, value in (
        ("head_dim", 64),
        ("sliding_window", 1024),
        ("num_key_value_heads", 4),
        ("rms_norm_eps", 1e-5),
        ("final_logit_softcapping", None),
        ("vocab_size", 152064),
        ("intermediate_size", 16384),
    ):
        out.append((field, {**checkpoint_config(), field: value}, None))
    config = checkpoint_config()
    config["layer_types"] = config["layer_types"][:51]
    out.append(("layer_types", config, None))
    config = checkpoint_config()
    config["rope_parameters"] = {"sliding_attention": {"rope_type": "yarn", "rope_theta": 1e6}}
    out.append(("rope", config, None))
    recipe = quant_config()
    recipe["quantization"]["group_size"] = 32
    out.append(("group_size", checkpoint_config(), recipe))
    recipe = quant_config()
    recipe["quantization"]["kv_cache_quant_algo"] = "FP8"
    out.append(("recipe_kv", checkpoint_config(), recipe))
    recipe = quant_config()
    recipe["quantization"]["exclude_modules"].remove("lm_head")
    out.append(("head_quantized", checkpoint_config(), recipe))
    out.append(("no_recipe", checkpoint_config(), False))
    return out


_REFUSED = _refused_checkpoints()


@pytest.mark.parametrize(
    "config,quant", [(c, q) for _, c, q in _REFUSED], ids=[i for i, _, _ in _REFUSED]
)
def test_selection_and_construction_refuse_the_same_checkpoint(tmp_path, config, quant):
    """Fields the route's quick checks do not read still decide selection, exactly as they
    decide construction: auto falls back (NoTarget), require fails with the same reason, and
    the target's own configuration step raises it too."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
        modeling,
    )

    directory = write_checkpoint(tmp_path / "ckpt", config=config, quant=quant)
    with pytest.raises(ValueError) as built:
        modeling._checkpoint_config(SimpleNamespace(name_or_path=directory))
    reason = str(built.value)
    route = routing_module("BeamForCausalLM").route
    with pytest.raises(NoTarget) as declined:
        route(_beam_config(name_or_path=directory), _mapping(), (10, 3), None, None)
    assert reason in str(declined.value)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_resolver_falls_back_or_fails_with_the_checkpoint_reason(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    directory = write_checkpoint(tmp_path / "ckpt", config={**checkpoint_config(), "head_dim": 64})
    config = SimpleNamespace(
        pretrained_config=_beam_config(name_or_path=directory),
        mapping=_mapping(),
        spec_config=None,
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
    )
    args = SimpleNamespace(modeling_v2=mode, kv_cache_config=SimpleNamespace(dtype="auto"))
    if mode == "auto":
        assert _router_index.modeling_v2_resolve(config, args) is None
    else:
        with pytest.raises(ValueError, match="unsupported head_dim=64"):
            _router_index.modeling_v2_resolve(config, args)


@pytest.mark.parametrize(
    "drafter,reason",
    [
        ({**drafter_config(), "hidden_size": 4096}, "hidden_size 4096"),
        (drafter_config(selector_top_k=0), "selector_top_k"),
        (drafter_config(final_logit_softcapping=None), "softcapping"),
        (None, "not a readable JSON file"),
    ],
    ids=["width", "selector", "softcap", "no-config"],
)
def test_selection_refuses_a_drafter_the_target_would_refuse(tmp_path, drafter, reason):
    directory = tmp_path / "drafter"
    directory.mkdir()
    if drafter is not None:
        (directory / "config.json").write_text(json.dumps(drafter))
    route = routing_module("BeamForCausalLM").route
    with pytest.raises(NoTarget, match=reason):
        route(_beam_config(), _mapping(), (10, 3), _dflash(speculative_model=str(directory)), None)


def test_supported_checkpoint_reaches_the_target_with_its_config():
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
        modeling,
    )

    route = routing_module("BeamForCausalLM").route
    assert route(_beam_config(), _mapping(), (10, 3), _dflash(), None).__name__ == (
        "ModelingV2BeamFullModelNvfp4Sm103Tp4"
    )
    directory, config = modeling._checkpoint_config(SimpleNamespace(name_or_path=_FILES["target"]))
    assert directory == _FILES["target"] and config == checkpoint_config()


# ---------------------------------------------------------------------------------------------
# FP8 KV cache: the scale table and the target's configuration
# ---------------------------------------------------------------------------------------------


def _scale_table(path, target=None, draft=None, **extra):
    table = {
        "target": {"layers": 52, "scales": {str(i): 1.0 for i in range(52)}}
        if target is None
        else target,
        **({} if draft is False else {"draft": draft or _DRAFT_SECTION}),
        **extra,
    }
    path.write_text(json.dumps(table))
    return str(path)


_DRAFT_SECTION = {"layers": 5, "scales": {"0": 2.0, "1": 2.0, "2": 2.0, "3": 8.0, "4": 4.0}}


def test_read_kv_cache_scales_returns_validated_per_layer_scales(tmp_path):
    import hashlib

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.weights import (  # noqa: E501
        read_kv_cache_scales,
    )

    path = _scale_table(tmp_path / "scales.json", format="note")
    table = read_kv_cache_scales(path)
    assert table["target"] == (1.0,) * 52
    assert table["draft"] == (2.0, 2.0, 2.0, 8.0, 4.0)
    assert table["sha256"] == hashlib.sha256((tmp_path / "scales.json").read_bytes()).hexdigest()
    assert read_kv_cache_scales(_scale_table(tmp_path / "t.json", draft=False))["draft"] is None


@pytest.mark.parametrize(
    "target,draft,reason",
    [
        ({"layers": 52, "scales": {str(i): 1.0 for i in range(51)}}, None, "0..51"),
        ({"layers": 52, "scales": {str(i): 1.0 for i in range(1, 53)}}, None, "0..51"),
        (None, {"layers": 5, "scales": {"0": 2.0, "1": 2.0, "2": 0.0, "3": 8.0, "4": 4.0}}, "> 0"),
        (None, {"layers": 5, "scales": {"0": 2, "1": -2, "2": 2, "3": 8, "4": 4}}, "> 0"),
        (None, {"layers": 5, "scales": {"0": True, "1": 2, "2": 2, "3": 8, "4": 4}}, "> 0"),
        (None, {"layers": 5, "scales": {"0": "2", "1": 2, "2": 2, "3": 8, "4": 4}}, "> 0"),
        (None, {"scales": {"0": 2.0}}, "'layers'"),
        ({}, None, "no 'target'"),
    ],
    ids=["missing", "shifted", "zero", "negative", "bool", "string", "no-layers", "no-target"],
)
def test_read_kv_cache_scales_refuses_unusable_tables(tmp_path, target, draft, reason):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.weights import (  # noqa: E501
        read_kv_cache_scales,
    )

    path = tmp_path / "scales.json"
    if target == {}:
        path.write_text(json.dumps({"draft": _DRAFT_SECTION}))
    else:
        _scale_table(path, target=target, draft=draft)
    with pytest.raises(ValueError, match=reason):
        read_kv_cache_scales(str(path))
    for bad in (float("nan"), float("inf")):
        path.write_text(json.dumps({"target": {"layers": 1, "scales": {"0": bad}}}))
        with pytest.raises(ValueError, match="> 0"):
            read_kv_cache_scales(str(path))


# Finite positive Python floats whose FP32 scale or reciprocal is zero, subnormal or infinite:
# 1e-300 -> s = 0; 1e300 -> s = inf, 1/s = 0; 1e-40 -> subnormal s, 1/s = inf.
_UNUSABLE_FP32_SCALES = [1e-300, 1e300, 1e-40]


@pytest.mark.parametrize("bad", _UNUSABLE_FP32_SCALES)
@pytest.mark.parametrize("section", ["target", "draft"])
def test_read_kv_cache_scales_refuses_scales_unusable_in_fp32(tmp_path, section, bad):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.weights import (  # noqa: E501
        read_kv_cache_scales,
    )

    target = {"layers": 52, "scales": {str(i): 1.0 for i in range(52)}}
    draft = json.loads(json.dumps(_DRAFT_SECTION))
    (target if section == "target" else draft)["scales"]["3"] = bad
    path = _scale_table(tmp_path / "scales.json", target=target, draft=draft)
    with pytest.raises(ValueError, match=f"{section}: layer 3 scale .* unusable in FP32"):
        read_kv_cache_scales(path)


def test_fp8_kv_scale_pairs_are_the_ops_fp32_factors():
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4._attention import (
        fp8_kv_scale_pairs,
    )

    approved = [1.0, 2.0, 2.0, 2.0, 8.0, 4.0, 0.5, 1.5]
    s, inv = fp8_kv_scale_pairs(approved, "scales")
    assert s.dtype == inv.dtype == torch.float32
    assert s.tolist() == approved
    assert inv.tolist() == [1.0, 0.5, 0.5, 0.5, 0.125, 0.25, 2.0, float(torch.tensor(1 / 1.5))]
    for bad in _UNUSABLE_FP32_SCALES + [0.0, -1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match=r"scales: layer 1 scale .* unusable in FP32"):
            fp8_kv_scale_pairs([1.0, bad, 2.0], "scales")


def test_target_kv_cache_fp8_refuses_scales_unusable_in_fp32():
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.component_bf16 import (  # noqa: E501
        BeamCheckpointLayoutBf16Sm103Tp4,
    )

    def component():
        layout = BeamCheckpointLayoutBf16Sm103Tp4.__new__(BeamCheckpointLayoutBf16Sm103Tp4)
        layout.__dict__.update(schedule=SimpleNamespace(num_layers=3), device=torch.device("cpu"))
        layout._init_kv_cache_mode()
        return layout

    for bad in _UNUSABLE_FP32_SCALES:
        layout = component()
        with pytest.raises(ValueError, match="target KV-cache scales: layer 2 .* unusable in FP32"):
            layout.set_kv_cache_fp8([1.0, 2.0, bad])
        assert layout.kv_cache_scales is None and layout.kv_scale_args(0) is None
    layout = component()
    layout.set_kv_cache_fp8([1.0, 2.0, 8.0])
    orig_quant, quant_orig = layout.kv_scale_args(2)
    assert quant_orig.tolist() == [8.0] and orig_quant.tolist() == [0.125]


def test_target_kv_cache_mode_and_scale_table_requirements():
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (  # noqa: E501
        ModelingV2BeamFullModelNvfp4Sm103Tp4,
        kv_cache_quant_algo,
    )
    from tensorrt_llm.quantization.mode import QuantAlgo

    def config(algo):
        return SimpleNamespace(quant_config=SimpleNamespace(kv_cache_quant_algo=algo))

    assert kv_cache_quant_algo(config(None)) is None
    assert kv_cache_quant_algo(config(QuantAlgo.FP8)) == QuantAlgo.FP8
    assert kv_cache_quant_algo(config("FP8")) == QuantAlgo.FP8
    with pytest.raises(ValueError, match="BF16 or FP8"):
        kv_cache_quant_algo(config(QuantAlgo.NVFP4))

    def target(algo, drafter):
        model = ModelingV2BeamFullModelNvfp4Sm103Tp4.__new__(ModelingV2BeamFullModelNvfp4Sm103Tp4)
        model.__dict__.update(kv_cache_quant_algo=algo, draft_model=drafter)
        return model

    table = {"target": (1.0,) * 52, "draft": (2.0, 2.0, 2.0, 8.0, 4.0)}
    target(QuantAlgo.FP8, object())._check_kv_cache_scales(table)
    target(None, object())._check_kv_cache_scales(None)
    target(QuantAlgo.FP8, None)._check_kv_cache_scales({"target": (1.0,) * 52, "draft": None})
    with pytest.raises(ValueError, match="needs a scale table"):
        target(QuantAlgo.FP8, object())._check_kv_cache_scales(None)
    with pytest.raises(ValueError, match="KV cache is BF16"):
        target(None, None)._check_kv_cache_scales(table)
    with pytest.raises(ValueError, match="draft"):
        target(QuantAlgo.FP8, object())._check_kv_cache_scales({"target": (1.0,) * 52})
