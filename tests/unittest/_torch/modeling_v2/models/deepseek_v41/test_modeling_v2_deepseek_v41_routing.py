# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash target routing, driven by synthetic configs on CPU.

Proved here: the three shipped targets (dep2 context role; tep8 moe tp4 x ep2 and dep, at 8 or 32 ranks, generation
roles) are claimed by their layouts under ``auto`` (also when ``modeling_v2`` is left unset: they are on by
default) and ``require`` and declined under ``off``, the vision-bearing composite config routes like the
text-only one, ``enable_lm_head_tp_in_adp`` is accepted by the generation roles that run it
and declined by the others, ``bounded_replay_on_generation`` and a vision checkpoint served with its image encoder
are declined by every target, a checkpoint whose routed experts are not NVFP4 is declined by every target, and every
near miss is declined with a message that names the criterion that failed. Every engine setting a target does not
implement (``limits.py``: the page size, the MoE backend, load balancer, combine and all-to-all knobs, the CSA2 options,
determinism, LoRA, torch.compile, dummy weights, Ray / no MPI, a draft longer than the Engram ring, piecewise prefill
graphs) is declined by that target: ``auto`` keeps the built-in model and ``require`` raises naming the setting.
"""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm.llmapi.llm_args import (
    CSA2SparseAttentionConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_DEP2 = dict(world_size=2, tp_size=2, enable_attention_dp=True, moe_ep_size=2, moe_tp_size=1)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=2, moe_tp_size=4)
_DEP8 = dict(world_size=8, tp_size=8, enable_attention_dp=True, moe_ep_size=8, moe_tp_size=1)
_DEP32 = dict(world_size=32, tp_size=32, enable_attention_dp=True, moe_ep_size=32, moe_tp_size=1)
_DEP2_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Dep2"
)
_TEP8_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2"
)
# One attention-DP generation target serves both worlds.
_DEP_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__dep.modeling.ModelingV2DeepseekV41FlashNvfp4Sm103Dep"
)
_DSPARK = DSparkDecodingConfig(max_draft_len=5)
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """``mode`` is the ``modeling_v2`` option; ``"unset"`` leaves it at its default, as a deployment that never
    names it."""
    kwargs.setdefault("kv_cache_config", KvCacheConfig(dtype="fp8", tokens_per_block=128))
    if mode != "unset":
        kwargs["modeling_v2"] = mode
    return TorchLlmArgs(model=_DUMMY_MODEL, **kwargs)


def _config(overrides=None, composite=False):
    """The shape DeepSeek-V4.1-Flash-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["DeepseekV41ForCausalLM"],
        num_hidden_layers=40,
        hidden_size=5120,
        n_routed_experts=384,
        q_lora_rank=1280,
        candidate_source_layer_id=2,
    )
    fields.update(overrides or {})
    text = PretrainedConfig(**fields)
    if not composite:
        return text
    # The released checkpoint carries a vision tower: the text model is nested.
    return PretrainedConfig(architectures=["DeepseekV41ForConditionalGeneration"], text_config=text)


def _experts(algo=QuantAlgo.NVFP4, group_size=16, layers=range(40)):
    """The per-layer routed-expert entries ``ModelConfig`` sets from the checkpoint's expert storage (U8 NVFP4 for
    DeepSeek-V4.1-Flash-NVFP4)."""
    return {
        f"model.layers.{i}.mlp.experts": QuantConfig(quant_algo=algo, group_size=group_size)
        for i in layers
    }


def _model_config(
    pretrained_config,
    spec_config=None,
    sparse="csa2",
    kv=QuantAlgo.FP8,
    experts="nvfp4",
    **mapping_kwargs,
):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        spec_config=spec_config,
        sparse_attention_config=CSA2SparseAttentionConfig() if sparse == "csa2" else None,
        quant_config=QuantConfig(kv_cache_quant_algo=kv),
        quant_config_dict=_experts() if experts == "nvfp4" else experts,
    )


@pytest.mark.parametrize("mode", ["unset", "auto", "require"])
@pytest.mark.parametrize("composite", [False, True], ids=["text", "vision-composite"])
@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, target",
    [
        (_DEP2, None, _DEP2_TARGET),
        (_TEP8, None, _TEP8_TARGET),
        (_TEP8, _DSPARK, _TEP8_TARGET),
        (_DEP8, None, _DEP_TARGET),
        (_DEP8, _DSPARK, _DEP_TARGET),
        (dict(_DEP8, enable_lm_head_tp_in_adp=True), _DSPARK, _DEP_TARGET),
        (_DEP32, None, _DEP_TARGET),
        (_DEP32, _DSPARK, _DEP_TARGET),
        (dict(_DEP32, enable_lm_head_tp_in_adp=True), _DSPARK, _DEP_TARGET),
    ],
    ids=[
        "dep2", "tep8", "tep8-dspark", "dep8", "dep8-dspark", "dep8-lm-head-tp", "dep32",
        "dep32-dspark", "dep32-lm-head-tp",
    ],
)  # fmt: skip
def test_targets_match(mode, composite, mapping_kwargs, spec_config, target):
    config = _model_config(_config(composite=composite), spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == target


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config",
    [(_DEP2, None), (_TEP8, _DSPARK), (_DEP8, _DSPARK), (_DEP32, _DSPARK)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)
def test_off_keeps_the_built_in_model(mapping_kwargs, spec_config):
    """The targets are on by default (no ``@disable_v2_by_default``): ``modeling_v2="off"`` is how a matching
    deployment keeps the built-in model."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("off")) is None


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, spec_config, missed",
    [
        (dict(num_hidden_layers=61), _DEP2, None, r"shape \(61, 5120, 384, 1280\)"),
        (dict(hidden_size=7168), _TEP8, None, r"shape \(40, 7168, 384, 1280\)"),
        (dict(), dict(_DEP2, enable_attention_dp=False), None, "attention_dp=False"),
        (dict(), dict(_TEP8, enable_attention_dp=True), None, "attention_dp=True"),
        (dict(), dict(_TEP8, moe_ep_size=8, moe_tp_size=1), None, "moe_ep=8"),
        (dict(), dict(world_size=4, tp_size=4, moe_ep_size=4), None, "world_size=4"),
        (dict(), _DEP2, _DSPARK, "DSPARK.*context role"),
        (dict(), _TEP8, MTPDecodingConfig(max_draft_len=3), "MTP.*DSpark"),
        (dict(), _TEP8, _NGRAM, "NGRAM.*DSpark"),
        (dict(), dict(_DEP8, enable_attention_dp=False), None, "attention_dp=False"),
        (dict(), dict(_DEP8, moe_ep_size=4, moe_tp_size=2), None, "moe_ep=4"),
        (dict(), dict(world_size=16, tp_size=16, enable_attention_dp=True, moe_ep_size=16), None, "world_size=16"),
        (dict(), _DEP8, MTPDecodingConfig(max_draft_len=3), "MTP.*DSpark"),
        (dict(), _DEP32, _NGRAM, "NGRAM.*DSpark"),
        (dict(hidden_size=7168), _DEP32, None, r"shape \(40, 7168, 384, 1280\)"),
    ],
    ids=[
        "other-depth", "other-width", "dep2-without-adp", "tep8-with-adp", "tep8-ep8", "tp4",
        "ctx-with-dspark", "mtp", "ngram", "dep8-without-adp", "dep8-moe-tp2-ep4", "dep16",
        "dep8-mtp", "dep32-ngram", "dep32-other-width",
    ],
)  # fmt: skip
def test_near_misses_do_not_match(config_kwargs, mapping_kwargs, spec_config, missed):
    config = _model_config(_config(config_kwargs), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "quant_config, quant_config_dict, missed",
    [
        (
            QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
            None,
            "routed experts unquantized in 40 layers",
        ),
        (
            QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, kv_cache_quant_algo=QuantAlgo.FP8),
            None,
            "routed experts FP8_BLOCK_SCALES in 40 layers",
        ),
        (
            QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
            _experts(QuantAlgo.W4A8_MXFP4_MXFP8, group_size=32),
            "routed experts W4A8_MXFP4_MXFP8 in 40 layers",
        ),
        (
            QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
            {**_experts(), **_experts(QuantAlgo.FP8_BLOCK_SCALES, layers=[39])},
            r"routed experts FP8_BLOCK_SCALES in 1 layers \(39\.\.39\)",
        ),
    ],
    ids=["bf16", "fp8", "mxfp4", "one-layer-fp8"],
)
@pytest.mark.parametrize(
    "mapping_kwargs, spec_config",
    [(_DEP2, None), (_TEP8, _DSPARK), (_DEP8, _DSPARK), (_DEP32, _DSPARK)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)
def test_non_nvfp4_routed_experts_do_not_match(
    quant_config, quant_config_dict, missed, mapping_kwargs, spec_config
):
    """The Flash shape with an fp8 KV cache but routed experts that are not NVFP4 (the native MXFP4 release, an
    FP8 or bf16 export): every target's loader reads NVFP4 expert tensors, so ``auto`` keeps the built-in model and
    ``require`` raises naming the experts' format."""
    config = ModelConfig(
        pretrained_config=_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        sparse_attention_config=CSA2SparseAttentionConfig(),
        quant_config=quant_config,
        quant_config_dict=quant_config_dict,
    )
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_a_global_nvfp4_quant_config_matches():
    """A checkpoint without a per-layer map whose global algorithm is NVFP4 routes (the built-in model reads the
    experts' format from the global config then)."""
    config = ModelConfig(
        pretrained_config=_config(),
        mapping=Mapping(**_DEP2),
        sparse_attention_config=CSA2SparseAttentionConfig(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=QuantAlgo.FP8),
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _DEP2_TARGET


def test_other_sm_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_config(), **_DEP2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "kv, dtype", [(None, "auto"), (QuantAlgo.NVFP4, "auto"), (QuantAlgo.FP8, "nvfp4")]
)
def test_only_an_fp8_kv_cache_matches(kv, dtype):
    config = _model_config(_config(), kv=kv, **_DEP2)
    args = dict(kv_cache_config=KvCacheConfig(dtype=dtype, tokens_per_block=128))
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="KV cache"):
        modeling_v2_resolve(config, _llm_args("require", **args))


def test_the_kv_cache_dtype_argument_overrides_the_checkpoint():
    config = _model_config(_config(), kv=None, **_DEP2)
    args = _llm_args("require", kv_cache_config=KvCacheConfig(dtype="fp8", tokens_per_block=128))
    assert _qualified(modeling_v2_resolve(config, args)) == _DEP2_TARGET


def test_swa_scratch_reuse_is_taken():
    # The engine's DeepSeek-V4.1 default (the official deployments leave it unset): the targets write only window rows.
    config = _model_config(_config(), **_DEP2)
    for value in (True, False):
        args = _llm_args(
            "require",
            kv_cache_config=KvCacheConfig(
                dtype="fp8", tokens_per_block=128, enable_swa_scratch_reuse=value
            ),
        )
        assert _qualified(modeling_v2_resolve(config, args)) == _DEP2_TARGET


def test_csa2_sparse_attention_is_required():
    config = _model_config(_config(), sparse=None, **_DEP2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="CSA2CacheManager"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_lm_head_tp_in_adp_is_declined():
    config = _model_config(_config(), **dict(_DEP2, enable_lm_head_tp_in_adp=True))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="enable_lm_head_tp_in_adp"):
        modeling_v2_resolve(config, _llm_args("require"))


_MEGAMOE = dict(moe_config={"backend": "MEGAMOE_CUTEDSL"})


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "mapping_kwargs, target",
    [(_DEP8, _DEP_TARGET), (_DEP32, _DEP_TARGET)],
    ids=["dep8", "dep32"],
)
def test_megamoe_backend_is_claimed_by_the_dep_generation_targets(mode, mapping_kwargs, target):
    config = _model_config(
        _config(), _DSPARK, **dict(mapping_kwargs, enable_lm_head_tp_in_adp=True)
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode, **_MEGAMOE))) == target


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config",
    [(_DEP2, None), (_TEP8, None), (_TEP8, _DSPARK)],
    ids=["dep2", "tep8", "tep8-dspark"],
)
def test_megamoe_backend_is_declined_where_it_is_not_implemented(mapping_kwargs, spec_config):
    # These targets run the TRTLLM fp4 runner: honouring the config silently would be a different MoE.
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", **_MEGAMOE)) is None
    with pytest.raises(ValueError, match="MEGAMOE_CUTEDSL"):
        modeling_v2_resolve(config, _llm_args("require", **_MEGAMOE))
    # and the same layout without the backend still routes
    assert modeling_v2_resolve(config, _llm_args("auto")) is not None


def test_per_layer_dynamic_placement_is_declined_with_megamoe():
    balancer = {"mode": "per_layer_dynamic", "num_slots": 416}
    config = _model_config(_config(), _DSPARK, **_DEP32)
    args = dict(moe_config={"backend": "MEGAMOE_CUTEDSL", "load_balancer": balancer})
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="per_layer_dynamic"):
        modeling_v2_resolve(config, _llm_args("require", **args))


def test_the_static_balancer_config_of_the_gen_deployments_is_accepted():
    warmup = dict(
        count_every_steps=4,
        decay=0.5,
        interval_steps=12000,
        max_migrations=100000,
        max_moved_fraction=0.1,
        min_gain=0.01,
        min_pos=2,
        min_round_gain=0.5,
        min_rows=1048576,
        planner="cost",
        start_step=0,
    )  # fmt: skip  (the gen-B deployment's block)
    balancer = {"num_slots": 416, "layer_updates_per_iter": 0, "warmup_placement": warmup}
    config = _model_config(_config(), _DSPARK, **_DEP32)
    args = dict(moe_config={"backend": "MEGAMOE_CUTEDSL", "load_balancer": balancer})
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", **args))) == _DEP_TARGET


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, target",
    [(_DEP2, None, _DEP2_TARGET), (_TEP8, _DSPARK, _TEP8_TARGET), (_DEP8, _DSPARK, _DEP_TARGET),
     (_DEP32, _DSPARK, _DEP_TARGET)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)  # fmt: skip
def test_bounded_replay_on_generation_is_declined(mapping_kwargs, spec_config, target):
    """`bounded_replay_on_generation` (disaggregated remote-tail replay) is implemented by no target: every layout
    declines it (``auto`` keeps the built-in model, ``require`` raises naming it); with it off, routing is unchanged."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", bounded_replay_on_generation=True)) is None
    with pytest.raises(ValueError, match="bounded_replay_on_generation"):
        modeling_v2_resolve(config, _llm_args("require", bounded_replay_on_generation=True))
    for mode in ("auto", "require"):
        args = _llm_args(mode, bounded_replay_on_generation=False)
        assert _qualified(modeling_v2_resolve(config, args)) == target


def _vision_config(composite: bool, layers: int = 32):
    """The checkpoint as the loader builds it with its vision tower: the composite (text nested) or the flat text
    config, either carrying the vision tower's config and the image token."""
    vision = PretrainedConfig(num_hidden_layers=layers, hidden_size=1024)
    if composite:
        return PretrainedConfig(
            architectures=["DeepseekV41ForConditionalGeneration"],
            text_config=_config(),
            vision_config=vision,
            image_token_id=129264,
        )
    return _config(
        dict(
            architectures=["DeepseekV41ForConditionalGeneration"],
            vision_config=vision,
            image_token_id=129264,
        )
    )


@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composite"])
@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, target",
    [(_DEP2, None, _DEP2_TARGET), (_TEP8, _DSPARK, _TEP8_TARGET), (_DEP8, _DSPARK, _DEP_TARGET),
     (_DEP32, _DSPARK, _DEP_TARGET)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)  # fmt: skip
def test_vision_checkpoint_with_the_image_encoder_is_declined(
    composite, mapping_kwargs, spec_config, target
):
    """The targets serve text only: a checkpoint with a vision tower served with the image encoder on is declined
    (``auto`` keeps the built-in model, ``require`` raises naming it); with ``disable_mm_encoder`` it routes to the
    target, which refuses image requests itself (test_modeling_v2_deepseek_v41_image_input.py)."""
    config = _model_config(_vision_config(composite), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        modeling_v2_resolve(config, _llm_args("require"))
    for mode in ("auto", "require"):
        args = _llm_args(mode, disable_mm_encoder=True)
        assert _qualified(modeling_v2_resolve(config, args)) == target


def test_vision_tower_of_zero_layers_is_text_only():
    """The loader reads a checkpoint whose vision config has no layer as text only; so does routing."""
    config = _model_config(_vision_config(composite=False, layers=0), **_DEP2)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _DEP2_TARGET


# ---- the refusal lists (`limits.py`) ----------------------------------------------------------------------------

_LAYOUTS = [(_DEP2, None), (_TEP8, _DSPARK), (_DEP8, _DSPARK), (_DEP32, _DSPARK)]
_LAYOUT_IDS = ["dep2", "tep8", "dep8", "dep32"]


def _with(args: TorchLlmArgs, **fields) -> TorchLlmArgs:
    """``args`` with ``fields`` set as given, past the argument validators (routing reads only the values)."""
    for name, value in fields.items():
        object.__setattr__(args, name, value)
    return args


def _declined(config, missed, args_of):
    """``args_of(mode)`` -> llm args: ``auto`` keeps the built-in model, ``require`` raises naming ``missed``."""
    assert modeling_v2_resolve(config, args_of("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, args_of("require"))


@pytest.mark.parametrize("tokens_per_block", [32, 64, 256])
@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
def test_pages_other_than_128_rows_are_declined(tokens_per_block, mapping_kwargs, spec_config):
    """Every family op addresses a CSA2 page as 128 rows; the cache manager also accepts 256."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    kv = KvCacheConfig(dtype="fp8", tokens_per_block=tokens_per_block)
    _declined(
        config, f"tokens_per_block {tokens_per_block}", lambda m: _llm_args(m, kv_cache_config=kv)
    )


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, backend, ok",
    [
        (_DEP2, None, "CUTEDSL", True), (_DEP2, None, "TRTLLM", False), (_DEP2, None, "CUTLASS", False),
        (_DEP2, None, "MEGAMOE_CUTEDSL", False),
        (_TEP8, _DSPARK, "TRTLLM", True), (_TEP8, _DSPARK, "CUTEDSL", False),
        (_TEP8, _DSPARK, "MEGAMOE_CUTEDSL", False),
        (_DEP8, _DSPARK, "TRTLLM", True), (_DEP8, _DSPARK, "MEGAMOE_CUTEDSL", True),
        (_DEP8, _DSPARK, "CUTLASS", False), (_DEP32, _DSPARK, "CUTEDSL", False),
    ],
    ids=[
        "dep2-cutedsl", "dep2-trtllm", "dep2-cutlass", "dep2-megamoe", "tep8-trtllm", "tep8-cutedsl",
        "tep8-megamoe", "dep8-trtllm", "dep8-megamoe", "dep8-cutlass", "dep32-cutedsl",
    ],
)  # fmt: skip
def test_only_the_moe_backends_a_target_runs_are_taken(mapping_kwargs, spec_config, backend, ok):
    """``AUTO`` (the default) and the backend each target runs route; any other backend is declined."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    args = dict(moe_config={"backend": backend})
    if ok:
        assert modeling_v2_resolve(config, _llm_args("require", **args)) is not None
    else:
        _declined(config, f"moe_config.backend {backend}", lambda m: _llm_args(m, **args))


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, backend",
    [(_DEP2, None, "AUTO"), (_TEP8, _DSPARK, "AUTO"), (_DEP8, _DSPARK, "TRTLLM"), (_DEP32, _DSPARK, "AUTO")],
    ids=_LAYOUT_IDS,
)  # fmt: skip
def test_a_load_balancer_off_megamoe_is_declined(mapping_kwargs, spec_config, backend):
    """Off MegaMoE every target places each expert once, contiguously: any load balancer (static slots, initial
    assignments, warmup placement, online) would be ignored, and the engine would build a balancer no layer joins."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    for balancer in ({"num_slots": 384}, {"num_slots": 384, "layer_updates_per_iter": 2}):
        args = dict(moe_config={"backend": backend, "load_balancer": balancer})
        _declined(config, "moe_config.load_balancer", lambda m: _llm_args(m, **args))


@pytest.mark.parametrize("mapping_kwargs", [_DEP8, _DEP32], ids=["dep8", "dep32"])
@pytest.mark.parametrize(
    "moe_config, missed",
    [
        (
            {"load_balancer": {"num_slots": 416, "layer_updates_per_iter": 2}},
            "layer_updates_per_iter 2",
        ),
        ({"load_balancer": {"num_slots": 390}}, "num_slots 390"),
        ({"load_balancer": {"num_slots": 352}}, "num_slots 352"),
        ({"max_num_tokens": 1024}, "moe_config.max_num_tokens 1024 below max_num_tokens 1280"),
    ],
    ids=[
        "online-eplb",
        "slots-not-multiple",
        "fewer-slots-than-experts",
        "moe-cap-below-engine-cap",
    ],
)
def test_what_megamoe_does_not_run_is_declined(mapping_kwargs, moe_config, missed):
    """MegaMoE on the dep target: the static and the warmup placement only, a slot count the kernel places, and every
    step launched whole (no MoE chunking: a MoE token cap below the engine's is declined)."""
    config = _model_config(_config(), _DSPARK, **mapping_kwargs)
    args = dict(moe_config={"backend": "MEGAMOE_CUTEDSL", **moe_config}, max_num_tokens=1280)
    _declined(config, missed, lambda m: _llm_args(m, **args))
    # the same deployment with the cap at the engine's routes
    ok = dict(
        moe_config={"backend": "MEGAMOE_CUTEDSL", "max_num_tokens": 1280}, max_num_tokens=1280
    )
    assert modeling_v2_resolve(config, _llm_args("require", **ok)) is not None


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, backend, ok",
    [
        (_DEP2, None, "AUTO", False), (_DEP8, _DSPARK, "TRTLLM", False),
        (_DEP8, _DSPARK, "MEGAMOE_CUTEDSL", True), (_TEP8, _DSPARK, "AUTO", True),
    ],
    ids=["dep2", "dep8-trtllm", "dep8-megamoe", "tep8"],
)  # fmt: skip
def test_low_precision_combine_where_a_target_combines_in_bf16_is_declined(
    mapping_kwargs, spec_config, backend, ok
):
    """dep2's and dep's all-to-all (TRTLLM runner) combine in bf16; stock's MegaMoE reads its combine format
    from MEGAMOE_COMBINE_FORMAT, not this switch, and tep8 has no MoE communication (no attention DP)."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    args = dict(moe_config={"backend": backend, "use_low_precision_moe_combine": True})
    if ok:
        assert modeling_v2_resolve(config, _llm_args("require", **args)) is not None
    else:
        _declined(config, "use_low_precision_moe_combine", lambda m: _llm_args(m, **args))


@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "fields, missed",
    [
        (dict(use_fp8_staging=False), "use_fp8_staging false"),
        (dict(use_cute_dsl_paged_mqa_logits=True), "use_cute_dsl_paged_mqa_logits"),
        (dict(enable_heuristic_topk=True, use_self_sampling_topk=False), "use_self_sampling_topk false"),
        (
            dict(
                enable_heuristic_topk=True, use_self_sampling_topk=False,
                use_cute_dsl_paged_mqa_logits=True, use_gvr_emission=True,
            ),
            "use_gvr_emission",
        ),
        (dict(indexer_max_chunk_size=16384), "indexer_max_chunk_size 16384"),
        (dict(skip_indexer_for_short_seqs=False), "skip_indexer_for_short_seqs false"),
        (dict(use_cute_dsl_topk=False), "use_cute_dsl_topk false"),
    ],
    ids=["bf16-staging", "cute-dsl-paged-logits", "temporal-hint-gvr", "gvr-emission", "indexer-chunk",
         "no-short-seq-skip", "cuda-radix-top-k"],
)  # fmt: skip
def test_csa2_options_the_targets_do_not_run_are_declined(
    mapping_kwargs, spec_config, fields, missed
):
    sparse = CSA2SparseAttentionConfig(**fields)
    config = ModelConfig(
        pretrained_config=_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        sparse_attention_config=sparse,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict=_experts(),
    )
    _declined(config, missed, lambda m: _llm_args(m))


@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
def test_the_default_indexer_chunk_size_is_taken(mapping_kwargs, spec_config):
    """``indexer_max_chunk_size`` 32768 is what the engine uses when it is left unset, and what the targets run."""
    config = ModelConfig(
        pretrained_config=_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        sparse_attention_config=CSA2SparseAttentionConfig(indexer_max_chunk_size=32768),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict=_experts(),
    )
    assert modeling_v2_resolve(config, _llm_args("require")) is not None


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, ok",
    [(_TEP8, _DSPARK, False), (_DEP2, None, True), (_DEP8, _DSPARK, True)],
    ids=["tep8", "dep2", "dep8"],
)
def test_a_q_split_threshold_is_declined_where_queries_could_split(mapping_kwargs, spec_config, ok):
    """``q_split_threshold`` splits prefill indexer queries across tensor-parallel ranks, never under attention DP:
    only tep8 could split, and it does not, so a non-default threshold is declined there alone."""
    config = ModelConfig(
        pretrained_config=_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        sparse_attention_config=CSA2SparseAttentionConfig(q_split_threshold=4096),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict=_experts(),
    )
    if ok:
        assert modeling_v2_resolve(config, _llm_args("require")) is not None
    else:
        _declined(config, "q_split_threshold 4096", lambda m: _llm_args(m))


@pytest.mark.parametrize("mapping_kwargs", [_TEP8, _DEP8, _DEP32], ids=["tep8", "dep8", "dep32"])
def test_confidence_scheduling_is_declined(mapping_kwargs):
    """Confidence scheduling captures verify graphs whose requests carry different numbers of draft tokens; the
    targets verify the same number for every request of a step."""
    spec = DSparkDecodingConfig(
        max_draft_len=5,
        enable_confidence_scheduling=True,
        confidence_cost_table_path="/tmp/costs.json",
    )
    config = _model_config(_config(), spec, **mapping_kwargs)
    _declined(config, "enable_confidence_scheduling", lambda m: _llm_args(m))


@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize("heuristic", [False, True], ids=["radix", "self-sampling-gvr"])
def test_both_exact_top_k_selections_are_taken(mapping_kwargs, spec_config, heuristic):
    """``enable_heuristic_topk`` picks the engine's exact radix top-k or its hint-free self-sampling GVR engine; the
    targets run the one asked for, so both route."""
    config = ModelConfig(
        pretrained_config=_config(),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        sparse_attention_config=CSA2SparseAttentionConfig(enable_heuristic_topk=heuristic),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict=_experts(),
    )
    assert modeling_v2_resolve(config, _llm_args("require")) is not None


@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "env, missed",
    [
        ({"FORCE_DETERMINISTIC": "1"}, "FORCE_DETERMINISTIC=1"),
        ({"FORCE_MHC_KERNEL_DETERMINISTIC": "1"}, "FORCE_MHC_KERNEL_DETERMINISTIC=1"),
        ({"TLLM_DISABLE_MPI": "1"}, "TLLM_DISABLE_MPI=1"),
    ],
    ids=["deterministic", "mhc-deterministic", "no-mpi"],
)
def test_environment_switches_the_targets_do_not_honor_are_declined(
    monkeypatch, mapping_kwargs, spec_config, env, missed
):
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _declined(config, missed, lambda m: _llm_args(m))


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, env, backend",
    [
        (_DEP2, None, {"TRTLLM_NVLINK_ONE_SIDED_A2A_SYNC_PROTOCOL": "fence"}, "AUTO"),
        (_DEP2, None, {"TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB": "512"}, "AUTO"),
        (_DEP8, _DSPARK, {"TRTLLM_NVLINK_ONE_SIDED_A2A_SENTINEL_MAX_TOKENS": "64"}, "TRTLLM"),
        (_DEP8, _DSPARK, {"MEGAMOE_COMBINE_FORMAT": "32e4m3xe8m0"}, "MEGAMOE_CUTEDSL"),
        (_DEP8, _DSPARK, {"MEGAMOE_UGPU_LAYOUT": "expert_half"}, "MEGAMOE_CUTEDSL"),
    ],
    ids=["dep2-sync-protocol", "dep2-workspace", "dep8-sentinel-limit", "dep8-megamoe-combine-format",
         "dep8-megamoe-ugpu-layout"],
)  # fmt: skip
def test_moe_communication_switches_the_targets_do_not_read_are_declined(
    monkeypatch, mapping_kwargs, spec_config, env, backend
):
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    ((name, value),) = env.items()
    monkeypatch.setenv(name, value)
    args = dict(moe_config={"backend": backend})
    _declined(config, name, lambda m: _llm_args(m, **args))


@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "fields, missed",
    [
        (dict(lora_config=object()), "LoRA"),
        (dict(torch_compile_config=object()), "torch_compile_config"),
        (dict(load_format="dummy"), "load_format dummy"),
        (dict(orchestrator_type="ray"), "orchestrator_type ray"),
        (dict(prefill_cuda_graph_backend="piecewise"), "piecewise"),
    ],
    ids=["lora", "torch-compile", "dummy-weights", "ray", "piecewise-prefill-graphs"],
)
def test_engine_arguments_the_targets_do_not_implement_are_declined(
    mapping_kwargs, spec_config, fields, missed
):
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    _declined(config, missed, lambda m: _with(_llm_args(m), **fields))


@pytest.mark.parametrize("mapping_kwargs", [_TEP8, _DEP8, _DEP32], ids=["tep8", "dep8", "dep32"])
def test_drafts_longer_than_the_engram_ring_are_declined(mapping_kwargs):
    """A step writes its 1 + drafts tokens into a 16-position look-back ring that the next step reads 3 positions back
    from: 13 drafts fit, 14 overwrite the look-back of a step that rejects them."""
    ok = _model_config(_config(), DSparkDecodingConfig(max_draft_len=13), **mapping_kwargs)
    assert modeling_v2_resolve(ok, _llm_args("require")) is not None
    config = _model_config(_config(), DSparkDecodingConfig(max_draft_len=14), **mapping_kwargs)
    _declined(config, "14 draft tokens per step", lambda m: _llm_args(m))


@pytest.mark.parametrize("mapping_kwargs, spec_config", _LAYOUTS, ids=_LAYOUT_IDS)
def test_a_checkpoint_without_candidate_layers_is_declined(mapping_kwargs, spec_config):
    """The targets' indexer reads the candidate source's candidates (`candidate_source_layer_id`); a checkpoint
    without one failed in the first attention layer's constructor instead of falling back."""
    config = _model_config(
        _config(dict(candidate_source_layer_id=None)), spec_config, **mapping_kwargs
    )
    _declined(config, "candidate_source_layer_id", lambda m: _llm_args(m))
