# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash target routing, driven by this checkpoint's config on CPU.

The target class is stubbed in ``sys.modules``: routing picks a class, and
these synthetic configs never construct it. Everything else -- the config
shape, the layer masks, the FP8 policy, the DSA geometry, the layouts and the
speculative config -- is the real one.
"""

import json
import os
import sys
from types import ModuleType

import pytest
import torch
from _hyv4_flash_config import config_dict  # noqa: E402

from tensorrt_llm._torch.configs import HYV4FlashConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import modeling_v2_resolve, registered_routes
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation import routing
from tensorrt_llm.llmapi.llm_args import (
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET_PACKAGE = (
    "tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation."
    "hf_003107f_orig__sm_103__unified"
)
_TARGET_MODULE = f"{_TARGET_PACKAGE}.modeling"
_TARGET = (
    "models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.modeling."
    "ModelingV2Hyv4FlashForConditionalGeneration"
)

#: The members, as the engine args lay them out:
#: (world size, attention DP, MoE TP, MoE EP). ``tep2`` is the disaggregated
#: deployment's context engine, two ranks with the experts split two ways.
_MEMBERS = {
    "tep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
    "tp4": dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
    "tep8": dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
}


@pytest.fixture(autouse=True)
def _on_sm103_with_a_stubbed_target(monkeypatch):
    """Route as a GB300, wherever the test runs, without the target's code.

    Registering the package as well as the module keeps the relative import in
    ``route`` resolving even while the target directory does not exist yet.
    """
    package = ModuleType(_TARGET_PACKAGE)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, _TARGET_PACKAGE, package)
    module = ModuleType(_TARGET_MODULE)
    module.ModelingV2Hyv4FlashForConditionalGeneration = type(
        "ModelingV2Hyv4FlashForConditionalGeneration", (), {"__module__": _TARGET_MODULE}
    )
    monkeypatch.setitem(sys.modules, _TARGET_MODULE, module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    return TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2=mode, **kwargs)


def _hyv4_config(**text_overrides) -> HYV4FlashConfig:
    """This checkpoint's composite config, with ``text_overrides`` on its decoder."""
    checkpoint = config_dict()
    checkpoint["text_config"].update(text_overrides)
    return HYV4FlashConfig.from_dict(checkpoint)


def _model_config(
    pretrained_config,
    spec_config=None,
    quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
    kv_cache_quant_algo=None,
    **mapping_kwargs,
) -> ModelConfig:
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=quant_algo, kv_cache_quant_algo=kv_cache_quant_algo),
    )


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


def _dflash_as_served() -> DFlashDecodingConfig:
    return DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=5, mask_token_id=120009)


def _dflash_geometry_from_checkpoint() -> DFlashDecodingConfig:
    return DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=5)


def _dflash_embedded_in_the_target() -> DFlashDecodingConfig:
    """The config as the engine leaves it: ``speculative_model`` is the model the args carry."""
    return DFlashDecodingConfig(
        decoding_type="DFlash",
        max_draft_len=5,
        mask_token_id=120009,
        speculative_model=_DUMMY_MODEL,
    )


_ACCEPTED_SPECS = {
    "no-spec": lambda: None,
    "dflash-as-served": _dflash_as_served,
    "dflash-embedded-in-target": _dflash_embedded_in_the_target,
    "dflash-geometry-from-checkpoint": _dflash_geometry_from_checkpoint,
}


def test_the_published_architecture_is_routed_here() -> None:
    """``models/__init__.py`` imports the family's routing, which registers the name."""
    assert registered_routes()["HYV4FlashForConditionalGeneration"] is routing.route


@pytest.mark.parametrize("member", list(_MEMBERS), ids=list(_MEMBERS))
@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize("spec_name", list(_ACCEPTED_SPECS), ids=list(_ACCEPTED_SPECS))
def test_the_served_members_route_to_the_target(member, mode, spec_name) -> None:
    """tep2, tp4, tep4, dep4 and tep8 all build the one target, with or without DFly2."""
    config = _model_config(
        _hyv4_config(), spec_config=_ACCEPTED_SPECS[spec_name](), **_MEMBERS[member]
    )
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _TARGET


@pytest.mark.parametrize("checkpoint_algo", [None, QuantAlgo.FP8], ids=["bf16", "fp8"])
@pytest.mark.parametrize("dtype", ["auto", "fp8"])
def test_bf16_and_fp8_kv_caches_route_to_the_target(checkpoint_algo, dtype) -> None:
    """The core member and its one FP8-KV variant both land on the target."""
    config = _model_config(_hyv4_config(), kv_cache_quant_algo=checkpoint_algo, **_MEMBERS["tp4"])
    args = _llm_args("require", kv_cache_config=KvCacheConfig(dtype=dtype))
    assert _qualified(modeling_v2_resolve(config, args)) == _TARGET


def test_layout_accepts_only_the_served_members() -> None:
    for member, mapping_kwargs in _MEMBERS.items():
        assert routing.layout(Mapping(**mapping_kwargs)) == member
    # The context engine's two ranks with the experts split by intermediate is
    # not the layout it runs: only tp2 *with* the two-way expert split is served.
    assert routing.layout(Mapping(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1)) is None
    # An ADP layout that splits the LM head over TP is another forward.
    assert (
        routing.layout(
            Mapping(
                world_size=4,
                tp_size=4,
                moe_tp_size=1,
                moe_ep_size=4,
                enable_attention_dp=True,
                enable_lm_head_tp_in_adp=True,
            )
        )
        is None
    )


@pytest.mark.parametrize(
    "config_factory, mapping_kwargs, missed",
    [
        (
            lambda: _hyv4_config(hidden_size=4096),
            _MEMBERS["tp4"],
            r"config shape \(42, 4096, 512, 16, 1536, 120832\)",
        ),
        (
            lambda: _hyv4_config(index_topk=512),
            _MEMBERS["tp4"],
            r"'index_topk': 512",
        ),
        (
            lambda: _hyv4_config(sliding_window=1024),
            _MEMBERS["tp4"],
            r"'sliding_window': 1024",
        ),
        (
            lambda: _hyv4_config(),
            dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1),
            r"world_size=2\b",
        ),
        (
            lambda: _hyv4_config(),
            dict(
                world_size=2,
                tp_size=2,
                moe_tp_size=1,
                moe_ep_size=2,
                enable_attention_dp=True,
            ),
            r"attention_dp=True",
        ),
        (
            lambda: _hyv4_config(),
            dict(
                world_size=16,
                tp_size=16,
                moe_tp_size=16,
                moe_ep_size=1,
                gpus_per_node=16,
            ),
            r"world_size=16\b",
        ),
    ],
    ids=[
        "other-shape",
        "other-index-topk",
        "other-window",
        "tp2-moe-tp2",
        "attention-dp-at-tep2",
        "tp16",
    ],
)
def test_near_miss_configs_do_not_match(config_factory, mapping_kwargs, missed) -> None:
    config = _model_config(config_factory(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_the_sparse_layer_pattern_is_part_of_the_identity() -> None:
    """A config with the same shape but a different sparse/windowed split is not this model."""
    checkpoint = config_dict()
    layer_types = checkpoint["text_config"]["layer_types"]
    layer_types[0], layer_types[2] = layer_types[2], layer_types[0]
    config = _model_config(HYV4FlashConfig.from_dict(checkpoint), **_MEMBERS["tp4"])

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"sparse layers \(1, 2, 5"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "mapping_kwargs, missed",
    [
        (
            dict(world_size=8, tp_size=4, pp_size=2, moe_tp_size=4, moe_ep_size=1),
            r"pp=2\b",
        ),
        (
            dict(world_size=8, tp_size=4, cp_size=2, moe_tp_size=4, moe_ep_size=1),
            r"cp=2\b",
        ),
        (
            dict(world_size=4, tp_size=4, moe_tp_size=2, moe_ep_size=2),
            r"moe_ep=2\b",
        ),
        (
            dict(
                world_size=4,
                tp_size=4,
                moe_tp_size=4,
                moe_ep_size=1,
                enable_attention_dp=True,
            ),
            r"attention_dp=True",
        ),
    ],
    ids=["pipeline", "context", "half-expert-parallel", "attention-dp-at-tp4"],
)
def test_other_parallelisms_do_not_match(mapping_kwargs, missed) -> None:
    config = _model_config(_hyv4_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_tep8_across_nodes_needs_mnnvl(monkeypatch) -> None:
    """tep8 is one engine over two nodes, which the attention allreduce must reach."""
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    config = _model_config(_hyv4_config(), **dict(_MEMBERS["tep8"], gpus_per_node=4))
    assert config.mapping.is_multi_node()

    monkeypatch.setattr(MNNVLAllReduce, "is_mnnvl", staticmethod(lambda *a, **k: False))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="tp_reaches_across_nodes=False"):
        modeling_v2_resolve(config, _llm_args("require"))

    monkeypatch.setattr(MNNVLAllReduce, "is_mnnvl", staticmethod(lambda *a, **k: True))
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


def test_another_capability_does_not_match(monkeypatch) -> None:
    """The target's kernels are sm103; any other GPU falls back to the built-in model."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))
    config = _model_config(_hyv4_config(), **_MEMBERS["tp4"])

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 7\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("quant_algo", [None, QuantAlgo.NVFP4], ids=["unquantized", "nvfp4"])
def test_another_weight_quantization_does_not_match(quant_algo) -> None:
    """The target reads the checkpoint's block-FP8 experts; other formats are other code."""
    config = _model_config(_hyv4_config(), quant_algo=quant_algo, **_MEMBERS["tp4"])

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="quantization"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "checkpoint_algo, dtype",
    [
        (None, "nvfp4"),
        (QuantAlgo.FP8, "nvfp4"),
        (None, "fp8_ds_mla"),
        (QuantAlgo.FP8, "fp8_ds_mla"),
    ],
    ids=["nvfp4", "fp8-to-nvfp4", "packed-sparse-mla", "fp8-to-packed-sparse-mla"],
)
def test_another_kv_cache_does_not_match(checkpoint_algo, dtype) -> None:
    config = _model_config(_hyv4_config(), kv_cache_quant_algo=checkpoint_algo, **_MEMBERS["tp4"])
    args = dict(kv_cache_config=KvCacheConfig(dtype=dtype))

    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="KV cache"):
        modeling_v2_resolve(config, _llm_args("require", **args))


_REFUSED_SPECS = {
    "mtp": lambda: MTPDecodingConfig(max_draft_len=3),
    "eagle3": lambda: Eagle3DecodingConfig(max_draft_len=3, speculative_model="/tmp/draft_model"),
    "dspark": lambda: DSparkDecodingConfig(max_draft_len=5),
    "three-drafts": lambda: DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=3),
    "variant-1": lambda: DFlashDecodingConfig(
        decoding_type="DFlash", max_draft_len=5, spec_dec_variant=1
    ),
    "separate-draft": lambda: DFlashDecodingConfig(
        decoding_type="DFlash", max_draft_len=5, speculative_model="/tmp/draft_model"
    ),
}


@pytest.mark.parametrize("spec_name", list(_REFUSED_SPECS), ids=list(_REFUSED_SPECS))
def test_other_speculation_does_not_match(spec_name) -> None:
    """Only the checkpoint's own DFly2, drafting five tokens from its own layers."""
    config = _model_config(
        _hyv4_config(), spec_config=_REFUSED_SPECS[spec_name](), **_MEMBERS["tep4"]
    )

    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="speculative decoding"):
        modeling_v2_resolve(config, _llm_args("require"))


def _checkpoint_dir(tmp_path, name: str, config: dict) -> str:
    """A directory holding ``config`` as the ``config.json`` of a checkpoint in it."""
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


def test_a_dflash_draft_that_names_the_target_checkpoint_routes(tmp_path) -> None:
    """Embeddedness is classified, not presence: only another checkpoint is a separate drafter.

    The loader swaps a hub repo id for the snapshot it downloaded, so the path a
    DFlash config names need not equal the model's byte for byte; and a
    checkpoint declaring the ``dfly_*`` draft geometry is the target checkpoint
    under another spelling.
    """
    spelled_otherwise = _dflash_embedded_in_the_target()
    spelled_otherwise.speculative_model = f"{_DUMMY_MODEL}{os.sep}"
    at_a_copy = _dflash_embedded_in_the_target()
    at_a_copy.speculative_model = _checkpoint_dir(tmp_path, "hf-003107f_orig", config_dict())

    for spec in (spelled_otherwise, at_a_copy):
        config = _model_config(_hyv4_config(), spec_config=spec, **_MEMBERS["tep4"])
        assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _TARGET


def test_a_separate_draft_checkpoint_is_still_refused(tmp_path) -> None:
    """A path naming no checkpoint of this family is a separate drafter, not the embedded one."""
    drafter = _checkpoint_dir(
        tmp_path,
        "draft_model",
        {
            "architectures": ["DFlashDraftModel"],
            "dflash_config": {"target_layer_ids": [25, 29, 33, 37, 41]},
        },
    )
    separate = _REFUSED_SPECS["separate-draft"]()
    separate.speculative_model = drafter
    embedded = _dflash_embedded_in_the_target()

    assert routing.draft_is_embedded_in_target(embedded, _DUMMY_MODEL)
    assert routing.speculation_refused(embedded, _DUMMY_MODEL) == []
    assert routing.speculation_refused(separate, _DUMMY_MODEL) == [
        "with a separate draft checkpoint"
    ]


def test_speculation_refused_names_a_draft_tree() -> None:
    """The one refusal the public config cannot express: a tree wider than its depth."""
    assert routing.speculation_refused(None) == []
    assert routing.speculation_refused(_dflash_as_served()) == []

    tree = DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=5)
    tree.max_total_draft_tokens = 7
    assert routing.speculation_refused(tree) == ["with a draft tree"]


def test_a_refusal_does_not_import_the_target(monkeypatch) -> None:
    """``route`` imports the target only on the branch that returns it."""
    monkeypatch.setitem(sys.modules, _TARGET_MODULE, None)
    config = _model_config(_hyv4_config(hidden_size=4096), **_MEMBERS["tp4"])

    with pytest.raises(ValueError, match="config shape"):
        modeling_v2_resolve(config, _llm_args("require"))
