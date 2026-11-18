# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The recurrent state a model declares, against the geometry the cache managers use."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import resolve_recurrent_state_spec
from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import (
    PLE_CONV_STATE,
    PLE_NGRAM_CONTEXT,
    _qwen4_exp_ple_state_bytes_per_rank,
)
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import (
    extract_mamba_kv_cache_params,
    mamba_effective_tp_size,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole
from tensorrt_llm._torch.recurrent_state_spec import (
    LayerKind,
    LayerStateSpec,
    PlaneShard,
    StatePlaneSpec,
    recurrent_state_spec_from_mamba_params,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Family -> (architecture, model_type) of its four-layer text config.
_TEXT_MODELS = {
    "nemotron": ("NemotronHForCausalLM", "nemotron_h"),
    "gdn": ("Qwen3_5MoeForCausalLM", "qwen3_5_moe"),
    "qwen4": ("Qwen4ExpForCausalLM", "qwen4_exp_text"),
    "kimi": ("KimiLinearForCausalLM", "kimi_linear"),
    "glm": ("Glm5NextForCausalLM", "glm5_next_text"),
}

# Every architecture name registered for a text model of a recurrent family.
_TEXT_ARCHITECTURES = [
    ("nemotron", "NemotronHForCausalLM"),
    ("nemotron", "NemotronHPuzzleForCausalLM"),
    ("gdn", "Qwen3NextForCausalLM"),
    ("gdn", "Qwen3_5MoeForCausalLM"),
    ("gdn", "Qwen3_5ForCausalLM"),
    ("qwen4", "Qwen4ExpForCausalLM"),
    ("kimi", "KimiLinearForCausalLM"),
    ("glm", "Glm5NextForCausalLM"),
]

# Wrapper architecture -> (family, architecture of the language model, field that holds its
# config, model_type of the composite config).
_WRAPPERS = {
    "NemotronH_Nano_VL_V2": (
        "nemotron",
        "NemotronHForCausalLM",
        "llm_config",
        "NemotronH_Nano_VL_V2",
    ),
    "NemotronH_Nano_Omni_Reasoning_V3": (
        "nemotron",
        "NemotronHForCausalLM",
        "llm_config",
        "NemotronH_Nano_Omni_Reasoning_V3",
    ),
    "NemotronH_Omni_Reasoning_V3": (
        "nemotron",
        "NemotronHForCausalLM",
        "llm_config",
        "NemotronH_Omni_Reasoning_V3",
    ),
    "Qwen3_5MoeForConditionalGeneration": (
        "gdn",
        "Qwen3_5MoeForCausalLM",
        "text_config",
        "qwen3_5_moe",
    ),
    "Qwen3_5ForConditionalGeneration": ("gdn", "Qwen3_5ForCausalLM", "text_config", "qwen3_5"),
    "MiniCPMV4_6ForConditionalGeneration": (
        "gdn",
        "Qwen3_5ForCausalLM",
        "text_config",
        "minicpmv4_6",
    ),
    "Qwen4ExpForConditionalGeneration": (
        "qwen4",
        "Qwen4ExpForCausalLM",
        "text_config",
        "qwen4_exp",
    ),
    "KimiK3ForConditionalGeneration": ("kimi", "KimiLinearForCausalLM", "text_config", "kimi_k3"),
    "Glm5NextForConditionalGeneration": (
        "glm",
        "Glm5NextForCausalLM",
        "text_config",
        "glm5_next",
    ),
}

# Family -> (conv dtype, SSM dtype, {ranks sharing a request: (conv shape, conv sections, SSM
# shape)}) of the state buffers the V2 managers allocate for `_text_config(family)`.
_MANAGER_LAYOUT = {
    "nemotron": (
        torch.bfloat16,
        torch.bfloat16,
        {1: ((64, 3), (32, 16, 16), (4, 8, 16)), 2: ((32, 3), (16, 8, 8), (2, 8, 16))},
    ),
    "gdn": (
        torch.bfloat16,
        torch.bfloat16,
        {1: ((48, 3), (8, 8, 32), (4, 8, 8)), 2: ((24, 3), (4, 4, 16), (2, 8, 8))},
    ),
    "qwen4": (
        torch.bfloat16,
        torch.bfloat16,
        {1: ((48, 3), (8, 8, 32), (4, 8, 8)), 2: ((24, 3), (4, 4, 16), (2, 8, 8))},
    ),
    "kimi": (
        torch.bfloat16,
        torch.float32,
        {1: ((96, 3), (32, 32, 32), (4, 8, 8)), 2: ((48, 3), (16, 16, 16), (2, 8, 8))},
    ),
    "glm": (
        torch.bfloat16,
        torch.float32,
        {1: ((96, 3), (32, 32, 32), (4, 8, 8)), 2: ((48, 3), (16, 16, 16), (2, 8, 8))},
    ),
}


def _text_config(
    family: str, *, architecture: str | None = None, recurrent: bool = True
) -> SimpleNamespace:
    """A four-layer text config: layers 0 and 2 are recurrent unless ``recurrent`` is False."""
    default_architecture, model_type = _TEXT_MODELS[family]
    config = SimpleNamespace(
        architectures=[architecture or default_architecture],
        model_type=model_type,
        vocab_size=163840,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
    )
    if family == "nemotron":
        config.hybrid_override_pattern = "M*M*" if recurrent else "****"
        config.ssm_state_size, config.conv_kernel = 16, 4
        config.mamba_num_heads, config.n_groups, config.mamba_head_dim = 4, 1, 8
    elif family in ("gdn", "qwen4"):
        config.layer_types = (
            ["linear_attention", "full_attention"] * 2 if recurrent else ["full_attention"] * 4
        )
        config.linear_key_head_dim, config.linear_conv_kernel_dim = 8, 4
        config.linear_num_value_heads, config.linear_num_key_heads = 4, 1
        config.linear_value_head_dim = 8
        if family == "qwen4":
            config.ple_layer_ids = [1] if recurrent else []
            config.ple_conv_kernel_size, config.ngram_size, config.hc_count = 2, 3, 1
    else:
        config.kv_lora_rank, config.qk_rope_head_dim, config.index_head_dim = 32, 8, 16
        config.linear_attn_config = {
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3] if recurrent else [],
            "full_attn_layers": [2, 4] if recurrent else [1, 2, 3, 4],
        }
        config.layer_types = (
            ["linear_attention", "deepseek_sparse_attention"] * 2
            if recurrent
            else ["deepseek_sparse_attention"] * 4
        )
    return config


def _wrapper_config(architecture: str, *, mirrored: bool) -> SimpleNamespace:
    """The composite config of a VLM wrapper around ``_text_config``.

    ``mirrored`` copies the text fields the composite config lacks onto it, as config loading does
    for a ``text_config``.
    """
    family, text_architecture, field, model_type = _WRAPPERS[architecture]
    text_config = _text_config(family, architecture=text_architecture)
    config = SimpleNamespace(
        architectures=[architecture],
        model_type=model_type,
        dtype=text_config.dtype,
        torch_dtype=text_config.torch_dtype,
        **{field: text_config},
    )
    if mirrored:
        for name, value in vars(text_config).items():
            if not hasattr(config, name):
                setattr(config, name, value)
    return config


def _base_planes(layer: LayerStateSpec) -> tuple[StatePlaneSpec, StatePlaneSpec]:
    planes = {plane.role: plane for plane in layer.planes}
    return planes[MambaRole.CONV_STATE], planes[MambaRole.SSM_STATE]


def _num_bytes(plane: StatePlaneSpec, num_shards: int) -> int:
    return math.prod(plane.local_shape(num_shards)) * plane.dtype.itemsize


def test_plane_splits_only_its_sharded_axis():
    conv = StatePlaneSpec(
        role=MambaRole.CONV_STATE,
        shape=(48, 3),
        dtype=torch.bfloat16,
        shard=PlaneShard(axis=0, sections=(32, 8, 8)),
    )
    assert conv.local_shape(4) == (12, 3)
    assert conv.local_sections(4) == (8, 2, 2)
    with pytest.raises(ValueError, match="cannot split evenly"):
        conv.local_shape(5)

    # Each section is split on its own, so sections that are not multiples of the rank count
    # leave part of the local axis outside every section.
    uneven = StatePlaneSpec(
        role=MambaRole.CONV_STATE,
        shape=(16, 3),
        dtype=torch.bfloat16,
        shard=PlaneShard(axis=0, sections=(5, 5, 6)),
    )
    assert uneven.local_shape(2) == (8, 3)
    assert uneven.local_sections(2) == (2, 2, 3)

    whole = StatePlaneSpec(role=PLE_NGRAM_CONTEXT, shape=(2,), dtype=torch.int64)
    assert whole.local_shape(4) == (2,)
    assert whole.local_sections(4) == ()


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: StatePlaneSpec(
                MambaRole.CONV_STATE, (48, 3), torch.bfloat16, PlaneShard(axis=0, sections=(32, 8))
            ),
            "do not add up",
        ),
        (
            lambda: StatePlaneSpec(
                MambaRole.CONV_STATE, (48, 3), torch.bfloat16, PlaneShard(sections=(48,))
            ),
            "no sharded axis",
        ),
        (
            lambda: StatePlaneSpec(
                MambaRole.CONV_STATE, (48, 3), torch.bfloat16, PlaneShard(axis=2)
            ),
            "outside its shape",
        ),
        (
            lambda: LayerStateSpec(
                LayerKind.ATTENTION, (StatePlaneSpec(MambaRole.SSM_STATE, (4,), torch.bfloat16),)
            ),
            "cannot carry recurrent state",
        ),
        (
            lambda: LayerStateSpec(
                LayerKind.RECURRENT,
                (StatePlaneSpec(MambaRole.SSM_STATE, (4,), torch.bfloat16),) * 2,
            ),
            "more than once",
        ),
        (
            lambda: recurrent_state_spec_from_mamba_params(
                extract_mamba_kv_cache_params(_text_config("nemotron")), conv_sections="b_c_x"
            ),
            "Unsupported convolution section order",
        ),
    ],
)
def test_malformed_declaration_is_rejected(build, message):
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("family, architecture", _TEXT_ARCHITECTURES)
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("ssm_cache_dtype", [None, "float32"])
def test_declaration_matches_cache_params_and_state_budget(
    family, architecture, recurrent, ssm_cache_dtype
):
    config = _text_config(family, architecture=architecture, recurrent=recurrent)
    quant_config = QuantConfig(mamba_ssm_cache_dtype=ssm_cache_dtype)
    params = extract_mamba_kv_cache_params(config, quant_config=quant_config)

    spec = resolve_recurrent_state_spec(config, quant_config=quant_config)

    assert list(spec.recurrent_layer_mask) == params.mamba_layer_mask
    assert list(spec.attention_layer_mask) == params.target_full_attention_layer_mask
    assert any(spec.recurrent_layer_mask) == recurrent
    for layer in spec.layers:
        if layer.kind is not LayerKind.RECURRENT:
            assert layer.planes == ()
            continue
        conv, ssm = _base_planes(layer)
        assert conv.dtype == params.dtype
        assert ssm.dtype == params.mamba_ssm_cache_dtype
        for mapping in (
            Mapping(),
            Mapping(world_size=2, tp_size=2),
            Mapping(world_size=4, tp_size=4),
            Mapping(world_size=2, tp_size=2, enable_attention_dp=True),
        ):
            num_shards = mamba_effective_tp_size(mapping)
            assert _num_bytes(conv, num_shards) + _num_bytes(
                ssm, num_shards
            ) == params.get_states_bytes_per_layer(mapping)


@pytest.mark.parametrize("family", sorted(_MANAGER_LAYOUT))
@pytest.mark.parametrize("num_shards", [1, 2])
def test_declared_geometry_matches_manager_layout(family, num_shards):
    conv_dtype, ssm_dtype, layouts = _MANAGER_LAYOUT[family]
    conv_shape, conv_sections, ssm_shape = layouts[num_shards]

    spec = resolve_recurrent_state_spec(_text_config(family), quant_config=QuantConfig())

    assert [layer.kind for layer in spec.layers] == [
        LayerKind.RECURRENT,
        LayerKind.ATTENTION,
        LayerKind.RECURRENT,
        LayerKind.ATTENTION,
    ]
    for layer_idx in (0, 2):
        conv, ssm = _base_planes(spec.layers[layer_idx])
        assert (conv.local_shape(num_shards), conv.dtype) == (conv_shape, conv_dtype)
        assert conv.local_sections(num_shards) == conv_sections
        assert (ssm.local_shape(num_shards), ssm.dtype) == (ssm_shape, ssm_dtype)


def test_nemotron_layers_without_cache_are_declared_as_none():
    config = _text_config("nemotron")
    config.hybrid_override_pattern = "M-*E"

    spec = resolve_recurrent_state_spec(config)

    assert [layer.kind for layer in spec.layers] == [
        LayerKind.RECURRENT,
        LayerKind.NONE,
        LayerKind.ATTENTION,
        LayerKind.NONE,
    ]


def test_qwen4_exp_declares_ple_state_on_its_ple_layer():
    config = _text_config("qwen4")
    params = extract_mamba_kv_cache_params(config, quant_config=QuantConfig())

    spec = resolve_recurrent_state_spec(config, quant_config=QuantConfig())

    assert [plane.role for plane in spec.layers[2].planes] == [
        MambaRole.CONV_STATE,
        MambaRole.SSM_STATE,
    ]
    ple = {plane.role: plane for plane in spec.layers[0].planes[2:]}
    assert (ple[PLE_CONV_STATE].shape, ple[PLE_CONV_STATE].dtype) == ((64, 3), torch.bfloat16)
    assert (ple[PLE_NGRAM_CONTEXT].shape, ple[PLE_NGRAM_CONTEXT].dtype) == ((2,), torch.int64)
    assert all(plane.local_shape(2) == plane.shape for plane in ple.values())
    assert sum(_num_bytes(plane, 1) for plane in ple.values()) == (
        _qwen4_exp_ple_state_bytes_per_rank(
            SimpleNamespace(pretrained_config=config),
            params,
            Mapping(),
            spec_config=None,
            use_separate_draft_kv_cache=False,
        )
    )


@pytest.mark.parametrize(
    "architecture, mirrored",
    [
        (architecture, mirrored)
        for architecture, (_, _, field, _) in sorted(_WRAPPERS.items())
        for mirrored in ((False, True) if field == "text_config" else (False,))
    ],
)
def test_wrapper_declares_the_state_of_its_language_model(architecture, mirrored):
    config = _wrapper_config(architecture, mirrored=mirrored)
    text_config = getattr(config, _WRAPPERS[architecture][2])

    spec = resolve_recurrent_state_spec(config, quant_config=QuantConfig())

    assert spec is not None
    assert spec == resolve_recurrent_state_spec(text_config, quant_config=QuantConfig())


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(architectures=["LlamaForCausalLM"], model_type="llama"),
        SimpleNamespace(architectures=["NoSuchArchitectureForCausalLM"]),
        SimpleNamespace(architectures=None, hybrid_override_pattern="M*M*"),
        SimpleNamespace(),
    ],
    ids=["no_recurrent_state", "unregistered", "architectures_none", "no_architectures"],
)
def test_no_declaration_without_a_recurrent_model_class(config):
    assert resolve_recurrent_state_spec(config) is None


@skip_no_cuda
@pytest.mark.parametrize("family", sorted(_TEXT_MODELS))
@pytest.mark.parametrize(
    "mapping_kwargs", [{}, {"world_size": 2, "tp_size": 2}], ids=["tp1", "tp2"]
)
def test_manager_state_buffers_match_declaration(monkeypatch, family, mapping_kwargs):
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    config = _text_config(family)
    mapping = Mapping(**mapping_kwargs)
    model_config = ModelConfig(
        pretrained_config=config, mapping=mapping, quant_config=QuantConfig()
    )
    kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True, enable_block_reuse=False, max_tokens=256, tokens_per_block=64
    )
    spec = resolve_recurrent_state_spec(config, quant_config=model_config.quant_config)
    num_shards = mamba_effective_tp_size(mapping)

    manager = _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=get_kv_cache_manager_cls(model_config, kv_cache_config),
        mapping=mapping,
        kv_cache_config=kv_cache_config,
        tokens_per_block=kv_cache_config.tokens_per_block,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=model_config.sparse_attention_config,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        is_draft=False,
    )
    try:
        recurrent_layers = [
            layer_idx
            for layer_idx, is_recurrent in enumerate(spec.recurrent_layer_mask)
            if is_recurrent
        ]
        assert len(manager.all_conv_states) == len(manager.all_ssm_states) == len(recurrent_layers)
        for layer_idx, conv_state, ssm_state in zip(
            recurrent_layers, manager.all_conv_states, manager.all_ssm_states, strict=True
        ):
            conv, ssm = _base_planes(spec.layers[layer_idx])
            assert (tuple(conv_state.shape[1:]), conv_state.dtype) == (
                conv.local_shape(num_shards),
                conv.dtype,
            )
            assert (tuple(ssm_state.shape[1:]), ssm_state.dtype) == (
                ssm.local_shape(num_shards),
                ssm.dtype,
            )
            assert tuple(manager.conv_section_dims) == conv.local_sections(num_shards)
        if family == "qwen4":
            ple = {plane.role: plane for plane in spec.layers[0].planes[2:]}
            ple_conv_state, ple_ngram_context = manager.ple_layer_cache(0)
            assert (tuple(ple_conv_state.shape[1:]), ple_conv_state.dtype) == (
                ple[PLE_CONV_STATE].local_shape(num_shards),
                ple[PLE_CONV_STATE].dtype,
            )
            assert (tuple(ple_ngram_context.shape[1:]), ple_ngram_context.dtype) == (
                ple[PLE_NGRAM_CONTEXT].local_shape(num_shards),
                ple[PLE_NGRAM_CONTEXT].dtype,
            )
    finally:
        manager.shutdown()
