# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of `InklingDims`: both config spellings give one geometry, and bad inputs fail."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.inkling2.config import ExpertFormat, InklingDims

pytestmark = pytest.mark.cpu_only

GLOBAL = tuple(range(5, 48, 6))
_COMMON = dict(
    num_hidden_layers=48,
    hidden_size=4608,
    vocab_size=201024,
    unpadded_vocab_size=200058,
    num_attention_heads=48,
    num_key_value_heads=8,
    head_dim=128,
    d_rel=16,
    rel_extent=1024,
    n_layers_per_attn_res_block=6,
    n_routed_experts=384,
    num_experts_per_tok=8,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    final_logit_softcapping=30.0,
    qk_l2_clip_threshold=22.627416997969522,
    rel_l2_clip_threshold=8.0,
    log_scaling_alpha=0.1,
    log_scaling_n_floor=8192,
    rms_norm_eps=1e-6,
)


def _trtllm_style(**overrides) -> SimpleNamespace:
    """The `Inkling2TextConfig` spelling: `local_layer_ids` and the checkpoint's field names."""
    fields = dict(
        _COMMON,
        local_layer_ids=[i for i in range(48) if i not in GLOBAL],
        sliding_window_size=1024,
        sconv_kernel_size=4,
        intermediate_size=2304,
        route_scale=8.0,
    )
    return SimpleNamespace(**{**fields, **overrides})


def _native_style() -> SimpleNamespace:
    """The Transformers spelling: `layer_types` (`hybrid` / `hybrid_sliding`) and its aliases."""
    return SimpleNamespace(
        **_COMMON,
        layer_types=["hybrid" if i in GLOBAL else "hybrid_sliding" for i in range(48)],
        sliding_window=1024,
        conv_kernel_size=4,
        moe_intermediate_size=2304,
        routed_scaling_factor=8.0,
    )


def _groups(**layers_by_format):
    return {
        "model": {
            "parameter_groups": [
                {
                    "compute_dtype": fmt,
                    "tensors": [f"model.llm.layers.{i}.mlp.experts.w13_weight" for i in layers],
                }
                for fmt, layers in layers_by_format.items()
            ]
        }
    }


def test_both_config_spellings_give_one_geometry():
    quant = _groups(nvfp4=range(40), mxfp8=range(40, 48))
    native = InklingDims.from_config(_native_style(), quant)
    assert native == InklingDims.from_config(_trtllm_style(), quant)
    assert native.global_layers == GLOBAL
    assert (native.conv_kernel_size, native.expert_intermediate_size, native.route_scale) == (
        4,
        2304,
        8.0,
    )
    assert native.expert_formats == (ExpertFormat.NVFP4,) * 40 + (ExpertFormat.MXFP8,) * 8


def test_layers_no_parameter_group_names_keep_bf16_experts():
    dims = InklingDims.from_config(_trtllm_style(), _groups(nvfp4=[0]))
    assert dims.expert_formats[:2] == (ExpertFormat.NVFP4, ExpertFormat.BF16)


@pytest.mark.parametrize(
    "quant, match",
    [
        (_groups(nvfp4=[3], mxfp8=[3]), "layer 3 two expert formats"),
        (_groups(nvfp4=[48]), r"layers \[48\], outside the 48 decoder layers"),
    ],
    ids=["two_formats_on_one_layer", "layer_out_of_range"],
)
def test_malformed_quant_config_is_refused(quant, match):
    with pytest.raises(ValueError, match=match):
        InklingDims.from_config(_trtllm_style(), quant)


def test_sliding_layers_with_another_attention_shape_are_refused():
    with pytest.raises(ValueError, match="different attention shape"):
        InklingDims.from_config(_trtllm_style(swa_num_key_value_heads=4))
