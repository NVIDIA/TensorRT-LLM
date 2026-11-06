# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the BFL <-> Diffusers FLUX.3 tensor name bijection."""

from typing import Any, Callable, Optional

import pytest
import torch
from flux3_test_utils import KLEIN_BASE_CONFIG, KLEIN_CONFIG, diffusers_shapes, tiny_config

from tensorrt_llm._torch.visual_gen.models.flux3.bfl_keys import (
    bfl_to_diffusers_key,
    bfl_to_diffusers_module,
    bfl_to_diffusers_state_dict,
    diffusers_to_bfl_key,
    diffusers_to_bfl_module,
    diffusers_to_bfl_state_dict,
    swap_norm_out_halves,
)

pytestmark = pytest.mark.cpu_only

BFL_BLOCK_LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
VARIANT_SCALES = {
    "nvfp4": ("weight_scale", "weight_scale_2", "input_scale"),
    "fp8r": ("weight_scale",),
}


def bfl_blocks(config: dict[str, Any]) -> list[str]:
    layers, single_layers = config["num_layers"], config["num_single_layers"]
    return [
        *(
            f"content_mode_blocks.{mode}.{i}"
            for mode in ("image", "image_cond")
            for i in range(layers)
        ),
        *(f"txt_mode_blocks.{i}" for i in range(layers)),
        *(f"single_blocks.{i}" for i in range(single_layers)),
    ]


def bfl_names(config: dict[str, Any]) -> list[str]:
    """BF16 tensor names of a BFL-native checkpoint, written from BFL's module layout."""
    names = [
        "txt_in.weight",
        "emb_in.image.weight",
        "emb_in.image_cond.weight",
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
        "final_layer.image.adaLN_modulation.1.weight",
        "final_layer.image.linear.weight",
    ]
    if config.get("flow_map", False):
        names += ["ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight"]
    names += [
        f"{stage}_stream_modulations.{mode}.lin.weight"
        for stage in ("early", "single")
        for mode in ("image", "image_cond", "txt")
    ]
    for block in bfl_blocks(config):
        names += [f"{block}.{linear}.weight" for linear in BFL_BLOCK_LINEARS]
        names += [f"{block}.norm.query_norm.scale", f"{block}.norm.key_norm.scale"]
    return names


def quantized_modules(config: dict[str, Any], variant: str) -> list[str]:
    """BFL modules the released variants quantize; the rest stay BF16 per hf_quant_config.json."""
    layers, single_layers = config["num_layers"], config["num_single_layers"]
    if variant == "nvfp4":
        blocks = [
            *(f"content_mode_blocks.image.{i}" for i in range(1, layers)),
            *(f"content_mode_blocks.image_cond.{i}" for i in range(layers)),
            *(f"single_blocks.{i}" for i in range(single_layers - 4)),
        ]
        extra = []
    else:
        blocks = [
            *(
                f"content_mode_blocks.{m}.{i}"
                for m in ("image", "image_cond")
                for i in range(layers)
            ),
            *(f"single_blocks.{i}" for i in range(single_layers)),
        ]
        extra = [
            f"{stage}_stream_modulations.{mode}.lin"
            for stage in ("early", "single")
            for mode in ("image", "image_cond", "txt")
        ]
    return [f"{block}.{linear}" for block in blocks for linear in BFL_BLOCK_LINEARS] + extra


def variant_names(config: dict[str, Any], variant: Optional[str]) -> list[str]:
    names = bfl_names(config)
    if variant is not None:
        names += [
            f"{module}.{scale}"
            for module in quantized_modules(config, variant)
            for scale in VARIANT_SCALES[variant]
        ]
    return names


@pytest.mark.parametrize(
    "config, variant, count",
    [
        pytest.param(KLEIN_CONFIG, None, 359, id="klein-bf16"),
        pytest.param(KLEIN_BASE_CONFIG, None, 357, id="base-bf16"),
        pytest.param(KLEIN_CONFIG, "nvfp4", 953, id="klein-nvfp4"),
        pytest.param(KLEIN_BASE_CONFIG, "nvfp4", 951, id="base-nvfp4"),
        pytest.param(KLEIN_CONFIG, "fp8r", 593, id="klein-fp8r"),
        pytest.param(KLEIN_BASE_CONFIG, "fp8r", 591, id="base-fp8r"),
    ],
)
def test_released_checkpoint_names_are_a_bijection(
    config: dict[str, Any], variant: Optional[str], count: int
) -> None:
    """Names laid out like the released packages map one to one onto the Diffusers checkpoint."""
    names = variant_names(config, variant)
    assert len(set(names)) == len(names) == count
    mapped = [bfl_to_diffusers_key(name) for name in names]
    assert len(set(mapped)) == count
    assert [diffusers_to_bfl_key(name) for name in mapped] == names
    assert {name for name in mapped if name.endswith(".weight")} == set(diffusers_shapes(config))


@pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "base"])
def test_tiny_checkpoint_names_round_trip(flow_map: bool) -> None:
    config = tiny_config(flow_map)
    names = sorted(diffusers_shapes(config))
    bfl = [diffusers_to_bfl_key(name) for name in names]
    assert len(set(bfl)) == len(names)
    assert set(bfl) == set(bfl_names(config))
    assert [bfl_to_diffusers_key(name) for name in bfl] == names


@pytest.mark.parametrize(
    "bfl, diffusers",
    [
        ("txt_in", "context_embedder"),
        ("emb_in.image_cond", "x_embedders.image_cond"),
        ("time_in.in_layer", "timestep_embedder.linear_1"),
        ("ref_time_in.out_layer", "target_timestep_embedder.linear_2"),
        ("early_stream_modulations.txt.lin", "early_stream_modulations.txt.linear"),
        ("final_layer.image.adaLN_modulation.1", "norm_out.linear"),
        ("final_layer.image.linear", "proj_out"),
        ("content_mode_blocks.image.0.attn_out", "mode_blocks.image.0.attn.to_out.0"),
        ("txt_mode_blocks.4.mlp_in", "txt_mode_blocks.4.ff.linear_in"),
        ("single_blocks.24.q_proj", "single_transformer_blocks.24.attn.to_q"),
    ],
)
def test_module_names_map_both_ways(bfl: str, diffusers: str) -> None:
    assert bfl_to_diffusers_module(bfl) == diffusers
    assert diffusers_to_bfl_module(diffusers) == bfl


def test_suffixes_and_qk_norm_names_map() -> None:
    assert (
        bfl_to_diffusers_key("content_mode_blocks.image_cond.4.attn_out.input_scale")
        == "mode_blocks.image_cond.4.attn.to_out.0.input_scale"
    )
    assert (
        bfl_to_diffusers_key("single_blocks.27.norm.key_norm.scale")
        == "single_transformer_blocks.27.attn.norm_k.weight"
    )
    assert (
        diffusers_to_bfl_key("single_transformer_blocks.3.attn.to_q.lora_A.weight")
        == "single_blocks.3.q_proj.lora_A.weight"
    )


def test_norm_out_rows_swap_between_shift_scale_orders() -> None:
    hidden_size = 4
    shift = torch.arange(hidden_size * 3, dtype=torch.float32).reshape(hidden_size, 3)
    scale = 100 + shift
    bias_shift = torch.arange(hidden_size, dtype=torch.float32)
    bias_scale = 10 + bias_shift
    bfl = {
        "final_layer.image.adaLN_modulation.1.weight": torch.cat([shift, scale]),
        "final_layer.image.adaLN_modulation.1.bias": torch.cat([bias_shift, bias_scale]),
        "txt_in.weight": torch.ones(2, 3),
    }

    diffusers = bfl_to_diffusers_state_dict(bfl)

    assert diffusers.keys() == {
        "norm_out.linear.weight",
        "norm_out.linear.bias",
        "context_embedder.weight",
    }
    assert torch.equal(diffusers["norm_out.linear.weight"], torch.cat([scale, shift]))
    assert torch.equal(diffusers["norm_out.linear.bias"], torch.cat([bias_scale, bias_shift]))
    assert diffusers["context_embedder.weight"] is bfl["txt_in.weight"]
    round_trip = diffusers_to_bfl_state_dict(diffusers)
    assert round_trip.keys() == bfl.keys()
    for key, value in bfl.items():
        assert torch.equal(round_trip[key], value), key


@pytest.mark.parametrize(
    "down, up", [("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight")]
)
def test_norm_out_lora_swaps_up_rows_only(down: str, up: str) -> None:
    """LoRA up rows follow the output rows; down rows and alpha are not split by output row."""
    hidden_size, rank = 4, 2
    up_shift = torch.arange(hidden_size * rank, dtype=torch.float32).reshape(hidden_size, rank)
    up_scale = 100 + up_shift
    bfl = {
        f"final_layer.image.adaLN_modulation.1.{down}": torch.randn(rank, hidden_size),
        f"final_layer.image.adaLN_modulation.1.{up}": torch.cat([up_shift, up_scale]),
        "final_layer.image.adaLN_modulation.1.alpha": torch.tensor(8.0),
    }

    diffusers = bfl_to_diffusers_state_dict(bfl)

    assert torch.equal(diffusers[f"norm_out.linear.{up}"], torch.cat([up_scale, up_shift]))
    assert (
        diffusers[f"norm_out.linear.{down}"] is bfl[f"final_layer.image.adaLN_modulation.1.{down}"]
    )
    assert diffusers["norm_out.linear.alpha"] is bfl["final_layer.image.adaLN_modulation.1.alpha"]
    round_trip = diffusers_to_bfl_state_dict(diffusers)
    assert round_trip.keys() == bfl.keys()
    for key, value in bfl.items():
        assert torch.equal(round_trip[key], value), key


@pytest.mark.parametrize(
    "convert, key",
    [
        (bfl_to_diffusers_state_dict, "final_layer.image.adaLN_modulation.1.weight_scale"),
        (diffusers_to_bfl_state_dict, "norm_out.linear.input_scale"),
    ],
)
def test_norm_out_tensors_without_a_known_row_order_raise(
    convert: Callable[[dict], dict], key: str
) -> None:
    with pytest.raises(ValueError, match="shift/scale order"):
        convert({key: torch.ones(4)})


def test_norm_out_swap_needs_even_rows() -> None:
    with pytest.raises(ValueError, match="even row count"):
        swap_norm_out_halves(torch.ones(3, 2))


@pytest.mark.parametrize(
    "convert, key",
    [
        (bfl_to_diffusers_key, "single_blocks.0.unknown.weight"),
        (bfl_to_diffusers_key, "repa_projectors.0.weight"),
        (bfl_to_diffusers_key, "single_blocks.0.q_proj"),
        (bfl_to_diffusers_key, "single_blocks.0.norm.query_norm.weight"),
        (bfl_to_diffusers_key, "x_embedders.image.weight"),
        (diffusers_to_bfl_key, "mode_blocks.image.0.attn.norm_q.bias"),
        (diffusers_to_bfl_key, "single_blocks.0.q_proj.weight"),
        (diffusers_to_bfl_key, "proj_out"),
    ],
)
def test_unknown_tensor_names_raise(convert: Callable[[str], str], key: str) -> None:
    with pytest.raises(ValueError, match="Unknown .* FLUX.3 tensor name"):
        convert(key)


@pytest.mark.parametrize(
    "convert, name",
    [
        (bfl_to_diffusers_module, "single_blocks.0.norm.query_norm"),
        (bfl_to_diffusers_module, "single_blocks.0.q_proj.weight"),
        (diffusers_to_bfl_module, "mode_blocks.image.0.attn"),
    ],
)
def test_unknown_module_names_raise(convert: Callable[[str], str], name: str) -> None:
    with pytest.raises(ValueError, match="Unknown .* FLUX.3 module name"):
        convert(name)
