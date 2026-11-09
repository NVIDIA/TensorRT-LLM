# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bijection between BFL-native and Diffusers FLUX.3 transformer tensor names.

The output norm also changes values: BFL orders its rows shift, scale and Diffusers scale, shift.
"""

import re
from typing import TYPE_CHECKING, Callable, Mapping, TypeVar

if TYPE_CHECKING:
    import torch

TensorT = TypeVar("TensorT", bound="torch.Tensor")

BFL_NORM_OUT = "final_layer.image.adaLN_modulation.1"
DIFFUSERS_NORM_OUT = "norm_out.linear"

_TOP_LEVEL_MODULES: tuple[tuple[str, str], ...] = (
    ("txt_in", "context_embedder"),
    ("emb_in.image", "x_embedders.image"),
    ("emb_in.image_cond", "x_embedders.image_cond"),
    ("time_in.in_layer", "timestep_embedder.linear_1"),
    ("time_in.out_layer", "timestep_embedder.linear_2"),
    ("ref_time_in.in_layer", "target_timestep_embedder.linear_1"),
    ("ref_time_in.out_layer", "target_timestep_embedder.linear_2"),
    *(
        (f"{stage}_stream_modulations.{mode}.lin", f"{stage}_stream_modulations.{mode}.linear")
        for stage in ("early", "single")
        for mode in ("image", "image_cond", "txt")
    ),
    (BFL_NORM_OUT, DIFFUSERS_NORM_OUT),
    ("final_layer.image.linear", "proj_out"),
)
_BLOCK_STACKS: tuple[tuple[str, str], ...] = (
    ("content_mode_blocks.image", "mode_blocks.image"),
    ("content_mode_blocks.image_cond", "mode_blocks.image_cond"),
    ("txt_mode_blocks", "txt_mode_blocks"),
    ("single_blocks", "single_transformer_blocks"),
)
_BLOCK_MODULES: tuple[tuple[str, str], ...] = (
    ("q_proj", "attn.to_q"),
    ("k_proj", "attn.to_k"),
    ("v_proj", "attn.to_v"),
    ("attn_out", "attn.to_out.0"),
    ("mlp_in", "ff.linear_in"),
    ("mlp_out", "ff.linear_out"),
)
_BLOCK_TENSORS: tuple[tuple[str, str], ...] = (
    ("norm.query_norm.scale", "attn.norm_q.weight"),
    ("norm.key_norm.scale", "attn.norm_k.weight"),
)
_SUFFIX = re.compile(r"[A-Za-z_]\w*(?:\.\w+)*")
# Output-norm tensors indexed by the shift and scale rows, and those that hold no such rows.
_NORM_OUT_SWAPPED_SUFFIXES = frozenset({"weight", "bias", "lora_B.weight", "lora_up.weight"})
_NORM_OUT_UNCHANGED_SUFFIXES = frozenset({"lora_A.weight", "lora_down.weight", "alpha"})


class _Direction:
    """Name tables for one direction of the mapping."""

    def __init__(self, source: str, pairs_index: int) -> None:
        def table(pairs: tuple[tuple[str, str], ...]) -> dict[str, str]:
            return {pair[pairs_index]: pair[1 - pairs_index] for pair in pairs}

        self.source = source
        self.top_level = table(_TOP_LEVEL_MODULES)
        self.stacks = table(_BLOCK_STACKS)
        self.block_modules = table(_BLOCK_MODULES)
        self.block_tensors = table(_BLOCK_TENSORS)
        stacks = "|".join(re.escape(name) for name in self.stacks)
        self.block = re.compile(rf"(?P<stack>{stacks})\.(?P<index>\d+)\.(?P<rest>.+)")

    def _split(self, name: str) -> tuple[str, str, dict[str, str]]:
        """Split ``name`` into a mapped block prefix, the name inside it, and its module table."""
        match = self.block.fullmatch(name)
        if match is None:
            return "", name, self.top_level
        prefix = f"{self.stacks[match['stack']]}.{match['index']}."
        return prefix, match["rest"], self.block_modules

    def module(self, name: str) -> str:
        prefix, rest, modules = self._split(name)
        if rest not in modules:
            raise ValueError(f"Unknown {self.source} FLUX.3 module name: {name!r}")
        return prefix + modules[rest]

    def key(self, key: str) -> str:
        prefix, rest, modules = self._split(key)
        if prefix and rest in self.block_tensors:
            return prefix + self.block_tensors[rest]
        # Longest module name first, so no shorter name can claim part of a longer one.
        for module_name in sorted(modules, key=len, reverse=True):
            suffix = rest[len(module_name) + 1 :]
            if rest.startswith(module_name + ".") and _SUFFIX.fullmatch(suffix):
                return f"{prefix}{modules[module_name]}.{suffix}"
        raise ValueError(f"Unknown {self.source} FLUX.3 tensor name: {key!r}")


_BFL_TO_DIFFUSERS = _Direction("BFL", 0)
_DIFFUSERS_TO_BFL = _Direction("Diffusers", 1)


def bfl_to_diffusers_module(name: str) -> str:
    """Map a BFL module name, e.g. ``single_blocks.3.q_proj``, to Diffusers.

    Raises ValueError if ``name`` is not a FLUX.3 BFL module.
    """
    return _BFL_TO_DIFFUSERS.module(name)


def diffusers_to_bfl_module(name: str) -> str:
    """Map a Diffusers module name, e.g. ``single_transformer_blocks.3.attn.to_q``, to BFL.

    Raises ValueError if ``name`` is not a FLUX.3 Diffusers module.
    """
    return _DIFFUSERS_TO_BFL.module(name)


def bfl_to_diffusers_key(key: str) -> str:
    """Map a BFL tensor name to Diffusers, keeping its suffix; ValueError if unknown."""
    return _BFL_TO_DIFFUSERS.key(key)


def diffusers_to_bfl_key(key: str) -> str:
    """Map a Diffusers tensor name to BFL, keeping its suffix; ValueError if unknown."""
    return _DIFFUSERS_TO_BFL.key(key)


def swap_norm_out_halves(value: TensorT) -> TensorT:
    """Swap the shift and scale row halves of an output-norm tensor [2 * hidden, ...].

    The swap is its own inverse.
    """
    rows = value.shape[0]
    if rows % 2:
        raise ValueError(f"FLUX.3 output-norm tensor needs an even row count, got {rows}")
    # Rolling dim 0 by half its length swaps the two halves.
    return value.roll(rows // 2, dims=0)


def _convert_state_dict(
    state_dict: Mapping[str, TensorT], convert_key: Callable[[str], str], norm_out: str
) -> dict[str, TensorT]:
    converted = {}
    for key, value in state_dict.items():
        if key.startswith(norm_out + "."):
            suffix = key[len(norm_out) + 1 :]
            if suffix in _NORM_OUT_SWAPPED_SUFFIXES:
                value = swap_norm_out_halves(value)
            elif suffix not in _NORM_OUT_UNCHANGED_SUFFIXES:
                raise ValueError(f"No shift/scale order transform for FLUX.3 tensor {key!r}")
        converted[convert_key(key)] = value
    return converted


def bfl_to_diffusers_state_dict(state_dict: Mapping[str, TensorT]) -> dict[str, TensorT]:
    """Rename a BFL state dict to Diffusers names and reorder the output norm to scale, shift.

    Raises ValueError for unknown names and for output-norm tensors without a known row order.
    """
    return _convert_state_dict(state_dict, bfl_to_diffusers_key, BFL_NORM_OUT)


def diffusers_to_bfl_state_dict(state_dict: Mapping[str, TensorT]) -> dict[str, TensorT]:
    """Rename a Diffusers state dict to BFL names and reorder the output norm to shift, scale.

    Raises ValueError for unknown names and for output-norm tensors without a known row order.
    """
    return _convert_state_dict(state_dict, diffusers_to_bfl_key, DIFFUSERS_NORM_OUT)
