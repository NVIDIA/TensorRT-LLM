# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-module attention precision of a MiniMax-H3 checkpoint.

A checkpoint may name attention modules whose operands stay unquantized while the pipeline runs
a quantized attention recipe on the others. The entry lives in the transformer's
``quantization_config`` beside ``quantized_layers``:

.. code-block:: json

    "attention_precision": {
        "schema_version": 1,
        "layers": {
            "transformer_blocks.45.attn": "bf16",
            "token_refiner.refiner_blocks.0.attn": "bf16"
        }
    }

* A listed module runs its attention on BF16 operands: it is built without the pipeline's
  ``quant_attention_config`` and without sparse attention. The backend stays the pipeline's.
* An unlisted module follows the pipeline's attention configuration.
* A checkpoint without the entry loads as it did before the entry existed.
* ``"bf16"`` is the only value: a quantized attention recipe takes its scales at run time, so a
  checkpoint has nothing to declare for it.

The same mapping may be given at run time as ``pipeline_config["minimax_h3_attention_precision"]``
for a checkpoint exported without the entry. When both are present they must be equal.
"""

from typing import Any, Mapping, Optional

import torch

SCHEMA_VERSION = 1
CHECKPOINT_KEY = "attention_precision"
RUNTIME_KEY = "minimax_h3_attention_precision"
# Ulysses sizes at which BF16 attention modules and MIXED_PRECISION Linears are served.
SUPPORTED_ULYSSES_SIZES = (1, 4)
SUPPORTED_ULYSSES_SIZES_TEXT = " and ".join(str(size) for size in SUPPORTED_ULYSSES_SIZES)
BF16 = "bf16"
_ENTRY_KEYS = frozenset({"schema_version", "layers"})
_PATTERN_CHARACTERS = frozenset("*?[]")


def parse_attention_precision(entry: Any, source: str) -> dict[str, str]:
    """Validate one ``attention_precision`` entry and return its ``layers`` mapping.

    ``source`` names where the entry came from, for the error message. Module names are exact;
    whether they exist in a model is checked by :func:`validate_attention_precision`.
    """
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"{source}: attention_precision must be a mapping, got {type(entry).__name__}"
        )
    unknown = sorted(set(entry) - _ENTRY_KEYS)
    if unknown:
        raise ValueError(f"{source}: attention_precision has unknown keys {unknown}")
    version = entry.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(
            f"{source}: attention_precision schema_version {version!r} is not supported "
            f"(expected the integer {SCHEMA_VERSION})"
        )
    layers = entry.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError(
            f"{source}: attention_precision.layers must be a mapping of module name to precision, "
            f"got {type(layers).__name__}"
        )
    parsed: dict[str, str] = {}
    for name, precision in layers.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{source}: attention_precision.layers has the invalid module name {name!r}"
            )
        if _PATTERN_CHARACTERS & set(name):
            raise ValueError(
                f"{source}: attention_precision.layers[{name!r}] is a pattern; module names are exact"
            )
        if precision != BF16 or not isinstance(precision, str):
            raise ValueError(
                f"{source}: attention_precision.layers[{name!r}] asks for {precision!r}; "
                f"only {BF16!r} is supported"
            )
        parsed[name] = precision
    return parsed


def resolve_attention_precision(
    pretrained_config: Any, extra_attrs: Optional[Mapping[str, Any]]
) -> Optional[dict[str, str]]:
    """The attention-precision mapping of a model: the checkpoint's entry, the run-time option, or
    None when neither is given. Both given and different is an error; neither overrides the other."""
    quantization_config = getattr(pretrained_config, "quantization_config", None)
    from_checkpoint = None
    if isinstance(quantization_config, Mapping) and CHECKPOINT_KEY in quantization_config:
        from_checkpoint = parse_attention_precision(
            quantization_config[CHECKPOINT_KEY], "checkpoint quantization_config"
        )
    from_runtime = None
    runtime_entry = (extra_attrs or {}).get(RUNTIME_KEY)
    if runtime_entry is not None:
        from_runtime = parse_attention_precision(runtime_entry, f"pipeline_config[{RUNTIME_KEY!r}]")
    if from_checkpoint is not None and from_runtime is not None and from_checkpoint != from_runtime:
        differing = sorted(set(from_checkpoint.items()) ^ set(from_runtime.items()))
        raise ValueError(
            f"the checkpoint's attention_precision and pipeline_config[{RUNTIME_KEY!r}] differ "
            f"in {sorted({name for name, _ in differing})}; give one, or make them equal"
        )
    return from_checkpoint if from_checkpoint is not None else from_runtime


def attention_module_names(num_layers: int, num_refiner_layers: int) -> list[str]:
    """Every attention module of the MiniMax-H3 transformer, by the name the entry uses."""
    return [f"token_refiner.refiner_blocks.{i}.attn" for i in range(num_refiner_layers)] + [
        f"transformer_blocks.{i}.attn" for i in range(num_layers)
    ]


def validate_attention_precision(
    layers: Mapping[str, str],
    *,
    num_layers: int,
    num_refiner_layers: int,
    model_dtype: torch.dtype,
    ulysses_size: int,
    async_ulysses: bool = False,
) -> None:
    """Check a parsed mapping against the model it is applied to, before the model is built."""
    if not layers:
        return
    known = set(attention_module_names(num_layers, num_refiner_layers))
    unknown = sorted(set(layers) - known)
    if unknown:
        raise ValueError(
            f"attention_precision names modules that are not attention modules of this model: "
            f"{unknown} ({num_refiner_layers} refiner blocks, {num_layers} transformer blocks)"
        )
    if model_dtype != torch.bfloat16:
        raise ValueError(
            f"attention_precision asks for {BF16!r} operands, but the model's dtype is {model_dtype}"
        )
    if ulysses_size not in SUPPORTED_ULYSSES_SIZES:
        raise NotImplementedError(
            f"attention_precision is supported at Ulysses {SUPPORTED_ULYSSES_SIZES_TEXT}, "
            f"found Ulysses {ulysses_size}"
        )
    if async_ulysses and ulysses_size > 1:
        raise NotImplementedError("attention_precision is not supported with async_ulysses")
