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
"""MiniMax-H3 scenarios: one optimized pipeline per deployment, in its own folder.

A scenario folder ``models/minimax_h3/<checkpoint>__<sm>__<parallel>/`` holds a complete variant of the
H3 transformer and pipeline with its own kernels, after Modeling V2's shape (one target per
configuration, model-specific kernels in the model directory, the shared code untouched). This module
is the family's routing: :func:`resolve_scenario` picks the scenario whose requirements the pipeline
config meets, or keeps the family pipeline.

``pipeline_config.extra_attrs["visual_gen_scenario"]`` selects the mode, as ``modeling_v2`` does for LLMs:

| mode | behaviour |
|---|---|
| ``"off"`` (default) | the family pipeline; no scenario code is imported |
| ``"auto"`` | a scenario when one matches, else the family pipeline |
| ``"require"`` | a scenario, or an error naming the requirement that failed |

``extra_attrs["minimax_h3_scenario"]`` (a mapping) overrides a scenario's settings for an A/B; see the
scenario's ``config.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional, Type

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm.models.modeling_utils import QuantConfig

    from ...config import DiffusionPipelineConfig
    from ...pipeline import BasePipeline

_MODES = ("off", "auto", "require")


class NoScenario(Exception):
    """A scenario declines the configuration; the message names the requirement that failed."""


def _device_capability() -> Optional[tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


def _device_multiprocessors() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).multi_processor_count)


def _transformer_quant_config(config: "DiffusionPipelineConfig") -> "QuantConfig":
    model_config = config.model_configs.get("transformer")
    quant_config = getattr(model_config, "quant_config", None) if model_config is not None else None
    if quant_config is None:
        quant_config = config.quant_config
    return quant_config


def _transformer_quant_algo(config: "DiffusionPipelineConfig") -> Optional[QuantAlgo]:
    return getattr(_transformer_quant_config(config), "quant_algo", None)


def _transformer_layer_quant(config: "DiffusionPipelineConfig") -> Optional[dict]:
    """The per-layer quant table of a MIXED_PRECISION transformer checkpoint, else None."""
    model_config = config.model_configs.get("transformer")
    layer_quant = (
        getattr(model_config, "quant_config_dict", None) if model_config is not None else None
    )
    if layer_quant is None:
        layer_quant = getattr(config, "quant_config_dict", None)
    return layer_quant


def _ulysses_size(config: "DiffusionPipelineConfig") -> int:
    vgm = config.visual_gen_mapping
    return int(getattr(vgm, "ulysses_size", 1) or 1) if vgm is not None else 1


def route_nvfp4_sm_107_u1_u4(config: "DiffusionPipelineConfig") -> Type["BasePipeline"]:
    """``minimax_h3_nvfp4__sm_107__u1_u4``: a pre-quantized NVFP4 checkpoint, SM 107, Ulysses 1 or 4,
    the cuDNN attention backend with the per-tensor FP8 recipe, no tensor parallelism.

    A MIXED_PRECISION checkpoint whose Linears are all static NVFP4 or static per-tensor FP8 is
    admitted explicitly, at the same Ulysses sizes; the global algorithm of such a checkpoint (the
    one most of its Linears carry) is not what admits it. An ``attention_precision`` entry is
    validated by the transformer, not here."""
    capability = _device_capability()
    if capability != (10, 7):
        raise NoScenario(f"needs an SM 107 device, found {capability}")
    quant_algo = _transformer_quant_algo(config)
    layer_quant = _transformer_layer_quant(config)
    mixed = layer_quant is not None
    if mixed:
        if config.dynamic_weight_quant:
            raise NoScenario("a MIXED_PRECISION checkpoint needs dynamic_weight_quant=False")
        other = sorted(
            {
                getattr(getattr(layer, "quant_algo", None), "name", str(None))
                for layer in layer_quant.values()
                if getattr(layer, "quant_algo", None) not in (QuantAlgo.NVFP4, QuantAlgo.FP8)
            }
        )
        if other or not layer_quant:
            raise NoScenario(
                "a MIXED_PRECISION checkpoint is served only when every Linear is static NVFP4 or "
                f"static per-tensor FP8; found {other or 'no layers'}"
            )
    elif quant_algo != QuantAlgo.NVFP4 or config.dynamic_weight_quant:
        raise NoScenario(
            f"needs a pre-quantized NVFP4 transformer checkpoint (dynamic_weight_quant=False); "
            f"found quant_algo={quant_algo}, dynamic_weight_quant={config.dynamic_weight_quant}"
        )
    backend = str(getattr(config.attention, "backend", "")).upper()
    if backend != "CUDNN":
        raise NoScenario(f"needs attention.backend=CUDNN, found {backend!r}")
    qac = getattr(config.attention, "quant_attention_config", None)
    if (
        qac is None
        or getattr(qac, "qk_dtype", None) != "fp8"
        or getattr(qac, "v_dtype", None) != "fp8"
    ):
        raise NoScenario("needs the per-tensor FP8 attention recipe (qk_dtype=v_dtype='fp8')")
    ulysses = _ulysses_size(config)
    if ulysses not in (1, 4):
        raise NoScenario(f"serves Ulysses 1 or 4, found {ulysses}")
    if config.mapping.tp_size > 1:
        raise NoScenario("serves tp_size=1")
    if getattr(config.parallel, "async_ulysses", False):
        raise NoScenario("serves the packed FP8 Ulysses exchange, not async_ulysses")

    from . import minimax_h3_nvfp4__sm_107__u1_u4 as pkg  # the config is applied before this import

    pkg.config.apply(config.extra_attrs.get("minimax_h3_scenario"))
    if mixed:
        logger.info(
            "MiniMax-H3 scenario minimax_h3_nvfp4__sm_107__u1_u4: admitting a MIXED_PRECISION "
            "checkpoint (static NVFP4 and static FP8 Linears)"
        )
    from .minimax_h3_nvfp4__sm_107__u1_u4.pipeline import MiniMaxH3Nvfp4Sm107U1U4Pipeline

    return MiniMaxH3Nvfp4Sm107U1U4Pipeline


def route_fp8_sm_107_u1(config: "DiffusionPipelineConfig") -> Type["BasePipeline"]:
    """Uniform static per-tensor FP8, 212-SM Rubin, Ulysses=TP=1 and compile on.

    Mixed checkpoints, asynchronous exchange, weight mutation/offload, cache
    approximations and whole-model CUDA Graphs are outside this fixed scenario.
    """
    if _device_capability() != (10, 7) or _device_multiprocessors() != 212:
        raise NoScenario("the static-FP8 fusion needs an SM 107 device with 212 SMs")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise NoScenario("the static-FP8 fusion is qualified with NVIDIA_TF32_OVERRIDE=0")
    if (
        _transformer_quant_algo(config) != QuantAlgo.FP8
        or _transformer_layer_quant(config) is not None
        or config.dynamic_weight_quant
        or config.force_dynamic_quantization
    ):
        raise NoScenario("the static-FP8 fusion needs a uniform pre-quantized FP8 checkpoint")
    quant_config = _transformer_quant_config(config)
    for block_index in range(50):
        for projection in ("attn.qkv_proj", "attn.to_out.0", "ff.gate_up_proj", "ff.down_proj"):
            name = f"transformer_blocks.{block_index}.{projection}"
            if quant_config.is_module_excluded_from_quantization(name):
                raise NoScenario(
                    "the static-FP8 fusion needs all 200 main Linears quantized; "
                    f"{name} is excluded from quantization"
                )
    qac = getattr(config.attention, "quant_attention_config", None)
    if (
        str(config.attention.backend).upper() != "CUDNN"
        or qac is None
        or getattr(qac, "qk_dtype", None) != "fp8"
        or getattr(qac, "v_dtype", None) != "fp8"
    ):
        raise NoScenario("the static-FP8 fusion needs CUDNN with qk_dtype=v_dtype='fp8'")
    if (
        _ulysses_size(config) != 1
        or config.mapping.tp_size != 1
        or config.parallel.parallel_vae_size != 1
    ):
        raise NoScenario("the static-FP8 fusion serves Ulysses=TP=parallel_vae_size=1")
    if config.parallel.async_ulysses:
        raise NoScenario("the static-FP8 fusion does not serve async_ulysses")
    if not config.torch_compile.enable or config.cuda_graph.enable:
        raise NoScenario("the static-FP8 fusion needs compile on and whole-model CUDA Graphs off")
    if (
        config.cpu_offload_config.enable
        or config.runtime_lora is not None
        or config.cache is not None
    ):
        raise NoScenario(
            "the static-FP8 fusion needs immutable resident weights without a diffusion cache"
        )
    if config.extra_attrs.get("minimax_h3_scenario"):
        raise NoScenario(
            "the static-FP8 fusion uses fixed component settings without scenario overrides"
        )
    from .minimax_h3_fp8__sm_107__u1.config import select

    select()
    from .minimax_h3_fp8__sm_107__u1.pipeline import MiniMaxH3Fp8Sm107U1Pipeline

    return MiniMaxH3Fp8Sm107U1Pipeline


# Scenario routes in the order they are tried; the first that does not raise wins.
ROUTES: tuple[Callable[["DiffusionPipelineConfig"], Type["BasePipeline"]], ...] = (
    route_nvfp4_sm_107_u1_u4,
    route_fp8_sm_107_u1,
)


def resolve_scenario(
    family_cls: Type["BasePipeline"], config: "DiffusionPipelineConfig"
) -> Type["BasePipeline"]:
    """The pipeline class for *config*: a scenario in ``auto`` / ``require`` mode, else *family_cls*."""
    mode = str(config.extra_attrs.get("visual_gen_scenario", "off")).lower()
    if mode not in _MODES:
        raise ValueError(f"visual_gen_scenario={mode!r}: expected one of {_MODES}")
    if mode == "off":
        return family_cls
    reasons: list[str] = []
    for route in ROUTES:
        try:
            cls = route(config)
        except NoScenario as e:
            reasons.append(f"{route.__name__}: {e}")
            continue
        logger.info(f"MiniMax-H3 scenario {cls.__name__} selected ({route.__name__})")
        return cls
    if mode == "require":
        raise RuntimeError(
            "visual_gen_scenario=require, but no MiniMax-H3 scenario matches this configuration: "
            + "; ".join(reasons)
        )
    logger.info(
        "MiniMax-H3: no scenario matches, using the family pipeline (" + "; ".join(reasons) + ")"
    )
    return family_cls
