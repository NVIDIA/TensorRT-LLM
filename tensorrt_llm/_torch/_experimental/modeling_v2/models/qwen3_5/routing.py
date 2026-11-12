# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen3_5ForConditionalGeneration.

`route` compares the facts that fix which code runs against the one deployment the
target was written for, and raises `NoTarget` stating them otherwise (`auto` falls
back to the built-in implementation; `require` raises). Shape criteria are read off
`text_config`: the dense Qwen3.5 text tower is what the target implements. It serves
text only and refuses a request carrying images or video, which needs
`modeling_v2='off'`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, MLP intermediate, attention heads, KV heads, head dim, linear key
# heads, linear value heads, vocab). Shape-sniffing also matches fine-tunes; the
# accuracy gate pins identity.
_TEXT_SHAPE = (64, 5120, 17408, 24, 4, 256, 16, 48, 248320)

# The recurrent-state pool the target reads and writes: e4m3 rows with one fp32 scale
# per row (`kv_cache_config.mamba_ssm_cache_dtype="int8"`).
_SSM_STATE_DTYPES = (torch.int8,)


class EngineChoices(NamedTuple):
    """Engine choices that fix which code the target runs but are not in the checkpoint's
    config, as the engine will resolve them."""

    torch_dtype: torch.dtype  # the dtype the engine runs the model in
    mamba_ssm_cache_dtype: torch.dtype  # the recurrent-state cache's dtype
    prefill_cuda_graph_backend: str  # `prefill_cuda_graph_backend`: disabled, piecewise, breakable


def engine_choices(model_config: ModelConfig, llm_args: TorchLlmArgs) -> EngineChoices:
    # Imported on the claimed branch only, as the target is.
    from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype

    return EngineChoices(
        torch_dtype=model_config.torch_dtype,
        mamba_ssm_cache_dtype=effective_mamba_ssm_cache_dtype(
            model_config, llm_args.kv_cache_config.mamba_ssm_cache_dtype
        ),
        prefill_cuda_graph_backend=str(llm_args.prefill_cuda_graph_backend.value),
    )


def _check_engine(kv_cache_quant_algo, engine: EngineChoices) -> None:
    """Decline what the target's constructor or first forward would refuse.

    Without this ``auto`` claims the deployment and then fails mid-build instead of
    falling back to the built-in implementation.
    """
    if kv_cache_quant_algo != QuantAlgo.FP8:
        raise NoTarget(
            f"a {kv_cache_quant_algo or 'bf16'} KV cache; the target's attention kernels read "
            "an fp8-e4m3 pool (kv_cache_config.dtype='fp8')"
        )
    if engine.torch_dtype != torch.bfloat16:
        raise NoTarget(f"the engine runs the model in {engine.torch_dtype}; the target is bf16")
    if engine.mamba_ssm_cache_dtype not in _SSM_STATE_DTYPES:
        raise NoTarget(
            f"a {engine.mamba_ssm_cache_dtype} recurrent-state cache; the target's Gated-DeltaNet "
            "kernels read an int8-typed pool (kv_cache_config.mamba_ssm_cache_dtype='int8')"
        )
    if engine.prefill_cuda_graph_backend != "disabled":
        raise NoTarget(
            f"prefill_cuda_graph_backend={engine.prefill_cuda_graph_backend!r}; the target runs "
            "steps with context rows eagerly"
        )


def _mixed_precision_recipe(model_config: ModelConfig) -> bool:
    """FP8 per-tensor attention and Gated-DeltaNet projections, NVFP4 MLP and lm_head: the
    ModelOpt MIXED_PRECISION export the target's weights.py reads."""
    quant_config = model_config.quant_config
    if quant_config is None or quant_config.quant_algo != QuantAlgo.MIXED_PRECISION:
        return False
    algos = {
        algo
        for name, cfg in (model_config.quant_config_dict or {}).items()
        for algo in (getattr(cfg, "quant_algo", None),)
        if algo is not None
    }
    return algos == {QuantAlgo.FP8, QuantAlgo.NVFP4}


@routes("Qwen3_5ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import of
    modeling code. It also declines the engine choices (`EngineChoices`) the target's
    constructor would refuse.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    text = getattr(pretrained_config, "text_config", None)
    if text is None:
        raise NoTarget(
            f"{type(pretrained_config).__name__} carries no text_config; the qwen3_5 target "
            "reads the composite checkpoint's text tower"
        )
    shape = (
        text.num_hidden_layers,
        text.hidden_size,
        text.intermediate_size,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        text.linear_num_key_heads,
        text.linear_num_value_heads,
        text.vocab_size,
    )
    recipe = _mixed_precision_recipe(model_config)

    # Qwen3.8-27B-NVFP4 on one RTX PRO 6000 Blackwell (sm120), no speculative decoding.
    if (
        compute_capability == (12, 0)
        and shape == _TEXT_SHAPE
        and recipe
        and mapping.world_size == 1
        and spec_config is None
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .qwen3_8_27b_nvfp4__sm_120__tp1.modeling import ModelingV2Qwen3827BNvfp4Sm120Tp1
        # isort: on
        # fmt: on

        _check_engine(kv_cache_quant_algo, engine_choices(model_config, llm_args))
        return ModelingV2Qwen3827BNvfp4Sm120Tp1

    raise NoTarget(
        f"compute capability {compute_capability}, text shape {shape}, "
        f"{'the' if recipe else 'not the'} FP8/NVFP4 MIXED_PRECISION recipe, "
        f"{mapping.world_size} ranks, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}; "
        "the one qwen3_5 target is Qwen3.8-27B-NVFP4 on sm120, tp1, without speculative decoding"
    )
