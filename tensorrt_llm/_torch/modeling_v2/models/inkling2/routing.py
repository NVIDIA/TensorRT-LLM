# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where an Inkling2ForConditionalGeneration config lands. Read this file and you know.

There is no built-in implementation of this architecture, so a deployment this route declines
fails in `auto` and `require` alike, with the reason `NoTarget` states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes
from .config import InklingDims

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

#: GB300 (sm_103).
_SMS = frozenset({(10, 3)})
#: Inkling-2-Small's text decoder: layers, hidden width, routed experts.
_SMALL = (48, 4608, 384)
#: The longest sequence the first target was gated on (the release supports 1M tokens).
MAX_SEQ_LEN = 262144
#: Inkling-2-Small's attention window per layer: 1024 tokens, full context on every sixth layer.
_SMALL_WINDOWS = tuple(None if layer % 6 == 5 else 1024 for layer in range(48))


def _windows(model_config: ModelConfig, text) -> tuple:
    """Each layer's window as the model applies it and as the KV cache keeps it (None: full).

    The KV cache reads the top-level config as `get_layer_attention_window` does; the model reads
    the text config. A field changed after loading can leave the two apart.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import get_layer_attention_window

    try:
        dims = InklingDims.from_config(text)
    except ValueError:
        return None, None
    model = tuple(
        None if dims.is_global(layer) else dims.sliding_window for layer in range(dims.num_layers)
    )
    cfg = model_config.pretrained_config
    try:
        cache = tuple(get_layer_attention_window(cfg, layer) for layer in range(dims.num_layers))
    except (NotImplementedError, ValueError):
        cache = None
    return model, cache


@routes("Inkling2ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise `NoTarget` saying why none fits."""
    cfg = model_config.pretrained_config
    text = getattr(cfg, "text_config", None) or cfg
    mapping = model_config.mapping
    shape = (text.num_hidden_layers, text.hidden_size, text.n_routed_experts)
    layout = (mapping.world_size, mapping.tp_size, mapping.moe_ep_size, mapping.moe_tp_size)
    kv_cache_config = llm_args.kv_cache_config
    model_windows, cache_windows = _windows(model_config, text)
    released_windows = model_windows == cache_windows == _SMALL_WINDOWS
    facts = {
        "GB300 (sm_103)": compute_capability in _SMS,
        "Inkling-2-Small's shape (48 layers, hidden 4608, 384 experts)": shape == _SMALL,
        "attention TP4 with MoE EP4 on 4 ranks": layout == (4, 4, 4, 1),
        "no pipeline, context or attention-data parallelism": mapping.pp_size == 1
        and mapping.cp_size == 1
        and not mapping.enable_attention_dp,
        "a BF16 KV cache": engine_kv_cache_quant_algo(model_config, llm_args) is None,
        "KV cache manager V2": kv_cache_config.use_kv_cache_manager_v2 is True,
        "no KV cache block reuse (the convolution windows are not cached)": not kv_cache_config.enable_block_reuse,
        "no SWA scratch reuse (the page tables do not map scratch slots)": not kv_cache_config.enable_swa_scratch_reuse,
        "Inkling-2-Small's attention windows (1024 tokens; full context on layers 5, 11, ..., 47), "
        "the same in the model and in the KV cache": released_windows,
        "no chunked prefill": not llm_args.enable_chunked_prefill,
        "no CUDA graphs (the forward runs eagerly)": llm_args.cuda_graph_config is None,
        "the overlap scheduler off": bool(llm_args.disable_overlap_scheduler),
        "no speculative decoding": model_config.spec_config is None,
        f"an explicit max_seq_len of at most {MAX_SEQ_LEN}": 0
        < (llm_args.max_seq_len or 0)
        <= MAX_SEQ_LEN,
    }
    unmet = [fact for fact, holds in facts.items() if not holds]
    if not unmet:
        from .inkling_2_small_nvfp4__sm_103__tep4.modeling import (
            ModelingV2Inkling2SmallNvfp4Sm103Tep4,
        )

        return ModelingV2Inkling2SmallNvfp4Sm103Tep4
    raise NoTarget(
        "the inkling2 target is Inkling-2-Small on GB300 with attention TP4 and MoE EP4, text only "
        f"and eager; this deployment lacks: {'; '.join(unmet)} (compute capability "
        f"{compute_capability}, shape {shape}, world {mapping.world_size} tp {mapping.tp_size} "
        f"moe_ep {mapping.moe_ep_size} moe_tp {mapping.moe_tp_size})"
    )
