# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for ``Spark2_5ForCausalLM``.

``route`` returns the dense sm_10x tp1 target for the published BF16
checkpoints, and raises ``NoTarget`` otherwise -- in ``auto`` the engine then
falls back to the built-in V1 implementation (``_torch/models/modeling_spark.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

# The GPUs the target is certified on: sm_100 (B200, GB200) and sm_103 (B300,
# GB300). They run the same kernels.
_SMS = frozenset({(10, 0), (10, 3)})

# The published BF16 checkpoint shapes the target has been checked against, as
# (num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads,
# head_dim): Spark-X2.5/X2.6 1.7B and 4B. A fine-tune with the same shape
# routes here too; the accuracy gate is what pins the identity.
_KNOWN_SHAPES = frozenset(
    {
        (28, 2048, 8, 2, 256),  # 1.7B
        (36, 2560, 16, 4, 256),  # 4B
    }
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@routes("Spark2_5ForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    cfg = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)

    shape = (
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )

    # Checkpoint facts the target's forward and weight table assume. Each one
    # would compute a plausible wrong answer rather than fail if it drifted.
    layer_types = list(getattr(cfg, "layer_types", None) or [])
    hybrid_ok = len(layer_types) == cfg.num_hidden_layers and all(
        t in ("sliding_attention", "full_attention") for t in layer_types
    )
    window = getattr(cfg, "sliding_window", None)
    window_ok = isinstance(window, int) and not isinstance(window, bool) and window > 0
    rope = getattr(cfg, "rope_parameters", None) or {}
    rope_ok = all(
        set(rope.get(kind, {})) <= {"rope_theta", "partial_rotary_factor"}
        for kind in ("sliding_attention", "full_attention")
    )
    checkpoint_ok = (
        shape in _KNOWN_SHAPES
        and hybrid_ok
        and window_ok
        and rope_ok
        and getattr(cfg, "gate_attn_act_mode", "sigmoid") == "sigmoid"
        and bool(getattr(cfg, "headwise_attn_output_gate", False))
        and getattr(cfg, "hidden_act", "gelu") == "gelu"
        and not getattr(cfg, "attention_bias", False)
        and not getattr(cfg, "mlp_bias", False)
        and bool(getattr(cfg, "tie_word_embeddings", False))
        # ModelConfig.quant_config defaults to an unquantized QuantConfig()
        # object rather than None, so check the algo, not the object.
        and getattr(model_config.quant_config, "quant_algo", None) is None
    )

    # One rank only: the target computes no collective. Pipeline and context
    # parallelism change which rank holds which layers or tokens, and the
    # target implements neither.
    single_rank = (
        mapping.world_size == 1
        and mapping.tp_size == 1
        and not mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
    )

    if (
        compute_capability in _SMS
        and checkpoint_ok
        and single_rank
        and spec_config is None
        and kv_cache_quant_algo is None
    ):
        from .spark2_5_dense__sm_10x__default.modeling import ModelingV2Spark25DenseSm10xDefault

        return ModelingV2Spark25DenseSm10xDefault

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"{mapping.world_size} rank(s) (tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} attention_dp={bool(mapping.enable_attention_dp)}), "
        f"speculative decoding "
        f"{type(spec_config).__name__ if spec_config is not None else 'off'}, "
        f"KV cache {kv_cache_quant_algo or 'bf16'}, "
        f"quantization {getattr(model_config.quant_config, 'quant_algo', None) or 'none'}; "
        "the spark2_5 target is a published BF16 Spark-X2.5/X2.6 checkpoint "
        "(1.7B or 4B shape) on sm100 or sm103, on one rank, without speculative "
        "decoding, quantization or a quantized KV cache"
    )
