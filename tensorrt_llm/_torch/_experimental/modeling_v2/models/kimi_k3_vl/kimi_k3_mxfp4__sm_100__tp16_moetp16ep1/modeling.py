# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3 (MXFP4) / sm_100 / tp16 attention, routed experts moe_tp 16 x moe_ep 1.

Kimi K3's language model: 93 layers, hidden 7168. Every fourth layer (3, 7, ..., 91) is MLA attention with 96
query heads; the others are Kimi Delta Attention (KDA), a gated linear-attention recurrence with per-request state.
Layer 0's MLP is dense. Every other layer's MLP is a latent MoE: 896 routed experts, top-16, expert width 3584.
Attention residuals mix each sublayer's input from a bank of earlier outputs. The routed experts are MXFP4 (the
checkpoint's compressed-tensors default, read as W4A8_MXFP4_MXFP8); everything else is bf16, and so is the KV pool.

`tp16_moetp16ep1` is `tensor_parallel_size: 16` with `moe_tensor_parallel_size: 16` and
`moe_expert_parallel_size: 1`, both set explicitly, no attention data parallelism, all 16 GPUs in one NVLink domain.
Attention is head-split (6 MLA query heads per rank), and every rank holds all 896 experts at a sixteenth of their
width (192 of 3072 intermediate values, zero-padded to 256 by the loader). With the expert split left unset the
built-in model runs the experts expert-parallel over the 16 ranks instead, and no target serves that layout.

**Each step takes one of two paths, chosen on the host from the step's shape** (`_step_path`):

* **The fused decode path**: pure decode steps of at most 8 tokens, on the K3 decode kernels' catalog entries. The
  routed experts of one and two tokens run as `moe/k3_moe_m1` and `moe/k3_moe_m2`, and of 3-8 tokens as k3_moe, each
  pushing its partial into the latent exchange that `trtllm::k3_latent_reduce` sums. The state those kernels share
  (MNNVL workspace, sandwich and MoE Lamport buffers, the latent exchange, the engines' workspaces, KDA / MLA scratch)
  lives in typed objects this target creates in `post_load_weights`, before any graph capture. Until those entries
  are wired `_fused_decode` stays None, and every step takes the generic path.
* **The generic path**: prefill, mixed steps, and decode steps above those bounds, on the built-in Kimi K3 text
  model, whose modules and ops have no catalog entries yet. `UNCERTIFIED_GENERIC_CALLS` names them.

**What this target asserts rather than adapts**: SM 10.0; the topology above, with its expert split explicit; no
speculative decoding; bf16 weights and a bf16 KV pool;
tokens_per_block 64 (the MLA generation kernels K3's 96 heads reach exist only at 64); the V2 hybrid KV / state
manager, which holds the KDA states, with block reuse off; an all-reduce strategy of AUTO or MNNVL. The
construction-time ones fail in `__init__`, the per-engine ones on the first forward, each naming the setting.

**Text only.** The checkpoint is the vision-language wrapper. This target builds and loads no vision tower (its
weights are a predicted non-load, `weights.py`), and a step carrying multimodal input raises.

**No speculative decoding.** This layout is the low-latency route for decoding without a drafter; DSpark runs on
the `tp16_moetp4ep4` target. A configuration with a speculative decoding config stops at construction.
"""

import copy
from typing import Any, Literal, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.models.modeling_utils import register_auto_model
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.functional import AllReduceStrategy

from . import weights as _weights

# The GPU architecture this target IS. Routing will not send another one here, but a direct instantiation could,
# and the certification is per arch.
_SM = (10, 0)

#: Every trtllm op this target reaches for, in its forward and in the weight load. Declared here, asserted in
#: tests/unittest/_torch/modeling_v2. Today these are the K3-specific ops of the generic path (attention residuals,
#: KDA, the router and fused-A GEMMs); the fused decode path adds its own.
REQUIRED_TRTLLM_OPS = (
    "attn_res_fwd",
    "attn_res_rmsnorm_fwd",
    "attn_res_add_rmsnorm_fwd",
    "attn_res_add_rmsnorm_persistent_fwd",
    "kda_prefill",
    "kda_decode",
    "kda_mtp_decode",
    "dsv3_router_gemm_op",
    "dsv3_fused_a_gemm_op",
)

#: The engine surface the first forward checks before this target relies on it: per object, the attributes read.
#: A renamed field upstream fails here, loudly, instead of reading as a default.
REQUIRED_ENGINE_FIELDS = {
    "attn_metadata": (
        "num_contexts",
        "num_seqs",
        "num_tokens",
        "tokens_per_block",
        "kv_cache_manager",
    ),
    "kv_cache_manager": ("enable_block_reuse",),
}

#: Calls the generic path makes outside the catalog, declared so they are not consumed silently. A call leaves this
#: list when a catalog entry replaces it.
UNCERTIFIED_GENERIC_CALLS = (
    "tensorrt_llm._torch.models.modeling_kimi_linear.KimiLinearForCausalLM",
)

# The fused decode path's token bound per step: one token per request (the K3 decode kernels are built for up to 8
# rows).
_FUSED_MAX_TOKENS = 8

# The MLA generation kernels for K3's 96 query heads exist only at a 64-token page (the built-in model's own
# get_model_defaults sets it for the same reason).
_TOKENS_PER_BLOCK = 64

_LANG_PREFIX = "language_model."


def _text_model_config(model_config: ModelConfig) -> ModelConfig:
    """The language model's ModelConfig: the checkpoint's text_config, with quant exclusions renamed to match.

    The checkpoint names its language-model modules `language_model.<x>`; the text model's are `<x>`, with
    `layers.*` under `model.`.
    """
    config = model_config.pretrained_config
    text = copy.copy(model_config)
    text._frozen = False
    text.pretrained_config = config.text_config
    excluded = text.quant_config.exclude_modules
    if excluded:
        text.quant_config = copy.copy(text.quant_config)
        renamed = []
        for name in excluded:
            if name.startswith(_LANG_PREFIX):
                name = name[len(_LANG_PREFIX) :]
                if name.startswith("layers."):
                    name = "model." + name
            renamed.append(name)
        text.quant_config.exclude_modules = renamed
    text.skip_create_weights_in_init = True
    text._frozen = True
    return text


def _check_construction(model_config: ModelConfig) -> None:
    """The settings this target is built for that are fixed before the first step."""
    capability = torch.cuda.get_device_capability()
    assert capability == _SM, (
        f"this target is certified on sm_{_SM[0]}{_SM[1]}, running on sm_{capability[0]}{capability[1]}"
    )
    mapping = model_config.mapping
    topology = (
        mapping.world_size,
        mapping.tp_size,
        mapping.pp_size,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        mapping.enable_attention_dp,
    )
    assert topology == (16, 16, 1, 16, 1, False), (
        "the tp16_moetp16ep1 target needs world_size 16, tensor_parallel_size 16, pipeline_parallel_size 1, "
        "moe_tensor_parallel_size 16, moe_expert_parallel_size 1 and enable_attention_dp false; the engine built "
        f"(world, tp, pp, moe_tp, moe_ep, attention_dp) = {topology}"
    )
    assert getattr(mapping, "moe_tp_ep_user_specified", False), (
        "the tp16_moetp16ep1 target needs moe_tensor_parallel_size 16 and moe_expert_parallel_size 1 set explicitly; "
        "with the split unset the built-in Kimi K3 model runs the experts expert-parallel over the 16 ranks"
    )
    assert model_config.spec_config is None, (
        "the tp16_moetp16ep1 target decodes without speculation; DSpark runs on the tp16_moetp4ep4 target "
        f"(moe_tensor_parallel_size 4, moe_expert_parallel_size 4); got {type(model_config.spec_config).__name__}"
    )
    assert model_config.torch_dtype == torch.bfloat16, (
        f"this target computes in bf16; the engine resolved dtype {model_config.torch_dtype}"
    )
    kv_algo = model_config.quant_config.kv_cache_quant_algo
    assert kv_algo is None, (
        f"this target's MLA kernels read a bf16 KV pool; kv_cache_config.dtype resolved to {kv_algo}"
    )
    strategy = model_config.allreduce_strategy
    assert strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL), (
        f"this target runs its all-reduces over MNNVL; allreduce_strategy is {strategy.name}"
    )


@register_auto_model("ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1")
class ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1(KimiLinearForCausalLM):
    """The registration shell: the built-in Kimi K3 text model as the generic path, behind this target's checks."""

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """The V2 hybrid manager holds the KDA states; the step contract requires it."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        assert getattr(config, "text_config", None) is not None, (
            "this target loads the KimiK3ForConditionalGeneration checkpoint, whose language model is its "
            "text_config"
        )
        _check_construction(model_config)
        super().__init__(_text_model_config(model_config))
        self._step_checked = False
        # The fused decode path and the state its kernels share, built in post_load_weights once the catalog entries
        # it calls exist. None: every step takes the generic path.
        self._fused_decode = None
        # The executor reads generation settings (eos_token_id, ...) off the model config the engine holds, which
        # must therefore be the text config, as the built-in wrapper leaves it.
        model_config._frozen = False
        model_config.pretrained_config = self.config
        model_config._frozen = True

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self, weights)

    def _check_step_contract(self, attn_metadata: AttentionMetadata) -> None:
        """First-forward checks of the engine surface and the per-engine settings."""
        objects = {
            "attn_metadata": attn_metadata,
            "kv_cache_manager": attn_metadata.kv_cache_manager,
        }
        missing = [
            f"{owner}.{name}"
            for owner, names in REQUIRED_ENGINE_FIELDS.items()
            for name in names
            if not hasattr(objects[owner], name)
        ]
        assert not missing, f"engine fields this target reads are missing: {missing}"
        assert attn_metadata.tokens_per_block == _TOKENS_PER_BLOCK, (
            f"this target needs kv_cache_config.tokens_per_block {_TOKENS_PER_BLOCK}; the engine built "
            f"{attn_metadata.tokens_per_block}"
        )
        manager = attn_metadata.kv_cache_manager
        assert isinstance(manager, MambaHybridCacheManagerV2), (
            "this target needs the V2 hybrid KV / state cache manager "
            "(kv_cache_config.use_kv_cache_manager_v2); the engine built "
            f"{type(manager).__name__}"
        )
        assert not manager.enable_block_reuse, (
            "this target runs with kv_cache_config.enable_block_reuse false; the engine enabled it"
        )
        self._step_checked = True

    def _step_path(self, attn_metadata: AttentionMetadata, spec_metadata) -> str:
        """`"fused"` for a pure decode step within the fused path's bounds, else `"generic"`.

        Read on the host from per-step integers only. A CUDA graph is captured per decode batch shape, and every
        input here is fixed by that shape, so a captured step and its replays take the same path.
        """
        if attn_metadata.num_contexts:
            return "generic"
        return "fused" if attn_metadata.num_tokens <= _FUSED_MAX_TOKENS else "generic"

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.pop("multimodal_params", None):
            raise ValueError(
                "this Kimi K3 target is text only: it loads no vision tower, and a request carried image input"
            )
        if not self._step_checked:
            self._check_step_contract(attn_metadata)
        if (
            self._fused_decode is not None
            and self._step_path(attn_metadata, spec_metadata) == "fused"
        ):
            return self._fused_decode(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                spec_metadata=spec_metadata,
                resource_manager=resource_manager,
                **kwargs,
            )
        return super().forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
            resource_manager,
            **kwargs,
        )
