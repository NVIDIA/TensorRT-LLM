# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3 (MXFP4) / sm_100 / tp16 attention, routed experts moe_tp 4 x moe_ep 4.

Kimi K3's language model: 93 layers, hidden 7168. Every fourth layer (3, 7, ..., 91) is MLA attention with 96
query heads; the others are Kimi Delta Attention (KDA), a gated linear-attention recurrence with per-request state.
Layer 0's MLP is dense. Every other layer's MLP is a latent MoE: 896 routed experts, top-16, expert width 3584.
Attention residuals mix each sublayer's input from a bank of earlier outputs. The routed experts are MXFP4 (the
checkpoint's compressed-tensors default, read as W4A8_MXFP4_MXFP8); everything else is bf16, and so is the KV pool.

`tp16_moetp4ep4` is `tensor_parallel_size: 16` with `moe_tensor_parallel_size: 4` and
`moe_expert_parallel_size: 4`, no attention data parallelism, all 16 GPUs in one NVLink domain. Attention is
head-split (6 MLA query heads per rank), and every rank holds a quarter of the width of a quarter of the experts.

**Each step is classified once, on the host, from its shape** (`decode_step`, a `DecodeStep` or None), and the
classification decides which kernels each module runs:

* **small**: at most 8 tokens (`DECODE_MAX_TOKENS`, one token tile), context requests included. The token-count
  kernels take it: decode GEMVs, the MoE front and routed experts, the sandwiches, the embedding and residual
  epilogues.
* **decode**: a pure decode step of R <= 8 generation requests with the same T <= 8 tokens each (one token without
  speculation, 1 + 7 drafts with DSpark) and no context request, so at most 64 tokens. The request-aware kernels
  take it: MLA attention and its KV store, the KDA verify, the drafter's attention.
* **wide**: a decode step of more than one token tile (DSpark verify of several requests). It keeps the decode
  layout's MoE head and tail, on M-general ops.

Every other step (prefill, mixed steps, decode steps above those bounds) runs the **generic path**: the built-in Kimi
K3 text model, whose modules and ops have no catalog entries yet. `UNCERTIFIED_GENERIC_CALLS` names them. The **fused
decode path** runs the steps `decode_step` classifies on the K3 decode kernels' catalog entries. The state those
kernels share (MNNVL workspace, sandwich and MoE Lamport buffers, KDA / MLA scratch) lives in typed objects this
target creates collectively in `post_load_weights`, before any graph capture. Until those entries exist
`_fused_decode` stays None, and every step takes the generic path.

**What this target asserts rather than adapts**: SM 10.0; the topology above; bf16 weights and a bf16 KV pool;
tokens_per_block 64 (the MLA generation kernels K3's 96 heads reach exist only at 64); the V2 hybrid KV / state
manager, which holds the KDA states, with block reuse off; an all-reduce strategy of AUTO or MNNVL. The
construction-time ones fail in `__init__`, the per-engine ones on the first forward, each naming the setting.

**Text only.** The checkpoint is the vision-language wrapper. This target builds and loads no vision tower (its
weights are a predicted non-load, `weights.py`), and a step carrying multimodal input raises.

**Speculative decoding** goes through the stock one-engine shell: DSpark or DFlash with an external drafter
checkpoint, and SA. The worker and its kernels stay upstream code; this target does not own a worker.
"""

import copy
from dataclasses import dataclass
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
        "num_generations",
        "num_seqs",
        "num_tokens",
        "seq_lens",
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

# The K3 decode kernels' bounds: the token-count kernels take one tile of DECODE_MAX_TOKENS rows; the request-aware
# kernels take MAX_REQUESTS generation requests of at most MAX_TOKENS_PER_REQUEST tokens (1 + 7 drafts with DSpark).
DECODE_MAX_TOKENS = 8
MAX_REQUESTS = 8
MAX_TOKENS_PER_REQUEST = 8

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


@dataclass(frozen=True)
class DecodeStep:
    """A step the Kimi K3 decode kernels take: ``num_tokens`` rows and, on a pure decode step, ``num_requests``
    generation requests of ``tokens_per_request`` tokens each (None on a step with context requests)."""

    num_tokens: int
    num_requests: Optional[int] = None
    tokens_per_request: Optional[int] = None

    @property
    def small(self) -> bool:
        """Whether the step fits one token tile of the token-count kernels."""
        return self.num_tokens <= DECODE_MAX_TOKENS

    @property
    def decode(self) -> bool:
        """Whether the step is a pure decode step the request-aware kernels take."""
        return self.num_requests is not None

    @property
    def wide(self) -> bool:
        """Whether the step is a pure decode step of more than one token tile: its token-count work keeps the decode
        layout's MoE head and tail, on M-general ops."""
        return self.decode and not self.small


def decode_step(attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[DecodeStep]:
    """The step's shape if any Kimi K3 decode kernel takes it, else None (the generic path runs).

    ``num_tokens`` is the step's token count (the rows of the model input). Read on the host from per-step integers
    and the host copy of the sequence lengths only. A CUDA graph is captured per decode batch shape, and every input
    here is fixed by that shape, so a captured step and its replays are classified alike.
    """
    if num_tokens <= 0:
        return None
    requests = _decode_requests(attn_metadata, num_tokens)
    if requests is not None:
        return DecodeStep(num_tokens, requests, num_tokens // requests)
    if num_tokens <= DECODE_MAX_TOKENS:
        return DecodeStep(num_tokens)
    return None


def _decode_requests(attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[int]:
    """R when the step is R <= 8 generation requests of the same T <= 8 tokens and no context request, else None."""
    if attn_metadata.num_contexts != 0:
        return None
    requests = attn_metadata.num_generations
    if not 0 < requests <= MAX_REQUESTS or num_tokens % requests != 0:
        return None
    tokens = num_tokens // requests
    if not 0 < tokens <= MAX_TOKENS_PER_REQUEST:
        return None
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is not None and seq_lens.device.type == "cpu":
        lens = seq_lens[:requests]
        if lens.numel() != requests or bool((lens != tokens).any()):
            return None
    return requests


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
    assert topology == (16, 16, 1, 4, 4, False), (
        "the tp16_moetp4ep4 target needs world_size 16, tensor_parallel_size 16, pipeline_parallel_size 1, "
        "moe_tensor_parallel_size 4, moe_expert_parallel_size 4 and enable_attention_dp false; the engine built "
        f"(world, tp, pp, moe_tp, moe_ep, attention_dp) = {topology}"
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


@register_auto_model("ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4")
class ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4(KimiLinearForCausalLM):
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
        if self._fused_decode is not None:
            rows = input_ids if input_ids is not None else inputs_embeds
            step = None if rows is None else decode_step(attn_metadata, rows.shape[0])
            if step is not None:
                return self._fused_decode(
                    step,
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
