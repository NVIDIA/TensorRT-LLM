# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --------------------------------------------------
# Portions of this code were derived from DeepSeek‑V3:
#   https://github.com/deepseek-ai/DeepSeek-V3
#
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------

"""GLM-5.2 MTP layers, head, and target-local draft construction."""

from typing import Dict, Iterator, List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams, allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    _copy_model_config_with_moe_backend,
    _enable_trtllm_moe_preload_for_draft,
    _get_mtp_moe_quant_config,
)
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModelForCausalLM,
    apply_layerwise_quant_config,
    apply_quant_config_exclude_modules,
)
from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer_set_repeated_for_next_layer
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import AuxStreamType, EventType, create_lm_head_tp_mapping

from ._model import Glm52DecoderLayer


class Glm52MTPHead(nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config

        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.mapping_lm_head_tp = None

    @torch.compile(options={"max-autotune": True})
    def get_last_token_states(self, hidden_states, attn_metadata):
        last_tokens = (
            torch.cumsum(
                attn_metadata.seq_lens_cuda,
                dim=0,
                dtype=torch.long,
            )
            - 1
        )
        return hidden_states[last_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = self.get_last_token_states(hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)

        enable_attention_dp = self.model_config.mapping.enable_attention_dp
        enable_lm_head_tp_in_adp = (
            enable_attention_dp and self.model_config.mapping.enable_lm_head_tp_in_adp
        )

        # Add pre-lm gather logic
        if enable_lm_head_tp_in_adp:
            # ADP + LM TP mode: perform All-Gather before LM_head
            self.mapping_lm_head_tp = create_lm_head_tp_mapping(
                self.model_config.mapping, hidden_states.shape[0]
            )
            hidden_states = allgather(hidden_states, self.mapping_lm_head_tp, dim=0)

        # Temporarily disable gather_output when not in ADP mode or (in ADP mode and LM TP is enabled)
        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = False
        logits = lm_head(
            hidden_states, mapping_lm_head_tp=self.mapping_lm_head_tp, is_spec_decoding_head=True
        )
        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = True
        return logits


class Glm52MTP(Glm52DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
    ):
        super().__init__(model_config, layer_idx, aux_stream_dict, is_separate_draft_engine)
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

        self.enorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.hnorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        if model_config.mapping.enable_attention_dp:
            self.eh_proj = Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
            )
        else:
            self.eh_proj = Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
            )

        self.shared_head = Glm52MTPHead(model_config)

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        def norm_embeds():
            return self.enorm(embed_tokens(input_ids))  # emdedding

        def norm_hidden():
            return self.hnorm(hidden_states)

        inputs_embeds, hidden_states = maybe_execute_in_parallel(
            norm_embeds,
            norm_hidden,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
            disable_on_compile=True,
        )
        hidden_states = torch.concat([inputs_embeds, hidden_states], dim=-1)
        # Split hidden_states columnwise based on TP
        tp_size = self.model_config.mapping.tp_size
        tp_rank = self.model_config.mapping.tp_rank

        if tp_size > 1 and not (self.model_config.mapping.enable_attention_dp):
            hidden_states = torch.chunk(hidden_states, tp_size, dim=-1)[tp_rank]
        hidden_states = self.eh_proj(hidden_states)

        # Input layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        # MTP Layer Must have sparse MOE
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ),
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                )
            ),
        )

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.shared_head.norm.weight,
                    eps=self.shared_head.norm.variance_epsilon,
                ),
            )
        else:
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)

        # It's for 2-model path, capture the hidden states
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)

        return hidden_states


class Glm52MTPForCausalLM(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
        lm_head: nn.Module = None,
        model: nn.Module = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.start_layer_idx = start_layer_idx
        spec_config = model_config.spec_config
        mtp_layer = Glm52MTP
        lm_head_quant_config = None

        if spec_config.uses_replacement_heads and getattr(mtp_layer, "uses_shared_lm_head", True):
            if model_config.quant_config_dict is not None:
                lm_head_quant_config = model_config.quant_config_dict.get("lm_head")
            elif (
                model_config.quant_config.quant_algo is not None
                and DecoderModelForCausalLM._checkpoint_has_lm_head_scale(
                    model_config, checkpoint_dir=str(spec_config.speculative_model)
                )
            ):
                lm_head_quant_config = model_config.quant_config
            if model_config.quant_config.is_module_excluded_from_quantization("lm_head"):
                lm_head_quant_config = None

        spec_dec_mode = spec_config.spec_dec_mode
        assert spec_dec_mode.is_mtp_one_model()
        checkpoint_mtp_num_layers = model_config.pretrained_config.num_nextn_predict_layers
        if spec_dec_mode.is_mtp_eagle_one_model():
            mtp_num_layers = 1
            mtp_repeat_count = spec_config.max_draft_len
        else:
            mtp_num_layers = min(spec_config.max_draft_len, checkpoint_mtp_num_layers)
            mtp_repeat_count = 1

        moe_load_balancer_set_repeated_for_next_layer(mtp_repeat_count)

        self.mtp_layers = nn.ModuleList(
            [
                mtp_layer(model_config, layer_idx + start_layer_idx, model.aux_stream_dict)
                for layer_idx in range(mtp_num_layers)
            ]
        )
        self.owns_lm_head = lm_head_quant_config is not None
        if self.owns_lm_head:
            if getattr(model_config.pretrained_config, "tie_word_embeddings", False):
                raise ValueError("A quantized replacement MTP head cannot tie embeddings")
            if (
                model_config.mapping.enable_attention_dp
                and model_config.mapping.enable_lm_head_tp_in_adp
            ):
                raise ValueError(
                    "Quantized replacement MTP LM heads require enable_lm_head_tp_in_adp=False"
                )
            self.lm_head = LMHead(
                model_config.pretrained_config.vocab_size,
                model_config.pretrained_config.hidden_size,
                dtype=model_config.pretrained_config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=model_config.lm_head_gather_output,
                reduce_output=False,
                use_custom_cublas_mm=getattr(lm_head, "use_custom_cublas_mm", False),
                quant_config=lm_head_quant_config,
            )
        else:
            self.lm_head = lm_head
        self.embed_tokens = model.embed_tokens

    def _quantization_named_modules(self) -> Iterator[tuple[str, nn.Module]]:
        for name, module in self.named_modules():
            if name.startswith("mtp_layers."):
                _, layer_idx, *suffix = name.split(".")
                prefix = f"model.layers.{self.start_layer_idx + int(layer_idx)}"
                yield ".".join([prefix, *suffix]), module
            elif self.owns_lm_head and (name == "lm_head" or name.startswith("lm_head.")):
                yield name, module

    def apply_quant_config(self) -> None:
        apply_layerwise_quant_config(self.model_config, self._quantization_named_modules())
        apply_quant_config_exclude_modules(self.model_config, self._quantization_named_modules())


def build_mtp_draft(model_config, draft_config, lm_head, model):
    """Build the one-model MTP drafter (vanilla MTP and MTP-Eagle share it)."""
    mtp_model_config = draft_config if draft_config is not None else model_config
    requested_moe_backend = getattr(model_config.spec_config, "moe_backend", None)
    if requested_moe_backend is not None:
        start_layer_idx = model_config.pretrained_config.num_hidden_layers
        mtp_model_config = _copy_model_config_with_moe_backend(
            mtp_model_config,
            requested_moe_backend,
            quant_config=_get_mtp_moe_quant_config(mtp_model_config, start_layer_idx),
        )
        _enable_trtllm_moe_preload_for_draft(model, mtp_model_config.moe_backend)
    return Glm52MTPForCausalLM(
        mtp_model_config, model_config.pretrained_config.num_hidden_layers, lm_head, model
    )
