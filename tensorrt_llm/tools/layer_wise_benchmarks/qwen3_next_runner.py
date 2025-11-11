from typing import List, Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3_next import ALL_DECODER_LAYER_TYPES
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from .runner_interface import RunnerBase
from .runner_utils import RunnerMixin


class Qwen3NextRunner(RunnerMixin, RunnerBase):
    @staticmethod
    def has_mamba_metadata() -> bool:
        return True

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        mapping: Mapping,
        *,
        moe_backend: str,
        layer_indices: List[int],
        scaled_from: Optional[int],
        max_seq_len: int,
        max_num_tokens: int,
        moe_max_num_tokens: int,
        use_cuda_graph: bool,
    ):
        self.model_config = ModelConfig.from_pretrained(
            pretrained_model_name_or_path,
            mapping=mapping,
            enable_min_latency=False,
            use_cuda_graph=use_cuda_graph,
            force_dynamic_quantization=False,
            spec_config=None,
            sparse_attention_config=None,  # To be loaded from config
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            moe_max_num_tokens=moe_max_num_tokens,
            moe_load_balancer=None,
            lora_config=None,
            allreduce_strategy=AllReduceStrategy.AUTO,
            mm_encoder_only=False,
            attn_backend="TRTLLM",
            moe_backend=moe_backend,
            moe_disable_finalize_fusion=False,
            use_low_precision_moe_combine=False,
            skip_create_weights_in_init=True,
        )
        pretrained_config = self.model_config.pretrained_config

        with self.scaled_from_ctx(scaled_from, mapping, pretrained_config):
            aux_stream = torch.cuda.Stream()
            layers = [
                ALL_DECODER_LAYER_TYPES[pretrained_config.layer_types[layer_idx]](
                    self.model_config,
                    layer_idx,
                    aux_stream,
                )
                for layer_idx in layer_indices
            ]
            next_layer_layernorm = RMSNorm(
                hidden_size=pretrained_config.hidden_size,
                eps=pretrained_config.rms_norm_eps,
                dtype=pretrained_config.torch_dtype,
                use_gemma=True,
            )

            # TODO: apply_layerwise_quant_config
            self.apply_quant_config_exclude_modules(layers, self.model_config.quant_config)
            for layer in layers:
                for module in layer.modules():
                    if callable(getattr(module, "create_weights", None)):
                        module.create_weights()
                layer.cuda()
                for module in layer.modules():
                    if hasattr(module, "post_load_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.post_load_weights()
            next_layer_layernorm.cuda()
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layer.next_layer_layernorm = next_layer.input_layernorm
            layers[-1].next_layer_layernorm = next_layer_layernorm

            self.layers = layers
