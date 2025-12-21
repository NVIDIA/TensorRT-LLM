from typing import List, Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gpt_oss import TransformerBlock
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from .runner_interface import RunnerBase
from .runner_utils import RunnerMixin, round_up


class GptOss120BRunner(RunnerMixin, RunnerBase):
    @staticmethod
    def has_mamba_metadata() -> bool:
        return False

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
        use_low_precision_moe_combine: bool,
        use_cuda_graph: bool,
    ):
        super().__init__()
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
            use_low_precision_moe_combine=use_low_precision_moe_combine,
            skip_create_weights_in_init=True,
        )
        pretrained_config = self.model_config.pretrained_config

        if pretrained_config.torch_dtype is None:
            pretrained_config.torch_dtype = torch.bfloat16

        with self.scaled_from_ctx(scaled_from, mapping, pretrained_config):
            layers = [
                TransformerBlock(
                    config=self.model_config,
                    layer_idx=layer_idx,
                )
                for layer_idx in layer_indices
            ]
            next_layer_layernorm = RMSNorm(
                hidden_size=pretrained_config.hidden_size,
                eps=pretrained_config.rms_norm_eps,
                dtype=pretrained_config.torch_dtype,
            )

            params_map_reverse = {"qkv": "qkv_proj", "out": "o_proj", "unembedding": "lm_head"}
            if self.model_config.quant_config.exclude_modules:
                for i, module in enumerate(self.model_config.quant_config.exclude_modules):
                    names = module.split(".")
                    if names[-1] in params_map_reverse:
                        names[-1] = params_map_reverse[names[-1]]
                    prefix = [] if names[0] == "model" else ["model"]
                    self.model_config.quant_config.exclude_modules[i] = '.'.join(prefix +names)

            # TODO: apply_layerwise_quant_config
            self.apply_quant_config_exclude_modules(layers, self.model_config.quant_config, prefix="block")
            for layer in layers:
                for module in layer.modules():
                    if callable(getattr(module, "create_weights", None)):
                        module.create_weights()
                layer.cuda()
                initialize_dummy_weights(layer)
                for module in layer.modules():
                    if hasattr(module, "post_load_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.post_load_weights()
            next_layer_layernorm.cuda()
            initialize_dummy_weights(next_layer_layernorm)

            # Link next_layer_layernorm: each layer's next_layer_layernorm points to
            # the next layer's input_layernorm
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layer.next_layer_layernorm = next_layer.input_layernorm
            layers[-1].next_layer_layernorm = next_layer_layernorm

            self.layers = layers

