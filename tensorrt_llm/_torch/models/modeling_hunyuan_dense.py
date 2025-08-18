import os
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import HunYuanPretrainedConfig, ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             duplicate_kv_weight, register_auto_model)


def _get_tensor_stats(tensor):
    dump_tensor_num = int(os.environ.get("DUMP_TENSOR_NUM", "10"))
    """获取张量的统计信息：前N个值、均值和方差"""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None, None, None, None, None

    flat = tensor.reshape(-1)
    length = len(flat)
    if len(flat) == 0:
        return [], [], [], None, None

    dump_tensor_num = min(dump_tensor_num, len(flat))
    first_values = flat[:dump_tensor_num].detach().cpu().tolist()

    # 获取中间n个值
    mid_start = max(0, (length - dump_tensor_num) // 2)
    middle_values = flat[mid_start:mid_start +
                         dump_tensor_num].detach().cpu().tolist()

    # 获取最后n个值
    last_values = flat[-dump_tensor_num:].detach().cpu().tolist()

    # 修复：对于整数类型张量，转换为浮点类型再计算统计量
    try:
        if flat.dtype in [
                torch.long, torch.int, torch.int32, torch.int64, torch.short,
                torch.int8
        ]:
            # 整数类型：转换为float计算，或者跳过统计计算
            flat_float = flat.float()
            mean = float(torch.mean(flat_float).item())
            var = float(torch.var(flat_float).item())
        else:
            # 浮点类型：直接计算
            mean = float(torch.mean(flat).item())
            var = float(torch.var(flat).item())
    except Exception as e:
        # 如果计算失败，返回None
        print(
            f"Warning: Could not compute statistics for tensor with dtype {flat.dtype}: {e}"
        )
        mean, var = None, None

    return first_values, middle_values, last_values, mean, var


class HunYuanTRTRotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 config: HunYuanPretrainedConfig,
                 device: Optional[torch.device] = None):
        "default" if config.rope_scaling is None else 'hunyuan'
        super().__init__(
            config,
            head_dim=config.hidden_size // config.num_attention_heads,
        )


class HunYuanAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[HunYuanPretrainedConfig],
        layer_idx: Optional[int] = None,
        nope_layer: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config

        self.use_rope = not nope_layer
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(config),
            is_neox=True,
        ) if self.use_rope else None

        self.use_qk_norm = config.use_qk_norm
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            # rotary_emb=HunYuanTRTRotaryEmbedding(config),
            # pos_embd_params=None,
            pos_embd_params=pos_embd_params,
            rope_fusion=not self.use_qk_norm,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )
        if self.use_qk_norm:
            if config.head_dim != None:
                self.head_dim = config.head_dim
            else:
                self.head_dim = config.hidden_size // config.num_attention_heads
            self.query_layernorm = RMSNorm(hidden_size=self.head_dim,
                                           eps=config.rms_norm_eps,
                                           dtype=config.torch_dtype)
            self.key_layernorm = RMSNorm(hidden_size=self.head_dim,
                                         eps=config.rms_norm_eps,
                                         dtype=config.torch_dtype)
            self.aux_stream = aux_stream
            self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        q, k, v = self.split_qkv(q, k, v)
        if position_ids is not None:
            q, k, v = super().apply_rope(q, k, v, position_ids)
        # Llama4 applies QK norm after RoPE.
        if self.use_qk_norm:
            q, k = self.apply_qk_norm(q, k)

        return q, k, v

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.query_layernorm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.key_layernorm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert lora_params is None, "LORA is not supported for HunYuanAttention"
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            mrope_config=mrope_config,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            **kwargs,
        )


class HunYuanMLP(GatedMLP):
    """
    继承自 GatedMLP 的 HunYuanMLP 实现, 兼容混合专家和tp
    """

    def __init__(self,
                 model_config: ModelConfig[HunYuanPretrainedConfig],
                 layer_idx: Optional[int] = None,
                 is_shared_mlp=False,
                 is_moe=False):
        hidden_size = model_config.hidden_size
        intermediate_size = model_config.intermediate_size

        if is_moe:
            # 优先使用 moe_intermediate_size
            if model_config.moe_intermediate_size is not None:
                intermediate_size = model_config.moe_intermediate_size if isinstance(
                    model_config.moe_intermediate_size,
                    int) else model_config.moe_intermediate_size[layer_idx]
        elif is_shared_mlp:  # gongxiang
            num_shared_expert = model_config.num_shared_expert if isinstance(
                model_config.num_shared_expert,
                int) else model_config.num_shared_expert[layer_idx]
            intermediate_size *= num_shared_expert

        if model_config.hidden_act == "silu":
            activation = F.silu
        else:
            raise ValueError(
                f"Unsupported activation: {model_config.hidden_act}")

        super().__init__(hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         bias=getattr(model_config, "mlp_bias", False),
                         activation=activation,
                         dtype=getattr(model_config, "torch_dtype",
                                       torch.bfloat16),
                         config=model_config,
                         overridden_tp_size=1,
                         is_expert=is_moe)

    def forward(self, x):
        return super().forward(x)


class HunYuanDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[HunYuanPretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        # attention
        self.self_attn = HunYuanAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            bias=config.mlp_bias,
                            dtype=config.torch_dtype,
                            config=model_config)

        norm_type = getattr(config, 'norm_type', 'rms')
        if norm_type == 'hf_rms' or norm_type == 'rms':
            self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                           eps=config.rms_norm_eps,
                                           dtype=config.torch_dtype)
            self.post_attention_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype)
        elif norm_type == 'fused' or norm_type == 'torch_nn':
            self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        # Fully Connected
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_hunyuan=True)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[HunYuanPretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.aux_stream_dict = {
            key: torch.cuda.Stream()
            for key in [
                AuxStreamType.Attention, AuxStreamType.MoeShared,
                AuxStreamType.MoeChunkingOverlap
            ]
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList([
            HunYuanDecoderLayer(model_config, layer_idx, self.aux_stream_dict)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            kwargs['layer_idx'] = layer_idx
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("HunYuanDenseV1ForCausalLM")
class HunYuanDenseV1ForCausalLM(DecoderModelForCausalLM[HunYuanModel,
                                                        HunYuanPretrainedConfig]
                                ):

    def __init__(self, model_config: ModelConfig[HunYuanPretrainedConfig]):
        super().__init__(HunYuanModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
        self._execution_stats = None

    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        }
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue
                names = name.split('.')
                if names[-1] in params_map:
                    # model.layers.{idx}.mlp.shared_mlp.gate_up_proj or model.layers.{idx}.self_attn.qkv_proj
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    num_kv_heads=v[:].shape[0] // head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    name = name.replace('gate', 'gate.wg')
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        # model.layers.{idx}.self_attn.o_proj or model.layers.{idx}.mlp.shared_mlp.down_proj
                        # or model.layers.{idx}.mlp.experts.gate
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        ###### DEBUG #####
        dump_decode_step = int(os.environ.get("DUMP_DECODE_STEP", "-1"))
        if dump_decode_step > 0:
            import json
            import time

            def _initialize_dump_trackers(self):
                # called once per session
                self.dump_modules_file = os.path.join(
                    os.environ.get(
                        "TRT_MODEL_DUMP_DIR",
                        "/apdcephfs_jn/share_302216743/jiarunliu/trt_model_dumps"
                    ),
                    f"trt_modules_{time.strftime('%Y%m%d_%H%M%S')}_{id(self):x}.jsonl"
                )
                os.makedirs(os.path.dirname(self.dump_modules_file),
                            exist_ok=True)
                open(self.dump_modules_file, 'w').close()
                self.hooks = []

            if not hasattr(self, 'dump_modules_file'):
                _initialize_dump_trackers(self)

            self.decode_step = getattr(self, "decode_step", 0) + 1
            exec_list = []

            if self.decode_step == dump_decode_step + 2:
                for name, module in self.named_modules():
                    if not list(module.children()):

                        def make_hook(mod_name):

                            def hook(mod, inp, out):
                                cls_name = mod.__class__.__name__

                                input_shape = None
                                input_values = None
                                input_mid_values = None
                                input_back_values = None
                                input_mean = None
                                input_var = None
                                dump_module_input = os.environ.get(
                                    "DUMP_MODULE_INPUT", "o_proj")
                                if dump_module_input in mod_name:
                                    input_shape = list(inp[0].shape)
                                    input_values, input_mid_values, input_back_values, input_mean, input_var = _get_tensor_stats(
                                        inp[0])

                                if isinstance(out, torch.Tensor):
                                    shape = list(out.shape)
                                    output_values, output_mid_values, output_back_values, output_mean, output_var = _get_tensor_stats(
                                        out)
                                elif isinstance(
                                        out,
                                    (tuple,
                                     list)) and len(out) > 0 and isinstance(
                                         out[0], torch.Tensor):
                                    shape = list(out[0].shape)
                                    output_values, output_mid_values, output_back_values, output_mean, output_var = _get_tensor_stats(
                                        out)
                                else:
                                    shape = None
                                    output_values = None
                                    output_mean = None
                                    output_var = None

                                weights = {}
                                weight_output_mean = 0.0
                                weight_output_var = 0.0
                                for param_name, param in mod.named_parameters(
                                        recurse=False):
                                    weights[
                                        param_name], weight_output_mid_values, weight_output_back_values, weight_output_mean, weight_output_var = _get_tensor_stats(
                                            param)

                                if input_shape is None:
                                    exec_list.append({
                                        "name": mod_name,
                                        "shape": shape,
                                        "cls_name": cls_name,
                                        "output_values": output_values,
                                        "output_middle_values":
                                        output_mid_values,
                                        "output_last_values":
                                        output_back_values,
                                        "output_stats": {
                                            "mean": output_mean,
                                            "var": output_var
                                        },
                                        "weights": weights,
                                        "weights_stats": {
                                            "mean": weight_output_mean,
                                            "var": weight_output_var
                                        },
                                    })
                                else:
                                    exec_list.append({
                                        "name": mod_name,
                                        "shape": shape,
                                        "cls_name": cls_name,
                                        "input_shape": input_shape,
                                        "input_values": input_values,
                                        "input_middle_values": input_mid_values,
                                        "input_last_values": input_back_values,
                                        "input_stats": {
                                            "mean": input_mean,
                                            "var": input_var
                                        },
                                        "output_values": output_values,
                                        "output_middle_values":
                                        output_mid_values,
                                        "output_last_values":
                                        output_back_values,
                                        "output_stats": {
                                            "mean": output_mean,
                                            "var": output_var
                                        },
                                        "weights": weights,
                                        "weights_stats": {
                                            "mean": weight_output_mean,
                                            "var": weight_output_var
                                        },
                                    })

                            return hook

                        self.hooks.append(
                            module.register_forward_hook(make_hook(name)))

            elif self.decode_step == dump_decode_step + 3:
                if hasattr(self, 'hooks'):
                    for hook in self.hooks:
                        hook.remove()
                    self.hooks = []
                    print(f"已完成第{self.decode_step - 3}次decode的数据收集，hook已移除")
        ###############
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        #### Save Debug Info ####
        if dump_decode_step > 0:
            if self.decode_step == dump_decode_step + 2:
                record = {
                    "timestamp": time.strftime("%Y%m%d_%H%M%S.%f")[:-3],
                    "decode_step": self.decode_step - 2,
                    "modules": exec_list
                }
                with open(self.dump_modules_file, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False, indent=2))
                    f.write("\n")
                print(
                    f"第{self.decode_step - 2}次decode的模块信息已dump到: {self.dump_modules_file}"
                )
        #### Save Debug Info ####

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
