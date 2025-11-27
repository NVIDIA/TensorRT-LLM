from typing import Dict, Optional

import torch
from torch import nn
from transformers import BertConfig

from tensorrt_llm.llmapi.utils import logger_debug
from tensorrt_llm.logger import logger

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear
from .modeling_utils import register_auto_model


class MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        """
        A simple two-layer Multi-Layer Perceptron (MLP) with GELU activation.
        """
        super(MLP, self).__init__()

        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass for inference.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings,
                                             config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size,
                                               config.hidden_size)
        self.embedding_ln = nn.LayerNorm(config.hidden_size,
                                         eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.IntTensor,
        token_type_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
    ):
        assert input_ids is not None
        x = self.word_embeddings(input_ids)
        assert x is not None
        x = x + self.position_embeddings(position_ids)
        assert x is not None
        x = x + self.token_type_embeddings(token_type_ids)
        x = self.embedding_ln(x)
        return x


class BertAttention(Attention):

    def __init__(self,
                 model_config: ModelConfig[BertConfig],
                 layer_idx: Optional[int] = None):
        config = model_config.pretrained_config
        pos_embd_params = None
        bias = True
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class BertEncoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[BertConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.self_attn = BertAttention(model_config=model_config,
                                       layer_idx=layer_idx)

        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps,
                                            dtype=config.torch_dtype)
        self.post_layernorm = nn.LayerNorm(config.hidden_size,
                                           eps=config.layer_norm_eps,
                                           dtype=config.torch_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            attention_mask=PredefinedAttentionMask.FULL,
        )

        hidden_states = residual + attention_output

        hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


class BertPooler(nn.Module):
    """from huggingface code"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if attn_metadata is not None:
            #NOTE: select the first tokens
            offset = attn_metadata.seq_lens_cuda
            selected_tokens = torch.cumsum(
                attn_metadata.seq_lens_cuda,
                dim=0,
                dtype=torch.long,
            ) - offset
            hidden_states = hidden_states[selected_tokens]
        else:
            # hidden_states: [B, N, H]
            hidden_states = hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):

    def __init__(self,
                 model_config: ModelConfig[BertConfig],
                 add_pooling_layer=True):
        super().__init__()
        self.dype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config

        config = self.model_config.pretrained_config
        self.add_pooling_layer = add_pooling_layer

        self.embedding = BertEmbeddings(config=config)
        self.layers = nn.ModuleList([
            BertEncoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.pooler = BertPooler(config) if self.add_pooling_layer else None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):

        hidden_states = self.embedding(input_ids, token_type_ids, position_ids)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attn_metadata)

        output = hidden_states
        if self.pooler is not None:
            output = self.pooler(hidden_states, attn_metadata)

        return output


@register_auto_model("BertForSequenceClassification")
class BertForSequenceClassification(nn.Module):

    def __init__(self, model_config: ModelConfig[BertConfig]):
        super().__init__()
        self.model_config = model_config
        config = self.model_config.pretrained_config
        self.model = BertModel(self.model_config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.create_weights()

    def create_weights(self):
        for _, module in self.named_modules():
            if callable(getattr(module, "_create_weights", None)):
                module._create_weights()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input=input_ids)
        position_ids = position_ids.flatten(start_dim=0)
        assert input_ids.shape == token_type_ids.shape and token_type_ids.shape == position_ids.shape, \
        f"Shapes are not the same! input_ids.shape={input_ids.shape},\ntoken_type_ids.shape={token_type_ids.shape},\nposition_ids.shape={position_ids.shape}"
        pooled_output = self.model(attn_metadata=attn_metadata,
                                   input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)
        logits = self.classifier(pooled_output)
        logits = logits.float()
        return logits

    @property
    def config(self):
        """
        We need to set this property because BertForSequenceClassification doesn't inherit from class DecoderModelForCausalLM in tensorrt_llm/models/modeling_utils.py
        This can be improved in the future.
        """
        # WAR because BertConfig in HF doesn't have some fields
        num_attention_heads = self.model_config.pretrained_config.num_attention_heads
        # kv head is same as num_attention_heads
        setattr(self.model_config.pretrained_config, "num_key_value_heads",
                num_attention_heads)
        logger.info(
            f"Set num_key_value_heads={self.model_config.pretrained_config.num_key_value_heads} for {type(self)} config"
        )
        # if pretrained_config torch_dtype is None or not set, set it to float32
        if not hasattr(
                self.model_config.pretrained_config, "torch_dtype"
        ) or self.model_config.pretrained_config.torch_dtype is None:
            self.model_config.pretrained_config.torch_dtype = torch.float32

        return self.model_config.pretrained_config

    def infer_max_seq_len(self) -> int:
        """
        We need to define this method because BertForSequenceClassification doesn't inherit from class DecoderModelForCausalLM in tensorrt_llm/models/modeling_utils.py
        This can be improved in the future.
        """
        inferred_max_seq_len = 2048
        if getattr(self.config, 'max_position_embeddings', None) is not None:
            inferred_max_seq_len = self.config.max_position_embeddings

        return inferred_max_seq_len

    def load_weights(self, weights: Dict):
        """
        We need to define this method because BertForSequenceClassification doesn't inherit from class DecoderModelForCausalLM in tensorrt_llm/models/modeling_utils.py
        This can be improved in the future.
        """

        #NOTE: the structure of BERT is old style, so we use customized func to mapping weights
        tllm_weights = convert_bert_to_tllm(
            weights,
            num_layers=self.model_config.pretrained_config.num_hidden_layers)
        loaded_weight = set()

        for name, module in self.named_modules():
            if len(module._parameters) > 0:
                logger_debug(f"loading for: {name}")
                try:
                    # the provided modules in TRTLLM
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=tllm_weights[name])
                    else:
                        logger_debug(f" use copy_ to load {name}")
                        module_weights = tllm_weights[name][0]
                        for n, p in module._parameters.items():
                            if p is not None:
                                weight = module_weights[n][:]
                                p.data.copy_(weight)

                except Exception as e:
                    raise e
            loaded_weight.add(name)
        # verify whether all the weights are loaded
        not_loaded_weights = set(tllm_weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

        self.half()


def convert_bert_to_tllm(bert_state_dict, num_layers):
    """
    Convert a standard BERT-style state_dict to a TLLM-style state_dict.
    """
    tllm_state_dict = {}

    # ----- 1) Embeddings -----
    # BERT -> TLLM

    tllm_state_dict['model.embedding.word_embeddings'] = [{
        'weight':
        bert_state_dict['bert.embeddings.word_embeddings.weight']
    }]
    tllm_state_dict['model.embedding.position_embeddings'] = [{
        'weight':
        bert_state_dict['bert.embeddings.position_embeddings.weight']
    }]
    tllm_state_dict['model.embedding.token_type_embeddings'] = [{
        'weight':
        bert_state_dict['bert.embeddings.token_type_embeddings.weight']
    }]

    # BERT embedding LayerNorm has both weight and bias
    tllm_state_dict['model.embedding.embedding_ln'] = [{
        'weight':
        bert_state_dict['bert.embeddings.LayerNorm.weight'],
        'bias':
        bert_state_dict['bert.embeddings.LayerNorm.bias']
    }]

    # ----- 2) Map each Transformer layer -----
    for i in range(num_layers):
        # (a) QKV: BERT has separate query/key/value
        q_w = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.self.query.weight']
        q_b = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.self.query.bias']
        q = {'weight': q_w, 'bias': q_b}

        k_w = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.self.key.weight']
        k_b = bert_state_dict[f'bert.encoder.layer.{i}.attention.self.key.bias']
        k = {'weight': k_w, 'bias': k_b}

        v_w = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.self.value.weight']
        v_b = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.self.value.bias']
        v = {'weight': v_w, 'bias': v_b}
        res = [q, k, v]

        tllm_state_dict[f'model.layers.{i}.self_attn.qkv_proj'] = res

        # (b) Output projection in attention

        o_w = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.output.dense.weight']
        o_b = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.output.dense.bias']
        tllm_state_dict[f'model.layers.{i}.self_attn.o_proj'] = [{
            'weight': o_w,
            'bias': o_b
        }]

        # (c) LayerNorm(s)
        # In BERT:
        #   - `attention.output.LayerNorm` normalizes after attn
        #   - `output.LayerNorm` normalizes after feed-forward
        # In TLLM:
        #   - `input_layernorm` (before attention)
        #   - `post_layernorm` (after attention, before MLP), etc.

        ln_in_w = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight']
        ln_in_b = bert_state_dict[
            f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias']
        tllm_state_dict[f'model.layers.{i}.input_layernorm'] = [{
            'weight': ln_in_w,
            'bias': ln_in_b
        }]

        ln_out_w = bert_state_dict[
            f'bert.encoder.layer.{i}.output.LayerNorm.weight']
        ln_out_b = bert_state_dict[
            f'bert.encoder.layer.{i}.output.LayerNorm.bias']
        tllm_state_dict[f'model.layers.{i}.post_layernorm'] = [{
            'weight': ln_out_w,
            'bias': ln_out_b
        }]

        # (d) MLP in BERT:
        #   intermediate.dense --> feed-forward up-projection
        #   output.dense       --> feed-forward down-projection
        # TLLM uses "gate_up_proj" + "down_proj".  Very often, TLLM’s MLP is bigger or “gated.”
        # In a *very naive* 1:1 mapping:

        fc1_w = bert_state_dict[
            f'bert.encoder.layer.{i}.intermediate.dense.weight']
        fc1_b = bert_state_dict[
            f'bert.encoder.layer.{i}.intermediate.dense.bias']
        tllm_state_dict[f'model.layers.{i}.mlp.fc1'] = [{
            'weight': fc1_w,
            'bias': fc1_b
        }]

        fc2_w = bert_state_dict[f'bert.encoder.layer.{i}.output.dense.weight']
        fc2_b = bert_state_dict[f'bert.encoder.layer.{i}.output.dense.bias']
        tllm_state_dict[f'model.layers.{i}.mlp.fc2'] = [{
            'weight': fc2_w,
            'bias': fc2_b
        }]

    # ----- 3) Pooler (if present in TLLM) -----

    if 'bert.pooler.dense.weight' in bert_state_dict:
        pooler_w = bert_state_dict['bert.pooler.dense.weight']
        pooler_b = bert_state_dict['bert.pooler.dense.bias']
        tllm_state_dict['model.pooler.dense'] = [{
            'weight': pooler_w,
            'bias': pooler_b
        }]

    # ----- 4) Classifier (if present) -----
    cls_w = bert_state_dict['classifier.weight']
    cls_b = bert_state_dict['classifier.bias']
    tllm_state_dict['classifier'] = [{'weight': cls_w, 'bias': cls_b}]

    return tllm_state_dict
