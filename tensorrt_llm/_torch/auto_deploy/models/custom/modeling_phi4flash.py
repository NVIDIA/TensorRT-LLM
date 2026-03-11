# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Slimmed down PyTorch Phi-4 Flash model implementation for auto_deploy export.

Source:
https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config class (Phi4FlashConfig) for transformers compatibility
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility (torch_attention,
  torch_causal_conv1d, torch_ssm)
* Replaces flash-attention / mamba custom kernels with PyTorch reference ops
* Removed training-only code paths and dropout
* Removed attention masks from the exported entry point; AD manages causal masking
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class Phi4FlashConfig(PretrainedConfig):
    """Configuration for Phi-4-mini-flash-reasoning."""

    model_type = "phi4flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 51200,
        hidden_size: int = 2560,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 40,
        num_key_value_heads: Optional[int] = 4,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        sliding_window: int = 2047,
        mb_per_layer: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: str | int = "auto",
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        mlp_bias: bool = False,
        lm_head_bias: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.mb_per_layer = mb_per_layer
        self.sliding_window = [
            sliding_window if layer_idx < num_hidden_layers // 2 and layer_idx % 2 == 1 else None
            for layer_idx in range(num_hidden_layers)
        ]
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = (
            math.ceil(hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        )
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mlp_bias = mlp_bias
        self.lm_head_bias = lm_head_bias
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        layer_block_types = []
        for idx in range(self.num_hidden_layers):
            if idx % 2 == 1:
                layer_block_type = (
                    "attention" if idx <= (self.num_hidden_layers // 2 + 1) else "shared_attention"
                )
            else:
                layer_block_type = "mamba"
            layer_block_types.append(layer_block_type)
        return layer_block_types


try:
    AutoConfig.register("phi4flash", Phi4FlashConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("phi4flash", Phi4FlashConfig)
    except ValueError:
        pass


def _lambda_init_fn(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def _swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return value * ACT2FN["silu"](gate)


def _split_heads(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.view(*x.shape[:-2], x.shape[-2] // 2, 2, x.shape[-1])
    return x[..., 0, :], x[..., 1, :]


class Phi4FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Phi4FlashMLP(nn.Module):
    def __init__(self, config: Phi4FlashConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        y = self.fc1(hidden_states)
        gate, up = y.chunk(2, dim=-1)
        up = up * self.activation_fn(gate)
        return self.fc2(up)


class Phi4FlashDiffAttention(nn.Module):
    def __init__(self, head_dim: int, depth: int):
        super().__init__()
        self.head_dim = head_dim
        self.lambda_init = _lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.subln = Phi4FlashRMSNorm(2 * head_dim, eps=1e-5)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        q1, q2 = _split_heads(q)
        k1, k2 = _split_heads(k)
        v1, v2 = _split_heads(v)

        attn11 = torch.ops.auto_deploy.torch_attention(
            q1, k1, v1, is_causal=True, layout="bsnd", sliding_window=sliding_window
        )
        attn12 = torch.ops.auto_deploy.torch_attention(
            q1, k1, v2, is_causal=True, layout="bsnd", sliding_window=sliding_window
        )
        attn21 = torch.ops.auto_deploy.torch_attention(
            q2, k2, v1, is_causal=True, layout="bsnd", sliding_window=sliding_window
        )
        attn22 = torch.ops.auto_deploy.torch_attention(
            q2, k2, v2, is_causal=True, layout="bsnd", sliding_window=sliding_window
        )

        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        return attn.view(*attn.shape[:-2], attn.shape[-2] * 2, self.head_dim)


class Phi4FlashAttention(nn.Module):
    def __init__(self, config: Phi4FlashConfig, layer_idx: int, yoco_cross: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.yoco_cross = yoco_cross

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        if yoco_cross:
            self.Wqkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=True)
        self.inner_cross_attn = Phi4FlashDiffAttention(self.head_dim, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        if self.yoco_cross:
            query_states = self.Wqkv(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
            key_states, value_states = yoco_key_values
            sliding_window = None
        else:
            qkv = self.Wqkv(hidden_states)
            query_pos = self.num_heads * self.head_dim
            kv_pos = self.num_key_value_heads * self.head_dim
            query_states = qkv[..., :query_pos].view(bsz, q_len, self.num_heads, self.head_dim)
            key_states = qkv[..., query_pos : query_pos + kv_pos].view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
            value_states = qkv[..., query_pos + kv_pos :].view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
            sliding_window = self.config.sliding_window[self.layer_idx]

        attn_output = self.inner_cross_attn(
            query_states,
            key_states,
            value_states,
            sliding_window=sliding_window,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return self.out_proj(attn_output), (key_states, value_states)


class Phi4FlashMamba(nn.Module):
    def __init__(
        self,
        config: Phi4FlashConfig,
        layer_idx: int,
        yoco_cross: bool = False,
        yoco_kv: bool = False,
    ):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.d_inner = int(self.expand * self.d_model)
        self.num_heads = self.d_inner
        self.head_dim = 1
        self.n_groups = 1
        self.dt_rank = config.mamba_dt_rank
        self.layer_idx = layer_idx
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.activation = "silu"
        self.act = nn.SiLU()

        if self.yoco_cross:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=config.mamba_proj_bias)
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.mamba_proj_bias)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.mamba_proj_bias)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=config.mamba_conv_bias,
                kernel_size=self.d_conv,
                groups=self.d_inner,
                padding=self.d_conv - 1,
            )
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            A = (
                torch.arange(1, self.d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.d_inner, -1)
                .contiguous()
            )
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.mamba_proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        yoco_key_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dtype = hidden_states.dtype
        if self.yoco_cross:
            out = self.in_proj(hidden_states)
            out = _swiglu(out, yoco_key_values)
            return self.out_proj(out), yoco_key_values

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = torch.ops.auto_deploy.torch_causal_conv1d(
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.conv1d.stride[0],
            self.conv1d.padding[0],
            self.conv1d.dilation[0],
            self.conv1d.groups,
            self.conv1d.padding_mode,
        )
        x = self.act(x)
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        y = torch.ops.auto_deploy.torch_ysamba_ssm(
            hidden_states=x.view(
                hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim
            ),
            A=-torch.exp(self.A_log.float()),
            B=B.view(hidden_states.shape[0], hidden_states.shape[1], self.n_groups, self.d_state),
            C=C.view(hidden_states.shape[0], hidden_states.shape[1], self.n_groups, self.d_state),
            D=self.D,
            dt=torch.matmul(dt, self.dt_proj.weight.t()),
            dt_bias=self.dt_proj.bias,
            time_step_limit=[0.0, float("inf")],
            chunk_size=256,
        ).view(hidden_states.shape[0], hidden_states.shape[1], -1)

        if self.yoco_kv:
            yoco_out = y
            y = _swiglu(z, y)
        else:
            yoco_out = None
            y = y * self.act(z)
        return self.out_proj(y.to(dtype)), yoco_out


class Phi4FlashDecoderLayer(nn.Module):
    def __init__(self, config: Phi4FlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = Phi4FlashMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.yoco_mb = layer_idx >= config.num_hidden_layers // 2
        self.yoco_cross = layer_idx >= (config.num_hidden_layers // 2 + 2)
        attn_config = config
        if layer_idx >= (config.num_hidden_layers // 2 + 1):
            attn_config = Phi4FlashConfig(**config.to_dict())
            attn_config.sliding_window = [None] * config.num_hidden_layers

        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            self.attn = Phi4FlashMamba(
                config=attn_config,
                layer_idx=layer_idx,
                yoco_cross=self.yoco_cross,
                yoco_kv=self.yoco_mb,
            )
        else:
            self.attn = Phi4FlashAttention(
                config=attn_config,
                layer_idx=layer_idx,
                yoco_cross=self.yoco_cross,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ssm_output: Optional[torch.Tensor] = None,
        yoco_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(hidden_states, yoco_key_values=ssm_output)
        else:
            attn_outputs, yoco_key_values = self.attn(
                hidden_states, yoco_key_values=yoco_key_values
            )

        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, ssm_output, yoco_key_values


@dataclass
class Phi4FlashModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Phi4FlashCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Phi4FlashPreTrainedModel(PreTrainedModel):
    config_class = Phi4FlashConfig
    base_model_prefix = "model"
    _no_split_modules = ["Phi4FlashDecoderLayer"]
    _supports_flash_attn_2 = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Phi4FlashModel(Phi4FlashPreTrainedModel):
    def __init__(self, config: Phi4FlashConfig):
        config._attn_implementation = "eager"
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Phi4FlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Phi4FlashModelOutput:
        del position_ids
        del kwargs
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        ssm_output = None
        yoco_key_values = None
        for decoder_layer in self.layers:
            hidden_states, ssm_output, yoco_key_values = decoder_layer(
                hidden_states,
                ssm_output=ssm_output,
                yoco_key_values=yoco_key_values,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return Phi4FlashModelOutput(last_hidden_state=hidden_states)


class Phi4FlashForCausalLM(Phi4FlashPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Phi4FlashConfig):
        config._attn_implementation = "eager"
        super().__init__(config)
        self.model = Phi4FlashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Phi4FlashCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return Phi4FlashCausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Phi4FlashConfig", Phi4FlashForCausalLM)
