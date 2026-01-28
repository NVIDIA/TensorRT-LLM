# Adapted from: https://huggingface.co/nvidia/Nemotron-Flash-3B-Instruct/tree/main
import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.inputs.utils import HF_CHAT_TEMPLATE_EXCEPTIONS

from ..nemotron_flash import NemotronFlashForCausalLMFactory


class NemotronFlashPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, num_memory_tokens: int = 0, vocab_size_model: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_memory_tokens = num_memory_tokens
        self.num_dummy_tokens = max(0, vocab_size_model - len(self))
        self.add_tokens([f"<dummy_{i}>" for i in range(self.num_dummy_tokens)], special_tokens=True)

        self.mem_tokens = [f"<mem_{i}>" for i in range(self.num_memory_tokens)]
        self.mem_token_ids = list(range(len(self), len(self) + self.num_memory_tokens))
        self.add_tokens(self.mem_tokens, special_tokens=True)

        if getattr(self, "chat_template", None) is None:
            self.chat_template = (
                "{% for m in messages %}"
                "{{ m['content'] }}"
                "{% if not loop.last %}\n{% endif %}"
                "{% endfor %}"
            )
        self.model_input_names = ["input_ids"]

    def _add_memory_tokens(self, input_ids) -> Union[List[List[int]], torch.Tensor]:
        is_unbatched = True
        if isinstance(input_ids, list) and isinstance(input_ids[0], int):
            input_ids = [input_ids]
        elif isinstance(input_ids, torch.Tensor) and input_ids.ndim == 1:
            input_ids = input_ids[None]
        else:
            is_unbatched = False

        if isinstance(input_ids, list):
            input_ids = [self.mem_token_ids + _ids for _ids in input_ids]
        elif isinstance(input_ids, torch.Tensor):
            mem_token_tnsr = torch.tensor(
                self.mem_token_ids, device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat((mem_token_tnsr[None], input_ids), dim=1)
        else:
            raise ValueError(f"Unsupported input type {type(input_ids)}")

        if is_unbatched:
            return input_ids[0]
        else:
            return input_ids

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        batch_encoding = super()._batch_encode_plus(*args, **kwargs)
        batch_encoding.data["input_ids"] = self._add_memory_tokens(batch_encoding.data["input_ids"])
        return batch_encoding

    def _decode(self, token_ids: Union[int, list[int]], *args, **kwargs) -> str:
        if isinstance(token_ids, list):
            token_ids = [_id for _id in token_ids if _id not in self.mem_token_ids]
        return super()._decode(token_ids, *args, **kwargs)


def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class NemotronFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> "NemotronFlashRMSNorm":
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x):
        return torch.ops.auto_deploy.triton_rms_norm(x, self.weight, self.eps).to(x.dtype)


class RMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: Optional[int] = None,
        norm_before_gate: bool = False,
    ) -> "RMSNormGated":
        super().__init__()

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.group_size = group_size or hidden_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.triton_rmsnorm_gated(
            x,
            weight=self.weight,
            gate=g,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


class CausalConv1d(nn.Conv1d):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_causal_conv1d(
            x,
            self.weight,
            self.bias,
            self.stride[0],
            self.padding[0],
            self.dilation[0],
            self.groups,
            self.padding_mode,
        )


class DeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        config=None,
        **kwargs,
    ) -> "DeltaNet":
        super().__init__()

        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # can't use ActivationType enum here,
        #  because there is no Elu defined in cpp/tensorrt_llm/kernels/cutlass_kernels/include/common.h
        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        self.silu = nn.SiLU()

        assert self.key_dim % num_heads == 0, (
            f"key dim must be divisible by num_heads of {num_heads}"
        )
        assert self.value_dim % num_heads == 0, (
            f"value dim must be divisible by num_heads of {num_heads}"
        )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if self.use_short_conv:
            self.q_conv1d = CausalConv1d(self.key_dim, kernel_size=conv_size, bias=conv_bias)
            self.k_conv1d = CausalConv1d(self.key_dim, kernel_size=conv_size, bias=conv_bias)
            self.v_conv1d = CausalConv1d(self.value_dim, kernel_size=conv_size, bias=conv_bias)

        if self.use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = RMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = NemotronFlashRMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.use_short_conv:
            q = self.q_conv1d(q)
            k = self.k_conv1d(k)
            v = self.v_conv1d(v)

        q = q.view(batch_size, seq_len, -1, self.head_k_dim)
        k = k.view(batch_size, seq_len, -1, self.head_k_dim)
        v = v.view(batch_size, seq_len, -1, self.head_v_dim)

        if self.qk_activation == "relu":
            q, k = q.relu(), k.relu()
        elif self.qk_activation == "elu":
            q, k = elu_p1(q), elu_p1(k)
        elif self.qk_activation == "identity":
            pass
        elif self.qk_activation == "silu":
            q, k = self.silu(q), self.silu(k)
        else:
            raise NotImplementedError

        v = self.silu(v)

        if self.use_beta:
            beta = self.b_proj(hidden_states)
            beta = beta.sigmoid()
        else:
            beta = q.new_ones()

        if self.allow_neg_eigval:
            beta = beta * 2.0

        if self.qk_norm == "l2":
            q = torch.ops.auto_deploy.torch_l2norm(q)
            k = torch.ops.auto_deploy.torch_l2norm(k)
        elif self.qk_norm == "sum":
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)
        else:
            raise NotImplementedError(f"Not supported qk_norm `{self.qk_norm}`.")

        o = torch.ops.auto_deploy.fla_delta_rule(q, k, v, beta)

        if self.use_gate:
            g = self.g_proj(hidden_states).view(batch_size, seq_len, -1, self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = o.view(batch_size, seq_len, -1)
        o = self.o_proj(o)

        return o.to(dtype)


class NemotronFlashMamba2(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        num_groups=1,
        rmsnorm=True,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=128,  # NOTE: original 256 gives us IMA in _chunk_scan_fwd_kernel for mamba2
    ):
        super().__init__()

        self.config = config
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv

        self.expand = config.mamba_expand
        self.d_inner = self.expand * self.d_model
        self.headdim = config.mamba2_headdim
        self.num_groups = num_groups
        self.num_heads = self.d_inner // self.headdim
        self.rmsnorm = rmsnorm
        self.dt_limit = dt_limit
        self.activation = ActivationType.Silu
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        assert self.d_inner % self.headdim == 0

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.num_groups * self.d_state + self.num_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        conv_dim = self.d_inner + 2 * self.num_groups * self.d_state
        self.conv1d = CausalConv1d(conv_dim, kernel_size=self.d_conv, bias=conv_bias)
        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                group_size=self.d_inner // num_groups,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        zxbcdt = self.in_proj(hidden_states)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.num_groups * self.d_state, self.num_heads],
            dim=-1,
        )

        xBC = self.act(self.conv1d(xBC))

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.num_groups * self.d_state, self.num_groups * self.d_state],
            dim=-1,
        )

        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=x.view(batch_size, seq_len, -1, self.headdim),
            A=-torch.exp(self.A_log.float()),
            B=B.view(batch_size, seq_len, -1, self.d_state),
            C=C.view(batch_size, seq_len, -1, self.d_state),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=self.dt_limit,
            chunk_size=self.chunk_size,
        )

        y = y.view(batch_size, seq_len, -1)

        if self.rmsnorm:
            y = self.norm(y, z)

        out = self.out_proj(y.to(dtype))

        return out


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        self.config = config

        self.rope_type = config.rope_type

        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if config.rope_type is None or config.rope_type == "default":
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == "ntk":
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            base = base * (
                (self.factor * max_position_embeddings / orig_max_position_embeddings)
                - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )

            self.max_seq_len_cached = orig_max_position_embeddings
        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NemotronFlashAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = (
            config.attn_hidden_size if config.attn_hidden_size > 0 else config.hidden_size
        )
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.kq_head_dim = config.kq_head_dim if config.kq_head_dim > 0 else self.head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim > 0 else self.head_dim

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (
            self.head_dim * self.num_heads
        ) != self.hidden_size and self.kq_head_dim == self.head_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.kq_head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.kq_head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        if self.config.kq_norm == "rms":
            self.k_norm = NemotronFlashRMSNorm(self.kq_head_dim)
            self.q_norm = NemotronFlashRMSNorm(self.kq_head_dim)
        elif self.config.kq_norm == "none":
            self.k_norm = None
            self.q_norm = None
        else:
            raise NotImplementedError(f"Unknown kq_norm: {self.config.kq_norm}")

        if self.config.rope:
            self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            dim=self.kq_head_dim,
            base=self.rope_theta,
            device=torch.device("cuda"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sliding_window: Optional[int] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.kq_head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.kq_head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.v_head_dim)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if self.config.rope:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            dropout_p=self.attention_dropout,
            sliding_window=sliding_window,
            layout="bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class NemotronFlashMLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.act_fn_name = config.mlp_hidden_act
        self.act_fn = ACT2FN[self.act_fn_name]

        if config.ffn_expand_ratio is not None:
            self.ffn_dim = int(config.ffn_expand_ratio * config.hidden_size) // 128 * 128
        else:
            self.ffn_dim = config.intermediate_size

        self.hidden_dim = config.hidden_size

        self.layer_idx = layer_idx

        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        if self.act_fn_name == "silu":
            output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        elif self.act_fn_name == "relu2":
            output = self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            raise NotImplementedError(f"No such hidden_act: {self.act_fn_name}")

        return output


class NemotronFlashAttentionDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        self.self_attn = NemotronFlashAttention(config, layer_idx)

        if self.config.intermediate_size > 0:
            self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
            self.pre_ffn_layernorm = NemotronFlashRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.ffn = None
            self.pre_ffn_layernorm = None

        self.input_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states, position_ids=position_ids, **kwargs
        )

        hidden_states = residual + hidden_states

        if self.ffn is not None:
            residual = hidden_states
            if self.pre_ffn_layernorm is not None:
                hidden_states = self.pre_ffn_layernorm(hidden_states)
            hidden_states = self.ffn(hidden_states)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class FFNDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)

        self.pre_ffn_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        residual = hidden_states
        if self.pre_ffn_layernorm is not None:
            hidden_states = self.pre_ffn_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class NemotronFlashMambaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.mamba = NemotronFlashMamba2(config=config, layer_idx=layer_idx)

        self.intermediate_size = config.intermediate_size
        if self.intermediate_size > 0:
            self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
            self.pre_ffn_layernorm = NemotronFlashRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.ffn = None
            self.pre_ffn_layernorm = None

        self.input_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        if self.intermediate_size > 0:
            residual = hidden_states

            if self.pre_ffn_layernorm is not None:
                hidden_states = self.pre_ffn_layernorm(hidden_states)

            hidden_states = self.ffn(hidden_states)

            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class NemotronFlashHybridDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        if config.hybrid_decoder_layer == "mamba":
            self.mamba = NemotronFlashMamba2(config=config, layer_idx=layer_idx)
        if config.hybrid_decoder_layer == "deltanet":
            ## this is to properly handle cache index
            if config.layer_types is not None:
                deltanet_idx = sum(
                    1 for i in range(layer_idx) if config.layer_types[i] == "deltanet"
                )
            else:
                deltanet_idx = layer_idx

            self.gla = DeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                layer_idx=deltanet_idx,
                config=self.config,
            )
        else:
            raise ValueError(f"Not supported: {config.hybrid_decoder_layer}")

        self.config = config

        if self.config.intermediate_size > 0:
            self.ffn = NemotronFlashMLP(config, layer_idx=layer_idx)
            self.pre_ffn_layernorm = NemotronFlashRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.ffn = None
            self.pre_ffn_layernorm = None

        self.input_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if self.config.hybrid_decoder_layer == "mamba":
            hybrid_op_hidden_states = self.mamba(hidden_states=hidden_states)

        else:
            hybrid_op_hidden_states = self.gla(hidden_states=hidden_states)

        hidden_states = residual + hybrid_op_hidden_states

        if self.ffn is not None:
            residual = hidden_states
            hidden_states = self.pre_ffn_layernorm(hidden_states)
            hidden_states = self.ffn(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class NemotronFlashPreTrainedModel(PreTrainedModel):
    # config_class = NemotronFlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NemotronFlashAttentionDecoderLayer", "NemotronFlashMambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class NemotronFlashModel(NemotronFlashPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`NemotronFlashDecoderLayer`]

    Args:
        config: NemotronFlashConfig
    """

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size + config.num_memory_tokens,
            config.hidden_size,
            self.padding_idx,
        )

        if self.config.num_memory_tokens > 0:
            # register an appropriate pre load hook to merge memory tokens into the embedding layer
            self.register_load_state_dict_pre_hook(self._merge_memory_tokens)

        decoder_layers = []

        layer_type = []
        for i in range(config.num_hidden_layers):
            if config.layer_types[i] in ["deltanet"]:
                layer_type.append("m")
                config_new = copy.deepcopy(config)
                config_new.hybrid_decoder_layer = "deltanet"
                decoder_layer = NemotronFlashHybridDecoderLayer(config_new, layer_idx=i)
            elif config.layer_types[i] in ["m", "m2"]:
                layer_type.append("m")
                decoder_layer = NemotronFlashMambaDecoderLayer(config, layer_idx=i)
            elif config.layer_types[i] == "a":
                layer_type.append("a")
                decoder_layer = NemotronFlashAttentionDecoderLayer(config, layer_idx=i)
            elif config.layer_types[i] == "f":
                layer_type.append("a")
                decoder_layer = FFNDecoderLayer(config, layer_idx=i)
            else:
                raise ValueError(f"Unsupported layer type {config.layer_types[i]}")

            decoder_layers.append(decoder_layer)

        config.layer_type = layer_type

        if config.sliding_window is not None:
            self.sliding_window = config.sliding_window
            self.global_attn_idx = config.global_attn_idx
        else:
            self.sliding_window = None
            self.global_attn_idx = None

        self.layers = nn.ModuleList(decoder_layers)

        self.final_layernorm = NemotronFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _merge_memory_tokens(
        module: "NemotronFlashModel",
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        embed_suffix = "embed_tokens.weight"
        embed_key = prefix + embed_suffix
        memory_key = prefix + "memory_tokens"

        if embed_key not in state_dict or memory_key not in state_dict:
            return

        embed_weight_from_state = state_dict[embed_key]

        # Already merged. Remove stale memory token weights if present.
        if embed_weight_from_state.shape[0] == module.get_parameter(embed_suffix).shape[0]:
            state_dict.pop(memory_key, None)
            return

        memory_weights = state_dict.pop(memory_key).to(embed_weight_from_state.dtype)
        state_dict[embed_key] = torch.cat([embed_weight_from_state, memory_weights], dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)

        hidden_states = inputs_embeds

        all_hidden_states = []

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sliding_window=self.sliding_window is not None and i not in self.global_attn_idx,
            )

            hidden_states = layer_outputs[0]

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return (hidden_states, all_hidden_states)


class TruncatedLinear(nn.Linear):
    def __init__(self, *args, out_truncated: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_truncated = out_truncated

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight[: self.out_truncated], self.bias)


class NemotronFlashForCausalLM(NemotronFlashPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.model = NemotronFlashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = TruncatedLinear(
            config.hidden_size,
            config.vocab_size + config.num_memory_tokens,
            bias=False,
            out_truncated=config.vocab_size,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        if return_dict:
            return CausalLMOutput(
                logits=logits,
                hidden_states=outputs[1] if output_hidden_states else None,
            )

        return (logits, *outputs[1:])


NemotronFlashForCausalLMFactory.register_custom_model_cls(
    "NemotronFlashConfig", NemotronFlashForCausalLM
)
HF_CHAT_TEMPLATE_EXCEPTIONS.append("nemotron_flash")
