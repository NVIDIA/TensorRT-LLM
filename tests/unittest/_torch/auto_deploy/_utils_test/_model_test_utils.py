import copy
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim
from utils.llm_data import llm_models_root


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    freqs_cis = freqs_cis[None, : x.shape[1], None]  #             --> [1, s,   1, h_d//2, 2]
    xshaped = x.float().unflatten(-1, (-1, 2))  # [b, s, n_h, h_d] --> [b, s, n_h, h_d//2, 2]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )  # [b, s, n_h, h_d//2, 2]

    return x_out2.flatten(-2).type_as(x)  # [b, s, n_h, h_d//2, 2] --> [b, s, n_h, h_d]


def repeat_kv(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep).

    This version avoid any memcopy.
    """
    # q, k, v is [b,s,n,d]
    n_heads = q.shape[2]
    bs, slen, n_kv_heads, head_dim = kv.shape
    n_rep = n_heads // n_kv_heads
    return (
        kv[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        .contiguous()
    )


class GQA(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.is_gqa = num_key_value_heads < num_attention_heads
        assert self.hidden_size == self.num_attention_heads * self.head_dim

        # key, query, value, out projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_key_value_heads, self.head_dim)

        if freqs_cis is not None:
            # q shape is [b,s,n,d]
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        if self.is_gqa:
            k = repeat_kv(q, k)
            v = repeat_kv(q, v)

        q, k, v = map(lambda x: x.transpose(1, 2).contiguous(), (q, k, v))  # [b,n,s,h_d]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [b,n,s,h_d]
        y = y.transpose(1, 2).contiguous().view(b, s, self.hidden_size)  # [b,s,n*h_d]

        return self.o_proj(y)


class MLP(nn.Module):
    """A simple 2 layer MLP example."""

    def __init__(self, in_channels, hidden_size, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, out_channels, bias=False)
        self.linear1._register_load_state_dict_pre_hook(self.load_hook)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)

    def load_hook(self, state_dict, *args):
        print("load hooks")


class TransformerLikeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Transpose embedding layer to project back to logits
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        hidden_states = self.mlp(embeddings)
        logits = self.output_projection(hidden_states)

        return logits


class VisionTransformerLikeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # Projection layer to map input channels to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection to map back to input_dim
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        embeddings = self.input_projection(x)
        hidden_states = self.mlp(embeddings)
        output = self.output_projection(hidden_states)

        return output


class Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.randn(hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.randn(intermediate_size, hidden_size))


class MoEOpModel(nn.Module):
    def __init__(self, hidden_size=32, intermediate_size=16, num_experts=4, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden_size, num_experts)

        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, hidden_size)
        Computes router logits via a gate, and then calls the MoE op via torch.moe.torch_moe.
        """

        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)

        w1_list = [expert.w1 for expert in self.experts]
        w2_list = [expert.w2 for expert in self.experts]
        w3_list = [expert.w3 for expert in self.experts]

        out = torch.ops.moe.torch_moe(
            x, selected_experts, routing_weights, w1_list, w2_list, w3_list
        )
        return out

    def get_input(self, device, dtype=torch.bfloat16):
        return torch.randn(2, self.hidden_size, device=device, dtype=dtype)


def generate_dynamic_shapes(max_batch_size, max_seq_len):
    dynamic_shapes = (
        {
            0: Dim("batch_size", max=max_batch_size),
            1: Dim("seq_len", max=max_seq_len),
        },
    )
    return dynamic_shapes


def _hf_model_dir_or_hub_id(
    hf_model_dir: str,
    hf_hub_id: str,
) -> str:
    if os.path.isdir(hf_model_dir):
        return hf_model_dir
    else:
        return hf_hub_id


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_explicit(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # Expected shape: (B, seq, head_dim//2) and complex dtype.
    unsqueeze_dim: int = 2,
):
    # Reshape the inputs to pair the last dimension.
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Multiply with frequencies. Note that freqs_cis is expected to broadcast with an extra head dim.
    freqs = freqs_cis.unsqueeze(unsqueeze_dim)
    xq_out = torch.view_as_real(xq_complex * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L339
def apply_rotary_pos_emb_ds(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Apply rotary positional embeddings by interleaving Q/K ,
    indexing cos/sin tables with position_ids, and returning rotated q, k.
    cos:  [seq_len, head_dim]
    sin:  [seq_len, head_dim]
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


_SMALL_MODEL_CONFIGS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "model": _hf_model_dir_or_hub_id(
            f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
        "model_kwargs": {
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        },
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "model": _hf_model_dir_or_hub_id(
            f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ),
        "model_kwargs": {
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_local_experts": 2,
        },
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "model": _hf_model_dir_or_hub_id(
            f"{llm_models_root()}/Phi-3/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-4k-instruct",
        ),
        "model_kwargs": {
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "model": _hf_model_dir_or_hub_id(
            f"{llm_models_root()}/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        ),
        "model_factory": "AutoModelForImageTextToText",
        "customize_tokenizer": True,
        "model_kwargs": {
            "text_config": {
                "num_hidden_layers": 1,
                "head_dim": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "intermediate_size_mlp": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_local_experts": 2,
            },
            "vision_config": {
                "num_hidden_layers": 1,
                "hidden_size": 64,
                "intermediate_size": 64,
                "num_attention_heads": 2,
            },
        },
    },
    "deepseek-ai/DeepSeek-V3": {
        "model": _hf_model_dir_or_hub_id(
            f"{llm_models_root()}/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V3",
        ),
        "model_kwargs": {
            "first_k_dense_replace": 1,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "kv_lora_rank": 128,
            "moe_intermediate_size": 128,
            "n_group": 2,
            "topk_group": 2,
            "n_routed_experts": 16,
            "n_shared_experts": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_experts_per_token": 2,
            "q_lora_rank": 128,
        },
    },
}


def get_small_model_config(model_hub_id: str, **config_kwargs) -> Dict[str, Any]:
    """
    Get the small model configuration for a given HuggingFace model hub ID.

    Args:
        model_hub_id: The HuggingFace model hub ID (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")

    Returns:
        Dictionary containing the model configuration

    Raises:
        KeyError: If the model_hub_id is not found in the configurations
    """
    if model_hub_id not in _SMALL_MODEL_CONFIGS:
        available_models = list(_SMALL_MODEL_CONFIGS.keys())
        raise KeyError(f"Model '{model_hub_id}' not found. Available models: {available_models}")

    config = copy.deepcopy(_SMALL_MODEL_CONFIGS[model_hub_id])

    # add default values for small model config
    config["skip_loading_weights"] = True  # No weight loading to speed up things
    config["free_mem_ratio"] = 0.00  # we don't need the cache and it may cause OOM issues
    config["benchmark"] = False  # No benchmark to speed up things
    config["max_tokens"] = 8  # Don't produce too many tokens to speed up things
    config["attn_page_size"] = 4  # Make sure paging is activated despite small max_tokens
    config["max_batch_size"] = 2  # Minimum batching to speed up things
    config["prompt"] = "Hello World"

    # add custom config kwargs
    config.update(config_kwargs)

    return config
