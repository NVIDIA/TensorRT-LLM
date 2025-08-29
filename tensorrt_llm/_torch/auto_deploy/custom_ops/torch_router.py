import torch
import torch.nn.functional as F


@torch.library.custom_op("auto_deploy::torch_moe_router", mutates_args=())
def torch_moe_router(
    hidden_states: torch.Tensor,  # [B, S, H]
    weight: torch.Tensor,  # [E, H]
    bias: torch.Tensor,  # [E]
    top_k: int = 2,
) -> torch.Tensor:
    """
    Reference router:
      - reshape tokens
      - linear to logits
      - topk
      - softmax over topk
      - scatter back to full expert space
    Returns:
      router_scores:  [B*S, E]
      router_indices: [B*S, top_k] (int64)
    """
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)  # [T, H]
    router_logits = F.linear(hidden_states, weight, bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(
        router_top_value, dim=1, dtype=router_top_value.dtype
    )
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    return router_scores


@torch_moe_router.register_fake
def _torch_moe_router_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int = 2,
) -> torch.Tensor:
    # Expect 3D input [B,S,H]; infer token count and experts from params
    B, S, H = hidden_states.shape
    E = weight.shape[0]
    T = B * S
    scores = torch.empty((T, E), device="meta", dtype=hidden_states.dtype)
    return scores
