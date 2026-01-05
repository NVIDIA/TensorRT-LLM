import torch


@torch.library.custom_op("auto_deploy::gather_logits_before_lm_head", mutates_args=())
def gather_logits_before_lm_head(
    hidden_states: torch.Tensor,
    logits_gather_indices: torch.Tensor,  # long tensor
    logits_gather_info_host: torch.Tensor,  # int tensor
) -> torch.Tensor:
    """Gather hidden states using logits_gather_indices before LM head.

    Args:
        hidden_states: Hidden states tensor [b, 1, hidden] or [1, s_total, hidden]
        logits_gather_indices: indices for gathering logits.
        logits_gather_info_host: info for gathering logits.
    Returns:
        Gathered and flattened hidden states [num_gathered_tokens, hidden]
    """
    # final shape is [total_tokens, hidden] from [b, 1, hidden] or [1, total_tokens, hidden]
    is_decode_only = hidden_states.shape[1] == 1
    hidden_states = hidden_states.squeeze(int(is_decode_only))

    # info object
    num_tokens_to_gather, gather_required = logits_gather_info_host.tolist()

    if gather_required:
        out = hidden_states.index_select(0, logits_gather_indices[:num_tokens_to_gather])
    else:
        out = hidden_states.clone(memory_format=torch.contiguous_format)
    return out.unsqueeze(int(is_decode_only))


@gather_logits_before_lm_head.register_fake
def gather_logits_before_lm_head_fake(
    hidden_states: torch.Tensor,
    logits_gather_indices: torch.Tensor,
    logits_gather_info_host: torch.Tensor,
) -> torch.Tensor:
    # NOTE: shape is not correct in fake mode
    # see https://github.com/NVIDIA/TensorRT-LLM/issues/9878
    return torch.empty_like(hidden_states)
