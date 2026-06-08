import os
from typing import Optional

import torch
from flashinfer.sampling import top_k_top_p_sampling_from_logits


def forward_native(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    PyTorch-native implementation of top-k and top-p sampling.

    The logits tensor may be updated in-place.
    """
    logits = apply_top_k_top_p(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs)


def random_sample(
    probs: torch.Tensor,
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs).exponential_()
    return probs.div_(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        top_k_mask = top_k_mask.clamp(min=0)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def apply_temperature(
    logits: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    return logits.div_(temp.unsqueeze(dim=1))


def apply_recent_token_penalties(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty_values: torch.Tensor,
) -> torch.Tensor:
    if os.environ.get("TRTLLM_SPEC_USE_PENALTY_OP", "0") == "1":
        torch.ops.trtllm.speculative_apply_token_penalties(
            logits, token_ids, penalty_values)
        return logits
    return logits.scatter_add_(1, token_ids, -penalty_values.to(logits.dtype))


def apply_history_frequency_penalty(
    logits: torch.Tensor,
    history_tokens: torch.Tensor,
    history_lens: torch.Tensor,
    row_slots: torch.Tensor,
    frequency_penalties: torch.Tensor,
) -> torch.Tensor:
    torch.ops.trtllm.speculative_apply_history_frequency_penalty(
        logits, history_tokens, history_lens, row_slots, frequency_penalties)
    return logits


def apply_count_frequency_penalty(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    row_slots: torch.Tensor,
    frequency_penalties: torch.Tensor,
) -> torch.Tensor:
    torch.ops.trtllm.speculative_apply_count_frequency_penalty(
        logits, token_counts, row_slots, frequency_penalties)
    return logits


def apply_sparse_count_frequency_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    count_lens: torch.Tensor,
    row_slots: torch.Tensor,
    frequency_penalties: torch.Tensor,
) -> torch.Tensor:
    torch.ops.trtllm.speculative_apply_sparse_count_frequency_penalty(
        logits, token_ids, token_counts, count_lens, row_slots,
        frequency_penalties)
    return logits


def append_accepted_tokens_to_history(
    history_tokens: torch.Tensor,
    history_lens: torch.Tensor,
    seq_slots: torch.Tensor,
    accepted_tokens: torch.Tensor,
    accepted_lens: torch.Tensor,
) -> None:
    torch.ops.trtllm.speculative_append_accepted_tokens(
        history_tokens, history_lens, seq_slots, accepted_tokens,
        accepted_lens)


def append_accepted_tokens_to_counts(
    token_counts: torch.Tensor,
    seq_slots: torch.Tensor,
    accepted_tokens: torch.Tensor,
    accepted_lens: torch.Tensor,
) -> None:
    torch.ops.trtllm.speculative_append_accepted_token_counts(
        token_counts, seq_slots, accepted_tokens, accepted_lens)


def append_accepted_tokens_to_sparse_counts(
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    count_lens: torch.Tensor,
    seq_slots: torch.Tensor,
    accepted_tokens: torch.Tensor,
    accepted_lens: torch.Tensor,
    vocab_size: int,
) -> None:
    torch.ops.trtllm.speculative_append_sparse_token_counts(
        token_ids, token_counts, count_lens, seq_slots, accepted_tokens,
        accepted_lens, vocab_size)


def init_sparse_token_counts(
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    count_lens: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    prompt_token_counts: torch.Tensor,
    prompt_lens: torch.Tensor,
    seq_slots: torch.Tensor,
    vocab_size: int,
) -> None:
    torch.ops.trtllm.speculative_init_sparse_token_counts(
        token_ids, token_counts, count_lens, prompt_token_ids,
        prompt_token_counts, prompt_lens, seq_slots, vocab_size)


@torch.compile(options={"max-autotune": True})
def sampling_batch_spec_dec_one_model(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    use_flashinfer: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-graph compatible sampling. Supports mixed sampling params.

    We can't do dynamic kernel selection inside graphs, so this might
    be slower than a torch.argmax for greedy requests. This is why advanced
    sampling is opt-in for now.
    """
    logits = apply_temperature(logits, temperatures)
    if use_flashinfer:
        return top_k_top_p_sampling_from_logits(logits, top_k, top_p, seed=seed, offset=offset)
    random_sampled = forward_native(logits, top_k, top_p)
    return random_sampled
