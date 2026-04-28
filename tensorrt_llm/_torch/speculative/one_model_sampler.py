from typing import Optional, Tuple

import torch

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE

if IS_FLASHINFER_AVAILABLE:
    from flashinfer.sampling import (
        chain_speculative_sampling,
        sampling_from_probs,
        top_k_mask_logits,
        top_k_top_p_sampling_from_logits,
        top_p_renorm_probs,
    )
    from flashinfer.sampling import softmax as flashinfer_softmax
else:
    chain_speculative_sampling = None
    sampling_from_probs = None
    flashinfer_softmax = None
    top_k_mask_logits = None
    top_p_renorm_probs = None
    top_k_top_p_sampling_from_logits = None


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


def sanitize_top_k(top_k: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map ``top_k`` into a backend-safe range before top-k filtering.

    Per ``SamplingParams``, ``top_k == 0`` means "all logits" (top-k disabled),
    but the flashinfer (``top_k_mask_logits``) and PyTorch-native top-k paths
    break on a literal 0 — they mask the entire row (all-zero probs) or gather
    out of bounds. Mirror the C++ op (``dynamicTreeKernels.cu``): map any
    non-positive value (and any oversized disable sentinel such as
    ``INT32_MAX``) to ``vocab_size`` (== keep all tokens), leaving genuine
    top_k values untouched.
    """
    return torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab_size)).clamp(max=vocab_size)


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
    Sampling for speculative decoding. Supports mixed sampling params.

    NOTE: torch.compile is intentionally omitted.  With TP > 1 each rank
    runs in a separate process; compiled sampling can produce slightly
    different results across ranks (e.g. different Triton kernel selections).
    In spec dec, sampled tokens determine acceptance counts and therefore
    the batch shape of subsequent draft-model NCCL collectives — divergent
    tokens cause a deadlock.
    """
    top_k = sanitize_top_k(top_k, logits.shape[-1])
    logits = apply_temperature(logits, temperatures)
    if use_flashinfer:
        if top_k_top_p_sampling_from_logits is None:
            raise RuntimeError("FlashInfer sampling requested but flashinfer is unavailable")
        return top_k_top_p_sampling_from_logits(logits, top_k, top_p, seed=seed, offset=offset)
    random_sampled = forward_native(logits, top_k, top_p)
    return random_sampled


@torch.compile(options={"max-autotune": True})
def compute_probs_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
    skip_temperature: bool = False,
) -> torch.Tensor:
    """
    Compute filtered + normalized probs from logits (temperature + top_k +
    top_p + softmax). Picks the fastest path for the input device:

    1. CUDA + flashinfer: ``top_k_mask_logits`` → fused ``softmax+temp`` →
       ``top_p_renorm_probs`` (all O(N) radix). ``skip_temperature`` ignored.
    2. CUDA, no flashinfer: ``compute_probs_from_logits_op`` (sort-based,
       O(N log N)).
    3. CPU: manual PyTorch fallback.
    """
    if top_k is not None:
        top_k = sanitize_top_k(top_k, logits.shape[-1])

    if logits.is_cuda and IS_FLASHINFER_AVAILABLE:
        # Fast path: flashinfer composition (O(N) per row, friendly to small
        # batch sizes). skip_temperature is ignored — flashinfer's softmax
        # always applies the temperature tensor.
        if top_k is not None:
            logits = top_k_mask_logits(logits, top_k)
        probs = flashinfer_softmax(logits, temperatures)
        if top_p is not None:
            probs = top_p_renorm_probs(probs, top_p)
        return probs

    if logits.is_cuda:
        # CUDA without flashinfer: fall back to the C++ op (slower sort-based
        # top-k path, but works without flashinfer).
        return torch.ops.trtllm.compute_probs_from_logits_op(
            logits, temperatures, top_k, top_p, skip_temperature
        )

    if not skip_temperature:
        logits = apply_temperature(logits, temperatures)
    logits = apply_top_k_top_p(logits, top_k, top_p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    # Greedy rows should remain exactly one-hot so rejection sampling does not
    # spuriously reject numerically-near argmax tokens.
    greedy_temp_threshold = 1e-4
    is_greedy = temperatures <= greedy_temp_threshold
    argmax_ids = logits.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(probs).scatter_(1, argmax_ids, 1.0)
    return torch.where(is_greedy.unsqueeze(1), one_hot, probs)


def sampling_batch_spec_dec_one_model_for_rejection(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rejection-sampling-aware draft sampler: returns BOTH the sampled tokens
    AND the prob distribution they were sampled from, so the downstream
    rejection-sampling path can reuse the probs without a second softmax +
    temp/top_k/top_p pass.
    """
    if sampling_from_probs is None:
        raise RuntimeError(
            "Rejection sampling for one-model speculative decoding requires flashinfer"
        )
    probs = compute_probs_from_logits(logits, temperatures, top_k, top_p)
    tokens = sampling_from_probs(probs, deterministic=True, seed=seed, offset=offset)
    return tokens, probs


def rejection_sampling_one_model(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    deterministic: bool = True,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-graph compatible rejection sampling for one-model speculative decoding.
    """
    batch_size = draft_token_ids.shape[0]
    device = draft_token_ids.device

    output_accepted_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)
    output_emitted_draft_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)

    if chain_speculative_sampling is None:
        raise RuntimeError(
            "Rejection sampling for one-model speculative decoding requires flashinfer"
        )

    accepted_tokens, _, output_emitted_draft_token_num = chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        maybe_output_accepted_token_num=output_accepted_token_num,
        maybe_output_emitted_draft_token_num=output_emitted_draft_token_num,
        deterministic=deterministic,
        generator=None,
        seed=seed,
        offset=offset,
    )

    return accepted_tokens, output_emitted_draft_token_num + 1
