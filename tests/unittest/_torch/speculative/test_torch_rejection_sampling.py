import unittest
from typing import cast

import numpy as np
import torch
from scipy.stats import entropy

from tensorrt_llm._torch.pyexecutor.sampler import (get_rejected_indices,
                                                    sample_rejected)


def test_get_rejected_indices():
    vocab_size = 500
    num_iter = 50000
    draft_probs = torch.rand(1, vocab_size)
    drop_idx = torch.topk(draft_probs[0], k=400, largest=False)[1]
    draft_probs[0, drop_idx] = 0.0
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    target_probs = torch.rand(2, vocab_size)
    drop_idx = torch.topk(target_probs[0], k=400, largest=False)[1]
    target_probs[0, drop_idx] = 0.0
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    generator = torch.Generator()
    sampled_tokens = []
    sampled_regular = []
    for _ in range(num_iter):
        draft_tokens = [
            cast(
                int,
                torch.multinomial(draft_probs,
                                  num_samples=1,
                                  generator=generator).item())
        ]
        rejected_indices = get_rejected_indices(draft_probs, target_probs,
                                                generator, draft_tokens)
        if rejected_indices.shape[0] == 0:
            sampled_tokens.append(draft_tokens[0])
        else:
            sampled_tokens.append(
                sample_rejected(draft_probs, target_probs, generator, 0))
        sampled_regular.append(
            torch.multinomial(target_probs[0],
                              num_samples=1,
                              generator=generator).item())
    bins = np.arange(vocab_size + 1) - 0.5  # Bins for histogram
    sampled_tokens, _ = np.histogram(sampled_tokens, bins=bins, density=True)
    sampled_regular, _ = np.histogram(sampled_regular, bins=bins, density=True)
    expected_prob = target_probs[0].squeeze().numpy()

    # KL Divergence check
    kl_divergence = entropy(expected_prob, sampled_tokens)
    kl_divergence_regular = entropy(expected_prob, sampled_regular)
    assert abs(kl_divergence - kl_divergence_regular) < 0.01


if __name__ == "__main__":
    unittest.main()
