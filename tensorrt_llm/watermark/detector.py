# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Detector for the Gumbel-max text watermark.

For each scored position the detector recomputes the uniform number ``u`` that the sampler
derived for the observed token from the secret key and the preceding tokens. Watermarked
text has unusually large ``u`` at its tokens. Under the null hypothesis (text not generated
with this key) ``u`` is uniform, ``-ln(1 - u)`` is Exp(1), and the sum over ``n`` scored
positions is Gamma(n, 1), which gives the p-value.

The detector only needs token ids; tokenize text with the tokenizer used for generation.
The configuration must match the generation-time one: key, ``context_width``, PRF and
deduplication settings. A fixed key reuses PRF values across documents, so the nominal
p-value assumes independent inputs; measure the false-positive rate on representative
unwatermarked text before relying on a decision.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import WatermarkConfig
from .prf import PAD_TOKEN, context_seeds, philox_uniform, secondary_key

__all__ = ["GumbelWatermarkDetector", "WatermarkDetection"]


@dataclass(frozen=True)
class WatermarkDetection:
    """Result of testing one token sequence."""

    score: float  # sum of -ln(1 - u) over the scored positions (larger of the two keys' per token)
    p_value: float  # P(score >= observed) under "not watermarked"
    num_scored: int  # positions that entered the score
    num_tokens: int  # tokens given
    is_watermarked: bool


def _gamma_upper_tail(n: float, x: float) -> float:
    """Regularized upper incomplete gamma ``Q(n, x)``, i.e. ``P(Gamma(n, 1) >= x)``."""
    if x <= 0.0:
        return 1.0
    if n <= 0.0:
        return 0.0
    log_prefactor = -x + n * math.log(x) - math.lgamma(n)
    if x < n + 1.0:
        # Series for the lower tail P(n, x), then complement.
        term = 1.0 / n
        total = term
        a = n
        for _ in range(100000):
            a += 1.0
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - math.exp(log_prefactor) * total)
    # Continued fraction (modified Lentz) for the upper tail.
    tiny = 1e-300
    b = x + 1.0 - n
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - n)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return min(1.0, math.exp(log_prefactor) * h)


def _max_pair_sum_tail(n: int, total: float) -> float:
    """``P(S >= total)`` for ``S`` the sum of ``n`` iid ``max(E1, E2)``, with ``E1, E2 ~ Exp(1)``.

    ``max(E1, E2)`` is ``Exp(rate 2) + Exp(rate 1)``, so ``S = Gamma(n, rate 2) + Gamma(n, rate 1)``
    and the tail is ``P(Gamma(n, 2) >= t) + integral_0^t f_Gamma(n, 2)(x) Q(n, t - x) dx``,
    integrated numerically.
    """
    if n <= 0 or total <= 0.0:
        return 1.0
    grid = np.linspace(0.0, total, 2001)[1:-1]
    log_density = n * math.log(2.0) + (n - 1) * np.log(grid) - 2.0 * grid - math.lgamma(n)
    tail = np.fromiter(
        (_gamma_upper_tail(float(n), float(total - x)) for x in grid), dtype=np.float64
    )
    integral = float(np.trapezoid(np.exp(log_density) * tail, grid))
    return min(1.0, _gamma_upper_tail(float(n), 2.0 * total) + integral)


class GumbelWatermarkDetector:
    """Test token sequences for a watermark produced with ``config``."""

    def __init__(self, config: WatermarkConfig, *, p_value_threshold: float = 1e-3):
        self.config = config
        self.p_value_threshold = p_value_threshold

    def detect(
        self,
        token_ids: Sequence[int],
        *,
        prompt_token_ids: Optional[Sequence[int]] = None,
    ) -> WatermarkDetection:
        """Test ``token_ids`` (the generated tokens).

        With ``prompt_token_ids`` the first positions are scored too, since their contexts
        reach into the prompt. Without it the first ``context_width`` positions are skipped.
        With ``deduplicate_contexts="all"`` the first ``context_width`` generated tokens are never
        scored, and the prompt is searched for repeated contexts, which needs the prompt.
        """
        config = self.config
        width = config.context_width
        generated = np.asarray(token_ids, dtype=np.int64)
        num_tokens = int(generated.shape[0])
        if prompt_token_ids is None:
            prefix = np.zeros(0, dtype=np.int64)
            first = width
        else:
            prompt = np.asarray(prompt_token_ids, dtype=np.int64)
            prefix = np.full(width, PAD_TOKEN, dtype=np.int64)
            prefix = np.concatenate([prefix, prompt])[-width:]
            first = 0
        history = np.concatenate([prefix, generated])
        offset = history.shape[0] - num_tokens  # index of generated[0] in history
        positions = np.arange(first, num_tokens)
        if positions.size == 0:
            return WatermarkDetection(0.0, 1.0, 0, num_tokens, False)

        contexts = np.stack([history[offset + positions - width + j] for j in range(width)], axis=1)
        scored = self._scored_mask(contexts, first, prompt_token_ids)
        num_scored = int(scored.sum())
        per_key = []
        keys = [config.key]
        if config.algorithm == "gumbel_dual_key":
            keys.append(secondary_key(config.key))
        for key in keys:
            seeds = context_seeds(key, contexts)
            uniforms = philox_uniform(seeds, generated[positions]).astype(np.float64)
            per_key.append(-np.log1p(-uniforms[scored]))
        if len(per_key) == 1:
            total = float(per_key[0].sum())
            p_value = _gamma_upper_tail(float(num_scored), total) if num_scored else 1.0
        else:
            # Dual-key text carries the first key in some tokens and the second in others,
            # and the detector cannot tell which. Score every token with the larger of its
            # two scores; the null distribution of that statistic is known exactly.
            total = float(np.maximum(per_key[0], per_key[1]).sum())
            p_value = _max_pair_sum_tail(num_scored, total) if num_scored else 1.0
        return WatermarkDetection(
            score=total,
            p_value=p_value,
            num_scored=num_scored,
            num_tokens=num_tokens,
            is_watermarked=num_scored > 0 and p_value < self.p_value_threshold,
        )

    def _scored_mask(
        self,
        contexts: np.ndarray,
        first_position: int,
        prompt_token_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Positions whose context is not a repeat within the deduplication window.

        The ``"all"`` scope also searches the contexts inside the prompt and never scores the
        first ``context_width`` generated tokens, which the sampler leaves unwatermarked.
        """
        config = self.config
        mask = np.ones(contexts.shape[0], dtype=bool)
        if config.deduplicate_contexts == "none":
            return mask
        window = config.deduplicate_contexts_max_history
        last_seen: dict[bytes, int] = {}
        if config.deduplicate_contexts == "all":
            width = config.context_width
            mask[: max(width - first_position, 0)] = False
            if prompt_token_ids is not None:
                prompt = np.asarray(prompt_token_ids, dtype=np.int64)
                if window is not None:
                    prompt = prompt[-(window + width) :]
                if prompt.shape[0] > width:
                    inside = np.lib.stride_tricks.sliding_window_view(prompt, width)[:-1]
                    # Virtual positions before the first generated one: -len(inside) .. -1.
                    for offset, context in enumerate(inside):
                        last_seen[context.tobytes()] = offset - inside.shape[0]
        for i in range(contexts.shape[0]):
            key = contexts[i].tobytes()
            previous = last_seen.get(key)
            if previous is not None and (window is None or i - previous <= window):
                mask[i] = False
            last_seen[key] = i
        return mask
