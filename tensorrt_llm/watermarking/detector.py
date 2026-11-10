# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU watermark detection from token ids, without model weights or a GPU."""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from .config import WatermarkConfig
from .prf import PRF_VERSION, token_uniform


@dataclass(frozen=True)
class WatermarkDetectionResult:
    """Scores and a null-model p-value for a token sequence.

    P-values assume independent pseudorandom outputs for distinct scored
    contexts. Detection scores each context once across the whole text, even
    if generation's bounded history has evicted it. Thresholds do not establish
    provenance or provide an adversarial guarantee.
    """

    detected: bool
    p_value: float
    score: float
    num_scored_tokens: int
    algorithm: str
    prf_version: int = PRF_VERSION
    key_scores: tuple[float, ...] = ()


def _gamma_survival(count: int, score: float) -> float:
    """Survival function for Gamma(integer shape=count, scale=1)."""
    if count == 0 or score <= 0:
        return 1.0
    log_score = math.log(score)
    terms = [index * log_score - math.lgamma(index + 1) for index in range(count)]
    largest = max(terms)
    log_p = -score + largest + math.log(sum(math.exp(term - largest) for term in terms))
    return min(1.0, math.exp(log_p))


class WatermarkDetector:
    """Detect a version-1 Gumbel watermark using the generation secret.

    Args:
        config: The exact watermark configuration used for generation.

    For dual-key Gumbel, generation randomly selects one of two domain-separated
    streams at each position. Detection scores both streams and uses a
    Bonferroni-corrected minimum of their Gamma-tail p-values. This conservative
    test remains calibrated under the null for any mixture alpha, unlike
    adding both scores and treating the result as Gamma(count, 1).
    """

    def __init__(self, config: WatermarkConfig):
        self.config = config

    def detect(
        self,
        token_ids: Sequence[int],
        *,
        prompt_token_ids: Sequence[int] = (),
        threshold: float = 0.01,
    ) -> WatermarkDetectionResult:
        """Score generated token ids with an optional original prompt.

        Args:
            token_ids: Generated ids, excluding the prompt.
            prompt_token_ids: Full prompt ids. Without these, the first
                ``context_width`` generated tokens provide unscored context.
            threshold: Reject the unwatermarked null below this p-value.
        """
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one")
        for sequence in (prompt_token_ids, token_ids):
            if any(
                isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token < 2**32
                for token in sequence
            ):
                raise ValueError("Token IDs must be unsigned 32-bit integers")
        width = self.config.context_width
        context = deque((int(token) for token in prompt_token_ids), maxlen=width)
        history: set[tuple[int, ...]] = set()
        num_keys = 2 if self.config.algorithm == "dual_key_gumbel" else 1
        scores = [0.0] * num_keys
        count = 0
        for token in token_ids:
            if len(context) == width:
                context_tuple = tuple(context)
                repeated = context_tuple in history
                history.add(context_tuple)
                if not repeated:
                    for domain in range(num_keys):
                        uniform = token_uniform(self.config.key, context_tuple, token, domain)
                        scores[domain] -= math.log1p(-uniform)
                    count += 1
            context.append(int(token))
        p_value = min(1.0, num_keys * min(_gamma_survival(count, score) for score in scores))
        return WatermarkDetectionResult(
            detected=count > 0 and p_value < threshold,
            p_value=p_value,
            score=max(scores),
            num_scored_tokens=count,
            algorithm=self.config.algorithm,
            key_scores=tuple(scores),
        )
