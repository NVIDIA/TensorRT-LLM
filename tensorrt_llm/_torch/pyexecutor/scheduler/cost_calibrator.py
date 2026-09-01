# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Online calibrator for equal-cost context chunking (see scheduler_v2.py).

Fits the per-iteration cost model ``T = A * (N / M) + c2 * S`` from observed
iteration wall times, where ``N`` is the scheduled token count, ``S`` is the
KV-weighted token count ``sum(n_i * (kv_i + n_i / 2))`` over context chunks,
``M`` is ``max_num_tokens``, ``A`` folds the KV-independent iteration cost and
``c2`` is the full-attention cost per token*kv. Publishes

    kv_cost_offset  = A / (c2 * M)          (kv-token equivalents of the fixed work)
    kv_depth_threshold = P_p of the observed token-weighted KV-depth distribution
    budget = (kv_cost_offset + kv_depth_threshold) * M  (token*kv units)

Notes on estimation:
  - Theil-Sen (median of pairwise slopes) tolerates the stall outliers
    (eviction/onboard pauses) that break least squares.
  - Under attention-DP lockstep the observed wall time is the max over
    ranks while (N, S) are rank-local; the resulting attenuation biases
    c2 towards zero by roughly the DP size (replaying a DEP16 run's logs
    through this estimator recovers kv_cost_offset ~16x above the exactly-attributed
    offline fit). Chunking therefore becomes LESS aggressive — the safe
    direction — but most of the balancing benefit needs either manual mode
    with an offline-fitted kv_cost_offset, or the follow-up that piggybacks per-rank
    (N, S) on the attention-DP allgather and fits on the argmax rank.
  - Until the fit passes the sanity gates, ``budget`` stays None and the
    scheduler keeps pure token budgeting.
"""

from collections import deque
from statistics import median
from typing import Optional

from tensorrt_llm.logger import logger

# Token-weighted KV-depth histogram: 64 buckets x 16k tokens covers 1M depth.
_HIST_BUCKET = 16384
_HIST_SIZE = 64


class ContextCostCalibrator:

    def __init__(
        self,
        max_num_tokens: int,
        kv_depth_threshold_percentile: float = 25.0,
        min_samples: int = 256,
        window: int = 2048,
        refresh_interval: int = 256,
        hysteresis: float = 0.1,
        offset_bounds: tuple = (1_000, 10_000_000),
    ):
        self._m = max_num_tokens
        self._percentile = kv_depth_threshold_percentile
        self._min_samples = min_samples
        self._refresh_interval = refresh_interval
        self._hysteresis = hysteresis
        self._offset_bounds = offset_bounds
        self._samples: deque = deque(maxlen=window)  # (N, S, T_ms)
        self._kv_hist = [0] * _HIST_SIZE
        self._since_refresh = 0

        # Published values; None until the fit passes the sanity gates.
        self.kv_cost_offset: Optional[float] = None
        self.kv_depth_threshold: Optional[float] = None
        self.budget: Optional[float] = None

    # ---- observation ----

    def observe_chunk(self, kv_start: int, num_tokens: int) -> None:
        """Record a scheduled context chunk for the KV-depth histogram."""
        bucket = min(_HIST_SIZE - 1, kv_start // _HIST_BUCKET)
        self._kv_hist[bucket] += num_tokens

    def observe_iteration(self, num_tokens: int, cost_features: float, wall_ms: float) -> None:
        """Record one iteration: scheduled tokens, S = sum(n*(kv+n/2)), wall time."""
        if num_tokens <= 0 or wall_ms <= 0.0:
            return
        self._samples.append((num_tokens, cost_features, wall_ms))
        self._since_refresh += 1

    # ---- refresh ----

    def maybe_refresh(self) -> bool:
        """Re-fit and republish the budget. Returns True when it changed."""
        if self._since_refresh < self._refresh_interval:
            return False
        self._since_refresh = 0
        kv_cost_offset = self._fit()
        if kv_cost_offset is None:
            return False
        kv_depth_threshold = self._kv_percentile(self._percentile)
        if kv_depth_threshold is None:
            return False
        budget = (kv_cost_offset + kv_depth_threshold) * self._m
        if self.budget is not None and abs(budget - self.budget) < self._hysteresis * self.budget:
            return False
        self.kv_cost_offset, self.kv_depth_threshold, self.budget = (
            kv_cost_offset, kv_depth_threshold, budget)
        logger.info(
            f"ContextCostCalibrator: kv_cost_offset={kv_cost_offset:.0f}, kv_depth_threshold={kv_depth_threshold:.0f} "
            f"(p{self._percentile:.0f}), cost_budget={budget:.4e} token*kv "
            f"({len(self._samples)} samples)"
        )
        return True

    def _fit(self) -> Optional[float]:
        if len(self._samples) < self._min_samples:
            return None
        t_med = median(t for _, _, t in self._samples)
        data = [(n, s, t) for n, s, t in self._samples if t < 4.0 * t_med and s > 0]
        if len(data) < self._min_samples // 2:
            return None
        # Theil-Sen slope: normalize T by the token count first (T*M/N), sort
        # by S and pair the k-th sample of the lower half with the k-th of
        # the upper half so every pair spans a large delta-S. The median of
        # pairwise slopes is robust to the stall outliers that survive the
        # 4x filter above.
        data.sort(key=lambda x: x[1])
        half = len(data) // 2
        slopes = []
        for (n0, s0, t0), (n1, s1, t1) in zip(data[:half], data[half:]):
            ds = s1 - s0
            if ds > 0:
                slopes.append((t1 * self._m / n1 - t0 * self._m / n0) / ds)
        if not slopes:
            return None
        c2 = median(slopes)
        if c2 <= 0.0:
            return None
        a = median((t - c2 * s) * self._m / n for n, s, t in data)
        if a <= 0.0:
            return None
        kv_cost_offset = a / (c2 * self._m)
        lo, hi = self._offset_bounds
        if not lo <= kv_cost_offset <= hi:
            return None
        return kv_cost_offset

    def _kv_percentile(self, p: float) -> Optional[float]:
        total = sum(self._kv_hist)
        if total == 0:
            return None
        target = total * p / 100.0
        acc = 0
        for i, cnt in enumerate(self._kv_hist):
            acc += cnt
            if acc >= target:
                return (i + 0.5) * _HIST_BUCKET
        return (_HIST_SIZE - 0.5) * _HIST_BUCKET
