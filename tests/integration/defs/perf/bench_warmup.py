# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Mapping, Optional


def bench_warmup_requests(
    num_gpus: int,
    concurrency: int,
    num_requests: int,
    max_batch_size: int,
    block_reuse: bool,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the unmeasured request count for trtllm-bench throughput."""
    env = os.environ if environ is None else environ
    legacy = 2 * num_gpus if num_gpus > 1 else 0
    if (env.get("TRTLLM_PERF_WARMUP_PASS", "1").strip().lower()
            in ("0", "false", "no", "off")) or block_reuse:
        return legacy
    wave = concurrency if concurrency > 0 else max_batch_size
    if num_requests > 0:
        wave = min(wave, num_requests)
    return max(legacy or 2, wave)
