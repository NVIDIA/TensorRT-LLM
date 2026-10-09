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
"""DeepGEMM support for ``jit_prefetch.py``: provider, record kind, stats.

Covers the SM100 FP8 block-scale ``Linear`` path
(``trtllm::fp8_swap_ab_gemm`` -> ``deep_gemm.fp8_gemm_nt``), the DeepGEMM call
on the default serving path of the FP8 DeepSeek / GLM models on Blackwell.

DeepGEMM picks a kernel variant from (M, N, K, dtypes, SM count): its tile
layout changes every 16 rows of M. The variant is never chosen here: the
helper calls DeepGEMM's compile-only entry point, which runs DeepGEMM's own
heuristic and writes the cubin into ``DG_JIT_CACHE_DIR`` under the key the
real launch will look up. A request is identified by its DeepGEMM arguments,
so two M values that select the same layout are compiled once (the second is
a disk-cache hit in the helper).

A provider (``FP8LinearDeepGemmProvider``) knows each FP8 Linear's (N, K) from
its weight, and the scheduled batch gives M (its token count; with attention
DP, each rank plans its own M). Record/replay logs every request this process
made or that its executor compiled, so the next process compiles them first.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

from tensorrt_llm.logger import logger

KIND = "deep_gemm"
_OP = "fp8_fp4_gemm_nt"
# How TRT-LLM calls it on SM100: FP8 A/B, BF16 out, UE8M0 int scaling factors,
# so DeepGEMM's default recipe for int SF is (1, 1, 128).
_A, _B, _D, _RECIPE = "float8_e4m3fn", "float8_e4m3fn", "bfloat16", (1, 1, 128)


def spec_for(m: int, n: int, k: int) -> str:
    """Canonical request for one GEMM; equal strings mean the same request."""
    return json.dumps(
        {
            "op": _OP,
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "a": _A,
            "b": _B,
            "d": _D,
            "recipe": list(_RECIPE),
        },
        sort_keys=True,
    )


def supported() -> bool:
    """DeepGEMM is built with the compile-only entry point and runs on SM100."""
    try:
        from tensorrt_llm import deep_gemm
        from tensorrt_llm._utils import get_sm_version
    except ImportError:
        return False
    return hasattr(deep_gemm, "compile_only_fp8_fp4_gemm_nt") and get_sm_version() in (100, 103)


def target() -> Tuple[int, int, int]:
    """(arch_major, arch_minor, num_sms) the helper must compile for.

    The SM count is DeepGEMM's own (``deep_gemm.get_num_sms()``), which is what
    the real launch passes to the heuristic, so it honours ``set_num_sms``.
    """
    import torch

    from tensorrt_llm import deep_gemm

    major, minor = torch.cuda.get_device_capability()
    return major, minor, int(deep_gemm.get_num_sms())


def package_dir() -> str:
    import tensorrt_llm

    return os.path.dirname(os.path.abspath(tensorrt_llm.__file__))


class FP8LinearDeepGemmProvider:
    """Plans the DeepGEMM FP8 GEMMs a batch with M tokens will run."""

    def __init__(self, model, max_num_tokens: int):
        from .modules.linear import FP8BlockScalesLinearMethod, Linear

        shapes = set()
        for mod in model.modules():
            if not isinstance(mod, Linear):
                continue
            if not isinstance(getattr(mod, "quant_method", None), FP8BlockScalesLinearMethod):
                continue
            if getattr(mod, "use_cute_dsl_blockscaling_mm", False) or getattr(
                mod, "disable_deep_gemm", False
            ):
                continue
            w = getattr(mod, "weight", None)
            if w is None or w.dim() != 2:
                continue
            n, k = int(w.shape[0]), int(w.shape[1])
            shapes.add((n, k))
        self.shapes = sorted(shapes)
        self.max_num_tokens = int(max_num_tokens)
        self._seen_m: set = set()

    def __bool__(self):
        return bool(self.shapes)

    def specs_for_m(self, m: int) -> List[str]:
        return [spec_for(m, n, k) for n, k in self.shapes]

    def plan_tokens(self, num_tokens: int) -> List[str]:
        """Requests for a batch of ``num_tokens``; each M is planned once."""
        if num_tokens <= 0 or num_tokens in self._seen_m:
            return []
        self._seen_m.add(num_tokens)
        return self.specs_for_m(num_tokens)

    def enumerate_specs(self, rank: int = 0, world: int = 1):
        """Background coverage: every M up to max_num_tokens.

        DeepGEMM's layout depends on M only through ceil(M / block_m) and the
        candidate block_m are multiples of 16, so one M per window of 16 rows
        reaches every layout: 1..16 individually (the smallest blocks), then
        every 16th M.

        Every rank enumerates every M, so coverage never depends on how ranks
        share ``DG_JIT_CACHE_DIR``. Ranks that do share one start at different
        points of the list (``rank``/``world``), so they compile different
        variants first and find the rest already on disk.
        """
        ms = list(range(1, 17)) + list(range(32, self.max_num_tokens + 1, 16))
        if self.max_num_tokens not in ms:
            ms.append(self.max_num_tokens)
        start = (len(ms) * (rank % max(1, world))) // max(1, world)
        for m in ms[start:] + ms[:start]:
            if m in self._seen_m:
                continue
            for s in self.specs_for_m(m):
                yield s


# Set by JitPrefetcher.enable_deep_gemm: called with (m, n, k) for every real
# FP8 swap-AB launch, so a shape no provider planned still reaches the record,
# and with (m, n, k, host_seconds) after it. The launch is asynchronous, so a
# host time far above a launch's (~10 us) is a compile on the calling thread.
launch_observer = None
done_observer = None
COMPILE_HOST_S = 0.2


def note_launch(m: int, n: int, k: int) -> float:
    obs = launch_observer
    if obs is not None:
        obs(m, n, k)
    return time.perf_counter()


def note_launch_done(m: int, n: int, k: int, t0: float) -> None:
    obs = done_observer
    if obs is not None:
        obs(m, n, k, time.perf_counter() - t0)


def batch_tokens(scheduled_requests) -> int:
    """Token count M the FP8 Linears see for this batch (pre-padding)."""
    n = 0
    for r in scheduled_requests.context_requests:
        n += r.context_chunk_size
    for r in scheduled_requests.generation_requests:
        n += 1 + len(getattr(r, "py_draft_tokens", None) or [])
    return n


def log_unsupported_once(reason: str) -> None:
    logger.info_once(
        f"[JIT prefetch] DeepGEMM prefetch disabled: {reason}", key="jitp_dg_unsupported"
    )
