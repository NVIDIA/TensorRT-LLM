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


_MOE_OP = "m_grouped_fp8_fp4_gemm_nt_masked"


def moe_spec_for(g: int, m: int, n: int, k: int, expected_m: int) -> str:
    """Request for one masked grouped GEMM (DeepGEMM MoE, ``[G, M, K] @ [G, N, K].mT``)."""
    return json.dumps(
        {
            "op": _MOE_OP,
            "g": int(g),
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "expected_m": int(expected_m),
            "a": _A,
            "b": _B,
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
    return (
        hasattr(deep_gemm, "compile_only_fp8_fp4_gemm_nt")
        and hasattr(deep_gemm, "compile_only_m_grouped_fp8_fp4_gemm_nt_masked")
        and get_sm_version() in (100, 103)
    )


def supports_mega_moe() -> bool:
    from tensorrt_llm import deep_gemm

    return supported() and hasattr(deep_gemm, "compile_only_fp8_fp4_mega_moe")


def supports_indexer() -> bool:
    """The patched DeepGEMM also has the DSA indexer compile-only entries."""
    from tensorrt_llm import deep_gemm

    return supported() and hasattr(deep_gemm, "compile_only_paged_mqa_logits")


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


class DeepGemmMoEProvider:
    """The masked grouped GEMMs of every DeepGEMM MoE layer.

    On SM100 the masked grouped GEMM has a single layout candidate (swap-AB,
    block M fixed by the contiguous-layout alignment, cluster N from N), and
    M is not a compiled dimension, so the kernel depends on (G, N, K) only:
    one request per weight shape covers every batch. ``M`` / ``expected_m``
    in the request are placeholders the kernel does not depend on.
    """

    def __init__(self, model):
        from .moe.fused_moe.fused_moe_deepgemm import DeepgemmCudaFp8BlockScalesImpl

        shapes = set()
        for mod in model.modules():
            if not isinstance(mod, DeepgemmCudaFp8BlockScalesImpl):
                continue
            for name in ("w3_w1_weight", "w2_weight"):
                w = getattr(mod, name, None)
                if w is not None and w.dim() == 3:
                    shapes.add(tuple(int(x) for x in w.shape))
        self.shapes = sorted(shapes)

    def __bool__(self):
        return bool(self.shapes)

    def enumerate_specs(self):
        for g, n, k in self.shapes:
            yield moe_spec_for(g, 128, n, k, 1)


def _spec(**kw) -> str:
    return json.dumps(kw, sort_keys=True)


class DsaIndexerProvider:
    """DeepGEMM kernels of the DSA indexer (DeepSeek V3.2 / V4, GLM-5).

    The prefill logits kernel has one variant per config; the paged decode
    logits kernel and its scheduling-metadata kernel vary only with
    ``next_n``: 1 for plain decode and MTP draft steps, ``1 + max_draft`` for
    the MTP verify step. Batch size and context length are runtime arguments.
    Each real call records its compiled-in values in
    ``indexer.DG_INDEXER_VARIANTS``; this provider queues every variant those
    imply, including the ``next_n`` the process has not run yet.
    """

    def __init__(self, max_draft_tokens: int, num_sms: int):
        self.next_ns = sorted({1, 1 + max(0, int(max_draft_tokens))})
        self.num_sms = int(num_sms)
        self._done: set = set()

    def pending_specs(self) -> List[str]:
        from .attention.backends.sparse.dsa.indexer import DG_INDEXER_VARIANTS

        specs = []
        for v in list(DG_INDEXER_VARIANTS):
            if v[0] == "mqa":
                _, h, d, fp4, mx, comp, lg, w = v
                specs.append(_spec(op="mqa_logits", num_heads=h, head_dim=d, is_fp4=fp4,
                                   is_mx_sf=mx, compressed=comp, logits=lg, weights=w))  # fmt: skip
            else:
                _, _n, h, d, bkv, fp4, mx, varlen, lg, w = v
                for n in self.next_ns:
                    specs.append(_spec(op="paged_mqa_logits", next_n=n, num_heads=h, head_dim=d,
                                       block_kv=bkv, is_fp4=fp4, is_mx_sf=mx, is_varlen=varlen,
                                       logits=lg, weights=w))  # fmt: skip
        for n in self.next_ns:
            specs.append(_spec(op="paged_mqa_logits_metadata", next_n=n, is_varlen=False,
                               num_sms=self.num_sms))  # fmt: skip
        out = [s for s in specs if s not in self._done]
        self._done.update(out)
        return out


# get_block_config_for_mega_moe switches its tile config at these expected
# tokens per expert (num_tokens * num_ranks * num_topk / num_experts).
_MEGA_MOE_BANDS = (8.5, 16.5, 32.5, 64.5, 96.5)


class MegaMoEProvider:
    """DeepGEMM MegaMoE (MEGAMOE_DEEPGEMM backend): one variant per token band.

    The kernel is compiled with the layer's fixed shape and the block config
    the heuristic picks from the per-rank token count, which changes in six
    bands; a rank's token count varies per batch (and per rank under
    attention DP). Each real call records its layer config; this provider
    queues one representative token count per band for each.
    """

    def __init__(self):
        self._done: set = set()

    @staticmethod
    def band_token_counts(
        num_ranks: int, num_experts: int, max_tokens: int, topk: int
    ) -> List[int]:
        per_token = num_ranks * topk / num_experts
        out, lo = [], 0.0
        for hi in _MEGA_MOE_BANDS + (float("inf"),):
            # Smallest token count whose expected tokens/expert lies in (lo, hi].
            n = max(1, int(lo / per_token) + 1)
            if n * per_token <= hi and n <= max_tokens:
                out.append(n)
            lo = hi
        return sorted(set(out))

    def pending_specs(self) -> List[str]:
        from .moe.fused_moe.mega_moe.mega_moe_deepgemm import MEGA_MOE_LAYER_CONFIGS

        specs = []
        for cfg in list(MEGA_MOE_LAYER_CONFIGS):
            ranks, experts, max_tok, topk, hidden, inter, act, clamp, fm, sb, slb = cfg
            for n in self.band_token_counts(ranks, experts, max_tok, topk):
                specs.append(_spec(op="mega_moe", num_ranks=ranks, num_experts=experts,
                                   max_tokens=max_tok, topk=topk, num_tokens=n, hidden=hidden,
                                   inter=inter, activation=act, clamp=clamp, fast_math=fm,
                                   situ_beta=sb, situ_linear_beta=slb))  # fmt: skip
        out = [s for s in specs if s not in self._done]
        self._done.update(out)
        return out


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
# and with (m, n, k, host_seconds) after it. The MoE masked grouped GEMM calls
# only ``done_observer``, with m = -num_groups. The launch is asynchronous, so a
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
