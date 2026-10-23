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
"""DeepGEMM support for ``jit_prefetch.py``: providers and stats.

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
its weight; M is a batch's token count, and the enumeration covers every M
up to ``max_num_tokens`` (``enumerate_specs``).
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


def target() -> Tuple[int, int, int, int, int]:
    """(arch_major, arch_minor, num_sms, smem_optin_bytes, l2_bytes) the
    helper must compile for (``DJ_COMPILE_TARGET``).

    The SM count is DeepGEMM's own (``deep_gemm.get_num_sms()``), which is what
    the real launch passes to the heuristic, so it honours ``set_num_sms``.
    """
    import torch

    from tensorrt_llm import deep_gemm

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (
        props.major,
        props.minor,
        int(deep_gemm.get_num_sms()),
        int(props.shared_memory_per_block_optin),
        int(props.L2_cache_size),
    )


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

    @classmethod
    def with_shapes(cls, base: "DeepGemmMoEProvider", shapes) -> "DeepGemmMoEProvider":
        """A provider over ``shapes`` only, otherwise like ``base``."""
        p = cls.__new__(cls)
        p.__dict__.update(base.__dict__)
        p.shapes = sorted(shapes)
        return p

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
                _, h, d, fp4, mx, comp, clean, lg, w = v
                specs.append(_spec(op="mqa_logits", num_heads=h, head_dim=d, is_fp4=fp4,
                                   is_mx_sf=mx, compressed=comp, clean_logits=clean,
                                   logits=lg, weights=w))  # fmt: skip
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


class MegaMoEProvider:
    """DeepGEMM MegaMoE (MEGAMOE_DEEPGEMM backend): one variant per BLOCK_M.

    The kernel is compiled with the layer's fixed shape and the block config
    DeepGEMM's heuristic picks from the per-rank token count. Of that config
    only BLOCK_M (and what follows from it) depends on the token count, and it
    takes a handful of values; a rank's token count varies per batch (and per
    rank under attention DP). Each real call records its layer config; this
    provider queues one token count per BLOCK_M the heuristic can pick.
    """

    def __init__(self):
        self._done: set = set()

    @staticmethod
    def band_token_counts(
        num_ranks: int, num_experts: int, max_tokens: int, topk: int, mma_type: str = "fp8xfp4"
    ) -> List[int]:
        """The smallest per-rank token count in [0, max_tokens] for each
        BLOCK_M DeepGEMM picks in that range (its own heuristic, host only)."""
        from tensorrt_llm import deep_gemm

        first = {}
        for n in range(max_tokens + 1):
            block_m = deep_gemm.get_block_m_for_mega_moe(
                num_ranks, num_experts, max_tokens, n, topk, mma_type
            )
            first.setdefault(block_m, n)
        return sorted(first.values())

    def pending_specs(self) -> List[str]:
        from .moe.fused_moe.mega_moe.mega_moe_deepgemm import MEGA_MOE_LAYER_CONFIGS

        specs = []
        for cfg in list(MEGA_MOE_LAYER_CONFIGS):
            ranks, experts, max_tok, topk, hidden, inter, act, clamp, fm, sb, slb, mma = cfg
            for n in self.band_token_counts(ranks, experts, max_tok, topk, mma):
                specs.append(_spec(op="mega_moe", num_ranks=ranks, num_experts=experts,
                                   max_tokens=max_tok, topk=topk, num_tokens=n, hidden=hidden,
                                   inter=inter, activation=act, clamp=clamp, fast_math=fm,
                                   situ_beta=sb, situ_linear_beta=slb, mma_type=mma))  # fmt: skip
        out = [s for s in specs if s not in self._done]
        self._done.update(out)
        return out


class FP8LinearDeepGemmProvider:
    """Plans the DeepGEMM FP8 GEMMs a batch with M tokens will run."""

    def __init__(self, model, max_num_tokens: int, priority_ms=(), first_ms=()):
        from .modules.linear import FP8BlockScalesLinearMethod, Linear

        # Kimi K3's FP8 weight-read projections (attention, shared experts)
        # call the same trtllm::fp8_swap_ab_gemm with BF16 output.
        try:
            from .models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear
        except ImportError:
            _Fp8BlockScaleWeightReadLinear = ()

        shapes = set()
        for mod in model.modules():
            if _Fp8BlockScaleWeightReadLinear and isinstance(mod, _Fp8BlockScaleWeightReadLinear):
                pass
            elif not isinstance(mod, Linear):
                continue
            elif not isinstance(getattr(mod, "quant_method", None), FP8BlockScalesLinearMethod):
                continue
            elif getattr(mod, "use_cute_dsl_blockscaling_mm", False) or getattr(
                mod, "disable_deep_gemm", False
            ):
                continue
            w = getattr(mod, "weight", None)
            if w is None or w.dim() != 2:
                continue
            n, k = int(w.shape[0]), int(w.shape[1])
            shapes.add((n, k))
        # GEMMs a model creates only while its weights load (Kimi K3's FP8
        # weight-read projections): the model names their shapes up front,
        # so a provider built before loading plans them too.
        for mod in model.modules():
            loaded = getattr(mod, "jit_prefetch_fp8_weight_read_shapes", None)
            if not callable(loaded):
                continue
            try:
                shapes.update((int(n), int(k)) for n, k in loaded())
            except Exception as e:  # noqa: BLE001 - planned once loaded instead
                logger.warning(
                    f"[JIT prefetch] FP8 weight-read shapes not predicted: {type(e).__name__}: {e}"
                )
        self.shapes = sorted(shapes)
        self.max_num_tokens = int(max_num_tokens)
        # Token counts known to run before serving (the CUDA graph shapes):
        # enumerated first, so they are built before warmup reaches them;
        # before them, the token counts of warmup's first forwards.
        self.first_ms = sorted({int(m) for m in first_ms if 0 < int(m) <= self.max_num_tokens})
        self.priority_ms = sorted(
            {int(m) for m in priority_ms if 0 < int(m) <= self.max_num_tokens} - set(self.first_ms)
        )

    @classmethod
    def with_shapes(cls, base: "FP8LinearDeepGemmProvider", shapes) -> "FP8LinearDeepGemmProvider":
        """A provider over ``shapes`` only, otherwise like ``base`` (its
        limits and priority token counts)."""
        p = cls.__new__(cls)
        p.__dict__.update(base.__dict__)
        p.shapes = sorted(shapes)
        return p

    def __bool__(self):
        return bool(self.shapes)

    def specs_for_m(self, m: int) -> List[str]:
        return [spec_for(m, n, k) for n, k in self.shapes]

    def enumerated_m(self, m: int) -> int:
        """The M ``enumerate_specs`` builds for ``m``'s layout window: ``m``
        itself up to 16 rows, else the window's last row (``max_num_tokens``
        in the last window). An M beyond ``max_num_tokens`` is returned as is."""
        if m <= 16 or m > self.max_num_tokens:
            return m
        return min(-(-m // 16) * 16, self.max_num_tokens)

    def enumerate_specs(self, rank: int = 0, world: int = 1):
        """Background coverage: every M up to max_num_tokens.

        DeepGEMM's layout depends on M only through ceil(M / block_m) and the
        candidate block_m are multiples of 16, so one M per window of 16 rows
        reaches every layout: 1..16 individually (the smallest blocks), then
        every 16th M.

        Every rank enumerates every M, so coverage never depends on how ranks
        share ``DG_JIT_CACHE_DIR``. Ranks that do share one start at different
        points of the list (``rank``/``world``), so they compile different
        variants first and find the rest already on disk. ``priority_ms``
        come before all of them, in the same rotated order, and ``first_ms``
        before those, in the same order on every rank: each rank launches
        them in its first forwards, and waits for its own helpers' builds.
        """
        ms = list(range(1, 17)) + list(range(32, self.max_num_tokens + 1, 16))
        if self.max_num_tokens not in ms:
            ms.append(self.max_num_tokens)
        world = max(1, world)
        early = set(self.first_ms) | set(self.priority_ms)
        rest = [m for m in ms if m not in early]
        order = list(self.first_ms)
        for part in (self.priority_ms, rest):
            start = (len(part) * (rank % world)) // world
            order += part[start:] + part[:start]
        for m in order:
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


def log_unsupported_once(reason: str) -> None:
    logger.info_once(
        f"[JIT prefetch] DeepGEMM prefetch disabled: {reason}", key="jitp_dg_unsupported"
    )
