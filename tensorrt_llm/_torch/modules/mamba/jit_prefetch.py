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
"""Variant provider for the Mamba2 SSD prefill path (Triton kernels).

Given a batch composition, reproduce on ``meta`` tensors exactly the Triton
launches that ``Mamba2Mixer.forward_core`` + ``Mamba2Metadata.prepare`` will
issue for its context requests, and return them as KernelCalls.

The launcher functions themselves (``_chunk_cumsum_fwd`` ... ``chunk_state_varlen``
and ``cu_seqlens_to_chunk_indices_offsets_triton``) are run unchanged under
``shadow_launches()``: their allocation, ``rearrange`` and grid code runs on
meta tensors, and every Triton ``run`` is recorded instead of executed. Only
the tensors entering the SSD are rebuilt here, with the same layout the mixer
produces (views into the in_proj output, token-major conv output).

Batch facts used:
  num context requests, their chunk lengths, whether any has cached tokens.
They decide every specialization that varies at runtime: ``HAS_INITSTATES``,
the multi-sequence chunk-index kernel, and Triton's integer specialization of
``seqlen`` / token counts (== 1, % 16 == 0, otherwise).

Tune ahead (``tune_batches`` / ``run_tune_batch``, see
``JitPrefetcher.tune``): the same launchers run once more on the GPU, on
scratch tensors, at the end of warmup, so the autotune keys of a batch with
cached prefixes are benchmarked before serving.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from ...jit_prefetch import KernelCall, shadow_launches


class _MixerShape:
    """The per-layer constants of one Mamba2Mixer that its kernels see."""

    def __init__(self, mixer):
        self.nheads = mixer.tp_nheads
        self.head_dim = mixer.head_dim
        self.ngroups = mixer.tp_ngroups
        self.d_state = mixer.d_state
        self.chunk_size = mixer.chunk_size
        self.d_inner = mixer.tp_d_inner
        self.conv_dim = mixer.tp_conv_dim
        self.delta_softplus = mixer.delta_softplus
        self.state_dtype = mixer._mamba_ssm_cache_dtype
        self.token_major_conv = mixer._token_major_conv
        self.has_dt_bias = mixer.dt_bias is not None
        self.has_d = mixer.D is not None
        self.io_dtype = (
            mixer.in_proj.weight.dtype
            if getattr(mixer.in_proj, "weight", None) is not None
            and mixer.in_proj.weight.dtype.is_floating_point
            and mixer.in_proj.weight.dtype.itemsize >= 2
            else torch.bfloat16
        )
        self.zxbcdt_width = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    @property
    def triton_ssd(self) -> bool:
        """Whether prefill runs the Triton SSD kernels: the same test as
        ``ssd_combined.mamba_chunk_scan_combined`` (the mixer passes z=None),
        else FlashInfer's fused SSD kernel runs instead."""
        from tensorrt_llm._utils import is_sm_100f

        from .ssd_combined import _flashinfer_ssd_supported, _use_flashinfer_ssd

        return not (
            is_sm_100f()
            and _use_flashinfer_ssd()
            and _flashinfer_ssd_supported(self.chunk_size, self.d_state, self.head_dim)
        )

    def key(self):
        return (
            self.nheads,
            self.head_dim,
            self.ngroups,
            self.d_state,
            self.chunk_size,
            self.d_inner,
            self.conv_dim,
            self.delta_softplus,
            str(self.state_dtype),
            self.token_major_conv,
            self.has_dt_bias,
            self.has_d,
            str(self.io_dtype),
        )


class MambaSSDProvider:
    """Plans the SSD prefill launches of every distinct Mamba2 layer shape."""

    # Largest batch tune ahead runs, in tokens (see tune_batches).
    TUNE_MAX_TOKENS = 4096
    # Warmup's prefills are planned before they run (JitPrefetcher.
    # plan_warmup_batch): the autotuner warmup's compiles every candidate
    # config of the SSD kernels' autotune keys.
    plans_warmup_batches = True

    def __init__(
        self, model, max_num_tokens: int = 0, max_batch_size: int = 0, max_seq_len: int = 0
    ):
        from .mamba2_mixer import Mamba2Mixer

        self.max_num_tokens = int(max_num_tokens)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)

        shapes = {}
        for m in model.modules():
            if isinstance(m, Mamba2Mixer):
                s = _MixerShape(m)
                shapes.setdefault(s.key(), s)
        self.shapes = list(shapes.values())
        self._seen_keys = set()

    def __bool__(self):
        return bool(self.shapes)

    # Modules defining the kernels this provider plans; helpers import them
    # during warmup (JitPrefetcher.register).
    kernel_modules = tuple(
        f"tensorrt_llm._torch.modules.mamba.{m}"
        for m in ("ssd_bmm", "ssd_chunk_state", "ssd_state_passing", "ssd_chunk_scan",
                  "mamba2_metadata")
    )  # fmt: skip

    @staticmethod
    def _int_class(v: int) -> int:
        # Triton's integer specialization: == 1, divisible by 16, other.
        return 1 if v == 1 else (16 if v % 16 == 0 else 0)

    def batch_key(self, ctx_lens: List[int], any_cached: bool) -> tuple:
        """Coarse dedup key: re-plan only when a planned launch could differ.

        Covers every batch-dependent int that reaches a kernel signature:
        token count, sequence count, and the chunk counts derived from them
        (``nchunks`` and the chunk-index length ``N``, which adds one chunk per
        sequence boundary that is not chunk-aligned). Planning itself is exact;
        this only decides whether to plan. A miss here costs one extra plan
        (tens of ms), never a wrong key.
        """
        from .mamba2_metadata import compute_extra_chunks_cpu

        n = len(ctx_lens)
        tokens = sum(ctx_lens)
        classes = []
        for s in self.shapes:
            nchunks = -(-tokens // s.chunk_size)
            extra = compute_extra_chunks_cpu(ctx_lens, n, s.chunk_size)
            classes.append((self._int_class(nchunks), self._int_class(nchunks + extra)))
        return (
            min(n, 2),
            any_cached,
            self._int_class(tokens),
            self._int_class(n),
            self._int_class(n + 1),
            tuple(classes),
        )

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        """Plan one batch. ``seen`` is the caller's dedup set (the background
        enumeration and tune ahead each keep their own)."""
        ctx_lens: List[int] = batch_ctx.ctx_chunk_lens
        if not ctx_lens:
            return []
        any_cached: bool = batch_ctx.any_ctx_cached
        seen = self._seen_keys if seen is None else seen
        key = self.batch_key(ctx_lens, any_cached)
        if key in seen:
            return []
        seen.add(key)
        calls: List[KernelCall] = []
        for s in self.shapes:
            calls.extend(self._plan_one(s, ctx_lens, any_cached))
        return calls

    def enumerate_batches(self) -> Iterator[Any]:
        """One representative batch per ``batch_key`` class reachable within
        the engine's limits (max_num_tokens context tokens, max_batch_size
        sequences).

        The class is everything in a batch that reaches a kernel signature
        (token count, sequence count and the derived chunk counts, through
        Triton's ==1 / %16 / other integer classes, plus whether any request
        has cached state). Candidates cover each class of the sequence count
        and of the token total, split evenly or with a remainder so that both
        chunk-aligned and ragged sequence boundaries occur; duplicates of a
        class are dropped.
        """
        T_max = self.max_num_tokens
        N_max = self.max_batch_size
        if T_max <= 0 or N_max <= 0:
            return
        chunk = self.shapes[0].chunk_size if self.shapes else 128
        n_cands = sorted({n for n in (1, 2, 3, 15, 16, 17, 31, 32, 33) if n <= N_max})
        t_base = (
            1,
            2,
            15,
            16,
            17,
            chunk - 1,
            chunk,
            chunk + 1,
            2 * chunk - 1,
            2 * chunk,
            2 * chunk + 1,
            T_max - 1,
            T_max,
        )
        t_cands = sorted({t for t in t_base if 1 <= t <= T_max})
        seen_cls = set()
        for any_cached in (False, True):
            for n in n_cands:
                for T in t_cands:
                    if T < n:
                        continue
                    q, r = divmod(T, n)
                    splits = [[q] * (n - 1) + [q + r]]
                    if n > 1 and q > 1:
                        # A ragged boundary: move one token between the first two.
                        splits.append(
                            [q - 1, q + 1] + [q] * (n - 3) + [q + r]
                            if n > 2
                            else [q - 1, q + 1 + r]
                        )
                    for lens in splits:
                        if sum(lens) != T or min(lens) < 1:
                            continue
                        cls = self.batch_key(lens, any_cached)
                        if cls in seen_cls:
                            continue
                        seen_cls.add(cls)
                        yield SimpleNamespace(ctx_chunk_lens=lens, any_ctx_cached=any_cached)

    def tune_batches(self) -> Iterator[Any]:
        """The batches tune ahead runs on the GPU (``JitPrefetcher.tune``).

        An SSD kernel's autotune key does not depend on the batch's size or
        sequence count, only on whether some request has a cached prefix (the
        initial states and the chunk indices then join the key of three
        kernels). So two batches reach every key: the largest context batch
        the engine admits, capped at ``TUNE_MAX_TOKENS`` tokens and split into
        ``max_batch_size`` sequences of near-equal length (the split warmup
        uses for a multi-sequence batch), once without and once with a cached
        prefix on every sequence. Each key's config is the autotuner's own
        benchmark on that batch; a batch whose keys warmup already tuned is
        not run. Nothing is yielded if no layer runs the Triton SSD kernels.
        """
        if not any(s.triton_ssd for s in self.shapes):
            return
        T = min(self.max_num_tokens, self.TUNE_MAX_TOKENS)
        if self.max_seq_len > 1:
            T = min(T, self.max_batch_size * (self.max_seq_len - 1))
        n = min(T, self.max_batch_size)
        if T <= 0 or n <= 0:
            return
        q, r = divmod(T, n)
        lens = [q + 1] * r + [q] * (n - r)
        for any_cached in (False, True):
            yield SimpleNamespace(ctx_chunk_lens=lens, any_ctx_cached=any_cached)

    def early_batches(self) -> Iterator[Any]:
        """The batches whose variants are queued before warmup's first
        forward (``JitPrefetcher.queue_early_batches``): tune ahead's batch
        with cached prefixes. No warmup forward reaches its initial-state
        autotune keys, so its candidate configs are compiled while warmup
        runs, and tune ahead only benchmarks them."""
        return (b for b in self.tune_batches() if b.any_ctx_cached)

    def run_tune_batch(self, batch_ctx: Any) -> None:
        """Run ``batch_ctx``'s SSD prefill launches on the GPU, on scratch
        tensors, for every layer shape: each untuned autotune key they reach
        runs the autotuner's benchmark. Nothing outside the scratch tensors
        is read or written."""
        device = torch.device("cuda", torch.cuda.current_device())
        with torch.inference_mode():
            for s in self.shapes:
                if s.triton_ssd:
                    self._launch(s, batch_ctx.ctx_chunk_lens, batch_ctx.any_ctx_cached, device)

    def _plan_one(self, s: _MixerShape, ctx_lens: List[int], any_cached: bool) -> List[KernelCall]:
        with shadow_launches() as rec:
            self._launch(s, ctx_lens, any_cached, torch.device("meta"))
        return list(rec)

    @staticmethod
    def _launch(s: _MixerShape, ctx_lens: List[int], any_cached: bool, device) -> None:
        """Issue one batch's SSD prefill launches for layer shape ``s``.

        The tensors entering the SSD have the layouts Mamba2Mixer.forward_core
        builds. On ``meta`` tensors, under ``shadow_launches``, the launches
        are recorded; on the GPU they run, on zero-filled tensors with the
        batch's sequence boundaries.
        """
        from .mamba2_metadata import (
            compute_extra_chunks_cpu,
            cu_seqlens_to_chunk_indices_offsets_triton,
        )
        from .ssd_combined import _mamba_chunk_scan_combined_fwd

        alloc = torch.empty if device.type == "meta" else torch.zeros
        n = len(ctx_lens)
        T = sum(ctx_lens)
        # in_proj output: [T, zxbcdt_width] contiguous; dt is a column slice.
        zxbcdt = alloc(T, s.zxbcdt_width, dtype=s.io_dtype, device=device)
        dt = zxbcdt[:, s.d_inner + s.conv_dim :].unsqueeze(0)
        bc = s.ngroups * s.d_state
        if s.token_major_conv:
            # Mamba2Mixer.forward_core: empty(T, conv_dim).t() is the conv's
            # channel-last output, and .t() again gives the token-major
            # [T, conv_dim] view that x/B/C are sliced from.
            xbc = alloc(T, s.conv_dim, dtype=s.io_dtype, device=device)
            x = xbc[:, : s.d_inner].view(T, s.nheads, s.head_dim).unsqueeze(0)
            B = xbc[:, s.d_inner : s.d_inner + bc].view(T, s.ngroups, s.d_state).unsqueeze(0)
            C = xbc[:, s.d_inner + bc :].view(T, s.ngroups, s.d_state).unsqueeze(0)
        else:
            x = alloc(T, s.d_inner, dtype=s.io_dtype, device=device).view(
                1, T, s.nheads, s.head_dim
            )
            B = alloc(T, bc, dtype=s.io_dtype, device=device).view(1, T, s.ngroups, s.d_state)
            C = alloc(T, bc, dtype=s.io_dtype, device=device).view(1, T, s.ngroups, s.d_state)
        A = alloc(s.nheads, dtype=torch.float32, device=device)
        D = alloc(s.nheads, dtype=torch.float32, device=device) if s.has_d else None
        dt_bias = alloc(s.nheads, dtype=torch.float32, device=device) if s.has_dt_bias else None
        if device.type == "meta":
            cu_seqlens = torch.empty(n + 1, dtype=torch.int, device=device)
            seq_idx = torch.empty(1, T, dtype=torch.int, device=device)
        else:
            # The kernels index with these: they hold the batch's real
            # boundaries, as Mamba2Metadata.prepare builds them.
            lens = torch.tensor(ctx_lens, dtype=torch.int, device=device)
            cu_seqlens = torch.zeros(n + 1, dtype=torch.int, device=device)
            torch.cumsum(lens, dim=0, dtype=torch.int, out=cu_seqlens[1:])
            seq_idx = torch.repeat_interleave(
                torch.arange(n, dtype=torch.int, device=device), lens, output_size=T
            ).unsqueeze(0)
        out = alloc(T, s.nheads * s.head_dim, dtype=s.io_dtype, device=device).view(
            1, T, -1, s.head_dim
        )
        initial_states: Optional[torch.Tensor] = None
        chunk_indices = chunk_offsets = None

        if any_cached:
            # Mamba2Metadata.prepare: only built when some request has a
            # cached prefix. The multi-seq branch launches the kernel; the
            # single-seq fast path is torch-only.
            extra = compute_extra_chunks_cpu(ctx_lens, n, s.chunk_size)
            chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
                cu_seqlens, s.chunk_size, total_seqlens=T, extra_chunks=extra
            )
            # forward_core: torch.where(...) over the gathered SSM states
            # yields a fresh contiguous tensor in the SSM-cache dtype.
            initial_states = alloc(
                n, s.nheads, s.head_dim, s.d_state, dtype=s.state_dtype, device=device
            )
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            s.chunk_size,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=s.delta_softplus,
            dt_limit=(0.0, float("inf")),
            out=out,
            state_dtype=s.state_dtype,
        )


class MambaGatedNormProvider:
    """Plans the Triton launches of every Mamba2 mixer's gated RMSNorm.

    ``Mamba2Mixer.forward`` normalizes the SSD output (a contiguous
    ``[tokens, d_inner]`` buffer) gated by ``z``, a column slice of the in_proj
    output, through ``RMSNormGated``. Unless the norm runs the fused CUDA NVFP4
    kernel, that is ``_layer_norm_fwd_1pass_kernel``, specialized on the row
    count's integer class (one row, a multiple of 16, other) and on the row
    strides the mixer's layout fixes. Every variant is therefore known from
    the model once it is loaded, and is planned before warmup's first forward
    (``early_batches``), whose prefill and decode steps launch the first ones.
    """

    kernel_modules = ("tensorrt_llm._torch.modules.mamba.layernorm_gated",)
    # A row count of each integer class Triton specializes on.
    ROWS = (1, 16, 17)

    def __init__(self, model):
        from .mamba2_mixer import Mamba2Mixer

        layers = {}
        for m in model.modules():
            if not isinstance(m, Mamba2Mixer) or m.norm.uses_fused_nvfp4_quant(gated=True):
                continue
            s = _MixerShape(m)
            norm = m.norm
            key = (
                s.d_inner,
                s.zxbcdt_width,
                str(s.io_dtype),
                norm.group_size,
                norm.norm_before_gate,
                norm.weight_is_delta,
                norm.fp8_scale is not None,
                str(norm.weight.dtype),
            )
            layers.setdefault(key, (s, norm))
        self.layers = list(layers.values())

    def __bool__(self):
        return bool(self.layers)

    def early_batches(self) -> Iterator[Any]:
        if self.layers:
            yield SimpleNamespace(mamba_gated_norm=True, ctx_chunk_lens=[])

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        if not getattr(batch_ctx, "mamba_gated_norm", False):
            return []
        meta = torch.device("meta")
        with shadow_launches() as rec:
            for s, norm in self.layers:
                for rows in self.ROWS:
                    # Mamba2Mixer.forward: the norm reads the SSD output buffer
                    # and the z columns of the in_proj output.
                    x = torch.empty(rows, s.d_inner, dtype=s.io_dtype, device=meta)
                    zxbcdt = torch.empty(rows, s.zxbcdt_width, dtype=s.io_dtype, device=meta)
                    norm(x, zxbcdt[:, : s.d_inner])
        return list(rec)


def register_providers(prefetcher, engine) -> None:
    """Register the Mamba2 SSD prefill provider of ``engine``'s model, which
    also tunes ahead (``tune_batches``), and its mixers' gated RMSNorm
    provider. Once per process: later calls change nothing."""
    from tensorrt_llm.logger import logger

    provider = MambaSSDProvider(
        engine.model,
        max_num_tokens=engine.max_num_tokens,
        max_batch_size=engine.batch_size,
        max_seq_len=engine.max_seq_len,
    )
    if provider and prefetcher.register("mamba_ssd", provider):
        logger.info(
            f"[JIT prefetch] Mamba SSD provider: {len(provider.shapes)} distinct layer shape(s)"
        )
    gated_norm = MambaGatedNormProvider(engine.model)
    if gated_norm:
        prefetcher.register("mamba_gated_norm", gated_norm)


def tune_ahead_covers_prefill(model) -> bool:
    """Whether JIT tune ahead covers the autotune keys of the Mamba hybrid
    warmup's forced-initial-states pass on this rank: every recurrent layer
    of ``model`` the rank holds is a Mamba2 mixer running the Triton SSD
    kernels (Nemotron-H), whose variants ``MambaSSDProvider`` compiles and
    tunes, or it holds none. GDN and KDA layers read the same initial-state
    metadata and are autotuned on keys no provider tunes ahead."""
    from tensorrt_llm.logger import logger

    from .gdn_mixer import Qwen3NextGatedDeltaNet

    other_mixers: tuple = (Qwen3NextGatedDeltaNet,)
    try:
        from ..kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

        other_mixers += (KimiKDALinearAttention,)
    except ImportError:  # optional dependency (CuTe DSL / FLA)
        pass
    try:
        shapes = MambaSSDProvider(model).shapes
        others = any(isinstance(m, other_mixers) for m in model.modules())
    except Exception as e:  # noqa: BLE001 - JIT compile ahead must never fail startup
        logger.warning(
            "Mamba hybrid warmup kept: cannot tell whether JIT tune ahead covers "
            f"it. {type(e).__name__}: {e}"
        )
        return False
    return all(s.triton_ssd for s in shapes) and not others
