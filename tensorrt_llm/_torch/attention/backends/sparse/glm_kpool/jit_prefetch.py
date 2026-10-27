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
"""Triton variant provider for the glm_kpool indexer kernels (GLM-5-Next).

``kpool_score`` (the costliest kernel to compile: several seconds),
``kpool_update`` and ``kpool_expand`` run in every sparse layer of every
phase. The batch-independent part of each launch class (phase, pool layout,
dtypes) is recorded by the warmup forwards in ``kernels.KPOOL_LAUNCH_TEMPLATE``.
What still varies from batch to batch is what Triton specializes on:

* ``kpool_score``'s row count ``num_rows``: == 1, divisible by 16, or other;
* the 16-byte alignment of per-row inputs that start at a batch-dependent
  element: in decode, ``kv_lens[num_contexts:]`` (also passed as the request
  ids); in verify with one request, a step of the positions.

Every combination is planned once, in the background, by replaying the real
launchers on ``meta`` tensors built from the template. Pointers that are fresh
allocations or fixed views (pools, block tables, the query and its head
weights) keep the alignment they were recorded with.

``flashinfer_modules`` names the FlashInfer JIT module the backend's native
decode path builds (``fmha_gen``) when the model's geometry selects that path,
so the prefetcher can build it in a CPU-only helper while the weights load.
"""

from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import torch

from .....jit_prefetch import KernelCall, shadow_launches
from . import kernels


def flashinfer_modules(model) -> List[str]:
    """FlashInfer JIT modules ``model``'s glm_kpool layers build at their first
    native decode: ``["fmha_gen"]`` if any layer runs small generation batches
    on FlashInfer's TRTLLM-GEN MLA decode kernel, else none.

    That path takes a BF16 latent cache and a TP-local query geometry the
    kernel supports (``GlmKpoolSparseAttention.native_decode_geometry``). Its
    launcher module is an nvcc build of about a minute that each rank pays in
    warmup's first forward with a generation request, before which only the
    model's geometry is needed: the layers' head counts and the KV cache
    dtype, which follows from the model's quantization (FP8 / NVFP4 KV) or
    else its dtype, as the cache manager derives it.
    """
    from .backend import GlmKpoolSparseAttention

    model_config = getattr(model, "model_config", None)
    if model_config is None:
        return []
    quant_mode = model_config.quant_config.quant_mode
    if quant_mode.has_fp8_kv_cache() or quant_mode.has_fp4_kv_cache():
        return []
    if model_config.torch_dtype != torch.bfloat16:
        return []
    for mod in model.modules():
        backend = getattr(mod, "attn_backend", None)
        if isinstance(backend, GlmKpoolSparseAttention) and backend.native_decode_geometry(
            backend.num_heads, backend.head_dim
        ):
            return ["fmha_gen"]
    return []


# Row counts covering each of Triton's integer classes (1, %16, other). The
# row count of kpool_update / kpool_expand only sizes the grid.
_SCORE_ROWS = (1, 16, 17)


def _strided(shape, stride, aligned: bool, dtype: torch.dtype) -> torch.Tensor:
    """A meta tensor with ``shape``/``stride``, 16-byte aligned or not."""
    offset = 0 if aligned else 1
    span = 1 + sum((s - 1) * st for s, st in zip(shape, stride) if s > 0)
    base = torch.empty(offset + span, dtype=dtype, device="meta")
    return base.as_strided(tuple(shape), tuple(stride), offset)


def _rows(n: int, dtype: torch.dtype, aligned: bool) -> torch.Tensor:
    """A contiguous ``[n]`` per-row input."""
    return _strided((n,), (1,), aligned, dtype)


def _alignments(row_input: Optional[tuple], always_both: bool = False) -> tuple:
    """Alignments a recorded per-row input can take at serve time: a slice of
    a longer buffer may start at any element, a whole buffer is aligned."""
    if row_input is None:
        return (None,)
    _dtype, is_slice = row_input
    return (True, False) if (is_slice or always_both) else (True,)


class GlmKpoolProvider:
    """Plans every batch-dependent variant of the glm_kpool indexer kernels."""

    kernel_modules = ("tensorrt_llm._torch.attention.backends.sparse.glm_kpool.kernels",)
    # Enumerated again after every warmup (JitPrefetcher.start_background):
    # the KV-cache estimation pass and the final executor warm up on
    # different pools, so each records launch classes of its own.
    plans_from_warmup = True

    def __init__(self, max_num_tokens: int):
        self.max_num_tokens = max(1, int(max_num_tokens))
        self._planned: set = set()

    def __bool__(self):
        return bool(kernels.KPOOL_LAUNCH_TEMPLATE)

    def __call__(self, batch_ctx: Any, seen: Optional[set] = None) -> List[KernelCall]:
        # Every variant is planned by the background enumeration.
        if not getattr(batch_ctx, "glm_kpool_all", False):
            return []
        return self.plan_all()

    def enumerate_batches(self) -> Iterator[Any]:
        if self:
            yield SimpleNamespace(glm_kpool_all=True, ctx_chunk_lens=[])

    def plan_all(self) -> List[KernelCall]:
        """Plan every recorded launch class not planned before."""
        planners = {"score": self._plan_score, "update": self._plan_update,
                    "expand": self._plan_expand}  # fmt: skip
        with shadow_launches() as rec:
            for name, plan in planners.items():
                for key, t in list(kernels.KPOOL_LAUNCH_TEMPLATE.get(name, {}).items()):
                    if (name, key) in self._planned:
                        continue
                    self._planned.add((name, key))
                    plan(t)
        return list(rec)

    @staticmethod
    def _pool(layout: tuple) -> torch.Tensor:
        shape, stride, aligned, dtype = layout
        return _strided(shape, stride, aligned, dtype)

    @staticmethod
    def _tables(n: int, layout: tuple) -> torch.Tensor:
        # The kernels read only the pointer and the row stride.
        row_stride, aligned, dtype = layout
        return _strided((n, 1), (row_stride, 1), aligned, dtype)

    def _plan_score(self, t: dict) -> None:
        pool = self._pool(t["index_pool"])
        hd, H = t["head_dim"], t["num_heads"]
        q_dtype, q_aligned = t["q"]
        w_row, w_head, w_aligned, w_dtype = t["weights"]
        for n in (r for r in _SCORE_ROWS if r <= self.max_num_tokens):
            q = _strided((n, H, hd), (H * hd, hd, 1), q_aligned, q_dtype)
            w = _strided((n, H), (w_row, w_head), w_aligned, w_dtype)
            tables = self._tables(n, t["block_tables"])
            for lens_aligned in _alignments(t["kv_lens"]):
                lens = _rows(n, t["kv_lens"][0], lens_aligned)
                for req_aligned in _alignments(t["request_ids"]):
                    req = (
                        None if req_aligned is None else _rows(n, t["request_ids"][0], req_aligned)
                    )
                    kernels.kpool_score(
                        q, w, pool, tables, lens, t["tokens_per_block"],
                        num_pools_max=t["num_pools_max"], head_dim=hd, kpool=t["kpool"],
                        q_scale=t["q_scale"], w_scale=t["w_scale"], precision=t["precision"],
                        rows_per_program=t["rows_per_program"], request_ids=req,
                    )  # fmt: skip

    def _plan_update(self, t: dict) -> None:
        pool = self._pool(t["index_pool"])
        ape = self._pool(t["ape"])
        n = 2  # the row count only sizes the grid
        tables = self._tables(n, t["block_tables"])
        # A one-request verify step passes a strided view of the positions,
        # so both alignments are planned whatever warmup recorded.
        for pos_aligned in _alignments(t["positions"], always_both=True):
            pos = _rows(n, t["positions"][0], pos_aligned)
            for req_aligned in _alignments(t["request_ids"]):
                req = None if req_aligned is None else _rows(n, t["request_ids"][0], req_aligned)
                kernels.kpool_update(pool, tables, pos, ape, t["tokens_per_block"],
                                     head_dim=t["head_dim"], kpool=t["kpool"],
                                     request_ids=req)  # fmt: skip

    def _plan_expand(self, t: dict) -> None:
        n = 2  # the row count only sizes the grid
        select_k, sel_stride, sel_aligned, sel_dtype = t["selected"]
        sel = _strided((n, select_k), (sel_stride, 1), sel_aligned, sel_dtype)
        tables = self._tables(n, t["block_tables"])
        for lens_aligned in _alignments(t["kv_lens"]):
            lens = _rows(n, t["kv_lens"][0], lens_aligned)
            for req_aligned in _alignments(t["request_ids"]):
                req = None if req_aligned is None else _rows(n, t["request_ids"][0], req_aligned)
                kernels.kpool_expand(sel, lens, tables, t["tokens_per_block"],
                                     base_row=t["base_row"], rows_per_slot=t["rows_per_slot"],
                                     kpool=t["kpool"], out_width=t["out_width"],
                                     request_ids=req)  # fmt: skip


def register_providers(prefetcher, engine) -> None:
    """Register the kpool provider at the end of warmup, if warmup's forwards
    launched the kpool kernels (it plans from those launches,
    ``kernels.KPOOL_LAUNCH_TEMPLATE``)."""
    if kernels.KPOOL_LAUNCH_TEMPLATE:
        prefetcher.register("glm_kpool", GlmKpoolProvider(engine.max_num_tokens))
