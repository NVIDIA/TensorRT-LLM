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
"""JIT prefetch provider for MiniMax-M3's MSA prefill kernels (``fmha_sm100``).

The MSA backend runs a step's context rows on the ``fmha_sm100`` package
(``fmha/msa_prefill.py``, ``msa_indexer.py``), which builds its native
kernels with nvcc the first time a call needs them, each into its own
directory of a cache keyed by the package sources (``fmha_sm100.jit
.CACHE_BASE``):

* the planner module, which every plan call runs;
* one FMHA variant per (q dtype, q tile, single warp group, sparse mode, page
  size, split-KV, pack factor). The indexer's proxy max-score pass ("score
  only") and the dense layers' paged attention ("off") pick theirs from the
  batch's longest context row; the proxy pass also has a KV-split form for
  rows with a long cached prefix. A variant is two nvcc compiles, 80-100 s;
* the NVFP4 sparse prefill's k2q CSR builder, a ``torch.utils.cpp_extension``
  module (about 50 s).

Which of them a step builds depends on its longest context row, so a cold
context server builds a dozen of them one after another: most in warmup, the
rest in the first serving steps that reach a new row class, each one stalling
every rank for one to two minutes.

:class:`MiniMaxM3FmhaProvider` lists every one of them a context step of this
engine can reach. It needs only the engine's geometry (each MSA layer's
per-rank head counts, the KV cache's dtypes and page size), so the prefetcher
starts building them at the beginning of warmup, before the first forward.
``jit_prefetch_fmha_sm100_helper.py`` builds them in CPU-only helper processes, into
the directories ``fmha_sm100`` itself reads, holding the same per-kernel file
lock its in-process build takes: a forward that needs a kernel a helper is
building waits for that build instead of compiling it again. Which variant a
call runs is still chosen by ``fmha_sm100`` itself; it finds it built.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

KIND = "fmha_sm100"

# fmha_sm100's sparse modes (fmha_sm100.api._fmha_sm100): block-sparse
# attention, max score only (the indexer's proxy pass), plain attention.
_SPARSE, _ONLY_SCORE, _OFF = 0, 2, 3

# DLPack dtype codes, as fmha_sm100.jit._dlpack_dtype_code encodes them.
_DTYPE_CODES = {"bfloat16": (4 << 16) | (16 << 8) | 1, "float8_e4m3fn": (12 << 16) | (8 << 8) | 1}

# fmha_sm100's head pack factors (fmha_sm100.jit._PACK_FACTORS), largest first.
_PACK_FACTORS = (16, 8, 6, 4, 2, 1)

# Context rows longer than this select the same variants as a row of this
# length: head packing stops at 32 rows, the 128-row q tile at 128 packed rows.
_LONGEST_DISTINCT_ROW = 129
# Block-sparse rows longer than this are planned for the CuTe DSL sparse
# prefill instead (fmha_sm100.api._prefill_qlen_threshold).
_SPARSE_PREFILL_ROWS = 32
# Queries per segment of a KV-split proxy plan (msa_indexer._PROXY_SEGMENT_LEN).
_PROXY_SEGMENT_ROWS = 128


def variant_spec(dtype_code, qo_tile_size, single_wg, sparse_mode, page_size, split_kv, pack):
    """Request for one FMHA variant, in ``fmha_sm100.jit.get_fmha_variant``'s
    arguments; equal strings mean the same variant."""
    args = [
        int(dtype_code),
        int(qo_tile_size),
        bool(single_wg),
        int(sparse_mode),
        int(page_size),
        bool(split_kv),
        int(pack),
    ]
    return json.dumps({"op": "variant", "args": args}, sort_keys=True)


def module_spec(name: str) -> str:
    """Request for a kernel module: fmha_sm100's planner (``plan``) or the
    NVFP4 sparse prefill's k2q CSR builder (``k2q_csr``)."""
    return json.dumps({"op": name}, sort_keys=True)


def pack_factor(max_qo_len: int, num_qo_heads: int, num_kv_heads: int) -> int:
    """``fmha_sm100.api._compute_pack_factor``: query heads of one KV head
    packed into a tile row when the context rows are short."""
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in _PACK_FACTORS:
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1


def row_variant(
    dtype: str, num_qo_heads: int, num_kv_heads: int, sparse_mode: int, page_size: int, qo_len: int
) -> Optional[Tuple]:
    """The ``get_fmha_variant`` key of a call whose longest context row has
    ``qo_len`` tokens, or None where that call runs no FMHA variant.

    Mirrors ``fmha_sm100.api._fmha_sm100_plan`` and ``_fmha_sm100`` with
    ``num_kv_splits=1``, as the MSA plans pass it except the indexer's
    segmented proxy plan (:func:`split_proxy_variant`). A block-sparse plan
    expands each query token into a row of ``pack`` packed heads; past
    ``_SPARSE_PREFILL_ROWS`` it is planned for the CuTe DSL sparse prefill,
    which runs no variant.
    """
    pack = pack_factor(qo_len, num_qo_heads, num_kv_heads)
    if sparse_mode == _SPARSE:
        if qo_len > _SPARSE_PREFILL_ROWS:
            return None
        packed = pack
    else:
        packed = qo_len * pack
    tile = 128 if packed <= 128 else 256
    return (_DTYPE_CODES[dtype], tile, packed <= 64, sparse_mode, page_size, False, pack)


def split_proxy_variant(dtype: str, page_size: int) -> Tuple:
    """The ``get_fmha_variant`` key of the indexer's KV-split proxy plan
    (``msa_indexer.segmented_proxy_plan``), whatever its split count.

    It cuts rows into segments of at most 128 queries and is used only when a
    row is longer than 64, so its longest segment selects the 128-query tile
    with both warp groups, and no heads are packed past 32 queries. With more
    than one KV split, ``fmha_sm100`` runs the split-KV form of that variant;
    with one, the unsplit one, which :func:`row_variant` gives for such a row.
    """
    return (_DTYPE_CODES[dtype], 128, False, _ONLY_SCORE, page_size, True, 1)


def _row_order() -> List[int]:
    """Context row lengths, one per variant class, in the order warmup first
    needs them: single rows, then long ones, then the rest."""
    return [1, _LONGEST_DISTINCT_ROW] + list(range(2, _LONGEST_DISTINCT_ROW))


def planned_variant_keys(sites, page_size: int) -> List[Tuple]:
    """Every ``get_fmha_variant`` key the MSA call sites ``sites`` ((q dtype,
    query heads, KV heads, sparse mode) per rank) can select, in build order:
    first the two variants of the indexer's segmented proxy plan, unsplit and
    KV-split, which warmup loads before its first forward
    (``msa_indexer.prewarm_split_proxy_variants``), then one per row class."""
    keys: List[Tuple] = []

    def add(key):
        if key is not None and key not in keys:
            keys.append(key)

    for dtype, heads, kv_heads, mode in sites:
        if mode == _ONLY_SCORE:
            add(row_variant(dtype, heads, kv_heads, mode, page_size, _PROXY_SEGMENT_ROWS))
            add(split_proxy_variant(dtype, page_size))
    for qo_len in _row_order():
        for dtype, heads, kv_heads, mode in sites:
            add(row_variant(dtype, heads, kv_heads, mode, page_size, qo_len))
    return keys


class MiniMaxM3FmhaProvider:
    """Every ``fmha_sm100`` kernel a MiniMax-M3 MSA context step can build.

    ``specs`` lists them in build order (the planner first: every context step
    plans before it attends). Built from the model's MSA attention layers and
    the KV cache manager, before the first warmup forward.
    """

    def __init__(self, model: Any, kv_cache_manager: Any):
        from tensorrt_llm._utils import is_sm_100f
        from tensorrt_llm.bindings import DataType

        from .kernels.msa_utils import msa_package_available
        from .msa_backend import MiniMaxM3MsaSparseAttention

        self.specs: List[str] = []
        self.sites: List[Tuple] = []
        if not is_sm_100f() or not msa_package_available() or kv_cache_manager is None:
            return
        page_size = int(getattr(kv_cache_manager, "tokens_per_block", 0) or 0)
        if page_size <= 0:
            return
        kv_dtype = getattr(kv_cache_manager, "dtype", None)
        # The main attention's q is cast to the paged K/V dtype: FP8 for an
        # FP8 cache and for the FP8 dense layers of an NVFP4 cache
        # (MiniMaxM3MsaSparseAttentionMetadata._msa_main_kv_is_fp8).
        main = "float8_e4m3fn" if kv_dtype in (DataType.FP8, DataType.NVFP4) else "bfloat16"
        index = (
            "float8_e4m3fn"
            if str(getattr(kv_cache_manager, "indexer_kv_dtype", "bf16")) == "fp8"
            else "bfloat16"
        )
        nvfp4 = kv_dtype == DataType.NVFP4
        sites = []
        for mod in model.modules() if model is not None else ():
            # The model's MiniMaxM3Attention layers: self.attn is the MSA
            # backend, is_sparse_attention_layer whether the layer has the
            # index branch (layers 3 and up) or attends densely.
            attn = mod.__dict__.get("attn")
            if not isinstance(attn, MiniMaxM3MsaSparseAttention):
                continue
            cfg = attn.m3_config
            if bool(getattr(mod, "is_sparse_attention_layer", False)):
                # The indexer's proxy pass over the MQA index-K cache.
                sites.append((index, cfg.num_index_heads, 1, _ONLY_SCORE))
                # The block-sparse attention of short rows; an NVFP4 cache's
                # sparse layers attend through the k2q CSR kernel instead.
                if not nvfp4:
                    sites.append((main, cfg.num_q_heads, cfg.num_kv_heads, _SPARSE))
            else:
                # Dense layers attend the full page table.
                sites.append((main, cfg.num_q_heads, cfg.num_kv_heads, _OFF))
        self.sites = list(dict.fromkeys(sites))
        if not self.sites:
            return
        self.specs.append(module_spec("plan"))
        if nvfp4 and any(site[3] == _ONLY_SCORE for site in self.sites):
            self.specs.append(module_spec("k2q_csr"))
        self.specs += [variant_spec(*key) for key in planned_variant_keys(self.sites, page_size)]

    def __bool__(self):
        return bool(self.specs)


def fmha_jit_module():
    """The ``fmha_sm100.jit`` module this process uses, imported the way the
    MSA backend imports it."""
    from .kernels.msa_utils import require_msa_module

    return require_msa_module().jit


def flashinfer_modules(model: Any) -> List[str]:
    """FlashInfer JIT modules a MiniMax-M3 MSA engine builds in warmup:
    ``["fmha_gen"]``, FlashInfer's TRTLLM-GEN attention launcher, for any
    model with MSA attention layers on an SM100-family GPU, else none.

    The MSA backend runs the decode rows of the dense layers (0-2) on
    ``flashinfer.decode.trtllm_batch_decode_with_kv_cache``
    (``kernels/trtllm_gen_dense_decode.py``; the backend refuses to start if
    that path cannot run). Its launcher module is an nvcc build of about a
    minute that each rank pays in warmup's first forward with a generation
    request (``_run_attention_warmup``), on context-only servers as well.
    Only the model's layers are needed to know this, so the prefetcher builds
    it in a CPU-only helper while the weights load.
    """
    from tensorrt_llm._utils import is_sm_100f

    from .kernels.msa_utils import msa_package_available
    from .kernels.trtllm_gen_dense_decode import flashinfer_available
    from .msa_backend import MiniMaxM3MsaSparseAttention

    if model is None or not is_sm_100f() or not msa_package_available():
        return []
    if not flashinfer_available():
        return []
    for mod in model.modules():
        if isinstance(mod.__dict__.get("attn"), MiniMaxM3MsaSparseAttention):
            return ["fmha_gen"]
    return []


def kv_cache_geometry(model: Any, llm_args: Any) -> Optional[Any]:
    """What :class:`MiniMaxM3FmhaProvider` reads from the KV cache manager
    (page size, cache dtype, indexer dtype), derived from the engine's
    arguments and the model's quantization the way the executor derives them
    when it creates the manager, so the builds can start once the model is
    constructed, while its weights load. None if the engine has no MSA
    attention or a value is missing."""
    from tensorrt_llm.bindings import DataType

    sparse_cfg = getattr(llm_args, "sparse_attention_config", None)
    if (
        getattr(sparse_cfg, "algorithm", None) != "minimax_m3"
        or getattr(sparse_cfg, "implementation", None) != "msa"
    ):
        return None
    # The executor sets kv_cache_config.tokens_per_block to MSA's page size
    # (128) before the model loads (py_executor_creator).
    tokens_per_block = getattr(getattr(llm_args, "kv_cache_config", None), "tokens_per_block", None)
    model_config = getattr(model, "model_config", None)
    if not tokens_per_block or model_config is None:
        return None
    # As create_kv_cache_manager: an FP8 or NVFP4 KV cache from the
    # (overridden) KV quantization, else an unquantized one, which the
    # planner treats alike (BF16 q, as fmha_sm100 has no other).
    quant_mode = model_config.quant_config.quant_mode
    if quant_mode.has_fp8_kv_cache():
        dtype = DataType.FP8
    elif quant_mode.has_fp4_kv_cache():
        dtype = DataType.NVFP4
    else:
        dtype = DataType.BF16
    return SimpleNamespace(
        tokens_per_block=int(tokens_per_block),
        dtype=dtype,
        indexer_kv_dtype=str(getattr(sparse_cfg, "indexer_kv_dtype", "bf16")),
    )


def start_model_builds(model: Any, llm_args: Any) -> bool:
    """Start building the MSA ``fmha_sm100`` kernels of ``model`` once it is
    constructed, while its weights load, from the KV cache geometry
    ``llm_args`` describes (``kv_cache_geometry``). Once per process; returns
    False if nothing was started (no MSA attention, or a value is missing:
    they then start at warmup, ``start_warmup_builds``)."""
    from .....jit_prefetch import FmhaSm100Builds

    if FmhaSm100Builds.get() is not None:
        return False
    geometry = kv_cache_geometry(model, llm_args)
    return geometry is not None and FmhaSm100Builds.start(MiniMaxM3FmhaProvider(model, geometry))


def start_warmup_builds(model: Any, kv_cache_manager: Any) -> None:
    """Start building the MSA ``fmha_sm100`` kernels of ``model`` at the
    start of warmup, before its first forward needs them, from the engine's
    KV cache manager, unless they started while the weights loaded. Once per
    process."""
    from .....jit_prefetch import FmhaSm100Builds

    if FmhaSm100Builds.get() is None:
        FmhaSm100Builds.start(MiniMaxM3FmhaProvider(model, kv_cache_manager))
