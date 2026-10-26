# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""DeepSeek-R1 MLA generation-attention microbenchmark (one layer, one GPU).

Times the per-layer decode attention of DS-R1 MLA (128 local heads, kv_lora_rank
512, rope 64) at generation-like batches for:

* ``fp8``          FP8 KV through the TRT-LLM attention backend with its
                   default FMHA-library dispatch (what a DS-R1 FP8-KV gen
                   engine runs); the chosen library is printed.
* ``fp8_thop``     FP8 KV forced through the fused THOP attention op.
* ``fp8_cutedsl``  FP8 KV through the CuTe-DSL MLA decode
                   (``flashinfer_mla_backend="cute-dsl"``).
* ``triton``       NVFP4 KV ("FP4 MLA") Triton decode.
* ``cutedsl``      NVFP4 KV CuTe-DSL (MUFU16) decode (when available).

KV lengths are either fixed (``--dist fixed:3000``) or drawn like the MLPerf
DS-R1 dataset in steady-state decode (``--dist dataset``): a request is picked
with probability proportional to its reference output length and observed at
a uniformly random point of its decode, so kv_len = ISL + U[1, OSL].

Two scopes are timed: ``decode`` (attention core: FP8 generation FMHA, or the
FP4 decode with pre-quantized Q) and ``step`` (adds the RoPE / Q-quant /
KV-append work that precedes the core in a generation step). Metrics:
``graph_us`` CUDA-graph replay time per call (CUDA events), ``kern_us`` sum of
CUDA kernel durations per call (torch profiler/CUPTI).

Example::

    python bench_dsr1_mla_decode.py --backend fp8 triton --batch 128 512 \
        --dist dataset fixed:1000 --isl-osl /path/dsr1_isl_osl.npz
"""

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

import tensorrt_llm

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
Q_LORA_RANK = 1536
V_HEAD_DIM = 128
HEAD_DIM_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
SM_SCALE = 1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)


# --------------------------------------------------------------------------- #
# KV-length distributions
# --------------------------------------------------------------------------- #
def load_isl_osl(path: str) -> tuple[np.ndarray, np.ndarray]:
    if path.endswith(".npz"):
        data = np.load(path)
        return data["isl"].astype(np.int64), data["osl"].astype(np.int64)
    import pandas as pd

    df = pd.read_pickle(path)
    isl = df["tok_input_len"].to_numpy().astype(np.int64)
    osl = df["tok_ref_output_len"].to_numpy().astype(np.int64)
    return isl, osl


def sample_kv_lens(
    dist: str, batch: int, seed: int, isl_osl, max_len: int, page_align: int = 1
) -> list[int]:
    rng = np.random.default_rng(seed)
    if dist.startswith("fixed:"):
        return [int(dist.split(":", 1)[1])] * batch
    if dist.startswith("uniform:"):
        lo, hi = (int(v) for v in dist.split(":", 1)[1].split("-"))
        return rng.integers(lo, hi + 1, size=batch).tolist()
    if dist in ("dataset", "dataset_u"):
        if isl_osl is None:
            raise ValueError("--dist dataset requires --isl-osl")
        isl, osl = isl_osl
        osl = np.minimum(osl, max_len - isl)
        # dataset: steady-state in-flight mix (a request is seen in proportion
        # to its decode length); dataset_u: every request equally likely.
        prob = osl / osl.sum() if dist == "dataset" else None
        idx = rng.choice(len(osl), size=batch, p=prob)
        progress = rng.integers(1, osl[idx] + 1)
        kv = np.minimum(isl[idx] + progress, max_len)
        return kv.astype(np.int64).tolist()
    raise ValueError(f"unknown distribution {dist!r}")


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def _capture_graph(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    return graph, stream


def _event_time_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _kernel_sum_us(fn, iters: int) -> tuple[float, list[tuple[str, int, float]]]:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    totals: dict[str, list] = {}
    for event in prof.events():
        if event.device_type != DeviceType.CUDA:
            continue
        rec = totals.setdefault(event.name, [0, 0.0])
        rec[0] += 1
        rec[1] += event.self_device_time_total
    total = sum(v[1] for v in totals.values()) / iters
    top = sorted(
        ((k, v[0] // iters, v[1] / iters) for k, v in totals.items()), key=lambda item: -item[2]
    )[:6]
    return total, top


# Profiler mode (--prof-range N): after warmup, run N eager calls inside an NVTX
# range named "prof_<scope>" and skip timing, for `ncu --nvtx --nvtx-include
# prof_decode/` (N=1: one launch per kernel) or `nsys -t cuda,nvtx` captures.
_PROF_RANGE_CALLS = 0


def time_fn(
    fn, *, warmup: int, iters: int, use_graph: bool, profile: bool, check=None, name: str = "decode"
) -> dict:
    result: dict = {}
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if _PROF_RANGE_CALLS:
        with torch.cuda.nvtx.range(f"prof_{name}"):
            for _ in range(_PROF_RANGE_CALLS):
                fn()
        torch.cuda.synchronize()
        return {"profiled_calls": _PROF_RANGE_CALLS}
    if check is not None:
        check()
    result["eager_us"] = _event_time_ms(fn, iters) * 1e3
    if profile:
        kern_us, top = _kernel_sum_us(fn, max(3, iters // 4))
        result["kern_us"] = kern_us
        result["top"] = top
    if use_graph:
        try:
            graph, stream = _capture_graph(fn)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            if check is not None:
                check()
            result["graph_us"] = _event_time_ms(graph.replay, iters) * 1e3
            del graph, stream
        except Exception as exc:  # graph capture is best effort; eager stays valid
            result["graph_err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            torch.cuda.synchronize()
    return result


# --------------------------------------------------------------------------- #
# FP8 KV (TRT-LLM attention backend)
# --------------------------------------------------------------------------- #
def run_fp8(seq_lens, heads, variant, args) -> dict:
    from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionInputType,
        MLAParams,
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.functional import PositionEmbeddingType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    device = torch.device("cuda")
    q_len = 1
    batch = len(seq_lens)
    page_size = args.fp8_page_size
    request_ids = list(range(batch))
    max_seq = max(seq_lens)
    max_tokens = (
        sum((s + page_size - 1) // page_size * page_size for s in seq_lens) + 2 * batch * page_size
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=HEAD_DIM_QK,
        tokens_per_block=page_size,
        max_seq_len=max_seq + page_size,
        max_batch_size=batch,
        mapping=mapping,
        dtype=_DataType.FP8,
    )
    try:
        if kv_cache_manager.add_dummy_requests(request_ids, list(seq_lens)) is None:
            raise RuntimeError("FP8 KV cache manager could not allocate the dummy requests")
        kv_cache_manager.get_buffers(0).zero_()
        past = [s - q_len for s in seq_lens]
        metadata = TrtllmAttention.Metadata(
            seq_lens=torch.full((batch,), q_len, dtype=torch.int32),
            request_ids=request_ids,
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=past,
            max_num_tokens=batch * q_len,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=past),
            mapping=mapping,
        )
        metadata.prepare()
        attention = TrtllmAttention(
            layer_idx=0,
            num_heads=heads,
            head_dim=HEAD_DIM_QK,
            num_kv_heads=1,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value),
            q_scaling=1.0,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams(
                    dim=QK_ROPE_HEAD_DIM,
                    max_positions=max_seq + 16,
                    original_max_positions=max_seq + 16,
                    duplicate_data=True,
                ),
                is_neox=False,
            ),
            mla_params=MLAParams(
                q_lora_rank=Q_LORA_RANK,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                predicted_tokens_per_seq=q_len,
            ),
            flashinfer_mla_backend="cute-dsl" if variant == "fp8_cutedsl" else "trtllm-gen",
        )
        if variant == "fp8_thop":
            attention.fmha_libs = [FallbackFmha(attention)]
        if not attention.fmha_libs:
            attention.create_fmha_libs()
        chosen = []
        for lib in attention.fmha_libs:
            original = lib.forward

            def wrapped(*a, _orig=original, _name=type(lib).__name__, **kw):
                if not chosen or chosen[-1] != _name:
                    chosen.append(_name)
                return _orig(*a, **kw)

            lib.forward = wrapped

        torch.manual_seed(8)
        num_tokens = batch * q_len
        fused_q = torch.randn(num_tokens, heads * HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(num_tokens, heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
        latent_cache = torch.randn(num_tokens, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
        cu_q_seqlens = torch.empty(batch + 1, dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.empty(batch + 1, dtype=torch.int32, device=device)
        counter = torch.empty(1, dtype=torch.uint32, device=device)
        bmm1 = torch.empty(2, dtype=torch.float32, device=device)
        bmm2 = torch.empty(1, dtype=torch.float32, device=device)
        quant_q = torch.empty(num_tokens, heads * HEAD_DIM_QK, dtype=torch.uint8, device=device)
        output = torch.empty(num_tokens, heads * KV_LORA_RANK, dtype=torch.bfloat16, device=device)

        def prepare():
            attention.mla_rope_generation(
                fused_q,
                q_pe,
                latent_cache,
                metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                counter,
                bmm1,
                bmm2,
                quant_q,
            )

        def decode():
            attention.forward(
                fused_q,
                None,
                None,
                metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1,
                mla_bmm2_scale=bmm2,
                quant_q_buffer=quant_q,
                output=output,
            )

        def step():
            prepare()
            decode()

        def check():
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("FP8 MLA output is not finite")

        prepare()
        decode()
        torch.cuda.synchronize()
        check()
        res = {"lib": ",".join(chosen)}
        res["decode"] = time_fn(
            decode,
            warmup=args.warmup,
            iters=args.iters,
            use_graph=not args.no_graph,
            profile=args.profile,
            check=check,
        )
        if not args.decode_only:
            res["step"] = time_fn(
                step,
                name="step",
                warmup=args.warmup,
                iters=args.iters,
                use_graph=not args.no_graph,
                profile=args.profile,
                check=check,
            )
        return res
    finally:
        kv_cache_manager.shutdown()


# --------------------------------------------------------------------------- #
# NVFP4 KV (FP4 MLA backend)
# --------------------------------------------------------------------------- #
def _build_fp4_metadata(kv_cache_manager, seq_lens, page_size, fp4):
    device = torch.device("cuda")
    num_seqs = len(seq_lens)
    request_ids = list(range(num_seqs))
    block_ids = kv_cache_manager.get_batch_cache_indices(request_ids, layer_idx=0)
    spec = kv_cache_manager.get_fp4_mla_page_table_spec(0)
    hp_block_ids = kv_cache_manager._get_batch_cache_indices_by_pool_id(
        request_ids, pool_id=spec.hp_pool_id, is_kv_aggregate=False
    )
    num_blocks = [(s + page_size - 1) // page_size for s in seq_lens]
    stride = max(num_blocks)
    rows, hp_rows = [], []
    for i, blocks in enumerate(block_ids):
        act = list(blocks[: num_blocks[i]])
        rows.extend(act + [0] * (stride - len(act)))
        hp_act = list(hp_block_ids[i][: num_blocks[i]])
        hp_rows.extend(hp_act + [0] * (stride - len(hp_act)))
    paged_kv_indices = torch.tensor(rows, dtype=torch.int32, device=device)
    paged_kv_indptr = torch.arange(num_seqs + 1, dtype=torch.int32, device=device) * stride
    hp_page_indices = torch.tensor(hp_rows, dtype=torch.int32, device=device)
    lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    batch_indices = torch.repeat_interleave(
        torch.arange(num_seqs, dtype=torch.int32, device=device), lens
    )
    starts = torch.cumsum(lens, 0) - lens
    positions = (
        torch.arange(int(lens.sum().item()), dtype=torch.int64, device=device)
        - torch.repeat_interleave(starts, lens)
    ).to(torch.int32)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_indices=paged_kv_indices,
        _paged_kv_indices=paged_kv_indices,
        _fp4_mla_hp_page_indices=hp_page_indices,
        paged_kv_indptr=paged_kv_indptr,
        _paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_decode=paged_kv_indptr.clone(),
        _fp4_mla_device_page_table=True,
        _fp4_mla_device_page_table_valid=True,
        fp4_mla_page_table_stride=stride,
        fp4_mla_context_repack_max_touched_pages=stride,
        page_size=page_size,
        num_context_blocks=num_seqs * stride,
        num_generation_blocks=0,
        num_contexts=num_seqs,
        num_seqs=num_seqs,
        num_blocks=None,
        high_precision_kv_pool=kv_cache_manager.get_fp4_mla_hp_pool(),
        fp4_mla_v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=kv_lens.clone(),
        prompt_lens_cpu_runtime=torch.tensor(seq_lens, dtype=torch.int32),
        fp4_mla_generation_kv_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
        fp4_mla_generation_append_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
        fp4_mla_generation_lengths_num_tokens=-1,
        fp4_mla_generation_lengths_num_seqs=-1,
        fp4_mla_generation_lengths_num_contexts=-1,
        _fp4_mla_generation_lengths_capture_recorded=False,
        _fp4_mla_q_global_scale=torch.tensor(
            [fp4.FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=device
        ),
        _fp4_mla_kv_global_scale=torch.tensor(
            [fp4.FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=device
        ),
        request_ids=request_ids,
        runtime_features=SimpleNamespace(has_speculative_draft_tokens=False),
        is_cuda_graph=False,
        is_warmup=False,
        num_ctx_tokens=int(sum(seq_lens)),
    )


def build_fp4_case(seq_lens, heads, backend, seed=8, q_len=1, keep_latent=False, skip_fill=False):
    """Build a 1-layer FP4 MLA cache filled by the real context writer.

    Returns (manager, metadata, q [Nq, H, 576] bf16). The metadata is switched
    to an all-generation batch whose kv_lens equal ``seq_lens``.
    """
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4
    from tensorrt_llm._torch.attention_backend.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    os.environ[fp4.FP4_MLA_ATTENTION_BACKEND_ENV] = backend
    if hasattr(fp4, "_q_quant_fusion_supported") and torch.cuda.get_device_capability() == (10, 7):
        # The stock k3_26.08 base gates the fused generation RoPE/Q-quant/KV
        # writer to SM103 only; its Triton kernel runs on SM107 as well.
        fp4._q_quant_fusion_supported = lambda: True
    torch.manual_seed(seed)
    device = torch.device("cuda")
    page_size = fp4.FP4_MLA_TOKENS_PER_BLOCK
    num_pages = sum((s + page_size - 1) // page_size for s in seq_lens)
    manager = None
    # The pool budget is shared with the per-request HP ring and is split by the
    # manager's sizing policy: grow it until every dummy request fits.
    for mult in (1.25, 1.5, 2.0, 3.0, 4.0):
        max_tokens = (int(num_pages * mult) + 2 * len(seq_lens) + 64) * page_size
        manager = Fp4MlaKVCacheManagerV2(
            KvCacheConfig(
                max_tokens=max_tokens, dtype="nvfp4", enable_block_reuse=False, host_cache_size=0
            ),
            _CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=HEAD_DIM_QK,
            tokens_per_block=page_size,
            max_seq_len=max(page_size, max(seq_lens)),
            max_batch_size=len(seq_lens),
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=_DataType.NVFP4,
            spec_config=None,
            max_num_tokens=max_tokens,
            pretrained_config=SimpleNamespace(kv_lora_rank=KV_LORA_RANK),
        )
        if manager.add_dummy_requests(list(range(len(seq_lens))), list(seq_lens)) is not None:
            break
        manager.shutdown()
        manager = None
    if manager is None:
        raise RuntimeError("FP4 MLA KV cache manager could not allocate the dummy requests")
    kv_cache, sf_cache = manager.get_fp4_mla_cache_buffers(0)
    kv_cache.zero_()
    sf_cache.zero_()
    metadata = _build_fp4_metadata(manager, seq_lens, page_size, fp4)
    get_v_base = getattr(manager, "get_mla_v_packed_pool_base", None)
    if callable(get_v_base) and get_v_base() is not None:
        get_v_base().zero_()
    if metadata.fp4_mla_v_scale_pool is not None:
        metadata.fp4_mla_v_scale_pool.zero_()
    if skip_fill:
        return manager, metadata, None
    # Fill in chunks to bound the BF16 latent staging buffer.
    latent = (
        torch.randn(int(sum(seq_lens)), HEAD_DIM_QK, dtype=torch.bfloat16, device=device) * 0.25
    ).clamp_(-1.0, 1.0)
    fp4.scatter_fp4_mla_kv_cache(
        metadata,
        latent,
        layer_idx=0,
        token_offset=0,
        phase="context",
        local_layer=0,
        v_head_dim=KV_LORA_RANK,
    )
    if keep_latent:
        metadata._bench_latent = latent
    del latent
    metadata.high_precision_kv_pool.zero_()
    torch.cuda.synchronize()
    # Switch to an all-generation decode batch.
    n = len(seq_lens)
    metadata.num_contexts = 0
    metadata.num_ctx_tokens = 0
    metadata.num_context_blocks = 0
    metadata.num_generation_blocks = n * metadata.fp4_mla_page_table_stride
    metadata.prompt_lens_cuda_runtime = torch.full((n,), q_len, dtype=torch.int32, device=device)
    metadata.prompt_lens_cpu_runtime = torch.full((n,), q_len, dtype=torch.int32)
    q = (
        (torch.randn(n * q_len, heads, HEAD_DIM_QK, dtype=torch.bfloat16, device=device) * 0.25)
        .clamp_(-1.0, 1.0)
        .contiguous()
    )
    return manager, metadata, q


def run_fp4(seq_lens, heads, backend, args) -> dict:
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    q_len = 1
    batch = len(seq_lens)
    manager, metadata, q = build_fp4_case(seq_lens, heads, backend, q_len=q_len)
    try:
        device = q.device
        # The synthetic batch never changes: pre-populate the generation lengths.
        metadata.fp4_mla_generation_kv_lens.copy_(metadata.kv_lens_cuda_runtime)
        metadata.fp4_mla_generation_append_lens.copy_(metadata.prompt_lens_cuda_runtime)
        metadata.fp4_mla_generation_lengths_num_tokens = batch * q_len
        metadata.fp4_mla_generation_lengths_num_seqs = batch
        metadata.fp4_mla_generation_lengths_num_contexts = 0
        metadata._fp4_mla_generation_lengths_capture_recorded = True
        q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
            q.view(-1, q.shape[-1]),
            metadata._fp4_mla_q_global_scale,
            fp4.FP4_MLA_Q_RESIDUAL_DIM,
            is_act=True,
        )
        q_sf = q_sf.view(torch.float8_e4m3fn).reshape(-1)
        output = torch.full(
            (q.shape[0], heads, KV_LORA_RANK), float("nan"), dtype=torch.bfloat16, device=device
        )

        def decode():
            pq = getattr(metadata, "fp4_mla_prequantized_q", None)
            if pq is not None:
                fq, fsf, cap = (
                    pq,
                    metadata.fp4_mla_prequantized_q_sf,
                    metadata.fp4_mla_q_batch_capacity,
                )
            else:
                fq, fsf, cap = q_fp4, q_sf, q.shape[0]
            fp4.run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q=q,
                output=output,
                sm_scale=SM_SCALE,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                prequantized_q=fq,
                prequantized_q_sf=fsf,
                q_batch_capacity=cap,
            )

        def check():
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("FP4 MLA output is not finite")

        decode()
        torch.cuda.synchronize()
        check()
        res = {"lib": f"fp4_mla/{backend}"}
        metadata.is_cuda_graph = not args.no_graph
        # The model sorts the longest-KV-first order once per forward step (first
        # layer); a per-layer timing must not replay that sort with every call.
        metadata._fp4_mla_lpt_capture_recorded = True
        res["decode"] = time_fn(
            decode,
            warmup=args.warmup,
            iters=args.iters,
            use_graph=not args.no_graph,
            profile=args.profile,
            check=check,
        )
        if not args.decode_only:
            context_lens = torch.tensor(
                [s - q_len for s in seq_lens], dtype=torch.int32, device=device
            )
            metadata.batch_indices = torch.arange(
                batch, dtype=torch.int32, device=device
            ).repeat_interleave(q_len)
            metadata.positions = (
                context_lens[:, None]
                + torch.arange(q_len, dtype=torch.int32, device=device)[None, :]
            ).reshape(-1)
            gen_latent = (
                torch.randn(batch * q_len, HEAD_DIM_QK, dtype=torch.bfloat16, device=device) * 0.25
            ).clamp_(-1.0, 1.0)
            cos_sin = torch.zeros(
                max(seq_lens) + q_len, QK_ROPE_HEAD_DIM, 2, dtype=torch.float32, device=device
            )
            cos_sin[..., 0] = 1.0

            def step():
                fp4.scatter_fp4_mla_kv_cache(
                    metadata,
                    gen_latent,
                    layer_idx=0,
                    token_offset=0,
                    phase="generation",
                    local_layer=0,
                    v_head_dim=KV_LORA_RANK,
                    rotary_cos_sin=cos_sin,
                    q_pe=q[..., KV_LORA_RANK:],
                    q_rope_out=q[..., KV_LORA_RANK:],
                    q_quant_input=q,
                )
                decode()

            try:
                res["step"] = time_fn(
                    step,
                    name="step",
                    warmup=args.warmup,
                    iters=args.iters,
                    use_graph=not args.no_graph,
                    profile=args.profile,
                    check=check,
                )
            except Exception as exc:
                res["step_err"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return res
    finally:
        manager.shutdown()


def _fmt(res: dict, scope: str) -> str:
    r = res.get(scope)
    if r is None:
        return f"{scope}=n/a"
    parts = [f"{scope}:"]
    for key in ("graph_us", "kern_us", "eager_us"):
        if key in r:
            parts.append(f"{key}={r[key]:.1f}")
    if "graph_err" in r:
        parts.append(f"graph_err=[{r['graph_err'][:80]}]")
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        nargs="+",
        default=["fp8", "triton"],
        choices=["fp8", "fp8_thop", "fp8_cutedsl", "triton", "cutedsl"],
    )
    p.add_argument("--batch", type=int, nargs="+", default=[128, 512, 1024, 1536])
    p.add_argument(
        "--dist", nargs="+", default=["dataset", "fixed:1000", "fixed:3000", "fixed:6000"]
    )
    p.add_argument("--isl-osl", default=None, help=".npz (isl, osl) or the MLPerf DS-R1 .pkl")
    p.add_argument("--max-len", type=int, default=23200)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--fp8-page-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument(
        "--profile",
        action="store_true",
        help="also sum kernel durations with the torch profiler (needs CUPTI)",
    )
    p.add_argument("--decode-only", action="store_true")
    p.add_argument("--json-out", default=None)
    p.add_argument(
        "--prof-range",
        type=int,
        default=0,
        help="profiler mode: N eager calls in an NVTX range prof_<scope>, no timing",
    )
    p.add_argument("--top", action="store_true", help="print top kernels")
    args = p.parse_args()

    global _PROF_RANGE_CALLS
    _PROF_RANGE_CALLS = args.prof_range
    if args.prof_range:
        args.no_graph = True
    isl_osl = load_isl_osl(args.isl_osl) if args.isl_osl else None
    if isl_osl is not None:
        isl, osl = isl_osl
        print(
            f"dataset: n={len(isl)} isl mean={isl.mean():.0f} osl mean={osl.mean():.0f} "
            f"osl max={osl.max()}",
            flush=True,
        )
    dev = torch.cuda.get_device_properties(0)
    print(
        f"device: {dev.name} sm{dev.major}{dev.minor} sms={dev.multi_processor_count} "
        f"torch={torch.__version__} trtllm={tensorrt_llm.__version__}",
        flush=True,
    )
    records = []
    for dist in args.dist:
        for batch in args.batch:
            seq_lens = sample_kv_lens(dist, batch, args.seed, isl_osl, args.max_len)
            mean_kv = float(np.mean(seq_lens))
            for be in args.backend:
                t0 = time.time()
                try:
                    if be.startswith("fp8"):
                        res = run_fp8(seq_lens, args.heads, be, args)
                    else:
                        res = run_fp4(seq_lens, args.heads, be, args)
                    err = None
                except Exception as exc:  # report and continue with the next case
                    import traceback

                    traceback.print_exc()
                    res, err = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
                    torch.cuda.synchronize()
                rec = {
                    "dist": dist,
                    "batch": batch,
                    "mean_kv": mean_kv,
                    "max_kv": int(max(seq_lens)),
                    "backend": be,
                    "err": err,
                    **res,
                }
                records.append(rec)
                line = (
                    f"RESULT dist={dist:>11s} B={batch:>4d} mean_kv={mean_kv:7.0f} "
                    f"max_kv={max(seq_lens):6d} backend={be:>11s} lib={res.get('lib', '-')} "
                    f"{_fmt(res, 'decode')} | {_fmt(res, 'step')}"
                )
                if err:
                    line += f" ERROR={err}"
                if "step_err" in res:
                    line += f" STEP_ERR={res['step_err'][:150]}"
                print(line + f" ({time.time() - t0:.0f}s)", flush=True)
                if args.top:
                    for scope in ("decode", "step"):
                        for name, cnt, us in res.get(scope, {}).get("top", []):
                            print(f"   top[{scope}] {us:9.1f} us x{cnt} {name[:140]}", flush=True)
                torch.cuda.empty_cache()
                if args.json_out:
                    with open(args.json_out, "w") as f:
                        json.dump(records, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
