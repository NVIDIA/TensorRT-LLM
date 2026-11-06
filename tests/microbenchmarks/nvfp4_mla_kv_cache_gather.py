# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark GLM-5.2 and DeepSeek-V4-Pro NVFP4 MLA gather followed by DSA.

The model shapes match their checkpoint paths: GLM-5.2 uses the pointer-table
gather with Top-K 2048 and a 576-wide KV row; DeepSeek-V4-Pro uses the direct
gather with Top-K 1024 and a 512-wide KV row. Each row reports mean latency for
the gather, DSA consuming the gathered FP8 scratch pool, and both operations
back-to-back. DSV4 attention includes its 128-token sliding window before the
1024 gathered compressed tokens.

Run on one Blackwell GPU; no model weights are needed::

    python tests/microbenchmarks/nvfp4_mla_kv_cache_gather.py

By default the benchmark sweeps batch sizes 1, 2, 4, 8, 16, 32, 64, and 128
with a 720 MiB GLM source pool; DSV4 defaults to the minimum pool needed for
the largest batch. Indices are unique across the largest batch, execution uses
CUDA Graph replay, and L2 is evicted before every timed sample
without charging eviction to the sample. Use ``--eager`` to disable graph
replay and ``--csv results.csv`` to also save machine-readable results.
"""

import argparse
import csv
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.quantization import QuantMode


@dataclass(frozen=True)
class ModelSpec:
    name: str
    operator: str
    topk: int
    head_dim: int
    residual_dim: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_append: bool
    direct: bool
    swa_window_size: int = 0

    @property
    def data_bytes_per_token(self) -> int:
        return (self.head_dim + self.residual_dim) // 2

    @property
    def scales_per_token(self) -> int:
        return (self.head_dim + self.residual_dim) // 16

    @property
    def source_bytes_per_token(self) -> int:
        return self.data_bytes_per_token + self.scales_per_token

    @property
    def bytes_per_pair(self) -> int:
        # Packed values + scales + FP8 output + index read/write.
        return self.source_bytes_per_token + self.head_dim + 2 * torch.int32.itemsize


MODEL_SPECS = {
    "glm52": ModelSpec(
        name="GLM-5.2",
        operator="trtllm::nvfp4_mla_kv_cache_gather",
        topk=2048,
        head_dim=576,
        residual_dim=64,
        num_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=True,
        direct=False,
    ),
    "dsv4-pro": ModelSpec(
        name="DeepSeek-V4-Pro",
        operator="trtllm::nvfp4_mla_kv_cache_gather_direct",
        topk=1024,
        head_dim=512,
        residual_dim=64,
        num_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=448,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=False,
        direct=True,
        swa_window_size=128,
    ),
}
DEFAULT_GLM_POOL_BYTES = 720 * 1024 * 1024
TOKENS_PER_BLOCK = 64


class GatherBenchmark:
    def __init__(
        self,
        spec: ModelSpec,
        max_batch_size: int,
        num_pool_tokens: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.max_batch_size = max_batch_size
        self.num_pool_tokens = num_pool_tokens
        num_indices = max_batch_size * spec.topk
        if num_pool_tokens < num_indices:
            raise ValueError(
                f"KV pool needs at least {num_indices} rows for unique indices, "
                f"got {num_pool_tokens}"
            )

        self.data_pool = torch.full(
            (num_pool_tokens, spec.data_bytes_per_token),
            0x22,
            dtype=torch.uint8,
            device=device,
        )
        self.scale_pool = torch.ones(
            (num_pool_tokens, spec.scales_per_token),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self.indices = (
            torch.randperm(num_pool_tokens, device=device, generator=generator)[:num_indices]
            .to(torch.int32)
            .view(max_batch_size, spec.topk)
        )
        self.indices_template = self.indices.clone() if spec.direct else None
        self.compact_indices = None if spec.direct else torch.empty_like(self.indices)
        self.output = torch.empty(
            (max_batch_size, spec.topk, spec.head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.attention_cache = torch.empty_like(self.output)
        self.num_sparse_topk = spec.swa_window_size + spec.topk
        self.sparse_attn_indices = self.compact_indices
        self.sparse_attn_kv_lens = None
        if spec.swa_window_size:
            self.attention_cache.fill_(1.0)
            # Dynamic DSV4 MLA reads tile 0 from the SWA pool and subsequent
            # tiles from the gathered FP8 pool. Its valid lengths select the
            # dynamic kernel family; omitting them selects GLM's static family.
            swa_indices = (
                torch.arange(max_batch_size, dtype=torch.int32, device=device)[:, None] * spec.topk
                + torch.arange(spec.swa_window_size, dtype=torch.int32, device=device)[None, :]
            )
            # All generated source indices are valid, so direct gather always
            # compacts them to this fixed row map. Precompute it outside timing.
            compact_indices = torch.arange(num_indices, dtype=torch.int32, device=device).view(
                max_batch_size, spec.topk
            )
            self.sparse_attn_indices = torch.cat((swa_indices, compact_indices), dim=1)
            self.sparse_attn_kv_lens = torch.full(
                (max_batch_size,), self.num_sparse_topk, dtype=torch.int32, device=device
            )
        self.global_dequant_scale = torch.ones(1, dtype=torch.float32, device=device)
        self.host_pool_pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
        self.host_pool_pointers[0, 0, 0] = self.data_pool.data_ptr()
        self.host_pool_pointers[0, 0, 1] = self.scale_pool.data_ptr()
        self.host_pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
        self.host_attention_pool_pointers = torch.tensor(
            [[self.attention_cache.data_ptr(), self.attention_cache.data_ptr()]],
            dtype=torch.int64,
        )

        self.query = torch.zeros(
            (max_batch_size, spec.num_heads * spec.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.attention_output = torch.empty(
            (max_batch_size, spec.num_heads * spec.v_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.quant_query = torch.zeros(
            (max_batch_size, spec.num_heads * spec.head_dim),
            dtype=torch.uint8,
            device=device,
        )
        self.latent_cache = torch.zeros(
            (max_batch_size, spec.head_dim), dtype=torch.bfloat16, device=device
        )
        self.q_pe = torch.zeros(
            (max_batch_size, spec.num_heads, spec.qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        bmm1_scale = 1.0 / math.sqrt(spec.qk_nope_head_dim + spec.qk_rope_head_dim)
        self.mla_bmm1_scale = torch.tensor(
            [bmm1_scale, bmm1_scale * math.log2(math.e)],
            dtype=torch.float32,
            device=device,
        )
        self.mla_bmm2_scale = torch.ones(1, dtype=torch.float32, device=device)
        self.kv_lens_cuda = torch.full(
            (max_batch_size,), spec.topk, dtype=torch.int32, device=device
        )
        self.kv_lens_host = torch.full((max_batch_size,), spec.topk - 1, dtype=torch.int32)
        self.prompt_lens_cuda = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        self.prompt_lens_host = torch.zeros(max_batch_size, dtype=torch.int32)
        self.host_request_types = torch.ones(max_batch_size, dtype=torch.int32)
        num_blocks = math.ceil(spec.topk / TOKENS_PER_BLOCK)
        self.kv_cache_block_offsets = torch.arange(
            max_batch_size * num_blocks, dtype=torch.int32, device=device
        ).view(1, max_batch_size, num_blocks)
        self.cu_q_seqlens = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.cu_q_seqlens.mul_(spec.num_heads)
        self.cu_kv_seqlens = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.cu_kv_seqlens.mul_(spec.topk)
        self.fmha_scheduler_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.attention_workspace = torch.empty(0, dtype=torch.uint8, device=device)

        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        self.l2_clear = torch.empty(2 * l2_bytes, dtype=torch.uint8, device=device)

    def call_gather_for_batch(self, batch_size: int) -> Callable[[], None]:
        spec = self.spec
        indices = self.indices[:batch_size]
        output = self.output[:batch_size]
        if spec.direct:

            def run() -> None:
                torch.ops.trtllm.nvfp4_mla_kv_cache_gather_direct(
                    self.data_pool,
                    self.scale_pool,
                    indices,
                    output,
                    self.global_dequant_scale,
                    spec.residual_dim,
                    self.num_pool_tokens,
                )

            return run

        assert self.compact_indices is not None
        compact_indices = self.compact_indices[:batch_size]

        def run() -> None:
            torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
                self.host_pool_pointers,
                self.host_pool_mapping,
                indices,
                output,
                compact_indices,
                self.global_dequant_scale,
                0,
                spec.residual_dim,
                self.num_pool_tokens,
            )

        return run

    def reset_indices_for_batch(self, batch_size: int) -> Callable[[], None] | None:
        if self.indices_template is None:
            return None

        def reset() -> None:
            self.indices[:batch_size].copy_(self.indices_template[:batch_size])

        return reset

    def call_dsa_for_batch(self, batch_size: int) -> Callable[[], None]:
        spec = self.spec
        host_total_kv_lens = torch.tensor([0, batch_size * spec.topk], dtype=torch.int32)
        sparse_indices = self.sparse_attn_indices
        assert sparse_indices is not None
        sparse_kv_lens = (
            self.sparse_attn_kv_lens[:batch_size] if self.sparse_attn_kv_lens is not None else None
        )

        def run() -> None:
            self.fmha_scheduler_counter.zero_()
            FallbackFmha.attention(
                q=self.query[:batch_size],
                k=None,
                v=None,
                output=self.attention_output[:batch_size],
                output_sf=None,
                workspace=self.attention_workspace,
                sequence_length=self.kv_lens_cuda[:batch_size],
                host_past_key_value_lengths=self.kv_lens_host[:batch_size],
                host_total_kv_lens=host_total_kv_lens,
                context_lengths=self.prompt_lens_cuda[:batch_size],
                host_context_lengths=self.prompt_lens_host[:batch_size],
                host_request_types=self.host_request_types[:batch_size],
                max_context_q_len_override=None,
                kv_cache_block_offsets=self.kv_cache_block_offsets[:, :batch_size],
                host_kv_cache_pool_pointers=self.host_attention_pool_pointers,
                host_kv_cache_pool_mapping=self.host_pool_mapping,
                cache_indirection=None,
                kv_scale_orig_quant=self.global_dequant_scale,
                kv_scale_quant_orig=self.global_dequant_scale,
                out_scale=None,
                rotary_inv_freq=None,
                rotary_cos_sin=None,
                latent_cache=self.latent_cache[:batch_size],
                q_pe=self.q_pe[:batch_size],
                block_ids_per_seq=None,
                attention_sinks=None,
                is_fused_qkv=False,
                update_kv_cache=True,
                predicted_tokens_per_seq=1,
                local_layer_idx=0,
                num_heads=spec.num_heads,
                num_kv_heads=1,
                head_size=spec.head_dim,
                tokens_per_block=TOKENS_PER_BLOCK,
                max_num_requests=self.max_batch_size,
                max_context_length=1,
                max_seq_len=spec.topk,
                attention_window_size=spec.topk,
                beam_width=1,
                mask_type=int(AttentionMaskType.causal),
                quant_mode=int(QuantMode.FP8_KV_CACHE),
                q_scaling=1.0,
                position_embedding_type=int(PositionEmbeddingType.learned_absolute),
                rope_dim=0,
                rope_base=10000.0,
                rope_scale_type=0,
                rope_scale=1.0,
                rope_short_m_scale=1.0,
                rope_long_m_scale=1.0,
                rope_max_positions=spec.topk,
                rope_original_max_positions=spec.topk,
                use_paged_context_fmha=True,
                attention_input_type=int(AttentionInputType.generation_only),
                is_mla_enable=True,
                chunked_prefill_buffer_batch_size=None,
                q_lora_rank=spec.q_lora_rank,
                kv_lora_rank=spec.kv_lora_rank,
                qk_nope_head_dim=spec.qk_nope_head_dim,
                qk_rope_head_dim=spec.qk_rope_head_dim,
                v_head_dim=spec.v_head_dim,
                rope_append=spec.rope_append,
                mrope_rotary_cos_sin=None,
                mrope_position_deltas=None,
                helix_position_offsets=None,
                helix_is_inactive_rank=None,
                attention_chunk_size=None,
                softmax_stats_tensor=None,
                is_spec_decoding_enabled=False,
                use_spec_decoding=False,
                is_spec_dec_tree=False,
                spec_decoding_generation_lengths=None,
                spec_decoding_position_offsets=None,
                spec_decoding_packed_mask=None,
                spec_decoding_bl_tree_mask_offset=None,
                spec_decoding_bl_tree_mask=None,
                spec_bl_tree_first_sparse_mask_offset_kv=None,
                sparse_kv_indices=None,
                sparse_kv_offsets=None,
                sparse_attn_indices=sparse_indices[:batch_size],
                sparse_attn_offsets=None,
                sparse_attn_indices_block_size=1,
                num_sparse_topk=self.num_sparse_topk,
                sparse_attn_kv_lens=sparse_kv_lens,
                cu_q_seqlens=self.cu_q_seqlens[: batch_size + 1],
                cu_kv_seqlens=self.cu_kv_seqlens[: batch_size + 1],
                fmha_scheduler_counter=self.fmha_scheduler_counter,
                mla_bmm1_scale=self.mla_bmm1_scale,
                mla_bmm2_scale=self.mla_bmm2_scale,
                quant_q_buffer=self.quant_query[:batch_size],
                num_contexts=0,
                num_ctx_tokens=0,
                aux_kv_cache_pool_ptr=self.output.data_ptr(),
            )

        return run

    def call_gather_and_dsa_for_batch(self, batch_size: int) -> Callable[[], None]:
        gather = self.call_gather_for_batch(batch_size)
        dsa = self.call_dsa_for_batch(batch_size)

        def run() -> None:
            gather()
            dsa()

        return run


def _capture_cuda_graph(
    fn: Callable[[], None],
    prepare: Callable[[], None] | None = None,
) -> Callable[[], None]:
    if prepare is not None:
        prepare()
    fn()
    torch.cuda.synchronize()
    if prepare is not None:
        prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay() -> None:
        graph.replay()

    return replay


def _time_cuda_us(
    fn: Callable[[], None],
    l2_clear: torch.Tensor,
    warmup: int,
    iters: int,
    prepare: Callable[[], None] | None = None,
) -> list[float]:
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        l2_clear.zero_()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        if prepare is not None:
            prepare()
        # Evict the source and index tensors from L2. The start event is
        # recorded afterwards, so preparation and eviction are excluded.
        l2_clear.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _run_model(
    spec: ModelSpec,
    batch_sizes: list[int],
    num_pool_tokens: int,
    seed: int,
    warmup: int,
    iters: int,
    device: torch.device,
    use_cuda_graph: bool,
) -> list[dict[str, int | float | str]]:
    max_batch_size = max(batch_sizes)
    benchmark = GatherBenchmark(spec, max_batch_size, num_pool_tokens, seed, device)
    execution = "CUDA Graph" if use_cuda_graph else "Eager"
    pool_mib = num_pool_tokens * spec.source_bytes_per_token / 1024**2
    print(
        f"\n{spec.name}: op={spec.operator} topk={spec.topk} "
        f"head_dim={spec.head_dim} swa_tokens={spec.swa_window_size} "
        f"source_pool={pool_mib:.1f} MiB mode={execution}"
    )
    print(" batch   topk  gather_mean_us  dsa_mean_us  total_mean_us  effective_GB/s")

    results = []
    for batch_size in batch_sizes:
        gather = benchmark.call_gather_for_batch(batch_size)
        dsa = benchmark.call_dsa_for_batch(batch_size)
        gather_and_dsa = benchmark.call_gather_and_dsa_for_batch(batch_size)
        reset_indices = benchmark.reset_indices_for_batch(batch_size)

        # Populate the FP8 scratch and compact indices before timing DSA alone.
        if reset_indices is not None:
            reset_indices()
        gather()
        torch.cuda.synchronize()
        if spec.direct:
            torch.testing.assert_close(
                benchmark.indices[:batch_size],
                benchmark.sparse_attn_indices[:batch_size, spec.swa_window_size :],
                rtol=0,
                atol=0,
            )
        if use_cuda_graph:
            gather = _capture_cuda_graph(gather, reset_indices)
            dsa = _capture_cuda_graph(dsa)
            gather_and_dsa = _capture_cuda_graph(gather_and_dsa, reset_indices)

        gather_samples = _time_cuda_us(gather, benchmark.l2_clear, warmup, iters, reset_indices)
        dsa_samples = _time_cuda_us(dsa, benchmark.l2_clear, warmup, iters)
        total_samples = _time_cuda_us(
            gather_and_dsa, benchmark.l2_clear, warmup, iters, reset_indices
        )
        gather_mean_us = statistics.fmean(gather_samples)
        dsa_mean_us = statistics.fmean(dsa_samples)
        total_mean_us = statistics.fmean(total_samples)
        num_pairs = batch_size * spec.topk
        effective_gb_s = num_pairs * spec.bytes_per_pair / gather_mean_us / 1000.0
        print(
            f"{batch_size:6d} {spec.topk:6d} {gather_mean_us:14.2f} "
            f"{dsa_mean_us:12.2f} {total_mean_us:14.2f} {effective_gb_s:15.2f}"
        )
        results.append(
            {
                "model": spec.name,
                "operator": spec.operator,
                "execution": execution,
                "batch_size": batch_size,
                "topk": spec.topk,
                "swa_tokens": spec.swa_window_size,
                "dsa_topk": benchmark.num_sparse_topk,
                "head_dim": spec.head_dim,
                "residual_dim": spec.residual_dim,
                "num_pairs": num_pairs,
                "gather_mean_us": gather_mean_us,
                "dsa_mean_us": dsa_mean_us,
                "total_mean_us": total_mean_us,
                "effective_gb_s": effective_gb_s,
            }
        )
    return results


def _write_csv(path: Path, results: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS,
        default=list(MODEL_SPECS),
        help="Models to benchmark (default: glm52 dsv4-pro).",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to sweep (default: 1 2 4 8 16 32 64 128).",
    )
    parser.add_argument(
        "--pool-tokens",
        type=int,
        default=None,
        help="KV pool rows; GLM defaults to 720 MiB, DSV4 to max(batch) * Top-K.",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Use eager execution instead of CUDA Graph replay.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path.")
    args = parser.parse_args()

    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes values must be positive")
    if args.pool_tokens is not None and args.pool_tokens <= 0:
        parser.error("--pool-tokens must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 10:
        raise RuntimeError(
            f"NVFP4 MLA gather requires a Blackwell GPU, got compute capability {capability}"
        )

    batch_sizes = args.batch_sizes
    max_batch_size = max(batch_sizes)
    all_results = []
    for model in args.models:
        spec = MODEL_SPECS[model]
        if args.pool_tokens is not None:
            num_pool_tokens = args.pool_tokens
        elif model == "glm52":
            num_pool_tokens = DEFAULT_GLM_POOL_BYTES // spec.source_bytes_per_token
        else:
            num_pool_tokens = max_batch_size * spec.topk
        all_results.extend(
            _run_model(
                spec,
                batch_sizes,
                num_pool_tokens,
                args.seed,
                args.warmup,
                args.iters,
                device,
                use_cuda_graph=not args.eager,
            )
        )
        torch.cuda.empty_cache()

    if args.csv is not None:
        _write_csv(args.csv, all_results)
        print(f"\nSaved {len(all_results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
