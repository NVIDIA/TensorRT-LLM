# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B2: batch-major vs token-major presentation of DeepSeek-V4 sparse-MLA generation.

This is the measurement the DSpark ragged-verification design hinges on
(``docs/dspark_confidence_schedule_goal.md`` §6.4 B2 / §7.1 U1).

The question
------------
Ragged verification means each request sends its own number of query tokens to
the target in one decode step. Every generation kernel that recovers ``(request,
offset)`` from a single ``next_n`` breaks under that. The deepest one is
trtllm-gen sparse-MLA generation, which cannot be read: its kernel selection
lives in a precompiled cubin, and the host side asserts a uniform layout::

    TLLM_CHECK_WITH_INFO(num_tokens % num_seqs == 0,
        "seq_len should be same for all generation requests", ...)
    int32_t const input_seq_length = num_tokens / num_seqs;
                                    -- cpp/tensorrt_llm/thop/attentionOp.cpp
    mMaxSeqLenQ = acc_q_len / batch_beam;
                                    -- cpp/tensorrt_llm/common/attentionOp.cpp

Both candidate ragged designs (D1 token-flat, D2 token-major) dodge that assert
the same way: present the generation half as ``num_gen_tokens`` sequences of
length one, so ``seq_len == 1`` and the check passes trivially. They are
therefore the same bet, and this script is how that bet gets priced.

What is compared
----------------
Identical total query tokens and an identical per-token sparse index budget,
laid out two ways:

* **batch-major** -- ``B`` requests, ``s_q = tier + 1``, each cached at ``L``.
  This is what the engine does today.
* **token-major** -- ``B * (tier + 1)`` requests, ``s_q = 1``, request ``(i, j)``
  cached at ``L + j``. Each query token is its own sequence, so it carries its
  own KV length and block table and no kernel has to divide anything.

Row ``j`` of request ``i`` attends over ``L + j + 1`` tokens in both layouts, so
the arithmetic is the same; only the presentation differs.

How to read the result
----------------------
``docs/dspark_p0_task_prompt.md`` §3: if token-major is materially slower or
incorrect, stop and fall back to the uniform tier ladder (D3 Stage A), which
needs no kernel change at all. The prior is that DeepSeek-V4-Pro under DEP8
keeps all 128 Q heads per rank, so BMM1's M dimension falls from
``128 * (tier+1)`` to ``128`` -- and because the sparse index set differs per
row there was never any KV tile sharing across the window to lose. That is an
inference, which is exactly why it needs measuring.

Scope: this times the whole attention layer, so it includes the indexer
(paged-MQA-logits + top-k), which itself changes presentation -- the batch-major
layout takes the expanded one-row-per-token path, the token-major layout takes
the strided ``next_n == 1`` path. That bundling is deliberate: it is what a
ragged step would actually pay, and it subsumes B3. The pure-attention delta
cannot be separated without a kernel-level profiler.

Usage (single GPU, no model weights)::

    python tests/microbenchmarks/dsv4_sparse_mla_presentation.py
    python tests/microbenchmarks/dsv4_sparse_mla_presentation.py \
        --batch-sizes 8 32 128 --tiers 1 3 5 --context-len 4096 --check
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

TESTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Reuse the sparse-MLA test harness verbatim rather than restating it: it is the
# only place that builds a real DeepseekV4TrtllmAttention plus cache manager,
# and a second copy would drift.
sys.path.insert(0, os.path.join(TESTS_ROOT, "unittest", "_torch", "attention",
                                "sparse", "deepseek_v4"))

from test_deepseek_v4_sparse_mla import (  # noqa: E402
    Scenario, _build_compressed_topk_indices, _create_cache_manager,
    _create_pos_embd_params, _create_rope_cos_sin, _prefill_compress_buffer)

from tensorrt_llm._torch.attention_backend.interface import (  # noqa: E402
    AttentionInputType, MLAParams)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import \
    DeepseekV4TrtllmAttention  # noqa: E402
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import \
    DeepseekV4TrtllmAttentionMetadata  # noqa: E402
from tensorrt_llm._torch.metadata import KVCacheParams  # noqa: E402
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest  # noqa: E402
from tensorrt_llm.bindings import SamplingConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

# Layer 1 of the shared Scenario is the compress_ratio=4 layer, i.e. the one
# that actually runs sparse selection. Layers 0 (dense) and 2 (ratio 128) do not
# exercise the path this question is about.
SPARSE_LAYER_IDX = 1


def _yarn_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class PresentationHarness:
    """One cache manager + attention layer, driven at a chosen presentation."""

    def __init__(self, scenario: Scenario, cached_lens: List[int],
                 seq_len_q: int, device: torch.device):
        """``cached_lens[i]`` is request i's KV length before this step; every
        request contributes ``seq_len_q`` query tokens."""
        self.scenario = scenario
        self.device = device
        self.cached_lens = list(cached_lens)
        self.seq_len_q = seq_len_q
        self.num_requests = len(cached_lens)
        self.total_q_tokens = self.num_requests * seq_len_q

        self.qk_rope_head_dim = scenario.qk_rope_head_dim
        # rope_append=False: the last 64 of the 512 lanes carry the rope part,
        # and the head count halves. Mirrors the test harness.
        self.kv_lora_rank = scenario.kv_lora_rank - self.qk_rope_head_dim
        self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.num_heads = 64

        max_seq_len = max(cached_lens) + seq_len_q + 1
        # The cache manager sizes its pool from the *context* lengths it is
        # handed, so pass the per-request KV lengths as if they were prompts.
        self.cache_manager, self.sparse_config = _create_cache_manager(
            scenario, self.cached_lens, max_seq_len)
        self.request_ids = list(range(self.num_requests))
        self.requests = [
            LlmRequest(
                request_id=i,
                max_new_tokens=seq_len_q + 1,
                input_tokens=list(range(cached_len)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            ) for i, cached_len in enumerate(self.cached_lens)
        ]
        for request in self.requests:
            self.cache_manager.prepare_context(request)
            self.cache_manager.resize_context(request,
                                              request.context_chunk_size)

        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        self.rope_cos_sin = _create_rope_cos_sin(scenario, device)
        pos_embd_params = _create_pos_embd_params(scenario)
        mscale = _yarn_mscale(pos_embd_params.rope.scale,
                              pos_embd_params.rope.mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.layer = DeepseekV4TrtllmAttention(
            layer_idx=SPARSE_LAYER_IDX,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=1,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=MLAParams(
                q_lora_rank=scenario.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                qk_nope_head_dim=scenario.qk_nope_head_dim,
                v_head_dim=scenario.v_head_dim,
                rope_append=scenario.rope_append,
                predicted_tokens_per_seq=1,
                hidden_size=scenario.hidden_size,
            ),
            sparse_attention_config=self.sparse_config,
            skip_create_weights_in_init=True,
        )
        self.layer.update_quant_config(None)
        self.layer.attn_sink = torch.nn.Parameter(torch.randn(
            self.num_heads, dtype=torch.float32, device=device).mul_(0.5),
                                                  requires_grad=False)

        _prefill_compress_buffer(self.cache_manager, SPARSE_LAYER_IDX,
                                 self.cached_lens, self.request_ids,
                                 self.head_dim, device)

        self._build_inputs()
        self._build_metadata()

    def _build_inputs(self) -> None:
        dtype = self.scenario.dtype
        device = self.device
        n = self.total_q_tokens
        self.compressed_kv = torch.empty([n, self.kv_lora_rank],
                                         dtype=dtype,
                                         device=device).uniform_(-1, 1)
        self.k_pe = torch.empty([n, self.qk_rope_head_dim],
                                dtype=dtype,
                                device=device).uniform_(-1, 1)
        q = torch.empty([n, self.num_heads, self.kv_lora_rank],
                        dtype=dtype,
                        device=device).uniform_(-1, 1)
        self.q_pe = torch.empty([n, self.num_heads, self.qk_rope_head_dim],
                                dtype=dtype,
                                device=device).uniform_(-1, 1)
        self.fused_q = torch.cat([q, self.q_pe],
                                 dim=-1).view(-1,
                                              self.num_heads * self.head_dim)
        self.latent_cache = torch.cat([self.compressed_kv, self.k_pe], dim=-1)

    def _build_metadata(self) -> None:
        device = self.device
        self.metadata = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor([self.seq_len_q] * self.num_requests,
                                  dtype=torch.int),
            request_ids=self.request_ids,
            max_num_requests=self.num_requests,
            num_contexts=0,
            prompt_lens=self.cached_lens,
            max_num_tokens=self.total_q_tokens,
            kv_cache_manager=self.cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=self.cached_lens,
            ),
            mapping=self.mapping,
            enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
            sparse_attention_config=self.sparse_config,
        )
        self.metadata.prepare()

        num_seqs = self.metadata.kv_lens_cuda_runtime.size(0)
        self.cu_q_seqlens = torch.empty(num_seqs + 1,
                                        dtype=torch.int32,
                                        device=device)
        self.cu_kv_seqlens = torch.empty(num_seqs + 1,
                                         dtype=torch.int32,
                                         device=device)
        self.fmha_scheduler_counter = torch.empty(1,
                                                  dtype=torch.uint32,
                                                  device=device)
        # Per-query-token KV extent, i.e. exactly what a correct ragged layout
        # must produce: row j of a request cached at L sees L + j + 1 tokens.
        per_row_kv_lens = [
            cached_len + j + 1 for cached_len in self.cached_lens
            for j in range(self.seq_len_q)
        ]
        self.topk_indices = _build_compressed_topk_indices(
            [kv_len - 1 for kv_len in per_row_kv_lens],
            self.scenario.compress_ratios[SPARSE_LAYER_IDX],
            self.scenario.index_topk,
            device,
        )

    def step(self) -> torch.Tensor:
        """One generation step. Repeatable: every write target comes from the
        metadata, which is fixed, so re-running rewrites the same slots."""
        self.layer.mla_rope_generation(
            self.fused_q,
            self.q_pe,
            self.latent_cache,
            self.metadata,
            self.cu_q_seqlens,
            self.cu_kv_seqlens,
            self.fmha_scheduler_counter,
            None,
            None,
            None,
        )
        return self.layer.forward(
            self.fused_q,
            None,
            None,
            self.metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=self.latent_cache,
            q_pe=self.q_pe,
            cu_q_seqlens=self.cu_q_seqlens,
            cu_kv_seqlens=self.cu_kv_seqlens,
            fmha_scheduler_counter=self.fmha_scheduler_counter,
            topk_indices=self.topk_indices,
        )

    def shutdown(self) -> None:
        self.cache_manager.shutdown()


def _time_step(harness: PresentationHarness, warmup: int,
               iters: int) -> Tuple[float, float]:
    """Median and minimum per-step latency in milliseconds.

    Median rather than mean: a single stray context switch on a shared node
    should not decide a design.
    """
    for _ in range(warmup):
        harness.step()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        harness.step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def _run_point(batch_size: int, tier: int, context_len: int, warmup: int,
               iters: int, check: bool) -> Dict[str, object]:
    next_n = tier + 1
    scenario = Scenario()
    device = torch.device("cuda")
    torch.manual_seed(42)

    record: Dict[str, object] = {
        "batch_size": batch_size,
        "tier": tier,
        "next_n": next_n,
        "context_len": context_len,
        "total_query_tokens": batch_size * next_n,
    }

    # Batch-major: B sequences, next_n query tokens each.
    batch_major = PresentationHarness(scenario, [context_len] * batch_size,
                                      next_n, device)
    # Token-major: one sequence per query token. Request (i, j) is cached at
    # L + j so its single query token has the same extent as row j above.
    token_cached = [
        context_len + j for _ in range(batch_size) for j in range(next_n)
    ]
    token_major = PresentationHarness(scenario, token_cached, 1, device)

    try:
        if check:
            out_batch = batch_major.step()
            out_token = token_major.step()
            record["batch_major_output_shape"] = list(out_batch.shape)
            record["token_major_output_shape"] = list(out_token.shape)
            record["shapes_match"] = (out_batch.shape == out_token.shape)
            for name, out in (("batch_major", out_batch), ("token_major",
                                                           out_token)):
                record[f"{name}_finite"] = bool(torch.isfinite(out).all())
                record[f"{name}_abs_mean"] = float(out.abs().mean())
            # The two runs use different random KV, so the outputs are not
            # expected to match elementwise -- only to be the same shape, finite,
            # and of the same magnitude. A token-major layout that mis-attributes
            # rows shows up here as a magnitude that is off, or as non-finite
            # values, because each pseudo-request's extent differs by one.

        batch_median, batch_min = _time_step(batch_major, warmup, iters)
        token_median, token_min = _time_step(token_major, warmup, iters)
    finally:
        batch_major.shutdown()
        token_major.shutdown()

    record["batch_major_ms_median"] = batch_median
    record["batch_major_ms_min"] = batch_min
    record["token_major_ms_median"] = token_median
    record["token_major_ms_min"] = token_min
    record["token_major_slowdown"] = (token_median /
                                      batch_median if batch_median else
                                      float("inf"))
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-sizes",
                        type=int,
                        nargs="+",
                        default=[8, 32, 128])
    parser.add_argument("--tiers",
                        type=int,
                        nargs="+",
                        default=[1, 3, 5],
                        help="verify-length tiers; next_n = tier + 1")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--check",
                        action="store_true",
                        help="also run a one-shot sanity check per point")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark", file=sys.stderr)
        return 2

    print(f"device: {torch.cuda.get_device_name(0)} "
          f"(sm{''.join(str(c) for c in torch.cuda.get_device_capability())})")
    print(f"context_len={args.context_len} warmup={args.warmup} "
          f"iters={args.iters}\n")
    header = (f"{'B':>5} {'tier':>5} {'next_n':>7} {'tokens':>7} "
              f"{'batch-major ms':>15} {'token-major ms':>15} {'slowdown':>9}")
    print(header)
    print("-" * len(header))

    records: List[Dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for tier in args.tiers:
            try:
                record = _run_point(batch_size, tier, args.context_len,
                                    args.warmup, args.iters, args.check)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"{batch_size:>5} {tier:>5} {'':>7} {'':>7} "
                      f"FAILED: {type(exc).__name__}: {exc}")
                records.append({
                    "batch_size": batch_size,
                    "tier": tier,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            records.append(record)
            print(f"{record['batch_size']:>5} {record['tier']:>5} "
                  f"{record['next_n']:>7} {record['total_query_tokens']:>7} "
                  f"{record['batch_major_ms_median']:>15.3f} "
                  f"{record['token_major_ms_median']:>15.3f} "
                  f"{record['token_major_slowdown']:>9.2f}x")

    ok = [r for r in records if "error" not in r]
    if ok:
        worst = max(r["token_major_slowdown"] for r in ok)
        best = min(r["token_major_slowdown"] for r in ok)
        print(f"\ntoken-major slowdown across points: "
              f"{best:.2f}x .. {worst:.2f}x")
        print("Read: >~1.3x sustained means the ragged (D1/D2) route is paying "
              "more than it can save; fall back to the uniform tier ladder "
              "(D3 Stage A), which needs no kernel change.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
