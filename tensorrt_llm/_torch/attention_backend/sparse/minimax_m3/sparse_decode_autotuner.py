# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-shape Triton/MSA tactic selection for MiniMax-M3 sparse decode."""

from __future__ import annotations

import contextlib

import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    DistributedTuningStrategy,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
    autotune,
)
from tensorrt_llm.logger import logger

_CUSTOM_OP = "trtllm::minimax_m3_sparse_decode"
# A failed graph-warmup attempt must not be retried by every sparse layer.
# The set stays bounded by the finite CUDA-graph shapes prepared at startup.
_attempted_tuning_keys: set[tuple] = set()


class MiniMaxM3SparseDecodeRunner(TunableRunner):
    """Run either complete sparse-decode kernel with identical inputs.

    The fallback tactic is Triton, which is the production route predating the
    MSA sparse-decode port.  MSA is considered only when the caller supplies a
    prebuilt plan; metadata preparation guarantees that for ``adaptive``.
    """

    # All TP ranks capture the same graph key and must embed the same tactic.
    # Profiling independently makes near-tie shapes vulnerable to rank-local
    # timer noise. Rank 0 is representative on homogeneous TP nodes, so tune
    # once and broadcast the cached choice before capture.
    tuning_config = TuningConfig(
        use_cuda_graph=True,
        distributed_tuning_strategy=DistributedTuningStrategy.BROADCAST,
    )

    def __init__(
        self,
        *,
        decode_query_len: int,
        input_layouts: tuple[tuple[torch.dtype, tuple[int, ...]], ...],
        sm_scale: float,
    ) -> None:
        self.decode_query_len = int(decode_query_len)
        self.input_layouts = input_layouts
        self.sm_scale = float(sm_scale)

    def unique_id(self) -> tuple:
        # AutoTuner already keys every input shape, which carries batch,
        # HQ/HKV, head dim, page size, and topK. Add the non-shape properties
        # that can change tactic performance. In particular, Q may be a
        # strided view of a fused projection even when its shape is unchanged.
        return (
            self.decode_query_len,
            self.input_layouts,
        )

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list[str]:
        del inputs, profile, kwargs
        return ["triton", "msa"]

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic: str | int = -1,
        plan: tuple | None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        q, k_paged, v_paged, block_indexes, block_table, seq_lens, output = inputs

        if tactic == -1:
            tactic = "triton"
        if tactic == "triton":
            from .triton_sparse_decode import minimax_m3_sparse_attn_decode

            minimax_m3_sparse_attn_decode(
                q,
                k_paged,
                v_paged,
                block_indexes.permute(1, 0, 2),
                block_table,
                seq_lens,
                sm_scale=self.sm_scale,
                output=output,
                decode_query_len=self.decode_query_len,
            )
            return output
        if tactic == "msa":
            if plan is None:
                raise RuntimeError(
                    "MiniMax-M3 selected the MSA sparse decode tactic without "
                    "a preplanned GQA plan."
                )
            from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_sparse_gqa

            use_fp8 = k_paged.dtype == torch.float8_e4m3fn
            msa_q = q
            if use_fp8 and msa_q.dtype != torch.float8_e4m3fn:
                msa_q = msa_q.to(torch.float8_e4m3fn)
            run_msa_sparse_gqa(
                msa_q,
                k_paged,
                v_paged,
                block_indexes,
                kv_indices=block_table.flatten(),
                sm_scale=self.sm_scale,
                causal=True,
                head_dim=int(q.shape[2]),
                plan=plan,
                out=output,
                use_fp8=use_fp8,
            )
            return output
        raise ValueError(f"Unsupported MiniMax-M3 sparse decode tactic: {tactic!r}.")


def run_adaptive_sparse_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    block_indexes: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    sm_scale: float,
    decode_query_len: int,
    plan: tuple | None,
    is_cuda_graph_metadata: bool,
) -> str | None:
    """Profile once per exact shape, cache the winner, and run it.

    Only a non-capturing call made with CUDA-graph metadata may seed the cache.
    Eager startup warmups can have the same tensor shapes but nonrepresentative
    sequence lengths, so allowing them to profile could poison the tactic later
    embedded in the graph.  During capture, a cache miss deliberately takes
    AutoTuner's Triton fallback instead of attempting nested profiling; the
    captured graph therefore never changes tactic across replays. Returns the
    stable cached tactic, or None when the call used an uncached fallback.
    """
    inputs = [q, k_paged, v_paged, block_indexes, block_table, seq_lens, output]
    runner = MiniMaxM3SparseDecodeRunner(
        decode_query_len=decode_query_len,
        input_layouts=tuple((tensor.dtype, tuple(tensor.stride())) for tensor in inputs),
        sm_scale=sm_scale,
    )
    tuner = AutoTuner.get()
    input_shapes = tuple(tensor.size() for tensor in inputs)
    cache_hit, _, cached_tactic, _ = tuner.profiling_cache.search_cache(
        _CUSTOM_OP,
        [runner],
        input_shapes,
        runner.tuning_config,
    )
    if cache_hit and cached_tactic == "msa" and plan is None:
        raise RuntimeError(
            "MiniMax-M3 cached the MSA sparse decode tactic without a preplanned GQA plan."
        )
    if not cache_hit and not is_cuda_graph_metadata:
        # An eager startup warmup is intentionally ineligible to seed a tactic.
        # Bypass choose_one() on its miss so AutoTuner does not report the
        # expected Triton fallback as a graph-setup cache failure.
        runner(inputs, tactic=-1, plan=plan)
        return None

    tuning_key = tuner.profiling_cache.get_cache_key(
        _CUSTOM_OP,
        runner,
        input_shapes,
        runner.tuning_config,
    )
    # AutoTuner's tune context performs pipeline cache handoff.  Graph warmup
    # does not run that coordinator, so PP configurations conservatively keep
    # the Triton fallback until graph-tuning orchestration supports them.
    can_profile = (
        is_cuda_graph_metadata
        and q.is_cuda
        and not torch.cuda.is_current_stream_capturing()
        and not tuner.mapping.has_pp()
    )
    should_profile = (
        not cache_hit
        and tuning_key not in _attempted_tuning_keys
        and can_profile
        and not tuner.is_tuning_mode
    )
    if should_profile:
        if plan is None:
            raise RuntimeError(
                "MiniMax-M3 adaptive sparse decode cannot profile MSA without "
                "a preplanned GQA plan."
            )
        _attempted_tuning_keys.add(tuning_key)

    tune_context = autotune() if should_profile else contextlib.nullcontext()
    with tune_context:
        _, tactic = tuner.choose_one(
            _CUSTOM_OP,
            [runner],
            runner.tuning_config,
            inputs,
            plan=plan,
        )
    stable_cache_hit, _, stable_tactic, _ = tuner.profiling_cache.search_cache(
        _CUSTOM_OP,
        [runner],
        input_shapes,
        runner.tuning_config,
    )
    if stable_cache_hit:
        stable_tactic = "triton" if stable_tactic == -1 else stable_tactic
        logger.info_once(
            "MiniMax-M3 adaptive sparse decode selected "
            f"{stable_tactic} for B={int(block_table.shape[0])}, "
            f"DQL={decode_query_len}, total_q={int(q.shape[0])}, "
            f"local HQ/HKV={int(q.shape[1])}/{int(k_paged.shape[1])}.",
            key=(_CUSTOM_OP, tuning_key, stable_tactic),
        )
    runner(inputs, tactic=tactic, plan=plan)
    return stable_tactic if stable_cache_hit else None
