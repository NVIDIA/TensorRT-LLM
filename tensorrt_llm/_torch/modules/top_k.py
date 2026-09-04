# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable index-selection Top-K module for sparse inference paths."""

from __future__ import annotations

import os
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

from ..locality_domain.gvr_topk import (
    GvrTopKRowShardPlan,
    is_gvr_topk_locality_workload_large_enough,
    plan_gvr_topk_row_shards,
)
from ..memory_buffer_utils import get_memory_buffers

if TYPE_CHECKING:
    from ..locality_domain.runtime import LocalityDomainRuntime


class TopKImplementation(str, Enum):
    """Top-K implementations grouped by backend and algorithm."""

    TORCH = "torch"
    CUDA_RADIX = "cuda_radix"
    CUTE_DSL_RADIX = "cute_dsl_radix"
    CUTE_DSL_GVR = "cute_dsl_gvr"


_GVR_IMPLEMENTATIONS = {
    TopKImplementation.CUTE_DSL_GVR,
}
_MAX_RADIX_BLOCKS_PER_ROW = 10
# One 16-byte vector per copy, matching the Blackwell prefill op's contract.
_CUTE_DSL_PREFILL_COPY_BITS = 128


class TopK(nn.Module):
    """Select Top-K indices for sparse prefill and decode paths.

    GVR decode state is owned by the caller so it can be shared with the
    request metadata and retain a stable address across CUDA Graph replays.
    """

    _memory_buffers = get_memory_buffers()

    def __init__(
        self,
        top_k: int,
        *,
        prefill_implementation: TopKImplementation | None = None,
        decode_implementation: TopKImplementation | None = None,
        compress_ratio: int = 1,
        gvr_self_sampling: bool = True,
        use_gvr_locality_domain: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.prefill_implementation = TopKImplementation(
            prefill_implementation or TopKImplementation.CUDA_RADIX
        )
        self.decode_implementation = TopKImplementation(
            decode_implementation or TopKImplementation.CUDA_RADIX
        )
        self.compress_ratio = compress_ratio
        # Second-level GVR dispatch for CUTE_DSL_GVR: True selects the
        # hint-free self-sampling engine, False the temporal-hint engine.
        self.gvr_self_sampling = gvr_self_sampling
        # Rubin locality-domain row sharding is an explicit prototype opt-in.
        # Runtime resources remain lazy so default and unsupported paths keep
        # the original full-device GVR lifecycle.
        self.use_gvr_locality_domain = use_gvr_locality_domain
        self._gvr_locality_runtime: LocalityDomainRuntime | None = None
        self._gvr_locality_capability: dict[int, bool] = {}
        self._gvr_locality_topologies: dict[int, tuple[tuple[int, int], ...]] = {}
        self._gvr_locality_ready_launches: set[tuple] = set()
        # emission-assisted GVR (opt-in via prepare_gvr_emission): the module
        # owns the closed-loop emission state; only reachable on the temporal
        # (gvr_self_sampling=False) V1 path.
        self._gvr_emission_state = None
        self._gvr_emission_route = None
        self._gvr_emission_armed = False

    @property
    def needs_gvr_prior(self) -> bool:
        """Return whether decode consumes previous-step Top-K indices."""
        return (
            self.decode_implementation == TopKImplementation.CUTE_DSL_GVR
            and not self.gvr_self_sampling
        )

    def forward(
        self,
        scores: torch.Tensor,
        output_indices: torch.Tensor,
        *,
        is_prefill: bool,
        row_starts: torch.Tensor | None = None,
        row_ends: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        scan_lengths: torch.Tensor | None = None,
        next_n: int = 1,
        max_seq_len: int | None = None,
        gvr_ext_kwargs: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Write prefill or decode Top-K indices into ``output_indices``.

        Args:
            scores: Top-K input scores with shape ``[num_rows, num_columns]``.
            output_indices: Int32 output with shape ``[num_rows, top_k]``.
            is_prefill: Whether to run the prefill implementation.
            row_starts: Per-row inclusive starts for prefill.
            row_ends: Per-row exclusive ends for prefill.
            sequence_lengths: Per-request logical KV lengths for decode.
            scan_lengths: Per-request score-column lengths for decode.
            next_n: Number of decode rows per request.
            max_seq_len: Maximum decode score width used for GVR kernel tuning.
            gvr_ext_kwargs: GVR-only keyword arguments. ``gvr_prior_indices``
                is required by the temporal GVR path (``CUTE_DSL_GVR`` with
                ``gvr_self_sampling=False``). It is
                caller-owned int32 previous selection with shape
                ``[num_requests, top_k]`` on ``scores.device``. The
                self-sampling engine does not consume this state.
                ``gvr_row_order`` is an optional int32 request ordering with
                shape ``[num_requests]`` on the same device.

        Returns:
            ``output_indices`` after the selected implementation writes it.
        """
        if is_prefill:
            assert row_starts is not None and row_ends is not None
            return self._forward_prefill(scores, row_starts, row_ends, output_indices)

        assert sequence_lengths is not None and scan_lengths is not None
        return self._forward_decode(
            scores,
            sequence_lengths,
            scan_lengths,
            output_indices,
            next_n,
            max_seq_len,
            gvr_ext_kwargs,
        )

    def _forward_prefill(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.prefill_implementation == TopKImplementation.TORCH:
            return self._forward_prefill_torch(
                scores,
                row_starts,
                row_ends,
                output_indices,
            )
        if self.prefill_implementation == TopKImplementation.CUTE_DSL_GVR:
            # hint-free k derives from the output width; pin it to the module's k
            assert output_indices.shape[1] == self.top_k
            if not self.gvr_self_sampling:
                # the temporal (hint) GVR engine has no prefill form
                logger.warning_once(
                    "temporal GVR has no prefill engine; using the CUDA radix prefill Top-K.",
                    key="gvr_temporal_prefill_radix",
                )
            elif scores.shape[1] <= self.top_k:
                # every row is short (nv <= k): the exact radix path emits the
                # identity/-1 answer without reading logits — cheaper than a
                # zero-work self-sampling launch. Deliberate, no warning.
                pass
            elif self._selfsampling_prefill_ok(scores):
                from ..cute_dsl_kernels.blackwell.top_k import selfsampling_topk_run_prefill

                logger.info_once(
                    "self-sampling GVR prefill top-K engaged "
                    f"(K={self.top_k}, cr={self.compress_ratio}, hint-free).",
                    key="selfsampling_topk_prefill_engaged",
                )
                # ks/ke are already in compressed column units; run_prefill
                # writes the local (column - ks) frame with -1 pad and no host
                # reads (envelope from scores.shape[1]).
                selfsampling_topk_run_prefill(scores, row_starts, row_ends, output_indices)
                return output_indices
            else:
                # engine hardware-format gate missed (e.g. a non-fp4 layer with
                # an odd DeepGEMM width, or a bf16 producer): exact radix.
                logger.warning_once(
                    "self-sampling GVR prefill is selected but the scores do "
                    "not satisfy the engine's hardware-format gate "
                    f"(dtype={scores.dtype}, strides={tuple(scores.stride())}); "
                    "falling back to the CUDA radix prefill Top-K.",
                    key="selfsampling_topk_prefill_fallthrough",
                )
        elif self.prefill_implementation == TopKImplementation.CUTE_DSL_RADIX:
            # Keep the op's reread policy default; only its copy width is tuned.
            torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                scores,
                row_starts,
                row_ends,
                output_indices,
                self.top_k,
                _CUTE_DSL_PREFILL_COPY_BITS,
            )
            return output_indices
        elif self.prefill_implementation != TopKImplementation.CUDA_RADIX:
            raise NotImplementedError(
                f"{self.prefill_implementation.value} does not support prefill Top-K"
            )
        torch.ops.trtllm.indexer_topk_prefill(
            scores,
            row_starts,
            row_ends,
            output_indices,
            self.top_k,
        )
        return output_indices

    def _selfsampling_prefill_ok(self, scores: torch.Tensor) -> bool:
        """Engine hardware-format gate for the self-sampling prefill Top-K.

        fp32 row-major scores with a float4-aligned row stride and a 16B base
        (the DeepGEMM prefill logits arena, whose rows are 1024B-aligned). The
        all-short tile case is handled by the caller before this check."""
        return (
            scores.dtype == torch.float32
            and scores.stride(1) == 1
            and scores.stride(0) % 4 == 0
            and scores.data_ptr() % 16 == 0
        )

    def _forward_decode(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int | None,
        gvr_ext_kwargs: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor:
        if self.decode_implementation == TopKImplementation.TORCH:
            return self._forward_decode_torch(scores, scan_lengths, output_indices, next_n)

        if self.decode_implementation in _GVR_IMPLEMENTATIONS:
            return self._forward_decode_gvr(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                max_seq_len=max_seq_len,
                **(gvr_ext_kwargs or {}),
            )

        return self._forward_decode_radix(
            scores,
            sequence_lengths,
            scan_lengths,
            output_indices,
            next_n,
        )

    def _forward_decode_radix(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
    ) -> torch.Tensor:
        use_cute_dsl = self.decode_implementation == TopKImplementation.CUTE_DSL_RADIX and not (
            self.compress_ratio > 1 and next_n > 1
        )
        if use_cute_dsl:
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                scores,
                scan_lengths,
                output_indices,
                self.top_k,
                next_n,
            )
            return output_indices

        radix_indices, radix_values = self._get_radix_workspace(scores)
        torch.ops.trtllm.indexer_topk_decode(
            scores,
            sequence_lengths,
            output_indices,
            next_n,
            self.top_k,
            compress_ratio=self.compress_ratio,
            radix_aux_indices=radix_indices,
            radix_aux_logits=radix_values,
        )
        return output_indices

    def _get_workspace(
        self,
        scores: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        buffer_name: str,
    ) -> torch.Tensor:
        device_buffer_name = f"{buffer_name}_{scores.device}"
        if scores.is_cuda:
            with torch.cuda.device(scores.device):
                capture_graph = torch.cuda.is_current_stream_capturing()
                return self._memory_buffers.get_buffer(
                    shape,
                    dtype=dtype,
                    buffer_name=device_buffer_name,
                    reserve_buffer=capture_graph,
                )
        return self._memory_buffers.get_buffer(
            shape,
            dtype=dtype,
            buffer_name=device_buffer_name,
            reserve_buffer=False,
        )

    def _get_radix_workspace(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if scores.dtype != torch.float32:
            # The C++ bf16/fp16 entry has no split-work tier or aux-buffer
            # arguments and rejects widths that would require split work.
            return None, None

        shape = (scores.shape[0], _MAX_RADIX_BLOCKS_PER_ROW, self.top_k)
        radix_indices = self._get_workspace(
            scores,
            shape,
            torch.int32,
            "top_k_radix_indices_workspace",
        )
        radix_values = self._get_workspace(
            scores,
            shape,
            torch.float32,
            "top_k_radix_values_workspace",
        )
        return radix_indices, radix_values

    def _build_gvr_locality_launch(
        self,
        scores: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> tuple["LocalityDomainRuntime", GvrTopKRowShardPlan, tuple] | None:
        """Build a capture-stable Rubin row-sharding launch, if eligible."""
        if not self.use_gvr_locality_domain or not scores.is_cuda:
            return None

        score_width = min(int(max_seq_len), int(scores.shape[1]))
        if not is_gvr_topk_locality_workload_large_enough(
            num_rows=int(scores.shape[0]),
            next_n=int(next_n),
            score_width=score_width,
            top_k=self.top_k,
        ):
            # In particular, BS=1 and small score envelopes never initialize
            # locality-domain resources and retain the full-device launch.
            return None

        device_index = scores.get_device()
        with torch.cuda.device(device_index):
            from ..locality_domain_utils import (
                get_current_locality_domain,
                is_locality_domain_supported,
            )

            # Avoid recursively partitioning a Top-K already submitted under
            # another locality-domain composite.
            if get_current_locality_domain() is not None:
                return None

            capturing = torch.cuda.is_current_stream_capturing()
            capable = self._gvr_locality_capability.get(device_index)
            if capable is None:
                if capturing:
                    raise RuntimeError(
                        "GVR locality-domain capability is cold during CUDA "
                        "Graph capture; run this shape once eagerly before capture"
                    )
                from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

                properties = torch.cuda.get_device_properties(device_index)
                sm_version = int(properties.major) * 10 + int(properties.minor)
                capable = (
                    sm_version == 107
                    and IS_CUTLASS_DSL_RUBIN_AVAILABLE
                    and os.environ.get("DISABLE_LOCALITY_DOMAINS", "0") != "1"
                    and is_locality_domain_supported(device_index)
                )
                self._gvr_locality_capability[device_index] = capable
            if not capable:
                logger.warning_once(
                    "use_gvr_locality_domain=True but Rubin locality-domain "
                    "execution is unavailable; keeping self-sampling GVR on "
                    "the full-device stream.",
                    key="gvr_locality_domain_unavailable",
                )
                return None

            runtime = self._gvr_locality_runtime
            if runtime is None:
                if capturing:
                    raise RuntimeError(
                        "GVR locality-domain resources are cold during CUDA "
                        "Graph capture; run this shape once eagerly before capture"
                    )
                from ..locality_domain.runtime import LocalityDomainRuntime

                runtime = LocalityDomainRuntime(num_partitions=2)
                self._gvr_locality_runtime = runtime

            topology = self._gvr_locality_topologies.get(device_index)
            if topology is None:
                if capturing:
                    raise RuntimeError(
                        "GVR locality-domain topology is cold during CUDA Graph "
                        "capture; run this shape once eagerly before capture"
                    )
                topology = runtime.topology_identity()
                self._gvr_locality_topologies[device_index] = topology

            properties = torch.cuda.get_device_properties(device_index)
            device_num_sms = int(properties.multi_processor_count)
            if any(total_sms != device_num_sms for _, total_sms in topology):
                raise RuntimeError(
                    "GVR locality-domain topology does not match the score "
                    f"device: topology={topology}, device_num_sms={device_num_sms}"
                )
            try:
                plan = plan_gvr_topk_row_shards(
                    num_rows=int(scores.shape[0]),
                    next_n=int(next_n),
                    score_width=score_width,
                    top_k=self.top_k,
                    topology=topology,
                )
            except ValueError as error:
                raise RuntimeError(f"invalid GVR locality-domain topology: {error}") from error
            if plan is None:
                return None

            # run_varlen derives npad from shape[1] for a one-row launch. Do
            # not turn a legal multi-row strided view into an illegal shard.
            if any(shard.num_rows == 1 for shard in plan.shards) and scores.shape[1] % 4:
                return None

            shard_geometry = tuple(
                (
                    shard.num_rows,
                    int(scores.shape[1]) if shard.num_rows == 1 else int(scores.stride(0)),
                    shard.num_sms,
                )
                for shard in plan.shards
            )
            launch_key = (
                device_index,
                shard_geometry,
                self.top_k,
                int(max_seq_len),
                int(next_n),
                self.compress_ratio,
                plan.topology,
            )
            if capturing and launch_key not in self._gvr_locality_ready_launches:
                raise RuntimeError(
                    "GVR locality-domain launchers or workspaces are cold "
                    "during CUDA Graph capture; run this shape once eagerly "
                    "before capture"
                )
            return runtime, plan, launch_key

    def _run_gvr_locality_domain(
        self,
        runner: Callable[..., None],
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> bool:
        """Run two non-overlapping GVR row slices on Rubin locality domains."""
        launch = self._build_gvr_locality_launch(scores, next_n, max_seq_len)
        if launch is None:
            return False
        runtime, plan, launch_key = launch

        num_rows = int(scores.shape[0])
        num_requests = num_rows // int(next_n)
        if tuple(output_indices.shape) != (num_rows, self.top_k):
            raise RuntimeError(
                "GVR locality-domain output shape must match the unsplit "
                f"launch: expected {(num_rows, self.top_k)}, got "
                f"{tuple(output_indices.shape)}"
            )
        if sequence_lengths.dim() != 1 or int(sequence_lengths.shape[0]) != num_requests:
            raise RuntimeError(
                "GVR locality-domain sequence_lengths must have one entry "
                f"per request: expected {(num_requests,)}, got "
                f"{tuple(sequence_lengths.shape)}"
            )
        if sequence_lengths.device != scores.device or output_indices.device != scores.device:
            raise RuntimeError(
                "GVR locality-domain scores, sequence_lengths, and output_indices "
                "must be on the same device"
            )

        # CPU tensors are useful for orchestration tests with a mocked launch
        # plan/runtime. Production locality launches always enter the CUDA
        # device guard above.
        device_context = torch.cuda.device(scores.device) if scores.is_cuda else nullcontext()
        with device_context:
            runtime.fork()
            try:
                for shard in plan.shards:
                    with runtime.partition_context(shard.partition_id):
                        runner(
                            scores[shard.row_start : shard.row_end],
                            sequence_lengths[shard.request_start : shard.request_end],
                            output_indices[shard.row_start : shard.row_end],
                            next_n=next_n,
                            compress_ratio=self.compress_ratio,
                            max_seq_len=max_seq_len * self.compress_ratio,
                        )
            except Exception:
                # Preserve the launch failure if cleanup also fails; the
                # original exception is the actionable cause.
                try:
                    runtime.join()
                except Exception:
                    logger.exception(
                        "failed to join Rubin locality-domain streams while "
                        "handling a GVR Top-K launch failure"
                    )
                raise
            runtime.join()

        # Mark ready only after both launches and the join complete. The key
        # covers every leaf cache/workspace dimension needed during capture.
        self._gvr_locality_ready_launches.add(launch_key)
        logger.info_once(
            "Rubin locality-domain self-sampling GVR Top-K engaged; only "
            "the already-produced Top-K rows are sharded (the logits "
            "producer remains full-device).",
            key="gvr_locality_domain_engaged",
        )
        return True

    def _forward_decode_gvr(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int | None,
        gvr_prior_indices: torch.Tensor | None = None,
        gvr_row_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.decode_implementation == TopKImplementation.CUTE_DSL_GVR and self.gvr_self_sampling:
            assert max_seq_len is not None
            if (
                # engine hardware-format gate (falls through otherwise):
                # fp32 row-major scores with a float4-aligned row stride and
                # a 16B-aligned base (the DSL paged-MQA arena view — column-
                # sliced from a 256-aligned buffer — satisfies this; odd
                # max_seq_len DeepGEMM layouts do not). Single-row batches
                # derive their row window from shape[1] (arena last-row
                # safety), so that width must satisfy the same float4 rule —
                # otherwise run_varlen raises instead of falling through.
                scores.dtype == torch.float32
                and scores.stride(1) == 1
                and scores.stride(0) % 4 == 0
                and scores.data_ptr() % 16 == 0
                and (scores.shape[0] > 1 or scores.shape[1] % 4 == 0)
            ):
                # hint-free k derives from the output width; pin it to the module's k
                assert output_indices.shape[1] == self.top_k
                from ..cute_dsl_kernels.blackwell.top_k import selfsampling_topk_run_varlen

                logger.info_once(
                    "self-sampling GVR top-K engaged "
                    f"(K={self.top_k}, cr={self.compress_ratio}, "
                    f"next_n={next_n}, hint-free).",
                    key="selfsampling_topk_engaged",
                )
                # Self-sampling GVR varlen engine:
                # one launch for the batch; per-row n from device kv_lens,
                # capture-stable tuning from the max-seq-len engine constant
                # (no host reads — CUDA-graph safe). The module receives
                # max_seq_len in compressed index space; run_varlen's value
                # is in KV-token space like sequence_lengths, so multiply it
                # back by the compression ratio.
                if not (
                    self.use_gvr_locality_domain
                    and self._run_gvr_locality_domain(
                        selfsampling_topk_run_varlen,
                        scores,
                        sequence_lengths,
                        output_indices,
                        next_n,
                        max_seq_len,
                    )
                ):
                    selfsampling_topk_run_varlen(
                        scores,
                        sequence_lengths,
                        output_indices,
                        next_n=next_n,
                        compress_ratio=self.compress_ratio,
                        max_seq_len=max_seq_len * self.compress_ratio,
                    )
                return output_indices
            logger.warning_once(
                "self-sampling GVR is selected but the decode scores do not "
                "satisfy the engine's hardware-format gate "
                f"(dtype={scores.dtype}, strides={tuple(scores.stride())}); "
                "falling back to the CUDA insertion/radix Top-K path.",
                key="selfsampling_topk_fallthrough",
            )
            radix_indices, radix_values = self._get_radix_workspace(scores)
            torch.ops.trtllm.indexer_topk_decode(
                scores,
                sequence_lengths,
                output_indices,
                next_n,
                self.top_k,
                compress_ratio=self.compress_ratio,
                radix_aux_indices=radix_indices,
                radix_aux_logits=radix_values,
            )
            return output_indices

        assert gvr_prior_indices is not None
        assert max_seq_len is not None
        # V1 temporal (DSL). Emission-assisted candidates (opt-in) are only
        # armed on this hint-first path; the self-sampling V2 path above never
        # arms them.
        emission_kwargs: dict = {}
        if self._gvr_emission_armed:
            state = self._gvr_emission_state
            num_rows = scores.shape[0]
            emission_kwargs = state.topk_ext_kwargs(
                self._gvr_emission_route,
                num_rows,
                state.block_max[:num_rows] if state.block_max is not None else None,
            )
            self._gvr_emission_armed = False
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            scores,
            gvr_prior_indices,
            sequence_lengths,
            output_indices,
            self.top_k,
            next_n=next_n,
            compress_ratio=self.compress_ratio,
            max_seq_len=max_seq_len,
            order_row=gvr_row_order,
            **emission_kwargs,
        )
        return output_indices

    def prepare_gvr_emission(
        self,
        batch: int,
        n_comp: int,
        num_sms: int,
        gvr_prior_indices: torch.Tensor,
    ) -> dict:
        """Plan the emission-assisted GVR tier for this decode step.

        Returns the emission kwargs for the paged-MQA scoring op (empty
        when the planner declines this step); the matching consume-side
        kwargs are injected into the next GVR Top-K call automatically.
        Host arithmetic on engine-static shapes plus capturable device
        ops, so a captured graph bakes the tier and replays refresh the
        state buffers in place.

        Args:
            batch: Number of decode requests this step.
            n_comp: Engine-static compressed maximum sequence length.
            num_sms: Device SM count.
            gvr_prior_indices: Caller-owned previous-selection state;
                defines the emission state's row capacity and device.
        """
        # the emission/xstate writes are undeclared mutations (see the op's
        # schema note), so the tier is eager / CUDA-graph only
        if torch.compiler.is_dynamo_compiling():
            return {}
        from ..cute_dsl_kernels.blackwell.top_k.gvr_emission import (
            LIST_EMIT_MIN_N,
            GvrEmissionState,
        )

        if self._gvr_emission_state is None:
            self._gvr_emission_state = GvrEmissionState(
                max_rows=gvr_prior_indices.shape[0],
                top_k=self.top_k,
                device=gvr_prior_indices.device,
                enable_list_tier=n_comp >= LIST_EMIT_MIN_N,
                own_prior=False,
            )
        state = self._gvr_emission_state
        emit_tier, self._gvr_emission_route = state.plan(
            batch, n_comp, num_sms, compress_ratio=max(self.compress_ratio, 1)
        )
        self._gvr_emission_armed = self._gvr_emission_route.tier != "none"
        if emit_tier in ("counts", "list", "rungs"):
            state.update_seed_rows(batch, emit_tier)
        kwargs: dict = {}
        if emit_tier in ("counts", "list"):
            kwargs = state.indexer_emit_kwargs(emit_tier, batch)
        if self._gvr_emission_route.attach_block_max or emit_tier in ("counts", "list"):
            kwargs["block_max_out"] = state.ensure_block_max(n_comp)[:batch]
        return kwargs

    def reset_gvr_emission_rows(self, rows: slice) -> None:
        """Cold-start the emission closed-loop state for reused request
        slots (prefill-to-decode handoff): a zeroed xstate reads as
        invalid and routes those rows to the stock path in-kernel."""
        if self._gvr_emission_state is not None:
            self._gvr_emission_state.xstate[rows].zero_()

    def update_gvr_prior_from_prefill(
        self,
        output_indices: torch.Tensor,
        request_lengths: torch.Tensor,
        gvr_prior_indices: torch.Tensor | None,
        *,
        request_offset: int = 0,
    ) -> None:
        """Update GVR prior indices from each prefill request's last row.

        Args:
            output_indices: Int32 prefill selections with shape
                ``[num_prefill_rows, top_k]``.
            request_lengths: Per-request prefill row counts on
                ``output_indices.device``; a host tensor here makes the row
                gather a synchronous host-to-device copy.
            gvr_prior_indices: Int32 caller-owned state on
                ``output_indices.device`` with shape ``[capacity, top_k]``.
                The slice starting at ``request_offset`` is updated in place.
            request_offset: First request row to update in the prior state.
        """
        if not self.needs_gvr_prior:
            return
        assert gvr_prior_indices is not None
        last_rows = (torch.cumsum(request_lengths, dim=0) - 1).to(dtype=torch.long)
        num_requests = request_lengths.shape[0]
        gvr_prior_indices[request_offset : request_offset + num_requests].copy_(
            output_indices[last_rows]
        )

    def _forward_prefill_torch(
        self,
        scores: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        output_indices.fill_(-1)
        selected_count = min(self.top_k, scores.shape[1])
        if selected_count == 0:
            return output_indices
        columns = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        valid = (columns >= row_starts.unsqueeze(1)) & (columns < row_ends.unsqueeze(1))
        selected = scores.masked_fill(~valid, float("-inf")).topk(selected_count, dim=-1).indices
        selected_valid = torch.gather(valid, 1, selected)
        selected = selected - row_starts.unsqueeze(1)
        selected = selected.masked_fill(~selected_valid, -1)
        output_indices[:, :selected_count].copy_(selected.to(torch.int32))
        return output_indices

    def _forward_decode_torch(
        self,
        scores: torch.Tensor,
        scan_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
    ) -> torch.Tensor:
        output_indices.fill_(-1)
        selected_count = min(self.top_k, scores.shape[1])
        if selected_count == 0:
            return output_indices
        positions = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
        row_indices = torch.arange(scores.shape[0], device=scores.device) // next_n
        next_n_offsets = torch.arange(scores.shape[0], device=scores.device) % next_n
        row_ends = scan_lengths[row_indices] - next_n + next_n_offsets + 1
        valid = positions < row_ends.unsqueeze(1)
        selected = scores.masked_fill(~valid, float("-inf")).topk(selected_count, dim=-1).indices
        selected_valid = torch.gather(valid, 1, selected)
        selected = selected.masked_fill(~selected_valid, -1)
        output_indices[:, :selected_count].copy_(selected.to(torch.int32))
        return output_indices
