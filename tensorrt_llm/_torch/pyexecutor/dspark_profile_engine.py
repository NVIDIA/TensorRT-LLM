# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-serving fixed-cell DSpark profiling on the original model engine."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any

import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    DSparkCUDAGraphRunner,
)
from tensorrt_llm._torch.pyexecutor.dspark_profile_graph_storage import (
    snapshot_metadata_storage,
    validate_metadata_storage,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker, resolve_confidence_head
from tensorrt_llm._torch.speculative.dspark_cost_profile import (
    DSparkCostProfileConfig,
    DSparkProfileCell,
)
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, PrefillCudaGraphBackend
from tensorrt_llm.tools.verify_costs import EventLeasePool

__all__ = ["DSparkCostProfileEngine"]


def _require_shape_agreement(rows: Sequence[Sequence[int]]) -> None:
    if not rows or any(len(row) != 5 or row[0] != 1 for row in rows):
        raise ValueError("Every profile rank must provide a valid decode-only cell")
    if any(tuple(row[1:]) != tuple(rows[0][1:]) for row in rows):
        raise ValueError("Profile ranks disagree on G, physical K, V or replay ordinal")


def _profile_contract_row(profile: DSparkCostProfileConfig, valid_ladder: bool) -> list[int]:
    digest = hashlib.sha256(profile.model_dump_json().encode()).digest()
    return [
        int(valid_ladder),
        profile.physical_k,
        int(profile.record_confidence_head),
        profile.max_samples,
        *(int.from_bytes(digest[index : index + 8], signed=True) for index in range(0, 32, 8)),
    ]


def _require_profile_agreement(rows: Sequence[Sequence[int]]) -> None:
    if (
        not rows
        or any(len(row) != 8 or row[0] != 1 for row in rows)
        or any(tuple(row[1:]) != tuple(rows[0][1:]) for row in rows)
    ):
        raise ValueError(
            "Every profile rank must agree on the construction-only grid and head mode"
        )


def _valid_previous_state(state: Any, requests: Sequence[Any], physical_k: int) -> bool:
    """Validate original overlap tensor extents without reading device values."""
    if state is None:
        return all(len(request.py_draft_tokens) == physical_k for request in requests)
    if not isinstance(state, SampleStateTensorsSpec):
        return False
    tensors = (state.new_tokens, state.new_tokens_lens, state.next_draft_tokens)
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_cuda
        or value.dtype not in (torch.int32, torch.int64)
        or not value.is_contiguous()
        for value in tensors
    ):
        return False
    tokens, lengths, drafts = tensors
    if (
        tokens.ndim != 3
        or tokens.shape[0] != physical_k + 1
        or tokens.shape[2] != 1
        or lengths.ndim != 1
        or drafts.ndim != 2
        or drafts.shape[1] != physical_k
        or tokens.shape[1] != lengths.shape[0]
        or drafts.shape[0] != lengths.shape[0]
        or len({value.device for value in tensors}) != 1
    ):
        return False
    slots = [getattr(request, "py_batch_idx", None) for request in requests]
    return all(type(slot) is int and 0 <= slot < lengths.shape[0] for slot in slots) and len(
        set(slots)
    ) == len(slots)


class _DSparkProfileWorker(DSv4DSparkWorker):
    def __init__(
        self,
        spec_config: DSparkDecodingConfig,
        mapping: Any,
        *,
        record_head: bool,
        use_separate_draft_kv_cache: bool,
    ) -> None:
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        self.return_confidence = record_head

    def _build_verify_planner(self, draft_model: torch.nn.Module, block_size: int) -> None:
        if block_size != self.spec_config.max_draft_len:
            raise ValueError("Profile head width must match the physical draft block")
        if resolve_confidence_head(draft_model) is None:
            raise ValueError("Head-ON profiling requires the loaded confidence head")
        # Buffers and the original head forward are retained; no serving planner is constructed.
        self.verify_planner = None


class _DSparkProfileGraphs(DSparkCUDAGraphRunner):
    _snapshot_graph_tensor_ptrs = staticmethod(snapshot_metadata_storage)
    _assert_graph_tensor_ptrs_stable = staticmethod(validate_metadata_storage)

    def __init__(self, config: CUDAGraphRunnerConfig, profile: DSparkCostProfileConfig) -> None:
        super().__init__(config)
        self._profile = profile
        self._dspark_trims_submitted_tokens = True
        self.active_profile_cell: DSparkProfileCell | None = None
        self.active_request_ids: tuple[int, ...] = ()
        self._profile_pool = EventLeasePool(lambda: torch.cuda.Event(enable_timing=True))
        self._profile_pool.initialize()
        torch.cuda.current_stream().synchronize()
        self._profile_records: list[dict[str, Any]] = []
        self._profile_ptrs: dict[Any, tuple[dict[str, Any], dict[str, Any]]] = {}

    def capture(self, key: Any, *args: Any, **kwargs: Any) -> Any:
        output = super().capture(key, *args, **kwargs)
        metadata = self.graph_metadata[key]
        self._profile_ptrs[key] = (
            self._snapshot_graph_tensor_ptrs(metadata["attn_metadata"]),
            self._snapshot_graph_tensor_ptrs(metadata["spec_metadata"])
            if metadata["spec_metadata"] is not None
            else {},
        )
        return output

    def maybe_get_cuda_graph(self, batch: ScheduledRequests, *args: Any, **kwargs: Any) -> Any:
        result = super().maybe_get_cuda_graph(batch, *args, **kwargs)
        cell = self.active_profile_cell
        if cell is not None:
            key = result[2]
            if key is None or key not in self.graphs:
                raise RuntimeError("A real profile cell requires its previously captured graph")
            expected_v = None if cell.verifier_budget == 0 else cell.verifier_budget
            if (key.batch_size, key.draft_len, key.ragged_verify_bucket) != (
                cell.graph_batch_size,
                self._profile.physical_k,
                expected_v,
            ):
                raise RuntimeError("Observed replay key does not match the declared profile cell")
            if key.num_contexts != 0 or not key.is_all_greedy_sample:
                raise RuntimeError("The fixed-cell pilot measures greedy decode-only graphs")
        return result

    def replay(
        self,
        key: Any,
        current_inputs: dict[str, Any],
        *,
        position_id_offsets: torch.Tensor | None = None,
    ) -> Any:
        cell = self.active_profile_cell
        if cell is None:
            return super().replay(key, current_inputs, position_id_offsets=position_id_offsets)
        if len(self._profile_records) >= self._profile.max_samples:
            raise RuntimeError("Preallocated profile timing capacity exhausted")
        metadata = self.graph_metadata[key]
        attn_ptrs, spec_ptrs = self._profile_ptrs[key]
        storage_context = {
            "rank": self.config.mapping.rank,
            "graph_key": tuple(
                value.name if isinstance(value, SampleType) else value for value in key
            ),
        }
        self._assert_graph_tensor_ptrs_stable(
            metadata["attn_metadata"], attn_ptrs, context=storage_context
        )
        if metadata["spec_metadata"] is not None:
            self._assert_graph_tensor_ptrs_stable(
                metadata["spec_metadata"], spec_ptrs, context=storage_context
            )
        lease = self._profile_pool.begin()
        output = super().replay(key, current_inputs, position_id_offsets=position_id_offsets)
        self._profile_pool.end(lease)
        self._profile_records.append(
            {
                "sample_index": len(self._profile_records),
                "_event_lease": lease,
                "rank": self.config.mapping.rank,
                "graph_batch_size": cell.graph_batch_size,
                "physical_k": self._profile.physical_k,
                "verifier_budget": cell.verifier_budget,
                "request_ids": self.active_request_ids,
                "graph_key": tuple(
                    value.name if isinstance(value, SampleType) else value for value in key
                ),
            }
        )
        return output

    def complete_sample(self, index: int) -> None:
        """Release events after the original sampler/update completion boundary."""
        record = self._profile_records[index]
        lease = record["_event_lease"]
        elapsed = self._profile_pool.complete(lease)
        del record["_event_lease"]
        record["replay_entry_elapsed_ms"] = elapsed

    def completed_profile_samples(self) -> list[dict[str, Any]]:
        """Return completed raw measurements without waiting or inventing costs."""
        if self._profile_pool.pending:
            raise RuntimeError("Profile generation still owns unfinished event leases")
        if any("replay_entry_elapsed_ms" not in row for row in self._profile_records):
            raise RuntimeError("Profile replay has no original update completion")
        return [dict(row) for row in self._profile_records]


class _DSparkProfileDecoderRunner(DecoderRunner):
    """Offline-only capture/state owner, using the current decoder lifecycle."""

    def __init__(self, model, config, *, profile, **runner_kwargs):
        self._cost_profile = profile
        super().__init__(model, config, **runner_kwargs)

    @property
    def _ragged_verification_enabled(self) -> bool:
        return True

    def _initialize_cuda_graph_runner(self):
        old_worker = self.model.spec_worker
        if type(old_worker) is not DSv4DSparkWorker or old_worker._win_inited:
            raise RuntimeError("Profile worker must be installed before its first initialization")
        if old_worker.staged_confidence_buffer() is not None or self.attn_metadata is not None:
            raise RuntimeError(
                "Profile construction cannot replace initialized state or graph addresses"
            )
        matches = [index for index, entry in enumerate(self.model.epilogue) if entry is old_worker]
        if len(matches) != 1:
            raise RuntimeError("Original speculative worker must have one epilogue owner")
        worker = _DSparkProfileWorker(
            self._config.spec_config,
            self.mapping,
            record_head=self._cost_profile.record_confidence_head,
            use_separate_draft_kv_cache=self.model.use_separate_draft_kv_cache,
        )
        worker.set_draft_model(self.model.draft_model)
        self.model.spec_worker = worker
        self.model.epilogue[matches[0]] = worker
        self._dspark_trims_submitted_tokens = True
        self.previous_verify_lens = []
        return super()._initialize_cuda_graph_runner()

    def _create_cuda_graph_runner(self, config: CUDAGraphRunnerConfig) -> CUDAGraphRunner:
        return _DSparkProfileGraphs(config, self._cost_profile)

    def _get_graphs_to_capture(self, cuda_graph_batch_sizes: list[int]) -> list[tuple[int, ...]]:
        return self._cost_profile.graph_shapes(cuda_graph_batch_sizes)

    def ragged_verify_token_buckets(self, padded_bs: int) -> list[int]:
        return sorted(
            cell.verifier_budget
            for cell in self._cost_profile.cells
            if cell.graph_batch_size == padded_bs and cell.verifier_budget > 0
        )


class DSparkCostProfileEngine(PyTorchModelEngine):
    """Explicit offline engine; the ordinary public serving factories cannot create it.

    Args:
        profile: Validated construction-only cell grid.
        engine_kwargs: Original PyTorchModelEngine construction arguments. The
            DSpark serving configuration must keep confidence scheduling off.
    """

    def __init__(self, *, profile: DSparkCostProfileConfig, **engine_kwargs: Any) -> None:
        self._cost_profile = profile
        self._profile_call_active = False
        spec = engine_kwargs.get("spec_config")
        if not isinstance(spec, DSparkDecodingConfig):
            raise TypeError("Profile-only construction requires DSparkDecodingConfig")
        if spec.max_draft_len != profile.physical_k or not spec.draft_is_embedded_in_target:
            raise ValueError("Profile-only construction requires embedded DSpark with matching K")
        if (
            spec.enable_confidence_scheduling
            or spec.enable_fused_confidence_scheduler
            or getattr(spec, "confidence_mixed_evidence", None) is not None
            or spec.draft_len_schedule is not None
        ):
            raise ValueError("Profile-only construction cannot enable serving confidence policy")
        if any(
            getattr(spec, name, None)
            for name in (
                "confidence_sps_table_path",
                "confidence_sps_live_fingerprint_path",
                "confidence_admission_receipt_path",
                "confidence_admission_receipt_sha256",
                "confidence_sts_path",
            )
        ):
            raise ValueError(
                "Profile-only construction does not consume serving admission artifacts"
            )
        if os.environ.get("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"):
            raise ValueError("Fixed-cell costs require real acceptance, not forced acceptance")
        super().__init__(**engine_kwargs)

    def _initialize_decoder_runner(self, runner_cls, runner_config=None, **runner_kwargs):
        valid_ladder = True
        try:
            self._cost_profile.validate_graph_ladder(self._cuda_graph_batch_sizes)
        except ValueError:
            valid_ladder = False
        if self.mapping.tp_size > 1 and self.dist is None:
            raise ValueError("Multi-rank profiling requires the original distributed communicator")
        row = _profile_contract_row(self._cost_profile, valid_ladder)
        rows = self.dist.tp_allgather_int64(row).tolist() if self.mapping.tp_size > 1 else [row]
        _require_profile_agreement(rows)
        if (
            runner_cls is not DecoderRunner
            or self.is_multimodal
            or self._is_encoder_decoder_model()
            or self.max_beam_width != 1
            or self.mapping.pp_size != 1
            or self.mapping.cp_size != 1
            or self.cuda_graph_config is None
            or not self._cuda_graph_padding_enabled
            or self.prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
            or self.llm_args.lora_config is not None
            or self.llm_args.enable_in_graph_sampling
        ):
            raise ValueError("Unsupported profile-only model, topology or graph configuration")
        return super()._initialize_decoder_runner(
            _DSparkProfileDecoderRunner,
            runner_config,
            profile=self._cost_profile,
            **runner_kwargs,
        )

    def _set_warmup_ragged_windows(self, batch, verify_bucket, draft_len):
        if not isinstance(self._runner, _DSparkProfileDecoderRunner):
            raise RuntimeError("Prescribed windows require the offline decoder owner")
        return self._runner._set_warmup_ragged_windows(batch, verify_bucket, draft_len)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Profile-only engines cannot execute the ordinary serving forward API")

    def forward_profile_prefill(
        self, batch: ScheduledRequests, resource_manager: ResourceManager
    ) -> Any:
        """Initialize genuine request KV using the original unmeasured context forward."""
        valid = (
            batch.num_context_requests > 0
            and not batch.num_generation_requests
            and not batch.num_encoder_requests
            and not self._profile_call_active
        )
        row = [int(valid), -1, self._cost_profile.physical_k, 0, -1]
        rows = self.dist.tp_allgather_int64(row).tolist() if self.mapping.tp_size > 1 else [row]
        _require_shape_agreement(rows)
        return super().forward(batch, resource_manager=resource_manager)

    def forward_profile_cell(
        self,
        cell: DSparkProfileCell,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        new_tensors_device: SampleStateTensors | None = None,
        cache_indirection_buffer: torch.Tensor | None = None,
    ) -> Any:
        """Borrow real scheduled state and run one declared cell through the original forward.

        The caller retains ownership of request/sampler/KV updates and must
        advance them using the original executor lifecycle before the next call.
        """
        requests = batch.generation_requests
        local_valid = (
            cell in self._cost_profile.cells
            and not self._profile_call_active
            and batch.num_context_requests == 0
            and batch.num_encoder_requests == 0
            and len(requests) == cell.graph_batch_size
            and len({request.py_request_id for request in requests}) == len(requests)
            and all(
                not request.is_dummy
                and request.py_beam_width == 1
                and not request.py_is_first_draft
                and request.py_seq_slot is not None
                for request in requests
            )
            and self.runtime_draft_len == self._cost_profile.physical_k
            and _valid_previous_state(new_tensors_device, requests, self._cost_profile.physical_k)
        )
        row = [
            int(local_valid),
            cell.graph_batch_size,
            self._cost_profile.physical_k,
            cell.verifier_budget,
            len(self.cuda_graph_runner._profile_records),
        ]
        rows = self.dist.tp_allgather_int64(row).tolist() if self.mapping.tp_size > 1 else [row]
        _require_shape_agreement(rows)
        graph_runner = self.cuda_graph_runner
        graph_runner.active_profile_cell = cell
        graph_runner.active_request_ids = tuple(request.py_request_id for request in requests)
        self._profile_call_active = True
        try:
            if cell.verifier_budget:
                self._set_warmup_ragged_windows(
                    batch, cell.verifier_budget, self._cost_profile.physical_k
                )
            else:
                for request in requests:
                    request.py_verify_len = None
                graph_runner.agreed_ragged_bucket = None
            before = len(graph_runner._profile_records)
            output = super().forward(
                batch,
                resource_manager=resource_manager,
                new_tensors_device=new_tensors_device,
                cache_indirection_buffer=cache_indirection_buffer,
            )
            if len(graph_runner._profile_records) != before + 1:
                raise RuntimeError(
                    "Profile forward did not execute exactly one original graph replay"
                )
            return output
        finally:
            graph_runner.active_profile_cell = None
            graph_runner.active_request_ids = ()
            self._profile_call_active = False

    def profile_observations(self) -> dict[str, Any]:
        """Return raw observations, never an admitted table or enabled-performance result."""
        return {
            "schema": "dspark-cost-profile-observation-v1",
            "profile_only": True,
            "physical_k": self._cost_profile.physical_k,
            "record_confidence_head": self._cost_profile.record_confidence_head,
            "timing_scope": "original-graph-replay-entry-including-static-copies",
            "samples": self.cuda_graph_runner.completed_profile_samples(),
            "full_policy_h": None,
            "production_sps_table_eligible": False,
            "production_admission_qualified": False,
        }
