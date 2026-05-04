# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

# isort: off
# needed before trying to import bindings to load tensorrt_libs
import tensorrt as trt  # noqa
# isort: on

from tensorrt_llm.bindings import executor as tllme


class DisaggScheduleStyle(IntEnum):
    CONTEXT_FIRST = 0
    GENERATION_FIRST = 1


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
        draft_tokens (List[int]): The draft tokens of the generation request
        disagg_request_id (int): The disaggregated request id, if set, both context and generation requests will use it
         as underlying request id.
        first_gen_log_probs (List): The logprobs for first_gen_tokens, produced during prefill.
         Each entry is a list (one per beam) of TokenLogprobs (list of dict[int, Logprob]).
        first_gen_logits (List): The generation logits for first_gen_tokens, produced during prefill.
         Each entry is a torch.Tensor of shape [num_tokens, vocab_size] (one per beam/sequence).

        multimodal_embedding_handles (List[Dict[str, Any]]): The resulting multimodal embedding handles from ViT.
        multimodal_hashes (List[List[int]]): The multimodal hashes of each multimodal item in the request.
         Requires multimodal_item_runs so cache keys know the exact prompt coverage.
        multimodal_item_runs (List[List[Tuple[int, int, List[int]]]]): Exact prompt token runs covered by each
         multimodal item. The third tuple element lists local non-embed offsets for that run.
    """

    request_type: Optional[str] = None
    # P-D Disaggregated Params
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None
    # If disagg_request_id is set, both context and generation requests will use it as underlying request id.
    disagg_request_id: Optional[int] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None

    # E-P Disaggregated Params
    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = (
        None  # multimodal embedding handles should be a list of cudaIPC handles for each mm_embedding
    )
    multimodal_hashes: Optional[List[List[int]]] = (
        None  # user provided mm hashes should be a list of 8 integers
    )
    multimodal_item_runs: Optional[List[List[Tuple[int, int, List[int]]]]] = None
    mrope_position_ids_handle: Optional[Dict[str, Any]] = None
    mrope_position_deltas_handle: Optional[Dict[str, Any]] = None

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        # Prefer disagg_request_id over ctx_request_id
        request_id = (
            self.disagg_request_id if self.disagg_request_id is not None else self.ctx_request_id
        )
        # `first_gen_tokens` is now required by bindings and cannot be None.
        first_gen_tokens = self.first_gen_tokens if self.first_gen_tokens is not None else []
        return tllme.ContextPhaseParams(
            first_gen_tokens,
            request_id,
            self.opaque_state,
            self.draft_tokens,
            self.ctx_dp_rank,
            self.ctx_info_endpoint,
        )

    def get_request_type(self) -> tllme.RequestType:
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )

    def __post_init__(self):
        if self.request_type is not None:
            self.request_type = self.request_type.lower()
            if self.request_type not in [
                "context_only",
                "generation_only",
                "context_and_generation",
            ]:
                raise ValueError(
                    f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                    "context_and_generation"
                )
        if self.multimodal_embedding_handles is not None:
            if self.multimodal_hashes is not None:
                assert len(self.multimodal_embedding_handles) == len(self.multimodal_hashes), (
                    "multimodal_embedding_handles and multimodal_hashes must have the same length"
                )

        if self.multimodal_hashes is not None:
            for mm_hash in self.multimodal_hashes:
                assert isinstance(mm_hash, list), "mm_hash must be a list"
                assert len(mm_hash) == 8, "mm_hash must be a list of 8 integers"
                assert all(isinstance(x, int) for x in mm_hash), "mm_hash must contain integers"
            assert self.multimodal_item_runs is not None, (
                "multimodal_hashes requires multimodal_item_runs"
            )

        if self.multimodal_item_runs is not None:
            assert self.multimodal_hashes is not None, (
                "multimodal_item_runs requires multimodal_hashes"
            )
            assert len(self.multimodal_item_runs) == len(self.multimodal_hashes), (
                "multimodal_item_runs and multimodal_hashes must have the same length"
            )
            occupied_runs: List[Tuple[int, int, int, int]] = []
            for item_idx, item_runs in enumerate(self.multimodal_item_runs):
                assert isinstance(item_runs, list), "multimodal_item_runs item must be a list"
                assert len(item_runs) > 0, "multimodal_item_runs item must not be empty"
                previous_end = None
                for run_idx, run in enumerate(item_runs):
                    assert isinstance(run, (list, tuple)) and len(run) == 3, (
                        "multimodal_item_runs entries must be "
                        "(prompt_start, run_length, non_embed_offsets) tuples"
                    )
                    start, length, non_embed_offsets = run
                    assert isinstance(start, int) and isinstance(length, int), (
                        "multimodal_item_runs must contain integers"
                    )
                    assert start >= 0, "multimodal_item_runs must contain non-negative positions"
                    assert length > 0, "multimodal_item_runs must contain positive lengths"
                    assert isinstance(non_embed_offsets, list), (
                        "multimodal_item_runs non-embed offsets must be a list"
                    )
                    assert all(isinstance(offset, int) for offset in non_embed_offsets), (
                        "multimodal_item_runs non-embed offsets must contain integers"
                    )
                    assert non_embed_offsets == sorted(set(non_embed_offsets)), (
                        "multimodal_item_runs non-embed offsets must be ordered and unique"
                    )
                    assert all(0 <= offset < length for offset in non_embed_offsets), (
                        "multimodal_item_runs non-embed offsets must be within the run"
                    )
                    assert previous_end is None or start >= previous_end, (
                        "multimodal_item_runs must be ordered and non-overlapping"
                    )
                    end = start + length
                    for prev_item_idx, prev_run_idx, prev_start, prev_end in occupied_runs:
                        assert start >= prev_end or prev_start >= end, (
                            "multimodal_item_runs must be globally non-overlapping "
                            f"but [{item_idx}][{run_idx}] overlaps "
                            f"[{prev_item_idx}][{prev_run_idx}]"
                        )
                    previous_end = start + length
                    occupied_runs.append((item_idx, run_idx, start, end))
