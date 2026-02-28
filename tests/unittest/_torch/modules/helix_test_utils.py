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
"""Shared utilities for Helix context-parallelism unit tests (MHA & MLA)."""

import traceback

import torch
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

# Convenient aliases for the two KV-cache types used by MHA and MLA tests.
CACHE_TYPE_SELF = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
CACHE_TYPE_SELFKONLY = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def copy_weights_for_cp(weights, param_name, dim, rank, world_size):
    """Slice a weight tensor along *dim* for context-parallel rank."""
    w_dim_per_rank = weights[param_name].shape[dim] // world_size
    w_dim_start = rank * w_dim_per_rank
    w_dim_end = w_dim_start + w_dim_per_rank
    slices = [slice(None)] * weights[param_name].ndim
    slices[dim] = slice(w_dim_start, w_dim_end)
    weights[param_name] = weights[param_name][slices]


# ---------------------------------------------------------------------------
# Input splitting
# ---------------------------------------------------------------------------


def split_inputs_for_rank(input_ctx, position_ids_ctx, scenario, rank, world_size):
    """Split context inputs into per-rank chunks for CP.

    Returns ``(input_ctx_rank, position_ids_ctx_rank)``.
    """
    ctx_len_per_gpu = scenario.ctx_len // world_size
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len, scenario.hidden_size)
    input_ctx_rank = input_ctx_bs[:, rank * ctx_len_per_gpu : (rank + 1) * ctx_len_per_gpu, :]
    input_ctx_rank = input_ctx_rank.reshape(
        scenario.batch * ctx_len_per_gpu, scenario.hidden_size
    ).contiguous()
    position_ids_ctx_bs = position_ids_ctx.view(scenario.batch, scenario.ctx_len)
    position_ids_ctx_rank = position_ids_ctx_bs[
        :, rank * ctx_len_per_gpu : (rank + 1) * ctx_len_per_gpu
    ]
    position_ids_ctx_rank = position_ids_ctx_rank.reshape(
        scenario.batch * ctx_len_per_gpu
    ).contiguous()
    return input_ctx_rank, position_ids_ctx_rank


# ---------------------------------------------------------------------------
# Helix metadata helpers
# ---------------------------------------------------------------------------


def activate_all_ranks_for_context(attn_metadata, position_ids):
    """Override helix metadata so all ranks participate in the context step."""
    attn_metadata.helix_position_offsets = position_ids
    if (
        hasattr(attn_metadata, "helix_is_inactive_rank")
        and attn_metadata.helix_is_inactive_rank is not None
    ):
        attn_metadata.helix_is_inactive_rank.fill_(False)
    if (
        hasattr(attn_metadata, "helix_is_inactive_rank_cpu")
        and attn_metadata.helix_is_inactive_rank_cpu is not None
    ):
        attn_metadata.helix_is_inactive_rank_cpu.fill_(False)


def create_helix_gen_metadata(
    batch,
    ctx_len_per_gpu,
    kv_cache_manager,
    helix_is_inactive_rank,
    position_ids_gen,
    *,
    enable_context_mla_with_cached_kv=False,
):
    """Create generation-phase attention metadata with Helix fields.

    The KV cache is assumed to hold *ctx_len_per_gpu* tokens per request (one
    generation step after the context phase).
    """
    kwargs = {}
    if enable_context_mla_with_cached_kv:
        kwargs["enable_context_mla_with_cached_kv"] = True
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([1] * batch, dtype=torch.int),
        request_ids=list(range(batch)),
        max_num_requests=batch,
        num_contexts=0,
        prompt_lens=[ctx_len_per_gpu] * batch,
        max_num_tokens=ctx_len_per_gpu,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[ctx_len_per_gpu] * batch,
        ),
        **kwargs,
    )
    attn_metadata.enable_helix = True
    attn_metadata.helix_is_inactive_rank = torch.tensor(
        helix_is_inactive_rank, dtype=torch.bool, device="cuda"
    )
    attn_metadata.helix_is_inactive_rank_cpu = attn_metadata.helix_is_inactive_rank.to(
        device="cpu"
    ).pin_memory()
    attn_metadata.helix_position_offsets = torch.tensor(
        position_ids_gen, dtype=torch.int, device="cuda"
    )
    attn_metadata.helix_position_offsets_cpu = attn_metadata.helix_position_offsets.to(
        device="cpu"
    ).pin_memory()
    attn_metadata.prepare()
    return attn_metadata


# ---------------------------------------------------------------------------
# KV-cache / metadata setup
# ---------------------------------------------------------------------------


def setup_kv_and_metadata(
    scenario,
    mapping: Mapping,
    *,
    cache_type,
    num_kv_heads: int,
    head_dim: int,
):
    """Create a :class:`KVCacheManager` and context-phase attention metadata.

    The caller supplies the KV-cache geometry that differs between MHA and MLA:
    *cache_type*, *num_kv_heads*, and *head_dim*.  Everything else is read from
    *scenario* (which must expose ``ctx_len``, ``batch``,
    ``kv_cache_tokens_per_block``, ``num_layers``, and ``kv_cache_dtype``).

    The KV cache is sized for one generation step after the context phase.
    """
    n_gpu = mapping.world_size
    assert scenario.ctx_len % n_gpu == 0
    ctx_len_per_gpu = scenario.ctx_len // n_gpu
    max_tokens = (
        (ctx_len_per_gpu + 1 + scenario.kv_cache_tokens_per_block - 1)
        // scenario.kv_cache_tokens_per_block
        * scenario.kv_cache_tokens_per_block
        * scenario.batch
    )
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        cache_type,
        num_layers=scenario.num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=scenario.kv_cache_tokens_per_block,
        max_seq_len=ctx_len_per_gpu + 1,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    for req_id in range(scenario.batch):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=1,
            input_tokens=[1] * ctx_len_per_gpu,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.is_dummy_request = True
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len_per_gpu, beam_width, req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        req.prompt_len = ctx_len_per_gpu
        req.py_prompt_len = req.prompt_len
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([ctx_len_per_gpu] * scenario.batch, dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=scenario.batch,
        prompt_lens=[ctx_len_per_gpu] * scenario.batch,
        max_num_tokens=ctx_len_per_gpu,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in range(scenario.batch)],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()
    return kv_cache_manager, attn_metadata


# ---------------------------------------------------------------------------
# Error reporting / comparison
# ---------------------------------------------------------------------------


def error_report(output, ref_output, atol, rtol, prefix):
    """Print per-element error stats and return the number of mismatches."""
    err = torch.abs(output - ref_output)
    ref_abs = torch.abs(ref_output)
    ref_abs[ref_abs == 0] = torch.finfo(ref_abs.dtype).smallest_normal
    rel_err = err / ref_abs
    max_err_idx = torch.unravel_index(torch.argmax(err - atol - rtol * ref_abs), err.shape)
    values_err = (output[max_err_idx].item(), ref_output[max_err_idx].item())
    max_abs_err_idx = torch.unravel_index(torch.argmax(err), err.shape)
    values_abs = (output[max_abs_err_idx].item(), ref_output[max_abs_err_idx].item())
    max_rel_err_idx = torch.unravel_index(torch.argmax(rel_err), rel_err.shape)
    values_rel = (output[max_rel_err_idx].item(), ref_output[max_rel_err_idx].item())
    max_abs_err = err[max_abs_err_idx].item()
    max_rel_err = rel_err[max_rel_err_idx].item()
    max_err_idx = [x.item() for x in max_err_idx]
    max_abs_err_idx = [x.item() for x in max_abs_err_idx]
    max_rel_err_idx = [x.item() for x in max_rel_err_idx]
    isclose = err < atol + rtol * ref_abs
    n_error = (~isclose).sum().item()
    print(
        f"{prefix}: {n_error} errors, max error index: {max_err_idx} "
        f"(test/ref values: {values_err}), max abs error index: "
        f"{max_abs_err_idx} "
        f"(test/ref values: {values_abs}, err: {max_abs_err}), max rel error "
        f"index: {max_rel_err_idx} "
        f"(test/ref values: {values_rel}, err: {max_rel_err}), atol: {atol}, "
        f"rtol: {rtol}"
    )
    return n_error


def compute_mismatch_ratio(output, ref_output, atol, rtol, rank, world_size):
    """Compare *output* vs *ref_output* per batch element, return mismatch ratio."""
    mismatch_count = 0
    for b in range(output.shape[0]):
        mismatch_count += error_report(
            output[b], ref_output[b], atol, rtol, f"Rank {rank} {world_size}-GPU batch {b}"
        )
    ratio_mismatch = mismatch_count / output.numel()
    print(
        f"Rank {rank} {world_size}-GPU: "
        f"{mismatch_count}/{output.numel()} mismatches: {ratio_mismatch}"
    )
    return ratio_mismatch


# ---------------------------------------------------------------------------
# MPI orchestration
# ---------------------------------------------------------------------------


def run_single_rank(func, *args, **kwargs):
    """Worker entry-point: set the CUDA device for *rank* and call *func*."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    print(f"rank {rank} starting")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred. Original traceback is\n{tb}\n")


def parse_comms_medium(comms_medium: str):
    """Return ``(use_nccl_for_alltoall, fifo_version)`` for *comms_medium*."""
    if comms_medium == "nccl":
        return True, 2
    elif comms_medium == "fifo_v1":
        return False, 1
    elif comms_medium == "fifo_v2":
        return False, 2
    else:
        raise ValueError(f"Unknown comms_medium: {comms_medium}")


def run_helix_test(
    full_test_func,
    scenario,
    comms_medium: str,
    world_size: int = 2,
    max_mismatch_ratio: float = 0.02,
):
    """Parse *comms_medium*, launch MPI workers, and assert mismatch ratio."""
    use_nccl_for_alltoall, fifo_version = parse_comms_medium(comms_medium)
    print(f"Testing with comms_medium={comms_medium}.")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(
                *[
                    (
                        full_test_func,
                        world_size,
                        scenario,
                        use_nccl_for_alltoall,
                        fifo_version,
                    )
                ]
                * world_size
            ),
        )
        for ratio_mismatch in results:
            assert ratio_mismatch <= max_mismatch_ratio
