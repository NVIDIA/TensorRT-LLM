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
"""Eligibility policy of the indexer_topk indexer decode package (the index_selection contract).

This module imports only the standard library so the decision table can be
unit-tested without torch or the compiled extensions. Every input is a plain
host value that the caller reads from the native ``Indexer``, its ``TopK``
module, the DSA metadata and the tensors' dtypes/shapes before any indexer_topk kernel
runs; an ineligible call returns a reason and the caller keeps the native
path.

indexer_topk engages only inside the envelope that its GPU tests verify (sm_103, the
GLM-5.2 indexer shape, 64-token FP8 pages, ``next_n = 1 + K`` with K = 5,
up to four generation requests, the native self-sampling GVR selection,
indexer ``max_seq_len`` up to ``2**20 + 14``). Every precondition of the indexer_topk
kernel launch that the caller can observe on the host is checked here, so an
eligible call cannot fail after the decision.
"""

from dataclasses import dataclass
from typing import Tuple

# GPU-verified support envelope (the indexer_topk kernel tests on sm_103).
SUPPORTED_SM_VERSIONS = (103,)
SUPPORTED_NUM_HEADS = 32
SUPPORTED_HEAD_DIM = 128
SUPPORTED_TOKENS_PER_BLOCK = 64
# 1 target row + K = 5 MTP draft rows per request.
SUPPORTED_NEXT_N = 6
MAX_GENERATIONS = 4
# The indexer max_seq_len is the KV-cache manager's max_seq_len: the engine's
# net max_seq_len plus (tokens_per_gen_step - 1) with the overlap scheduler,
# plus (max_draft_len - 1) extra one-model KV tokens, plus
# (tokens_per_gen_step - 1). For K = 5 that slack is 5 + 4 + 5 = 14, so a 2**20
# net envelope serves an indexer max_seq_len of 2**20 + 14. Live KV lengths
# stay at or below it; the logits arena and the native TopK envelope are sized
# from it.
MTP_K5_MAX_SEQ_LEN_SLACK = 14
MAX_INDEXER_SEQ_LEN = (1 << 20) + MTP_K5_MAX_SEQ_LEN_SLACK
# The logits kernel computes 128-token tiles and loads both page IDs of every
# tile it computes, so a request whose last tile covers one page reads the
# table entry after that page. Requiring the table width to be a multiple of
# the pages per tile keeps that read inside the request's row for every KV
# length the row can describe.
KV_TILE_TOKENS = 128
SUPPORTED_INDEX_TOPK = 2048
SUPPORTED_COMPRESS_RATIO = 1
SUPPORTED_TOPK_DECODE_IMPL = "cute_dsl_gvr"

# Tensor formats of the indexer_topk launch: contiguous FP8 queries ``[tokens, H, D]``,
# contiguous fp32 per-head weights ``[tokens, H]``, fused uint8 K pages
# ``[pages, tokens_per_block, 1, D + 4]`` (D FP8 bytes plus one fp32 scale).
QUERY_DTYPE = "torch.float8_e4m3fn"
WEIGHTS_DTYPE = "torch.float32"
K_CACHE_DTYPE = "torch.uint8"
K_CACHE_SCALE_BYTES = 4


@dataclass(frozen=True)
class Decision:
    """Outcome of an indexer_topk eligibility check.

    Attributes:
        eligible: True when indexer_topk may replace the native logits + select region.
        reason: Short reason for a native fallback; empty when eligible.
    """

    eligible: bool
    reason: str = ""


def decide(
    *,
    sm_version: int,
    cute_dsl_available: bool,
    native_uses_cute_dsl_logits: bool,
    use_fp4_indexer: bool,
    use_gvr_emission: bool,
    needs_gvr_prior: bool,
    is_ragged_verify: bool,
    expanded_mtp_buffers: bool,
    in_mtp_draft_loop: bool,
    attention_dp: bool,
    topk_decode_impl: str,
    topk_self_sampling: bool,
    index_topk: int,
    compress_ratio: int,
    num_heads: int,
    head_dim: int,
    query_dtype: str,
    query_shape: Tuple[int, ...],
    query_contiguous: bool,
    weights_dtype: str,
    weights_shape: Tuple[int, ...],
    weights_contiguous: bool,
    k_cache_dtype: str,
    k_cache_shape: Tuple[int, ...],
    block_table_cols: int,
    max_seq_len: int,
    num_contexts: int,
    token_offset: int,
    num_gen_tokens: int,
    num_generations: int,
    uniform_verify_width: bool,
) -> Decision:
    """Decide whether indexer_topk may run the GEN indexer logits + top-k region.

    Args:
        sm_version: Compute capability as an integer, e.g. 103.
        cute_dsl_available: Whether the CUTLASS DSL runtime is importable.
        native_uses_cute_dsl_logits: The indexer's
            ``use_cute_dsl_paged_mqa_logits`` flag.
        use_fp4_indexer: Whether the indexer K cache is MXFP4.
        use_gvr_emission: Whether the temporal GVR emission tier is enabled.
        needs_gvr_prior: Whether the decode top-k consumes a previous-step
            prior (temporal GVR engine).
        is_ragged_verify: Whether generation rows have per-request widths.
        expanded_mtp_buffers: The metadata ``use_expanded_buffers_for_mtp``
            flag; when set, the per-request schedule indexer_topk consumes is not built.
        in_mtp_draft_loop: Whether this forward is a draft step of the MTP loop.
        attention_dp: Whether attention data parallelism is enabled.
        topk_decode_impl: Value of the layer ``TopK.decode_implementation``.
        topk_self_sampling: The layer ``TopK.gvr_self_sampling`` flag.
        index_topk: Selected IDs per row (``TopK.top_k``).
        compress_ratio: Index compression ratio (``TopK.compress_ratio``).
        num_heads: Indexer heads ``H``.
        head_dim: Indexer head dimension ``D``.
        query_dtype: ``str(dtype)`` of the projected indexer queries.
        query_shape: Shape of the projected queries ``[tokens, H, D]``.
        query_contiguous: Whether the projected queries are contiguous.
        weights_dtype: ``str(dtype)`` of the per-head weights.
        weights_shape: Shape of the per-head weights ``[tokens, H]``.
        weights_contiguous: Whether the per-head weights are contiguous.
        k_cache_dtype: ``str(dtype)`` of the fused indexer K page view.
        k_cache_shape: Shape of the fused K page view
            ``[pages, tokens_per_block, 1, D + 4]``.
        block_table_cols: Columns (pages per request) of the indexer
            block table ``indexer_k_cache_block_offsets``.
        max_seq_len: Capture-stable indexer max sequence length.
        num_contexts: Context requests in this forward.
        token_offset: First generation row in the query/weight tensors.
        num_gen_tokens: Generation query rows in this call.
        num_generations: Generation requests in this call.
        uniform_verify_width: Whether every generation request has the same
            number of query rows (from the host ``seq_lens``).

    Returns:
        The decision with a fallback reason when ineligible.
    """
    if sm_version not in SUPPORTED_SM_VERSIONS:
        return Decision(False, f"sm{sm_version} unsupported")
    if not cute_dsl_available:
        return Decision(False, "CUTLASS DSL unavailable")
    if not native_uses_cute_dsl_logits:
        return Decision(False, "native path is not CuTe-DSL paged-MQA logits")
    if use_fp4_indexer:
        return Decision(False, "FP4 indexer keys")
    if use_gvr_emission:
        return Decision(False, "GVR emission tier enabled")
    if needs_gvr_prior:
        return Decision(False, "temporal GVR prior in use")
    if is_ragged_verify:
        return Decision(False, "ragged verification")
    if expanded_mtp_buffers:
        return Decision(False, "expanded MTP buffers (no per-request schedule)")
    if in_mtp_draft_loop:
        return Decision(False, "MTP draft step (target layers only)")
    if attention_dp:
        return Decision(False, "attention data parallelism")
    if topk_decode_impl != SUPPORTED_TOPK_DECODE_IMPL or not topk_self_sampling:
        return Decision(
            False, f"top-k engine {topk_decode_impl} (self_sampling={topk_self_sampling})"
        )
    if index_topk != SUPPORTED_INDEX_TOPK:
        return Decision(False, f"index_topk {index_topk} != {SUPPORTED_INDEX_TOPK}")
    if compress_ratio != SUPPORTED_COMPRESS_RATIO:
        return Decision(False, f"index compress_ratio {compress_ratio}")
    if num_heads != SUPPORTED_NUM_HEADS:
        return Decision(False, f"num_heads {num_heads} != {SUPPORTED_NUM_HEADS}")
    if head_dim != SUPPORTED_HEAD_DIM:
        return Decision(False, f"head_dim {head_dim} != {SUPPORTED_HEAD_DIM}")
    k_cache_shape = tuple(k_cache_shape)
    if len(k_cache_shape) != 4 or k_cache_shape[2] != 1:
        return Decision(False, f"K cache view shape {k_cache_shape}")
    if k_cache_shape[1] != SUPPORTED_TOKENS_PER_BLOCK:
        return Decision(
            False, f"tokens_per_block {k_cache_shape[1]} != {SUPPORTED_TOKENS_PER_BLOCK}"
        )
    if k_cache_dtype != K_CACHE_DTYPE or k_cache_shape[3] != head_dim + K_CACHE_SCALE_BYTES:
        return Decision(
            False, f"K cache layout {k_cache_dtype} x {k_cache_shape[3]} bytes per token"
        )
    if query_dtype != QUERY_DTYPE:
        return Decision(False, f"query dtype {query_dtype}")
    if weights_dtype != WEIGHTS_DTYPE:
        return Decision(False, f"weights dtype {weights_dtype}")
    query_shape, weights_shape = tuple(query_shape), tuple(weights_shape)
    if (
        len(query_shape) != 3
        or query_shape[1:] != (num_heads, head_dim)
        or len(weights_shape) != 2
        or weights_shape[1] != num_heads
        or not query_contiguous
        or not weights_contiguous
    ):
        return Decision(False, f"query {query_shape} / weights {weights_shape} layout")
    if max_seq_len <= 0 or max_seq_len > MAX_INDEXER_SEQ_LEN:
        return Decision(False, f"indexer max_seq_len {max_seq_len} > {MAX_INDEXER_SEQ_LEN}")
    pages_per_tile = KV_TILE_TOKENS // k_cache_shape[1]
    if block_table_cols <= 0 or block_table_cols % pages_per_tile != 0:
        return Decision(
            False,
            f"block table width {block_table_cols} is not a multiple of {pages_per_tile} pages",
        )
    if num_contexts != 0:
        return Decision(False, "context requests in the same forward")
    if num_generations <= 0 or num_gen_tokens <= 0:
        return Decision(False, "no generation rows")
    if num_generations > MAX_GENERATIONS:
        return Decision(False, f"{num_generations} generation requests > {MAX_GENERATIONS}")
    if token_offset < 0 or token_offset + num_gen_tokens > min(query_shape[0], weights_shape[0]):
        return Decision(False, "generation rows exceed the query/weight tensors")
    if num_gen_tokens % num_generations != 0 or not uniform_verify_width:
        return Decision(False, "non-uniform verify width")
    next_n = num_gen_tokens // num_generations
    if next_n != SUPPORTED_NEXT_N:
        return Decision(False, f"next_n {next_n} != {SUPPORTED_NEXT_N}")
    return Decision(True)
