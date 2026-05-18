# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Dynamic Tree Operations for EAGLE3 Speculative Decoding

This module provides high-performance CUDA kernel wrappers for building and verifying
dynamic tree structures used in EAGLE3 speculative decoding. It integrates SGLang's
optimized CUDA kernels into TensorRT-LLM's PyTorch backend.

Key Features:
- Efficient tree construction from layer-local parent indices
- Greedy tree verification with parallel traversal
- Buffer pre-allocation and reuse for minimal runtime overhead
"""

import torch


class DynamicTreeOpsConverter:
    """
    Converter for dynamic tree operations using CUDA kernels.

    This class handles data format conversion and CUDA kernel invocation for
    building and verifying dynamic trees in EAGLE3 speculative decoding.

    Args:
        dynamic_tree_max_topK: Maximum top-K tokens per node.
        max_draft_len: Maximum draft length (tree depth).
        max_total_draft_tokens: Total number of draft tokens.
        max_batch_size: Maximum batch size.
        device: CUDA device.
    """

    def __init__(
        self,
        dynamic_tree_max_topK: int,
        max_draft_len: int,
        max_total_draft_tokens: int,
        max_batch_size: int,
        device: torch.device,
    ):
        """Allocate reusable CUDA buffers for dynamic-tree build/verify ops."""
        self.K = dynamic_tree_max_topK
        self.depth = max_draft_len

        # Pre-allocated output buffers for verify_dynamic_tree_greedy_out_op
        N = max_total_draft_tokens + 1  # tokens_per_gen_step (includes root)
        max_path_len = max_draft_len + 1
        self._verify_predicts_buf = torch.zeros(
            max_batch_size * N, dtype=torch.int64, device=device
        )
        self._verify_accept_index_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int64, device=device
        )
        self._verify_accept_token_num_buf = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._verify_accept_token_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int64, device=device
        )

        # Pre-allocated output buffers for verify_dynamic_tree_rejection_out
        self._rej_accept_index_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int64, device=device
        )
        self._rej_accept_token_num_buf = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._rej_accept_token_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int64, device=device
        )
        self._rej_seed_buf = torch.zeros(1, dtype=torch.int64, device=device)
        self._rej_offset_buf = torch.zeros(1, dtype=torch.int64, device=device)

    def _get_rejection_rng_tensor(
        self,
        value: int | torch.Tensor,
        buffer: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Normalize a rejection RNG input to a one-element int64 CUDA tensor."""
        if isinstance(value, int):
            buffer.fill_(value)
            return buffer
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be an int or torch.Tensor, got {type(value)!r}")
        if value.dtype != torch.int64:
            raise TypeError(f"{name} must be int64 tensor, got {value.dtype}")
        if value.numel() != 1:
            raise ValueError(f"{name} tensor must have exactly one element, got {value.numel()}")
        if value.device != buffer.device:
            raise ValueError(f"{name} tensor must be on device {buffer.device}, got {value.device}")
        return value.reshape(-1)

    def build_dynamic_tree(
        self,
        history_draft_tokens_parent_buffer: torch.Tensor,
        topk_score_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        use_packed_mask: bool = False,
    ) -> None:
        """
        Build dynamic tree structure using CUDA kernel (in-place, writes to pre-allocated buffers).

        All output tensors are written in-place; nothing is returned.

        Args:
            history_draft_tokens_parent_buffer: [bs, history_size] int64
                Parent indices (directly used as parentList).
            topk_score_indices: [bs, max_total_draft_tokens] int64
                Selected token indices (directly used as selectedIndex).
            tree_mask: [bs, N, packed_bits] int32 (packed) or [bs, N, N] bool
                Pre-allocated output buffer for attention mask.
            positions: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for position IDs.
            retrieve_index: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for token retrieval indices.
            retrieve_next_token: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for first child indices.
            retrieve_next_sibling: [bs, num_draft_tokens] int32
                Pre-allocated output buffer for next sibling indices.
            use_packed_mask: bool
                Use bit-packed mask for memory efficiency.
        """
        bs = topk_score_indices.shape[0]
        # +1 because num_draft_tokens includes root node in SGLang's convention
        num_draft_tokens = topk_score_indices.shape[1] + 1
        tree_mask_mode = 2 if use_packed_mask else 1  # QLEN_ONLY_BITPACKING / QLEN_ONLY
        # Packed layout: last dim is int32 row stride (may exceed ceil(N/32) if padded).
        # Non-packed path ignores this (bool [bs,N,N] still has a last dim).
        num_int32_per_row = tree_mask.shape[-1]

        # The CUDA builder writes only active tree links/bits. Clear the reused
        # work buffers first so stale slot/tree state cannot leak into this tree.
        tree_mask[:bs].zero_()
        positions[:bs].zero_()
        retrieve_index[:bs].zero_()
        retrieve_next_token[:bs].fill_(-1)
        retrieve_next_sibling[:bs].fill_(-1)

        # Call CUDA kernel in-place
        try:
            torch.ops.trtllm.build_dynamic_tree_op(
                history_draft_tokens_parent_buffer[:bs],
                topk_score_indices,
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                self.K,
                self.depth,
                num_draft_tokens,
                tree_mask_mode,
                num_int32_per_row,
            )
        except Exception as e:
            raise RuntimeError(
                f"build_dynamic_tree_op failed: {e}\n"
                f"Inputs: bs={bs}, K={self.K}, depth={self.depth}, "
                f"num_draft_tokens={num_draft_tokens}"
            ) from e

    def verify_dynamic_tree_greedy_out(
        self,
        candidates: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        target_predict: torch.Tensor,
        num_gens: int,
        num_spec_step: int,
        tree_valid: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        In-place verify using pre-allocated output buffers (CUDA graph friendly).

        Args:
            candidates: [num_gens, N] int64 candidate tokens.
            retrieve_index: [num_gens, N] int32 retrieval indices.
            retrieve_next_token: [num_gens, N] int32 next token indices.
            retrieve_next_sibling: [num_gens, N] int32 next sibling indices.
            target_predict: [num_gens, N] int64 target predictions.
            num_gens: Number of generation requests.
            num_spec_step: Number of speculative steps.
            tree_valid: [num_gens] bool per-request flag.  When False the
                kernel early-returns with acceptTokenNum=0 (first-gen /
                dummy requests).  None means all trees are valid.

        Returns:
            Tuple of (predicts, accept_index, accept_token_num, accept_token)
            as slices of pre-allocated buffers.
        """
        N = candidates.size(1)
        predicts = self._verify_predicts_buf[: num_gens * N]
        accept_index = self._verify_accept_index_buf[:num_gens]
        accept_token_num = self._verify_accept_token_num_buf[:num_gens]
        accept_token = self._verify_accept_token_buf[:num_gens]

        if tree_valid is None:
            tree_valid = torch.ones(num_gens, dtype=torch.bool, device=candidates.device)

        try:
            torch.ops.trtllm.verify_dynamic_tree_greedy_out_op(
                candidates,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                target_predict,
                predicts,
                accept_index,
                accept_token_num,
                accept_token,
                tree_valid,
                num_spec_step,
            )
        except Exception as e:
            raise RuntimeError(
                f"verify_dynamic_tree_greedy_out_op failed: {e}\n"
                f"Inputs: num_gens={num_gens}, N={N}, "
                f"num_spec_step={num_spec_step}"
            ) from e

        return predicts, accept_index, accept_token_num, accept_token

    def verify_dynamic_tree_rejection_from_logits_out(
        self,
        candidates: torch.Tensor,
        draft_logits_tree: torch.Tensor,
        target_logits_tree: torch.Tensor,
        draft_prob_indices: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        tree_valid: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        skip_temperature: bool,
        num_gens: int,
        num_spec_step: int,
        seed: int | torch.Tensor = 0,
        offset: int | torch.Tensor = 0,
        d2t: torch.Tensor | None = None,
        skip_all_sampling_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tree-aware rejection sampling from logits (three CUDA ops).

        This path keeps draft/target logits as inputs, computes unique draft
        and target probabilities with separate CUDA ops, then runs the tree
        rejection kernel as a third CUDA op. `draft_prob_indices` maps each
        tree position to its shared draft-prob row. `tree_valid` guards
        first-gen and dummy requests that do not have a usable tree yet.
        """
        accept_index = self._rej_accept_index_buf[:num_gens]
        accept_token = self._rej_accept_token_buf[:num_gens]
        accept_tok_num = self._rej_accept_token_num_buf[:num_gens]
        seed_tensor = self._get_rejection_rng_tensor(seed, self._rej_seed_buf, "seed")
        offset_tensor = self._get_rejection_rng_tensor(offset, self._rej_offset_buf, "offset")
        num_draft_tokens = candidates.shape[1]
        if num_gens <= 0:
            raise ValueError(f"num_gens must be positive, got {num_gens}")
        if draft_logits_tree.shape[0] % num_gens != 0:
            raise ValueError(
                f"draft_logits_tree rows ({draft_logits_tree.shape[0]}) must be divisible by "
                f"num_gens ({num_gens})"
            )
        num_draft_prob_rows = draft_logits_tree.shape[0] // num_gens
        target_vocab_size = target_logits_tree.shape[-1]

        if tree_valid is None:
            tree_valid = torch.ones(num_gens, dtype=torch.bool, device=candidates.device)
        tree_valid = tree_valid.contiguous()

        if top_k is None:
            top_k_max = 0
        else:
            enabled_top_k = top_k[(top_k > 0) & (top_k < target_vocab_size)]
            top_k_max = int(enabled_top_k.max().item()) if enabled_top_k.numel() > 0 else 0

        try:
            draft_probs_tree = torch.ops.trtllm.compute_draft_probs_for_dynamic_tree_rejection_op(
                draft_logits_tree,
                temperatures,
                num_draft_prob_rows,
                target_vocab_size,
                top_k,
                top_p,
                skip_temperature,
                d2t=d2t,
                top_k_max=top_k_max,
                skip_all_sampling_params=skip_all_sampling_params,
            )

            (
                target_probs_tree,
                target_support_indices,
                target_support_lengths,
            ) = torch.ops.trtllm.compute_target_probs_for_dynamic_tree_rejection_op(
                target_logits_tree,
                temperatures,
                num_draft_tokens,
                top_k,
                top_p,
                skip_temperature,
                top_k_max=top_k_max,
                skip_all_sampling_params=skip_all_sampling_params,
            )

            torch.ops.trtllm.verify_dynamic_tree_rejection_out_op(
                candidates,
                draft_probs_tree,
                target_probs_tree,
                target_support_indices,
                target_support_lengths,
                draft_prob_indices,
                retrieve_next_token,
                retrieve_next_sibling,
                tree_valid,
                accept_index,
                accept_tok_num,
                accept_token,
                num_spec_step,
                seed_tensor,
                offset_tensor,
            )
        except Exception as e:
            raise RuntimeError(
                f"dynamic tree rejection op chain failed: {e}\n"
                f"Inputs: num_gens={num_gens}, N={candidates.shape[1]}, "
                f"draft_vocab={draft_logits_tree.shape[-1]}, "
                f"target_vocab={target_logits_tree.shape[-1]}, num_spec_step={num_spec_step}"
            ) from e

        return target_support_indices, accept_index, accept_tok_num, accept_token
