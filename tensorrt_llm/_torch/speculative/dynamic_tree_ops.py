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

    def verify_dynamic_tree_rejection_out(
        self,
        candidates: torch.Tensor,
        draft_probs_tree: torch.Tensor,
        target_probs_tree: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        num_gens: int,
        num_spec_step: int,
        seed: int = 0,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tree-aware rejection sampling for speculative decoding (CUDA kernel).

        Traverses the draft tree for each request, accepting or rejecting tokens
        at each node using rejection sampling. At each depth, siblings are tried
        in order; the first accepted sibling continues the path. If all siblings
        are rejected, a correction token is sampled from (target - draft)_+.

        This method does NOT require a prior greedy verification pass.

        Args:
            candidates: [num_gens, N] int64 — col 0 = root (target's prediction),
                cols 1..N-1 = draft tokens at each tree position.
            draft_probs_tree: [num_gens, N-1, vocab] float32 — draft prob distribution
                for each draft node (index 0 = tree position 1, i.e. candidates[:, 1]).
            target_probs_tree: [num_gens, N, vocab] float32 — target prob distribution
                at each tree position (index 0 = root position).
            retrieve_next_token: [num_gens, N] int32 — first child position (-1 = none).
            retrieve_next_sibling: [num_gens, N] int32 — next sibling position (-1 = none).
            num_gens: Number of generation requests.
            num_spec_step: max_path_len (= max_draft_len + 1).
            seed: Philox RNG seed for reproducibility.
            offset: Philox RNG offset (increment between calls).

        Returns:
            Tuple of (None, accept_index, accept_token_num, accept_token) matching
            the interface of verify_dynamic_tree_greedy_out, where:
              accept_index[req, d]     = tree position of the d-th accepted draft token
                                        with slot 0 reserved for the root position
              accept_token_num[req]    = number of accepted draft tokens
                                        (+1 in caller for total emitted tokens)
              accept_token[req, 0..D] = emitted token sequence:
                                        accepted draft tokens followed by the
                                        final bonus/correction token
        """
        accept_index = self._rej_accept_index_buf[:num_gens]
        accept_token = self._rej_accept_token_buf[:num_gens]
        accept_tok_num = self._rej_accept_token_num_buf[:num_gens]

        try:
            torch.ops.trtllm.verify_dynamic_tree_rejection_out_op(
                candidates,
                draft_probs_tree,
                target_probs_tree,
                retrieve_next_token,
                retrieve_next_sibling,
                accept_index,
                accept_tok_num,
                accept_token,
                num_spec_step,
                seed,
                offset,
            )
        except Exception as e:
            raise RuntimeError(
                f"verify_dynamic_tree_rejection_out_op failed: {e}\n"
                f"Inputs: num_gens={num_gens}, N={candidates.shape[1]}, "
                f"vocab={target_probs_tree.shape[-1]}, num_spec_step={num_spec_step}"
            ) from e

        return None, accept_index, accept_tok_num, accept_token
