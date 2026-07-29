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

from tensorrt_llm._torch.pyexecutor.sampler.sampling_utils import compute_probs_from_logits


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

        # Pre-allocated output buffers for verify_dynamic_tree_greedy_out_packed_op
        max_path_len = max_draft_len + 1
        self._verify_accept_index_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int32, device=device
        )
        self._verify_accept_token_num_buf = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._verify_accept_token_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int32, device=device
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
        if bs == 0:
            return
        # +1 because num_draft_tokens includes root node in SGLang's convention
        num_draft_tokens = topk_score_indices.shape[1] + 1
        tree_mask_mode = 2 if use_packed_mask else 1  # QLEN_ONLY_BITPACKING / QLEN_ONLY

        # Actual buffer row stride (int32s); kernel otherwise computes ceil(num_draft_tokens / 32).
        num_int32_per_row = tree_mask.shape[-1] if use_packed_mask else 0

        # CUDA kernel indexes as ptr[bid * draftTokenNum + tid], so dim1 must equal num_draft_tokens.
        assert positions.shape[-1] == num_draft_tokens, (
            f"positions dim1 ({positions.shape[-1]}) != num_draft_tokens ({num_draft_tokens})"
        )

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

    def verify_dynamic_tree_greedy_out_packed(
        self,
        candidates: torch.Tensor,
        retrieve_packed: torch.Tensor,
        target_predict: torch.Tensor,
        num_gens: int,
        num_spec_step: int,
        tree_valid: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """In-place verify with int32 token tensors and packed int32 retrieve layout."""
        N = candidates.size(1)
        accept_index = self._verify_accept_index_buf[:num_gens]
        accept_token_num = self._verify_accept_token_num_buf[:num_gens]
        accept_token = self._verify_accept_token_buf[:num_gens]

        if tree_valid is None:
            tree_valid = torch.ones(num_gens, dtype=torch.bool, device=candidates.device)

        retrieve_packed_contig = retrieve_packed[:, :N, :].contiguous()
        try:
            torch.ops.trtllm.verify_dynamic_tree_greedy_out_packed_op(
                candidates,
                retrieve_packed_contig,
                target_predict,
                accept_index,
                accept_token_num,
                accept_token,
                tree_valid,
                num_spec_step,
            )
        except Exception as e:
            raise RuntimeError(
                f"verify_dynamic_tree_greedy_out_packed_op failed: {e}\n"
                f"Inputs: num_gens={num_gens}, N={N}, num_spec_step={num_spec_step}"
            ) from e

        return accept_index, accept_token_num, accept_token

    def verify_dynamic_tree_rejection_out(
        self,
        draft_tokens: torch.Tensor,
        target_logits_tree: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        tree_valid: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        num_gens: int,
        num_spec_step: int,
        seed: int | torch.Tensor = 0,
        offset: int | torch.Tensor = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamic tree rejection sampling.

        Computes target probabilities from logits, then runs the target-only
        rejection kernel. No draft probabilities are needed.
        `tree_valid` guards first-gen and dummy requests without a valid tree.
        """
        accept_index = self._rej_accept_index_buf[:num_gens]
        accept_token = self._rej_accept_token_buf[:num_gens]
        accept_tok_num = self._rej_accept_token_num_buf[:num_gens]
        seed_tensor = self._get_rejection_rng_tensor(seed, self._rej_seed_buf, "seed")
        offset_tensor = self._get_rejection_rng_tensor(offset, self._rej_offset_buf, "offset")
        if num_gens <= 0:
            raise ValueError(f"num_gens must be positive, got {num_gens}")
        if target_logits_tree.shape[0] % num_gens != 0:
            raise ValueError(
                "target_logits_tree rows must be divisible by num_gens, got "
                f"{target_logits_tree.shape[0]} and {num_gens}"
            )
        # draft_tokens has shape [num_gens, N-1]; derive total tree nodes N from target_logits_tree.
        num_draft_tokens = target_logits_tree.shape[0] // num_gens

        if tree_valid is None:
            tree_valid = torch.ones(num_gens, dtype=torch.bool, device=draft_tokens.device)
        tree_valid = tree_valid.contiguous()

        # Expand per-request sampling params to per-tree-position (num_gens * N rows).
        temps_exp = temperatures.repeat_interleave(num_draft_tokens)
        top_k_exp = top_k.repeat_interleave(num_draft_tokens) if top_k is not None else None
        top_p_exp = top_p.repeat_interleave(num_draft_tokens) if top_p is not None else None

        # Compute target probs using the shared linear-path interface (FlashInfer fast
        # path when available, sort-based fallback otherwise). Returns dense full-vocab
        # probs [num_gens * N, vocab_size]; no sparse support indices needed.
        target_probs_flat = compute_probs_from_logits(
            target_logits_tree,
            temps_exp,
            top_k_exp,
            top_p_exp,
        )
        vocab_size = target_probs_flat.shape[-1]
        target_probs_tree = target_probs_flat.reshape(num_gens, num_draft_tokens, vocab_size)

        try:
            torch.ops.trtllm.verify_dynamic_tree_rejection_out_op(
                draft_tokens,
                target_probs_tree,
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
                f"dynamic tree rejection target-only op chain failed: {e}\n"
                f"Inputs: num_gens={num_gens}, N={draft_tokens.shape[1] + 1}, "
                f"target_vocab={target_logits_tree.shape[-1]}, num_spec_step={num_spec_step}"
            ) from e

        return accept_index, accept_tok_num, accept_token
