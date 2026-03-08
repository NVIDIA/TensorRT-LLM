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

from dataclasses import dataclass

import torch


@dataclass
class DynamicTreeBuffers:
    """
    Output buffers from dynamic tree building.

    Attributes:
        tree_mask: Attention mask for tree structure. Shape varies by mode.
        positions: Position IDs for each draft token [bs, num_draft_tokens].
        retrieve_index: Indices for token retrieval [bs, num_draft_tokens].
        retrieve_next_token: First child index for each node [bs, num_draft_tokens].
        retrieve_next_sibling: Next sibling index for each node [bs, num_draft_tokens].
    """

    tree_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor


@dataclass
class VerifyTreeResults:
    """
    Results from tree verification.

    Attributes:
        predicts: Verified token predictions [seq_lens_sum].
        accept_index: Indices of accepted tokens [bs, num_spec_step].
        accept_token_num: Number of accepted tokens per request [bs].
    """

    predicts: torch.Tensor
    accept_index: torch.Tensor
    accept_token_num: torch.Tensor


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
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_batch_size = max_batch_size
        self.device = device

        # Pre-allocate buffers for tree building
        self._preallocate_buffers()

    def _preallocate_buffers(self):
        """Pre-allocate reusable buffers to minimize runtime allocation."""
        # Preallocate parent_list buffer (max size)
        # Size: [max_batch_size, K * (depth - 1) + 1]
        # Note: Only first max_total_draft_tokens are used
        self.parent_list_buffer = torch.full(
            (self.max_batch_size, self.K * (self.depth - 1) + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
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
    ) -> DynamicTreeBuffers:
        """
        Build dynamic tree structure using CUDA kernel (in-place, writes to pre-allocated buffers).

        Args:
            history_draft_tokens_parent_buffer: [bs, history_size]
                Parent indices (directly used as parentList).
            topk_score_indices: [bs, max_total_draft_tokens]
                Selected token indices (directly used as selectedIndex).
            tree_mask: [bs, num_draft_tokens, num_draft_tokens] bool
                Pre-allocated output buffer for attention mask.
            positions: [bs, num_draft_tokens] int64
                Pre-allocated output buffer for position IDs.
            retrieve_index: [bs, num_draft_tokens] int64
                Pre-allocated output buffer for token retrieval indices.
            retrieve_next_token: [bs, num_draft_tokens] int64
                Pre-allocated output buffer for first child indices.
            retrieve_next_sibling: [bs, num_draft_tokens] int64
                Pre-allocated output buffer for next sibling indices.
            use_packed_mask: bool
                Use bit-packed mask for memory efficiency.

        Returns:
            DynamicTreeBuffers wrapping the same output buffers.
        """
        bs = topk_score_indices.shape[0]
        # +1 because num_draft_tokens includes root node in SGLang's convention
        num_draft_tokens = topk_score_indices.shape[1] + 1

        parent_list = history_draft_tokens_parent_buffer[:bs]
        selected_index = topk_score_indices

        # Determine tree mask mode
        if use_packed_mask:
            tree_mask_mode = 2  # QLEN_ONLY_BITPACKING
        else:
            tree_mask_mode = 1  # QLEN_ONLY

        # Call CUDA kernel in-place
        try:
            torch.ops.trtllm.build_dynamic_tree_op(
                parent_list,
                selected_index,
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                self.K,
                self.depth,
                num_draft_tokens,
                tree_mask_mode,
            )
        except Exception as e:
            raise RuntimeError(
                f"build_dynamic_tree_op failed: {e}\n"
                f"Inputs: bs={bs}, K={self.K}, depth={self.depth}, "
                f"num_draft_tokens={num_draft_tokens}"
            ) from e

        return DynamicTreeBuffers(
            tree_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
        )

    def verify_dynamic_tree_greedy(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        tree_buffers: DynamicTreeBuffers,
        num_spec_step: int,
    ) -> VerifyTreeResults:
        """
        Verify dynamic tree using greedy strategy.

        Args:
            draft_tokens: [bs, num_draft_tokens]
                Candidate draft tokens.
            target_logits: [bs, num_draft_tokens, vocab_size]
                Target model logits for verification.
            tree_buffers: DynamicTreeBuffers
                Tree structure from build_dynamic_tree.
            num_spec_step: int, optional
                Number of speculative steps. Defaults to num_draft_tokens.

        Returns:
            VerifyTreeResults containing verification outputs.
        """
        bs, num_draft_tokens = draft_tokens.shape

        assert num_spec_step is not None, (
            "num_spec_step must be explicitly provided (= max_draft_len + 1)"
        )

        # Get target predictions (greedy)
        target_predict = target_logits.argmax(dim=-1)  # [bs, num_draft_tokens]

        # Call CUDA kernel
        try:
            predicts, accept_index, accept_token_num = (
                torch.ops.trtllm.verify_dynamic_tree_greedy_op(
                    draft_tokens,
                    tree_buffers.retrieve_index,
                    tree_buffers.retrieve_next_token,
                    tree_buffers.retrieve_next_sibling,
                    target_predict,
                    num_spec_step,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"verify_dynamic_tree_greedy_op failed: {e}\n"
                f"Inputs: bs={bs}, num_draft_tokens={num_draft_tokens}, "
                f"num_spec_step={num_spec_step}"
            ) from e

        return VerifyTreeResults(
            predicts=predicts, accept_index=accept_index, accept_token_num=accept_token_num
        )


def create_dynamic_tree_ops_converter(
    dynamic_tree_max_topK: int,
    max_draft_len: int,
    max_total_draft_tokens: int,
    max_batch_size: int,
    device: torch.device,
) -> DynamicTreeOpsConverter:
    """
    Factory function to create a DynamicTreeOpsConverter instance.

    Args:
        dynamic_tree_max_topK: Maximum top-K tokens per node.
        max_draft_len: Maximum draft length (tree depth).
        max_total_draft_tokens: Total number of draft tokens.
        max_batch_size: Maximum batch size.
        device: CUDA device.

    Returns:
        DynamicTreeOpsConverter instance.
    """
    return DynamicTreeOpsConverter(
        dynamic_tree_max_topK=dynamic_tree_max_topK,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_batch_size=max_batch_size,
        device=device,
    )
