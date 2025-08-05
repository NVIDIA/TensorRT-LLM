"""Torch attention reference implementations for testing.

This module provides clean reference implementations using the torch backend
that can be used across all attention operation test files to eliminate
code duplication and ensure consistency.
"""

import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


class TorchAttentionReference:
    """Reference implementation using the torch backend for consistency."""

    @staticmethod
    def basic_mha_with_cache(q, k, v, k_cache, v_cache, input_positions, scale=None):
        """Reference implementation for basic MHA with cache (generate phase).

        This matches the signature of triton_attention_fused_mha_with_cache.

        Args:
            q: Query tensor [batch, seq, n_heads, head_dim]
            k: Key tensor [batch, seq, n_kv_heads, head_dim]
            v: Value tensor [batch, seq, n_kv_heads, head_dim]
            k_cache: Key cache [batch, max_seq_len, n_kv_heads, head_dim]
            v_cache: Value cache [batch, max_seq_len, n_kv_heads, head_dim]
            input_positions: Positions to update cache [batch]
            scale: Optional attention scale

        Returns:
            Attention output [batch, seq, n_heads, head_dim] (same shape as q)
        """
        batch_size, seq_len = q.shape[:2]

        # Convert to flattened format for torch backend
        seq_len_tensor = torch.full((batch_size,), seq_len, device=q.device, dtype=torch.int32)
        cache_loc = torch.arange(batch_size, device=q.device, dtype=torch.int32)
        seq_start = torch.arange(
            0, batch_size * seq_len, seq_len, device=q.device, dtype=torch.int32
        )

        # Flatten inputs to [1, total_seq_len, ...] format
        q_flat = q.view(1, batch_size * seq_len, -1)
        k_flat = k.view(1, batch_size * seq_len, -1)
        v_flat = v.view(1, batch_size * seq_len, -1)

        # Call torch backend via custom op registry
        output_flat = torch.ops.auto_deploy.torch_cached_attention_with_cache(
            q_flat,
            k_flat,
            v_flat,
            seq_len_tensor,
            input_positions,
            cache_loc,
            seq_start,
            k_cache,
            v_cache,
            scale,
        )

        # Reshape back to original format [batch, seq, n_heads, head_dim]
        if q.ndim == 4:
            # Input was [batch, seq, n_heads, head_dim], but triton always returns flattened
            # So return [batch, seq, n_heads * head_dim] to match triton behavior
            return output_flat.view(batch_size, seq_len, -1)
        else:
            # Input was [batch, seq, n_heads * head_dim], return same shape
            return output_flat.view(batch_size, seq_len, -1)

    @staticmethod
    def flattened_mha_with_cache(
        q, k, v, seq_len, input_positions, cache_loc, seq_start, k_cache, v_cache, scale=None
    ):
        """Reference implementation following triton flattened MHA pattern.

        This function directly calls the torch backend implementation via custom op registry.
        """
        return torch.ops.auto_deploy.torch_cached_attention_with_cache(
            q, k, v, seq_len, input_positions, cache_loc, seq_start, k_cache, v_cache, scale
        )

    @staticmethod
    def decode_with_prefilled_cache(q, k_ref, v_ref, k_cache, v_cache, prefill_lengths):
        """Reference for decode phase with pre-filled cache (flashinfer tests).

        Args:
            q: Query tensor [batch, seq=1, n_heads, head_dim]
            k_ref: Reference keys (full context including prefill + new token)
            v_ref: Reference values (full context including prefill + new token)
            k_cache: Key cache [batch, max_seq_len, n_heads, head_dim]
            v_cache: Value cache [batch, max_seq_len, n_heads, head_dim]
            prefill_lengths: Number of pre-filled tokens per batch [batch]

        Returns:
            Attention output [batch, seq=1, n_heads * head_dim]
        """
        batch_size = q.shape[0]
        seq_len = torch.ones(batch_size, device=q.device, dtype=torch.int32)
        cache_loc = torch.arange(batch_size, device=q.device, dtype=torch.int32)
        # Fix: Each sequence starts at its own position in the flattened tensor
        seq_start = torch.arange(batch_size, device=q.device, dtype=torch.int32)

        # For decode phase, input_positions should be the prefill_lengths (where to append new token)
        input_positions = prefill_lengths.to(torch.int32)

        # Extract the new k,v tokens from k_ref, v_ref (last token for each batch)
        k_new = k_ref[:, -1:, :, :]  # [batch, 1, n_heads, head_dim]
        v_new = v_ref[:, -1:, :, :]  # [batch, 1, n_heads, head_dim]

        # Convert to flattened format [1, total_seq_len, ...]
        q_flat = q.view(1, batch_size, -1)
        k_flat = k_new.view(1, batch_size, -1)
        v_flat = v_new.view(1, batch_size, -1)

        # Call torch backend via custom op registry
        output_flat = torch.ops.auto_deploy.torch_cached_attention_with_cache(
            q_flat,
            k_flat,
            v_flat,
            seq_len,
            input_positions,
            cache_loc,
            seq_start,
            k_cache,
            v_cache,
            None,
        )

        # Return in flattened format to match flashinfer backend behavior [batch, seq=1, n_heads * head_dim]
        return output_flat.view(batch_size, 1, -1)

    @staticmethod
    def mha_with_features(
        q,
        k,
        v,
        seq_len,
        input_positions,
        cache_loc,
        seq_start,
        k_cache,
        v_cache,
        scale=None,
        logit_cap=None,
        sliding_window_size=None,
    ):
        """Reference implementation with advanced features (logit capping, sliding window).

        This demonstrates how to use the torch backend with additional features.
        """
        return torch.ops.auto_deploy.torch_cached_attention_with_cache(
            q,
            k,
            v,
            seq_len,
            input_positions,
            cache_loc,
            seq_start,
            k_cache,
            v_cache,
            scale,
            None,  # sinks
            sliding_window_size,
            logit_cap,
        )

    @staticmethod
    def prepare_flattened_inputs(q_list, k_list, v_list, input_positions_list):
        """Helper to convert list of per-sequence tensors to flattened format.

        Args:
            q_list: List of query tensors per sequence
            k_list: List of key tensors per sequence
            v_list: List of value tensors per sequence
            input_positions_list: List of input positions per sequence

        Returns:
            Tuple of (q_flat, k_flat, v_flat, seq_len, input_positions, cache_loc, seq_start)
        """
        device = q_list[0].device

        # Compute sequence metadata
        seq_lengths = [q.shape[0] for q in q_list]
        seq_len = torch.tensor(seq_lengths, device=device, dtype=torch.int32)
        seq_start = torch.tensor(
            [sum(seq_lengths[:i]) for i in range(len(seq_lengths))],
            device=device,
            dtype=torch.int32,
        )

        # Flatten tensors
        q_flat = torch.cat(q_list, dim=0).unsqueeze(0)  # [1, total_seq_len, ...]
        k_flat = torch.cat(k_list, dim=0).unsqueeze(0)  # [1, total_seq_len, ...]
        v_flat = torch.cat(v_list, dim=0).unsqueeze(0)  # [1, total_seq_len, ...]

        # Create metadata tensors
        input_positions = torch.tensor(input_positions_list, device=device, dtype=torch.int32)
        cache_loc = torch.arange(len(q_list), device=device, dtype=torch.int32)

        return q_flat, k_flat, v_flat, seq_len, input_positions, cache_loc, seq_start
