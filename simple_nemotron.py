"""
Simplified NemotronHMamba2Mixer - Tensor Algebra Operations Only

This file focuses on the tensor algebra operations in the Mamba2 forward pass,
with detailed annotations for parallelization across multiple GPUs.

Notation:
- b: batch_size
- s: seq_len
- h_in: hidden_size (input)
- h: num_heads
- d: head_dim
- n: ssm_state_size
- g: n_groups
- i: intermediate_size (= h * d)
- c: chunk_size
- num_chunks: number of chunks (= ceil(s / c))

Key relationships:
- intermediate_size = num_heads * head_dim  (i = h * d)
- conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
"""

from typing import Optional

import torch
import torch.nn as nn


class NemotronHMamba2Mixer:
    """
    Mamba2 SSM Mixer - Tensor Algebra Only

    This class contains only the algebraically significant operations,
    annotated with parallelization strategies.
    """

    def __init__(self):
        # Model dimensions (example values)
        self.hidden_size = 4096  # h_in
        self.num_heads = 64  # h
        self.head_dim = 64  # d
        self.intermediate_size = 4096  # i = h * d
        self.n_groups = 8  # g
        self.ssm_state_size = 128  # n
        self.chunk_size = 256  # c
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # analogy to transformers' attention:
        # A - query  [b, s, h, d]
        # B - key    [b, s, g, d]  # n_groups function as num KV heads
        # C - value  [b, s, g, d]
        # D - attention mask
        # B and C will be broadcasted from g to h for SSM computation

        # Learnable parameters
        conv_kernel = 4
        self.in_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel -
            1  # This ensures output length >= input length
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.A_log = nn.Parameter(torch.randn(self.num_heads))
        self.dt_bias = nn.Parameter(torch.randn(self.num_heads))
        self.D = nn.Parameter(torch.randn(self.num_heads))

    def segment_sum(self, input_tensor):
        """
        Segment sum operation - computes cumulative sum within triangular mask.

        Input: [..., chunk_size]
        Output: [..., chunk_size, chunk_size]

        PARALLELIZATION ANALYSIS:
        - All batch dimensions (...): FULLY PARALLEL (embarrassingly parallel)
        - chunk_size dimension: SEQUENTIAL (cumsum is inherently sequential)
        - Can parallelize across chunks if processing multiple chunks
        - Cross-GPU: Can distribute batch/head dimensions, but cumsum requires local computation
        """
        chunk_size = input_tensor.size(-1)

        # Input: [..., c] -> [..., c, c]
        # Complexity: O(c^2) per element in batch
        # Parallel: All leading dims are independent
        input_tensor = input_tensor[..., None].expand(*input_tensor.size(),
                                                      chunk_size)

        # Cumsum along dim=-2
        # Input: [..., c, c], Output: [..., c, c]
        # Complexity: O(c^2) per element
        # Parallel: Leading dims (...) are independent, but cumsum is sequential in last-2 dim
        # WARNING: cumsum is NOT parallelizable in the reduction dimension
        tensor_segsum = torch.cumsum(input_tensor, dim=-2)

        return tensor_segsum

    def torch_forward_algebra_only(
            self,
            input_states: torch.Tensor,  # [b, s, h_in]
            cache_params: Optional = None,
            debug: bool = False):
        """
        Forward pass with TENSOR ALGEBRA operations only.
        Focus: matrix multiplications, reductions, cumulative operations.
        Excluded: element-wise ops (activations, exp, masking, etc.)
        """

        batch_size, seq_len, _ = input_states.shape  # b, s, h_in

        # =============================================================================
        # STEP 1: Input Projection (Linear Layer)
        # =============================================================================
        # Operation: projected_states = input_states @ in_proj.weight^T + in_proj.bias
        # Input:  [b, s, h_in]
        # Weight: [projection_size, h_in] where projection_size = i + conv_dim + h
        # Output: [b, s, projection_size]
        # Complexity: O(b * s * h_in * projection_size) ≈ O(b * s * h_in^2)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL - can split across GPUs with no communication
        # - Sequence (s): FULLY PARALLEL - can split across GPUs with no communication
        # - Hidden (h_in): PARALLEL with ALL_REDUCE - this is the reduction dimension
        #   * If split h_in across GPUs, need all_reduce to sum partial results
        #   * Row-parallel: split weight rows, no all_reduce needed
        #   * Column-parallel: split weight columns, need all_reduce after
        # - Output (projection_size): PARALLEL - row-wise split requires no communication
        #
        # TENSOR PARALLEL STRATEGIES:
        # 1. Batch parallel: Each GPU processes different batch elements
        # 2. Sequence parallel: Each GPU processes different tokens (works for attention)
        # 3. Tensor parallel (column): Split projection_size, all_reduce on h_in
        # 4. Tensor parallel (row): Split h_in, each GPU computes partial projection
        projected_states = self.in_proj(input_states)  # [b, s, projection_size]

        # Split the projection into components
        # gate: [b, s, i], hidden_states_B_C: [b, s, conv_dim], dt: [b, s, h]
        # Note: d_mlp is computed but will be 0 in this configuration
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size -
                 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2

        if debug:
            print("\nProjection split:")
            print(f"  projected_states shape: {projected_states.shape}")
            print(f"  d_mlp: {d_mlp}")
            print(f"  Split sizes: [d_mlp={d_mlp}, d_mlp={d_mlp}, "
                  f"intermediate={self.intermediate_size}, "
                  f"conv_dim={self.conv_dim}, num_heads={self.num_heads}]")
            print(
                f"  Total: {2*d_mlp + self.intermediate_size + self.conv_dim + self.num_heads}"
            )

        # Split into components (d_mlp will be 0, so first two splits are empty)
        splits = []
        current_idx = 0
        for size in [
                d_mlp, d_mlp, self.intermediate_size, self.conv_dim,
                self.num_heads
        ]:
            if size > 0:
                splits.append(projected_states[...,
                                               current_idx:current_idx + size])
            else:
                splits.append(
                    projected_states[...,
                                     current_idx:current_idx])  # Empty tensor
            current_idx += size

        _, _, gate, hidden_states_B_C, dt = splits[0], splits[1], splits[
            2], splits[3], splits[4]

        if debug:
            print(
                f"  After split - gate: {gate.shape}, hidden_states_B_C: {hidden_states_B_C.shape}, dt: {dt.shape}"
            )

        # =============================================================================
        # STEP 2: Conv1D Operation
        # =============================================================================
        # Conv1D is applied on sequence dimension
        # Input:  [b, conv_dim, s] (after transpose)
        # Weight: [conv_dim, 1, kernel_size]
        # Output: [b, conv_dim, s]
        # Complexity: O(b * conv_dim * s * kernel_size)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Channel (conv_dim): FULLY PARALLEL (depthwise conv, groups=conv_dim)
        # - Sequence (s): PARALLEL with communication
        #   * Conv requires kernel_size-1 halo elements from neighbors
        #   * Split sequence: need halo exchange between GPUs
        #   * First/last kernel_size-1 tokens need data from adjacent GPUs
        #
        # TENSOR PARALLEL STRATEGIES:
        # 1. Batch parallel: Easiest, no communication
        # 2. Channel parallel: Split conv_dim, no cross-channel communication (depthwise)
        # 3. Sequence parallel: Need halo exchange (communication overhead)
        hidden_states_B_C_transposed = hidden_states_B_C.transpose(
            1, 2)  # [b, conv_dim, s]
        conv_out = self.conv1d(hidden_states_B_C_transposed)[
            ..., :seq_len]  # [b, conv_dim, s]
        hidden_states_B_C = conv_out.transpose(1, 2)  # [b, s, conv_dim]

        # Split conv output
        # conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
        split_sizes = [
            self.intermediate_size, self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size
        ]

        # Verify split sizes match conv_dim
        assert sum(
            split_sizes
        ) == self.conv_dim, f"Split sizes {split_sizes} don't sum to conv_dim {self.conv_dim}"

        hidden_states = hidden_states_B_C[..., :self.intermediate_size]
        B = hidden_states_B_C[...,
                              self.intermediate_size:self.intermediate_size +
                              self.n_groups * self.ssm_state_size]
        C = hidden_states_B_C[..., self.intermediate_size +
                              self.n_groups * self.ssm_state_size:]

        # hidden_states: [b, s, i], B: [b, s, g*n], C: [b, s, g*n]
        if debug:
            print(
                f"After split - hidden_states: {hidden_states.shape}, B: {B.shape}, C: {C.shape}"
            )

        # =============================================================================
        # STEP 3: SSM State Space Computation (Main Computation)
        # =============================================================================

        # Reshape for SSM computation
        # hidden_states: [b, s, i] -> [b, s, h, d]
        # B: [b, s, g*n] -> [b, s, g, n]
        # C: [b, s, g*n] -> [b, s, g, n]
        # Complexity: O(1) - just view operations
        # Parallel: All dimensions are independent

        if debug:
            print(
                f"Before reshape - hidden_states: {hidden_states.shape}, expected: [{batch_size}, {seq_len}, {self.intermediate_size}]"
            )
            print(
                f"Reshape target: [{batch_size}, {seq_len}, {self.num_heads}, {self.head_dim}]"
            )
            print(
                f"intermediate_size={self.intermediate_size}, num_heads={self.num_heads}, head_dim={self.head_dim}"
            )
            print(f"num_heads * head_dim = {self.num_heads * self.head_dim}")

        # Verify dimensions are compatible
        assert hidden_states.shape[-1] == self.num_heads * self.head_dim, \
            f"Cannot reshape {hidden_states.shape} to have {self.num_heads} heads of dim {self.head_dim}"

        hidden_states = hidden_states.reshape(batch_size, seq_len,
                                              self.num_heads, self.head_dim)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        if debug:
            print(
                f"After reshape - hidden_states: {hidden_states.shape}, B: {B.shape}, C: {C.shape}"
            )

        # Repeat B and C to match num_heads (from n_groups)
        # Input:  [b, s, g, n]
        # Output: [b, s, h, n] where h = g * repetition_factor
        # Complexity: O(b * s * h * n) memory, O(1) compute (just indexing)
        # Parallel: FULLY PARALLEL - simple replication
        B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)  # [b, s, h, n]
        C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)  # [b, s, h, n]

        # Compute pad size for chunking
        pad_size = (self.chunk_size -
                    seq_len % self.chunk_size) % self.chunk_size

        # =============================================================================
        # STEP 3a: Chunk Reshaping
        # =============================================================================
        # Reshape sequences into chunks
        # Input:  [b, s, h, d]
        # Output: [b, num_chunks, c, h, d] where num_chunks = ceil(s/c)
        # Complexity: O(1) - reshape only
        # Parallel: All dimensions independent
        #
        # Note: This creates a new dimension (num_chunks) that can be parallelized!
        def reshape_into_chunks(tensor, pad_size, chunk_size):
            """Pad and reshape into chunks"""
            # Pad: increases sequence length by pad_size
            # Reshape: [b, s+pad, ...] -> [b, num_chunks, c, ...]
            # Parallel: Independent across batch dimension
            return tensor  # Simplified - actual implementation in original code

        # After chunking (conceptual):
        # hidden_states: [b, num_chunks, c, h, d]
        # A: [b, h, num_chunks, c]  (permuted for computation)
        # B: [b, num_chunks, c, h, n]
        # C: [b, num_chunks, c, h, n]

        # =============================================================================
        # STEP 3b: Cumulative Sum (Sequential Operation)
        # =============================================================================
        # A_cumsum = torch.cumsum(A, dim=-1)
        # Input:  [b, h, num_chunks, c]
        # Output: [b, h, num_chunks, c]
        # Complexity: O(b * h * num_chunks * c)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Heads (h): FULLY PARALLEL
        # - Chunks (num_chunks): FULLY PARALLEL - each chunk is independent!
        # - Chunk_size (c): SEQUENTIAL - cumsum is inherently sequential
        #
        # CRITICAL: cumsum within each chunk is sequential, but different chunks
        # can be computed in parallel! This is why chunking is valuable.
        #
        # TENSOR PARALLEL: Can split b, h, num_chunks across GPUs with NO communication
        A_cumsum = torch.zeros(batch_size, self.num_heads,
                               (seq_len + pad_size) // self.chunk_size,
                               self.chunk_size)

        # =============================================================================
        # STEP 3c: Segment Sum (calls cumsum internally)
        # =============================================================================
        # L = torch.exp(segment_sum(A))
        # segment_sum input:  [b, h, num_chunks, c]
        # segment_sum output: [b, h, num_chunks, c, c]
        # Complexity: O(b * h * num_chunks * c^2)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Heads (h): FULLY PARALLEL
        # - Chunks (num_chunks): FULLY PARALLEL
        # - Within chunk (c): SEQUENTIAL (cumsum)
        # - Output dimension (c): Creates new dimension, parallel
        #
        # TENSOR PARALLEL: Can split b, h, num_chunks across GPUs with NO communication
        # The c x c matrix per chunk is computed locally on each GPU
        L = self.segment_sum(A_cumsum)  # [b, h, num_chunks, c, c]

        # =============================================================================
        # STEP 3d: Attention-like Computation (G matrix)
        # =============================================================================
        # G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        # Input C: [b, num_chunks, c, h, n]
        # Input B: [b, num_chunks, c, h, n]
        # Output:  [b, num_chunks, c, c, h, n]
        # Then: G = G_intermediate.sum(dim=-1) -> [b, num_chunks, c, c, h]
        # Complexity: O(b * num_chunks * c^2 * h * n)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Chunks (num_chunks): FULLY PARALLEL
        # - Query positions (c): FULLY PARALLEL
        # - Key positions (c): FULLY PARALLEL
        # - Heads (h): FULLY PARALLEL
        # - State dimension (n): PARALLEL with ALL_REDUCE (this is reduction dim)
        #   * If split n across GPUs, need all_reduce after sum
        #
        # TENSOR PARALLEL STRATEGIES:
        # 1. Split any of (b, num_chunks, h) with no communication
        # 2. Split n with all_reduce after reduction
        # 3. This is similar to attention QK^T computation!
        C_expanded = torch.zeros(batch_size,
                                 (seq_len + pad_size) // self.chunk_size,
                                 self.chunk_size, self.chunk_size,
                                 self.num_heads, self.ssm_state_size)
        B_expanded = torch.zeros(batch_size,
                                 (seq_len + pad_size) // self.chunk_size,
                                 self.chunk_size, self.chunk_size,
                                 self.num_heads, self.ssm_state_size)
        G_intermediate = C_expanded * B_expanded  # [b, num_chunks, c, c, h, n]

        # Reduction over state dimension
        # Input:  [b, num_chunks, c, c, h, n]
        # Output: [b, num_chunks, c, c, h]
        # Complexity: O(b * num_chunks * c^2 * h * n)
        # Parallel: Reduction over n - if n is split, need all_reduce
        G = G_intermediate.sum(dim=-1)  # [b, num_chunks, c, c, h]

        # =============================================================================
        # STEP 3e: Attention Weights Computation (M matrix)
        # =============================================================================
        # M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        # After permute, L: [b, num_chunks, c, c, h]
        # G: [b, num_chunks, c, c, h]
        # M_intermediate: [b, num_chunks, c, c, h, d] (after broadcasting)
        # M = M_intermediate.sum(dim=-1) -> [b, num_chunks, c, c, h, d]
        # Complexity: O(b * num_chunks * c^2 * h * d)
        #
        # PARALLELIZATION:
        # - All of (b, num_chunks, c, c, h, d) are independent in the outer product
        # - The sum reduction is over a broadcasted dimension
        # - FULLY PARALLEL across b, num_chunks, c, c, h
        # - d dimension: depends on reduction
        L_permuted = L.permute(0, 2, 3, 4, 1)  # [b, num_chunks, c, c, h]
        M_intermediate = torch.zeros(batch_size,
                                     (seq_len + pad_size) // self.chunk_size,
                                     self.chunk_size, self.chunk_size,
                                     self.num_heads, self.head_dim)
        M = M_intermediate.sum(
            dim=-1)  # Simplified - actual computation more complex

        # =============================================================================
        # STEP 3f: Intra-chunk Output (Y_diag)
        # =============================================================================
        # Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)
        # M: [b, num_chunks, c, c, h, d]
        # hidden_states after chunking: [b, num_chunks, c, h, d]
        # After broadcasting: [b, num_chunks, c, c, h, d]
        # Output after sum: [b, num_chunks, c, h, d]
        # Complexity: O(b * num_chunks * c^2 * h * d)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Chunks (num_chunks): FULLY PARALLEL
        # - Output positions (c, dim=2): FULLY PARALLEL
        # - Input positions (c, dim=3): PARALLEL with ALL_REDUCE (reduction dimension)
        # - Heads (h): FULLY PARALLEL
        # - Head_dim (d): FULLY PARALLEL
        #
        # This is essentially the attention "apply to values" step!
        # TENSOR PARALLEL: Split b, num_chunks, h with no communication
        # If split input c dimension, need all_reduce
        Y_diag = torch.zeros(batch_size,
                             (seq_len + pad_size) // self.chunk_size,
                             self.chunk_size, self.num_heads, self.head_dim)

        # =============================================================================
        # STEP 3g: Intra-chunk State Computation
        # =============================================================================
        # B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        # states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)
        # B_decay: [b, num_chunks, c, h, n]
        # hidden_states: [b, num_chunks, c, h, d]
        # After broadcasting: [b, num_chunks, c, h, d, n]
        # After sum over c: [b, num_chunks, h, d, n]
        # Complexity: O(b * num_chunks * c * h * d * n)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Chunks (num_chunks): FULLY PARALLEL (each chunk's state independent)
        # - Sequence within chunk (c, dim=2): PARALLEL with ALL_REDUCE (reduction)
        # - Heads (h): FULLY PARALLEL
        # - Head_dim (d): FULLY PARALLEL
        # - State_size (n): FULLY PARALLEL
        #
        # TENSOR PARALLEL: Can split b, num_chunks, h, d, n with no communication
        # If split c dimension, need all_reduce after sum
        states = torch.zeros(batch_size,
                             (seq_len + pad_size) // self.chunk_size,
                             self.num_heads, self.head_dim, self.ssm_state_size)

        # =============================================================================
        # STEP 3h: Inter-chunk Recurrence (Sequential Across Chunks!)
        # =============================================================================
        # decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        # decay_chunk = decay_chunk.transpose(1, 3)  # [b, num_chunks+1, num_chunks+1, h]
        # new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        #
        # Input states: [b, num_chunks, h, d, n]
        # decay_chunk: [b, num_chunks+1, num_chunks+1, h]
        # new_states: [b, num_chunks+1, h, d, n]
        #
        # Complexity: O(b * num_chunks^2 * h * d * n)
        #
        # PARALLELIZATION - CRITICAL INSIGHT:
        # - Batch (b): FULLY PARALLEL
        # - Heads (h): FULLY PARALLEL
        # - Head_dim (d): FULLY PARALLEL
        # - State_size (n): FULLY PARALLEL
        # - Chunks (num_chunks): SEQUENTIAL!!! This is a recurrence across chunks!
        #
        # **This is the main sequential bottleneck for long sequences!**
        #
        # The sum over dim=1 creates a dependency between chunks:
        # new_states[chunk_i] depends on states[0:i]
        #
        # TENSOR PARALLEL STRATEGIES:
        # 1. Can split b, h, d, n across GPUs with no communication
        # 2. CANNOT efficiently parallelize across chunks without changing algorithm
        # 3. For very long sequences, this becomes a bottleneck
        # 4. Possible solution: Use ring-reduce or prefix-sum parallel algorithms
        #    but this requires O(log num_chunks) communication rounds
        #
        # Alternative: Pipeline parallelism - process chunks sequentially but
        # overlap computation of different layers
        new_states = torch.zeros(batch_size,
                                 (seq_len + pad_size) // self.chunk_size + 1,
                                 self.num_heads, self.head_dim,
                                 self.ssm_state_size)

        # Extract final state and intermediate states
        states = new_states[:, :-1]  # [b, num_chunks, h, d, n]
        ssm_state = new_states[:, -1]  # [b, h, d, n] - final state for caching

        # =============================================================================
        # STEP 3i: State to Output (Y_off)
        # =============================================================================
        # C_times_states = (C[..., None, :] * states[:, :, None, ...])
        # Input C: [b, num_chunks, c, h, n]
        # Input states: [b, num_chunks, h, d, n]
        # After broadcast: [b, num_chunks, c, h, d, n]
        # Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
        # After sum over n: [b, num_chunks, c, h, d]
        # Complexity: O(b * num_chunks * c * h * d * n)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Chunks (num_chunks): FULLY PARALLEL (using precomputed states)
        # - Positions (c): FULLY PARALLEL
        # - Heads (h): FULLY PARALLEL
        # - Head_dim (d): FULLY PARALLEL
        # - State_size (n): PARALLEL with ALL_REDUCE (reduction dimension)
        #
        # TENSOR PARALLEL: Split b, num_chunks, c, h, d with no communication
        # Split n requires all_reduce after sum
        Y_off = torch.zeros(batch_size, (seq_len + pad_size) // self.chunk_size,
                            self.chunk_size, self.num_heads, self.head_dim)

        # =============================================================================
        # STEP 3j: Combine Intra-chunk and Inter-chunk Outputs
        # =============================================================================
        # y = Y_diag + Y_off
        # Both: [b, num_chunks, c, h, d]
        # Output: [b, num_chunks, c, h, d] -> [b, s, h, d] -> [b, s, i]
        # Complexity: O(b * s * i) for reshape
        # Parallel: FULLY PARALLEL (element-wise addition)
        y = Y_diag + Y_off  # [b, num_chunks, c, h, d]
        y = y.reshape(batch_size, -1, self.num_heads,
                      self.head_dim)  # [b, s_padded, h, d]
        y = y[:, :seq_len, :, :]  # Remove padding: [b, s, h, d]
        y = y.reshape(batch_size, seq_len, self.intermediate_size)  # [b, s, i]

        # =============================================================================
        # STEP 4: Output Projection (Linear Layer)
        # =============================================================================
        # contextualized_states = y @ out_proj.weight^T + out_proj.bias
        # Input:  [b, s, i]
        # Weight: [h_in, i]
        # Output: [b, s, h_in]
        # Complexity: O(b * s * i * h_in)
        #
        # PARALLELIZATION:
        # - Batch (b): FULLY PARALLEL
        # - Sequence (s): FULLY PARALLEL
        # - Input dim (i): PARALLEL with ALL_REDUCE (reduction dimension)
        # - Output dim (h_in): FULLY PARALLEL (row-parallel)
        #
        # TENSOR PARALLEL STRATEGIES:
        # 1. Column parallel on i: split weight columns, all_reduce after matmul
        # 2. Row parallel on h_in: split weight rows, no all_reduce needed
        # 3. Typically: in_proj is column-parallel, out_proj is row-parallel
        #    This minimizes communication (1 all_reduce per layer)
        contextualized_states = self.out_proj(y)  # [b, s, h_in]

        return contextualized_states

    def summarize_parallelization_strategies(self):
        """
        SUMMARY OF PARALLELIZATION STRATEGIES FOR MULTI-GPU DEPLOYMENT
        ================================================================

        DIMENSIONS AND THEIR PARALLELIZABILITY:

        1. BATCH (b) - EMBARRASSINGLY PARALLEL
           - Can split across GPUs with ZERO communication
           - Each GPU processes different examples
           - Strategy: Data Parallelism

        2. SEQUENCE (s) - MOSTLY PARALLEL with caveats
           - Linear layers: FULLY PARALLEL
           - Conv1d: Needs halo exchange (kernel_size-1 elements)
           - Attention-like ops: FULLY PARALLEL
           - Within chunks: FULLY PARALLEL
           - Across chunks: SEQUENTIAL (recurrence)
           - Strategy: Sequence Parallelism (limited by chunk recurrence)

        3. HEADS (h) - FULLY PARALLEL
           - All operations independent across heads
           - No communication needed
           - Strategy: Tensor Parallelism on head dimension

        4. HEAD_DIM (d) - PARALLEL (no reductions in this dim)
           - Can split with no all_reduce
           - Strategy: Tensor Parallelism on head_dim

        5. HIDDEN_DIM (h_in, i) - PARALLEL with ALL_REDUCE
           - Linear layers: reduction dimension
           - Need all_reduce when splitting this dimension
           - Strategy: Tensor Parallelism (column-parallel in, row-parallel out)

        6. STATE_SIZE (n) - PARALLEL with ALL_REDUCE
           - Reduction dimension in attention-like operations
           - Need all_reduce when computing G and Y_off
           - Strategy: Tensor Parallelism on state dimension

        7. NUM_CHUNKS - MOSTLY PARALLEL
           - Each chunk computation: FULLY PARALLEL
           - Chunk recurrence (Step 3h): SEQUENTIAL
           - Strategy: Pipeline or sequential processing

        8. CHUNK_SIZE (c) - MIXED
           - Cumsum/segment_sum: SEQUENTIAL within chunk
           - Other ops: PARALLEL
           - Cannot split within chunk effectively

        RECOMMENDED MULTI-GPU STRATEGIES:
        ==================================

        Strategy 1: TENSOR + DATA PARALLEL (Most Common)
        -------------------------------------------------
        - Split batch across data-parallel GPUs (no communication)
        - Within each data-parallel group, use tensor parallelism:
          * Split num_heads across GPUs (no communication in compute)
          * Column-parallel in_proj, row-parallel out_proj (1 all_reduce per layer)
        - Works well for moderate sequence lengths
        - Communication: O(b * s * h_in) per layer for all_reduce

        Strategy 2: SEQUENCE PARALLEL (For Very Long Sequences)
        --------------------------------------------------------
        - Split sequence dimension across GPUs
        - Requires:
          * Halo exchange for conv1d (small overhead)
          * Sequential processing of chunk recurrence (pipelined)
        - Best for: seq_len >> hidden_size
        - Communication: O(conv_kernel * features) for halo + pipeline latency

        Strategy 3: EXPERT PARALLEL (If MOE layers present)
        ---------------------------------------------------
        - Not shown in this code, but relevant for full model
        - Split experts across GPUs
        - All-to-all communication for routing

        Strategy 4: PIPELINE PARALLEL (For Very Large Models)
        -----------------------------------------------------
        - Split layers across GPUs
        - Process micro-batches in pipeline
        - Communication: O(b * s * h_in) per pipeline stage boundary

        CRITICAL BOTTLENECKS:
        =====================

        1. CHUNK RECURRENCE (Step 3h)
           - Sequential across chunks
           - Cannot parallelize without algorithmic changes
           - For long sequences with many chunks, this limits speedup
           - Mitigation: Use larger chunk_size (but increases memory)

        2. CUMSUM OPERATIONS
           - Sequential within each chunk
           - Limits parallelism to chunk_size granularity
           - Cannot split chunk_size dimension across GPUs

        3. ALL_REDUCE COMMUNICATION
           - Required when splitting reduction dimensions
           - Latency increases with number of GPUs
           - Bandwidth-bound for large tensors

        4. CONV1D HALO EXCHANGE
           - Required for sequence parallelism
           - Small overhead but adds latency

        OPTIMAL CONFIGURATION (Example for 8 GPUs):
        ===========================================
        - Use 4-way tensor parallelism on heads (split 64 heads -> 16 per GPU)
        - Use 2-way data parallelism on batch
        - Keep sequence on single GPU (if possible)
        - If sequence too long:
          * Use sequence parallelism with 2-4 way split
          * Accept chunk recurrence as sequential bottleneck

        This gives:
        - ~4x speedup from tensor parallelism (limited by all_reduce)
        - ~2x speedup from data parallelism (perfect scaling)
        - Total: ~6-7x speedup on 8 GPUs (75-85% efficiency)
        """


def main():
    """
    Example usage showing the tensor shapes through the forward pass.
    """
    import sys

    # Check if debug mode is requested
    debug = "--debug" in sys.argv

    mixer = NemotronHMamba2Mixer()

    # Example input
    batch_size = 4
    seq_len = 1024
    hidden_size = 4096

    input_states = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output = mixer.torch_forward_algebra_only(input_states, debug=debug)

    print(f"\n{'='*80}")
    print("NEMOTRON-H MAMBA2 MIXER - TENSOR ALGEBRA ANALYSIS")
    print(f"{'='*80}")
    print(f"\nInput shape:  {input_states.shape}")
    print(f"Output shape: {output.shape}")
    print("\nConfiguration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Num heads: {mixer.num_heads}")
    print(f"  - Head dim: {mixer.head_dim}")
    print(f"  - Intermediate size: {mixer.intermediate_size}")
    print(f"  - Chunk size: {mixer.chunk_size}")
    print(
        f"  - Num chunks: {(seq_len + mixer.chunk_size - 1) // mixer.chunk_size}"
    )
    print(f"\n{'='*80}")
    print("See docstrings in the code for detailed parallelization analysis.")
    print("Run with --debug flag to see intermediate tensor shapes.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
