# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import defaultdict
from typing import List

import torch


class Block(object):

    def __init__(self, block_idx: int):
        self.idx = block_idx
        self.ref_count = 0

    def add_link(self):
        self.ref_count += 1

    def remove_link(self):
        self.ref_count -= 1

    def has_link(self) -> bool:
        return self.ref_count > 0

    def is_shared(self) -> bool:
        return self.ref_count > 1


class GenerationSequence(object):

    def __init__(self, seq_idx, batch_idx):
        self.seq_idx = seq_idx
        self.batch_idx = batch_idx

    def get_batch_idx(self) -> int:
        """
        Returns idx of sequence in batch
        """
        return self.batch_idx

    def get_seq_idx(self) -> int:
        """
        Returns sequence idx
        """
        return self.seq_idx

    def __eq__(self, another):
        return hasattr(another, 'seq_idx') and self.seq_idx == another.seq_idx and \
            hasattr(another, 'batch_idx') and self.batch_idx == another.batch_idx

    def __hash__(self):
        return self.seq_idx


class BlocksManager(object):
    _sizeof = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }

    def __init__(self,
                 *,
                 num_layers: int,
                 num_blocks: int,
                 block_size: int,
                 max_blocks_per_seq: int = 128,
                 beam_width: int = 1):
        """
        If layers are homogeneous then the expected block pool shape is: [num_blocks, num_layers, 2, block_size]
        Otherwise, the expected block pool shape is: [num_blocks, 2, block_size]
        """

        self.max_blocks_per_seq = max_blocks_per_seq

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.beam_width = beam_width

        self.free_blocks = []
        for bi in range(num_blocks):
            self.free_blocks.append(Block(bi))

        beam_width = self.beam_width
        # Here use beam_width instead of self.beam_width to remove cyclic reference between self and
        # self.allocated_blocks by preventing capture self, which may cause memory leak.
        self.allocated_blocks = defaultdict(
            lambda: [[] for _ in range(beam_width)])

    def has_free_block(self) -> bool:
        """
        Returns True if we have at least 1 free block
        """
        return len(self.free_blocks) > 0

    def allocate(self,
                 owner: GenerationSequence,
                 share_across_beam: bool = False):
        """
        Add block to owner and increase ref count
        """
        # Add blocks for whole beam width
        block = None
        for bi in range(self.beam_width):
            if not self.has_free_block():
                raise RuntimeError("Can't allocate new block for KV cache")

            # Use the same block for all seqs in beam if share_across_beam
            if block is None or share_across_beam == False:
                block = self.free_blocks.pop(0)
            # Add one reference to the block
            block.add_link()
            self.allocated_blocks[owner][bi].append(block)

    def replace_shared_block(self, owner: GenerationSequence, block_idx: int):
        """
        Replace the shared block.
        Free the shared block, and allocate blocks with share_across_beam=False
        """
        if not self.allocated_blocks[owner][0][block_idx].is_shared():
            return

        # Free shared block
        for bi in range(self.beam_width):
            block = self.allocated_blocks[owner][bi][block_idx]
            block.remove_link()
            if not block.has_link():
                self.free_blocks.append(block)

        # Allocate new block
        for bi in range(self.beam_width):
            if not self.has_free_block():
                raise RuntimeError("Can't allocate new block for KV cache")
            block = self.free_blocks.pop(0)
            block.add_link()
            self.allocated_blocks[owner][bi][block_idx] = block
        return

    def free(self, owner: GenerationSequence):
        """
        Unlink all blocks of given owner.
        Moves blocks with ref_count == 0 to free.
        Removes owner from allocated blocks.
        """
        for bi in range(self.beam_width):
            for block in self.allocated_blocks[owner][bi]:
                # Move block to free if no one refers to it
                block.remove_link()

                # Move block to free if no one refers to it
                if not block.has_link():
                    self.free_blocks.append(block)
        # Remove owner from allocated blocks
        self.allocated_blocks.pop(owner)

    def get_number_blocks(self, owner: GenerationSequence) -> int:
        """
        Returns number of blocks allocated to the sequence owner
        """
        return len(self.allocated_blocks[owner][0])

    def get_k_or_v_block_offset(self, block_idx, field_idx):
        """
        Get offset in memory pool to K or V block. field_idx should be 0 (K) or 1 (V).
        """
        return block_idx * self.num_layers * 2 + field_idx

    def get_offset_array(self, beam_width: int) -> torch.Tensor:
        """
        Returns array of [batch size, beam_width, 2, max_blocks_per_seq] of offsets
        to the allocated blocks in memory pool
        """
        assert (beam_width <= self.beam_width)

        def create_nested_list(dims):
            """Recursive function to generate nested list."""
            if len(dims) == 1:
                return [0 for _ in range(dims[0])]
            return [create_nested_list(dims[1:]) for _ in range(dims[0])]

        offset_array = create_nested_list(
            (len(self.allocated_blocks), beam_width, 2,
             self.max_blocks_per_seq))

        k_idx = 0
        v_idx = 1
        for owner, beams_blocks in self.allocated_blocks.items():
            for bi in range(beam_width):
                for block_linear_idx, block in enumerate(beams_blocks[bi]):
                    for x_idx in [k_idx, v_idx]:
                        offset_array[owner.get_batch_idx()][bi][x_idx][
                            block_linear_idx] = self.get_k_or_v_block_offset(
                                block.idx, x_idx)

        self.offset_array = torch.tensor(offset_array, dtype=torch.int32)
        return self.offset_array

    def get_continuous_caches(self, memory_pool: torch.Tensor) -> torch.Tensor:
        """
        Returns continuous KV caches.
        Used only for debug purposes.
        """
        assert self.beam_width == 1

        pool = memory_pool.flatten()
        continuous_kv_cache = torch.zeros(len(self.allocated_blocks),
                                          2,
                                          self.max_blocks_per_seq *
                                          self.block_size,
                                          dtype=pool.dtype,
                                          device="cuda")
        k_idx = 0
        v_idx = 1
        for owner, beam_blocks in self.allocated_blocks.items():
            for bi in range(self.beam_width):
                for block_linear_idx, block in enumerate(beam_blocks[bi]):
                    # The batch index.
                    batch_idx = owner.get_batch_idx()
                    # The first index in the sequence.
                    block_offset = block_linear_idx * self.block_size

                    for x_idx in [k_idx, v_idx]:
                        x_start = self.get_k_or_v_block_offset(
                            block.idx, x_idx) * self.block_size

                        continuous_kv_cache[batch_idx][
                            x_idx][block_offset:block_offset +
                                   self.block_size] = pool[x_start:x_start +
                                                           self.block_size]

        return continuous_kv_cache


class KVCacheManager(object):

    def __init__(self,
                 *,
                 num_layers: int,
                 num_blocks: int,
                 block_size: int,
                 tokens_per_block: int,
                 max_blocks_per_seq: int,
                 max_attention_window_size: int,
                 sink_token_len: int,
                 beam_width: int = 1,
                 use_one_more_block: bool = False):

        self.blocks_manager = BlocksManager(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
            beam_width=beam_width)

        self.tokens_per_block = tokens_per_block
        self.max_attention_window_size = max_attention_window_size
        self.sink_token_len = sink_token_len
        self.beam_width = beam_width

        # The sink tokens are not stored into the same block with other tokens.
        # Need to add the bubble after the sink tokens.
        if sink_token_len % tokens_per_block == 0:
            self.bubble_len = 0
        else:
            self.bubble_len = tokens_per_block - sink_token_len % tokens_per_block

        # Token num in the sink blocks
        self.sink_block_token_num = self.sink_token_len + self.bubble_len

        # Max token num in the cache
        self.max_token_num = self.max_attention_window_size + self.bubble_len
        if use_one_more_block:
            self.max_token_num += self.tokens_per_block

        self.lens = []
        self.sequences = []

    def step(self, finished: List[bool]):
        """
        Iterate to the next generation step.
        Add new blocks where needed and clear finished sequences.
        """
        for seq in self.sequences:
            batch_idx = seq.get_batch_idx()
            # Enable cyclic kv cache when it exceeds the max_token_num
            cyclic_token_num = self.max_token_num - self.sink_block_token_num
            next_token_idx_in_cache = self.sink_block_token_num + \
                        (self.lens[batch_idx] - self.sink_block_token_num) % cyclic_token_num
            if not finished[batch_idx] and (
                    next_token_idx_in_cache % self.tokens_per_block == 0 or
                (next_token_idx_in_cache - self.sink_block_token_num) %
                    cyclic_token_num == 0):
                if self.lens[batch_idx] < self.max_token_num:
                    self.blocks_manager.allocate(seq)
                elif self.beam_width > 1:
                    # Get next block index
                    next_block_idx = next_token_idx_in_cache // self.tokens_per_block
                    # Replace the shared block with the unshared ones
                    self.blocks_manager.replace_shared_block(
                        seq, next_block_idx)

            self.lens[batch_idx] += 1

        # Remove finished sequences
        for fi in range(len(finished)):
            if finished[fi]:
                self.blocks_manager.free(self.sequences[fi])
        self.lens = [l for l, f in zip(self.lens, finished) if not f]

        # Remap sequence ids
        new_sequences = []
        batch_idx = 0
        for seq, finish in zip(self.sequences, finished):
            if not finish:
                seq.batch_idx = batch_idx
                new_sequences.append(seq)
                batch_idx += 1
        self.sequences = new_sequences

    def add_sequence(self,
                     sequence: GenerationSequence,
                     context_len: int,
                     always_share_across_beam: bool = False):
        """
        Add sequence to the manager and allocate minimum amount of blocks for context
        """
        seq_len = context_len + self.bubble_len
        self.lens.append(seq_len)
        self.sequences.append(sequence)

        # Enable cyclic kv cache when inputLength exceeds maxAttentionWindow.
        # Note that currently cyclic kv cache doesn't work with shared kv cache of different beams.
        enable_cyclic_kv_cache = seq_len >= self.max_token_num

        # Get the final token index in kv cache
        final_token_kv_index = self.sink_block_token_num + (
            (seq_len - 1 - self.sink_block_token_num) %
            (self.max_token_num - self.sink_block_token_num))

        # Get block index that with shareAmongBeams=False.
        unshared_block_idx = -1
        if (not enable_cyclic_kv_cache or self.beam_width > 1
                or final_token_kv_index % self.tokens_per_block > 0):
            unshared_block_idx = final_token_kv_index // self.tokens_per_block + 1 if (
                final_token_kv_index + 1
            ) % self.tokens_per_block == 0 else final_token_kv_index // self.tokens_per_block

        # Get context block num.
        # Allocate one more block if there are tokens that can't be shared across beams.
        seq_len = min(seq_len, self.max_token_num)
        context_blocks = seq_len // self.tokens_per_block
        if seq_len % self.tokens_per_block > 0:
            context_blocks += 1

        # Allocate blocks
        for i in range(context_blocks):
            self.blocks_manager.allocate(
                sequence,
                share_across_beam=True if always_share_across_beam else
                (i != unshared_block_idx))

    def get_block_offsets(self, beam_width: int) -> torch.Tensor:
        """
        Returns array of offsets into memory pools
        """
        return self.blocks_manager.get_offset_array(beam_width)


class KVCacheUpdater:

    def __init__(self):
        self.use_paged_kv_cache = None
        self.num_layers = None
        self.num_kv_heads = None
        self.head_dim = None
        self.elt_size = None
        self.past_key_value_list = None
        self.max_kv_cache_length = None
        self.kv_cache_manager = None
        self.host_kv_cache_pool_pointers = None

    def init_linear_kv_cache(self, num_layers, num_kv_heads, head_dim,
                             kv_cache_type, past_key_value_list):
        self.use_paged_kv_cache = False
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.past_key_value_list = past_key_value_list
        self.elt_size = torch.zeros(1, dtype=kv_cache_type).element_size()
        self.max_kv_cache_length = past_key_value_list[0].shape[3]

    def init_paged_kv_cache(self, num_layers, num_kv_heads, head_dim,
                            kv_cache_type, kv_cache_manager,
                            host_kv_cache_pool_pointers):
        self.use_paged_kv_cache = True
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_cache_manager = kv_cache_manager
        self.host_kv_cache_pool_pointers = host_kv_cache_pool_pointers
        self.elt_size = torch.zeros(1, dtype=kv_cache_type).element_size()

    def update(self, accepted_draft_token_offsets,
               packed_accepted_draft_tokens_indices, sequence_length_buffer,
               rewind_tokens):
        assert isinstance(rewind_tokens, torch.Tensor) or isinstance(
            rewind_tokens, int)
        rewind_tokens_tensor = rewind_tokens if isinstance(
            rewind_tokens, torch.Tensor) else None
        rewind_tokens_count = rewind_tokens if isinstance(rewind_tokens,
                                                          int) else 0
        assert self.use_paged_kv_cache is not None
        if self.use_paged_kv_cache:
            if self.kv_cache_manager.has_single_pool():
                kv_cache_manager = self.kv_cache_manager.get_single_kv_cache_manager(
                )
            else:
                raise RuntimeError(
                    "Currently, using KVCacheUpdater with more then single memory pool is not supported"
                )

            host_kv_cache_block_offsets = kv_cache_manager.get_block_offsets(1)
            kv_cache_block_offsets = host_kv_cache_block_offsets.to('cuda')
            torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
                accepted_draft_token_offsets,
                packed_accepted_draft_tokens_indices,
                sequence_length_buffer,
                True,
                self.num_layers,
                self.num_kv_heads,
                self.head_dim * self.elt_size,
                rewind_tokens_count,
                kv_cache_manager.max_attention_window_size,
                rewind_tokens_tensor,
                None,
                self.host_kv_cache_pool_pointers,
                kv_cache_block_offsets,
                kv_cache_manager.blocks_manager.max_blocks_per_seq,
                kv_cache_manager.tokens_per_block,
                None,
            )
        else:
            torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
                accepted_draft_token_offsets,
                packed_accepted_draft_tokens_indices,
                sequence_length_buffer,
                False,
                self.num_layers,
                self.num_kv_heads,
                self.head_dim * self.elt_size,
                rewind_tokens_count,
                self.max_kv_cache_length,
                rewind_tokens_tensor,
                self.past_key_value_list,
                None,
                None,
                None,
                None,
                None,
            )
