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
import unittest

import torch

import tensorrt_llm
from tensorrt_llm.runtime.kv_cache_manager import (Block, BlocksManager,
                                                   GenerationSequence,
                                                   KVCacheManager)


class TestKVCacheManager(unittest.TestCase):
    _sizeof = {torch.float32: 4, torch.float16: 2, torch.int8: 1}

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def test_block(self):
        block = Block(block_idx=0, k_ptrs=[123321], v_ptrs=[321123])
        block.add_link()
        self.assertEqual(block.ref_count, 1)

        block.add_link()
        self.assertEqual(block.ref_count, 2)
        self.assertTrue(block.has_link())

        block.remove_link()
        self.assertEqual(block.ref_count, 1)

        block.remove_link()
        self.assertEqual(block.ref_count, 0)
        self.assertFalse(block.has_link())

        self.assertEqual(block.get_k_ptr(0), 123321)
        self.assertEqual(block.get_v_ptr(0), 321123)

    def test_sequence(self):
        seq = GenerationSequence(seq_idx=1, batch_idx=0)
        self.assertEqual(seq.get_batch_idx(), 0)
        self.assertEqual(seq.get_seq_idx(), 1)

        seq1 = GenerationSequence(seq_idx=1, batch_idx=1)
        seq2 = GenerationSequence(seq_idx=1, batch_idx=0)
        seq3 = GenerationSequence(seq_idx=0, batch_idx=0)

        self.assertNotEqual(seq, seq1)
        self.assertEqual(seq, seq2)
        self.assertNotEqual(seq, seq3)

    def allocate_blocks(self, manager, sequences, block_len):
        for _ in range(block_len):
            for seq in sequences:
                self.assertTrue(manager.has_free_block())
                manager.allocate(seq)
        # All blocks should be allocated by now
        self.assertFalse(manager.has_free_block())

    def verify_pointer_array(self,
                             manager,
                             sequences,
                             block_len,
                             total_blocks,
                             max_blocks_per_seq,
                             block_elts,
                             memory_pool,
                             pool_idx=0):
        pointers = manager.get_pointer_array(pool_idx, beam_width=1)

        self.assertEqual(pointers.shape,
                         torch.Size([len(sequences), 1, 2, max_blocks_per_seq]))

        # Check if pointer array is correct
        for seq in sequences:
            for block in range(block_len):
                linear_block_idx = (block * len(sequences) +
                                    seq.get_batch_idx())
                self.assertEqual(pointers[seq.get_batch_idx()][0][0][block], memory_pool.data_ptr() + \
                                 linear_block_idx * block_elts * self._sizeof[memory_pool.dtype])
                self.assertEqual(pointers[seq.get_batch_idx()][0][1][block], memory_pool.data_ptr() + \
                                 (linear_block_idx * block_elts + total_blocks * block_elts) * \
                                    self._sizeof[memory_pool.dtype])

    def free_blocks(self, manager, sequences, block_len):
        for seq in sequences:
            manager.free(seq)
            # We don't have double references to the blocks for now
            self.assertEqual(len(manager.free_blocks),
                             (seq.get_batch_idx() + 1) * block_len)

    def full_allocate_free_test(self, manager, sequences, block_len,
                                total_blocks, max_blocks_per_seq, block_elts,
                                memory_pool):
        self.allocate_blocks(manager, sequences, block_len)

        self.verify_pointer_array(manager, sequences, block_len, total_blocks,
                                  max_blocks_per_seq, block_elts, memory_pool)

        self.free_blocks(manager, sequences, block_len)

    def test_blocks_manager_single_pool(self):
        max_seq = 32
        max_blocks_per_seq = 32
        block_elts = 64
        memory_pool = torch.zeros(max_seq,
                                  2,
                                  max_blocks_per_seq,
                                  block_elts,
                                  dtype=torch.float,
                                  device='cuda')

        sequences = [
            GenerationSequence(seq_idx=idx, batch_idx=idx)
            for idx in range(max_seq)
        ]

        manager = BlocksManager(memory_pools=[memory_pool],
                                blocks=max_seq * max_blocks_per_seq,
                                max_blocks_per_seq=max_blocks_per_seq)

        self.assertEqual(len(manager.free_blocks), max_seq * max_blocks_per_seq)
        self.assertTrue(manager.has_free_block())

        # Allocate maximum amount of blocks for maximum amount of sequences
        self.full_allocate_free_test(manager, sequences, max_blocks_per_seq,
                                     max_seq * max_blocks_per_seq,
                                     max_blocks_per_seq, block_elts,
                                     memory_pool)

        manager = BlocksManager(memory_pools=[memory_pool],
                                blocks=max_seq * max_blocks_per_seq,
                                max_blocks_per_seq=max_blocks_per_seq)

        # Allocate 2x more sequences with 2 times smaller num of blocks
        sequences_2x = [
            GenerationSequence(seq_idx=idx, batch_idx=idx)
            for idx in range(2 * max_seq)
        ]
        self.full_allocate_free_test(manager, sequences_2x,
                                     max_blocks_per_seq // 2,
                                     max_seq * max_blocks_per_seq,
                                     max_blocks_per_seq, block_elts,
                                     memory_pool)

        manager = BlocksManager(memory_pools=[memory_pool],
                                blocks=max_seq * max_blocks_per_seq,
                                max_blocks_per_seq=max_blocks_per_seq)

        # Allocate maximum amount of blocks for maximum amount of sequences
        self.allocate_blocks(manager, sequences, max_blocks_per_seq)

        # Can't allocate more blocks
        with self.assertRaises(RuntimeError) as context:
            manager.allocate(sequences[0])
        self.assertEqual("Can't allocate new block for KV cache",
                         str(context.exception))

    def test_blocks_manager_multi_pool(self):
        max_seq = 32
        max_blocks_per_seq = 32
        block_elts_1 = 64
        block_elts_2 = 128
        memory_pool_1 = torch.zeros(max_seq,
                                    2,
                                    max_blocks_per_seq,
                                    block_elts_1,
                                    dtype=torch.float,
                                    device='cuda')
        memory_pool_2 = torch.zeros(max_seq,
                                    2,
                                    max_blocks_per_seq,
                                    block_elts_2,
                                    dtype=torch.float,
                                    device='cuda')

        sequences = [
            GenerationSequence(seq_idx=idx, batch_idx=idx)
            for idx in range(max_seq)
        ]

        manager = BlocksManager(memory_pools=[memory_pool_1, memory_pool_2],
                                blocks=max_seq * max_blocks_per_seq,
                                max_blocks_per_seq=max_blocks_per_seq)

        self.allocate_blocks(manager, sequences, max_blocks_per_seq)

        # Verify that pointers to the both pools are ok
        self.verify_pointer_array(manager,
                                  sequences,
                                  max_blocks_per_seq,
                                  max_seq * max_blocks_per_seq,
                                  max_blocks_per_seq,
                                  block_elts_1,
                                  memory_pool_1,
                                  pool_idx=0)
        self.verify_pointer_array(manager,
                                  sequences,
                                  max_blocks_per_seq,
                                  max_seq * max_blocks_per_seq,
                                  max_blocks_per_seq,
                                  block_elts_2,
                                  memory_pool_2,
                                  pool_idx=1)

    def test_blocks_manager_beam(self):
        max_seq = 32
        max_blocks_per_seq = 32
        block_elts = 64
        beam_width = 4
        blocks = max_seq * max_blocks_per_seq
        memory_pool = torch.zeros(max_seq * beam_width,
                                  2,
                                  max_blocks_per_seq,
                                  block_elts,
                                  dtype=torch.float,
                                  device='cuda')

        sequences = [
            GenerationSequence(seq_idx=idx, batch_idx=idx)
            for idx in range(max_seq)
        ]

        manager = BlocksManager(memory_pools=[memory_pool],
                                blocks=blocks,
                                max_blocks_per_seq=max_blocks_per_seq,
                                beam_width=beam_width)

        manager.allocate(sequences[0], share_across_beam=True)

        beams_blocks = manager.allocated_blocks[sequences[0]]
        self.assertEqual(beams_blocks[0][0].idx, beams_blocks[1][0].idx)
        self.assertEqual(beams_blocks[1][0].idx, beams_blocks[2][0].idx)
        self.assertEqual(beams_blocks[2][0].idx, beams_blocks[3][0].idx)
        self.assertEqual(beams_blocks[1][0].ref_count, beam_width)

        manager.allocate(sequences[1], share_across_beam=False)
        beams_blocks = manager.allocated_blocks[sequences[1]]
        self.assertNotEqual(beams_blocks[0][0].idx, beams_blocks[1][0].idx)
        self.assertNotEqual(beams_blocks[1][0].idx, beams_blocks[2][0].idx)
        self.assertNotEqual(beams_blocks[2][0].idx, beams_blocks[3][0].idx)
        self.assertEqual(beams_blocks[0][0].ref_count, 1)
        self.assertEqual(beams_blocks[1][0].ref_count, 1)
        self.assertEqual(beams_blocks[2][0].ref_count, 1)
        self.assertEqual(beams_blocks[3][0].ref_count, 1)

        manager.free(sequences[1])
        self.assertEqual(len(manager.free_blocks), blocks - 1)

        manager.free(sequences[0])
        self.assertEqual(len(manager.free_blocks), blocks)

    def test_kv_cache_manager(self):
        blocks = 128
        tokens_per_block = 32
        max_blocks_per_seq = 16
        dims_per_head_1 = 64
        dims_per_head_2 = 128
        memory_pool_1 = torch.zeros(2,
                                    blocks,
                                    tokens_per_block,
                                    dims_per_head_1,
                                    dtype=torch.float,
                                    device='cuda')
        memory_pool_2 = torch.zeros(2,
                                    blocks,
                                    tokens_per_block,
                                    dims_per_head_2,
                                    dtype=torch.float,
                                    device='cuda')
        manager = KVCacheManager(memory_pools=[memory_pool_1, memory_pool_2],
                                 blocks=blocks,
                                 tokens_per_block=tokens_per_block,
                                 max_blocks_per_seq=max_blocks_per_seq,
                                 max_attention_window_size=max_blocks_per_seq *
                                 tokens_per_block,
                                 sink_token_len=0)
        manager.add_sequence(GenerationSequence(seq_idx=0, batch_idx=0), 30)
        manager.add_sequence(GenerationSequence(seq_idx=1, batch_idx=1), 35)
        manager.add_sequence(GenerationSequence(seq_idx=2, batch_idx=2), 31)

        def check_amount_of_blocks(sequence, expected_blocks):
            for bi in range(max_blocks_per_seq):
                if bi < expected_blocks:
                    self.assertNotEqual(sequence[bi], 0)
                else:
                    self.assertEqual(sequence[bi], 0)

        arrays = manager.get_pointer_arrays(beam_width=1)
        arrays = [arr.view(dtype=torch.int64) for arr in arrays]

        # Expect 2 arrays for 2 memory pools
        self.assertEqual(len(arrays), 2)
        check_amount_of_blocks(arrays[0][0][0][0], 1)
        check_amount_of_blocks(arrays[0][1][0][0], 2)
        check_amount_of_blocks(arrays[0][2][0][0], 1)
        self.assertEqual(manager.lens[0], 30)
        self.assertEqual(manager.lens[1], 35)
        self.assertEqual(manager.lens[2], 31)

        # After this loop sequence 1 should have 33 tokens and 2 blocks
        for _ in range(3):
            manager.step([False, False, False])

        arrays = manager.get_pointer_arrays(beam_width=1)
        arrays = [arr.view(dtype=torch.int64) for arr in arrays]
        check_amount_of_blocks(arrays[0][0][0][0], 2)
        check_amount_of_blocks(arrays[0][1][0][0], 2)
        check_amount_of_blocks(arrays[0][2][0][0], 2)
        self.assertEqual(manager.lens[0], 33)
        self.assertEqual(manager.lens[1], 38)
        self.assertEqual(manager.lens[2], 34)

        # Second sequence finishes
        manager.step([False, True, False])

        self.assertEqual(len(manager.sequences), 2)
        self.assertEqual(len(manager.lens), 2)
        arrays = manager.get_pointer_arrays(beam_width=1)
        arrays = [arr.view(dtype=torch.int64) for arr in arrays]

        self.assertEqual(manager.lens[0], 34)
        self.assertEqual(manager.lens[1], 35)

        check_amount_of_blocks(arrays[0][0][0][0], 2)
        check_amount_of_blocks(arrays[0][1][0][0], 2)

        # Second sequence finishes
        manager.step([False, True])

        self.assertEqual(len(manager.sequences), 1)
        self.assertEqual(len(manager.lens), 1)
        arrays = manager.get_pointer_arrays(beam_width=1)
        arrays = [arr.view(dtype=torch.int64) for arr in arrays]

        self.assertEqual(manager.lens[0], 35)

        check_amount_of_blocks(arrays[0][0][0][0], 2)


if __name__ == '__main__':
    unittest.main()
