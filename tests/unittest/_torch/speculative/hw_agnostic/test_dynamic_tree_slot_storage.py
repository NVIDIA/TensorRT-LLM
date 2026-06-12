"""Unit tests for DynamicTreeSlotStorage no-tree link fallback.

The Mamba tree-aware verify conv1d/SSU path reads retrieve_next_token /
retrieve_next_sibling unconditionally (it does not gate on has_tree).  For
gen slots without a built tree (CUDA-graph/warmup dummies, and a real slot's
first decode), the gathered links are sentinels; apply_no_tree_linear_chain
must replace those rows with a valid degenerate linear chain so the kernel
indexing stays in-bounds.  Real-tree rows must be left untouched.
"""

import unittest

import pytest
import torch

from tensorrt_llm._torch.speculative.spec_tree_manager import DynamicTreeSlotStorage


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DynamicTreeSlotStorage allocates CUDA buffers"
)
class TestNoTreeLinearChain(unittest.TestCase):
    def _make(self, num_slots=4, n_dt=6):
        # mask_width is unused by the link path; any positive value is fine.
        return DynamicTreeSlotStorage(num_slots=num_slots, n_dt=n_dt, mask_width=1)

    def test_chain_template_is_valid(self):
        n_dt = 6
        ss = self._make(n_dt=n_dt)
        # next_token[i] = i+1, leaf has no child (-1).
        expected = torch.tensor([1, 2, 3, 4, 5, -1], dtype=torch.int32, device="cuda")
        self.assertTrue(torch.equal(ss._no_tree_next_token, expected))

    def test_no_tree_rows_get_chain_real_rows_untouched(self):
        num_slots, n_dt = 4, 6
        ss = self._make(num_slots=num_slots, n_dt=n_dt)

        # Slot 1 has a (synthetic) real tree; mark it valid and give it links
        # distinct from the chain so we can detect any accidental overwrite.
        real_next_token = torch.full((n_dt,), 3, dtype=torch.int32, device="cuda")
        real_next_sibling = torch.full((n_dt,), 2, dtype=torch.int32, device="cuda")
        ss.retrieve_next_token[1] = real_next_token
        ss.retrieve_next_sibling[1] = real_next_sibling
        ss.has_tree[1] = True

        # Gather slots [dummy, real, dummy] -> rows 0 and 2 are no-tree.
        slot_ids = torch.tensor(
            [ss.dummy_slot_id, 1, ss.dummy_slot_id], dtype=torch.long, device="cuda"
        )
        count = 3
        next_token, next_sibling = ss.next_links_from_slots(slot_ids, count)
        ss.apply_no_tree_linear_chain(next_token, next_sibling, slot_ids, count)

        chain = ss._no_tree_next_token
        # No-tree rows -> linear chain, sibling all -1.
        self.assertTrue(torch.equal(next_token[0], chain))
        self.assertTrue(torch.equal(next_token[2], chain))
        self.assertTrue(
            torch.equal(next_sibling[0], torch.full((n_dt,), -1, dtype=torch.int32, device="cuda"))
        )
        self.assertTrue(
            torch.equal(next_sibling[2], torch.full((n_dt,), -1, dtype=torch.int32, device="cuda"))
        )
        # Real-tree row -> original links preserved.
        self.assertTrue(torch.equal(next_token[1], real_next_token))
        self.assertTrue(torch.equal(next_sibling[1], real_next_sibling))

    def test_chain_links_are_in_bounds(self):
        # Every link is either a valid token index in [0, n_dt) or the -1 stop
        # sentinel; nothing points outside the per-request token range.
        n_dt = 8
        ss = self._make(n_dt=n_dt)
        slot_ids = torch.tensor([ss.dummy_slot_id], dtype=torch.long, device="cuda")
        next_token, next_sibling = ss.next_links_from_slots(slot_ids, 1)
        ss.apply_no_tree_linear_chain(next_token, next_sibling, slot_ids, 1)
        valid = (next_token == -1) | ((next_token >= 0) & (next_token < n_dt))
        self.assertTrue(bool(valid.all().item()))


if __name__ == "__main__":
    unittest.main()
