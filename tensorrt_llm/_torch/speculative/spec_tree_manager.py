import logging
import math

import torch

from tensorrt_llm._utils import prefer_pinned

logger = logging.getLogger(__name__)


class DynamicTreeSlotStorage:
    """Per-slot GPU storage for dynamic tree data, indexed by py_seq_slot.

    Buffers are [S, ...] where S = num_slots + 1 (+1 for CUDA graph dummy).
    """

    def __init__(self,
                 num_slots: int,
                 n_dt: int,
                 mask_width: int,
                 top_k: int = 1):
        S = num_slots + 1
        self.dummy_slot_id = num_slots

        # Bootstrap/reused slots may not have a tree yet; keep their metadata
        # as a valid linear chain so verification kernels can read it directly.
        no_tree_position_offsets, no_tree_packed_mask = self._make_kary_tree_metadata(
            n_dt, mask_width, top_k=1)
        self.position_offsets = no_tree_position_offsets.unsqueeze(0).repeat(
            S, 1).contiguous()
        self.packed_mask = no_tree_packed_mask.unsqueeze(0).repeat(
            S, 1, 1).contiguous()
        self._no_tree_position_offsets = no_tree_position_offsets
        self._no_tree_packed_mask = no_tree_packed_mask

        # CUDA-graph dummies use a deterministic K-ary tree, matching real
        # dynamic-tree mask/position shapes without depending on request state.
        dummy_position_offsets, dummy_packed_mask = self._make_kary_tree_metadata(
            n_dt, mask_width, top_k)
        self.position_offsets[self.dummy_slot_id] = dummy_position_offsets
        self.packed_mask[self.dummy_slot_id] = dummy_packed_mask
        self.retrieve_index = torch.zeros((S, n_dt),
                                          dtype=torch.int32,
                                          device='cuda')

        # Mamba verify reads next links unconditionally, so no-tree rows must be
        # valid linear chains instead of sentinels.
        self._no_tree_next_token = self._make_no_tree_next_token(n_dt)
        self.retrieve_next_token = self._no_tree_next_token.unsqueeze(0).repeat(
            S, 1)
        self.retrieve_next_sibling = torch.full((S, n_dt),
                                                -1,
                                                dtype=torch.int32,
                                                device='cuda')
        self.has_tree = torch.zeros(S, dtype=torch.bool, device='cuda')

        # Slot-ID buffers
        self.all_ids_buf = torch.zeros(num_slots,
                                       dtype=torch.long,
                                       device='cuda')
        self._pin_batch = torch.empty(num_slots,
                                      dtype=torch.long,
                                      pin_memory=prefer_pinned())
        self._verify_staging = torch.empty((num_slots, n_dt, 3),
                                           dtype=torch.int32,
                                           device='cuda')
        self._next_token_staging = torch.empty((num_slots, n_dt),
                                               dtype=torch.int32,
                                               device='cuda')
        self._next_sibling_staging = torch.empty((num_slots, n_dt),
                                                 dtype=torch.int32,
                                                 device='cuda')

    @staticmethod
    def _make_kary_tree_metadata(
            n_dt: int, mask_width: int,
            top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = max(int(top_k), 1)
        token_ids = torch.arange(n_dt, device='cuda')
        parents = torch.where(token_ids > 0, (token_ids - 1) // top_k,
                              token_ids)
        ancestor_chain = torch.empty((n_dt, n_dt),
                                     dtype=torch.long,
                                     device='cuda')
        current = token_ids
        for depth in range(n_dt):
            ancestor_chain[:, depth] = current
            current = parents[current]

        # Pack bits directly from the parent chain instead of materializing a
        # dense bool mask and repacking it.
        valid_ancestors = torch.ones((n_dt, n_dt),
                                     dtype=torch.bool,
                                     device='cuda')
        valid_ancestors[:, 1:] = ancestor_chain[:, 1:] != ancestor_chain[:, :-1]
        bit_values = (1 << (ancestor_chain % 32)).to(torch.int32)
        bit_values.masked_fill_(~valid_ancestors, 0)
        packed_mask = torch.zeros((n_dt, mask_width),
                                  dtype=torch.int32,
                                  device='cuda')
        packed_mask.scatter_add_(1, ancestor_chain // 32, bit_values)
        position_offsets = valid_ancestors.sum(-1).to(torch.int32) - 1
        return position_offsets, packed_mask

    @staticmethod
    def _make_no_tree_next_token(n_dt: int) -> torch.Tensor:
        next_token = torch.arange(1, n_dt + 1, dtype=torch.int32, device='cuda')
        next_token[n_dt - 1] = -1
        return next_token

    def fill_all_slot_ids(self, context_requests, generation_requests):
        """Fill all_ids_buf for full batch [ctx | gen] via one HtoD copy."""
        dummy_slot = self.dummy_slot_id
        pin = self._pin_batch
        cursor = 0
        for req in context_requests:
            pin[cursor] = req.py_seq_slot if req.py_seq_slot is not None else dummy_slot
            cursor += 1
        for req in generation_requests:
            slot = req.py_seq_slot if (
                not getattr(req, 'is_cuda_graph_dummy', False)
                and req.py_seq_slot is not None) else dummy_slot
            pin[cursor] = slot
            cursor += 1
        if cursor > 0:
            self.all_ids_buf[:cursor].copy_(pin[:cursor], non_blocking=True)

    def mark_valid(self, slot_ids, count):
        if count == 0:
            return
        self.has_tree.index_fill_(0, slot_ids[:count], True)
        self.has_tree.narrow(0, self.dummy_slot_id, 1).fill_(False)

    def mark_invalid(self, slot_id):
        """Clear validity and restore valid no-tree metadata."""
        self.has_tree[slot_id] = False
        self.packed_mask[slot_id] = self._no_tree_packed_mask
        self.position_offsets[slot_id] = self._no_tree_position_offsets
        self.retrieve_index[slot_id] = 0
        self.retrieve_next_token[slot_id] = self._no_tree_next_token
        self.retrieve_next_sibling[slot_id] = -1

    def pack_retrieve_from_slots(self, slot_ids, count):
        """Pack retrieve data into [count, n_dt, 3] staging buffer."""
        if count == 0:
            return self._verify_staging[:0]
        ids = slot_ids[:count]
        staging = self._verify_staging[:count]
        staging[:, :, 0] = self.retrieve_index[ids]
        staging[:, :, 1] = self.retrieve_next_token[ids]
        staging[:, :, 2] = self.retrieve_next_sibling[ids]
        return staging

    def next_links_from_slots(self, slot_ids, count):
        """Gather next-token and next-sibling links into contiguous staging buffers."""
        if count == 0:
            return self._next_token_staging[:0], self._next_sibling_staging[:0]
        ids = slot_ids[:count]
        next_token = self._next_token_staging[:count]
        next_sibling = self._next_sibling_staging[:count]
        torch.index_select(self.retrieve_next_token, 0, ids, out=next_token)
        torch.index_select(self.retrieve_next_sibling, 0, ids, out=next_sibling)
        return next_token, next_sibling


class SpecTreeManager:
    """Per-request tree metadata for dynamic tree (Eagle-2 style) speculation."""

    max_total_draft_tokens: int  # The number of all nodes in the tree (except the root)
    dynamic_tree_max_topK: int  # The number of nodes to expand each time.
    max_draft_len: int  # The number of drafter layer.

    # Auxiliary buffers
    # The top k  list for each draft layer.
    top_k_list: list
    # Each request has their own tree.
    num_trees: int

    # The packed decoding mask for the target model to verify the draft tokens. Pad the 0-1 matrix to int32 vector.
    # shape: [num_trees, max_total_draft_tokens + 1], device tensor.
    spec_dec_packed_mask: torch.Tensor = None

    # The spec position offsets for the target model to verify the draft tokens.
    # shape: [num_trees, max_total_draft_tokens + 1], device tensor.
    spec_dec_position_offsets: torch.Tensor = None

    # Work buffers for dynamic tree build kernel output
    retrieve_index: torch.Tensor = None
    retrieve_next_token: torch.Tensor = None
    retrieve_next_sibling: torch.Tensor = None
    slot_storage: 'DynamicTreeSlotStorage | None' = None

    def __init__(self, max_num_requests: int, max_total_draft_tokens: int,
                 max_draft_len: int, dynamic_tree_max_topK: int):

        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len

        # The draft loop can produce up to K * max_draft_len tokens, which may
        # exceed max_total_draft_tokens+1. Size the working buffers to the
        # larger of the two so the masks and position-offset tensors never run
        # out of columns/rows.
        if dynamic_tree_max_topK and dynamic_tree_max_topK > 0:
            self._internal_buf_dim = max(max_total_draft_tokens + 1,
                                         dynamic_tree_max_topK * max_draft_len)
        else:
            self._internal_buf_dim = max_total_draft_tokens + 1
        self.num_trees = max_num_requests
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.top_k_list = []

        n_dt = self.max_total_draft_tokens + 1
        self.spec_dec_packed_mask = torch.zeros(
            (self.num_trees, n_dt, math.ceil(n_dt / 32)),
            dtype=torch.int32,
            device='cuda',
        )
        self.spec_dec_position_offsets = torch.zeros(
            (self.num_trees, n_dt),
            dtype=torch.int32,
            device='cuda',
        )

        # Cached constants for compute_spec_dec_packed_mask (avoids per-call allocation)
        self._pack_weights = (
            1 << torch.arange(32, device='cuda', dtype=torch.int32))
        # Python-only internal buffers — enlarged to _internal_buf_dim
        num_blocks = math.ceil(self._internal_buf_dim / 32)
        total_bits = num_blocks * 32
        self._padded_mask_buf = torch.zeros(self.num_trees,
                                            self._internal_buf_dim,
                                            total_bits,
                                            dtype=torch.int32,
                                            device='cuda')
        self._pack_result_buf = torch.zeros(self.num_trees,
                                            self._internal_buf_dim,
                                            num_blocks,
                                            dtype=torch.int32,
                                            device='cuda')

        self.init_tree_info_for_dynamic_tree()

    def init_tree_info_for_dynamic_tree(self):
        num_draft_with_root = self.max_total_draft_tokens + 1

        self.top_k_list = [
            torch.ones(self.dynamic_tree_max_topK,
                       dtype=torch.int32,
                       device='cpu',
                       pin_memory=prefer_pinned()) * self.dynamic_tree_max_topK
        ]

        # Work buffers for build_dynamic_tree kernel output
        self.retrieve_index = torch.zeros((self.num_trees, num_draft_with_root),
                                          dtype=torch.int32,
                                          device='cuda')
        self.retrieve_next_token = torch.full(
            (self.num_trees, num_draft_with_root),
            -1,
            dtype=torch.int32,
            device='cuda')
        self.retrieve_next_sibling = torch.full(
            (self.num_trees, num_draft_with_root),
            -1,
            dtype=torch.int32,
            device='cuda')

        mask_width = math.ceil(num_draft_with_root / 32)
        self.slot_storage = DynamicTreeSlotStorage(
            num_slots=self.num_trees,
            n_dt=num_draft_with_root,
            mask_width=mask_width,
            top_k=self.dynamic_tree_max_topK,
        )

    def scatter_to_slot_storage(self, ss, gen_slots, num_gens):
        """Scatter work buffers to slot storage via index_copy_."""
        if num_gens == 0:
            return
        ids = gen_slots[:num_gens]
        ss.packed_mask.index_copy_(0, ids, self.spec_dec_packed_mask[:num_gens])
        ss.position_offsets.index_copy_(
            0, ids, self.spec_dec_position_offsets[:num_gens])
        ss.retrieve_index.index_copy_(0, ids, self.retrieve_index[:num_gens])
        ss.retrieve_next_token.index_copy_(0, ids,
                                           self.retrieve_next_token[:num_gens])
        ss.retrieve_next_sibling.index_copy_(
            0, ids, self.retrieve_next_sibling[:num_gens])
        ss.mark_valid(ids, num_gens)

    def compute_spec_dec_packed_mask(self, mask_matrix, packed_mask):
        bs, num_tokens, num_tokens_attend = mask_matrix.shape
        assert mask_matrix.ndim == 3, f"Expected 3D mask_matrix, got {mask_matrix.ndim}D"
        assert packed_mask.ndim == 3, f"Expected 3D packed_mask, got {packed_mask.ndim}D"
        assert bs <= self._padded_mask_buf.shape[0], \
            f"batch size {bs} exceeds pre-allocated buffer size {self._padded_mask_buf.shape[0]}"
        num_blocks = packed_mask.shape[-1]

        # Use cached bit weights
        weights = self._pack_weights
        src = mask_matrix if mask_matrix.dtype == torch.int32 else mask_matrix.to(
            torch.int32)

        if num_blocks == 1 and num_tokens_attend <= 32:
            result = self._pack_result_buf[:bs, :num_tokens, :1]
            torch.sum(src * weights[:num_tokens_attend],
                      dim=-1,
                      out=result[:, :, 0])
            packed_mask[:, :num_tokens, :1] = result
            return packed_mask

        # Pad into pre-allocated buffer
        total_bits = num_blocks * 32
        padded_m = self._padded_mask_buf[:bs, :num_tokens, :total_bits]
        padded_m.zero_()
        padded_m[:, :, :num_tokens_attend].copy_(src)

        # Reshape last dim into [num_blocks, 32] for blocked packing
        blocked_matrix = padded_m.view(bs, num_tokens, num_blocks, 32)

        # Vectorized dot product into pre-allocated result buffer
        result = self._pack_result_buf[:bs, :num_tokens, :num_blocks]
        torch.sum(blocked_matrix * weights, dim=-1, out=result)

        # Write results back to the output buffer
        packed_mask[:, :num_tokens, :] = result
        return packed_mask

    def dump_tree_info(self):
        logger.debug("TopK list: %s", self.top_k_list)
        logger.debug("Dynamic max top k: %s", self.dynamic_tree_max_topK)
