import logging
import math
from itertools import accumulate
from typing import List

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
        no_tree_position_offsets, no_tree_packed_mask = self._make_no_tree_metadata(
            n_dt, mask_width)
        self.position_offsets = no_tree_position_offsets.unsqueeze(0).repeat(
            S, 1).contiguous()
        self.packed_mask = no_tree_packed_mask.unsqueeze(0).repeat(
            S, 1, 1).contiguous()
        self._no_tree_position_offsets = no_tree_position_offsets
        self._no_tree_packed_mask = no_tree_packed_mask

        # CUDA-graph dummies use a deterministic K-ary tree, matching real
        # dynamic-tree mask/position shapes without depending on request state.
        dummy_position_offsets, dummy_packed_mask = self._make_dummy_tree_metadata(
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
    def _pack_bool_mask(mask: torch.Tensor, mask_width: int) -> torch.Tensor:
        """Pack a bool attention mask into int32 words."""
        num_rows, num_bits = mask.shape
        total_bits = mask_width * 32
        padded_mask = torch.zeros((num_rows, total_bits),
                                  dtype=torch.bool,
                                  device=mask.device)
        padded_mask[:, :num_bits] = mask
        weights = 2**torch.arange(32, dtype=torch.int64, device=mask.device)
        return (padded_mask.view(num_rows, mask_width, 32).to(torch.int64) *
                weights).sum(-1).to(torch.int32)

    @classmethod
    def _make_no_tree_metadata(
            cls, n_dt: int,
            mask_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.arange(n_dt, device='cuda')
        position_offsets = token_ids.to(torch.int32)
        causal_mask = token_ids.unsqueeze(1) >= token_ids.unsqueeze(0)
        return position_offsets, cls._pack_bool_mask(causal_mask, mask_width)

    @classmethod
    def _make_dummy_tree_metadata(
            cls, n_dt: int, mask_width: int,
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
        # Avoid advanced-indexing temporaries on the verification path.
        torch.index_select(self.retrieve_index, 0, ids, out=staging[:, :, 0])
        torch.index_select(self.retrieve_next_token,
                           0,
                           ids,
                           out=staging[:, :, 1])
        torch.index_select(self.retrieve_next_sibling,
                           0,
                           ids,
                           out=staging[:, :, 2])
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
    use_dynamic_tree: bool  # Whether using dynamic tree
    max_total_draft_tokens: int  # The number of all nodes in the tree (except the root)
    dynamic_tree_max_topK: int  # If using dynamic tree, the number of nodes to expand each time.
    max_draft_len: int  # The number of drafter layer. When using linear-tree, the max_draft_len is the same as max_total_draft_tokens.
    cur_draft_layer_idx: int  # The current index of the drafter layer

    # Auxiliary buffers
    # The top k  list for each draft layer.
    top_k_list: list
    # The user input eagle choices, only available when using static tree.
    eagle_choices: List[List[int]]
    # If dynamic tree, each request has their own tree. If static tree, all requests share the same tree.
    num_trees: int

    # Convert the choice to a path. Each path is an array of indices from the root to other nodes in the tree.
    # shape: [num_trees, max_total_draft_tokens + 1, max_draft_len + 1]
    eagle_paths: torch.Tensor = None

    # The spec decoding mask. Include the root node.
    # shape: [num_trees, max_total_draft_tokens + 1, max_total_draft_tokens + 1], device tensor.
    spec_dec_mask_matrix: torch.Tensor = None

    # The packed decoding mask for the target model to verify the draft tokens. Pad the 0-1 matrix to int32 vector.
    # shape: [num_trees, max_total_draft_tokens + 1], device tensor.
    spec_dec_packed_mask: torch.Tensor = None

    # The spec position offsets for the target model to verify the draft tokens.
    # shape: [num_trees, max_total_draft_tokens + 1], device tensor.
    spec_dec_position_offsets: torch.Tensor = None

    ############################ Auxiliary buffers for the static tree. ############################
    # Considering that the static tree does not modify the tree structure during inference, we can calculate some buffers in advance.
    # NOTE: Most of these buffers are introduced due to limitations of XQA:
    #       With tree attention, XQA cannot simply take the tokens to be processed in the next round as input. Instead, it needs to take ALL of their parent nodes as input.
    #       This incurs additional computation, but it is unavoidable.

    # NOTE: The reason why most of these auxiliary buffers are with `len == max_draft_len - 1` is that: we do not need to prepare specific input data for the first draft layer.
    # The top k value for each draft layer. Device tensor.
    top_k_list_cuda: list[torch.Tensor] = None

    # The max top k value for all draft layers. Which is used for torch.topk and cuda graph.
    max_top_k = -1

    # Gather the required draft tokens among the 'max_total_draft_tokens + 1' tokens.
    # Only the nodes has child(s) this layer and all their parents nodes will be gathered.
    tokens_gather_idx_for_drafter_model: list[torch.Tensor] = None

    # The packed mask for the drafter model's attention (i.e., xqa).
    # shape: [1, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32)], device tensor.
    spec_dec_packed_mask_for_drafter_model: torch.Tensor = None

    # The read indices offset for the drafter model.
    # shape: [max_total_draft_tokens + 1], device tensor.
    hidden_states_read_indices_offset_for_drafter_model: torch.Tensor = None

    # The write back start indices for the drafter tokens between different draft layers.
    # shape: [max_draft_len + 1], device tensor.
    draft_tokens_indices_cumsum: torch.Tensor = None

    ############################ Auxiliary buffers for the dynamic tree. ############################
    # CUDA kernel outputs for dynamic tree verification.
    # These are produced by build_dynamic_tree CUDA kernel and used by verify_dynamic_tree_greedy.
    # shape: [num_trees, max_total_draft_tokens + 1], int32, device tensor.
    retrieve_index: torch.Tensor = None
    retrieve_next_token: torch.Tensor = None
    retrieve_next_sibling: torch.Tensor = None
    slot_storage: 'DynamicTreeSlotStorage | None' = None

    def __init__(self, max_num_requests: int, use_dynamic_tree: bool,
                 max_total_draft_tokens: int, max_draft_len: int,
                 eagle_choices: List[List[int]] | None,
                 dynamic_tree_max_topK: int):

        self.use_dynamic_tree = use_dynamic_tree
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len

        # In dynamic tree mode the draft loop can produce up to
        # K * max_draft_len tokens, which may exceed max_total_draft_tokens+1.
        # Size the working buffers to the larger of the two so the masks and
        # position-offset tensors never run out of columns/rows.
        if use_dynamic_tree and dynamic_tree_max_topK > 0:
            self._internal_buf_dim = max(max_total_draft_tokens + 1,
                                         dynamic_tree_max_topK * max_draft_len)
        else:
            self._internal_buf_dim = max_total_draft_tokens + 1
        self.eagle_choices = eagle_choices
        self.num_trees = max_num_requests if use_dynamic_tree else 1
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.cur_draft_layer_idx = 0
        self.top_k_list = []

        # Initialize the buffers
        # eagle_paths and spec_dec_mask_matrix are only used in static tree mode.
        # Skip allocation in dynamic tree mode to save memory.
        if not use_dynamic_tree:
            self.eagle_paths = torch.ones(
                (self.num_trees, self.max_total_draft_tokens + 1,
                 self.max_draft_len + 1),
                dtype=torch.int32,
                device='cpu',
                pin_memory=prefer_pinned(),
            ) * -1

            self.spec_dec_mask_matrix = torch.eye(
                self.max_total_draft_tokens + 1,
                dtype=torch.int32,
                device='cuda',
            ).unsqueeze(0).repeat(self.num_trees, 1, 1)

        # CUDA kernel facing — rows = max_total_draft_tokens + 1,
        # columns widened to match attn_metadata mask_width so that the
        # Hopper flat copy in update_spec_dec_param needs no per-row padding.
        self.spec_dec_packed_mask = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1,
             math.ceil(self._internal_buf_dim / 32)),
            dtype=torch.int32,
            device='cuda',
        )
        self.spec_dec_position_offsets = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1),
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

        if self.use_dynamic_tree:
            self.init_tree_info_for_dynamic_tree()
        else:
            self.init_tree_info_for_static_tree()

    def init_tree_info_for_dynamic_tree(self):
        # Allocate retrieve buffers for CUDA kernel outputs
        num_draft_with_root = self.max_total_draft_tokens + 1
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

        # For the dynamic tree
        # To the internal layer, the number of nodes is the same as the dynamic_tree_max_topK.
        self.top_k_list = [
            torch.ones(self.dynamic_tree_max_topK,
                       dtype=torch.int32,
                       device='cpu',
                       pin_memory=prefer_pinned()) * self.dynamic_tree_max_topK
        ]

        self.slot_storage = DynamicTreeSlotStorage(
            num_slots=self.num_trees,
            n_dt=num_draft_with_root,
            mask_width=self.spec_dec_packed_mask.shape[-1],
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

    def init_tree_info_for_static_tree(self):
        self.index_mapping_set = {}
        self.nodes_list_per_layer = [[] for _ in range(self.max_draft_len + 1)]
        child_nodes_list = [[] for _ in range(self.max_total_draft_tokens + 1)]

        # 1) Map the index
        for i, choice in enumerate(self.eagle_choices):
            self.index_mapping_set[str(choice)] = i + 1

        # 2) Reconstruct the eagle_paths
        self.eagle_paths.fill_(-1)
        self.eagle_paths[0][0][0] = 0  # root node
        for i, choice in enumerate(self.eagle_choices):
            self.eagle_paths[0][i + 1][0] = 0
            for j in range(len(choice)):
                self.eagle_paths[0][i + 1][j + 1] = self.index_mapping_set[str(
                    choice[:j + 1])]

        # 3) Compute node_list_per_layer
        self.nodes_list_per_layer[0].append(0)  # root node
        for choice in self.eagle_choices:
            cur_layer = len(choice)
            self.nodes_list_per_layer[cur_layer].append(
                self.index_mapping_set[str(choice)])

        # 4) Compute child_nodes_list
        for choice in self.eagle_choices:
            if len(choice) == 1:  # root node's children
                child_nodes_list[0].append(self.index_mapping_set[str(choice)])
            else:
                child_nodes_list[self.index_mapping_set[str(
                    choice[:-1])]].append(self.index_mapping_set[str(choice)])

        # 5) Compute top_k_list
        for i in range(self.max_draft_len):
            cur_layer_nodes = self.nodes_list_per_layer[i]
            tmp_top_k_list = [
                len(child_nodes_list[node]) for node in cur_layer_nodes
                if len(child_nodes_list[node]) > 0
            ]
            assert sum(tmp_top_k_list) == len(self.nodes_list_per_layer[i + 1])
            self.top_k_list.append(
                torch.tensor(tmp_top_k_list,
                             dtype=torch.int32,
                             device='cpu',
                             pin_memory=prefer_pinned()))

        # 6) Compute the spec decoding according to the eagle_paths for the target model
        self.compute_spec_dec_mask_matrix(0)
        self.compute_spec_dec_packed_mask(self.spec_dec_mask_matrix,
                                          self.spec_dec_packed_mask)

        # 7) Compute the spec position offsets for the target model
        start_idx = 0
        for i in range(self.max_draft_len + 1):
            num_nodes_this_layer = len(self.nodes_list_per_layer[i])
            self.spec_dec_position_offsets[:, start_idx:start_idx +
                                           num_nodes_this_layer] = i
            start_idx += num_nodes_this_layer

        ### Compute the auxiliary buffers for the drafter model
        # 8) Copy top_k_list_cuda
        self.top_k_list_cuda = []
        for i in range(self.max_draft_len):
            self.top_k_list_cuda.append(self.top_k_list[i].to(
                device='cuda', dtype=torch.int32))
        # Compute the max top k value for all draft layers
        self.max_top_k = -1
        for top_k_list in self.top_k_list_cuda:
            self.max_top_k = max(self.max_top_k, top_k_list.max().item())

        # 9) Compute the tokens_gather_idx for the drafter model
        self.tokens_gather_idx_for_drafter_model = []
        self.tokens_gather_idx_for_drafter_model.append(
            torch.tensor([0], dtype=torch.int32,
                         device='cuda'))  # For the 1-st drafer layer

        for cur_layer_nodes in self.nodes_list_per_layer[1:]:
            tmp_gather_list = [(node - 1) for node in cur_layer_nodes
                               if len(child_nodes_list[node]) > 0]
            self.tokens_gather_idx_for_drafter_model.append(
                torch.tensor(tmp_gather_list, dtype=torch.int32, device='cuda'))

        # 10) Compute the draft_tokens_indices_cumsum for the drafter model
        num_nodes_per_layer = [0]
        num_nodes_per_layer.extend(
            [len(node_list) for node_list in self.nodes_list_per_layer[1:]])
        self.draft_tokens_indices_cumsum = torch.tensor(list(
            accumulate(num_nodes_per_layer)),
                                                        dtype=torch.int32,
                                                        device='cuda')

        # 11) Compute the spec_dec_packed_mask_for_drafter_model for the drafter model
        self.spec_dec_packed_mask_for_drafter_model = torch.zeros(
            (1, self.max_total_draft_tokens + 1,
             math.ceil((self.max_total_draft_tokens + 1) / 32)),
            dtype=torch.int32,
            device='cuda')
        tmp_mask_matrix = torch.zeros_like(
            self.spec_dec_mask_matrix[0]
        ).unsqueeze(
            0
        )  # [1, self.max_total_draft_tokens + 1, self.max_total_draft_tokens + 1]
        tmp_mask_matrix[0, :-1, :-1] = self.spec_dec_mask_matrix[0, 1:, 1:]
        self.compute_spec_dec_packed_mask(
            tmp_mask_matrix, self.spec_dec_packed_mask_for_drafter_model)

        self.hidden_states_read_indices_offset_for_drafter_model = torch.zeros(
            (self.max_total_draft_tokens + 1), dtype=torch.int32, device='cuda')
        tmp_parent_nodes = []
        for choice in self.eagle_choices:
            if len(choice) == 1:
                tmp_parent_nodes.append(0)
            else:
                tmp_parent_nodes.append(self.index_mapping_set[str(
                    choice[:-1])])
        self.hidden_states_read_indices_offset_for_drafter_model[:self.
                                                                 max_total_draft_tokens] = torch.tensor(
                                                                     tmp_parent_nodes,
                                                                     dtype=torch
                                                                     .int32,
                                                                     device=
                                                                     'cuda')

    def get_eagle_paths(self, tree_idx=0):
        if self.eagle_paths is None:
            raise RuntimeError(
                "get_eagle_paths() is not supported in dynamic tree mode; "
                "use retrieve_index/retrieve_next_token/retrieve_next_sibling instead"
            )
        return self.eagle_paths[0]

    # Get the topK list for the specific draft layer
    def get_top_k_list(self, draft_layer_id):
        assert draft_layer_id >= 0
        return self.top_k_list[draft_layer_id]

    def compute_spec_dec_mask_matrix(self, tree_idx=0):
        if self.eagle_paths is None:
            raise RuntimeError(
                "compute_spec_dec_mask_matrix() is not supported in dynamic tree mode"
            )
        for i, path in enumerate(self.eagle_paths[0]):
            indices = path[path > -1]
            self.spec_dec_mask_matrix[0][i, indices] = 1

    def compute_spec_dec_packed_mask(self, mask_matrix, packed_mask):
        bs, num_tokens, num_tokens_attend = mask_matrix.shape
        assert mask_matrix.ndim == 3, f"Expected 3D mask_matrix, got {mask_matrix.ndim}D"
        assert packed_mask.ndim == 3, f"Expected 3D packed_mask, got {packed_mask.ndim}D"
        assert bs <= self._padded_mask_buf.shape[0], \
            f"batch size {bs} exceeds pre-allocated buffer size {self._padded_mask_buf.shape[0]}"
        num_blocks = packed_mask.shape[-1]

        # Use cached bit weights
        weights = self._pack_weights

        # Pad into pre-allocated buffer
        total_bits = num_blocks * 32
        padded_m = self._padded_mask_buf[:bs, :num_tokens, :total_bits]
        padded_m.zero_()
        src = mask_matrix if mask_matrix.dtype == torch.int32 else mask_matrix.to(
            torch.int32)
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
        if self.use_dynamic_tree:
            logger.debug("Dynamic max top k: %s", self.dynamic_tree_max_topK)
        else:
            logger.debug("Max top k list cuda: %s", self.max_top_k)
            logger.debug("Eagle paths: %s", self.eagle_paths)
            logger.debug("Index mapping set: %s", self.index_mapping_set)
            logger.debug("Nodes list per layer: %s", self.nodes_list_per_layer)
            logger.debug("Spec dec position offsets: %s",
                         self.spec_dec_position_offsets)
            logger.debug("Spec dec mask matrix: %s",
                         self.spec_dec_mask_matrix.int())
            logger.debug("Spec dec pack mask: %s", self.spec_dec_packed_mask)
            logger.debug("Auxiliary buffers for the static tree.")
            logger.debug("TopK list cuda: %s", self.top_k_list_cuda)
            logger.debug("Tokens gather idx for drafter model: %s",
                         self.tokens_gather_idx_for_drafter_model)
            logger.debug("Draft tokens indices cumsum: %s",
                         self.draft_tokens_indices_cumsum)
            logger.debug("Spec dec packed mask for drafter model: %s",
                         self.spec_dec_packed_mask_for_drafter_model)
            logger.debug(
                "Hidden states read indices offset for drafter model: %s",
                self.hidden_states_read_indices_offset_for_drafter_model,
            )
