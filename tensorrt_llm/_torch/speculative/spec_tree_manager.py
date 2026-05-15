import logging
import math
from itertools import accumulate
from typing import List

import torch

from tensorrt_llm._utils import prefer_pinned

logger = logging.getLogger(__name__)


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

        # Per-py_seq_slot storage (+1 dummy row for graph); survives batch reorder.
        S = self.num_trees + 1
        self._slot_packed_mask = torch.zeros(
            (S, ) + self.spec_dec_packed_mask.shape[1:],
            dtype=self.spec_dec_packed_mask.dtype,
            device='cuda')
        self._slot_position_offsets = torch.zeros(
            (S, ) + self.spec_dec_position_offsets.shape[1:],
            dtype=self.spec_dec_position_offsets.dtype,
            device='cuda')
        # [S, n_dt, 3]: index / next_token / next_sibling in one tensor (fewer gathers).
        self._slot_retrieve_packed = torch.full((S, num_draft_with_root, 3),
                                                -1,
                                                dtype=torch.int32,
                                                device='cuda')
        self._slot_retrieve_packed[:, :, 0] = 0  # match retrieve_index default
        self._all_slot_ids_buf = torch.zeros(self.num_trees,
                                             dtype=torch.long,
                                             device='cuda')
        self._dummy_slot_id = self.num_trees

        # True after scatter; dummy row stays False.
        self.slot_has_tree = torch.zeros(S, dtype=torch.bool, device='cuda')

        # Graph-safe: no per-forward host-to-device slot-id tensor alloc.
        self._gather_gen_slot_ids_buf = torch.zeros(self.num_trees,
                                                    dtype=torch.long,
                                                    device='cuda')

        # Scatter staging (no per-call torch.stack).
        self._scatter_retrieve_staging = torch.empty(
            (self.num_trees, num_draft_with_root, 3),
            dtype=torch.int32,
            device='cuda')

    def fill_gen_slot_ids(self, gen_requests):
        """Fill _gather_gen_slot_ids_buf; return (buf[:count], count). Gen LlmRequest only."""
        buf = self._gather_gen_slot_ids_buf
        dummy = self._dummy_slot_id
        count = 0
        for r in gen_requests:
            buf[count] = r.py_seq_slot if r.py_seq_slot is not None else dummy
            count += 1
        return buf[:count], count

    def mark_tree_valid(self, slot_ids, count):
        """Set slot_has_tree True for scattered slots."""
        if count == 0:
            return
        # index_fill_: graph-capture-safe (no fancy indexing).
        self.slot_has_tree.index_fill_(0, slot_ids[:count], True)
        # CUDA graph padding may use the dummy slot; it must never carry a real tree.
        self.slot_has_tree.narrow(0, self._dummy_slot_id, 1).fill_(False)

    def mark_tree_invalid(self, slot_id):
        """Clear validity when a slot is freed."""
        self.slot_has_tree[slot_id] = False

    def scatter_trees_to_slots(self, slot_ids, count):
        """Copy work buffers [:count] into per-slot storage (index_copy_ for graph capture)."""
        if count == 0:
            return
        ids = slot_ids[:count]
        self._slot_packed_mask.index_copy_(0, ids,
                                           self.spec_dec_packed_mask[:count])
        self._slot_position_offsets.index_copy_(
            0, ids, self.spec_dec_position_offsets[:count])
        staging = self._scatter_retrieve_staging[:count]
        staging[:, :, 0] = self.retrieve_index[:count]
        staging[:, :, 1] = self.retrieve_next_token[:count]
        staging[:, :, 2] = self.retrieve_next_sibling[:count]
        self._slot_retrieve_packed.index_copy_(0, ids, staging)

    def gather_attn_params_from_slots(self, slot_ids, count):
        """Copy mask and position offsets from slots into work buffers [:count]."""
        if count == 0:
            return
        ids = slot_ids[:count]
        self.spec_dec_packed_mask[:count] = self._slot_packed_mask[ids]
        self.spec_dec_position_offsets[:count] = self._slot_position_offsets[
            ids]

    def gather_retrieve_from_slots(self, slot_ids, count):
        """Copy retrieve tensors from slots into work buffers [:count]."""
        if count == 0:
            return
        ids = slot_ids[:count]
        packed = self._slot_retrieve_packed[ids]
        self.retrieve_index[:count] = packed[..., 0]
        self.retrieve_next_token[:count] = packed[..., 1]
        self.retrieve_next_sibling[:count] = packed[..., 2]

    def gather_trees_from_slots(self, slot_ids, count):
        """gather_attn_params_from_slots + gather_retrieve_from_slots."""
        self.gather_attn_params_from_slots(slot_ids, count)
        self.gather_retrieve_from_slots(slot_ids, count)

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
