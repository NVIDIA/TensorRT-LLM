import math
from typing import List

import torch


class SpecTreeManager:
    use_dynamic_tree: bool  # Whether using dynamic tree
    max_total_draft_tokens: int  # The number of all nodes in the tree (except the root)
    dynamic_tree_max_topK: int  # If using dynamic tree, the number of nodes to expand each time.
    max_draft_len: int  # The number of drafter layer. When using linear-tree, the max_draft_len is the same as max_total_draft_tokens.
    cur_draft_layer_idx: int  # The current index of the drafter layer

    # Auxiliary buffers
    # The top k  list for each draft layer.
    top_k_list = []
    # The user input eagle choices, only available when using static tree.
    eagle_choices: List[List[int]] = None
    # If dynamice tree, each request has their own tree. If static tree, all requests share the same tree.
    num_trees: int = None

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

    # TODO: Optimized together with the subsequent dynamic tree.
    # Auxiliary buffers for the static tree.
    # Considering that the static tree does not modify the tree structure during inference, we can calculate some buffers in advance.
    # NOTE: Most of these buffers are introduced due to limitations of XQA:
    #       With tree attention, XQA cannot simply take the tokens to be processed in the next round as input. Instead, it needs to take ALL of their parent nodes as input.
    #       This incurs additional computation, but it is unavoidable.

    # NOTE: The reason why most of these auxiliary buffers are with `len == max_draft_len - 1` is that: we do not need to prepare specific input data for the first draft layer.

    # The top k value for each draft layer. Device tensor.
    top_k_list_cuda: list[torch.Tensor] = None

    # The max top k value for all draft layers. Which is used for torch.topk and cuda graph.
    max_top_k = -1

    # Gather the required draft tokens from all currently generated draft tokens as the input of the next draft layer.
    # Only the nodes has child(s) this layer and all their parents nodes will be gathered.
    # Device tensor. len(tokens_gather_idx) == max_draft_len - 1. Each element is a tensor with shape [num_tokens_for_next_layer].
    tokens_gather_idx_for_drafter_model: list[torch.Tensor] = None

    # Gather the required logits from all currently generated logits.
    # Device tensor. len(tokens_gather_idx) == max_draft_len - 1.
    logits_gather_idx: list[torch.Tensor] = None

    # The packed mask for the drafter model's attention (i.e., xqa).
    spec_dec_packed_mask_for_drafter_model: torch.Tensor = None

    # The read indices offset for the drafter model.
    hidden_states_read_indices_offset_for_drafter_model: torch.Tensor = None

    # The write back start indices for the drafter tokens between different draft layers.
    draft_tokens_indices_cumsum: torch.Tensor = None

    def __init__(self, max_num_requests: int, use_dynamic_tree: bool,
                 max_total_draft_tokens: int, max_draft_len: int,
                 eagle_choices: [List[List[int]]], dynamic_tree_max_topK: int):

        self.use_dynamic_tree = use_dynamic_tree
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len
        self.eagle_choices = eagle_choices
        self.num_trees = max_num_requests if use_dynamic_tree else 1
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.cur_draft_layer_idx = 0
        self.top_k_list = []

        # Initialize the buffers
        self.eagle_paths = torch.ones(
            (self.num_trees, self.max_total_draft_tokens + 1,
             self.max_draft_len + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) * -1

        self.spec_dec_mask_matrix = torch.eye(
            self.max_total_draft_tokens + 1,
            dtype=torch.int32,
            device='cuda',
        ).unsqueeze(0).repeat(self.num_trees, 1, 1)

        self.spec_dec_packed_mask = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1,
             math.ceil((self.max_total_draft_tokens + 1) / 32)),
            dtype=torch.int32,
            device='cuda',
        )
        self.spec_dec_position_offsets = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cuda',
        )

        if self.use_dynamic_tree:
            self.init_tree_info_for_dynamic_tree()
        else:
            self.init_tree_info_for_static_tree()

        # self.dump_tree_info()

    def init_tree_info_for_dynamic_tree(self):
        # For the dynamic tree
        # To the internal layer, the number of nodes is the same as the dynamic_tree_max_topK.
        self.top_k_list = [
            torch.ones(self.dynamic_tree_max_topK,
                       dtype=torch.int32,
                       device='cpu',
                       pin_memory=True) * self.dynamic_tree_max_topK
        ]

    # For the static tree
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
                             pin_memory=True))

        # 6) Compute the spec decoding according to the eagle_paths for the target model
        for i, path in enumerate(self.eagle_paths[0]):
            indices = path[path > -1]
            self.spec_dec_mask_matrix[0][i, indices] = 1
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
        from itertools import accumulate
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

    # Get the eagle_paths
    def get_eagle_paths(self, tree_idx=0):
        if self.use_dynamic_tree:
            self.eagle_paths[tree_idx].fill_(-1)
            # If dynamic tree, return the eagle_paths according to the mask.
            for i in range(self.max_total_draft_tokens + 1):
                self.eagle_paths[tree_idx][:, i, :] = self.spec_dec_mask_matrix[
                    tree_idx][i, :].nonzero()
            return self.eagle_paths[tree_idx]
        else:
            # If static tree, return the prepared eagle_paths. These paths are immutable.
            return self.eagle_paths[0]

    # Get the topK list for the specific draft layer
    def get_top_k_list(self, draft_layer_id):
        assert draft_layer_id >= 0
        return self.top_k_list[draft_layer_id]

    # Compute the packed mask according to the mask matrix
    def compute_spec_dec_packed_mask(self, mask_matrix, packed_mask):
        # mask_matrix: shape: [num_trees, max_total_draft_tokens + 1, max_total_draft_tokens + 1]
        # packed_mask: shape: [num_trees, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32)]
        assert mask_matrix.ndim == 3
        assert packed_mask.ndim == 3

        num_trees = mask_matrix.shape[0]
        assert mask_matrix.shape == (num_trees, self.max_total_draft_tokens + 1,
                                     self.max_total_draft_tokens + 1)
        assert packed_mask.shape == (num_trees, self.max_total_draft_tokens + 1,
                                     math.ceil(
                                         (self.max_total_draft_tokens + 1) /
                                         32))

        num_blocks = math.ceil((self.max_total_draft_tokens + 1) / 32)
        int_tensor = mask_matrix.reshape(
            -1, self.max_total_draft_tokens + 1
        )  # shape: [num_trees * (self.max_total_draft_tokens + 1), self.max_total_draft_tokens + 1]
        packed_mask = packed_mask.reshape(
            -1, num_blocks
        )  # shape: [num_trees * (self.max_total_draft_tokens + 1), num_blocks]

        for block_idx in range(num_blocks):
            start_idx = block_idx * 32
            end_idx = min(start_idx + 32, self.max_total_draft_tokens + 1)
            if end_idx < start_idx:
                break
            block_bits = int_tensor[:, start_idx:end_idx]
            weight = torch.pow(
                2,
                torch.arange(end_idx - start_idx,
                             dtype=torch.int32,
                             device=int_tensor.device))
            block_value = torch.sum(block_bits * weight, dim=-1)
            packed_mask[:, block_idx] = block_value

        packed_mask = packed_mask.reshape(num_trees,
                                          self.max_total_draft_tokens + 1,
                                          num_blocks)

    # Print the tree info
    def dump_tree_info(self):
        print(f"TopK list: {self.top_k_list}")
        if not self.use_dynamic_tree:
            print(f"Max top k list cuda: {self.max_top_k}")
            print(f"Static tree: {self.eagle_paths}")
            print(f"Index mapping set: {self.index_mapping_set}")
            print(f"Nodes list per layer: {self.nodes_list_per_layer}")
            print(
                f"Spec dec position offsets: {self.spec_dec_position_offsets}")
            print(f"Spec dec mask matrix: {self.spec_dec_mask_matrix.int()}")
            print(f"Spec dec pack mask: {self.spec_dec_packed_mask}")
            print("Auxiliary buffers for the static tree.")
            print(f"TopK list cuda: {self.top_k_list_cuda}")
            print(
                f"Tokens gather idx for drafter model: {self.tokens_gather_idx_for_drafter_model}"
            )
            print(
                f"Draft tokens indices cumsum: {self.draft_tokens_indices_cumsum}"
            )
            print(f"Logits gather idx: {self.logits_gather_idx}")
            print(
                f"Spec dec packed mask for drafter model: {self.spec_dec_packed_mask_for_drafter_model}"
            )
            print(
                f"Hidden states read indices offset for drafter model: {self.hidden_states_read_indices_offset_for_drafter_model}"
            )
