import math
from typing import List, Optional

import torch


class SpecTreeManager:
    use_dynamic_tree: bool  # Whether using dynamic tree
    max_total_draft_tokens: int  # The number of all nodes in the tree (except the root)
    dynamic_tree_max_topK: int  # If using dynamic tree, the number of nodes to expand each time.
    max_draft_len: int  # The number of drafter layer.
    cur_draft_layer_idx: int  # The current index of the drafter layer

    # Auxiliary buffers
    # The user input eagle choices, only available when using static tree.
    eagle_choices: Optional[List[List[int]]] = None
    # If dynamice tree, each request has their own tree. If static tree, all requests share the same tree.
    num_trees: Optional[int] = None

    # Convert the choice to a path. Each path is an array of indices from the root to other nodes in the tree.
    # shape: [num_trees, max_total_draft_tokens + 1, max_draft_len + 1]
    eagle_paths: Optional[torch.Tensor] = None

    # The spec decoding mask.
    # shape: [num_trees, max_total_draft_tokens + 1, max_total_draft_tokens + 1], include the root node.
    spec_dec_mask_matrix: Optional[torch.Tensor] = None
    # shape: [num_trees, max_total_draft_tokens + 1], pad the 0-1 matrix to int32 vector.
    spec_dec_pack_mask: Optional[torch.Tensor] = None

    # The spec position offsets.
    # shape: [num_trees, max_total_draft_tokens + 1].
    spec_dec_position_offsets: Optional[torch.Tensor] = None

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
            dtype=torch.bool,
            device='cpu',
            pin_memory=True,
        ).unsqueeze(0).repeat(self.num_trees, 1, 1)
        self.spec_dec_pack_mask = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1,
             math.ceil((self.max_total_draft_tokens + 1) / 32)),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.spec_dec_position_offsets = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )

        # top_k_list[i] means the topK values for the i-th layer.
        # Examples:
        #   top_k_list[0] = [3] means the topK values for the 0-th layer (aka, expand the root node) is 3.
        #   top_k_list[1] = [3, 2, 1] means the topK values for the 1-th layer's nodes is 3, 2, 1, respectively.
        self.top_k_list = []

        # cumulative_draft_lens[i] means the cumulative draft token lengths AFTER the i-th layer. DO NOT include the root node.
        # Examples:
        #   cumulative_draft_lens[0] = 3 means the draft tokens generated AFTER the 0-th layer will be 3.
        #   cumulative_draft_lens[1] = 9 (3 + 3 + 2 + 1 = 9) means the cumulative generated AFTER the 1-st layer will be 9.
        self.cumulative_draft_lens = []

        # For each drafter's generation requests, we take all selected draft tokens from the current layer and above (excluding the root node) as input tokens.
        # However, only some tokens in the current layer will have child nodes, i.e., they will continue to expand and generated the next layer.
        # We only need to set the gather_ids for these tokens.
        # NOTE: For the static tree, each node is selected.
        # gather_ids_per_layer is a list of offsets for the draft tokens that need to gather_ids in draft_layer_id layer.
        # The offset is relative to the root node - 1 (-1 is because the root node is excluded).
        self.gather_ids_per_layer = []

        # Auxiliary variable for static tree.
        #Considering that the static tree is a fixed tree, we can use some auxiliary variables to record some
        # information in advance to avoid repeated calculations between different iterations.
        # nodes_list_per_layer[i] means the nodes list for the i-th layer. Include the root node.
        self.nodes_list_per_layer = []
        # Mapping choices to unique indices.
        self.index_mapping_set = {}

        if self.use_dynamic_tree:
            self.init_tree_info_for_dynamic_tree()
        else:
            self.init_tree_info_for_static_tree()

        # self.dump_tree_info()

    def init_tree_info_for_dynamic_tree(self):
        # For the dynamic tree
        # To the internal layer, the number of nodes is the same as the dynamic_tree_max_topK.
        self.top_k_list.append(
            torch.tensor([self.dynamic_tree_max_topK],
                         dtype=torch.int32,
                         device='cpu',
                         pin_memory=True))
        for i in range(self.max_draft_len - 1):
            self.topK_list.append(
                torch.tensor([
                    self.dynamic_tree_max_topK
                    for _ in range(self.dynamic_tree_max_topK)
                ],
                             dtype=torch.int32,
                             device='cpu',
                             pin_memory=True))

        self.cumulative_draft_lens = [
            i * self.dynamic_tree_max_topK
            for i in range(1, self.max_draft_len + 1)
        ]

        self.gather_ids_per_layer.append([0])
        for i in range(1, self.max_draft_len):
            self.gather_ids_per_layer.append(
                list(
                    range(self.dynamic_tree_max_topK * (i - 1),
                          self.dynamic_tree_max_topK * i)))

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
        # 6) Compute cumulative_draft_lens
        self.cumulative_draft_lens.append(len(self.nodes_list_per_layer[1]))
        for i in range(1, self.max_draft_len):
            self.cumulative_draft_lens.append(
                self.cumulative_draft_lens[i - 1] +
                len(self.nodes_list_per_layer[i + 1]))

        # 7) Compute the spec decoding according to the eagle_paths
        for i, path in enumerate(self.eagle_paths[0]):
            indices = path[path > -1]
            self.spec_dec_mask_matrix[0][i, indices] = 1
        self.compute_spec_dec_pack_mask(self.spec_dec_mask_matrix,
                                        self.spec_dec_pack_mask)

        # 8) Compute the spec position offsets
        start_idx = 0
        for i in range(self.max_draft_len + 1):
            num_nodes_this_layer = len(self.nodes_list_per_layer[i])
            self.spec_dec_position_offsets[:, start_idx:start_idx +
                                           num_nodes_this_layer] = i
            start_idx += num_nodes_this_layer

        # 9) Compute the gather_ids_per_layer
        self.gather_ids_per_layer.append([0])
        for i in range(1, self.max_draft_len):
            cur_gather_ids = []
            for path in self.eagle_choices:
                if len(path) == i:
                    # Has child node(s)
                    if (len(child_nodes_list[self.index_mapping_set[str(path)]])
                            > 0):
                        cur_gather_ids.append(self.index_mapping_set[str(path)])
            self.gather_ids_per_layer.append(cur_gather_ids)

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
        assert draft_layer_id >= 0 and draft_layer_id < self.max_draft_len
        return self.top_k_list[draft_layer_id]

    # Get the cumulative draft token lengths AFTER the i-th layer. DO NOT include the root node.
    def get_cumulative_draft_lens(self, draft_layer_id):
        assert draft_layer_id >= 0 and draft_layer_id < self.max_draft_len
        return self.cumulative_draft_lens[draft_layer_id]

    # Get the draft token lengths for the specific draft layer.
    def get_current_layer_draft_len(self, draft_layer_id):
        assert draft_layer_id >= 0 and draft_layer_id < self.max_draft_len
        if self.use_dynamic_tree:
            return self.dynamic_tree_max_topK
        else:
            return len(self.nodes_list_per_layer[draft_layer_id +
                                                 1])  # +1 to skip the root node

    # Get the gather ids for the specific draft layer.
    # Return: A list [].
    def get_gather_ids(self, draft_layer_id):
        assert draft_layer_id > 0 and draft_layer_id < self.max_draft_len
        return self.gather_ids_per_layer[draft_layer_id]

    # Compute the packed mask according to the mask matrix
    def compute_spec_dec_pack_mask(self, mask_matrix, packed_mask):
        # mask_matrix: shape: [num_trees, max_total_draft_tokens + 1, max_total_draft_tokens + 1]
        # packed_mask: shape: [num_trees, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32)]
        num_blocks = math.ceil((self.max_total_draft_tokens + 1) / 32)
        int_tensor = mask_matrix.to(torch.int32)
        int_tensor = int_tensor.reshape(-1, self.max_total_draft_tokens + 1)
        packed_mask = packed_mask.reshape(-1, num_blocks)

        for block_idx in range(num_blocks):
            start_idx = block_idx * 32
            end_idx = min(start_idx + 32, self.max_total_draft_tokens + 1)
            block_bits = int_tensor[:, start_idx:end_idx]
            weight = torch.pow(
                2,
                torch.arange(end_idx - start_idx,
                             dtype=torch.int32,
                             device=int_tensor.device))
            block_value = torch.sum(block_bits * weight, dim=-1)
            packed_mask[:, block_idx] = block_value

        packed_mask = packed_mask.reshape(self.num_trees,
                                          self.max_total_draft_tokens + 1,
                                          num_blocks)

    # Print the tree info
    def dump_tree_info(self):
        if not self.use_dynamic_tree:
            print(f"Static tree: {self.eagle_paths}")
            print(f"Nodes list per layer: {self.nodes_list_per_layer}")
            print(f"Index mapping set: {self.index_mapping_set}")
        print(f"TopK list: {self.top_k_list}")
        print(f"Cumulative draft lens: {self.cumulative_draft_lens}")
        print(f"Gather ids per layer: {self.gather_ids_per_layer}")
        print(f"Spec dec mask matrix: {self.spec_dec_mask_matrix.int()}")
        print(f"Spec dec pack mask: {self.spec_dec_pack_mask}")
        print(f"Spec dec position offsets: {self.spec_dec_position_offsets}")

    def print_mask_matrix_from_packed_mask(self):
        for i in range(self.num_trees):
            for j in range(self.max_total_draft_tokens + 1):
                num_blocks = math.ceil((self.max_total_draft_tokens + 1) / 32)
                for k in range(num_blocks - 1, -1, -1):
                    print(bin(self.spec_dec_pack_mask[i, j, k])[2:], end='')
