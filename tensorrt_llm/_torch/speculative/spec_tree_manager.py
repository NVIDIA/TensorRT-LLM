from typing import List, Optional

import torch


class SpecTreeManager:
    use_dynamic_tree: bool  # Whether using dynamic tree
    max_total_draft_tokens: int  # The number of all nodes in the tree (except the root)
    dynamic_tree_max_topK: int  # If using dynamic tree, the number of nodes to expand each time.
    max_draft_len: int  # The number of drafter layer. When using linear-tree, the max_draft_len is the same as max_total_draft_tokens.
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

    def __init__(self, max_num_requests: int, use_dynamic_tree: bool,
                 max_total_draft_tokens: int, max_draft_len: int,
                 eagle_choices: [List[List[int]]], dynamic_tree_max_topK: int):

        self.use_dynamic_tree = use_dynamic_tree
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len
        self.eagle_choices = eagle_choices
        self.num_trees = max_num_requests if use_dynamic_tree else 1
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.cur_draft_layer_idx = -1

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
            (self.num_trees + 1, self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.top_k_list = []

        if self.use_dynamic_tree:
            self.init_tree_info_for_dynamic_tree()
        else:
            self.init_tree_info_for_static_tree()

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
        index_mapping_set = {}
        nodes_list_per_layer = [[] for _ in range(self.max_draft_len + 1)]
        child_nodes_list = [[] for _ in range(self.max_total_draft_tokens + 1)]

        # 1) Map the index
        for i, choice in enumerate(self.eagle_choices):
            index_mapping_set[str(choice)] = i + 1

        # 2) Reconstruct the eagle_paths
        self.eagle_paths.fill_(-1)
        self.eagle_paths[0][0][0] = 0  # root node
        for i, choice in enumerate(self.eagle_choices):
            self.eagle_paths[0][i + 1][0] = 0
            for j in range(len(choice)):
                self.eagle_paths[0][i + 1][j + 1] = index_mapping_set[str(
                    choice[:j + 1])]

        # 3) Compute node_list_per_layer
        nodes_list_per_layer[0].append(0)  # root node
        for choice in self.eagle_choices:
            cur_layer = len(choice)
            nodes_list_per_layer[cur_layer].append(
                index_mapping_set[str(choice)])

        # 4) Compute child_nodes_list
        for choice in self.eagle_choices:
            if len(choice) == 1:  # root node's children
                child_nodes_list[0].append(index_mapping_set[str(choice)])
            else:
                child_nodes_list[index_mapping_set[str(choice[:-1])]].append(
                    index_mapping_set[str(choice)])

        # 5) Compute top_k_list
        for i in range(self.max_draft_len):
            cur_layer_nodes = nodes_list_per_layer[i]
            tmp_top_k_list = [
                len(child_nodes_list[node]) for node in cur_layer_nodes
                if len(child_nodes_list[node]) > 0
            ]
            assert sum(tmp_top_k_list) == len(nodes_list_per_layer[i + 1])
            self.top_k_list.append(
                torch.tensor(tmp_top_k_list,
                             dtype=torch.int32,
                             device='cpu',
                             pin_memory=True))

        # 6) Compute the spec decoding according to the eagle_paths
        for i, path in enumerate(self.eagle_paths[0]):
            indices = path[path > -1]
            self.spec_dec_mask_matrix[0][i, indices] = 1
        self.spec_dec_pack_mask = self.compute_spec_dec_pack_mask(
            self.spec_dec_mask_matrix)

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
    def compute_spec_dec_pack_mask(self, mask_matrix):
        # mask_matrix: shape: [num_trees, max_total_draft_tokens + 1, max_total_draft_tokens + 1]
        int_tensor = mask_matrix.to(torch.int32)
        weights = torch.pow(
            2, torch.arange(mask_matrix.shape[-1], device=mask_matrix.device))
        packed_mask = torch.sum(int_tensor * weights, dim=-1)
        return packed_mask

    # Print the tree info
    def dump_tree_info(self):
        if not self.use_dynamic_tree:
            print(f"Static tree: {self.eagle_paths}")
        print(f"TopK list: {self.top_k_list}")
        print(f"Spec dec mask matrix: {self.spec_dec_mask_matrix.int()}")
        print(f"Spec dec pack mask: {self.spec_dec_pack_mask}")
