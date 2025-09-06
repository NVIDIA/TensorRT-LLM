from typing import Dict, List, Optional

import torch


class SpecTreeManager:
    dynamic_tree: bool  # Whether using dynamic tree
    max_total_draft_tokens: int  # The number of all nodes in the tree (except the root)
    dynamic_tree_max_topK: int  # If using dynamic tree, the number of nodes to expand each time.
    max_draft_len: int  # The number of drafter layer. When using linear-tree, the max_draft_len is the same as max_total_draft_tokens.
    cur_draft_layer_idx: int  # The current index of the drafter layer

    # Auxiliary buffers
    # The user input eagle choices, only available when using static tree.
    eagle_choices: Optional[List[List[int]]] = None
    # If dynamice tree, each request has their own tree. If static tree, all requests share the same tree.
    num_trees: Optional[int] = None
    # Mapping each choice/path to an index, which can also be treated as the node index.
    # Each tree will have an associated mapping list. Root node is '0'.
    index_mapping_list: Optional[List[Dict[str, int]]] = None

    # Convert the choice to a path. Each path is an array of indices from the root to other nodes in the tree.
    # shape: [num_trees, max_total_draft_tokens + 1, max_draft_len + 1]
    eagle_paths: Optional[torch.Tensor] = None

    # The nodes list of each layer. Padding with -1.
    # shape: [num_trees, max_draft_len + 1, max_total_draft_tokens + 1]
    nodes_list_per_layer: Optional[torch.Tensor] = None

    # The number of nodes in each layer. Actually a count of nodes_list_per_layer values that are not -1.
    # shape: [num_trees, max_draft_len + 1]
    num_nodes_per_layer: Optional[torch.Tensor] = None

    # The parent node of each node. For the root node, it is -1.
    # shape: [num_trees, max_total_draft_tokens + 1]
    parent_node_index: Optional[torch.Tensor] = None

    # The child nodes of each node. Padding with -1.
    # shape: [num_trees, max_total_draft_tokens + 1, max_total_draft_tokens]
    child_nodes_list: Optional[torch.Tensor] = None

    # The number of child nodes of each node. Actually a count of child_nodes_list values that are not -1.
    # shape: [num_trees, max_draft_len + 1]
    num_child_nodes: Optional[torch.Tensor] = None

    def __init__(self, max_num_requests: int, use_dynamic_tree: bool,
                 max_total_draft_tokens: int, max_draft_len: int,
                 eagle_choices: [List[List[int]]], dynamic_tree_max_topK: int):

        self.dynamic_tree = use_dynamic_tree
        self.max_total_draft_tokens = max_total_draft_tokens
        self.max_draft_len = max_draft_len
        self.eagle_choices = eagle_choices
        self.num_trees = max_num_requests if use_dynamic_tree else 1
        self.dynamic_tree_max_topK = dynamic_tree_max_topK
        self.cur_draft_layer_idx = -1

        # Initialize the buffers
        self.index_mapping_list = [{} for _ in range(self.num_trees)]
        self.eagle_paths = torch.ones(
            (self.num_trees, self.max_total_draft_tokens + 1,
             self.max_draft_len + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) * -1
        self.nodes_list_per_layer = torch.ones(
            (self.num_trees, self.max_draft_len + 1,
             self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) * -1
        self.num_nodes_per_layer = torch.zeros(
            (self.num_trees, self.max_draft_len + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.parent_node_index = torch.ones(
            (self.num_trees, self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) * -1
        self.child_nodes_list = torch.ones(
            (self.num_trees, self.max_total_draft_tokens + 1,
             self.max_total_draft_tokens),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) * -1
        self.num_child_nodes = torch.zeros(
            (self.num_trees, self.max_total_draft_tokens + 1),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.init_tree_from_input_choices()

    def init_tree_from_input_choices(self):
        if self.dynamic_tree:
            return

        # For the static tree
        tree_idx = 0
        # 1) Map the index
        self.index_mapping_list[tree_idx].clear()
        for i, choice in enumerate(self.eagle_choices):
            self.index_mapping_list[tree_idx][str(choice)] = i + 1

        # 2) Reconstruct the eagle_paths
        self.eagle_paths[tree_idx][0][0] = 0  # root node
        for i, choice in enumerate(self.eagle_choices):
            self.eagle_paths[tree_idx][i + 1][0] = 0
            for j, token in enumerate(choice):
                self.eagle_paths[tree_idx][i + 1][
                    j + 1] = self.index_mapping_list[tree_idx][str(choice[:j +
                                                                          1])]

        # 3) Compute num_nodes_per_layer
        self.nodes_list_per_layer[tree_idx][0][0] = 0  # root node
        self.num_nodes_per_layer[tree_idx][0] = 1
        for choice in self.eagle_choices:
            cur_layer = len(choice)
            self.nodes_list_per_layer[tree_idx][cur_layer][
                self.num_nodes_per_layer[tree_idx]
                [cur_layer]] = self.index_mapping_list[tree_idx][str(choice)]
            self.num_nodes_per_layer[tree_idx][cur_layer] += 1

        # Compute the parent node for each node
        self.parent_node_index[tree_idx][0] = -1
        for choice in self.eagle_choices:
            cur_node_index = self.index_mapping_list[tree_idx][str(choice)]
            if len(choice) == 1:
                self.parent_node_index[tree_idx][cur_node_index] = 0
            else:
                self.parent_node_index[tree_idx][
                    cur_node_index] = self.index_mapping_list[tree_idx][str(
                        choice[:-1])]

        # Compute the child nodes for each node
        tmp_child_nodes_list = [[]
                                for _ in range(self.max_total_draft_tokens + 1)]
        for choice in self.eagle_choices:
            if len(choice) == 1:
                tmp_child_nodes_list[0].append(
                    self.index_mapping_list[tree_idx][str(choice)])
            else:
                tmp_child_nodes_list[self.index_mapping_list[tree_idx][str(
                    choice[:-1])]].append(
                        self.index_mapping_list[tree_idx][str(choice)])
        for i in range(self.max_total_draft_tokens + 1):
            self.child_nodes_list[tree_idx][i][:len(tmp_child_nodes_list[i]
                                                    )] = torch.tensor(
                                                        tmp_child_nodes_list[i])
            self.num_child_nodes[tree_idx][i] = len(tmp_child_nodes_list[i])
