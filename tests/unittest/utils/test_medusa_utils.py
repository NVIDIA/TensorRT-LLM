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

from parameterized import parameterized

# isort: off
import torch
# isort: on

import copy
from argparse import Namespace
from typing import List

import tensorrt_llm
from tensorrt_llm.runtime.medusa_utils import (_medusa_setup, choices_2_paths,
                                               expand_choices_if_needed,
                                               get_packed_mask,
                                               path_sorting_key)

#############################################
## Taken from Medusa and slightly modified ##
## Used for testing only ##
#############################################


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_medusa_info(medusa_choices,
                         num_medusa_heads=None,
                         device="cuda"):  # MODIFIED
    """
    [MODIFIED]
    NOTE:
    tree_ids is different than the vanilla implementation.
    Instead of TOPK to be fixed for all the heads, we can vary it based on the given choices.
    Example: [[0], [1], [0,0]] would lead to topk for the 2 Medusa heads to be [2, 1].
    And tree_ids would be [0, 1, 2, 3], instead of vanilla [0, 1, 2, 11]

    Generate buffers for the Medusa structure based on the provided choices.

    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".

    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    # MODIFIED
    if num_medusa_heads is None:
        num_medusa_heads = max([len(c) for c in medusa_choices])
    medusa_topks = [0] * num_medusa_heads
    #
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        medusa_topks[depth - 1] = max(medusa_topks[depth - 1],
                                      path[-1] + 1)  # MODIFIED
        prev_depth = depth

    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(
                    sorted_medusa_choices.index(cur_medusa_choice[:c + 1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + sum(
                medusa_topks[:i]) + 1  # MODIFIED
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1:start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i - 1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(
                    sorted_medusa_choices.index(cur_medusa_choice[:c + 1]))
                retrieve_paths.append(cur_medusa_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [
        pad_path(path, max_length) for path in retrieve_indices_nest
    ]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([
        torch.zeros(
            (retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices
    ],
                                 dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask":
        medusa_attn_mask,  #.unsqueeze(0).unsqueeze(0), # MODIFIED
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        # MODIFIED
        "medusa_topks": medusa_topks,
    }

    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k:
        v.clone().to(device)
        if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers


#############################################
## End of imported code ##
#############################################


def _vanilla_medusa_setup(medusa_choices: List[List[int]],
                          num_medusa_heads=None):
    """
    Just a wrapper around the vanilla (modified) method, in case we write our own in the future.
    One flexibility added: medusa choices can also be given as only the full paths e.g.
    Path-only style:
        [[0,0,0,0],[0,1,0],[1,0],[1,1]] is equivalent to
    Vanilla-style:
        [[0], [0,0], [0,0,0], [0,0,0,0], [0,1], [0,1,0], [1], [1,0], [1,1]]
    """
    medusa_choices = copy.deepcopy(medusa_choices)
    medusa_choices = expand_choices_if_needed(medusa_choices)
    medusa_info = generate_medusa_info(medusa_choices, num_medusa_heads)
    medusa_topks = medusa_info['medusa_topks']
    medusa_mask = medusa_info['medusa_attn_mask']
    medusa_paths = medusa_info['retrieve_indices']
    medusa_tree_ids = medusa_info['tree_indices']
    medusa_position_offsets = medusa_info['medusa_position_ids']
    medusa_packed_mask = get_packed_mask(len(medusa_choices), medusa_mask[1:,
                                                                          1:])
    return Namespace(
        medusa_mask=medusa_mask,
        medusa_packed_mask=medusa_packed_mask,
        medusa_topks=medusa_topks,
        medusa_paths=medusa_paths,
        medusa_tree_ids=medusa_tree_ids,
        medusa_position_offsets=medusa_position_offsets,
    )


class TestMedusaUtils(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ([[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 1], [0, 1, 0], [1], [1, 0],
          [1, 1]], ),
        ([[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 1], [0, 1, 0], [1], [1, 0],
          [1, 1], [2], [2, 0], [0, 2], [0, 2, 0]], ),
        ([[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3],
          [4], [0, 4], [2, 0]], ),
        (
            [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 1], [0, 1, 0], [1],
             [1, 0], [1, 1]],
            [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]],
        ),
        ([[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], ),
        # test the original choices
        (
            [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3],
             [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6],
             [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9],
             [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1],
             [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6],
             [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0],
             [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2],
             [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0],
             [0, 0, 0, 1], [1, 6], [0, 7, 0]], )
    ])
    def test_all_buffers(self, choices, paths=None):
        num_medusa_heads = max([len(c) for c in choices])
        info = _vanilla_medusa_setup(choices)
        vanilla_paths = sorted(info.medusa_paths.tolist(), key=path_sorting_key)
        cinfo = _medusa_setup(choices)
        test_paths = sorted(cinfo.medusa_paths.tolist(), key=path_sorting_key)
        assert info.medusa_tree_ids.equal(
            cinfo.medusa_tree_ids
        ), f"{info.medusa_tree_ids} \n\n!=\n\n {cinfo.medusa_tree_ids}"
        assert vanilla_paths == test_paths, f"\n{vanilla_paths} \n\n!=\n\n {test_paths}"
        assert info.medusa_position_offsets.equal(
            cinfo.medusa_position_offsets
        ), f"{info.medusa_position_offsets} != {cinfo.medusa_position_offsets}"
        assert info.medusa_mask.equal(
            cinfo.medusa_mask
        ), f"{info.medusa_mask} \n\n!=\n\n {cinfo.medusa_mask}"
        assert info.medusa_packed_mask.cuda().equal(
            cinfo.medusa_packed_mask
        ), f"{info.medusa_packed_mask} != {cinfo.medusa_packed_mask}"
        if paths is not None:
            deduced_choices = expand_choices_if_needed(paths)
            assert sorted(deduced_choices) == sorted(
                choices), f"{deduced_choices} != {choices}"
            deduced_paths, _, _, _ = choices_2_paths(num_medusa_heads, choices)
            assert sorted(deduced_paths, key=path_sorting_key) == sorted(
                paths, key=path_sorting_key), f"{deduced_paths} != {paths}"
            pinfo = _medusa_setup(paths)
            assert info.medusa_paths.sort(dim=0).values.equal(
                pinfo.medusa_paths.sort(dim=0).values
            ), f"{info.medusa_paths.sort(dim=0).values} != {pinfo.medusa_paths.sort(dim=0).values}"
            assert info.medusa_tree_ids.equal(
                pinfo.medusa_tree_ids
            ), f"{info.medusa_tree_ids} != {pinfo.medusa_tree_ids}"
            assert info.medusa_position_offsets.equal(
                pinfo.medusa_position_offsets
            ), f"{info.medusa_position_offsets} != {pinfo.medusa_position_offsets}"
            assert info.medusa_mask.equal(
                pinfo.medusa_mask
            ), f"{info.medusa_mask} \n\n!=\n\n {pinfo.medusa_mask}"
            assert info.medusa_packed_mask.cuda().equal(
                pinfo.medusa_packed_mask
            ), f"{info.medusa_packed_mask} != {pinfo.medusa_packed_mask}"
        return
