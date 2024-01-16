import copy
from argparse import Namespace
from functools import cmp_to_key
from typing import List

import numpy as np
import torch

from tensorrt_llm.logger import logger


def path_sorter(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return -1 if a[i] < b[i] else 1
    return 0  # shouldn't reach


path_sorting_key = cmp_to_key(path_sorter)


def expand_choices_if_needed(medusa_choices: List[List[int]]):
    """
    Do a simple check to see if the given choices are path-only or vanilla.
    """
    assert len(medusa_choices) > 0
    for c in medusa_choices:
        if len(c) > 1:
            try:
                _ = medusa_choices.index(
                    [c[0]])  # find the first parent of current path
                logger.debug(
                    "Detected vanilla-style of Medusa choices. No need to expand."
                )
                return medusa_choices  # if found, just return assuming it is already expanded
            except ValueError:
                logger.debug(
                    "Detected path-only style of Medusa choices. Expanding ...")
                break
    expanded_choices = set()
    for c in medusa_choices:
        cur = ()
        for n in c:
            cur = (*cur, n)
            expanded_choices.add(cur)
    expanded_choices = [list(c) for c in expanded_choices]
    return expanded_choices


def get_packed_mask(num_medusa_tokens, medusa_mask):
    num_packed_masks = (num_medusa_tokens + 1 + 32 - 1) // 32
    medusa_packed_mask = torch.zeros((num_medusa_tokens + 1, num_packed_masks),
                                     dtype=torch.int32)
    for token_idx in range(num_medusa_tokens + 1):
        if token_idx == 0:
            medusa_packed_mask[0, 0] = 1
        else:
            mask_list = medusa_mask[token_idx - 1, :].tolist()
            # insert 1 as there is one extra new token from the original lm head.
            mask_list.insert(0, True)
            # convert binary bits into 4 int32_t
            mask_str_list = [str(int(val)) for val in mask_list]
            mask_str_list.reverse()

            for mask_idx in range(num_packed_masks):
                if mask_idx * 32 >= len(mask_str_list):
                    break
                mask_32bits_str = ''.join(mask_str_list[-(mask_idx + 1) * 32:
                                                        (-mask_idx * 32 - 1)] +
                                          [mask_str_list[(-mask_idx * 32 - 1)]])
                valid_num_bits = len(mask_32bits_str)
                first_bit1 = mask_32bits_str[0] == '1'
                mask_31bits_str = mask_32bits_str[1:]
                mask_31bits = int(mask_31bits_str, 2)
                if valid_num_bits == 32:
                    mask_32bits = mask_31bits - first_bit1 * (2**(
                        valid_num_bits - 1))
                else:
                    mask_32bits = mask_31bits + first_bit1 * (2**(
                        valid_num_bits - 1))
                medusa_packed_mask[token_idx, mask_idx] = mask_32bits
    return medusa_packed_mask


def choices_2_paths(num_medusa_heads, choices):
    paths = {}
    all_paths = {}
    level_counts = [0] * num_medusa_heads
    choices.sort(key=len, reverse=True)
    for c in choices:
        k = ":".join([str(ci) for ci in c])
        if k not in all_paths:
            paths[k] = c
        for i in range(len(c)):
            k = ":".join([str(ci) for ci in c[:i + 1]])
            if k not in all_paths:
                all_paths[k] = c[:i + 1]
                level_counts[i] += 1
    # print(level_counts)
    return list(paths.values()), level_counts, paths, all_paths


def get_medusa_topks(num_medusa_heads, paths):
    medusa_topks = [0] * num_medusa_heads
    for p in paths:
        for i, k in enumerate(p):
            medusa_topks[i] = max(medusa_topks[i], k + 1)
    # print(medusa_topks)
    return medusa_topks


def get_medusa_tree(num_medusa_heads, medusa_topks, level_counts, paths):
    cum_topks = np.cumsum([0] + medusa_topks)
    cum_level_counts = np.cumsum([0] + level_counts)
    tree_paths = copy.deepcopy(paths)
    medusa_tree_ids = list(np.arange(medusa_topks[0]))
    medusa_position_offsets = [0] * medusa_topks[0]
    for i in range(1, num_medusa_heads):
        last_prefix = "-1"
        last = -1
        c = -1
        for pi, p in enumerate(paths):
            if i < len(p):
                prefix_str = ":".join([str(k) for k in p[:i]])
                if last_prefix != prefix_str or last != p[i]:
                    # new path
                    medusa_position_offsets.append(i)
                    medusa_tree_ids.append(p[i] + cum_topks[i])
                    last_prefix = prefix_str
                    last = p[i]
                    c += 1
                tree_paths[pi][i] = cum_level_counts[i] + c
    return medusa_tree_ids, medusa_position_offsets, tree_paths


def get_medusa_mask(medusa_tree_ids, medusa_paths):
    medusa_mask = torch.zeros((len(medusa_tree_ids), len(medusa_tree_ids)))
    medusa_mask[:, 0] = 1
    for p in medusa_paths:
        for i, idx in enumerate(p):
            if idx < 0:
                continue
            for j in range(i + 1):
                medusa_mask[idx, p[j]] = 1
    return medusa_mask


def _medusa_setup(choices_or_paths, num_medusa_heads=None):
    choices = copy.deepcopy(choices_or_paths)
    sorted_choices = sorted(choices, key=path_sorting_key)
    if num_medusa_heads is None:
        num_medusa_heads = max([len(c) for c in sorted_choices])
    paths, level_counts, _, _ = choices_2_paths(num_medusa_heads,
                                                sorted_choices)
    paths = sorted(paths, key=path_sorting_key)
    # print(paths)
    medusa_topks = get_medusa_topks(num_medusa_heads, paths)
    medusa_tree_ids, medusa_position_offsets, tree_paths = get_medusa_tree(
        num_medusa_heads, medusa_topks, level_counts, paths)

    num_medusa_tokens = len(medusa_tree_ids)
    # now do the padding before converting to torch.Tensor
    medusa_paths = []
    for p in tree_paths:
        medusa_paths.append(
            torch.tensor([-1] + p + ([-2] * (num_medusa_heads - len(p)))))
    medusa_topks = torch.tensor(medusa_topks)
    medusa_paths = torch.stack(medusa_paths) + 1
    medusa_tree_ids = torch.tensor([-1] + medusa_tree_ids) + 1
    medusa_position_offsets = torch.tensor([-1] + medusa_position_offsets) + 1
    medusa_mask = get_medusa_mask(medusa_tree_ids, medusa_paths)
    medusa_packed_mask = get_packed_mask(num_medusa_tokens, medusa_mask[1:, 1:])
    return Namespace(
        medusa_mask=medusa_mask.cuda(),
        medusa_packed_mask=medusa_packed_mask.cuda(),
        medusa_topks=medusa_topks.cuda(),
        medusa_paths=medusa_paths.cuda(),
        medusa_tree_ids=medusa_tree_ids.cuda(),
        medusa_position_offsets=medusa_position_offsets.cuda(),
    )
