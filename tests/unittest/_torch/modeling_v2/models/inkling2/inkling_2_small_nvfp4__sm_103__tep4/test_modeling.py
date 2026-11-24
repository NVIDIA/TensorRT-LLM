# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Inkling 2 target's engine glue: the page tables against the KV cache's
retention, and the config errors raised before any weight is declared."""

import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.inkling_2_small_nvfp4__sm_103__tep4 import (
    modeling,
)

pytestmark = pytest.mark.cpu_only

#: Layer 0 slides over 8 tokens and layer 1 is global, with 4 tokens per KV cache block.
_WINDOW = 8
_TOKENS_PER_BLOCK = 4


class _Manager:
    """The rows `KVCacheManagerV2.get_batch_cache_indices` returns: per request, a sliding layer's
    pages (-1 for a block it has recycled) and a global layer's."""

    tokens_per_block = _TOKENS_PER_BLOCK

    def __init__(self, sliding, full):
        self.rows = {False: sliding, True: full}

    def get_batch_cache_indices(self, request_ids, layer_idx, num_blocks_per_seq):
        rows = self.rows[layer_idx == 1]
        return [rows[r][:n] for r, n in zip(request_ids, num_blocks_per_seq)]


def _page_tables(manager, cached, new):
    core = SimpleNamespace(
        dims=SimpleNamespace(
            num_layers=2, sliding_window=_WINDOW, is_global=lambda layer: layer == 1
        ),
        _shared_tables=None,
        _check_live_pages=modeling.Inkling2SmallCore._check_live_pages,
    )
    blocks = [-(-(c + n) // _TOKENS_PER_BLOCK) for c, n in zip(cached, new)]
    request_ids = list(range(len(cached)))
    return modeling.Inkling2SmallCore._page_tables(
        core, manager, request_ids, blocks, cached, torch.device("cpu")
    )


@pytest.mark.parametrize("history", [20, 23], ids=["window_starts_in_block_3", "in_block_4"])
def test_pages_recycled_before_the_window_are_not_read(history):
    """A decode step at `history` reads tokens `history - 7 .. history`: from block 3 at 20 and
    from block 4 at 23, where block 3 may be gone too."""
    first = (history + 1 - _WINDOW) // _TOKENS_PER_BLOCK
    sliding = [-1] * first + list(range(first, 6))
    manager = _Manager(sliding=[sliding, [10, 11]], full=[list(range(6)), [10, 11]])
    tables = _page_tables(manager, cached=[history, 0], new=[1, 6])
    # A table is as wide as the longest request's blocks; a shorter request's row ends in -1.
    assert tables[0].tolist() == [sliding, [10, 11, -1, -1, -1, -1]]
    assert tables[1].tolist() == [list(range(6)), [10, 11, -1, -1, -1, -1]]


@pytest.mark.parametrize(
    "sliding, full, hole",
    [
        ([-1, -1, -1, -1, 4, 5], list(range(6)), "layer 0: request 0 reads KV cache block 3,"),
        ([-1, -1, -1, 3, 4, 5], [-1, 1, 2, 3, 4, 5], "layer 1: request 0 reads KV cache block 0,"),
        ([-1, -1, -1, 3, 4], list(range(6)), "layer 0: request 0 reads KV cache block 5,"),
    ],
    ids=["sliding_page_inside_the_window", "global_page", "row_shorter_than_the_read"],
)
def test_a_missing_page_the_step_reads_is_an_error(sliding, full, hole):
    manager = _Manager(sliding=[sliding], full=[full])
    with pytest.raises(RuntimeError, match=re.escape(hole)):
        _page_tables(manager, cached=[20], new=[1])


def test_the_model_names_the_quant_config_it_could_not_find(tmp_path):
    config = SimpleNamespace(
        text_config=SimpleNamespace(), _name_or_path=str(tmp_path), torch_dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match=re.escape(f"no hf_quant_config.json in '{tmp_path}'")):
        modeling.Inkling2SmallCore(SimpleNamespace(pretrained_config=config))


def test_the_model_refuses_a_config_without_a_dtype():
    config = SimpleNamespace(text_config=SimpleNamespace(), torch_dtype=None)
    with pytest.raises(ValueError, match="declares no dtype"):
        modeling.ModelingV2Inkling2SmallNvfp4Sm103Tep4(SimpleNamespace(pretrained_config=config))
