# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

from tensorrt_llm._torch.modules.gemma_attention_mask import build_gemma_attention_mask


def _reference(types: list[int], prefix: int, window: int | None) -> torch.Tensor:
    blobs = []
    blob = 0
    previous = 0
    for token_type in types:
        vision = token_type in (1, 2)
        if vision and token_type != previous:
            blob += 1
        blobs.append(blob if vision else 0)
        previous = token_type
    rows = []
    for query in range(len(types)):
        row = []
        for key in range(prefix + len(types)):
            causal = key <= prefix + query and (window is None or key > prefix + query - window)
            same_blob = key >= prefix and blobs[query] != 0 and blobs[query] == blobs[key - prefix]
            row.append(causal or same_blob)
        rows.extend(row)
    return torch.tensor(rows, dtype=torch.bool)


@pytest.mark.parametrize("gemma3", [False, True])
@pytest.mark.parametrize("window", [None, 1, 4, 128])
def test_gemma_attention_mask_rules(gemma3: bool, window: int | None) -> None:
    # Adjacent image/video blobs, audio, empty requests, and trailing decode token.
    requests = [[1, 1, 2, 2, 3, 3, 0, 1, 1], [], [0, 2, 2, 0, 1]]
    if gemma3:
        requests = [[int(t == 1) for t in request] for request in requests]
    prefixes = [0, 0, 0] if gemma3 else [7, 0, 2]
    flat = [t for request in requests for t in request] + [1]
    # Exercise input views with a nonzero storage offset and nonunit stride.
    storage = torch.zeros(
        2 * len(flat) + 1, dtype=torch.bool if gemma3 else torch.int64, device="cuda"
    )
    types = storage[1::2]
    types.copy_(torch.tensor(flat, device="cuda"))
    original_storage = storage.clone()
    actual = build_gemma_attention_mask(types, [len(r) for r in requests], prefixes, window)
    expected = torch.cat([_reference(r, p, window) for r, p in zip(requests, prefixes)])
    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("length", "prefix"),
    [(0, 0), (0, 5), (1, 0), (17, 1007), (17, 1008), (17, 1009), (17, 2049), (1025, 3)],
)
def test_gemma_attention_mask_tile_boundaries(length: int, prefix: int) -> None:
    # Prefix-only tiles, partially filled final tiles, and vision across a tile boundary.
    types = ([1, 1, 2, 2, 3, 0, 1] * ((length + 6) // 7))[:length]
    actual = build_gemma_attention_mask(
        torch.tensor(types, dtype=torch.int64, device="cuda"), [length], [prefix], 4
    )
    torch.testing.assert_close(actual.cpu(), _reference(types, prefix, 4), rtol=0, atol=0)


def test_gemma_attention_mask_empty_batch() -> None:
    types = torch.tensor([1, 2, 3], device="cuda")
    result = build_gemma_attention_mask(types, [], [])
    assert result.shape == (0,)
    assert result.dtype == torch.bool
    assert result.device == types.device


def test_gemma_attention_mask_capture() -> None:
    types = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], device="cuda")
    build_gemma_attention_mask(types, [8], [5], 4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = build_gemma_attention_mask(types, [8], [5], 4)
    types.zero_()
    graph.replay()
    torch.testing.assert_close(result.cpu(), _reference([0] * 8, 5, 4), rtol=0, atol=0)


def test_gemma_attention_mask_peak() -> None:
    # A second quadratic tensor exceeds the allowance by tens of MiB.
    length = 4096
    types = torch.zeros(2 * length, dtype=torch.int64, device="cuda")
    types[64:128] = 1
    warm = build_gemma_attention_mask(types, [length, length], [0, 1024], 512)
    del warm
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = build_gemma_attention_mask(types, [length, length], [0, 1024], 512)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert peak <= result.numel() + 2 * 1024**2
