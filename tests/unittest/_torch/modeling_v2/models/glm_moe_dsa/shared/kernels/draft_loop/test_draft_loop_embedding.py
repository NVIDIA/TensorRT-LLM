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
"""draft_loop draft embedding versus the native ``Embedding`` (bitwise)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _draft_loop_fixtures import bits, load_checkpoint_tensor, real_token_ids  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop.embedding import (  # noqa: E402, E501
    DraftEmbedding,
    draft_embedding,
    wrap_draft_embedding,
)
from tensorrt_llm._torch.modules.embedding import Embedding  # noqa: E402

VOCAB = 154880
HIDDEN = 6144


def _native_embedding(weight: torch.Tensor) -> Embedding:
    module = Embedding(weight.shape[0], weight.shape[1], dtype=weight.dtype).cuda()
    module.weight.data = weight
    return module


def _edge_ids(vocab: int, dtype: torch.dtype) -> torch.Tensor:
    info = torch.iinfo(dtype)
    values = [0, 1, vocab - 1, -1, vocab, vocab + 7, info.min, info.max, 12345]
    return torch.tensor(values, dtype=dtype, device="cuda")


@pytest.fixture(scope="module")
def synthetic_weight():
    gen = torch.Generator(device="cuda").manual_seed(8)
    weight = torch.randn((VOCAB, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=gen)
    # Keep special bit patterns in the table so a bit-copy is required.
    weight[3, :4] = torch.tensor([-0.0, float("inf"), float("-inf"), float("nan")])
    return weight


@pytest.fixture(scope="module")
def real_weight():
    weight = load_checkpoint_tensor("model.embed_tokens.weight")
    if weight is None:
        pytest.skip("GLM-5.2 checkpoint not available")
    assert weight.shape == (VOCAB, HIDDEN) and weight.dtype == torch.bfloat16
    return weight


@pytest.mark.parametrize("num_rows", [1, 2, 5, 6, 7, 64])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_matches_native_synthetic(synthetic_weight, num_rows, id_dtype):
    native = _native_embedding(synthetic_weight)
    gen = torch.Generator(device="cuda").manual_seed(num_rows)
    ids = torch.randint(0, VOCAB, (num_rows,), generator=gen, device="cuda", dtype=id_dtype)
    ids[0] = 3
    expected = native(ids)
    got = draft_embedding(ids, synthetic_weight)
    assert got.shape == expected.shape and got.dtype == expected.dtype
    assert torch.equal(bits(got), bits(expected))


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_out_of_range_ids_give_zero_rows(synthetic_weight, id_dtype):
    native = _native_embedding(synthetic_weight)
    ids = _edge_ids(VOCAB, id_dtype)
    expected = native(ids)
    got = draft_embedding(ids, synthetic_weight)
    assert torch.equal(bits(got), bits(expected))
    # +0.0, not -0.0, for every invalid row.
    invalid = (ids < 0) | (ids >= VOCAB)
    assert torch.all(bits(got[invalid]) == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_other_dtypes(dtype):
    weight = torch.randn((1000, 384), dtype=dtype, device="cuda")
    native = _native_embedding(weight)
    ids = torch.tensor([0, 999, 1000, -5, 17], dtype=torch.int32, device="cuda")
    assert torch.equal(bits(draft_embedding(ids, weight)), bits(native(ids)))


def test_real_weights_real_ids(real_weight):
    native = _native_embedding(real_weight)
    ids_host = real_token_ids(512)
    assert ids_host is not None
    for chunk in (1, 6):
        for start in range(0, 512, 37):
            ids = ids_host[start : start + chunk].cuda()
            assert torch.equal(bits(draft_embedding(ids, real_weight)), bits(native(ids)))
    # Every vocabulary row once.
    all_ids = torch.arange(VOCAB, dtype=torch.int32, device="cuda")
    assert torch.equal(bits(draft_embedding(all_ids, real_weight)), bits(native(all_ids)))


def test_cuda_graph_replay(synthetic_weight):
    """Replays read the static id buffer; each replay matches the native result."""
    native = _native_embedding(synthetic_weight)
    static_ids = torch.zeros((6,), dtype=torch.int32, device="cuda")
    draft_embedding(static_ids, synthetic_weight)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = draft_embedding(static_ids, synthetic_weight)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for replay in range(20):
        new_ids = torch.randint(
            -3, VOCAB + 3, (6,), generator=gen, device="cuda", dtype=torch.int32
        )
        static_ids.copy_(new_ids)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(static_out), bits(native(new_ids))), f"replay {replay}"


def test_wrapper_uses_fused_path_and_proxies_attributes(synthetic_weight, monkeypatch):
    native = _native_embedding(synthetic_weight)
    wrapper = wrap_draft_embedding(native)
    assert wrap_draft_embedding(native) is wrapper
    assert wrapper.ineligible_reason is None
    assert wrapper.weight is native.weight
    assert wrapper.num_embeddings == VOCAB

    calls = []
    import tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop.embedding as emb  # noqa: E501

    real = emb.draft_embedding
    monkeypatch.setattr(emb, "draft_embedding", lambda *a: calls.append(1) or real(*a))
    ids = torch.tensor([5, 6], dtype=torch.int32, device="cuda")
    assert torch.equal(bits(wrapper(ids)), bits(native(ids)))
    assert calls == [1]


def test_wrapper_native_fallbacks(synthetic_weight, monkeypatch):
    native = _native_embedding(synthetic_weight)
    wrapper = DraftEmbedding(native)
    import tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop.embedding as emb  # noqa: E501

    def _fail(*_):
        raise AssertionError("fused path taken for an ineligible input")

    monkeypatch.setattr(emb, "draft_embedding", _fail)
    # 2-D ids and non-contiguous ids take the native module.
    ids_2d = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda")
    assert torch.equal(bits(wrapper(ids_2d)), bits(native(ids_2d)))
    ids_strided = torch.arange(10, dtype=torch.int32, device="cuda")[::2]
    assert torch.equal(bits(wrapper(ids_strided)), bits(native(ids_strided)))
    # A tensor-parallel module is ineligible as a whole.
    native.tp_size = 2
    assert DraftEmbedding(native).ineligible_reason == "embedding is tensor-parallel"
    native.tp_size = 1


def test_empty_ids(synthetic_weight):
    ids = torch.empty((0,), dtype=torch.int32, device="cuda")
    out = draft_embedding(ids, synthetic_weight)
    assert out.shape == (0, HIDDEN)
