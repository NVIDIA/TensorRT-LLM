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
"""CPU tests for the MiniMax-M3 VL forward under speculative decoding and KV reuse.

``MiniMaxM3VLForConditionalGeneration.forward`` owes the executor and the
speculative-decoding workers this contract:

* a multimodal prefill fuses the vision rows into ``inputs_embeds`` in prompt
  order and forwards the pre-fusion token ids as ``orig_input_ids``
  (``input_ids`` is None after fusion, and ``SpecDecOneEngineForCausalLM.forward``
  reads ``orig_input_ids`` for the DSpark/DFlash, MTP and Eagle workers);
* a request whose placeholder rows are all cached runs no vision forward; when
  every request in the batch is, fusion is skipped and ``input_ids`` passes
  through unchanged;
* a partial reuse hit or a chunk boundary inside the placeholder span keeps
  exactly the rows ``find_input_mm_embeds`` selects
  (``[num_cached : num_cached + in_chunk]``), advancing the pre-concatenated row
  cursor by ``total_embeds_in_request`` per request.

The model is built with ``object.__new__`` plus ``nn.Module.__init__`` so no
checkpoint, config or CUDA is needed; the vision tower is a fake that maps each
``merge_size**2`` block of ``pixel_values`` rows to one embedding row whose value
is the block's tag, so a wrong slice or offset shows up as a wrong value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_speculative
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData

pytestmark = pytest.mark.cpu_only

_VOCAB = 64
_HIDDEN = 8
_IMG = 50  # image placeholder id (in-vocab, as in the checkpoint)
_VID = 51  # video placeholder id
_TXT = 7  # an ordinary text token id
_MERGE = 2  # fake tower: merge_size**2 = 4 pixel rows per embedding row
_PATCH_FEATURES = 6

_VLModel = Tuple[MiniMaxM3VLForConditionalGeneration, Dict[str, Any]]


class _FakeVisionTower(nn.Module):
    """Returns one row per ``merge**2`` pixel rows, valued by the block's tag."""

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.calls: List[int] = []

    def forward(self, pixel_values: torch.Tensor, grid_thw: Any) -> torch.Tensor:
        tags = pixel_values[:: _MERGE * _MERGE, 0]
        self.calls.append(int(tags.numel()))
        return tags.unsqueeze(1).expand(-1, _HIDDEN).to(torch.float32).clone()


def _pixel_values(tags: List[float]) -> torch.Tensor:
    """``len(tags) * merge**2`` pixel rows; every row of a block carries its tag."""
    t = torch.tensor(tags, dtype=torch.float32).repeat_interleave(_MERGE * _MERGE)
    return t.unsqueeze(1).expand(-1, _PATCH_FEATURES).clone()


def _grid(n_tokens: int) -> torch.Tensor:
    """``[1, 3]`` grid: t=1, h*w = ``n_tokens * merge**2`` patches."""
    return torch.tensor([[1, _MERGE, n_tokens * _MERGE]], dtype=torch.int64)


def _runtime(full_ids: List[int], past_seen: int, chunk_end: int) -> MultimodalRuntimeData:
    mask = torch.tensor([1 if t in (_IMG, _VID) else 0 for t in full_ids], dtype=torch.int64)
    return MultimodalRuntimeData(
        past_seen_token_num=past_seen,
        chunk_end_pos=chunk_end,
        embed_mask_cumsum=torch.cumsum(mask, dim=0),
    )


def _params(
    image_tags: List[List[float]],
    runtime: Optional[MultimodalRuntimeData] = None,
) -> MultimodalParams:
    """One request carrying ``len(image_tags)`` images (rows concatenated in order)."""
    pv = torch.cat([_pixel_values(tags) for tags in image_tags], dim=0)
    grid = torch.cat([_grid(len(tags)) for tags in image_tags], dim=0)
    return MultimodalParams(
        multimodal_data={"image": {"pixel_values": pv, "image_grid_thw": grid}},
        multimodal_runtime=runtime,
    )


@pytest.fixture
def vl_model(monkeypatch: pytest.MonkeyPatch) -> _VLModel:
    """A VL wrapper with fake tower/embedding and a captured text forward."""
    model = object.__new__(MiniMaxM3VLForConditionalGeneration)
    nn.Module.__init__(model)
    embed = nn.Embedding(_VOCAB, _HIDDEN)
    with torch.no_grad():
        # Text embedding of token t is the constant -(t + 1): distinct from any
        # (non-negative) vision tag.
        embed.weight.copy_(
            -(torch.arange(_VOCAB, dtype=torch.float32) + 1).unsqueeze(1).expand(-1, _HIDDEN)
        )
    model.model = nn.Module()
    model.model.embed_tokens = embed
    model.vision_tower = _FakeVisionTower()
    model.register_buffer(
        "mm_token_ids", torch.tensor([_IMG, _VID], dtype=torch.int32), persistent=False
    )

    captured: Dict[str, Any] = {}

    def fake_text_forward(
        self: Any,
        attn_metadata: Any = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        captured.update(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kwargs=kwargs,
        )
        return torch.zeros(1)

    # ``MiniMaxM3ForCausalLM`` does not override ``forward``; the wrapper's
    # ``super().forward`` resolves to ``SpecDecOneEngineForCausalLM.forward``.
    monkeypatch.setattr(
        modeling_speculative.SpecDecOneEngineForCausalLM, "forward", fake_text_forward
    )
    return model, captured


def _run(
    model: MiniMaxM3VLForConditionalGeneration,
    ids: List[int],
    params: List[Optional[MultimodalParams]],
    **extra: Any,
) -> torch.Tensor:
    input_ids = torch.tensor(ids, dtype=torch.int32)
    position_ids = torch.arange(len(ids), dtype=torch.int32).unsqueeze(0)
    model.forward(
        attn_metadata=object(),
        input_ids=input_ids,
        position_ids=position_ids,
        multimodal_params=params,
        **extra,
    )
    return input_ids


def _fused_rows(captured: Dict[str, Any], ids: List[int]) -> List[float]:
    """First feature of every row of the fused embeddings (one value per position)."""
    embeds = captured["inputs_embeds"]
    assert embeds is not None and embeds.shape == (len(ids), _HIDDEN)
    return embeds[:, 0].tolist()


def test_multimodal_prefill_fuses_images_in_order_and_forwards_orig_input_ids(
    vl_model: _VLModel,
) -> None:
    """Two images (4 + 2 tokens), no runtime data: rows fused in prompt order, ids kept for spec."""
    model, captured = vl_model
    ids = [_TXT, _IMG, _IMG, _IMG, _IMG, _TXT, _IMG, _IMG, _TXT]
    spec_metadata = object()
    input_ids = _run(
        model, ids, [_params([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]])], spec_metadata=spec_metadata
    )

    assert captured["input_ids"] is None
    assert torch.equal(captured["kwargs"]["orig_input_ids"], input_ids)
    assert captured["kwargs"]["spec_metadata"] is spec_metadata
    assert "multimodal_params" not in captured["kwargs"]
    assert _fused_rows(captured, ids) == [-8.0, 1.0, 2.0, 3.0, 4.0, -8.0, 5.0, 6.0, -8.0]


def test_full_image_prefix_reuse_hit_skips_fusion(vl_model: _VLModel) -> None:
    """Every placeholder row is cached: no vision forward, no fusion, ``input_ids`` unchanged."""
    model, captured = vl_model
    full = [_TXT, _IMG, _IMG, _IMG, _IMG, _TXT, _TXT]
    # The cached prefix covers the whole image span; this forward computes the tail only.
    runtime = _runtime(full, past_seen=5, chunk_end=len(full))
    assert runtime.num_cached_mm_tokens == 4 and runtime.num_mm_tokens_in_chunk == 0
    chunk_ids = full[5:]
    input_ids = _run(model, chunk_ids, [_params([[1.0, 2.0, 3.0, 4.0]], runtime)])

    assert torch.equal(captured["input_ids"], input_ids)
    assert captured["inputs_embeds"] is None
    assert "orig_input_ids" not in captured["kwargs"]
    assert model.vision_tower.calls == []


def test_fully_reused_request_in_a_batch_runs_no_vision_forward(vl_model: _VLModel) -> None:
    """A fully cached request next to a fresh one: only the fresh image is encoded and fused."""
    model, captured = vl_model
    full_a = [_TXT, _IMG, _IMG, _IMG, _IMG, _TXT, _TXT]
    full_b = [_TXT, _IMG, _IMG, _IMG, _IMG]
    runtime_a = _runtime(full_a, past_seen=5, chunk_end=len(full_a))
    runtime_b = _runtime(full_b, past_seen=0, chunk_end=len(full_b))
    assert runtime_a.num_mm_tokens_in_chunk == 0
    ids = full_a[5:] + full_b
    params: List[Optional[MultimodalParams]] = [
        _params([[1.0, 2.0, 3.0, 4.0]], runtime_a),
        _params([[10.0, 11.0, 12.0, 13.0]], runtime_b),
    ]
    input_ids = _run(model, ids, params)

    assert torch.equal(captured["kwargs"]["orig_input_ids"], input_ids)
    assert _fused_rows(captured, ids) == [-8.0, -8.0, -8.0, 10.0, 11.0, 12.0, 13.0]
    assert model.vision_tower.calls == [4]


def test_chunk_boundary_inside_image_span_keeps_only_the_chunk_rows(vl_model: _VLModel) -> None:
    """Chunked prefill: the chunk ends inside the span; cached rows and rows beyond the chunk are dropped."""
    model, captured = vl_model
    full = [_TXT, _IMG, _IMG, _IMG, _IMG, _IMG, _IMG, _TXT]
    runtime = _runtime(full, past_seen=2, chunk_end=5)
    assert (runtime.num_cached_mm_tokens, runtime.num_mm_tokens_in_chunk) == (1, 3)
    chunk_ids = full[2:5]
    _run(model, chunk_ids, [_params([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], runtime)])

    assert _fused_rows(captured, chunk_ids) == [2.0, 3.0, 4.0]


def test_batched_requests_advance_the_row_cursor_per_request(vl_model: _VLModel) -> None:
    """Two requests, one image each; the second has two cached rows. Rows are taken per request."""
    model, captured = vl_model
    full_a = [_TXT, _IMG, _IMG, _IMG, _IMG]
    full_b = [_TXT, _IMG, _IMG, _IMG, _IMG, _IMG, _IMG, _TXT]
    runtime_a = _runtime(full_a, past_seen=0, chunk_end=len(full_a))
    runtime_b = _runtime(full_b, past_seen=3, chunk_end=len(full_b))
    assert (runtime_b.num_cached_mm_tokens, runtime_b.num_mm_tokens_in_chunk) == (2, 4)
    ids = full_a + full_b[3:]
    params: List[Optional[MultimodalParams]] = [
        _params([[10.0, 11.0, 12.0, 13.0]], runtime_a),
        _params([[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]], runtime_b),
    ]
    input_ids = _run(model, ids, params)

    assert torch.equal(captured["kwargs"]["orig_input_ids"], input_ids)
    assert _fused_rows(captured, ids) == [
        -8.0, 10.0, 11.0, 12.0, 13.0,  # request A, nothing cached
        22.0, 23.0, 24.0, 25.0, -8.0,  # request B, rows [2:6]
    ]  # fmt: skip
