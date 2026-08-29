# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for speculative attention-metadata save and restore lifecycles."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.draft_target import DraftTargetOneModelWorker
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.pard import PARDWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _FailingSpecWorker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return 1

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        attn_metadata = kwargs["attn_metadata"]
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)
        _mutate_context_metadata(attn_metadata)
        raise RuntimeError("simulated draft failure")


def _make_metadata(*, is_cuda_graph: bool = False) -> TrtllmAttentionMetadata:
    metadata = TrtllmAttentionMetadata(
        seq_lens=None,
        seq_lens_kv=None,
        num_contexts=1,
        max_num_requests=2,
        max_num_tokens=4,
        is_cuda_graph=is_cuda_graph,
    )
    metadata._seq_lens = torch.tensor([3, 1], dtype=torch.int32)
    metadata._seq_lens_cuda = metadata._seq_lens.cuda()
    metadata.kv_lens_cuda = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    metadata.host_request_types[:1].fill_(0)
    metadata.host_request_types[1:2].fill_(1)
    metadata.on_update()
    return metadata


def _mutate_context_metadata(metadata: TrtllmAttentionMetadata) -> None:
    metadata._seq_lens.fill_(1)
    metadata._seq_lens_cuda.fill_(1)
    metadata.on_update()
    metadata.host_request_types[: metadata.num_contexts].fill_(1)
    metadata.num_contexts = 0
    assert metadata.num_contexts == 0
    assert metadata.num_ctx_tokens == 0
    assert metadata.num_generations == 2
    assert metadata.context_lens.numel() == 0


def _assert_metadata_restored(metadata: TrtllmAttentionMetadata) -> None:
    assert metadata.num_contexts == 1
    assert metadata.num_tokens == 4
    assert metadata.num_ctx_tokens == 3
    assert metadata.num_generations == 1
    assert metadata.context_lens.tolist() == [3]
    assert metadata.host_request_types[:2].tolist() == [0, 1]


def _make_worker(worker_type: type[SpecWorkerBase]) -> SpecWorkerBase:
    worker = object.__new__(worker_type)
    SpecWorkerBase.__init__(worker)
    return worker


def test_spec_worker_metadata_restore_preserves_context_metadata() -> None:
    metadata = _make_metadata()
    worker = _make_worker(Eagle3OneModelWorker)
    original_host_request_types = metadata.host_request_types

    with pytest.raises(AssertionError, match="requires a paired prepare"):
        worker._restore_attn_metadata_from_spec_dec(metadata)
    worker._prepare_attn_metadata_for_spec_dec(metadata)
    with pytest.raises(AssertionError, match="prepare and restore calls must be paired"):
        worker._prepare_attn_metadata_for_spec_dec(metadata)
    other_metadata = _make_metadata()
    with pytest.raises(AssertionError, match="same object"):
        worker._restore_attn_metadata_from_spec_dec(other_metadata)
    _mutate_context_metadata(metadata)
    worker._restore_attn_metadata_from_spec_dec(metadata)

    _assert_metadata_restored(metadata)
    assert metadata.host_request_types is original_host_request_types
    assert worker._saved_num_contexts is None
    assert worker._saved_attn_metadata is None


def test_partial_prepare_restores_metadata() -> None:
    metadata = _make_metadata()
    original_seq_lens = metadata._seq_lens
    worker = _make_worker(Eagle3OneModelWorker)

    with pytest.raises(AttributeError):
        worker._prepare_attn_metadata_for_spec_dec(metadata, "missing_field")

    assert metadata._seq_lens is original_seq_lens
    assert not metadata.has_spec_dec_saved_state
    assert worker._saved_num_contexts is None
    assert worker._saved_attn_metadata is None


def test_forward_failure_restores_metadata() -> None:
    metadata = _make_metadata()
    worker = _make_worker(_FailingSpecWorker)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="simulated draft failure"):
            SpecWorkerBase.forward(worker, attn_metadata=metadata, spec_metadata=None)
        _assert_metadata_restored(metadata)
        assert not metadata.has_spec_dec_saved_state
        assert worker._saved_num_contexts is None
        assert worker._saved_attn_metadata is None


@pytest.mark.parametrize(
    ("worker_type", "prepare_method"),
    [
        (DraftTargetOneModelWorker, "_prepare_attn_metadata_for_draft_target"),
        (PARDWorker, "_prepare_attn_metadata_for_pard"),
        (DFlashWorker, "_prepare_attn_metadata_for_dflash"),
    ],
)
@pytest.mark.parametrize(
    ("is_cuda_graph", "is_capturing", "restores_kv_lens"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_worker_specific_prepare_uses_base_metadata_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    worker_type: type[SpecWorkerBase],
    prepare_method: str,
    is_cuda_graph: bool,
    is_capturing: bool,
    restores_kv_lens: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: is_capturing)
    metadata = _make_metadata(is_cuda_graph=is_cuda_graph)
    worker = _make_worker(worker_type)
    spec_metadata = SimpleNamespace(is_cuda_graph=is_cuda_graph)
    original_kv_lens_tensor = metadata.kv_lens_cuda
    original_kv_lens = original_kv_lens_tensor.clone()

    getattr(worker, prepare_method)(metadata, spec_metadata)
    metadata.kv_lens_cuda.fill_(9)
    _mutate_context_metadata(metadata)
    worker._restore_attn_metadata_from_spec_dec(metadata)

    _assert_metadata_restored(metadata)
    assert worker._saved_num_contexts is None
    assert worker._saved_attn_metadata is None
    assert metadata.kv_lens_cuda is original_kv_lens_tensor
    if restores_kv_lens:
        assert torch.equal(metadata.kv_lens_cuda, original_kv_lens)
    else:
        assert torch.equal(metadata.kv_lens_cuda, torch.full_like(original_kv_lens, 9))
