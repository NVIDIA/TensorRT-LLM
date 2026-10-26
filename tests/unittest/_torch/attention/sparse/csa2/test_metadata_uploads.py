# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host metadata uploads retain live geometry and per-owner request state."""

from contextlib import nullcontext
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.features.spec_decoding import SpecDecodingMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2Metadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            skip_pre_blackwell,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ],
    ),
]


def _bare_csa2_metadata() -> TrtllmAttentionMetadata:
    """Metadata built without ``__init__``, carrying a bare CSA2 extension."""
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata.replace_ext(CSA2Metadata(metadata))
    metadata.replace_ext(SpecDecodingMetadata(metadata))
    return metadata


def _metadata(device):
    metadata = _bare_csa2_metadata()
    metadata.is_cuda_graph = False
    metadata.num_sparse_topk = 256
    for name in ("kv_lens", "prompt_lens_cpu", "host_request_types"):
        setattr(metadata, name, torch.empty(4, dtype=torch.int32, device="cpu"))
    for name in ("kv_lens_cuda", "prompt_lens_cuda"):
        setattr(metadata, name, torch.empty(4, dtype=torch.int32, device=device))
    metadata.host_total_kv_lens = torch.zeros(2, dtype=torch.int32, device="cpu")
    csa2 = metadata.ext(CSA2Metadata)
    csa2.query_lens_host = torch.empty(4, dtype=torch.int32, device="cpu")
    csa2.query_lens_device = torch.empty(4, dtype=torch.int32, device=device)
    csa2.prepared_cu_q = torch.empty(5, dtype=torch.int32, device=device)
    csa2.prepared_cu_kv = torch.empty(5, dtype=torch.int32, device=device)
    return metadata


@pytest.mark.parametrize("device", _DEVICES)
def test_context_uploads_refresh_changing_geometry(device):
    metadata = _metadata(device)
    stream = torch.cuda.Stream() if device == "cuda" else None
    observed = []
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        if stream is not None:
            torch.cuda._sleep(2_000_000)
        for lengths in ([2, 1], [1, 2], [3], [1, 1, 1], [2, 1], [2, 1]):
            metadata.ext(CSA2Metadata)._bind_context_tile(list(lengths))
            observed.append(
                (
                    list(lengths),
                    tuple(
                        tensor.clone()
                        for tensor in (
                            metadata.seq_lens_cuda,
                            metadata.kv_lens_cuda_runtime,
                            metadata.prompt_lens_cuda_runtime,
                            metadata.cu_q_seqlens,
                            metadata.cu_kv_seqlens,
                        )
                    ),
                )
            )
    if stream is not None:
        stream.synchronize()
    for lengths, actual in observed:
        kv = [256 + length - 1 for length in lengths]
        expected = [
            lengths,
            kv,
            lengths,
            list(accumulate(lengths, initial=0)),
            list(accumulate(kv, initial=0)),
        ]
        for tensor, values in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                tensor.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0
            )


@pytest.mark.parametrize("device", _DEVICES)
def test_shared_context_uploads_preserve_owners_and_live_positions(device):
    metadata = _metadata(device)
    layout = SimpleNamespace(
        window_size=128,
        compress_ratios=(1, 2),
        layer=lambda index: SimpleNamespace(kv_source=(0, 1, None)[index]),
    )
    metadata.kv_cache_manager = SimpleNamespace(layout=layout, tokens_per_block=128)
    stream = torch.cuda.Stream() if device == "cuda" else None
    observed = []
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        if stream is not None:
            torch.cuda._sleep(2_000_000)
        for step, lengths in enumerate(([2, 1], [1, 2], [3], [1, 1, 1], [2, 1])):
            requests = len(lengths)
            first_positions = [190 + step * 11 + request * 37 for request in range(requests)]
            floors = [80 + step + request for request in range(requests)]
            positions = [
                start + offset
                for start, length in zip(first_positions, lengths)
                for offset in range(length)
            ]
            metadata._num_ctx_tokens = sum(lengths)
            metadata.ext(CSA2Metadata).csa2_num_context_requests = requests
            metadata.ext(CSA2Metadata).csa2_request_lengths = lengths
            metadata.ext(CSA2Metadata).csa2_positions = torch.tensor(
                positions, dtype=torch.int32, device=device
            )
            metadata.ext(CSA2Metadata).csa2_replay_start_positions = torch.tensor(
                floors, device=device
            )
            metadata.ext(CSA2Metadata).csa2_token_requests = torch.tensor(
                [request for request, length in enumerate(lengths) for _ in range(length)],
                device=device,
            )
            metadata.ext(CSA2Metadata)._csa2_main_domain_counts = {
                owner: [step + owner + request + 1 for request in range(requests)]
                for owner in (0, 1)
            }
            metadata.ext(CSA2Metadata)._csa2_shared_domains = {}
            metadata.ext(CSA2Metadata)._csa2_swa_page_tables = {
                layer: torch.full((requests, 2), 20 + layer + step, device=device)
                for layer in (0, 1, 2)
            }
            metadata.ext(CSA2Metadata).csa2_global_page_tables = {
                owner: torch.full((requests, 2), 40 + owner + step, device=device)
                for owner in (0, 1)
            }
            q = torch.empty(sum(lengths), 1, 512, device=device)
            plans = [
                metadata.ext(CSA2Metadata)._shared_context_plan(q, 64, layer) for layer in (0, 1, 2)
            ]
            swa_offsets = list(accumulate((length + 127 for length in lengths), initial=1))
            # Read all owner plans only after the next owner's uploads have been submitted.
            for owner, plan in zip((0, 1, None), plans, strict=True):
                main_counts = (
                    [0] * requests
                    if owner is None
                    else metadata.ext(CSA2Metadata)._csa2_main_domain_counts[owner]
                )
                expected = (
                    [max(0, start - 127, floor) for start, floor in zip(first_positions, floors)],
                    swa_offsets,
                    list(accumulate(main_counts, initial=swa_offsets[-1])),
                )
                observed.append(
                    (
                        tuple(
                            value.clone()
                            for value in (plan.swa_starts, plan.swa_offsets, plan.main_offsets)
                        ),
                        expected,
                    )
                )
    if stream is not None:
        stream.synchronize()
    for actual, expected in observed:
        for tensor, values in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                tensor.cpu(), torch.tensor(values, dtype=torch.int64), rtol=0, atol=0
            )
