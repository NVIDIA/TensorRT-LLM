# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call-owned pre-quantized MXFP8 activation buffers across CUDA streams.

The pair is produced on one stream (the main stream's fused norm) and can be taken on another (the MoE shared
expert's aux stream). Access records the consuming stream on the pair, so once the carrier's references are gone the
caching allocator does not hand the pair's memory to a new tensor on the producing stream while the consumer still
reads it.
"""

import pytest
import torch

from tensorrt_llm._torch.utils import MxFp8QuantizedTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

N = 1 << 22


def _take_on(stream: torch.cuda.Stream, source: MxFp8QuantizedTensor):
    with torch.cuda.stream(stream):
        return source.quantized()


def test_a_take_on_another_stream_keeps_the_pair_alive_until_the_consumer_reads_it() -> None:
    main = torch.cuda.current_stream()
    aux = torch.cuda.Stream()
    q = torch.full((N,), 7, dtype=torch.uint8, device="cuda")
    sf = torch.full((N // 32,), 9, dtype=torch.uint8, device="cuda")
    source = MxFp8QuantizedTensor(q, sf, producer_stream=main)
    del q, sf  # only the carrier holds the pair now
    aux.wait_stream(main)
    pair = _take_on(aux, source)
    assert pair is not None
    with torch.cuda.stream(aux):
        # The consumer is still busy when the producing stream allocates again.
        torch.cuda._sleep(200_000_000)
        got_q = pair[0].clone()
        got_sf = pair[1].clone()
    del pair, source
    # The producing stream allocates and writes same-sized tensors while the consumer sleeps.
    overwrite = [torch.full((N,), 0x55, dtype=torch.uint8, device="cuda") for _ in range(2)]
    overwrite += [torch.full((N // 32,), 0x55, dtype=torch.uint8, device="cuda") for _ in range(2)]
    torch.cuda.synchronize()
    del overwrite
    assert bool((got_q == 7).all()), (
        "the consumer read MXFP8 values the producing stream had reused"
    )
    assert bool((got_sf == 9).all()), "the consumer read scales the producing stream had reused"


def test_a_take_on_the_producing_stream_records_nothing(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, s: recorded.append((id(self), s))
    )
    q = torch.zeros((1024,), dtype=torch.uint8, device="cuda")
    sf = torch.zeros((32,), dtype=torch.uint8, device="cuda")
    source = MxFp8QuantizedTensor(q, sf, producer_stream=torch.cuda.current_stream())
    assert source.quantized() is not None
    assert recorded == []
    aux = torch.cuda.Stream()
    assert _take_on(aux, source) is not None
    assert sorted(tensor_id for tensor_id, _ in recorded) == sorted((id(q), id(sf)))
    assert all(stream == aux for _, stream in recorded)
