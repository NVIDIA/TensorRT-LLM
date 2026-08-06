# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The device fill/row-map path must reproduce the host path token-for-token.

``fill_bucket_device`` and ``build_row_maps_device`` are the capture-safe
replacements for ``RaggedVerifyLayout.fill_bucket`` and the host row-map
staging in prepare(). Any divergence in the per-row allocation or the
per-token KV correction is an attention-reads-wrong-KV bug that surfaces as
silent quality loss (or an IMA), never as an exception -- so the parity here
is exact equality against the host implementation over the whole feasible
space, not spot checks.
"""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_ragged import (
    RaggedVerifyLayout, build_row_maps_device, fill_bucket_device)


def _host_fill(lens, padded_bs, bucket, max_verify_len):
    layout = RaggedVerifyLayout.from_verify_lens(
        torch.tensor(lens, dtype=torch.int32),
        graph_num_tokens=bucket,
        total_verify_tokens=sum(lens),
    )
    return layout.fill_bucket(max_verify_len=max_verify_len,
                              padded_bs=padded_bs)


def _device_fill(lens, padded_bs, bucket, max_verify_len):
    padded = torch.ones(padded_bs, dtype=torch.int32)
    padded[:len(lens)] = torch.tensor(lens, dtype=torch.int32)
    return fill_bucket_device(
        padded,
        num_real=torch.tensor(len(lens)),
        graph_num_tokens=bucket,
        max_verify_len=max_verify_len,
    )


def _feasible(lens, padded_bs, bucket, max_verify_len):
    n_pad = padded_bs - len(lens)
    return (sum(lens) + n_pad <= bucket <= padded_bs * max_verify_len)


def _all_lens(n_real, max_verify_len):
    """Every [1, max]^n_real length vector."""
    if n_real == 0:
        yield []
        return
    for head in range(1, max_verify_len + 1):
        for tail in _all_lens(n_real - 1, max_verify_len):
            yield [head] + tail


@pytest.mark.parametrize("max_verify_len", [2, 4, 6])
def test_fill_parity_exhaustive_small(max_verify_len):
    """Exact host/device agreement over every feasible small case."""
    checked = 0
    for n_real in (1, 2, 3):
        for padded_bs in (n_real, n_real + 1, n_real + 3):
            for lens in _all_lens(n_real, max_verify_len):
                for bucket in range(padded_bs,
                                    padded_bs * max_verify_len + 1):
                    if not _feasible(lens, padded_bs, bucket, max_verify_len):
                        continue
                    host = _host_fill(lens, padded_bs, bucket, max_verify_len)
                    dev = _device_fill(lens, padded_bs, bucket, max_verify_len)
                    assert torch.equal(dev, host.verify_lens), (
                        f"lens={lens} padded_bs={padded_bs} bucket={bucket} "
                        f"max={max_verify_len}: device {dev.tolist()} != "
                        f"host {host.verify_lens.tolist()}")
                    checked += 1
    assert checked > 100


def test_fill_parity_randomized_large():
    """Production-shaped cases: bs up to 128, max_verify_len 6."""
    gen = torch.Generator().manual_seed(20260806)
    max_verify_len = 6
    for _ in range(200):
        n_real = int(torch.randint(1, 129, (1,), generator=gen))
        padded_bs = n_real + int(torch.randint(0, 9, (1,), generator=gen))
        lens = torch.randint(1, max_verify_len + 1, (n_real,),
                             generator=gen).tolist()
        lo, hi = sum(lens) + (padded_bs - n_real), padded_bs * max_verify_len
        bucket = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        host = _host_fill(lens, padded_bs, bucket, max_verify_len)
        dev = _device_fill(lens, padded_bs, bucket, max_verify_len)
        assert torch.equal(dev, host.verify_lens)
        assert int(dev.sum()) == bucket


def test_row_maps_match_prepare_semantics():
    """req_idx/correction must equal what prepare() stages from host lists:
    token o of a request with window v gets correction o - v + 1, so the
    gathered extent walks kv_len - v + 1 .. kv_len."""
    gen = torch.Generator().manual_seed(7)
    for _ in range(50):
        bs = int(torch.randint(1, 65, (1,), generator=gen))
        lens = torch.randint(1, 7, (bs,), generator=gen).to(torch.int32)
        total = int(lens.sum())
        req_idx, corr = build_row_maps_device(lens, graph_num_tokens=total)
        want_req, want_corr = [], []
        for r, v in enumerate(lens.tolist()):
            for o in range(v):
                want_req.append(r)
                want_corr.append(o - v + 1)
        assert req_idx.tolist() == want_req
        assert corr.tolist() == want_corr
        # Composing with a kv_lens gather is refresh_ragged_row_kv_lens:
        # the extents must end exactly at each request's kv_len.
        kv_lens = torch.randint(10, 1000, (bs,), generator=gen).to(torch.int32)
        extents = kv_lens[req_idx] + corr
        ends = torch.cumsum(lens.to(torch.long), 0) - 1
        assert torch.equal(extents[ends], kv_lens)
        assert int(extents.min()) >= 1
