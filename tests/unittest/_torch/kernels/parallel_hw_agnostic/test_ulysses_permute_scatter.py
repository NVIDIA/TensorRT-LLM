# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch


def torch_ref(input_4d, send_ref, recv_ref, my_rank, P):
    """CPU/eager reference for ulyssesPermuteScatterKernel.

    input_4d : [B, S_local, H, D]      bf16
    send/recv: [P, B, S_local, H/P, D] bf16
    For each (b, s, h, d):
      peer    = h // (H/P)
      h_local = h %  (H/P)
      dst     = recv_ref if peer == my_rank else send_ref
      dst[peer, b, s, h_local, d] = input[b, s, h, d]
    """
    B, S_local, H, D = input_4d.shape
    P_check, _, _, H_local, _ = send_ref.shape
    assert P == P_check
    assert H == H_local * P

    # 4D -> [B, S_local, P, H_local, D] via H = P * H_local
    x = input_4d.view(B, S_local, P, H_local, D)
    # All-peer (P, B, S_local, H_local, D)
    all_dst = x.permute(2, 0, 1, 3, 4).contiguous()

    # Split: peer != my_rank -> send_ref, peer == my_rank -> recv_ref
    send_ref.copy_(all_dst)
    recv_ref.zero_()
    recv_ref[my_rank].copy_(all_dst[my_rank])
    send_ref[my_rank].zero_()
    return send_ref, recv_ref


@pytest.mark.parametrize(
    "P,B,S_local,H,D,my_rank",
    [
        # LTX-2 ws=8 shape: heads-per-rank in input is H_full=32, P=4 ulysses
        (4, 2, 1024, 32, 64, 0),
        (4, 2, 1024, 32, 64, 1),
        (4, 2, 1024, 32, 64, 2),
        (4, 2, 1024, 32, 64, 3),
        # WAN-like (alternate H, D)
        (4, 2, 512, 16, 128, 0),
        (8, 1, 256, 16, 128, 5),
        # Smaller smoke shapes
        (2, 1, 128, 8, 64, 0),
        (2, 1, 128, 8, 64, 1),
    ],
)
@torch.inference_mode()
def test_ulysses_permute_scatter_exact_match(P, B, S_local, H, D, my_rank):
    """The kernel is a pure data-movement op (no float arithmetic), so the
    output must match the eager reference byte-exact (max_diff == 0)."""
    torch.manual_seed(0)
    H_local = H // P
    input_4d = torch.randn(B, S_local, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    send = torch.zeros(P, B, S_local, H_local, D, device="cuda", dtype=torch.bfloat16)
    recv = torch.zeros(P, B, S_local, H_local, D, device="cuda", dtype=torch.bfloat16)

    # Reference
    send_ref = torch.zeros_like(send)
    recv_ref = torch.zeros_like(recv)
    torch_ref(input_4d, send_ref, recv_ref, my_rank, P)

    # Kernel
    torch.ops.trtllm.ulysses_permute_scatter(input_4d, send, recv, my_rank, P)

    # 1. The peer==my_rank slot must be in recv (not send), and equal to input's slice
    assert torch.equal(recv[my_rank], recv_ref[my_rank])
    # 2. The peer!=my_rank slots must be in send (not recv), and equal to input's slice
    for peer in range(P):
        if peer == my_rank:
            continue
        assert torch.equal(send[peer], send_ref[peer]), (
            f"send[peer={peer}] mismatch for my_rank={my_rank}"
        )

    # 3. The kernel must leave send[my_rank] and recv[peer!=my_rank] UNTOUCHED
    # (we initialised them to zero; check they're still zero).
    assert torch.count_nonzero(send[my_rank]) == 0
    for peer in range(P):
        if peer == my_rank:
            continue
        assert torch.count_nonzero(recv[peer]) == 0
