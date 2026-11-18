# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU regression for NIXL preflight of writes spanning CUDA VMM chunks.

Run in a fresh process with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""

import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent
from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
    MemoryDescs,
    MemoryType,
    TransferOp,
    TransferRequest,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_vmm_cross_chunk_preflight_and_invalid_remote_range():
    size = 64 * 1024 * 1024
    buffer = torch.empty((size,), dtype=torch.uint8, device="cuda")
    buffer[: size // 2].fill_(37)
    buffer[size // 2 :].zero_()
    torch.cuda.synchronize()
    sender = BindingsNixlTransferAgent("vmm_test_sender", num_threads=1)
    receiver = BindingsNixlTransferAgent("vmm_test_receiver", num_threads=1)

    def mem(ptr, length):
        return MemoryDescs(MemoryType.VRAM, [(ptr, length, 0)])

    try:
        sender.register_memory(RegMemoryDescs("VRAM", [(buffer.data_ptr(), size, 0, "src")]))
        receiver.register_memory(RegMemoryDescs("VRAM", [(buffer.data_ptr(), size, 0, "dst")]))
        sender.load_remote_agent("vmm_test_receiver", receiver.get_local_agent_desc())
        # The destination is an interior span across at least one VMM chunk;
        # it is wholly within registered memory, and must be accepted.
        assert sender.check_remote_descs(
            "vmm_test_receiver", mem(buffer.data_ptr() + size // 2, size // 2)
        )
        # This span extends past the end of the registered region; it must not
        # be mistaken for a valid allocation merely because its start is valid.
        assert not sender.check_remote_descs(
            "vmm_test_receiver", mem(buffer.data_ptr() + size - 4096, 8192)
        )
        request = TransferRequest(
            TransferOp.WRITE,
            mem(buffer.data_ptr(), size // 2),
            mem(buffer.data_ptr() + size // 2, size // 2),
            "vmm_test_receiver",
            None,
        )
        status = sender.submit_transfer_requests(request)
        assert status.wait(timeout_ms=30_000)
        assert status.release()
        torch.cuda.synchronize()
        assert torch.equal(buffer[: size // 2], buffer[size // 2 :])
    finally:
        sender.shutdown()
        receiver.shutdown()
