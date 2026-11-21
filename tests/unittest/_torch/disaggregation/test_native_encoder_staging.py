# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GPU registration under both PyTorch allocator modes."""

import gc
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("expandable", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_native_encoder_staging_registers_with_expandable_segments(expandable):
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["PYTORCH_ALLOC_CONF"] = f"backend:native,expandable_segments:{expandable}"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1])['_exercise']()",
            str(Path(__file__).resolve()),
        ],
        env=env,
        check=True,
        timeout=120,
    )


def _exercise():
    from tensorrt_llm._torch.disaggregation import encoder_nixl as transport
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

    device = torch.device("cuda:0")
    # A nonzero offset into a multi-chunk allocation reproduces native VMM rejection.
    shape = (2304, 16384)
    size = shape[0] * shape[1] * torch.bfloat16.itemsize
    backing = torch.empty(size + 256, dtype=torch.uint8, device=device)
    source = backing[256:].view(torch.bfloat16).view(shape).fill_(7)
    agent = transport._native_agent("encoder-staging-regression")
    try:
        registration, _, _ = transport._descriptors([source], agent.name)
        if "expandable_segments:True" in os.environ["PYTORCH_ALLOC_CONF"]:
            with pytest.raises(RuntimeError, match="chunk boundary"):
                agent.register_memory(registration)
        else:
            agent.register_memory(registration)
            agent.deregister_memory(registration)

        staged = transport._copy_staging(source, device)
        received = transport._empty_staging(source.shape, source.dtype, device)
        assert staged.data_ptr() != received.data_ptr()
        received.copy_(staged)
        torch.cuda.synchronize(device)
        registration, _, _ = transport._descriptors([staged, received], agent.name)
        agent.register_memory(registration)
        try:
            assert torch.equal(received, source)
            # Receive staging is copied into PyTorch-owned storage for worker IPC.
            exported = received.clone()
            handle = SharedTensorContainer.from_tensor(exported).dump_to_dict()
            SharedTensorContainer.release_dict(handle)
        finally:
            agent.deregister_memory(registration)
        before_free = torch.cuda.mem_get_info(device)[0]
        del staged, received
        gc.collect()
        assert torch.cuda.mem_get_info(device)[0] >= before_free + 2 * size
    finally:
        agent.shutdown()
