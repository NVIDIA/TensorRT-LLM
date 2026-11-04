# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA's launch wrappers release the GIL.

A CUDA host function that takes the GIL (the capturable guided decoder's, for one) can be queued ahead of
FlashMLA's kernels on the same stream. Once the launch queue is full, a FlashMLA call that kept the GIL while
it blocks in a launch would wait on the host function, and the host function on the GIL.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

_CHILD = textwrap.dedent("""
    import time

    import torch

    import tensorrt_llm.flash_mla_cpp_tllm as flash_mla
    from tensorrt_llm._torch.hostfunc import launch_hostfunc

    q = torch.randn(4, 64, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(256, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 256, (4, 1, 64), device="cuda", dtype=torch.int32)
    flash_mla.sparse_prefill_fwd(q, kv, indices, 0.05, 512, None, None)
    torch.cuda.synchronize()

    ran = []
    torch.cuda._sleep(int(3e9))  # the host function runs only after the device works through this
    launch_hostfunc(lambda: ran.append(time.time()))
    for _ in range(20000):  # far more launches than the queue holds
        flash_mla.sparse_prefill_fwd(q, kv, indices, 0.05, 512, None, None)
    torch.cuda.synchronize()
    print("host function ran", len(ran))
""")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FlashMLA sparse prefill on SM100-family GPUs",
)
def test_sparse_prefill_does_not_hold_the_gil_while_launching():
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "host function ran 1" in result.stdout
