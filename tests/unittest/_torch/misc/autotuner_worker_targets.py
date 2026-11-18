# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A runner for test_autotuner.py's tune-mode builds, whose worker processes import this
module."""

import os
import time

from tensorrt_llm._torch.autotuner import TunableRunner


class RecordingRunner(TunableRunner):
    """A runner whose build takes ``build_s`` seconds and records the process that built it, as a
    line of the file in ``directory`` named for the rows and the tactic; its tactics double the
    input."""

    def __init__(self, directory: str, build_s: float = 0.0):
        self.directory = directory
        self.build_s = build_s

    def unique_id(self):
        return (self.directory, self.build_s)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return [0, 1]

    def build(self, inputs, *, tactic=-1, **kwargs) -> None:
        time.sleep(self.build_s)
        name = f"m{inputs[0].shape[0]}-t{tactic}"
        with open(os.path.join(self.directory, name), "a") as f:
            f.write(f"{os.getpid()}\n")

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0] * 2


class FromShapesRunner(RecordingRunner):
    """A RecordingRunner whose build needs only its inputs' shapes; it records the device and the
    columns of the input it built on."""

    builds_from_shapes = True

    def build(self, inputs, *, tactic=-1, **kwargs) -> None:
        name = f"m{inputs[0].shape[0]}-t{tactic}"
        with open(os.path.join(self.directory, name), "a") as f:
            f.write(f"{os.getpid()} {inputs[0].device.type} {inputs[0].shape[1]}\n")


def allocate(nbytes: int) -> None:
    """A build that allocates ``nbytes`` of GPU memory while it runs, as running a kernel at a
    profile's shape does."""
    import torch

    torch.ones(nbytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()


def record_reserved(path: str) -> None:
    """A build that records the GPU memory its worker's allocator holds when it starts."""
    import torch

    with open(path, "w") as f:
        f.write(str(torch.cuda.memory_reserved()))


class PrebuiltRunner(TunableRunner):
    """A runner whose kernel ships prebuilt (as cuBLAS's): its build does nothing."""

    def unique_id(self):
        return ()

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return [0]

    def build(self, inputs, *, tactic=-1, **kwargs) -> None:
        """Nothing to build."""

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0] * 2
