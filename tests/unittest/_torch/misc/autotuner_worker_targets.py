# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A runner for test_autotuner.py's tune-mode builds, whose worker processes import this
module."""

import os

from tensorrt_llm._torch.autotuner import TunableRunner


class RecordingRunner(TunableRunner):
    """A runner whose build records, as a file in ``directory``, the rows and the tactic it
    builds and the process that builds them; its tactics double the input."""

    def __init__(self, directory: str):
        self.directory = directory

    def unique_id(self):
        return (self.directory,)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return [0, 1]

    def build(self, inputs, *, tactic=-1, **kwargs) -> None:
        name = f"m{inputs[0].shape[0]}-t{tactic}"
        with open(os.path.join(self.directory, name), "w") as f:
            f.write(str(os.getpid()))

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0] * 2
