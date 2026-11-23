# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The aux stream two independent pieces of a captured generation step run on side by side."""

from __future__ import annotations

from typing import Callable, TypeVar

import torch

_Main = TypeVar("_Main")
_Aux = TypeVar("_Aux")


class AuxStream:
    """One aux stream and the fork and join events that order it against the current stream.

    Made once per model after the weights load: the engine constructs the model on the meta
    device, and a stream made inside a CUDA graph capture is not capturable.
    """

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
        self.fork = torch.cuda.Event()
        self.join = torch.cuda.Event()

    def fork_join(self, main: Callable[[], _Main], aux: Callable[[], _Aux]) -> tuple[_Main, _Aux]:
        """``(main(), aux())``: inside a CUDA graph capture ``aux`` runs on the aux stream beside
        ``main``; otherwise both run in order on the current stream."""
        if not torch.cuda.is_current_stream_capturing():
            return main(), aux()
        # In a capture the aux stream's outputs need no record_stream: every use of the aux
        # stream starts by waiting for the current stream (the fork).
        self.fork.record()
        main_out = main()
        with torch.cuda.stream(self.stream):
            self.fork.wait()
            aux_out = aux()
            self.join.record()
        self.join.wait()
        return main_out, aux_out
