# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Kimi K3.1 moonvit entry tests share: the bf16 band, a cell's generator and the cell IDs.

Not a test file: the entry tests import it by name from their own directory.
"""

import zlib

import torch

#: bf16 results against an fp32 / fp64 reference rounded once: the entries round intermediate bf16
#: values, each half an ulp; four ulps of the row's scale covers them.
BF16_BAND = dict(rtol=4 * 2.0**-8, atol=4 * 2.0**-8)


def cell_generator(cell) -> torch.Generator:
    """A CUDA generator seeded from the cell's description."""
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def cell_ids(cells) -> list[str]:
    """The test IDs of ``cells``."""
    return [c.why[:48] for c in cells]
