# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input-embedding lookup (thin mirror of torch.nn.functional.embedding)."""

import torch
import torch.nn.functional as F


def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Look up rows of `weight` by token id: [*] ids -> [*, H] embeddings."""
    return F.embedding(input_ids, weight)
