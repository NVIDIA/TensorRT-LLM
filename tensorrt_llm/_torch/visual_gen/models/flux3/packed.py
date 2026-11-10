# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode-major FLUX.3 tokens and immutable per-sample layout."""

from dataclasses import dataclass

import torch
from torch import Tensor

Layout = tuple[tuple[str, tuple[int, ...]], ...]
Bounds = tuple[tuple[int, int], ...]
# Position ids are (t, h, w, l).
NUM_POSITION_AXES = 4


def bounds(lengths: tuple[int, ...]) -> Bounds:
    """(start, end) of consecutive segments with the given lengths."""
    offset = 0
    result = []
    for length in lengths:
        result.append((offset, offset + length))
        offset += length
    return tuple(result)


@dataclass
class Packed:
    """Per modality: tokens [N, C] and position ids [N, 4]."""

    data: dict[str, Tensor]
    ids: dict[str, Tensor]
    layout: Layout

    def kwargs(self) -> dict:
        """Transformer inputs: ``layout`` and ``x_{modality}{,_ids}``."""
        result = {"layout": self.layout}
        for name, _ in self.layout:
            result[f"x_{name}"] = self.data[name]
            result[f"x_{name}_ids"] = self.ids[name]
        return result


def pack(
    samples: list[dict[str, tuple[Tensor, Tensor] | None]], channels: dict[str, int]
) -> Packed:
    """Concatenate each modality's (tokens [n, C], ids [n, 4]) across samples, modalities sorted.

    ``channels`` gives every modality's width C.
    """
    if not samples:
        raise ValueError("FLUX.3 requires at least one packed sample")
    first = next((item for sample in samples for item in sample.values() if item is not None), None)
    if first is None:
        raise ValueError("FLUX.3 requires nonempty packed conditioning")
    # Empty modalities take the device and dtypes of the first present sample.
    first_tokens, first_ids = first
    data, ids, layout = {}, {}, []
    for sample in samples:
        if sample.keys() - channels.keys():
            raise ValueError("Unknown FLUX.3 packed modality")
    for name, width in sorted(channels.items()):
        tokens, positions, lengths = [], [], []
        for sample in samples:
            item = sample.get(name)
            length = 0
            if item is not None:
                value, position = item
                if (
                    value.ndim != 2
                    or value.shape[1] != width
                    or position.shape != (value.shape[0], NUM_POSITION_AXES)
                ):
                    raise ValueError(f"Invalid packed FLUX.3 {name} tokens or positions")
                tokens.append(value)
                positions.append(position)
                length = value.shape[0]
            lengths.append(length)
        data[name] = torch.cat(tokens) if tokens else first_tokens.new_zeros(0, width)
        ids[name] = torch.cat(positions) if positions else first_ids.new_zeros(0, NUM_POSITION_AXES)
        layout.append((name, tuple(lengths)))
    return Packed(data, ids, tuple(layout))
