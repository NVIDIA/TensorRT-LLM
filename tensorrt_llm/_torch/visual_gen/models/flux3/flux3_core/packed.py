# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode-major FLUX.3 tokens and immutable per-sample layout."""

from dataclasses import dataclass

import torch
from torch import Tensor

Layout = tuple[tuple[str, tuple[int, ...]], ...]
Bounds = tuple[tuple[int, int], ...]


def bounds(lengths: tuple[int, ...]) -> Bounds:
    offset = 0
    result = []
    for length in lengths:
        result.append((offset, offset + length))
        offset += length
    return tuple(result)


@dataclass
class Packed:
    data: dict[str, Tensor]
    ids: dict[str, Tensor]
    timesteps: dict[str, Tensor]
    layout: Layout

    def kwargs(self) -> dict:
        result = {"layout": self.layout}
        for name, _ in self.layout:
            result[f"x_{name}"] = self.data[name]
            result[f"x_{name}_ids"] = self.ids[name]
            result[f"x_{name}_timesteps"] = self.timesteps[name]
        return result


def pack(
    samples: list[dict[str, tuple[Tensor, Tensor] | None]], channels: dict[str, int]
) -> Packed:
    if not samples:
        raise ValueError("FLUX.3 requires at least one packed sample")
    device = next(
        (item[0].device for sample in samples for item in sample.values() if item is not None),
        None,
    )
    if device is None:
        raise ValueError("FLUX.3 requires nonempty packed conditioning")
    data, ids, timesteps, layout = {}, {}, {}, []
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
                    or position.shape != (value.shape[0], 4)
                ):
                    raise ValueError(f"Invalid packed FLUX.3 {name} tokens or positions")
                tokens.append(value)
                positions.append(position)
                length = value.shape[0]
            lengths.append(length)
        data[name] = (
            torch.cat(tokens)
            if tokens
            else torch.zeros(0, width, device=device, dtype=torch.bfloat16)
        )
        ids[name] = torch.cat(positions) if positions else torch.zeros(0, 4, device=device)
        timesteps[name] = torch.zeros(sum(lengths), device=device, dtype=torch.bfloat16)
        layout.append((name, tuple(lengths)))
    return Packed(data, ids, timesteps, tuple(layout))
