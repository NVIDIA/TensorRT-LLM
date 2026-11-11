# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent-layer state of the attention metadata of a hybrid model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type

from ..extensions import MetadataExtension

if TYPE_CHECKING:
    from ....modules.mamba.mamba2_metadata import Mamba2Metadata


@dataclass(frozen=True)
class RecurrentConfig:
    """Construction-time configuration of ``RecurrentMetadata``.

    ``chunk_size`` is the model's SSM chunk size; ``mamba_metadata_cls`` is the
    ``Mamba2Metadata`` subclass the model needs, or None for the base class.
    """

    chunk_size: int = 128
    mamba_metadata_cls: Optional[Type["Mamba2Metadata"]] = None


class RecurrentMetadata(MetadataExtension):
    """Per-step metadata of the Mamba and linear-attention layers.

    Attached when the KV cache manager also holds recurrent state. Its chunk
    size and ``Mamba2Metadata`` class come from ``recurrent_config``, given at
    construction; the instance is built on the first ``prepare()``.
    """

    # Every CUDA graph copy shares the recurrent-state metadata the first
    # prepare built.
    shared = frozenset({"mamba_metadata"})
    fields = frozenset(
        {
            "recurrent_config",
            "chunk_size",
            "mamba_metadata",
            "mamba_metadata_cls",
        }
    )
    methods = frozenset()
    init_fields = {
        "recurrent_config": RecurrentConfig(),
    }

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.mamba_metadata: Optional["Mamba2Metadata"] = self.shared_or(
            "mamba_metadata", lambda: None
        )

    @property
    def chunk_size(self) -> int:
        return self.recurrent_config.chunk_size

    @property
    def mamba_metadata_cls(self) -> Optional[Type["Mamba2Metadata"]]:
        return self.recurrent_config.mamba_metadata_cls

    def prepare(self) -> None:
        md = self.metadata
        if self.mamba_metadata is None:
            metadata_cls = self.mamba_metadata_cls
            if metadata_cls is None:
                from ....modules.mamba.mamba2_metadata import Mamba2Metadata

                metadata_cls = Mamba2Metadata
            self.mamba_metadata = metadata_cls(
                md.max_num_requests, self.chunk_size, max_num_tokens=md.max_num_tokens
            )
        self.mamba_metadata.prepare(md)
