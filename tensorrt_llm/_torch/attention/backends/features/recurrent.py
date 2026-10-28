# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent-layer state of the attention metadata of a hybrid model."""

from typing import TYPE_CHECKING, Optional, Type

from ..extensions import MetadataExtension

if TYPE_CHECKING:
    from ....modules.mamba.mamba2_metadata import Mamba2Metadata


class RecurrentMetadata(MetadataExtension):
    """Per-step metadata of the Mamba and linear-attention layers.

    Attached when the KV cache manager also holds recurrent state. The model
    may choose a ``Mamba2Metadata`` subclass through ``mamba_metadata_cls``;
    the instance is built on the first ``prepare()``.
    """

    def __init__(self, metadata) -> None:
        super().__init__(metadata)
        self.chunk_size = 128
        self.mamba_metadata_cls: Optional[Type["Mamba2Metadata"]] = None
        self.mamba_metadata: Optional["Mamba2Metadata"] = None

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
