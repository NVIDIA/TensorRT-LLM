# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal model contracts shared by the executor and model implementations."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeGuard

__all__ = ["EncoderParallelMode", "SupportsMultimodal", "is_multimodal_model"]

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
    from tensorrt_llm.inputs.multimodal import MultimodalParams

    from .modeling_multimodal_mixin import EncoderGroup, MultimodalModelMixin, PreparedLlmInputs

EncoderParallelMode = Literal["tp", "dp"]


class SupportsMultimodal(Protocol):
    """The model-facing multimodal contract, independent of mixin inheritance.

    Models opt in with ``supports_multimodal = True`` and implement this contract.
    ``MultimodalModelMixin`` supplies the common implementation. Feature declarations
    describe support, while runtime properties/configuration determine what is enabled.
    Item scheduling additionally requires the processor's item-metadata contract.

    Optional features keep their declarations false (or return an empty support mapping)
    until implemented. Embedding properties may raise ``NotImplementedError`` when the
    executor can derive their values from the text embedding layer or model config.
    Encoder groups and native encoder metadata/graphs retain their existing roles.
    Transport serialization belongs outside the model contract.
    """

    supports_multimodal: ClassVar[Literal[True]]
    supports_encoder_cache: ClassVar[bool]
    supports_mm_encoder_item_scheduling: ClassVar[bool]
    supports_encoder_attention_backend: ClassVar[bool]
    support_mm_disagg: ClassVar[bool]

    model_config: ModelConfig[Any]

    @property
    def mm_encoder_groups(self) -> Sequence[EncoderGroup]:
        """Read the model's existing encoder batching declarations."""
        ...

    @classmethod
    def get_multimodal_encoder_parallel_modes(
        cls, model_config: ModelConfig[Any]
    ) -> dict[str, frozenset[EncoderParallelMode]]:
        """Return supported TP/DP modes per enabled logical encoder, before loading weights."""
        ...

    @classmethod
    def get_supported_encoder_attention_backends(
        cls, model_config: ModelConfig[Any]
    ) -> dict[str, frozenset[str]]:
        """Return supported attention backends per enabled logical encoder."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Width of one encoder output row, including model-specific extra features."""
        ...

    @property
    def embedding_dtype(self) -> torch.dtype:
        """Dtype of encoder output rows."""
        ...

    @property
    def text_embedding_layer(self) -> torch.nn.Module:
        """Text embedding layer used for fusion and output-size inference."""
        ...

    @property
    def encoder_cache_active(self) -> bool:
        """Whether persistent encoder-output reuse is enabled in this configuration."""
        ...

    @property
    def multimodal_encoder_cache(self) -> TensorLRUCache[Hashable] | None:
        """The existing model-owned output cache, also used by item scheduling."""
        ...

    def initialize_multimodal_encoder_cache(
        self, max_bytes: int
    ) -> TensorLRUCache[Hashable] | None:
        """Initialize that cache once at startup using the executor's resolved byte budget."""
        ...

    @classmethod
    def build_encoder_cache_item_keys(
        cls,
        multimodal_hashes: Sequence[Sequence[int]] | None,
        item_refs: Sequence[tuple[str, int]],
        embedding_lengths: Sequence[int],
        kwargs_hash: str | None,
    ) -> list[Hashable] | None:
        """Return prompt-ordered item keys, or None when persistent reuse is unavailable."""
        ...

    def validate_multimodal_request_data(self, mm_data: dict[str, Any]) -> None:
        """Reject invalid model inputs before admitting the request to a shared batch."""
        ...

    def encode_multimodal_inputs(
        self, multimodal_params: Sequence[MultimodalParams]
    ) -> torch.Tensor:
        """Return encoder embeddings concatenated in request order.

        TODO: Adopt the encoded Params-view return contract when batching #20897 lands.
        """
        ...

    def prepare_multimodal_encoder_inputs(
        self, selected_items: Sequence[tuple[MultimodalParams, int]]
    ) -> list[tuple[MultimodalParams, list[int], str]]:
        """Prepare selected items before H2D, retaining scheduler order."""
        ...

    def forward_multimodal_encoder_items(
        self, encoder_inputs: Sequence[tuple[MultimodalParams, list[int], str]]
    ) -> list[torch.Tensor]:
        """Return one embedding tensor per selected item, in scheduler order."""
        ...

    def prepare_multimodal_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        multimodal_params: Sequence[MultimodalParams] | None,
        num_context_requests: int,
        **forward_kwargs: Any,
    ) -> PreparedLlmInputs:
        """Fuse model-specific multimodal features into the current LLM inputs."""
        ...


def is_multimodal_model(model: object) -> TypeGuard[SupportsMultimodal]:
    """Read the explicit class marker without evaluating encoder properties.

    This is a declaration, not runtime signature validation. Implementations are checked
    statically against ``SupportsMultimodal``; feature gates still check the relevant
    declaration and configuration. Avoid structural reflection on each executor step.
    """
    return getattr(type(model), "supports_multimodal", False) is True


if TYPE_CHECKING:
    # Check the shared implementation without imposing Protocol inheritance at runtime.
    _model_contract: SupportsMultimodal = MultimodalModelMixin()
