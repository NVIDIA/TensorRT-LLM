# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal model contracts shared by the executor and model implementations.

This module imports no model implementation, so any model can implement the contract
without depending on the shared mixin.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeGuard

from strenum import StrEnum

__all__ = [
    "MultimodalFeature",
    "SupportsMultimodal",
    "get_multimodal_data_device_paths",
    "is_multimodal_model",
    "supports_feature",
]

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
    from tensorrt_llm.inputs.multimodal import MultimodalParams
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    from .modeling_multimodal_mixin import EncoderGroup, PreparedLlmInputs


class MultimodalFeature(StrEnum):
    ITEM_SCHEDULING = "item_scheduling"
    ENCODER_CACHE = "encoder_cache"  # Independent of item scheduling.
    ENCODER_CUDA_GRAPH = "encoder_cuda_graph"
    ENCODER_DP = "encoder_dp"
    ENCODER_TP = "encoder_tp"
    LLM_BREAKABLE_CUDA_GRAPH = "llm_breakable_cuda_graph"
    LLM_PIECEWISE_CUDA_GRAPH = "llm_piecewise_cuda_graph"  # Includes LLM-scoped torch.compile.
    CHUNKED_PREFILL_MEDIA = "chunked_prefill_media"
    KV_REUSE_MEDIA = "kv_reuse_media"
    DISAGG_E_TO_P = "disagg_e_to_p"
    DISAGG_P_TO_D = "disagg_p_to_d"
    UINT8_MEDIA = "uint8_media"


_MultimodalOptOuts = Mapping[
    MultimodalFeature | Literal["supports_vision_gpu_normalize", "encoder_attention_backend"], str
]


class SupportsMultimodal(Protocol):
    """The model-facing multimodal contract, independent of mixin inheritance.

    Models opt in with ``supports_multimodal = True`` and implement this contract.
    ``MultimodalModelMixin`` supplies the common implementation. Feature declarations
    describe support, while runtime properties/configuration determine what is enabled.
    Item scheduling additionally requires the processor's item-metadata contract.

    Optional features keep their declarations false (or return an empty support mapping)
    until implemented. Embedding properties and ``language_model`` may raise
    ``NotImplementedError`` when the executor can derive or skip their values.
    Encoder groups and native encoder metadata/graphs retain their existing roles.
    Transport serialization belongs outside the model contract.

    ``multimodal_data_device_paths`` is optional for every model, multimodal or not;
    read it through ``get_multimodal_data_device_paths``.
    """

    supports_multimodal: ClassVar[Literal[True]]
    multimodal_opt_outs: ClassVar[_MultimodalOptOuts]
    supports_encoder_cache: ClassVar[bool]
    supports_mm_encoder_item_scheduling: ClassVar[bool]
    supported_encoder_attention_backends: ClassVar[dict[str, frozenset[str]]]
    supports_vision_gpu_normalize: ClassVar[bool]
    support_mm_disagg: ClassVar[bool]

    model_config: ModelConfig[Any]

    @classmethod
    def multimodal_feature_limits(
        cls, model_config: ModelConfig[Any] | None, llm_args: TorchLlmArgs | None
    ) -> Mapping[MultimodalFeature, str]:
        """Return deployment-specific exclusions, each with a reason and tracking ID."""
        ...

    @property
    def mm_encoder_groups(self) -> Sequence[EncoderGroup]:
        """Read the model's existing encoder batching declarations."""
        ...

    @property
    def language_model(self) -> torch.nn.Module:
        """The inner language model, read for its capabilities and weight-transfer scope."""
        ...

    @property
    def samples_from_vocab_shards(self) -> bool:
        """Whether the inner model's speculative worker samples target rows from vocabulary
        shards."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Width of an encoder embedding, including model-specific extra features."""
        ...

    @property
    def embedding_dtype(self) -> torch.dtype:
        """Dtype of encoder embeddings."""
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
        """Return encoder embeddings concatenated in request order."""
        ...

    def prepare_multimodal_encoder_inputs(
        self, selected_items: Sequence[tuple[MultimodalParams, int]]
    ) -> list[tuple[MultimodalParams, Sequence[int], str]]:
        """Prepare selected items before they are copied to the GPU, in scheduler order."""
        ...

    def forward_multimodal_encoder_items(
        self, encoder_inputs: Sequence[tuple[MultimodalParams, Sequence[int], str]]
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

    The marker is a declaration; it does not check method signatures. Implementations are
    checked statically against ``SupportsMultimodal``; feature gates still check the
    relevant declaration and configuration. Avoid structural introspection on each executor step.
    The marker is read from ``model.__class__``, as ``isinstance`` does.
    """
    return getattr(model.__class__, "supports_multimodal", False) is True


_FEATURE_FLAGS = {
    MultimodalFeature.ITEM_SCHEDULING: "supports_mm_encoder_item_scheduling",
    MultimodalFeature.ENCODER_CACHE: "supports_encoder_cache",
    MultimodalFeature.DISAGG_E_TO_P: "support_mm_disagg",
    MultimodalFeature.ENCODER_DP: "supports_mm_encoder_data_parallel",
}


def supports_feature(
    model_cls: type | torch.nn.Module,
    feature: MultimodalFeature,
    model_config: ModelConfig[Any] | None = None,
    llm_args: TorchLlmArgs | None = None,
) -> bool:
    """Read class declarations or inspect a built model under its deployment limits.

    Encoder TP requires local tower weights sharded by their encoder mapping.
    The breakable graph declaration is on the wrapper. At startup the engine also
    validates the decoder instance and its layer registries before allowing replay.
    """
    model = None
    if not isinstance(model_cls, type):
        model = model_cls
        model_cls = type(model)
        if model_config is None:
            model_config = getattr(model, "model_config", None)
    if getattr(model_cls, "supports_multimodal", False) is not True:
        return False
    if feature in getattr(model_cls, "multimodal_opt_outs", {}):
        return False
    limits = getattr(model_cls, "multimodal_feature_limits", None)
    if limits is not None and feature in limits(model_config, llm_args):
        return False

    if feature in _FEATURE_FLAGS:
        flag = getattr(model_cls, _FEATURE_FLAGS[feature], False)
        return flag is True if feature == MultimodalFeature.ENCODER_DP else bool(flag)
    if feature == MultimodalFeature.ENCODER_TP:
        from .modeling_multimodal_encoder import _has_sharded_encoder_weights

        return model is not None and _has_sharded_encoder_weights(model)
    if feature == MultimodalFeature.LLM_BREAKABLE_CUDA_GRAPH:
        from .modeling_multimodal_mixin import MultimodalModelMixin

        return (
            issubclass(model_cls, MultimodalModelMixin)
            and getattr(model_cls, "supports_multimodal_breakable_cuda_graph", False) is True
            and (model_config is None or model_config.mapping.pp_size == 1)
        )
    if feature == MultimodalFeature.LLM_PIECEWISE_CUDA_GRAPH:
        from .modeling_utils import DecoderModelForCausalLM

        return issubclass(model_cls, DecoderModelForCausalLM) or callable(
            getattr(model_cls, "apply_llm_torch_compile", None)
        )
    return True


def get_multimodal_data_device_paths(model: object) -> list[str] | None:
    """Return the ``multimodal_data`` paths the executor moves to the device for ``model``.

    Any model may declare ``multimodal_data_device_paths``; ``None`` (also when it is
    absent) moves every tensor.
    """
    return getattr(model, "multimodal_data_device_paths", None)
