"""The interface pipelines use to build a model from checkpoint weights."""

from __future__ import annotations

from typing import Protocol, TypeVar

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.registry import RegistryKeys

_Model_co = TypeVar("_Model_co", covariant=True)


class ModelBuilder(Protocol[_Model_co]):
    """Build a model and describe what that build exchanges with its registry.
    ``build`` takes builder-specific keyword arguments; a builder ignores the ones it does not use.
    ``registry_keys`` describes a ``build`` with the same requested dtype.
    """

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: object
    ) -> _Model_co:
        """Build the model on ``device`` at the requested ``dtype``."""
        ...

    def registry_keys(self, dtype: torch.dtype | None = None) -> RegistryKeys:
        """Return the registry keys a build at ``dtype`` queries and retains."""
        ...

    def model_config(self) -> dict[str, object]:
        """Return the checkpoint's model configuration."""
        ...

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Whether this builder guarantees stable GPU weight storage across builds."""
        ...

    @property
    def builds_per_shape(self) -> bool:
        """Whether a built model serves only the latent shape it was built for."""
        ...
