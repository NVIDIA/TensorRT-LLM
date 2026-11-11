from typing import Protocol, Tuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import AudioLatentShape, VideoLatentShape


class Patchifier(Protocol):
    """
    Protocol for patchifiers that convert latent tensors into patches and assemble them back.
    """

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        ...
        """
        Convert latent tensors into flattened patch tokens.
        Args:
            latents: Latent tensor to patchify.
        Returns:
            Flattened patch tokens tensor.
        """

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: AudioLatentShape | VideoLatentShape,
    ) -> torch.Tensor:
        """
        Converts latent tensors between spatio-temporal formats and flattened sequence representations.
        Args:
            latents: Patch tokens that must be rearranged back into the latent grid constructed by `patchify`.
            output_shape: Shape of the output tensor. Note that output_shape is either AudioLatentShape or
            VideoLatentShape.
        Returns:
            Dense latent tensor restored from the flattened representation.
        """

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        ...
        """
        Returns the patch size as a tuple of (temporal, height, width) dimensions
        """

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ...
        """
        Compute metadata describing where each latent patch resides within the
        grid specified by `output_shape`.
        Args:
            output_shape: Target grid layout for the patches.
            device: Target device for the returned tensor.
        Returns:
            Tensor containing patch coordinate metadata such as spatial or temporal intervals.
        """


class SchedulerProtocol(Protocol):
    """
    Protocol for schedulers that provide a sigmas schedule tensor for a
    given number of steps. Device is cpu.
    """

    def execute(self, steps: int, **kwargs) -> torch.FloatTensor: ...


class GuiderProtocol(Protocol):
    """
    Protocol for guiders that compute a delta tensor given conditioning inputs.
    The returned delta should be added to the conditional output (cond), enabling
    multiple guiders to be chained together by accumulating their deltas.
    """

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor: ...

    def enabled(self) -> bool:
        """
        Returns whether the corresponding perturbation is enabled. E.g. for CFG, this should return False if the scale
        is 1.0.
        """
        ...


class DiffusionStepProtocol(Protocol):
    """
    Protocol for diffusion steps that provide a next sample tensor for a given current sample tensor,
    current denoised sample tensor, and sigmas tensor.
    """

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int, **kwargs
    ) -> torch.Tensor: ...
