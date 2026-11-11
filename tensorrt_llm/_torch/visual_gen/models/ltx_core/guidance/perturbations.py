from dataclasses import dataclass
from enum import IntEnum

import torch
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.visual_gen.models.ltx_core.types import HostDeviceTensorPair


class PerturbationType(IntEnum):
    """Types of attention perturbations for STG (Spatio-Temporal Guidance).
    The integer value is the row index into ``BatchedPerturbationConfig._block_masks`` dim 0.
    """

    SKIP_VIDEO_SELF_ATTN = 0
    SKIP_AUDIO_SELF_ATTN = 1
    SKIP_A2V_CROSS_ATTN = 2
    SKIP_V2A_CROSS_ATTN = 3


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation specifying which attention type to skip and in which blocks."""

    type: PerturbationType
    blocks: list[int] | None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False

        if self.blocks is None:
            return True

        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration holding a list of perturbations for a single sample."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False

        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty() -> "PerturbationConfig":
        return PerturbationConfig([])


class BatchedPerturbationConfig:
    """Per-block attention keep-masks for a batch, built once from a list of per-sample configs.
    Construction materializes ``_block_masks`` -- paired device and host
    ``(len(PerturbationType), num_blocks, B)`` tensors (1 = keep, 0 = perturbed) whose dim-0 row
    index is the ``PerturbationType`` value -- from the perturbation structure. The per-sample
    config list is NOT retained: every consumer reads the tensor pair.
    The host build (reading the Python structure) happens here, in ``__init__``, so it MUST be run
    eagerly OUTSIDE any ``torch.compile`` / CUDA-graph-capture region. The compiled block then reads
    perturbation purely as the runtime device tensor and never recompiles per config.
    """

    _block_masks: HostDeviceTensorPair

    def __init__(
        self,
        perturbations: list[PerturbationConfig],
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        keep = [
            [
                [not pc.is_perturbed(PerturbationType(direction), block) for pc in perturbations]
                for block in range(num_blocks)
            ]
            for direction in range(len(PerturbationType))
        ]
        host = torch.tensor(keep, dtype=dtype, device="cpu")
        if device is not None and torch.device(device).type == "cuda":
            # CUDA non-blocking copies require pinned source memory. The host mirror is read-only
            # after construction.
            host = host.pin_memory()
            device_mask = host.to(device, non_blocking=True)
        else:
            device_mask = host if device is None else host.to(device)
        self._block_masks = HostDeviceTensorPair(host_tensor=host, device_tensor=device_mask)

    @classmethod
    def from_masks(cls, block_masks: HostDeviceTensorPair) -> "BatchedPerturbationConfig":
        """Construct from prebuilt mask tensors (e.g. a batch-dim slice), bypassing the host build.
        The host tensor is consumed by ``any_in_batch`` / ``all_in_batch`` while the compiled
        processor reads only the device tensor.
        """
        obj = cls.__new__(cls)
        obj._block_masks = block_masks
        return obj

    @property
    def block_masks(self) -> torch.Tensor:
        """Keep-mask on the compute device (1 = keep, 0 = perturbed), indexed [type, block, sample]."""
        return self._block_masks.device_tensor

    @property
    def block_masks_cpu(self) -> torch.Tensor:
        """Host mirror of the keep-mask."""
        return self._block_masks.host_tensor

    def batch_slice(self, start: int, end: int) -> "BatchedPerturbationConfig":
        """A view over samples ``[start:end]`` of the batch, by slicing the mask tensors.
        Slicing (never rebuilding) keeps the host mask build outside any compiled / capture region.
        """
        return BatchedPerturbationConfig.from_masks(
            HostDeviceTensorPair(
                host_tensor=self._block_masks.host_tensor[:, :, start:end],
                device_tensor=self._block_masks.device_tensor[:, :, start:end],
            )
        )

    def mask(self, perturbation_type: PerturbationType, block: int) -> torch.Tensor:
        """This block's ``(B, 1, 1)`` keep-mask for one perturbation type, as an OWNED tensor.
        A ``clone`` (not a view into ``_block_masks``) so the masks attached to a block
        (e.g. self- and cross-attention) don't alias the same storage -- aliased graph inputs are
        fragile under ``torch.compile``.
        """
        return self._block_masks.device_tensor[perturbation_type, block].reshape(-1, 1, 1).clone()

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return bool((self._block_masks.host_tensor[perturbation_type, block] == 0).any())

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return bool((self._block_masks.host_tensor[perturbation_type, block] == 0).all())

    @staticmethod
    def empty(
        batch_size: int,
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BatchedPerturbationConfig":
        return BatchedPerturbationConfig(
            [PerturbationConfig.empty() for _ in range(batch_size)], num_blocks, device, dtype
        )
