from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class SplitSpec:
    """Describes how tensors are split across ranks for parallel VAE."""

    split_dim: Literal["height", "width"]
    input_dim: int
    conv3d_dim: int
    conv2d_dim: int
    attn_dim: int


class ParallelVAEBase(nn.Module):
    """nn.Module wrapper that parallelises a VAE across a process group.

    Subclasses implement ``_encode_impl`` / ``_decode_impl`` for their
    specific VAE family and override ``_parallelize_modules`` to swap
    internal layers (convolutions, attention, norm) with parallel variants.
    """

    def __init__(
        self,
        vae_backend: nn.Module,
        pg: dist.ProcessGroup,
        spec: SplitSpec,
    ) -> None:
        super().__init__()
        self.vae_backend = vae_backend
        self.pg = pg
        self.spec = spec
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        self._adj_groups = self._build_adj_groups(pg)
        self._parallelize_modules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        if kwargs.get("return_dict", True):
            raise NotImplementedError(
                "ParallelVAEBase does not support return_dict=True. Pass return_dict=False."
            )

        result = self._encode_impl(x, **kwargs)

        if not isinstance(result, torch.Tensor):
            raise TypeError(f"_encode_impl must return a torch.Tensor, got {type(result).__name__}")
        if result.ndim != x.ndim:
            raise ValueError(
                f"_encode_impl changed tensor rank: input {x.ndim}D, output {result.ndim}D"
            )

        return (result,)

    def decode(self, z: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        if kwargs.get("return_dict", True):
            raise NotImplementedError(
                "ParallelVAEBase does not support return_dict=True. Pass return_dict=False."
            )

        result = self._decode_impl(z, **kwargs)

        if not isinstance(result, torch.Tensor):
            raise TypeError(f"_decode_impl must return a torch.Tensor, got {type(result).__name__}")
        if result.ndim != z.ndim:
            raise ValueError(
                f"_decode_impl changed tensor rank: input {z.ndim}D, output {result.ndim}D"
            )

        return (result,)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _encode_impl(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _decode_impl(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _parallelize_modules(self) -> None:
        """Replace internal layers with parallel variants.  Called at end of ``__init__``."""

    @staticmethod
    def make_spec(split_dim: Literal["height", "width"]) -> "SplitSpec":
        """Build a ``SplitSpec`` for the given split dimension.

        Every concrete subclass must override this.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vae_backend, name)

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _split_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Chunk ``x`` along the input split dimension and return this rank's slice.

        Returns:
            (local_chunk, full_size) where *full_size* is the original extent
            along the split dim so callers can pass it to ``_gather_tensor``.
        """
        dim = self.spec.input_dim
        full_size = x.shape[dim]
        if full_size % self.world_size != 0:
            raise ValueError(
                f"Dim {dim} (size {full_size}) not divisible by world_size {self.world_size}"
            )
        return x.chunk(self.world_size, dim=dim)[self.rank], full_size

    def _gather_tensor(self, x_local: torch.Tensor) -> torch.Tensor:
        """All-gather ``x_local`` along the input split dimension."""
        dim = self.spec.input_dim
        gathered = [torch.empty_like(x_local) for _ in range(self.world_size)]
        dist.all_gather(gathered, x_local, group=self.pg)
        return torch.cat(gathered, dim=dim)

    # ------------------------------------------------------------------
    # Module-replacement helper
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_module(root: nn.Module, target_name: str, new_module: nn.Module):
        """Replace a named submodule inside *root* in-place."""
        attrs = target_name.split(".")
        parent = root
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], new_module)

    # ------------------------------------------------------------------
    # Process-group helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_adj_groups(pg: dist.ProcessGroup) -> List[dist.ProcessGroup]:
        """Create pairwise adjacent-rank groups from *pg*.

        Returns a list where ``adj_groups[i]`` is a group containing global
        ranks ``ranks[i]`` and ``ranks[i+1]`` from *pg*.
        """
        world_size = dist.get_world_size(pg)
        ranks = list(range(world_size))

        adj_groups: List[dist.ProcessGroup] = []
        for i in range(world_size - 1):
            adj_groups.append(
                dist.new_group(
                    [ranks[i], ranks[i + 1]],
                    use_local_synchronization=False,
                )
            )
        return adj_groups


class ParallelVAEFactory:
    """Factory that maps VAE classes to their parallel wrappers via lazy imports.

    The mapping is keyed by the fully-qualified VAE class name
    (``module.ClassName``) so the parallel implementation module is only
    imported when actually needed -- no side-effect imports required.
    """

    # "vae_module.VaeClass" -> (parallel_module, parallel_class)
    _LAZY_REGISTRY: Dict[str, Tuple[str, str]] = {
        "diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan": (
            "tensorrt_llm._torch.visual_gen.models.wan.parallel_vae",
            "ParallelVAE_Wan",
        ),
    }

    @classmethod
    def from_vae(
        cls,
        vae: nn.Module,
        split_dim: Literal["height", "width"],
        pg: dist.ProcessGroup,
    ) -> ParallelVAEBase:
        parallel_cls = cls._resolve(type(vae))
        if parallel_cls is None:
            raise ValueError(
                f"No parallel VAE registered for {type(vae).__name__}. "
                f"Known VAE types: {list(cls._LAZY_REGISTRY.keys())}"
            )
        spec = parallel_cls.make_spec(split_dim)
        return parallel_cls(vae, pg, spec)

    @classmethod
    def _resolve(cls, vae_type: type) -> Type[ParallelVAEBase] | None:
        """Walk the MRO of *vae_type* and return the first matching parallel class."""
        import importlib

        for klass in vae_type.__mro__:
            key = f"{klass.__module__}.{klass.__qualname__}"
            if key in cls._LAZY_REGISTRY:
                mod_path, cls_name = cls._LAZY_REGISTRY[key]
                mod = importlib.import_module(mod_path)
                return getattr(mod, cls_name)
        return None
