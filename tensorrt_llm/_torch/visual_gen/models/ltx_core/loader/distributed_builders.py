"""Thin distributed model wrappers around the new loading builders."""

from __future__ import annotations

import itertools
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, TypeVar

import torch
import torch.distributed as dist
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.distributed import build_distributed, distributed_registry_keys
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.gemma import gemma3_builder, gemma4_builder
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.helpers import assign_state_dict
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.registry import DummyRegistry, ModelRegistry, RegistryKeys
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.simple_builder import SimpleBuilder
from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.broadcast import BroadcastCoordinator
from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.gemma.accelerate_wrapper import AccelerateGemmaWrapper
from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.gemma.batch_parallel_wrapper import BatchParallelGemmaWrapper
from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.transformer.attention import AttentionManager
from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.transformer.sequence_parallel import (
    SequenceParallelModelWrapper,
    create_video_self_attention_module_ops,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.transformer.tiled_data_parallel import TiledDataParallelModelWrapper
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma import gemma_model_type
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import TileCountConfig
from tensorrt_llm._torch.visual_gen.models.ltx_core.tools import VideoLatentTools

_Model = TypeVar("_Model", bound=torch.nn.Module)

# Sequence numbers naming each attention manager's module operation. Ranks run the same program, so the
# n-th manager wrapped here gets the same number on every rank: the name stays distinct per manager for
# the local shell cache and equal across ranks for the build agreement (an object id is not).
_MANAGER_SEQUENCE = itertools.count()
_MANAGER_NUMBERS: weakref.WeakKeyDictionary[AttentionManager, int] = weakref.WeakKeyDictionary()


def _manager_number(manager: AttentionManager) -> int:
    number = _MANAGER_NUMBERS.get(manager)
    if number is None:
        number = _MANAGER_NUMBERS[manager] = next(_MANAGER_SEQUENCE)
    return number


def _agree_gemma(group: dist.ProcessGroup, phase: str, error: str | None) -> None:
    if not dist.is_initialized():
        if error:
            raise RuntimeError(error)
        return
    errors: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError(f"Gemma {phase} failed on one or more ranks: {errors}")


# Bounds a non-source rank's host wait on the source rank (its cold Gemma load, an encode or an
# enhancement), as BroadcastGemmaWrapper does before each broadcast.
_GEMMA_SRC_WAIT_TIMEOUT_S = 600.0
# Numbers Accelerate Gemma builders in construction order, which every rank shares, to key their stores.
_GEMMA_BUILDER_IDS = itertools.count()


def _gemma_coordinator(group: dist.ProcessGroup, src_rank: int) -> BroadcastCoordinator | None:
    """One store signal per builder, driven by the source rank; None without a process group."""
    if not dist.is_initialized():
        return None
    store = dist.distributed_c10d.PrefixStore(
        f"ltx_gemma_broadcast/{next(_GEMMA_BUILDER_IDS)}/", dist.distributed_c10d._get_default_store()
    )
    return BroadcastCoordinator(store=store, is_driver=dist.get_rank(group) == src_rank)


def _hold_until_source(coordinator: BroadcastCoordinator | None) -> None:
    """Keep non-source ranks on the host until the source rank signals, before a collective with it."""
    if coordinator is None:
        return
    if coordinator.is_driver:
        coordinator.notify()
    else:
        coordinator.wait_for_signal_change(poll_interval=0.001, timeout=_GEMMA_SRC_WAIT_TIMEOUT_S)


@dataclass(frozen=True, slots=True)
class SequenceParallelBuilder(Generic[_Model]):
    """Build a sequence-parallel wrapper around a registry-retained transformer."""

    base: SimpleBuilder[_Model]
    attention_manager: AttentionManager
    dst: dict[str, torch.Tensor] = field(default_factory=dict)
    """Active weights the builds fill once and then rewrite in place; pass one dict to builders that share them."""

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None, **_kwargs: object
    ) -> SequenceParallelModelWrapper:
        """Build the retained model with sequence-parallel attention and wrap it."""
        model = build_distributed(self.base, self.attention_manager.group, device, dtype, dst=self.dst)
        return SequenceParallelModelWrapper(model, self.attention_manager)

    def registry_keys(self, dtype: torch.dtype | None = None) -> RegistryKeys:
        """Return the registry keys this rank's build at ``dtype`` queries and retains."""
        return distributed_registry_keys(self.base, self.attention_manager.group, dtype)

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the base builder."""
        return self.base.model_config()

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """The registry reuses the bound device allocation across builds."""
        return True

    @property
    def builds_per_shape(self) -> bool:
        """A built model serves any latent shape."""
        return False

    def __post_init__(self) -> None:
        # The attention operation joins the base builder here, not in ``build``: a stage composes
        # compilation onto ``base`` afterwards, and compiled blocks no longer match the operation.
        attention = create_video_self_attention_module_ops(self.attention_manager)
        operation = ModuleOps(
            f"{attention.name}:manager={_manager_number(self.attention_manager)}", attention.matcher, attention.mutator
        )
        if all(existing.name != operation.name for existing in self.base.module_ops):
            object.__setattr__(self, "base", self.base.with_module_ops((*self.base.module_ops, operation)))


@dataclass(frozen=True, slots=True)
class TiledDataParallelBuilder(Generic[_Model]):
    """Build a registry-retained model with tiled data-parallel inference behavior."""

    base: SimpleBuilder[_Model]
    group: dist.ProcessGroup
    tiling: TileCountConfig
    normalize_positions: bool = True
    dst: dict[str, torch.Tensor] = field(default_factory=dict)
    """Active weights the builds fill once and then rewrite in place; pass one dict to builders that share them."""

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        video_tools: VideoLatentTools,
        **_kwargs: object,
    ) -> TiledDataParallelModelWrapper:
        """Build the model and bind tiling, process group, and position settings."""
        model = build_distributed(self.base, self.group, device, dtype, dst=self.dst)
        return TiledDataParallelModelWrapper(
            model,
            video_tools=video_tools,
            tiling=self.tiling,
            group=self.group,
            normalize_positions=self.normalize_positions,
        )

    def registry_keys(self, dtype: torch.dtype | None = None) -> RegistryKeys:
        """Return the registry keys this rank's build at ``dtype`` queries and retains."""
        return distributed_registry_keys(self.base, self.group, dtype)

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the base builder."""
        return self.base.model_config()

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """The registry reuses the bound device allocation across builds."""
        return True

    @property
    def builds_per_shape(self) -> bool:
        """Each build wraps the model for one call's video tools, so it cannot serve another shape."""
        return True


class _AgreedEnhancement:
    """Broadcast either a source-only generation result or its error."""

    def _enhance(self, method: str, *args: object, **kwargs: object) -> str:
        outcome: tuple[str | None, str | None] | None = None
        if self._rank == self._src_rank:
            try:
                outcome = getattr(self._encoder, method)(*args, **kwargs), None
            except Exception as exc:
                outcome = None, f"{type(exc).__name__}: {exc}"
        received = [outcome]
        self._wait_for_src()
        dist.broadcast_object_list(received, src=self._src_rank, group=self._group)
        assert received[0] is not None
        result, error = received[0]
        if error is not None:
            raise RuntimeError(f"Gemma {method} failed on source rank: {error}")
        assert result is not None
        return result

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        """Generate on the source rank and agree on errors before broadcasting."""
        return self._enhance(
            "enhance_t2v",
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            seed=seed,
            static_cache=static_cache,
        )

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        """Generate on the source rank and agree on errors before broadcasting."""
        return self._enhance(
            "enhance_i2v",
            prompt,
            image,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            seed=seed,
            static_cache=static_cache,
        )


def _validate_gemma_group(group: dist.ProcessGroup) -> None:
    """The existing Gemma wrappers pass group-local sources to global broadcasts."""
    if dist.is_initialized() and group is not None:
        world = dist.get_world_size(group)
        if any(dist.get_global_rank(group, rank) != rank for rank in range(world)):
            raise ValueError("Gemma wrappers require group-local ranks to match global ranks")


class _BatchParallelGemmaWrapper(_AgreedEnhancement, BatchParallelGemmaWrapper):
    """Ported batch wrapper with collective agreement before result exchange."""

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Partition the batch, then agree that local encodes succeeded."""
        if not prompts:
            return []
        base, remainder = divmod(len(prompts), self._world_size)
        counts = [base + (rank < remainder) for rank in range(self._world_size)]
        start = sum(counts[: self._rank])
        local_prompts = prompts[start : start + counts[self._rank]]
        local_outputs = []
        error = None
        try:
            if local_prompts:
                local_outputs = self._encoder.encode(local_prompts)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _agree_gemma(self._group, "batch encode", error)
        return [
            self._broadcast_encoder_output(*(local_outputs[slot] if rank == self._rank else (None, None)), rank)
            for rank, count in enumerate(counts)
            for slot in range(count)
        ]


@dataclass(frozen=True, slots=True)
class BatchParallelGemmaBuilder:
    """Build a full Gemma encoder on every rank for batch-parallel serving."""

    base: SimpleBuilder[LTXGemmaTextEncoder]
    group: dist.ProcessGroup
    src_rank: int

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16, **_kwargs: object
    ) -> BatchParallelGemmaWrapper:
        """Build and evaluate the encoder, agreeing across ranks before wrapping it."""
        _validate_gemma_group(self.group)
        encoder = None
        error = None
        try:
            encoder = self.base.build(device, dtype).eval()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _agree_gemma(self.group, "build", error)
        assert encoder is not None
        return _BatchParallelGemmaWrapper(encoder, self.group, self.src_rank, dtype, device)

    def registry_keys(self, dtype: torch.dtype = torch.bfloat16) -> RegistryKeys:
        """Return the registry keys the base build at ``dtype`` queries and retains on every rank."""
        return self.base.registry_keys(dtype)

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the base builder."""
        return self.base.model_config()

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Each build loads the encoder through the base builder."""
        return False

    @property
    def builds_per_shape(self) -> bool:
        """A built encoder serves any prompt batch."""
        return False


class _AccelerateGemmaWrapper(_AgreedEnhancement, AccelerateGemmaWrapper):
    """Ported source wrapper with collective agreement before result exchange."""

    def __init__(
        self,
        encoder: LTXGemmaTextEncoder | None,
        group: dist.ProcessGroup | None,
        src_rank: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        coordinator: BroadcastCoordinator | None = None,
        *,
        trim_cached_weights: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(encoder, group, src_rank, dtype, device, coordinator)
        self._trim_cached_weights = trim_cached_weights

    def trim_cached_weights(self) -> None:
        """Discard the source builder's dispatched weights after this wrapper is disposed."""
        if self._trim_cached_weights is not None:
            self._trim_cached_weights()

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Agree on source encode errors before broadcasting each output."""
        outputs = [(None, None)] * len(prompts)
        error = None
        if self._rank == self._src_rank:
            try:
                outputs = self._encoder.encode(prompts)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._wait_for_src(*(tensor for hidden, mask in outputs for tensor in (*(hidden or ()), mask)))
        _agree_gemma(self._group, "source encode", error)
        return [self._broadcast_encoder_output(hs, mask, self._src_rank) for hs, mask in outputs]


@dataclass(frozen=True, slots=True)
class AccelerateGemmaBuilder:
    """Build on the designated rank through SimpleBuilder, then dispatch by device map.
    The source rank keeps dispatched weights per dtype for a warm rebuild after
    the wrapper is disposed. An explicit allocator trim discards that cache,
    so the next build reads and dispatches again.
    """

    base: SimpleBuilder[LTXGemmaTextEncoder]
    group: dist.ProcessGroup
    src_rank: int
    _dispatched: dict[torch.dtype, tuple[LTXGemmaTextEncoder, dict[str, torch.Tensor]]] = field(
        default_factory=dict, compare=False, repr=False
    )
    # Every rank constructs its builders in the same order; copies keep the coordinator so signals stay in step.
    _coordinator: BroadcastCoordinator | None = field(default=None, compare=False, repr=False)

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16, **_kwargs: object
    ) -> AccelerateGemmaWrapper:
        """Build on the source rank, dispatch it by device map, and wrap all ranks."""
        _validate_gemma_group(self.group)
        rank = dist.get_rank(self.group)
        encoder = None
        error = None
        try:
            if rank == self.src_rank:
                dispatched = self._dispatched.get(dtype)
                if dispatched is None:
                    # A shell shared through the registry could be rebuilt under the dispatch hooks;
                    # this one belongs to the builder.
                    encoder = self.base.with_registry(DummyRegistry()).build(torch.device("cpu"), dtype).eval()
                    encoder.model.tie_weights()
                    no_split = list(getattr(encoder.model, "_no_split_modules", None) or ())
                    max_memory = get_balanced_memory(encoder.model, dtype=dtype, no_split_module_classes=no_split)
                    device_map = infer_auto_device_map(
                        encoder.model,
                        max_memory=max_memory,
                        dtype=dtype,
                        no_split_module_classes=no_split,
                    )
                    encoder.model = dispatch_model(encoder.model, device_map)
                    dispatched = encoder, encoder.state_dict()
                    self._dispatched[dtype] = dispatched
                encoder, weights = dispatched
                assign_state_dict(encoder, weights)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _hold_until_source(self._coordinator)
        _agree_gemma(self.group, "build", error)
        return _AccelerateGemmaWrapper(
            encoder,
            self.group,
            self.src_rank,
            dtype,
            device,
            coordinator=self._coordinator,
            trim_cached_weights=self._dispatched.clear,
        )

    def registry_keys(self, dtype: torch.dtype = torch.bfloat16) -> RegistryKeys:
        """The source rank builds through a ``DummyRegistry``, so no registry is queried or filled."""
        del dtype
        return RegistryKeys()

    @classmethod
    def from_root(
        cls, root: str | Path, registry: ModelRegistry, group: dist.ProcessGroup, src_rank: int
    ) -> AccelerateGemmaBuilder:
        """Select the Gemma family and share a unit registry across ranks."""
        factory = gemma3_builder if gemma_model_type(str(root)) == "gemma3" else gemma4_builder
        return cls(factory(root).with_registry(registry), group, src_rank)

    def model_config(self) -> dict[str, object]:
        """Read checkpoint configuration through the base builder."""
        return self.base.model_config()

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Weights are dispatched by device map, outside the unit loader's storage."""
        return False

    @property
    def builds_per_shape(self) -> bool:
        """A built encoder serves any prompt batch."""
        return False

    def __post_init__(self) -> None:
        if self._coordinator is None:
            object.__setattr__(self, "_coordinator", _gemma_coordinator(self.group, self.src_rank))
