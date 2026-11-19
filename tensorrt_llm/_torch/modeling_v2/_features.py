# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine features and the step targets that serve them.

A deployment's core hands each step to one of its step targets (``_select_target``): every
step with context rows to ``PrefillTarget``, every step without to ``DecodeTarget``. A
deployment class whose step targets declare features names them in ``step_targets``; a
deployment class without it runs every step itself and is its own step target.

Each feature is a class whose instance holds the facts that decide support, read from the
engine arguments by :meth:`Feature.from_args`, which returns None while the feature is off.
A step target declares itself with the feature's class decorator, optionally with ``when``,
which sees only the feature instance and limits the declaration to the settings it holds
for. A step target serves a feature when it runs correctly with the feature on, including
when the feature leaves its steps alone: a ``DecodeTarget`` serves a breakable prefill CUDA
graph, which captures only the steps with context rows.

Features come in three kinds, and ``modeling_v2_resolve`` treats a step target that does not
serve one the deployment turns on by its kind:

- :class:`BasicFeature`: every step target serves it unless it declares otherwise with
  ``@Feature.unsupported(when=...)``. The deployment is an error, in every mode.
- :class:`AdvancedFunctionalFeature`: a step target serves it only once it declares
  ``@Feature.supported(when=...)``. The route's target is refused: ``auto`` builds the
  built-in implementation and ``require`` raises.
- :class:`AdvancedPerfFeature`: declared like a functional feature, but the route's target
  is built anyway, with a warning that it does not declare the feature.

Picking a different step target when a feature is on is the core's decision, made with an
ordinary branch on ``Feature.from_args`` in ``_select_target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Dict, List, Optional, Tuple, Type

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend, TorchLlmArgs


def step_targets(deployment: type) -> Tuple[type, ...]:
    """The step targets that run the steps of ``deployment``."""
    return getattr(deployment, "step_targets", None) or (deployment,)


class Feature:
    """An engine feature; subclass one of its three kinds, not this class."""

    # step target class -> its ``when``, or None for every setting; one dict per feature class.
    _declared: ClassVar[Dict[type, Optional[Callable[["Feature"], bool]]]]
    # The features of each kind, filled as each is defined.
    _features: ClassVar[List[Type["Feature"]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._declared = {}
        if Feature in cls.__bases__:
            cls._features = []
        else:
            cls._features.append(cls)

    @classmethod
    def from_args(
        cls, model_config: "ModelConfig", llm_args: "TorchLlmArgs"
    ) -> Optional["Feature"]:
        """The feature as this deployment sets it, or None when the deployment leaves it off."""
        raise NotImplementedError

    @classmethod
    def _declare(cls, when: Optional[Callable[["Feature"], bool]]):
        def declare(target: type) -> type:
            cls._declared[target] = when
            return target

        return declare

    @classmethod
    def _declares(cls, target: type, feature: "Feature") -> bool:
        if target not in cls._declared:
            return False
        when = cls._declared[target]
        return when is None or when(feature)

    @classmethod
    def _serves(cls, target: type, feature: "Feature") -> bool:
        raise NotImplementedError

    @classmethod
    def gap(
        cls, deployment: type, model_config: "ModelConfig", llm_args: "TorchLlmArgs"
    ) -> Optional[str]:
        """Which step target of ``deployment`` does not serve this feature as set, or None."""
        feature = cls.from_args(model_config, llm_args)
        if feature is None:
            return None
        for target in step_targets(deployment):
            if cls._serves(target, feature):
                continue
            name = deployment.__name__
            if target is not deployment:
                name = f"{name}'s {target.__name__}"
            return f"{name} does not support {feature}"
        return None

    @classmethod
    def gaps(
        cls, deployment: type, model_config: "ModelConfig", llm_args: "TorchLlmArgs"
    ) -> List[str]:
        """:meth:`gap` of every feature of this kind that ``deployment`` does not serve."""
        return [
            why
            for feature in cls._features
            if (why := feature.gap(deployment, model_config, llm_args))
        ]


class BasicFeature(Feature):
    """A feature every step target serves unless it declares that it does not."""

    @classmethod
    def unsupported(cls, when: Optional[Callable[["Feature"], bool]] = None):
        """Declare that the decorated step target does not serve this feature, where ``when`` holds if given."""
        return cls._declare(when)

    @classmethod
    def _serves(cls, target: type, feature: Feature) -> bool:
        return not cls._declares(target, feature)


class AdvancedFunctionalFeature(Feature):
    """A feature that changes what the engine computes; only the step targets that declare it serve it."""

    @classmethod
    def supported(cls, when: Optional[Callable[["Feature"], bool]] = None):
        """Declare that the decorated step target serves this feature, where ``when`` holds if given."""
        return cls._declare(when)

    @classmethod
    def _serves(cls, target: type, feature: Feature) -> bool:
        return cls._declares(target, feature)


class AdvancedPerfFeature(Feature):
    """A feature that only changes how fast the engine runs; only the step targets that declare it serve it."""

    @classmethod
    def supported(cls, when: Optional[Callable[["Feature"], bool]] = None):
        """Declare that the decorated step target serves this feature, where ``when`` holds if given."""
        return cls._declare(when)

    @classmethod
    def _serves(cls, target: type, feature: Feature) -> bool:
        return cls._declares(target, feature)


@dataclass(frozen=True)
class DecodeCudaGraph(BasicFeature):
    """The decode CUDA graph (``cuda_graph_config`` set)."""

    @classmethod
    def from_args(cls, model_config, llm_args) -> Optional["DecodeCudaGraph"]:
        return None if llm_args.cuda_graph_config is None else cls()

    def __str__(self) -> str:
        return "the decode CUDA graph (cuda_graph_config)"


@dataclass(frozen=True)
class ChunkedPrefill(BasicFeature):
    """Chunked prefill (``enable_chunked_prefill``)."""

    @classmethod
    def from_args(cls, model_config, llm_args) -> Optional["ChunkedPrefill"]:
        return cls() if llm_args.enable_chunked_prefill else None

    def __str__(self) -> str:
        return "enable_chunked_prefill"


@dataclass(frozen=True)
class OverlapScheduler(BasicFeature):
    """The overlap scheduler (``disable_overlap_scheduler`` unset)."""

    @classmethod
    def from_args(cls, model_config, llm_args) -> Optional["OverlapScheduler"]:
        return None if llm_args.disable_overlap_scheduler else cls()

    def __str__(self) -> str:
        return "the overlap scheduler (disable_overlap_scheduler=False)"


@dataclass(frozen=True)
class KvCacheBlockReuse(BasicFeature):
    """KV cache block reuse (``kv_cache_config.enable_block_reuse``)."""

    @classmethod
    def from_args(cls, model_config, llm_args) -> Optional["KvCacheBlockReuse"]:
        return cls() if llm_args.kv_cache_config.enable_block_reuse else None

    def __str__(self) -> str:
        return "kv_cache_config.enable_block_reuse"


@dataclass(frozen=True)
class PrefillCudaGraph(AdvancedPerfFeature):
    """A prefill CUDA graph (``prefill_cuda_graph_backend`` other than ``disabled``).

    The engine captures and replays it for steps with context rows only, but ``piecewise``
    builds it from a torch.compile of the model, which also runs the steps without context
    rows unless ``torch_compile_config.compile_only_piecewise_graphs`` limits it to the
    graphed ones.
    """

    backend: "PrefillCudaGraphBackend"
    attention_dp: bool
    #: Whether the torch.compile that ``piecewise`` needs also runs the decode steps.
    compiles_decode: bool

    @classmethod
    def from_args(cls, model_config, llm_args) -> Optional["PrefillCudaGraph"]:
        backend = llm_args.prefill_cuda_graph_backend
        if backend == "disabled":
            return None
        compile_config = llm_args.torch_compile_config
        compiles_decode = backend == "piecewise" and not (
            compile_config is not None and compile_config.compile_only_piecewise_graphs
        )
        return cls(backend, model_config.mapping.enable_attention_dp, compiles_decode)

    def __str__(self) -> str:
        words = f"prefill_cuda_graph_backend='{self.backend.value}'"
        return f"{words} with attention DP" if self.attention_dp else words
