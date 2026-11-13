# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine features a step target must declare before a deployment that turns them on can build it.

A deployment's core hands each step to one of its step targets (``_select_target``): every
step with context rows to ``PrefillTarget``, every step without to ``DecodeTarget``. A
deployment class whose step targets declare features names them in ``step_targets``; a
deployment class without it runs every step itself and is its own step target.

Each feature is a class whose instance holds the facts that decide support, read from the
engine arguments by :meth:`Feature.from_args`, which returns None while the feature is off.
A step target declares support with the feature's class decorator::

    @PrefillCudaGraph.supported(when=_prefill_cuda_graph_supported)
    class PrefillTarget: ...

``when`` sees only the feature instance and returns whether the target serves it; without
``when`` the target serves every setting of the feature. A step target serves a feature when
it runs correctly with the feature on, including when the feature leaves its steps alone: a
``DecodeTarget`` serves a breakable prefill CUDA graph, which captures only the steps with
context rows. A step target that does not declare a feature does not serve it:
``modeling_v2_resolve`` refuses every deployment one of whose step targets does not serve a
feature it turns on, so a feature added here is refused everywhere until step targets
declare it.

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


# Every feature class, filled as each is defined.
_FEATURES: List[Type["Feature"]] = []


class Feature:
    """An engine feature a step target serves only once it declares it."""

    # step target class -> its ``when``, or None for every setting; one dict per feature class.
    _supported_by: ClassVar[Dict[type, Optional[Callable[["Feature"], bool]]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._supported_by = {}
        _FEATURES.append(cls)

    @classmethod
    def from_args(
        cls, model_config: "ModelConfig", llm_args: "TorchLlmArgs"
    ) -> Optional["Feature"]:
        """The feature as this deployment sets it, or None when the deployment leaves it off."""
        raise NotImplementedError

    @classmethod
    def supported(cls, when: Optional[Callable[["Feature"], bool]] = None):
        """Declare that the decorated step target serves this feature, where ``when`` holds if given."""

        def declare(target: type) -> type:
            cls._supported_by[target] = when
            return target

        return declare

    @classmethod
    def refusal(
        cls, deployment: type, model_config: "ModelConfig", llm_args: "TorchLlmArgs"
    ) -> Optional[str]:
        """Why ``deployment`` cannot serve this feature as the deployment sets it, or None."""
        feature = cls.from_args(model_config, llm_args)
        if feature is None:
            return None
        for target in step_targets(deployment):
            if target in cls._supported_by:
                when = cls._supported_by[target]
                if when is None or when(feature):
                    continue
            name = deployment.__name__
            if target is not deployment:
                name = f"{name}'s {target.__name__}"
            return f"{name} does not support {feature}"
        return None


def feature_refusal(
    deployment: type, model_config: "ModelConfig", llm_args: "TorchLlmArgs"
) -> Optional[str]:
    """Why ``deployment`` cannot serve the features this deployment turns on, or None."""
    refusals = [
        why for feature in _FEATURES if (why := feature.refusal(deployment, model_config, llm_args))
    ]
    return "; ".join(refusals) or None


@dataclass(frozen=True)
class PrefillCudaGraph(Feature):
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
