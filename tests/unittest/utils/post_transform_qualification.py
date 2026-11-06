# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reusable unit harness for post-transform receiver qualification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.moe.fused_moe.fused_moe_vanilla import VanillaMoE
from tensorrt_llm._torch.moe.fused_moe.weight_owner import is_moe_weight_owner
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import (
    PostTransformQualificationDecision,
    PostTransformQualificationReason,
    canonical_tensor_bytes,
)

ModelFactory = Callable[[], nn.Module]
ModelQualifier = Callable[[nn.Module], PostTransformQualificationDecision]
ReceiverPreparer = Callable[[nn.Module, nn.Module], None]
ProducerPreparer = Callable[[nn.Module], None]
StateProbe = Callable[[nn.Module], object]
OutputProbe = Callable[[nn.Module], object]

# Runtime constraints that a profile qualifying a model with routed experts
# must pin: the routed-expert layout depends on the MoE backend and partition.
_MOE_RUNTIME_CONSTRAINTS = ("moe_backends", "moe_tp_sizes", "moe_ep_sizes", "moe_cluster_sizes")


class AllReduceStub(nn.Module):
    """Identity all-reduce for simulating one tensor-parallel rank in a single process."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()

    def uses_nccl_symmetric_memory_window(self) -> bool:
        """Report that no NCCL symmetric memory window is in use."""
        return False

    def forward(self, tensor: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        """Return the local partial result unchanged."""
        return tensor


def mx_canonical_parameter_catalog(
    model: nn.Module,
) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    """Mirror the parameter view that MX registers and transfers, without importing MX.

    MX walks `named_parameters(remove_duplicate=False)`, skips the runtime
    alias names, and deduplicates by storage.
    """
    catalog = {}
    seen_storages = set()
    runtime_alias_components = {"next_attn", "next_layer_layernorm"}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        storage = (parameter.device.type, parameter.device.index, parameter.data_ptr())
        if runtime_alias_components.intersection(name.split(".")):
            continue
        if storage in seen_storages:
            continue
        seen_storages.add(storage)
        catalog[name] = (parameter.data_ptr(), tuple(parameter.shape), parameter.dtype)
    return catalog


def module_state_snapshot(model: nn.Module) -> dict[str, object]:
    """Snapshot the non-tensor state of every module for a state probe.

    Captures scalar, string, enum, and `None` attributes of each module and the
    contents of every Hugging Face config that a module holds. A staged
    receiver never runs `load_weights()`, so any of this state that only the
    load path sets differs from a producer prepared with `prepare_producer`.
    """
    state: dict[str, object] = {}
    seen_configs: set[int] = set()
    for module_name, module in model.named_modules():
        for attribute, value in vars(module).items():
            key = f"{module_name}.{attribute}"
            if isinstance(value, (bool, int, float, str, Enum, type(None))):
                state[key] = value
            elif isinstance(value, PretrainedConfig) and id(value) not in seen_configs:
                seen_configs.add(id(value))
                state[key] = value.to_dict()
    return state


def keep_factory_parameters(
    prepare_producer: ProducerPreparer,
) -> tuple[ProducerPreparer, dict[str, torch.Tensor]]:
    """Wrap `prepare_producer` to copy the factory parameters before it overwrites them.

    Returns the wrapped preparer for `PostTransformQualificationCase` and the
    copies, keyed by parameter name, for `assert_every_parameter_was_loaded`.
    """
    factory_parameters: dict[str, torch.Tensor] = {}

    def prepare(model: nn.Module) -> None:
        factory_parameters.update(
            (name, parameter.detach().clone()) for name, parameter in model.named_parameters()
        )
        prepare_producer(model)

    return prepare, factory_parameters


def assert_every_parameter_was_loaded(
    producer: nn.Module, factory_parameters: dict[str, torch.Tensor]
) -> None:
    """Assert that `prepare_producer` overwrote every parameter of the factory fill.

    The lifecycle probes then compare bytes that came from the checkpoint
    through the weight mapper rather than the factory's placeholder values.
    `factory_parameters` comes from `keep_factory_parameters`.
    """
    assert factory_parameters, "The producer was never prepared"
    # Look parameters up by the factory's canonical names: after
    # `setup_aliases()` the producer also reaches some of them through
    # `next_layer_layernorm` alias names.
    unchanged = [
        name
        for name, factory_parameter in factory_parameters.items()
        if torch.equal(producer.get_parameter(name), factory_parameter)
    ]
    assert unchanged == [], f"Parameters not written by the producer load: {unchanged}"


def copy_post_transform_parameters(producer: nn.Module, receiver: nn.Module) -> None:
    """Simulate an exact-name post-transform parameter transfer."""
    producer_parameters = dict(producer.named_parameters())
    receiver_parameters = dict(receiver.named_parameters())
    assert receiver_parameters.keys() == producer_parameters.keys(), (
        "Producer and receiver parameter names differ before staged finalization"
    )

    with torch.no_grad():
        for name, producer_parameter in producer_parameters.items():
            receiver_parameter = receiver_parameters[name]
            assert receiver_parameter.shape == producer_parameter.shape, (
                f"Parameter {name!r} shape differs: receiver "
                f"{tuple(receiver_parameter.shape)} != producer "
                f"{tuple(producer_parameter.shape)}"
            )
            assert receiver_parameter.dtype == producer_parameter.dtype, (
                f"Parameter {name!r} dtype differs: receiver "
                f"{receiver_parameter.dtype} != producer {producer_parameter.dtype}"
            )
            assert receiver_parameter.layout == producer_parameter.layout, (
                f"Parameter {name!r} layout differs: receiver "
                f"{receiver_parameter.layout} != producer {producer_parameter.layout}"
            )
            if producer_parameter.layout is torch.strided:
                assert receiver_parameter.stride() == producer_parameter.stride(), (
                    f"Parameter {name!r} stride differs: receiver "
                    f"{receiver_parameter.stride()} != producer "
                    f"{producer_parameter.stride()}"
                )
                assert receiver_parameter.storage_offset() == producer_parameter.storage_offset(), (
                    f"Parameter {name!r} storage offset differs: receiver "
                    f"{receiver_parameter.storage_offset()} != producer "
                    f"{producer_parameter.storage_offset()}"
                )
            receiver_parameter.copy_(producer_parameter)


@dataclass(frozen=True)
class PostTransformQualificationCase:
    """Inputs needed to prove full and staged post-load lifecycle equivalence.

    `unqualified_model_factory` must preserve the candidate config dimensions
    while returning an unregistered exact root type. `output_probes` should be
    deterministic and exercise the deepest practical output path for the unit
    fixture; real donor/receiver GPU output coverage remains a separate gate.

    `prepare_receiver` simulates installation of a producer's post-transform
    tensors when a family needs value-level coverage. A real MX donor/receiver
    test is still required before enabling a production profile.

    `prepare_producer`, when set, runs the family's real checkpoint load path
    on the producer before its full post-load walk. The receiver never loads,
    as on a staged MX receiver, so the state and output probes then also cover
    any state that only `load_weights()` or the weight mapper sets.
    """

    profile_id: str
    model_factory: ModelFactory
    unqualified_model_factory: ModelFactory
    qualify_model: ModelQualifier
    state_probes: tuple[tuple[str, StateProbe], ...]
    output_probes: tuple[tuple[str, OutputProbe], ...]
    prepare_receiver: ReceiverPreparer = copy_post_transform_parameters
    prepare_producer: ProducerPreparer | None = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Qualification case profile_id must not be empty")
        if not self.state_probes:
            raise ValueError("Qualification case must define at least one state probe")
        if not self.output_probes:
            raise ValueError("Qualification case must define at least one output probe")
        state_probe_names = tuple(name for name, _probe in self.state_probes)
        if len(state_probe_names) != len(set(state_probe_names)):
            raise ValueError("Qualification case state probe names must be unique")
        output_probe_names = tuple(name for name, _probe in self.output_probes)
        if len(output_probe_names) != len(set(output_probe_names)):
            raise ValueError("Qualification case output probe names must be unique")


def _transform_guard_state(model: nn.Module) -> dict[str, bool]:
    return {
        name: module._weights_transformed
        for name, module in model.named_modules()
        if hasattr(module, "_weights_transformed")
        and not getattr(module, "_weights_removed", False)
    }


def _assert_named_tensors_equal(
    producer_tensors: dict[str, torch.Tensor],
    receiver_tensors: dict[str, torch.Tensor],
    *,
    tensor_kind: str,
) -> None:
    assert receiver_tensors.keys() == producer_tensors.keys(), (
        f"Producer and receiver {tensor_kind} names differ after staged finalization"
    )
    for name, producer_tensor in producer_tensors.items():
        receiver_tensor = receiver_tensors[name]
        assert receiver_tensor.shape == producer_tensor.shape, (
            f"Post-transform {tensor_kind} {name!r} shape differs: receiver "
            f"{tuple(receiver_tensor.shape)} != producer {tuple(producer_tensor.shape)}"
        )
        assert receiver_tensor.dtype == producer_tensor.dtype, (
            f"Post-transform {tensor_kind} {name!r} dtype differs: receiver "
            f"{receiver_tensor.dtype} != producer {producer_tensor.dtype}"
        )
        assert receiver_tensor.layout == producer_tensor.layout, (
            f"Post-transform {tensor_kind} {name!r} layout differs: receiver "
            f"{receiver_tensor.layout} != producer {producer_tensor.layout}"
        )
        if producer_tensor.layout is torch.strided:
            assert receiver_tensor.stride() == producer_tensor.stride(), (
                f"Post-transform {tensor_kind} {name!r} stride differs: receiver "
                f"{receiver_tensor.stride()} != producer {producer_tensor.stride()}"
            )
            assert receiver_tensor.storage_offset() == producer_tensor.storage_offset(), (
                f"Post-transform {tensor_kind} {name!r} storage offset differs: "
                f"receiver {receiver_tensor.storage_offset()} != producer "
                f"{producer_tensor.storage_offset()}"
            )
        try:
            torch.testing.assert_close(
                receiver_tensor,
                producer_tensor,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        except AssertionError as error:
            raise AssertionError(f"Post-transform {tensor_kind} {name!r} values differ") from error
        # Exact-value equality still accepts `+0.0 == -0.0` and any-NaN == any-NaN;
        # the byte comparison below is the contract the MX weight manifest enforces.
        if not torch.equal(
            canonical_tensor_bytes(receiver_tensor), canonical_tensor_bytes(producer_tensor)
        ):
            raise AssertionError(f"Post-transform {tensor_kind} {name!r} bytes differ")


def _holds_routed_experts(model: nn.Module) -> bool:
    return any(
        is_moe_weight_owner(module) or isinstance(module, VanillaMoE) for module in model.modules()
    )


def _runs_real_transform(module: nn.Module) -> bool:
    """Whether `ModelLoader._walk_transform` would run a transform on `module`.

    Reads the hook from the class, because the call recorders replace it on the
    instance; the no-op default of `DecoderModelForCausalLM` transforms nothing.
    """
    hook = getattr(type(module), "transform_weights", None)
    return (
        hook is not None
        and hook is not DecoderModelForCausalLM.transform_weights
        and not getattr(module, "_weights_removed", False)
    )


def _install_transform_call_recorders(model: nn.Module) -> list[str]:
    calls: list[str] = []
    for name, module in model.named_modules():
        if getattr(module, "transform_weights", None) is None:
            continue

        def _record_transform_weights(
            *_args: object,
            module_name: str = name,
            module_type_name: str = type(module).__name__,
            **_kwargs: object,
        ) -> None:
            calls.append(module_name or module_type_name)

        module.transform_weights = _record_transform_weights
    return calls


def assert_post_transform_lifecycle_equivalent(
    case: PostTransformQualificationCase,
) -> tuple[nn.Module, nn.Module]:
    """Assert one exact profile's full and staged lifecycle states agree.

    Args:
        case: Model construction, registry lookup, transfer simulator, and
            family-specific comparable state probes.

    Returns:
        The fully post-loaded producer and staged receiver for optional
        additional assertions by the caller.
    """
    producer = case.model_factory()
    receiver = case.model_factory()
    assert producer is not receiver, "model_factory must return a fresh model"

    decision = case.qualify_model(receiver)
    assert decision.qualified, (
        f"Profile {case.profile_id!r} is not registered: {decision.reason.value}"
    )
    assert decision.profile is not None
    assert decision.profile.profile_id == case.profile_id
    # MoE-ness comes from the constructed model, not from the profile, so a
    # profile that leaves the routed-expert backend or partition open cannot
    # pass as a dense one.
    if _holds_routed_experts(receiver):
        constraints = decision.profile.runtime_constraints
        unpinned = [name for name in _MOE_RUNTIME_CONSTRAINTS if getattr(constraints, name) is None]
        assert not unpinned, (
            f"Profile {case.profile_id!r} qualifies a model with routed experts but leaves "
            f"{unpinned} unconstrained"
        )

    unqualified_model = case.unqualified_model_factory()
    unqualified_decision = case.qualify_model(unqualified_model)
    assert not unqualified_decision.qualified, (
        f"Negative-control root {type(unqualified_model).__name__!r} unexpectedly "
        f"inherited profile {case.profile_id!r}"
    )
    assert (
        unqualified_decision.reason
        is PostTransformQualificationReason.ROOT_MODEL_CLASS_NOT_REGISTERED
    ), (
        f"Negative-control root {type(unqualified_model).__name__!r} was rejected for "
        f"{unqualified_decision.reason.value!r}, not exact root-class registration"
    )

    transform_calls = _install_transform_call_recorders(receiver)
    if case.prepare_producer is not None:
        case.prepare_producer(producer)
    ModelLoader._walk_full_post_load(producer)
    ModelLoader._setup_aliases(receiver)
    case.prepare_receiver(producer, receiver)

    # Runtime finalization repeats setup_aliases(), whose contract is
    # idempotent, after the checkpoint loader's pre-transfer preparation.
    ModelLoader._setup_aliases(receiver)
    ModelLoader._mark_weights_transformed(receiver)
    ModelLoader._walk_cache_state(receiver)

    assert transform_calls == [], f"Staged receiver invoked transform_weights(): {transform_calls}"
    # Staged finalization marks only modules that already carry the guard, and
    # a later transform walk runs every module left unmarked, so each module
    # with a transform must declare the guard before any weight load.
    unmarked = [
        name or type(module).__name__
        for name, module in receiver.named_modules()
        if _runs_real_transform(module)
        and getattr(module, "_weights_transformed", False) is not True
    ]
    assert unmarked == [], f"Staged finalization left transform modules unmarked: {unmarked}"
    assert _transform_guard_state(receiver) == _transform_guard_state(producer)

    _assert_named_tensors_equal(
        dict(producer.named_parameters()),
        dict(receiver.named_parameters()),
        tensor_kind="parameter",
    )
    _assert_named_tensors_equal(
        dict(producer.named_buffers()),
        dict(receiver.named_buffers()),
        tensor_kind="buffer",
    )

    for probe_name, probe in case.state_probes:
        assert probe(receiver) == probe(producer), (
            f"Post-transform qualification probe {probe_name!r} differs "
            f"for profile {case.profile_id!r}"
        )

    for probe_name, probe in case.output_probes:
        with torch.no_grad():
            producer_output = probe(producer)
            receiver_output = probe(receiver)
        try:
            torch.testing.assert_close(
                receiver_output,
                producer_output,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        except AssertionError as error:
            raise AssertionError(
                f"Post-transform output probe {probe_name!r} differs "
                f"for profile {case.profile_id!r}"
            ) from error

    return producer, receiver
