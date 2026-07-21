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

import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import PostTransformQualificationDecision

ModelFactory = Callable[[], nn.Module]
ModelQualifier = Callable[[nn.Module], PostTransformQualificationDecision]
ReceiverPreparer = Callable[[nn.Module, nn.Module], None]
StateProbe = Callable[[nn.Module], object]


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

    `prepare_receiver` simulates installation of a producer's post-transform
    tensors when a family needs value-level coverage. A real MX donor/receiver
    test is still required before enabling a production profile.
    """

    profile_id: str
    model_factory: ModelFactory
    qualify_model: ModelQualifier
    state_probes: tuple[tuple[str, StateProbe], ...]
    prepare_receiver: ReceiverPreparer = copy_post_transform_parameters

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Qualification case profile_id must not be empty")
        if not self.state_probes:
            raise ValueError("Qualification case must define at least one state probe")
        probe_names = tuple(name for name, _probe in self.state_probes)
        if len(probe_names) != len(set(probe_names)):
            raise ValueError("Qualification case state probe names must be unique")


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

    transform_calls = _install_transform_call_recorders(receiver)
    ModelLoader._walk_full_post_load(producer)
    ModelLoader._setup_aliases(receiver)
    case.prepare_receiver(producer, receiver)

    # Runtime finalization repeats setup_aliases(), whose contract is
    # idempotent, after the checkpoint loader's pre-transfer preparation.
    ModelLoader._setup_aliases(receiver)
    ModelLoader._mark_weights_transformed(receiver)
    ModelLoader._walk_cache_state(receiver)

    assert transform_calls == [], f"Staged receiver invoked transform_weights(): {transform_calls}"
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

    return producer, receiver
