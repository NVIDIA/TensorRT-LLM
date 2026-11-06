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
"""What a KV cache manager's PageTableSpec says of its roles."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import (
    ByHead,
    Own,
    PageTableSpec,
    Replicated,
    kv_dtype_name,
    layer_groups,
)

pytestmark = pytest.mark.cpu_only


class _Role(str):
    """A role as the runtime names one: a string subclass."""


def test_spec_names_roles_and_layer_ids_by_plain_values():
    spec = PageTableSpec(
        role_layouts={_Role("k"): ByHead()},
        untransferred_roles=[_Role("pad")],
        derived_roles={_Role("v_scale")},
        request_rebuilt_roles=(_Role("swa"),),
        layer_ids={3: (6.0, 3)},
    )

    assert dict(spec.role_layouts) == {"k": ByHead()}
    assert all(type(role) is str for role in spec.role_layouts)
    assert spec.untransferred_roles == frozenset({"pad"})
    assert spec.derived_roles == frozenset({"v_scale"})
    assert spec.request_rebuilt_roles == frozenset({"swa"})
    assert dict(spec.layer_ids) == {3: (6, 3)}


def test_spec_names_mixed_precision_types_as_a_pair():
    assert PageTableSpec(mixed_precision=["nvfp4", "bf16"]).mixed_precision == ("nvfp4", "bf16")
    assert PageTableSpec().mixed_precision is None
    with pytest.raises(ValueError, match="low and a high type"):
        PageTableSpec(mixed_precision=("nvfp4",))


def test_spec_is_immutable():
    spec = PageTableSpec(role_layouts={"k": ByHead()}, layer_ids={0: (0, 0)})

    with pytest.raises(TypeError):
        spec.role_layouts["v"] = ByHead()
    with pytest.raises(TypeError):
        spec.layer_ids[1] = (1, 1)
    with pytest.raises(FrozenInstanceError):
        spec.quant_mode = "fp8"


def test_spec_with_fields_replaced_keeps_the_others():
    base = PageTableSpec(role_layouts={"k": ByHead()}, quant_mode="fp8")

    spec = replace(base, role_layouts={**base.role_layouts, "index_k": Replicated()})

    assert dict(spec.role_layouts) == {"k": ByHead(), "index_k": Replicated()}
    assert spec.quant_mode == "fp8"


def test_layout_of_a_role_falls_back_to_the_default():
    spec = PageTableSpec(
        role_layouts={"k": ByHead()},
        untransferred_roles={"pad"},
        derived_roles={"v_scale"},
    )

    assert spec.layout_of("k") == ByHead()
    # A role that never moves is this rank's own share.
    assert spec.layout_of("pad") == Own()
    assert spec.layout_of("v_scale") == Own()
    # A role that moves and that the spec does not lay out has no layout.
    assert spec.layout_of("index_k") is None
    assert replace(spec, default_layout=Replicated()).layout_of("index_k") == Replicated()


def test_a_role_that_never_moves_is_the_rank_s_own_share_whatever_else_lays_it_out():
    spec = PageTableSpec(
        role_layouts={"pad": ByHead()},
        default_layout=Replicated(),
        untransferred_roles={"pad"},
        derived_roles={"v_scale"},
    )

    assert spec.layout_of("pad") == Own()
    assert spec.layout_of("v_scale") == Own()


def test_layer_groups_list_each_layer_once_in_buffer_order():
    def buffer(layer, role):
        return SimpleNamespace(layer_id=layer, role=role)

    manager = SimpleNamespace(
        impl=SimpleNamespace(
            buffer_grouping=[
                [buffer(2, "k"), buffer(0, "k"), buffer(2, "v"), buffer(0, "v")],
                [buffer(1, "ssm")],
                # A layer whose buffers the reuse policies split lies in each of its groups.
                [buffer(0, "index_k")],
            ]
        )
    )

    assert layer_groups(manager) == [[2, 0], [1], [0]]


def test_kv_dtype_name_reads_the_manager_s_data_type():
    assert kv_dtype_name(SimpleNamespace(dtype=SimpleNamespace(name="FP8"))) == "FP8"
    assert kv_dtype_name(SimpleNamespace(dtype="bfloat16")) == "bfloat16"
    assert kv_dtype_name(SimpleNamespace()) is None
