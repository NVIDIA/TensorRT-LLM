# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the Eagle modeling code (modeling_eagle.py)."""

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    EagleConfig,
    EagleWrapper,
    get_eagle_layers,
)


def _make_graph_module_with_placeholders(*names):
    graph = torch.fx.Graph()
    output = None
    for name in names:
        placeholder = graph.placeholder(name)
        if output is None:
            output = placeholder
    graph.output(output)
    return torch.fx.GraphModule(nn.Module(), graph)


def _eager_wrapper_around(inner_graph_module):
    """An eager wrapper mimicking a VLM (vision tower eager, exported inner text graph nested)."""
    wrapper = nn.Module()
    wrapper.model = nn.Module()
    wrapper.model.language_model = inner_graph_module
    return wrapper


def _wrapper_tree(*submodules):
    """A minimal stand-in for the EagleWrapper module tree (the ``self`` the filter searches)."""
    tree = nn.Module()
    for i, submodule in enumerate(submodules):
        setattr(tree, f"submodule_{i}", submodule)
    return tree


def test_graph_submodules_retrieval():
    """_graph_submodules recursively collects every GraphModule in a module (including nested)."""
    inner_gm = _make_graph_module_with_placeholders("inputs_embeds")
    eager = _eager_wrapper_around(inner_gm)  # inner_gm nested at .model.language_model (depth 2)
    draft = _make_graph_module_with_placeholders("hidden_states")
    tree = _wrapper_tree(eager, draft)

    assert set(EagleWrapper._graph_submodules(tree)) == {inner_gm, draft}
    assert EagleWrapper._graph_submodules(draft) == [draft]  # a GraphModule reports itself
    assert EagleWrapper._graph_submodules(eager) == [inner_gm]  # recurses into the eager wrapper


# The submodule under test declares OWN; a second submodule in the tree declares OTHER (sharing the
# IO placeholders); ORPHANS are runtime inputs that belong to no submodule's graph -- the kind the
# eager VLM forward consumes (vision metadata + the mrope_delta_cache resource).
_OWN = ["inputs_embeds", "position_ids", "r0_kv_cache"]
_OTHER = ["inputs_embeds", "position_ids", "r1_kv_cache"]
_ORPHANS = ["pixel_values", "r2_mrope_delta_cache"]


@pytest.mark.parametrize(
    "eager_wrapper, include_unassigned, expect_orphans",
    [
        pytest.param(False, False, False, id="graph_module-flag_off"),
        pytest.param(False, True, False, id="graph_module-flag_on_ignored"),
        pytest.param(True, False, False, id="eager_wrapper-flag_off"),
        pytest.param(True, True, True, id="eager_wrapper-flag_on_collects_unassigned"),
    ],
)
def test_filter_kwargs_for_submodule(eager_wrapper, include_unassigned, expect_orphans):
    """A submodule gets its own graph placeholders, plus orphans only for an eager wrapper when asked.

    The "unassigned" (orphan) inputs belong to no submodule's graph -- the kind the eager VLM forward
    consumes (vision metadata + the mrope_delta_cache). A strict GraphModule gets only its
    placeholders regardless of the flag; the eager wrapper here mimics a VLM and collects the orphans
    only when include_unassigned_kwargs=True.
    """
    own_gm = _make_graph_module_with_placeholders(*_OWN)
    other_gm = _make_graph_module_with_placeholders(*_OTHER)
    submodule = _eager_wrapper_around(own_gm) if eager_wrapper else own_gm
    tree = _wrapper_tree(submodule, other_gm)

    kwargs = {name: torch.empty(1) for name in (*_OWN, *_OTHER, *_ORPHANS)}
    filtered = EagleWrapper._filter_kwargs_for_submodule(
        tree, kwargs, submodule, include_unassigned_kwargs=include_unassigned
    )

    expected = set(_OWN) | (set(_ORPHANS) if expect_orphans else set())
    assert set(filtered) == expected  # never the other submodule's r1_kv_cache
    for name, value in filtered.items():
        assert value is kwargs[name]


def test_mtp_builder_rejects_multiple_layers():
    config = PretrainedConfig(mtp_num_hidden_layers=2)
    model_type = "qwen3_5_moe_text"
    eagle_config = EagleConfig.from_base_config(config, model_type)

    with pytest.raises(ValueError):
        get_eagle_layers(eagle_config, model_type)
