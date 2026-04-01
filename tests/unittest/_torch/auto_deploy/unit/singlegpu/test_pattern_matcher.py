# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.utils import pattern_matcher as ad_pattern_matcher
from tensorrt_llm._torch.auto_deploy.utils.pattern_matcher import (
    ADPatternMatcherPass,
    register_ad_pattern,
)


class _LateCastMultiOutputModule(nn.Module):
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add = residual + x
        cast = weight.to(torch.float32)
        out = add * cast
        return out, add


def _multi_output_search(
    x: torch.Tensor, residual: torch.Tensor, cast: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    add = residual + x
    out = add * cast
    return out, add


def _multi_output_replace(
    x: torch.Tensor, residual: torch.Tensor, cast: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    add = residual + x
    out = add * cast
    return out, add


def _build_repro_gm() -> tuple[torch.fx.GraphModule, tuple[torch.Tensor, ...]]:
    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)
    weight = torch.randn(4, 8, dtype=torch.float16)
    gm = torch_export_to_gm(_LateCastMultiOutputModule(), args=(x, residual, weight), clone=True)
    return gm, (x, residual, weight)


def _apply_multi_output_replacement(gm: torch.fx.GraphModule) -> int:
    patterns = ADPatternMatcherPass()
    register_ad_pattern(
        search_fn=_multi_output_search,
        replace_fn=_multi_output_replace,
        patterns=patterns,
        dummy_args=[
            torch.randn(4, 8),
            torch.randn(4, 8),
            torch.randn(4, 8),
        ],
    )
    return patterns.apply(gm.graph)


def test_multi_output_replacement_needs_topological_repair(monkeypatch: pytest.MonkeyPatch):
    gm, _ = _build_repro_gm()
    monkeypatch.setattr(ad_pattern_matcher, "stable_topological_sort", lambda graph: None)

    assert _apply_multi_output_replacement(gm) == 1

    with pytest.raises(RuntimeError, match="used before it has been defined"):
        gm.graph.lint()


def test_multi_output_replacement_stable_sort_restores_topological_order():
    gm, args = _build_repro_gm()
    expected = _LateCastMultiOutputModule()(*args)

    assert _apply_multi_output_replacement(gm) == 1

    gm.graph.lint()
    gm.recompile()
    actual = gm(*args)

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])
