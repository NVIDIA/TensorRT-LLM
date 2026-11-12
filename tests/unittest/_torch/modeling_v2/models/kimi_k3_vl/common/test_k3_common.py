# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's ``common/`` glue behaves as the Modeling V1 helpers.

* ``parallel.TensorParallelMode`` and ``modules.linear.TensorParallelMode``: same members, equal
  and hash-equal across the two classes (``LMHead.tp_mode`` is the V1 enum), same ``split_dim``
  and ``flip``.
* ``multi_stream``: ``do_multi_stream`` is the engine's flag itself; the parallel helpers are the
  V1 code token for token; with the flag off both callables run in order on the calling stream.
* ``weight_io``: ``filter_weights`` / ``run_concurrently`` as V1's; ``text_model_weights`` gives the
  text loader exactly what the vision-language wrapper gave it, without the vision tensors.
"""

import importlib
from pathlib import Path

import pytest

__extra_import_path__ = [".."]
from _k3_source_parity import REPO, function_dump

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_MULTI_STREAM_V1 = REPO / "tensorrt_llm/_torch/modules/multi_stream_utils.py"
_MULTI_STREAM = (
    REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/common/multi_stream.py"
)


def _common(name):
    return importlib.import_module(f"{_FAMILY}.common.{name}")


def test_tensor_parallel_mode_matches_v1():
    from tensorrt_llm._torch.modules.linear import TensorParallelMode as V1

    Fam = _common("parallel").TensorParallelMode
    assert [(m.name, m.value) for m in Fam] == [(m.name, m.value) for m in V1]
    for fam, v1 in zip(Fam, V1):
        assert fam == v1 and v1 == fam and hash(fam) == hash(v1)
        assert fam != next(m for m in V1 if m is not v1)
        assert {fam: 1}[v1] == 1
    for mode in (*Fam, *V1, None):
        assert Fam.split_dim(mode) == V1.split_dim(mode)
        assert Fam.flip(mode) == V1.flip(mode)
        assert isinstance(Fam.flip(mode), Fam)


def test_do_multi_stream_is_the_engine_flag():
    from tensorrt_llm._torch.modules import multi_stream_utils as v1

    fam = _common("multi_stream")
    assert fam.do_multi_stream is v1.do_multi_stream
    assert not fam.do_multi_stream()
    with v1.with_multi_stream(True):
        assert fam.do_multi_stream()
    assert not fam.do_multi_stream()


@pytest.mark.parametrize(
    "name",
    ["_record_stream", "maybe_execute_in_parallel_deferred_join", "maybe_execute_in_parallel"],
)
def test_multi_stream_helpers_are_the_v1_code(name):
    assert function_dump(_MULTI_STREAM, name) == function_dump(_MULTI_STREAM_V1, name)


def test_single_stream_runs_in_order():
    fam = _common("multi_stream")
    calls = []
    out = fam.maybe_execute_in_parallel(
        lambda: calls.append(0) or "a", lambda: calls.append(1) or "b", None, None, aux_stream=None
    )
    assert out == ("a", "b") and calls == [0, 1]
    assert fam.maybe_execute_in_parallel_deferred_join(lambda: 1, lambda: 2, None, None) == (
        1,
        2,
        None,
    )


def test_filter_weights_and_run_concurrently_match_v1():
    from tensorrt_llm._torch.models import modeling_utils as v1

    fam = _common("weight_io")
    weights = {"language_model.a": 1, "language_model.b.c": 2, "vision_tower.x": 3, "other": 4}
    assert fam.filter_weights("language_model", weights) == v1.filter_weights(
        "language_model", weights
    )
    jobs = [(i, i + 1) for i in range(32)]
    got, want = [], []
    fam.run_concurrently(lambda a, b: a * b, jobs, reduce_func=got.append, num_workers=4)
    v1.run_concurrently(lambda a, b: a * b, jobs, reduce_func=want.append, num_workers=4)
    assert sorted(got) == sorted(want)

    def fail(i):
        raise ValueError(f"job {i}")

    with pytest.raises(ValueError, match="job"):
        fam.run_concurrently(fail, [(0,)], num_workers=1)


class _Lazy(dict):
    """The engine's lazily mapped checkpoint: a mapping with ``checkpoint_dir``."""

    checkpoint_dir = "/ckpt"


def test_text_model_weights_skip_the_vision_tower():
    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
    from tensorrt_llm._torch.models.modeling_utils import filter_weights

    fam = _common("weight_io")
    weights = _Lazy(
        {
            "language_model.model.layers.0.mlp.w": 1,
            "language_model.lm_head.weight": 2,
            "vision_tower.blocks.0.attn.qkv.weight": 3,
            "mm_projector.proj.0.weight": 4,
        }
    )
    got = fam.text_model_weights(weights)
    # What the vision-language wrapper handed the text model (KimiK3DefaultWeights.load_weights).
    want = ConsumableWeightsDict(filter_weights("language_model", weights))
    assert isinstance(got, ConsumableWeightsDict)
    assert dict(got.items()) == dict(want.items())
    assert not any(k.startswith(("vision_tower.", "mm_projector.")) for k in got.keys())
    assert got.checkpoint_dir == "/ckpt" and got.checkpoint_prefix == "language_model."

    # A text-only checkpoint (no language_model.* keys) goes to the text loader as is.
    plain = {"model.layers.0.mlp.w": 1}
    assert fam.text_model_weights(plain) is plain


def test_check_text_only():
    fam = _common("weight_io")
    fam.check_text_only(None)
    fam.check_text_only([])
    with pytest.raises(ValueError, match="text only"):
        fam.check_text_only([object()])


def test_common_imports_no_target_code():
    """``common/`` is family code: it imports no target package."""
    import ast

    root = Path(_MULTI_STREAM).parent
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "__sm_" not in module, f"{path.name} imports {module}"
