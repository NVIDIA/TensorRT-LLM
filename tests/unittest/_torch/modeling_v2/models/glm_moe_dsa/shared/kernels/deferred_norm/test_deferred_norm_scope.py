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
"""Single-process tests of the deferred_norm scope (``deferred_norm/scope.py``).

The scope logic (this target's ``shared`` scope, the next layer's indexer type, the constructing
decoder layer found from the calling frames) needs only the standard library and the target's
``dispatch`` and runs without a built ``tensorrt_llm``. The remaining tests need the installed
package and the GLM-5.2 checkpoint:

* the GLM-5.2 indexer layout seen by the scope (the model config's
  ``to_sparse_params``) has 18 full-indexer layers among the 74 next layers of
  the deferred_norm boundaries (6, 10, ..., 74) and agrees with the native attention
  module (an ``Indexer`` exactly on full layers);
* the consumer guard: under ``shared``, this target's scope, the deferred_norm projection
  declines a full-indexer layer (so a packet there is materialized for the
  native path) and accepts shared layers.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[9]
TARGET = (
    REPO
    / "tensorrt_llm"
    / "_torch"
    / "_experimental"
    / "modeling_v2"
    / "models"
    / "glm_moe_dsa"
    / "shared"
)
_STANDALONE = "_deferred_norm_scope_standalone"
# GLM-5.2: 78 target layers, full-indexer layers where max(i - 2, 0) % 4 == 0.
GLM52 = dict(
    num_hidden_layers=78, index_topk_freq=4, index_skip_topk_offset=3, index_topk_pattern=None
)
FULL_NEXT = list(range(6, 78, 4))
BOUNDARIES = list(range(3, 77))  # L -> L+1 with a deferred_norm hop (first_k_dense_replace = 3)


def _load_dispatch_standalone(name):
    """The target's ``dispatch.py`` as module ``name``, without importing ``tensorrt_llm``."""
    spec = importlib.util.spec_from_file_location(
        "deferred_norm_scope_input_prep_testlib",
        Path(__file__).resolve().parents[1] / "deferred_input_prep" / "input_prep_testlib.py",
    )
    testlib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(testlib)
    return testlib.load_dispatch_standalone(name)


def _load_scope():
    """``scope.py``: the installed package, else the source tree under a stand-in package."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
            scope,
        )

        return scope
    except (ImportError, OSError):
        pass
    if _STANDALONE not in sys.modules:
        for name, path in (
            (_STANDALONE, TARGET),
            (f"{_STANDALONE}.kernels", TARGET / "kernels"),
            (f"{_STANDALONE}.kernels.deferred_norm", TARGET / "kernels" / "deferred_norm"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        sys.modules[f"{_STANDALONE}.dispatch"] = _load_dispatch_standalone(
            f"{_STANDALONE}.dispatch"
        )
    return importlib.import_module(f"{_STANDALONE}.kernels.deferred_norm.scope")


@pytest.fixture
def scope():
    return _load_scope()


class _Sparse:
    """``to_sparse_params`` with GLM-5.2's frequency rule; records its calls."""

    def __init__(self):
        self.calls = []

    def to_sparse_params(self, pretrained_config=None, layer_idx=None):
        self.calls.append((pretrained_config, layer_idx))
        offset = pretrained_config.index_skip_topk_offset
        full = max(layer_idx - offset + 1, 0) % pretrained_config.index_topk_freq == 0
        return types.SimpleNamespace(is_full_indexer_layer=full)


def _model_config(sparse=None):
    return types.SimpleNamespace(
        sparse_attention_config=sparse or _Sparse(),
        pretrained_config=types.SimpleNamespace(**GLM52),
    )


class _Hop:
    """Stands in for ``DeferredNormHop.__init__``: resolves the next layer like it."""

    def __init__(self, scope, post_moe_hop, arg=None):
        self.kind = scope.next_layer_kind(post_moe_hop, arg)


class _DecoderLayer:
    """The order of ``GlmDecoderLayer._adopt_post_moe``: post_moe_hop first, then deferred_norm."""

    def __init__(self, scope, layer_idx, model_config, post_moe_hop):
        self.model_config = model_config
        self.layer_idx = layer_idx
        self.post_moe_hop = post_moe_hop
        self.deferred_norm = _Hop(scope, self.post_moe_hop)


def test_the_scope_of_this_target(scope):
    assert scope.SCOPES == ("all", "shared") and scope.DEFAULT_SCOPE == "shared"
    assert scope.UNRESTRICTED_SCOPE == "all"
    assert scope.SCOPE_LABEL == "TRTLLM_STAIRCASE_GLM_LL_DEFERRED_NORM_SCOPE"
    assert scope.selected_scope() == "shared"


def test_the_record_names_the_scope_of_this_target(scope):
    """The record's deferred_norm variant is the scope the hop runs, whatever the environment
    holds (``resolve`` refuses the variable)."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    except (ImportError, OSError):
        dispatch = _load_dispatch_standalone("dispatch_standalone")
    assert dispatch.NUMERICAL_VARIANTS["deferred_norm"] == (None, (scope.DEFAULT_SCOPE,))
    for raw in (None, "", " ", "SHARED", " shared ", "all", "bogus"):
        env = {} if raw is None else {scope.SCOPE_LABEL: raw}
        assert dispatch.feature_variants(["deferred_norm"], env) == {
            "deferred_norm": scope.DEFAULT_SCOPE
        }


def test_glm52_frequency_rule_boundaries(scope):
    cfg = _model_config()
    full = {i for i in range(78) if scope.is_full_indexer_layer(cfg, i)}
    assert full == {0, 1, 2} | set(FULL_NEXT)
    into_full = [b for b in BOUNDARIES if scope.is_full_indexer_layer(cfg, b + 1)]
    assert into_full == [n - 1 for n in FULL_NEXT] and len(into_full) == 18
    assert len(BOUNDARIES) - len(into_full) == 56


def test_unknown_layout(scope):
    assert (
        scope.is_full_indexer_layer(types.SimpleNamespace(sparse_attention_config=None), 6) is None
    )
    params = types.SimpleNamespace(to_sparse_params=lambda **kw: types.SimpleNamespace())
    assert (
        scope.is_full_indexer_layer(
            types.SimpleNamespace(sparse_attention_config=params, pretrained_config=None), 6
        )
        is None
    )


@pytest.mark.parametrize("layer_idx", [3, 5, 6, 9, 73, 75, 76])
def test_next_layer_from_the_constructing_layer(scope, layer_idx):
    hop_stub = object()
    cfg = _model_config()
    layer = _DecoderLayer(scope, layer_idx, cfg, hop_stub)
    full, nxt, how = layer.deferred_norm.kind
    assert nxt == layer_idx + 1 and how == "model config"
    assert full is ((layer_idx + 1) in FULL_NEXT)
    # The same call as MLA.__init__ of the next layer.
    assert cfg.sparse_attention_config.calls == [(cfg.pretrained_config, layer_idx + 1)]


def test_constructing_layer_must_hold_this_post_moe_hop(scope):
    hop_stub, other = object(), object()
    unresolved = (None, None, "no constructing decoder layer found")
    assert _Hop(scope, hop_stub).kind == unresolved

    class _OtherHop:
        def __init__(self):
            self.model_config = _model_config()
            self.layer_idx = 5
            self.post_moe_hop = other
            self.deferred_norm = _Hop(scope, hop_stub)

    assert _OtherHop().deferred_norm.kind == unresolved

    class _NoIndex:
        def __init__(self):
            self.post_moe_hop = hop_stub
            self.layer_idx = None
            self.model_config = _model_config()
            self.deferred_norm = _Hop(scope, hop_stub)

    assert _NoIndex().deferred_norm.kind == unresolved


def test_explicit_argument_wins(scope):
    hop_stub = object()

    class _Layer:
        def __init__(self, arg):
            self.model_config = _model_config()
            self.layer_idx = 5  # next layer 6 is full
            self.post_moe_hop = hop_stub
            self.deferred_norm = _Hop(scope, hop_stub, arg)

    assert _Layer(False).deferred_norm.kind == (False, None, "argument")
    assert _Layer(True).deferred_norm.kind == (True, None, "argument")
    assert _Layer(None).deferred_norm.kind == (True, 6, "model config")


def test_unresolved_config(scope):
    hop_stub = object()
    cfg = types.SimpleNamespace(sparse_attention_config=None, pretrained_config=None)
    layer = _DecoderLayer(scope, 5, cfg, hop_stub)
    assert layer.deferred_norm.kind[0] is None and layer.deferred_norm.kind[1] == 6


def test_decline_reason(scope):
    assert scope.decline_reason("all", True, 6, "model config") is None
    assert scope.decline_reason("all", None, None, "x") is None
    assert scope.decline_reason("shared", False, 7, "model config") is None
    why = scope.decline_reason("shared", True, 6, "model config")
    assert "layer 6 is a full-indexer layer" in why and "post_moe_hop hop" in why
    why = scope.decline_reason("shared", None, None, "no constructing decoder layer found")
    assert "unknown" in why and "no constructing decoder layer found" in why


# ---- installed package and checkpoint ------------------------------------------------


def _installed():
    try:
        import torch  # noqa: F401

        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501, F401
            projection,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(f"needs the installed tensorrt_llm ({exc!r})")
    import absorbed_query_fixture

    if not absorbed_query_fixture.checkpoint_available():
        pytest.skip("needs the GLM-5.2 checkpoint")
    return absorbed_query_fixture


_ATTN = {}


def _attention(absorbed_query_fixture, layer):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if "cfg" not in _ATTN:
        _ATTN["cfg"] = absorbed_query_fixture.build_model_config(0)
    if layer not in _ATTN:
        with torch.inference_mode():
            _ATTN[layer] = absorbed_query_fixture.build_native_attention(layer, 0, _ATTN["cfg"])
    return _ATTN["cfg"], _ATTN[layer]


def test_glm52_model_config_layout(scope):
    """The scope's view of the real GLM-5.2 model config (no GPU work)."""
    absorbed_query_fixture = _installed()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        scope as installed,
    )

    cfg = absorbed_query_fixture.build_model_config(0)
    full = [b + 1 for b in BOUNDARIES if installed.is_full_indexer_layer(cfg, b + 1)]
    assert full == FULL_NEXT, full
    assert len(BOUNDARIES) - len(full) == 56


@pytest.mark.parametrize("layer", [6, 7])
def test_scope_layout_matches_the_attention_module(layer):
    """``is_full_indexer_layer`` is true exactly where the native module has an indexer."""
    absorbed_query_fixture = _installed()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        scope as installed,
    )

    cfg, attn = _attention(absorbed_query_fixture, layer)
    has_indexer = attn.mqa.indexer is not None
    assert installed.is_full_indexer_layer(cfg, layer) is has_indexer
    assert has_indexer is (layer in FULL_NEXT)


@pytest.mark.parametrize("layer", [6, 7])
def test_consumer_guard(layer):
    """Under ``shared`` the deferred_norm projection declines full-indexer layers only."""
    absorbed_query_fixture = _installed()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.scope import (  # noqa: E501
        SCOPE_LABEL,
    )

    _, attn = _attention(absorbed_query_fixture, layer)
    ok, reason, kw = projection_eligibility(attn)
    if layer in FULL_NEXT:
        assert not ok and kw is None and f"{SCOPE_LABEL}=shared" in reason, reason
    else:
        assert ok and kw is None, reason
