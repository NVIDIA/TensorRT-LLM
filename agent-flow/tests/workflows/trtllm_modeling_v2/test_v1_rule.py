# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target reads Modeling V1 model code and never imports it: the guide's rule, found by definition."""

from __future__ import annotations

from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.operators import v1_rule

_GUIDE = """# Modeling V2

## Modeling V1 code

A target builds its model from operations. From the rest of the engine,
it uses only runtime interfaces, for example:

* `DecoderModel`, `DecoderModelForCausalLM`, `ModelConfig` and `Mapping`.
* Attention and speculative metadata, and the KV cache managers.

A target does not import, subclass or call Modeling V1 model code: the model
classes in `{models}/`, the layers in `{modules}/`, MLA and the
attention backends in `{attention}/`, and fused MoE in `{moe}/`.
This applies to every path in the target, weight loading included.
"""

_FILES = {
    "models/modeling_utils.py": (
        "def register_auto_model(name):\n    return lambda cls: cls\n\n"
        "class DecoderModel:\n    pass\n\nclass DecoderModelForCausalLM:\n    pass\n"
    ),
    "models/modeling_toy.py": (
        "from ..modules.head import LMHead\n"
        "from .modeling_utils import DecoderModelForCausalLM, register_auto_model\n\n"
        "def remap_toy_keys(weights):\n    return weights\n\n"
        "class ToyWeightLoader:\n    pass\n\n"
        '@register_auto_model("ToyForCausalLM")\n'
        "class ToyForCausalLM(DecoderModelForCausalLM):\n    pass\n"
    ),
    # A registration above another decorator that spans lines, as the engine writes a
    # model with an input processor; and one in a docstring, which registers nothing.
    "models/modeling_toy_vision.py": (
        "from .modeling_utils import register_auto_model\n\n"
        '@register_auto_model("ToyForConditionalGeneration")\n'
        '@register_input_processor(\n    ToyProcessor,\n    model_type="toy",\n)\n'
        "class ToyForConditionalGeneration:\n"
        '    """Use it as::\n\n        @register_auto_model("SomeVLModel")\n'
        '        class SomeVLModel: ...\n    """\n'
    ),
    "modules/linear.py": (
        "from torch import nn\n\nclass Linear(nn.Module):\n    pass\n\n"
        "class WeightMode:\n    pass\n\ndef split(x):\n    return x\n"
    ),
    "modules/__init__.py": "from .linear import Linear\n",
    # A layer on a layer, a container of layers, and the base classes' module handing a layer on.
    "modules/head.py": (
        "from torch import nn\nfrom .linear import Linear\n\n"
        "class LMHead(Linear):\n    pass\n\n"
        "class Stack(nn.ModuleList):\n    pass\n"
    ),
    "attention/backends/interface.py": (
        "class AttentionMetadata:\n    pass\n\nclass AttentionBackend:\n    pass\n"
    ),
    "attention/backends/trtllm.py": (
        "from .interface import AttentionBackend, AttentionMetadata\n\n"
        "class TrtllmAttentionMetadata(AttentionMetadata):\n    pass\n\n"
        "class TrtllmAttention(AttentionBackend[TrtllmAttentionMetadata]):\n    pass\n\n"
        "class PagedKVCacheManager:\n    pass\n"
    ),
    # A fused MoE implementation on another, and a name bound to one.
    "moe/fused_moe.py": (
        "class MoE:\n    pass\n\nclass CutlassFusedMoEBase(MoE):\n    pass\n\n"
        "class TunedFusedMoE(CutlassFusedMoEBase):\n    pass\n\n"
        "CutlassFusedMoE = CutlassFusedMoEBase\n"
    ),
    "speculative/interface.py": "class SpecMetadata:\n    pass\n",
    "kernels/thop.py": "def thop_decode(x):\n    return x\n",
}

_TARGET = """import torch
from typing import TYPE_CHECKING
from engine.core.attention.backends.interface import AttentionMetadata
from engine.core.attention.backends.trtllm import (
    PagedKVCacheManager,
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from engine.core.kernels.thop import thop_decode
from engine.core.models.modeling_toy import ToyForCausalLM, ToyWeightLoader, remap_toy_keys
from engine.core.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from engine.core.modules import Linear
from engine.core.modules.linear import WeightMode, split
from engine.core.models.modeling_toy import LMHead as Head
from engine.core.modules.head import Stack
from engine.core.moe.fused_moe import CutlassFusedMoE, TunedFusedMoE
from engine.core.speculative.interface import SpecMetadata

from . import weights


if TYPE_CHECKING:
    from engine.core.modules.linear import Linear as OnlyTyped


def forward():
    from .....modules import linear
    from engine.core.kernels import thop
    return linear, thop
"""


def _checkout(tmp_path: Path, core: str = "engine/core") -> Path:
    """A checkout whose engine keeps its code under ``core``: no directory is the real one's."""
    repo = tmp_path / "repo"
    for rel, source in _FILES.items():
        path = repo / core / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    top = core.split("/")[-1]
    guide = repo / core / "v2" / "README.md"
    guide.parent.mkdir(parents=True)
    guide.write_text(
        _GUIDE.format(
            models=f"{top}/models", modules=f"{top}/modules",
            attention=f"{top}/attention", moe=f"{top}/moe",
        ),
        encoding="utf-8",
    )  # fmt: skip
    target = repo / core / "v2" / "models" / "toy" / "toy_1b"
    target.mkdir(parents=True)
    (target / "modeling.py").write_text(_TARGET, encoding="utf-8")
    (target / "weights.py").write_text("WEIGHTS = {}\n", encoding="utf-8")
    return repo


def test_the_rule_is_read_from_the_guide_and_from_what_registers_a_model(tmp_path):
    """The places are the guide's own, resolved against where the guide is; nothing is remembered."""
    repo = _checkout(tmp_path)
    found = v1_rule.rule(repo)
    assert found.guide == "engine/core/v2/README.md"
    assert found.sentence.startswith("does not import, subclass or call Modeling V1 model code:")
    assert found.listed == ("core/models/", "core/modules/", "core/attention/", "core/moe/")
    assert found.places == (
        "engine/core/models", "engine/core/modules", "engine/core/attention", "engine/core/moe",
    )  # fmt: skip
    assert found.interfaces == ("DecoderModel", "DecoderModelForCausalLM", "ModelConfig", "Mapping")
    # A model's module is one that registers a class for an architecture: the base classes'
    # module defines the decorator and registers nothing.
    assert found.model_modules == (
        "engine/core/models/modeling_toy.py",
        "engine/core/models/modeling_toy_vision.py",
    )
    vision = (repo / "engine/core/models/modeling_toy_vision.py").read_text()
    assert v1_rule.registered_classes(vision) == [
        ("ToyForConditionalGeneration", "ToyForConditionalGeneration")
    ]
    # The same engine under another directory: the rule follows the guide.
    moved = v1_rule.rule(_checkout(tmp_path / "moved", core="src/trt"))
    assert moved.places[0] == "src/trt/models"
    assert moved.model_modules[0] == "src/trt/models/modeling_toy.py"
    # A checkout whose guide states no such rule has none, and no import is judged.
    (repo / "engine/core/v2/README.md").write_text("# Modeling V2\n\nNo rule.\n")
    assert v1_rule.rule(repo) is None
    assert v1_rule.v1_imports(repo, "engine/core/v2/models/toy/toy_1b/modeling.py") == []
    assert v1_rule.rule(tmp_path / "nowhere") is None


def test_an_import_is_judged_by_what_the_imported_name_is(tmp_path):
    """Runtime interfaces pass; a model class, its module, a backend, fused MoE and a layer do not."""
    repo = _checkout(tmp_path)
    path = "engine/core/v2/models/toy/toy_1b/modeling.py"
    found = {item.name: item for item in v1_rule.v1_imports(repo, path)}
    # What the guide lets a target use from the rest of the engine, and what is in no listed place.
    for allowed in ("AttentionMetadata", "TrtllmAttentionMetadata", "PagedKVCacheManager",
                    "DecoderModel", "DecoderModelForCausalLM", "SpecMetadata", "thop_decode",
                    "weights"):  # fmt: skip
        assert allowed not in found, allowed
    assert {name: (item.what, item.certain) for name, item in found.items()} == {
        "ToyForCausalLM": (v1_rule.MODEL_CLASS, True),
        # A model's weight loader and helpers are that model's code: weight loading included.
        "ToyWeightLoader": (v1_rule.MODEL_MODULE, True),
        "TrtllmAttention": (v1_rule.ATTENTION_BACKEND, True),
        "Linear": (v1_rule.LAYER, True),
        # A class is what its ancestors are: a layer on a layer, a container of layers, a
        # fused MoE on another, and a name bound to one. A name a module only hands on is
        # judged where it is defined.
        "LMHead": (v1_rule.LAYER, True),
        "Stack": (v1_rule.LAYER, True),
        "TunedFusedMoE": (v1_rule.MOE_MODULE, True),
        "CutlassFusedMoE": (v1_rule.MOE_MODULE, True),
        # A module imported whole is what it defines.
        "linear": (v1_rule.LAYER, True),
        # The guide lists the place, not such a thing: reported, and left to a person. A
        # function of a model's module may be a kernel or an op, which a target may call.
        "remap_toy_keys": (v1_rule.LISTED_PLACE, False),
        "WeightMode": (v1_rule.LISTED_PLACE, False),
        "split": (v1_rule.LISTED_PLACE, False),
    }
    assert found["LMHead"].path == "engine/core/modules/head.py"
    # What only a type checker reads is not an import of the target.
    assert "OnlyTyped" not in {item.name for item in v1_rule.v1_imports(repo, path)}
    # A name a package re-exports is judged where it is defined.
    assert found["Linear"].path == "engine/core/modules/linear.py"
    assert found["Linear"].render() == (
        f"line {found['Linear'].line}: `Linear` from `engine.core.modules` is a Modeling V1 layer"
    )
    # An import inside a function is an import, and a relative one is resolved from the file.
    assert found["linear"].module == ".....modules"
    # A commit's version of a file is judged as given; one that does not parse has no finding.
    text = "from engine.core.modules.linear import Linear\n"
    (one,) = v1_rule.v1_imports(repo, "engine/core/v2/models/toy/toy_1b/new.py", text=text)
    assert (one.line, one.name, one.what) == (1, "Linear", v1_rule.LAYER)
    assert v1_rule.v1_imports(repo, path, text="def broken(:\n") == []
    # Every name of a module, a registration reached through a module, and a name no file
    # system holds: judged, or passed over, and never an exception.
    star = v1_rule.v1_imports(repo, path, text="from engine.core.modules.linear import *\n")
    assert [(i.name, i.what, i.certain) for i in star] == [("*", v1_rule.LAYER, True)]
    assert v1_rule.v1_imports(repo, path, text=f"import {'x' * 300}.y\n") == []
    assert v1_rule.registered_classes(
        "@zoo.register_auto_model(Config.architecture)\nclass Lazy:\n    pass\n"
    ) == [("Config.architecture", "Lazy")]
