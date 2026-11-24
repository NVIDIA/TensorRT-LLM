# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image prompts on the specialized Kimi K3.1 Rubin targets, on the CPU.

dep16 and tep8 build the MoonViT tower and its projector (their ``w4_rows``) on a server that can
run a context step and is not text-only (``disable_mm_encoder``); helix8 builds none. Checked here:

* the weight tables: the vision rows are the export's 168 tower and projector tensors (names,
  shapes, BF16) where the tower is built, and nothing changes where it is not (the table, the
  leftovers); ``load`` reads them and leaves no vision key; ``media_proc_cfg`` reads the
  checkpoint processor's limits;
* the vision section and the shell hooks of each target's ``modeling.py``, run from their source
  with the GPU entries stood in by their ``reference``: the flat ``vision_encode`` equals the
  generic target's ``K31VisionTower.encode`` (also run from its source) bit for bit; a request
  past the encoder budget is encoded in groups within it; the shell's forward hands the core the
  ids and ``inputs_embeds=None`` on a text step and the spliced embeddings on an image step (the
  worker keeps the ids); the KV-capacity estimate's text context step holds one largest image;
  admission refuses what the tower cannot encode and media without a tower unless the request is
  generation-only (helix8 and the ULL generation target helix16_moetp16ep1 too).

The same comparisons on the GPU entries, with the checkpoint's own weights and images, are in
``test_rubin_vision_tower.py``.
"""

from __future__ import annotations
import __future__

import ast
import copy
import functools
import importlib
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
    KimiK31TextConfig,
    KimiK31VisionConfig,
)
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only

_FAMILY_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_GENERIC = _FAMILY / "kimi_k31_pro_mxfp4__sm_10x__default/modeling.py"
_INPUT_PROCESSOR = _FAMILY / "input_processor.py"
_TOWER_TARGETS = ("dep16", "tep8")
_LAYOUTS = {
    "dep16": dict(
        world_size=16, tp_size=16, moe_tp_size=1, moe_ep_size=16, enable_attention_dp=True
    ),
    "tep8": dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
    "helix8": dict(
        world_size=8,
        tp_size=1,
        cp_size=8,
        cp_config={"cp_type": CpType.HELIX},
        moe_tp_size=2,
        moe_ep_size=4,
    ),
}

# --- the export's tower, as its safetensors index and shard headers state it (rev 1d82032) ---

_HIDDEN, _QKV, _MLP, _BLOCKS, _TEXT = 1024, 1536, 4096, 27, 7168
_EXPORT_TOWER = {
    "vision_tower.patch_embed.proj.weight": (_HIDDEN, 3, 14, 14),
    "vision_tower.patch_embed.pos_emb.weight": (64, 64, _HIDDEN),
    "vision_tower.encoder.final_layernorm.weight": (_HIDDEN,),
    "mm_projector.proj.0.weight": (4 * _HIDDEN, 4 * _HIDDEN),
    "mm_projector.proj.2.weight": (_TEXT, 4 * _HIDDEN),
    "mm_projector.post_norm.weight": (_TEXT,),
    **{
        f"vision_tower.encoder.blocks.{j}.{name}": shape
        for j in range(_BLOCKS)
        for name, shape in (
            ("norm0.weight", (_HIDDEN,)),
            ("norm1.weight", (_HIDDEN,)),
            ("wqkv.weight", (3 * _QKV, _HIDDEN)),
            ("wo.weight", (_HIDDEN, _QKV)),
            ("mlp.fc0.weight", (_MLP, _HIDDEN)),
            ("mlp.fc1.weight", (_HIDDEN, _MLP)),
        )
    },
}
#: K3.1's checkpoint processor limits (``preprocessor_config.json`` ``media_proc_cfg``).
_K31_MEDIA = {
    "in_patch_limit": 65536,
    "patch_size": 14,
    "merge_kernel_size": 2,
    "patch_limit_on_one_side": 512,
}

# The text config the weight tables read (the export's, trimmed to what they read).
_MLA = (*range(3, 88, 4), 91, 92)
_TEXT_CONFIG = dict(
    hidden_size=_TEXT,
    num_hidden_layers=93,
    vocab_size=163840,
    num_attention_heads=96,
    num_key_value_heads=96,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    intermediate_size=33792,
    moe_intermediate_size=3072,
    num_experts=896,
    num_experts_per_token=16,
    num_shared_experts=2,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    routed_expert_hidden_size=3584,
    attn_res_block_size=12,
    rms_norm_eps=1e-5,
    linear_attn_config={
        "full_attn_layers": [i + 1 for i in _MLA],
        "kda_layers": [i + 1 for i in range(93) if i not in _MLA],
        "head_dim": 128,
        "num_heads": 96,
        "short_conv_kernel_size": 4,
    },
    dsa_config={
        "dsa_layers": [i + 1 for i in _MLA],
        "index_head_dim": 128,
        "index_k_norm_type": "layernorm",
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_skip_topk_offset": 1,
        "index_topk": 2048,
        "index_topk_freq": 1,
    },
)


def _weights(layout: str):
    return importlib.import_module(f"{_FAMILY_PKG}.kimi_k31_pro_nvfp4__sm_107__{layout}.weights")


def _defs(tree: ast.Module, names: set[str]) -> list[ast.stmt]:
    """The module-level definitions ``names`` of ``tree`` in file order, decorators dropped."""
    out, seen = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            out.append(node)
            seen.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found = {t.id for t in targets if isinstance(t, ast.Name)} & names
            if found:
                out.append(copy.deepcopy(node))
                seen |= found
    assert not names - seen, f"{sorted(names - seen)} not found"
    return out


def _exec(nodes: list[ast.stmt], ns: dict, filename: str) -> dict:
    """Run ``nodes`` in ``ns`` as their file runs them (annotations postponed)."""
    module = ast.fix_missing_locations(ast.Module(nodes, []))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, filename, "exec", flags=flags, dont_inherit=True), ns)
    return ns


@functools.lru_cache(maxsize=None)
def _geometry() -> dict:
    """The input processor's grid geometry (``moonvit_max_image_grid``) and placeholder id, from
    its source (its module imports the engine's input stack, which this CPU stage does not load)."""
    tree = ast.parse(_INPUT_PROCESSOR.read_text())
    names = {
        "MEDIA_PLACEHOLDER_TOKEN_ID",
        "_DEFAULT_IN_PATCH_LIMIT",
        "_DEFAULT_PATCH_LIMIT_ON_ONE_SIDE",
        "_navit_max_patches",
        "_MoonViTGeometry",
        "_moonvit_geometry",
        "_max_image_patches",
        "_largest_grid",
        "moonvit_max_image_grid",
    }
    nodes = _defs(tree, names)
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name == "_navit_max_patches":
            node.decorator_list = [ast.parse("functools.lru_cache(maxsize=16)", mode="eval").body]
    ns = dict(
        np=np,
        functools=functools,
        NamedTuple=NamedTuple,
        Dict=Dict,
        Any=Any,
        Optional=Optional,
        Tuple=Tuple,
    )
    return _exec(nodes, ns, str(_INPUT_PROCESSOR))


def moonvit_max_image_grid(media_proc_cfg, merge_kernel_size, vision_patch_size):
    return _geometry()["moonvit_max_image_grid"](
        media_proc_cfg, merge_kernel_size, vision_patch_size
    )


MEDIA_PLACEHOLDER_TOKEN_ID = 163605


def _core(layout: str, *, vision: bool, text_only: bool = False, role: str = "context"):
    """A core as ``weights.dims`` reads it: the role, the model config, the composite's vision
    config (``vision``) and ``disable_mm_encoder`` (``text_only``)."""
    core = SimpleNamespace(
        role=role,
        vision_config=KimiK31VisionConfig() if vision else None,
        model_config=SimpleNamespace(
            pretrained_config=KimiK31TextConfig(**_TEXT_CONFIG),
            mapping=Mapping(rank=0, **copy.deepcopy(_LAYOUTS[layout])),
            torch_dtype=torch.bfloat16,
            disable_mm_encoder=text_only,
        ),
    )
    core.dims = _weights(layout).dims(core)
    return core


# --- the weight tables ---


@pytest.mark.parametrize("role", ["context", "generation"])
@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_the_vision_rows_are_the_exports_tower(layout, role):
    """Both roles hold the tower: each export tensor once, whole, BF16, into its own ``v_*``."""
    W = _weights(layout)
    d = _core(layout, vision=True, role=role).dims
    assert d.vision
    rows = W.w4_rows(d)
    sources = {}
    for row in rows:
        assert row.layer_indices(d) is None and row.cut is None and row.transform is None, row.name
        assert row.dtype == torch.bfloat16, row.name
        assert row.name.startswith("v") and not row.name.startswith(W.DRAFT_KEY_PREFIX), row.name
        [(key, index)] = row.sources(d, None)
        assert index is None and key.startswith(W.CKPT_VISION)
        sources[key] = tuple(row.shape(d))
    assert sources == _EXPORT_TOWER
    assert len({row.name for row in rows}) == len(rows) == 168
    assert {row.name for row in rows} <= {row.name for row in W.table(d)}


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_a_text_only_server_keeps_todays_table(layout):
    """Without the composite's vision config, or with ``disable_mm_encoder``, the table and the
    leftovers are those of the text-only target: no vision row, every vision key a leftover."""
    W = _weights(layout)
    weights = dict.fromkeys(_EXPORT_TOWER)
    for core in (_core(layout, vision=False), _core(layout, vision=True, text_only=True)):
        d = core.dims
        assert not d.vision and W.w4_rows(d) == ()
        names = [row.name for row in W.table(d)]
        assert names == [row.name for row in (*W.w1_rows(d), *W.w2_rows(d), *W.w3_rows(d))]
        experts = W.w2_offwindow_keys(core, weights)
        assert W.expected_unconsumed(core, weights) - experts == set(_EXPORT_TOWER)


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_the_tower_rows_read_as_declared(layout):
    """Each vision row reads its key whole into a parameter of the declared shape and dtype (meta
    tensors: nothing is allocated); with the tower, no vision key is a leftover."""
    W = _weights(layout)
    core = _core(layout, vision=True)
    rows = W.w4_rows(core.dims)
    with torch.device("meta"):
        params = {f"w.{k}": v for k, v in W.declare(core.dims, rows).items()}
    ckpt = {
        k: torch.empty(s, dtype=torch.bfloat16, device="meta") for k, s in _EXPORT_TOWER.items()
    }
    consumed = W.fill_table(core, ckpt, rows, params)
    assert consumed == set(_EXPORT_TOWER)
    assert not W.expected_unconsumed(core, ckpt) & set(_EXPORT_TOWER)


def test_helix8_builds_no_tower():
    """helix8 serves generation steps only: its table has no vision rows and every load leaves
    the tower."""
    W = _weights("helix8")
    assert not hasattr(W, "w4_rows") and "vision" not in W.Dims._fields
    core = _core("helix8", vision=True, role="generation")
    keys = dict.fromkeys(_EXPORT_TOWER)
    assert W.expected_unconsumed(core, keys) - W.w2_offwindow_keys(core, keys) == set(_EXPORT_TOWER)


class _Lazy:
    """Stands in for a safetensors slice."""

    def __init__(self, full: torch.Tensor):
        self._full = full

    def get_shape(self) -> list[int]:
        return list(self._full.shape)

    def __getitem__(self, index):
        return self._full[index]


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_load_reads_the_tower_where_it_is_built(layout, monkeypatch):
    """``load`` fills the vision rows with the text rows and leaves no vision key; without the
    tower the same checkpoint leaves exactly the vision keys (tiny tables stand in for the model)."""
    W = _weights(layout)
    monkeypatch.setattr(W, "check_checkpoint", lambda model_config: None)
    monkeypatch.setattr(W, "w2_offwindow_keys", lambda core, weights: set())
    text_row = W.W("embed", lambda d: (2, 4), "t.embed", layers=None)
    tower_rows = (
        W.W("v_a", lambda d: (3,), "vision_tower.a.weight", dtype=torch.bfloat16, layers=None),
        W.W("v_b", lambda d: (2, 2), "mm_projector.b.weight", dtype=torch.bfloat16, layers=None),
    )
    monkeypatch.setattr(W, "w1_rows", lambda d: (text_row,))
    monkeypatch.setattr(W, "w2_rows", lambda d: ())
    monkeypatch.setattr(W, "w4_rows", lambda d: tower_rows if d.vision else ())
    g = torch.Generator().manual_seed(0)
    full = {
        "t.embed": torch.randn(2, 4, generator=g).bfloat16(),
        "vision_tower.a.weight": torch.randn(3, generator=g).bfloat16(),
        "mm_projector.b.weight": torch.randn(2, 2, generator=g).bfloat16(),
        W.CKPT_LM_HEAD: torch.randn(6, 4, generator=g).bfloat16(),
    }
    for vision in (True, False):
        d = W.Dims(*([0] * len(W.Dims._fields)))._replace(
            num_layers=1, hidden=4, vocab=6, dtype=torch.bfloat16, kda_tp=1, vision=vision
        )

        class _Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.dims = d
                self.model_config = None
                self.w = W.declare(d, W.table(d))

        class _Shell(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Core()
                self.lm_head = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)

        monkeypatch.setattr(W, "w3_rows", lambda d: ())
        shell = _Shell()
        ckpt = {k: _Lazy(v) for k, v in full.items()}
        W.load(shell, ckpt)
        if vision:
            assert torch.equal(shell.model.w["v_a"], full["vision_tower.a.weight"])
            assert torch.equal(shell.model.w["v_b"], full["mm_projector.b.weight"])
        else:
            assert set(dict(shell.model.w.named_parameters())) == {"embed"}


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_media_proc_cfg_reads_the_checkpoint_processor(layout, tmp_path):
    W = _weights(layout)
    weights = SimpleNamespace(checkpoint_dir=str(tmp_path))
    assert W.media_proc_cfg(weights) is None
    assert W.media_proc_cfg(SimpleNamespace()) is None
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"auto_map": {}}))
    assert W.media_proc_cfg(weights) == {}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"media_proc_cfg": _K31_MEDIA}))
    assert W.media_proc_cfg(weights) == _K31_MEDIA


# --- the vision section and the shell hooks, run from each target's source ---

#: The vision section's definitions and the shell's hooks each test reads off a target's source.
_VISION_DEFS = {
    "_K31_VISION",
    "_VISION_HEADS",
    "_VISION_HEAD_DIM",
    "_VISION_MERGE",
    "_VISION_PATCH",
    "_VISION_FRAMES",
    "_VISION_NO_TOWER",
    "_VISION_DEVICE_PATHS",
    "_VisionBlockW",
    "_VisionW",
    "_check_vision",
    "_vision_time_rows",
    "_vision_view",
    "_vision_attention",
    "vision_encode",
    "_vision_groups",
    "vision_forward",
    "image_inputs_embeds",
    "vision_profiling_rows",
    "vision_admit",
}
_SHELL_METHODS = ("forward", "validate_request", "load_weights")
_SHELL_PROPERTIES = ("multimodal_token_ids", "multimodal_data_device_paths")


def _source(layout: str) -> Path:
    return _FAMILY / f"kimi_k31_pro_nvfp4__sm_107__{layout}/modeling.py"


def _shell(tree: ast.Module) -> ast.ClassDef:
    return next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any("DecoderModelForCausalLM" in ast.unparse(b) for b in n.bases)
    )


def _attention_reference(
    qkv, out, seg_lens, max_seq_len, max_num_requests, workspace, num_heads, head_dim
):
    """The MoonViT attention entry's ``reference`` (fp32 softmax per image and head), into ``out``."""
    width = num_heads * head_dim
    rows, start = [], 0
    for n in seg_lens:
        q, k, v = (
            qkv[start : start + n, i * width : (i + 1) * width]
            .float()
            .view(n, num_heads, head_dim)
            .transpose(0, 1)
            for i in range(3)
        )
        scores = torch.matmul(q, k.transpose(1, 2)) * head_dim**-0.5
        rows.append(
            torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1).reshape(n, width)
        )
        start += n
    out.copy_(torch.cat(rows).to(qkv.dtype))
    return out


def _cpu_entries() -> dict:
    """The entries the tower calls, on CPU: the family's own where they import here, the attention
    and the BF16 linear by their references (the GPU test runs the real ones)."""
    rope = importlib.import_module(f"{_FAMILY_PKG}.ops.vision.moonvit_rope_2d")
    patch = importlib.import_module(f"{_FAMILY_PKG}.ops.vision.moonvit_patch_embed")
    norm = importlib.import_module(f"{_FAMILY_PKG}.ops.vision.moonvit_rms_norm")
    gelu = importlib.import_module(f"{_FAMILY_PKG}.ops.vision.moonvit_gelu")
    return dict(
        TABLE_SIDE=rope.TABLE_SIDE,
        THETA=rope.THETA,
        moonvit_rope_2d=rope.moonvit_rope_2d,
        moonvit_rope_angles=rope.moonvit_rope_angles,
        moonvit_rope_table=rope.moonvit_rope_table,
        moonvit_patch_embed=patch.moonvit_patch_embed,
        moonvit_rms_norm=norm.moonvit_rms_norm,
        moonvit_gelu=gelu.moonvit_gelu,
        moonvit_attention=_attention_reference,
        bf16_linear=lambda x, weight, cublas: F.linear(x, weight),
    )


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args, **kwargs):
        self.lines.append(msg)

    info_once = warning = info


def _namespace(layout: str, **overrides) -> dict:
    """The target's vision section and shell hooks, executed with CPU stand-ins."""
    tree = ast.parse(_source(layout).read_text())
    shell = _shell(tree)
    methods = [
        copy.deepcopy(n)
        for n in shell.body
        if isinstance(n, ast.FunctionDef) and n.name in _SHELL_METHODS + _SHELL_PROPERTIES
    ]
    for m in methods:
        m.decorator_list = []
    ns = dict(
        torch=torch,
        nn=nn,
        np=np,
        Any=Any,
        NamedTuple=NamedTuple,
        MappingProxyType=MappingProxyType,
        AttentionMetadata=Any,
        MEDIA_PLACEHOLDER_TOKEN_ID=MEDIA_PLACEHOLDER_TOKEN_ID,
        moonvit_max_image_grid=moonvit_max_image_grid,
        logger=_Log(),
        _TARGET=f"kimi_k31_pro_nvfp4__sm_107__{layout}",
        get_multimodal_embeddings=None,
        find_input_mm_embeds=None,
        _weights=None,
        **_cpu_entries(),
    )
    ns.update(overrides)
    return _exec(_defs(tree, _VISION_DEFS) + methods, ns, str(_source(layout)))


def _generic_tower_namespace() -> dict:
    """The generic target's ``K31VisionTower`` from its source, with the same CPU stand-ins."""
    tree = ast.parse(_GENERIC.read_text())
    names = {"_SUPPORTED_VARIANT", "_config_dict", "_sincos_rows", "_Block", "K31VisionTower"}
    ns = dict(
        torch=torch,
        nn=nn,
        np=np,
        Any=Any,
        Dict=dict,
        List=list,
        Tuple=tuple,
        Optional=Any,
        MultimodalParams=Any,
        moonvit_max_image_grid=moonvit_max_image_grid,
        logger=_Log(),
        VISION_TOWER_PREFIX="vision_tower.",
        PROJECTOR_PREFIX="mm_projector.",
        **_cpu_entries(),
    )
    return _exec(_defs(tree, names), ns, str(_GENERIC))


def _random_tower(seed: int = 0) -> dict:
    """Random tower weights at the export's shapes, under the checkpoint's names."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for key, shape in _EXPORT_TOWER.items():
        if key.endswith(
            ("norm0.weight", "norm1.weight", "final_layernorm.weight", "post_norm.weight")
        ):
            out[key] = (1.0 + 0.1 * torch.randn(shape, generator=gen)).bfloat16()
        elif key.endswith("pos_emb.weight"):
            out[key] = torch.randn(shape, generator=gen).bfloat16()
        else:
            out[key] = (0.02 * torch.randn(shape, generator=gen)).bfloat16()
    return out


def _flat_core(layout: str, ns: dict, tower: dict, media: dict | None):
    """A core holding the tower as ``w4_rows`` parameters, as the target's ``_vision_view`` reads it."""
    W = _weights(layout)
    d = W.Dims(*([0] * len(W.Dims._fields)))._replace(vision=True)
    w = {row.name: tower[row.sources(d, None)[0][0]] for row in W.w4_rows(d)}
    return SimpleNamespace(dims=d, w=w, media_proc_cfg=media)


@pytest.fixture(scope="module")
def tower():
    return _random_tower()


#: The images of each request: resampled position tables and the smallest 2x2 grid (small enough
#: for the CPU references; the GPU test also runs the table's own 64x64 grid).
_REQUEST_GRIDS = [[(1, 4, 6), (1, 2, 2)], [(1, 6, 10)], [(1, 8, 8)]]


def _params(grids_per_request, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for grids in grids_per_request:
        patches = sum(t * h * w for t, h, w in grids)
        image = {
            "pixel_values": torch.randn(patches, 3, 14, 14, generator=gen),
            "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
        }
        out.append(SimpleNamespace(multimodal_data={"image": image}))
    return out


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_the_flat_encode_is_the_generic_towers_bit_for_bit(layout, tower):
    """``vision_forward`` / ``vision_encode`` from the target's source against ``K31VisionTower``
    from the generic target's source, on the same weights, images and entries: equal rows."""
    ns = _namespace(layout)
    generic = _generic_tower_namespace()["K31VisionTower"](
        dict(KimiK31VisionConfig().to_dict()),
        _TEXT,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert generic.load_weights(tower)
    flat = ns["_vision_view"](_flat_core(layout, ns, tower, {}))
    assert flat.max_grid == generic.max_image_grid and flat.capacity == generic.attention_capacity
    assert torch.equal(flat.time_weight, generic.time_weight)
    params = _params(_REQUEST_GRIDS)
    [expected] = generic.forward(params)
    [rows] = ns["vision_forward"](flat, params)
    assert rows.dtype == torch.bfloat16 and rows.shape == expected.shape
    assert torch.isfinite(rows.float()).all()
    assert torch.equal(rows, expected)


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_the_largest_grid_is_the_checkpoint_processors(layout, tower):
    ns = _namespace(layout)
    flat = ns["_vision_view"](_flat_core(layout, ns, tower, _K31_MEDIA))
    assert flat.max_grid == moonvit_max_image_grid(_K31_MEDIA, (2, 2), 14) == (1, 220, 306)
    assert flat.capacity == (67320, 16830)
    missing = ns["_vision_view"](_flat_core(layout, ns, tower, None))
    assert missing.max_grid == moonvit_max_image_grid({}, (2, 2), 14)
    assert any("no preprocessor_config.json" in line for line in ns["logger"].lines)


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_a_request_past_the_budget_is_encoded_in_groups(layout, tower):
    """Grids are grouped in order within the budget; each group is one encode, so the rows are
    those of encoding each group alone."""
    ns = _namespace(layout)
    groups = ns["_vision_groups"]
    assert groups([(1, 2, 2), (1, 4, 4)], 100) == [[(1, 2, 2), (1, 4, 4)]]
    assert groups([(1, 8, 8), (1, 6, 6), (1, 4, 4), (1, 8, 8), (1, 10, 10)], 100) == [
        [(1, 8, 8), (1, 6, 6)],
        [(1, 4, 4), (1, 8, 8)],
        [(1, 10, 10)],
    ]
    flat = ns["_vision_view"](_flat_core(layout, ns, tower, {}))._replace(capacity=(40, 10))
    grids = [(1, 4, 6), (1, 2, 2), (1, 4, 4), (1, 6, 6)]
    [param] = _params([grids], seed=3)
    [rows] = ns["vision_forward"](flat, [param])
    pixels = param.multimodal_data["image"]["pixel_values"]
    expected = torch.cat(
        [
            ns["vision_encode"](flat, pixels[0:28], [(1, 4, 6), (1, 2, 2)]),
            ns["vision_encode"](flat, pixels[28:44], [(1, 4, 4)]),
            ns["vision_encode"](flat, pixels[44:80], [(1, 6, 6)]),
        ]
    )
    assert torch.equal(rows, expected)


def test_the_vision_config_is_checked():
    ns = _namespace("dep16")
    ns["_check_vision"](KimiK31VisionConfig())
    ns["_check_vision"](KimiK31VisionConfig().to_dict())
    with pytest.raises(AssertionError, match="vt_num_hidden_layers"):
        ns["_check_vision"](KimiK31VisionConfig(vt_num_hidden_layers=26))
    with pytest.raises(AssertionError, match="merge_type"):
        ns["_check_vision"](KimiK31VisionConfig(merge_type="sd2"))


# --- the shell's forward, on stand-ins ---

_PLACEHOLDER = MEDIA_PLACEHOLDER_TOKEN_ID


class _Core:
    """Stands in for the core: records what the shell hands it."""

    def __init__(self, vision, embed):
        self._vision = vision
        self._head = SimpleNamespace(embed=embed)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(kwargs["input_ids"].shape[0], 4)


class _Worker:
    def __init__(self):
        self.calls = []

    def target_head_rows(self, hidden, attn_metadata, spec_metadata):
        return hidden

    def target_logits(self, rows, *args):
        return rows

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["logits"]


def _shell_stub(ns: dict, vision) -> SimpleNamespace:
    embed = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)
    stub = SimpleNamespace(
        model=_Core(vision, embed),
        spec_worker=_Worker(),
        lm_head=None,
        logits_processor=None,
        draft_model=None,
    )
    stub.forward = lambda *a, **k: ns["forward"](stub, *a, **k)
    return stub


def _metadata(num_contexts: int = 1, estimating: bool = False):
    return SimpleNamespace(
        num_contexts=num_contexts,
        num_tokens=4,
        padded_num_tokens=None,
        kv_cache_manager=SimpleNamespace(is_estimating_kv_cache=estimating),
    )


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_a_text_step_hands_the_core_the_ids_alone(layout):
    ns = _namespace(layout)
    stub = _shell_stub(ns, vision=SimpleNamespace(name="tower"))
    input_ids = torch.tensor([1, 2, 3, 4])
    for kwargs in (
        {},
        {"multimodal_params": []},
        {"multimodal_params": None, "mm_token_indices": None},
    ):
        stub.forward(_metadata(), input_ids=input_ids, position_ids=torch.arange(4)[None], **kwargs)
        call = stub.model.calls[-1]
        assert call["input_ids"] is input_ids and call["inputs_embeds"] is None
        assert stub.spec_worker.calls[-1]["input_ids"] is input_ids


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_an_image_step_splices_the_rows_and_keeps_the_ids(layout):
    """The rows the engine helpers return go over the placeholder positions; every other row is
    the token's embedding; the core and the worker keep the ids."""
    image_rows = torch.full((2, 4), -7.0)
    seen = {}

    def get_embeddings(encoder_forward_fn, multimodal_params):
        seen["params"] = multimodal_params
        return [image_rows]

    ns = _namespace(
        layout,
        get_multimodal_embeddings=get_embeddings,
        find_input_mm_embeds=lambda embeds, params: embeds,
    )
    stub = _shell_stub(ns, vision=SimpleNamespace(name="tower"))
    input_ids = torch.tensor([1, 5, 5, 2])
    params = [SimpleNamespace(name="request 0"), SimpleNamespace(name="generation delta")]
    stub.forward(
        _metadata(num_contexts=1),
        input_ids=input_ids,
        position_ids=torch.arange(4)[None],
        multimodal_params=params,
        mm_token_indices=torch.tensor([1, 2]),
    )
    call = stub.model.calls[-1]
    embeds = call["inputs_embeds"]
    assert call["input_ids"] is input_ids and stub.spec_worker.calls[-1]["input_ids"] is input_ids
    assert seen["params"] == params[:1]
    assert torch.equal(embeds[1:3], image_rows)
    assert torch.equal(embeds[[0, 3]], F.embedding(input_ids[[0, 3]], stub.model._head.embed))


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_an_image_step_without_matching_placeholders_fails(layout):
    ns = _namespace(
        layout,
        get_multimodal_embeddings=lambda encoder_forward_fn, multimodal_params: [torch.zeros(3, 4)],
        find_input_mm_embeds=lambda embeds, params: embeds,
    )
    embed = torch.zeros(6, 4)
    with pytest.raises(ValueError, match="token count mismatch"):
        ns["image_inputs_embeds"](
            object(), embed, torch.tensor([5, 5]), [object()], torch.tensor([0, 1]), 1
        )
    with pytest.raises(ValueError, match="mm_token_indices"):
        ns["image_inputs_embeds"](object(), embed, torch.tensor([5, 5]), [object()], None, 1)
    with pytest.raises(NotImplementedError, match="no vision tower"):
        ns["image_inputs_embeds"](None, embed, torch.tensor([5, 5]), [object()], None, 1)


class _HoldingCore(_Core):
    """A core that records, when it runs, how many profiling encodes are held."""

    def __init__(self, vision, embed, encodes):
        super().__init__(vision, embed)
        self.encodes = encodes
        self.held = []

    def __call__(self, **kwargs):
        self.held.append(len(self.encodes))
        return super().__call__(**kwargs)


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_the_kv_estimate_holds_one_largest_image(layout, tower):
    """Only a text context step of the KV-capacity estimate encodes (one image of the largest
    grid), before the core runs."""
    encodes = []
    ns = _namespace(layout)
    vision = ns["_vision_view"](_flat_core(layout, ns, tower, {}))
    ns["vision_encode"] = lambda V, pixels, grids: encodes.append(
        (tuple(pixels.shape), grids)
    ) or torch.zeros(3, 4)
    input_ids = torch.tensor([1, 2, 3, 4])
    for tower_view, metadata, want in (
        (vision, _metadata(1, estimating=True), 1),
        (vision, _metadata(1, estimating=False), 0),
        (vision, _metadata(0, estimating=True), 0),
        (None, _metadata(1, estimating=True), 0),
    ):
        encodes.clear()
        stub = _shell_stub(ns, vision=tower_view)
        stub.model = _HoldingCore(tower_view, stub.model._head.embed, encodes)
        stub.forward(metadata, input_ids=input_ids, position_ids=torch.arange(4)[None])
        assert stub.model.held == [want]
        if want:
            t, h, w = vision.max_grid
            assert encodes == [((t * h * w, 3, 14, 14), [vision.max_grid])]


# --- admission ---


def _request(grids=((1, 2, 2),), patches=None, generation_only=False, media=True):
    """A request as the admission hook sees it (meta pixels: only their shape is read)."""
    data = None
    if media:
        n = sum(t * h * w for t, h, w in grids) if patches is None else patches
        pixels = torch.empty(n, 3, 14, 14, device="meta")
        data = {"image": {"pixel_values": pixels, "image_grid_thw": torch.tensor(grids)}}
    return SimpleNamespace(py_multimodal_data=data, is_generation_only_request=generation_only)


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_admission(layout, tower):
    ns = _namespace(layout)
    admit = ns["vision_admit"]
    vision = ns["_vision_view"](_flat_core(layout, ns, tower, _K31_MEDIA))
    admit(vision, _request(media=False))
    admit(vision, _request([(1, 220, 306)]))
    admit(vision, _request([(1, 220, 306), (1, 220, 306)]))  # past the budget: encoded in groups
    for grids in ([(5, 2, 2)], [(1, 514, 2)], [(1, 3, 2)], [(1, 2, 5)], [(1, 512, 512)]):
        with pytest.raises(ValueError, match="cannot encode the image grid"):
            admit(vision, _request(grids))
    with pytest.raises(ValueError, match="pixel_values must have shape"):
        admit(vision, _request([(1, 2, 2)], patches=5))
    with pytest.raises(ValueError, match=r"\[num_grids, 3\]"):
        admit(vision, _request([(1, 2)], patches=4))
    # Without a tower: media only on a generation-only request.
    admit(None, _request(media=False))
    admit(None, _request(generation_only=True))
    with pytest.raises(ValueError, match="no vision tower"):
        admit(None, _request())
    # The shell's hook reads the core's tower.
    stub = SimpleNamespace(model=SimpleNamespace(_vision=None))
    with pytest.raises(ValueError, match="no vision tower"):
        ns["validate_request"](stub, _request())


@pytest.mark.parametrize("layout", _TOWER_TARGETS)
def test_the_multimodal_surface(layout):
    ns = _namespace(layout)
    stub = SimpleNamespace(_mm_token_id=_PLACEHOLDER)
    ids = ns["multimodal_token_ids"](stub)
    assert ids.dtype == torch.int32 and ids.tolist() == [163605]
    assert ns["multimodal_data_device_paths"](stub) == [
        "image.pixel_values",
        "image.image_grid_thw",
        "multimodal_embedding",
    ]


#: The generation-only targets: no tower, no vision code; admission takes media only on a
#: generation-only request.
_GENERATION_ONLY_TARGETS = ("helix8", "helix16_moetp16ep1")


@pytest.mark.parametrize("layout", _GENERATION_ONLY_TARGETS)
def test_generation_only_target_admits_media_only_on_generation_only_requests(layout):
    tree = ast.parse(_source(layout).read_text())
    [method] = [
        copy.deepcopy(n)
        for n in _shell(tree).body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_request"
    ]
    name = f"kimi_k31_pro_nvfp4__sm_107__{layout}"
    ns = _exec([method], {"Any": Any, "_TARGET": name, "_TARGET_NAME": name}, layout)
    validate = ns["validate_request"]
    validate(None, _request(media=False))
    validate(None, _request(generation_only=True))
    with pytest.raises(ValueError, match="runs no vision tower"):
        validate(None, _request())


@pytest.mark.parametrize("layout", _GENERATION_ONLY_TARGETS)
def test_generation_only_target_reads_no_vision_code(layout):
    """A generation-only target's modeling.py has no tower: none of the vision section's names."""
    text = _source(layout).read_text()
    for name in ("vision_encode", "image_inputs_embeds", "w4_rows", "_VisionW", "moonvit_"):
        assert name not in text, name
