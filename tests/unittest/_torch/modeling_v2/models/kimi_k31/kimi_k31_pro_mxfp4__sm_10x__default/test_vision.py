# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's vision tower (the default target's ``K31VisionTower``): against the engine MoonViT3d tower, its
attention op reuse, its media checks, the image-row splice and the KV-capacity profiling hook.

Both towers are built at the K3.1 vision config, filled from the same checkpoint weights and run on the same
images: the loaded parameters, every attention call's arguments and the image rows must match bit for bit. The engine
tower runs with dense attention (no DSA), the engine's no-KV-cache encoder metadata, and the attention ``max_seq_len``
either the batch's longest image or a fixed capacity, the K3.1 tower's. The K3.1 tower keeps one native attention op
across image batches of any shape.
"""

import json
import os
import subprocess
import sys
import types

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.input_processor import (
    moonvit_max_image_grid,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    modeling as target_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31VisionTower,
    build_vision_tower,
    validate_image_request,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.vision.moonvit_attention import (
    vision_step,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_k3_vl
from tensorrt_llm._torch.models.modeling_multimodal_utils import fuse_input_embeds
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.inputs.multimodal import MultimodalParams

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

#: Kimi K3.1's checkpoint ``vision_config``.
K31_VISION_CONFIG = {
    "activation_func": "gelu_pytorch_tanh",
    "attn_bias": False,
    "init_pos_emb_height": 64,
    "init_pos_emb_time": 4,
    "init_pos_emb_width": 64,
    "linear_bias": False,
    "merge_kernel_size": [2, 2],
    "merge_type": "sd2_tpool",
    "mlp_type": "mlp2",
    "mm_hidden_size": 1024,
    "mm_projector_type": "patchmergerv2",
    "norm_type": "rmsnorm",
    "patch_embed_proj_bias": False,
    "patch_size": 14,
    "pos_emb_type": "divided_fixed",
    "projector_hidden_act": "gelu",
    "projector_ln_eps": 1e-05,
    "qkv_hidden_size": 1536,
    "text_hidden_size": 7168,
    "vt_hidden_size": 1024,
    "vt_intermediate_size": 4096,
    "vt_num_attention_heads": 12,
    "vt_num_hidden_layers": 27,
}
TEXT_HIDDEN = 7168
#: The engine's encoder budget and the processor's attention capacity for it.
ENCODER_TOKENS = 16384
#: Kimi K3.1's checkpoint ``preprocessor_config.json`` grid limits.
K31_MEDIA_PROC_CFG = {
    "in_patch_limit": 65536,
    "patch_size": 14,
    "merge_kernel_size": 2,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 16384,
    "temporal_merge_kernel_size": 4,
}
#: Images of one call, per request: the table's own 64x64 grid, resampled grids, the smallest 2x2 grid.
REQUEST_GRIDS = [[(1, 32, 48), (1, 2, 2)], [(1, 64, 64)], [(1, 20, 34)]]


def _composite_config() -> PretrainedConfig:
    config = PretrainedConfig.from_dict(
        {"model_type": "kimi_k3", "torch_dtype": "bfloat16", "media_placeholder_token_id": 163605}
    )
    config.vision_config = dict(K31_VISION_CONFIG)
    config.text_config = PretrainedConfig.from_dict({"hidden_size": TEXT_HIDDEN})
    return config


def _checkpoint(seed: int = 0) -> dict:
    """Random ``vision_tower.*`` / ``mm_projector.*`` weights at the K3.1 shapes, as the checkpoint holds them."""
    gen = torch.Generator().manual_seed(seed)
    cfg = K31_VISION_CONFIG
    hidden, qkv, mlp = cfg["vt_hidden_size"], cfg["qkv_hidden_size"], cfg["vt_intermediate_size"]
    merged = 4 * cfg["mm_hidden_size"]

    def w(*shape, std=0.02):
        return (std * torch.randn(*shape, generator=gen)).bfloat16()

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=gen)).bfloat16()

    weights = {
        "vision_tower.patch_embed.proj.weight": w(hidden, 3, 14, 14),
        "vision_tower.patch_embed.pos_emb.weight": w(64, 64, hidden, std=1.0),
        "vision_tower.encoder.final_layernorm.weight": norm(hidden),
        "mm_projector.proj.0.weight": w(merged, merged),
        "mm_projector.proj.2.weight": w(TEXT_HIDDEN, merged),
        "mm_projector.post_norm.weight": norm(TEXT_HIDDEN),
    }
    for i in range(cfg["vt_num_hidden_layers"]):
        prefix = f"vision_tower.encoder.blocks.{i}."
        weights.update(
            {
                f"{prefix}norm0.weight": norm(hidden),
                f"{prefix}norm1.weight": norm(hidden),
                f"{prefix}wqkv.weight": w(3 * qkv, hidden),
                f"{prefix}wo.weight": w(hidden, qkv),
                f"{prefix}mlp.fc0.weight": w(mlp, hidden),
                f"{prefix}mlp.fc1.weight": w(hidden, mlp),
            }
        )
    return weights


def _requests(seed: int = 1) -> list:
    """The processed images of each request, on the GPU (the executor's device paths)."""
    gen = torch.Generator().manual_seed(seed)
    params = []
    for grids in REQUEST_GRIDS:
        patches = sum(t * h * w for t, h, w in grids)
        params.append(
            MultimodalParams(
                multimodal_data={
                    "image": {
                        "pixel_values": torch.randn(patches, 3, 14, 14, generator=gen).cuda(),
                        "image_grid_thw": torch.tensor(grids, dtype=torch.int64).cuda(),
                    }
                }
            )
        )
    return params


def _fixed_capacity_prepare(self, seq_lens, attn_metadata, *, capacity: int):
    """Encoder metadata with ``max_seq_len`` fixed at ``capacity`` (the engine's ``set_attn_max_seq_len`` contract)."""
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, pin_memory=prefer_pinned())
    attn_metadata.num_contexts = len(seq_lens)
    attn_metadata.request_ids = list(range(1, len(seq_lens) + 1))
    attn_metadata.prompt_lens = seq_lens
    attn_metadata.seq_lens = seq_lens_tensor
    attn_metadata.max_seq_len = max(capacity, max(seq_lens))
    attn_metadata.prepare()
    return attn_metadata


def _engine_tower(weights: dict, capacity: int, max_num_requests: int):
    model_config = ModelConfig(pretrained_config=_composite_config())
    tower = modeling_kimi_k3_vl.KimiK3VisionModel(model_config)
    tower.load_weights(weights)
    tower.encoder.setup_attn_metadata(
        max_num_tokens=capacity or ENCODER_TOKENS,
        attention_metadata_capacity={"attention": max_num_requests},
    )
    if capacity:
        tower.encoder.prepare_attn_metadata = types.MethodType(
            lambda self, seq_lens, md: _fixed_capacity_prepare(
                self, seq_lens, md, capacity=capacity
            ),
            tower.encoder,
        )
    return tower


def _engine_parameters(tower) -> dict:
    """The engine tower's parameters under the checkpoint's names (its fused qkv is the checkpoint's wqkv)."""
    renames = {
        ".attn.qkv_proj.": ".wqkv.",
        ".attn.o_proj.": ".wo.",
        ".mlp.up_proj.": ".mlp.fc0.",
        ".mlp.down_proj.": ".mlp.fc1.",
    }
    out = {}
    for name, param in tower.named_parameters():
        for old, new in renames.items():
            name = name.replace(old, new)
        key = name if name.startswith("mm_projector.") else f"vision_tower.{name}"
        out[key] = param
    return out


class _AttentionCalls:
    """Record the arguments of every ``thop.attention`` call, then run it.

    The compared tensors are copied at call time: a tower may reuse its metadata tensors from one call to the next.
    """

    def __init__(self, monkeypatch):
        self.calls = []
        original = thop.attention

        def record(*args, **kwargs):
            compared = _KERNEL_ARGS + _CAPACITY_ARGS
            self.calls.append(
                {
                    k: v.clone() if isinstance(v, torch.Tensor) and k in compared else v
                    for k, v in kwargs.items()
                }
            )
            return original(*args, **kwargs)

        monkeypatch.setattr(thop, "attention", record)


#: Arguments that select or parameterize the attention kernel; the rest address buffers or are inert here.
_KERNEL_ARGS = (
    "num_heads",
    "num_kv_heads",
    "head_size",
    "mask_type",
    "quant_mode",
    "q_scaling",
    "position_embedding_type",
    "is_fused_qkv",
    "update_kv_cache",
    "attention_input_type",
    "is_mla_enable",
    "use_paged_context_fmha",
    "beam_width",
    "tokens_per_block",
    "num_contexts",
    "num_ctx_tokens",
    "sequence_length",
    "host_past_key_value_lengths",
    "host_total_kv_lens",
    "context_lengths",
    "host_context_lengths",
    "host_request_types",
    "attention_sinks",
    "out_scale",
    "is_spec_decoding_enabled",
    "use_spec_decoding",
    "is_cross",
)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and torch.equal(a.cpu(), b.cpu())
        )
    return a == b


#: The capacity arguments, equal when the engine tower runs at the K3.1 tower's capacity.
_CAPACITY_ARGS = ("max_seq_len", "attention_window_size", "max_context_length", "max_num_requests")


@pytest.fixture(scope="module")
def checkpoint():
    return _checkpoint()


def _k31_tower(tmp_path) -> K31VisionTower:
    """The K3.1 tower as the loader builds it beside the K3.1 checkpoint's processor config."""
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"media_proc_cfg": K31_MEDIA_PROC_CFG})
    )
    config = _composite_config()
    config._name_or_path = str(tmp_path)
    return build_vision_tower(config)


@_GPU
@pytest.mark.parametrize("fixed", [False, True], ids=["longest_image", "tower_capacity"])
def test_image_rows_match_the_engine_tower_bit_for_bit(checkpoint, fixed, monkeypatch, tmp_path):
    tower = _k31_tower(tmp_path)
    tower.load_weights(checkpoint)
    capacity, max_num_requests = tower.attention_capacity
    engine = _engine_tower(checkpoint, capacity if fixed else 0, max_num_requests)

    engine_params = _engine_parameters(engine)
    family_params = tower._checkpoint_parameters()
    assert set(family_params) == set(checkpoint)
    for name, param in family_params.items():
        assert param.dtype == engine_params[name].dtype == torch.bfloat16, name
        assert torch.equal(param, engine_params[name]), name

    # The family tower encodes request by request (one encoder call holds at most the admitted budget), so the
    # engine tower runs each request on its own as the reference: the same shapes per call on both sides.
    params = _requests()
    calls = _AttentionCalls(monkeypatch)
    expected_rows = []
    for param in params:
        [request_rows] = engine.forward([param])
        expected_rows.append(request_rows)
    expected = torch.cat(expected_rows, dim=0)
    engine_calls, calls.calls = calls.calls, []
    [rows] = tower.forward(params)
    family_calls = calls.calls

    # The engine attention runs on the thop op (FlashInfer TRTLLM-Gen declines a cache-less encoder), once per block
    # and request.
    layers = K31_VISION_CONFIG["vt_num_hidden_layers"]
    assert len(engine_calls) == len(family_calls) == layers * len(params)
    for call, (ours, theirs) in enumerate(zip(family_calls, engine_calls)):
        for arg in _KERNEL_ARGS + (_CAPACITY_ARGS if fixed else ()):
            assert _same(ours.get(arg), theirs.get(arg)), (
                divmod(call, layers),
                arg,
                ours.get(arg),
                theirs.get(arg),
            )

    merged = sum((h // 2) * (w // 2) for grids in REQUEST_GRIDS for _, h, w in grids)
    assert rows.shape == expected.shape == (merged, TEXT_HIDDEN)
    assert rows.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(expected.float()).all()
    assert torch.equal(rows, expected)


@_GPU
def test_largest_image_encodes_at_the_processors_maximum_grid(checkpoint, tmp_path):
    """The KV-capacity profiling image: the checkpoint processor's largest grid, through the whole tower."""
    tower = _k31_tower(tmp_path)
    assert tower.max_image_grid == moonvit_max_image_grid(K31_MEDIA_PROC_CFG, (2, 2), 14)
    tower.load_weights(checkpoint)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    rows = tower.encode_largest_image()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    t, h, w = tower.max_image_grid
    print(
        f"largest image grid {(t, h, w)}: {t * h * w} patches, {rows.shape[0]} rows, peak {peak / 2**30:.2f} GiB"
    )
    assert rows.shape == ((h // 2) * (w // 2), TEXT_HIDDEN)
    assert torch.isfinite(rows.float()).all()


@_GPU
def test_a_checkpoint_without_processor_geometry_takes_the_default_limits(tmp_path):
    config = _composite_config()
    config._name_or_path = str(tmp_path)
    assert build_vision_tower(config).max_image_grid == moonvit_max_image_grid({}, (2, 2), 14)


@_GPU
def test_a_processor_config_that_admits_no_image_grid_fails_the_build(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"media_proc_cfg": dict(K31_MEDIA_PROC_CFG, patch_limit_on_one_side=1)})
    )
    config = _composite_config()
    config._name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match="admit no image grid"):
        build_vision_tower(config)


@_GPU
def test_attention_capacity_is_the_largest_image(tmp_path):
    tower = _k31_tower(tmp_path)
    t, h, w = tower.max_image_grid
    assert tower.attention_capacity == (t * h * w, t * h * w // 4)
    with pytest.raises(ValueError, match="attention capacity"):
        vision_step([t * h * w + 1], *tower.attention_capacity, "cuda", tower._workspace)


#: Run in a subprocess with the attention op's trace log on: image batches of different shapes, each followed by a
#: marker line. The op logs "is not cached" whenever it builds a native op for a new configuration.
_OP_REUSE_SCRIPT = r"""
import sys

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31VisionTower,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams

config = dict(VISION_CONFIG, vt_num_hidden_layers=2)
tower = K31VisionTower(config, 7168, device=torch.device("cuda"))
with torch.no_grad():
    for param in tower.parameters():
        param.normal_(0.0, 0.02)
for i, grids in enumerate([[(1, 8, 12)], [(1, 4, 4), (1, 32, 20)], [(1, 2, 2), (1, 6, 6), (1, 64, 64)]]):
    patches = sum(t * h * w for t, h, w in grids)
    image = {"pixel_values": torch.randn(patches, 3, 14, 14, device="cuda"),
             "image_grid_thw": torch.tensor(grids, device="cuda")}
    tower.forward([MultimodalParams(multimodal_data={"image": image})])
    torch.cuda.synchronize()
    print(f"MOONVIT_BATCH_DONE {i}", flush=True)
"""


@_GPU
def test_image_batches_of_any_shape_reuse_one_attention_op():
    """The native attention op cache is keyed by the call's capacities and never evicts: the tower's fixed
    capacities build one op on the first image batch and reuse it for every later batch shape."""
    script = _OP_REUSE_SCRIPT.replace("VISION_CONFIG", repr(K31_VISION_CONFIG))
    env = dict(os.environ, TLLM_LOG_LEVEL_BY_MODULE="trace:thop")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]
    segments = result.stdout.split("MOONVIT_BATCH_DONE")
    built = [segment.count("is not cached") for segment in segments[:3]]
    assert built[0] >= 1, "the trace log did not report the first op build"
    assert built[1:] == [0, 0], f"native attention ops built per batch: {built}"


@_GPU
def test_a_checkpoint_without_tower_weights_reports_no_load():
    tower = K31VisionTower(K31_VISION_CONFIG, TEXT_HIDDEN, device=torch.device("cuda"))
    assert tower.load_weights({"language_model.model.embed_tokens.weight": torch.zeros(1)}) is False


@_GPU
def test_a_partial_tower_checkpoint_fails_the_load(checkpoint):
    tower = K31VisionTower(K31_VISION_CONFIG, TEXT_HIDDEN, device=torch.device("cuda"))
    partial = dict(checkpoint)
    del partial["mm_projector.post_norm.weight"]
    with pytest.raises(ValueError, match="missing"):
        tower.load_weights(partial)


@_GPU
def test_unsupported_vision_variants_are_rejected():
    with pytest.raises(ValueError, match="norm_type"):
        K31VisionTower(dict(K31_VISION_CONFIG, norm_type="layernorm"), TEXT_HIDDEN)


# ---------------------------------------------------------------------------
# Media checks, the image-row splice and the profiling hook (no GPU)
# ---------------------------------------------------------------------------

_PATCH = 14
_PLACEHOLDER = 9
_HIDDEN = 4


def _payload(*grids, pixel_rows=None):
    if pixel_rows is None:
        pixel_rows = sum(t * h * w for t, h, w in grids)
    return {
        "pixel_values": torch.zeros(pixel_rows, 3, _PATCH, _PATCH),
        "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
    }


def _unbuilt_tower():
    """An unbuilt MoonViT tower with the geometry ``validate_multimodal_data`` reads."""
    tower = K31VisionTower.__new__(K31VisionTower)
    tower.merge_kernel_size = (2, 2)
    tower.num_frames = 4
    tower.patch_size = _PATCH
    tower.max_image_grid = moonvit_max_image_grid({}, tower.merge_kernel_size, _PATCH)
    return tower


@pytest.mark.cpu_only
@pytest.mark.parametrize("grid", [(5, 2, 2), (1, 514, 2), (1, 2, 3)])
def test_tower_rejects_grids_it_cannot_encode(grid):
    tower = _unbuilt_tower()
    tower.validate_multimodal_data({"image": _payload((4, 512, 2))})

    with pytest.raises(ValueError, match="MoonViT cannot encode"):
        tower.validate_multimodal_data({"image": _payload(grid)})


@pytest.mark.cpu_only
def test_tower_rejects_an_image_beyond_its_attention_capacity():
    """A grid larger than the processor's largest image would exceed the attention op's fixed capacity: the request
    is refused at admission, not in the step that encodes it."""
    tower = _unbuilt_tower()
    t, h, w = tower.max_image_grid
    tower.validate_multimodal_data({"image": _payload((t, h, w))})

    with pytest.raises(ValueError, match="at most .* patches"):
        tower.validate_multimodal_data({"image": _payload((t, h + 2, w))})


@pytest.mark.cpu_only
def test_tower_rejects_a_request_over_the_encoder_budget():
    """All of a request's images together must fit the encoder budget, the patches of the processor's largest image
    (what the KV-capacity estimate reserves): two images of half the largest grid pass, two largest images do not."""
    tower = _unbuilt_tower()
    t, h, w = tower.max_image_grid
    half_h = (h // 4) * 2
    tower.validate_multimodal_data({"image": _payload((t, half_h, w), (t, half_h, w))})

    with pytest.raises(ValueError, match="more than the vision encoder budget"):
        tower.validate_multimodal_data({"image": _payload((t, h, w), (t, h, w))})


@pytest.mark.cpu_only
def test_forward_encodes_each_request_on_its_own():
    """One encoder call per request (at most the admitted budget each), the rows concatenated in request order."""
    tower = _unbuilt_tower()
    calls = []

    def encode(pixel_values, grids):
        calls.append((pixel_values.shape[0], list(grids)))
        return torch.full((pixel_values.shape[0], _HIDDEN), float(len(calls)))

    tower.encode = encode
    params = [
        types.SimpleNamespace(multimodal_data={"image": _payload((1, 2, 2))}),
        types.SimpleNamespace(multimodal_data={}),
        types.SimpleNamespace(multimodal_data={"image": _payload((1, 4, 2), (1, 2, 2))}),
    ]
    (rows,) = tower.forward(params)

    assert calls == [(4, [(1, 2, 2)]), (12, [(1, 4, 2), (1, 2, 2)])]
    assert rows.shape == (16, _HIDDEN)
    assert rows[:4].eq(1).all() and rows[4:].eq(2).all()
    assert tower.forward([types.SimpleNamespace(multimodal_data={})]) == []


@pytest.mark.cpu_only
def test_processor_limits_that_admit_no_image_grid_are_refused():
    with pytest.raises(ValueError, match="admit no image grid"):
        _unbuilt_tower().largest_image_grid({"patch_limit_on_one_side": 1})


def _admitted(mm_data, generation_only=False):
    return types.SimpleNamespace(
        py_multimodal_data=mm_data, is_generation_only_request=generation_only
    )


@pytest.mark.cpu_only
def test_admission_refuses_media_without_a_local_tower():
    validate_image_request(None, _admitted(None))
    validate_image_request(_unbuilt_tower(), _admitted({"image": _payload((1, 4, 6))}))

    with pytest.raises(ValueError, match="runs no vision tower"):
        validate_image_request(None, _admitted({"image": _payload((1, 4, 6))}))
    with pytest.raises(ValueError, match="MoonViT cannot encode"):
        validate_image_request(_unbuilt_tower(), _admitted({"image": _payload((1, 2, 3))}))


@pytest.mark.cpu_only
def test_admission_passes_generation_only_media_without_a_local_tower():
    """A generation-only request (P/D decode side) never encodes, so it needs no tower; a local tower still checks
    its grids."""
    validate_image_request(None, _admitted({"image": _payload((1, 4, 6))}, generation_only=True))
    with pytest.raises(ValueError, match="MoonViT cannot encode"):
        validate_image_request(
            _unbuilt_tower(), _admitted({"image": _payload((1, 2, 3))}, generation_only=True)
        )


@pytest.mark.cpu_only
def test_tower_rejects_pixels_that_do_not_match_the_grids():
    with pytest.raises(ValueError, match="pixel_values must have shape"):
        _unbuilt_tower().validate_multimodal_data({"image": _payload((1, 4, 6), pixel_rows=25)})


@pytest.mark.cpu_only
def test_splice_gives_the_engine_fusion():
    """The image-row splice writes the rows the engine's ``fuse_input_embeds`` writes, over the same embeddings."""
    gen = torch.Generator().manual_seed(0)
    embed_tokens = torch.nn.Embedding(16, _HIDDEN).to(torch.bfloat16)
    input_ids = torch.tensor([1, _PLACEHOLDER, _PLACEHOLDER, 2, _PLACEHOLDER, 3], dtype=torch.int32)
    mm_token_indices = torch.tensor([1, 2, 4])
    text_token_indices = torch.tensor([0, 3, 5])
    rows = [torch.randn(2, _HIDDEN, generator=gen), torch.randn(1, _HIDDEN, generator=gen)]

    spliced = target_modeling.splice_image_rows(embed_tokens, input_ids, rows, mm_token_indices)
    fused_ids, fused = fuse_input_embeds(
        embed_tokens,
        input_ids,
        rows,
        mm_token_ids=torch.tensor([_PLACEHOLDER], dtype=torch.int32),
        mm_token_indices=mm_token_indices,
        text_token_indices=text_token_indices,
    )
    assert fused_ids is None
    assert spliced.dtype == fused.dtype == torch.bfloat16
    assert torch.equal(spliced, fused)


@pytest.mark.cpu_only
def test_splice_rejects_a_row_count_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        target_modeling.splice_image_rows(
            torch.nn.Embedding(16, _HIDDEN),
            torch.tensor([1, _PLACEHOLDER], dtype=torch.int32),
            [torch.zeros(2, _HIDDEN)],
            torch.tensor([1]),
        )


class _CountingTower:
    """Stand-in for the vision tower: counts the profiling encodes."""

    max_image_grid = (1, 4, 6)

    def __init__(self):
        self.encodes = 0

    def encode_largest_image(self):
        self.encodes += 1
        return torch.zeros(6, _HIDDEN)


def _metadata(num_contexts, estimating):
    return types.SimpleNamespace(
        num_contexts=num_contexts,
        kv_cache_manager=types.SimpleNamespace(is_estimating_kv_cache=estimating),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_contexts, estimating, has_images, encodes",
    [
        (1, True, False, 1),  # the KV-capacity estimate's profiling step
        (1, False, False, 0),  # a served text step
        (0, True, False, 0),  # a generation-only step
        (1, True, True, 0),  # a step that encodes its own images
    ],
)
def test_profiling_step_holds_one_largest_image(num_contexts, estimating, has_images, encodes):
    tower = _CountingTower()
    with target_modeling.profiling_image_rows(
        tower, _metadata(num_contexts, estimating), has_images
    ):
        pass
    assert tower.encodes == encodes


@pytest.mark.cpu_only
def test_profiling_without_a_local_tower_encodes_nothing():
    with target_modeling.profiling_image_rows(None, _metadata(1, True), False):
        pass
