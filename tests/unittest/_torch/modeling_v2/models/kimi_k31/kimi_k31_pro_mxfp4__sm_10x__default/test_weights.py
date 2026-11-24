# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The targets' ``weights.py`` on CPU, each target's copy: the checkpoint helpers against the engine helpers they stand
in for (the same tensor views, the same keys in the same order, the same jobs), the key split, lazy reads, the
routed-expert layout, trunk dtypes and shards, and each target's post-load step."""

import importlib
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_utils import filter_weights, run_concurrently
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
#: target -> whether it packs each MLA layer's input projections after the load.
_TARGET_PACKS = {
    "kimi_k31_pro_mxfp4__sm_103__tep": True,
    "kimi_k31_pro_mxfp4__sm_103__dep": True,
    "kimi_k31_pro_mxfp4__sm_10x__default": False,
}


@pytest.fixture(params=list(_TARGET_PACKS), ids=["tep", "dep", "default"])
def target(request):
    return request.param


@pytest.fixture
def target_weights(target):
    """The target's ``weights`` module."""
    return importlib.import_module(f"{_FAMILY}.{target}.weights")


# ---------------------------------------------------------------------------
# The checkpoint helpers.
# ---------------------------------------------------------------------------

_KDA_HEADS = 96
_KDA_WIDTH = _KDA_HEADS * 128  # KDA heads x head dim
_DENSE = 33792  # dense MLP intermediate size
_SHARED = 2 * 3072  # shared experts x their intermediate size
_OTHER = 64  # width of the axis that is not split; it does not change the slicing

# (shape, split dim): the split axis has the width of the checkpoint tensor it stands for.
_SHARD_CASES = [
    ((_KDA_WIDTH, _OTHER), 0),  # KDA q / k / v / f_b / g_b projections: output rows
    ((_OTHER, _KDA_WIDTH), 1),  # KDA o_proj: input columns
    ((_KDA_HEADS, _OTHER), 0),  # KDA b_proj
    ((_KDA_WIDTH, 1, 4), 0),  # KDA short-conv weight
    ((_KDA_WIDTH,), 0),  # KDA dt_bias
    ((_KDA_WIDTH,), 1),  # a 1-D tensor has no input columns: kept whole
    ((1, _OTHER), 0),  # a width-1 axis: kept whole
    ((_DENSE, _OTHER), 0),  # dense MLP gate / up
    ((_OTHER, _DENSE), 1),  # dense MLP down
    ((_SHARED, _OTHER), 0),  # shared-expert gate / up
    ((_OTHER, _SHARED), 1),  # shared-expert down
    ((_OTHER, 100), 1),  # uneven: the last part is shorter
    ((18, _OTHER), 0),  # uneven: the parts past the end are empty
]
_TP_SIZES = (1, 2, 3, 4, 5, 8, 12, 16, 32)


def _random(shape, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
    return torch.randn(shape, generator=gen).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.uint8])
@pytest.mark.parametrize("shape, dim", _SHARD_CASES)
def test_tp_shard_is_load_weight_shard(target_weights, shape, dim, dtype):
    tensor = _random(shape, dtype)
    mode = TensorParallelMode.ROW if dim == 1 else TensorParallelMode.COLUMN
    cpu = torch.device("cpu")
    for tp_size in _TP_SIZES:
        for tp_rank in range(tp_size):
            expected = load_weight_shard(tensor, tp_size, tp_rank, mode, device=cpu)
            got = target_weights.tp_shard(tensor, tp_size, tp_rank, dim, cpu)
            assert got.dtype == expected.dtype
            assert got.shape == expected.shape, (tp_size, tp_rank)
            # The same view of the same storage, so the same bytes reach the parameter.
            assert got.stride() == expected.stride()
            assert got.storage_offset() == expected.storage_offset()
            assert got.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
            assert torch.equal(got, expected)


def _checkpoint_names():
    names = []
    for layer in range(3):
        base = f"language_model.model.layers.{layer}"
        names += [f"{base}.self_attn.o_proj.weight", f"{base}.mlp.experts.0.w1.weight_packed"]
    names += ["language_model.lm_head.weight", "language_model.model.embed_tokens.weight"]
    names += ["vision_tower.patch_embed.proj.weight", "mm_projector.proj.0.weight"]
    return names


@pytest.mark.parametrize("source_kind", ["family", "engine", "dict"])
def test_strip_prefix_is_filter_weights(target_weights, source_kind):
    names = _checkpoint_names()
    values = {name: torch.full((2,), float(i)) for i, name in enumerate(names)}
    if source_kind == "family":
        source = target_weights.CheckpointWeights(values)
        source.checkpoint_dir = "/checkpoint"
    elif source_kind == "engine":
        source = ConsumableWeightsDict(dict(values))
        source.checkpoint_dir = "/checkpoint"
    else:
        source = dict(values)

    # What the targets built before: the engine filter in an engine dict, with the directory and the prefix.
    expected = ConsumableWeightsDict(filter_weights("language_model", source))
    checkpoint_dir = getattr(source, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        expected.checkpoint_dir = checkpoint_dir
    expected.checkpoint_prefix = "language_model."

    got = target_weights.strip_prefix(source, "language_model.")
    assert list(got.keys()) == list(expected.keys())
    assert all(got[name] is expected[name] for name in got)
    assert getattr(got, "checkpoint_dir", None) == getattr(expected, "checkpoint_dir", None)
    assert got.checkpoint_prefix == expected.checkpoint_prefix
    # Neither consumes a source entry.
    assert list(source.keys()) == names


def test_checkpoint_weights_reads_like_the_engine_dict(target_weights):
    names = _checkpoint_names()
    values = {name: torch.full((2,), float(i)) for i, name in enumerate(names)}
    family, engine = target_weights.CheckpointWeights(values), ConsumableWeightsDict(dict(values))
    assert len(family) == len(engine)
    assert list(family) == list(engine) == list(family.keys()) == list(engine.keys())
    assert [(k, v) for k, v in family.items()] == [(k, v) for k, v in engine.items()]
    assert all(family[name] is engine[name] and name in family for name in names)
    # The loader reads these with getattr defaults; a fresh container has neither set.
    assert getattr(family, "checkpoint_dir", None) is getattr(engine, "checkpoint_dir", None)
    assert getattr(family, "checkpoint_prefix", "") == getattr(engine, "checkpoint_prefix", "")
    family.clear()
    engine.clear()
    assert len(family) == len(engine) == 0


@pytest.mark.parametrize("num_workers", [1, 8])
def test_run_in_threads_is_run_concurrently(target_weights, num_workers):
    sources = [_random((_OTHER, 33), torch.bfloat16, seed=i) for i in range(32)]

    def load_all(run):
        params = [torch.zeros(_OTHER, 33, dtype=torch.float32) for _ in sources]
        calls, threads = [], set()
        lock = threading.Lock()

        def load_param(index, param):
            param.copy_(
                target_weights.tp_shard(sources[index], 1, 0, 0, torch.device("cpu")).to(
                    param.dtype
                )
            )
            with lock:
                calls.append(index)
                threads.add(threading.get_ident())

        run(load_param, [(i, p) for i, p in enumerate(params)], num_workers=num_workers)
        return params, sorted(calls), threads

    got, got_calls, got_threads = load_all(target_weights.run_in_threads)
    expected, expected_calls, _ = load_all(run_concurrently)
    assert got_calls == expected_calls == list(range(len(sources)))
    assert len(got_threads) <= num_workers
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_run_in_threads_raises_after_every_job_ran(target_weights, monkeypatch):
    logged = []
    monkeypatch.setattr(target_weights, "logger", SimpleNamespace(error=logged.append))

    def run_jobs(run):
        ran = []

        def job(index):
            ran.append(index)
            if index == 3:
                raise ValueError("job 3 failed")

        with pytest.raises(ValueError, match="job 3 failed"):
            run(job, [(i,) for i in range(8)], num_workers=2)
        return sorted(ran)

    assert run_jobs(target_weights.run_in_threads) == run_jobs(run_concurrently) == list(range(8))
    # Like the engine helper, it names the function and the job that failed.
    assert logged == ["job failed on job (3)"]


# ---------------------------------------------------------------------------
# The checkpoint read, its split, the routed-expert layout and the post-load step.
# ---------------------------------------------------------------------------

_LANG = ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight"]
_VISION = ["vision_tower.patch_embed.proj.weight", "mm_projector.proj.0.weight"]


class _VisionTower(nn.Module):
    def __init__(self, has_weights: bool = True) -> None:
        super().__init__()
        self.loaded = None
        self.has_weights = has_weights

    def load_weights(self, weights) -> bool:
        self.loaded = sorted(weights.keys())
        return self.has_weights


def _model(skip_mm_encoder: bool, mm_encoder=None) -> SimpleNamespace:
    return SimpleNamespace(
        _skip_mm_encoder=skip_mm_encoder,
        mm_encoder=mm_encoder,
        llm=SimpleNamespace(name="text model"),
    )


@pytest.fixture
def text_loads(monkeypatch, target_weights):
    """Record what the text model's loader receives instead of loading it."""
    calls = []

    def load_text_model(llm, weights):
        calls.append((llm, sorted(weights.keys()), getattr(weights, "checkpoint_prefix", None)))

    monkeypatch.setattr(target_weights, "load_text_model", load_text_model)
    return calls


def test_split_keys(target_weights):
    lang, vision, unknown = target_weights.split_keys(_LANG + _VISION + ["mtp.0.weight"])
    assert lang == _LANG
    assert vision == _VISION
    assert unknown == ["mtp.0.weight"]


def test_unknown_key_fails_naming_it(target_weights, text_loads):
    weights = {k: None for k in _LANG + _VISION + ["model.layers.0.extra.weight"]}
    with pytest.raises(ValueError, match="model.layers.0.extra.weight"):
        target_weights.load(_model(skip_mm_encoder=True), weights)
    assert not text_loads


def test_no_language_model_keys_fails(target_weights, text_loads):
    with pytest.raises(ValueError, match="language_model"):
        target_weights.load(_model(skip_mm_encoder=True), {k: None for k in _VISION})


def test_vision_keys_are_a_non_load_without_a_local_tower(target_weights, text_loads):
    model = _model(skip_mm_encoder=True)
    target_weights.load(model, {k: None for k in _LANG + _VISION})
    assert model.mm_encoder is None
    ((llm, keys, prefix),) = text_loads
    assert llm is model.llm
    # The text loader sees only the language-model keys, with the prefix stripped.
    assert keys == ["lm_head.weight", "model.embed_tokens.weight"]
    assert prefix == target_weights.LANG_PREFIX


def test_vision_keys_go_to_the_local_tower(target_weights, text_loads):
    tower = _VisionTower()
    model = _model(skip_mm_encoder=False, mm_encoder=tower)
    target_weights.load(model, {k: None for k in _LANG + _VISION})
    assert set(_VISION) <= set(tower.loaded)
    assert model.mm_encoder is tower
    ((_, keys, _),) = text_loads
    assert keys == ["lm_head.weight", "model.embed_tokens.weight"]


def test_a_checkpoint_without_tower_weights_runs_without_a_tower(target_weights, text_loads):
    """Image requests are then refused instead of encoded by an unfilled tower."""
    model = _model(skip_mm_encoder=False, mm_encoder=_VisionTower(has_weights=False))
    target_weights.load(model, {k: None for k in _LANG})
    assert model.mm_encoder is None


def test_open_checkpoint_weights_reads_lazily(target_weights, tmp_path):
    save_file({"a.weight": torch.arange(4.0)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b.weight": torch.ones(2, 3)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    save_file({"a.weight": torch.zeros(4)}, str(tmp_path / "consolidated.safetensors"))
    with target_weights.open_checkpoint_weights(str(tmp_path)) as weights:
        assert sorted(weights.keys()) == ["a.weight", "b.weight"]
        assert weights.checkpoint_dir == str(tmp_path)
        # A slice reads only when indexed.
        assert not isinstance(weights["a.weight"], torch.Tensor)
        assert torch.equal(weights["a.weight"][:], torch.arange(4.0))
        assert weights["b.weight"].get_shape() == [2, 3]
    assert len(weights) == 0


def test_open_checkpoint_weights_needs_safetensors(target_weights, tmp_path):
    with pytest.raises(RuntimeError, match="No safetensors"):
        with target_weights.open_checkpoint_weights(str(tmp_path)):
            pass


def test_the_engine_reads_the_checkpoint_through_the_target(tmp_path):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        ModelingV2KimiK31ProMxfp4Sm10xDefault as Target,
    )
    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
    from tensorrt_llm._torch.pyexecutor.model_loader import _open_checkpoint_weight_session

    save_file({"a.weight": torch.arange(4.0)}, str(tmp_path / "model.safetensors"))
    model = Target.__new__(Target)  # the class is what the loader asks; nothing is built
    with _open_checkpoint_weight_session(
        HfCheckpointLoader(), str(tmp_path), mapping=None, model=model
    ) as weights:
        assert weights.checkpoint_dir == str(tmp_path)
        assert not isinstance(weights["a.weight"], torch.Tensor)


def test_routed_experts_load_only_the_mxfp4_layout(target_weights):
    """Only the MXFP4 checkpoint's packed experts have a layout; any other format fails the model build."""
    spec = target_weights.expert_ckpt_spec(QuantAlgo.W4A8_MXFP4_MXFP8)
    assert spec.kinds == ("weight_packed", "weight_scale")
    assert spec.loader is target_weights._load_packed_mxfp4_expert
    for quant_algo in (QuantAlgo.NVFP4, QuantAlgo.NVFP4_AWQ, QuantAlgo.FP8, None):
        with pytest.raises(NotImplementedError, match="no per-expert checkpoint layout"):
            target_weights.expert_ckpt_spec(quant_algo)


def test_packed_mxfp4_expert_reads_the_checkpoint_expert_into_its_slot(target_weights, monkeypatch):
    calls = []
    monkeypatch.setattr(
        target_weights,
        "load_packed_mxfp4_expert",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    experts = SimpleNamespace(
        initial_local_expert_ids=[1, 2],
        w3_w1_weight="w31",
        w3_w1_weight_scale="w31s",
        w2_weight="w2",
        w2_weight_scale="w2s",
        _packed_mxfp4_loaded_slots=set(),
        intermediate_size=3072,
        tp_size=4,
        tp_rank=3,
    )
    target_weights._load_packed_mxfp4_expert(experts, "experts", 5, 2, 1, lambda key: key)
    ((args, kwargs),) = calls
    assert args == ("w31", "w31s", "w2", "w2s", 1, experts._packed_mxfp4_loaded_slots)
    assert kwargs == {
        "intermediate_size": 3072,
        "moe_tp_size": 4,
        "moe_tp_rank": 3,
        **{
            f"{w}_{name}": f"experts.5.{w}.{kind}"
            for w in ("w1", "w2", "w3")
            for name, kind in (("weight", "weight_packed"), ("weight_scale", "weight_scale"))
        },
    }
    # The slot serves the expert the routed experts' partition assigns it.
    with pytest.raises(ValueError, match="expects global expert 2"):
        target_weights._load_packed_mxfp4_expert(experts, "experts", 5, 7, 1, lambda key: key)


def test_fp8_trunk_tensor_fails_the_load(target_weights, tmp_path):
    bf16 = torch.ones(2, 2, dtype=torch.bfloat16)
    assert target_weights._read_unquantized({"w.weight": bf16}, "w.weight") is bf16
    save_file({"w.weight": bf16.to(torch.float8_e4m3fn)}, str(tmp_path / "model.safetensors"))
    with target_weights.open_checkpoint_weights(str(tmp_path)) as weights:
        with pytest.raises(ValueError, match="w.weight is FP8"):
            target_weights._read_unquantized(weights, "w.weight")


def test_finalize_runs_the_kda_steps_and_the_post_load_step_in_layer_order(
    target, target_weights, monkeypatch
):
    """Each KDA layer builds its decode weights and, on the targets that pack them, each MLA layer packs its input
    projections (with the output gate), in layer order; layers whose weights were removed are skipped."""
    packs = _TARGET_PACKS[target]
    calls = []
    if packs:
        monkeypatch.setattr(
            target_weights,
            "pack_dsa_hidden_projection",
            lambda mixer, pack_gate: calls.append(("pack", mixer.index, pack_gate)),
        )

    def kda_layer(i):
        attention = SimpleNamespace(
            finalize_decode_weights=lambda: calls.append(("finalize", i)),
        )
        return SimpleNamespace(is_kda=True, linear_attn=attention)

    def mla_layer(i, removed=False):
        return SimpleNamespace(
            is_kda=False,
            self_attn=SimpleNamespace(mixer=SimpleNamespace(index=i)),
            _weights_removed=removed,
        )

    layers = [kda_layer(0), mla_layer(1), kda_layer(2), mla_layer(3, removed=True), mla_layer(4)]
    llm = SimpleNamespace(model=SimpleNamespace(layers=layers))
    target_weights._finalize_weight_load(llm, 0, 0)
    if packs:
        assert calls == [("finalize", 0), ("pack", 1, True), ("finalize", 2), ("pack", 4, True)]
    else:
        assert calls == [("finalize", 0), ("finalize", 2)]


# ---------------------------------------------------------------------------
# Host paths of the text-model loader: the key check, the trunk shard rules, the safetensors range reader and the
# routed-expert streaming, on small CPU tensors.
# ---------------------------------------------------------------------------


def _llm(layers, tp_rank: int = 0) -> SimpleNamespace:
    anchor = nn.Parameter(torch.zeros(1))
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        parameters=lambda: iter([anchor]),
        model_config=SimpleNamespace(
            mapping=SimpleNamespace(tp_rank=tp_rank),
            pretrained_config=SimpleNamespace(_name_or_path=""),
        ),
    )


@pytest.mark.parametrize("prefix", ["", "language_model."])
def test_key_check_reports_unknown_keys_of_the_layers_the_model_holds(
    target_weights, prefix, monkeypatch
):
    warnings = []
    monkeypatch.setattr(target_weights, "logger", SimpleNamespace(warning=warnings.append))
    llm = _llm([SimpleNamespace(), SimpleNamespace(_weights_removed=True)])
    expected = {f"{prefix}model.layers.0.input_layernorm.weight", f"{prefix}lm_head.weight"}
    leftovers = [
        f"{prefix}model.layers.0.block_sparse_moe.experts.7.w1.weight_packed",  # another rank's expert
        f"{prefix}model.layers.1.input_layernorm.weight",  # a layer whose weights were removed
        f"{prefix}model.layers.2.input_layernorm.weight",  # past a truncated config's last layer
        "vision_tower.blocks.0.weight",
    ]
    surprising = [f"{prefix}model.layers.0.unknown.weight", f"{prefix}model.unknown.weight"]
    weights = dict.fromkeys([*expected, *leftovers, *surprising])
    target_weights._validate_checkpoint_keys(llm, weights, expected, prefix)
    (message,) = warnings
    assert "2 unmatched" in message
    assert all(key in message for key in surprising)
    assert not any(key in message for key in leftovers)


def test_key_check_fails_on_a_missing_key(target_weights):
    llm = _llm([SimpleNamespace()])
    with pytest.raises(KeyError, match="model.layers.0.input_layernorm.weight"):
        target_weights._validate_checkpoint_keys(
            llm, {}, {"model.layers.0.input_layernorm.weight"}, ""
        )


def test_trunk_parameters_take_this_ranks_shard(target_weights):
    """KDA head shards (rows; o_proj columns; A_log heads), the dense MLP's gate / up rows and down columns, and
    replicated tensors, at KDA TP 2 / MLP TP 2, rank 1."""
    hidden, kda, inter = 8, 8, 6
    gen = torch.Generator().manual_seed(0)
    ckpt = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(kda, hidden, generator=gen),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, kda, generator=gen),
        "model.layers.0.self_attn.dt_bias": torch.randn(kda, generator=gen),
        # 4 heads padded to 8 entries.
        "model.layers.0.self_attn.A_log": torch.cat(
            [torch.randn(4, generator=gen), torch.zeros(4)]
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(inter, hidden, generator=gen),
        "model.layers.0.mlp.up_proj.weight": torch.randn(inter, hidden, generator=gen),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, inter, generator=gen),
        "model.layers.0.input_layernorm.weight": torch.randn(hidden, generator=gen),
    }
    ckpt = {k: v.to(torch.bfloat16) if v.dim() == 2 else v for k, v in ckpt.items()}
    params = {
        "model.layers.0.linear_attn.q_proj.weight": torch.zeros(
            kda // 2, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.linear_attn.o_proj.weight": torch.zeros(
            hidden, kda // 2, dtype=torch.bfloat16
        ),
        "model.layers.0.linear_attn.dt_bias": torch.zeros(kda // 2),
        "model.layers.0.linear_attn.A_log": torch.zeros(2),
        "model.layers.0.mlp.gate_up_proj.weight": torch.zeros(inter, hidden, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(
            hidden, inter // 2, dtype=torch.bfloat16
        ),
        "model.layers.0.input_layernorm.weight": torch.zeros(hidden),
    }
    params = {name: nn.Parameter(p, requires_grad=False) for name, p in params.items()}
    name_map = {name: name.replace(".linear_attn.", ".self_attn.") for name in params}
    kda_layer = SimpleNamespace(
        is_kda=True, linear_attn=SimpleNamespace(_kda_tp_size=2, _kda_tp_rank=1)
    )
    llm = _llm([kda_layer], tp_rank=1)

    assert target_weights._load_trunk_params(llm, ckpt, params, name_map) == len(params)

    def got(name):
        return params[f"model.layers.0.{name}"].data

    def src(name):
        return ckpt[f"model.layers.0.{name}"]

    assert torch.equal(got("linear_attn.q_proj.weight"), src("self_attn.q_proj.weight")[4:])
    assert torch.equal(got("linear_attn.o_proj.weight"), src("self_attn.o_proj.weight")[:, 4:])
    assert torch.equal(got("linear_attn.dt_bias"), src("self_attn.dt_bias")[4:])
    assert torch.equal(got("linear_attn.A_log"), src("self_attn.A_log")[2:4])
    gate_up = torch.cat([src("mlp.gate_proj.weight")[3:], src("mlp.up_proj.weight")[3:]])
    assert torch.equal(got("mlp.gate_up_proj.weight"), gate_up)
    assert torch.equal(got("mlp.down_proj.weight"), src("mlp.down_proj.weight")[:, 3:])
    assert torch.equal(got("input_layernorm.weight"), src("input_layernorm.weight"))


def _safetensors_header(path):
    import json

    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(header_len))


def test_read_safetensors_slices_returns_the_file_tensors(target_weights, tmp_path):
    tensors = {
        "a": torch.randn(3, 5).to(torch.bfloat16),
        "b": torch.randn(7),
        "c": torch.randint(0, 256, (4, 9), dtype=torch.uint8),
        "d": torch.arange(6, dtype=torch.int32).view(2, 3),
        "e": torch.randn(300_000),  # 1.2 MB, not read: the ranges on either side of it stay apart
        "f": torch.randn(2, 2).half(),
    }
    path = str(tmp_path / "shard.safetensors")
    save_file(tensors, path)
    buffers = []

    def get_buffer(nbytes):
        buffers.append(torch.empty(nbytes, dtype=torch.uint8))
        return buffers[-1]

    names = ["f", "a", "c", "d", "b", "not-in-file"]
    out = target_weights.read_safetensors_slices(path, names, get_buffer)
    assert sorted(out) == ["a", "b", "c", "d", "f"]
    (buf,) = buffers
    header = _safetensors_header(path)
    for name, tensor in out.items():
        assert tensor.dtype == tensors[name].dtype
        assert torch.equal(tensor, tensors[name])
        # Every view keeps its file offset's alignment.
        start = header[name]["data_offsets"][0]
        assert (tensor.data_ptr() - buf.data_ptr()) % 64 == start % 64


def test_read_safetensors_slices_declines_unknown_dtypes_and_short_files(target_weights, tmp_path):
    path = str(tmp_path / "shard.safetensors")
    save_file({"f64": torch.randn(4, dtype=torch.float64), "f32": torch.randn(64)}, path)

    def get_buffer(nbytes):
        return torch.empty(nbytes, dtype=torch.uint8)

    assert target_weights.read_safetensors_slices(path, ["f64"], get_buffer) is None
    with open(path, "rb+") as fh:
        fh.truncate(fh.seek(0, 2) - 4)
    assert target_weights.read_safetensors_slices(path, ["f32", "f64"], get_buffer) is None


_KINDS = ("weight_packed", "weight_scale")
_EXPERT_BASE = "model.layers.0.block_sparse_moe.experts"


def _expert_tensor(expert: int, w: str, kind: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(100 * expert + 10 * int(w[1]) + _KINDS.index(kind))
    if kind == "weight_packed":
        return torch.randint(0, 256, (4, 8), generator=gen, dtype=torch.uint8)
    return torch.randint(0, 256, (4, 1), generator=gen, dtype=torch.uint8)


def _record_expert(experts, base, ckpt_expert_idx, slot_expert_idx, local_slot_id, get_tensor):
    prefix = f"{base}.{ckpt_expert_idx}"
    # A copy: the streamed views live in a buffer the next shard overwrites.
    experts.got[local_slot_id] = {
        f"{w}.{kind}": get_tensor(f"{prefix}.{w}.{kind}").clone()
        for w in ("w1", "w2", "w3")
        for kind in _KINDS
    }
    experts.loaded.add(local_slot_id)


def _write_expert_checkpoint(directory, with_index: bool = True):
    """Experts 0-7 of layer 0 in two shards, with expert 5's w2 in the other shard, under ``language_model.``."""
    shards = {"a.safetensors": {}, "b.safetensors": {}}
    for expert in range(8):
        for w in ("w1", "w2", "w3"):
            for kind in _KINDS:
                shard = (
                    "a.safetensors"
                    if (expert < 4) != (expert == 5 and w == "w2")
                    else "b.safetensors"
                )
                key = f"language_model.{_EXPERT_BASE}.{expert}.{w}.{kind}"
                shards[shard][key] = _expert_tensor(expert, w, kind)
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(directory / name))
        weight_map.update(dict.fromkeys(tensors, name))
    if with_index:
        import json

        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )


def _moe(target_weights, expert_ids, slots=None):
    spec = target_weights.ExpertCkptSpec(
        kinds=_KINDS, loader=_record_expert, loaded_slots_attr="loaded"
    )
    experts = SimpleNamespace(
        got={}, loaded=set(), expert_size_per_partition=len(expert_ids) if slots is None else slots
    )
    return SimpleNamespace(
        expert_ckpt_spec=spec,
        ckpt_expert_ids=list(expert_ids),
        local_expert_ids=list(expert_ids),
        routed_experts=experts,
    )


def _assert_loaded(moe):
    for slot, expert in enumerate(moe.ckpt_expert_ids):
        got = moe.routed_experts.got[slot]
        for w in ("w1", "w2", "w3"):
            for kind in _KINDS:
                assert torch.equal(got[f"{w}.{kind}"], _expert_tensor(expert, w, kind)), (
                    expert,
                    w,
                    kind,
                )


@pytest.mark.parametrize(
    "path", ["preadv", "mapped file", "lazy dict"], ids=["preadv", "safe_open", "no-index"]
)
def test_routed_experts_load_this_ranks_slots(target_weights, path, tmp_path, monkeypatch):
    """Experts 2, 3 and 6 stream per shard file, expert 5 reads its two shards; every slot gets its checkpoint
    tensors whichever path reads them."""
    _write_expert_checkpoint(tmp_path, with_index=path != "lazy dict")
    served = []
    if path == "mapped file":
        monkeypatch.setattr(target_weights, "read_safetensors_slices", lambda *args: None)
    elif path == "preadv":
        read = target_weights.read_safetensors_slices

        def counted_read(*args):
            tensors = read(*args)
            served.append(tensors is not None)
            return tensors

        monkeypatch.setattr(target_weights, "read_safetensors_slices", counted_read)
    moe = _moe(target_weights, [2, 3, 5, 6])
    llm = _llm([SimpleNamespace(is_moe=True, block_sparse_moe=moe)])
    with target_weights.open_checkpoint_weights(str(tmp_path)) as weights:
        lm_weights = target_weights.strip_prefix(weights, target_weights.LANG_PREFIX)
        target_weights._load_expert_slices(llm, lm_weights, [(moe, _EXPERT_BASE)])
    assert moe.routed_experts.loaded == {0, 1, 2, 3}
    _assert_loaded(moe)
    # Both shards with whole experts went through the range reader.
    assert served == ([True, True] if path == "preadv" else [])


def test_a_slot_left_empty_fails_the_load(target_weights, tmp_path):
    _write_expert_checkpoint(tmp_path)
    moe = _moe(target_weights, [2, 3], slots=3)
    llm = _llm([SimpleNamespace(is_moe=True, block_sparse_moe=moe)])
    with target_weights.open_checkpoint_weights(str(tmp_path)) as weights:
        lm_weights = target_weights.strip_prefix(weights, target_weights.LANG_PREFIX)
        with pytest.raises(RuntimeError, match=r"missing \[2\]"):
            target_weights._load_expert_slices(llm, lm_weights, [(moe, _EXPERT_BASE)])
