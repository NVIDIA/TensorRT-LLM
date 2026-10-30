# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Beam target's standard paths: registration, scale input, recipes, ``LLM()``.

``trtllm-serve <checkpoint> --config <recipe>`` and ``LLM(...)`` reach the target through three
names in the shared registries (``checkpoint_format: beam_nvfp4`` from
``_torch/models/checkpoints/beam``, the ``custom_tokenizer`` import path of ``BeamFastTokenizer``
and the ``beam_harmony_text`` reasoning parser of ``llmapi/reasoning_parser.py``) and one
environment setting (``TRTLLM_MODELING_V2_KV_CACHE_SCALES``). Generated tokenizers and scale
tables only; no checkpoint, no GPU.
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    weights,
)
from tensorrt_llm._torch.models.checkpoints.beam.checkpoint_loader import (
    BEAM_NVFP4_CHECKPOINT_FORMAT,
)
from tensorrt_llm._torch.models.checkpoints.beam.tokenizer import BeamFastTokenizer
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

pytestmark = pytest.mark.cpu_only

#: The recipes' ``custom_tokenizer`` import path.
_TOKENIZER = "tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer"
_ENV = weights.KV_CACHE_SCALES_ENV
_REPO = Path(__file__).resolve().parents[7]
_RECIPES = {
    mode: _REPO / f"examples/configs/curated/beam-nvfp4-dflash2-fp8kv-{mode}.yaml"
    for mode in ("aggregated", "disagg-ctx", "disagg-gen")
}
_TARGET_SCALES = {"layers": 52, "scales": {str(i): 1.0 for i in range(52)}}
_DRAFT_SCALES = {"layers": 5, "scales": {"0": 2.0, "1": 2.0, "2": 2.0, "3": 8.0, "4": 4.0}}


@pytest.fixture(autouse=True)
def _no_inherited_scale_table(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)


def _scale_file(path: Path, **sections) -> str:
    table = {"target": _TARGET_SCALES, "draft": _DRAFT_SCALES}
    table.update(sections)
    path.write_text(json.dumps({k: v for k, v in table.items() if v is not None}))
    return str(path)


def _tokenizer_dir(path: Path) -> str:
    """A checkpoint-like directory holding only a small fast tokenizer (tokenizer.json)."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["<unk>", "What", "is", "12", "plus", "7", "?"]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path / "tokenizer.json"))
    return str(path)


def _drafter_dir(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return str(path)


def _run_fresh(tmp_path: Path, source: str, env: dict) -> dict:
    """Run ``source`` in a fresh interpreter; it prints one JSON object as its last line."""
    script = tmp_path / "fresh_process.py"
    script.write_text(source)
    result = subprocess.run(  # nosec B603
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------
# Registration on the import paths of a fresh process
# ---------------------------------------------------------------------------------------------

_WORKER_PATH = """
import json
import sys

import tensorrt_llm._torch.pyexecutor.model_loader  # what a rank imports before the factories
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import (
    CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING, get_checkpoint_weight_loader, get_config_loader)
from tensorrt_llm._torch.pyexecutor.model_loader import _construct_checkpoint_loader
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

target = "tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4"
record = {
    "format": "beam_nvfp4" in CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING,
    "hf_weight_loader": get_checkpoint_weight_loader("beam_nvfp4") is HfWeightLoader,
    "hf_config_loader": get_config_loader("beam_nvfp4") is HfConfigLoader,
    "parser": "beam_harmony_text" in ReasoningParserFactory.keys(),
    "cli_imported": "tensorrt_llm.commands.serve" in sys.modules,
    "weights_before": target + ".weights" in sys.modules,
}
loader = _construct_checkpoint_loader(None, "beam_nvfp4")
record.update(
    loader=type(loader).__name__,
    format_on_instance=loader.checkpoint_format,
    weight_loader=type(loader.weight_loader).__name__,
    config_loader=type(loader.config_loader).__name__,
    sha256=None if loader.kv_cache_scales is None else loader.kv_cache_scales["sha256"],
    modeling_imported=target + ".modeling" in sys.modules,
)
print(json.dumps(record))
"""


def test_fresh_rank_import_path_resolves_the_checkpoint_format(tmp_path):
    """A rank resolves ``beam_nvfp4`` from the model loader's own imports, with the scale table
    of the environment setting, before any modeling code or the serving CLI is imported."""
    import hashlib

    scales = _scale_file(tmp_path / "scales.json")
    env = dict(os.environ, **{_ENV: scales})
    record = _run_fresh(tmp_path, _WORKER_PATH, env)
    assert record == {
        "format": True,
        "hf_weight_loader": True,
        "hf_config_loader": True,
        "parser": True,
        "cli_imported": False,
        "weights_before": False,
        "loader": "BeamNvfp4CheckpointLoader",
        "format_on_instance": "HF",
        "weight_loader": "HfWeightLoader",
        "config_loader": "HfConfigLoader",
        "sha256": hashlib.sha256(Path(scales).read_bytes()).hexdigest(),
        "modeling_imported": False,
    }
    env.pop(_ENV)
    assert _run_fresh(tmp_path, _WORKER_PATH, env)["sha256"] is None


_SERVER_PATH = """
import json
import sys

from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.tokenizer import load_custom_tokenizer

parser = "beam_harmony_text" in ReasoningParserFactory.keys()
tokenizer = load_custom_tokenizer({tokenizer!r}, {directory!r}, trust_remote_code=False)
print(json.dumps({{
    "parser": parser,
    "parser_class": type(ReasoningParserFactory.create_reasoning_parser("beam_harmony_text")).__name__,
    "tokenizer": type(tokenizer).__name__,
    "ids": tokenizer.encode("What is 12 plus 7 ?"),
    "model_loader_imported": "tensorrt_llm._torch.pyexecutor.model_loader" in sys.modules,
    "modeling_v2_imported": "tensorrt_llm._torch._experimental.modeling_v2" in sys.modules,
}}))
"""


def test_fresh_server_resolves_the_parser_and_tokenizer_from_shared_registries(tmp_path):
    """The server process has the parser as soon as it imports the parser registry, and loads
    the tokenizer from its ``custom_tokenizer`` path, without the model loader or ModelingV2."""
    source = _SERVER_PATH.format(tokenizer=_TOKENIZER, directory=_tokenizer_dir(tmp_path / "ckpt"))
    record = _run_fresh(tmp_path, source, dict(os.environ))
    assert record == {
        "parser": True,
        "parser_class": "BeamHarmonyTextParser",
        "tokenizer": "BeamFastTokenizer",
        "ids": [1, 2, 3, 4, 5, 6],
        "model_loader_imported": False,
        "modeling_v2_imported": False,
    }


def test_custom_tokenizer_is_the_concrete_fast_class(tmp_path):
    from transformers import PreTrainedTokenizerFast

    from tensorrt_llm.tokenizer import TokenizerBase, load_custom_tokenizer

    directory = _tokenizer_dir(tmp_path / "ckpt")
    with mock.patch("transformers.AutoTokenizer.from_pretrained") as auto:
        tokenizer = load_custom_tokenizer(_TOKENIZER, directory)
    auto.assert_not_called()
    assert isinstance(tokenizer, BeamFastTokenizer)
    assert isinstance(tokenizer, TokenizerBase)
    assert isinstance(tokenizer.tokenizer, PreTrainedTokenizerFast)
    copy = pickle.loads(pickle.dumps(tokenizer))
    assert type(copy) is BeamFastTokenizer
    assert copy.encode("What is 12 plus 7 ?") == tokenizer.encode("What is 12 plus 7 ?")
    with pytest.raises(ValueError, match="slow"):
        BeamFastTokenizer.from_pretrained(directory, use_fast=False)


# ---------------------------------------------------------------------------------------------
# TRTLLM_MODELING_V2_KV_CACHE_SCALES: read once, explicit instance wins, failures are visible
# ---------------------------------------------------------------------------------------------


def _counting_reader(monkeypatch):
    calls = []
    read = weights.read_kv_cache_scales

    def counted(path):
        calls.append(path)
        return read(path)

    monkeypatch.setattr(weights, "read_kv_cache_scales", counted)
    return calls


def test_environment_table_is_read_once_per_loader_and_travels_resolved(tmp_path, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.model_loader import _construct_checkpoint_loader

    calls = _counting_reader(monkeypatch)
    scales = _scale_file(tmp_path / "scales.json")
    monkeypatch.setenv(_ENV, scales)
    loader = _construct_checkpoint_loader(None, BEAM_NVFP4_CHECKPOINT_FORMAT)
    assert isinstance(loader, weights.BeamNvfp4CheckpointLoader)
    assert calls == [scales]
    table = loader.kv_cache_scales
    assert table["target"] == (1.0,) * 52 and table["draft"] == (2.0, 2.0, 2.0, 8.0, 4.0)
    monkeypatch.setenv(_ENV, _scale_file(tmp_path / "other.json", draft=None))
    copy = pickle.loads(pickle.dumps(loader))
    assert type(copy) is weights.BeamNvfp4CheckpointLoader
    assert copy.kv_cache_scales == table
    assert calls == [scales], "a worker's copy carries the resolved table"
    monkeypatch.delenv(_ENV)
    assert _construct_checkpoint_loader(None, BEAM_NVFP4_CHECKPOINT_FORMAT).kv_cache_scales is None


def test_explicit_loader_instance_wins_over_the_environment(tmp_path, monkeypatch):
    from tensorrt_llm._torch.pyexecutor.model_loader import _construct_checkpoint_loader
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    explicit = weights.read_kv_cache_scales(_scale_file(tmp_path / "explicit.json", draft=None))
    calls = _counting_reader(monkeypatch)
    monkeypatch.setenv(_ENV, _scale_file(tmp_path / "environment.json"))
    instance = weights.BeamNvfp4CheckpointLoader(kv_cache_scales=explicit)
    assert _construct_checkpoint_loader(instance, None) is instance
    assert instance.kv_cache_scales is explicit and calls == []
    assert weights.BeamNvfp4CheckpointLoader().kv_cache_scales is None and calls == []
    # The native precedence is unchanged: a format name and an instance together drop the instance.
    args = TorchLlmArgs(
        model="unused",
        skip_tokenizer_init=True,
        checkpoint_format=BEAM_NVFP4_CHECKPOINT_FORMAT,
        checkpoint_loader=instance,
    )
    assert args.checkpoint_loader is None and args.checkpoint_format == "beam_nvfp4"


@pytest.mark.parametrize(
    "contents,reason",
    [
        (None, "No such file"),
        ("{not json", "Expecting"),
        ("[1, 2]", "JSON object"),
        (json.dumps({"draft": _DRAFT_SCALES}), "no 'target' section"),
        (json.dumps({"target": {"layers": 52, "scales": {"0": 1.0}}}), "0..51"),
        (json.dumps({"target": _TARGET_SCALES, "draft": {"layers": 5}}), "'scales' map"),
        (json.dumps({"target": {"layers": 1, "scales": {"0": -1.0}}}), "> 0"),
    ],
    ids=["missing", "malformed", "not-object", "no-target", "partial", "bad-draft", "negative"],
)
def test_unusable_environment_table_fails_the_loader_build(tmp_path, monkeypatch, contents, reason):
    from tensorrt_llm._torch.pyexecutor.model_loader import _construct_checkpoint_loader

    path = tmp_path / "scales.json"
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setenv(_ENV, str(path))
    with pytest.raises(ValueError, match=reason) as raised:
        _construct_checkpoint_loader(None, BEAM_NVFP4_CHECKPOINT_FORMAT)
    assert str(raised.value).startswith(f"{_ENV}=")


def test_empty_environment_setting_is_unset(monkeypatch):
    monkeypatch.setenv(_ENV, "")
    assert weights.kv_cache_scales_from_environment() is None


class _Gather:
    """A stand-in for ``Distributed.get(mapping)`` recording the all-rank gather."""

    def __init__(self, peers=None):
        self.peers = peers
        self.calls = []

    def allgather(self, identity):
        self.calls.append(identity)
        return [identity] * 4 if self.peers is None else self.peers(identity)


def _target(algo, drafter=None):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (  # noqa: E501
        ModelingV2BeamFullModelNvfp4Sm103Tp4,
    )

    model = ModelingV2BeamFullModelNvfp4Sm103Tp4.__new__(ModelingV2BeamFullModelNvfp4Sm103Tp4)
    loads = []
    model.__dict__.update(
        kv_cache_quant_algo=algo,
        draft_model=drafter,
        loads=loads,
        load_checkpoint=lambda path, kv_cache_scales=None: loads.append((path, kv_cache_scales)),
    )
    return model


@pytest.fixture
def gather(monkeypatch):
    from tensorrt_llm._torch.distributed import Distributed

    stand_in = _Gather()
    monkeypatch.setattr(Distributed, "get", staticmethod(lambda mapping: stand_in))
    return stand_in


def _load(loader, model, path="/target"):
    return loader.load_weights(path, mapping=SimpleNamespace(), model=model)


def test_target_load_carries_the_agreed_table(tmp_path, gather):
    from tensorrt_llm.quantization.mode import QuantAlgo

    table = weights.read_kv_cache_scales(_scale_file(tmp_path / "scales.json"))
    loader = weights.BeamNvfp4CheckpointLoader(kv_cache_scales=table)
    model = _target(QuantAlgo.FP8, drafter=object())
    assert _load(loader, model, str(tmp_path)) == {}
    assert model.loads == [(str(tmp_path), table)] and loader.is_weights_preloaded()
    assert gather.calls == [(table["sha256"], table["target"], table["draft"])]
    bf16 = _target(None)
    _load(weights.BeamNvfp4CheckpointLoader(), bf16)
    assert bf16.loads == [("/target", None)] and gather.calls[-1] is None


@pytest.mark.parametrize(
    "algo,table,drafter,reason",
    [
        ("FP8", None, True, f"set {weights.KV_CACHE_SCALES_ENV}"),
        (None, "full", False, "the KV cache is BF16"),
        ("FP8", "target-only", True, "no 'draft' section"),
    ],
    ids=["fp8-without-table", "bf16-with-table", "drafter-without-draft-scales"],
)
def test_target_load_refuses_a_table_that_does_not_match_the_kv_cache(
    tmp_path, gather, algo, table, drafter, reason
):
    from tensorrt_llm.quantization.mode import QuantAlgo

    tables = {
        None: None,
        "full": weights.read_kv_cache_scales(_scale_file(tmp_path / "full.json")),
        "target-only": weights.read_kv_cache_scales(
            _scale_file(tmp_path / "target.json", draft=None)
        ),
    }
    model = _target(QuantAlgo.FP8 if algo else None, drafter=object() if drafter else None)
    loader = weights.BeamNvfp4CheckpointLoader(kv_cache_scales=tables[table])
    with pytest.raises(ValueError, match=reason):
        _load(loader, model)
    assert model.loads == [] and not loader.is_weights_preloaded()
    assert len(gather.calls) == 1, "every rank gathers before any rank refuses"


def test_target_load_refuses_ranks_with_different_tables(tmp_path, gather):
    from tensorrt_llm.quantization.mode import QuantAlgo

    table = weights.read_kv_cache_scales(_scale_file(tmp_path / "scales.json"))
    gather.peers = lambda identity: [identity, None, identity, identity]
    model = _target(QuantAlgo.FP8, drafter=object())
    with pytest.raises(ValueError, match="ranks hold different FP8 KV-cache scale tables"):
        _load(weights.BeamNvfp4CheckpointLoader(kv_cache_scales=table), model)
    assert model.loads == []


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_scale_settings_reach_spawned_ranks(tmp_path, monkeypatch):
    """The MPI session forwards both recipe settings to every spawned rank."""
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)
    scales = _scale_file(tmp_path / "scales.json")
    monkeypatch.setenv(_ENV, scales)
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")
    session = MpiPoolSession(n_workers=2, wait_shutdown=True)
    try:
        assert session.submit_sync(os.getenv, _ENV) == [scales, scales]
        assert session.submit_sync(os.getenv, "TRTLLM_SKIP_KV_CACHE_ESTIMATION") == ["1", "1"]
    finally:
        session.shutdown()


# ---------------------------------------------------------------------------------------------
# The curated recipes through the real trtllm-serve argument path
# ---------------------------------------------------------------------------------------------


def _materialize(recipe: Path, tmp_path: Path, drafter: str) -> str:
    text = recipe.read_text()
    assert yaml.safe_load(text)["speculative_config"]["speculative_model"] == "<DRAFTER_CHECKPOINT>"
    path = tmp_path / recipe.name
    path.write_text(text.replace("<DRAFTER_CHECKPOINT>", drafter))
    return str(path)


def _serve_llm_kwargs(model: str, config: str, *extra: str) -> dict:
    """The keyword arguments trtllm-serve hands to the LLM constructor (server mocked)."""
    from tensorrt_llm.commands.serve import main as serve_main

    llm = mock.Mock()
    with (
        mock.patch("tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=4),
        mock.patch("tensorrt_llm.commands.serve.PyTorchLLM", llm),
        mock.patch("tensorrt_llm.commands.serve.OpenAIServer"),
        mock.patch("tensorrt_llm.commands.serve.uvloop.run"),
    ):
        serve_main(
            args=[model, "--config", config, "--port", "18999", *extra], standalone_mode=False
        )
    llm.assert_called_once()
    return llm.call_args.kwargs


@pytest.fixture
def cuda_shape(monkeypatch):
    props = SimpleNamespace(major=10, minor=3)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda *a, **k: props)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (10, 3))


_COMMON = {
    "modeling_v2": "require",
    "checkpoint_format": "beam_nvfp4",
    "checkpoint_loader": None,
    "custom_tokenizer": _TOKENIZER,
    "reasoning_parser": "beam_harmony_text",
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1,
    "moe_tensor_parallel_size": 4,
    "moe_expert_parallel_size": 1,
    "enable_attention_dp": False,
    "attn_backend": "TRTLLM",
    "disable_overlap_scheduler": True,
    "enable_chunked_prefill": False,
    "cuda_graph_config": None,
    "max_batch_size": None,
    "max_seq_len": 69632,
    "max_num_tokens": 8192,
}
_KV = {
    "enable_block_reuse": True,
    "enable_partial_reuse": False,
    "use_kv_cache_manager_v2": True,
    "max_tokens": None,
    "dtype": "fp8",
}
# Qualified concurrency (max_batch_size) and KV-cache capacity (max_tokens) of each recipe.
_CAPACITY = {
    "aggregated": (64, 4456448),
    "disagg-ctx": (64, 4456448),
    "disagg-gen": (64, 4456448),
}


def _expected_settings(mode: str) -> dict:
    batch, max_tokens = _CAPACITY[mode]
    return {**_COMMON, "max_batch_size": batch, "kv": {**_KV, "max_tokens": max_tokens}}


def _effective(args) -> dict:
    spec = args.speculative_config
    transceiver = args.cache_transceiver_config
    return {
        **{name: getattr(args, name) for name in _COMMON},
        "kv": {name: getattr(args.kv_cache_config, name) for name in _KV},
        "spec": (
            type(spec).__name__,
            spec.max_draft_len,
            str(spec.speculative_model),
            spec.attention_backend,
        ),
        "transceiver": None
        if transceiver is None
        else (
            transceiver.backend,
            transceiver.transceiver_runtime,
            transceiver.kv_transfer_timeout_ms,
        ),
    }


@pytest.mark.parametrize("mode", sorted(_RECIPES))
def test_recipe_effective_settings_through_trtllm_serve(tmp_path, cuda_shape, mode):
    """Every pinned value, including explicit false/null, survives the CLI's defaults."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    model = _tokenizer_dir(tmp_path / "ckpt")
    drafter = _drafter_dir(tmp_path / "drafter")
    config = _materialize(_RECIPES[mode], tmp_path, drafter)
    kwargs = _serve_llm_kwargs(model, config)
    assert kwargs["cuda_graph_config"] is None and kwargs["enable_chunked_prefill"] is False
    args = TorchLlmArgs(**kwargs)
    assert isinstance(args.tokenizer, BeamFastTokenizer)
    disagg = mode != "aggregated"
    assert _effective(args) == {
        **_expected_settings(mode),
        "spec": ("DFlashDecodingConfig", 7, drafter, "TRTLLM"),
        "transceiver": ("NIXL", "PYTHON", 60000) if disagg else None,
    }


def test_explicit_cli_flags_win_over_the_recipe(tmp_path, cuda_shape):
    """Documented precedence: an explicit flag overrides its recipe value, nothing else."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    model = _tokenizer_dir(tmp_path / "ckpt")
    config = _materialize(_RECIPES["aggregated"], tmp_path, _drafter_dir(tmp_path / "drafter"))
    args = TorchLlmArgs(**_serve_llm_kwargs(model, config, "--max_batch_size", "4"))
    assert args.max_batch_size == 4
    assert args.cuda_graph_config is None and args.disable_overlap_scheduler
    assert args.kv_cache_config.dtype == "fp8" and not args.enable_chunked_prefill


def _documented_llm_kwargs(model: str, drafter: str, mode: str) -> dict:
    """The public ``LLM()`` arguments of the deployment guide's Python example for ``mode``."""
    from tensorrt_llm.llmapi import CacheTransceiverConfig, DFlashDecodingConfig, KvCacheConfig

    batch, max_tokens = _CAPACITY[mode]
    kwargs = dict(
        model=model,
        modeling_v2="require",
        checkpoint_format="beam_nvfp4",
        custom_tokenizer=_TOKENIZER,
        reasoning_parser="beam_harmony_text",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        enable_attention_dp=False,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        cuda_graph_config=None,
        max_batch_size=batch,
        max_seq_len=69632,
        max_num_tokens=8192,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            enable_partial_reuse=False,
            use_kv_cache_manager_v2=True,
            max_tokens=max_tokens,
            dtype="fp8",
        ),
        speculative_config=DFlashDecodingConfig(
            max_draft_len=7, speculative_model=drafter, attention_backend="TRTLLM"
        ),
    )
    if mode != "aggregated":
        kwargs["cache_transceiver_config"] = CacheTransceiverConfig(
            backend="NIXL", transceiver_runtime="PYTHON", kv_transfer_timeout_ms=60000
        )
    return kwargs


@pytest.mark.parametrize("mode", sorted(_RECIPES))
def test_documented_llm_api_example_matches_the_recipe(tmp_path, cuda_shape, mode):
    """The guide's public ``LLM()`` example and the recipe through ``trtllm-serve`` describe one
    deployment: the same registered names, engine settings, KV cache and speculation."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    model = _tokenizer_dir(tmp_path / "ckpt")
    drafter = _drafter_dir(tmp_path / "drafter")
    recipe = TorchLlmArgs(
        **_serve_llm_kwargs(model, _materialize(_RECIPES[mode], tmp_path, drafter))
    )
    example = TorchLlmArgs(**_documented_llm_kwargs(model, drafter, mode))
    assert _effective(example) == _effective(recipe)
    assert type(example.tokenizer) is type(recipe.tokenizer) is BeamFastTokenizer


def test_recipes_hold_only_public_placeholders():
    for path in _RECIPES.values():
        text = path.read_text()
        config = yaml.safe_load(text)
        assert config["speculative_config"]["speculative_model"] == "<DRAFTER_CHECKPOINT>"
        assert "/" not in config["speculative_config"]["speculative_model"]
        assert "<KV_SCALES>" in text and "<TARGET_CHECKPOINT>" in text
