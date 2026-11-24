# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine hooks the specialized Kimi K3.1 Rubin targets rely on, and each target's use of them.

The hook, on stand-ins: the model loader routes before it checks the KV cache quantization, and
what it writes into ``config.extra_attrs`` reaches the check and the copy the model is built from;
on sm_107 the check runs an NVFP4 KV cache of these models as FP8, as no specialized target
declares an NVFP4 cache of its own (``handles_nvfp4_kv_cache``).

Each specialized target (``kimi_k31_pro_nvfp4__sm_107__{dep16,helix8,tep8}``):

* ``attention_backend_for`` returns the family's DSA backend. Its metadata class, which every
  step and the KV cache estimation build, is the family's; it declares ``kv_lens_refresh_once``
  exactly where the target's launch constants ask for it. The engine checks that resolve the
  stock backend by name (the overlap KV-length guard, the attention workspace) get the same
  answers from the stock classes as from the family's;
* the family ships no FMHA library classes: no FMHA library serves a DSA layer (the CuTe DSL
  MLA library is unavailable on a sparse layer, TRTLLM-Gen declines sparse attention), every MLA
  layer of these targets is a DSA layer, and the flat targets call the sparse MLA themselves;
* the target's recipe is an FP8 KV cache, which the check keeps.
"""

import copy
import importlib
import inspect
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm import _utils as trtllm_utils
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor import model_loader
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_LAYOUTS = ("dep16", "helix8", "tep8")
# Classmethods the engine reads off the stock TRTLLM backend, resolved by name
# (pyexecutor/_util.py).
_WORKSPACE_DECLARATIONS = (
    "runtime_workspace_bytes_per_token",
    "runtime_workspace_is_chunked_prefill_bounded",
    "runtime_workspace_fixed_bytes",
)


def _report_sm(monkeypatch, sm):
    """Report a GPU of this SM version, also on a host without one (the KV cache check reads it)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trtllm_utils, "get_sm_version", lambda: sm)
    monkeypatch.setattr(model_loader, "get_sm_version", lambda: sm)


@pytest.fixture
def sm_107(monkeypatch):
    """Report a Rubin GPU."""
    _report_sm(monkeypatch, 107)


def _k31_config():
    """Kimi K3.1's composite config on 8 layers: MLA with a DSA indexer on layers 4 and 8
    (numbered from 1), KDA on the rest."""
    text = dict(
        model_type="kimi_linear",
        num_hidden_layers=8,
        hidden_size=7168,
        num_experts=896,
        num_experts_per_token=16,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        vocab_size=163840,
        linear_attn_config=dict(full_attn_layers=[4, 8], kda_layers=[1, 2, 3, 5, 6, 7]),
        dsa_config=dict(dsa_layers=[4, 8], index_topk=2048, index_num_heads=32, index_head_dim=128),
    )
    return PretrainedConfig(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=PretrainedConfig(**text),
    )


def _model_config(target=None):
    return ModelConfig(
        pretrained_config=_k31_config(),
        mapping=Mapping(),
        quant_config=QuantConfig(),
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype="fp8"),
        attn_backend="TRTLLM",
        modeling_v2_target=target,
    )


def _kv_algo(config):
    return QuantAlgo(config.quant_config.kv_cache_quant_algo)


# --- The KV cache check and the loader's route ---


class _Target:
    """A stand-in for the routed target."""


def test_on_sm_107_an_nvfp4_kv_cache_runs_as_fp8(sm_107):
    config = _model_config(_Target)
    model_loader.validate_and_set_kv_cache_quant(config, "nvfp4")
    assert _kv_algo(config) is QuantAlgo.FP8


def test_elsewhere_an_nvfp4_kv_cache_stays(monkeypatch):
    _report_sm(monkeypatch, 100)
    config = _model_config(_Target)
    model_loader.validate_and_set_kv_cache_quant(config, "nvfp4")
    assert _kv_algo(config) is QuantAlgo.NVFP4


class _Stop(Exception):
    pass


class _CheckpointLoader:
    """Hands the loader a ready model config."""

    def __init__(self, config):
        self._config = config

    def load_config(self, checkpoint_dir, **kwargs):
        return self._config


def _loader(llm_args):
    loader = object.__new__(model_loader.ModelLoader)
    loader.llm_args = llm_args
    loader.mapping = Mapping()
    loader.spec_config = None
    loader.sparse_attention_config = llm_args.sparse_attention_config
    loader.max_num_tokens = llm_args.max_num_tokens
    loader.max_seq_len = llm_args.max_seq_len
    loader.lora_config = None
    return loader


def test_the_loader_routes_before_the_kv_cache_check(monkeypatch, tmp_path):
    """The target is on the config the KV cache check gets, together with what the loader
    wrote into ``extra_attrs``, and the copy the model is built from keeps both."""
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="fp8"),
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype="fp8"),
        max_batch_size=12,
    )
    seen = {}

    def check(config, kv_cache_dtype):
        seen["config"] = config
        seen["kv_cache_dtype"] = kv_cache_dtype
        raise _Stop

    monkeypatch.setattr(model_loader, "modeling_v2_resolve", lambda config, args: _Target)
    monkeypatch.setattr(model_loader, "validate_and_set_kv_cache_quant", check)
    with pytest.raises(_Stop):
        _loader(llm_args)._load_and_validate_config(
            str(tmp_path), _CheckpointLoader(_model_config())
        )
    config = seen["config"]
    assert config.modeling_v2_target is _Target
    assert seen["kv_cache_dtype"] == "fp8"
    assert config.extra_attrs["kv_cache_dtype"] == "fp8"
    built_from = copy.deepcopy(config)  # ModelLoader.load builds the model from a deep copy
    assert built_from.modeling_v2_target is _Target
    assert built_from.extra_attrs == config.extra_attrs


# --- Each specialized target ---


def _import(name):
    """Import ``name``; skip only when this environment lacks a third-party dependency."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        missing = exc.name or ""
        if missing and not missing.startswith("tensorrt_llm"):
            pytest.skip(f"this environment lacks {missing}")
        raise


def _target(layout):
    """The target's modeling module and shell class."""
    package = f"kimi_k31_pro_nvfp4__sm_107__{layout}"
    module = _import(f"{_FAMILY}.{package}.modeling")
    shells = [
        cls
        for name, cls in vars(module).items()
        if name.startswith("ModelingV2")
        and inspect.isclass(cls)
        and cls.__module__ == module.__name__
    ]
    assert len(shells) == 1, f"{package}: expected one ModelingV2 shell class, got {shells}"
    return module, shells[0]


def _engine_sparse_params():
    """The DSA parameters the engine hands ``attention_backend_for`` (model_engine.py)."""
    config = _model_config()
    return config.sparse_attention_config.to_sparse_params(
        pretrained_config=config.pretrained_config
    )


def _backend(layout):
    """The backend class the target's ``attention_backend_for`` returns, asked on a shell that was
    not built: the answer comes from the target's constants."""
    module, shell = _target(layout)
    backend = shell.attention_backend_for(object.__new__(shell), "TRTLLM", _engine_sparse_params())
    return module, backend


def _refresh_once_expected(module):
    expected = {
        bool(module._CONSTANTS[role].get("kv_lens_refresh_once", False)) for role in module._ROLES
    }
    assert len(expected) == 1, "the roles of one target need one metadata class"
    return expected.pop()


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_the_target_runs_the_familys_dsa_backend_and_metadata(layout):
    module, backend = _backend(layout)
    assert inspect.isclass(backend) and issubclass(backend, DSATrtllmAttention)
    assert backend.__module__.startswith(f"{_FAMILY}.ops.attention.rubin_"), backend.__module__
    metadata = backend.Metadata
    assert issubclass(metadata, DSAtrtllmAttentionMetadata)
    assert metadata.__module__.startswith(f"{_FAMILY}.ops.attention.rubin_"), metadata.__module__

    # The class declares the refresh-once choice the target's constants ask for.
    expected = _refresh_once_expected(module)
    assert bool(getattr(metadata, "kv_lens_refresh_once", False)) is expected

    # Engine checks that resolve the stock backend by name see the same answers.
    stock = get_attention_backend("TRTLLM", sparse_params=_engine_sparse_params())
    hook = "apply_spec_decode_kv_lens_offsets"
    assert hasattr(metadata, hook) == hasattr(stock.Metadata, hook)
    dense = get_attention_backend("TRTLLM")
    for name in _WORKSPACE_DECLARATIONS:
        assert inspect.getattr_static(backend, name) is inspect.getattr_static(dense, name), name


def test_the_family_ships_no_fmha_library_classes():
    """No module of the family's attention ops defines an FMHA library class."""
    ops = importlib.import_module(f"{_FAMILY}.ops.attention")
    found = []
    for path in sorted(Path(ops.__file__).parent.glob("*.py")):
        module = _import(f"{ops.__name__}.{path.stem}")
        found += [
            f"{path.stem}.{name}"
            for name, cls in vars(module).items()
            if inspect.isclass(cls) and issubclass(cls, Fmha) and cls.__module__ == module.__name__
        ]
    assert not found, found


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_the_target_keeps_its_fp8_kv_cache(layout, sm_107):
    module, shell = _target(layout)
    assert not getattr(shell, "handles_nvfp4_kv_cache", False)
    config = _model_config(shell)
    model_loader.validate_and_set_kv_cache_quant(config, "fp8")
    assert _kv_algo(config) is QuantAlgo.FP8
    # Asked for NVFP4 (routing refuses that first), the loader would run FP8.
    config = _model_config(shell)
    model_loader.validate_and_set_kv_cache_quant(config, "nvfp4")
    assert _kv_algo(config) is QuantAlgo.FP8
