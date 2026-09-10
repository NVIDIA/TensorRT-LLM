# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vocab_size has to reach KVCacheManagerV2 whatever config the model ships.

V2 mints the synthetic ids of a multimodal block-reuse cache key above
``vocab_size``. ``_create_kv_cache_manager`` resolves it once and every branch
forwards that one value, so what is left to guard is narrow: the two branches
that rebuild the kwargs dict rather than spread it, the lookup order inside
``resolve_vocab_size``, and the error raised when nothing resolves.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.pyexecutor._util as _util
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager
from tensorrt_llm._torch.pyexecutor.config_utils import resolve_vocab_size
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

# Nemotron 3.5 Super VL's inner language model.
_VOCAB_SIZE = 131072


def _nemotron_hybrid_config():
    """Flat Nemotron-H text config: two mamba layers, two attention layers."""
    return SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        hybrid_override_pattern="M*M*",
        ssm_state_size=16,
        conv_kernel=4,
        mamba_num_heads=4,
        n_groups=1,
        mamba_head_dim=8,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        vocab_size=_VOCAB_SIZE,
        torch_dtype=torch.bfloat16,
    )


def _qwen3_5_hybrid_config():
    return SimpleNamespace(
        architectures=["Qwen3_5ForCausalLM"],
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_key_head_dim=8,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=4,
        linear_num_key_heads=1,
        linear_value_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        vocab_size=_VOCAB_SIZE,
        torch_dtype=torch.float16,
    )


def _dense_config():
    return SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        vocab_size=_VOCAB_SIZE,
        torch_dtype=torch.bfloat16,
    )


def _capture_manager_kwargs(pretrained_config, base_cls, kv_cache_config=None):
    """Route a config through _create_kv_cache_manager, capture the ctor kwargs."""
    captured: dict[str, object] = {}
    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(enable_block_reuse=True, use_kv_cache_manager_v2=True)

    class RecordingManager(base_cls):
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=kv_cache_config,
        tokens_per_block=64,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=SimpleNamespace(
            pretrained_config=pretrained_config,
            quant_config=None,
            sparse_attention_config=None,
        ),
        dtype=torch.bfloat16,
        is_draft=False,
    )
    return captured["kwargs"]


@pytest.mark.parametrize(
    "config_factory, base_cls",
    [
        # The hybrid branches copy the shared kwargs dict instead of spreading
        # it, once each, so both need a case of their own.
        pytest.param(_nemotron_hybrid_config, MambaHybridCacheManagerV2, id="nemotron_hybrid"),
        pytest.param(_qwen3_5_hybrid_config, MambaHybridCacheManagerV2, id="qwen3_5_hybrid"),
        # Stands in for every branch that spreads the dict directly: Kimi, MLA, dense.
        pytest.param(_dense_config, KVCacheManagerV2, id="dense"),
    ],
)
def test_branch_forwards_vocab_size_to_the_manager(config_factory, base_cls):
    kwargs = _capture_manager_kwargs(config_factory(), base_cls)
    assert kwargs["vocab_size"] == _VOCAB_SIZE


def test_v1_hybrid_manager_is_not_given_vocab_size():
    """The forwarding is gated on KVCacheManagerV2, and it has to stay that way.

    ``MixedMambaHybridCacheManager`` takes no ``**kwargs``, so lifting
    ``vocab_size`` out of that ``issubclass`` gate would not degrade to a
    silently ignored argument: it would raise TypeError while the manager is
    being built, before the model serves anything.
    """
    kwargs = _capture_manager_kwargs(
        _qwen3_5_hybrid_config(),
        MixedMambaHybridCacheManager,
        KvCacheConfig(enable_block_reuse=False, use_kv_cache_manager_v2=False),
    )
    assert "vocab_size" not in kwargs
    # The V1 managers pick their conv-state layout by model_type, so this also
    # pins that the V1 branch still gets its own kwargs.
    assert kwargs["model_type"] == "qwen3_next"


def test_util_resolves_vocab_size_instead_of_reading_the_attribute():
    """``llm_config`` is a name only resolve_vocab_size knows, so reaching it proves the call."""
    config = _nemotron_hybrid_config()
    config.vocab_size = None
    config.llm_config = SimpleNamespace(vocab_size=_VOCAB_SIZE)

    kwargs = _capture_manager_kwargs(config, MambaHybridCacheManagerV2)
    assert kwargs["vocab_size"] == _VOCAB_SIZE


def test_unresolvable_vocab_size_warns_when_block_reuse_is_on(monkeypatch: pytest.MonkeyPatch):
    """Startup says so once; the request that needs the value raises."""
    warnings: list[str] = []
    monkeypatch.setattr(
        _util.logger, "warning", lambda *msg: warnings.append(" ".join(map(str, msg)))
    )

    config = _dense_config()
    config.vocab_size = None

    kwargs = _capture_manager_kwargs(config, KVCacheManagerV2)
    assert kwargs["vocab_size"] is None
    assert any("vocab_size" in message for message in warnings)


def test_resolve_vocab_size_reads_a_text_config_attribute():
    """A non-HF config carries text_config plainly, with no get_text_config to find it."""
    config = SimpleNamespace(vocab_size=None, text_config=SimpleNamespace(vocab_size=_VOCAB_SIZE))
    assert resolve_vocab_size(config) == _VOCAB_SIZE


def test_resolve_vocab_size_reads_a_dict_valued_nested_config():
    """HyperCLOVAX stores language_config as a plain dict, not a config."""
    config = SimpleNamespace(vocab_size=None, language_config={"vocab_size": _VOCAB_SIZE})
    assert resolve_vocab_size(config) == _VOCAB_SIZE


def test_resolve_vocab_size_consults_get_text_config():
    """HF composite configs expose the text config through a method, not an attribute."""
    text_config = SimpleNamespace(vocab_size=_VOCAB_SIZE)
    config = SimpleNamespace(vocab_size=None, get_text_config=lambda: text_config)
    assert resolve_vocab_size(config) == _VOCAB_SIZE


def test_resolve_vocab_size_falls_back_when_get_text_config_yields_nothing():
    """get_text_config() knows nothing about llm_config, so the loop must still run."""
    config = SimpleNamespace(
        vocab_size=None,
        get_text_config=lambda: SimpleNamespace(vocab_size=None),
        llm_config=SimpleNamespace(vocab_size=_VOCAB_SIZE),
    )
    assert resolve_vocab_size(config) == _VOCAB_SIZE


def test_resolve_vocab_size_prefers_the_top_level_value_over_the_nested_loop():
    config = SimpleNamespace(vocab_size=_VOCAB_SIZE, text_config=SimpleNamespace(vocab_size=7))
    assert resolve_vocab_size(config) == _VOCAB_SIZE


def test_resolve_vocab_size_prefers_the_top_level_value_over_get_text_config():
    """Lowering the id floor would let synthetic multimodal ids collide with real ones."""
    config = SimpleNamespace(
        vocab_size=_VOCAB_SIZE,
        get_text_config=lambda: SimpleNamespace(vocab_size=7),
    )
    assert resolve_vocab_size(config) == _VOCAB_SIZE


def _multimodal_request():
    return SimpleNamespace(
        multimodal_hashes=[[1, 2, 3, 4]],
        multimodal_positions=[0],
        multimodal_lengths=[2],
    )


def _text_only_request():
    return SimpleNamespace(
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
    )


def test_multimodal_request_raises_when_vocab_size_is_unresolved():
    """Without this the None reaches the key generator as a TypeError."""
    manager = SimpleNamespace(vocab_size=None)

    with pytest.raises(ValueError, match="resolve_vocab_size"):
        KVCacheManagerV2._augment_tokens_for_block_reuse(
            manager, [11, 12, 13], _multimodal_request()
        )


def test_text_only_request_is_served_when_vocab_size_is_unresolved():
    """Only multimodal keys need the id floor; text-only serving must stay up."""
    manager = SimpleNamespace(vocab_size=None)
    tokens = [11, 12, 13]

    assert (
        KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, _text_only_request())
        == tokens
    )
