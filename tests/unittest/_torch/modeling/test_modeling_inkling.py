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
"""CPU-only structural tests for the Inkling text tower.

The config-and-geometry tests run anywhere; the weight-accounting and tensor-shape
tests read the checkpoint's safetensors *index* only (a few hundred KB of JSON, no
weights, no GPU) and skip when the checkpoint is not available. Together they
guarantee no required text tensor (q/k norm, relative bias, short conv,
route/global scale, unpadded logits) can silently go missing.
"""

import inspect
import json
import os
import struct
from types import SimpleNamespace
from unittest import mock

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.configs.inkling import InklingConfig, InklingTextConfig
from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
    inkling_account_checkpoint,
    inkling_nvfp4_expert_layers,
)

_models_root = llm_models_root()
CHECKPOINT = os.environ.get(
    "INKLING_CHECKPOINT", str(_models_root / "Inkling-NVFP4") if _models_root else ""
)

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT or not os.path.isdir(CHECKPOINT), reason="Inkling checkpoint not available"
)

# The checkpoint's hybrid attention pattern: every 6th layer at offset 5 is a
# global (full-causal) layer, the rest are local (sliding-window).
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]
LOCAL_LAYER_IDS = [n for n in range(66) if n not in GLOBAL_LAYERS]


def _index(ckpt: str) -> dict[str, str]:
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _is_quantized(ckpt: str) -> bool:
    """A checkpoint is quantized iff it ships hf_quant_config.json.

    The BF16 release does not, and its empty exclusion list means "nothing is
    quantized" rather than "everything is".
    """
    return os.path.isfile(os.path.join(ckpt, "hf_quant_config.json"))


def _exclude_modules(ckpt: str) -> set[str]:
    if not _is_quantized(ckpt):
        return set()
    with open(os.path.join(ckpt, "hf_quant_config.json")) as f:
        return set(json.load(f)["quantization"].get("exclude_modules", []))


def _safetensors_shape(ckpt: str, key: str) -> tuple[list[int], str]:
    """Read one tensor's shape/dtype from its shard header (no tensor data)."""
    shard = _index(ckpt)[key]
    with open(os.path.join(ckpt, shard), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header[key]["shape"], header[key]["dtype"]


@pytest.fixture(scope="module")
def ckpt_config() -> InklingTextConfig:
    """The real checkpoint's text sub-config (checkpoint-gated tests only)."""
    return InklingConfig.from_pretrained(CHECKPOINT).text_config


def test_model_and_config_are_registered():
    # Importing the model module registers the auto-model and the weight mapper.
    import tensorrt_llm._torch.models.modeling_inkling  # noqa: F401
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    assert "InklingForConditionalGeneration" in MODEL_CLASS_MAPPING
    cfg = InklingConfig()
    assert cfg.model_type == "inkling_mm_model"
    assert isinstance(cfg.text_config, InklingTextConfig)
    assert cfg.text_config.model_type == "inkling_text"


def test_model_defaults_pin_v2_and_disable_block_reuse():
    """Both defaults are correctness requirements, not tuning.

    V2 because the per-layer KV-head split (local 16 / global 8) needs
    per-layer geometry V1 cannot represent; block reuse off because the four
    depthwise short convs per layer hold a ``kernel_size - 1`` window outside
    the KV cache, and ``InklingConvRuntime`` seeds every context request
    with ``has_initial_state=False`` -- a reused prefix would restart the
    convolutions from zeros while attention resumed from real history.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration

    defaults = InklingForConditionalGeneration.get_model_defaults(None)

    assert defaults["kv_cache_config"]["use_kv_cache_manager_v2"] is True
    assert defaults["kv_cache_config"]["enable_block_reuse"] is False


# ---------------------------------------------------------------------------
# The two features that must be refused, not defaulted off.
#
# Both leave a CONTEXT request with history it should attend to and convolve
# against, and Inkling gets that wrong twice: _run_context attends only to the
# tokens of its own call (inkling_prefill_attention has no paged-KV argument),
# and InklingConvRuntime seeds every context request with
# has_initial_state=False. Neither raises on its own -- both emit wrong logits.
# ---------------------------------------------------------------------------
def test_disaggregated_serving_is_rejected():
    """The C++ transceiver route is already refused for every V2 manager, but
    the Python one (KvCacheTransceiverV2) is not. It would move the paged KV and
    silently leave the short-conv windows behind -- they are a plain torch pool
    with no page table -- so the generation instance would resume from zeros."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        reject_unsupported_inkling_kv_cache_features,
    )

    with pytest.raises(NotImplementedError, match="disaggregated"):
        reject_unsupported_inkling_kv_cache_features(
            InklingConfig(),
            enable_block_reuse=False,
            enable_cache_transceiver=True,
        )

    # Default off: an aggregate deployment pays nothing.
    reject_unsupported_inkling_kv_cache_features(InklingConfig(), enable_block_reuse=False)


def test_the_supported_configuration_is_accepted():
    """The default configuration must stay silent, including on the text
    sub-config the KV cache is sized from."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        reject_unsupported_inkling_kv_cache_features,
    )

    for cfg in (InklingConfig(), InklingTextConfig()):
        reject_unsupported_inkling_kv_cache_features(cfg, enable_block_reuse=False)


def test_the_rejection_is_scoped_to_inkling():
    """The guard sits on a shared path (KvCacheCreator picks the manager class
    for every model), so it must be inert for anything that is not Inkling."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        reject_unsupported_inkling_kv_cache_features,
    )

    not_inkling = SimpleNamespace(model_type="llama")
    reject_unsupported_inkling_kv_cache_features(not_inkling, enable_block_reuse=True)


def test_block_reuse_is_rejected_without_a_snapshot_policy():
    """An explicit enable_block_reuse=True wins the deep-merge over the model
    default, so the default alone cannot be the guarantee. Without a snapshot
    policy a prefix hit still has no conv window to restore."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        reject_unsupported_inkling_kv_cache_features,
    )

    with pytest.raises(NotImplementedError, match="block reuse"):
        reject_unsupported_inkling_kv_cache_features(InklingConfig(), enable_block_reuse=True)


def test_block_reuse_is_allowed_once_snapshots_are_configured():
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        reject_unsupported_inkling_kv_cache_features,
    )

    reject_unsupported_inkling_kv_cache_features(
        InklingConfig(), enable_block_reuse=True, periodic_snapshot_interval=256
    )


def _warnings_from_guard(**kwargs):
    """Run the guard and return what it warned.

    Captured off tensorrt_llm's logger, not with caplog: that logger sets
    propagate=False, so a caplog assertion would pass while asserting nothing.
    """
    from tensorrt_llm._torch.pyexecutor import config_utils

    said = []
    with mock.patch.object(config_utils.logger, "warning", said.append):
        config_utils.reject_unsupported_inkling_kv_cache_features(InklingConfig(), **kwargs)
    return " ".join(said)


def test_enabling_reuse_warns_that_it_is_text_only():
    """The manager's per-request warning needs a multimodal request to arrive,
    and rides on the same probe as the rule it describes."""
    assert "text prompts only" in _warnings_from_guard(
        enable_block_reuse=True, periodic_snapshot_interval=256
    )


def test_no_text_only_warning_when_reuse_is_off():
    """The negative control. A warning on every deployment, including the ones
    that never asked for reuse, is a warning operators learn to skip."""
    assert "text prompts only" not in _warnings_from_guard(enable_block_reuse=False)


def test_the_multimodal_probe_field_still_exists():
    """The guard reaches py_multimodal_data through getattr, so the NAME is the
    contract: a rename makes every multimodal request look like a text one and
    silently restores the defect, with nothing else failing."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    assert "py_multimodal_data" in inspect.getsource(LlmRequest.__init__)


def test_inkling_needs_block_aligned_chunks_without_being_hybrid_linear():
    """Folding Inkling into is_hybrid_linear would also route it through
    extract_mamba_kv_cache_params and the Mamba conv-state layouts, neither of
    which it can satisfy. The predicate names the property instead."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        is_hybrid_linear,
        needs_block_aligned_context_chunks,
    )

    cfg = InklingConfig()
    assert needs_block_aligned_context_chunks(cfg)
    assert not is_hybrid_linear(cfg)


# Snapshot points: where a window may be captured, so where a hit may land.


def _snapshot_manager(interval, reuse=True):
    """A stub: the method reads two attributes and writes one field per
    request, so a real manager would test the framework, not this policy."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )

    mgr = object.__new__(InklingHybridCacheManager)
    mgr.enable_block_reuse = reuse
    mgr._kv_cache_config = SimpleNamespace(
        mamba_state_config=SimpleNamespace(periodic_snapshot_interval=interval)
    )
    return mgr


def _snapshot_reqs(*prompt_lens):
    return [SimpleNamespace(prompt_len=n, expect_snapshot_points=None) for n in prompt_lens]


def test_snapshot_points_land_on_multiples_of_the_interval():
    reqs = _snapshot_reqs(700)
    _snapshot_manager(256).prepare_expect_snapshot_points(reqs)
    assert reqs[0].expect_snapshot_points == [256, 512]


def test_a_snapshot_point_never_exceeds_the_prompt():
    """Past prompt_len there is no prefix to key a snapshot by, and the
    scheduler would be asked to end a chunk beyond the request."""
    reqs = _snapshot_reqs(100, 256, 512)
    _snapshot_manager(256).prepare_expect_snapshot_points(reqs)
    assert reqs[0].expect_snapshot_points == []
    assert reqs[1].expect_snapshot_points == [256]
    assert reqs[2].expect_snapshot_points == [256, 512]


@pytest.mark.parametrize("interval,reuse", [(256, False), (0, True)])
def test_no_snapshot_points_when_reuse_cannot_run(interval, reuse):
    """The field is still assigned: the scheduler reads it unconditionally, and
    a leftover list from a previous batch would force chunk boundaries for a
    feature that is not running."""
    reqs = _snapshot_reqs(700)
    _snapshot_manager(interval, reuse=reuse).prepare_expect_snapshot_points(reqs)
    assert reqs[0].expect_snapshot_points == []


def _manager_with_caches(caches):
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )

    mgr = object.__new__(InklingHybridCacheManager)
    mgr.kv_cache_map = caches
    mgr._conv_layer_group_id = 4
    return mgr


def test_the_conv_slot_comes_from_v2_not_a_private_counter():
    """V2 restores a hit into the slot it assigned, so the pool must be indexed
    its way; a second numbering reads a row nobody restored."""
    seen = []

    class _Cache:
        def get_ssm_block_base_index(self, group_id):
            seen.append(group_id)
            return 3

    assert _manager_with_caches({7: _Cache()})._conv_slot_for_request(7) == 3
    assert seen == [4], "the conv layer group id must be the one queried"


def test_a_request_without_a_v2_slot_falls_back():
    """Padding sentinels have no cache entry, and V2 reports a negative index
    for a non-resident request; both must read as "no slot"."""

    class _NoBlock:
        def get_ssm_block_base_index(self, group_id):
            return -1

    mgr = _manager_with_caches({9: _NoBlock()})
    assert mgr._conv_slot_for_request(9) is None
    assert mgr._conv_slot_for_request(12345) is None


def test_the_pool_allocates_nothing_when_v2_resolves_slots():
    """The private free list must not run in parallel with V2's allocation:
    whichever row it handed out would be the wrong one."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling.conv_state import (
        InklingConvStateCache,
    )

    cache = object.__new__(InklingConvStateCache)
    cache._resolve_slot = {5: 2, 6: 0}.get
    cache._free = []  # empty: the private path would raise if it ran
    cache._slot_of = {}
    cache._padding_slot = 8
    cache._attention_dp_dummy_slot = None
    cache._max_draft_len = 0
    cache.num_request_slots = 8

    assert cache.slots_for([5, 6, 5]) == [2, 0, 2]
    assert cache._slot_of == {}, "nothing may be recorded in the private map"


def test_py_executor_finds_the_snapshot_hook_by_name():
    """py_executor reaches this through hasattr, so the NAME is the contract --
    a rename would silently stop forcing chunk boundaries rather than fail."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )

    assert hasattr(InklingHybridCacheManager, "prepare_expect_snapshot_points")


def test_the_snapshot_commit_methods_override_the_base():
    """The base commits as prefill advances and never sets history_length,
    which the SSM commit path asserts on. These overrides are what maintain
    that contract."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    for name in ("try_commit_blocks", "update_context_resources"):
        assert getattr(InklingHybridCacheManager, name) is not getattr(KVCacheManagerV2, name), (
            f"{name} no longer overrides the base"
        )
    assert hasattr(InklingHybridCacheManager, "_mark_context_position_as_history")


def test_the_snapshot_commit_methods_match_mambas():
    """A copy of MambaHybridCacheManagerV2's, so it can drift. Compared as
    ASTs: the two files use different formatters, and the error strings name
    the model."""
    import ast
    import textwrap

    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2

    def body(cls, name):
        fn = ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name)))).body[0]
        stmts = fn.body
        if (
            stmts
            and isinstance(stmts[0], ast.Expr)
            and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)
        ):
            stmts = stmts[1:]
        module = ast.Module(body=stmts, type_ignores=[])
        for node in ast.walk(module):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = "<msg>"
        return ast.dump(module)

    for name in (
        "try_commit_blocks",
        "update_context_resources",
        "_mark_context_position_as_history",
    ):
        assert body(InklingHybridCacheManager, name) == body(MambaHybridCacheManagerV2, name), (
            f"{name} has drifted from the Mamba copy it was taken from"
        )


# Multimodal requests must not share the radix tree: their image spans carry no
# content digest, measured as MMMU 81.3% -> 31.3% over the same 32 items.


def _reuse_manager(reuse=True):
    from tensorrt_llm._torch.attention_backend.sparse.inkling.cache_manager import (
        InklingHybridCacheManager,
    )

    mgr = object.__new__(InklingHybridCacheManager)
    mgr.enable_block_reuse = reuse
    mgr._warned_multimodal_reuse = False
    return mgr


def _mm_request(hashes=None, mm_data=None, request_id=7):
    return SimpleNamespace(
        multimodal_hashes=hashes, py_multimodal_data=mm_data, request_id=request_id
    )


@pytest.fixture
def passthrough_base(monkeypatch):
    """Make the base augmentation the identity, so what these assert on is the
    Inkling override alone rather than the multimodal rewrite underneath it."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    monkeypatch.setattr(
        KVCacheManagerV2,
        "_augment_tokens_for_block_reuse",
        lambda self, tokens, req, start=0, end=None: list(tokens),
    )


def test_multimodal_request_without_digests_gets_a_private_chain(passthrough_base):
    tokens = [1, 2, 3, 4]
    out = _reuse_manager()._augment_tokens_for_block_reuse(
        tokens, _mm_request(mm_data={"image": object()})
    )

    assert isinstance(out[0], bytes), "position 0 must stop matching real ids"
    assert out[1:] == tokens[1:], "only the first key token is salted"


def test_two_multimodal_requests_do_not_match_each_other(passthrough_base):
    """The failure this prevents is one image being served another's KV, so it
    is not enough that the chain is private -- two requests must differ."""
    mgr = _reuse_manager()
    first = mgr._augment_tokens_for_block_reuse(
        [1, 2, 3], _mm_request(mm_data={"image": object()}, request_id=1)
    )
    second = mgr._augment_tokens_for_block_reuse(
        [1, 2, 3], _mm_request(mm_data={"image": object()}, request_id=2)
    )

    assert first[0] != second[0]
    # ...and a continuation chunk of the SAME request keys the same way, or the
    # request could not reuse its own committed blocks.
    again = mgr._augment_tokens_for_block_reuse(
        [1, 2, 3], _mm_request(mm_data={"image": object()}, request_id=1)
    )
    assert again[0] == first[0]


def test_text_requests_keep_reuse(passthrough_base):
    """The negative control. Text is the path the end-to-end runs measured as
    good, and a salt applied there would silently switch the feature off."""
    tokens = [1, 2, 3]
    assert _reuse_manager()._augment_tokens_for_block_reuse(tokens, _mm_request()) == tokens


def test_multimodal_request_with_digests_is_left_to_the_base(passthrough_base):
    """Once hashes exist the base rewrite distinguishes items by content, which
    is strictly better than no reuse -- so the override must step aside."""
    tokens = [1, 2, 3]
    assert (
        _reuse_manager()._augment_tokens_for_block_reuse(
            tokens, _mm_request(hashes=[[1, 2, 3, 4]], mm_data={"image": object()})
        )
        == tokens
    )


def test_continuation_chunks_are_not_salted(passthrough_base):
    """The radix tree chains a block's key through its parent, so salting a
    later chunk would only change keys the first chunk already committed."""
    tokens = [5, 6, 7]
    assert (
        _reuse_manager()._augment_tokens_for_block_reuse(
            tokens, _mm_request(mm_data={"image": object()}), start=256
        )
        == tokens
    )


def test_nothing_is_salted_when_reuse_is_off(passthrough_base):
    tokens = [1, 2, 3]
    assert (
        _reuse_manager(reuse=False)._augment_tokens_for_block_reuse(
            tokens, _mm_request(mm_data={"image": object()})
        )
        == tokens
    )


def test_the_prefill_path_always_reads_the_pages():
    """One prefill kernel, and it takes the page table. A packed variant beside
    it would mean proving the two agree bit-for-bit at num_cached == 0."""
    import inspect

    from tensorrt_llm._torch.attention_backend.sparse.inkling import kernels

    params = inspect.signature(kernels.inkling_prefill_attention).parameters
    for name in ("k_cache", "v_cache", "num_cached", "page_table", "page_size"):
        assert name in params, f"prefill entry lost its paged argument {name}"
    assert not hasattr(kernels, "inkling_chunked_prefill_attention"), (
        "the second prefill kernel is gone; nothing should re-export it"
    )


def test_chunked_prefill_is_supported_and_no_longer_refused():
    """The raise came out once the feature was validated: bit-identical to
    one-shot prefill at the layer level, 63 GPU parity tests, and GSM8K over the
    full 1319 questions at -0.08 pt with exact McNemar p=1.000."""
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        reject_unsupported_inkling_kv_cache_features,
    )

    reject_unsupported_inkling_kv_cache_features(InklingConfig(), enable_block_reuse=False)


def test_text_geometry():
    """The config defaults are the checkpoint's text-tower geometry."""
    tc = InklingTextConfig()
    assert (tc.num_hidden_layers, tc.hidden_size, tc.head_dim) == (66, 6144, 128)
    assert (tc.num_attention_heads, tc.num_key_value_heads) == (64, 8)
    assert (tc.vocab_size, tc.unpadded_vocab_size) == (201024, 200058)
    assert tc.logits_mup_width_multiplier == 24.0
    assert tc.use_embed_norm is True
    assert (tc.n_routed_experts, tc.num_experts_per_tok, tc.n_shared_experts) == (256, 6, 2)
    assert tc.dense_mlp_idx == 2
    assert (tc.sliding_window_size, tc.swa_num_key_value_heads) == (512, 16)


def test_layer_classification():
    """Dense = {0, 1}; local/global geometry follows ``local_layer_ids``."""
    tc = InklingTextConfig(local_layer_ids=LOCAL_LAYER_IDS)
    assert [n for n in range(tc.num_hidden_layers) if tc.is_dense_layer(n)] == [0, 1]
    assert [n for n in range(tc.num_hidden_layers) if not tc.is_local_layer(n)] == GLOBAL_LAYERS
    # local: 16 kv-heads behind a 512 window; global: 8 kv-heads, no window.
    assert tc.layer_num_kv_heads(0) == 16 and tc.layer_window(0) == 512
    assert tc.layer_num_kv_heads(5) == 8 and tc.layer_window(5) is None
    # The paged KV cache is sized per layer from exactly this hybrid split.
    assert tc.num_kv_heads_per_layer() == [
        tc.layer_num_kv_heads(n) for n in range(tc.num_hidden_layers)
    ]


@requires_checkpoint
def test_text_weight_accounting(ckpt_config):
    """Every checkpoint key is consumed-text or intentionally deferred; the text
    tower is exactly and fully covered (nothing missing, nothing unaccounted)."""
    exclude = _exclude_modules(CHECKPOINT)
    all_keys = set(_index(CHECKPOINT))

    acct = inkling_account_checkpoint(
        all_keys, ckpt_config, exclude, quantized=_is_quantized(CHECKPOINT)
    )
    assert not acct["unaccounted"], sorted(acct["unaccounted"])[:10]
    assert not acct["missing"], sorted(acct["missing"])[:10]
    assert all(
        k.startswith(("model.audio.", "model.visual.", "model.mtp.")) for k in acct["deferred"]
    )
    assert len(acct["consumed_text"]) + len(acct["deferred"]) == len(all_keys)

    if _is_quantized(CHECKPOINT):
        # NVFP4 routed experts are exactly layers 3..65 (layer-2 experts are bf16).
        assert inkling_nvfp4_expert_layers(ckpt_config, exclude) == list(range(3, 66))
        assert "model.llm.layers.2.mlp.experts" in exclude
        assert "model.llm.layers.3.mlp.experts" not in exclude
    else:
        assert inkling_nvfp4_expert_layers(ckpt_config, exclude, quantized=False) == []


BF16_CHECKPOINT = os.environ.get(
    "INKLING_BF16_CHECKPOINT", str(_models_root / "Inkling-Small") if _models_root else ""
)

requires_bf16_checkpoint = pytest.mark.skipif(
    not BF16_CHECKPOINT or not os.path.isdir(BF16_CHECKPOINT),
    reason="Inkling BF16 checkpoint not available",
)


@requires_bf16_checkpoint
def test_bf16_text_weight_accounting():
    """The unquantized release accounts exactly, with no scale sidecars expected.

    Guards the failure mode the ``quantized`` flag exists for: an empty
    exclusion list read as "every layer is NVFP4" makes the expected-key set ask
    for ``.scale`` / ``.scale2`` / ``.input_amax`` / ``.original_shape`` on every
    routed-expert tensor, none of which a BF16 checkpoint ships.
    """
    config = InklingConfig.from_pretrained(BF16_CHECKPOINT).text_config
    all_keys = set(_index(BF16_CHECKPOINT))
    assert not _is_quantized(BF16_CHECKPOINT)

    acct = inkling_account_checkpoint(all_keys, config, set(), quantized=False)
    assert not acct["unaccounted"], sorted(acct["unaccounted"])[:10]
    assert not acct["missing"], sorted(acct["missing"])[:10]
    assert len(acct["consumed_text"]) + len(acct["deferred"]) == len(all_keys)

    # No routed-expert tensor carries a scale sidecar.
    assert not [k for k in all_keys if k.endswith((".scale", ".scale2", ".input_amax"))]


@requires_checkpoint
def test_checkpoint_tensor_shapes_match_geometry(ckpt_config):
    """Sampled checkpoint tensors have the shapes the modules construct."""
    tc = ckpt_config
    hd, hidden = tc.head_dim, tc.hidden_size

    # q/k/v/r projection out-dims (layer 0 is local: 16 kv-heads).
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wq_du.weight")[0] == [
        tc.num_attention_heads * hd,
        hidden,
    ]
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wk_dv.weight")[0] == [
        tc.swa_num_key_value_heads * hd,
        hidden,
    ]
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.5.attn.wk_dv.weight")[0] == [
        tc.num_key_value_heads * hd,
        hidden,
    ]  # global layer
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.wr_du.weight")[0] == [
        tc.num_attention_heads * tc.d_rel,
        hidden,
    ]

    # short-conv depthwise weight: [channels, 1, kernel].
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.0.attn.k_sconv.weight")[0] == [
        tc.swa_num_key_value_heads * hd,
        1,
        tc.sconv_kernel_size,
    ]

    # NVFP4 routed experts: [E, 2*inter, hidden/2] packed uint8 + block scale.
    shape, dtype = _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.experts.w13_weight")
    assert shape == [tc.n_routed_experts, 2 * tc.intermediate_size, hidden // 2]
    assert dtype in ("U8", "UINT8")
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.experts.w13_weight.scale")[0] == [
        tc.n_routed_experts,
        2 * tc.intermediate_size,
        hidden // 16,
    ]

    # The router covers routed + shared experts.
    assert _safetensors_shape(CHECKPOINT, "model.llm.layers.3.mlp.gate.weight")[0] == [
        tc.n_routed_experts + tc.n_shared_experts,
        hidden,
    ]


# ---------------------------------------------------------------------------
# Decode inputs: views into the stock TrtllmAttentionMetadata
# ---------------------------------------------------------------------------
#: Pages per logical slot, i.e. ``num_pool_layers * kv_factor``. Arbitrary here;
#: what matters is that it is > 1 and even, so a test that forgot to apply
#: ``page_div`` produces a visibly wrong page id rather than an off-by-a-bit.
_FAKE_INDEX_SCALE = 6


class _FakeKvManager:
    """Stand-in for the parts of KVCacheManagerV2 the decode metadata reads.

    Much smaller than the version this replaced, which had to fake
    ``get_batch_cache_indices`` because the metadata staged its own page table.
    The table is now a slice of the base ``kv_cache_block_offsets``, so all that
    is left to fake is the *static* layer -> row mapping and the encoding
    constants. ``calls`` stays so a test can assert the per-step path asks the
    manager for nothing at all.
    """

    kv_factor = 2
    enable_swa_scratch_reuse = False

    def __init__(
        self,
        pp_layers=(0, 1),
        layer_pools=None,
        index_scale=_FAKE_INDEX_SCALE,
        layer_scales=None,
        max_blocks_per_seq=None,
        batch_cache_indices=None,
    ):
        self.pp_layers = list(pp_layers)
        self.layer_offsets = {layer: i for i, layer in enumerate(self.pp_layers)}
        # Default: every layer its own pool, so a test that means to exercise
        # sharing has to say so.
        pools = layer_pools or {layer: i for i, layer in enumerate(self.pp_layers)}
        self.layer_to_pool_mapping_dict = {
            self.layer_offsets[layer]: pool for layer, pool in pools.items()
        }
        self.num_pools = len(set(self.layer_to_pool_mapping_dict.values()))
        self._index_scale = index_scale
        self.index_scales = [index_scale] * self.num_pools
        # Per-layer scale overrides: a layer absent here matches its pool (the
        # common case). A present, differing scale is a mixed-scale pool -- the
        # Inkling reality (global vs local geometry) the staging path exists for.
        self._layer_scales = dict(layer_scales or {})
        # Only the mixed-scale staging path reads these two.
        self.max_blocks_per_seq = max_blocks_per_seq
        self._batch_cache_indices = dict(batch_cache_indices or {})
        self.calls = []

    def get_layer_page_index_scale(self, layer_idx):
        self.calls.append(("scale", layer_idx))
        return self._layer_scales.get(layer_idx, self._index_scale)

    def get_batch_cache_indices(self, request_ids, layer_idx):
        # Block indices per request for this layer's *own* scale (already divided
        # by kv_factor), as the V2 manager returns them for the staging path.
        self.calls.append(("indices", layer_idx))
        return [list(self._batch_cache_indices[req_id]) for req_id in request_ids]


def _fake_block_offsets(num_pools, num_seqs, max_blocks, index_scale=_FAKE_INDEX_SCALE):
    """What ``copyBatchBlockOffsetsToDeviceKernel`` would have written.

    Plane 0 holds ``index_scale * base_page_index``, plane 1 the same plus
    ``kv_offset``; ``BAD_PAGE_INDEX`` entries are already 0 rather than negative
    (kvCacheManagerV2Utils.cu:224-227). Base pages are salted by pool and by
    sequence so reading the wrong row or the wrong plane is visible, and row
    ``seq`` owns ``seq + 1`` blocks so the padded tail is exercised.
    """
    import torch

    offs = torch.zeros((num_pools, num_seqs, 2, max_blocks), dtype=torch.int32)
    for pool in range(num_pools):
        for seq in range(num_seqs):
            n = min(seq + 1, max_blocks)
            base = torch.arange(1, n + 1, dtype=torch.int32) + 10 * pool + 100 * seq
            offs[pool, seq, 0, :n] = base * index_scale
            offs[pool, seq, 1, :n] = base * index_scale + 1
    return offs


class _FakeBuffers:
    """Stand-in for the captured-graph buffer cache. get_empty routes through
    get_buffer; handing back a fresh CPU tensor keeps the staging test CPU-only."""

    def get_buffer(self, tensor_shape, dtype, cache_name, capture_graph):
        import torch

        return torch.zeros(tensor_shape, dtype=dtype)


def _ptmod():
    from tensorrt_llm._torch.attention_backend.sparse.inkling import page_table

    return page_table


def _gen_page_table(md, layer):
    return _ptmod().gen_page_table(md, layer)


def _gen_seq_lens(md, num_gen):
    return _ptmod().gen_seq_lens(md, num_gen)


def _page_div(md):
    return _ptmod().page_div(md)


def _pt_row(md, layer):
    return _ptmod().pt_row(md, layer)


def _validate(md):
    """What the backend runs per layer before touching the borrowed buffers."""
    pt = _ptmod()
    num_gen = len(md.request_ids) - md.num_contexts
    for layer in pt.owned_layers(md):
        pt.validate_decode_layout(md, layer, num_gen)


def _ink_metadata(
    num_contexts=0,
    request_ids=(7, 9),
    num_cached=(3, 130),
    pp_layers=(0, 1),
    max_blocks_per_seq=4,
    layer_pools=None,
    max_num_sequences=None,
    index_scale=_FAKE_INDEX_SCALE,
    mgr=None,
):
    """An InklingAttentionMetadata with prepare()'s outputs stubbed in.

    Builds the object without running AttentionMetadata's dataclass __init__ so
    the test stays CPU-only and independent of the KV-cache stack; only the
    fields the Inkling page-table helpers read are populated.
    """
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingAttentionMetadata

    md = object.__new__(InklingAttentionMetadata)
    md.is_cuda_graph = False
    md.request_ids = list(request_ids)
    md._num_contexts = num_contexts
    # model_engine assigns _num_generations and request_ids from the same
    # scheduled batch, so the fake keeps them consistent -- the accessors slice
    # by num_generations now that the private counter is gone.
    md._num_generations = len(request_ids) - num_contexts
    md.kv_cache_params = SimpleNamespace(num_cached_tokens_per_seq=list(num_cached))
    # What TrtllmAttentionMetadata.prepare() would have published: total KV
    # length per request, num_cached + the one new generation token.
    md.kv_lens_cuda = torch.tensor([n + 1 for n in num_cached], dtype=torch.int32)
    md.kv_cache_manager = (
        mgr if mgr is not None else _FakeKvManager(pp_layers, layer_pools, index_scale)
    )
    # ... and the block offsets, the other buffer prepare() would have refreshed.
    md.kv_cache_block_offsets = _fake_block_offsets(
        md.kv_cache_manager.num_pools,
        max_num_sequences if max_num_sequences is not None else len(request_ids),
        max_blocks_per_seq,
        index_scale,
    )
    md._seq_lens_cuda = torch.zeros(1, dtype=torch.int32)
    return md


def test_metadata_reads_both_decode_inputs_from_the_base_buffers():
    """Neither seq lengths nor the page table are staged privately.

    ``kv_lens_cuda`` and ``kv_cache_block_offsets`` are both filled by
    ``TrtllmAttentionMetadata.prepare``, so the Inkling accessors slice them.
    The page table must come back as a *view* -- an equal-valued copy would
    reintroduce the per-step H2D this refactor deleted, and would go stale under
    CUDA-graph replay.
    """
    md = _ink_metadata()

    _validate(md)

    # num_cached + 1, straight off the base buffer.
    assert _gen_seq_lens(md, 2).tolist() == [4, 131]
    pt = _gen_page_table(md, 0)
    # Same storage as the base tensor: a view, not a copy.
    assert pt.untyped_storage().data_ptr() == md.kv_cache_block_offsets.untyped_storage().data_ptr()
    # Plane 0 of pool 0, generation rows. Encoded values, not block indices --
    # the kernel divides by page_div.
    assert pt.tolist() == md.kv_cache_block_offsets[0, 0:2, 0].tolist()
    assert pt.tolist() == [[1 * 6, 0, 0, 0], [101 * 6, 102 * 6, 0, 0]]


def test_metadata_page_div_recovers_the_block_index():
    """The one arithmetic difference from the base encoding, stated as a test.

    Entries count pages (K and V separately); the kernel's K/V views are the two
    planes of a ``[blocks, kv_factor, ...]`` buffer and count blocks. So
    ``entry // page_div`` must be ``base_page_index * num_pool_layers``, and
    ``page_div`` must be the manager's ``kv_factor`` -- not a hardcoded 2.
    """
    md = _ink_metadata()
    _validate(md)

    assert _page_div(md) == md.kv_cache_manager.kv_factor == 2
    pt = _gen_page_table(md, 0)
    # index_scale=6, kv_factor=2 -> block index is base_page_index * 3.
    assert (pt // _page_div(md)).tolist() == [[3, 0, 0, 0], [303, 306, 0, 0]]

    md.kv_cache_manager.kv_factor = 1
    assert _page_div(md) == 1


def test_metadata_stages_no_page_table_of_its_own():
    """Regression on the reason for this refactor: the previous version built a
    pinned host table in a Python loop and issued one H2D per KV geometry, every
    decode step, on the host-bound path. This is the scale-matched borrow path
    every layer takes unless its scale disagrees with its pool (the mixed-scale
    layers that still stage are covered separately)."""
    md = _ink_metadata(pp_layers=(0, 1), layer_pools={0: 0, 1: 0})

    _validate(md)

    assert not hasattr(md, "_ink_pt_host")
    assert not hasattr(md, "ink_page_table")
    # Only the cheap row-mapping constants are read; get_batch_cache_indices --
    # the accessor the H2D path needed -- is never reached.
    assert {kind for kind, _ in md.kv_cache_manager.calls} == {"scale"}


def test_metadata_shares_one_page_table_row_per_kv_geometry():
    """Layers of one KV geometry land in one pool, and the base tensor's leading
    axis is per-pool -- so the deduplication an earlier version did by hand (a
    ``(pool_id, index_scale)`` group key, one staged table per group) now comes
    for free from the borrowed layout. Inkling has two geometries and 66 layers.
    """
    md = _ink_metadata(
        pp_layers=(0, 1, 2, 3),
        # Inkling's shape: local layers in one pool, global layers in another.
        layer_pools={0: 0, 1: 0, 2: 1, 3: 1},
    )

    _validate(md)

    assert {layer: _pt_row(md, layer) for layer in (0, 1, 2, 3)} == {0: 0, 1: 0, 2: 1, 3: 1}
    # Layers sharing a pool address the same memory...
    assert _gen_page_table(md, 0).data_ptr() == _gen_page_table(md, 1).data_ptr()
    assert _gen_page_table(md, 2).data_ptr() == _gen_page_table(md, 3).data_ptr()
    # ...and the two geometries do not.
    assert _gen_page_table(md, 0).data_ptr() != _gen_page_table(md, 2).data_ptr()
    assert _gen_page_table(md, 0).tolist() != _gen_page_table(md, 2).tolist()


def test_metadata_rows_are_per_layer_under_swa_scratch_reuse():
    """``num_attention_op_pools`` is per-pool by default but per attention-op
    *layer* when scratch reuse is on, and the copy then keys rows by
    ``layer_offsets``. Reading a pool id into a layer-keyed tensor would silently
    return another layer's pages, so the mode has to be honoured."""
    mgr = _FakeKvManager(pp_layers=(0, 1, 2, 3), layer_pools={0: 0, 1: 0, 2: 1, 3: 1})
    mgr.enable_swa_scratch_reuse = True
    md = _ink_metadata(pp_layers=(0, 1, 2, 3), mgr=mgr, max_num_sequences=2)
    # One row per layer now, not per pool.
    md.kv_cache_block_offsets = _fake_block_offsets(4, 2, 4)

    _validate(md)

    assert {layer: _pt_row(md, layer) for layer in (0, 1, 2, 3)} == {0: 0, 1: 1, 2: 2, 3: 3}
    # Every layer distinct -- no sharing in this mode.
    ptrs = {_gen_page_table(md, layer).data_ptr() for layer in (0, 1, 2, 3)}
    assert len(ptrs) == 4


def test_metadata_skips_the_context_slice():
    """Only generation rows get decode metadata; context rows run the prefill
    kernel and are excluded by num_contexts."""
    md = _ink_metadata(num_contexts=1, request_ids=(7, 9), num_cached=(0, 130))

    _validate(md)

    assert _gen_seq_lens(md, 1).tolist() == [131]
    # Row 1 of the base tensor, not row 0 -- the context request's row is skipped.
    assert _gen_page_table(md, 0).tolist() == md.kv_cache_block_offsets[0, 1:2, 0].tolist()


def test_metadata_yields_no_generation_rows_for_a_context_only_batch():
    """A prefill-only step must not expose the previous step's rows.

    This used to be enforced by a private counter reset at the top of prepare.
    It now falls out of the borrowed layout: the accessors slice by the base's
    num_generations, which the framework sets to 0 for a context-only batch, so
    there is no stale value to leak. Asserting the resulting slice is empty is
    the property the counter was there to provide.
    """
    md = _ink_metadata()
    _validate(md)
    assert _gen_page_table(md, 0).shape[0] == 2

    # Same object, now a context-only batch. model_engine sets both of these
    # from one scheduled batch, so the fake moves both.
    md._num_contexts = 2
    md._num_generations = 0
    _validate(md)

    assert _gen_page_table(md, 0).shape[0] == 0
    assert _gen_seq_lens(md, 0).shape[0] == 0


def test_metadata_refuses_a_cuda_graph_batch_carrying_contexts():
    """The captured kernel reads both borrowed buffers at a fixed generation
    offset, so that offset has to be constant across replays. Decode graphs are
    pure generation, which makes it 0."""
    md = _ink_metadata(num_contexts=1, request_ids=(7, 9), num_cached=(0, 130))
    md.is_cuda_graph = True

    with pytest.raises(RuntimeError, match="context requests"):
        _validate(md)


def test_metadata_rejects_a_batch_wider_than_the_borrowed_block_offsets():
    """Borrowing the base tensor means inheriting its bounds. A batch past
    max_num_sequences must raise, not slice short and silently attend over
    another request's pages."""
    md = _ink_metadata(request_ids=(7, 9), num_cached=(3, 130), max_num_sequences=2)
    md.request_ids = [7, 9, 11]
    md.kv_cache_params = SimpleNamespace(num_cached_tokens_per_seq=[3, 130, 5])

    with pytest.raises(RuntimeError, match="sequence rows"):
        _validate(md)


def test_metadata_rejects_a_layer_whose_scale_disagrees_with_its_pool():
    """A scale-mismatched layer must never fall back to the borrowed pool row.

    The C++ copy encodes with the pool-level index_scale, while
    ``get_layer_page_index_scale`` documents that layers in one pool may differ --
    and for Inkling they *do* (global 8 kv-head layers share a pool with local
    16 kv-head ones). ``prepare`` stages such layers a private table; this is the
    backstop for when validation reaches one that was never staged -- prepare did
    not run, or did not own the layer. Borrowing the pool row there recovers a
    wrong page address rather than crashing, so it is rejected loudly."""
    md = _ink_metadata(pp_layers=(0, 1), layer_pools={0: 0, 1: 0})
    md.kv_cache_manager.index_scales = [_FAKE_INDEX_SCALE * 2]

    with pytest.raises(RuntimeError, match="page-index scale"):
        _validate(md)


def test_metadata_stages_a_private_table_for_a_scale_mismatched_layer():
    """The Inkling case the borrow cannot serve, and the whole reason the staging
    helper exists.

    Layer 1's page-index scale differs from its pool's shared row (main's V2
    manager co-locates global- and local-geometry layers in one pool). prepare()
    must stage that layer a private table built from its *own* scale via
    get_batch_cache_indices -- already block indices, so divisor 1 -- while the
    scale-matched layer 0 keeps the zero-copy pool borrow (divisor kv_factor)."""
    pt = _ptmod()

    # Layer 0 matches pool 0's scale; layer 1's own scale is doubled, so it does
    # not, exactly like an Inkling global layer sharing a local-encoded pool.
    mgr = _FakeKvManager(
        pp_layers=(0, 1),
        layer_pools={0: 0, 1: 0},
        layer_scales={1: _FAKE_INDEX_SCALE * 2},
        max_blocks_per_seq=4,
        batch_cache_indices={7: [4, 5], 9: [6, 7, 8]},
    )
    md = _ink_metadata(
        pp_layers=(0, 1),
        request_ids=(7, 9),
        num_cached=(3, 130),
        layer_pools={0: 0, 1: 0},
        mgr=mgr,
        max_num_sequences=2,
    )
    md.max_num_sequences = 2
    md.cuda_graph_buffers = _FakeBuffers()

    md._stage_scale_fixed_page_tables()

    # Only the mismatched layer is staged; the matched one still borrows.
    assert set(md._scale_fixed_page_tables) == {1}
    assert not pt.uses_pool_row(md, 1) and pt.uses_pool_row(md, 0)
    # It was built from the layer's own block indices, zero-padded to width.
    assert md._scale_fixed_page_tables[1].tolist() == [[4, 5, 0, 0], [6, 7, 8, 0]]
    assert ("indices", 1) in mgr.calls

    # decode_page_table routes each layer correctly.
    fixed_pt, fixed_div = pt.decode_page_table(md, 1, 2)
    assert fixed_div == 1
    assert fixed_pt.tolist() == [[4, 5, 0, 0], [6, 7, 8, 0]]

    borrowed_pt, borrowed_div = pt.decode_page_table(md, 0, 2)
    assert borrowed_div == mgr.kv_factor
    # A view of the base offsets, not a private copy.
    assert (
        borrowed_pt.untyped_storage().data_ptr()
        == md.kv_cache_block_offsets.untyped_storage().data_ptr()
    )


def test_metadata_rejects_a_missing_block_offsets_tensor():
    """The page table is no longer this class's to allocate, so its absence is a
    setup error to report rather than something to paper over."""
    md = _ink_metadata()
    md.kv_cache_block_offsets = None

    with pytest.raises(RuntimeError, match="kv_cache_block_offsets"):
        _validate(md)


def test_backend_and_cache_manager_both_resolve_from_the_sparse_registry():
    """One selection path for every consumer.

    Inkling populates ``sparse_attention_config`` from the checkpoint
    architecture, so the registry answers for the module layer AND for the
    consumers that hold no ModelConfig. Both entries are required: without the
    cache-manager one, ``get_kv_cache_manager_cls`` takes its sparse branch and
    raises "Unsupported sparse attention algorithm" at engine startup.
    """
    from tensorrt_llm._torch.attention_backend.sparse import (
        get_sparse_attn_kv_cache_manager,
        get_trtllm_sparse_attn_attention_backend,
    )
    from tensorrt_llm._torch.attention_backend.sparse.inkling import (
        InklingAttentionMetadata,
        InklingHybridCacheManager,
        InklingSparseAttentionConfig,
        InklingSparseParams,
        InklingTritonAttention,
    )

    params = InklingSparseParams()
    assert get_trtllm_sparse_attn_attention_backend(params) is InklingTritonAttention
    assert InklingTritonAttention.Metadata is InklingAttentionMetadata
    assert (
        get_sparse_attn_kv_cache_manager(InklingSparseAttentionConfig())
        is InklingHybridCacheManager
    )


def test_the_fake_config_stays_out_of_the_user_facing_schema():
    """The config exists to reach sparse/registry.py, not to give users a knob.

    Adding it to the SparseAttentionConfig union in llmapi/llm_args.py would
    change an API-stability snapshot, stale the golden manifest and need
    telemetry sign-off -- for a field with one legal value. It is a standalone
    BaseModel instead, and ModelConfig never validates against the union.

    It must still be a BaseModel, because modules/attention.py calls
    .model_dump() on whatever sits in that field.

    Note what is asserted and what is NOT. "Out of the union" is the property
    that matters; "not a subclass of the interface" is a different property, and
    an earlier version of this test asserted the latter as if it implied the
    former. It does not, the union is explicitly enumerated, and forbidding
    inheritance is what let the config ship without ``to_sparse_metadata_params``
    -- a base method the engine calls at executor init. Subclassing is now
    required, and :func:`test_the_fake_config_implements_the_whole_interface`
    covers why.
    """
    import inspect
    import typing

    from pydantic import BaseModel

    from tensorrt_llm._torch.attention_backend.sparse.inkling import (
        InklingSparseAttentionConfig,
        InklingSparseParams,
    )
    from tensorrt_llm.llmapi import llm_args as la

    cfg = InklingSparseAttentionConfig()
    assert isinstance(cfg, BaseModel)
    assert cfg.model_dump() == {"algorithm": "inkling"}
    assert isinstance(cfg.to_sparse_params(), InklingSparseParams)

    # The actual invariant: absent from the user-facing discriminated union, so
    # no API-stability snapshot, golden manifest or telemetry review is touched.
    members = typing.get_args(typing.get_args(la.SparseAttentionConfig)[0])
    assert InklingSparseAttentionConfig not in members
    assert "inkling" not in inspect.getsource(la).lower()


def test_the_fake_config_implements_the_whole_interface():
    """Every public method the engine may call must exist, not just today's set.

    This is the regression test for the failure that cost an end-to-end run: the
    config duck-typed ``BaseSparseAttentionConfig`` and was missing
    ``to_sparse_metadata_params``, which ``model_engine._set_up_attn_metadata``
    calls unconditionally on a non-None ``sparse_attention_config``. It surfaced
    as ``AttributeError`` during executor init, i.e. only end to end.

    Asserting the subclass relation rather than enumerating method names is the
    point: an enumeration would go stale exactly the way the imitation did.
    """
    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingSparseAttentionConfig
    from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig

    assert issubclass(InklingSparseAttentionConfig, BaseSparseAttentionConfig)

    cfg = InklingSparseAttentionConfig()
    # Spot-check the two the engine calls, including the one that was missing.
    assert cfg.to_sparse_metadata_params(pretrained_config=None) is None
    assert cfg.algorithm == "inkling"
    # Nothing on the interface may be left unbound.
    missing = [
        name
        for name in dir(BaseSparseAttentionConfig)
        if not name.startswith("_") and not hasattr(cfg, name)
    ]
    assert missing == [], missing


def test_the_config_is_injected_from_the_checkpoint_architecture():
    """Never user-supplied: derived in from_pretrained, before the instance is
    frozen (sparse_attention_config is not on __setattr__'s allow-list)."""
    import inspect

    from tensorrt_llm._torch import model_config as mc

    assert "InklingForCausalLM" in mc._INKLING_ARCHITECTURES
    assert "InklingForConditionalGeneration" in mc._INKLING_ARCHITECTURES
    src = inspect.getsource(mc.ModelConfig.from_pretrained)
    assert "_INKLING_ARCHITECTURES" in src
    # Injected into kwargs before cls(...) -- assignment afterwards would hit the
    # frozen guard.
    assert src.index("_INKLING_ARCHITECTURES") < src.index("model_config._frozen = True")


def test_rel_logits_rides_the_registered_backend_args_slot():
    """rel_logits is [num_query_tokens, heads, rel_extent] -- content-dependent,
    with a leading per-query axis. AttentionForwardArgs.relative_attention_bias
    is T5's [num_heads, num_buckets] table broadcast across the batch, so it
    cannot carry this; sparse_backend_args is the registered slot for exactly
    this kind of model-specific input."""
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import (
        InklingBackendForwardArgs,
        inkling_forward_args,
    )

    rel = torch.zeros(3, 2, 5)
    args = inkling_forward_args(rel, allow_mixed=True)
    assert isinstance(args.sparse_backend_args, InklingBackendForwardArgs)
    assert args.sparse_backend_args.rel_logits is rel
    assert args.sparse_backend_args.allow_mixed is True
    # The shared T5 field stays untouched -- it has no room for the query axis.
    assert args.relative_attention_bias is None
    # allow_mixed is REQUIRED, not defaulted. It certifies that the short-conv
    # state pool is active for this forward; a caller that forgets it would get
    # a mixed context+generation batch convolved across the packed boundary,
    # which is silently wrong output rather than an error. There is one call
    # site and it must state which case it is in, so the keyword has no default
    # even though the dataclass field does.
    with pytest.raises(TypeError, match="allow_mixed"):
        inkling_forward_args(rel)


def test_backend_forward_refuses_foreign_backend_args():
    """The standard forward() signature means anything can call it; a plain
    AttentionForwardArgs carries no rel_logits, and silently attending without
    the bias would produce plausible, wrong text."""
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingTritonAttention

    backend = object.__new__(InklingTritonAttention)  # no CUDA needed for this guard
    with pytest.raises(TypeError, match="sparse_backend_args"):
        backend.forward(None, None, None, None, forward_args=AttentionForwardArgs())


def test_model_defaults_do_not_pin_an_attn_backend():
    """Selection comes from the architecture-derived sparse_attention_config, so
    the default carries no backend name; a family override is caught by
    _assert_inkling_attn_backend."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration

    defaults = InklingForConditionalGeneration.get_model_defaults(None)
    assert "attn_backend" not in defaults
    assert defaults["kv_cache_config"]["use_kv_cache_manager_v2"] is True


def test_attn_backend_family_override_fails_loudly():
    """The VANILLA and FLASHINFER registries do not know the "inkling" algorithm
    and would surface an error naming neither Inkling nor the setting."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    check = InklingForCausalLM._assert_inkling_attn_backend
    check(SimpleNamespace(attn_backend="TRTLLM"))  # the family Inkling runs under
    check(SimpleNamespace(attn_backend="trtllm"))  # case-insensitive
    check(SimpleNamespace(attn_backend=None))  # unset -> framework default

    for bad in ("FLASHINFER", "VANILLA", "FLASHINFER_STAR_ATTENTION"):
        with pytest.raises(ValueError, match="TRTLLM attention backend family"):
            check(SimpleNamespace(attn_backend=bad))


def test_conv_pool_is_owned_by_the_kv_cache_manager():
    """The pool must be part of the cache manager, not a resource manager beside
    it: that is what frees the conv row with the request's KV blocks and lets
    the model reach it through attn_metadata.kv_cache_manager."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingHybridCacheManager
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    assert issubclass(InklingHybridCacheManager, KVCacheManagerV2)
    assert not getattr(InklingHybridCacheManager, "__abstractmethods__", frozenset())
    # free_resources must be overridden, or conv rows outlive their KV blocks.
    assert InklingHybridCacheManager.free_resources is not KVCacheManagerV2.free_resources


def test_inkling_selects_the_hybrid_cache_manager():
    """The manager comes from sparse/registry.py, not from a branch in
    _non_hybrid_kv_cache_manager_cls.

    get_kv_cache_manager_cls takes its sparse branch first (Inkling populates
    sparse_attention_config), so a branch in the non-hybrid helper would be dead
    code that disagrees with the registry the day one of them changes. What that
    helper must still do is force V2: an explicit use_kv_cache_manager_v2=False
    would otherwise mis-size the per-layer pool, and this runs before the model
    (and therefore its get_model_defaults) exists.
    """
    import inspect

    from tensorrt_llm._torch.attention_backend.sparse import get_sparse_attn_kv_cache_manager
    from tensorrt_llm._torch.attention_backend.sparse.inkling import (
        InklingHybridCacheManager,
        InklingSparseAttentionConfig,
    )
    from tensorrt_llm._torch.pyexecutor import _util
    from tensorrt_llm._torch.pyexecutor._util import _non_hybrid_kv_cache_manager_cls
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    assert (
        get_sparse_attn_kv_cache_manager(InklingSparseAttentionConfig())
        is InklingHybridCacheManager
    )

    # The sparse branch runs before the non-hybrid helper is ever consulted.
    outer = inspect.getsource(_util.get_kv_cache_manager_cls)
    assert outer.index("get_sparse_attn_kv_cache_manager") < outer.index(
        "_non_hybrid_kv_cache_manager_cls"
    )

    # V2 backstop survives: is_inkling still forces V2 even though the manager
    # itself now comes from the registry.
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    cfg = SimpleNamespace(model_type="inkling_text")
    assert _non_hybrid_kv_cache_manager_cls(cfg, KvCacheConfig()) is KVCacheManagerV2

    # And it returns no Inkling manager of its own. Comments are stripped: the
    # helper *mentions* the class to explain where the routing went.
    src = inspect.getsource(_non_hybrid_kv_cache_manager_cls)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "InklingHybridCacheManager" not in code


def test_conv_pool_dtype_is_a_torch_dtype_not_the_kv_cache_dtype():
    """_create_kv_cache_manager passes dtype= as the KV cache dtype, a C++
    tensorrt_llm.bindings.DataType. Feeding that to torch.zeros raises
    "invalid combination of arguments" and kills the server at startup, so the
    conv pool must take its dtype from the model config instead."""
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as csm

    src = inspect.getsource(csm.InklingHybridCacheManager.__init__)
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert 'kwargs["dtype"]' not in code and 'kwargs.get("dtype")' not in code

    # A torch dtype wins; HF's string spelling of the same thing resolves to it.
    assert csm._resolve_conv_dtype(SimpleNamespace(torch_dtype=torch.float16)) is torch.float16
    assert csm._resolve_conv_dtype(SimpleNamespace(torch_dtype="float16")) is torch.float16
    # The multimodal config defers to its text sub-config.
    assert (
        csm._resolve_conv_dtype(SimpleNamespace(text_config=SimpleNamespace(torch_dtype="float16")))
        is torch.float16
    )


def test_conv_pool_dtype_refuses_to_guess():
    """An unresolvable torch_dtype must raise here, not fall back to bfloat16.

    The pool holds pre-conv activations, so a silent bf16 default on an fp16
    release builds the whole pool in the wrong dtype. Nothing checks it until
    causal_conv1d_update reports a dtype mismatch, layers deep and with no
    mention of the config field that caused it."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as csm

    for cfg in (SimpleNamespace(torch_dtype="not-a-dtype"), SimpleNamespace()):
        with pytest.raises(ValueError, match="torch_dtype"):
            csm._resolve_conv_dtype(cfg)


def test_prepare_publishes_the_pool_rows_and_nothing_else():
    """The metadata subclass overrides ``prepare`` and adds exactly one helper.

    prepare() does host->device writes into buffers the captured decode graph
    aliases, so they run every step outside that region: the short-conv slot
    write, and -- via ``_stage_scale_fixed_page_tables`` -- a private page table
    for any layer whose page-index scale differs from its pool's shared row.
    Inkling needs the second because main's V2 manager co-locates its 55 local
    (16 kv-head) and 11 global (8 kv-head) attention layers in one pool, so one
    geometry cannot borrow the other's scale. Everything else the decode path
    needs is still a view of the base's buffers, so the subclass adds no further
    method or field -- bound the surface here so a future addition that should
    have been a base-buffer view is caught."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingAttentionMetadata
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

    pool = _conv_pool(tp_size=1, num_request_slots=4)
    md = object.__new__(InklingAttentionMetadata)
    md.kv_cache_manager = SimpleNamespace(conv_state_cache=pool)
    md.request_ids = [7, 9]

    # prepare() plus the mixed-scale staging helper, and nothing else.
    assert set(vars(InklingAttentionMetadata)) - set(vars(TrtllmAttentionMetadata)) <= {
        "prepare",
        "_stage_scale_fixed_page_tables",
        "__doc__",
        "__module__",
        "__qualname__",
    }

    with mock.patch.object(TrtllmAttentionMetadata, "prepare", lambda self: None):
        md.prepare()

    assert pool.state_indices[:2].tolist() == pool.slots_for([7, 9])


def test_prepare_leaves_a_plain_kv_manager_alone():
    """A manager with no conv pool gets no publish -- other models pay nothing."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingAttentionMetadata
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

    md = object.__new__(InklingAttentionMetadata)
    md.kv_cache_manager = SimpleNamespace()
    md.request_ids = [7, 9]

    with mock.patch.object(TrtllmAttentionMetadata, "prepare", lambda self: None):
        md.prepare()  # must not raise


def test_the_conv_runtime_slices_the_published_rows():
    """Built inside model.forward, so the pool rows must come back as a view of
    the buffer prepare() already refreshed -- copying them would go stale under
    CUDA-graph replay."""
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingConvRuntime

    pool = _conv_pool(tp_size=1, num_request_slots=4)
    md = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(conv_state_cache=pool),
        request_ids=[7, 9, 11, 13],
        num_contexts=2,
        seq_lens=torch.tensor([3, 4, 1, 1], dtype=torch.int32),
    )
    pool.write_state_indices(md.request_ids)

    rt = InklingConvRuntime.from_metadata(md)

    assert rt.num_ctx_tokens == 7
    assert rt.ctx_indices.tolist() == pool.state_indices[:2].tolist()
    assert rt.gen_indices.tolist() == pool.state_indices[2:4].tolist()
    # Views of the pool's own buffer, not copies.
    for view in (rt.ctx_indices, rt.gen_indices):
        assert view.untyped_storage().data_ptr() == pool.state_indices.untyped_storage().data_ptr()
    # Varlen offsets over the context prefix only, and no carried conv window.
    assert rt.query_start_loc.tolist() == [0, 3, 7]
    assert rt.has_initial_state.tolist() == [False, False]


def test_the_conv_runtime_is_absent_without_a_pool():
    """No pool on the manager is the stateless short-conv path."""
    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingConvRuntime

    assert InklingConvRuntime.from_metadata(SimpleNamespace(kv_cache_manager=None)) is None
    assert InklingConvRuntime.from_metadata(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# Phase 0 for DP / EP: the per-rank token counts must reach the fused MoE
# ---------------------------------------------------------------------------
def _decoder_layer_stub(is_moe: bool):
    """An InklingDecoderLayer with only the mlp dispatch wired up."""
    import torch.nn as nn

    from tensorrt_llm._torch.models.modeling_inkling import InklingDecoderLayer, InklingMoE

    layer = object.__new__(InklingDecoderLayer)
    # nn.Module.__setattr__ refuses submodule assignment until Module.__init__
    # has run, and object.__new__ skips it.
    nn.Module.__init__(layer)
    seen = {}

    if is_moe:

        class _Moe(InklingMoE):
            def __init__(self):
                pass

            def __call__(self, hidden_states, all_rank_num_tokens=None):
                seen["arnt"] = all_rank_num_tokens
                return hidden_states

        layer.mlp = _Moe()
    else:

        class _Dense:
            def __call__(self, hidden_states):
                seen["called"] = True
                return hidden_states

        layer.mlp = _Dense()
    return layer, seen


def test_moe_layers_receive_the_per_rank_token_counts():
    """FusedMoE sets use_dp from mapping.enable_attention_dp and needs this list
    to pad and gather across ranks; without it a DP or EP-with-DP layout cannot
    know how much each peer contributed."""
    layer, seen = _decoder_layer_stub(is_moe=True)

    layer._run_mlp("H", [4, 7, 4, 5])

    assert seen["arnt"] == [4, 7, 4, 5]


def test_dense_layers_are_not_handed_the_token_counts():
    """Layers 0 and 1 are InklingDenseMLP, whose forward takes activations only;
    passing the DP list would be a TypeError."""
    layer, seen = _decoder_layer_stub(is_moe=False)

    layer._run_mlp("H", [4, 7, 4, 5])

    assert seen.get("called") is True


def test_non_dp_passes_none_and_keeps_the_old_call_shape():
    """Single-rank / pure-TP has no all_rank_num_tokens on attn_metadata, and
    the expert call must be exactly what it was before this plumbing."""
    layer, seen = _decoder_layer_stub(is_moe=True)

    layer._run_mlp("H", None)

    assert seen["arnt"] is None


# ---------------------------------------------------------------------------
# Phase 1: expert parallelism
# ---------------------------------------------------------------------------
def _ep_model_config(ep_size, n_routed=256, world_size=4, use_cuda_graph=True):
    """Mapping derives moe_tp_size = world_size // moe_ep_size, so the stub must
    too. The only expert-parallel guard left is divisibility -- ep_size must
    divide the routed-expert count -- so moe_tp_size and use_cuda_graph are
    carried only to document the layout, not because the check reads them."""
    return SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_size=ep_size, moe_tp_size=max(1, world_size // ep_size)),
        use_cuda_graph=use_cuda_graph,
        pretrained_config=SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=n_routed)),
    )


@pytest.mark.parametrize("ep_size", [1, 2])
def test_expert_parallel_accepts_the_measured_layouts(ep_size):
    """On 4 GPUs these are the layouts actually run: both reproduce the TP-only
    GSM8K result per item (acc 0.9667, zero score flips)."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    InklingForCausalLM._assert_inkling_moe_parallel(_ep_model_config(ep_size))


@pytest.mark.parametrize("use_cuda_graph", [True, False])
def test_expert_parallel_accepts_whole_width_experts(use_cuda_graph):
    """ep_size 4 on 4 GPUs leaves moe_tp_size 1 (whole-width experts). With and
    without CUDA graphs this reproduces the TP-only GSM8K result per item (acc
    0.9667, zero score flips). The pure-EP + CUDA-graph capture segfault that
    once made this combination raise is fixed (verified end to end at tp_size=4,
    moe_ep_size=4, --use_cuda_graph), so no layout is refused here."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    InklingForCausalLM._assert_inkling_moe_parallel(
        _ep_model_config(4, use_cuda_graph=use_cuda_graph)
    )


@pytest.mark.parametrize("ep_size", [3, 5, 6, 7, 24])
def test_expert_parallel_rejects_a_non_divisor(ep_size):
    """FusedMoE._supports_non_divisible_ep is opt-in and CUTLASS -- Inkling's
    only routed-expert backend -- does not opt in, so an uneven split would
    fail inside expert-slot bookkeeping instead of at load."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    with pytest.raises(ValueError, match="does not divide evenly"):
        InklingForCausalLM._assert_inkling_moe_parallel(_ep_model_config(ep_size))


def test_expert_parallel_rejects_more_ranks_than_experts():
    """Zero-expert ranks are unsupported by every MoE backend.

    This case is also non-divisible, so it exercises the check ORDER: the
    user needs to hear "more ranks than experts", not "pick a divisor of 4".
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    with pytest.raises(ValueError, match="exceeds"):
        InklingForCausalLM._assert_inkling_moe_parallel(_ep_model_config(8, n_routed=4))


def test_expert_parallel_guard_is_inert_when_ep_is_off():
    """ep_size == 1 is the default and is what every accuracy run measured; the
    guard must not constrain moe_tp_size."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    cfg = _ep_model_config(1)
    cfg.mapping.moe_tp_size = 3  # deliberately not a divisor
    InklingForCausalLM._assert_inkling_moe_parallel(cfg)


def test_every_deferred_import_in_the_inkling_package_resolves():
    """Deferred (function-body) imports must resolve, including their level.

    This is the check py_compile and plain module-import tests cannot do.
    cache_manager.py holds two ``from ...models.modeling_inkling import ...``
    inside method bodies -- written that way on purpose, to keep a module-level
    cycle back through pyexecutor from forming. Moving the file one directory
    deeper silently changed what ``..`` resolved to, and nothing caught it
    until a GPU run died with ModuleNotFoundError during KV-cache creation,
    because neither import executes until the manager is instantiated.

    Resolve every relative import in the package against the real module tree.
    """
    import ast
    import importlib.util
    import pathlib

    import tensorrt_llm

    root = pathlib.Path(tensorrt_llm.__file__).parent
    pkg_dir = root / "_torch" / "attention_backend" / "sparse" / "inkling"
    assert pkg_dir.is_dir(), pkg_dir

    unresolved = []
    checked = 0
    for path in sorted(pkg_dir.rglob("*.py")):
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            checked += 1
            target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            if importlib.util.find_spec(target) is None:
                unresolved.append(f"{path.name}:{node.lineno} -> {target}")

    assert checked >= 10, f"expected to inspect the package's imports, saw {checked}"
    assert not unresolved, unresolved


# ---------------------------------------------------------------------------
# Phase 2: attention data parallelism
#
# Under ADP every rank runs the FULL attention over its OWN requests, and only
# the routed experts stay sharded. So the rule for everything outside the
# experts is "replicate", and the failure mode when a tensor keeps the global TP
# split is not a slowdown -- it is either a shape mismatch against the base
# Attention (which already scoped itself to attention TP) or an all-reduce that
# sums activations belonging to different requests.
# ---------------------------------------------------------------------------
def _adp_mapping(adp, tp_size=4, pp_size=1):
    from tensorrt_llm.mapping import Mapping

    return Mapping(
        world_size=tp_size * pp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        rank=0,
        enable_attention_dp=adp,
    )


def test_short_conv_keeps_every_channel_under_attention_dp():
    """The k/v convs act on the k/v stream out of the fused qkv projection. Under
    ADP that projection keeps every kv head, so the convs must keep every
    channel -- a 1/tp slice here would silently convolve a quarter of the
    stream."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingShortConv

    sc = InklingShortConv(32, 4, mapping=_adp_mapping(True), tp_shard=False)

    assert sc.tp_size == 1
    assert sc.channels == 32 == sc.channels_full


def test_short_conv_still_shards_by_kv_head_under_pure_tp():
    """Regression guard for the ADP change: pure TP must keep the split that
    every Inkling accuracy run to date measured."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingShortConv

    sc = InklingShortConv(32, 4, mapping=_adp_mapping(False), tp_shard=True)

    assert sc.tp_size == 4
    assert sc.channels == 8
    assert sc.channels_full == 32


def _tiny_dense_mlp(adp, hidden=8, inter=16):
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_inkling import InklingDenseMLP

    cfg = SimpleNamespace(
        hidden_size=hidden, dense_intermediate_size=inter, torch_dtype=torch.bfloat16
    )
    return InklingDenseMLP(ModelConfig(pretrained_config=cfg, mapping=_adp_mapping(adp)))


class _StubAllReduce:
    """Stands in for the real collective so a 4-rank Linear can be built in a
    single-process pytest.

    ``AllReduce.__init__`` opens a communicator, which calls
    ``_validate_world_size`` and refuses a Mapping wider than the actual MPI
    world (1 here). Only the collective is stubbed -- the sharding arithmetic
    under test is Linear's own and runs untouched.
    """

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def no_collectives(monkeypatch):
    # Patch the DEFINING module, not linear.py: Linear.__init__ does
    # ``from ..distributed import AllReduce`` in its own body, so the name is
    # resolved fresh from tensorrt_llm._torch.distributed on every call and a
    # patch on linear.AllReduce would be silently ignored.
    from tensorrt_llm._torch import distributed as dist_mod

    monkeypatch.setattr(dist_mod, "AllReduce", _StubAllReduce)


def test_dense_mlp_is_replicated_under_attention_dp(no_collectives):
    """Layers 0/1 are dense. Under ADP the row-parallel down_proj would
    all-reduce its partial sum across the TP group -- but each rank's partial
    belongs to DIFFERENT requests, so the reduce would add unrelated tokens
    together. Replicating the weights is the only correct answer, and it is what
    DeepSeek-V3's _compute_mlp_tp_size does (returns 1 under ADP)."""
    mlp = _tiny_dense_mlp(adp=True)

    # tp_mode is what Linear.forward dispatches on: the ROW branch is the only
    # one that calls all_reduce, so None means no cross-rank reduce can happen.
    assert mlp.gate_up_proj.tp_mode is None
    assert mlp.down_proj.tp_mode is None
    # And the module must not believe it owns a shard of anything.
    assert mlp.down_proj.tp_size == 1
    # Full width on every rank: 2*inter rows out, hidden columns in.
    assert tuple(mlp.gate_up_proj.weight.shape) == (32, 8)
    assert tuple(mlp.down_proj.weight.shape) == (8, 16)


def test_dense_mlp_still_tp_shards_and_reduces_without_attention_dp(no_collectives):
    """Regression guard: pure TP must keep column/row parallelism AND the
    down_proj all-reduce. Dropping that reduce is the bug this pair is here to
    make impossible to introduce by accident."""
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    mlp = _tiny_dense_mlp(adp=False)

    assert mlp.gate_up_proj.tp_mode == TensorParallelMode.COLUMN
    assert mlp.down_proj.tp_mode == TensorParallelMode.ROW
    assert isinstance(mlp.down_proj.all_reduce, _StubAllReduce)
    assert tuple(mlp.gate_up_proj.weight.shape) == (8, 8)  # 2*16 / 4 ranks
    assert tuple(mlp.down_proj.weight.shape) == (8, 4)  # 16 / 4 ranks


def _conv_pool(tp_size, kv_heads=16, head_dim=8, num_request_slots=2, **kwargs):
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingConvStateCache

    cfg = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=8,
        sconv_kernel_size=4,
        layer_num_kv_heads=lambda i: kv_heads,
        layer_head_dim=lambda i: head_dim,
    )
    return InklingConvStateCache(
        cfg, tp_size, num_request_slots, torch.device("cpu"), torch.bfloat16, **kwargs
    )


def test_conv_pool_zeroes_a_row_even_when_the_allocator_hands_back_garbage():
    """V2 pool memory arrives uninitialized, unlike the torch.zeros this used to do.

    Backing the pool with V2 SSM-layer buffers moved allocation out of this class,
    and with it the free zero-fill. Correctness now rests on slots_for zeroing a
    row the first time it hands it out -- if that ever regresses, a fresh request
    convolves against whatever the allocator last left there, which is wrong
    output rather than a crash. Simulate a dirty allocator and assert the
    invariant directly.
    """
    import torch

    def dirty(_layer_idx, _role, state_shape):
        # num_slots is 2 request rows + 1 padding row here; hand back more than
        # needed so the slice in __init__ is exercised too.
        return torch.full((8, *state_shape), 7.0, dtype=torch.bfloat16)

    pool = _conv_pool(tp_size=1, num_request_slots=2, allocate=dirty)
    slot = pool.slots_for([11])[0]

    for layer in range(2):
        for buf in pool.layer_state(layer):
            assert float(buf[slot].abs().sum()) == 0.0, (layer, slot)

    # A row that was never handed out keeps the allocator's contents -- that is
    # fine, nothing reads it, and asserting otherwise would re-impose the
    # whole-pool zero-fill this change deliberately dropped.
    other = next(s for s in range(2) if s != slot)
    assert float(pool.layer_state(0).k[other].abs().sum()) > 0.0


def test_reserved_slot_count_is_defined_once_and_asked_for():
    """The pool and the manager must agree on the row count exactly: the manager
    declares a min-slots floor to V2, and a V2 buffer sized from a smaller number
    is indexed out of bounds by slots_for. Two independent copies of
    `1 + int(adp)` would drift silently, so the layout owner exposes it and the
    manager asks."""
    import inspect

    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingConvStateCache
    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm

    assert InklingConvStateCache.reserved_slot_count(reserve_attention_dp_slot=False) == 1
    assert InklingConvStateCache.reserved_slot_count(reserve_attention_dp_slot=True) == 2

    # The manager calls it rather than recomputing the formula.
    src = inspect.getsource(cm.InklingHybridCacheManager._num_reserved_conv_slots.fget)
    assert "reserved_slot_count(" in src
    assert "1 + int(" not in src, "the manager re-derived the count instead of asking"

    # And the pool actually sizes itself by it.
    for adp in (False, True):
        pool = _conv_pool(tp_size=1, num_request_slots=3, reserve_attention_dp_slot=adp)
        assert pool.num_slots == 3 + InklingConvStateCache.reserved_slot_count(
            reserve_attention_dp_slot=adp
        )


def test_conv_pool_refuses_an_allocator_that_returns_too_few_slots():
    """The row count is declared twice -- here and in the manager, which feeds V2
    the min-slots constraint. They must agree; a V2 buffer sized from a
    smaller count would be indexed out of bounds by slots_for."""
    import torch

    def too_small(_layer_idx, _role, state_shape):
        return torch.zeros((2, *state_shape), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="slots but the pool needs"):
        _conv_pool(tp_size=1, num_request_slots=4, allocate=too_small)


def test_conv_pool_k_v_width_follows_the_split_it_is_given():
    """The pool's k/v rows must be exactly as wide as the convs that write them.
    tp_size 1 (the ADP case) is full width; tp_size 4 is the TP slice.
    Row count is 2 request rows + the reserved CUDA-graph padding row."""
    assert tuple(_conv_pool(1).layer_state(0).k.shape) == (3, 128, 3)
    assert tuple(_conv_pool(4).layer_state(0).k.shape) == (3, 32, 3)


def test_conv_pool_attn_and_mlp_rows_are_never_split():
    """The post-attention / post-MLP convs run on the full residual stream and
    are replicated under both TP and ADP, so their width is hidden_size
    regardless of the split."""
    for tp in (1, 4):
        st = _conv_pool(tp).layer_state(0)
        assert tuple(st.attn.shape) == (3, 8, 3), tp
        assert tuple(st.mlp.shape) == (3, 8, 3), tp


def test_conv_pool_reserves_a_row_for_padding_and_one_for_attention_dp():
    """Reserved rows sit above the real ones and are not handed to requests.

    Charging a padding sentinel a real row silently shrinks the servable batch:
    cuda_graph_runner keeps one dummy id per runtime draft length, and the
    attention-DP idle dummy is a third id on top of those."""
    pool = _conv_pool(1, num_request_slots=2, reserve_attention_dp_slot=True)

    assert (pool.num_request_slots, pool.num_slots) == (2, 4)
    assert tuple(pool.layer_state(0).attn.shape) == (4, 8, 3)


def test_conv_pool_aliases_every_padding_sentinel_to_one_reserved_row():
    """All CUDA-graph sentinel ids -- one per runtime draft length -- share the
    padding row, and the attention-DP dummy has its own."""
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID

    pool = _conv_pool(1, num_request_slots=2, reserve_attention_dp_slot=True, max_draft_len=3)

    sentinels = [CUDA_GRAPH_DUMMY_REQUEST_ID - d for d in range(4)]
    assert pool.slots_for(sentinels) == [2, 2, 2, 2]
    assert pool.slots_for([ATTENTION_DP_DUMMY_REQUEST_ID]) == [3]
    # Real rows are untouched by all that padding: both requests get real rows,
    # handed out from the bottom of the pool.
    assert pool.slots_for([101, 102]) == [0, 1]


def test_conv_pool_raises_when_it_runs_out_instead_of_growing():
    """A grown pool reallocates every buffer, stranding the pointers a captured
    CUDA graph holds -- and a graph replaying against a freed buffer does not
    raise. Refuse loudly instead; the pool is sized to every sequence that can
    be resident at once, so exhaustion means a row leaked."""
    pool = _conv_pool(1, num_request_slots=2)
    pool.slots_for([1, 2])

    with pytest.raises(RuntimeError, match="out of rows"):
        pool.slots_for([3])

    assert not hasattr(pool, "_grow")


def test_conv_pool_keeps_a_requests_row_across_steps():
    """The carried short-conv window is the whole point of the pool: a request
    must keep its row until free(), and the row must be zeroed on first use."""
    pool = _conv_pool(1, num_request_slots=2)

    assert pool.slots_for([5]) == pool.slots_for([5])
    pool.layer_state(0).k[pool.slots_for([5])[0]].fill_(1.0)
    slot = pool.slots_for([5])[0]
    pool.free([5])
    pool.slots_for([6])  # reuses the row

    assert float(pool.layer_state(0).k[slot].abs().sum()) == 0.0


def test_cache_manager_sizes_the_conv_pool_by_attention_tp(monkeypatch):
    """The manager must hand the pool the ATTENTION TP, not the global one --
    the same rule KVCacheManagerV2 already applies to the paged pool. Passing
    mapping.tp_size under ADP would allocate quarter-width conv rows for
    full-width convs, which shows up as a shape error deep in the conv kernel
    rather than at load."""
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm

    seen = _patch_pool(monkeypatch, cm)
    # The manager now sizes the V2 SSM buffers itself, so it reads the conv
    # geometry off the config before the pool is built.
    cfg = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        sconv_kernel_size=4,
        hidden_size=8,
        num_hidden_layers=2,
        layer_num_kv_heads=lambda i: 2,
        layer_head_dim=lambda i: 4,
    )

    cm.InklingHybridCacheManager(
        pretrained_config=cfg, mapping=_adp_mapping(True), max_batch_size=8
    )
    assert seen["tp_size"] == 1, "under ADP the conv pool must be full width"

    cm.InklingHybridCacheManager(
        pretrained_config=cfg, mapping=_adp_mapping(False), max_batch_size=8
    )
    assert seen["tp_size"] == 4, "without ADP the conv pool must keep the TP slice"


def test_build_cache_config_appends_ssm_layers_for_the_conv_state(monkeypatch):
    """The conv rows are declared to V2 as SSM layers so their bytes land inside
    the same quota as the paged KV.

    They used to be a side torch allocation that no byte quota knew about,
    counted only because the throwaway manager built during capacity estimation
    also held one -- correct only while the estimation pool and the serving pool
    were exactly the same size, which nothing enforced.

    Appended, not interleaved: keeping every attention layer_id at its original
    index is what lets get_buffers, get_batch_cache_indices, and the
    layer_offsets -> layer_to_pool_mapping_dict route the metadata takes into the
    borrowed kv_cache_block_offsets all stay untouched. DeepSeek-V4 has to
    interleave because its KV is itself split across cache layers; Inkling's is
    not.
    """
    from dataclasses import dataclass, field

    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig

    _patch_pool(monkeypatch, cm)
    cfg = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        sconv_kernel_size=4,  # kwin = 3
        hidden_size=8,
        num_hidden_layers=2,
        layer_num_kv_heads=lambda i: 2,
        layer_head_dim=lambda i: 4,  # kv_dim = 8, /tp_size 1 -> 8
    )
    mgr = cm.InklingHybridCacheManager(
        pretrained_config=cfg, mapping=_adp_mapping(False, tp_size=1), max_batch_size=3
    )
    mgr.pp_layers = [0, 1]
    mgr.num_local_layers = 2

    @dataclass
    class _Cfg:
        layers: list
        constraints: list = field(default_factory=list)
        commit_min_snapshot: bool = False

    base = _Cfg(layers=[AttentionLayerConfig(layer_id=i, buffers=[]) for i in range(2)])
    out = mgr._build_cache_config(base)

    # Attention layers survive at their original ids; SSM layers follow.
    assert [type(layer) for layer in out.layers] == [
        AttentionLayerConfig,
        AttentionLayerConfig,
        SsmLayerConfig,
        SsmLayerConfig,
    ]
    assert [int(layer.layer_id) for layer in out.layers] == [0, 1, 2, 3]
    assert mgr._conv_layer_id(0) == 2 and mgr._conv_layer_id(1) == 3

    # One buffer per layer holding [k | v | attn | mlp], sized per request (not
    # per block): summed channels * kwin. One pool is what a state transfer
    # wants -- the section widths travel beside it, as Mamba's [x | B | C] do.
    ssm = out.layers[2]
    assert [b.role for b in ssm.buffers] == [cm.CONV_ROLE]
    itemsize = torch.empty((), dtype=torch.bfloat16).element_size()
    sections = [
        8 * 3 * itemsize,  # k, kv_dim
        8 * 3 * itemsize,  # v, kv_dim
        8 * 3 * itemsize,  # attn, hidden
        8 * 3 * itemsize,  # mlp, hidden
    ]
    assert [b.size for b in ssm.buffers] == [sum(sections)]
    assert mgr._conv_section_bytes(0) == sections
    # SsmLayerConfig forbids a per-block override -- these are not paged.
    assert all(b.tokens_per_block_override is None for b in ssm.buffers)

    # A min-slots floor, so the pool cannot be sized for fewer sequences than
    # the scheduler admits. Zero-capacity requests cost no attention pages.
    floor = out.constraints[-1]
    assert len(floor.kv_caches) == 3 + 1  # max_batch_size*pp_size + 1 padding row
    assert all(d.capacity == 0 for d in floor.kv_caches)

    # KVCacheManagerConfig hard-asserts this whenever an SSM layer is present.
    # Omitting it took a full serve job to surface (job 6244892 died in
    # _create_kv_cache_manager), because nothing on the CPU-only path builds a
    # real KVCacheManagerConfig.
    assert out.commit_min_snapshot is True


def _patch_pool(monkeypatch, cm):
    """Replace the pool with a recorder and the V2 base __init__ with a no-op,
    so the manager's sizing arithmetic can be read off without a GPU."""
    seen = {}

    from tensorrt_llm._torch.attention_backend.sparse.inkling import InklingConvStateCache

    class _FakePool:
        num_slots = 0

        # Delegated, not reimplemented: the manager asks the pool class
        # for this, and a hand-rolled copy in the double would let the real
        # formula drift without any test noticing.
        reserved_slot_count = InklingConvStateCache.reserved_slot_count

        def __init__(self, pretrained_config, tp_size, num_request_slots, device, dtype, **kwargs):
            seen.update(tp_size=tp_size, num_request_slots=num_request_slots, dtype=dtype, **kwargs)

        def conv_state_bytes(self):
            return 0

    def _fake_base_init(self, *args, **kwargs):
        # The manager derives its conv sizing from what the base resolves before
        # _build_cache_config runs, so the double has to provide exactly those.
        self.mapping = kwargs["mapping"]
        self.max_batch_size = kwargs["max_batch_size"]
        self.max_draft_len = getattr(kwargs.get("spec_config"), "max_draft_len", 0) or 0

    monkeypatch.setattr(cm.KVCacheManagerV2, "__init__", _fake_base_init)
    monkeypatch.setattr(cm, "InklingConvStateCache", _FakePool)
    return seen


def test_cache_manager_sizes_the_conv_pool_for_every_resident_sequence(monkeypatch):
    """One row per sequence that can be resident at once -- max_batch_size per
    pipeline stage, the count MambaHybridCacheManagerV2 calls
    _max_resident_sequences -- plus the reserved rows the pool adds itself.

    Sizing it at max_batch_size alone leaves a PP>1 or attention-DP deployment
    depending on pool growth, which is exactly what cannot happen once CUDA
    graphs have captured the pool's pointers."""
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm

    seen = _patch_pool(monkeypatch, cm)
    # The manager now sizes the V2 SSM buffers itself, so it reads the conv
    # geometry off the config before the pool is built.
    cfg = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        sconv_kernel_size=4,
        hidden_size=8,
        num_hidden_layers=2,
        layer_num_kv_heads=lambda i: 2,
        layer_head_dim=lambda i: 4,
    )

    cm.InklingHybridCacheManager(
        pretrained_config=cfg,
        mapping=_adp_mapping(False, pp_size=2),
        max_batch_size=8,
    )

    assert seen["num_request_slots"] == 16
    assert seen["reserve_attention_dp_slot"] is False

    cm.InklingHybridCacheManager(
        pretrained_config=cfg, mapping=_adp_mapping(True), max_batch_size=8
    )
    assert seen["reserve_attention_dp_slot"] is True


def test_cache_manager_requires_its_three_pool_arguments_by_name(monkeypatch):
    """Omitting one used to raise a bare KeyError from inside the constructor,
    because they were read back out of **kwargs."""
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm

    _patch_pool(monkeypatch, cm)
    cfg = SimpleNamespace(torch_dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="pretrained_config"):
        cm.InklingHybridCacheManager(mapping=_adp_mapping(False), max_batch_size=8)
    with pytest.raises(TypeError, match="max_batch_size"):
        cm.InklingHybridCacheManager(pretrained_config=cfg, mapping=_adp_mapping(False))


_ADP_RANKS = 4


def _tiny_attention(adp, layer_idx=0):
    """A real InklingAttention on a miniature config; returns (config, module).

    Small enough to build in a unit test, but every dimension the ADP rule
    touches is still divisible by 4 ranks so the pure-TP arm really shards.
    ``local_layer_ids=[0]`` pins layer 0 as a local (sliding-window) layer, so
    which of the swa_* / plain fields apply is fixed rather than inherited from
    whatever pattern a 2-layer config would default to.

    Expected widths are derived from the config's own per-layer accessors, not
    hardcoded: the claim under test is "full under ADP, 1/tp under TP", and
    restating the geometry by hand would only add a second thing to get wrong.
    """
    from tensorrt_llm._torch.configs.inkling import InklingTextConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    cfg = InklingTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        local_layer_ids=[0],
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        swa_num_attention_heads=8,
        swa_num_key_value_heads=8,
        swa_head_dim=8,
        d_rel=4,
    )
    mc = ModelConfig(pretrained_config=cfg, mapping=_adp_mapping(adp))
    return cfg, InklingAttention(mc, layer_idx)


def _expected_widths(cfg, layer_idx=0):
    """(full head count, full k/v channel count) for a layer, per the config."""
    heads = cfg.layer_num_heads(layer_idx)
    kv_dim = cfg.layer_num_kv_heads(layer_idx) * cfg.layer_head_dim(layer_idx)
    # If these were not divisible the TP arm below would prove nothing.
    assert heads % _ADP_RANKS == 0, heads
    assert kv_dim % _ADP_RANKS == 0, kv_dim
    assert (heads * cfg.d_rel) % _ADP_RANKS == 0, heads * cfg.d_rel
    return heads, kv_dim


def test_attention_keeps_every_head_and_channel_under_attention_dp(no_collectives):
    """The whole point of ADP: this rank owns the full head set for its own
    requests. r_proj, the k/v short convs and local_num_heads all hang off that
    head split, so each must be full width -- if any one of them kept the global
    1/tp slice it would disagree with the qkv projection the base already built
    at full width, and the relative-bias einsum would be the first thing to
    notice."""
    cfg, attn = _tiny_attention(adp=True)
    heads, kv_dim = _expected_widths(cfg)

    assert attn.local_num_heads == heads  # every head, not heads/4
    assert attn.num_heads == heads  # and the base agrees
    assert attn.r_proj.tp_mode is None
    assert tuple(attn.r_proj.weight.shape) == (heads * cfg.d_rel, cfg.hidden_size)
    assert attn.k_sconv.channels == kv_dim == attn.k_sconv.channels_full
    assert attn.v_sconv.channels == kv_dim == attn.v_sconv.channels_full


def test_the_backend_gets_this_layer_s_own_attention_scalars(no_collectives):
    """The compute lives in InklingTritonAttention.forward, which reads
    sm_scale / rel_extent / window_left off itself -- so InklingAttention has to
    put them there (create_attention() takes a fixed kwarg list, with no
    passthrough for model-specific values).

    Both layer families are checked because the two differ and getting one wrong
    does not raise: a local layer that inherited a global layer's window_left=-1
    would silently attend outside its sliding window and produce plausible,
    wrong text. The tiny config pins layer 0 local and layer 1 global.
    """
    _, local = _tiny_attention(adp=False, layer_idx=0)
    _, glob = _tiny_attention(adp=False, layer_idx=1)

    for module in (local, glob):
        assert module.attn.sm_scale == module.sm_scale
        assert module.attn.rel_extent == module.rel_extent
        assert module.attn.window_left == module.window_left
        # The moved code indexes the KV cache with the backend's layer_idx, and
        # KVCacheManagerV2 expects the GLOBAL index there.
        assert module.attn.layer_idx == module.layer_idx

    # And the two layers really do disagree, or the loop above proves nothing.
    assert local.window_left >= 0, "layer 0 should be a sliding-window layer"
    assert glob.window_left == -1, "layer 1 should be full causal"
    assert local.rel_extent != glob.rel_extent


def test_attention_still_shards_heads_and_channels_without_attention_dp(no_collectives):
    """Regression guard: pure TP is what every Inkling accuracy run to date
    measured, and the ADP change must not have moved it."""
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    cfg, attn = _tiny_attention(adp=False)
    heads, kv_dim = _expected_widths(cfg)

    assert attn.local_num_heads == heads // _ADP_RANKS
    assert attn.num_heads == heads // _ADP_RANKS
    assert attn.r_proj.tp_mode == TensorParallelMode.COLUMN
    assert tuple(attn.r_proj.weight.shape) == (
        heads * cfg.d_rel // _ADP_RANKS,
        cfg.hidden_size,
    )
    assert attn.k_sconv.channels == kv_dim // _ADP_RANKS
    assert attn.k_sconv.channels_full == kv_dim
