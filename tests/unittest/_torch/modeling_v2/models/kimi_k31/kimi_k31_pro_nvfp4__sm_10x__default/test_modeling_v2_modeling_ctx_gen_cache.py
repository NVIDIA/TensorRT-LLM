# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both sides of a disaggregated sm_103 pair of the NVFP4 default target lay out every region the cache
transceiver moves alike. CPU only.

A TEP8 context server (attention TP 8, MoE EP 8, no vision tower) hands each prompt's cache to a DEP16 generation
server (attention DP 16, MoE EP 16). The transceiver moves the MLA latent pages (FP8), the DSA index-key pages (FP8
keys and their FP32 scales), each KDA layer's convolution and recurrent state (FP32; head-sharded under attention TP,
whole under attention DP) and the DFlash2 drafter's draft pages (its BF16 history). What differs between the
servers of a pair is what each is told: the role the runtime publishes (``TRTLLM_DISAGG_ROLE``), from which the
context server's cache manager keeps no KDA replay state (``kda_context_only``) and its drafter is context-only
(``dflash2_context_kv_only``); the attention TP or DP mapping; ``disable_mm_encoder`` on the context server; and the
batch sizes of the recipe's server YAMLs.

Each server is built from its own arguments as the engine builds it, on CPU:

* the export's config and the server's arguments route to the NVFP4 default target, whose ``get_model_defaults``
  the arguments take; the model loader's config path (``ModelLoader._load_and_validate_config``) gives the config
  the model is built from, and the vision wrapper hands its text config to the engine (``K31VLWrapper``);
* the target's ``K31ForCausalLM._build_dflash2`` builds the drafter (the DFlash worker, which holds no cache
  layout, stood in);
* the engine's ``_create_kv_cache_manager`` builds the target's ``runtime_hooks.kv_cache_manager_cls`` with the
  ``draft_*`` arguments the class declares for the deployment (``unified_draft_cache_kwargs``), up to the native
  pool allocation. The native config it would allocate from (every layer's buffers and their bytes per page or
  slot, the page size, the windows) and the manager's transfer declarations (ignored roles, mapper kinds, bytes
  per head and per convolution section) are what the tests compare.

The drafter is the checkpoint's config at reduced widths: its layer count, latent width, window and dtype, which
shape its draft pages, are the checkpoint's. The pools are not allocated, so the page table the transceiver builds
from them and the peer registration checks are not run here; the byte transfer of the manager between a TEP and a
DEP server is a GPU test of the family's KDA + DSA cache manager.
"""

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default import (
    modeling,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm import bf16_linear
from tensorrt_llm._torch.pyexecutor import _util as engine_util
from tensorrt_llm._torch.pyexecutor import model_loader
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole
from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args
from tensorrt_llm.runtime.kv_cache_manager_v2 import SsmLayerConfig

__extra_import_path__ = [".."]
from _k31_builders import (  # noqa: E402 — needs the path declared above
    CHECKPOINT_MLA_LAYERS,
    SM103,
    HfConfigOnly,
    checkpoint_dir,
    on_gpu,
)

pytestmark = pytest.mark.cpu_only

_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
_DRAFT_LEN = 7
#: The arguments every server of the recipe sets alike.
_COMMON = dict(
    modeling_v2="require",
    max_num_tokens=8192,
    max_seq_len=8192,
    kv_cache_config=dict(dtype="fp8", mamba_ssm_cache_dtype="float32"),
    moe_config=dict(backend="CUTEDSL"),
    sparse_attention_config=dict(algorithm="dsa", indexer_k_dtype="fp8"),
    cache_transceiver_config=dict(backend="NIXL", transceiver_runtime="PYTHON"),
)
#: Each server: (the role the runtime publishes, its own arguments).
_SERVERS = {
    "tep8-context": (
        "context",
        dict(
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            enable_attention_dp=False,
            max_batch_size=16,
            disable_mm_encoder=True,
        ),
    ),
    "dep16-generation": (
        "generation",
        dict(
            tensor_parallel_size=16,
            moe_expert_parallel_size=16,
            enable_attention_dp=True,
            max_batch_size=12,
        ),
    ),
}
_PAIRS = (("tep8-context", "dep16-generation"),)
_PAIR_IDS = ("tep8-ctx-to-dep16-gen",)

#: The DFlash2 MLA drafter checkpoint's config.json at reduced widths (hidden, heads, projections).
_DRAFTER = dict(
    architectures=["DFlash2DraftModel"],
    model_type="qwen3",
    dtype="bfloat16",
    hidden_size=256,
    intermediate_size=256,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=64,
    q_lora_rank=64,
    kv_lora_rank=512,
    qk_nope_head_dim=32,
    qk_rope_head_dim=64,
    v_head_dim=32,
    num_hidden_layers=5,
    layer_types=["sliding_attention"] * 5,
    sliding_window=1024,
    use_sliding_window=True,
    max_window_layers=5,
    is_causal=True,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    vocab_size=163840,
    max_position_embeddings=1048576,
    rope_theta=50000.0,
    rope_parameters=dict(
        rope_type="yarn",
        rope_theta=50000.0,
        factor=32.0,
        original_max_position_embeddings=32768,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    ),
    dflash_config=dict(
        attention_mode="mla",
        block_size=5,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=32,
        mask_token_id=163824,
        target_layer_ids=[7, 23, 51, 67, 83, 91],
    ),
)

# The export's layers (0-based): MLA with a DSA indexer, KDA on the rest, then the drafter's five.
_MLA_LAYERS = [layer - 1 for layer in CHECKPOINT_MLA_LAYERS]
_KDA_LAYERS = [layer for layer in range(93) if layer not in _MLA_LAYERS]
_DRAFT_LAYERS = list(range(93, 98))
_TOKENS_PER_BLOCK = 64
#: KDA heads, their recurrent state (head_dim x head_dim FP32 per head) and convolution width.
_KDA_HEADS, _KDA_HEAD_BYTES, _KDA_CONV_KERNEL = 96, 128 * 128 * 4, 4
#: Bytes per token: the FP8 latent (kv_lora_rank + qk_rope_head_dim), the FP8 index keys with one FP32 scale per
#: 128 of them, and the drafter's BF16 latent.
_LATENT_BYTES, _INDEX_KEY_BYTES, _DRAFT_LATENT_BYTES = 576, 128 + 4, 576 * 2


@dataclasses.dataclass
class _Server:
    name: str
    role: str
    args: TorchLlmArgs
    drafter: nn.Module
    #: Whether the drafter's KV goes to a separate draft manager (it does not: it is the target manager's).
    separate_draft_manager: bool
    #: The target's cache manager, built up to its native pool allocation.
    manager: object
    #: The native config the manager would allocate its pools from.
    config: object


class _LaidOut(Exception):
    """The native pool allocation was reached: the manager and the config it would allocate from."""

    def __init__(self, manager, config):
        super().__init__("the cache manager reached its pool allocation")
        self.manager, self.config = manager, config


def _stop_at_allocation(self, config, **kwargs):
    raise _LaidOut(self, config)


class _Worker:
    """Stands in for the DFlash worker, which keeps no cache layout."""

    def __init__(self, *args, **kwargs):
        pass

    def set_draft_model(self, draft_model):
        pass

    def set_target_model(self, target_model):
        pass


def _text_model_config(config: ModelConfig) -> ModelConfig:
    """The model config the engine reads once the vision wrapper is built (``K31VLWrapper`` points its
    ``pretrained_config`` at the text config)."""
    text = copy.copy(config)
    text._frozen = False
    text.pretrained_config = config.pretrained_config.text_config
    text._frozen = True
    return text


def _build_drafter(mp, model_config: ModelConfig, spec_config):
    """The drafter ``K31ForCausalLM._build_dflash2`` builds on this server, and its config."""
    mp.setattr(modeling, "K31DFlashWorker", _Worker)
    mp.setattr(modeling, "K31HelixDFlashWorker", _Worker)
    mp.setattr(modeling, "get_sm_version", lambda: 103)
    # One process stands in for one rank: no TP group to build communication workspaces for.
    allreduce = SimpleNamespace(
        mnnvl_allreduce=None, uses_nccl_symmetric_memory_window=lambda: False
    )
    mp.setattr(bf16_linear, "tp_allreduce", lambda *args, **kwargs: allreduce)
    causal_lm = modeling.K31ForCausalLM.__new__(modeling.K31ForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.logits_processor = None
    causal_lm.epilogue = []
    causal_lm._build_dflash2(model_config, spec_config)
    return causal_lm


def _build_manager(mp, target, model_config, args, spec_config, causal_lm):
    """The target's cache manager as the engine creates it, up to the native pool allocation."""
    manager_cls = runtime_hooks_of(target).kv_cache_manager_cls
    mapping = model_config.mapping
    # The token budget sizes the pools, not their pages; the host tier and the event buffer are left out.
    kv_cache_config = args.kv_cache_config.model_copy(
        update=dict(max_tokens=1 << 16, host_cache_size=0, event_buffer_max_size=0)
    )
    draft_kwargs = manager_cls.unified_draft_cache_kwargs(
        model_engine=SimpleNamespace(model=SimpleNamespace(draft_model=causal_lm.draft_model)),
        draft_config=causal_lm.draft_config,
        spec_config=spec_config,
        mapping=mapping,
        kv_cache_config=kv_cache_config,
        is_disagg=True,
    )
    mp.setattr(KVCacheManagerV2, "_allocate_cache_pools", _stop_at_allocation)
    try:
        engine_util._create_kv_cache_manager(
            model_engine=None,
            kv_cache_manager_cls=manager_cls,
            mapping=mapping,
            kv_cache_config=kv_cache_config,
            tokens_per_block=kv_cache_config.tokens_per_block,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            spec_config=spec_config,
            sparse_attention_config=args.sparse_attention_config,
            max_num_tokens=args.max_num_tokens,
            max_beam_width=1,
            kv_connector_manager=None,
            execution_stream=object(),
            model_config=model_config,
            dtype=torch.bfloat16,
            is_disagg=True,
            **draft_kwargs,
        )
    except _LaidOut as laid_out:
        return laid_out.manager, laid_out.config
    raise AssertionError("the cache manager was built without reaching its pool allocation")


def _build_server(name: str, checkpoint: str, drafter_dir: str) -> _Server:
    role, options = _SERVERS[name]
    args = TorchLlmArgs(model=checkpoint, **copy.deepcopy(_COMMON), **options)
    spec_config = DFlashDecodingConfig(max_draft_len=_DRAFT_LEN, speculative_model=drafter_dir)
    spec_config.resolve_from_checkpoint()
    mapping = args.parallel_config.to_mapping()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_ROLE_ENV, role)
        with pytest.MonkeyPatch.context() as gpu:
            on_gpu(gpu, SM103)
            # The config read before the model defaults apply routes the target, whose defaults the
            # arguments then take; the model is built from the config the loader reads after that.
            first = ModelConfig.from_pretrained(
                checkpoint,
                trust_remote_code=args.trust_remote_code,
                mm_encoder_only=args.mm_encoder_only,
                mapping=mapping,
                spec_config=spec_config,
            )
            target = modeling_v2_resolve(first, args)
            apply_model_defaults_to_llm_args(args, target.get_model_defaults(args))
            loader = object.__new__(model_loader.ModelLoader)
            loader.llm_args = args
            loader.mapping = mapping
            loader.spec_config = spec_config
            loader.sparse_attention_config = args.sparse_attention_config
            loader.max_num_tokens = args.max_num_tokens
            loader.max_seq_len = args.max_seq_len
            loader.lora_config = None
            config = loader._load_and_validate_config(checkpoint, HfConfigOnly())
        assert target is modeling.ModelingV2KimiK31ProNvfp4Sm10xDefault, (name, target)
        assert config.modeling_v2_target is target, (name, config.modeling_v2_target)
        model_config = _text_model_config(config)
        causal_lm = _build_drafter(mp, model_config, spec_config)
        manager, laid_out = _build_manager(mp, target, model_config, args, spec_config, causal_lm)
    return _Server(
        name=name,
        role=role,
        args=args,
        drafter=causal_lm.draft_model,
        separate_draft_manager=causal_lm.use_separate_draft_kv_cache,
        manager=manager,
        config=laid_out,
    )


@pytest.fixture(scope="module")
def servers(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("k31_ctx_gen_cache")
    checkpoint = checkpoint_dir(root)
    # The export's text config also states what the cache manager reads and routing does not: the KDA
    # convolution width and the model dtype.
    config_path = root / "checkpoint" / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["linear_attn_config"]["short_conv_kernel_size"] = _KDA_CONV_KERNEL
    config["text_config"]["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config))
    drafter_dir = root / "drafter"
    drafter_dir.mkdir()
    (drafter_dir / "config.json").write_text(json.dumps(_DRAFTER))
    return {name: _build_server(name, checkpoint, str(drafter_dir)) for name in _SERVERS}


# --- What a server lays out -------------------------------------------------------------------------------------


def _layers(server: _Server, kind: str) -> dict:
    """Global layer id -> native layer config of the ``kind`` layers: ``mla``, ``kda`` or ``draft``."""
    manager = server.manager
    global_of = {local: layer for layer, local in manager.layer_offsets.items()}
    out = {}
    for layer in server.config.layers:
        local = int(layer.layer_id)
        if local in manager._unified_draft_layer_specs:
            layer_kind = "draft"
        elif isinstance(layer, SsmLayerConfig):
            layer_kind = "kda"
        else:
            layer_kind = "mla"
        if layer_kind == kind:
            out[global_of[local]] = layer
    return out


def _moved(server: _Server, layer) -> list:
    """The buffers of a layer the transceiver moves, as (role, bytes per page or slot), in layer order."""
    ignored = server.manager.get_disagg_ignored_roles()
    return [
        (str(buffer.role), buffer.size) for buffer in layer.buffers if buffer.role not in ignored
    ]


def _mapper_kind(server: _Server, role):
    kinds = server.manager.get_disagg_role_mapper_kinds()
    return kinds.get(role, kinds[Role.ALL])


def _attention_tp(server: _Server) -> int:
    """The ranks one attention group's heads are split over: TP, or 1 under attention DP."""
    mapping = server.manager.mapping
    return 1 if mapping.enable_attention_dp else mapping.tp_size


def _pair(servers, ctx: str, gen: str):
    return servers[ctx], servers[gen]


# --- The tests --------------------------------------------------------------------------------------------------


def test_the_role_reaches_the_cache_manager_and_the_drafter(servers):
    """The context server's manager keeps no KDA replay state and its drafter is context-only; each generation
    server keeps the replay state of the drafts it verifies and drafts. The drafter's pages are the target
    manager's (no separate draft manager) on every server.

    Regression: the published role stops reaching the manager or the drafter, so a context server builds the
    generation layout (and the pair checks below compare two generation builds) or the reverse."""
    for name, server in servers.items():
        context = server.role == "context"
        manager = server.manager
        assert manager.kda_context_only is context, name
        assert (manager._kda_replay is None) is context, name
        assert server.drafter.context_kv_only is context, name
        assert server.args.disable_mm_encoder is context, name
        assert server.separate_draft_manager is False, name
        assert manager.has_standalone_draft_cache, name


@pytest.mark.parametrize("ctx, gen", _PAIRS, ids=_PAIR_IDS)
def test_latent_and_index_key_pages_match(servers, ctx, gen):
    """Every MLA layer moves the same pages on both sides: the FP8 latent and the DSA index keys (FP8 keys and
    their FP32 scales) of the same layers, with the same bytes per page, page size, transfer mapping and KV-head
    geometry (one latent head on every rank).

    Regression: a side that lays out a latent or index-key page differently (dtype, width, scale block, page size
    or the set of indexer layers), so the generation server reads the transferred pages as other bytes."""
    c, g = _pair(servers, ctx, gen)
    c_layers, g_layers = _layers(c, "mla"), _layers(g, "mla")
    assert sorted(c_layers) == sorted(g_layers) == _MLA_LAYERS
    assert c.config.tokens_per_block == g.config.tokens_per_block == _TOKENS_PER_BLOCK
    assert c.manager.tokens_per_block == g.manager.tokens_per_block == _TOKENS_PER_BLOCK
    for layer in _MLA_LAYERS:
        assert _moved(c, c_layers[layer]) == _moved(g, g_layers[layer]), f"MLA layer {layer}"
    assert _moved(c, c_layers[_MLA_LAYERS[0]]) == [
        (str(Role.KEY), _TOKENS_PER_BLOCK * _LATENT_BYTES),
        (str(Role.INDEX_KEY), _TOKENS_PER_BLOCK * _INDEX_KEY_BYTES),
    ]
    assert c.manager.dtype == g.manager.dtype == DataType.FP8
    for role in (Role.KEY, Role.INDEX_KEY):
        assert _mapper_kind(c, role) == _mapper_kind(g, role), role
    for side, layers in ((c, c_layers), (g, g_layers)):
        manager = side.manager
        heads = {manager.num_kv_heads_per_layer[manager.layer_offsets[layer]] for layer in layers}
        assert (heads, manager.head_dim, manager.kv_factor) == ({1}, _LATENT_BYTES, 1), side.name


@pytest.mark.parametrize("ctx, gen", _PAIRS, ids=_PAIR_IDS)
def test_kda_state_matches_head_by_head(servers, ctx, gen):
    """Every KDA layer moves the recurrent and the convolution state of its slot, and nothing else (the
    generation server's replay buffers in the same slot stay behind), laid out so that the context ranks' head
    shards make up the generation rank's heads: the same dtypes, the same bytes per recurrent head, the same
    q / k / v convolution sections split over the attention TP ranks, and the same transfer mapping.

    Regression: a side that lays out the KDA state differently (state or convolution dtype, head width, section
    order or the head split), so the reassembled state is wrong, or a replay buffer that is moved."""
    c, g = _pair(servers, ctx, gen)
    c_layers, g_layers = _layers(c, "kda"), _layers(g, "kda")
    assert sorted(c_layers) == sorted(g_layers) == _KDA_LAYERS
    state, conv = str(MambaRole.SSM_STATE), str(MambaRole.CONV_STATE)
    c_mgr, g_mgr = c.manager, g.manager
    assert (c_mgr.ssm_state_dtype, c_mgr.conv_state_dtype) == (
        g_mgr.ssm_state_dtype,
        g_mgr.conv_state_dtype,
    )
    assert c_mgr.ssm_state_dtype == torch.float32
    assert c_mgr._state_layout.conv_state_layout == g_mgr._state_layout.conv_state_layout == "q_k_v"
    c_roles, g_roles = c_mgr.get_disagg_role_layouts(), g_mgr.get_disagg_role_layouts()
    head_bytes = c_roles[MambaRole.SSM_STATE].bytes_per_head
    assert head_bytes == g_roles[MambaRole.SSM_STATE].bytes_per_head == _KDA_HEAD_BYTES
    assert tuple(c_mgr.ssm_state_shape[1:]) == tuple(g_mgr.ssm_state_shape[1:]) == (128, 128)
    c_heads, g_heads = c_mgr.ssm_state_shape[0], g_mgr.ssm_state_shape[0]
    assert c_heads * _attention_tp(c) == g_heads * _attention_tp(g) == _KDA_HEADS
    c_sections = c_roles[MambaRole.CONV_STATE].section_bytes
    g_sections = g_roles[MambaRole.CONV_STATE].section_bytes
    # Each of q, k and v holds a rank's heads x head_dim rows of the convolution window.
    row_bytes = 128 * (_KDA_CONV_KERNEL - 1) * c_mgr.conv_state_dtype.itemsize
    assert list(c_sections) == [c_heads * row_bytes] * 3
    assert list(g_sections) == [g_heads * row_bytes] * 3
    for role in (MambaRole.SSM_STATE, MambaRole.CONV_STATE):
        assert _mapper_kind(c, role) == _mapper_kind(g, role), role
    for side, layers, heads, sections in (
        (c, c_layers, c_heads, c_sections),
        (g, g_layers, g_heads, g_sections),
    ):
        expected = [(state, heads * head_bytes), (conv, sum(sections))]
        for layer, config in layers.items():
            assert _moved(side, config) == expected, f"{side.name} KDA layer {layer}"
        # A generation server's slots also hold the replay buffers its verify reads; a context server's
        # hold the moved state alone.
        kept = {str(b.role) for config in layers.values() for b in config.buffers} - {state, conv}
        assert bool(kept) is (side.role == "generation"), (side.name, sorted(kept))


@pytest.mark.parametrize("ctx, gen", _PAIRS, ids=_PAIR_IDS)
def test_draft_pages_match(servers, ctx, gen):
    """The drafter's pages move with the target's: the same draft layers after the model's, the same BF16
    latent per token and layer on every rank (one K-only head), the same page size and the same window of the
    drafter's history, whether the drafter is context-only (context server) or drafts (generation server).

    Regression: a context-only drafter that keeps its history in another layout than the drafting one (layer
    count, latent width, dtype or window), so the generation server's drafter decodes from other bytes."""
    c, g = _pair(servers, ctx, gen)
    c_layers, g_layers = _layers(c, "draft"), _layers(g, "draft")
    assert sorted(c_layers) == sorted(g_layers) == _DRAFT_LAYERS
    for layer in _DRAFT_LAYERS:
        c_config, g_config = c_layers[layer], g_layers[layer]
        assert _moved(c, c_config) == _moved(g, g_config), f"draft layer {layer}"
        assert c_config.sliding_window_size == g_config.sliding_window_size, f"draft layer {layer}"
        assert _mapper_kind(c, Role.KEY) == _mapper_kind(g, Role.KEY)
    first = _DRAFT_LAYERS[0]
    assert _moved(c, c_layers[first]) == [(str(Role.KEY), _TOKENS_PER_BLOCK * _DRAFT_LATENT_BYTES)]
    assert c_layers[first].sliding_window_size == _DRAFTER["sliding_window"] + 1
    for side in (c, g):
        manager = side.manager
        assert (manager._draft_dtype, manager._draft_num_kv_heads, manager._draft_kv_factor) == (
            DataType.BF16,
            1,
            1,
        ), side.name
