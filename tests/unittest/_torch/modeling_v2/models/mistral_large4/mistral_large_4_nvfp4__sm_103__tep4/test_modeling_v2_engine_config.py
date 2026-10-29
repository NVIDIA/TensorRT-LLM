# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine's configuration of Mistral-Large-4-1T-A52B-NVFP4 at tep4, as a served engine reads it.

No GPU and no weights. The target serves the checkpoint's own context: max_seq_len 1048576 (its
max_position_embeddings), chunked prefill on in steps of max_num_tokens 32768, 16 requests, CUDA
graphs over batches 1, 2, 4, 8 and 16 padded, the fp8_ds_mla latent KV cache
(`MistralLarge4CacheManager`, 656 bytes a token on each KV-owning layer) at kv_cache_config's
default dtype as the CI class runs it and at the gates' fp8, block reuse off. The delivered
config (max_seq_len = max_num_tokens = 32768, chunked prefill off) stays pinned beside it.
At each config the checkpoint goes through the engine's own configuration path, with the LLM-API
options of its accuracy gate:

* the LLM API front end: ``TorchLlmArgs`` and ``ModelLoader._update_from_hf_quant_config``
  (``llmapi/llm_utils.py``), which settle ``llm.args.quant_config``, the key the accuracy
  harness looks its reference row up by;
* each rank: ``_load_config_and_create_checkpoint_loader`` (``py_executor_creator``), then the
  model loader's ``_load_and_validate_config``, which builds the ``ModelConfig`` through
  ``ModelConfig.from_pretrained`` and asks the modeling_v2 router for a target. The router's
  ``route`` is replaced by a recorder, so what it is handed is observed rather than judged.

What the engine then does with that configuration -- the sizes the target takes from it on its
first step (the indexer's logits budget and the held context workspace's rows), the context path
a chunk of a long prompt takes, the query split across the four TP ranks, the scheduler's chunks,
the buffers sized by max_seq_len, and what warmup, the KV-capacity estimation and CUDA-graph
capture send -- is pinned on the engine's own host code. Where that code allocates on the device,
the allocation is taken on PyTorch's meta device; where it copies to the device, the copy is the
identity. Nothing asserted depends on a visible device or on its SM count.

The checkpoint is Mistral-native, as published (params.json, tekken.json, consolidated
safetensors), and is loaded with checkpoint_format mistral_large_4: the gates' LLM-API options
state it and the boot gate's command passes --checkpoint_format=mistral_large_4. It is the
ModelOpt MIXED_PRECISION export, whose params.json quantization_config names Mistral modules
(NVFP4 routed experts, FP8_PB_WO block-FP8 linears, group_size 16). The checkpoint at dep4 and
at tp4 is checked to resolve and route as it did before the 1M config was added.

The same target serves the NVFP4 checkpoint at tep8, eight ranks over two GB300 nodes of four
(attention tensor parallel over all eight, attention DP off, the routed experts expert-parallel
eight ways, 16 requests over the whole engine), and tep8 is checked at the 1M config and at
the delivered 32768 config. At 1M the gates spell the latent KV cache fp8 and the CI class
leaves it at its default, auto (the checkpoint's fp8_kv_cache); the delivered gates leave it at
its default; every spelling builds the fp8_ds_mla pool of MistralLarge4CacheManager. At each
config the checkpoint goes through the same path with the two spellings the gates use for the
tep8 core unit:

* the accuracy gate's LLM-API options (spec-1m-tep8/options/core.yaml at 1M,
  spec-tep8/options/core.yaml delivered), handed to ``TorchLlmArgs``;
* the boot gate's command, parsed and turned into ``LLM`` keyword arguments by the tree's own
  ``examples/llm-api/quickstart_advanced.py`` (``parse_arguments`` and ``setup_llm``), as the
  gate harness builds its engine, plus the ``modeling_v2`` argument the harness adds.

What the engine does with the 1M configuration at eight ranks -- the prompts admitted, the
context path, the query split across the eight TP ranks and each rank's indexer chunk plan, the
scheduler's chunks, the buffers sized by max_seq_len, warmup, the KV-capacity estimation, CUDA
graph capture and the load the measured tep8 pools hold -- is pinned as at tep4.

Neither gate states ``gpus_per_node``: ``TorchLlmArgs`` takes it from
``torch.cuda.device_count()`` on rank 0, which on a GB300 node of this cluster is 4. The count is
set to 4 where the tep8 arguments are built rather than read from the test host, which may see
no device or a whole node. Where the engine reads the node layout on its way to a modeling_v2
target -- the device each rank selects, the communicators, the causal-LM shell's lm_head, the
MNNVL communicator split, the AllReduce strategy's node test, the MoE autotuner, the KV host
tier -- is pinned on the engine's own host code, eight ranks simulated in one process where they
exchange anything. tep8 is checked against the NVFP4 checkpoint at tep4 (on the options of
either config) and at dep4.

The checkpoint is looked up under ``LLM_MODELS_ROOT`` (as ``Mistral-Large-4-1T-A52B-NVFP4`` or
``mistral-large-4-nvfp4``) by its params.json, and its tests are skipped where it is absent.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml
from test_common.llm_data import llm_models_root

import tensorrt_llm.bindings
from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
    modeling as tep4_modeling,
)
from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as dsa_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as dsa_metadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    Indexer,
    plan_mla_chunked_prefill_chunks,
    select_indexer_chunk_size,
    split_prefill_chunks,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.autotuner import AutoTuner, DistributedTuningStrategy
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from tensorrt_llm._torch.distributed import ops as distributed_ops
from tensorrt_llm._torch.distributed.communicator import Distributed, MPIDist
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import maybe_create_moe_load_balancer
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor._util import (
    CacheCost,
    KvCacheCreator,
    get_attention_workspace_bytes_per_token,
    get_attention_workspace_fixed_bytes,
    get_attention_workspace_is_chunked_prefill_bounded,
    get_kv_cache_manager_cls,
    get_mla_context_workspace_kv_len_cap,
    validate_feature_combination,
)
from tensorrt_llm._torch.pyexecutor.engine.cuda_graph import filter_cuda_graph_batch_sizes
from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
    _compute_auto_host_tier_quota,
    _get_generation_kv_capacity,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _filter_piecewise_capture_num_tokens,
)
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader as RankModelLoader
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.executor import base_worker
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    ContextChunkingPolicy,
    DeepSeekSparseAttentionConfig,
    TorchLlmArgs,
    _ParallelConfig,
)
from tensorrt_llm.llmapi.llm_utils import ModelLoader as FrontEndModelLoader
from tensorrt_llm.mapping import Mapping, MpiTopology
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_LAYERS = 72
_SM103 = (10, 3)
_RANKS = 4

# The serving config (spec-1m-kvds-tep4/options/core.yaml with task.yaml's runtime flags).
_MAX_SEQ_LEN = 1048576
_MAX_NUM_TOKENS = 32768
_MAX_BATCH_SIZE = 16
_TOKENS_PER_BLOCK = 32
_INDEX_TOPK = 2048
# The 1M gate's long prompt: max_seq_len less the 16 tokens it generates, less one.
_ENVELOPE_PROMPT = _MAX_SEQ_LEN - 16 - 1
# The pool the delivered tep4 boot (32768 unchunked, the 576-byte latent pool on every layer)
# gave each rank, in tokens (NOTES.md), and that pool's bytes a token.
_DELIVERED_POOL_TOKENS = 2_373_024
_DELIVERED_BYTES_PER_TOKEN = 576 * 72 + 132 * 44
# One token of the DSA pool per rank: the fp8_ds_mla latent (512 e4m3, four fp32 tile scales,
# 64 bf16 rope: 656 bytes) on the 63 KV-owning layers, the fp8 indexer key (128) and its fp32
# scale on the 44 full-indexer layers.
_KV_BYTES_PER_TOKEN = 656 * 63 + 132 * 44
# The vision tower's bf16 weights, which every rank of the NVFP4 targets loads whole.
_VISION_TOWER_BYTES = 3_408_437_248
# The pool a boot of the 1M accuracy gate (the Tep4Ctx1m class, KV fraction 0.85) gave each
# rank on the native checkpoint, the vision tower loaded, in tokens: 66,734 pages.
_TEP4_1M_POOL_TOKENS = 66_734 * 32

# The tep4 topology, and the other members of this checkpoint at their own base options
# (spec-dep4/options/core.yaml, spec-tp4/options/core.yaml), with the latent KV cache at its
# default: the checkpoint's fp8_kv_cache, the fp8_ds_mla pool.
_TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
_DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
_TP4 = dict(
    tensor_parallel_size=4,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)


def _options(
    max_seq_len: int,
    max_num_tokens: int,
    chunked: bool,
    topology: dict = _TEP4,
    kv_dtype: str | None = None,
) -> dict:
    """The LLM-API options of a Mistral Large 4 accuracy gate, the checkpoint in its Mistral-native
    format, the latent KV cache at its default dtype (the checkpoint's fp8_kv_cache) unless
    ``kv_dtype`` names one."""
    kv = dict(enable_block_reuse=False, free_gpu_memory_fraction=0.85)
    if kv_dtype is not None:
        kv.update(dtype=kv_dtype)
    return dict(
        checkpoint_format="mistral_large_4",
        **topology,
        max_batch_size=_MAX_BATCH_SIZE,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=chunked,
        cuda_graph_config=dict(enable_padding=True, max_batch_size=16),
        kv_cache_config=kv,
    )


@dataclass(frozen=True)
class _Case:
    """One config of the checkpoint, the bounds its accuracy gate runs with, and what it must
    resolve to."""

    names: tuple[str, ...]
    max_seq_len: int
    max_num_tokens: int
    chunked: bool
    quant_algo: QuantAlgo
    group_size: int
    # exclude_modules as llm.args reports it, and as the ModelConfig the ranks build holds it
    llm_exclude_modules: list
    engine_exclude_modules: list
    # (quant_algo, group_size) -> number of per-layer configs
    layer_algos: dict
    rope_factor: float
    max_position_embeddings: int
    anchor: float
    target: str
    # kv_cache_config.dtype as the gate passes it; None leaves the default (auto)
    kv_dtype: str | None = None

    @property
    def options(self) -> dict:
        return _options(self.max_seq_len, self.max_num_tokens, self.chunked, kv_dtype=self.kv_dtype)


_NVFP4_NAMES = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The serving config.
_NVFP4 = _Case(
    names=_NVFP4_NAMES,
    max_seq_len=_MAX_SEQ_LEN,
    max_num_tokens=_MAX_NUM_TOKENS,
    chunked=True,
    quant_algo=QuantAlgo.MIXED_PRECISION,
    group_size=16,
    llm_exclude_modules=[],
    engine_exclude_modules=[],
    layer_algos={(QuantAlgo.NVFP4, 16): 71, (QuantAlgo.FP8_BLOCK_SCALES, None): 404},
    rope_factor=2048.0,
    max_position_embeddings=1048576,
    anchor=91.926,
    target="ModelingV2MistralLarge4Nvfp4Sm103Tep4",
)
# The serving config as the gates run it: spec-1m-kvds-tep4/options/core.yaml and the
# runtime's --kv_cache_dtype=fp8.
_NVFP4_GATE = dataclasses.replace(_NVFP4, kv_dtype="fp8")
# The delivered config: 32768 tokens, one step per prompt.
_NVFP4_32K = dataclasses.replace(_NVFP4, max_seq_len=32768, chunked=False)

# The GSM8K reference rows, and the key the harness reads them by.
_GSM8K_REFERENCES = (
    Path(__file__).resolve().parents[6] / "integration/defs/accuracy/references/gsm8k.yaml"
)
_MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"


def _checkpoint_dir(names: tuple[str, ...]) -> Path:
    """The first of ``names`` under LLM_MODELS_ROOT holding a Mistral-native params.json."""
    root = llm_models_root()
    for name in names:
        if root is not None and (root / name / "params.json").is_file():
            return root / name
    pytest.skip(f"none of {names} is under LLM_MODELS_ROOT ({root})")


class _Routed:
    """What the recording `route` returns: no target is built here."""


def _resolve(checkpoint_dir: Path, options: dict) -> SimpleNamespace:
    """The configuration a served engine reaches for this checkpoint under ``options``."""
    llm_args = TorchLlmArgs(model=str(checkpoint_dir), modeling_v2="require", **options)
    return _resolve_args(checkpoint_dir, llm_args)


def _resolve_args(checkpoint_dir: Path, llm_args: TorchLlmArgs) -> SimpleNamespace:
    """The configuration a served engine reaches for this checkpoint, front end then rank."""
    model = str(checkpoint_dir)
    prequantized = FrontEndModelLoader(llm_args)._update_from_hf_quant_config()

    routed = []

    def route(pretrained_config, mapping, compute_capability, spec_config):
        routed.append(
            SimpleNamespace(
                pretrained_config=pretrained_config,
                quantization_config=copy.deepcopy(pretrained_config.quantization_config),
                mapping=mapping,
                compute_capability=compute_capability,
                spec_config=spec_config,
            )
        )
        return _Routed

    with pytest.MonkeyPatch.context() as mp:
        # The rank's config pass pins the CUDA memory fraction first; there is no device here.
        mp.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *args, **kwargs: None)
        mp.setattr(_router_index, "_device_compute_capability", lambda: _SM103)
        mp.setattr(_router_index, "routing_module", lambda arch: SimpleNamespace(route=route))
        llm_args, checkpoint_loader = py_executor_creator._load_config_and_create_checkpoint_loader(
            llm_args, model
        )
        _, max_num_tokens, max_seq_len, max_batch_size = llm_args.get_runtime_sizes()
        mapping = llm_args.parallel_config.to_mapping()
        model_config = RankModelLoader(
            llm_args=llm_args,
            mapping=mapping,
            spec_config=llm_args.speculative_config,
            sparse_attention_config=llm_args.sparse_attention_config,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
        )._load_and_validate_config(model, checkpoint_loader)
    return SimpleNamespace(
        path=checkpoint_dir,
        llm_args=llm_args,
        prequantized=prequantized,
        model_config=model_config,
        mapping=mapping,
        routed=routed,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


def _resolve_case(case: _Case) -> SimpleNamespace:
    path = _checkpoint_dir(case.names)
    return SimpleNamespace(
        case=case,
        # params.json as the native loader adapts it (models/checkpoints/mistral/config_loader.py)
        native=MistralConfigLoader().load_pretrained_config(str(path)),
        **vars(_resolve(path, case.options)),
    )


@pytest.fixture(scope="module")
def envelope():
    return _resolve_case(_NVFP4)


@pytest.fixture(scope="module")
def gate():
    return _resolve_case(_NVFP4_GATE)


@pytest.fixture(scope="module")
def delivered():
    return _resolve_case(_NVFP4_32K)


@pytest.fixture(scope="module", params=["envelope", "gate", "delivered"])
def resolved(request):
    return request.getfixturevalue(request.param)


def _anchor(llm_args) -> float:
    """The reference row `AccuracyTask.get_hypothesis_testing_params` selects for this engine."""
    quant = llm_args.quant_config
    acc_specs = dict(
        dtype=llm_args.dtype,
        quant_algo=quant.quant_algo,
        kv_cache_quant_algo=quant.kv_cache_quant_algo,
        spec_dec_algo=None,
        extra_acc_spec=None,
    )
    for entry in yaml.safe_load(_GSM8K_REFERENCES.read_text())[_MODEL_NAME]:
        if all(
            entry.get(key, "auto" if key == "dtype" else None) == value
            for key, value in acc_specs.items()
        ):
            return entry["accuracy"]
    raise AssertionError(f"no {_MODEL_NAME} GSM8K reference row for {acc_specs}")


def _rank_mapping(mapping, rank: int):
    rank_mapping = copy.deepcopy(mapping)
    rank_mapping.rank = rank
    return rank_mapping


def _kv_cost(resolved, rank: int = 0) -> CacheCost:
    """The engine's affine KV cost on ``rank``, as the KV-cache creator sizes the pool with it."""
    model_config, llm_args = resolved.model_config, resolved.llm_args
    kv_cache_config = llm_args.kv_cache_config
    manager = get_kv_cache_manager_cls(model_config, kv_cache_config)
    return CacheCost.from_raw(
        manager.get_cache_size_per_token(
            model_config,
            _rank_mapping(resolved.mapping, rank),
            tokens_per_block=kv_cache_config.tokens_per_block,
            max_seq_len=resolved.max_seq_len,
            max_batch_size=resolved.max_batch_size,
            max_num_tokens=resolved.max_num_tokens,
            kv_cache_config=kv_cache_config,
            spec_config=llm_args.speculative_config,
            is_draft=False,
        )
    )


def _runtime_features(resolved) -> AttentionRuntimeFeatures:
    """The attention runtime features `_create_py_executor` (py_executor_creator.py) builds:
    chunked prefill from the LLM API's enable_chunked_prefill, cache reuse from its
    kv_cache_config.enable_block_reuse, the chunk size max_num_tokens. They are what the target
    reads chunked prefill and block reuse off on its first step."""
    llm_args = resolved.llm_args
    return AttentionRuntimeFeatures(
        chunked_prefill=llm_args.enable_chunked_prefill,
        cache_reuse=llm_args.kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=False,
        chunk_size=resolved.max_num_tokens,
    )


def _target_sizing(resolved) -> SimpleNamespace:
    """What the target sizes its indexer logits budget and its held context workspace from on
    its first step (`_derive_first_step`): the attention metadata `build_attention_metadata`
    (pyexecutor/engine/metadata.py) makes for this engine -- max_num_requests (max_batch_size),
    max_num_tokens and the runtime features (`_runtime_features`, chunked prefill and block
    reuse) -- whose max_seq_len is the KV cache manager's (`TrtllmAttentionMetadata.max_seq_len`),
    the one thing the target's sizing reads through that manager."""

    def metadata(**kwargs):
        return SimpleNamespace(**kwargs, max_seq_len=kwargs["kv_cache_manager"].max_seq_len)

    return build_attention_metadata(
        resolved.model_config,
        max_batch_size=resolved.max_batch_size,
        max_num_tokens=resolved.max_num_tokens,
        max_beam_width=1,
        attention_backend=SimpleNamespace(Metadata=metadata),
        attention_runtime_features=_runtime_features(resolved),
        mapping=resolved.mapping,
        cache_indirection=None,
        kv_cache_manager=SimpleNamespace(max_seq_len=resolved.max_seq_len),
    )


def _metadata_params(resolved):
    model_config = resolved.model_config
    return model_config.sparse_attention_config.to_sparse_metadata_params(
        pretrained_config=model_config.pretrained_config
    )


def _max_blocks_per_seq(max_seq_len: int) -> int:
    """`KVCacheManagerV2.__init__`'s block-table width: max_seq_len plus the generation headroom
    `_get_generation_kv_capacity` gives a target without speculation, in pages, padded to a
    multiple of 4 for the copy_block_offsets kernel."""
    _, headroom = _get_generation_kv_capacity(None, is_draft=False)
    blocks = (max_seq_len + headroom + _TOKENS_PER_BLOCK - 1) // _TOKENS_PER_BLOCK
    return (blocks + 3) // 4 * 4


# ------------------------------------------------------- (1) LLM-API resolution, the anchor


def test_llm_args_quantization_is_the_anchor_key(resolved):
    """`llm.args.quant_config`, as `ModelLoader._update_from_hf_quant_config` (llmapi/
    llm_utils.py) settles it, is what the accuracy harness keys the reference row on
    (`AccuracyTask.get_hypothesis_testing_params`): the same row at every config."""
    case, quant = resolved.case, resolved.llm_args.quant_config
    assert resolved.prequantized is True
    assert quant.quant_algo == case.quant_algo
    assert quant.kv_cache_quant_algo == QuantAlgo.FP8
    assert quant.group_size == case.group_size
    assert quant.exclude_modules == case.llm_exclude_modules
    assert _anchor(resolved.llm_args) == case.anchor


def test_engine_quantization(resolved):
    """The ranks' ModelConfig (`ModelConfig.from_pretrained`, _torch/model_config.py, through
    `ModelLoader._load_and_validate_config`): the same global algorithm, the fp8 latent cache,
    and one per-layer config for every Mistral-named quantized layer, the ModelOpt spellings
    canonicalised (FP8_PB_WO is block-128 FP8) and group_size 16 accepted for the NVFP4
    experts."""
    case, model_config = resolved.case, resolved.model_config
    quant = model_config.quant_config
    assert quant.quant_algo == case.quant_algo
    assert quant.kv_cache_quant_algo == QuantAlgo.FP8
    assert quant.group_size == case.group_size
    assert quant.exclude_modules == case.engine_exclude_modules

    layers = model_config.quant_config_dict or {}
    assert Counter((q.quant_algo, q.group_size) for q in layers.values()) == case.layer_algos
    # The latent cache's algorithm is the engine-wide one above (the checkpoint's fp8_kv_cache
    # under the default kv_cache_config.dtype); a per-layer config sets none of its own.
    assert {q.kv_cache_quant_algo for q in layers.values()} <= {QuantAlgo.FP8, None}
    quantized_layers = resolved.native.quantization_config.get("quantized_layers", {})
    assert set(layers) == set(quantized_layers)
    assert {name for name, q in layers.items() if q.quant_algo == QuantAlgo.NVFP4} == {
        f"layers.{layer}.experts" for layer in range(1, _LAYERS)
    }


def test_envelope_resolves_as_configured(envelope, delivered):
    """`TorchLlmArgs` (llmapi/llm_args.py) and `get_runtime_sizes` keep the 1M config as given:
    tp 4, moe_ep 4, attention DP off; max_seq_len 1048576, max_num_tokens 32768 (the chunk),
    max_batch_size 16; chunked prefill accepted with the DSA sparse attention the checkpoint
    configures, the latent KV cache at its default dtype and block reuse off, on the V2 KV
    manager the sparse registry picks for a MistralLarge4Config, `MistralLarge4CacheManager`
    (attention/backends/sparse/registry.py, the fp8_ds_mla pool). The DSA config is
    the checkpoint's, with its q_split_threshold default. `validate_feature_combination`
    (pyexecutor/_util.py) rejects nothing, and `_create_py_executor` (py_executor_creator.py)
    keeps chunked prefill for MLA on sm_103 (`_MLA_CHUNKED_PREFILL_SUPPORTED_SM_VERSIONS`).
    Beside it the delivered config: 32768 tokens, one step per prompt."""
    for resolved, max_seq_len, chunked in (
        (envelope, _MAX_SEQ_LEN, True),
        (delivered, 32768, False),
    ):
        llm_args, mapping = resolved.llm_args, resolved.mapping
        assert (llm_args.tensor_parallel_size, llm_args.moe_expert_parallel_size) == (4, 4)
        assert llm_args.enable_attention_dp is False
        assert (mapping.world_size, mapping.tp_size, mapping.pp_size, mapping.cp_size) == (
            4,
            4,
            1,
            1,
        )
        assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.enable_attention_dp) == (
            4,
            1,
            False,
        )
        assert (resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size) == (
            _MAX_NUM_TOKENS,
            max_seq_len,
            _MAX_BATCH_SIZE,
        )
        assert llm_args.get_runtime_sizes() == (1, _MAX_NUM_TOKENS, max_seq_len, _MAX_BATCH_SIZE)
        assert (llm_args.max_seq_len, llm_args.max_num_tokens) == (max_seq_len, _MAX_NUM_TOKENS)
        assert llm_args.enable_chunked_prefill is chunked
        assert llm_args.disable_overlap_scheduler is False

        kv_cache_config = llm_args.kv_cache_config
        assert kv_cache_config.dtype == "auto" and kv_cache_config.enable_block_reuse is False
        assert kv_cache_config.tokens_per_block == _TOKENS_PER_BLOCK
        assert kv_cache_config.use_kv_cache_manager_v2 is True
        assert kv_cache_config.max_attention_window is None
        manager = get_kv_cache_manager_cls(resolved.model_config, kv_cache_config)
        assert manager is MistralLarge4CacheManager and issubclass(manager, KVCacheManagerV2)

        graphs = llm_args.cuda_graph_config
        assert (graphs.batch_sizes, graphs.enable_padding) == ([1, 2, 4, 8, 16], True)
        assert llm_args.prefill_capture_num_tokens is None
        assert llm_args.torch_compile_config is None

        sparse = resolved.model_config.sparse_attention_config
        assert type(sparse) is DeepSeekSparseAttentionConfig and sparse.algorithm == "dsa"
        assert sparse.q_split_threshold == 8192
        assert sparse.skip_indexer_for_short_seqs is True
        assert sparse.use_cute_dsl_topk is False and sparse.enable_heuristic_topk is False
        assert resolved.model_config.attn_backend == "TRTLLM"

        validate_feature_combination(
            llm_args, SimpleNamespace(model=SimpleNamespace(model_config=resolved.model_config))
        )
    assert 103 in py_executor_creator._MLA_CHUNKED_PREFILL_SUPPORTED_SM_VERSIONS


def _manager_arguments(resolved, dtype) -> dict:
    """What `MistralLarge4CacheManager.__init__` (attention/backends/sparse/mistral_large4/
    cache_manager.py) hands the DSA V2 manager it builds on, given what `_create_kv_cache_manager`
    (pyexecutor/_util.py) passes it for this engine: the LLM API's kv_cache_config, one 576-wide
    MLA head on each of the 72 layers, the latent ``dtype``. The base is not built (it allocates
    the pool on the device): its constructor records its arguments and stops."""

    class _Recorded(Exception):
        pass

    seen = {}

    def record(self, kv_cache_config, kv_cache_type, **kwargs):
        seen.update(kwargs, kv_cache_config=kv_cache_config, kv_cache_type=kv_cache_type)
        raise _Recorded

    model_config = resolved.model_config
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DSACacheManagerV2, "__init__", record)
        with pytest.raises(_Recorded):
            MistralLarge4CacheManager(
                resolved.llm_args.kv_cache_config,
                CacheType.SELFKONLY,
                num_layers=_LAYERS,
                num_kv_heads=1,
                head_dim=576,
                tokens_per_block=_TOKENS_PER_BLOCK,
                max_seq_len=resolved.max_seq_len,
                max_batch_size=resolved.max_batch_size,
                mapping=resolved.mapping,
                dtype=dtype,
                spec_config=None,
                layer_mask=None,
                max_num_tokens=resolved.max_num_tokens,
                max_beam_width=1,
                sparse_attention_config=model_config.sparse_attention_config,
                pretrained_config=model_config.pretrained_config,
            )
    return seen


def test_kv_cache_dtype_auto_and_fp8_build_the_fp8_ds_mla_pool(envelope, gate):
    """kv_cache_config.dtype auto (the CI class: the checkpoint's fp8_kv_cache) and fp8 (the
    gates: spec-1m-kvds-tep4/options/core.yaml, the runtime's --kv_cache_dtype=fp8) build the
    same cache. Both resolve the latent to FP8 (`kv_cache_quant_algo`, so `_create_kv_cache_
    manager` hands the manager `DataType.FP8`, `quant_mode.has_fp8_kv_cache()`); the sparse
    registry picks `MistralLarge4CacheManager`, whose `latent_layout` is fp8_ds_mla, at 47136
    bytes a token; and the manager builds its base over all 72 layers, one 656-byte head on each
    of the 63 KV owners and none on the nine sharers, as FP8, with the dtype it was asked for. A
    latent dtype the layout cannot hold, bf16, is refused at construction."""
    for resolved, requested in ((envelope, "auto"), (gate, "fp8")):
        model_config, llm_args = resolved.model_config, resolved.llm_args
        assert llm_args.kv_cache_config.dtype == requested
        assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert model_config.quant_config.quant_mode.has_fp8_kv_cache()
        manager = get_kv_cache_manager_cls(model_config, llm_args.kv_cache_config)
        assert manager is MistralLarge4CacheManager and manager.latent_layout == "fp8_ds_mla"
        assert (_kv_cost(resolved).slope, _kv_cost(resolved).intercept) == (47136, 0)

        sources = model_config.pretrained_config.kv_source_layers
        base = _manager_arguments(resolved, tensorrt_llm.bindings.DataType.FP8)
        assert (base["num_layers"], base["head_dim"]) == (_LAYERS, 656)
        assert base["num_kv_heads"] == [1 if s == i else None for i, s in enumerate(sources)]
        assert base["num_kv_heads"].count(1) == 63
        assert base["dtype"] == tensorrt_llm.bindings.DataType.FP8
        assert base["layer_mask"] is None
        assert base["kv_cache_config"].dtype == requested
        assert (base["tokens_per_block"], base["max_seq_len"]) == (32, _MAX_SEQ_LEN)
        with pytest.raises(ValueError, match="fp8_ds_mla"):
            _manager_arguments(resolved, tensorrt_llm.bindings.DataType.BF16)


def test_model_engine_keeps_max_seq_len(envelope, delivered):
    """`PyTorchModelEngine._init_model_capacity` (pyexecutor/model_engine.py) lowers a
    max_seq_len above the model's inferred limit and a max_num_tokens above max_batch_size x
    max_seq_len. The target's shell infers the limit with `DecoderModelForCausalLM
    .infer_max_seq_len`: YaRN does not scale it, so it is max_position_embeddings, 1048576, and
    the 1M config's max_seq_len is kept whole, as the delivered one is."""
    for resolved in (envelope, delivered):
        cfg = resolved.model_config.pretrained_config
        inferred = DecoderModelForCausalLM.infer_max_seq_len(SimpleNamespace(config=cfg))
        assert inferred == cfg.max_position_embeddings == _MAX_SEQ_LEN

        engine = SimpleNamespace(
            model=SimpleNamespace(infer_max_seq_len=lambda inferred=inferred: inferred),
            max_seq_len=resolved.max_seq_len,
            max_num_tokens=resolved.max_num_tokens,
            batch_size=resolved.max_batch_size,
        )
        for name in ("_init_max_seq_len", "_init_max_num_tokens", "_infer_max_seq_len_from_config"):
            setattr(engine, name, MethodType(getattr(PyTorchModelEngine, name), engine))
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", raising=False)
            PyTorchModelEngine._init_model_capacity(engine)
        assert (engine.max_seq_len, engine.max_num_tokens) == (
            resolved.max_seq_len,
            resolved.max_num_tokens,
        )
    assert envelope.model_config.max_seq_len == _MAX_SEQ_LEN
    assert envelope.model_config.max_num_tokens == _MAX_NUM_TOKENS


def _check_arguments(resolved, prompt_len: int) -> None:
    llm = SimpleNamespace(args=resolved.llm_args)
    llm._check_one_model_speculative_sampling = MethodType(
        BaseLLM._check_one_model_speculative_sampling, llm
    )
    BaseLLM._check_arguments(llm, prompt_len, SamplingParams(max_tokens=16), is_gen_only=False)


def _enqueue(resolved, prompt_len: int, max_tokens: int):
    """The executor request `BaseWorker._enqueue_request` hands the engine for a prompt of
    ``prompt_len`` tokens."""
    enqueued = []
    worker = SimpleNamespace(
        _lora_manager=None,
        _is_pytorch_backend=True,
        max_seq_len=resolved.max_seq_len,
        mapping=resolved.mapping,
        engine=SimpleNamespace(enqueue_request=lambda request, **_: enqueued.append(request) or 1),
    )
    request = GenerationRequest(
        [1] * prompt_len, sampling_params=SamplingParams(max_tokens=max_tokens, end_id=2)
    )
    request.set_id(1)
    BaseWorker._enqueue_request(worker, request)
    return enqueued[0]


def test_llm_api_admits_the_envelope_prompt(envelope, delivered):
    """`BaseLLM._check_arguments` (llmapi/llm.py) refuses a prompt longer than max_num_tokens only
    without chunked prefill: the delivered engine refuses 32769 tokens, the 1M engine admits the
    1M gate's 1048559. The bound on max_seq_len is the worker's: `BaseWorker
    ._enqueue_request` (executor/base_worker.py) gives a request at most max_seq_len less its
    prompt to generate, and refuses a prompt that leaves no room for one token."""
    _check_arguments(envelope, _ENVELOPE_PROMPT)
    _check_arguments(delivered, _MAX_NUM_TOKENS)
    with pytest.raises(RequestError, match=r"should not exceed max_num_tokens \(32768\)"):
        _check_arguments(delivered, _MAX_NUM_TOKENS + 1)

    assert _enqueue(envelope, _ENVELOPE_PROMPT, 16).max_tokens == 16
    assert _enqueue(envelope, _MAX_SEQ_LEN - 1, 16).max_tokens == 1
    for prompt_len in (_MAX_SEQ_LEN, _MAX_SEQ_LEN + 1):
        with pytest.raises(RequestError, match="maximum context length is 1048576 tokens"):
            _enqueue(envelope, prompt_len, 16)


# ------------------------------------------------------- (2) the context path of a long prompt


def test_chunked_prefill_hands_the_target_the_cached_kv_context_path(envelope, delivered):
    """`build_attention_metadata` (pyexecutor/engine/metadata.py) turns on
    `enable_context_mla_with_cached_kv` for MLA under cache reuse or chunked prefill: on at the
    1M config, off at the delivered engine. With it and `runtime_features.chunked_prefill`,
    `Indexer.prepare_for_chunked_prefill` (attention/backends/sparse/dsa/indexer.py) takes its
    MLA-chunked-prefill branch. The metadata is sized by the engine bounds, not max_seq_len."""
    for resolved, cached_kv in ((envelope, True), (delivered, False)):
        made = build_attention_metadata(
            resolved.model_config,
            max_batch_size=resolved.max_batch_size,
            max_num_tokens=resolved.max_num_tokens,
            max_beam_width=1,
            attention_backend=SimpleNamespace(Metadata=lambda **kwargs: SimpleNamespace(**kwargs)),
            attention_runtime_features=_runtime_features(resolved),
            mapping=resolved.mapping,
            cache_indirection=None,
        )
        assert made.enable_context_mla_with_cached_kv is cached_kv
        assert made.runtime_features.chunked_prefill is cached_kv
        assert made.runtime_features.cache_reuse is False
        assert made.runtime_features.chunk_size == _MAX_NUM_TOKENS
        assert (made.max_num_requests, made.max_num_tokens, made.max_num_sequences) == (
            16,
            32768,
            16,
        )
        assert made.sparse_metadata_params == _metadata_params(resolved)


def test_target_sizes_from_the_engine(monkeypatch, envelope, gate, delivered):
    """What the target takes from the engine on its first step, read off the metadata this engine
    builds (`_target_sizing`), whose runtime features are the executor's: chunked prefill as
    enable_chunked_prefill, block reuse as kv_cache_config.enable_block_reuse, off in every
    config here. The 1M config (max_seq_len 1048576, chunked prefill, 16 requests, in either KV
    dtype spelling): the indexer's logits budget at its 2**28 ceiling (`_prefill_logits_budget`,
    256 query rows at 1048576 keys), and a step may attend min(16, 32768) x 1048576 rows of
    context KV (`_most_ctx_kv`, chunked prefill bounds no gather), so the held context workspace
    takes its 2**19 ceiling (`_ctx_workspace_rows`). The delivered config (32768 unchunked, block
    reuse off): 2**24 logits (512 rows at 32768 keys), at most max_num_tokens of context KV a
    step, all of it new, and 65536 rows held, twice that."""
    monkeypatch.delenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", raising=False)
    for resolved, most, rows, budget in (
        (envelope, 16 * _MAX_SEQ_LEN, 2**19, 2**28),
        (gate, 16 * _MAX_SEQ_LEN, 2**19, 2**28),
        (delivered, _MAX_NUM_TOKENS, 2**16, 2**24),
    ):
        md = _target_sizing(resolved)
        assert (md.max_num_requests, md.max_num_tokens, md.max_seq_len) == (
            16,
            32768,
            resolved.max_seq_len,
        )
        features, llm_args = md.runtime_features, resolved.llm_args
        assert features.chunked_prefill is llm_args.enable_chunked_prefill is resolved.case.chunked
        assert features.cache_reuse is llm_args.kv_cache_config.enable_block_reuse is False
        assert tep4_modeling._most_ctx_kv(md) == most
        assert tep4_modeling._ctx_workspace_rows(md) == rows
        assert tep4_modeling._prefill_logits_budget(md.max_num_tokens, md.max_seq_len) == budget
    assert tep4_modeling._CTX_WORKSPACE_ROWS == 2**19
    assert tep4_modeling._PREFILL_LOGITS_CEILING == 2**28
    assert (2**28 // _MAX_SEQ_LEN, 2**24 // 32768) == (256, 512)


def test_target_sizes_block_reuse_from_the_executors_runtime_features(
    monkeypatch, envelope, delivered
):
    """Block reuse reaches the target through the attention runtime features the executor builds
    from the LLM API's kv_cache_config (`_runtime_features`: cache_reuse is
    kv_cache_config.enable_block_reuse). The delivered and the 1M options with block reuse on,
    resolved as the engine resolves them: the runtime features carry it, the metadata takes the
    cached-KV MLA path at both, and a step may attend min(16, 32768) x max_seq_len rows of
    context KV, cached prefixes included (`_most_ctx_kv`), so the held context workspace takes
    its 2**19 ceiling at both: the delivered config's 65536 rows grow to it, the 1M config's stay.
    The logits budget, from max_num_tokens and max_seq_len alone, is 2**24 and 2**28 as without
    block reuse."""
    monkeypatch.delenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", raising=False)
    for base, most, budget in (
        (delivered, 16 * _MAX_NUM_TOKENS, 2**24),
        (envelope, 16 * _MAX_SEQ_LEN, 2**28),
    ):
        options = base.case.options
        options["kv_cache_config"] = dict(options["kv_cache_config"], enable_block_reuse=True)
        resolved = _resolve(base.path, options)
        assert resolved.llm_args.kv_cache_config.enable_block_reuse is True
        assert (resolved.max_num_tokens, resolved.max_seq_len) == (
            base.max_num_tokens,
            base.max_seq_len,
        )
        md = _target_sizing(resolved)
        assert md.runtime_features.cache_reuse is True
        assert md.runtime_features.chunked_prefill is base.case.chunked
        assert md.enable_context_mla_with_cached_kv is True
        assert tep4_modeling._most_ctx_kv(md) == most
        assert tep4_modeling._ctx_workspace_rows(md) == 2**19
        assert tep4_modeling._prefill_logits_budget(md.max_num_tokens, md.max_seq_len) == budget


def _block_table(num_seqs: int, max_blocks: int) -> torch.Tensor:
    """A page table whose every entry is a distinct page id, none in its own row position."""
    pages = torch.arange(num_seqs * max_blocks, dtype=torch.int32).flip(0) * 3 + 7
    return pages.view(num_seqs, max_blocks).contiguous()


def _indexer_step(resolved, rank: int, ctx, gen=()):
    """One step's DSA metadata, prepared on the host by the engine's own code.

    ``ctx`` holds (cached, new) tokens per context request, ``gen`` the cached tokens of each
    decode row behind them. `DSAtrtllmAttentionMetadata.prepare_for_mla_rope_append`,
    `.prepare_for_skip_indexer` and `Indexer.prepare` run as `DSAtrtllmAttentionMetadata.prepare`
    calls them, on rank ``rank``'s mapping; DeepGEMM's decode schedule is recorded, not built.
    """
    params = _metadata_params(resolved)
    num_contexts, num_generations = len(ctx), len(gen)
    num_seqs = num_contexts + num_generations
    seq_lens = torch.tensor([new for _, new in ctx] + [1] * num_generations, dtype=torch.int32)
    cached = torch.tensor([c for c, _ in ctx] + list(gen), dtype=torch.int32)
    num_tokens, num_ctx_tokens = int(seq_lens.sum()), sum(new for _, new in ctx)
    topk = params.max_sparse_topk

    def buffer(*shape, dtype=torch.int64):
        return torch.zeros(*shape, dtype=dtype)

    md = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(index_head_dim=128, max_seq_len=resolved.max_seq_len),
        sparse_metadata_params=params,
        indexer_max_chunk_size=params.indexer_max_chunk_size,
        num_sparse_topk=topk,
        enable_indexer_skip=params.enable_indexer_skip,
        indexer_head_dim=params.index_head_dim,
        indexer_quant_block_size=128,
        compress_ratios=[1],
        _tokens_per_block=_TOKENS_PER_BLOCK,
        max_num_tokens=resolved.max_num_tokens,
        mapping=_rank_mapping(resolved.mapping, rank),
        enable_context_mla_with_cached_kv=resolved.llm_args.enable_chunked_prefill,
        runtime_features=_runtime_features(resolved),
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_seqs=num_seqs,
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_tokens,
        seq_lens=seq_lens,
        request_ids=list(range(num_seqs)),
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=cached.tolist(), num_extra_kv_tokens=0
        ),
        host_indexer_k_cache_block_offsets=_block_table(
            num_seqs, _max_blocks_per_seq(resolved.max_seq_len)
        ),
        host_slot_mapping_fp8=buffer(num_tokens),
        host_slot_mapping_scale=buffer(num_tokens),
        slot_mapping_fp8=buffer(num_tokens),
        slot_mapping_scale=buffer(num_tokens),
        cu_seqlen_ks=buffer(max(num_ctx_tokens, 1), dtype=torch.int32),
        cu_seqlen_ke=buffer(max(num_ctx_tokens, 1), dtype=torch.int32),
        host_ctx_cached_token_indptr=buffer(num_contexts + 1),
        ctx_cached_token_indptr=buffer(num_contexts + 1),
        host_ctx_kv_indptr=buffer(num_contexts + 1),
        ctx_kv_indptr=buffer(num_contexts + 1),
        host_gen_cached_token_indptr=buffer(num_generations + 1),
        gen_cached_token_indptr=buffer(num_generations + 1),
        host_gen_kv_indptr=buffer(num_generations + 1),
        gen_kv_indptr=buffer(num_generations + 1),
        host_topk_indices_buffer=buffer(num_tokens, topk, dtype=torch.int32),
        topk_indices_buffer=buffer(num_tokens, topk, dtype=torch.int32),
        indexer_prefill_chunks=None,
        indexer_prefill_q_split=None,
        decode_schedules=[],
    )
    md.prepare_dense_topk_indices = MethodType(
        DSAtrtllmAttentionMetadata.prepare_dense_topk_indices, md
    )
    md._get_dense_topk_indices = MethodType(DSAtrtllmAttentionMetadata._get_dense_topk_indices, md)
    kv_lens = cached + seq_lens
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dsa_indexer, "maybe_pin_memory", lambda tensor: tensor)
        mp.setattr(dsa_indexer, "prefer_pinned", lambda: False)
        mp.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
        mp.setattr(
            Indexer,
            "prepare_scheduler_metadata",
            staticmethod(
                lambda metadata: metadata.decode_schedules.append(metadata.num_generations)
            ),
        )
        DSAtrtllmAttentionMetadata.prepare_for_mla_rope_append(md, cached, kv_lens)
        DSAtrtllmAttentionMetadata.prepare_for_skip_indexer(md, kv_lens)
        Indexer.prepare(md)
    return md


def _key_slots(block_table: torch.Tensor, row: int, positions: torch.Tensor):
    """Where an fp8 indexer key and its fp32 scale live: pages of 32 tokens of 128 key bytes
    followed by 32 scales of 4 bytes, 4224 bytes a page."""
    pages = block_table[row, positions // _TOKENS_PER_BLOCK].to(torch.int64)
    offset = positions % _TOKENS_PER_BLOCK
    page_bytes = _TOKENS_PER_BLOCK * (128 + 4)
    assert page_bytes == 4224
    return (
        pages * page_bytes + offset * 128,
        pages * page_bytes + _TOKENS_PER_BLOCK * 128 + offset * 4,
    )


# The chunks of the 1M gate's prompt that the scheduler admits alone: 31 of 32768 tokens
# and a last of 32751.
_CHUNKS = [_MAX_NUM_TOKENS] * 31 + [_ENVELOPE_PROMPT - 31 * _MAX_NUM_TOKENS]


@pytest.mark.parametrize("chunk", [0, 1, 2, 31], ids=["first", "second", "third", "last"])
def test_later_chunks_carry_their_cached_prefix(envelope, chunk):
    """Every chunk of one request after its first is a context request with a cached prefix.
    `DSAtrtllmAttentionMetadata.prepare_for_mla_rope_append` (attention/backends/sparse/dsa/
    metadata.py) hands the target its cached and total prefix sums, `ctx_cached_token_indptr`
    [0, cached] and `ctx_kv_indptr` [0, cached + new], `num_ctx_cached_tokens` = cached and
    `max_ctx_seq_len` = new; `Indexer.prepare` counts `num_ctx_kv_tokens` = new keys written this
    step, and `Indexer.recompute_context_kv_gather_mappings` builds `slot_mapping_fp8_fullkv` /
    `slot_mapping_scale_fullkv` over the cached and the new keys, position p at page p // 32 of
    the request's indexer page table. On a first chunk there is nothing cached and the full
    mappings are the step's own."""
    cached, new = sum(_CHUNKS[:chunk]), _CHUNKS[chunk]
    md = _indexer_step(envelope, 0, ctx=[(cached, new)])
    total = cached + new
    assert md.host_ctx_cached_token_indptr.tolist() == [0, cached]
    assert md.ctx_cached_token_indptr.tolist() == [0, cached]
    assert md.host_ctx_kv_indptr.tolist() == [0, total]
    assert md.ctx_kv_indptr.tolist() == [0, total]
    assert (md.num_ctx_cached_tokens, md.max_ctx_seq_len, md.max_ctx_kv_len) == (cached, new, total)
    assert md.num_ctx_mla_kv_tokens == total
    assert md.num_ctx_kv_tokens == new
    # The target's context workspace holds the request's whole KV, the step's cached count plus
    # its context tokens (`PrefillTarget._ctx_workspace`), placed by the total prefix sums.
    assert md.num_ctx_cached_tokens + md.num_ctx_tokens == int(md.ctx_kv_indptr[-1]) == total
    assert md.skip_indexer_for_ctx_reqs is False

    table = md.host_indexer_k_cache_block_offsets
    fp8, scale = _key_slots(table, 0, torch.arange(cached, total))
    assert torch.equal(md.slot_mapping_fp8, fp8) and torch.equal(md.slot_mapping_scale, scale)
    if chunk == 0:
        assert md.slot_mapping_fp8_fullkv is md.slot_mapping_fp8
        assert md.slot_mapping_scale_fullkv is md.slot_mapping_scale
    else:
        fp8, scale = _key_slots(table, 0, torch.arange(total))
        assert md.slot_mapping_fp8_fullkv.shape == (total,)
        assert torch.equal(md.slot_mapping_fp8_fullkv, fp8)
        assert torch.equal(md.slot_mapping_scale_fullkv, scale)
        assert torch.equal(md.slot_mapping_fp8_fullkv[cached:], md.slot_mapping_fp8)
    # The last chunk ends one token short of max_seq_len less the tokens it generates.
    assert total <= _ENVELOPE_PROMPT and (chunk != 31 or total == _ENVELOPE_PROMPT)


# ------------------------------------------------------- (3) the query split across TP ranks


def test_q_split_threshold_is_8192(envelope):
    """`DeepSeekSparseAttentionConfig.q_split_threshold` (llmapi/llm_args.py) defaults to 8192 and
    reaches `DSAMetadataParams`, which `_prefill_q_split_threshold` (dsa/indexer.py) reads; the
    split is on, not disabled by a negative value."""
    params = _metadata_params(envelope)
    assert envelope.model_config.sparse_attention_config.q_split_threshold == 8192
    assert params.q_split_threshold == 8192
    assert (
        dsa_indexer._prefill_q_split_threshold(SimpleNamespace(sparse_metadata_params=params))
        == 8192
    )


# context tokens in a step -> per-rank sizes of the split (None: no split)
_SPLITS = {
    8191: None,
    8192: [2048, 2048, 2048, 2048],
    8193: [2048, 2048, 2048, 2049],
    32767: [8191, 8192, 8192, 8192],
    32768: [8192, 8192, 8192, 8192],
    # several short prompts, 3000 + 3000 + 3000 + 500 tokens
    9500: [2375, 2375, 2375, 2375],
}


@pytest.mark.parametrize("count", list(_SPLITS))
def test_split_prefill_queries_per_rank(envelope, count):
    """`_split_prefill_queries` (dsa/indexer.py) on each TP rank's mapping: below 8192 context
    tokens every rank takes them all; from 8192 on, rank r takes [count r / 4, count (r + 1) / 4)
    and every rank knows every share's size. Attention DP (dep4's mapping) never splits."""
    sizes = _SPLITS[count]
    begins = [0] if sizes is None else [sum(sizes[:r]) for r in range(_RANKS + 1)]
    for rank in range(_RANKS):
        mapping = _rank_mapping(envelope.mapping, rank)
        assert mapping.tp_rank == rank
        split = dsa_indexer._split_prefill_queries(count, mapping, 8192)
        if sizes is None:
            assert split == (0, count, None)
        else:
            assert split == (begins[rank], begins[rank + 1], sizes)
    adp = copy.deepcopy(envelope.mapping)
    adp.enable_attention_dp = True
    assert dsa_indexer._split_prefill_queries(count, adp, 8192) == (0, count, None)


def _covered(chunks, q_lens):
    q_starts = [sum(q_lens[:i]) for i in range(len(q_lens))]
    spans = []
    for chunk in chunks:
        for req_idx, start, end, req_start in chunk:
            assert req_start == q_starts[req_idx]
            spans.append((req_start + start, req_start + end))
    return spans


@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        pytest.param([32768], [3 * 32768], id="one-chunk-over-a-cached-prefix"),
        pytest.param([32751], [_ENVELOPE_PROMPT], id="last-chunk"),
        pytest.param([3000, 3000, 3000, 500], [3000, 3000, 3000, 500], id="short-prompts"),
        pytest.param([2048] * 16, [2048] * 16, id="sixteen-prompts"),
        pytest.param([600] * 15 + [23768], [600] * 15 + [23768 + 5 * 32768], id="mixed-lengths"),
    ],
)
def test_chunk_plan_covers_each_rank_share(envelope, delivered, q_lens, kv_lens):
    """`_packed_logits_elem_budget` (dsa/indexer.py) is 1 << 28 at both configs (the packed cap;
    `_INDEXER_MQA_LOGITS_ELEM_BUDGET` x max_num_tokens x max_seq_len do not bind). Under it
    `plan_mla_chunked_prefill_chunks` covers exactly the rank's share [q_begin, q_end) of the
    context tokens, in order, once each; a request the share boundary cuts is a chunk of its
    own, and whole requests share a chunk only within the budget."""
    for resolved in (envelope, delivered):
        budget = dsa_indexer._packed_logits_elem_budget(
            SimpleNamespace(
                max_num_tokens=resolved.max_num_tokens,
                kv_cache_manager=SimpleNamespace(max_seq_len=resolved.max_seq_len),
            )
        )
        assert budget == dsa_indexer._INDEXER_PACKED_LOGITS_ELEM_BUDGET == 1 << 28
    assert dsa_indexer._INDEXER_MQA_LOGITS_ELEM_BUDGET == 1 << 31

    total = sum(q_lens)
    covered = []
    for rank in range(_RANKS):
        q_begin, q_end, sizes = dsa_indexer._split_prefill_queries(
            total, _rank_mapping(envelope.mapping, rank), 8192
        )
        assert sizes is not None
        chunks = plan_mla_chunked_prefill_chunks(q_lens, kv_lens, 1 << 28, q_begin, q_end)
        spans = _covered(chunks, q_lens)
        assert spans[0][0] == q_begin and spans[-1][1] == q_end
        assert all(prev[1] == cur[0] for prev, cur in zip(spans, spans[1:]))
        for chunk in chunks:
            whole = all(start == 0 and end == q_lens[i] for i, start, end, _ in chunk)
            if len(chunk) > 1:
                assert whole
                assert sum(q_lens[i] for i, *_ in chunk) * sum(kv_lens[i] for i, *_ in chunk) <= (
                    1 << 28
                )
            elif not whole:
                req_idx, start, end, req_start = chunk[0]
                assert req_start + start == q_begin or req_start + end == q_end
        covered += spans
    assert covered[0][0] == 0 and covered[-1][1] == total
    assert all(prev[1] == cur[0] for prev, cur in zip(covered, covered[1:]))


@pytest.mark.parametrize("chunk", [1, 31], ids=["second", "last"])
def test_each_rank_plans_its_share_of_a_chunk(envelope, chunk):
    """Through `Indexer.prepare` on each TP rank: a chunk of the long prompt (32768 or 32751
    tokens over its cached prefix) sets `indexer_prefill_q_split` = (q_begin, q_end, sizes) and
    one `indexer_prefill_chunks` entry for the rank's share, queries [q_begin, q_end) of the
    chunk against keys [0, cached + q_end) of the request, each query's window ending at its own
    position. The four shares tile the chunk."""
    cached, new = sum(_CHUNKS[:chunk]), _CHUNKS[chunk]
    sizes = _SPLITS[32768] if new == 32768 else [8187, 8188, 8188, 8188]
    assert sum(sizes) == new
    for rank in range(_RANKS):
        md = _indexer_step(envelope, rank, ctx=[(cached, new)])
        q_begin, q_end = sum(sizes[:rank]), sum(sizes[: rank + 1])
        assert md.indexer_prefill_q_split == (q_begin, q_end, sizes)
        assert len(md.indexer_prefill_chunks) == 1
        plan = md.indexer_prefill_chunks[0]
        assert (plan.token_start, plan.token_end) == (q_begin, q_end)
        assert (plan.k_token_start, plan.k_token_end) == (0, cached + q_end)
        assert torch.equal(plan.cu_seqlen_ks, torch.zeros(q_end - q_begin, dtype=torch.int32))
        assert torch.equal(
            plan.cu_seqlen_ke,
            torch.arange(cached + q_begin + 1, cached + q_end + 1, dtype=torch.int32),
        )
        assert md.decode_schedules == []


def test_q_split_on_skipped_and_mixed_steps(envelope):
    """What `Indexer.prepare_for_chunked_prefill` does on the two steps the target must not
    misread. (a) A step whose every context is within index_topk 2048: `prepare_for_skip_indexer`
    sets `skip_indexer_for_ctx_reqs` and fills the dense top-k rows, and the engine STILL sets
    `indexer_prefill_q_split` once the context tokens reach 8192 (and plans the share's chunks);
    only the full-KV gather mappings are skipped. A target that gathers on the split must not do
    so on a skipped step. (b) A mixed step, one chunk with 15 decode rows behind it: the split
    counts the context tokens only and covers the context rows; the decode rows keep their own
    schedule."""
    skipped = _indexer_step(envelope, 1, ctx=[(0, 1000)] * 9)
    assert skipped.skip_indexer_for_ctx_reqs is True
    assert skipped.indexer_prefill_q_split == (2250, 4500, [2250] * 4)
    # Rank 1's share cuts the third prompt after 250 tokens and the fifth after 500.
    assert [(c.token_start, c.token_end) for c in skipped.indexer_prefill_chunks] == [
        (2250, 3000),
        (3000, 4000),
        (4000, 4500),
    ]
    assert skipped.slot_mapping_fp8_fullkv is skipped.slot_mapping_fp8
    dense = skipped.host_topk_indices_buffer[:9000]
    assert torch.equal(dense[999, :1000], torch.arange(1000, dtype=torch.int32))
    assert bool((dense[999, 1000:] == -1).all())

    cached, new = 2 * 32768, 32768 - 32
    decode = [1048000 - 64 * i for i in range(15)]
    mixed = _indexer_step(envelope, 3, ctx=[(cached, new)], gen=decode)
    assert (mixed.num_contexts, mixed.num_generations, mixed.num_tokens) == (1, 15, 32751)
    assert mixed.indexer_prefill_q_split == (3 * 8184, 32736, [8184] * 4)
    (plan,) = mixed.indexer_prefill_chunks
    assert (plan.token_start, plan.token_end, plan.k_token_end) == (24552, 32736, cached + 32736)
    assert mixed.decode_schedules == [15]
    assert mixed.max_gen_seq_len == 1
    assert mixed.host_gen_kv_indptr[-1].item() == sum(decode) + 15


def test_context_rows_take_the_path_of_their_kv(envelope):
    """The target's context-row path (`PrefillTarget._ctx_workspace`) on the engine's own host
    metadata at the 1M config, rank by rank, over the held workspace's rows the target takes from
    this engine (`_ctx_workspace_rows`: 2**19, chunked prefill letting a step attend cached
    prefixes). It reads a step's context KV as cached + new over its context requests
    (`_ctx_kv_offsets`, the host `ctx_kv_indptr[num_contexts]`), never `num_ctx_kv_tokens`, which
    counts the new tokens only:

    * the KV-capacity estimate's step, one fresh 32768-token prompt split 8192 a rank, takes the
      workspace path, and that first context step allocates the held bf16 [2**19, 576] workspace;
    * the 1M gate prompt's chunks, each a query-split step with the indexer running: the first
      16 (context KV up to 2**19) on the workspace path, from the 17th on the decode path in
      sub-batches of 8192, the last (32751 rows behind 1015808 cached) 8192 x 3 and 8175;
    * one row of context KV either side of 2**19, with a decode row behind;
    * a packed step of four chunks over 1032192, 524288, 0 and 2048 cached keys with twelve
      decode rows behind them: the decode path over the 32752 context rows only;
    * nine prompts within index_topk (the context indexer skipped, the split still planned): the
      workspace path.

    A workspace-path step keeps layer 35's gather beside its rows when its context KV fits the
    held rows twice (the first 8 chunks), and gathers it again for each sharer otherwise. The
    workspace is never reallocated after the estimate's step."""
    rows_b = tep4_modeling._ctx_workspace_rows(_target_sizing(envelope))
    assert (rows_b, tep4_modeling._CTX_DECODE_ROWS) == (2**19, 8192)
    target = tep4_modeling.PrefillTarget.__new__(tep4_modeling.PrefillTarget)
    target.core = SimpleNamespace(lat_dim=576, ctx_workspace_rows=rows_b)

    def path(rank, ctx, gen=()):
        md = _indexer_step(envelope, rank, ctx=ctx, gen=gen)
        step = dict(
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            **tep4_modeling._ctx_kv_offsets(md, md.num_contexts),
        )
        n_kv = sum(c + n for c, n in ctx)
        assert int(md.host_ctx_kv_indptr[md.num_contexts]) == n_kv
        assert not any(c for c, _ in ctx) or md.num_ctx_kv_tokens == md.num_ctx_tokens < n_kv
        ws = target._ctx_workspace(step, md.num_ctx_tokens, torch.device("cpu"))
        split = md.indexer_prefill_q_split
        if ws.ws_buf is not None:
            assert n_kv <= rows_b and ws.ws_buf.shape == (n_kv, 576)
            assert ws.ws_buf.data_ptr() == target._ws_buf.data_ptr()
            # Layer 35's gather is kept beside the step's rows where they fit twice.
            assert (ws.kept is not None) == (2 * n_kv <= rows_b)
            return "workspace", split, None
        assert n_kv > rows_b and sorted(ws.sched) == sorted({hi - lo for lo, hi in ws.sub_batches})
        return "decode", split, ws.sub_batches

    full = [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
    assert path(1, [(0, _MAX_NUM_TOKENS)]) == ("workspace", (8192, 16384, [8192] * 4), None)
    held = target._ws_buf
    assert held.shape == (rows_b, 576) and held.dtype == torch.bfloat16
    for k, new in enumerate(_CHUNKS):
        got = path(k % _RANKS, [(sum(_CHUNKS[:k]), new)])
        q_begin, q_end, sizes = got[1]
        assert sum(sizes) == new and q_end - q_begin == sizes[k % _RANKS]
        if k < 16:
            assert got[0] == "workspace", k
        elif k < 31:
            assert got[0::2] == ("decode", full), k
        else:
            assert got[0::2] == ("decode", full[:3] + [(24576, 32751)])
    edge = rows_b - 32767
    assert path(0, [(edge, 32767)], gen=[4096])[0] == "workspace"
    assert path(0, [(edge + 1, 32767)], gen=[4096])[0::2] == ("decode", full[:3] + [(24576, 32767)])
    packed = [(1032192, 16384), (524288, 8192), (0, 4096), (2048, 4080)]
    got = path(2, packed, gen=[1048000 - 64 * i for i in range(12)])
    assert got[0::2] == ("decode", full[:3] + [(24576, 32752)])
    assert got[1] is not None
    skipped = path(3, [(0, 1000)] * 9)
    assert skipped == ("workspace", (6750, 9000, [2250] * 4), None)
    assert target._ws_buf is held


# ------------------------------------------------------- (4) the scheduler's chunks


def _scheduler_pool() -> Mock:
    """The surface of a KVCacheManagerV2 the V2 scheduler reads, with room for everything."""
    pool = Mock()
    pool.kv_connector_manager = None
    pool.tokens_per_block = _TOKENS_PER_BLOCK
    pool.block_reuse_policy = BlockReusePolicy.PER_REQUEST
    pool.enable_partial_reuse = True
    pool.enable_joint_kv_cache_reuse = False
    pool.enable_block_reuse = False
    pool.num_extra_kv_tokens = 0
    pool._spec_recompute_tail = 0
    pool.is_draft = False
    pool.context_replay_tokens.return_value = None
    pool.can_evict = False
    pool._has_cp_helix = False
    pool.is_vswa = False
    pool.probe_first_new_block_key.return_value = None
    pool.kv_cache_map = {}
    pool.prepare_context.return_value = True
    pool.resize_context.return_value = True
    pool.try_allocate_generation.return_value = True
    pool.has_cache_tier_below_gpu = True
    pool.has_pending_preemption.return_value = False
    return pool


def _request(request_id: int, state: LlmRequestState, **fields) -> Mock:
    request = Mock()
    request.request_id = request.py_request_id = request_id
    request.state_value = state.value
    request.py_csa2_remote_tail_mode = None
    request.py_encoder_output_ready_event = None
    request.py_multimodal_data = None
    request.py_ced_replay = None
    request.lora_task_id = None
    request.num_draft_tokens = 0
    request.has_draft_tokens = False
    request.py_draft_tokens = []
    request.get_beam_width_by_iter.return_value = 1
    request.is_context_init_state = state == LlmRequestState.CONTEXT_INIT
    request.is_generation_in_progress_state = state == LlmRequestState.GENERATION_IN_PROGRESS
    request.is_generation_to_complete_state = False
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def _schedule_prompt(envelope, prompt_len: int, decode_rows: int) -> list[tuple[int, int]]:
    """Steps of the scheduler `create_py_executor_instance` (pyexecutor/_util.py) builds for a V2
    KV manager, admitting one prompt of ``prompt_len`` tokens while ``decode_rows`` requests
    decode beside it: (context tokens, generation rows) per step."""
    llm_args = envelope.llm_args
    pool = _scheduler_pool()
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(pool),
    ):
        scheduler = KVCacheV2Scheduler(
            max_batch_size=envelope.max_batch_size,
            max_num_tokens=envelope.max_num_tokens,
            kv_cache_manager=pool,
            scheduler_policy=llm_args.scheduler_config.capacity_scheduler_policy,
            # `_create_py_executor` (py_executor_creator.py): first come first served, in units of
            # tokens_per_block when no attention window is shorter than max_seq_len.
            ctx_chunk_config=(ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, _TOKENS_PER_BLOCK),
            scheduler_capacity=envelope.max_batch_size,
        )
    assert scheduler.policy == CapacitySchedulerPolicy.MAX_UTILIZATION
    assert (scheduler.chunking_enabled, scheduler.chunk_unit_size) == (True, _TOKENS_PER_BLOCK)
    assert scheduler.max_context_length == _MAX_NUM_TOKENS

    prompt = _request(
        0,
        LlmRequestState.CONTEXT_INIT,
        prompt_len=prompt_len,
        context_remaining_length=prompt_len,
        context_current_position=0,
        context_chunk_size=0,
        py_served_prefix_end=0,
        expect_snapshot_points=[],
        is_first_context_chunk=True,
        is_last_context_chunk=False,
        encoder_output_len=None,
        py_skip_cross_kv_projection=False,
    )
    decoding = [
        _request(1 + i, LlmRequestState.GENERATION_IN_PROGRESS, is_first_context_chunk=False)
        for i in range(decode_rows)
    ]
    steps = []
    while prompt.context_remaining_length > 0:
        out = scheduler.schedule_request([prompt] + decoding, set())
        assert out.context_requests == [prompt]
        chunk = prompt.context_chunk_size
        steps.append((chunk, len(out.generation_requests)))
        prompt.context_remaining_length -= chunk
        prompt.context_current_position += chunk
        prompt.is_first_context_chunk = False
    return steps


def test_scheduler_admits_the_envelope_prompt_in_chunks(envelope):
    """`KVCacheV2Scheduler` (pyexecutor/scheduler/scheduler_v2.py), the scheduler of a V2 KV
    manager, with the chunking `_create_py_executor` configures. It schedules the decode rows
    first, then cuts the context chunk from what the step's 32768 tokens leave, rounded down to
    the 32-token unit unless it is the prompt's last. The configured GUARANTEED_NO_EVICT is
    served as MAX_UTILIZATION. Alone, the 1048559-token prompt takes 32 steps, 31 of 32768 and a
    last of 32751; beside 15 decode rows each chunk is 32736, context and decode rows 32751 of
    the 32768, and the prompt takes 33 steps, the last 1007."""
    assert envelope.llm_args.scheduler_config.capacity_scheduler_policy == (
        CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    )
    alone = _schedule_prompt(envelope, _ENVELOPE_PROMPT, 0)
    assert alone == [(c, 0) for c in _CHUNKS]
    assert len(alone) == 32 and alone[-1] == (32751, 0)

    beside = _schedule_prompt(envelope, _ENVELOPE_PROMPT, 15)
    assert beside == [(32736, 15)] * 32 + [(1007, 15)]
    assert (32768 - 15) // 32 * 32 == 32736
    assert all(chunk + rows <= _MAX_NUM_TOKENS for chunk, rows in beside)
    assert sum(chunk for chunk, _ in beside) == _ENVELOPE_PROMPT


# ------------------------------------------------------- (5) what max_seq_len sizes


def test_kv_cache_sizing(resolved):
    """The DSA pool per rank (`MistralLarge4CacheManager.get_cache_size_per_token`, attention/
    backends/sparse/mistral_large4/cache_manager.py): the 63 KV owners keep an fp8_ds_mla latent
    of 656 bytes (512 e4m3, four fp32 tile scales, 64 bf16 rope), the 44 full-indexer layers an
    indexer key of 128 fp8 bytes and one fp32 scale, pages of 32; the nine sharers keep no
    latent. 47136 bytes a token on every rank, whatever max_seq_len; 16 requests at the
    engine's sequence length need that per token times their tokens."""
    model_config, llm_args = resolved.model_config, resolved.llm_args
    kv_cache_config = llm_args.kv_cache_config
    manager = get_kv_cache_manager_cls(model_config, kv_cache_config)
    assert manager is MistralLarge4CacheManager
    assert kv_cache_config.tokens_per_block == _TOKENS_PER_BLOCK

    cfg = model_config.pretrained_config
    assert cfg.kv_lora_rank + cfg.qk_rope_head_dim == 576
    assert sum(1 for i, s in enumerate(cfg.kv_source_layers) if s == i) == 63
    assert cfg.index_topk_pattern.count("F") == 44
    for rank in range(_RANKS):
        cost = _kv_cost(resolved, rank)
        assert (cost.slope, cost.intercept) == (656 * 63 + 132 * 44, 0) == (47136, 0)
        tokens = resolved.max_batch_size * resolved.max_seq_len
        pool = cost.bytes_for_tokens(tokens)
        assert pool == tokens * 47136
        if resolved.max_seq_len == _MAX_SEQ_LEN:
            assert tokens == 16 * _MAX_SEQ_LEN == 16_777_216
            assert pool == 16_777_216 * 47136 == 790_810_853_376
            assert pool / 2**30 == 736.5
        if resolved.case is _NVFP4_32K:
            assert tokens == 524288
            assert pool == 524288 * 47136 == 24_712_839_168
            assert math.isclose(pool / 2**30, 23.016, abs_tol=5e-4)


class _Pool(MistralLarge4CacheManager):
    """The fp8_ds_mla KV manager as the metadata reads it at construction: its page geometry
    only, its local layers the 63 KV owners."""

    tokens_per_block = _TOKENS_PER_BLOCK
    num_pools = 1
    num_attention_op_pools = 1
    num_local_layers = 63
    host_kv_cache_block_offsets = None
    dtype = tensorrt_llm.bindings.DataType.FP8
    get_fp4_mla_page_table_spec = None
    use_fp8_ds_mla = False

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.max_blocks_per_seq = _max_blocks_per_seq(max_seq_len)


def _metadata_buffers(resolved) -> dict:
    """Every buffer `DSAtrtllmAttentionMetadata.__post_init__` (dsa/metadata.py), with
    `TrtllmAttentionMetadata._post_init_with_buffers` (attention/backends/trtllm.py), allocates
    for this engine: cache name -> (shape, dtype). Allocated on the meta device."""
    allocated = {}

    def get_empty(buffers, tensor_shape, dtype, cache_name, capture_graph=False):
        allocated[cache_name] = (tuple(tensor_shape), dtype)
        return torch.empty(tuple(tensor_shape), dtype=dtype, device="meta")

    md = object.__new__(DSAtrtllmAttentionMetadata)
    md.__dict__.update(
        max_num_requests=resolved.max_batch_size,
        max_num_sequences=resolved.max_batch_size,
        max_num_tokens=resolved.max_num_tokens,
        kv_cache_manager=object.__new__(_Pool),
        draft_kv_cache_manager=None,
        mapping=resolved.mapping,
        runtime_features=_runtime_features(resolved),
        enable_context_mla_with_cached_kv=resolved.llm_args.enable_chunked_prefill,
        enable_flash_mla=False,
        sparse_metadata_params=_metadata_params(resolved),
        is_cross=False,
        cross=None,
        is_cuda_graph=False,
        cuda_graph_buffers=None,
        workspace=torch.empty(0, device="meta"),
        cuda_graph_workspace=torch.empty(0, device="meta"),
        max_draft_tokens=0,
        num_sms=148,
        enable_ragged_verification=False,
        ragged_verify_lens=None,
        _radix_rows_per_sequence=1,
    )
    _Pool.__init__(md.kv_cache_manager, resolved.max_seq_len)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AttentionMetadata, "get_empty", staticmethod(get_empty))
        mp.setattr(dsa_metadata, "prefer_pinned", lambda: False)
        mp.setattr(trtllm_backend, "prefer_pinned", lambda: False)
        mp.setattr(dsa_metadata, "get_sm_version", lambda: 103)
        md.__post_init__()
    assert md.use_cute_dsl_topk is False and md.needs_gvr_prior is False
    return allocated


def test_buffers_sized_by_max_seq_len(envelope, delivered):
    """The KV manager's page tables are max_seq_len wide: `KVCacheManagerV2.__init__`
    (pyexecutor/kv_cache/kv_cache_manager_v2.py) takes max_seq_len plus one token of generation
    headroom (`_get_generation_kv_capacity`) in pages of 32 and pads to a multiple of 4: 32772
    columns at 1048576 (1028 at 32768). The attention metadata the target reads (`kv_cache_block_
    offsets`, the DSA `block_table`, `indexer_k_cache_block_offsets`, `block_table_expanded`) is
    16 sequences by that width. Every other buffer -- `topk_indices_buffer` [32768, 2048], the
    radix top-k's `radix_aux_indices` / `radix_aux_logits` [16, 10, 2048], `scheduler_metadata_
    buffer` [num_sms + 1, 2], the slot mappings and query bounds [32768] -- is the same at both
    configs; none is max_batch_size x max_seq_len."""
    assert _get_generation_kv_capacity(None, is_draft=False) == (0, 1)
    assert _max_blocks_per_seq(_MAX_SEQ_LEN) == 32772 and _max_blocks_per_seq(32768) == 1028
    wide, narrow = _metadata_buffers(envelope), _metadata_buffers(delivered)

    tables = {
        "kv_cache_block_offsets": ((1, 16, 2), torch.int32),
        "block_table": ((16,), torch.int32),
        "indexer_k_cache_block_offsets": ((16,), torch.int32),
        "block_table_expanded": ((16,), torch.int32),
    }
    for name, (rows, dtype) in tables.items():
        assert wide[name] == (rows + (32772,), dtype)
        assert narrow[name] == (rows + (1028,), dtype)
    assert wide["topk_indices_buffer"] == ((32768, 2048), torch.int32)
    assert wide["radix_aux_indices"] == (
        (16, dsa_metadata._MAX_RADIX_BLOCKS_PER_ROW, 2048),
        torch.int32,
    )
    assert wide["radix_aux_logits"] == ((16, 10, 2048), torch.float32)
    assert wide["scheduler_metadata_buffer"] == ((149, 2), torch.int32)
    assert wide["slot_mapping_fp8"] == wide["slot_mapping_scale"] == ((32768,), torch.int64)
    assert wide["cu_seqlen_ks"] == wide["cu_seqlen_ke"] == ((32768,), torch.int32)
    assert wide["shared_topk_indices"] == ((32768, 2048), torch.int32)
    assert wide["ctx_cached_token_indptr"] == wide["ctx_kv_indptr"] == ((17,), torch.int64)

    # Under cached-KV MLA the base metadata also keeps the uncached prefix sums.
    assert set(wide) - set(narrow) == {"ctx_uncached_token_indptr"}
    assert set(narrow) <= set(wide)
    for name, (shape, dtype) in wide.items():
        assert _MAX_SEQ_LEN not in shape, name
        if name not in tables and name in narrow:
            assert (shape, dtype) == narrow[name], name


class _KvPool:
    """The KV manager surface warmup and capture read: ``tokens`` free tokens in pages of 32,
    dummies recorded as the token counts they were asked for."""

    tokens_per_block = _TOKENS_PER_BLOCK
    num_extra_kv_tokens = 0

    def __init__(self, tokens: int, max_seq_len: int):
        self.tokens = tokens
        self.max_seq_len = max_seq_len
        self.added = []

    def get_num_available_tokens(self, *, token_num_upper_bound, batch_size=1, **_):
        return min(token_num_upper_bound, self.tokens)

    def get_num_free_blocks(self) -> int:
        return self.tokens // self.tokens_per_block

    def add_dummy_requests(self, request_ids, token_nums=None, is_gen=False, **kwargs):
        if token_nums is None:
            token_nums = [1 + kwargs.get("max_num_draft_tokens", 0)] * len(request_ids)
        self.added.append((list(token_nums), is_gen))
        return [
            SimpleNamespace(
                py_request_id=request_id,
                token_num=token_num,
                is_gen=is_gen,
                is_last_context_chunk=True,
                is_attention_dp_dummy=False,
                py_draft_tokens=[],
                py_needs_onehot_draft_probs=False,
            )
            for request_id, token_num in zip(request_ids, token_nums)
        ]

    def free_resources(self, request) -> None:
        pass


class _Resources:
    def __init__(self, kv: _KvPool):
        self.kv = kv

    def get_resource_manager(self, key):
        return self.kv if key == ResourceManagerType.KV_CACHE_MANAGER else None


def _runner_config(resolved) -> SimpleNamespace:
    """The decoder runner's sizes for this engine. No speculative decoding: every draft width is
    zero, and beams are one wide."""
    graphs = resolved.llm_args.cuda_graph_config
    return SimpleNamespace(
        max_num_tokens=resolved.max_num_tokens,
        max_seq_len=resolved.max_seq_len,
        max_batch_size=resolved.max_batch_size,
        cuda_graph_batch_sizes=filter_cuda_graph_batch_sizes(
            graphs.batch_sizes,
            resolved.max_batch_size,
            resolved.max_num_tokens,
            1,
            graphs.enable_padding,
        ),
        max_beam_width=1,
        spec_config=None,
        use_mrope=False,
        max_draft_len=0,
        original_max_draft_len=0,
        max_total_draft_tokens=0,
        max_draft_loop_tokens=0,
    )


def _dummy_factory(resolved) -> DummyRequestFactory:
    return DummyRequestFactory(
        config=_runner_config(resolved),
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        model=SimpleNamespace(model_config=resolved.model_config),
        get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
        release_request_state=None,
    )


def _kv_creator(resolved) -> KvCacheCreator:
    creator = KvCacheCreator.__new__(KvCacheCreator)
    config = _runner_config(resolved)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=resolved.model_config),
        use_mrope=False,
        batch_size=resolved.max_batch_size,
        max_seq_len=resolved.max_seq_len,
        _max_cuda_graph_batch_size=config.cuda_graph_batch_sizes[-1],
    )
    creator._max_num_tokens = resolved.max_num_tokens
    creator._max_beam_width = 1
    creator._mapping = resolved.mapping
    creator._speculative_config = None
    creator._llm_args = resolved.llm_args
    creator._tokens_per_block = _TOKENS_PER_BLOCK
    creator._is_kv_cache_manager_v2 = True
    creator._kv_cache_config = resolved.llm_args.kv_cache_config
    creator._net_max_seq_len = resolved.max_seq_len
    creator._dummy_reqs = None
    return creator


def test_kv_estimation_profiles_one_full_step(envelope, delivered):
    """Before the pool exists `KvCacheCreator` (pyexecutor/_util.py) profiles dummy context
    requests as long as a request may be, net max_seq_len - 1, but no more than max_num_tokens in
    all: at the 1M config one 32768-token prompt (the delivered engine: a 32767- and a 1-token
    prompt). The DSA config keeps it off the chunked-MLA profile
    (`_get_mla_chunked_profile_length`), so the profiled step is one unchunked 32768-token
    context. `_get_token_num_for_estimation` sizes the temporary pool for the larger of those
    prompts and CUDA-graph capture's longest decode row: ceil((max_seq_len + 1) / 32) + 16 - 1
    pages, 32784 at the 1M config -- 1049088 tokens, 46.05 GiB at the fp8_ds_mla pool's 47136
    bytes a token (46.19 GiB at the 576-byte pool's). It holds a max_seq_len request, so the
    estimation KV manager does not lower max_seq_len
    (`KVCacheManagerV2.__init__` caps it by `get_num_available_tokens`, which a V2 estimate caps
    by these max_tokens, and `KvCacheCreator._create_kv_cache_manager` would carry a lowered
    value to every later manager)."""
    expected = {
        id(envelope): ([32768], 32784),
        id(delivered): ([32767, 1], 1040),
    }
    for resolved in (envelope, delivered):
        creator = _kv_creator(resolved)
        requests = creator._create_dummy_context_requests(max(1, resolved.max_seq_len - 1))
        assert creator._mla_chunked_profile_length is None
        lengths = [len(request.input_token_ids) for request in requests]
        prompts, pages = expected[id(resolved)]
        assert lengths == prompts and sum(lengths) == resolved.max_num_tokens
        creator._dummy_reqs = requests
        tokens = creator._get_token_num_for_estimation()
        assert tokens == pages * _TOKENS_PER_BLOCK
        assert tokens >= resolved.max_seq_len + 1
    assert 32784 * _TOKENS_PER_BLOCK == 1_049_088
    assert _KV_BYTES_PER_TOKEN == 47136
    assert 1_049_088 * _KV_BYTES_PER_TOKEN == 49_449_811_968
    assert math.isclose(1_049_088 * _KV_BYTES_PER_TOKEN / 2**30, 46.05, abs_tol=5e-3)
    assert 32784 >= math.ceil((_MAX_SEQ_LEN + 1) / _TOKENS_PER_BLOCK) == 32769


def test_kv_estimation_reserves_no_workspace(envelope):
    """What `KvCacheCreator.configure_kv_cache_capacity` (pyexecutor/_util.py) takes out of the KV
    budget beside the profiled peak, at the 1M config on sm_103: the indexer MQA-logits tile
    (`_INDEXER_MQA_LOGITS_ELEM_BUDGET`) is reserved only for the deepseek_v4 algorithm, not DSA;
    the TRTLLM backend declares no context-MLA workspace for fp8 DSA on sm_103
    (`TrtllmAttention.runtime_workspace_bytes_per_token`, attention/backends/trtllm.py: sparse
    MLA stages no buffer), so `get_mla_context_workspace_kv_len_cap` installs no attended-KV cap
    though chunked prefill is on; and no fixed capacity-sized reserve. The pool is the profiled
    budget whole."""
    model_config, mapping = envelope.model_config, envelope.mapping
    assert model_config.sparse_attention_config.algorithm == "dsa" != "deepseek_v4"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trtllm_backend, "get_sm_version", lambda: 103)
        w = get_attention_workspace_bytes_per_token(model_config, mapping)
        bounded = get_attention_workspace_is_chunked_prefill_bounded(model_config)
        fixed = get_attention_workspace_fixed_bytes(
            model_config,
            mapping,
            max_batch_size=_MAX_BATCH_SIZE,
            max_num_tokens=_MAX_NUM_TOKENS,
            max_seq_len=_MAX_SEQ_LEN,
            enable_cuda_graph=True,
        )
    assert (w, bounded, fixed) == (0, True, 0)
    cap = get_mla_context_workspace_kv_len_cap(
        envelope.llm_args.kv_cache_config,
        _MAX_BATCH_SIZE,
        _MAX_NUM_TOKENS,
        _MAX_SEQ_LEN,
        True,
        bounded,
        chunked_workspace_profiled=False,
        require_chunked_workspace_profile=w > 0,
    )
    assert cap is None


def test_warmup_runs_one_full_step(envelope, delivered):
    """`DecoderWarmup._get_full_general_warmup_requests` (pyexecutor/engine/runners/decoder/
    warmup.py) bounds its largest shape by min(max_num_tokens, max_batch_size x (max_seq_len -
    1)) = 32768; `DummyRequestFactory.create_warmup_request` (decoder/dummy_requests.py) builds
    it as prompts of at most max_seq_len - 1: one 32768-token prompt at the 1M config (a 32767-
    and a 1-token prompt delivered), never a 1M prompt -- warmup bypasses the scheduler and runs
    each shape as one step. No prefill CUDA graph is captured (`_filter_piecewise_capture_num_
    tokens`, model_engine.py: its 16 x 1048575 ceiling caps only capture candidates)."""
    for resolved, prompts in ((envelope, [32768]), (delivered, [32767, 1])):
        config = _runner_config(resolved)
        resources = _Resources(_KvPool(2 * _MAX_SEQ_LEN, resolved.max_seq_len))
        warmup = DecoderWarmup.__new__(DecoderWarmup)
        warmup._config = config
        warmup.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        shapes = warmup._get_full_general_warmup_requests(resources)
        assert shapes == [(1, 0), (1, 1), (32768, 0), (16, 16), (2, 0)]
        batch = _dummy_factory(resolved).create_warmup_request(resources, 32768, 0)
        assert [request.token_num for request in batch.context_requests] == prompts

        kept, unrecordable = _filter_piecewise_capture_num_tokens(
            config.cuda_graph_batch_sizes,
            resolved.max_num_tokens,
            _MAX_BATCH_SIZE,
            resolved.max_seq_len,
        )
        assert (kept, unrecordable) == ([1, 2, 4, 8, 16], [])
        assert resolved.llm_args.prefill_cuda_graph_backend == "disabled"
    assert _MAX_BATCH_SIZE * (_MAX_SEQ_LEN - 1) == 16_777_200


def test_cuda_graph_capture_decodes_at_max_seq_len(envelope):
    """`DecoderCudaGraphs.warmup_and_capture` (decoder/graphs.py) captures every graph batch (1,
    2, 4, 8, 16) twice for DSA, whose config asks for separate short and long graphs
    (`needs_separate_short_long_cuda_graphs`, its threshold index_topk 2048): once at max_seq_len
    and once at 2048 less the one token a decode step adds. `create_cuda_graph_warmup_request`
    fills each batch with one-token decode rows and one row of the family's longest history
    (`add_longest_dummy_request`): 1048575 tokens -- the decode kernels are captured at KV length
    1048576 -- and 2046. At batch 16 the long family needs 32768 + 15 pages, within the 32784
    the estimation pool holds."""
    sparse = copy.deepcopy(envelope.model_config.sparse_attention_config)
    assert sparse.needs_separate_short_long_cuda_graphs() is True
    families = [_MAX_SEQ_LEN, sparse.seq_len_threshold - 1]
    assert families == [1048576, 2047]

    config = _runner_config(envelope)
    assert config.cuda_graph_batch_sizes == [1, 2, 4, 8, 16]
    for family, longest in zip(families, (1048575, 2046)):
        for batch_size in config.cuda_graph_batch_sizes:
            pool = _KvPool(32784 * _TOKENS_PER_BLOCK, envelope.max_seq_len)
            batch = _dummy_factory(envelope).create_cuda_graph_warmup_request(
                _Resources(pool), batch_size, 0, family
            )
            rows = [request.token_num for request in batch.generation_requests]
            assert rows == [longest] + [1] * (batch_size - 1)
            assert all(request.is_gen for request in batch.generation_requests)
            pages = math.ceil((longest + 1) / _TOKENS_PER_BLOCK) + (batch_size - 1)
            assert pages <= 32784
    assert math.ceil(1048576 / _TOKENS_PER_BLOCK) + 15 == 32783


def test_pool_holds_the_envelope_load():
    """Arithmetic on the delivered pool, 2,373,024 tokens a rank of the 576-byte pool (47280 bytes
    a token), re-read at the fp8_ds_mla pool's 47136 (`MistralLarge4CacheManager.get_cache_size_
    per_token`; the V2 manager gives every request ceil((tokens + 1) / 32) pages,
    `KVCacheManagerV2.__init__`'s generation headroom): the same bytes hold 2,380,273 tokens.
    Charged to them whole: the width-sized allocations the measured 32768 engine did not have --
    the target's rope tables (indexer 64 x fp32 and identity 64 x fp32 at 1048576 rows, the MLA
    rope reading the indexer's: 512 MiB, against the 32 MiB that engine held at 32768 rows, the
    MLA rope ops' duplicated 2 x 64 x fp32 layout among them, which the target no longer
    builds), one gathered key block of 1048576 x 132 bytes, the indexer logits tile at the 1M
    config's 1 << 28 budget (1 GiB against 64 MiB at the 32768 config's 1 << 24,
    `_prefill_logits_budget`) and the decode scorer's [16, 1048576] fp32 arena --; the
    fp8_ds_mla attention's larger scratch in the profiled 32768-token step (the 64-head bf16
    query FlashMLA takes, `PrefillTarget._scratch`, against the 576-byte path's 16-head bf16 and
    e4m3 queries and zero rope tail); and the held context workspace (the 1M config's 2**19 rows
    of bf16 576, `_ctx_workspace_rows`, `PrefillTarget._ctx_workspace`; a step with more context
    KV attends its context rows straight off the pool), charged whole though a 32768 engine now
    holds 65536 rows of its own; and the vision tower every rank now loads whole (3,408,437,248
    bytes of bf16), which the delivered engine, the language model only, did not hold.
    Together under 160,000 tokens; what is left holds the 1M gate's load -- the
    1048559-token prompt generating 16 tokens beside 15 requests of at most a full step each --
    and two requests of max_seq_len, as does the pool a boot of the 1M accuracy gate measured on
    the native checkpoint (2,135,488 tokens a rank at its KV fraction 0.85)."""
    mib = 2**20
    rows, heads, lat = _MAX_NUM_TOKENS, 16, 576
    grown = {
        "rope tables": 1048576 * (64 + 64) * 4 - 32768 * (2 * 64 + 64 + 64) * 4,
        "gathered keys": (1048576 - 32768) * 132,
        "logits tile": ((1 << 28) - (1 << 24)) * 4,
        "decode arena": 16 * (1048576 - 32768) * 4,
        "64-head query": rows * 64 * lat * 2 - rows * heads * (lat * 2 + lat + 64 * 2),
        "context workspace": 2**19 * lat * 2,
        "vision tower": _VISION_TOWER_BYTES,
    }
    assert 1048576 * (64 + 64) * 4 == 512 * mib
    assert grown["64-head query"] == 1_442_840_576
    assert grown["context workspace"] == 576 * mib
    assert sum(grown.values()) == 7_164_305_408 < 6.7 * 2**30
    budget = _DELIVERED_POOL_TOKENS * _DELIVERED_BYTES_PER_TOKEN
    assert _DELIVERED_BYTES_PER_TOKEN == 47280 and budget == 112_196_574_720
    assert budget // _KV_BYTES_PER_TOKEN == 2_380_273
    lost = math.ceil(sum(grown.values()) / _KV_BYTES_PER_TOKEN)
    assert lost == 151_993 < 160_000
    pool = budget // _KV_BYTES_PER_TOKEN - lost

    def pages(tokens):
        return math.ceil((tokens + 1) / _TOKENS_PER_BLOCK)

    gate = pages(_ENVELOPE_PROMPT + 16) + 15 * pages(_MAX_NUM_TOKENS + 16)
    assert gate * _TOKENS_PER_BLOCK == 1_540_576 <= pool
    two = 2 * pages(_MAX_SEQ_LEN)
    assert two * _TOKENS_PER_BLOCK == 2_097_216 <= pool
    assert pool - two * _TOKENS_PER_BLOCK > 130_000
    # The measured 1M pool holds the same load.
    assert gate * _TOKENS_PER_BLOCK <= two * _TOKENS_PER_BLOCK <= _TEP4_1M_POOL_TOKENS


# ------------------------------------------------------- (6) what the router reads, the members


def test_router_reads_the_quantization_config_intact(resolved):
    """`route`, which `ModelLoader._load_and_validate_config` (pyexecutor/model_loader.py) asks
    through `_router_index` for a target, is handed the checkpoint's `quantization_config`
    exactly as the native loader reads it from params.json -- params.json's own, the whole
    `quantized_layers` map in its ModelOpt spelling -- and it stays so on the ModelConfig the
    target is built from, on the tep4 mapping. The copies the LLM API and ModelConfig normalise
    (and pop `quantized_layers` from) are their own."""
    assert len(resolved.routed) == 1
    seen = resolved.routed[0]
    on_disk = resolved.native.quantization_config
    params = json.loads((resolved.path / "params.json").read_text())
    assert on_disk == params["quantization_config"]
    assert seen.quantization_config == on_disk
    assert resolved.model_config.pretrained_config.quantization_config == on_disk
    assert seen.pretrained_config is resolved.model_config.pretrained_config
    assert resolved.model_config.modeling_v2_target is _Routed
    assert seen.compute_capability == _SM103 and seen.spec_config is None
    mapping = seen.mapping
    assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (4, 4, 1)
    assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.enable_attention_dp) == (4, 1, False)


def test_router_quantization_map_of_the_nvfp4_export():
    """The map the NVFP4 branch (`_nvfp4_mixed_precision`, models/mistral_large4/routing.py)
    compares as data: 475 Mistral module names, NVFP4 group 16 for the routed experts of layers
    1..71, FP8_PB_WO for wq, wo, the 44 full indexers' wq_b, the shared experts and layer 0's
    dense MLP, nothing else."""
    quant = json.loads((_checkpoint_dir(_NVFP4_NAMES) / "params.json").read_text())[
        "quantization_config"
    ]
    full = [i < 17 or i % 4 in (0, 3) for i in range(_LAYERS)]
    fp8 = {"quant_algo": "FP8_PB_WO"}
    expected = {
        f"layers.{i}.experts": {"quant_algo": "NVFP4", "group_size": 16} for i in range(1, _LAYERS)
    }
    for i in range(_LAYERS):
        expected[f"layers.{i}.attention.wq"] = fp8
        expected[f"layers.{i}.attention.wo"] = fp8
        if full[i]:
            expected[f"layers.{i}.attention.indexer.wq_b"] = fp8
        mlp = "feed_forward" if i == 0 else "shared_experts"
        for w in ("w1", "w2", "w3"):
            expected[f"layers.{i}.{mlp}.{w}"] = fp8
    assert {k: v for k, v in quant.items() if k != "quantized_layers"} == {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "ignore": [],
    }
    assert quant["quantized_layers"] == expected
    assert len(expected) == 475


def test_tep4_branch_claims_the_checkpoint_at_both_envelopes(envelope, delivered):
    """The routing (`route`, models/mistral_large4/routing.py), on what each engine hands it: the
    NVFP4 tep4 branch claims the NVFP4 export, at the 1M config as delivered -- no criterion
    reads max_seq_len or chunked prefill. The quantization map, read off the `pretrained_config`
    the router is handed, where it survives intact, is the one the branch compares."""
    for resolved in (envelope, delivered):
        seen = resolved.routed[0]
        target = routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
        assert target.__name__ == _NVFP4.target
        assert routing._nvfp4_mixed_precision(seen.pretrained_config)


def test_other_members_resolve_and_route_as_at_base():
    """The members the 1M config does not move, at their own base options: the NVFP4 checkpoint
    at dep4 (attention DP) and at tp4 (the experts cut along their width), each 32768 tokens with
    chunked prefill off. Through the same engine path (`_load_config_and_create_checkpoint_
    loader`, `ModelLoader._load_and_validate_config`) each is claimed by its own target (`route`,
    models/mistral_large4/routing.py), keyed on the checkpoint's anchor, on the same fp8_ds_mla
    pool, 47136 bytes a token (`MistralLarge4CacheManager.get_cache_size_per_token`), the DSA
    config unchanged."""
    nvfp4 = _checkpoint_dir(_NVFP4_NAMES)
    dep4 = _resolve(nvfp4, _options(32768, 32768, False, _DEP4))
    tp4 = _resolve(nvfp4, _options(32768, 32768, False, _TP4))
    cases = (
        (dep4, "ModelingV2MistralLarge4Nvfp4Sm103Dep4", (4, 1, True)),
        (tp4, "ModelingV2MistralLarge4Nvfp4Sm103Tp4", (1, 4, False)),
    )
    for resolved, target, (moe_ep, moe_tp, adp) in cases:
        seen = resolved.routed[0]
        mapping = seen.mapping
        assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (4, 4, 1)
        assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.enable_attention_dp) == (
            moe_ep,
            moe_tp,
            adp,
        )
        assert routing.route(seen.pretrained_config, mapping, _SM103, None).__name__ == target
        assert _anchor(resolved.llm_args) == 91.926
        assert (resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size) == (
            32768,
            32768,
            16,
        )
        assert resolved.llm_args.enable_chunked_prefill is False
        manager = get_kv_cache_manager_cls(resolved.model_config, resolved.llm_args.kv_cache_config)
        assert manager is MistralLarge4CacheManager
        assert (_kv_cost(resolved).slope, _kv_cost(resolved).intercept) == (47136, 0)
        sparse = resolved.model_config.sparse_attention_config
        assert (sparse.algorithm, sparse.q_split_threshold, sparse.index_topk) == (
            "dsa",
            8192,
            2048,
        )


# ------------------------------------------------------- DSA, rope, positions, params.json


def test_dsa_rope_and_positions_under_the_envelope(resolved):
    """The native config's normalised DSA keys, its YaRN parameters and its maximum position,
    adapted from params.json, are read without error under each gate's bounds; the engine keeps
    max_seq_len and max_num_tokens as configured, chunked prefill on at the 1M config and off at
    the delivered one, and max_seq_len is within the inferred limit, the
    checkpoint's maximum position (`DecoderModelForCausalLM.infer_max_seq_len`,
    _torch/models/modeling_utils.py: YaRN does not scale it). At the 1M config max_seq_len is
    that limit."""
    case, model_config, llm_args = resolved.case, resolved.model_config, resolved.llm_args
    cfg = model_config.pretrained_config
    assert (cfg.index_topk, cfg.index_n_heads, cfg.index_head_dim) == (2048, 32, 128)
    assert cfg.indexer_norm_eps == 1e-06 and cfg.indexer_ssmax_scaling is True
    assert (
        cfg.indexer_nope_layer_flags == cfg.nope_layer_flags == [i % 4 == 3 for i in range(_LAYERS)]
    )
    assert cfg.index_topk_pattern == "".join(
        "F" if i < 17 or i % 4 in (0, 3) else "S" for i in range(_LAYERS)
    )
    assert cfg.rope_parameters == {
        "rope_type": "yarn",
        "factor": case.rope_factor,
        "original_max_position_embeddings": 512,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
        "mscale_all_dim": 1,
        "rope_theta": 10000.0,
    }
    assert cfg.max_position_embeddings == case.max_position_embeddings

    sparse = model_config.sparse_attention_config
    assert type(sparse) is DeepSeekSparseAttentionConfig
    assert (sparse.index_n_heads, sparse.index_head_dim, sparse.index_topk) == (32, 128, 2048)
    assert sparse.indexer_rope_interleave is True and sparse.indexer_k_dtype == "fp8"

    assert llm_args.enable_chunked_prefill is case.chunked
    assert (resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size) == (
        case.max_num_tokens,
        case.max_seq_len,
        16,
    )
    assert (model_config.max_num_tokens, model_config.max_seq_len) == (
        case.max_num_tokens,
        case.max_seq_len,
    )
    inferred = DecoderModelForCausalLM.infer_max_seq_len(SimpleNamespace(config=cfg))
    assert inferred == case.max_position_embeddings >= resolved.max_seq_len
    if case is _NVFP4:
        assert resolved.max_seq_len == inferred == 1048576


def test_indexer_chunking_under_each_envelope(envelope, delivered):
    """`Indexer.prepare_for_chunked_prefill` (dsa/indexer.py) on the largest step of each engine.
    Delivered (32768 unchunked, no cached KV) it takes the indexer's own chunking:
    `indexer_max_chunk_size` resolves to its 32768 default, `select_indexer_chunk_size` keeps it
    below 256K keys, and a whole 32768-token prompt is one chunk, so no chunk list is built. At
    the 1M config the MLA-chunked-prefill branch plans the chunks instead (by request and packed
    logits, see the q-split tests); `indexer_max_chunk_size` is not read."""
    params = _metadata_params(delivered)
    assert params.indexer_max_chunk_size == 32768 >= delivered.max_num_tokens
    assert select_indexer_chunk_size(params.indexer_max_chunk_size, 32768) == 32768
    groups = split_prefill_chunks(torch.tensor([32768], dtype=torch.int32), 32768)
    assert groups == [[(0, 0, 32768, 0)]]
    step = _indexer_step(delivered, 0, ctx=[(0, 32768)])
    assert step.indexer_prefill_chunks is None and step.indexer_prefill_q_split is None
    assert step.slot_mapping_fp8_fullkv is step.slot_mapping_fp8

    step = _indexer_step(envelope, 0, ctx=[(0, 32768)])
    assert step.indexer_prefill_q_split == (0, 8192, [8192] * 4)
    assert [(c.token_start, c.token_end) for c in step.indexer_prefill_chunks] == [(0, 8192)]


def test_config_resolution_reads_params_json_only(resolved, tmp_path):
    """Nothing on this path (the front end's `_update_from_hf_quant_config`, then
    `_load_config_and_create_checkpoint_loader`, py_executor_creator.py, and
    `ModelLoader._load_and_validate_config`) reads any file of the checkpoint beside params.json:
    a directory holding params.json alone -- no tekken.json, no weights or their index --
    resolves to the same configuration, at each config. Its `dsa` block is parsed by the native
    loader alone (`_adapt_mistral_large4_config`)."""
    (tmp_path / "params.json").symlink_to(resolved.path / "params.json")
    alone = _resolve(tmp_path, resolved.case.options)

    def summary(r):
        cfg = r.model_config.pretrained_config.to_dict()
        cfg.pop("_name_or_path", None)
        return dict(
            llm_quant=repr(r.llm_args.quant_config),
            quant=repr(r.model_config.quant_config),
            layers={k: repr(v) for k, v in (r.model_config.quant_config_dict or {}).items()},
            pretrained=cfg,
            sparse=r.model_config.sparse_attention_config.model_dump(),
            routed=r.routed[0].quantization_config,
            sizes=(r.max_num_tokens, r.max_seq_len, r.max_batch_size),
            chunked=r.llm_args.enable_chunked_prefill,
        )

    assert summary(alone) == summary(resolved)


def test_params_json_dsa_block_is_the_engines_config(resolved):
    """params.json's `dsa` block is what the engine's config carries normalised: the native
    loader (`_adapt_mistral_large4_config`, models/checkpoints/mistral/config_loader.py) reads it
    into the keys the indexer, its sharing pattern and the latent cache are built from, and the
    DSA sparse config the forward reads takes the indexer's from them."""
    params = json.loads((resolved.path / "params.json").read_text())
    dsa, cfg = params["dsa"], resolved.model_config.pretrained_config
    assert dsa["topk"] == cfg.index_topk
    assert dsa["n_heads"] == cfg.index_n_heads
    assert dsa["head_dim"] == cfg.index_head_dim
    assert dsa["rope_head_dim"] == cfg.qk_rope_head_dim
    assert dsa["norm_eps"] == cfg.indexer_norm_eps
    assert dsa["ssmax_scaling"] == cfg.indexer_ssmax_scaling
    assert dsa["match_nope"] is True
    assert cfg.indexer_nope_layer_flags == cfg.nope_layer_flags
    assert dsa["sharing"].split(",") == cfg.index_sharing
    assert dsa["fp8_kv_cache"] is cfg.fp8_kv_cache is True
    sparse = resolved.model_config.sparse_attention_config
    assert (sparse.index_topk, sparse.index_n_heads, sparse.index_head_dim) == (
        dsa["topk"],
        dsa["n_heads"],
        dsa["head_dim"],
    )


# ------------------------------------------------------- tep8: eight ranks, two nodes of four

_TEP8_RANKS = 8
# Devices per GB300 node on this cluster: two nodes hold the eight ranks.
_GPUS_PER_NODE = 4
# The delivered config's max_seq_len (32768 tokens, one step per prompt).
_DELIVERED_MAX_SEQ_LEN = 32768
# The KV owners: the layers whose kv_source_layers entry is themselves, every layer but the
# nine NoPE sharers 39, 43, ..., 71 (layer 35's latent is theirs).
_KV_OWNERS = 63
# The per-rank KV pools measured at tep8 on the fp8_ds_mla cache, with the 64-head FlashMLA
# query in the profiled step and the native checkpoint's vision tower on every rank: at 32768
# at the accuracy gate's KV fraction (core.yaml's, 0.85), 106,325 pages; at the 1M config at
# the boot gate's (the quickstart's, 0.9), 111,923 pages, its load-time rope tables and the
# held context workspace already out of it.
_TEP8_ACCURACY_POOL_TOKENS = 106_325 * 32
_TEP8_BOOT_POOL_TOKENS = 111_923 * 32

# The topologies beside tep4 and dep4 (above). tep8 is the core unit's (spec-1m-tep8/options/
# core.yaml at 1M, spec-tep8/options/core.yaml delivered); tensor_parallel_size 8 alone is what
# a command stating no MoE split builds.
_TEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=False)
_TP8_ALONE = dict(tensor_parallel_size=8)

# spec-1m-tep8/options/core.yaml: the 1M config, the latent KV cache spelled fp8.
_CORE = _options(_MAX_SEQ_LEN, _MAX_NUM_TOKENS, True, _TEP8, kv_dtype="fp8")
# The CI class's (TestModelingV2MistralLarge4Nvfp4Sm103Tep8): the same config, its
# KvCacheConfig without a dtype.
_CORE_CI = _options(_MAX_SEQ_LEN, _MAX_NUM_TOKENS, True, _TEP8)
# spec-tep8/options/core.yaml: the delivered config, 32768 unchunked, the latent KV cache at
# its default.
_CORE_32K = _options(_DELIVERED_MAX_SEQ_LEN, _MAX_NUM_TOKENS, False, _TEP8)

# The boot gate's quickstart command after --model_dir: the checkpoint's Mistral-native format,
# the tep8 topology flags, the core unit's CUDA-graph capture list, chunked prefill on and block
# reuse off (task.yaml's runtime block), then task.yaml's extra flags.
_GATE_ARGV = (
    "--checkpoint_format=mistral_large_4",
    "--max_tokens=4",
    "--tp_size=8",
    "--moe_ep_size=8",
    "--use_cuda_graph",
    "--cuda_graph_batch_sizes",
    "1",
    "2",
    "4",
    "8",
    "16",
    "--enable_chunked_prefill",
    "--disable_kv_cache_reuse",
    "--max_seq_len=1048576",
    "--max_batch_size=16",
    "--max_num_tokens=32768",
    "--cuda_graph_padding_enabled",
    "--kv_cache_dtype=fp8",
)
# The delivered boot gate's: 32768 tokens, chunked prefill off. No --kv_cache_dtype: the latent
# cache is the checkpoint's default (quickstart's "auto"), the fp8_ds_mla pool.
_GATE_ARGV_32K = (
    "--checkpoint_format=mistral_large_4",
    "--max_tokens=4",
    "--tp_size=8",
    "--moe_ep_size=8",
    "--use_cuda_graph",
    "--cuda_graph_batch_sizes",
    "1",
    "2",
    "4",
    "8",
    "16",
    "--disable_kv_cache_reuse",
    "--max_seq_len=32768",
    "--max_batch_size=16",
    "--max_num_tokens=32768",
    "--cuda_graph_padding_enabled",
)
_QUICKSTART = Path(__file__).resolve().parents[7] / "examples/llm-api/quickstart_advanced.py"

# The NVFP4 target that serves tep4 and tep8 alike.
_NVFP4_TEP4_TARGET = "ModelingV2MistralLarge4Nvfp4Sm103Tep4"


def _llm_args(devices: int = _GPUS_PER_NODE, **kwargs) -> TorchLlmArgs:
    """``TorchLlmArgs`` as the LLM constructor builds it on rank 0 of a node with ``devices``
    visible devices (outside Ray)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("RAY_LOCAL_WORLD_SIZE", raising=False)
        mp.setattr(torch.cuda, "device_count", lambda: devices)
        return TorchLlmArgs(**kwargs)


def _gate_kwargs(checkpoint_dir: Path, argv: tuple[str, ...]) -> dict:
    """The ``LLM`` keyword arguments of a quickstart gate command: the tree's own
    quickstart_advanced.py parses ``argv`` and builds them in ``setup_llm``, and the gate harness
    adds ``modeling_v2``. Every KV-cache key of core.yaml has a quickstart flag, so the harness
    hands over no other."""
    if not _QUICKSTART.is_file():
        pytest.skip(f"{_QUICKSTART} is not in this checkout")
    spec = importlib.util.spec_from_file_location("quickstart_advanced", _QUICKSTART)
    quickstart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quickstart)
    built = {}

    def llm(**kwargs):
        built.update(kwargs)
        return SimpleNamespace()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["quickstart_advanced.py", f"--model_dir={checkpoint_dir}", *argv])
        args = quickstart.parse_arguments()
        mp.setattr(quickstart, "LLM", llm)
        quickstart.setup_llm(args)
    return dict(built, modeling_v2="require")


def _resolve_options(names: tuple[str, ...], options: dict) -> SimpleNamespace:
    """The configuration a served engine reaches for the checkpoint under ``names`` from the
    LLM-API ``options``, its arguments built as on rank 0 of a node of four."""
    path = _checkpoint_dir(names)
    return _resolve_args(path, _llm_args(model=str(path), modeling_v2="require", **options))


@pytest.fixture(scope="module")
def tep8():
    """The accuracy gate's engine at the 1M config: spec-1m-tep8/options/core.yaml."""
    return _resolve_options(_NVFP4_NAMES, _CORE)


@pytest.fixture(scope="module")
def tep8_gate():
    """The boot gate's engine at the 1M config: its quickstart command."""
    path = _checkpoint_dir(_NVFP4_NAMES)
    return _resolve_args(path, _llm_args(**_gate_kwargs(path, _GATE_ARGV)))


@pytest.fixture(scope="module")
def tep8_ci():
    """The CI class's engine at the 1M config: core.yaml's options with the KV dtype left at its
    default."""
    return _resolve_options(_NVFP4_NAMES, _CORE_CI)


@pytest.fixture(scope="module")
def tep8_32k():
    """The delivered accuracy gate's engine: spec-tep8/options/core.yaml, 32768 unchunked."""
    return _resolve_options(_NVFP4_NAMES, _CORE_32K)


@pytest.fixture(scope="module")
def tep8_32k_gate():
    """The delivered boot gate's engine: its quickstart command at 32768, unchunked."""
    path = _checkpoint_dir(_NVFP4_NAMES)
    return _resolve_args(path, _llm_args(**_gate_kwargs(path, _GATE_ARGV_32K)))


# Each tep8 engine: the fixture, its max_seq_len, whether chunked prefill is on, and the latent
# KV dtype it states (fp8 at 1M; the delivered gates leave it at its default).
_DEPLOYMENTS = {
    "1m-core-yaml": ("tep8", _MAX_SEQ_LEN, True, "fp8"),
    "1m-quickstart": ("tep8_gate", _MAX_SEQ_LEN, True, "fp8"),
    "32k-core-yaml": ("tep8_32k", _DELIVERED_MAX_SEQ_LEN, False, "auto"),
    "32k-quickstart": ("tep8_32k_gate", _DELIVERED_MAX_SEQ_LEN, False, "auto"),
}


@pytest.fixture(scope="module", params=list(_DEPLOYMENTS))
def deployment(request):
    """One tep8 engine -- the 1M config or the delivered one, from core.yaml or from the boot
    gate's command -- with the max_seq_len, the chunked prefill and the KV dtype it is configured
    with."""
    name, max_seq_len, chunked, kv_dtype = _DEPLOYMENTS[request.param]
    return request.getfixturevalue(name), max_seq_len, chunked, kv_dtype


@pytest.fixture(
    scope="module", params=["1m-options", "1m-quickstart", "32k-options", "32k-quickstart"]
)
def tp8_alone(request):
    """tensor_parallel_size 8 and no MoE split, as LLM-API options and as ``--tp_size=8``, at
    each config; with the tep8 engine of the same config."""
    path = _checkpoint_dir(_NVFP4_NAMES)
    size, spelling = request.param.split("-")
    wide = size == "1m"
    reference = request.getfixturevalue("tep8" if wide else "tep8_32k")
    if spelling == "options":
        if wide:
            options = _options(_MAX_SEQ_LEN, _MAX_NUM_TOKENS, True, _TP8_ALONE, kv_dtype="fp8")
        else:
            options = _options(_DELIVERED_MAX_SEQ_LEN, _MAX_NUM_TOKENS, False, _TP8_ALONE)
        return _resolve_options(_NVFP4_NAMES, options), reference
    argv = _GATE_ARGV if wide else _GATE_ARGV_32K
    argv = tuple(word for word in argv if word != "--moe_ep_size=8")
    return _resolve_args(path, _llm_args(**_gate_kwargs(path, argv))), reference


@pytest.fixture(scope="module")
def tep4():
    """The NVFP4 checkpoint at tep4 on the delivered config, built as the tep8 arguments are."""
    return _resolve_options(_NVFP4_NAMES, _options(_DELIVERED_MAX_SEQ_LEN, _MAX_NUM_TOKENS, False))


@pytest.fixture(scope="module")
def tep4_1m():
    """The NVFP4 checkpoint at tep4 on the 1M config's gate options (spec-1m-kvds-tep4/options/
    core.yaml, the latent KV cache spelled fp8), built as the tep8 arguments are: the
    configuration tep8's is compared against."""
    return _resolve_options(
        _NVFP4_NAMES, _options(_MAX_SEQ_LEN, _MAX_NUM_TOKENS, True, kv_dtype="fp8")
    )


@pytest.fixture(scope="module")
def dep4():
    return _resolve_options(
        _NVFP4_NAMES, _options(_DELIVERED_MAX_SEQ_LEN, _MAX_NUM_TOKENS, False, _DEP4)
    )


def _summary(resolved) -> dict:
    """Everything the engine resolved for a checkpoint except its mapping."""
    pretrained = resolved.model_config.pretrained_config.to_dict()
    pretrained.pop("_name_or_path", None)
    llm_args, model_config = resolved.llm_args, resolved.model_config
    return dict(
        llm_quant=repr(llm_args.quant_config),
        quant=repr(model_config.quant_config),
        layers={k: repr(v) for k, v in (model_config.quant_config_dict or {}).items()},
        pretrained=pretrained,
        sparse=model_config.sparse_attention_config.model_dump(),
        routed=resolved.routed[0].quantization_config,
        sizes=(resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size),
        chunked=llm_args.enable_chunked_prefill,
        cuda_graphs=llm_args.cuda_graph_config.model_dump(),
        kv_cache=llm_args.kv_cache_config.model_dump(),
        moe=llm_args.moe_config.model_dump(),
        moe_backend=model_config.moe_backend,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
    )


def _route_or_none(seen):
    try:
        return routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
    except NoTarget:
        return None


def _pages(tokens: int) -> int:
    """The pages the V2 manager holds for a request of ``tokens`` tokens: its one token of
    generation headroom included (`KVCacheManagerV2.__init__`)."""
    return math.ceil((tokens + 1) / _TOKENS_PER_BLOCK)


# ------------------------------------------------------- eight ranks in one process


class _Group:
    """The eight ranks in one process: rank r is a thread, and every exchange is a barrier, so
    each rank's call sees what every rank sent, as over MPI."""

    def __init__(self, size: int = _TEP8_RANKS):
        self.size = size
        self._barrier = threading.Barrier(size, timeout=60)
        self._slots = [None] * size

    def exchange(self, rank: int, obj):
        self._slots[rank] = obj
        self._barrier.wait()
        gathered = list(self._slots)
        self._barrier.wait()
        return gathered

    def run(self, step):
        """``step(rank)`` on every rank at once; the per-rank results in rank order."""
        results, errors = [None] * self.size, [None] * self.size

        def body(rank):
            try:
                results[rank] = step(rank)
            except BaseException as error:  # noqa: BLE001 - re-raised below, after every rank
                errors[rank] = error
                self._barrier.abort()

        threads = [threading.Thread(target=body, args=(rank,)) for rank in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # The rank that failed first, not a peer it left waiting at the barrier.
        raised = [error for error in errors if error is not None]
        for error in raised:
            if not isinstance(error, threading.BrokenBarrierError):
                raise error
        if raised:
            raise raised[0]
        return results


class _RankDist:
    """One rank's communicator: the world gather the warmup agrees its shapes over."""

    def __init__(self, group: _Group, rank: int):
        self.group = group
        self.rank = self.tp_rank = rank
        self.world_size = self.tp_size = group.size

    def allgather(self, obj):
        return self.group.exchange(self.rank, obj)


# ------------------------------------------------------- tep8 (1): topology, node layout, limits


def test_tep8_topology_and_limits(deployment):
    """Eight ranks, tensor parallel attention over all eight with no attention DP (8 of the 64
    heads per rank), the routed experts expert-parallel eight ways and not split along their
    width, one pipeline stage -- from core.yaml and from the boot gate's command alike, at the
    1M config and at the delivered one. Rank r is TP rank r and MoE EP rank r, MoE TP rank 0:
    its expert window is [16 r, 16 r + 16). One scheduler drives the eight ranks: 16 requests and
    32768 tokens per step in all (`TorchLlmArgs.get_runtime_sizes`, llmapi/llm_args.py), up to
    max_seq_len tokens a request: 1048576 in chunks, or 32768 unchunked as delivered."""
    resolved, max_seq_len, chunked, _ = deployment
    llm_args, mapping = resolved.llm_args, resolved.mapping
    parallel = llm_args.parallel_config
    assert (parallel.tp_size, parallel.pp_size, parallel.cp_size) == (8, 1, 1)
    # core.yaml and the command state the EP split only; the MoE TP size is left to Mapping.
    moe_split = (parallel.moe_ep_size, parallel.moe_tp_size, parallel.moe_cluster_size)
    assert moe_split == (8, -1, -1)
    assert llm_args.enable_attention_dp is False and parallel.enable_attention_dp is False

    assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (8, 8, 1)
    assert mapping.cp_size == 1
    assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.moe_cluster_size) == (8, 1, 1)
    assert mapping.enable_attention_dp is False and mapping.dp_size == 1
    assert mapping.attn_tp_size == 8 and mapping.enable_lm_head_tp_in_adp is False
    assert mapping.has_moe_ep() and not mapping.has_moe_tp()
    assert resolved.model_config.mapping is mapping

    cfg = resolved.model_config.pretrained_config
    heads = cfg.num_attention_heads // mapping.tp_size
    local_experts = cfg.n_routed_experts // mapping.moe_ep_size
    assert (cfg.num_attention_heads, cfg.n_routed_experts) == (64, 128)
    assert (heads, local_experts) == (8, 16)
    for rank in range(_TEP8_RANKS):
        rank_mapping = _rank_mapping(mapping, rank)
        assert (
            rank_mapping.tp_rank,
            rank_mapping.moe_ep_rank,
            rank_mapping.moe_tp_rank,
            rank_mapping.moe_cluster_rank,
        ) == (rank, rank, 0, 0)
        assert rank_mapping.ep_experts(cfg.n_routed_experts) == list(
            range(16 * rank, 16 * rank + 16)
        )
        assert local_experts * rank_mapping.moe_ep_rank == 16 * rank
        assert rank_mapping.tp_group == list(range(_TEP8_RANKS))

    assert (resolved.max_batch_size, resolved.max_num_tokens, resolved.max_seq_len) == (
        16,
        32768,
        max_seq_len,
    )
    assert llm_args.get_runtime_sizes() == (1, 32768, max_seq_len, 16)
    model_config = resolved.model_config
    assert (model_config.max_num_tokens, model_config.max_seq_len) == (32768, max_seq_len)
    assert llm_args.enable_chunked_prefill is chunked


def test_gpus_per_node_is_the_visible_device_count(deployment):
    """Neither gate states gpus_per_node. ``TorchLlmArgs.gpus_per_node`` defaults to None and its
    validator (outside Ray) replaces None with ``torch.cuda.device_count()`` of the process that
    builds the arguments, rank 0; each node's ranks see that node's four held devices, so it is 4.
    The mapping every rank builds from ``parallel_config`` carries it: rank r is local rank r % 4
    on node r // 4, and the deployment is multi-node. The LLM constructor's device check reads it
    too: world_size_per_node is 4, which four visible devices satisfy.

    The defaults of ``_ParallelConfig`` and ``Mapping`` are 8, not the device count: an
    eight-rank mapping built without it reads one node of eight -- local rank r, never multi-node
    -- and would demand eight devices on rank 0's node."""
    resolved = deployment[0]
    field = TorchLlmArgs.model_fields["gpus_per_node"]
    assert field.default is None and field.validate_default is True
    for options in (_CORE, _CORE_32K):
        assert "gpus_per_node" not in options
    for argv in (_GATE_ARGV, _GATE_ARGV_32K):
        assert "gpus_per_node" not in _gate_kwargs(resolved.path, argv)

    llm_args, mapping = resolved.llm_args, resolved.mapping
    assert llm_args.gpus_per_node == _GPUS_PER_NODE == 4
    assert llm_args.parallel_config.gpus_per_node == 4
    assert mapping.gpus_per_node == 4 and resolved.model_config.mapping.gpus_per_node == 4
    assert llm_args.parallel_config.world_size_per_node == 4
    for rank in range(_TEP8_RANKS):
        rank_mapping = _rank_mapping(mapping, rank)
        assert (rank_mapping.local_rank, rank_mapping.node_rank) == (rank % 4, rank // 4)
        assert rank_mapping.get_node_rank(rank) == rank // 4
        assert rank_mapping.is_multi_node() is True

    # The count is the visible devices' count, read when the arguments are built.
    path = str(resolved.path)
    for devices in (4, 8):
        built = _llm_args(devices, model=path, **_CORE)
        assert built.gpus_per_node == devices
        assert built.parallel_config.world_size_per_node == devices
    stated = _llm_args(1, model=path, gpus_per_node=4, **_CORE)
    assert stated.gpus_per_node == 4

    assert _ParallelConfig.model_fields["gpus_per_node"].default == 8
    assert _ParallelConfig(tp_size=8, moe_ep_size=8).world_size_per_node == 8
    unstated = Mapping(world_size=8, rank=5, tp_size=8, moe_ep_size=8)
    assert unstated.gpus_per_node == 8
    assert (unstated.local_rank, unstated.node_rank, unstated.is_multi_node()) == (5, 0, False)


def test_each_rank_selects_its_own_nodes_device(tep8):
    """The device each worker sets before building its executor
    (``BaseWorker._get_comm_ranks_device_id``): its global MPI rank modulo the devices its node
    shows, so rank r drives device r % 4 of its node -- the mapping's local rank. The ranks
    allgather their global ranks and devices, and each pins its CPU affinity to its device."""
    group = _Group()
    local = threading.local()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "device_count", lambda: _GPUS_PER_NODE)
        mp.setattr(torch.cuda, "set_device", lambda device: setattr(local, "device", device))
        mp.setattr(base_worker, "global_mpi_rank", lambda: local.rank)
        mp.setattr(
            base_worker,
            "mpi_comm",
            lambda: SimpleNamespace(allgather=lambda obj: group.exchange(local.rank, obj)),
        )
        mp.setattr(
            base_worker, "configure_cpu_affinity", lambda device: setattr(local, "cpus", device)
        )

        def step(rank):
            local.rank = rank
            ranks, devices = BaseWorker._get_comm_ranks_device_id(SimpleNamespace(global_rank=rank))
            return ranks, devices, local.device, local.cpus

        results = group.run(step)
    for rank, (ranks, devices, device, cpus) in enumerate(results):
        assert ranks == list(range(_TEP8_RANKS))
        assert devices == [0, 1, 2, 3, 0, 1, 2, 3]
        assert device == cpus == rank % 4 == _rank_mapping(tep8.mapping, rank).local_rank


def test_tp_size_8_alone_is_not_this_deployment(tp8_alone):
    """tensor_parallel_size 8 with no MoE split -- as LLM-API options or as ``--tp_size=8``,
    whose MoE sizes the quickstart passes as -1 -- gives the MoE the whole TP width: the routed
    experts tensor parallel eight ways (MoE TP rank r) and not expert-parallel, every rank's
    window all 128 experts. Everything else resolves as at tep8, at either config. Neither the
    tep4 branch, at either size, nor the dep4 branch of the routing claims it."""
    alone, reference = tp8_alone
    mapping = alone.mapping
    assert (mapping.world_size, mapping.tp_size, mapping.moe_tp_size, mapping.moe_ep_size) == (
        8,
        8,
        8,
        1,
    )
    assert mapping.enable_attention_dp is False and mapping.has_moe_tp()
    for rank in range(_TEP8_RANKS):
        rank_mapping = _rank_mapping(mapping, rank)
        assert (rank_mapping.moe_tp_rank, rank_mapping.moe_ep_rank) == (rank, 0)
        assert rank_mapping.ep_experts(128) == list(range(128))
    differs = {k for k, v in mapping.to_dict().items() if v != reference.mapping.to_dict()[k]}
    assert differs == {"moe_tp_size", "moe_ep_size"}
    assert _summary(alone)["sizes"] == _summary(reference)["sizes"]
    assert _summary(alone)["chunked"] == _summary(reference)["chunked"]
    assert len(alone.routed) == 1
    target = _route_or_none(alone.routed[0])
    assert target is None or target.__name__ not in (
        _NVFP4_TEP4_TARGET,
        "ModelingV2MistralLarge4Nvfp4Sm103Dep4",
    )


@pytest.mark.parametrize("tep8_envelope", ["1m", "32k"])
def test_boot_command_and_core_yaml_build_one_deployment(request, tep8_envelope):
    """The boot gate's quickstart command and core.yaml resolve to the same mapping on every rank
    and the same model, quantization, DSA, limits, chunked prefill, CUDA graphs, KV dtype (fp8 at
    the 1M config, the default auto as delivered), page size, block-reuse setting and KV
    manager; the router is handed equal mappings -- at the 1M config as at the delivered one.
    The KV fraction is not shared: the quickstart's --kv_cache_fraction default, 0.9, against
    core.yaml's 0.85, so the boot gate's pool is sized at 0.9 of the free memory and the
    accuracy gate's at 0.85."""
    wide = tep8_envelope == "1m"
    suffix = "" if wide else "_32k"
    core = request.getfixturevalue(f"tep8{suffix}")
    gate = request.getfixturevalue(f"tep8{suffix}_gate")
    assert gate.mapping.to_dict() == core.mapping.to_dict()
    assert gate.routed[0].mapping.to_dict() == core.routed[0].mapping.to_dict()
    gate_summary, core_summary = _summary(gate), _summary(core)
    shared = ("llm_quant", "quant", "layers", "pretrained", "sparse", "routed", "sizes", "chunked")
    for key in shared:
        assert gate_summary[key] == core_summary[key], key
    assert gate_summary["moe_max_num_tokens"] == core_summary["moe_max_num_tokens"] == 32768
    kv_dtype = "fp8" if wide else "auto"
    for resolved in (core, gate):
        graphs = resolved.llm_args.cuda_graph_config
        assert graphs.batch_sizes == [1, 2, 4, 8, 16] and graphs.enable_padding is True
        kv = resolved.llm_args.kv_cache_config
        assert (kv.dtype, kv.enable_block_reuse, kv.tokens_per_block) == (kv_dtype, False, 32)
        assert kv.use_kv_cache_manager_v2 is True
        assert resolved.llm_args.disable_overlap_scheduler is False
    assert core.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.85
    assert gate.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.9


def test_tep8_envelope_resolves_as_configured(tep8, tep8_gate, tep8_32k):
    """`TorchLlmArgs` (llmapi/llm_args.py) keeps the 1M config as given at tep8: max_seq_len
    1048576, max_num_tokens 32768 (the chunk), max_batch_size 16; chunked prefill accepted with
    the DSA sparse attention the checkpoint configures, the latent KV cache spelled fp8 and block
    reuse off, on MistralLarge4CacheManager, the V2 KV manager the sparse registry picks for a
    MistralLarge4Config (`get_kv_cache_manager_cls`, pyexecutor/_util.py). The DSA config is the
    checkpoint's, with its `q_split_threshold` default. `validate_feature_combination`
    (pyexecutor/_util.py) rejects nothing, and `_create_py_executor` (py_executor_creator.py)
    keeps chunked prefill for MLA on sm_103 (`_MLA_CHUNKED_PREFILL_SUPPORTED_SM_VERSIONS`).
    Beside it the delivered config: 32768 tokens, one step per prompt, the KV dtype at its
    default."""
    for resolved, max_seq_len, chunked, kv_dtype in (
        (tep8, _MAX_SEQ_LEN, True, "fp8"),
        (tep8_gate, _MAX_SEQ_LEN, True, "fp8"),
        (tep8_32k, _DELIVERED_MAX_SEQ_LEN, False, "auto"),
    ):
        llm_args = resolved.llm_args
        assert (llm_args.max_seq_len, llm_args.max_num_tokens) == (max_seq_len, _MAX_NUM_TOKENS)
        assert llm_args.enable_chunked_prefill is chunked
        assert llm_args.disable_overlap_scheduler is False

        kv_cache_config = llm_args.kv_cache_config
        assert kv_cache_config.dtype == kv_dtype and kv_cache_config.enable_block_reuse is False
        assert kv_cache_config.tokens_per_block == _TOKENS_PER_BLOCK
        assert kv_cache_config.use_kv_cache_manager_v2 is True
        assert kv_cache_config.max_attention_window is None
        manager = get_kv_cache_manager_cls(resolved.model_config, kv_cache_config)
        assert manager is MistralLarge4CacheManager and issubclass(manager, KVCacheManagerV2)

        assert llm_args.prefill_capture_num_tokens is None
        assert llm_args.torch_compile_config is None

        sparse = resolved.model_config.sparse_attention_config
        assert type(sparse) is DeepSeekSparseAttentionConfig and sparse.algorithm == "dsa"
        assert sparse.q_split_threshold == 8192
        assert sparse.skip_indexer_for_short_seqs is True
        assert sparse.use_cute_dsl_topk is False and sparse.enable_heuristic_topk is False
        assert resolved.model_config.attn_backend == "TRTLLM"

        validate_feature_combination(
            llm_args, SimpleNamespace(model=SimpleNamespace(model_config=resolved.model_config))
        )
    assert 103 in py_executor_creator._MLA_CHUNKED_PREFILL_SUPPORTED_SM_VERSIONS


def test_tep8_model_engine_keeps_max_seq_len(tep8, tep8_32k):
    """`PyTorchModelEngine._init_model_capacity` (pyexecutor/model_engine.py) lowers a
    max_seq_len above the model's inferred limit and a max_num_tokens above max_batch_size x
    max_seq_len. The target's shell infers the limit with
    `DecoderModelForCausalLM.infer_max_seq_len` (_torch/models/modeling_utils.py): YaRN
    (512 x 2048) does not scale it, so it is max_position_embeddings, 1048576, and the 1M
    config's max_seq_len is kept whole at tep8, as the delivered one is."""
    for resolved in (tep8, tep8_32k):
        cfg = resolved.model_config.pretrained_config
        assert cfg.rope_parameters["rope_type"] == "yarn"
        assert cfg.rope_parameters["factor"] == 2048.0
        assert cfg.rope_parameters["original_max_position_embeddings"] == 512
        inferred = DecoderModelForCausalLM.infer_max_seq_len(SimpleNamespace(config=cfg))
        assert inferred == cfg.max_position_embeddings == _MAX_SEQ_LEN

        engine = SimpleNamespace(
            model=SimpleNamespace(infer_max_seq_len=lambda inferred=inferred: inferred),
            max_seq_len=resolved.max_seq_len,
            max_num_tokens=resolved.max_num_tokens,
            batch_size=resolved.max_batch_size,
        )
        for name in ("_init_max_seq_len", "_init_max_num_tokens", "_infer_max_seq_len_from_config"):
            setattr(engine, name, MethodType(getattr(PyTorchModelEngine, name), engine))
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", raising=False)
            PyTorchModelEngine._init_model_capacity(engine)
        assert (engine.max_seq_len, engine.max_num_tokens) == (
            resolved.max_seq_len,
            resolved.max_num_tokens,
        )
    assert tep8.model_config.max_seq_len == _MAX_SEQ_LEN
    assert tep8.model_config.max_num_tokens == _MAX_NUM_TOKENS


def test_tep8_llm_api_admits_the_envelope_prompt(tep8, tep8_gate, tep8_32k):
    """`BaseLLM._check_arguments` (llmapi/llm.py) refuses a prompt longer than max_num_tokens only
    without chunked prefill: the delivered tep8 engine refuses 32769 tokens, the 1M engine admits
    the 1M gate's 1048559. The bound on max_seq_len is the worker's:
    `BaseWorker._enqueue_request` (executor/base_worker.py) gives a request at most max_seq_len
    less its prompt to generate, and refuses a prompt that leaves no room for one token."""
    for resolved in (tep8, tep8_gate):
        _check_arguments(resolved, _ENVELOPE_PROMPT)
        assert _enqueue(resolved, _ENVELOPE_PROMPT, 16).max_tokens == 16
        assert _enqueue(resolved, _MAX_SEQ_LEN - 1, 16).max_tokens == 1
        for prompt_len in (_MAX_SEQ_LEN, _MAX_SEQ_LEN + 1):
            with pytest.raises(RequestError, match="maximum context length is 1048576 tokens"):
                _enqueue(resolved, prompt_len, 16)
    _check_arguments(tep8_32k, _MAX_NUM_TOKENS)
    with pytest.raises(RequestError, match=r"should not exceed max_num_tokens \(32768\)"):
        _check_arguments(tep8_32k, _MAX_NUM_TOKENS + 1)


# ------------------------------------------------------- tep8 (2): quantization, the anchor's key


def test_tep8_quantization_is_the_anchor_key(deployment):
    """At tep8 as at tep4, at the 1M config as delivered: MIXED_PRECISION with the FP8 KV
    cache, the GSM8K anchor row's key, and the router handed params.json's quantization_config
    with its 475 quantized_layers intact, on the tep8 mapping -- from core.yaml and from the boot
    gate's command alike, whether they spell the latent fp8 (1M) or leave it at auto
    (delivered)."""
    resolved, _, _, kv_dtype = deployment
    on_disk = json.loads((resolved.path / "params.json").read_text())["quantization_config"]
    assert len(on_disk["quantized_layers"]) == 475
    assert resolved.prequantized is True
    for quant in (resolved.llm_args.quant_config, resolved.model_config.quant_config):
        assert quant.quant_algo == QuantAlgo.MIXED_PRECISION
        assert quant.kv_cache_quant_algo == QuantAlgo.FP8
        assert quant.group_size == 16
    assert resolved.llm_args.kv_cache_config.dtype == kv_dtype
    assert _anchor(resolved.llm_args) == 91.926

    assert len(resolved.routed) == 1
    seen = resolved.routed[0]
    assert seen.quantization_config == on_disk
    assert resolved.model_config.pretrained_config.quantization_config == on_disk
    assert seen.pretrained_config is resolved.model_config.pretrained_config
    assert resolved.model_config.modeling_v2_target is _Routed
    assert seen.compute_capability == _SM103 and seen.spec_config is None
    assert seen.mapping is resolved.model_config.mapping
    assert (seen.mapping.world_size, seen.mapping.tp_size, seen.mapping.pp_size) == (8, 8, 1)
    assert (seen.mapping.moe_ep_size, seen.mapping.moe_tp_size) == (8, 1)
    assert seen.mapping.enable_attention_dp is False


def test_tep8_kv_dtype_fp8_and_auto_build_the_fp8_ds_mla_pool(tep8, tep8_gate, tep8_ci):
    """The gates spell the latent KV cache fp8 (core.yaml's kv_cache_config.dtype, the boot gate's
    --kv_cache_dtype=fp8); the CI class leaves it at its default, auto. Both reach the same pool:
    `validate_and_set_kv_cache_quant` (pyexecutor/model_loader.py) sets the KV quant algo FP8 under
    fp8 and keeps the checkpoint's, FP8, under auto, so `_create_kv_cache_manager` hands the
    manager the FP8 element type; the sparse registry picks MistralLarge4CacheManager for the
    MistralLarge4Config whatever the dtype (`get_kv_cache_manager_cls`), and it accepts either
    spelling over FP8 and builds the fp8_ds_mla layout -- its base over all 72 layers, one
    656-byte head on each of the 63 KV owners and none on the nine sharers, as FP8 -- 47136 bytes
    a token on every rank, the same anchor row. Any other element type is refused at
    construction."""
    assert tep8_ci.llm_args.kv_cache_config.dtype == "auto"
    assert tep8.llm_args.kv_cache_config.dtype == tep8_gate.llm_args.kv_cache_config.dtype == "fp8"
    cfg = tep8.model_config.pretrained_config
    owners = [s == i for i, s in enumerate(cfg.kv_source_layers)]
    assert sum(owners) == _KV_OWNERS
    for resolved in (tep8, tep8_gate, tep8_ci):
        model_config, kv_cache_config = resolved.model_config, resolved.llm_args.kv_cache_config
        assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert model_config.quant_config.quant_mode.has_fp8_kv_cache()
        manager = get_kv_cache_manager_cls(model_config, kv_cache_config)
        assert manager is MistralLarge4CacheManager and manager.latent_layout == "fp8_ds_mla"
        base = _manager_arguments(resolved, tensorrt_llm.bindings.DataType.FP8)
        assert base["kv_cache_config"].dtype == kv_cache_config.dtype
        assert base["dtype"] == tensorrt_llm.bindings.DataType.FP8
        assert (base["num_layers"], base["head_dim"]) == (_LAYERS, 656)
        assert base["num_kv_heads"] == [1 if owner else None for owner in owners]
        assert base["layer_mask"] is None
        for rank in range(_TEP8_RANKS):
            assert _kv_cost(resolved, rank) == _kv_cost(tep8, 0)
        assert (_kv_cost(resolved, 0).slope, _kv_cost(resolved, 0).intercept) == (
            _KV_BYTES_PER_TOKEN,
            0,
        )
        assert _anchor(resolved.llm_args) == 91.926
        with pytest.raises(ValueError, match="fp8_ds_mla"):
            _manager_arguments(resolved, tensorrt_llm.bindings.DataType.BF16)
    assert _KV_BYTES_PER_TOKEN == 47136


# ------------------------------------------------------- tep8 (3): a long prompt's context path


def test_tep8_chunked_prefill_hands_the_target_the_cached_kv_context_path(tep8, tep8_32k):
    """`build_attention_metadata` (pyexecutor/engine/metadata.py) turns on
    `enable_context_mla_with_cached_kv` for MLA under cache reuse or chunked prefill: on at the
    1M config, off at the delivered engine. With it and `runtime_features.chunked_prefill`,
    `Indexer.prepare_for_chunked_prefill` (attention/backends/sparse/dsa/indexer.py) takes its
    MLA-chunked-prefill branch. The metadata is sized by the engine bounds, not max_seq_len."""
    for resolved, cached_kv in ((tep8, True), (tep8_32k, False)):
        made = build_attention_metadata(
            resolved.model_config,
            max_batch_size=resolved.max_batch_size,
            max_num_tokens=resolved.max_num_tokens,
            max_beam_width=1,
            attention_backend=SimpleNamespace(Metadata=lambda **kwargs: SimpleNamespace(**kwargs)),
            attention_runtime_features=_runtime_features(resolved),
            mapping=resolved.mapping,
            cache_indirection=None,
        )
        assert made.enable_context_mla_with_cached_kv is cached_kv
        assert made.runtime_features.chunked_prefill is cached_kv
        assert made.runtime_features.cache_reuse is False
        assert made.runtime_features.chunk_size == _MAX_NUM_TOKENS
        assert (made.max_num_requests, made.max_num_tokens, made.max_num_sequences) == (
            16,
            32768,
            16,
        )
        assert made.sparse_metadata_params == _metadata_params(resolved)


def test_tep8_target_sizes_from_the_engine(monkeypatch, tep8, tep8_gate, tep8_32k):
    """What the target takes from the engine on its first step at tep8, as at tep4: one
    scheduler holds the 16 requests and the 32768 tokens of a step for all eight ranks, so every
    rank reads the same metadata (`_target_sizing`), its runtime features the executor's: chunked
    prefill as enable_chunked_prefill, block reuse as kv_cache_config.enable_block_reuse (off in
    core.yaml, and by the boot gate's --disable_kv_cache_reuse). At the 1M config the logits
    budget at its 2**28 ceiling and the held context workspace at its 2**19; delivered 2**24 and
    65536."""
    monkeypatch.delenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", raising=False)
    for resolved, chunked, rows, budget in (
        (tep8, True, 2**19, 2**28),
        (tep8_gate, True, 2**19, 2**28),
        (tep8_32k, False, 2**16, 2**24),
    ):
        md = _target_sizing(resolved)
        assert (md.max_num_requests, md.max_num_tokens) == (16, 32768)
        assert md.max_seq_len == resolved.max_seq_len
        features, llm_args = md.runtime_features, resolved.llm_args
        assert features.chunked_prefill is llm_args.enable_chunked_prefill is chunked
        assert features.cache_reuse is llm_args.kv_cache_config.enable_block_reuse is False
        assert tep4_modeling._ctx_workspace_rows(md) == rows
        assert tep4_modeling._prefill_logits_budget(md.max_num_tokens, md.max_seq_len) == budget


def _chunk_ranges(md) -> list[tuple[int, int, int, int]]:
    """Each planned indexer chunk as (token_start, token_end, k_token_start, k_token_end)."""
    return [
        (chunk.token_start, chunk.token_end, chunk.k_token_start, chunk.k_token_end)
        for chunk in md.indexer_prefill_chunks
    ]


@pytest.mark.parametrize("chunk", [0, 1, 2, 31], ids=["first", "second", "third", "last"])
def test_tep8_later_chunks_carry_their_cached_prefix(tep8, chunk):
    """Every chunk of one request after its first is a context request with a cached prefix.
    `DSAtrtllmAttentionMetadata.prepare_for_mla_rope_append` (attention/backends/sparse/dsa/
    metadata.py) hands the target its cached and total prefix sums, `ctx_cached_token_indptr`
    [0, cached] and `ctx_kv_indptr` [0, cached + new], `num_ctx_cached_tokens` = cached and
    `max_ctx_seq_len` = new; `Indexer.prepare` counts `num_ctx_kv_tokens` = new keys written this
    step, and `Indexer.recompute_context_kv_gather_mappings` builds `slot_mapping_fp8_fullkv` /
    `slot_mapping_scale_fullkv` over the cached and the new keys, position p at page p // 32 of
    the request's indexer page table. On a first chunk there is nothing cached and the full
    mappings are the step's own. Every TP rank gets the same: the query split (below) cuts only
    the indexer's queries, not the keys they are scored against. The target's context workspace
    (`PrefillTarget._ctx_workspace`, from `_ctx_kv_offsets`, held at the 2**19 rows it takes from
    this engine) holds the request's whole KV on each chunk whose KV fits those rows, its cached
    prefix and the chunk, each row at its own position: `num_ctx_kv_tokens` (the new keys) would
    leave the prefix out. The last chunk's KV is past them, so its rows attend off the pool in
    sub-batches of 8192 instead."""
    cached, new = sum(_CHUNKS[:chunk]), _CHUNKS[chunk]
    total = cached + new
    rows = tep4_modeling._ctx_workspace_rows(_target_sizing(tep8))
    assert rows == 2**19
    for rank in (0, _TEP8_RANKS - 1):
        md = _indexer_step(tep8, rank, ctx=[(cached, new)])
        assert md.host_ctx_cached_token_indptr.tolist() == [0, cached]
        assert md.ctx_cached_token_indptr.tolist() == [0, cached]
        assert md.host_ctx_kv_indptr.tolist() == [0, total]
        assert md.ctx_kv_indptr.tolist() == [0, total]
        assert (md.num_ctx_cached_tokens, md.max_ctx_seq_len, md.max_ctx_kv_len) == (
            cached,
            new,
            total,
        )
        assert md.num_ctx_mla_kv_tokens == total
        assert md.num_ctx_kv_tokens == new
        assert md.skip_indexer_for_ctx_reqs is False

        offsets = tep4_modeling._ctx_kv_offsets(md, md.num_contexts)
        assert offsets["ctx_cached_tokens"] == cached
        assert offsets["cu_ctx_kv"].tolist() == [0, total]
        prefill = object.__new__(tep4_modeling.PrefillTarget)
        prefill.core = SimpleNamespace(lat_dim=1, ctx_workspace_rows=rows)
        step = dict(num_contexts=1, num_ctx_tokens=md.num_ctx_tokens, **offsets)
        ws = prefill._ctx_workspace(step, new, "cpu")
        if total > rows:
            assert chunk == 31 and ws.ws_buf is None
            assert ws.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, new)]
        else:
            assert ws.ws_buf.shape[0] == ws.kv_req.shape[0] == ws.kv_pos.shape[0] == total
            assert int(ws.kv_pos[0]) == 0 and int(ws.kv_pos[-1]) == total - 1
            assert ws.ws_off.tolist() == [0] and int(ws.kv_req.max()) == 0

        table = md.host_indexer_k_cache_block_offsets
        fp8, scale = _key_slots(table, 0, torch.arange(cached, total))
        assert torch.equal(md.slot_mapping_fp8, fp8) and torch.equal(md.slot_mapping_scale, scale)
        if chunk == 0:
            assert md.slot_mapping_fp8_fullkv is md.slot_mapping_fp8
            assert md.slot_mapping_scale_fullkv is md.slot_mapping_scale
        else:
            fp8, scale = _key_slots(table, 0, torch.arange(total))
            assert md.slot_mapping_fp8_fullkv.shape == (total,)
            assert torch.equal(md.slot_mapping_fp8_fullkv, fp8)
            assert torch.equal(md.slot_mapping_scale_fullkv, scale)
            assert torch.equal(md.slot_mapping_fp8_fullkv[cached:], md.slot_mapping_fp8)
    # The last chunk ends one token short of max_seq_len less the tokens it generates.
    assert total <= _ENVELOPE_PROMPT and (chunk != 31 or total == _ENVELOPE_PROMPT)


# ------------------------------------------------------- tep8 (4): the query split over eight ranks


def test_tep8_q_split_threshold_is_8192(tep8, tep8_gate):
    """`DeepSeekSparseAttentionConfig.q_split_threshold` (llmapi/llm_args.py) defaults to 8192 and
    reaches `DSAMetadataParams`, which `_prefill_q_split_threshold` (dsa/indexer.py) reads; the
    split is on, not disabled by a negative value, from core.yaml and the boot gate's command
    alike."""
    for resolved in (tep8, tep8_gate):
        params = _metadata_params(resolved)
        assert resolved.model_config.sparse_attention_config.q_split_threshold == 8192
        assert params.q_split_threshold == 8192
        assert (
            dsa_indexer._prefill_q_split_threshold(SimpleNamespace(sparse_metadata_params=params))
            == 8192
        )


# context tokens in a step -> per-rank sizes of the split at tp 8 (None: no split)
_TEP8_SPLITS = {
    8191: None,
    8192: [1024] * 8,
    8193: [1024] * 7 + [1025],
    32767: [4095] + [4096] * 7,
    # a whole chunk, the warmup's and the KV estimation's step
    32768: [4096] * 8,
    # the 1M gate prompt's last chunk
    32751: [4093] + [4094] * 7,
    # a chunk beside 15 decode rows
    32736: [4092] * 8,
    # several short prompts, 3000 + 3000 + 3000 + 500 tokens
    9500: [1187, 1188] * 4,
}


@pytest.mark.parametrize("count", list(_TEP8_SPLITS))
def test_tep8_split_prefill_queries_per_rank(tep8, count):
    """`_split_prefill_queries` (dsa/indexer.py) on each of the eight TP ranks' mappings: below
    8192 context tokens every rank takes them all; from 8192 on, rank r takes
    [count r // 8, count (r + 1) // 8) and every rank knows every share's size (4096 rows a rank
    of a 32768-token step). Attention DP (dep4's mapping) never splits."""
    sizes = _TEP8_SPLITS[count]
    assert sizes is None or sum(sizes) == count
    begins = [0] if sizes is None else [sum(sizes[:r]) for r in range(_TEP8_RANKS + 1)]
    for rank in range(_TEP8_RANKS):
        mapping = _rank_mapping(tep8.mapping, rank)
        assert mapping.tp_rank == rank
        split = dsa_indexer._split_prefill_queries(count, mapping, 8192)
        if sizes is None:
            assert split == (0, count, None)
        else:
            assert split == (begins[rank], begins[rank + 1], sizes)
    adp = copy.deepcopy(tep8.mapping)
    adp.enable_attention_dp = True
    assert dsa_indexer._split_prefill_queries(count, adp, 8192) == (0, count, None)


@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        pytest.param([32768], [3 * 32768], id="one-chunk-over-a-cached-prefix"),
        pytest.param([32751], [_ENVELOPE_PROMPT], id="last-chunk"),
        pytest.param([3000, 3000, 3000, 500], [3000, 3000, 3000, 500], id="short-prompts"),
        pytest.param([2048] * 16, [2048] * 16, id="sixteen-prompts"),
        pytest.param([600] * 15 + [23768], [600] * 15 + [23768 + 5 * 32768], id="mixed-lengths"),
        pytest.param([5, 5, 8182], [983045, 983045, 8182], id="packed-tails-beside-a-prompt"),
    ],
)
def test_tep8_chunk_plan_covers_each_rank_share(tep8, tep8_32k, q_lens, kv_lens):
    """`_packed_logits_elem_budget` (dsa/indexer.py) is 1 << 28 at both configs (the packed cap;
    `_INDEXER_MQA_LOGITS_ELEM_BUDGET` x max_num_tokens x max_seq_len do not bind). Under it
    `plan_mla_chunked_prefill_chunks` covers exactly each of the eight ranks' shares
    [q_begin, q_end) of the context tokens, in order, once each; a request the share boundary
    cuts is a chunk of its own, and whole requests share a chunk only within the budget."""
    for resolved in (tep8, tep8_32k):
        budget = dsa_indexer._packed_logits_elem_budget(
            SimpleNamespace(
                max_num_tokens=resolved.max_num_tokens,
                kv_cache_manager=SimpleNamespace(max_seq_len=resolved.max_seq_len),
            )
        )
        assert budget == dsa_indexer._INDEXER_PACKED_LOGITS_ELEM_BUDGET == 1 << 28
    assert dsa_indexer._INDEXER_MQA_LOGITS_ELEM_BUDGET == 1 << 31

    total = sum(q_lens)
    covered = []
    for rank in range(_TEP8_RANKS):
        q_begin, q_end, sizes = dsa_indexer._split_prefill_queries(
            total, _rank_mapping(tep8.mapping, rank), 8192
        )
        assert sizes is not None and len(sizes) == _TEP8_RANKS
        chunks = plan_mla_chunked_prefill_chunks(q_lens, kv_lens, 1 << 28, q_begin, q_end)
        spans = _covered(chunks, q_lens)
        assert spans[0][0] == q_begin and spans[-1][1] == q_end
        assert all(prev[1] == cur[0] for prev, cur in zip(spans, spans[1:]))
        for chunk in chunks:
            whole = all(start == 0 and end == q_lens[i] for i, start, end, _ in chunk)
            if len(chunk) > 1:
                assert whole
                assert sum(q_lens[i] for i, *_ in chunk) * sum(kv_lens[i] for i, *_ in chunk) <= (
                    1 << 28
                )
            elif not whole:
                req_idx, start, end, req_start = chunk[0]
                assert req_start + start == q_begin or req_start + end == q_end
        covered += spans
    assert covered[0][0] == 0 and covered[-1][1] == total
    assert all(prev[1] == cur[0] for prev, cur in zip(covered, covered[1:]))


@pytest.mark.parametrize("chunk", [1, 31], ids=["second", "last"])
def test_tep8_each_rank_plans_its_share_of_a_chunk(tep8, chunk):
    """Through `Indexer.prepare` on each of the eight TP ranks: a chunk of the long prompt (32768
    or 32751 tokens over its cached prefix) sets `indexer_prefill_q_split` = (q_begin, q_end,
    sizes) and one `indexer_prefill_chunks` entry for the rank's share, queries [q_begin, q_end)
    of the chunk against keys [0, cached + q_end) of the request, each query's window ending at
    its own position. The eight shares tile the chunk."""
    cached, new = sum(_CHUNKS[:chunk]), _CHUNKS[chunk]
    sizes = _TEP8_SPLITS[new]
    assert sum(sizes) == new
    for rank in range(_TEP8_RANKS):
        md = _indexer_step(tep8, rank, ctx=[(cached, new)])
        q_begin, q_end = sum(sizes[:rank]), sum(sizes[: rank + 1])
        assert md.indexer_prefill_q_split == (q_begin, q_end, sizes)
        assert _chunk_ranges(md) == [(q_begin, q_end, 0, cached + q_end)]
        (plan,) = md.indexer_prefill_chunks
        assert torch.equal(plan.cu_seqlen_ks, torch.zeros(q_end - q_begin, dtype=torch.int32))
        assert torch.equal(
            plan.cu_seqlen_ke,
            torch.arange(cached + q_begin + 1, cached + q_end + 1, dtype=torch.int32),
        )
        assert md.decode_schedules == []


def test_tep8_q_split_on_skipped_and_mixed_steps(tep8):
    """What `Indexer.prepare_for_chunked_prefill` does on the two steps the target must not
    misread. (a) A step whose every context is within index_topk 2048: `prepare_for_skip_indexer`
    sets `skip_indexer_for_ctx_reqs` and fills the dense top-k rows, and the engine STILL sets
    `indexer_prefill_q_split` once the context tokens reach 8192 (and plans each rank's share);
    only the full-KV gather mappings are skipped. A target that gathers on the split must not do
    so on a skipped step. (b) A mixed step, one chunk with 15 decode rows behind it: the split
    counts the context tokens only and covers the context rows; the decode rows keep their own
    schedule."""
    for rank in range(_TEP8_RANKS):
        skipped = _indexer_step(tep8, rank, ctx=[(0, 1000)] * 9)
        assert skipped.skip_indexer_for_ctx_reqs is True
        assert skipped.indexer_prefill_q_split == (1125 * rank, 1125 * (rank + 1), [1125] * 8)
        # Rank r's share runs from inside prompt r to inside prompt r + 1 (rank 0 holds prompt 0
        # whole, rank 7 prompt 8): two chunks, cut at the prompt boundary 1000 (r + 1).
        boundary = 1000 * (rank + 1)
        assert [(c.token_start, c.token_end) for c in skipped.indexer_prefill_chunks] == [
            (1125 * rank, boundary),
            (boundary, 1125 * (rank + 1)),
        ]
        assert skipped.slot_mapping_fp8_fullkv is skipped.slot_mapping_fp8
    dense = skipped.host_topk_indices_buffer[:9000]
    assert torch.equal(dense[999, :1000], torch.arange(1000, dtype=torch.int32))
    assert bool((dense[999, 1000:] == -1).all())

    cached, new = 2 * 32768, 32768 - 32
    decode = [1048000 - 64 * i for i in range(15)]
    mixed = _indexer_step(tep8, 3, ctx=[(cached, new)], gen=decode)
    assert (mixed.num_contexts, mixed.num_generations, mixed.num_tokens) == (1, 15, 32751)
    assert mixed.indexer_prefill_q_split == (3 * 4092, 4 * 4092, _TEP8_SPLITS[32736])
    assert _chunk_ranges(mixed) == [(12276, 16368, 0, cached + 16368)]
    assert mixed.decode_schedules == [15]
    assert mixed.max_gen_seq_len == 1
    assert mixed.host_gen_kv_indptr[-1].item() == sum(decode) + 15


def test_packed_chunks_pass_max_seq_len_keys(tep8):
    """`plan_mla_chunked_prefill_chunks` (dsa/indexer.py) packs consecutive whole context requests
    into one indexer chunk while sum(q) x sum(kv) stays within 1 << 28, and the chunk's keys are
    every packed request's whole KV, cached and new: a chunk can carry more than max_seq_len keys.
    (a) Two requests' 5-token last chunks over 983040-token prefixes (10 context tokens, no
    split): one chunk of n_k 1966090 on every rank. (b) A 50-token last chunk over a 1048480-token
    prefix beside a fresh 200-token prompt: one chunk of n_k 1048730. (c) The two tails of (a)
    beside a fresh 8182-token prompt, 8192 context tokens: split 1024 a rank; rank 0's share
    holds both tails whole, packed into the 1966090-key chunk, then the fresh prompt's first 1014
    rows as a chunk of their own; ranks 1..7 each plan a slice of the fresh prompt over its own
    keys, past the tails' 1966090. Each packed chunk's n_k is bounded by the budget over its new
    tokens and by the KV pool, not by max_seq_len; the gather mappings cover every key."""
    tails = [(983040, 5), (983040, 5)]
    for rank in range(_TEP8_RANKS):
        packed = _indexer_step(tep8, rank, ctx=tails)
        assert packed.indexer_prefill_q_split is None
        assert _chunk_ranges(packed) == [(0, 10, 0, 1966090)]
    assert 10 * 1966090 <= 1 << 28
    assert tuple(packed.slot_mapping_fp8_fullkv.shape) == (1966090,)
    table = packed.host_indexer_k_cache_block_offsets
    for request, position in ((0, 0), (0, 983044), (1, 0), (1, 983044)):
        fp8, scale = _key_slots(table, request, torch.tensor([position]))
        flat = request * 983045 + position
        assert int(packed.slot_mapping_fp8_fullkv[flat]) == int(fp8)
        assert int(packed.slot_mapping_scale_fullkv[flat]) == int(scale)

    beside = _indexer_step(tep8, 0, ctx=[(1048480, 50), (0, 200)])
    assert beside.indexer_prefill_q_split is None
    assert _chunk_ranges(beside) == [(0, 250, 0, 1048730)]
    assert 1048730 > _MAX_SEQ_LEN and 250 * 1048730 <= 1 << 28
    assert 1048480 % _TOKENS_PER_BLOCK == 0 and 1048480 + 50 < _MAX_SEQ_LEN

    for rank in range(_TEP8_RANKS):
        split = _indexer_step(tep8, rank, ctx=[*tails, (0, 8182)])
        assert split.indexer_prefill_q_split == (
            1024 * rank,
            1024 * (rank + 1),
            _TEP8_SPLITS[8192],
        )
        if rank == 0:
            assert _chunk_ranges(split) == [(0, 10, 0, 1966090), (10, 1024, 1966090, 1967104)]
        else:
            q_end = 1024 * (rank + 1)
            assert _chunk_ranges(split) == [(1024 * rank, q_end, 1966090, 1966090 + q_end - 10)]
        assert tuple(split.slot_mapping_fp8_fullkv.shape) == (1966090 + 8182,)


def test_tep8_indexer_chunking_under_each_envelope(tep8, tep8_32k):
    """`Indexer.prepare_for_chunked_prefill` (dsa/indexer.py) on the largest step of each engine.
    Delivered (32768 unchunked, no cached KV) it takes the indexer's own chunking:
    `indexer_max_chunk_size` resolves to its 32768 default, `select_indexer_chunk_size` keeps it
    below 256K keys, and a whole 32768-token prompt (or the warmup's and KV estimation's 32767 +
    1) is one chunk, so no chunk list is built and the query is never split. At the 1M config the
    MLA-chunked-prefill branch plans the chunks instead, by request and packed logits, and the
    warmup's and KV estimation's one 32768-token prompt is itself a split step: each of the eight
    ranks plans 4096 of its rows, so every boot runs the split before the pool is sized.
    `indexer_max_chunk_size` is not read there."""
    params = _metadata_params(tep8_32k)
    assert params.indexer_max_chunk_size == 32768 >= tep8_32k.max_num_tokens
    assert select_indexer_chunk_size(params.indexer_max_chunk_size, 32768) == 32768
    groups = split_prefill_chunks(torch.tensor([32768], dtype=torch.int32), 32768)
    assert groups == [[(0, 0, 32768, 0)]]
    for ctx in ([(0, 32768)], [(0, 32767), (0, 1)]):
        for rank in (0, _TEP8_RANKS - 1):
            step = _indexer_step(tep8_32k, rank, ctx=ctx)
            assert step.indexer_prefill_chunks is None and step.indexer_prefill_q_split is None
            assert step.slot_mapping_fp8_fullkv is step.slot_mapping_fp8

    for rank in range(_TEP8_RANKS):
        step = _indexer_step(tep8, rank, ctx=[(0, 32768)])
        assert step.indexer_prefill_q_split == (4096 * rank, 4096 * (rank + 1), [4096] * 8)
        assert _chunk_ranges(step) == [(4096 * rank, 4096 * (rank + 1), 0, 4096 * (rank + 1))]
        assert step.slot_mapping_fp8_fullkv is step.slot_mapping_fp8


def test_tep8_context_rows_take_the_path_of_their_kv(tep8):
    """The target's context-row path (`PrefillTarget._ctx_workspace`) on the engine's own host
    metadata at the 1M config, on the eight TP ranks, over the held workspace's rows the target
    takes from this engine (`_ctx_workspace_rows`: 2**19). It reads a step's context KV as
    cached + new over its context requests (`_ctx_kv_offsets`, the host
    `ctx_kv_indptr[num_contexts]`), never `num_ctx_kv_tokens`, which counts the new tokens only;
    the query split changes only which rows a rank selects for, so every rank takes the same
    path:

    * the KV-capacity estimate's step, one fresh 32768-token prompt split 4096 a rank, takes the
      workspace path, and the first context step allocates the held bf16 [2**19, 576] workspace;
    * the 1M gate prompt's chunks, each a query-split step with the indexer running: the first
      16 (context KV up to 2**19) on the workspace path, from the 17th on the decode path in
      sub-batches of 8192, the last (32751 rows behind 1015808 cached) 8192 x 3 and 8175;
    * one row of context KV either side of 2**19, with a decode row behind;
    * a packed step of four chunks over 1032192, 524288, 0 and 2048 cached keys with twelve
      decode rows behind them: the decode path over the 32752 context rows only;
    * nine prompts within index_topk (the context indexer skipped, the split still planned): the
      workspace path.

    A workspace-path step keeps layer 35's gather beside its rows when its context KV fits the
    held rows twice, and gathers it again for each sharer otherwise. The workspace is never
    reallocated after the first step."""
    rows_b = tep4_modeling._ctx_workspace_rows(_target_sizing(tep8))
    assert (rows_b, tep4_modeling._CTX_DECODE_ROWS) == (2**19, 8192)

    def path(target, rank, ctx, gen=()):
        md = _indexer_step(tep8, rank, ctx=ctx, gen=gen)
        step = dict(
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            **tep4_modeling._ctx_kv_offsets(md, md.num_contexts),
        )
        n_kv = sum(c + n for c, n in ctx)
        assert int(md.host_ctx_kv_indptr[md.num_contexts]) == n_kv
        assert not any(c for c, _ in ctx) or md.num_ctx_kv_tokens == md.num_ctx_tokens < n_kv
        ws = target._ctx_workspace(step, md.num_ctx_tokens, torch.device("cpu"))
        split = md.indexer_prefill_q_split
        if ws.ws_buf is not None:
            assert n_kv <= rows_b and ws.ws_buf.shape == (n_kv, 576)
            assert ws.ws_buf.data_ptr() == target._ws_buf.data_ptr()
            # Layer 35's gather is kept beside the step's rows where they fit twice.
            assert (ws.kept is not None) == (2 * n_kv <= rows_b)
            return "workspace", split, None
        assert n_kv > rows_b and sorted(ws.sched) == sorted({hi - lo for lo, hi in ws.sub_batches})
        return "decode", split, ws.sub_batches

    full = [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
    targets = []
    for rank in range(_TEP8_RANKS):
        target = tep4_modeling.PrefillTarget.__new__(tep4_modeling.PrefillTarget)
        target.core = SimpleNamespace(lat_dim=576, ctx_workspace_rows=rows_b)
        share = (4096 * rank, 4096 * (rank + 1), [4096] * 8)
        assert path(target, rank, [(0, _MAX_NUM_TOKENS)]) == ("workspace", share, None)
        held = target._ws_buf
        assert held.shape == (rows_b, 576) and held.dtype == torch.bfloat16
        targets.append((target, held))
    for k, new in enumerate(_CHUNKS):
        # Every rank on the chunks either side of 2**19 and on the last; one rank on the rest.
        for rank in range(_TEP8_RANKS) if k in (15, 16, 31) else [k % _TEP8_RANKS]:
            target = targets[rank][0]
            got = path(target, rank, [(sum(_CHUNKS[:k]), new)])
            q_begin, q_end, sizes = got[1]
            assert sizes == _TEP8_SPLITS[new] and q_end - q_begin == sizes[rank]
            if k < 16:
                assert got[0] == "workspace", k
            elif k < 31:
                assert got[0::2] == ("decode", full), k
            else:
                assert got[0::2] == ("decode", full[:3] + [(24576, 32751)])
    target, held = targets[0]
    edge = rows_b - 32767
    assert path(target, 0, [(edge, 32767)], gen=[4096])[0] == "workspace"
    past = path(target, 0, [(edge + 1, 32767)], gen=[4096])
    assert past[0::2] == ("decode", full[:3] + [(24576, 32767)])
    packed = [(1032192, 16384), (524288, 8192), (0, 4096), (2048, 4080)]
    for rank, (target, _) in enumerate(targets):
        got = path(target, rank, packed, gen=[1048000 - 64 * i for i in range(12)])
        assert got[0::2] == ("decode", full[:3] + [(24576, 32752)])
        assert got[1] is not None
        skipped = path(target, rank, [(0, 1000)] * 9)
        assert skipped == ("workspace", (1125 * rank, 1125 * (rank + 1), [1125] * 8), None)
    assert all(target._ws_buf is held for target, held in targets)


# ------------------------------------------------------- tep8 (5): the scheduler's chunks


def test_tep8_scheduler_admits_the_envelope_prompt_in_chunks(tep8):
    """`KVCacheV2Scheduler` (pyexecutor/scheduler/scheduler_v2.py), the scheduler of a V2 KV
    manager, with the chunking `_create_py_executor` configures. One scheduler serves the eight
    ranks. It schedules the decode rows first, then cuts the context chunk from what the step's
    32768 tokens leave, rounded down to the 32-token unit unless it is the prompt's last. The
    configured GUARANTEED_NO_EVICT is served as MAX_UTILIZATION. Alone, the 1048559-token prompt
    takes 32 steps, 31 of 32768 and a last of 32751; beside 15 decode rows each chunk is 32736,
    context and decode rows 32751 of the 32768, and the prompt takes 33 steps, the last 1007."""
    assert tep8.llm_args.scheduler_config.capacity_scheduler_policy == (
        CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    )
    alone = _schedule_prompt(tep8, _ENVELOPE_PROMPT, 0)
    assert alone == [(c, 0) for c in _CHUNKS]
    assert len(alone) == 32 and alone[-1] == (32751, 0)

    beside = _schedule_prompt(tep8, _ENVELOPE_PROMPT, 15)
    assert beside == [(32736, 15)] * 32 + [(1007, 15)]
    assert (32768 - 15) // 32 * 32 == 32736
    assert all(chunk + rows <= _MAX_NUM_TOKENS for chunk, rows in beside)
    assert sum(chunk for chunk, _ in beside) == _ENVELOPE_PROMPT


# ------------------------------------------------------- tep8 (6): KV pool, what max_seq_len sizes


def test_kv_pool_as_at_tep4(tep8, tep8_gate, tep8_32k, tep8_32k_gate, tep4):
    """MLA's latent and the indexer keys are not head-sharded, and every rank holds all 16
    requests: a token costs the same on each of the eight ranks as on each tep4 rank, at the 1M
    config as delivered (`MistralLarge4CacheManager.get_cache_size_per_token`,
    attention/backends/sparse/mistral_large4/cache_manager.py) -- an fp8_ds_mla latent of 656
    bytes (512 e4m3, four fp32 tile scales, 64 bf16 rope) on each of the 63 KV owners (the nine
    sharers keep no latent) plus a 128-byte fp8 indexer key and its fp32 scale on each of the 44
    full-indexer layers, in pages of 32, in the manager the sparse registry picks for a
    MistralLarge4Config, with no per-request intercept. The delivered pool is that
    times 16 x 32768 tokens; at 1M the pool is sized by free memory, not by 16 x
    max_seq_len."""
    for resolved in (tep8, tep8_gate, tep8_32k, tep8_32k_gate, tep4):
        model_config, kv_cache_config = resolved.model_config, resolved.llm_args.kv_cache_config
        assert get_kv_cache_manager_cls(model_config, kv_cache_config) is (
            MistralLarge4CacheManager
        )
        assert kv_cache_config.tokens_per_block == 32

    cfg, sparse = tep8.model_config.pretrained_config, tep8.model_config.sparse_attention_config
    latent = MistralLarge4CacheManager.latent_token_bytes
    owners = sum(1 for i, s in enumerate(cfg.kv_source_layers) if s == i)
    indexer_key = sparse.index_head_dim + sparse.index_head_dim // 128 * 4
    full_indexer_layers = cfg.index_topk_pattern.count("F")
    per_token = latent * owners + indexer_key * full_indexer_layers
    assert (latent, owners, indexer_key, full_indexer_layers) == (656, 63, 132, 44)
    assert per_token == _KV_BYTES_PER_TOKEN == 47136

    for rank in range(_TEP8_RANKS):
        cost = _kv_cost(tep8, rank)
        assert (cost.slope, cost.intercept) == (per_token, 0)
        assert cost == _kv_cost(tep8_gate, rank) == _kv_cost(tep8_32k, rank)
        assert cost == _kv_cost(tep8_32k_gate, rank) == _kv_cost(tep4, rank % 4)
        assert cost.bytes_for_tokens(32) == 32 * per_token == 1_508_352

    tokens = tep8_32k.max_batch_size * tep8_32k.max_seq_len
    pool = _kv_cost(tep8_32k, 0).bytes_for_tokens(tokens)
    assert pool == tokens * per_token == 24_712_839_168
    assert math.isclose(pool / 2**30, 23.016, abs_tol=5e-4)
    request = _kv_cost(tep8, 0).bytes_for_tokens(_MAX_SEQ_LEN)
    assert request == 49_425_678_336 and math.isclose(request / 2**30, 46.03, abs_tol=5e-3)


def test_tep8_buffers_sized_by_max_seq_len(tep8, tep8_32k):
    """The KV manager's page tables are max_seq_len wide: `KVCacheManagerV2.__init__`
    (pyexecutor/kv_cache/kv_cache_manager_v2.py) takes max_seq_len plus one token of generation
    headroom (`_get_generation_kv_capacity`) in pages of 32 and pads to a multiple of 4: 32772
    columns at 1048576 (1028 at 32768). The attention metadata the target reads
    (`kv_cache_block_offsets`, the DSA `block_table`, `indexer_k_cache_block_offsets`,
    `block_table_expanded`) is 16 sequences by that width. Every other buffer -- the selection
    buffer `topk_indices_buffer` [32768, 2048], the radix top-k's `radix_aux_indices` /
    `radix_aux_logits` [16, 10, 2048], DeepGEMM's `scheduler_metadata_buffer` [num_sms + 1, 2],
    the slot mappings and query bounds [32768] -- is the same at both configs; none is
    max_batch_size x max_seq_len, and no buffer has a max_seq_len dimension."""
    assert _get_generation_kv_capacity(None, is_draft=False) == (0, 1)
    assert _max_blocks_per_seq(_MAX_SEQ_LEN) == 32772 and _max_blocks_per_seq(32768) == 1028
    wide, narrow = _metadata_buffers(tep8), _metadata_buffers(tep8_32k)

    tables = {
        "kv_cache_block_offsets": ((1, 16, 2), torch.int32),
        "block_table": ((16,), torch.int32),
        "indexer_k_cache_block_offsets": ((16,), torch.int32),
        "block_table_expanded": ((16,), torch.int32),
    }
    for name, (rows, dtype) in tables.items():
        assert wide[name] == (rows + (32772,), dtype)
        assert narrow[name] == (rows + (1028,), dtype)
    assert wide["topk_indices_buffer"] == ((32768, 2048), torch.int32)
    assert wide["radix_aux_indices"] == (
        (16, dsa_metadata._MAX_RADIX_BLOCKS_PER_ROW, 2048),
        torch.int32,
    )
    assert wide["radix_aux_logits"] == ((16, 10, 2048), torch.float32)
    # `_metadata_buffers` stands in 148 for the device's SM count.
    assert wide["scheduler_metadata_buffer"] == ((149, 2), torch.int32)
    assert wide["slot_mapping_fp8"] == wide["slot_mapping_scale"] == ((32768,), torch.int64)
    assert wide["cu_seqlen_ks"] == wide["cu_seqlen_ke"] == ((32768,), torch.int32)
    assert wide["shared_topk_indices"] == ((32768, 2048), torch.int32)
    assert wide["ctx_cached_token_indptr"] == wide["ctx_kv_indptr"] == ((17,), torch.int64)

    # Under cached-KV MLA the base metadata also keeps the uncached prefix sums.
    assert set(wide) - set(narrow) == {"ctx_uncached_token_indptr"}
    assert set(narrow) <= set(wide)
    for name, (shape, dtype) in wide.items():
        assert _MAX_SEQ_LEN not in shape, name
        assert _MAX_BATCH_SIZE * _MAX_SEQ_LEN not in shape, name
        if name not in tables and name in narrow:
            assert (shape, dtype) == narrow[name], name


# The KV estimation's temporary pool at the 1M config, in pages and tokens.
_ESTIMATION_PAGES = 32784
_ESTIMATION_TOKENS = _ESTIMATION_PAGES * _TOKENS_PER_BLOCK


def test_tep8_kv_estimation_profiles_one_full_step(tep8, tep8_32k):
    """Before the pool exists `KvCacheCreator` (pyexecutor/_util.py) profiles dummy context
    requests as long as a request may be, net max_seq_len - 1, but no more than max_num_tokens in
    all, for the one scheduler the eight ranks share: at the 1M config one 32768-token prompt (the
    delivered engine: a 32767- and a 1-token prompt). The DSA config keeps it off the chunked-MLA
    profile (`_get_mla_chunked_profile_length`), so the profiled step is one 32768-token context
    with nothing cached -- a split step at the 1M config (see the indexer chunking test).
    `_get_token_num_for_estimation` sizes the temporary pool for the larger of those prompts and
    CUDA-graph capture's longest decode row: ceil((max_seq_len + 1) / 32) + 16 - 1 pages, 32784 at
    the 1M config -- 1049088 tokens, 46.05 GiB at 47136 bytes a token. It holds a max_seq_len
    request, so the estimation KV manager does not lower max_seq_len (`KVCacheManagerV2.__init__`
    caps it by `get_num_available_tokens`, which a V2 estimate caps by these max_tokens, and
    `KvCacheCreator._create_kv_cache_manager` would carry a lowered value to every later
    manager). No buffer is max_batch_size x max_seq_len."""
    expected = {
        id(tep8): ([32768], _ESTIMATION_PAGES),
        id(tep8_32k): ([32767, 1], 1040),
    }
    for resolved in (tep8, tep8_32k):
        creator = _kv_creator(resolved)
        requests = creator._create_dummy_context_requests(max(1, resolved.max_seq_len - 1))
        assert creator._mla_chunked_profile_length is None
        lengths = [len(request.input_token_ids) for request in requests]
        prompts, pages = expected[id(resolved)]
        assert lengths == prompts and sum(lengths) == resolved.max_num_tokens
        creator._dummy_reqs = requests
        tokens = creator._get_token_num_for_estimation()
        assert tokens == pages * _TOKENS_PER_BLOCK
        assert tokens >= resolved.max_seq_len + 1
    assert _ESTIMATION_TOKENS == 1_049_088 < _MAX_BATCH_SIZE * _MAX_SEQ_LEN
    assert math.isclose(_ESTIMATION_TOKENS * _KV_BYTES_PER_TOKEN / 2**30, 46.05, abs_tol=5e-3)


def test_tep8_kv_estimation_reserves_no_workspace(tep8):
    """What `KvCacheCreator.configure_kv_cache_capacity` (pyexecutor/_util.py) takes out of the KV
    budget beside the profiled peak, at the 1M config on sm_103, on every rank's mapping: the
    indexer MQA-logits workspace (`get_indexer_mqa_logits_workspace_bytes`, a tile of
    `_INDEXER_MQA_LOGITS_ELEM_BUDGET`) is reserved only when the model engine's sparse attention
    config is the deepseek_v4 algorithm, and this one is DSA; the TRTLLM backend declares no
    context-MLA workspace for fp8 DSA on sm_103
    (`TrtllmAttention.runtime_workspace_bytes_per_token`, attention/backends/trtllm.py: sparse
    MLA stages no buffer), so `get_mla_context_workspace_kv_len_cap` installs no attended-KV cap
    though chunked prefill is on (with no workspace the creator keeps the chunk-bounded default
    and asks no chunk profile), `KvCacheCreator._fp8_ctx_mla_kv_len_cap` stays None and no fixed
    capacity-sized reserve is taken. The pool is the profiled budget whole."""
    model_config = tep8.model_config
    assert model_config.sparse_attention_config.algorithm == "dsa" != "deepseek_v4"
    quant_mode = model_config.quant_config.quant_mode
    assert quant_mode.has_fp8_kv_cache() and not quant_mode.has_fp4_kv_cache()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trtllm_backend, "get_sm_version", lambda: 103)
        mp.delenv("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", raising=False)
        for rank in range(_TEP8_RANKS):
            mapping = _rank_mapping(tep8.mapping, rank)
            w = get_attention_workspace_bytes_per_token(model_config, mapping)
            fixed = get_attention_workspace_fixed_bytes(
                model_config,
                mapping,
                max_batch_size=_MAX_BATCH_SIZE,
                max_num_tokens=_MAX_NUM_TOKENS,
                max_seq_len=_MAX_SEQ_LEN,
                enable_cuda_graph=True,
            )
            assert (w, fixed) == (0, 0)
        assert get_attention_workspace_is_chunked_prefill_bounded(model_config) is True
    cap = get_mla_context_workspace_kv_len_cap(
        tep8.llm_args.kv_cache_config,
        _MAX_BATCH_SIZE,
        _MAX_NUM_TOKENS,
        _MAX_SEQ_LEN,
        tep8.llm_args.enable_chunked_prefill,
        True,
        chunked_workspace_profiled=False,
        require_chunked_workspace_profile=w > 0,
    )
    assert cap is None
    assert tep8.llm_args.kv_cache_config.fp8_context_mla_kv_len_cap is None


@pytest.mark.parametrize("tep8_envelope", ["1m", "32k"])
def test_warmup_runs_one_full_step_on_every_rank(request, tep8_envelope):
    """The warmup's shapes at tep8 (`DecoderWarmup._get_full_general_warmup_requests`,
    pyexecutor/engine/runners/decoder/warmup.py), on the estimation's pool at the 1M config (the
    warmup runs in the profiling executor first, then again in the final one). Each rank proposes
    its shapes from its own free KV, the eight ranks keep the ones all proposed
    (`WarmupAgreement.agree_shapes`), and every rank enters each shape's forward. The largest is
    min(max_num_tokens, max_batch_size x (max_seq_len - 1)) = 32768 tokens of context in one
    step, which `DummyRequestFactory.create_warmup_request` (decoder/dummy_requests.py) builds as
    prompts of at most max_seq_len - 1: one 32768-token prompt at the 1M config (a 32767- and a
    1-token prompt delivered), never a 1M prompt -- warmup bypasses the scheduler. All eight ranks
    run it on the same tokens, so each rank's routed experts see 32768 tokens, and when every
    token's top-4 falls in one rank's 16 experts, 4 x 32768 = 131072 permuted rows on that rank.
    No prefill CUDA graph is captured (`_filter_piecewise_capture_num_tokens`, model_engine.py:
    its 16 x (max_seq_len - 1) ceiling caps only capture candidates)."""
    wide = tep8_envelope == "1m"
    resolved = request.getfixturevalue("tep8" if wide else "tep8_32k")
    prompts = [32768] if wide else [32767, 1]
    pool_tokens = _ESTIMATION_TOKENS if wide else 16 * resolved.max_seq_len
    config = _runner_config(resolved)
    group = _Group()

    def rank_warmup(rank):
        resources = _Resources(_KvPool(pool_tokens, config.max_seq_len))
        warmup = DecoderWarmup.__new__(DecoderWarmup)
        warmup._config = config
        warmup.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        agreement = WarmupAgreement(_RankDist(group, rank), _rank_mapping(resolved.mapping, rank))
        assert agreement.is_distributed_forward()
        shapes = agreement.agree_shapes(warmup._get_full_general_warmup_requests(resources))
        max_context = max(tokens for tokens, gen in shapes if gen == 0)
        batch = _dummy_factory(resolved).create_warmup_request(resources, max_context, 0)
        enters = agreement.should_run_warmup_batch(batch, max_context, "max context")
        return shapes, enters, [prompt.token_num for prompt in batch.context_requests]

    results = group.run(rank_warmup)
    for shapes, enters, rank_prompts in results:
        assert shapes == [(1, 0), (1, 1), (32768, 0), (16, 16), (2, 0)]
        assert enters is True
        assert rank_prompts == prompts

    cfg = resolved.model_config.pretrained_config
    rows = sum(results[0][2])
    assert rows == resolved.max_num_tokens == 32768
    local_experts = cfg.n_routed_experts // resolved.mapping.moe_ep_size
    assert local_experts == 16 >= cfg.num_experts_per_tok == 4
    assert rows * cfg.num_experts_per_tok == 131072

    kept, unrecordable = _filter_piecewise_capture_num_tokens(
        config.cuda_graph_batch_sizes,
        resolved.max_num_tokens,
        _MAX_BATCH_SIZE,
        resolved.max_seq_len,
    )
    assert (kept, unrecordable) == ([1, 2, 4, 8, 16], [])
    assert resolved.llm_args.prefill_cuda_graph_backend == "disabled"


@pytest.mark.parametrize("tep8_envelope", ["1m", "32k"])
def test_tep8_cuda_graph_capture_decodes_at_max_seq_len(request, tep8_envelope):
    """`DecoderCudaGraphs.warmup_and_capture` (pyexecutor/engine/runners/decoder/graphs.py)
    captures every graph batch (1, 2, 4, 8, 16) twice for DSA, whose config asks for separate
    short and long graphs (`needs_separate_short_long_cuda_graphs`, its threshold index_topk
    2048): once at max_seq_len and once at 2048 less the one token a decode step adds.
    `create_cuda_graph_warmup_request` (decoder/dummy_requests.py) fills each batch with one-token
    decode rows and one row of the family's longest history (`add_longest_dummy_request`):
    1048575 tokens at the 1M config -- every boot captures the decode kernels at KV length 1048576
    -- and 2046 (32767 and 2046 delivered). At batch 16 the long family needs 32768 + 15 pages,
    within the 32784 the estimation pool holds, so capture sends no request the pool cannot
    hold."""
    wide = tep8_envelope == "1m"
    resolved = request.getfixturevalue("tep8" if wide else "tep8_32k")
    sparse = copy.deepcopy(resolved.model_config.sparse_attention_config)
    assert sparse.needs_separate_short_long_cuda_graphs() is True
    families = [resolved.max_seq_len, sparse.seq_len_threshold - 1]
    longest_rows = (resolved.max_seq_len - 1, 2046)
    assert families == ([1048576, 2047] if wide else [32768, 2047])

    config = _runner_config(resolved)
    assert config.cuda_graph_batch_sizes == [1, 2, 4, 8, 16]
    for family, longest in zip(families, longest_rows):
        for batch_size in config.cuda_graph_batch_sizes:
            pool = _KvPool(_ESTIMATION_TOKENS, resolved.max_seq_len)
            batch = _dummy_factory(resolved).create_cuda_graph_warmup_request(
                _Resources(pool), batch_size, 0, family
            )
            rows = [row.token_num for row in batch.generation_requests]
            assert rows == [longest] + [1] * (batch_size - 1)
            assert all(row.is_gen for row in batch.generation_requests)
            pages = math.ceil((longest + 1) / _TOKENS_PER_BLOCK) + (batch_size - 1)
            assert pages <= _ESTIMATION_PAGES
    assert math.ceil(1048576 / _TOKENS_PER_BLOCK) + 15 == 32783


def test_tep8_pool_holds_the_envelope_load(monkeypatch, tep8, tep8_gate, tep8_32k):
    """Arithmetic on the pools measured at tep8 on the fp8_ds_mla cache, at 47136 bytes a token
    (`MistralLarge4CacheManager.get_cache_size_per_token`): 3,402,400 tokens a rank at 32768 at the
    accuracy gate's KV fraction 0.85 (core.yaml's), and 3,581,536 at the 1M config at the boot
    gate's 0.9 (the quickstart's). Each is also read at `KvCacheConfig.max_util_for_resume` (0.95)
    of its size, the share of its quota the V2 manager (`KVCacheManagerV2.__init__`) syncs as the
    ranks' resumable token capacity, in case the scheduler holds requests to it. The V2 manager
    gives a request ceil((tokens + 1) / 32) pages. The 1M config adds width-sized allocations the
    measured 32768 engine did not have (the 1M pool has its rope tables out already; charging them
    again only understates what is left): the target's rope tables at 1048576 rows (indexer 64 x
    fp32, identity 64 x fp32: 512 MiB against their 16 MiB at 32768 rows; the 16 MiB `mla` layout
    that engine also held, which the target no longer builds, not credited), one gathered indexer
    key block of 1048576 x 132 bytes, the indexer logits tile at the budget the target takes from
    each engine (`_prefill_logits_budget`: 1 << 28, 1 GiB, at 1M against 1 << 24, 64 MiB, at 32768),
    the decode scorer's [16, 1048576] fp32 arena, and the held context workspace
    (`PrefillTarget._ctx_workspace`, `_ctx_workspace_rows` at 1M): 2**19 rows of bf16 576 with the
    int32 request and position maps of a step that fills it (a step with more context KV attends its
    context rows straight off the pool). Charged to the pool whole, they cost 49,517 tokens. What is
    left of either pool, read either way, holds the 1M gate's load -- the 1048559-token prompt
    generating 16 tokens beside 15 short requests (its probes and fillers), each bounded here by a
    full step -- and three requests of max_seq_len, though not four: the latent is replicated over
    the eight ranks, so the engine holds three 1M requests at once."""
    fractions = {
        tep8_gate.llm_args.kv_cache_config.free_gpu_memory_fraction: _TEP8_BOOT_POOL_TOKENS,
        tep8.llm_args.kv_cache_config.free_gpu_memory_fraction: _TEP8_ACCURACY_POOL_TOKENS,
    }
    assert fractions == {0.9: 3_581_536, 0.85: 3_402_400}
    resume = tep8.llm_args.kv_cache_config.max_util_for_resume
    assert resume == tep8_gate.llm_args.kv_cache_config.max_util_for_resume == 0.95

    mib = 2**20
    monkeypatch.delenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", raising=False)
    wide, narrow = _target_sizing(tep8), _target_sizing(tep8_32k)
    budgets = [
        tep4_modeling._prefill_logits_budget(md.max_num_tokens, md.max_seq_len)
        for md in (wide, narrow)
    ]
    assert budgets == [1 << 28, 1 << 24]
    held = tep4_modeling._ctx_workspace_rows(wide)
    grown = {
        "rope tables": (1048576 - 32768) * (64 + 64) * 4,
        "gathered keys": (1048576 - 32768) * 132,
        "logits tile": (budgets[0] - budgets[1]) * 4,
        "decode arena": 16 * (1048576 - 32768) * 4,
        "context workspace": held * (576 * 2 + 2 * 4),
    }
    assert 1048576 * (64 + 64) * 4 == 512 * mib
    assert held * 576 * 2 == 576 * mib
    assert sum(grown.values()) == 2_333_999_104 < 2.2 * 2**30
    lost = math.ceil(sum(grown.values()) / _KV_BYTES_PER_TOKEN)
    assert lost == 49_517

    gate = _pages(_ENVELOPE_PROMPT + 16) + 15 * _pages(_MAX_NUM_TOKENS + 16)
    assert gate * _TOKENS_PER_BLOCK == 1_540_576
    three = 3 * _pages(_MAX_SEQ_LEN) * _TOKENS_PER_BLOCK
    four = 4 * _pages(_MAX_SEQ_LEN) * _TOKENS_PER_BLOCK
    assert (three, four) == (3_145_824, 4_194_432)
    for measured in (_TEP8_BOOT_POOL_TOKENS, _TEP8_ACCURACY_POOL_TOKENS):
        assert measured < four
        for pool in (measured - lost, measured * 95 // 100 - lost):
            assert gate * _TOKENS_PER_BLOCK <= pool
            assert three <= pool < four
    # The tightest reading: the accuracy gate's pool at 0.95, 36,939 tokens past three.
    assert _TEP8_ACCURACY_POOL_TOKENS * 95 // 100 - lost - three == 36_939


# ------------------------------------------------------- tep8 (7): the engine's node layout


def test_shell_lm_head_at_tp8(tep8, tep4):
    """The causal-LM shell every target subclasses builds the vocab-parallel lm_head from the TP
    width: on each rank an eighth of the 204800-token vocabulary, 25600 rows of 6144, gathered
    over the TP group of all eight ranks (``allgather`` over ``mapping.tp_group``) and with no
    AllReduce of its own. The node layout does not enter. Built on the meta device: no weights."""

    def lm_head(resolved, rank):
        config = dataclasses.replace(
            resolved.model_config, mapping=_rank_mapping(resolved.mapping, rank)
        )
        cfg = config.pretrained_config
        with torch.device("meta"):
            shell = DecoderModelForCausalLM(
                DecoderModel(config),
                config=config,
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
            )
        head = shell.lm_head
        return dict(
            tp=(head.tp_size, head.tp_rank, head.tp_mode),
            shape=tuple(head.weight.shape),
            gather=(head.gather_output, config.lm_head_gather_output),
            all_reduce=head.all_reduce,
            group=head.mapping.tp_group,
            quant=head.quant_config,
            pp=(shell.pp_rank, shell.pp_size),
        )

    for rank in range(_TEP8_RANKS):
        head = lm_head(tep8, rank)
        assert head["tp"] == (8, rank, TensorParallelMode.COLUMN)
        assert head["shape"] == (204800 // 8, 6144) == (25600, 6144)
        assert head["gather"] == (True, True) and head["all_reduce"] is None
        assert head["group"] == list(range(_TEP8_RANKS)) and head["pp"] == (0, 1)
    tep4_head = lm_head(tep4, 1)
    assert tep4_head["tp"] == (4, 1, TensorParallelMode.COLUMN)
    assert tep4_head["shape"] == (51200, 6144) and tep4_head["all_reduce"] is None


def test_collective_paths_read_the_node_layout(tep8, tep4):
    """Where the engine's collectives read the node layout at tep8, on each rank's mapping:

    * the rank's mapping is an ``MpiTopology`` and its communicator an ``MPIDist`` (the MPI
      launch leaves TLLM_DISABLE_MPI unset; ``DeviceMeshTopology`` and ``TorchDist`` are the
      non-MPI path);
    * the MNNVL communicator (``MnnvlMemory.get_comm``) splits by pipeline, context and MoE TP
      rank, keyed by TP rank: one communicator of all eight ranks, in TP order; an MNNVL buffer
      sits on device ``mapping.local_rank``, the device the worker set;
    * an engine ``AllReduce`` at the default AUTO strategy takes MNNVL only for a group that spans
      nodes (``MNNVLAllReduce.is_mnnvl`` asks ``mapping.is_multi_node()``: true at tep8, false at
      tep4), and the one-node IPC workspace (``IpcMemory``) stays closed past one node's devices.
      No userbuffers: the strategy is AUTO, not NCCL_SYMMETRIC, and torch.compile is off."""
    llm_args = tep8.llm_args
    assert llm_args.allreduce_strategy == "AUTO" and llm_args.torch_compile_config is None
    signature = MnnvlMemory.comm_layout_signature(tep8.mapping)
    assert signature == (8, 1, 1, 8, 1, 8, 1)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("TLLM_DISABLE_MPI", raising=False)
        for rank in range(_TEP8_RANKS):
            rank_mapping = _rank_mapping(tep8.mapping, rank)
            assert type(rank_mapping) is MpiTopology
            assert type(Distributed.get(rank_mapping)) is MPIDist
            assert MnnvlMemory.comm_split_color_key(rank_mapping) == (0, rank)
            assert distributed_ops._mnnvl_device_index(rank_mapping) == rank % 4
            assert rank_mapping.is_multi_node() is True
            assert IpcMemory(rank_mapping, 1024).open_ipc is False
        for rank in range(4):
            rank_mapping = _rank_mapping(tep4.mapping, rank)
            assert MnnvlMemory.comm_split_color_key(rank_mapping) == (0, rank)
            assert rank_mapping.is_multi_node() is False


def test_moe_autotuner_at_sixteen_local_experts(tep8, tep8_32k, tep4):
    """The autotuner's tuning group is the TP group, ``(tp_size, tp_rank)``: all eight ranks over
    both nodes. The NVFP4 MoE runner tunes INDEPENDENT, each rank alone, so tuning sends nothing
    between the nodes; its expert-parallel size, num_experts // local_num_experts, is 8 at tep8
    and moves no bucket without attention DP: the buckets and the token-to-bucket rule are tep4's,
    32768 its top bucket when the target tunes to max_num_tokens. ModelConfig's MoE token budget
    reads the attention-DP width only (32768, at the 1M config as delivered: a chunk is at most
    max_num_tokens), and no MoE load balancer is built."""
    cfg = tep8.model_config.pretrained_config
    for rank in range(_TEP8_RANKS):
        tuner = SimpleNamespace(mapping=_rank_mapping(tep8.mapping, rank))
        assert AutoTuner._tuning_group(tuner) == (8, rank)

    ep_size = cfg.n_routed_experts // (cfg.n_routed_experts // tep8.mapping.moe_ep_size)
    assert ep_size == 8
    tep8_tuning = FP4BlockScaleMoERunner.get_tuning_config(
        ep_size, tune_max_num_tokens=tep8.max_num_tokens, use_dp=False
    )
    tep4_tuning = FP4BlockScaleMoERunner.get_tuning_config(
        4, tune_max_num_tokens=tep4.max_num_tokens, use_dp=False
    )
    assert tep8_tuning.distributed_tuning_strategy == DistributedTuningStrategy.INDEPENDENT
    (tep8_spec,) = tep8_tuning.dynamic_tensor_specs
    (tep4_spec,) = tep4_tuning.dynamic_tensor_specs
    assert tep8_spec.gen_tuning_buckets == tep4_spec.gen_tuning_buckets
    assert max(tep8_spec.gen_tuning_buckets) == 32768
    for tokens in (1, 3, 16, 17, 4097, 8192, 8193, 16384, 32767, 32768):
        assert tep8_spec.map_to_tuning_buckets(tokens) == tep4_spec.map_to_tuning_buckets(tokens)
    assert tep8_spec.map_to_tuning_buckets(32768) == 32768

    for resolved in (tep8, tep8_32k):
        model_config = resolved.model_config
        assert model_config.moe_max_num_tokens == resolved.max_num_tokens == 32768
        assert resolved.mapping.dp_size == 1
        assert resolved.llm_args.moe_config.load_balancer is None
        assert model_config.moe_load_balancer is None
        for rank in range(_TEP8_RANKS):
            with maybe_create_moe_load_balancer(
                model_config, _rank_mapping(resolved.mapping, rank)
            ) as lb:
                assert lb is None


def test_kv_host_tier_is_shared_by_the_node_four_ways():
    """The V2 KV manager provisions a host tier per rank (no host_cache_size is set): its device
    quota, capped at half the node's available physical pages divided by the ranks on the node --
    ``Distributed.local_world_size``, the size of the MPI communicator split by host, four at
    tep8 -- then the minimum over all eight ranks (`_compute_auto_host_tier_quota`,
    pyexecutor/kv_cache/kv_cache_manager_v2.py). So a tep8 node's four tiers together hold at most
    half the host memory available when the pool is built."""
    gib = 2**30
    quota = 160 * gib
    assert _compute_auto_host_tier_quota(quota, 4, 800 * gib, float("inf")) == 100 * gib
    assert _compute_auto_host_tier_quota(quota, 4, 2000 * gib, float("inf")) == quota
    assert _compute_auto_host_tier_quota(quota, 4, float("inf"), float("inf")) == quota


# ------------------------------------------------------- tep8 (8): beside tep4 and dep4


def test_tep4_and_dep4_route_on_one_node_of_four(tep4, dep4):
    """The NVFP4 checkpoint at tep4 and at dep4, built as the tep8 arguments are, on their own
    base options (32768 tokens, chunked prefill off), resolve and route on one node of four: each
    claimed by its own target (`route`, models/mistral_large4/routing.py), the routed experts
    expert-parallel four ways and not split along their width, each on the checkpoint's anchor,
    the same 47136-byte token of the fp8_ds_mla pool and the DSA config unchanged."""
    cases = (
        (tep4, False, _NVFP4_TEP4_TARGET),
        (dep4, True, "ModelingV2MistralLarge4Nvfp4Sm103Dep4"),
    )
    for resolved, attention_dp, target in cases:
        seen = resolved.routed[0]
        mapping = seen.mapping
        assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (4, 4, 1)
        assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.enable_attention_dp) == (
            4,
            1,
            attention_dp,
        )
        assert mapping.gpus_per_node == 4 and mapping.is_multi_node() is False
        for rank in range(4):
            rank_mapping = _rank_mapping(mapping, rank)
            assert (rank_mapping.moe_ep_rank, rank_mapping.moe_tp_rank) == (rank, 0)
            assert (rank_mapping.local_rank, rank_mapping.node_rank) == (rank, 0)
        assert routing.route(seen.pretrained_config, mapping, _SM103, None).__name__ == target
        assert resolved.llm_args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert resolved.llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert _anchor(resolved.llm_args) == 91.926
        assert (_kv_cost(resolved, 0).slope, _kv_cost(resolved, 0).intercept) == (47136, 0)
        assert resolved.llm_args.enable_chunked_prefill is False
        sparse = resolved.model_config.sparse_attention_config
        assert (sparse.algorithm, sparse.q_split_threshold, sparse.index_topk) == (
            "dsa",
            8192,
            2048,
        )
    assert (tep4.max_num_tokens, tep4.max_seq_len, tep4.max_batch_size) == (32768, 32768, 16)
    assert (dep4.max_num_tokens, dep4.max_seq_len, dep4.max_batch_size) == (32768, 32768, 16)


@pytest.mark.parametrize("tep8_envelope", ["1m", "32k"])
def test_tep8_differs_from_tep4_in_the_world_only(request, tep8_envelope):
    """The same checkpoint at tep8 and at tep4 on the same options resolves to the same
    configuration -- quantization, per-layer quantization, model config, DSA config, limits,
    chunked prefill, CUDA graphs, KV config, the MoE config and token budget, what the router
    reads -- except the mapping's world: its size, the TP and attention-TP widths and the EP
    width; at the 1M config (tep4 on its 1M gate's options) as delivered."""
    wide = tep8_envelope == "1m"
    tep8 = request.getfixturevalue("tep8" if wide else "tep8_32k")
    tep4 = request.getfixturevalue("tep4_1m" if wide else "tep4")
    assert _summary(tep8) == _summary(tep4)
    tep8_mapping, tep4_mapping = tep8.mapping.to_dict(), tep4.mapping.to_dict()
    assert {k for k in tep8_mapping if tep8_mapping[k] != tep4_mapping[k]} == {
        "world_size",
        "tp_size",
        "attn_tp_size",
        "moe_ep_size",
    }


def test_tep4_branch_claims_tep8_at_both_envelopes(deployment):
    """The routing (`route`, models/mistral_large4/routing.py), on what each tep8 engine hands it:
    the tep4 branch, whose target serves eight ranks as well as four, claims the NVFP4 export at
    tep8, at the 1M config as delivered, from core.yaml and from the boot gate's command -- no
    criterion reads max_seq_len or chunked prefill."""
    resolved = deployment[0]
    seen = resolved.routed[0]
    target = routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
    assert target.__name__ == _NVFP4_TEP4_TARGET
    assert routing._nvfp4_mixed_precision(seen.pretrained_config)
