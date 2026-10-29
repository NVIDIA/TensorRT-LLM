# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B-NVFP4 at dep4: the engine's configuration at the checkpoint's own
context, and the attention-DP contract the dep4 forward is written against.

No GPU and no weights. The checkpoint goes through the engine's own configuration path with the
LLM-API options of the dep4 core unit (four ranks, attention data parallel, the routed experts
expert-parallel four ways, 16 requests and 32768 tokens per rank, the fp8_ds_mla latent KV
cache, KV block reuse off) at the 1M config: max_seq_len 1048576 (``max_position_embeddings``)
with chunked prefill on, so a context step carries at most one 32768-token chunk per rank. The
same configuration is reached three ways: from ``spec-1m-kvds-dep4/options/core.yaml`` (the
accuracy and serve gates) and from the argv the boot gate hands
``examples/llm-api/quickstart_advanced.py``, through that script's own parser and
``setup_llm``, both with kv_cache_config.dtype fp8; and with the dtype left at auto, as the CI
class passes its KvCacheConfig. All three build ``MistralLarge4CacheManager``.

* the LLM API front end, ``TorchLlmArgs`` and ``ModelLoader._update_from_hf_quant_config``
  (``llmapi/llm_utils.py``), which settle the key the accuracy harness looks its anchor up by;
* each rank, ``_load_config_and_create_checkpoint_loader`` (``py_executor_creator``) then the
  model loader's ``_load_and_validate_config``, which builds the ``ModelConfig`` and asks the
  modeling_v2 router for a target. The router's ``route`` is replaced by a recorder, so what it
  is handed is observed rather than judged.

What the 1M config changes is then pinned on the engine's own host-side code: the request
admission (``BaseLLM._check_arguments`` and the worker's ``max_tokens`` deduction), the
context-MLA-with-cached-KV switch and the DSA indexer's prefill chunk plan
(``Indexer.prepare_for_chunked_prefill``) for chunks over a cached prefix up to 1M keys, the DSA
KV tables' width, and what boot runs before the KV pool is sized (the KV-capacity estimation's
dummy requests, the context-MLA workspace reserve, the sizes the target fixes on its first step
from this engine's metadata -- its held context workspace, its indexer logits budget and its
expert calls -- the warmup's shapes and the CUDA-graph capture's longest row). Under attention
DP four ranks are simulated in one process (each rank a thread, each collective a barrier).
Device-dependent values (the SM version) are fixed to sm_103 here; nothing is asserted that a
visible device would decide.

The checkpoint is Mistral-native, as published (params.json, tekken.json, consolidated
safetensors), and every spelling loads it with checkpoint_format mistral_large_4 (the options'
key, the boot gate's --checkpoint_format=mistral_large_4). The delivered 32K dep4 configuration
(unchunked) and the checkpoint at tep4 and tp4 are checked to resolve and route as they did
before the 1M config. The checkpoint is looked up under ``LLM_MODELS_ROOT`` (as
``Mistral-Large-4-1T-A52B-NVFP4`` or ``mistral-large-4-nvfp4``) by its params.json, and its tests
are skipped where it is absent.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import sys
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import yaml
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as dsa_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    Indexer,
    _packed_logits_elem_budget,
    _split_prefill_queries,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    _MAX_RADIX_BLOCKS_PER_ROW,
    DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor._util import (
    CacheCost,
    KvCacheCreator,
    _get_num_pool_groups_for_estimation,
    get_attention_workspace_bytes_per_token,
    get_attention_workspace_fixed_bytes,
    get_attention_workspace_is_chunked_prefill_bounded,
    get_kv_cache_manager_cls,
    get_mla_context_workspace_kv_len_cap,
    should_enable_adp_dummy_fixes,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_scheduler_aware_adp_dummy,
)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.pyexecutor.engine.metadata import build_attention_metadata
from tensorrt_llm._torch.pyexecutor.engine.runners.common import (
    get_all_rank_num_tokens,
    pad_input_tokens,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BASE_GENERATION_TOKEN_COUNT,
    KVCacheManagerV2,
    _get_generation_kv_capacity,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader as RankModelLoader
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import derive_attention_dp_per_rank_request_cap
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import ADPRouter, DefaultADPRouter
from tensorrt_llm._torch.pyexecutor.scheduler.rank_state import RankState
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.utils import RequestError, is_context_length_exceeded_message
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend, TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import ModelLoader as FrontEndModelLoader
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_RANKS = 4
# max_num_tokens per rank: under chunked prefill, the most context rows one step carries.
_MAX_NUM_TOKENS = 32768
# The checkpoint's max_position_embeddings, the 1M config's max_seq_len.
_MAX_SEQ_LEN = 1 << 20
_TOKENS_PER_BLOCK = 32

# The topologies. dep4 is the core unit's; tep4 and tp4 are the other NVFP4 targets'
# (spec-tep4 / spec-tp4 options).
_DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
_TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
_TP4 = dict(
    tensor_parallel_size=4,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)


def _options(
    topology: dict,
    max_num_tokens: int,
    max_seq_len: int | None = None,
    chunked=False,
    kv_dtype: str | None = None,
) -> dict:
    """The LLM-API options of a Mistral Large 4 gate on ``topology``, the checkpoint in its
    Mistral-native format, KV block reuse off and the latent KV cache at kv_cache_config's
    default dtype (auto: the checkpoint's fp8_kv_cache) unless ``kv_dtype`` names one. Under
    attention DP max_batch_size and max_num_tokens bind each rank."""
    kv_cache_config = dict(enable_block_reuse=False, free_gpu_memory_fraction=0.85)
    if kv_dtype is not None:
        kv_cache_config.update(dtype=kv_dtype)
    return dict(
        checkpoint_format="mistral_large_4",
        **topology,
        max_batch_size=16,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len or max_num_tokens,
        enable_chunked_prefill=chunked,
        cuda_graph_config=dict(enable_padding=True, max_batch_size=16),
        kv_cache_config=kv_cache_config,
    )


# spec-1m-kvds-dep4/options/core.yaml: the gates pass kv_cache_config.dtype fp8.
_DEP4_1M = _options(_DEP4, _MAX_NUM_TOKENS, _MAX_SEQ_LEN, chunked=True, kv_dtype="fp8")
# The CI class's (TestModelingV2MistralLarge4Nvfp4Sm103Dep4): the same config, the
# dtype left at auto.
_DEP4_1M_CI = _options(_DEP4, _MAX_NUM_TOKENS, _MAX_SEQ_LEN, chunked=True)

# The argv the boot gate hands quickstart_advanced.py (workspace
# gates/geometry/geometry.core.dep4.json), after its --model_dir.
_QUICKSTART_ARGV = (
    "--checkpoint_format=mistral_large_4",
    "--max_tokens=2",
    "--tp_size=4",
    "--moe_ep_size=4",
    "--enable_attention_dp",
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
_QUICKSTART = Path(__file__).resolve().parents[7] / "examples/llm-api/quickstart_advanced.py"

_NVFP4_NAMES = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The GSM8K reference rows, and the key the harness reads them by.
_GSM8K_REFERENCES = (
    Path(__file__).resolve().parents[6] / "integration/defs/accuracy/references/gsm8k.yaml"
)
_MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"

# The per-rank KV pool the delivered 32K dep4 boot gate measured, in tokens, and the bytes a
# token of that pool took: the 576-byte latent on all 72 layers and the indexer key on the 44
# full-indexer layers.
_DELIVERED_POOL_TOKENS = 2_004_896
_DELIVERED_BYTES_PER_TOKEN = 576 * 72 + 132 * 44
# The per-rank pool a boot of the 1M accuracy gate (the Dep4Ctx1m class, KV fraction 0.85)
# measured on the native checkpoint, the vision tower loaded on every rank: 58,747 pages.
_BOOT_POOL_TOKENS = 58_747 * 32


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
    """The configuration a served engine reaches for this checkpoint, front end then rank."""
    model = str(checkpoint_dir)
    llm_args = TorchLlmArgs(model=model, modeling_v2="require", **options)
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


def _quickstart_options(checkpoint_dir: Path) -> dict:
    """The keyword arguments quickstart_advanced.py hands the LLM constructor for the boot
    gate's argv: the script's own ``parse_arguments`` and ``setup_llm``, with ``LLM`` replaced
    by a recorder."""
    if not _QUICKSTART.is_file():
        pytest.skip(f"{_QUICKSTART} is not in this checkout")
    spec = importlib.util.spec_from_file_location("quickstart_advanced", _QUICKSTART)
    quickstart = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quickstart)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys,
            "argv",
            ["quickstart_advanced.py", f"--model_dir={checkpoint_dir}", *_QUICKSTART_ARGV],
        )
        mp.setattr(quickstart, "LLM", lambda **kwargs: kwargs)
        kwargs, _ = quickstart.setup_llm(quickstart.parse_arguments())
    assert kwargs.pop("model") == str(checkpoint_dir)
    return kwargs


@pytest.fixture(scope="module")
def dep4():
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _DEP4_1M)


@pytest.fixture(scope="module")
def dep4_quickstart():
    checkpoint_dir = _checkpoint_dir(_NVFP4_NAMES)
    return _resolve(checkpoint_dir, _quickstart_options(checkpoint_dir))


@pytest.fixture(scope="module", params=["core_yaml", "quickstart"])
def dep4_1m(request):
    """The 1M dep4 engine, as the accuracy gate's options and as the boot gate's argv build it."""
    return request.getfixturevalue("dep4" if request.param == "core_yaml" else "dep4_quickstart")


@pytest.fixture(scope="module")
def dep4_ci():
    """The 1M dep4 engine as the CI class builds it: kv_cache_config.dtype left at auto."""
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _DEP4_1M_CI)


@pytest.fixture(scope="module")
def dep4_32k():
    """The delivered dep4 configuration: 32768 tokens per rank, unchunked."""
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _options(_DEP4, _MAX_NUM_TOKENS))


@pytest.fixture(scope="module")
def tep4():
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _options(_TEP4, _MAX_NUM_TOKENS))


@pytest.fixture(scope="module")
def tp4():
    return _resolve(_checkpoint_dir(_NVFP4_NAMES), _options(_TP4, _MAX_NUM_TOKENS))


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


def _kv_cost(resolved, rank: int) -> CacheCost:
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


def _summary(resolved) -> dict:
    """Everything the engine resolved for a checkpoint except its mapping."""
    pretrained = resolved.model_config.pretrained_config.to_dict()
    pretrained.pop("_name_or_path", None)
    llm_args = resolved.llm_args
    return dict(
        llm_quant=repr(llm_args.quant_config),
        quant=repr(resolved.model_config.quant_config),
        layers={k: repr(v) for k, v in (resolved.model_config.quant_config_dict or {}).items()},
        pretrained=pretrained,
        sparse=resolved.model_config.sparse_attention_config.model_dump(),
        routed=resolved.routed[0].quantization_config,
        sizes=(resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size),
        chunked_prefill=llm_args.enable_chunked_prefill,
        cuda_graphs=llm_args.cuda_graph_config.model_dump(),
        kv_cache=llm_args.kv_cache_config.model_dump(),
    )


def _runtime_features(resolved) -> AttentionRuntimeFeatures:
    """The attention runtime features `_create_py_executor` builds from the LLM arguments."""
    return AttentionRuntimeFeatures(
        chunked_prefill=resolved.llm_args.enable_chunked_prefill,
        cache_reuse=resolved.llm_args.kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=False,
        chunk_size=resolved.max_num_tokens,
    )


def _attention_metadata_kwargs(resolved, rank: int = 0) -> SimpleNamespace:
    """What the decoder runner's ``build_attention_metadata`` hands the DSA metadata class, with
    the class replaced by a recorder (no buffers are allocated)."""
    return build_attention_metadata(
        resolved.model_config,
        max_batch_size=resolved.max_batch_size,
        max_num_tokens=resolved.max_num_tokens,
        max_beam_width=1,
        attention_backend=SimpleNamespace(Metadata=lambda **kwargs: SimpleNamespace(**kwargs)),
        attention_runtime_features=_runtime_features(resolved),
        mapping=_rank_mapping(resolved.mapping, rank),
        cache_indirection=None,
    )


def _v2_max_blocks_per_seq(max_seq_len: int) -> int:
    """``KVCacheManagerV2``'s block-table width: the blocks of ``max_seq_len`` plus the generation
    headroom (``_get_generation_kv_capacity``), padded to a multiple of 4 for the
    copy_block_offsets kernel."""
    _, headroom = _get_generation_kv_capacity(None, is_draft=False)
    blocks = -(-(max_seq_len + headroom) // _TOKENS_PER_BLOCK)
    return -(-blocks // 4) * 4


# ------------------------------------------------------- four ranks in one process


class _TpGroup:
    """The four ranks of the attention-DP group in one process: rank r is a thread, and every
    exchange is a barrier, so each rank's call sees what every rank sent, as over MPI."""

    def __init__(self, size: int = _RANKS):
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

    def dist(self, rank: int) -> "_RankDist":
        return _RankDist(self, rank)


class _RankDist:
    """One rank's communicator: the gathers the engine's host code issues under attention DP."""

    # The engine's own int64 gather, over the `tp_allgather` below.
    tp_allgather_int64 = Distributed.tp_allgather_int64

    def __init__(self, group: _TpGroup, rank: int):
        self.group = group
        self.rank = self.tp_rank = rank
        self.world_size = self.tp_size = group.size
        self.has_cp_helix = False

    def tp_allgather(self, obj, *, small_payload: bool = False):
        return self.group.exchange(self.rank, obj)

    def allgather(self, obj):
        return self.group.exchange(self.rank, obj)


_request_ids = itertools.count(1)


def _decode_row(kv_len: int, *, dummy: bool = False) -> SimpleNamespace:
    """A generation request whose step attends ``kv_len`` keys (its own token included)."""
    return SimpleNamespace(
        py_request_id=next(_request_ids),
        py_draft_tokens=[],
        max_beam_num_tokens=kv_len,
        is_dummy=dummy,
        is_cuda_graph_dummy=False,
        py_batch_idx=None,
    )


def _context_row(tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=next(_request_ids),
        is_last_context_chunk=True,
        context_chunk_size=tokens,
        encoder_output_len=0,
        py_skip_cross_kv_projection=False,
    )


def _batch(*, decode=(), context=()) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.reset_context_requests([_context_row(n) for n in context])
    batch.generation_requests = [
        row if isinstance(row, SimpleNamespace) else _decode_row(row) for row in decode
    ]
    return batch


class _KvPool:
    """The KV manager surface the warmup's dummy requests read: ``tokens`` free tokens in pages of
    32, and dummies recorded as the token counts they were asked for (a generation dummy with no
    count holds one token, as the engine's managers default)."""

    tokens_per_block = _TOKENS_PER_BLOCK
    num_extra_kv_tokens = 0

    def __init__(self, tokens: int, max_seq_len: int):
        self.tokens = tokens
        self.max_seq_len = max_seq_len
        self.calls = []

    def get_num_available_tokens(self, *, token_num_upper_bound, max_num_draft_tokens=0, **_):
        return min(token_num_upper_bound, self.tokens)

    def get_num_free_blocks(self) -> int:
        return self.tokens // self.tokens_per_block

    def add_dummy_requests(self, request_ids, token_nums=None, is_gen=False, **kwargs):
        self.calls.append(
            dict(request_ids=request_ids, token_nums=token_nums, is_gen=is_gen, **kwargs)
        )
        if token_nums is None:
            token_nums = [1 + kwargs.get("max_num_draft_tokens", 0)] * len(request_ids)
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
        cuda_graph_batch_sizes=graphs.batch_sizes,
        max_beam_width=1,
        spec_config=None,
        use_mrope=False,
        max_draft_len=0,
        original_max_draft_len=0,
        max_total_draft_tokens=0,
        max_draft_loop_tokens=0,
    )


def _estimation_creator(resolved) -> KvCacheCreator:
    """The KV-cache creator's state its estimation workload reads, for this engine: the model
    engine's sizes after ``_init_model_capacity``, the V2 manager, no speculation."""
    creator = KvCacheCreator.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=resolved.model_config),
        use_mrope=False,
        batch_size=resolved.max_batch_size,
        _max_cuda_graph_batch_size=resolved.llm_args.cuda_graph_config.max_batch_size,
        max_seq_len=resolved.max_seq_len,
    )
    creator._max_num_tokens = resolved.max_num_tokens
    creator._max_beam_width = 1
    creator._mapping = resolved.mapping
    creator._net_max_seq_len = resolved.max_seq_len
    creator._dummy_reqs = None
    creator._speculative_config = None
    creator._llm_args = resolved.llm_args
    creator._tokens_per_block = resolved.llm_args.kv_cache_config.tokens_per_block
    creator._kv_cache_config = resolved.llm_args.kv_cache_config
    creator._is_kv_cache_manager_v2 = issubclass(
        get_kv_cache_manager_cls(resolved.model_config, creator._kv_cache_config), KVCacheManagerV2
    )
    return creator


# The estimation's KV quota at the 1M config, per rank: the CUDA-graph capture's longest row
# (ceil((1048576 + 1) / 32) blocks) beside the 15 other rows of the largest captured batch.
_ESTIMATION_BLOCKS = -(-(_MAX_SEQ_LEN + 1) // _TOKENS_PER_BLOCK) + 16 - 1
_ESTIMATION_TOKENS = _ESTIMATION_BLOCKS * _TOKENS_PER_BLOCK


# ------------------------------------------------------- (1) topology, limits, quantization


def test_dep4_topology_and_per_rank_limits(dep4_1m):
    """Four ranks, tensor_parallel_size 4 spent on attention data parallelism (each rank holds
    every head; attention TP is 1), the routed experts expert-parallel four ways and not split
    along their width, one pipeline stage. lm_head is not vocab-parallel under attention DP.
    max_batch_size and max_num_tokens are each rank's own: 16 requests and 32768 tokens per rank,
    64 requests in flight over the four. max_seq_len is the checkpoint's 1048576, chunked prefill
    is on (a step carries at most one 32768-token chunk per rank) and KV block reuse is off. The
    accuracy gate's options and the boot gate's quickstart argv agree on all of it."""
    resolved = dep4_1m
    llm_args, mapping = resolved.llm_args, resolved.mapping
    parallel = llm_args.parallel_config
    assert (parallel.tp_size, parallel.moe_ep_size, parallel.pp_size) == (4, 4, 1)
    assert llm_args.enable_attention_dp is True and parallel.enable_attention_dp is True

    assert (mapping.world_size, mapping.tp_size, mapping.pp_size, mapping.cp_size) == (4, 4, 1, 1)
    assert (mapping.moe_ep_size, mapping.moe_tp_size) == (4, 1)
    assert mapping.enable_attention_dp is True and mapping.dp_size == 4
    assert mapping.tp_size // mapping.dp_size == 1
    # The flag, not attn_tp_size, says attention is replicated: attn_tp_size still reads 4.
    assert mapping.attn_tp_size == 4
    assert mapping.enable_lm_head_tp_in_adp is False
    # Rank r holds the expert window [32 r, 32 r + 32).
    for rank in range(_RANKS):
        rank_mapping = _rank_mapping(mapping, rank)
        assert (rank_mapping.tp_rank, rank_mapping.moe_ep_rank, rank_mapping.moe_tp_rank) == (
            rank,
            rank,
            0,
        )
    assert resolved.model_config.pretrained_config.n_routed_experts // mapping.moe_ep_size == 32

    assert (resolved.max_batch_size, resolved.max_num_tokens, resolved.max_seq_len) == (
        16,
        32768,
        1048576,
    )
    assert (resolved.model_config.max_num_tokens, resolved.model_config.max_seq_len) == (
        32768,
        1048576,
    )
    assert llm_args.enable_chunked_prefill is True
    assert llm_args.kv_cache_config.enable_block_reuse is False
    assert llm_args.kv_cache_config.dtype == "fp8"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is True
    assert llm_args.disable_overlap_scheduler is False
    # PyExecutor's per-rank cap on active requests under attention DP.
    cap = derive_attention_dp_per_rank_request_cap(
        base_cap=mapping.pp_size * resolved.max_batch_size,
        max_num_tokens=resolved.max_num_tokens,
        max_total_draft_tokens=0,
    )
    assert cap == 16 and cap * mapping.dp_size == 64
    # No attention-DP balancing or KV-aware routing: the default router places requests (the
    # quickstart passes an AttentionDpConfig with balancing off, core.yaml none).
    router = ADPRouter.create(
        dist=None, has_seq_slot_headroom=False, attention_dp_config=llm_args.attention_dp_config
    )
    assert type(router) is DefaultADPRouter


def test_max_seq_len_is_not_capped(dep4):
    """Nothing lowers max_seq_len below 1048576. The checkpoint's max_position_embeddings is
    1048576 and its rope is YaRN (factor 2048 over 512 original positions), which the engine's
    inference does not scale, so the model engine's ``_init_model_capacity``, reading the
    modeling_v2 shell's ``infer_max_seq_len``, keeps 1048576 and max_num_tokens 32768. The target
    builds its rope tables at ``model_config.max_seq_len`` rows in ``post_load_weights``: 1048576
    rows, before the KV pool is sized."""
    cfg = dep4.model_config.pretrained_config
    assert cfg.max_position_embeddings == _MAX_SEQ_LEN
    assert (cfg.rope_scaling["rope_type"], cfg.rope_scaling["factor"]) == ("yarn", 2048.0)
    assert cfg.rope_scaling["original_max_position_embeddings"] == 512

    shell = SimpleNamespace(config=cfg)
    shell.infer_max_seq_len = MethodType(DecoderModelForCausalLM.infer_max_seq_len, shell)
    engine = SimpleNamespace(
        model=shell,
        max_seq_len=dep4.max_seq_len,
        max_num_tokens=dep4.max_num_tokens,
        batch_size=dep4.max_batch_size,
    )
    for name in ("_init_max_seq_len", "_init_max_num_tokens", "_init_model_capacity"):
        setattr(engine, name, MethodType(getattr(PyTorchModelEngine, name), engine))
    assert shell.infer_max_seq_len() == _MAX_SEQ_LEN
    engine._init_model_capacity()
    assert (engine.max_seq_len, engine.max_num_tokens) == (_MAX_SEQ_LEN, _MAX_NUM_TOKENS)
    assert dep4.model_config.max_seq_len == _MAX_SEQ_LEN


def test_quantization_is_the_anchor_key(dep4_1m):
    """At 1M as at 32K and at tep4: MIXED_PRECISION with the fp8 KV cache, the GSM8K anchor row's
    key, and the router handed params.json's quantization_config with its 475 quantized_layers
    intact."""
    resolved = dep4_1m
    assert resolved.prequantized is True
    for quant in (resolved.llm_args.quant_config, resolved.model_config.quant_config):
        assert quant.quant_algo == QuantAlgo.MIXED_PRECISION
        assert quant.kv_cache_quant_algo == QuantAlgo.FP8
        assert quant.group_size == 16
    assert resolved.llm_args.kv_cache_config.dtype == "fp8"
    assert _anchor(resolved.llm_args) == 91.926

    on_disk = json.loads((resolved.path / "params.json").read_text())["quantization_config"]
    assert len(resolved.routed) == 1
    seen = resolved.routed[0]
    assert seen.quantization_config == on_disk and len(on_disk["quantized_layers"]) == 475
    assert resolved.model_config.pretrained_config.quantization_config == on_disk
    assert seen.pretrained_config is resolved.model_config.pretrained_config
    assert resolved.model_config.modeling_v2_target is _Routed
    assert seen.compute_capability == _SM103 and seen.spec_config is None
    assert seen.mapping.enable_attention_dp is True
    assert (seen.mapping.world_size, seen.mapping.tp_size, seen.mapping.moe_ep_size) == (4, 4, 4)


def test_core_yaml_and_quickstart_build_the_same_engine(dep4, dep4_quickstart):
    """The boot gate's quickstart argv and the accuracy gate's core.yaml give the same mapping,
    model config, limits and CUDA graphs. They differ in two settings that do not reach the
    forward: the KV cache's free_gpu_memory_fraction (quickstart's default 0.9 against 0.85) and
    the attention-DP config (quickstart passes one with balancing off, core.yaml none); the
    quickstart's mamba cache dtype default is inert on this model."""
    assert dep4.mapping.to_dict() == dep4_quickstart.mapping.to_dict()
    core, quick = _summary(dep4), _summary(dep4_quickstart)
    assert {key for key in core if core[key] != quick[key]} == {"kv_cache"}
    kv_core, kv_quick = core["kv_cache"], quick["kv_cache"]
    assert {key for key in kv_core if kv_core[key] != kv_quick[key]} == {
        "free_gpu_memory_fraction",
        "mamba_ssm_cache_dtype",
    }
    assert (kv_core["free_gpu_memory_fraction"], kv_quick["free_gpu_memory_fraction"]) == (
        0.85,
        0.9,
    )
    assert dep4.llm_args.attention_dp_config is None
    quick_adp = dep4_quickstart.llm_args.attention_dp_config
    assert quick_adp.enable_balance is False
    assert quick_adp.enable_kv_cache_aware_routing is False
    for resolved in (dep4, dep4_quickstart):
        kv_cache_config = resolved.llm_args.kv_cache_config
        manager = get_kv_cache_manager_cls(resolved.model_config, kv_cache_config)
        assert manager is MistralLarge4CacheManager


def _manager_arguments(resolved, dtype) -> dict:
    """What ``MistralLarge4CacheManager.__init__`` (attention/backends/sparse/mistral_large4/
    cache_manager.py) hands the DSA V2 manager it builds on, given what ``_create_kv_cache_manager``
    (pyexecutor/_util.py) passes it for rank 0 of this engine: the LLM API's kv_cache_config, one
    576-wide MLA head on each of the 72 layers, the latent ``dtype``. The base is not built (it
    allocates the pool on the device): its constructor records its arguments and stops."""

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
                num_layers=model_config.pretrained_config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=576,
                tokens_per_block=_TOKENS_PER_BLOCK,
                max_seq_len=resolved.max_seq_len,
                max_batch_size=resolved.max_batch_size,
                mapping=_rank_mapping(resolved.mapping, 0),
                dtype=dtype,
                spec_config=None,
                layer_mask=None,
                max_num_tokens=resolved.max_num_tokens,
                max_beam_width=1,
                sparse_attention_config=model_config.sparse_attention_config,
                pretrained_config=model_config.pretrained_config,
            )
    return seen


def test_kv_cache_dtype_auto_and_fp8_build_the_fp8_ds_mla_pool(dep4, dep4_quickstart, dep4_ci):
    """kv_cache_config.dtype fp8 (the gates: core.yaml's and the quickstart's --kv_cache_dtype=fp8)
    and auto (the CI class: the checkpoint's fp8_kv_cache) build the same cache. Each resolves the
    latent to FP8 (``kv_cache_quant_algo``, so ``_create_kv_cache_manager`` hands the manager
    ``DataType.FP8``, ``quant_mode.has_fp8_kv_cache()``) on the same GSM8K anchor row; the sparse
    registry picks ``MistralLarge4CacheManager``, whose ``latent_layout`` is fp8_ds_mla, at 47136
    bytes a token on every rank; and the manager builds its base over all 72 layers, a 656-byte
    head on the 63 KV owners only, as FP8, with the dtype it was asked for. A latent dtype the
    layout cannot hold, bf16, is refused at construction. The CI class's engine is the gates'
    in everything else but where the FP8 KV algo is recorded: fp8 writes it onto each of the 475
    per-layer quant configs as well, auto leaves it on the global one, which the latent's dtype is
    read from."""
    for resolved, requested in ((dep4, "fp8"), (dep4_quickstart, "fp8"), (dep4_ci, "auto")):
        model_config, llm_args = resolved.model_config, resolved.llm_args
        assert llm_args.kv_cache_config.dtype == requested
        for quant in (llm_args.quant_config, model_config.quant_config):
            assert quant.kv_cache_quant_algo == QuantAlgo.FP8
        assert model_config.quant_config.quant_mode.has_fp8_kv_cache()
        assert _anchor(llm_args) == 91.926
        manager = get_kv_cache_manager_cls(model_config, llm_args.kv_cache_config)
        assert manager is MistralLarge4CacheManager and manager.latent_layout == "fp8_ds_mla"
        assert issubclass(manager, KVCacheManagerV2)
        for rank in range(_RANKS):
            cost = _kv_cost(resolved, rank)
            assert (cost.slope, cost.intercept) == (47136, 0)

        sources = model_config.pretrained_config.kv_source_layers
        base = _manager_arguments(resolved, DataType.FP8)
        assert (base["num_layers"], base["head_dim"]) == (72, 656)
        assert base["num_kv_heads"] == [1 if s == i else None for i, s in enumerate(sources)]
        assert base["num_kv_heads"].count(1) == 63
        assert base["dtype"] == DataType.FP8
        assert base["layer_mask"] is None
        assert base["kv_cache_config"].dtype == requested
        assert (base["tokens_per_block"], base["max_seq_len"]) == (_TOKENS_PER_BLOCK, _MAX_SEQ_LEN)
        with pytest.raises(ValueError, match="fp8_ds_mla"):
            _manager_arguments(resolved, DataType.BF16)

    gate, ci = _summary(dep4), _summary(dep4_ci)
    assert {key for key in gate if gate[key] != ci[key]} == {"kv_cache", "quant", "layers"}
    for resolved, layer_algo in ((dep4, QuantAlgo.FP8), (dep4_ci, None)):
        layers = resolved.model_config.quant_config_dict
        assert len(layers) == 475
        assert all(quant.kv_cache_quant_algo == layer_algo for quant in layers.values())
    assert {key for key in gate["kv_cache"] if gate["kv_cache"][key] != ci["kv_cache"][key]} == {
        "dtype"
    }
    assert dep4.mapping.to_dict() == dep4_ci.mapping.to_dict()


# ------------------------------------------------------- (2) admitting a 1M prompt


def _check_arguments(llm_args, prompt_len: int) -> None:
    """``BaseLLM._check_arguments`` on an LLM holding ``llm_args``, for a plain request."""
    llm = SimpleNamespace(args=llm_args)
    llm._check_one_model_speculative_sampling = MethodType(
        BaseLLM._check_one_model_speculative_sampling, llm
    )
    BaseLLM._check_arguments(llm, prompt_len, SamplingParams(max_tokens=8), is_gen_only=False)


def _deduced_max_tokens(resolved, prompt_len: int, max_tokens: int | None) -> int:
    """The ``max_tokens`` the worker's ``_enqueue_request`` gives the executor request, for a
    ``prompt_len``-token prompt asking ``max_tokens`` (None: as many as fit)."""
    enqueued = []
    worker = SimpleNamespace(
        _lora_manager=None,
        _is_pytorch_backend=True,
        mapping=_rank_mapping(resolved.mapping, 0),
        max_seq_len=resolved.max_seq_len,
        engine=SimpleNamespace(enqueue_request=lambda request, **_: enqueued.append(request) or 1),
    )
    request = GenerationRequest(
        [7] * prompt_len, sampling_params=SamplingParams(max_tokens=max_tokens, end_id=2, pad_id=2)
    )
    request.id = 1
    BaseWorker._enqueue_request(worker, request)
    (executor_request,) = enqueued
    return executor_request.max_tokens


def test_a_1m_prompt_passes_the_front_end(dep4, dep4_32k):
    """``_check_arguments`` refuses a prompt longer than max_num_tokens only without chunked
    prefill: at the 1M config a 1,040,000-token prompt passes (no length is refused there),
    while the delivered unchunked engine refuses one token past 32768."""
    for prompt_len in (32769, 1_040_000, _MAX_SEQ_LEN - 1, _MAX_SEQ_LEN):
        _check_arguments(dep4.llm_args, prompt_len)

    _check_arguments(dep4_32k.llm_args, _MAX_NUM_TOKENS)
    with pytest.raises(RequestError, match=r"should not exceed max_num_tokens \(32768\)"):
        _check_arguments(dep4_32k.llm_args, _MAX_NUM_TOKENS + 1)


def test_the_worker_refuses_a_prompt_of_max_seq_len_tokens(dep4):
    """The length the engine refuses is max_seq_len against the prompt alone: the worker deduces
    ``max_tokens`` as ``max_seq_len - prompt_len`` and refuses the request only when that is not
    positive, so a 1048576-token prompt is refused with the context-length message and a
    1048575-token prompt generates one token. A prompt plus max_tokens past max_seq_len is not
    refused: max_tokens is cut to what fits (a 1,040,000-token prompt gets at most 8576)."""
    assert _deduced_max_tokens(dep4, 1_040_000, None) == 8576
    assert _deduced_max_tokens(dep4, 1_040_000, 64) == 64
    assert _deduced_max_tokens(dep4, 1_040_000, 10_000) == 8576
    assert _deduced_max_tokens(dep4, _MAX_SEQ_LEN - 1, None) == 1
    with pytest.raises(RequestError) as refused:
        _deduced_max_tokens(dep4, _MAX_SEQ_LEN, 1)
    assert is_context_length_exceeded_message(str(refused.value))
    assert "maximum context length is 1048576 tokens" in str(refused.value)
    assert "resulted in 1048576 tokens" in str(refused.value)


# ------------------------------------------------------- (3) chunked prefill under MLA and DSA


def test_chunked_prefill_turns_on_context_mla_with_cached_kv(dep4, dep4_32k):
    """``build_attention_metadata`` turns ``enable_context_mla_with_cached_kv`` on for MLA when
    the runtime features carry chunked prefill (or KV reuse): on at the 1M config, off at the
    delivered 32K one. The DSA metadata is sized per rank (16 sequences, 32768 tokens) and its
    params keep the indexer's own chunk size 32768, the TP query-split threshold 8192, top-k
    2048, the short-sequence indexer skip and the shared-indexer layers."""
    for resolved, cached_kv in ((dep4, True), (dep4_32k, False)):
        features = _runtime_features(resolved)
        assert (features.chunked_prefill, features.cache_reuse) == (cached_kv, False)
        assert features.chunk_size == _MAX_NUM_TOKENS
        kwargs = _attention_metadata_kwargs(resolved)
        assert kwargs.enable_context_mla_with_cached_kv is cached_kv
        assert (kwargs.max_num_requests, kwargs.max_num_sequences) == (16, 16)
        assert kwargs.max_num_tokens == _MAX_NUM_TOKENS
        params = kwargs.sparse_metadata_params
        assert (params.indexer_max_chunk_size, params.q_split_threshold) == (32768, 8192)
        assert (params.max_sparse_topk, params.index_head_dim) == (2048, 128)
        assert params.enable_indexer_skip is True and params.has_shared_indexer_layers is True

    sparse_params = dep4.model_config.sparse_attention_config.to_sparse_params(
        pretrained_config=dep4.model_config.pretrained_config
    )
    backend = get_attention_backend(dep4.llm_args.attn_backend, sparse_params=sparse_params)
    assert backend.Metadata is DSAtrtllmAttentionMetadata


def test_attention_dp_leaves_the_indexer_queries_unsplit(dep4, tep4):
    """``_split_prefill_queries`` splits a step's context queries across TP ranks from 8192 of
    them on, but never under attention DP: every dep4 rank computes all 32768 of its own, while a
    tep4 rank takes a quarter."""
    for rank in range(_RANKS):
        assert _split_prefill_queries(_MAX_NUM_TOKENS, _rank_mapping(dep4.mapping, rank), 8192) == (
            0,
            _MAX_NUM_TOKENS,
            None,
        )
        assert _split_prefill_queries(_MAX_NUM_TOKENS, _rank_mapping(tep4.mapping, rank), 8192) == (
            8192 * rank,
            8192 * (rank + 1),
            [8192] * _RANKS,
        )


def _indexer_prefill_step(resolved, contexts, decode_kv_lens=(), rank: int = 0):
    """``Indexer.prepare_for_chunked_prefill`` on one dep4 rank's step: ``contexts`` lists each
    context request's (cached prefix, new tokens) and ``decode_kv_lens`` each generation row's KV
    length. The metadata holds what that method and its gather mappings read; its block table
    gives request i the pages i * width + j. Pinned host memory and the device copy of the gather
    mappings are skipped: there is no device here."""
    width = _v2_max_blocks_per_seq(resolved.max_seq_len)
    num_contexts, num_generations = len(contexts), len(decode_kv_lens)
    seq_lens = [new for _, new in contexts] + [1] * num_generations
    cached = [prefix for prefix, _ in contexts] + [kv_len - 1 for kv_len in decode_kv_lens]
    cached_indptr = [0, *itertools.accumulate(prefix for prefix, _ in contexts)]
    kv_indptr = [0, *itertools.accumulate(prefix + new for prefix, new in contexts)]
    num_seqs = num_contexts + num_generations
    md = SimpleNamespace(
        mapping=_rank_mapping(resolved.mapping, rank),
        enable_context_mla_with_cached_kv=True,
        runtime_features=_runtime_features(resolved),
        sparse_metadata_params=_attention_metadata_kwargs(resolved, rank).sparse_metadata_params,
        compress_ratios=[1],
        max_num_tokens=resolved.max_num_tokens,
        kv_cache_manager=SimpleNamespace(
            max_seq_len=resolved.max_seq_len, index_head_dim=128, tokens_per_block=32
        ),
        indexer_head_dim=128,
        indexer_quant_block_size=128,
        _tokens_per_block=_TOKENS_PER_BLOCK,
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=sum(new for _, new in contexts),
        request_ids=list(range(num_seqs)),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=cached),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        host_ctx_cached_token_indptr=torch.tensor(cached_indptr, dtype=torch.int64),
        host_ctx_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int64),
        cu_seqlen_ks=torch.empty(resolved.max_num_tokens, dtype=torch.int32),
        skip_indexer_for_ctx_reqs=False,
        slot_mapping_fp8=torch.empty(resolved.max_num_tokens, dtype=torch.int64),
        slot_mapping_scale=torch.empty(resolved.max_num_tokens, dtype=torch.int64),
        host_indexer_k_cache_block_offsets=torch.arange(
            resolved.max_batch_size * width, dtype=torch.int32
        ).reshape(resolved.max_batch_size, width),
        indexer_prefill_q_split="unset",
        indexer_prefill_chunks=None,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dsa_indexer, "prefer_pinned", lambda: False)
        mp.setattr(dsa_indexer, "maybe_pin_memory", lambda tensor: tensor)
        mp.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
        Indexer.prepare_for_chunked_prefill(md, Indexer.build_indexer_params(md))
    return md


def _chunk_ranges(md) -> list[tuple[int, int, int, int]]:
    return [
        (chunk.token_start, chunk.token_end, chunk.k_token_start, chunk.k_token_end)
        for chunk in md.indexer_prefill_chunks
    ]


def _indexer_slot(md, request: int, position: int) -> tuple[int, int]:
    """The indexer key pool's byte offsets of ``position`` in ``request``: 128 fp8 codes then one
    fp32 scale per token, pages of 32 tokens, page stride 32 * 132 bytes."""
    page = int(md.host_indexer_k_cache_block_offsets[request, position // _TOKENS_PER_BLOCK])
    base = page * _TOKENS_PER_BLOCK * (128 + 4)
    offset = position % _TOKENS_PER_BLOCK
    return base + offset * 128, base + _TOKENS_PER_BLOCK * 128 + offset * 4


def test_indexer_prefill_chunks_over_a_1m_prefix(dep4):
    """What ``Indexer.prepare_for_chunked_prefill`` plans on a dep4 rank under chunked prefill,
    as (token_start, token_end, k_token_start, k_token_end) per chunk. The planner is
    ``plan_mla_chunked_prefill_chunks`` over each request's whole KV (cached + new), packing
    requests while their (new tokens) x (KV) stays within ``_packed_logits_elem_budget``: 1 << 28
    here, the packed cap, below the 1 << 31 single-request tile and 32768 x 1048576. The query
    split stays None on every step. The target re-tiles each chunk's queries at the logits budget
    it takes from this engine (``_prefill_logits_budget(32768, 1048576)``, the ceiling 2**26,
    ``test_target_sizes_from_the_engines_metadata``) against the chunk's n_k = k_token_end -
    k_token_start gathered keys, or, for a chunk past max_seq_len keys (d), against each packed
    request's own (``_prefill_blocks``).

    (a) a 1M request's last chunk, 32768 new tokens over 1048576 - 32768 cached: one chunk,
    n_k 1048576; its rows attend keys [0, 1015809) to [0, 1048576).
    (b) its first chunk, 32768 over no prefix: one chunk, n_k 32768.
    (c) a mixed step, a 16384-token chunk over a 500000-token prefix with 15 decode rows behind
    it: one chunk of the context rows, n_k 516384; the decode rows are not planned here.
    (d) two requests' 5-token last chunks over 983040-token prefixes on one rank: packed into one
    chunk (10 x 1966090 is under the budget), so that indexer call sees n_k 1966090, more than
    max_seq_len; a packed chunk's n_k is bounded only by the budget over its new tokens and by
    the rank's KV pool.

    The key gather mappings (``slot_mapping_fp8_fullkv`` / ``slot_mapping_scale_fullkv``) are
    int64 over every context key of the step, cached and new, in request order: 1048576 long in
    (a). With no cached prefix (b) they are the step's own slot mappings."""
    budget_md = SimpleNamespace(
        max_num_tokens=dep4.max_num_tokens,
        kv_cache_manager=SimpleNamespace(max_seq_len=dep4.max_seq_len),
    )
    assert dsa_indexer._INDEXER_PACKED_LOGITS_ELEM_BUDGET == 1 << 28
    assert dsa_indexer._INDEXER_MQA_LOGITS_ELEM_BUDGET == 1 << 31
    assert _packed_logits_elem_budget(budget_md) == 1 << 28

    prefix = _MAX_SEQ_LEN - _MAX_NUM_TOKENS
    last = _indexer_prefill_step(dep4, [(prefix, _MAX_NUM_TOKENS)])
    assert last.indexer_prefill_q_split is None
    assert _chunk_ranges(last) == [(0, _MAX_NUM_TOKENS, 0, _MAX_SEQ_LEN)]
    (chunk,) = last.indexer_prefill_chunks
    assert torch.equal(chunk.cu_seqlen_ks, torch.zeros(_MAX_NUM_TOKENS, dtype=torch.int32))
    assert torch.equal(
        chunk.cu_seqlen_ke, torch.arange(prefix + 1, _MAX_SEQ_LEN + 1, dtype=torch.int32)
    )
    for mapping in (last.slot_mapping_fp8_fullkv, last.slot_mapping_scale_fullkv):
        assert mapping.dtype == torch.int64 and tuple(mapping.shape) == (_MAX_SEQ_LEN,)
    for position in (0, 31, 32, prefix, _MAX_SEQ_LEN - 1):
        assert (
            int(last.slot_mapping_fp8_fullkv[position]),
            int(last.slot_mapping_scale_fullkv[position]),
        ) == _indexer_slot(last, 0, position)

    first = _indexer_prefill_step(dep4, [(0, _MAX_NUM_TOKENS)])
    assert first.indexer_prefill_q_split is None
    assert _chunk_ranges(first) == [(0, _MAX_NUM_TOKENS, 0, _MAX_NUM_TOKENS)]
    assert first.slot_mapping_fp8_fullkv is first.slot_mapping_fp8
    assert first.slot_mapping_scale_fullkv is first.slot_mapping_scale

    mixed = _indexer_prefill_step(dep4, [(500_000, 16384)], decode_kv_lens=[100_000] * 15)
    assert mixed.indexer_prefill_q_split is None
    assert _chunk_ranges(mixed) == [(0, 16384, 0, 516_384)]
    assert tuple(mixed.slot_mapping_fp8_fullkv.shape) == (516_384,)
    assert 16384 + 15 <= dep4.max_num_tokens

    packed = _indexer_prefill_step(dep4, [(983_040, 5), (983_040, 5)])
    assert packed.indexer_prefill_q_split is None
    assert _chunk_ranges(packed) == [(0, 10, 0, 1_966_090)]
    assert 10 * 1_966_090 <= 1 << 28
    assert tuple(packed.slot_mapping_fp8_fullkv.shape) == (1_966_090,)
    assert int(packed.slot_mapping_fp8_fullkv[983_045]) == _indexer_slot(packed, 1, 0)[0]
    # The delivered per-rank pool (measured at 47280 bytes a token, the language model only) held
    # two such requests; the 1M boot's, with the vision tower on every rank, holds one.
    two = 2 * -(-(983_045 + 1) // _TOKENS_PER_BLOCK) * _TOKENS_PER_BLOCK
    assert _BOOT_POOL_TOKENS < two <= _DELIVERED_POOL_TOKENS


# ------------------------------------------------------- (4) the KV pool and its tables


def test_dsa_kv_tables_at_1m(dep4, dep4_32k):
    """The engine builds ``MistralLarge4CacheManager``, a V2 manager (``use_kv_cache_manager_v2``
    defaults on). Its block-table width is ceil((max_seq_len + 1) / 32), the one-token
    generation headroom included, padded to a multiple of 4: 32772 at 1048576 (1028 at 32768).
    The DSA metadata sizes ``block_table`` and ``indexer_k_cache_block_offsets`` [16, width] and
    ``kv_cache_block_offsets`` [pools, 16, 2, width] from it. The radix top-k's aux buffers are
    [16, 10, 2048] (one row per sequence without speculation) and ``topk_indices_buffer``, kept
    for the short-sequence indexer skip, is [32768, 2048]: neither grows with max_seq_len."""
    for resolved in (dep4, dep4_32k):
        kv_cache_config = resolved.llm_args.kv_cache_config
        manager = get_kv_cache_manager_cls(resolved.model_config, kv_cache_config)
        assert manager is MistralLarge4CacheManager and issubclass(manager, KVCacheManagerV2)
        assert kv_cache_config.tokens_per_block == _TOKENS_PER_BLOCK
    assert _get_generation_kv_capacity(None, is_draft=False) == (0, BASE_GENERATION_TOKEN_COUNT)
    assert BASE_GENERATION_TOKEN_COUNT == 1
    assert _v2_max_blocks_per_seq(dep4.max_seq_len) == 32772
    assert _v2_max_blocks_per_seq(dep4_32k.max_seq_len) == 1028

    kwargs = _attention_metadata_kwargs(dep4)
    params = kwargs.sparse_metadata_params
    # One radix row per sequence: `_radix_rows_per_sequence` is 1 without speculation.
    radix = (kwargs.max_num_sequences * 1, _MAX_RADIX_BLOCKS_PER_ROW, params.max_sparse_topk)
    assert radix == (16, 10, 2048)
    assert params.enable_indexer_skip is True
    assert (kwargs.max_num_tokens, params.max_sparse_topk) == (32768, 2048)


def test_kv_pool_has_no_rank_factor(dep4, dep4_32k, tep4):
    """Each rank holds the latent and indexer-key caches of its own requests only, and the cost of
    a token is the same on every rank, at 1M as at 32K and at tep4: an fp8_ds_mla latent of
    656 bytes (512 e4m3, four fp32 tile scales, 64 bf16 rope) on each of the 63 KV owners plus a
    128-byte fp8 indexer key and its fp32 scale on each of the 44 full-indexer layers, in pages
    of 32, with no per-request intercept. One 1,048,576-token request is 49,425,678,336 B
    (46.03 GiB) of a rank's pool, against 49,576,673,280 B (46.17 GiB) at the 576-byte latent
    of all 72 layers. The bytes of the delivered 32K boot's pool (2,004,896 tokens at 47280
    bytes) hold one such request at either token, not two, as does the 1M boot's pool on the
    native checkpoint (1,879,904 tokens). The configured batch is another
    figure at this config: 16 requests of max_seq_len would be 736.5 GiB of a rank's pool,
    which no rank holds, so the pool and not max_batch_size bounds how many 1M requests a rank
    holds at once; at 32K the same 16 requests of max_seq_len are 23.016 GiB."""
    model_config, kv_cache_config = dep4.model_config, dep4.llm_args.kv_cache_config
    manager = get_kv_cache_manager_cls(model_config, kv_cache_config)
    assert manager is MistralLarge4CacheManager
    assert manager is get_kv_cache_manager_cls(tep4.model_config, tep4.llm_args.kv_cache_config)
    assert kv_cache_config.tokens_per_block == _TOKENS_PER_BLOCK

    cfg, sparse = model_config.pretrained_config, model_config.sparse_attention_config
    latent = 656  # fp8_ds_mla: 512 e4m3 + 4 fp32 tile scales + 64 bf16 rope
    owners = sum(1 for i, s in enumerate(cfg.kv_source_layers) if s == i)
    indexer_key = sparse.index_head_dim + sparse.index_head_dim // 128 * 4
    full_indexer_layers = cfg.index_topk_pattern.count("F")
    per_token = latent * owners + indexer_key * full_indexer_layers
    assert (latent, owners, indexer_key, full_indexer_layers) == (656, 63, 132, 44)
    assert per_token == 47136

    for rank in range(_RANKS):
        cost = _kv_cost(dep4, rank)
        assert (cost.slope, cost.intercept) == (per_token, 0)
        assert cost == _kv_cost(dep4_32k, rank) == _kv_cost(tep4, rank)

    request = _kv_cost(dep4, 0).bytes_for_tokens(_MAX_SEQ_LEN)
    assert request == _MAX_SEQ_LEN * per_token == 49_425_678_336
    assert math.isclose(request / 2**30, 46.03, abs_tol=5e-3)
    assert _MAX_SEQ_LEN * _DELIVERED_BYTES_PER_TOKEN == 49_576_673_280
    # The measured pool, read through its bytes: one 1M request at either token, not two.
    delivered = _DELIVERED_POOL_TOKENS * _DELIVERED_BYTES_PER_TOKEN
    assert delivered // request == delivered // (_MAX_SEQ_LEN * _DELIVERED_BYTES_PER_TOKEN) == 1
    assert _BOOT_POOL_TOKENS * per_token // request == 1

    # The per-rank cost of the configured batch, 16 requests of max_seq_len: at 1M past any
    # rank's pool, at 32K the 23.016 GiB the 32K manager test pins.
    batch = _kv_cost(dep4, 0).bytes_for_tokens(dep4.max_batch_size * dep4.max_seq_len)
    assert batch == 16 * request and math.isclose(batch / 2**30, 736.5, abs_tol=5e-2)
    assert batch > delivered
    tokens_32k = dep4_32k.max_batch_size * dep4_32k.max_seq_len
    batch_32k = _kv_cost(dep4_32k, 0).bytes_for_tokens(tokens_32k)
    assert batch_32k == tokens_32k * per_token
    assert math.isclose(batch_32k / 2**30, 23.016, abs_tol=5e-4)


# ------------------------------------------------------- (5) boot, before the pool is sized


def test_kv_estimation_profiles_one_fresh_chunk_per_rank(dep4):
    """Before the real pool exists the engine runs a profiling pass. Its dummies are as long as a
    request may be (net max_seq_len - 1 = 1048575), cut to max_num_tokens: one 32768-token
    prompt, one copy per rank under attention DP, and the default router hands every rank one.
    MLA chunk-aware profiling does not apply to DSA, so each rank's profiled context step is that
    one fresh 32768-token prompt in a single chunk: no cached prefix and no KV length above
    32768. The estimation's own V2 quota is sized for the CUDA-graph capture's longest row
    instead: ceil((1048576 + 1) / 32) + 16 - 1 = 32784 blocks, 1,049,088 tokens per rank."""
    creator = _estimation_creator(dep4)
    assert creator._is_kv_cache_manager_v2 is True
    pretrained = dep4.model_config.pretrained_config
    assert _get_num_pool_groups_for_estimation(pretrained, _MAX_SEQ_LEN, None) == 1
    assert creator._get_token_num_for_estimation() == _ESTIMATION_TOKENS == 1_049_088
    assert creator._mla_chunked_profile_length is None
    lengths = [len(request.input_token_ids) for request in creator._dummy_reqs]
    assert lengths == [_MAX_NUM_TOKENS] * _RANKS

    router = ADPRouter.create(
        dist=None,
        has_seq_slot_headroom=False,
        attention_dp_config=dep4.llm_args.attention_dp_config,
    )
    assigned, expected = router.route_requests(
        [RankState(rank=rank) for rank in range(_RANKS)],
        [
            SimpleNamespace(request=SimpleNamespace(num_input_tokens=n, py_scheduling_params=None))
            for n in lengths
        ],
        max_num_active_requests=dep4.max_batch_size,
    )
    assert expected == 1
    for rank in range(_RANKS):
        assert [item.request.num_input_tokens for item in assigned[rank]] == [_MAX_NUM_TOKENS]


def test_no_context_mla_workspace_is_reserved(dep4, monkeypatch):
    """At sm_103 the fp8-KV context MLA of a DSA model takes the sparse path, which stages no
    K/V: the attention backend declares 0 bytes per attended token and no capacity-sized
    workspace, so the KV-capacity estimation subtracts no context-MLA reserve from the pool and
    the scheduler gets no attended-KV admission cap, chunked prefill or not. (The DeepSeek-V4
    logits reserve is for that algorithm only; this one is "dsa".) The target's own fp8_ds_mla
    context workspace (``PrefillTarget._ctx_workspace``) is not the backend's and is not reserved
    here either."""
    monkeypatch.setattr(trtllm_backend, "get_sm_version", lambda: 103)
    monkeypatch.delenv("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", raising=False)
    model_config = dep4.model_config
    assert model_config.sparse_attention_config.algorithm == "dsa"
    assert model_config.attn_backend == "TRTLLM"
    quant_mode = model_config.quant_config.quant_mode
    assert quant_mode.has_fp8_kv_cache() and not quant_mode.has_fp4_kv_cache()
    mapping = _rank_mapping(dep4.mapping, 0)
    bytes_per_token = get_attention_workspace_bytes_per_token(model_config, mapping)
    assert bytes_per_token == 0
    assert get_attention_workspace_is_chunked_prefill_bounded(model_config) is True
    assert (
        get_attention_workspace_fixed_bytes(
            model_config,
            mapping,
            max_batch_size=dep4.max_batch_size,
            max_num_tokens=dep4.max_num_tokens,
            max_seq_len=dep4.max_seq_len,
            enable_cuda_graph=True,
        )
        == 0
    )
    assert (
        get_mla_context_workspace_kv_len_cap(
            dep4.llm_args.kv_cache_config,
            dep4.max_batch_size,
            dep4.max_num_tokens,
            dep4.max_seq_len,
            dep4.llm_args.enable_chunked_prefill,
            workspace_is_chunked_prefill_bounded=True,
            # DSA takes no chunk-aware profiling; the requirement holds only for w > 0.
            chunked_workspace_profiled=False,
            require_chunked_workspace_profile=bytes_per_token > 0,
        )
        is None
    )


def test_target_sizes_from_the_engines_metadata(dep4, dep4_32k, monkeypatch):
    """What the dep4 target fixes on the first step (``_derive_first_step``) from the metadata
    this engine builds -- ``build_attention_metadata``'s max_num_requests, max_num_tokens and
    runtime features, which ``_create_py_executor`` builds from the LLM arguments (chunked
    prefill from enable_chunked_prefill, cache reuse from the KvCacheConfig's
    enable_block_reuse, off at both configs), and the KV cache manager's max_seq_len. At the 1M
    config, chunked prefill lets a step attend min(16, 32768) x 1048576 = 2**24 rows of context
    KV, so: the held context workspace at its ceiling, 2**19 rows (bf16 [2**19, 576], 576 MiB,
    allocated on the first context step -- the KV-capacity estimation's profiling forward -- and
    held, so the pool is sized with it); the indexer's logits budget at its ceiling, 2**26 (64
    query rows of 1048576 keys); and, under chunked prefill, expert calls of 32768 of the up to
    131072 gathered rows, the autotuner's bucket 32768. At the delivered 32K configuration
    (neither chunked prefill nor block reuse) a step attends at most its 32768 new rows: 65536
    rows, 2**24 and one call of 131072. Built on the meta device, with stand-ins for its kernel
    builds."""
    monkeypatch.delenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", raising=False)
    monkeypatch.setattr(
        M, "_rope_tables", lambda cfg, alpha, rows, dtype, device: SimpleNamespace(rows=rows)
    )
    monkeypatch.setattr(M, "build_ix_q_side", lambda dev, is_neox: None)
    monkeypatch.setattr(M, "build_ix_k_cache", lambda dev, is_neox, cache_contiguous: None)
    monkeypatch.setattr(
        M, "build_ix_topk_decode", lambda dev, next_n: SimpleNamespace(next_n=next_n)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    sources = list(dep4.model_config.pretrained_config.kv_source_layers)
    owners = [i for i, s in enumerate(sources) if s == i]
    assert (1 << 19) * 576 * 2 == 576 * 2**20
    for resolved, most, sizes in (
        (dep4, 1 << 24, (1 << 19, 1 << 26, 32768, 32768)),
        (dep4_32k, 32768, (65536, 1 << 24, 131072, 131072)),
    ):
        llm_args = resolved.llm_args
        kwargs = _attention_metadata_kwargs(resolved)
        features = kwargs.runtime_features
        assert features == _runtime_features(resolved)
        assert features.chunked_prefill is llm_args.enable_chunked_prefill is (resolved is dep4)
        assert features.cache_reuse is llm_args.kv_cache_config.enable_block_reuse is False
        md = SimpleNamespace(
            max_num_requests=kwargs.max_num_requests,
            max_num_tokens=kwargs.max_num_tokens,
            runtime_features=features,
            max_seq_len=resolved.max_seq_len,
            tokens_per_block=_TOKENS_PER_BLOCK,
            kv_cache_manager=SimpleNamespace(
                latent_layout=MistralLarge4CacheManager.latent_layout,
                kv_source_layers=sources,
                latent_pool_row=lambda i: owners.index(sources[i]),
                kv_owner_layers=owners,
            ),
        )
        assert M._most_ctx_kv(md) == most
        with torch.device("meta"):
            core = M.MistralLarge4Dep4ModelingV2(resolved.model_config)
        core.build_kernels(0)
        core._derive_first_step(md)
        got = (
            core.ctx_workspace_rows,
            core.prefill_logits_budget,
            core.expert_call_tokens,
            core.expert_tune_tokens,
        )
        assert got == sizes, (resolved.max_seq_len, got)


def test_warmup_runs_one_chunk_per_rank_and_captures_a_1m_decode_row(dep4):
    """The warmup's shapes under attention DP, on the estimation's 1,049,088-token pool (the
    warmup and the capture run in the profiling executor first, then again in the final one).
    Each rank proposes its shapes from its own free KV, the ranks keep the ones all proposed, and
    every rank enters each shape's forward. The max context shape is max_num_tokens of context
    on every rank at once, now one fresh 32768-token prompt (a prompt may be up to
    max_seq_len - 1 long), so the routed experts see 131072 gathered rows. Each decode graph is
    captured on a batch whose first row is max_seq_len - 1 = 1048575 tokens long (long family;
    its step attends 1048576 keys) or threshold - 2 = 2046 (short family), the other rows one
    token. So the 1M decode row runs in the profiling executor's capture, before the pool is
    sized, and the decode scorer's [16, 1048576] fp32 arena -- sized by max_seq_len, whatever
    the KV lengths -- is reserved there."""
    config = _runner_config(dep4)
    sparse = copy.deepcopy(dep4.model_config.sparse_attention_config)
    assert sparse.needs_separate_short_long_cuda_graphs() is True
    model = SimpleNamespace(model_config=dep4.model_config)
    group = _TpGroup()

    def rank_warmup(rank):
        resources = _Resources(_KvPool(_ESTIMATION_TOKENS, config.max_seq_len))
        warmup = DecoderWarmup.__new__(DecoderWarmup)
        warmup._config = config
        warmup.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        agreement = WarmupAgreement(group.dist(rank), _rank_mapping(dep4.mapping, rank))
        shapes = agreement.agree_shapes(warmup._get_full_general_warmup_requests(resources))
        factory = DummyRequestFactory(
            config=config,
            kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
            model=model,
            get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
            release_request_state=None,
        )
        max_context = max(tokens for tokens, gen in shapes if gen == 0)
        batch = factory.create_warmup_request(resources, max_context, 0)
        enters = agreement.should_run_warmup_batch(batch, max_context, "max context")
        prompts = [request.token_num for request in batch.context_requests]
        captures = {}
        for family, max_seq_len in (
            ("long", config.max_seq_len),
            ("short", sparse.seq_len_threshold - 1),
        ):
            capture = factory.create_cuda_graph_warmup_request(
                resources, config.cuda_graph_batch_sizes[-1], 0, max_seq_len
            )
            captures[family] = [request.token_num for request in capture.generation_requests]
        return shapes, enters, prompts, captures

    results = group.run(rank_warmup)
    for shapes, enters, prompts, captures in results:
        assert shapes == [(1, 0), (1, 1), (32768, 0), (16, 16), (2, 0)]
        assert enters is True
        assert prompts == [32768]
        assert captures == {"long": [1048575] + [1] * 15, "short": [2046] + [1] * 15}
    assert sum(sum(prompts) for _, _, prompts, _ in results) == 131072
    # The capture's batch fits the estimation quota: the longest row's pages and one per other row.
    assert -(-(1048575 + 1) // _TOKENS_PER_BLOCK) + 15 <= _ESTIMATION_BLOCKS


# ------------------------------------------------------- what the forward receives


def test_every_rank_is_handed_every_rank_token_count(dep4):
    """``attn_metadata.all_rank_num_tokens`` is set on every step of every rank, by the gather in
    the decoder's input preparation (eager, warmup and capture steps alike): each rank's own
    token count, in rank order, identical on every rank and NOT padded (no prefill CUDA graph
    backend is configured). An idle rank counts the one token of its generation dummy; a mixed
    step counts its chunk and its decode rows. Padding to the largest count before the experts'
    collectives is the forward's own job."""
    llm_args, model_config = dep4.llm_args, dep4.model_config
    sparse_params = model_config.sparse_attention_config.to_sparse_params(
        pretrained_config=model_config.pretrained_config
    )
    metadata_cls = get_attention_backend(
        llm_args.attn_backend, sparse_params=sparse_params
    ).Metadata
    assert metadata_cls is DSAtrtllmAttentionMetadata
    # DSA exchanges the one plain count; it has no per-stage counts of its own.
    assert metadata_cls.get_adp_token_counts is AttentionMetadata.get_adp_token_counts
    assert metadata_cls.set_adp_token_counts is AttentionMetadata.set_adp_token_counts
    assert llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.DISABLED

    group = _TpGroup()
    steps = {
        # every rank runs a whole 32768-token chunk, as the warmup's max shape does
        "four chunks": ([32768] * 4, [1] * 4),
        # a chunk with 15 decode rows behind it, an idle rank's dummy, a chunk, sixteen decode rows
        "unequal": ([16384 + 15, 1, 32768, 16], [1, 0, 1, 0]),
    }
    for counts, contexts in steps.values():

        def rank_step(rank, counts=counts, contexts=contexts):
            md = SimpleNamespace(
                num_tokens=counts[rank], padded_num_tokens=None, all_rank_num_tokens=None
            )
            md.get_adp_token_counts = MethodType(metadata_cls.get_adp_token_counts, md)
            md.set_adp_token_counts = MethodType(metadata_cls.set_adp_token_counts, md)
            dist = group.dist(rank)
            gathered = get_all_rank_num_tokens(
                md,
                enable_attention_dp=True,
                mapping=_rank_mapping(dep4.mapping, rank),
                dist=dist,
            )
            num_tokens = pad_input_tokens(
                md,
                counts[rank],
                contexts[rank],
                gathered,
                input_ids=torch.zeros(counts[rank], dtype=torch.int32),
                position_ids=torch.zeros(counts[rank], dtype=torch.int32),
                dist=dist,
                enable_attention_dp=True,
                prefill_cuda_graph_backend=llm_args.prefill_cuda_graph_backend,
                prefill_cuda_graph_num_tokens=[],
            )
            return num_tokens, md.all_rank_num_tokens, md.padded_num_tokens

        for rank, (num_tokens, all_rank, padded) in enumerate(group.run(rank_step)):
            assert num_tokens == counts[rank] and padded is None
            assert all_rank == counts

    # Four 32768-token chunks in one step: the experts see 131072 gathered rows, 524288 after
    # top-4, as at 32K.
    rows = sum(steps["four chunks"][0])
    assert rows == _RANKS * dep4.max_num_tokens == 131072
    assert rows * model_config.pretrained_config.num_experts_per_tok == 524288


def test_an_idle_rank_steps_with_a_one_token_generation_dummy(dep4):
    """When some rank has work and this one has none, the executor gives it one dummy before
    scheduling: request id ATTENTION_DP_DUMMY_REQUEST_ID, a generation request with the KV
    manager's default of one token (no draft tokens), its KV prepared like a real request's.
    Aggregated serving with the overlap scheduler takes this plain path: no KV-cache transceiver,
    and neither the scheduler-aware nor the forward-intent variant. The rank's step is then
    generation-only with one token, so it runs the decode target while its peers may prefill
    (one 1M prompt runs its whole prefill on one rank, the others stepping beside it). When no
    rank has work, no rank gets a dummy and none steps."""
    llm_args, mapping = dep4.llm_args, dep4.mapping
    assert llm_args.disable_overlap_scheduler is False
    assert should_enable_adp_dummy_fixes(mapping) is True
    model_type = dep4.model_config.pretrained_config.model_type
    assert not should_enable_scheduler_aware_adp_dummy(
        model_type, mapping, llm_args.disable_overlap_scheduler
    )
    assert not should_enable_non_overlap_adp_forward_intent(
        mapping, llm_args.disable_overlap_scheduler
    )

    def executor(expected_num_active_requests, active_requests):
        """The executor state `_pad_attention_dp_dummy_request` reads, with its own helpers."""
        executor = SimpleNamespace(
            enable_attention_dp=mapping.enable_attention_dp,
            _enable_remote_tail_adp=False,
            # No speculative decoding: no draft KV cache joins the target's reuse.
            enable_joint_kv_cache_reuse=False,
            _enable_adp_dummy_fixes=should_enable_adp_dummy_fixes(mapping),
            _enable_scheduler_aware_adp_dummy=False,
            _enable_non_overlap_adp_forward_intent=False,
            _benchmark_fill_phase_active=False,
            is_warmup=False,
            kv_cache_transceiver=None,
            # PyExecutor's initial dummy role; only disaggregated serving changes it.
            _adp_dummy_is_gen=True,
            max_total_draft_tokens=0,
            expected_num_active_requests=expected_num_active_requests,
            active_requests=list(active_requests),
            adp_router=SimpleNamespace(exclude_retiring_requests=False),
            kv_cache_manager=_KvPool(_DELIVERED_POOL_TOKENS, dep4.max_seq_len),
            resource_manager=_Resources(None),
        )
        for name in (
            "_count_schedulable_active_requests",
            "_should_skip_dummy_for_benchmark_disagg",
            "_has_adp_dummy_kv_capacity",
        ):
            setattr(executor, name, MethodType(getattr(PyExecutor, name), executor))
        PyExecutor._pad_attention_dp_dummy_request(executor)
        return executor

    idle = executor(expected_num_active_requests=1, active_requests=[])
    assert idle.kv_cache_manager.calls == [
        dict(
            request_ids=[ATTENTION_DP_DUMMY_REQUEST_ID],
            token_nums=None,
            is_gen=True,
            prepare_resource=True,
            max_num_draft_tokens=0,
            draft_kv_cache_manager=None,
        )
    ]
    (dummy,) = idle.active_requests
    assert dummy.is_attention_dp_dummy is True and dummy.is_gen is True
    assert (dummy.py_request_id, dummy.token_num) == (ATTENTION_DP_DUMMY_REQUEST_ID, 1)

    busy = executor(expected_num_active_requests=1, active_requests=[_decode_row(100)])
    assert busy.kv_cache_manager.calls == [] and len(busy.active_requests) == 1
    fleet_idle = executor(expected_num_active_requests=0, active_requests=[])
    assert fleet_idle.kv_cache_manager.calls == [] and fleet_idle.active_requests == []


def test_decode_graphs_replay_only_when_every_rank_decodes(dep4, monkeypatch):
    """The CUDA-graph rule under attention DP, on the engine's own runner: a step replays only
    when every rank's batch is generation-only. Then each rank pads to the largest batch across
    the ranks, rounded up to a captured size, so all replay the same batch size; an idle rank's
    one-row dummy is padded like any batch. When any rank has a context request (a chunk), no
    rank replays; a decode rank still pads its own batch to its own captured size and runs it
    eagerly. The ranks agree on eligibility and batch size only: DSA captures a short- and a
    long-sequence family at each size, and each rank picks its family from its own sequences, so
    ranks can replay different families in one step (a 1M sequence takes the long one)."""
    llm_args, model_config = dep4.llm_args, dep4.model_config
    graphs = llm_args.cuda_graph_config
    assert graphs.batch_sizes == [1, 2, 4, 8, 16]
    assert graphs.enable_padding is True and graphs.max_batch_size == 16
    # Asking for the families writes the threshold onto the config; ask a copy.
    sparse = copy.deepcopy(model_config.sparse_attention_config)
    assert sparse.needs_separate_short_long_cuda_graphs() is True
    threshold = sparse.seq_len_threshold
    assert threshold == sparse.index_topk == 2048

    # No device here: the static input buffers are not built.
    monkeypatch.setattr(CUDAGraphRunner, "_create_shared_static_tensors", lambda self: None)
    group = _TpGroup()
    captured = SimpleNamespace(is_cuda_graph=True)
    runners = []
    for rank in range(_RANKS):
        runner = CUDAGraphRunner(
            CUDAGraphRunnerConfig(
                use_cuda_graph=True,
                cuda_graph_padding_enabled=graphs.enable_padding,
                cuda_graph_batch_sizes=graphs.batch_sizes,
                max_cuda_graph_batch_size=graphs.max_batch_size,
                max_beam_width=1,
                max_num_tokens=dep4.max_num_tokens,
                spec_config=None,
                cuda_graph_mem_pool=None,
                use_mrope=False,
                original_max_draft_len=0,
                original_max_total_draft_tokens=0,
                enable_attention_dp=dep4.mapping.enable_attention_dp,
                is_encoder_decoder=False,
                batch_size=dep4.max_batch_size,
                mapping=_rank_mapping(dep4.mapping, rank),
                dist=group.dist(rank),
                kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
                sparse_attention_config=sparse,
            )
        )
        # What warmup leaves behind: the padding dummy (one generation token) and a graph per
        # captured size in each family, captured with a longest row of max_seq_len - 1 (long)
        # and threshold - 1 - 1 (short).
        runner.padding_dummy_requests[0] = _decode_row(1, dummy=True)
        for size in graphs.batch_sizes:
            for longest in (dep4.max_seq_len - 1, threshold - 2):
                key = runner.get_graph_key(_batch(decode=[longest] + [1] * (size - 1)))
                runner.graph_metadata[key] = {"attn_metadata": captured, "spec_metadata": None}
        runners.append(runner)
    assert len(runners[0].graph_metadata) == 2 * len(graphs.batch_sizes)

    def step(batches):
        def rank_step(rank):
            runner = runners[rank]
            with runner.pad_batch(batches[rank], None, 0) as padded:
                attn_metadata, _, key = runner.maybe_get_cuda_graph(
                    padded, enable_spec_decode=False, attn_metadata=None
                )
                return padded.batch_size, key, attn_metadata

        return group.run(rank_step)

    def idle():
        return _batch(decode=[_decode_row(1, dummy=True)])

    # Every rank decodes, short sequences, unequal batches: all pad to 16 and replay it.
    for sizes in ([3, None, 9, 16], [3, None, 9, 5]):
        batches = [idle() if n is None else _batch(decode=[100] * n) for n in sizes]
        for size, key, attn_metadata in step(batches):
            assert size == 16 and attn_metadata is captured
            assert (key.batch_size, key.draft_len, key.short_seq_len_mode) == (16, 0, True)

    # Small batches pad to the smallest captured size that holds the largest.
    batches = [_batch(decode=[100, 100]), idle(), _batch(decode=[100, 100]), idle()]
    assert [(size, key.batch_size) for size, key, _ in step(batches)] == [(2, 2)] * 4

    # Equal sizes, different families: the ranks holding a 30000- and a 1048575-token sequence
    # replay the long graph while their peers replay the short one.
    long_rows = {1: 30000, 2: dep4.max_seq_len - 1}
    batches = [_batch(decode=[long_rows.get(rank, 100)] * 16) for rank in range(_RANKS)]
    modes = [key.short_seq_len_mode for _, key, _ in step(batches)]
    assert modes == [True, False, False, True]

    # One rank runs a chunk: nobody replays, and a decode rank runs its own padded batch eagerly.
    batches = [_batch(context=[32768]), idle(), _batch(decode=[100] * 9), _batch(decode=[100] * 16)]
    results = step(batches)
    assert [size for size, _, _ in results] == [1, 1, 16, 16]
    assert all(key is None and attn_metadata is None for _, key, attn_metadata in results)


# ------------------------------------------------------- (6) the other configurations as before


def test_the_1m_config_moves_max_seq_len_and_chunked_prefill_only(dep4_ci, dep4_32k):
    """Against the delivered 32K dep4 configuration the 1M config changes exactly two LLM-API
    settings, max_seq_len (32768 -> 1048576) and enable_chunked_prefill (False -> True), and
    nothing else the engine resolves: same mapping, quantization, model config, DSA config,
    CUDA graphs and KV-cache config. Both as the CI classes pass them, the KV dtype at auto; the
    gates' fp8 differs from it in that field alone
    (``test_kv_cache_dtype_auto_and_fp8_build_the_fp8_ds_mla_pool``)."""
    fields_1m, fields_32k = (
        {name: repr(getattr(resolved.llm_args, name)) for name in TorchLlmArgs.model_fields}
        for resolved in (dep4_ci, dep4_32k)
    )
    assert {name for name in fields_1m if fields_1m[name] != fields_32k[name]} == {
        "max_seq_len",
        "enable_chunked_prefill",
    }
    summary_1m, summary_32k = _summary(dep4_ci), _summary(dep4_32k)
    assert {key for key in summary_1m if summary_1m[key] != summary_32k[key]} == {
        "sizes",
        "chunked_prefill",
    }
    assert (summary_1m["sizes"], summary_32k["sizes"]) == ((32768, 1048576, 16), (32768, 32768, 16))
    assert dep4_ci.mapping.to_dict() == dep4_32k.mapping.to_dict()


def test_tep4_and_tp4_route_as_before(tep4, tp4):
    """The NVFP4 checkpoint at tep4 and at tp4 resolves and routes as at base: attention DP off,
    unchunked at 32768, each claimed by its own target, each on the checkpoint's anchor and on
    the fp8_ds_mla pool (``MistralLarge4CacheManager``), the same 47136-byte token."""
    cases = (
        (tep4, (4, 1), "ModelingV2MistralLarge4Nvfp4Sm103Tep4"),
        (tp4, (1, 4), "ModelingV2MistralLarge4Nvfp4Sm103Tp4"),
    )
    for resolved, moe_split, target in cases:
        seen = resolved.routed[0]
        mapping = seen.mapping
        assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (4, 4, 1)
        assert (mapping.moe_ep_size, mapping.moe_tp_size) == moe_split
        assert mapping.enable_attention_dp is False
        assert routing.route(seen.pretrained_config, mapping, _SM103, None).__name__ == target
        assert resolved.llm_args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        assert resolved.llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
        assert _anchor(resolved.llm_args) == 91.926
        manager = get_kv_cache_manager_cls(resolved.model_config, resolved.llm_args.kv_cache_config)
        assert manager is MistralLarge4CacheManager
        assert (_kv_cost(resolved, 0).slope, _kv_cost(resolved, 0).intercept) == (47136, 0)
        assert resolved.llm_args.enable_chunked_prefill is False
        assert (resolved.max_num_tokens, resolved.max_seq_len, resolved.max_batch_size) == (
            32768,
            32768,
            16,
        )


def test_dep4_differs_from_tep4_in_attention_dp_only(dep4, dep4_32k, tep4):
    """The same checkpoint at the delivered dep4 configuration and at tep4 resolves to the same
    configuration -- quantization, per-layer quantization, model config, DSA config, limits, what
    the router reads -- except the mapping's attention-DP bit. Routing reads neither max_seq_len
    nor chunked prefill: the dep4 branch claims dep4 at 32K and at 1M alike, and the tep4 branch
    claims neither."""
    assert _summary(dep4_32k) == _summary(tep4)
    dep4_mapping, tep4_mapping = dep4_32k.mapping.to_dict(), tep4.mapping.to_dict()
    assert {k for k in dep4_mapping if dep4_mapping[k] != tep4_mapping[k]} == {
        "enable_attention_dp"
    }

    for resolved in (dep4_32k, dep4):
        seen = resolved.routed[0]
        try:
            target = routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
        except NoTarget:
            target = None
        assert target is not None
        assert target.__name__ == "ModelingV2MistralLarge4Nvfp4Sm103Dep4"
