# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B-NVFP4 at dep4: the engine's configuration, and the attention-DP contract
the dep4 forward is written against, at both worlds the target serves: four ranks on one GB300
node, and eight over two nodes of four (dep8).

No GPU and no weights. The checkpoint goes through the engine's own configuration path with the
LLM-API options of the core unit (four or eight ranks, attention data parallel, the routed
experts expert-parallel over every rank, 32768 tokens and 16 requests per rank, the latent KV
cache at its default dtype, the checkpoint's fp8_kv_cache: the fp8_ds_mla pool), in the
spellings the gates use:

* the accuracy and serve gates' LLM-API options (spec-dep4/options/core.yaml and
  spec-dep8/options/core.yaml), handed to ``TorchLlmArgs``;
* at eight ranks also the boot gate's command, parsed and turned into ``LLM`` keyword arguments
  by the tree's own ``examples/llm-api/quickstart_advanced.py`` (``parse_arguments`` and
  ``setup_llm``), as the gate harness builds its engine, plus the ``modeling_v2`` argument the
  harness adds.

Each then runs the LLM API front end (``ModelLoader._update_from_hf_quant_config``,
``llmapi/llm_utils.py``), which settles the key the accuracy harness looks its anchor up by, and
each rank's configuration pass (``_load_config_and_create_checkpoint_loader`` in
``py_executor_creator``, then the model loader's ``_load_and_validate_config``), which builds the
``ModelConfig`` and asks the modeling_v2 router for a target. The router's ``route`` is replaced
by a recorder, so what it is handed is observed rather than judged.

No spelling states ``gpus_per_node``: ``TorchLlmArgs`` takes it from
``torch.cuda.device_count()`` on rank 0, which on a GB300 node of this cluster is 4. The count is
set to 4 where the arguments are built rather than read from the test host, which may see no
device or a whole node. What the forward receives under attention DP -- the per-rank token counts
in ``attn_metadata.all_rank_num_tokens``, an idle rank's dummy request, the CUDA-graph padding and
replay rule, the warmup's shapes, the KV-cache estimation's context step from the request queue
to the ranks -- is pinned on the engine's own host code at each world, its ranks simulated in
one process where they exchange anything (each rank a thread, each collective a barrier); so is
the expert path's call size, which the core fixes on the first step from the world and the
engine's runtime features, and at eight ranks where the engine reads the node layout on its way
to a modeling_v2 target.

The checkpoint is Mistral-native, as published (params.json, tekken.json, consolidated
safetensors), and is loaded with checkpoint_format mistral_large_4: the gates' LLM-API options
state it and the boot gate's command passes --checkpoint_format=mistral_large_4. The checkpoint
at tep4 is checked to resolve and route as it did before this target, and at dep4 and at dep8
to route to it. The checkpoint is looked up under ``LLM_MODELS_ROOT`` (as
``Mistral-Large-4-1T-A52B-NVFP4`` or ``mistral-large-4-nvfp4``) by its params.json, and its
tests are skipped where it is absent.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import itertools
import json
import math
import queue
import sys
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import yaml
from test_common.llm_data import llm_models_root

from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.autotuner import AutoTuner, DistributedTuningStrategy
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from tensorrt_llm._torch.distributed.communicator import Distributed, MPIDist
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import maybe_create_moe_load_balancer
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor._util import (
    CacheCost,
    KvCacheCreator,
    get_kv_cache_manager_cls,
    should_enable_adp_dummy_fixes,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_scheduler_aware_adp_dummy,
)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.pyexecutor.engine.runners.common import (
    get_all_rank_num_tokens,
    pad_input_tokens,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_auto_host_tier_quota,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader as RankModelLoader
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import (
    derive_attention_dp_per_rank_request_cap,
    get_from_waiting_queue,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import ADPRouter, DefaultADPRouter
from tensorrt_llm._torch.pyexecutor.scheduler.rank_state import RankState
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm.executor import base_worker
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend, TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import ModelLoader as FrontEndModelLoader
from tensorrt_llm.mapping import Mapping, MpiTopology
from tensorrt_llm.models.modeling_utils import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
# The attention-DP worlds the target serves: four ranks on one node, eight over two.
_WORLDS = (4, 8)
# dep8's ranks: two nodes of four.
_DEP8_RANKS = 8
# Devices per GB300 node on this cluster: one node holds four ranks, two hold eight.
_GPUS_PER_NODE = 4
# max_num_tokens = max_seq_len per rank: the 32K serving config.
_TOKENS = 32768

# The topologies. dep4 and dep8 are the core units' (spec-dep4/options/core.yaml,
# spec-dep8/options/core.yaml); tep4 is where the checkpoint already serves.
_DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
_DEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=True)
_TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)

# Each world, and the fixtures of its engine's spellings: core.yaml's, and at eight ranks the
# boot gate's command.
_each_world = pytest.mark.parametrize("world", _WORLDS, ids=lambda w: f"world{w}")
_ENGINES = {4: ("dep4",), 8: ("dep8", "dep8_gate")}

# The dep8 boot gate's quickstart command after --model_dir: the checkpoint's Mistral-native
# format, the dep8 topology flags, the core unit's CUDA-graph capture list, and task.yaml's
# runtime flags.
_GATE_ARGV = (
    "--checkpoint_format=mistral_large_4",
    "--max_tokens=64",
    "--tp_size=8",
    "--moe_ep_size=8",
    "--enable_attention_dp",
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


def _options(topology: dict, tokens: int) -> dict:
    """The LLM-API options of a Mistral Large 4 accuracy gate on ``topology``, the checkpoint in its
    Mistral-native format, with the latent KV cache at its default (the checkpoint's
    fp8_kv_cache: the fp8_ds_mla pool). Under attention DP max_batch_size and max_num_tokens bind
    each rank."""
    return dict(
        checkpoint_format="mistral_large_4",
        **topology,
        max_batch_size=16,
        max_num_tokens=tokens,
        max_seq_len=tokens,
        enable_chunked_prefill=False,
        cuda_graph_config=dict(enable_padding=True, max_batch_size=16),
        kv_cache_config=dict(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
    )


_NVFP4_NAMES = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

# The GSM8K reference rows, and the key the harness reads them by.
_GSM8K_REFERENCES = (
    Path(__file__).resolve().parents[6] / "integration/defs/accuracy/references/gsm8k.yaml"
)
_MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"

_DEP4_TARGET = "ModelingV2MistralLarge4Nvfp4Sm103Dep4"


def _checkpoint_dir(names: tuple[str, ...]) -> Path:
    """The first of ``names`` under LLM_MODELS_ROOT holding a Mistral-native params.json."""
    root = llm_models_root()
    for name in names:
        if root is not None and (root / name / "params.json").is_file():
            return root / name
    pytest.skip(f"none of {names} is under LLM_MODELS_ROOT ({root})")


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


class _Routed:
    """What the recording `route` returns: no target is built here."""


def _resolve(checkpoint_dir: Path, llm_args: TorchLlmArgs) -> SimpleNamespace:
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


def _resolve_options(names: tuple[str, ...], topology: dict, tokens: int) -> SimpleNamespace:
    path = _checkpoint_dir(names)
    options = _options(topology, tokens)
    return _resolve(path, _llm_args(model=str(path), modeling_v2="require", **options))


@pytest.fixture(scope="module")
def dep4():
    """The four-rank accuracy and serve gates' engine: spec-dep4/options/core.yaml."""
    return _resolve_options(_NVFP4_NAMES, _DEP4, _TOKENS)


@pytest.fixture(scope="module")
def dep8():
    """The eight-rank accuracy and serve gates' engine: spec-dep8/options/core.yaml."""
    return _resolve_options(_NVFP4_NAMES, _DEP8, _TOKENS)


@pytest.fixture(scope="module")
def dep8_gate():
    """The eight-rank boot gate's engine: its quickstart command."""
    path = _checkpoint_dir(_NVFP4_NAMES)
    return _resolve(path, _llm_args(**_gate_kwargs(path, _GATE_ARGV)))


@pytest.fixture(scope="module")
def tep4():
    return _resolve_options(_NVFP4_NAMES, _TEP4, _TOKENS)


@pytest.fixture
def dep(request, world):
    """The attention-DP engine at ``world`` ranks, from its core.yaml."""
    return request.getfixturevalue(_ENGINES[world][0])


@pytest.fixture
def engines(request, world):
    """Every spelling of the attention-DP engine at ``world`` ranks, core.yaml's first."""
    return [request.getfixturevalue(name) for name in _ENGINES[world]]


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
    """Everything the engine resolved for a checkpoint except its mapping and the MoE token
    budget, which reads the mapping's attention-DP width."""
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
        cuda_graphs=llm_args.cuda_graph_config.model_dump(),
        kv_cache=llm_args.kv_cache_config.model_dump(),
        moe=llm_args.moe_config.model_dump(),
        moe_backend=model_config.moe_backend,
    )


def _route_or_none(seen):
    try:
        return routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
    except NoTarget:
        return None


# ------------------------------------------------------- the world's ranks in one process


class _Group:
    """The ranks of the attention-DP group in one process: rank r is a thread, and every exchange
    is a barrier, so each rank's call sees what every rank sent, as over MPI."""

    def __init__(self, size: int):
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

    def __init__(self, group: _Group, rank: int):
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

    tokens_per_block = 32
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


# ------------------------------------------------------- (1) topology, node layout, limits


@_each_world
def test_dep4_topology_and_per_rank_limits(world, engines):
    """``world`` ranks -- four on one node, or eight over two nodes of four (dep8) --
    tensor_parallel_size ``world`` spent on attention data parallelism (each rank holds every head
    and schedules its own requests; attention TP is 1), the routed experts expert-parallel over
    every rank and not split along their width, one pipeline stage -- at eight ranks from
    core.yaml and from the boot gate's command alike. lm_head is not vocab-parallel under
    attention DP. Rank r is TP rank r and MoE EP rank r, MoE TP rank 0: its expert window is
    [E r, E r + E), E = 128 / world (32 at four ranks, 16 at eight). mapping.tp_size,
    attn_tp_size and dp_size all read the world, so no width outside the expert window may be
    divided by them. max_batch_size and max_num_tokens are each rank's own: 16 requests and 32768
    tokens per rank, 16 x world requests in flight (64 over four ranks, 128 over eight),
    unchunked."""
    local = {4: 32, 8: 16}[world]
    for resolved in engines:
        llm_args, mapping = resolved.llm_args, resolved.mapping
        parallel = llm_args.parallel_config
        assert (parallel.tp_size, parallel.pp_size, parallel.cp_size) == (world, 1, 1)
        # Every spelling states the EP split only; the MoE TP size is left to Mapping.
        assert (parallel.moe_ep_size, parallel.moe_tp_size) == (world, -1)
        assert llm_args.enable_attention_dp is True and parallel.enable_attention_dp is True

        assert (mapping.world_size, mapping.tp_size, mapping.pp_size, mapping.cp_size) == (
            world,
            world,
            1,
            1,
        )
        assert (mapping.moe_ep_size, mapping.moe_tp_size, mapping.moe_cluster_size) == (world, 1, 1)
        assert mapping.enable_attention_dp is True
        # The flag, not a width, says attention is replicated: all three read the world.
        assert (mapping.tp_size, mapping.attn_tp_size, mapping.dp_size) == (world, world, world)
        assert mapping.tp_size // mapping.dp_size == 1
        assert mapping.enable_lm_head_tp_in_adp is False
        assert mapping.has_moe_ep() and not mapping.has_moe_tp()
        assert resolved.model_config.mapping is mapping

        cfg = resolved.model_config.pretrained_config
        assert (cfg.num_attention_heads, cfg.n_routed_experts) == (64, 128)
        assert cfg.n_routed_experts // mapping.moe_ep_size == local
        for rank in range(world):
            rank_mapping = _rank_mapping(mapping, rank)
            assert (
                rank_mapping.tp_rank,
                rank_mapping.moe_ep_rank,
                rank_mapping.moe_tp_rank,
                rank_mapping.moe_cluster_rank,
            ) == (rank, rank, 0, 0)
            assert rank_mapping.ep_experts(cfg.n_routed_experts) == list(
                range(local * rank, local * rank + local)
            )
            assert rank_mapping.tp_group == list(range(world))

        assert (resolved.max_batch_size, resolved.max_num_tokens, resolved.max_seq_len) == (
            16,
            32768,
            32768,
        )
        model_config = resolved.model_config
        assert (model_config.max_num_tokens, model_config.max_seq_len) == (32768, 32768)
        assert llm_args.enable_chunked_prefill is False
        # PyExecutor's per-rank cap on active requests under attention DP.
        cap = derive_attention_dp_per_rank_request_cap(
            base_cap=mapping.pp_size * resolved.max_batch_size,
            max_num_tokens=resolved.max_num_tokens,
            max_total_draft_tokens=0,
        )
        assert cap == 16 and cap * mapping.dp_size == {4: 64, 8: 128}[world]
        # No attention-DP balancing or KV-aware routing: the default router places requests.
        # core.yaml states no attention_dp_config; the quickstart builds one at its defaults.
        adp = llm_args.attention_dp_config
        assert (adp is None) == (resolved is engines[0])
        assert adp is None or (adp.enable_balance, adp.enable_kv_cache_aware_routing) == (
            False,
            False,
        )
        router = ADPRouter.create(dist=None, has_seq_slot_headroom=False, attention_dp_config=adp)
        assert type(router) is DefaultADPRouter


def test_gpus_per_node_is_the_visible_device_count(dep8, dep8_gate):
    """At eight ranks over two nodes of four (dep8), neither gate states gpus_per_node.
    ``TorchLlmArgs.gpus_per_node`` defaults to None and its validator (outside Ray) replaces None
    with ``torch.cuda.device_count()`` of the process that builds the arguments, rank 0; each node's
    ranks see that node's four held devices, so it is 4. The mapping every rank builds from
    ``parallel_config`` carries it: rank r is local rank r % 4 on node r // 4, and the deployment is
    multi-node. The LLM constructor's device check reads it too: world_size_per_node is 4, which
    four visible devices satisfy.

    The defaults of ``_ParallelConfig`` and ``Mapping`` are 8, not the device count: an
    eight-rank mapping built without it reads one node of eight."""
    field = TorchLlmArgs.model_fields["gpus_per_node"]
    assert field.default is None and field.validate_default is True
    assert "gpus_per_node" not in _options(_DEP8, _TOKENS)
    assert "gpus_per_node" not in _gate_kwargs(dep8_gate.path, _GATE_ARGV)

    for resolved in (dep8, dep8_gate):
        llm_args, mapping = resolved.llm_args, resolved.mapping
        assert llm_args.gpus_per_node == _GPUS_PER_NODE == 4
        assert llm_args.parallel_config.gpus_per_node == 4
        assert mapping.gpus_per_node == 4 and resolved.model_config.mapping.gpus_per_node == 4
        assert llm_args.parallel_config.world_size_per_node == 4
        for rank in range(_DEP8_RANKS):
            rank_mapping = _rank_mapping(mapping, rank)
            assert (rank_mapping.local_rank, rank_mapping.node_rank) == (rank % 4, rank // 4)
            assert rank_mapping.get_node_rank(rank) == rank // 4
            assert rank_mapping.is_multi_node() is True

    # The count is the visible devices' count, read when the arguments are built.
    path = str(dep8.path)
    for devices in (4, 8):
        llm_args = _llm_args(devices, model=path, **_options(_DEP8, _TOKENS))
        assert llm_args.gpus_per_node == devices
        assert llm_args.parallel_config.world_size_per_node == devices
    stated = _llm_args(1, model=path, gpus_per_node=4, **_options(_DEP8, _TOKENS))
    assert stated.gpus_per_node == 4


@_each_world
def test_each_rank_selects_its_own_nodes_device(world, dep):
    """The device each worker sets before building its executor
    (``BaseWorker._get_comm_ranks_device_id``): its global MPI rank modulo the devices its node
    shows, so rank r drives device r % 4 of its node -- the mapping's local rank, at four ranks
    on the one node and at eight on either. The ranks allgather their global ranks and devices,
    and each pins its CPU affinity to its device."""
    group = _Group(world)
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
        assert ranks == list(range(world))
        assert devices == {4: [0, 1, 2, 3], 8: [0, 1, 2, 3, 0, 1, 2, 3]}[world]
        assert device == cpus == rank % 4 == _rank_mapping(dep.mapping, rank).local_rank


def test_boot_command_and_core_yaml_build_one_deployment(dep8, dep8_gate):
    """At eight ranks, the boot gate's quickstart command and core.yaml resolve to the same mapping
    on every rank and the same model, quantization, DSA, limits, CUDA graphs, KV dtype, page size,
    block-reuse setting, KV manager and MoE token budget; the router is handed equal mappings. The
    KV fraction is not shared: the quickstart's --kv_cache_fraction default, 0.9, against
    core.yaml's 0.85, so the boot gate's pool is sized at 0.9 of the free memory and the accuracy
    gate's at 0.85."""
    assert dep8_gate.mapping.to_dict() == dep8.mapping.to_dict()
    assert dep8_gate.routed[0].mapping.to_dict() == dep8.routed[0].mapping.to_dict()
    gate, core = _summary(dep8_gate), _summary(dep8)
    for key in ("llm_quant", "quant", "layers", "pretrained", "sparse", "routed", "sizes"):
        assert gate[key] == core[key], key
    assert dep8_gate.model_config.moe_max_num_tokens == dep8.model_config.moe_max_num_tokens
    for resolved in (dep8, dep8_gate):
        graphs = resolved.llm_args.cuda_graph_config
        assert graphs.batch_sizes == [1, 2, 4, 8, 16] and graphs.enable_padding is True
        kv = resolved.llm_args.kv_cache_config
        assert (kv.dtype, kv.enable_block_reuse, kv.tokens_per_block) == ("auto", False, 32)
        assert kv.use_kv_cache_manager_v2 is True
        assert resolved.llm_args.disable_overlap_scheduler is False
    assert dep8.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.85
    assert dep8_gate.llm_args.kv_cache_config.free_gpu_memory_fraction == 0.9


# ------------------------------------------------------- (1) quantization, the anchor's key


@_each_world
def test_quantization_is_the_anchor_key(world, engines):
    """At dep4 and dep8 as at tep4: MIXED_PRECISION with the FP8 KV cache, the GSM8K anchor row's
    key, and the router handed params.json's quantization_config with its 475 quantized_layers
    intact, on the world's mapping -- at eight ranks from core.yaml and from the boot gate's
    command alike."""
    on_disk = json.loads((engines[0].path / "params.json").read_text())["quantization_config"]
    assert len(on_disk["quantized_layers"]) == 475
    for resolved in engines:
        assert resolved.prequantized is True
        for quant in (resolved.llm_args.quant_config, resolved.model_config.quant_config):
            assert quant.quant_algo == QuantAlgo.MIXED_PRECISION
            assert quant.kv_cache_quant_algo == QuantAlgo.FP8
            assert quant.group_size == 16
        assert resolved.llm_args.kv_cache_config.dtype == "auto"
        assert _anchor(resolved.llm_args) == 91.926

        assert len(resolved.routed) == 1
        seen = resolved.routed[0]
        assert seen.quantization_config == on_disk
        assert resolved.model_config.pretrained_config.quantization_config == on_disk
        assert seen.pretrained_config is resolved.model_config.pretrained_config
        assert resolved.model_config.modeling_v2_target is _Routed
        assert seen.compute_capability == _SM103 and seen.spec_config is None
        assert seen.mapping is resolved.model_config.mapping
        mapping = seen.mapping
        assert (mapping.world_size, mapping.tp_size, mapping.pp_size) == (world, world, 1)
        assert (mapping.moe_ep_size, mapping.moe_tp_size) == (world, 1)
        assert mapping.enable_attention_dp is True
        assert mapping.enable_lm_head_tp_in_adp is False


# ------------------------------------------------------- (2) the KV pool, per rank


@_each_world
def test_kv_pool_has_no_rank_factor(world, engines, dep4, tep4):
    """Each rank holds the latent and indexer-key caches of its own requests only, and the cost of
    a token is the same on each of the world's ranks, at four and at eight, as on each tep4 rank:
    an fp8_ds_mla latent of 656 bytes (512 e4m3, four fp32 tile scales, 64 bf16 rope) on each of
    the 63 KV owners plus a 128-byte fp8 indexer key and its fp32 scale on each of the 44
    full-indexer layers, in pages of 32, in the MistralLarge4CacheManager the engine builds for
    this checkpoint; the nine KV sharers keep no latent. A rank's own 16 requests at 32768 tokens
    need that times their tokens."""
    for resolved in (*engines, dep4, tep4):
        model_config, kv_cache_config = resolved.model_config, resolved.llm_args.kv_cache_config
        assert get_kv_cache_manager_cls(model_config, kv_cache_config) is MistralLarge4CacheManager
        assert kv_cache_config.tokens_per_block == 32

    dep = engines[0]
    cfg, sparse = dep.model_config.pretrained_config, dep.model_config.sparse_attention_config
    latent = 656  # fp8_ds_mla: 512 e4m3 + 4 fp32 tile scales + 64 bf16 rope
    owners = sum(1 for i, s in enumerate(cfg.kv_source_layers) if s == i)
    indexer_key = sparse.index_head_dim + sparse.index_head_dim // 128 * 4
    full_indexer_layers = cfg.index_topk_pattern.count("F")
    per_token = latent * owners + indexer_key * full_indexer_layers
    assert (latent, owners, indexer_key, full_indexer_layers) == (656, 63, 132, 44)
    assert per_token == 47136

    for rank in range(world):
        cost = _kv_cost(dep, rank)
        assert (cost.slope, cost.intercept) == (per_token, 0)
        assert all(cost == _kv_cost(other, rank) for other in engines[1:])
        assert cost == _kv_cost(dep4, rank % 4) == _kv_cost(tep4, rank % 4)
        assert cost.bytes_for_tokens(32) == 32 * per_token

    tokens = dep.max_batch_size * dep.max_seq_len
    pool = _kv_cost(dep, 0).bytes_for_tokens(tokens)
    assert pool == tokens * per_token
    assert math.isclose(pool / 2**30, 23.016, abs_tol=5e-4)


@_each_world
def test_indexer_chunk_holds_a_whole_prompt(world, dep):
    """The indexer's [T, N] logits are cut at indexer_max_chunk_size: a rank's whole 32K prompt,
    its largest unchunked step, is one chunk."""
    model_config = dep.model_config
    params = model_config.sparse_attention_config.to_sparse_metadata_params(
        pretrained_config=model_config.pretrained_config
    )
    assert params.indexer_max_chunk_size >= 32768
    assert params.indexer_max_chunk_size >= dep.max_num_tokens


# ------------------------------------------------------- (3) what the forward receives


@_each_world
def test_every_rank_is_handed_every_rank_token_count(world, dep):
    """``attn_metadata.all_rank_num_tokens`` is set on every step of every rank, by the gather in
    the decoder's input preparation (eager, warmup and capture steps alike): each rank's own
    token count, one entry per rank in rank order, identical on every rank and NOT padded (no
    prefill CUDA graph backend is configured). An idle rank counts the one token of its
    generation dummy. Padding to the largest count before the experts' collectives is the
    forward's own job. When every rank prefills a whole step at once, the experts see world x
    32768 gathered rows: 131072 at four ranks, 524288 after top-4; 262144 at eight, 1048576
    after top-4."""
    llm_args, model_config = dep.llm_args, dep.model_config
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

    group = _Group(world)
    unequal = {
        # a 32K prefill, an idle rank's dummy, a mixed step, sixteen decode rows
        4: ([32768, 1, 517, 16], [1, 0, 1, 0]),
        # 32K prefills on both nodes, idle ranks' dummies, a mixed step, sixteen decode rows
        8: ([32768, 1, 517, 16, 1, 32768, 3, 1], [1, 0, 1, 0, 0, 1, 0, 0]),
    }
    steps = {
        # every rank prefills a whole 32K step, as the warmup's max shape does
        "prefills": ([32768] * world, [2] * world),
        "unequal": unequal[world],
        # a decode graph capture: every rank at the same captured size
        "capture": ([16] * world, [0] * world),
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
                mapping=_rank_mapping(dep.mapping, rank),
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
            assert all_rank == counts and len(all_rank) == world

    rows = sum(steps["prefills"][0])
    assert rows == world * dep.max_num_tokens == {4: 131072, 8: 262144}[world]
    top4 = rows * model_config.pretrained_config.num_experts_per_tok
    assert top4 == {4: 524288, 8: 1048576}[world]


@_each_world
def test_an_idle_rank_steps_with_a_one_token_generation_dummy(world, dep):
    """When some rank has work and this one has none, the executor gives it one dummy before
    scheduling: request id ATTENTION_DP_DUMMY_REQUEST_ID, a generation request with the KV
    manager's default of one token (no draft tokens), its KV prepared like a real request's.
    Aggregated serving with the overlap scheduler takes this plain path: no KV-cache transceiver,
    and neither the scheduler-aware nor the forward-intent variant. The rank's step is then
    generation-only with one token, so it runs the decode target while its peers may prefill.
    When no rank has work, no rank gets a dummy and none steps."""
    llm_args, mapping = dep.llm_args, dep.mapping
    assert llm_args.disable_overlap_scheduler is False
    assert should_enable_adp_dummy_fixes(mapping) is True
    model_type = dep.model_config.pretrained_config.model_type
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
            kv_cache_manager=_KvPool(dep.max_batch_size * dep.max_seq_len, dep.max_seq_len),
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


@_each_world
def test_decode_graphs_replay_only_when_every_rank_decodes(world, dep, monkeypatch):
    """The CUDA-graph rule under attention DP, on the engine's own runner: a step replays only
    when every rank's batch, over the four or the eight, is generation-only. Then each rank pads
    to the largest batch across the ranks, rounded up to a captured size, so all replay the same
    batch size; an idle rank's one-row dummy is padded like any batch. When any rank has a
    context request, no rank replays; a decode rank still pads its own batch to its own captured
    size and runs it eagerly. The ranks agree on eligibility and batch size only: DSA captures a
    short- and a long-sequence family at each size, and each rank picks its family from its own
    sequences, so ranks can replay different families in one step."""
    llm_args, model_config = dep.llm_args, dep.model_config
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
    group = _Group(world)
    captured = SimpleNamespace(is_cuda_graph=True)
    runners = []
    for rank in range(world):
        runner = CUDAGraphRunner(
            CUDAGraphRunnerConfig(
                use_cuda_graph=True,
                cuda_graph_padding_enabled=graphs.enable_padding,
                cuda_graph_batch_sizes=graphs.batch_sizes,
                max_cuda_graph_batch_size=graphs.max_batch_size,
                max_beam_width=1,
                max_num_tokens=dep.max_num_tokens,
                spec_config=None,
                cuda_graph_mem_pool=None,
                use_mrope=False,
                original_max_draft_len=0,
                original_max_total_draft_tokens=0,
                enable_attention_dp=dep.mapping.enable_attention_dp,
                is_encoder_decoder=False,
                batch_size=dep.max_batch_size,
                mapping=_rank_mapping(dep.mapping, rank),
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
            for longest in (dep.max_seq_len - 1, threshold - 2):
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

    # Every rank decodes, short sequences, unequal batches (on both nodes at eight ranks): all
    # pad to 16 and replay it.
    unequal = {
        4: ([3, None, 9, 16], [3, None, 9, 5]),
        8: ([3, None, 9, 16, 1, None, 5, 2], [3, None, 9, 5, None, None, 1, 12]),
    }
    for sizes in unequal[world]:
        batches = [idle() if n is None else _batch(decode=[100] * n) for n in sizes]
        for size, key, attn_metadata in step(batches):
            assert size == 16 and attn_metadata is captured
            assert (key.batch_size, key.draft_len, key.short_seq_len_mode) == (16, 0, True)

    # Small batches pad to the smallest captured size that holds the largest.
    batches = [_batch(decode=[100, 100]) if rank % 2 == 0 else idle() for rank in range(world)]
    assert [(size, key.batch_size) for size, key, _ in step(batches)] == [(2, 2)] * world

    # Equal sizes, different families: the rank holding a 30000-token sequence (rank 1 at four
    # ranks, rank 5 on the second node at eight) replays the long graph while its peers replay
    # the short one.
    long_rank = {4: 1, 8: 5}[world]
    batches = [_batch(decode=[30000 if rank == long_rank else 100] * 16) for rank in range(world)]
    modes = [key.short_seq_len_mode for _, key, _ in step(batches)]
    assert modes == {4: [True, False, True, True], 8: [True] * 5 + [False] + [True] * 2}[world]

    # One rank prefills: nobody replays, and a decode rank runs its own padded batch eagerly.
    prefill_step = {
        4: ((None, 9, 16), [1, 1, 16, 16]),
        8: ((None, 9, 16, None, None, 3, 1), [1, 1, 16, 16, 1, 1, 4, 1]),
    }
    others, sizes = prefill_step[world]
    batches = [_batch(context=[32768])] + [
        idle() if n is None else _batch(decode=[100] * n) for n in others
    ]
    results = step(batches)
    assert [size for size, _, _ in results] == sizes
    assert all(key is None and attn_metadata is None for _, key, attn_metadata in results)


@_each_world
def test_warmup_runs_the_max_context_step_on_every_rank_at_once(world, dep):
    """The warmup's shapes under attention DP. Each rank proposes its shapes from its own free
    KV, the ranks keep the ones all proposed, and every rank enters each shape's forward. The max
    context shape is max_num_tokens of context on every rank in the same step -- a 32767-token
    prompt and a 1-token prompt, since a prompt leaves room for one decode token -- so the routed
    experts see a 32K prefill on every rank at boot: 131072 gathered rows at four ranks, 262144 at
    eight. Each decode graph is captured on a batch whose first row attends max_seq_len - 1 keys
    (long family) or threshold - 2 (short family) and whose other rows attend one."""
    config = _runner_config(dep)
    sparse = copy.deepcopy(dep.model_config.sparse_attention_config)
    assert sparse.needs_separate_short_long_cuda_graphs() is True
    model = SimpleNamespace(model_config=dep.model_config)
    group = _Group(world)

    def rank_warmup(rank):
        # A rank's own pool holds at least its 16 requests of max_seq_len.
        resources = _Resources(
            _KvPool(config.max_batch_size * config.max_seq_len, config.max_seq_len)
        )
        warmup = DecoderWarmup.__new__(DecoderWarmup)
        warmup._config = config
        warmup.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        agreement = WarmupAgreement(group.dist(rank), _rank_mapping(dep.mapping, rank))
        assert agreement.is_distributed_forward()
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
        assert prompts == [32767, 1]
        assert captures == {"long": [32767] + [1] * 15, "short": [2046] + [1] * 15}
    total = sum(sum(prompts) for _, _, prompts, _ in results)
    assert total == world * dep.max_num_tokens == {4: 131072, 8: 262144}[world]


@_each_world
def test_kv_estimation_puts_a_whole_context_step_on_every_rank(world, dep):
    """Before the real pool exists the engine runs a profiling pass: under attention DP it builds
    one copy of its max_num_tokens of dummy context per rank (a 32767-token and a 1-token prompt
    each, 2 x world requests) and rank 0 enqueues all of them before the executor loop starts.
    The loop's first fetch finds the queue non-empty and drains it whole (no batch wait is
    configured); the waiting queue hands all of them on, within the world x 16 admission capacity
    (64 at four ranks, 128 at eight); and the default router gives every rank one of each. So the
    estimation's first step is a 32768-token context step on every rank at once -- 131072
    gathered rows at every MoE layer at four ranks, 262144 at eight -- and each rank's scheduler
    takes its two prompts together (32767 + 1 tokens within max_num_tokens, unchunked)."""
    llm_args = dep.llm_args
    creator = KvCacheCreator.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=dep.model_config), use_mrope=False
    )
    creator._max_num_tokens = dep.max_num_tokens
    creator._max_beam_width = 1
    creator._mapping = dep.mapping
    # The estimation's dummies are as long as a request may be: net max_seq_len - 1.
    requests = creator._create_dummy_context_requests(max(1, dep.max_seq_len - 1))
    assert creator._mla_chunked_profile_length is None
    lengths = [len(request.input_token_ids) for request in requests]
    assert lengths == [32767, 1] * world

    # Rank 0's request queue, filled before the loop: the first fetch takes every item.
    assert llm_args.batch_wait_timeout_ms == 0
    request_queue = queue.Queue()
    for request_id, request in enumerate(requests, start=1):
        request_queue.put(RequestQueueItem(request_id, request))
    fetched = ExecutorRequestQueue.get_from_request_queue(
        SimpleNamespace(
            request_queue=request_queue,
            batch_wait_timeout_ms=llm_args.batch_wait_timeout_ms,
            max_batch_size=dep.max_batch_size,
        ),
        None,
    )
    assert len(fetched) == 2 * world and request_queue.empty()

    # Every rank's copy of the waiting queue gives them all up at once.
    waiting = FCFSWaitingQueue()
    waiting.add_requests(fetched)
    capacity = PyExecutor._get_request_admission_capacity(
        SimpleNamespace(
            enable_attention_dp=True,
            dist=SimpleNamespace(tp_size=dep.mapping.tp_size),
            max_num_active_requests=dep.max_batch_size,
        )
    )
    assert capacity == 16 * world == {4: 64, 8: 128}[world]
    popped = get_from_waiting_queue(
        waiting,
        capacity,
        enable_attention_dp=True,
        max_num_active_requests=dep.max_batch_size,
        all_ranks_num_active_requests=[0] * world,
    )
    assert len(popped) == 2 * world and not waiting

    router = ADPRouter.create(
        dist=None,
        has_seq_slot_headroom=False,
        attention_dp_config=llm_args.attention_dp_config,
    )
    assigned, expected = router.route_requests(
        [RankState(rank=rank) for rank in range(world)], popped, dep.max_batch_size
    )
    assert expected == 2
    for rank in range(world):
        prompts = sorted((len(item.request.input_token_ids) for item in assigned[rank]))
        assert prompts == [1, 32767] and sum(prompts) == dep.max_num_tokens


# ------------------------------------------------------- (4) the engine's reads of the node layout


@_each_world
def test_shell_lm_head_is_whole_under_attention_dp(world, dep):
    """The causal-LM shell every target subclasses builds, under attention DP with lm_head TP
    off, a replicated lm_head: the full 204800-token vocabulary of 6144 on every rank, at four
    ranks and at eight, on a one-rank mapping with no tensor-parallel mode, so no gather and no
    reduction. The node layout does not enter. Built on the meta device."""
    for rank in range(world):
        config = dataclasses.replace(dep.model_config, mapping=_rank_mapping(dep.mapping, rank))
        cfg = config.pretrained_config
        with torch.device("meta"):
            shell = DecoderModelForCausalLM(
                DecoderModel(config),
                config=config,
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
            )
        head = shell.lm_head
        assert tuple(head.weight.shape) == (204800, 6144)
        assert (head.tp_size, head.tp_mode, head.gather_output) == (1, None, False)
        assert head.mapping.world_size == 1
        assert (shell.pp_rank, shell.pp_size) == (0, 1)


def test_collective_paths_read_the_node_layout(dep8, monkeypatch):
    """Where the engine's host exchanges and the target's collectives meet the node layout at
    dep8, on each rank's mapping:

    * the rank's mapping is an ``MpiTopology`` and its communicator an ``MPIDist`` (the MPI
      launch leaves TLLM_DISABLE_MPI unset); every host exchange of attention DP -- the
      token-count gather, the CUDA-graph batch agreement, the warmup's agreement -- runs over
      MPI on ``MPIDist.tp_comm``, the group of ``mapping.tp_group``, all eight ranks over both
      nodes;
    * the catalog's comm entries with ``process_group=None`` (the flavor a target picks under
      MPI) call the MPI-session symbols ``torch.ops.trtllm.allgather`` / ``reducescatter`` with
      the group they are handed, whose NCCL communicator ``getComm`` builds for that rank set
      through the MPI session: it reads no node layout itself;
    * the mapping says multi-node, and the one-node IPC workspace an engine AllReduce would
      open stays closed past one node's devices; no engine AllReduce is built on this path."""
    assert dep8.llm_args.allreduce_strategy == "AUTO"
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)
    for rank in range(_DEP8_RANKS):
        rank_mapping = _rank_mapping(dep8.mapping, rank)
        assert type(rank_mapping) is MpiTopology
        assert type(Distributed.get(rank_mapping)) is MPIDist
        assert rank_mapping.tp_group == list(range(_DEP8_RANKS))
        assert rank_mapping.is_multi_node() is True
        assert IpcMemory(rank_mapping, 1024).open_ipc is False

    calls = []
    for name in ("allgather", "reducescatter", "allgather_pg", "reducescatter_pg"):
        monkeypatch.setattr(
            torch.ops.trtllm,
            name,
            lambda *args, name=name: calls.append((name, args[2], args[3:])) or args[0],
        )
    rows = torch.zeros(4, 6144, dtype=torch.bfloat16)
    group = list(range(_DEP8_RANKS))
    allgather(rows, None, group, None)
    reducescatter(rows, None, group, None)
    allgather(rows, None, group, "pg")
    assert calls == [
        ("allgather", group, ()),
        ("reducescatter", group, ()),
        ("allgather_pg", group, ("pg",)),
    ]


@pytest.mark.parametrize("chunk", [32768, 65536])
def test_moe_autotuner_at_sixteen_local_experts(dep8, dep4, chunk):
    """At eight ranks (dep8), the autotuner's tuning group is the TP group, ``(tp_size, tp_rank)``:
    all eight ranks over both nodes. The NVFP4 MoE runner tunes INDEPENDENT, each rank alone, so
    tuning sends nothing between the nodes. Its expert-parallel size, num_experts //
    local_num_experts, is 8; with ``use_dp`` False, as the dep4 target passes it, that moves no
    bucket: an expert call's rows map to the buckets 1..8192 and the ``tune_max_num_tokens`` bucket
    exactly as at dep4, so a chunk of ``chunk`` gathered rows tunes and runs at ``chunk``.
    ``use_dp`` True would deflate the rows by 8 first. The engine's own MoE token budget,
    ModelConfig.moe_max_num_tokens, reads the attention-DP width (8 x 32768), and no MoE load
    balancer is built."""
    for rank in range(_DEP8_RANKS):
        tuner = SimpleNamespace(mapping=_rank_mapping(dep8.mapping, rank))
        assert AutoTuner._tuning_group(tuner) == (8, rank)

    cfg = dep8.model_config.pretrained_config
    local_experts = cfg.n_routed_experts // dep8.mapping.moe_ep_size
    ep_size = cfg.n_routed_experts // local_experts
    assert (local_experts, ep_size) == (16, 8)
    dep8_tuning = FP4BlockScaleMoERunner.get_tuning_config(
        ep_size, tune_max_num_tokens=chunk, use_dp=False
    )
    dep4_tuning = FP4BlockScaleMoERunner.get_tuning_config(
        4, tune_max_num_tokens=chunk, use_dp=False
    )
    assert dep8_tuning.distributed_tuning_strategy == DistributedTuningStrategy.INDEPENDENT
    (dep8_spec,) = dep8_tuning.dynamic_tensor_specs
    (dep4_spec,) = dep4_tuning.dynamic_tensor_specs
    assert dep8_spec.gen_tuning_buckets == dep4_spec.gen_tuning_buckets
    assert max(dep8_spec.gen_tuning_buckets) == chunk
    for tokens in (1, 3, 16, 17, 4097, 8192, 8193, 16384, chunk - 1, chunk):
        assert dep8_spec.map_to_tuning_buckets(tokens) == dep4_spec.map_to_tuning_buckets(tokens)
    assert dep8_spec.map_to_tuning_buckets(chunk) == chunk
    (deflated,) = FP4BlockScaleMoERunner.get_tuning_config(
        ep_size, tune_max_num_tokens=chunk, use_dp=True
    ).dynamic_tensor_specs
    assert deflated.map_to_tuning_buckets(chunk) == min(chunk // ep_size, 8192)

    model_config = dep8.model_config
    assert model_config.moe_max_num_tokens == dep8.max_num_tokens * dep8.mapping.dp_size == 262144
    assert dep4.model_config.moe_max_num_tokens == 131072
    assert dep8.llm_args.moe_config.load_balancer is None and model_config.moe_load_balancer is None
    for rank in range(_DEP8_RANKS):
        with maybe_create_moe_load_balancer(model_config, _rank_mapping(dep8.mapping, rank)) as lb:
            assert lb is None


def _executor_runtime_features(llm_args, max_num_tokens: int) -> AttentionRuntimeFeatures:
    """The attention runtime features ``_create_py_executor`` builds once for the service from
    the LLM arguments: chunked prefill from enable_chunked_prefill, cache reuse from the
    KvCacheConfig's enable_block_reuse, no speculative drafter, chunks of max_num_tokens."""
    return AttentionRuntimeFeatures(
        chunked_prefill=llm_args.enable_chunked_prefill,
        cache_reuse=llm_args.kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=False,
        chunk_size=max_num_tokens,
    )


def _first_step_md(
    resolved,
    max_seq_len: int | None = None,
    enable_chunked_prefill: bool | None = None,
    enable_block_reuse: bool | None = None,
) -> SimpleNamespace:
    """The attention metadata of the first step ``resolved``'s engine hands the forward, as
    ``_derive_first_step`` reads it: max_num_tokens, max_batch_size (``max_num_requests``),
    max_seq_len, the runtime features the executor builds from the LLM arguments
    (``_executor_runtime_features``), the pool's pages and a fresh fp8_ds_mla KV cache manager
    over the checkpoint's kv_source_layers. ``max_seq_len`` overrides the engine's;
    ``enable_chunked_prefill`` and ``enable_block_reuse`` override the LLM arguments' (the
    latter their KvCacheConfig's) before the runtime features are built from them."""
    llm_args = resolved.llm_args
    kv_cache_config = llm_args.kv_cache_config
    if enable_block_reuse is not None:
        kv_cache_config = kv_cache_config.model_copy(
            update=dict(enable_block_reuse=enable_block_reuse)
        )
    update = dict(kv_cache_config=kv_cache_config)
    if enable_chunked_prefill is not None:
        update.update(enable_chunked_prefill=enable_chunked_prefill)
    llm_args = llm_args.model_copy(update=update)
    sources = list(resolved.model_config.pretrained_config.kv_source_layers)
    owners = [i for i, s in enumerate(sources) if s == i]
    return SimpleNamespace(
        max_num_tokens=resolved.max_num_tokens,
        max_num_requests=resolved.max_batch_size,
        max_seq_len=resolved.max_seq_len if max_seq_len is None else max_seq_len,
        runtime_features=_executor_runtime_features(llm_args, resolved.max_num_tokens),
        tokens_per_block=llm_args.kv_cache_config.tokens_per_block,
        kv_cache_manager=SimpleNamespace(
            latent_layout=MistralLarge4CacheManager.latent_layout,
            kv_source_layers=sources,
            latent_pool_row=lambda i: owners.index(sources[i]),
            kv_owner_layers=owners,
        ),
    )


@_each_world
def test_expert_call_size_is_the_worlds(world, dep, monkeypatch):
    """The expert path's call -- the most gathered rows one router / routing / NVFP4-quantize /
    expert call takes, and the autotuner's largest bucket -- is fixed on the first step the
    engine hands over (``_derive_first_step``), from the world and the engine's chunked prefill
    flag alone, as the runtime features the executor builds from the LLM arguments carry it: on
    this engine (32768 tokens per rank, neither chunked prefill nor block reuse) at four ranks
    all 131072 gathered rows of a 32K step on every rank, one call per MoE layer; at eight ranks
    32768 of the up to 262144, in calls of 32768, the dep8 autotuner's 32768 chunk. The same
    engine with the KvCacheConfig's enable_block_reuse (the runtime features' cache_reuse) calls
    the same, though a step may then attend more context KV than max_num_tokens (its held
    context workspace grows from 65536 rows to 2**19). With enable_chunked_prefill, at
    max_seq_len 1048576 (the 1M config) or at 32768, block reuse or not, it calls 32768 at both:
    at four ranks the up to 131072 gathered rows run in four calls, one call's workspace over all
    of them costing the engine about 5.6 GiB of free memory, more than the 1M config's margin
    spares. The core's expert window and attention-DP group are the world's. Built on the meta
    device, with stand-ins for its kernel builds."""
    assert M._EXPERT_CALL_TOKENS == {4: 131072, 8: 32768}
    assert M._EXPERT_CALL_TOKENS_CHUNKED == {4: 32768, 8: 32768}
    assert (dep.max_num_tokens, dep.max_seq_len, dep.max_batch_size) == (32768, 32768, 16)
    assert not dep.llm_args.enable_chunked_prefill
    assert not dep.llm_args.kv_cache_config.enable_block_reuse
    monkeypatch.setattr(
        M, "_rope_tables", lambda cfg, alpha, rows, dtype, device: SimpleNamespace(rows=rows)
    )
    monkeypatch.setattr(M, "build_ix_q_side", lambda dev, is_neox: None)
    monkeypatch.setattr(M, "build_ix_k_cache", lambda dev, is_neox, cache_contiguous: None)
    monkeypatch.setattr(
        M, "build_ix_topk_decode", lambda dev, next_n: SimpleNamespace(next_n=next_n)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with torch.device("meta"):
        core = M.MistralLarge4Dep4ModelingV2(dep.model_config)
    assert (core.expert_call_tokens, core.expert_tune_tokens) == (0, 0)
    core.build_kernels(0)
    # (overrides of the engine's max_seq_len and LLM arguments, the expert call, the held
    # context workspace's rows)
    for engine, call, rows in (
        (dict(), {4: 131072, 8: 32768}[world], 65536),
        (dict(enable_block_reuse=True), {4: 131072, 8: 32768}[world], 1 << 19),
        (dict(max_seq_len=1048576, enable_chunked_prefill=True), 32768, 1 << 19),
        (dict(enable_chunked_prefill=True), 32768, 1 << 19),
        (dict(enable_chunked_prefill=True, enable_block_reuse=True), 32768, 1 << 19),
    ):
        # Each with a KV cache manager of its own: the core derives its sizes once per manager.
        md = _first_step_md(dep, **engine)
        features = md.runtime_features
        assert features.chunked_prefill is engine.get("enable_chunked_prefill", False), engine
        assert features.cache_reuse is engine.get("enable_block_reuse", False), engine
        core._derive_first_step(md)
        assert core.expert_call_tokens == call, engine
        assert core.ctx_workspace_rows == rows, engine
        assert core.expert_tune_tokens == min(call, world * dep.max_num_tokens) == call
    assert (core.dp_group, core.dp_rank) == (list(range(world)), 0)
    assert (core.local_experts, core.expert_offset) == (128 // world, 0)


@pytest.mark.parametrize(
    "world_size, tp_size, pp_size",
    [(2, 2, 1), (16, 16, 1), (8, 4, 2)],
    ids=["world2", "world16", "world8-pp2"],
)
def test_expert_path_refuses_another_world(dep4, world_size, tp_size, pp_size):
    """The core serves four ranks on one node or eight over two nodes of four, every rank in the
    one attention-DP group: another world size has no expert call size, and a group short of the
    world (two pipeline stages of four) is not the expert path's communicator. Both are refused
    before anything is built."""
    mapping = Mapping(
        world_size=world_size,
        rank=0,
        gpus_per_node=_GPUS_PER_NODE,
        tp_size=tp_size,
        pp_size=pp_size,
        moe_tp_size=1,
        moe_ep_size=tp_size,
        enable_attention_dp=True,
    )
    assert len(mapping.tp_group) == tp_size
    config = dataclasses.replace(dep4.model_config, mapping=mapping)
    with torch.device("meta"), pytest.raises(ValueError, match="serves 4 ranks on one node or 8"):
        M.MistralLarge4Dep4ModelingV2(config)


def test_kv_host_tier_is_shared_by_the_node_four_ways():
    """At eight ranks over two nodes of four (dep8), the V2 KV manager provisions a host tier per
    rank (no host_cache_size is set): its device quota, capped at half the node's free physical
    pages divided by the ranks on the node -- ``Distributed.local_world_size``, the size of the MPI
    communicator split by host, four at dep8 -- then the minimum over all eight ranks. So a dep8
    node's four tiers together hold at most half the host memory free when the pool is built."""
    gib = 2**30
    quota = 160 * gib
    assert _compute_auto_host_tier_quota(quota, 4, 800 * gib, float("inf")) == 100 * gib
    assert _compute_auto_host_tier_quota(quota, 4, 2000 * gib, float("inf")) == quota
    assert _compute_auto_host_tier_quota(quota, 4, float("inf"), float("inf")) == quota


# ------------------------------------------------------- (5) the four-rank engines, and dep8


def test_tep4_and_dep4_route_as_before(tep4, dep4):
    """The NVFP4 checkpoint at tep4 and at dep4 resolves and routes: tep4 as before this target,
    dep4 to it, each claimed by its own target, on one node of four, the routed experts
    expert-parallel four ways and not split along their width, each on the checkpoint's anchor
    and the same 47136-byte token."""
    cases = (
        (tep4, False, "ModelingV2MistralLarge4Nvfp4Sm103Tep4"),
        (dep4, True, _DEP4_TARGET),
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
    assert (tep4.max_num_tokens, tep4.max_seq_len, tep4.max_batch_size) == (32768, 32768, 16)
    assert (dep4.max_num_tokens, dep4.max_seq_len, dep4.max_batch_size) == (32768, 32768, 16)


def test_dep4_differs_from_tep4_in_attention_dp_only(dep4, tep4):
    """The same checkpoint at dep4 and at tep4 resolves to the same configuration -- quantization,
    per-layer quantization, model config, DSA config, limits, CUDA graphs, KV config, the MoE
    config, what the router reads -- except the mapping's attention-DP bit, and the tep4 branch
    of the routing does not claim dep4."""
    assert _summary(dep4) == _summary(tep4)
    dep4_mapping, tep4_mapping = dep4.mapping.to_dict(), tep4.mapping.to_dict()
    assert {k for k in dep4_mapping if dep4_mapping[k] != tep4_mapping[k]} == {
        "enable_attention_dp"
    }

    target = _route_or_none(dep4.routed[0])
    assert target is None or target.__name__ != "ModelingV2MistralLarge4Nvfp4Sm103Tep4"


def test_dep8_differs_from_dep4_in_the_world_only(dep8, dep8_gate, dep4):
    """The same checkpoint at dep8 and at dep4 resolves to the same configuration --
    quantization, per-layer quantization, model config, DSA config, limits, CUDA graphs, KV
    config, the MoE config, what the router reads -- except the mapping's world: its size, the TP
    and attention-TP widths and the EP width (and the MoE token budget that reads them). The tep4
    branch of the routing does not claim dep8: the dep4 target does, from core.yaml and from the
    boot gate's command alike, the same class as at four ranks."""
    assert _summary(dep8) == _summary(dep4)
    dep8_mapping, dep4_mapping = dep8.mapping.to_dict(), dep4.mapping.to_dict()
    assert {k for k in dep8_mapping if dep8_mapping[k] != dep4_mapping[k]} == {
        "world_size",
        "tp_size",
        "attn_tp_size",
        "moe_ep_size",
    }
    for resolved in (dep8, dep8_gate):
        seen = resolved.routed[0]
        target = routing.route(seen.pretrained_config, seen.mapping, _SM103, None)
        assert target.__name__ == _DEP4_TARGET
