# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the bit-identical target-side fast paths together compute the built-in decoder's bits.

TP8, real weights.

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes and the
checkpoint, under the serving switches, e.g.::

    MUSESPARK_REFERENCE_CKPT=<Muse-Spark-Open-816B-A42B-NVFP4 checkpoint> \\
    MUSESPARK_OPEN_ALLREDUCE=MNNVL MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL=1 \\
    MUSESPARK_OPEN_SHARDED_PROJECTIONS=1 TRTLLM_ENABLE_PDL=1 \\
    srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>

Anywhere else every test skips.

The text model is built truncated to the checkpoint's first six layers (layer 0
BF16, layers 1..5 NVFP4; the six-layer schedule keeps the full model's NoPE
layers 1 and 5) and loaded with those layers' real weights. One copy is built
per arm: the built-in model, the target's core under ``NO_PATHS``, each of the
bit-identical layer paths alone, and all of them. Every arm runs the same steps:
a prefill of M requests (the built-in path in every arm), then decode steps of
M rows -- M requests of one token each, M = 1..8, the row counts of a decode
step with up to 7 drafts -- each fed real embedding rows, once eagerly and
once replayed from a CUDA graph captured per arm (the serving mode, with the
metadata and inputs updated in place before every replay). Every arm's final
output and every layer's outputs (the FP32 residual and the next layer's
attention input) must equal the built-in model's in the same mode as bits;
the built-in model run twice must agree with itself, and its outputs must be
finite. The paths' entry points are counted on the decode steps (on the
capture, in graph mode), so a planned path that silently fell back to the
built-in path is caught too.

``MODELING_V2_COMPOSE_OUT=<dir>`` also writes rank 0's per-arm, per-row-count
result table there as JSON.
"""

import json
import os
import types

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)

WORLD = 8
#: Layers of the truncated model: the checkpoint's layers 0..5.
LAYERS = 6
#: Decode steps after the prefill.
STEPS = 3
TOKENS_PER_BLOCK = 32
MAX_ROWS = 8
#: Real embedding rows the inputs are drawn from.
TOKEN_POOL = 48
#: The serving switches the features need at construction.
REQUIRED_ENV = {
    "MUSESPARK_OPEN_ALLREDUCE": "MNNVL",
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL": "1",
    "MUSESPARK_OPEN_SHARDED_PROJECTIONS": "1",
}
#: (arm, the fast paths it is adopted with); None is the built-in model, not adopted.
ARMS = (
    ("builtin", None),
    ("off", ()),
    ("preattn_prep", ("preattn_prep",)),
    ("post_attention_hop", ("post_attention_hop",)),
    ("projection_gathers", ("projection_gathers",)),
    ("full", ("preattn_prep", "post_attention_hop", "projection_gathers")),
)
#: Entry point -> the fast paths that run it.
ENTRIES = {
    "preattn_prep": {"preattn_prep"},
    "post_attn_hop": {"post_attention_hop"},
    "unpool_gather_boundary": {"projection_gathers"},
    "pool_gather_fp4quant": {"projection_gathers"},
}

_RESULTS: dict = {}
_REFERENCE: dict = {}
_COUNTS: dict = {}


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
    checkpoint = os.environ.get("MUSESPARK_REFERENCE_CKPT", "")
    if not os.path.isfile(os.path.join(checkpoint, "model.safetensors.index.json")):
        pytest.skip(
            "needs MUSESPARK_REFERENCE_CKPT=<the Muse-Spark-Open-816B-A42B-NVFP4 checkpoint>"
        )
    missing = [f"{k}={v}" for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
    if missing:
        pytest.skip(f"needs the serving switches: {' '.join(missing)}")
    return comm, rank, checkpoint


class _Lazy:
    """One checkpoint tensor, read on first use per index and kept on the host for the next copy."""

    def __init__(self, handle, key):
        self._handle = handle
        self._key = key
        self._read = {}

    def get_shape(self):
        return self._handle.get_slice(self._key).get_shape()

    def __getitem__(self, index):
        tag = repr(index)
        if tag not in self._read:
            self._read[tag] = self._handle.get_slice(self._key)[index]
        return self._read[tag].clone()


def _checkpoint_weights(checkpoint: str):
    """Every key of the checkpoint, lazily, renamed to the source spelling; and the rename flag."""
    from safetensors import safe_open

    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        rename_open_format_weights,
    )

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    handles = {}
    weights = {}
    for key, shard in weight_map.items():
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(checkpoint, shard), framework="pt")
        weights[key] = _Lazy(handles[shard], key)
    open_format = rename_open_format_weights(weights)
    return weights, open_format


def _model_config(checkpoint: str, mapping):
    """The checkpoint's model config, truncated to its first ``LAYERS`` layers."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.functional import AllReduceStrategy

    config = ModelConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        mapping=mapping,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
        allreduce_strategy=AllReduceStrategy.NCCL,
        max_num_tokens=8192,
    )
    pretrained = config.pretrained_config
    full = pretrained.num_hidden_layers
    # The per-layer schedule (window, RoPE) is computed for the full model; keep its head.
    for name, value in list(vars(pretrained).items()):
        if isinstance(value, list) and len(value) == full:
            setattr(pretrained, name, value[:LAYERS])
    pretrained.num_hidden_layers = LAYERS
    return config


class _Counted:
    """A callable that counts its calls into ``_COUNTS[name]`` and calls through."""

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args, **kwargs):
        _COUNTS[self.name] = _COUNTS.get(self.name, 0) + 1
        return self.fn(*args, **kwargs)

    def __getattr__(self, name):
        # Everything else (``router_gemv.precompile``, say) is the wrapped object's.
        return getattr(self.fn, name)


def _build_arm(checkpoint, mapping, weights, open_format, selection, sink):
    """One copy of the truncated model: built, adopted with the paths ``selection``, loaded,
    post-loaded."""
    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
        WeightAccounting,
        load_layer,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenForCausalLM
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    config = _model_config(checkpoint, mapping)
    with torch.device("cuda"):
        model = MuseSparkOpenForCausalLM(config).model
    if selection is not None:
        modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of(*selection))
        if model._decode.preattn_prep is not None:
            model._decode.preattn_prep = _Counted("preattn_prep", model._decode.preattn_prep)
        if model._post_attn_hop is not None:
            model._post_attn_hop.op = _Counted("post_attn_hop", model._post_attn_hop.op)
    owner = types.SimpleNamespace(config=config.pretrained_config)
    accounting = WeightAccounting()
    for index, layer in enumerate(model.layers):
        load_layer(owner, layer, index, weights, accounting, open_format=open_format)
    with torch.no_grad():
        model.norm.weight.copy_(weights["model.norm.weight"][:].to(model.norm.weight.dtype))
    ModelLoader._walk_full_post_load(model)

    def capture(module, args, output):
        outputs = output if isinstance(output, tuple) else (output,)
        sink.append([t.clone() for t in outputs])

    for layer in model.layers:
        layer.register_forward_hook(capture)
    return model, config


@pytest.fixture(scope="module")
def env():
    """Every arm's model copy, the KV-cache geometry, and the real embedding rows (collective)."""
    comm, rank, checkpoint = _world()
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
        gathers as projection_gathers,
    )
    from tensorrt_llm.mapping import Mapping

    e = types.SimpleNamespace(comm=comm, rank=rank, sink=[], models={})
    e.mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=4,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
    )
    weights, open_format = _checkpoint_weights(checkpoint)
    # Module-level entry points, counted for every arm (the call sites look them up by name).
    patched = [
        (projection_gathers._unpool, "unpool_gather_boundary", "unpool_gather_boundary"),
        (projection_gathers._pool, "pool_gather_rmsnorm_fp4quant", "pool_gather_fp4quant"),
    ]
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patched]
    for module, attr, name in patched:
        setattr(module, attr, _Counted(name, getattr(module, attr)))
    try:
        for arm, selection in ARMS:
            e.models[arm], config = _build_arm(
                checkpoint, e.mapping, weights, open_format, selection, e.sink
            )
            comm.Barrier()
        e.pretrained = config.pretrained_config
        embed = weights["model.embed_tokens.weight"]
        vocab = embed.get_shape()[0]
        generator = torch.Generator().manual_seed(2026)
        tokens = torch.randint(0, vocab, (TOKEN_POOL,), generator=generator).tolist()
        e.table = torch.cat([embed[t : t + 1] for t in tokens]).to("cuda", torch.bfloat16)
        torch.cuda.synchronize()
        comm.Barrier()
        yield e
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        for module, attr, original in originals:
            setattr(module, attr, original)


def _kv_cache(e):
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    return KVCacheManager(
        KvCacheConfig(max_tokens=2 * MAX_ROWS * TOKENS_PER_BLOCK, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=LAYERS,
        num_kv_heads=e.pretrained.num_key_value_heads,
        head_dim=e.pretrained.head_dim,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=2 * TOKENS_PER_BLOCK,
        max_batch_size=MAX_ROWS,
        mapping=e.mapping,
        dtype=DataType.BF16,
    )


def _metadata(e, kv, request_ids, seq_lens, cached, prompt_lens, num_contexts):
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
        kv_cache_manager=kv,
        request_ids=list(request_ids),
        prompt_lens=list(prompt_lens),
        max_num_requests=MAX_ROWS,
        max_num_tokens=8192,
        mapping=e.mapping,
    )
    metadata.prepare()
    return metadata


def _plan(rows: int):
    """Prompt lengths and token ids (indices into ``table``) of a run at ``rows`` rows."""
    prompt_lens = [9 + 2 * i for i in range(rows)]
    generator = torch.Generator().manual_seed(rows)
    prompts = [torch.randint(0, TOKEN_POOL, (n,), generator=generator) for n in prompt_lens]
    decode = [torch.randint(0, TOKEN_POOL, (rows,), generator=generator) for _ in range(STEPS)]
    return prompt_lens, prompts, decode


def _collect(out: dict, prefix: str, layer_outputs) -> None:
    for index, layer_out in enumerate(layer_outputs):
        for part, tensor in enumerate(layer_out):
            out[f"{prefix}/layer{index}.{part}"] = tensor.clone()


@torch.inference_mode()
def _run(e, model, rows: int, graph: bool = False):
    """Prefill, then ``STEPS`` decode steps at ``rows`` rows; every output and the decode counts.

    With ``graph`` the decode steps replay one CUDA graph, captured after an
    eager warm-up as the engine captures its decode graphs, with the attention
    metadata and the inputs updated in place before every replay; the counts
    are then those of the capture.
    """
    from tensorrt_llm._torch.metadata import KVCacheParams

    prompt_lens, prompts, decode = _plan(rows)
    request_ids = list(range(rows))
    kv = _kv_cache(e)
    out = {}
    cuda_graph = None
    try:
        kv.add_dummy_requests(request_ids, [n + STEPS for n in prompt_lens])
        metadata = _metadata(
            e, kv, request_ids, prompt_lens, [0] * rows, prompt_lens, num_contexts=rows
        )
        position_ids = torch.cat([torch.arange(n) for n in prompt_lens]).int().cuda()[None]
        e.sink.clear()
        out["prefill"] = model(
            attn_metadata=metadata,
            position_ids=position_ids,
            inputs_embeds=e.table[torch.cat(prompts).cuda()],
        ).clone()
        _collect(out, "prefill", e.sink)
        _COUNTS.clear()
        if not graph:
            for step in range(STEPS):
                cached = [n + step for n in prompt_lens]
                metadata = _metadata(e, kv, request_ids, [1] * rows, cached, prompt_lens, 0)
                e.sink.clear()
                out[f"decode{step}"] = model(
                    attn_metadata=metadata,
                    position_ids=torch.tensor([cached], dtype=torch.int, device="cuda"),
                    inputs_embeds=e.table[decode[step].cuda()],
                ).clone()
                _collect(out, f"decode{step}", e.sink)
            counts = dict(_COUNTS)
        else:
            metadata = _metadata(e, kv, request_ids, [1] * rows, prompt_lens, prompt_lens, 0)
            metadata = metadata.create_cuda_graph_metadata(rows)
            metadata.prepare()
            position_ids = torch.tensor([prompt_lens], dtype=torch.int, device="cuda")
            embeds = e.table[decode[0].cuda()].clone()

            def forward():
                return model(
                    attn_metadata=metadata, position_ids=position_ids, inputs_embeds=embeds
                )

            forward()  # warm-up: compiles and allocates outside the capture
            torch.cuda.synchronize()
            e.comm.Barrier()
            _COUNTS.clear()
            e.sink.clear()
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                static_out = forward()
            counts = dict(_COUNTS)
            static_layers = list(e.sink)
            torch.cuda.synchronize()
            e.comm.Barrier()
            for step in range(STEPS):
                cached = [n + step for n in prompt_lens]
                metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=cached
                )
                metadata.prepare()
                position_ids.copy_(torch.tensor([cached], dtype=torch.int))
                embeds.copy_(e.table[decode[step].cuda()])
                cuda_graph.replay()
                out[f"decode{step}"] = static_out.clone()
                _collect(out, f"decode{step}", static_layers)
        torch.cuda.synchronize()
    finally:
        e.sink.clear()
        del cuda_graph
        kv.shutdown()
    return out, counts


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    return tensor.view({4: torch.int32, 2: torch.int16, 1: torch.int8}[tensor.element_size()])


def _differences(got: dict, want: dict) -> list:
    """Each output of ``got`` that is not ``want``'s, as bits, in step order."""
    found = []
    for name, reference in want.items():
        tensor = got.get(name)
        if tensor is None:
            found.append(f"{name}: missing")
        elif tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            found.append(
                f"{name}: {tuple(tensor.shape)} {tensor.dtype} vs "
                f"{tuple(reference.shape)} {reference.dtype}"
            )
        else:
            differ = int((_bits(tensor) != _bits(reference)).sum())
            if differ:
                delta = (tensor.float() - reference.float()).abs().max().item()
                found.append(f"{name}: {differ}/{tensor.numel()} differ, max |d| {delta:.3e}")
    extra = sorted(set(got) - set(want))
    if extra:
        found.append(f"unexpected outputs {extra[:4]}")
    return found


def _compare_arms(e, rows: int, mode: str) -> None:
    """Every arm against the built-in model at ``rows`` rows in ``mode`` (collective)."""
    graph = mode == "graph"
    reference, _ = _run(e, e.models["builtin"], rows, graph)
    again, _ = _run(e, e.models["builtin"], rows, graph)
    report = {
        "builtin (second run)": _differences(again, reference),
        "builtin (finite)": [
            f"{name}: not finite" for name, t in reference.items() if not torch.isfinite(t).all()
        ],
    }
    _REFERENCE[(mode, rows)] = {
        name: round(t.float().abs().max().item(), 4)
        for name, t in reference.items()
        if "/" not in name
    }
    for arm, _selection in ARMS[1:]:
        got, counts = _run(e, e.models[arm], rows, graph)
        report[arm] = _differences(got, reference)
        _RESULTS[(mode, arm, rows)] = {"differences": len(report[arm]), "counts": counts}
        e.comm.Barrier()
    everyone = e.comm.allgather({arm: found for arm, found in report.items() if found})
    failed = {
        f"rank {rank} {arm}": found[:6]
        for rank, bad in enumerate(everyone)
        for arm, found in bad.items()
    }
    assert not failed, f"{mode} outputs that are not the built-in model's at {rows} rows: {failed}"


@pytest.mark.parametrize("rows", list(range(1, MAX_ROWS + 1)))
def test_every_arm_is_the_builtin_decoder_bit_for_bit(env, rows):
    """Prefill and eager decode outputs of every arm, every layer, against the built-in model's bits."""
    _compare_arms(env, rows, "eager")


@pytest.mark.parametrize("rows", list(range(1, MAX_ROWS + 1)))
def test_every_arm_is_the_builtin_decoder_bit_for_bit_in_cuda_graphs(env, rows):
    """The same with the decode steps replayed from one CUDA graph per arm (the serving mode)."""
    _compare_arms(env, rows, "graph")


def test_every_selected_feature_ran_on_the_decode_steps(env):
    """Each arm's decode steps ran the entry points of exactly its planned paths, in both modes."""
    e = env
    selections = dict(ARMS)
    wrong = []
    for (mode, arm, rows), result in sorted(_RESULTS.items()):
        selected = set(selections[arm] or ())
        # Eager: once per decode step at least; graph: in the one capture.
        least = STEPS if mode == "eager" else 1
        for entry, paths in ENTRIES.items():
            calls = result["counts"].get(entry, 0)
            if bool(selected & paths) != (calls >= least):
                wrong.append(f"{mode} {arm} at {rows} rows: {entry} ran {calls} times")
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and e.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        table = {
            f"{mode}:{arm}@{rows}": result for (mode, arm, rows), result in sorted(_RESULTS.items())
        }
        table["reference_abs_max"] = {
            f"{mode}@{rows}": values for (mode, rows), values in sorted(_REFERENCE.items())
        }
        with open(os.path.join(out_dir, "compose_layers.json"), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
    assert _RESULTS, "the bitwise test recorded nothing"
    assert not wrong, "; ".join(wrong)
