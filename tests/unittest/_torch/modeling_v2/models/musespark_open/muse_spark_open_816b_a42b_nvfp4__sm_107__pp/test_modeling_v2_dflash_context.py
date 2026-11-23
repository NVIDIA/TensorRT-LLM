# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 PP target with a DFlash drafter, on the host only (CPU torch, no device).

The target package is loaded from its files under a test package name, with the leaf
modules it imports stubbed for the load (the router index, the communicator, the logger,
the built-in model and shell, the quantization enum, the DFlash worker). The TEP target's
lever resolver and dispatch are loaded from their files under their real names, so the
lever table is the real one. The tests pin:

* the composition table: role x speculative config -> served or not, the drafter context
  the stages agree on (its arena is paged), and a stage whose drafter differs splitting the
  decision; the lever table with the paged arena is the drafterless one;
* the context-step stage forward (``carry.CaptureCarry``) over the four stages of the
  ``[15, 16, 16, 15]`` split, run one after another with a queue for the transport: the
  message shapes and order per edge, and the last stage's capture buffer, bit for bit, against the
  residuals each sending stage dumped for its own target layers (keyed by layer id), with
  every stage's capture buffer NaN-filled before the step;
* the shell constructor: the built-in model is built inside the
  ``pp_dflash_context_server()`` scope only with a drafter, ``PrefillTarget`` carries the
  captures and ``DecodeTarget`` keeps the built-in two-tensor boundary, and the last stage's
  worker is re-classed;
* the context-server worker: a step with generation rows runs no drafter forward, a
  context-only step is the built-in step;
* the drafter reduction: what is kept, what is released, and that the context K/V the
  reduced drafter computes is bitwise the full drafter's.
"""

from __future__ import annotations

import collections
import contextlib
import enum
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_MODELS = _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
_TARGET = _MODELS / "muse_spark_open_816b_a42b_nvfp4__sm_107__pp"
_TEP = _MODELS / "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
_TEP_PACKAGE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
_PACKAGE = "_mso_pp_dflash_under_test"

NUM_LAYERS, HIDDEN, VOCAB = 62, 16, 32
TARGET_LAYERS = (1, 16, 30, 44, 59)
SPLIT_15_16_16_15 = [15, 16, 16, 15]
MAX_ROWS = 12
DTYPE = torch.bfloat16


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------
class _Mode(str, enum.Enum):
    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"


class _Log:
    def __init__(self) -> None:
        self.records: list = []

    def info(self, message, *args, **kwargs) -> None:
        self.records.append(("info", str(message)))

    def warning(self, message, *args, **kwargs) -> None:
        self.records.append(("warning", str(message)))

    def debug(self, message, *args, **kwargs) -> None:
        self.records.append(("debug", str(message)))

    def error(self, message, *args, **kwargs) -> None:
        self.records.append(("error", str(message)))


class _Group:
    """The PP group: this stage's item, and each other stage's as ``others`` makes it."""

    def __init__(self) -> None:
        self.world = 1
        self.others = None

    def pp_allgather(self, item) -> list:
        others = self.others or (lambda x: x)
        return [item] + [others(item) for _ in range(self.world - 1)]

    def tp_allgather(self, items, small_payload: bool = False) -> list:
        return [list(items)]


class _Distributed:
    group = _Group()

    @classmethod
    def get(cls, mapping):
        cls.group.world = int(getattr(mapping, "pp_size", 1))
        return cls.group


class _QuantAlgo(enum.Enum):
    NVFP4 = "NVFP4"


class _Transport:
    """``pp_send_tensors`` / ``pp_recv_tensors``: one queue per edge, a stage at a time."""

    def __init__(self) -> None:
        self.queues: dict = {}
        self.sent: dict = {}
        self.stage = 0

    def send(self, tensors) -> None:
        self.queues.setdefault(self.stage, []).append([t.clone() for t in tensors])
        self.sent.setdefault(self.stage, []).append([(tuple(t.shape), t.dtype) for t in tensors])

    def recv(self, tensors) -> None:
        message = self.queues[self.stage - 1].pop(0)
        assert [(tuple(t.shape), t.dtype) for t in tensors] == [
            (tuple(m.shape), m.dtype) for m in message
        ], "the two ends of an edge disagree on the message"
        for tensor, value in zip(tensors, message):
            tensor.copy_(value)


TRANSPORT = _Transport()
LOG = _Log()
SCOPE = {"depth": 0, "entered": 0}


@contextlib.contextmanager
def _pp_dflash_context_server():
    SCOPE["depth"] += 1
    SCOPE["entered"] += 1
    try:
        yield
    finally:
        SCOPE["depth"] -= 1


class _BuiltinModel(nn.Module):
    """``MuseSparkOpenModel``'s stage contract: ``forward`` -> ``_forward_pipeline_stage``,
    whose boundary is the residual and the attention input."""

    def __init__(self, stage_range, layers, pretrained, mapping) -> None:
        super().__init__()
        self._pp_stage = stage_range
        self.layers = layers
        self.model_config = types.SimpleNamespace(pretrained_config=pretrained, mapping=mapping)
        self.dtype = DTYPE
        self.m_emb = 1.0
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.embed_norm = nn.Identity()
        self.norm = nn.Identity()
        self.builtin_calls = 0

    def forward(self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None, **kw):
        return self._forward_pipeline_stage(
            attn_metadata, input_ids, position_ids, inputs_embeds, **kw
        )

    def _forward_pipeline_stage(self, attn_metadata, input_ids, position_ids, inputs_embeds, **kw):
        self.builtin_calls += 1
        mapping = self.model_config.mapping
        rows = input_ids.shape[0]
        hidden = torch.zeros(rows, HIDDEN, dtype=torch.float32)
        attn = torch.zeros(rows, HIDDEN, dtype=DTYPE)
        if not mapping.is_first_pp_rank():
            TRANSPORT.recv([hidden, attn])
        if not mapping.is_last_pp_rank():
            TRANSPORT.send([hidden, attn])
        return hidden


class _Omni(nn.Module):
    """The built-in shell: builds ``llm`` (the stage text model, the drafter, the worker)."""

    built: list = []

    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__()
        _Omni.built.append(SCOPE["depth"])
        self.model_config = model_config
        mapping = model_config.mapping
        stage = mapping.pp_rank
        ranges = _STAGES.stage_ranges(NUM_LAYERS, mapping.pp_size, mapping.pp_partition)
        layers = nn.ModuleList(_StubLayer(i) for i in range(NUM_LAYERS))
        text = _BuiltinModel(ranges[stage], layers, model_config.pretrained_config, mapping)
        worker = _StubWorker() if model_config.spec_config is not None else None
        self.llm = types.SimpleNamespace(model=text, spec_worker=worker, draft_model=None)


class _StubWorkerBase:
    """``DFlashWorker``: the built-in step, recorded."""

    def __init__(self) -> None:
        self.calls: list = []

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        self.calls.append("builtin")
        if attn_metadata.num_seqs > attn_metadata.num_contexts:
            draft_model.dflash_forward()
        return {"builtin": True}

    def _execute_guided_decoder_if_present(self, logits) -> None:
        self.calls.append("guided")

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        batch = attn_metadata.num_seqs
        k = spec_metadata.runtime_draft_len
        accepted = torch.arange(batch * (k + 1), dtype=torch.int32).view(batch, k + 1)
        return accepted, torch.ones(batch, dtype=torch.int32)

    def write_context_onehot_draft_probs(
        self, spec_metadata, num_contexts, num_gens, k, vocab=None
    ):
        self.calls.append(("onehot", num_contexts, num_gens, k))

    def _rollback_guided_decoder_after_verify(self, num_accepted) -> None:
        self.calls.append("rollback")

    def _prepare_next_new_tokens(self, accepted, drafts, batch_indices, batch_size, num_accepted):
        rows = accepted[batch_indices[:batch_size].long(), num_accepted.long() - 1].unsqueeze(1)
        return torch.cat([rows, drafts], dim=1)

    def _store_prefill_context(self, *args):
        self.calls.append("builtin_store")

    def _store_context_kv_paged(self, *args):
        self.calls.append("builtin_paged_store")


class _StubWorker(_StubWorkerBase):
    pass


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _load():
    communicator = _module(
        "communicator",
        Distributed=_Distributed,
        pp_send_tensors=TRANSPORT.send,
        pp_recv_tensors=TRANSPORT.recv,
        pp_comm_buffered=lambda: False,
        _pp_comm=None,
    )
    router_index = _module(
        "_router_index",
        ModelingV2Mode=_Mode,
        resolved_mode=lambda cls: _Mode.AUTO,
        resolved_facts=lambda cls: None,
    )
    pp_buffered = _module("pp_buffered", TENSOR_ALIGN=256)
    stubs = {
        "tensorrt_llm._torch._experimental.modeling_v2._router_index": router_index,
        "tensorrt_llm._torch.distributed": _module(
            "distributed", communicator=communicator, pp_buffered=pp_buffered
        ),
        "tensorrt_llm._torch.distributed.communicator": communicator,
        "tensorrt_llm._torch.distributed.pp_buffered": pp_buffered,
        "tensorrt_llm.logger": _module("logger", logger=LOG),
        "tensorrt_llm.quantization.mode": _module("mode", QuantAlgo=_QuantAlgo),
        "tensorrt_llm._torch.attention.backends.interface": _module(
            "interface", AttentionMetadata=object
        ),
        "tensorrt_llm._torch.model_config": _module("model_config", ModelConfig=object),
        "tensorrt_llm._torch.models.modeling_musespark_open": _module(
            "modeling_musespark_open",
            MuseSparkOpenModel=_BuiltinModel,
            pp_dflash_context_server=_pp_dflash_context_server,
            _fused_preattention_enabled=lambda: False,
        ),
        "tensorrt_llm._torch.models.modeling_musespark_open_omni": _module(
            "modeling_musespark_open_omni", MuseSparkOpenOmniForConditionalGeneration=_Omni
        ),
        "tensorrt_llm._torch.flashinfer_utils": _module(
            "flashinfer_utils", get_env_enable_pdl=lambda: True
        ),
        "tensorrt_llm._torch.speculative.dflash": _module("dflash", DFlashWorker=_StubWorkerBase),
    }
    sys.modules.update(stubs)
    tep = types.ModuleType(_TEP_PACKAGE)
    tep.__path__ = [str(_TEP)]
    sys.modules[_TEP_PACKAGE] = tep
    importlib.import_module(f"{_TEP_PACKAGE}.levers")
    importlib.import_module(f"{_TEP_PACKAGE}.dispatch")
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_TARGET)]
    sys.modules[_PACKAGE] = package
    names = (
        "stages",
        "dispatch",
        "modeling",
        "kernels.dflash_context.route",
        "kernels.dflash_context.carry",
        "kernels.dflash_context.worker",
        "kernels.dflash_context.slim",
        "kernels.dflash_context.install",
    )
    return [importlib.import_module(f"{_PACKAGE}.{name}") for name in names]


(_STAGES, DISPATCH, MODELING, ROUTE, CARRY, WORKER, SLIM, INSTALL) = _load()
LEVERS = sys.modules[f"{_TEP_PACKAGE}.levers"]


# --------------------------------------------------------------------------
# Model stand-ins
# --------------------------------------------------------------------------
class _Norm(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual * self.scale).to(DTYPE)


class _StubLayer(nn.Module):
    """One residual update per layer with the built-in layer's call contract and DFlash tap."""

    dumps: dict = {}

    def __init__(self, index: int) -> None:
        super().__init__()
        self.layer_idx = index
        self.pre_qkv_proj_norm = _Norm(1.0 + index / 64)

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        **kwargs,
    ):
        attn = (
            precomputed_attn_input
            if precomputed_attn_input is not None
            else self.pre_qkv_proj_norm(hidden_states)
        )
        hidden_states = hidden_states * 0.75 + attn.float() * 0.5 + (self.layer_idx + 1) / 7
        _StubLayer.dumps[self.layer_idx] = hidden_states.clone()
        spec_metadata = kwargs.get("spec_metadata")
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, None)
        if next_layer_norm is None:
            return hidden_states
        return hidden_states, next_layer_norm(hidden_states)


class _SpecMetadata:
    """``DFlashSpecMetadata``'s capture buffer, layer map and tap (the tap is
    ``inplace_slice_copy``: contiguous 2-D destination, rows from the source)."""

    def __init__(self, layers=TARGET_LAYERS, max_rows=MAX_ROWS, hidden=HIDDEN) -> None:
        self.layers_to_capture = sorted(layers)
        self.hidden_size = hidden
        self.num_capture_layers = len(layers)
        self._layer_to_idx = {layer: i for i, layer in enumerate(self.layers_to_capture)}
        self.captured_hidden_states = torch.full(
            (max_rows, hidden * len(layers)), float("nan"), dtype=DTYPE
        )
        self.runtime_draft_len = 3

    def maybe_capture_hidden_states(self, layer_id, hidden_states, residual=None) -> None:
        i = self._layer_to_idx.get(layer_id)
        if i is None:
            return
        dest = self.captured_hidden_states
        src = hidden_states.to(dest.dtype).contiguous()
        assert dest.is_contiguous() and dest.dim() == 2 and src.shape[0] <= dest.shape[0]
        dest[: src.shape[0], i * self.hidden_size : (i + 1) * self.hidden_size].copy_(src)

    def get_hidden_states(self, num_tokens: int) -> torch.Tensor:
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class _Mapping:
    def __init__(self, stage: int, pp_size: int = 4, partition=SPLIT_15_16_16_15) -> None:
        self.pp_rank = stage
        self.pp_size = pp_size
        self.tp_size = 1
        self.gpus_per_node = 4
        self.pp_partition = partition
        self.pp_group = list(range(pp_size))
        self.rank = stage

    def is_first_pp_rank(self) -> bool:
        return self.pp_rank == 0

    def is_last_pp_rank(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    def has_pp(self) -> bool:
        return self.pp_size > 1


def _spec(decoding="DFlash", backend="TRTLLM", layers=TARGET_LAYERS, draft_len=7):
    return types.SimpleNamespace(
        decoding_type=decoding,
        attention_backend=backend,
        target_layer_ids=list(layers) if layers is not None else None,
        max_draft_len=draft_len,
    )


def _config(stage=0, spec=None, partition=SPLIT_15_16_16_15, max_num_tokens=8192):
    pretrained = types.SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        every_n_layers_nope=4,
        hidden_size=HIDDEN,
        num_attention_heads=128,
        num_key_value_heads=16,
        torch_dtype=DTYPE,
    )
    return types.SimpleNamespace(
        pretrained_config=pretrained,
        mapping=_Mapping(stage, partition=partition),
        spec_config=spec,
        quant_config=types.SimpleNamespace(quant_algo=_QuantAlgo.NVFP4),
        torch_dtype=DTYPE,
        max_num_tokens=max_num_tokens,
    )


CTX = {LEVERS.ROLE_KNOB: "ctx"}
SM107 = (10, 7)


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "role, spec, served",
    [
        ("ctx", None, True),
        ("ctx", _spec(), True),
        ("ctx", _spec(backend="VANILLA"), False),
        ("ctx", _spec(backend="AUTO"), False),
        ("ctx", _spec(backend="FA4"), False),
        ("ctx", _spec(decoding="Eagle3"), False),
        ("ctx", _spec(decoding="MTP"), False),
        ("ctx", _spec(layers=None), False),
        ("ctx", _spec(layers=(1, 62)), False),
        ("ctx", _spec(draft_len=0), False),
        ("gen", _spec(), False),
        ("gen", None, False),
        ("auto", _spec(), False),
    ],
)
def test_composition_table(role, spec, served):
    config = _config(spec=spec)
    snapshot = {LEVERS.ROLE_KNOB: role}
    got = DISPATCH.composition(config, MODELING.TARGET_PATHS, SM107, snapshot)
    assert (got is not None) == served
    decision = DISPATCH.agreed_composition(config, MODELING.TARGET_PATHS, SM107, snapshot)
    assert (decision.served is not None) == served
    if served and spec is not None:
        assert decision.drafter == DISPATCH.DrafterContext(TARGET_LAYERS, 7)
        assert decision.drafter.arena is LEVERS.Arena.PAGED
    else:
        assert decision.drafter is None
    if not served and spec is not None and role == "ctx":
        assert DISPATCH.drafter_refusal(config) in decision.reason


def test_unserved_reason_names_the_drafter_refusal():
    config = _config(spec=_spec(backend="VANILLA"))
    reason = DISPATCH.unserved_reason(config, MODELING.TARGET_PATHS, SM107, CTX)
    assert "attention backend is VANILLA" in reason
    assert "with a DFlash drafter on the TRTLLM attention backend" in reason


def test_stages_with_another_drafter_split_the_decision():
    """The drafter (its arena, target layers and draft length) is part of what the stages agree."""
    config = _config(spec=_spec())
    _Distributed.group.others = lambda key: key.replace("draft length 7", "draft length 3")
    try:
        decision = DISPATCH.agreed_composition(config, MODELING.TARGET_PATHS, SM107, CTX)
    finally:
        _Distributed.group.others = None
    assert decision.served is None and decision.split
    assert "paged arena" in decision.reason


def test_lever_table_with_the_paged_arena_is_the_drafterless_one():
    common = dict(
        compositions=MODELING.TARGET_PATHS,
        sm=107,
        tp=1,
        pp=4,
        heads=(128, 16),
        max_batch_size=None,
        role="ctx",
        max_num_tokens=8192,
    )
    none = DISPATCH.table_for(**common)
    paged = DISPATCH.table_for(**common, arena="paged")
    assert paged.paths() == none.paths()
    assert paged.paths() <= MODELING.TARGET_PATHS[107, 1].served
    assert {n: d.on for n, d in paged.decisions.items()} == {
        n: d.on for n, d in none.decisions.items()
    }
    assert paged.deployment.arena is LEVERS.Arena.PAGED
    # Every served lever under its knob set on resolves alike under both arenas.
    knobs = {LEVERS.knob_name(name): "on" for name in MODELING.TARGET_PATHS[107, 1].served}
    assert (
        DISPATCH.table_for(**common, knobs=knobs, arena="paged").paths()
        == DISPATCH.table_for(**common, knobs=knobs).paths()
    )


def test_deployment_arena():
    config = _config(spec=_spec())
    drafter = DISPATCH.drafter_of(config)
    with_drafter = DISPATCH.deployment_of(config, SM107, None, CTX, drafter)
    without = DISPATCH.deployment_of(config, SM107, None, CTX)
    assert with_drafter.arena is LEVERS.Arena.PAGED
    assert without.arena is LEVERS.Arena.NONE
    with pytest.raises(ValueError):
        DISPATCH.drafter_of(_config(spec=_spec(backend="VANILLA")))


# --------------------------------------------------------------------------
# The context-step stage forward over four stages
# --------------------------------------------------------------------------
def _core(stage: int, partition=SPLIT_15_16_16_15) -> _BuiltinModel:
    mapping = _Mapping(stage, partition=partition)
    ranges = _STAGES.stage_ranges(NUM_LAYERS, 4, partition)
    layers = nn.ModuleList(_StubLayer(i) for i in range(NUM_LAYERS))
    pretrained = types.SimpleNamespace(hidden_size=HIDDEN)
    torch.manual_seed(0)
    return _BuiltinModel(ranges[stage], layers, pretrained, mapping)


def _route(partition=SPLIT_15_16_16_15, layers=TARGET_LAYERS):
    ranges = _STAGES.stage_ranges(NUM_LAYERS, 4, partition)
    return ROUTE.capture_route(layers, ranges, HIDDEN)


def _run_chain(rows: int, partition=SPLIT_15_16_16_15, layers=TARGET_LAYERS, metadata=True):
    """The four stages of one context step, one after another; returns each stage's
    metadata and output and, per sending stage, the residual dumps of its own layers."""
    route = _route(partition, layers)
    TRANSPORT.queues.clear()
    TRANSPORT.sent.clear()
    input_ids = torch.arange(rows) % VOCAB
    outputs, metas, dumps = [], [], []
    for stage in range(4):
        TRANSPORT.stage = stage
        carry = CARRY.CaptureCarry(route, stage, send=TRANSPORT.send, recv=TRANSPORT.recv)
        core = _core(stage, partition)
        meta = _SpecMetadata(layers) if metadata else None
        _StubLayer.dumps = {}
        kwargs = {"spec_metadata": meta} if metadata else {}
        outputs.append(carry.forward_stage(core, None, input_ids, None, None, **kwargs))
        dumps.append(dict(_StubLayer.dumps))
        metas.append(meta)
    return route, metas, outputs, dumps


@pytest.mark.parametrize("rows", [1, 7, MAX_ROWS])
def test_last_stage_assembles_the_captures_bitwise_from_the_sending_stages(rows):
    route, metas, outputs, dumps = _run_chain(rows)
    step = metas[3].get_hidden_states(rows)
    assert torch.isfinite(step.float()).all(), "a capture column was read before it was written"
    for index, layer in enumerate(TARGET_LAYERS):
        stage = next(s.stage for s in route.stages if layer in route.layers_of(s.own))
        expected = dumps[stage][layer].to(DTYPE)
        got = step[:, index * HIDDEN : (index + 1) * HIDDEN]
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), (layer, stage)
    # Rows past the step's are untouched (still the NaN fill).
    assert torch.isnan(metas[3].captured_hidden_states[rows:].float()).all()


def test_message_shapes_and_order_per_edge():
    rows = 5
    route, _, _, _ = _run_chain(rows)
    for stage in range(3):
        (message,) = TRANSPORT.sent[stage]
        expected = [
            (shape, torch.float32 if i == 0 else DTYPE)
            for i, (shape, _) in enumerate(route.edge_tensors(stage, rows))
        ]
        assert message == expected
    assert [len(TRANSPORT.sent[s][0]) for s in range(3)] == [3, 4, 5]


def test_sending_stages_carry_inside_their_capture_buffer_and_restore_its_map():
    rows = 6
    _, metas, _, dumps = _run_chain(rows)
    for stage in range(3):
        meta = metas[stage]
        # The buffer object and the layer map are the metadata's own again after the step.
        assert meta._layer_to_idx == {layer: i for i, layer in enumerate(TARGET_LAYERS)}
        assert meta.captured_hidden_states.shape == (MAX_ROWS, HIDDEN * len(TARGET_LAYERS))
    # Stage 1 taps layers 16 and 30 into its group, layer-major from capture index 1.
    flat = metas[1].captured_hidden_states.view(-1)
    start = 1 * MAX_ROWS * HIDDEN
    group = flat[start : start + rows * 2 * HIDDEN].view(rows, 2 * HIDDEN)
    assert torch.equal(group[:, :HIDDEN], dumps[1][16].to(DTYPE))
    assert torch.equal(group[:, HIDDEN:], dumps[1][30].to(DTYPE))


def test_stage_forward_without_metadata_keeps_the_message_shapes():
    rows = 4
    route, _, outputs, _ = _run_chain(rows, metadata=False)
    assert [len(TRANSPORT.sent[s][0]) for s in range(3)] == [3, 4, 5]
    assert outputs[3].shape == (rows, HIDDEN)


@pytest.mark.parametrize("partition", [[14, 16, 16, 16], [16, 16, 16, 14], [16, 15, 16, 15]])
def test_other_served_splits(partition):
    rows = 3
    route, metas, _, dumps = _run_chain(rows, partition)
    step = metas[3].get_hidden_states(rows)
    for index, layer in enumerate(TARGET_LAYERS):
        stage = next(s.stage for s in route.stages if layer in route.layers_of(s.own))
        assert torch.equal(
            step[:, index * HIDDEN : (index + 1) * HIDDEN], dumps[stage][layer].to(DTYPE)
        )


def test_stage_forward_matches_the_one_stage_residual():
    """The carried forward runs the same per-layer calls: the last stage's output is the
    one-stage forward's final residual."""
    rows = 5
    _, _, outputs, _ = _run_chain(rows)
    core = _core(0)
    hidden = core.embed_tokens(torch.arange(rows) % VOCAB).to(DTYPE).float()
    attn = None
    for index in range(NUM_LAYERS):
        nxt = core.layers[index + 1].pre_qkv_proj_norm if index + 1 < NUM_LAYERS else None
        result = core.layers[index](None, hidden, None, attn, nxt)
        hidden, attn = result if nxt is not None else (result, None)
    assert torch.equal(outputs[3], hidden.to(DTYPE))


def test_capture_buffer_too_small_for_the_step():
    route = _route()
    carry = CARRY.CaptureCarry(route, 0, send=TRANSPORT.send, recv=TRANSPORT.recv)
    with pytest.raises(ValueError, match="exceeds"):
        carry.forward_stage(
            _core(0),
            None,
            torch.zeros(MAX_ROWS + 1, dtype=torch.long),
            None,
            None,
            spec_metadata=_SpecMetadata(),
        )


# --------------------------------------------------------------------------
# The shell: scope, targets, worker
# --------------------------------------------------------------------------
def _attn(num_contexts: int, num_seqs: int):
    return types.SimpleNamespace(num_contexts=num_contexts, num_seqs=num_seqs)


def _shells(spec, monkeypatch):
    monkeypatch.setattr(MODELING, "device_compute_capability", lambda: SM107)
    monkeypatch.setenv(LEVERS.ROLE_KNOB, "ctx")
    # Every lever of the stage off: the shell then installs no fast path, only the drafter
    # context, which no knob reaches.
    for name in DISPATCH.FAST_PATH_NAMES:
        monkeypatch.setenv(LEVERS.knob_name(name), "off")
    _Omni.built.clear()
    shells = []
    for stage in range(4):
        shells.append(MODELING.ModelingV2MuseSparkOpen816bA42bNvfp4Sm107Pp(_config(stage, spec)))
    return shells


def test_shell_with_a_drafter(monkeypatch):
    entered = SCOPE["entered"]
    shells = _shells(_spec(), monkeypatch)
    # Each stage built the built-in model inside the scope, and left it.
    assert _Omni.built == [1, 1, 1, 1] and SCOPE["depth"] == 0
    assert SCOPE["entered"] == entered + 4
    for stage, shell in enumerate(shells):
        text = shell.llm.model
        assert isinstance(text, MODELING.MuseSparkOpenPpModelingV2)
        assert text._capture_carry is not None and not text._capture_carry.active
        assert shell._drafter == DISPATCH.DrafterContext(TARGET_LAYERS, 7)
    assert isinstance(shells[3].llm.spec_worker, WORKER.ContextServerDFlashWorker)
    for stage in range(3):
        assert not isinstance(shells[stage].llm.spec_worker, WORKER.ContextServerDFlashWorker)

    # PrefillTarget carries the captures; DecodeTarget keeps the built-in boundary.
    TRANSPORT.queues.clear()
    TRANSPORT.sent.clear()
    rows = 4
    for stage, shell in enumerate(shells):
        TRANSPORT.stage = stage
        shell.llm.model.forward(
            _attn(1, 1), input_ids=torch.arange(rows), spec_metadata=_SpecMetadata()
        )
        assert not shell.llm.model._capture_carry.active
    assert [len(TRANSPORT.sent[s][0]) for s in range(3)] == [3, 4, 5]
    assert all(shell.llm.model.builtin_calls == 0 for shell in shells)
    TRANSPORT.queues.clear()
    TRANSPORT.sent.clear()
    for stage, shell in enumerate(shells):
        TRANSPORT.stage = stage
        shell.llm.model.forward(
            _attn(0, 2), input_ids=torch.arange(2), spec_metadata=_SpecMetadata()
        )
    assert [len(TRANSPORT.sent[s][0]) for s in range(3)] == [2, 2, 2]
    assert all(shell.llm.model.builtin_calls == 1 for shell in shells)


def test_shell_without_a_drafter(monkeypatch):
    entered = SCOPE["entered"]
    shells = _shells(None, monkeypatch)
    assert _Omni.built == [0, 0, 0, 0] and SCOPE["entered"] == entered
    for shell in shells:
        assert shell.llm.model._capture_carry is None and shell._drafter is None
    TRANSPORT.queues.clear()
    TRANSPORT.sent.clear()
    for stage, shell in enumerate(shells):
        TRANSPORT.stage = stage
        shell.llm.model.forward(_attn(1, 1), input_ids=torch.arange(3))
    assert [len(TRANSPORT.sent[s][0]) for s in range(3)] == [2, 2, 2]


_OLD_SPEC_REFUSAL = (
    "MuseSparkOpen pipeline parallelism: speculative decoding is not supported; serve the "
    "stages without a speculative_config"
)


def test_a_refused_drafter_stage_is_logged_with_the_built_in_opt_in(monkeypatch):
    """The built-in decoder's refusal propagates unchanged, after one hint naming its opt-in."""

    def refuse(self, model_config, *args, **kwargs):
        raise ValueError(_OLD_SPEC_REFUSAL)

    monkeypatch.setattr(_Omni, "__init__", refuse)
    LOG.records.clear()
    with pytest.raises(ValueError) as error:
        _shells(_spec(), monkeypatch)
    assert str(error.value) == _OLD_SPEC_REFUSAL
    errors = [message for level, message in LOG.records if level == "error"]
    assert len(errors) == 1 and "pipeline_parallel_dflash_context" in errors[0]
    assert SCOPE["depth"] == 0

    LOG.records.clear()
    with pytest.raises(ValueError):
        _shells(None, monkeypatch)
    assert not [message for level, message in LOG.records if level == "error"]


def test_shell_warns_of_a_ring_too_small(monkeypatch):
    """At hidden 16 and 8192 tokens the boundary into the last stage is 1.75 MiB: residual
    512 KiB, attention input 256 KiB and four captures of 256 KiB. Such steps go through
    NCCL; the shells are still built, with one warning."""
    ring = types.SimpleNamespace(slot_bytes=1 << 20, send_on_ring=True, recv_on_ring=True)
    communicator = sys.modules["tensorrt_llm._torch.distributed.communicator"]
    monkeypatch.setattr(communicator, "pp_comm_buffered", lambda: True)
    monkeypatch.setattr(communicator, "_pp_comm", ring)

    def ring_warnings():
        return [m for level, m in LOG.records if level == "warning" and "go through NCCL" in m]

    LOG.records.clear()
    assert len(_shells(_spec(), monkeypatch)) == 4
    assert ring_warnings() and "takes up to 2 MiB" in ring_warnings()[0]
    ring.slot_bytes = 2 << 20
    LOG.records.clear()
    assert len(_shells(_spec(), monkeypatch)) == 4
    assert not ring_warnings()


def test_shell_refuses_a_last_stage_without_a_dflash_worker(monkeypatch):
    monkeypatch.setattr(_Omni, "__init__", _omni_with_worker(object()))
    with pytest.raises(ValueError, match="not a _StubWorkerBase"):
        _shells(_spec(), monkeypatch)


def _omni_with_worker(worker):
    original = _Omni.__init__

    def init(self, model_config, *args, **kwargs):
        original(self, model_config, *args, **kwargs)
        self.llm.spec_worker = worker

    return init


# --------------------------------------------------------------------------
# The context-server worker
# --------------------------------------------------------------------------
class _Drafter:
    def __init__(self) -> None:
        self.forwards = 0

    def dflash_forward(self) -> None:
        self.forwards += 1


def _context_worker():
    worker = _StubWorker()
    assert WORKER.refusal(worker) is None
    WORKER.install(worker)
    return worker


def test_worker_with_generation_rows_runs_no_draft_step():
    worker = _context_worker()
    drafter = _Drafter()
    meta = types.SimpleNamespace(runtime_draft_len=3, batch_indices_cuda=torch.arange(4))
    out = worker.forward_impl_for_test = None
    out = worker._forward_impl(None, None, None, torch.zeros(4, VOCAB), _attn(1, 4), meta, drafter)
    assert drafter.forwards == 0
    assert "builtin" not in worker.calls
    assert worker.calls == ["guided", ("onehot", 4, 0, 3), "rollback"]
    assert out["next_draft_tokens"].shape == (4, 3) and not out["next_draft_tokens"].any()
    assert out["new_tokens"].shape == (4, 4)
    assert out["next_new_tokens"].shape == (4, 4)
    assert torch.equal(out["next_new_tokens"][:, 0], out["new_tokens"][:, 0])


def test_worker_context_only_and_zero_draft_steps_are_the_builtin_step():
    worker = _context_worker()
    drafter = _Drafter()
    meta = types.SimpleNamespace(runtime_draft_len=3)
    assert worker._forward_impl(None, None, None, None, _attn(2, 2), meta, drafter) == {
        "builtin": True
    }
    meta0 = types.SimpleNamespace(runtime_draft_len=0)
    worker._forward_impl(None, None, None, None, _attn(0, 2), meta0, _Drafter())
    assert worker.calls == ["builtin", "builtin"]


def test_worker_class_is_composed_once():
    cls = WORKER.worker_class(_StubWorker)
    assert WORKER.worker_class(_StubWorker) is cls
    assert WORKER.worker_class(cls) is cls
    assert cls.__mro__[1] is WORKER.ContextServerDFlashWorker
    assert WORKER.refusal(None) is not None
    assert "not a" in WORKER.refusal(object())


# --------------------------------------------------------------------------
# The reduced drafter
# --------------------------------------------------------------------------
def _iter_modules(module, ignore_modules=None):
    """``models.modeling_utils.iter_modules``."""
    ignore_modules = ignore_modules or []
    ignore_names = []
    for name, mod in module.named_modules():
        if mod in ignore_modules:
            ignore_names.append(name)
        elif any(name.startswith(ignore_name) for ignore_name in ignore_names):
            continue
        else:
            yield mod


def _remove_weights(module, ignore_modules=None):
    """``models.modeling_utils.remove_weights``."""
    for mod in _iter_modules(module, ignore_modules):
        mod._parameters.clear()
        mod._buffers.clear()
        mod._weights_removed = True


class _Rotary(nn.Module):
    def __init__(self, positions: int, head_dim: int) -> None:
        super().__init__()
        self.register_buffer("rotary_cos_sin", torch.randn(positions, 2, head_dim // 2))


class _DraftAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, rope: bool) -> None:
        super().__init__()
        self.q_size, self.kv_size = heads * head_dim, kv_heads * head_dim
        self.qkv_proj = nn.Linear(hidden, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, hidden, bias=False)
        self.k_norm = nn.RMSNorm(head_dim)
        self.rotary_emb = _Rotary(64, head_dim) if rope else None


class _DraftLayer(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.self_attn = _DraftAttention(HIDDEN, 4, 2, 4, rope=True)
        self.mlp = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.input_layernorm = nn.RMSNorm(HIDDEN)


class _Drafter2(nn.Module):
    """A DFlash drafter's module layout and context-write methods."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_DraftLayer(i) for i in range(5))
        self.model.norm = nn.RMSNorm(HIDDEN)
        self.fc = nn.Linear(5 * HIDDEN, HIDDEN, bias=False)
        self.hidden_norm = nn.RMSNorm(HIDDEN)
        self.candidate_selector = nn.Linear(HIDDEN, 8)
        self.attention_convs = None
        self._fused_kv_weight = None
        self._cos_sin = None

    def _build_fused_kv_buffers(self) -> None:
        if self._fused_kv_weight is not None:
            return
        attn = [layer.self_attn for layer in self.model.layers]
        self._fused_kv_weight = torch.cat(
            [a.qkv_proj.weight[a.q_size : a.q_size + 2 * a.kv_size] for a in attn]
        ).contiguous()
        self._k_norm_stacked = torch.stack([a.k_norm.weight.data for a in attn])

    def _get_cos_sin_cache(self) -> torch.Tensor:
        if self._cos_sin is None:
            table = self.model.layers[0].self_attn.rotary_emb.rotary_cos_sin
            self._cos_sin = table.view(table.shape[0], -1).to(torch.float32).contiguous()
        return self._cos_sin

    def project_target_hidden(self, captures: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(captures))

    def precompute_context_kv(self, projected, positions):
        self._build_fused_kv_buffers()
        kv = torch.nn.functional.linear(projected, self._fused_kv_weight)
        return kv * self._get_cos_sin_cache()[positions][:, :1], self._k_norm_stacked.sum()


def test_slim_keeps_the_context_write_and_releases_the_rest():
    torch.manual_seed(1)
    drafter = _Drafter2()
    captures = torch.randn(6, 5 * HIDDEN)
    positions = torch.arange(6)
    full = drafter.precompute_context_kv(drafter.project_target_hidden(captures), positions)
    drafter._fused_kv_weight = None
    drafter._cos_sin = None
    expected = sum(
        p.numel() * p.element_size()
        for name, p in list(drafter.named_parameters()) + list(drafter.named_buffers())
        if not name.startswith(("fc.", "hidden_norm.", "model.layers.0.self_attn.rotary_emb."))
    )
    released = SLIM.slim(drafter, _remove_weights, _iter_modules)
    assert released == expected
    kept = dict(drafter.named_parameters())
    assert set(kept) == {"fc.weight", "hidden_norm.weight"}
    assert dict(drafter.named_buffers()).keys() == {
        "model.layers.0.self_attn.rotary_emb.rotary_cos_sin"
    }
    assert all(getattr(layer, "_weights_removed", False) for layer in drafter.model.layers)
    assert getattr(drafter.model.layers[0].self_attn.rotary_emb, "_weights_removed", False) is False
    slim_kv = drafter.precompute_context_kv(drafter.project_target_hidden(captures), positions)
    assert torch.equal(slim_kv[0], full[0]) and torch.equal(slim_kv[1], full[1])


def test_slim_refuses_a_drafter_without_the_context_write():
    drafter = _Drafter2()
    del drafter.fc
    with pytest.raises(ValueError, match="fc"):
        SLIM.slim(drafter, _remove_weights, _iter_modules)


# --------------------------------------------------------------------------
# The context-server worker's drafter-context store: bookkeeping and fallback
# --------------------------------------------------------------------------
class _StoreWorker(_StubWorkerBase):
    """``DFlashWorker``'s context-store state and slot helpers, as the built-in defines them."""

    def __init__(self, slots: int = 3, max_ctx: int = 256) -> None:
        super().__init__()
        self._ctx_paged = True
        self._ctx_kv_buf = None
        self._ctx_block_tables = None
        self._ctx_page_table = None
        self._ctx_page_size = 32
        self._ctx_kv_manager = None
        self._ctx_reuse_addressable = False
        self._max_ctx = max_ctx
        self._req_to_slot: dict = {}
        self._req_ctx_pos: dict = {}
        self._free_slots = collections.deque(range(slots))
        self._ctx_len_host = [0] * (slots + 1)
        self.written: list = []

    def _assign_slot(self, req_id, reset=False, updates=None):
        def clear(slot):
            if updates is None:
                self._ctx_len_host[slot] = 0
            else:
                updates[slot] = 0

        if reset and req_id in self._req_to_slot:
            old_slot = self._req_to_slot.pop(req_id)
            clear(old_slot)
            self._free_slots.append(old_slot)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                return None
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            clear(slot)
        return self._req_to_slot[req_id]

    def _ctx_alloc_tokens(self, request_ids):
        page = self._ctx_page_size
        caches = [self._ctx_kv_manager.kv_cache_map[req_id] for req_id in request_ids]
        return [-(-cache.capacity // page) * page for cache in caches]

    def _write_ctx_len(self, updates, base_updates=None):
        self.written.append(dict(updates))
        for slot, value in updates.items():
            self._ctx_len_host[slot] = value


def _store_worker(**kwargs):
    worker = _StoreWorker(**kwargs)
    WORKER.install(worker)
    return worker


def _step(worker, requests):
    """``(request id, first position, rows)`` per context request -> ``(runs, updates)``."""
    lens = [rows for _, _, rows in requests]
    host_pos = [first + t for _, first, rows in requests for t in range(rows)]
    meta = types.SimpleNamespace(
        request_ids=[req_id for req_id, _, _ in requests], host_position_ids=host_pos
    )
    attn = types.SimpleNamespace(
        num_contexts=len(requests), _seq_lens=torch.tensor(lens, dtype=torch.int32)
    )
    runs, updates = worker._context_runs(meta, attn, torch.tensor(host_pos))
    worker._write_ctx_len(updates)
    return [(r.offset, r.rows, r.table_row, r.first) for r in runs], updates


def test_store_bookkeeping_fresh_continued_and_reset_requests():
    worker = _store_worker()
    runs, updates = _step(worker, [(10, 0, 5), (11, 0, 0), (12, 0, 3)])
    # Private arena: the table row is the slot; an empty chunk takes a slot but stores nothing.
    assert runs == [(0, 5, 0, 0), (5, 3, 2, 0)]
    assert updates == {0: 5, 1: 0, 2: 3}
    assert worker._req_ctx_pos == {10: 5, 11: 0, 12: 3}
    # A chunk that starts where the stored context ended continues it.
    runs, updates = _step(worker, [(10, 5, 4)])
    assert runs == [(0, 4, 0, 5)] and updates == {0: 9}
    # One that does not starts over from position 0: its slot is released and taken again
    # (the only free one), at length 0.
    runs, updates = _step(worker, [(12, 7, 2)])
    assert runs == [(0, 2, 2, 0)] and updates == {2: 2}
    assert worker._req_ctx_pos[12] == 9 and worker._ctx_len_host[2] == 2
    assert not worker._free_slots


def test_store_bookkeeping_with_the_managers_tables():
    worker = _store_worker()
    worker._ctx_block_tables = torch.zeros(4, 8, dtype=torch.int32)
    worker._ctx_reuse_addressable = True
    worker._ctx_kv_manager = types.SimpleNamespace(
        kv_cache_map={
            20: types.SimpleNamespace(capacity=100),
            21: types.SimpleNamespace(capacity=64),
            22: types.SimpleNamespace(capacity=256),
        }
    )
    # The table row is the batch index. Reuse skipped [0, 40) of request 22: its first stored
    # position is 40. Request 21 does not fit its allocation (64 < 0 + 65): no store, slot freed.
    runs, updates = _step(worker, [(20, 0, 33), (21, 0, 65), (22, 40, 31)])
    assert runs == [(0, 33, 0, 0), (98, 31, 2, 40)]
    assert updates[0] == 33 and updates[2] == 71
    slot_21 = 1
    assert updates[slot_21] == 0 and 21 not in worker._req_to_slot
    assert 21 not in worker._req_ctx_pos and worker._free_slots[-1] == slot_21
    assert worker._req_ctx_pos == {20: 33, 22: 71}


def test_store_bookkeeping_without_a_free_slot():
    worker = _store_worker(slots=1)
    _step(worker, [(30, 0, 4)])
    runs, updates = _step(worker, [(31, 0, 4)])
    assert runs == [] and updates == {}
    assert 31 not in worker._req_ctx_pos
    assert ("warning", "DFlash: no free slots, skipping context store") in LOG.records


def test_store_falls_back_to_the_built_in_store_when_refused():
    worker = _store_worker()
    worker._ctx_paged = False
    LOG.records.clear()
    worker._store_prefill_context(None, None, None, None, 0)
    worker._store_prefill_context(None, None, None, None, 0)
    assert worker.calls == ["builtin_store", "builtin_store"]
    notes = [message for level, message in LOG.records if "drafter context store" in message]
    assert notes == [
        "DFlash context server: built-in drafter context store (the drafter context is not paged)"
    ]


class _OverridingStoreWorker(_StoreWorker):
    """A worker class with a store of its own: the context server must not bypass it."""

    def _store_prefill_context(self, *args):
        self.calls.append("own_store")


@pytest.mark.parametrize("method", ["_store_prefill_context", "_store_context_kv_paged"])
def test_store_override_names_a_class_that_overrides_the_built_in_store(method):
    assert WORKER.store_override(WORKER.worker_class(_StoreWorker)) is None
    overriding = type("_Overriding", (_StoreWorker,), {method: lambda self, *args: None})
    assert WORKER.store_override(WORKER.worker_class(overriding)) == (
        f"_Overriding defines {method}"
    )
    assert WORKER.store_override(object) == "no class defines _store_prefill_context"


def test_store_keeps_a_worker_classs_own_store():
    worker = _OverridingStoreWorker()
    WORKER.install(worker)
    # A paged arena: the class check refuses before the drafter is looked at.
    worker._ctx_kv_buf = object()
    LOG.records.clear()
    worker._store_prefill_context(None, None, None, None, 0)
    assert worker.calls == ["own_store"]
    notes = [message for level, message in LOG.records if "drafter context store" in message]
    assert notes == [
        "DFlash context server: built-in drafter context store "
        "(_OverridingStoreWorker defines _store_prefill_context)"
    ]


# --------------------------------------------------------------------------
# A drafter context window: the kept rows' captures only
# --------------------------------------------------------------------------
WINDOW = sys.modules[f"{_PACKAGE}.kernels.dflash_context.window"]


@pytest.mark.parametrize(
    "lengths, starts, keep_from, skips",
    [
        ([5], [0], [0], (0,)),  # everything kept
        ([5], [0], [3], (3,)),  # the window starts inside the chunk
        ([5], [10], [3], (0,)),  # a chunk past the window start
        ([5], [0], [40], (5,)),  # a chunk before the window start
        ([4, 0, 6], [8, 0, 100], [10, 0, 103], (2, 0, 3)),  # an empty chunk keeps nothing
        ([3, 3], [0, 0], [1, 2], (1, 2)),
    ],
)
def test_plan_window(lengths, starts, keep_from, skips):
    """Floor 0: no lead rows, the carried rows are the kept rows."""
    plan = WINDOW.plan_window(lengths, starts, keep_from, floor=0)
    assert plan.skips == skips
    assert plan.leads == (0,) * len(lengths)
    assert plan.kept == tuple(n - s for n, s in zip(lengths, skips))
    assert plan.carried == plan.kept
    assert plan.count == sum(plan.kept)
    offset, rows, kept_offset = 0, [], 0
    for length, skip, kept, got in zip(lengths, skips, plan.kept, plan.offsets()):
        rows.extend(range(offset + skip, offset + length))
        assert got == kept_offset
        offset += length
        kept_offset += kept
    assert plan.index("cpu").tolist() == rows


@pytest.mark.parametrize(
    "lengths, keep_from, floor, leads",
    [
        ([8192], [8192 - 1], 512, (511,)),  # one kept row: the 511 rows ahead are carried
        ([8192], [8192 - 512], 512, (0,)),  # kept rows at the floor: no lead
        ([8192], [8192 - 2592], 512, (0,)),
        ([300], [299], 512, (299,)),  # a chunk under the floor: the whole chunk
        ([300], [0], 512, (0,)),  # everything kept
        ([300], [400], 512, (0,)),  # nothing kept: nothing carried
        ([4096, 4096], [4095, 4096 - 2592], 512, (511, 0)),
        ([8, 8, 8], [7, 0, 5], 512, (7, 0, 5)),  # every chunk whole: the step's 24 rows
        # A short whole chunk beside a chunk that keeps nothing: the step is padded with that
        # chunk's last rows up to the floor.
        ([300, 7892], [0, 10_000], 512, (0, 212)),
        ([100, 50, 7000], [0, 10_000, 10_000], 512, (0, 50, 362)),
        ([100, 7000], [0, 10_000], 0, (0, 0)),
        ([100, 7000], [0, 10_000], 64, (0, 0)),  # already at the floor
        ([0, 5], [0, 4], 3, (0, 2)),
    ],
)
def test_plan_window_lead_rows(lengths, keep_from, floor, leads):
    plan = WINDOW.plan_window(lengths, [0] * len(lengths), keep_from, floor=floor)
    assert plan.leads == leads
    rows = []
    for (begin, end), length, skip, lead, offset in zip(
        plan.ranges(), plan.lengths, plan.skips, plan.leads, [0, *_cumsum(lengths)]
    ):
        assert (begin, end) == (offset + skip - lead, offset + length)
        rows.extend(range(begin, end))
    assert plan.index("cpu").tolist() == rows
    assert plan.offsets() == [0, *_cumsum(plan.carried)][: len(lengths)]


def _cumsum(values):
    out, total = [], 0
    for value in values:
        total += value
        out.append(total)
    return out


@pytest.mark.parametrize("seed", range(200))
def test_plan_window_runs_every_gemm_over_its_whole_rows_or_the_floor(seed):
    """Random steps of the served sizes: each request that keeps rows carries its whole chunk
    or at least the floor's rows (its K/V GEMM), the step carries all its rows or at least the
    floor's (the projection), lead rows sit just ahead of the kept rows, and at most
    ``floor - 1`` rows per request and per step are carried for this."""
    generator = torch.Generator().manual_seed(seed)

    def draw(low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=generator))

    floor = WINDOW.MIN_GEMM_ROWS
    count = draw(1, 5)
    lengths = [draw(0, 8193) if draw(0, 3) else draw(0, 600) for _ in range(count)]
    starts = [draw(0, 60_000) for _ in range(count)]
    keep_from = [start + draw(-3000, 9000) for start in starts]
    plan = WINDOW.plan_window(lengths, starts, keep_from)
    whole = sum(lengths)
    padding = 0
    for length, skip, lead, kept, carried in zip(
        plan.lengths, plan.skips, plan.leads, plan.kept, plan.carried
    ):
        assert 0 <= lead <= skip <= length
        if kept:
            assert carried == length or carried >= floor, (length, kept, carried)
            assert lead == 0 or carried == min(length, floor)
        else:
            padding += lead
    if plan.count == 0:
        assert sum(plan.kept) == 0
    else:
        assert plan.count == whole or plan.count >= floor, (lengths, plan)
    assert padding <= floor - 1
    if padding:
        assert plan.count == min(whole, floor)
        assert all(
            carried == length
            for length, kept, carried in zip(plan.lengths, plan.kept, plan.carried)
            if kept
        )


def test_plan_window_refuses_mismatched_lists():
    with pytest.raises(ValueError, match="keep positions"):
        WINDOW.plan_window([1, 2], [0, 0], [0])


def _window_meta(lengths, starts, keep_from, layers=TARGET_LAYERS):
    meta = _SpecMetadata(layers)
    meta.draft_context_keep_from = None if keep_from is None else list(keep_from)
    host = []
    for start, length in zip(starts, lengths):
        host.extend(range(start, start + length))
    meta.host_position_ids = torch.tensor(host, dtype=torch.int32)
    return meta


def _window_attn(lengths):
    return types.SimpleNamespace(
        num_contexts=len(lengths), seq_lens=torch.tensor(lengths, dtype=torch.int32)
    )


def _window_carry(route, stage):
    return CARRY.CaptureCarry(
        route,
        stage,
        send=TRANSPORT.send,
        recv=TRANSPORT.recv,
        raw_send=TRANSPORT.send,
        raw_recv=TRANSPORT.recv,
        eager_break=lambda fn: fn,
    )


def _run_window_chain(lengths, starts, keep_from, layers=TARGET_LAYERS):
    """The four stages of one context step with the step planned first on every stage (the
    shell's forward); returns the route, each stage's carry, metadata and own-layer dumps."""
    route = _route(SPLIT_15_16_16_15, layers)
    TRANSPORT.queues.clear()
    TRANSPORT.sent.clear()
    rows = sum(lengths)
    input_ids = torch.arange(rows) % VOCAB
    carries, metas, dumps = [], [], []
    for stage in range(4):
        TRANSPORT.stage = stage
        carry = _window_carry(route, stage)
        meta = _window_meta(lengths, starts, keep_from, layers)
        carry.plan_step(_window_attn(lengths), meta)
        _StubLayer.dumps = {}
        carry.forward_stage(_core(stage), None, input_ids, None, None, spec_metadata=meta)
        carries.append(carry)
        metas.append(meta)
        dumps.append(dict(_StubLayer.dumps))
    assert all(not queue for queue in TRANSPORT.queues.values()), "a message was not received"
    return route, carries, metas, dumps


def _schedules(seed: int):
    """A random step: 1-3 context requests of 0-5 rows, chunk starts and windows anywhere."""
    generator = torch.Generator().manual_seed(seed)

    def draw(high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=generator))

    count = 1 + draw(3)
    lengths = [draw(6) for _ in range(count)]
    lengths[0] = max(lengths[0], 1)
    # An empty chunk has no first row: its start reads as 0, as the write reads it.
    starts = [draw(40) if length else 0 for length in lengths]
    keep_from = [start + draw(9) - 2 for start in starts]
    return lengths, starts, [max(0, first) for first in keep_from]


@pytest.mark.parametrize("floor", [0, 4, None], ids=["no-lead", "floor-4", "served-floor"])
@pytest.mark.parametrize("seed", range(40))
def test_window_messages_and_last_stage_captures_over_random_schedules(seed, floor, monkeypatch):
    if floor is not None:
        monkeypatch.setattr(WINDOW, "MIN_GEMM_ROWS", floor)
    lengths, starts, keep_from = _schedules(seed)
    rows = sum(lengths)
    route, carries, metas, dumps = _run_window_chain(lengths, starts, keep_from)
    plan = WINDOW.plan_window(lengths, starts, keep_from)
    assert all(carry.plan == plan for carry in carries), "the stages planned different rows"
    for stage in range(3):
        (message,) = TRANSPORT.sent[stage]
        boundary = [((rows, HIDDEN), torch.float32), ((rows, HIDDEN), DTYPE)]
        groups = (
            [
                ((plan.count, (end - first) * HIDDEN), DTYPE)
                for first, end in route.stages[stage].sent
            ]
            if plan.count
            else []
        )
        assert message == boundary + groups, (stage, message)
    window = carries[3].window_capture(metas[3].captured_hidden_states)
    assert window is not None and window[0] == plan
    index = plan.index("cpu")
    if not plan.count:
        assert window[1] is None
        return
    captures = window[1]
    assert captures.shape == (plan.count, HIDDEN * len(TARGET_LAYERS))
    for column, layer in enumerate(TARGET_LAYERS):
        owner = next(s.stage for s in route.stages if layer in route.layers_of(s.own))
        expected = dumps[owner][layer].to(DTYPE).index_select(0, index)
        got = captures[:, column * HIDDEN : (column + 1) * HIDDEN]
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), (layer, owner)
    # The held captures are handed over once.
    assert carries[3].window_capture(metas[3].captured_hidden_states) is not None
    assert carries[3]._held == []


@pytest.mark.parametrize("floor, carried", [(0, 3 + 4 + 1), (4, 4 + 4 + 3), (None, 5 + 4 + 3)])
def test_window_captures_equal_the_whole_steps_at_the_kept_rows(floor, carried, monkeypatch):
    if floor is not None:
        monkeypatch.setattr(WINDOW, "MIN_GEMM_ROWS", floor)
    lengths, starts, keep_from = [5, 4, 3], [0, 64, 7], [2, 0, 9]
    _, full_metas, _, _ = _run_chain(sum(lengths))
    whole = full_metas[3].get_hidden_states(sum(lengths))
    _, carries, metas, _ = _run_window_chain(lengths, starts, keep_from)
    plan, captures = carries[3].window_capture(metas[3].captured_hidden_states)
    assert plan.kept == (3, 4, 1)
    assert plan.count == carried
    expected = whole.index_select(0, plan.index("cpu"))
    assert torch.equal(captures.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("keep_from", [None, "missing"])
def test_without_keep_positions_the_step_is_the_whole_one(keep_from):
    """No keep positions (the manager keeps every row, or metadata without the field):
    no plan, and the same messages and assembled capture buffer as the whole step."""
    rows = 7
    lengths, starts = [3, 4], [0, 50]
    _, full_metas, _, _ = _run_chain(rows)
    full_sent = {stage: list(TRANSPORT.sent[stage]) for stage in range(3)}
    route = _route()
    TRANSPORT.queues.clear()
    TRANSPORT.sent.clear()
    metas = []
    for stage in range(4):
        TRANSPORT.stage = stage
        carry = _window_carry(route, stage)
        meta = _window_meta(lengths, starts, None)
        if keep_from == "missing":
            del meta.draft_context_keep_from
        assert carry.plan_step(_window_attn(lengths), meta) is None
        carry.forward_stage(
            _core(stage), None, torch.arange(rows) % VOCAB, None, None, spec_metadata=meta
        )
        metas.append(meta)
        if stage == 3:
            assert carry.window_capture(meta.captured_hidden_states) is None
    assert {stage: TRANSPORT.sent[stage] for stage in range(3)} == full_sent
    assert torch.equal(
        metas[3].get_hidden_states(rows).view(torch.int16),
        full_metas[3].get_hidden_states(rows).view(torch.int16),
    )


def test_window_step_without_kept_rows_sends_the_boundary_alone():
    _, carries, metas, _ = _run_window_chain([4, 2], [0, 10], [100, 100])
    assert [TRANSPORT.sent[stage][0] for stage in range(3)] == [
        [((6, HIDDEN), torch.float32), ((6, HIDDEN), DTYPE)]
    ] * 3
    plan, captures = carries[3].window_capture(metas[3].captured_hidden_states)
    assert plan.count == 0 and captures is None


def test_window_breaks_refuse_a_step_without_a_plan():
    """A prefill graph captured with the window runs its window breaks on every replay; a
    replayed step without a plan raises instead of sending a message the peer does not expect."""
    route = _route()
    carry = _window_carry(route, 1)
    carry._window_ops()
    carry.plan_step(_window_attn([3]), _window_meta([3], [0], None))
    boundary = [torch.zeros(3, HIDDEN), torch.zeros(3, HIDDEN, dtype=DTYPE)]
    with pytest.raises(RuntimeError, match="without one"):
        carry._recv_window(boundary)
    with pytest.raises(RuntimeError, match="without one"):
        carry._send_window(boundary)


@pytest.mark.parametrize(
    "lengths, keep_from, floor, span",
    [
        ([8], [3], 0, (3, 5)),  # one request
        ([4, 6], [0, 0], 0, (0, 10)),  # whole chunks follow each other
        ([4, 6], [0, 2], 0, None),  # a gap: the second request drops its first rows
        ([4, 6], [2, 0], 0, (2, 8)),  # the first request's kept rows end where the next starts
        ([4, 0, 6], [2, 0, 0], 0, (2, 8)),  # an empty chunk in between
        ([4, 6], [99, 3], 0, (7, 3)),  # a request that keeps nothing ahead of the run
        ([4, 6], [99, 99], 0, None),  # nothing carried
        ([4, 6], [99, 5], 4, (6, 4)),  # lead rows extend the run backwards
        ([3, 2], [0, 99], 5, (0, 5)),  # padding rows of a request that keeps nothing
        ([3, 6], [0, 99], 5, None),  # padding rows at the end of a later chunk: a gap
    ],
)
def test_plan_span_is_the_carried_rows_when_they_are_one_run(lengths, keep_from, floor, span):
    plan = WINDOW.plan_window(lengths, [0] * len(lengths), keep_from, floor=floor)
    assert plan.span() == span
    if span is not None:
        assert plan.index("cpu").tolist() == list(range(span[0], span[0] + span[1]))


@pytest.mark.parametrize("seed", range(200))
def test_plan_span_agrees_with_the_index(seed):
    generator = torch.Generator().manual_seed(seed)

    def draw(high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=generator))

    count = 1 + draw(4)
    lengths = [draw(12) for _ in range(count)]
    keep_from = [draw(16) for _ in range(count)]
    plan = WINDOW.plan_window(lengths, [0] * count, keep_from, floor=draw(8))
    rows = plan.index("cpu").tolist()
    span = plan.span()
    contiguous = bool(rows) and rows == list(range(rows[0], rows[0] + len(rows)))
    assert (span is not None) == contiguous
    if span is not None:
        assert span == (rows[0], len(rows))


@pytest.mark.parametrize("floor", [0, 4, None], ids=["no-lead", "floor-4", "served-floor"])
@pytest.mark.parametrize("seed", range(40))
def test_window_carry_slices_a_run_of_kept_rows_and_indexes_the_rest(seed, floor, monkeypatch):
    """The carry builds the plan's row index only for a step whose kept rows are not one run;
    the messages and the last stage's captures are those of the index either way."""
    if floor is not None:
        monkeypatch.setattr(WINDOW, "MIN_GEMM_ROWS", floor)
    lengths, starts, keep_from = _schedules(seed)
    calls = []
    index = WINDOW.WindowPlan.index

    def counted(plan, device):
        calls.append(plan)
        return index(plan, device)

    monkeypatch.setattr(WINDOW.WindowPlan, "index", counted)
    _, carries, metas, dumps = _run_window_chain(lengths, starts, keep_from)
    window = carries[3].window_capture(metas[3].captured_hidden_states)
    plan = window[0]
    if plan.count == 0 or plan.span() is not None:
        assert calls == []
    else:
        assert calls and all(call == plan for call in calls)
    calls.clear()
    if plan.count:
        rows = index(plan, "cpu")
        for column, layer in enumerate(TARGET_LAYERS):
            stage = next(
                s.stage
                for s in carries[3].route.stages
                if layer in carries[3].route.layers_of(s.own)
            )
            expected = dumps[stage][layer].to(DTYPE).index_select(0, rows)
            got = window[1][:, column * HIDDEN : (column + 1) * HIDDEN]
            assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def test_window_send_of_a_run_is_a_slice_of_the_own_group(monkeypatch):
    """A sending stage's own group rows of a one-run plan go out as a slice of its group."""
    monkeypatch.setattr(WINDOW, "MIN_GEMM_ROWS", 0)
    route = _route()
    carry = _window_carry(route, 0)
    carry.plan_step(_window_attn([6]), _window_meta([6], [0], [2]))
    assert carry.plan.span() == (2, 4)
    own = torch.arange(6 * 4, dtype=torch.float32).view(6, 4).to(DTYPE)
    sent = []
    carry._raw_send = sent.append
    carry._send_kept([torch.zeros(6, HIDDEN), torch.zeros(6, HIDDEN, dtype=DTYPE), own])
    (message,) = sent
    kept = message[2]
    assert kept.data_ptr() == own[2].data_ptr() and kept.shape == (4, 4)
    assert torch.equal(kept.view(torch.int16), own[2:].view(torch.int16))


def test_window_send_of_rows_with_a_gap_is_a_gather_of_the_own_group(monkeypatch):
    """A sending stage's own group rows of a plan with a gap go out as a new tensor of the
    plan's rows in order (the index path), not as a view of its group."""
    monkeypatch.setattr(WINDOW, "MIN_GEMM_ROWS", 0)
    route = _route()
    carry = _window_carry(route, 0)
    carry.plan_step(_window_attn([3, 3]), _window_meta([3, 3], [0, 0], [0, 2]))
    assert carry.plan.span() is None
    own = torch.arange(6 * 4, dtype=torch.float32).view(6, 4).to(DTYPE)
    sent = []
    carry._raw_send = sent.append
    carry._send_kept([torch.zeros(6, HIDDEN), torch.zeros(6, HIDDEN, dtype=DTYPE), own])
    (message,) = sent
    kept = message[2]
    assert kept.untyped_storage().data_ptr() != own.untyped_storage().data_ptr()
    expected = own.index_select(0, torch.tensor([0, 1, 2, 5]))
    assert torch.equal(kept.view(torch.int16), expected.view(torch.int16))


def test_window_step_plan_refuses_a_keep_list_of_another_step():
    meta = _window_meta([3, 2], [0, 0], [0])
    with pytest.raises(ValueError, match="2 context requests"):
        WINDOW.step_plan(_window_attn([3, 2]), meta)


def test_window_starts_fall_back_to_the_cached_token_counts():
    meta = _window_meta([3, 2], [0, 0], [6, 0])
    meta.host_position_ids = None
    attn = _window_attn([3, 2])
    attn.kv_cache_params = types.SimpleNamespace(num_cached_tokens_per_seq=[5, 9])
    plan = WINDOW.step_plan(attn, meta)
    assert plan.starts == (5, 9) and plan.skips == (1, 0)


# --------------------------------------------------------------------------
# The window's drafter context write
# --------------------------------------------------------------------------
class _StoreDrafter:
    """Row-wise stand-ins for ``project_target_hidden`` and ``precompute_context_kv``."""

    fc = hidden_norm = True

    def project_target_hidden(self, captures):
        return captures.float().sum(dim=1, keepdim=True) * torch.tensor([[1.0, -0.5, 0.25]])

    def precompute_context_kv(self, projected, positions):
        offset = positions.to(projected.dtype).unsqueeze(1)
        return projected + offset, projected - offset


class _StoreWorkerBase:
    """The paged arena bookkeeping of ``DFlashWorker`` the window write uses (manager-bound
    block tables: rows are batch positions)."""

    def __init__(self, bound: bool = True) -> None:
        import collections

        self._ctx_paged = True
        self._ctx_block_tables = torch.zeros(4, 4) if bound else None
        self._ctx_kv_buf = [torch.zeros(1, dtype=torch.float32)]
        self._max_ctx = 1000
        self._ctx_reuse_addressable = True
        self._req_ctx_pos: dict = {}
        self._req_to_slot: dict = {}
        self._free_slots = collections.deque(range(4))
        self._ctx_len_host = [0] * 4
        self.rows: dict = {}

    def _ctx_alloc_tokens(self, request_ids):
        return [self._max_ctx] * len(request_ids)

    def _assign_slot(self, req_id, reset=False, updates=None):
        if reset and req_id in self._req_to_slot:
            old = self._req_to_slot.pop(req_id)
            updates[old] = 0
            self._free_slots.append(old)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                return None
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            updates[slot] = 0
        return self._req_to_slot[req_id]

    def _write_ctx_len(self, updates, base_updates=None):
        for slot, value in updates.items():
            self._ctx_len_host[slot] = value

    def _store_context_kv_paged(self, k, v, rows, positions):
        for row, position, key, value in zip(rows.tolist(), positions.tolist(), k, v):
            self.rows[(row, position)] = (key.clone(), value.clone())


def _store_steps(steps, prompts, keep, one_launch=False, bound=True, floor=None):
    """Run the window write over ``steps`` (each a list of (request id, chunk start, rows)) of
    requests with ``prompts`` lengths, keeping the last ``keep`` positions; returns the worker
    and the rows it wrote, keyed by (request id, position)."""
    worker = WORKER.worker_class(_StoreWorkerBase)(bound)
    drafter = _StoreDrafter()
    written = {}
    for step in steps:
        ids = [request for request, _, _ in step]
        lengths = [rows for _, _, rows in step]
        starts = [start for _, start, _ in step]
        keep_from = [max(0, prompts[request] - keep) for request in ids]
        plan = WINDOW.plan_window(lengths, starts, keep_from, floor=floor)
        # Every row's capture is a function of (request, position) only.
        whole = torch.tensor(
            [
                [request * 1000.0 + start + j, j * 0.5]
                for request, start, rows in step
                for j in range(rows)
            ]
        ).reshape(-1, 2)
        captures = whole.index_select(0, plan.index("cpu")) if plan.count else None
        meta = types.SimpleNamespace(request_ids=ids)
        attn = types.SimpleNamespace(num_contexts=len(step), num_ctx_tokens=sum(lengths))
        worker.rows = {}
        worker._store_window_context(drafter, meta, attn, plan, captures, one_launch=one_launch)
        slots = {slot: request for request, slot in worker._req_to_slot.items()}
        for (row, position), value in worker.rows.items():
            written[(ids[row] if bound else slots[row], position)] = value
    return worker, written


@pytest.mark.parametrize("floor", [0, 3, None], ids=["no-lead", "floor-3", "served-floor"])
def test_window_write_stores_the_kept_rows_of_the_whole_write(floor):
    """Chunked prompts of two requests, one of them reusing a prefix: the window write stores
    exactly the rows of the last ``keep`` positions (no lead row), each equal to the whole
    write's row, and leaves the slots' lengths and chunk positions where the whole write leaves
    them."""
    prompts = {7: 20, 9: 13}
    steps = [
        [(7, 0, 8), (9, 4, 5)],  # request 9 reused a 4-token prefix
        [(7, 8, 8), (9, 9, 4)],
        [(7, 16, 4)],
    ]
    keep = 6
    whole_worker, whole = _store_steps(steps, prompts, keep=10_000)
    window_worker, window = _store_steps(steps, prompts, keep=keep, floor=floor)
    expected = {key: value for key, value in whole.items() if key[1] >= prompts[key[0]] - keep}
    assert window.keys() == expected.keys()
    for key, (k, v) in expected.items():
        assert torch.equal(window[key][0], k) and torch.equal(window[key][1], v), key
    assert window_worker._ctx_len_host == whole_worker._ctx_len_host
    assert window_worker._req_ctx_pos == whole_worker._req_ctx_pos == {7: 20, 9: 13}
    # Request 9's slot starts at its reused prefix: positions 4.. are its rows.
    assert min(position for request, position in whole if request == 9) == 4


def test_window_write_refuses_a_dense_arena():
    worker = WORKER.worker_class(_StoreWorkerBase)()
    worker._ctx_paged = False
    plan = WINDOW.plan_window([2], [0], [0])
    with pytest.raises(RuntimeError, match="dense arena"):
        worker._store_window_context(
            _StoreDrafter(),
            types.SimpleNamespace(request_ids=[1]),
            types.SimpleNamespace(num_contexts=1, num_ctx_tokens=2),
            plan,
            torch.zeros(2, 2),
        )


def test_worker_store_takes_the_window_only_when_the_step_was_received_with_it():
    calls = []

    class _Base(_StoreWorkerBase):
        def _store_prefill_context(self, *args):
            calls.append("builtin")

    worker = WORKER.worker_class(_Base)()
    meta = types.SimpleNamespace(captured_hidden_states=torch.zeros(4, 2), request_ids=[1])
    attn = types.SimpleNamespace(num_contexts=1, num_ctx_tokens=2)
    worker._store_prefill_context(_StoreDrafter(), meta, attn, None, 2)
    assert calls == ["builtin"]  # no carry
    carry = types.SimpleNamespace(window_capture=lambda buffer: None)
    worker._capture_carry = carry
    worker._store_prefill_context(_StoreDrafter(), meta, attn, None, 2)
    assert calls == ["builtin", "builtin"]  # a step received without the window
    plan = WINDOW.plan_window([2], [0], [1])
    assert plan.leads == (1,)
    carry.window_capture = lambda buffer: (plan, torch.ones(plan.count, 2))
    worker._store_prefill_context(_StoreDrafter(), meta, attn, None, 2)
    assert calls == ["builtin", "builtin"] and list(worker.rows) == [(0, 1)]


def _runs_through_the_built_in_calls(calls):
    """A ``store.store_runs`` stand-in: every run through ``precompute_context_kv`` and
    ``_store_context_kv_paged`` at its table row and positions, as the store writes it."""

    def store_runs(worker, drafter, projected, runs):
        calls.append([(run.offset, run.rows, run.table_row, run.first, run.lead) for run in runs])
        for run in runs:
            chunk = projected[run.offset : run.offset + run.lead + run.rows]
            chunk = chunk.to(worker._ctx_kv_buf[0].dtype)
            positions = torch.arange(run.first - run.lead, run.first + run.rows, dtype=torch.long)
            k, v = drafter.precompute_context_kv(chunk, positions)
            rows = torch.full((run.rows,), run.table_row, dtype=torch.long)
            worker._store_context_kv_paged(
                k[run.lead :], v[run.lead :], rows, positions[run.lead :]
            )

    return store_runs


@pytest.mark.parametrize("floor", [0, 3, None], ids=["no-lead", "floor-3", "served-floor"])
@pytest.mark.parametrize("bound", [True, False], ids=["manager-tables", "private-arena"])
@pytest.mark.parametrize("keep", [1, 6, 10_000])
def test_window_write_in_one_launch_stores_the_rows_of_the_built_in_calls(
    monkeypatch, keep, bound, floor
):
    """With the one-launch store the window write hands ``store.store_runs`` one run per request
    with kept rows (its offset among the kept captures, its rows, its table row, its first kept
    position): the store writes the rows, slot lengths and chunk positions the per-request
    built-in calls write."""
    calls = []
    monkeypatch.setattr(WORKER._store, "store_runs", _runs_through_the_built_in_calls(calls))
    prompts = {7: 20, 9: 13}
    steps = [
        [(7, 0, 8), (9, 4, 5)],  # request 9 reused a 4-token prefix
        [(9, 9, 4), (7, 8, 8)],  # batch positions that are not the requests' slots
        [(7, 16, 4)],
    ]
    calls_worker, calls_rows = _store_steps(steps, prompts, keep, bound=bound, floor=floor)
    assert not calls
    store_worker, store_rows = _store_steps(
        steps, prompts, keep, one_launch=True, bound=bound, floor=floor
    )
    assert calls and all(calls)
    assert store_rows.keys() == calls_rows.keys()
    for key, (k, v) in calls_rows.items():
        assert torch.equal(store_rows[key][0], k) and torch.equal(store_rows[key][1], v), key
    assert store_worker._ctx_len_host == calls_worker._ctx_len_host
    assert store_worker._req_ctx_pos == calls_worker._req_ctx_pos == {7: 20, 9: 13}


def test_window_write_takes_the_one_launch_store_unless_it_is_refused(monkeypatch):
    calls = []
    monkeypatch.setattr(WORKER._store, "store_runs", _runs_through_the_built_in_calls(calls))
    monkeypatch.setattr(WORKER._store, "refusal", lambda worker, drafter: None)
    monkeypatch.setattr(WORKER, "store_override", lambda cls: None)
    worker = WORKER.worker_class(_StoreWorkerBase)()
    plan = WINDOW.plan_window([3], [0], [1])
    assert plan.leads == (1,)
    worker._capture_carry = types.SimpleNamespace(
        window_capture=lambda buffer: (plan, torch.ones(plan.count, 2))
    )
    meta = types.SimpleNamespace(captured_hidden_states=torch.zeros(4, 2), request_ids=[1])
    attn = types.SimpleNamespace(num_contexts=1, num_ctx_tokens=3)
    worker._store_prefill_context(_StoreDrafter(), meta, attn, None, 3)
    assert calls == [[(0, 2, 0, 1, 1)]] and sorted(worker.rows) == [(0, 1), (0, 2)]
    # A refused store keeps the per-request built-in calls on the kept rows.
    monkeypatch.setattr(WORKER._store, "refusal", lambda worker, drafter: "refused here")
    worker = WORKER.worker_class(_StoreWorkerBase)()
    worker._capture_carry = types.SimpleNamespace(
        window_capture=lambda buffer: (plan, torch.ones(plan.count, 2))
    )
    worker._store_prefill_context(_StoreDrafter(), meta, attn, None, 3)
    assert len(calls) == 1 and sorted(worker.rows) == [(0, 1), (0, 2)]


def test_install_hands_the_carry_to_the_last_stage_worker(monkeypatch):
    shells = _shells(_spec(), monkeypatch)
    assert shells[3].llm.spec_worker._capture_carry is shells[3].llm.model._capture_carry
