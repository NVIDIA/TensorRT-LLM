# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam NVFP4-expert checkpoint-layout component / sm_103 / tp4: packed experts and blocks.

Owned by the routed NVFP4 target (this directory); the full model (``model``) builds on it.
Constructed on its own it is a component, not a target: a caller on each of four ranks
constructs this class with the checkpoint's configuration and its rank's ``Mapping`` (attention
TP4, MoE TP4/EP1, PP1), loads a caller-supplied ModelOpt NVFP4 checkpoint directory into the
rank's slices, and runs representative computations. It does not run a full-depth forward,
prefill/decode loop or sampler.

This is the BF16 layout component (``component_bf16``) with NVFP4 expert
storage and execution. Everything outside the routed experts -- ownership, the embedding,
attention on TRTLLM attention and KVCacheManagerV2 metadata, the dense feed-forward, the
replicated FP32 router with its diagnostic-mode all-rank agreement check, the norms, depth
residuals, head and every collective -- is that component's code, unchanged, on the
checkpoint's BF16/FP32 tensors copied byte for byte. The experts stay packed:

    storage   per sparse layer and projection kind, banks indexed by global expert id: packed
              E2M1 codes ``[256, rows, K/2]`` u8 and E4M3 block scales in the GEMM's 128x4
              swizzled order ``[256, rows * K/16]`` u8 (the rank's slice, see ``layout``), and
              ``input_scale``, ``weight_scale_2``, ``g = 1/input_scale``, ``alpha =
              input_scale * weight_scale_2`` ``[256]`` fp32. The up and gate banks of a layer
              are the two halves of one ``[up; gate]`` storage (``[256, 2 * rows, K/2]`` codes,
              ``[256, 2 * rows * K/16]`` scales), the FC1 operand of the grouped GEMM, so the
              fused operand is a view rather than a second copy
    compute   ``routed_experts.nvfp4_routed_experts``: every (token, route) as one row of the
              CUTLASS NVFP4 grouped GEMM (fp32 SwiGLU of bf16 gate and up, rounded to bf16
              before the FC2 quantization), then the combination (bf16 routes, fp32
              ascending-expert sum, x2.5, bf16) on the device; the one MoE all-reduce sums the
              four bf16 rank partials in fp32 and rounds the sum to bf16 once

Loading streams bounded tiles straight into that storage and is a transaction: metadata
(config.json, hf_quant_config.json, every index/header entry) is validated against the whole
model before any storage changes, and the component is usable only after every rank loaded
every selected tensor. The grouped GEMM takes one FC1 input scale and one alpha for gate and up
and one FC2 input scale per layer, so loading also proves that every sparse layer's stored
scales are exactly that (``check_expert_scales``) and refuses the checkpoint otherwise; no scale
is recomputed or merged. Every parameter, bank and derived scalar is a registered parameter, so
``weight_state`` covers packed codes, block scales and scalars.

``expert.nvfp4_swiglu_expert`` under ``dispatch_combine.beam_dispatch_combine`` computes the same
experts one selected expert at a time with host-enumerated dispatch; it stays as the independent
reference the provider is compared with, not as a serving path.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping as ParallelMapping

from . import routed_experts
from .component_bf16 import (
    MAX_TOKENS,
    BeamCheckpointLayoutBf16Sm103Tp4,
    LoadError,
    _attr,
    _check_mapping,
    _rec,
    check_max_tokens,
)
from .layout import (
    EMBED,
    PROJECTIONS,
    QUANT_CONFIG_FILE,
    WORLD_SIZE,
    QuantSpec,
    check_config,
    check_partition,
    check_recipe,
    ownership_ledger,
    stored_entries,
    swizzled_numel,
    tensor_specs,
)
from .loader import (
    CheckpointFormatError,
    SafetensorsCatalog,
    TileReader,
    check_block_scales,
    read_packed_into,
    read_scalars,
    read_slice_into,
    validate,
)

_SM = (10, 3)
#: Bank fields of one projection kind of one sparse layer.
BANK_FIELDS = ("weight", "weight_scale", "input_scale", "weight_scale_2", "g", "alpha")
_KINDS = tuple(p.removesuffix("_proj") for p in PROJECTIONS)
#: The kinds whose banks share one FC1 storage, in its row order: the grouped GEMM reads
#: ``[w3; w1]`` = ``[up; gate]``.
FC1_KINDS = ("up", "gate")


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def check_expert_scales(layer: int, scales: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    """Raise ``CheckpointFormatError`` unless sparse layer ``layer``'s scales fit the grouped GEMM.

    ``scales[kind][field]`` is the ``[256]`` fp32 ``input_scale`` / ``weight_scale_2`` of
    projection kind ``gate``, ``up`` or ``down``. The operator takes one FC1 input scale and one
    alpha per expert for gate and up together, and one FC2 input scale, so required, bit for
    bit: gate and up input scales equal and the same for every expert, gate and up
    ``weight_scale_2`` equal per expert, and one down input scale for every expert. The stored
    values are used as they are; none is replaced by a maximum or recomputed to fit.
    """

    def bits(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(torch.float32).cpu().view(torch.int32)

    gate_in, up_in = bits(scales["gate"]["input_scale"]), bits(scales["up"]["input_scale"])
    gate_w2, up_w2 = bits(scales["gate"]["weight_scale_2"]), bits(scales["up"]["weight_scale_2"])
    down_in = bits(scales["down"]["input_scale"])
    problems = []
    if not torch.equal(gate_in, up_in):
        problems.append("gate and up input scales differ")
    if not bool((gate_in == gate_in[0]).all()):
        problems.append("the FC1 input scale differs between experts")
    if not torch.equal(gate_w2, up_w2):
        problems.append("gate and up weight_scale_2 differ")
    if not bool((down_in == down_in[0]).all()):
        problems.append("the FC2 input scale differs between experts")
    if problems:
        raise CheckpointFormatError(
            f"sparse layer {layer}: the grouped-GEMM experts need one FC1 input scale and alpha "
            f"for gate and up and one FC2 input scale per layer; {', '.join(problems)}"
        )


def _joined(first: torch.Tensor, second: torch.Tensor, dim: int) -> Optional[torch.Tensor]:
    """``first`` and ``second`` concatenated along ``dim`` as one contiguous view of their storage.

    None unless both are equally shaped views of one storage in which ``second`` starts where
    ``first`` ends along ``dim`` and the joined tensor is contiguous -- the layout of the two
    halves of a fused bank. Reads tensor metadata only.
    """
    if (
        first.shape != second.shape
        or first.stride() != second.stride()
        or first.dtype != second.dtype
        or first.device != second.device
        or first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
        or second.storage_offset() != first.storage_offset() + first.shape[dim] * first.stride(dim)
    ):
        return None
    size = list(first.shape)
    size[dim] *= 2
    joined = first.as_strided(size, first.stride(), first.storage_offset())
    return joined if joined.is_contiguous() else None


class BeamCheckpointLayoutNvfp4Sm103Tp4(BeamCheckpointLayoutBf16Sm103Tp4):
    """Rank-local real-checkpoint weights of the Beam model with packed NVFP4 experts.

    ``config`` is the checkpoint's config.json (a mapping); ``mapping`` this rank's
    ``tensorrt_llm.mapping.Mapping`` with world size 4, TP4, PP1, MoE TP4/EP1; ``layers`` the
    logical layers to hold (default all 52; the embedding, final norm and head are always
    held); ``max_tokens`` the most tokens one call accepts (default ``MAX_TOKENS``).
    Construction allocates uninitialized rank-local storage on the current CUDA device and
    validates the configuration and layout; it reads no checkpoint data.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        mapping: ParallelMapping,
        layers: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        *,
        max_tokens: int = MAX_TOKENS,
    ):
        nn.Module.__init__(self)
        self.max_tokens = check_max_tokens(max_tokens)
        self.schedule = check_config(config)
        _check_mapping(mapping)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this component is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        self.config = dict(config)
        self.mapping = mapping
        self.rank = mapping.rank
        self.device = device
        self.layers = tuple(sorted(range(self.schedule.num_layers) if layers is None else layers))
        self.specs = tensor_specs(config, self.layers)
        check_partition(self.specs)
        self.hidden = config["hidden_size"]
        self.head_dim = config["head_dim"]
        self.local_heads = config["num_attention_heads"] // WORLD_SIZE
        self.local_kv_heads = config["num_key_value_heads"] // WORLD_SIZE
        self.eps = config["rms_norm_eps"]
        self.window = config["sliding_window"]
        self.rope_theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]
        self.softcap = config["final_logit_softcapping"]
        self.routed_scaling_factor = config["routed_scaling_factor"]
        self.vocab_interval = self.specs[EMBED].interval(self.rank)[1:]
        self.tensor_names: list = []
        self.bank_names: list = []
        self.expert_names: list = []
        experts = config["num_experts"]
        # Per sparse layer: the [up; gate] codes and swizzled block scales the FC1 banks view.
        fc1_storage = {}
        for spec in self.specs.values():
            if isinstance(spec, QuantSpec):
                self.expert_names.append(spec.name)
                if spec.bank in self.bank_names:
                    continue
                rows, half = spec.local_packed_shape(self.rank)
                _, scale_cols = spec.local_scale_shape(self.rank)
                scale_bytes = swizzled_numel(rows, scale_cols)
                kind = spec.proj.removesuffix("_proj")
                if kind in FC1_KINDS:
                    if spec.layer not in fc1_storage:
                        fc1_storage[spec.layer] = (
                            torch.empty(
                                (experts, 2 * rows, half), dtype=torch.uint8, device=device
                            ),
                            torch.empty(
                                (experts, 2 * scale_bytes), dtype=torch.uint8, device=device
                            ),
                        )
                    codes, scales = fc1_storage[spec.layer]
                    part = FC1_KINDS.index(kind)
                    if codes.shape[1:] != (2 * rows, half) or scales.shape[1] != 2 * scale_bytes:
                        raise ValueError(f"layer {spec.layer}: gate and up slices differ in shape")
                    storage = {
                        "weight": codes[:, part * rows : (part + 1) * rows],
                        "weight_scale": scales[:, part * scale_bytes : (part + 1) * scale_bytes],
                    }
                else:
                    storage = {
                        "weight": torch.empty(
                            (experts, rows, half), dtype=torch.uint8, device=device
                        ),
                        "weight_scale": torch.empty(
                            (experts, scale_bytes), dtype=torch.uint8, device=device
                        ),
                    }
                for field in BANK_FIELDS[2:]:
                    storage[field] = torch.empty((experts,), dtype=torch.float32, device=device)
                for field in BANK_FIELDS:
                    self.register_parameter(
                        _attr(f"{spec.bank}.{field}"),
                        nn.Parameter(storage[field], requires_grad=False),
                    )
                self.bank_names.append(spec.bank)
                continue
            dtype = torch.bfloat16 if spec.effective_dtype == "BF16" else torch.float32
            param = nn.Parameter(
                torch.empty(spec.local_shape(self.rank), dtype=dtype, device=device),
                requires_grad=False,
            )
            self.register_parameter(_attr(spec.name), param)
            self.tensor_names.append(spec.name)
        self._allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
        self.usable = False
        self.load_record: Optional[dict] = None
        self._init_kv_cache_mode()
        self._init_diagnostics()
        #: Per sparse layer, per global expert: gate and up input scales bitwise equal.
        self._shared_fc1: dict = {}
        #: Per sparse layer: (the bank parameters, the grouped GEMM's operands viewing them).
        self._operands: dict = {}

    # ------------------------------------------------------------------ storage

    def tensor(self, name: str) -> torch.Tensor:
        """The rank-local storage of higher-precision tensor ``name``."""
        spec = self.specs[name]
        if isinstance(spec, QuantSpec):
            raise ValueError(f"{name} is an NVFP4 expert projection; see expert_storage()")
        return getattr(self, _attr(name))

    def bank(self, bank: str, field: str = "weight") -> torch.Tensor:
        """The current ``[256, ...]`` storage of one field of one sparse layer's expert bank."""
        if field not in BANK_FIELDS:
            raise ValueError(f"unknown bank field {field!r}; expected one of {BANK_FIELDS}")
        return getattr(self, _attr(f"{bank}.{field}"))

    def expert_bank(self, layer: int) -> dict:
        """``{"gate"|"up"|"down": {field: [256, ...] storage}}`` of sparse layer ``layer``."""
        self._check_usable(layer)
        if not self.schedule.is_sparse(layer):
            raise ValueError(f"layer {layer} has a dense feed-forward")
        return {
            kind: {f: self.bank(f"layers.{layer}.experts.{kind}", f) for f in BANK_FIELDS}
            for kind in _KINDS
        }

    def expert_operands(self, layer: int) -> routed_experts.ExpertOperands:
        """The grouped GEMM's operands of sparse layer ``layer``: views of its current banks.

        Resolved from the registered parameters, so a replaced bank is what the next call uses;
        the views are rebuilt only when a parameter object changes. The up and gate banks must
        still be the two adjacent halves of one storage (as construction allocates them), else
        ``RuntimeError``. Reads tensor metadata only.
        """
        bank = self.expert_bank(layer)
        params = tuple(bank[kind][field] for kind in _KINDS for field in BANK_FIELDS)
        cached = self._operands.get(layer)
        if cached is not None and len(cached[0]) == len(params):
            if all(a is b for a, b in zip(cached[0], params)):
                return cached[1]
        up, gate, down = bank["up"], bank["gate"], bank["down"]
        codes = _joined(up["weight"], gate["weight"], 1)
        scales = _joined(up["weight_scale"], gate["weight_scale"], 1)
        if codes is None or scales is None:
            raise RuntimeError(
                f"layer {layer}: the up and gate banks are no longer the two halves of one FC1 "
                "storage; the grouped GEMM reads them as one operand"
            )
        experts, fc1_rows, half = codes.shape
        hidden = down["weight"].shape[1]
        operands = routed_experts.ExpertOperands(
            fc1_weight=codes.view(torch.int64),
            fc1_block_scales=scales.view(experts, fc1_rows, -1).view(torch.int32),
            fc1_alpha=gate["alpha"],
            fc1_input_scale_inv=gate["g"][0],
            fc2_weight=down["weight"].view(torch.int64),
            fc2_block_scales=down["weight_scale"].view(experts, hidden, -1).view(torch.int32),
            fc2_input_scale_inv=down["g"],
            fc2_alpha=down["alpha"],
        )
        self._operands[layer] = (params, operands)
        return operands

    def expert_storage(self, layer: int, expert: int, proj: str) -> dict:
        """Views of one expert projection's rank-local storage (``proj`` in ``PROJECTIONS``)."""
        spec = self.specs.get(f"model.layers.{layer}.mlp.experts.{expert}.{proj}")
        if not isinstance(spec, QuantSpec):
            raise ValueError(f"layer {layer} expert {expert} {proj!r} is not held")
        return {f: self.bank(spec.bank, f)[expert] for f in BANK_FIELDS}

    def ledger(self) -> list:
        """This component's ownership rows (see ``layout.ownership_ledger``)."""
        return ownership_ledger(self.specs)

    # ------------------------------------------------------------------ loading

    def _load_expert(self, catalog, reader, spec: QuantSpec, digest) -> dict:
        dim, p0, p1 = spec.packed_interval(self.rank)
        _, s0, s1 = spec.scale_interval(self.rank)
        if dim == 0:
            rows, packed_cols, scale_cols = (p0, p1), None, None
        else:
            rows, packed_cols, scale_cols = (0, spec.shape[0]), (p0, p1), (s0, s1)
        packed = self.bank(spec.bank, "weight")[spec.expert]
        read_packed_into(
            catalog, reader, spec.name + ".weight", rows, packed_cols, packed, digest=digest
        )
        linear = torch.empty(
            spec.local_scale_shape(self.rank), dtype=torch.uint8, device=self.device
        )
        read_packed_into(
            catalog, reader, spec.name + ".weight_scale", rows, scale_cols, linear, digest=digest
        )
        check_block_scales(spec.name + ".weight_scale", linear)
        swizzled = torch.ops.trtllm.block_scale_interleave(linear).reshape(-1)
        dst = self.bank(spec.bank, "weight_scale")[spec.expert]
        if swizzled.numel() != dst.numel():
            raise CheckpointFormatError(
                f"{spec.name}: swizzled scales hold {swizzled.numel()} bytes, storage {dst.numel()}"
            )
        dst.copy_(swizzled)
        torch.cuda.current_stream(self.device).synchronize()
        return {
            "interval": list(spec.interval(self.rank)),
            "packed_interval": [dim, p0, p1],
            "scale_interval": [dim, s0, s1],
            "local_packed_shape": list(packed.shape),
            "local_scale_shape": list(linear.shape),
            "bank": spec.bank,
            "slot": spec.expert,
            "source_sha256": digest.hexdigest() if digest is not None else None,
        }

    def _load_scalars(self, catalog, reader, layer: int) -> dict:
        experts = [s for s in self.specs.values() if isinstance(s, QuantSpec) and s.layer == layer]
        names = [s.name + suffix for s in experts for suffix in (".input_scale", ".weight_scale_2")]
        values = read_scalars(catalog, reader, names)
        bad = [n for n, v in values.items() if not (v > 0.0 and v != float("inf"))]
        if bad:
            raise CheckpointFormatError(f"non-finite or non-positive scalars: {bad[:4]}")
        n_experts = self.config["num_experts"]
        shared = [False] * n_experts
        stored = {}
        for kind in _KINDS:
            s = torch.empty(n_experts, dtype=torch.float32)
            w2 = torch.empty(n_experts, dtype=torch.float32)
            for e in range(n_experts):
                p = f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj"
                s[e], w2[e] = values[p + ".input_scale"], values[p + ".weight_scale_2"]
            stored[kind] = {"input_scale": s, "weight_scale_2": w2}
        check_expert_scales(layer, stored)
        for kind in _KINDS:
            s, w2 = stored[kind]["input_scale"], stored[kind]["weight_scale_2"]
            g = torch.ones(n_experts, dtype=torch.float32) / s
            alpha = s * w2
            bank = f"layers.{layer}.experts.{kind}"
            for field, value in (
                ("input_scale", s),
                ("weight_scale_2", w2),
                ("g", g),
                ("alpha", alpha),
            ):
                self.bank(bank, field).copy_(value)
        for e in range(n_experts):
            p = f"model.layers.{layer}.mlp.experts.{e}."
            shared[e] = _f32_bits(values[p + "gate_proj.input_scale"]) == _f32_bits(
                values[p + "up_proj.input_scale"]
            )
        self._shared_fc1[layer] = shared
        return {"scalars": len(values), "gate_up_input_scales_shared": sum(shared)}

    @torch.no_grad()
    def load(self, checkpoint_dir: str, *, record_digests: bool = True) -> dict:
        """Stream the selected tensors of a caller-supplied checkpoint into rank-local storage.

        Collective: every rank calls it. config.json must equal the constructed configuration,
        hf_quant_config.json must be the NVFP4 recipe (``layout.check_recipe``), and every
        stored tensor of the whole model must be exactly what the layout needs before any
        storage changes. Expert codes and block-scale bytes are copied unchanged (block scales
        then swizzled), scalars must be finite and positive, block scales finite and
        non-negative, and every sparse layer's scales must fit the grouped GEMM
        (``check_expert_scales``). The component is unusable from the first write until every
        rank reports a complete load; a failure on any rank raises ``LoadError`` on all ranks.
        Returns the load record (recipe, coverage, source digests, bytes read).
        """
        self._check_diagnostics_agree()
        error: Optional[BaseException] = None
        catalog = None
        recipe = None
        try:
            with open(os.path.join(checkpoint_dir, "config.json")) as f:
                stored = json.load(f)
            if stored != self.config:
                differing = sorted(
                    k for k in set(stored) | set(self.config) if stored.get(k) != self.config.get(k)
                )
                raise CheckpointFormatError(
                    f"config.json differs from the constructed configuration: {differing[:8]}"
                )
            with open(os.path.join(checkpoint_dir, QUANT_CONFIG_FILE)) as f:
                quant = json.load(f)
            full = tensor_specs(self.config)
            recipe = check_recipe(quant, full)
            catalog = SafetensorsCatalog(checkpoint_dir)
            validate(catalog, stored_entries(full), complete=True)
        except Exception as exc:  # Keep rank-local failures inside the collective protocol.
            error = exc
        if not self._agree(error is None):
            raise LoadError(
                f"rank {self.rank}: checkpoint rejected before loading: {error!r}"
            ) from error

        self.usable = False
        self.load_record = None
        self._shared_fc1 = {}
        self._operands = {}
        coverage = OrderedDict()
        scalars = OrderedDict()
        reader = None
        bytes_read = reads = 0

        def first_entry(spec):
            return catalog.entries[
                spec.name + ".weight" if isinstance(spec, QuantSpec) else spec.name
            ]

        try:
            reader = TileReader()
            order = sorted(
                self.specs.values(), key=lambda s: (first_entry(s).path, first_entry(s).begin)
            )
            sparse = [layer for layer in self.layers if self.schedule.is_sparse(layer)]
            for spec in order:
                digest = hashlib.sha256() if record_digests else None
                if isinstance(spec, QuantSpec):
                    coverage[spec.name] = self._load_expert(catalog, reader, spec, digest)
                    continue
                dst = self.tensor(spec.name)
                read_slice_into(catalog, reader, spec, self.rank, dst, digest=digest)
                coverage[spec.name] = {
                    "interval": spec.interval(self.rank),
                    "local_shape": list(dst.shape),
                    "dtype": str(dst.dtype),
                    "source_sha256": digest.hexdigest() if digest is not None else None,
                }
            for layer in sparse:
                scalars[layer] = self._load_scalars(catalog, reader, layer)
        except Exception as exc:  # Every rank must reach the completion agreement.
            error = exc
        finally:
            if reader is not None:
                bytes_read, reads = reader.bytes_read, reader.reads
                try:
                    reader.close()
                except Exception as exc:  # Cleanup failure must also reach the agreement.
                    if error is None:
                        error = exc
        complete = (
            error is None and list(coverage) == [s.name for s in order] and list(scalars) == sparse
        )
        if not self._agree(complete):
            self._shared_fc1 = {}
            self._operands = {}
            raise LoadError(
                f"rank {self.rank}: load did not complete on every rank: {error!r}"
            ) from error
        self.load_record = {
            "rank": self.rank,
            "checkpoint_dir": os.fspath(checkpoint_dir),
            "layers": list(self.layers),
            "recipe": recipe,
            "contract": dict(routed_experts.CONTRACT),
            "tensors": len(coverage),
            "expert_projections": len(self.expert_names),
            "scalars": dict(scalars),
            "bytes_read": bytes_read,
            "reads": reads,
            "coverage": coverage,
        }
        self.usable = True
        return self.load_record

    # ------------------------------------------------------------------ computations

    @torch.no_grad()
    def moe(self, layer: int, h: torch.Tensor, capture: Optional[dict] = None) -> torch.Tensor:
        """Normalized input ``[T, hidden]`` -> MoE output after the one all-reduce.

        The replicated FP32 router and the diagnostic-mode all-rank agreement on its input, ids
        and weights are the BF16 component's; the experts and their combination run on the
        device (``routed_experts.nvfp4_routed_experts``) on this rank's packed slice. The rank
        partials are reduced once, in FP32, then rounded to BF16.
        """
        routing = self.route(layer, h)
        ids, weights = routing["ids"], routing["weights"]
        digests = self._digests_agree((h, ids, weights)) if self.diagnostics else None
        record = {} if capture is not None else None
        local = routed_experts.nvfp4_routed_experts(
            h,
            ids,
            weights,
            self.expert_operands(layer),
            tune_max_rows=routed_experts.EXPERTS_PER_TOKEN * self.max_tokens,
            record=record,
        )
        # The four BF16 rank partials are summed once in FP32 and the sum is rounded once.
        out = self.all_reduce(local.float()).to(torch.bfloat16)
        if capture is not None:
            _rec(capture, _moe_capture(routing, record, out))
            if digests is not None:
                capture["moe.agreement_sha256"] = digests
        return out


def _moe_capture(routing: dict, record: dict, out: torch.Tensor) -> dict:
    """Routing intermediates, the provider's operator inputs and per-route outputs, the output.

    ``moe.routes`` ``[T, 8, 5120]`` holds each route's unweighted expert output in the router's
    slot order (``moe.ids``); ``moe.route_fc1_fp4`` / ``moe.route_fc1_sf_swizzled`` are the
    quantized operator rows (row ``8t + j`` is token ``t``'s route ``j``).
    """
    return {
        "moe.logits": routing["logits"],
        "moe.scores": routing["scores"],
        "moe.selection": routing["selection_scores"],
        "moe.group_scores": routing["group_scores"],
        "moe.groups": routing["groups"],
        "moe.masked": routing["masked_selection_scores"],
        "moe.ids": routing["ids"],
        "moe.weights": routing["weights"],
        "moe.route_fc1_fp4": record["route_fc1_fp4"],
        "moe.route_fc1_sf_swizzled": record["route_fc1_sf_swizzled"],
        "moe.routes": record["routes"],
        "moe.tp.local_out": record["local"],
        "moe.output": out,
        "mlp_out": out,
    }


__all__ = [
    "BANK_FIELDS",
    "FC1_KINDS",
    "BeamCheckpointLayoutNvfp4Sm103Tp4",
    "LoadError",
    "check_expert_scales",
]
