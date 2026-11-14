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
replicated FP32 router with its diagnostic-mode all-rank agreement check, dispatch/combine, the
norms, depth residuals and head -- uses that component's code on
the checkpoint's BF16/FP32 tensors copied byte for byte. The experts stay packed:

    storage   per sparse layer and projection kind, banks indexed by global expert id: packed
              E2M1 codes ``[256, rows, K/2]`` u8 and E4M3 block scales in the GEMM's 128x4
              swizzled order ``[256, rows * K/16]`` u8 (the rank's slice, see ``layout``), and
              ``input_scale``, ``weight_scale_2``, ``g = 1/input_scale``, ``alpha =
              input_scale * weight_scale_2`` ``[256]`` fp32
    compute   ``expert.nvfp4_swiglu_expert``: per selected expert, activation quantization,
              three CUTLASS NVFP4 GEMMs and the fp32 SwiGLU, as the expert provider of
              ``beam_dispatch_combine``; the combination (bf16 routes, fp32 ascending-expert
              sum, x2.5, bf16) is unchanged, and the one MoE all-reduce sums the four bf16 rank
              partials in fp32 and rounds the sum to bf16 once

Loading streams bounded tiles straight into that storage and is a transaction: metadata
(config.json, hf_quant_config.json, every index/header entry) is validated against the whole
model before any storage changes, and the component is usable only after every rank loaded
every selected tensor. Packed codes, block scales and derived scalars are all registered
parameters.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional

import torch
from torch import nn

from .. import dispatch_combine
from .._tp4.component_bf16 import BeamCheckpointLayoutBf16Sm103Tp4, LoadError, _attr
from . import expert as nvfp4_expert
from .layout import (
    PROJECTIONS,
    QuantSpec,
    TensorSpec,
    check_partition,
    check_recipe,
    read_quant_config,
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
    validate,
)

#: Bank fields of one projection kind of one sparse layer.
BANK_FIELDS = ("weight", "weight_scale", "input_scale", "weight_scale_2", "g", "alpha")
_KINDS = tuple(p.removesuffix("_proj") for p in PROJECTIONS)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class BeamCheckpointLayoutNvfp4Sm103Tp4(BeamCheckpointLayoutBf16Sm103Tp4):
    """Rank-local real-checkpoint weights of the Beam model with packed NVFP4 experts.

    ``config`` is the checkpoint's config.json (a mapping); ``mapping`` this rank's
    ``tensorrt_llm.mapping.Mapping`` with world size 4, TP4, PP1, MoE TP4/EP1; ``layers`` the
    logical layers to hold (default all 52; the embedding, final norm and head are always
    held); ``max_tokens`` the most tokens one call accepts (default ``MAX_TOKENS``).
    Construction allocates uninitialized rank-local storage on the current CUDA device and
    validates the configuration and layout; it reads no checkpoint data.
    """

    _tensor_specs = staticmethod(tensor_specs)
    _check_partition = staticmethod(check_partition)

    def _allocate_storage(self) -> None:
        self.bank_names: list = []
        self.expert_names: list = []
        experts = self.config["num_experts"]
        for spec in self.specs.values():
            if isinstance(spec, QuantSpec):
                self.expert_names.append(spec.name)
                if spec.bank in self.bank_names:
                    continue
                rows, half = spec.local_packed_shape(self.rank)
                _, scale_cols = spec.local_scale_shape(self.rank)
                fields = {
                    "weight": ((experts, rows, half), torch.uint8),
                    "weight_scale": ((experts, swizzled_numel(rows, scale_cols)), torch.uint8),
                    "input_scale": ((experts,), torch.float32),
                    "weight_scale_2": ((experts,), torch.float32),
                    "g": ((experts,), torch.float32),
                    "alpha": ((experts,), torch.float32),
                }
                for field, (shape, dtype) in fields.items():
                    self.register_parameter(
                        _attr(f"{spec.bank}.{field}"),
                        nn.Parameter(
                            torch.empty(shape, dtype=dtype, device=self.device), requires_grad=False
                        ),
                    )
                self.bank_names.append(spec.bank)
                continue
            dtype = torch.bfloat16 if spec.effective_dtype == "BF16" else torch.float32
            param = nn.Parameter(
                torch.empty(spec.local_shape(self.rank), dtype=dtype, device=self.device),
                requires_grad=False,
            )
            self.register_parameter(_attr(spec.name), param)
        #: Per sparse layer, per global expert: gate and up input scales bitwise equal.
        self._shared_fc1: dict = {}

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

    def expert_storage(self, layer: int, expert: int, proj: str) -> dict:
        """Views of one expert projection's rank-local storage (``proj`` in ``PROJECTIONS``)."""
        spec = self.specs.get(f"model.layers.{layer}.mlp.experts.{expert}.{proj}")
        if not isinstance(spec, QuantSpec):
            raise ValueError(f"layer {layer} expert {expert} {proj!r} is not held")
        return {f: self.bank(spec.bank, f)[expert] for f in BANK_FIELDS}

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
        for kind in _KINDS:
            s = torch.empty(n_experts, dtype=torch.float32)
            w2 = torch.empty(n_experts, dtype=torch.float32)
            for e in range(n_experts):
                p = f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj"
                s[e], w2[e] = values[p + ".input_scale"], values[p + ".weight_scale_2"]
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

    def _validate_catalog(
        self, checkpoint_dir: str, catalog: SafetensorsCatalog, stored: dict
    ) -> dict:
        full = tensor_specs(self.config)
        recipe = check_recipe(read_quant_config(checkpoint_dir, stored), full)
        validate(catalog, stored_entries(full), complete=True)
        return {
            "recipe": recipe,
            "contract": dict(nvfp4_expert.CONTRACT),
            "expert_projections": len(self.expert_names),
        }

    def _reset_loaded_state(self) -> None:
        self._shared_fc1 = {}

    @staticmethod
    def _entry_name(spec: TensorSpec | QuantSpec) -> str:
        return spec.name + ".weight" if isinstance(spec, QuantSpec) else spec.name

    def _load_tensor(
        self,
        catalog: SafetensorsCatalog,
        reader: TileReader,
        spec: TensorSpec | QuantSpec,
        digest: Optional["hashlib._Hash"],
    ) -> dict:
        if isinstance(spec, QuantSpec):
            return self._load_expert(catalog, reader, spec, digest)
        record = super()._load_tensor(catalog, reader, spec, digest)
        # Packed expert ownership is recorded by _load_expert; dense tensors are unbanked.
        record.pop("bank")
        record.pop("slot")
        return record

    def _load_extra(self, catalog: SafetensorsCatalog, reader: TileReader) -> dict:
        scalars = {
            layer: self._load_scalars(catalog, reader, layer)
            for layer in self.layers
            if self.schedule.is_sparse(layer)
        }
        return {"scalars": scalars}

    # ------------------------------------------------------------------ computations

    @torch.no_grad()
    def moe(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """Normalized input ``[T, hidden]`` -> MoE output after the one all-reduce.

        The replicated FP32 router, the diagnostic-mode all-rank agreement on its input, ids
        and weights and the combination are the BF16 component's; each selected expert runs
        ``expert.nvfp4_swiglu_expert`` on this rank's packed slice. The rank partials are
        reduced once, in FP32, then rounded to BF16.
        """
        routing = self.route(layer, h)
        ids, weights = routing["ids"], routing["weights"]
        if self.diagnostics:
            self._digests_agree((h, ids, weights))
        bank = self.expert_bank(layer)
        shared = self._shared_fc1[layer]

        def provider(expert_id: int, rows: torch.Tensor) -> torch.Tensor:
            return nvfp4_expert.nvfp4_swiglu_expert(rows, bank, expert_id, shared[expert_id])

        local = dispatch_combine.dispatch_and_combine(h, ids, weights, provider)
        # The four BF16 rank partials are summed once in FP32 and the sum is rounded once.
        out = self.all_reduce(local.float()).to(torch.bfloat16)
        return out


__all__ = [
    "BANK_FIELDS",
    "BeamCheckpointLayoutNvfp4Sm103Tp4",
    "LoadError",
]
