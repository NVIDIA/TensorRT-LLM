# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam BF16 checkpoint-layout component / sm_103 / tp4: rank-local real weights and blocks.

Owned by the routed NVFP4 target (this directory): the NVFP4 component (``component``) builds
on it and the full model (``model``) on that. Constructed on its own it is a component, not a
target: a caller on each of four ranks constructs this class with the checkpoint's
configuration and its rank's ``Mapping`` (attention TP4, MoE TP4/EP1, PP1), loads a
caller-supplied checkpoint directory into the rank's slices, and runs representative
computations. It does not run a full-depth forward, prefill/decode loop or sampler.

Storage is rank-local and final: each logical tensor's slice (``layout_bf16``) lives in its own
parameter, except that each sparse layer keeps its 256 experts' slices in three contiguous
banks. Loading streams bounded tiles straight into that storage (``loader``) and is a
transaction: the component is usable only after every rank loaded every selected tensor.

Per call, rank r computes its 12 query heads / 2 kv heads, its dense or expert intermediate
slice and its vocabulary rows; the enclosing method applies exactly one collective at each
boundary, on the TRT-LLM NCCL all-reduce or all-gather::

    embed          masked local lookup, one all-reduce (sum)
    attention      local q/k/v/gate, q/k norms, RoPE or NoPE scale, TRTLLM attention on the
                   rank's kv heads, output gate, local o_proj partial, one all-reduce
    dense FFN      local gate/up, fp32 SwiGLU, local down partial, one all-reduce
    MoE            replicated fp32 router (global expert ids), in diagnostic mode an
                   all-rank agreement on the router input, ids and weights, rank-local
                   expert slices combined in ascending global expert order (bf16 routes,
                   fp32 sum, x2.5, bf16), one all-reduce
    head           final norm, local vocabulary logits, one all-gather along the
                   vocabulary in rank order, fp32 softcap once

"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed import AllReduce, allgather
from tensorrt_llm._torch.modeling_v2._router_index import (
    MODELING_V2_VALIDATE_ENV,
    step_contract_enabled,
)
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.beam.ops.activation.beam_logit_softcap import (
    beam_logit_softcap,
)
from tensorrt_llm._torch.modeling_v2.models.beam.ops.activation.swiglu_fp32 import swiglu_fp32
from tensorrt_llm._torch.modeling_v2.models.beam.ops.moe.beam_grouped_topk_router import (
    beam_grouped_topk_router,
)
from tensorrt_llm._torch.modeling_v2.models.beam.ops.norm.beam_rmsnorm_fp32 import beam_rmsnorm_fp32
from tensorrt_llm._torch.modeling_v2.models.beam.ops.residual.beam_depth_residual import (
    beam_depth_residual,
    depth_scales,
)
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping as ParallelMapping

from .. import _attention, dispatch_combine
from .._constants import ROUTER_EPS
from .._constants import SM as _SM
from .._load import load_failure_details
from .._validation import check_positions, explicit_device
from .layout_bf16 import (
    EMBED,
    FINAL_NORM,
    HEAD,
    WORLD_SIZE,
    TensorSpec,
    check_config,
    check_partition,
    check_rank,
    tensor_specs,
)
from .loader_bf16 import CheckpointFormatError, SafetensorsCatalog, TileReader, read_slice_into

#: Default tokens per call; the representative cases use at most 129. A caller that runs longer
#: calls (a full-model prefill) passes its own ``max_tokens`` at construction.
MAX_TOKENS = 1024


class LoadError(RuntimeError):
    """A load failed on at least one rank; the component is not usable on any rank."""


def check_max_tokens(max_tokens: int) -> int:
    """Return ``max_tokens`` if it is a positive int (the most tokens one call accepts)."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
    return max_tokens


def _check_mapping(mapping: ParallelMapping) -> None:
    check_rank(mapping.rank, mapping.world_size)
    expected = dict(tp_size=4, pp_size=1, moe_tp_size=4, moe_ep_size=1, cp_size=1)
    for field, value in expected.items():
        actual = getattr(mapping, field)
        if actual != value:
            raise ValueError(f"this layout needs Mapping {field}={value}, got {actual}")
    if mapping.enable_attention_dp:
        raise ValueError("this layout does not use attention data parallelism")
    if list(mapping.tp_group) != list(range(WORLD_SIZE)):
        raise ValueError(f"the TP group must be ranks 0..3, got {mapping.tp_group}")


class BeamCheckpointLayoutBf16Sm103Tp4(nn.Module):
    """Rank-local real-checkpoint weights of the Beam model on one of four TP ranks.

    ``config`` is the checkpoint's config.json (a mapping); ``mapping`` this rank's
    ``tensorrt_llm.mapping.Mapping`` with world size 4, TP4, PP1, MoE TP4/EP1; ``layers``
    the logical layers to hold (default all 52; the embedding, final norm and head are
    always held); ``max_tokens`` the most tokens one call accepts (default ``MAX_TOKENS``).
    Construction allocates uninitialized rank-local storage on the current CUDA device and
    validates the configuration and layout; it reads no checkpoint data.
    """

    _tensor_specs = staticmethod(tensor_specs)
    _check_partition = staticmethod(check_partition)

    def __init__(
        self,
        config: Mapping[str, Any],
        mapping: ParallelMapping,
        layers: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        *,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        super().__init__()
        self.max_tokens = check_max_tokens(max_tokens)
        self.schedule = check_config(config)
        _check_mapping(mapping)
        device = explicit_device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this component is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        self.config = dict(config)
        self.mapping = mapping
        self.rank = mapping.rank
        self.device = device
        self.layers = tuple(sorted(range(self.schedule.num_layers) if layers is None else layers))
        self.specs = self._tensor_specs(config, self.layers)
        self._check_partition(self.specs)
        self.hidden = config["hidden_size"]
        self.head_dim = config["head_dim"]
        self.local_heads = config["num_attention_heads"] // WORLD_SIZE
        self.local_kv_heads = config["num_key_value_heads"] // WORLD_SIZE
        self.eps = config["rms_norm_eps"]
        self.window = config["sliding_window"]
        self.rope_theta = config["rope_parameters"]["sliding_attention"]["rope_theta"]
        self.softcap = config["final_logit_softcapping"]
        self.vocab_interval = self.specs[EMBED].interval(self.rank)[1:]
        self._allocate_storage()
        self._allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
        self.usable = False
        self.load_record: Optional[dict] = None
        self._init_diagnostics()

    def _allocate_storage(self) -> None:
        #: Storage is always resolved from the currently registered parameters, never from
        #: references taken here, so a replaced parameter or bank is what every call uses.
        self.bank_names: list = []
        for spec in self.specs.values():
            if spec.bank is not None:
                if spec.bank not in self.bank_names:
                    local = spec.local_shape(self.rank)
                    bank = torch.empty(
                        (self.config["num_experts"], *local),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    self.register_parameter(
                        _attr(spec.bank), nn.Parameter(bank, requires_grad=False)
                    )
                    self.bank_names.append(spec.bank)
                continue
            dtype = torch.bfloat16 if spec.effective_dtype == "BF16" else torch.float32
            param = nn.Parameter(
                torch.empty(spec.local_shape(self.rank), dtype=dtype, device=self.device),
                requires_grad=False,
            )
            self.register_parameter(_attr(spec.name), param)

    # ------------------------------------------------------------------ diagnostics

    def _init_diagnostics(self) -> None:
        """Diagnostic validation mode: ``step_contract_enabled()``, read once per instance.

        On, each MoE layer also proves the four ranks routed identical bytes (an all-gather of
        host digests) and callers re-check per-call facts their own validation already fixed.
        Off (the default), none of that host work or diagnostic collective runs. Every rank
        must run in the same mode, since the agreement is a collective; ``load`` checks that
        before the component can run.
        """
        self.diagnostics = step_contract_enabled()

    def _check_diagnostics_agree(self) -> None:
        """Raise on every rank unless all four ranks run the same diagnostic mode (collective)."""
        flag = torch.tensor([1.0 if self.diagnostics else 0.0], device=self.device)
        on = int(self._allreduce(flag).item())
        if on not in (0, WORLD_SIZE):
            raise LoadError(
                f"rank {self.rank}: diagnostic validation is on for {on} of {WORLD_SIZE} ranks; "
                f"set {MODELING_V2_VALIDATE_ENV} alike on every rank"
            )

    # ------------------------------------------------------------------ storage

    def tensor(self, name: str) -> torch.Tensor:
        """The rank-local storage of logical tensor ``name`` (an expert's view of its bank slot)."""
        spec = self.specs[name]
        if spec.bank is not None:
            return self.bank(spec.bank)[spec.expert]
        return getattr(self, _attr(name))

    def bank(self, bank: str) -> torch.Tensor:
        """The current ``[256, ...]`` storage of one sparse layer's expert bank."""
        return getattr(self, _attr(bank))

    # ------------------------------------------------------------------ loading

    @torch.no_grad()
    def load(self, checkpoint_dir: str, *, record_digests: bool = True) -> dict:
        """Stream the selected tensors of a caller-supplied checkpoint into rank-local storage.

        Collective: every rank calls it. Metadata is validated against the whole model
        (every expected tensor and scale, nothing unsupported, exact dtypes and shapes) and
        the directory's config.json against the constructed configuration before any
        storage changes. The component is unusable from the first write until every rank
        reports a complete load; a failure on any rank raises ``LoadError`` on all ranks
        and leaves the component unusable. Returns the load record (coverage, source
        digests, bytes read).
        """
        self._check_diagnostics_agree()
        error: Optional[BaseException] = None
        catalog = None
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
            catalog = SafetensorsCatalog(checkpoint_dir)
            metadata = self._validate_catalog(checkpoint_dir, catalog, stored)
        except Exception as exc:  # Keep rank-local failures inside the collective protocol.
            error = exc
        if failure := load_failure_details(self.mapping, error is None, error):
            raise LoadError(
                f"rank {self.rank}: checkpoint rejected before loading: {failure}"
            ) from error

        self.usable = False
        self.load_record = None
        self._reset_loaded_state()
        coverage = OrderedDict()
        reader = None
        bytes_read = reads = 0
        try:
            reader = TileReader()
            order = sorted(
                self.specs.values(),
                key=lambda s: (
                    catalog.entries[self._entry_name(s)].path,
                    catalog.entries[self._entry_name(s)].begin,
                ),
            )
            for spec in order:
                digest = hashlib.sha256() if record_digests else None
                coverage[spec.name] = self._load_tensor(catalog, reader, spec, digest)
            extra = self._load_extra(catalog, reader)
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
        complete = error is None and list(coverage) == [s.name for s in order]
        if failure := load_failure_details(self.mapping, complete, error):
            self._reset_loaded_state()
            raise LoadError(
                f"rank {self.rank}: load did not complete on every rank: {failure}"
            ) from error
        self.load_record = {
            "rank": self.rank,
            "checkpoint_dir": os.fspath(checkpoint_dir),
            "layers": list(self.layers),
            "tensors": len(coverage),
            "bytes_read": bytes_read,
            "reads": reads,
            "coverage": coverage,
            **metadata,
            **extra,
        }
        self.usable = True
        return self.load_record

    def _validate_catalog(
        self, checkpoint_dir: str, catalog: SafetensorsCatalog, stored: dict
    ) -> dict:
        catalog.validate(self._tensor_specs(self.config), complete=True)
        return {}

    def _reset_loaded_state(self) -> None:
        """Clear format-specific state after storage changes or a failed transaction."""

    @staticmethod
    def _entry_name(spec: TensorSpec) -> str:
        return spec.name

    def _load_tensor(
        self,
        catalog: SafetensorsCatalog,
        reader: TileReader,
        spec: TensorSpec,
        digest: Optional["hashlib._Hash"],
    ) -> dict:
        dst = self.tensor(spec.name)
        read_slice_into(catalog, reader, spec, self.rank, dst, digest=digest)
        return {
            "interval": spec.interval(self.rank),
            "local_shape": list(dst.shape),
            "dtype": str(dst.dtype),
            "source_sha256": digest.hexdigest() if digest is not None else None,
            "bank": spec.bank,
            "slot": spec.expert,
        }

    def _load_extra(self, catalog: SafetensorsCatalog, reader: TileReader) -> dict:
        return {}

    # ------------------------------------------------------------------ helpers

    def _check_usable(self, layer: Optional[int] = None) -> None:
        if not self.usable:
            raise RuntimeError("the component is not usable: no complete load on every rank")
        if layer is not None and layer not in self.layers:
            raise ValueError(f"logical layer {layer} is not held by this component ({self.layers})")

    def _layer(self, layer: int, name: str) -> torch.Tensor:
        return self.tensor(f"model.layers.{layer}.{name}")

    def _bank(self, layer: int, which: str) -> torch.Tensor:
        return self.bank(f"layers.{layer}.experts.{which}")

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """The one sum over the four ranks at a partial-result boundary (NCCL)."""
        return self._allreduce(tensor)

    def _digests_agree(self, tensors: Sequence[torch.Tensor]) -> list:
        """Raise unless every rank holds byte-identical ``tensors``; return this rank's digests.

        Diagnostic (``self.diagnostics``): host hashing of device copies and one all-gather,
        called on every rank alike or on none.
        """
        digests = []
        for t in tensors:
            h = hashlib.sha256(f"{t.dtype}{list(t.shape)}".encode())
            h.update(memoryview(t.contiguous().cpu().view(torch.uint8).numpy().reshape(-1)))
            digests.append(h.digest())
        local = torch.frombuffer(bytearray(b"".join(digests)), dtype=torch.uint8).to(self.device)
        gathered = allgather(local, self.mapping, dim=0).view(WORLD_SIZE, -1).cpu()
        if not all(torch.equal(gathered[0], gathered[r]) for r in range(1, WORLD_SIZE)):
            raise RuntimeError(
                "replicated MoE routing differs between ranks (input, ids or weights)"
            )
        return [d.hex() for d in digests]

    # ------------------------------------------------------------------ computations

    @torch.no_grad()
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Global token ids ``[T]`` int64 -> replicated embeddings ``[T, hidden]`` bf16."""
        self._check_usable()
        _check_ids(token_ids, self.config["vocab_size"], self.device, self.max_tokens)
        lo, hi = self.vocab_interval
        inside = (token_ids >= lo) & (token_ids < hi)
        local = nn.functional.embedding(
            torch.where(inside, token_ids - lo, torch.zeros_like(token_ids)), self.tensor(EMBED)
        )
        part = torch.where(inside.unsqueeze(-1), local, torch.zeros_like(local))
        out = self.all_reduce(part)
        return out

    @torch.no_grad()
    def attention(
        self,
        layer: int,
        h: torch.Tensor,
        positions: torch.Tensor,
        md: TrtllmAttentionMetadata,
        *,
        call_checked: bool = False,
    ) -> torch.Tensor:
        """Normalized input ``[T, hidden]`` -> attention output after the one o_proj all-reduce.

        ``md`` is one prepared step of one or more requests (``_attention.check_batch_metadata``
        with one row per generation request) whose new tokens are the T rows, contexts first; ``positions`` are their
        absolute positions. ``call_checked`` states that the caller validated this call's
        metadata and its cache manager's every layer already (a full-depth call does so once);
        the per-layer re-check then runs only in diagnostic mode.
        """
        self._check_usable(layer)
        _check_hidden(h, self.hidden, self.device, self.max_tokens)
        T = h.shape[0]
        check_positions(positions, (T,), self.device)
        if not call_checked or self.diagnostics:
            _attention.check_batch_metadata(
                md,
                T,
                layer_idx=layer,
                num_kv_heads=self.local_kv_heads,
                head_dim=self.head_dim,
            )
        step = _attention.step_args(md, layer)
        sliding = self.schedule.is_sliding(layer)
        part = _attention.attention(
            h,
            positions,
            step,
            q_proj=self._layer(layer, "self_attn.q_proj.weight"),
            k_proj=self._layer(layer, "self_attn.k_proj.weight"),
            v_proj=self._layer(layer, "self_attn.v_proj.weight"),
            gate_proj=self._layer(layer, "self_attn.gate_proj.weight"),
            o_proj=self._layer(layer, "self_attn.o_proj.weight"),
            q_norm=self._layer(layer, "self_attn.q_norm.weight"),
            k_norm=self._layer(layer, "self_attn.k_norm.weight"),
            gate_bias=self._layer(layer, "self_attn.gate_bias"),
            num_heads=self.local_heads,
            num_kv_heads=self.local_kv_heads,
            head_dim=self.head_dim,
            eps=self.eps,
            rope_theta=self.rope_theta if sliding else None,
            window=self.window if sliding else step["max_seq_len"],
        )
        out = self.all_reduce(part)
        return out

    @torch.no_grad()
    def dense_ffn(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """Normalized input ``[T, hidden]`` -> dense SwiGLU output after the one down all-reduce."""
        self._check_usable(layer)
        _check_hidden(h, self.hidden, self.device, self.max_tokens)
        if self.schedule.is_sparse(layer):
            raise ValueError(f"layer {layer} has a sparse feed-forward")
        gate = cublas_mm(h, self._layer(layer, "mlp.gate_proj.weight").t())
        up = cublas_mm(h, self._layer(layer, "mlp.up_proj.weight").t())
        act = swiglu_fp32(gate, up)
        activation = act["act"]
        del gate, up, act
        part = cublas_mm(activation, self._layer(layer, "mlp.down_proj.weight").t())
        out = self.all_reduce(part)
        return out

    @torch.no_grad()
    def route(self, layer: int, h: torch.Tensor) -> dict:
        """The replicated FP32 router on ``h`` ``[T, hidden]``: every routing intermediate."""
        self._check_usable(layer)
        _check_hidden(h, self.hidden, self.device, self.max_tokens)
        if not self.schedule.is_sparse(layer):
            raise ValueError(f"layer {layer} has a dense feed-forward")
        return beam_grouped_topk_router(
            h,
            self._layer(layer, "mlp.gate.weight"),
            self._layer(layer, "mlp.gate.e_score_correction_bias"),
            n_group=self.config["n_group"],
            topk_group=self.config["topk_group"],
            top_k=self.config["num_experts_per_tok"],
            eps=ROUTER_EPS,
        )

    @torch.no_grad()
    def moe(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """Normalized input ``[T, hidden]`` -> MoE output after the one all-reduce.

        In diagnostic mode, before any expert runs, every rank's router input bytes, ordered
        expert ids and FP32 weights must be identical (one all-gather of their digests).
        """
        routing = self.route(layer, h)
        ids, weights = routing["ids"], routing["weights"]
        if self.diagnostics:
            self._digests_agree((h, ids, weights))
        gate_bank, up_bank, down_bank = (self._bank(layer, w) for w in ("gate", "up", "down"))

        def local_expert(expert_id: int, rows: torch.Tensor) -> torch.Tensor:
            g = cublas_mm(rows, gate_bank[expert_id].t())
            u = cublas_mm(rows, up_bank[expert_id].t())
            act = swiglu_fp32(g, u)
            y = cublas_mm(act["act"], down_bank[expert_id].t())
            return y

        local = dispatch_combine.dispatch_and_combine(h, ids, weights, local_expert)
        out = self.all_reduce(local)
        return out

    @torch.no_grad()
    def block(
        self,
        layer: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        md: TrtllmAttentionMetadata,
        *,
        call_checked: bool = False,
    ) -> torch.Tensor:
        """One decoder block of logical layer ``layer`` on ``x`` ``[T, hidden]``, one engine step.

        The input norm runs the source model's torch op order (``beam_rmsnorm_fp32``), so the
        q/k/v/gate projections read the source's normalized bytes for the same block input.
        The other norms use the flashinfer kernel. ``call_checked`` states that the
        full-model caller already validated the step metadata (see ``attention``).
        """
        self._check_usable(layer)
        p = f"model.layers.{layer}."
        h = beam_rmsnorm_fp32(x, self.tensor(p + "input_layernorm.weight"), self.eps)
        a = self.attention(
            layer,
            h,
            positions,
            md,
            call_checked=call_checked,
        )
        pa = flashinfer_rmsnorm(a, self.tensor(p + "post_attention_layernorm.weight"), self.eps)
        x1 = beam_depth_residual(pa, x, *depth_scales(layer, is_attention=True))
        pf = flashinfer_rmsnorm(x1, self.tensor(p + "pre_feedforward_layernorm.weight"), self.eps)
        m = self.moe(layer, pf) if self.schedule.is_sparse(layer) else self.dense_ffn(layer, pf)
        pm = flashinfer_rmsnorm(m, self.tensor(p + "post_feedforward_layernorm.weight"), self.eps)
        out = beam_depth_residual(pm, x1, *depth_scales(layer, is_attention=False))
        return out

    @torch.no_grad()
    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        self._check_usable()
        _check_hidden(h, self.hidden, self.device, self.max_tokens)
        out = flashinfer_rmsnorm(h, self.tensor(FINAL_NORM), self.eps)
        return out

    @torch.no_grad()
    def head(self, n: torch.Tensor) -> torch.Tensor:
        """Final-norm output ``[T, hidden]`` -> softcapped fp32 logits over the full vocabulary.

        The rank's vocabulary rows give local logits; one all-gather along the vocabulary
        axis restores global order (rank 0's rows first), and the softcap is applied once.
        """
        self._check_usable()
        _check_hidden(n, self.hidden, self.device, self.max_tokens)
        local = cublas_mm(n, self.tensor(HEAD).t())
        gathered = allgather(local, self.mapping, dim=-1)
        logits = beam_logit_softcap(gathered, self.softcap)
        return logits


def _attr(name: str) -> str:
    return name.replace(".", "__")


def _check_hidden(
    h: torch.Tensor, hidden: int, device: torch.device, max_tokens: int = MAX_TOKENS
) -> None:
    if not isinstance(h, torch.Tensor) or h.dtype != torch.bfloat16:
        raise ValueError("hidden states must be a torch.bfloat16 tensor")
    if h.dim() != 2 or h.shape[1] != hidden or not 1 <= h.shape[0] <= max_tokens:
        raise ValueError(
            f"hidden states must be [T, {hidden}] with 1 <= T <= {max_tokens}, got {list(h.shape)}"
        )
    if h.device != device or not h.is_contiguous():
        raise ValueError("hidden states must be contiguous on the component's device")


def _check_ids(
    ids: torch.Tensor, vocab: int, device: torch.device, max_tokens: int = MAX_TOKENS
) -> None:
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.dim() != 1:
        raise ValueError("token ids must be a 1-D torch.int64 tensor")
    if ids.device != device or not 1 <= ids.shape[0] <= max_tokens:
        raise ValueError(f"token ids must hold 1..{max_tokens} ids on the component's device")
    if bool(((ids < 0) | (ids >= vocab)).any()):
        raise ValueError(f"token ids must be in [0, {vocab})")


__all__ = [
    "BeamCheckpointLayoutBf16Sm103Tp4",
    "LoadError",
    "MAX_TOKENS",
    "check_max_tokens",
    "TensorSpec",
]
