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

"""Routed-expert MoE LoRA for the CUTLASS leaves that can fuse it.

``torch.ops.trtllm.fused_moe`` runs the per-expert adapter GEMMs inside the
op, around its FC1 and FC2 grouped GEMMs, and only for unquantized and
per-tensor-FP8 weights -- every other format is rejected by the C++ op. Only
those two leaves mix this in, and only they declare ``supports_moe_lora=True``,
which the scheduler checks before it reaches for ``_moe_lora_active``.
"""

from collections.abc import Callable

import torch

from ....model_config import ModelConfig
from ....peft.lora.layer import (
    MOE_LORA_MODULE_NAMES,
    MOE_LORA_MODULE_TO_KERNEL_SLOT,
    LoraModuleType,
    MoeLoraLayer,
)
from ....peft.lora.validation import has_moe_lora_targets
from ..impl_contract import MoERunContext

#: An MoE LoRA module's ``(ranks, weight_ptrs)``.
_RanksAndPtrs = tuple[torch.Tensor, torch.Tensor]


def raise_moe_lora_multichunk_unsupported(num_chunks: int) -> None:
    """Reject multi-chunk execution for routed-expert MoE LoRA.

    Routed-expert MoE LoRA passes per-request/slot adapter metadata that is not
    re-sliced per token-chunk, so multi-chunk execution would mismatch the
    kernel's per-token expansion. Raised by the MoE scheduler.
    """
    raise NotImplementedError(
        f"Routed-expert MoE LoRA does not support multi-chunk execution "
        f"(num_chunks={num_chunks}). Reduce the per-forward token count or "
        f"increase `moe_max_num_tokens` so the MoE runs in a single chunk."
    )


def moe_lora_active(lora_params: dict | None, layer_idx: int | None) -> bool:
    """Return True when ``lora_params`` carries routed-expert MoE LoRA tensors
    for layer ``layer_idx``.

    ``lora_params`` describes every adapter of the forward, so a backend that
    cannot fuse MoE LoRA asks this rather than refusing any ``lora_params``.
    """
    if not lora_params or layer_idx is None:
        return False
    # CUDA-graph slot-indexed mode carries MoE LoRA in cuda_graph_params
    # rather than a per-layer eager dict (mirrors _extract_moe_lora_tensors),
    # so consult the graph layer map to keep the multi-chunk guard effective
    # during capture/replay.
    if lora_params.get("use_cuda_graph_mode", False):
        cuda_graph_params = lora_params.get("cuda_graph_params")
        if cuda_graph_params is None:
            return False
        layer_module2key = getattr(cuda_graph_params, "layer_module2key", {})
        return any(
            (layer_idx, int(LoraModuleType.from_string(name))) in layer_module2key
            for name in MOE_LORA_MODULE_NAMES
        )
    layer_params = lora_params.get(layer_idx, {})
    if not layer_params:
        return False
    return any(
        int(LoraModuleType.from_string(name)) in layer_params for name in MOE_LORA_MODULE_NAMES
    )


class CutlassMoELoraMixin:
    """The LoRA half of a CUTLASS leaf that supports routed-expert LoRA.

    Mixed in ahead of ``CutlassFusedMoEBase`` so ``_init_moe_lora`` and
    ``run_moe`` override the base's versions.
    """

    #: The activation dtype the leaf's ``quantize_input`` hands the op, when it
    #: is not the layer dtype (per-tensor FP8 casts to e4m3). Part of the
    #: FusedMoeRunner key the CUDA-graph workspace reservation must match.
    LORA_OP_INPUT_DTYPE: torch.dtype | None = None

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # Cutlass allocates its own intermediates.
        return self._run_fused_moe_op(
            ctx, lora_kwargs=self._extract_moe_lora_tensors(ctx.lora_params)
        )

    def _init_moe_lora(self, model_config: ModelConfig) -> None:
        self._moe_lora_enabled = self._has_moe_lora_targets(model_config)

        # Discovery-only marker submodule. The actual LoRA GEMMs are fused into
        # torch.ops.trtllm.fused_moe; MoeLoraLayer exists purely so that
        # CudaGraphLoraManager and the target-module validator can find this MoE
        # layer via isinstance(child, LoraLayer) traversal and read its
        # lora_module_types / output_hidden_sizes when building slot tables.
        self.lora = self._maybe_make_lora_marker(model_config)

    def _has_moe_lora_targets(self, model_config: ModelConfig) -> bool:
        """Whether this layer is in the routed-expert LoRA target-module set."""
        return has_moe_lora_targets(getattr(model_config, "lora_config", None))

    def _maybe_make_lora_marker(self, model_config: ModelConfig) -> MoeLoraLayer | None:
        """Construct a MoeLoraLayer marker iff this MoE layer is in the LoRA
        target-module set. The marker is a discovery-only submodule; the actual
        LoRA application is fused into torch.ops.trtllm.fused_moe.

        The output_hidden_sizes recorded here are the per-token outputs of the
        LoRA-side GEMM (not per-expert weight shapes): MOE_H_TO_4H / MOE_GATE
        produce intermediate_size, MOE_4H_TO_H produces hidden_size.
        """
        lora_config = getattr(model_config, "lora_config", None)
        if lora_config is None:
            return None
        # Normalize to lowercase to match has_moe_lora_targets (which lowercases
        # before comparing), so a mixed-case config marks the layer and builds
        # the discovery marker consistently.
        targets = {name.lower() for name in (getattr(lora_config, "lora_target_modules", []) or [])}
        active_modules: list[LoraModuleType] = []
        active_out_sizes: list[int] = []
        for name in MOE_LORA_MODULE_NAMES:
            if name not in targets:
                continue
            module_type = LoraModuleType.from_string(name)
            if name == "moe_4h_to_h":
                active_out_sizes.append(self.hidden_size)
            else:
                active_out_sizes.append(self.intermediate_size)
            active_modules.append(module_type)
        if not active_modules:
            return None
        return MoeLoraLayer(active_modules, active_out_sizes)

    def reserve_moe_lora_cuda_graph_workspace(
        self, max_num_tokens: int, max_lora_rank: int, max_lora_size: int
    ) -> None:
        """Pre-size the C++ FusedMoeRunner's MoE-LoRA scratch to the engine's
        worst case so no (re)allocation happens during CUDA graph capture or
        replay (which would dangle addresses baked into earlier graphs).

        No-op for layers without MoE LoRA targets; idempotent and grow-only.
        Call during warmup, before any capture that exercises MoE LoRA;
        CudaGraphLoraManager does this automatically.

        Args:
            max_num_tokens: Larger of the captured-decode capacity
                (max_batch_size * max_tokens_per_seq) and the engine-wide token limit.
            max_lora_rank: Largest LoRA rank across adapters.
            max_lora_size: Adapter-slot pool size for the slot-indexed device tables.
        """
        if not self._moe_lora_enabled or max_num_tokens <= 0:
            return
        # Weights must exist to read the runner's weight dtype. If they have not
        # been created yet, skip; the lazy sizing + in-capture guard still
        # protect correctness.
        if getattr(self, "w3_w1_weight", None) is None:
            return

        # The reservation must cover the engine's worst case, otherwise the first
        # capture hits a lazy allocation that the C++ in-capture guard rejects.
        assert max_lora_rank > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_rank > 0 "
            f"(got {max_lora_rank}); set lora_config.max_lora_rank."
        )
        assert max_lora_size > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_size > 0 "
            f"(got {max_lora_size})."
        )

        # The reservation must land on the *same* cached C++ FusedMoeRunner that
        # the runtime torch.ops.trtllm.fused_moe op uses on this layer, so the
        # MoERunner instance key is built from what _run_fused_moe_op hands the
        # op: the leaf's FUSED_MOE_OP_FLAGS, its weight view, and the
        # activation dtype its quantize_input produces. The output (LoRA
        # compute) dtype is the layer dtype. If a runtime call ever uses a
        # different key, the C++ capture guard surfaces a clear error rather
        # than corrupting replay.
        x_dtype = self.LORA_OP_INPUT_DTYPE
        if x_dtype is None:
            assert self.dtype in (torch.float16, torch.bfloat16), (
                "MoE LoRA requires fp16/bf16 activations to reserve a "
                f"deterministic FusedMoeRunner key; got {self.dtype}."
            )
            x_dtype = self.dtype
        flags = self.FUSED_MOE_OP_FLAGS

        from ....custom_ops.torch_custom_ops import MoERunner

        runner = MoERunner(
            x_dtype=x_dtype,
            weight_dtype=flags.weight_dtype or self.w3_w1_weight.dtype,
            output_dtype=self.dtype,
            top_k=self.routing_method.experts_per_token,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            **flags.op_kwargs(),
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            activation_type=self.activation_type,
        )
        runner.fused_moe_runner.reserve_lora_host_buffers(
            int(max_num_tokens),
            int(self.routing_method.experts_per_token),
            int(max_lora_rank),
            int(max_lora_size),
            bool(self.is_gated_activation),
        )

    def _moe_lora_active(self, lora_params: dict | None) -> bool:
        """Return True when lora_params carries routed-expert MoE LoRA tensors
        for this layer, meaning run_moe would fuse a LoRA delta.
        """
        return moe_lora_active(lora_params, self.layer_idx)

    @staticmethod
    def _empty_kernel_slot_dict() -> dict[str, torch.Tensor | None]:
        return {"fc1": None, "fc2": None, "gated": None}

    def _gather_moe_lora_slots(
        self, source: Callable[[LoraModuleType], _RanksAndPtrs | None]
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:
        """Gather per-kernel-slot (ranks, weight_ptrs) tensors.

        `source(module_type)` returns the (ranks, weight_ptrs) pair for an MoE
        LoRA module, or None if absent. Returns (ranks_by_slot, ptrs_by_slot)
        dicts keyed by the kernel slot ("fc1"/"gated"/"fc2"); see
        MOE_LORA_MODULE_TO_KERNEL_SLOT for the module->slot convention. Shared by
        the eager (per-request) and CUDA-graph (slot-indexed) extraction paths.
        """
        ranks = self._empty_kernel_slot_dict()
        ptrs = self._empty_kernel_slot_dict()
        for module_type, slot in MOE_LORA_MODULE_TO_KERNEL_SLOT.items():
            got = source(module_type)
            if got is None:
                continue
            ranks[slot], ptrs[slot] = got
        return ranks, ptrs

    @staticmethod
    def _require_fc1_fc2(ranks: dict[str, torch.Tensor | None]) -> None:
        """The kernel always dereferences the fc1 and fc2 rank/pointer arrays
        (see setupLoraWorkspace in moe_kernels.cu), so moe_h_to_4h (fc1/gate) and
        moe_4h_to_h (fc2/down) must both be present when MoE LoRA is active. The
        gated slot (moe_gate) is only read for gated activations.
        """
        if ranks["fc1"] is None or ranks["fc2"] is None:
            raise ValueError(
                "MoE LoRA requires both `moe_h_to_4h` (gate/SiLU) and "
                "`moe_4h_to_h` (down) in lora_target_modules."
            )

    def _extract_moe_lora_tensors(self, lora_params: dict | None) -> dict[str, object] | None:
        """Pick the MoE-side LoRA tensors out of the global `lora_params` dict
        for this layer. Returns a dict with the kwargs expected by
        `torch.ops.trtllm.fused_moe`, or None when no MoE LoRA applies.

        Each entry is a CPU tensor:
            *_lora_ranks         : int32  [num_seqs]
            *_lora_weight_ptrs   : int64  [num_seqs, 3]   (A, B, DoRA_unused)
            host_request_types   : int32  [num_seqs]      (0=CTX, 1=GEN)
            host_context_lengths : int32  [num_seqs]
            lora_max_low_rank    : int (max rank across the active modules)
        """
        if not lora_params:
            return None
        # Slot-indexed (CUDA-graph decode) path: the per-token expansion is
        # driven inside the op by token_to_slot indexed into stable slot
        # tables owned by CudaGraphLoraParams (see _extract_moe_lora_tensors_cuda_graph).
        if lora_params.get("use_cuda_graph_mode", False):
            return self._extract_moe_lora_tensors_cuda_graph(lora_params)
        layer_params = lora_params.get(self.layer_idx, {}) if self.layer_idx is not None else {}
        if not layer_params:
            return None

        # Gather (ranks, weight_ptrs) per kernel slot. weight_pointers is built
        # flat ([num_seqs * 3], row-major (A, B, DoRA) per seq) in
        # ModelEngine._build_lora_params; the op expects [num_seqs, 3].
        active_max_rank = 0

        def _source(module_type: LoraModuleType) -> _RanksAndPtrs | None:
            nonlocal active_max_rank
            entry = layer_params.get(int(module_type))
            if entry is None:
                return None
            rank_t = entry["adapter_size"]
            if rank_t.numel() > 0:
                active_max_rank = max(active_max_rank, int(rank_t.max().item()))
            return rank_t, entry["weight_pointers"].reshape(-1, 3)

        ranks, ptrs = self._gather_moe_lora_slots(_source)
        if all(v is None for v in ranks.values()):
            return None
        self._require_fc1_fc2(ranks)

        num_seqs = lora_params["num_seqs"]

        def _slice(t: torch.Tensor | None) -> torch.Tensor | None:
            return t[:num_seqs].contiguous() if t is not None else None

        return {
            "fc1_lora_ranks": _slice(ranks["fc1"]),
            "fc1_lora_weight_ptrs": _slice(ptrs["fc1"]),
            "fc2_lora_ranks": _slice(ranks["fc2"]),
            "fc2_lora_weight_ptrs": _slice(ptrs["fc2"]),
            "gated_lora_ranks": _slice(ranks["gated"]),
            "gated_lora_weight_ptrs": _slice(ptrs["gated"]),
            "host_request_types": _slice(lora_params["host_request_types"]),
            "host_context_lengths": _slice(lora_params["prompt_lens_cpu"]),
            "lora_max_low_rank": active_max_rank,
            "lora_dtype": lora_params.get("data_type"),
        }

    def _extract_moe_lora_tensors_cuda_graph(self, lora_params: dict) -> dict[str, object] | None:
        """CUDA-graph slot-indexed extraction for routed-expert MoE LoRA.

        Pulls per-module slot tables and token_to_slot out of
        CudaGraphLoraParams and returns the slot-indexed kwargs accepted by
        torch.ops.trtllm.fused_moe. Returns None when this layer does not
        carry any MoE LoRA modules in the graph layer map, and raises
        ``ValueError`` when it carries only one of fc1 / fc2, as the
        per-request path does.

        Returned tensor addresses are stable across captures and replays: they
        come from persistent pinned host buffers owned by CudaGraphLoraParams
        and the per-module packed pointer cache. Uses the same module->kernel
        slot convention as the per-request path (moe_h_to_4h -> fc1,
        moe_gate -> gated, moe_4h_to_h -> fc2).
        """
        if self.layer_idx is None:
            return None
        cuda_graph_params = lora_params.get("cuda_graph_params")
        if cuda_graph_params is None:
            return None

        def _source(module_type: LoraModuleType) -> _RanksAndPtrs | None:
            return cuda_graph_params.get_moe_slot_inputs(self.layer_idx, int(module_type))

        slot_ranks, slot_ptrs = self._gather_moe_lora_slots(_source)
        if slot_ranks["fc1"] is None and slot_ranks["fc2"] is None:
            return None
        self._require_fc1_fc2(slot_ranks)

        num_seqs = lora_params["num_seqs"]
        tokens_per_seq = getattr(cuda_graph_params, "max_tokens_per_seq", 1)
        num_tokens = num_seqs * max(int(tokens_per_seq), 1)
        token_to_slot = cuda_graph_params.token_to_slot_host[:num_tokens].contiguous()

        # Pass the global max LoRA rank, not the per-step active max: the device
        # path uses it only to size the low-rank workspace strides baked into the
        # captured graph, so the global max keeps them valid for any per-slot
        # rank across replays. The actual per-token rank is read on-device from
        # the slot table, so a smaller rank just runs a smaller GEMM.
        max_rank = int(getattr(cuda_graph_params, "max_rank", 0))
        if max_rank <= 0:
            return None

        return {
            "fc1_slot_lora_ranks": slot_ranks["fc1"].contiguous(),
            "fc1_slot_lora_weight_ptrs": slot_ptrs["fc1"].contiguous(),
            "fc2_slot_lora_ranks": slot_ranks["fc2"].contiguous(),
            "fc2_slot_lora_weight_ptrs": slot_ptrs["fc2"].contiguous(),
            "gated_slot_lora_ranks": (
                slot_ranks["gated"].contiguous() if slot_ranks["gated"] is not None else None
            ),
            "gated_slot_lora_weight_ptrs": (
                slot_ptrs["gated"].contiguous() if slot_ptrs["gated"] is not None else None
            ),
            "token_to_slot": token_to_slot,
            "lora_max_low_rank": max_rank,
            "lora_dtype": lora_params.get("data_type"),
        }
