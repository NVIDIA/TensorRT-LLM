# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts of one MoE layer on this rank (SM107 targets), as a chain of kernels.

:class:`K3RoutedExperts` runs one forward on its NVFP4 weights (:class:`K3Nvfp4Experts`, the
``backend`` attribute DWDP binds) with one of two CuTe DSL kernel sets:

``CUTEDSL`` (the context target)::

    routing (the caller's ``route``)
    fp4_quantize                                     [after the routing, on this stream]
    moe_sort                                         [unless the routing wrote the tables]
    moe_output_memset_inplace                        [aux stream]
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_{rubin,blackwell}   FC1 + SiTU + FP4 requant
    cute_dsl_nvfp4_grouped_gemm_finalize_inplace_{rubin,blackwell}    FC2 + weighted scatter-add

``CUTEDSL_FC12`` (the helix generation targets, SM107), at every token count::

    routing, fp4_quantize, moe_sort (as above)
    [counter fill + moe_output_memset_inplace]       unless zeroed ahead (``zeroed_output``)
    fused FC12 kernel                                FC1 + SiTU + FC2 + scatter-add

The forward routes first (the routing reads only the router logits), then joins the stream that
wrote ``x`` (``input_ready_event``), then quantizes ``x``: the routing overlaps the producer of
``x``. The output is this rank's partial sum (its experts, its slice of the intermediate dim); the
caller reduces it.

FC12 options, fixed by the target that builds the layer:

* ``sm_reserve``: SMs the FC12 grid leaves to the kernels another stream runs beside it.
* ``die_major``: on a two-die GPU, ``post_load_weights`` re-homes ``w3_w1_weight`` and
  ``w2_weight`` into die-alternating memory (2 MiB chunks alternating between the dies; same
  bytes and layout). ``die_sched_min_tokens``: from that many tokens per call the kernel's CTAs
  claim the tiles whose weight rows sit on their own die first.
* Pre-zeroed output: the caller allocates :meth:`K3RoutedExperts.new_zeroed_output` before it
  forks a side stream, zeroes it there (:func:`zero_on_side_stream`) and passes it to the forward,
  so the kernel directly follows the routing on this stream.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...kernels.k3r_moe.fc12 import fc12_counter_words


@dataclass(frozen=True)
class SortTables:
    """The outputs of ``trtllm::moe_sort`` (or of a routing kernel that writes the same tables) for
    this rank's experts at ``tile_size`` rows per tile: ``(tile_idx_to_group_idx,
    tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
    total_num_padded_tokens, num_non_exiting_tiles)``."""

    tile_size: int
    outputs: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class SortedRouting:
    """The routing of one forward: the top-k ids and weights, with this rank's ``moe_sort`` tables
    where the routing kernel wrote them (``None``: the experts run ``moe_sort``)."""

    token_selected_experts: torch.Tensor  # int32 [num_tokens, top_k]
    token_final_scales: torch.Tensor  # fp32 [num_tokens, top_k]
    sort_tables: Optional[SortTables]


@dataclass
class Fc12ZeroedOutput:
    """Output and FC12 counters of one forward, zeroed ahead of it.

    ``buffer`` (int32) holds the BF16 ``output`` ``[num_tokens, hidden_size]``, then, from a
    128-byte boundary, ``counters`` (enough words for every tile count the routing can produce).
    The caller zeroes ``buffer`` on a stream ordered after its allocation and records ``ready``
    there (:func:`zero_on_side_stream`).
    """

    buffer: torch.Tensor
    output: torch.Tensor
    counters: torch.Tensor
    ready: torch.cuda.Event


#: Values per NVFP4 block scale.
SF_VEC_SIZE = 16
#: FC1 rows (gate + up) are padded to a multiple of this.
_ROW_ALIGNMENT = 128

#: A route: ``route(router_logits, local_expert_offset, local_num_experts)``.
Route = Callable[[torch.Tensor, int, int], SortedRouting]

#: The kernel sets :class:`K3RoutedExperts` runs, by ``moe_config.backend``.
KERNEL_SETS = ("CUTEDSL", "CUTEDSL_FC12")
#: AutoTuner keys of the routing tile of the two kernel sets.
TILE_TUNER_KEYS = {
    "CUTEDSL": "CuteDslFusedMoE::run_moe_nvfp4",
    "CUTEDSL_FC12": "CuteDslFc12FusedMoE::run_moe_nvfp4",
}
#: Routing tiles the fused FC12 kernel runs (1-CTA and 2-CTA clusters).
FC12_TILE_SIZES = (128, 256)


def ep_partition(num_experts: int, ep_size: int, ep_rank: int) -> Tuple[int, int]:
    """``[start, end)`` of ``ep_rank``'s experts: ranks below ``num_experts % ep_size`` hold one
    expert more."""
    if num_experts < ep_size:
        raise ValueError(f"{num_experts} experts cannot be split over {ep_size} ranks")
    base, remainder = divmod(num_experts, ep_size)
    start = ep_rank * base + min(ep_rank, remainder)
    return start, start + base + (1 if ep_rank < remainder else 0)


def fc1_rows_for_kernels(x: torch.Tensor) -> torch.Tensor:
    """FC1 rows (dim 0: the up half, then the gate half) in the order the CuTe DSL FC1 kernels of
    this GPU read: Rubin (SM107) groups of 16 gate rows then 16 up rows, Blackwell groups of 64
    up rows then 64 gate rows. Weights and their block scales take the same order."""
    rows = x.shape[0]
    group, gate_first = (16, True) if get_sm_version() == 107 else (64, False)
    assert rows % (2 * group) == 0, (rows, group)
    halves = x.view(2, rows // (2 * group), group, *x.shape[1:])
    if gate_first:
        halves = halves.flip(0)
    return halves.transpose(0, 1).contiguous().view(x.shape)


def tp_slice(weight: torch.Tensor, tp_size: int, tp_rank: int, dim: int) -> torch.Tensor:
    """This rank's ceil-divided slice of ``weight`` along ``dim`` (``ceil(width / tp)`` per rank,
    the last rank the remainder); a dim of width 1 is not split."""
    if tp_size <= 1 or weight.shape[dim] == 1:
        return weight
    width = weight.shape[dim]
    step = -(-width // tp_size)
    return weight.narrow(dim, tp_rank * step, min((tp_rank + 1) * step, width) - tp_rank * step)


def _pad_to(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """``src`` zero-padded at the end of both dims to ``dst``'s 2D shape."""
    rows, cols = src.shape
    if (rows, cols) != tuple(dst.shape):
        src = torch.nn.functional.pad(
            src, (0, dst.shape[1] - cols, 0, dst.shape[0] - rows), "constant", 0
        ).contiguous()
    return src


def _swizzle_block_scales(dst: torch.Tensor) -> None:
    """Swizzle one expert's linear E4M3 block scales (int32-packed, in place) to the 128x4 layout."""
    swizzled = (
        torch.ops.trtllm.block_scale_interleave(dst.view(torch.uint8))
        .view(dst.dtype)
        .reshape(dst.shape)
    )
    torch.cuda.synchronize()
    dst.copy_(swizzled)


class K3Nvfp4Experts(nn.Module):
    """One MoE layer's NVFP4 routed experts on this rank.

    Holds experts ``[slot_start, slot_end)`` (all of them under DWDP once the setup widened the
    range) as the kernels take them:

    * ``w3_w1_weight`` ``[E, N1, H / 16]`` int64 (16 E2M1 values each) and ``w3_w1_weight_scale``
      ``[E, N1, H / 64]`` int32 (4 E4M3 block scales each, 128x4-swizzled), ``N1 = align128(2 I)``;
      FC1 rows in the gate/up order of the kernels of this GPU (:func:`fc1_rows_for_kernels`).
    * ``w2_weight`` ``[E, H, N1 / 32]`` int64 and ``w2_weight_scale`` ``[E, H, N1 / 128]`` int32
      (swizzled).
    * ``fc31_input_scale`` / ``fc2_input_scale``: the reciprocal of the largest checkpoint
      ``input_scale`` of the rank's experts (FC1 input, FC2 input); ``fc31_alpha`` / ``fc2_alpha``
      ``[E]``: the per-expert dequantization scales of the two GEMMs.

    The checkpoint streams in expert by expert (:meth:`load_expert`); :meth:`finalize` runs once
    the last of the rank's experts is in. The parameter names are the ones DWDP rebinds.

    ``tp_size`` / ``tp_rank``: MoE tensor parallelism, slicing every expert's intermediate dim (FC1
    rows, FC2 columns); ``ep_size`` / ``ep_rank``: which experts the rank holds.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
    ) -> None:
        super().__init__()
        if hidden_size % 4 != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by 4")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.ep_size, self.ep_rank = ep_size, ep_rank
        self.intermediate_size_per_partition = intermediate_size // tp_size
        self.scaling_vector_size = SF_VEC_SIZE
        # DWDP's fixup re-materializes activation constants only on modules that declare one.
        self.activation = None
        self.set_expert_range(*ep_partition(num_experts, ep_size, ep_rank))
        self._lock = threading.Lock()
        self._staged: Optional[Dict] = None
        self._streamed_expert_slots: set = set()

    def set_expert_range(self, start: int, end: int) -> None:
        """Hold experts ``[start, end)`` (before the weights are created)."""
        self.num_slots = self.num_experts
        self.expert_size_per_partition = end - start
        self.slot_start, self.slot_end = start, end
        self.initial_local_expert_ids = list(range(start, end))
        self.initial_global_assignments = list(range(self.num_experts))

    def create_weights(self) -> None:
        num_local = self.expert_size_per_partition
        n1 = -(-2 * self.intermediate_size_per_partition // _ROW_ALIGNMENT) * _ROW_ALIGNMENT
        h = self.hidden_size
        packed = 16  # E2M1 values per int64
        scales = 4  # E4M3 block scales per int32

        def param(shape, dtype, fill=None):
            data = (
                torch.empty(shape, dtype=dtype)
                if fill is None
                else torch.full(shape, fill, dtype=dtype)
            )
            return nn.Parameter(data, requires_grad=False)

        self.w3_w1_weight = param((num_local, n1, h // packed), torch.int64)
        self.w2_weight = param((num_local, h, n1 // 2 // packed), torch.int64)
        self.w3_w1_weight_scale = param(
            (num_local, n1, h // SF_VEC_SIZE // scales), torch.int32, fill=1
        )
        self.w2_weight_scale = param(
            (num_local, h, n1 // 2 // SF_VEC_SIZE // scales), torch.int32, fill=1
        )
        self.fc31_input_scale = param((), torch.float32, fill=1.0)
        self.fc2_input_scale = param((), torch.float32, fill=1.0)
        self.fc31_alpha = param((num_local,), torch.float32, fill=1.0)
        self.fc2_alpha = param((num_local,), torch.float32, fill=1.0)

    # ------------------------------------------------------------------
    # Loading: expert by expert, then one finalize per layer.
    # ------------------------------------------------------------------
    def prepare_streaming_load(self) -> None:
        """Create the per-layer staging the loader threads fill (before any of them runs)."""
        self._staged = {"input_scales": {}, "weight_scale_2": {}}
        self._streamed_expert_slots = set()

    def load_expert(
        self,
        *,
        global_expert_id: int,
        local_slot_id: int,
        w1_weight: torch.Tensor,
        w1_weight_scale: torch.Tensor,
        w1_weight_scale_2: torch.Tensor,
        w1_input_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_weight_scale_2: torch.Tensor,
        w2_input_scale: torch.Tensor,
        w3_weight: torch.Tensor,
        w3_weight_scale: torch.Tensor,
        w3_weight_scale_2: torch.Tensor,
        w3_input_scale: torch.Tensor,
    ) -> None:
        """Load one checkpoint expert (packed E2M1 ``uint8`` weights, E4M3 block scales, FP32
        global scales) into ``local_slot_id``: this rank's slice of the intermediate dim, FC1 as
        ``[w3 | w1]`` padded to ``N1`` rows, block scales swizzled."""
        if not 0 <= local_slot_id < self.expert_size_per_partition:
            raise IndexError(
                f"local_slot_id={local_slot_id} is outside [0, {self.expert_size_per_partition})"
            )
        if global_expert_id != self.initial_local_expert_ids[local_slot_id]:
            raise ValueError(
                f"local slot {local_slot_id} holds expert "
                f"{self.initial_local_expert_ids[local_slot_id]}, got {global_expert_id}"
            )
        if self._staged is None:
            raise RuntimeError("prepare_streaming_load() must run before load_expert()")
        if local_slot_id in self._streamed_expert_slots:
            raise ValueError(f"NVFP4 local slot {local_slot_id} was loaded twice")
        for name, value in (("w1", w1_weight), ("w2", w2_weight), ("w3", w3_weight)):
            if value.dtype != torch.uint8:
                raise TypeError(f"{name}_weight must hold packed NVFP4 uint8, got {value.dtype}")

        tp, rank = self.tp_size, self.tp_rank

        def column(t: torch.Tensor, device: torch.device) -> torch.Tensor:
            return tp_slice(t, tp, rank, 0).to(device)

        def row(t: torch.Tensor, device: torch.device) -> torch.Tensor:
            return tp_slice(t, tp, rank, 1).to(device)

        dst = self.w3_w1_weight.data[local_slot_id]
        w1 = column(w1_weight, dst.device)
        w3 = column(w3_weight, dst.device)
        fc1 = torch.cat([w3.contiguous().view(dst.dtype), w1.contiguous().view(dst.dtype)], dim=0)
        dst.copy_(_pad_to(fc1, dst), non_blocking=True)

        dst = self.w2_weight.data[local_slot_id]
        w2 = row(w2_weight, dst.device)
        dst.copy_(_pad_to(w2.contiguous().view(dst.dtype), dst), non_blocking=True)

        dst = self.w3_w1_weight_scale.data[local_slot_id]
        w1_s = column(w1_weight_scale, dst.device)
        w3_s = column(w3_weight_scale, dst.device)
        fc1_s = torch.cat(
            [w3_s.contiguous().view(dst.dtype), w1_s.contiguous().view(dst.dtype)], dim=0
        )
        dst.copy_(_pad_to(fc1_s, dst))
        _swizzle_block_scales(dst)

        dst = self.w2_weight_scale.data[local_slot_id]
        w2_s = row(w2_weight_scale, dst.device)
        cols = dst.shape[1] * 4
        if cols > w2_s.shape[1]:
            w2_s = torch.nn.functional.pad(
                w2_s, (0, cols - w2_s.shape[1]), "constant", 0
            ).contiguous()
        dst.copy_(_pad_to(w2_s.view(dst.dtype), dst))

        self._staged["weight_scale_2"][local_slot_id] = {
            "w1": w1_weight_scale_2,
            "w3": w3_weight_scale_2,
            "w2": w2_weight_scale_2,
        }
        self._staged["input_scales"][global_expert_id] = {
            "w1": w1_input_scale[...].reshape([]),
            "w3": w3_input_scale[...].reshape([]),
            "w2": w2_input_scale[...].reshape([]),
        }
        self._streamed_expert_slots.add(local_slot_id)

    def loaded_slots(self) -> set:
        return self._streamed_expert_slots

    def finalize(self) -> None:
        """After the last expert of the layer: swizzle the FC2 block scales, compute the global
        input scales and the per-expert alphas, and put FC1 rows and block scales in the kernels'
        gate/up order."""
        staged, self._staged = self._staged, None
        if staged is None:
            return
        for slot in range(self.w2_weight_scale.shape[0]):
            _swizzle_block_scales(self.w2_weight_scale.data[slot])

        fc31_values, fc2_values = [], []
        for expert_id, entry in staged["input_scales"].items():
            assert torch.allclose(entry["w1"], entry["w3"]), (
                f"expert {expert_id}: w1 input_scale != w3 input_scale"
            )
            fc31_values.append(entry["w1"])
            fc2_values.append(entry["w2"])
        if fc31_values:
            self.fc31_input_scale.data.copy_(torch.stack(fc31_values).max().reciprocal())
        if fc2_values:
            self.fc2_input_scale.data.copy_(torch.stack(fc2_values).max().reciprocal())

        fc31_input_scale = self.fc31_input_scale.data
        fc2_input_scale = self.fc2_input_scale.data
        for slot, scales in staged["weight_scale_2"].items():
            w1_ws2, w3_ws2 = scales["w1"], scales["w3"]
            if not torch.allclose(w1_ws2, w3_ws2):
                logger.warning(
                    f"w1_weight_scale_2 != w3_weight_scale_2 ({w1_ws2} != {w3_ws2}), "
                    "selecting the larger value. Accuracy may be affected."
                )
                w1_ws2 = torch.max(w1_ws2, w3_ws2)
            gate_ws2 = 1.0 / w1_ws2[...].reshape([])
            self.fc31_alpha.data[slot].copy_(1.0 / (fc31_input_scale * gate_ws2))
            w2_ws2 = 1.0 / scales["w2"][...].reshape([])
            self.fc2_alpha.data[slot].copy_(1.0 / (fc2_input_scale * w2_ws2))

        n = self.intermediate_size_per_partition * 2
        k = self.hidden_size
        sf_cols = k // SF_VEC_SIZE
        for slot in range(self.w3_w1_weight.shape[0]):
            weight = self.w3_w1_weight.data[slot]
            weight.copy_(fc1_rows_for_kernels(weight.view(torch.uint8)).view(weight.dtype))
            scale = self.w3_w1_weight_scale.data[slot]
            linear = torch.ops.trtllm.block_scale_interleave_reverse(
                scale.view(torch.uint8).view(-1, n, sf_cols)
            ).view(n, sf_cols)
            swizzled = torch.ops.trtllm.block_scale_interleave(
                fc1_rows_for_kernels(linear).view(-1, n, sf_cols)
            ).view(n, sf_cols)
            scale.copy_(swizzled.view(scale.dtype))

    # ------------------------------------------------------------------
    # Die placement (two-die GPUs).
    # ------------------------------------------------------------------
    def place_die_alternating(self, measure_sm_dies: bool) -> bool:
        """Re-home ``w3_w1_weight`` and ``w2_weight`` into die-alternating memory.

        Each weight is copied (same shape, dtype and bytes) into one VA range whose 2 MiB chunks
        alternate between the two dies, its parameter is pointed at the copy and the original is
        released. With ``measure_sm_dies`` the SM -> die table is measured here too, ahead of
        CUDA graph capture. Returns whether the weights are die-alternating (placing twice is a
        no-op): not on devices with one die, or when the driver does not home a chunk as
        requested.
        """
        from ...kernels.k3r_moe.die_locality import (
            die_alternating_copy,
            is_die_alternating,
            sm_die_table,
        )

        names = ("w3_w1_weight", "w2_weight")
        if all(is_die_alternating(getattr(self, name).data) for name in names):
            return True
        copies = []
        for name in names:
            copy = die_alternating_copy(getattr(self, name).data)
            if copy is None:
                logger.warning_once(
                    "Kimi K3 FC12 die-major weights: the device has no die-alternating memory; "
                    "the expert weights keep the default placement",
                    key="k3r_fc12_die_layout_unavailable",
                )
                return False
            copies.append(copy)
        for name, copy in zip(names, copies):
            getattr(self, name).data = copy
        del copies
        # The originals' blocks back to the driver: the next layer's copies are driver
        # allocations outside the caching allocator.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if measure_sm_dies:
            sm_die_table(self.w3_w1_weight.device.index)
        return True


# ----------------------------------------------------------------------
# Routing-tile autotuning: the tuned tactic of a forward is its routing tile.
# ----------------------------------------------------------------------
_TILE_RUNNERS: Dict[str, type] = {}


def _tile_runner_class(kernels: str) -> type:
    """The AutoTuner runner whose tactics are ``kernels``' routing tiles (built on first use: it
    needs the CuTe DSL)."""
    runner_cls = _TILE_RUNNERS.get(kernels)
    if runner_cls is not None:
        return runner_cls
    from tensorrt_llm._torch.autotuner import (
        ConstraintSpec,
        DynamicTensorSpec,
        OptimizationProfile,
        TunableRunner,
        TuningConfig,
    )
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
    from tensorrt_llm._torch.utils import (
        get_last_power_of_2_num_tokens_buckets,
        last_positive_power_of_2,
    )

    class _InputsHelper(GroupedGemmInputsHelper):
        """Profiles the routing of ``num_tokens`` tokens over the rank's experts, each expert at
        about the largest load."""

        def __init__(self, num_experts, top_k, num_local_experts, local_expert_offset) -> None:
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset

        def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
            return input_shapes[0][0]

        def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
            x, token_selected_experts, *others = inputs
            num_tokens = token_selected_experts.size(0)
            per_expert = self.generate_num_tokens_per_expert(num_tokens, approx_max_load=True)
            ids = []
            for expert, count in enumerate(per_expert, start=self.local_expert_offset):
                ids.extend([expert] * count)
            ids += [-1] * (num_tokens * self.top_k - len(ids))
            ids = torch.tensor(
                ids, dtype=token_selected_experts.dtype, device=token_selected_experts.device
            )
            return x, ids.view(self.top_k, num_tokens).transpose(0, 1).contiguous(), *others

    class _TileRunner(TunableRunner):
        """Tactic = routing tile; ``forward_impl(x, ids, scales, x_sf, moe_output, weights,
        enable_alltoall=False, tile_size=...)`` runs the kernel chain at it."""

        tuning_config_cache: Dict = {}

        def __init__(
            self,
            forward_impl: Callable,
            num_experts: int,
            top_k: int,
            num_local_experts: int,
            local_expert_offset: int,
            workload_identity: Optional[tuple] = None,
        ) -> None:
            super().__init__()
            self.forward_impl = forward_impl
            self.num_experts = num_experts
            self.top_k = top_k
            self.num_local_experts = num_local_experts
            self.local_expert_offset = local_expert_offset
            self.workload_identity = workload_identity

        def unique_id(self) -> tuple:
            # (num_experts, top_k, local experts, offset, finalize fusion, alltoall, output dtype,
            # scale vector), plus the epilogue where it tells two layers apart.
            identity = (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                True,
                False,
                torch.bfloat16,
                SF_VEC_SIZE,
            )
            if self.workload_identity is not None:
                identity += (self.workload_identity,)
            return identity

        def get_valid_tactics(
            self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
        ) -> List[int]:
            return self.tile_sizes()

        @staticmethod
        def tile_sizes() -> List[int]:
            # tile_size=512 runs on Rubin (SM107) only.
            return [128, 256, 512] if get_sm_version() == 107 else [128, 256]

        @staticmethod
        def tile_size_of(tactic: Optional[int]) -> int:
            """Routing tile of ``tactic``; the fallback tactic runs 128-row tiles."""
            return tactic if isinstance(tactic, int) and tactic > 0 else 128

        def get_tuning_config(self) -> TuningConfig:
            key = self.unique_id()
            if key not in self.__class__.tuning_config_cache:
                helper = _InputsHelper(
                    self.num_experts, self.top_k, self.num_local_experts, self.local_expert_offset
                )
                self.__class__.tuning_config_cache[key] = TuningConfig(
                    dynamic_tensor_specs=(
                        DynamicTensorSpec(
                            0, 0, get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2
                        ),
                    ),
                    constraint_specs=(
                        ConstraintSpec(1, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(3, 0, helper.infer_shape_num_tokens),
                        ConstraintSpec(4, 0, helper.infer_shape_num_tokens),
                    ),
                    inputs_pre_hook=helper.inputs_pre_hook,
                    use_cold_l2_cache=True,
                )
            return self.__class__.tuning_config_cache[key]

        def forward(
            self, inputs: List[torch.Tensor], tactic: Optional[int], do_preparation: bool = False
        ) -> torch.Tensor:
            if do_preparation:
                return inputs[4]
            return self.forward_impl(
                *inputs, enable_alltoall=False, tile_size=self.tile_size_of(tactic)
            )

    class _Fc12TileRunner(_TileRunner):
        """The fused FC12 kernel's routing tiles (1-CTA and 2-CTA clusters)."""

        tuning_config_cache: Dict = {}

        @staticmethod
        def tile_sizes() -> List[int]:
            return list(FC12_TILE_SIZES)

    _TILE_RUNNERS.update({"CUTEDSL": _TileRunner, "CUTEDSL_FC12": _Fc12TileRunner})
    return _TILE_RUNNERS[kernels]


# The FC12 pre-zeroing streams, by device index; every MoE layer uses the same one.
_FC12_PREZERO_STREAMS: Dict[int, torch.cuda.Stream] = {}


def fc12_prezero_stream() -> torch.cuda.Stream:
    """The FC12 pre-zeroing stream of the current device."""
    device = torch.cuda.current_device()
    stream = _FC12_PREZERO_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream()
        _FC12_PREZERO_STREAMS[device] = stream
    return stream


def zero_on_side_stream(zeroed: Fc12ZeroedOutput, fork_event: torch.cuda.Event) -> None:
    """Zero ``zeroed`` on the pre-zeroing stream once ``fork_event`` (recorded on the stream that
    allocated it, after the allocation) completes, and record ``zeroed.ready`` there."""
    with torch.cuda.stream(fc12_prezero_stream()):
        fork_event.wait()
        zeroed.buffer.zero_()
        zeroed.ready.record()


def _wait(event: Optional[torch.cuda.Event]) -> None:
    if event is not None:
        event.wait()


class K3RoutedExperts(nn.Module):
    """One MoE layer's routed experts on this rank (see the module docstring).

    ``mapping``: the routed experts' mapping (``moe_tp_size`` x ``moe_ep_size`` over the TP
    group). ``aux_stream``: the stream the CUTEDSL output memset forks to. ``max_chunk_tokens``:
    rows per launch chain (``moe_max_num_tokens``). ``sm_reserve``, ``die_major``,
    ``die_sched_min_tokens``: FC12 constants of the target (see the module docstring).
    """

    def __init__(
        self,
        *,
        kernels: str,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        layer_idx: int,
        situ_beta: float,
        situ_linear_beta: float,
        aux_stream: Optional[torch.cuda.Stream],
        max_chunk_tokens: int,
        sm_reserve: int = 0,
        die_major: bool = False,
        die_sched_min_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        if kernels not in KERNEL_SETS:
            raise ValueError(f"Kimi K3 routed experts run {KERNEL_SETS}, not {kernels!r}")
        fc12 = kernels == "CUTEDSL_FC12"
        if fc12 and (get_sm_version() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE):
            raise ValueError("CUTEDSL_FC12 runs on SM107 with the Rubin CuTe DSL only")
        if not fc12 and (sm_reserve or die_major or die_sched_min_tokens is not None):
            raise ValueError("the SM reserve and die-major options apply to CUTEDSL_FC12 only")
        if die_sched_min_tokens is not None and not die_major:
            raise ValueError("die-local tile claiming needs die-major expert weights")
        if sm_reserve < 0:
            raise ValueError(f"sm_reserve must be non-negative, got {sm_reserve}")
        if mapping.enable_attention_dp and mapping.tp_size > 1:
            raise NotImplementedError(
                "Kimi K3 CuTe DSL routed experts take the tokens of their own rank only "
                "(no attention-DP dispatch)"
            )
        self.kernels = kernels
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.situ_beta, self.situ_linear_beta = float(situ_beta), float(situ_linear_beta)
        self.moe_ep_size = mapping.moe_ep_size
        self.max_chunk_tokens = max(1, int(max_chunk_tokens))
        self.sm_reserve = int(sm_reserve)
        self.die_major = bool(die_major)
        self.die_sched_min_tokens = die_sched_min_tokens
        self.backend = K3Nvfp4Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=mapping.moe_tp_size,
            tp_rank=mapping.moe_tp_rank,
            ep_size=mapping.moe_ep_size,
            ep_rank=mapping.moe_ep_rank,
        )
        self.dwdp_manager = None
        self.enable_dwdp = False
        self._decide_dwdp()
        if self.enable_dwdp and self.die_major:
            raise NotImplementedError(
                "die-major expert weights do not combine with DWDP, which rebinds the weights"
            )
        self.backend.create_weights()
        self._aux_stream = aux_stream
        self._main_event = torch.cuda.Event()
        self._memset_event = torch.cuda.Event()
        # Signals that a pre-zeroed output of this layer is zero.
        self.zeroed_ready_event = torch.cuda.Event() if fc12 else None

    def _decide_dwdp(self) -> None:
        """Under a DWDP manager, store only this rank's range of its expert partition and register
        the layer; DWDP widens the range to all experts once it is set up."""
        from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager

        manager = get_global_dwdp_manager()
        if manager is None:
            return
        start, end = manager.expert_partition(self.backend.num_experts).range_of(manager.dwdp_rank)
        self.backend.set_expert_range(start, end)
        self.dwdp_manager = manager
        self.enable_dwdp = True
        manager.add_layer(layer_idx=self.layer_idx)

    @property
    def consumes_sort_tables(self) -> bool:
        """Whether the kernels can run on tables a routing kernel wrote (no slot remap)."""
        return self.backend.num_slots == self.backend.num_experts

    def post_load_weights(self) -> None:
        """Re-home the expert weights die-major where the target asks for it (and measure the
        SM -> die table for die-local claiming), ahead of CUDA graph capture."""
        if self.die_major:
            self.backend.place_die_alternating(
                measure_sm_dies=self.die_sched_min_tokens is not None
            )

    # ------------------------------------------------------------------
    # Output and FC12 counters zeroed ahead of a forward.
    # ------------------------------------------------------------------
    def new_zeroed_output(self, num_tokens: int, device: torch.device) -> Fc12ZeroedOutput:
        """Allocate, on the current stream, the output and FC12 counters of a forward of
        ``num_tokens`` tokens (not zeroed yet; see :func:`zero_on_side_stream`)."""
        if self.zeroed_ready_event is None:
            raise RuntimeError("a pre-zeroed output serves the CUTEDSL_FC12 kernels only")
        counter_words = fc12_counter_words(
            self.backend.expert_size_per_partition,
            num_tokens,
            self.top_k,
            die_sched=self.die_sched_min_tokens is not None,
        )
        out_elems = num_tokens * self.hidden_size
        # BF16 pairs per int32 word; the counters start on a 128-byte line.
        out_words = -(-(-(-out_elems // 2)) // 32) * 32
        buffer = torch.empty(out_words + counter_words, dtype=torch.int32, device=device)
        output = (
            buffer[:out_words].view(torch.bfloat16)[:out_elems].view(num_tokens, self.hidden_size)
        )
        return Fc12ZeroedOutput(buffer, output, buffer[out_words:], self.zeroed_ready_event)

    # ------------------------------------------------------------------
    # Input quantization.
    # ------------------------------------------------------------------
    def quantize_input(self, x: Union[torch.Tensor, Fp4QuantizedTensor]):
        """``(x, x_sf)``: packed E2M1 ``[M, H / 2]`` and linear E4M3 block scales ``[M, H / 16]``
        of ``x`` (BF16, quantized against ``fc31_input_scale``; or an unswizzled
        ``Fp4QuantizedTensor`` produced upstream with the same scale)."""
        rows = x.shape[0]
        if isinstance(x, Fp4QuantizedTensor):
            assert not x.is_sf_swizzled, "the routed experts take linear (unswizzled) block scales"
            x, x_sf = x.fp4_tensor, x.scaling_factor
        else:
            x, x_sf = fp4_quantize(x, self.backend.fc31_input_scale, SF_VEC_SIZE, False, False)
        return x, x_sf.view(rows, -(-self.hidden_size // SF_VEC_SIZE))

    def _aux(self) -> torch.cuda.Stream:
        if self._aux_stream is None:
            self._aux_stream = torch.cuda.Stream()
        return self._aux_stream

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        route: Route,
        routing: Optional[SortedRouting] = None,
        input_ready_event: Optional[torch.cuda.Event] = None,
        zeroed_output: Optional[Fc12ZeroedOutput] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """This rank's routed output ``[M, hidden]`` (BF16) of ``x`` (``[M, hidden]`` BF16 or
        NVFP4), routed by ``routing`` or else by ``route`` on ``router_logits``.

        ``input_ready_event``: ``x`` is written on another stream, joined after the routing and
        before the first read of ``x``. ``zeroed_output``: the FC12 output and counters zeroed
        ahead of this forward (:meth:`new_zeroed_output`); joined before the kernels and used
        where the forward is one FC12 launch of its shape, otherwise dropped.
        """
        if (
            routing is not None
            and routing.sort_tables is not None
            and not self.consumes_sort_tables
        ):
            raise ValueError("sort tables of a routing need experts without a slot remap")
        if self.enable_dwdp:
            self.dwdp_manager.wait_and_bind(self.backend, self.layer_idx)
        out = self._forward_chunked(
            x, router_logits, route, routing, input_ready_event, zeroed_output, all_rank_num_tokens
        )
        if self.enable_dwdp:
            self.dwdp_manager.finish_layer(self.layer_idx)
        return out

    def _forward_chunked(
        self, x, router_logits, route, routing, input_ready_event, zeroed, all_rank_num_tokens
    ):
        num_tokens = router_logits.shape[0]
        rows = max(all_rank_num_tokens) if self.enable_dwdp and all_rank_num_tokens else num_tokens
        num_chunks = max(1, -(-rows // self.max_chunk_tokens))
        if num_chunks == 1:
            return self._forward_chunk(x, router_logits, route, routing, input_ready_event, zeroed)
        if routing is not None:
            raise RuntimeError("a caller routing covers one launch chain, not several")
        _wait(input_ready_event)
        if zeroed is not None:
            zeroed.ready.wait()
        base, extra = divmod(num_tokens, num_chunks)
        sizes = [base + 1] * extra + [base] * (num_chunks - extra)
        if isinstance(x, Fp4QuantizedTensor):
            x_chunks = [
                Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)
                for codes, scales in zip(
                    x.fp4_tensor.split(sizes), x.scaling_factor.view(num_tokens, -1).split(sizes)
                )
            ]
        else:
            x_chunks = x.split(sizes)
        outs = [
            self._forward_chunk(xc, lc, route, None, None, None)
            for xc, lc in zip(x_chunks, router_logits.split(sizes))
        ]
        return torch.cat(outs)

    def _forward_chunk(self, x, router_logits, route, routing, input_ready_event, zeroed):
        """Route (reads only the router logits), join ``x``'s stream, quantize, run."""
        if routing is None:
            routing = route(
                router_logits, self.backend.slot_start, self.backend.expert_size_per_partition
            )
        _wait(input_ready_event)
        x_q, x_sf = self.quantize_input(x)
        return self._run_tuned(
            x_q,
            x_sf,
            routing.token_selected_experts,
            routing.token_final_scales,
            routing.sort_tables,
            zeroed,
        )

    def _take_zeroed(self, zeroed: Optional[Fc12ZeroedOutput], num_tokens: int):
        """``(output, counters)`` of ``zeroed`` where it serves this forward's FC12 launch (it is
        joined either way), else ``(None, None)``."""
        if zeroed is None:
            return None, None
        zeroed.ready.wait()
        if self.kernels != "CUTEDSL_FC12" or tuple(zeroed.output.shape) != (
            num_tokens,
            self.hidden_size,
        ):
            return None, None
        return zeroed.output, zeroed.counters

    def _run_tuned(self, x, x_sf, ids, scales, tables: Optional[SortTables], zeroed):
        """The AutoTuner's routing tile for this shape, then the kernel chain at it (on ``tables``
        where they are at that tile). The AutoTuner's profiling launches run on their own
        tensors; only the final launch takes the pre-zeroed counters."""
        b = self.backend
        num_tokens = scales.size(0)
        moe_output, counters = self._take_zeroed(zeroed, num_tokens)
        if moe_output is None:
            moe_output = torch.empty(
                (num_tokens, self.hidden_size), dtype=torch.bfloat16, device=x.device
            )
        if ids.size(0) == 0:
            return moe_output
        inputs = [x, ids, scales, x_sf, moe_output, b]
        fc12 = self.kernels == "CUTEDSL_FC12"
        from tensorrt_llm._torch.autotuner import AutoTuner

        runner = _tile_runner_class(self.kernels)(
            forward_impl=self._run,
            num_experts=b.num_slots,
            top_k=ids.size(-1),
            num_local_experts=b.expert_size_per_partition,
            local_expert_offset=b.slot_start,
            workload_identity=(
                (int(ActivationType.SiTu), self.situ_beta, self.situ_linear_beta) if fc12 else None
            ),
        )
        _, tactic = AutoTuner.get().choose_one(
            TILE_TUNER_KEYS[self.kernels], [runner], runner.get_tuning_config(), inputs
        )
        tile_size = runner.tile_size_of(tactic)
        if tables is not None and tile_size == tables.tile_size:
            return self._run(
                *inputs, tile_size=tile_size, sort_tables=tables, zeroed_counters=counters
            )
        return self._run(*inputs, tile_size=tile_size, zeroed_counters=counters)

    def _run(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        scales: torch.Tensor,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weights: K3Nvfp4Experts,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        sort_tables: Optional[SortTables] = None,
        zeroed_counters: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The kernel chain at ``tile_size`` (on ``sort_tables`` if given, else ``moe_sort``)."""
        del enable_alltoall
        top_k = ids.size(1)
        offset, num_local = weights.slot_start, weights.expert_size_per_partition
        if sort_tables is not None:
            assert sort_tables.tile_size == tile_size
            outs = sort_tables.outputs
        else:
            outs = torch.ops.trtllm.moe_sort(
                token_selected_experts=ids,
                token_final_scales=scales,
                num_experts=weights.num_slots,
                top_k=top_k,
                local_expert_offset=offset,
                local_num_experts=num_local,
                tile_tokens_dim=tile_size,
            )
        tile_to_expert, tile_mn_limit, expanded_to_permuted, permuted_to_expanded, _, num_tiles = (
            outs
        )
        fc1_weight = weights.w3_w1_weight.view(torch.float4_e2m1fn_x2)
        fc1_scale = weights.w3_w1_weight_scale.view(torch.uint8)
        fc2_weight = weights.w2_weight.view(torch.float4_e2m1fn_x2)
        fc2_scale = weights.w2_weight_scale.view(torch.uint8)
        if self.kernels == "CUTEDSL_FC12":
            from ...kernels.k3r_moe.fc12_runner import run_fc12

            run_fc12(
                x.view(torch.float4_e2m1fn_x2),
                fc1_weight,
                x_sf.view(torch.uint8),
                fc1_scale,
                weights.fc31_alpha,
                tile_to_expert,
                tile_mn_limit,
                permuted_to_expanded,
                num_tiles,
                weights.fc2_input_scale,
                fc2_weight,
                fc2_scale,
                weights.fc2_alpha,
                moe_output,
                scales,
                expanded_to_permuted,
                num_experts=weights.num_slots,
                top_k=top_k,
                num_local_experts=num_local,
                local_expert_offset=offset,
                tile_size=tile_size,
                ep_size=self.moe_ep_size,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                reserved_sms=self.sm_reserve,
                die_sched_min_tokens=self.die_sched_min_tokens,
                zeroed_counters=zeroed_counters,
            )
            return moe_output

        rubin = get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE
        memset_stream = self._aux()
        self._main_event.record()
        moe_output.record_stream(memset_stream)
        with torch.cuda.stream(memset_stream):
            self._main_event.wait()
            torch.ops.trtllm.moe_output_memset_inplace(
                input=moe_output,
                tile_idx_to_mn_limit=tile_mn_limit,
                expanded_idx_to_permuted_idx=expanded_to_permuted,
                permuted_idx_to_expanded_idx=permuted_to_expanded,
                num_non_exiting_tiles=num_tiles,
                tile_tokens_dim=tile_size,
                top_k=top_k,
                ep_size=self.moe_ep_size,
                enable_alltoall=False,
            )
            self._memset_event.record()
        fc1 = (
            torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin
            if rubin
            else torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell
        )
        fc1_kwargs = {"output_tensor": None, "output_sf_tensor": None} if rubin else {}
        h, h_sf = fc1(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=fc1_weight,
            input_scale=x_sf.view(torch.uint8),
            weight_scale=fc1_scale,
            alpha=weights.fc31_alpha,
            tile_idx_to_group_idx=tile_to_expert,
            tile_idx_to_mn_limit=tile_mn_limit,
            permuted_idx_to_expanded_idx=permuted_to_expanded,
            num_non_exiting_tiles=num_tiles,
            global_sf=weights.fc2_input_scale,
            num_experts=weights.num_slots,
            top_k=top_k,
            num_local_experts=num_local,
            local_expert_offset=offset,
            tile_size=tile_size,
            **fc1_kwargs,
            activation_type=ActivationType.SiTu,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )
        self._memset_event.wait()
        fc2 = (
            torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin
            if rubin
            else torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell
        )
        fc2_kwargs = {} if rubin else {"expert_counts": None, "expert_capacity": 0}
        fc2(
            input=h.view(torch.float4_e2m1fn_x2),
            weight=fc2_weight,
            input_scale=h_sf.view(torch.uint8),
            weight_scale=fc2_scale,
            alpha=weights.fc2_alpha,
            output=moe_output,
            tile_idx_to_group_idx=tile_to_expert,
            tile_idx_to_mn_limit=tile_mn_limit,
            permuted_idx_to_expanded_idx=permuted_to_expanded,
            num_non_exiting_tiles=num_tiles,
            token_final_scales=scales,
            num_experts=weights.num_slots,
            top_k=top_k,
            num_local_experts=num_local,
            local_expert_offset=offset,
            tile_size=tile_size,
            output_dtype=torch.bfloat16,
            **fc2_kwargs,
        )
        return moe_output


__all__ = [
    "FC12_TILE_SIZES",
    "Fc12ZeroedOutput",
    "K3Nvfp4Experts",
    "K3RoutedExperts",
    "KERNEL_SETS",
    "SortTables",
    "SortedRouting",
    "fc12_prezero_stream",
    "zero_on_side_stream",
]
