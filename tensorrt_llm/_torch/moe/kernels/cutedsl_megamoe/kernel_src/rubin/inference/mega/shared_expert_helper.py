# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-side shared-expert wiring common to the Rubin generation-phase and prefill MegaMoE kernels."""

import dataclasses
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cutlass_dsl import Int32

from .....api import ImplDesc, OptionalRequirement, ProblemDesc
from .....helpers.cute_py_helpers import Tcgen05MmaInstruction, make_tcgen05_tmem_plan
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.utils import ceil_div, round_up
from .....quant_def import SHARED_QUANT_KINDS, CombineFormat, QuantKind
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SharedEpilogueBindings, SwapABGatedActEpilogue


def shared_kernel_problem_requirements() -> dict:
    """Optional problem keys of a kernel with a fused shared expert."""
    return {
        # Shared expert fusion (optional): a dense mxfp8 x mxfp8 FFN over this rank's tokens with its own widths.
        "shared_expert": OptionalRequirement(bool),
        "shared_quant_kind": OptionalRequirement(Optional[str]),
        "shared_hidden_size": OptionalRequirement(Optional[int]),
        "shared_intermediate_gateup_size": OptionalRequirement(Optional[int]),
        "shared_output_size": OptionalRequirement(Optional[int]),
        # Add ``shared_output`` into ``output_activation`` in the separate reduce launch.
        "reduce_shared_output": OptionalRequirement(bool),
    }


def shared_kernel_impl_requirements() -> dict:
    """Optional impl keys of a kernel with a fused shared expert."""
    return {
        # Shared FC2 before ("early") or after ("tail") the routed work; "auto" (default) lets the kernel choose.
        "shared_l2_position": OptionalRequirement(Optional[str]),
        # Shared tile ids from their own atomic counters (default under atomic routed work) or by grid stride.
        "shared_work_id_mode": OptionalRequirement(Optional[str]),
        # Shared FC1 L2 swizzle: walk the tiles G feature blocks at a time so fewer weight bytes are live (0: off,
        # unset: the kernel's default).
        "shared_fc1_swizzle": OptionalRequirement(Optional[int]),
    }


def _rank3_view(tensor: cute.Tensor, rows, cols, row_stride, col_stride) -> cute.Tensor:
    # (rows, cols, 1) with a non-zero L stride: every TMA load map has rank 3 (one atom type per group).
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout((rows, cols, Int32(1)), stride=(row_stride, col_stride, Int32(row_stride))),
    )


class SharedExpertKernelMixin:
    """Shared-expert binding for a MegaMoE kernel class that declares the three ``shared_*_region`` names."""

    @staticmethod
    def check_shared_dims(shared: dict) -> None:
        """Raise unless the shared kind is an mxfp8 kind and the shared GEMM sizes are positive static ints in whole SF
        atoms / tiles."""
        if shared["shared_quant_kind"] not in SHARED_QUANT_KINDS:
            raise ValueError(
                f"shared_quant_kind must be one of {sorted(SHARED_QUANT_KINDS)}, got {shared['shared_quant_kind']!r}."
            )
        for key in ("shared_hidden_size", "shared_intermediate_gateup_size", "shared_output_size"):
            value = shared[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive static int, got {value!r}.")
        sf_atom = QuantKind(shared["shared_quant_kind"]).sf_vec_size * 4
        gateup = shared["shared_intermediate_gateup_size"]
        if shared["shared_hidden_size"] % sf_atom != 0:
            raise ValueError(f"shared_hidden_size must be a multiple of {sf_atom}, got {shared['shared_hidden_size']}.")
        if gateup % 128 != 0 or (gateup // 2) % sf_atom != 0:
            raise ValueError(
                f"shared_intermediate_gateup_size must be a multiple of 128 whose half is a multiple of {sf_atom}, "
                f"got {gateup}."
            )
        if shared["shared_output_size"] % 64 != 0:
            raise ValueError(f"shared_output_size must be a multiple of 64, got {shared['shared_output_size']}.")

    def _resolve_shared_expert(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        """Bind the optional shared-expert fields; partial or unsupported combinations raise."""
        self.shared_expert = bool(problem_desc.get("shared_expert", False))
        shared_keys = (
            "shared_quant_kind",
            "shared_hidden_size",
            "shared_intermediate_gateup_size",
            "shared_output_size",
        )
        supplied = {key: problem_desc.get(key) for key in shared_keys}
        position = impl_desc.get("shared_l2_position")
        self.shared_work_id_mode = impl_desc.get("shared_work_id_mode")
        self.shared_fc1_swizzle = impl_desc.get("shared_fc1_swizzle")
        self.reduce_shared_output = bool(problem_desc.get("reduce_shared_output", False))
        if not self.shared_expert:
            stray = [key for key, value in supplied.items() if value is not None]
            if self.reduce_shared_output:
                stray.append("reduce_shared_output")
            if position is not None:
                stray.append("shared_l2_position")
            if self.shared_work_id_mode is not None:
                stray.append("shared_work_id_mode")
            if self.shared_fc1_swizzle is not None:
                stray.append("shared_fc1_swizzle")
            if stray:
                raise ValueError(f"{', '.join(stray)} require shared_expert=True.")
            self.shared_quant_kind = None
            self.shared_hidden_size = None
            self.shared_intermediate_gateup_size = None
            self.shared_output_size = None
            self._shared_l2_position_request = None
            self.heterogeneous_shared = False
            return
        missing = [key for key, value in supplied.items() if value is None]
        if missing:
            raise ValueError(f"shared_expert=True requires {', '.join(missing)}.")
        self.check_shared_dims(supplied)
        self.shared_quant_kind = QuantKind(supplied["shared_quant_kind"])
        self.shared_hidden_size = supplied["shared_hidden_size"]
        self.shared_intermediate_gateup_size = supplied["shared_intermediate_gateup_size"]
        self.shared_output_size = supplied["shared_output_size"]
        self.heterogeneous_shared = self.shared_quant_kind != self.quant_kind
        if self.helper_expert_count > 0:
            raise ValueError("shared_expert is not supported together with helper (DLB) experts.")
        if self.reduce_shared_output:
            if self.reduce_topk_in_kernel:
                raise ValueError("reduce_shared_output adds the shared term in the TopkReduce launch; it needs "
                                 "reduce_topk_in_kernel=False.")
            if self.shared_output_size != self.hidden_size:
                raise ValueError(f"reduce_shared_output needs shared_output_size == hidden_size, got "
                                 f"{self.shared_output_size} vs {self.hidden_size}.")
        # The scheduler validates and resolves the request into ``shared_l2_position``.
        self._shared_l2_position_request = position

    @property
    def shared_l2_position(self) -> Optional[str]:
        """The resolved shared FC2 position (``early`` or ``tail``), or None without a shared expert."""
        return self.scheduler.shared_fc2_position

    @property
    def shared_tile_k(self) -> int:
        """K of one shared MMA tile: the routed instruction count per stage times the shared kind's instruction K."""
        return (self.mma_tiler_mnk[2] // self.mma_instruction_mnk[2]) * self.shared_quant_kind.instruction_k(
            self.mma_k_mode
        )

    def _merge_shared_tmem_plan(self, routed_plan, mma_cta_count: int):
        """The routed TMEM plan with the larger of the two kinds' scale-factor windows."""
        shared_instruction_k = self.shared_quant_kind.instruction_k(self.mma_k_mode)
        shared_tile_k = self.shared_tile_k
        shared_instruction = Tcgen05MmaInstruction(
            a_type=self.shared_quant_kind.weight_dtype,
            b_type=self.shared_quant_kind.activation_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=(*self.mma_instruction_mnk[:2], shared_instruction_k),
            participates=mma_cta_count,
            sfa_type=self.shared_quant_kind.sf_dtype,
            sfb_type=self.shared_quant_kind.sf_dtype,
            sf_vec_size=self.shared_quant_kind.sf_vec_size,
        )
        shared_plan = make_tcgen05_tmem_plan(
            shared_instruction, self.architecture, (*self.mma_tiler_mnk[:2], shared_tile_k)
        )
        for field_name in (
            "accumulator_columns",
            "accumulator_stage_columns",
            "accumulator_stage_count",
            "accumulator_stage_stride_columns",
            "accumulator_pipeline_stages",
        ):
            if getattr(routed_plan, field_name) != getattr(shared_plan, field_name):
                raise ValueError(
                    f"shared_quant_kind {self.shared_quant_kind} plans TMEM {field_name}="
                    f"{getattr(shared_plan, field_name)} while quant_kind {self.quant_kind} plans "
                    f"{getattr(routed_plan, field_name)}; the two kinds cannot share one accumulator staging."
                )
        return dataclasses.replace(
            routed_plan,
            allocation_columns=max(routed_plan.allocation_columns, shared_plan.allocation_columns),
            sfa_columns=max(routed_plan.sfa_columns, shared_plan.sfa_columns),
            sfb_columns=max(routed_plan.sfb_columns, shared_plan.sfb_columns),
        )

    def _make_shared_epilogue(self, fc12_problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc) -> None:
        """Second epilogue instance for the shared expert; it borrows the routed scratch SMEM."""
        self.shared_epilogue = None
        if not self.shared_expert:
            return
        shared_epilogue_problem_desc = ProblemDesc(
            {
                **fc12_problem_desc,
                "quant_kind": self.shared_quant_kind.value,
                "hidden_size": self.shared_output_size,
                "intermediate_gateup_size": self.shared_intermediate_gateup_size,
                "expert_count": 1,
                # shared_output is a local bf16 tensor whatever the routed combine wire format is.
                "combine_format": CombineFormat.parse("bf16"),
            }
        )
        shared_epilogue_impl_desc = ImplDesc(
            {
                **resolved_impl_desc,
                "communication_enabled": False,
                # Communication off makes the bulk path the local FC2 TMA store.
                "fc2_use_bulk": True,
                "reduce_topk_in_kernel": False,
                "token_back_push_data": False,
                # shared_output is a plain local tensor; only the routed combine plane is self-validating.
                "combine_ll_encoded": False,
                # Shared FC2 tiles keep their own rows: the ring of the local FC2 output holds routed tiles only.
                "fc2_output_ring_blocks": 0,
            }
        )
        self.shared_epilogue = SwapABGatedActEpilogue(shared_epilogue_problem_desc, shared_epilogue_impl_desc)
        self.shared_fc2_spin_threshold = ceil_div(self.shared_intermediate_gateup_size, self.cta_tile_m)

    def _check_shared_tile_k(self) -> None:
        if self.shared_expert and self.shared_tile_k % self._mainloop.shared_mma_tile_k != 0:
            # The TMEM plan sized the shared scale-factor window for this K; the mainloop may load a divisor of it.
            raise ValueError(
                f"Shared tile K differs: mainloop {self._mainloop.shared_mma_tile_k} vs kernel {self.shared_tile_k}."
            )

    def _shared_scheduler_impl_fields(self, position: Optional[str] = None, fc1_swizzle: Optional[int] = None) -> dict:
        """Impl keys that hand the shared tile stream to the scheduler; the arguments override the requested values."""
        return {
            "shared_expert_stream": self.shared_expert,
            "shared_l2_position": position if position is not None else self._shared_l2_position_request,
            "shared_work_id_mode": self.shared_work_id_mode,
            "shared_fc1_swizzle": fc1_swizzle if fc1_swizzle is not None else self.shared_fc1_swizzle,
        }

    def _shared_smem_kwargs(self) -> dict:
        """Epilogue SMEM arguments that widen the routed scratch for the shared FC1 / FC2 staging."""
        return dict(
            shared_fc1_staging_bytes=self.shared_epilogue.fc1_staging_bytes if self.shared_expert else 0,
            shared_fc2_staging_spec=self.shared_epilogue.fc2_staging_spec if self.shared_expert else None,
        )

    def _register_shared_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """The shared FC1 output pool, its scale factors and one done counter per shared token block."""
        if not self.shared_expert:
            return
        shared_intermediate_downproj = self.shared_intermediate_gateup_size // 2
        shared_sf_vec = self.shared_epilogue.sf_vec_size
        device_workspace.register(
            self.shared_fc1_output_region,
            self.shared_epilogue.fc1_output_dtype,
            (round_up(self.max_tokens_per_rank, 64), shared_intermediate_downproj),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.shared_fc1_output_sf_region,
            self.shared_epilogue.fc1_output_sf_dtype,
            (
                round_up(self.max_tokens_per_rank, 128),
                round_up(ceil_div(shared_intermediate_downproj, shared_sf_vec), 4),
            ),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.shared_fc1_done_counter_region,
            cutlass.Int32,
            (self.scheduler.shared_token_blocks,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )

    def _shared_name_suffix(self) -> str:
        if not self.shared_expert:
            return ""
        return (
            f"_shared{self.shared_quant_kind}_k{self.shared_hidden_size}_i{self.shared_intermediate_gateup_size}"
            f"_n{self.shared_output_size}"
            + ("" if self.shared_work_id_mode is None else f"_ids{self.shared_work_id_mode}")
            + f"_l2{self.shared_l2_position}"
            + (f"_fc1sw{self.scheduler._shared_fc1_swizzle}" if self.scheduler._shared_fc1_swizzle else "")
            + ("_reduced" if self.reduce_shared_output else "")
        )

    def _shared_fake_arguments(self, fake_tensor, tokens: int) -> dict:
        """AOT fake tensors for the trailing shared-expert operands (empty without a shared expert)."""
        if not self.shared_expert:
            return {}
        shared_hidden = self.shared_hidden_size
        shared_gateup = self.shared_intermediate_gateup_size
        shared_downproj = shared_gateup // 2
        shared_output_size = self.shared_output_size
        shared_vec = self.shared_quant_kind.sf_vec_size
        shared_activation_sf_elements = round_up(tokens, 128) * round_up(shared_hidden // shared_vec, 4)
        shared_fc1_weight_sf_columns = round_up(shared_gateup, 128) * round_up(shared_hidden // shared_vec, 4)
        shared_fc2_weight_sf_columns = round_up(shared_output_size, 128) * round_up(shared_downproj // shared_vec, 4)
        return dict(
            shared_activation=fake_tensor(
                self.shared_quant_kind.activation_dtype, (tokens, shared_hidden), (1, 0), {}, 16
            ),
            shared_activation_sf=fake_tensor(
                self.shared_quant_kind.sf_dtype, (shared_activation_sf_elements,), (0,), set(), 16
            ),
            shared_fc1_weight=fake_tensor(
                self.shared_quant_kind.weight_dtype, (1, shared_hidden, shared_gateup), (2, 0, 1), set(), 16
            ),
            shared_fc1_weight_sf=fake_tensor(
                self.shared_quant_kind.sf_dtype, (1, shared_fc1_weight_sf_columns), (1, 0), set(), 16
            ),
            shared_fc2_weight=fake_tensor(
                self.shared_quant_kind.weight_dtype, (1, shared_downproj, shared_output_size), (2, 0, 1), set(), 16
            ),
            shared_fc2_weight_sf=fake_tensor(
                self.shared_quant_kind.sf_dtype, (1, shared_fc2_weight_sf_columns), (1, 0), set(), 16
            ),
            # 32 B, like output_activation: the top-k reduce reads it in up to 256-bit accesses.
            shared_output=fake_tensor(cutlass.BFloat16, (tokens, shared_output_size), (1, 0), {}, 32),
        )

    def _check_shared_operands(self, operands: dict) -> None:
        """All seven shared operands with a shared expert and none without; dtypes follow ``shared_quant_kind``."""
        if not self.shared_expert:
            for operand_name, operand in operands.items():
                if operand is not None:
                    raise ValueError(f"{operand_name} requires a kernel built with shared_expert=True.")
            return
        for operand_name, operand in operands.items():
            if operand is None:
                raise ValueError(f"shared_expert=True requires {operand_name}.")
        kind = self.shared_quant_kind
        expected = {
            "shared_activation": kind.activation_dtype,
            "shared_activation_sf": kind.sf_dtype,
            "shared_fc1_weight": kind.weight_dtype,
            "shared_fc1_weight_sf": kind.sf_dtype,
            "shared_fc2_weight": kind.weight_dtype,
            "shared_fc2_weight_sf": kind.sf_dtype,
            "shared_output": cutlass.BFloat16,
        }
        for operand_name, operand in operands.items():
            if operand.element_type is not expected[operand_name]:
                raise TypeError(
                    f"shared {kind} requires {operand_name} to be {expected[operand_name]}, got {operand.element_type}."
                )

    def _prepare_shared_operands(
        self,
        shared_activation: cute.Tensor,
        shared_activation_sf: cute.Tensor,
        shared_fc1_weight: cute.Tensor,
        shared_fc1_weight_sf: cute.Tensor,
        shared_fc2_weight: cute.Tensor,
        shared_fc2_weight_sf: cute.Tensor,
        shared_output: cute.Tensor,
    ) -> Tuple:
        """Shared TMA load params (16), the shared FC1/FC2 TMA store atoms and tensors, and the FC2 output view."""
        if not self.shared_expert:
            return (None,) * 16, None, None, None, None, None
        singleton = Int32(1)
        shared_hidden = self.shared_hidden_size
        shared_gateup = self.shared_intermediate_gateup_size
        shared_downproj = shared_gateup // 2
        shared_output_size = self.shared_output_size
        shared_vec = self.shared_quant_kind.sf_vec_size
        tokens = self.max_tokens_per_rank
        shared_fc1_output = self._device_workspace.tensor(self.shared_fc1_output_region)
        shared_fc1_output_sf = self._device_workspace.tensor(self.shared_fc1_output_sf_region)

        def rewrite_tensor_shape(tensor: cute.Tensor, shape: Tuple) -> cute.Tensor:
            return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))

        shared_fc1_weight = rewrite_tensor_shape(shared_fc1_weight, (1, shared_hidden, shared_gateup))
        shared_fc2_weight = rewrite_tensor_shape(shared_fc2_weight, (1, shared_downproj, shared_output_size))
        shared_fc1_a = _rank3_view(
            shared_fc1_weight, Int32(shared_gateup), Int32(shared_hidden), shared_fc1_weight.stride[2],
            shared_fc1_weight.stride[1],
        )
        # Shared FC1 B is the dense token tensor: a plain tiled TMA leg, no gather.
        shared_fc1_b = _rank3_view(
            shared_activation, Int32(tokens), Int32(shared_hidden), shared_activation.stride[0],
            shared_activation.stride[1],
        )
        padded_shared_tokens = round_up(tokens, 128)
        expected_shared_activation_sf_elements = padded_shared_tokens * round_up(shared_hidden // shared_vec, 4)
        if isinstance(shared_activation_sf.shape[0], int):
            if shared_activation_sf.shape[0] != expected_shared_activation_sf_elements:
                raise ValueError(
                    f"shared_activation_sf must hold {expected_shared_activation_sf_elements} canonical elements, "
                    f"got {shared_activation_sf.shape[0]}."
                )
        shared_fc1_sfb = cute.make_tensor(
            shared_activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (Int32(padded_shared_tokens), Int32(shared_hidden), singleton), shared_vec
            ),
        )
        padded_shared_gateup = cute.round_up(shared_gateup, shared_vec * 4)
        expected_shared_fc1_weight_sf_columns = padded_shared_gateup * shared_hidden // shared_vec
        if isinstance(shared_fc1_weight_sf.shape[1], int):
            if shared_fc1_weight_sf.shape[1] != expected_shared_fc1_weight_sf_columns:
                raise ValueError("shared_fc1_weight_sf has an incompatible column count.")
        shared_fc1_sfa = cute.make_tensor(
            shared_fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (Int32(padded_shared_gateup), Int32(shared_hidden), singleton), shared_vec
            ),
        )
        shared_fc1_output_gemm = cute.make_tensor(
            shared_fc1_output.iterator,
            cute.make_layout(
                (shared_fc1_output.shape[0], Int32(shared_downproj), singleton),
                stride=(shared_fc1_output.stride[0], shared_fc1_output.stride[1], 0),
            ),
        )
        shared_fc2_b = _rank3_view(
            shared_fc1_output, Int32(shared_fc1_output.shape[0]), Int32(shared_downproj),
            shared_fc1_output.stride[0], shared_fc1_output.stride[1],
        )
        shared_fc2_a = _rank3_view(
            shared_fc2_weight, Int32(shared_output_size), Int32(shared_downproj), shared_fc2_weight.stride[2],
            shared_fc2_weight.stride[1],
        )
        padded_shared_output = cute.round_up(shared_output_size, 128)
        padded_shared_downproj = cute.round_up(shared_downproj, shared_vec * 4)
        expected_shared_fc2_weight_sf_columns = padded_shared_output * padded_shared_downproj // shared_vec
        if isinstance(shared_fc2_weight_sf.shape[1], int):
            if shared_fc2_weight_sf.shape[1] != expected_shared_fc2_weight_sf_columns:
                raise ValueError("shared_fc2_weight_sf has an incompatible column count.")
        shared_fc2_sfa = cute.make_tensor(
            shared_fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (Int32(padded_shared_output), Int32(padded_shared_downproj), singleton), shared_vec
            ),
        )
        shared_fc2_sfb = cute.make_tensor(
            shared_fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (Int32(shared_fc1_output_sf.shape[0]), Int32(shared_fc1_output_sf.shape[1] * shared_vec), singleton),
                shared_vec,
            ),
        )
        shared_fc2_output = cute.make_tensor(
            shared_output.iterator,
            cute.make_layout(
                (tokens, 1, shared_output_size),
                stride=(shared_output.stride[0], shared_output.stride[0], shared_output.stride[1]),
            ),
        )
        shared_tma_params = self._mainloop.prepare_shared_tma_load_params(
            shared_fc1_a=shared_fc1_a,
            shared_fc1_b=shared_fc1_b,
            shared_fc1_sfa=shared_fc1_sfa,
            shared_fc1_sfb=shared_fc1_sfb,
            shared_fc2_a=shared_fc2_a,
            shared_fc2_b=shared_fc2_b,
            shared_fc2_sfa=shared_fc2_sfa,
            shared_fc2_sfb=shared_fc2_sfb,
        )
        store_params = self.shared_epilogue.prepare_tma_store_params(shared_fc1_output_gemm, shared_fc2_output)
        return (shared_tma_params, *store_params, shared_fc2_output)

    def _shared_counter_bindings(self) -> Tuple:
        """Device-side shared FC1 done counter pointer and the count a shared FC2 tile waits for."""
        if not self.shared_expert:
            return None, None
        shared_fc1_done_counter = self._device_workspace.tensor(self.shared_fc1_done_counter_region)
        return shared_fc1_done_counter.iterator, Int32(self.shared_fc2_spin_threshold)

    def _shared_device_bindings(
        self,
        kernel_extension,
        shared_fc1_output_tma_atom,
        shared_fc1_output_tma_tensor,
        shared_fc2_output_tma_atom,
        shared_fc2_output_tma_tensor,
        shared_fc2_output,
    ) -> Tuple:
        """The extension for the shared SF views and the shared epilogue bindings (both None-free only when enabled)."""
        if not self.shared_expert:
            return kernel_extension, None
        shared_fc1_done_counter = self._device_workspace.tensor(self.shared_fc1_done_counter_region)
        shared_fc1_output_sf_storage = self._device_workspace.tensor(self.shared_fc1_output_sf_region)
        shared_fc1_output_sf = cute.make_tensor(
            shared_fc1_output_sf_storage.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (shared_fc1_output_sf_storage.shape[0], self.shared_intermediate_gateup_size // 2, Int32(1)),
                self.shared_epilogue.sf_vec_size,
            ),
        )
        # The extension derives SF views from its vector size, so the shared kind needs its own instance.
        shared_kernel_extension = kernel_extension
        if self.heterogeneous_shared:
            shared_kernel_extension = dataclasses.replace(
                kernel_extension, sf_vec_size=self.shared_quant_kind.sf_vec_size
            )
        bindings = SharedEpilogueBindings(
            base=self.shared_epilogue,
            kernel_extension=shared_kernel_extension,
            tma_atom_fc1_output=shared_fc1_output_tma_atom,
            fc1_output=shared_fc1_output_tma_tensor,
            fc1_output_sf=shared_fc1_output_sf,
            fc1_done_counter=shared_fc1_done_counter,
            fc2_output=shared_fc2_output,
            tma_atom_fc2_output=shared_fc2_output_tma_atom,
            fc2_tma_output=shared_fc2_output_tma_tensor,
            # mxfp8 carries its whole rescale in the e8m0 block scales: no alpha or norm constant.
            epi_args=GatedActEpilogueArgs(fc1_alpha=None, fc2_alpha=None, fc1_norm_const=None, topk_scores=None),
        )
        return shared_kernel_extension, bindings

__all__ = [
    "SHARED_QUANT_KINDS",
    "SharedExpertKernelMixin",
    "shared_kernel_impl_requirements",
    "shared_kernel_problem_requirements",
]
