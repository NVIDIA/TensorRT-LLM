import argparse
import enum
import os
from itertools import chain, product

from cutlass_library import *


################################################################################
# Epilogue Tag enum and string utils
class TrtLlm_EpilogueTag(enum.Enum):
    epilogue_op_default = enum_auto()
    epilogue_op_bias = enum_auto()
    epilogue_op_silu = enum_auto()
    epilogue_op_gelu = enum_auto()


class TrtLlm_EpilogueFusion(enum.Enum):
    epilogue_fusion_none = enum_auto()
    epilogue_fusion_finalize = enum_auto()


EpiTagNames = {
    TrtLlm_EpilogueTag.epilogue_op_default: "lc",  # linear combination
    TrtLlm_EpilogueTag.epilogue_op_bias:
    "lc_bias",  # linear combination with bias addition
    TrtLlm_EpilogueTag.epilogue_op_silu: "silu",  # silu or swiglu
    TrtLlm_EpilogueTag.epilogue_op_gelu: "gelu"  # gelu or geglu
}

EpiTag = {
    TrtLlm_EpilogueTag.epilogue_op_default:
    "tensorrt_llm::cutlass_extensions::EpilogueOpDefault",
    TrtLlm_EpilogueTag.epilogue_op_bias:
    "tensorrt_llm::cutlass_extensions::EpilogueOpBias",
    TrtLlm_EpilogueTag.epilogue_op_silu:
    "tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu",
    TrtLlm_EpilogueTag.epilogue_op_gelu:
    "tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu"
}

EpiFusion = {
    TrtLlm_EpilogueFusion.epilogue_fusion_none:
    "tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE",
    TrtLlm_EpilogueFusion.epilogue_fusion_finalize:
    "tensorrt_llm::TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE",
}

EpiFusionSuffixes = {
    None: "",
    TrtLlm_EpilogueFusion.epilogue_fusion_none: "EpilogueFusion_NONE",
    TrtLlm_EpilogueFusion.epilogue_fusion_finalize: "EpilogueFusion_FINALIZE",
}


################################################################################
# Quantization Operation and string utils
class TrtLlm_QuantOp(enum.Enum):
    per_column_scale_only = enum_auto()
    finegrained_scale_only = enum_auto()
    finegrained_scale_and_zeros = enum_auto()
    none = enum_auto()


QuantOpNames = {
    TrtLlm_QuantOp.per_column_scale_only: "cs",
    TrtLlm_QuantOp.finegrained_scale_only: "fgs",
    TrtLlm_QuantOp.finegrained_scale_and_zeros: "fgsz",
    TrtLlm_QuantOp.none: "noquant"
}

QuantOpTag = {
    TrtLlm_QuantOp.per_column_scale_only:
    "cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY",
    TrtLlm_QuantOp.finegrained_scale_only:
    "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY",
    TrtLlm_QuantOp.finegrained_scale_and_zeros:
    "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS",
    TrtLlm_QuantOp.none: "void"
}

################################################################################
# The activations, biases, scales and zeros are instantiated using CUDA types,
# not CUTLASS types. This map materializes the name of the CUDA type.


class e2m1_type:  # WAR until we have upgraded everything to a supported version
    pass


e2m1 = e2m1_type()


def GetDataTypeBits(type):
    if isinstance(type, e2m1_type):
        return 4
    return DataTypeSize[type]


def GetDataTypeNames(type, is_mx_fpx=None):
    mxprefix = ""
    if is_mx_fpx is not None:
        mxprefix = "mx_" if is_mx_fpx else "nv_"
    if isinstance(type, e2m1_type):
        return mxprefix + "e2m1"
    return mxprefix + DataTypeNames[type]


CudaTypeName = {
    e2m1: "SafeFP4",
    DataType.e4m3: "__nv_fp8_e4m3",
    DataType.bf16: "__nv_bfloat16",
    DataType.f16: "half",
    DataType.f32: "float",
    DataType.e2m1: "__nv_fp4_e2m1",
    DataType.ue8m0: "cutlass::float_ue8m0_t",
    DataType.u4: "cutlass::uint4b_t"
}


################################################################################
# A data structure holding all info to instantiate gemm launchers in TRT LLM.
class TrtLlm_GemmLauncher:

    def __init__(self,
                 gemm_kind,
                 arch,
                 act_type,
                 weight_type,
                 scalezero_type,
                 bias_type,
                 output_type,
                 quant_op,
                 epi_tag,
                 cta_shape,
                 warp_shape,
                 stages,
                 cga_shape,
                 mainloop_schedule,
                 epi_schedule,
                 epi_fusion=None,
                 is_mx_fpx=False,
                 dynamic_cga=False,
                 swap_ab=False):
        self.gemm_kind = gemm_kind
        self.arch = arch
        self.act_type = act_type
        self.weight_type = weight_type
        self.scalezero_type = scalezero_type
        self.bias_type = bias_type
        self.output_type = output_type
        self.quant_op = quant_op
        self.epi_tag = epi_tag
        self.cta_shape = cta_shape
        self.warp_shape = warp_shape
        self.stages = stages
        self.cga_shape = cga_shape
        self.dynamic_cga = dynamic_cga
        self.mainloop_schedule = mainloop_schedule
        self.epi_schedule = epi_schedule
        self.epi_fusion = epi_fusion
        self.is_mx_fpx = is_mx_fpx
        self.swap_ab = swap_ab

    def __repr__(self):
        kernel_prefix = "{}_sm{}_{}_{}_{}_{}_{}_{}_{}_{}x{}x{}_{}x{}x{}_{}".format(
            GemmKindNames[self.gemm_kind], self.arch,
            GetDataTypeNames(self.act_type, self.is_mx_fpx),
            GetDataTypeNames(self.weight_type, self.is_mx_fpx),
            GetDataTypeNames(self.scalezero_type),
            GetDataTypeNames(self.bias_type),
            GetDataTypeNames(self.output_type), QuantOpNames[self.quant_op],
            EpiTagNames[self.epi_tag], self.cta_shape[0], self.cta_shape[1],
            self.cta_shape[2], self.warp_shape[0], self.warp_shape[1],
            self.warp_shape[2], self.stages)

        hopper_suffix = "_{}x{}x{}{}{}{}{}{}".format(
            self.cga_shape[0], self.cga_shape[1], self.cga_shape[2],
            KernelScheduleSuffixes[self.mainloop_schedule],
            EpilogueScheduleSuffixes[self.epi_schedule],
            EpiFusionSuffixes[self.epi_fusion],
            "_mxfpx_" if self.is_mx_fpx else "",
            "_swap_ab" if self.swap_ab else "")

        if self.arch >= 90:
            return kernel_prefix + hopper_suffix
        elif self.arch > 100:
            raise ValueError(f"SM{self.arch} not supported yet.")
        return kernel_prefix


################################################################################
def tuple_to_cute_shape(shape):
    return f"cute::Shape<cute::Int<{shape[0]}>, cute::Int<{shape[1]}>, cute::Int<{shape[2]}>>"


def instantiate_operation_tma_warp_specialized(operation):
    act_tag = CudaTypeName[operation.act_type]
    scale_zero_tag = CudaTypeName[operation.scalezero_type]
    bias_tag = CudaTypeName[operation.bias_type]
    out_tag = CudaTypeName[operation.output_type]

    quant_op = QuantOpTag[operation.quant_op]
    epi_tag = EpiTag[operation.epi_tag]

    cute_cta_shape = tuple_to_cute_shape(operation.cta_shape)
    cute_cga_shape = tuple_to_cute_shape(operation.cga_shape)

    kernel_sched = KernelScheduleTag[operation.mainloop_schedule]
    epi_sched = "void"
    if operation.epi_schedule is not None:
        epi_sched = EpilogueScheduleTag[operation.epi_schedule]

    if operation.gemm_kind == GemmKind.Gemm:
        weight_tag = DataTypeTag[operation.weight_type]
        instantiation = f"""
template void sm90_generic_mixed_gemm_kernelLauncher<{act_tag}, {weight_tag}, {scale_zero_tag}, {bias_tag}, {out_tag},
{quant_op}, {epi_tag},
{cute_cta_shape}, {cute_cga_shape},
{kernel_sched}, {epi_sched}> (
const {act_tag}*, const {weight_tag}*, const {scale_zero_tag}*, const {scale_zero_tag}*, const {bias_tag}*, const float,
{out_tag}*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);"""
    elif operation.gemm_kind == GemmKind.Grouped:
        if operation.act_type != operation.weight_type and (
                operation.act_type != DataType.e4m3
                or operation.weight_type != e2m1):
            # Mixed MoE GEMM
            weight_tag = CudaTypeName[operation.weight_type]
            instantiation = f"""
template void sm90_generic_mixed_moe_gemm_kernelLauncher<{act_tag}, {weight_tag}, {out_tag},
{epi_tag}, {cute_cta_shape}, {cute_cga_shape}, {kernel_sched}, {epi_sched}, {quant_op}> (
GroupedGemmInput<{act_tag}, {weight_tag}, {out_tag}, {out_tag}>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);
"""
        else:
            # Similar to MixedInput above, we must modify the tags for grouped gemm as CUTLASS library does not have the updated schedules
            assert operation.mainloop_schedule in [
                KernelScheduleType.TmaWarpSpecializedCooperative,
                KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum
            ]
            kernel_sched.replace("::Kernel", "::KernelGrouped")
            # epi_sched += "Grouped"
            # arch_tag = f"cutlass::arch::Sm{operation.arch}"
            arch_tag = f"Sm{operation.arch}"
            weight_tag = CudaTypeName[operation.weight_type]
            assert operation.epi_fusion is not None
            epi_fusion = EpiFusion[operation.epi_fusion]

            # We need to remove the '::' because this will break the instantiation macro
            epi_fusion = epi_fusion.split(':')[-1]
            epi_tag = epi_tag.split(':')[-1]
            epi_sched = epi_sched.split(':')[-1]
            epi_sched = epi_sched.replace(
                "1Sm",
                "")  # Hack to WAR missing `PtrArrayTmaWarpSpecialized` type

            guard_map = {
                e2m1: "defined(ENABLE_FP4)",
                DataType.e4m3: "defined(ENABLE_FP8)",
                DataType.bf16: "defined(ENABLE_BF16)"
            }
            guard_act = guard_map[
                operation.act_type] if operation.act_type in guard_map else "1"
            guard_weight = guard_map[
                operation.
                weight_type] if operation.weight_type in guard_map else "1"
            # TODO Revert this once compiler bug is fixed so we can use template instead of macro again
            #         instantiation = f"""
            #         template void tma_warp_specialized_generic_moe_gemm_kernelLauncher<{arch_tag}, {act_tag}, {weight_tag}, {out_tag},
            #                 {epi_tag}, {epi_fusion}, {cute_cta_shape}, {cute_cga_shape}, false>
            #                 (TmaWarpSpecializedGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);
            # """
            instantiation = f"""
#if {guard_act} && {guard_weight}
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM({arch_tag}, {act_tag}, {weight_tag}, {out_tag}, {epi_sched}, {epi_tag}, {epi_fusion}, {operation.cta_shape[0]}, {operation.cta_shape[1]}, {operation.cta_shape[2]}, {operation.cga_shape[0]}, {operation.cga_shape[1]}, {operation.cga_shape[2]}, {"true" if operation.is_mx_fpx else "false"}, {"true" if operation.dynamic_cga else "false"}, false, {"true" if operation.swap_ab else "false"});
#endif"""
    return instantiation


def instantiate_operation_sm80(operation):
    act_tag = DataTypeTag[operation.dtype]
    weight_tag = DataTypeTag[operation.dtype]
    epi_tag = EpiTag[operation.epi_tag]

    instantiation = f"""
            template void sm80_generic_fused_moe_gemm_kernelLauncher<{act_tag}, {weight_tag}, {operation.cta_shape[0]}, {operation.cta_shape[1]}, {operation.cta_shape[2]}, {operation.stage}, {epi_tag}>
                    ({act_tag} const* A, {weight_tag} const* B, {act_tag} const* biases, bool bias_is_broadcast, {act_tag}* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);"""
    return instantiation


def instantiate_operation(operation):
    if operation.arch == 80:
        return instantiate_operation_sm80(operation)
    elif operation.arch >= 90:
        return instantiate_operation_tma_warp_specialized(operation)


def get_file_content(launcher_inl_files, operations):
    assert operations
    include_list = list()
    for file in launcher_inl_files:
        include_list.append(f"#include \"{file}\"")
    includes = "\n".join(include_list)

    insts_list = list()
    for op in operations:
        insts_list.append(instantiate_operation(op))
    instantiations = "\n".join(insts_list)

    file_content = f"""{includes}
namespace tensorrt_llm
{{
namespace kernels
{{
namespace cutlass_kernels_oss
{{

{instantiations}

}} // namespace cutlass_kernels_oss
}} // namespace kernels
}} // namespace tensorrt_llm
"""
    return file_content


def clean_leftover_files(output_dir, generated_files):
    """Remove leftover generated files that weren't created in this run."""
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path not in generated_files:
                os.remove(file_path)


def write_file(launcher_inl_files, operations, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Avoid changing modified time if file content is up to date
    content = get_file_content(launcher_inl_files, operations)
    try:
        with open(output_file, mode="r") as f:
            if f.read() == content:
                return
    except FileNotFoundError:
        pass
    with open(output_file, mode="w") as f:
        f.write(content)


def is_gemm_op_valid_sm100(op):
    # TODO These are much more restricted than theory dictates, investigate if more can be enabled in future
    tile_m, tile_n, _ = op.cta_shape
    cga_m, cga_n, cga_k = op.cga_shape

    if op.epi_fusion == TrtLlm_EpilogueFusion.epilogue_fusion_finalize and op.epi_schedule != EpilogueScheduleType.PtrArrayTmaWarpSpecialized1Sm:
        return False

    # We use a runtime cluster shape for SM100, so we only use cluster shapes to distinguish between 1SM and 2SM variants.
    if cga_m > 2 or cga_n != 1 or cga_k != 1:
        return False

    if op.arch == 103:
        return op.act_type == e2m1 and op.weight_type == e2m1 and tile_m == 128 and tile_n in [
            128, 256
        ]

    # Default shapes
    # This is epilogue tile size. For two CTA this is actually size 128/256 for the MMA
    if tile_m not in [64, 128]:
        return False

    # FP4 Has some much more limited sizes
    if op.act_type == e2m1 or op.weight_type == e2m1:
        if tile_n not in [64, 128, 256] or tile_m != 128:
            return False
        # TODO Revert this once cutlass adds support for blockscaled + no smem
        if op.arch == 100 and op.epi_schedule == EpilogueScheduleType.PtrArrayNoSmemWarpSpecialized1Sm:
            return False

    # Shapes for fp8 small N shapes
    if (op.act_type == DataType.e4m3) and (tile_n == 16
                                           or tile_n == 8) and (cga_m == 1
                                                                and cga_n == 1):
        # todo: double check why tile_n = 8 is disabled in CUTLASS backend. @yuhan
        return tile_m != 128 or tile_n % 16 == 0

    # Default alignment requirements
    if tile_n % 32 != 0 or tile_n < 32 or tile_n > 256:
        return False

    # Two CTA mode needs bigger tile n alignment
    if cga_m % 2 == 0 and tile_n % 64 != 0:
        return False

    return True


def is_gemm_op_valid(op):
    tile_m, tile_n, _ = op.cta_shape
    cga_m, cga_n, _ = op.cga_shape

    if cga_m == 1 and cga_n == 1:
        return True

    if cga_m == 2 and cga_n == 1 and tile_m >= 128:
        return True

    if cga_m == 1 and cga_n == 2 and tile_n >= 128:
        return True

    if cga_m == 2 and cga_n == 2 and tile_m >= 128 and tile_n >= 128:
        return True

    return False


def is_grouped_gemm_op_valid(op):
    if not is_gemm_op_valid(op):
        return False

    if op.epi_tag != TrtLlm_EpilogueTag.epilogue_op_default:
        return False

    if op.epi_schedule is not None and op.epi_schedule != EpilogueScheduleType.NoSmemWarpSpecialized:
        return False

    if op.mainloop_schedule not in [
            KernelScheduleType.TmaWarpSpecializedCooperative,
            KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum
    ]:
        return False

    return True


def is_op_valid(op):
    if op.arch >= 100:
        return is_gemm_op_valid_sm100(op)

    if op.gemm_kind == GemmKind.Gemm:
        return is_gemm_op_valid(op)
    if op.gemm_kind == GemmKind.Grouped:
        return is_grouped_gemm_op_valid(op)


################################################################################
def generate_sm90_mixed_gemm_operations():
    arch = 90

    # For legacy reasons, we use unsigned types for the weights. The instanitated template
    # will remap those back to the signed type.
    # Takes the form (activation_type, weight_type, scalezero_type, bias_type, output_type)
    supported_dtypes = [
        (DataType.e4m3, DataType.u4, DataType.f16, DataType.f16, DataType.f16),
        (DataType.e4m3, DataType.u4, DataType.f16, DataType.bf16,
         DataType.bf16),
        (DataType.f16, DataType.u4, DataType.f16, DataType.f16, DataType.f16),
        (DataType.bf16, DataType.u4, DataType.bf16, DataType.bf16,
         DataType.bf16),
        (DataType.f16, DataType.u8, DataType.f16, DataType.f16, DataType.f16),
        (DataType.bf16, DataType.u8, DataType.bf16, DataType.bf16,
         DataType.bf16)
    ]

    quant_ops = [
        TrtLlm_QuantOp.per_column_scale_only,
        TrtLlm_QuantOp.finegrained_scale_only,
        TrtLlm_QuantOp.finegrained_scale_and_zeros
    ]

    epi_tags = [TrtLlm_EpilogueTag.epilogue_op_bias]

    M_TILES = [64, 128]
    N_TILES = [16, 32, 64, 128, 256]
    cta_shapes_mn = product(M_TILES, N_TILES)

    warp_shape = [4, 1, 1]
    stages = 0  # auto

    cga_shapes = product([1, 2], [1, 2], [1])

    partial_args = product(supported_dtypes, quant_ops, epi_tags, cta_shapes_mn,
                           cga_shapes)

    operations = list()
    for dtype_combo, quant_op, epi_tag, cta_shape_mn, cga_shape in partial_args:
        max_k_bits = 128 * 8
        cta_shape_k = max_k_bits // GetDataTypeBits(dtype_combo[0])
        cta_shape_mnk = cta_shape_mn + (cta_shape_k, )

        use_coop = cta_shape_mn[0] == 128
        mainloop_schedule = KernelScheduleType.TmaWarpSpecializedCooperative if use_coop else KernelScheduleType.TmaWarpSpecializedPingpong
        epi_schedule = EpilogueScheduleType.TmaWarpSpecializedCooperative if use_coop else EpilogueScheduleType.TmaWarpSpecialized

        fpA_intB_operation = TrtLlm_GemmLauncher(GemmKind.Gemm, arch, *dtype_combo, quant_op, epi_tag, cta_shape_mnk, \
                                                 warp_shape, stages, cga_shape, mainloop_schedule, epi_schedule)

        if is_op_valid(fpA_intB_operation):
            operations.append(fpA_intB_operation)

    return operations


def generate_sm90_grouped_gemm_operations(is_arch_enabled):
    if not is_arch_enabled:
        return []
    arch = 90
    supported_dtypes = [
        DataType.f16, DataType.bf16, DataType.f32, DataType.e4m3
    ]
    quant_ops = [TrtLlm_QuantOp.none]
    epi_tags = [TrtLlm_EpilogueTag.epilogue_op_default]
    M_TILES = [128]  # Currently M tile must be 128 for Grouped GEMM
    N_TILES = [16, 32, 64, 128, 256]
    cta_shapes_mn = list(product(M_TILES, N_TILES)) + [(256, 128)]

    warp_shape = [0, 0, 0]  # ignored except for naming
    stages = 0  # auto

    epi_fusions = [
        TrtLlm_EpilogueFusion.epilogue_fusion_none,
        TrtLlm_EpilogueFusion.epilogue_fusion_finalize
    ]

    swap_ab = [True, False]

    cga_shapes = product([1, 2], [1, 2], [1])

    partial_args = product(supported_dtypes, quant_ops, epi_tags, epi_fusions,
                           cta_shapes_mn, cga_shapes, swap_ab)

    operations = list()
    for dtype, quant_op, epi_tag, epi_fusion, cta_shape_mn, cga_shape, swap_ab in partial_args:
        max_k_bits = 128 * 8
        cta_shape_k = max_k_bits // GetDataTypeBits(dtype)
        cta_shape_mnk = cta_shape_mn + (cta_shape_k, )

        mainloop_schedule = KernelScheduleType.TmaWarpSpecializedCooperative if dtype != DataType.e4m3 else KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum
        epi_schedule = None

        otypes = [dtype]
        if dtype == DataType.e4m3:
            otypes = [DataType.f16, DataType.bf16]

        for otype in otypes:
            moe_gemm_operation = TrtLlm_GemmLauncher(GemmKind.Grouped,
                                                     arch,
                                                     dtype,
                                                     dtype,
                                                     dtype,
                                                     dtype,
                                                     otype,
                                                     quant_op,
                                                     epi_tag,
                                                     cta_shape_mnk,
                                                     warp_shape,
                                                     stages,
                                                     cga_shape,
                                                     mainloop_schedule,
                                                     epi_schedule,
                                                     epi_fusion,
                                                     swap_ab=swap_ab)

            if is_op_valid(moe_gemm_operation):
                operations.append(moe_gemm_operation)
    return operations


def generate_sm90_mixed_type_grouped_gemm_operations(is_arch_enabled):
    if not is_arch_enabled:
        return []
    arch = 90

    # act_type, weight_type, scalezero_type, bias_type, output_type
    supported_dtypes_int4 = [
        (DataType.e4m3, DataType.u4, DataType.f16, DataType.f16, DataType.f16),
        (DataType.e4m3, DataType.u4, DataType.bf16, DataType.bf16,
         DataType.bf16),
    ]
    supported_dtypes_fp4 = [
        (DataType.f16, DataType.e2m1, DataType.ue8m0, DataType.f16,
         DataType.f16),
        (DataType.bf16, DataType.e2m1, DataType.ue8m0, DataType.bf16,
         DataType.bf16),
    ]

    quant_ops = [TrtLlm_QuantOp.finegrained_scale_only]

    epi_tags = [TrtLlm_EpilogueTag.epilogue_op_default]

    M_TILES = [64, 128]  # Currently M tile must be 128 for Grouped GEMM
    N_TILES = [16, 32, 64, 128]
    K_TILES = [128, 256, 512]
    cta_shapes_mnk_int4 = list(product(M_TILES, N_TILES, K_TILES))

    M_TILES = [64, 128]  # Currently M tile must be 128 for Grouped GEMM
    N_TILES = [16, 32, 64]
    K_TILES = [128, 256]
    cta_shapes_mnk_fp4 = list(product(M_TILES, N_TILES, K_TILES))
    cta_shapes_mnk_fp4.append((128, 128, 128))

    warp_shape = [0, 0, 0]  # ignored except for naming
    stages = 0  # auto

    cga_shapes = list(product([1, 2], [1, 2], [1]))

    partial_args_int4 = product(supported_dtypes_int4, quant_ops, epi_tags,
                                cta_shapes_mnk_int4, cga_shapes)
    partial_args_fp4 = product(supported_dtypes_fp4, quant_ops, epi_tags,
                               cta_shapes_mnk_fp4, cga_shapes)
    partial_args = chain(partial_args_int4, partial_args_fp4)

    operations = list()
    for dtype_combo, quant_op, epi_tag, cta_shape_mnk, cga_shape in partial_args:
        use_coop = cta_shape_mnk[0] >= 128
        mainloop_schedules = [
            KernelScheduleType.TmaWarpSpecializedCooperative,
            KernelScheduleType.TmaWarpSpecializedPingpong
        ] if use_coop else [KernelScheduleType.TmaWarpSpecializedPingpong]
        epi_schedule = EpilogueScheduleType.TmaWarpSpecializedCooperative
        for mainloop_schedule in mainloop_schedules:
            if (cta_shape_mnk[0] == 128 and cta_shape_mnk[1] == 128
                    and mainloop_schedule
                    == KernelScheduleType.TmaWarpSpecializedCooperative):
                continue
            moe_gemm_operation = TrtLlm_GemmLauncher(GemmKind.Grouped, arch, *dtype_combo, quant_op, epi_tag, cta_shape_mnk, \
                                                    warp_shape, stages, cga_shape, mainloop_schedule, epi_schedule)
            operations.append(moe_gemm_operation)
    return operations


def generate_sm90_operations(is_arch_enabled):
    operations = generate_sm90_mixed_gemm_operations()
    operations.extend(generate_sm90_grouped_gemm_operations(is_arch_enabled))
    operations.extend(
        generate_sm90_mixed_type_grouped_gemm_operations(is_arch_enabled))
    return operations


def calc_shape_mnk_sm100_grouped_gemm(cta_shape_mn, dtype):
    max_k_bits = 128 * 8
    cta_shape_k = max_k_bits // GetDataTypeBits(dtype)
    if dtype == DataType.e4m3 and (cta_shape_mn[1] == 8):
        cta_shape_k = 256
    return cta_shape_mn + (cta_shape_k, )


def generate_sm120_grouped_gemm_operations(is_arch_enabled):

    if not is_arch_enabled:
        return []
    arch = 120
    supported_dtypes = [e2m1, (DataType.e4m3, e2m1)]
    quant_ops = [TrtLlm_QuantOp.none]
    epi_tags = [TrtLlm_EpilogueTag.epilogue_op_default]
    cta_shapes_mnk = [[128, 128, 128], [128, 128, 256], [256, 128, 128],
                      [128, 256, 128]]

    warp_shape = [0, 0, 0]  # ignored except for naming
    stages = 0  # auto

    epi_fusions = [
        TrtLlm_EpilogueFusion.epilogue_fusion_none,
        TrtLlm_EpilogueFusion.epilogue_fusion_finalize
    ]

    cga_shapes = [[1, 1, 1]]

    swap_ab = [True, False]

    partial_args = product(supported_dtypes, quant_ops, epi_tags, epi_fusions,
                           cta_shapes_mnk, cga_shapes, swap_ab)

    operations = list()
    for dtype, quant_op, epi_tag, epi_fusion, cta_shape_mnk, cga_shape, swap_ab in partial_args:

        # Ignored
        mainloop_schedule = KernelScheduleType.TmaWarpSpecializedCooperative
        epi_schedule = None

        if isinstance(dtype, tuple):
            act_type, weight_type = dtype
        else:
            act_type, weight_type = dtype, dtype

        # Minimal filter: for mixed FP8xFP4 on SM120, only emit 128x128x128
        if act_type == DataType.e4m3 and weight_type == e2m1:
            if cta_shape_mnk != [128, 128, 128]:
                continue

        otypes = [act_type]
        if act_type in [DataType.e4m3, e2m1]:
            otypes = [DataType.f16, DataType.bf16]

        for otype in otypes:
            moe_gemm_operation = TrtLlm_GemmLauncher(
                GemmKind.Grouped,
                arch,
                act_type,
                weight_type,
                act_type,
                act_type,
                otype,
                quant_op,
                epi_tag,
                cta_shape_mnk,
                warp_shape,
                stages,
                cga_shape,
                mainloop_schedule,
                epi_schedule,
                epi_fusion,
                is_mx_fpx=(act_type == DataType.e4m3 and weight_type == e2m1),
                swap_ab=swap_ab)

            operations.append(moe_gemm_operation)
    return operations


def generate_sm120_operations(is_arch_enabled):
    operations = generate_sm120_grouped_gemm_operations(is_arch_enabled)
    return operations


def generate_sm100_grouped_gemm_operations(is_arch_enabled, arch):
    if not is_arch_enabled:
        return []
    supported_dtypes = [
        DataType.f16, DataType.bf16, DataType.f32, DataType.e4m3, e2m1,
        (DataType.e4m3, e2m1)
    ]
    quant_ops = [TrtLlm_QuantOp.none]
    epi_tags = [TrtLlm_EpilogueTag.epilogue_op_default]
    cta_shapes_m = [64, 128]
    cta_shapes_n = [8, 16, 32, 64, 128, 192, 256]
    cta_shapes_mn = product(cta_shapes_m, cta_shapes_n)

    warp_shape = [0, 0, 0]  # ignored except for naming
    stages = 0  # auto

    epi_fusions = [
        TrtLlm_EpilogueFusion.epilogue_fusion_none,
        TrtLlm_EpilogueFusion.epilogue_fusion_finalize
    ]

    # Some shapes for SM100 are better with NoSmem, note the kernel will internally map to the 1 or 2 SM variants based on the cga_shape[0]
    epi_schedules = [
        EpilogueScheduleType.PtrArrayNoSmemWarpSpecialized1Sm,
        EpilogueScheduleType.PtrArrayTmaWarpSpecialized1Sm
    ]

    # We will use dynamic cluster shapes for SM100, so we only need to indicate if we are using 1 or 2 SM version
    cga_shapes = [(1, 1, 1), (2, 1, 1)]

    swap_ab = [True, False]

    dynamic_cga = [True, False]

    partial_args = product(supported_dtypes, quant_ops, epi_tags, epi_fusions,
                           cta_shapes_mn, cga_shapes, epi_schedules,
                           dynamic_cga, swap_ab)

    operations = list()
    for dtype, quant_op, epi_tag, epi_fusion, cta_shape_mn, cga_shape, epi_schedule, dynamic_cga, swap_ab in partial_args:
        if isinstance(dtype, tuple):
            dtype, weight_type = dtype
        else:
            weight_type = dtype

        cta_shape_mnk = calc_shape_mnk_sm100_grouped_gemm(cta_shape_mn, dtype)

        # Ignored
        mainloop_schedule = KernelScheduleType.TmaWarpSpecializedCooperative

        otypes = [dtype]
        if dtype in [DataType.e4m3, e2m1]:
            otypes = [DataType.f16, DataType.bf16]

        for otype in otypes:
            moe_gemm_operation = TrtLlm_GemmLauncher(
                GemmKind.Grouped,
                arch,
                dtype,
                weight_type,
                otype,
                otype,
                otype,
                quant_op,
                epi_tag,
                cta_shape_mnk,
                warp_shape,
                stages,
                cga_shape,
                mainloop_schedule,
                epi_schedule,
                epi_fusion,
                is_mx_fpx=(dtype == DataType.e4m3 and weight_type == e2m1),
                dynamic_cga=dynamic_cga,
                swap_ab=swap_ab)

            if is_op_valid(moe_gemm_operation):
                operations.append(moe_gemm_operation)
    return operations


def generate_sm103_operations(is_arch_enabled):
    operations = generate_sm100_grouped_gemm_operations(is_arch_enabled, 103)
    return operations


def generate_sm100_operations(is_arch_enabled):
    operations = generate_sm100_grouped_gemm_operations(is_arch_enabled, 100)
    return operations


class GemmSm80LauncherConfig:

    def __init__(self, gemm_kind, arch, dtype, epi_tag, cta_shape, stage):
        self.gemm_kind = gemm_kind
        self.arch = arch
        self.dtype = dtype
        self.epi_tag = epi_tag
        self.cta_shape = cta_shape
        self.stage = stage


def generate_sm80_fused_grouped_gemm_operations():
    arch = 80
    supported_dtypes = [DataType.f16, DataType.bf16]
    epi_tags = [
        TrtLlm_EpilogueTag.epilogue_op_silu, TrtLlm_EpilogueTag.epilogue_op_gelu
    ]
    cta_shapes_mnk = [(16, 128, 64), (16, 256, 64), (32, 128, 64),
                      (64, 128, 64), (128, 128, 64)]

    stages = [2, 3, 4]

    partial_args = product(supported_dtypes, epi_tags, cta_shapes_mnk, stages)

    operations = list()
    for dtype, epi_tag, cta_shape_mnk, stage in partial_args:
        item = GemmSm80LauncherConfig(GemmKind.Grouped, arch, dtype, epi_tag,
                                      cta_shape_mnk, stage)
        operations.append(item)
    return operations


def generate_sm80_operations(is_arch_enabled):
    operations = generate_sm80_fused_grouped_gemm_operations()
    return operations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print the output directory')

    # Add the output_dir argument with short and long options
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help='Path to the output directory')
    parser.add_argument("-a",
                        "--architectures",
                        type=str,
                        required=True,
                        help="Architectures to generate kernels for")
    parser.add_argument("-i",
                        "--internal",
                        action='store_true',
                        required=False,
                        help="Generate the internal build kernels")

    # Parse the command line arguments
    args = parser.parse_args()

    arches = args.architectures.split(';')
    # Get the absolute path of the provided directory
    output_dir = os.path.abspath(args.output_dir)

    fpA_intB_inl = "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"
    moe_gemm_inl = "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
    # moe_gemm_inl = "tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
    moe_mixed_gemm_inl = "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl"
    # moe_mixed_gemm_inl = "tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl"
    sm80_moe_gemm_inl = "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"
    # sm80_moe_gemm_inl = "tensorrt_llm/kernels/internal_cutlass_kernels/src/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"

    inl_map = {
        (GemmKind.Gemm, 90): [fpA_intB_inl],
        (GemmKind.Grouped, 90): [moe_gemm_inl],
        (GemmKind.Grouped, 100): [moe_gemm_inl],
        (GemmKind.Grouped, 103): [moe_gemm_inl],
        (GemmKind.Grouped, 120): [moe_gemm_inl],
        (GemmKind.Grouped, 80): [sm80_moe_gemm_inl]
    }

    def has_arch(sm):
        return f"{sm}" in arches or f"{sm}-real" in arches or f"{sm}f-real" in arches or f"{sm}f" in arches

    # The goal here is to group kernels with common instantiations together in order to reduce template instantiation overheads.
    # Template instantiation dominates the time in a compilation unit, so it is the most important factor to improve.
    operations = []
    operations += generate_sm120_operations(has_arch(120) or has_arch(121))
    operations += generate_sm103_operations(has_arch(103))
    operations += generate_sm100_operations(has_arch(100) or has_arch(103))
    operations += generate_sm90_operations(has_arch(90))
    operations += generate_sm80_operations(has_arch(80) or has_arch(89))

    def should_skip(op):
        return False  # All kernels have a public implementation

    # The mixed dtype grouped gemm for w4afp8 has a different launcher
    def is_mixed_dtype_grouped(op):
        if isinstance(op, GemmSm80LauncherConfig):
            return False
        # Only w4a8fp8 and not wfp4afp8
        return (op.act_type != op.weight_type) and (
            op.gemm_kind == GemmKind.Grouped) and (op.act_type != DataType.e4m3
                                                   or op.weight_type != e2m1)

    # Fix OOM error in CI. If len(operations) is more than GROUP_SIZE, it will be split into multiple sub groups.
    GROUP_SIZE = 8
    op_groups = dict()
    for op in operations:
        if should_skip(op):
            continue
        # This dict key is used to group kernels with common instantiations together
        # Similar implementations should live in the same file so the compiler can share the cutlass state
        # Without this we see significant memory consumption, and separating them also does not reduce the compilation time
        # because most time is spent parsing the same cutlass files
        # We separate by: Architecture, Leading dimension of the CTA shape, FP4 (i.e. block scaled MMA), mixed input
        # TODO Do a more scientific analysis of this
        dict_key = (op.gemm_kind, op.arch, op.cta_shape[0], op.arch >= 100
                    and (op.weight_type == e2m1 or op.is_mx_fpx),
                    is_mixed_dtype_grouped(op))
        op_group = op_groups.get(dict_key, [])
        if len(op_group) == 0 or len(op_group[-1]) >= GROUP_SIZE:
            op_group.append([op])
        else:
            op_group[-1].append(op)
        op_groups[dict_key] = op_group

    file_list = []
    for key, value in op_groups.items():
        gemm_kind, arch, m, block_scale, is_mixed = key
        for i, op_sub_group in enumerate(value):
            out_file = os.path.join(
                output_dir, GemmKindNames[gemm_kind], str(arch),
                f"cutlass_kernel_file_{GemmKindNames[gemm_kind]}_sm{arch}_M{m}{'_BS' if block_scale else ''}{'_Mixed' if is_mixed else ''}_group{i}.generated.cu"
            )
            inl_file = [moe_mixed_gemm_inl] if is_mixed else inl_map[key[:2]]
            write_file(inl_file, op_sub_group, out_file)
            file_list.append(out_file)

    # Clean up any leftover files from previous runs
    clean_leftover_files(output_dir, set(file_list))
