import argparse
import enum
import os
from itertools import product

from cutlass_library import *


################################################################################
# Epilogue Tag enum and string utils
class TrtLlm_EpilogueTag(enum.Enum):
    epilogue_op_default = enum_auto()
    epilogue_op_bias = enum_auto()
    epilogue_op_silu = enum_auto()
    epilogue_op_gelu = enum_auto()


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
CudaTypeName = {
    DataType.e4m3: "__nv_fp8_e4m3",
    DataType.bf16: "__nv_bfloat16",
    DataType.f16: "half",
    DataType.f32: "float"
}


################################################################################
# A data structure holding all info to instantiate gemm launchers in TRT LLM.
class TrtLlm_GemmLauncher:

    def __init__(self, gemm_kind, arch, act_type, weight_type, scalezero_type,
                 bias_type, output_type, quant_op, epi_tag, cta_shape,
                 warp_shape, stages, cga_shape, mainloop_schedule,
                 epi_schedule):
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
        self.mainloop_schedule = mainloop_schedule
        self.epi_schedule = epi_schedule

    def __repr__(self):
        kernel_prefix = "{}_sm{}_{}_{}_{}_{}_{}_{}_{}_{}x{}x{}_{}x{}x{}_{}".format(
            GemmKindNames[self.gemm_kind], self.arch,
            DataTypeNames[self.act_type], DataTypeNames[self.weight_type],
            DataTypeNames[self.scalezero_type], DataTypeNames[self.bias_type],
            DataTypeNames[self.output_type], QuantOpNames[self.quant_op],
            EpiTagNames[self.epi_tag], self.cta_shape[0], self.cta_shape[1],
            self.cta_shape[2], self.warp_shape[0], self.warp_shape[1],
            self.warp_shape[2], self.stages)

        hopper_suffix = "_{}x{}x{}{}{}".format(
            self.cga_shape[0], self.cga_shape[1], self.cga_shape[2],
            KernelScheduleSuffixes[self.mainloop_schedule],
            EpilogueScheduleSuffixes[self.epi_schedule])

        if self.arch == 90:
            return kernel_prefix + hopper_suffix
        elif self.arch > 90:
            raise ValueError(f"SM{self.arch} not supported yet.")
        return kernel_prefix


################################################################################
def tuple_to_cute_shape(shape):
    return f"cute::Shape<cute::Int<{shape[0]}>, cute::Int<{shape[1]}>, cute::Int<{shape[2]}>>"


def instantiate_operation(operation):
    act_tag = CudaTypeName[operation.act_type]
    scale_zero_tag = CudaTypeName[operation.scalezero_type]
    bias_tag = CudaTypeName[operation.bias_type]
    out_tag = CudaTypeName[operation.output_type]

    quant_op = QuantOpTag[operation.quant_op]
    epi_tag = EpiTag[operation.epi_tag]

    cute_cta_shape = tuple_to_cute_shape(operation.cta_shape)
    cute_cga_shape = tuple_to_cute_shape(operation.cga_shape)

    kernel_sched = KernelScheduleTag[operation.mainloop_schedule]
    epi_sched = EpilogueScheduleTag[operation.epi_schedule]

    if operation.gemm_kind == GemmKind.Gemm:
        if operation.mainloop_schedule in [
                KernelScheduleType.TmaWarpSpecializedCooperative,
                KernelScheduleType.TmaWarpSpecializedPingpong,
                KernelScheduleType.TmaWarpSpecialized
        ] and DataTypeSize[operation.act_type] != DataTypeSize[
                operation.weight_type]:
            # Here, we must append MixedInput depending on the schedule, since we know the types are different.
            # It is a work around since the CUTLASS library did not have the MixedInput schedules at the time of writing.
            kernel_sched += "MixedInput"

        weight_tag = DataTypeTag[operation.weight_type]
        instantiation = f"""
template void sm90_generic_mixed_gemm_kernelLauncher<{act_tag}, {weight_tag}, {scale_zero_tag}, {bias_tag}, {out_tag},
{quant_op}, {epi_tag},
{cute_cta_shape}, {cute_cga_shape},
{kernel_sched}, {epi_sched}> (
const {act_tag}*, const {weight_tag}*, const {scale_zero_tag}*, const {scale_zero_tag}*, const {bias_tag}*, const float,
{out_tag}*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);
"""
    elif operation.gemm_kind == GemmKind.Grouped:
        # Similar to MixedInput above, we must modify the tags for grouped gemm as CUTLASS library does not have the updated schedules
        assert operation.mainloop_schedule in [
            KernelScheduleType.TmaWarpSpecializedCooperative,
            KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum
        ]
        assert operation.epi_schedule == EpilogueScheduleType.NoSmemWarpSpecialized
        kernel_sched.replace("::Kernel", "::KernelGrouped")
        epi_sched += "Grouped"

        weight_tag = CudaTypeName[operation.weight_type]

        instantiation = f"""
        template void sm90_generic_moe_gemm_kernelLauncher<{act_tag}, {weight_tag},
                {epi_tag}, {cute_cta_shape}, {cute_cga_shape}, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);
"""
    return instantiation


def get_file_content(launcher_inl_files, operations):
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
namespace cutlass_kernels
{{

{instantiations}

}} // namespace cutlass_kernels
}} // namespace kernels
}} // namespace tensorrt_llm
"""
    return file_content


def write_file(launcher_inl_files, operations, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode="w") as f:
        f.write(get_file_content(launcher_inl_files, operations))


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

    if op.epi_schedule != EpilogueScheduleType.NoSmemWarpSpecialized:
        return False

    if op.mainloop_schedule not in [
            KernelScheduleType.TmaWarpSpecializedCooperative,
            KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum
    ]:
        return False

    return True


def is_op_valid(op):
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
        cta_shape_k = max_k_bits // DataTypeSize[dtype_combo[0]]
        cta_shape_mnk = cta_shape_mn + (cta_shape_k, )

        use_coop = cta_shape_mn[0] == 128
        mainloop_schedule = KernelScheduleType.TmaWarpSpecializedCooperative if use_coop else KernelScheduleType.TmaWarpSpecializedPingpong
        epi_schedule = EpilogueScheduleType.TmaWarpSpecializedCooperative if use_coop else EpilogueScheduleType.TmaWarpSpecialized

        fpA_intB_operation = TrtLlm_GemmLauncher(GemmKind.Gemm, arch, *dtype_combo, quant_op, epi_tag, cta_shape_mnk, \
                                                 warp_shape, stages, cga_shape, mainloop_schedule, epi_schedule)

        if is_op_valid(fpA_intB_operation):
            operations.append(fpA_intB_operation)

    return operations


def generate_sm90_grouped_gemm_operations():
    arch = 90
    supported_dtypes = [
        DataType.f16, DataType.bf16, DataType.f32, DataType.e4m3
    ]
    quant_ops = [TrtLlm_QuantOp.none]
    epi_tags = [TrtLlm_EpilogueTag.epilogue_op_default]
    M_TILES = [128]  # Currently M tile must be 128 for Grouped GEMM
    N_TILES = [16, 32, 64, 128, 256]
    cta_shapes_mn = product(M_TILES, N_TILES)

    warp_shape = [0, 0, 0]  # ignored except for naming
    stages = 0  # auto

    cga_shapes = product([1, 2], [1, 2], [1])

    partial_args = product(supported_dtypes, quant_ops, epi_tags, cta_shapes_mn,
                           cga_shapes)

    operations = list()
    for dtype, quant_op, epi_tag, cta_shape_mn, cga_shape in partial_args:
        max_k_bits = 128 * 8
        cta_shape_k = max_k_bits // DataTypeSize[dtype]
        cta_shape_mnk = cta_shape_mn + (cta_shape_k, )

        mainloop_schedule = KernelScheduleType.TmaWarpSpecializedCooperative if dtype else KernelScheduleType.TmaWarpSpecializedCooperativeFP8FastAccum
        epi_schedule = EpilogueScheduleType.NoSmemWarpSpecialized

        moe_gemm_operation = TrtLlm_GemmLauncher(
            GemmKind.Grouped, arch, dtype, dtype, dtype, dtype, dtype, quant_op,
            epi_tag, cta_shape_mnk, warp_shape, stages, cga_shape,
            mainloop_schedule, epi_schedule)

        if is_op_valid(moe_gemm_operation):
            operations.append(moe_gemm_operation)
    return operations


def generate_sm90_operations():
    operations = generate_sm90_mixed_gemm_operations()
    operations.extend(generate_sm90_grouped_gemm_operations())
    return operations


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
        item = {
            "arch": arch,
            "dtype": dtype,
            "epi_tag": epi_tag,
            "cta_shape": cta_shape_mnk,
            "stage": stage
        }
        operations.append(item)
    return operations


def generate_sm80_operations():
    operations = generate_sm80_fused_grouped_gemm_operations()
    return operations


def get_sm80_file_content(op_item):
    includes = f"#include <tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl>"
    act_tag = DataTypeTag[op_item['dtype']]
    weight_tag = DataTypeTag[op_item['dtype']]
    epi_tag = EpiTag[op_item['epi_tag']]

    instantiations = f"""
        template void sm80_generic_fused_moe_gemm_kernelLauncher<{act_tag}, {weight_tag}, {op_item['cta_shape'][0]}, {op_item['cta_shape'][1]}, {op_item['cta_shape'][2]}, {op_item['stage']}, {epi_tag}>
                ({act_tag} const* A, {weight_tag} const* B, {act_tag} const* biases, {act_tag}* C, int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
"""
    file_content = f"""{includes}
namespace tensorrt_llm
{{
namespace kernels
{{
namespace cutlass_kernels
{{

{instantiations}

}} // namespace cutlass_kernels
}} // namespace kernels
}} // namespace tensorrt_llm
"""
    return file_content


def write_sm80_file(op_item, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w") as f:
        f.write(get_sm80_file_content(op_item))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print the output directory')

    # Add the output_dir argument with short and long options
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help='Path to the output directory')

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the absolute path of the provided directory
    output_dir = os.path.abspath(args.output_dir)

    fpA_intB_inl = "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"
    moe_gemm_inl = "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.inl"

    inl_map = {GemmKind.Gemm: [fpA_intB_inl], GemmKind.Grouped: [moe_gemm_inl]}

    # The goal here is to group kernels with common instantiations together in order to reduce template instantiation overheads.
    # Template instantiation dominates the time in a compilation unit, so it is the most important factor to improve.
    operations = generate_sm90_operations()
    op_groups = dict()
    for op in operations:
        dict_key = (op.gemm_kind, op.arch, op.cta_shape[0])
        op_group = op_groups.get(dict_key, list())
        op_group.append(op)
        op_groups[dict_key] = op_group

    file_counter = 1
    for key, value in op_groups.items():
        gemm_kind, _, _ = key
        out_file = os.path.join(
            output_dir, GemmKindNames[gemm_kind],
            f"cutlass_kernel_file_{file_counter}.generated.cu")
        write_file(inl_map[gemm_kind], value, out_file)
        file_counter += 1

    # Since GemmKind.Grouped is used for gen sm90 moe code.
    sm80_operations = generate_sm80_operations()
    for op_item in sm80_operations:
        # print(op_item)
        out_file_path = os.path.join(
            output_dir, "gemm_grouped",
            f"fused_moe_sm{op_item['arch']}_{op_item['cta_shape'][0]}_{op_item['cta_shape'][1]}_{op_item['cta_shape'][2]}_{op_item['stage']}_{DataTypeNames[op_item['dtype']]}_{EpiTagNames[op_item['epi_tag']]}.generated.cu"
        )
        write_sm80_file(op_item, out_file_path)
        # print(out_file_path)
