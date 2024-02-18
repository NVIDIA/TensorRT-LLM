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


EpiTagNames = {
    TrtLlm_EpilogueTag.epilogue_op_default: "lc",  # linear combination
    TrtLlm_EpilogueTag.epilogue_op_bias:
    "lc_bias"  # linear combination with bias addition
}

EpiTag = {
    TrtLlm_EpilogueTag.epilogue_op_default:
    "tensorrt_llm::cutlass_extensions::EpilogueOpDefault",
    TrtLlm_EpilogueTag.epilogue_op_bias:
    "tensorrt_llm::cutlass_extensions::EpilogueOpBias"
}


################################################################################
# Quantization Operation and string utils
class TrtLlm_QuantOp(enum.Enum):
    per_column_scale_only = enum_auto()
    finegrained_scale_only = enum_auto()
    finegrained_scale_and_zeros = enum_auto()


QuantOpNames = {
    TrtLlm_QuantOp.per_column_scale_only: "cs",
    TrtLlm_QuantOp.finegrained_scale_only: "fgs",
    TrtLlm_QuantOp.finegrained_scale_and_zeros: "fgsz"
}

QuantOpTag = {
    TrtLlm_QuantOp.per_column_scale_only:
    "cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY",
    TrtLlm_QuantOp.finegrained_scale_only:
    "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY",
    TrtLlm_QuantOp.finegrained_scale_and_zeros:
    "cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS"
}

################################################################################
# The activations, biases, scales and zeros are instantiated using CUDA types,
# not CUTLASS types. This map materializes the name of the CUDA type.
CudaTypeName = {
    DataType.e4m3: "__nv_fp8_e4m3",
    DataType.bf16: "__nv_bfloat16",
    DataType.f16: "half"
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
                 cga_shape=None,
                 mainloop_schedule=None,
                 epi_schedule=None):
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
    weight_tag = DataTypeTag[operation.weight_type]
    scale_zero_tag = CudaTypeName[operation.scalezero_type]
    bias_tag = CudaTypeName[operation.bias_type]
    out_tag = CudaTypeName[operation.output_type]

    quant_op = QuantOpTag[operation.quant_op]
    epi_tag = EpiTag[operation.epi_tag]

    cute_cta_shape = tuple_to_cute_shape(operation.cta_shape)
    cute_cga_shape = tuple_to_cute_shape(operation.cga_shape)

    kernel_sched = KernelScheduleTag[operation.mainloop_schedule]

    # Here, we must append MixedInput depending on the schedule, since we know the types are different.
    # It is a work around since the CUTLASS library did not have the MixedInput schedules at the time of writing.
    if operation.mainloop_schedule in [
            KernelScheduleType.TmaWarpSpecializedCooperative,
            KernelScheduleType.TmaWarpSpecializedPingpong,
            KernelScheduleType.TmaWarpSpecialized
    ]:
        kernel_sched += "MixedInput"
    epi_sched = EpilogueScheduleTag[operation.epi_schedule]

    instantiation = f"""
template void sm90_generic_mixed_gemm_kernelLauncher<{act_tag}, {weight_tag}, {scale_zero_tag}, {bias_tag}, {out_tag},
{quant_op}, {epi_tag},
{cute_cta_shape}, {cute_cga_shape},
{kernel_sched}, {epi_sched}> (
const {act_tag}*, const {weight_tag}*, const {scale_zero_tag}*, const {scale_zero_tag}*, const {bias_tag}*, const float,
{out_tag}*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);
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
    with open(output_file, mode="w") as f:
        f.write(get_file_content(launcher_inl_files, operations))


def is_op_valid(op):
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


################################################################################
def generate_sm90_operations():
    arch = 90

    # For legacy reasons, we use unsigned types for fp16 / bf16 activations.
    # Takes the form (activation_type, weight_type, scalezero_type, bias_type, output_type)
    supported_dtypes = [
        (DataType.e4m3, DataType.s4, DataType.f16, DataType.f16, DataType.f16),
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

        operation = TrtLlm_GemmLauncher(GemmKind.Gemm, arch, *dtype_combo, quant_op, epi_tag, cta_shape_mnk, \
                                        warp_shape, stages, cga_shape, mainloop_schedule, epi_schedule)

        if is_op_valid(operation):
            operations.append(operation)
    return operations


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

    hopper_inl = "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"

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
        out_file = os.path.join(
            output_dir, f"cutlass_kernel_file_{file_counter}.generated.cu")
        write_file([hopper_inl], value, out_file)
        file_counter += 1
